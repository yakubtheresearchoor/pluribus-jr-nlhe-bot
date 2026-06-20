//! G5: fully GPU-native bucketed iteration — the Fix-C async
//! orchestrator. One command buffer per traverser pass; one encoder per
//! dispatch; ordering comes from Metal hazard tracking on the shared
//! buffers (the same pattern that took the exact-solver pipeline to
//! zero intermediate waits); commit without wait, single flush at
//! readback. No reach prepass on the CPU, no packing, no per-walk
//! readbacks — the G5-1 eliminable budget (reach prepass 8.6 GB/iter +
//! packing 2.0 GB/iter at 4×4) is gone by construction.
//!
//! Bit-exactness design (the standard every prior link met): each
//! output element is written by exactly one thread; inner loops run in
//! the CPU walk's iteration order; walk-vs-walk reordering touches only
//! independent buffers; chance accumulation preserves the CPU's
//! outcome-ascending sum. The identity gate (G5-3) holds this to
//! bit-exact at B = nh through full DCFR iterations.
//!
//! State lives GPU-resident in the ZoneDims-concatenated layout
//! ([flop | turn ti-major | river outcome-major] — identical to the
//! CPU solver's per-zone buffers laid end to end), so gate readback is
//! a direct slice comparison.

use crate::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, BuDesc, FlopBucketing, InfosetDesc, ReachEdge,
};
use crate::solver::flop_start_game::FlopChanceTable;
use crate::solver::flop_start_vector_cfr::{DcfrParams, Zone};
use crate::tree::flat::{FlatTree, MAX_NA_POSTFLOP};
use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoderRef,
    ComputePipelineState, MTLResourceOptions, MTLSize,
};

use super::bucketed_terminal::{BucketedTerminalGpu, WalkKey};
use super::context::{MetalContext, MetalError, MetalResult};

/// Field-for-field mirror of Metal `NativeWalkParams`.
#[repr(C)]
#[derive(Clone, Copy)]
struct WalkParamsGpu {
    count: u32,
    np: u32,
    nh: u32,
    nb: u32,
    map_off: u32,
    omega_off: u32,
    traverser: u32,
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
    regret_floor: f32,
    _pad: u32,
}

/// Nil-checked allocation. Metal's `new_buffer` returns a NIL object
/// when the allocation fails (too large / out of memory); writing
/// through its `contents()` (NULL) was the SIGSEGV the v1 live-6 cell
/// pricing hit 2026-06-12 (45,711-node tree → 27 GB reach_mega).
fn alloc_buf(ctx: &MetalContext, bytes: usize, what: &str) -> MetalResult<Buffer> {
    let max = ctx.device().max_buffer_length() as usize;
    if bytes > max {
        return Err(MetalError::CapacityExceeded(format!(
            "{what}: {bytes} bytes exceeds device max_buffer_length {max}"
        )));
    }
    let buf = ctx
        .device()
        .new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared);
    if buf.contents().is_null() {
        return Err(MetalError::CapacityExceeded(format!(
            "{what}: allocation of {bytes} bytes returned nil (out of GPU memory)"
        )));
    }
    Ok(buf)
}

fn buf_from<T: Copy>(ctx: &MetalContext, data: &[T]) -> MetalResult<Buffer> {
    let bytes = std::mem::size_of_val(data);
    let buf = alloc_buf(ctx, bytes, std::any::type_name::<T>())?;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.contents() as *mut u8, bytes);
    }
    Ok(buf)
}

fn zeros_buf(ctx: &MetalContext, floats: usize, what: &str) -> MetalResult<Buffer> {
    let buf = alloc_buf(ctx, floats * 4, what)?;
    unsafe {
        std::ptr::write_bytes(buf.contents() as *mut u8, 0, floats * 4);
    }
    Ok(buf)
}

/// (element offset, count) ranges into a concatenated upload buffer.
type Range = (usize, usize);

pub struct BucketedNativeGpu {
    queue: CommandQueue,
    p_strat: ComputePipelineState,
    p_zero: ComputePipelineState,
    p_root_init: ComputePipelineState,
    p_seed: ComputePipelineState,
    p_reach: ComputePipelineState,
    p_cfv: ComputePipelineState,
    p_regret: ComputePipelineState,
    p_chance: ComputePipelineState,
    p_accum_step: ComputePipelineState,
    p_publish: ComputePipelineState,
    p_root_accum: ComputePipelineState,
    /// Terminal resources + the batched collapsed kernel (resident
    /// mode); single-source with the G3-gated hybrid path.
    term: BucketedTerminalGpu,

    // ── Persistent state (ZoneDims concatenated) ──
    regrets: Buffer,
    cum: Buffer,
    strategy: Buffer,
    flop_stride: usize,
    turn_total: usize,
    river_total: usize,
    turn_offset_f: usize,
    river_offset_f: usize,
    turn_stride: usize,
    river_stride: usize,

    // ── Per-pass scratch (resident) ──
    reach_mega: Buffer,
    cfv_mega: Buffer,
    root_cfv: Buffer,
    /// SHARED-BUFFER SEQUENTIAL MODE (2026-06-12 big-tree unlock): when
    /// the per-walk mega-buffer layout (n_walks × nn × np × nh) exceeds
    /// the u32 walk-offset range or the device buffer limit, allocate
    /// ONE nn-sized reach/cfv buffer (all walk offsets 0, absolute node
    /// indexing unchanged) and SERIALIZE walks by encode order — Metal
    /// hazard tracking on the shared buffer enforces it. Chance sums
    /// accumulate per outcome into `acc_buf` (outcome-ascending — the
    /// mega kernel's exact f32 order) and publish to the boundary rows.
    /// Free by the G5 wall=busy finding: one walk at these widths
    /// already saturates the GPU, so walk concurrency buys nothing.
    /// Regrets always encode inline per walk here (the walk's cfv rows
    /// are overwritten by the next outcome).
    shared_mode: bool,
    /// Per-ti river chance accumulator (Σ_ri cp·river_cfv at the river
    /// chance-children rows). MUST be separate from `acc_turn`: the
    /// turn accumulation spans the whole ti loop while river
    /// accumulation runs inside each ti — one shared buffer overlapped
    /// them (the at-size live-6 parity failure, drift ~1.0).
    acc_river: Buffer,
    /// Cross-ti turn chance accumulator (Σ_ti cp·turn_cfv).
    acc_turn: Buffer,

    // ── Plan-derived (uploaded once) ──
    descs_buf: Buffer,
    flop_desc: Range,
    turn_desc_all: Range,
    river_desc_all: Range,
    turn_desc: Vec<Range>,  // per ti
    river_desc: Vec<Range>, // per outcome (ti·max_river + ri)
    edges_buf: Buffer,
    flop_edge_lv: Vec<Range>,
    turn_edge_lv: Vec<Range>,
    river_edge_lv: Vec<Range>,
    bu_buf: Buffer,
    flop_bu_lv: Vec<Range>,
    turn_bu_lv: Vec<Vec<Range>>,  // [ti][level]
    river_bu_lv: Vec<Vec<Range>>, // [outcome][level]
    children_buf: Buffer,
    node_child_off_buf: Buffer,
    node_local_base_buf: Buffer,
    maps_buf: Buffer,   // ridx·nh scheme (same as the terminal's)
    omegas_buf: Buffer, // ridx·nh scheme
    weights_buf: Buffer,
    cp_turn_buf: Buffer,
    cp_river_buf: Buffer,
    cp_river_off: Vec<usize>, // float offset per ti
    turn_seed_buf: Buffer,
    ts_count: usize,
    river_seed_buf: Buffer,
    rs_count: usize,
    /// cfv float offsets of river walks, ti-grouped (chance accum).
    river_cfv_offs_buf: Buffer,
    river_offs_off: Vec<usize>, // element offset per ti
    /// cfv float offsets of turn walks (flop chance accum).
    turn_cfv_offs_buf: Buffer,

    walks: Vec<WalkKey>,
    reach_offs: Vec<u32>, // float offset per walk
    cfv_offs: Vec<u32>,
    n_rivers: Vec<usize>,
    river_walk_base: Vec<usize>,
    total_rivers: usize,

    np: usize,
    nh: usize,
    n_turn: usize,
    max_river: usize,
    nb_flop: usize,
    nb_turn: usize,
    nb_river: usize,
    reach_per_walk: usize,

    iteration: u32,
    regret_floor: f32,
    inflight: Vec<CommandBuffer>,
    /// Parallel to `inflight`: stage tag (usize::MAX = untagged).
    inflight_stages: Vec<usize>,
    gpu_busy_s: f64,
    stage_timing: bool,
    stage_busy: [f64; 5],
}

impl BucketedNativeGpu {
    /// Runout index in the concatenated map/omega buffers — the SAME
    /// flattening the terminal uses (flop 0; turn 1+ti; river
    /// 1+n_turn+ti·max_river+ri).
    fn runout_index(&self, zone: Zone, tc: Option<usize>, rc: Option<usize>) -> usize {
        match zone {
            Zone::Flop => 0,
            Zone::Turn => 1 + tc.unwrap(),
            Zone::River => 1 + self.n_turn + tc.unwrap() * self.max_river + rc.unwrap(),
            Zone::Preflop => unreachable!(),
        }
    }

    pub fn new(
        ctx: &MetalContext,
        tree: &FlatTree,
        table: &FlopChanceTable,
        bk: &FlopBucketing,
        solver: &BucketedFlopCfr,
        stripes: u32,
    ) -> MetalResult<Self> {
        Self::new_inner(ctx, tree, table, bk, solver, stripes, false)
    }

    /// Force the SHARED-BUFFER SEQUENTIAL pass regardless of size —
    /// for the bit-exactness gates (small trees auto-select mega mode,
    /// so the shared path needs an explicit handle to be gated on the
    /// same fixtures).
    pub fn new_forced_shared(
        ctx: &MetalContext,
        tree: &FlatTree,
        table: &FlopChanceTable,
        bk: &FlopBucketing,
        solver: &BucketedFlopCfr,
        stripes: u32,
    ) -> MetalResult<Self> {
        Self::new_inner(ctx, tree, table, bk, solver, stripes, true)
    }

    pub fn is_shared_mode(&self) -> bool {
        self.shared_mode
    }

    fn new_inner(
        ctx: &MetalContext,
        tree: &FlatTree,
        table: &FlopChanceTable,
        bk: &FlopBucketing,
        solver: &BucketedFlopCfr,
        stripes: u32,
        force_shared: bool,
    ) -> MetalResult<Self> {
        let plan = solver.native_plan(tree, bk);
        let layout = solver.gpu_layout(bk);
        let dims = layout.dims;
        let term = BucketedTerminalGpu::new(ctx, tree, table, bk, solver, stripes)?;

        let nn = tree.num_nodes();
        let np = tree.num_players as usize;
        let nh = table.num_valid;
        let n_turn = table.remaining_deck.len();
        let max_river = dims.max_river;
        // ── Walk bookkeeping (plan order: rivers, turns, flop) ──
        let n_rivers: Vec<usize> = (0..n_turn).map(|ti| bk.river_map[ti].len()).collect();
        let mut river_walk_base = vec![0usize; n_turn];
        let mut acc = 0usize;
        for ti in 0..n_turn {
            river_walk_base[ti] = acc;
            acc += n_rivers[ti];
        }
        let total_rivers = acc;
        let n_walks = total_rivers + n_turn + 1;
        assert_eq!(plan.walks.len(), n_walks);
        let reach_per_walk = nn * np * nh;
        let cfv_per_walk = nn * nh;

        // ── CAPACITY PRE-FLIGHT + MODE SELECTION (2026-06-12, after
        // the v1 live-6 cell SIGSEGV: 45,711-node tree → 27 GB
        // reach_mega → nil buffer → bzero through NULL). Limits:
        //   1. Walk offsets are u32 FLOAT indices in the GPU
        //      descriptors (WalkDescGpu.reach_off) — the mega layout
        //      caps at 4 Gi floats per buffer REGARDLESS of device
        //      memory. Overflow would silently corrupt — worse than
        //      the crash.
        //   2. Per-buffer device limit (max_buffer_length), enforced
        //      in alloc_buf at every allocation.
        // When the mega layout (n_walks × nn × np × nh) exceeds either
        // limit, fall back to SHARED-BUFFER SEQUENTIAL MODE: one
        // nn-sized reach/cfv buffer, all walk offsets 0, walks
        // serialized by encode order (see the `shared_mode` field doc).
        // Zone-size probe verdict (zone_size_probe.rs): per-zone
        // buffers do NOT unlock big trees — the river zone is ~70% of
        // nn and 16 concurrent river walks would still need 16 GB+;
        // the shared buffer is the layout that actually fits (live-6
        // limp: 27 GB → 1.29 GB).
        let max_buf = ctx.device().max_buffer_length() as u64;
        let mega_reach_floats = n_walks as u64 * reach_per_walk as u64;
        let mega_cfv_floats = n_walks as u64 * cfv_per_walk as u64;
        let shared_mode = force_shared
            || mega_reach_floats > u32::MAX as u64
            || mega_cfv_floats > u32::MAX as u64
            || mega_reach_floats * 4 > max_buf
            || mega_cfv_floats * 4 > max_buf;
        if shared_mode
            && ((reach_per_walk as u64) > u32::MAX as u64
                || (reach_per_walk as u64) * 4 > max_buf)
        {
            return Err(MetalError::CapacityExceeded(format!(
                "even the shared-buffer layout does not fit: nn {nn} × np {np} × nh {nh} \
                 = {reach_per_walk} floats per reach buffer (u32 cap {}, device cap {} \
                 bytes)",
                u32::MAX,
                max_buf
            )));
        }
        let (reach_offs, cfv_offs): (Vec<u32>, Vec<u32>) = if shared_mode {
            (vec![0; n_walks], vec![0; n_walks])
        } else {
            (
                (0..n_walks).map(|w| (w * reach_per_walk) as u32).collect(),
                (0..n_walks).map(|w| (w * cfv_per_walk) as u32).collect(),
            )
        };

        // ── Infoset descriptors: [flop | turn ti-major | river outcome-major] ──
        let mut all_descs: Vec<InfosetDesc> = Vec::new();
        let flop_desc: Range = (0, plan.flop_infosets.len());
        all_descs.extend_from_slice(&plan.flop_infosets);
        let turn_start = all_descs.len();
        let mut turn_desc: Vec<Range> = Vec::with_capacity(n_turn);
        for ti in 0..n_turn {
            turn_desc.push((all_descs.len(), plan.turn_infosets[ti].len()));
            all_descs.extend_from_slice(&plan.turn_infosets[ti]);
        }
        let turn_desc_all: Range = (turn_start, all_descs.len() - turn_start);
        let river_start = all_descs.len();
        let mut river_desc: Vec<Range> = Vec::with_capacity(n_turn * max_river);
        for o in 0..n_turn * max_river {
            river_desc.push((all_descs.len(), plan.river_infosets[o].len()));
            all_descs.extend_from_slice(&plan.river_infosets[o]);
        }
        let river_desc_all: Range = (river_start, all_descs.len() - river_start);
        let descs_buf = buf_from(ctx, &all_descs)?;

        // ── Reach edges per (zone, level), concatenated ──
        let mut all_edges: Vec<ReachEdge> = Vec::new();
        let mut edge_ranges = |levels: &[Vec<ReachEdge>]| -> Vec<Range> {
            levels
                .iter()
                .map(|es| {
                    let r = (all_edges.len(), es.len());
                    all_edges.extend_from_slice(es);
                    r
                })
                .collect()
        };
        let flop_edge_lv = edge_ranges(&plan.flop_edges);
        let turn_edge_lv = edge_ranges(&plan.turn_edges);
        let river_edge_lv = edge_ranges(&plan.river_edges);
        let edges_buf = buf_from(ctx, &all_edges)?;

        // ── Bottom-up descriptors per (zone, outcome, level) ──
        let mut all_bu: Vec<BuDesc> = Vec::new();
        let mut bu_ranges = |levels: &[Vec<BuDesc>]| -> Vec<Range> {
            levels
                .iter()
                .map(|ds| {
                    let r = (all_bu.len(), ds.len());
                    all_bu.extend_from_slice(ds);
                    r
                })
                .collect()
        };
        let flop_bu_lv = bu_ranges(&plan.flop_bu);
        let turn_bu_lv: Vec<Vec<Range>> =
            plan.turn_bu.iter().map(|lv| bu_ranges(lv)).collect();
        let river_bu_lv: Vec<Vec<Range>> =
            plan.river_bu.iter().map(|lv| bu_ranges(lv)).collect();
        let bu_buf = buf_from(ctx, &all_bu)?;

        let children_buf = buf_from(ctx, &plan.children_flat)?;
        let node_child_off_buf = buf_from(ctx, &plan.node_child_off)?;
        let node_local_base_buf = buf_from(ctx, &layout.node_local_base)?;

        // ── Maps + omegas, runout-indexed (ridx·nh) ──
        let n_runouts = 1 + n_turn + n_turn * max_river;
        let mut maps = vec![u16::MAX; n_runouts * nh];
        let mut omegas = vec![0.0f32; n_runouts * nh];
        maps[0..nh].copy_from_slice(&bk.flop_map);
        omegas[0..nh].copy_from_slice(&bk.flop_omega);
        for ti in 0..n_turn {
            let r = 1 + ti;
            maps[r * nh..(r + 1) * nh].copy_from_slice(&bk.turn_map[ti]);
            omegas[r * nh..(r + 1) * nh].copy_from_slice(&bk.turn_omega[ti]);
            for ri in 0..n_rivers[ti] {
                let r = 1 + n_turn + ti * max_river + ri;
                maps[r * nh..(r + 1) * nh].copy_from_slice(&bk.river_map[ti][ri]);
                omegas[r * nh..(r + 1) * nh].copy_from_slice(&bk.river_omega[ti][ri]);
            }
        }
        let maps_buf = buf_from(ctx, &maps)?;
        let omegas_buf = buf_from(ctx, &omegas)?;

        let weights_buf = buf_from(ctx, &table.initial_weight_flat())?;

        // ── Chance probabilities (identical call chain to the CPU run) ──
        let mut cp_turn = vec![0.0f32; n_turn * nh];
        for ti in 0..n_turn {
            for h in 0..nh {
                cp_turn[ti * nh + h] = table.chance_probability_turn(ti, h);
            }
        }
        let cp_turn_buf = buf_from(ctx, &cp_turn)?;
        let mut cp_river: Vec<f32> = Vec::new();
        let mut cp_river_off = vec![0usize; n_turn];
        for ti in 0..n_turn {
            cp_river_off[ti] = cp_river.len();
            let tc_card = table.remaining_deck[ti];
            for ri in 0..n_rivers[ti] {
                for h in 0..nh {
                    cp_river.push(table.chance_probability_river(tc_card, ri, h));
                }
            }
        }
        let cp_river_buf = buf_from(ctx, &cp_river)?;

        let ts_count = plan.turn_seed_nodes.len();
        let rs_count = plan.river_seed_nodes.len();
        let turn_seed_buf = buf_from(ctx, &plan.turn_seed_nodes)?;
        let river_seed_buf = buf_from(ctx, &plan.river_seed_nodes)?;

        // ── cfv-offset lists for the chance accumulations ──
        let mut river_offs: Vec<u32> = Vec::with_capacity(total_rivers);
        let mut river_offs_off = vec![0usize; n_turn];
        for ti in 0..n_turn {
            river_offs_off[ti] = river_offs.len();
            for ri in 0..n_rivers[ti] {
                river_offs.push(cfv_offs[river_walk_base[ti] + ri]);
            }
        }
        let river_cfv_offs_buf = buf_from(ctx, &river_offs)?;
        let turn_offs: Vec<u32> =
            (0..n_turn).map(|ti| cfv_offs[total_rivers + ti]).collect();
        let turn_cfv_offs_buf = buf_from(ctx, &turn_offs)?;

        Ok(Self {
            queue: ctx.queue().to_owned(),
            p_strat: ctx.create_pipeline("bucketed_native_strategies")?,
            p_zero: ctx.create_pipeline("bucketed_native_zero")?,
            p_root_init: ctx.create_pipeline("bucketed_native_root_init")?,
            p_seed: ctx.create_pipeline("bucketed_native_seed")?,
            p_reach: ctx.create_pipeline("bucketed_native_reach_level")?,
            p_cfv: ctx.create_pipeline("bucketed_native_cfv_level")?,
            p_regret: ctx.create_pipeline("bucketed_native_regret_update")?,
            p_chance: ctx.create_pipeline("bucketed_native_chance_accum")?,
            p_accum_step: ctx.create_pipeline("bucketed_native_chance_accum_step")?,
            p_publish: ctx.create_pipeline("bucketed_native_rows_publish")?,
            p_root_accum: ctx.create_pipeline("bucketed_native_root_accum")?,
            term,
            regrets: zeros_buf(ctx, dims.total_floats(), "regrets")?,
            cum: zeros_buf(ctx, dims.total_floats(), "cum")?,
            strategy: zeros_buf(ctx, dims.total_floats(), "strategy")?,
            flop_stride: dims.flop_stride(),
            turn_total: dims.turn_total(),
            river_total: dims.river_total(),
            turn_offset_f: dims.turn_offset(),
            river_offset_f: dims.river_offset(),
            turn_stride: dims.turn_stride(),
            river_stride: dims.river_stride(),
            reach_mega: if shared_mode {
                zeros_buf(ctx, reach_per_walk, "reach_shared")?
            } else {
                zeros_buf(ctx, n_walks * reach_per_walk, "reach_mega")?
            },
            cfv_mega: if shared_mode {
                zeros_buf(ctx, cfv_per_walk, "cfv_shared")?
            } else {
                zeros_buf(ctx, n_walks * cfv_per_walk, "cfv_mega")?
            },
            root_cfv: zeros_buf(ctx, nh, "root_cfv")?,
            shared_mode,
            acc_river: zeros_buf(ctx, rs_count.max(1) * nh, "acc_river")?,
            acc_turn: zeros_buf(ctx, ts_count.max(1) * nh, "acc_turn")?,
            descs_buf,
            flop_desc,
            turn_desc_all,
            river_desc_all,
            turn_desc,
            river_desc,
            edges_buf,
            flop_edge_lv,
            turn_edge_lv,
            river_edge_lv,
            bu_buf,
            flop_bu_lv,
            turn_bu_lv,
            river_bu_lv,
            children_buf,
            node_child_off_buf,
            node_local_base_buf,
            maps_buf,
            omegas_buf,
            weights_buf,
            cp_turn_buf,
            cp_river_buf,
            cp_river_off,
            turn_seed_buf,
            ts_count,
            river_seed_buf,
            rs_count,
            river_cfv_offs_buf,
            river_offs_off,
            turn_cfv_offs_buf,
            walks: plan.walks,
            reach_offs,
            cfv_offs,
            n_rivers,
            river_walk_base,
            total_rivers,
            np,
            nh,
            n_turn,
            max_river,
            nb_flop: bk.nb_flop,
            nb_turn: bk.nb_turn,
            nb_river: bk.nb_river,
            reach_per_walk,
            iteration: 0,
            // Mirror of BucketedFlopCfr::new's default.
            regret_floor: -1e30,
            inflight: Vec::new(),
            inflight_stages: Vec::new(),
            gpu_busy_s: 0.0,
            stage_timing: false,
            stage_busy: [0.0; 5],
        })
    }

    fn wparams(
        &self,
        count: usize,
        nb: usize,
        map_off: usize,
        traverser: u8,
        d: &DcfrParams,
    ) -> WalkParamsGpu {
        WalkParamsGpu {
            count: count as u32,
            np: self.np as u32,
            nh: self.nh as u32,
            nb: nb as u32,
            map_off: map_off as u32,
            omega_off: map_off as u32, // same ridx·nh scheme
            traverser: traverser as u32,
            alpha_t: d.alpha_t(),
            beta_t: d.beta_t(),
            gamma_t: d.gamma_t(),
            regret_floor: self.regret_floor,
            _pad: 0,
        }
    }

    /// One encoder, one dispatch, ceil(n/256)×256 threads; the kernels
    /// all guard with `if (idx >= count) return`.
    fn dispatch(
        &self,
        cmd: &metal::CommandBufferRef,
        pipe: &ComputePipelineState,
        n: usize,
        setup: impl FnOnce(&ComputeCommandEncoderRef),
    ) {
        if n == 0 {
            return;
        }
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        setup(enc);
        let tg = 256u64.min(n as u64).max(1);
        let groups = (n as u64).div_ceil(tg);
        enc.dispatch_thread_groups(
            MTLSize { width: groups, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        enc.end_encoding();
    }

    fn set_params(enc: &ComputeCommandEncoderRef, idx: u64, p: &WalkParamsGpu) {
        enc.set_bytes(
            idx,
            std::mem::size_of::<WalkParamsGpu>() as u64,
            p as *const _ as *const std::ffi::c_void,
        );
    }

    fn set_u32(enc: &ComputeCommandEncoderRef, idx: u64, v: u32) {
        enc.set_bytes(idx, 4, &v as *const _ as *const std::ffi::c_void);
    }

    /// Strategy bases for the reach kernel (`zone_outcome_base`).
    fn zone_outcome_base(&self, zone: Zone, ti: Option<usize>, ri: Option<usize>) -> usize {
        match zone {
            Zone::Flop => 0,
            Zone::Turn => self.turn_offset_f + ti.unwrap() * self.turn_stride,
            Zone::River => {
                self.river_offset_f
                    + (ti.unwrap() * self.max_river + ri.unwrap()) * self.river_stride
            }
            Zone::Preflop => unreachable!(),
        }
    }

    fn walk_index(&self, zone: Zone, ti: Option<usize>, ri: Option<usize>) -> usize {
        match zone {
            Zone::River => self.river_walk_base[ti.unwrap()] + ri.unwrap(),
            Zone::Turn => self.total_rivers + ti.unwrap(),
            Zone::Flop => self.total_rivers + self.n_turn,
            Zone::Preflop => unreachable!(),
        }
    }

    /// Encode one walk's reach: seed (or root init) + ascending-level
    /// edge propagation. The region is zeroed by the pass-wide zero.
    fn encode_reach_walk(
        &self,
        cmd: &metal::CommandBufferRef,
        zone: Zone,
        ti: Option<usize>,
        ri: Option<usize>,
        traverser: u8,
        d: &DcfrParams,
    ) {
        let w = self.walk_index(zone, ti, ri);
        let reach_byte = (self.reach_offs[w] as u64) * 4;
        let (nb, edge_lv): (usize, &Vec<Range>) = match zone {
            Zone::Flop => (self.nb_flop, &self.flop_edge_lv),
            Zone::Turn => (self.nb_turn, &self.turn_edge_lv),
            Zone::River => (self.nb_river, &self.river_edge_lv),
            Zone::Preflop => unreachable!(),
        };
        match zone {
            Zone::Flop => {
                let p = self.wparams(0, nb, 0, traverser, d);
                self.dispatch(cmd, &self.p_root_init, self.np * self.nh, |enc| {
                    enc.set_buffer(0, Some(&self.reach_mega), reach_byte);
                    enc.set_buffer(1, Some(&self.weights_buf), 0);
                    Self::set_params(enc, 2, &p);
                });
            }
            // In SHARED mode the parent walk already wrote the seed
            // rows in place (same buffer, offset 0) — the copy would
            // be a bit-identical self-copy; skip the dispatch.
            Zone::Turn if !self.shared_mode => {
                // Seed from the flop walk's reach.
                let src_byte =
                    (self.reach_offs[self.walk_index(Zone::Flop, None, None)] as u64) * 4;
                let p = self.wparams(self.ts_count, nb, 0, traverser, d);
                self.dispatch(
                    cmd,
                    &self.p_seed,
                    self.ts_count * self.np * self.nh,
                    |enc| {
                        enc.set_buffer(0, Some(&self.reach_mega), reach_byte);
                        enc.set_buffer(1, Some(&self.reach_mega), src_byte);
                        enc.set_buffer(2, Some(&self.turn_seed_buf), 0);
                        Self::set_params(enc, 3, &p);
                    },
                );
            }
            Zone::River if !self.shared_mode => {
                // Seed from this ti's turn walk reach.
                let src_byte =
                    (self.reach_offs[self.walk_index(Zone::Turn, ti, None)] as u64) * 4;
                let p = self.wparams(self.rs_count, nb, 0, traverser, d);
                self.dispatch(
                    cmd,
                    &self.p_seed,
                    self.rs_count * self.np * self.nh,
                    |enc| {
                        enc.set_buffer(0, Some(&self.reach_mega), reach_byte);
                        enc.set_buffer(1, Some(&self.reach_mega), src_byte);
                        enc.set_buffer(2, Some(&self.river_seed_buf), 0);
                        Self::set_params(enc, 3, &p);
                    },
                );
            }
            Zone::Turn | Zone::River => {}
            Zone::Preflop => unreachable!(),
        }
        let map_off = self.runout_index(zone, ti, ri) * self.nh;
        let zob = self.zone_outcome_base(zone, ti, ri) as u32;
        for &(e_off, e_len) in edge_lv.iter() {
            if e_len == 0 {
                continue;
            }
            let p = self.wparams(e_len, nb, map_off, traverser, d);
            let edge_byte = (e_off * std::mem::size_of::<ReachEdge>()) as u64;
            self.dispatch(cmd, &self.p_reach, e_len * self.nh, |enc| {
                enc.set_buffer(0, Some(&self.reach_mega), reach_byte);
                enc.set_buffer(1, Some(&self.strategy), 0);
                enc.set_buffer(2, Some(&self.node_local_base_buf), 0);
                enc.set_buffer(3, Some(&self.maps_buf), 0);
                enc.set_buffer(4, Some(&self.edges_buf), edge_byte);
                Self::set_params(enc, 5, &p);
                Self::set_u32(enc, 6, zob);
            });
        }
    }

    /// Encode one walk's bottom-up: descending-level cfv dispatches,
    /// then ONE regret/cum update over the walk's infosets (deferred to
    /// after the levels — value-identical: every cfv row it reads is
    /// final once its level is done, and nothing in the walk reads
    /// regrets or cum).
    fn encode_bu_walk(
        &self,
        cmd: &metal::CommandBufferRef,
        zone: Zone,
        ti: Option<usize>,
        ri: Option<usize>,
        traverser: u8,
        d: &DcfrParams,
        with_regrets: bool,
    ) {
        let w = self.walk_index(zone, ti, ri);
        let cfv_byte = (self.cfv_offs[w] as u64) * 4;
        let (nb, bu_lv): (usize, &Vec<Range>) = match zone {
            Zone::Flop => (self.nb_flop, &self.flop_bu_lv),
            Zone::Turn => (self.nb_turn, &self.turn_bu_lv[ti.unwrap()]),
            Zone::River => {
                let o = ti.unwrap() * self.max_river + ri.unwrap();
                (self.nb_river, &self.river_bu_lv[o])
            }
            Zone::Preflop => unreachable!(),
        };
        let map_off = self.runout_index(zone, ti, ri) * self.nh;
        for &(b_off, b_len) in bu_lv.iter().rev() {
            if b_len == 0 {
                continue;
            }
            let p = self.wparams(b_len, nb, map_off, traverser, d);
            let bu_byte = (b_off * std::mem::size_of::<BuDesc>()) as u64;
            self.dispatch(cmd, &self.p_cfv, b_len * self.nh, |enc| {
                enc.set_buffer(0, Some(&self.cfv_mega), cfv_byte);
                enc.set_buffer(1, Some(&self.strategy), 0);
                enc.set_buffer(2, Some(&self.maps_buf), 0);
                enc.set_buffer(3, Some(&self.bu_buf), bu_byte);
                enc.set_buffer(4, Some(&self.children_buf), 0);
                Self::set_params(enc, 5, &p);
            });
        }
        if with_regrets {
            self.encode_regret_walk(cmd, zone, ti, ri, traverser, d);
        }
    }

    /// The walk's regret/cum update. Reads only the walk's FINAL cfv
    /// rows; regrets/cum are consumed by nothing until the next pass's
    /// strategies — so this may encode any time after the walk's cfv
    /// levels (stage-timing mode defers all of them to a tagged
    /// buffer; values identical).
    fn encode_regret_walk(
        &self,
        cmd: &metal::CommandBufferRef,
        zone: Zone,
        ti: Option<usize>,
        ri: Option<usize>,
        traverser: u8,
        d: &DcfrParams,
    ) {
        let w = self.walk_index(zone, ti, ri);
        let cfv_byte = (self.cfv_offs[w] as u64) * 4;
        let (nb, desc): (usize, Range) = match zone {
            Zone::Flop => (self.nb_flop, self.flop_desc),
            Zone::Turn => (self.nb_turn, self.turn_desc[ti.unwrap()]),
            Zone::River => {
                let o = ti.unwrap() * self.max_river + ri.unwrap();
                (self.nb_river, self.river_desc[o])
            }
            Zone::Preflop => unreachable!(),
        };
        let map_off = self.runout_index(zone, ti, ri) * self.nh;
        let (d_off, d_len) = desc;
        if d_len == 0 {
            return;
        }
        let p = self.wparams(d_len, nb, map_off, traverser, d);
        let desc_byte = (d_off * std::mem::size_of::<InfosetDesc>()) as u64;
        self.dispatch(cmd, &self.p_regret, d_len * MAX_NA_POSTFLOP * nb, |enc| {
            enc.set_buffer(0, Some(&self.cfv_mega), cfv_byte);
            enc.set_buffer(1, Some(&self.regrets), 0);
            enc.set_buffer(2, Some(&self.cum), 0);
            enc.set_buffer(3, Some(&self.strategy), 0);
            enc.set_buffer(4, Some(&self.maps_buf), 0);
            enc.set_buffer(5, Some(&self.omegas_buf), 0);
            enc.set_buffer(6, Some(&self.descs_buf), desc_byte);
            enc.set_buffer(7, Some(&self.children_buf), 0);
            enc.set_buffer(8, Some(&self.node_child_off_buf), 0);
            Self::set_params(enc, 9, &p);
        });
    }

    /// Stage-timing cut: in timing mode, commit the current buffer
    /// tagged with its stage and start a fresh one (same queue —
    /// submission order + hazard tracking keep semantics identical).
    fn stage_cut(&mut self, cmd: CommandBuffer, stage: usize) -> CommandBuffer {
        if !self.stage_timing {
            return cmd;
        }
        cmd.commit();
        self.inflight.push(cmd);
        self.inflight_stages.push(stage);
        self.queue.new_command_buffer().to_owned()
    }

    /// SHARED-BUFFER SEQUENTIAL pass: one nn-sized reach/cfv buffer,
    /// walks serialized by encode order (hazard tracking on the shared
    /// buffers enforces it — every dispatch references them). Identical
    /// arithmetic to the mega pass:
    ///   - seeds are in place (parent walks write boundary rows
    ///     directly; the mega mode's seed copy is a self-copy here);
    ///   - chance sums accumulate per outcome in ascending order into
    ///     acc_buf (the mega kernel's exact f32 op sequence), then
    ///     publish to the boundary rows;
    ///   - regrets encode inline per walk (its cfv rows are overwritten
    ///     by the next outcome), which is the mega mode's default too.
    /// Write-sets per zone are outcome-invariant (edges/BU descs are
    /// per-zone), so the single pass-wide zero gives the same
    /// unwritten-rows-read-zero semantics as the CPU's fresh per-walk
    /// arrays.
    fn encode_traverser_pass_shared(&mut self, traverser: u8, d: &DcfrParams) {
        let cmd = self.queue.new_command_buffer().to_owned();

        // 1. Strategies (all zones/outcomes — regret-only input).
        self.encode_strategies(&cmd, traverser, d);

        // 2. Zero shared reach (one walk's worth IS the buffer).
        self.dispatch(&cmd, &self.p_zero, self.reach_per_walk, |enc| {
            enc.set_buffer(0, Some(&self.reach_mega), 0);
            Self::set_u32(enc, 1, self.reach_per_walk as u32);
        });

        // 3. Flop subtree: reach → terminals (flop rows stay intact
        //    below — turn/river walks write disjoint rows).
        self.encode_reach_walk(&cmd, Zone::Flop, None, None, traverser, d);
        self.term.encode_walks_resident(
            &cmd,
            &self.walks[self.walk_index(Zone::Flop, None, None)..][..1],
            traverser,
            &self.reach_mega,
            &self.cfv_mega,
            &[0],
            &[0],
        );

        // 4. Per-turn subtrees, sequentially.
        for ti in 0..self.n_turn {
            self.encode_reach_walk(&cmd, Zone::Turn, Some(ti), None, traverser, d);
            self.term.encode_walks_resident(
                &cmd,
                &self.walks[self.walk_index(Zone::Turn, Some(ti), None)..][..1],
                traverser,
                &self.reach_mega,
                &self.cfv_mega,
                &[0],
                &[0],
            );
            for ri in 0..self.n_rivers[ti] {
                self.encode_reach_walk(&cmd, Zone::River, Some(ti), Some(ri), traverser, d);
                self.term.encode_walks_resident(
                    &cmd,
                    &self.walks[self.walk_index(Zone::River, Some(ti), Some(ri))..][..1],
                    traverser,
                    &self.reach_mega,
                    &self.cfv_mega,
                    &[0],
                    &[0],
                );
                self.encode_bu_walk(
                    &cmd, Zone::River, Some(ti), Some(ri), traverser, d, true,
                );
                // acc[r,h] += cp_river[ti][ri][h] · cfv[river_root_r, h]
                let p = self.wparams(self.rs_count, self.nb_turn, 0, traverser, d);
                let cp_byte = ((self.cp_river_off[ti] + ri * self.nh) * 4) as u64;
                self.dispatch(&cmd, &self.p_accum_step, self.rs_count * self.nh, |enc| {
                    enc.set_buffer(0, Some(&self.acc_river), 0);
                    enc.set_buffer(1, Some(&self.cfv_mega), 0);
                    enc.set_buffer(2, Some(&self.cp_river_buf), cp_byte);
                    enc.set_buffer(3, Some(&self.river_seed_buf), 0);
                    Self::set_params(enc, 4, &p);
                });
            }
            // Publish Σ_ri into the river-root rows; re-zero acc.
            let p = self.wparams(self.rs_count, self.nb_turn, 0, traverser, d);
            self.dispatch(&cmd, &self.p_publish, self.rs_count * self.nh, |enc| {
                enc.set_buffer(0, Some(&self.cfv_mega), 0);
                enc.set_buffer(1, Some(&self.acc_river), 0);
                enc.set_buffer(2, Some(&self.river_seed_buf), 0);
                Self::set_params(enc, 3, &p);
            });
            self.encode_bu_walk(&cmd, Zone::Turn, Some(ti), None, traverser, d, true);
            // acc[r,h] += cp_turn[ti][h] · cfv[turn_root_r, h]
            let p = self.wparams(self.ts_count, self.nb_flop, 0, traverser, d);
            let cp_byte = ((ti * self.nh) * 4) as u64;
            self.dispatch(&cmd, &self.p_accum_step, self.ts_count * self.nh, |enc| {
                enc.set_buffer(0, Some(&self.acc_turn), 0);
                enc.set_buffer(1, Some(&self.cfv_mega), 0);
                enc.set_buffer(2, Some(&self.cp_turn_buf), cp_byte);
                enc.set_buffer(3, Some(&self.turn_seed_buf), 0);
                Self::set_params(enc, 4, &p);
            });
        }
        // Publish Σ_ti into the turn-root rows; flop bottom-up; root.
        let p = self.wparams(self.ts_count, self.nb_flop, 0, traverser, d);
        self.dispatch(&cmd, &self.p_publish, self.ts_count * self.nh, |enc| {
            enc.set_buffer(0, Some(&self.cfv_mega), 0);
            enc.set_buffer(1, Some(&self.acc_turn), 0);
            enc.set_buffer(2, Some(&self.turn_seed_buf), 0);
            Self::set_params(enc, 3, &p);
        });
        self.encode_bu_walk(&cmd, Zone::Flop, None, None, traverser, d, true);
        if traverser == 0 {
            self.dispatch(&cmd, &self.p_root_accum, self.nh, |enc| {
                enc.set_buffer(0, Some(&self.root_cfv), 0);
                enc.set_buffer(1, Some(&self.cfv_mega), 0);
                Self::set_u32(enc, 2, self.nh as u32);
            });
        }
        cmd.commit();
        self.inflight.push(cmd);
        self.inflight_stages.push(usize::MAX);
    }

    /// Strategy dispatches for all zones/outcomes (shared by both pass
    /// modes — strategies read only regrets, fixed within a pass).
    fn encode_strategies(&self, cmd: &metal::CommandBufferRef, traverser: u8, d: &DcfrParams) {
        for (range, nb) in [
            (self.flop_desc, self.nb_flop),
            (self.turn_desc_all, self.nb_turn),
            (self.river_desc_all, self.nb_river),
        ] {
            let (d_off, d_len) = range;
            if d_len == 0 {
                continue;
            }
            let p = self.wparams(d_len, nb, 0, traverser, d);
            let desc_byte = (d_off * std::mem::size_of::<InfosetDesc>()) as u64;
            self.dispatch(cmd, &self.p_strat, d_len * nb, |enc| {
                enc.set_buffer(0, Some(&self.regrets), 0);
                enc.set_buffer(1, Some(&self.strategy), 0);
                enc.set_buffer(2, Some(&self.descs_buf), desc_byte);
                Self::set_params(enc, 3, &p);
            });
        }
    }

    /// One traverser pass, one command buffer (Fix-C: encode
    /// everything, commit, NO wait — hazard tracking orders dependent
    /// dispatches; cross-command-buffer order is queue submission
    /// order on tracked resources).
    fn encode_traverser_pass(&mut self, traverser: u8, d: &DcfrParams) {
        if self.shared_mode {
            return self.encode_traverser_pass_shared(traverser, d);
        }
        let mut cmd = self.queue.new_command_buffer().to_owned();

        // ── 1. Strategies (all zones, all outcomes; regrets fixed) ──
        self.encode_strategies(&cmd, traverser, d);

        cmd = self.stage_cut(cmd, 0); // strategies

        // ── 2. Zero all reach (CPU allocates fresh-zeroed per walk;
        //       unwritten rows — e.g. below UNUSED-offset skips — must
        //       read as zero at terminals exactly like the CPU's) ──
        let total_reach = self.walks.len() * self.reach_per_walk;
        self.dispatch(&cmd, &self.p_zero, total_reach, |enc| {
            enc.set_buffer(0, Some(&self.reach_mega), 0);
            Self::set_u32(enc, 1, total_reach as u32);
        });

        // ── 3. Reach in dependency order: flop → turns → rivers ──
        self.encode_reach_walk(&cmd, Zone::Flop, None, None, traverser, d);
        for ti in 0..self.n_turn {
            self.encode_reach_walk(&cmd, Zone::Turn, Some(ti), None, traverser, d);
        }
        for ti in 0..self.n_turn {
            for ri in 0..self.n_rivers[ti] {
                self.encode_reach_walk(&cmd, Zone::River, Some(ti), Some(ri), traverser, d);
            }
        }

        cmd = self.stage_cut(cmd, 1); // reach (zero + seeds + levels)

        // ── 4. ALL terminals, one batched dispatch (resident mode) ──
        self.term.encode_walks_resident(
            &cmd,
            &self.walks,
            traverser,
            &self.reach_mega,
            &self.cfv_mega,
            &self.reach_offs,
            &self.cfv_offs,
        );

        cmd = self.stage_cut(cmd, 2); // terminal

        // ── 5. Bottom-up in run() order: rivers → per-ti chance accum
        //       → turns → flop chance accum → flop. In stage-timing
        //       mode regrets defer to a tagged buffer (value-identical
        //       — see encode_regret_walk).
        let with_regrets = !self.stage_timing;
        for ti in 0..self.n_turn {
            for ri in 0..self.n_rivers[ti] {
                self.encode_bu_walk(
                    &cmd, Zone::River, Some(ti), Some(ri), traverser, d, with_regrets,
                );
            }
        }
        for ti in 0..self.n_turn {
            // turn_cfv[child] = Σ_ri cp_river · river_cfv[child]
            // (outcome-ascending — the CPU's accumulation order).
            let dst_byte =
                (self.cfv_offs[self.walk_index(Zone::Turn, Some(ti), None)] as u64) * 4;
            let p = self.wparams(self.rs_count, self.nb_turn, 0, traverser, d);
            let offs_byte = (self.river_offs_off[ti] * 4) as u64;
            let cp_byte = (self.cp_river_off[ti] * 4) as u64;
            let n_out = self.n_rivers[ti] as u32;
            self.dispatch(&cmd, &self.p_chance, self.rs_count * self.nh, |enc| {
                enc.set_buffer(0, Some(&self.cfv_mega), dst_byte);
                enc.set_buffer(1, Some(&self.cfv_mega), 0);
                enc.set_buffer(2, Some(&self.river_cfv_offs_buf), offs_byte);
                enc.set_buffer(3, Some(&self.cp_river_buf), cp_byte);
                enc.set_buffer(4, Some(&self.river_seed_buf), 0);
                Self::set_params(enc, 5, &p);
                Self::set_u32(enc, 6, n_out);
            });
        }
        for ti in 0..self.n_turn {
            self.encode_bu_walk(&cmd, Zone::Turn, Some(ti), None, traverser, d, with_regrets);
        }
        {
            // flop_cfv[child] = Σ_ti cp_turn · turn_cfv[child]
            let dst_byte =
                (self.cfv_offs[self.walk_index(Zone::Flop, None, None)] as u64) * 4;
            let p = self.wparams(self.ts_count, self.nb_flop, 0, traverser, d);
            self.dispatch(&cmd, &self.p_chance, self.ts_count * self.nh, |enc| {
                enc.set_buffer(0, Some(&self.cfv_mega), dst_byte);
                enc.set_buffer(1, Some(&self.cfv_mega), 0);
                enc.set_buffer(2, Some(&self.turn_cfv_offs_buf), 0);
                enc.set_buffer(3, Some(&self.cp_turn_buf), 0);
                enc.set_buffer(4, Some(&self.turn_seed_buf), 0);
                Self::set_params(enc, 5, &p);
                Self::set_u32(enc, 6, self.n_turn as u32);
            });
        }
        self.encode_bu_walk(&cmd, Zone::Flop, None, None, traverser, d, with_regrets);

        // ── 6. Root accumulation (traverser 0 only, like run()) ──
        if traverser == 0 {
            let flop_cfv_byte =
                (self.cfv_offs[self.walk_index(Zone::Flop, None, None)] as u64) * 4;
            self.dispatch(&cmd, &self.p_root_accum, self.nh, |enc| {
                enc.set_buffer(0, Some(&self.root_cfv), 0);
                enc.set_buffer(1, Some(&self.cfv_mega), flop_cfv_byte);
                Self::set_u32(enc, 2, self.nh as u32);
            });
        }

        // ── 7. Deferred regret updates (stage-timing mode only) ──
        if self.stage_timing {
            cmd = self.stage_cut(cmd, 3); // cfv levels + chance + root
            for ti in 0..self.n_turn {
                for ri in 0..self.n_rivers[ti] {
                    self.encode_regret_walk(&cmd, Zone::River, Some(ti), Some(ri), traverser, d);
                }
            }
            for ti in 0..self.n_turn {
                self.encode_regret_walk(&cmd, Zone::Turn, Some(ti), None, traverser, d);
            }
            self.encode_regret_walk(&cmd, Zone::Flop, None, None, traverser, d);
            cmd.commit();
            self.inflight.push(cmd);
            self.inflight_stages.push(4); // regret updates
        } else {
            cmd.commit();
            self.inflight.push(cmd);
            self.inflight_stages.push(usize::MAX);
        }
    }

    /// Flush: wait for the last committed command buffer (same-queue
    /// completion is in submission order) and harvest GPU-busy
    /// attribution from every in-flight buffer.
    pub fn sync(&mut self) {
        if let Some(last) = self.inflight.last() {
            last.wait_until_completed();
        }
        for (cmd, stage) in self.inflight.drain(..).zip(self.inflight_stages.drain(..)) {
            unsafe {
                use objc::{msg_send, sel, sel_impl};
                let start: f64 = msg_send![&*cmd, GPUStartTime];
                let end: f64 = msg_send![&*cmd, GPUEndTime];
                if end > start {
                    self.gpu_busy_s += end - start;
                    if stage < self.stage_busy.len() {
                        self.stage_busy[stage] += end - start;
                    }
                }
            }
        }
    }

    /// Inject a new per-player entry range (flat `[np*nh]`, player-major) into the
    /// reach-seed buffer — for a connected DCFR co-solve that refreshes the range
    /// (e.g. the preflop calling range) between WARM GPU solves. Device regrets
    /// persist; only the flop-root reach seed changes.
    pub fn set_initial_weights(&mut self, flat: &[f32]) {
        assert!(self.inflight.is_empty(), "set_initial_weights while passes in flight");
        assert_eq!(flat.len(), self.np * self.nh, "weights must be flat [np*nh]");
        unsafe {
            std::ptr::copy_nonoverlapping(
                flat.as_ptr(),
                self.weights_buf.contents() as *mut f32,
                flat.len(),
            );
        }
    }

    /// G6 lever-1 toggle passthrough (specialized terminal pipeline).
    pub fn set_use_specialized(&mut self, on: bool) {
        self.term.set_use_specialized(on);
    }

    /// MC terminal-sampling passthrough. `m == 0` keeps exhaustive B^K
    /// enumeration; `m > 0` samples `m` opponent tuples per bucket. REQUIRED
    /// for B > 16 — an exhaustive (m=0) terminal at B≈32 explodes
    /// (32^5 ≈ 33M tuples/lane) and crashes the device watchdog.
    pub fn set_sampling(&mut self, m: u32, seed: u32) {
        self.term.set_sampling(m, seed);
    }

    /// Stage-timing probe controls (attribution: per-stage command
    /// buffers on one queue — same hazard semantics, slight wall
    /// distortion, busy attribution valid).
    pub fn set_stage_timing(&mut self, on: bool) {
        self.stage_timing = on;
    }
    pub const STAGE_NAMES: [&'static str; 5] =
        ["strategies", "reach", "terminal", "cfv+chance", "regret"];
    pub fn stage_busy(&self) -> [f64; 5] {
        self.stage_busy
    }
    pub fn reset_stage_busy(&mut self) {
        self.stage_busy = [0.0; 5];
    }

    /// The fully GPU-native mirror of `BucketedFlopCfr::run`: per
    /// iteration, per traverser, one async command buffer; single
    /// flush at the end; returns the averaged root cfv exactly like
    /// the CPU (sum over traverser-0 passes / iteration count).
    pub fn run(&mut self, num_iterations: u32) -> Vec<f32> {
        assert!(self.inflight.is_empty(), "run() while passes in flight");
        // root_cfv starts at zero each run (fresh CPU-side vec mirror).
        unsafe {
            std::ptr::write_bytes(self.root_cfv.contents() as *mut u8, 0, self.nh * 4);
        }
        let mut count = 0u32;
        for _ in 0..num_iterations {
            let d = DcfrParams::new(self.iteration);
            self.iteration += 1;
            for traverser in 0..self.np {
                self.encode_traverser_pass(traverser as u8, &d);
                if traverser == 0 {
                    count += 1;
                }
            }
        }
        self.sync();
        let out = unsafe {
            std::slice::from_raw_parts(self.root_cfv.contents() as *const f32, self.nh)
        };
        out.iter().map(|&v| v / count as f32).collect()
    }

    // ── Gate readback (call after run/sync; slices mirror the CPU
    //    solver's per-zone buffers exactly — ZoneDims concatenation) ──

    fn state_slice<'a>(&self, buf: &'a Buffer, off: usize, len: usize) -> &'a [f32] {
        assert!(self.inflight.is_empty(), "readback while passes in flight");
        unsafe {
            std::slice::from_raw_parts((buf.contents() as *const f32).add(off), len)
        }
    }

    pub fn regrets_flop(&self) -> &[f32] {
        self.state_slice(&self.regrets, 0, self.flop_stride)
    }
    pub fn cum_strategy_flop(&self) -> &[f32] {
        self.state_slice(&self.cum, 0, self.flop_stride)
    }
    pub fn regrets_turn(&self) -> &[f32] {
        self.state_slice(&self.regrets, self.turn_offset_f, self.turn_total)
    }
    pub fn cum_strategy_turn(&self) -> &[f32] {
        self.state_slice(&self.cum, self.turn_offset_f, self.turn_total)
    }
    pub fn regrets_river(&self) -> &[f32] {
        self.state_slice(&self.regrets, self.river_offset_f, self.river_total)
    }
    pub fn cum_strategy_river(&self) -> &[f32] {
        self.state_slice(&self.cum, self.river_offset_f, self.river_total)
    }

    /// Cumulative GPU-busy seconds (the G4 attribution channel,
    /// carried over: pair with wall-clock for occupancy claims).
    pub fn gpu_busy_seconds(&self) -> f64 {
        self.gpu_busy_s + self.term.gpu_busy_seconds()
    }
}
