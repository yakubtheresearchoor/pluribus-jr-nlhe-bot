//! Host side of the bucketed Design-1-collapsed terminal offload
//! (G2 step 3). One `BucketedTerminalGpu` per (flop, bucketing);
//! installs as `BucketedFlopCfr`'s terminal offload hook. The CPU
//! keeps the (measured-nearly-free, bucket-granular) walk; the GPU
//! takes filter→reduce→enumerate→expand for every terminal of a zone
//! walk in ONE dispatch (one sync per zone walk — Fix-B at the
//! offload boundary; Fix-A tg_size=32; outcome batching across flops
//! is G4's multi-flop concurrency axis).
//!
//! Dispatch configs: `stripes = 1` is the unstriped CPU-order
//! reference (the G3 bit-exact middle link); `stripes > 1` is
//! production (f64-reference + trajectory standard).

use crate::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalOffloadHook};
use crate::solver::bucketed_showdown::MAX_BUCKETS_GPU;
use crate::solver::flop_start_game::FlopChanceTable;
use crate::solver::flop_start_vector_cfr::Zone;
use crate::tree::flat::FlatTree;
use metal::{Buffer, CommandQueue, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::collections::HashMap;

use super::context::{MetalContext, MetalResult};

/// Field-for-field mirror of the Metal-side `BucketedTermParams`.
/// SINGLE Rust definition; any layout drift produces loudly-garbage
/// values caught by the G3 functional gates (and the struct is ≤ the
/// 4 KB set_bytes limit).
#[repr(C)]
#[derive(Clone, Copy)]
struct BucketedTermParams {
    num_terminals: u32,
    np: u32,
    nh: u32,
    nb: u32,
    num_opp: u32,
    stripes: u32,
    traverser: u32,
    board0: u32,
    board1: u32,
    table_off: u32,
    map_off: u32,
    nc: f32,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    /// 0 = exhaustive B^K enumeration (byte-identical to the historical
    /// path); >0 = Monte-Carlo terminal sampling at this many opponent
    /// tuples per traverser bucket (the higher-B fidelity unlock).
    sample_m: u32,
    /// Host seed; the per-(node,bt) RNG stream is a pure function of
    /// (rng_seed, node, bt) — run-to-run deterministic.
    rng_seed: u32,
}

/// Field-for-field mirror of Metal `BucketedWalkDesc`.
#[repr(C)]
#[derive(Clone, Copy)]
struct WalkDescGpu {
    reach_off: u32,
    cfv_off: u32,
    contrib_off: u32,
    fold_off: u32,
    table_off: u32,
    map_off: u32,
    board0: u32,
    board1: u32,
    nb: u32,
    stripes: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Field-for-field mirror of Metal `BucketedBatchedParams`.
#[repr(C)]
#[derive(Clone, Copy)]
struct BatchedParamsGpu {
    num_jobs: u32,
    np: u32,
    nh: u32,
    num_opp: u32,
    traverser: u32,
    nc: f32,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    /// 0 = packed (hybrid), 1 = resident (G5 native).
    mode: u32,
    /// 0 = exhaustive; >0 = MC terminal sampling (per-bucket tuples).
    sample_m: u32,
    /// Per-(node,bt) deterministic RNG seed.
    rng_seed: u32,
}

/// Walk identity for the batched cache.
pub type WalkKey = (Zone, Option<usize>, Option<usize>);

struct ZoneTerminals {
    node_ids: Buffer,
    contribs: Buffer,
    fold_masks: Buffer,
    count: u32,
}

pub struct BucketedTerminalGpu {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    fractions: Buffer,
    maps: Buffer,
    hand_cards: Buffer,
    reach_buf: Buffer,
    cfv_buf: Buffer,
    flop_terms: ZoneTerminals,
    turn_terms: ZoneTerminals,
    river_terms: ZoneTerminals,
    /// Terminal node ids per zone (host copy, for cfv readback).
    flop_term_ids: Vec<u32>,
    turn_term_ids: Vec<u32>,
    river_term_ids: Vec<u32>,
    nh: usize,
    np: usize,
    nb_flop: usize,
    nb_turn: usize,
    nb_river: usize,
    /// Per-runout float offsets into `fractions` (runout-indexed).
    table_offs: Vec<u32>,
    n_turn: usize,
    max_river: usize,
    remaining_deck: Vec<u8>,
    river_decks: Vec<Vec<u8>>,
    nc: f32,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    stripes: u32,
    /// Terminal sampling controls (single-kernel path). 0 = exhaustive.
    sample_m: u32,
    rng_seed: u32,
    gpu_busy_s: f64,
    // ── Batched path (G4 step 3) ──
    batched_pipeline: ComputePipelineState,
    /// G6 lever 1: function-constant-specialized batched pipeline
    /// (uniform per-street B only). Bit-exact with the generic one by
    /// construction; toggle for keep/discard measurement.
    batched_pipeline_spec: Option<ComputePipelineState>,
    use_specialized: bool,
    /// Concatenated [flop | turn | river] zone contribs / fold masks
    /// (i32 / u32), with per-zone element offsets.
    contribs_all: Buffer,
    folds_all: Buffer,
    zone_contrib_off: [u32; 3], // int offset per zone (flop, turn, river)
    zone_fold_off: [u32; 3],
    node_ids_all: Buffer,
    zone_nodeid_off: [u32; 3],
    /// Per-walk packed terminal cfv, filled by `fill_walks`, drained by
    /// `fill_terminals` (cache-first).
    cache: HashMap<(u8, usize, usize, u8), Vec<f32>>,
}

fn zone_code(z: Zone) -> u8 {
    match z {
        Zone::Flop => 0,
        Zone::Turn => 1,
        Zone::River => 2,
        Zone::Preflop => unreachable!(),
    }
}

fn buf_from<T: Copy>(ctx: &MetalContext, data: &[T]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(4);
    let buf = ctx
        .device()
        .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents() as *mut u8,
            std::mem::size_of_val(data),
        );
    }
    buf
}

impl BucketedTerminalGpu {
    /// Runout index in the concatenated fraction/map buffers:
    /// flop = 0, turn ti = 1 + ti, river (ti, ri) = 1 + n_turn +
    /// ti·max_river + ri. (The same outcome flattening ZoneDims uses.)
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
        // np ≥ 3 since 2026-06-12 (scope extension with the CPU
        // bucketed terminal; gated by np3_bucketed_terminal_gate on
        // the CPU side + cell_parity_live3 GPU↔CPU at size).
        assert!(tree.num_players >= 3, "bucketed terminal: np ≥ 3 (HU stays exact)");
        for nb in [bk.nb_flop, bk.nb_turn, bk.nb_river] {
            assert!(nb <= MAX_BUCKETS_GPU, "B > MAX_BUCKETS_GPU stays CPU-only");
        }
        let nb_max = bk.nb_flop.max(bk.nb_turn).max(bk.nb_river);
        assert!(
            stripes >= 1 && (32 / nb_max as u32) >= stripes,
            "stripes × max(nb) ≤ 32"
        );

        let nh = table.num_valid;
        let np = tree.num_players as usize;
        let nn = tree.num_nodes();
        let n_turn = table.remaining_deck.len();
        let max_river = table
            .remaining_deck
            .iter()
            .map(|&tc| table.river_decks[tc as usize].len())
            .max()
            .unwrap_or(0);

        let pipeline = ctx.create_pipeline("bucketed_terminal_collapsed")?;
        // The batched kernel declares optional function constants, so
        // even the GENERIC pipeline must be built via the
        // constantValues path (empty set ⇒ all constants undefined ⇒
        // runtime params — Metal validation requires this).
        let batched_pipeline =
            ctx.create_pipeline_with_constants("bucketed_terminal_collapsed_batched", &[])?;
        // Specialized variant: only when per-street B is uniform (the
        // production config) — per-zone stripes are then uniform too.
        // Specialized variant: only when per-street B is uniform (the
        // production config) — per-zone stripes are then uniform too.
        // (G6 lever 2, FMA contraction, was built on top of this and
        // DISCARDED: contraction measured ACTIVE by bit-divergence yet
        // 1.00× — the odometer is latency/divergence-bound, not
        // FP-throughput-bound. See p1_5_4_gpu_bucketed_g6_levers.rs.)
        let batched_pipeline_spec = if bk.nb_flop == bk.nb_turn && bk.nb_turn == bk.nb_river {
            let s_uniform = stripes.min(32 / bk.nb_flop as u32).max(1);
            Some(ctx.create_pipeline_with_constants(
                "bucketed_terminal_collapsed_batched",
                &[
                    (0, bk.nb_flop as u32),             // FC_NB
                    (1, tree.num_players as u32),       // FC_NP
                    (2, (tree.num_players - 1) as u32), // FC_NUM_OPP
                    (3, s_uniform),                     // FC_STRIPES
                ],
            )?)
        } else {
            None
        };

        // ── Concatenated fraction tables + maps, runout-indexed.
        //    Per-runout table size is 4·nb_zone² (divergent per-street
        //    B is the layout's reason for existing); offsets accumulate. ──
        let n_runouts = 1 + n_turn + n_turn * max_river;
        let mut fractions: Vec<f32> = Vec::new();
        let mut table_offs = vec![0u32; n_runouts];
        let mut maps = vec![u16::MAX; n_runouts * nh];
        {
            let mut put = |ridx: usize,
                           nb: usize,
                           t: &crate::solver::bucketed_showdown::BucketedRunoutTables,
                           m: &[u16],
                           fractions: &mut Vec<f32>,
                           table_offs: &mut Vec<u32>,
                           maps: &mut Vec<u16>| {
                assert_eq!(t.nb, nb);
                table_offs[ridx] = fractions.len() as u32;
                fractions.extend_from_slice(&t.f_w);
                fractions.extend_from_slice(&t.f_t);
                fractions.extend_from_slice(&t.f_l);
                fractions.extend_from_slice(&t.f_n);
                maps[ridx * nh..(ridx + 1) * nh].copy_from_slice(m);
            };
            put(0, bk.nb_flop, &bk.flop_tables, &bk.flop_map, &mut fractions, &mut table_offs, &mut maps);
            for ti in 0..n_turn {
                put(1 + ti, bk.nb_turn, &bk.turn_tables[ti], &bk.turn_map[ti], &mut fractions, &mut table_offs, &mut maps);
                for ri in 0..bk.river_map[ti].len() {
                    put(
                        1 + n_turn + ti * max_river + ri,
                        bk.nb_river,
                        &bk.river_tables[ti][ri],
                        &bk.river_map[ti][ri],
                        &mut fractions,
                        &mut table_offs,
                        &mut maps,
                    );
                }
            }
        }

        // ── Per-zone terminal metadata (zone via the gated layout) ──
        let layout = solver.gpu_layout(bk);
        let mut zone_terms =
            |zone: Zone| -> (Vec<u32>, Vec<i32>, Vec<u32>) {
                let mut ids = Vec::new();
                let mut contribs = Vec::new();
                let mut folds = Vec::new();
                for idx in 0..nn {
                    if layout.zone_of(idx) != zone || !tree.nodes[idx].is_terminal() {
                        continue;
                    }
                    ids.push(idx as u32);
                    for p in 0..np {
                        contribs.push(tree.get_contribution(idx, p as u8));
                    }
                    folds.push(tree.get_folded_mask(idx) as u32);
                }
                (ids, contribs, folds)
            };
        let (f_ids, f_c, f_m) = zone_terms(Zone::Flop);
        let (t_ids, t_c, t_m) = zone_terms(Zone::Turn);
        let (r_ids, r_c, r_m) = zone_terms(Zone::River);
        let mk = |ids: &Vec<u32>, c: &Vec<i32>, m: &Vec<u32>| ZoneTerminals {
            node_ids: buf_from(ctx, ids),
            contribs: buf_from(ctx, c),
            fold_masks: buf_from(ctx, m),
            count: ids.len() as u32,
        };

        // Concatenated zone contribs/folds for the batched kernel.
        let mut contribs_cat: Vec<i32> = Vec::new();
        let mut folds_cat: Vec<u32> = Vec::new();
        let mut zone_contrib_off = [0u32; 3];
        let mut zone_fold_off = [0u32; 3];
        for (zi, (c, m)) in [(&f_c, &f_m), (&t_c, &t_m), (&r_c, &r_m)].iter().enumerate() {
            zone_contrib_off[zi] = contribs_cat.len() as u32;
            zone_fold_off[zi] = folds_cat.len() as u32;
            contribs_cat.extend_from_slice(c);
            folds_cat.extend_from_slice(m);
        }
        let contribs_all = buf_from(ctx, &contribs_cat);
        let folds_all = buf_from(ctx, &folds_cat);
        // Concatenated zone node ids (resident-mode terminal indexing);
        // zone offsets parallel zone_fold_off (per-terminal grain).
        let mut node_ids_cat: Vec<u32> = Vec::new();
        let mut zone_nodeid_off = [0u32; 3];
        for (zi, ids) in [&f_ids, &t_ids, &r_ids].iter().enumerate() {
            zone_nodeid_off[zi] = node_ids_cat.len() as u32;
            node_ids_cat.extend_from_slice(ids);
        }
        let node_ids_all = buf_from(ctx, &node_ids_cat);

        let reach_buf = ctx.device().new_buffer(
            (nn * np * nh * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cfv_buf = ctx
            .device()
            .new_buffer((nn * nh * 4) as u64, MTLResourceOptions::StorageModeShared);

        Ok(Self {
            queue: ctx.queue().to_owned(),
            pipeline,
            fractions: buf_from(ctx, &fractions),
            maps: buf_from(ctx, &maps),
            hand_cards: buf_from(ctx, &table.hand_cards),
            reach_buf,
            cfv_buf,
            flop_terms: mk(&f_ids, &f_c, &f_m),
            turn_terms: mk(&t_ids, &t_c, &t_m),
            river_terms: mk(&r_ids, &r_c, &r_m),
            flop_term_ids: f_ids,
            turn_term_ids: t_ids,
            river_term_ids: r_ids,
            nh,
            np,
            nb_flop: bk.nb_flop,
            nb_turn: bk.nb_turn,
            nb_river: bk.nb_river,
            table_offs,
            n_turn,
            max_river,
            remaining_deck: table.remaining_deck.clone(),
            river_decks: table.river_decks.clone(),
            nc: table.num_combinations as f32,
            starting_pot: tree.starting_pot,
            rake_rate: tree.rake_rate as f32,
            rake_cap: tree.rake_cap as f32,
            stripes,
            sample_m: 0,
            rng_seed: 0,
            gpu_busy_s: 0.0,
            batched_pipeline,
            batched_pipeline_spec,
            use_specialized: true,
            contribs_all,
            folds_all,
            zone_contrib_off,
            zone_fold_off,
            node_ids_all,
            zone_nodeid_off,
            cache: HashMap::new(),
        })
    }

    /// Enable/disable Monte-Carlo terminal sampling on the single-kernel
    /// path (`fill_terminals`). `m == 0` restores exhaustive enumeration
    /// (byte-identical to the historical path). `m > 0` draws `m` opponent
    /// tuples per traverser bucket from the reach marginal, importance-
    /// weighted (unbiased). `seed` keys the deterministic RNG stream.
    pub fn set_sampling(&mut self, m: u32, seed: u32) {
        self.sample_m = m;
        self.rng_seed = seed;
    }

    /// Fill every terminal cfv slice for one zone walk. Returns true
    /// (handled) — the walk skips its terminal branch.
    pub fn fill_terminals(
        &mut self,
        zone: Zone,
        tc: Option<usize>,
        rc: Option<usize>,
        traverser: u8,
        reach: &[f32],
        cfv: &mut [f32],
    ) -> bool {
        // Batched cache: if fill_walks already computed this walk, drain it.
        let key = (zone_code(zone), tc.unwrap_or(usize::MAX), rc.unwrap_or(usize::MAX), traverser);
        if let Some(packed) = self.cache.remove(&key) {
            let ids: &[u32] = match zone {
                Zone::Flop => &self.flop_term_ids,
                Zone::Turn => &self.turn_term_ids,
                Zone::River => &self.river_term_ids,
                Zone::Preflop => unreachable!(),
            };
            for (lt, &nid) in ids.iter().enumerate() {
                let dst = nid as usize * self.nh;
                cfv[dst..dst + self.nh]
                    .copy_from_slice(&packed[lt * self.nh..(lt + 1) * self.nh]);
            }
            return true;
        }
        let (terms, ids): (&ZoneTerminals, &[u32]) = match zone {
            Zone::Flop => (&self.flop_terms, &self.flop_term_ids),
            Zone::Turn => (&self.turn_terms, &self.turn_term_ids),
            Zone::River => (&self.river_terms, &self.river_term_ids),
            Zone::Preflop => unreachable!(),
        };
        if terms.count == 0 {
            return true;
        }
        // Board cards for the filter (verbatim CPU board_cards logic).
        let (board0, board1) = match zone {
            Zone::Flop => (0xFFFFu32, 0xFFFFu32),
            Zone::Turn => (self.remaining_deck[tc.unwrap()] as u32, 0xFFFF),
            Zone::River => {
                let tcc = self.remaining_deck[tc.unwrap()];
                let rcc = self.river_decks[tcc as usize][rc.unwrap()];
                (tcc as u32, rcc as u32)
            }
            Zone::Preflop => unreachable!(),
        };
        let ridx = self.runout_index(zone, tc, rc);
        let nb_zone = match zone {
            Zone::Flop => self.nb_flop,
            Zone::Turn => self.nb_turn,
            Zone::River => self.nb_river,
            Zone::Preflop => unreachable!(),
        };
        // Per-zone stripes: as wide as the lane budget allows, capped
        // at the requested production setting (1 = unstriped reference
        // everywhere — the gate config).
        let stripes_zone = self.stripes.min(32 / nb_zone as u32).max(1);
        let params = BucketedTermParams {
            num_terminals: terms.count,
            np: self.np as u32,
            nh: self.nh as u32,
            nb: nb_zone as u32,
            num_opp: (self.np - 1) as u32,
            stripes: stripes_zone,
            traverser: traverser as u32,
            board0,
            board1,
            table_off: self.table_offs[ridx],
            map_off: (ridx * self.nh) as u32,
            nc: self.nc,
            starting_pot: self.starting_pot,
            rake_rate: self.rake_rate,
            rake_cap: self.rake_cap,
            sample_m: self.sample_m,
            rng_seed: self.rng_seed,
        };

        // Stage reach (unified memory; v1 copies, the no-copy variant
        // writes the walk's reach directly into this buffer later).
        unsafe {
            std::ptr::copy_nonoverlapping(
                reach.as_ptr(),
                self.reach_buf.contents() as *mut f32,
                reach.len(),
            );
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<BucketedTermParams>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        enc.set_buffer(1, Some(&terms.node_ids), 0);
        enc.set_buffer(2, Some(&terms.contribs), 0);
        enc.set_buffer(3, Some(&terms.fold_masks), 0);
        enc.set_buffer(4, Some(&self.reach_buf), 0);
        enc.set_buffer(5, Some(&self.fractions), 0);
        enc.set_buffer(6, Some(&self.maps), 0);
        enc.set_buffer(7, Some(&self.hand_cards), 0);
        enc.set_buffer(8, Some(&self.cfv_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize { width: terms.count as u64, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        // GPU-busy attribution: per-command-buffer GPU timestamps.
        // metal-0.33 doesn't expose GPUStartTime/GPUEndTime; raw objc
        // selectors (return CFTimeInterval seconds, valid post-completion).
        unsafe {
            use objc::{msg_send, sel, sel_impl};
            let start: f64 = msg_send![cmd, GPUStartTime];
            let end: f64 = msg_send![cmd, GPUEndTime];
            if end > start {
                self.gpu_busy_s += end - start;
            }
        }

        // Read back only the terminal slices.
        let out = unsafe {
            std::slice::from_raw_parts(self.cfv_buf.contents() as *const f32, cfv.len())
        };
        for &nid in ids {
            let base = nid as usize * self.nh;
            cfv[base..base + self.nh].copy_from_slice(&out[base..base + self.nh]);
        }
        true
    }

    /// Replace the command queue (G4 multi-flop concurrency: one
    /// INDEPENDENT queue per concurrent flop, so cross-flop submission
    /// is parallel by construction — the orchestration-serialization
    /// failure mode designed out before it is measured).
    pub fn set_queue(&mut self, queue: CommandQueue) {
        self.queue = queue;
    }

    /// The batched pipeline in effect: specialized when available and
    /// enabled, generic otherwise (and always for divergent dims).
    fn batched_pipe(&self) -> &ComputePipelineState {
        if self.use_specialized {
            self.batched_pipeline_spec.as_ref().unwrap_or(&self.batched_pipeline)
        } else {
            &self.batched_pipeline
        }
    }

    /// G6 lever-1 toggle (keep/discard measurement).
    pub fn set_use_specialized(&mut self, on: bool) {
        self.use_specialized = on;
    }

    /// Cumulative GPU-busy seconds across this object's dispatches
    /// (per-command-buffer GPU timestamps). The attribution channel:
    /// wall-clock ratio ≈ 1 with LOW busy-fraction = orchestration
    /// serialization; with HIGH busy-fraction = chip saturation.
    pub fn gpu_busy_seconds(&self) -> f64 {
        self.gpu_busy_s
    }

    /// G4 step 3: compute ALL walks' terminals in ONE dispatch (the
    /// occupancy fix — ~1k-3.3k threadgroups vs 100-550 per-walk).
    /// Results land in the cache, drained by `fill_terminals` as the
    /// CPU walk reaches each (zone, outcome). `reaches[w]` is the full
    /// [nn × np × nh] reach for walk w (the prepass keeps them).
    pub fn fill_walks(
        &mut self,
        walks: &[WalkKey],
        traverser: u8,
        reaches: &[&[f32]],
    ) -> () {
        assert_eq!(walks.len(), reaches.len());
        let nh = self.nh;
        let np = self.np;

        // Build jobs + descs + packed reach.
        let mut jobs: Vec<[u32; 2]> = Vec::new();
        let mut descs: Vec<WalkDescGpu> = Vec::with_capacity(walks.len());
        let mut reach_packed: Vec<f32> = Vec::new();
        let mut cfv_off = 0u32;
        let mut cfv_total = 0usize;
        for (w, &(zone, tc, rc)) in walks.iter().enumerate() {
            let (ids, zi): (&[u32], usize) = match zone {
                Zone::Flop => (&self.flop_term_ids, 0),
                Zone::Turn => (&self.turn_term_ids, 1),
                Zone::River => (&self.river_term_ids, 2),
                Zone::Preflop => unreachable!(),
            };
            let nb_zone = match zone {
                Zone::Flop => self.nb_flop,
                Zone::Turn => self.nb_turn,
                Zone::River => self.nb_river,
                Zone::Preflop => unreachable!(),
            };
            let stripes_zone = self.stripes.min(32 / nb_zone as u32).max(1);
            let (board0, board1) = match zone {
                Zone::Flop => (0xFFFFu32, 0xFFFFu32),
                Zone::Turn => (self.remaining_deck[tc.unwrap()] as u32, 0xFFFF),
                Zone::River => {
                    let tcc = self.remaining_deck[tc.unwrap()];
                    let rcc = self.river_decks[tcc as usize][rc.unwrap()];
                    (tcc as u32, rcc as u32)
                }
                Zone::Preflop => unreachable!(),
            };
            let ridx = self.runout_index(zone, tc, rc);
            let reach_off = reach_packed.len() as u32;
            let reach = reaches[w];
            for &nid in ids {
                let base = nid as usize * np * nh;
                reach_packed.extend_from_slice(&reach[base..base + np * nh]);
            }
            descs.push(WalkDescGpu {
                reach_off,
                cfv_off,
                contrib_off: self.zone_contrib_off[zi],
                fold_off: self.zone_fold_off[zi],
                table_off: self.table_offs[ridx],
                map_off: (ridx * nh) as u32,
                board0,
                board1,
                nb: nb_zone as u32,
                stripes: stripes_zone,
                _pad0: self.zone_nodeid_off[zi], // node_off (resident mode)
                _pad1: 0,
            });
            for lt in 0..ids.len() {
                jobs.push([w as u32, lt as u32]);
            }
            cfv_off += (ids.len() * nh) as u32;
            cfv_total += ids.len() * nh;
        }
        if jobs.is_empty() {
            return;
        }

        let params = BatchedParamsGpu {
            num_jobs: jobs.len() as u32,
            np: np as u32,
            nh: nh as u32,
            num_opp: (np - 1) as u32,
            traverser: traverser as u32,
            nc: self.nc,
            starting_pot: self.starting_pot,
            rake_rate: self.rake_rate,
            rake_cap: self.rake_cap,
            mode: 0,
            sample_m: self.sample_m,
            rng_seed: self.rng_seed,
        };

        // Per-call buffers (job/desc small; reach the big one).
        let jobs_buf = {
            let flat: Vec<u32> = jobs.iter().flatten().copied().collect();
            let b = self
                .queue
                .device()
                .new_buffer((flat.len() * 4).max(4) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(flat.as_ptr(), b.contents() as *mut u32, flat.len());
            }
            b
        };
        let descs_buf = {
            let bytes = std::mem::size_of::<WalkDescGpu>() * descs.len();
            let b = self
                .queue
                .device()
                .new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    descs.as_ptr() as *const u8,
                    b.contents() as *mut u8,
                    bytes,
                );
            }
            b
        };
        let reach_pk_buf = {
            let b = self.queue.device().new_buffer(
                (reach_packed.len() * 4).max(4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    reach_packed.as_ptr(),
                    b.contents() as *mut f32,
                    reach_packed.len(),
                );
            }
            b
        };
        let cfv_pk_buf = self
            .queue
            .device()
            .new_buffer((cfv_total * 4).max(4) as u64, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.batched_pipe());
        enc.set_bytes(
            0,
            std::mem::size_of::<BatchedParamsGpu>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        enc.set_buffer(1, Some(&jobs_buf), 0);
        enc.set_buffer(2, Some(&descs_buf), 0);
        enc.set_buffer(3, Some(&self.contribs_all), 0);
        enc.set_buffer(4, Some(&self.folds_all), 0);
        enc.set_buffer(5, Some(&reach_pk_buf), 0);
        enc.set_buffer(6, Some(&self.fractions), 0);
        enc.set_buffer(7, Some(&self.maps), 0);
        enc.set_buffer(8, Some(&self.hand_cards), 0);
        enc.set_buffer(9, Some(&cfv_pk_buf), 0);
        enc.set_buffer(10, Some(&self.node_ids_all), 0);
        enc.dispatch_thread_groups(
            MTLSize { width: jobs.len() as u64, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        unsafe {
            use objc::{msg_send, sel, sel_impl};
            let start: f64 = msg_send![cmd, GPUStartTime];
            let end: f64 = msg_send![cmd, GPUEndTime];
            if end > start {
                self.gpu_busy_s += end - start;
            }
        }

        // Slice packed cfv into per-walk cache entries.
        let out = unsafe {
            std::slice::from_raw_parts(cfv_pk_buf.contents() as *const f32, cfv_total)
        };
        for (w, &(zone, tc, rc)) in walks.iter().enumerate() {
            let n_terms = match zone {
                Zone::Flop => self.flop_term_ids.len(),
                Zone::Turn => self.turn_term_ids.len(),
                Zone::River => self.river_term_ids.len(),
                Zone::Preflop => unreachable!(),
            };
            let off = descs[w].cfv_off as usize;
            let key = (
                zone_code(zone),
                tc.unwrap_or(usize::MAX),
                rc.unwrap_or(usize::MAX),
                traverser,
            );
            self.cache.insert(key, out[off..off + n_terms * nh].to_vec());
        }
    }

    /// G5: encode the batched terminal in RESIDENT mode (mode = 1) onto
    /// the caller's command buffer — reach read from each walk's
    /// resident region BY NODE ID (`node_ids_all[node_off + lt]`), cfv
    /// written to each walk's resident region at the node row. No
    /// commit, no wait, no packing: ordering comes from the caller's
    /// hazard-tracked command stream (the Fix-C pattern), which is the
    /// entire point of the G5 native path.
    ///
    /// `reach_offs[w]` / `cfv_offs[w]` are FLOAT offsets of walk w's
    /// region in the mega buffers (walk order = `walks`).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_walks_resident(
        &self,
        cmd: &metal::CommandBufferRef,
        walks: &[WalkKey],
        traverser: u8,
        reach_mega: &Buffer,
        cfv_mega: &Buffer,
        reach_offs: &[u32],
        cfv_offs: &[u32],
    ) {
        assert_eq!(walks.len(), reach_offs.len());
        assert_eq!(walks.len(), cfv_offs.len());
        let nh = self.nh;
        let mut jobs: Vec<[u32; 2]> = Vec::new();
        let mut descs: Vec<WalkDescGpu> = Vec::with_capacity(walks.len());
        for (w, &(zone, tc, rc)) in walks.iter().enumerate() {
            let (n_terms, zi): (usize, usize) = match zone {
                Zone::Flop => (self.flop_term_ids.len(), 0),
                Zone::Turn => (self.turn_term_ids.len(), 1),
                Zone::River => (self.river_term_ids.len(), 2),
                Zone::Preflop => unreachable!(),
            };
            let nb_zone = match zone {
                Zone::Flop => self.nb_flop,
                Zone::Turn => self.nb_turn,
                Zone::River => self.nb_river,
                Zone::Preflop => unreachable!(),
            };
            let stripes_zone = self.stripes.min(32 / nb_zone as u32).max(1);
            let (board0, board1) = match zone {
                Zone::Flop => (0xFFFFu32, 0xFFFFu32),
                Zone::Turn => (self.remaining_deck[tc.unwrap()] as u32, 0xFFFF),
                Zone::River => {
                    let tcc = self.remaining_deck[tc.unwrap()];
                    let rcc = self.river_decks[tcc as usize][rc.unwrap()];
                    (tcc as u32, rcc as u32)
                }
                Zone::Preflop => unreachable!(),
            };
            let ridx = self.runout_index(zone, tc, rc);
            descs.push(WalkDescGpu {
                reach_off: reach_offs[w],
                cfv_off: cfv_offs[w],
                contrib_off: self.zone_contrib_off[zi],
                fold_off: self.zone_fold_off[zi],
                table_off: self.table_offs[ridx],
                map_off: (ridx * nh) as u32,
                board0,
                board1,
                nb: nb_zone as u32,
                stripes: stripes_zone,
                _pad0: self.zone_nodeid_off[zi], // node_off (resident mode)
                _pad1: 0,
            });
            for lt in 0..n_terms {
                jobs.push([w as u32, lt as u32]);
            }
        }
        if jobs.is_empty() {
            return;
        }
        let params = BatchedParamsGpu {
            num_jobs: jobs.len() as u32,
            np: self.np as u32,
            nh: nh as u32,
            num_opp: (self.np - 1) as u32,
            traverser: traverser as u32,
            nc: self.nc,
            starting_pot: self.starting_pot,
            rake_rate: self.rake_rate,
            rake_cap: self.rake_cap,
            mode: 1,
            sample_m: self.sample_m,
            rng_seed: self.rng_seed,
        };
        let jobs_flat: Vec<u32> = jobs.iter().flatten().copied().collect();
        let jobs_buf = {
            let b = self.queue.device().new_buffer(
                (jobs_flat.len() * 4).max(4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    jobs_flat.as_ptr(),
                    b.contents() as *mut u32,
                    jobs_flat.len(),
                );
            }
            b
        };
        let descs_buf = {
            let bytes = std::mem::size_of::<WalkDescGpu>() * descs.len();
            let b = self
                .queue
                .device()
                .new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    descs.as_ptr() as *const u8,
                    b.contents() as *mut u8,
                    bytes,
                );
            }
            b
        };
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.batched_pipe());
        enc.set_bytes(
            0,
            std::mem::size_of::<BatchedParamsGpu>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        enc.set_buffer(1, Some(&jobs_buf), 0);
        enc.set_buffer(2, Some(&descs_buf), 0);
        enc.set_buffer(3, Some(&self.contribs_all), 0);
        enc.set_buffer(4, Some(&self.folds_all), 0);
        enc.set_buffer(5, Some(reach_mega), 0);
        enc.set_buffer(6, Some(&self.fractions), 0);
        enc.set_buffer(7, Some(&self.maps), 0);
        enc.set_buffer(8, Some(&self.hand_cards), 0);
        enc.set_buffer(9, Some(cfv_mega), 0);
        enc.set_buffer(10, Some(&self.node_ids_all), 0);
        enc.dispatch_thread_groups(
            MTLSize { width: jobs.len() as u64, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.end_encoding();
    }

    /// Consume self into a `BucketedFlopCfr` terminal offload hook.
    pub fn into_hook(mut self) -> TerminalOffloadHook {
        Box::new(move |zone, tc, rc, traverser, reach, cfv| {
            self.fill_terminals(zone, tc, rc, traverser, reach, cfv)
        })
    }
}
