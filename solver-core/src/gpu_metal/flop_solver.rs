/// Metal-backed flop-start vector CFR solver.
///
/// Implements the three-zone pipeline (river → turn → flop) with PER-OUTCOME
/// dimensional regrets matching the validated CPU FlopStartVectorCfr.
///
/// Architecture: Each outcome (turn card, river card) has its own independent
/// regret slot in the dimensional layout. The batched kernel writes directly
/// to per-outcome regret slots — no atomic accumulation needed.
///
/// Regret layout (contiguous buffer: flop | turn | river):
///   Flop:  regrets[infoset * MAX_NA_POSTFLOP * nh + a * nh + h]
///   Turn:  regrets[flop_total + tc * turn_stride + infoset * MAX_NA_POSTFLOP * nh + a * nh + h]
///   River: regrets[flop_total + turn_total + (tc*max_river+rc) * river_stride + infoset * MAX_NA_POSTFLOP * nh + a * nh + h]
///
/// Strategy computed from regrets via regret matching, separately per outcome.
/// DCFR discount applied inline during the bottom-up pass.

use crate::gpu_metal::{MetalBuffer, MetalContext};
use crate::solver::flop_start_game::FlopStartGame;
use crate::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use crate::tree::flat::{FlatTree, MAX_NA_POSTFLOP};
use metal::ComputePipelineState;
use std::time::{Duration, Instant};

/// GPU-side mirror of CPU's `RiverPersistenceMode`. In `InMemory` (default),
/// the river region of `d_regrets`/`d_strategy`/`d_cum_strategy` holds all
/// `(tc, rc)` pairs simultaneously — same layout as before 2.B landed.
/// In `DiskBacked`, two files hold the per-pair persistent state and the
/// caller is responsible for invoking `load_river_pair_gpu(ti, ri)` before
/// any per-pair read and `save_river_pair_gpu(ti, ri)` after any per-pair
/// mutation. The file format is the same as CPU's `DiskBacked` so the same
/// files can in principle be shared (not yet exercised; future work).
///
/// Step 2.B.1 (foundation): file I/O round-trips into the existing
/// in-place buffer layout. Buffers stay full-sized; the disk path is
/// equivalent to a no-op modulo seek/read/write. Validation: write known
/// pattern → save → zero in-buffer → load → verify recovered.
///
/// Step 2.B.2 (integration): shrink river buffers to `river_stride`
/// scratch, change kernel dispatch offsets to use `outcome_idx = 0` in
/// `DiskBacked`, wire load/save into `run_one_iter`, add I/O fields to
/// `StageProfile`.
/// Zone selector for inter-zone offset computations on the postflop
/// d_regrets / d_strategy / d_cum_strategy buffers. All three buffers share
/// the same layout — they were allocated in lockstep with identical strides —
/// so a single set of zone offsets describes all three.
///
/// The variants carry the outcome index because each zone other than Flop
/// has per-outcome substructure:
///   - Flop has exactly one outcome (no chance dimension at the flop root)
///   - Turn has one outcome per turn card index `ti`
///   - River has one outcome per (ti, ri) pair, encoded as `outcome_idx`
///
/// See `MetalFlopStartSolver::infoset_float_offset` /
/// `infoset_byte_offset` — the consolidated offset helpers (Phase 2). The
/// inline form that this replaces was a known stride-bug source — flop_solver
/// historically miscounted outcome offsets at sites like `disk_backed_load`
/// because each site re-derived `zone_offset + outcome_idx * zone_stride`
/// from raw fields, so centralizing it eliminates the duplication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferZone {
    Flop,
    Turn { ti: usize },
    River { outcome_idx: usize },
}

#[derive(Clone, Debug)]
pub enum GpuRiverMode {
    /// All [n_turn × max_river × river_stride] f32 stays in GPU memory.
    /// Default; matches the layout the solver used before 2.B.
    InMemory,
    /// Two binary files hold the full per-(tc, rc) state for the river
    /// region of d_regrets and d_cum_strategy. The GPU buffers are
    /// streamed via load/save_river_pair_gpu.
    DiskBacked {
        regrets_path: std::path::PathBuf,
        cum_strategy_path: std::path::PathBuf,
    },
}

/// Per-stage wall-clock budget accumulated across one or more iterations of
/// `run_profiled()`. Each field is the total time spent in that stage,
/// summed across iters and traversers.
///
/// Fields mirror the structure of the production GPU iter loop in
/// `MetalFlopStartSolver::run_one_iter` so adding/removing stages stays
/// localized. Step 2.B will add I/O stages (`load_river_pair`,
/// `save_river_pair`) here when DiskBacked GPU mode lands.
#[derive(Default, Clone, Debug)]
pub struct StageProfile {
    /// Total wall-clock for the profiled run (covers all iters and stages,
    /// including host overhead not attributed to any stage).
    pub total: Duration,
    pub compute_strategies: Duration,
    pub compute_reach_flop: Duration,
    pub compute_reach_turn: Duration,
    pub compute_reach_river: Duration,
    /// `bottom_up_river` includes the fused unified factored showdown CFV
    /// kernel — the kernels are not separable at the dispatch level.
    pub bottom_up_river: Duration,
    pub bottom_up_turn: Duration,
    pub bottom_up_flop: Duration,
    pub chance_accumulate_river: Duration,
    pub chance_finalize_river: Duration,
    pub chance_accumulate_turn: Duration,
    pub chance_finalize_turn: Duration,
    pub zero_buffer_total: Duration,
    /// 2.B.2: time spent in `load_river_pair_gpu` (file → GPU scratch).
    /// Zero in InMemory mode (load is a no-op there).
    pub load_river_pair: Duration,
    /// 2.B.2: time spent in `save_river_pair_gpu` (GPU scratch → file).
    /// Zero in InMemory mode.
    pub save_river_pair: Duration,
    /// 2.B.2: time spent in `compute_river_strategy_pair` (per-pair river
    /// strategy compute). Zero in InMemory mode — there the batched call
    /// inside `compute_all_strategies` does this work and the time lands
    /// in `compute_strategies` instead.
    pub compute_river_strategy_pair: Duration,
}

impl StageProfile {
    pub fn new() -> Self { Self::default() }

    /// Sum of all attributed stage durations. The gap between `total` and
    /// `attributed()` is host orchestration overhead (command-buffer setup,
    /// dispatch overhead not bracketed inside a stage, etc.).
    pub fn attributed(&self) -> Duration {
        self.compute_strategies
            + self.compute_reach_flop
            + self.compute_reach_turn
            + self.compute_reach_river
            + self.bottom_up_river
            + self.bottom_up_turn
            + self.bottom_up_flop
            + self.chance_accumulate_river
            + self.chance_finalize_river
            + self.chance_accumulate_turn
            + self.chance_finalize_turn
            + self.zero_buffer_total
            + self.load_river_pair
            + self.save_river_pair
            + self.compute_river_strategy_pair
    }

    /// Format a one-line-per-stage report with percentages of `self.total`.
    pub fn report(&self) -> String {
        let total_s = self.total.as_secs_f64().max(1e-12);
        let fmt = |name: &str, d: Duration| -> String {
            let s = d.as_secs_f64();
            format!("  {:30} {:>10.4} s  ({:>5.1}%)\n", name, s, s / total_s * 100.0)
        };
        let mut out = String::new();
        out.push_str(&format!("=== StageProfile (total {:.4} s) ===\n", total_s));
        out.push_str(&fmt("compute_strategies", self.compute_strategies));
        out.push_str(&fmt("compute_reach_flop", self.compute_reach_flop));
        out.push_str(&fmt("compute_reach_turn", self.compute_reach_turn));
        out.push_str(&fmt("compute_reach_river", self.compute_reach_river));
        out.push_str(&fmt("bottom_up_river", self.bottom_up_river));
        out.push_str(&fmt("bottom_up_turn", self.bottom_up_turn));
        out.push_str(&fmt("bottom_up_flop", self.bottom_up_flop));
        out.push_str(&fmt("chance_accumulate_river", self.chance_accumulate_river));
        out.push_str(&fmt("chance_finalize_river", self.chance_finalize_river));
        out.push_str(&fmt("chance_accumulate_turn", self.chance_accumulate_turn));
        out.push_str(&fmt("chance_finalize_turn", self.chance_finalize_turn));
        out.push_str(&fmt("zero_buffer_total", self.zero_buffer_total));
        out.push_str(&fmt("load_river_pair (I/O)", self.load_river_pair));
        out.push_str(&fmt("save_river_pair (I/O)", self.save_river_pair));
        out.push_str(&fmt("compute_river_strategy_pair", self.compute_river_strategy_pair));
        out.push_str(&fmt("attributed (sum)", self.attributed()));
        let unattributed = self.total.saturating_sub(self.attributed());
        out.push_str(&fmt("unattributed (host overhead)", unattributed));
        out
    }
}

/// Wrap `$body` with `Instant::now()` bracketing IFF `$profile` is `Some`,
/// adding the elapsed time to `$profile.$field`. When `$profile` is `None`,
/// the body runs without any timing overhead (zero-cost-when-disabled).
///
/// Designed for use inside `run_one_iter` so the same loop body serves both
/// `run()` (profile = None) and `run_profiled()` (profile = Some).
macro_rules! time_stage {
    ($profile:expr, $field:ident, $body:expr) => {{
        if let Some(p) = $profile.as_deref_mut() {
            let __t = std::time::Instant::now();
            let __r = $body;
            p.$field += __t.elapsed();
            __r
        } else {
            $body
        }
    }};
}

const UNUSED: u32 = u32::MAX;

pub struct DcfrParams {
    pub alpha_t: f32,
    pub beta_t: f32,
    pub gamma_t: f32,
}

impl DcfrParams {
    pub fn new(current_iteration: u32) -> Self {
        let nearest_lower_power_of_4 = match current_iteration {
            0 => 0u32,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };
        let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
        let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;
        let pow_alpha = t_alpha * t_alpha.sqrt();
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);
        Self {
            alpha_t: (pow_alpha / (pow_alpha + 1.0)) as f32,
            beta_t: 0.5,
            gamma_t: pow_gamma as f32,
        }
    }
}

/// Metal-backed flop-start VCFR solver.
pub struct MetalFlopStartSolver {
    // Pipelines
    strategies_pipeline: ComputePipelineState,          // single outcome
    strategies_batched_pipeline: ComputePipelineState,   // per-outcome batched
    init_reach_pipeline: ComputePipelineState,
    top_down_pipeline: ComputePipelineState,
    bottom_up_pipeline: ComputePipelineState,
    // Flop port (M2 follow-up): tg-parallel version of vcfr_bottom_up.
    // Used by bottom_up_flop when num_opp >= 2. HU keeps the serial kernel.
    bottom_up_tg_parallel_pipeline: ComputePipelineState,
    batched_pipeline: ComputePipelineState,
    // Step 2.D.28: threadgroup-parallel kernel for 6-max (num_opp >= 3).
    // Same buffer layout as batched_pipeline; dispatch uses 1 threadgroup
    // per (outcome, node) with TG_SIZE threads cooperating on each node.
    batched_tg_parallel_pipeline: ComputePipelineState,
    chance_accum_pipeline: ComputePipelineState,
    chance_final_pipeline: ComputePipelineState,
    chance_grouped_pipeline: ComputePipelineState,
    seed_reach_pipeline: ComputePipelineState,
    zero_pipeline: ComputePipelineState,

    // Tree structure (read-only)
    d_nodes: MetalBuffer<crate::tree::flat::FlatNode>,
    d_children: MetalBuffer<u32>,
    d_contributions: MetalBuffer<i32>,
    d_folded_masks: MetalBuffer<u16>,
    d_infoset_offsets: MetalBuffer<u32>,

    // Per-zone decision node IDs and infoset offsets for strategy computation
    d_flop_decision_ids: MetalBuffer<u32>,
    d_flop_infoset_offsets: MetalBuffer<u32>,
    d_turn_decision_ids: MetalBuffer<u32>,
    d_turn_infoset_offsets: MetalBuffer<u32>,
    d_river_decision_ids: MetalBuffer<u32>,
    d_river_infoset_offsets: MetalBuffer<u32>,

    // Solver state: per-outcome dimensional layout matching CPU
    d_regrets: MetalBuffer<f32>,
    d_strategy: MetalBuffer<f32>,
    d_cum_strategy: MetalBuffer<f32>,
    d_reach: MetalBuffer<f32>,   // [nn * np * nh]
    d_turn_reach: MetalBuffer<f32>, // [nn * np * nh] for turn zone per-tc
    d_river_reach: MetalBuffer<f32>, // [nn * np * nh] for river zone per-(tc,rc)
    d_cfv: MetalBuffer<f32>,     // [nn * nh]

    // ─── Slice 2 Phase B Step 5: chokepoint instrumentation marker ───
    // Per-(terminal-node, hand) marker buffer sized [nn × nh] u8. Written
    // by the two production payoff-computing kernels (vcfr_bottom_up at
    // buffer 18, vcfr_bottom_up_batched at buffer 20) right after the
    // multiway_brute_force_showdown helper returns. After a solve, the
    // host downloads this and asserts every terminal-node-hand cell is
    // marked (1 = rake-applied, 2 = rake-correctly-skipped per
    // no-flop-no-drop). An unmarked cell (0) means that terminal
    // bypassed the chokepoint — a bug. The standing CI guard against
    // any future change reintroducing a payoff path that doesn't apply
    // rake. Per the lead: "the instrumentation is the protection for all
    // future kernel work, not just a Phase B closeout."
    d_rake_marker: MetalBuffer<u8>,

    // Game data (read-only)
    d_initial_weight: MetalBuffer<f32>,
    d_hand_cards: MetalBuffer<u8>,
    d_sorted_opp_strength: MetalBuffer<u16>,
    d_sorted_opp_indices: MetalBuffer<u16>,
    d_sorted_pl_strength: MetalBuffer<u16>,
    d_sorted_pl_indices: MetalBuffer<u16>,

    // Per-zone sorted arrays (full 52-card deck indexed)
    d_river_sorted_str: MetalBuffer<u16>,
    d_river_sorted_idx: MetalBuffer<u16>,
    d_river_pl_str: MetalBuffer<u16>,
    d_river_pl_idx: MetalBuffer<u16>,
    d_turn_sorted_str: MetalBuffer<u16>,
    d_turn_sorted_idx: MetalBuffer<u16>,
    d_turn_pl_str: MetalBuffer<u16>,
    d_turn_pl_idx: MetalBuffer<u16>,
    d_river_chance_prob: MetalBuffer<f32>,
    d_river_board_mask: MetalBuffer<f32>,
    d_turn_chance_prob: MetalBuffer<f32>,

    // Batched CFV buffers
    d_river_cfv_batch: MetalBuffer<f32>,
    d_river_accum: MetalBuffer<f32>,
    d_turn_cfv_batch: MetalBuffer<f32>,
    d_turn_accum: MetalBuffer<f32>,

    // Chance children
    d_river_chance_children: MetalBuffer<u32>,
    d_debug_out: MetalBuffer<f32>,
    d_turn_chance_children: MetalBuffer<u32>,

    // Zone nodes per level
    d_river_zone_nodes: Vec<Option<MetalBuffer<u32>>>,
    d_turn_zone_nodes: Vec<Option<MetalBuffer<u32>>>,
    d_flop_zone_nodes: Vec<Option<MetalBuffer<u32>>>,

    // Flop zone level nodes (for top-down reach through flop zone)
    d_flop_level_nodes: Vec<Option<MetalBuffer<u32>>>,

    // Parameters buffer for passing params to kernels
    d_params_buf: std::cell::UnsafeCell<MetalBuffer<u8>>,

    /// #117 Fix C: GPU-resident orchestration mode. When true, stage
    /// functions commit command buffers but skip the per-stage
    /// wait_until_completed; the caller (run()) flushes at end of all
    /// iterations via a final empty-buffer wait. This eliminates ~14
    /// CPU↔GPU sync points per iter (one per stage) × num_iters × np
    /// traversers — at production scale, this is thousands of sync
    /// points per blueprint.
    ///
    /// Profiling (run_profiled) keeps async_mode=false because per-
    /// stage timing requires synchronous waits to attribute GPU time
    /// to stages.
    async_mode: std::cell::Cell<bool>,

    // Layout parameters
    num_players: u8,
    nh: usize,
    nn: usize,
    starting_pot: i32,
    // ─── Slice 2: rake plumbing (CPU↔Metal parity) ───
    // Mirror CPU `tree.rake_rate, tree.rake_cap` as f32 for kernel scalars.
    // Apply gates: `eff_rake = flop_seen ? rake : 0.0` (no-flop-no-drop),
    // then `rake_amount = (pot * eff_rate).min(eff_cap).max(0.0)`.
    // Today flop_seen is always true (no preflop terminals); the gating is
    // wired now so it's correct when preflop integration brings preflop-end
    // terminals — not a later gap.
    rake_rate: f32,
    rake_cap: f32,
    num_combinations: f32,
    regret_floor: f32,
    iteration: u32,
    // ─── Phase 1.A negative-regret pruning (Option A) ───
    // Default off — when enabled, kernel skips per-(action, hand) regret
    // updates where the regret is below pruning_threshold, with Pluribus
    // carve-outs (never river, never terminal-leading, every Kth iter all).
    pruning_enabled: bool,
    pruning_threshold: f32,
    pruning_stride: u32,
    n_turn: usize,
    max_river: usize,
    max_depth: usize,
    river_cc_count: usize,
    turn_cc_count: usize,

    // Strides for per-outcome dimensional layout.
    // INVARIANT: these fields are populated FROM `dims` at construction
    // and must stay equal to it — `dims` (ZoneDims) is the single source
    // of stride/offset math shared with the CPU side (Phase B3). The
    // offset helper delegates to `dims`; the offset_helper test
    // cross-checks field values against an independently-constructed
    // ZoneDims.
    dims: crate::solver::zone_dims::ZoneDims,
    flop_stride: usize,
    turn_stride: usize,
    river_stride: usize,
    turn_total: usize,
    river_total: usize,
    flop_infosets: usize,
    turn_infosets: usize,
    river_infosets: usize,

    // Zone node counts per level
    river_zone_counts: Vec<usize>,
    turn_zone_counts: Vec<usize>,
    flop_zone_counts: Vec<usize>,

    // Offsets into contiguous regrets buffer
    flop_offset: usize,    // 0
    turn_offset: usize,    // flop_stride
    river_offset: usize,   // flop_stride + turn_total

    river_outcomes_per_turn: Vec<usize>,
    turn_deck: Vec<u8>,
    river_decks: Vec<Vec<u8>>,

    // ─── Step 2.B.1: DiskBacked river persistence (foundation) ───
    // Mirror of CPU's RiverPersistenceMode + river_files. See GpuRiverMode
    // docstring at top of this file. In InMemory (default), gpu_river_files
    // is None and load/save_river_pair_gpu are no-ops. In DiskBacked, the
    // files hold the persistent per-pair state and load/save round-trip
    // them into the in-place buffer slots.
    gpu_river_mode: GpuRiverMode,
    gpu_river_files: Option<(std::fs::File, std::fs::File)>,
    /// Tracks which (ti, ri) pair is currently loaded into the river region
    /// of d_regrets/d_cum_strategy. Used by `save_river_pair_gpu` as a
    /// debug-only assertion that the caller is saving the same pair they
    /// loaded.
    gpu_current_river_pair: Option<(usize, usize)>,

    // Zone nodes per level for reach computation (flat lists filtered by zone)
    flop_zone_nodes_flat: Vec<MetalBuffer<u32>>,
    turn_zone_nodes_flat: Vec<MetalBuffer<u32>>,
    river_zone_nodes_flat: Vec<MetalBuffer<u32>>,
}

impl MetalFlopStartSolver {
    pub fn new(
        ctx: &MetalContext,
        tree: &FlatTree,
        game: &FlopStartGame,
        cpu_solver: &FlopStartVectorCfr,
    ) -> Self {
        let table = game.table();
        let np = cpu_solver.num_players();
        let nh = cpu_solver.num_hands();
        let nn = tree.num_nodes();
        let n_turn = cpu_solver.n_turn_outcomes();
        let max_river = cpu_solver.max_river_outcomes();
        let max_depth = tree.max_depth as usize;
        let num_opp = (np - 1) as usize;

        // Layout dimensions — sourced from ZoneDims (the single source
        // both CPU and GPU consume; Phase B3 dims-threading step). At
        // uniform nh this reproduces the pre-bucketing formulas exactly
        // (pinned by zone_dims::tests::uniform_matches_prebucketing_formulas).
        let flop_infosets = cpu_solver.flop_infosets();
        let turn_infosets = cpu_solver.turn_infosets();
        let river_infosets = cpu_solver.river_infosets();
        let dims = crate::solver::zone_dims::ZoneDims::uniform(
            MAX_NA_POSTFLOP, nh, flop_infosets, turn_infosets, river_infosets,
            n_turn, max_river,
        );
        let flop_stride = dims.flop_stride();
        let turn_stride = dims.turn_stride();
        let river_stride = dims.river_stride();
        let turn_total = dims.turn_total();
        let river_total = dims.river_total();
        let flop_offset = dims.flop_offset();
        let turn_offset = dims.turn_offset();
        let river_offset = dims.river_offset();

        // Load pipelines
        let strategies_pipeline = ctx.create_pipeline("vcfr_compute_strategies").expect("strategies");
        let strategies_batched_pipeline = ctx.create_pipeline("vcfr_compute_strategies_batched").expect("strategies_batched");
        let init_reach_pipeline = ctx.create_pipeline("vcfr_init_reach").expect("init_reach");
        let top_down_pipeline = ctx.create_pipeline("vcfr_top_down_reach").expect("top_down");
        let bottom_up_pipeline = ctx.create_pipeline("vcfr_bottom_up").expect("bottom_up");
        let bottom_up_tg_parallel_pipeline = ctx.create_pipeline("vcfr_bottom_up_tg_parallel").expect("bottom_up_tg_parallel");
        let batched_pipeline = ctx.create_pipeline("vcfr_bottom_up_batched").expect("batched");
        let batched_tg_parallel_pipeline = ctx.create_pipeline("vcfr_bottom_up_batched_tg_parallel").expect("batched_tg_parallel");
        let chance_accum_pipeline = ctx.create_pipeline("vcfr_chance_accumulate").expect("chance_accum");
        let chance_final_pipeline = ctx.create_pipeline("vcfr_chance_finalize").expect("chance_final");
        let chance_grouped_pipeline = ctx.create_pipeline("vcfr_chance_accumulate_grouped").expect("chance_grouped");
        let seed_reach_pipeline = ctx.create_pipeline("vcfr_seed_reach").expect("seed_reach");
        let zero_pipeline = ctx.create_pipeline("vcfr_zero_buffer").expect("zero");

        let device = ctx.device();

        // Tree structure
        let d_nodes = MetalBuffer::from_slice(device, &tree.nodes);
        let d_children = MetalBuffer::from_slice(device, &tree.children);
        let d_contributions = MetalBuffer::from_slice(device, &tree.contributions);
        let d_folded_masks = MetalBuffer::from_slice(device, &tree.folded_masks);
        let d_infoset_offsets = MetalBuffer::from_slice(device, &compute_infoset_offsets(tree, cpu_solver));

        // Per-zone decision node IDs
        let (flop_ids, flop_offs, turn_ids, turn_offs, river_ids, river_offs) =
            cpu_solver.per_zone_decision_nodes(tree);
        let d_flop_decision_ids = MetalBuffer::from_slice(device, &flop_ids);
        let d_flop_infoset_offsets = MetalBuffer::from_slice(device, &flop_offs);
        let d_turn_decision_ids = MetalBuffer::from_slice(device, &turn_ids);
        let d_turn_infoset_offsets = MetalBuffer::from_slice(device, &turn_offs);
        let d_river_decision_ids = MetalBuffer::from_slice(device, &river_ids);
        let d_river_infoset_offsets = MetalBuffer::from_slice(device, &river_offs);

        // Solver state.
        //
        // GPU IS INDEPENDENT OF CPU (architecture: CPU as lossless reference,
        // GPU as production, each running its own complete solve, validated
        // by parity comparison — no CPU→GPU data crossing the boundary).
        //
        // The three solver state buffers are zero-initialized. Justification:
        //   - d_regrets: at iter-0 CFR all regrets are zero. After iter-0 the
        //     GPU updates regrets in-place; the init value isn't transported
        //     from CPU.
        //   - d_strategy: derived from regrets. The GPU run loop begins each
        //     iter with compute_all_strategies(ctx) which overwrites every
        //     slot before any read. Disturbance verified 2026-06-05: NaN
        //     init produced iter-0 max_rel=0.00% divergence vs CPU and
        //     identical iter-99 exploitability (0.037445 GPU vs 0.037472 CPU).
        //   - d_cum_strategy: at iter-0 zero, accumulated in-place by GPU
        //     across iters. Init value not transported from CPU.
        let d_regrets = MetalBuffer::zeros(device, flop_stride + turn_total + river_total);
        let d_strategy = MetalBuffer::zeros(device, flop_stride + turn_total + river_total);
        let d_cum_strategy = MetalBuffer::zeros(device, flop_stride + turn_total + river_total);
        let d_reach = MetalBuffer::zeros(device, nn * np as usize * nh);
        let d_turn_reach = MetalBuffer::zeros(device, nn * np as usize * nh);
        let d_river_reach = MetalBuffer::zeros(device, nn * np as usize * nh);
        let d_cfv = MetalBuffer::zeros(device, nn * nh);

        // Game data
        let d_initial_weight = {
            let mut w = Vec::new();
            for p in 0..np as usize {
                w.extend_from_slice(&table.initial_weights[p]);
            }
            MetalBuffer::from_slice(device, &w)
        };
        let d_hand_cards = MetalBuffer::from_slice(device, &table.hand_cards);

        // Step 5 chokepoint instrumentation buffer: [nn × nh] u8 markers,
        // initialized to zero (= unmarked). Kernel writes 1 (rake-applied)
        // or 2 (rake-correctly-skipped per no-flop-no-drop) at every
        // terminal it processes via multiway_brute_force_showdown.
        let d_rake_marker = MetalBuffer::<u8>::zeros(device, nn * nh);

        let (opp_str, opp_idx, pl_str, pl_idx, _) = table.sorted_opp_arrays_base();
        let d_sorted_opp_strength = MetalBuffer::from_slice(device, &opp_str);
        let d_sorted_opp_indices = MetalBuffer::from_slice(device, &opp_idx);
        let d_sorted_pl_strength = MetalBuffer::from_slice(device, &pl_str);
        let d_sorted_pl_indices = MetalBuffer::from_slice(device, &pl_idx);

        let d_river_sorted_str = MetalBuffer::from_slice(device, &table.river_sorted_str);
        let d_river_sorted_idx = MetalBuffer::from_slice(device, &table.river_sorted_idx);
        let (river_pl_str, river_pl_idx) = table.compute_river_pl_sorted();
        let d_river_pl_str = MetalBuffer::from_slice(device, &river_pl_str);
        let d_river_pl_idx = MetalBuffer::from_slice(device, &river_pl_idx);
        let d_turn_sorted_str = MetalBuffer::from_slice(device, &table.turn_sorted_str);
        let d_turn_sorted_idx = MetalBuffer::from_slice(device, &table.turn_sorted_idx);
        let (turn_pl_str, turn_pl_idx) = table.compute_turn_pl_sorted();
        let d_turn_pl_str = MetalBuffer::from_slice(device, &turn_pl_str);
        let d_turn_pl_idx = MetalBuffer::from_slice(device, &turn_pl_idx);

        let river_chance_prob = table.compute_river_chance_prob();
        let turn_chance_prob = table.compute_turn_chance_prob();
        let d_river_chance_prob = MetalBuffer::from_slice(device, &river_chance_prob);
        let river_board_mask = table.compute_river_board_mask();
        let d_river_board_mask = MetalBuffer::from_slice(device, &river_board_mask);
        let d_turn_chance_prob = MetalBuffer::from_slice(device, &turn_chance_prob);

        let d_river_cfv_batch = MetalBuffer::zeros(device, max_river * nn * nh);
        let d_river_accum = MetalBuffer::zeros(device, nn * nh);
        let d_turn_cfv_batch = MetalBuffer::zeros(device, n_turn * nn * nh);
        let d_turn_accum = MetalBuffer::zeros(device, nn * nh);

        let river_cc = cpu_solver.river_chance_children();
        let turn_cc = cpu_solver.turn_chance_children();
        let d_river_chance_children = MetalBuffer::from_slice(device, river_cc);
        let d_debug_out = MetalBuffer::zeros(device, 64);
        let d_turn_chance_children = MetalBuffer::from_slice(device, turn_cc);

        // Zone nodes per level
        let (river_zone, turn_zone, flop_zone) = cpu_solver.zone_nodes_per_level();
        let river_zone_counts: Vec<usize> = river_zone.iter().map(|v| v.len()).collect();
        let turn_zone_counts: Vec<usize> = turn_zone.iter().map(|v| v.len()).collect();
        let flop_zone_counts: Vec<usize> = flop_zone.iter().map(|v| v.len()).collect();

        let d_river_zone_nodes: Vec<Option<MetalBuffer<u32>>> = river_zone.iter()
            .map(|v| if v.is_empty() { None } else { Some(MetalBuffer::from_slice(device, v)) })
            .collect();
        let d_turn_zone_nodes: Vec<Option<MetalBuffer<u32>>> = turn_zone.iter()
            .map(|v| if v.is_empty() { None } else { Some(MetalBuffer::from_slice(device, v)) })
            .collect();
        let d_flop_zone_nodes: Vec<Option<MetalBuffer<u32>>> = flop_zone.iter()
            .map(|v| if v.is_empty() { None } else { Some(MetalBuffer::from_slice(device, v)) })
            .collect();

        // Flat zone node lists per level (for top-down reach within zones)
        let d_flop_level_nodes: Vec<Option<MetalBuffer<u32>>> = (0..=max_depth)
            .map(|level| {
                let nodes: Vec<u32> = tree.nodes_at_level(level as u32).iter()
                    .filter(|&&nid| {
                        let zone = cpu_solver.zones()[nid as usize];
                        matches!(zone, crate::solver::flop_start_vector_cfr::Zone::Flop)
                    })
                    .copied()
                    .collect();
                if nodes.is_empty() { None }
                else { Some(MetalBuffer::from_slice(device, &nodes)) }
            })
            .collect();

        let river_outcomes_per_turn: Vec<usize> = table.remaining_deck.iter()
            .map(|&tc| table.river_decks[tc as usize].len())
            .collect();

        // Parameters buffer (reused across kernel launches)
        let d_params_buf = std::cell::UnsafeCell::new(MetalBuffer::zeros(device, 256));

        Self {
            dims,
            strategies_pipeline,
            strategies_batched_pipeline,
            init_reach_pipeline,
            top_down_pipeline,
            bottom_up_pipeline,
            bottom_up_tg_parallel_pipeline,
            batched_pipeline,
            batched_tg_parallel_pipeline,
            chance_accum_pipeline,
            chance_final_pipeline,
            chance_grouped_pipeline,
            seed_reach_pipeline,
            zero_pipeline,
            d_nodes,
            d_children,
            d_contributions,
            d_folded_masks,
            d_infoset_offsets,
            d_flop_decision_ids,
            d_flop_infoset_offsets,
            d_turn_decision_ids,
            d_turn_infoset_offsets,
            d_river_decision_ids,
            d_river_infoset_offsets,
            d_regrets,
            d_strategy,
            d_cum_strategy,
            d_reach,
            d_turn_reach,
            d_river_reach,
            d_cfv,
            d_rake_marker,
            d_initial_weight,
            d_hand_cards,
            d_sorted_opp_strength,
            d_sorted_opp_indices,
            d_sorted_pl_strength,
            d_sorted_pl_indices,
            d_river_sorted_str,
            d_river_sorted_idx,
            d_river_pl_str,
            d_river_pl_idx,
            d_turn_sorted_str,
            d_turn_sorted_idx,
            d_turn_pl_str,
            d_turn_pl_idx,
            d_river_chance_prob,
            d_river_board_mask,
            d_turn_chance_prob,
            d_river_cfv_batch,
            d_river_accum,
            d_turn_cfv_batch,
            d_turn_accum,
            d_river_chance_children,
            d_debug_out,
            d_turn_chance_children,
            d_river_zone_nodes,
            d_turn_zone_nodes,
            d_flop_zone_nodes,
            d_flop_level_nodes,
            d_params_buf,
            async_mode: std::cell::Cell::new(false),
            num_players: np,
            nh,
            nn,
            starting_pot: tree.starting_pot,
            // Slice 2 rake: stash the f64 tree fields as f32 scalars for kernel use.
            rake_rate: tree.rake_rate as f32,
            rake_cap: tree.rake_cap as f32,
            num_combinations: table.num_combinations as f32,
            regret_floor: -1e30,
            iteration: 0,
            // Phase 1.A pruning: default OFF. Existing tests unchanged.
            pruning_enabled: false,
            pruning_threshold: -1e30,  // never trips when disabled
            pruning_stride: 20,        // Pluribus's "every 5%" → 1/20
            n_turn,
            max_river,
            max_depth,
            river_cc_count: river_cc.len(),
            turn_cc_count: turn_cc.len(),
            flop_stride,
            turn_stride,
            river_stride,
            turn_total,
            river_total,
            flop_infosets,
            turn_infosets,
            river_infosets,
            river_zone_counts,
            turn_zone_counts,
            flop_zone_counts,
            flop_offset,
            turn_offset,
            river_offset,
            river_outcomes_per_turn,
            turn_deck: table.remaining_deck.clone(),
            river_decks: table.river_decks.clone(),
            // These aren't used yet but will be for per-zone reach
            flop_zone_nodes_flat: vec![],
            turn_zone_nodes_flat: vec![],
            river_zone_nodes_flat: vec![],
            // Step 2.B.1: default InMemory; promote via into_disk_backed_gpu().
            gpu_river_mode: GpuRiverMode::InMemory,
            gpu_river_files: None,
            gpu_current_river_pair: None,
        }
    }

    // ─── Step 2.B.1: DiskBacked river persistence (foundation) ───
    //
    // Mirror of CPU's into_disk_backed + load_river_pair + save_river_pair.
    // File format is bit-compatible with CPU's: f32 little-endian, offset
    // `(ti * max_river + ri) * river_stride * 4` bytes per pair, total file
    // size `n_turn * max_river * river_stride * 4` bytes.
    //
    // Step 2.B.1 keeps the GPU river buffers full-sized — load/save round-
    // trip the in-place per-pair slot. Step 2.B.2 will shrink to a single
    // scratch slot and change kernel dispatch offsets accordingly.

    pub fn gpu_river_mode(&self) -> &GpuRiverMode { &self.gpu_river_mode }
    // max_river() defined further below; not duplicating here.
    pub fn river_offset(&self) -> usize { self.river_offset }
    pub fn river_stride(&self) -> usize { self.river_stride }
    /// Public accessors mirroring river_*, added in Phase 2 so the offset-
    /// helper unit tests can hand-compute reference values.
    pub fn turn_offset(&self) -> usize { self.turn_offset }
    pub fn turn_stride(&self) -> usize { self.turn_stride }
    pub fn flop_offset(&self) -> usize { self.flop_offset }

    /// Float-offset into d_regrets / d_strategy / d_cum_strategy for the start
    /// of the requested zone's outcome slice. All three buffers share this
    /// layout — they were allocated together with identical stride math, so a
    /// single helper covers all three.
    ///
    /// This is the consolidation of the pattern that used to be inlined at
    /// every dispatch site as `(self.river_offset + outcome_idx * self.river_stride) * 4`,
    /// etc. The inline form was a desync hazard — Phase 2's per-stage MAX_NA
    /// makes the stride math stage-dependent, and the user's directive
    /// explicitly called out stride-math sites as "historically bug-prone".
    /// Centralizing into this helper + unit testing it per zone is the
    /// disambiguator.
    pub fn infoset_float_offset(&self, zone: BufferZone) -> usize {
        // Delegates to ZoneDims (single source, Phase B3). BufferZone ↔
        // ZoneRef is a 1:1 rename.
        let zr = match zone {
            BufferZone::Flop => crate::solver::zone_dims::ZoneRef::Flop,
            BufferZone::Turn { ti } => crate::solver::zone_dims::ZoneRef::Turn { ti },
            BufferZone::River { outcome_idx } => {
                crate::solver::zone_dims::ZoneRef::River { outcome_idx }
            }
        };
        self.dims.zone_float_offset(zr)
    }

    /// Byte offset (suitable for set_buffer offset args). Equals
    /// `infoset_float_offset(zone) * size_of::<f32>()`.
    pub fn infoset_byte_offset(&self, zone: BufferZone) -> u64 {
        (self.infoset_float_offset(zone) * std::mem::size_of::<f32>()) as u64
    }

    /// Returns the buffer outcome index for river kernel dispatches.
    ///
    /// In InMemory mode the river region of d_regrets / d_strategy /
    /// d_cum_strategy holds all (ti, ri) pairs at
    /// `river_offset + outcome_idx * river_stride` — return
    /// `ti * max_river + ri`.
    ///
    /// In DiskBacked mode the river region acts as a single scratch slot
    /// holding only the currently-loaded (ti, ri) pair, so every river
    /// kernel must address slot 0 — return 0. The caller is responsible
    /// for invoking `load_river_pair_gpu(ti, ri)` before any kernel reads
    /// the scratch and `save_river_pair_gpu(ti, ri)` after any kernel
    /// writes it.
    ///
    /// CONSOLIDATION PRINCIPLE: this is the single source of truth for
    /// the river-buffer outcome index across modes. Every site that
    /// previously computed `ti * self.max_river + ri` for river-buffer
    /// addressing now calls this method instead. An offset bug now
    /// surfaces at EVERY dispatch site simultaneously rather than
    /// quietly at one — loud-everywhere instead of quiet-at-one-site.
    ///
    /// Note: this returns the kernel/buffer outcome index, NOT the file
    /// offset. File offsets in load/save_river_pair_gpu always use
    /// `ti * max_river + ri` because the file holds all pairs.
    #[inline]
    pub fn river_outcome_idx(&self, ti: usize, ri: usize) -> usize {
        match self.gpu_river_mode {
            GpuRiverMode::InMemory => ti * self.max_river + ri,
            GpuRiverMode::DiskBacked { .. } => 0,
        }
    }

    /// Convert this solver from `InMemory` to `DiskBacked`. Creates the
    /// two persistence files (sized to hold the full river state), seeds
    /// them with the current in-buffer river state, then SHRINKS the
    /// d_regrets / d_strategy / d_cum_strategy GPU buffers so their river
    /// region holds only `river_stride` (a single scratch slot for the
    /// currently-loaded pair). This is the load-bearing optimization for
    /// production scale: at HU OptB nh=1176 the river region drops from
    /// 175 GB to 76 MB per buffer.
    ///
    /// SHRINK SAFETY (per the audit-arc discipline): the buffer shrink
    /// is potentially-bug-exposing — any code path that secretly
    /// addresses river-region slots beyond slot 0 in DiskBacked will now
    /// access OOB. The discriminator is the 2.B.2 bit-exact parity gate
    /// (`p1_5_4_step2b2_disk_backed_gpu_bit_exact_parity`), which is
    /// re-run after this slice. If a kernel was reading slot 1+ in
    /// DiskBacked, the OOB now produces nondeterministic f32 → parity
    /// mismatch → test fails loudly. The shrink is not assumed safe; it
    /// is VALIDATED by the gate.
    pub fn into_disk_backed_gpu<P: AsRef<std::path::Path>>(
        &mut self,
        ctx: &MetalContext,
        regrets_path: P,
        cum_strategy_path: P,
    ) -> std::io::Result<()> {
        use std::fs::OpenOptions;
        use std::io::{Seek, SeekFrom, Write};

        assert!(
            matches!(self.gpu_river_mode, GpuRiverMode::InMemory),
            "into_disk_backed_gpu called on a solver already in DiskBacked mode"
        );

        let regrets_path = regrets_path.as_ref().to_path_buf();
        let cum_strategy_path = cum_strategy_path.as_ref().to_path_buf();

        // Full-size file (holds all (ti, ri) pairs).
        let total_bytes = (self.n_turn * self.max_river * self.river_stride
            * std::mem::size_of::<f32>()) as u64;

        let mut regrets_file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(&regrets_path)?;
        let mut cum_strategy_file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(&cum_strategy_path)?;

        regrets_file.set_len(total_bytes)?;
        cum_strategy_file.set_len(total_bytes)?;

        // Seed both files with the current FULL in-buffer river state.
        // This preserves any per-pair state from prior iters and seeds the
        // (zero) starting state at iter 0. Done BEFORE the shrink (we
        // still have access to all per-pair data here).
        regrets_file.seek(SeekFrom::Start(0))?;
        let river_byte_off = self.river_offset * std::mem::size_of::<f32>();
        let river_byte_len = self.n_turn * self.max_river * self.river_stride
            * std::mem::size_of::<f32>();
        let regrets_bytes = unsafe {
            let p = self.d_regrets.as_slice().as_ptr() as *const u8;
            std::slice::from_raw_parts(p.add(river_byte_off), river_byte_len)
        };
        regrets_file.write_all(regrets_bytes)?;
        regrets_file.flush()?;

        cum_strategy_file.seek(SeekFrom::Start(0))?;
        let cum_bytes = unsafe {
            let p = self.d_cum_strategy.as_slice().as_ptr() as *const u8;
            std::slice::from_raw_parts(p.add(river_byte_off), river_byte_len)
        };
        cum_strategy_file.write_all(cum_bytes)?;
        cum_strategy_file.flush()?;

        // ──── SHRINK: reallocate d_regrets, d_strategy, d_cum_strategy
        //              at (flop_stride + turn_total + river_stride). The
        //              river region beyond slot 0 is no longer addressable;
        //              any kernel that secretly accesses it will now go OOB.
        //
        // The mode field must be flipped BEFORE the shrink so that
        // `river_outcome_idx` returns 0 from this point on. The kernels
        // we just shrunk for already use the helper, so they're correct.
        // Subsequent calls follow the DiskBacked code path.
        self.gpu_river_mode = GpuRiverMode::DiskBacked {
            regrets_path,
            cum_strategy_path,
        };
        self.gpu_river_files = Some((regrets_file, cum_strategy_file));
        self.gpu_current_river_pair = None;

        let device = ctx.device();
        let new_size = self.flop_stride + self.turn_total + self.river_stride;

        // Allocate new buffers; copy flop+turn region from old to new.
        // Slot 0 of the new river region stays zeros — first
        // `load_river_pair_gpu` will populate it from the file.
        let mut new_regrets = MetalBuffer::<f32>::zeros(device, new_size);
        let mut new_strategy = MetalBuffer::<f32>::zeros(device, new_size);
        let mut new_cum_strategy = MetalBuffer::<f32>::zeros(device, new_size);
        new_regrets.as_mut_slice()[..self.river_offset]
            .copy_from_slice(&self.d_regrets.as_slice()[..self.river_offset]);
        new_strategy.as_mut_slice()[..self.river_offset]
            .copy_from_slice(&self.d_strategy.as_slice()[..self.river_offset]);
        new_cum_strategy.as_mut_slice()[..self.river_offset]
            .copy_from_slice(&self.d_cum_strategy.as_slice()[..self.river_offset]);

        // Replace. Old buffers drop here.
        self.d_regrets = new_regrets;
        self.d_strategy = new_strategy;
        self.d_cum_strategy = new_cum_strategy;

        Ok(())
    }

    /// Load the (ti, ri) slice of regrets and cum_strategy from disk
    /// into the in-buffer river slot. No-op in InMemory.
    ///
    /// Buffer slot uses `river_outcome_idx(ti, ri)` (0 in DiskBacked —
    /// the scratch). File offset uses `ti * max_river + ri` regardless
    /// of mode because the file holds all pairs at canonical offsets.
    pub fn load_river_pair_gpu(&mut self, ti: usize, ri: usize) -> std::io::Result<()> {
        if !matches!(self.gpu_river_mode, GpuRiverMode::DiskBacked { .. }) {
            return Ok(());
        }
        use std::io::{Read, Seek, SeekFrom};
        let stride = self.river_stride;
        let buf_off = self.river_offset + self.river_outcome_idx(ti, ri) * stride;
        let file_off = ((ti * self.max_river + ri) * stride
            * std::mem::size_of::<f32>()) as u64;
        let byte_len = stride * std::mem::size_of::<f32>();

        let (rf, cf) = self.gpu_river_files.as_mut()
            .expect("DiskBacked mode requires gpu_river_files");

        rf.seek(SeekFrom::Start(file_off))?;
        let regrets_bytes = unsafe {
            let p = self.d_regrets.as_mut_slice().as_mut_ptr() as *mut u8;
            std::slice::from_raw_parts_mut(p.add(buf_off * std::mem::size_of::<f32>()), byte_len)
        };
        rf.read_exact(regrets_bytes)?;

        cf.seek(SeekFrom::Start(file_off))?;
        let cum_bytes = unsafe {
            let p = self.d_cum_strategy.as_mut_slice().as_mut_ptr() as *mut u8;
            std::slice::from_raw_parts_mut(p.add(buf_off * std::mem::size_of::<f32>()), byte_len)
        };
        cf.read_exact(cum_bytes)?;

        self.gpu_current_river_pair = Some((ti, ri));
        Ok(())
    }

    /// Save the (ti, ri) slice of regrets and cum_strategy from the
    /// in-buffer river slot to disk. No-op in InMemory.
    ///
    /// Buffer slot uses `river_outcome_idx(ti, ri)`; file offset uses
    /// `ti * max_river + ri`. See `river_outcome_idx` for the consolidation
    /// rationale.
    pub fn save_river_pair_gpu(&mut self, ti: usize, ri: usize) -> std::io::Result<()> {
        if !matches!(self.gpu_river_mode, GpuRiverMode::DiskBacked { .. }) {
            return Ok(());
        }
        use std::io::{Seek, SeekFrom, Write};
        debug_assert_eq!(self.gpu_current_river_pair, Some((ti, ri)),
            "save_river_pair_gpu({}, {}) called but current loaded pair is {:?}",
            ti, ri, self.gpu_current_river_pair);

        let stride = self.river_stride;
        let buf_off = self.river_offset + self.river_outcome_idx(ti, ri) * stride;
        let file_off = ((ti * self.max_river + ri) * stride
            * std::mem::size_of::<f32>()) as u64;
        let byte_len = stride * std::mem::size_of::<f32>();

        let (rf, cf) = self.gpu_river_files.as_mut()
            .expect("DiskBacked mode requires gpu_river_files");

        rf.seek(SeekFrom::Start(file_off))?;
        let regrets_bytes = unsafe {
            let p = self.d_regrets.as_slice().as_ptr() as *const u8;
            std::slice::from_raw_parts(p.add(buf_off * std::mem::size_of::<f32>()), byte_len)
        };
        rf.write_all(regrets_bytes)?;

        cf.seek(SeekFrom::Start(file_off))?;
        let cum_bytes = unsafe {
            let p = self.d_cum_strategy.as_slice().as_ptr() as *const u8;
            std::slice::from_raw_parts(p.add(buf_off * std::mem::size_of::<f32>()), byte_len)
        };
        cf.write_all(cum_bytes)?;

        Ok(())
    }

    /// Debug: run one traverser pass on GPU and return intermediate CFVs.
    /// Computes strategies, reach, and CFVs for the given traverser.
    /// Returns: (river_cfvs_per_outcome, river_reach, turn_reach, flop_reach, flop_cfv)
    /// Does NOT update iteration counter or regrets (saves/restores).
    pub fn debug_traverser_cfvs(
        &mut self,
        ctx: &MetalContext,
        tree: &FlatTree,
        game: &FlopStartGame,
        traverser: usize,
    ) -> (Vec<(usize, usize, Vec<f32>)>, Vec<f32>, Vec<f32>) {
        let np = self.num_players as usize;
        let nh = self.nh;
        let nn = self.nn;

        // Save regrets & cum before debug pass
        let saved_reg = self.download_regrets();
        let saved_cum = self.download_cum_strategy();

        let params = DcfrParams::new(self.iteration);

        self.compute_all_strategies(ctx);
        self.compute_reach_flop(ctx);
        let flop_reach = self.download_reach();

        self.zero_buffer_name(ctx, 100); // d_cfv
        self.zero_buffer_name(ctx, 2);   // turn_cfv_batch

        let mut river_cfvs = Vec::new();

        for ti in 0..self.n_turn {
            let n_river = self.river_outcomes_per_turn[ti];
            self.zero_buffer_name(ctx, 0); // river_cfv_batch
            self.zero_buffer_name(ctx, 1); // river_accum

            self.compute_reach_turn(ctx, ti);
            let _turn_reach = self.download_turn_reach();

            for ri in 0..n_river {
                self.compute_reach_river(ctx, ti, ri);
                self.bottom_up_river(ctx, ti, ri, traverser as u32, &params);

                // Download river CFV for this outcome
                let rcfv = self.download_river_cfv_batch();
                // The CFV for this ri is at offset ri * nn * nh
                let start = ri * nn * nh;
                let end = start + nn * nh;
                river_cfvs.push((ti, ri, rcfv[start..end].to_vec()));
            }

            self.chance_accumulate_river(ctx, ti, n_river);
            self.chance_finalize_river(ctx, ti);
            self.bottom_up_turn(ctx, ti, traverser as u32, &params);
        }

        self.chance_accumulate_turn(ctx);
        self.chance_finalize_turn(ctx);
        self.bottom_up_flop(ctx, traverser as u32, &params);

        let flop_cfv = self.download_cfv();

        // Restore regrets & cum
        self.upload_regrets(&saved_reg);
        self.d_cum_strategy.as_mut_slice().copy_from_slice(&saved_cum);

        (river_cfvs, flop_reach, flop_cfv)
    }

    /// Run a single traverser pass (strategies → reach → bottom-up → regret update).
    /// Increments iteration counter on first traverser. Matches one step of the inner loop of run().
    pub fn run_single_traverser(
        &mut self,
        ctx: &MetalContext,
        traverser: usize,
        increment_iter: bool,
    ) {
        if increment_iter {
            let _params = DcfrParams::new(self.iteration);
            self.iteration += 1;
        }
        let params = DcfrParams::new(self.iteration - 1); // use the params for this iteration

        self.compute_all_strategies(ctx);
        self.compute_reach_flop(ctx);
        self.zero_buffer_name(ctx, 100);
        self.zero_buffer_name(ctx, 2);

        for ti in 0..self.n_turn {
            let n_river = self.river_outcomes_per_turn[ti];
            self.zero_buffer_name(ctx, 0);
            self.zero_buffer_name(ctx, 1);
            self.compute_reach_turn(ctx, ti);
            for ri in 0..n_river {
                self.compute_reach_river(ctx, ti, ri);
                self.bottom_up_river(ctx, ti, ri, traverser as u32, &params);
            }
            self.chance_accumulate_river(ctx, ti, n_river);
            self.chance_finalize_river(ctx, ti);
            self.bottom_up_turn(ctx, ti, traverser as u32, &params);
        }
        self.chance_accumulate_turn(ctx);
        self.chance_finalize_turn(ctx);
        self.bottom_up_flop(ctx, traverser as u32, &params);
    }

    /// Run N iterations of the flop-start VCFR solver on Metal.
    /// Mirrors the CPU FlopStartVectorCfr::run() exactly.
    /// Conditional wait. In async_mode (Fix C orchestration), skip the
    /// per-stage GPU sync — the caller flushes at end. In sync mode
    /// (default / run_profiled / debug paths), wait as before.
    #[inline]
    fn maybe_wait(&self, buf: &metal::CommandBufferRef) {
        if !self.async_mode.get() {
            buf.wait_until_completed();
        }
    }

    /// Flush queued GPU work by committing an empty command buffer and
    /// waiting on it. Metal serializes command buffers within a queue,
    /// so waiting on a LATER buffer guarantees all prior buffers have
    /// completed. Used to terminate async-mode runs.
    fn flush(&self, ctx: &MetalContext) {
        let final_buf = ctx.new_command_buffer();
        final_buf.commit();
        final_buf.wait_until_completed();
    }

    pub fn run(
        &mut self,
        ctx: &MetalContext,
        tree: &FlatTree,
        game: &FlopStartGame,
        num_iterations: u32,
    ) {
        // #117 Fix C: GPU-resident orchestration. Stage functions commit
        // command buffers but skip per-stage waits; final flush at end
        // amortizes ~14 sync points per iter × np × num_iterations into
        // ONE sync at the end.
        self.async_mode.set(true);
        for _ in 0..num_iterations {
            self.run_one_iter(ctx, tree, game, None);
        }
        self.async_mode.set(false);
        self.flush(ctx);
    }

    /// Profiled variant of `run()`. Returns a `StageProfile` accumulating
    /// per-stage wall-clock across all `num_iterations`. Use this in
    /// performance tests instead of duplicating the run loop with bespoke
    /// `Instant::now()` bracketing — the duplicated loops drift whenever
    /// the production `run()` is refactored. (Phase 0.C / Slice 7a tests
    /// previously did exactly that; 2.C formalizes the timing surface.)
    ///
    /// Overhead: per-stage `Instant::now()` calls add ~30 ns each on Apple
    /// Silicon. Negligible for diagnostic runs; not for hot-path production
    /// (use plain `run()` there).
    pub fn run_profiled(
        &mut self,
        ctx: &MetalContext,
        tree: &FlatTree,
        game: &FlopStartGame,
        num_iterations: u32,
    ) -> StageProfile {
        let mut profile = StageProfile::default();
        let t_total = Instant::now();
        for _ in 0..num_iterations {
            self.run_one_iter(ctx, tree, game, Some(&mut profile));
        }
        profile.total = t_total.elapsed();
        profile
    }

    /// Body of a single CFR iter — shared between `run()` (profile = None,
    /// zero-cost) and `run_profiled()` (profile = Some, per-stage timing
    /// via the `time_stage!` macro).
    ///
    /// The structure mirrors the original `run()` loop body verbatim;
    /// only the timing wrappers are new. Adding a stage means adding a
    /// field to `StageProfile` and wrapping the call site in
    /// `time_stage!(profile, new_field, { ... });`.
    fn run_one_iter(
        &mut self,
        ctx: &MetalContext,
        _tree: &FlatTree,
        _game: &FlopStartGame,
        mut profile: Option<&mut StageProfile>,
    ) {
        let np = self.num_players as usize;
        let params = DcfrParams::new(self.iteration);
        self.iteration += 1;

        for traverser in 0..np {
            // Sequential: recompute strategies before each traverser
            time_stage!(profile, compute_strategies, {
                self.compute_all_strategies(ctx);
            });

            // Flop zone reach
            time_stage!(profile, compute_reach_flop, {
                self.compute_reach_flop(ctx);
            });

            // Zero main CFV (flop_cfv in CPU) and turn CFV batch for this traverser
            time_stage!(profile, zero_buffer_total, {
                self.zero_buffer_name(ctx, 100);
                self.zero_buffer_name(ctx, 2);
            });

            // Per turn card: river → turn pipeline
            //
            // 2.B.2: the per-(ti, ri) inner block branches on
            // `gpu_river_mode`. InMemory uses the original flow (strategy
            // already populated by `compute_all_strategies`'s batched call).
            // DiskBacked brackets each pair with load/save and computes
            // the per-pair strategy from the just-loaded regrets via
            // `compute_river_strategy_pair`. The kernel-dispatch offsets
            // are routed through `river_outcome_idx`, so the kernels are
            // identical across modes; only the buffer slot differs (full
            // per-pair vs scratch).
            let is_disk_backed = matches!(self.gpu_river_mode, GpuRiverMode::DiskBacked { .. });
            for ti in 0..self.n_turn {
                let n_river = self.river_outcomes_per_turn[ti];

                // Zero river CFV batch and accum
                time_stage!(profile, zero_buffer_total, {
                    self.zero_buffer_name(ctx, 0);
                    self.zero_buffer_name(ctx, 1);
                });

                // Compute turn reach for this tc
                time_stage!(profile, compute_reach_turn, {
                    self.compute_reach_turn(ctx, ti);
                });

                // River zone: per river card
                for ri in 0..n_river {
                    if is_disk_backed {
                        // DiskBacked: cycle the scratch through file I/O.
                        time_stage!(profile, load_river_pair, {
                            self.load_river_pair_gpu(ti, ri)
                                .expect("load_river_pair_gpu in run_one_iter");
                        });
                        // Per-pair river strategy from freshly-loaded regrets.
                        time_stage!(profile, compute_river_strategy_pair, {
                            self.compute_river_strategy_pair(ctx, ti, ri);
                        });
                    }

                    // Compute river reach for this (tc, rc)
                    time_stage!(profile, compute_reach_river, {
                        self.compute_reach_river(ctx, ti, ri);
                    });

                    // Bottom-up river zone (single outcome: ri)
                    time_stage!(profile, bottom_up_river, {
                        self.bottom_up_river(ctx, ti, ri, traverser as u32, &params);
                    });

                    if is_disk_backed {
                        // Persist the per-pair mutations (regrets + cum) back
                        // to file before the next pair overwrites the scratch.
                        time_stage!(profile, save_river_pair, {
                            self.save_river_pair_gpu(ti, ri)
                                .expect("save_river_pair_gpu in run_one_iter");
                        });
                    }
                }

                // Chance accumulate: weight river CFVs by chance probability → river_accum
                time_stage!(profile, chance_accumulate_river, {
                    self.chance_accumulate_river(ctx, ti, n_river);
                });

                // Chance finalize: copy river_accum into turn CFV batch at river chance children
                time_stage!(profile, chance_finalize_river, {
                    self.chance_finalize_river(ctx, ti);
                });

                // Bottom-up turn zone for this turn card
                time_stage!(profile, bottom_up_turn, {
                    self.bottom_up_turn(ctx, ti, traverser as u32, &params);
                });
            }

            // Chance accumulate turn: weight turn CFV batch by turn chance probability
            time_stage!(profile, chance_accumulate_turn, {
                self.chance_accumulate_turn(ctx);
            });

            // Chance finalize turn: sum into main CFV at turn chance children
            time_stage!(profile, chance_finalize_turn, {
                self.chance_finalize_turn(ctx);
            });

            // Bottom-up flop zone
            time_stage!(profile, bottom_up_flop, {
                self.bottom_up_flop(ctx, traverser as u32, &params);
            });
        }
    }

    // ─── Strategy computation ───

    fn compute_strategies_single(
        &self, ctx: &MetalContext,
        decision_ids: &MetalBuffer<u32>, infoset_offsets: &MetalBuffer<u32>,
        num_infosets: usize, base_offset: usize, _outcome_stride: usize,
    ) {
        if num_infosets == 0 { return; }
        let nh = self.nh;

        #[repr(C)]
        #[repr(C)]

        #[derive(Clone, Copy)]
        struct Params { num_infosets: i32, nh: i32, base_offset: i32 }
        let p = Params { num_infosets: num_infosets as i32, nh: nh as i32, base_offset: base_offset as i32 };

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.strategies_pipeline);
        enc.set_buffer(0, Some(self.d_regrets.as_ref()), 0);
        enc.set_buffer(1, Some(self.d_strategy.as_ref()), 0);
        enc.set_buffer(2, Some(decision_ids.as_ref()), 0);
        enc.set_buffer(3, Some(self.d_nodes.as_ref()), 0);
        enc.set_buffer(4, Some(infoset_offsets.as_ref()), 0);
        // #117 Fix C race-fix: inline params per-encoder to avoid shared
        // d_params_buf race in async mode (CPU writes faster than GPU reads).
        enc.set_bytes(5, std::mem::size_of::<Params>() as u64,
            &p as *const Params as *const std::ffi::c_void);

        let max_tpg = self.strategies_pipeline.max_total_threads_per_threadgroup() as usize;
        let (grid, tg) = ctx.dispatch_2d(num_infosets, nh, max_tpg);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        self.maybe_wait(cmd);
    }

    fn compute_strategies_batched(
        &self, ctx: &MetalContext,
        decision_ids: &MetalBuffer<u32>, infoset_offsets: &MetalBuffer<u32>,
        num_infosets: usize, num_outcomes: usize,
        base_offset: usize, outcome_stride: usize,
    ) {
        if num_infosets == 0 || num_outcomes == 0 { return; }
        let nh = self.nh;

        #[repr(C)]
        #[repr(C)]

        #[derive(Clone, Copy)]
        struct Params { num_outcomes: i32, num_infosets: i32, nh: i32, outcome_stride: i32, base_offset: i32 }
        let p = Params {
            num_outcomes: num_outcomes as i32,
            num_infosets: num_infosets as i32,
            nh: nh as i32,
            outcome_stride: outcome_stride as i32,
            base_offset: base_offset as i32,
        };
        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.strategies_batched_pipeline);
        enc.set_buffer(0, Some(self.d_regrets.as_ref()), 0);
        enc.set_buffer(1, Some(self.d_strategy.as_ref()), 0);
        enc.set_buffer(2, Some(decision_ids.as_ref()), 0);
        enc.set_buffer(3, Some(self.d_nodes.as_ref()), 0);
        enc.set_buffer(4, Some(infoset_offsets.as_ref()), 0);
        enc.set_bytes(5, std::mem::size_of::<Params>() as u64,
            &p as *const Params as *const std::ffi::c_void);

        let max_tpg = self.strategies_batched_pipeline.max_total_threads_per_threadgroup() as usize;
        // Map 3D problem to 2D: x = outcome * num_infosets + infoset, y = hand
        let total_x = num_outcomes * num_infosets;
        let (grid, tg) = ctx.dispatch_2d(total_x, nh, max_tpg);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        self.maybe_wait(cmd);
    }

    // ─── Reach computation ───

    pub fn compute_reach_flop(&mut self, ctx: &MetalContext) {
        let np = self.num_players as usize;
        let nh = self.nh;
        let nn = self.nn;

        // Zero reach buffer
        self.zero_buffer_name(ctx, 99);

        // Init reach at root from initial weights
        {
            let np_nh = np * nh;
            #[repr(C)]
            #[repr(C)]

            #[derive(Clone, Copy)]
            struct Params { total_reach: i32, np_nh: i32 }
            let p = Params { total_reach: (nn * np_nh) as i32, np_nh: np_nh as i32 };

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.init_reach_pipeline);
            enc.set_buffer(0, Some(self.d_reach.as_ref()), 0);
            enc.set_buffer(1, Some(self.d_initial_weight.as_ref()), 0);
            enc.set_bytes(2, std::mem::size_of::<Params>() as u64,
                &p as *const Params as *const std::ffi::c_void);

            let (grid, tg) = ctx.dispatch_1d(nn * np_nh, 256);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            self.maybe_wait(cmd);
        }

        // Top-down through flop zone using flop strategy
        // We use the main level_nodes but only process flop zone / chance nodes
        // The existing top_down kernel processes all node types (player + chance)
        for level in 0..=self.max_depth {
            let ln = self.d_flop_level_nodes[level].as_ref();
            if ln.is_none() { continue; }
            let ln = ln.unwrap();
            let count = ln.len();
            if count == 0 { continue; }

            #[repr(C)]
            #[repr(C)]

            #[derive(Clone, Copy)]
            struct Params { level_count: i32, num_players: i32, nh: i32, strategy_base: i32 }
            let p = Params { level_count: count as i32, num_players: np as i32, nh: nh as i32, strategy_base: self.flop_offset as i32 };

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.top_down_pipeline);
            enc.set_buffer(0, Some(ln.as_ref()), 0);
            enc.set_bytes(1, std::mem::size_of::<Params>() as u64,
                &p as *const Params as *const std::ffi::c_void);
            enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_children.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_strategy.as_ref()), 0);
            enc.set_buffer(5, Some(self.d_infoset_offsets.as_ref()), 0);
            enc.set_buffer(6, Some(self.d_reach.as_ref()), 0);

            let max_tpg = self.top_down_pipeline.max_total_threads_per_threadgroup() as usize;
            let (grid, tg) = ctx.dispatch_2d(count, nh, max_tpg);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            self.maybe_wait(cmd);
        }
    }

    pub fn compute_reach_turn(&self, ctx: &MetalContext, ti: usize) {
        let nh = self.nh;
        let np = self.num_players as usize;
        let nn = self.nn;
        let np_nh = np * nh;

        // Zero turn reach buffer
        self.zero_buffer_name(ctx, 6); // 6 = d_turn_reach

        // Seed at turn chance children from flop reach
        let cc_count = self.turn_cc_count;
        self.launch_seed_reach(ctx, &self.d_turn_reach, &self.d_reach,
                               &self.d_turn_chance_children, cc_count, np_nh);

        // Top-down through turn zone using turn[ti] strategy
        // Strategy byte offset: turn_offset + ti * turn_stride, each element is f32 (4 bytes)
        let strat_byte_offset = self.infoset_byte_offset(BufferZone::Turn { ti });
        for level in 0..=self.max_depth {
            let count = self.turn_zone_counts[level];
            if count == 0 { continue; }
            let ln = self.d_turn_zone_nodes[level].as_ref().unwrap();
            self.launch_top_down_zone(ctx, ln, count, &self.d_turn_reach,
                                       strat_byte_offset);
        }
    }

    pub fn compute_reach_river(&self, ctx: &MetalContext, ti: usize, ri: usize) {
        let np = self.num_players as usize;
        let nh = self.nh;
        let np_nh = np * nh;

        // Zero river reach buffer
        self.zero_buffer_name(ctx, 7); // 7 = d_river_reach

        // Seed at river chance children from turn reach
        let cc_count = self.river_cc_count;
        self.launch_seed_reach(ctx, &self.d_river_reach, &self.d_turn_reach,
                               &self.d_river_chance_children, cc_count, np_nh);

        // Top-down through river zone using strategy at the kernel-buffer
        // outcome slot. In DiskBacked, `river_outcome_idx` returns 0 (the
        // scratch); the caller must have loaded (ti, ri) into the scratch
        // first via `load_river_pair_gpu`.
        let outcome_idx = self.river_outcome_idx(ti, ri);
        let strat_byte_offset = self.infoset_byte_offset(BufferZone::River { outcome_idx });
        for level in 0..=self.max_depth {
            let count = self.river_zone_counts[level];
            if count == 0 { continue; }
            let ln = self.d_river_zone_nodes[level].as_ref().unwrap();
            self.launch_top_down_zone(ctx, ln, count, &self.d_river_reach,
                                       strat_byte_offset);
        }
    }

    // ─── Bottom-up zone processing ───

    pub fn bottom_up_river(
        &self, ctx: &MetalContext,
        ti: usize, ri: usize, traverser: u32, params: &DcfrParams,
    ) {
        let nh = self.nh;
        let np = self.num_players as usize;
        let nn = self.nn;
        let num_opp = np - 1;
        let cfv_batch_stride = nn * nh;
        let sorted_opp_stride = num_opp * nh;

        let outcome_idx = self.river_outcome_idx(ti, ri);
        // Consolidated zone-offset helper (Phase 2). All three buffers
        // (d_regrets / d_strategy / d_cum_strategy) share the same layout,
        // so one call sets the byte offset for all three.
        let buf_off = self.infoset_byte_offset(BufferZone::River { outcome_idx }) as usize;
        let strat_byte_off = buf_off;
        let regret_byte_off = buf_off;
        let cum_byte_off = buf_off;
        let cfv_byte_off = ri * nn * nh * 4;
        let tc_card = self.turn_deck[ti] as usize;
        let rc_card = self.river_decks[tc_card][ri] as usize;
        let sos_byte_off = ((tc_card * 52 + rc_card) * num_opp * nh) * 2;
        let sps_byte_off = ((tc_card * 52 + rc_card) * num_opp * nh) * 2;
        let prob_byte_off = ((ti * self.max_river + ri) * nh) * 4;

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct BParams {
            level_count: i32, num_outcomes: i32, cfv_batch_stride: i32,
            sorted_opp_stride: i32, num_players: i32, nh: i32,
            traverser: u32, alpha_t: f32, beta_t: f32, gamma_t: f32,
            regret_floor: f32, starting_pot: i32, num_combinations: f32,
            regret_outcome_stride: i32, cum_outcome_stride: i32,
            pruning_enabled: i32, pruning_threshold: f32,
            iteration: i32, pruning_stride: i32, board_state: i32,
            rake_rate: f32, rake_cap: f32,
        }

        // #117 FIX B (per-level wait elimination): encode ALL levels'
        // dispatches into a SINGLE command buffer with a SINGLE wait at the
        // end. Metal's command-buffer ordering preserves intra-buffer
        // sequencing: dispatch N+1 sees the results of dispatch N within
        // the same buffer (compute encoder dependency tracking handles this).
        // Pre-fix: 1 cmd buffer + 1 wait per level (~10 levels per pair ×
        // 2162 pairs = ~20k CPU↔GPU round trips per iter at production).
        // Post-fix: 1 cmd buffer + 1 wait per pair.
        //
        // PARAMS NOTE: per-level params previously uploaded via
        // self.upload_params(&bp) which writes to a shared params buffer.
        // Since multiple levels' kernels read this buffer at different
        // GPU-time points within the same cmd buffer, we'd race on the
        // shared params slot. Fix by uploading each level's params to a
        // per-level offset in a larger params staging buffer... but that
        // requires a buffer refactor. SIMPLER FIRST CUT: only the
        // level_count varies per level; everything else is constant per
        // bottom_up_river call. If we can encode level_count as a
        // function-constant or via a per-level small uniform, the shared
        // params buffer holds the constant fields. For now, since
        // level_count is the only varying field and the kernel uses it
        // for the work-divisor (outcome = idx / level_count), we need
        // per-level. Push this through via PER-LEVEL params buffer
        // allocation; tiny cost, eliminates the race.
        //
        // SIMPLEST CORRECT IMPL: collect all dispatches into one cmd
        // buffer using SEPARATE compute encoders, each with its own
        // upload_params call. Each encoder's params upload happens at
        // encode-time (host side, sequential). The GPU then reads the
        // current params-buffer contents at dispatch time. PROBLEM:
        // each encoder may execute on GPU after later encoders have
        // overwritten the params buffer. RACE.
        //
        // CORRECT FIX: upload params for ALL levels FIRST (to per-level
        // slots in a staging buffer), then encode dispatches that read
        // from the per-level offsets. For this first cut, fall back to
        // per-level command buffer commit (no wait) — Metal queues them
        // and GPU runs them in order, but CPU doesn't block. Wait only
        // at the end (single waitUntilCompleted on the LAST cmd buffer).
        let mut last_cmd: Option<&metal::CommandBufferRef> = None;

        for level in (0..=self.max_depth).rev() {
            let count = self.river_zone_counts[level];
            if count == 0 { continue; }
            let ln = self.d_river_zone_nodes[level].as_ref().unwrap();

            let bp = BParams {
                level_count: count as i32,
                num_outcomes: 1,
                cfv_batch_stride: cfv_batch_stride as i32,
                sorted_opp_stride: sorted_opp_stride as i32,
                num_players: np as i32,
                nh: nh as i32,
                traverser,
                alpha_t: params.alpha_t,
                beta_t: params.beta_t,
                gamma_t: params.gamma_t,
                regret_floor: self.regret_floor,
                starting_pot: self.starting_pot,
                num_combinations: self.num_combinations,
                regret_outcome_stride: self.river_stride as i32,
                cum_outcome_stride: self.river_stride as i32,
                pruning_enabled: if self.pruning_enabled { 1 } else { 0 },
                pruning_threshold: self.pruning_threshold,
                iteration: self.iteration as i32,
                pruning_stride: self.pruning_stride as i32,
                board_state: 2,
                rake_rate: self.rake_rate,
                rake_cap: self.rake_cap,
            };
            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            // Step 2.D.28: branch on num_opp. K>=2 (3p, 4p, 5p, 6p) uses the
            // tg-parallel kernel (1 threadgroup per node, TG_SIZE threads
            // cooperating). HU (num_opp==1) keeps the serial kernel because
            // its per-node work is too small (~nh ops, sweep + output) to
            // benefit from within-node parallelism — 32-nodes-per-group
            // parallelism on the OLD kernel wins by ~32× at HU scale.
            //
            // K=2 routes through the factored share helper (k2_tg), which
            // produces the SAME mathematical result as the CPU's K=2 brute
            // force but with different float ordering. Bit-exact CPU↔Metal
            // parity is intentionally relaxed for K=2; rules-oracle gate
            // (zero-sum, card-conflict, walker-agreement) is the
            // disambiguator. The K=2 brute-force rake parity gates
            // (site_a/b/c, --ignored by default) document this gap.
            //
            // Env override SOLVER_DISABLE_TG_PARALLEL=1 forces serial for
            // A/B measurement; remove once production speedup is validated.
            let use_tg_parallel = num_opp >= 2
                && std::env::var_os("SOLVER_DISABLE_TG_PARALLEL").is_none();
            let pipeline = if use_tg_parallel {
                &self.batched_tg_parallel_pipeline
            } else {
                &self.batched_pipeline
            };
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(ln.as_ref()), 0);
            // #117 FIX B race-fix: use set_bytes to inline params per-encoder.
            // The shared d_params_buf gets overwritten between commit and GPU
            // execution if we don't wait; set_bytes copies the data into the
            // command buffer at encode time, eliminating the race.
            let bp_bytes = &bp as *const BParams as *const std::ffi::c_void;
            enc.set_bytes(1, std::mem::size_of::<BParams>() as u64, bp_bytes);
            enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_children.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_contributions.as_ref()), 0);
            enc.set_buffer(5, Some(self.d_folded_masks.as_ref()), 0);
            enc.set_buffer(6, Some(self.d_strategy.as_ref()), strat_byte_off as u64);
            enc.set_buffer(7, Some(self.d_infoset_offsets.as_ref()), 0);
            enc.set_buffer(8, Some(self.d_river_reach.as_ref()), 0);
            enc.set_buffer(9, Some(self.d_river_cfv_batch.as_ref()), cfv_byte_off as u64);
            enc.set_buffer(10, Some(self.d_regrets.as_ref()), regret_byte_off as u64);
            enc.set_buffer(11, Some(self.d_cum_strategy.as_ref()), cum_byte_off as u64);
            enc.set_buffer(12, Some(self.d_initial_weight.as_ref()), 0);
            enc.set_buffer(13, Some(self.d_river_sorted_str.as_ref()), sos_byte_off as u64);
            enc.set_buffer(14, Some(self.d_river_sorted_idx.as_ref()), sos_byte_off as u64);
            enc.set_buffer(15, Some(self.d_river_pl_str.as_ref()), sps_byte_off as u64);
            enc.set_buffer(16, Some(self.d_river_pl_idx.as_ref()), sps_byte_off as u64);
            enc.set_buffer(17, Some(self.d_hand_cards.as_ref()), 0);
            enc.set_buffer(18, Some(self.d_river_board_mask.as_ref()), prob_byte_off as u64);
            enc.set_buffer(19, Some(self.d_debug_out.as_ref()), 0);
            enc.set_buffer(20, Some(self.d_rake_marker.as_ref()), 0);

            let (n_groups, tg_width): (u64, u64) = if use_tg_parallel {
                // 1 threadgroup per node; threads cooperate on per-h work.
                // num_outcomes is 1 for the river path so total groups = count.
                let max_tpg = pipeline.max_total_threads_per_threadgroup() as u64;
                // Env tunable for fast iteration on tg_size optimum.
                let env_tg = std::env::var("SOLVER_TG_SIZE")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok());
                let w = env_tg.unwrap_or(64).min(max_tpg).max(32);
                (count as u64, w)
            } else {
                // Serial kernel: 32 threads per group, 1 thread per node.
                let w: u64 = 32;
                ((count as u64 + w - 1) / w, w)
            };
            let grid_size = metal::MTLSize { width: n_groups, height: 1, depth: 1 };
            let tg_size = metal::MTLSize { width: tg_width, height: 1, depth: 1 };
            enc.dispatch_thread_groups(grid_size, tg_size);
            enc.end_encoding();
            cmd.commit();  // commit but DON'T wait — GPU queues sequentially
            last_cmd = Some(cmd);
        }

        // Wait only on the LAST committed command buffer. Metal serializes
        // command-buffer execution within a queue, so waiting on the last one
        // waits for all prior ones to complete.
        if let Some(cmd) = last_cmd {
            self.maybe_wait(cmd);
        }
    }

    pub fn bottom_up_turn(
        &self, ctx: &MetalContext,
        ti: usize, traverser: u32, params: &DcfrParams,
    ) {
        let nh = self.nh;
        let np = self.num_players as usize;
        let num_opp = np - 1;

        let buf_off = self.infoset_byte_offset(BufferZone::Turn { ti }) as usize;
        let strat_byte_off = buf_off;
        let regret_byte_off = buf_off;
        let cum_byte_off = buf_off;
        let cfv_byte_off = ti * self.nn * self.nh * 4;

        // Turn zone uses sorted arrays for this turn card (indexed by raw card value)
        let tc_card = self.turn_deck[ti] as usize;
        let sos_byte_off = (tc_card * num_opp * nh) * 2;
        let sps_byte_off = (tc_card * num_opp * nh) * 2;

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct BParams {
            level_count: i32, num_outcomes: i32, cfv_batch_stride: i32,
            sorted_opp_stride: i32, num_players: i32, nh: i32,
            traverser: u32, alpha_t: f32, beta_t: f32, gamma_t: f32,
            regret_floor: f32, starting_pot: i32, num_combinations: f32,
            regret_outcome_stride: i32, cum_outcome_stride: i32,
            pruning_enabled: i32, pruning_threshold: f32,
            iteration: i32, pruning_stride: i32, board_state: i32,
            rake_rate: f32, rake_cap: f32,
        }

        // #117 Fix A+B applied to turn (mirror of bottom_up_river fix).
        let mut last_cmd: Option<&metal::CommandBufferRef> = None;

        for level in (0..=self.max_depth).rev() {
            let count = self.turn_zone_counts[level];
            if count == 0 { continue; }
            let ln = self.d_turn_zone_nodes[level].as_ref().unwrap();

            let bp = BParams {
                level_count: count as i32,
                num_outcomes: 1,
                cfv_batch_stride: (self.nn * nh) as i32,
                sorted_opp_stride: (num_opp * nh) as i32,
                num_players: np as i32,
                nh: nh as i32,
                traverser,
                alpha_t: params.alpha_t,
                beta_t: params.beta_t,
                gamma_t: params.gamma_t,
                regret_floor: self.regret_floor,
                starting_pot: self.starting_pot,
                num_combinations: self.num_combinations,
                regret_outcome_stride: self.turn_stride as i32,
                cum_outcome_stride: self.turn_stride as i32,
                pruning_enabled: if self.pruning_enabled { 1 } else { 0 },
                pruning_threshold: self.pruning_threshold,
                iteration: self.iteration as i32,
                pruning_stride: self.pruning_stride as i32,
                board_state: 1,
                rake_rate: self.rake_rate,
                rake_cap: self.rake_cap,
            };

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            // Step 2.D.28: branch on num_opp (see bottom_up_river for rationale).
            // num_opp >= 2: use new kernel. HU (num_opp==1) stays on serial.
            let use_tg_parallel = num_opp >= 2
                && std::env::var_os("SOLVER_DISABLE_TG_PARALLEL").is_none();
            let pipeline = if use_tg_parallel {
                &self.batched_tg_parallel_pipeline
            } else {
                &self.batched_pipeline
            };
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(ln.as_ref()), 0);
            let bp_bytes = &bp as *const BParams as *const std::ffi::c_void;
            enc.set_bytes(1, std::mem::size_of::<BParams>() as u64, bp_bytes);
            enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_children.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_contributions.as_ref()), 0);
            enc.set_buffer(5, Some(self.d_folded_masks.as_ref()), 0);
            enc.set_buffer(6, Some(self.d_strategy.as_ref()), strat_byte_off as u64);
            enc.set_buffer(7, Some(self.d_infoset_offsets.as_ref()), 0);
            enc.set_buffer(8, Some(self.d_turn_reach.as_ref()), 0);
            enc.set_buffer(9, Some(self.d_turn_cfv_batch.as_ref()), cfv_byte_off as u64);
            enc.set_buffer(10, Some(self.d_regrets.as_ref()), regret_byte_off as u64);
            enc.set_buffer(11, Some(self.d_cum_strategy.as_ref()), cum_byte_off as u64);
            enc.set_buffer(12, Some(self.d_initial_weight.as_ref()), 0);
            enc.set_buffer(13, Some(self.d_turn_sorted_str.as_ref()), sos_byte_off as u64);
            enc.set_buffer(14, Some(self.d_turn_sorted_idx.as_ref()), sos_byte_off as u64);
            enc.set_buffer(15, Some(self.d_turn_pl_str.as_ref()), sps_byte_off as u64);
            enc.set_buffer(16, Some(self.d_turn_pl_idx.as_ref()), sps_byte_off as u64);
            enc.set_buffer(17, Some(self.d_hand_cards.as_ref()), 0);
            enc.set_buffer(18, Some(self.d_turn_chance_prob.as_ref()), (ti * nh * 4) as u64);
            enc.set_buffer(19, Some(self.d_debug_out.as_ref()), 0);
            enc.set_buffer(20, Some(self.d_rake_marker.as_ref()), 0);

            let (n_groups, tg_width): (u64, u64) = if use_tg_parallel {
                let max_tpg = pipeline.max_total_threads_per_threadgroup() as u64;
                // Env tunable for fast iteration on tg_size optimum.
                let env_tg = std::env::var("SOLVER_TG_SIZE")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok());
                let w = env_tg.unwrap_or(64).min(max_tpg).max(32);
                (count as u64, w)
            } else {
                let w: u64 = 32;
                ((count as u64 + w - 1) / w, w)
            };
            let grid_size = metal::MTLSize { width: n_groups, height: 1, depth: 1 };
            let tg_size = metal::MTLSize { width: tg_width, height: 1, depth: 1 };
            enc.dispatch_thread_groups(grid_size, tg_size);
            enc.end_encoding();
            cmd.commit();
            last_cmd = Some(cmd);
        }

        if let Some(cmd) = last_cmd {
            self.maybe_wait(cmd);
        }
    }

    pub fn bottom_up_flop(&self, ctx: &MetalContext, traverser: u32, params: &DcfrParams) {
        let nh = self.nh;
        let np = self.num_players as usize;

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct BuParams {
            level_count: i32, num_players: i32, nh: i32,
            traverser: u32, alpha_t: f32, beta_t: f32, gamma_t: f32,
            regret_floor: f32, starting_pot: i32, num_combinations: f32,
            rake_rate: f32, rake_cap: f32,
            // P1: Pluribus pruning fields (mirror BatchedParams). The flop
            // kernel reads board_state=0 (Flop), so the "no pruning on river"
            // carve-out is automatically satisfied at the flop level.
            pruning_enabled: i32, pruning_threshold: f32,
            iteration: i32, pruning_stride: i32, board_state: i32,
        }

        // #117 Fix A+B applied to flop (mirror of bottom_up_river fix).
        let mut last_cmd: Option<&metal::CommandBufferRef> = None;

        for level in (0..=self.max_depth).rev() {
            let count = self.flop_zone_counts[level];
            if count == 0 { continue; }
            let ln = self.d_flop_zone_nodes[level].as_ref().unwrap();

            let bp = BuParams {
                level_count: count as i32,
                num_players: np as i32,
                nh: nh as i32,
                traverser,
                alpha_t: params.alpha_t,
                beta_t: params.beta_t,
                gamma_t: params.gamma_t,
                regret_floor: self.regret_floor,
                starting_pot: self.starting_pot,
                num_combinations: self.num_combinations,
                rake_rate: self.rake_rate,
                rake_cap: self.rake_cap,
                // Flop zone runs at board_state=0 (Flop). board_state != 2
                // is automatically true → pruning carve-out for river
                // doesn't trigger here. The kernel still respects the
                // re_enable_iter and action-leads-to-terminal carve-outs.
                pruning_enabled: if self.pruning_enabled { 1 } else { 0 },
                pruning_threshold: self.pruning_threshold,
                iteration: self.iteration as i32,
                pruning_stride: self.pruning_stride as i32,
                board_state: 0,
            };

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            // M2 follow-up: branch on num_opp. K>=2 (3p+) uses the tg-parallel
            // flop kernel. HU stays on serial (same reasoning as the batched
            // kernel — per-node showdown work too small for within-node
            // parallelism to win).
            let num_opp = np - 1;
            let use_tg_parallel = num_opp >= 2
                && std::env::var_os("SOLVER_DISABLE_TG_PARALLEL").is_none();
            let pipeline = if use_tg_parallel {
                &self.bottom_up_tg_parallel_pipeline
            } else {
                &self.bottom_up_pipeline
            };
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(ln.as_ref()), 0);
            let bp_bytes = &bp as *const BuParams as *const std::ffi::c_void;
            enc.set_bytes(1, std::mem::size_of::<BuParams>() as u64, bp_bytes);
            enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_children.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_contributions.as_ref()), 0);
            enc.set_buffer(5, Some(self.d_folded_masks.as_ref()), 0);
            enc.set_buffer(6, Some(self.d_strategy.as_ref()), 0);
            enc.set_buffer(7, Some(self.d_infoset_offsets.as_ref()), 0);
            enc.set_buffer(8, Some(self.d_reach.as_ref()), 0);
            enc.set_buffer(9, Some(self.d_cfv.as_ref()), 0);
            enc.set_buffer(10, Some(self.d_regrets.as_ref()), 0);
            enc.set_buffer(11, Some(self.d_cum_strategy.as_ref()), 0);
            enc.set_buffer(12, Some(self.d_initial_weight.as_ref()), 0);
            enc.set_buffer(13, Some(self.d_sorted_opp_strength.as_ref()), 0);
            enc.set_buffer(14, Some(self.d_sorted_opp_indices.as_ref()), 0);
            enc.set_buffer(15, Some(self.d_sorted_pl_strength.as_ref()), 0);
            enc.set_buffer(16, Some(self.d_sorted_pl_indices.as_ref()), 0);
            enc.set_buffer(17, Some(self.d_hand_cards.as_ref()), 0);
            enc.set_buffer(18, Some(self.d_rake_marker.as_ref()), 0);

            let (n_groups, tg_width): (u64, u64) = if use_tg_parallel {
                // 1 threadgroup per node.
                let max_tpg = pipeline.max_total_threads_per_threadgroup() as u64;
                let w = max_tpg.min(64).max(32);
                (count as u64, w)
            } else {
                let w: u64 = 32;
                ((count as u64 + w - 1) / w, w)
            };
            let grid_size = metal::MTLSize { width: n_groups, height: 1, depth: 1 };
            let tg_size = metal::MTLSize { width: tg_width, height: 1, depth: 1 };
            enc.dispatch_thread_groups(grid_size, tg_size);
            enc.end_encoding();
            cmd.commit();
            last_cmd = Some(cmd);
        }

        if let Some(cmd) = last_cmd {
            self.maybe_wait(cmd);
        }
    }

    // ─── Chance node transitions ───

    pub fn chance_accumulate_river(&self, ctx: &MetalContext, ti: usize, n_river: usize) {
        // CPU: for each ri: river_accum[child*nh+h] += cp(tc,ri,h) * cfv[child*nh+h]
        // Metal: call vcfr_chance_accumulate once per river outcome ri.
        // The kernel takes a single outcome index and dispatches cc_count*nh threads.
        let nh = self.nh;
        let cc = self.river_cc_count;
        let nn = self.nn;

        // Zero river_accum first
        self.zero_buffer_name(ctx, 1);

        for ri in 0..n_river {
            // Upload params for this outcome

            let num_cc_val: i32 = cc as i32;
            let nh_val: i32 = nh as i32;
            let outcome_val: i32 = ri as i32;

            let total = cc * nh;
            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.chance_accum_pipeline);
            enc.set_buffer(0, Some(self.d_river_accum.as_ref()), 0);
            let cfv_byte_off = (ri * nn * nh) * 4;
            enc.set_buffer(1, Some(self.d_river_cfv_batch.as_ref()), cfv_byte_off as u64);
            let prob_byte_off = (ti * self.max_river * nh) * 4;
            enc.set_buffer(2, Some(self.d_river_chance_prob.as_ref()), prob_byte_off as u64);
            enc.set_buffer(3, Some(self.d_river_chance_children.as_ref()), 0);
            // #117 Fix C race-fix: inline scalar params per-encoder.
            enc.set_bytes(4, 4, &num_cc_val as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &nh_val as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &outcome_val as *const i32 as *const std::ffi::c_void);

            let (grid, tg) = ctx.dispatch_1d(total, 256);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            self.maybe_wait(cmd);
        }
    }

    pub fn chance_finalize_river(&self, ctx: &MetalContext, ti: usize) {
        // Copy river_accum into turn CFV batch at river chance children positions.
        // CPU: turn_cfv[child*nh+h] = river_cfv_accum[child*nh+h]
        // Uses vcfr_chance_finalize: cfv[child*nh+h] = cfv_accum[child*nh+h]
        let nh = self.nh;
        let cc = self.river_cc_count;
        let total = cc * nh;


        let num_cc_val: i32 = cc as i32;
        let nh_val_local: i32 = nh as i32;

        let turn_cfv_byte_off = (ti * self.nn * nh) * 4;

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.chance_final_pipeline);
        enc.set_buffer(0, Some(self.d_turn_cfv_batch.as_ref()), turn_cfv_byte_off as u64);
        enc.set_buffer(1, Some(self.d_river_accum.as_ref()), 0);
        enc.set_buffer(2, Some(self.d_river_chance_children.as_ref()), 0);
        enc.set_bytes(3, 4, &num_cc_val as *const i32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &nh_val_local as *const i32 as *const std::ffi::c_void);

        let (grid, tg) = ctx.dispatch_1d(total, 256);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        self.maybe_wait(cmd);
    }

    pub fn chance_accumulate_turn(&self, ctx: &MetalContext) {
        // CPU: for each ti: flop_cfv[child*nh+h] += cp(ti,h) * turn_cfv[child*nh+h]
        // Metal: call vcfr_chance_accumulate once per turn outcome ti.
        let nh = self.nh;
        let cc = self.turn_cc_count;
        let nn = self.nn;
        let n_turn = self.n_turn;

        for ti in 0..n_turn {

            let num_cc_val: i32 = cc as i32;
            let nh_val_local: i32 = nh as i32;
            let outcome_val: i32 = ti as i32;

            let total = cc * nh;
            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.chance_accum_pipeline);
            enc.set_buffer(0, Some(self.d_cfv.as_ref()), 0);
            let cfv_byte_off = (ti * nn * nh) * 4;
            enc.set_buffer(1, Some(self.d_turn_cfv_batch.as_ref()), cfv_byte_off as u64);
            enc.set_buffer(2, Some(self.d_turn_chance_prob.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_turn_chance_children.as_ref()), 0);
            enc.set_bytes(4, 4, &num_cc_val as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &nh_val_local as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &outcome_val as *const i32 as *const std::ffi::c_void);

            let (grid, tg) = ctx.dispatch_1d(total, 256);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            self.maybe_wait(cmd);
        }
    }

    pub fn chance_finalize_turn(&self, ctx: &MetalContext) {
        // No-op: chance_accumulate_turn now writes directly to d_cfv.
        // d_cfv was zeroed at the start of each traverser iteration.
        let _ = ctx;
    }

    // ─── Helpers ───

    pub fn zero_buffer_name(&self, ctx: &MetalContext, name: u8) {
        let len = match name {
            0 => self.d_river_cfv_batch.len(),
            1 => self.d_river_accum.len(),
            2 => self.d_turn_cfv_batch.len(),
            3 => self.d_turn_accum.len(),
            6 => self.d_turn_reach.len(),
            7 => self.d_river_reach.len(),
            99 => self.d_reach.len(),
            100 => self.d_cfv.len(),
            _ => panic!("unknown buffer"),
        };

        let count_val: i32 = len as i32;

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.zero_pipeline);
        let buf_ref: &metal::BufferRef = match name {
            0 => self.d_river_cfv_batch.as_ref(),
            1 => self.d_river_accum.as_ref(),
            2 => self.d_turn_cfv_batch.as_ref(),
            3 => self.d_turn_accum.as_ref(),
            6 => self.d_turn_reach.as_ref(),
            7 => self.d_river_reach.as_ref(),
            99 => self.d_reach.as_ref(),
            100 => self.d_cfv.as_ref(),
            _ => panic!("unknown buffer"),
        };
        enc.set_buffer(0, Some(buf_ref), 0);
        enc.set_bytes(1, 4, &count_val as *const i32 as *const std::ffi::c_void);
        let (grid, tg) = ctx.dispatch_1d(len, 256);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        self.maybe_wait(cmd);
    }

    fn upload_params<T: Copy>(&self, params: &T) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                params as *const T as *const u8,
                std::mem::size_of::<T>(),
            )
        };
        let slice = unsafe { (*self.d_params_buf.get()).as_mut_slice() };
        slice[..bytes.len()].copy_from_slice(bytes);
    }

    fn params_buf_ref(&self) -> &metal::BufferRef {
        unsafe { (*self.d_params_buf.get()).as_ref() }
    }

    fn launch_seed_reach(
        &self, ctx: &MetalContext,
        dst: &MetalBuffer<f32>, src: &MetalBuffer<f32>,
        chance_children: &MetalBuffer<u32>, count: usize, np_nh: usize,
    ) {
        if count == 0 { return; }
        let total = count * np_nh;


        let count_val: i32 = count as i32;
        let np_nh_val: i32 = np_nh as i32;

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.seed_reach_pipeline);
        enc.set_buffer(0, Some(dst.as_ref()), 0);
        enc.set_buffer(1, Some(src.as_ref()), 0);
        enc.set_buffer(2, Some(chance_children.as_ref()), 0);
        enc.set_bytes(3, 4, &count_val as *const i32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &np_nh_val as *const i32 as *const std::ffi::c_void);

        let (grid, tg) = ctx.dispatch_1d(total, 256);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        self.maybe_wait(cmd);
    }

    fn launch_top_down_zone(
        &self, ctx: &MetalContext,
        level_nodes: &MetalBuffer<u32>, count: usize,
        reach: &MetalBuffer<f32>,
        strategy_byte_offset: u64,
    ) {
        let nh = self.nh;
        let np = self.num_players as usize;

        #[repr(C)]
        #[repr(C)]

        #[derive(Clone, Copy)]
        struct Params { level_count: i32, num_players: i32, nh: i32 }
        let p = Params { level_count: count as i32, num_players: np as i32, nh: nh as i32 };

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.top_down_pipeline);
        enc.set_buffer(0, Some(level_nodes.as_ref()), 0);
        enc.set_bytes(1, std::mem::size_of::<Params>() as u64,
            &p as *const Params as *const std::ffi::c_void);
        enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
        enc.set_buffer(3, Some(self.d_children.as_ref()), 0);
        enc.set_buffer(4, Some(self.d_strategy.as_ref()), strategy_byte_offset);
        enc.set_buffer(5, Some(self.d_infoset_offsets.as_ref()), 0);
        enc.set_buffer(6, Some(reach.as_ref()), 0);

        let max_tpg = self.top_down_pipeline.max_total_threads_per_threadgroup() as usize;
        let (grid, tg) = ctx.dispatch_2d(count, nh, max_tpg);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        self.maybe_wait(cmd);
    }

    pub fn download_regrets(&self) -> Vec<f32> { self.d_regrets.to_vec() }
    pub fn download_cfv(&self) -> Vec<f32> { self.d_cfv.to_vec() }
    pub fn download_reach(&self) -> Vec<f32> { self.d_reach.to_vec() }
    /// Step 5 chokepoint instrumentation: download the per-(terminal-node,
    /// hand) rake marker buffer. Returns Vec<u8> of length nn × nh. Cell
    /// values: 0 = unmarked (BUG — terminal bypassed the chokepoint),
    /// 1 = rake-applied (flop_seen=true), 2 = rake-correctly-skipped
    /// (flop_seen=false per no-flop-no-drop). After a representative
    /// solve, ALL terminal-node-hand cells must be 1 or 2; any 0 at a
    /// terminal indicates a payoff path bypassed the rake-applying
    /// chokepoint multiway_brute_force_showdown.
    pub fn download_rake_marker(&self) -> Vec<u8> { self.d_rake_marker.to_vec() }
    pub fn download_turn_reach(&self) -> Vec<f32> { self.d_turn_reach.to_vec() }
    pub fn download_river_reach(&self) -> Vec<f32> { self.d_river_reach.to_vec() }
    pub fn download_turn_cfv_batch(&self) -> Vec<f32> { self.d_turn_cfv_batch.to_vec() }
    pub fn download_river_cfv_batch(&self) -> Vec<f32> { self.d_river_cfv_batch.to_vec() }
    pub fn download_river_accum(&self) -> Vec<f32> { self.d_river_accum.to_vec() }
    pub fn download_turn_accum(&self) -> Vec<f32> { self.d_turn_accum.to_vec() }
    pub fn upload_regrets(&mut self, data: &[f32]) {
        self.d_regrets.as_mut_slice().copy_from_slice(data);
    }
    pub fn upload_cum_strategy(&mut self, data: &[f32]) {
        self.d_cum_strategy.as_mut_slice().copy_from_slice(data);
    }
    /// Test-only: overwrite d_strategy with arbitrary values. Used by the
    /// disturbance test to prove that the GPU's compute_all_strategies(ctx)
    /// pass at the start of run() overwrites the strategy buffer before
    /// anything reads it — if true, init values (including NaN) leak nowhere.
    pub fn poison_strategy(&mut self, data: &[f32]) {
        self.d_strategy.as_mut_slice().copy_from_slice(data);
    }
    pub fn strategy_buffer_len(&self) -> usize { self.d_strategy.to_vec().len() }
    pub fn river_outcomes_per_turn(&self) -> &[usize] { &self.river_outcomes_per_turn }
    pub fn compute_all_strategies(&self, ctx: &MetalContext) {
        // Flop zone: single outcome
        self.compute_strategies_single(
            ctx, &self.d_flop_decision_ids, &self.d_flop_infoset_offsets,
            self.flop_infosets, self.flop_offset, self.flop_stride,
        );

        // Turn zone: n_turn outcomes
        self.compute_strategies_batched(
            ctx, &self.d_turn_decision_ids, &self.d_turn_infoset_offsets,
            self.turn_infosets, self.n_turn, self.turn_offset, self.turn_stride,
        );

        // River zone:
        //   InMemory: batched over all n_turn * max_river outcomes.
        //   DiskBacked: SKIPPED here. The per-pair flow in `run_one_iter`
        //     calls `compute_river_strategy_pair(ti, ri)` after each
        //     `load_river_pair_gpu(ti, ri)` so the strategy is computed
        //     from the regrets currently loaded into the scratch.
        match self.gpu_river_mode {
            GpuRiverMode::InMemory => {
                self.compute_strategies_batched(
                    ctx, &self.d_river_decision_ids, &self.d_river_infoset_offsets,
                    self.river_infosets, self.n_turn * self.max_river,
                    self.river_offset, self.river_stride,
                );
            }
            GpuRiverMode::DiskBacked { .. } => {
                // No-op; compute_river_strategy_pair handles per-pair.
            }
        }
    }

    /// Compute river strategy for the single (ti, ri) pair currently
    /// loaded into the scratch (DiskBacked) or for the per-pair slot in
    /// the full buffer (InMemory). One-pair version of the batched call
    /// in `compute_all_strategies` — same kernel, num_outcomes = 1, base
    /// offset routed through `river_outcome_idx`.
    ///
    /// In DiskBacked, the caller must invoke `load_river_pair_gpu(ti, ri)`
    /// before calling this so the scratch holds the right regrets.
    pub fn compute_river_strategy_pair(&self, ctx: &MetalContext, ti: usize, ri: usize) {
        let outcome_idx = self.river_outcome_idx(ti, ri);
        let base_offset = self.infoset_float_offset(BufferZone::River { outcome_idx });
        self.compute_strategies_batched(
            ctx, &self.d_river_decision_ids, &self.d_river_infoset_offsets,
            self.river_infosets, 1, base_offset, self.river_stride,
        );
    }
    pub fn download_strategy(&self) -> Vec<f32> { self.d_strategy.to_vec() }
    pub fn download_cum_strategy(&self) -> Vec<f32> { self.d_cum_strategy.to_vec() }
    pub fn download_river_sorted_str(&self) -> Vec<u16> { self.d_river_sorted_str.to_vec() }
    pub fn download_river_sorted_idx(&self) -> Vec<u16> { self.d_river_sorted_idx.to_vec() }
    pub fn download_debug(&self) -> Vec<f32> { self.d_debug_out.to_vec() }

    pub fn layout(&self) -> (usize, usize, usize, usize, usize, usize) {
        (self.flop_stride, self.turn_stride, self.river_stride,
         self.turn_total, self.river_total, self.n_turn)
    }

    pub fn iteration(&self) -> u32 { self.iteration }
    pub fn set_iteration(&mut self, i: u32) { self.iteration = i; }

    // ─── Phase 1.A pruning accessors ───
    pub fn pruning_enabled(&self) -> bool { self.pruning_enabled }
    pub fn set_pruning(&mut self, enabled: bool, threshold: f32, stride: u32) {
        self.pruning_enabled = enabled;
        self.pruning_threshold = threshold;
        self.pruning_stride = stride.max(1);
    }
    pub fn pruning_threshold(&self) -> f32 { self.pruning_threshold }
    pub fn pruning_stride(&self) -> u32 { self.pruning_stride }
    pub fn debug_params(&self, ctx: &MetalContext) -> Vec<f32> {
        let device = ctx.device();
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let workspace_target = format!("{}/../target", manifest);
        let metallib_path = std::env::var("METALLIB_PATH")
            .unwrap_or_else(|_| format!("{}/debug/build/solver-core-78298d1a81aec300/out/solver.metallib", workspace_target));
        let lib = device.new_library_with_data(
            &std::fs::read(&metallib_path).unwrap_or_else(|e| panic!("read metallib {:?}: {}", metallib_path, e))
        ).expect("load lib");
        let func = lib.get_function("debug_params", None).expect("function");
        let pipeline = device.new_compute_pipeline_state_with_function(&func).expect("pipeline");

        let d_output = MetalBuffer::<f32>::zeros(device, 15);
        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(d_output.as_ref()), 0);
        enc.set_buffer(1, Some(unsafe { (*self.d_params_buf.get()).as_ref() }), 0);
        let (grid, tg) = ctx.dispatch_1d(1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        self.maybe_wait(cmd);

        d_output.to_vec()
    }

    pub fn river_infosets(&self) -> usize { self.river_infosets }
    pub fn n_turn(&self) -> usize { self.n_turn }
    pub fn max_river(&self) -> usize { self.max_river }

    /// Run the debug_multiway_sweep kernel with given inputs, return product-formula output
    pub fn debug_multiway_sweep(
        &self, ctx: &MetalContext,
        opp0_reach: &[f32], opp1_reach: &[f32], nh: usize,
        opp_str: &[u16], opp_idx: &[u16],
        pl_str: &[u16], pl_idx: &[u16],
        half_pot: f32, num_active_opp: i32,
    ) -> Vec<f32> {
        use metal::ComputePipelineState;
        let device = ctx.device();
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let workspace_target = format!("{}/../target", manifest);
        let metallib_path = std::env::var("METALLIB_PATH")
            .unwrap_or_else(|_| format!("{}/debug/build/solver-core-78298d1a81aec300/out/solver.metallib", workspace_target));
        let lib = device.new_library_with_data(
            &std::fs::read(&metallib_path).unwrap_or_else(|e| panic!("read metallib {:?}: {}", metallib_path, e))
        ).expect("load lib");
        let func = lib.get_function("debug_multiway_sweep", None).expect("function");
        let pipeline = device.new_compute_pipeline_state_with_function(&func).expect("pipeline");

        let d_output = MetalBuffer::<f32>::zeros(device, nh);
        let d_opp0_reach = MetalBuffer::from_slice(device, opp0_reach);
        let d_opp1_reach = MetalBuffer::from_slice(device, opp1_reach);
        let d_opp_str = MetalBuffer::from_slice(device, opp_str);
        let d_opp_idx = MetalBuffer::from_slice(device, opp_idx);
        let d_pl_str = MetalBuffer::from_slice(device, pl_str);
        let d_pl_idx = MetalBuffer::from_slice(device, pl_idx);
        let d_hand_cards = MetalBuffer::from_slice(device, &self.d_hand_cards.as_slice());

        let nh_buf = MetalBuffer::<i32>::from_slice(device, &[nh as i32]);
        let hp_buf = MetalBuffer::<f32>::from_slice(device, &[half_pot]);
        let nao_buf = MetalBuffer::<i32>::from_slice(device, &[num_active_opp]);

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(d_output.as_ref()), 0);
        enc.set_buffer(1, Some(d_opp0_reach.as_ref()), 0);
        enc.set_buffer(2, Some(d_opp1_reach.as_ref()), 0);
        enc.set_buffer(3, Some(d_opp_str.as_ref()), 0);
        enc.set_buffer(4, Some(d_opp_idx.as_ref()), 0);
        enc.set_buffer(5, Some(d_pl_str.as_ref()), 0);
        enc.set_buffer(6, Some(d_pl_idx.as_ref()), 0);
        enc.set_buffer(7, Some(d_hand_cards.as_ref()), 0);
        enc.set_buffer(8, Some(nh_buf.as_ref()), 0);
        enc.set_buffer(9, Some(hp_buf.as_ref()), 0);
        enc.set_buffer(10, Some(nao_buf.as_ref()), 0);

        let grid = metal::MTLSize { width: 1, height: 1, depth: 1 };
        let tg = metal::MTLSize { width: 1, height: 1, depth: 1 };
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        self.maybe_wait(cmd);

        d_output.to_vec()
    }

    /// Run the debug_sweep kernel with given inputs, return sweep output
    pub fn debug_sweep(
        &self, ctx: &MetalContext,
        opp_reach: &[f32], nh: usize,
        opp_str: &[u16], opp_idx: &[u16],
        pl_str: &[u16], pl_idx: &[u16],
    ) -> Vec<f32> {
        use metal::ComputePipelineState;
        let device = ctx.device();
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        // CARGO_MANIFEST_DIR is solver-core/, target/ is one level up in workspace
        let workspace_target = format!("{}/../target", manifest);
        let metallib_path = std::env::var("METALLIB_PATH")
            .unwrap_or_else(|_| format!("{}/debug/build/solver-core-78298d1a81aec300/out/solver.metallib", workspace_target));
        let lib = device.new_library_with_data(
            &std::fs::read(&metallib_path).unwrap_or_else(|e| panic!("read metallib {:?}: {}", metallib_path, e))
        ).expect("load lib");
        let func = lib.get_function("debug_sweep", None).expect("function");
        let pipeline = device.new_compute_pipeline_state_with_function(&func).expect("pipeline");

        let d_output = MetalBuffer::<f32>::zeros(device, nh);
        let d_opp_reach = MetalBuffer::from_slice(device, opp_reach);
        let d_opp_str = MetalBuffer::from_slice(device, opp_str);
        let d_opp_idx = MetalBuffer::from_slice(device, opp_idx);
        let d_pl_str = MetalBuffer::from_slice(device, pl_str);
        let d_pl_idx = MetalBuffer::from_slice(device, pl_idx);
        let d_hand_cards = MetalBuffer::from_slice(device, &self.d_hand_cards.as_slice());

        let nh_buf = MetalBuffer::<i32>::from_slice(device, &[nh as i32]);

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(d_output.as_ref()), 0);
        enc.set_buffer(1, Some(d_opp_reach.as_ref()), 0);
        enc.set_buffer(2, Some(d_opp_str.as_ref()), 0);
        enc.set_buffer(3, Some(d_opp_idx.as_ref()), 0);
        enc.set_buffer(4, Some(d_pl_str.as_ref()), 0);
        enc.set_buffer(5, Some(d_pl_idx.as_ref()), 0);
        enc.set_buffer(6, Some(d_hand_cards.as_ref()), 0);
        enc.set_buffer(7, Some(nh_buf.as_ref()), 0);
        let (grid, tg) = ctx.dispatch_1d(1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        self.maybe_wait(cmd);

        d_output.to_vec()
    }
}

fn compute_infoset_offsets(tree: &FlatTree, solver: &FlopStartVectorCfr) -> Vec<u32> {
    let nn = tree.num_nodes();
    let mut offsets = vec![UNUSED; nn];
    for idx in 0..nn {
        let node = &tree.nodes[idx];
        if !node.is_player() { continue; }
        let zone = solver.zones()[idx];
        let local = match zone {
            crate::solver::flop_start_vector_cfr::Zone::Flop => solver.flop_local_offset()[idx],
            crate::solver::flop_start_vector_cfr::Zone::Turn => solver.turn_local_offset()[idx],
            crate::solver::flop_start_vector_cfr::Zone::River => solver.river_local_offset()[idx],
            crate::solver::flop_start_vector_cfr::Zone::Preflop => unreachable!(
                "Zone::Preflop in Metal flop-start offset table; preflop \
                 processing lives in P1.5.4 (#44)"
            ),
        };
        offsets[idx] = local as u32;
    }
    offsets
}
