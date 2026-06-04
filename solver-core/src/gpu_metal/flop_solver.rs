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
///   Flop:  regrets[infoset * MAX_NA * nh + a * nh + h]
///   Turn:  regrets[flop_total + tc * turn_stride + infoset * MAX_NA * nh + a * nh + h]
///   River: regrets[flop_total + turn_total + (tc*max_river+rc) * river_stride + infoset * MAX_NA * nh + a * nh + h]
///
/// Strategy computed from regrets via regret matching, separately per outcome.
/// DCFR discount applied inline during the bottom-up pass.

use crate::gpu_metal::{MetalBuffer, MetalContext};
use crate::solver::flop_start_game::FlopStartGame;
use crate::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use crate::tree::flat::{FlatTree, MAX_NA};
use metal::ComputePipelineState;

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
    batched_pipeline: ComputePipelineState,
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

    // Strides for per-outcome dimensional layout
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

        // Layout dimensions
        let flop_infosets = cpu_solver.flop_infosets();
        let turn_infosets = cpu_solver.turn_infosets();
        let river_infosets = cpu_solver.river_infosets();
        let flop_stride = flop_infosets * MAX_NA * nh;
        let turn_stride = turn_infosets * MAX_NA * nh;
        let river_stride = river_infosets * MAX_NA * nh;
        let turn_total = n_turn * turn_stride;
        let river_total = n_turn * max_river * river_stride;
        let flop_offset = 0;
        let turn_offset = flop_stride;
        let river_offset = flop_stride + turn_total;

        // Load pipelines
        let strategies_pipeline = ctx.create_pipeline("vcfr_compute_strategies").expect("strategies");
        let strategies_batched_pipeline = ctx.create_pipeline("vcfr_compute_strategies_batched").expect("strategies_batched");
        let init_reach_pipeline = ctx.create_pipeline("vcfr_init_reach").expect("init_reach");
        let top_down_pipeline = ctx.create_pipeline("vcfr_top_down_reach").expect("top_down");
        let bottom_up_pipeline = ctx.create_pipeline("vcfr_bottom_up").expect("bottom_up");
        let batched_pipeline = ctx.create_pipeline("vcfr_bottom_up_batched").expect("batched");
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

        // Solver state
        let d_regrets = {
            let mut buf = Vec::with_capacity(flop_stride + turn_total + river_total);
            buf.extend_from_slice(cpu_solver.regrets_flop());
            buf.extend_from_slice(cpu_solver.regrets_turn());
            buf.extend_from_slice(cpu_solver.regrets_river());
            MetalBuffer::from_slice(device, &buf)
        };
        let d_strategy = {
            let mut buf = Vec::with_capacity(flop_stride + turn_total + river_total);
            buf.extend_from_slice(cpu_solver.strategy_flop());
            buf.extend_from_slice(cpu_solver.strategy_turn());
            buf.extend_from_slice(cpu_solver.strategy_river());
            MetalBuffer::from_slice(device, &buf)
        };
        let d_cum_strategy = {
            let mut buf = Vec::with_capacity(flop_stride + turn_total + river_total);
            buf.extend_from_slice(cpu_solver.cum_strategy_flop());
            buf.extend_from_slice(cpu_solver.cum_strategy_turn());
            buf.extend_from_slice(cpu_solver.cum_strategy_river());
            MetalBuffer::from_slice(device, &buf)
        };
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
            strategies_pipeline,
            strategies_batched_pipeline,
            init_reach_pipeline,
            top_down_pipeline,
            bottom_up_pipeline,
            batched_pipeline,
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
        }
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
    pub fn run(
        &mut self,
        ctx: &MetalContext,
        tree: &FlatTree,
        game: &FlopStartGame,
        num_iterations: u32,
    ) {
        let np = self.num_players as usize;
        let nh = self.nh;
        let nn = self.nn;

        for _ in 0..num_iterations {
            let params = DcfrParams::new(self.iteration);
            self.iteration += 1;

            for traverser in 0..np {
                // Sequential: recompute strategies before each traverser
                self.compute_all_strategies(ctx);

                // Flop zone reach
                self.compute_reach_flop(ctx);

                // Zero main CFV (flop_cfv in CPU) for this traverser
                self.zero_buffer_name(ctx, 100);

                // Zero turn CFV batch
                self.zero_buffer_name(ctx, 2);

                // Per turn card: river → turn pipeline
                for ti in 0..self.n_turn {
                    let n_river = self.river_outcomes_per_turn[ti];

                    // Zero river CFV batch and accum
                    self.zero_buffer_name(ctx, 0);
                    self.zero_buffer_name(ctx, 1);

                    // Compute turn reach for this tc
                    self.compute_reach_turn(ctx, ti);

                    // River zone: per river card
                    for ri in 0..n_river {
                        // Compute river reach for this (tc, rc)
                        self.compute_reach_river(ctx, ti, ri);

                        // Bottom-up river zone (single outcome: ri)
                        self.bottom_up_river(ctx, ti, ri, traverser as u32, &params);
                    }

                    // Chance accumulate: weight river CFVs by chance probability → river_accum
                    self.chance_accumulate_river(ctx, ti, n_river);

                    // Chance finalize: copy river_accum into turn CFV batch at river chance children
                    self.chance_finalize_river(ctx, ti);

                    // Bottom-up turn zone for this turn card
                    self.bottom_up_turn(ctx, ti, traverser as u32, &params);
                }

                // Chance accumulate turn: weight turn CFV batch by turn chance probability
                self.chance_accumulate_turn(ctx);

                // Chance finalize turn: sum into main CFV at turn chance children
                self.chance_finalize_turn(ctx);

                // Bottom-up flop zone
                self.bottom_up_flop(ctx, traverser as u32, &params);
            }
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
        self.upload_params(&p);
        let params_buf = self.params_buf_ref();

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.strategies_pipeline);
        enc.set_buffer(0, Some(self.d_regrets.as_ref()), 0);
        enc.set_buffer(1, Some(self.d_strategy.as_ref()), 0);
        enc.set_buffer(2, Some(decision_ids.as_ref()), 0);
        enc.set_buffer(3, Some(self.d_nodes.as_ref()), 0);
        enc.set_buffer(4, Some(infoset_offsets.as_ref()), 0);
        enc.set_buffer(5, Some(params_buf), 0);

        let max_tpg = self.strategies_pipeline.max_total_threads_per_threadgroup() as usize;
        let (grid, tg) = ctx.dispatch_2d(num_infosets, nh, max_tpg);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
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
        self.upload_params(&p);
        let params_buf = self.params_buf_ref();

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.strategies_batched_pipeline);
        enc.set_buffer(0, Some(self.d_regrets.as_ref()), 0);
        enc.set_buffer(1, Some(self.d_strategy.as_ref()), 0);
        enc.set_buffer(2, Some(decision_ids.as_ref()), 0);
        enc.set_buffer(3, Some(self.d_nodes.as_ref()), 0);
        enc.set_buffer(4, Some(infoset_offsets.as_ref()), 0);
        enc.set_buffer(5, Some(params_buf), 0);

        let max_tpg = self.strategies_batched_pipeline.max_total_threads_per_threadgroup() as usize;
        // Map 3D problem to 2D: x = outcome * num_infosets + infoset, y = hand
        let total_x = num_outcomes * num_infosets;
        let (grid, tg) = ctx.dispatch_2d(total_x, nh, max_tpg);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
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
            self.upload_params(&p);

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.init_reach_pipeline);
            enc.set_buffer(0, Some(self.d_reach.as_ref()), 0);
            enc.set_buffer(1, Some(self.d_initial_weight.as_ref()), 0);
            enc.set_buffer(2, Some(unsafe { (*self.d_params_buf.get()).as_ref() }), 0);

            let (grid, tg) = ctx.dispatch_1d(nn * np_nh, 256);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
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
            self.upload_params(&p);

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.top_down_pipeline);
            enc.set_buffer(0, Some(ln.as_ref()), 0);
            enc.set_buffer(1, Some(unsafe { (*self.d_params_buf.get()).as_ref() }), 0);
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
            cmd.wait_until_completed();
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
        let strat_byte_offset = ((self.turn_offset + ti * self.turn_stride) * 4) as u64;
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

        // Top-down through river zone using river[ti*max_river+ri] strategy
        let outcome_idx = ti * self.max_river + ri;
        let strat_byte_offset = ((self.river_offset + outcome_idx * self.river_stride) * 4) as u64;
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

        let outcome_idx = ti * self.max_river + ri;

        // Byte offsets into contiguous strategy/regret/cum buffers
        let strat_byte_off = (self.river_offset + outcome_idx * self.river_stride) * 4;
        let regret_byte_off = (self.river_offset + outcome_idx * self.river_stride) * 4;
        let cum_byte_off = (self.river_offset + outcome_idx * self.river_stride) * 4;
        let cfv_byte_off = ri * nn * nh * 4;

        // Sorted array offsets: table uses tc_card * 52 + rc_card indexing
        let tc_card = self.turn_deck[ti] as usize;
        let rc_card = self.river_decks[tc_card][ri] as usize;
        let sos_byte_off = ((tc_card * 52 + rc_card) * num_opp * nh) * 2; // u16 = 2 bytes
        let sps_byte_off = ((tc_card * 52 + rc_card) * num_opp * nh) * 2;
        let prob_byte_off = ((ti * self.max_river + ri) * nh) * 4; // f32 = 4 bytes

        for level in (0..=self.max_depth).rev() {
            let count = self.river_zone_counts[level];
            if count == 0 { continue; }
            let ln = self.d_river_zone_nodes[level].as_ref().unwrap();

            #[repr(C)]
            #[derive(Clone, Copy)]
            struct BParams {
                level_count: i32, num_outcomes: i32, cfv_batch_stride: i32,
                sorted_opp_stride: i32, num_players: i32, nh: i32,
                traverser: u32, alpha_t: f32, beta_t: f32, gamma_t: f32,
                regret_floor: f32, starting_pot: i32, num_combinations: f32,
                regret_outcome_stride: i32, cum_outcome_stride: i32,
                // ─── Phase 1.A pruning (Option A) ───
                pruning_enabled: i32, pruning_threshold: f32,
                iteration: i32, pruning_stride: i32, board_state: i32,
                // ─── Slice 2 rake (CPU↔Metal parity) ───
                rake_rate: f32, rake_cap: f32,
            }
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
                // Phase 1.A pruning (river: board_state=2 → kernel never prunes river anyway)
                pruning_enabled: if self.pruning_enabled { 1 } else { 0 },
                pruning_threshold: self.pruning_threshold,
                iteration: self.iteration as i32,
                pruning_stride: self.pruning_stride as i32,
                board_state: 2,  // RIVER
                rake_rate: self.rake_rate,
                rake_cap: self.rake_cap,
            };
            self.upload_params(&bp);
            let params_buf = unsafe { (*self.d_params_buf.get()).as_ref() };

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.batched_pipeline);
            enc.set_buffer(0, Some(ln.as_ref()), 0);
            enc.set_buffer(1, Some(params_buf), 0);
            enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_children.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_contributions.as_ref()), 0);
            enc.set_buffer(5, Some(self.d_folded_masks.as_ref()), 0);
            // Strategy/Cumul/Regrets with per-outcome byte offset
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
            // Step 5 chokepoint instrumentation marker (river dispatch).
            enc.set_buffer(20, Some(self.d_rake_marker.as_ref()), 0);

            let grid_size = metal::MTLSize { width: count as u64, height: 1, depth: 1 };
            let tg_size = metal::MTLSize { width: 1, height: 1, depth: 1 };
            enc.dispatch_thread_groups(grid_size, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    pub fn bottom_up_turn(
        &self, ctx: &MetalContext,
        ti: usize, traverser: u32, params: &DcfrParams,
    ) {
        let nh = self.nh;
        let np = self.num_players as usize;
        let num_opp = np - 1;

        let strat_byte_off = (self.turn_offset + ti * self.turn_stride) * 4;
        let regret_byte_off = (self.turn_offset + ti * self.turn_stride) * 4;
        let cum_byte_off = (self.turn_offset + ti * self.turn_stride) * 4;
        let cfv_byte_off = ti * self.nn * self.nh * 4;

        // Turn zone uses sorted arrays for this turn card (indexed by raw card value)
        let tc_card = self.turn_deck[ti] as usize;
        let sos_byte_off = (tc_card * num_opp * nh) * 2;
        let sps_byte_off = (tc_card * num_opp * nh) * 2;

        for level in (0..=self.max_depth).rev() {
            let count = self.turn_zone_counts[level];
            if count == 0 { continue; }
            let ln = self.d_turn_zone_nodes[level].as_ref().unwrap();

            #[repr(C)]
            #[derive(Clone, Copy)]
            struct BParams {
                level_count: i32, num_outcomes: i32, cfv_batch_stride: i32,
                sorted_opp_stride: i32, num_players: i32, nh: i32,
                traverser: u32, alpha_t: f32, beta_t: f32, gamma_t: f32,
                regret_floor: f32, starting_pot: i32, num_combinations: f32,
                regret_outcome_stride: i32, cum_outcome_stride: i32,
                // ─── Phase 1.A pruning (Option A) ───
                pruning_enabled: i32, pruning_threshold: f32,
                iteration: i32, pruning_stride: i32, board_state: i32,
                // ─── Slice 2 rake (CPU↔Metal parity) ───
                rake_rate: f32, rake_cap: f32,
            }
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
                // Phase 1.A pruning (turn: board_state=1)
                pruning_enabled: if self.pruning_enabled { 1 } else { 0 },
                pruning_threshold: self.pruning_threshold,
                iteration: self.iteration as i32,
                pruning_stride: self.pruning_stride as i32,
                board_state: 1,  // TURN
                rake_rate: self.rake_rate,
                rake_cap: self.rake_cap,
            };
            self.upload_params(&bp);
            let params_buf = unsafe { (*self.d_params_buf.get()).as_ref() };

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.batched_pipeline);
            enc.set_buffer(0, Some(ln.as_ref()), 0);
            enc.set_buffer(1, Some(params_buf), 0);
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
            // debug_out at buffer(19) — bind to debug buffer for completeness
            // (turn dispatch did not previously bind this, but adding rake_marker
            // at buffer(20) means we need 19 bound too for Metal to validate).
            enc.set_buffer(19, Some(self.d_debug_out.as_ref()), 0);
            // Step 5 chokepoint instrumentation marker (turn dispatch).
            enc.set_buffer(20, Some(self.d_rake_marker.as_ref()), 0);

            let grid_size = metal::MTLSize { width: count as u64, height: 1, depth: 1 };
            let tg_size = metal::MTLSize { width: 1, height: 1, depth: 1 };
            enc.dispatch_thread_groups(grid_size, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    pub fn bottom_up_flop(&self, ctx: &MetalContext, traverser: u32, params: &DcfrParams) {
        // Flop zone uses the single-outcome bottom_up kernel
        let nh = self.nh;
        let np = self.num_players as usize;

        for level in (0..=self.max_depth).rev() {
            let count = self.flop_zone_counts[level];
            if count == 0 { continue; }
            let ln = self.d_flop_zone_nodes[level].as_ref().unwrap();

            #[repr(C)]
            #[repr(C)]

            #[derive(Clone, Copy)]
            struct BuParams {
                level_count: i32, num_players: i32, nh: i32,
                traverser: u32, alpha_t: f32, beta_t: f32, gamma_t: f32,
                regret_floor: f32, starting_pot: i32, num_combinations: f32,
                // ─── Slice 2 rake (CPU↔Metal parity) ───
                rake_rate: f32, rake_cap: f32,
            }
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
            };
            self.upload_params(&bp);

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.bottom_up_pipeline);
            enc.set_buffer(0, Some(ln.as_ref()), 0);
            enc.set_buffer(1, Some(unsafe { (*self.d_params_buf.get()).as_ref() }), 0);
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
            // Step 5 chokepoint instrumentation marker (flop dispatch).
            enc.set_buffer(18, Some(self.d_rake_marker.as_ref()), 0);

            let grid_size = metal::MTLSize { width: count as u64, height: 1, depth: 1 };
            let tg_size = metal::MTLSize { width: 1, height: 1, depth: 1 };
            enc.dispatch_thread_groups(grid_size, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
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
            #[repr(C)]
            #[repr(C)]

            #[derive(Clone, Copy)]
            struct Params { num_chance_children: i32, nh: i32, outcome: i32 }
            let p = Params {
                num_chance_children: cc as i32,
                nh: nh as i32,
                outcome: ri as i32, // river outcome index for chance_prob lookup
            };
            self.upload_params(&p);
            let params_buf = unsafe { (*self.d_params_buf.get()).as_ref() };

            let total = cc * nh;
            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.chance_accum_pipeline);
            // cfv_accum = river_accum
            enc.set_buffer(0, Some(self.d_river_accum.as_ref()), 0);
            // cfv = river_cfv_batch offset to this ri's CFVs
            let cfv_byte_off = (ri * nn * nh) * 4;
            enc.set_buffer(1, Some(self.d_river_cfv_batch.as_ref()), cfv_byte_off as u64);
            // chance_prob = river_chance_prob offset to this turn card's probs
            let prob_byte_off = (ti * self.max_river * nh) * 4;
            enc.set_buffer(2, Some(self.d_river_chance_prob.as_ref()), prob_byte_off as u64);
            enc.set_buffer(3, Some(self.d_river_chance_children.as_ref()), 0);
            enc.set_buffer(4, Some(params_buf), 0); // num_chance_children
            enc.set_buffer(5, Some(params_buf), 4); // nh (same struct, offset 4)
            enc.set_buffer(6, Some(params_buf), 8); // outcome (same struct, offset 8)

            let (grid, tg) = ctx.dispatch_1d(total, 256);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    pub fn chance_finalize_river(&self, ctx: &MetalContext, ti: usize) {
        // Copy river_accum into turn CFV batch at river chance children positions.
        // CPU: turn_cfv[child*nh+h] = river_cfv_accum[child*nh+h]
        // Uses vcfr_chance_finalize: cfv[child*nh+h] = cfv_accum[child*nh+h]
        let nh = self.nh;
        let cc = self.river_cc_count;
        let total = cc * nh;

        #[repr(C)]
        #[repr(C)]

        #[derive(Clone, Copy)]
        struct Params { num_chance_children: i32, nh_val: i32 }
        let p = Params { num_chance_children: cc as i32, nh_val: nh as i32 };
        self.upload_params(&p);
        let params_buf = unsafe { (*self.d_params_buf.get()).as_ref() };

        // The turn CFV batch is indexed by turn card: ti * nn * nh
        let turn_cfv_byte_off = (ti * self.nn * nh) * 4;

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.chance_final_pipeline);
        // cfv = turn_cfv_batch offset to this ti
        enc.set_buffer(0, Some(self.d_turn_cfv_batch.as_ref()), turn_cfv_byte_off as u64);
        // cfv_accum = river_accum
        enc.set_buffer(1, Some(self.d_river_accum.as_ref()), 0);
        enc.set_buffer(2, Some(self.d_river_chance_children.as_ref()), 0);
        enc.set_buffer(3, Some(params_buf), 0); // num_chance_children at offset 0
        enc.set_buffer(4, Some(params_buf), 4); // nh_val at offset 4

        let (grid, tg) = ctx.dispatch_1d(total, 256);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    pub fn chance_accumulate_turn(&self, ctx: &MetalContext) {
        // CPU: for each ti: flop_cfv[child*nh+h] += cp(ti,h) * turn_cfv[child*nh+h]
        // Metal: call vcfr_chance_accumulate once per turn outcome ti.
        let nh = self.nh;
        let cc = self.turn_cc_count;
        let nn = self.nn;
        let n_turn = self.n_turn;

        for ti in 0..n_turn {
            #[repr(C)]
            #[repr(C)]

            #[derive(Clone, Copy)]
            struct Params { num_chance_children: i32, nh: i32, outcome: i32 }
            let p = Params {
                num_chance_children: cc as i32,
                nh: nh as i32,
                outcome: ti as i32,
            };
            self.upload_params(&p);
            let params_buf = unsafe { (*self.d_params_buf.get()).as_ref() };

            let total = cc * nh;
            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.chance_accum_pipeline);
            // cfv_accum = d_cfv (main CFV buffer, accumulated across turn cards)
            enc.set_buffer(0, Some(self.d_cfv.as_ref()), 0);
            // cfv = turn_cfv_batch offset to this ti's CFVs
            let cfv_byte_off = (ti * nn * nh) * 4;
            enc.set_buffer(1, Some(self.d_turn_cfv_batch.as_ref()), cfv_byte_off as u64);
            // chance_prob = turn_chance_prob
            enc.set_buffer(2, Some(self.d_turn_chance_prob.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_turn_chance_children.as_ref()), 0);
            enc.set_buffer(4, Some(params_buf), 0); // num_chance_children
            enc.set_buffer(5, Some(params_buf), 4); // nh (same struct, offset 4)
            enc.set_buffer(6, Some(params_buf), 8); // outcome (same struct, offset 8)

            let (grid, tg) = ctx.dispatch_1d(total, 256);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
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
        #[repr(C)]
        #[repr(C)]

        #[derive(Clone, Copy)]
        struct Params { count: i32 }
        let p = Params { count: len as i32 };
        self.upload_params(&p);

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
        enc.set_buffer(1, Some(unsafe { (*self.d_params_buf.get()).as_ref() }), 0);
        let (grid, tg) = ctx.dispatch_1d(len, 256);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
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

        #[repr(C)]
        #[repr(C)]

        #[derive(Clone, Copy)]
        struct Params { count: i32, np_nh: i32 }
        let p = Params { count: count as i32, np_nh: np_nh as i32 };
        self.upload_params(&p);

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.seed_reach_pipeline);
        enc.set_buffer(0, Some(dst.as_ref()), 0);
        enc.set_buffer(1, Some(src.as_ref()), 0);
        enc.set_buffer(2, Some(chance_children.as_ref()), 0);
        enc.set_buffer(3, Some(unsafe { (*self.d_params_buf.get()).as_ref() }), 0);
        enc.set_buffer(4, Some(unsafe { (*self.d_params_buf.get()).as_ref() }), 4); // np_nh at offset 4

        let (grid, tg) = ctx.dispatch_1d(total, 256);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
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
        self.upload_params(&p);

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.top_down_pipeline);
        enc.set_buffer(0, Some(level_nodes.as_ref()), 0);
        enc.set_buffer(1, Some(unsafe { (*self.d_params_buf.get()).as_ref() }), 0);
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
        cmd.wait_until_completed();
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

        // River zone: n_turn * max_river outcomes
        self.compute_strategies_batched(
            ctx, &self.d_river_decision_ids, &self.d_river_infoset_offsets,
            self.river_infosets, self.n_turn * self.max_river,
            self.river_offset, self.river_stride,
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
        cmd.wait_until_completed();

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
        cmd.wait_until_completed();

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
        cmd.wait_until_completed();

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
