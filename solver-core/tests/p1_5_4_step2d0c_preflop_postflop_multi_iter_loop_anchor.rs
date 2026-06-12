// Step 2.D.0c: multi-iter CPU preflop+postflop loop-composition oracle.
//
// THE GAP THIS CLOSES (banked from 2.D scaffolding pass):
//
// The existing preflop oracles (slice A.1 strategy, A.2 reach, P2.5a chance
// composition, P5a aggregate) are all SINGLE-STAGE anchors. The closest
// multi-iter test is slice_a3b_single_iter_engine.rs (one iter only).
// There is NO existing CPU oracle that anchors the FULL preflop+postflop
// MULTI-ITER loop against rules-derived ground truth.
//
// The 2.A.2 sweep-vs-brute compounder proved this matters: at iter 1 the
// divergence was 1 ULP per terminal CFV entry (looks like f32 noise);
// compounded through CFR feedback it surfaced as orders-of-magnitude
// divergence by iter 10. Per-stage gates AT IT 1 STRUCTURALLY CANNOT
// catch what compounds through the loop.
//
// What this test anchors:
// - Production PreflopVectorCfr::run_one_iteration vs independent textbook
//   DCFR reference.
// - 10 iters under realistic asymmetric inputs.
// - Comparison after EACH iter on regrets, cum_strategy, strategy.
//
// What this test SHARES with production (and that the per-stage anchors
// already validate):
// - PreflopChanceTable construction (anchored by P5a, P5b primitives).
// - expand_reach_class_to_combo (P5b).
// - reduce_cfv_combo_to_class (P5b).
// - aggregate_preflop_chance (P5a, f64-anchored).
// - The asymmetric stub leaf-value function (DEFINED HERE, shared by both
//   sides — the stub is the leaf, not the loop).
//
// What this test does NOT share with production (the load-bearing
// independence — this is the part being validated):
// - Regret-matching: independent textbook formula.
// - Reach propagation: independent recursive top-down walk.
// - DCFR bottom-up update: independent textbook formula.
//
// If a compounder hides in production's loop wiring (wrong order, wrong
// sign, wrong accumulator, wrong regret denominator), the production state
// will diverge from the reference state — at iter 1 maybe by 1 ULP, by
// iter 10 by orders of magnitude. The bug is caught.
//
// Once this passes, the CPU side's multi-iter loop is rules-anchored, and
// 2.D.5 (CPU↔GPU multi-iter parity at production cell) becomes a
// well-defined REPLICATION gate, not an implicit "we hope CPU is right"
// gate.

use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::Card;
use solver_core::solver::flop_start_vector_cfr::DcfrParams;
use solver_core::solver::postflop_oracle::{ClosureOracle, PostflopValueOracle};
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::solver::preflop_start_game::{
    aggregate_preflop_chance, expand_reach_class_to_combo, flop_combo_layout,
    reduce_cfv_combo_to_class, PreflopChanceTable,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

// Same epsilon as production preflop_cfr.rs REGRET_MATCH_EPS. Keep in sync
// if production changes it (the reference must match production's
// uniform-fallback threshold to be a valid reference).
const REGRET_MATCH_EPS: f32 = 1e-5;

const UNUSED: usize = usize::MAX;

// ─────────────────────────────────────────────────────────────────────────
// Stub leaf value function: asymmetric, deterministic, reach-independent.
// ─────────────────────────────────────────────────────────────────────────
//
// The leaf is what each canonical flop's postflop solve "returns" at the
// root for the traverser. For loop-composition testing we want the leaf
// to be:
//   (a) deterministic — same input → same output, bit-exact (so reference
//       and production see the same leaf);
//   (b) asymmetric per (canonical, traverser, combo_index) — so the
//       composition logic propagates a non-trivial signal;
//   (c) reach-INDEPENDENT — leaf values don't depend on combo_ranges, so
//       any divergence between production and reference comes purely from
//       loop composition, not leaf-value feedback;
//   (d) varied magnitudes — not all in a tight range, so any partial
//       cancellation bugs surface.

fn stub_leaf(canonical: [Card; 3], _combo_ranges: &[Vec<f32>], traverser: u8) -> Vec<f32> {
    let layout = flop_combo_layout(canonical);
    // Hash the canonical + traverser + combo into a small deterministic
    // signal in [-1, +1].
    let canon_seed: u64 = (canonical[0] as u64) << 16
        | (canonical[1] as u64) << 8
        | (canonical[2] as u64);
    layout.iter().enumerate().map(|(li, &(c1, c2))| {
        let combo_seed: u64 = ((c1 as u64) << 8) | (c2 as u64);
        let trav_seed: u64 = traverser as u64;
        let mix: u64 = canon_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ combo_seed.wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ trav_seed.wrapping_mul(0x94D0_49BB_1331_11EB)
            ^ (li as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
        // Project to [-1, 1] via the high 24 bits.
        let bits = ((mix >> 32) & 0xFFFFFF) as i64 - (1 << 23);
        (bits as f32) / ((1 << 23) as f32)
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// Tree: minimal HU preflop.
// ─────────────────────────────────────────────────────────────────────────

fn build_minimal_hu_preflop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        // HU blinds: SB=1, BB=2, pot=3 after antes.
        starting_pot: 3,
        starting_stacks: vec![20, 19],
        initial_contributions: vec![1, 2],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            // Minimal: one open size + allin.
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&cfg).expect("preflop tree builds")
}

// ─────────────────────────────────────────────────────────────────────────
// Reference state: independently computes strategy / reach / DCFR update.
// ─────────────────────────────────────────────────────────────────────────

struct RefState {
    num_players: u8,
    nn: usize,
    local_offset: Vec<usize>,
    infoset_count: usize,
    // Same storage convention as PreflopVectorCfr:
    //   [infoset_count * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES]
    //   indexed as [local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES + a * NUM_PREFLOP_CLASSES + c]
    strategy: Vec<f32>,
    regrets: Vec<f32>,
    cum_strategy: Vec<f32>,
    iteration: u32,
}

impl RefState {
    fn new(tree: &FlatTree) -> Self {
        use solver_core::tree::action::BoardState;
        let nn = tree.num_nodes();
        let mut local_offset = vec![UNUSED; nn];
        let mut infoset_count = 0usize;
        for idx in 0..nn {
            let node = &tree.nodes[idx];
            if node.board_state != BoardState::Preflop as u8 { continue; }
            if !node.is_player() { continue; }
            local_offset[idx] = infoset_count;
            infoset_count += 1;
        }
        let stride = MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
        let total = infoset_count * stride;
        let mut strategy = vec![0.0f32; total];
        // Uniform init per actual na.
        for idx in 0..nn {
            let local = local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            if na == 0 { continue; }
            let off = local * stride;
            let uniform = 1.0_f32 / na as f32;
            for a in 0..na {
                for c in 0..NUM_PREFLOP_CLASSES {
                    strategy[off + a * NUM_PREFLOP_CLASSES + c] = uniform;
                }
            }
        }
        Self {
            num_players: tree.num_players,
            nn,
            local_offset,
            infoset_count,
            strategy,
            regrets: vec![0.0f32; total],
            cum_strategy: vec![0.0f32; total],
            iteration: 0,
        }
    }

    // Textbook regret-matching per (infoset, class).
    fn ref_compute_strategy(&mut self, tree: &FlatTree) {
        for idx in 0..self.nn {
            let local = self.local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            if na == 0 { continue; }
            let off = local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
            for c in 0..NUM_PREFLOP_CLASSES {
                // Sum of positive regrets (above eps) for this class.
                let mut sum_pos = 0.0_f32;
                for a in 0..na {
                    let r = self.regrets[off + a * NUM_PREFLOP_CLASSES + c];
                    if r > REGRET_MATCH_EPS { sum_pos += r; }
                }
                if sum_pos <= 0.0 {
                    // Uniform fallback.
                    let uniform = 1.0_f32 / na as f32;
                    for a in 0..na {
                        self.strategy[off + a * NUM_PREFLOP_CLASSES + c] = uniform;
                    }
                } else {
                    for a in 0..na {
                        let r = self.regrets[off + a * NUM_PREFLOP_CLASSES + c];
                        let p = if r > REGRET_MATCH_EPS { r / sum_pos } else { 0.0 };
                        self.strategy[off + a * NUM_PREFLOP_CLASSES + c] = p;
                    }
                }
            }
        }
    }

    // Independent recursive top-down reach walk.
    fn ref_compute_reach(&self, tree: &FlatTree) -> Vec<Vec<f32>> {
        let np = self.num_players as usize;
        let n_classes = NUM_PREFLOP_CLASSES;
        let mut reach: Vec<Vec<f32>> = (0..np)
            .map(|_| vec![0.0f32; self.nn * n_classes]).collect();
        // Root: reach = 1.0 per class per player.
        for p in 0..np {
            for c in 0..n_classes {
                reach[p][c] = 1.0;
            }
        }
        self.ref_propagate(tree, 0, &mut reach);
        reach
    }

    fn ref_propagate(&self, tree: &FlatTree, node_idx: usize, reach: &mut [Vec<f32>]) {
        use solver_core::tree::action::BoardState;
        let node = &tree.nodes[node_idx];
        if node.board_state != BoardState::Preflop as u8 { return; }
        let children = tree.node_children(node_idx).to_vec();
        if children.is_empty() { return; }
        let np = self.num_players as usize;
        let n_classes = NUM_PREFLOP_CLASSES;
        let parent_base = node_idx * n_classes;
        if node.is_player() {
            let pid = node.player_id as usize;
            let local = self.local_offset[node_idx];
            assert!(local != UNUSED);
            let na = node.num_children as usize;
            let off = local * MAX_NA_PREFLOP * n_classes;
            for (a, &child_u32) in children.iter().enumerate() {
                let child = child_u32 as usize;
                let child_base = child * n_classes;
                for c in 0..n_classes {
                    reach[pid][child_base + c] =
                        reach[pid][parent_base + c]
                        * self.strategy[off + a * n_classes + c];
                }
                for p in 0..np {
                    if p == pid { continue; }
                    for c in 0..n_classes {
                        reach[p][child_base + c] = reach[p][parent_base + c];
                    }
                }
                // No-op for c that exceeds na is not needed since strategy
                // is zero for padded actions; reach copying still works.
                self.ref_propagate(tree, child, reach);
            }
        } else {
            // Chance or terminal pass-through.
            for &child_u32 in &children {
                let child = child_u32 as usize;
                let child_base = child * n_classes;
                for p in 0..np {
                    for c in 0..n_classes {
                        reach[p][child_base + c] = reach[p][parent_base + c];
                    }
                }
                self.ref_propagate(tree, child, reach);
            }
        }
    }

    // Independent textbook DCFR bottom-up update for traverser. Mirrors
    // PreflopVectorCfr::bottom_up_recursive but uses RefState's own
    // strategy/regrets/cum_strategy buffers and recomputes everything
    // independently.
    fn ref_bottom_up(
        &mut self,
        tree: &FlatTree,
        node_idx: usize,
        traverser: u8,
        is_chance_leaf: &impl Fn(usize) -> bool,
        reach: &[Vec<f32>],
        chance_cfvs: &[Vec<f32>],   // indexed by node_idx; chance nodes filled, others empty
        terminal_value_fn: &impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
        cfv: &mut [Vec<f32>],
        params: &DcfrParams,
    ) {
        if is_chance_leaf(node_idx) {
            cfv[node_idx] = chance_cfvs[node_idx].clone();
            return;
        }
        let node = &tree.nodes[node_idx];
        let n_classes = NUM_PREFLOP_CLASSES;
        if node.is_terminal() {
            let np = self.num_players as usize;
            let mut reach_at_terminal: Vec<Vec<f32>> = Vec::with_capacity(np);
            let base = node_idx * n_classes;
            for p in 0..np {
                reach_at_terminal.push(reach[p][base..base + n_classes].to_vec());
            }
            cfv[node_idx] = terminal_value_fn(node_idx, traverser, &reach_at_terminal);
            return;
        }
        let children: Vec<u32> = tree.node_children(node_idx).to_vec();
        for &child_u32 in &children {
            self.ref_bottom_up(
                tree, child_u32 as usize, traverser, is_chance_leaf,
                reach, chance_cfvs, terminal_value_fn, cfv, params,
            );
        }
        let local = self.local_offset[node_idx];
        let na = node.num_children as usize;
        let off = local * MAX_NA_PREFLOP * n_classes;
        let mut cfv_avg = vec![0.0f32; n_classes];
        if node.player_id == traverser {
            for (a, &child_u32) in children.iter().enumerate() {
                let child = child_u32 as usize;
                let s_base = off + a * n_classes;
                for c in 0..n_classes {
                    cfv_avg[c] += self.strategy[s_base + c] * cfv[child][c];
                }
            }
            for (a, &child_u32) in children.iter().enumerate() {
                let child = child_u32 as usize;
                for c in 0..n_classes {
                    let inst_regret = cfv[child][c] - cfv_avg[c];
                    let ridx = off + a * n_classes + c;
                    let old_r = self.regrets[ridx];
                    let coef = if old_r >= 0.0 { params.alpha_t() } else { params.beta_t() };
                    self.regrets[ridx] = coef * old_r + inst_regret;
                    self.cum_strategy[ridx] = params.gamma_t() * self.cum_strategy[ridx]
                        + self.strategy[ridx];
                }
            }
        } else {
            for &child_u32 in &children {
                let child = child_u32 as usize;
                for c in 0..n_classes {
                    cfv_avg[c] += cfv[child][c];
                }
            }
        }
        cfv[node_idx] = cfv_avg;
    }

    fn ref_run_one_iteration(
        &mut self,
        tree: &FlatTree,
        table: &PreflopChanceTable,
        oracle: &mut impl PostflopValueOracle,
        chance_node_indices: &[usize],
        terminal_value_fn: impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
    ) {
        let nn = self.nn;
        let n_classes = NUM_PREFLOP_CLASSES;
        let np = self.num_players as usize;

        self.ref_compute_strategy(tree);
        let reach = self.ref_compute_reach(tree);
        let params = DcfrParams::new(self.iteration);
        oracle.begin_preflop_iter(self.iteration);

        for t in 0..np as u8 {
            let mut cfv: Vec<Vec<f32>> = vec![vec![0.0f32; n_classes]; nn];
            // Per-chance-node CFV via the per-stage-anchored primitives
            // (P5b expand/reduce, P5a aggregate). These are SHARED with
            // production but already anchored by their own structural
            // independence tests; the loop-composition piece below is
            // the independent part.
            for &chance_idx in chance_node_indices {
                let chance_base = chance_idx * n_classes;
                let n_canon = table.num_canonical_flops();
                let mut per_canonical_v_class: Vec<Vec<f32>> = Vec::with_capacity(n_canon);
                for canonical_idx in 0..n_canon {
                    let f_canon = table.canonical_flops[canonical_idx];
                    let layout = flop_combo_layout(f_canon);
                    let mut combo_reaches: Vec<Vec<f32>> = Vec::with_capacity(np);
                    for p in 0..np {
                        let class_reach = &reach[p][chance_base..chance_base + n_classes];
                        combo_reaches.push(
                            expand_reach_class_to_combo(f_canon, class_reach, &layout)
                        );
                    }
                    let v_combo = oracle.flop_root_cfv(f_canon, &combo_reaches, t);
                    assert_eq!(v_combo.len(), layout.len());
                    let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
                    per_canonical_v_class.push(v_class);
                }
                cfv[chance_idx] = aggregate_preflop_chance(table, &per_canonical_v_class);
            }
            let is_chance_leaf = |idx: usize| chance_node_indices.binary_search(&idx).is_ok();
            let chance_cfvs_snapshot = cfv.clone();
            self.ref_bottom_up(
                tree, 0, t, &is_chance_leaf, &reach, &chance_cfvs_snapshot,
                &terminal_value_fn, &mut cfv, &params,
            );
        }
        oracle.end_preflop_iter(self.iteration);
        self.iteration += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The anchor test.
// ─────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "Step 2.D.0c: multi-iter preflop+postflop loop composition oracle (slow: full 1755-canonical chance over N iters)"]
fn preflop_postflop_multi_iter_loop_composition_anchor() {
    let tree = build_minimal_hu_preflop_tree();
    let np = tree.num_players;
    eprintln!("\n=== Step 2.D.0c: preflop+postflop multi-iter loop-composition anchor ===");
    eprintln!("Tree: {} nodes, np={}", tree.num_nodes(), np);

    // Realistic asymmetric class weights — same shape pattern as 2.A.2
    // strata harnesses (P0 tight value-heavy, P1 wider linear).
    let mut class_weights: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();
    for k in 0..NUM_PREFLOP_CLASSES {
        let strength_frac = k as f32 / NUM_PREFLOP_CLASSES as f32;
        let p0_w = (strength_frac - 0.3).max(0.05) * 1.5;
        let p0_w = p0_w.min(1.0);
        let p1_w = 0.6 + 0.4 * strength_frac;
        class_weights[0][k] = p0_w;
        class_weights[1][k] = p1_w;
    }
    eprintln!("Class weights: P0 sigmoid-tight, P1 wider-linear (realistic asymmetric)");
    eprintln!("Building PreflopChanceTable (1755 canonical orbits; ~few seconds)...");

    let table = PreflopChanceTable::new(np, class_weights);
    let n_canon = table.num_canonical_flops();
    assert_eq!(n_canon, 1755, "expected 1755 canonical flops");

    let mut prod = PreflopVectorCfr::new(&tree);
    let mut refs = RefState::new(&tree);

    // Preflop chance-node indices: a CHANCE node whose PARENT is in the
    // Preflop zone. The chance node itself transitions to Flop, so its
    // own board_state is Flop, not Preflop. Use the production helper for
    // identification (sharing identification logic is fine; it's the LOOP
    // composition we're validating independently).
    let mut chance_node_indices: Vec<usize> = prod.preflop_chance_node_indices(&tree);
    chance_node_indices.sort();
    eprintln!("Preflop chance node count: {}", chance_node_indices.len());
    assert_eq!(prod.infoset_count, refs.infoset_count,
        "production and reference disagree on infoset_count");

    let n_iters = 10u32;
    // Stub leaf for terminals (preflop terminals like preflop-fold endpoints).
    // Same deterministic shape as stub_leaf but per-class instead of
    // per-combo (terminals are evaluated per-class in the bottom-up walk).
    let terminal_value_fn = |term_idx: usize, traverser: u8, _reach_at_term: &[Vec<f32>]| -> Vec<f32> {
        (0..NUM_PREFLOP_CLASSES).map(|c| {
            let seed: u64 = (term_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (traverser as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                ^ (c as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            let bits = ((seed >> 32) & 0xFFFFFF) as i64 - (1 << 23);
            (bits as f32) / ((1 << 23) as f32)
        }).collect()
    };

    let mut max_strat_abs_overall = 0.0f32;
    let mut max_regret_abs_overall = 0.0f32;
    let mut max_cum_abs_overall = 0.0f32;
    let mut growth_per_iter: Vec<f32> = Vec::new();

    for iter in 0..n_iters {
        eprintln!("\n--- iter {} ---", iter);
        // Run production. Use a fresh ClosureOracle wrapping stub_leaf.
        let mut prod_oracle = ClosureOracle::new(stub_leaf);
        prod.run_one_iteration(&tree, &table, &mut prod_oracle, &terminal_value_fn);

        // Run reference.
        let mut ref_oracle = ClosureOracle::new(stub_leaf);
        refs.ref_run_one_iteration(
            &tree, &table, &mut ref_oracle,
            &chance_node_indices, &terminal_value_fn,
        );

        // Compare strategy, regrets, cum_strategy after this iter.
        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        let s_diff = max_abs(&prod.strategy, &refs.strategy);
        let r_diff = max_abs(&prod.regrets, &refs.regrets);
        let c_diff = max_abs(&prod.cum_strategy, &refs.cum_strategy);
        eprintln!("  strategy   max_abs = {:.6e}", s_diff);
        eprintln!("  regrets    max_abs = {:.6e}", r_diff);
        eprintln!("  cum_strat  max_abs = {:.6e}", c_diff);
        if r_diff > max_regret_abs_overall {
            if max_regret_abs_overall > 0.0 {
                growth_per_iter.push(r_diff / max_regret_abs_overall);
            }
        }
        max_strat_abs_overall = max_strat_abs_overall.max(s_diff);
        max_regret_abs_overall = max_regret_abs_overall.max(r_diff);
        max_cum_abs_overall = max_cum_abs_overall.max(c_diff);
    }

    eprintln!("\nOver {} iters:", n_iters);
    eprintln!("  max strategy diff:    {:.6e}", max_strat_abs_overall);
    eprintln!("  max regrets diff:     {:.6e}", max_regret_abs_overall);
    eprintln!("  max cum_strategy diff: {:.6e}", max_cum_abs_overall);
    if !growth_per_iter.is_empty() {
        let avg_growth: f32 = growth_per_iter.iter().sum::<f32>() / growth_per_iter.len() as f32;
        eprintln!("  avg regret-growth-per-iter ratio: {:.3} (compounding fingerprint > 1.5x is suspect)",
            avg_growth);
    }

    // f32 floor tolerance. Production and reference share the per-stage
    // anchored primitives (expand/reduce/aggregate at f64) and the stub
    // leaf, so the only float-ordering source is the loop composition
    // pieces (regret-matching, reach, bottom-up), which use the same
    // sequential operations on both sides.
    let tol = 1e-4_f32;
    assert!(max_strat_abs_overall < tol,
        "LOOP-COMPOSITION DIVERGENCE: strategy diverges by {:.3e} > {}. \
         CPU preflop loop-composition has a bug that single-iter gates missed.",
        max_strat_abs_overall, tol);
    assert!(max_regret_abs_overall < tol,
        "LOOP-COMPOSITION DIVERGENCE: regrets diverge by {:.3e} > {}. \
         CPU preflop loop-composition has a compounding bug invisible per-stage.",
        max_regret_abs_overall, tol);
    assert!(max_cum_abs_overall < tol,
        "LOOP-COMPOSITION DIVERGENCE: cum_strategy diverges by {:.3e} > {}.",
        max_cum_abs_overall, tol);

    eprintln!("\n=== STEP 2.D.0c PASS ===");
    eprintln!("CPU preflop+postflop multi-iter loop is anchored against rules.");
    eprintln!("2.D.1-2.D.5 GPU port can now proceed as pure replication against this loop.");
    eprintln!("Per #84: if 2.D.5 parity breaks, run THIS test first to disambiguate");
    eprintln!("replication drift vs loop-composition regression on CPU.");
}
