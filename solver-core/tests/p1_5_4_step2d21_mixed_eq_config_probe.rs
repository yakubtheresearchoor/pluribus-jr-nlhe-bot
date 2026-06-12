// Step 2.D.21 (#113 config-validity probe): determine WHAT makes the
// preflop equilibrium MIXED vs DIRAC.
//
// Background: #112 (variance-vs-iter test) produced a misleading "favorable"
// verdict because the test config converged to PURE DIRAC strategies (mixed
// slots = 0 at preflop_iters=30). Dirac is robust to CFV perturbation, so
// any K reaches the same fixed point bit-exactly — that's not warm-start
// favorability, that's degenerate convergence.
//
// User hypothesis: the dirac was driven by random terminal_value_fn (no
// hand-strength structure → one action dominates per class). Likely scale
// alone won't fix it; the fix is realistic terminal values.
//
// This probe varies ONE knob at a time on the SAME small config to isolate:
//   (a) Original random terminal_value_fn + asymmetric class weights (baseline = #112 config)
//   (b) Production terminal_value_fn (make_production_terminal_value_fn_hu)
//       with realistic blocking + fold chip-delta
//   (c) Symmetric class weights (vs the asymmetric "p0 narrow, p1 wide")
//
// PASS = a config that produces meaningfully MIXED strategies (>5% mixed
// slots) at preflop_iters=30. Without this, #113's variance measurement
// can't be done at all.

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::flop_start_vector_cfr::DcfrParams;
use solver_core::solver::preflop_cfr::{
    make_production_terminal_value_fn_hu, PreflopVectorCfr,
};
use solver_core::solver::preflop_start_game::{
    flop_combo_layout, reduce_cfv_combo_to_class, PreflopChanceTable,
};
use solver_core::solver::preflop_terminal::build_class_blocking_matrix;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const PREFLOP_ITERS: u32 = 30;
const CANONICAL_SUBSET_SIZE: usize = 10;

fn build_minimal_hu_preflop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![20, 19],
        initial_contributions: vec![1, 2],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
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

/// Strategy stat: counts of zero / one / mixed slots.
fn strategy_stats(strategy: &[f32]) -> (usize, usize, usize) {
    let n_zero = strategy.iter().filter(|&&v| v == 0.0).count();
    let n_one = strategy.iter().filter(|&&v| v == 1.0).count();
    let n_mixed = strategy.iter().filter(|&&v| v != 0.0 && v != 1.0).count();
    (n_zero, n_one, n_mixed)
}

/// Helper: dummy postflop CFV — uniform per class. Mimics #112's setup
/// without depending on the per-canonical postflop solver (we're testing
/// PREFLOP dynamics in isolation).
///
/// For the probe, we don't need a real per-canonical postflop solver —
/// we just need the preflop loop to run. Use a constant 0 per-canonical
/// CFV; the preflop convergence is driven entirely by terminal_value_fn.
fn run_preflop_only(
    tree: &FlatTree,
    table: &PreflopChanceTable,
    canonical_subset: &[usize],
    terminal_value_fn: impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32> + Copy,
    preflop_iters: u32,
) -> Vec<f32> {
    let np = tree.num_players as usize;
    let n_classes = NUM_PREFLOP_CLASSES;
    let nn = tree.num_nodes();
    let mut prod = PreflopVectorCfr::new(tree);
    let chance_node_indices = prod.preflop_chance_node_indices(tree);

    for iter in 0..preflop_iters {
        prod.compute_preflop_strategy(tree);
        let reach = prod.compute_preflop_reach(tree, None);
        let params = DcfrParams::new(iter);

        for t in 0..np as u8 {
            let mut cfv: Vec<Vec<f32>> = vec![vec![0.0f32; n_classes]; nn];
            for &chance_idx in &chance_node_indices {
                let chance_base = chance_idx * n_classes;
                let mut per_canon_v: Vec<Vec<f32>> = Vec::with_capacity(canonical_subset.len());
                for &canonical_idx in canonical_subset {
                    let f_canon = table.canonical_flops[canonical_idx];
                    let layout = flop_combo_layout(f_canon);
                    let _layout_len = layout.len();
                    // Dummy postflop CFV: constant 0 per combo. Preflop
                    // dynamics are driven by terminal_value_fn alone.
                    let v_combo: Vec<f32> = vec![0.0f32; layout.len()];
                    let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
                    let _ = chance_base;
                    per_canon_v.push(v_class);
                }
                cfv[chance_idx] = solver_core::solver::preflop_start_game::aggregate_preflop_chance_subset(
                    table, canonical_subset, &per_canon_v);
            }
            prod.bottom_up_preflop_for_traverser(
                tree, t, &chance_node_indices, &reach,
                |term_idx, tr, r| terminal_value_fn(term_idx, tr, r),
                &mut cfv, &params,
            );
        }
    }
    prod.strategy.clone()
}

#[test]
#[ignore = "Step 2.D.21: config-validity probe — find what produces MIXED equilibria"]
fn step2d21_mixed_eq_config_probe() {
    let tree = build_minimal_hu_preflop_tree();
    eprintln!("\n=== Step 2.D.21: config-validity probe for mixed equilibria ===");
    eprintln!("Tree: {} nodes, {} preflop iters, {} canonicals", tree.num_nodes(), PREFLOP_ITERS, CANONICAL_SUBSET_SIZE);
    eprintln!();

    let asymmetric_weights: Vec<Vec<f32>> = {
        let mut w: Vec<Vec<f32>> = (0..2).map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();
        for k in 0..NUM_PREFLOP_CLASSES {
            let s = k as f32 / NUM_PREFLOP_CLASSES as f32;
            w[0][k] = ((s - 0.3).max(0.05) * 1.5).min(1.0);
            w[1][k] = 0.6 + 0.4 * s;
        }
        w
    };
    let symmetric_uniform: Vec<Vec<f32>> = vec![vec![1.0_f32; NUM_PREFLOP_CLASSES]; 2];

    let canonical_subset: Vec<usize> = (0..CANONICAL_SUBSET_SIZE).collect();

    // The OLD random terminal_value_fn (#112 baseline).
    let random_tvf = |term_idx: usize, traverser: u8, _r: &[Vec<f32>]| -> Vec<f32> {
        (0..NUM_PREFLOP_CLASSES).map(|c| {
            let seed: u64 = (term_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (traverser as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                ^ (c as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            let bits = ((seed >> 32) & 0xFFFFFF) as i64 - (1 << 23);
            (bits as f32) / ((1 << 23) as f32)
        }).collect()
    };

    let blocking = build_class_blocking_matrix();
    let realistic_tvf = make_production_terminal_value_fn_hu(&tree, &blocking);

    let report = |label: &str, strategy: &[f32]| {
        let (z, o, m) = strategy_stats(strategy);
        let total = strategy.len();
        let mixed_pct = 100.0 * m as f32 / total as f32;
        eprintln!("  {:60} zero={:>4} one={:>4} mixed={:>4} ({:5.2}% mixed)",
            label, z, o, m, mixed_pct);
        m
    };

    // ----- (a) baseline: random TVF + asymmetric weights (#112 config) -----
    eprintln!("\n── (a) BASELINE: random terminal_value_fn + asymmetric class weights ──");
    let t = Instant::now();
    let table_a = PreflopChanceTable::new(2, asymmetric_weights.clone());
    let strat_a = run_preflop_only(&tree, &table_a, &canonical_subset, random_tvf, PREFLOP_ITERS);
    let dur_a = t.elapsed().as_secs_f64();
    let mixed_a = report(&format!("(a) random TVF + asymmetric ({:>5.1}s)", dur_a), &strat_a);

    // ----- (b) realistic TVF + asymmetric weights -----
    eprintln!("\n── (b) realistic terminal_value_fn (production fold-CFV) + asymmetric weights ──");
    let t = Instant::now();
    let table_b = PreflopChanceTable::new(2, asymmetric_weights.clone());
    let strat_b = run_preflop_only(&tree, &table_b, &canonical_subset, &realistic_tvf, PREFLOP_ITERS);
    let dur_b = t.elapsed().as_secs_f64();
    let mixed_b = report(&format!("(b) realistic TVF + asymmetric ({:>5.1}s)", dur_b), &strat_b);

    // ----- (c) random TVF + symmetric uniform weights -----
    eprintln!("\n── (c) random terminal_value_fn + symmetric uniform class weights ──");
    let t = Instant::now();
    let table_c = PreflopChanceTable::new(2, symmetric_uniform.clone());
    let strat_c = run_preflop_only(&tree, &table_c, &canonical_subset, random_tvf, PREFLOP_ITERS);
    let dur_c = t.elapsed().as_secs_f64();
    let mixed_c = report(&format!("(c) random TVF + symmetric ({:>5.1}s)", dur_c), &strat_c);

    // ----- (d) realistic TVF + symmetric uniform weights -----
    eprintln!("\n── (d) realistic terminal_value_fn + symmetric uniform class weights ──");
    let t = Instant::now();
    let table_d = PreflopChanceTable::new(2, symmetric_uniform.clone());
    let strat_d = run_preflop_only(&tree, &table_d, &canonical_subset, &realistic_tvf, PREFLOP_ITERS);
    let dur_d = t.elapsed().as_secs_f64();
    let mixed_d = report(&format!("(d) realistic TVF + symmetric ({:>5.1}s)", dur_d), &strat_d);

    let total = strat_a.len();
    eprintln!("\n══════════ DIAGNOSIS ══════════");
    eprintln!("Mixed slots (of {} total):", total);
    eprintln!("  (a) random TVF    + asymmetric: {} ({:.2}%)", mixed_a, 100.0 * mixed_a as f32 / total as f32);
    eprintln!("  (b) realistic TVF + asymmetric: {} ({:.2}%)", mixed_b, 100.0 * mixed_b as f32 / total as f32);
    eprintln!("  (c) random TVF    + symmetric:  {} ({:.2}%)", mixed_c, 100.0 * mixed_c as f32 / total as f32);
    eprintln!("  (d) realistic TVF + symmetric:  {} ({:.2}%)", mixed_d, 100.0 * mixed_d as f32 / total as f32);
    eprintln!();
    eprintln!("Interpretation:");
    eprintln!("- If (b) or (d) have substantially more mixed than (a) → realistic TVF is the");
    eprintln!("  fix; #113 variance measurement should use it.");
    eprintln!("- If (c) or (d) have substantially more mixed than (a) → symmetric ranges are");
    eprintln!("  the fix; #113 should use those.");
    eprintln!("- If NONE produce >5% mixed → the small subset (10 canonicals) is too degenerate;");
    eprintln!("  need to scale up canonical count or use a different game structure.");
}
