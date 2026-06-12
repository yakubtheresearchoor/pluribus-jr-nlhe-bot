// P1.5.4 Slice B: multi-iter loop at reduced flop count, correctness-baseline-first.
//
// Per the lead's directive: before reading N off the convergence trajectory,
// confirm the preflop loop converges to a correct equilibrium (low
// exploitability via best-response), the way convergence_audit confirmed
// FlopStartVectorCfr. A loop that converges WRONG isn't a reference.
//
// Reduce flop count (the cheap dimension that preserves convergence
// dynamics per the earlier directive) rather than tree depth (which
// would change the dynamics and break transferability).
//
// Lessons carried from the proxy study:
//   - Measure TIME-AVERAGED cum_strategy convergence, not current-iter
//     strategy (which is pure best-response from iter 1).
//   - Confirm convergence holds over a WINDOW, not a single iter
//     dropping below threshold.
//   - Anomaly check across multiple instances of the held-fixed random
//     dimension before sizing any long run.
//
// What's NEW in this slice that wasn't in the proxy: the preflop loop
// aggregates signal over the canonical SET (each iter's regret update
// accumulates CFVs from all canonicals via the chance integration),
// so the preflop N may differ from the single-flop postflop proxy's
// rate. This is the FIRST time we measure convergence on the actual
// preflop engine.
//
// This test is #[ignore] because each iter runs real per-canonical
// solves (~seconds at production nh, ms at restricted nh). Run on
// demand with `--ignored --nocapture`.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::postflop_oracle::UnabstractedPostflopOracle;
use solver_core::solver::preflop_cfr::{
    make_production_terminal_value_fn_hu, PreflopVectorCfr,
};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::solver::preflop_terminal::build_class_blocking_matrix;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// Build the HU preflop tree (used by PreflopVectorCfr).
fn build_hu_preflop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![99, 98],
        initial_contributions: vec![1, 2],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
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

/// Build the HU FLOP tree (used by per_flop_solver, same postflop bet
/// sizes as the preflop tree's subtree shape for consistency).
fn build_hu_flop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 6,  // pre-flop pot at flop start (placeholder; per-flop solver re-keys to its own state)
        starting_stacks: vec![97, 97],
        initial_contributions: vec![3, 3],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    build_tree(&cfg).expect("flop tree builds")
}

#[test]
#[ignore = "Slice B: multi-iter preflop CFR on reduced canonical subset, ~minutes wall-clock. \
            Run on demand: cargo test --release --test \
            p1_5_4_slice_b_multi_iter_correctness_baseline -- --ignored --nocapture"]
fn slice_b_multi_iter_correctness_baseline_then_n_readout() {
    eprintln!("\n═══ P1.5.4 Slice B: multi-iter at reduced flop count, correctness-baseline-first ═══");

    let preflop_tree = build_hu_preflop_tree();
    let flop_tree = build_hu_flop_tree();

    // PreflopChanceTable with uniform class weights (all-1 initial range).
    let table = PreflopChanceTable::new(
        2,
        vec![vec![1.0_f32; NUM_PREFLOP_CLASSES]; 2],
    );
    let n_canonical_full = table.num_canonical_flops();
    eprintln!("Full canonical count: {}", n_canonical_full);

    // Reduced subset: take 10 canonicals spread across the canonical list.
    // (the lead's "reduce flop count, the cheap dimension"; tree depth unchanged.)
    let subset_size = 10_usize;
    let subset_indices: Vec<usize> = (0..subset_size)
        .map(|i| (i * n_canonical_full) / subset_size)
        .collect();
    eprintln!("Subset size: {} canonicals → indices: {:?}", subset_size, subset_indices);

    let blocking = build_class_blocking_matrix();

    let mut oracle = UnabstractedPostflopOracle::new(&flop_tree, 50);
    let term_val_fn = make_production_terminal_value_fn_hu(&preflop_tree, &blocking);

    let mut solver = PreflopVectorCfr::new(&preflop_tree);
    eprintln!("Preflop infoset count: {}", solver.infoset_count());

    // ─────────────────────────────────────────────────────────────────────
    // Multi-iter loop with periodic BR exploitability check.
    // ─────────────────────────────────────────────────────────────────────
    //
    // Correctness baseline: at every checkpoint, compute traverser's BR
    // value at root (per class), then take L1 norm across classes as a
    // scalar "exploitability proxy". Sum across both traversers (for HU
    // zero-sum, this approximates the Nash gap, modulo sign convention
    // and per-class weighting).
    //
    // What we look for:
    //   - Exploitability proxy DECREASES (or stabilizes at a floor) over iterations
    //   - cum_strategy iter-over-iter delta drops below a threshold AND
    //     stays below over a window (per the proxy-study lesson)

    let num_iters: u32 = 50;
    let checkpoint_iters: Vec<u32> = vec![1, 2, 5, 10, 20, 30, 40, 50];

    eprintln!("\n── Running {} iterations on subset of {} canonicals ──", num_iters, subset_size);

    let chance_nodes = solver.preflop_chance_node_indices(&preflop_tree);
    eprintln!("Preflop chance nodes: {}", chance_nodes.len());

    let mut prev_cum_strategy_snapshot: Option<Vec<f32>> = None;

    for iter in 1..=num_iters {
        let t0 = std::time::Instant::now();
        solver.run_one_iteration_subset(
            &preflop_tree, &table, &subset_indices,
            &mut oracle, &term_val_fn,
        );
        let iter_secs = t0.elapsed().as_secs_f64();

        if checkpoint_iters.contains(&iter) {
            // Compute BR exploitability proxy at this checkpoint.
            // For each traverser:
            //   1. Compute reach using current strategy (already in solver.strategy after iter)
            //   2. Compute chance_cfv at each preflop chance node using engine's per_flop_solver
            //   3. Compute BR value at root via compute_traverser_br_value
            //   4. Sum |v[c]| across classes for an exploitability scalar
            let reach = solver.compute_preflop_reach(&preflop_tree, None);
            let mut br_proxy_per_traverser = vec![0.0_f32; solver.num_players as usize];
            for t in 0..solver.num_players {
                let mut chance_cfv: Vec<Vec<f32>> = vec![vec![0.0_f32; NUM_PREFLOP_CLASSES]; preflop_tree.num_nodes()];
                for &c_idx in &chance_nodes {
                    chance_cfv[c_idx] = solver.compute_chance_node_cfv_with_expansion_subset(
                        c_idx, t, &reach, &table, &subset_indices, &mut oracle,
                    );
                }
                let br_v = solver.compute_traverser_br_value(
                    &preflop_tree, t, &chance_nodes, &reach, &chance_cfv, &term_val_fn,
                );
                let l1: f32 = br_v.iter().map(|v| v.abs()).sum();
                br_proxy_per_traverser[t as usize] = l1;
            }
            let exploitability_proxy: f32 = br_proxy_per_traverser.iter().sum::<f32>()
                / solver.num_players as f32;

            // cum_strategy iter-delta (compare snapshot iter-over-iter).
            let cur_snapshot = solver.cum_strategy.clone();
            let cum_delta = if let Some(prev) = &prev_cum_strategy_snapshot {
                cur_snapshot.iter().zip(prev.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max)
            } else {
                f32::INFINITY
            };
            prev_cum_strategy_snapshot = Some(cur_snapshot);

            eprintln!("  iter {:>3}: per-iter={:.2}s, BR L1 per trav = {:?}, expl proxy = {:.4e}, cum max-delta = {:.4e}",
                iter, iter_secs, br_proxy_per_traverser, exploitability_proxy, cum_delta);
        }
    }

    eprintln!("\n── Convergence-correctness baseline interpretation ──");
    eprintln!("");
    eprintln!("If the BR exploitability proxy DECREASES over the checkpoint sequence,");
    eprintln!("the preflop loop is converging correctly toward equilibrium (on the");
    eprintln!("subset). The N read-off is the iter where:");
    eprintln!("  (a) exploitability proxy < some threshold of starting value, AND");
    eprintln!("  (b) cum_strategy max-delta < threshold sustained over a window.");
    eprintln!("");
    eprintln!("If exploitability INCREASES or stays high, the loop is converging to");
    eprintln!("the WRONG equilibrium (engine bug, terminal-value sign, factored-CFR");
    eprintln!("convention issue) — investigate before trusting any N readout.");
    eprintln!("");
    eprintln!("Carry from proxy study (DO NOT FORGET):");
    eprintln!("  - cum_strategy iter-delta must hold over a WINDOW, not a single iter");
    eprintln!("    transient slowdown (proxy study's iter-7 milestone was misleading).");
    eprintln!("  - Preflop signal aggregates over canonicals; per-iter signal may be");
    eprintln!("    weaker than postflop single-flop, slowing emergence.");

    // Sanity assertion: iterations actually ran.
    assert!(solver.iteration() == num_iters,
        "expected {} iters, solver reports {}", num_iters, solver.iteration());
}

#[test]
#[ignore = "Slice B quick variant: 20 iters on 2 canonicals, ~minutes wall-clock. \
            For first-look correctness signal. Run on demand: cargo test --release --test \
            p1_5_4_slice_b_multi_iter_correctness_baseline slice_b_quick -- --ignored --nocapture"]
fn slice_b_quick_correctness_signal_2_canonicals_20_iters() {
    eprintln!("\n═══ Slice B quick variant: 2 canonicals × 20 iters ═══");
    eprintln!("Purpose: get an end-to-end correctness signal fast (~minutes), before");
    eprintln!("committing to the full 10-canonical × 50-iter baseline (~hours).");

    let preflop_tree = build_hu_preflop_tree();
    let flop_tree = build_hu_flop_tree();
    let table = PreflopChanceTable::new(2, vec![vec![1.0_f32; NUM_PREFLOP_CLASSES]; 2]);
    let subset_indices = vec![0_usize, 877];  // two canonicals from opposite ends
    let blocking = build_class_blocking_matrix();
    let mut oracle = UnabstractedPostflopOracle::new(&flop_tree, 50);
    let term_val_fn = make_production_terminal_value_fn_hu(&preflop_tree, &blocking);
    let mut solver = PreflopVectorCfr::new(&preflop_tree);
    let chance_nodes = solver.preflop_chance_node_indices(&preflop_tree);

    let num_iters: u32 = 20;
    let checkpoints: Vec<u32> = vec![1, 2, 5, 10, 15, 20];
    let mut prev_snap: Option<Vec<f32>> = None;

    for iter in 1..=num_iters {
        let t0 = std::time::Instant::now();
        solver.run_one_iteration_subset(&preflop_tree, &table, &subset_indices,
            &mut oracle, &term_val_fn);
        let iter_secs = t0.elapsed().as_secs_f64();

        if checkpoints.contains(&iter) {
            let reach = solver.compute_preflop_reach(&preflop_tree, None);
            let mut br_proxy = vec![0.0_f32; solver.num_players as usize];
            for t in 0..solver.num_players {
                let mut chance_cfv: Vec<Vec<f32>> = vec![vec![0.0; NUM_PREFLOP_CLASSES]; preflop_tree.num_nodes()];
                for &c_idx in &chance_nodes {
                    chance_cfv[c_idx] = solver.compute_chance_node_cfv_with_expansion_subset(
                        c_idx, t, &reach, &table, &subset_indices, &mut oracle,
                    );
                }
                let br_v = solver.compute_traverser_br_value(
                    &preflop_tree, t, &chance_nodes, &reach, &chance_cfv, &term_val_fn,
                );
                br_proxy[t as usize] = br_v.iter().map(|v| v.abs()).sum::<f32>();
            }
            let expl: f32 = br_proxy.iter().sum::<f32>() / solver.num_players as f32;
            let cur_snap = solver.cum_strategy.clone();
            let cum_delta = if let Some(p) = &prev_snap {
                cur_snap.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
            } else { f32::INFINITY };
            prev_snap = Some(cur_snap);
            eprintln!("  iter {:>2}: iter={:.2}s, BR L1 per trav = [{:.3e}, {:.3e}], expl = {:.3e}, cum-delta = {:.3e}",
                iter, iter_secs, br_proxy[0], br_proxy[1], expl, cum_delta);
        }
    }

    eprintln!("\nSee trajectory above: if exploitability proxy decreases over iters,");
    eprintln!("correctness baseline is on track for the full slice_b run.");
}
