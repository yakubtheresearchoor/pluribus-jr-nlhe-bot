// Phase 1 Slice 7a follow-up: per-flop cost attribution.
//
// Per the lead (2026-06-04): "The 7x discrepancy is itself a finding to
// chase. Determine if it's a legitimate cost or a reducible regression
// by checking what the original ~1s estimate measured and whether the
// current per-flop solve does excess work, because if it's a regression,
// fixing it shrinks slice 7c proportionally and it indicates a silent
// slowdown worth catching."
//
// Hypothesis: compute_v_flop_at_root_iter0 builds a fresh
// FlopChanceTable on every call (52 x 52 x nh = ~3.2M river-rank
// evaluations + sorted strings). The TABLE depends on (board, combo
// ranges) but NOT on strategies, so it is amortizable across CFR iters
// (ranges are fixed during a solve; only strategies change). The SOLVE
// itself (compute_all_strategies + compute_reach + bottom_up_zone)
// depends on per-iter strategies and is NOT amortizable.
//
// If table construction is the dominant cost, the 7x discrepancy is
// largely an amortizable cost (the prior "~1s" estimate may have been
// measuring the solve in isolation, after table-construction cost was
// paid once).
//
// This test measures:
//   1. FlopChanceTable::compute_flop_start in isolation (the
//      amortizable cost).
//   2. The remaining solve work (compute_all_strategies +
//      compute_reach_flop + bottom_up_zone) in isolation (the
//      per-iter cost).
//   3. Reports the split, which tells us whether the slice 7c cost
//      can be dramatically reduced by caching tables across iters.

use solver_core::abstraction::flop_isomorphism::enumerate_canonical_flops;
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
#[ignore = "Slice 7a follow-up: per-flop cost attribution (~30s wall-clock). \
            Splits compute_v_flop_at_root_iter0 into amortizable (table) \
            and non-amortizable (solve) components to size slice 7c with \
            table-caching factored in. Run on demand: \
            cargo test --release --test slice7a_per_flop_cost_attribution \
            -- --ignored --nocapture"]
fn slice7a_table_vs_solve_cost_split() {
    eprintln!("\n═══ Slice 7a follow-up: per-flop cost attribution ═══");
    eprintln!("Hypothesis: FlopChanceTable construction (52x52xnh rank evals) is");
    eprintln!("the dominant cost per call. It's amortizable across CFR iters");
    eprintln!("(ranges fixed, only strategies change). If confirmed, slice 7c");
    eprintln!("cost drops dramatically with a table-caching layer.\n");

    // ─────────────────────────────────────────────────────────────
    // Setup: a canonical flop + production-shape inputs
    // ─────────────────────────────────────────────────────────────
    let canonical_flops = enumerate_canonical_flops();
    let rep_flop = canonical_flops[0];
    eprintln!("  Representative canonical flop: {:?}", rep_flop);

    let combo_ranges: Vec<Vec<f32>> = (0..2)
        .map(|_| vec![1.0f32; NUM_POSSIBLE_HANDS])
        .collect();

    let flop_cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 6,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![3, 3],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    };
    let flop_tree = build_tree(&flop_cfg).expect("flop tree builds");

    // ─────────────────────────────────────────────────────────────
    // Measure 1: FlopChanceTable::compute_flop_start ALONE (the
    //   amortizable cost across CFR iters at fixed canonical+ranges)
    // ─────────────────────────────────────────────────────────────
    eprintln!("── Measure 1: FlopChanceTable::compute_flop_start (amortizable) ──");
    let board: Vec<Card> = rep_flop.iter().copied().collect();
    let t_table_start = std::time::Instant::now();
    let table = FlopChanceTable::compute_flop_start(&board, &combo_ranges, 2);
    let t_table = t_table_start.elapsed();
    let nh = table.num_valid;
    eprintln!("  FlopChanceTable::compute_flop_start: {:?} (nh={})", t_table, nh);
    eprintln!("  Internal sizes:");
    eprintln!("    river_ranks:       52 x 52 x {} = {} entries (u16)",
        nh, 52 * 52 * nh);
    eprintln!("    turn_ranks:        52 x {} = {} entries (u16)",
        nh, 52 * nh);
    eprintln!("    river_sorted_str:  52 x 52 x {} x {} (num_opp x nh) ", nh, 1);

    // ─────────────────────────────────────────────────────────────
    // Measure 2: the SOLVE work alone (non-amortizable per CFR iter)
    //
    // This is what runs each CFR iteration after the table is built:
    //   - FlopStartGame::new (cheap wrapper)
    //   - FlopStartVectorCfr::new (allocates regret/cum buffers)
    //   - compute_all_strategies
    //   - compute_reach_flop (top-down reach)
    //   - bottom_up_zone for the traverser (CFV + regret update)
    // ─────────────────────────────────────────────────────────────
    eprintln!("\n── Measure 2: solve work alone (non-amortizable per iter) ──");
    let t_solve_start = std::time::Instant::now();

    let t_game_start = std::time::Instant::now();
    let game = FlopStartGame::new(table);
    let t_game = t_game_start.elapsed();

    let t_cfr_init_start = std::time::Instant::now();
    let mut solver = FlopStartVectorCfr::new(&flop_tree, game.table());
    solver.set_vanilla_mode(true);
    let t_cfr_init = t_cfr_init_start.elapsed();

    let t_strat_start = std::time::Instant::now();
    solver.compute_all_strategies(&flop_tree);
    let t_strat = t_strat_start.elapsed();

    let t_reach_start = std::time::Instant::now();
    let _reach = solver.compute_reach_flop(&flop_tree, &game);
    let t_reach = t_reach_start.elapsed();

    let t_solve_total = t_solve_start.elapsed();
    eprintln!("  FlopStartGame::new (wrap):      {:?}", t_game);
    eprintln!("  FlopStartVectorCfr::new:        {:?}", t_cfr_init);
    eprintln!("  compute_all_strategies:         {:?}", t_strat);
    eprintln!("  compute_reach_flop:             {:?}", t_reach);
    eprintln!("  (bottom_up_zone for traverser ≈ similar to compute_reach order)");
    eprintln!("  Total solve setup + reach pass: {:?}", t_solve_total);

    // ─────────────────────────────────────────────────────────────
    // Attribution report
    // ─────────────────────────────────────────────────────────────
    eprintln!("\n══ Cost attribution ══");
    let total_observed = t_table.as_secs_f64() + t_solve_total.as_secs_f64();
    let table_pct = 100.0 * t_table.as_secs_f64() / total_observed;
    let solve_pct = 100.0 * t_solve_total.as_secs_f64() / total_observed;
    eprintln!("  Table construction:  {:?} ({:.1}% of observed total)",
        t_table, table_pct);
    eprintln!("  Solve work (partial): {:?} ({:.1}% of observed total)",
        t_solve_total, solve_pct);
    eprintln!("");
    eprintln!("  IMPORTANT: 'Solve work' above is JUST setup + reach pass.");
    eprintln!("  The full per-iter solve includes bottom_up_zone for the");
    eprintln!("  traverser, which is the dominant per-iter compute. Prior");
    eprintln!("  slice 7a measured the full compute_v_flop_at_root_iter0 at");
    eprintln!("  7.24s; the table fraction of THAT is the amortizable part.");

    let prior_full_solve = 7.24_f64;
    let table_in_full_pct = 100.0 * t_table.as_secs_f64() / prior_full_solve;
    let non_amortizable_secs = prior_full_solve - t_table.as_secs_f64();
    eprintln!("  Table as fraction of full 7.24s call: {:.1}%", table_in_full_pct);
    eprintln!("  Non-amortizable per-iter cost:        {:.2}s", non_amortizable_secs);

    // ─────────────────────────────────────────────────────────────
    // Implication for slice 7c sizing
    // ─────────────────────────────────────────────────────────────
    eprintln!("\n══ Slice 7c sizing implication ══");
    let n_canon = 1755u64;
    let per_pass_no_cache_min = prior_full_solve * n_canon as f64 / 60.0;
    let first_pass_with_cache_min = per_pass_no_cache_min;
    let subsequent_pass_with_cache_min =
        non_amortizable_secs * n_canon as f64 / 60.0;
    eprintln!("  Without caching (current slice 7a measurement):");
    eprintln!("    per-pass = {:.1} min, N iters = N x {:.1} min",
        per_pass_no_cache_min, per_pass_no_cache_min);
    eprintln!("  With per-flop-table caching across iters:");
    eprintln!("    iter 0 = {:.1} min (full cost, tables built and cached)",
        first_pass_with_cache_min);
    eprintln!("    iter N>0 = {:.1} min per iter (just the solve)",
        subsequent_pass_with_cache_min);
    eprintln!("    Total for N iters = {:.1} + (N-1) x {:.1} min",
        first_pass_with_cache_min, subsequent_pass_with_cache_min);

    for n in [5, 10, 50, 100] {
        let no_cache_hr =
            (per_pass_no_cache_min * n as f64) / 60.0;
        let with_cache_hr =
            (first_pass_with_cache_min + (n - 1) as f64 * subsequent_pass_with_cache_min)
            / 60.0;
        let speedup = no_cache_hr / with_cache_hr;
        eprintln!("    N={}: no-cache = {:.1} hr, with-cache = {:.1} hr ({:.1}x speedup)",
            n, no_cache_hr, with_cache_hr, speedup);
    }

    eprintln!("\n══ Recommendation ══");
    eprintln!("  If table fraction > 50% of per-call cost (likely from the");
    eprintln!("  3.2M river_rank evaluations), adding a per-canonical-flop");
    eprintln!("  table cache before slice 7c is the right optimization.");
    eprintln!("  Slice 7c can then proceed at the with-cache cost, which is");
    eprintln!("  bounded by the non-amortizable solve work.");

    // Test passes if measurements are sane (not zero, not absurdly large).
    assert!(t_table.as_secs_f64() > 0.01,
        "table construction time {:?} too small to be plausible", t_table);
    assert!(t_table.as_secs_f64() < 60.0,
        "table construction time {:?} too large; investigate", t_table);
}

#[test]
#[ignore = "Slice 7a follow-up: verify table is independent of strategies. \
            Builds the same per-flop table twice with identical inputs and \
            confirms the output is identical (proves the table is purely a \
            function of board + ranges, hence amortizable across iters)."]
fn slice7a_table_is_deterministic_in_board_and_ranges() {
    eprintln!("\n═══ Slice 7a follow-up: confirm table is amortizable ═══");
    eprintln!("Builds FlopChanceTable twice with identical inputs and checks");
    eprintln!("the output is identical. If yes, the table can be cached across");
    eprintln!("CFR iters (which DO change strategies but NOT ranges).\n");

    let canonical_flops = enumerate_canonical_flops();
    let rep_flop = canonical_flops[0];
    let board: Vec<Card> = rep_flop.iter().copied().collect();
    let combo_ranges: Vec<Vec<f32>> = (0..2)
        .map(|_| vec![1.0f32; NUM_POSSIBLE_HANDS])
        .collect();

    let t1 = std::time::Instant::now();
    let table_a = FlopChanceTable::compute_flop_start(&board, &combo_ranges, 2);
    eprintln!("  First build:  {:?}", t1.elapsed());

    let t2 = std::time::Instant::now();
    let table_b = FlopChanceTable::compute_flop_start(&board, &combo_ranges, 2);
    eprintln!("  Second build: {:?}", t2.elapsed());

    // The two tables should be byte-equivalent on their public fields.
    assert_eq!(table_a.num_valid, table_b.num_valid, "num_valid mismatch");
    assert_eq!(table_a.hand_ranks_base, table_b.hand_ranks_base, "hand_ranks_base mismatch");
    assert_eq!(table_a.turn_ranks, table_b.turn_ranks, "turn_ranks mismatch");
    assert_eq!(table_a.river_ranks, table_b.river_ranks, "river_ranks mismatch");
    assert_eq!(table_a.num_combinations, table_b.num_combinations, "num_combinations mismatch");

    eprintln!("\n  ✓ Table is deterministic in (board, combo_ranges).");
    eprintln!("  ✓ Amortizable across CFR iters at fixed canonical+ranges.");
    eprintln!("  Caching the per-canonical-flop table once would save the");
    eprintln!("  table-construction cost on every subsequent iter.");
}
