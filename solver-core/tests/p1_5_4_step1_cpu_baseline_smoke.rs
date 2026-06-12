// Step 1 (CPU baseline) — Stage A smoke test.
//
// the lead's directive: measure per-iteration cost of the complete solve on the
// corrected 162,650-node Option-B 6-max preflop tree using vector CFR. BEFORE
// the number is trusted as the baseline, confirm the iteration actually
// exercises the full preflop-to-postflop composition at production richness
// (postflop solves running, not stubbed; corrected tree, not stale).
//
// This is the does-this-signal-exercise-what-I-claim check applied to the
// measurement instrument itself.
//
// Stage A (this test): SMOKE only.
//   - Build the verified 162,650-node Option-B 6-max preflop tree.
//   - Wrap UnabstractedPostflopOracle in a counting wrapper to verify the
//     oracle is invoked the expected number of times.
//   - Run ONE preflop iteration over a SMALL subset of canonical flops
//     (10 of 1755) with num_postflop_iters=1 (minimal postflop work) to
//     keep wall-clock tractable while still exercising the full wiring.
//   - Report: oracle call count, total wall-clock, per-oracle-call mean
//     time, sample of returned CFVs (to verify non-trivial values).
//
// Stage B (future): scale up to full 1755 × num_postflop_iters=100 once
// the smoke confirms the wiring is correct and we have a per-call cost
// number to project from.

use std::time::Instant;

use solver_core::card::Card;
use solver_core::solver::postflop_oracle::{PostflopValueOracle, UnabstractedPostflopOracle};
use solver_core::solver::preflop_cfr::{
    make_production_terminal_value_fn_multiway, PreflopVectorCfr,
};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// Wrapper that counts oracle calls and records aggregate timing.
/// This is the measurement-instrument verification: if call_count == 0,
/// the iteration didn't actually exercise the oracle (stubbed). If the
/// returned values are all zeros, the oracle returned trivial values
/// (the cached-to-zero failure mode). Both would invalidate the measurement.
struct CountingOracle<'a> {
    inner: UnabstractedPostflopOracle<'a>,
    call_count: u64,
    total_micros: u64,
    nontrivial_samples: Vec<(Card, Card, Card, u8, Vec<f32>)>,
    sample_budget: usize,
}

impl<'a> CountingOracle<'a> {
    fn new(flop_tree: &'a FlatTree, num_postflop_iters: u32) -> Self {
        Self {
            inner: UnabstractedPostflopOracle::new(flop_tree, num_postflop_iters),
            call_count: 0,
            total_micros: 0,
            nontrivial_samples: Vec::new(),
            sample_budget: 5,
        }
    }
}

impl<'a> PostflopValueOracle for CountingOracle<'a> {
    fn flop_root_cfv(
        &mut self,
        canonical_flop: [Card; 3],
        combo_ranges: &[Vec<f32>],
        traverser: u8,
    ) -> Vec<f32> {
        let start = Instant::now();
        let v = self.inner.flop_root_cfv(canonical_flop, combo_ranges, traverser);
        let elapsed = start.elapsed().as_micros() as u64;

        self.call_count += 1;
        self.total_micros += elapsed;

        if self.nontrivial_samples.len() < self.sample_budget {
            if v.iter().any(|&x| x.abs() > 1e-9) {
                let mut sample: Vec<f32> = v.iter().cloned().take(4).collect();
                sample.push(v.len() as f32); // record length at the end
                self.nontrivial_samples.push((
                    canonical_flop[0], canonical_flop[1], canonical_flop[2],
                    traverser, sample,
                ));
            }
        }

        v
    }
}

fn build_optb_6max_preflop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; 6],
        initial_contributions: vec![1, 2, 0, 0, 0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
            max_bets_per_street: None,
    };
    build_tree(&cfg).expect("Option-B 6-max preflop tree must build (corrected foundation)")
}

fn build_optb_6max_flop_tree() -> FlatTree {
    // Representative flop subgame: 6 players at the flop with pot = 12
    // (everyone called BB pre, common multi-way limped pot) and ~94 stack
    // remaining. Option-B betting (2 bet, 2 raise) matching preflop.
    //
    // For the smoke test, the exact pot/stack are not the measurement —
    // what matters is the flop tree builds and the oracle can run on it.
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 12,
        starting_stacks: vec![94; 6],
        initial_contributions: vec![0; 6],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&cfg).expect("Option-B 6-max flop tree must build")
}

#[test]
#[ignore = "Step 1 Stage A: CPU baseline smoke. ~minutes to tens of minutes; run on demand: \
            cargo test --release --test p1_5_4_step1_cpu_baseline_smoke -- --ignored --nocapture"]
fn step1_stage_a_smoke_one_iter_subset() {
    eprintln!("\n=== Step 1 Stage A: CPU baseline smoke (ONE preflop iter, small subset) ===\n");

    // 1. Build the verified 162,650-node Option-B 6-max preflop tree.
    let t0 = Instant::now();
    let preflop_tree = build_optb_6max_preflop_tree();
    let preflop_build_ms = t0.elapsed().as_millis();
    eprintln!("Preflop tree built: {} nodes in {} ms", preflop_tree.num_nodes(), preflop_build_ms);
    assert_eq!(
        preflop_tree.num_nodes(),
        162650,
        "must be the verified-correct 162,650-node tree (foundation gate)"
    );

    // 2. Build a representative flop subgame tree.
    let t0 = Instant::now();
    let flop_tree = build_optb_6max_flop_tree();
    let flop_build_ms = t0.elapsed().as_millis();
    eprintln!("Flop tree built:   {} nodes in {} ms", flop_tree.num_nodes(), flop_build_ms);

    // 3. Build PreflopChanceTable (uniform class weights for the smoke test).
    use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
    let np = 6u8;
    let class_weights: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let t0 = Instant::now();
    let table = PreflopChanceTable::new(np, class_weights);
    eprintln!(
        "PreflopChanceTable built: {} canonical flops in {} ms",
        table.num_canonical_flops(),
        t0.elapsed().as_millis()
    );

    // 4. Wrap UnabstractedPostflopOracle with counting wrapper. Use
    //    num_postflop_iters=1 for minimal per-call work in the smoke.
    let num_postflop_iters = 1u32;
    let mut oracle = CountingOracle::new(&flop_tree, num_postflop_iters);

    // 5. Build the preflop CFR engine + terminal value fn.
    let mut engine = PreflopVectorCfr::new(&preflop_tree);
    let terminal_fn = make_production_terminal_value_fn_multiway(&preflop_tree);

    // 6. Subset: first 10 canonical flops (of 1755). This bounds the
    //    smoke at 6 traversers × 87 chance nodes × 10 canonicals = 5,220
    //    oracle calls, each running 1 postflop iter on a 6-max flop tree.
    let subset_indices: Vec<usize> = (0..10).collect();
    let chance_node_count = engine.preflop_chance_node_indices(&preflop_tree).len();
    let expected_oracle_calls = (np as usize) * chance_node_count * subset_indices.len();
    eprintln!(
        "\nSubset: {} canonical flops (of {} total)",
        subset_indices.len(),
        table.num_canonical_flops()
    );
    eprintln!(
        "Expected oracle calls this iter: {} traversers × {} chance nodes × {} canonicals = {}",
        np, chance_node_count, subset_indices.len(), expected_oracle_calls
    );

    // 7. Run ONE iteration. Time it.
    eprintln!("\nRunning one preflop iteration...");
    let t0 = Instant::now();
    engine.run_one_iteration_subset(
        &preflop_tree, &table, &subset_indices, &mut oracle, terminal_fn,
    );
    let iter_ms = t0.elapsed().as_millis();
    let iter_secs = t0.elapsed().as_secs_f64();

    // 8. Report.
    let call_count = oracle.call_count;
    let total_oracle_micros = oracle.total_micros;
    let mean_oracle_micros = if call_count > 0 { total_oracle_micros / call_count } else { 0 };
    let samples = &oracle.nontrivial_samples;

    eprintln!("\n=== Results ===");
    eprintln!("  Total iter wall-clock:      {} ms ({:.2} s)", iter_ms, iter_secs);
    eprintln!("  Oracle calls (counted):     {}", call_count);
    eprintln!("  Oracle calls (expected):    {}", expected_oracle_calls);
    eprintln!("  Mean oracle call time:      {} μs", mean_oracle_micros);
    eprintln!("  Total oracle time:          {} ms", total_oracle_micros / 1000);
    eprintln!("  Oracle / iter ratio:        {:.1}%",
              100.0 * (total_oracle_micros as f64 / 1000.0) / iter_ms as f64);
    eprintln!("  Engine overhead (iter - oracle): {} ms",
              iter_ms.saturating_sub((total_oracle_micros / 1000) as u128));

    eprintln!("\n=== Verification (the does-this-signal-exercise-what-I-claim check) ===");
    let exercised = call_count > 0;
    let count_matches = call_count as usize == expected_oracle_calls;
    let nontrivial_returns = !samples.is_empty();

    eprintln!("  (a) Oracle was invoked:                    {}", if exercised { "PASS" } else { "FAIL" });
    eprintln!("  (b) Call count == expected:                {} ({} vs {})",
              if count_matches { "PASS" } else { "FAIL (instrumentation mismatch)" },
              call_count, expected_oracle_calls);
    eprintln!("  (c) Oracle returned non-trivial values:    {} ({} samples non-zero)",
              if nontrivial_returns { "PASS" } else { "FAIL (returning all-zero CFVs)" },
              samples.len());

    if !samples.is_empty() {
        eprintln!("\n  Sample non-trivial CFVs (first {} oracle returns):", samples.len());
        for (c1, c2, c3, t, v) in samples.iter() {
            let len = v.last().copied().unwrap_or(0.0) as usize;
            let head: Vec<String> = v.iter().take(4).map(|x| format!("{:.3e}", x)).collect();
            eprintln!("    flop={:?},{:?},{:?} traverser={} v.len={} v[0..4]=[{}]",
                      c1, c2, c3, t, len, head.join(", "));
        }
    }

    assert!(exercised, "oracle was never invoked — iteration did not exercise the postflop composition (STUBBED)");
    assert!(count_matches, "call count {} != expected {} — instrumentation or loop structure mismatch", call_count, expected_oracle_calls);
    assert!(nontrivial_returns, "all oracle returns were trivial (zeros) — postflop solve is returning stub values");

    // 9. Projection to full-scale per-iter cost.
    let full_canonicals = table.num_canonical_flops() as u64;
    let subset_size = subset_indices.len() as u64;
    let scale_canonicals = full_canonicals as f64 / subset_size as f64;
    let projected_iter_ms_n1 = iter_ms as f64 * scale_canonicals;
    let projected_iter_ms_n100 = projected_iter_ms_n1 * 100.0; // num_postflop_iters scaling

    eprintln!("\n=== Linear projection to full-scale baseline ===");
    eprintln!("  Subset → full canonicals (1755) scale: {:.1}×", scale_canonicals);
    eprintln!("  Projected per-iter at full 1755 + num_postflop_iters=1:  {:.1} s", projected_iter_ms_n1 / 1000.0);
    eprintln!("  Projected per-iter at full 1755 + num_postflop_iters=100: {:.1} s", projected_iter_ms_n100 / 1000.0);
    eprintln!("  (Projections are LINEAR scaling estimates — measure directly for the trusted baseline.)");
}
