// Step 1 — Probe 3: reduced-scale end-to-end preflop iteration.
//
// Probe 1 measured ONE shared-tree call cost; Probe 3 measures ONE full
// preflop iteration end-to-end at a REDUCED scale to confirm the pipeline
// composes and to give a CPU-vs-GPU parity target that's feasible to run
// on both sides.
//
// Scale reductions vs production:
//   - Flop tree: 1 bet + 0 raise (simplest) instead of Option-B (2+2).
//     This dramatically shrinks the flop tree (probably to tens of
//     thousands of nodes instead of 1.52M).
//   - Subset: small number of canonical flops (5 of 1755).
//   - num_postflop_iters: 1 (minimal postflop work per call).
//   - Preflop tree: FULL Option-B 162,650 nodes (the foundation we want
//     the engine to handle).
//
// Purpose: prove the composition runs end-to-end at 6-max on the
// corrected foundation, measure the wall-clock at reduced scale,
// instrument the oracle invocations to verify real exercise. This is
// the does-this-signal-exercise-what-I-claim check at a scale where the
// answer can actually arrive.

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::solver::postflop_oracle::{PostflopValueOracle, UnabstractedPostflopOracle};
use solver_core::solver::preflop_cfr::{
    make_production_terminal_value_fn_multiway, PreflopVectorCfr,
};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

struct CountingOracle<'a> {
    inner: UnabstractedPostflopOracle<'a>,
    call_count: u64,
    total_micros: u64,
    first_nontrivial_sample: Option<(Card, Card, Card, u8, Vec<f32>)>,
}

impl<'a> CountingOracle<'a> {
    fn new(flop_tree: &'a FlatTree, n: u32) -> Self {
        Self {
            inner: UnabstractedPostflopOracle::new(flop_tree, n),
            call_count: 0,
            total_micros: 0,
            first_nontrivial_sample: None,
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
        let t = Instant::now();
        let v = self.inner.flop_root_cfv(canonical_flop, combo_ranges, traverser);
        self.call_count += 1;
        self.total_micros += t.elapsed().as_micros() as u64;
        if self.first_nontrivial_sample.is_none() && v.iter().any(|&x| x.abs() > 1e-9) {
            let sample: Vec<f32> = v.iter().take(4).cloned().collect();
            self.first_nontrivial_sample = Some((
                canonical_flop[0], canonical_flop[1], canonical_flop[2], traverser, sample,
            ));
        }
        v
    }
}

#[test]
#[ignore = "Step 1 Probe 3: reduced-scale end-to-end. Run on demand."]
fn probe_three_reduced_scale_endtoend() {
    eprintln!("\n=== Probe 3: reduced-scale end-to-end preflop iter ===\n");

    // 1. Full Option-B 6-max preflop tree (the foundation we want to validate on).
    let pre_cfg = TreeConfig {
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
    let t0 = Instant::now();
    let preflop_tree = build_tree(&pre_cfg).expect("preflop builds");
    eprintln!("Preflop tree: {} nodes ({} ms)", preflop_tree.num_nodes(), t0.elapsed().as_millis());

    // 2. COARSE flop tree: 1 bet + 0 raise. Smallest possible 6-max flop
    //    abstraction; reduces the per-call cost dramatically.
    let flop_cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 12,
        starting_stacks: vec![94; 6],
        initial_contributions: vec![0; 6],
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
    let t0 = Instant::now();
    let flop_tree = build_tree(&flop_cfg).expect("flop builds");
    eprintln!("Flop tree (1+0 abstraction): {} nodes ({} ms)",
              flop_tree.num_nodes(), t0.elapsed().as_millis());

    // 3. PreflopChanceTable + engine + oracle wrapper.
    let np = 6u8;
    let class_weights: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let table = PreflopChanceTable::new(np, class_weights);
    eprintln!("Canonical flops: {}", table.num_canonical_flops());

    let mut oracle = CountingOracle::new(&flop_tree, 1);
    let mut engine = PreflopVectorCfr::new(&preflop_tree);
    let term_fn = make_production_terminal_value_fn_multiway(&preflop_tree);

    let subset: Vec<usize> = (0..5).collect();
    let chance_count = engine.preflop_chance_node_indices(&preflop_tree).len();
    let expected_calls = (np as usize) * chance_count * subset.len();
    eprintln!(
        "\nSubset: {} canonicals × {} traversers × {} preflop chance nodes = {} expected oracle calls\n",
        subset.len(), np, chance_count, expected_calls
    );

    eprintln!("Running ONE preflop iteration...");
    let t0 = Instant::now();
    engine.run_one_iteration_subset(
        &preflop_tree, &table, &subset, &mut oracle, term_fn,
    );
    let iter_ms = t0.elapsed().as_millis();
    let iter_s = iter_ms as f64 / 1000.0;

    eprintln!("\n=== Results ===");
    eprintln!("  Iter wall-clock:       {} ms ({:.2} s)", iter_ms, iter_s);
    eprintln!("  Oracle calls:          {} (expected {})", oracle.call_count, expected_calls);
    eprintln!("  Mean per-call:         {} μs",
              if oracle.call_count > 0 { oracle.total_micros / oracle.call_count } else { 0 });
    eprintln!("  Total oracle time:     {} ms ({:.1}% of iter)",
              oracle.total_micros / 1000,
              100.0 * (oracle.total_micros as f64 / 1000.0) / iter_ms as f64);
    eprintln!("  Engine overhead:       {} ms",
              iter_ms.saturating_sub((oracle.total_micros / 1000) as u128));
    if let Some((c1, c2, c3, t, head)) = &oracle.first_nontrivial_sample {
        eprintln!("  First non-trivial v sample: flop=[{:?},{:?},{:?}] traverser={} v[0..4]={:?}",
                  c1, c2, c3, t, head);
    }

    assert!(oracle.call_count > 0, "oracle never invoked (composition stubbed)");
    assert_eq!(oracle.call_count as usize, expected_calls,
               "call count mismatch — instrumentation or loop bug");
    assert!(oracle.first_nontrivial_sample.is_some(),
            "all oracle returns were trivial (zeros) — postflop solve stub or all-zero range");

    eprintln!("\n=== Verification ===");
    eprintln!("  (a) Oracle was invoked:                    PASS");
    eprintln!("  (b) Call count matches expected:           PASS ({})", oracle.call_count);
    eprintln!("  (c) Oracle returned non-trivial values:    PASS");
    eprintln!("\nThis is the end-to-end wiring confirmation at reduced scale.");
    eprintln!("Per-iter cost at REDUCED scale (5 canonicals, 1+0 flop, n=1):");
    eprintln!("  {} ms = {:.2} s", iter_ms, iter_s);
    eprintln!("\nThis is the GPU parity target at this scale. Production projections come");
    eprintln!("from Probe 1's unit cost on the production flop tree.");
}
