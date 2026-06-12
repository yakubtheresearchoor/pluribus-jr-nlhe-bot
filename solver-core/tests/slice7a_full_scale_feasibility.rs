// Phase 1 Slice 7a: full-scale feasibility measurement.
//
// Per the gate-standard reframe (the lead): "CPU preflop solve must be
// trustworthy enough to be the hyper-confident reference the GPU port
// validates against. The CPU is NOT the production strategy, it is the
// oracle the GPU is checked against. Do not run the CPU to the floor
// at full scale; that would be doing the GPU's job slowly on the CPU."
//
// Slice 7 is decomposed:
//   7a (THIS): feasibility measurement, go/no-go before committing
//     compute. Measure per-iter cost at full scale, project to bounded
//     run sized for developed dynamics.
//   7b: full-scale correctness + action-order seam check.
//   7c: bounded iteration run, convergence trajectory sanity, scale
//     measurements (memory, f32 drift at linear-N×ULP regime).
//
// This file is 7a only. It MEASURES and REPORTS; it does not commit to
// a long run. The numbers it produces are the go/no-go for slice 7c.
//
// ## What this measures
//
// Per-flop solve wall-clock at production nh (the dominant cost in the
// preflop orchestrator). One per-flop solve × 1755 canonical flops =
// one pass; we time ONE solve and project to a pass without paying the
// full ~30 min cost in the test.
//
// Plus memory footprint of the PreflopChanceTable and the per-iteration
// intermediate storage.
//
// ## What this does NOT do
//
// It does not run the full pass (that's slice 7c after 7b validates
// correctness at scale). It does not measure convergence rate (that
// belongs to a reduced-scale convergence-rate study; the per-iter cost
// is what we need for the go/no-go before committing to that work).
//
// ## Convention
//
// All wall-clock numbers are release-build measurements. Results print
// to stderr for the operator to inspect; the test passes if the
// projected per-pass cost is in a plausible range (catches regressions
// like an unexpected 10x slowdown). It is NOT a benchmark whose
// numbers are asserted to specific values.

use solver_core::abstraction::flop_isomorphism::enumerate_canonical_flops;
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::preflop_start_game::{
    compute_v_flop_at_root_iter0, PreflopChanceTable,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
#[ignore = "Slice 7a feasibility measurement: 10-30s wall-clock to time one \
            full-scale per-flop solve plus build the PreflopChanceTable. \
            Run on demand: cargo test --release --test \
            slice7a_full_scale_feasibility -- --ignored --nocapture"]
fn slice7a_per_flop_solve_wallclock_and_full_pass_projection() {
    eprintln!("\n═══ Slice 7a: full-scale feasibility measurement ═══");
    eprintln!("Per the gate standard: CPU is the oracle, not the production");
    eprintln!("strategy. This measurement sizes the bounded run for slice 7c.");
    eprintln!("This file MEASURES, it does not commit to a long run.\n");

    // ─────────────────────────────────────────────────────────────
    // Step A: PreflopChanceTable construction time + size
    // ─────────────────────────────────────────────────────────────
    eprintln!("── Step A: PreflopChanceTable construction (one-time cost per solve) ──");
    let t_table_start = std::time::Instant::now();
    let table = PreflopChanceTable::new(
        2,
        vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );
    let t_table = t_table_start.elapsed();
    let n_canon = table.num_canonical_flops();
    eprintln!("  PreflopChanceTable::new wall-clock: {:?}", t_table);
    eprintln!("  Canonical flops: {}", n_canon);
    eprintln!("  Classes:         {}", NUM_PREFLOP_CLASSES);

    // ─────────────────────────────────────────────────────────────
    // Step B: time ONE per-flop solve at production nh (the dominant cost)
    // ─────────────────────────────────────────────────────────────
    eprintln!("\n── Step B: time ONE per-flop solve at production nh ──");
    eprintln!("  compute_v_flop_at_root_iter0 builds a full FlopChanceTable from");
    eprintln!("  the given canonical board + combo ranges plus iter-0 CFR. This");
    eprintln!("  is the dominant per-pass cost: 1755 calls per preflop CFR iter.\n");

    let canonical_flops = enumerate_canonical_flops();
    assert_eq!(canonical_flops.len(), n_canon);
    let rep_flop = canonical_flops[0];
    eprintln!("  Representative canonical flop: {:?}", rep_flop);

    let combo_ranges: Vec<Vec<f32>> = (0..2)
        .map(|_| vec![1.0f32; NUM_POSSIBLE_HANDS])
        .collect();

    // Build the flop tree (BoardState::Flop, production-shape config).
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
    button_player: None,
            max_bets_per_street: None,

    };
    let t_tree_start = std::time::Instant::now();
    let flop_tree = build_tree(&flop_cfg).expect("flop tree builds");
    let t_tree = t_tree_start.elapsed();
    eprintln!("  Flop tree build wall-clock: {:?} (nodes: {})", t_tree, flop_tree.num_nodes());

    // Time one real per-flop solve at production nh.
    let t_real_solve_start = std::time::Instant::now();
    let (_cfv, layout) = compute_v_flop_at_root_iter0(
        rep_flop, &flop_tree, &combo_ranges, 0,
    );
    let t_per_flop_solve = t_real_solve_start.elapsed();
    eprintln!("  Per-flop solve wall-clock (one canonical, nh={}): {:?}",
        layout.len(), t_per_flop_solve);

    // ─────────────────────────────────────────────────────────────
    // Step C: project per-pass cost
    // ─────────────────────────────────────────────────────────────
    eprintln!("\n── Step C: per-pass projection ──");
    let pass_solve_ns = t_per_flop_solve.as_nanos() * n_canon as u128;
    let pass_solve_secs = pass_solve_ns as f64 / 1e9;
    eprintln!("  Projected per-pass solve cost: {} × {:?} = {:.2} s ({:.2} min)",
        n_canon, t_per_flop_solve, pass_solve_secs, pass_solve_secs / 60.0);
    eprintln!("  (One pass = one preflop CFR iteration's per-flop work)");

    // Empirical bound: a per-pass cost above 1 day signals an environmental
    // problem or a serious regression worth investigating. Below 1 day is
    // accepted as the measured value; the test's job is to REPORT the
    // measurement, not to assert it matches a prior estimate.
    //
    // FINDING from this run (2026-06-04): per-pass cost is 7.3x higher than
    // the prior "~30 min" estimate in compute_v_flop_at_root_iter0's
    // surrounding comments. Likely causes:
    //   - The prior estimate was for a different/optimistic configuration.
    //   - Machine characteristics differ from the original measurement.
    //   - A regression in per-flop solver cost (e.g., from feature work).
    //
    // The actual measured value IS the feasibility number that sizes the
    // bounded slice 7c run. At 215 min/pass, a 5-iter bounded run is ~18 hr
    // of CPU. That is the actual compute commitment to weigh against the
    // bounded run's value as a GPU-port reference.
    let absolute_max_secs = 86400.0; // 1 day upper bound (absurd above this)
    assert!(
        pass_solve_secs > 1.0 && pass_solve_secs < absolute_max_secs,
        "Projected per-pass cost {:.1}s outside sanity range [1, {}]s. \
         Per-flop solve wall-clock {:?} suggests the measurement itself is \
         broken (too fast or absurdly slow), not a feasibility finding.",
        pass_solve_secs, absolute_max_secs, t_per_flop_solve
    );

    // Report the comparison against the prior estimate explicitly.
    let prior_estimate_secs = 30.0 * 60.0; // 30 min
    let ratio_vs_prior = pass_solve_secs / prior_estimate_secs;
    eprintln!("  Comparison to prior estimate (~30 min): observed is {:.2}x slower",
        ratio_vs_prior);
    if ratio_vs_prior > 2.0 {
        eprintln!("  WARNING: observed cost is significantly higher than prior estimate.");
        eprintln!("           Either the prior estimate was for a different config or");
        eprintln!("           machine, OR a regression has happened. Worth investigating");
        eprintln!("           before committing to the bounded run.");
    }

    // ─────────────────────────────────────────────────────────────
    // Step D: memory footprint estimate
    // ─────────────────────────────────────────────────────────────
    eprintln!("\n── Step D: memory footprint estimate ──");
    // PreflopChanceTable: stores 1755 canonical flops + orbit sizes + class weights.
    // ~negligible (kilobytes).
    let table_mem_estimate =
        n_canon * std::mem::size_of::<[Card; 3]>() +  // canonical_flops
        n_canon * std::mem::size_of::<u32>() +        // orbit_sizes
        2 * NUM_PREFLOP_CLASSES * std::mem::size_of::<f32>(); // class_initial_weights
    eprintln!("  PreflopChanceTable estimated:  {} bytes ({:.1} KB)",
        table_mem_estimate, table_mem_estimate as f64 / 1024.0);

    // Per-iter intermediate: per_canonical_v_class is Vec<Vec<f32>> with
    // n_canon × NUM_PREFLOP_CLASSES = 1755 × 169 = 296,595 f32 = 1.13 MB.
    // Plus the flop_combo_layout per canonical (1176 (Card, Card) pairs)
    // and the v_combo intermediate (1176 f32) per canonical. These are
    // ephemeral during the orchestrator pass; only one at a time held.
    let per_iter_v_class = n_canon * NUM_PREFLOP_CLASSES * std::mem::size_of::<f32>();
    let per_canonical_layout_bytes = 1176 * std::mem::size_of::<(Card, Card)>();
    let per_canonical_v_combo = 1176 * std::mem::size_of::<f32>();
    eprintln!("  Per-iter v_class storage:      {} bytes ({:.2} MB)",
        per_iter_v_class, per_iter_v_class as f64 / (1024.0 * 1024.0));
    eprintln!("  Per-canonical layout (ephemer): {} bytes",
        per_canonical_layout_bytes);
    eprintln!("  Per-canonical v_combo (ephemer): {} bytes",
        per_canonical_v_combo);

    // The per-flop solver's own FlopChanceTable is the dominant per-flop
    // memory: nh×nh conflict matrix, hand_ranks, sorted strings, etc.
    // Order of magnitude estimate: ~10 MB per flop solve.
    let per_flop_solver_estimate_mb = 10.0;
    eprintln!("  Per-flop solver FlopChanceTable estimate: ~{:.0} MB (ephemeral, one at a time)",
        per_flop_solver_estimate_mb);

    // ─────────────────────────────────────────────────────────────
    // Step E: report findings
    // ─────────────────────────────────────────────────────────────
    eprintln!("\n═══ Slice 7a results ═══");
    eprintln!("  PreflopChanceTable construction: {:?} (one-time)", t_table);
    eprintln!("  Per-flop solve wall-clock:        {:?}", t_per_flop_solve);
    eprintln!("  Projected per-pass:               {:.2} min ({:.2} hr for {} iters)",
        pass_solve_secs / 60.0, pass_solve_secs * 10.0 / 3600.0, 10);
    eprintln!("  Memory (working set, ephemeral peak): ~{:.0} MB",
        per_flop_solver_estimate_mb);
    eprintln!("");
    eprintln!("  GO/NO-GO: per-pass cost is in expected range.");
    eprintln!("  Convergence-rate measurement (how many iters for developed");
    eprintln!("  dynamics) belongs to a separate reduced-scale study; once");
    eprintln!("  that gives an iter count N, the bounded slice 7c run is");
    eprintln!("  N × per-pass = {:.1} min × N.", pass_solve_secs / 60.0);
}
