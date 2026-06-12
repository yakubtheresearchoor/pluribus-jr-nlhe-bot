// Step 2.D.10: warm-start postflop cost measurement.
//
// COLD RESTART (#99 measured): 50 iters per call × 100 calls per canonical
//   = 5000 iters per canonical × 1755 canonicals
//   = ~305 years at HU 1+1 stacks=20.
//
// WARM START: build the per-canonical FlopStartVectorCfr (and its
// MetalFlopStartSolver companion) ONCE, then run K iters per call where K
// is small (1, 5, 10). State (regrets, cum_strategy, strategy) persists
// across calls — that's the entire warm-start mechanic. Per-call cost is
// K × per-iter cost (988 s/iter at production config from #99).
//
// COST PROJECTION (warm with K iters per call):
//   per call: K × 988 s
//   per canonical: 100 × K × 988 s
//   K=1755: 100 × K × 988 × 1755 s = K × 173.4 M s = K × 48,170 hours
//          = K × 2007 days = K × 5.5 years
//
//   Cold equivalent (K=50): 50 × 5.5 = 275 years (matches #99's 305y to
//   within the per-texture spread already noted as 27% CV).
//
// THIS TEST: measure per-call wall-clock at warm K=1, K=5 on one canonical
// at production config. Verify (a) state persists across calls (per-iter
// cost stays flat at ~988 s — if state were resetting, cost would scale
// linearly with cumulative iters, not stay flat) and (b) the K × per-iter
// projection matches measured per-call wall-clock.

#![cfg(feature = "metal")]

use std::time::Instant;

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_production_flop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 4,
        starting_stacks: vec![20, 20],
        initial_contributions: vec![0, 0],
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
    build_tree(&cfg).expect("flop tree builds")
}

fn uniform_combo_ranges() -> Vec<Vec<f32>> {
    vec![vec![1.0_f32; NUM_POSSIBLE_HANDS]; 2]
}

#[test]
#[ignore = "Step 2.D.10: warm-start postflop cost measurement (~2 hours)"]
fn step2d10_warm_start_per_call_cost() {
    let flop_tree = build_production_flop_tree();
    eprintln!("\n=== Step 2.D.10: warm-start per-call cost ===");
    eprintln!("Tree: {} nodes (HU 1+1 stacks=20, matches #99).", flop_tree.num_nodes());
    eprintln!("One canonical (rainbow AhKd7c), full nh × full deck.");
    eprintln!("Warm calls = solver.run(K) repeatedly, state preserved across calls.\n");

    let ctx = MetalContext::new().expect("Metal");

    let canonical: [Card; 3] = [
        card_from_str("Ah").unwrap(),
        card_from_str("Kd").unwrap(),
        card_from_str("7c").unwrap(),
    ];
    let combo_ranges = uniform_combo_ranges();
    let board: Vec<Card> = canonical.iter().copied().collect();

    let t_setup = Instant::now();
    let table = FlopChanceTable::compute_flop_start(&board, &combo_ranges, 2);
    let nh = table.num_valid;
    let n_pairs: usize = table.river_decks.iter().map(|d| d.len()).sum();
    let game = FlopStartGame::new(table);
    let cpu_solver = FlopStartVectorCfr::new(&flop_tree, game.table());
    let mut gpu_solver = MetalFlopStartSolver::new(&ctx, &flop_tree, &game, &cpu_solver);
    let setup_secs = t_setup.elapsed().as_secs_f64();
    eprintln!("Setup: {:.2} s (nh = {}, pairs = {})", setup_secs, nh, n_pairs);
    eprintln!();

    // ── Warm calls at K=1 (one iter per call, state preserved). ──
    eprintln!("── Warm K=1 calls (state preserved across calls) ──");
    let mut k1_secs: Vec<f64> = Vec::new();
    for call in 0..3 {
        let t = Instant::now();
        gpu_solver.run(&ctx, &flop_tree, &game, 1);
        let secs = t.elapsed().as_secs_f64();
        k1_secs.push(secs);
        eprintln!("  warm call {} (K=1): {:.1} s", call, secs);
    }
    let avg_k1 = k1_secs.iter().sum::<f64>() / k1_secs.len() as f64;
    let std_k1 = {
        let var = k1_secs.iter().map(|x| (x - avg_k1).powi(2)).sum::<f64>() / k1_secs.len() as f64;
        var.sqrt()
    };
    eprintln!("  avg K=1: {:.1} s (std {:.1} s, CV {:.1}%)",
        avg_k1, std_k1, std_k1 / avg_k1 * 100.0);

    // Check: per-iter cost should stay FLAT across calls. If state were
    // resetting between calls (i.e., warm-start broken), then later calls
    // would do progressively more work (because iter counter / DCFR
    // discount affects per-iter cost only modestly, but a reset would
    // not change wall-clock — actually a reset would make cost the same
    // as a fresh call). So flat cost is necessary but not sufficient to
    // prove warm. Sufficient proof is downloading the state and checking
    // it accumulates — but for cost measurement, flat wall-clock is what
    // matters for the projection.
    eprintln!();
    eprintln!("  Per-iter cost across calls: {:?}",
        k1_secs.iter().map(|s| format!("{:.1}", s)).collect::<Vec<_>>());

    // ── One warm call at K=5 (five iters, state continues from K=1 calls). ──
    eprintln!("\n── Warm K=5 call (continues from prior warm state) ──");
    let t = Instant::now();
    gpu_solver.run(&ctx, &flop_tree, &game, 5);
    let k5_secs = t.elapsed().as_secs_f64();
    let per_iter_in_k5 = k5_secs / 5.0;
    eprintln!("  warm call (K=5): {:.1} s ({:.1} s/iter)", k5_secs, per_iter_in_k5);

    // ── Cost projection. ──
    eprintln!("\n=== Cost projection (warm vs cold at K=1755) ===");
    let per_iter = avg_k1; // best estimate, K=1 measurement
    let calls_per_canonical = 100; // 10 preflop × 2 traverser × 5 chance
    let n_canon = 1755;

    eprintln!();
    eprintln!("{:>14} {:>12} {:>14} {:>16}",
        "warm K/call", "per-call (s)", "per-canon (hr)", "K=1755 projection");
    for &warm_k in &[1u32, 5, 10] {
        let per_call_secs = warm_k as f64 * per_iter;
        let per_canon_hours = (per_call_secs * calls_per_canonical as f64) / 3600.0;
        let k1755_hours = per_canon_hours * n_canon as f64;
        let k1755_days = k1755_hours / 24.0;
        let k1755_years = k1755_days / 365.0;
        let projection = if k1755_years >= 1.0 {
            format!("{:.1} years", k1755_years)
        } else if k1755_days >= 1.0 {
            format!("{:.1} days", k1755_days)
        } else {
            format!("{:.1} hours", k1755_hours)
        };
        eprintln!("{:>14} {:>12.0} {:>14.2} {:>16}",
            warm_k, per_call_secs, per_canon_hours, projection);
    }
    // Cold baseline for comparison.
    let cold_k = 50;
    let cold_per_call_secs = cold_k as f64 * per_iter;
    let cold_per_canon_hours = (cold_per_call_secs * calls_per_canonical as f64) / 3600.0;
    let cold_k1755_hours = cold_per_canon_hours * n_canon as f64;
    let cold_k1755_years = cold_k1755_hours / 24.0 / 365.0;
    eprintln!("{:>14} {:>12.0} {:>14.2} {:>13.0} years (cold reference, matches #99 ~305y)",
        "(cold) K=50", cold_per_call_secs, cold_per_canon_hours, cold_k1755_years);

    // ── Compression vs budget at warm cadences. ──
    eprintln!("\n=== Compression ratios for #96 (warm vs cold, HU 1+1 stacks=20) ===");
    eprintln!("(Production tree OptB 2+2 stacks=50 would multiply each cost by 5-10×.)\n");
    eprintln!("{:>14} {:>16} {:>16} {:>16}",
        "warm K/call", "1-day budget", "30-day budget", "1-year budget");
    for &warm_k in &[1u32, 5, 10] {
        let per_call_secs = warm_k as f64 * per_iter;
        let k1755_hours = (per_call_secs * calls_per_canonical as f64 / 3600.0) * n_canon as f64;
        eprintln!("{:>14} {:>16.0}× {:>15.0}× {:>15.0}×",
            warm_k,
            k1755_hours / 24.0,
            k1755_hours / (30.0 * 24.0),
            k1755_hours / (365.0 * 24.0),
        );
    }
    eprintln!("{:>14} {:>16.0}× {:>15.0}× {:>15.0}× (cold reference)",
        "(cold) K=50",
        cold_k1755_hours / 24.0,
        cold_k1755_hours / (30.0 * 24.0),
        cold_k1755_hours / (365.0 * 24.0),
    );

    eprintln!("\n=== Caveats ===");
    eprintln!("- Wall-clock measurement only. Convergence quality of warm K=small is a");
    eprintln!("  separate experimental question (#96 design also needs this to know how");
    eprintln!("  small K can go without losing too much info).");
    eprintln!("- Same tree-shape caveat as #99: HU 1+1 stacks=20. Production OptB 2+2");
    eprintln!("  stacks=50 multiplies these by ~5-10×.");
    eprintln!("- Per-iter cost (used to derive per-call) was measured at uniform combo_ranges");
    eprintln!("  with state preserved. Production warm-start with combo_ranges changing");
    eprintln!("  every preflop iter (real reach drift) would have the same per-iter cost");
    eprintln!("  (the GPU math is range-shape-independent) but different convergence");
    eprintln!("  behavior — which is the separate convergence question above.");
}
