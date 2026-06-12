// Step 2.D.9 (#96 PREREQUISITE): production-config per-canonical wall-clock
// — direct measurement across textures.
//
// PER USER (banked 2026-06): the tiny-config 2.64-day baseline from #98 is
// a FLOOR. The dominant term is production-config postflop, and the only
// honest way to get it is direct measurement at full nh × full deck. Across
// a few representative textures to confirm per-canonical uniformity (or
// measure spread).
//
// MEASUREMENT METHODOLOGY:
//   - Run ONE per-canonical postflop solve on GPU at production config:
//     HU OptB-shape tree (stacks=50, 2+2 bet sizes), full nh=1176, full
//     deck (~47 turn × 46 river).
//   - 5 postflop iters per measurement (per-iter cost was established
//     linear in #98 + 2.D.8c, so 5 iters scales reliably to 50).
//   - 3 representative textures: rainbow uncoordinated (AhKd7c), paired
//     (8h8d2c), monotone (AsKsQs). Spread across these covers the texture
//     variation that affects per-canonical work.
//
// EXTRAPOLATION:
//   per_canonical_at_5_iters / 5 = per_iter_cost
//   per_canonical_at_50_iters = per_iter_cost × 50 + fixed_overhead
//   With negligible fixed overhead (established in #98c profile), this is
//   just × 10.
//
// OUTPUTS:
//   - Per-texture wall-clock at 5 iters
//   - Per-iter cost = wall-clock / 5
//   - Production-config per-canonical cost = per-iter × 50 (production iters)
//   - Spread across textures (CV) → confirms uniformity or measures
//   - Per-canonical "unified pipeline" cost = per-call × 100 calls
//     (10 preflop × 2 traversers × 5 chance nodes, matching #98)
//   - K=1755 production projection
//   - Ratio table for #96, framed against typical budget options.

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

// Production iter count target (extrapolated to). Per-iter is measured at
// MEASURE_ITERS and linearly scaled to PRODUCTION_POSTFLOP_ITERS.
const MEASURE_ITERS: u32 = 5;
const PRODUCTION_POSTFLOP_ITERS: u32 = 50;

// 10 preflop × 2 traversers × 5 chance nodes from the HU minimal preflop
// tree used in #98. Same shape, so the unified-pipeline per-canonical cost
// scales the same way.
const CALLS_PER_CANONICAL: u32 = 10 * 2 * 5;

fn build_production_flop_tree() -> FlatTree {
    // Match the 2.A.2 production cell tree shape (already validated to run
    // at full nh × full deck in InMemory mode on this machine): HU 1+1
    // bet sizes, stacks=20, starting_pot=4. That's not the deepest possible
    // tree, but it IS what 2.A.2 demonstrated runs at full nh × full deck;
    // the OptB 2+2 stacks=50 tree (5453 nodes) overflows InMemory state at
    // full nh × full deck (~40+ GB river-zone buffer) and would need
    // DiskBacked. For the per-canonical cost framing, the 2.A.2-shape tree
    // gives a defensible production-config number; OptB 2+2 stacks=50
    // would scale this up further (separate follow-up if #96 sits at a
    // feasibility boundary that needs the bigger-tree number).
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

fn uniform_combo_ranges(canonical: [Card; 3]) -> Vec<Vec<f32>> {
    // Production combo ranges: uniform 1.0 across all NUM_POSSIBLE_HANDS
    // (compute_flop_start will internally mask out hands that conflict with
    // the canonical board cards). Cost measurement is range-shape-
    // independent at the per-iter level.
    let np = 2;
    vec![vec![1.0_f32; NUM_POSSIBLE_HANDS]; np]
}

fn measure_one_canonical(
    ctx: &MetalContext,
    flop_tree: &FlatTree,
    canonical: [Card; 3],
    label: &str,
) -> (f64, usize) {
    let combo_ranges = uniform_combo_ranges(canonical);
    let board: Vec<Card> = canonical.iter().copied().collect();

    let t_setup = Instant::now();
    let table = FlopChanceTable::compute_flop_start(&board, &combo_ranges, 2);
    let nh = table.num_valid;
    let n_pairs: usize = table.river_decks.iter().map(|d| d.len()).sum();
    let game = FlopStartGame::new(table);
    let cpu_solver = FlopStartVectorCfr::new(flop_tree, &game.table());
    let mut gpu_solver = MetalFlopStartSolver::new(ctx, flop_tree, &game, &cpu_solver);
    let setup_secs = t_setup.elapsed().as_secs_f64();

    eprintln!("\n── {} (canonical {:?}) ──", label, canonical);
    eprintln!("  nh = {}, total pairs = {}, tree = {} nodes",
        nh, n_pairs, flop_tree.num_nodes());
    eprintln!("  setup: {:.2} s", setup_secs);

    let t_run = Instant::now();
    gpu_solver.run(ctx, flop_tree, &game, MEASURE_ITERS);
    let run_secs = t_run.elapsed().as_secs_f64();
    let per_iter = run_secs / MEASURE_ITERS as f64;
    eprintln!("  run({} iters): {:.1} s ({:.1} s/iter)", MEASURE_ITERS, run_secs, per_iter);
    (run_secs, n_pairs)
}

#[test]
#[ignore = "Step 2.D.9: production-config per-canonical measurement (hours)"]
fn step2d9_production_config_per_canonical_across_textures() {
    let flop_tree = build_production_flop_tree();
    eprintln!("\n=== Step 2.D.9: production-config per-canonical wall-clock ===");
    eprintln!("Tree: {} nodes (HU 1+1 stacks=20 — matches 2.A.2 production cell shape).",
        flop_tree.num_nodes());
    eprintln!("Per-canonical postflop solve at full nh × full deck.");
    eprintln!("Measuring {} iters then scaling linearly to {} iters (production).",
        MEASURE_ITERS, PRODUCTION_POSTFLOP_ITERS);

    let ctx = MetalContext::new().expect("Metal");

    // 3 representative textures:
    //   - rainbow uncoordinated (max suit variety, no straight/pair texture)
    //   - paired (one rank repeated, restricts opp hand space differently)
    //   - monotone (all one suit, dramatic flush-draw texture)
    let textures: Vec<(&str, [Card; 3])> = vec![
        ("rainbow uncoordinated AhKd7c", [
            card_from_str("Ah").unwrap(),
            card_from_str("Kd").unwrap(),
            card_from_str("7c").unwrap(),
        ]),
        ("paired 8h8d2c", [
            card_from_str("8h").unwrap(),
            card_from_str("8d").unwrap(),
            card_from_str("2c").unwrap(),
        ]),
        ("monotone AsKsQs", [
            card_from_str("As").unwrap(),
            card_from_str("Ks").unwrap(),
            card_from_str("Qs").unwrap(),
        ]),
    ];

    let mut measurements: Vec<(String, f64, usize)> = Vec::new();
    for (label, canonical) in &textures {
        let (run_secs, n_pairs) = measure_one_canonical(&ctx, &flop_tree, *canonical, label);
        measurements.push((label.to_string(), run_secs, n_pairs));
    }

    // Per-iter and per-canonical-at-50-iters per texture.
    eprintln!("\n=== Cross-texture summary ===");
    eprintln!("{:>32} {:>9} {:>11} {:>12} {:>14}",
        "texture", "n_pairs", "5-iter (s)", "per-iter (s)", "50-iter (h)");
    let mut per_iters: Vec<f64> = Vec::new();
    for (label, run_secs, n_pairs) in &measurements {
        let per_iter = run_secs / MEASURE_ITERS as f64;
        let production_secs = per_iter * PRODUCTION_POSTFLOP_ITERS as f64;
        let production_hours = production_secs / 3600.0;
        per_iters.push(per_iter);
        eprintln!("{:>32} {:>9} {:>11.1} {:>12.2} {:>14.2}",
            label, n_pairs, run_secs, per_iter, production_hours);
    }

    let avg_per_iter: f64 = per_iters.iter().sum::<f64>() / per_iters.len() as f64;
    let min_per_iter: f64 = per_iters.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_per_iter: f64 = per_iters.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let std_per_iter: f64 = {
        let var = per_iters.iter()
            .map(|x| (x - avg_per_iter).powi(2)).sum::<f64>() / per_iters.len() as f64;
        var.sqrt()
    };
    let cv = std_per_iter / avg_per_iter * 100.0;
    eprintln!("\n  per-iter stats: avg={:.2}s, min={:.2}s, max={:.2}s, std={:.2}s, CV={:.1}%",
        avg_per_iter, min_per_iter, max_per_iter, std_per_iter, cv);
    if cv < 15.0 {
        eprintln!("  → per-canonical uniformity HOLDS at production config (CV {:.1}% < 15%).", cv);
        eprintln!("    Average is the right summary for the projection.");
    } else {
        eprintln!("  → per-canonical uniformity DOES NOT hold (CV {:.1}% > 15%).", cv);
        eprintln!("    The projection should use avg ± spread; per-texture variation is real.");
    }

    // ── Production projection. ──
    let per_canonical_50_iters_secs = avg_per_iter * PRODUCTION_POSTFLOP_ITERS as f64;
    let per_canonical_50_iters_hours = per_canonical_50_iters_secs / 3600.0;
    eprintln!("\n=== Production-config per-canonical (single postflop call at 50 iters) ===");
    eprintln!("  avg per-canonical cost: {:.0} s = {:.2} hours per call",
        per_canonical_50_iters_secs, per_canonical_50_iters_hours);

    // Unified-pipeline production cost.
    let per_canonical_unified_hours = per_canonical_50_iters_hours * CALLS_PER_CANONICAL as f64;
    let k1755_hours = per_canonical_unified_hours * 1755.0;
    let k1755_days = k1755_hours / 24.0;
    let k1755_years = k1755_days / 365.0;
    eprintln!("\n=== K=1755 production unified-pipeline projection ===");
    eprintln!("  per-canonical (× {} calls): {:.1} hours = {:.2} days per canonical",
        CALLS_PER_CANONICAL, per_canonical_unified_hours, per_canonical_unified_hours / 24.0);
    eprintln!("  K=1755: {:.0} hours = {:.0} days = {:.1} years",
        k1755_hours, k1755_days, k1755_years);

    // ── Ratio framings. ──
    eprintln!("\n=== Cost ratios for #96 (real production-config; the lead's budget = INPUT) ===");
    for budget_hours in &[24.0_f64, 24.0 * 7.0, 24.0 * 30.0, 24.0 * 90.0, 24.0 * 365.0] {
        let ratio = k1755_hours / budget_hours;
        let budget_desc = if *budget_hours < 24.0 * 2.0 {
            format!("{:.0} hours", budget_hours)
        } else if *budget_hours < 24.0 * 35.0 {
            format!("{:.0} days", budget_hours / 24.0)
        } else if *budget_hours < 24.0 * 100.0 {
            format!("~{:.0} weeks", budget_hours / 24.0 / 7.0)
        } else if *budget_hours < 24.0 * 200.0 {
            format!("~{:.0} months", budget_hours / 24.0 / 30.0)
        } else {
            format!("~{:.0} year", budget_hours / 24.0 / 365.0)
        };
        eprintln!("  If budget = {:<14}: ratio = {:>10.0}×", budget_desc, ratio);
    }
    eprintln!("\n  The ratio number is the compression factor #96's abstraction has to deliver.");
    eprintln!("  Bucket counts per street are chosen to fit that compression factor at acceptable info loss.");

    eprintln!("\n=== CAVEATS ===");
    eprintln!("  - Per-iter cost was measured linear-in-iters in #98+#98c; the 5→50 extrapolation");
    eprintln!("    is straight-line linear at f32 floor for postflop iter scaling.");
    eprintln!("  - The 10 preflop × 2 traversers × 5 chance nodes = {} calls per canonical is", CALLS_PER_CANONICAL);
    eprintln!("    from the HU minimal preflop tree in #98. Real production preflop tree shape");
    eprintln!("    could change this multiplier (more/fewer chance nodes, more traversers).");
    eprintln!("  - Per-canonical uniformity (this measurement) doesn't capture preflop reach");
    eprintln!("    asymmetry effects on per-canonical cost — uniform combo_ranges is a worst-case-");
    eprintln!("    saturation upper bound; realistic asymmetric reach may be modestly faster.");
}
