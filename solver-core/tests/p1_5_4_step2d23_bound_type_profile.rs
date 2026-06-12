// Step 2.D.23 (#115 CRITICAL PATH): bound-type profile of production-config
// per-canonical solve.
//
// 24h budget requires the quality-free GPU efficiency lever — warm-start
// (the quality-cost lever) was ruled out by #113/#114. This profile
// determines which GPU efficiency play applies:
//
//   - BARRIER-bound (host orchestration dominates) → batch independent
//     canonical solves into fewer dispatches, quality-free, potentially
//     large compression.
//   - BANDWIDTH-bound (memory wall) → fix data layout.
//   - COMPUTE-bound (kernel math dominates) → GPU efficiency exhausted,
//     only abstraction reduces cost (bucketing + action abstraction must
//     carry the full compression).
//
// PRIMARY DISCRIMINATOR: StageProfile.unattributed (total wall-clock minus
// per-stage attributed time). Per-stage timing brackets each GPU dispatch
// with waitUntilCompleted, so attributed time IS GPU busy time per stage.
// The gap = host orchestration overhead between dispatches. If gap is
// large relative to attributed, barrier-bound. If gap is small, GPU-bound.
//
// SECONDARY DISCRIMINATOR: scaling with iter count. If unattributed scales
// linearly with K (per-iter fixed overhead per dispatch cycle), barrier-
// bound. If unattributed is constant (one-time setup), the per-iter cost
// IS the compute/bandwidth cost.

#![cfg(feature = "metal")]

use std::time::Instant;

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::StageProfile;
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

fn measure_profile(
    ctx: &MetalContext,
    flop_tree: &FlatTree,
    canonical: [Card; 3],
    iters: u32,
    label: &str,
) -> (StageProfile, f64) {
    let np = 2;
    let combo_ranges: Vec<Vec<f32>> = vec![vec![1.0_f32; NUM_POSSIBLE_HANDS]; np];
    let board: Vec<Card> = canonical.iter().copied().collect();
    let table = FlopChanceTable::compute_flop_start(&board, &combo_ranges, np as u8);
    let nh = table.num_valid;
    let n_pairs: usize = table.river_decks.iter().map(|d| d.len()).sum();
    let game = FlopStartGame::new(table);
    let cpu_solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let mut gpu_solver = MetalFlopStartSolver::new(ctx, flop_tree, &game, &cpu_solver);

    eprintln!("\n── {} — K={} iters, nh={}, n_pairs={}, tree={} nodes ──",
        label, iters, nh, n_pairs, flop_tree.num_nodes());

    let t_wall = Instant::now();
    let profile = gpu_solver.run_profiled(ctx, flop_tree, &game, iters);
    let wall_secs = t_wall.elapsed().as_secs_f64();

    eprint!("{}", profile.report());
    eprintln!("  outer wall-clock: {:.4} s (matches profile.total={:.4} s)",
        wall_secs, profile.total.as_secs_f64());
    (profile, wall_secs)
}

#[test]
#[ignore = "Step 2.D.23 (#115): bound-type profile at production config — minutes wall-clock"]
fn step2d23_bound_type_profile_at_production_config() {
    let flop_tree = build_production_flop_tree();
    eprintln!("\n=== Step 2.D.23: bound-type profile of production-config per-canonical ===");
    eprintln!("Tree: {} nodes (HU 1+1 stacks=20).", flop_tree.num_nodes());
    eprintln!("Full nh × full deck per-canonical solve at multiple K to discriminate");
    eprintln!("barrier-bound vs bandwidth-bound vs compute-bound.");
    eprintln!();
    eprintln!("Interpretation key:");
    eprintln!("  - unattributed (host overhead) > 30% of total → BARRIER-BOUND");
    eprintln!("    → batch canonicals: quality-free compression");
    eprintln!("  - unattributed < 10% AND per-iter wall scales linearly with K → GPU-BOUND");
    eprintln!("    → next: bandwidth vs compute discrimination needed");
    eprintln!();

    let ctx = MetalContext::new().expect("Metal");

    // Single texture for the profile — rainbow uncoordinated, representative.
    let canonical: [Card; 3] = [
        card_from_str("Ah").unwrap(),
        card_from_str("Kd").unwrap(),
        card_from_str("7c").unwrap(),
    ];

    // LEAN VERSION: single K=10 measurement, one solver allocation.
    // (Multi-K linearity check was too expensive at production config —
    // 3× allocation of full nh × full deck state took 3+ hours total.
    // The primary discriminator is unattributed % at the largest K; the
    // K-scaling linearity check is a confirmatory secondary signal
    // filed for follow-up if needed.)
    let k_values: Vec<u32> = vec![10];

    let mut measurements: Vec<(u32, StageProfile, f64)> = Vec::new();
    for &k in &k_values {
        let (profile, wall) = measure_profile(&ctx, &flop_tree, canonical,
            k, &format!("K={}", k));
        measurements.push((k, profile, wall));
    }

    // ── Bound-type discrimination ──
    eprintln!("\n\n══════════ BOUND-TYPE ANALYSIS ══════════");
    eprintln!();
    eprintln!("{:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "K", "wall (s)", "attr (s)", "attr%", "unattr (s)", "unattr%", "ms/iter");
    for (k, p, _wall) in &measurements {
        let total = p.total.as_secs_f64();
        let attr = p.attributed().as_secs_f64();
        let unattr = total - attr;
        let attr_pct = attr / total.max(1e-12) * 100.0;
        let unattr_pct = unattr / total.max(1e-12) * 100.0;
        let ms_per_iter = total * 1000.0 / *k as f64;
        eprintln!("{:>5}  {:>10.3}  {:>10.3}  {:>9.1}%  {:>10.3}  {:>9.1}%  {:>10.1}",
            k, total, attr, attr_pct, unattr, unattr_pct, ms_per_iter);
    }

    // Per-stage breakdown — pick K=10 (longest, most stable signal).
    let largest = measurements.last().unwrap();
    let p = &largest.1;
    let total = p.total.as_secs_f64();
    eprintln!("\n── Per-stage breakdown @ K={} (dominant stages) ──", largest.0);
    let stages: Vec<(&str, f64)> = vec![
        ("compute_strategies",       p.compute_strategies.as_secs_f64()),
        ("compute_reach_flop",       p.compute_reach_flop.as_secs_f64()),
        ("compute_reach_turn",       p.compute_reach_turn.as_secs_f64()),
        ("compute_reach_river",      p.compute_reach_river.as_secs_f64()),
        ("bottom_up_river (showdown)", p.bottom_up_river.as_secs_f64()),
        ("bottom_up_turn",           p.bottom_up_turn.as_secs_f64()),
        ("bottom_up_flop",           p.bottom_up_flop.as_secs_f64()),
        ("chance_accum_river",       p.chance_accumulate_river.as_secs_f64()),
        ("chance_finalize_river",    p.chance_finalize_river.as_secs_f64()),
        ("chance_accum_turn",        p.chance_accumulate_turn.as_secs_f64()),
        ("chance_finalize_turn",     p.chance_finalize_turn.as_secs_f64()),
        ("zero_buffer_total",        p.zero_buffer_total.as_secs_f64()),
    ];
    let mut sorted = stages.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, s) in sorted.iter().take(6) {
        let pct = s / total.max(1e-12) * 100.0;
        eprintln!("  {:30} {:>10.3} s ({:>5.1}%)", name, s, pct);
    }

    // ── Linearity / per-iter scaling ──
    eprintln!("\n── Per-iter scaling (test for barrier-bound signature) ──");
    let per_iter_total: Vec<f64> = measurements.iter()
        .map(|(k, p, _)| p.total.as_secs_f64() / *k as f64).collect();
    let per_iter_attr: Vec<f64> = measurements.iter()
        .map(|(k, p, _)| p.attributed().as_secs_f64() / *k as f64).collect();
    let per_iter_unattr: Vec<f64> = measurements.iter()
        .map(|(k, p, _)| (p.total - p.attributed()).as_secs_f64() / *k as f64).collect();

    eprintln!("{:>5}  {:>14}  {:>14}  {:>14}", "K", "total/iter (s)", "attr/iter (s)", "unattr/iter (s)");
    for i in 0..measurements.len() {
        eprintln!("{:>5}  {:>14.4}  {:>14.4}  {:>14.4}",
            measurements[i].0, per_iter_total[i], per_iter_attr[i], per_iter_unattr[i]);
    }

    // ── VERDICT ──
    eprintln!("\n══════════ VERDICT ══════════");
    let k_largest = measurements.last().unwrap();
    let total_largest = k_largest.1.total.as_secs_f64();
    let unattr_largest = (k_largest.1.total - k_largest.1.attributed()).as_secs_f64();
    let unattr_pct_largest = unattr_largest / total_largest.max(1e-12) * 100.0;

    if unattr_pct_largest > 30.0 {
        eprintln!("→ BARRIER-BOUND (unattributed {:.1}% of total at K={}).", unattr_pct_largest, k_largest.0);
        eprintln!("  Host orchestration between GPU dispatches dominates wall-clock.");
        eprintln!("  Lever: BATCH canonical solves (multiple per-canonical solves into one command");
        eprintln!("  buffer / one dispatch wave). Quality-free compression. Potential: up to 1/unattr×");
        eprintln!("  if dispatch overhead amortizes; bounded by GPU work per batch.");
    } else if unattr_pct_largest < 10.0 {
        eprintln!("→ GPU-BOUND (unattributed only {:.1}% — GPU is the bottleneck).", unattr_pct_largest);
        if measurements.len() >= 2 {
            let scaling_ratio = per_iter_total.last().unwrap() / per_iter_total.first().unwrap();
            if (0.85..=1.15).contains(&scaling_ratio) {
                eprintln!("  Per-iter time near-constant across K ({:.2}× from K={} to K={}).",
                    scaling_ratio, measurements.first().unwrap().0, k_largest.0);
                eprintln!("  GPU is fully utilized. Next: bandwidth vs compute discrimination needed");
                eprintln!("  (roofline analysis or kernel-level Metal Counters).");
            } else {
                eprintln!("  Per-iter time NOT constant ({:.2}× scaling) — investigate before banking.",
                    scaling_ratio);
            }
        } else {
            eprintln!("  (Only one K measured — multi-K linearity check skipped. The primary");
            eprintln!("   discriminator (unattributed %) still applies. Next: bandwidth vs compute.)");
        }
    } else {
        eprintln!("→ MIXED ({:.1}% unattributed). Partial barrier overhead, partial GPU work.", unattr_pct_largest);
        eprintln!("  Both batching (some compression) and GPU efficiency are levers.");
    }

    eprintln!();
    eprintln!("CAVEATS:");
    eprintln!("- Tree is HU 1+1 stacks=20 ({} nodes). Full HU OptB stacks=50 is 5453 nodes (~12× bigger);", flop_tree.num_nodes());
    eprintln!("  per-iter compute scales with tree size, so production tree shifts toward GPU-bound.");
    eprintln!("- Single texture (rainbow). Per-iter cost is texture-uniform per #99 (CV<15%).");
    eprintln!("- Profile timing is wall-clock around dispatches (waitUntilCompleted blocks);");
    eprintln!("  attributed time IS GPU busy time per stage. Unattributed IS host-side gap.");
}
