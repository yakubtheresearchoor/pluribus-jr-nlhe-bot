// Step 1 — Probe 1: shared-tree single-oracle-call unit cost.
//
// The smoke test as designed (10 canonicals × 6 traversers × 662 chance =
// 39,720 oracle calls on the 1.52M-node 6-max Option-B flop tree) was
// infeasible — killed after 9 hours of CPU time. This probe measures the
// fundamental unit: ONE oracle call (one canonical flop, uniform combo
// ranges) at num_postflop_iters=1. From this, the full per-iter cost is
// a linear projection.

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::postflop_oracle::{PostflopValueOracle, UnabstractedPostflopOracle};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_optb_6max_flop_tree() -> FlatTree {
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
    build_tree(&cfg).expect("flop tree builds")
}

#[test]
#[ignore = "Step 1 Probe 1: shared-tree unit-cost probe. Run on demand."]
fn probe_one_shared_tree_unit_cost() {
    eprintln!("\n=== Probe 1: shared-tree single-oracle-call unit cost ===\n");

    let t0 = Instant::now();
    let flop_tree = build_optb_6max_flop_tree();
    eprintln!("Flop tree built: {} nodes in {} ms", flop_tree.num_nodes(), t0.elapsed().as_millis());

    // Build PreflopChanceTable just to get a valid canonical_flop.
    let class_weights: Vec<Vec<f32>> = (0..6)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let table = PreflopChanceTable::new(6, class_weights);
    let canonical: [Card; 3] = table.canonical_flops[0];
    eprintln!("Test canonical flop: {:?}", canonical);

    // Build uniform combo_ranges in flop_combo_layout order.
    use solver_core::solver::preflop_start_game::flop_combo_layout;
    let layout = flop_combo_layout(canonical);
    eprintln!("combo layout has {} entries", layout.len());
    let combo_ranges: Vec<Vec<f32>> = (0..6)
        .map(|_| vec![1.0_f32 / layout.len() as f32; layout.len()])
        .collect();

    // Probe with num_postflop_iters = 1.
    eprintln!("\n--- Probe num_postflop_iters = 1 ---");
    let mut oracle = UnabstractedPostflopOracle::new(&flop_tree, 1);
    let t0 = Instant::now();
    let v = oracle.flop_root_cfv(canonical, &combo_ranges, 0);
    let elapsed_ms = t0.elapsed().as_millis();
    let elapsed_s = t0.elapsed().as_secs_f64();
    eprintln!("  Wall-clock: {} ms ({:.2} s)", elapsed_ms, elapsed_s);
    eprintln!("  Returned v.len() = {}", v.len());
    eprintln!("  v[0..4] = {:?}", v.iter().take(4).collect::<Vec<_>>());
    eprintln!("  Any non-zero: {}", v.iter().any(|&x| x.abs() > 1e-9));
    let unit_cost_s = elapsed_s;

    eprintln!("\n=== Projection to full per-iter cost ===");
    let np = 6;
    let chance_nodes = 662; // from the smoke output
    let canonicals = 1755;
    for npfi in [1u32, 10, 100] {
        let scale = (np as f64) * (chance_nodes as f64) * (canonicals as f64) * (npfi as f64);
        let projected_iter_s = unit_cost_s * scale;
        let projected_iter_h = projected_iter_s / 3600.0;
        let projected_iter_d = projected_iter_h / 24.0;
        eprintln!(
            "  num_postflop_iters={:3}: per-iter ≈ {:.1} s = {:.2} h = {:.2} d (single CPU core, single-thread)",
            npfi, projected_iter_s, projected_iter_h, projected_iter_d
        );
    }
    eprintln!("\n(Projection is linear in (calls × num_postflop_iters), assuming per-call cost\n  is independent of postflop iter count beyond the trivial proportionality.\n  Actual production needs both per-iter cost AND iterations-to-converge —\n  this probe gives the per-iter scale only.)");
}
