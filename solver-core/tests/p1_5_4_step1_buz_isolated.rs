// Step 1 — Disturbance B (isolated): time ONE bottom_up_zone(River, 0, 0)
// call WITHOUT running compute_all_strategies first.
//
// What this discriminates:
//   - If the call completes fast (≤ seconds): per-call work is reasonable,
//     and the streaming-strategy fix (compute strategy per (tc,rc) just-in-
//     time, eliminate compute_all_strategies's 175 GB up-front materialize)
//     would convert the >5-min hang into per-call latency × n_pairs ≈ 2352 ×
//     (seconds per call) wall-clock per iter. Tractable.
//
//   - If the call also hangs (>5 min): per-call work is the dominant cost,
//     streaming-strategy wouldn't help, and the issue is somewhere else
//     (terminal CFV, regret update, reach computation, etc.).
//
// What this does NOT do: produce correct CFV values. The strategy buffer
// is left uninitialized (all-zero) so the strategy reads return uniform/zero
// instead of regret-matched. Doesn't matter for timing the call structure.

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
#[ignore = "Step 1 B-isolated: one bottom_up_zone(River, 0, 0) without compute_all_strategies. Run on demand."]
fn time_one_bottom_up_zone_river_isolated() {
    let np = 2u8;
    let flop_cfg = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 6,
        starting_stacks: vec![97; np as usize],
        initial_contributions: vec![0; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&flop_cfg).expect("HU OptB flop builds");

    let class_weights: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let pre_table = PreflopChanceTable::new(np, class_weights);
    let canonical: [Card; 3] = pre_table.canonical_flops[0];
    let combo_ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS])
        .collect();
    let board: Vec<Card> = canonical.iter().copied().collect();
    let table = FlopChanceTable::compute_flop_start(&board, &combo_ranges, np);

    let nh = table.num_valid;
    let nn = tree.num_nodes();
    eprintln!("\n=== B-isolated: ONE bottom_up_zone(River, 0, 0) without compute_all_strategies ===");
    eprintln!("  tree: {} nodes, nh={}", nn, nh);

    let game = FlopStartGame::new(table);

    eprintln!("\n  Building solver (allocates ~526 GB of virtual zero-pages)...");
    let t0 = Instant::now();
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());
    eprintln!("  solver::new: {} ms", t0.elapsed().as_millis());

    // Skip compute_all_strategies (the >5 min hang).
    // Strategy buffer stays at allocator-zero. We're not validating CFV
    // correctness, only timing the call.

    eprintln!("\n  compute_reach_flop...");
    let t = Instant::now();
    let flop_reach = solver.compute_reach_flop(&tree, &game);
    eprintln!("    {} ms (returned {} f32 = {} MB)",
              t.elapsed().as_millis(), flop_reach.len(), flop_reach.len() * 4 / (1 << 20));

    eprintln!("\n  compute_reach_turn(ti=0)...");
    let t = Instant::now();
    let turn_reach = solver.compute_reach_turn(&tree, 0, &flop_reach);
    eprintln!("    {} ms", t.elapsed().as_millis());

    eprintln!("\n  compute_reach_river(ti=0, ri=0)...");
    let t = Instant::now();
    let river_reach = solver.compute_reach_river(&tree, 0, 0, &turn_reach);
    eprintln!("    {} ms", t.elapsed().as_millis());

    let mut cfv = vec![0.0f32; nn * nh];
    let params = DcfrParams::new(0);

    eprintln!("\n  bottom_up_zone(River, ti=0, ri=0)  <-- THE MEASUREMENT");
    let t = Instant::now();
    solver.bottom_up_zone(
        &tree, game.table(), 0, &river_reach, &mut cfv,
        Zone::River, Some(0), Some(0), &params,
    );
    let buz_ms = t.elapsed().as_millis();
    eprintln!("    {} ms ({:.3} s)", buz_ms, buz_ms as f64 / 1000.0);

    let n_turn = solver.n_turn_outcomes();
    let max_n_river = solver.max_river_outcomes();
    let n_pairs = (n_turn * max_n_river) as u128;
    let projected_per_iter_ms = buz_ms as u128 * n_pairs;
    eprintln!("\n=== Projection ===");
    eprintln!("  ONE bottom_up_zone(River, 0, 0): {} ms", buz_ms);
    eprintln!("  Projected per-iter (×{} pairs): {} ms = {:.2} s = {:.2} min",
              n_pairs, projected_per_iter_ms,
              projected_per_iter_ms as f64 / 1000.0,
              projected_per_iter_ms as f64 / 60000.0);
    eprintln!("  (Not including compute_reach_river per pair, turn/flop calls, np traversers.)");

    eprintln!("\n=== Discriminator outcome ===");
    if buz_ms < 5000 {
        eprintln!("  → per-call cost is {} ms (≤5 sec). Streaming-strategy fix would convert", buz_ms);
        eprintln!("    the >5-min compute_all_strategies hang into per-call latency × n_pairs.");
        eprintln!("    Storage layout is INCIDENTAL, not structural.");
    } else if buz_ms < 60_000 {
        eprintln!("  → per-call cost is {} ms (5-60 sec). Streaming helps but per-call also needs", buz_ms);
        eprintln!("    work. Hybrid fix.");
    } else {
        eprintln!("  → per-call cost is {} ms (>1 min). Streaming wouldn't fix the issue;", buz_ms);
        eprintln!("    per-call compute is the dominant cost (terminal eval, regret update, etc.)");
    }
}
