// Step 1 — Inner-call timer probe (profile-driven per the lead's directive).
//
// The run-loop allocation fix removed 4 nn*nh allocations per iter but HU
// OptB at nh=1176 still SIGKILLs at 5 min. This probe times each individual
// primitive call inside run()'s body (compute_all_strategies, compute_reach_*,
// bottom_up_zone) to identify the dominant remaining cost — instead of
// guessing which allocation to patch next.
//
// Each call is exercised ONCE (1 traverser, 1 turn, 1 river) so even if a
// single call is multi-second, the probe completes in well under the 5-min
// harness limit.
//
// From these per-call times we can compute per-iter cost analytically:
//   per_iter ≈ np × (
//       compute_all_strategies +
//       compute_reach_flop +
//       n_turn × (compute_reach_turn + n_river × (compute_reach_river +
//                                                  bottom_up_zone_River) +
//                 bottom_up_zone_Turn) +
//       bottom_up_zone_Flop
//   )
// and identify which term dominates.

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
#[ignore = "Step 1 inner-call timer: pinpoint dominant cost after run-loop alloc fix. Run on demand."]
fn time_each_inner_call_hu_optb_nh1176() {
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
    eprintln!("\n=== Inner-call timer probe: HU OptB flop tree ===");
    eprintln!("  flop tree: {} nodes", tree.num_nodes());

    // Build table at uniform ranges.
    let class_weights: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let pre_table = PreflopChanceTable::new(np, class_weights);
    let canonical: [Card; 3] = pre_table.canonical_flops[0];
    let combo_ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS])
        .collect();
    let board: Vec<Card> = canonical.iter().copied().collect();
    let t = Instant::now();
    let table = FlopChanceTable::compute_flop_start(&board, &combo_ranges, np);
    eprintln!("  table build:           {:>8} ms (nh={})", t.elapsed().as_millis(), table.num_valid);

    let game = FlopStartGame::new(table);
    let nh = game.table().num_valid;
    let nn = tree.num_nodes();
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    eprintln!("  nh={}, nn={}, per-buffer alloc cost = nn*nh*4 = {} KB ({} MB)",
              nh, nn, nn*nh*4/1024, nn*nh*4/1024/1024);

    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();
    let traverser: u8 = 0;
    let params = DcfrParams::new(0);

    eprintln!("\n--- Per-call timings (single-call, traverser={}) ---", traverser);

    // 1. compute_all_strategies
    let t = Instant::now();
    solver.compute_all_strategies(&tree);
    let cas_ms = t.elapsed().as_millis();
    eprintln!("  compute_all_strategies:                {:>8} ms", cas_ms);

    // 2. compute_reach_flop
    let t = Instant::now();
    let flop_reach = solver.compute_reach_flop(&tree, &game);
    let crf_ms = t.elapsed().as_millis();
    eprintln!("  compute_reach_flop:                    {:>8} ms (returned {} f32)",
              crf_ms, flop_reach.len());

    // 3. compute_reach_turn (turn 0)
    let ti = 0;
    let tc = turn_deck[ti];
    let t = Instant::now();
    let turn_reach = solver.compute_reach_turn(&tree, ti, &flop_reach);
    let crt_ms = t.elapsed().as_millis();
    eprintln!("  compute_reach_turn (ti=0):             {:>8} ms", crt_ms);

    // 4. compute_reach_river (turn 0, river 0)
    let ri = 0;
    let t = Instant::now();
    let river_reach = solver.compute_reach_river(&tree, ti, ri, &turn_reach);
    let crr_ms = t.elapsed().as_millis();
    eprintln!("  compute_reach_river (ti=0, ri=0):      {:>8} ms", crr_ms);

    // 5. bottom_up_zone(River, ti=0, ri=0)
    let mut cfv = vec![0.0f32; nn * nh];
    let t = Instant::now();
    solver.bottom_up_zone(
        &tree, table_ref, traverser, &river_reach, &mut cfv,
        Zone::River, Some(ti), Some(ri), &params,
    );
    let buz_river_ms = t.elapsed().as_millis();
    eprintln!("  bottom_up_zone(River, ti=0, ri=0):     {:>8} ms", buz_river_ms);

    // 6. bottom_up_zone(Turn, ti=0)
    let mut turn_cfv = vec![0.0f32; nn * nh];
    let t = Instant::now();
    solver.bottom_up_zone(
        &tree, table_ref, traverser, &turn_reach, &mut turn_cfv,
        Zone::Turn, Some(ti), None, &params,
    );
    let buz_turn_ms = t.elapsed().as_millis();
    eprintln!("  bottom_up_zone(Turn, ti=0):            {:>8} ms", buz_turn_ms);

    // 7. bottom_up_zone(Flop)
    let mut flop_cfv = vec![0.0f32; nn * nh];
    let t = Instant::now();
    solver.bottom_up_zone(
        &tree, table_ref, traverser, &flop_reach, &mut flop_cfv,
        Zone::Flop, None, None, &params,
    );
    let buz_flop_ms = t.elapsed().as_millis();
    eprintln!("  bottom_up_zone(Flop):                  {:>8} ms", buz_flop_ms);

    // Analytical projection of full per-iter cost.
    let n_turn = turn_deck.len() as u64;
    let mut n_river_sum: u64 = 0;
    for &tc in turn_deck.iter() {
        n_river_sum += table_ref.river_decks[tc as usize].len() as u64;
    }
    eprintln!("\n--- Analytical per-iter projection ---");
    eprintln!("  n_turn = {}, sum(n_river over turns) = {}", n_turn, n_river_sum);

    // run() does for each traverser:
    //   1 × compute_all_strategies
    //   1 × compute_reach_flop
    //   n_turn × compute_reach_turn
    //   sum(n_river) × (compute_reach_river + bottom_up_zone(River))
    //   n_turn × bottom_up_zone(Turn)
    //   1 × bottom_up_zone(Flop)
    let per_trav_ms: u128 =
        cas_ms
        + crf_ms
        + (crt_ms as u128) * (n_turn as u128)
        + ((crr_ms + buz_river_ms) as u128) * (n_river_sum as u128)
        + (buz_turn_ms as u128) * (n_turn as u128)
        + buz_flop_ms;
    let per_iter_ms: u128 = per_trav_ms * (np as u128);
    eprintln!("  per-traverser ms: {} ({:.2} s)", per_trav_ms, per_trav_ms as f64 / 1000.0);
    eprintln!("  per-iter ms (×np={}): {} ({:.2} s, {:.2} min)",
              np, per_iter_ms, per_iter_ms as f64 / 1000.0, per_iter_ms as f64 / 60000.0);

    eprintln!("\n--- Dominant cost ---");
    let river_total = ((crr_ms + buz_river_ms) as u128) * (n_river_sum as u128);
    let turn_total = (buz_turn_ms as u128) * (n_turn as u128);
    let flop_total = buz_flop_ms as u128;
    let reach_turn_total = (crt_ms as u128) * (n_turn as u128);
    let strategy_total = cas_ms;
    let reach_flop_total = crf_ms;
    eprintln!("  compute_all_strategies (1×):           {:>10} ms", strategy_total);
    eprintln!("  compute_reach_flop (1×):               {:>10} ms", reach_flop_total);
    eprintln!("  compute_reach_turn × n_turn:           {:>10} ms", reach_turn_total);
    eprintln!("  (compute_reach_river + River-zone) × sum: {:>10} ms  <-- expected dominant", river_total);
    eprintln!("  bottom_up_zone(Turn) × n_turn:         {:>10} ms", turn_total);
    eprintln!("  bottom_up_zone(Flop) × 1:              {:>10} ms", flop_total);
}
