// Step 1 — Profiling probe: find where the hour goes in ONE shared-tree
// postflop solve at num_postflop_iters=1.
//
// the lead's discipline: this project's track record says "1 hour for the
// lightest postflop solve" is a BUG, not a real cost. Switching to per-
// terminal architecture without finding the bug is the MAX_DEPTH pattern
// (making the symptom less visible while the bug carries forward).
//
// Prime suspects per the lead:
//   (a) O(n^5) terminal CFV cost (already found, per-terminal trees would
//       NOT fix this — every terminal in any tree still computes multiway
//       CFV).
//   (b) Loop or per-node bug that would carry forward to per-terminal trees.
//
// This probe re-implements compute_v_flop_at_root_converged inline with
// per-stage timers, run at TWO scales:
//   - HU (2-player, Option-B flop tree, small) — baseline cost per stage
//   - 6-max OptB (6-player, 1.52M-node flop tree) — production scale
// Comparing stage timings between the two reveals which stage scales worse
// than its big-O would predict — that's the bug location.
//
// Stage breakdown:
//   1. FlopChanceTable construction
//   2. solver.run (num_postflop_iters CFR iters)
//   3. freeze_average_strategy
//   4. compute_reach_flop
//   5. Per-river bottom_up_zone (48*47 = 2256 calls) ← largest call count
//   6. Per-turn bottom_up_zone (48 calls)
//   7. Flop bottom_up_zone (1 call)

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn profile_one_solve(label: &str, flop_tree: &FlatTree, num_players: u8, num_postflop_iters: u32) {
    eprintln!("\n=================================================================");
    eprintln!("=== Profiling: {} ({} players, {} flop-tree nodes, n_iters={}) ===",
              label, num_players, flop_tree.num_nodes(), num_postflop_iters);
    eprintln!("=================================================================\n");

    // Get a canonical flop and uniform combo ranges (per-player NUM_POSSIBLE_HANDS).
    let class_weights: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let pre_table = PreflopChanceTable::new(num_players, class_weights);
    let canonical_flop: [Card; 3] = pre_table.canonical_flops[0];
    eprintln!("Canonical flop: {:?}", canonical_flop);

    // Uniform full-NUM_POSSIBLE_HANDS combo ranges (the format
    // compute_v_flop_at_root_converged expects).
    let combo_ranges_per_player: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![1.0_f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS])
        .collect();
    let board: Vec<Card> = canonical_flop.iter().copied().collect();

    let traverser = 0u8;
    let mut total = 0u128;

    // -------- Stage 1: FlopChanceTable::compute_flop_start --------
    let t = Instant::now();
    let table = FlopChanceTable::compute_flop_start(&board, &combo_ranges_per_player, num_players);
    let s1 = t.elapsed().as_millis();
    total += s1;
    let nh = table.num_valid;
    eprintln!("Stage 1: FlopChanceTable::compute_flop_start  {:>10} ms   (nh = {})", s1, nh);

    // -------- Stage 2: solver.run with num_postflop_iters CFR iters --------
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let t = Instant::now();
    let _ = solver.run(flop_tree, &game, num_postflop_iters);
    let s2 = t.elapsed().as_millis();
    total += s2;
    eprintln!("Stage 2: solver.run ({} iters)                {:>10} ms   ({:.1} ms/iter)",
              num_postflop_iters, s2, s2 as f64 / num_postflop_iters as f64);

    // -------- Stage 3: freeze_average_strategy --------
    let t = Instant::now();
    solver.freeze_average_strategy(flop_tree);
    let s3 = t.elapsed().as_millis();
    total += s3;
    eprintln!("Stage 3: freeze_average_strategy             {:>10} ms", s3);

    // -------- Stage 4: compute_reach_flop --------
    let t = Instant::now();
    let reach = solver.compute_reach_flop(flop_tree, &game);
    let s4 = t.elapsed().as_millis();
    total += s4;
    eprintln!("Stage 4: compute_reach_flop                  {:>10} ms", s4);

    // -------- Stage 5: per-river bottom_up_zone --------
    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();
    let nn = flop_tree.num_nodes();
    let mut cfv = vec![0.0_f32; nn * nh];
    let params = DcfrParams::new(0);

    let mut s5_call_count = 0u64;
    let mut s5_total_us = 0u128;
    let mut s5_per_turn_ms: Vec<u128> = Vec::with_capacity(turn_deck.len());

    let t_all = Instant::now();
    for (ti, &tc_card) in turn_deck.iter().enumerate() {
        let t_turn = Instant::now();
        let river_deck = &table_ref.river_decks[tc_card as usize];
        for ri in 0..river_deck.len() {
            let t1 = Instant::now();
            solver.bottom_up_zone(
                flop_tree, table_ref, traverser, &reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params,
            );
            s5_total_us += t1.elapsed().as_micros();
            s5_call_count += 1;
        }
        s5_per_turn_ms.push(t_turn.elapsed().as_millis());

        // Long-running mitigation: dump intermediate progress every 8 turn cards.
        if (ti + 1) % 8 == 0 {
            eprintln!("   ...{} turn cards processed in {} ms ({} river calls so far, mean {} μs/call)",
                ti + 1, t_all.elapsed().as_millis(), s5_call_count,
                if s5_call_count > 0 { s5_total_us / s5_call_count as u128 } else { 0 });
        }
    }
    let s5 = t_all.elapsed().as_millis();
    total += s5;
    let mean_river_us = if s5_call_count > 0 { s5_total_us / s5_call_count as u128 } else { 0 };
    eprintln!("Stage 5: per-(turn,river) bottom_up_zone     {:>10} ms   ({} calls, {} μs/call mean)",
              s5, s5_call_count, mean_river_us);
    if !s5_per_turn_ms.is_empty() {
        let min = s5_per_turn_ms.iter().min().unwrap();
        let max = s5_per_turn_ms.iter().max().unwrap();
        let sum: u128 = s5_per_turn_ms.iter().sum();
        eprintln!("           per-turn-block: min={} ms max={} ms mean={} ms",
                  min, max, sum / s5_per_turn_ms.len() as u128);
    }

    // -------- Stage 6: per-turn bottom_up_zone --------
    let t = Instant::now();
    for ti in 0..turn_deck.len() {
        solver.bottom_up_zone(
            flop_tree, table_ref, traverser, &reach, &mut cfv,
            Zone::Turn, Some(ti), None, &params,
        );
    }
    let s6 = t.elapsed().as_millis();
    total += s6;
    eprintln!("Stage 6: per-turn bottom_up_zone             {:>10} ms   ({} calls)",
              s6, turn_deck.len());

    // -------- Stage 7: flop bottom_up_zone --------
    let t = Instant::now();
    solver.bottom_up_zone(
        flop_tree, table_ref, traverser, &reach, &mut cfv,
        Zone::Flop, None, None, &params,
    );
    let s7 = t.elapsed().as_millis();
    total += s7;
    eprintln!("Stage 7: flop bottom_up_zone                 {:>10} ms", s7);

    eprintln!("---");
    eprintln!("TOTAL:                                          {:>10} ms ({:.2} s)\n",
              total, total as f64 / 1000.0);

    // Stage shares (rounded for readability).
    let pct = |s: u128| 100.0 * (s as f64) / (total as f64);
    eprintln!("Stage shares of total:");
    eprintln!("  S1 table:        {:6.2}%", pct(s1));
    eprintln!("  S2 solver.run:   {:6.2}%", pct(s2));
    eprintln!("  S3 freeze:       {:6.2}%", pct(s3));
    eprintln!("  S4 reach:        {:6.2}%", pct(s4));
    eprintln!("  S5 river-zones:  {:6.2}%   <-- largest call count (2256 at HU/OptB)", pct(s5));
    eprintln!("  S6 turn-zones:   {:6.2}%", pct(s6));
    eprintln!("  S7 flop-zone:    {:6.2}%", pct(s7));
}

#[test]
#[ignore = "Step 1 profiling: HU OptB flop tree. Fast (~seconds). Run on demand."]
fn profile_hu_optb_one_solve() {
    let flop_cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 6,
        starting_stacks: vec![97; 2],
        initial_contributions: vec![0; 2],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    };
    let flop_tree = build_tree(&flop_cfg).expect("HU OptB flop builds");
    profile_one_solve("HU OptB", &flop_tree, 2, 1);
}

#[test]
#[ignore = "Step 1 profiling: 6-max OptB flop tree. Slow (likely many minutes — kill if needed). Run on demand."]
fn profile_6max_optb_one_solve() {
    let flop_cfg = TreeConfig {
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
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    };
    let flop_tree = build_tree(&flop_cfg).expect("6-max OptB flop builds");
    profile_one_solve("6-max OptB", &flop_tree, 6, 1);
}
