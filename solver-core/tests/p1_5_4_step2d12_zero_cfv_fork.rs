// Step 2.D.12: chase the zero-CFV bug. Three paths at the same canonical
// + ranges, compare:
//   (A) PRODUCTION iter-0 path:   compute_v_flop_at_root_iter0 (FULL deck)
//   (B) PRODUCTION converged:     compute_v_flop_at_root_converged (FULL deck, K=10)
//   (C) MY SUBSET CONVERGED:      compute_flop_start_subset_with_decks +
//                                  run(K=10) + freeze + bottom_up_extract
//                                  (this is what #11b found returning zero)
//
// Fork:
//   A,B nonzero AND C zero → bug in (C): subset path or my freeze+extract
//     mirror is broken. Test problem, not production.
//   A,B nonzero AND C nonzero (but my reading showed zero?) → indexing
//     mistake in my prior diagnostic. Test problem.
//   A,B,C all zero → production extraction path is producing zero CFV at
//     this canonical+ranges, which is a real finding about the production
//     CFV handoff (would be a serious bug in the unified loop).
//   A,B zero → degeneracy: this canonical or ranges produce zero CFV
//     legitimately (unlikely but possible).

use solver_core::card::{card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::solver::preflop_start_game::{
    compute_v_flop_at_root_converged, compute_v_flop_at_root_iter0,
    flop_combo_layout,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_tiny_flop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 4,
        starting_stacks: vec![10, 10],
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

fn pick_subset(canonical: [Card; 3]) -> (Vec<u16>, Vec<u8>, Vec<Vec<u8>>) {
    let board_mask: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut chosen: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = solver_core::card::index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        chosen.push(idx as u16);
        if chosen.len() == 8 { break; }
    }
    let mut hand_mask = board_mask;
    for &i in &chosen {
        let (c1, c2) = solver_core::card::index_to_card_pair(i as usize);
        hand_mask |= 1u64 << c1;
        hand_mask |= 1u64 << c2;
    }
    let mut turn_cards: Vec<u8> = Vec::new();
    for c in 0u8..52u8 {
        if hand_mask & (1u64 << c) != 0 { continue; }
        turn_cards.push(c);
        if turn_cards.len() == 2 { break; }
    }
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let mut rivers: Vec<u8> = Vec::new();
        for c in 0u8..52u8 {
            if hand_mask & (1u64 << c) != 0 { continue; }
            if c == tc { continue; }
            rivers.push(c);
            if rivers.len() == 2 { break; }
        }
        river_decks[tc as usize] = rivers;
    }
    (chosen, turn_cards, river_decks)
}

#[test]
#[ignore = "Step 2.D.12: zero-CFV fork diagnostic"]
fn step2d12_zero_cfv_fork() {
    let flop_tree = build_tiny_flop_tree();
    eprintln!("\n=== Step 2.D.12: zero-CFV fork ===");
    eprintln!("Tree: {} nodes (HU 1+1 stacks=10).", flop_tree.num_nodes());

    use solver_core::abstraction::flop_isomorphism::enumerate_canonical_flops;
    let canonicals = enumerate_canonical_flops();
    let canonical = canonicals[0];
    eprintln!("Canonical: {:?}", canonical);

    // Asymmetric combo_ranges (NUM_POSSIBLE_HANDS-indexed, the production format).
    let np = 2;
    let mut combo_ranges_full: Vec<Vec<f32>> = vec![vec![0.0f32; NUM_POSSIBLE_HANDS]; np];
    let layout_engine = flop_combo_layout(canonical);
    for p in 0..np {
        for (li, &(c1, c2)) in layout_engine.iter().enumerate() {
            let h = (li as f32 * 0.1 + p as f32 * 0.3).sin() * 0.5 + 0.5;
            combo_ranges_full[p][card_pair_to_index(c1, c2)] = h;
        }
    }

    let traverser = 0u8;

    // ── (A) production iter-0 path, FULL deck ──
    eprintln!("\n── (A) compute_v_flop_at_root_iter0 (production iter-0, FULL deck) ──");
    let t = std::time::Instant::now();
    let (v_a_table, layout_a) = compute_v_flop_at_root_iter0(
        canonical, &flop_tree, &combo_ranges_full, traverser);
    let secs_a = t.elapsed().as_secs_f64();
    let sum_a: f32 = v_a_table.iter().sum();
    let min_a: f32 = v_a_table.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_a: f32 = v_a_table.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("  {:.2}s, nh = {}, sum = {:.3e}, range = [{:.3e}, {:.3e}]",
        secs_a, v_a_table.len(), sum_a, min_a, max_a);
    eprintln!("  cfv[0..5] = {:?}",
        v_a_table[..5.min(v_a_table.len())].iter().map(|x| format!("{:.3e}", x)).collect::<Vec<_>>());

    // ── (B) production converged, FULL deck, K=10 ──
    eprintln!("\n── (B) compute_v_flop_at_root_converged (production, FULL deck, K=10) ──");
    let t = std::time::Instant::now();
    let (v_b_table, layout_b) = compute_v_flop_at_root_converged(
        canonical, &flop_tree, &combo_ranges_full, traverser, 10);
    let secs_b = t.elapsed().as_secs_f64();
    let sum_b: f32 = v_b_table.iter().sum();
    let min_b: f32 = v_b_table.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_b: f32 = v_b_table.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("  {:.2}s, nh = {}, sum = {:.3e}, range = [{:.3e}, {:.3e}]",
        secs_b, v_b_table.len(), sum_b, min_b, max_b);
    eprintln!("  cfv[0..5] = {:?}",
        v_b_table[..5.min(v_b_table.len())].iter().map(|x| format!("{:.3e}", x)).collect::<Vec<_>>());

    // ── (C) my subset path (matches 2.D.11) ──
    eprintln!("\n── (C) subset path (compute_flop_start_subset_with_decks + freeze+extract, K=10) ──");
    let t = std::time::Instant::now();
    let board: Vec<Card> = canonical.iter().copied().collect();
    let (chosen, turn_cards, river_decks) = pick_subset(canonical);
    let table_c = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &combo_ranges_full, np as u8, &chosen, &turn_cards, &river_decks);
    let nh_c = table_c.num_valid;
    let game_c = FlopStartGame::new(table_c);
    let mut solver = FlopStartVectorCfr::new(&flop_tree, game_c.table());
    let _ = solver.run(&flop_tree, &game_c, 10);
    solver.freeze_average_strategy_flop(&flop_tree);
    let reach = solver.compute_reach_flop(&flop_tree, &game_c);
    let nn = flop_tree.num_nodes();
    let mut cfv = vec![0.0f32; nn * nh_c];
    let params = DcfrParams::new(0);
    let table_ref = game_c.table();
    let turn_deck = table_ref.remaining_deck.clone();
    for (ti, &tc_card) in turn_deck.iter().enumerate() {
        let river_deck = &table_ref.river_decks[tc_card as usize];
        for ri in 0..river_deck.len() {
            solver.load_river_pair(ti, ri).unwrap();
            solver.freeze_average_strategy_for_river_pair(&flop_tree, ti, ri);
            solver.bottom_up_zone(&flop_tree, table_ref, traverser, &reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params);
            solver.save_river_pair(ti, ri).unwrap();
        }
    }
    for ti in 0..turn_deck.len() {
        solver.freeze_average_strategy_for_turn(&flop_tree, ti);
        solver.bottom_up_zone(&flop_tree, table_ref, traverser, &reach, &mut cfv,
            Zone::Turn, Some(ti), None, &params);
    }
    solver.bottom_up_zone(&flop_tree, table_ref, traverser, &reach, &mut cfv,
        Zone::Flop, None, None, &params);
    let v_c_table = cfv[0..nh_c].to_vec();
    let secs_c = t.elapsed().as_secs_f64();
    let sum_c: f32 = v_c_table.iter().sum();
    let min_c: f32 = v_c_table.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_c: f32 = v_c_table.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("  {:.2}s, nh = {}, sum = {:.3e}, range = [{:.3e}, {:.3e}]",
        secs_c, nh_c, sum_c, min_c, max_c);
    eprintln!("  cfv[0..5] = {:?}",
        v_c_table[..5.min(v_c_table.len())].iter().map(|x| format!("{:.3e}", x)).collect::<Vec<_>>());

    // Also: check the reach buffer at flop root for the subset path
    // (debug: is reach all zero, which would explain zero CFV?).
    eprintln!("\n── (C debug) reach at flop root, subset path ──");
    let nh = nh_c;
    eprintln!("  reach[p0][0..5] = {:?}",
        (0..5.min(nh)).map(|h| reach[0 * nh + h]).collect::<Vec<_>>());
    eprintln!("  reach[p1][0..5] = {:?}",
        (0..5.min(nh)).map(|h| reach[1 * nh + h]).collect::<Vec<_>>());
    eprintln!("  reach[p0] sum = {:.3e}, max = {:.3e}",
        reach[0..nh].iter().sum::<f32>(),
        reach[0..nh].iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    eprintln!("  reach[p1] sum = {:.3e}, max = {:.3e}",
        reach[nh..2*nh].iter().sum::<f32>(),
        reach[nh..2*nh].iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // ── Verdict ──
    eprintln!("\n=== VERDICT ===");
    let a_zero = sum_a.abs() < 1e-9 && max_a.abs() < 1e-9;
    let b_zero = sum_b.abs() < 1e-9 && max_b.abs() < 1e-9;
    let c_zero = sum_c.abs() < 1e-9 && max_c.abs() < 1e-9;
    eprintln!("  (A) production iter-0:     {}", if a_zero { "ZERO" } else { "NONZERO" });
    eprintln!("  (B) production converged:  {}", if b_zero { "ZERO" } else { "NONZERO" });
    eprintln!("  (C) subset converged:      {}", if c_zero { "ZERO" } else { "NONZERO" });
    eprintln!();
    if !a_zero && !b_zero && c_zero {
        eprintln!("→ TEST BUG: my subset extraction path returns zero while production paths");
        eprintln!("  return nonzero. The subset path (compute_flop_start_subset_with_decks +");
        eprintln!("  freeze+extract) has an indexing issue or subset interaction issue.");
        eprintln!("  Fix the subset path; then warm-K convergence test (#101 redesign) can");
        eprintln!("  proceed.");
    } else if a_zero && b_zero && c_zero {
        eprintln!("→ PRODUCTION BUG: all three paths return zero. The production CFV extraction");
        eprintln!("  is degenerate at this canonical+ranges. Real finding in the unified loop's");
        eprintln!("  CFV handoff — affects every blueprint that uses this canonical with these");
        eprintln!("  ranges.");
    } else if a_zero || b_zero {
        eprintln!("→ MIXED: one production path returned zero but not the other. Inspect");
        eprintln!("  which (A or B). Likely a freeze-related bug in the path that returned zero.");
    } else {
        eprintln!("→ Unexpected: at least one production path nonzero, subset path was claimed");
        eprintln!("  to be zero in #11b but now shows {} — re-check the #11b setup.",
            if c_zero { "ZERO" } else { "NONZERO" });
    }
    let _ = (layout_a, layout_b);
}
