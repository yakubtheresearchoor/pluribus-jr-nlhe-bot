// Step 2.D.11b: diagnostic — is the postflop subgame trivially convergent
// at our test config, or is the warm-K=1 = cold-K=200 result from #2d11
// a real warm-start convergence signal?
//
// Method: call flop_root_cfv with varying K (cold restart each time) at
// the SAME combo_ranges. Compare K=1 vs K=200 CFV directly.
//
// If K=1 ≈ K=200 → postflop converges in 1 iter at this config → 2.D.11
//   is non-informative at this scale; need harder postflop.
// If K=1 ≠ K=200 → postflop has real convergence dynamics; 2.D.11's
//   "all variants match" is a real warm-start signal.

use std::collections::HashMap;

use solver_core::card::{card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::solver::preflop_start_game::flop_combo_layout;
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

/// Cold call at K iters (no warm state). Returns flop-root CFV in
/// flop_combo_layout(canonical) order.
fn cold_flop_root_cfv_subset(
    flop_tree: &FlatTree,
    canonical: [Card; 3],
    combo_ranges_per_player: &[Vec<f32>],
    traverser: u8,
    iters: u32,
) -> Vec<f32> {
    let np = combo_ranges_per_player.len();
    let layout_engine = flop_combo_layout(canonical);
    let mut combo_ranges_full: Vec<Vec<f32>> =
        vec![vec![0.0f32; NUM_POSSIBLE_HANDS]; np];
    for p in 0..np {
        for (li, &(c1, c2)) in layout_engine.iter().enumerate() {
            combo_ranges_full[p][card_pair_to_index(c1, c2)] = combo_ranges_per_player[p][li];
        }
    }
    let board: Vec<Card> = canonical.iter().copied().collect();
    let (chosen, turn_cards, river_decks) = pick_subset(canonical);
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &combo_ranges_full, np as u8, &chosen, &turn_cards, &river_decks);
    let nh = table.num_valid;
    let layout_table: Vec<(Card, Card)> = (0..nh)
        .map(|i| (table.hand_cards[i * 2], table.hand_cards[i * 2 + 1]))
        .collect();
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let _ = solver.run(flop_tree, &game, iters);

    solver.freeze_average_strategy_flop(flop_tree);
    let reach = solver.compute_reach_flop(flop_tree, &game);
    let nn = flop_tree.num_nodes();
    let mut cfv = vec![0.0f32; nn * nh];
    let params = DcfrParams::new(0);
    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();
    for (ti, &tc_card) in turn_deck.iter().enumerate() {
        let river_deck = &table_ref.river_decks[tc_card as usize];
        for ri in 0..river_deck.len() {
            solver.load_river_pair(ti, ri).unwrap();
            solver.freeze_average_strategy_for_river_pair(flop_tree, ti, ri);
            solver.bottom_up_zone(flop_tree, table_ref, traverser, &reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params);
            solver.save_river_pair(ti, ri).unwrap();
        }
    }
    for ti in 0..turn_deck.len() {
        solver.freeze_average_strategy_for_turn(flop_tree, ti);
        solver.bottom_up_zone(flop_tree, table_ref, traverser, &reach, &mut cfv,
            Zone::Turn, Some(ti), None, &params);
    }
    solver.bottom_up_zone(flop_tree, table_ref, traverser, &reach, &mut cfv,
        Zone::Flop, None, None, &params);
    let v_table = cfv[0..nh].to_vec();
    // Diagnostic: return v_table DIRECTLY (no layout-engine reordering).
    // This shows the per-table-hand CFV at flop root.
    let _ = (layout_engine, layout_table);
    v_table
}

#[test]
#[ignore = "Step 2.D.11b: postflop convergence diagnostic at subset config"]
fn step2d11b_postflop_convergence_at_subset() {
    let flop_tree = build_tiny_flop_tree();
    eprintln!("\n=== Step 2.D.11b: postflop K-sensitivity at subset config ===");
    eprintln!("Tree: {} nodes (HU 1+1 stacks=10).", flop_tree.num_nodes());
    eprintln!("Subset: 8 hands × 2 turn × 2 river per canonical.");

    // A canonical from the first orbit.
    use solver_core::abstraction::flop_isomorphism::enumerate_canonical_flops;
    let canonicals = enumerate_canonical_flops();
    let canonical = canonicals[0];
    let np = 2;

    // Asymmetric combo_ranges (mimics what the preflop loop's reach produces).
    let layout = flop_combo_layout(canonical);
    let mut combo_ranges: Vec<Vec<f32>> = (0..np).map(|p| {
        layout.iter().enumerate().map(|(li, &(c1, c2))| {
            let h = (li as f32 * 0.1 + p as f32 * 0.3).sin() * 0.5 + 0.5;
            h
        }).collect()
    }).collect();
    // Don't normalize — production combo_ranges aren't normalized either.

    // Compute CFV at varying K, cold each time.
    let traverser = 0u8;
    let mut cfvs: Vec<(u32, Vec<f32>)> = Vec::new();
    for &k in &[1u32, 5, 10, 50, 200, 500] {
        let t = std::time::Instant::now();
        let cfv = cold_flop_root_cfv_subset(&flop_tree, canonical, &combo_ranges, traverser, k);
        let secs = t.elapsed().as_secs_f64();
        let sum: f32 = cfv.iter().sum();
        let max: f32 = cfv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min: f32 = cfv.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("  K={:>4} cold: {:>6.2} s, sum={:.4}, range=[{:.4}, {:.4}], cfv[0..5]={:?}",
            k, secs, sum, min, max,
            cfv[..5.min(cfv.len())].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
        cfvs.push((k, cfv));
    }

    // Compare each K to K=500 reference.
    eprintln!("\n=== Per-CFV deviation from K=500 reference ===");
    let ref_cfv = &cfvs.last().unwrap().1;
    eprintln!("{:>4}  {:>14}  {:>14}", "K", "max_abs", "rms");
    for (k, cfv) in &cfvs {
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        for i in 0..cfv.len().min(ref_cfv.len()) {
            let d = (cfv[i] - ref_cfv[i]).abs();
            if d > max_abs { max_abs = d; }
            sum_sq += (d as f64) * (d as f64);
        }
        let rms = (sum_sq / cfv.len() as f64).sqrt() as f32;
        eprintln!("{:>4}  {:>14.4e}  {:>14.4e}", k, max_abs, rms);
    }

    eprintln!("\n=== Interpretation ===");
    eprintln!("If K=1 ≈ K=500, postflop converges in 1 iter at this config → 2.D.11 was");
    eprintln!("  non-informative at this scale; need a more complex postflop subset to");
    eprintln!("  discriminate warm-K convergence.");
    eprintln!("If K=1 ≠ K=500, postflop has real convergence dynamics → 2.D.11's matching");
    eprintln!("  variants result is a real warm-start convergence signal.");
}
