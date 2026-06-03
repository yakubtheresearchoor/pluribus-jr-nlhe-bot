// Focused diagnostic: why does CPU bottom_up_zone return 0 at node[13]
// for the HU symmetric [5,5] river-showdown terminal where the hand-trace
// proved the correct value is [0.234, 0.078, -0.078, -0.234]?
//
// Check the reach values at node[13] when compute_reach_river(ti=0, ri=0)
// is run, and call bottom_up_zone for the river zone to confirm/reproduce
// the divergence.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{FlopStartVectorCfr, Zone, DcfrParams};
use solver_core::solver::showdown::{side_pot_showdown_cfv, sorted_sweep_showdown};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_minimal_table() -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
    let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let chosen_hands: Vec<u16> = vec![
        find_pair_index(card_from_str("Ah").unwrap(), card_from_str("Kh").unwrap()),
        find_pair_index(card_from_str("Qh").unwrap(), card_from_str("Jh").unwrap()),
        find_pair_index(card_from_str("Th").unwrap(), card_from_str("9h").unwrap()),
        find_pair_index(card_from_str("8h").unwrap(), card_from_str("6h").unwrap()),
    ];
    let nh = chosen_hands.len();
    let num_players = 2u8;
    let num_opp = 1;
    let valid_hand_indices = chosen_hands.clone();
    let num_valid = nh;
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in valid_hand_indices.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1; hand_cards[i * 2 + 1] = c2;
    }
    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh {
        for j in 0..nh {
            if i == j { conflict[i * nh + j] = 1; continue; }
            let (c1a, c1b) = index_to_card_pair(valid_hand_indices[i] as usize);
            let (c2a, c2b) = index_to_card_pair(valid_hand_indices[j] as usize);
            if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b {
                conflict[i * nh + j] = 1;
            }
        }
    }
    let mut hand_ranks_base = vec![0u16; nh];
    for (i, &hi) in valid_hand_indices.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut hand = Hand::new();
        hand = hand.add_card(c1 as usize); hand = hand.add_card(c2 as usize);
        for &bc in &board { hand = hand.add_card(bc as usize); }
        hand_ranks_base[i] = hand.evaluate_internal() as u16;
    }
    let turn_cards: Vec<u8> = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("4c").unwrap() as u8,
    ];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("5c").unwrap() as u8,
        card_from_str("6c").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("5c").unwrap() as u8,
    ];
    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 { continue; }
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize); hand = hand.add_card(c2 as usize);
            for &bc in &board { hand = hand.add_card(bc as usize); }
            hand = hand.add_card(tc as usize);
            turn_ranks[tc as usize * nh + i] = hand.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh).filter(|&h| {
            let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
            turn_mask & (1u64 << c1) == 0 && turn_mask & (1u64 << c2) == 0
        }).map(|h| (turn_ranks[tc as usize * nh + h], h as u16)).collect();
        items.sort_by_key(|&(s, _)| s); // ASCENDING — matches production FlopStartGame::new
        for (k, &(r, idx)) in items.iter().enumerate() {
            turn_sorted_str[(tc as usize) * num_opp * nh + 0 * nh + k] = r;
            turn_sorted_idx[(tc as usize) * num_opp * nh + 0 * nh + k] = idx;
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        for &rc in &river_decks[tc as usize] {
            let combined = board_mask | (1u64 << tc) | (1u64 << rc);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if combined & (1u64 << c1) != 0 || combined & (1u64 << c2) != 0 { continue; }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize); hand = hand.add_card(c2 as usize);
                for &bc in &board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize); hand = hand.add_card(rc as usize);
                let r = hand.evaluate_internal() as u16;
                let key = (tc as usize) * 52 + (rc as usize);
                river_ranks[key * nh + i] = r;
            }
            let key = (tc as usize) * 52 + (rc as usize);
            let mut items: Vec<(u16, u16)> = (0..nh).filter(|&h| {
                let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
                combined & (1u64 << c1) == 0 && combined & (1u64 << c2) == 0
            }).map(|h| (river_ranks[key * nh + h], h as u16)).collect();
            items.sort_by_key(|&(s, _)| s); // ASCENDING — matches production FlopStartGame::new
            for (k, &(r, idx)) in items.iter().enumerate() {
                river_sorted_str[key * num_opp * nh + 0 * nh + k] = r;
                river_sorted_idx[key * num_opp * nh + 0 * nh + k] = idx;
            }
        }
    }
    let initial_weights: Vec<Vec<f32>> = (0..num_players).map(|_| {
        let mut w = vec![0.0f32; nh];
        for h in 0..nh {
            let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
            let mut blocked = 0;
            for h2 in 0..nh {
                if h2 == h { continue; }
                let (c3, c4) = index_to_card_pair(valid_hand_indices[h2] as usize);
                if c1 == c3 || c1 == c4 || c2 == c3 || c2 == c4 { blocked += 1; }
            }
            w[h] = if blocked < (nh - 1) as i32 { 1.0 } else { 0.0 };
        }
        w
    }).collect();
    let num_combinations = initial_weights[0].iter().sum::<f32>() * initial_weights[1].iter().sum::<f32>();
    let table = FlopChanceTable {
        hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
        remaining_deck: turn_cards, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights, num_players,
        num_combinations: num_combinations as f64, river_decks,
    };
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

fn find_pair_index(c1: Card, c2: Card) -> u16 {
    let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
    let mut idx = 0u16;
    for i in 0..52u8 {
        for j in (i+1)..52u8 {
            if i == lo && j == hi { return idx; }
            idx += 1;
        }
    }
    panic!("pair not found");
}

#[test]
fn diagnose_cpu_node13_zero_cfv() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let nh = 4usize;
    let np = 2usize;
    let nn = tree.num_nodes();

    cpu.compute_all_strategies(&tree);

    let flop_reach = cpu.compute_reach_flop(&tree, &game);
    let turn_reach_0 = cpu.compute_reach_turn(&tree, 0, &flop_reach);
    let river_reach_00 = cpu.compute_reach_river(&tree, 0, 0, &turn_reach_0);

    // Dump reach at node[13] for both players from each stage.
    let base = 13 * np * nh;
    let p0_flop: Vec<f32> = (0..nh).map(|h| flop_reach[base + 0 * nh + h]).collect();
    let p1_flop: Vec<f32> = (0..nh).map(|h| flop_reach[base + 1 * nh + h]).collect();
    let p0_turn0: Vec<f32> = (0..nh).map(|h| turn_reach_0[base + 0 * nh + h]).collect();
    let p1_turn0: Vec<f32> = (0..nh).map(|h| turn_reach_0[base + 1 * nh + h]).collect();
    let p0_river00: Vec<f32> = (0..nh).map(|h| river_reach_00[base + 0 * nh + h]).collect();
    let p1_river00: Vec<f32> = (0..nh).map(|h| river_reach_00[base + 1 * nh + h]).collect();

    eprintln!("\n=== Reach at node[13] for traverser=0, ti=0, ri=0 ===");
    eprintln!("flop_reach:    p0={:?} p1={:?}", p0_flop, p1_flop);
    eprintln!("turn_reach[0]: p0={:?} p1={:?}", p0_turn0, p1_turn0);
    eprintln!("river_reach[0,0]: p0={:?} p1={:?}", p0_river00, p1_river00);

    // Hand-trace expected reach at node 13:
    // Path: 0:K (p0) → 1:K (p1) → 3:+ → 5:K (p0) → 6:K (p1) → 8:+ → 10:K (p0) → 11:K (p1)
    // p0 acts 3 times (nodes 0, 5, 10), p1 acts 3 times (nodes 1, 6, 11). All sigma=1/2.
    // Initial weight = 1.0 for all 4 hands (no card conflicts).
    // So p0_reach = 1.0 * (1/2)^3 = 0.125 for all hands.
    // p1_reach = 0.125 for all hands.
    // Chance nodes don't change reach (they just propagate).
    eprintln!("\nExpected hand-traced reach at node[13]: 0.125 for both players, all hands.");

    // Now run CPU bottom_up_zone(Zone::River, ti=0, ri=0, traverser=0) and check
    // the cfv at node[13].
    let params = DcfrParams::new(0);
    let mut cfv = vec![0.0f32; nn * nh];
    cpu.bottom_up_zone(
        &tree, game.table(), 0,
        &river_reach_00, &mut cfv,
        Zone::River, Some(0), Some(0),
        &params,
    );
    let cpu_cfv_n13: Vec<f32> = (0..nh).map(|h| cfv[13 * nh + h]).collect();
    eprintln!("\nCPU bottom_up_zone(River, 0, 0, traverser=0) cfv at node[13]: {:?}", cpu_cfv_n13);
    eprintln!("Hand-trace expected: [0.234, 0.078, -0.078, -0.234] (GPU matches this)");

    // If CPU returns 0, hypothesis: river_reach is 0 at node[13].
    let any_reach_nonzero = p0_river00.iter().any(|&v| v != 0.0)
        || p1_river00.iter().any(|&v| v != 0.0);
    eprintln!("\nDiagnosis: river_reach at node[13] nonzero? {}", any_reach_nonzero);

    // Reach is correct. Bug must be downstream. Call side_pot_showdown_cfv
    // directly with exact arguments bottom_up_zone would build for node[13].
    let opp_reach_p1: Vec<f32> = p1_river00.clone();
    let opp_reach_views: Vec<&[f32]> = vec![&opp_reach_p1];
    let contributions: Vec<i32> = (0..np)
        .map(|p| tree.get_contribution(13, p as u8))
        .collect();
    let fold_mask = tree.get_folded_mask(13);

    let table_ref = game.table();
    let tc_card = table_ref.remaining_deck[0];
    let rc_card = table_ref.river_decks[tc_card as usize][0];
    let (opp_str, opp_idx, pl_str, pl_idx) = table_ref.river_sorted_arrays(tc_card, rc_card);

    eprintln!("\n=== Sorted arrays inspection at (tc=3c={}, rc=5c={}) ===", tc_card, rc_card);
    eprintln!("opp_str ({} elements): {:?}", opp_str.len(), opp_str);
    eprintln!("opp_idx ({} elements): {:?}", opp_idx.len(), opp_idx);
    eprintln!("pl_str  ({} elements): {:?}", pl_str.len(), pl_str);
    eprintln!("pl_idx  ({} elements): {:?}", pl_idx.len(), pl_idx);
    eprintln!("hand_cards: {:?}", &table_ref.hand_cards);
    eprintln!("contributions[node13]: {:?}", contributions);
    eprintln!("fold_mask[node13]: {:b}", fold_mask);
    eprintln!("starting_pot: {}", tree.starting_pot);

    let cfv_out = side_pot_showdown_cfv(
        &opp_reach_views, &table_ref.hand_cards, nh,
        opp_str, opp_idx, pl_str, pl_idx,
        &contributions, fold_mask, 0, 2,
        tree.starting_pot,
    );
    let nc = table_ref.num_combinations as f32;
    let cfv_normalized: Vec<f32> = cfv_out.iter().map(|&v| v / nc).collect();
    eprintln!("\nDirect side_pot_showdown_cfv call:");
    eprintln!("  raw cfv_out = {:?}", cfv_out);
    eprintln!("  / num_combinations({}) = {:?}", nc, cfv_normalized);
    eprintln!("  hand-trace expected: [0.234, 0.078, -0.078, -0.234]");
    eprintln!("  GPU produces:        [0.234, 0.078, -0.078, -0.234]");

    if cfv_out == vec![0.0; nh] {
        eprintln!("  → side_pot_showdown_cfv returns ZEROS. Bug is IN this function.");
    } else if cfv_normalized.iter().zip([0.234375f32, 0.078125, -0.078125, -0.234375].iter())
        .all(|(a, b)| (a - b).abs() < 1e-4)
    {
        eprintln!("  → side_pot_showdown_cfv returns CORRECT values directly!");
        eprintln!("  → Bug is in how bottom_up_zone WIRES the inputs (passes wrong args).");
    } else {
        eprintln!("  → side_pot_showdown_cfv returns nonzero but doesn't match expected.");
        eprintln!("  → Need to investigate which args differ from what bottom_up_zone passes.");
    }

    // Call sorted_sweep_showdown DIRECTLY (one level deeper than
    // side_pot_showdown_cfv) to see if the sweep returns 0 or nonzero.
    let sweep_out = sorted_sweep_showdown(
        &opp_reach_views, &table_ref.hand_cards, nh,
        opp_str, opp_idx, pl_str, pl_idx,
    );
    eprintln!("\nDirect sorted_sweep_showdown call (DESCENDING-sorted input):");
    eprintln!("  raw sweep_out = {:?}", sweep_out);
    eprintln!("  expected: [+0.375, +0.125, -0.125, -0.375]");

    // Verify the bug: flip the sort direction (descending → ascending) and
    // re-run. If the sweep then produces correct values, the algorithm is
    // designed for ASCENDING-sorted inputs but the table populates DESCENDING.
    let mut asc_str: Vec<u16> = pl_str.to_vec();
    let mut asc_idx: Vec<u16> = pl_idx.to_vec();
    // Sort ascending by strength while keeping idx aligned.
    let mut pairs: Vec<(u16, u16)> = asc_str.iter().zip(asc_idx.iter()).map(|(a, b)| (*a, *b)).collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    for (i, (s, idx)) in pairs.iter().enumerate() {
        asc_str[i] = *s;
        asc_idx[i] = *idx;
    }
    eprintln!("\n  Re-sorted ASCENDING: str={:?} idx={:?}", asc_str, asc_idx);
    let sweep_asc = sorted_sweep_showdown(
        &opp_reach_views, &table_ref.hand_cards, nh,
        &asc_str, &asc_idx, &asc_str, &asc_idx,
    );
    eprintln!("  sweep_out (ascending) = {:?}", sweep_asc);
    if sweep_asc.iter().zip([0.375f32, 0.125, -0.125, -0.375].iter())
        .all(|(a, b)| (a - b).abs() < 1e-4)
    {
        eprintln!("  ✓ CONFIRMED: sweep works correctly with ASCENDING input.");
        eprintln!("  → Bug: sweep algorithm designed for ASCENDING; table populates DESCENDING.");
        eprintln!("  → Fix options:");
        eprintln!("    A) Change table setup to sort ascending (affects GPU which uses brute-force; benign)");
        eprintln!("    B) Change sweep algorithm to handle descending (more contained)");
    }
}
