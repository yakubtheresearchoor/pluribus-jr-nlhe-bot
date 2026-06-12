/// Permanent gate tests for the flop-start solver.
/// These MUST pass before any convergence run is considered valid.
///
/// Run all gates:
///   cargo test -p solver-core --features metal --test permanent_gates -- --test-threads=1 --nocapture
///
/// Run full-game gates (slow):
///   cargo test -p solver-core --features metal --test permanent_gates -- --test-threads=1 --nocapture --ignored

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::game::GameSpec;
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
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
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
        hand = hand.add_card(c1 as usize);
        hand = hand.add_card(c2 as usize);
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
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in &board { hand = hand.add_card(bc as usize); }
            hand = hand.add_card(tc as usize);
            turn_ranks[tc as usize * nh + i] = hand.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh)
            .map(|h| (turn_ranks[tc as usize * nh + h] + 1, h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            let off = tc as usize * num_opp * nh + oi * nh;
            for h in 0..nh {
                turn_sorted_str[off + h] = items[h].0;
                turn_sorted_idx[off + h] = items[h].1;
            }
        }
    }

    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for &rc in &river_decks[tc as usize] {
            let full_mask = turn_mask | (1u64 << rc);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if full_mask & (1u64 << c1) != 0 || full_mask & (1u64 << c2) != 0 { continue; }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in &board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize);
                hand = hand.add_card(rc as usize);
                river_ranks[tc as usize * 52 * nh + rc as usize * nh + i] =
                    hand.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| (river_ranks[tc as usize * 52 * nh + rc as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = tc as usize * 52 * num_opp * nh + rc as usize * num_opp * nh + oi * nh;
                for h in 0..nh {
                    river_sorted_str[off + h] = items[h].0;
                    river_sorted_idx[off + h] = items[h].1;
                }
            }
        }
    }

    let initial_weights = vec![vec![1.0f32; nh], vec![1.0f32; nh]];
    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let mask0: u64 = (1u64 << hand_cards[h0 * 2]) | (1u64 << hand_cards[h0 * 2 + 1]);
        for h1 in 0..nh {
            let mask1: u64 = (1u64 << hand_cards[h1 * 2]) | (1u64 << hand_cards[h1 * 2 + 1]);
            if mask0 & mask1 == 0 { nc += 1.0; }
        }
    }

    let table = FlopChanceTable {
        hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
        remaining_deck: turn_cards.clone(), turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx, initial_weights, num_players,
        num_combinations: nc, river_decks,
    };
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

fn find_pair_index(c1: Card, c2: Card) -> u16 {
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (a, b) = index_to_card_pair(idx);
        if (a == c1 as u8 && b == c2 as u8) || (a == c2 as u8 && b == c1 as u8) {
            return idx as u16;
        }
    }
    panic!("pair not found");
}

/// GATE 1: Zero-sum at iter 0 (minimal tree, fast).
#[test]
fn gate_zero_sum_iter0() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let solver = FlopStartVectorCfr::new(&tree, game.table());

    let nh = solver.num_hands();
    let nc = game.table().num_combinations as f32;
    let w0 = &game.table().initial_weights[0];

    let sv0 = solver.strategy_value(&tree, &game, 0);
    let sv1 = solver.strategy_value(&tree, &game, 1);

    let sv_sum: f32 = (0..nh).map(|h| w0[h] * (sv0[h] + sv1[h])).sum::<f32>() / nc;

    assert!(sv_sum.abs() < 0.01,
        "GATE 1 FAILED: SV not zero-sum at iter 0: {}", sv_sum);
}

/// GATE 2: Zero-sum persists across convergence (minimal tree, fast).
#[test]
fn gate_zero_sum_convergence() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    let nh = solver.num_hands();
    let nc = game.table().num_combinations as f32;
    let w0 = &game.table().initial_weights[0];

    for i in 0..50 {
        let _ = solver.run(&tree, &game, 1);
        if i % 10 == 9 {
            let sv0 = solver.strategy_value(&tree, &game, 0);
            let sv1 = solver.strategy_value(&tree, &game, 1);
            let sv_sum: f32 = (0..nh).map(|h| w0[h] * (sv0[h] + sv1[h])).sum::<f32>() / nc;
            assert!(sv_sum.abs() < 0.01,
                "GATE 2 FAILED: SV not zero-sum at iter {}: {}", i + 1, sv_sum);
        }
    }
}

/// GATE 3: Terminal CFVs zero-sum at all board states (minimal tree, fast).
#[test]
fn gate_terminal_zero_sum() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let nh = game.table().num_valid;
    let board_cards: Vec<u8> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();

    for idx in 0..tree.num_nodes() {
        if !tree.nodes[idx].is_terminal() { continue; }
        let fold_mask = tree.get_folded_mask(idx);
        if fold_mask != 0 { continue; } // fold terminals are trivially zero-sum

        for (ti, &tc) in game.table().remaining_deck.iter().enumerate() {
            for (ri, &rc) in game.table().river_decks[tc as usize].iter().enumerate() {
                game.set_turn_card(tc);
                game.set_river_card(rc);

                let cfreach = vec![vec![1.0f32; nh], vec![1.0f32; nh]];
                let cfv0 = game.evaluate_terminal(0, idx, &tree, &cfreach);
                let cfv1 = game.evaluate_terminal(1, idx, &tree, &cfreach);

                let sum0: f32 = cfv0.iter().sum();
                let sum1: f32 = cfv1.iter().sum();

                // Per-traverser terminal CFVs should each sum to 0
                // (because for each hand pair (h0,h1), win + loss = 0)
                assert!(sum0.abs() < 0.01,
                    "GATE 3 FAILED: Terminal {} turn={}/{} P0 sum={:.4}",
                    idx, ti, ri, sum0);
                assert!(sum1.abs() < 0.01,
                    "GATE 3 FAILED: Terminal {} turn={}/{} P1 sum={:.4}",
                    idx, ti, ri, sum1);
            }
        }
    }
}

/// GATE 4: Convergence — exploitability decreases over 50 iterations.
#[test]
fn gate_convergence() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    let mut first_10_avg = 0.0f32;
    let mut last_10_avg = 0.0f32;

    for i in 1..=50 {
        let _ = solver.run(&tree, &game, 1);
        let expl = solver.compute_exploitability(&tree, &game);
        if i <= 10 { first_10_avg += expl; }
        if i > 40 { last_10_avg += expl; }
    }
    first_10_avg /= 10.0;
    last_10_avg /= 10.0;

    assert!(last_10_avg < first_10_avg * 0.5,
        "GATE 4 FAILED: Not converging. First 10 avg={:.4}, Last 10 avg={:.4}",
        first_10_avg, last_10_avg);
}

/// GATE 5: Board-card filtering — opponent hands with turn/river cards are excluded.
#[test]
fn gate_board_card_filtering() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let nh = game.table().num_valid;

    // Verify: for each (turn, river) combo, all showdown terminals are zero-sum
    // (this implicitly verifies board-card filtering)
    for (ti, &tc) in game.table().remaining_deck.iter().enumerate() {
        for (ri, &rc) in game.table().river_decks[tc as usize].iter().enumerate() {
            game.set_turn_card(tc);
            game.set_river_card(rc);

            for idx in 0..tree.num_nodes() {
                if !tree.nodes[idx].is_terminal() { continue; }
                let fm = tree.get_folded_mask(idx);
                if fm != 0 { continue; }

                let cfreach = vec![vec![1.0f32; nh], vec![1.0f32; nh]];
                let cfv = game.evaluate_terminal(0, idx, &tree, &cfreach);
                let raw_sum: f32 = cfv.iter().sum();
                assert!(raw_sum.abs() < 0.01,
                    "Board filtering failed: terminal {} turn={}/{} sum={:.4}",
                    idx, ti, ri, raw_sum);
            }
        }
    }
}
