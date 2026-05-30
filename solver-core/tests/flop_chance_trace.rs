use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{FlopStartVectorCfr, Zone};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn find_pair(c1: Card, c2: Card) -> u16 {
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (a, b) = index_to_card_pair(idx);
        if (a == c1 as u8 && b == c2 as u8) || (a == c2 as u8 && b == c1 as u8) { return idx as u16; }
    }
    panic!("pair not found")
}

fn build_minimal() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
    let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let chosen_hands: Vec<u16> = vec![
        find_pair(card_from_str("Ah").unwrap(), card_from_str("Kh").unwrap()),
        find_pair(card_from_str("Qh").unwrap(), card_from_str("Jh").unwrap()),
        find_pair(card_from_str("Th").unwrap(), card_from_str("9h").unwrap()),
        find_pair(card_from_str("8h").unwrap(), card_from_str("6h").unwrap()),
    ];
    let nh = chosen_hands.len();
    let num_opp = 1;
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen_hands.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
    }
    let turn_cards: Vec<u8> = vec![card_from_str("3c").unwrap() as u8, card_from_str("4c").unwrap() as u8];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![card_from_str("5c").unwrap() as u8, card_from_str("6c").unwrap() as u8];
    river_decks[turn_cards[1] as usize] = vec![card_from_str("3c").unwrap() as u8, card_from_str("5c").unwrap() as u8];
    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for (i, &hi) in chosen_hands.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 { continue; }
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize).add_card(c2 as usize);
            for &bc in &board { hand = hand.add_card(bc as usize); }
            hand = hand.add_card(tc as usize);
            turn_ranks[tc as usize * nh + i] = hand.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (turn_ranks[tc as usize * nh + h] + 1, h as u16)).collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            let off = tc as usize * num_opp * nh + oi * nh;
            for h in 0..nh { turn_sorted_str[off + h] = items[h].0; turn_sorted_idx[off + h] = items[h].1; }
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for &rc in &river_decks[tc as usize] {
            let full_mask = turn_mask | (1u64 << rc);
            for (i, &hi) in chosen_hands.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if full_mask & (1u64 << c1) != 0 || full_mask & (1u64 << c2) != 0 { continue; }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize).add_card(c2 as usize);
                for &bc in &board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize).add_card(rc as usize);
                river_ranks[tc as usize * 52 * nh + rc as usize * nh + i] = hand.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| (river_ranks[tc as usize * 52 * nh + rc as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = tc as usize * 52 * num_opp * nh + rc as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let mut hand_ranks_base = vec![0u16; nh];
    for (i, &hi) in chosen_hands.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut hand = Hand::new();
        hand = hand.add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { hand = hand.add_card(bc as usize); }
        hand_ranks_base[i] = hand.evaluate_internal() as u16;
    }
    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh {
        for j in 0..nh {
            if i == j { conflict[i * nh + j] = 1; continue; }
            let (c1a, c1b) = index_to_card_pair(chosen_hands[i] as usize);
            let (c2a, c2b) = index_to_card_pair(chosen_hands[j] as usize);
            if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b { conflict[i * nh + j] = 1; }
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
        hand_ranks_base, valid_hand_indices: chosen_hands, num_valid: nh,
        conflict, hand_cards, remaining_deck: turn_cards.clone(),
        turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights, num_players: 2, num_combinations: nc, river_decks,
    };
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).unwrap();
    let game = FlopStartGame::new(table);
    (tree, game)
}

use solver_core::tree::flat::FlatTree;

#[test]
fn trace_chance() {
    let (tree, game) = build_minimal();
    let nn = tree.num_nodes();
    let nh = 4;

    println!("\nTree: {} nodes, {} hands", nn, nh);

    // Print ALL chance nodes and their children
    for i in 0..nn {
        let n = &tree.nodes[i];
        if n.is_chance() {
            let children = tree.node_children(i);
            println!("  Chance node {} (board_state={}): {} children = {:?}",
                i, n.board_state, children.len(), children);
        }
    }

    let solver = FlopStartVectorCfr::new(&tree, game.table());
    println!("\nriver_chance_children: {:?}", solver.river_chance_children());
    println!("turn_chance: (see zones array)");

    // Print all nodes with zone classification
    println!("\nAll nodes:");
    for i in 0..nn {
        let n = &tree.nodes[i];
        let zone = solver.zones()[i];
        let zone_str = match zone {
            Zone::Flop => "FLOP",
            Zone::Turn => "TURN",
            Zone::River => "RIVE",
        };
        if n.is_chance() {
            let children = tree.node_children(i);
            println!("  [{:>4}] {:>4} children={:?}",
                zone_str, i, children);
        } else if n.is_terminal() {
            println!("  [{:>4}] {:>4} TERMINAL", zone_str, i);
        } else {
            let children = tree.node_children(i);
            println!("  [{:>4}] {:>4} P{} children={:?}",
                zone_str, i, n.player_id, children);
        }
    }

    // Print turn zone local offsets
    println!("\nTurn zone nodes (local offsets):");
    for i in 0..nn {
        if solver.zones()[i] == Zone::Turn && !tree.nodes[i].is_chance() {
            let lo = solver.turn_local_offset()[i];
            println!("  Node {}: local_offset={}", i, lo);
        }
    }

    // Print river zone local offsets
    println!("\nRiver zone nodes (local offsets):");
    for i in 0..nn {
        if solver.zones()[i] == Zone::River && !tree.nodes[i].is_chance() {
            let lo = solver.turn_local_offset()[i];
            println!("  Node {}: local_offset={}", i, lo);
        }
    }
}
