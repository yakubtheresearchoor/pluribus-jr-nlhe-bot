/// Direct CFV comparison between SimpleCfr and FlopStartVectorCfr at the root.
use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{FlopStartVectorCfr, Zone};
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::collections::HashMap;

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

/// Check: for traverser P0 with uniform strategy, what are the CFVs at root children?
#[test]
fn trace_root_cfv() {
    let (tree, game) = build_minimal();
    let nh = 4;

    // SimpleCfr: compute CFV at root's children by calling cfr_recursive with dummy traverser
    // that doesn't update regrets.
    let np = 2;
    let reach: Vec<Vec<f32>> = (0..np).map(|p| game.initial_weight(p as u8)).collect();

    // Just compute CFV using evaluate_terminal-like walk without updating regrets
    // For uniform strategy at all nodes, compute CFV for traverser P0 at root children 1 and 2.

    // Actually, let's just use FlopStartVectorCfr's compute_reach + a manual CFV walk.
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());
    solver.compute_all_strategies(&tree);

    let flop_reach = solver.compute_reach_flop(&tree, &game);
    let nn = tree.num_nodes();

    // Print reach at root's children
    for &child in &tree.node_children(0) {
        let off = child as usize * np * nh;
        println!("\n  Reach at node {} (child of root):", child);
        for p in 0..np {
            println!("    P{}: {:?}", p, &flop_reach[off + p * nh..off + (p + 1) * nh]);
        }
    }

    // Print reach at flop zone terminals
    println!("\n  Flop zone terminals:");
    for i in 0..nn {
        if solver.zones()[i] == Zone::Flop && tree.nodes[i].is_terminal() {
            let off = i * np * nh;
            let p0_reach = &flop_reach[off..off + nh];
            let p1_reach = &flop_reach[off + nh..off + 2 * nh];
            println!("    Node {}: P0_reach={:?} P1_reach={:?}",
                i, p0_reach, p1_reach);
        }
    }

    // Now run bottom_up_zone for the flop zone with traverser P0
    // and see the CFV at root children
    let traverser: u8 = 0;
    let params = solver_core::solver::dcfr_params::DcfrParams { alpha_t: 1.0, beta_t: 1.0, gamma_t: 1.0 };
    let mut cfv = vec![0.0f32; nn * nh];

    // Seed CFV at turn chance children from accumulated turn/river CFVs
    // For iter 0 with uniform strategy, let's just run the full run()
    let _root_cfv = solver.run(&tree, &game, 1);

    // Now check the exploitability
    let expl = solver.compute_exploitability(&tree, &game);
    println!("\n  After 1 iter (vanilla): exploitability = {:.6}", expl);

    // Let's also compute exploitability using the simple recursive approach
    // with the same regrets to check if the measurement agrees.

    // Check terminal CFV at a specific terminal using the game's evaluate_terminal
    // For node 134 (flop zone terminal, child of node 2 action 0)
    let node_134_reach_base = 134 * np * nh;
    let cfreach: Vec<Vec<f32>> = (0..np)
        .map(|p| {
            let off = node_134_reach_base + p * nh;
            flop_reach[off..off + nh].to_vec()
        })
        .collect();
    let terminal_cfv = game.evaluate_terminal(0, 134, &tree, &cfreach);
    println!("\n  Terminal 134 CFV (evaluate_terminal, P0 traverser): {:?}", terminal_cfv);
}
