/// Compare CFV at root between FlopStartVectorCfr and a manual recursive computation.
/// Both use uniform strategy (iter 0).
use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{FlopStartVectorCfr, Zone};
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

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
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).unwrap();
    let game = FlopStartGame::new(table);
    (tree, game)
}

/// Recursive CFV computation (no regret update) for a specific traverser.
fn compute_cfv_recursive(
    game: &FlopStartGame, tree: &FlatTree,
    traverser: u8, node_idx: usize,
    reach: &[Vec<f32>], tc_idx: Option<usize>, rc_idx: Option<usize>,
) -> Vec<f32> {
    let nh = game.num_hands(traverser);
    let np = 2;
    let node = &tree.nodes[node_idx];
    if node.is_terminal() {
        let cfreach = reach.to_vec();
        return game.evaluate_terminal(traverser, node_idx, tree, &cfreach);
    }
    if node.is_chance() {
        let children = tree.node_children(node_idx);
        let mut cfv = vec![0.0f32; nh];
        let n_outcomes = game.num_chance_outcomes();
        let child = if !children.is_empty() { children[0] as usize } else { return cfv; };
        for outcome in 0..n_outcomes {
            if outcome > 0 { game.clear_chance_outcome(); }
            let probs: Vec<f32> = (0..nh).map(|h| game.chance_probability(outcome, h)).collect();
            game.set_chance_outcome(outcome);
            let (new_tc, new_rc) = if node.board_state == 1 { (Some(outcome), rc_idx) }
                else if node.board_state == 2 { (tc_idx, Some(outcome)) } else { (tc_idx, rc_idx) };
            let child_cfv = compute_cfv_recursive(game, tree, traverser, child, reach, new_tc, new_rc);
            for h in 0..nh { cfv[h] += probs[h] * child_cfv[h]; }
        }
        game.clear_chance_outcome();
        return cfv;
    }
    let player = node.player_id;
    let na = node.num_children as usize;
    let children = tree.node_children(node_idx);
    // Use uniform strategy
    let sigma = vec![1.0 / na as f32; na * nh];
    let mut cfv_all: Vec<Vec<f32>> = Vec::with_capacity(na);
    for (a, &child) in children.iter().enumerate() {
        let mut new_reach = reach.to_vec();
        for h in 0..nh { new_reach[player as usize][h] = reach[player as usize][h] * sigma[a * nh + h]; }
        cfv_all.push(compute_cfv_recursive(game, tree, traverser, child as usize, &new_reach, tc_idx, rc_idx));
    }
    let mut cfv = vec![0.0f32; nh];
    if player == traverser {
        // Traverser: sigma-weighted sum
        for a in 0..na { for h in 0..nh { cfv[h] += sigma[a * nh + h] * cfv_all[a][h]; } }
    } else {
        // Non-traverser: unweighted sum (sigma in reach)
        for a in 0..na { for h in 0..nh { cfv[h] += cfv_all[a][h]; } }
    }
    cfv
}

#[test]
fn compare_cfv_uniform() {
    let (tree, game) = build_minimal();
    let nh = 4;
    let np = 2;

    // Recursive CFV for P0 traverser with uniform strategy
    let reach: Vec<Vec<f32>> = (0..np).map(|p| game.initial_weight(p as u8)).collect();
    let recursive_cfv = compute_cfv_recursive(&game, &tree, 0, 0, &reach, None, None);

    // FlopStartVectorCfr CFV
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());
    solver.compute_all_strategies(&tree);
    // Use best_response_value / strategy_value to get CFV-like values
    let br_val = solver.best_response_value(&tree, &game, 0);
    let sv_val = solver.strategy_value(&tree, &game, 0);

    println!("\n  CFV comparison (uniform strategy, P0 traverser):");
    println!("    Recursive CFV at root: {:?}", recursive_cfv);
    println!("    BR value: {:?}", br_val);
    println!("    SV value: {:?}", sv_val);
    println!("    BR - SV: {:?}", br_val.iter().zip(sv_val.iter()).map(|(b, s)| b - s).collect::<Vec<f32>>());

    // The recursive CFV should match... something
    // Actually, the recursive CFV is the COUNTERFACTUAL value for P0 traverser.
    // The SV value should also be the counterfactual value.
    // The BR value picks the best action at P0 nodes.
    let diff: f32 = recursive_cfv.iter().zip(sv_val.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("    |recursive - SV| sum: {:.6}", diff);
    let br_diff: f32 = recursive_cfv.iter().zip(br_val.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("    |recursive - BR| sum: {:.6}", br_diff);

    // Run FlopStartVectorCfr for 1 iter and check root regret
    solver.set_vanilla_mode(true);
    let root_cfv = solver.run(&tree, &game, 1);
    println!("\n  FlopStart run() root CFV: {:?}", root_cfv);

    // Check root regret
    let root_regret = solver.regrets_flop();
    println!("  Root regret after 1 iter: {:?}", &root_regret[0..8]);

    // Expected inst_regret from recursive: cfv[child_0] - cfv_avg
    // cfv_avg = 0.5 * cfv[child_0] + 0.5 * cfv[child_1]
    // inst_regret[0] = 0.5 * (cfv[child_0] - cfv[child_1])
    // But we need cfv at children, not at root.
    let reach_c0: Vec<Vec<f32>> = vec![
        vec![0.5; nh],  // P0 reach * 0.5
        vec![1.0; nh],  // P1 reach unchanged
    ];
    let cfv_child0 = compute_cfv_recursive(&game, &tree, 0, 1, &reach_c0, None, None);
    let reach_c1: Vec<Vec<f32>> = vec![
        vec![0.5; nh],
        vec![1.0; nh],
    ];
    let cfv_child1 = compute_cfv_recursive(&game, &tree, 0, 2, &reach_c1, None, None);
    println!("\n  Recursive CFV at child 0 (node 1): {:?}", cfv_child0);
    println!("  Recursive CFV at child 1 (node 2): {:?}", cfv_child1);
    let inst_regret: Vec<f32> = cfv_child0.iter().zip(cfv_child1.iter())
        .map(|(c0, c1)| 0.5 * (c0 - c1)).collect();
    println!("  Expected inst_regret[0]: {:?}", inst_regret);
    println!("  Actual regret[0]: {:?}", &root_regret[0..4]);
    println!("  Actual regret[1]: {:?}", &root_regret[4..8]);
}
