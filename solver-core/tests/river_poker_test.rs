use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::action::BoardState;
use solver_core::tree::flat::{FlatNode, FlatTree};

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

fn build_river_tree() -> FlatTree {
    let mut tree = FlatTree::new(2, 10, vec![95, 95], 0.0, 0.0);

    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n0, 0, 5);
    tree.set_contribution(n0, 1, 5);

    let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n1, 0, 5);
    tree.set_contribution(n1, 1, 5);

    let n2 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n2, 0, 10);
    tree.set_contribution(n2, 1, 5);

    let n3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n3, 0, 5);
    tree.set_contribution(n3, 1, 5);

    let n4 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n4, 0, 5);
    tree.set_contribution(n4, 1, 10);

    let n5 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n5, 0, 10);
    tree.set_contribution(n5, 1, 5);

    let n6 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n6, 0, 10);
    tree.set_contribution(n6, 1, 10);

    let n7 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n7, 0, 5);
    tree.set_contribution(n7, 1, 10);

    let n8 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n8, 0, 10);
    tree.set_contribution(n8, 1, 10);

    tree.set_children(n0, vec![1, 2]);
    tree.set_children(n1, vec![3, 4]);
    tree.set_children(n2, vec![5, 6]);
    tree.set_children(n4, vec![7, 8]);

    tree.set_folded_mask(n5, 0b10);
    tree.set_folded_mask(n7, 0b01);

    assert_eq!(tree.num_nodes(), 9);
    tree
}

#[test]
fn river_poker_cpu_smoke() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    println!("Valid hands: {}", nh);

    let tree = build_river_tree();
    let mut solver = CpuMccfr::new(&tree, vec![nh, nh]);

    let root_cfv = solver.run(&tree, &game, 100);

    let avg_strat = solver.get_average_strategy(0, 2, nh);

    let game_value: f32 = root_cfv.iter().sum::<f32>() / nh as f32;
    println!("Game value (avg cfv): {:.4}", game_value);

    let avg_bet_prob: f32 = avg_strat[1].iter().sum::<f32>() / nh as f32;
    println!("Average bet probability at root: {:.4}", avg_bet_prob);

    assert!(
        game_value.abs() < 500.0,
        "game value {} seems out of range",
        game_value
    );
    assert!(
        avg_bet_prob > 0.01 && avg_bet_prob < 0.99,
        "bet prob {} seems degenerate",
        avg_bet_prob
    );
}

#[test]
fn river_poker_strong_hands_bet_more() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    let tree = build_river_tree();
    let mut solver = CpuMccfr::new(&tree, vec![nh, nh]);
    let _root_cfv = solver.run(&tree, &game, 500);

    let strat = solver.get_current_strategy(0, 2, nh);

    let hand_ranks = game.hand_ranks_gpu();
    let mut ranked: Vec<(u16, f32)> = (0..nh).map(|i| (hand_ranks[i], strat[1][i])).collect();
    ranked.sort_by_key(|&(r, _)| std::cmp::Reverse(r));

    let top_quarter = nh / 4;
    let bottom_quarter = nh / 4;
    let top_bet: f32 = ranked[..top_quarter].iter().map(|(_, b)| *b).sum::<f32>() / top_quarter as f32;
    let bottom_bet: f32 = ranked[nh - bottom_quarter..].iter().map(|(_, b)| *b).sum::<f32>() / bottom_quarter as f32;

    println!("Top 25% bet prob: {:.4}", top_bet);
    println!("Bottom 25% bet prob: {:.4}", bottom_bet);

    assert!(
        top_bet > bottom_bet,
        "strong hands should bet more: top={:.4} bottom={:.4}",
        top_bet, bottom_bet
    );
}
