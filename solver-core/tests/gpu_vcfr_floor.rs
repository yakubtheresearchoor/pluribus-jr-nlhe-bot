#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::flat::{FlatNode, FlatTree, MAX_NA};
use solver_core::tree::action::BoardState;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

fn make_board() -> Vec<Card> {
    ["2h", "7d", "Ks", "4c", "9s"]
        .iter().map(|s| card_from_str(s).unwrap()).collect()
}

fn build_river_tree() -> FlatTree {
    let mut tree = FlatTree::new(2, 10, vec![95, 95], 0.0, 0.0);
    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n0, 0, 5); tree.set_contribution(n0, 1, 5);
    let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n1, 0, 5); tree.set_contribution(n1, 1, 5);
    let n2 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n2, 0, 10); tree.set_contribution(n2, 1, 5);
    let n3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n3, 0, 5); tree.set_contribution(n3, 1, 5);
    let n4 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n4, 0, 5); tree.set_contribution(n4, 1, 10);
    let n5 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n5, 0, 10); tree.set_contribution(n5, 1, 5);
    let n6 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n6, 0, 10); tree.set_contribution(n6, 1, 10);
    let n7 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n7, 0, 5); tree.set_contribution(n7, 1, 10);
    let n8 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n8, 0, 10); tree.set_contribution(n8, 1, 10);
    tree.set_children(n0, vec![1, 2]);
    tree.set_children(n1, vec![3, 4]);
    tree.set_children(n2, vec![5, 6]);
    tree.set_children(n4, vec![7, 8]);
    tree.set_folded_mask(n5, 0b10);
    tree.set_folded_mask(n7, 0b01);
    tree.compute_levels();
    tree
}

#[test]
fn regret_magnitude_at_floor() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();
    let num_infosets = tree.num_infosets as usize;
    let data_len = num_infosets * MAX_NA * nh;

    println!("nh={}, floor=-1e7", nh);
    println!("{:>6} {:>12} {:>12} {:>8} {:>8} {:>8}", 
        "iters", "min_regret", "max_regret", "at_floor", "abs_max", "T*(T+1)/2*5400");

    for &n_iters in &[1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 500] {
        let mut cpu = VectorCfr::new(&tree, vec![nh, nh]);
        cpu.run_sequential(&tree, &game, n_iters);
        let regrets = cpu.regrets_slice();

        let mut min_r = f32::MAX;
        let mut max_r = f32::MIN;
        let mut at_floor = 0;
        for i in 0..data_len {
            let v = regrets[i];
            min_r = min_r.min(v);
            max_r = max_r.max(v);
            if v <= -1e7 + 1.0 { at_floor += 1; }
        }

        let predicted = n_iters as f64 * (n_iters as f64 + 1.0) / 2.0 * 5400.0;

        println!("{:>6} {:>12.1} {:>12.1} {:>8} {:>8.1} {:>8.0}",
            n_iters, min_r, max_r, at_floor, max_r.abs().max(min_r.abs()), predicted);
    }
}
