#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
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
fn exploitability_convergence_trajectory() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let gpu = GpuContext::new().expect("GPU init failed");

    println!("{:>6} {:>12} {:>12} {:>10}", "iters", "cpu_exp", "gpu_exp", "rel_gap");
    for &n_iters in &[100, 200, 500, 1000, 2000, 5000, 10000] {
        let mut cpu = VectorCfr::new(&tree, vec![nh, nh]);
        cpu.run_sequential(&tree, &game, n_iters);
        let cpu_profile = StrategyProfile::from_usize_offsets(
            cpu.cum_strategy_slice(), cpu.node_offsets(), nh,
        );
        let cpu_exp = exploitability(&tree, &game, &cpu_profile);

        let mut gpu_solver = gpu
            .create_vcfr_solver(&tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None)
            .expect("vcfr solver creation failed");
        gpu_solver.run(n_iters).expect("GPU run failed");
        let cum = gpu_solver.download_cum_strategy().expect("download failed");
        let data_per_infoset = MAX_NA * nh;
        let node_offsets: Vec<usize> = (0..tree.num_nodes())
            .map(|i| {
                let is = tree.infoset_offsets[i];
                if is == u32::MAX { usize::MAX } else { is as usize * data_per_infoset }
            })
            .collect();
        let gpu_profile = StrategyProfile::from_usize_offsets(&cum, &node_offsets, nh);
        let gpu_exp = exploitability(&tree, &game, &gpu_profile);

        let rel_gap = if cpu_exp > 0.0 { (gpu_exp - cpu_exp) / cpu_exp } else { 0.0 };
        println!("{:>6} {:>12.6} {:>12.6} {:>10.4}", n_iters, cpu_exp, gpu_exp, rel_gap);
    }
}
