#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::flat::{FlatNode, FlatTree, MAX_NA};
use solver_core::tree::action::BoardState;

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

fn make_board() -> Vec<Card> {
    ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect()
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

    tree.compute_levels();
    tree
}

#[test]
fn measure_drift_growth_rate() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let num_infosets = tree.num_infosets as usize;
    let data_len = num_infosets * MAX_NA * nh;

    let gpu = GpuContext::new().expect("GPU init failed");

    for &n_iters in &[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000] {
        let mut cpu = VectorCfr::new(&tree, vec![nh, nh]);
        cpu.run_sequential(&tree, &game, n_iters);
        let cpu_reg = cpu.regrets_slice().to_vec();
        let cpu_cum = cpu.cum_strategy_slice().to_vec();

        let mut gpu_solver = gpu
            .create_vcfr_solver(&tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight)
            .expect("vcfr solver creation failed");
        gpu_solver.run(n_iters).expect("GPU run failed");
        let gpu_reg = gpu_solver.download_regrets().expect("download failed");
        let gpu_cum = gpu_solver.download_cum_strategy().expect("download failed");

        let mut max_reg_diff = 0.0f32;
        let mut max_val = 0.0f32;
        for i in 0..data_len {
            let diff = (cpu_reg[i] - gpu_reg[i]).abs();
            max_reg_diff = max_reg_diff.max(diff);
            max_val = max_val.max(cpu_reg[i].abs()).max(gpu_reg[i].abs());
        }

        let mut max_cum_diff = 0.0f32;
        for i in 0..data_len {
            max_cum_diff = max_cum_diff.max((cpu_cum[i] - gpu_cum[i]).abs());
        }

        let rel_diff = if max_val > 0.0 { max_reg_diff / max_val } else { 0.0 };

        println!("iters={:5} max_reg_diff={:.6} max_val={:.1} rel_diff={:.2e} max_cum_diff={:.6}",
            n_iters, max_reg_diff, max_val, rel_diff, max_cum_diff);
    }
}
