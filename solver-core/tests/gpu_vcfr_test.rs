#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
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
fn gpu_vcfr_river_single_iter_bit_exact() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let (opp_str, opp_idx, pl_str, pl_idx, _same_hand) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let mut cpu = VectorCfr::new(&tree, vec![nh, nh]);
    cpu.run_sequential(&tree, &game, 1);
    let cpu_regrets = cpu.regrets_slice().to_vec();
    let cpu_cum = cpu.cum_strategy_slice().to_vec();

    let gpu = GpuContext::new().expect("GPU init failed");
    let mut gpu_solver = gpu
        .create_vcfr_solver(
            &tree,
            nh,
            &opp_str,
            &opp_idx,
            &pl_str,
            &pl_idx,
            &hand_cards,
            &initial_weight,
            None,
        )
        .expect("vcfr solver creation failed");

    gpu_solver.run(1).expect("GPU run failed");

    let gpu_regrets = gpu_solver.download_regrets().expect("download regrets failed");
    let gpu_cum = gpu_solver.download_cum_strategy().expect("download cum failed");

    let num_infosets = tree.num_infosets as usize;
    let data_len = num_infosets * MAX_NA * nh;

    let mut max_regret_diff = 0.0f32;
    let mut diff_idx = 0;
    for i in 0..data_len {
        let diff = (cpu_regrets[i] - gpu_regrets[i]).abs();
        if diff > max_regret_diff {
            max_regret_diff = diff;
            diff_idx = i;
        }
    }
    println!("Max regret diff: {} at idx {}", max_regret_diff, diff_idx);
    if max_regret_diff > 1e-3 {
        let infoset = diff_idx / (MAX_NA * nh);
        let action = (diff_idx % (MAX_NA * nh)) / nh;
        let hand = diff_idx % nh;
        println!("  infoset={}, action={}, hand={}", infoset, action, hand);
        println!("  cpu={}, gpu={}", cpu_regrets[diff_idx], gpu_regrets[diff_idx]);

        let first_20_cpu: Vec<f32> = cpu_regrets[..20.min(cpu_regrets.len())].to_vec();
        let first_20_gpu: Vec<f32> = gpu_regrets[..20.min(gpu_regrets.len())].to_vec();
        println!("CPU first 20: {:?}", first_20_cpu);
        println!("GPU first 20: {:?}", first_20_gpu);
    }
    assert!(max_regret_diff < 1e-3, "regret diff too large: {}", max_regret_diff);

    let mut max_cum_diff = 0.0f32;
    for i in 0..data_len {
        let diff = (cpu_cum[i] - gpu_cum[i]).abs();
        if diff > max_cum_diff {
            max_cum_diff = diff;
        }
    }
    println!("Max cum_strategy diff: {}", max_cum_diff);
    assert!(max_cum_diff < 1e-3, "cum_strategy diff too large: {}", max_cum_diff);
}

#[test]
fn gpu_vcfr_river_10_iter_bit_exact() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let mut cpu = VectorCfr::new(&tree, vec![nh, nh]);
    cpu.run_sequential(&tree, &game, 10);
    let cpu_regrets = cpu.regrets_slice().to_vec();
    let cpu_cum = cpu.cum_strategy_slice().to_vec();

    let gpu = GpuContext::new().expect("GPU init failed");
    let mut gpu_solver = gpu
        .create_vcfr_solver(
            &tree,
            nh,
            &opp_str,
            &opp_idx,
            &pl_str,
            &pl_idx,
            &hand_cards,
            &initial_weight,
            None,
        )
        .expect("vcfr solver creation failed");

    gpu_solver.run(10).expect("GPU run failed");

    let gpu_regrets = gpu_solver.download_regrets().expect("download regrets failed");
    let gpu_cum = gpu_solver.download_cum_strategy().expect("download cum failed");

    let num_infosets = tree.num_infosets as usize;
    let data_len = num_infosets * MAX_NA * nh;

    let mut max_regret_diff = 0.0f32;
    let mut diff_idx = 0;
    for i in 0..data_len {
        let diff = (cpu_regrets[i] - gpu_regrets[i]).abs();
        if diff > max_regret_diff {
            max_regret_diff = diff;
            diff_idx = i;
        }
    }
    println!("10-iter max regret diff: {} at idx {}", max_regret_diff, diff_idx);
    assert!(max_regret_diff < 1.0, "regret diff too large: {}", max_regret_diff);

    let mut max_cum_diff = 0.0f32;
    for i in 0..data_len {
        let diff = (cpu_cum[i] - gpu_cum[i]).abs();
        if diff > max_cum_diff {
            max_cum_diff = diff;
        }
    }
    println!("10-iter max cum_strategy diff: {}", max_cum_diff);
    assert!(max_cum_diff < 1.0, "cum_strategy diff too large: {}", max_cum_diff);
}

#[test]
fn gpu_vcfr_river_convergence() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let gpu = GpuContext::new().expect("GPU init failed");
    let mut gpu_solver = gpu
        .create_vcfr_solver(
            &tree,
            nh,
            &opp_str,
            &opp_idx,
            &pl_str,
            &pl_idx,
            &hand_cards,
            &initial_weight,
            None,
        )
        .expect("vcfr solver creation failed");

    gpu_solver.run(2000).expect("GPU run failed");

    let cum = gpu_solver.download_cum_strategy().expect("download cum failed");
    let data_per_infoset = MAX_NA * nh;
    let node_offsets: Vec<usize> = (0..tree.num_nodes())
        .map(|i| {
            let is = tree.infoset_offsets[i];
            if is == u32::MAX { usize::MAX } else { is as usize * data_per_infoset }
        })
        .collect();
    let profile = StrategyProfile::from_usize_offsets(&cum, &node_offsets, nh);
    let exp = exploitability(&tree, &game, &profile);
    println!("GPU vcfr river exploitability (2k iters): {:.6}", exp);

    let mut cpu = VectorCfr::new(&tree, vec![nh, nh]);
    cpu.run_sequential(&tree, &game, 2000);
    let cpu_profile = StrategyProfile::from_usize_offsets(
        cpu.cum_strategy_slice(),
        cpu.node_offsets(),
        nh,
    );
    let cpu_exp = exploitability(&tree, &game, &cpu_profile);
    println!("CPU vcfr river exploitability (2k iters): {:.6}", cpu_exp);

    assert!(exp < 0.5, "GPU exploitability too high: {}", exp);
    assert!(
        (exp - cpu_exp).abs() < 0.05,
        "GPU vs CPU exploitability mismatch: gpu={:.6} cpu={:.6}",
        exp, cpu_exp
    );
}
