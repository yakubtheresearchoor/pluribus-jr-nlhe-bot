#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::{ChanceGpuData, GpuContext};
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::game::GameSpec;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatNode, FlatTree, MAX_NA};

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

fn build_3player_river_tree() -> FlatTree {
    let mut tree = FlatTree::new(3, 200, vec![200, 200, 200], 0.0, 0.0);
    let n_p0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n_p0, 0, 5);
    tree.set_contribution(n_p0, 1, 5);
    tree.set_contribution(n_p0, 2, 5);
    let n_check = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n_check, 0, 5);
    tree.set_contribution(n_check, 1, 5);
    tree.set_contribution(n_check, 2, 5);
    let n_p1_resp = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n_p1_resp, 0, 15);
    tree.set_contribution(n_p1_resp, 1, 5);
    tree.set_contribution(n_p1_resp, 2, 5);
    let n_p2_resp = tree.alloc_node(FlatNode::player(2, BoardState::River, 0));
    tree.set_contribution(n_p2_resp, 0, 15);
    tree.set_contribution(n_p2_resp, 1, 15);
    tree.set_contribution(n_p2_resp, 2, 5);
    let n_showdown3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n_showdown3, 0, 15);
    tree.set_contribution(n_showdown3, 1, 15);
    tree.set_contribution(n_showdown3, 2, 15);
    let n_p2_fold = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n_p2_fold, 0, 15);
    tree.set_contribution(n_p2_fold, 1, 15);
    tree.set_contribution(n_p2_fold, 2, 5);
    tree.set_folded_mask(n_p2_fold, 4);
    let n_p1_fold = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n_p1_fold, 0, 15);
    tree.set_contribution(n_p1_fold, 1, 5);
    tree.set_contribution(n_p1_fold, 2, 5);
    tree.set_folded_mask(n_p1_fold, 2);
    tree.set_children(n_p0, vec![n_check as u32, n_p1_resp as u32]);
    tree.set_children(n_p1_resp, vec![n_p2_resp as u32, n_p1_fold as u32]);
    tree.set_children(n_p2_resp, vec![n_showdown3 as u32, n_p2_fold as u32]);
    tree.compute_levels();
    tree
}

#[test]
fn gpu_vcfr_3player_parity() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 3);
    let nh = game.num_valid_hands();

    let tree = build_3player_river_tree();

    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let mut initial_weight = game.initial_weight_flat(&ranges);

    let gpu = GpuContext::new().expect("GPU init failed");
    let mut gpu_solver = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
    ).expect("gpu vcfr create failed");

    let iters = 200;
    gpu_solver.run(iters).expect("gpu vcfr run failed");
    let gpu_cum = gpu_solver.download_cum_strategy().expect("download failed");

    let gpu_offsets: Vec<usize> = (0..tree.num_nodes()).map(|i| {
        let is = tree.infoset_offsets[i];
        if is == u32::MAX { usize::MAX } else { is as usize * MAX_NA * nh }
    }).collect();

    let gpu_profile = StrategyProfile::from_usize_offsets(&gpu_cum, &gpu_offsets, nh);
    let gpu_exp = exploitability(&tree, &game, &gpu_profile);
    println!("GPU VCFR 3p exploitability ({} iters): {:.6}", iters, gpu_exp);

    let mut cpu_solver = VectorCfr::new(&tree, vec![nh, nh, nh]);
    cpu_solver.run(&tree, &game, iters);
    let cpu_profile = StrategyProfile::from_usize_offsets(
        cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh,
    );
    let cpu_exp = exploitability(&tree, &game, &cpu_profile);
    println!("CPU VCFR 3p exploitability ({} iters): {:.6}", iters, cpu_exp);

    let as_kh_raw = card_pair_to_index(card_from_str("As").unwrap(), card_from_str("Kh").unwrap());
    let valid = game.valid_hand_indices();
    let hi = valid.iter().position(|&vi| vi as usize == as_kh_raw).unwrap();

    fn extract_bet(cum: &[f32], offsets: &[usize], nh: usize, node: usize) -> Vec<f32> {
        let off = offsets[node];
        if off == usize::MAX { return vec![0.5; nh]; }
        let mut bet = vec![0.0f32; nh];
        for h in 0..nh {
            let mut t = 0.0f32;
            for a in 0..2 { t += cum[off + a * nh + h]; }
            bet[h] = if t > 0.0 { cum[off + 1 * nh + h] / t } else { 0.5 };
        }
        bet
    }

    let gpu_bet = extract_bet(&gpu_cum, &gpu_offsets, nh, 0);
    let cpu_bet = extract_bet(cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh, 0);
    println!("GPU P0 bet prob for AsKh: {:.4}", gpu_bet[hi]);
    println!("CPU P0 bet prob for AsKh: {:.4}", cpu_bet[hi]);

    let strat_diff: f32 = (0..nh)
        .map(|h| (gpu_bet[h] - cpu_bet[h]).abs())
        .sum::<f32>() / nh as f32;
    println!("Avg P0 bet prob diff: {:.6}", strat_diff);

    assert!(gpu_exp < cpu_exp * 5.0 + 100.0,
        "GPU exp ({:.6}) should be reasonable vs CPU ({:.6})", gpu_exp, cpu_exp);
    assert!(strat_diff < 0.1,
        "GPU/CPU strategy too different: avg diff = {:.6}", strat_diff);
}

#[test]
fn gpu_vcfr_3player_smoke() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 3);
    let nh = game.num_valid_hands();

    let tree = build_3player_river_tree();

    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let gpu = GpuContext::new().expect("GPU init failed");
    let mut gpu_solver = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
    ).expect("gpu vcfr create failed");

    gpu_solver.run(50).expect("gpu vcfr run failed");
    let gpu_cum = gpu_solver.download_cum_strategy().expect("download failed");

    let gpu_offsets: Vec<usize> = (0..tree.num_nodes()).map(|i| {
        let is = tree.infoset_offsets[i];
        if is == u32::MAX { usize::MAX } else { is as usize * MAX_NA * nh }
    }).collect();

    let off = gpu_offsets[0];
    let mut avg_bet = 0.0f32;
    for h in 0..nh {
        let mut t = 0.0f32;
        for a in 0..2 { t += gpu_cum[off + a * nh + h]; }
        avg_bet += if t > 0.0 { gpu_cum[off + 1 * nh + h] / t } else { 0.5 };
    }
    avg_bet /= nh as f32;
    println!("GPU 3p P0 avg bet prob (50 iters): {:.4}", avg_bet);
    assert!(avg_bet > 0.01 && avg_bet < 0.99, "bet prob degenerate: {}", avg_bet);
}
