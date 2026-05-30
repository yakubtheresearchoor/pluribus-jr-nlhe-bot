/// Diagnostic test with the original 9-node river tree that showed 9.4x divergence.
/// This tree has fold terminals (nodes 5, 7) which are the likely divergence source.

use solver_core::gpu_metal::{MetalContext, MetalVectorCfr};
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
use solver_core::tree::flat::{FlatNode, FlatTree};
use solver_core::tree::action::BoardState;
use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

/// Original 9-node tree from metal_vcfr_convergence.rs
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

    tree.set_folded_mask(n5, 0b10); // P1 folded
    tree.set_folded_mask(n7, 0b01); // P0 folded

    tree.compute_levels();
    tree
}

#[test]
fn diagnostic_9node_1iter() {
    let ctx = MetalContext::new().expect("MetalContext creation failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();
    let np = 2usize;

    let (sorted_opp_strength, sorted_opp_indices, sorted_pl_strength, sorted_pl_indices, _) =
        game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weights: Vec<Vec<f32>> = (0..2)
        .map(|p| game.initial_weight(p as u8))
        .collect();

    let mut gpu_solver = MetalVectorCfr::new(
        &ctx, &tree, nh,
        &initial_weights,
        &sorted_opp_strength, &sorted_opp_indices,
        &sorted_pl_strength, &sorted_pl_indices,
        &hand_cards,
        game.num_combinations(),
    );

    let mut cpu_solver = VectorCfr::new(&tree, vec![nh, nh]);

    // Run 1 iteration for traverser 0
    let gpu_snap = gpu_solver.run_one_iteration_diagnostic(&ctx, &tree, 0);
    let cpu_snap = cpu_solver.run_one_iteration_diagnostic(&tree, &game, 0);

    println!("=== 9-NODE DIAGNOSTIC: Iteration 0, Traverser 0 ===");
    println!("nh = {}, nn = {}, np = {}", nh, tree.num_nodes(), np);

    // Strategies
    let stride = solver_core::tree::flat::MAX_NA * nh;
    let max_strat_diff: f32 = (0..cpu_snap.strategies.len())
        .map(|i| (cpu_snap.strategies[i] - gpu_snap.strategies[i]).abs())
        .fold(0.0f32, f32::max);
    println!("Strategy max diff: {:.10}", max_strat_diff);

    // Reach
    let max_reach_diff: f32 = (0..cpu_snap.reach.len())
        .map(|i| (cpu_snap.reach[i] - gpu_snap.reach_after_topdown[i]).abs())
        .fold(0.0f32, f32::max);
    println!("Reach max diff: {:.10}", max_reach_diff);

    // Root CFV
    let root_cfv_diff: f32 = (0..nh)
        .map(|h| (cpu_snap.cfv[h] - gpu_snap.cfv[h]).abs())
        .fold(0.0f32, f32::max);
    println!("Root CFV max diff: {:.10}", root_cfv_diff);

    // Per-terminal CFVs
    println!("\n--- Per-terminal-node CFVs ---");
    for nid in 0..tree.num_nodes() {
        let node = &tree.nodes[nid];
        if !node.is_terminal() { continue; }
        let c0 = tree.get_contribution(nid, 0);
        let c1 = tree.get_contribution(nid, 1);
        let fm = tree.get_folded_mask(nid);

        let reach_base = nid * np * nh;
        let cfreach: Vec<Vec<f32>> = (0..np)
            .map(|p| (0..nh).map(|h| cpu_snap.reach[reach_base + p * nh + h]).collect())
            .collect();
        let cpu_cfv = game.evaluate_terminal(0, nid, &tree, &cfreach);

        let mut max_diff = 0.0f32;
        let mut first_diff = None;
        for h in 0..nh {
            let diff = (cpu_cfv[h] - gpu_snap.cfv[nid * nh + h]).abs();
            if diff > max_diff {
                max_diff = diff;
                if first_diff.is_none() && diff > 1e-6 {
                    first_diff = Some((h, cpu_cfv[h], gpu_snap.cfv[nid * nh + h]));
                }
            }
        }

        let status = if max_diff < 1e-6 { "OK" } else { "*** DIVERGENCE ***" };
        println!("  Node {} (c={}/{}, fm=0b{:02}): max_diff={:.10}  {}",
            nid, c0, c1, fm, max_diff, status);
        if let Some((h, cv, gv)) = first_diff {
            println!("    First diff at h={}: cpu={:.10}, gpu={:.10}", h, cv, gv);
            // Show reach at this terminal node
            for p in 0..np {
                println!("    Reach P{} [h={}] = {:.10}", p, h,
                    cpu_snap.reach[reach_base + p * nh + h]);
            }
        }
    }

    // Regrets
    let max_reg_diff: f32 = (0..cpu_snap.regrets.len())
        .map(|i| (cpu_snap.regrets[i] - gpu_snap.regrets[i]).abs())
        .fold(0.0f32, f32::max);
    println!("\nRegret max diff: {:.10}", max_reg_diff);
}

#[test]
fn diagnostic_9node_100iters_run() {
    // Use run() for both CPU and Metal (same algorithm)
    let ctx = MetalContext::new().expect("MetalContext creation failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    // CPU with run() (same algorithm as Metal)
    let mut cpu_solver = VectorCfr::new(&tree, vec![nh, nh]);
    cpu_solver.run(&tree, &game, 2000);

    // Metal
    let (sorted_opp_strength, sorted_opp_indices, sorted_pl_strength, sorted_pl_indices, _) =
        game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weights: Vec<Vec<f32>> = (0..2)
        .map(|p| game.initial_weight(p as u8))
        .collect();

    let mut gpu_solver = MetalVectorCfr::new(
        &ctx, &tree, nh,
        &initial_weights,
        &sorted_opp_strength, &sorted_opp_indices,
        &sorted_pl_strength, &sorted_pl_indices,
        &hand_cards,
        game.num_combinations(),
    );
    gpu_solver.run(&ctx, &tree, 2000);

    let cpu_profile = StrategyProfile::from_usize_offsets(
        cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh,
    );
    let gpu_cum = gpu_solver.cum_strategy_slice();
    let gpu_offsets = gpu_solver.node_offsets();
    let gpu_profile = StrategyProfile::from_usize_offsets(&gpu_cum, &gpu_offsets, nh);

    let cpu_exp = exploitability(&tree, &game, &cpu_profile);
    let gpu_exp = exploitability(&tree, &game, &gpu_profile);

    println!("=== 9-NODE, 2000 iters, run() for both ===");
    println!("CPU  exploitability: {:.10}", cpu_exp);
    println!("Metal exploitability: {:.10}", gpu_exp);
    if cpu_exp > 0.0 {
        println!("ACTUAL RATIO (no clamping): {:.6}", gpu_exp / cpu_exp);
    }
}

#[test]
fn diagnostic_9node_100iters_sequential() {
    // Compare: what if CPU uses run_sequential (as original test did)?
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    // CPU run()
    let mut cpu_run = VectorCfr::new(&tree, vec![nh, nh]);
    cpu_run.run(&tree, &game, 2000);

    // CPU run_sequential()
    let mut cpu_seq = VectorCfr::new(&tree, vec![nh, nh]);
    cpu_seq.run_sequential(&tree, &game, 2000);

    let profile_run = StrategyProfile::from_usize_offsets(
        cpu_run.cum_strategy_slice(), cpu_run.node_offsets(), nh,
    );
    let profile_seq = StrategyProfile::from_usize_offsets(
        cpu_seq.cum_strategy_slice(), cpu_seq.node_offsets(), nh,
    );
    let exp_run = exploitability(&tree, &game, &profile_run);
    let exp_seq = exploitability(&tree, &game, &profile_seq);

    println!("=== CPU run() vs run_sequential() comparison, 2000 iters ===");
    println!("CPU run()           exploitability: {:.10}", exp_run);
    println!("CPU run_sequential() exploitability: {:.10}", exp_seq);
    println!("run()/run_sequential() ratio: {:.6}", exp_run / exp_seq.max(1e-10));
}
