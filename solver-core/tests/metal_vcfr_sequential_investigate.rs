/// Focused investigation: Why does Metal sequential converge faster than CPU sequential?
///
/// Hypothesis: The Metal kernel processes levels bottom-up, and the bottom-up
/// results for earlier traversers are visible when computing strategies for
/// the next traverser in the SAME iteration. But something subtle differs.

use solver_core::gpu_metal::{MetalContext, MetalVectorCfr};
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::flat::{FlatNode, FlatTree};
use solver_core::tree::action::BoardState;
use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

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
fn investigate_sequential_iteration1_regrets() {
    // Run 1 iteration on both solvers, dump regrets after each traverser.
    // If the algorithms are identical, regrets must match exactly.

    let ctx = MetalContext::new().expect("MetalContext creation failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let (sos, soi, sps, spi, _) = game.sorted_opp_arrays();
    let hc = game.hand_cards_gpu();
    let iw: Vec<Vec<f32>> = (0..2).map(|p| game.initial_weight(p as u8)).collect();

    let mut gpu_solver = MetalVectorCfr::new(
        &ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &hc, game.num_combinations(),
    );
    let mut cpu_solver = VectorCfr::new(&tree, vec![nh, nh]);

    // Iteration 0: params alpha=0, beta=0.5, gamma=0
    // For np=2, run_sequential does:
    //   traverser 0: compute_strategies, compute_reach, bottom_up(traverser=0)
    //   traverser 1: compute_strategies, compute_reach, bottom_up(traverser=1)

    // Run traverser 0 on both
    let gpu_snap0 = gpu_solver.run_one_iteration_diagnostic(&ctx, &tree, 0);
    let cpu_snap0 = cpu_solver.run_one_iteration_diagnostic(&tree, &game, 0);

    let stride = solver_core::tree::flat::MAX_NA * nh;

    // Compare regrets after traverser 0
    let max_reg_diff_t0: f32 = cpu_snap0.regrets.iter().zip(gpu_snap0.regrets.iter())
        .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    println!("After traverser 0: regret max diff = {:.10}", max_reg_diff_t0);

    // Compare strategies after traverser 0
    let max_strat_diff_t0: f32 = cpu_snap0.strategies.iter().zip(gpu_snap0.strategies.iter())
        .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    println!("After traverser 0: strategy max diff = {:.10}", max_strat_diff_t0);

    // Now run traverser 1 on both (note: both start from same regret state after T0)
    let gpu_snap1 = gpu_solver.run_one_iteration_diagnostic(&ctx, &tree, 1);
    let cpu_snap1 = cpu_solver.run_one_iteration_diagnostic(&tree, &game, 1);

    let max_reg_diff_t1: f32 = cpu_snap1.regrets.iter().zip(gpu_snap1.regrets.iter())
        .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    println!("After traverser 1: regret max diff = {:.10}", max_reg_diff_t1);

    let max_strat_diff_t1: f32 = cpu_snap1.strategies.iter().zip(gpu_snap1.strategies.iter())
        .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    println!("After traverser 1: strategy max diff = {:.10}", max_strat_diff_t1);

    // Check if strategies before T1 match (they should, since regrets match after T0)
    // The strategies used for T1 should be identical if both solvers see the same regrets
    println!("\nStrategies before T1 (first 5 values per infoset):");
    for infoset in 0..tree.num_infosets as usize {
        let cpu_s = &cpu_snap1.strategies[infoset * stride..infoset * stride + 4.min(nh)];
        let gpu_s = &gpu_snap1.strategies[infoset * stride..infoset * stride + 4.min(nh)];
        let diff: f32 = cpu_s.iter().zip(gpu_s.iter()).map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        println!("  Infoset {} check-action-0: cpu={:?}, gpu={:?}, diff={:.10}",
            infoset, &cpu_s[..2.min(nh)], &gpu_s[..2.min(nh)], diff);
    }

    // Now let's look at the cum_strategy after both traversers
    let cum_diff: f32 = cpu_snap1.cum_strategy.iter().zip(gpu_snap1.cum_strategy.iter())
        .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    println!("\nCum strategy after T1: max diff = {:.10}", cum_diff);

    // Print first iteration gamma_t
    // gamma_0 = 0.0, so cum_strategy should just be sigma (current strategy)
    println!("gamma_t = {:.6} (should be 0.0 for iter 0)", gpu_snap0.gamma_t);

    // CRITICAL: print non-zero regret entries
    println!("\nNon-zero regrets after iteration 0 (both traversers):");
    let num_infosets = tree.num_infosets as usize;
    for infoset in 0..num_infosets {
        let na = tree.nodes[tree.decision_node_ids[infoset] as usize].num_children as usize;
        for a in 0..na {
            let mut first_diffs = Vec::new();
            for h in 0..5.min(nh) {
                let idx = infoset * stride + a * nh + h;
                let cv = cpu_snap1.regrets[idx];
                let gv = gpu_snap1.regrets[idx];
                if cv != 0.0 || gv != 0.0 {
                    first_diffs.push((h, cv, gv));
                }
            }
            if !first_diffs.is_empty() {
                println!("  infoset={}, a={}: {:?}", infoset, a,
                    first_diffs.iter().take(3).map(|(h,c,g)| format!("h{}: cpu={:.8}, gpu={:.8}", h, c, g)).collect::<Vec<_>>());
            }
        }
    }
}

#[test]
fn investigate_convergence_curves() {
    // Run both solvers for 500 iterations, measure exploitability at checkpoints.
    // This shows whether the convergence rate differs systematically.

    use solver_core::solver::best_response::{StrategyProfile, exploitability};

    let ctx = MetalContext::new().expect("MetalContext creation failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let (sos, soi, sps, spi, _) = game.sorted_opp_arrays();
    let hc = game.hand_cards_gpu();
    let iw: Vec<Vec<f32>> = (0..2).map(|p| game.initial_weight(p as u8)).collect();

    let mut gpu_solver = MetalVectorCfr::new(
        &ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &hc, game.num_combinations(),
    );

    let mut cpu_solver = VectorCfr::new(&tree, vec![nh, nh]);

    let checkpoints = [10, 50, 100, 250, 500, 1000, 2000];
    let mut prev = 0;
    let mut gpu_exploitabilities = Vec::new();
    let mut cpu_exploitabilities = Vec::new();

    for &cp in &checkpoints {
        let iters = cp - prev;
        prev = cp;

        // CPU
        cpu_solver.run_sequential(&tree, &game, iters);
        let cpu_profile = StrategyProfile::from_usize_offsets(
            cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh,
        );
        let cpu_exp = exploitability(&tree, &game, &cpu_profile);
        cpu_exploitabilities.push(cpu_exp);

        // Metal
        gpu_solver.run(&ctx, &tree, iters);
        let gc = gpu_solver.cum_strategy_slice();
        let go = gpu_solver.node_offsets();
        let gpu_profile = StrategyProfile::from_usize_offsets(&gc, &go, nh);
        let gpu_exp = exploitability(&tree, &game, &gpu_profile);
        gpu_exploitabilities.push(gpu_exp);

        let ratio = gpu_exp / cpu_exp.max(1e-10);
        println!("{:5} iters: CPU={:.10}, Metal={:.10}, ratio={:.4}", cp, cpu_exp, gpu_exp, ratio);
    }

    // Check convergence direction
    for i in 1..gpu_exploitabilities.len() {
        assert!(gpu_exploitabilities[i] <= gpu_exploitabilities[i-1] * 1.01,
            "Metal exploitability increased at iteration {}: {:.6} -> {:.6}",
            checkpoints[i-1], gpu_exploitabilities[i-1], gpu_exploitabilities[i]);
    }
}
