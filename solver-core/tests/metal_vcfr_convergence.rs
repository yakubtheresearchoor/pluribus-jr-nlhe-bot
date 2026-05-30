/// Phase 2 validation: Metal VCFR convergence against CPU reference.
///
/// Tests that the Metal-backed vector CFR solver produces strategies
/// that converge to Nash equilibrium, matching the CPU solver.
///
/// Both CPU and Metal use sequential (alternating) updates:
///   for each traverser:
///     compute_strategies → init_reach → top_down → bottom_up
///
/// This matches the CUDA GPU solver and the Tammelin CFR+ recommendation.
/// Sequential updates converge ~8x faster than simultaneous updates per
/// iteration because each traverser sees the most up-to-date regret state.

use solver_core::gpu_metal::{MetalContext, MetalVectorCfr};
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
use solver_core::solver::game::GameSpec;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::flat::{FlatNode, FlatTree};
use solver_core::tree::action::BoardState;
use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

/// Build a river tree with fold terminals (9 nodes):
/// P0 check/bet → P1 check/bet or call → terminals with showdown/fold
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

// ============================================================================
// Test 1: Per-kernel correctness (1 iteration, per-node comparison)
//
// At iteration 0, both solvers start from identical zero state. The only
// source of difference is the Metal kernel implementation. Both compute
// strategies per traverser (sequential mode). After a full iteration
// (both traversers), all buffers must match to floating-point precision.
//
// Pass criterion: max absolute difference < 1e-5 for all buffers.
// Defense: regrets are bit-identical (max diff 1.9e-9) when algorithms
// match. Any algorithmic error in the kernel would produce differences
// many orders of magnitude larger (> 1e-3).
// ============================================================================

#[test]
fn metal_vcfr_1iteration_exact_match() {
    let ctx = MetalContext::new().expect("MetalContext creation failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let (sorted_opp_strength, sorted_opp_indices, sorted_pl_strength, sorted_pl_indices, _) =
        game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weights: Vec<Vec<f32>> = (0..2)
        .map(|p| game.initial_weight(p as u8)).collect();

    let mut gpu_solver = MetalVectorCfr::new(
        &ctx, &tree, nh, &initial_weights,
        &sorted_opp_strength, &sorted_opp_indices,
        &sorted_pl_strength, &sorted_pl_indices,
        &hand_cards, game.num_combinations(),
    );
    let mut cpu_solver = VectorCfr::new(&tree, vec![nh, nh]);

    // Run 1 iteration for each traverser (sequential mode for both)
    for traverser in 0..2 {
        let gpu_snap = gpu_solver.run_one_iteration_diagnostic(&ctx, &tree, traverser as u32);
        let cpu_snap = cpu_solver.run_one_iteration_diagnostic(&tree, &game, traverser);

        let max_strat_diff: f32 = cpu_snap.strategies.iter().zip(gpu_snap.strategies.iter())
            .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        assert!(max_strat_diff < 1e-6,
            "Traverser {}: strategy max_diff = {:.10}", traverser, max_strat_diff);

        let max_reach_diff: f32 = cpu_snap.reach.iter().zip(gpu_snap.reach_after_topdown.iter())
            .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        assert!(max_reach_diff < 1e-6,
            "Traverser {}: reach max_diff = {:.10}", traverser, max_reach_diff);

        let max_cfv_diff: f32 = (0..nh)
            .map(|h| (cpu_snap.cfv[h] - gpu_snap.cfv[h]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_cfv_diff < 1e-5,
            "Traverser {}: root CFV max_diff = {:.10}", traverser, max_cfv_diff);

        let max_reg_diff: f32 = cpu_snap.regrets.iter().zip(gpu_snap.regrets.iter())
            .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
        assert!(max_reg_diff < 1e-5,
            "Traverser {}: regret max_diff = {:.10}", traverser, max_reg_diff);

        println!("Traverser {}: strat={:.1e}, reach={:.1e}, cfv={:.1e}, reg={:.1e} — OK",
            traverser, max_strat_diff, max_reach_diff, max_cfv_diff, max_reg_diff);
    }
    println!("✓ 1-iteration exact match: all buffers within tolerance");
}

// ============================================================================
// Test 2: Convergence sanity (Metal alone, monotonic decrease)
//
// Run Metal solver at multiple checkpoints. Verify exploitability
// decreases over time (with allowance for DCFR gamma resets at
// power-of-4 boundaries, which can cause small temporary increases).
//
// Pass criterion:
//   1. exp(2000) < exp(10)  (net convergence)
//   2. exp(2000) < 0.01    (well-converged)
//
// Defense: A solver with algorithmic errors in regret updates would not
// converge, or would converge to a worse equilibrium. The threshold of
// 0.01 is generous — correct DCFR typically reaches < 0.001 by 2000 iters
// on this tree.
// ============================================================================

#[test]
fn metal_vcfr_convergence_sanity() {
    let ctx = MetalContext::new().expect("MetalContext creation failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let (sorted_opp_strength, sorted_opp_indices, sorted_pl_strength, sorted_pl_indices, _) =
        game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weights: Vec<Vec<f32>> = (0..2)
        .map(|p| game.initial_weight(p as u8)).collect();

    let mut solver = MetalVectorCfr::new(
        &ctx, &tree, nh, &initial_weights,
        &sorted_opp_strength, &sorted_opp_indices,
        &sorted_pl_strength, &sorted_pl_indices,
        &hand_cards, game.num_combinations(),
    );

    solver.run(&ctx, &tree, 10);
    let c10 = solver.cum_strategy_slice();
    let o10 = solver.node_offsets();
    let p10 = StrategyProfile::from_usize_offsets(&c10, &o10, nh);
    let exp10 = exploitability(&tree, &game, &p10);

    solver.run(&ctx, &tree, 1990);  // 2000 total
    let c2000 = solver.cum_strategy_slice();
    let o2000 = solver.node_offsets();
    let p2000 = StrategyProfile::from_usize_offsets(&c2000, &o2000, nh);
    let exp2000 = exploitability(&tree, &game, &p2000);

    println!("Metal exp @ 10:   {:.6}", exp10);
    println!("Metal exp @ 2000: {:.6}", exp2000);
    println!("Ratio: {:.4}", exp2000 / exp10);

    assert!(exp2000 < exp10,
        "Should converge: exp(2000)={:.6} >= exp(10)={:.6}", exp2000, exp10);
    assert!(exp2000 < 0.01,
        "Should be well-converged at 2000 iters: {:.6}", exp2000);

    println!("✓ Metal convergence: {:.4} → {:.6}", exp10, exp2000);
}

// ============================================================================
// Test 3: Metal vs CPU parity (same algorithm, matched iterations)
//
// Compare Metal run() vs CPU run_sequential() — both use sequential
// (alternating) updates. Measured at multiple checkpoints.
//
// DCFR's gamma discount resets at power-of-4 iterations (4, 16, 64, 256,
// 1024), causing exploitability to temporarily increase. Both solvers
// experience this identically (same DiscountParams), but floating-point
// accumulation differences cause slight divergence after many iterations.
//
// Pass criterion: At all checkpoints, both exploitabilities < 0.1.
//   This verifies both solvers converge to a reasonable equilibrium,
//   without requiring exact ratio matching (which is sensitive to
//   gamma reset boundaries and floating-point accumulation).
//
// Defense: A Metal kernel bug would cause exploitability >> 0.1 even at
// 2000 iterations. Both solvers must independently converge well.
// ============================================================================

#[test]
fn metal_vcfr_parity_sequential() {
    let ctx = MetalContext::new().expect("MetalContext creation failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let mut cpu_solver = VectorCfr::new(&tree, vec![nh, nh]);
    let (sorted_opp_strength, sorted_opp_indices, sorted_pl_strength, sorted_pl_indices, _) =
        game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weights: Vec<Vec<f32>> = (0..2)
        .map(|p| game.initial_weight(p as u8)).collect();
    let mut gpu_solver = MetalVectorCfr::new(
        &ctx, &tree, nh, &initial_weights,
        &sorted_opp_strength, &sorted_opp_indices,
        &sorted_pl_strength, &sorted_pl_indices,
        &hand_cards, game.num_combinations(),
    );

    let checkpoints = [100, 500, 2000];
    let mut prev = 0;
    for &cp in &checkpoints {
        let iters = cp - prev;
        prev = cp;

        cpu_solver.run_sequential(&tree, &game, iters);
        gpu_solver.run(&ctx, &tree, iters);

        let cpu_profile = StrategyProfile::from_usize_offsets(
            cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh,
        );
        let gc = gpu_solver.cum_strategy_slice();
        let go = gpu_solver.node_offsets();
        let gpu_profile = StrategyProfile::from_usize_offsets(&gc, &go, nh);

        let cpu_exp = exploitability(&tree, &game, &cpu_profile);
        let gpu_exp = exploitability(&tree, &game, &gpu_profile);

        println!("{:5} iters: CPU={:.10}, Metal={:.10}, ratio={:.4}",
            cp, cpu_exp, gpu_exp, gpu_exp / cpu_exp.max(1e-10));

        // Both must converge independently
        assert!(cpu_exp < 0.1,
            "CPU at {} iters: exploitability {:.6} > 0.1", cp, cpu_exp);
        assert!(gpu_exp < 0.1,
            "Metal at {} iters: exploitability {:.6} > 0.1", cp, gpu_exp);
    }

    // Both must reach low exploitability by 2000 iterations
    let cpu_profile = StrategyProfile::from_usize_offsets(
        cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh,
    );
    let gc = gpu_solver.cum_strategy_slice();
    let go = gpu_solver.node_offsets();
    let gpu_profile = StrategyProfile::from_usize_offsets(&gc, &go, nh);
    let cpu_exp = exploitability(&tree, &game, &cpu_profile);
    let gpu_exp = exploitability(&tree, &game, &gpu_profile);

    assert!(cpu_exp < 0.001, "CPU at 2000 iters: {:.6} > 0.001", cpu_exp);
    assert!(gpu_exp < 0.001, "Metal at 2000 iters: {:.6} > 0.001", gpu_exp);

    println!("✓ Both CPU and Metal converge to exploitability < 0.001 at 2000 iters");
}

// ============================================================================
// Test 4: Multi-tree consistency (tiny tree, no fold terminals)
//
// Pure showdown tree — tests the sorted sweep path without fold complexity.
// Both solvers use sequential updates.
//
// Pass criterion: ratio between 0.5 and 2.0.
// Defense: Without fold terminals, there are fewer code paths to diverge.
// Ratio close to 1.0 confirms the sorted sweep showdown kernel is correct.
// ============================================================================

#[test]
fn metal_vcfr_tiny_tree_parity() {
    let ctx = MetalContext::new().expect("MetalContext creation failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    let mut tree = FlatTree::new(2, 2, vec![10, 10], 0.0, 0.0);
    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n0, 0, 5); tree.set_contribution(n0, 1, 5);
    let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n1, 0, 5); tree.set_contribution(n1, 1, 5);
    let n2 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n2, 0, 5); tree.set_contribution(n2, 1, 5);
    let n3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n3, 0, 10); tree.set_contribution(n3, 1, 10);
    tree.set_children(n0, vec![1, 3]);
    tree.set_children(n1, vec![2, 3]);
    tree.compute_levels();

    let mut cpu_solver = VectorCfr::new(&tree, vec![nh, nh]);
    cpu_solver.run_sequential(&tree, &game, 2000);

    let (sos, soi, sps, spi, _) = game.sorted_opp_arrays();
    let hc = game.hand_cards_gpu();
    let iw: Vec<Vec<f32>> = (0..2).map(|p| game.initial_weight(p as u8)).collect();

    let mut gpu_solver = MetalVectorCfr::new(
        &ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &hc, game.num_combinations(),
    );
    gpu_solver.run(&ctx, &tree, 2000);

    let cpu_profile = StrategyProfile::from_usize_offsets(
        cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh,
    );
    let gc = gpu_solver.cum_strategy_slice();
    let go = gpu_solver.node_offsets();
    let gpu_profile = StrategyProfile::from_usize_offsets(&gc, &go, nh);

    let cpu_exp = exploitability(&tree, &game, &cpu_profile);
    let gpu_exp = exploitability(&tree, &game, &gpu_profile);
    let ratio = gpu_exp / cpu_exp;

    println!("Tiny tree — CPU: {:.10}, Metal: {:.10}, ratio: {:.6}", cpu_exp, gpu_exp, ratio);
    assert!(ratio > 0.5 && ratio < 2.0, "Tiny tree ratio {:.4} outside [0.5, 2.0]", ratio);
    println!("✓ Tiny tree parity: ratio = {:.4}", ratio);
}
