/// Diagnostic test: compare CPU and Metal VCFR intermediate values
/// to find the first kernel where divergence occurs.
///
/// Strategy: Run 1 iteration on both solvers, dump per-node state at each
/// stage, compare with exact float tolerances.

use solver_core::gpu_metal::{MetalContext, MetalVectorCfr};
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::flat::{FlatNode, FlatTree};
use solver_core::tree::action::BoardState;
use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

/// Minimal 4-node river tree: P0 checks/bets → P1 checks/bets
/// Nodes: [0: P0 check/bet] → [1: P1 check/bet], [2: terminal(check-check)], [3: terminal(bet)]
fn build_tiny_tree() -> FlatTree {
    let mut tree = FlatTree::new(2, 2, vec![10, 10], 0.0, 0.0);

    // Node 0: P0 decision (check or bet 5)
    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n0, 0, 5);
    tree.set_contribution(n0, 1, 5);

    // Node 1: P1 decision (check or bet 5) after P0 checks
    let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n1, 0, 5);
    tree.set_contribution(n1, 1, 5);

    // Node 2: terminal (check-check) — showdown, equal contributions
    let n2 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n2, 0, 5);
    tree.set_contribution(n2, 1, 5);

    // Node 3: terminal (bet) — showdown after P0 bets, P1 calls
    let n3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n3, 0, 10);
    tree.set_contribution(n3, 1, 10);

    tree.set_children(n0, vec![1, 3]); // check → node 1, bet → node 3
    tree.set_children(n1, vec![2, 3]); // check → node 2, bet → node 3

    tree.compute_levels();
    tree
}

#[test]
fn diagnostic_strategy_computation() {
    // Stage 1: Verify compute_strategies matches between CPU and Metal
    // After init (all regrets zero), both should produce uniform 0.5/0.5 strategies
    let ctx = MetalContext::new().expect("MetalContext creation failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_tiny_tree();

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

    println!("=== DIAGNOSTIC: Iteration 0, Traverser 0 ===");
    println!("nh = {}, nn = {}, np = {}", nh, tree.num_nodes(), tree.num_players);
    println!("num_infosets = {}", tree.num_infosets);
    println!("alpha_t = {:.6}, beta_t = {:.6}, gamma_t = {:.6}", gpu_snap.alpha_t, gpu_snap.beta_t, gpu_snap.gamma_t);
    println!("CPU alpha_t = {:.6}, beta_t = {:.6}, gamma_t = {:.6}", cpu_snap.alpha_t, cpu_snap.beta_t, cpu_snap.gamma_t);

    // ---- Step 1: Compare strategies ----
    println!("\n--- Strategy comparison ---");
    let (mut s_match, mut s_mismatch) = (0usize, 0usize);
    let mut max_strat_diff = 0.0f32;
    let stride = solver_core::tree::flat::MAX_NA_POSTFLOP * nh;
    for infoset in 0..tree.num_infosets as usize {
        for a in 0..2 {
            for h in 0..nh {
                let cpu_val = cpu_snap.strategies[infoset * stride + a * nh + h];
                let gpu_val = gpu_snap.strategies[infoset * stride + a * nh + h];
                let diff = (cpu_val - gpu_val).abs();
                if diff > max_strat_diff { max_strat_diff = diff; }
                if diff > 1e-6 { s_mismatch += 1; } else { s_match += 1; }
            }
        }
    }
    println!("Strategy: {} match, {} mismatch, max_diff = {:.10}", s_match, s_mismatch, max_strat_diff);
    assert!(max_strat_diff < 1e-6, "Strategy mismatch: max_diff = {}", max_strat_diff);

    // ---- Step 2: Compare reach after top-down ----
    println!("\n--- Reach comparison (after top-down) ---");
    let nn = tree.num_nodes();
    let np = 2usize;
    let (mut r_match, mut r_mismatch) = (0usize, 0usize);
    let mut max_reach_diff = 0.0f32;
    let mut first_reach_mismatch = None;
    for node in 0..nn {
        for p in 0..np {
            for h in 0..nh {
                let idx = node * np * nh + p * nh + h;
                let cpu_val = cpu_snap.reach[idx];
                let gpu_val = gpu_snap.reach_after_topdown[idx];
                let diff = (cpu_val - gpu_val).abs();
                if diff > max_reach_diff { max_reach_diff = diff; }
                if diff > 1e-5 {
                    r_mismatch += 1;
                    if first_reach_mismatch.is_none() {
                        first_reach_mismatch = Some((node, p, h, cpu_val, gpu_val, diff));
                    }
                } else {
                    r_match += 1;
                }
            }
        }
    }
    println!("Reach: {} match, {} mismatch, max_diff = {:.10}", r_match, r_mismatch, max_reach_diff);
    if let Some((node, p, h, cv, gv, d)) = first_reach_mismatch {
        println!("  First mismatch: node={}, p={}, h={}: cpu={:.10}, gpu={:.10}, diff={:.10}", node, p, h, cv, gv, d);
    }
    assert!(max_reach_diff < 1e-4, "Reach mismatch: max_diff = {}", max_reach_diff);

    // ---- Step 3: Compare CFV (terminal node evaluation) ----
    println!("\n--- CFV comparison (after bottom-up, traverser {}) ---", gpu_snap.traverser);
    // CPU's cfv is only the root cfv (nh values). Metal stores per-node cfv.
    // We need to reconstruct per-node CFVs from the CPU.
    // Instead, let's compare the root CFV first.
    println!("Root CFV (CPU): first 5 = {:?}", &cpu_snap.cfv[..5.min(nh)]);
    println!("Root CFV (GPU): first 5 = {:?}", &gpu_snap.cfv[..5.min(nh)]);
    let root_cfv_diff: f32 = (0..nh)
        .map(|h| (cpu_snap.cfv[h] - gpu_snap.cfv[h]).abs())
        .fold(0.0f32, f32::max);
    let root_cfv_rel = if cpu_snap.cfv.iter().map(|v| v.abs()).fold(0.0f32, f32::max) > 0.0 {
        root_cfv_diff / cpu_snap.cfv.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    } else { 0.0 };
    println!("Root CFV max abs diff = {:.10}", root_cfv_diff);
    println!("Root CFV max rel diff = {:.10}", root_cfv_rel);

    // ---- Step 3b: Compare per-terminal-node CFVs ----
    // Run CPU bottom-up and capture per-terminal-node CFVs
    // The CPU doesn't store per-node CFVs easily. Instead, let's compare regrets directly.
    println!("\n--- Regrets comparison (after traverser 0 bottom-up) ---");
    let (mut reg_match, mut reg_mismatch) = (0usize, 0usize);
    let mut max_regret_diff = 0.0f32;
    let mut first_regret_mismatch = None;
    for infoset in 0..tree.num_infosets as usize {
        for a in 0..2 {
            for h in 0..nh.min(10) {
                let idx = infoset * stride + a * nh + h;
                let cpu_val = cpu_snap.regrets[idx];
                let gpu_val = gpu_snap.regrets[idx];
                let diff = (cpu_val - gpu_val).abs();
                if diff > max_regret_diff { max_regret_diff = diff; }
                if diff > 1e-4 {
                    reg_mismatch += 1;
                    if first_regret_mismatch.is_none() {
                        first_regret_mismatch = Some((infoset, a, h, cpu_val, gpu_val, diff));
                    }
                } else {
                    reg_match += 1;
                }
            }
        }
    }
    println!("Regrets (first 10 hands): {} match, {} mismatch, max_diff = {:.10}", reg_match, reg_mismatch, max_regret_diff);
    if let Some((infoset, a, h, cv, gv, d)) = first_regret_mismatch {
        println!("  First mismatch: infoset={}, a={}, h={}: cpu={:.10}, gpu={:.10}, diff={:.10}", infoset, a, h, cv, gv, d);
    }

    // ---- Step 3c: Detailed terminal node CFV comparison ----
    // We need to manually compute what the terminal CFVs should be for traverser 0
    // and compare against what the Metal kernel produced.
    println!("\n--- Per-terminal-node CFV comparison ---");
    let node_ids: Vec<usize> = (0..tree.num_nodes()).collect();
    for &nid in &node_ids {
        let node = &tree.nodes[nid];
        if !node.is_terminal() { continue; }
        let c0 = tree.get_contribution(nid, 0);
        let c1 = tree.get_contribution(nid, 1);
        let fm = tree.get_folded_mask(nid);
        println!("  Terminal node {}: contrib=({},{}), fold_mask=0b{:08b}", nid, c0, c1, fm);
        let gpu_cfv_h0 = gpu_snap.cfv[nid * nh..nid * nh + 3.min(nh)].to_vec();
        // CPU doesn't store per-node CFVs, but we can compute them
        // using the game's evaluate_terminal
        let reach_base = nid * np * nh;
        let cfreach: Vec<Vec<f32>> = (0..np)
            .map(|p| {
                (0..nh).map(|h| cpu_snap.reach[reach_base + p * nh + h]).collect()
            })
            .collect();
        let cpu_cfv = game.evaluate_terminal(0, nid, &tree, &cfreach);
        let cpu_cfv_h0 = cpu_cfv[..3.min(nh)].to_vec();
        let diff: f32 = (0..nh).map(|h| (cpu_cfv[h] - gpu_snap.cfv[nid * nh + h]).abs()).fold(0.0f32, f32::max);
        println!("    CPU CFV (first 3): {:?}", cpu_cfv_h0);
        println!("    GPU CFV (first 3): {:?}", gpu_cfv_h0);
        println!("    Max diff: {:.10}", diff);
        if diff > 0.01 {
            println!("    *** SIGNIFICANT DIVERGENCE ***");
            // Dump reach values for this terminal node
            for p in 0..np {
                println!("    Reach P{} (first 5): {:?}", p,
                    &cpu_snap.reach[reach_base + p * nh..reach_base + p * nh + 5.min(nh)]);
            }
        }
    }

    // Don't assert on CFVs yet — just print the diagnostics.
    // The test framework will show the output.
    println!("\n=== Diagnostic complete ===");
}

#[test]
fn diagnostic_after_100_iterations() {
    // Compare exploitability after 100 iterations with CORRECT ratio calculation.
    // Uses run() (not run_sequential) for CPU to match Metal's algorithm.
    let ctx = MetalContext::new().expect("MetalContext creation failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    let tree = build_tiny_tree();

    // Run CPU with run() (same algorithm as Metal)
    let mut cpu_solver = VectorCfr::new(&tree, vec![nh, nh]);
    cpu_solver.run(&tree, &game, 100);

    // Run Metal
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

    gpu_solver.run(&ctx, &tree, 100);

    // Compare regrets directly
    let cpu_reg = cpu_solver.regrets_slice();
    let gpu_reg = gpu_solver.regrets_slice();
    let max_reg_diff: f32 = cpu_reg.iter().zip(gpu_reg.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let max_reg_val: f32 = cpu_reg.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-10);
    println!("Regret max abs diff: {:.10}", max_reg_diff);
    println!("Regret max value: {:.10}", max_reg_val);
    println!("Regret relative diff: {:.6}", max_reg_diff / max_reg_val);

    // Compare cum_strategy directly
    let cpu_cum = cpu_solver.cum_strategy_slice();
    let gpu_cum = gpu_solver.cum_strategy_slice();
    let max_cum_diff: f32 = cpu_cum.iter().zip(gpu_cum.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let max_cum_val: f32 = cpu_cum.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-10);
    println!("CumStrategy max abs diff: {:.10}", max_cum_diff);
    println!("CumStrategy max value: {:.10}", max_cum_val);
    println!("CumStrategy relative diff: {:.6}", max_cum_diff / max_cum_val);

    // Print average strategies for comparison
    let cpu_strat = cpu_solver.get_average_strategy(0, 2, nh);
    let gpu_strat = gpu_solver.get_average_strategy(0, 2, nh);
    println!("\nP0 strategy (check/bet):");
    println!("  CPU first 5 hands check: {:?}", &cpu_strat[0][..5.min(nh)]);
    println!("  GPU first 5 hands check: {:?}", &gpu_strat[0][..5.min(nh)]);
    println!("  CPU first 5 hands bet:   {:?}", &cpu_strat[1][..5.min(nh)]);
    println!("  GPU first 5 hands bet:   {:?}", &gpu_strat[1][..5.min(nh)]);

    // Show actual exploitability comparison
    use solver_core::solver::best_response::{StrategyProfile, exploitability};

    let cpu_profile = StrategyProfile::from_usize_offsets(
        cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh,
    );
    let gpu_cum_vec = gpu_solver.cum_strategy_slice();
    let gpu_offsets = gpu_solver.node_offsets();
    let gpu_profile = StrategyProfile::from_usize_offsets(
        &gpu_cum_vec, &gpu_offsets, nh,
    );
    let cpu_exp = exploitability(&tree, &game, &cpu_profile);
    let gpu_exp = exploitability(&tree, &game, &gpu_profile);

    println!("\n--- Exploitability (100 iters, same algorithm: run()) ---");
    println!("CPU VCFR  exploitability:  {:.10}", cpu_exp);
    println!("Metal VCFR exploitability: {:.10}", gpu_exp);
    if cpu_exp > 0.0 {
        println!("ACTUAL RATIO (no clamping): {:.4}", gpu_exp / cpu_exp);
    }
}
