/// Cross-check: Verify the "Metal 8x better" result is genuine.
///
/// 1. Take Metal's final strategy at iteration 10000, evaluate exploitability
///    using the same CPU exploitability code as the CPU solver.
/// 2. Take CPU's final strategy, evaluate using the same code.
/// 3. Compare: is Metal's strategy actually lower-exploitability?
///
/// Also: compare the raw cum_strategy vectors to see if the strategies
/// themselves are similar or fundamentally different.

use solver_core::gpu_metal::{MetalContext, MetalVectorCfr};
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
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
fn cross_check_exploitability() {
    let ctx = MetalContext::new().expect("MetalContext failed");
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let (sos, soi, sps, spi, _) = game.sorted_opp_arrays();
    let hc = game.hand_cards_gpu();
    let iw: Vec<Vec<f32>> = (0..2).map(|p| game.initial_weight(p as u8)).collect();

    // Run both for 10000 iterations
    let mut cpu = VectorCfr::new(&tree, vec![nh, nh]);
    cpu.run_sequential(&tree, &game, 10000);

    let mut gpu = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &hc, game.num_combinations());
    gpu.run(&ctx, &tree, 10000);

    // Get strategies
    let cpu_cum = cpu.cum_strategy_slice().to_vec();
    let gpu_cum = gpu.cum_strategy_slice();
    let cpu_offs = cpu.node_offsets();
    let gpu_offs = gpu.node_offsets();

    // Both use same exploitability code
    let cpu_prof = StrategyProfile::from_usize_offsets(&cpu_cum, &cpu_offs, nh);
    let gpu_prof = StrategyProfile::from_usize_offsets(&gpu_cum, &gpu_offs, nh);

    let cpu_exp = exploitability(&tree, &game, &cpu_prof);
    let gpu_exp = exploitability(&tree, &game, &gpu_prof);

    // Cross-check: evaluate CPU strategy with Metal's offsets, and vice versa
    // (should be no-ops since both use same exploitability code)
    println!("CPU  exploitability: {:.10}", cpu_exp);
    println!("Metal exploitability: {:.10}", gpu_exp);
    println!("Ratio: {:.4}", gpu_exp / cpu_exp);

    // Compare raw strategies at each decision node
    println!("\n--- Raw strategy comparison ---");
    let stride = solver_core::tree::flat::MAX_NA * nh;
    let num_infosets = tree.num_infosets as usize;
    let mut max_cum_diff = 0.0f32;
    let mut max_cum_idx = 0;
    for infoset in 0..num_infosets {
        let base = infoset * stride;
        let na = tree.nodes[tree.decision_node_ids[infoset] as usize].num_children as usize;
        for a in 0..na {
            for h in 0..nh.min(10) {
                let idx = base + a * nh + h;
                let diff = (cpu_cum[idx] - gpu_cum[idx]).abs();
                if diff > max_cum_diff {
                    max_cum_diff = diff;
                    max_cum_idx = idx;
                }
            }
        }
    }
    println!("Max cum_strategy diff: {:.10} at index {}", max_cum_diff, max_cum_idx);

    // Compare normalized average strategies at root node
    let cpu_strat = cpu.get_average_strategy(0, 2, nh);
    let gpu_strat = gpu.get_average_strategy(0, 2, nh);

    // Show distribution of strategy differences
    let mut strat_diffs: Vec<f32> = Vec::new();
    for a in 0..2 {
        for h in 0..nh {
            strat_diffs.push((cpu_strat[a][h] - gpu_strat[a][h]).abs());
        }
    }
    strat_diffs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    println!("\nStrategy diff distribution (node 0, check/bet):");
    println!("  Max diff:  {:.10}", strat_diffs[0]);
    println!("  P50 diff:  {:.10}", strat_diffs[strat_diffs.len() / 2]);
    println!("  P99 diff:  {:.10}", strat_diffs[(strat_diffs.len() as f64 * 0.99) as usize]);
    println!("  Mean diff: {:.10}", strat_diffs.iter().sum::<f32>() / strat_diffs.len() as f32);

    // Show specific strategy values for a few hands
    println!("\nNode 0 P0 bet probability (first 10 hands):");
    println!("  CPU:  {:?}", &cpu_strat[1][..10.min(nh)]);
    println!("  GPU:  {:?}", &gpu_strat[1][..10.min(nh)]);

    println!("\n--- Per-infoset regret divergence ---");
    let stride = solver_core::tree::flat::MAX_NA * nh;
    let num_infosets = tree.num_infosets as usize;
    let cpu_reg = cpu.regrets_slice();
    let gpu_reg = gpu.regrets_slice();
    for infoset in 0..num_infosets {
        let na = tree.nodes[tree.decision_node_ids[infoset] as usize].num_children as usize;
        for a in 0..na {
            // Find hand with max relative regret divergence
            let mut max_rel = 0.0f32;
            let mut max_h = 0;
            for h in 0..nh {
                let idx = infoset * stride + a * nh + h;
                let cv = cpu_reg[idx];
                let gv = gpu_reg[idx];
                let denom = cv.abs().max(gv.abs()).max(1e-10);
                let rel = (cv - gv).abs() / denom;
                if rel > max_rel { max_rel = rel; max_h = h; }
            }
            let idx = infoset * stride + a * max_h;
            println!("  infoset={} a={}: max_rel_diff={:.4} at h={}, cpu={:.8}, gpu={:.8}",
                infoset, a, max_rel, max_h, cpu_reg[idx], gpu_reg[idx]);
        }
    }

    // Check whether it's the regret or cum_strategy driving the difference
    // Compare regrets
    let max_reg_diff: f32 = cpu_reg.iter().zip(gpu_reg.iter())
        .map(|(c, g)| (c - g).abs()).fold(0.0f32, f32::max);
    let max_reg_val: f32 = cpu_reg.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("\nMax regret diff: {:.10}", max_reg_diff);
    println!("Max regret val:  {:.10}", max_reg_val);
    println!("Relative diff:   {:.6}", max_reg_diff / max_reg_val.max(1e-10));

    // The definitive test: is Metal's strategy genuinely better?
    // Both are evaluated through the SAME exploitability function.
    // If gpu_exp < cpu_exp, Metal's strategy IS lower exploitability.
    let genuine_advantage = gpu_exp < cpu_exp;
    println!("\nMetal genuinely lower exploitability: {}", genuine_advantage);
    if genuine_advantage {
        println!("Metal's strategy is {:.1}x better (ratio {:.4})",
            cpu_exp / gpu_exp, gpu_exp / cpu_exp);
    }
}
