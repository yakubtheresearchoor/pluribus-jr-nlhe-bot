/// Convergence plateau investigation.
/// Measures CPU and Metal exploitability at fine-grained checkpoints
/// to determine whether the 0.12 ratio at 2000 iterations is a snapshot
/// effect or a persistent difference.

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

fn make_gpu_solver(ctx: &MetalContext, tree: &FlatTree, game: &RiverPokerGame) -> MetalVectorCfr {
    let nh = game.num_valid_hands();
    let (sos, soi, sps, spi, _) = game.sorted_opp_arrays();
    let hc = game.hand_cards_gpu();
    let iw: Vec<Vec<f32>> = (0..2).map(|p| game.initial_weight(p as u8)).collect();
    MetalVectorCfr::new(ctx, tree, nh, &iw, &sos, &soi, &sps, &spi, &hc, game.num_combinations())
}

#[test]
fn convergence_plateau_10000() {
    let ctx = MetalContext::new().expect("MetalContext failed");
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let mut cpu = VectorCfr::new(&tree, vec![nh, nh]);
    let mut gpu = make_gpu_solver(&ctx, &tree, &game);

    let checkpoints: Vec<u32> = vec![
        100, 250, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000
    ];
    let mut prev = 0u32;

    println!("iter      CPU_exp          Metal_exp        ratio");
    println!("----      -------          ---------        -----");

    for cp in &checkpoints {
        let iters = cp - prev;
        prev = *cp;

        cpu.run_sequential(&tree, &game, iters);
        gpu.run(&ctx, &tree, iters);

        let cpu_prof = StrategyProfile::from_usize_offsets(
            cpu.cum_strategy_slice(), cpu.node_offsets(), nh,
        );
        let gc = gpu.cum_strategy_slice();
        let go = gpu.node_offsets();
        let gpu_prof = StrategyProfile::from_usize_offsets(&gc, &go, nh);

        let ce = exploitability(&tree, &game, &cpu_prof);
        let ge = exploitability(&tree, &game, &gpu_prof);
        let ratio = ge / ce.max(1e-10);

        println!("{:5}      {:.10}  {:.10}  {:.4}", cp, ce, ge, ratio);
    }

    // Print final state analysis
    let cpu_prof = StrategyProfile::from_usize_offsets(
        cpu.cum_strategy_slice(), cpu.node_offsets(), nh,
    );
    let gc = gpu.cum_strategy_slice();
    let go = gpu.node_offsets();
    let gpu_prof = StrategyProfile::from_usize_offsets(&gc, &go, nh);
    let ce = exploitability(&tree, &game, &cpu_prof);
    let ge = exploitability(&tree, &game, &gpu_prof);

    println!("\n--- Final state @ 10000 iters ---");
    println!("CPU:  {:.10}", ce);
    println!("Metal: {:.10}", ge);
    println!("Min of both: {:.10}", ce.min(ge));
    println!("Ratio: {:.4}", ge / ce.max(1e-10));

    // Both should converge well
    assert!(ce < 0.001, "CPU should be < 0.001 at 10000 iters, got {:.6}", ce);
    assert!(ge < 0.001, "Metal should be < 0.001 at 10000 iters, got {:.6}", ge);
}
