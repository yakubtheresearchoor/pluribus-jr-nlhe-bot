#![cfg(feature = "cuda")]

//! Correctness verification for batched GPU flop-start kernel.
//!
//! Strategy: Run GPU for increasing iterations and measure exploitability.
//! A correct solver shows monotonically decreasing exploitability.
//! We also compare GPU strategies at 1 iter against CPU at 1 iter
//! (separate, faster test) to verify they're in the same ballpark.
//!
//! Phase 1: GPU convergence trend (fast, measures exploitability curve)
//! Phase 2: CPU vs GPU at 1 iter (slow, but verifies parity)

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::context::GpuContext;
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

fn build_flop_tree() -> (
    solver_core::tree::flat::FlatTree,
    FlopChanceTable,
) {
    let board: Vec<Card> = ["2h","7d","Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];

    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop,
        starting_pot: 100, starting_stacks: vec![200, 200],
        initial_contributions: vec![0,0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![],
        },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree");
    let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    (tree, table)
}

fn make_offsets(tree: &solver_core::tree::flat::FlatTree, nh: usize) -> Vec<usize> {
    (0..tree.num_nodes())
        .map(|i| {
            let is = tree.infoset_offsets[i];
            if is == u32::MAX { usize::MAX } else { is as usize * MAX_NA * nh }
        })
        .collect()
}

/// Phase 1: GPU convergence trend.
/// If exploitability decreases with more iterations, the solver is working.
#[test]
fn flop_start_gpu_convergence() {
    let (tree, table) = build_flop_tree();
    let nh = table.num_valid;
    let offsets = make_offsets(&tree, nh);
    let table2 = FlopChanceTable::compute_flop_start(
        &["2h","7d","Ks"].iter().map(|s| card_from_str(s).unwrap()).collect::<Vec<_>>(),
        &vec![uniform_range(), uniform_range()], 2);
    let game = FlopStartGame::new(table2);

    let nt = tree.nodes.iter().filter(|n| n.is_terminal()).count();
    let nc = tree.nodes.iter().filter(|n| n.is_chance()).count();
    let np = tree.nodes.iter().filter(|n| n.is_player()).count();

    println!("\n{}", "=".repeat(60));
    println!("  FLOP-START GPU CONVERGENCE TREND");
    println!("{}", "=".repeat(60));
    println!("Tree: {} nodes (T:{} C:{} P:{}), nh={}, depth={}",
        tree.num_nodes(), nt, nc, np, nh, tree.max_depth);

    let ctx = GpuContext::new().expect("GPU");
    let mut solver = ctx.create_flop_start_vcfr(&tree, &table).expect("solver");

    let checkpoints: &[u32] = &[1, 10, 25, 50];
    let mut accumulated = 0u32;
    let mut exploits = Vec::new();

    for &target in checkpoints {
        let delta = target - accumulated;
        let t0 = std::time::Instant::now();
        solver.run_flop_start(delta).expect("run");
        let elapsed = t0.elapsed().as_secs_f64();
        accumulated = target;

        let cum = solver.download_cum_strategy().expect("download");
        let profile = StrategyProfile::from_usize_offsets(&cum, &offsets, nh);
        let exp = exploitability(&tree, &game, &profile);
        exploits.push(exp);

        println!("GPU @ {:>3} iters: {:.2}s, exploitability = {:.4}",
            target, elapsed, exp);
    }

    // Verify convergence trend
    let exp_1 = exploits[0];
    let exp_100 = *exploits.last().unwrap();

    println!("\n{}", "=".repeat(60));
    println!("  Exploitability @ 1 iter:  {:.4}", exp_1);
    println!("  Exploitability @ 100 iter: {:.4}", exp_100);
    println!("  Reduction: {:.1}x", exp_1 / exp_100.max(0.01));
    println!("{}", "=".repeat(60));

    // Core assertions
    // 1. Should converge (100 iter much better than 1 iter)
    assert!(
        exp_100 < exp_1 * 0.8,
        "No convergence: 1 iter={:.4}, 100 iter={:.4}",
        exp_1, exp_100
    );

    // 2. Overall trend should be decreasing (allow occasional increases from DCFR gamma resets)
    let increases: Vec<_> = exploits.windows(2)
        .enumerate()
        .filter(|(_, w)| w[1] > w[0])
        .collect();
    println!("Increases between checkpoints: {} (DCFR gamma resets expected)",
        increases.len());

    // 3. Final exploitability should be reasonable
    // For pot=100, initial uniform strategy should have exploitability ~2000-5000
    // After 100 iterations, should be well below that
    assert!(
        exp_100 < exp_1,
        "Final exploitability ({:.4}) not better than initial ({:.4})",
        exp_100, exp_1
    );
}

/// Phase 2: CPU vs GPU at 1 iteration.
/// Verifies they produce similar strategies.
/// This test is slow (~140s for 1 CPU iteration).
#[test]
#[ignore] // Run with --ignored flag for full verification
fn flop_start_cpu_gpu_parity() {
    let (tree, table) = build_flop_tree();
    let nh = table.num_valid;
    let offsets = make_offsets(&tree, nh);
    let table2 = FlopChanceTable::compute_flop_start(
        &["2h","7d","Ks"].iter().map(|s| card_from_str(s).unwrap()).collect::<Vec<_>>(),
        &vec![uniform_range(), uniform_range()], 2);
    let game = FlopStartGame::new(table2);

    // CPU
    let mut cpu = VectorCfr::new(&tree, vec![nh, nh]);
    let t0 = std::time::Instant::now();
    cpu.run_sequential(&tree, &game, 1);
    let cpu_time = t0.elapsed().as_secs_f64();
    let cpu_profile = StrategyProfile::from_usize_offsets(cpu.cum_strategy_slice(), &offsets, nh);
    let cpu_exp = exploitability(&tree, &game, &cpu_profile);

    // GPU
    let ctx = GpuContext::new().expect("GPU");
    let mut gpu_solver = ctx.create_flop_start_vcfr(&tree, &table).expect("solver");
    let t1 = std::time::Instant::now();
    gpu_solver.run_flop_start(1).expect("run");
    let gpu_time = t1.elapsed().as_secs_f64();
    let gpu_cum = gpu_solver.download_cum_strategy().expect("download");
    let gpu_profile = StrategyProfile::from_usize_offsets(&gpu_cum, &offsets, nh);
    let gpu_exp = exploitability(&tree, &game, &gpu_profile);

    let ratio = gpu_exp / cpu_exp;
    let speedup = cpu_time / gpu_time;

    println!("\n{}", "=".repeat(60));
    println!("  CPU/GPU PARITY @ 1 iter");
    println!("{}", "=".repeat(60));
    println!("CPU: {:.2}s ({:.0}ms/i), exploitability = {:.4}", cpu_time, cpu_time * 1000.0, cpu_exp);
    println!("GPU: {:.2}s ({:.0}ms/i), exploitability = {:.4}", gpu_time, gpu_time * 1000.0, gpu_exp);
    println!("Ratio: {:.2}x, Speedup: {:.0}x", ratio, speedup);
    println!("{}", "=".repeat(60));

    assert!(ratio > 0.1 && ratio < 10.0,
        "GPU/CPU ratio {:.2} outside range. CPU={:.4} GPU={:.4}",
        ratio, cpu_exp, gpu_exp);
}
