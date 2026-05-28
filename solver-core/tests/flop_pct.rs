#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::context::GpuContext;
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

fn make_offsets(tree: &solver_core::tree::flat::FlatTree, nh: usize) -> Vec<usize> {
    (0..tree.num_nodes())
        .map(|i| {
            let is = tree.infoset_offsets[i];
            if is == u32::MAX { usize::MAX } else { is as usize * MAX_NA * nh }
        })
        .collect()
}

#[test]
fn flop_convergence_pct() {
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
    let nh = table.num_valid;
    let starting_pot = tree.starting_pot;
    let offsets = make_offsets(&tree, nh);

    let table2 = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    let game = FlopStartGame::new(table2);

    let ctx = GpuContext::new().expect("GPU");
    let mut solver = ctx.create_flop_start_vcfr(&tree, &table).expect("solver");

    // Timing-only run first (no exploitability overhead)
    let t_timing = std::time::Instant::now();
    solver.run_flop_start(10).expect("run");
    let gpu_time = t_timing.elapsed().as_secs_f64();
    let ms_per_iter = gpu_time / 10.0 * 1000.0;
    let iters_in_25s = (25000.0 / ms_per_iter) as u32;
    println!("\nGPU timing: 10 iters = {:.3}s ({:.1}ms/i), in 25s: ~{} iters",
        gpu_time, ms_per_iter, iters_in_25s);

    // Reset solver for convergence curve
    // (Can't easily reset, so create a new one)
    let mut solver2 = ctx.create_flop_start_vcfr(&tree, &table).expect("solver2");

    let checkpoints: &[u32] = &[1, 5, 10, 25];
    let mut accumulated = 0u32;

    println!("\n{:<12} {:<12} {:<12} {:<12}", "Iters", "Exploit", "% of pot", "GPU time");
    println!("{}", "-".repeat(48));

    for &target in checkpoints {
        let delta = target - accumulated;
        let t0 = std::time::Instant::now();
        solver2.run_flop_start(delta).expect("run");
        let elapsed = t0.elapsed().as_secs_f64();
        accumulated = target;

        let cum = solver2.download_cum_strategy().expect("download");
        let profile = StrategyProfile::from_usize_offsets(&cum, &offsets, nh);
        let exp = exploitability(&tree, &game, &profile);
        let pct = exp as f64 / starting_pot as f64 * 100.0;

        println!("{:<12} {:<12.2} {:<8.2}% {:<12.2}s",
            target, exp, pct, elapsed);
    }

    let final_pct = {
        let cum = solver2.download_cum_strategy().expect("download");
        let profile = StrategyProfile::from_usize_offsets(&cum, &offsets, nh);
        let exp = exploitability(&tree, &game, &profile);
        exp as f64 / starting_pot as f64 * 100.0
    };

    println!("\n{}", "=".repeat(60));
    println!("  Timing:     {:.1}ms/iter (definitive)", ms_per_iter);
    println!("  In 25s:     ~{} iterations", iters_in_25s);
    println!("  50 iter exp: {:.2}% of pot", final_pct);
    println!("  Target:     3-5% of pot");
    if final_pct < 5.0 {
        println!("  VERDICT:    VIABLE at 50 iters");
    } else if final_pct < 10.0 {
        println!("  VERDICT:    MARGINAL — needs more iters or better convergence");
    } else {
        println!("  VERDICT:    NOT VIABLE — convergence too slow");
    }
    println!("{}", "=".repeat(60));
}
