#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::context::GpuContext;
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
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

/// Compare CPU vs GPU convergence on the same flop-start tree.
/// CPU uses exact sequential DCFR. GPU uses batched atomic accumulation.
/// This measures the convergence penalty of the atomic approximation.
#[test]
fn cpu_vs_gpu_convergence() {
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
    let game = FlopStartGame::new(FlopChanceTable::compute_flop_start(&board, &ranges, 2));

    println!("\n{}", "=".repeat(70));
    println!("  CPU vs GPU CONVERGENCE COMPARISON (flop-start, 2h7dKs)");
    println!("{}", "=".repeat(70));
    println!("Tree: {} nodes, nh={}, pot={}", tree.num_nodes(), nh, starting_pot);

    // CPU convergence
    println!("\n--- CPU VCFR (exact sequential DCFR) ---");
    let num_hands_vec = (0..tree.num_players).map(|_| nh).collect();
    let mut cpu_solver = VectorCfr::new(&tree, num_hands_vec);
    let checkpoints_cpu: &[u32] = &[1, 5];
    let mut cpu_accumulated = 0u32;

    println!("{:<8} {:<12} {:<10} {:<10}", "Iters", "Exploit", "% pot", "Time");
    for &target in checkpoints_cpu {
        let delta = target - cpu_accumulated;
        let t0 = std::time::Instant::now();
        cpu_solver.run_sequential(&tree, &game, delta);
        let elapsed = t0.elapsed().as_secs_f64();
        cpu_accumulated = target;

        let cum = cpu_solver.cum_strategy_slice();
        let profile = StrategyProfile::from_usize_offsets(cum, &offsets, nh);
        let exp = exploitability(&tree, &game, &profile);
        let pct = exp as f64 / starting_pot as f64 * 100.0;
        println!("{:<8} {:<12.4} {:<10.2}% {:<10.1}s", target, exp, pct, elapsed);
    }

    // GPU convergence
    println!("\n--- GPU VCFR (batched atomic accumulation) ---");
    let ctx = GpuContext::new().expect("GPU");
    let mut gpu_solver = ctx.create_flop_start_vcfr(&tree, &table).expect("solver");

    // Warmup
    gpu_solver.run_flop_start(1).expect("warmup");

    let checkpoints_gpu: &[u32] = &[1, 5, 10];
    let mut gpu_accumulated = 0u32;
    println!("{:<8} {:<12} {:<10} {:<10}", "Iters", "Exploit", "% pot", "Time");
    for &target in checkpoints_gpu {
        let delta = target - gpu_accumulated;
        let t0 = std::time::Instant::now();
        gpu_solver.run_flop_start(delta).expect("run");
        let elapsed = t0.elapsed().as_secs_f64();
        gpu_accumulated = target;

        let cum = gpu_solver.download_cum_strategy().expect("download");
        let profile = StrategyProfile::from_usize_offsets(&cum, &offsets, nh);
        let exp = exploitability(&tree, &game, &profile);
        let pct = exp as f64 / starting_pot as f64 * 100.0;
        println!("{:<8} {:<12.4} {:<10.2}% {:<10.1}s", target, exp, pct, elapsed);
    }

    println!("\n{}", "=".repeat(70));
    println!("  If CPU converges significantly faster than GPU,");
    println!("  the atomic approximation is the convergence bottleneck.");
    println!("{}", "=".repeat(70));
}
