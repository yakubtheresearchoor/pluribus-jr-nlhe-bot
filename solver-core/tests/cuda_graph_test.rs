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

/// Test CUDA graph capture/replay for flop-start VCFR.
/// Measures whether CUDA graphs eliminate launch overhead.
#[test]
fn flop_start_cuda_graph() {
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

    let ctx = GpuContext::new().expect("GPU");

    // Test 1: Without CUDA graphs (baseline)
    println!("\n=== WITHOUT CUDA GRAPHS ===");
    let mut solver1 = ctx.create_flop_start_vcfr(&tree, &table).expect("solver1");
    solver1.run_flop_start(1).expect("warmup");
    let t1 = std::time::Instant::now();
    solver1.run_flop_start(10).expect("run");
    let without_graph = t1.elapsed().as_secs_f64() / 10.0 * 1000.0;
    println!("Without graph: {:.1}ms/i", without_graph);

    // Test 2: With CUDA graphs
    println!("\n=== WITH CUDA GRAPHS ===");
    let mut solver2 = ctx.create_flop_start_vcfr(&tree, &table).expect("solver2");
    std::env::set_var("RUST_BACKTRACE", "1");
    let t2 = std::time::Instant::now();
    solver2.run_flop_start_graph(11).expect("run");
    let with_graph = t2.elapsed().as_secs_f64() / 11.0 * 1000.0;
    println!("With graph: {:.1}ms/i (includes capture overhead on first iter)", with_graph);

    // Measure graph replay only (skip first iteration which includes capture)
    let t3 = std::time::Instant::now();
    solver2.run_flop_start_graph(10).expect("replay");
    let replay_only = t3.elapsed().as_secs_f64() / 10.0 * 1000.0;
    println!("Graph replay: {:.1}ms/i (pure replay, no capture)", replay_only);

    let speedup = without_graph / replay_only;
    println!("\nSpeedup from CUDA graphs: {:.1}x", speedup);

    // Convergence check
    let cum = solver2.download_cum_strategy().expect("download");
    let profile = StrategyProfile::from_usize_offsets(&cum, &offsets, nh);
    let exp = exploitability(&tree, &game, &profile);
    let pct = exp as f64 / starting_pot as f64 * 100.0;
    println!("After 21 iters (graph): exploitability = {:.4} ({:.2}% of pot)", exp, pct);

    println!("\n{}", "=".repeat(60));
    if replay_only < 200.0 {
        println!("  VERDICT: CUDA graphs VIABLE ({:.1}ms/i)", replay_only);
    } else if replay_only < without_graph * 0.5 {
        println!("  VERDICT: CUDA graphs HELPFUL ({:.1}ms/i, {:.1}x faster)", replay_only, speedup);
    } else {
        println!("  VERDICT: CUDA graphs NOT EFFECTIVE ({:.1}ms/i, {:.1}x faster)", replay_only, speedup);
    }
    println!("{}", "=".repeat(60));

    // Note: the graph captures fixed params (alpha, beta, gamma from iter 0).
    // This means DCFR discounting is wrong for iter > 0. This test measures
    // throughput only; a production version would need updatable graph params.
}
