#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::context::GpuContext;
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
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

/// Clean timing measurement.
/// Measures ONLY run_flop_start, no exploitability mixed in.
/// Multiple runs to get a stable number.
#[test]
fn flop_start_clean_timing() {
    let (tree, table) = build_flop_tree();
    let nh = table.num_valid;
    let starting_pot = tree.starting_pot as f64;

    let nt = tree.nodes.iter().filter(|n| n.is_terminal()).count();
    let nc = tree.nodes.iter().filter(|n| n.is_chance()).count();
    let np = tree.nodes.iter().filter(|n| n.is_player()).count();

    println!("\n{}", "=".repeat(60));
    println!("  CLEAN TIMING: GPU flop-start batched VCFR");
    println!("{}", "=".repeat(60));
    println!("Tree: {} nodes (T:{} C:{} P:{}), nh={}, depth={}",
        tree.num_nodes(), nt, nc, np, nh, tree.max_depth);
    println!("Starting pot: {}", starting_pot);
    println!("Chance: {} turn × {} river = {} total",
        table.remaining_deck.len(), table.river_decks[0].len(),
        table.remaining_deck.len() * table.river_decks[0].len());

    let ctx = GpuContext::new().expect("GPU");
    let mut solver = ctx.create_flop_start_vcfr(&tree, &table).expect("solver");

    // Warmup: 1 iter (includes JIT/shader compilation overhead)
    let t0 = std::time::Instant::now();
    solver.run_flop_start(1).expect("warmup");
    let warmup = t0.elapsed().as_secs_f64();
    println!("\nWarmup: 1 iter = {:.3}s ({:.0}ms/i)", warmup, warmup * 1000.0);

    // Measurement run 1: 10 iters
    let t1 = std::time::Instant::now();
    solver.run_flop_start(10).expect("run1");
    let r1 = t1.elapsed().as_secs_f64();
    let ms1 = r1 / 10.0 * 1000.0;
    println!("Run 1: 10 iters = {:.3}s ({:.1}ms/i)", r1, ms1);

    // Measurement run 2: 10 iters
    let t2 = std::time::Instant::now();
    solver.run_flop_start(10).expect("run2");
    let r2 = t2.elapsed().as_secs_f64();
    let ms2 = r2 / 10.0 * 1000.0;
    println!("Run 2: 10 iters = {:.3}s ({:.1}ms/i)", r2, ms2);

    // Measurement run 3: 10 iters
    let t3 = std::time::Instant::now();
    solver.run_flop_start(10).expect("run3");
    let r3 = t3.elapsed().as_secs_f64();
    let ms3 = r3 / 10.0 * 1000.0;
    println!("Run 3: 10 iters = {:.3}s ({:.1}ms/i)", r3, ms3);

    let avg_ms = (ms1 + ms2 + ms3) / 3.0;
    let iters_25s = (25000.0 / avg_ms) as u32;

    println!("\n{}", "=".repeat(60));
    println!("  DEFINITIVE: {:.1}ms/iter (avg of 3 × 10 iter runs)", avg_ms);
    println!("  In 25s budget: ~{} iterations", iters_25s);
    println!("  CPU comparison: ~138,000ms/i → {:.0}x speedup", 138000.0 / avg_ms);
    println!("{}", "=".repeat(60));

    // Also compute exploitability at current state (31 iters total)
    let table2 = FlopChanceTable::compute_flop_start(
        &["2h","7d","Ks"].iter().map(|s| card_from_str(s).unwrap()).collect::<Vec<_>>(),
        &vec![uniform_range(), uniform_range()], 2);
    let game = FlopStartGame::new(table2);
    let offsets = make_offsets(&tree, nh);

    let cum = solver.download_cum_strategy().expect("download");
    let profile = StrategyProfile::from_usize_offsets(&cum, &offsets, nh);
    let exp = exploitability(&tree, &game, &profile);
    let exp_pct = exp as f64 / starting_pot * 100.0;

    println!("\n  After 31 iters: exploitability = {:.4} ({:.2}% of pot)",
        exp, exp_pct);
    println!("{}", "=".repeat(60));
}
