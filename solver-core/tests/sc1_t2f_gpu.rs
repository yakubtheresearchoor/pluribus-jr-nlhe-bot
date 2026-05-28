#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::context::GpuContext;
use solver_core::solver::flop_start_game::FlopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

#[test]
fn sc1_t2f_gpu_actual() {
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

    let nt = tree.nodes.iter().filter(|n| n.is_terminal()).count();
    let nc = tree.nodes.iter().filter(|n| n.is_chance()).count();
    let np = tree.nodes.iter().filter(|n| n.is_player()).count();

    println!("\n{}", "=".repeat(60));
    println!("  SC1: T2F GPU ACTUAL MEASUREMENT");
    println!("{}", "=".repeat(60));
    println!("Tree: {} nodes (T:{} C:{} P:{}), nh={}, depth={}",
        tree.num_nodes(), nt, nc, np, table.num_valid, tree.max_depth);
    println!("Chance: {} turn × {} river = {} total",
        table.remaining_deck.len(), table.river_decks[0].len(),
        table.remaining_deck.len() * table.river_decks[0].len());

    let ctx = GpuContext::new().expect("GPU");
    let mut solver = ctx.create_flop_start_vcfr(&tree, &table).expect("solver");

    // Warm up (1 iter)
    let t0 = std::time::Instant::now();
    solver.run_flop_start(1).expect("warmup");
    let warmup = t0.elapsed().as_secs_f64();
    println!("Warmup: 1 iter {:.2}s", warmup);

    // Measurement
    let n = 3u32;
    let t1 = std::time::Instant::now();
    solver.run_flop_start(n).expect("run");
    let elapsed = t1.elapsed().as_secs_f64();
    let ms_per_iter = elapsed / n as f64 * 1000.0;

    println!("GPU flop-start: {} iters {:.2}s ({:.0}ms/i)", n, elapsed, ms_per_iter);

    let iters_25s = (25000.0 / ms_per_iter) as u32;
    println!("In 25s: ~{} iterations", iters_25s);

    println!("\n{}", "=".repeat(60));
    if ms_per_iter < 500.0 {
        println!("  VERDICT: GPU flop-start VIABLE ({:.0}ms/i)", ms_per_iter);
    } else if ms_per_iter < 3000.0 {
        println!("  VERDICT: GPU flop-start MARGINAL ({:.0}ms/i)", ms_per_iter);
    } else {
        println!("  VERDICT: GPU flop-start TOO SLOW ({:.0}ms/i)", ms_per_iter);
    }
    println!("  Serial kernel launches: ~{} per iteration",
        table.remaining_deck.len() * (table.river_decks[0].len() * 6 + 10));
    println!("  Next: batched kernel to reduce launch count");
    println!("{}", "=".repeat(60));
}
