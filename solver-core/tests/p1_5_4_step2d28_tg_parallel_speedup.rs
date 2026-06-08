// Step 2.D.28: tg-parallel kernel speedup measurement.
//
// Measures wall-clock time for 6p (K=5) GPU iterations at small nh,
// comparing the new threadgroup-parallel kernel against the serial
// batched kernel. Toggle via env SOLVER_DISABLE_TG_PARALLEL=1.
//
// Run sequence:
//   cargo test --release --features metal --test p1_5_4_step2d28_tg_parallel_speedup -- --ignored --nocapture
//
// Then for A/B against old kernel:
//   SOLVER_DISABLE_TG_PARALLEL=1 cargo test --release --features metal --test p1_5_4_step2d28_tg_parallel_speedup -- --ignored --nocapture
#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_np_table(np: u8, nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();

    let mut ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..np as usize {
        for &hi in &chosen {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
            ranges[p][pair_idx] = 1.0;
        }
    }

    let turn_cards = vec![card_from_str("3c").unwrap() as u8];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![card_from_str("5s").unwrap() as u8];

    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, np, &chosen, &turn_cards, &river_decks,
    );
    let starting_pot: i32 = (np as i32) * 5;
    let stacks: Vec<i32> = vec![100; np as usize];
    let contribs: Vec<i32> = vec![5; np as usize];
    let config = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot,
        starting_stacks: stacks,
        initial_contributions: contribs,
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

fn measure_gpu_iters(np: u8, nh: usize, n_iters: u32) -> f64 {
    let (tree, table) = build_np_table(np, nh);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Warmup (cold pipelines).
    gpu.run(&ctx, &tree, &game, 2);

    let t = std::time::Instant::now();
    gpu.run(&ctx, &tree, &game, n_iters);
    let secs = t.elapsed().as_secs_f64();
    secs * 1000.0 / n_iters as f64
}

#[test]
#[ignore = "Step 2.D.28: tg-parallel speedup measurement (run explicitly)"]
fn step2d28_tg_parallel_6p_speedup() {
    let env = std::env::var_os("SOLVER_DISABLE_TG_PARALLEL").is_some();
    let mode = if env { "OLD (serial)" } else { "NEW (tg-parallel)" };

    eprintln!("\n=== Step 2.D.28: 6p kernel-mode timing — {} ===", mode);
    eprintln!("Set SOLVER_DISABLE_TG_PARALLEL=1 to measure old kernel.\n");

    eprintln!("{:>4}  {:>5}  {:>14}", "np", "nh", "ms/iter");

    for &(np, nh, n_iters) in &[
        (4u8, 6usize, 10u32),
        (4, 10, 10),
        (5, 6, 10),
        (6, 6, 10),
        (6, 7, 10),
    ] {
        let ms = measure_gpu_iters(np, nh, n_iters);
        eprintln!("{:>4}  {:>5}  {:>14.3}", np, nh, ms);
    }
}

#[test]
#[ignore = "Step 2.D.28: direct A/B (auto-toggles env between runs)"]
fn step2d28_tg_parallel_ab_compare() {
    // A/B sweep across np and nh, in-process, both modes back-to-back.
    eprintln!("\n=== Step 2.D.28: A/B sweep ===");
    eprintln!("{:>4}  {:>4}  {:>12}  {:>12}  {:>10}", "np", "nh", "OLD ms/it", "NEW ms/it", "speedup");

    for &(np, nh, n_iters) in &[
        (4u8, 20usize, 5u32),
        (4, 30, 5),
        (4, 50, 3),
        (5, 16, 5),
        (5, 20, 3),
        (6, 12, 5),
        (6, 16, 3),
    ] {
        std::env::set_var("SOLVER_DISABLE_TG_PARALLEL", "1");
        let old_ms = measure_gpu_iters(np, nh, n_iters);
        std::env::remove_var("SOLVER_DISABLE_TG_PARALLEL");
        let new_ms = measure_gpu_iters(np, nh, n_iters);
        let speedup = old_ms / new_ms.max(1e-9);
        eprintln!("{:>4}  {:>4}  {:>12.3}  {:>12.3}  {:>9.2}x", np, nh, old_ms, new_ms, speedup);
    }
}
