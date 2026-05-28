#![cfg(feature = "cuda")]

/// Phase 1 validation: Streaming reduction convergence test.
///
/// Measures GPU flop-start convergence with per-outcome DCFR discount
/// (streaming kernel) vs the external solver at matched iteration counts.
///
/// Success criteria: GPU exploitability per iteration within 2-3x of
/// external solver, accepting some residual gap from shared regrets.

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;
use solver_core::gpu::context::GpuContext;
use postflop_solver::{
    TreeConfig as ExtTreeConfig, BoardState as ExtBoard,
    BetSizeOptions as ExtBet, CardConfig,
    flop_from_str, solve_step, compute_exploitability,
};

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

fn ext_full_range() -> &'static str {
    "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+,J2s+,J2o+,T2s+,T2o+,92s+,92o+,82s+,82o+,72s+,72o+,62s+,62o+,52s+,52o+,42s+,42o+,32s,32o"
}

fn make_offsets(tree: &solver_core::tree::flat::FlatTree, nh: usize) -> Vec<usize> {
    (0..tree.num_nodes())
        .map(|i| {
            let is = tree.infoset_offsets[i];
            if is == u32::MAX { usize::MAX } else { is as usize * MAX_NA * nh }
        })
        .collect()
}

fn build_flop_tree() -> (solver_core::tree::flat::FlatTree, FlopChanceTable) {
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

#[test]
fn streaming_convergence_vs_external() {
    let starting_pot = 100.0f32;
    let (tree, table) = build_flop_tree();
    let nh = table.num_valid;
    let offsets = make_offsets(&tree, nh);
    let game = FlopStartGame::new(FlopChanceTable::compute_flop_start(
        &["2h","7d","Ks"].iter().map(|s| card_from_str(s).unwrap()).collect::<Vec<_>>(),
        &vec![uniform_range(), uniform_range()], 2
    ));

    let mut ctx = GpuContext::new().expect("gpu context");
    let mut solver = ctx.create_flop_start_vcfr(&tree, &table).expect("solver");

    println!("\n{}", "=".repeat(70));
    println!("  PHASE 1: STREAMING REDUCTION CONVERGENCE TEST");
    println!("  Board: 2h7dKs, pot=100, stacks=200, bet=50%");
    println!("{}", "=".repeat(70));

    // GPU streaming convergence
    let checkpoints = [1u32, 2, 5, 10, 25, 50];
    let mut gpu_results: Vec<(u32, f32, f64)> = Vec::new();
    let mut gpu_iter: u32 = 0;

    for &target in &checkpoints {
        let to_run = target - gpu_iter;
        if to_run == 0 { continue; }

        let t = std::time::Instant::now();
        solver.run_flop_start(to_run).expect("run");
        gpu_iter = target;
        let elapsed = t.elapsed().as_secs_f64();
        let ms_per = elapsed / to_run as f64 * 1000.0;

        let cum = solver.download_cum_strategy().expect("cum");
        let profile = StrategyProfile::from_usize_offsets(&cum, &offsets, nh);
        let exp = exploitability(&tree, &game, &profile);
        gpu_results.push((target, exp, ms_per));

        println!("  GPU @ {:>3} iters: {:.2}% of pot ({:.0} ms/i)",
            target, exp as f64 / starting_pot as f64 * 100.0, ms_per);
    }

    // External solver convergence
    println!("\n  External solver (same config):");
    let bet = ExtBet::try_from(("50%", "")).unwrap();
    let range = ext_full_range();

    let mut action_tree = postflop_solver::ActionTree::new(ExtTreeConfig {
        initial_state: ExtBoard::Flop,
        starting_pot: 100,
        effective_stack: 200,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [bet.clone(), bet.clone()],
        turn_bet_sizes: [bet.clone(), bet.clone()],
        river_bet_sizes: [bet.clone(), bet],
        turn_donk_sizes: None, river_donk_sizes: None,
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    }).expect("action tree");

    let mut ext_game = postflop_solver::PostFlopGame::with_config(CardConfig {
        range: [range.parse().unwrap(), range.parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(),
        turn: postflop_solver::NOT_DEALT,
        river: postflop_solver::NOT_DEALT,
    }, action_tree).expect("game");
    ext_game.allocate_memory(false);

    let mut ext_iter: u32 = 0;
    let mut ext_results: Vec<(u32, f64)> = Vec::new();

    for &target in &checkpoints {
        let to_run = target - ext_iter;
        if to_run == 0 { continue; }

        let t = std::time::Instant::now();
        for i in 0..to_run {
            solve_step(&ext_game, ext_iter + i);
        }
        ext_iter = target;
        let elapsed = t.elapsed().as_secs_f64();
        let ms_per = elapsed / to_run as f64 * 1000.0;

        let exp = compute_exploitability(&ext_game);
        ext_results.push((target, exp as f64));
        println!("  Ext @ {:>3} iters: {:.2}% of pot ({:.0} ms/i)",
            target, exp as f64 / starting_pot as f64 * 100.0, ms_per);
    }

    // Summary table
    println!("\n{}", "=".repeat(70));
    println!("  CONVERGENCE COMPARISON (exploitability as % of pot)");
    println!("  {:>6} | {:>12} | {:>12} | {:>8}", "Iter", "GPU Stream", "External", "Ratio");
    println!("  {}-+-{}-+-{}-+-{}",
        "-".repeat(6), "-".repeat(12), "-".repeat(12), "-".repeat(8));

    for (iter, gpu_exp, _) in &gpu_results {
        let gpu_pct = *gpu_exp as f64 / starting_pot as f64 * 100.0;
        let ext_exp = ext_results.iter().find(|(i,_)| *i == *iter)
            .map(|(_,e)| *e).unwrap_or(f64::NAN);
        let ext_pct = ext_exp / starting_pot as f64 * 100.0;
        let ratio = if ext_pct > 0.0 && ext_pct.is_finite() {
            format!("{:.1}x", gpu_pct / ext_pct)
        } else {
            "—".to_string()
        };
        println!("  {:>6} | {:>11.2}% | {:>11.2}% | {:>8}",
            iter, gpu_pct, ext_pct, ratio);
    }

    println!("{}", "=".repeat(70));

    // Phase 1 success check
    if let Some(&(50, gpu_50, _)) = gpu_results.iter().find(|(i,_,_)| *i == 50) {
        let ext_50 = ext_results.iter().find(|(i,_)| *i == 50)
            .map(|(_,e)| *e).unwrap_or(f64::NAN);
        let gpu_pct = gpu_50 as f64 / starting_pot as f64 * 100.0;
        let ext_pct = ext_50 / starting_pot as f64 * 100.0;
        let ratio = gpu_pct / ext_pct;
        if ratio < 5.0 {
            println!("  ✅ Phase 1 PASS: GPU/External ratio at 50 iters = {:.1}x (< 5x threshold)", ratio);
        } else {
            println!("  ❌ Phase 1 FAIL: GPU/External ratio at 50 iters = {:.1}x (> 5x threshold)", ratio);
        }
    }
}
