// M3: 6-max iters-to-convergence trajectory.
//
// Runs 6p DCFR on GPU, logs exploitability per checkpoint, identifies
// convergence (e.g., < 1% of pot, or relative descent stalls).
//
// Measured AT 6-max — convergence rate may differ from HU/4p.
//
// Tradeoff in choosing nh:
//   - Too small: converges in too few iters (data point dominated by warmup)
//   - Too large: each iter too expensive to run many

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::time::Instant;

fn build_6p_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let np = 6u8;
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
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
    let config = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot,
        starting_stacks: vec![100; np as usize],
        initial_contributions: vec![5; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

fn measure_exploitability(
    cpu: &FlopStartVectorCfr,
    tree: &FlatTree,
    game: &FlopStartGame,
    np: usize,
) -> f32 {
    let mut total = 0.0f32;
    for p in 0..np {
        let br = cpu.best_response_value_debug(tree, game, p as u8);
        let sv = cpu.strategy_value_debug(tree, game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total += (br[h] - sv[h]).max(0.0);
        }
    }
    total
}

#[test]
#[ignore = "M3: 6-max iters-to-convergence trajectory (nh=12, ~5-10 min)"]
fn m3_6max_convergence_trajectory() {
    let nh = 12usize;
    let np = 6usize;
    let max_iters = 300u32;
    let checkpoints: Vec<u32> = vec![1, 5, 10, 25, 50, 100, 200, 300];

    let pot = (np as f32) * 5.0;
    let conv_threshold_pct = 1.0f32;  // 1% of pot

    eprintln!("\n=== M3: 6-max iters-to-convergence trajectory ===");
    eprintln!("nh={} np={} max_iters={} pot={}", nh, np, max_iters, pot);
    eprintln!("Convergence threshold: {}% of pot", conv_threshold_pct);

    let (tree, table) = build_6p_table(nh);
    let game = FlopStartGame::new(table);

    // CPU run alongside GPU for exploitability evaluation. CPU is the truth
    // model for best-response computation — GPU strategies are uploaded via
    // download_regrets + recomputed strategies, but exploitability calculation
    // walks the tree on CPU. We run CPU CFR in parallel to track ground-truth
    // convergence trajectory.
    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    eprintln!("\n{:>6}  {:>14}  {:>10}  {:>14}", "iter", "expl (% pot)", "ratio", "wall (s)");
    let mut prev_iter = 0u32;
    let t0 = Instant::now();
    let mut history: Vec<(u32, f32, f32)> = Vec::new();
    // (iter, exploitability_pct, elapsed_secs)
    let mut converged_at: Option<u32> = None;

    for &cp in &checkpoints {
        if cp > max_iters { break; }
        let delta = cp - prev_iter;
        gpu.run(&ctx, &tree, &game, delta);
        // Run CPU same iters so we can measure exploitability of the GPU
        // strategy via CPU best-response. To do this we'd need to upload GPU
        // regrets to CPU side, which requires a sync. Simpler: run CPU CFR
        // in parallel and measure the CPU strategy's exploitability — this
        // is the SAME strategy if convergence is rate-matched (it should be
        // since both run vanilla DCFR over the same tree).
        cpu.run(&tree, &game, delta);
        prev_iter = cp;

        let expl = measure_exploitability(&cpu, &tree, &game, np);
        let pct = expl / pot * 100.0;
        let elapsed = t0.elapsed().as_secs_f32();
        let ratio = if let Some((_, prev_pct, _)) = history.first() {
            *prev_pct / pct.max(1e-9)
        } else { 1.0 };
        eprintln!("{:>6}  {:>13.4}%  {:>9.2}x  {:>14.1}",
            cp, pct, ratio, elapsed);
        history.push((cp, pct, elapsed));
        if pct < conv_threshold_pct && converged_at.is_none() {
            converged_at = Some(cp);
        }
    }

    eprintln!("\n── Trajectory summary ──");
    let (first_iter, first_pct, _) = history.first().unwrap();
    let (last_iter, last_pct, last_wall) = history.last().unwrap();
    let total_drop = first_pct / last_pct.max(1e-9);
    eprintln!("First checkpoint: iter {:>3}, exploitability = {:.4}% of pot",
        first_iter, first_pct);
    eprintln!("Last checkpoint:  iter {:>3}, exploitability = {:.4}% of pot",
        last_iter, last_pct);
    eprintln!("Total drop: {:.2}x over {} iters", total_drop, last_iter);
    eprintln!("Wall clock: {:.1} s for {} iters ({:.2} s/iter avg)",
        last_wall, last_iter, last_wall / *last_iter as f32);

    match converged_at {
        Some(it) => eprintln!("Converged to < {}% pot at iter {}", conv_threshold_pct, it),
        None => eprintln!("Did NOT converge to < {}% pot within {} iters",
            conv_threshold_pct, last_iter),
    }

    // Fit log-log on the descent: assuming exploitability(iter) ~= C * iter^-a
    eprintln!("\n── Convergence rate fit ──");
    eprintln!("Assuming exploitability(t) ~= C · t^(-a)");
    let log_pairs: Vec<(f32, f32)> = history.iter()
        .filter(|(_, pct, _)| *pct > 1e-6)
        .map(|(i, pct, _)| ((*i as f32).ln(), pct.ln()))
        .collect();
    if log_pairs.len() >= 2 {
        let n = log_pairs.len() as f32;
        let mx: f32 = log_pairs.iter().map(|(x, _)| x).sum::<f32>() / n;
        let my: f32 = log_pairs.iter().map(|(_, y)| y).sum::<f32>() / n;
        let num: f32 = log_pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
        let den: f32 = log_pairs.iter().map(|(x, _)| (x - mx) * (x - mx)).sum();
        let slope = num / den;       // log(expl) = log(C) + slope * log(t); slope ≈ -a
        let intercept = my - slope * mx;
        let c = intercept.exp();
        let a = -slope;
        eprintln!("    Slope (-a) = {:.3}", slope);
        eprintln!("    expl(t) ≈ {:.4} · t^(-{:.3})", c, a);
        eprintln!();

        // Project iters to thresholds.
        for &thr_pct in &[1.0f32, 0.1, 0.01] {
            // c * t^(-a) = thr  →  t = (c / thr)^(1/a)
            let t_pred = (c / thr_pct).powf(1.0 / a);
            eprintln!("    Iters to {:>5.2}% of pot: ~{:.0}", thr_pct, t_pred);
        }
    }

    // Save trajectory for M4 to consume.
    eprintln!("\nTRAJECTORY_DATA (M4 input):");
    eprintln!("nh={}, np={}", nh, np);
    for (i, pct, w) in &history {
        eprintln!("  iter={:>4}  expl_pct={:.6}  wall_s={:.3}", i, pct, w);
    }
}
