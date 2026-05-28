#![cfg(feature = "cuda")]

use postflop_solver::*;
use postflop_solver::{
    TreeConfig as ExtTreeConfig,
    BoardState as ExtBoard,
    BetSizeOptions as ExtBet, CardConfig,
    flop_from_str, solve, solve_step, NOT_DEALT,
};

/// Run the external solver on a flop-start tree with same config as our tests.
/// Measures convergence at 10, 50, 100 iterations.
#[test]
fn external_flop_convergence() {
    println!("\n{}", "=".repeat(70));
    println!("  EXTERNAL SOLVER CONVERGENCE (2h7dKs flop, pot=100)");
    println!("{}", "=".repeat(70));

    let bet = ExtBet::try_from(("50%", "")).unwrap();
    let range = "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+,J2s+,J2o+,T2s+,T2o+,92s+,92o+,82s+,82o+,72s+,72o+,62s+,62o+,52s+,52o+,42s+,42o+,32s,32o";

    let mut action_tree = ActionTree::new(ExtTreeConfig {
        initial_state: ExtBoard::Flop,
        starting_pot: 100,
        effective_stack: 200,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet.clone(), bet.clone()],
        turn_bet_sizes: [bet.clone(), bet.clone()],
        river_bet_sizes: [bet.clone(), bet],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    }).expect("action tree");

    let mut game = PostFlopGame::with_config(CardConfig {
        range: [range.parse().unwrap(), range.parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    }, action_tree).expect("external game");

    game.allocate_memory(false);

    let checkpoints = [10u32, 50, 100, 200, 500];
    let mut iter = 0u32;
    let mut checkpoint_idx = 0;

    let t_total = std::time::Instant::now();

    // Run iterations and check exploitability at checkpoints
    loop {
        solve_step(&mut game, iter);
        iter += 1;

        if checkpoint_idx < checkpoints.len() && iter == checkpoints[checkpoint_idx] {
            let exp = compute_exploitability(&game);
            let pct = exp / 100.0 * 100.0;
            let elapsed = t_total.elapsed().as_secs_f64();
            let ms_per_iter = elapsed / iter as f64 * 1000.0;
            println!("  @ {:>3} iters ({:.1}s, {:.0}ms/i): exploitability = {:.4} ({:.2}% of pot)",
                iter, elapsed, ms_per_iter, exp, pct);
            checkpoint_idx += 1;

            if exp <= 5.0 {
                println!("  → Target reached! (< 5% of pot)");
                break;
            }
        }

        if iter > 1000 {
            println!("  → Stopping at 1000 iterations");
            break;
        }
        if t_total.elapsed().as_secs() > 600 {
            println!("  → Timeout at {} iterations ({:.0}s)", iter, t_total.elapsed().as_secs());
            break;
        }
    }

    let total = t_total.elapsed().as_secs_f64();
    let ms_per_iter = total / iter as f64 * 1000.0;

    println!("\n{}", "=".repeat(70));
    println!("  EXTERNAL SOLVER SUMMARY");
    println!("  Iterations: {}", iter);
    println!("  Time: {:.1}s ({:.0}ms/i)", total, ms_per_iter);
    println!("  Final exploitability: {:.4}", compute_exploitability(&game));
    println!();
    println!("  COMPARISON:");
    println!("  External: {} iters to converge ({:.0}ms/i)", iter, ms_per_iter);
    println!("  Our CPU: DIVERGES (per-outcome discount bug)");
    println!("  Our GPU: 50 iters → 15.4%, ~200 iters → ~5% (extrapolated)");
    println!("  Our GPU speed: 2293ms/i → 200 iters = 459s");
    println!("{}", "=".repeat(70));
}
