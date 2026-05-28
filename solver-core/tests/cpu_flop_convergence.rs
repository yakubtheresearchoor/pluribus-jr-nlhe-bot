#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
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

/// Measure CPU VCFR convergence on the flop-start tree.
/// The key question: does CPU reach acceptable exploitability in ~10 iterations?
/// If yes, fixing GPU convergence to match makes the solver viable at current speed.
/// If no, both convergence AND throughput need fixing.
#[test]
fn cpu_flop_convergence() {
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
    let starting_pot = tree.starting_pot as f64;
    let offsets = make_offsets(&tree, nh);
    let game = FlopStartGame::new(FlopChanceTable::compute_flop_start(&board, &ranges, 2));

    println!("\n{}", "=".repeat(70));
    println!("  CPU VCFR CONVERGENCE ON FLOP-START (2h7dKs, pot=100)");
    println!("{}", "=".repeat(70));
    println!("Tree: {} nodes, nh={}", tree.num_nodes(), nh);
    println!("Target: 3-5% of pot in ~10 iterations");
    println!();

    let num_hands_vec = (0..tree.num_players).map(|_| nh).collect();
    let mut solver = VectorCfr::new(&tree, num_hands_vec);

    // DCFR params at key iterations (for reference)
    for t in 0..5 {
        let p = solver_core::gpu::context::DcfrParams::new(t);
        println!("  DCFR iter {}: alpha={:.4} beta={:.4} gamma={:.4}",
            t, p.alpha_t, p.beta_t, p.gamma_t);
    }
    println!();

    println!("{:<8} {:<12} {:<10} {:<12} {:<10}", "Iters", "Exploit", "% pot", "Solve(s)", "BR(s)");
    println!("{}", "-".repeat(55));

    let checkpoints: &[u32] = &[1, 2, 5, 10];
    let mut accumulated = 0u32;

    for &target in checkpoints {
        let delta = target - accumulated;
        let t_solve = std::time::Instant::now();
        solver.run_sequential(&tree, &game, delta);
        let solve_time = t_solve.elapsed().as_secs_f64();
        accumulated = target;

        let t_br = std::time::Instant::now();
        let cum = solver.cum_strategy_slice();
        let profile = StrategyProfile::from_usize_offsets(cum, &offsets, nh);
        let exp = exploitability(&tree, &game, &profile);
        let br_time = t_br.elapsed().as_secs_f64();
        let pct = exp as f64 / starting_pot * 100.0;

        println!("{:<8} {:<12.4} {:<10.2}% {:<12.1} {:<10.1}",
            target, exp, pct, solve_time, br_time);
    }

    println!("\n{}", "=".repeat(70));
    println!("  GPU comparison (from earlier measurement):");
    println!("  GPU @  1 iter: 167.66% (same as CPU — no discount at iter 0,1)");
    println!("  GPU @ 10 iter: 111.32%");
    println!("  GPU @ 25 iter:  27.75%");
    println!("  GPU @ 50 iter:  15.45%");
    println!();
    println!("  If CPU @ 10 iter < 30% pot:");
    println!("    → Fixing GPU convergence to match CPU makes solver VIABLE");
    println!("    → 10 iters × 2293ms = 23s, within 25s budget");
    println!("  If CPU @ 10 iter > 100% pot:");
    println!("    → Need both convergence AND throughput improvement");
    println!("    → Streaming reduction + isomorphism + compression");
    println!("{}", "=".repeat(70));
}
