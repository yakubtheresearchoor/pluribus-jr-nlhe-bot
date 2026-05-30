/// T2F Production Verification: Our solver vs b1nary, side-by-side.
///
/// This is the architectural moment of truth. Both solvers run on the same
/// board (2h 7d Ks), same pot (10), same stacks (100), same bet sizes (pot).
/// Full hand ranges (1326 per player).
///
/// Run:
///   cargo test -p solver-core --features metal --test t2f_verification -- --test-threads=1 --nocapture --ignored

use solver_core::card::{card_from_str, Card};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn setup_our_solver() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![vec![1.0f32; 1326], vec![1.0; 1326]];

    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 10,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    };

    let tree = build_tree(&config).expect("tree build");
    let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    let game = FlopStartGame::new(table);
    (tree, game)
}

#[test]
#[ignore]
fn t2f_our_solver_50_iters() {
    let (tree, game) = setup_our_solver();
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  T2F Production Verification: Our Per-Outcome Solver       ║");
    println!("║  Board: 2h 7d Ks | Pot: 10 | Stack: 100 | PSB             ║");
    println!("║  Full ranges (1326 hands), DCFR                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let checkpoints = [0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50];
    let mut total_iters = 0u32;
    let mut results: Vec<(u32, f32, u64, u64)> = Vec::new(); // (iter, expl, solve_ms, measure_ms)

    for &target in &checkpoints {
        if target == 0 {
            // Just measure at iter 0 (uniform strategy)
            let t0 = std::time::Instant::now();
            let expl = solver.compute_exploitability(&tree, &game);
            let measure_ms = t0.elapsed().as_millis() as u64;
            results.push((0, expl, 0, measure_ms));
            println!("  iter {:>3}: expl = {:.6e}  (measure: {:.1}s)", 0, expl, measure_ms as f32 / 1000.0);
            continue;
        }

        let batch = target - total_iters;
        if batch > 0 {
            let t0 = std::time::Instant::now();
            let _cfv = solver.run(&tree, &game, batch);
            let solve_ms = t0.elapsed().as_millis() as u64;
            total_iters = target;

            let t1 = std::time::Instant::now();
            let expl = solver.compute_exploitability(&tree, &game);
            let measure_ms = t1.elapsed().as_millis() as u64;

            results.push((total_iters, expl, solve_ms, measure_ms));
            println!("  iter {:>3}: expl = {:.6e}  (solve: {:.1}s, measure: {:.1}s)",
                total_iters, expl, solve_ms as f32 / 1000.0, measure_ms as f32 / 1000.0);
        }
    }

    println!("\n  ══════════════════════════════════════════════════");
    println!("  Summary Table");
    println!("  ══════════════════════════════════════════════════");
    println!("  {:>6}  {:>14}  {:>10}  {:>5}", "iter", "expl", "solve(s)", "meas(s)");
    for (iter, expl, solve_ms, measure_ms) in &results {
        println!("  {:>6}  {:>14.6e}  {:>10.1}  {:>5.1}",
            iter, expl, *solve_ms as f32 / 1000.0, *measure_ms as f32 / 1000.0);
    }

    // Convergence check: final should be < initial
    let initial = results.first().unwrap().1;
    let final_expl = results.last().unwrap().1;
    assert!(final_expl < initial,
        "Solver diverged: initial={:.6e}, final={:.6e}", initial, final_expl);
    println!("\n  Convergence: {:.6e} → {:.6e} ({:.1}x reduction)", initial, final_expl, initial / final_expl);
}

#[test]
#[ignore]
fn t2f_our_solver_vanilla_50_iters() {
    let (tree, game) = setup_our_solver();
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());
    solver.set_vanilla_mode(true);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  T2F Production Verification: Vanilla CFR                   ║");
    println!("║  Board: 2h 7d Ks | Pot: 10 | Stack: 100 | PSB             ║");
    println!("║  Full ranges (1326 hands), alpha=beta=gamma=1               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let checkpoints = [0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50];
    let mut total_iters = 0u32;

    for &target in &checkpoints {
        if target == 0 {
            let t0 = std::time::Instant::now();
            let expl = solver.compute_exploitability(&tree, &game);
            println!("  iter {:>3}: expl = {:.6e}  (measure: {:.1}s)", 0, expl, t0.elapsed().as_secs_f32());
            continue;
        }

        let batch = target - total_iters;
        if batch > 0 {
            let t0 = std::time::Instant::now();
            let _cfv = solver.run(&tree, &game, batch);
            total_iters = target;

            let t1 = std::time::Instant::now();
            let expl = solver.compute_exploitability(&tree, &game);
            println!("  iter {:>3}: expl = {:.6e}  (solve: {:.1}s, measure: {:.1}s)",
                total_iters, expl, t0.elapsed().as_secs_f32(), t1.elapsed().as_secs_f32());
        }
    }
}

#[test]
#[ignore]
fn t2f_b1nary_50_iters() {
    use postflop_solver::{PostFlopGame, Range, CardConfig, TreeConfig as B1TreeConfig, ActionTree};
    use postflop_solver::{solve_step, compute_exploitability, flop_from_str, BetSize, BetSizeOptions as BBetSizeOptions};

    let one_pot = BBetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: vec![],
    };

    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("2h7dKs").unwrap(),
        ..Default::default()
    };

    let tree_config = B1TreeConfig {
        starting_pot: 10,
        effective_stack: 95,
        flop_bet_sizes: [one_pot.clone(), one_pot.clone()],
        turn_bet_sizes: [one_pot.clone(), one_pot.clone()],
        river_bet_sizes: [one_pot.clone(), one_pot.clone()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  T2F Production Verification: b1nary Reference              ║");
    println!("║  Board: 2h 7d Ks | Pot: 10 | Stack: 100 | PSB             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let checkpoints = [0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50];

    for &target in &checkpoints {
        if target == 0 {
            let t0 = std::time::Instant::now();
            let expl = compute_exploitability(&game);
            println!("  iter {:>3}: expl = {:.6e}  ({:.1}s)", 0, expl, t0.elapsed().as_secs_f32());
            continue;
        }
        let t0 = std::time::Instant::now();
        solve_step(&game, target as u32 - 1);
        let solve_s = t0.elapsed().as_secs_f32();

        let t1 = std::time::Instant::now();
        let expl = compute_exploitability(&game);
        let meas_s = t1.elapsed().as_secs_f32();
        println!("  iter {:>3}: expl = {:.6e}  (solve: {:.1}s, measure: {:.1}s)",
            target, expl, solve_s, meas_s);
    }
}
