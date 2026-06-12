/// CORRECTED Full-game comparison: our solver vs b1nary on the SAME game.
/// Both use 2h7dKs, pot=10, stacks=100, 1PSB, 1176 hands.
///
/// This is the actual architectural validation.
///
/// Run:
///   cargo test -p solver-core --features metal --test full_game_corrected -- --test-threads=1 --nocapture --ignored

use solver_core::card::card_from_str;
use solver_core::solver::flop_start_game::FlopStartGame;
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn setup_our() -> (solver_core::tree::flat::FlatTree, FlopStartGame) {
    let board: Vec<solver_core::card::Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![vec![1.0f32; 1326], vec![1.0; 1326]];
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).unwrap();
    let table = solver_core::solver::flop_start_game::FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    (tree, FlopStartGame::new(table))
}

fn setup_b1nary() -> postflop_solver::PostFlopGame {
    use postflop_solver::*;
    let one_pot = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("2h7dKs").unwrap(),
        ..Default::default()
    };
    let tree_config = TreeConfig {
        starting_pot: 10, effective_stack: 95,
        flop_bet_sizes: [one_pot.clone(), one_pot.clone()],
        turn_bet_sizes: [one_pot.clone(), one_pot.clone()],
        river_bet_sizes: [one_pot.clone(), one_pot.clone()],
        ..Default::default()
};
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);
    game
}

/// GATE: Zero-sum on the FULL game at iter 0.
#[test]
#[ignore]
fn gate_zero_sum_full_game() {
    let (tree, game) = setup_our();
    let solver = FlopStartVectorCfr::new(&tree, game.table());

    let nh = solver.num_hands();
    let nc = game.table().num_combinations as f32;
    let w0 = &game.table().initial_weights[0];

    let sv0 = solver.strategy_value(&tree, &game, 0);
    let sv1 = solver.strategy_value(&tree, &game, 1);

    let ev_sv0: f32 = (0..nh).map(|h| w0[h] * sv0[h]).sum::<f32>() / nc;
    let ev_sv1: f32 = (0..nh).map(|h| w0[h] * sv1[h]).sum::<f32>() / nc;
    let sv_sum = ev_sv0 + ev_sv1;

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  GATE: Zero-sum on FULL game at iter 0          ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("  nh={}, nc={:.0}", nh, nc);
    println!("  EV[SV,P0] = {:.6}", ev_sv0);
    println!("  EV[SV,P1] = {:.6}", ev_sv1);
    println!("  SV_sum    = {:.6e}", sv_sum);

    let pass = sv_sum.abs() < 0.1;
    if pass {
        println!("  ✅ PASS: |SV_sum| < 0.1");
    } else {
        println!("  ❌ FAIL: |SV_sum| = {:.6e} >= 0.1", sv_sum.abs());
    }
    assert!(pass, "Zero-sum violated on full game: {}", sv_sum);
}

/// ARCHITECTURAL VALIDATION: Full game comparison at iter 0.
#[test]
#[ignore]
fn full_game_iter0_comparison() {
    let (tree, game) = setup_our();
    let solver = FlopStartVectorCfr::new(&tree, game.table());
    let mut b1game = setup_b1nary();

    use postflop_solver::{compute_exploitability as b1_expl};

    let our_expl = solver.compute_exploitability(&tree, &game);
    let b1_expl_val = b1_expl(&b1game);

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  FULL GAME iter 0 comparison                    ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("  Our exploitability:  {:.6e}", our_expl);
    println!("  b1nary exploitability: {:.6e}", b1_expl_val);
    println!("  Ratio: {:.2}x", our_expl / b1_expl_val);

    let nh = solver.num_hands();
    let nc = game.table().num_combinations as f32;
    let w0 = &game.table().initial_weights[0];
    let sv0 = solver.strategy_value(&tree, &game, 0);
    let sv1 = solver.strategy_value(&tree, &game, 1);
    let ev_sv0: f32 = (0..nh).map(|h| w0[h] * sv0[h]).sum::<f32>() / nc;
    let ev_sv1: f32 = (0..nh).map(|h| w0[h] * sv1[h]).sum::<f32>() / nc;
    println!("  Zero-sum check: SV_sum = {:.6e}", ev_sv0 + ev_sv1);

    let ratio = our_expl / b1_expl_val;
    if ratio < 1.5 {
        println!("  ✅ Ratio < 1.5x — same scale");
    } else if ratio < 3.0 {
        println!("  ⚠️  Ratio {:.2}x — different scale, investigating needed", ratio);
    } else {
        println!("  ❌ Ratio {:.2}x — significant discrepancy", ratio);
    }
}

/// Full convergence run: our solver at iters 1, 5, 10, 25, 50.
/// Zero-sum checked at every point.
#[test]
#[ignore]
fn full_game_convergence_with_gates() {
    let (tree, game) = setup_our();
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    let nh = solver.num_hands();
    let nc = game.table().num_combinations as f32;
    let w0 = &game.table().initial_weights[0];

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  Full game convergence with zero-sum gates      ║");
    println!("╚══════════════════════════════════════════════════╝");

    let check_iters = [0, 1, 5, 10, 25, 50];
    let mut iter_count = 0;

    println!("  {:>4} | {:>12} | {:>12} | {}", "iter", "expl", "SV_sum", "status");
    println!("  -----+--------------+--------------+-------");

    for &target_iter in &check_iters {
        while iter_count < target_iter {
            let _ = solver.run(&tree, &game, 1);
            iter_count += 1;
        }

        let expl = solver.compute_exploitability(&tree, &game);
        let sv0 = solver.strategy_value(&tree, &game, 0);
        let sv1 = solver.strategy_value(&tree, &game, 1);
        let ev0: f32 = (0..nh).map(|h| w0[h] * sv0[h]).sum::<f32>() / nc;
        let ev1: f32 = (0..nh).map(|h| w0[h] * sv1[h]).sum::<f32>() / nc;
        let zs = (ev0 + ev1).abs();
        let status = if zs < 0.1 { "✅" } else { "❌ BUG" };

        println!("  {:>4} | {:.6e} | {:.6e} | {}", target_iter, expl, ev0 + ev1, status);
    }
}
