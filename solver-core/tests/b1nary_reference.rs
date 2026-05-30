/// b1nary reference: run postflop-solver on a matching flop-start game.
/// This provides the baseline convergence numbers to compare against.
///
/// Run:
///   cargo test -p solver-core --features metal --test b1nary_reference -- --test-threads=1 --nocapture --ignored

use postflop_solver::{PostFlopGame, Range, CardConfig, TreeConfig, ActionTree, Game};
use postflop_solver::{solve_step, compute_exploitability};
use postflop_solver::{flop_from_str, Card, BetSize, BetSizeOptions};

fn make_b1nary_game() -> PostFlopGame {
    let one_pot = BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: vec![],
    };

    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("2h7dKs").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
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
    game
}

#[test]
fn b1nary_flop_start_5_iters() {
    let mut game = make_b1nary_game();

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  b1nary reference convergence (5 iters)         ║");
    println!("║  Board: 2h 7d Ks | Pot: 10 | Stack: 100        ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    let initial_expl = compute_exploitability(&game);
    println!("  iter   0: exploitability = {:.6e}", initial_expl);

    for t in 1..=5 {
        solve_step(&game, t as u32 - 1);
        let expl = compute_exploitability(&game);
        println!("  iter {:>3}: exploitability = {:.6e}", t, expl);
    }
}

#[test]
#[ignore]
fn b1nary_flop_start_20_iters() {
    let mut game = make_b1nary_game();

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  b1nary reference convergence (20 iters)        ║");
    println!("║  Board: 2h 7d Ks | Pot: 10 | Stack: 100        ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    let mut results: Vec<(u32, f32)> = Vec::new();
    let initial_expl = compute_exploitability(&game);
    results.push((0, initial_expl));
    println!("  iter   0: exploitability = {:.6e}", initial_expl);

    for t in 1..=20 {
        solve_step(&game, t as u32 - 1);
        let expl = compute_exploitability(&game);
        results.push((t, expl));
        if t <= 5 || t % 5 == 0 {
            println!("  iter {:>3}: exploitability = {:.6e}", t, expl);
        }
    }

    println!("\n  ══ Summary ══");
    for (iter, expl) in &results {
        println!("    iter {}: {:.6e}", iter, expl);
    }

    let first = results[0].1;
    let last = results.last().unwrap().1;
    println!("\n  Reduction: {:.2}x (final/initial)", last / first);
}
