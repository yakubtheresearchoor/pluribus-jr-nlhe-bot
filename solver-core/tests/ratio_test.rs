/// Compare exploitability between our solver and b1nary at multiple iterations.
/// If the ratio is constant, it's a scaling difference (not a bug).
/// If the ratio changes, there's a real bug.
///
/// Run:
///   cargo test -p solver-core --features metal --test ratio_test -- --test-threads=1 --nocapture --ignored

use solver_core::card::{card_from_str, Card};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn setup_our() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
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
    let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
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

#[test]
#[ignore]
fn ratio_across_iters() {
    use postflop_solver::{solve_step, compute_exploitability as b1_expl};

    let (tree, game) = setup_our();
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());
    let mut b1game = setup_b1nary();

    println!("\n  iter | ours     | b1nary   | ratio");
    println!("  -----+----------+----------+-------");

    for iter in 0..=3 {
        if iter > 0 {
            let _ = solver.run(&tree, &game, 1);
            solve_step(&b1game, iter as u32 - 1);
        }

        let our_expl = solver.compute_exploitability(&tree, &game);
        let b1_expl_val = b1_expl(&b1game);
        let ratio = our_expl / b1_expl_val;

        println!("  {:>4} | {:.6e} | {:.6e} | {:.4}", iter, our_expl, b1_expl_val, ratio);
    }
}
