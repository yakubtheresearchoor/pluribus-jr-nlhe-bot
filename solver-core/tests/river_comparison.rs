/// Compare exploitability on a RIVER-start game (no chance nodes).
/// If the ratio is ~1.0, the issue is in chance node handling.
/// If the ratio is >>1.0, the issue is in the showdown/terminal evaluation.
///
/// Run:
///   cargo test -p solver-core --features metal --test river_comparison -- --test-threads=1 --nocapture --ignored

use solver_core::card::{card_from_str, Card};
use solver_core::solver::best_response::{self, exploitability, StrategyProfile};
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::tree::flat::FlatTree;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn uniform_range() -> Vec<f32> { vec![1.0; 1326] }

fn build_river_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::River,
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
    button_player: None,
            max_bets_per_street: None,

    };
    build_tree(&config).unwrap()
}

fn setup_b1nary_river() -> postflop_solver::PostFlopGame {
    use postflop_solver::*;
    let one_pot = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("2h7dKs").unwrap(),
        turn: card_from_str("3c").unwrap(),
        river: card_from_str("5c").unwrap(),
        ..Default::default()
    };
    let tree_config = TreeConfig {
        starting_pot: 10,
        effective_stack: 95,
        initial_state: postflop_solver::BoardState::River,
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
fn river_start_comparison() {
    // Board: 2h 7d Ks 3c 5c (same as b1nary)
    let board: Vec<Card> = ["2h", "7d", "Ks", "3c", "5c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_hands(0);
    let tree = build_river_tree();

    let solver = VectorCfr::new(&tree, vec![nh, nh]);
    let b1game = setup_b1nary_river();

    use postflop_solver::{compute_exploitability as b1_expl};

    // Our exploitability at iter 0 (uniform strategy)
    let profile = StrategyProfile::from_usize_offsets(
        solver.cum_strategy_slice(), solver.node_offsets(), nh,
    );
    let our_expl = exploitability(&tree, &game as &dyn GameSpec, &profile);
    let b1_expl_val = b1_expl(&b1game);

    // Zero-sum check
    let sv0 = best_response::strategy_value(&tree, &game as &dyn GameSpec, &profile, 0);
    let sv1 = best_response::strategy_value(&tree, &game as &dyn GameSpec, &profile, 1);
    let nc = game.num_combinations() as f32;
    let w0 = game.initial_weight(0);
    let ev0: f32 = (0..nh).map(|h| w0[h] * sv0[h]).sum::<f32>() / nc;
    let ev1: f32 = (0..nh).map(|h| w0[h] * sv1[h]).sum::<f32>() / nc;

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  RIVER-START comparison (no chance nodes)       ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("  Board: 2h 7d Ks 3c 5c, nh={}, nc={:.0}", nh, nc);
    println!("  Our exploitability:  {:.6e}", our_expl);
    println!("  b1nary exploitability: {:.6e}", b1_expl_val);
    println!("  Ratio: {:.4}x", our_expl / b1_expl_val);
    println!("  Zero-sum: SV_sum = {:.6e}", ev0 + ev1);

    let ratio = our_expl / b1_expl_val;
    if ratio < 1.5 {
        println!("  ✅ Ratio < 1.5x — terminal evaluation matches");
    } else {
        println!("  ❌ Ratio {:.2}x — terminal evaluation differs", ratio);
    }
}
