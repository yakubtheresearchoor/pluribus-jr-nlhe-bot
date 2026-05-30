/// Architectural validation: our solver vs b1nary on the EXACT SAME game.
///
/// We manually construct a 9-node river tree, then run both solvers and
/// compare exploitability at the same iteration counts.
///
/// Game: River poker, board=2h7dKs3c5c, pot=10, eff_stack=95, 1PSB, no raises.
/// Tree: 9 nodes (check/bet/call/fold only).
///
/// Run:
///   cargo test -p solver-core --features metal --test same_game -- --test-threads=1 --nocapture --ignored

use solver_core::card::{card_from_str, Card};
use solver_core::solver::best_response::{self, exploitability, StrategyProfile};
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::flat::{FlatNode, FlatTree};
use solver_core::tree::action::BoardState;

fn uniform_range() -> Vec<f32> { vec![1.0; 1326] }

/// Build 9-node river tree:
/// [0] P0 → check[1], bet[2]
/// [1] P1 → check[3], bet[4]
/// [2] P1 → call[5], fold[6]
/// [3] TERMINAL (check-check showdown)
/// [4] P0 → call[7], fold[8]
/// [5] TERMINAL (bet-call showdown)
/// [6] TERMINAL (P1 folds)
/// [7] TERMINAL (check-bet-call showdown)
/// [8] TERMINAL (P0 folds)
fn build_9node_river_tree() -> FlatTree {
    let mut tree = FlatTree::new(2, 10, vec![95, 95], 0.0, 0.0);

    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n0, 0, 5);
    tree.set_contribution(n0, 1, 5);

    let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n1, 0, 5);
    tree.set_contribution(n1, 1, 5);

    let n2 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n2, 0, 15); // P0 bet 10
    tree.set_contribution(n2, 1, 5);

    let n3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n3, 0, 5);
    tree.set_contribution(n3, 1, 5);

    let n4 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n4, 0, 5);
    tree.set_contribution(n4, 1, 15); // P1 bet 10

    let n5 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n5, 0, 15);
    tree.set_contribution(n5, 1, 15);

    let n6 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n6, 0, 15);
    tree.set_contribution(n6, 1, 5);
    tree.set_folded_mask(n6, 0b10); // P1 folded

    let n7 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n7, 0, 15);
    tree.set_contribution(n7, 1, 15);

    let n8 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n8, 0, 5);
    tree.set_contribution(n8, 1, 15);
    tree.set_folded_mask(n8, 0b01); // P0 folded

    tree.set_children(n0, vec![n1 as u32, n2 as u32]);
    tree.set_children(n1, vec![n3 as u32, n4 as u32]);
    tree.set_children(n2, vec![n5 as u32, n6 as u32]);
    tree.set_children(n4, vec![n7 as u32, n8 as u32]);

    tree.compute_levels();
    tree
}

#[test]
#[ignore]
fn validate_architecture() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "3c", "5c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_hands(0);
    let nc = game.num_combinations() as f32;
    let tree = build_9node_river_tree();

    assert_eq!(tree.num_nodes(), 9);

    // Our solver
    let mut solver = VectorCfr::new(&tree, vec![nh, nh]);

    // b1nary solver - recreate for each checkpoint since solve() finalizes
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
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
        ..Default::default()
    };
    let action_tree = ActionTree::new(tree_config).unwrap();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  ARCHITECTURAL VALIDATION: Our Solver vs b1nary            ║");
    println!("║  Board: 2h 7d Ks 3c 5c | Pot: 10 | Stack: 95 | nh: {:<5}  ║", nh);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  Tree nodes (ours): {}", tree.num_nodes());
    println!("  num_combinations: {:.0}", nc);

    println!("\n  {:>6} {:>14} {:>14} {:>10}", "Iters", "Ours", "b1nary", "Ratio");

    let checkpoints: &[u32] = &[1, 5, 10, 25, 50, 100, 200, 500, 1000];
    let mut our_total = 0usize;
    let mut b1_total = 0u32;

    for &n_iters in checkpoints {
        // Our solver
        let remaining = n_iters as usize - our_total;
        solver.run_sequential(&tree, &game as &dyn GameSpec, remaining as u32);
        our_total = n_iters as usize;

        let profile = StrategyProfile::from_usize_offsets(
            solver.cum_strategy_slice(), solver.node_offsets(), nh,
        );
        let our_expl = exploitability(&tree, &game as &dyn GameSpec, &profile);

        // b1nary: recreate game fresh and solve for exactly n_iters
        let action_tree2 = ActionTree::new(TreeConfig {
            starting_pot: 10, effective_stack: 95,
            initial_state: postflop_solver::BoardState::River,
            river_bet_sizes: [one_pot.clone(), one_pot.clone()],
            add_allin_threshold: 0.0, force_allin_threshold: 0.0, merging_threshold: 0.0,
            ..Default::default()
        }).unwrap();
        let mut b1game = PostFlopGame::with_config(
            CardConfig {
                range: [Range::ones(); 2],
                flop: flop_from_str("2h7dKs").unwrap(),
                turn: card_from_str("3c").unwrap(),
                river: card_from_str("5c").unwrap(),
                ..Default::default()
            },
            action_tree2,
        ).unwrap();
        b1game.allocate_memory(false);
        let b1_expl = if n_iters > 0 {
            solve(&mut b1game, n_iters, 0.0, false)
        } else {
            compute_exploitability(&b1game)
        };

        let ratio = if b1_expl > 1e-10 { our_expl / b1_expl } else { f32::NAN };

        println!("  {:>6} {:>14.6} {:>14.6} {:>10.2}x",
            n_iters, our_expl, b1_expl, ratio);
    }

    // Zero-sum check
    let profile = StrategyProfile::from_usize_offsets(
        solver.cum_strategy_slice(), solver.node_offsets(), nh,
    );
    let sv0 = best_response::strategy_value(&tree, &game as &dyn GameSpec, &profile, 0);
    let sv1 = best_response::strategy_value(&tree, &game as &dyn GameSpec, &profile, 1);
    let w0 = game.initial_weight(0);
    let ev_sum: f32 = (0..nh).map(|h| w0[h] * (sv0[h] + sv1[h])).sum::<f32>() / nc;
    println!("\n  Zero-sum (EV[SV0] + EV[SV1]): {:.2e}", ev_sum);
}
