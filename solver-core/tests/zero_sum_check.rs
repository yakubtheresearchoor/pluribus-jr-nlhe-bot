/// Check if strategy_value is zero-sum for both players at iter 0.
/// In a zero-sum game, sum_h w0[h]*SV0[h] + sum_h w1[h]*SV1[h] should be 0.
///
/// Run:
///   cargo test -p solver-core --features metal --test zero_sum_check -- --test-threads=1 --nocapture --ignored

use solver_core::card::{card_from_str, Card, index_to_card_pair};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn setup_our_game() -> (FlatTree, FlopStartGame) {
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
fn check_zero_sum_full() {
    let (tree, game) = setup_our_game();
    let solver = FlopStartVectorCfr::new(&tree, game.table());
    let nh = game.num_hands(0);
    let nc = game.table().num_combinations as f32;

    let sv0 = solver.strategy_value(&tree, &game, 0);
    let sv1 = solver.strategy_value(&tree, &game, 1);
    let br0 = solver.best_response_value(&tree, &game, 0);
    let br1 = solver.best_response_value(&tree, &game, 1);

    let w0 = game.initial_weight(0);
    let w1 = game.initial_weight(1);

    let ev0_sv: f64 = sv0.iter().zip(w0.iter()).map(|(&v, &w)| v as f64 * w as f64).sum();
    let ev1_sv: f64 = sv1.iter().zip(w1.iter()).map(|(&v, &w)| v as f64 * w as f64).sum();
    let ev0_br: f64 = br0.iter().zip(w0.iter()).map(|(&v, &w)| v as f64 * w as f64).sum();
    let ev1_br: f64 = br1.iter().zip(w1.iter()).map(|(&v, &w)| v as f64 * w as f64).sum();

    println!("\n  === Zero-Sum Check ===");
    println!("  EV[SV,P0] = {:.6}", ev0_sv);
    println!("  EV[SV,P1] = {:.6}", ev1_sv);
    println!("  EV[SV,P0] + EV[SV,P1] = {:.6}  (should be 0)", ev0_sv + ev1_sv);
    println!();
    println!("  EV[BR,P0] = {:.6}", ev0_br);
    println!("  EV[BR,P1] = {:.6}", ev1_br);
    println!("  EV[BR,P0] + EV[BR,P1] = {:.6}  (may not be 0 - BR is exploitative)", ev0_br + ev1_br);
    println!();
    println!("  Exploitability from EV:");
    println!("    (EV[BR,P0]-EV[SV,P0]) + (EV[BR,P1]-EV[SV,P1]) = {:.6}",
        (ev0_br - ev0_sv) + (ev1_br - ev1_sv));
    println!("    / 2 = {:.6}", ((ev0_br - ev0_sv) + (ev1_br - ev1_sv)) / 2.0);
    println!();
    println!("  Our compute_exploitability = {:.6}", solver.compute_exploitability(&tree, &game));

    // Check per-hand: SV0[h] + SV1[h] should relate to each other
    // SV0[h] is P0's CFV when holding hand h, summing over all P1 hands
    // SV1[h'] is P1's CFV when holding hand h', summing over all P0 hands
    // They can't be directly compared per-hand, but the total EV should zero-sum.

    // Let me check if the issue is in walk_sv or in evaluate_terminal
    // By checking a simple all-in terminal node
    println!("\n  === Check: P0 all-in at root (node 3, action 0) ===");
    // Node 3 is DEC p=0 nch=1 (fold/call?). Let me find a terminal node.
    for (i, n) in tree.nodes.iter().enumerate() {
        if n.is_terminal() {
            // Evaluate terminal from both players' perspective
            let mut cfreach_0: Vec<Vec<f32>> = vec![vec![1.0; nh], w0.to_vec()];
            let mut cfreach_1: Vec<Vec<f32>> = vec![w1.to_vec(), vec![1.0; nh]];

            // Set a specific turn and river card
            game.set_turn_card(game.table().remaining_deck[0]);
            game.set_river_card(game.table().river_decks[game.table().remaining_deck[0] as usize][0]);

            let cfv_p0 = game.evaluate_terminal(0, i, &tree, &cfreach_0);
            let cfv_p1 = game.evaluate_terminal(1, i, &tree, &cfreach_1);

            let sum_p0: f64 = cfv_p0.iter().map(|&v| v as f64).sum();
            let sum_p1: f64 = cfv_p1.iter().map(|&v| v as f64).sum();

            game.clear_chance_outcome();

            // Only print first few terminals
            if i < 20 || (sum_p0 + sum_p1).abs() > 0.01 {
                println!("  Terminal {}: sum_P0={:.4} sum_P1={:.4} total={:.4}",
                    i, sum_p0, sum_p1, sum_p0 + sum_p1);
            }
        }
    }
}
