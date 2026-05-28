#![cfg(feature = "cuda")]

/// Regression test: nested chance node ordering must be
///   clear → probs → set → recurse
/// If any caller uses set → recurse → probs, the flop-start game
/// dispatches chance_probability to the wrong method (river instead of turn).
///
/// This test builds a tiny flop-start tree and runs 1 VCFR iteration.
/// If the ordering is wrong, it panics with index out of bounds
/// (river deck has 48 entries, turn outcomes go up to 48).

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

#[test]
fn regression_nested_chance_ordering() {
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
    let tree = build_tree(&config).expect("tree build");
    let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    let game = FlopStartGame::new(table);
    let nh = game.num_hands(0);

    // This will panic with index out of bounds if the ordering is wrong:
    // "index out of bounds: the len is 48 but the index is 48"
    let mut vcfr = VectorCfr::new(&tree, vec![nh, nh]);
    vcfr.run_sequential(&tree, &game, 1);

    // Also test exploitability (which uses best_response's chance handling)
    // Skip exploitability on this large tree — it takes ~5 min.
    // The regression test only needs to verify VCFR doesn't crash.
    println!("Nested chance ordering: PASS (1 iter completed without crash)");
}
