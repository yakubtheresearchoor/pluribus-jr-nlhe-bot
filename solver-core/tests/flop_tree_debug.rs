#![cfg(feature = "cuda")]
use solver_core::card::card_from_str;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
fn flop_tree_debug() {
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop,
        starting_pot: 100, starting_stacks: vec![500, 500],
        initial_contributions: vec![0,0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).unwrap();
    
    for i in 0..tree.num_nodes() {
        let n = &tree.nodes[i];
        if n.is_chance() {
            println!("Chance node {}: board_state={}, children={}", i, n.board_state, n.num_children);
        }
    }
}
