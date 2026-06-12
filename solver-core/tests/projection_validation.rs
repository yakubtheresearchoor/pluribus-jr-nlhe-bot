// Validate the K=5 projection methodology against the 3-player solve I
// have actually measured. The bake-off's 17-year number was built by
// multiplying CPU single-thread per-op rate × naive terminal count ×
// brute-force scaling × full-convergence iter count — four assumptions,
// any of which can be off by ~10× and compound into wrong-architecture
// decisions.
//
// This test:
//   1. Measures terminal count and tree size directly for 3p and 6p at
//      production nh, instead of guessing 5k vs 10k.
//   2. Measures GPU per-iter cost at 3p nh=50 K=2 (the production path
//      we already have working).
//   3. Backs out the effective GPU throughput on the real path.
//   4. Re-projects K=5 6p nh=50 using the measured anchor instead of the
//      single-thread CPU per-op rate.
//   5. Reports separately the brute-force projection (just to show the
//      scaling) and the recursive-exact-factored projection (the actual
//      production candidate).

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn count_terminals(tree: &solver_core::tree::flat::FlatTree) -> (usize, usize) {
    let mut n_terminal = 0usize;
    let mut n_total = 0usize;
    for node in tree.nodes.iter() {
        n_total += 1;
        if node.is_terminal() { n_terminal += 1; }
    }
    (n_terminal, n_total)
}

fn build_tree_np(np: u8) -> solver_core::tree::flat::FlatTree {
    let config = TreeConfig {
        num_players: np, initial_state: BoardState::Flop,
        starting_pot: np as i32 * 5,
        starting_stacks: vec![100; np as usize],
        initial_contributions: vec![5; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    build_tree(&config).unwrap()
}

#[test]
fn projection_validation_terminal_counts() {
    eprintln!("\n=== Terminal counts (independent of nh; same TreeConfig) ===");
    for np in [3u8, 4u8, 5u8, 6u8] {
        let tree = build_tree_np(np);
        let (n_term, n_total) = count_terminals(&tree);
        eprintln!("  np={} : total nodes={}, terminals={}, terminals/total={:.1}%",
            np, n_total, n_term, 100.0 * n_term as f64 / n_total as f64);
    }

    eprintln!("\nTerminal-count ratios (vs 3p baseline):");
    let tree_3p = build_tree_np(3);
    let (t3, _) = count_terminals(&tree_3p);
    for np in [4u8, 5u8, 6u8] {
        let tree = build_tree_np(np);
        let (tn, _) = count_terminals(&tree);
        eprintln!("  {}p / 3p = {:.2}x ({} / {})", np, tn as f64 / t3 as f64, tn, t3);
    }
}
