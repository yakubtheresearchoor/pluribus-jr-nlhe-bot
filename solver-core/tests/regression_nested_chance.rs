#![cfg(feature = "cuda")]

/// Regression test for nested chance ordering: clear → probs → set → recurse.
/// Uses a wrapper that limits outcomes to 3 per level (instead of 49/48),
/// so the test runs in seconds, not minutes.
/// The ordering bug doesn't depend on tree size or outcome count.

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

/// Wrapper that caps chance outcomes at N per level.
/// Delegates everything else to the inner game.
struct LimitedOutcomes<'a> {
    inner: &'a FlopStartGame,
    max_turn: usize,
    max_river: usize,
}

impl<'a> GameSpec for LimitedOutcomes<'a> {
    fn num_hands(&self, player: u8) -> usize { self.inner.num_hands(player) }
    fn initial_weight(&self, player: u8) -> Vec<f32> { self.inner.initial_weight(player) }

    fn evaluate_terminal(
        &self, traverser: u8, node_idx: usize, tree: &FlatTree,
        cfreach: &[Vec<f32>],
    ) -> Vec<f32> {
        self.inner.evaluate_terminal(traverser, node_idx, tree, cfreach)
    }

    fn chance_probability(&self, outcome: usize, hand: usize) -> f32 {
        self.inner.chance_probability(outcome, hand)
    }

    fn num_chance_outcomes(&self) -> usize {
        let full = self.inner.num_chance_outcomes();
        // If inner has a turn card set (river level), cap at max_river; else max_turn
        // Heuristic: if full > 40 it's the turn deck (49), else river (48)
        if full > 40 {
            self.max_turn.min(full)
        } else {
            self.max_river.min(full)
        }
    }

    fn set_chance_outcome(&self, outcome: usize) {
        self.inner.set_chance_outcome(outcome)
    }
    fn clear_chance_outcome(&self) {
        self.inner.clear_chance_outcome()
    }
    fn num_combinations(&self) -> f64 { self.inner.num_combinations() }
}

#[test]
fn regression_nested_chance_ordering_fast() {
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

    // Wrap with limited outcomes: 3 turn × 3 river = 9 total (instead of 2352)
    let limited = LimitedOutcomes { inner: &game, max_turn: 3, max_river: 3 };

    let mut vcfr = VectorCfr::new(&tree, vec![nh, nh]);
    // This will panic with "index out of bounds: len is 48, index is 48"
    // if the clear→probs→set ordering is broken
    vcfr.run_sequential(&tree, &limited, 1);

    println!("Nested chance ordering: PASS (3×3 outcomes, completed without crash)");
}
