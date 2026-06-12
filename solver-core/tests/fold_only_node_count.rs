// Task #26 closure: re-count FOLD-only PLAYER nodes on the CORRECTED tree.
//
// History: the pre-rewrite builder produced ~138k PLAYER nodes whose only
// child was a FOLD action. That is strictly illegal — a player facing a bet
// always has at least {FOLD, CALL}, never just {FOLD}, because CALL (or
// short-call-all-in) is always available when you owe chips. The 138k count
// was a known correctness bug deferred during the rewrite.
//
// The rewritten builder now passes the standing gate at 0 violations across
// 8 configs, including the gate's `facing_bet_missing_call` category which
// flags any PLAYER node whose action set has FOLD but no CALL. So FOLD-only
// nodes are caught implicitly by the gate. This test makes that explicit
// per the discipline learned in the rewrite arc: "likely fixed" becomes
// "measured at zero" before being closed.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::ACTION_LABEL_FOLD;

fn count_fold_only(cfg: &TreeConfig) -> usize {
    let tree = build_tree(cfg).unwrap();
    let mut count = 0;
    for n in &tree.nodes {
        if n.is_player() && n.num_children == 1 {
            let child_idx = tree.children[n.children_start as usize] as usize;
            if tree.nodes[child_idx].action_label == ACTION_LABEL_FOLD {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn fold_only_nodes_zero_across_all_gate_configs() {
    // Same 8 configs as the standing gate.
    let configs: Vec<(&str, TreeConfig)> = vec![
        ("6p asymmetric [10,5,5,5,5,5]", TreeConfig {
            num_players: 6, initial_state: BoardState::Flop, starting_pot: 35,
            starting_stacks: vec![200; 6],
            initial_contributions: vec![10, 5, 5, 5, 5, 5],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
            add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,

        }),
        ("6p symmetric [5;6]", TreeConfig {
            num_players: 6, initial_state: BoardState::Flop, starting_pot: 30,
            starting_stacks: vec![200; 6],
            initial_contributions: vec![5; 6],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
            add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,

        }),
        ("3p asymmetric [10,5,5]", TreeConfig {
            num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
            starting_stacks: vec![200; 3],
            initial_contributions: vec![10, 5, 5],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
            add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,

        }),
        ("2p HU symmetric [5,5]", TreeConfig {
            num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
            starting_stacks: vec![100; 2],
            initial_contributions: vec![5, 5],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
            add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,

        }),
        ("2p HU asymmetric [2,1]", TreeConfig {
            num_players: 2, initial_state: BoardState::Flop, starting_pot: 3,
            starting_stacks: vec![100; 2],
            initial_contributions: vec![2, 1],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
            add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,

        }),
        ("6p asym allin_thr=0.5", TreeConfig {
            num_players: 6, initial_state: BoardState::Flop, starting_pot: 35,
            starting_stacks: vec![200; 6],
            initial_contributions: vec![10, 5, 5, 5, 5, 5],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
            add_allin_threshold: 0.5, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,

        }),
        ("6p asym starting at TURN", TreeConfig {
            num_players: 6, initial_state: BoardState::Turn, starting_pot: 35,
            starting_stacks: vec![200; 6],
            initial_contributions: vec![10, 5, 5, 5, 5, 5],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
            add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,

        }),
        ("6p asym starting at RIVER", TreeConfig {
            num_players: 6, initial_state: BoardState::River, starting_pot: 35,
            starting_stacks: vec![200; 6],
            initial_contributions: vec![10, 5, 5, 5, 5, 5],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
            add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,

        }),
    ];

    let mut total = 0usize;
    eprintln!("\n=== FOLD-only PLAYER node count (#26 closure) ===");
    for (label, cfg) in &configs {
        let c = count_fold_only(cfg);
        eprintln!("  {} → {} FOLD-only nodes", label, c);
        total += c;
    }
    eprintln!("Total: {}", total);
    eprintln!(
        "Pre-rewrite count was ~138,000. Corrected-tree count must be 0 \
         (a player facing a bet always has FOLD + CALL minimum)."
    );
    assert_eq!(
        total, 0,
        "Task #26 regression: {} FOLD-only PLAYER nodes found on corrected \
         tree (was ~138k before rewrite, must be 0 after). Indicates the \
         FacingBet classifier dropped the CALL action somewhere.",
        total
    );
}
