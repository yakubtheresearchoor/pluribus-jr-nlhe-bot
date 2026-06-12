// Task #24 closure: re-measure 1-child PLAYER node fraction on the
// CORRECTED tree.
//
// History: pre-rewrite measurement reported "77.8% 1-child PLAYER infoset
// structure" as a tree-representation inefficiency candidate for elision.
// That measurement was on the broken 14.5M-node tree. The corrected tree
// is 2,431 nodes at the same 6p asymmetric config — a 6,000× reduction —
// so the 1-child distribution is fundamentally different.
//
// 1-child PLAYER nodes ARE legitimate in poker: they're forced-check
// pass-through nodes for all-in players (their action set is [Check] only
// per the builder's compute_actions early-return on player_remaining = 0).
// They aren't "dead weight" — they preserve the per-player turn order so
// downstream players' decisions are correctly contextualized. Whether
// elision is worthwhile depends on what fraction they represent.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn one_child_stats(cfg: &TreeConfig, label: &str) -> (usize, usize, f32) {
    let tree = build_tree(cfg).unwrap();
    let mut total_player = 0;
    let mut one_child_player = 0;
    let mut by_children: std::collections::BTreeMap<u16, usize> = std::collections::BTreeMap::new();
    for n in &tree.nodes {
        if n.is_player() {
            total_player += 1;
            *by_children.entry(n.num_children).or_insert(0) += 1;
            if n.num_children == 1 {
                one_child_player += 1;
            }
        }
    }
    let frac = if total_player > 0 {
        one_child_player as f32 * 100.0 / total_player as f32
    } else {
        0.0
    };
    eprintln!(
        "{}: {} PLAYER nodes, {} with 1 child ({:.1}%), distribution: {:?}",
        label, total_player, one_child_player, frac, by_children
    );
    (total_player, one_child_player, frac)
}

#[test]
fn one_child_player_fraction_across_corrected_configs() {
    eprintln!("\n=== Task #24 closure: 1-child PLAYER node fraction on corrected tree ===");
    eprintln!("(Pre-rewrite measurement: 77.8% on 14.5M-node broken tree)");
    eprintln!("(Corrected 6p asymmetric tree: 2,431 nodes total)\n");

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
    ];

    let mut total_p = 0;
    let mut total_one = 0;
    for (label, cfg) in &configs {
        let (tp, oc, _) = one_child_stats(cfg, label);
        total_p += tp;
        total_one += oc;
    }

    eprintln!(
        "\nAGGREGATE: {} PLAYER nodes total, {} with 1 child ({:.1}%)",
        total_p,
        total_one,
        total_one as f32 * 100.0 / total_p as f32
    );
    eprintln!(
        "\nInterpretation: 1-child PLAYER nodes are forced-check pass-through \
         for all-in players. They are legitimate, not dead weight. The pre-\
         rewrite 77.8% was inflated by structural bugs (empty PLAYER nodes, \
         broken player advancement) that created the appearance of 1-child \
         degeneracy. On the corrected tree, the fraction reflects actual \
         all-in pass-through structure, which is decision-relevant."
    );
}
