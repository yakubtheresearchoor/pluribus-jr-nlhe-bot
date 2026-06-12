// Phase 0.B detail: count tree nodes by type to identify whether the
// 4.3M total is dominated by chance enumeration, terminal enumeration,
// or player decisions. Determines whether action abstraction is a
// useful Phase 1 lever (only if decisions are a non-trivial fraction).

use solver_core::card::card_from_str;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn count_by_type(label: &str, config: TreeConfig) {
    let tree = build_tree(&config).unwrap();
    let total = tree.nodes.len();
    let player = tree.nodes.iter().filter(|n| n.is_player()).count();
    let chance = tree.nodes.iter().filter(|n| n.is_chance()).count();
    let terminal = tree.nodes.iter().filter(|n| n.is_terminal()).count();
    let pct = |n: usize| n as f64 / total as f64 * 100.0;
    eprintln!("{:>30} | total={:>9} | players={:>8} ({:5.1}%) | chance={:>5} ({:4.1}%) | terminal={:>9} ({:5.1}%)",
        label, total, player, pct(player), chance, pct(chance), terminal, pct(terminal));
}

#[test]
fn tree_node_breakdown_phase0() {
    eprintln!("\n=== Phase 0.B detail: tree node breakdown by type ===\n");

    // Baseline used in all measurements
    let _board: Vec<u8> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let base = || TreeConfig {
        num_players: 6, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };

    count_by_type("baseline (1 bet, no raise)", base());

    // Variants for comparison
    let mut c2 = base(); c2.bet_sizes.bet = vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)];
    count_by_type("2 bets {0.5x,1x}, no raise", c2);

    let mut c3 = base(); c3.bet_sizes.bet = vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)];
    c3.bet_sizes.raise = vec![BetSize::PotRelative(1.0)];
    count_by_type("2 bets, 1 raise", c3);

    // Pluribus-like later-street abstraction: {0.5x, 1x, allin via threshold}
    let mut c_p = base();
    c_p.bet_sizes.bet = vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)];
    c_p.add_allin_threshold = 2.0;  // force allin
    count_by_type("Pluribus-like {0.5x,1x,allin}", c_p);

    // 3-player for scaling reference
    let mut c_3p = base();
    c_3p.num_players = 3;
    c_3p.starting_stacks = vec![200; 3];
    c_3p.initial_contributions = vec![10, 5, 5];
    count_by_type("3-player baseline", c_3p);

    // 6p with smaller stacks to constrain depth
    let mut c_small = base();
    c_small.starting_stacks = vec![50; 6];  // less room for many actions
    count_by_type("6p stack=50 (constrained)", c_small);

    eprintln!();
    eprintln!("Key questions answered:");
    eprintln!("  - What fraction of 4.3M is decisions vs chance vs terminals?");
    eprintln!("  - How does node count scale with bet sizes added?");
    eprintln!("  - Would Pluribus-like abstraction grow or shrink the tree?");
}
