// Final follow-up: do different bet_sizes produce different terminal
// contribution distributions, even if node count happens to match?
// This is the only remaining test to confirm bet_sizes is fully applied.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn summary(label: &str, cfg: TreeConfig) {
    let tree = build_tree(&cfg).unwrap();
    let np = 6usize;
    let mut patterns: std::collections::HashSet<Vec<i32>> = std::collections::HashSet::new();
    let mut distinct_max_contribs: std::collections::HashSet<i32> = std::collections::HashSet::new();
    let mut player_amounts_seen: Vec<std::collections::HashSet<i32>> = vec![std::collections::HashSet::new(); np];
    for (idx, node) in tree.nodes.iter().enumerate() {
        if !node.is_terminal() { continue; }
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        for (p, &c) in contribs.iter().enumerate() {
            player_amounts_seen[p].insert(c);
        }
        let max_c = *contribs.iter().max().unwrap();
        distinct_max_contribs.insert(max_c);
        patterns.insert(contribs);
    }
    eprintln!();
    eprintln!("CONFIG: {}", label);
    eprintln!("  total nodes: {}", tree.nodes.len());
    eprintln!("  distinct terminal contribution patterns: {}", patterns.len());
    eprintln!("  distinct max-contribution values across terminals: {}", distinct_max_contribs.len());
    let mut sorted: Vec<i32> = distinct_max_contribs.iter().copied().collect();
    sorted.sort();
    eprintln!("  distinct max values: {:?}",
        if sorted.len() <= 20 { format!("{:?}", sorted) } else { format!("{:?}...{:?}", &sorted[..5], &sorted[sorted.len()-5..]) });
    for p in 0..np {
        let mut amts: Vec<i32> = player_amounts_seen[p].iter().copied().collect();
        amts.sort();
        eprintln!("  player {} distinct contributions ({}): {}",
            p, amts.len(),
            if amts.len() <= 15 { format!("{:?}", amts) } else { format!("{:?}...{:?}", &amts[..5], &amts[amts.len()-5..]) });
    }
}

#[test]
fn betsizes_produce_different_terminal_distributions() {
    let base = |bet: Vec<BetSize>, raise: Vec<BetSize>| TreeConfig {
        num_players: 6, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet, raise },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };

    eprintln!("\n=== Does bet_sizes produce DIFFERENT terminal contribution distributions? ===");
    summary("1 bet PotRel(1.0)", base(vec![BetSize::PotRelative(1.0)], vec![]));
    summary("1 bet PotRel(0.5)", base(vec![BetSize::PotRelative(0.5)], vec![]));
    summary("2 bets {0.5, 1.0}", base(vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)], vec![]));
    summary("2 bets {0.5, 2.0}", base(vec![BetSize::PotRelative(0.5), BetSize::PotRelative(2.0)], vec![]));
    summary("1 bet + 1 raise (both 1x)", base(vec![BetSize::PotRelative(1.0)], vec![BetSize::PotRelative(1.0)]));
}
