// Follow-up: even though FlatNode.amount=0 everywhere, the per-player
// contributions at terminals tell us whether bets actually happened.
// If contributions at terminals VARY (different terminals have different
// per-player contributions), then betting IS happening through some other
// mechanism. If contributions are uniform at all terminals (just the
// starting [10,5,5,5,5,5]), the tree is degenerate.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
fn check_terminal_contributions_for_betting_signal() {
    let cfg = TreeConfig {
        num_players: 6, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&cfg).unwrap();

    let np = 6usize;
    // Gather distinct per-player contribution patterns at terminals
    let mut patterns: std::collections::HashMap<Vec<i32>, usize> = std::collections::HashMap::new();
    let mut total_terminals = 0usize;
    let mut max_per_player: Vec<i32> = vec![0; np];
    let mut min_per_player: Vec<i32> = vec![i32::MAX; np];
    for (idx, node) in tree.nodes.iter().enumerate() {
        if !node.is_terminal() { continue; }
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        for (p, &c) in contribs.iter().enumerate() {
            max_per_player[p] = max_per_player[p].max(c);
            min_per_player[p] = min_per_player[p].min(c);
        }
        *patterns.entry(contribs.clone()).or_insert(0) += 1;
        total_terminals += 1;
    }
    eprintln!("\n=== Terminal contribution diagnostic ===");
    eprintln!("Total terminals: {}", total_terminals);
    eprintln!("Distinct contribution patterns: {}", patterns.len());
    eprintln!("Per-player contribution range:");
    for p in 0..np {
        eprintln!("  player {}: min={} max={}", p, min_per_player[p], max_per_player[p]);
    }
    eprintln!();
    eprintln!("Sample distinct patterns (up to 10):");
    let mut sorted: Vec<_> = patterns.iter().collect();
    sorted.sort_by_key(|&(_, &c)| std::cmp::Reverse(c));
    for (i, (pattern, count)) in sorted.iter().take(10).enumerate() {
        eprintln!("  #{}: {:?} (appears in {} terminals)", i, pattern, count);
    }

    // The smoking gun: if all contributions equal [10,5,5,5,5,5], no betting happened
    let starting = vec![10i32, 5, 5, 5, 5, 5];
    let starting_count = patterns.get(&starting).copied().unwrap_or(0);
    let pct_unchanged = starting_count as f64 / total_terminals as f64 * 100.0;
    eprintln!();
    eprintln!("Terminals with contributions UNCHANGED from start [10,5,5,5,5,5]: {} ({:.1}%)",
        starting_count, pct_unchanged);

    // If ALL terminals are unchanged → no betting. If SOME are unchanged → mixed.
    if pct_unchanged == 100.0 {
        eprintln!("*** SMOKING GUN: tree contains NO BETTING. All terminals are check-down. ***");
        eprintln!("    This means the 4.3M-node tree is a degenerate trivial game.");
    } else if patterns.len() == 1 {
        eprintln!("*** All terminals have IDENTICAL contributions {:?}. No diversity. ***",
            patterns.keys().next().unwrap());
    } else {
        eprintln!("Tree DOES contain betting (contributions vary). bet_sizes was applied.");
        eprintln!("The FlatNode.amount=0 finding may be a misinterpretation of that field.");
    }
}
