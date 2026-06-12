// Confirm the second anomaly: terminals contain players with contributions
// exceeding their starting stack. A player cannot commit more than their
// stack in real poker, so such terminals represent physically impossible
// game states. If confirmed, the tree has been solving phantom states the
// whole time — a correctness bug deeper than the bet-collapse.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
fn check_contributions_dont_exceed_starting_stack() {
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
    let starting_stacks = vec![200i32; 6];
    let initial_contribs = vec![10i32, 5, 5, 5, 5, 5];
    let max_possible: Vec<i32> = starting_stacks.iter().zip(&initial_contribs)
        .map(|(s, c)| s + c).collect();  // total chips player has = stack + already-contributed

    eprintln!("\n=== Physical-validity check: contributions vs max possible (starting_stack + initial_contrib) ===");
    for p in 0..np {
        eprintln!("  player {}: starting_stack={}, initial_contrib={}, MAX possible total commitment = {}",
            p, starting_stacks[p], initial_contribs[p], max_possible[p]);
    }

    let mut violations_by_player: Vec<usize> = vec![0; np];
    let mut max_contrib_seen: Vec<i32> = vec![0; np];
    let mut worst_violation_terminal: Option<(usize, Vec<i32>)> = None;
    let mut worst_excess = 0i32;
    let mut total_terminals = 0usize;
    let mut any_violation_terminals = 0usize;
    for (idx, node) in tree.nodes.iter().enumerate() {
        if !node.is_terminal() { continue; }
        total_terminals += 1;
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let mut had_violation = false;
        for p in 0..np {
            if contribs[p] > max_possible[p] {
                violations_by_player[p] += 1;
                had_violation = true;
                let excess = contribs[p] - max_possible[p];
                if excess > worst_excess {
                    worst_excess = excess;
                    worst_violation_terminal = Some((idx, contribs.clone()));
                }
            }
            max_contrib_seen[p] = max_contrib_seen[p].max(contribs[p]);
        }
        if had_violation { any_violation_terminals += 1; }
    }

    eprintln!();
    eprintln!("Total terminals: {}", total_terminals);
    eprintln!("Terminals with at least one player exceeding max possible: {} ({:.1}%)",
        any_violation_terminals, any_violation_terminals as f64 / total_terminals as f64 * 100.0);
    eprintln!();
    eprintln!("Per-player breakdown:");
    for p in 0..np {
        let excess = max_contrib_seen[p] - max_possible[p];
        eprintln!("  player {}: max_seen={} (max_possible={}), excess={}{}, violations_at={} terminals ({:.1}%)",
            p, max_contrib_seen[p], max_possible[p], excess,
            if excess > 0 { " (PHYSICALLY IMPOSSIBLE)" } else { "" },
            violations_by_player[p],
            violations_by_player[p] as f64 / total_terminals as f64 * 100.0);
    }

    if let Some((idx, contribs)) = worst_violation_terminal {
        eprintln!();
        eprintln!("Worst violation: terminal node {} with contributions {:?}", idx, contribs);
        eprintln!("  Excess over max_possible: {} chips", worst_excess);
    }

    eprintln!();
    eprintln!("=== INTERPRETATION ===");
    if any_violation_terminals == 0 {
        eprintln!("✓ All terminals have contributions ≤ starting_stack. Tree is physically valid.");
    } else {
        let pct = any_violation_terminals as f64 / total_terminals as f64 * 100.0;
        eprintln!("✗ {:.1}% of terminals contain players committed for MORE chips than they have.", pct);
        eprintln!("  These terminals represent physically impossible game states.");
        eprintln!("  Tree has been solving phantom game states.");
        eprintln!("  Correctness bug deeper than bet-collapse: regret/strategy data at these");
        eprintln!("  nodes corresponds to game states that cannot actually occur.");
    }
}
