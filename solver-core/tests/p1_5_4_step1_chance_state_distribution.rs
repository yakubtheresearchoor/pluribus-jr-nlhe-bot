// Step 1 — Probe 2: preflop chance-node state distribution.
//
// The shared-tree architecture uses ONE 1.52M-node flop tree for every
// preflop chance node. A per-terminal architecture would build a flop
// tree sized to each chance node's specific (pot, remaining-stack-tuple)
// state. This probe counts how many DISTINCT such states the 162,650-node
// Option-B 6-max preflop tree produces — that's how many flop trees a
// per-terminal architecture would need, and the reduction-factor estimate
// for per-call cost.
//
// If distinct-state count is small (e.g., ~10) and per-state flop trees
// are ~10x smaller each, per-terminal could reduce total work substantially
// even with the build overhead. If distinct-state count is large (~hundreds),
// the build overhead dominates and shared-tree might still win.

use std::collections::BTreeMap;

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_optb_6max_preflop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; 6],
        initial_contributions: vec![1, 2, 0, 0, 0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
            max_bets_per_street: None,
    };
    build_tree(&cfg).expect("preflop tree builds")
}

#[test]
fn probe_two_chance_node_state_distribution() {
    eprintln!("\n=== Probe 2: distinct chance-node states in 162,650-node Option-B preflop tree ===\n");

    let tree = build_optb_6max_preflop_tree();
    let np = tree.num_players as usize;
    eprintln!("Preflop tree: {} nodes, {} players", tree.num_nodes(), np);

    // Walk every chance node; extract its commits → (pot, stacks-tuple).
    // For each chance node, the state at preflop→flop is:
    //   - per-player committed (i32 each)
    //   - implied remaining stack = starting_stack - committed
    //   - pot = sum of committed
    // Two chance nodes are "same state" if their commits-tuples are equal
    // (after sorting if player identity doesn't matter — but it might matter
    // for postflop ordering, so we keep ordered).
    //
    // We count BOTH ordered (player-identity-distinct) and unordered (commit-
    // multiset-distinct) distinct states. Unordered is what a position-invariant
    // flop solver could reuse; ordered is what a position-aware solver needs.
    let mut ordered_counts: BTreeMap<Vec<i32>, u32> = BTreeMap::new();
    let mut unordered_counts: BTreeMap<Vec<i32>, u32> = BTreeMap::new();
    let mut pot_distribution: BTreeMap<i32, u32> = BTreeMap::new();
    let mut total_chance_nodes = 0u32;

    for (idx, node) in tree.nodes.iter().enumerate() {
        if !node.is_chance() { continue; }
        total_chance_nodes += 1;
        let commits: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let pot: i32 = commits.iter().sum();
        *pot_distribution.entry(pot).or_insert(0) += 1;
        *ordered_counts.entry(commits.clone()).or_insert(0) += 1;
        let mut sorted = commits.clone();
        sorted.sort();
        *unordered_counts.entry(sorted).or_insert(0) += 1;
    }

    eprintln!("Total chance nodes: {}", total_chance_nodes);
    eprintln!("Distinct (ordered) commit-tuples:   {}", ordered_counts.len());
    eprintln!("Distinct (unordered) commit-multisets: {}", unordered_counts.len());
    eprintln!("Distinct pot sizes:                 {}", pot_distribution.len());

    eprintln!("\nPot-size distribution (pot → count of chance nodes):");
    for (pot, count) in &pot_distribution {
        eprintln!("  pot={:4} : {:5} chance nodes ({:.1}%)",
                  pot, count, 100.0 * (*count as f64) / (total_chance_nodes as f64));
    }

    eprintln!("\nTop 20 most common (ordered) commit-tuples:");
    let mut by_count: Vec<_> = ordered_counts.iter().collect();
    by_count.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (commits, count) in by_count.iter().take(20) {
        eprintln!("  {:?} : {:5} chance nodes", commits, count);
    }

    eprintln!("\n=== Per-terminal flop-tree architecture estimate ===");
    eprintln!("  Shared-tree:    1 flop tree shared by {} chance nodes", total_chance_nodes);
    eprintln!("  Per-terminal:   {} flop trees (one per distinct ordered state),", ordered_counts.len());
    eprintln!("                  shared by avg {:.1} chance nodes each",
              total_chance_nodes as f64 / ordered_counts.len() as f64);
    eprintln!("  Position-aware: {} flop trees (one per distinct unordered state),", unordered_counts.len());
    eprintln!("                  if the postflop solver is position-invariant in some sense");
    eprintln!("\nEach per-terminal flop tree would be sized to ITS specific (pot, stacks),");
    eprintln!("which is dramatically smaller than the shared 1.52M-node tree for the");
    eprintln!("largest-stack-depth case. The unit-cost reduction is what the architecture");
    eprintln!("trade-off depends on.");
}
