// Probe: inspect the preflop-rooted tree structure to understand where
// chance nodes live and how to identify them. Diagnostic; can be deleted
// after Slice A.3a's chance-node identification is settled.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
fn probe_hu_preflop_tree_structure() {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![99, 98],
        initial_contributions: vec![1, 2],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&cfg).expect("preflop tree builds");
    eprintln!("Total nodes: {}", tree.num_nodes());
    eprintln!("Decision nodes: {}", tree.decision_node_ids.len());

    let mut by_state: [(usize, usize, usize); 4] = [(0, 0, 0); 4]; // (player, chance, terminal)
    for idx in 0..tree.num_nodes() {
        let n = &tree.nodes[idx];
        let s = n.board_state as usize;
        if s < 4 {
            if n.is_player() { by_state[s].0 += 1; }
            else if n.is_chance() { by_state[s].1 += 1; }
            else { by_state[s].2 += 1; }
        }
    }
    eprintln!("");
    eprintln!("board_state breakdown (player / chance / terminal):");
    eprintln!("  0 (Flop):    {:?}", by_state[0]);
    eprintln!("  1 (Turn):    {:?}", by_state[1]);
    eprintln!("  2 (River):   {:?}", by_state[2]);
    eprintln!("  3 (Preflop): {:?}", by_state[3]);

    // For chance nodes: print which preflop player parent (if any) they
    // descend from.
    eprintln!("\nChance nodes and their immediate parents:");
    let mut parents_of = vec![None::<u32>; tree.num_nodes()];
    for parent_idx in 0..tree.num_nodes() {
        for &child in tree.node_children(parent_idx) {
            parents_of[child as usize] = Some(parent_idx as u32);
        }
    }
    let mut chance_idxs = Vec::new();
    for idx in 0..tree.num_nodes() {
        if tree.nodes[idx].is_chance() {
            chance_idxs.push(idx);
        }
    }
    eprintln!("Total chance nodes: {}", chance_idxs.len());
    // Print at most 20 chance nodes (just to see structure).
    for &idx in chance_idxs.iter().take(20) {
        let n = &tree.nodes[idx];
        let parent_idx = parents_of[idx];
        let parent_state = parent_idx.map(|p| tree.nodes[p as usize].board_state);
        eprintln!("  node {}: board_state={}, parent={:?} (parent board_state={:?})",
            idx, n.board_state, parent_idx, parent_state);
    }

    // Specifically count chance nodes whose parent is in the preflop zone
    // (board_state == 3) — these are the preflop→flop chance nodes.
    let preflop_to_flop_chance: Vec<usize> = chance_idxs.iter()
        .filter(|&&idx| {
            let pp = parents_of[idx];
            pp.map(|p| tree.nodes[p as usize].board_state == BoardState::Preflop as u8)
              .unwrap_or(false)
        })
        .copied()
        .collect();
    eprintln!("\nPreflop→Flop chance nodes (chance whose parent is preflop): {} found",
        preflop_to_flop_chance.len());
    for &idx in &preflop_to_flop_chance {
        eprintln!("  node {}: board_state={} (its own), parent={:?}",
            idx, tree.nodes[idx].board_state, parents_of[idx]);
    }
}
