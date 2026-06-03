// Check whether nodes are referenced by multiple parents (would explain
// the impossible contribs we see at empty PLAYER nodes).

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
fn check_multiparent() {
    let cfg = TreeConfig {
        num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![200; 3],
        initial_contributions: vec![10, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&cfg).unwrap();

    // For each child index, count how many parents reference it
    let mut parent_count: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
    for (i, n) in tree.nodes.iter().enumerate() {
        for j in 0..n.num_children as usize {
            let c = tree.children[n.children_start as usize + j];
            parent_count.entry(c).or_insert_with(Vec::new).push(i);
        }
    }

    let multi: Vec<_> = parent_count.iter().filter(|(_, v)| v.len() > 1).collect();
    eprintln!("\n=== Multi-parent nodes (would indicate tree corruption): {} ===", multi.len());
    for (child, parents) in multi.iter().take(20) {
        eprintln!("  node[{}] has {} parents: {:?}", child, parents.len(), parents);
    }

    // For node[129] specifically
    eprintln!("\n=== node[129] all parents: ===");
    if let Some(parents) = parent_count.get(&129u32) {
        for &p in parents {
            let pn = &tree.nodes[p];
            let pcontribs: Vec<i32> = (0..3).map(|i| tree.get_contribution(p, i as u8)).collect();
            eprintln!("  parent: node[{}] type={} p{} nc={} act={} contribs={:?}",
                p, pn.node_type, pn.player_id, pn.num_children, pn.action_label, pcontribs);
        }
    } else {
        eprintln!("  no parents (orphan?)");
    }

    // Total orphan check
    let mut reachable = std::collections::HashSet::new();
    let mut stack = vec![0usize];
    while let Some(idx) = stack.pop() {
        if !reachable.insert(idx) { continue; }
        let n = &tree.nodes[idx];
        for j in 0..n.num_children as usize {
            let c = tree.children[n.children_start as usize + j] as usize;
            stack.push(c);
        }
    }
    let orphans: usize = (0..tree.num_nodes()).filter(|i| !reachable.contains(i)).count();
    eprintln!("\nReachable from root: {} / {} (orphans: {})",
        reachable.len(), tree.num_nodes(), orphans);
}
