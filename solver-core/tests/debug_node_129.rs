// Focused debug on the empty PLAYER nodes. node[129] in 3p config is one
// such node. Let me look at its full state and trace its origin.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{ACTION_LABEL_FOLD, ACTION_LABEL_CHECK, ACTION_LABEL_CALL,
    ACTION_LABEL_BET, ACTION_LABEL_RAISE, ACTION_LABEL_ALLIN, ACTION_LABEL_CHANCE};

fn label_name(l: u8) -> &'static str {
    match l {
        ACTION_LABEL_FOLD => "FOLD", ACTION_LABEL_CHECK => "CHECK",
        ACTION_LABEL_CALL => "CALL", ACTION_LABEL_BET => "BET",
        ACTION_LABEL_RAISE => "RAISE", ACTION_LABEL_ALLIN => "ALLIN",
        ACTION_LABEL_CHANCE => "CHANCE", _ => "?",
    }
}

fn type_name(t: u8) -> &'static str {
    match t { 0 => "TERM", 1 => "CHCE", 2 => "PLAY", _ => "?" }
}

#[test]
fn trace_empty_player_node() {
    let cfg = TreeConfig {
        num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![200; 3],
        initial_contributions: vec![10, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&cfg).unwrap();

    // Walk down from root, find parent of node 129
    eprintln!("\n=== Trace to node[129] ===");

    fn dump(tree: &solver_core::tree::flat::FlatTree, idx: usize, depth: usize) {
        let n = &tree.nodes[idx];
        let contribs: Vec<i32> = (0..3).map(|p| tree.get_contribution(idx, p as u8)).collect();
        eprintln!("{:>indent$}node[{:5}] {} p{} act={} amt={} nc={} contribs={:?}",
            "", idx, type_name(n.node_type), n.player_id, label_name(n.action_label),
            n.amount, n.num_children, contribs, indent=depth*2);
    }

    fn walk(tree: &solver_core::tree::flat::FlatTree, idx: usize, depth: usize, target: usize, max_depth: usize) -> bool {
        if depth > max_depth { return false; }
        dump(tree, idx, depth);
        if idx == target { return true; }
        let n = &tree.nodes[idx];
        for i in 0..n.num_children as usize {
            let c = tree.children[n.children_start as usize + i] as usize;
            if walk(tree, c, depth + 1, target, max_depth) { return true; }
        }
        false
    }

    walk(&tree, 0, 0, 129, 15);

    eprintln!();
    eprintln!("--- Direct dump of node[129] ---");
    let n = &tree.nodes[129];
    eprintln!("  node_type = {} ({})", n.node_type, type_name(n.node_type));
    eprintln!("  player_id = {}", n.player_id);
    eprintln!("  num_children = {}", n.num_children);
    eprintln!("  children_start = {}", n.children_start);
    eprintln!("  action_label = {} ({})", n.action_label, label_name(n.action_label));
    eprintln!("  amount = {}", n.amount);
    eprintln!("  board_state = {}", n.board_state);
    let contribs: Vec<i32> = (0..3).map(|p| tree.get_contribution(129, p as u8)).collect();
    eprintln!("  contribs = {:?}", contribs);
    if n.num_children > 0 {
        eprintln!("  children:");
        for i in 0..n.num_children as usize {
            let c = tree.children[n.children_start as usize + i] as usize;
            let cn = &tree.nodes[c];
            eprintln!("    [{}] node[{}] {} p{} act={}", i, c, type_name(cn.node_type), cn.player_id, label_name(cn.action_label));
        }
    }
}
