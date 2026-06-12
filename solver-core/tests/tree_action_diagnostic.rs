// Direct diagnostic: does bet_sizes actually change the tree?
// The 4.3M-identical-across-all-configs signal from Phase 0.B is direct
// evidence that the tree-builder MAY be ignoring its config. This test
// inspects the actual actions and amounts at known decision nodes for
// several distinct bet_sizes configurations and reports whether the tree
// differs.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree,
    ACTION_LABEL_FOLD, ACTION_LABEL_CHECK, ACTION_LABEL_CALL,
    ACTION_LABEL_BET, ACTION_LABEL_RAISE, ACTION_LABEL_ALLIN};

fn action_label_name(label: u8) -> &'static str {
    match label {
        ACTION_LABEL_FOLD => "FOLD",
        ACTION_LABEL_CHECK => "CHECK",
        ACTION_LABEL_CALL => "CALL",
        ACTION_LABEL_BET => "BET",
        ACTION_LABEL_RAISE => "RAISE",
        ACTION_LABEL_ALLIN => "ALLIN",
        _ => "?",
    }
}

fn describe_node(tree: &FlatTree, idx: usize, depth: usize) -> String {
    let n = &tree.nodes[idx];
    let kind = if n.is_player() { "PLAYER" }
               else if n.is_chance() { "CHANCE" }
               else { "TERMINAL" };
    let label = action_label_name(n.action_label);
    format!("{:>indent$}[{idx}] {kind} p{} amt={} action={}({})",
        "", n.player_id, n.amount, label, n.action_label, indent=depth*2)
}

fn walk_first_few(tree: &FlatTree, root: usize, max_depth: usize) {
    fn rec(tree: &FlatTree, idx: usize, depth: usize, max_depth: usize) {
        eprintln!("{}", describe_node(tree, idx, depth));
        if depth >= max_depth { return; }
        let children = tree.node_children(idx);
        for &c in children {
            rec(tree, c as usize, depth + 1, max_depth);
        }
    }
    rec(tree, root, 0, max_depth);
}

fn config_with(bet: Vec<BetSize>, raise: Vec<BetSize>, label: &str) -> (String, TreeConfig) {
    let cfg = TreeConfig {
        num_players: 6, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet, raise },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    (label.to_string(), cfg)
}

#[test]
fn diagnose_root_children_across_configs() {
    let configs = vec![
        config_with(vec![BetSize::PotRelative(1.0)], vec![], "1 bet PotRel(1.0)"),
        config_with(vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)], vec![], "2 bets {0.5,1.0}"),
        config_with(vec![BetSize::PotRelative(0.25), BetSize::PotRelative(0.5),
                          BetSize::PotRelative(1.0), BetSize::PotRelative(2.0)], vec![], "4 bets {0.25,0.5,1,2}"),
        config_with(vec![BetSize::PotRelative(1.0)], vec![BetSize::PotRelative(1.0)], "1 bet, 1 raise"),
        // No bets at all — should ideally not even produce decisions
        config_with(vec![], vec![], "ZERO bets"),
    ];

    eprintln!("\n=== Diagnostic: do bet_sizes actually change the tree? ===\n");

    for (label, cfg) in &configs {
        let tree = build_tree(cfg).unwrap();
        let n_total = tree.nodes.len();
        let n_player = tree.nodes.iter().filter(|n| n.is_player()).count();
        let n_terminal = tree.nodes.iter().filter(|n| n.is_terminal()).count();

        let root_children: Vec<u32> = tree.node_children(0).to_vec();
        eprintln!("CONFIG: {}", label);
        eprintln!("  total nodes: {}, players: {}, terminals: {}", n_total, n_player, n_terminal);
        eprintln!("  root (node 0) has {} children:", root_children.len());
        for &c in &root_children {
            let n = &tree.nodes[c as usize];
            eprintln!("    child[{}] p{} amount={} action={}",
                c, n.player_id, n.amount, action_label_name(n.action_label));
        }

        // Walk first 2 levels to see branching
        let first_child = root_children.first().copied().unwrap_or(0);
        if first_child > 0 {
            let grandchildren = tree.node_children(first_child as usize);
            eprintln!("  first-child[{}] has {} grandchildren:", first_child, grandchildren.len());
            for &gc in grandchildren {
                let n = &tree.nodes[gc as usize];
                eprintln!("    grandchild[{}] p{} amount={} action={}",
                    gc, n.player_id, n.amount, action_label_name(n.action_label));
            }
        }
        eprintln!();
    }
}

#[test]
fn diagnose_action_diversity_global() {
    // Across the entire tree, count unique (action_label, amount) pairs.
    // If the bet_sizes config really expands the action space, we should
    // see more distinct (label, amount) pairs as we add bet sizes.
    let configs = vec![
        config_with(vec![BetSize::PotRelative(1.0)], vec![], "1 bet"),
        config_with(vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)], vec![], "2 bets"),
        config_with(vec![BetSize::PotRelative(0.25), BetSize::PotRelative(0.5),
                          BetSize::PotRelative(1.0), BetSize::PotRelative(2.0)], vec![], "4 bets"),
        config_with(vec![BetSize::PotRelative(1.0)], vec![BetSize::PotRelative(1.0)], "1 bet + 1 raise"),
    ];

    eprintln!("\n=== Diversity of (action, amount) pairs across whole tree ===");
    for (label, cfg) in &configs {
        let tree = build_tree(cfg).unwrap();
        let mut unique_pairs: std::collections::HashSet<(u8, i32)> = std::collections::HashSet::new();
        let mut amount_histogram: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for n in &tree.nodes {
            unique_pairs.insert((n.action_label, n.amount));
            *amount_histogram.entry(n.amount).or_insert(0) += 1;
        }
        let mut amounts: Vec<i32> = amount_histogram.keys().copied().collect();
        amounts.sort();
        eprintln!("\nCONFIG: {} (total nodes {})", label, tree.nodes.len());
        eprintln!("  unique (action, amount) pairs: {}", unique_pairs.len());
        eprintln!("  distinct amounts seen ({}): {:?}", amounts.len(), amounts);
    }
}
