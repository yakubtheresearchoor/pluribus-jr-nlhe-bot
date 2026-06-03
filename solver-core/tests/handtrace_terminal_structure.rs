// Dump structural info for terminals in the HU symmetric [5,5] test config.
// Identify a terminal where per-hand CFV variation is PROVABLY REQUIRED
// (non-uniform opp_reach, hands that distinguish clearly against the opp
// distribution). This is the discriminating terminal for the hand-trace
// adjudication of #37.

use solver_core::card::{card_from_str, index_to_card_pair, Card};
use solver_core::hand::eval::Hand;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{
    ACTION_LABEL_ALLIN, ACTION_LABEL_BET, ACTION_LABEL_CALL, ACTION_LABEL_CHANCE,
    ACTION_LABEL_CHECK, ACTION_LABEL_FOLD,
};

fn label(l: u8) -> &'static str {
    match l {
        ACTION_LABEL_FOLD => "F", ACTION_LABEL_CHECK => "K", ACTION_LABEL_CALL => "C",
        ACTION_LABEL_BET => "B", ACTION_LABEL_ALLIN => "A", ACTION_LABEL_CHANCE => "+",
        _ => "?",
    }
}

#[test]
fn dump_river_terminal_structures() {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 10,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&cfg).unwrap();

    // Build parent map so we can reconstruct the path to any node.
    let mut parent: std::collections::HashMap<u32, (usize, u8)> = std::collections::HashMap::new();
    for (i, n) in tree.nodes.iter().enumerate() {
        for j in 0..n.num_children as usize {
            let c = tree.children[n.children_start as usize + j];
            let cn = &tree.nodes[c as usize];
            parent.insert(c, (i, cn.action_label));
        }
    }

    fn path_to(node_idx: usize, parent: &std::collections::HashMap<u32, (usize, u8)>) -> String {
        let mut steps: Vec<String> = Vec::new();
        let mut cur = node_idx as u32;
        while let Some(&(p, lab)) = parent.get(&cur) {
            steps.push(format!("{}:{}", p, label(lab)));
            cur = p as u32;
        }
        steps.reverse();
        steps.join(" -> ")
    }

    eprintln!("\n=== HU symmetric [5,5]: terminal nodes structural dump ===\n");

    let mut shown = 0;
    let mut by_kind: std::collections::BTreeMap<String, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, n) in tree.nodes.iter().enumerate() {
        if !n.is_terminal() { continue; }
        let contribs: Vec<i32> = (0..2).map(|p| tree.get_contribution(i, p as u8)).collect();
        let fm = tree.get_folded_mask(i);
        let bs = n.board_state;
        let kind = if fm != 0 {
            format!("fold (fm={:b}, bs={})", fm, bs)
        } else if bs == 2 {
            format!("river-showdown bs={} contribs={:?}", bs, contribs)
        } else {
            format!("unknown-terminal bs={} fm={:b}", bs, fm)
        };
        by_kind.entry(kind.clone()).or_default().push(i);
        if shown < 10 {
            eprintln!(
                "  node[{:3}]: {} contribs={:?} fm={:b} path: {}",
                i, kind, contribs, fm, path_to(i, &parent)
            );
            shown += 1;
        }
    }

    eprintln!("\nBy category:");
    for (k, v) in &by_kind {
        eprintln!("  {} terminals: {:?}{}",
            v.len(), &v[..v.len().min(8)],
            if v.len() > 8 { ", ..." } else { "" });
    }

    // Focus on node[13] specifically (the discriminating divergent terminal
    // identified in the parity-walk diagnostic).
    eprintln!("\n=== node[13] focused dump ===");
    let n = &tree.nodes[13];
    let contribs: Vec<i32> = (0..2).map(|p| tree.get_contribution(13, p as u8)).collect();
    eprintln!(
        "  node[13]: type={} bs={} contribs={:?} fm={:b} action_label={}",
        n.node_type, n.board_state, contribs, tree.get_folded_mask(13), label(n.action_label)
    );
    eprintln!("  path: {}", path_to(13, &parent));

    // Print hand info to know what's at stake on the board.
    eprintln!("\n=== Hands and their flop strengths (board 2h 7d Ks) ===");
    let board: Vec<u8> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let hand_strs = ["AhKh", "QhJh", "Th9h", "8h6h"];
    for (i, &hs) in hand_strs.iter().enumerate() {
        let cs: Vec<&str> = vec![&hs[0..2], &hs[2..4]];
        let c1 = card_from_str(cs[0]).unwrap();
        let c2 = card_from_str(cs[1]).unwrap();
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        let rank = h.evaluate_internal();
        eprintln!("  hand[{}] = {} : flop rank = {}", i, hs, rank);
    }
}
