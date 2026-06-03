// Find the parents of empty PLAYER nodes — what was their compute_actions
// producing?

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
fn empty_player_parents_3p() {
    let cfg = TreeConfig {
        num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![200; 3],
        initial_contributions: vec![10, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&cfg).unwrap();

    // Build parent map
    let mut parent: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for (i, n) in tree.nodes.iter().enumerate() {
        for j in 0..n.num_children as usize {
            let c = tree.children[n.children_start as usize + j];
            parent.insert(c, i);
        }
    }

    let mut empties: Vec<usize> = Vec::new();
    for (i, n) in tree.nodes.iter().enumerate() {
        if n.is_player() && n.num_children == 0 {
            empties.push(i);
        }
    }
    eprintln!("\n=== {} empty PLAYER nodes; first 10 with parents ===\n", empties.len());
    for &idx in empties.iter().take(10) {
        let n = &tree.nodes[idx];
        let contribs: Vec<i32> = (0..3).map(|p| tree.get_contribution(idx, p as u8)).collect();
        eprintln!("Empty: node[{}] p{} bs={} act={} contribs={:?}",
            idx, n.player_id, n.board_state, label_name(n.action_label), contribs);

        if let Some(&pidx) = parent.get(&(idx as u32)) {
            let pn = &tree.nodes[pidx];
            let pcontribs: Vec<i32> = (0..3).map(|p| tree.get_contribution(pidx, p as u8)).collect();
            eprintln!("  parent: node[{}] {} p{} bs={} act={} nc={} contribs={:?}",
                pidx, type_name(pn.node_type), pn.player_id, pn.board_state,
                label_name(pn.action_label), pn.num_children, pcontribs);
            eprintln!("  parent's children:");
            for j in 0..pn.num_children as usize {
                let c = tree.children[pn.children_start as usize + j] as usize;
                let cn = &tree.nodes[c];
                let ccontribs: Vec<i32> = (0..3).map(|p| tree.get_contribution(c, p as u8)).collect();
                eprintln!("    [{}] node[{}] {} p{} act={} nc={} contribs={:?}",
                    j, c, type_name(cn.node_type), cn.player_id,
                    label_name(cn.action_label), cn.num_children, ccontribs);
            }
        } else {
            eprintln!("  NO PARENT (orphan)");
        }
        eprintln!();
    }
}
