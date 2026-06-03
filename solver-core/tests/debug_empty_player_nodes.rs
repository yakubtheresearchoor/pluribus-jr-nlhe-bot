// Diagnostic: find where empty PLAYER nodes are being created in the tree
// after Phase 3 restructure. The standing gate reports 110,893 such nodes
// across 3 configs — they're the only remaining violation class.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{
    ACTION_LABEL_FOLD, ACTION_LABEL_CHECK, ACTION_LABEL_CALL,
    ACTION_LABEL_BET, ACTION_LABEL_RAISE, ACTION_LABEL_ALLIN, ACTION_LABEL_CHANCE,
};

fn label_name(l: u8) -> &'static str {
    match l {
        ACTION_LABEL_FOLD => "FOLD",
        ACTION_LABEL_CHECK => "CHECK",
        ACTION_LABEL_CALL => "CALL",
        ACTION_LABEL_BET => "BET",
        ACTION_LABEL_RAISE => "RAISE",
        ACTION_LABEL_ALLIN => "ALLIN",
        ACTION_LABEL_CHANCE => "CHANCE",
        _ => "?",
    }
}

#[test]
fn dump_empty_player_node_samples() {
    let cfg = TreeConfig {
        num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![200; 3],
        initial_contributions: vec![10, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&cfg).unwrap();
    let np = 3;

    let mut empty_players: Vec<usize> = Vec::new();
    for (idx, n) in tree.nodes.iter().enumerate() {
        if n.is_player() && n.num_children == 0 {
            empty_players.push(idx);
        }
    }

    eprintln!("\n=== {} empty PLAYER nodes in 3p config ===", empty_players.len());
    eprintln!("First 20 samples:");
    for &idx in empty_players.iter().take(20) {
        let n = &tree.nodes[idx];
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        eprintln!("  node[{:6}] player_id={} board_state={} action_label={}({}) amount={} contribs={:?}",
            idx, n.player_id, n.board_state,
            label_name(n.action_label), n.action_label, n.amount, contribs);
    }

    // Aggregate by action_label
    let mut by_label: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut by_player: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut by_board: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    for &idx in &empty_players {
        let n = &tree.nodes[idx];
        *by_label.entry(n.action_label).or_insert(0) += 1;
        *by_player.entry(n.player_id).or_insert(0) += 1;
        *by_board.entry(n.board_state).or_insert(0) += 1;
    }
    eprintln!("\nBy action_label:");
    let mut labels: Vec<u8> = by_label.keys().copied().collect();
    labels.sort();
    for l in &labels {
        eprintln!("  {}({}): {}", label_name(*l), l, by_label[l]);
    }
    eprintln!("\nBy player_id:");
    let mut players: Vec<u8> = by_player.keys().copied().collect();
    players.sort();
    for p in &players {
        eprintln!("  p{}: {}", p, by_player[p]);
    }
    eprintln!("\nBy board_state:");
    let mut bs: Vec<u8> = by_board.keys().copied().collect();
    bs.sort();
    for s in &bs {
        let name = match *s { 0 => "Flop", 1 => "Turn", 2 => "River", _ => "?" };
        eprintln!("  {}({}): {}", name, s, by_board[s]);
    }
}
