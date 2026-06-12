// Diagnostic: trace one violating sequence back to root to find the
// remaining wrong-game logic the partial fix didn't catch.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

#[test]
fn b4_pre_trace_violating_path() {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![2, 1],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).unwrap();
    let np = tree.num_players;

    // Find first violating flop player node.
    let mut violator: Option<usize> = None;
    for idx in 0..tree.num_nodes() {
        let n = &tree.nodes[idx];
        if !n.is_player() || n.board_state != BoardState::Flop as u8 { continue; }
        if tree.get_folded_mask(idx) != 0 { continue; }
        let c: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p)).collect();
        if !c.iter().all(|&x| x == c[0]) {
            violator = Some(idx);
            break;
        }
    }
    let violator = violator.expect("expected at least one violation");
    eprintln!("First violating flop node: {}", violator);

    // Build parent map.
    let mut parent_of = vec![None::<usize>; tree.num_nodes()];
    for p in 0..tree.num_nodes() {
        for &c in tree.node_children(p) {
            parent_of[c as usize] = Some(p);
        }
    }

    // Walk back to root.
    let mut chain: Vec<usize> = Vec::new();
    let mut cur = violator;
    loop {
        chain.push(cur);
        if cur == 0 { break; }
        cur = parent_of[cur].expect("non-root should have parent");
    }
    chain.reverse();

    eprintln!("\nPath from root to violation:");
    eprintln!("{:<7} {:<10} {:<8} {:<10} {:<20} {:<15} {}",
        "node", "type", "state", "player_id", "contributions", "fold_mask",
        "act_label");
    eprintln!("{}", "-".repeat(100));
    for &idx in &chain {
        let n = &tree.nodes[idx];
        let kind = if n.is_player() { "PLAYER" }
                   else if n.is_chance() { "CHANCE" }
                   else { "TERMINAL" };
        let bs = match n.board_state {
            0 => "Flop", 1 => "Turn", 2 => "River", 3 => "Preflop",
            _ => "?",
        };
        let c: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p)).collect();
        let mask = tree.get_folded_mask(idx);
        eprintln!("{:<7} {:<10} {:<8} {:<10} {:<20} {:<15} {}",
            idx, kind, bs, n.player_id, format!("{:?}", c), format!("0b{:0b}", mask),
            n.action_label);
    }
    eprintln!("");
    eprintln!("ACTION LABELS: 0=Fold, 1=Check, 2=Call, 3=Bet, 4=Raise, 5=AllIn, 6=Chance");
}
