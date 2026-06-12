//! Probe: do Call children record the caller's matched contribution?
//! Oracle config (6p, pot 12, stacks 94, 1.0x pot bet, no raises).
//! Walk root -> bet (child 0) -> call (child 0) and print contributions.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
fn call_contribution_probe() {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 12,
        starting_stacks: vec![94; 6],
        initial_contributions: vec![0; 6],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("tree");

    let dump = |label: &str, idx: usize| {
        let n = &tree.nodes[idx];
        let contribs: Vec<i32> = (0..6).map(|p| tree.get_contribution(idx, p as u8)).collect();
        println!(
            "{label}: node {idx} type={} player={} label={} contribs={:?} folded={:06b}",
            n.node_type, n.player_id, n.action_label, contribs, tree.get_folded_mask(idx)
        );
    };

    dump("root", 0);
    for (i, &c) in tree.node_children(0).iter().enumerate() {
        dump(&format!("root->c{i}"), c as usize);
    }
    // Find the BET child of the root (label 3) and walk it.
    let bet = *tree
        .node_children(0)
        .iter()
        .find(|&&c| tree.nodes[c as usize].action_label == 3)
        .expect("bet child") as usize;
    for (i, &c) in tree.node_children(bet).iter().enumerate() {
        dump(&format!("bet->c{i}"), c as usize);
    }
    // Find the CALL child of the bet node (label 2) and walk it.
    let call = *tree
        .node_children(bet)
        .iter()
        .find(|&&c| tree.nodes[c as usize].action_label == 2)
        .expect("call child") as usize;
    for (i, &c) in tree.node_children(call).iter().enumerate() {
        dump(&format!("call->c{i}"), c as usize);
    }
    // Pinned semantics: a Call child must record the caller matching
    // the bettor's contribution.
    let caller = tree.nodes[bet].player_id;
    assert_eq!(
        tree.get_contribution(call, caller),
        12,
        "caller (player {caller}) must record 12 at the call child"
    );

    // FOLD-CONTINUATION (2026-06-12 fix): p1 folding to p0's bet must
    // NOT end the hand — p2..p5 still act. The fold child is a PLAYER
    // node with the next active player to act and p1's fold bit set.
    let fold = *tree
        .node_children(bet)
        .iter()
        .find(|&&c| tree.nodes[c as usize].action_label == 0)
        .expect("fold child") as usize;
    dump("bet->fold", fold);
    assert_eq!(tree.nodes[fold].node_type, 2, "fold child must stay a player node multiway");
    assert_eq!(tree.nodes[fold].player_id, 2, "after p1 folds, p2 acts");
    assert_eq!(tree.get_folded_mask(fold), 0b000010);
    // Walk the all-fold line: p1..p5 fold in turn → terminal with all
    // five fold bits set (the mask the harness needs for settlement).
    let mut n = bet;
    for _ in 0..5 {
        n = *tree
            .node_children(n)
            .iter()
            .find(|&&c| tree.nodes[c as usize].action_label == 0)
            .expect("fold child") as usize;
    }
    dump("all-fold terminal", n);
    assert_eq!(tree.nodes[n].node_type, 0, "five folds leave one player: terminal");
    assert_eq!(tree.get_folded_mask(n), 0b111110, "all five fold bits must be present");
    println!("total nodes: {}", tree.nodes.len());
}
