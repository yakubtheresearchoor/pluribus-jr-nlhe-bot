// Verify whether the 77.8% 1-child player infosets are GENUINELY rules-forced
// (player all-in or other narrow situation) or BUILDER-COLLAPSED (real decisions
// missing because the tree-builder under-generates actions).
//
// The distinguishing test:
//   - If the single child's action is CHECK: legitimate when player has 0
//     remaining stack (all-in forced). Suspicious if player has chips
//     remaining and was first-to-act — should have {CHECK, BET sizes}.
//   - If the single child's action is CALL: BUG — facing a bet always has
//     at minimum {FOLD, CALL}. A 1-child CALL means FOLD was not generated.
//   - If the single child's action is FOLD: BUG — same as above, CALL not
//     generated.
//   - If the single child's action is ALLIN: usually correct (forced to allin
//     when max_amount equals player_remaining, no room to bet less).

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{
    ACTION_LABEL_CHECK, ACTION_LABEL_CALL, ACTION_LABEL_FOLD,
    ACTION_LABEL_BET, ACTION_LABEL_RAISE, ACTION_LABEL_ALLIN,
};

fn label_name(l: u8) -> &'static str {
    match l {
        ACTION_LABEL_FOLD => "FOLD",
        ACTION_LABEL_CHECK => "CHECK",
        ACTION_LABEL_CALL => "CALL",
        ACTION_LABEL_BET => "BET",
        ACTION_LABEL_RAISE => "RAISE",
        ACTION_LABEL_ALLIN => "ALLIN",
        _ => "OTHER",
    }
}

#[test]
fn verify_one_child_nodes_are_forced_or_bugs() {
    let cfg = TreeConfig {
        num_players: 6, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&cfg).unwrap();
    let np = 6usize;
    let starting_stacks = vec![200i32; 6];
    let initial_contribs = vec![10i32, 5, 5, 5, 5, 5];
    let max_committable: Vec<i32> = starting_stacks.iter().zip(&initial_contribs)
        .map(|(s, c)| s + c).collect();

    eprintln!("\n=== 1-child player node verification ===\n");

    // Categorize all 1-child player nodes by the action label of their child.
    let mut counts_by_action: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut by_player_remaining: std::collections::HashMap<u8, (usize, usize)>
        = std::collections::HashMap::new(); // (action_label, n_with_chips_remaining)

    let mut total_one_child = 0usize;
    for (idx, n) in tree.nodes.iter().enumerate() {
        if !n.is_player() || n.num_children != 1 { continue; }
        total_one_child += 1;
        let child_idx = tree.children[n.children_start as usize] as usize;
        let child = &tree.nodes[child_idx];
        let action = child.action_label;
        *counts_by_action.entry(action).or_insert(0) += 1;

        // Look up player's chip state at THIS node from tree.contributions
        let player = n.player_id;
        let player_committed = tree.get_contribution(idx, player);
        let player_remaining = max_committable[player as usize] - player_committed;

        let entry = by_player_remaining.entry(action).or_insert((0, 0));
        entry.0 += 1;
        if player_remaining > 0 { entry.1 += 1; }
    }

    eprintln!("Total 1-child player nodes: {}\n", total_one_child);
    eprintln!("Breakdown by child's action label:");
    eprintln!("{:>10} | {:>12} | {:>12} | {:>14} | {:>10}",
        "action", "count", "% of 1ch", "w/ chips left", "% w/chips");
    eprintln!("{}", "-".repeat(75));
    let mut keys: Vec<u8> = counts_by_action.keys().copied().collect();
    keys.sort();
    for a in &keys {
        let count = counts_by_action[a];
        let with_chips = by_player_remaining[a].1;
        let pct_of_total = count as f64 / total_one_child as f64 * 100.0;
        let pct_with_chips = with_chips as f64 / count as f64 * 100.0;
        eprintln!("{:>10} | {:>12} | {:>11.2}% | {:>14} | {:>9.2}%",
            label_name(*a), count, pct_of_total, with_chips, pct_with_chips);
    }

    // Interpretation
    eprintln!();
    eprintln!("=== Interpretation ===");
    let check_count = counts_by_action.get(&ACTION_LABEL_CHECK).copied().unwrap_or(0);
    let call_count = counts_by_action.get(&ACTION_LABEL_CALL).copied().unwrap_or(0);
    let fold_count = counts_by_action.get(&ACTION_LABEL_FOLD).copied().unwrap_or(0);
    let bet_count = counts_by_action.get(&ACTION_LABEL_BET).copied().unwrap_or(0);
    let allin_count = counts_by_action.get(&ACTION_LABEL_ALLIN).copied().unwrap_or(0);

    let check_with_chips = by_player_remaining.get(&ACTION_LABEL_CHECK).map(|e| e.1).unwrap_or(0);
    let check_forced = check_count - check_with_chips;

    eprintln!();
    eprintln!("CHECK as only action: {}", check_count);
    eprintln!("  → forced (player has 0 chips left, all-in): {}", check_forced);
    eprintln!("  → SUSPICIOUS (player has chips, should have BET option): {}", check_with_chips);
    eprintln!();
    eprintln!("CALL as only action: {} (BUG if non-zero: facing-bet always has FOLD + CALL minimum)", call_count);
    eprintln!("FOLD as only action: {} (BUG if non-zero: facing-bet always has FOLD + CALL minimum)", fold_count);
    eprintln!("BET as only action:  {} (suspicious: first-to-act should have CHECK option too)", bet_count);
    eprintln!("ALLIN as only action: {} (usually OK: forced when bet would exceed remaining)", allin_count);

    let definitively_buggy = call_count + fold_count;
    let suspicious = check_with_chips + bet_count;
    eprintln!();
    if definitively_buggy > 0 {
        eprintln!("✗ DEFINITIVELY BUG: {} player nodes where the SINGLE child is CALL or FOLD",
            definitively_buggy);
        eprintln!("  facing-a-bet always has FOLD + CALL minimum. Builder is collapsing.");
    }
    if suspicious > 0 {
        eprintln!("⚠ SUSPICIOUS: {} player nodes where the single child is CHECK (player has chips) or BET",
            suspicious);
        eprintln!("  These ARE generated by builder but the alternative wasn't — collapse possible.");
    }
    if definitively_buggy == 0 && check_with_chips == 0 && bet_count == 0 {
        eprintln!("✓ All 1-child nodes are explained by forced situations (all-in CHECK or forced ALLIN).");
        eprintln!("  The 77.8%-1-child finding is genuine redundancy — elision opportunity is real.");
    }

    // Print 10 sample suspicious nodes for hand-verification
    if check_with_chips > 0 || call_count > 0 || fold_count > 0 {
        eprintln!();
        eprintln!("=== Sample suspicious 1-child nodes (up to 10) ===");
        let mut printed = 0;
        for (idx, n) in tree.nodes.iter().enumerate() {
            if printed >= 10 { break; }
            if !n.is_player() || n.num_children != 1 { continue; }
            let child_idx = tree.children[n.children_start as usize] as usize;
            let child = &tree.nodes[child_idx];
            let action = child.action_label;
            let player = n.player_id;
            let player_committed = tree.get_contribution(idx, player);
            let player_remaining = max_committable[player as usize] - player_committed;
            let is_suspicious = match action {
                ACTION_LABEL_CHECK => player_remaining > 0,
                ACTION_LABEL_CALL | ACTION_LABEL_FOLD => true,
                _ => false,
            };
            if !is_suspicious { continue; }
            let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
            eprintln!("  node[{}] player={} board_state={} contribs={:?} player_remaining={} → single child action={}",
                idx, player, n.board_state, contribs, player_remaining, label_name(action));
            printed += 1;
        }
    }
}
