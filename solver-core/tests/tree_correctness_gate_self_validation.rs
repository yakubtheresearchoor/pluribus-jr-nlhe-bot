// Validate the standing tree-correctness gate against tiny, hand-checkable
// trees BEFORE trusting its count to drive the tree-builder rewrite scope.
//
// The gate is the newest untested component. It already has one known
// classification bug (cumulative-vs-per-street in the original audit) and
// flagged 43% of nodes in 3p as "folded player acts" which is suspicious.
// A gate you have not validated cannot be the thing you gate on.
//
// Methodology:
//   1. Build a tiny tree (2p heads-up, equal blinds, single bet structure)
//      that is small enough to hand-enumerate every node's correct legal
//      action set per poker rules.
//   2. Run the gate and confirm: zero violations on this tree.
//   3. Cross-check by walking the tree manually and comparing against
//      what the gate reports.
//
// If the gate reports zero on a hand-verified correct tree: gate logic is
// trustworthy for the at-least-this-simple case.
// If the gate reports violations on a tree that's actually correct: gate
// has false-positives, fix the gate before using its count.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{
    FlatTree,
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

/// Walk tree depth-first, printing each player node with its acting state and
/// generated actions. For HAND verification against poker rules.
fn dump_player_nodes(
    tree: &FlatTree, np: usize, max_committable: &[i32],
    max_nodes_to_print: usize,
) {
    let initial_snapshot: Vec<i32> = (0..np)
        .map(|p| tree.get_contribution(0, p as u8))
        .collect();
    let mut stack: Vec<(usize, Vec<i32>, u16, usize)> = vec![(0, initial_snapshot, 0, 0)];
    let mut printed = 0;

    eprintln!("Hand-verification dump (up to {} player nodes):", max_nodes_to_print);
    eprintln!("{:>5} | {:>5} | {:>5} | {:>13} | {:>8} | {:>10} | actions",
        "node", "depth", "type", "contribs", "player", "remaining");
    eprintln!("{}", "-".repeat(100));

    while let Some((idx, round_start, folded_mask, depth)) = stack.pop() {
        let n = &tree.nodes[idx];
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();

        if n.is_chance() {
            if printed < max_nodes_to_print {
                eprintln!("{:>5} | {:>5} | {:>5} | {:?}",
                    idx, depth, "CHCE", contribs);
                printed += 1;
            }
            let new_snapshot: Vec<i32> = contribs.clone();
            for i in 0..n.num_children as usize {
                let c = tree.children[n.children_start as usize + i] as usize;
                stack.push((c, new_snapshot.clone(), folded_mask, depth + 1));
            }
            continue;
        }
        if n.is_terminal() {
            if printed < max_nodes_to_print {
                eprintln!("{:>5} | {:>5} | {:>5} | {:?}",
                    idx, depth, "TERM", contribs);
                printed += 1;
            }
            continue;
        }

        let player = n.player_id;
        let nc = n.num_children as usize;
        let actions: Vec<String> = (0..nc)
            .map(|i| {
                let c = &tree.nodes[tree.children[n.children_start as usize + i] as usize];
                label_name(c.action_label).to_string()
            })
            .collect();

        let per_street: Vec<i32> = (0..np).map(|p| contribs[p] - round_start[p]).collect();
        let max_other_street: i32 = (0..np)
            .filter(|&p| p != player as usize && (folded_mask & (1u16 << p)) == 0)
            .map(|p| per_street[p])
            .max().unwrap_or(0);
        let player_remaining = max_committable[player as usize] - contribs[player as usize];
        let player_street = per_street[player as usize];
        let facing = player_street < max_other_street;

        if printed < max_nodes_to_print {
            let kind = if player_remaining == 0 { "ALLIN" }
                       else if facing { "FACE" }
                       else { "OPEN" };
            eprintln!("{:>5} | {:>5} | {:>5} | {:?} | p{} | {:>3} {:5} | {:?}",
                idx, depth, "PLAY", contribs, player, player_remaining, kind, actions);
            printed += 1;
        }

        for i in 0..nc {
            let c_idx = tree.children[n.children_start as usize + i] as usize;
            let c = &tree.nodes[c_idx];
            let new_mask = if c.action_label == ACTION_LABEL_FOLD {
                folded_mask | (1u16 << player)
            } else { folded_mask };
            stack.push((c_idx, round_start.clone(), new_mask, depth + 1));
        }
    }
    eprintln!();
}

#[test]
fn gate_self_validation_2p_simple() {
    // Smallest meaningful test: 2p heads-up, equal blinds, single bet PotRel(1.0)
    // No raises, no all-in threshold (large stacks).
    //
    // EXPECTED tree shape (hand-enumerated):
    //   - Root: p0 acts, not facing bet (commits equal). Legal: {CHECK, BET}
    //   - After p0 CHECK: p1 acts, not facing bet. Legal: {CHECK, BET}
    //     - After p1 CHECK: round complete → chance turn
    //     - After p1 BET: p0 acts, facing bet. Legal: {FOLD, CALL}
    //   - After p0 BET: p1 acts, facing bet. Legal: {FOLD, CALL}
    //
    // Each terminal (FOLD or river chance subtree end) must have contribs
    // ≤ max_committable for each player.
    let cfg = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100; 2],
        initial_contributions: vec![5, 5],  // equal — simplest case
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 0.0, force_allin_threshold: 0.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&cfg).unwrap();
    let np = cfg.num_players as usize;
    let max_committable: Vec<i32> = cfg.starting_stacks.iter()
        .zip(&cfg.initial_contributions)
        .map(|(s, c)| s + c).collect();

    eprintln!("\n=== Gate self-validation: 2p equal blinds, hand-checkable ===");
    eprintln!("Tree: {} nodes, max_committable = {:?}", tree.num_nodes(), max_committable);
    eprintln!();

    // Dump the first ~50 nodes for hand-verification
    dump_player_nodes(&tree, np, &max_committable, 50);

    // Now run the GATE logic (replicated here from tree_correctness_gate.rs)
    let initial_snapshot: Vec<i32> = (0..np)
        .map(|p| tree.get_contribution(0, p as u8))
        .collect();
    let mut stack: Vec<(usize, Vec<i32>, u16)> = vec![(0, initial_snapshot, 0)];

    let mut violation_count = 0;
    let mut violations_by_kind: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
    let mut detail: Vec<(usize, &'static str, String)> = Vec::new();

    while let Some((idx, round_start, folded_mask)) = stack.pop() {
        let n = &tree.nodes[idx];
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        if n.is_chance() {
            let new_snapshot: Vec<i32> = contribs.clone();
            for i in 0..n.num_children as usize {
                let c = tree.children[n.children_start as usize + i] as usize;
                stack.push((c, new_snapshot.clone(), folded_mask));
            }
            continue;
        }
        if n.is_terminal() {
            for p in 0..np {
                if contribs[p] > max_committable[p] {
                    violation_count += 1;
                    *violations_by_kind.entry("contrib_over_stack").or_insert(0) += 1;
                    if detail.len() < 20 {
                        detail.push((idx, "contrib_over_stack",
                            format!("p{} contrib={} > max={}", p, contribs[p], max_committable[p])));
                    }
                }
            }
            continue;
        }
        let player = n.player_id;
        let nc = n.num_children as usize;
        if nc == 0 {
            violation_count += 1;
            *violations_by_kind.entry("empty_player").or_insert(0) += 1;
            if detail.len() < 20 {
                detail.push((idx, "empty_player", format!("player p{}", player)));
            }
            continue;
        }
        if (folded_mask & (1u16 << player)) != 0 {
            violation_count += 1;
            *violations_by_kind.entry("folded_acts").or_insert(0) += 1;
            if detail.len() < 20 {
                detail.push((idx, "folded_acts", format!("player p{} in folded_mask={:#b}", player, folded_mask)));
            }
        }
        let per_street: Vec<i32> = (0..np).map(|p| contribs[p] - round_start[p]).collect();
        let max_other: i32 = (0..np)
            .filter(|&p| p != player as usize && (folded_mask & (1u16 << p)) == 0)
            .map(|p| per_street[p])
            .max().unwrap_or(0);
        let player_remaining = max_committable[player as usize] - contribs[player as usize];
        let player_street = per_street[player as usize];
        let facing_bet = player_street < max_other;
        let actions: Vec<u8> = (0..nc)
            .map(|i| tree.nodes[tree.children[n.children_start as usize + i] as usize].action_label)
            .collect();
        let has = |l: u8| actions.contains(&l);

        if player_remaining == 0 {
            if !(actions.len() == 1 && actions[0] == ACTION_LABEL_CHECK) {
                violation_count += 1;
                *violations_by_kind.entry("allin_not_check").or_insert(0) += 1;
                if detail.len() < 20 {
                    let names: Vec<&str> = actions.iter().map(|&a| label_name(a)).collect();
                    detail.push((idx, "allin_not_check",
                        format!("p{} allin, actions={:?}", player, names)));
                }
            }
        } else if facing_bet {
            if !has(ACTION_LABEL_FOLD) {
                violation_count += 1;
                *violations_by_kind.entry("face_missing_fold").or_insert(0) += 1;
                if detail.len() < 20 {
                    let names: Vec<&str> = actions.iter().map(|&a| label_name(a)).collect();
                    detail.push((idx, "face_missing_fold",
                        format!("p{} facing, actions={:?}, contribs={:?}, per_street={:?}",
                            player, names, contribs, per_street)));
                }
            }
            if !has(ACTION_LABEL_CALL) && !has(ACTION_LABEL_ALLIN) {
                violation_count += 1;
                *violations_by_kind.entry("face_missing_call").or_insert(0) += 1;
                if detail.len() < 20 {
                    let names: Vec<&str> = actions.iter().map(|&a| label_name(a)).collect();
                    detail.push((idx, "face_missing_call",
                        format!("p{} facing, actions={:?}, contribs={:?}, per_street={:?}",
                            player, names, contribs, per_street)));
                }
            }
        } else {
            if !has(ACTION_LABEL_CHECK) {
                violation_count += 1;
                *violations_by_kind.entry("open_missing_check").or_insert(0) += 1;
                if detail.len() < 20 {
                    let names: Vec<&str> = actions.iter().map(|&a| label_name(a)).collect();
                    detail.push((idx, "open_missing_check",
                        format!("p{} open, actions={:?}, contribs={:?}, per_street={:?}",
                            player, names, contribs, per_street)));
                }
            }
            if has(ACTION_LABEL_FOLD) {
                violation_count += 1;
                *violations_by_kind.entry("open_has_fold").or_insert(0) += 1;
                if detail.len() < 20 {
                    let names: Vec<&str> = actions.iter().map(|&a| label_name(a)).collect();
                    detail.push((idx, "open_has_fold",
                        format!("p{} open, actions={:?}, contribs={:?}, per_street={:?}",
                            player, names, contribs, per_street)));
                }
            }
        }
        for i in 0..nc {
            let c_idx = tree.children[n.children_start as usize + i] as usize;
            let c = &tree.nodes[c_idx];
            let new_mask = if c.action_label == ACTION_LABEL_FOLD {
                folded_mask | (1u16 << player)
            } else { folded_mask };
            stack.push((c_idx, round_start.clone(), new_mask));
        }
    }

    eprintln!();
    eprintln!("=== Gate verdict on 2p simple tree ===");
    eprintln!("Total violations: {}", violation_count);
    if violation_count == 0 {
        eprintln!("✓ Gate finds zero violations.");
        eprintln!("  Hand-verify above dump: do the action sets match poker rules?");
        eprintln!("  If yes → gate logic trustworthy on simple cases.");
        eprintln!("  If no → gate is producing false negatives; needs review.");
    } else {
        eprintln!("Violations by kind:");
        for (k, n) in &violations_by_kind {
            eprintln!("  {}: {}", k, n);
        }
        eprintln!();
        eprintln!("Sample (first 20):");
        for (idx, kind, msg) in &detail {
            eprintln!("  node[{}] {}: {}", idx, kind, msg);
        }
        eprintln!();
        eprintln!("Determining: is this a tree-builder bug or a gate bug?");
        eprintln!("  Cross-reference each violation against the hand-verified dump above.");
        eprintln!("  If the listed violations actually violate poker rules → tree-builder bug.");
        eprintln!("  If they match poker rules → gate has false-positives, fix gate.");
    }
}
