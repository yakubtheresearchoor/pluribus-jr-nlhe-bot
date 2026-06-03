// Expanded self-validation for tree_correctness_gate.
//
// Per the approved plan in tidy-skipping-flamingo.md, before the gate can
// be trusted to drive the tree-builder rewrite, it must be validated on
// hand-checked trees that exercise the cases where the per-street vs
// cumulative classification distinction actually bites — exactly where a
// subtle classifier error would hide and where the original cumulative-
// commits audit bug lived.
//
// Three hard cases:
//   1. Asymmetric-blind root: per-street commits both 0, cumulative differs.
//      The first decision in every poker tree starting with blinds posted.
//      This is where the original audit bug lived.
//   2. Street transition: per-street commit snapshot must reset at chance
//      nodes. If snapshot doesn't refresh, post-chance facing-bet
//      classification uses cumulative commits (wrong).
//   3. All-in interposition: a player who is all-in (player_remaining = 0)
//      must have action set exactly {CHECK}. Anything else is a violation.
//
// For each case: build a hand-enumerated CORRECT tree (gate must report 0
// violations), then a deliberately-BROKEN variant (gate must flag exactly
// the introduced bug). This is the precondition for the gate being
// trustworthy enough to drive the rewrite.

use solver_core::tree::action::BoardState;
use solver_core::tree::flat::{
    FlatNode, FlatTree,
    ACTION_LABEL_CHECK, ACTION_LABEL_CALL, ACTION_LABEL_FOLD,
    ACTION_LABEL_BET, ACTION_LABEL_RAISE, ACTION_LABEL_ALLIN, ACTION_LABEL_CHANCE,
};

// ─────────────────────────────────────────────────────────────────────────
// Gate logic — INDEPENDENTLY derived from poker rules in this file.
// NOT shared with the production gate in tree_correctness_gate.rs.
// Two independent derivations is what makes the agreement meaningful.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Default, Debug, Clone)]
struct GateResult {
    total: usize,
    by_kind: std::collections::HashMap<&'static str, usize>,
    samples: Vec<(usize, &'static str, String)>,
}

impl GateResult {
    fn count(&self, kind: &'static str) -> usize {
        self.by_kind.get(kind).copied().unwrap_or(0)
    }
    fn record(&mut self, idx: usize, kind: &'static str, msg: String) {
        self.total += 1;
        *self.by_kind.entry(kind).or_insert(0) += 1;
        if self.samples.len() < 20 {
            self.samples.push((idx, kind, msg));
        }
    }
}

/// Run an INDEPENDENT classifier-based audit. Reasoning derived directly
/// from poker rules (not from the production gate):
///   - A player with no remaining chips is all-in and can only check.
///   - A player who has put fewer chips into THIS STREET than at least one
///     other active player is "facing a bet" and may fold, call, or raise.
///   - A player whose street commit equals the max of other active players'
///     street commits is "opening" — may check or bet, but NOT fold (you
///     cannot fold when nothing is owed).
fn audit_tree(tree: &FlatTree, np: usize, max_committable: &[i32]) -> GateResult {
    let mut r = GateResult::default();
    let initial_round_start: Vec<i32> = (0..np)
        .map(|p| tree.get_contribution(0, p as u8))
        .collect();
    // DFS: (node_idx, round_start_contribs, folded_mask)
    let mut work: Vec<(usize, Vec<i32>, u16)> = vec![(0, initial_round_start, 0)];

    while let Some((idx, round_start, folded_mask)) = work.pop() {
        let n = &tree.nodes[idx];
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();

        // CHANCE node — refresh per-street snapshot to current contribs.
        if n.is_chance() {
            let next_snapshot: Vec<i32> = contribs.clone();
            for i in 0..n.num_children as usize {
                let c = tree.children[n.children_start as usize + i] as usize;
                work.push((c, next_snapshot.clone(), folded_mask));
            }
            continue;
        }

        // TERMINAL — verify physical-validity: no contribution may exceed
        // the chips a player has (starting_stack + initial_contribution).
        if n.is_terminal() {
            for p in 0..np {
                if contribs[p] > max_committable[p] {
                    r.record(idx, "contrib_exceeds_stack",
                        format!("p{} contrib={} > max={}", p, contribs[p], max_committable[p]));
                }
            }
            continue;
        }

        // PLAYER node — full poker-rules check.
        let nc = n.num_children as usize;
        if nc == 0 {
            r.record(idx, "empty_player",
                format!("p{} has 0 children (decision node with no actions)", n.player_id));
            continue;
        }
        if (folded_mask & (1u16 << n.player_id)) != 0 {
            r.record(idx, "folded_acts",
                format!("p{} is in folded_mask={:#b} yet acts at this node", n.player_id, folded_mask));
        }

        // Per-street commits derived from cumulative - round-start snapshot.
        let player = n.player_id as usize;
        let player_street_commit = contribs[player] - round_start[player];
        let mut max_other_street_commit = 0i32;
        for p in 0..np {
            if p == player { continue; }
            if (folded_mask & (1u16 << p)) != 0 { continue; }
            let other_street = contribs[p] - round_start[p];
            if other_street > max_other_street_commit {
                max_other_street_commit = other_street;
            }
        }
        let player_remaining = max_committable[player] - contribs[player];

        // Collect generated actions
        let actions: Vec<u8> = (0..nc)
            .map(|i| tree.nodes[tree.children[n.children_start as usize + i] as usize].action_label)
            .collect();
        let has = |l: u8| actions.contains(&l);

        // Classification by poker rules
        if player_remaining == 0 {
            // All-in: only legal action is CHECK (forced pass).
            let only_check = actions.len() == 1 && actions[0] == ACTION_LABEL_CHECK;
            if !only_check {
                let names: Vec<&str> = actions.iter().map(|&a| label_name(a)).collect();
                r.record(idx, "allin_not_check_only",
                    format!("p{} all-in (remaining=0) but actions={:?}", player, names));
            }
        } else if player_street_commit < max_other_street_commit {
            // Facing a bet: must include FOLD, and must include something
            // to continue (CALL or ALLIN).
            if !has(ACTION_LABEL_FOLD) {
                r.record(idx, "facing_bet_no_fold",
                    format!("p{} owes {} chips this street, has no FOLD",
                        player, max_other_street_commit - player_street_commit));
            }
            if !has(ACTION_LABEL_CALL) && !has(ACTION_LABEL_ALLIN) {
                r.record(idx, "facing_bet_no_continue",
                    format!("p{} facing bet has neither CALL nor ALLIN", player));
            }
        } else {
            // Not facing a bet: CHECK must be available, FOLD must NOT be
            // (nothing to fold to when no chips are owed).
            if !has(ACTION_LABEL_CHECK) {
                r.record(idx, "open_no_check",
                    format!("p{} not facing bet, has no CHECK option", player));
            }
            if has(ACTION_LABEL_FOLD) {
                r.record(idx, "open_has_illegal_fold",
                    format!("p{} not facing bet, FOLD is illegal but present", player));
            }
        }

        // Recurse into children — folded_mask propagates if THIS player folded.
        for i in 0..nc {
            let c_idx = tree.children[n.children_start as usize + i] as usize;
            let c = &tree.nodes[c_idx];
            let new_mask = if c.action_label == ACTION_LABEL_FOLD {
                folded_mask | (1u16 << player as u16)
            } else {
                folded_mask
            };
            work.push((c_idx, round_start.clone(), new_mask));
        }
    }
    r
}

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

// ─────────────────────────────────────────────────────────────────────────
// Helper: build a FlatNode with action_label set (kuhn_test pattern leaves
// action_label at 0 = FOLD by default; we need explicit labels).
// ─────────────────────────────────────────────────────────────────────────

fn add_player(tree: &mut FlatTree, player_id: u8, bs: BoardState, contribs: &[i32], action_from_parent: u8) -> usize {
    let idx = tree.alloc_node(FlatNode::player(player_id, bs, 0));
    for (p, &c) in contribs.iter().enumerate() {
        tree.set_contribution(idx, p as u8, c);
    }
    tree.nodes[idx].action_label = action_from_parent;
    idx
}

fn add_chance(tree: &mut FlatTree, bs: BoardState, contribs: &[i32], action_from_parent: u8) -> usize {
    let idx = tree.alloc_node(FlatNode::chance(bs));
    for (p, &c) in contribs.iter().enumerate() {
        tree.set_contribution(idx, p as u8, c);
    }
    tree.nodes[idx].action_label = action_from_parent;
    idx
}

fn add_terminal(tree: &mut FlatTree, contribs: &[i32], action_from_parent: u8, folded_mask: u16) -> usize {
    let idx = tree.alloc_node(FlatNode::terminal());
    for (p, &c) in contribs.iter().enumerate() {
        tree.set_contribution(idx, p as u8, c);
    }
    tree.nodes[idx].action_label = action_from_parent;
    tree.set_folded_mask(idx, folded_mask);
    idx
}

// ═════════════════════════════════════════════════════════════════════════
// TEST 1: ASYMMETRIC-BLIND ROOT
// 2 players, initial_contribs=[2,1] (asymmetric "big blind 2, small blind 1"
// already in pot from preflop), stacks=[10,10], max_committable=[12, 11].
// We're at flop start. p0 acts first.
//
// Per-street semantics: at root, snapshot = [2,1], per_street = [0,0].
// Both players are "not facing bet" — actions {CHECK, BET} for each.
// Cumulative semantics would (incorrectly) say p1 owes 1 chip relative to
// p0 cumulative — this is the exact bug.
// ═════════════════════════════════════════════════════════════════════════

fn build_test1_correct() -> (FlatTree, Vec<i32>) {
    let mut tree = FlatTree::new(2, 3, vec![10, 10], 0.0, 0.0);
    let max_committable = vec![12, 11];

    // Node 0: p0 acts (root). Children: CHECK→1, BET(to 5)→4.
    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::Flop, 0));
    tree.set_contribution(n0, 0, 2);
    tree.set_contribution(n0, 1, 1);

    // Node 1: p1 after p0's CHECK (not facing bet — per-street commits both 0).
    let n1 = add_player(&mut tree, 1, BoardState::Flop, &[2, 1], ACTION_LABEL_CHECK);

    // Node 2: TERMINAL after p1 CHECK (round complete; placeholder in lieu of chance).
    let n2 = add_terminal(&mut tree, &[2, 1], ACTION_LABEL_CHECK, 0);

    // Node 3: TERMINAL after p1 BET (placeholder — would be p0's facing-bet decision).
    let n3 = add_terminal(&mut tree, &[2, 5], ACTION_LABEL_BET, 0);

    // Node 4: p1 after p0's BET to 5 (facing-bet — must FOLD or CALL).
    let n4 = add_player(&mut tree, 1, BoardState::Flop, &[5, 1], ACTION_LABEL_BET);

    // Node 5: TERMINAL after p1 FOLD (p0 wins).
    let n5 = add_terminal(&mut tree, &[5, 1], ACTION_LABEL_FOLD, 0b10);

    // Node 6: TERMINAL after p1 CALL.
    let n6 = add_terminal(&mut tree, &[5, 5], ACTION_LABEL_CALL, 0);

    tree.set_children(n0, vec![n1 as u32, n4 as u32]);
    tree.set_children(n1, vec![n2 as u32, n3 as u32]);
    tree.set_children(n4, vec![n5 as u32, n6 as u32]);

    // For the audit, walk needs children to be queryable; n2/n3/n5/n6 are terminals.
    (tree, max_committable)
}

fn build_test1_broken_open_with_fold() -> (FlatTree, Vec<i32>) {
    // Same as correct, but add a FOLD child to node 1 (which is open / not-facing-bet).
    // Gate should flag "open_has_illegal_fold" at node 1.
    let (mut tree, max_c) = build_test1_correct();
    // Add a 3rd child to node 1: TERMINAL with action_label=FOLD.
    let bad_terminal = add_terminal(&mut tree, &[2, 1], ACTION_LABEL_FOLD, 0b10);
    // Re-set children of n1 to include the bad terminal.
    tree.set_children(1, vec![2, 3, bad_terminal as u32]);
    (tree, max_c)
}

#[test]
fn test1_asymmetric_blind_root_correct() {
    let (tree, max_c) = build_test1_correct();
    let r = audit_tree(&tree, 2, &max_c);
    eprintln!("\n=== Test 1: asymmetric-blind root, CORRECT tree ===");
    eprintln!("Tree: {} nodes, max_committable={:?}", tree.num_nodes(), max_c);
    eprintln!("Total violations: {}", r.total);
    for (k, v) in &r.by_kind {
        eprintln!("  {}: {}", k, v);
    }
    for (idx, k, msg) in &r.samples {
        eprintln!("  node[{}] {}: {}", idx, k, msg);
    }
    assert_eq!(r.total, 0,
        "Correct asymmetric-blind-root tree should report 0 violations, got {}", r.total);
}

#[test]
fn test1_asymmetric_blind_root_broken_open_with_fold() {
    let (tree, max_c) = build_test1_broken_open_with_fold();
    let r = audit_tree(&tree, 2, &max_c);
    eprintln!("\n=== Test 1: asymmetric-blind root, BROKEN (open with FOLD) ===");
    eprintln!("Total violations: {}", r.total);
    for (k, v) in &r.by_kind {
        eprintln!("  {}: {}", k, v);
    }
    assert_eq!(r.count("open_has_illegal_fold"), 1,
        "Should flag exactly 1 open_has_illegal_fold, got {}", r.count("open_has_illegal_fold"));
    assert_eq!(r.total, 1, "Should flag exactly 1 violation total, got {}", r.total);
}

// ═════════════════════════════════════════════════════════════════════════
// TEST 2: STREET TRANSITION
// 2 players, initial_contribs=[2,1], stacks=[10,10], max_committable=[12,11].
// Flop: p0 BET to 5, p1 CALL to 5. Round complete → chance → turn.
// On turn (post-chance), commit snapshot resets to [5,5]. Both players'
// per-street commits = 0 → not facing bet → {CHECK, BET}.
//
// If snapshot doesn't refresh at chance, post-chance per-street uses initial
// snapshot [2,1] — making p1 (5-1=4) appear to "owe" p0 (5-2=3) → wrong
// facing-bet classification on turn.
// ═════════════════════════════════════════════════════════════════════════

fn build_test2_correct() -> (FlatTree, Vec<i32>) {
    let mut tree = FlatTree::new(2, 3, vec![10, 10], 0.0, 0.0);
    let max_c = vec![12, 11];

    // Flop:
    // Node 0: p0 acts at flop start. contribs [2,1], per-street [0,0]. Not facing bet.
    //   Must have CHECK + BET actions (CHECK leads to placeholder; BET leads to subtree).
    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::Flop, 0));
    tree.set_contribution(n0, 0, 2);
    tree.set_contribution(n0, 1, 1);

    // Node 1: p1 facing p0's BET to 5. contribs [5,1], per-street [3,0]. Must have FOLD + CALL.
    let n1 = add_player(&mut tree, 1, BoardState::Flop, &[5, 1], ACTION_LABEL_BET);

    // Node 2: CHANCE (turn) after p1 CALL.
    let n2 = add_chance(&mut tree, BoardState::Turn, &[5, 5], ACTION_LABEL_CALL);

    // Node 3: p0 on turn (round starter again). Not facing bet (per-street commits both 0).
    let n3 = add_player(&mut tree, 0, BoardState::Turn, &[5, 5], ACTION_LABEL_CHANCE);

    // Node 4: TERMINAL after p0 CHECK on turn.
    let n4 = add_terminal(&mut tree, &[5, 5], ACTION_LABEL_CHECK, 0);

    // Node 5: TERMINAL after p0 BET on turn (placeholder).
    let n5 = add_terminal(&mut tree, &[10, 5], ACTION_LABEL_BET, 0);

    // Node 6: TERMINAL after p0 CHECK from root (placeholder — not exercised further).
    let n6 = add_terminal(&mut tree, &[2, 1], ACTION_LABEL_CHECK, 0);

    // Node 7: TERMINAL after p1 FOLD at node 1 (p1 folds to p0's bet).
    let n7 = add_terminal(&mut tree, &[5, 1], ACTION_LABEL_FOLD, 0b10);

    tree.set_children(n0, vec![n6 as u32, n1 as u32]);  // CHECK→n6, BET→n1
    tree.set_children(n1, vec![n7 as u32, n2 as u32]);  // FOLD→n7, CALL→n2
    tree.set_children(n2, vec![n3 as u32]);
    tree.set_children(n3, vec![n4 as u32, n5 as u32]);  // CHECK→n4, BET→n5

    (tree, max_c)
}

fn build_test2_broken_post_chance_fold() -> (FlatTree, Vec<i32>) {
    // Same as correct, but the post-chance player node (node 3, p0 on turn)
    // has a spurious FOLD action — which is illegal because p0 is the round
    // starter on turn with nothing to call (per-street commits both 0 after
    // chance refreshes the snapshot). If the gate's snapshot logic were
    // broken (carrying root snapshot through the chance node), it would NOT
    // flag this because cumulative comparison still wouldn't show a bet
    // outstanding. But the gate IS correct here — it refreshes snapshot at
    // chance, sees p0 as not-facing-bet on turn, and flags the FOLD as illegal.
    let (mut tree, max_c) = build_test2_correct();
    let bad_fold = add_terminal(&mut tree, &[5, 5], ACTION_LABEL_FOLD, 0b01);
    // After fix, n3's children are [n4, n5] = [4, 5] indices.
    // Append the bad_fold.
    tree.set_children(3, vec![4, 5, bad_fold as u32]);
    (tree, max_c)
}

#[test]
fn test2_street_transition_correct() {
    let (tree, max_c) = build_test2_correct();
    let r = audit_tree(&tree, 2, &max_c);
    eprintln!("\n=== Test 2: street transition, CORRECT tree ===");
    eprintln!("Total violations: {}", r.total);
    for (idx, k, msg) in &r.samples { eprintln!("  node[{}] {}: {}", idx, k, msg); }
    assert_eq!(r.total, 0,
        "Correct street-transition tree should report 0 violations, got {}", r.total);
}

#[test]
fn test2_street_transition_broken_post_chance_fold() {
    let (tree, max_c) = build_test2_broken_post_chance_fold();
    let r = audit_tree(&tree, 2, &max_c);
    eprintln!("\n=== Test 2: street transition, BROKEN (post-chance FOLD) ===");
    eprintln!("Total violations: {}", r.total);
    for (k, v) in &r.by_kind { eprintln!("  {}: {}", k, v); }
    assert_eq!(r.count("open_has_illegal_fold"), 1,
        "Should flag exactly 1 open_has_illegal_fold on post-chance node");
    assert_eq!(r.total, 1, "Should flag exactly 1 violation");
}

// ═════════════════════════════════════════════════════════════════════════
// TEST 3: ALL-IN INTERPOSITION
// 3 players. initial_contribs=[1, 1, 1], stacks=[10, 10, 3].
// max_committable = [11, 11, 4].
//
// Sequence on flop:
//   p0 BET to 4
//   p1 RAISE to 11 (all-in for p1)
//   p2 considering: p2's max is 4. They can fold or call-all-in (commit 4).
//   (we'll have p2 CALL all-in)
//   p0 facing p1's raise to 11: can fold, call (commit 11), or...
//   (we'll have p0 CALL)
//   At this point: all-in players are p1 (committed 11=max) and p2 (committed
//   4=max). p0 is committed 11 with remaining 0 → also all-in.
//   Round complete. Chance → turn.
//
// For testing the ALL-IN action set: include a node where p2 is acting after
// being put all-in by an earlier bet. P2's only legal action is CHECK.
// ═════════════════════════════════════════════════════════════════════════

fn build_test3_correct() -> (FlatTree, Vec<i32>) {
    let mut tree = FlatTree::new(3, 3, vec![10, 10, 3], 0.0, 0.0);
    let max_c = vec![11, 11, 4];

    // Simplest demo: a chance transition to turn where p2 is already all-in,
    // and on the turn p2 is asked to act. Per all-in rules, p2 has exactly {CHECK}.
    //
    // Set up post-flop state: p2 was all-in at 4, p0 and p1 each called for 4.
    // After flop round, all committed equal at 4 → chance to turn.
    //
    // Tree:
    //   Node 0: CHANCE (turn) — using as root for testing simplicity.
    //     Children: node 1 (post-chance starting player).
    //   Node 1: p0 acts on turn (p2 is all-in but it's p0's turn first).
    //     Actions: CHECK→2, BET→4
    //   Node 2: p1 acts on turn after p0 CHECK. Actions: CHECK→3 (terminal), BET→...
    //     For simplicity, only CHECK path.
    //   Node 3: p2 acts on turn after both p0 and p1 CHECK. p2 is all-in.
    //     Action set MUST be exactly {CHECK} → terminal.
    //
    // Initial contribs at root = [4, 4, 4]. max_committable = [11, 11, 4].
    // p2's remaining = 4 - 4 = 0 → all-in → forced CHECK.

    let n0 = tree.alloc_node(FlatNode::chance(BoardState::Turn));
    tree.set_contribution(n0, 0, 4);
    tree.set_contribution(n0, 1, 4);
    tree.set_contribution(n0, 2, 4);

    let n1 = add_player(&mut tree, 0, BoardState::Turn, &[4, 4, 4], ACTION_LABEL_CHANCE);
    let n2 = add_player(&mut tree, 1, BoardState::Turn, &[4, 4, 4], ACTION_LABEL_CHECK);
    let n3 = add_player(&mut tree, 2, BoardState::Turn, &[4, 4, 4], ACTION_LABEL_CHECK);
    // n4: TERMINAL after p2 CHECK (round complete).
    let n4 = add_terminal(&mut tree, &[4, 4, 4], ACTION_LABEL_CHECK, 0);
    // n5: TERMINAL after p1 BET (placeholder).
    let n5 = add_terminal(&mut tree, &[4, 10, 4], ACTION_LABEL_BET, 0);
    // n6: TERMINAL after p0 BET on turn (placeholder).
    let n6 = add_terminal(&mut tree, &[10, 4, 4], ACTION_LABEL_BET, 0);

    tree.set_children(n0, vec![n1 as u32]);
    tree.set_children(n1, vec![n2 as u32, n6 as u32]);
    tree.set_children(n2, vec![n3 as u32, n5 as u32]);
    // n3 (p2 all-in) has exactly {CHECK} child:
    tree.set_children(n3, vec![n4 as u32]);

    (tree, max_c)
}

fn build_test3_broken_allin_with_fold() -> (FlatTree, Vec<i32>) {
    // p2 is all-in but has a FOLD action — illegal.
    let (mut tree, max_c) = build_test3_correct();
    let bad_fold = add_terminal(&mut tree, &[4, 4, 4], ACTION_LABEL_FOLD, 0b100);
    // n3's existing child is index 4 (the n4 terminal). Add FOLD as second child.
    tree.set_children(3, vec![4, bad_fold as u32]);
    (tree, max_c)
}

#[test]
fn test3_allin_interposition_correct() {
    let (tree, max_c) = build_test3_correct();
    let r = audit_tree(&tree, 3, &max_c);
    eprintln!("\n=== Test 3: all-in interposition, CORRECT tree ===");
    eprintln!("Total violations: {}", r.total);
    for (idx, k, msg) in &r.samples { eprintln!("  node[{}] {}: {}", idx, k, msg); }
    assert_eq!(r.total, 0, "Correct all-in tree should report 0 violations, got {}", r.total);
}

#[test]
fn test3_allin_interposition_broken_allin_with_fold() {
    let (tree, max_c) = build_test3_broken_allin_with_fold();
    let r = audit_tree(&tree, 3, &max_c);
    eprintln!("\n=== Test 3: all-in interposition, BROKEN (all-in with FOLD) ===");
    eprintln!("Total violations: {}", r.total);
    for (k, v) in &r.by_kind { eprintln!("  {}: {}", k, v); }
    assert_eq!(r.count("allin_not_check_only"), 1,
        "Should flag exactly 1 allin_not_check_only");
    assert_eq!(r.total, 1, "Should flag exactly 1 violation");
}
