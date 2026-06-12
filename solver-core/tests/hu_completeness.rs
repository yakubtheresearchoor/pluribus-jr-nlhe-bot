// HU completeness check (responds to the standing concern that gate-pass
// proves only that no ILLEGAL nodes exist, not that all LEGAL lines are
// present). The rewrite collapsed the 6p asymmetric tree from 14.5M nodes
// → 2,431 nodes (~5,900× reduction). That ratio is large enough to warrant
// suspicion that we swung from "too-many states" (phantom nodes the gate
// would have flagged) to "too-few states" (legal subtrees missing entirely,
// which the gate's per-node-rules check would NOT detect).
//
// HU is the natural minimal test case:
//   - HU postflop action order is REVERSED from multiway (BB at index 0
//     acts first OOP, button at index 1 acts second IP), and the rewrite's
//     player-advancement logic was built/tested only against multiway
//     configs (6p asymmetric, 6p symmetric, 3p asymmetric). "It'll work
//     fine for HU" was untested.
//   - Small (~85–117 nodes) so we can hand-trace specific lines.
//
// The gate covers per-PLAYER-node rules. It does NOT cover:
//   (a) wrong action ORDER: which player_id acts at a given state
//   (b) missing legal LINES: subtrees that should exist below an action
//   (c) wrong NODE TYPE: TERMINAL where PLAYER should be, vice versa
//   (d) missing/extra CHANCE nodes (street advance correctness)
//
// This file addresses (a)–(d):
//   Check 1  — Action order: at every street-start PLAYER node, player_id == 0
//              (HU postflop convention: BB acts first OOP).
//   Check 2  — Sample-path completeness: a handful of specific legal HU
//              lines (check-down through 3 streets; bet-call; bet-fold;
//              bet-allin-call; check-bet on later street) must be
//              navigable from the root, with the actor's cumulative
//              contribution after each action matching the C1 total.
//   Check 3  — Node-type classification: every CHANCE node represents a
//              completed round at street ≤ river, and every TERMINAL node
//              represents either a fold or the end of the river.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{
    FlatTree, ACTION_LABEL_ALLIN, ACTION_LABEL_BET, ACTION_LABEL_CALL,
    ACTION_LABEL_CHECK, ACTION_LABEL_FOLD,
};

fn hu_symmetric_cfg() -> TreeConfig {
    TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        // 2026-06-12 semantics: starting_pot is dead money ADDITIVE with
        // contributions; this legacy config double-counted (pot 10 = [5,5]).
        starting_pot: 0,
        starting_stacks: vec![100; 2],
        initial_contributions: vec![5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,
    }

}

fn hu_asymmetric_cfg() -> TreeConfig {
    TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 0, // was 3: double-counted [2,1] (2026-06-12 semantics)
        starting_stacks: vec![100; 2],
        initial_contributions: vec![2, 1],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,
    }

}

// ============================================================================
// Scope note: this file's checks are FLOP-START only. The BoardState enum
// (src/tree/action.rs) has only {Flop, Turn, River} — there is no Preflop
// variant, so no tree this builder produces contains preflop structure.
// Preflop ranges enter the solver as inputs, not as a betting-tree prefix.
//
// IMPORTANT (flagged for the deferred preflop workstream): HU has a within-
// hand action-order reversal. POSTflop, BB (OOP) acts first — verified
// below at every street-start. PREflop, the button (= the small blind, in
// HU) acts first. These are OPPOSITE conventions. The postflop check is
// the complete check for this tree, but when the preflop workstream lands,
// the preflop root must verify that the button is to_act, not the BB. The
// postflop-order machinery in this file CANNOT be reused for that — it
// would assert the wrong player.
//
// ============================================================================
// Check 1 — HU action order (postflop street-starts)
// ============================================================================

#[test]
fn hu_action_order_symmetric() {
    check_hu_action_order(&hu_symmetric_cfg(), "HU symmetric [5,5]");
}

#[test]
fn hu_action_order_asymmetric() {
    check_hu_action_order(&hu_asymmetric_cfg(), "HU asymmetric [2,1]");
}

fn check_hu_action_order(cfg: &TreeConfig, label: &str) {
    let tree = build_tree(cfg).unwrap();
    eprintln!("\n=== HU action order: {} ({} nodes) ===", label, tree.nodes.len());

    // Root must be a PLAYER node owned by p0 (BB, OOP postflop).
    assert!(
        tree.nodes[0].is_player(),
        "[{}] root must be PLAYER (got node_type={})",
        label,
        tree.nodes[0].node_type
    );
    assert_eq!(
        tree.nodes[0].player_id, 0,
        "[{}] root player must be 0 (BB acts first OOP postflop in HU); \
         got player {}. The rewrite uses multiway action order; HU \
         reverses it.",
        label, tree.nodes[0].player_id
    );

    // Every CHANCE node represents a street advance. Its sole PLAYER
    // successor must be owned by p0 (the still-active leftmost player,
    // which in HU is always BB at index 0 unless folded — but you can't
    // street-advance after a fold, so it's always p0 here).
    let mut chance_count = 0;
    let mut street_starts_to_p0 = 0;
    for (i, n) in tree.nodes.iter().enumerate() {
        if !n.is_chance() {
            continue;
        }
        chance_count += 1;
        let nc = n.num_children as usize;
        assert!(
            nc >= 1,
            "[{}] CHANCE node[{}] has {} children",
            label, i, nc
        );
        for j in 0..nc {
            let child_idx = tree.children[n.children_start as usize + j] as usize;
            let child = &tree.nodes[child_idx];
            if child.is_player() {
                street_starts_to_p0 += 1;
                assert_eq!(
                    child.player_id, 0,
                    "[{}] CHANCE node[{}] → PLAYER child[{}] has player_id={} \
                     (HU street-start must be p0; this would be MULTIWAY order \
                     incorrectly applied to HU)",
                    label, i, child_idx, child.player_id
                );
            }
        }
    }
    eprintln!(
        "  {} CHANCE nodes; {} street-start PLAYER children; all owned by p0 ✓",
        chance_count, street_starts_to_p0
    );
    assert!(
        street_starts_to_p0 > 0,
        "[{}] no PLAYER children found below CHANCE nodes — tree may have no betting at all",
        label
    );
}

// ============================================================================
// Check 2 — Sample-path completeness
// ============================================================================

/// One step on a path: descend through a child of the current node.
#[derive(Debug, Clone)]
enum Step {
    /// Pick the player child with this action_label; after the descent,
    /// the child's cumulative contribution for `actor` (= parent's
    /// player_id) must equal `expected_total`.
    Action {
        label: u8,
        expected_total: Option<i32>,
    },
    /// Descend through a CHANCE node (street advance). CHANCE nodes have
    /// a single child here because the builder collapses chance to one
    /// representative outcome at structure level.
    StreetAdvance,
}

fn follow_path(tree: &FlatTree, np: u8, path: &[Step], label: &str) -> usize {
    let mut idx = 0usize;
    for (i, step) in path.iter().enumerate() {
        let n = &tree.nodes[idx];
        match step {
            Step::Action {
                label: want_label,
                expected_total,
            } => {
                assert!(
                    n.is_player(),
                    "[{}] step {} expected PLAYER node at idx {}, got node_type={} action_label={}",
                    label, i, idx, n.node_type, n.action_label
                );
                let actor = n.player_id;
                let mut found: Option<usize> = None;
                for j in 0..n.num_children as usize {
                    let c = tree.children[n.children_start as usize + j] as usize;
                    let cn = &tree.nodes[c];
                    if cn.action_label == *want_label {
                        // For BET/RAISE/CALL/ALLIN, also check the actor's contribution
                        // on the child matches expected_total if provided.
                        if let Some(want) = expected_total {
                            let got = tree.get_contribution(c, actor);
                            if got != *want {
                                continue; // try next match (shouldn't be multiple, but safe)
                            }
                        }
                        found = Some(c);
                        break;
                    }
                }
                idx = found.unwrap_or_else(|| {
                    let avail: Vec<(u8, Vec<i32>)> = (0..n.num_children as usize)
                        .map(|j| {
                            let c = tree.children[n.children_start as usize + j] as usize;
                            let cn = &tree.nodes[c];
                            let contribs: Vec<i32> =
                                (0..np).map(|p| tree.get_contribution(c, p)).collect();
                            (cn.action_label, contribs)
                        })
                        .collect();
                    panic!(
                        "[{}] step {} at PLAYER node[{}] actor=p{}: expected \
                         action_label={} expected_total={:?}, available children: \
                         {:?}. This legal HU line is MISSING from the tree \
                         (over-pruning).",
                        label, i, idx, actor, want_label, expected_total, avail
                    );
                });
            }
            Step::StreetAdvance => {
                assert!(
                    n.is_chance(),
                    "[{}] step {} expected CHANCE node at idx {}, got node_type={} \
                     player_id={} action_label={}",
                    label, i, idx, n.node_type, n.player_id, n.action_label
                );
                assert_eq!(
                    n.num_children, 1,
                    "[{}] step {} CHANCE node[{}] has {} children (expected 1)",
                    label, i, idx, n.num_children
                );
                idx = tree.children[n.children_start as usize] as usize;
            }
        }
    }
    idx
}

#[test]
fn hu_sample_path_completeness_symmetric() {
    let cfg = hu_symmetric_cfg();
    let tree = build_tree(&cfg).unwrap();
    let label = "HU symmetric [5,5]";
    eprintln!("\n=== Sample-path completeness: {} ===", label);

    // Bet sizing under PotRelative(1.0): total = prev_amount + pot*1.0.
    // At root: prev_amount = 5 (p1's commit), pot = 10. So a bet by p0
    // commits 5 + 10 = 15. The contribution stored on the resulting
    // child's slot for p0 is 15.
    //
    // After flop check-check, pot still = 10 at start of turn; p0's
    // commit = 5; bet of PotRel(1.0) adds 10 → total commit 15.

    // ---- Line A: 3 streets of pure check-check → showdown terminal ----
    eprintln!("Line A: check-down through 3 streets → terminal");
    let leaf = follow_path(
        &tree,
        2,
        &[
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None }, // p0 check
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None }, // p1 check
            Step::StreetAdvance,                                              // → TURN
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::StreetAdvance,                                              // → RIVER
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
        ],
        label,
    );
    assert!(
        tree.nodes[leaf].is_terminal(),
        "Line A end must be terminal; got node[{}] type={}",
        leaf, tree.nodes[leaf].node_type
    );

    // ---- Line B: flop check-bet-call, turn check-check, river check-check ----
    eprintln!("Line B: flop check / bet→15 / call→15, turn check-check, river check-check → terminal");
    let leaf = follow_path(
        &tree,
        2,
        &[
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },     // p0 check
            Step::Action { label: ACTION_LABEL_BET,   expected_total: Some(15) }, // p1 bet to 15
            Step::Action { label: ACTION_LABEL_CALL,  expected_total: Some(15) }, // p0 call → 15
            Step::StreetAdvance,
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::StreetAdvance,
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
        ],
        label,
    );
    assert!(tree.nodes[leaf].is_terminal(), "Line B end must be terminal");

    // ---- Line C: p0 bets flop, p1 folds (immediate terminal) ----
    eprintln!("Line C: p0 bet→15, p1 fold → terminal");
    let leaf = follow_path(
        &tree,
        2,
        &[
            Step::Action { label: ACTION_LABEL_BET,  expected_total: Some(15) },
            Step::Action { label: ACTION_LABEL_FOLD, expected_total: None },
        ],
        label,
    );
    assert!(tree.nodes[leaf].is_terminal(), "Line C fold leaf must be terminal");

    // ---- Line D: flop check-check, turn check-check, river bet-fold ----
    eprintln!("Line D: flop+turn check-check, river p0 bet→15 p1 fold → terminal");
    let leaf = follow_path(
        &tree,
        2,
        &[
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::StreetAdvance,
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::StreetAdvance,
            Step::Action { label: ACTION_LABEL_BET,  expected_total: Some(15) }, // p0 bets river
            Step::Action { label: ACTION_LABEL_FOLD, expected_total: None },     // p1 folds river
        ],
        label,
    );
    assert!(tree.nodes[leaf].is_terminal(), "Line D end must be terminal");

    // ---- Line E: flop check-check, turn bet-call, river check-check ----
    eprintln!("Line E: flop check-check, turn p0 bet→15 p1 call→15, river check-check → terminal");
    let leaf = follow_path(
        &tree,
        2,
        &[
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::StreetAdvance,
            Step::Action { label: ACTION_LABEL_BET,  expected_total: Some(15) }, // p0 bets turn
            Step::Action { label: ACTION_LABEL_CALL, expected_total: Some(15) }, // p1 calls
            Step::StreetAdvance,
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
        ],
        label,
    );
    assert!(tree.nodes[leaf].is_terminal(), "Line E end must be terminal");

    eprintln!("All 5 sample lines navigable ✓");
}

#[test]
fn hu_sample_path_completeness_asymmetric() {
    let cfg = hu_asymmetric_cfg();
    let tree = build_tree(&cfg).unwrap();
    let label = "HU asymmetric [2,1]";
    eprintln!("\n=== Sample-path completeness: {} ===", label);

    // HU asymmetric [2,1] starting flop: per-street-start = [2,1], current = [2,1].
    // Per-street commits this street = [0,0] → neither facing-bet → p0 acts first.
    // BET formula (builder add_bet_size_action C1): total = prev_amount + delta
    // where prev_amount = max OTHER committed (= 1, p1's contrib) and delta =
    // pot * PotRel(1.0). Pot = sum_commits = 3. delta = 3 → total = 1 + 3 = 4.
    // So p0's bet target is 4, NOT 5 (would only be 5 if formula used p0's own
    // commit of 2 + delta of 3 = 5, which is symmetric-case coincidence).

    // Line A: pure check-down through 3 streets.
    eprintln!("Line A: check-check across 3 streets → terminal");
    let leaf = follow_path(
        &tree,
        2,
        &[
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::StreetAdvance,
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::StreetAdvance,
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
        ],
        label,
    );
    assert!(tree.nodes[leaf].is_terminal());

    // Line B: p0 bet→4 (=prev_amount 1 + delta 3), p1 fold.
    eprintln!("Line B: p0 bet→4, p1 fold → terminal");
    let leaf = follow_path(
        &tree,
        2,
        &[
            Step::Action { label: ACTION_LABEL_BET,  expected_total: Some(4) },
            Step::Action { label: ACTION_LABEL_FOLD, expected_total: None },
        ],
        label,
    );
    assert!(tree.nodes[leaf].is_terminal());

    // Line C: p0 bet→4, p1 call→4, [turn], check-check, [river], check-check.
    // After p1 calls 4: both at 4. Turn pot = 8.
    eprintln!("Line C: bet-call flop, check-check turn+river → terminal");
    let leaf = follow_path(
        &tree,
        2,
        &[
            Step::Action { label: ACTION_LABEL_BET,  expected_total: Some(4) },
            Step::Action { label: ACTION_LABEL_CALL, expected_total: Some(4) },
            Step::StreetAdvance,
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::StreetAdvance,
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
            Step::Action { label: ACTION_LABEL_CHECK, expected_total: None },
        ],
        label,
    );
    assert!(tree.nodes[leaf].is_terminal());

    eprintln!("All sample lines navigable for HU asymmetric ✓");
}

// ============================================================================
// Check 3 — Node-type sanity + reachability
// ============================================================================
//
// For every non-PLAYER node, verify the node type is justified by the
// state. The classifier accommodates two builder conventions that initially
// flagged false positives but are intentional:
//   - CHANCE node's board_state is the DESTINATION street (so a CHANCE
//     node at board=River means "advancing TO river"). Validity check is
//     "has at least one child" + "child's board state matches".
//   - TERMINAL with cumulative-unequal commits is legitimate under per-
//     street semantics when no betting occurred this street (asymmetric
//     blind carryover). We don't enforce the cumulative-equal rule.
//
// What we DO enforce (catches real bugs):
//   - Every CHANCE has ≥1 child
//   - Every node (except root) is reachable from root exactly once (no
//     orphans, no shared subtrees corrupting reach computation)
//   - Terminal-reason multiplicity: both fold-terminals AND non-fold
//     terminals exist (zero of either indicates over-pruning of one
//     branch category)

/// Categorize a terminal by which legal end-of-hand reason produced it.
/// All categories are valid; the function returns which one. Pure
/// classification — no errors.
fn classify_terminal_category(tree: &FlatTree, idx: usize, cfg: &TreeConfig) -> &'static str {
    let np = cfg.num_players as usize;
    let folded = tree.get_folded_mask(idx);
    let num_folded = (0..np).filter(|p| folded & (1 << p) != 0).count();
    let num_active = np - num_folded;
    let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
    let max_committable: Vec<i32> = (0..np)
        .map(|p| cfg.starting_stacks[p] + cfg.initial_contributions[p])
        .collect();
    let num_allin_active = (0..np)
        .filter(|&p| folded & (1 << p) == 0 && contribs[p] >= max_committable[p])
        .count();
    if num_active <= 1 {
        "fold-forfeit"
    } else if num_allin_active >= num_active {
        "all-in-runout"
    } else {
        // Everyone still alive and not all-in → river end with at least
        // one street's betting resolved (under per-street semantics, this
        // is valid even if cumulative commits aren't equal because of
        // asymmetric blind carryover from no-betting streets).
        "river-showdown"
    }
}

#[test]
fn hu_node_type_sanity_symmetric() {
    check_node_type_sanity(&hu_symmetric_cfg(), "HU symmetric [5,5]");
}

#[test]
fn hu_node_type_sanity_asymmetric() {
    check_node_type_sanity(&hu_asymmetric_cfg(), "HU asymmetric [2,1]");
}

fn check_node_type_sanity(cfg: &TreeConfig, label: &str) {
    let tree = build_tree(cfg).unwrap();
    eprintln!("\n=== Node-type sanity + reachability: {} ===", label);

    // (A) Every CHANCE has ≥1 child. Every PLAYER has ≥1 child. Every TERMINAL has 0.
    for (i, n) in tree.nodes.iter().enumerate() {
        if n.is_chance() {
            assert!(
                n.num_children >= 1,
                "[{}] CHANCE node[{}] has 0 children",
                label, i
            );
        } else if n.is_player() {
            assert!(
                n.num_children >= 1,
                "[{}] PLAYER node[{}] has 0 children (empty player node)",
                label, i
            );
        } else if n.is_terminal() {
            assert_eq!(
                n.num_children, 0,
                "[{}] TERMINAL node[{}] has {} children (should have 0)",
                label, i, n.num_children
            );
        }
    }

    // (B) Reachability: every node reachable from root exactly once.
    let total = tree.nodes.len();
    let mut visit_count = vec![0usize; total];
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        visit_count[i] += 1;
        if visit_count[i] > 1 {
            continue; // don't recurse multiple times
        }
        let n = &tree.nodes[i];
        for j in 0..n.num_children as usize {
            stack.push(tree.children[n.children_start as usize + j] as usize);
        }
    }
    let unreached: Vec<usize> = visit_count
        .iter()
        .enumerate()
        .filter(|(_, &c)| c == 0)
        .map(|(i, _)| i)
        .collect();
    let multi_visited: Vec<(usize, usize)> = visit_count
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 1)
        .map(|(i, c)| (i, *c))
        .collect();
    assert!(
        unreached.is_empty(),
        "[{}] {} orphan nodes (unreachable from root): first 10 = {:?}",
        label,
        unreached.len(),
        &unreached[..unreached.len().min(10)]
    );
    assert!(
        multi_visited.is_empty(),
        "[{}] {} multi-parent nodes (shared subtree corrupts reach computation): \
         first 10 = {:?}",
        label,
        multi_visited.len(),
        &multi_visited[..multi_visited.len().min(10)]
    );
    eprintln!("  All {} nodes reachable exactly once from root ✓", total);

    // (C) Terminal categories: must have both fold-forfeit AND at least one
    // non-fold category (river-showdown or all-in-runout). If only folds
    // exist, the tree never reaches showdown — over-pruning indicator. If
    // only showdowns exist, FOLD action was dropped from facing-bet nodes.
    let mut cats: std::collections::BTreeMap<&'static str, usize> = std::collections::BTreeMap::new();
    for (i, n) in tree.nodes.iter().enumerate() {
        if n.is_terminal() {
            *cats.entry(classify_terminal_category(&tree, i, cfg)).or_insert(0) += 1;
        }
    }
    eprintln!("  Terminal categories: {:?}", cats);
    let n_fold = cats.get("fold-forfeit").copied().unwrap_or(0);
    let n_show = cats.get("river-showdown").copied().unwrap_or(0);
    let n_allin = cats.get("all-in-runout").copied().unwrap_or(0);
    assert!(
        n_fold > 0,
        "[{}] no fold-forfeit terminals — tree has no fold lines (FOLD pruned from facing-bet nodes?)",
        label
    );
    assert!(
        n_show + n_allin > 0,
        "[{}] no showdown OR all-in run-out terminals — tree never reaches end of hand naturally",
        label
    );
    eprintln!("All node-type sanity checks ✓");
}

// ============================================================================
// Check 4 — Abstraction-aware reference count
// ============================================================================
//
// The point of this check is the one sample-paths cannot make: closing
// the over-correction failure mode raised by the 5,900× node-count drop.
// Sample-paths verify that specific lines exist; they cannot rule out
// missing lines nobody sampled. A global count check can.
//
// The completeness question, stated precisely, is: "given the abstraction
// policy (PotRel bet sizes, add_allin_threshold, force_allin_threshold,
// merging_threshold), does the built tree contain ALL the sequences that
// abstraction implies, or has the rewrite dropped some?"
//
// We answer it by writing a small REFERENCE enumerator that:
//   (a) ACCEPTS the abstraction as given (matches the production builder's
//       compute_actions semantics — bet target = prev_amount + pot*ratio;
//       clamp + force_allin if max <= clamped + new_pot*force_thr; explicit
//       allin if max <= prev + pot*add_thr); we are not validating the
//       abstraction, only completeness under it.
//   (b) ALWAYS RECURSES on every action it generates (no early returns,
//       no short-circuits — those are exactly the rewrite-time bug class
//       that could over-prune).
//   (c) COUNTS the resulting nodes (PLAYER, CHANCE, TERMINAL).
//
// Counts match → the builder fully expands its abstraction (completeness
// confirmed). Reference > tree → the builder is dropping lines the
// abstraction says should exist. Reference < tree → the builder is adding
// nodes the abstraction does not imply (the gate should already catch
// most of these, but a numeric mismatch here is still a signal).
//
// NOTE on independence: unlike the gate (which validates poker-rules
// LEGALITY and must be derived independently from the builder so they can
// disagree), this reference validates ABSTRACTION COVERAGE — i.e. it asks
// whether the builder did its own job, not whether the job is right. Same-
// logic-twice is therefore NOT a tautology problem here: a builder bug
// that short-circuits a branch will still produce a smaller count than
// the reference, because the reference does not short-circuit.

#[derive(Clone, Debug)]
struct RefState {
    street: u8,            // 0=Flop, 1=Turn, 2=River
    commits: Vec<i32>,     // per-player cumulative commitment
    round_start: Vec<i32>, // commits at start of current street (per-street snapshot)
    folded: Vec<bool>,
    has_acted: Vec<bool>,  // per-player; builder requires ALL active to have acted
    allin_flag: bool,      // persists across streets; gates re-raise/over-allin
    to_act: u8,            // player index 0..num_players
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefAct {
    Fold,
    Check,
    Call,
    Bet(i32),
    AllIn(i32),
}

fn ref_pot(cfg: &TreeConfig, s: &RefState) -> i32 {
    // Mirrors builder get_pot (2026-06-12 fix): starting_pot is dead
    // money additive with cumulative commits.
    cfg.starting_pot + s.commits.iter().sum::<i32>()
}

fn ref_max_committable(cfg: &TreeConfig, p: usize) -> i32 {
    cfg.starting_stacks[p] + cfg.initial_contributions[p]
}

/// Multi-player next-active traversal. Returns the next non-folded player
/// after `current` in circular index order, or None if no other active.
/// Mirrors builder::next_active_player. All-in players are still "active"
/// (folded=false) and DO get a forced-check turn.
fn ref_next_active(s: &RefState, current: usize) -> Option<usize> {
    let np = s.commits.len();
    for offset in 1..=np {
        let next = (current + offset) % np;
        if !s.folded[next] {
            return Some(next);
        }
    }
    None
}

/// First non-folded player by index (the "round starter" on a new street).
/// In multiway postflop this is the leftmost active player (typically SB,
/// or BB if SB folded). In HU postflop this is always p0 (BB).
fn ref_first_active(s: &RefState) -> Option<usize> {
    (0..s.commits.len()).find(|&p| !s.folded[p])
}

/// Reference compute_actions. Mirrors src/tree/builder.rs::compute_actions
/// for the (PotRelative bet, no raise) abstraction; deliberately does NOT
/// dedupe within-vec duplicates (the production builder doesn't either,
/// with merging_threshold=0, so duplicate AllIn entries become duplicate
/// child nodes and the count check needs to mirror that).
fn ref_actions(cfg: &TreeConfig, s: &RefState) -> Vec<RefAct> {
    let np = s.commits.len();
    let p = s.to_act as usize;
    let player_committed = s.commits[p];
    let max_amount = ref_max_committable(cfg, p);
    let player_remaining = (max_amount - player_committed).max(0);
    let pot = ref_pot(cfg, s);
    // prev_amount for bet/raise/call target = max CUMULATIVE commit among
    // OTHER non-folded players (Model A semantics — Call matches it, Bet
    // adds delta over it). Mirrors builder line 535.
    let max_other_cumulative = (0..np)
        .filter(|&q| q != p && !s.folded[q])
        .map(|q| s.commits[q])
        .max()
        .unwrap_or(0);
    let prev_amount = max_other_cumulative;
    // Facing-bet classification = PER-STREET comparison among OTHER non-
    // folded players (mirrors builder's compute_legal_action_class).
    // Asymmetric blinds with no betting this street have per-street
    // commits = 0 for everyone, so nobody is facing-bet despite cumulative
    // differences.
    let per_street_p = s.commits[p] - s.round_start[p];
    let max_other_per_street = (0..np)
        .filter(|&q| q != p && !s.folded[q])
        .map(|q| s.commits[q] - s.round_start[q])
        .max()
        .unwrap_or(0);

    let mut actions: Vec<RefAct> = Vec::new();

    if player_remaining <= 0 {
        // AllInForcedCheck path (early-return in builder).
        return vec![RefAct::Check];
    }

    let facing_bet = per_street_p < max_other_per_street;
    let _ = player_committed; // referenced via per_street_p and prev_amount

    if !facing_bet {
        // NotFacingBet: Check, bet sizes, optional AllIn.
        actions.push(RefAct::Check);
        for bs in &cfg.bet_sizes.bet {
            let delta = match bs {
                BetSize::PotRelative(r) => (pot as f64 * r).round() as i32,
                _ => unimplemented!("only PotRelative bet sizes supported in reference"),
            };
            let target = prev_amount + delta;
            actions.push(RefAct::Bet(target));
        }
        // Explicit allin add-on: max_amount <= pot * add_allin_threshold
        if max_amount <= (pot as f64 * cfg.add_allin_threshold).round() as i32 {
            actions.push(RefAct::AllIn(max_amount));
        }
    } else {
        // FacingBet: Fold, Call, raise sizes if any, optional AllIn.
        // Raises AND open-allin gated on !allin_flag — once someone has gone
        // all-in this hand, you can only Fold or Call against it (you cannot
        // re-raise into an existing all-in commitment that you can't exceed
        // by definition).
        actions.push(RefAct::Fold);
        actions.push(RefAct::Call);
        assert!(cfg.bet_sizes.raise.is_empty(), "ref enumerator only supports empty raise sizes");
        if !s.allin_flag {
            let allin_thr = (pot as f64 * cfg.add_allin_threshold).round() as i32;
            if max_amount <= prev_amount + allin_thr {
                actions.push(RefAct::AllIn(max_amount));
            }
        }
    }

    // clamp_and_force_allin: convert Bet/Raise to AllIn when force_allin
    // threshold is met. min_amount = max(1, player_committed + to_call).
    // The builder's new_pot formula adds 2*new_diff which under-counts
    // multi-player pots (it assumes one other matches the difference);
    // we mirror it verbatim since this is an abstraction-policy detail
    // we are validating coverage of, not correctness of.
    let to_call = (max_other_cumulative - player_committed).max(0);
    let min_amount = (player_committed + to_call.min(player_remaining))
        .max(1)
        .min(max_amount);
    for a in actions.iter_mut() {
        if let RefAct::Bet(amt) = *a {
            let clamped = amt.clamp(min_amount, max_amount);
            let new_diff = clamped - prev_amount;
            let new_pot = pot + 2 * new_diff;
            let force_thr = (new_pot as f64 * cfg.force_allin_threshold).round() as i32;
            if max_amount <= clamped + force_thr {
                *a = RefAct::AllIn(max_amount);
            } else if clamped != amt {
                *a = RefAct::Bet(clamped);
            }
        }
    }

    // SORT + DEDUP (builder lines 639-640) — removes exact-duplicate
    // actions. Critical case: force-allin clamp converts a Bet to AllIn(max),
    // AND the explicit-allin push also added AllIn(max). Both target the
    // same total commit; the builder sorts and dedups to a single AllIn.
    // HU never triggers both conditions simultaneously (pot doesn't grow
    // large enough), so this is a no-op there — but in 6p with larger pots
    // and larger max_committable it fires regularly and accounts for
    // significant count divergence if omitted.
    actions.sort_by(|a, b| {
        let key = |x: &RefAct| match x {
            RefAct::Fold => (0, 0),
            RefAct::Check => (1, 0),
            RefAct::Call => (2, 0),
            RefAct::Bet(amt) => (3, *amt),
            RefAct::AllIn(amt) => (5, *amt),
        };
        key(a).cmp(&key(b))
    });
    actions.dedup();

    // merge_bet_actions is a no-op when merging_threshold = 0.0.
    assert_eq!(
        cfg.merging_threshold, 0.0,
        "ref enumerator only handles merging_threshold = 0.0"
    );

    actions
}

/// Apply an action to a state, producing the child state. Does NOT yet
/// classify the child as PLAYER/CHANCE/TERMINAL — that happens at the
/// recursion level so we can also count chance/terminal nodes the builder
/// would emit between this state and the next PLAYER state.
fn ref_apply(cfg: &TreeConfig, s: &RefState, a: RefAct) -> RefState {
    let np = s.commits.len();
    let mut n = s.clone();
    let p = s.to_act as usize;
    let max_amount = ref_max_committable(cfg, p);
    let max_other = (0..np)
        .filter(|&q| q != p && !s.folded[q])
        .map(|q| s.commits[q])
        .max()
        .unwrap_or(0);
    match a {
        RefAct::Fold => {
            n.folded[p] = true;
            n.has_acted[p] = true;
        }
        RefAct::Check => {
            n.has_acted[p] = true;
        }
        RefAct::Call => {
            n.commits[p] = max_other.min(max_amount);
            n.has_acted[p] = true;
        }
        RefAct::Bet(amt) => {
            n.commits[p] = amt;
            n.has_acted = vec![false; np];
            n.has_acted[p] = true;
        }
        RefAct::AllIn(amt) => {
            n.commits[p] = amt;
            n.has_acted = vec![false; np];
            n.has_acted[p] = true;
            n.allin_flag = true;
        }
    }
    // Next acting player: builder uses next_active_player (skips folded
    // only; all-in players still take their forced-check turn).
    n.to_act = ref_next_active(&n, p).unwrap_or(0) as u8;
    n
}

/// Recursively count nodes in the subtree the builder WOULD emit rooted at
/// `s`. The state passed in represents a node about to be classified: it
/// may become a PLAYER node (with action children), a CHANCE node (street
/// advance with one child), or a TERMINAL node (leaf).
fn ref_count(cfg: &TreeConfig, s: &RefState) -> (usize, usize, usize) {
    // Returns (player_nodes, chance_nodes, terminal_nodes) in subtree
    // including this node.
    let np = s.commits.len();
    let num_unfolded = s.folded.iter().filter(|&&f| !f).count();

    // Terminal classification 1: only one (or zero) unfolded player.
    if num_unfolded <= 1 {
        return (0, 0, 1);
    }

    // ALL-IN SHORTCUT (mirrors builder lines 359-388): when allin_flag is
    // true AND every non-folded player has committed their max, skip the
    // round of forced-check PLAYER nodes entirely. River → TERMINAL,
    // earlier streets → CHANCE.
    let unfolded: Vec<usize> = (0..np).filter(|p| !s.folded[*p]).collect();
    if s.allin_flag {
        let all_allin = unfolded
            .iter()
            .all(|&p| s.commits[p] >= ref_max_committable(cfg, p));
        if all_allin {
            // Chronological street ordinal (Preflop=0, Flop=1, Turn=2, River=3).
            if s.street == 3 {
                return (0, 0, 1);
            }
            let mut next = s.clone();
            next.street += 1;
            next.round_start = next.commits.clone();
            next.has_acted = vec![false; np];
            next.to_act = ref_first_active(&next).unwrap_or(0) as u8;
            let (cp, cc, ct) = ref_count(cfg, &next);
            return (cp, cc + 1, ct);
        }
    }

    // Round-complete check (mirrors builder is_round_complete):
    //   1. Every non-folded player must have acted this round.
    //   2. AND either (a) all non-folded have equal cumulative commits, or
    //      (b) all non-folded have per-street commit 0 (no-betting case
    //      for asymmetric blinds).
    let all_acted = unfolded.iter().all(|&p| s.has_acted[p]);
    // Standing-bet rule (parallel fix to builder.rs is_round_complete,
    // 2026-06-04): matched standing bet OR all-in at max_committable.
    let standing_bet = unfolded.iter().map(|&p| s.commits[p]).max().unwrap();
    let matched_or_allin = unfolded.iter().all(|&p| {
        s.commits[p] == standing_bet || s.commits[p] >= ref_max_committable(cfg, p)
    });
    let no_betting = unfolded
        .iter()
        .all(|&p| s.commits[p] - s.round_start[p] == 0);
    let round_complete = all_acted && (matched_or_allin || no_betting);

    if round_complete {
        // River-end check: use chronological street ordinal (Preflop=0,
        // Flop=1, Turn=2, River=3 via BoardState::street()). River is
        // street == 3 regardless of where the tree started.
        if s.street == 3 {
            return (0, 0, 1); // river end → terminal
        }
        let mut next = s.clone();
        next.street += 1;
        next.round_start = next.commits.clone();
        next.has_acted = vec![false; np];
        next.to_act = ref_first_active(&next).unwrap_or(0) as u8;
        let (cp, cc, ct) = ref_count(cfg, &next);
        return (cp, cc + 1, ct);
    }

    // PLAYER node. Generate actions, recurse on each — INCLUDING Fold.
    //
    // HISTORY (2026-06-12): this enumerator (and the builder) used to
    // collapse EVERY fold to a terminal, justified as "standard CFR
    // pruning — other branches enumerate equivalent positions". That
    // justification was false: after p1 folds to p0's bet, no other
    // branch puts p2 facing that bet with p1 out. The collapse silently
    // deleted all multiway fold-continuation subtrees (a wrong GAME,
    // caught by the play-harness fold-mask audit). Folds now recurse
    // through ref_apply like any action; the num_unfolded <= 1 check at
    // the top of ref_count terminates the genuine hand-ending folds.
    let acts = ref_actions(cfg, s);
    let mut tot_p = 1usize; // this PLAYER node
    let mut tot_c = 0usize;
    let mut tot_t = 0usize;
    for a in acts {
        let child = ref_apply(cfg, s, a);
        let (cp, cc, ct) = ref_count(cfg, &child);
        tot_p += cp;
        tot_c += cc;
        tot_t += ct;
    }
    (tot_p, tot_c, tot_t)
}

fn ref_initial(cfg: &TreeConfig) -> RefState {
    let np = cfg.num_players as usize;
    // Use chronological street ordinal (Preflop=0, Flop=1, Turn=2, River=3
    // via BoardState::street()), NOT the enum repr value (where Preflop=3
    // for backward compat with existing hardcoded board_state == 1/2 checks).
    // This lets the same increment-based street logic handle both
    // postflop-start (street 1..3) and preflop-start (street 0..3) chains.
    //
    // First-actor dispatch: button in HU preflop, leftmost-active in postflop
    // (matches builder's first_preflop_player vs first_postflop_player).
    let to_act: u8 = match cfg.initial_state {
        BoardState::Preflop => {
            // HU button = highest-indexed active = num_players - 1.
            // Multiway preflop ordering is deferred (matches builder caveat).
            cfg.num_players - 1
        }
        _ => 0,
    };
    // round_start: parallel fix to builder.rs's committed_at_round_start.
    // At preflop, no round preceded — blinds are first-round actions,
    // so round_start = 0 for everyone. At postflop, initial_contributions
    // represent pre-this-street commits → round_start = initial_contribs.
    let round_start = match cfg.initial_state {
        BoardState::Preflop => vec![0_i32; np],
        _ => cfg.initial_contributions.clone(),
    };
    RefState {
        street: cfg.initial_state.street(),
        commits: cfg.initial_contributions.clone(),
        round_start,
        folded: vec![false; np],
        has_acted: vec![false; np],
        allin_flag: false,
        to_act,
    }
}

fn count_tree(tree: &FlatTree) -> (usize, usize, usize) {
    let mut p = 0;
    let mut c = 0;
    let mut t = 0;
    for n in &tree.nodes {
        if n.is_player() { p += 1; }
        else if n.is_chance() { c += 1; }
        else { t += 1; }
    }
    (p, c, t)
}

#[test]
fn hu_abstraction_count_symmetric() {
    let cfg = hu_symmetric_cfg();
    let tree = build_tree(&cfg).unwrap();
    let (tp, tc, tt) = count_tree(&tree);
    let init = ref_initial(&cfg);
    let (rp, rc, rt) = ref_count(&cfg, &init);

    eprintln!("\n=== Abstraction-aware count: HU symmetric [5,5] ===");
    eprintln!("Tree:      player={:3}  chance={:3}  terminal={:3}  total={}",
        tp, tc, tt, tp + tc + tt);
    eprintln!("Reference: player={:3}  chance={:3}  terminal={:3}  total={}",
        rp, rc, rt, rp + rc + rt);
    eprintln!("Delta:     player={:+}  chance={:+}  terminal={:+}  total={:+}",
        tp as i64 - rp as i64,
        tc as i64 - rc as i64,
        tt as i64 - rt as i64,
        (tp + tc + tt) as i64 - (rp + rc + rt) as i64);

    assert_eq!(tp, rp, "PLAYER count mismatch — builder vs abstraction-reference \
        diverged. Tree<ref ⇒ builder is dropping lines its own abstraction says exist \
        (the over-correction failure mode the 5,900× reduction warned about). \
        Tree>ref ⇒ builder is adding nodes the abstraction does not imply.");
    assert_eq!(tc, rc, "CHANCE count mismatch — see PLAYER message above.");
    assert_eq!(tt, rt, "TERMINAL count mismatch — see PLAYER message above.");
    eprintln!("Tree fully expands its abstraction (no over-pruning, no extras) ✓");
}

#[test]
fn hu_abstraction_count_asymmetric() {
    let cfg = hu_asymmetric_cfg();
    let tree = build_tree(&cfg).unwrap();
    let (tp, tc, tt) = count_tree(&tree);
    let init = ref_initial(&cfg);
    let (rp, rc, rt) = ref_count(&cfg, &init);

    eprintln!("\n=== Abstraction-aware count: HU asymmetric [2,1] ===");
    eprintln!("Tree:      player={:3}  chance={:3}  terminal={:3}  total={}",
        tp, tc, tt, tp + tc + tt);
    eprintln!("Reference: player={:3}  chance={:3}  terminal={:3}  total={}",
        rp, rc, rt, rp + rc + rt);
    eprintln!("Delta:     player={:+}  chance={:+}  terminal={:+}  total={:+}",
        tp as i64 - rp as i64,
        tc as i64 - rc as i64,
        tt as i64 - rt as i64,
        (tp + tc + tt) as i64 - (rp + rc + rt) as i64);

    assert_eq!(tp, rp, "PLAYER count mismatch on HU asymmetric.");
    assert_eq!(tc, rc, "CHANCE count mismatch on HU asymmetric.");
    assert_eq!(tt, rt, "TERMINAL count mismatch on HU asymmetric.");
    eprintln!("Tree fully expands its abstraction (no over-pruning, no extras) ✓");
}

// ============================================================================
// Check 5 — Abstraction-aware count for the multiway configs that drove the
//           rewrite. 6p asymmetric is THE config whose 14.5M → 2,431 (~5,900×)
//           reduction raised the over-correction worry; the HU count match
//           confirms the rewrite is sound in HU but doesn't, by itself,
//           prove the 6p reduction was real bug-removal vs over-correction.
//           Pointing the same validated reference enumerator at the 6p
//           configs is the direct measurement.
// ============================================================================

fn config_6p_asymmetric() -> TreeConfig {
    // The original "5,900× reduction" config.
    TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 0, // was 35: double-counted contribs (2026-06-12 semantics)
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,
    }

}

fn config_6p_symmetric() -> TreeConfig {
    TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 0, // was 30: double-counted contribs (2026-06-12 semantics)
        starting_stacks: vec![200; 6],
        initial_contributions: vec![5; 6],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,
    }

}

fn config_3p_asymmetric() -> TreeConfig {
    TreeConfig {
        num_players: 3,
        initial_state: BoardState::Flop,
        starting_pot: 0, // was 15: double-counted contribs (2026-06-12 semantics)
        starting_stacks: vec![200; 3],
        initial_contributions: vec![10, 5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,
    }

}

fn run_count_check(cfg: &TreeConfig, label: &str) {
    let tree = build_tree(cfg).unwrap();
    let (tp, tc, tt) = count_tree(&tree);
    let init = ref_initial(cfg);
    let (rp, rc, rt) = ref_count(cfg, &init);
    eprintln!("\n=== Abstraction-aware count: {} ===", label);
    eprintln!(
        "Tree:      player={:5}  chance={:4}  terminal={:5}  total={}",
        tp,
        tc,
        tt,
        tp + tc + tt
    );
    eprintln!(
        "Reference: player={:5}  chance={:4}  terminal={:5}  total={}",
        rp,
        rc,
        rt,
        rp + rc + rt
    );
    eprintln!(
        "Delta:     player={:+}  chance={:+}  terminal={:+}  total={:+}",
        tp as i64 - rp as i64,
        tc as i64 - rc as i64,
        tt as i64 - rt as i64,
        (tp + tc + tt) as i64 - (rp + rc + rt) as i64
    );
    assert_eq!(
        tp, rp,
        "[{}] PLAYER count mismatch. Tree<ref ⇒ builder is dropping lines its own \
         abstraction says exist (the over-correction failure mode); Tree>ref ⇒ \
         builder is adding nodes the abstraction does not imply.",
        label
    );
    assert_eq!(tc, rc, "[{}] CHANCE count mismatch.", label);
    assert_eq!(tt, rt, "[{}] TERMINAL count mismatch.", label);
    eprintln!("Tree fully expands its abstraction (no over-pruning, no extras) ✓");
}

#[test]
fn count_6p_asymmetric() {
    // THE config the 5,900× reduction came from. Direct measurement that
    // the reduction was real bug-removal vs over-correction.
    run_count_check(&config_6p_asymmetric(), "6p asymmetric [10,5,5,5,5,5]");
}

#[test]
fn count_6p_symmetric() {
    run_count_check(&config_6p_symmetric(), "6p symmetric [5;6]");
}

#[test]
fn count_3p_asymmetric() {
    run_count_check(&config_3p_asymmetric(), "3p asymmetric [10,5,5]");
}

// ============================================================================
// Preflop completeness check (#41 P2.3)
// ============================================================================
// HU preflop config: same blinds as `hu_asymmetric_cfg` but starting at
// Preflop. The four-zone tree (PRE → CHANCE → FLOP → CHANCE → TURN →
// CHANCE → RIVER) must contain exactly what the abstraction implies —
// the same completeness discipline applied to postflop, now extended to
// the new preflop component.
//
// The convenient structural fact to verify: HU preflop tree has 308
// nodes (per preflop_tree_smoke.rs). The unshortcuted reference must
// match exactly. If it doesn't, the preflop tree is silently dropping
// lines (the over-correction failure mode the validation arc taught us
// to verify, not assume).

fn config_hu_preflop_asymmetric() -> TreeConfig {
    TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 0, // was 3: double-counted the live blinds (2026-06-12 semantics)
        starting_stacks: vec![100, 100],
        initial_contributions: vec![2, 1],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,
    }

}

#[test]
fn count_hu_preflop_asymmetric() {
    run_count_check(&config_hu_preflop_asymmetric(), "HU PREFLOP asymmetric [2,1]");
}
