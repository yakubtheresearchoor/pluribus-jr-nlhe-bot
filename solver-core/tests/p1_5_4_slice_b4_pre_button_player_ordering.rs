// P1.5.4 Slice B.4 prerequisite: button_player ordering verification.
//
// Per the lead (2026-06-04): the schema change added `button_player:
// Option<u8>` with `None` preserving legacy behavior. The 49-test
// regression that passed after the refactor confirms the legacy `None`
// path is intact (no behavioral change for existing tests). It does
// NOT confirm the new button-aware ordering is correct, because no
// existing test sets `button_player` to a real value.
//
// This test exercises the NEW path explicitly. It verifies:
//
//   1. Preflop UTG-first at 6-max: with button=5, first preflop actor
//      is UTG = (5+3) mod 6 = player 2.
//
//   2. Postflop SB-first at 6-max: with button=5, first postflop actor
//      is SB = (5+1) mod 6 = player 0.
//
//   3. The flip at the preflop-to-flop transition: the same tree has
//      preflop nodes ordered UTG-first AND postflop nodes (children of
//      the preflop chance) ordered SB-first. This is the multiway seam
//      where preflop ordering could be right and postflop wrong (or
//      vice versa).
//
//   4. HU collapse under explicit button: with button=1 at HU (np=2),
//      preflop first actor is button=SB=player 1 (HU special case),
//      preserving the historical HU behavior under the new path.
//
// Without this test, B.4 would build lossless-collapse verification on
// an ordering that's asserted-by-construction but not verified-by-test.
// If the `(button+3)%np` or `(button+1)%np` arithmetic is off (off-by-
// one, wrong modular reduction, wrong button-position assumption), B.4
// would certify the lossless collapse of a wrong-ordering tree — the
// same wasted-validation-on-wrong-game risk that stopping before B.4
// was meant to avoid.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// Walk DOWN from `node_idx`, following children indices, until finding
/// the first PLAYER node. Used to find the first postflop player node
/// below a preflop chance node (chance's children are postflop player
/// or chance nodes; we drill until we hit a player).
fn first_player_descendant(tree: &FlatTree, node_idx: usize) -> Option<usize> {
    let n = &tree.nodes[node_idx];
    if n.is_player() { return Some(node_idx); }
    for &child in tree.node_children(node_idx) {
        if let Some(found) = first_player_descendant(tree, child as usize) {
            return Some(found);
        }
    }
    None
}

/// Find the first preflop→flop chance node in the tree (chance whose
/// parent is in the preflop zone). Returns the chance node's index.
fn first_preflop_chance(tree: &FlatTree) -> Option<usize> {
    let mut parents = vec![None::<u32>; tree.num_nodes()];
    for p in 0..tree.num_nodes() {
        for &c in tree.node_children(p) {
            parents[c as usize] = Some(p as u32);
        }
    }
    for idx in 0..tree.num_nodes() {
        if !tree.nodes[idx].is_chance() { continue; }
        if let Some(par) = parents[idx] {
            if tree.nodes[par as usize].board_state == BoardState::Preflop as u8 {
                return Some(idx);
            }
        }
    }
    None
}

/// Walk the FIRST preflop action path (always take child[1] = call/check)
/// and record the player_id at each player node, until we hit a chance
/// or terminal. Returns the ordered sequence of player_ids.
fn first_preflop_action_path_players(tree: &FlatTree) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cur = 0usize;
    loop {
        let n = &tree.nodes[cur];
        if !n.is_player() { break; }
        out.push(n.player_id);
        let children = tree.node_children(cur);
        if children.is_empty() { break; }
        // Action layout (per builder): for a player facing-bet-from-blinds
        // node, the action set typically includes fold (idx 0), call (idx
        // 1), and maybe raise (higher idx). Walking child[1] takes the
        // CALL action which keeps the round alive without folding (so
        // the next player gets to act, exercising the action ordering).
        // For an opening player (UTG with no facing-bet), child[1]
        // similarly takes the second action (typically check/call or
        // open). Either way, child[1] keeps us in the preflop round.
        let next_idx = if children.len() > 1 { children[1] } else { children[0] };
        cur = next_idx as usize;
    }
    out
}

#[test]
fn slice_b4_pre_6max_preflop_first_actor_is_utg() {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; 6],
        // SB=player 0 (1 chip), BB=player 1 (2 chips), others=0
        initial_contributions: vec![1, 2, 0, 0, 0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
            max_bets_per_street: None,  // Explicit button = player 5 (BTN)
    };
    let tree = build_tree(&cfg).expect("6-max preflop tree builds");

    // 1. Root should be UTG = (5 + 3) % 6 = 2.
    assert_eq!(tree.nodes[0].player_id, 2,
        "root preflop player should be UTG = (button+3)%6 = (5+3)%6 = 2; got {}",
        tree.nodes[0].player_id);

    // 2. Walking the preflop action path (children[1] at each step) should
    //    produce the rotation UTG → MP → CO → BTN → SB → BB
    //    = 2 → 3 → 4 → 5 → 0 → 1.
    let path = first_preflop_action_path_players(&tree);
    eprintln!("6-max preflop action path: {:?}", path);
    // The first 6 entries should match the expected rotation (1 action per player).
    let expected_prefix: Vec<u8> = vec![2, 3, 4, 5, 0, 1];
    assert!(path.len() >= 6,
        "expected at least 6 preflop player nodes in the action path; got {} ({:?})",
        path.len(), path);
    assert_eq!(&path[..6], &expected_prefix[..],
        "preflop action order at 6-max with button=5 should be UTG,MP,CO,BTN,SB,BB \
         = [2,3,4,5,0,1]; got {:?}",
        &path[..6]);

    eprintln!("Slice B.4 pre PASS (6-max preflop): root = UTG = 2; action rotation \
              follows UTG,MP,CO,BTN,SB,BB.");
}

#[test]
fn slice_b4_pre_6max_postflop_first_actor_is_sb_via_transition() {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; 6],
        initial_contributions: vec![1, 2, 0, 0, 0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("6-max preflop tree builds");

    // 3. The preflop-to-flop transition flip:
    //    Find the first preflop chance node, walk into its children,
    //    find the first player descendant. That should be SB = (5+1)%6 = 0.
    //    This verifies the action order FLIPS from UTG-first (preflop) to
    //    SB-first (postflop) across the chance boundary.
    let chance_idx = first_preflop_chance(&tree)
        .expect("6-max preflop tree should have at least one preflop chance node");
    eprintln!("first preflop chance node at idx {} (board_state={}, children={:?})",
        chance_idx, tree.nodes[chance_idx].board_state, tree.node_children(chance_idx));

    let first_postflop_player_idx = first_player_descendant(&tree, chance_idx)
        .expect("preflop chance node should have a player descendant (the first postflop actor)");
    let first_postflop_player_id = tree.nodes[first_postflop_player_idx].player_id;
    eprintln!("first postflop player below chance: node {}, player_id = {}",
        first_postflop_player_idx, first_postflop_player_id);

    assert_eq!(first_postflop_player_id, 0,
        "first postflop player at 6-max with button=5 should be SB = (button+1)%6 = (5+1)%6 = 0; \
         got {}. If wrong, the preflop-to-flop transition is NOT flipping to SB-first ordering; \
         the multiway action-order seam is broken.",
        first_postflop_player_id);

    eprintln!("Slice B.4 pre PASS (6-max transition): preflop chance → first postflop \
              player = SB = 0. The action order CORRECTLY flips from UTG-first preflop to \
              SB-first postflop at the chance boundary.");
}

#[test]
fn slice_b4_pre_hu_explicit_button_preserves_preflop_button_first() {
    // HU is the special case: button = SB (same player). With
    // explicit button_player=Some(1) at HU, the first preflop actor
    // should be the button = player 1. This collapses correctly via
    // the HU branch (np==2: return button) and preserves the existing
    // HU behavior that worked under the None legacy path (which
    // returned highest-indexed-active = player 1 by coincidence).
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![2, 1],  // BB at player 0, SB at player 1
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(1),
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("HU preflop tree builds");

    // HU preflop first actor: button = SB = player 1.
    assert_eq!(tree.nodes[0].player_id, 1,
        "HU preflop first actor with button=1 should be player 1 (button=SB); got {}",
        tree.nodes[0].player_id);

    // HU postflop first actor: BB = (button+1) % 2 = (1+1) % 2 = 0.
    let chance_idx = first_preflop_chance(&tree)
        .expect("HU preflop tree should have a preflop chance node");
    let first_postflop_player_idx = first_player_descendant(&tree, chance_idx)
        .expect("preflop chance node should have a player descendant");
    let first_postflop_player_id = tree.nodes[first_postflop_player_idx].player_id;
    assert_eq!(first_postflop_player_id, 0,
        "HU postflop first actor with button=1 should be player 0 (BB); got {}",
        first_postflop_player_id);

    eprintln!("Slice B.4 pre PASS (HU explicit button): preflop first = button=SB=1, \
              postflop first = BB=0. HU collapses correctly under explicit button_player.");
}

#[test]
fn slice_b4_pre_legacy_none_path_unchanged_for_hu() {
    // The legacy None path: existing HU tests use the highest-indexed-
    // active inference. With initial_contributions=[2, 1] (BB at 0, SB
    // at 1), highest-indexed-active = player 1 = SB. So legacy None
    // gives player 1 as first preflop actor. This matches what
    // button=Some(1) gives (per the test above), so the legacy and
    // explicit paths agree at HU. This is the no-regression check that
    // the 49-test suite implicitly verifies; documenting it here as a
    // standalone assertion makes it explicit.
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![2, 1],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,  // legacy inference
    };
    let tree = build_tree(&cfg).expect("HU preflop tree builds with None button_player");
    // Legacy: highest-indexed-active = player 1 = SB = first preflop actor.
    assert_eq!(tree.nodes[0].player_id, 1,
        "HU preflop with button_player=None should still give player 1 (legacy inference); got {}",
        tree.nodes[0].player_id);
    eprintln!("Slice B.4 pre PASS (HU legacy None): legacy inference still gives player 1 at HU.");
}
