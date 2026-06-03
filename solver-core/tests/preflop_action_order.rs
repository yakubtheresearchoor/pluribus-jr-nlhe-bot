// P2.1 of preflop integration (#41): the discriminating button-first test.
//
// The cheapest possible discriminator for "preflop action order is HU-correct,
// not multiway-correct." In HU postflop, the BB (= player 0, the leftmost
// active player) acts first. In HU preflop, the BUTTON (= SB = player 1,
// the highest-indexed active player) acts first. These are OPPOSITE
// conventions; getting them wrong silently produces a tree that solves a
// different game than intended.
//
// `hu_completeness.rs` scope note explicitly flagged this: "the postflop-
// order machinery in this file CANNOT be reused for that — it would assert
// the wrong player." This test verifies the root acts player_id = button
// (NOT BB) for preflop-start trees, distinguishing the new preflop machinery
// from the existing postflop machinery.
//
// Run this test FIRST in the preflop component validation sequence — it's
// the cheapest discriminator and catches the most likely bug (accidentally
// reusing postflop ordering).

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn hu_preflop_cfg() -> TreeConfig {
    // HU preflop with standard blinds: p0 = BB (contributes 2), p1 = SB
    // (contributes 1). Initial state Preflop.
    TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
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
    }
}

fn hu_flop_cfg() -> TreeConfig {
    // Same blinds, but starting at Flop. Used as the contrastive check:
    // postflop root must act player 0 (BB), preflop root must act player 1
    // (button). The contrast confirms the new machinery actually does
    // something different, not just "always returns 1."
    TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 3,
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
    }
}

#[test]
fn preflop_root_acts_button() {
    let cfg = hu_preflop_cfg();
    let tree = build_tree(&cfg).expect("preflop tree build");
    let root = &tree.nodes[0];
    eprintln!(
        "HU preflop root: node_type={} player_id={} board_state={}",
        root.node_type, root.player_id, root.board_state
    );
    assert!(
        root.is_player(),
        "preflop root must be PLAYER (got node_type={})",
        root.node_type
    );
    assert_eq!(
        root.player_id, 1,
        "HU preflop root must act player_id = 1 (button = SB), got player {}. \
         If this is 0, the preflop machinery is using postflop ordering — the \
         exact bug hu_completeness.rs scope note warned about.",
        root.player_id
    );
    assert_eq!(
        root.board_state,
        BoardState::Preflop as u8,
        "preflop root board_state must be Preflop"
    );
}

#[test]
fn postflop_root_acts_bb_unchanged() {
    // Contrastive check: postflop machinery unchanged. HU flop root acts
    // player 0 (BB), the existing convention. If this regressed, the new
    // preflop dispatch broke postflop, which would be a serious regression
    // worth catching immediately.
    let cfg = hu_flop_cfg();
    let tree = build_tree(&cfg).expect("flop tree build");
    let root = &tree.nodes[0];
    eprintln!(
        "HU flop root: node_type={} player_id={} board_state={}",
        root.node_type, root.player_id, root.board_state
    );
    assert_eq!(
        root.player_id, 0,
        "HU flop root must act player_id = 0 (BB) — existing postflop \
         convention. If this changed, the preflop dispatch in build_tree \
         broke the postflop path."
    );
}

#[test]
fn preflop_vs_postflop_order_distinct() {
    // The actual discriminator: preflop root player ≠ postflop root player.
    // This is the structural property that fails if preflop ordering is
    // accidentally reused from postflop. If both return the same player,
    // the action-order reversal isn't actually happening.
    let pre = build_tree(&hu_preflop_cfg()).expect("preflop tree");
    let post = build_tree(&hu_flop_cfg()).expect("flop tree");
    assert_ne!(
        pre.nodes[0].player_id, post.nodes[0].player_id,
        "preflop root and postflop root must act DIFFERENT players in HU \
         (preflop: button = p1; postflop: BB = p0). If they agree, the \
         preflop machinery is just reusing postflop ordering."
    );
}
