// P1.5.4 Slice B.4 prerequisite: preflop→flop seam chip conservation.
//
// Per the lead (2026-06-04): "the durable fix isn't just changing
// committed_at_round_start, it's adding a test that connects the
// preflop builder's flop-start output to the postflop validator's
// expected input, an end-to-end chip-conservation check across the
// preflop-to-flop seam. That test would have caught this bug, and it
// would catch any future preflop-to-flop handoff error."
//
// This is the seam test. It connects the preflop builder's output
// (chip state at the first flop-zone node below the preflop chance)
// to the chip state that EVERY VALID PREFLOP SEQUENCE should produce
// per actual poker rules: every player who entered the flop must have
// matched the high bet (or be all-in). For the simplest sequence (no
// raises, everyone enters cheaply), that means everyone called the BB.
//
// The bug surfaced by the prior chip trace: SB could "Check" at the
// preflop root and see the flop having paid only 1 chip (the SB
// blind), without matching the BB's 2 chips. The seam test asserts
// that at every flop-zone player node reached from a preflop sequence
// where no one folded, all active players have equal contributions
// (everyone matched the same bet level). If SB enters the flop with
// 1 chip when BB has 2, the assertion fires.
//
// This is the connecting test that was missing: the postflop tests
// validated against EXPLICIT [5,5] contributions; nothing tested that
// the preflop builder's output matches that shape. The seam test fills
// that gap.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// For a given preflop tree, find every preflop→flop CHANCE NODE
/// (chance node whose parent is in the preflop zone) and verify that
/// at that chance node, all NON-FOLDED players have equal contributions
/// (they've matched the standing preflop bet — anyone who didn't match
/// must have folded). This is the seam: chips entering the flop from
/// preflop must be conserved per the standing-bet rule.
///
/// **Important correction from the initial too-broad test:** intermediate
/// flop PLAYER nodes (where flop betting is in progress) validly have
/// unequal contributions (the bettor just bet, the next player hasn't
/// responded yet). The seam check is at the CHANCE NODE itself — the
/// point where preflop ENDS and the flop subtree begins. At that node,
/// preflop is complete: every non-folded player has either matched the
/// standing bet or is all-in.
fn verify_no_fold_flop_entries(tree: &FlatTree, label: &str) {
    let np = tree.num_players as usize;

    // Build a parent map to identify preflop→flop chance nodes.
    let mut parent_of = vec![None::<u32>; tree.num_nodes()];
    for p in 0..tree.num_nodes() {
        for &c in tree.node_children(p) {
            parent_of[c as usize] = Some(p as u32);
        }
    }

    let mut violations: Vec<(usize, Vec<i32>, u16)> = Vec::new();
    for idx in 0..tree.num_nodes() {
        let n = &tree.nodes[idx];
        if !n.is_chance() { continue; }
        // Preflop→flop chance: parent in preflop zone, chance itself carries
        // destination state = Flop.
        let par = match parent_of[idx] {
            Some(p) => p as usize,
            None => continue,
        };
        if tree.nodes[par].board_state != BoardState::Preflop as u8 { continue; }
        // This is a preflop→flop seam node. Check chip conservation
        // among NON-FOLDED players (folded players validly have lower
        // contributions — their forfeited blinds/bets are still in the pot).
        let fold_mask = tree.get_folded_mask(idx);
        let contribs: Vec<i32> = (0..np)
            .map(|p| tree.get_contribution(idx, p as u8))
            .collect();
        // Among non-folded players: each must EITHER match the standing
        // bet (the max contribution) OR be all-in at their personal
        // max_committable (short-all-in with the excess from a deeper-
        // stacked opponent being uncalled-bet-returned per poker rules).
        let active_players: Vec<usize> = (0..np)
            .filter(|&p| (fold_mask & (1 << p)) == 0)
            .collect();
        if active_players.is_empty() { continue; }
        let standing_bet = active_players.iter()
            .map(|&p| contribs[p])
            .max().unwrap();
        // Determine each non-folded player's max_committable.
        // max_committable = starting_stack + initial_contribution.
        let starting_stacks = vec![100; np];  // matches the test configs
        let initial_contributions: Vec<i32> = {
            // Recover from the tree's root contributions (they ARE the
            // initial contributions at the preflop root).
            (0..np).map(|p| tree.get_contribution(0, p as u8)).collect()
        };
        let max_committable: Vec<i32> = (0..np)
            .map(|p| starting_stacks[p] + initial_contributions[p])
            .collect();
        let mut ok = true;
        for &p in &active_players {
            let matched = contribs[p] == standing_bet;
            let all_in = contribs[p] >= max_committable[p];
            if !matched && !all_in {
                ok = false;
                break;
            }
        }
        if !ok {
            violations.push((idx, contribs, fold_mask));
        }
    }

    eprintln!("\n══ {} ══", label);
    eprintln!("Total flop player nodes with no-fold mask: checked.");
    if violations.is_empty() {
        eprintln!("  ✓ NO VIOLATIONS: every no-fold flop entry has equal contributions");
        eprintln!("    across all active players. Preflop→flop chip handoff is correct.");
    } else {
        eprintln!("  ✗ {} VIOLATIONS detected:", violations.len());
        for (idx, contribs, mask) in violations.iter().take(5) {
            eprintln!("    node {}: fold_mask=0b{:0b}, contributions={:?}",
                idx, mask, contribs);
        }
        if violations.len() > 5 {
            eprintln!("    ... and {} more", violations.len() - 5);
        }
        eprintln!("");
        eprintln!("  EXPECTED: at a flop-zone player node with no-fold mask, all");
        eprintln!("  players are still active and must have matched the same bet");
        eprintln!("  level (all contributions equal). UNEQUAL contributions mean");
        eprintln!("  the preflop builder let a player see the flop without matching");
        eprintln!("  the high bet — a FREE FLOP, the wrong-game finding from the");
        eprintln!("  earlier chip trace.");
        panic!("seam test FAILED: {} flop entries have unequal contributions \
                with no folds (wrong-game / free-flop bug). See log above.",
            violations.len());
    }
}

#[test]
fn b4_pre_seam_hu_preflop_to_flop_chip_conservation() {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![2, 1],  // BB=player 0, SB=player 1
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
    let tree = build_tree(&cfg).expect("HU preflop builds");
    verify_no_fold_flop_entries(&tree, "HU preflop, 1 bet, 0 raise (legacy button)");
}

#[test]
fn b4_pre_seam_hu_preflop_with_raise_chip_conservation() {
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
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("HU preflop builds");
    verify_no_fold_flop_entries(&tree, "HU preflop, 1 bet + 1 raise (legacy button)");
}

#[test]
fn b4_pre_seam_6max_preflop_to_flop_chip_conservation() {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; 6],
        initial_contributions: vec![1, 2, 0, 0, 0, 0],  // SB=p0, BB=p1
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
            max_bets_per_street: None,  // UTG = (5+3)%6 = 2 acts first
    };
    let tree = build_tree(&cfg).expect("6-max preflop builds");
    verify_no_fold_flop_entries(&tree, "6-max preflop, button=5, 1 bet, 0 raise");
}
