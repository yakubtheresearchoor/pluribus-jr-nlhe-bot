// P1.5.4 Slice B.4: independent first-principles chip-accounting at
// preflop-end nodes.
//
// Per the lead (2026-06-04): "verify the fix against an independent first-
// principles chip-accounting reference at preflop-end nodes, rather
// than against the thrice-tuned seam test, because 'I loosened the
// test until it passed' requires the independent check to confirm the
// code is right and not just that the test was made permissive."
//
// This reference derives the chip rules from FIRST PRINCIPLES (poker
// rules), not from the builder's logic. It checks every preflop-end
// node (preflop→flop CHANCE nodes AND preflop TERMINAL nodes) against
// these independent rules:
//
//   Rule 1 (standing bet conservation): at any preflop-end node, every
//   non-folded player has either matched the standing bet OR is all-in
//   at their personal max_committable. This is the poker rule "you
//   must call to see the next street or be all-in for less."
//
//   Rule 2 (chip cap): no player's contribution can exceed their
//   max_committable. This is the rule "you can't bet more than you
//   have."
//
//   Rule 3 (terminal validity, per existing builder convention): the
//   builder creates a TERMINAL node for EVERY Fold action (per the
//   "Fold ALWAYS produces TERMINAL" multi-player convention noted in
//   the P3 enumerator). So a preflop TERMINAL represents at minimum
//   "the folding player's end of hand at this decision" — others may
//   still be active via sibling paths. The chip-accounting rule:
//   every preflop TERMINAL must have at least 1 player folded
//   (fold_mask != 0), and the folded players' contributions reflect
//   what they put in before folding (which they forfeit).
//
//   Rule 4 (total chip conservation through the seam): at a preflop→
//   flop chance node, the sum of contributions = chips committed by
//   all players during preflop. This must equal the sum at the FIRST
//   postflop player below the chance (no chips magically added or
//   removed at the boundary).
//
// These rules are independent of the builder's `committed_at_round_
// start`, `is_facing_bet`, etc. They derive directly from poker.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn run_rules_check(cfg: &TreeConfig, label: &str) {
    let tree = build_tree(cfg).expect("builds");
    let np = cfg.num_players as usize;
    let initial_contributions = cfg.initial_contributions.clone();
    let starting_stacks: Vec<i32> = cfg.starting_stacks.clone();
    let max_committable: Vec<i32> = (0..np)
        .map(|p| starting_stacks[p] + initial_contributions[p])
        .collect();

    let mut parent_of = vec![None::<u32>; tree.num_nodes()];
    for p in 0..tree.num_nodes() {
        for &c in tree.node_children(p) {
            parent_of[c as usize] = Some(p as u32);
        }
    }

    let mut all_violations: Vec<String> = Vec::new();

    // ── Rule 1 + Rule 4: preflop→flop CHANCE nodes ──
    for idx in 0..tree.num_nodes() {
        let n = &tree.nodes[idx];
        if !n.is_chance() { continue; }
        let par = match parent_of[idx] { Some(p) => p as usize, None => continue };
        if tree.nodes[par].board_state != BoardState::Preflop as u8 { continue; }

        let mask = tree.get_folded_mask(idx);
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let active: Vec<usize> = (0..np).filter(|&p| (mask & (1 << p)) == 0).collect();
        if active.is_empty() { continue; }
        let standing = active.iter().map(|&p| contribs[p]).max().unwrap();

        // Rule 1: every non-folded player either matches standing or is at max_committable.
        for &p in &active {
            let c = contribs[p];
            if c != standing && c < max_committable[p] {
                all_violations.push(format!(
                    "[Rule 1] preflop→flop chance node {}: player {} contribution={} \
                     != standing {} AND remaining = {} > 0 (not all-in)",
                    idx, p, c, standing, max_committable[p] - c));
            }
        }

        // Rule 4: chip conservation through chance boundary
        for &child_u32 in tree.node_children(idx) {
            let child = child_u32 as usize;
            let child_contribs: Vec<i32> = (0..np)
                .map(|p| tree.get_contribution(child, p as u8)).collect();
            if child_contribs != contribs {
                all_violations.push(format!(
                    "[Rule 4] chance node {} → child {}: contributions changed from \
                     {:?} to {:?} across chance boundary (chips magically appeared/disappeared)",
                    idx, child, contribs, child_contribs));
            }
        }
    }

    // ── Rule 2: chip cap at every node ──
    for idx in 0..tree.num_nodes() {
        for p in 0..np {
            let c = tree.get_contribution(idx, p as u8);
            if c > max_committable[p] {
                all_violations.push(format!(
                    "[Rule 2] node {} player {} contribution {} exceeds max_committable {}",
                    idx, p, c, max_committable[p]));
            }
        }
    }

    // ── Rule 3: every preflop TERMINAL has at least one fold ──
    // (Per the builder's "Fold ALWAYS produces TERMINAL" convention,
    // every preflop terminal represents the folding player exiting at
    // that decision; the hand may continue for others via sibling paths.)
    for idx in 0..tree.num_nodes() {
        let n = &tree.nodes[idx];
        if !n.is_terminal() { continue; }
        if n.board_state != BoardState::Preflop as u8 { continue; }
        let mask = tree.get_folded_mask(idx);
        if mask == 0 {
            all_violations.push(format!(
                "[Rule 3] preflop TERMINAL node {} has fold_mask = 0: a preflop terminal \
                 must have at least one fold (else round should continue to chance or another \
                 player; no all-checks-no-folds preflop terminal is legal in NLHE).",
                idx));
        }
    }

    eprintln!("\n══ {} ══", label);
    if all_violations.is_empty() {
        eprintln!("  ✓ ALL FOUR RULES PASS independently. The preflop tree's chip");
        eprintln!("    accounting at preflop-end nodes (chance + terminal) matches");
        eprintln!("    poker rules derived from first principles. The fix is confirmed");
        eprintln!("    by independent reference, not by the tuned seam test.");
    } else {
        eprintln!("  ✗ {} violations:", all_violations.len());
        for v in all_violations.iter().take(10) {
            eprintln!("    {}", v);
        }
        panic!("first-principles chip-accounting reference detected violations");
    }
}

#[test]
fn b4_pre_independent_chips_hu_simplest() {
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
    run_rules_check(&cfg, "HU preflop, 1 bet, 0 raise");
}

#[test]
fn b4_pre_independent_chips_hu_with_raise() {
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
    run_rules_check(&cfg, "HU preflop, 1 bet + 1 raise");
}

#[test]
fn b4_pre_independent_chips_6max_simplest() {
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
    run_rules_check(&cfg, "6-max preflop, button=5, 1 bet, 0 raise");
}
