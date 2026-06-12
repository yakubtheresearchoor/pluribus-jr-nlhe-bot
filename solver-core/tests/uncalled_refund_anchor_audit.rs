//! UNCALLED-REFUND ANCHOR AUDIT (2026-06-12, ordered after the np=3
//! gate found the exact evaluator returning per-player-WRONG values
//! for fold terminals where a FOLDED player out-contributed the max
//! active player — the constant-payoff fast path, gated num_opp ≤ 2,
//! takes no uncalled refund; documented zero-sum-only).
//!
//! THE AUDIT QUESTION: is that state REACHABLE in legal trees? If not,
//! every gate that compares evaluators on real tree terminals is
//! unaffected and the hole exists only off the reachable manifold.
//!
//! CLAIM (proof sketch, verified empirically below): at every legal
//! terminal, max cumulative commitment is attained by a NON-folded
//! player. Argument: (a) folding is only legal facing a strictly
//! higher per-street commitment; (b) betting rounds start with all
//! non-all-in actives equalized (round-complete requires matched or
//! all-in); so at the moment of folding, the folder is STRICTLY below
//! the bettor they faced; (c) commitments only grow, and applying (a)+
//! (b) to each subsequent folder gives a chain that terminates at a
//! non-folder ≥ every folder. Hence folded > max-active is impossible,
//! and (refund = max(0, folded − max_active)) ≡ 0 on reachable states.
//!
//! This test VERIFIES the claim on every terminal of representative
//! built trees (all standing-gate configs + v1 seam cells + the v1
//! production preflop tree's flop-entry boundary) rather than trusting
//! the proof — if the builder ever produces a violating terminal, this
//! gate fails and every exact-referenced gate is suspect.
//!
//! AUDIT VERDICT for the gate inventory (2026-06-12): all gates that
//! feed the exact evaluator REAL tree terminals are safe iff this
//! invariant holds. Gates with SYNTHETIC fixtures were inspected for
//! folded-out-contributor states: only np3_bucketed_terminal_gate's
//! "lone survivor w/ uncalled refund" hits the hole, deliberately, as
//! a robustness case off the manifold, and it documents that the
//! bucketed arm (refund) and exact arm (no refund) disagree THERE by
//! design choice — the bucketed treatment matches clean-rules'
//! settle_pots (strict-max refunds to second-max); on the manifold the
//! strict max is never folded so both notions coincide.

use solver_core::tree::action::{
    production_game_v1, BetSize, BetSizeOptions, BoardState, TreeConfig,
};
use solver_core::tree::builder::{build_tree, build_tree_preflop_only};
use solver_core::tree::flat::{FlatTree, NODE_TYPE_CHANCE, NODE_TYPE_TERMINAL};

/// Assert the invariant on every terminal (and, for preflop-only
/// trees, every flop-entry chance leaf): max commit is non-folded.
fn assert_max_commit_active(tree: &FlatTree, np: usize, label: &str) -> (usize, i32) {
    let mut checked = 0usize;
    let mut max_folded_gap = i32::MIN; // max over terminals of (folded_max − active_max)
    for idx in 0..tree.nodes.len() {
        let n = &tree.nodes[idx];
        let boundary = n.node_type == NODE_TYPE_TERMINAL
            || (n.node_type == NODE_TYPE_CHANCE && n.num_children == 0);
        if !boundary {
            continue;
        }
        let mask = tree.get_folded_mask(idx);
        let mut active_max = i32::MIN;
        let mut folded_max = i32::MIN;
        for p in 0..np {
            let c = tree.get_contribution(idx, p as u8);
            if mask & (1 << p) != 0 {
                folded_max = folded_max.max(c);
            } else {
                active_max = active_max.max(c);
            }
        }
        if folded_max > i32::MIN {
            let gap = folded_max - active_max;
            max_folded_gap = max_folded_gap.max(gap);
            assert!(
                folded_max <= active_max,
                "{label}: node {idx} VIOLATES the anchor invariant — folded max \
                 {folded_max} > active max {active_max} (mask {mask:#08b}). The \
                 constant-payoff fast path's no-refund accounting is per-player- \
                 WRONG here; every gate using the exact evaluator on this tree \
                 is suspect.",
            );
        }
        checked += 1;
    }
    (checked, max_folded_gap)
}

fn oracle_bets() -> BetSizeOptions {
    BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] }
}

#[test]
fn uncalled_refund_anchor_audit() {
    let spec = production_game_v1();

    // 1. Every v1 seam-cell family shape (the priced cells).
    for &(live, commit, pot, label) in &[
        (3u8, 7i32, 29i32, "live-3 raised"),
        (4, 7, 29, "live-4 raised"),
        (5, 2, 10, "live-5 limp"),
        (6, 2, 12, "live-6 limp"),
        (6, 7, 42, "live-6 raised"),
    ] {
        let tree = build_tree(&spec.flop_seam_config(live, commit, pot, oracle_bets())).unwrap();
        let (n, gap) = assert_max_commit_active(&tree, live as usize, label);
        eprintln!("{label}: {n} terminals OK (max folded−active gap {gap})");
    }

    // 2. Raise-bearing config (raises are where folds chase bets).
    let raisey = TreeConfig {
        num_players: 4,
        initial_state: BoardState::Flop,
        starting_pot: 8,
        starting_stacks: vec![120; 4],
        initial_contributions: vec![0; 4],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
        max_bets_per_street: None,
    };
    let tree = build_tree(&raisey).unwrap();
    let (n, gap) = assert_max_commit_active(&tree, 4, "4p raises");
    eprintln!("4p raises: {n} terminals OK (max folded−active gap {gap})");

    // 3. Asymmetric blinds postflop (unequal starting commits).
    let blinds = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 0,
        starting_stacks: vec![100; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: oracle_bets(),
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
        max_bets_per_street: None,
    };
    let tree = build_tree(&blinds).unwrap();
    let (n, gap) = assert_max_commit_active(&tree, 6, "6p asym blinds");
    eprintln!("6p asym blinds: {n} terminals OK (max folded−active gap {gap})");

    // 4. THE v1 PRODUCTION PREFLOP TREE (cap 3, reduced ladder for
    //    gate speed — fold terminals AND flop-entry boundaries; blinds
    //    are forced partial commits, the likeliest violation source).
    use solver_core::tree::action::BetCap;
    let mut pre = spec.preflop_tree_config(BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: vec![
            BetSize::PotRelative(0.5),
            BetSize::PotRelative(1.0),
            BetSize::PotRelative(2.0),
        ],
    });
    pre.max_bets_per_street = BetCap::all(3);
    let tree = build_tree_preflop_only(&pre).unwrap();
    let (n, gap) = assert_max_commit_active(&tree, 6, "v1 preflop (cap 3)");
    eprintln!("v1 preflop (cap 3): {n} boundary nodes OK (max folded−active gap {gap})");

    eprintln!(
        "ANCHOR AUDIT PASSED: folded max-commit ≤ active max-commit at every \
         boundary — the exact fast path's no-refund accounting is correct on \
         the reachable manifold; the np3 gate's refund case is off-manifold \
         robustness only."
    );
}
