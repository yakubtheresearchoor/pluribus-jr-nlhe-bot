//! PHASE S0 — MULTIWAY reduction check (2026-06-13, pre-S1). Kuhn
//! proved QRE→Nash HEADS-UP, but the per-seat λ threading was exercised
//! only in the trivial one-opponent case, and the multiway families are
//! the entire reason QRE exists here. Multiway also has NO Nash
//! convergence guarantee (the foundational caveat), so "QRE reduces to
//! Nash" must be CONFIRMED in the np=3 regime, not assumed — if it
//! reduces less cleanly multiway, that's a finding wanted now at S0, not
//! discovered in S2 when a weird result is un-attributable.
//!
//! Minimal np=3 game (all 3 seats decide; mixed equilibrium so λ bites):
//! 3 cards {0,1,2} dealt as a permutation, each antes 1 (pot 3).
//!   P0: CHECK → 3-way showdown | BET(+1) →
//!     P1: FOLD → P2: FOLD (P0 wins) / CALL (P0 vs P2)
//!         CALL → P2: FOLD (P0 vs P1) / CALL (3-way)
//! Showdown: highest card among players at the max contribution wins the
//! pot; folders lose their contribution. Self-consistent zero-sum game
//! (the reduction needs only that the SAME terminal feeds Nash, QRE, and
//! best-response — not poker-accuracy).

use solver_core::solver::game::GameSpec;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::tree::action::BoardState;
use solver_core::tree::flat::{FlatNode, FlatTree};

const NH: usize = 3; // cards 0,1,2

struct ThreePlayerGame;

impl GameSpec for ThreePlayerGame {
    fn num_hands(&self, _p: u8) -> usize { NH }
    fn initial_weight(&self, _p: u8) -> Vec<f32> { vec![1.0; NH] }
    fn chance_probability(&self, _o: usize, _h: usize) -> f32 { 0.0 }

    fn evaluate_terminal(
        &self, traverser: u8, node_idx: usize, tree: &FlatTree, cfreach: &[Vec<f32>],
    ) -> Vec<f32> {
        let t = traverser as usize;
        let c: Vec<i32> = (0..3).map(|p| tree.get_contribution(node_idx, p as u8)).collect();
        let pot: i32 = c.iter().sum();
        let maxc = *c.iter().max().unwrap();
        let in_player = |p: usize| c[p] == maxc; // at max contribution = not folded
        let mut cfv = vec![0.0f32; NH];

        // Sum over the two opponents' card assignments (a permutation of
        // the two cards not held by the traverser).
        let opps: Vec<usize> = (0..3).filter(|&p| p != t).collect();
        let (o1, o2) = (opps[0], opps[1]);
        for h in 0..NH {
            let mut acc = 0.0f32;
            for h1 in 0..NH {
                if h1 == h { continue; }
                for h2 in 0..NH {
                    if h2 == h || h2 == h1 { continue; }
                    let reach = cfreach[o1][h1] * cfreach[o2][h2];
                    if reach == 0.0 { continue; }
                    // Cards by player.
                    let mut card = [0usize; 3];
                    card[t] = h; card[o1] = h1; card[o2] = h2;
                    // Winner = highest card among in-players.
                    let winner = (0..3).filter(|&p| in_player(p))
                        .max_by_key(|&p| card[p]).unwrap();
                    let payoff = if winner == t { (pot - c[t]) as f32 } else { -(c[t] as f32) };
                    acc += reach * payoff;
                }
            }
            cfv[h] = acc;
        }
        cfv
    }
}

/// Build the np=3 one-bet game tree. Antes 1 each (pot 3 baseline).
fn build_3p_tree() -> FlatTree {
    let mut tree = FlatTree::new(3, 3, vec![0, 0, 0], 0.0, 0.0);
    let player = |pl: u8| FlatNode::player(pl, BoardState::River, 0);
    let set = |t: &mut FlatTree, n: usize, c: [i32; 3]| {
        for p in 0..3 { t.set_contribution(n, p as u8, c[p]); }
    };

    // root: P0 decides. antes [1,1,1].
    let n0 = tree.alloc_node(player(0)); set(&mut tree, n0, [1,1,1]);

    // P0 CHECK → 3-way showdown terminal (all in at 1).
    let t_check = tree.alloc_node(FlatNode::terminal()); set(&mut tree, t_check, [1,1,1]);

    // P0 BET (+1 → 2): P1 decides.
    let n1 = tree.alloc_node(player(1)); set(&mut tree, n1, [2,1,1]);
    //   P1 FOLD (stays at 1): P2 decides.
    let n2 = tree.alloc_node(player(2)); set(&mut tree, n2, [2,1,1]);
    //     P2 FOLD: P0 wins uncontested. [2,1,1]
    let t_a = tree.alloc_node(FlatNode::terminal()); set(&mut tree, t_a, [2,1,1]);
    //     P2 CALL (+1 → 2): P0 vs P2 (P1 folded). [2,1,2]
    let t_b = tree.alloc_node(FlatNode::terminal()); set(&mut tree, t_b, [2,1,2]);
    //   P1 CALL (+1 → 2): P2 decides. [2,2,1]
    let n3 = tree.alloc_node(player(2)); set(&mut tree, n3, [2,2,1]);
    //     P2 FOLD: P0 vs P1. [2,2,1]
    let t_c = tree.alloc_node(FlatNode::terminal()); set(&mut tree, t_c, [2,2,1]);
    //     P2 CALL: 3-way. [2,2,2]
    let t_d = tree.alloc_node(FlatNode::terminal()); set(&mut tree, t_d, [2,2,2]);

    tree.set_children(n0, vec![t_check as u32, n1 as u32]);     // check, bet
    tree.set_children(n1, vec![n2 as u32, n3 as u32]);          // fold, call
    tree.set_children(n2, vec![t_a as u32, t_b as u32]);        // fold, call
    tree.set_children(n3, vec![t_c as u32, t_d as u32]);        // fold, call
    tree.compute_levels();
    tree
}

#[test]
fn s0_multiway_qre_reduces_to_nash() {
    let tree = build_3p_tree();
    let game = ThreePlayerGame;
    const ITERS: u32 = 30000;

    // Nash reference: regret matching (lambda = None). np=3 has no Nash
    // GUARANTEE, but regret-matching converges empirically on these
    // games — that empirical Nash is exactly what QRE must reduce to.
    let mut nash = CpuMccfr::new(&tree, vec![NH; 3]);
    nash.run(&tree, &game, ITERS);
    let nash_expl = {
        let p = StrategyProfile::from_usize_offsets(nash.cum_strategy_slice(), nash.node_offsets(), NH);
        exploitability(&tree, &game, &p)
    };
    eprintln!("\n═══ S0 MULTIWAY QRE→Nash (np=3, {ITERS} iters) ═══");
    eprintln!("NASH (regret matching): expl {nash_expl:.5}");

    eprintln!("QRE λ-curve (ALL 3 seats at λ — exercises per-seat threading):");
    let mut hi_expl = f32::INFINITY;
    for &lam in &[1.0f32, 3.0, 10.0, 30.0, 100.0, 300.0] {
        let mut qre = CpuMccfr::new(&tree, vec![NH; 3]);
        qre.set_lambda(vec![lam; 3]); // per-seat, all high → Nash limit
        qre.run(&tree, &game, ITERS);
        let expl = {
            let p = StrategyProfile::from_usize_offsets(qre.cum_strategy_slice(), qre.node_offsets(), NH);
            exploitability(&tree, &game, &p)
        };
        eprintln!("  λ=[{lam:.0},{lam:.0},{lam:.0}]: expl {expl:.5}");
        hi_expl = expl;
    }

    // PER-SEAT routing check: a LOW λ on seat 2 only must change the
    // result (proves λ is threaded per-seat, not globally) — seat 2
    // plays quantally-randomly while 0,1 stay near-Nash.
    let mut mixed = CpuMccfr::new(&tree, vec![NH; 3]);
    mixed.set_lambda(vec![300.0, 300.0, 1.0]);
    mixed.run(&tree, &game, ITERS);
    let mixed_expl = {
        let p = StrategyProfile::from_usize_offsets(mixed.cum_strategy_slice(), mixed.node_offsets(), NH);
        exploitability(&tree, &game, &p)
    };
    eprintln!("PER-SEAT routing: λ=[300,300,1] expl {mixed_expl:.5} (must differ from all-300)");

    // REDUCTION: all-seats-high-λ QRE reduces to the empirical np=3 Nash.
    assert!(nash_expl < 0.05, "np=3 regret-matching didn't converge (expl {nash_expl:.5}) — no Nash to reduce TO");
    assert!(hi_expl < nash_expl + 0.03, "MULTIWAY QRE high-λ expl {hi_expl:.5} not near Nash {nash_expl:.5} — reduction FAILED at np=3");
    // Per-seat λ genuinely routes: a rough seat-2 must raise exploitability
    // meaningfully above the all-high-λ near-Nash.
    assert!(mixed_expl > hi_expl + 0.02, "per-seat λ not routing: [300,300,1] ({mixed_expl:.5}) ≈ all-300 ({hi_expl:.5})");
    eprintln!("S0 MULTIWAY PASSED: QRE reduces to np=3 Nash; per-seat λ routes correctly.");
}
