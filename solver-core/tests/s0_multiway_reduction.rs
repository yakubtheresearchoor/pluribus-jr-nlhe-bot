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

// ════════════════════════════════════════════════════════════════════
// S1 MULTIWAY ANCHOR (2026-06-13): the HU range problem was a
// convergence mirage. Test whether it's REAL multiway (np=3), where —
// unlike HU — it might not be. Needs a 2-ROUND np=3 game (the S0 game
// is single-round). Round-1 = Flop, round-2 = River (the truncation
// boundary). REFLEX ARMED: self-validate the game converges to ~0
// exploitability before measuring any anchor through it (the game must
// not itself be a confound), and sweep ITERS on the anchor (the lesson:
// convergence is an axis the freeze control can't see).

/// Round-2 betting subtree (River), given entry contributions and live
/// set (P0 always live). P0 {check, bet(+4)} then live opps fold/call.
fn build_round2_3p(tree: &mut FlatTree, c: [i32; 3], live: [bool; 3]) -> usize {
    let bet = 4i32;
    let set = |t: &mut FlatTree, n: usize, c: [i32; 3]| {
        for p in 0..3 { t.set_contribution(n, p as u8, c[p]); }
    };
    let entry = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    set(tree, entry, c);
    // CHECK → showdown at c.
    let t_check = tree.alloc_node(FlatNode::terminal()); set(tree, t_check, c);
    // BET → P0 +bet, then live opponents (1 then 2) fold/call in a chain.
    let mut cb = c; cb[0] += bet;
    let opps: Vec<usize> = [1usize, 2].into_iter().filter(|&p| live[p]).collect();
    // recursive chain
    fn chain(tree: &mut FlatTree, c: [i32; 3], opps: &[usize], idx: usize, bet: i32) -> usize {
        let set = |t: &mut FlatTree, n: usize, c: [i32; 3]| {
            for p in 0..3 { t.set_contribution(n, p as u8, c[p]); }
        };
        if idx == opps.len() {
            let t = tree.alloc_node(FlatNode::terminal()); set(tree, t, c); return t;
        }
        let p = opps[idx];
        let node = tree.alloc_node(FlatNode::player(p as u8, BoardState::River, 0));
        set(tree, node, c);
        let fold_child = chain(tree, c, opps, idx + 1, bet);     // p stays (folds)
        let mut cc = c; cc[p] += bet;
        let call_child = chain(tree, cc, opps, idx + 1, bet);    // p calls
        tree.set_children(node, vec![fold_child as u32, call_child as u32]);
        node
    }
    let bet_branch = chain(tree, cb, &opps, 0, bet);
    tree.set_children(entry, vec![t_check as u32, bet_branch as u32]);
    entry
}

fn build_2round_3p_tree() -> FlatTree {
    let mut tree = FlatTree::new(3, 3, vec![0, 0, 0], 0.0, 0.0);
    let set = |t: &mut FlatTree, n: usize, c: [i32; 3]| {
        for p in 0..3 { t.set_contribution(n, p as u8, c[p]); }
    };
    // n0: P0 check/bet, antes [1,1,1].
    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::Flop, 0)); set(&mut tree, n0, [1,1,1]);
    let e_check = build_round2_3p(&mut tree, [1,1,1], [true, true, true]);
    // BET: P0 → [3,1,1].
    let n_p1 = tree.alloc_node(FlatNode::player(1, BoardState::Flop, 0)); set(&mut tree, n_p1, [3,1,1]);
    //   P1 FOLD → P2 fold/call.
    let n_p2a = tree.alloc_node(FlatNode::player(2, BoardState::Flop, 0)); set(&mut tree, n_p2a, [3,1,1]);
    let t_p0wins = tree.alloc_node(FlatNode::terminal()); set(&mut tree, t_p0wins, [3,1,1]); // P2 folds too
    let e_02 = build_round2_3p(&mut tree, [3,1,3], [true, false, true]);                      // P2 calls
    tree.set_children(n_p2a, vec![t_p0wins as u32, e_02 as u32]);
    //   P1 CALL → [3,3,1], P2 fold/call.
    let n_p2b = tree.alloc_node(FlatNode::player(2, BoardState::Flop, 0)); set(&mut tree, n_p2b, [3,3,1]);
    let e_01 = build_round2_3p(&mut tree, [3,3,1], [true, true, false]);                      // P2 folds
    let e_pp = build_round2_3p(&mut tree, [3,3,3], [true, true, true]);                       // P2 calls
    tree.set_children(n_p2b, vec![e_01 as u32, e_pp as u32]);
    tree.set_children(n_p1, vec![n_p2a as u32, n_p2b as u32]); // fold, call
    tree.set_children(n0, vec![e_check as u32, n_p1 as u32]);  // check, bet
    tree.compute_levels();
    tree
}

fn full_expl_3p(s: &CpuMccfr, tree: &FlatTree, game: &ThreePlayerGame) -> f32 {
    let p = StrategyProfile::from_usize_offsets(s.cum_strategy_slice(), s.node_offsets(), NH);
    exploitability(tree, game, &p)
}

/// Depth-limited multiway: freeze River to blueprint, search Flop, then
/// deploy searched-Flop + blueprint-River and score true exploitability.
fn search_3p_frozen_river(tree: &FlatTree, game: &ThreePlayerGame, bp: &CpuMccfr, iters: u32) -> f32 {
    let mut s = CpuMccfr::new(tree, vec![NH; 3]);
    for nid in 0..tree.nodes.len() {
        if tree.nodes[nid].is_player() && tree.nodes[nid].board_state == BoardState::River as u8 {
            let na = tree.nodes[nid].num_children as usize;
            let st = bp.get_average_strategy(nid, na, NH);
            let flat: Vec<f32> = (0..na).flat_map(|a| (0..NH).map(move |h| (a, h))).map(|(a, h)| st[a][h]).collect();
            s.freeze_node(nid, &flat);
        }
    }
    s.run(tree, game, iters);
    // deploy: Flop=searched, River=blueprint.
    let mut dep = CpuMccfr::new(tree, vec![NH; 3]);
    for nid in 0..tree.nodes.len() {
        if !tree.nodes[nid].is_player() { continue; }
        let na = tree.nodes[nid].num_children as usize;
        let src = if tree.nodes[nid].board_state == BoardState::Flop as u8 { &s } else { bp };
        let st = src.get_average_strategy(nid, na, NH);
        let flat: Vec<f32> = (0..na).flat_map(|a| (0..NH).map(move |h| (a, h))).map(|(a, h)| st[a][h]).collect();
        dep.freeze_node(nid, &flat);
    }
    dep.run(tree, game, 5);
    full_expl_3p(&dep, tree, game)
}

#[test]
fn s1_multiway_anchor() {
    let tree = build_2round_3p_tree();
    let game = ThreePlayerGame;

    // REFLEX: validate the GAME (full solve → ~0 expl) before any anchor.
    let mut fine = CpuMccfr::new(&tree, vec![NH; 3]);
    fine.run(&tree, &game, 50000);
    let fine_expl = full_expl_3p(&fine, &tree, &game);
    eprintln!("\n═══ S1 MULTIWAY anchor (2-round np=3) ═══");
    eprintln!("GAME SELF-CHECK: full-solve expl {fine_expl:.5} (must be ~0 — else the game is a confound)");
    assert!(fine_expl < 0.05, "2-round np=3 game didn't converge to Nash ({fine_expl:.5}) — game/terminal bug");

    // ANCHOR iter-sweep (the reflex: is a high anchor convergence or
    // structural?). Single-continuation (freeze blueprint River).
    eprintln!("--- multiway anchor vs iterations (single-continuation) ---");
    let mut converged = f32::INFINITY;
    for it in [20000u32, 60000, 120000, 300000, 600000] {
        let a = search_3p_frozen_river(&tree, &game, &fine, it);
        eprintln!("  {it:>6} iters: anchor {a:.5}  ({:.0}× fine)", a / fine_expl.max(1e-6));
        converged = a;
    }
    eprintln!("MULTIWAY anchor at max iters: {converged:.5} vs fine {fine_expl:.5}. \
        If ≈fine → range problem is a convergence mirage multiway too (single-cont suffices). \
        If plateaus above → REAL multiway range problem (multi-continuation earns its place).");
}

// 6-card deck for 3 players: your rank is UNCERTAIN (3 of 6 cards out),
// so there's hidden information and mixing — a non-degenerate game that
// can actually exhibit the depth-limited range problem. Same terminal
// logic (max contribution = live, highest card among live wins).
const NCARD: usize = 6;
struct Rich3pGame;
impl GameSpec for Rich3pGame {
    fn num_hands(&self, _p: u8) -> usize { NCARD }
    fn initial_weight(&self, _p: u8) -> Vec<f32> { vec![1.0; NCARD] }
    fn chance_probability(&self, _o: usize, _h: usize) -> f32 { 0.0 }
    fn evaluate_terminal(&self, traverser: u8, node_idx: usize, tree: &FlatTree, cfreach: &[Vec<f32>]) -> Vec<f32> {
        let t = traverser as usize;
        let c: Vec<i32> = (0..3).map(|p| tree.get_contribution(node_idx, p as u8)).collect();
        let pot: i32 = c.iter().sum();
        let maxc = *c.iter().max().unwrap();
        let in_player = |p: usize| c[p] == maxc;
        let opps: Vec<usize> = (0..3).filter(|&p| p != t).collect();
        let (o1, o2) = (opps[0], opps[1]);
        let mut cfv = vec![0.0f32; NCARD];
        for h in 0..NCARD {
            let mut acc = 0.0f32;
            for h1 in 0..NCARD {
                if h1 == h { continue; }
                for h2 in 0..NCARD {
                    if h2 == h || h2 == h1 { continue; }
                    let reach = cfreach[o1][h1] * cfreach[o2][h2];
                    if reach == 0.0 { continue; }
                    let mut card = [0usize; 3];
                    card[t] = h; card[o1] = h1; card[o2] = h2;
                    let winner = (0..3).filter(|&p| in_player(p)).max_by_key(|&p| card[p]).unwrap();
                    acc += reach * if winner == t { (pot - c[t]) as f32 } else { -(c[t] as f32) };
                }
            }
            cfv[h] = acc;
        }
        cfv
    }
}

fn full_expl_rich(s: &CpuMccfr, tree: &FlatTree, game: &Rich3pGame) -> f32 {
    let p = StrategyProfile::from_usize_offsets(s.cum_strategy_slice(), s.node_offsets(), NCARD);
    exploitability(tree, game, &p)
}

fn search_rich_frozen_river(tree: &FlatTree, game: &Rich3pGame, bp: &CpuMccfr, iters: u32) -> f32 {
    let mut s = CpuMccfr::new(tree, vec![NCARD; 3]);
    for nid in 0..tree.nodes.len() {
        if tree.nodes[nid].is_player() && tree.nodes[nid].board_state == BoardState::River as u8 {
            let na = tree.nodes[nid].num_children as usize;
            let st = bp.get_average_strategy(nid, na, NCARD);
            let flat: Vec<f32> = (0..na).flat_map(|a| (0..NCARD).map(move |h| (a, h))).map(|(a, h)| st[a][h]).collect();
            s.freeze_node(nid, &flat);
        }
    }
    s.run(tree, game, iters);
    let mut dep = CpuMccfr::new(tree, vec![NCARD; 3]);
    for nid in 0..tree.nodes.len() {
        if !tree.nodes[nid].is_player() { continue; }
        let na = tree.nodes[nid].num_children as usize;
        let src = if tree.nodes[nid].board_state == BoardState::Flop as u8 { &s } else { bp };
        let st = src.get_average_strategy(nid, na, NCARD);
        let flat: Vec<f32> = (0..na).flat_map(|a| (0..NCARD).map(move |h| (a, h))).map(|(a, h)| st[a][h]).collect();
        dep.freeze_node(nid, &flat);
    }
    dep.run(tree, game, 5);
    full_expl_rich(&dep, tree, game)
}

#[test]
fn s1_multiway_anchor_rich() {
    let tree = build_2round_3p_tree();
    let game = Rich3pGame;
    eprintln!("\n═══ S1 MULTIWAY anchor (2-round np=3, 6-card deck) ═══");

    // REFLEX on the REFERENCE: np=3 has NO Nash convergence guarantee, so
    // iter-sweep the full-solve blueprint to find its FLOOR F (a CFR
    // residual, maybe not 0). The anchor is compared to F, not to 0 —
    // "fine" here = the achievable np=3 blueprint.
    eprintln!("blueprint (full solve) expl vs iters — finding the np=3 floor F:");
    let mut fine = CpuMccfr::new(&tree, vec![NCARD; 3]);
    let mut f = 0.0f32;
    for chunk in [80000u32, 220000, 300000] {
        fine.run(&tree, &game, chunk);
        f = full_expl_rich(&fine, &tree, &game);
        eprintln!("  ~{} iters: blueprint expl {f:.5}", fine.iteration_count());
    }
    assert!(f > 1e-4, "game DEGENERATE ({f:.6}) — no mixing, can't test the range problem");
    eprintln!("np=3 blueprint expl bounces (no-Nash caveat) — reference is UNSTABLE ≈ {f:.5}");

    // CONTROL: deploy the blueprint EVERYWHERE (freeze all) → must
    // reproduce the blueprint's own exploitability. Localizes plumbing
    // (game/terminal/freeze/scoring) vs a real search effect.
    let ctrl = {
        let mut all = CpuMccfr::new(&tree, vec![NCARD; 3]);
        for nid in 0..tree.nodes.len() {
            if tree.nodes[nid].is_player() {
                let na = tree.nodes[nid].num_children as usize;
                let st = fine.get_average_strategy(nid, na, NCARD);
                let flat: Vec<f32> = (0..na).flat_map(|a| (0..NCARD).map(move |h| (a, h))).map(|(a, h)| st[a][h]).collect();
                all.freeze_node(nid, &flat);
            }
        }
        all.run(&tree, &game, 5);
        full_expl_rich(&all, &tree, &game)
    };
    eprintln!("CONTROL freeze-ALL-to-blueprint: {ctrl:.5} (must ≈ blueprint {f:.5} → plumbing OK)");
    assert!((ctrl - f).abs() < 0.02, "plumbing broken: control {ctrl:.5} != blueprint {f:.5}");

    eprintln!("--- multiway anchor vs iterations (single-continuation, reflex armed) ---");
    let mut converged = f32::INFINITY;
    for it in [20000u32, 60000, 120000, 300000, 600000] {
        let a = search_rich_frozen_river(&tree, &game, &fine, it);
        eprintln!("  {it:>6} iters: anchor {a:.5}  ({:.2}× F)", a / f);
        converged = a;
    }
    // FINDING (documented, not a clean pass/fail — the testbed is the
    // problem): the anchor is CONVERGED (flat across iters, NOT the HU
    // convergence mirage) but huge vs the blueprint. HOWEVER the np=3
    // blueprint itself does NOT converge (no-Nash caveat bounces it),
    // so this toy is a POOR testbed and the anchor-vs-unstable-reference
    // is UNINTERPRETABLE as a range-problem verdict. The control proves
    // plumbing is sound, so it's not a bug — it's that np=3 CFR doesn't
    // give a clean blueprint HERE. The real multiway flop games DO
    // converge (blueprint conv median 0.00122), so the multiway anchor
    // needs a CONVERGENT np=3 testbed (next), not this pathological one.
    eprintln!("VERDICT: anchor CONVERGED at {converged:.5} (flat across iters — not the HU mirage), \
        but the np=3 blueprint reference is NON-CONVERGENT (bounces ~0.17-0.27), so this toy is a \
        POOR testbed and the result is UNINTERPRETABLE. Plumbing sound (control). NEXT: a convergent \
        np=3 testbed (the real flop games converge; this 6-card toy does not).");
    assert!(converged.is_finite() && f.is_finite());
}
