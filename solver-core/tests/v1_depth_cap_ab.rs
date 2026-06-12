//! RAISE-DEPTH CAP A/B (2026-06-12, user directive): the cap VALUE is
//! MEASURED, not assumed — the postflop saga's rule that action-
//! abstraction changes get validated (pruning the wrong action creates
//! catastrophic exploitability). Pluribus's argument says raise-WAR
//! depth is naturally shallow because the equilibrium folds most hands
//! early; this instrument checks whether capping OUR depth at k costs
//! EV against an opponent who keeps FULL depth.
//!
//! Measurement design (no action-translation lift needed): the
//! ASYMMETRIC game — seat 0's aggression capped at k bets/street
//! (BetCap::seat_only), seat 1 unrestricted. Every defensive infoset
//! seat 0 needs (facing deep raises) stays IN the tree, so the
//! equilibrium is well-defined and the cap's cost is exactly
//! EV_0(k) − EV_0(∞) at the root. The smallest k whose gap is within
//! the solve-noise floor is defense-complete. Run mirrored (seat 1
//! capped) for the position average.
//!
//! NAMED PROXY: the game is HU FLOP-START at deep SPR (pot 4, stacks
//! 200 ⇒ SPR 50; 1.0×pot escalation reaches num_bets ≈ 5-6
//! naturally), because the four-zone preflop runtime is not built yet.
//! The escalation structure (raise-war depth per street) is the same
//! question; re-confirm on the preflop street via the harness
//! head-to-head once the preflop runtime exists.
//!
//! Solve-quality control: per-arm exploitability via brute-force best
//! response (both seats) must be small and comparable across arms,
//! otherwise EV differences are solve noise, not structure. Noise
//! floor: the uncapped arm re-solved at a different iteration count.
//!
//! ═══ MEASURED VERDICT 2026-06-12 (this instrument + refinement) ═══
//! Deep shape (pot 2, stacks 200, 2-rung ladder, natural depth 6):
//!   k=1: gap 5.5e-4 / 2.1e-3  — REAL (>> every bar)
//!   k=2: gap 3.4e-5 / 3.4e-6  — REAL (holds above bars tightened to
//!        2.7e-6 / 4.0e-7 at 12k iters; resolved by the refinement)
//!   k=3: gap −1.8e-9 / +2.6e-9 — ZERO at solve precision (bar 5e-7)
//!   k=4: gap 0 (first shape: k=4 tree bit-identical to uncapped)
//! ⇒ PRODUCTION CAP = 3 (smallest defense-complete). Census payoff:
//! v1 preflop tree 4.33M → 1.90M nodes at cap 3 (cap 4: 3.92M — depth
//! ≥5 barely exists naturally, the cap-4 knob is near-vacuous).

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetCap, BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const NH: usize = 30;
const NP: u8 = 2;

fn build_table() -> FlopChanceTable {
    let board: Vec<Card> =
        ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
            continue;
        }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let ranges: Vec<Vec<f32>> = (0..NP as usize)
        .map(|_| {
            let mut r = vec![0.0f32; NUM_POSSIBLE_HANDS];
            for &h in &chosen {
                r[h as usize] = 1.0;
            }
            r
        })
        .collect();
    let turn_cards = vec![card_from_str("3c").unwrap() as u8];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![card_from_str("5s").unwrap() as u8];
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

fn cfg(cap: Option<BetCap>) -> TreeConfig {
    TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        // Deep SPR + a slow escalation rung so raise wars reach
        // num_bets 5-6 naturally — the cap must BIND for 3-vs-4 to be
        // a measurement (first shape tried: pot 4, single 1.0x rung —
        // natural depth only 3, k=3 was vacuously free).
        starting_pot: 2,
        starting_stacks: vec![200; NP as usize],
        initial_contributions: vec![0; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
        max_bets_per_street: cap,
    }
}

/// Max consecutive-aggression depth (num_bets high-water) per seat in
/// the tree, by walking with the builder's num_bets semantics.
fn max_depth_by_seat(tree: &FlatTree) -> [u32; 2] {
    use solver_core::tree::flat::{ACTION_LABEL_BET, ACTION_LABEL_RAISE};
    let mut max_by_seat = [0u32; 2];
    // stack: (node, bets_this_street)
    let mut stack: Vec<(usize, u32)> = vec![(0, 0)];
    while let Some((idx, bets)) = stack.pop() {
        let n = &tree.nodes[idx];
        if n.is_terminal() {
            continue;
        }
        let bets = if n.is_chance() { 0 } else { bets };
        let actor = tree.nodes[idx].player_id as usize;
        for &c in tree.node_children(idx) {
            let cl = tree.nodes[c as usize].action_label;
            let nb = if cl == ACTION_LABEL_BET || cl == ACTION_LABEL_RAISE {
                let b = bets + 1;
                if n.is_player() && actor < 2 && b > max_by_seat[actor] {
                    max_by_seat[actor] = b;
                }
                b
            } else {
                bets
            };
            stack.push((c as usize, nb));
        }
    }
    max_by_seat
}

struct Arm {
    label: String,
    ev: [f64; 2],
    expl: f64,
    nodes: usize,
}

fn solve_arm(label: &str, cap: Option<BetCap>, iters: u32) -> Arm {
    let tree = build_tree(&cfg(cap)).expect("tree");
    let game = FlopStartGame::new(build_table());
    let mut s = FlopStartVectorCfr::new(&tree, game.table());
    s.run(&tree, &game, iters);
    let mut ev = [0.0f64; 2];
    let mut expl = 0.0f64;
    for p in 0..2u8 {
        let sv = s.strategy_value_debug(&tree, &game, p);
        let br = s.best_response_value_debug(&tree, &game, p);
        ev[p as usize] = sv.iter().map(|&v| v as f64).sum::<f64>() / sv.len() as f64;
        expl += br
            .iter()
            .zip(&sv)
            .map(|(&b, &v)| ((b - v) as f64).max(0.0))
            .sum::<f64>()
            / br.len() as f64;
    }
    Arm { label: label.to_string(), ev, expl, nodes: tree.nodes.len() }
}

#[test]
#[ignore = "depth-cap A/B measurement; run with --ignored --nocapture --release"]
fn depth_cap_ab() {
    const ITERS: u32 = 2000;

    // Natural depth of the uncapped game (does the cap even bind?).
    let t_un = build_tree(&cfg(None)).expect("tree");
    let nat = max_depth_by_seat(&t_un);
    eprintln!("uncapped tree: {} nodes, natural max bets/street by seat: {:?}", t_un.nodes.len(), nat);

    // Baseline + noise floor (same game, different iteration count).
    let base = solve_arm("uncapped", None, ITERS);
    let base_b = solve_arm("uncapped (noise re-solve)", None, ITERS + ITERS / 2);
    let noise = (base.ev[0] - base_b.ev[0]).abs().max((base.ev[1] - base_b.ev[1]).abs());
    eprintln!(
        "baseline: EV {:?} expl {:.3e} | noise floor (iter-count jitter): {:.3e}",
        base.ev, base.expl, noise
    );

    // Capped arms: seat 0 capped (seat 1 free), and mirrored.
    for k in [1u8, 2, 3, 4] {
        let a0 = solve_arm(&format!("k={k} seat0-capped"), BetCap::seat_only(k, 0), ITERS);
        let a1 = solve_arm(&format!("k={k} seat1-capped"), BetCap::seat_only(k, 1), ITERS);
        let gap0 = base.ev[0] - a0.ev[0]; // EV the capped seat lost
        let gap1 = base.ev[1] - a1.ev[1];
        eprintln!(
            "k={k}: capped-seat EV gap (s0 {gap0:+.4e}, s1 {gap1:+.4e}) vs noise {noise:.3e} \
             | expl (a0 {:.3e}, a1 {:.3e}) | nodes {} / {}",
            a0.expl, a1.expl, a0.nodes, a1.nodes
        );
    }
    eprintln!(
        "verdict rule: the honest error bar is the per-arm EXPLOITABILITY (an \
         eps-equilibrium pins the game value to within eps), not iteration \
         jitter — smallest k with both gaps within the expl bar is \
         defense-complete for this game shape"
    );
}

/// Refinement: resolve k=2 vs k=3 specifically — at 2000 iters the
/// k=2 gap (3.4e-5) sat BELOW the exploitability bar (6.5e-5),
/// unresolvable. Longer solves shrink the bar under the gap (or the
/// gap under everything).
#[test]
#[ignore = "depth-cap refinement (longer solves); run with --ignored --nocapture --release"]
fn depth_cap_k2_refinement() {
    const ITERS: u32 = 12_000;
    let base = solve_arm("uncapped", None, ITERS);
    eprintln!("baseline: EV {:?} expl {:.3e}", base.ev, base.expl);
    for k in [2u8, 3] {
        let a0 = solve_arm(&format!("k={k} s0"), BetCap::seat_only(k, 0), ITERS);
        let a1 = solve_arm(&format!("k={k} s1"), BetCap::seat_only(k, 1), ITERS);
        eprintln!(
            "k={k}: gaps (s0 {:+.4e}, s1 {:+.4e}) | expl bar (base {:.3e}, a0 {:.3e}, a1 {:.3e})",
            base.ev[0] - a0.ev[0],
            base.ev[1] - a1.ev[1],
            base.expl,
            a0.expl,
            a1.expl
        );
    }
}
