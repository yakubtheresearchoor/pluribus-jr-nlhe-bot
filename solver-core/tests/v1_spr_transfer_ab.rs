//! WITHIN-BUCKET SPR TRANSFER A/B (2026-06-12): the cell-width
//! analysis found production cost is ~119 SPR-bucket solves instead of
//! 9,521 cells — but that banks a BUCKETING CLAIM, and bucketing
//! claims get the exploitability gate before they're trusted (the
//! standing discipline). This instrument measures the claim directly:
//!
//!   solve the bucket REPRESENTATIVE's game → transplant its average
//!   strategy into a MEMBER cell's game (same tree shape — the
//!   tree-shape collapse check makes the buffers layout-identical) →
//!   exploitability of the transplanted strategy in the member game,
//!   against the member's OWN equilibrium exploitability as the bar.
//!
//!   transfer residual = expl(transplant in member) − expl(member eq)
//!
//! Pairs are chosen at MAXIMUM within-bin SPR separation (worst case
//! for the bucket), for log2-SPR widths 0.25 (the 119-bucket policy)
//! and 0.5 (the 66-bucket policy). HU cells with the exact solver
//! (BR machinery exists); multiway transfer inherits the same
//! geometry — re-confirm via harness head-to-head at production.
//! Residuals reported in raw chips/hand and as a fraction of the
//! member pot.
//!
//! ═══ MEASURED VERDICT 2026-06-12 ═══
//!   width 0.25, worst pair (SPR ratio 1.13×):  0.132% of pot
//!   width 0.25, half distance (1.07×):         0.046% of pot
//!     → residual ~ distance^1.6: bin-CENTER representatives cut the
//!       worst case ~3× (≈0.04-0.05% pot)
//!   width 0.25, 3-bet family (1.10×):          0.017% of pot
//!     (low-SPR families transfer better — less future betting)
//!   width 0.5: SHAPE MISMATCH at the edge pair — 66-bucket policy
//!     REJECTED (not single-shape; transfer undefined without action
//!     translation).
//! Bars: the baseline blueprint's own convergence was median 0.122% /
//! p90 0.372% of pot, and the B=8 abstraction error is ~7-10% pot at
//! research scale. The worst-pair transfer residual ≈ the convergence
//! MEDIAN, and centered-rep residual is well under it.
//! ⇒ POLICY VALIDATED: 119 buckets (log2-SPR width 0.25), bin-CENTER
//! representatives, all-in cells free, the 6 multi-shape buckets
//! shape-split (→ ~125 solves). NAMED approximation, residual
//! measured here at HU research scale; production confirmation rides
//! the harness head-to-head like every other A/B.
//!
//! RESIDUAL RISK (user-flagged 2026-06-12): these are HU numbers, and
//! the families the policy most needs to hold for are live-5/6 (96%
//! of the GPU bill) — multiway pots have more future betting
//! structure, so within-bucket transfer is most likely to be looser
//! exactly where solve cost concentrates. THE HARNESS CONFIRMATION
//! MUST BE STRATIFIED BY FAMILY: a transfer that is free HU and loose
//! at live-6 would be invisible in a pooled number.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

const NH: usize = 30;
const ITERS: u32 = 2000;

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
    let ranges: Vec<Vec<f32>> = (0..2usize)
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
        &board, &ranges, 2, &chosen, &turn_cards, &river_decks,
    )
}

struct Solved {
    solver: FlopStartVectorCfr,
    tree: solver_core::tree::flat::FlatTree,
    game: FlopStartGame,
    own_expl: f64,
}

fn solve_cell(commit: i32, pot: i32) -> Solved {
    let spec = production_game_v1();
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let cfg = spec.flop_seam_config(2, commit, pot, bets);
    let tree = build_tree(&cfg).expect("tree");
    let game = FlopStartGame::new(build_table());
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());
    solver.run(&tree, &game, ITERS);
    let own_expl = expl_of(&solver, &tree, &game);
    Solved { solver, tree, game, own_expl }
}

/// Mean per-hand positive BR gap, summed over both seats (chips/hand).
fn expl_of(s: &FlopStartVectorCfr, tree: &solver_core::tree::flat::FlatTree, game: &FlopStartGame) -> f64 {
    let mut total = 0.0f64;
    for p in 0..2u8 {
        let br = s.best_response_value_debug(tree, game, p);
        let sv = s.strategy_value_debug(tree, game, p);
        total += br
            .iter()
            .zip(&sv)
            .map(|(&b, &v)| ((b - v) as f64).max(0.0))
            .sum::<f64>()
            / br.len() as f64;
    }
    total
}

/// Transplant src's AVERAGE strategy (cum buffers) into dst's solver
/// and return the transplanted strategy's exploitability in dst's game.
fn transfer_expl(src: &Solved, dst: &mut Solved) -> f64 {
    assert_eq!(
        src.tree.nodes.len(),
        dst.tree.nodes.len(),
        "transfer requires identical tree shape (the bucket's collapse property)"
    );
    assert_eq!(src.solver.cum_strategy_flop().len(), dst.solver.cum_strategy_flop().len());
    dst.solver
        .cum_strategy_flop_mut()
        .copy_from_slice(src.solver.cum_strategy_flop());
    dst.solver
        .cum_strategy_turn_mut()
        .copy_from_slice(src.solver.cum_strategy_turn());
    dst.solver
        .cum_strategy_river_mut()
        .copy_from_slice(src.solver.cum_strategy_river());
    expl_of(&dst.solver, &dst.tree, &dst.game)
}

fn run_pair(width: f64, rep: (i32, i32), mem: (i32, i32)) {
    let spec = production_game_v1();
    let spr = |(c, p): (i32, i32)| (spec.stack - c) as f64 / p as f64;
    let r = solve_cell(rep.0, rep.1);
    let mut m = solve_cell(mem.0, mem.1);
    if r.tree.nodes.len() != m.tree.nodes.len() {
        eprintln!(
            "width {width}: rep (c{},p{}) {} nodes vs member (c{},p{}) {} nodes — \
             SHAPE MISMATCH: transfer undefined without action translation. \
             FINDING: this width's buckets are not single-shape; the bucket \
             policy must shape-split (or stay at width 0.25 where 113/119 \
             buckets are single-shape).",
            rep.0,
            rep.1,
            r.tree.nodes.len(),
            mem.0,
            mem.1,
            m.tree.nodes.len()
        );
        return;
    }
    let t = transfer_expl(&r, &mut m);
    let residual = t - m.own_expl;
    let pot = mem.1 as f64;
    eprintln!(
        "width {width}: rep (c{},p{}) SPR {:.2} → member (c{},p{}) SPR {:.2} \
         [ratio {:.2}×]\n  member own expl {:.3e} | transplant expl {:.3e} | \
         RESIDUAL {:+.3e} chips/hand = {:.4}% of member pot",
        rep.0,
        rep.1,
        spr(rep),
        mem.0,
        mem.1,
        spr(mem),
        spr(rep).max(spr(mem)) / spr(rep).min(spr(mem)),
        m.own_expl,
        t,
        residual,
        100.0 * residual / pot
    );
}

#[test]
#[ignore = "SPR transfer A/B; run with --ignored --nocapture --release"]
fn spr_transfer_ab() {
    eprintln!("\n════ within-bucket SPR transfer A/B (HU exact, nh={NH}, {ITERS} iters) ════");

    // Width 0.25 (the 119-bucket policy), max within-bin separation.
    // Single-raised family bin [SPR 11.31, 13.45):
    run_pair(0.25, (7, 15), (7, 17)); // 12.87 vs 11.35
    // Half-distance control (residual-vs-distance scaling: if ~d²,
    // bin-CENTER representatives cut the worst case ~4×):
    run_pair(0.25, (7, 15), (7, 16)); // 12.87 vs 12.06
    // 3-bet family bin [SPR 3.36, 4.00):
    run_pair(0.25, (24, 49), (22, 45)); // 3.59 vs 3.96

    // Width 0.5 (the 66-bucket policy), edge-to-edge:
    run_pair(0.5, (7, 17), (5, 13)); // 11.35 vs 15.0 (ratio 1.32×)

    eprintln!(
        "verdict rule: the bucket policy is trusted iff residuals sit at/below \
         the own-equilibrium expl bar (solve noise) or are small enough in \
         %-of-pot to be a named, accepted approximation (then measure again \
         at production via harness head-to-head)."
    );
}
