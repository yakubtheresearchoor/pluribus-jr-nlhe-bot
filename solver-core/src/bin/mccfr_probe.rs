//! BATCHED MCCFR vs DCFR — WALL-ESCAPE MEASUREMENT (2026-06-17, branch
//! mccfr-cosolve-probe). Does external-sampling MCCFR's per-iter cost skip the
//! nb^num_opp multiway-showdown enumeration that makes the DCFR connected solve
//! cost explode with player count? Measurement branch, NOT production. The DCFR
//! path (BucketedFlopCfr) is untouched; this binary measures it head-to-head
//! against a batched external-sampling MCCFR on the IDENTICAL shrunk game.
//!
//! This file (step 1): the shrunk-game harness + the DCFR per-iter baseline by
//! live-count, establishing the wall (DCFR iter time should grow steeply with
//! live-count as nb^num_opp). MCCFR engine + the full-tree-enumerated
//! exploitability anchor land in subsequent steps.
//!
//! SHRINK KNOBS (recorded explicitly, env-overridable):
//!   MC_NB     buckets per street (default 6)
//!   MC_ITERS  DCFR iters for the per-iter timing (default 16)
//!   MC_NT/NR  turn/river runout samples (default 1/1 — single runout)
//! Both algorithms run on the game these knobs define; identical footing.

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// The shrunk game for one live-count: small bucketed flop→river subgame on a
/// single canonical flop with an nt×nr runout. Returns the tree + table +
/// bucketing so DCFR and (later) MCCFR run on the IDENTICAL object.
pub struct ShrunkGame {
    pub tree: FlatTree,
    pub game: FlopStartGame,
    pub bk: FlopBucketing,
    pub live: u8,
    pub nb: usize,
}

pub fn build_shrunk(live: u8, nb: usize, nt: usize, nr: usize) -> ShrunkGame {
    let spec = production_game_v1();
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let ptable = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let canonical = ptable.canonical_flops[0];
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    // nt turn samples, nr river samples per turn (deterministic positions).
    let tp: &[usize] = match nt { 1 => &[12], 2 => &[12, 36], _ => &[12] };
    let turns: Vec<Card> = tp.iter().map(|&p| deck[p]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turns {
        let rd: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        let rp: &[usize] = match nr { 1 => &[10], 2 => &[10, 30], _ => &[10] };
        river_decks[tc as usize] = rp.iter().map(|&p| rd[p]).collect();
    }
    let tree = build_tree(&spec.flop_seam_config(live, 2, 12, bets)).unwrap();
    let table = FlopChanceTable::build_full_nh_sampled(canonical, live, &turns, &river_decks);
    let bk = FlopBucketing::quantile(&table, nb);
    let game = FlopStartGame::new(table);
    ShrunkGame { tree, game, bk, live, nb }
}

/// DCFR per-iter timing. live==2 is exact HU (FlopStartVectorCfr, O(nh^2)
/// showdown — the bucketed designs force HU to exact); live>=3 is the bucketed
/// multiway wall (BucketedFlopCfr Design1Collapsed, O(nb^num_opp) showdown).
fn dcfr_per_iter(g: &ShrunkGame, iters: u32) -> (f64, usize) {
    use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
    let nn = g.tree.num_nodes();
    if g.live == 2 {
        let mut s = FlopStartVectorCfr::new(&g.tree, g.game.table());
        s.run(&g.tree, &g.game, 1);
        let t0 = Instant::now();
        s.run(&g.tree, &g.game, iters);
        return (t0.elapsed().as_secs_f64() / iters as f64, nn);
    }
    let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
    s.set_terminal_design(TerminalDesign::Design1Collapsed);
    s.run(&g.tree, &g.game, &g.bk, 1);
    let t0 = Instant::now();
    s.run(&g.tree, &g.game, &g.bk, iters);
    (t0.elapsed().as_secs_f64() / iters as f64, nn)
}

fn main() {
    let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let iters: u32 = std::env::var("MC_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let nt: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let nr: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("SHRUNK GAME: nb={nb} runout={nt}x{nr} | DCFR per-iter baseline (the nb^num_opp wall)\n");
    println!("{:>5} {:>8} {:>10} {:>12} {:>14}", "live", "num_opp", "nodes", "DCFR s/iter", "nb^num_opp");
    for live in 2u8..=5 {
        let g = build_shrunk(live, nb, nt, nr);
        let (per, nodes) = dcfr_per_iter(&g, iters);
        let wall = (nb as f64).powi((live - 1) as i32);
        println!("{:>5} {:>8} {:>10} {:>12.4} {:>14.0}", live, live - 1, nodes, per, wall);
    }
    println!("\n(if DCFR s/iter tracks nb^num_opp, that's the wall MCCFR aims to skip by sampling.)");
}
