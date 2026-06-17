//! LIVE-2 REAL-TIME SEARCH LATENCY (2026-06-15): how long does it take to
//! SOLVE a live-2 flop spot LIVE — i.e. the deployment search, not the fill.
//! The fill bakes a 1×1 sampled runout; a real-time resolve must cover the
//! FULL runout (every turn × every river) at full nh. This probe builds that
//! full-runout subgame and times the exact CPU solver per iteration, so we can
//! project the act latency at any convergence depth. (FlopStartVectorCfr is
//! CPU-only — no GPU path exists — so this is the CPU number; the "~12.5s GPU"
//! figure was a projection on an UNBUILT path.)
//!
//! Run: cargo test -p solver-core --release --test live2_search_latency -- \
//!        --ignored --nocapture   (L2_ITERS=N to change the timed iter count)

use solver_core::card::{card_from_str, Card};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use std::time::Instant;

#[test]
#[ignore = "live-2 real-time search latency; --ignored --nocapture --release"]
fn live2_search_latency() {
    let spec = production_game_v1();
    // A representative flop-entry spot (commit/pot ~ a mid postflop node).
    let (commit, pot) = (10i32, 32i32);
    let tree = build_tree(&spec.flop_seam_config(
        2, commit, pot,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
    )).unwrap();
    eprintln!("\n═══ LIVE-2 REAL-TIME SEARCH LATENCY (commit {commit}, pot {pot}) ═══");
    eprintln!("seam tree: {} nodes", tree.num_nodes());

    // FULL RUNOUT: every remaining card as a turn, every remaining-after-turn
    // card as a river. This is what a live resolve must actually search.
    let board: [Card; 3] = ["Ks", "7d", "2c"].map(|s| card_from_str(s).unwrap());
    let dead: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let turns: Vec<u8> = (0..52u8).filter(|c| dead & (1u64 << c) == 0).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turns {
        river_decks[tc as usize] =
            (0..52u8).filter(|&c| dead & (1u64 << c) == 0 && c != tc).collect();
    }
    let n_runouts: usize = turns.iter().map(|&tc| river_decks[tc as usize].len()).sum();
    eprintln!("full runout: {} turns × ~{} rivers = {} (turn,river) outcomes",
        turns.len(), river_decks[turns[0] as usize].len(), n_runouts);

    let t_build = Instant::now();
    let table = FlopChanceTable::build_full_nh_sampled(board, 2, &turns, &river_decks);
    let nh = table.num_valid;
    let game = FlopStartGame::new(table);
    let build_s = t_build.elapsed().as_secs_f64();
    eprintln!("nh = {nh} | table build {build_s:.2}s (one-time per resolve)");

    let iters: u32 = std::env::var("L2_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let mut s = FlopStartVectorCfr::new(&tree, game.table());
    let t0 = Instant::now();
    s.run(&tree, &game, iters);
    let solve_s = t0.elapsed().as_secs_f64();
    let per_iter = solve_s / iters as f64;
    eprintln!("\nsolved {iters} iters in {solve_s:.2}s → {per_iter:.3}s / iter");

    eprintln!("\nACT LATENCY = build + iters×per_iter (+ strategy read, negligible):");
    for &it in &[20u32, 34, 60, 100, 200] {
        let lat = build_s + it as f64 * per_iter;
        eprintln!("  @{it:>3} iters: {lat:6.1}s");
    }
    eprintln!("\nbudget check: a live decision wants ≲ a few seconds. \
        If even @20 iters ≫ that, exact CPU resolve is not viable live → \
        bank live-2, or depth-limit the search.");
}
