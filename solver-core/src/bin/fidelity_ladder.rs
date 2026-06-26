//! FIDELITY/TIMING LADDER (2026-06-18, dcfr-inplace-fidelity branch): measure the
//! per-family CFR cost as a function of bucket count B on the REAL cap-3 abstraction
//! (bet=pot, raises={0.5,1.0}pot, BetCap::all(3)), so we can find the highest B per
//! live-count that still fits the per-iter timing budget.
//!
//! The transferable quantity is the LADDER SHAPE (cost ratio across B); absolutes get
//! anchored to the banked ground-truth hours (joint_solve: live-4@B15=11h, live-5@B8=7.1h).
//! CPU BucketedFlopCfr + Design1Collapsed (the production terminal). Reports ms/iter.
//!
//! Env: FL_ITERS (default 6), FL_COMMIT (10), FL_POT (60), FL_LIVES (3,4,5), FL_BS (8,10,12,15).

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
use solver_core::tree::flat::MAX_NA_POSTFLOP;
#[cfg(feature = "metal")]
use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
#[cfg(feature = "metal")]
use solver_core::gpu_metal::context::MetalContext;
use solver_core::tree::builder::build_tree;

fn parse_list(key: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(key)
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<_>>())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn main() {
    let iters: u32 = std::env::var("FL_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let commit: i32 = std::env::var("FL_COMMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let pot: i32 = std::env::var("FL_POT").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let lives = parse_list("FL_LIVES", &[3, 4, 5]);
    let bs = parse_list("FL_BS", &[8, 10, 12, 15]);

    let spec = production_game_v1();
    // REAL cap-3 action set (mirrors mccfr_probe MC_REAL).
    let mrc = MAX_NA_POSTFLOP.saturating_sub(2);
    let bets = BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect(),
    };

    let ptable = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let canonical = ptable.canonical_flops[0];
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let turns = vec![deck[0]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[deck[0] as usize] = vec![deck[1]];

    #[cfg(feature = "metal")]
    let engine = "GPU (BucketedNativeGpu — production multiway engine)";
    #[cfg(not(feature = "metal"))]
    let engine = "CPU (BucketedFlopCfr)";
    eprintln!(
        "FIDELITY LADDER [{engine}] — real cap-3 (bet=pot, raises={:?}, cap3), cell commit={commit} pot={pot}, {iters} iters",
        (0..mrc).map(|i| 0.5 + 0.5 * i as f64).collect::<Vec<_>>()
    );
    eprintln!("{:<8} {:>6} {:>6} {:>12} {:>12}", "live", "B", "nh", "ms/iter", "x vs B[0]");

    #[cfg(feature = "metal")]
    let ctx = MetalContext::new().expect("Metal context (set OUT_DIR to dir holding solver.metallib)");

    for &live in &lives {
        let mut cfg = spec.flop_seam_config(live as u8, commit, pot, bets.clone());
        cfg.max_bets_per_street = BetCap::all(3);
        let tree = build_tree(&cfg).unwrap();
        let tbl = FlopChanceTable::build_full_nh_sampled(canonical, live as u8, &turns, &river_decks);
        let nh = tbl.num_valid;
        let game = FlopStartGame::new(tbl);

        let mut base_ms = 0.0f64;
        for (bi, &nb) in bs.iter().enumerate() {
            let bk = FlopBucketing::quantile(game.table(), nb);
            let mut s = BucketedFlopCfr::new(&tree, game.table(), &bk);
            // FL_TERM=factored ⇒ the precomputed-equity (Pluribus-style) terminal
            // (O(M·K), flat in B); default = the production Design1Collapsed (B^K).
            // Design2Factored is np≥4 only, so it applies to live-4/5.
            let term = match std::env::var("FL_TERM").as_deref() {
                Ok("factored") if live >= 4 => TerminalDesign::Design2Factored,
                _ => TerminalDesign::Design1Collapsed,
            };
            s.set_terminal_design(term);

            #[cfg(feature = "metal")]
            let ms = {
                let mut n = BucketedNativeGpu::new(&ctx, &tree, game.table(), &bk, &s, ((32 / nb).max(1)) as u32)
                    .expect("native gpu");
                n.run(1); // warm
                let t0 = Instant::now();
                n.run(iters);
                t0.elapsed().as_secs_f64() / iters as f64 * 1e3
            };
            #[cfg(not(feature = "metal"))]
            let ms = {
                s.run(&tree, &game, &bk, 1); // warm
                let t0 = Instant::now();
                s.run(&tree, &game, &bk, iters);
                t0.elapsed().as_secs_f64() / iters as f64 * 1e3
            };

            if bi == 0 {
                base_ms = ms;
            }
            eprintln!("{live:<8} {nb:>6} {nh:>6} {ms:>12.2} {:>11.2}x", ms / base_ms);
        }
        eprintln!();
    }
}
