//! CPU-vs-GPU PER-PATH BENCH (2026-06-17): live2_cut_probe found exact HU is
//! 0.40s on CPU vs 11.4s on the GPU MetalFlopStartSolver — a 28× GPU
//! PESSIMIZATION for the tiny 1×1 subgames. So re-measure ALL paths on CPU and
//! compare to the GPU times (joint_probe: L3 0.15 / L4 1.46 / L5 4.70 / L2 11.4),
//! because if the GPU is slow for every small subgame, the whole joint solve
//! should be CPU and the 219h breakdown is badly wrong.

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::{extract_v_flop_root_from_cum, PreflopChanceTable};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

fn main() {
    let spec = production_game_v1();
    let iters: u32 = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(34);
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
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };

    // GPU times from joint_probe, for the comparison column.
    let gpu = [("live-2 exact", 11.4f64), ("live-3 B15", 0.15), ("live-4 B15", 1.46), ("live-5 B8", 4.70)];

    eprintln!("CPU-vs-GPU per-path bench ({iters} iters)\n");
    eprintln!("{:<14} {:>10} {:>10} {:>12}", "path", "CPU s", "GPU s", "CPU speedup");

    // live-2 exact (CPU FlopStartVectorCfr)
    {
        let tree = build_tree(&spec.flop_seam_config(2, 2, 12, bets.clone())).unwrap();
        let t = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
        let game = FlopStartGame::new(t);
        let mut s = FlopStartVectorCfr::new(&tree, game.table());
        let t0 = Instant::now();
        s.run(&tree, &game, iters);
        let _ = extract_v_flop_root_from_cum(&mut s, &tree, &game, 0);
        let cpu = t0.elapsed().as_secs_f64();
        eprintln!("{:<14} {:>10.3} {:>10.2} {:>11.1}x", gpu[0].0, cpu, gpu[0].1, gpu[0].1 / cpu);
    }

    // live-3/4/5 bucketed (CPU BucketedFlopCfr)
    for (i, (live, nb)) in [(3u8, 15usize), (4, 15), (5, 8)].into_iter().enumerate() {
        let tree = build_tree(&spec.flop_seam_config(live, 2, 12, bets.clone())).unwrap();
        let t = FlopChanceTable::build_full_nh_sampled(canonical, live, &turns, &river_decks);
        let bk = FlopBucketing::quantile(&t, nb);
        let game = FlopStartGame::new(t);
        let mut s = BucketedFlopCfr::new(&tree, game.table(), &bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        let t0 = Instant::now();
        s.run(&tree, &game, &bk, iters);
        let _ = s.root_cfv_from_avg(&tree, &game, &bk);
        let cpu = t0.elapsed().as_secs_f64();
        let (label, gs) = gpu[i + 1];
        eprintln!("{label:<14} {cpu:>10.3} {gs:>10.2} {:>11.1}x", gs / cpu);
    }
    eprintln!("\n(>1x ⇒ CPU faster than the GPU path the joint factory currently uses)");
}
