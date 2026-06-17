//! GPU live-2 fidelity gate (2026-06-16): the joint factory routes live-2
//! (exact HU) through `MetalFlopStartSolver` (GPU run) + the shared gated
//! extraction (`extract_v_flop_root_from_cum`). This gate confirms the GPU path
//! reproduces the deployed CPU path (`compute_v_flop_at_root_converged_with_table`)
//! per-hand flop-root CFV — same flop, same seeded 1×1 runout, same iters.
//!
//! Run: cargo test -p solver-core --features metal --test live2_gpu_gate -- --ignored --nocapture
//! (OUT_DIR must point at the dir holding solver.metallib.)
#![cfg(feature = "metal")]

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::flop_start_game::FlopChanceTable;
use solver_core::solver::preflop_start_game::{
    compute_v_flop_at_root_converged_with_table, compute_v_flop_root_gpu_per_traverser,
    PreflopChanceTable,
};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

#[test]
#[ignore = "gpu gate; --ignored --nocapture --features metal (needs OUT_DIR=metallib dir)"]
fn live2_gpu_matches_cpu() {
    let spec = production_game_v1();
    let ptable = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let canonical = ptable.canonical_flops[0];

    // seeded 1×1 runout (deterministic: first two non-board deck cards).
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let turns = vec![deck[0]];
    let mut river_decks = vec![vec![]; 52];
    river_decks[deck[0] as usize] = vec![deck[1]];

    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let tree = build_tree(&spec.flop_seam_config(2, 2, 12, bets)).unwrap();
    let iters = 34;

    // CPU reference (the deployed path), per traverser.
    let cpu: Vec<Vec<f32>> = (0..2u8)
        .map(|t| {
            let tbl = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
            compute_v_flop_at_root_converged_with_table(tbl, &tree, t, iters).0
        })
        .collect();

    // GPU: one solve, both traversers.
    let ctx = MetalContext::new().expect("Metal context (set OUT_DIR to metallib dir)");
    let tbl = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
    let (gpu, _lay) = compute_v_flop_root_gpu_per_traverser(&ctx, tbl, &tree, 2, iters);

    let mut worst_mad = 0.0f64;
    for t in 0..2usize {
        let c = &cpu[t];
        let g = &gpu[t];
        assert_eq!(c.len(), g.len(), "len mismatch trav {t}");
        let n = c.len();
        let mad = c.iter().zip(g).map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>() / n as f64;
        let maxd = c.iter().zip(g).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
        let cmin = c.iter().cloned().fold(f32::INFINITY, f32::min);
        let cmax = c.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        worst_mad = worst_mad.max(mad);
        eprintln!(
            "trav {t}: nh={n} | CPU range [{cmin:.4}, {cmax:.4}] | mean|Δ| {mad:.6}  max|Δ| {maxd:.6}"
        );
        assert!(g.iter().all(|x| x.is_finite()), "GPU CFV must be finite (trav {t})");
        assert!(cmax - cmin > 1e-4, "CFV must vary across hands, not be constant (trav {t})");
    }
    eprintln!("GPU live-2 gate: finite + non-constant OK; worst mean|Δ| vs CPU = {worst_mad:.6}");
    // Same equilibrium → CFVs should agree closely. Loose bar (CPU flop-start
    // DCFR has known ordering drift, GPU is the more-correct discount); tighten
    // if the printed mean|Δ| is tiny.
    assert!(worst_mad < 0.05, "GPU vs CPU live-2 CFV diverged: mean|Δ| {worst_mad:.6} ≥ 0.05");
}
