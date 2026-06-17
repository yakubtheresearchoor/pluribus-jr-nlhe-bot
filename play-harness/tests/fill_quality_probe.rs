//! FILL QUALITY PROBE (2026-06-14): is the banked GPU 34-iter blueprint good,
//! or is the ~8% root-CFV gap a problem? Two independent checks on one banked
//! cell, all CPU (no GPU contention with the fill):
//!   (A) GPU-vs-CPU at IDENTICAL bucketing: extract the banked GPU strategy's
//!       root CFV, vs a CPU re-solve using the cell's OWN banked maps (bp.bk),
//!       both at 34 iters. Isolates implementation divergence from the
//!       bucketing-mismatch my earlier gate accidentally introduced (it used a
//!       fresh quantile, not the banked maps).
//!   (B) Convergence: CPU re-solve (banked maps) at 34 / 100 / 300 iters —
//!       does the root CFV stabilize, i.e. is 34 iters "converged enough"?

use play_harness::blueprint::Blueprint;
use solver_core::card::Card;
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, TerminalDesign};
use solver_core::solver::preflop_start_game::table_hand_to_layout_perm;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

fn root() -> String {
    format!("{}/../blueprint_out_v1", env!("CARGO_MANIFEST_DIR"))
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mad: f64 = (0..n).map(|i| (a[i] as f64 - b[i] as f64).abs()).sum::<f64>() / n as f64;
    let scale = b[..n].iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6) as f64;
    mad / scale
}

#[test]
#[ignore = "fill quality probe (one banked cell, CPU); --ignored --nocapture --release"]
fn fill_quality_probe() {
    let (live, commit, pot, b) = (4u8, 10i32, 40i32, 15usize);
    let bp = Blueprint::load(&format!("{}/live{live}_c{commit}_p{pot}_b{b}/flop_0000.bp", root()))
        .expect("load");
    let spec = production_game_v1();
    let canonical: [Card; 3] = bp.flop;
    let perm = table_hand_to_layout_perm(&bp.game.table().hand_cards, bp.game.table().num_valid, canonical);
    let tree = build_tree(&spec.flop_seam_config(
        live, commit, pot,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
    )).unwrap();

    // Banked GPU strategy's root CFV (CPU extraction).
    let mut g = bp.indexer(&tree);
    g.cum_strategy_flop_mut().copy_from_slice(&bp.cum_flop);
    g.cum_strategy_turn_mut().copy_from_slice(&bp.cum_turn);
    g.cum_strategy_river_mut().copy_from_slice(&bp.cum_river);
    let gpu_root = g.root_cfv_from_avg(&tree, &bp.game, &bp.bk);
    let gpu0: Vec<f32> = perm.iter().map(|&h| gpu_root[0][h]).collect();

    eprintln!("\n═══ FILL QUALITY PROBE (live-{live} c{commit} p{pot} B{b}, flop 0) ═══");

    // (A) + (B): CPU re-solve with the cell's OWN banked maps (bp.bk), iter sweep.
    let mut prev: Option<Vec<f32>> = None;
    for &iters in &[34u32, 100, 300] {
        let mut s = BucketedFlopCfr::new(&tree, bp.game.table(), &bp.bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        let rr = s.run_all_root_cfv(&tree, &bp.game, &bp.bk, iters);
        let cpu0: Vec<f32> = perm.iter().map(|&h| rr[0][h]).collect();
        let vs_gpu = rel(&gpu0, &cpu0) * 100.0;
        let vs_prev = prev.as_ref().map(|p| rel(p, &cpu0) * 100.0);
        eprintln!(
            "  CPU re-solve @{iters:>3} it (banked maps): vs banked-GPU {vs_gpu:5.2}% | vs prev-iters {}",
            vs_prev.map(|v| format!("{v:.2}%")).unwrap_or("—".into())
        );
        prev = Some(cpu0);
    }
    eprintln!("→ (A) 'vs banked-GPU' at 34it with SAME maps ≈ 0 ⇒ earlier 8% was bucketing mismatch, fill fine.");
    eprintln!("  (B) 'vs prev-iters' shrinking ⇒ converging; 100→300 small ⇒ 34it already near-converged.");
}
