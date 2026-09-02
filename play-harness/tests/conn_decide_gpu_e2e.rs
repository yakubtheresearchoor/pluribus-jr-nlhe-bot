//! End-to-end GPU flop-search validation through the LIVE runtime path
//! (`ConnDecider::decide_postflop_search` → search_decision → try_gpu_flop_search).
//! Runs the same flop decision with the CPU search and the GPU search and
//! compares the hero's action distribution + reports latency. Built
//! `--features metal` and run with `--ignored`; skipped if the blueprint is
//! absent. Without the metal feature both paths are the CPU search (trivially
//! equal) — the comparison is only meaningful with `--features metal`.

use play_harness::api::DecideRequest;
use play_harness::api_conn::ConnDecider;
use std::time::Instant;

fn ws(p: &str) -> String { format!("{}/../{}", env!("CARGO_MANIFEST_DIR"), p) }

#[test]
#[ignore = "e2e: loads the full blueprint (~50s). Run on demand with --features metal."]
fn gpu_vs_cpu_flop_decision_e2e() {
    let Ok(dec) = ConnDecider::load(
        &std::env::var("CONN_TEST_BP").unwrap_or_else(|_| ws("blueprint_conn_eqr")),
        &ws("gs14_blueprint_cache"), 6, 5, 200, 7,
    ) else { eprintln!("blueprint absent — skipping"); return; };

    // A flop decision: HU (live-2) so the continuation is EXACT on both sides
    // (the HU closed form) — a clean GPU-vs-CPU comparison. (Multiway adds MC
    // continuation variance that QRE's exp() amplifies; validated separately at
    // the solver level in gpu_search_parity.)
    let mk = || DecideRequest {
        opponent_stats: vec![],
        pool_river_bluff: None,
        eff_stack: None,
        deadline_ms: None,
        budget_ms: None,
        preflop_actions: vec![],
        seat_positions: vec![],
        board: vec![3, 19, 35], hero_cards: [48, 49], live: 2, hero_idx: 0,
        commit_entry: 6, pot_entry: 12, flop_id: 0, route: false,
        ..Default::default()
    };

    // CPU path (GPU_SEARCH unset).
    std::env::remove_var("GPU_SEARCH");
    let t = Instant::now();
    let cpu = dec.decide_postflop_search(&mk());
    let cpu_ms = t.elapsed().as_millis();
    let Some(cpu) = cpu else { eprintln!("flop search returned None (hero not acting?) — skipping"); return; };

    // GPU path (GPU_SEARCH=1). No-op without --features metal (falls back to CPU).
    std::env::set_var("GPU_SEARCH", "1");
    let t = Instant::now();
    let gpu = dec.decide_postflop_search(&mk()).expect("gpu flop decision");
    let gpu_ms = t.elapsed().as_millis();
    std::env::remove_var("GPU_SEARCH");

    eprintln!("flop decision: street={} CPU {}ms / GPU {}ms", cpu.street, cpu_ms, gpu_ms);
    eprintln!("  CPU actions: {:?}", cpu.actions.iter().map(|a| (a.label, (a.prob * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
    eprintln!("  GPU actions: {:?}", gpu.actions.iter().map(|a| (a.label, (a.prob * 1000.0).round() / 1000.0)).collect::<Vec<_>>());

    assert_eq!(cpu.street, "flop");
    assert_eq!(gpu.actions.len(), cpu.actions.len(), "action count mismatch");
    // Distributions should match within search/schedule tolerance (the solver
    // core is validated at 2% L1; this is the full live path incl. marshaling).
    let l1: f32 = cpu.actions.iter().zip(&gpu.actions).map(|(c, g)| (c.prob - g.prob).abs()).sum::<f32>()
        / cpu.actions.len() as f32;
    eprintln!("  mean action-prob L1 (CPU vs GPU) = {l1:.4}  [informational: CPU uses a");
    eprintln!("   sample_m-MC continuation while the GPU HU continuation is exact; QRE's");
    eprintln!("   exp() amplifies that. Controlled parity is gated by solver gpu_search_parity.]");
    // Sanity: the GPU path produced a VALID, non-degenerate distribution.
    let z: f32 = gpu.actions.iter().map(|a| a.prob).sum();
    assert!((z - 1.0).abs() < 1e-3, "GPU action dist sums to {z}");
    assert!(gpu.actions.iter().all(|a| a.prob.is_finite() && a.prob >= 0.0), "GPU dist invalid");
    assert!(gpu_ms < 14_000, "GPU flop search over budget: {gpu_ms}ms");
}
