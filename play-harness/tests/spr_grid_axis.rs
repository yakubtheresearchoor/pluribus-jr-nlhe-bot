//! SPR / CELL-GRID AXIS (1×1 vs 2×2 runouts) as a SECOND priced dial in the
//! fidelity table, alongside bucket count. Measured cheaply on a common family
//! at research scale, BUCKET COUNT HELD FIXED (exact) so the runout-grid effect
//! is isolated (the de-confound discipline). Quality = mean |Δσ| of the FLOP
//! strategy vs a 4×4 reference (how far each grid is from the converged-runout
//! strategy — the "is 1×1 too crappy" question, empirically).
//!
//! Caveats carried: research-scale magnitudes transfer as a RELATIONSHIP not
//! absolute; SPR sensitivity may be DEPTH-dependent (deep stacks have wider SPR
//! range ⇒ 2×2 may matter more at 200bb than 30bb) — flagged for the matrix.
//!
//! ⛔ RETRACTED / DO NOT TRUST THESE FOR THE DECISION (2026-06-14): all three
//! tests below are STRUCTURALLY UNABLE to answer the production runout grid.
//! Production 1×1 = 1 seeded runout PER FLOP across 1,755 flops (an ENSEMBLE;
//! runout luck averages across flops, never clairvoyant). The single-flop
//! research proxy here can only build the DEGENERATE 1×1 (1 turn + 1 river ⇒
//! the flop strategy KNOWS the runout ⇒ clairvoyant + catastrophic), which
//! production 1×1 does NOT inherit. So: `spr_grid_sensitivity_and_pricing`
//! was sample-choice CONFOUNDED; `spr_grid_nested_sensitivity`'s clean "1×1 2×
//! worse" Δσ is RETRACTED (measured clairvoyance, not production);
//! `spr_grid_nested_ev`'s 188–368% Δ-EV is a seam-mismatch ARTIFACT. The
//! runout grid is decidable ONLY by the production head-to-head / field-model
//! (the real per-flop ensemble). Do not re-run a proxy to settle it — proven
//! three ways it can't. See memory project_runout_grid.md.

use play_harness::v1_seam::{SeamBlueprint, SeamGame};

fn flop() -> [u8; 3] { [47, 21, 0] }

/// NESTED re-measurement (2026-06-14): the confounded `spr_grid_sensitivity`
/// sign-flipped (live-4 1×1 came out CLOSER to 4×4 than 2×2 — impossible if
/// the metric measured runout resolution) because 1×1/2×2/4×4 drew disjoint
/// river cards. Here the grids are NESTED (4×4 ⊇ 2×2 ⊇ 1×1), so Δσ vs 4×4 is
/// controlled and MUST be monotone (finer ⇒ closer to the converged-runout
/// reference). Monotone 1×1 ≥ 2×2 is the sanity the confounded version failed;
/// the MAGNITUDE of 1×1's gap from 4×4 is the real "is 1×1 too crappy" answer.
/// Δ-EV re-score (2026-06-14): the nested Δσ measurement showed 1×1's flop
/// strategy is ~2× farther from converged-runout than 2×2's — but Δσ is
/// strategy DISTANCE, not EV. A strategy can move a lot between grids while
/// costing ~0 EV if the moved spots are near-indifferent (the
/// exploitability-vs-play lesson, restated in σ-vs-EV terms). The 4× matrix
/// cost (≈26 h → ≈100 h) is only justified if 1×1 costs real EV. So score the
/// SAME nested solves by EXPLOITABILITY in the 4×4 "real" game with a
/// converged continuation — the decision measure.
#[test]
#[ignore = "SPR-grid Δ-EV (exploitability, the DECISION measure); --ignored --nocapture --release"]
fn spr_grid_nested_ev() {
    let nh = 16;
    const ITERS: u32 = 600;
    let pot = 12u32;
    eprintln!("\n═══ SPR/CELL-GRID Δ-EV (exploitability in 4×4 'real' game, converged continuation) ═══");
    eprintln!("Δσ showed the strategy MOVES; this shows whether the movement COSTS EV.");
    eprintln!("family | expl(1×1) | expl(2×2) | expl(ref) | ΔEV(1×1) | ΔEV(2×2)  [pt of pot]");
    for live in 3u8..=4 {
        let g = SeamGame::new(live, 2, 12, flop());
        let b1 = SeamBlueprint::solve_research_grid_nested(&g, nh, 1, 1, ITERS);
        let b2 = SeamBlueprint::solve_research_grid_nested(&g, nh, 2, 2, ITERS);
        let bref = SeamBlueprint::solve_research_grid_nested(&g, nh, 4, 4, ITERS);
        let e1 = b1.deploy_exploitability(&bref, &g, nh, pot);
        let e2 = b2.deploy_exploitability(&bref, &g, nh, pot);
        let er = bref.deploy_exploitability(&bref, &g, nh, pot);
        eprintln!("  live-{live} | {e1:7.3}% | {e2:7.3}% | {er:7.3}% | {:+.3} | {:+.3}", e1 - er, e2 - er);
    }
    eprintln!("\n→ ΔEV = excess exploitability over the converged-continuation baseline. If ΔEV(1×1)≈ΔEV(2×2)");
    eprintln!("  (both small), 1×1 is fine on EV despite the 2× Δσ — launch 1×1. If ΔEV(1×1)≫ΔEV(2×2),");
    eprintln!("  the runout grid costs real EV and 2×2 earns its ~4× cost.");
}

#[test]
#[ignore = "SPR-grid NESTED sensitivity (research-scale, minutes); --ignored --nocapture --release"]
fn spr_grid_nested_sensitivity() {
    let nh = 16;
    const ITERS: u32 = 600;
    eprintln!("\n═══ SPR/CELL-GRID NESTED (1×1 vs 2×2 vs 4×4-ref, controlled samples) ═══");
    eprintln!("family | |Δσ flop| vs 4×4-ref: 1×1 | 2×2 | (coarser ⊂ finer; lower = closer)");
    for live in 3u8..=4 {
        let g = SeamGame::new(live, 2, 12, flop());
        let g1 = SeamBlueprint::solve_research_grid_nested(&g, nh, 1, 1, ITERS);
        let g2 = SeamBlueprint::solve_research_grid_nested(&g, nh, 2, 2, ITERS);
        let gref = SeamBlueprint::solve_research_grid_nested(&g, nh, 4, 4, ITERS);
        let d1 = g1.flop_sigma_delta(&gref, g.tree_ref());
        let d2 = g2.flop_sigma_delta(&gref, g.tree_ref());
        let ratio = d1 / d2.max(1e-9);
        let mono = if d1 + 1e-9 >= d2 { "MONOTONE ✓" } else { "STILL INVERTED ✗" };
        eprintln!("  live-{live} | {d1:.4} | {d2:.4} | 1×1 {ratio:.2}× of 2×2's gap | {mono}");
    }
    eprintln!("\n→ monotone 1×1 ≥ 2×2 = confound removed. Then read MAGNITUDE: 2×2 ≪ 1×1 ⇒ runout");
    eprintln!("  grid MATTERS (2×2 earns its ~3× cost); 2×2 ≈ 1×1 ⇒ 1×1 is fine for the matrix.");
}

#[test]
#[ignore = "SPR-grid sensitivity (research-scale, minutes); --ignored --nocapture --release"]
fn spr_grid_sensitivity_and_pricing() {
    // Common family, fixed nh + exact bucketing — only the runout grid varies.
    let nh = 16;
    const ITERS: u32 = 600;
    eprintln!("\n═══ SPR/CELL-GRID SENSITIVITY (1×1 vs 2×2 vs 4×4-ref), bucket count fixed ═══");
    eprintln!("family | |Δσ flop| vs 4×4-ref: 1×1 | 2×2 | (lower = closer to converged runout)");
    for live in 3u8..=4 {
        let g = SeamGame::new(live, 2, 12, flop());
        let g1 = SeamBlueprint::solve_research_grid(&g, nh, 1, 1, ITERS);
        let g2 = SeamBlueprint::solve_research_grid(&g, nh, 2, 2, ITERS);
        let gref = SeamBlueprint::solve_research_grid(&g, nh, 4, 4, ITERS);
        let d1 = g1.flop_sigma_delta(&gref, g.tree_ref());
        let d2 = g2.flop_sigma_delta(&gref, g.tree_ref());
        let ratio = d1 / d2.max(1e-9);
        eprintln!("  live-{live} | {d1:.4} | {d2:.4} | 1×1 is {ratio:.1}× farther from 4×4 than 2×2");
    }
    eprintln!("\n→ 2×2 ≪ 1×1 distance ⇒ runout grid MATTERS (2×2 earns cost); 2×2 ≈ 1×1 ⇒ 1×1 fine.");

    // COST of the 2×2 increment over 1×1 (banked: runout cost is ~linear in the
    // cell count; 2×2 = 4 cells vs 1, ~4× per-family solve — memory: B8 full-set
    // 1×1 10.5h / 2×2 42h ≈ 4×). Multiplied across the production matrix.
    eprintln!("\nCOST of 2×2 over 1×1 (GPU-hours), reach-weighted base (live-6 rollout):");
    let base_1x1 = 18.7f64; // the B8 reach-weighted floor (live-2/3/4@B8 + live-5@B8 + live-6 rollout)
    eprintln!("  per cell (1 stake×depth): 1×1 {base_1x1:.1}h → 2×2 ~{:.1}h (~4×)", base_1x1 * 4.0);
    for cells in [6usize, 10] {
        eprintln!("  ×{cells}-cell matrix (depths×stakes): 1×1 {:.0}h | 2×2 ~{:.0}h (+{:.0}h)",
            base_1x1 * cells as f64, base_1x1 * 4.0 * cells as f64, base_1x1 * 3.0 * cells as f64);
    }
    eprintln!("\n→ the 2×2 increment (+3× per cell) is paid ACROSS the whole matrix, so a small");
    eprintln!("  per-cell delta becomes large at 6–10 cells. DEPTH FLAG: if 2×2 helps, check whether");
    eprintln!("  it helps MORE at deep stacks (wide SPR range) — if so, cell-grid is depth-dependent");
    eprintln!("  (1×1 short, 2×2 deep) = a reach-weighting on the SECOND axis too.");
}
