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

use play_harness::v1_seam::{SeamBlueprint, SeamGame};

fn flop() -> [u8; 3] { [47, 21, 0] }

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
