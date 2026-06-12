// P1.5.4 Slice B.3b: terminal CFV cost vs. opp reach sparsity.
//
// Per the lead (2026-06-04): "the 94s [observed in B.3's dense-reach
// 6-max smoke] is likely a synthetic-worst-case artifact, so measure
// the terminal CFV cost at realistic sparse fold-terminal reach...
// the real cost may be fine and the 94s may not occur in practice".
//
// Same measure-the-real-cost discipline that caught the slice 7a 7x
// and the 59-day waste. Before treating dense-reach cost as a
// constraint that needs optimization, confirm what the cost is at
// realistic sparsity.
//
// At a real preflop fold terminal at equilibrium, each opp's reach is
// concentrated on the classes that take the specific action sequence
// leading there. For folders: their fold-range classes (varies by
// position/situation; typically 30-100 classes out of 169). For non-
// folders (one player at fold terminals): their play-range classes
// for the action sequence.
//
// The production multiway terminal CFV already exploits sparsity at
// two levels:
//   - `if r == 0.0 { continue; }` per opp class (skip zero-reach branches)
//   - `if accumulated_reach == 0.0 { return; }` at recursion entry
//     (cumulative-product zero prune)
//
// So this test measures how cost actually scales with sparsity, to
// confirm that realistic fold-terminal sparsity gives reasonable
// per-terminal cost (or expose if it doesn't).

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::preflop_terminal::preflop_fold_terminal_cfv_multiway;

/// Build a synthetic opp reach with exactly `n_nonzero` non-zero
/// classes (positions chosen deterministically by stride). Each non-
/// zero class gets reach value 1.0 (uniform).
fn sparse_reach(n_nonzero: usize) -> Vec<f32> {
    let mut r = vec![0.0_f32; NUM_PREFLOP_CLASSES];
    if n_nonzero == 0 { return r; }
    let stride = NUM_PREFLOP_CLASSES / n_nonzero.max(1);
    for i in 0..n_nonzero {
        let idx = (i * stride).min(NUM_PREFLOP_CLASSES - 1);
        r[idx] = 1.0;
    }
    r
}

#[test]
#[ignore = "Slice B.3b: cost-vs-sparsity measurement, multiple minutes wall-clock at \
            higher densities. Run on demand: cargo test --release --test \
            p1_5_4_slice_b3b_terminal_cfv_sparsity_cost -- --ignored --nocapture"]
fn slice_b3b_terminal_cfv_cost_scales_with_opp_reach_sparsity() {
    eprintln!("\n═══ Slice B.3b: multiway terminal CFV cost vs. opp reach sparsity ═══");
    eprintln!("Measure: 6-max fold terminal (5 opps), varying non-zero classes per opp.");
    eprintln!("Production code exploits sparsity via per-class continue + cumulative-product");
    eprintln!("prune (preflop_terminal.rs:243,251). This test measures whether the");
    eprintln!("exploitation is sufficient at realistic fold-terminal sparsity.\n");

    let chip_delta = 1.5_f32;

    // Sparsity ladder. At a real preflop fold terminal, each opp's
    // reach is concentrated on classes that took the action sequence
    // leading here. For folders: their fold-range — say 30-100 classes.
    // For non-folders: their play-range — say 30-80 classes.
    let densities: Vec<usize> = vec![1, 2, 5, 10, 20, 50, 100, 169];

    eprintln!("{:<12} {:<12} {:<20} {:<12}",
        "n_nonzero", "n_opps", "wall_clock_secs", "max|v|");
    eprintln!("{}", "-".repeat(60));

    for &n_nonzero in &densities {
        let r = sparse_reach(n_nonzero);
        let opp_reaches: Vec<&[f32]> = (0..5).map(|_| r.as_slice()).collect();

        let t0 = std::time::Instant::now();
        let v = preflop_fold_terminal_cfv_multiway(&opp_reaches, chip_delta);
        let secs = t0.elapsed().as_secs_f64();
        let max_abs: f32 = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

        eprintln!("{:<12} {:<12} {:<20.3} {:<12.3e}", n_nonzero, 5, secs, max_abs);

        // Sanity: finite
        for (c, &val) in v.iter().enumerate() {
            assert!(val.is_finite(),
                "v[{}] not finite at n_nonzero={}", c, n_nonzero);
        }
    }

    eprintln!("\n══ Interpretation ══");
    eprintln!("If cost grows roughly as O(n_nonzero^5) (5 opps, naive enumeration), the");
    eprintln!("sparsity exploit isn't enough by itself for dense reaches. If cost grows");
    eprintln!("subexponentially (early-termination on conflicts cuts the tree), realistic");
    eprintln!("fold-terminal sparsity (30-100 classes/opp) may be fast enough.");
    eprintln!("");
    eprintln!("Reference: the lead's framing — measure the real cost before treating the");
    eprintln!("dense synthetic worst case as a constraint. If sparse-reach cost is");
    eprintln!("acceptable (e.g., sub-second per terminal at realistic sparsity), the");
    eprintln!("94s dense-reach number from B.3 is the synthetic-worst-case artifact,");
    eprintln!("not a real production constraint.");
}

/// Same measurement at HU (1 opp) — establishes the baseline cost
/// scaling and shows the multiway combinatorial blowup specifically.
#[test]
#[ignore = "Slice B.3b HU baseline. Run on demand: cargo test --release --test \
            p1_5_4_slice_b3b_terminal_cfv_sparsity_cost slice_b3b_hu -- --ignored --nocapture"]
fn slice_b3b_hu_baseline_cost_vs_sparsity() {
    eprintln!("\n═══ Slice B.3b HU baseline: cost vs. opp reach sparsity, 1 opp ═══");

    let chip_delta = 1.5_f32;
    let densities: Vec<usize> = vec![1, 10, 50, 169];

    eprintln!("{:<12} {:<12} {:<20}", "n_nonzero", "n_opps", "wall_clock_secs");
    eprintln!("{}", "-".repeat(48));

    for &n_nonzero in &densities {
        let r = sparse_reach(n_nonzero);
        let opp_reaches: Vec<&[f32]> = vec![r.as_slice()];
        let t0 = std::time::Instant::now();
        let _v = preflop_fold_terminal_cfv_multiway(&opp_reaches, chip_delta);
        let secs = t0.elapsed().as_secs_f64();
        eprintln!("{:<12} {:<12} {:<20.6}", n_nonzero, 1, secs);
    }
    eprintln!("\nHU is the floor: 1 opp = O(169 * joint enumeration per pair).");
    eprintln!("Multiway at 5 opps scales by the opp-tuple enumeration cost.");
}
