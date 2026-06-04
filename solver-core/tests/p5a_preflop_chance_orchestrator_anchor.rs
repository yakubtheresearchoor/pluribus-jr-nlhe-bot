// Phase 1 P5a: preflop chance integration anchored at the running
// orchestrator with f64 discrimination.
//
// Per the preflop completion plan: "the preflop chance integration first
// (the three-card deal and orbit-weighted aggregation in the real
// orchestrator, anchored against an independent reference with f64
// discrimination proving the aggregation arithmetic is exact, the P2.5a
// discipline applied to the running orchestrator not the stub)."
//
// P2.5a (commit history: prior preflop work) used a `v_flop_stub`
// function to verify the aggregation pipeline produced the right answer
// for arbitrary leaf values. That validated the algorithmic structure.
// P5a applies the same f64 discipline directly to the PRODUCTION
// `aggregate_preflop_chance` function (the orchestrator's actual entry
// point), proving its f32 arithmetic introduces no error beyond f32
// representational precision.
//
// ## What this anchors
//
// `aggregate_preflop_chance(table, flop_cfvs)` computes
//   out[h] = Σ over canonical flops F of p(F, h) × flop_cfvs[F][h]
//
// where p(F, h) = chance_probability_flop(F, h) = orbit-weighted
// probability of class h reaching canonical flop F.
//
// Production sums in f32. The reference sums in f64 (both p and
// flop_cfvs cast to f64 before the multiply-accumulate), then casts
// the final result to f32. If the production f32 arithmetic introduces
// no algorithmic error (only representational), the two results
// should agree at f32 floor (single-ULP precision at scale ~1.0,
// somewhat larger as accumulation across 1755 terms can chip away
// at f32 precision).
//
// ## Why this matters for P5
//
// P5 integrates preflop into the real blueprint loop. The aggregation
// step is where 1755 per-flop CFV vectors collapse into one preflop
// CFV vector. If the f32 accumulation drifts beyond f32 floor (e.g.,
// catastrophic cancellation or systematic bias from the order of
// addition), the preflop CFVs at the running loop are wrong by more
// than precision allows, and downstream regret updates compound the
// error. P5a verifies this CAN'T happen at the aggregation step.
//
// A diff above the f32-precision tolerance is unambiguously a
// production-code arithmetic issue (the reference is f64-exact for
// the aggregation step given f32 inputs).
//
// ## On the f64-discrimination limit at this stage
//
// The per-flop leaf values themselves (the inputs flop_cfvs) come
// from f32 per-flop solves in production. The f64 reference takes
// these f32 inputs and uses f64 accumulators. So the reference can
// only validate the AGGREGATION step is exact, not that the inputs
// themselves are f64-exact (they're not; they're f32). This is the
// same limitation noted in P2.5b ("the f64 mirror was partially
// neutered because both sides shared f32 leaf values").
//
// P5a's specific claim: GIVEN f32 leaf values from per-flop solves,
// the orchestrator's aggregation step preserves f32 precision exactly.
// The per-flop solver's f32 floor is a separate property already
// validated by the postflop work.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::preflop_start_game::{
    aggregate_preflop_chance, PreflopChanceTable,
};

/// f64-precision reference for aggregate_preflop_chance.
///
/// Implements the same formula as production but with f64 accumulators
/// and f64-precision multiplication. The final cast to f32 happens
/// only at the very end (per-class), so the only f32 rounding is at
/// the final assignment. Production rounds at every accumulation step.
///
/// If production has no algorithmic error (only representational), the
/// two should agree to at most a few ULPs per output element after
/// the production's many f32 roundings vs the reference's single f32
/// rounding.
fn aggregate_preflop_chance_f64_reference(
    table: &PreflopChanceTable,
    flop_cfvs: &[Vec<f32>],
) -> Vec<f32> {
    let n_canon = table.num_canonical_flops();
    let nh = NUM_PREFLOP_CLASSES;
    assert_eq!(flop_cfvs.len(), n_canon, "flop_cfvs length mismatch");

    let mut out_f64 = vec![0.0f64; nh];
    for (canonical_idx, cfvs_for_flop) in flop_cfvs.iter().enumerate() {
        assert_eq!(cfvs_for_flop.len(), nh);
        for class_idx in 0..nh {
            let p = table.chance_probability_flop(canonical_idx, class_idx) as f64;
            let v = cfvs_for_flop[class_idx] as f64;
            out_f64[class_idx] += p * v;
        }
    }
    out_f64.into_iter().map(|x| x as f32).collect()
}

/// Build a deterministic synthetic flop_cfvs input that exercises all
/// (canonical_flop, class) cells with non-degenerate values.
///
/// Values are bounded small (avoid f32 overflow) and vary across both
/// axes so the test discriminates the aggregation arithmetic (vs
/// trivial values like all-1.0 where any aggregation would give the
/// same answer regardless of arithmetic precision).
///
/// Pattern: v(F, h) = small_scale × ((F as f32) + h as f32 × 0.1)
/// adjusted to a mix of positive and negative values so cancellation
/// effects are exercised.
fn build_synthetic_flop_cfvs(n_canon: usize, nh: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(n_canon);
    let scale = 1e-3_f32;
    for f in 0..n_canon {
        let mut row = Vec::with_capacity(nh);
        for h in 0..nh {
            // Mixed-sign values centered around zero so the aggregation
            // exercises both positive and negative accumulation
            // (catches one-sided rounding bias if any).
            let raw = ((f as f32) - (n_canon as f32 / 2.0))
                + ((h as f32) - (nh as f32 / 2.0)) * 0.1;
            row.push(raw * scale);
        }
        out.push(row);
    }
    out
}

#[test]
fn p5a_aggregate_preflop_chance_f32_matches_f64_reference() {
    eprintln!("\n=== P5a: aggregate_preflop_chance f64-discrimination anchor ===");
    eprintln!("Proves: production f32 aggregate_preflop_chance matches the f64");
    eprintln!("        reference at f32 floor across all 1755 canonical flops × 169 classes.");
    eprintln!("        Confirms the aggregation arithmetic introduces no error beyond");
    eprintln!("        f32 representational precision (P2.5a discipline applied to the");
    eprintln!("        running orchestrator, not the stub).\n");

    // Build a real PreflopChanceTable with uniform HU class weights.
    // This exercises the FULL 1755-canonical-flop space (no subsetting),
    // per the lead's "at the running orchestrator" requirement.
    eprintln!("  Building PreflopChanceTable (HU uniform; ~few seconds for 22,100-orbit enumeration)...");
    let t0 = std::time::Instant::now();
    let table = PreflopChanceTable::new(
        2,
        vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );
    let n_canon = table.num_canonical_flops();
    eprintln!("  Table built in {:?}: {} canonical flops × {} classes",
        t0.elapsed(), n_canon, NUM_PREFLOP_CLASSES);

    // Construct synthetic deterministic input that exercises every
    // (canonical_flop, class) cell with a non-trivial value.
    let flop_cfvs = build_synthetic_flop_cfvs(n_canon, NUM_PREFLOP_CLASSES);

    // Production f32 aggregation.
    let prod_f32 = aggregate_preflop_chance(&table, &flop_cfvs);

    // Reference: same formula with f64 accumulators, final cast to f32.
    let ref_f64_cast = aggregate_preflop_chance_f64_reference(&table, &flop_cfvs);

    assert_eq!(prod_f32.len(), NUM_PREFLOP_CLASSES);
    assert_eq!(ref_f64_cast.len(), NUM_PREFLOP_CLASSES);

    // Discrimination check: at every class, |production f32 - reference f64-cast|
    // must be at f32 floor (≤ several ULPs at the result scale).
    //
    // The result scale: synthetic input values are O(1e-3 × n_canon) ≈ 2 per
    // entry, p is O(1/n_canon) ≈ 6e-4, so each per-flop contribution is
    // O(1e-3). Summed across 1755 terms (alternating signs), result is
    // O(1e-1 to 1). A single f32 ULP at scale 1.0 is ~1.2e-7. Across 1755
    // accumulations, accumulated f32 ULPs can be ~2e-4 in the worst case.
    let tol = 1e-4_f32;
    let mut max_abs_diff = 0.0_f32;
    let mut max_h = 0usize;
    let mut max_prod = 0.0_f32;
    let mut max_ref = 0.0_f32;
    for h in 0..NUM_PREFLOP_CLASSES {
        let d = (prod_f32[h] - ref_f64_cast[h]).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            max_h = h;
            max_prod = prod_f32[h];
            max_ref = ref_f64_cast[h];
        }
    }
    eprintln!("\n  Result diagnostics:");
    eprintln!("    max |prod_f32 - ref_f64_cast| = {:.3e} at class h={}", max_abs_diff, max_h);
    eprintln!("    prod_f32[{}]    = {:.8e}", max_h, max_prod);
    eprintln!("    ref_f64_cast[{}] = {:.8e}", max_h, max_ref);
    eprintln!("    tolerance       = {:.3e} (f32 floor across 1755-term accumulation)", tol);

    assert!(
        max_abs_diff < tol,
        "P5a FAIL: max diff between f32 production and f64 reference is {} > {} at class {}. \
         The aggregation arithmetic is NOT exact to f32 precision. \
         Either f32 production has algorithmic error, or the input scale is too large \
         and accumulation overflows f32 precision (in which case the tolerance needs \
         to be relaxed to match the actual scale, NOT the production code accepted as \
         correct without investigation).",
        max_abs_diff, tol, max_h
    );

    eprintln!("\n  ✓ P5a: aggregate_preflop_chance f32 production matches f64 reference");
    eprintln!("    at f32 floor across all 1755 canonical flops × 169 classes.");
    eprintln!("    Aggregation arithmetic exact to f32 representational precision.");
}

#[test]
fn p5a_aggregate_preflop_chance_with_zeros_is_exact() {
    // Edge case: when flop_cfvs are all zero, the aggregation must
    // produce exactly zero (regardless of probability distribution).
    // Tests there's no spurious nonzero from accumulating-then-rounding.
    eprintln!("\n=== P5a edge case: zero-input aggregation is exactly zero ===");
    let table = PreflopChanceTable::new(
        2,
        vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );
    let n_canon = table.num_canonical_flops();
    let zero_flop_cfvs: Vec<Vec<f32>> = (0..n_canon)
        .map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();

    let prod = aggregate_preflop_chance(&table, &zero_flop_cfvs);
    for (h, &v) in prod.iter().enumerate() {
        assert_eq!(v, 0.0_f32,
            "zero input produced nonzero output at class {}: {}", h, v);
    }
    eprintln!("  ✓ zero-input aggregation produces exact zero at all 169 classes");
}

#[test]
fn p5a_aggregate_preflop_chance_with_constant_input() {
    // Discriminating case: constant input v(F, h) = c for all (F, h).
    // Then out[h] = c × Σ_F p(F, h). The sum of probabilities over all
    // canonical flops for a fixed class h equals 1 by construction
    // (each class's reach is fully distributed across the canonical
    // flops). So out[h] should equal c for every class.
    //
    // This is a hand-computable invariant that catches:
    //   - Wrong orbit weighting (sum != 1 across F for fixed h)
    //   - Wrong probability computation
    //   - Off-by-one over the canonical flop enumeration
    //
    // The expected value is exactly c at every class; any drift indicates
    // a real bug, not just f32 floor noise.
    eprintln!("\n=== P5a invariant: Σ p(F, h) over all F equals 1 for every class h ===");
    let table = PreflopChanceTable::new(
        2,
        vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );
    let n_canon = table.num_canonical_flops();
    let c = 0.5_f32;
    let const_flop_cfvs: Vec<Vec<f32>> = (0..n_canon)
        .map(|_| vec![c; NUM_PREFLOP_CLASSES]).collect();

    let prod = aggregate_preflop_chance(&table, &const_flop_cfvs);

    // For each class h: out[h] should equal c (because Σ_F p(F, h) = 1).
    // f32 accumulation across 1755 terms each ~6e-4 has bounded error.
    // The orbit weighting + class expansion should sum to exactly 1
    // mathematically, with f32 error at the floor.
    let tol = 1e-4_f32;
    let mut max_abs_diff = 0.0_f32;
    let mut max_h = 0usize;
    for h in 0..NUM_PREFLOP_CLASSES {
        let d = (prod[h] - c).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            max_h = h;
        }
    }
    eprintln!("  max |prod[h] - {}| = {:.3e} at class h={}", c, max_abs_diff, max_h);
    eprintln!("  prod[{}] = {:.8}", max_h, prod[max_h]);
    assert!(
        max_abs_diff < tol,
        "P5a invariant FAIL: constant input c={} aggregated to {} at class h={} (diff {}). \
         Σ_F p(F, h) should equal 1 by construction; the deviation indicates the \
         orbit-weighting or chance-probability arithmetic has a real bug, not f32 noise.",
        c, prod[max_h], max_h, max_abs_diff
    );
    eprintln!("  ✓ Σ p(F, h) = 1 for every class h (orbit weighting correct, f32 floor preserved)");
}
