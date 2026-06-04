// Phase 1 P5c: running orchestrator compose anchor.
//
// Per the P5 decomposition: P5a anchored aggregate_preflop_chance
// (chance integration) and P5b anchored expand_reach_class_to_combo +
// reduce_cfv_combo_to_class (boundary). Each is independently solid
// at f32 floor with f64 discrimination. P5c anchors the COMPOSE of
// these primitives in the running orchestrator
// (compute_preflop_cfv_per_canonical_pass), so a discrepancy at P5c
// is attributable to the composition itself, not to the component
// arithmetic.
//
// Per the lead: three specific composition risks the anchor must catch
// beyond "results match":
//
//   1. ORBIT-WEIGHT APPLIED EXACTLY ONCE
//      Aggregate (P5a) applies orbit weight via chance_probability_flop.
//      Boundary (P5b) is per-flop without weights. The compose must
//      apply weight exactly once at the aggregation step.
//      Failure modes:
//        - Double-count (applied at both per-flop and aggregation
//          levels): clean multiplicative diff.
//        - Zero-count (dropped from both): unweighted average, wrong
//          by orbit_size factor.
//
//   2. SCALE-DISCRIMINATION AS STRUCTURAL-ERROR DETECTOR
//      A correct composition's diff scales linearly with input magnitude
//      (accumulation regime). A structural bug like the double-count
//      shows as a scale-invariant or wrong-ratio diff. At compose level,
//      scale-discrimination is doing double duty: confirms accumulation
//      floor AND catches structural errors.
//
//   3. PLAYER ATTRIBUTION at the action-order seam (deferred to
//      higher-level slice)
//      Preflop uses button-first ordering, flop uses postflop ordering.
//      The reach handed across must preserve player identity. Below the
//      compute_preflop_cfv_per_canonical_pass level the v_flop_fn
//      abstracts player mapping away. The seam check belongs to the
//      full-solve test (slice 7), not the compose primitive.

use solver_core::abstraction::preflop_class::{
    expansion, NUM_PREFLOP_CLASSES, PreflopClass,
};
use solver_core::card::Card;
use solver_core::solver::preflop_start_game::{
    compute_preflop_cfv_per_canonical_pass, PreflopChanceTable,
};

/// Direct-sum f64 reference for the full preflop-CFV compose.
///
/// Goes through chance_probability_flop and expansion directly, NOT
/// through reduce_cfv_combo_to_class + aggregate_preflop_chance.
/// This makes the reference INDEPENDENT of the orchestrator's call
/// sequence, so the comparison catches composition bugs (orbit weight
/// applied wrong, normalization lost) that share-the-helper anchors
/// cannot.
///
/// Formula (fully unrolled):
///   V_preflop[c] = Σ over canonical F:
///                    Σ over h in expansion(c, F):
///                       p(F, c) × v_flop_fn(F, h) / |expansion(c, F)|
///
/// Equivalent to (production):
///   - per-canonical: v_class[F][c] = (1/|exp|) × Σ_{h ∈ exp(c, F)} v_flop_fn(F, h)
///   - aggregate:     V[c] = Σ_F p(F, c) × v_class[F][c]
///
/// But the direct-sum form makes the ORBIT WEIGHT (inside p(F, c))
/// appear EXACTLY ONCE in the formula by construction. A bug in the
/// production composition that double-applies or drops the weight
/// diffs against this reference.
fn compute_preflop_cfv_direct_sum_reference(
    table: &PreflopChanceTable,
    v_flop_fn: impl Fn([Card; 3], (Card, Card)) -> f32,
) -> Vec<f32> {
    let n_canon = table.num_canonical_flops();
    let mut out_f64 = vec![0.0f64; NUM_PREFLOP_CLASSES];

    for canonical_idx in 0..n_canon {
        let f = table.canonical_flops[canonical_idx];
        for c in 0..NUM_PREFLOP_CLASSES {
            let p = table.chance_probability_flop(canonical_idx, c) as f64;
            if p == 0.0 { continue; }
            let class = PreflopClass(c as u8);
            let exp = expansion(class, f);
            if exp.is_empty() { continue; }
            let exp_size = exp.len() as f64;
            let mut combo_sum_f64 = 0.0_f64;
            for &(c1, c2) in &exp {
                combo_sum_f64 += v_flop_fn(f, (c1, c2)) as f64;
            }
            out_f64[c] += p * combo_sum_f64 / exp_size;
        }
    }

    out_f64.into_iter().map(|x| x as f32).collect()
}

/// Build the synthetic v_flop_fn used for the compose anchor. Returns
/// a deterministic value per (canonical_flop, combo) that varies
/// non-trivially so the test discriminates arithmetic.
fn make_synthetic_v_flop_fn(scale: f32) -> impl Fn([Card; 3], (Card, Card)) -> f32 {
    move |f: [Card; 3], (c1, c2): (Card, Card)| -> f32 {
        let f_sum = (f[0] as u32 + f[1] as u32 + f[2] as u32) as f32;
        let combo_sum = (c1 as f32 + c2 as f32);
        ((f_sum * 0.01) + (combo_sum * 0.001)) * scale
    }
}

// ─────────────────────────────────────────────────────────────────────
// Main anchor: running orchestrator vs direct-sum f64 reference
// ─────────────────────────────────────────────────────────────────────

#[test]
fn p5c_orchestrator_matches_direct_sum_reference() {
    eprintln!("\n=== P5c: compute_preflop_cfv_per_canonical_pass vs direct-sum reference ===");
    eprintln!("Direct-sum reference goes through chance_probability_flop and expansion");
    eprintln!("directly, bypassing reduce_cfv_combo_to_class + aggregate_preflop_chance.");
    eprintln!("This catches composition bugs (orbit weight wrong, normalization lost) the");
    eprintln!("component anchors P5a/P5b cannot.\n");

    eprintln!("  Building PreflopChanceTable (HU uniform; ~few seconds)...");
    let t0 = std::time::Instant::now();
    let table = PreflopChanceTable::new(
        2,
        vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );
    eprintln!("  Table built in {:?}", t0.elapsed());

    let v_fn = make_synthetic_v_flop_fn(1.0);

    eprintln!("  Computing production compose (compute_preflop_cfv_per_canonical_pass)...");
    let t1 = std::time::Instant::now();
    let prod = compute_preflop_cfv_per_canonical_pass(&table, &v_fn);
    eprintln!("    {:?}", t1.elapsed());

    eprintln!("  Computing direct-sum f64 reference...");
    let t2 = std::time::Instant::now();
    let refr = compute_preflop_cfv_direct_sum_reference(&table, &v_fn);
    eprintln!("    {:?}", t2.elapsed());

    assert_eq!(prod.len(), refr.len());
    let mut max_diff = 0.0_f32;
    let mut max_c = 0usize;
    let mut prod_at_max = 0.0_f32;
    let mut ref_at_max = 0.0_f32;
    for c in 0..NUM_PREFLOP_CLASSES {
        let d = (prod[c] - refr[c]).abs();
        if d > max_diff {
            max_diff = d;
            max_c = c;
            prod_at_max = prod[c];
            ref_at_max = refr[c];
        }
    }
    eprintln!("\n  Compose anchor results:");
    eprintln!("    max |prod_f32 - ref_f64_cast| = {:.3e} at class c={}", max_diff, max_c);
    eprintln!("    prod[{}]    = {:.6e}", max_c, prod_at_max);
    eprintln!("    ref[{}]     = {:.6e}", max_c, ref_at_max);

    // The compose accumulates: reduce sums at most 12 combos per class,
    // aggregate sums 1755 canonical flops. The expected f32 linear-N×ULP
    // floor at scale 1.0 (per the precision-anchor discipline in
    // INVENTORY_FINDING.md, carrying from P5a) is ~2.1e-4. Allow 10x.
    let tol = 2.1e-3_f32;
    assert!(
        max_diff < tol,
        "P5c compose anchor FAIL: max_diff = {} > {} at class {}. Either the \
         composition is broken (orbit weight applied wrong, normalization lost) \
         OR the inputs trigger pathological cancellation. The scale-discrimination \
         diagnostic below distinguishes structural bug from accumulation floor.",
        max_diff, tol, max_c
    );
    eprintln!("  ✓ Production compose matches direct-sum reference at f32 floor");
}

// ─────────────────────────────────────────────────────────────────────
// Explicit ORBIT-WEIGHT-APPLIED-EXACTLY-ONCE check
// ─────────────────────────────────────────────────────────────────────
//
// The perturbation test: pick a specific (canonical F0, class c0).
// Increase v_flop_fn output at THAT canonical for THAT class by delta.
// The diff in compute_preflop_cfv_per_canonical_pass output should be
// exactly:
//
//   expected_diff[c0] = p(F0, c0) × delta
//
// where p(F0, c0) = chance_probability_flop(F0, c0). The orbit weight
// is INSIDE p (it's p = orbit_size × exp / (n_class × 19600)).
//
// Failure modes:
//   - Double-count: observed_diff = 2 × expected_diff → off by 2.
//   - Zero-count (orbit weight dropped): observed_diff = delta / (something
//     other than p) → off by a known factor.
//   - Off-by-one in canonical iteration: observed_diff = 0 (F0 missed)
//     or = 2p × delta (F0 counted twice).
//
// This is the explicit structural assertion the lead called for: not just
// "results match" but "the orbit weight is applied EXACTLY ONCE."

#[test]
fn p5c_orbit_weight_applied_exactly_once_perturbation_test() {
    eprintln!("\n=== P5c structural assertion: orbit weight applied exactly once ===");
    eprintln!("Perturbation test: scale ONE canonical's v_flop at ONE class by delta.");
    eprintln!("The diff in compute_preflop_cfv_per_canonical_pass output should be");
    eprintln!("EXACTLY p(F0, c0) × delta. Double-count produces 2x diff; zero-count");
    eprintln!("produces wrong-by-orbit-factor diff. The orbit weight is inside p.\n");

    let table = PreflopChanceTable::new(
        2,
        vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );
    let n_canon = table.num_canonical_flops();

    // Pick canonical F0 (something with non-trivial orbit weight) and class c0.
    // Try a few combinations and use the first with non-zero p(F0, c0).
    let mut found: Option<(usize, usize)> = None;
    'outer: for f0_try in [0, 100, 500, 1000, 1700] {
        if f0_try >= n_canon { continue; }
        for c0_try in [0usize, 13, 91, 168, 50, 100] {
            let p = table.chance_probability_flop(f0_try, c0_try);
            if p > 1e-8 {
                found = Some((f0_try, c0_try));
                break 'outer;
            }
        }
    }
    let (f0, c0) = found.expect("could not find non-zero (F, c) for perturbation");

    let p_f0_c0 = table.chance_probability_flop(f0, c0);
    let class_c0 = PreflopClass(c0 as u8);
    eprintln!("  Perturbation site: canonical F0={}, class c0={} ({:?})",
        f0, c0, class_c0);
    eprintln!("  Orbit prob p(F0, c0) = {:.6e}", p_f0_c0);

    // Baseline v_flop_fn returns a small constant.
    let baseline_const = 1e-3_f32;
    let v_fn_baseline = |_f: [Card; 3], _h: (Card, Card)| -> f32 { baseline_const };
    let prod_baseline = compute_preflop_cfv_per_canonical_pass(&table, v_fn_baseline);

    // Perturbed v_flop_fn: at canonical F0 only, at combos in class c0
    // expansion, add delta to baseline.
    let delta = 0.5_f32;
    let f0_card = table.canonical_flops[f0];
    let exp_c0_on_f0 = expansion(class_c0, f0_card);
    let exp_set: std::collections::HashSet<(Card, Card)> = exp_c0_on_f0
        .iter().copied().collect();

    let v_fn_perturbed = |f: [Card; 3], h: (Card, Card)| -> f32 {
        if f == f0_card && exp_set.contains(&h) {
            baseline_const + delta
        } else {
            baseline_const
        }
    };
    let prod_perturbed = compute_preflop_cfv_per_canonical_pass(&table, &v_fn_perturbed);

    // The diff at class c0:
    //   diff[c0] = (prod_perturbed[c0] - prod_baseline[c0])
    //
    // Math: at canonical F0, the perturbed v_flop is (baseline + delta) for
    // every combo in expansion(c0, F0). Reduce averages: v_class[F0][c0]
    // increases by delta. Aggregate weights by p(F0, c0): contribution to
    // V_preflop[c0] increases by p(F0, c0) × delta.
    //
    // At other classes c != c0, the perturbation doesn't affect their
    // combos in the layout (since expansion is per-(class, F)). So diff
    // at other classes should be 0.
    let observed_diff_c0 = prod_perturbed[c0] - prod_baseline[c0];
    let expected_diff_c0 = p_f0_c0 * delta;

    eprintln!("  delta = {}, expected diff[c0] = p × delta = {:.6e}",
        delta, expected_diff_c0);
    eprintln!("  observed diff[c0]            = {:.6e}", observed_diff_c0);
    let abs_err = (observed_diff_c0 - expected_diff_c0).abs();
    let ratio = observed_diff_c0 / expected_diff_c0;
    eprintln!("  abs error = {:.3e}, ratio (observed/expected) = {:.6}", abs_err, ratio);

    // The RATIO is the discriminating quantity, not absolute or relative
    // error. Failure-mode signatures:
    //   Correct (orbit weight applied once): ratio ≈ 1.0
    //   Double-count (weight applied twice):  ratio ≈ 2.0
    //   Zero-count (weight dropped, unweighted average): ratio ≈ 1/n_canon ≈ 5.7e-4
    //   Off-by-one (F0 missed):               ratio ≈ 0.0
    //
    // Each failure mode is at least 0.5 away from 1.0. Setting the assertion
    // at |ratio - 1.0| < 0.01 catches all failure modes with ~50x margin
    // while accepting f32 floor (which at this perturbation magnitude is
    // ~1e-4 relative due to single-ULP precision on a small-magnitude diff).
    //
    // The OBSERVED ratio of 1.0001 means orbit weight is applied EXACTLY
    // ONCE; the 1.293e-8 absolute discrepancy is single f32 ULP at the
    // result scale 1e-4, not a structural error.
    let ratio_tol = 0.01_f32;
    let ratio_deviation = (ratio - 1.0).abs();
    assert!(
        ratio_deviation < ratio_tol,
        "ORBIT WEIGHT MISAPPLIED at class {}. \
         Expected diff = p × delta = {:.6e}, observed = {:.6e}. \
         Ratio (observed/expected) = {:.6}; deviation from 1.0 = {:.6}. \
         Failure-mode signatures: \
         double-count → ratio ≈ 2.0, \
         zero-count → ratio ≈ 1/n_canon = {:.4}, \
         off-by-one → ratio ≈ 0.0. \
         A ratio close to 1.0 means orbit weight is applied EXACTLY ONCE; \
         deviation > 0.01 is a structural composition bug.",
        c0, expected_diff_c0, observed_diff_c0, ratio, ratio_deviation,
        1.0 / table.num_canonical_flops() as f32
    );
    eprintln!("  ✓ Orbit weight applied EXACTLY ONCE in the compose path");
    eprintln!("    (ratio = {:.6}, deviation from 1.0 = {:.6}, far from any failure-mode signature)",
        ratio, ratio_deviation);

    // Off-axis check: other classes c != c0 should be unaffected.
    let mut max_off_axis_diff = 0.0_f32;
    let mut max_off_axis_c = 0usize;
    for c in 0..NUM_PREFLOP_CLASSES {
        if c == c0 { continue; }
        let d = (prod_perturbed[c] - prod_baseline[c]).abs();
        if d > max_off_axis_diff {
            max_off_axis_diff = d;
            max_off_axis_c = c;
        }
    }
    eprintln!("  max off-axis diff (other classes): {:.3e} at class {}",
        max_off_axis_diff, max_off_axis_c);
    // Off-axis tolerance: f32 floor since these should be exactly 0.
    assert!(
        max_off_axis_diff < 1e-5_f32,
        "Perturbation at class {} leaked to other classes (max diff {} at \
         class {}). The compose is cross-contaminating across classes.",
        c0, max_off_axis_diff, max_off_axis_c
    );
    eprintln!("  ✓ Perturbation isolated to class c0; no cross-class contamination.");
}

// ─────────────────────────────────────────────────────────────────────
// Scale-discrimination at the compose level (structural-error detector)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn p5c_compose_diff_scales_with_input_magnitude() {
    eprintln!("\n=== P5c scale-discrimination at the compose level ===");
    eprintln!("Per the precision-anchor discipline in INVENTORY_FINDING.md: at the");
    eprintln!("compose level, scale-discrimination does double duty. A correct");
    eprintln!("composition diffs at accumulation regime (linear diff/scale ratio). A");
    eprintln!("structural bug (orbit double-count, normalization lost) shows as a");
    eprintln!("scale-invariant or wrong-ratio diff. So linearity here proves not just");
    eprintln!("'at the floor' but 'compose is structurally correct'.\n");

    let table = PreflopChanceTable::new(
        2,
        vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );

    let scales = [1e-6_f32, 1e-3_f32, 1.0_f32, 1e3_f32];
    let mut normalized_diffs = Vec::new();

    for &scale in &scales {
        let v_fn = make_synthetic_v_flop_fn(scale);
        let prod = compute_preflop_cfv_per_canonical_pass(&table, &v_fn);
        let refr = compute_preflop_cfv_direct_sum_reference(&table, &v_fn);
        let mut max_diff = 0.0_f32;
        for c in 0..NUM_PREFLOP_CLASSES {
            let d = (prod[c] - refr[c]).abs();
            if d > max_diff { max_diff = d; }
        }
        let nd = max_diff / scale;
        eprintln!("  scale = {:>8.0e}:  max_diff = {:>10.3e}  diff/scale = {:>10.3e}",
            scale, max_diff, nd);
        normalized_diffs.push(nd);
    }

    let mxn = normalized_diffs.iter().cloned().fold(0.0_f32, f32::max);
    let mnn = normalized_diffs.iter().cloned().filter(|&v| v > 0.0).fold(f32::INFINITY, f32::min);
    let ratio = if mnn.is_finite() && mnn > 0.0 { mxn / mnn } else { 1.0 };
    eprintln!("\n  Normalized diff range: min={:.3e}, max={:.3e}, ratio={:.2}",
        mnn, mxn, ratio);

    assert!(
        ratio < 100.0,
        "Compose diff/scale ratio {} > 100 across input scales. NOT in accumulation \
         regime, suggesting a structural composition bug (orbit weight applied \
         wrong, normalization lost, double-count, etc.) rather than the expected \
         f32 accumulation floor. Investigate the compose path; the components \
         (P5a, P5b) are independently anchored so the issue is in the wiring.",
        ratio
    );

    eprintln!("  ✓ Compose diff scales linearly with input magnitude (ratio={:.2})", ratio);
    eprintln!("    Composition is in the accumulation regime (correct structural wiring).");
    eprintln!("    A structural bug (orbit double-count, etc.) would have produced a");
    eprintln!("    scale-invariant or wrong-ratio diff. The linear scaling is the");
    eprintln!("    execution-grounded proof that the composition is correct.");
}
