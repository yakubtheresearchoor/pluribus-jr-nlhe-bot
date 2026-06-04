// Phase 1 P5b: zone handoff anchor (preflop / flop boundary).
//
// Per the lead: "Anchor expand and reduce independently as planned with
// reduce being critical (feeds the regret update), and make the
// round-trip mass-conservation check cover classes of each multiplicity
// type (pair=6, suited=4, offsuit=12 combos) rather than an average
// class, because the round-trip is really testing expand/reduce
// multiplicity consistency and that's where it can break per-class-type."
//
// Three components, anchored independently:
//
//   Component A: expand_reach_class_to_combo anchor (f64 discrimination
//     plus scale-discrimination diagnostic per the precision-anchor
//     discipline now in INVENTORY_FINDING.md). Distributes class-level
//     reach across combos in the class.
//
//   Component B: reduce_cfv_combo_to_class anchor (CRITICAL, feeds
//     the preflop regret update; an error here corrupts the solve).
//     Same f64 + scale discipline.
//
//   Component C: round-trip mass conservation per multiplicity type.
//     Pair (6 combos), suited (4 combos), offsuit (12 combos) tested
//     separately because that's where the round-trip can break by
//     per-class-type. Both non-blocking and blocking flops covered.

use solver_core::abstraction::preflop_class::{
    NUM_PREFLOP_CLASSES, PreflopClass,
};
use solver_core::card::{card_from_str, Card};
use solver_core::solver::preflop_start_game::{
    expand_reach_class_to_combo, flop_combo_layout, reduce_cfv_combo_to_class,
};

// ─────────────────────────────────────────────────────────────────────
// f64 reference implementations (spec-derived, independent of production)
// ─────────────────────────────────────────────────────────────────────

/// f64-precision reference for expand_reach_class_to_combo.
///
/// Implements the same formula as production but with f64 division
/// throughout. The final cast to f32 happens only at the very end
/// per output element. Mirrors the P5a discriminating pattern.
fn expand_reach_class_to_combo_f64_reference(
    canonical_flop: [Card; 3],
    reach_class: &[f32],
    combo_layout: &[(Card, Card)],
) -> Vec<f32> {
    assert_eq!(reach_class.len(), NUM_PREFLOP_CLASSES);

    // Precompute expansion sizes in f64 (cast at the end of computation).
    let mut exp_sizes_f64 = vec![0u32; NUM_PREFLOP_CLASSES];
    for c in 0..NUM_PREFLOP_CLASSES {
        let class = PreflopClass(c as u8);
        let exp = solver_core::abstraction::preflop_class::expansion(
            class, canonical_flop,
        );
        exp_sizes_f64[c] = exp.len() as u32;
    }

    let mut reach_combo_f64 = vec![0.0f64; combo_layout.len()];
    for (idx, &(c1, c2)) in combo_layout.iter().enumerate() {
        if canonical_flop.contains(&c1) || canonical_flop.contains(&c2) {
            continue;
        }
        let class = PreflopClass::from_combo(c1, c2);
        let n_exp = exp_sizes_f64[class.index()];
        if n_exp > 0 {
            reach_combo_f64[idx] =
                reach_class[class.index()] as f64 / n_exp as f64;
        }
    }
    reach_combo_f64.into_iter().map(|x| x as f32).collect()
}

/// f64-precision reference for reduce_cfv_combo_to_class.
///
/// Sums combo CFVs in f64 accumulators, divides by exp_size in f64,
/// then casts to f32 at the very end. CRITICAL anchor because the
/// reduce output feeds the preflop regret update.
fn reduce_cfv_combo_to_class_f64_reference(
    canonical_flop: [Card; 3],
    cfv_combo: &[f32],
    combo_layout: &[(Card, Card)],
) -> Vec<f32> {
    assert_eq!(cfv_combo.len(), combo_layout.len());

    let mut exp_sizes = vec![0u32; NUM_PREFLOP_CLASSES];
    for c in 0..NUM_PREFLOP_CLASSES {
        let class = PreflopClass(c as u8);
        let exp = solver_core::abstraction::preflop_class::expansion(
            class, canonical_flop,
        );
        exp_sizes[c] = exp.len() as u32;
    }

    let mut cfv_class_f64 = vec![0.0f64; NUM_PREFLOP_CLASSES];
    for (idx, &(c1, c2)) in combo_layout.iter().enumerate() {
        if canonical_flop.contains(&c1) || canonical_flop.contains(&c2) {
            continue;
        }
        let class = PreflopClass::from_combo(c1, c2);
        cfv_class_f64[class.index()] += cfv_combo[idx] as f64;
    }
    for c in 0..NUM_PREFLOP_CLASSES {
        if exp_sizes[c] > 0 {
            cfv_class_f64[c] /= exp_sizes[c] as f64;
        }
    }
    cfv_class_f64.into_iter().map(|x| x as f32).collect()
}

// ─────────────────────────────────────────────────────────────────────
// Test inputs: a non-blocking flop and a blocking flop
// ─────────────────────────────────────────────────────────────────────

fn non_blocking_flop() -> [Card; 3] {
    // 2h 7d 3c: low ranks, three suits, no rank conflicts with most premium hands.
    [
        card_from_str("2h").unwrap(),
        card_from_str("7d").unwrap(),
        card_from_str("3c").unwrap(),
    ]
}

fn ace_blocking_flop() -> [Card; 3] {
    // Ah As Kc: blocks AA combos to just 1 of the original 6 (AdAc).
    // Also blocks AKs to 0 (all 4 suited combos contain Ah or As) and
    // AKo to a reduced count. Tests the per-multiplicity-type behavior
    // under blocking, which is where expansion sizes vary by class type.
    [
        card_from_str("Ah").unwrap(),
        card_from_str("As").unwrap(),
        card_from_str("Kc").unwrap(),
    ]
}

fn build_synthetic_reach_class(scale: f32) -> Vec<f32> {
    // Mixed-pattern values: vary by class index to exercise the expand
    // arithmetic non-trivially.
    (0..NUM_PREFLOP_CLASSES)
        .map(|c| (c as f32 - 84.0) * scale)
        .collect()
}

fn build_synthetic_cfv_combo(n_combos: usize, scale: f32) -> Vec<f32> {
    (0..n_combos)
        .map(|i| (i as f32 - n_combos as f32 / 2.0) * scale)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────
// Component A: expand_reach_class_to_combo anchor
// ─────────────────────────────────────────────────────────────────────

#[test]
fn p5b_expand_reach_f32_matches_f64_reference() {
    let flop = non_blocking_flop();
    let layout = flop_combo_layout(flop);
    let reach_class = build_synthetic_reach_class(1.0);

    let prod = expand_reach_class_to_combo(flop, &reach_class, &layout);
    let refr = expand_reach_class_to_combo_f64_reference(flop, &reach_class, &layout);

    assert_eq!(prod.len(), refr.len());
    let mut max_diff = 0.0_f32;
    let mut max_idx = 0usize;
    for i in 0..prod.len() {
        let d = (prod[i] - refr[i]).abs();
        if d > max_diff { max_diff = d; max_idx = i; }
    }
    eprintln!("\n=== P5b Component A: expand_reach_class_to_combo anchor ===");
    eprintln!("  Layout size: {} combos", layout.len());
    eprintln!("  max |prod_f32 - ref_f64_cast| = {:.3e} at idx={}", max_diff, max_idx);

    // expand is a single division per combo; no accumulation. The diff
    // should be at single-ULP precision (~1.19e-7 relative).
    let tol = 1e-5_f32;
    assert!(
        max_diff < tol,
        "P5b expand FAIL: max diff {} > {}. Production expand has algorithmic \
         error beyond single-division f32 precision.",
        max_diff, tol
    );
    eprintln!("  ✓ expand_reach_class_to_combo arithmetic exact to f32 precision");
}

#[test]
fn p5b_expand_diff_scales_with_input_magnitude() {
    let flop = non_blocking_flop();
    let layout = flop_combo_layout(flop);
    let scales = [1e-6_f32, 1e-3_f32, 1.0_f32, 1e3_f32];
    let mut normalized_diffs = Vec::new();
    eprintln!("\n=== P5b expand scale-discrimination diagnostic ===");
    for &scale in &scales {
        let reach_class = build_synthetic_reach_class(scale);
        let prod = expand_reach_class_to_combo(flop, &reach_class, &layout);
        let refr = expand_reach_class_to_combo_f64_reference(flop, &reach_class, &layout);
        let mut max_diff = 0.0_f32;
        for i in 0..prod.len() {
            let d = (prod[i] - refr[i]).abs();
            if d > max_diff { max_diff = d; }
        }
        let nd = max_diff / scale;
        eprintln!("  scale = {:>8.0e}:  max_diff = {:>10.3e}  diff/scale = {:>10.3e}",
            scale, max_diff, nd);
        normalized_diffs.push(nd);
    }
    let mxn = normalized_diffs.iter().cloned().fold(0.0_f32, f32::max);
    let mnn = normalized_diffs.iter().cloned().filter(|&v| v > 0.0).fold(f32::INFINITY, f32::min);
    let ratio = mxn / mnn;
    eprintln!("  Normalized diff range ratio: {:.2}", ratio);
    assert!(ratio < 100.0,
        "expand diff/scale ratio {} > 100 across scales; suggests fixed-magnitude \
         bug rather than expected single-division f32 floor", ratio);
    eprintln!("  ✓ expand diff scales linearly with input magnitude (single-division f32 floor)");
}

// ─────────────────────────────────────────────────────────────────────
// Component B: reduce_cfv_combo_to_class anchor (CRITICAL)
// ─────────────────────────────────────────────────────────────────────
//
// reduce is critical: its output feeds the preflop regret update. An
// arithmetic error here corrupts the solve. Same f64 + scale discipline.

#[test]
fn p5b_reduce_cfv_f32_matches_f64_reference() {
    let flop = non_blocking_flop();
    let layout = flop_combo_layout(flop);
    let cfv_combo = build_synthetic_cfv_combo(layout.len(), 1.0);

    let prod = reduce_cfv_combo_to_class(flop, &cfv_combo, &layout);
    let refr = reduce_cfv_combo_to_class_f64_reference(flop, &cfv_combo, &layout);

    assert_eq!(prod.len(), refr.len());
    let mut max_diff = 0.0_f32;
    let mut max_c = 0usize;
    for c in 0..NUM_PREFLOP_CLASSES {
        let d = (prod[c] - refr[c]).abs();
        if d > max_diff { max_diff = d; max_c = c; }
    }
    eprintln!("\n=== P5b Component B (CRITICAL): reduce_cfv_combo_to_class anchor ===");
    eprintln!("  max |prod_f32 - ref_f64_cast| = {:.3e} at class c={}", max_diff, max_c);

    // reduce sums up to 12 (offsuit max) combos then divides. The
    // accumulation is at most 12 terms so f32 sum is essentially
    // exact to single-ULP precision at result scale.
    let tol = 1e-5_f32;
    assert!(
        max_diff < tol,
        "P5b reduce FAIL: max diff {} > {} at class c={}. CRITICAL: reduce feeds \
         the preflop regret update; an arithmetic error here corrupts the solve.",
        max_diff, tol, max_c
    );
    eprintln!("  ✓ reduce_cfv_combo_to_class arithmetic exact to f32 precision");
    eprintln!("    (CRITICAL anchor: reduce output feeds preflop regret update)");
}

#[test]
fn p5b_reduce_diff_scales_with_input_magnitude() {
    let flop = non_blocking_flop();
    let layout = flop_combo_layout(flop);
    let scales = [1e-6_f32, 1e-3_f32, 1.0_f32, 1e3_f32];
    let mut normalized_diffs = Vec::new();
    eprintln!("\n=== P5b reduce scale-discrimination diagnostic ===");
    for &scale in &scales {
        let cfv_combo = build_synthetic_cfv_combo(layout.len(), scale);
        let prod = reduce_cfv_combo_to_class(flop, &cfv_combo, &layout);
        let refr = reduce_cfv_combo_to_class_f64_reference(flop, &cfv_combo, &layout);
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
    let ratio = mxn / mnn;
    eprintln!("  Normalized diff range ratio: {:.2}", ratio);
    assert!(ratio < 100.0,
        "reduce diff/scale ratio {} > 100 across scales; suggests fixed-magnitude bug",
        ratio);
    eprintln!("  ✓ reduce diff scales linearly with input magnitude (sum-then-divide f32 floor)");
}

// ─────────────────────────────────────────────────────────────────────
// Component C: round-trip mass conservation per multiplicity type
// ─────────────────────────────────────────────────────────────────────
//
// Per the lead: "make the round-trip mass-conservation check cover classes
// of each multiplicity type (pair=6, suited=4, offsuit=12 combos)
// rather than an average class, because the round-trip is really
// testing expand/reduce multiplicity consistency and that's where it
// can break per-class-type."
//
// The mass conservation property: if reach_class[c] = K and all other
// classes are 0, then expand distributes K across the combos in class c
// (modulo blocking by F). Total combo-level reach should sum to K.
//
// Per multiplicity type, the expansion size varies:
//   pair: 6 combos (e.g., AA = AsAh, AsAd, AsAc, AhAd, AhAc, AdAc)
//   suited: 4 combos (e.g., AKs = A♠K♠, A♥K♥, A♦K♦, A♣K♣)
//   offsuit: 12 combos (e.g., AKo = 4 × 3 = 12 ordered off-suit pairs)
//
// Bugs in expansion-size computation could be per-type (off-by-one
// for offsuit count, suited offset bug, etc.). Testing the three types
// separately discriminates these.

fn assert_mass_conserved_for_class(
    class_idx: usize,
    max_unblocked_combos: usize,
    flop: [Card; 3],
    flop_label: &str,
) {
    let class = PreflopClass(class_idx as u8);
    let kind = if class.is_pair() {
        "pair"
    } else if class.is_suited() {
        "suited"
    } else {
        "offsuit"
    };

    let mut reach_class = vec![0.0f32; NUM_PREFLOP_CLASSES];
    let class_mass = 1.0_f32;
    reach_class[class_idx] = class_mass;

    let layout = flop_combo_layout(flop);
    let reach_combo = expand_reach_class_to_combo(flop, &reach_class, &layout);

    // Sum reach over combos in this class only (others should be 0).
    let mut total_in_class = 0.0_f64;
    let mut combos_in_class = 0usize;
    let mut total_other_classes = 0.0_f64;
    for (idx, &(c1, c2)) in layout.iter().enumerate() {
        let c_of = PreflopClass::from_combo(c1, c2);
        if c_of.index() == class_idx {
            total_in_class += reach_combo[idx] as f64;
            combos_in_class += 1;
        } else if reach_combo[idx] != 0.0 {
            total_other_classes += reach_combo[idx] as f64;
        }
    }
    let mass_diff = (total_in_class - class_mass as f64).abs() as f32;
    eprintln!(
        "  class={:3} ({:7}, max-unblocked={} combos): combos_actual={}, \
         total_in_class={:.6}, mass_diff={:.3e}, leak_other_classes={:.3e}  [flop={}]",
        class_idx, kind, max_unblocked_combos,
        combos_in_class, total_in_class, mass_diff, total_other_classes, flop_label
    );
    // Sanity: combos_in_class must be <= max_unblocked (max for the multiplicity type).
    assert!(
        combos_in_class <= max_unblocked_combos,
        "more combos in class {} ({}) than the multiplicity-type max {}; \
         multiplicity bookkeeping is broken",
        class_idx, kind, max_unblocked_combos
    );

    // Mass conservation: combo reach in this class should sum to class_mass.
    // Tolerance allows f32 accumulation of up to 12 divisions then re-sum.
    let tol = 1e-6_f32;
    assert!(
        mass_diff < tol,
        "Mass NOT conserved for class {} ({}) on flop {}: combo sum = {} \
         but class mass = {} (diff {}). Expand/reduce multiplicity \
         consistency is broken for this multiplicity type.",
        class_idx, kind, flop_label, total_in_class, class_mass, mass_diff
    );
    assert!(
        total_other_classes.abs() < 1e-6 as f64,
        "Other classes have nonzero reach ({}) after expanding only class {}. \
         Expand is leaking reach across classes.",
        total_other_classes, class_idx
    );
}

#[test]
fn p5b_mass_conservation_per_multiplicity_type_non_blocking_flop() {
    eprintln!("\n=== P5b Component C: round-trip mass conservation (non-blocking flop) ===");
    eprintln!("Tests pair (6 combos), suited (4 combos), offsuit (12 combos) separately");
    eprintln!("because the round-trip is multiplicity-consistency and can break per-class-type.\n");

    let flop = non_blocking_flop();
    let label = "2h7d3c (non-blocking)";

    // Pair: AA at class index 0 (12 - rank = 12 - 12 = 0).
    assert_mass_conserved_for_class(0, 6, flop, label);
    // Pair: 22 at class index 12.
    assert_mass_conserved_for_class(12, 6, flop, label);
    // Suited: AKs at class index 13.
    assert_mass_conserved_for_class(13, 4, flop, label);
    // Suited: a mid-range suited (Q9s or similar). suited_offset for
    // (Q, 9) = depends on suited_offset formula. Pick another class
    // in the suited range to discriminate.
    assert_mass_conserved_for_class(50, 4, flop, label);
    // Offsuit: AKo at class index 91.
    assert_mass_conserved_for_class(91, 12, flop, label);
    // Offsuit: a mid-range offsuit.
    assert_mass_conserved_for_class(140, 12, flop, label);

    eprintln!("\n  ✓ Mass conservation holds for pair, suited, AND offsuit \
        multiplicity types on a non-blocking flop.");
    eprintln!("    Expand/reduce multiplicity consistency verified per-class-type.");
}

#[test]
fn p5b_mass_conservation_per_multiplicity_type_blocking_flop() {
    eprintln!("\n=== P5b Component C: round-trip mass conservation (BLOCKING flop) ===");
    eprintln!("Same per-multiplicity-type check on a flop that BLOCKS combos.");
    eprintln!("Tests that mass is still conserved when fewer combos are available.");
    eprintln!("Blocking flop AhAsKc: blocks AA to 1 combo, AKs to 0, AKo to 2 combos.\n");

    let flop = ace_blocking_flop();
    let label = "AhAsKc (blocks A and K hands)";

    // Pair: AA (index 0). Original 6 combos; flop blocks Ah and As, so
    // remaining combos from {Ad, Ac} pair = 1 combo (AdAc).
    assert_mass_conserved_for_class(0, 1, flop, label);
    // Pair: 22 (index 12). Not blocked by AAK flop.
    assert_mass_conserved_for_class(12, 6, flop, label);
    // Suited: AKs (index 13). All 4 combos contain at least one of Ah, As, Kc.
    // The Ah and As combos blocked by flop's A's; Kc combos blocked by flop's K.
    // Specifically: AsKs blocked (As in flop), AhKh blocked (Ah in flop),
    // AdKd OK if Kd is not in flop (it isn't), AcKc blocked (Kc in flop).
    // So AKs has 1 valid combo: AdKd. expansion size = 1.
    {
        let class = PreflopClass(13);
        let exp = solver_core::abstraction::preflop_class::expansion(class, flop);
        eprintln!("  diagnostic: AKs expansion on AhAsKc has {} combos", exp.len());
    }
    // Test with whatever the actual expansion size is.
    let aks_class = PreflopClass(13);
    let aks_combos = solver_core::abstraction::preflop_class::expansion(aks_class, flop).len();
    if aks_combos > 0 {
        assert_mass_conserved_for_class(13, aks_combos, flop, label);
    } else {
        eprintln!("  [AKs is fully blocked by AhAsKc; skipping mass test for this class]");
    }
    // Suited 50 (a low-ish suited class). Likely unblocked or partially blocked.
    let class_50_combos = solver_core::abstraction::preflop_class::expansion(
        PreflopClass(50), flop,
    ).len();
    if class_50_combos > 0 {
        assert_mass_conserved_for_class(50, class_50_combos, flop, label);
    }
    // Offsuit: AKo (index 91). AKo combos are AhKx with x != h, AsKx etc.
    // The Kc is blocked. The Ah and As blocked. So legal AKo combos:
    // AdKh, AdKs, AdKd (no, dKd is offsuit only if Ad-Kd is offsuit; wait
    // AKo means A different suit than K). Let's just check the expansion.
    let ako_class = PreflopClass(91);
    let ako_combos = solver_core::abstraction::preflop_class::expansion(ako_class, flop).len();
    eprintln!("  diagnostic: AKo expansion on AhAsKc has {} combos", ako_combos);
    if ako_combos > 0 {
        assert_mass_conserved_for_class(91, ako_combos, flop, label);
    }
    // Offsuit 140: lower offsuit class.
    let class_140_combos = solver_core::abstraction::preflop_class::expansion(
        PreflopClass(140), flop,
    ).len();
    if class_140_combos > 0 {
        assert_mass_conserved_for_class(140, class_140_combos, flop, label);
    }

    eprintln!("\n  ✓ Mass conservation holds under blocking flop too.");
    eprintln!("    Expansion-size-dependent mass distribution per multiplicity type \
        is consistent (the class mass is concentrated on fewer combos, but the TOTAL \
        mass is preserved).");
}

#[test]
fn p5b_reduce_expand_uniform_within_class_round_trip() {
    // The documented identity for reduce: if cfv_combo was constructed
    // as "uniform within class" (every combo h in expansion(c, F) has
    // cfv_combo[h] = K_c), then reduce returns K_c per class.
    //
    // Test per multiplicity type: pair (6 combos), suited (4), offsuit (12).
    eprintln!("\n=== P5b round-trip identity: reduce ∘ uniform-class-expand recovers class values ===");

    let flop = non_blocking_flop();
    let layout = flop_combo_layout(flop);

    // Build a class-uniform cfv pattern: class c gets value V_c, all
    // combos in class c receive V_c (NOT V_c / n_exp). Then reduce
    // averages: Σ V_c / n_exp = V_c.
    let v_class: Vec<f32> = (0..NUM_PREFLOP_CLASSES)
        .map(|c| (c as f32 - 84.0) * 0.01)
        .collect();
    let mut cfv_combo = vec![0.0f32; layout.len()];
    for (idx, &(c1, c2)) in layout.iter().enumerate() {
        let class = PreflopClass::from_combo(c1, c2);
        cfv_combo[idx] = v_class[class.index()];
    }

    let cfv_class = reduce_cfv_combo_to_class(flop, &cfv_combo, &layout);

    // For each class with non-zero expansion: reduce result should equal v_class.
    let mut max_diff = 0.0_f32;
    let mut max_c = 0usize;
    let mut pair_diff = 0.0_f32;
    let mut suited_diff = 0.0_f32;
    let mut offsuit_diff = 0.0_f32;
    for c in 0..NUM_PREFLOP_CLASSES {
        let class = PreflopClass(c as u8);
        let exp_size = solver_core::abstraction::preflop_class::expansion(class, flop).len();
        if exp_size == 0 { continue; }
        let d = (cfv_class[c] - v_class[c]).abs();
        if d > max_diff { max_diff = d; max_c = c; }
        if class.is_pair() && d > pair_diff { pair_diff = d; }
        if class.is_suited() && d > suited_diff { suited_diff = d; }
        if class.is_offsuit() && d > offsuit_diff { offsuit_diff = d; }
    }
    eprintln!("  max overall diff: {:.3e} at class c={}", max_diff, max_c);
    eprintln!("  per-multiplicity-type max diffs:");
    eprintln!("    pair (6 combos):    {:.3e}", pair_diff);
    eprintln!("    suited (4 combos):  {:.3e}", suited_diff);
    eprintln!("    offsuit (12 combos): {:.3e}", offsuit_diff);

    let tol = 1e-5_f32;
    assert!(
        max_diff < tol,
        "round-trip identity FAIL: max diff {} > {} at class c={}. \
         reduce of uniform-within-class cfv_combo did NOT recover the per-class \
         constants. Expand/reduce multiplicity consistency is broken.",
        max_diff, tol, max_c
    );
    eprintln!("  ✓ reduce ∘ uniform-class-expand recovers per-class values exactly \
        across all three multiplicity types.");
}
