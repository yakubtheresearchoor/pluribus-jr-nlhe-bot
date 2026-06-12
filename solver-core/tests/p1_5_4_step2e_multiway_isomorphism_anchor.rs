// Step 2.E: multiway isomorphism anchor — extend the existing per-class
// orbit-weighted = un-canonicalized ground-truth check to np ∈ {2, 3, 6}.
//
// PER USER (banked 2026-06): "verify it's correct, especially at multiway
// scale (the same find-by-measurement discipline — this project's
// canonicalization has had bugs). Anchor against ground truth (does the
// isomorphism-reduced tree represent the same game as the full tree),
// not against another implementation."
//
// EXISTING ANCHORS (all pass, all structurally np-independent):
// - flop_isomorphism::canonical_form_is_deterministic
// - flop_isomorphism::orbit_respecting_under_suit_permutation
// - flop_isomorphism::canonical_count_is_1755 (literature ground-truth)
// - flop_isomorphism::orbits_partition_all_flops (22,100 combinatorial GT)
// - preflop_class::per_class_compat_count_is_orbit_invariant
// - preflop_class::orbit_weighted_total_matches_uncanonicalized (the
//   load-bearing structural-independence anchor)
//
// WHAT THIS TEST ADDS: the existing anchors operate on COUNTS (compat
// combos). The runtime uses PROBABILITIES (chance_probability_flop). This
// test anchors the PROBABILITY-LEVEL aggregation at np ∈ {2, 3, 6}.
//
// Structural argument up front: the per-canonical chance probability
// formula P(F | class h) = (orbit_size × expansion_size) / (n_class × 19600)
// is per-class, np-independent. The np ONLY enters via class_weights
// (one row per player) which the chance_probability_flop computation
// does NOT consume. So the per-(canonical, class) probabilities are
// identical for np=2, 3, 6. This test verifies that empirically and
// then verifies the load-bearing aggregation property: for each class,
// the orbit-weighted sum of probabilities equals 1.0 (the sum-to-one
// property), which is the ground-truth that distinguishes a correct
// chance distribution from a corrupted one.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::preflop_start_game::PreflopChanceTable;

fn build_table_with_np(np: u8, asymmetric: bool) -> PreflopChanceTable {
    let mut class_weights: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();
    for p in 0..(np as usize) {
        for k in 0..NUM_PREFLOP_CLASSES {
            let s = k as f32 / NUM_PREFLOP_CLASSES as f32;
            let w = if asymmetric {
                // Per-player asymmetric sigmoid — different shape per player
                // to ensure no accidental per-player symmetry could mask a bug.
                ((s - 0.3 - 0.05 * p as f32).max(0.05) * (1.5 - 0.1 * p as f32)).min(1.0)
            } else {
                1.0
            };
            class_weights[p][k] = w;
        }
    }
    PreflopChanceTable::new(np, class_weights)
}

#[test]
fn step2e_multiway_chance_probability_is_np_independent() {
    // The per-(canonical, class) probability formula is structurally
    // np-independent. Verify empirically across np ∈ {2, 3, 6}.

    let tables: Vec<(u8, PreflopChanceTable)> = vec![
        (2, build_table_with_np(2, true)),
        (3, build_table_with_np(3, true)),
        (6, build_table_with_np(6, true)),
    ];

    let n_canon = tables[0].1.num_canonical_flops();
    assert_eq!(n_canon, 1755);

    // Sample 50 canonicals × all 169 classes — exhaustive within tractability.
    let sample_indices: Vec<usize> = (0..50).map(|i| i * (n_canon / 50)).collect();

    for &canonical_idx in &sample_indices {
        for class_idx in 0..NUM_PREFLOP_CLASSES {
            let p_ref = tables[0].1.chance_probability_flop(canonical_idx, class_idx);
            for (np, t) in tables.iter().skip(1) {
                let p_here = t.chance_probability_flop(canonical_idx, class_idx);
                assert_eq!(p_ref.to_bits(), p_here.to_bits(),
                    "np={}: chance_probability_flop differs at canonical {}, class {}: \
                     np=2 → {}, np={} → {}. The per-(canonical, class) probability \
                     is supposed to be np-independent; this would corrupt multiway \
                     chance integration.",
                    np, canonical_idx, class_idx, p_ref, np, p_here);
            }
        }
    }
}

#[test]
fn step2e_sum_over_canonicals_is_one_per_class_for_each_np() {
    // The load-bearing ground-truth property: for each class h, the sum
    // over canonical flops of P(F | h) must equal 1.0 (within f32
    // rounding). This is the sum-to-one property of a valid conditional
    // probability distribution.
    //
    // Verifying this for np ∈ {2, 3, 6} establishes that the chance
    // distribution is structurally well-formed at every multiway scale.

    for &np in &[2u8, 3, 6] {
        let table = build_table_with_np(np, true);
        let n_canon = table.num_canonical_flops();
        for class_idx in 0..NUM_PREFLOP_CLASSES {
            // Sum in f64 to separate accumulation precision from the
            // underlying probability quality.
            let sum_f64: f64 = (0..n_canon)
                .map(|ci| table.chance_probability_flop(ci, class_idx) as f64)
                .sum();
            let err = (sum_f64 - 1.0).abs();
            assert!(err < 1e-5,
                "np={}: class {}: Σ P(F | h) = {:.10}, expected 1.0 (err {:.2e}). \
                 The chance distribution is malformed; multiway aggregation will \
                 silently produce wrong values.",
                np, class_idx, sum_f64, err);
        }
    }
}

#[test]
fn step2e_aggregate_preflop_chance_is_np_independent() {
    // Discriminating multiway anchor: aggregate_preflop_chance must
    // produce the SAME output regardless of np when fed the same per-
    // canonical leaf values. This is because the formula
    //   out[h] = Σ_F P(F | h) × flop_cfvs[F][h]
    // depends only on the per-(canonical, class) probability, which
    // step2e_multiway_chance_probability_is_np_independent established
    // is np-independent.
    //
    // If a future change introduces np-specific code into the aggregate
    // path (e.g., a multiway-only correction term), this test catches it
    // immediately. Anchoring at f32 floor against the np=2 reference.

    use solver_core::solver::preflop_start_game::aggregate_preflop_chance;

    let table_2 = build_table_with_np(2, true);
    let table_3 = build_table_with_np(3, true);
    let table_6 = build_table_with_np(6, true);
    let n_canon = table_2.num_canonical_flops();

    // Asymmetric per-(canonical, class) leaf values — deterministic.
    let make_leaves = || -> Vec<Vec<f32>> {
        (0..n_canon).map(|ci| {
            let f = table_2.canonical_flops[ci];
            let canon_seed: u64 = (f[0] as u64) << 16 | (f[1] as u64) << 8 | (f[2] as u64);
            (0..NUM_PREFLOP_CLASSES).map(|c| {
                let mix: u64 = canon_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                let bits = ((mix >> 32) & 0xFFFFFF) as i64 - (1 << 23);
                (bits as f32) / ((1 << 23) as f32)
            }).collect()
        }).collect()
    };
    let leaves = make_leaves();

    let agg_2 = aggregate_preflop_chance(&table_2, &leaves);
    let agg_3 = aggregate_preflop_chance(&table_3, &leaves);
    let agg_6 = aggregate_preflop_chance(&table_6, &leaves);

    for c in 0..NUM_PREFLOP_CLASSES {
        assert_eq!(agg_2[c].to_bits(), agg_3[c].to_bits(),
            "aggregate_preflop_chance not np-independent: np=2 → {}, np=3 → {} at class {}",
            agg_2[c], agg_3[c], c);
        assert_eq!(agg_2[c].to_bits(), agg_6[c].to_bits(),
            "aggregate_preflop_chance not np-independent: np=2 → {}, np=6 → {} at class {}",
            agg_2[c], agg_6[c], c);
    }
}

#[test]
fn step2e_aggregate_against_uncanonicalized_anchor_multiway() {
    // The load-bearing ground-truth anchor at the PRODUCTION arithmetic
    // level. Mirrors P5a (which is HU at f64) but extended to multiway
    // np ∈ {2, 3, 6}. Since aggregate_preflop_chance is structurally
    // np-independent (proven above), this test confirms the result is
    // ALSO correct against un-canonicalized ground truth at every np.
    //
    // Un-canonicalized reference: for each class h, the chance-integrated
    // value is Σ_f (compat(h, f) / (n_h × 19600)) × leaf(canonicalize(f)).
    // The canonicalize step picks each f's orbit representative; multiple
    // f's map to the same canonical leaf value. This is the un-canon
    // expansion of Σ_F P(F | h) × leaf(F).

    use solver_core::abstraction::flop_isomorphism::{canonicalize_flop, enumerate_all_flops};
    use solver_core::abstraction::preflop_class::{expansion, PreflopClass};
    use solver_core::solver::preflop_start_game::aggregate_preflop_chance;
    use std::collections::HashMap;

    for &np in &[2u8, 3, 6] {
        let table = build_table_with_np(np, true);
        let n_canon = table.num_canonical_flops();

        // Build per-canonical leaf values.
        let leaves: Vec<Vec<f32>> = (0..n_canon).map(|ci| {
            let f = table.canonical_flops[ci];
            let canon_seed: u64 = (f[0] as u64) << 16 | (f[1] as u64) << 8 | (f[2] as u64);
            (0..NUM_PREFLOP_CLASSES).map(|c| {
                let mix: u64 = canon_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                let bits = ((mix >> 32) & 0xFFFFFF) as i64 - (1 << 23);
                (bits as f32) / ((1 << 23) as f32)
            }).collect()
        }).collect();

        // Production aggregate.
        let prod = aggregate_preflop_chance(&table, &leaves);

        // Independent reference: enumerate all 22100 flops; per flop,
        // accumulate compat(h, f) × leaf(canonicalize(f)). Then divide
        // by (n_h × 19600). Built in f64 to keep precision distinct from
        // production's f32.
        let canon_to_idx: HashMap<[u8; 3], usize> = (0..n_canon)
            .map(|ci| (table.canonical_flops[ci], ci)).collect();
        let mut ref_out = vec![0.0f64; NUM_PREFLOP_CLASSES];
        let all_flops = enumerate_all_flops();
        for class_idx in 0..NUM_PREFLOP_CLASSES {
            let class = PreflopClass(class_idx as u8);
            let n_h = class.num_combos() as f64;
            let mut sum = 0.0f64;
            for &f in &all_flops {
                let compat = expansion(class, f).len() as f64;
                if compat == 0.0 { continue; }
                let canon = canonicalize_flop(f);
                let ci = canon_to_idx[&canon];
                let leaf = leaves[ci][class_idx] as f64;
                sum += compat * leaf;
            }
            ref_out[class_idx] = sum / (n_h * 19_600.0);
        }

        // Compare. Tolerance must accept f32-vs-f64 accumulation drift —
        // production uses f32 throughout, reference uses f64. P5a chose
        // ~5e-7; the same tolerance is appropriate at every np.
        for c in 0..NUM_PREFLOP_CLASSES {
            let p_f64 = prod[c] as f64;
            let err = (p_f64 - ref_out[c]).abs();
            assert!(err < 5e-7,
                "np={}: class {}: production = {:.10}, un-canonicalized reference = {:.10}, \
                 err = {:.3e}. The orbit-weighted aggregation at multiway is NOT equal to \
                 the un-canonicalized enumeration. This is the ground-truth anchor; \
                 failure means canonicalization corrupts the multiway distribution.",
                np, c, p_f64, ref_out[c], err);
        }
    }
}
