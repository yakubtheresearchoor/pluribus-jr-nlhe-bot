// P1.5.4 Slice A.3c: class×class blocking matrix + per-class fold-terminal CFV.
//
// Validates (1) the 169×169 non-blocking-fraction matrix against direct
// combo enumeration on known class pairs, and (2) the fold-terminal CFV
// helper's composition arithmetic.

use solver_core::abstraction::preflop_class::{class_combos, NUM_PREFLOP_CLASSES, PreflopClass};
use solver_core::solver::preflop_terminal::{
    build_class_blocking_matrix, preflop_fold_terminal_cfv_hu,
};

/// Independent reference: count non-blocking pairs by direct enumeration.
fn direct_non_blocking_fraction(c: usize, cp: usize) -> f32 {
    let combos_c = class_combos(PreflopClass(c as u8));
    let combos_cp = class_combos(PreflopClass(cp as u8));
    let total = combos_c.len() * combos_cp.len();
    if total == 0 { return 0.0; }
    let mut non_blocking = 0_usize;
    for &(a1, a2) in &combos_c {
        for &(b1, b2) in &combos_cp {
            if a1 != b1 && a1 != b2 && a2 != b1 && a2 != b2 {
                non_blocking += 1;
            }
        }
    }
    non_blocking as f32 / total as f32
}

#[test]
fn slice_a3c_blocking_matrix_matches_direct_enumeration_on_all_pairs() {
    let m = build_class_blocking_matrix();
    assert_eq!(m.len(), NUM_PREFLOP_CLASSES * NUM_PREFLOP_CLASSES);

    let mut max_diff = 0.0_f32;
    let mut max_loc = (0_usize, 0_usize);
    for c in 0..NUM_PREFLOP_CLASSES {
        for cp in 0..NUM_PREFLOP_CLASSES {
            let got = m[c * NUM_PREFLOP_CLASSES + cp];
            let expected = direct_non_blocking_fraction(c, cp);
            let d = (got - expected).abs();
            if d > max_diff { max_diff = d; max_loc = (c, cp); }
        }
    }
    eprintln!("blocking matrix vs direct enumeration: max_diff = {:.4e} at ({}, {})",
        max_diff, max_loc.0, max_loc.1);
    assert_eq!(max_diff, 0.0,
        "blocking matrix diverges from direct enumeration");
}

#[test]
fn slice_a3c_blocking_matrix_known_invariants() {
    let m = build_class_blocking_matrix();

    // Invariant 1: pair vs pair of different rank (e.g., AA = class 0 vs 22 = class 12)
    // → ranks differ, no card overlap possible → 1.0.
    //
    // Class index convention: pair AA = class 0 (rank 12 << ranks down), 22 = class 12.
    // Let's not assume that and verify against direct enumeration.
    let aa_idx = 0;  // First pair = AA per the codebase's convention (rank 12 = "A")
    let combos_aa = class_combos(PreflopClass(aa_idx as u8));
    assert_eq!(combos_aa.len(), 6, "AA should have 6 combos (pair)");
    // Find a pair-class that doesn't share any rank with AA. Try 22 etc by iterating.
    let mut found_disjoint_pair = false;
    for cp in 0..NUM_PREFLOP_CLASSES {
        if !PreflopClass(cp as u8).is_pair() { continue; }
        if cp == aa_idx { continue; }
        let combos_cp = class_combos(PreflopClass(cp as u8));
        // Ranks of pair cp:
        let p_rank = combos_cp[0].0 >> 2;
        let aa_rank = combos_aa[0].0 >> 2;
        if p_rank != aa_rank {
            // Different ranks → no overlap possible.
            let frac = m[aa_idx * NUM_PREFLOP_CLASSES + cp];
            assert_eq!(frac, 1.0,
                "pair {:?}(rank {}) vs pair {:?}(rank {}): different ranks, expect M = 1.0, got {}",
                PreflopClass(aa_idx as u8), aa_rank, PreflopClass(cp as u8), p_rank, frac);
            found_disjoint_pair = true;
            break;
        }
    }
    assert!(found_disjoint_pair, "couldn't find another pair class with rank ≠ AA's rank");

    // Invariant 2: pair vs itself (e.g., AA vs AA): self-blocking is high.
    // For 6 AA combos, the (combo, combo) pairs that DON'T share any card: pick combo A1 = AsAh, combo A2 = AcAd → no shared card.
    // Count directly: of the 6×6 = 36 (combo_AA, combo_AA) pairs, how many have no shared card?
    // Pairs of disjoint combos from AA: 6 combos use 4 suits in pairs (AsAh, AsAd, AsAc, AhAd, AhAc, AdAc).
    // Two combos are disjoint iff they use 4 distinct suits. The 6 combos pair up into 3 disjoint pairs:
    //   (AsAh, AcAd), (AsAd, AhAc), (AsAc, AhAd).
    // Each disjoint pair contributes 2 ordered pairs (a,b) and (b,a). Total = 6 non-blocking out of 36.
    let aa_self = m[aa_idx * NUM_PREFLOP_CLASSES + aa_idx];
    let expected_aa_self = 6.0 / 36.0;
    assert!((aa_self - expected_aa_self).abs() < 1e-7,
        "AA vs AA self-blocking: got {}, expected {} (= 6/36)", aa_self, expected_aa_self);
    eprintln!("AA vs AA self-blocking = {} (= 6/36 ≈ {:.4})", aa_self, expected_aa_self);

    // Invariant 3: every entry is in [0, 1].
    for &v in &m {
        assert!(v >= 0.0 && v <= 1.0, "entry out of [0,1]: {}", v);
    }
}

#[test]
fn slice_a3c_fold_terminal_cfv_composition() {
    let m = build_class_blocking_matrix();

    // Case 1: uniform opp_reach = all-ones, chip_delta = 1.0.
    // v[c] = Σ_c' M[c, c'] = row sum.
    let opp_reach = vec![1.0_f32; NUM_PREFLOP_CLASSES];
    let v = preflop_fold_terminal_cfv_hu(&opp_reach, &m, 1.0);
    assert_eq!(v.len(), NUM_PREFLOP_CLASSES);

    // Verify against direct row-sum.
    let mut max_diff = 0.0_f32;
    for c in 0..NUM_PREFLOP_CLASSES {
        let expected: f32 = (0..NUM_PREFLOP_CLASSES)
            .map(|cp| m[c * NUM_PREFLOP_CLASSES + cp])
            .sum();
        let d = (v[c] - expected).abs();
        if d > max_diff { max_diff = d; }
    }
    eprintln!("v vs row sums: max_diff = {:.4e}", max_diff);
    assert!(max_diff < 1e-5,
        "fold-terminal CFV vs direct row sum: max_diff {:.4e} too large", max_diff);

    // Case 2: chip_delta = 0 → v all-zero.
    let v_zero = preflop_fold_terminal_cfv_hu(&opp_reach, &m, 0.0);
    for &val in &v_zero {
        assert_eq!(val, 0.0, "chip_delta=0 should produce v all-zero, got {}", val);
    }

    // Case 3: opp_reach zero everywhere except class 0 = 1.0.
    // v[c] = chip_delta × M[c, 0]
    let mut single_class_reach = vec![0.0_f32; NUM_PREFLOP_CLASSES];
    single_class_reach[0] = 1.0;
    let chip_delta = 2.5_f32;
    let v_single = preflop_fold_terminal_cfv_hu(&single_class_reach, &m, chip_delta);
    for c in 0..NUM_PREFLOP_CLASSES {
        let expected = chip_delta * m[c * NUM_PREFLOP_CLASSES + 0];
        let d = (v_single[c] - expected).abs();
        assert!(d < 1e-6,
            "single-class opp reach: v[{}] = {}, expected {} (chip_delta × M[{}, 0])",
            c, v_single[c], expected, c);
    }

    // Case 4: chip_delta sign propagates.
    let v_pos = preflop_fold_terminal_cfv_hu(&opp_reach, &m, 1.0);
    let v_neg = preflop_fold_terminal_cfv_hu(&opp_reach, &m, -1.0);
    for c in 0..NUM_PREFLOP_CLASSES {
        assert!((v_pos[c] + v_neg[c]).abs() < 1e-6,
            "sign symmetry: v_pos[{}] = {}, v_neg[{}] = {} (should sum to ~0)",
            c, v_pos[c], c, v_neg[c]);
    }

    eprintln!("Slice A.3c PASS: blocking matrix matches direct enumeration exactly; \
              row-sum, single-class, zero-delta, and sign-symmetry invariants hold on the \
              fold-terminal CFV helper.");
}
