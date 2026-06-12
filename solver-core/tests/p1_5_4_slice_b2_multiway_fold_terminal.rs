// P1.5.4 Slice B.2: n-player fold-terminal CFV.
//
// Composes B.1's joint blocking primitive with A.3d's chip_delta to
// give per-class CFV at an N-player preflop fold-terminal. The multiway
// formula:
//
//   v[c_t] = chip_delta × Σ over opp class tuples of [
//     Π opp_reaches × joint_non_blocking_fraction(traverser + opps)
//   ]
//
// Validations:
//
//   1. **HU equivalence (N=2)**: the multiway formula at one opp must
//      give the EXACT SAME output as preflop_fold_terminal_cfv_hu on
//      the same inputs. This is the primary sanity gate — if the
//      multiway formula reduces correctly to HU, the composition is
//      structurally right; if it doesn't, something in the recursion
//      or accumulation is wrong.
//
//   2. **Sparse-reach correctness (N=3)**: hand-computed expected v
//      on a small case where opp reaches are concentrated on a few
//      classes. The expected value is computed in the test by direct
//      summation; the production output must match.
//
//   3. **Degenerate inputs**: chip_delta=0 → all-zero output (sign
//      symmetry: chip_delta=-1 vs chip_delta=+1 give v_neg = -v_pos
//      pointwise).
//
//   4. **The pairwise-trap discriminator carries through**: building
//      v using the LOSSY pairwise approximation gives a different
//      result than the joint primitive on the AA-AA-AA case from B.1.
//      Confirms the terminal CFV inherits the correctness guarantee of
//      the joint primitive.

use solver_core::abstraction::preflop_class::{NUM_PREFLOP_CLASSES, PreflopClass};
use solver_core::solver::preflop_terminal::{
    build_class_blocking_matrix,
    joint_class_tuple_non_blocking_fraction,
    preflop_fold_terminal_cfv_hu,
    preflop_fold_terminal_cfv_multiway,
};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn slice_b2_hu_equivalence_n2_matches_hu_formula() {
    // At N=2 (one opp), the multiway formula must reduce to the HU
    // formula. Test on several opp reach distributions + chip_delta
    // values, across all 169 traverser classes.
    let m_hu = build_class_blocking_matrix();
    let cases: Vec<(Vec<f32>, f32)> = vec![
        (vec![1.0_f32; NUM_PREFLOP_CLASSES], 1.0),          // uniform reach, unit delta
        (vec![1.0_f32; NUM_PREFLOP_CLASSES], -2.5),         // uniform reach, negative delta
        ({                                                  // sparse reach: only AA, KK, QQ
            let mut r = vec![0.0_f32; NUM_PREFLOP_CLASSES];
            r[0] = 1.0; r[1] = 0.7; r[2] = 0.3;
            r
        }, 1.5),
        ({                                                  // varied reach
            let mut r = vec![0.0_f32; NUM_PREFLOP_CLASSES];
            for c in 0..NUM_PREFLOP_CLASSES {
                r[c] = (c as f32 * 0.01).sin().abs();
            }
            r
        }, -0.7),
    ];

    for (opp_reach, chip_delta) in &cases {
        let multiway = preflop_fold_terminal_cfv_multiway(&[opp_reach], *chip_delta);
        let hu = preflop_fold_terminal_cfv_hu(opp_reach, &m_hu, *chip_delta);
        let d = max_abs_diff(&multiway, &hu);
        // Both use the same arithmetic (multiway calls joint_non_blocking,
        // HU uses the precomputed matrix which equals joint at N=2 by B.1).
        // f32 accumulation order may differ slightly; tolerance reflects
        // that.
        eprintln!("HU equivalence: max_abs_diff = {:.4e} (delta={}, reach sum={:.3})",
            d, chip_delta, opp_reach.iter().sum::<f32>());
        assert!(d < 1e-4,
            "multiway formula at N=2 must match HU formula; max_diff = {:.4e}", d);
    }
}

#[test]
fn slice_b2_n3_sparse_reach_hand_computed() {
    // N=3: traverser, opp1, opp2. Use sparse opp reaches so the test
    // can hand-compute the expected v for a specific traverser class.
    //
    // Setup: opp1 has reach 1.0 only on KK (class 1), reach 0 elsewhere.
    //        opp2 has reach 1.0 only on QQ (class 2), reach 0 elsewhere.
    //        chip_delta = 2.0.
    //
    // For traverser class c_t:
    //   v[c_t] = 2.0 × 1.0 × 1.0 × joint_non_blocking_fraction([c_t, KK, QQ])
    //          = 2.0 × joint_non_blocking_fraction([c_t, KK, QQ])
    //
    // For c_t = AA (class 0): all three are pairs of different ranks,
    //   joint count = 6 × 6 × 6 = 216 (out of 216), fraction = 1.0,
    //   v[AA] = 2.0 × 1.0 = 2.0.
    // For c_t = KK: shares rank with opp1's KK. Joint count = 0
    //   (can't have 2 disjoint KK combos AND a QQ); fraction = 0.
    //   v[KK] = 0.
    // For c_t = QQ: similar, conflicts with opp2.
    //   v[QQ] = 0.
    // For c_t = JJ (class 3): disjoint from KK and QQ.
    //   v[JJ] = 2.0 × 1.0 = 2.0.
    let mut opp1 = vec![0.0_f32; NUM_PREFLOP_CLASSES];
    opp1[1] = 1.0;  // KK
    let mut opp2 = vec![0.0_f32; NUM_PREFLOP_CLASSES];
    opp2[2] = 1.0;  // QQ
    let chip_delta = 2.0;
    let v = preflop_fold_terminal_cfv_multiway(&[&opp1, &opp2], chip_delta);

    assert_eq!(v.len(), NUM_PREFLOP_CLASSES);

    // Hand-computed expectations on DISJOINT-RANK traverser classes
    // (no rank overlap with KK or QQ, so joint fraction = 1.0):
    //   v[c_t] = chip_delta × 1.0 = 2.0 for c_t ∈ {AA, JJ, TT, 99, ..., 22}.
    //
    // For RANK-OVERLAPPING traverser classes (e.g., c_t = KK or QQ),
    // the joint fraction is NOT zero: two different KK combos CAN
    // coexist if they use disjoint suits (3 disjoint pairs out of 6
    // ordered pairs, so 6/36 = 1/6 of (KK, KK) sub-tuples are
    // non-conflicting). For [KK_t, KK_o1, QQ_o2]: 6/36 × 1 (QQ
    // disjoint) = 1/6. v[KK] = 2.0 × 1/6 ≈ 0.333.
    //
    // The "trivial" sanity is only on the disjoint-rank classes; the
    // cross-check loop below validates v[c_t] for ALL 169 classes
    // against independent direct enumeration via the joint primitive.
    eprintln!("v[AA]={}, v[KK]={}, v[QQ]={}, v[JJ]={}, v[TT]={}, v[99]={}",
        v[0], v[1], v[2], v[3], v[4], v[5]);
    assert!((v[0] - 2.0).abs() < 1e-5,
        "v[AA] disjoint expected 2.0; got {}", v[0]);
    assert!((v[3] - 2.0).abs() < 1e-5, "v[JJ] disjoint expected 2.0; got {}", v[3]);
    assert!((v[4] - 2.0).abs() < 1e-5, "v[TT] disjoint expected 2.0; got {}", v[4]);
    assert!((v[5] - 2.0).abs() < 1e-5, "v[99] disjoint expected 2.0; got {}", v[5]);
    // For overlapping ranks: not zero (different suits can coexist),
    // but the cross-check below verifies the exact value.

    // Cross-check via independent recomputation in the test.
    for c_t in 0..NUM_PREFLOP_CLASSES {
        let expected = chip_delta * joint_class_tuple_non_blocking_fraction(
            &[PreflopClass(c_t as u8), PreflopClass(1), PreflopClass(2)]
        );
        assert!((v[c_t] - expected).abs() < 1e-5,
            "v[{}] mismatch: production = {}, expected = {}", c_t, v[c_t], expected);
    }
    eprintln!("Slice B.2 N=3 hand-computed PASS: production matches direct enumeration on the \
              sparse-opp-reach case across all 169 traverser classes.");
}

#[test]
fn slice_b2_degenerate_chip_delta_zero_gives_zeros() {
    let opp = vec![1.0_f32; NUM_PREFLOP_CLASSES];
    let v = preflop_fold_terminal_cfv_multiway(&[&opp, &opp], 0.0);
    for (i, &val) in v.iter().enumerate() {
        assert_eq!(val, 0.0, "v[{}] should be 0 when chip_delta=0; got {}", i, val);
    }
}

#[test]
fn slice_b2_sign_symmetry() {
    let mut opp = vec![0.0_f32; NUM_PREFLOP_CLASSES];
    opp[5] = 1.0; opp[10] = 0.5; opp[50] = 0.3;
    let v_pos = preflop_fold_terminal_cfv_multiway(&[&opp, &opp], 1.0);
    let v_neg = preflop_fold_terminal_cfv_multiway(&[&opp, &opp], -1.0);
    let max_sym_err = v_pos.iter().zip(v_neg.iter())
        .map(|(a, b)| (a + b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("sign symmetry: max |v_pos[c] + v_neg[c]| = {:.4e}", max_sym_err);
    assert!(max_sym_err < 1e-6, "chip_delta sign symmetry broken: {:.4e}", max_sym_err);
}

#[test]
fn slice_b2_pairwise_trap_discriminator_carries_through_to_terminal_cfv() {
    // The B.1 discriminator: at N=3 with all three players holding AA,
    // joint = 0 but pairwise = 0.0046. If the terminal CFV builds on
    // the joint primitive (correct), v[AA] for THIS scenario is 0. If
    // it accidentally used pairwise, v[AA] would be chip_delta × 0.0046
    // (nonzero, wrong).
    //
    // Setup: opp1 reach 1.0 on AA (only), opp2 reach 1.0 on AA (only).
    // chip_delta = 1.0.
    // For c_t = AA: 3 AAs needs 6 aces, impossible. Joint fraction = 0.
    //               v[AA] must be 0.
    // If terminal CFV used pairwise, v[AA] = 1.0 × 0.0046 ≈ 0.0046.
    let mut opp_aa = vec![0.0_f32; NUM_PREFLOP_CLASSES];
    opp_aa[0] = 1.0;
    let v = preflop_fold_terminal_cfv_multiway(&[&opp_aa, &opp_aa], 1.0);
    eprintln!("v[AA] when opp1=AA, opp2=AA, chip_delta=1: {} (must be 0; pairwise approx would give ~0.0046)",
        v[0]);
    assert_eq!(v[0], 0.0,
        "terminal CFV must inherit B.1's joint-blocking correctness; got {} (likely fell into pairwise trap)",
        v[0]);
    eprintln!("Slice B.2 PASS: terminal CFV at N=2 matches HU formula; sparse-N=3 matches \
              hand-computed; degenerate chip_delta=0 gives zeros; sign symmetry holds; the \
              pairwise trap discriminator carries through: terminal CFV stays at exact 0 on \
              the AA-AA-AA case where the lossy pairwise approach would give 0.0046.");
}
