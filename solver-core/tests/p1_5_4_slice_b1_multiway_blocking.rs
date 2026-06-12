// P1.5.4 Slice B.1: multiway blocking primitive.
//
// First piece of the 6-max rework. Per the lead's directive (2026-06-04):
// at multiway, blocking does NOT decompose pairwise — events
// "hand_i conflicts with hand_j" are correlated through shared cards
// across multiple pairs, and a pairwise-product approximation
// systematically over-counts non-blocking tuples by missing cross-pair
// card conflicts. This is the same inclusion-exclusion shape that
// produced the postflop showdown bug (site (d) num_opp=1 fold-fast-path
// error); the multiway blocking primitive must be anchored against
// formula-free enumeration the way the showdown oracle was.
//
// Three validations:
//
//   1. **N=2 (HU sanity)**: joint_class_tuple_non_blocking_fraction
//      matches build_class_blocking_matrix entries exactly. At N=2 there
//      are no higher-order terms, so the joint primitive must reduce to
//      the existing HU blocking matrix.
//
//   2. **N=3 (the pairwise-trap exposure)**: on AA-AA-AA, the joint
//      answer is 0 (three AA combos need 6 aces, only 4 exist), while
//      the pairwise-product approximation gives (6/36)^3 ≈ 0.46%
//      non-zero. The two MUST disagree; if they don't, the joint
//      primitive is buggy (specifically, it has fallen into the
//      pairwise-decomposition trap).
//
//   3. **N=3 (small per-class samples)**: a handful of triples
//      cross-checked against an independent re-enumeration in the test
//      file. Sanity that the joint counts match independent enumeration
//      on cases the formula approach would not get wrong by accident.

use solver_core::abstraction::preflop_class::{class_combos, NUM_PREFLOP_CLASSES, PreflopClass};
use solver_core::card::Card;
use solver_core::solver::preflop_terminal::{
    build_class_blocking_matrix,
    joint_class_tuple_non_blocking_count,
    joint_class_tuple_non_blocking_fraction,
};

/// Independent reference: brute-force count joint non-blocking tuples
/// for a class tuple. Mirrors the production algorithm; included so the
/// test is self-contained and any production refactor must still match.
fn reference_joint_non_blocking_count(classes: &[PreflopClass]) -> u64 {
    let lists: Vec<Vec<(Card, Card)>> = classes.iter()
        .map(|&c| class_combos(c))
        .collect();
    fn enumerate(
        lists: &[Vec<(Card, Card)>],
        used: u64,
        depth: usize,
        count: &mut u64,
    ) {
        if depth == lists.len() { *count += 1; return; }
        for &(c1, c2) in &lists[depth] {
            let m = (1_u64 << c1) | (1_u64 << c2);
            if used & m == 0 {
                enumerate(lists, used | m, depth + 1, count);
            }
        }
    }
    let mut count = 0_u64;
    enumerate(&lists, 0, 0, &mut count);
    count
}

/// The LOSSY pairwise-product approximation. Computes
/// `Π_{i<j} M_HU[c_i, c_j]` where M_HU is the existing HU blocking
/// matrix. This is the trap — included in the test so the discriminating
/// test can show it produces WRONG answers.
fn lossy_pairwise_product_approximation(classes: &[PreflopClass]) -> f32 {
    let m = build_class_blocking_matrix();
    let mut prod = 1.0_f32;
    for i in 0..classes.len() {
        for j in (i + 1)..classes.len() {
            let ci = classes[i].index();
            let cj = classes[j].index();
            prod *= m[ci * NUM_PREFLOP_CLASSES + cj];
        }
    }
    prod
}

#[test]
fn slice_b1_n2_matches_hu_blocking_matrix() {
    // At N=2 the joint primitive must reduce to the HU blocking matrix
    // (no higher-order terms; joint == pairwise). Check on all 169 × 169
    // class pairs at exact 0 diff.
    let m_hu = build_class_blocking_matrix();
    let mut max_diff = 0.0_f32;
    let mut max_loc = (0_usize, 0_usize);
    for c in 0..NUM_PREFLOP_CLASSES {
        for cp in 0..NUM_PREFLOP_CLASSES {
            let joint = joint_class_tuple_non_blocking_fraction(
                &[PreflopClass(c as u8), PreflopClass(cp as u8)]
            );
            let pairwise = m_hu[c * NUM_PREFLOP_CLASSES + cp];
            let d = (joint - pairwise).abs();
            if d > max_diff { max_diff = d; max_loc = (c, cp); }
        }
    }
    eprintln!("N=2 joint vs HU blocking matrix: max_diff = {:.4e} at ({}, {})",
        max_diff, max_loc.0, max_loc.1);
    assert_eq!(max_diff, 0.0,
        "joint primitive at N=2 must exactly equal HU blocking matrix");
}

#[test]
fn slice_b1_n3_aa_aa_aa_exposes_pairwise_trap() {
    // The discriminating test. Three players all holding AA need 6
    // aces to be all disjoint; only 4 aces exist. Joint count = 0.
    // The pairwise approximation gives (M[AA,AA])^3 ≈ (1/6)^3 ≈ 0.46%,
    // confidently non-zero. The two MUST disagree.
    let aa = PreflopClass(0);
    let joint_count = joint_class_tuple_non_blocking_count(&[aa, aa, aa]);
    let joint = joint_class_tuple_non_blocking_fraction(&[aa, aa, aa]);
    let pairwise = lossy_pairwise_product_approximation(&[aa, aa, aa]);

    eprintln!("AA-AA-AA joint count = {} (must be 0; 3 AA combos need 6 aces)", joint_count);
    eprintln!("AA-AA-AA joint fraction = {:.4e}", joint);
    eprintln!("AA-AA-AA pairwise approximation = {:.4e} ({:.4} basis points)",
        pairwise, pairwise * 10000.0);

    assert_eq!(joint_count, 0,
        "AA-AA-AA joint count must be 0 (4 aces total, 3 hands × 2 cards each = 6 needed)");
    assert_eq!(joint, 0.0, "joint fraction must be exactly 0");
    assert!(pairwise > 0.001,
        "pairwise approximation must be confidently nonzero ({}) for this test to discriminate",
        pairwise);
    let gap = (joint - pairwise).abs();
    assert!(gap > 0.001,
        "joint and pairwise MUST disagree (gap = {:.4e}). If they agree, the joint primitive \
         has fallen into the pairwise-decomposition trap and is structurally wrong.",
        gap);

    eprintln!("Slice B.1 N=3 discriminator PASS: joint = {} (truth), pairwise = {} (lossy approximation), \
              gap = {} confirms the joint primitive avoids the pairwise trap.",
        joint, pairwise, gap);
}

#[test]
fn slice_b1_n3_kk_qq_jj_disjoint_ranks_all_combos_compatible() {
    // Three players with KK, QQ, JJ — disjoint ranks, so no card
    // conflict possible. Joint count = |KK combos| × |QQ combos| ×
    // |JJ combos| = 6 × 6 × 6 = 216. Joint fraction = 216/216 = 1.0.
    let kk = PreflopClass(1);  // KK = second pair after AA
    let qq = PreflopClass(2);
    let jj = PreflopClass(3);
    let joint_count = joint_class_tuple_non_blocking_count(&[kk, qq, jj]);
    let joint = joint_class_tuple_non_blocking_fraction(&[kk, qq, jj]);

    eprintln!("KK-QQ-JJ joint count = {} (expected 216 = 6³)", joint_count);
    eprintln!("KK-QQ-JJ joint fraction = {} (expected 1.0)", joint);
    assert_eq!(joint_count, 216);
    assert_eq!(joint, 1.0);

    // Pairwise gives the same answer when there's no joint constraint to miss:
    // KK vs QQ pairwise is 1.0 (no shared rank); same for KK vs JJ and QQ vs JJ;
    // product is 1.0. So on disjoint-rank triples, pairwise and joint AGREE.
    // The disagreement on AA-AA-AA was specifically the joint-constraint case.
    let pairwise = lossy_pairwise_product_approximation(&[kk, qq, jj]);
    eprintln!("KK-QQ-JJ pairwise approximation = {} (should agree with joint on disjoint ranks)",
        pairwise);
    assert!((pairwise - 1.0).abs() < 1e-6);
}

#[test]
fn slice_b1_matches_independent_reference_on_diverse_triples() {
    // Cross-check the production primitive against the in-test reference
    // enumeration on a handful of triples spanning class types
    // (pair / suited / offsuit).
    let cases: Vec<[PreflopClass; 3]> = vec![
        [PreflopClass(0), PreflopClass(0), PreflopClass(1)],   // AA-AA-KK
        [PreflopClass(0), PreflopClass(12), PreflopClass(24)], // AA-22-? offsuit-region edge
        [PreflopClass(13), PreflopClass(13), PreflopClass(13)],// first suited × 3 (same class)
        [PreflopClass(13), PreflopClass(14), PreflopClass(15)],// three different suited
        [PreflopClass(91), PreflopClass(92), PreflopClass(93)],// three offsuit (different)
        [PreflopClass(91), PreflopClass(91), PreflopClass(91)],// same offsuit × 3
    ];
    for case in &cases {
        let prod_count = joint_class_tuple_non_blocking_count(case);
        let ref_count = reference_joint_non_blocking_count(case);
        assert_eq!(prod_count, ref_count,
            "production joint count {} != reference {} for case {:?}",
            prod_count, ref_count, case);
    }
    eprintln!("Slice B.1 PASS: production joint primitive matches independent enumeration on {} \
              diverse class triples spanning pair/suited/offsuit; HU sanity at N=2 holds across \
              all 169×169 pairs at exact 0 diff; pairwise-decomposition trap is exposed on \
              AA-AA-AA where joint = 0 and pairwise = 0.0046.",
        cases.len());
}

#[test]
fn slice_b1_n4_aa_aa_aa_aa_impossible() {
    // 4 players all holding AA: need 8 aces, only 4 exist. Joint = 0.
    // Pairwise approximation would give (1/6)^6 ≈ 2.1e-5, still nonzero
    // (proving the pairwise trap gets WORSE at higher N as more
    // higher-order conflicts are missed).
    let aa = PreflopClass(0);
    let joint = joint_class_tuple_non_blocking_count(&[aa, aa, aa, aa]);
    let pairwise = lossy_pairwise_product_approximation(&[aa, aa, aa, aa]);
    assert_eq!(joint, 0, "AA × 4 needs 8 aces; impossible");
    assert!(pairwise > 1e-6,
        "pairwise approx must be nonzero for the discriminator to bite");
    eprintln!("N=4 AA quad: joint = 0 (impossible), pairwise = {:.4e} (wrong but nonzero)",
        pairwise);
}
