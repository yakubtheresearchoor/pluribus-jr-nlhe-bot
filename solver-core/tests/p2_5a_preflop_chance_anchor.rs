//! P2.5a: stub-based anchor for the preflop chance integration arithmetic.
//!
//! THE LOAD-BEARING REQUIREMENT IS STRUCTURAL INDEPENDENCE.
//!
//! The runtime preflop CFV computation calls THREE pieces of new arithmetic:
//!   1. P1.5.5a `reduce_cfv_combo_to_class` (combo → class averaging,
//!      with per-(class, canonical-flop) survivor count denominator)
//!   2. P1.5.4 `aggregate_preflop_chance` (orbit-weighted sum over
//!      1,755 canonicals)
//!   3. P1.5.1 `chance_probability_flop` (orbit weight × expansion size
//!      probability formula)
//!
//! All three are anchored at the primitive level by their own targeted
//! checks. P2.5a anchors their COMPOSITION: does the full runtime path
//! give the right answer when fed an arbitrary leaf-value stub?
//!
//! The validation arc's inclusion-exclusion lesson — agreement between
//! un-anchored sides is not correctness — applies here doubly: the
//! reference for P2.5a must NOT call any of the three primitives. Both
//! sides share ONLY the stub V_flop function (the leaf values); every
//! piece of arithmetic above the leaf is computed independently on each
//! side. The structural property to verify by READING the reference code:
//!
//!   - Reference loops over the 22,100 actual flops (not 1,755 canonicals)
//!   - Reference uses unit weight 1/19,600 per actual flop (not orbit weights)
//!   - Reference computes per-class average by explicit enumeration of
//!     class.num_combos() combos with explicit compatibility filtering
//!     and explicit averaging (not `reduce_cfv_combo_to_class` call)
//!
//! If the reference loops over canonicals OR uses orbit weights OR calls
//! reduce_cfv_combo_to_class, it is running the runtime's logic and
//! agreement proves nothing.
//!
//! ## Stub design
//!
//! The stub must satisfy:
//!   - Joint suit symmetry: V_stub(perm(F), perm(h)) = V_stub(F, h)
//!     for any suit permutation perm. This is the lossless-169
//!     property that makes the orbit-weighted aggregation exact.
//!   - NOT board-only symmetry: V_stub(perm(F), h) ≠ V_stub(F, h) in
//!     general (when perm doesn't fix h). Real V_flop satisfies the
//!     joint symmetry but NOT the spurious board-only symmetry (the
//!     nh=1326 finding: AsAc sees different backdoor flush draws in
//!     [KcKdKh] vs [KcKdKs] which are in the same flop orbit).
//!
//! A stub that is too symmetric (e.g., depending only on canonical flop
//! ignoring combo's specific suits) would mask bugs that only surface
//! against realistically asymmetric leaves. The stub below uses a
//! self-contained joint canonicalization (24 suit perms enumerated
//! inline, NOT calling `flop_isomorphism::canonicalize_flop`) so the
//! stub's symmetry computation is independent of the runtime's
//! canonicalization path.

use solver_core::abstraction::preflop_class::{
    class_combos, PreflopClass, NUM_PREFLOP_CLASSES,
};
use solver_core::abstraction::flop_isomorphism::enumerate_all_flops;
use solver_core::card::Card;
use solver_core::solver::preflop_start_game::{
    compute_preflop_cfv_per_canonical_pass, FLOPS_PER_HAND, PreflopChanceTable,
};

// ─────────────────────────────────────────────────────────────────────
// Self-contained joint canonicalization (does NOT call canonicalize_flop)
// ─────────────────────────────────────────────────────────────────────

/// All 24 permutations of {0, 1, 2, 3}. Listed explicitly so the test
/// has zero dependency on flop_isomorphism's internal SUIT_PERMUTATIONS
/// (which the test could share with the runtime if imported, breaking
/// the structural-independence requirement).
const SUIT_PERMS_24: [[u8; 4]; 24] = [
    [0,1,2,3], [0,1,3,2], [0,2,1,3], [0,2,3,1], [0,3,1,2], [0,3,2,1],
    [1,0,2,3], [1,0,3,2], [1,2,0,3], [1,2,3,0], [1,3,0,2], [1,3,2,0],
    [2,0,1,3], [2,0,3,1], [2,1,0,3], [2,1,3,0], [2,3,0,1], [2,3,1,0],
    [3,0,1,2], [3,0,2,1], [3,1,0,2], [3,1,2,0], [3,2,0,1], [3,2,1,0],
];

fn perm_card(c: Card, perm: &[u8; 4]) -> Card {
    let rank = c >> 2;
    let suit = (c & 3) as usize;
    (rank << 2) | perm[suit]
}

/// Joint canonical form of (F, h) under suit isomorphism: find the suit
/// permutation that lex-minimizes (sorted(perm(F)), normalized(perm(h)))
/// and return the permuted tuple.
///
/// The result is invariant under joint suit permutation by construction.
/// Self-contained: does NOT call `flop_isomorphism::canonicalize_flop`,
/// so the stub's symmetry computation is independent of the runtime's
/// flop canonicalization path.
fn joint_canonical(f: [Card; 3], h: (Card, Card)) -> ([Card; 3], (Card, Card)) {
    let mut best: Option<([Card; 3], (Card, Card))> = None;
    for perm in &SUIT_PERMS_24 {
        let mut p_f = [
            perm_card(f[0], perm),
            perm_card(f[1], perm),
            perm_card(f[2], perm),
        ];
        p_f.sort();
        let p_h0 = perm_card(h.0, perm);
        let p_h1 = perm_card(h.1, perm);
        let p_h = if p_h0 < p_h1 { (p_h0, p_h1) } else { (p_h1, p_h0) };
        let candidate = (p_f, p_h);
        match best {
            None => best = Some(candidate),
            Some(b) if candidate < b => best = Some(candidate),
            _ => {}
        }
    }
    best.expect("at least one suit perm always exists")
}

/// Test stub V_flop(F, h): deterministic function of the joint canonical
/// (F, h). Encodes the joint canonical as a u32 hash mapped to f32 in a
/// stable range. Has joint suit symmetry by construction (hash depends
/// only on joint canonical). Does NOT have spurious per-combo board
/// symmetry (different h's give different joint canonicals for the same
/// F's orbit).
fn v_flop_stub(f: [Card; 3], h: (Card, Card)) -> f32 {
    let (cf, ch) = joint_canonical(f, h);
    // Encode joint canonical as u32: 5 cards × 6 bits = 30 bits.
    let key = (cf[0] as u32)
            | ((cf[1] as u32) << 6)
            | ((cf[2] as u32) << 12)
            | ((ch.0 as u32) << 18)
            | ((ch.1 as u32) << 24);
    // Map to f32 in a reasonable range. Division by 2^20 ≈ 1e6 keeps
    // values < ~1024 with reasonable precision.
    (key as f32) / (1u32 << 20) as f32
}

// ─────────────────────────────────────────────────────────────────────
// REFERENCE: explicit 22,100-actual-flop enumeration, NO calls to
// reduce_cfv_combo_to_class or aggregate_preflop_chance or
// chance_probability_flop. Structurally independent of runtime path.
// ─────────────────────────────────────────────────────────────────────

/// Compute per-class preflop CFV by explicit enumeration of all 22,100
/// actual flops with unit weight per flop, and explicit per-class
/// per-actual-flop averaging via inline summation over surviving combos.
///
/// Math reference:
///
/// For class c with n_c combos:
///   V_class[c] = (1 / (n_c × 19,600)) ×
///                Σ over h in class c, over f in 22,100 with f compat h
///                of V_flop_stub(f, h)
///
/// This is the textbook un-canonicalized CFV: probability of each
/// (combo, flop) pair is uniform 1/(1326 × 19,600), conditional on
/// holding class c gives 1/(n_c × 19,600) per compatible (h, f) pair.
///
/// **STRUCTURAL VERIFICATION (read this loop):**
///   - Outer loop: `for f in enumerate_all_flops()` — 22,100 actual flops
///   - Inner loop over `class_combos(class)` — explicit per-class combos
///   - Filter via `!f.contains(c1) && !f.contains(c2)` — explicit
///     compatibility check (the OR-union conflict logic, but applied
///     here directly, not via expansion())
///   - Sum and average inline, no `reduce_cfv_combo_to_class` call
///   - Unit weight `1.0 / 19,600` per flop, no `chance_probability_flop`
///     call, no `aggregate_preflop_chance` call
///
/// If the loop is reading "for canonical_flop in 1,755 canonicals" or
/// the weight is "orbit_size × something", the reference is running
/// the runtime's logic and the validation is self-validating.
fn compute_preflop_cfv_reference(
    v_flop_fn: impl Fn([Card; 3], (Card, Card)) -> f32,
) -> Vec<f32> {
    let mut v_class = vec![0.0f64; NUM_PREFLOP_CLASSES];
    // STRUCTURAL CHECK: this loop is over enumerate_all_flops (22,100
    // actual flops). NOT over enumerate_canonical_flops (1,755).
    for f in enumerate_all_flops() {
        for class_idx in 0..NUM_PREFLOP_CLASSES {
            let class = PreflopClass(class_idx as u8);
            // Explicit per-class combo enumeration. NOT calling expansion()
            // (which is the same primitive reduce_cfv_combo_to_class uses).
            for &(c1, c2) in &class_combos(class) {
                if f.contains(&c1) || f.contains(&c2) {
                    continue; // explicit OR-union conflict check
                }
                // Unit weight 1/19,600 per flop, divided by n_c for the
                // per-class average. NOT calling chance_probability_flop
                // (which uses orbit_size). NOT calling aggregate (which
                // sums over canonicals). Direct math.
                v_class[class_idx] += (v_flop_fn(f, (c1, c2)) as f64)
                    / (class.num_combos() as f64 * FLOPS_PER_HAND as f64);
            }
        }
    }
    v_class.into_iter().map(|x| x as f32).collect()
}

// ─────────────────────────────────────────────────────────────────────
// THE TEST
// ─────────────────────────────────────────────────────────────────────

/// Discriminator: at constant stub K, the math derivation says
/// V_class[c] = K for every class. If both sides give K within f32
/// precision, the structural arithmetic is right. If they don't, there's
/// a real bug.
#[test]
fn constant_stub_gives_constant_class_cfv_on_both_sides() {
    let table = PreflopChanceTable::new(
        2, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );
    let k = 42.0_f32;
    let const_stub = |_: [Card; 3], _: (Card, Card)| k;

    let runtime = compute_preflop_cfv_per_canonical_pass(&table, &const_stub);
    let reference = compute_preflop_cfv_reference(&const_stub);

    let mut max_rt_diff: f32 = 0.0;
    let mut max_ref_diff: f32 = 0.0;
    let mut max_pair_diff: f32 = 0.0;
    for c in 0..NUM_PREFLOP_CLASSES {
        max_rt_diff = max_rt_diff.max((runtime[c] - k).abs());
        max_ref_diff = max_ref_diff.max((reference[c] - k).abs());
        max_pair_diff = max_pair_diff.max((runtime[c] - reference[c]).abs());
    }
    eprintln!("constant stub K = {}: max diff from K — runtime {}, reference {}; runtime vs reference {}",
        k, max_rt_diff, max_ref_diff, max_pair_diff);

    // At constant K, V_class = K exactly (by derivation). The arithmetic
    // path should reproduce K within f32 floor (a few ULPs on the
    // accumulator). If max_rt_diff >> 1e-3 at K=42 (relative 2e-5), the
    // runtime arithmetic has a real bug.
    assert!(max_rt_diff < 1e-3,
        "runtime path deviates from constant K = {} by {} (relative {}). \
         The constant-stub math derivation says V_class[c] = K exactly; \
         deviation > f32 floor indicates a real bug in the runtime path \
         (aggregate or reduce or chance_probability_flop normalization).",
        k, max_rt_diff, max_rt_diff / k);
    assert!(max_ref_diff < 1e-3,
        "reference path deviates from constant K = {} by {} (relative {}). \
         The reference enumerates 22,100 actual flops with unit weight; \
         the per-class normalization should collapse to K exactly.",
        k, max_ref_diff, max_ref_diff / k);
}

#[test]
fn p2_5a_preflop_chance_arithmetic_anchored_against_uncanonicalized_truth() {
    // Build the runtime-side preflop chance table.
    let table = PreflopChanceTable::new(
        2,
        vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );

    eprintln!("\n=== P2.5a: stub-based anchor on preflop chance arithmetic ===");
    eprintln!("Runtime: 1,755 canonicals × reduce_cfv × aggregate_preflop_chance");
    eprintln!("Reference: 22,100 actual flops × explicit-combo-enumeration × unit weight");
    eprintln!("Stub: self-contained joint canonicalization (24 perms inline)");

    let t0 = std::time::Instant::now();
    let runtime = compute_preflop_cfv_per_canonical_pass(&table, v_flop_stub);
    let t_runtime = t0.elapsed();
    eprintln!("Runtime path: {} canonical flops × ~1176 combos in {:?}",
        table.num_canonical_flops(), t_runtime);

    let t0 = std::time::Instant::now();
    let reference = compute_preflop_cfv_reference(v_flop_stub);
    let t_ref = t0.elapsed();
    eprintln!("Reference path: 22,100 actual flops × 1326 combos in {:?}", t_ref);

    // Compare per-class.
    //
    // PRECISION DISCUSSION:
    // The runtime uses f32 throughout (matching production). The
    // reference uses f64 accumulators internally (cast to f32 at the
    // end). Both compute the same true value, but the runtime
    // accumulates ~2M f32 additions on values of magnitude ~10³,
    // which gives cumulative error ~sqrt(2M) × ε_f32 × magnitude ≈
    // sqrt(2e6) × 1.2e-7 × 10³ ≈ 1.7e-1 worst case, ~1e-2 typical.
    // The reference's f64 accumulator has ~3e-9 typical error,
    // negligible. So the runtime-vs-reference diff IS the runtime's
    // f32 floor.
    //
    // The constant-stub discriminator above (passes at K=42 within
    // 1e-3) proves the arithmetic is structurally right; the diff
    // here measures f32 precision, not bug.
    //
    // Tolerance: 1e-4 RELATIVE to per-class magnitude is f32 floor
    // for this accumulation pattern. Larger tolerance would mask
    // real bugs; smaller would fail on f32 noise.
    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut argmax_class: usize = 0;
    for c in 0..NUM_PREFLOP_CLASSES {
        let abs = (runtime[c] - reference[c]).abs();
        let rel = if reference[c].abs() > 1.0 { abs / reference[c].abs() } else { abs };
        if rel > max_rel_diff {
            max_rel_diff = rel;
            argmax_class = c;
            max_abs_diff = abs;
        }
    }

    eprintln!("\nmax_rel_diff = {} (class {}, abs {})", max_rel_diff, argmax_class, max_abs_diff);
    eprintln!("  runtime[{}]   = {}", argmax_class, runtime[argmax_class]);
    eprintln!("  reference[{}] = {}", argmax_class, reference[argmax_class]);

    let f32_floor_relative = 1e-4_f32;
    assert!(
        max_rel_diff < f32_floor_relative,
        "P2.5a FAILED: max_rel_diff = {} (class {}, abs diff {}), tolerance = {} (f32 floor). \
         Runtime path (1,755 canonicals × reduce_cfv × aggregate) does \
         NOT match reference (22,100 actual flops × explicit enumeration × \
         unit weight) within f32 precision. \
         If max_rel_diff is just above f32 floor: likely accumulation order \
         differences worth investigating but maybe not a real bug. \
         If max_rel_diff is much above f32 floor: the orbit-weighted aggregation, \
         or the per-class survivor-count reduce, or the orbit-weight chance \
         probability, is computing something different from the un-canonicalized \
         truth — real bug, anchored by independent reference.",
        max_rel_diff, argmax_class, max_abs_diff, f32_floor_relative,
    );

    eprintln!("✓ P2.5a PASS: runtime path agrees with un-canonicalized reference at f32 floor");
    eprintln!("  (constant-stub discriminator confirmed math is structurally exact;");
    eprintln!("   this diff is the runtime's f32 accumulator precision floor)");
}

// ─────────────────────────────────────────────────────────────────────
// Sanity checks on the stub (separate tests, run fast)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn stub_has_joint_suit_symmetry() {
    // For any (F, h) and any suit perm p, v_flop_stub(p(F), p(h)) = v_flop_stub(F, h).
    let test_cases: Vec<([Card; 3], (Card, Card))> = vec![
        ([0, 5, 10], (20, 25)),
        ([12, 17, 22], (30, 35)),
        ([0, 13, 26], (39, 51)),
        ([4, 9, 14], (19, 24)),
    ];
    for (f, h) in test_cases {
        let baseline = v_flop_stub(f, h);
        for perm in &SUIT_PERMS_24 {
            let p_f = [perm_card(f[0], perm), perm_card(f[1], perm), perm_card(f[2], perm)];
            let p_h = (perm_card(h.0, perm), perm_card(h.1, perm));
            // Skip if perm produced card conflict (shouldn't happen but be safe)
            let v = v_flop_stub(p_f, p_h);
            assert!(
                (v - baseline).abs() < 1e-7,
                "joint symmetry broken: F={:?} h={:?} → {}, perm {:?} → F={:?} h={:?} → {}",
                f, h, baseline, perm, p_f, p_h, v,
            );
        }
    }
}

#[test]
fn stub_lacks_board_only_symmetry() {
    // For a fixed h that does NOT use all 4 suits, there must exist a
    // suit perm p such that p moves a suit not in h, producing a
    // different (F, h) joint canonical and therefore a different stub
    // value. If stub were board-only-symmetric (depending only on
    // canonical(F), ignoring h's suit interaction), this would fail.
    //
    // Test: h = (As, Kh) using suits {s, h}. Permute suit c → d (which
    // doesn't move anything in h) — joint canonical SHOULD change for
    // flops that have a club, because the perm moves clubs.
    let h: (Card, Card) = (
        (12 << 2) | 3, // As
        (11 << 2) | 2, // Kh
    );
    // Find a flop where permuting c↔d changes the joint canonical.
    let f_with_club: [Card; 3] = [
        (10 << 2) | 0, // Qc
        (9 << 2) | 1,  // Jd
        (8 << 2) | 2,  // Th
    ];
    let baseline = v_flop_stub(f_with_club, h);

    // Perm swapping c↔d: [d, c, h, s] = [1, 0, 2, 3]
    let swap_cd = [1u8, 0, 2, 3];
    let p_f = [
        perm_card(f_with_club[0], &swap_cd),
        perm_card(f_with_club[1], &swap_cd),
        perm_card(f_with_club[2], &swap_cd),
    ];
    // h is unchanged (no clubs or diamonds in h)
    let p_h = (perm_card(h.0, &swap_cd), perm_card(h.1, &swap_cd));
    assert_eq!(p_h, h, "perm c↔d should not affect h=(As,Kh)");

    let board_only_perm_value = v_flop_stub(p_f, p_h);

    // Both should produce the same value if board-only symmetric.
    // For a NON-orbit-invariant stub, they'd be EQUAL (the joint
    // canonical is the same since h is unchanged). Hmm — that's the
    // OPPOSITE of what I want to test.
    //
    // Actually: joint_canonical(perm(F), h) where h has no c or d:
    // applying c↔d to F changes F, but the joint canonical of
    // (perm(F), h) is computed via 24 perms; one of those perms is
    // the c↔d that undoes the change → joint canonical is the same.
    // So the stub value is the SAME — by joint symmetry.
    //
    // To test "lacks board-only symmetry", I need a perm that changes
    // BOTH F and h such that the joint canonical changes vs baseline.
    // But by joint symmetry, perm(F), perm(h) gives same joint canonical
    // as F, h. So the stub is invariant under joint perm.
    //
    // The "no spurious board-only symmetry" property means: if I
    // change F's suits WITHOUT changing h's suits, the stub value
    // SHOULD differ in general. Above test SHOULD show difference,
    // but for h that doesn't use c or d, applying c↔d to F leaves h
    // unchanged AND the joint canonical of (F', h) under any perm
    // including the inverse c↔d gives back the original canonical.
    // So the stub IS invariant.
    //
    // OK so the stub has joint symmetry which makes it invariant
    // under (F → perm(F), h fixed) when h's suits are not in perm's
    // moved set. That's correct behavior. The "lacks board-only
    // symmetry" property means: when h's suits ARE in perm's moved
    // set, perm(F) with h-fixed should give a different value.
    //
    // Let me test that: pick h using all 4 suits? Impossible (h has
    // 2 cards). Pick h using 2 distinct suits, perm swaps one of h's
    // suits with one not in h.
    //
    // h = (As, Kh): suits {s, h}. Perm s↔c (swap suit 3 and suit 0):
    // [c, h, d, s] becomes... wait the perm[i] gives the new suit for
    // old suit i. So [3, 1, 2, 0] swaps 0↔3 i.e. c↔s. Apply to h:
    // As (suit 3) → suit perm[3] = 0 = c. Ac. Kh (suit 2) → suit
    // perm[2] = 2 = h. Kh. So permed h = (Ac, Kh). DIFFERENT from
    // original (As, Kh). So this perm DOES change h.
    //
    // The joint canonical of (perm(F), perm(h)) equals joint canonical
    // of (F, h) (by joint symmetry). So stub value is the same.
    //
    // What about (perm(F), h) — keeping h fixed but applying perm to
    // F? The joint canonical of (perm(F), h) may differ from joint
    // canonical of (F, h) because we can't always find a perm to undo
    // perm(F) without also changing h.
    //
    // For h = (As, Kh), perm swap c↔s of F:
    // F = [Qc, Jd, Th] becomes perm(F) = [Qs, Jd, Th]. Joint
    // canonical of ([Qs, Jd, Th], (As, Kh)) vs ([Qc, Jd, Th], (As, Kh)).
    // To make these equal under joint symmetry, we'd need a perm p
    // such that p(perm(F)) sorted = p(F) sorted AND p((As, Kh)) =
    // (As, Kh). The h-fixing perm = (s fixed, h fixed) = perm of
    // {c, d} only. With p = swap c↔d: p(perm(F)) = swap-cd([Qs, Jd,
    // Th]) = [Qs, Jc, Th]. Not equal to p(F) = swap-cd([Qc, Jd, Th])
    // = [Qd, Jc, Th]. So the h-fixing subgroup can't undo the c↔s
    // swap. The joint canonicals are different. Stub values differ.
    //
    // So this version DOES discriminate. Let me code it.

    let swap_cs = [3u8, 1, 2, 0]; // c↔s, d and h fixed
    let p_f = [
        perm_card(f_with_club[0], &swap_cs),
        perm_card(f_with_club[1], &swap_cs),
        perm_card(f_with_club[2], &swap_cs),
    ];
    // h's suits are {s, h}; swap_cs moves s→c and c→s. So h's s-suit
    // card (As) maps to Ac. h changes.
    // For "board-only symmetry" test, we want to KEEP h fixed and
    // see if perm(F) gives different stub value.
    let v_fixed_h = v_flop_stub(p_f, h);

    // baseline = v_flop_stub(f_with_club, h)
    // v_fixed_h = v_flop_stub(perm(f_with_club), h) — DIFFERENT h would imply
    // joint canonical changes. Stub should differ.
    assert_ne!(
        v_fixed_h, baseline,
        "stub has spurious board-only symmetry: v(perm(F), h) = v(F, h) = {} \
         when h was held fixed across a suit perm that does NOT fix h's suits. \
         A board-only-symmetric stub would mask bugs that surface against \
         realistically asymmetric V_flop values (the nh=1326 finding).",
        baseline
    );

    // Also re-use the v_flop_stub variable to avoid unused-var warning:
    let _ = board_only_perm_value;
}
