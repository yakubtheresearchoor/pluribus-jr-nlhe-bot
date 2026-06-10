// P1.5.4 Slice A.3c: class×class blocking matrix + per-class fold-terminal
// CFV helper for preflop.
//
// At a preflop fold-terminal (lone non-folder wins the pot, no flop dealt),
// the per-class CFV for the traverser depends on the OPP's per-class reach
// AND on the fraction of opp combos that don't share a card with traverser's
// combo (card blocking). The lossless 169-class layout averages this blocking
// across combos within each class pair, producing a 169×169 matrix M where:
//
//   M[c, c'] = (# of (combo_c, combo_c') pairs that DON'T share any card) /
//              (|combos_c| × |combos_c'|)
//
// This matrix is constant (depends only on the card structure, not strategies
// or reaches), so we compute it once and reuse.
//
// Per-class fold-terminal CFV for traverser:
//
//   v[c] = chip_delta × Σ_c' opp_reach[c'] × M[c, c']
//
// where chip_delta is traverser's chip gain (positive for the non-folder,
// negative for the folder). The chip_delta computation from tree.contributions
// + tree.folded_masks involves preflop-specific blind-asymmetry reconciliation
// (the showdown_oracle's `starting_pot / np + c_t` formula assumes symmetric
// blind-or-zero contributions, which doesn't hold for HU preflop SB=1/BB=2);
// that reconciliation is a separate slice (deferred). This module takes
// chip_delta as a parameter; the caller computes it from the tree's
// per-terminal state.
//
// Validation:
//
//   1. Blocking matrix vs direct combo enumeration on known class pairs.
//      - Pair vs itself (e.g., AA vs AA): blocked entries are pairs that
//        share an A; direct count.
//      - Pair vs unrelated pair (e.g., AA vs 22): no card overlap possible
//        because ranks differ → all (6 × 6 = 36) pairs are non-blocking →
//        M[AA, 22] = 1.0.
//      - Suited vs same suited (e.g., AKs vs AKs): exact ranks AND suits
//        overlap; direct count.
//   2. Per-class fold-terminal CFV applied identity: with uniform opp_reach
//      (all-ones) and chip_delta=1, v[c] = Σ_c' M[c, c'] (the "average
//      surviving opponent classes" per traverser class). Verifies the
//      composition arithmetic.

use crate::abstraction::preflop_class::{class_combos, NUM_PREFLOP_CLASSES, PreflopClass};
use crate::card::Card;
use crate::tree::flat::FlatTree;

// ─────────────────────────────────────────────────────────────────────
// Multiway blocking primitive (P1.5.4 6-max rework)
// ─────────────────────────────────────────────────────────────────────
//
// At 2-max, blocking decomposes pairwise: M[c, c'] = P(combo_c and
// combo_c' don't share a card). The 169×169 matrix captures this fully
// because there are no higher-order terms with only 2 players.
//
// At 3+ players, blocking does NOT decompose pairwise. The events
// "hand_i conflicts with hand_j" are CORRELATED through shared cards
// across multiple pairs. A pairwise-product approximation systematically
// over-counts non-blocking tuples because it misses cross-pair card
// conflicts.
//
// Concrete failure of the pairwise approximation (the test below makes
// this discriminating):
//
//   3 players all holding class AA. Each pair has pairwise non-blocking
//   fraction 6/36 (from build_class_blocking_matrix). The pairwise
//   product gives (6/36)^3 ≈ 0.0046, i.e., "0.5% of triple tuples are
//   non-blocking". The JOINT answer: ZERO. Three AA combos need 6
//   aces; only 4 exist. The pairwise approach is structurally wrong.
//
// So the multiway primitive computes the JOINT non-blocking count
// directly via brute-force enumeration of (combo_1, combo_2, ...,
// combo_N) tuples with a used-cards bitmask, returning the exact joint
// non-blocking fraction. No formula, no inclusion-exclusion shortcut.
// Validated by the (a) it MATCHES enumeration at N=2 (HU sanity), and
// (b) the lossy pairwise approximation is SHOWN to disagree at N=3 on
// the AA-AA-AA case (proving the pairwise trap is real).

/// Joint non-blocking fraction across N hand classes.
///
/// Given N preflop classes `[c_1, c_2, ..., c_N]`, returns the
/// fraction of ordered (combo_1, combo_2, ..., combo_N) tuples (where
/// combo_i ∈ class_combos(c_i)) that have no card conflicts across all
/// N combos.
///
/// **Computation:** direct brute-force enumeration. Recursive DFS with
/// a 52-bit used-cards bitmask; early-terminates as soon as a conflict
/// is detected at any depth. For N=6 with class sizes up to 12, the
/// worst-case enumeration is 12^6 ≈ 3M tuples per call; in practice
/// far fewer survive the early-termination prune.
///
/// **Correctness:** by construction — this IS the joint enumeration.
/// There is no formula to validate; the validation is showing that the
/// pairwise approximation DIVERGES from this on N=3 cases where joint
/// constraints come into play.
///
/// **Symmetric** in the class arguments (joint conflict is invariant
/// under permutation of the hands). Callers may sort `classes` to
/// canonicalize for memoization keys.
pub fn joint_class_tuple_non_blocking_fraction(classes: &[PreflopClass]) -> f32 {
    let count = joint_class_tuple_non_blocking_count(classes);
    let total: u64 = classes.iter().map(|c| c.num_combos() as u64).product();
    if total == 0 { return 0.0; }
    count as f32 / total as f32
}

/// Joint non-blocking COUNT (raw integer, not normalized). The
/// fraction-form's numerator; exposed because integer math is exact
/// (avoids any f32 rounding when callers want to chain joint
/// counts before dividing).
pub fn joint_class_tuple_non_blocking_count(classes: &[PreflopClass]) -> u64 {
    let n = classes.len();
    if n == 0 { return 1; }
    // Pre-fetch combo lists once (cheap; class_combos returns Vec).
    let combo_lists: Vec<Vec<(Card, Card)>> = classes.iter()
        .map(|&c| class_combos(c))
        .collect();
    let mut count: u64 = 0;
    enumerate_joint(&combo_lists, 0_u64, 0, &mut count);
    count
}

/// Recursive joint-tuple enumerator. `used_cards` is a 52-bit bitmask
/// of cards already committed by hands at depths 0..depth. At each
/// depth, iterate this depth's combo list; for each combo whose cards
/// don't overlap `used_cards`, recurse with the updated mask. When
/// depth reaches N, count this tuple as non-blocking.
fn enumerate_joint(
    combo_lists: &[Vec<(Card, Card)>],
    used_cards: u64,
    depth: usize,
    count: &mut u64,
) {
    if depth == combo_lists.len() {
        *count += 1;
        return;
    }
    for &(c1, c2) in &combo_lists[depth] {
        let mask = (1_u64 << c1) | (1_u64 << c2);
        if used_cards & mask == 0 {
            enumerate_joint(combo_lists, used_cards | mask, depth + 1, count);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// N-player fold-terminal CFV (P1.5.4 6-max rework, Slice B.2)
// ─────────────────────────────────────────────────────────────────────
//
// Composes the multiway blocking primitive (B.1) with the chip_delta
// machinery (A.3d) to give per-class CFV at an N-player preflop fold-
// terminal. Formula:
//
//   v[c_t] = chip_delta ×
//     Σ over (c_opp_1, c_opp_2, ..., c_opp_{N-1}) of [
//       Π_i opp_reach[i][c_opp_i]
//       × joint_class_tuple_non_blocking_fraction(
//           [c_t, c_opp_1, c_opp_2, ..., c_opp_{N-1}])
//     ]
//
// At N=2 (HU), this collapses to the existing HU formula:
//   v[c_t] = chip_delta × Σ_{c_opp} opp_reach[c_opp] × M_HU[c_t, c_opp]
//
// At N=6 with all-opps-fold-terminal, this is a sum over 169^5 ≈ 138
// billion class tuples per traverser class — intractable in dense form.
// IN PRACTICE: at a preflop fold terminal, opps have folded specific
// hand subsets, so their reaches are concentrated in a small set of
// classes (the ones they fold). Sparse-reach early termination
// (`accumulated_reach == 0` prune) handles this; the cost scales with
// the SPARSITY of opp reaches, not 169^{N-1}.
//
// For dense opp reaches (e.g., all-1 initial range at iter 0), the cost
// is large by construction; this isn't an optimization slice. Callers
// at 6-max either (a) supply sparse reaches from the engine's actual
// preflop walk (most opp classes are zero at any fold terminal because
// only specific classes were "still in" before the final fold), or (b)
// use a future optimization (caching, decomposition) once the
// correctness primitive exists.
//
// Correctness FIRST: this function implements the math directly,
// validated against the HU formula at N=2 and against hand-computed
// expected values at N=3 with sparse opp reaches.

/// Compute per-class CFV at an N-player preflop fold-terminal.
///
/// `opp_reaches`: `[N-1]` opponent reach distributions, each of length
///   `NUM_PREFLOP_CLASSES`. Order doesn't matter (the joint blocking
///   primitive is symmetric in opp classes); the engine supplies them
///   in some canonical order.
///
/// `chip_delta`: traverser's chip gain at this terminal (from
///   `preflop_fold_terminal_chip_delta` / `_from_state`, A.3d). Sign
///   convention is the oracle's: positive for non-folder, negative for
///   folder.
///
/// Returns `Vec<f32>` of length `NUM_PREFLOP_CLASSES` giving v[c_t]
/// per traverser class.
pub fn preflop_fold_terminal_cfv_multiway(
    opp_reaches: &[&[f32]],
    chip_delta: f32,
) -> Vec<f32> {
    let n_opps = opp_reaches.len();
    for (i, r) in opp_reaches.iter().enumerate() {
        assert_eq!(r.len(), NUM_PREFLOP_CLASSES,
            "opp_reaches[{}] length {} != NUM_PREFLOP_CLASSES",
            i, r.len());
    }
    if chip_delta == 0.0 || n_opps == 0 {
        // Degenerate cases: chip_delta=0 → zeros. n_opps=0 → no opps,
        // joint non-blocking fraction is trivially 1 across all classes
        // (1 hand, no conflict possible), v[c_t] = chip_delta × 1 ×
        // (empty product = 1) = chip_delta. Caller-driven semantics.
        return vec![chip_delta; NUM_PREFLOP_CLASSES];
    }

    let mut v = vec![0.0_f32; NUM_PREFLOP_CLASSES];
    let mut class_tuple_buf: Vec<PreflopClass> = Vec::with_capacity(1 + n_opps);
    for c_t in 0..NUM_PREFLOP_CLASSES {
        class_tuple_buf.clear();
        class_tuple_buf.push(PreflopClass(c_t as u8));
        let mut sum = 0.0_f32;
        accumulate_opp_classes(
            opp_reaches, 0, &mut class_tuple_buf, 1.0_f32, &mut sum,
        );
        v[c_t] = chip_delta * sum;
    }
    v
}

/// PAIRWISE-APPROXIMATE variant of `preflop_fold_terminal_cfv_multiway`
/// for dense-reach callers (P8 preflop bootstrap).
///
///   v[c_t] = chip_delta × Π_i ( Σ_c opp_reach[i][c] × M[c_t, c] )
///
/// Each opponent's card-blocking against the traverser is treated
/// independently; opp-opp cross-blocking is IGNORED. Per the module
/// docs above, this "systematically over-counts non-blocking tuples" —
/// the bias is a smooth multiplicative over-estimate of surviving
/// combinations, exact at N=2 (one opp, no cross-pairs to miss).
///
/// WHY IT EXISTS (measured 2026-06-09/10): the exact joint enumeration
/// is O(Π_i nnz(opp_i)) per traverser class. At 6-max iteration 1, all
/// reaches are DENSE (uniform strategy ⇒ every class reachable), so a
/// single fold terminal costs ~169^5 ≈ 1.4×10^11 tuples × 169 classes.
/// A profile (`sample`) of the P8b bootstrap showed 100% of an 11-hour
/// run inside `accumulate_opp_classes` — the engine never finished ONE
/// iteration on the 82k-terminal rich preflop tree. The pairwise form
/// is 5 matvecs of the fixed 169×169 matrix per terminal (~10^5 FLOPs,
/// ~10^8× cheaper) and keeps the same unnormalized reach-weighted
/// semantics CFR expects.
///
/// Use for bootstrap-grade size observation; the exact primitive stays
/// the ground-truth reference for sparse-reach validation cases.
///
/// `blocking_matrix`: from `build_class_blocking_matrix()`, computed
/// once by the caller.
pub fn preflop_fold_terminal_cfv_multiway_pairwise(
    opp_reaches: &[&[f32]],
    chip_delta: f32,
    blocking_matrix: &[f32],
) -> Vec<f32> {
    let n_opps = opp_reaches.len();
    assert_eq!(blocking_matrix.len(), NUM_PREFLOP_CLASSES * NUM_PREFLOP_CLASSES);
    for (i, r) in opp_reaches.iter().enumerate() {
        assert_eq!(r.len(), NUM_PREFLOP_CLASSES,
            "opp_reaches[{}] length {} != NUM_PREFLOP_CLASSES", i, r.len());
    }
    if chip_delta == 0.0 || n_opps == 0 {
        return vec![chip_delta; NUM_PREFLOP_CLASSES];
    }
    let mut v = vec![chip_delta; NUM_PREFLOP_CLASSES];
    for r in opp_reaches {
        for c_t in 0..NUM_PREFLOP_CLASSES {
            let row = &blocking_matrix[c_t * NUM_PREFLOP_CLASSES..(c_t + 1) * NUM_PREFLOP_CLASSES];
            let mut q = 0.0_f32;
            for c in 0..NUM_PREFLOP_CLASSES {
                q += r[c] * row[c];
            }
            v[c_t] *= q;
        }
    }
    v
}

/// Recursive opp-class-tuple enumeration with sparse-reach early
/// termination. At each depth, iterate this opp's classes with non-zero
/// reach; recurse, accumulating the reach product; at the leaf
/// (all opps assigned), compute the joint non-blocking fraction across
/// the full [traverser, opp_1, ..., opp_{N-1}] class tuple and add the
/// weighted contribution.
fn accumulate_opp_classes(
    opp_reaches: &[&[f32]],
    opp_idx: usize,
    class_tuple: &mut Vec<PreflopClass>,
    accumulated_reach: f32,
    sum: &mut f32,
) {
    if accumulated_reach == 0.0 { return; }
    if opp_idx == opp_reaches.len() {
        let frac = joint_class_tuple_non_blocking_fraction(class_tuple);
        *sum += accumulated_reach * frac;
        return;
    }
    for c_opp in 0..NUM_PREFLOP_CLASSES {
        let r = opp_reaches[opp_idx][c_opp];
        if r == 0.0 { continue; }
        class_tuple.push(PreflopClass(c_opp as u8));
        accumulate_opp_classes(
            opp_reaches, opp_idx + 1, class_tuple,
            accumulated_reach * r, sum,
        );
        class_tuple.pop();
    }
}

// ─────────────────────────────────────────────────────────────────────
// (Existing HU blocking matrix + per-class fold-terminal CFV below)
// ─────────────────────────────────────────────────────────────────────


/// Class×class non-blocking-fraction matrix, flattened as
/// `m[c * NUM_PREFLOP_CLASSES + c']`. Each entry is in [0, 1].
///
/// Self-blocking (`c == c'` for pairs): the diagonal entry is the
/// fraction of (combo, combo') pairs in the same class that don't share
/// any card. For pairs (6 combos): a combo blocks itself trivially
/// (1 pair) and other combos that share a card (4 of the remaining 5).
/// So 1 of 6 = "non-self" combos are non-blocking when combo equals
/// itself. Formal computation is what `build_class_blocking_matrix`
/// returns; this comment is illustrative.
pub fn build_class_blocking_matrix() -> Vec<f32> {
    let mut m = vec![0.0_f32; NUM_PREFLOP_CLASSES * NUM_PREFLOP_CLASSES];
    let all_combos: Vec<Vec<(u8, u8)>> = (0..NUM_PREFLOP_CLASSES)
        .map(|c| class_combos(PreflopClass(c as u8)))
        .collect();
    for c in 0..NUM_PREFLOP_CLASSES {
        let combos_c = &all_combos[c];
        for cp in 0..NUM_PREFLOP_CLASSES {
            let combos_cp = &all_combos[cp];
            let total = combos_c.len() * combos_cp.len();
            if total == 0 {
                m[c * NUM_PREFLOP_CLASSES + cp] = 0.0;
                continue;
            }
            let mut non_blocking = 0_usize;
            for &(a1, a2) in combos_c {
                for &(b1, b2) in combos_cp {
                    if a1 != b1 && a1 != b2 && a2 != b1 && a2 != b2 {
                        non_blocking += 1;
                    }
                }
            }
            m[c * NUM_PREFLOP_CLASSES + cp] = non_blocking as f32 / total as f32;
        }
    }
    m
}

/// Compute the chip delta for one traverser at a preflop fold-terminal
/// from the tree's per-node contribution/fold state.
///
/// Per the lead's directive (2026-06-04): `tree.contributions` is the
/// authoritative source of who put in what chips. The formula matches
/// the showdown_oracle's constant-payoff fast path (so the symmetric
/// overlap cross-check holds by construction) but is implemented as a
/// preflop-specific function — the validated oracle is NOT extended;
/// new code is isolated.
///
/// Formula (mirrors `showdown.rs` fast path, lines 484–500-ish):
///
/// ```text
///   investment = starting_pot / np + contributions[traverser]
///   total_pot  = starting_pot + Σ contributions
///   if traverser folded:        delta = -investment
///   else (traverser non-folder): delta = total_pot - investment
/// ```
///
/// Sign convention is the oracle's convention — zero-sum across
/// traversers at a fold terminal. The per-player delta is NOT
/// the "literal chip movement" (see showdown.rs:467-472 for the
/// detailed acknowledgement) but is what CFR's regret update
/// expects, and Σ_p delta_p = 0 holds.
///
/// HU-only for now: multi-way fold terminals (where all but one player
/// folded; that lone player wins) can extend this by replacing the
/// folder/non-folder branch with "lone active wins all".
pub fn preflop_fold_terminal_chip_delta_from_state(
    contributions: &[i32],
    fold_mask: u16,
    starting_pot: i32,
    traverser: u8,
    num_players: u8,
) -> f32 {
    let np = num_players as usize;
    assert_eq!(contributions.len(), np,
        "contributions has {} slots; expected num_players = {}", contributions.len(), np);
    let t = traverser as usize;
    assert!(t < np, "traverser {} out of range for num_players {}", t, np);
    let traverser_folded = (fold_mask & (1u16 << t)) != 0;
    let total_pot: i32 = starting_pot + contributions.iter().sum::<i32>();
    let investment = starting_pot as f32 / np as f32 + contributions[t] as f32;
    if traverser_folded {
        -investment
    } else {
        total_pot as f32 - investment
    }
}

/// Convenience: read terminal state directly from the tree, then call
/// `preflop_fold_terminal_chip_delta_from_state`. Production callers
/// (e.g., the `terminal_value_fn` wired through `PreflopVectorCfr::run_one_iteration`)
/// use this form.
pub fn preflop_fold_terminal_chip_delta(
    tree: &FlatTree,
    terminal_node_idx: usize,
    traverser: u8,
) -> f32 {
    let np = tree.num_players;
    let contributions: Vec<i32> = (0..np)
        .map(|p| tree.get_contribution(terminal_node_idx, p))
        .collect();
    let fold_mask = tree.get_folded_mask(terminal_node_idx);
    preflop_fold_terminal_chip_delta_from_state(
        &contributions, fold_mask, tree.starting_pot, traverser, np,
    )
}

/// Compute the per-class CFV for one traverser at a preflop fold-terminal.
///
///   v[c] = chip_delta × Σ_c' opp_reach[c'] × M[c, c']
///
/// `opp_reach`: traverser's opponent's per-class reach at the terminal
/// (length `NUM_PREFLOP_CLASSES`). For factored CFR with vector reach,
/// this includes opp's strategy factors propagated down the tree.
///
/// `blocking_matrix`: from `build_class_blocking_matrix`. Pre-compute
/// once per process and reuse.
///
/// `chip_delta`: traverser's chip gain at this terminal (positive when
/// traverser wins, negative when traverser folds). Sign convention is
/// the caller's choice; the helper just multiplies through.
///
/// HU-only for now: multi-way preflop fold-terminals (where some opps
/// fold but more than one player remains, requiring further play) are
/// not terminal nodes by definition. Multi-way fold-terminals where ALL
/// but one player folded can be handled with this helper too by summing
/// over multiple opp reaches; the API generalization is a separate
/// slice when N-way preflop lands.
pub fn preflop_fold_terminal_cfv_hu(
    opp_reach: &[f32],
    blocking_matrix: &[f32],
    chip_delta: f32,
) -> Vec<f32> {
    assert_eq!(opp_reach.len(), NUM_PREFLOP_CLASSES,
        "opp_reach length {} != NUM_PREFLOP_CLASSES {}",
        opp_reach.len(), NUM_PREFLOP_CLASSES);
    assert_eq!(blocking_matrix.len(), NUM_PREFLOP_CLASSES * NUM_PREFLOP_CLASSES,
        "blocking_matrix length {} != {}^2", blocking_matrix.len(), NUM_PREFLOP_CLASSES);
    let mut v = vec![0.0_f32; NUM_PREFLOP_CLASSES];
    for c in 0..NUM_PREFLOP_CLASSES {
        let row_base = c * NUM_PREFLOP_CLASSES;
        let mut acc = 0.0_f32;
        for cp in 0..NUM_PREFLOP_CLASSES {
            acc += opp_reach[cp] * blocking_matrix[row_base + cp];
        }
        v[c] = chip_delta * acc;
    }
    v
}

#[cfg(test)]
mod pairwise_tests {
    use super::*;

    /// At N=2 (one opp) there are no cross-pairs to miss: pairwise must
    /// equal the exact joint enumeration bit-for-bit (both reduce to one
    /// matvec against M).
    #[test]
    fn pairwise_exact_at_one_opp() {
        let m = build_class_blocking_matrix();
        // Sparse but multi-class opp reach.
        let mut r = vec![0.0_f32; NUM_PREFLOP_CLASSES];
        r[0] = 0.7;   // AA
        r[20] = 0.3;  // some suited class
        r[100] = 1.2; // some offsuit class (unnormalized on purpose)
        let opp: Vec<&[f32]> = vec![&r];
        let exact = preflop_fold_terminal_cfv_multiway(&opp, 2.5);
        let pair = preflop_fold_terminal_cfv_multiway_pairwise(&opp, 2.5, &m);
        for c in 0..NUM_PREFLOP_CLASSES {
            assert!((exact[c] - pair[c]).abs() < 1e-5,
                "N=2 mismatch at class {}: exact={} pairwise={}", c, exact[c], pair[c]);
        }
    }

    /// At N=3 (two opps) the pairwise form ignores opp-opp blocking and
    /// must OVER-estimate |v| (over-counts non-blocking tuples). Checked
    /// on a small sparse case the exact enumeration can afford.
    #[test]
    fn pairwise_overcounts_at_two_opps() {
        let m = build_class_blocking_matrix();
        let mut r1 = vec![0.0_f32; NUM_PREFLOP_CLASSES];
        let mut r2 = vec![0.0_f32; NUM_PREFLOP_CLASSES];
        r1[0] = 1.0;  // AA
        r1[12] = 0.5; // 22
        r2[0] = 0.8;  // AA
        r2[50] = 0.4; // suited class
        let opps: Vec<&[f32]> = vec![&r1, &r2];
        let exact = preflop_fold_terminal_cfv_multiway(&opps, 1.0);
        let pair = preflop_fold_terminal_cfv_multiway_pairwise(&opps, 1.0, &m);
        let mut any_positive = false;
        for c in 0..NUM_PREFLOP_CLASSES {
            assert!(pair[c] >= exact[c] - 1e-6,
                "pairwise under exact at class {}: exact={} pairwise={}", c, exact[c], pair[c]);
            if exact[c] > 0.0 { any_positive = true; }
        }
        assert!(any_positive, "degenerate test: exact all zero");
    }
}
