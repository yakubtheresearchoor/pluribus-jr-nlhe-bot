//! Preflop-start chance integration over the lossless 169-class hand
//! layout and the lossless flop-isomorphism canonicalization.
//!
//! ## What this module owns
//!
//! `PreflopChanceTable` holds the data needed for the preflop → flop
//! chance step:
//!   - The 1,755 canonical flops + their orbit sizes (×4 / ×12 / ×24)
//!   - The 169-class preflop hand layout
//!   - Per-player initial weights at the preflop class level
//!
//! `chance_probability_flop(canonical_idx, class_idx)` returns the
//! probability that the chance outcome is the given canonical flop,
//! conditional on the preflop class being held. Math is exact (no
//! orbit-weighted approximation) because the underlying class layout
//! is suit-iso lossless — see `crate::abstraction::preflop_class`.
//!
//! ## What this module does NOT yet own (sub-stepping plan, #44)
//!
//! Per the user-directed sub-stepping for P1.5:
//!   - P1.5.1 (THIS COMMIT): PreflopChanceTable data + chance_probability_flop
//!     + conflict-OR + probability-sum check
//!   - P1.5.2: four-zone zone_nodes_per_level (extension of
//!     flop_start_vector_cfr.rs; routine wiring, rides on P2.5)
//!   - P1.5.3: compute_reach_preflop + button-first reach order check
//!   - P1.5.4: bottom_up_zone preflop extension + aggregation isolation
//!     check (the per-flop FlopChanceTable wrapping lives here)
//!   - P1.5.5: PreflopStartGame GameSpec impl (routine wiring)
//!   - P2.5:  orchestration oracle four-zone walk with un-canonicalized
//!     22,100-flop anchor (the reference is the un-canonicalized
//!     enumeration validated in P1.5-pre, NOT the canonicalized path
//!     the runtime uses — agreement must be against truth, not
//!     against the same orbit-weight logic)
//!
//! ## Why exact (no approximation)
//!
//! P1.5-pre validated empirically that for every (canonical flop C,
//! class H) pair, all orbit members of C give the same compat-combo
//! count. This is the lossless-ness property: the joint flop-hand
//! suit-iso group acts trivially on class memberships, so the
//! orbit-weighted aggregation equals the un-canonicalized sum exactly.
//! At full nh = 1326 this property does not hold — see #44 history.
//!
//! Total probability normalization check is in `tests`: for every
//! class H, Σ over canonical flops F of chance_probability_flop(F, H)
//! must equal 1.0 (within f32 rounding). Conflict-OR check confirms
//! the function returns 0 exactly when every combo in the class
//! shares some card with every orbit member of the canonical.

use crate::abstraction::flop_isomorphism::{
    enumerate_canonical_flops, orbit_of,
};
use crate::abstraction::preflop_class::{
    expansion, PreflopClass, NUM_PREFLOP_CLASSES,
};
use crate::card::Card;
use crate::tree::flat::FlatTree;
use crate::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use crate::solver::flop_start_vector_cfr::FlopStartVectorCfr;

/// C(50, 3) = number of distinct 3-card flops the dealer can deal
/// conditional on a 2-card hand being held (52 - 2 = 50 cards left).
pub const FLOPS_PER_HAND: u32 = 19_600;

/// Per-player chance integration data for the preflop → flop transition.
pub struct PreflopChanceTable {
    pub num_players: u8,

    /// 1,755 canonical flops in lex order (from
    /// `flop_isomorphism::enumerate_canonical_flops`).
    pub canonical_flops: Vec<[Card; 3]>,

    /// Orbit size per canonical flop (× 4 / × 12 / × 24). Index parallels
    /// `canonical_flops`. The orbit-weighted sum across these equals the
    /// un-canonicalized 22,100 sum exactly (P1.5-pre orbit-weighted total
    /// anchor proved this).
    pub orbit_sizes: Vec<u32>,

    /// Per-player initial weights at the preflop class level.
    /// Shape: `[num_players][169]`. Class index follows
    /// `PreflopClass::index()` (0..13 pairs, 13..91 suited, 91..169 offsuit).
    pub class_initial_weights: Vec<Vec<f32>>,

    /// Total reach-weighted joint count (for showdown normalization in
    /// the preflop terminal evaluation, downstream piece). Mirrors
    /// `FlopChanceTable::num_combinations` semantics. Computed once at
    /// construction.
    pub num_combinations: f64,
}

impl PreflopChanceTable {
    /// Construct the preflop chance integration table from per-player
    /// preflop class weights.
    ///
    /// `class_weights[p][k]` = weight of class k for player p. Use
    /// uniform 1.0 for a "deal full range" setup, or non-uniform for
    /// pre-specified preflop ranges.
    pub fn new(num_players: u8, class_weights: Vec<Vec<f32>>) -> Self {
        assert_eq!(class_weights.len(), num_players as usize,
            "class_weights must have one row per player");
        for (p, w) in class_weights.iter().enumerate() {
            assert_eq!(w.len(), NUM_PREFLOP_CLASSES,
                "class_weights[{}] has length {}, expected {}",
                p, w.len(), NUM_PREFLOP_CLASSES);
        }

        let canonical_flops = enumerate_canonical_flops();
        // Compute orbit sizes once. orbit_of is O(22,100) per call so this
        // is the slow path (~ a few seconds in release); cache once at
        // construction. Downstream consumers read orbit_sizes directly.
        let orbit_sizes: Vec<u32> = canonical_flops.iter()
            .map(|&c| orbit_of(c).len() as u32)
            .collect();

        // num_combinations placeholder: at preflop, this is the reach-
        // weighted joint count over all player class assignments. The
        // showdown normalization piece lives at the flop level (per-flop
        // FlopChanceTable already computes num_combinations); the preflop
        // analog is whatever the preflop-terminal showdown needs.
        //
        // For preflop terminals (folds, all-ins), the right normalization
        // depends on the terminal kind. Compute placeholder = product of
        // per-player reach sums for now; refine in P1.5.5 (PreflopStartGame
        // impl) when terminal evaluation gets wired up.
        let num_combinations: f64 = class_weights.iter()
            .map(|w| {
                w.iter().enumerate()
                    .map(|(k, &wt)| wt as f64 * PreflopClass(k as u8).num_combos() as f64)
                    .sum::<f64>()
            })
            .product();

        PreflopChanceTable {
            num_players,
            canonical_flops,
            orbit_sizes,
            class_initial_weights: class_weights,
            num_combinations,
        }
    }

    /// Probability that the dealt flop is in the canonical orbit
    /// `canonical_idx`, conditional on the preflop class `class_idx`
    /// being held.
    ///
    /// Math (lossless 169 layout):
    /// ```text
    ///   P(F | H) = (orbit_size(F) × |expansion(H, F)|) / (n_H × 19,600)
    /// ```
    /// where `n_H = class.num_combos()` (6 / 4 / 12 for pair / suited /
    /// offsuit), and `|expansion(H, F)|` counts combos in H that don't
    /// share any card with the canonical representative F.
    ///
    /// **Conflict logic:** `expansion()` filters via OR-union over the
    /// three flop cards (a combo is excluded if it shares ANY of the
    /// three flop cards). The user-flagged check during planning.
    ///
    /// **Orbit invariance:** P1.5-pre's
    /// `per_class_compat_count_is_orbit_invariant` test proves the
    /// expansion size is the same for every orbit member, so picking
    /// the canonical representative F is fine — any orbit member gives
    /// the same count.
    ///
    /// **Sum-to-one:** Σ over canonical F of P(F | H) = 1.0 (within
    /// f32 rounding). Verified in `tests::sum_over_canonicals_is_one`.
    pub fn chance_probability_flop(&self, canonical_idx: usize, class_idx: usize) -> f32 {
        let orbit_size = self.orbit_sizes[canonical_idx] as f32;
        let class = PreflopClass(class_idx as u8);
        let n_c = class.num_combos() as f32;
        let canonical = self.canonical_flops[canonical_idx];
        let expansion_size = expansion(class, canonical).len() as f32;
        (orbit_size * expansion_size) / (n_c * FLOPS_PER_HAND as f32)
    }

    /// Number of canonical flop outcomes (always 1,755 for a full deck).
    pub fn num_canonical_flops(&self) -> usize { self.canonical_flops.len() }

    /// Number of preflop classes (always 169 for the lossless layout).
    pub fn num_classes(&self) -> usize { NUM_PREFLOP_CLASSES }
}

// ─────────────────────────────────────────────────────────────────────
// P1.5.5a: expand/reduce arithmetic at the preflop → flop boundary
// ─────────────────────────────────────────────────────────────────────
//
// The lossless-169 architecture uses class-level representation at
// preflop (nh = 169) and combo-level at flop (nh = per-canonical
// surviving combos, ≈ 1,176 for a typical flop). The preflop → flop
// chance transition needs two distinct boundary operations:
//
//   expand_reach_class_to_combo:
//     reach_combo[h] = reach_class[class(h)] / |expansion(class(h), F)|
//     for h ∈ expansion(class(h), F), else 0.
//     The /|expansion| factor is probability mass conservation:
//     uniformly distributing the class's reach across its surviving
//     combos (the lossless-169 equiprobability property from P1.5-pre).
//
//   reduce_cfv_combo_to_class:
//     CFV_class[c] = (1 / |expansion(c, F)|) ×
//                    Σ over h ∈ expansion(c, F) of CFV_combo[h]
//     CFV class-level = average of CFV combo-level over the surviving
//     combos. The averaging weight comes from the same P(h | class, F)
//     = 1/|expansion(class, F)| distribution.
//
// These are NOT inverses on the same data type (one is for probability
// mass, the other for value averaging). They are well-defined separately
// and each has a clean isolated check. The targeted anchors below verify
// (a) expand preserves probability mass: Σ over expansion combos =
// reach_class for the source class; (b) reduce of a uniform-per-class
// combo array returns the original class values.

/// Build the combo layout at a given canonical flop: all 2-card combos
/// compatible with F (no shared card). Returns combos in deterministic
/// order (low_card, high_card) ascending by low_card then high_card.
///
/// This is the natural per-flop combo indexing for the runtime: the
/// flop subtree solver operates on this layout (per-flop nh =
/// `flop_combo_layout(F).len()`, which equals 1326 - 150 = 1,176 for
/// typical 3-card flops with distinct ranks/suits).
pub fn flop_combo_layout(canonical_flop: [Card; 3]) -> Vec<(Card, Card)> {
    let mut out = Vec::with_capacity(1_176);
    for c1 in 0u8..52 {
        if canonical_flop.contains(&c1) { continue; }
        for c2 in (c1 + 1)..52 {
            if canonical_flop.contains(&c2) { continue; }
            out.push((c1, c2));
        }
    }
    out
}

/// Reconcile a `FlopChanceTable`'s per-hand vector (in the table's
/// `hand_cards` order, length `num_valid`) into `flop_combo_layout`
/// order (2026-06-13, the bucketed-oracle keystone). Returns
/// `perm[layout_idx] = table_hand_idx`. Both index the SAME set of
/// board-compatible combos, so this is a bijection — gated as such by
/// `bucketed_oracle_reconciliation_gate`. The bucketed solver
/// (`run_all_root_cfv`) returns per-hand in table order; the oracle
/// must hand back per-combo in layout order, and this is the one
/// silent-bug-prone seam between them.
pub fn table_hand_to_layout_perm(
    hand_cards: &[u8],
    num_valid: usize,
    canonical_flop: [Card; 3],
) -> Vec<usize> {
    use std::collections::HashMap;
    // table hand h → its (min,max) card pair → table index.
    let mut by_pair: HashMap<(u8, u8), usize> = HashMap::with_capacity(num_valid);
    for h in 0..num_valid {
        let a = hand_cards[h * 2];
        let b = hand_cards[h * 2 + 1];
        by_pair.insert((a.min(b), a.max(b)), h);
    }
    let layout = flop_combo_layout(canonical_flop);
    assert_eq!(
        layout.len(),
        num_valid,
        "layout/table combo-count mismatch ({} vs {}) — different board-compatible \
         sets, reconciliation undefined",
        layout.len(),
        num_valid
    );
    layout
        .iter()
        .map(|&(c1, c2)| {
            *by_pair.get(&(c1.min(c2), c1.max(c2))).unwrap_or_else(|| {
                panic!("layout combo ({c1},{c2}) absent from the table — not the same combo set")
            })
        })
        .collect()
}

/// Per-class expansion sizes at a given canonical flop. Precomputed
/// once to avoid repeated expansion() calls. Index 0..169.
fn expansion_sizes_for_flop(canonical_flop: [Card; 3]) -> Vec<u32> {
    (0..NUM_PREFLOP_CLASSES)
        .map(|c| expansion(PreflopClass(c as u8), canonical_flop).len() as u32)
        .collect()
}

/// Expand class-level reach to combo-level reach at a canonical flop.
///
/// ```text
///   reach_combo[h] = reach_class[class(h)] / |expansion(class(h), F)|
///                    for h ∈ combo_layout AND h ∈ expansion(class(h), F)
///                  = 0  otherwise
/// ```
///
/// `combo_layout` typically comes from `flop_combo_layout(F)`. The
/// function is general — passing any layout works, but combos NOT in
/// `expansion(class(h), F)` (i.e., conflicting with F) get reach 0.
///
/// **Probability mass conservation:** for any class C, Σ over combos
/// h in combo_layout with class(h)==C of reach_combo[h] equals
/// reach_class[C] (modulo numerical precision). Verified by the
/// `expand_reach_preserves_class_mass` test.
pub fn expand_reach_class_to_combo(
    canonical_flop: [Card; 3],
    reach_class: &[f32],
    combo_layout: &[(Card, Card)],
) -> Vec<f32> {
    assert_eq!(reach_class.len(), NUM_PREFLOP_CLASSES,
        "reach_class length {} != NUM_PREFLOP_CLASSES {}",
        reach_class.len(), NUM_PREFLOP_CLASSES);
    let exp_sizes = expansion_sizes_for_flop(canonical_flop);
    let mut reach_combo = vec![0.0f32; combo_layout.len()];
    for (idx, &(c1, c2)) in combo_layout.iter().enumerate() {
        // Skip if combo conflicts with the flop (shouldn't happen if
        // combo_layout was built from flop_combo_layout(F), but be
        // defensive — preserves the 0-for-blocked semantic).
        if canonical_flop.contains(&c1) || canonical_flop.contains(&c2) {
            continue;
        }
        let class = PreflopClass::from_combo(c1, c2);
        let n_exp = exp_sizes[class.index()];
        if n_exp > 0 {
            reach_combo[idx] = reach_class[class.index()] / (n_exp as f32);
        }
    }
    reach_combo
}

/// Reduce per-combo CFV to per-class CFV at a canonical flop.
///
/// ```text
///   CFV_class[c] = (1 / |expansion(c, F)|) ×
///                  Σ over h in combo_layout with class(h)==c of CFV_combo[h]
///              = 0 if |expansion(c, F)| == 0 (class fully blocked by F)
/// ```
///
/// **Identity property under expand_value:** if CFV_combo was
/// constructed as "uniform within class" (every combo h in
/// expansion(c, F) has CFV_combo[h] = K_c for some per-class
/// constant K_c), then reduce returns K_c per class (the averaging
/// collapses to the constant). Verified by the
/// `reduce_recovers_class_uniform_values` test.
pub fn reduce_cfv_combo_to_class(
    canonical_flop: [Card; 3],
    cfv_combo: &[f32],
    combo_layout: &[(Card, Card)],
) -> Vec<f32> {
    assert_eq!(cfv_combo.len(), combo_layout.len(),
        "cfv_combo length {} != combo_layout length {}",
        cfv_combo.len(), combo_layout.len());
    let exp_sizes = expansion_sizes_for_flop(canonical_flop);
    let mut cfv_class = vec![0.0f32; NUM_PREFLOP_CLASSES];
    // Sum CFV per class
    for (idx, &(c1, c2)) in combo_layout.iter().enumerate() {
        if canonical_flop.contains(&c1) || canonical_flop.contains(&c2) {
            continue;
        }
        let class = PreflopClass::from_combo(c1, c2);
        cfv_class[class.index()] += cfv_combo[idx];
    }
    // Average by dividing by |expansion|
    for c in 0..NUM_PREFLOP_CLASSES {
        if exp_sizes[c] > 0 {
            cfv_class[c] /= exp_sizes[c] as f32;
        }
        // else: cfv_class[c] stays 0 (class fully blocked by F)
    }
    cfv_class
}

// ─────────────────────────────────────────────────────────────────────
// P1.5.5b Slice 2: stratified canonical subset + subset aggregation
// ─────────────────────────────────────────────────────────────────────
//
// Per the wall-clock observation in Slice 1 (~1s per per-flop solve at
// nh=1176, ×1755 ≈ 30 min full-scale), CI tests sub-sample canonicals.
// Sub-sampling is stratified, not random: the user-emphasized rule is
// that "subset passes" must imply "full passes" by COVERAGE rather than
// LUCK, because the highest-risk primitive (the per-(class, F)
// survivor count) is exactly the one most exercised in the cases a
// random subset would under-sample.
//
// Stratification axes:
//   - Orbit size (×4 / ×12 / ×24): catches bugs sensitive to orbit
//     multiplicity. Each stratum has different orbit counts:
//     299 / 1170 / 286 canonicals respectively (from P0).
//   - Flop rank structure (trip / paired / rainbow): catches
//     survivor-count edge cases. Trip flops drive class survivor count
//     to 0 for the trip-rank class (most stress on blocked-class
//     handling in expand/reduce). Paired flops are intermediate.
//
// The subset selector below picks samples from each stratum + each
// rank structure, giving a small (~50-100) but representative
// subset that exercises the survivor-count arithmetic across its
// stressful corners.

/// Categorize a 3-card flop's rank structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FlopRankShape {
    /// All three cards have the same rank (e.g., KcKdKh).
    Trip,
    /// Exactly two cards share a rank (e.g., KcKd7s).
    Paired,
    /// All three cards have distinct ranks (e.g., Kc7d2s).
    Rainbow,
}

pub fn flop_rank_shape(flop: [Card; 3]) -> FlopRankShape {
    let r0 = flop[0] >> 2;
    let r1 = flop[1] >> 2;
    let r2 = flop[2] >> 2;
    if r0 == r1 && r1 == r2 { FlopRankShape::Trip }
    else if r0 == r1 || r1 == r2 || r0 == r2 { FlopRankShape::Paired }
    else { FlopRankShape::Rainbow }
}

/// Build a stratified subset of canonical-flop indices covering all
/// (orbit_size, rank_shape) cells. For each (orbit_size, rank_shape)
/// cell, picks up to `n_per_cell` indices in deterministic order
/// (lex order on canonical_flops, which is the order
/// `enumerate_canonical_flops` returns).
///
/// Cells:
///   3 orbit sizes (4, 12, 24) × 3 rank shapes (Trip, Paired, Rainbow)
///   = 9 cells maximum. Some cells may be empty (e.g., Trip + orbit 12)
///   and are silently skipped.
///
/// Returns sorted, deduplicated subset of canonical indices.
pub fn stratified_canonical_subset(
    table: &PreflopChanceTable,
    n_per_cell: usize,
) -> Vec<usize> {
    use std::collections::BTreeMap;
    // Group indices by (orbit_size, rank_shape) cell
    let mut by_cell: BTreeMap<(u32, FlopRankShape), Vec<usize>> = BTreeMap::new();
    for (idx, &flop) in table.canonical_flops.iter().enumerate() {
        let orbit = table.orbit_sizes[idx];
        let shape = flop_rank_shape(flop);
        by_cell.entry((orbit, shape)).or_default().push(idx);
    }
    let mut subset = Vec::new();
    for (_, indices) in by_cell {
        subset.extend(indices.into_iter().take(n_per_cell));
    }
    subset.sort();
    subset.dedup();
    subset
}

/// Subset variant of `aggregate_preflop_chance`. Sums orbit-weighted
/// chance-probability × per-canonical V_class_at_F over a subset of
/// canonical-flop indices.
///
/// Use case: validation tests that run per-flop solves only on a
/// stratified subset of canonicals. The per-class CFV produced is the
/// PARTIAL sum (over the subset, weighted by orbit), not the full
/// preflop CFV. Both the runtime side (this function) and the
/// reference side (in tests) sum over the same subset; their agreement
/// validates the orchestration arithmetic.
///
/// **Args:**
///   - `table`: PreflopChanceTable (the full 1,755-canonical context;
///     orbit weights are read from here)
///   - `subset_indices`: canonical indices to aggregate over (subset
///     of 0..1755). Same indices the per-flop solves were run on.
///   - `flop_cfvs_subset`: `[subset_indices.len()][NUM_PREFLOP_CLASSES]`,
///     V_class_at_F for each canonical in the subset
///
/// **Returns:** `[NUM_PREFLOP_CLASSES]` partial preflop CFV.
pub fn aggregate_preflop_chance_subset(
    table: &PreflopChanceTable,
    subset_indices: &[usize],
    flop_cfvs_subset: &[Vec<f32>],
) -> Vec<f32> {
    let nh = NUM_PREFLOP_CLASSES;
    assert_eq!(subset_indices.len(), flop_cfvs_subset.len(),
        "subset_indices length {} != flop_cfvs_subset length {}",
        subset_indices.len(), flop_cfvs_subset.len());

    let mut out = vec![0.0f32; nh];
    for (i, &canonical_idx) in subset_indices.iter().enumerate() {
        let cfvs_for_flop = &flop_cfvs_subset[i];
        assert_eq!(cfvs_for_flop.len(), nh,
            "flop_cfvs_subset[{}] length {} != NUM_PREFLOP_CLASSES {}",
            i, cfvs_for_flop.len(), nh);
        for class_idx in 0..nh {
            let p = table.chance_probability_flop(canonical_idx, class_idx);
            out[class_idx] += p * cfvs_for_flop[class_idx];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// P1.5.5b Slice 1: per-canonical real V_flop computation
// ─────────────────────────────────────────────────────────────────────
//
// The "real" replacement for P2.5a's V_flop_stub. Given a canonical
// flop, per-player combo-level initial weights, and a flop-rooted
// FlatTree, runs the existing FlopStartVectorCfr for one CFR
// iteration and returns the per-combo CFV at the flop root for a
// specified traverser.
//
// Architectural choices made unilaterally (surface for reversal if wrong):
//   - Fresh flop-rooted FlatTree built by the CALLER (not extracted
//     from a preflop tree). Simpler, decouples the orchestrator from
//     a specific preflop tree's pot context.
//   - Uses existing FlopStartVectorCfr (anchored by its own tests).
//     No new flop-solver code.
//   - Iter-0 only (one call to .run with num_iterations=1). Validates
//     the orchestration chance-integration math; full CFR convergence
//     is a separate future piece.
//
// Memory note: instantiates one FlopStartVectorCfr per call. The
// caller (orchestrator in Slice 2) will hold 1,755 of these at
// validation scale (tractable at small nh). Production memory remains
// the deferred #43 footprint work; see #44 task description for the
// trail.

/// Compute combo-level CFV at the flop root for a single canonical
/// flop, at iter-0 with uniform strategy initialization.
///
/// **Args:**
///   - `canonical_flop`: the 3-card flop board for this per-flop solve
///   - `flop_tree`: a FlatTree rooted at BoardState::Flop with the
///     desired action structure. Caller's responsibility to build.
///   - `combo_ranges_per_player`: per-player initial reach at the
///     combo level for this canonical, shape `[num_players][1326]`
///     (NUM_POSSIBLE_HANDS), zero for combos blocked by the flop or
///     outside the player's range.
///   - `traverser`: player index 0..num_players whose CFV we return
///
/// **Returns:** `Vec<f32>` of length per-flop nh (= number of valid
/// hands at this flop = combos not blocking the flop board), giving
/// V_combo at the flop root for `traverser`. Plus the layout:
/// `Vec<(Card, Card)>` of the corresponding combos (so the caller
/// can match each V_combo[i] back to a specific combo).
pub fn compute_v_flop_at_root_iter0(
    canonical_flop: [Card; 3],
    flop_tree: &FlatTree,
    combo_ranges_per_player: &[Vec<f32>],
    traverser: u8,
) -> (Vec<f32>, Vec<(Card, Card)>) {
    let num_players = combo_ranges_per_player.len() as u8;
    assert!(num_players >= 2, "need at least 2 players");
    for (p, r) in combo_ranges_per_player.iter().enumerate() {
        assert_eq!(r.len(), crate::card::NUM_POSSIBLE_HANDS,
            "combo_ranges_per_player[{}] length {} != NUM_POSSIBLE_HANDS {}",
            p, r.len(), crate::card::NUM_POSSIBLE_HANDS);
    }

    // Build FlopChanceTable from the canonical flop board + the
    // caller's per-player ranges. The table internally filters to
    // valid hands (those not blocking the flop), giving the per-flop
    // hand layout.
    let board: Vec<Card> = canonical_flop.iter().copied().collect();
    let table = FlopChanceTable::compute_flop_start(
        &board, combo_ranges_per_player, num_players,
    );
    let nh = table.num_valid;
    let layout: Vec<(Card, Card)> = (0..nh)
        .map(|i| {
            let c1 = table.hand_cards[i * 2];
            let c2 = table.hand_cards[i * 2 + 1];
            (c1, c2)
        })
        .collect();

    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(flop_tree, game.table());

    // Use vanilla CFR (alpha=beta=gamma=1) for iter-0 — the iter-0
    // CFV doesn't depend on DCFR params (those weight regret
    // accumulation, not the per-iter CFV computation), so vanilla
    // keeps the math simple and deterministic.
    solver.set_vanilla_mode(true);

    // Run one iteration. Note: .run() averages CFV across traversers
    // internally. For a per-traverser CFV at iter-0, we instead use
    // the lower-level path (compute reach + bottom_up_zone) once for
    // the specified traverser.
    //
    // At iter-0, regrets are all zero, so regret_matching produces uniform
    // strategy. Uniform is identical across all (tc, rc) boards, so the
    // per-pair compute is also called inline below (matching the converged
    // variant's pattern) to make this independent of the iter-0-uniform
    // coincidence: any future regret state would be picked up correctly.
    solver.compute_flop_strategy(flop_tree);
    let reach = solver.compute_reach_flop(flop_tree, &game);

    let nn = flop_tree.num_nodes();
    let mut cfv = vec![0.0f32; nn * nh];
    // DcfrParams at iteration 0 — params weight regret accumulation,
    // not the per-iter CFV computation, so the specific values don't
    // affect the CFV at root that we return. We use iteration=0 for
    // determinism.
    let params = crate::solver::flop_start_vector_cfr::DcfrParams::new(0);

    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();

    use crate::solver::flop_start_vector_cfr::Zone;

    // River: per-pair strategy compute + bottom_up_zone(River).
    for (ti, &tc_card) in turn_deck.iter().enumerate() {
        let river_deck = &table_ref.river_decks[tc_card as usize];
        for ri in 0..river_deck.len() {
            solver.compute_river_strategy_for_pair(flop_tree, ti, ri);
            solver.bottom_up_zone(
                flop_tree, table_ref, traverser, &reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params,
            );
        }
    }
    // Turn: per-tc strategy compute + bottom_up_zone(Turn).
    for ti in 0..turn_deck.len() {
        solver.compute_turn_strategy_for_tc(flop_tree, ti);
        solver.bottom_up_zone(
            flop_tree, table_ref, traverser, &reach, &mut cfv,
            Zone::Turn, Some(ti), None, &params,
        );
    }
    // Flop: single pass.
    solver.bottom_up_zone(
        flop_tree, table_ref, traverser, &reach, &mut cfv,
        Zone::Flop, None, None, &params,
    );

    // Root CFV is at cfv[0..nh].
    let root_cfv = cfv[0..nh].to_vec();
    (root_cfv, layout)
}

/// Per the lead's directive (2026-06-04): the engine's per_flop_solver
/// MUST use converged postflop, not iter-0. Iter-0 postflop means the
/// engine is solving "preflop CFR with fixed uniform postflop", a
/// game that isn't NLHE — and a faithful GPU reproduction of an
/// iter-0-postflop engine is a faithful reproduction of a strategy
/// you can't play. The fix is localized to the per_flop_solver
/// boundary; the P5a/b/c composition anchoring is unchanged because
/// it validated the composition, not the postflop source.
///
/// `num_postflop_iters`: how many CFR iterations to run on the per-
/// flop subgame before reading the converged CFV. Convergence rate
/// depends on per-flop nh and tree complexity; the convergence_audit
/// reference uses 100 iters at nh=4 reaching 0.37% pot exploitability.
/// At production nh=1176, more iters are needed; the CPU cost scales
/// accordingly (each per-flop solve becomes a 100+ iter postflop CFR
/// run instead of a single iter-0 pass — order-of-magnitude increase,
/// which is the expensive thing the GPU exists to make feasible).
///
/// Returns `(root_cfv, layout)` matching `compute_v_flop_at_root_iter0`'s
/// signature for drop-in API compatibility; semantics differ in that
/// the CFV is computed under the time-averaged Nash-converging
/// strategy rather than uniform iter-0.
pub fn compute_v_flop_at_root_converged(
    canonical_flop: [Card; 3],
    flop_tree: &FlatTree,
    combo_ranges_per_player: &[Vec<f32>],
    traverser: u8,
    num_postflop_iters: u32,
) -> (Vec<f32>, Vec<(Card, Card)>) {
    let num_players = combo_ranges_per_player.len() as u8;
    assert!(num_players >= 2);
    for (p, r) in combo_ranges_per_player.iter().enumerate() {
        assert_eq!(r.len(), crate::card::NUM_POSSIBLE_HANDS,
            "combo_ranges_per_player[{}] length {} != NUM_POSSIBLE_HANDS",
            p, r.len());
    }

    let board: Vec<Card> = canonical_flop.iter().copied().collect();
    let table = FlopChanceTable::compute_flop_start(
        &board, combo_ranges_per_player, num_players,
    );
    compute_v_flop_at_root_converged_with_table(table, flop_tree, traverser, num_postflop_iters)
}

/// Table-generic variant of `compute_v_flop_at_root_converged`: same
/// solve-freeze-extract pipeline, but the caller provides the
/// `FlopChanceTable` directly. This is the seam that lets cheap
/// SAMPLED tables (built via `compute_flop_start_subset_with_decks`
/// with a small `chosen` hand set and sampled turn/river decks) run
/// through the SAME post-#105 extraction code instead of reimplementing
/// it. Motivating use: the preflop bootstrap's oracle (P8) — the
/// full-range path (nh ≈ 1176, 47×46 runouts) measured ≥ 42 min per
/// call, while sampled tables solve in ms-to-seconds.
pub fn compute_v_flop_at_root_converged_with_table(
    table: FlopChanceTable,
    flop_tree: &FlatTree,
    traverser: u8,
    num_postflop_iters: u32,
) -> (Vec<f32>, Vec<(Card, Card)>) {
    let nh = table.num_valid;
    let layout: Vec<(Card, Card)> = (0..nh)
        .map(|i| (table.hand_cards[i * 2], table.hand_cards[i * 2 + 1]))
        .collect();

    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(flop_tree, game.table());

    // CFR convergence run on the postflop subgame.
    //
    // DCFR (not vanilla): use the production solver's discount schedule
    // for faster convergence. The convergence_audit measured 0.37% pot
    // at 100 iters on nh=4, so 100+ iters is a reasonable starting
    // budget; caller picks `num_postflop_iters` based on the desired
    // convergence depth.
    let _ = solver.run(flop_tree, &game, num_postflop_iters);

    // Freeze the time-averaged FLOP strategy (small, fully materialized).
    // Turn/river freezes happen per-outcome just before they're read,
    // because strategy_turn/river are SCRATCH (one outcome's worth each)
    // post the 2026-06-05 streaming-strategy refactor.
    solver.freeze_average_strategy_flop(flop_tree);

    // BUGFIX 2026-06-07 (#105): the prior implementation called
    // compute_reach_flop ONCE and passed that flop_reach buffer to
    // bottom_up_zone for all three zones. compute_reach_flop only fills
    // Flop-zone reach (plus turn-chance-children at the flop/turn
    // boundary); turn-zone and river-zone player/terminal nodes had
    // reach = 0 in the buffer. bottom_up_zone reads opp_reach from this
    // buffer at terminals, so river/turn terminals received opp_reach = 0
    // → showdown CFV = 0 → only flop-fold terminals contributed to v_root.
    //
    // The CFR run() loop (FlopStartVectorCfr::run) does this correctly:
    // per (tc, ri) it computes turn_reach + river_reach explicitly,
    // weights child CFVs by chance_probability_turn/river when bubbling
    // up, and uses separate accumulator buffers per zone. This rewrite
    // mirrors run()'s structure but with FROZEN averaged strategies
    // instead of regret-matched current strategies.
    //
    // Anchor: tests/p1_5_4_step2d14_freeze_extract_anchor.rs builds an
    // independent walker (top-down/bottom-up + standing showdown oracle)
    // and a fixed extraction (this code's shape); both produce the same
    // CFV bit-exactly at K=1 on a tiny config, and both satisfy the
    // card-conflict game-theory invariant (v[weak_hand] < 0 < v[strong_hand]).
    // The pre-fix output was constant per hand, violating that invariant.
    //
    // Impact: UnabstractedPostflopOracle (production blueprint pipeline)
    // calls this function. Pre-fix, the unified preflop+postflop loop has
    // been optimizing preflop against postflop CFVs missing the showdown
    // contribution. The bug was hidden by the same degenerate showdown
    // (mutually-blocking pick_subset) that hid bug 2 in #104.
    let nn = flop_tree.num_nodes();
    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();
    let params = crate::solver::flop_start_vector_cfr::DcfrParams::new(0);

    use crate::solver::flop_start_vector_cfr::Zone;

    // Flop reach (frozen strategy_flop populated by freeze above).
    let mut flop_reach = solver.compute_reach_flop(flop_tree, &game);
    // Per-hand EV: normalize the opponent's reach to sum 1 (HU has one
    // opponent; the terminal is linear in its reach) so the live-2 CFV is on
    // the SAME chip scale as the live-3/4/5 banked path (root_cfv_from_avg) and
    // the rollouts. Without this, live-2's CFV is ~nh× too large and the
    // preflop can't compare HU against multiway. Reach: [node][player][hand].
    let np_eval = flop_reach.len() / (nn * nh);
    crate::solver::bucketed_flop_cfr::normalize_opponent_reach(&mut flop_reach, np_eval, nh, traverser as usize);

    // Buffers — mirror run()'s zone-separated accumulator pattern.
    let mut cfv = vec![0.0f32; nn * nh];
    let mut river_cfv_accum = vec![0.0f32; nn * nh];
    let mut turn_cfv = vec![0.0f32; nn * nh];
    let mut flop_cfv = vec![0.0f32; nn * nh];

    // Reset flop_cfv at turn-chance-children before per-tc accumulation.
    for &child_id in solver.turn_chance_children() {
        let off = child_id as usize * nh;
        for h in 0..nh { flop_cfv[off + h] = 0.0; }
    }

    for (ti, &tc_card) in turn_deck.iter().enumerate() {
        // STREAMING-FREEZE: populate turn scratch with the time-averaged
        // strategy for THIS ti before compute_reach_turn / bottom_up_zone(Turn) read it.
        solver.freeze_average_strategy_for_turn(flop_tree, ti);
        // Compute turn-zone reach using the FROZEN strategy_turn.
        let turn_reach = solver.compute_reach_turn(flop_tree, ti, &flop_reach);
        let river_deck = &table_ref.river_decks[tc_card as usize];

        // Reset river accumulator slots for this turn.
        for &child_id in solver.river_chance_children() {
            let off = child_id as usize * nh;
            for h in 0..nh { river_cfv_accum[off + h] = 0.0; }
        }

        for ri in 0..river_deck.len() {
            // HYBRID A: load this pair's persistent regrets+cum_strategy from
            // disk (no-op in InMemory mode).
            solver.load_river_pair(ti, ri)
                .expect("load_river_pair failed (DiskBacked I/O) in compute_v_flop_at_root_converged");
            // STREAMING-FREEZE: populate river scratch with the time-averaged
            // strategy for THIS (ti, ri) before compute_reach_river / bottom_up_zone(River) read it.
            solver.freeze_average_strategy_for_river_pair(flop_tree, ti, ri);
            // Compute river-zone reach using the FROZEN strategy_river.
            let river_reach = solver.compute_reach_river(flop_tree, ti, ri, &turn_reach);

            solver.bottom_up_zone(
                flop_tree, table_ref, traverser, &river_reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params,
            );

            // bottom_up_zone(River) mutates regrets+cum_strategy in the traverser
            // branch (DCFR update). Save to keep disk in sync.
            solver.save_river_pair(ti, ri)
                .expect("save_river_pair failed (DiskBacked I/O) in compute_v_flop_at_root_converged");

            // Weight river-chance-children CFVs by chance_probability_river, accumulate.
            for &child_id in solver.river_chance_children() {
                for h in 0..nh {
                    let cp = table_ref.chance_probability_river(tc_card, ri, h);
                    river_cfv_accum[child_id as usize * nh + h] +=
                        cp * cfv[child_id as usize * nh + h];
                }
            }
        }

        // Seed turn_cfv at river-chance-children from river accumulator.
        for &child_id in solver.river_chance_children() {
            for h in 0..nh {
                turn_cfv[child_id as usize * nh + h] =
                    river_cfv_accum[child_id as usize * nh + h];
            }
        }

        solver.bottom_up_zone(
            flop_tree, table_ref, traverser, &turn_reach, &mut turn_cfv,
            Zone::Turn, Some(ti), None, &params,
        );

        // Weight turn-chance-children CFVs by chance_probability_turn, accumulate.
        for &child_id in solver.turn_chance_children() {
            for h in 0..nh {
                let cp = table_ref.chance_probability_turn(ti, h);
                flop_cfv[child_id as usize * nh + h] +=
                    cp * turn_cfv[child_id as usize * nh + h];
            }
        }
    }

    // Seed cfv at turn-chance-children from flop accumulator.
    for &child_id in solver.turn_chance_children() {
        for h in 0..nh {
            cfv[child_id as usize * nh + h] = flop_cfv[child_id as usize * nh + h];
        }
    }

    solver.bottom_up_zone(
        flop_tree, table_ref, traverser, &flop_reach, &mut cfv,
        Zone::Flop, None, None, &params,
    );

    let root_cfv = cfv[0..nh].to_vec();
    (root_cfv, layout)
}

/// Extract the flop-root CFV per hand (traverser) from a solver whose
/// `cum_strategy_*` is ALREADY the converged time-average — the gated
/// extraction tail shared by the CPU path (after `solver.run`) and the GPU
/// path (after injecting `MetalFlopStartSolver`'s downloaded cum). Identical
/// to the body of `compute_v_flop_at_root_converged_with_table` post-run:
/// freeze avg → per-zone reach + bottom-up CFV → opponent-reach-sum-1
/// normalization (so live-2 is on the same chip scale as the live-3/4/5 banked
/// path and the rollouts).
pub fn extract_v_flop_root_from_cum(
    solver: &mut FlopStartVectorCfr,
    flop_tree: &FlatTree,
    game: &FlopStartGame,
    traverser: u8,
) -> (Vec<f32>, Vec<(Card, Card)>) {
    let table_ref = game.table();
    let nh = table_ref.num_valid;
    let layout: Vec<(Card, Card)> = (0..nh)
        .map(|i| (table_ref.hand_cards[i * 2], table_ref.hand_cards[i * 2 + 1]))
        .collect();

    solver.freeze_average_strategy_flop(flop_tree);

    let nn = flop_tree.num_nodes();
    let turn_deck = table_ref.remaining_deck.clone();
    let params = crate::solver::flop_start_vector_cfr::DcfrParams::new(0);
    use crate::solver::flop_start_vector_cfr::Zone;

    let mut flop_reach = solver.compute_reach_flop(flop_tree, game);
    let np_eval = flop_reach.len() / (nn * nh);
    crate::solver::bucketed_flop_cfr::normalize_opponent_reach(
        &mut flop_reach,
        np_eval,
        nh,
        traverser as usize,
    );

    let mut cfv = vec![0.0f32; nn * nh];
    let mut river_cfv_accum = vec![0.0f32; nn * nh];
    let mut turn_cfv = vec![0.0f32; nn * nh];
    let mut flop_cfv = vec![0.0f32; nn * nh];

    for &child_id in solver.turn_chance_children() {
        let off = child_id as usize * nh;
        for h in 0..nh {
            flop_cfv[off + h] = 0.0;
        }
    }

    for (ti, &tc_card) in turn_deck.iter().enumerate() {
        solver.freeze_average_strategy_for_turn(flop_tree, ti);
        let turn_reach = solver.compute_reach_turn(flop_tree, ti, &flop_reach);
        let river_deck = &table_ref.river_decks[tc_card as usize];

        for &child_id in solver.river_chance_children() {
            let off = child_id as usize * nh;
            for h in 0..nh {
                river_cfv_accum[off + h] = 0.0;
            }
        }

        for ri in 0..river_deck.len() {
            solver.load_river_pair(ti, ri).expect("load_river_pair (extract_from_cum)");
            solver.freeze_average_strategy_for_river_pair(flop_tree, ti, ri);
            let river_reach = solver.compute_reach_river(flop_tree, ti, ri, &turn_reach);
            solver.bottom_up_zone(
                flop_tree, table_ref, traverser, &river_reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params,
            );
            solver.save_river_pair(ti, ri).expect("save_river_pair (extract_from_cum)");
            for &child_id in solver.river_chance_children() {
                for h in 0..nh {
                    let cp = table_ref.chance_probability_river(tc_card, ri, h);
                    river_cfv_accum[child_id as usize * nh + h] +=
                        cp * cfv[child_id as usize * nh + h];
                }
            }
        }

        for &child_id in solver.river_chance_children() {
            for h in 0..nh {
                turn_cfv[child_id as usize * nh + h] =
                    river_cfv_accum[child_id as usize * nh + h];
            }
        }

        solver.bottom_up_zone(
            flop_tree, table_ref, traverser, &turn_reach, &mut turn_cfv,
            Zone::Turn, Some(ti), None, &params,
        );

        for &child_id in solver.turn_chance_children() {
            for h in 0..nh {
                let cp = table_ref.chance_probability_turn(ti, h);
                flop_cfv[child_id as usize * nh + h] +=
                    cp * turn_cfv[child_id as usize * nh + h];
            }
        }

        if std::env::var("TRACE_AGG").is_ok() {
            let (mut tc_sq, mut fc_sq) = (0.0f32, 0.0f32);
            for &c in solver.turn_chance_children() {
                for h in 0..nh {
                    let t = turn_cfv[c as usize * nh + h];
                    tc_sq += t * t;
                    let f = flop_cfv[c as usize * nh + h];
                    fc_sq += f * f;
                }
            }
            eprintln!("TRACE_AGG ti {ti}: turn_cfv@tcc sumsq {tc_sq:.6} | flop_cfv@tcc(accum) sumsq {fc_sq:.6}");
        }
    }

    for &child_id in solver.turn_chance_children() {
        for h in 0..nh {
            cfv[child_id as usize * nh + h] = flop_cfv[child_id as usize * nh + h];
        }
    }

    solver.bottom_up_zone(
        flop_tree, table_ref, traverser, &flop_reach, &mut cfv,
        Zone::Flop, None, None, &params,
    );

    if std::env::var("TRACE_AGG").is_ok() {
        let root_sq: f32 = (0..nh).map(|h| cfv[h] * cfv[h]).sum();
        eprintln!("TRACE_AGG root cfv sumsq {root_sq:.6}");
    }

    let root_cfv = cfv[0..nh].to_vec();
    (root_cfv, layout)
}

/// GPU exact live-2 (HU): solve the flop-start subgame on `MetalFlopStartSolver`
/// (the measured bottleneck — the 34-iter exact showdown sweep moves to GPU),
/// then extract per-traverser flop-root CFVs via the gated CPU extraction by
/// injecting the GPU's converged cum into a CPU solver (same ZoneDims layout,
/// [flop|turn|river] contiguous). ONE GPU solve, then re-inject the clean cum
/// before each traverser's extraction (bottom_up_zone mutates cum). Returns
/// per-traverser CFV (table/layout order) + the shared combo layout.
#[cfg(feature = "metal")]
pub fn compute_v_flop_root_gpu_per_traverser(
    ctx: &crate::gpu_metal::context::MetalContext,
    table: FlopChanceTable,
    flop_tree: &FlatTree,
    np: u8,
    num_postflop_iters: u32,
) -> (Vec<Vec<f32>>, Vec<(Card, Card)>) {
    use crate::gpu_metal::flop_solver::MetalFlopStartSolver;
    let game = FlopStartGame::new(table);
    let mut cpu = FlopStartVectorCfr::new(flop_tree, game.table());
    let mut gpu = MetalFlopStartSolver::new(ctx, flop_tree, &game, &cpu);
    gpu.run(ctx, flop_tree, &game, num_postflop_iters);

    let cum = gpu.download_cum_strategy();
    let nf = cpu.cum_strategy_flop().len();
    let nt = cpu.cum_strategy_turn().len();
    let nr = cpu.cum_strategy_river().len();
    assert_eq!(
        nf + nt + nr,
        cum.len(),
        "GPU cum length {} != CPU flop+turn+river {}+{}+{}",
        cum.len(), nf, nt, nr
    );

    let mut layout = Vec::new();
    let per_trav: Vec<Vec<f32>> = (0..np)
        .map(|t| {
            // re-inject the clean converged cum (extraction mutates it).
            cpu.cum_strategy_flop_mut().copy_from_slice(&cum[0..nf]);
            cpu.cum_strategy_turn_mut().copy_from_slice(&cum[nf..nf + nt]);
            cpu.cum_strategy_river_mut().copy_from_slice(&cum[nf + nt..]);
            let (v, lay) = extract_v_flop_root_from_cum(&mut cpu, flop_tree, &game, t);
            if layout.is_empty() {
                layout = lay;
            }
            v
        })
        .collect();
    (per_trav, layout)
}

// ─────────────────────────────────────────────────────────────────────
// P1.5.5a runtime-shape function (test target for P2.5a)
// ─────────────────────────────────────────────────────────────────────

/// Compute per-class preflop CFV via the runtime path: loop over 1,755
/// canonical flops, call `v_flop_fn(F_canon, combo)` for each surviving
/// combo, reduce combo CFV to class CFV via `reduce_cfv_combo_to_class`,
/// aggregate across canonicals via `aggregate_preflop_chance` with
/// orbit weights.
///
/// This is the RUNTIME path P2.5a tests. It will also be the bottom-up
/// pass used by P1.5.5b's orchestrator (with `v_flop_fn` replaced by a
/// real per-flop solver instead of the test stub).
///
/// Returns `[NUM_PREFLOP_CLASSES]` per-class CFV.
pub fn compute_preflop_cfv_per_canonical_pass(
    table: &PreflopChanceTable,
    v_flop_fn: impl Fn([Card; 3], (Card, Card)) -> f32,
) -> Vec<f32> {
    let n_canon = table.num_canonical_flops();
    let mut per_canonical_v_class: Vec<Vec<f32>> = Vec::with_capacity(n_canon);
    for canonical_idx in 0..n_canon {
        let f_canon = table.canonical_flops[canonical_idx];
        let layout = flop_combo_layout(f_canon);
        let v_combo: Vec<f32> = layout.iter()
            .map(|&combo| v_flop_fn(f_canon, combo))
            .collect();
        // Runtime uses P1.5.5a's reduce. P2.5a's reference does NOT use this;
        // it computes class CFV by explicit enumeration so the per-class
        // survivor-count arithmetic is independently anchored.
        let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
        per_canonical_v_class.push(v_class);
    }
    // Runtime uses P1.5.4's orbit-weighted aggregation. P2.5a's reference
    // does NOT use this; it sums with explicit 1/19,600 unit weights per
    // actual flop so the orbit weighting is independently anchored.
    aggregate_preflop_chance(table, &per_canonical_v_class)
}

// ─────────────────────────────────────────────────────────────────────
// P1.5.4: aggregate_preflop_chance (orbit-weighted CFV aggregation)
// ─────────────────────────────────────────────────────────────────────

/// Aggregate per-canonical-flop CFVs into the preflop chance node's
/// CFV at the lossless 169-class layout.
///
/// Math:
/// ```text
///   V_preflop[class] = Σ over canonical F of P(F | class) × V_flop[F, class]
/// ```
/// where `P(F | class) = (orbit_size(F) × |expansion(class, F)|) /
/// (n_class × 19,600)` from `chance_probability_flop`.
///
/// This is the NEW ORCHESTRATION SHAPE for preflop→flop chance
/// integration: the existing flop→turn and turn→river transitions
/// aggregate over single chance cards; the preflop→flop transition
/// aggregates over 1,755 canonical flops weighted by orbit sizes
/// (× 4 / × 12 / × 24).
///
/// **Args:**
///   - `table`: PreflopChanceTable holding the orbit weights and
///     canonical-flop list
///   - `flop_cfvs`: per-canonical-flop value vectors, shape
///     `[1755][NUM_PREFLOP_CLASSES]`. `flop_cfvs[canonical][class]`
///     is the value at the start of canonical flop's subtree for a
///     player holding `class`. Must equal `table.num_canonical_flops()`
///     length; each inner vec must have NUM_PREFLOP_CLASSES entries.
///
/// **Returns:** `[NUM_PREFLOP_CLASSES]` aggregated CFV at the preflop
/// chance node.
///
/// **Isolation check coverage (`aggregate_preflop_chance_orbit_weighted`):**
/// Hand-build a flop_cfvs map where exactly one canonical has a known
/// non-zero value and verify the result matches the manually-computed
/// orbit-weighted product. This discriminates against off-by-one in the
/// loop (visits all canonicals exactly once), wrong probability lookup
/// (uses the right (canonical, class) probability), and accumulator
/// wrong-init (starts at 0, sums correctly).
pub fn aggregate_preflop_chance(
    table: &PreflopChanceTable,
    flop_cfvs: &[Vec<f32>],
) -> Vec<f32> {
    let n_canon = table.num_canonical_flops();
    let nh = NUM_PREFLOP_CLASSES;
    assert_eq!(flop_cfvs.len(), n_canon,
        "flop_cfvs length {} does not match num_canonical_flops {}",
        flop_cfvs.len(), n_canon);

    let mut out = vec![0.0f32; nh];
    for (canonical_idx, cfvs_for_flop) in flop_cfvs.iter().enumerate() {
        assert_eq!(cfvs_for_flop.len(), nh,
            "flop_cfvs[{}] length {} != NUM_PREFLOP_CLASSES {}",
            canonical_idx, cfvs_for_flop.len(), nh);
        for class_idx in 0..nh {
            let p = table.chance_probability_flop(canonical_idx, class_idx);
            out[class_idx] += p * cfvs_for_flop[class_idx];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// P1.5.3: compute_reach_preflop (free function, button-first reach order)
// ─────────────────────────────────────────────────────────────────────

/// Top-down reach propagation through the preflop zone at the lossless
/// 169-class layout.
///
/// Returns a Vec<f32> of shape `[nn * num_players * NUM_PREFLOP_CLASSES]`,
/// where `reach[nid * np * nh + p * nh + h]` is the probability that
/// player `p`, holding class `h`, reaches node `nid` along the chosen
/// strategy profile.
///
/// **Button-first action order:** the function reads `node.player_id`
/// for the acting player, which the tree builder set to the
/// button-first convention for preflop nodes (P1 in HU = SB = button,
/// validated by P2.1's `preflop_action_order` tests). This function
/// therefore inherits button-first ordering automatically — provided
/// it correctly dispatches on `node.player_id` and does NOT hardcode
/// any player index. That is the property `compute_reach_preflop_uses_
/// button_player_at_root` in `tests` discriminates: a hand-built
/// strategy that scales player 1's reach by σ at the root must produce
/// a child whose player-1 reach equals σ × initial weight, while
/// player 0's reach is unchanged — failure of this property would
/// catch the bug of hardcoding player 0 (the postflop-first-actor
/// assumption leaking into the preflop reach path).
///
/// **Stops at preflop chance:** at the preflop → flop chance node,
/// reach is propagated unchanged to the chance child (which is in
/// the Flop zone). The chance-probability weighting enters the CFV
/// pass downstream (P1.5.4's bottom_up_zone), not the reach pass.
///
/// **Args:**
///   - `tree`: the preflop-start FlatTree (root must be at
///     `BoardState::Preflop`)
///   - `num_players`: player count (matches `tree.num_players` for
///     real trees; explicit for testability)
///   - `zones`: zone classification from FlopStartVectorCfr::new (so
///     we filter to Preflop-zone nodes). Pass `&[]` to skip filtering
///     and walk all nodes (useful for isolated tests; the production
///     path should pass the real zones).
///   - `initial_class_weights`: `[num_players][NUM_PREFLOP_CLASSES]`
///   - `preflop_strategies`: one optional `Vec<f32>` per tree node;
///     `Some(sigma)` at preflop decision nodes (sigma layout
///     `[num_actions * NUM_PREFLOP_CLASSES]`), `None` elsewhere.
///     Decoupled from FlopStartVectorCfr's `self.strategy_*` storage
///     because preflop strategy storage doesn't exist yet (P1.5.4 wires it).
pub fn compute_reach_preflop(
    tree: &FlatTree,
    num_players: u8,
    zones: &[crate::solver::flop_start_vector_cfr::Zone],
    initial_class_weights: &[Vec<f32>],
    preflop_strategies: &[Option<Vec<f32>>],
) -> Vec<f32> {
    use crate::solver::flop_start_vector_cfr::Zone;
    let nn = tree.num_nodes();
    let np = num_players as usize;
    let nh = NUM_PREFLOP_CLASSES;
    let mut reach = vec![0.0f32; nn * np * nh];

    assert_eq!(initial_class_weights.len(), np);
    for p in 0..np {
        assert_eq!(initial_class_weights[p].len(), nh);
        for h in 0..nh { reach[p * nh + h] = initial_class_weights[p][h]; }
    }
    assert_eq!(preflop_strategies.len(), nn);

    let use_zone_filter = !zones.is_empty();

    for level in 0..=tree.max_depth {
        for &nid in tree.nodes_at_level(level as u32) {
            let idx = nid as usize;
            if use_zone_filter && zones[idx] != Zone::Preflop { continue; }
            let node = &tree.nodes[idx];

            if node.is_player() {
                let player = node.player_id as usize;
                let na = node.num_children as usize;
                let sigma = match &preflop_strategies[idx] {
                    Some(s) => s,
                    None => continue, // no strategy: leave children with zero reach
                };
                assert_eq!(sigma.len(), na * nh,
                    "strategy at node {} has length {}, expected {} ({} actions × {} classes)",
                    idx, sigma.len(), na * nh, na, nh);

                for (a, &child) in tree.node_children(idx).iter().enumerate() {
                    let src = idx * np * nh;
                    let dst = child as usize * np * nh;
                    // Inherit reach from parent for every (player, class).
                    for p in 0..np {
                        for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                    }
                    // Multiply the ACTING player's reach by their strategy
                    // for this action. The acting player is `node.player_id`
                    // — the button-first preflop convention is baked into
                    // the tree builder (P2.1 anchor), not redundantly here.
                    for h in 0..nh {
                        reach[dst + player * nh + h] *= sigma[a * nh + h];
                    }
                }
            } else if node.is_chance() {
                // Preflop → Flop chance: propagate reach unchanged to the
                // chance child (which is in Flop zone). Chance-prob
                // weighting enters CFV pass, not reach pass.
                for &child in tree.node_children(idx) {
                    let src = idx * np * nh;
                    let dst = child as usize * np * nh;
                    for p in 0..np {
                        for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                    }
                }
            }
        }
    }

    reach
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Targeted P1.5.1 anchor: probability sums to 1.0 for every class.
    /// This is the normalization check on `chance_probability_flop`. If
    /// the orbit weighting + expansion arithmetic is consistent with the
    /// underlying counts, every class's probabilities across all 1,755
    /// canonical flops must sum to exactly 1.0 (within f32 rounding).
    #[test]
    fn sum_over_canonicals_is_one() {
        // HU with uniform 1.0 weights. The probability function doesn't
        // depend on weights, so this validates the math independent of
        // any reach setup.
        let table = PreflopChanceTable::new(
            2,
            vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
        );
        for class_idx in 0..NUM_PREFLOP_CLASSES {
            let sum: f64 = (0..table.num_canonical_flops())
                .map(|c| table.chance_probability_flop(c, class_idx) as f64)
                .sum();
            // f32 rounding tolerance — 1,755 additions of f32 values
            // accumulates a few ULPs. 1e-5 is well within f32 floor.
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "class {} probability sum = {}, expected 1.0",
                class_idx, sum,
            );
        }
    }

    /// Targeted P1.5.1 anchor: conflict OR-union returns zero exactly
    /// when every combo in the class shares at least one card with
    /// every orbit member of the canonical.
    ///
    /// Two concrete cases (the third is exhaustive across all classes
    /// for one specific canonical):
    ///
    /// 1. Class AA paired with canonical flop containing three aces of
    ///    three distinct suits. Every AA combo (AhAd, AhAc, ..., 6 of
    ///    them) uses two of {Ah, Ad, Ac, As}, and every orbit member
    ///    of a "three aces" canonical uses three of those four cards.
    ///    Some AA combos will use a card not on a specific orbit
    ///    member (the 4th ace), but other orbit members swap suits.
    ///    The orbit-invariance result means the per-(class, canonical)
    ///    compat count is determined by structure, not which orbit
    ///    member we picked.
    ///
    /// 2. Class 22 paired with canonical [As, Kh, Qd]: every 22 combo
    ///    uses two cards from {2c, 2d, 2h, 2s}, none of which appear
    ///    on the flop. Expansion = 6 (all combos compatible).
    #[test]
    fn conflict_or_zero_when_all_combos_blocked() {
        let table = PreflopChanceTable::new(
            2,
            vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
        );

        // Find the canonical with three aces of distinct suits.
        // Cards: rank 12 (A), suits 0/1/2/3 → cards 48, 49, 50, 51.
        // Any 3-of-4 aces is a flop; canonicalizes to one canonical.
        let three_aces_flop: [Card; 3] = [48, 49, 50];
        let three_aces_canonical =
            crate::abstraction::flop_isomorphism::canonicalize_flop(three_aces_flop);
        let aces_idx = table.canonical_flops.iter()
            .position(|&f| f == three_aces_canonical)
            .expect("three-aces canonical must exist");

        // Class AA: index 0 (pair of aces, since 12 - 12 = 0).
        let aa = PreflopClass::from_combo(48, 49).index(); // AhAd
        assert_eq!(aa, 0, "AA must be class 0");

        // Probability of three-aces canonical given AA: every AA combo
        // uses 2 of {Ah, Ad, Ac, As}; the canonical representative uses
        // 3 of those. A combo (X, Y) is compat with the flop iff neither
        // X nor Y is on the flop. Since the flop has 3 of 4 aces, only
        // combos using BOTH non-flop aces are compat — but no AA combo
        // uses just the 1 non-flop ace twice (would need that ace + itself).
        // So expansion count = 0, probability = 0.
        let p = table.chance_probability_flop(aces_idx, aa);
        assert_eq!(p, 0.0,
            "AA + three-aces canonical: probability must be 0 (every AA combo blocked)");

        // 22 + [As, Kh, Qd] (non-2 flop): every 22 combo compat.
        let mixed_flop: [Card; 3] = [
            (12 << 2) | 3, // As
            (11 << 2) | 2, // Kh
            (10 << 2) | 1, // Qd
        ];
        let mixed_canonical =
            crate::abstraction::flop_isomorphism::canonicalize_flop(mixed_flop);
        let mixed_idx = table.canonical_flops.iter()
            .position(|&f| f == mixed_canonical)
            .expect("mixed canonical must exist");

        let twos = PreflopClass::from_combo(0, 1).index(); // 2c2d
        // 22 is class index 12 (pair of rank 0 = "2" → 12 - 0 = 12).
        assert_eq!(twos, 12, "22 must be class 12");

        let p = table.chance_probability_flop(mixed_idx, twos);
        assert!(p > 0.0,
            "22 + non-2 canonical: probability must be > 0 (all 6 combos compat)");
        // Specifically: expansion = 6, orbit size for AsKhQd-canonical depends
        // on canonical_flop's symmetry. P = (orbit_size × 6) / (6 × 19600).
        let orbit_size = table.orbit_sizes[mixed_idx] as f32;
        let expected = orbit_size * 6.0 / (6.0 * FLOPS_PER_HAND as f32);
        assert!((p - expected).abs() < 1e-7,
            "P({} | 22) = {}, expected {} (orbit_size {} × 6 / (6 × 19600))",
            mixed_idx, p, expected, orbit_size);
    }

    /// Probabilities are non-negative.
    #[test]
    fn probabilities_non_negative() {
        let table = PreflopChanceTable::new(
            2,
            vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
        );
        for c in 0..table.num_canonical_flops() {
            for k in 0..NUM_PREFLOP_CLASSES {
                let p = table.chance_probability_flop(c, k);
                assert!(p >= 0.0, "P({} | class {}) = {} < 0", c, k, p);
                assert!(p <= 1.0, "P({} | class {}) = {} > 1", c, k, p);
            }
        }
    }

    /// Targeted P1.5.3 anchor: compute_reach_preflop applies the acting
    /// player's strategy to the acting player's reach slot, where the
    /// acting player is `node.player_id` (button-first at preflop root
    /// per P2.1's tree-side anchor).
    ///
    /// THE DISCRIMINATOR: build a preflop tree, set a non-uniform
    /// strategy at the root. The root acts player_id = 1 (button = SB in
    /// HU). After reach propagation:
    ///   - child[0]: player 1's reach scaled by σ[0] (≈ 0.7); player 0's
    ///     reach unchanged (still = initial = 1.0)
    ///   - child[1]: player 1's reach scaled by σ[1] (≈ 0.3); player 0
    ///     unchanged
    ///
    /// If the function bug hardcoded player 0 (the postflop-first-actor
    /// assumption leaking in), player 0's reach at the child would be
    /// 0.7 / 0.3 instead of 1.0. That's the failure this catches.
    #[test]
    fn compute_reach_preflop_uses_button_player_at_root() {
        use crate::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
        use crate::tree::builder::build_tree;

        // HU preflop, simple action set (matches preflop_tree_smoke).
        let cfg = TreeConfig {
            num_players: 2,
            initial_state: BoardState::Preflop,
            starting_pot: 3,
            starting_stacks: vec![100, 100],
            initial_contributions: vec![2, 1],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions {
                bet: vec![BetSize::PotRelative(1.0)],
                raise: vec![],
            },
            add_allin_threshold: 1.0,
            force_allin_threshold: 1.0,
            merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
            no_open_limp: false,
            threebet_or_fold: false,

        };
        let tree = build_tree(&cfg).expect("preflop tree builds");

        let np = cfg.num_players as usize;
        let nh = NUM_PREFLOP_CLASSES;
        let nn = tree.num_nodes();

        // Sanity: root acts player 1 (button, validated by
        // preflop_action_order tests).
        assert_eq!(tree.nodes[0].player_id, 1,
            "preflop root must act player 1 (button); if this changed, \
             the P2.1 anchor regressed and this test's discriminator \
             premise breaks");
        let root_na = tree.nodes[0].num_children as usize;

        // Non-uniform strategy at the root only, uniform per class.
        // σ[a, h] is action `a`'s probability for class `h`. To make
        // the discriminator unambiguous, use σ[0] = 0.7, σ[1] = 0.3 for
        // every class (or split evenly if root has more children — but
        // for HU preflop with our cfg, root has 2 children).
        let mut preflop_strategies: Vec<Option<Vec<f32>>> = vec![None; nn];
        let mut root_sigma = vec![0.0f32; root_na * nh];
        let sigma_per_action: Vec<f32> = match root_na {
            2 => vec![0.7, 0.3],
            3 => vec![0.5, 0.3, 0.2],
            4 => vec![0.4, 0.3, 0.2, 0.1],
            _ => panic!("unexpected root_na = {}", root_na),
        };
        for a in 0..root_na {
            for h in 0..nh {
                root_sigma[a * nh + h] = sigma_per_action[a];
            }
        }
        preflop_strategies[0] = Some(root_sigma);

        // Initial class weights: uniform 1.0 for both players. (The
        // probability function doesn't care about normalization for
        // this discriminator; we just need a known starting value.)
        let initial_class_weights = vec![vec![1.0f32; nh]; np];

        // Run reach propagation with no zone filter (walk all nodes,
        // strategies at non-preflop nodes are None so they'd be skipped
        // anyway). The zones-arg-empty mode is documented for testability.
        let reach = compute_reach_preflop(
            &tree,
            cfg.num_players,
            &[],
            &initial_class_weights,
            &preflop_strategies,
        );

        // Identify the first chance descendant's TWO PARENT player-children
        // — root has root_na = 2 children, indices read from the tree.
        let root_children: Vec<u32> = tree.node_children(0).to_vec();
        assert_eq!(root_children.len(), 2, "HU preflop root has 2 children");

        // For each child of root, verify:
        //   reach[child][player=1][h] == sigma_per_action[a]  (button scaled)
        //   reach[child][player=0][h] == 1.0                  (BB unchanged)
        for (a, &child) in root_children.iter().enumerate() {
            let base = child as usize * np * nh;
            let p1_reach = reach[base + 1 * nh + 0];
            let p0_reach = reach[base + 0 * nh + 0];
            assert!(
                (p1_reach - sigma_per_action[a]).abs() < 1e-7,
                "child {} (action {}): player 1 (button) reach = {}, expected {} \
                 (strategy applied to acting player's slot)",
                child, a, p1_reach, sigma_per_action[a],
            );
            assert!(
                (p0_reach - 1.0).abs() < 1e-7,
                "child {} (action {}): player 0 (BB) reach = {}, expected 1.0 \
                 (non-acting player's reach should pass through unchanged). \
                 If this is {}, the function hardcoded player 0 as the actor \
                 instead of reading node.player_id — the postflop-first-actor \
                 bug leaking into preflop, exactly the failure mode P1.5.3 \
                 anchors against.",
                child, a, p0_reach, sigma_per_action[a],
            );
        }
    }

    /// Targeted P1.5.4 anchor: aggregate_preflop_chance correctly
    /// orbit-weights per-canonical-flop CFVs into the preflop-zone
    /// CFV at the chance node.
    ///
    /// THE DISCRIMINATOR: pick one specific canonical flop F_pick with
    /// known orbit size, and one specific class H_pick with known
    /// expansion count for F_pick. Set flop_cfvs[F_pick][H_pick] =
    /// K (some non-zero), all others = 0. The aggregated CFV must
    /// equal:
    /// ```
    ///   aggregate[H_pick] = K × P(F_pick | H_pick)
    ///                     = K × (orbit_size(F_pick) × |expansion(H_pick, F_pick)|)
    ///                       / (n_H_pick × 19,600)
    /// ```
    ///
    /// What this catches:
    ///   - off-by-one in the canonical-flop loop (visits each exactly once)
    ///   - wrong (canonical, class) lookup (right probability per pair)
    ///   - accumulator not initialized to 0 (sum starts clean)
    ///   - typo in the multiply-accumulate (correct product)
    ///
    /// And a second case where flop_cfvs is constant across all flops:
    /// aggregate[class] = constant × Σ P(F | class) = constant × 1.0 = constant
    /// (this is the sum-to-one normalization re-applied at the aggregation
    /// level — confirms the aggregation respects the chance-prob normalization).
    #[test]
    fn aggregate_preflop_chance_orbit_weighted() {
        let table = PreflopChanceTable::new(
            2,
            vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
        );
        let n_canon = table.num_canonical_flops();
        let nh = NUM_PREFLOP_CLASSES;

        // ── Discriminator 1: single non-zero canonical ──
        // Pick canonical at index 0 (some lex-first canonical flop).
        // Pick class AA (index 0). K = 7.5 to avoid round-trip with
        // simple powers of 2.
        let pick_canon = 0;
        let pick_class = 0; // AA
        let k = 7.5_f32;

        let mut flop_cfvs: Vec<Vec<f32>> = vec![vec![0.0f32; nh]; n_canon];
        flop_cfvs[pick_canon][pick_class] = k;

        let aggregated = aggregate_preflop_chance(&table, &flop_cfvs);

        let expected = k * table.chance_probability_flop(pick_canon, pick_class);
        assert!(
            (aggregated[pick_class] - expected).abs() < 1e-6,
            "single-non-zero discriminator: aggregated[AA] = {}, expected K × P = {} × {} = {}",
            aggregated[pick_class], k,
            table.chance_probability_flop(pick_canon, pick_class), expected,
        );

        // All OTHER classes must remain 0 at the picked canonical (since
        // we only put a non-zero at AA), and all other canonicals
        // contribute 0 to AA — so aggregated[other_class] = 0.
        for c in 0..nh {
            if c == pick_class { continue; }
            assert!(
                aggregated[c].abs() < 1e-7,
                "aggregated[{}] = {}, expected 0 (only AA at canonical {} was non-zero)",
                c, aggregated[c], pick_canon,
            );
        }

        // ── Discriminator 2: constant across all canonicals → constant out ──
        // Sets flop_cfvs[any][AA] = K. By the normalization
        // Σ P(F | AA) = 1, the aggregate must equal K for AA.
        let mut flop_cfvs_const: Vec<Vec<f32>> = vec![vec![0.0f32; nh]; n_canon];
        let k2 = 3.25_f32;
        for canonical_idx in 0..n_canon {
            flop_cfvs_const[canonical_idx][pick_class] = k2;
        }
        let aggregated_const = aggregate_preflop_chance(&table, &flop_cfvs_const);
        assert!(
            (aggregated_const[pick_class] - k2).abs() < 1e-4, // f32 rounding ~1e-5 from 1755 adds
            "constant discriminator: aggregated[AA] = {}, expected K2 = {} \
             (Σ P = 1.0 normalization should produce this exactly within f32)",
            aggregated_const[pick_class], k2,
        );
    }

    /// Targeted P1.5.5a anchor (1/2): expand_reach_class_to_combo
    /// preserves probability mass per class.
    ///
    /// For any class C with reach R_C, the sum of reach_combo over
    /// the combos in C's expansion must equal R_C. This is THE
    /// probability mass conservation property that makes the expand
    /// step a valid representation of the same probability across
    /// the class → combo level change.
    ///
    /// Failure mode this catches: forgetting the /|expansion| factor
    /// (sum would equal R_C × |expansion| instead of R_C), or applying
    /// the wrong factor (sum would be off by some ratio).
    #[test]
    fn expand_reach_preserves_class_mass() {
        // Use a typical 3-card flop (non-paired, non-suited) to exercise
        // a generic expansion shape.
        let flop: [Card; 3] = [
            (12 << 2) | 3, // As
            (11 << 2) | 2, // Kh
            (10 << 2) | 1, // Qd
        ];
        let layout = flop_combo_layout(flop);
        assert_eq!(layout.len(), 1_176,
            "flop combo layout for non-blocking flop has 1326 - 150 = 1176 combos");

        // Set reach uniformly to 1.0 for all classes — each class's
        // surviving combos should sum to 1.0.
        let reach_class: Vec<f32> = vec![1.0f32; NUM_PREFLOP_CLASSES];
        let reach_combo = expand_reach_class_to_combo(flop, &reach_class, &layout);

        // For each class with non-zero expansion, sum the reach_combo
        // over its combos in layout. Must equal reach_class[c] = 1.0.
        for c in 0..NUM_PREFLOP_CLASSES {
            let class = PreflopClass(c as u8);
            let exp = expansion(class, flop);
            let exp_size = exp.len();
            if exp_size == 0 { continue; }  // fully blocked, skip

            let sum_in_class: f32 = layout.iter().enumerate()
                .filter(|(_, &combo)| {
                    PreflopClass::from_combo(combo.0, combo.1) == class
                })
                .map(|(idx, _)| reach_combo[idx])
                .sum();

            assert!(
                (sum_in_class - 1.0).abs() < 1e-6,
                "class {} (n_combos={}, expansion size={}): combo-level sum = {}, expected 1.0 \
                 (probability mass conservation). If this is 1.0 × expansion_size = {}, \
                 the /|expansion| factor was omitted (the expand-without-divide bug).",
                c, class.num_combos(), exp_size, sum_in_class, exp_size as f32
            );
        }

        // Also check a non-uniform class-level reach: only AA gets R = 5.
        let mut reach_class2 = vec![0.0f32; NUM_PREFLOP_CLASSES];
        let aa = 0;
        reach_class2[aa] = 5.0;
        let reach_combo2 = expand_reach_class_to_combo(flop, &reach_class2, &layout);

        // Sum over AA-class combos in layout must equal 5.0.
        let aa_class = PreflopClass(0);
        let aa_sum: f32 = layout.iter().enumerate()
            .filter(|(_, &combo)| PreflopClass::from_combo(combo.0, combo.1) == aa_class)
            .map(|(idx, _)| reach_combo2[idx])
            .sum();
        assert!(
            (aa_sum - 5.0).abs() < 1e-6,
            "AA combo-level sum = {}, expected 5.0 (single-class non-uniform mass)",
            aa_sum
        );

        // All non-AA-class combos must have reach_combo = 0.
        for (idx, &combo) in layout.iter().enumerate() {
            if PreflopClass::from_combo(combo.0, combo.1) != aa_class {
                assert_eq!(reach_combo2[idx], 0.0,
                    "non-AA combo {:?} got reach {} from AA-only class-level reach",
                    combo, reach_combo2[idx]);
            }
        }
    }

    /// Targeted P1.5.5a anchor (2/2): reduce_cfv_combo_to_class
    /// recovers per-class values when combo-level values are uniform
    /// within class.
    ///
    /// Construct CFV_combo[h] = K_class for h in expansion(class, F),
    /// for some per-class constants K_c. Then reduce(CFV_combo) must
    /// return K_c per class (the averaging collapses to the constant
    /// when the inputs are all equal within a class).
    ///
    /// Failure modes this catches: wrong averaging count (off by one in
    /// |expansion|), wrong combo-to-class mapping (combos assigned to
    /// wrong classes), forgetting to skip blocked combos (would change
    /// the average).
    #[test]
    fn reduce_recovers_class_uniform_values() {
        let flop: [Card; 3] = [
            (12 << 2) | 3, // As
            (11 << 2) | 2, // Kh
            (10 << 2) | 1, // Qd
        ];
        let layout = flop_combo_layout(flop);

        // Construct CFV_combo as "uniform within class" using class
        // index as the constant (so class 0 has value 0.0, class 1
        // has value 1.0, ..., class 168 has value 168.0).
        let cfv_combo: Vec<f32> = layout.iter()
            .map(|&(c1, c2)| {
                let class = PreflopClass::from_combo(c1, c2);
                class.index() as f32
            })
            .collect();

        let cfv_class = reduce_cfv_combo_to_class(flop, &cfv_combo, &layout);

        // For every non-blocked class, reduce must return the class index.
        for c in 0..NUM_PREFLOP_CLASSES {
            let class = PreflopClass(c as u8);
            let exp_size = expansion(class, flop).len();
            if exp_size == 0 {
                // Blocked class: reduce returns 0.
                assert_eq!(cfv_class[c], 0.0,
                    "class {} fully blocked by flop {:?}, reduce should return 0",
                    c, flop);
            } else {
                assert!(
                    (cfv_class[c] - c as f32).abs() < 1e-5,
                    "class {} (expansion size {}): reduce returned {}, expected {} \
                     (uniform-within-class average must collapse to the constant)",
                    c, exp_size, cfv_class[c], c
                );
            }
        }
    }

    /// Sanity: 1,755 canonical flops, orbit-size histogram matches the
    /// flop_isomorphism::orbit_size_histogram result from P0.
    #[test]
    fn canonical_count_and_orbit_histogram() {
        let table = PreflopChanceTable::new(
            2,
            vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
        );
        assert_eq!(table.num_canonical_flops(), 1_755);
        let mut hist: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for &os in &table.orbit_sizes {
            *hist.entry(os).or_insert(0) += 1;
        }
        // Expected from P0: {4: 299, 12: 1170, 24: 286}.
        assert_eq!(hist.get(&4), Some(&299), "orbit-size-4 count");
        assert_eq!(hist.get(&12), Some(&1170), "orbit-size-12 count");
        assert_eq!(hist.get(&24), Some(&286), "orbit-size-24 count");
    }
}
