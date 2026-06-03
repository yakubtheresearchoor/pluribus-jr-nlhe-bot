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
