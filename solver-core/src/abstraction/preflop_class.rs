//! Lossless 169-class preflop hand abstraction.
//!
//! Texas Hold'em has 1,326 distinct 2-card combos. Under suit-isomorphism
//! (with the off-suit/suited distinction preserved), these collapse to
//! exactly 169 equivalence classes:
//!   - 13 pocket pairs (AA, KK, ..., 22), 6 combos each
//!   - 78 suited classes (AKs, AQs, ..., 32s), 4 combos each
//!   - 78 offsuit classes (AKo, AQo, ..., 32o), 12 combos each
//!   - Total: 13*6 + 78*4 + 78*12 = 78 + 312 + 936 = 1326 ✓
//!
//! Why "lossless": the game-from-preflop value is invariant under the
//! joint suit-iso group (S_4) acting on (hole-cards, board). At nh = 169,
//! each class is its own suit-orbit, so the joint flop-iso × hand-iso
//! aggregation is exact, not approximate. This is what makes
//! `chance_probability_flop` math equal the un-canonicalized 22,100 ×
//! 1326 enumeration at max_diff = 0 (the P2.5 anchor target).
//!
//! At full nh = 1326 (every combo as its own hand), the math is only
//! approximate — see #41 discussion. That's why this layout is the
//! prerequisite for the orbit-weighted preflop chance integration.
//!
//! ## Class index layout (matters for downstream array indexing)
//! ```text
//!   0..13:   Pocket pairs:  index = 12 - rank.  AA=0, KK=1, ..., 22=12.
//!   13..91:  Suited:        index = 13 + suited_offset(r_high, r_low).
//!   91..169: Offsuit:       index = 91 + suited_offset(r_high, r_low).
//! ```
//!
//! `suited_offset(r_high, r_low)` for r_high > r_low (rank values 0..13,
//! 12 = Ace, 0 = "2"):
//!   AKs(r=12,11) = 0, AQs = 1, ..., A2s = 11,
//!   KQs = 12, KJs = 13, ..., K2s = 22,
//!   ..., 32s = 77.

use crate::card::Card;

pub const NUM_PREFLOP_CLASSES: usize = 169;
pub const NUM_PAIRS: usize = 13;
pub const NUM_SUITED: usize = 78;
pub const NUM_OFFSUIT: usize = 78;
pub const NUM_COMBOS: usize = 1326;
pub const PAIR_START: u8 = 0;
pub const SUITED_START: u8 = 13;
pub const OFFSUIT_START: u8 = 91;

/// One of the 169 lossless preflop classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PreflopClass(pub u8);

impl PreflopClass {
    pub const fn new(idx: u8) -> Self {
        assert!((idx as usize) < NUM_PREFLOP_CLASSES);
        PreflopClass(idx)
    }
    pub fn index(self) -> usize { self.0 as usize }
    pub fn is_pair(self) -> bool { self.0 < SUITED_START }
    pub fn is_suited(self) -> bool { SUITED_START <= self.0 && self.0 < OFFSUIT_START }
    pub fn is_offsuit(self) -> bool { OFFSUIT_START <= self.0 && (self.0 as usize) < NUM_PREFLOP_CLASSES }

    /// Number of underlying 2-card combos this class represents.
    /// Pairs: 6 = C(4, 2). Suited: 4 (one per suit). Offsuit: 12 = 4×3.
    pub fn num_combos(self) -> usize {
        if self.is_pair() { 6 }
        else if self.is_suited() { 4 }
        else { 12 }
    }

    /// Construct the class containing a given (c1, c2) combo. Combo is
    /// order-independent (PreflopClass::from_combo(a, b) == from_combo(b, a)).
    pub fn from_combo(c1: Card, c2: Card) -> Self {
        let (r1, s1) = (c1 >> 2, c1 & 3);
        let (r2, s2) = (c2 >> 2, c2 & 3);
        let (r_high, r_low) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
        if r_high == r_low {
            // pair: index = 12 - rank
            PreflopClass(12 - r_high)
        } else {
            let off = suited_offset(r_high, r_low);
            if s1 == s2 {
                PreflopClass(SUITED_START + off)
            } else {
                PreflopClass(OFFSUIT_START + off)
            }
        }
    }
}

/// `suited_offset(r_high, r_low)` returns 0..78 for r_high > r_low.
/// Layout: lex-descending by r_high then by r_high - 1 - r_low (so that
/// within a fixed r_high, r_low = r_high - 1 is first). See module doc.
fn suited_offset(r_high: u8, r_low: u8) -> u8 {
    debug_assert!(r_high > r_low);
    debug_assert!((r_high as usize) < 13);
    // pairs_above(r_high) = number of (r1, r2) with r1 > r_high
    //                    = Σ over r=r_high+1..=12 of r
    //                    = (12*13)/2 - r_high*(r_high+1)/2
    //                    = 78 - r_high*(r_high+1)/2
    let pairs_above = 78 - (r_high as u16) * (r_high as u16 + 1) / 2;
    let within = r_high - 1 - r_low;
    (pairs_above + within as u16) as u8
}

/// Invert `suited_offset`: given an offset 0..78, return (r_high, r_low).
fn invert_suited_offset(off: u8) -> (u8, u8) {
    debug_assert!((off as usize) < NUM_SUITED);
    let off = off as u16;
    // Find r_high s.t. pairs_above(r_high) <= off < pairs_above(r_high) + r_high
    // (where r_high is the count of classes with first rank = r_high).
    for r_high in (1..=12u16).rev() {
        let pa = 78 - r_high * (r_high + 1) / 2;
        let count_this_rank = r_high;
        if pa <= off && off < pa + count_this_rank {
            let within = off - pa;
            let r_low = (r_high - 1 - within) as u8;
            return (r_high as u8, r_low);
        }
    }
    panic!("invalid suited offset {}", off);
}

/// All 2-card combos in this preflop class. Combos are returned with
/// `c1` being the higher-rank card (or, for pairs, the lower-suit card).
/// Order within the returned vector is deterministic but not specified
/// to consumers; treat it as an unordered set.
pub fn class_combos(class: PreflopClass) -> Vec<(Card, Card)> {
    if class.is_pair() {
        let rank = 12 - class.0;
        let mut out = Vec::with_capacity(6);
        for s1 in 0u8..4 {
            for s2 in (s1 + 1)..4 {
                out.push(((rank << 2) | s1, (rank << 2) | s2));
            }
        }
        out
    } else {
        let suited = class.is_suited();
        let off = class.0 - if suited { SUITED_START } else { OFFSUIT_START };
        let (r_high, r_low) = invert_suited_offset(off);
        let cap = if suited { 4 } else { 12 };
        let mut out = Vec::with_capacity(cap);
        for s_high in 0u8..4 {
            for s_low in 0u8..4 {
                let is_suited_combo = s_high == s_low;
                if is_suited_combo == suited {
                    out.push(((r_high << 2) | s_high, (r_low << 2) | s_low));
                }
            }
        }
        out
    }
}

/// Combos in the class that are compatible with a given 3-card flop
/// (no shared card). This is the per-flop expansion the runtime uses
/// to map a preflop class to the surviving combos at the flop level.
///
/// Conflict logic is OR over the three flop cards (a combo is excluded
/// if it shares ANY of the three flop cards), as the user flagged
/// during the preflop integration planning.
pub fn expansion(class: PreflopClass, flop: [Card; 3]) -> Vec<(Card, Card)> {
    class_combos(class)
        .into_iter()
        .filter(|&(c1, c2)| !flop.contains(&c1) && !flop.contains(&c2))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::flop_isomorphism::{
        enumerate_all_flops, enumerate_canonical_flops, orbit_of,
    };

    #[test]
    fn class_counts_sum_to_1326() {
        let mut total = 0usize;
        for c in 0..NUM_PREFLOP_CLASSES {
            total += PreflopClass(c as u8).num_combos();
        }
        assert_eq!(total, NUM_COMBOS, "total combos across all classes must be 1326");
    }

    #[test]
    fn class_buckets_have_expected_sizes() {
        let mut n_pair = 0;
        let mut n_suited = 0;
        let mut n_offsuit = 0;
        for c in 0..NUM_PREFLOP_CLASSES {
            let pc = PreflopClass(c as u8);
            if pc.is_pair() { n_pair += 1; assert_eq!(pc.num_combos(), 6); }
            else if pc.is_suited() { n_suited += 1; assert_eq!(pc.num_combos(), 4); }
            else if pc.is_offsuit() { n_offsuit += 1; assert_eq!(pc.num_combos(), 12); }
            else { panic!("class {} fell out of all three buckets", c); }
        }
        assert_eq!(n_pair, NUM_PAIRS);
        assert_eq!(n_suited, NUM_SUITED);
        assert_eq!(n_offsuit, NUM_OFFSUIT);
    }

    #[test]
    fn combo_roundtrip_for_all_1326_combos() {
        // Every (c1, c2) combo with c1 != c2 (52 * 51 / 2 = 1326) must
        // roundtrip through from_combo -> class_combos. Each combo must
        // appear in exactly one class's combo list.
        let mut counts: std::collections::HashMap<(u8, u8), usize> = std::collections::HashMap::new();
        for c1 in 0u8..52 {
            for c2 in (c1 + 1)..52 {
                let class = PreflopClass::from_combo(c1, c2);
                let combos = class_combos(class);
                let canonical = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                let found = combos.iter().any(|&(a, b)| {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    (lo, hi) == canonical
                });
                assert!(
                    found,
                    "combo ({}, {}) classified as {:?} but not in its combo list",
                    c1, c2, class
                );
                *counts.entry(canonical).or_insert(0) += 1;
            }
        }
        assert_eq!(counts.len(), NUM_COMBOS, "all 1326 unique combos roundtripped");
        for (combo, n) in &counts {
            assert_eq!(*n, 1, "combo {:?} classified {} times (should be exactly 1)", combo, n);
        }
    }

    #[test]
    fn suited_offset_invertible() {
        for r_high in 1u8..=12 {
            for r_low in 0u8..r_high {
                let off = suited_offset(r_high, r_low);
                let (rh, rl) = invert_suited_offset(off);
                assert_eq!((rh, rl), (r_high, r_low),
                    "suited_offset({}, {}) = {} inverted incorrectly to ({}, {})",
                    r_high, r_low, off, rh, rl);
            }
        }
    }

    #[test]
    fn expansion_count_matches_per_flop_total() {
        // For any 3-card flop, the total number of compatible 2-card
        // combos across all 169 classes equals 1326 - 150 = 1176
        // (cards not on the flop choose 2 minus pairs of flop cards;
        // by inclusion-exclusion: hands using ≥1 flop card = 3*51 - 3 = 150).
        let test_flops: Vec<[Card; 3]> = vec![
            [0, 1, 2],
            [5, 13, 28],
            [12, 13, 14], // adjacent rank, same suit progression
            [0, 17, 34],  // three suited
            [12, 24, 36], // three "K" cards of three suits (ranks differ)
        ];
        for flop in test_flops {
            let total: usize = (0..NUM_PREFLOP_CLASSES)
                .map(|c| expansion(PreflopClass(c as u8), flop).len())
                .sum();
            assert_eq!(total, 1326 - 150,
                "flop {:?} has {} compatible combos across all classes, expected 1176",
                flop, total);
        }
    }

    /// Per-class compat orbit invariance: for every (canonical, class),
    /// all orbit members of the canonical have the SAME number of
    /// compatible combos in the class.
    ///
    /// This is THE lossless-ness property that makes the orbit-weighted
    /// chance integration exact: the count depends only on the canonical
    /// (not the specific orbit member), because the joint flop-hand
    /// suit-iso group acts trivially on class memberships.
    ///
    /// If this test fails, the flop_isomorphism + preflop_class layers
    /// cannot be combined into a max_diff = 0 preflop chance integration,
    /// and the user-flagged "convenient structural fact" wasn't actually
    /// a fact.
    #[test]
    fn per_class_compat_count_is_orbit_invariant() {
        let canonicals = enumerate_canonical_flops();
        for canon in canonicals {
            let orbit = orbit_of(canon);
            for c in 0..NUM_PREFLOP_CLASSES {
                let class = PreflopClass(c as u8);
                let counts: Vec<usize> = orbit.iter()
                    .map(|&f| expansion(class, f).len())
                    .collect();
                let first = counts[0];
                for (i, &n) in counts.iter().enumerate() {
                    assert_eq!(n, first,
                        "class {:?} orbit-of-canonical {:?}: member {} ({:?}) has {} compat combos, member 0 ({:?}) has {} — orbit invariance broken",
                        class, canon, i, orbit[i], n, orbit[0], first);
                }
            }
        }
    }

    /// Orbit-weighted total: for each class, the sum over canonical
    /// flops weighted by orbit size of "compat combos in this class"
    /// must equal the un-canonicalized sum over all 22,100 flops of
    /// "compat combos in this class".
    ///
    /// This is the explicit P1.5-pre anchor of the orbit weighting
    /// arithmetic. The validation arc's discipline: validate the
    /// orbit-weighted aggregation against the un-canonicalized
    /// enumeration directly, not against another canonicalized
    /// implementation (the inclusion-exclusion lesson, applied to
    /// orbit weights).
    #[test]
    fn orbit_weighted_total_matches_uncanonicalized() {
        let all_flops = enumerate_all_flops();
        let canonicals = enumerate_canonical_flops();

        for c in 0..NUM_PREFLOP_CLASSES {
            let class = PreflopClass(c as u8);
            let un_canon: usize = all_flops.iter()
                .map(|&f| expansion(class, f).len())
                .sum();
            let canon_weighted: usize = canonicals.iter()
                .map(|&canon| {
                    let orbit_size = orbit_of(canon).len();
                    let compat = expansion(class, canon).len();
                    orbit_size * compat
                })
                .sum();
            assert_eq!(un_canon, canon_weighted,
                "class {:?}: un-canonicalized total = {}, orbit-weighted canonical total = {}",
                class, un_canon, canon_weighted);
        }
    }

    /// Total across all classes × all flops = 22,100 × 1176 = 25,989,600.
    /// A cheap aggregate sanity check on top of the per-class anchor.
    #[test]
    fn total_class_flop_compat_count() {
        let total: usize = enumerate_all_flops().iter()
            .map(|&f| (0..NUM_PREFLOP_CLASSES)
                .map(|c| expansion(PreflopClass(c as u8), f).len())
                .sum::<usize>())
            .sum();
        assert_eq!(total, 22_100 * 1176, "total compat (flop, combo) pairs");
    }
}
