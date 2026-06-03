//! Lossless flop suit-isomorphism canonicalization.
//!
//! Two flops are SUIT-ISOMORPHIC iff they are related by a permutation
//! of the four suits. For example, 2♥7♦K♠ and 2♣7♥K♦ are isomorphic
//! (the same flop "shape": three rainbow cards of ranks 2, 7, K), and a
//! strategy that respects suit symmetry plays them identically. There
//! are 22,100 raw flops (C(52, 3)) and 1,755 distinct equivalence
//! classes under suit isomorphism — a 12.6× compression with zero
//! information loss.
//!
//! This is the LOSSLESS half of the board-abstraction layer. The lossy
//! half (postflop information bucketing) is a separate workstream (#42)
//! with its own validation discipline. Both live in the `abstraction`
//! module but they have different correctness requirements: lossless
//! isomorphism just needs to be deterministic and orbit-respecting;
//! lossy bucketing additionally needs an EV-cost measurement against
//! the unabstracted game.
//!
//! ALGORITHM (canonical form by first-appearance suit relabeling):
//!
//! 1. Sort the 3 cards by (rank descending, suit ascending). This gives
//!    a deterministic within-flop ordering for any flop.
//! 2. Walk the sorted cards; the FIRST distinct suit encountered gets
//!    canonical label 0, the second new suit gets 1, the third gets 2.
//!    (Flops only ever use at most 3 of 4 suits, so we never need 3.)
//! 3. Re-emit each card with rank unchanged and suit replaced by its
//!    canonical label.
//!
//! Two flops produce the SAME canonical form iff they are in the same
//! suit-isomorphism orbit. The canonical form is the orbit's chosen
//! representative.
//!
//! REFERENCES: this is a standard poker-abstraction technique; the
//! count of 1,755 distinct flops is well-known in the literature.

use crate::card::Card;

/// Canonicalize a flop (3 cards) into its suit-isomorphism canonical
/// representative. The output is a sorted [3]Card array with suit labels
/// relabeled by first-appearance (suit-0 = first new suit encountered,
/// suit-1 = second new suit, etc).
///
/// Property (canonical-form determinism): for any flop `f`, running this
/// twice on `f` yields the same result.
///
/// Property (orbit-respecting): for any two flops `f1`, `f2` related by
/// a suit permutation, this returns the same canonical form.
///
/// Property (lossless): the canonical form uniquely identifies the
/// orbit; distinct orbits produce distinct canonical forms.
///
/// ALGORITHM: a single-pass first-appearance relabeling is NOT orbit-
/// respecting when within-rank-tie sort order interacts with which
/// suits get assigned which canonical labels (e.g., 9c9d8c and 9c9d8d
/// are in the same orbit under c↔d swap but a single-pass relabel
/// gives them different outputs because the sort places 9c first in
/// both cases regardless of which 9 the 8 shares suit with).
///
/// The correct approach: enumerate all 24 suit permutations of the
/// input flop, sort each canonically (rank-desc, then by canonical
/// suit), apply first-appearance relabeling to each, and pick the
/// lexicographically smallest result. Brute-force but correct, and
/// 24 perms × 3 cards is trivially fast.
pub fn canonicalize_flop(flop: [Card; 3]) -> [Card; 3] {
    let mut best: Option<[Card; 3]> = None;
    for perm in &SUIT_PERMUTATIONS {
        let mut permuted = [0u8; 3];
        for (i, &c) in flop.iter().enumerate() {
            let rank = c >> 2;
            let suit = c & 3;
            permuted[i] = (rank << 2) | perm[suit as usize];
        }
        // Sort the permuted flop by (rank desc, suit asc). Using
        // ascending sort on a key that puts higher ranks first and
        // then suit ascending.
        permuted.sort_by_key(|&c| {
            let rank = c >> 2;
            let suit = c & 3;
            (255 - rank, suit)
        });
        // After permutation, the suits may not be {0,1,2,..} starting
        // from 0. Apply first-appearance relabeling to canonicalize
        // suit labels so different permutations that produce the same
        // multiset structure agree.
        let mut canon_suits = [u8::MAX; 4];
        let mut next_canon = 0u8;
        for &c in permuted.iter() {
            let suit = c & 3;
            if canon_suits[suit as usize] == u8::MAX {
                canon_suits[suit as usize] = next_canon;
                next_canon += 1;
            }
        }
        let mut relabeled = [0u8; 3];
        for (i, &c) in permuted.iter().enumerate() {
            let rank = c >> 2;
            let suit = c & 3;
            relabeled[i] = (rank << 2) | canon_suits[suit as usize];
        }
        // Re-sort after relabeling (canonical suits may reshuffle
        // within-tie order).
        relabeled.sort_by_key(|&c| {
            let rank = c >> 2;
            let suit = c & 3;
            (255 - rank, suit)
        });
        match best {
            None => best = Some(relabeled),
            Some(b) if relabeled < b => best = Some(relabeled),
            _ => {}
        }
    }
    best.expect("at least one permutation always exists")
}

/// All 24 permutations of {0, 1, 2, 3}, used by canonicalize_flop.
const SUIT_PERMUTATIONS: [[u8; 4]; 24] = [
    [0,1,2,3], [0,1,3,2], [0,2,1,3], [0,2,3,1], [0,3,1,2], [0,3,2,1],
    [1,0,2,3], [1,0,3,2], [1,2,0,3], [1,2,3,0], [1,3,0,2], [1,3,2,0],
    [2,0,1,3], [2,0,3,1], [2,1,0,3], [2,1,3,0], [2,3,0,1], [2,3,1,0],
    [3,0,1,2], [3,0,2,1], [3,1,0,2], [3,1,2,0], [3,2,0,1], [3,2,1,0],
];

/// Enumerate every possible flop (C(52, 3) = 22,100 combinations).
/// Each flop is returned with cards sorted by raw card index ascending
/// (the conventional un-canonicalized form).
pub fn enumerate_all_flops() -> Vec<[Card; 3]> {
    let mut out = Vec::with_capacity(22_100);
    for a in 0u8..50 {
        for b in (a + 1)..51 {
            for c in (b + 1)..52 {
                out.push([a, b, c]);
            }
        }
    }
    out
}

/// Group all flops by canonical form and return the distinct canonical
/// representatives (one per orbit). The expected count is 1,755.
pub fn enumerate_canonical_flops() -> Vec<[Card; 3]> {
    use std::collections::BTreeSet;
    let mut canonicals: BTreeSet<[Card; 3]> = BTreeSet::new();
    for flop in enumerate_all_flops() {
        canonicals.insert(canonicalize_flop(flop));
    }
    canonicals.into_iter().collect()
}

/// Return all flops in the suit-isomorphism orbit of the given flop
/// (the set of all flops that canonicalize to the same form). Useful
/// for validation: every member of the orbit must canonicalize to the
/// shared canonical form.
pub fn orbit_of(flop: [Card; 3]) -> Vec<[Card; 3]> {
    let canon = canonicalize_flop(flop);
    let mut orbit = Vec::new();
    for f in enumerate_all_flops() {
        if canonicalize_flop(f) == canon {
            orbit.push(f);
        }
    }
    orbit
}

/// Count flops per canonical form. Returns a histogram: how many raw
/// flops belong to orbits of each size. Sanity check: the total across
/// all orbits should equal 22,100 (every flop accounted for).
pub fn orbit_size_histogram() -> std::collections::BTreeMap<usize, usize> {
    use std::collections::BTreeMap;
    let mut orbits: BTreeMap<[Card; 3], usize> = BTreeMap::new();
    for f in enumerate_all_flops() {
        *orbits.entry(canonicalize_flop(f)).or_insert(0) += 1;
    }
    let mut histogram: BTreeMap<usize, usize> = BTreeMap::new();
    for (_, orbit_size) in orbits {
        *histogram.entry(orbit_size).or_insert(0) += 1;
    }
    histogram
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_form_is_deterministic() {
        // Running canonicalize_flop twice on the same input gives the
        // same output.
        for f in enumerate_all_flops() {
            let canon = canonicalize_flop(f);
            let canon2 = canonicalize_flop(canon);
            assert_eq!(canon, canon2,
                "canonicalize_flop not idempotent on flop {:?}: {:?} vs {:?}",
                f, canon, canon2);
        }
    }

    #[test]
    fn orbit_respecting_under_suit_permutation() {
        // For each flop, apply every suit permutation (4! = 24) and
        // verify all permutations canonicalize to the same form.
        let perms = all_suit_permutations();
        // Test a representative sample for performance (full sweep is
        // 22,100 × 24 = 530,400 canonicalizations; takes a few seconds
        // in release mode but is OK).
        for f in enumerate_all_flops() {
            let canon = canonicalize_flop(f);
            for perm in &perms {
                let permuted = apply_suit_permutation(f, perm);
                let canon_p = canonicalize_flop(permuted);
                assert_eq!(canon, canon_p,
                    "flop {:?} canonicalized to {:?}; suit-permuted to {:?} → {:?}",
                    f, canon, permuted, canon_p);
            }
        }
    }

    #[test]
    fn canonical_count_is_1755() {
        // The number of distinct flops up to suit isomorphism is 1,755.
        // This is a well-known count from the poker abstraction
        // literature; if we get a different number, the canonicalization
        // is wrong.
        let canonicals = enumerate_canonical_flops();
        assert_eq!(canonicals.len(), 1755,
            "expected 1755 canonical flops, got {}", canonicals.len());
    }

    #[test]
    fn orbits_partition_all_flops() {
        // Every flop belongs to exactly one orbit, and total flops
        // across all orbits = 22,100. No flop is double-counted or
        // missing.
        let histogram = orbit_size_histogram();
        let total: usize = histogram.iter()
            .map(|(orbit_size, num_orbits)| orbit_size * num_orbits)
            .sum();
        assert_eq!(total, 22_100,
            "orbits do not partition flops: total = {}, expected 22,100", total);
        eprintln!("orbit size histogram: {:?}", histogram);
        let total_orbits: usize = histogram.values().sum();
        assert_eq!(total_orbits, 1755,
            "total orbit count = {}, expected 1755", total_orbits);
    }

    #[test]
    fn canonical_form_uses_lowest_suits() {
        // Canonical forms use suits 0, 1, 2 only (never 3), because
        // flops have at most 3 distinct suits.
        for canon in enumerate_canonical_flops() {
            for &c in &canon {
                let suit = c & 3;
                assert!(suit < 3,
                    "canonical flop {:?} uses suit {} (should be 0-2 only)",
                    canon, suit);
            }
        }
    }

    #[test]
    fn examples_canonicalize_as_expected() {
        // Hand-verified expected canonical forms. Card encoding:
        // c = (rank << 2) | suit, rank 0..12 (12=A), suit 0..3 (0=c, 1=d, 2=h, 3=s).
        use crate::card::flop_from_str;

        // Rainbow flop: 2h7dKs — three different suits, ranks K, 7, 2.
        // After sorting by rank-desc: K, 7, 2. Suits encountered in
        // order: s, d, h → canonical 0, 1, 2.
        // Canonical: K with suit 0 (= Kc), 7 with suit 1 (= 7d), 2 with suit 2 (= 2h).
        let f = flop_from_str("Ks7d2h").unwrap();
        let canon = canonicalize_flop(f);
        let expected = flop_from_str("Kc7d2h").unwrap();
        assert_eq!(canon, expected, "rainbow Ks7d2h canonical form");

        // Equivalent rainbow flop in a different suit perm: should
        // canonicalize to the same form.
        let f2 = flop_from_str("Kd7c2s").unwrap();
        let canon2 = canonicalize_flop(f2);
        assert_eq!(canon, canon2, "rainbow orbit collapse");

        // Monotone flop: KhQh7h — all same suit.
        // After sorting by rank-desc: K, Q, 7 all suit h. Canonical suit = 0.
        let f = flop_from_str("KhQh7h").unwrap();
        let canon = canonicalize_flop(f);
        let expected = flop_from_str("KcQc7c").unwrap();
        assert_eq!(canon, expected, "monotone KhQh7h canonical form");

        // Two-tone flop: KhQh7s — K and Q share suit, 7 different.
        // After sorting by rank-desc: K, Q (both h), 7 (s).
        // First suit encountered = h → 0. Second = s → 1.
        // Canonical: Kc Qc 7d.
        let f = flop_from_str("KhQh7s").unwrap();
        let canon = canonicalize_flop(f);
        let expected = flop_from_str("KcQc7d").unwrap();
        assert_eq!(canon, expected, "two-tone KhQh7s canonical form");

        // Trips on flop (impossible since only 4 of each rank but a
        // ranks/suits sanity: 5h5d5c — all rank 5, three suits.
        // After sorting: rank-tied 5,5,5; suit-asc breaks tie → 5c, 5d, 5h.
        // Canonical: 5c 5d 5h (already canonical).
        let f = flop_from_str("5h5d5c").unwrap();
        let canon = canonicalize_flop(f);
        let expected = flop_from_str("5c5d5h").unwrap();
        assert_eq!(canon, expected, "trips 5h5d5c canonical form");
    }

    // ---- helpers ----

    fn all_suit_permutations() -> Vec<[u8; 4]> {
        SUIT_PERMUTATIONS.to_vec()
    }

    fn apply_suit_permutation(flop: [Card; 3], perm: &[u8; 4]) -> [Card; 3] {
        let mut out = [0u8; 3];
        for (i, &c) in flop.iter().enumerate() {
            let rank = c >> 2;
            let suit = c & 3;
            let new_suit = perm[suit as usize];
            out[i] = (rank << 2) | new_suit;
        }
        out
    }
}
