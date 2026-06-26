//! Preflop ALL-IN equity — the one primitive the production blueprint lacks
//! (it has no explicit preflop all-in; see `blueprint::build_conn_preflop_tree`).
//! Used by the real-time "preflop jam-subgame" search to value a called jam: a
//! preflop all-in that gets called skips every postflop street and goes straight
//! to a 5-card runout, so its value is pure all-in equity — no flop continuation.
//!
//! Card encoding (crate-wide): `card = rank * 4 + suit`, rank 0..=12 (2..A),
//! suit 0..=3. Matches `hand::eval::Hand` (`rank = card/4`, `suit = card%4`).
//!
//! Two tiers:
//!   - `hu_equity_exact`: exact C(48,5) runout enumeration for ONE matchup.
//!     The validation/ground-truth path (≈1.7M boards, a few ms).
//!   - `class_hu_equity_table`: a lazily-built, disk-cached 169×169 class-vs-
//!     class HU equity matrix (MC-sampled per class pair) for runtime lookup.

use crate::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use crate::card::Card;
use crate::hand::eval::Hand;

/// Exact heads-up all-in equity of `hero` vs `opp` over a uniform 5-card runout.
/// Returns hero's equity = P(win) + 0.5·P(tie), enumerating all C(48,5) boards.
/// Cards must be distinct and in `0..52`.
pub fn hu_equity_exact(hero: [u8; 2], opp: [u8; 2]) -> f32 {
    debug_assert!(hero[0] != hero[1] && opp[0] != opp[1]);
    debug_assert!(![hero[0], hero[1]].contains(&opp[0]) && ![hero[0], hero[1]].contains(&opp[1]));
    // Remaining 48-card deck.
    let dead = [hero[0], hero[1], opp[0], opp[1]];
    let deck: Vec<usize> = (0..52usize).filter(|c| !dead.contains(&(*c as u8))).collect();
    let h_base = Hand::new()
        .add_card(hero[0] as usize)
        .add_card(hero[1] as usize);
    let o_base = Hand::new()
        .add_card(opp[0] as usize)
        .add_card(opp[1] as usize);
    let n = deck.len(); // 48
    let mut win = 0u64;
    let mut tie = 0u64;
    let mut total = 0u64;
    for a in 0..n {
        let ha = h_base.add_card(deck[a]);
        let oa = o_base.add_card(deck[a]);
        for b in (a + 1)..n {
            let hb = ha.add_card(deck[b]);
            let ob = oa.add_card(deck[b]);
            for c in (b + 1)..n {
                let hc = hb.add_card(deck[c]);
                let oc = ob.add_card(deck[c]);
                for d in (c + 1)..n {
                    let hd = hc.add_card(deck[d]);
                    let od = oc.add_card(deck[d]);
                    for e in (d + 1)..n {
                        let hr = hd.add_card(deck[e]).evaluate_full();
                        let or = od.add_card(deck[e]).evaluate_full();
                        total += 1;
                        if hr > or {
                            win += 1;
                        } else if hr == or {
                            tie += 1;
                        }
                    }
                }
            }
        }
    }
    (win as f64 + 0.5 * tie as f64) as f32 / total as f32
}

/// All combos (card pairs) belonging to a preflop class, as `[Card; 2]`.
fn class_combos(cls: usize) -> Vec<[u8; 2]> {
    let mut out = Vec::new();
    for a in 0..52u8 {
        for b in (a + 1)..52u8 {
            if PreflopClass::from_combo(a as Card, b as Card).index() == cls {
                out.push([a, b]);
            }
        }
    }
    out
}

/// MC-sampled heads-up all-in equity of class `hc` vs class `oc`, averaged over
/// all non-blocking combo pairs and `samples` random runouts each. Used to build
/// the cached table; deterministic given `seed`.
pub fn class_hu_equity_mc(hc: usize, oc: usize, samples: usize, seed: u64) -> f32 {
    let hero_combos = class_combos(hc);
    let opp_combos = class_combos(oc);
    let mut rng = seed ^ ((hc as u64) << 32) ^ (oc as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut next = |rng: &mut u64| {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng
    };
    let mut sum = 0.0f64;
    let mut cnt = 0u64;
    for &h in &hero_combos {
        for &o in &opp_combos {
            if h[0] == o[0] || h[0] == o[1] || h[1] == o[0] || h[1] == o[1] {
                continue; // blocked combo pair
            }
            let dead = [h[0], h[1], o[0], o[1]];
            let deck: Vec<usize> =
                (0..52usize).filter(|c| !dead.contains(&(*c as u8))).collect();
            let h_base = Hand::new().add_card(h[0] as usize).add_card(h[1] as usize);
            let o_base = Hand::new().add_card(o[0] as usize).add_card(o[1] as usize);
            let mut win = 0.0f64;
            for _ in 0..samples {
                // sample 5 distinct board cards from `deck` (Fisher–Yates prefix)
                let mut idx = [0usize; 5];
                let mut pool: Vec<usize> = deck.clone();
                for k in 0..5 {
                    let r = (next(&mut rng) as usize) % pool.len();
                    idx[k] = pool.swap_remove(r);
                }
                let mut hh = h_base;
                let mut oh = o_base;
                for &bc in &idx {
                    hh = hh.add_card(bc);
                    oh = oh.add_card(bc);
                }
                let hr = hh.evaluate_full();
                let or = oh.evaluate_full();
                if hr > or {
                    win += 1.0;
                } else if hr == or {
                    win += 0.5;
                }
            }
            sum += win / samples as f64;
            cnt += 1;
        }
    }
    if cnt == 0 {
        0.5
    } else {
        (sum / cnt as f64) as f32
    }
}

/// Number of distinct card-pair combos in each of the 169 preflop classes
/// (pairs=6, suited=4, offsuit=12). Class-level reach must be weighted by these
/// so cross-class terminal aggregation matches a combo-level sum.
pub fn class_combo_counts() -> Vec<f32> {
    let mut c = vec![0.0f32; NUM_PREFLOP_CLASSES];
    for a in 0..52u8 {
        for b in (a + 1)..52u8 {
            c[PreflopClass::from_combo(a as Card, b as Card).index()] += 1.0;
        }
    }
    c
}

/// 169×169 class-vs-class HU all-in equity matrix (row-major, `[hc*169+oc]`),
/// MC-sampled `samples` per pair. Serial — for tests; use the `_par` build for
/// the runtime artifact.
pub fn class_hu_equity_table(samples: usize, seed: u64) -> Vec<f32> {
    let n = NUM_PREFLOP_CLASSES;
    let mut t = vec![0.5f32; n * n];
    for hc in 0..n {
        for oc in 0..n {
            t[hc * n + oc] = class_hu_equity_mc(hc, oc, samples, seed);
        }
    }
    t
}

/// Parallel (rayon, over hero classes) build of the 169×169 HU equity table.
pub fn class_hu_equity_table_par(samples: usize, seed: u64) -> Vec<f32> {
    use rayon::prelude::*;
    let n = NUM_PREFLOP_CLASSES;
    let rows: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|hc| (0..n).map(|oc| class_hu_equity_mc(hc, oc, samples, seed)).collect())
        .collect();
    let mut t = vec![0.0f32; n * n];
    for (hc, row) in rows.into_iter().enumerate() {
        t[hc * n..(hc + 1) * n].copy_from_slice(&row);
    }
    t
}

/// Load the 169×169 HU equity table from `path` (raw little-endian f32), or build
/// it in parallel and persist it. Built once; reused across runtime restarts.
pub fn load_or_build_class_equity_table(path: &str, samples: usize, seed: u64) -> Vec<f32> {
    let n = NUM_PREFLOP_CLASSES;
    let want = n * n * 4;
    if let Ok(bytes) = std::fs::read(path) {
        if bytes.len() == want {
            let mut t = vec![0.0f32; n * n];
            for i in 0..n * n {
                t[i] = f32::from_le_bytes([
                    bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3],
                ]);
            }
            return t;
        }
    }
    let t = class_hu_equity_table_par(samples, seed);
    let mut out = Vec::with_capacity(want);
    for &v in &t {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let _ = std::fs::write(path, &out);
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    // card = rank*4 + suit; rank 0=2 .. 8=T, 9=J, 10=Q, 11=K, 12=A.
    fn c(rank: u8, suit: u8) -> u8 {
        rank * 4 + suit
    }

    /// Exact average HU equity of class `hc` vs class `oc` over ALL non-blocking
    /// combo pairs (suit-independent ground truth = the textbook class number).
    fn class_avg_exact(hc: usize, oc: usize) -> f32 {
        let hcs = class_combos(hc);
        let ocs = class_combos(oc);
        let mut s = 0.0f64;
        let mut n = 0u64;
        for &h in &hcs {
            for &o in &ocs {
                if h[0] == o[0] || h[0] == o[1] || h[1] == o[0] || h[1] == o[1] {
                    continue;
                }
                s += hu_equity_exact(h, o) as f64;
                n += 1;
            }
        }
        (s / n as f64) as f32
    }

    /// Validate against canonical (suit-independent) class equities + structure.
    /// The enumerator is exact; the gate catches gross bugs (evaluator, card
    /// encoding, win/lose swap), not third-decimal PPT drift.
    #[test]
    fn known_hu_equities() {
        let aa = PreflopClass::from_combo(c(12, 0) as Card, c(12, 1) as Card).index();
        let kk = PreflopClass::from_combo(c(11, 0) as Card, c(11, 1) as Card).index();
        let qq = PreflopClass::from_combo(c(10, 0) as Card, c(10, 1) as Card).index();
        let aks = PreflopClass::from_combo(c(12, 0) as Card, c(11, 0) as Card).index();
        let ako = PreflopClass::from_combo(c(12, 0) as Card, c(11, 1) as Card).index();
        let p22 = PreflopClass::from_combo(c(0, 0) as Card, c(0, 1) as Card).index();

        // AA vs KK class avg ≈ 81.9% (textbook); AKs vs QQ ≈ 46.2% (race, dog);
        // 22 vs AKo ≈ 52.5% (small pair edges two overcards).
        let e_aakk = class_avg_exact(aa, kk);
        assert!((e_aakk - 0.819).abs() < 0.004, "AA vs KK class avg = {e_aakk}");
        let e_aksqq = class_avg_exact(aks, qq);
        assert!((e_aksqq - 0.462).abs() < 0.006, "AKs vs QQ class avg = {e_aksqq}");
        let e_22ako = class_avg_exact(p22, ako);
        assert!((e_22ako - 0.525).abs() < 0.006, "22 vs AKo class avg = {e_22ako}");

        // Per-combo flush-blocker direction: shared-suit AA vs KK beats distinct.
        let e_sh = hu_equity_exact([c(12, 0), c(12, 1)], [c(11, 0), c(11, 1)]);
        let e_di = hu_equity_exact([c(12, 0), c(12, 1)], [c(11, 2), c(11, 3)]);
        assert!(e_sh > e_di, "shared {e_sh} should beat distinct {e_di}");
        // The class avg must lie between the suit extremes.
        assert!(e_di < e_aakk && e_aakk < e_sh, "{e_di} < {e_aakk} < {e_sh}");

        // Symmetry: equity(A,B) + equity(B,A) ≈ 1 exactly.
        let s = hu_equity_exact([c(12, 0), c(12, 1)], [c(11, 0), c(11, 1)])
            + hu_equity_exact([c(11, 0), c(11, 1)], [c(12, 0), c(12, 1)]);
        assert!((s - 1.0).abs() < 1e-4, "symmetry {s}");
    }

    /// The MC class-equity path (used to build the runtime table) must match the
    /// exact class average within sampling noise — validates the table builder.
    #[test]
    fn mc_matches_exact_class() {
        let aa = PreflopClass::from_combo(c(12, 0) as Card, c(12, 1) as Card).index();
        let kk = PreflopClass::from_combo(c(11, 0) as Card, c(11, 1) as Card).index();
        let aks = PreflopClass::from_combo(c(12, 0) as Card, c(11, 0) as Card).index();
        let qq = PreflopClass::from_combo(c(10, 0) as Card, c(10, 1) as Card).index();
        for &(h, o) in &[(aa, kk), (aks, qq)] {
            let exact = class_avg_exact(h, o);
            let mc = class_hu_equity_mc(h, o, 4000, 0xC0FFEE);
            assert!((mc - exact).abs() < 0.01, "MC {mc} vs exact {exact} ({h},{o})");
        }
    }
}
