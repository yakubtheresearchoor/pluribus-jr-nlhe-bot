//! B4 collapse gate: `bucketed_showdown_cfv_design1_collapsed` vs the
//! enumerated Design-1 arm.
//!
//! The collapse is CONTROL FLOW, not approximation: same B^K tuple
//! enumeration with pairwise conflict fractions, with the 3^K_active
//! relation enumeration replaced by a per-level tie-count DP
//! (linearity of expectation over pot levels). It IS Design 1
//! semantically, so it inherits the Design-1 standard — with one
//! float-reality refinement, stated openly rather than negotiated:
//!
//!   - BIT-EXACT wherever the computation graphs coincide: at
//!     singletons (B = nh) every probability is binary, the DP is
//!     point-mass selection, every extra multiply is ×1.0 and every
//!     add is 0.0 + x — so the collapsed arm must match the enumerated
//!     arm AND the exact CPU evaluator to the bit, including through
//!     the full identity walk (gated in the identity-gate test file).
//!   - At general B the same mathematical quantity is summed in a
//!     different order; float addition is not associative, so bit
//!     equality between the two orders is unachievable BY ANY
//!     implementation. The gate instead PINS the measured drift at
//!     ulp scale; drift beyond ulp scale is the bug the gate hunts.
//!
//! ═══ MEASURED 2026-06-10 (16-card/120-hand fixture, K=5) ═══
//!   singletons (10-hand sub-fixture, 5 arm-2 scenarios × 6 seats):
//!     bit-exact ✓ everywhere. Full-walk identity gate with the
//!     collapsed terminal: bit-exact ✓ (identity-gate test file).
//!   general B ∈ {3, 5, 8}, distance from the f64 reference:
//!     enumerated arm  1.9e-7 .. 3.9e-5   (longer f32 chains)
//!     collapsed arm   1.4e-7 .. 3.6e-6   (CLOSER to the true value —
//!       fewer f32 ops; the apparent 1.1e-5 enum↔collapsed "drift" was
//!       the enumerated arm's own rounding)
//!   Both at f32 accumulated-rounding distance from one f64 quantity →
//!   same computation, different order. Bug line 1e-4.

use solver_core::abstraction::postflop_buckets::compute_wtl_for_runout;
use solver_core::card::Card;
use solver_core::solver::bucketed_showdown::{
    bucketed_showdown_cfv, bucketed_showdown_cfv_design1_collapsed, BucketedRunoutTables,
};

const NP: u8 = 6;
const K: usize = 5;

// ── Singleton sub-fixture: 10 hands over 12+ cards (physically
// realizable for 6 disjoint hands), unit weights, singleton buckets —
// binary fractions, graphs coincide, bit-exact required. ──

fn singleton_fixture() -> (Vec<(Card, Card)>, Vec<i32>) {
    let hands: Vec<(Card, Card)> = vec![
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (0, 2),
        (1, 3),
        (4, 6),
    ];
    let strengths: Vec<i32> = vec![10, 20, 20, 5, 30, 20, 30, 7, 30, 12];
    (hands, strengths)
}

fn singleton_tables() -> BucketedRunoutTables {
    let (hands, strengths) = singleton_fixture();
    let nh = hands.len();
    let map: Vec<u16> = (0..nh as u16).collect();
    let wtl = compute_wtl_for_runout(&hands, &strengths, &vec![1.0f64; nh], &map, nh);
    BucketedRunoutTables::from_wtl(&wtl, &vec![1.0f64; nh])
}

fn make_reaches(num_opp: usize, nb: usize, salt: u32) -> Vec<Vec<f32>> {
    (0..num_opp)
        .map(|oi| {
            (0..nb)
                .map(|b| {
                    let v = (oi as u32 * 31 + b as u32 * 17 + salt * 7) % 11;
                    if v == 0 { 0.0 } else { v as f32 / 16.0 }
                })
                .collect()
        })
        .collect()
}

/// Arm-2 scenarios (the collapse only changes arm 2; arm 1 is shared
/// code and bit-identical by construction).
fn arm2_scenarios() -> Vec<(&'static str, [i32; 6], u16)> {
    vec![
        ("unequal/no-folds", [10, 25, 40, 40, 25, 10], 0),
        ("folds+sidepots", [20, 35, 35, 5, 20, 35], (1 << 3) | (1 << 4)),
        ("equal+fold", [20, 20, 20, 20, 20, 20], 1 << 2),
        ("lone survivor", [8, 8, 30, 8, 8, 16], 0b011111),
        ("folded traverser seat0", [20, 35, 35, 5, 20, 35], (1 << 0) | (1 << 3)),
    ]
}

#[test]
fn collapse_singleton_bit_exact() {
    let tables = singleton_tables();
    let nb = tables.nb;
    for (name, contribs, fold_mask) in arm2_scenarios() {
        for traverser in 0..NP as usize {
            let reaches = make_reaches(K, nb, 3 + traverser as u32);
            let views: Vec<&[f32]> = reaches.iter().map(|v| v.as_slice()).collect();
            let enumerated = bucketed_showdown_cfv(
                &views, &tables, &contribs, fold_mask, traverser, NP, 30, 0.05, 3.0, true,
            );
            let collapsed = bucketed_showdown_cfv_design1_collapsed(
                &views, &tables, &contribs, fold_mask, traverser, NP, 30, 0.05, 3.0, true,
            );
            for b in 0..nb {
                assert_eq!(
                    collapsed[b].to_bits(),
                    enumerated[b].to_bits(),
                    "{name} trav={traverser} bucket {b}: collapsed {} vs enumerated {} — \
                     at singletons the graphs coincide; any drift is a bug",
                    collapsed[b],
                    enumerated[b],
                );
            }
        }
    }
    eprintln!("collapse singleton gate PASSED: bit-exact across {} scenarios × 6 traversers",
        arm2_scenarios().len());
}

// ── General-B drift: 16-card/120-hand fixture (same as the Design-2
// parity fixture), quantile buckets — fractions interior, orders
// differ, drift must be ulp-scale. ──

fn general_fixture(nb: usize) -> (BucketedRunoutTables, Vec<Vec<f32>>) {
    let mut hands = Vec::new();
    for a in 0u8..16 {
        for b in (a + 1)..16 {
            hands.push((a, b));
        }
    }
    let strengths: Vec<i32> = hands
        .iter()
        .map(|&(a, b)| ((a as i32 * 7 + b as i32 * 13) % 11) + 1)
        .collect();
    let nh = hands.len();
    let mut order: Vec<usize> = (0..nh).collect();
    order.sort_by_key(|&i| strengths[i]);
    let mut map = vec![0u16; nh];
    for (pos, &h) in order.iter().enumerate() {
        map[h] = ((pos * nb) / nh) as u16;
    }
    let wtl = compute_wtl_for_runout(&hands, &strengths, &vec![1.0f64; nh], &map, nb);
    let mut sums = vec![0.0f64; nb];
    for h in 0..nh {
        sums[map[h] as usize] += 1.0;
    }
    let tables = BucketedRunoutTables::from_wtl(&wtl, &sums);
    let reach = make_reaches(K, nb, 5);
    (tables, reach)
}

/// f64 reference for arm 2: same tuple × relation enumeration as the
/// f32 enumerated arm, all arithmetic in f64. Both f32 arms must sit
/// at f32 accumulated-rounding distance from THIS — proving they
/// compute the same quantity and that their mutual drift is rounding.
#[allow(clippy::too_many_arguments)]
fn arm2_reference_f64(
    bucket_reach: &[&[f32]],
    tables: &BucketedRunoutTables,
    contributions: &[i32],
    fold_mask: u16,
    traverser: usize,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
) -> Vec<f64> {
    let nb = tables.nb;
    let np = NP as usize;
    let num_opp = np - 1;
    let c_t = contributions[traverser];
    let mut levels: Vec<i32> = (0..np).map(|p| contributions[p]).collect();
    levels.sort();
    levels.dedup();
    let main_pot_amount: i32 = {
        let nmc = (0..np).filter(|&p| contributions[p] >= levels[0]).count();
        levels[0] * nmc as i32 + starting_pot
    };
    let main_pot_rake =
        (main_pot_amount as f64 * rake_rate as f64).min(rake_cap as f64).max(0.0);
    let traverser_stake = starting_pot as f64 / np as f64 + c_t as f64;
    let traverser_folded = fold_mask & (1u16 << traverser) != 0;
    let opp_player: Vec<usize> =
        (0..num_opp).map(|oi| if oi < traverser { oi } else { oi + 1 }).collect();
    let opp_contrib: Vec<i32> = opp_player.iter().map(|&p| contributions[p]).collect();
    let opp_folded: Vec<bool> =
        opp_player.iter().map(|&p| fold_mask & (1u16 << p) != 0).collect();

    let net = |rel: &[u8]| -> f64 {
        let mut cash = 0.0f64;
        let mut prev_l = 0i32;
        for (li, &lev) in levels.iter().enumerate() {
            let pc = lev - prev_l;
            let nc = (0..np).filter(|&p| contributions[p] >= lev).count();
            let mut pot_l = (pc * nc as i32) as f64;
            if li == 0 {
                pot_l += starting_pot as f64;
            }
            if pot_l == 0.0 {
                prev_l = lev;
                continue;
            }
            let trav_elig = !traverser_folded && c_t >= lev;
            let mut elig: u32 = trav_elig as u32;
            let mut beats = false;
            for oi in 0..num_opp {
                if opp_folded[oi] || opp_contrib[oi] < lev {
                    continue;
                }
                elig += 1;
                if rel[oi] == 2 {
                    beats = true;
                }
            }
            if elig == 0 {
                if contributions[traverser] >= lev {
                    cash += pc as f64
                        + if li == 0 { starting_pot as f64 / np as f64 } else { 0.0 };
                }
                prev_l = lev;
                continue;
            }
            if !trav_elig {
                prev_l = lev;
                continue;
            }
            if !beats {
                let mut tied = 1u32;
                for oi in 0..num_opp {
                    if opp_folded[oi] || opp_contrib[oi] < lev {
                        continue;
                    }
                    if rel[oi] == 1 {
                        tied += 1;
                    }
                }
                let par = if li == 0 { pot_l - main_pot_rake } else { pot_l };
                cash += par / tied as f64;
            }
            prev_l = lev;
        }
        cash - traverser_stake
    };

    #[allow(clippy::too_many_arguments)]
    fn rec(
        oi: usize,
        num_opp: usize,
        nb: usize,
        bt: usize,
        w: f64,
        prefix: &mut Vec<usize>,
        rel: &mut [u8],
        bucket_reach: &[&[f32]],
        tables: &BucketedRunoutTables,
        opp_folded: &[bool],
        net: &dyn Fn(&[u8]) -> f64,
        accum: &mut f64,
    ) {
        if oi == num_opp {
            *accum += w * net(rel);
            return;
        }
        for bo in 0..nb {
            let r = bucket_reach[oi][bo] as f64;
            if r == 0.0 {
                continue;
            }
            let mut base = w * r;
            let mut blocked = false;
            for &pb in prefix.iter() {
                let f = tables.f_n[pb * nb + bo] as f64;
                if f == 0.0 {
                    blocked = true;
                    break;
                }
                base *= f;
            }
            if blocked {
                continue;
            }
            let i = bt * nb + bo;
            prefix.push(bo);
            if opp_folded[oi] {
                let n = tables.f_n[i] as f64;
                if n != 0.0 {
                    rel[oi] = 3;
                    rec(oi + 1, num_opp, nb, bt, base * n, prefix, rel, bucket_reach,
                        tables, opp_folded, net, accum);
                }
            } else {
                for (code, f) in
                    [(0u8, tables.f_w[i]), (1, tables.f_t[i]), (2, tables.f_l[i])]
                {
                    if f == 0.0 {
                        continue;
                    }
                    rel[oi] = code;
                    rec(oi + 1, num_opp, nb, bt, base * f as f64, prefix, rel,
                        bucket_reach, tables, opp_folded, net, accum);
                }
            }
            prefix.pop();
        }
    }

    let mut cfv = vec![0.0f64; nb];
    let mut rel = vec![0u8; num_opp];
    for (bt, slot) in cfv.iter_mut().enumerate() {
        let mut accum = 0.0f64;
        rec(0, num_opp, nb, bt, 1.0, &mut Vec::new(), &mut rel, bucket_reach, tables,
            &opp_folded, &net, &mut accum);
        *slot = accum;
    }
    cfv
}

#[test]
fn collapse_general_b_drift_pinned() {
    for nb in [3usize, 5, 8] {
        let (tables, reach) = general_fixture(nb);
        let views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
        for (name, contribs, fold_mask) in arm2_scenarios() {
            let enumerated = bucketed_showdown_cfv(
                &views, &tables, &contribs, fold_mask, 0, NP, 30, 0.05, 3.0, true,
            );
            let collapsed = bucketed_showdown_cfv_design1_collapsed(
                &views, &tables, &contribs, fold_mask, 0, NP, 30, 0.05, 3.0, true,
            );
            let reference =
                arm2_reference_f64(&views, &tables, &contribs, fold_mask, 0, 30, 0.05, 3.0);
            let scale = reference.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            assert!(scale > 0.0);
            let dist = |xs: &[f32]| -> f64 {
                xs.iter()
                    .zip(&reference)
                    .map(|(a, r)| (*a as f64 - r).abs() / scale)
                    .fold(0.0, f64::max)
            };
            let de = dist(&enumerated);
            let dc = dist(&collapsed);
            eprintln!("B={nb} {name}: |enum−f64ref| {de:.2e}  |collapsed−f64ref| {dc:.2e}");
            // Same-quantity proof: BOTH f32 arms at f32 accumulated-
            // rounding distance from the one f64 quantity. Reorder spans
            // ~B^K×3^K ≈ 10⁴-10⁶ terms; √N·ulp ≈ 5e-5 at B=5. Bug line
            // 1e-4 (set from that arithmetic, then checked against the
            // measured values printed above).
            for (arm, d) in [("enumerated", de), ("collapsed", dc)] {
                assert!(
                    d < 1e-4,
                    "B={nb} {name}: {arm} arm is {d:.2e} from the f64 reference — \
                     beyond accumulated f32 rounding; same-quantity violation \
                     (a real bug, not float reordering)"
                );
            }
        }
    }
}
