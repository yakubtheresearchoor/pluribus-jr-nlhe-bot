//! np = 3 SCOPE-EXTENSION GATE for the bucketed terminal (2026-06-12,
//! v1 live-3 seam cells — the most common multiway family, unpriceable
//! on the exact path: its np=3 arms are O(nh³) tie-aware brute force
//! per terminal, measured 95+ min for ONE cell at nh=1176).
//!
//! Claim being gated: VALUE-correctness of the Design-1 bucketed
//! terminal at np = 3 (the arms are num_opp-generic; what's np ≥ 4
//! specific was only the bit-exact dispatch-mirror claim — the exact
//! function's np=3 arms use different float-op ORDER, so this gate is
//! tolerance-based against the exact evaluator at SINGLETON bucketing,
//! the B3 fixture):
//!   1. bucketed(singletons) ≈ exact, rel ≤ 1e-5 per entry, every
//!      traverser, across the np=3 scenario surface (3-active equal
//!      raked/unraked/checkdown; folds + side pots; folded traverser;
//!      lone survivor; all-in ladder; no-flop-no-drop);
//!   2. Design1Brute ↔ Design1Collapsed BIT-exact at np=3 (collapse is
//!      control flow, not approximation — same claim as np ≥ 4);
//!   3. zero-sum: Σ over traversers of Σ_h cfv (singleton, uniform
//!      reach) = 0 within f32 accumulation noise, unraked.

use solver_core::abstraction::postflop_buckets::compute_wtl_for_runout;
use solver_core::card::Card;
use solver_core::solver::bucketed_showdown::{
    bucketed_showdown_cfv, bucketed_showdown_cfv_design1_collapsed, BucketedRunoutTables,
};
use solver_core::solver::showdown::side_pot_showdown_cfv_with_rake;

fn hand_universe() -> (Vec<(Card, Card)>, Vec<i32>) {
    let hands: Vec<(Card, Card)> = vec![
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (1, 8),
    ];
    let strengths: Vec<i32> = vec![10, 20, 20, 5, 30, 20, 30, 7, 30, 12];
    (hands, strengths)
}

struct Fixture {
    nh: usize,
    hand_cards: Vec<u8>,
    sorted_str: Vec<u16>,
    sorted_idx: Vec<u16>,
    tables: BucketedRunoutTables,
}

fn build_fixture() -> Fixture {
    let (hands, strengths) = hand_universe();
    let nh = hands.len();
    let bucket_map: Vec<u16> = (0..nh as u16).collect();
    let weights = vec![1.0f64; nh];
    let wtl = compute_wtl_for_runout(&hands, &strengths, &weights, &bucket_map, nh);
    let tables = BucketedRunoutTables::from_wtl(&wtl, &vec![1.0f64; nh]);
    let hand_cards: Vec<u8> = hands.iter().flat_map(|&(a, b)| [a, b]).collect();
    let mut order: Vec<usize> = (0..nh).collect();
    order.sort_by_key(|&i| strengths[i]);
    let sorted_str: Vec<u16> = order.iter().map(|&i| strengths[i] as u16).collect();
    let sorted_idx: Vec<u16> = order.iter().map(|&i| i as u16).collect();
    Fixture { nh, hand_cards, sorted_str, sorted_idx, tables }
}

fn make_reaches(num_opp: usize, nh: usize, salt: u32) -> Vec<Vec<f32>> {
    (0..num_opp)
        .map(|oi| {
            (0..nh)
                .map(|h| {
                    let v = (oi as u32 * 31 + h as u32 * 17 + salt * 7) % 11;
                    if v == 0 {
                        0.0
                    } else {
                        v as f32 / 16.0
                    }
                })
                .collect()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn assert_np3_scenario(
    fx: &Fixture,
    contributions: &[i32; 3],
    fold_mask: u16,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    flop_seen: bool,
    salt: u32,
    label: &str,
) {
    const REL_TOL: f64 = 1e-5;
    for traverser in 0..3usize {
        let reaches = make_reaches(2, fx.nh, salt + traverser as u32);
        let views: Vec<&[f32]> = reaches.iter().map(|v| v.as_slice()).collect();

        let exact = side_pot_showdown_cfv_with_rake(
            &views,
            &fx.hand_cards,
            fx.nh,
            &fx.sorted_str,
            &fx.sorted_idx,
            &fx.sorted_str,
            &fx.sorted_idx,
            contributions,
            fold_mask,
            traverser,
            3,
            starting_pot,
            rake_rate,
            rake_cap,
            flop_seen,
        );
        let brute = bucketed_showdown_cfv(
            &views,
            &fx.tables,
            contributions,
            fold_mask,
            traverser,
            3,
            starting_pot,
            rake_rate,
            rake_cap,
            flop_seen,
        );
        let collapsed = bucketed_showdown_cfv_design1_collapsed(
            &views,
            &fx.tables,
            contributions,
            fold_mask,
            traverser,
            3,
            starting_pot,
            rake_rate,
            rake_cap,
            flop_seen,
        );

        let scale = exact.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-30) as f64;
        for h in 0..fx.nh {
            // 2. collapsed ≡ brute, bit-exact (control-flow claim).
            assert_eq!(
                collapsed[h].to_bits(),
                brute[h].to_bits(),
                "{label}: t{traverser} h{h}: collapsed {} vs brute {} — the \
                 collapse must stay bit-exact at np=3",
                collapsed[h],
                brute[h]
            );
            // 1. brute ≈ exact (value claim, order-tolerant).
            let d = (brute[h] as f64 - exact[h] as f64).abs() / scale;
            assert!(
                d <= REL_TOL,
                "{label}: t{traverser} h{h}: bucketed {} vs exact {} (rel {d:.2e} \
                 > {REL_TOL:.0e}) — np=3 VALUE disagreement, not op-order noise",
                brute[h],
                exact[h]
            );
        }
    }
    eprintln!("  np3 scenario OK: {label}");
}

#[test]
fn np3_bucketed_terminal_gate() {
    let fx = build_fixture();
    // 3-active showdowns (the O(nh^3) exact arm this replaces).
    assert_np3_scenario(&fx, &[20; 3], 0, 12, 0.05, 3.0, true, 1, "equal raked");
    assert_np3_scenario(&fx, &[20; 3], 0, 12, 0.0, 0.0, true, 2, "equal unraked");
    assert_np3_scenario(&fx, &[0; 3], 0, 18, 0.05, 3.0, true, 3, "checkdown");
    // Per-level arm: folds, side pots, folded traverser, survivors.
    assert_np3_scenario(&fx, &[20, 35, 5], 0, 12, 0.05, 3.0, true, 4, "allin ladder");
    assert_np3_scenario(&fx, &[20, 20, 20], 1 << 1, 12, 0.05, 3.0, true, 5, "equal + fold");
    assert_np3_scenario(&fx, &[20, 35, 35], 1 << 0, 12, 0.05, 3.0, true, 7, "folded traverser seat0");
    assert_np3_scenario(&fx, &[20, 35, 5], 0, 12, 0.05, 3.0, false, 8, "no flop no drop");

    // ═══ FINDING (this gate's first run, 2026-06-12): the exact
    // function is NOT a valid reference for np=3 fold terminals where
    // a FOLDED player out-contributed the max active player. At np=3
    // those terminals take the constant-payoff fast path (gated
    // num_opp ≤ 2) whose own header documents per-player-WRONG values
    // (zero-sum-correct only): the folded player's uncalled excess is
    // not refunded per-player. The BUCKETED per-level arm computes the
    // refund correctly — measured: t1 folded commit 30, survivor 16,
    // pot 12 ⇒ correct unit loss = 12/3 + 30 − (30−16) = 20; bucketed
    // = −20·cfreach exactly; exact fast path = −34·cfreach. So for
    // this scenario class the reference is HAND-COMPUTED (per-unit
    // payoff × brute-force cfreach), not the exact arm. At np ≥ 4 the
    // exact falls through to the per-level evaluator and is correct —
    // np=3 consumers of per-hand CFVs (best response, harness EV)
    // should prefer the bucketed terminal for fold terminals too.
    {
        let (hands, _) = hand_universe();
        let contributions = [8i32, 30, 16];
        let fold_mask: u16 = 0b011; // seats 0,1 folded; survivor seat 2
        let starting_pot = 12;
        for traverser in 0..3usize {
            let reaches = make_reaches(2, fx.nh, 6 + traverser as u32);
            let views: Vec<&[f32]> = reaches.iter().map(|v| v.as_slice()).collect();
            let brute = bucketed_showdown_cfv(
                &views, &fx.tables, &contributions, fold_mask, traverser, 3,
                starting_pot, 0.0, 0.0, true,
            );
            // Hand-computed per-unit payoff (unraked):
            //   investment_t = pot/3 + c_t; refund_t (folded) =
            //   max(0, c_t − max active commit); survivor wins
            //   total_pot − refunds − own investment.
            let max_active = 16.0f32;
            let inv: Vec<f32> =
                (0..3).map(|p| starting_pot as f32 / 3.0 + contributions[p] as f32).collect();
            let refund: Vec<f32> = (0..3)
                .map(|p| {
                    if fold_mask & (1 << p) != 0 {
                        (contributions[p] as f32 - max_active).max(0.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            let total_pot = starting_pot as f32 + 8.0 + 30.0 + 16.0;
            let unit = if fold_mask & (1 << traverser) != 0 {
                -(inv[traverser] - refund[traverser])
            } else {
                total_pot - refund.iter().sum::<f32>() - inv[traverser]
            };
            // cfreach by brute force over compatible opponent pairs.
            for h in 0..fx.nh {
                let (hc1, hc2) = hands[h];
                let mut cfreach = 0.0f32;
                for g0 in 0..fx.nh {
                    let (a, b) = hands[g0];
                    if a == hc1 || a == hc2 || b == hc1 || b == hc2 {
                        continue;
                    }
                    if reaches[0][g0] == 0.0 {
                        continue;
                    }
                    for g1 in 0..fx.nh {
                        let (c, d) = hands[g1];
                        if c == hc1 || c == hc2 || d == hc1 || d == hc2 {
                            continue;
                        }
                        if c == a || c == b || d == a || d == b {
                            continue;
                        }
                        cfreach += reaches[0][g0] * reaches[1][g1];
                    }
                }
                let want = unit * cfreach;
                let d = (brute[h] - want).abs() / want.abs().max(1e-6);
                assert!(
                    d <= 1e-5,
                    "lone-survivor hand-computed: t{traverser} h{h}: bucketed {} \
                     vs hand {} (rel {d:.2e})",
                    brute[h],
                    want
                );
            }
        }
        eprintln!("  np3 scenario OK: lone survivor w/ uncalled refund (HAND-COMPUTED ref)");
    }

    // 3. Zero-sum (unraked, uniform reach): Σ_t Σ_h cfv ≈ 0.
    let uniform: Vec<Vec<f32>> = vec![vec![1.0; fx.nh]; 2];
    let views: Vec<&[f32]> = uniform.iter().map(|v| v.as_slice()).collect();
    let mut total = 0.0f64;
    let mut scale = 0.0f64;
    for t in 0..3usize {
        let cfv = bucketed_showdown_cfv(
            &views, &fx.tables, &[20; 3], 0, t, 3, 12, 0.0, 0.0, true,
        );
        for v in cfv {
            total += v as f64;
            scale += (v as f64).abs();
        }
    }
    assert!(
        total.abs() <= 1e-5 * scale.max(1.0),
        "np=3 zero-sum violated: Σ {total:.3e} (scale {scale:.3e})"
    );
    eprintln!("np3 gate PASSED: value vs exact, collapse bit-exact, zero-sum");
}
