//! B3 step 3 gate: the Design-1 bucketed terminal at B = nh (singleton
//! buckets, unit weights) must reproduce `side_pot_showdown_cfv_with_rake`
//! BIT-EXACTLY (f32::to_bits equality), through the same code path that
//! B < nh will use — no tolerance, per the phase directive ("if it comes
//! back merely close, that's a finding to chase, not a fallback to
//! accept").
//!
//! This runs BEFORE any walk integration (the walk gate is step 4).
//!
//! Scope: np ≥ 4 — the only regime the bucketed terminal serves (the two
//! dispatch arms that carry the fifth power). Scenarios cover:
//!   - arm 1: equal contributions, no folds (recurse_eq mirror), with
//!     and without rake, every traverser seat;
//!   - arm 2: folds + unequal contributions (per-level mirror), folded
//!     traverser, lone-survivor-by-folds, all-in unequal-no-fold,
//!     flop_seen = false, every traverser seat.

use solver_core::abstraction::postflop_buckets::compute_wtl_for_runout;
use solver_core::card::Card;
use solver_core::solver::bucketed_showdown::{bucketed_showdown_cfv, BucketedRunoutTables};
use solver_core::solver::showdown::side_pot_showdown_cfv_with_rake;

/// Small synthetic hand universe with real card-conflict structure and
/// strength ties. Cards are raw u8 ids; conflicts are card-identity,
/// which is all either implementation reads.
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
    // Deliberate ties (20, 30 repeated) to exercise tie-count payoffs.
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

    // Singleton bucketing: bucket i = hand i, unit weights.
    let bucket_map: Vec<u16> = (0..nh as u16).collect();
    let weights = vec![1.0f64; nh];
    let wtl = compute_wtl_for_runout(&hands, &strengths, &weights, &bucket_map, nh);
    let tables = BucketedRunoutTables::from_wtl(&wtl, &vec![1.0f64; nh]);

    // Identity-degeneracy precondition: every fraction must be EXACTLY
    // 0.0 or 1.0, and must agree with a direct pairwise recomputation.
    // If this fails, the bit-exact comparison below is meaningless.
    for bt in 0..nh {
        for bo in 0..nh {
            let i = bt * nh + bo;
            let (h1, h2) = hands[bt];
            let (g1, g2) = hands[bo];
            let conflict =
                bt == bo || g1 == h1 || g1 == h2 || g2 == h1 || g2 == h2;
            let (ew, et, el, en) = if conflict {
                (0.0, 0.0, 0.0, 0.0)
            } else if strengths[bt] > strengths[bo] {
                (1.0, 0.0, 0.0, 1.0)
            } else if strengths[bt] == strengths[bo] {
                (0.0, 1.0, 0.0, 1.0)
            } else {
                (0.0, 0.0, 1.0, 1.0)
            };
            assert_eq!(tables.f_w[i].to_bits(), (ew as f32).to_bits(), "f_w[{bt}][{bo}]");
            assert_eq!(tables.f_t[i].to_bits(), (et as f32).to_bits(), "f_t[{bt}][{bo}]");
            assert_eq!(tables.f_l[i].to_bits(), (el as f32).to_bits(), "f_l[{bt}][{bo}]");
            assert_eq!(tables.f_n[i].to_bits(), (en as f32).to_bits(), "f_n[{bt}][{bo}]");
        }
    }

    let hand_cards: Vec<u8> = hands.iter().flat_map(|&(a, b)| [a, b]).collect();
    let mut order: Vec<usize> = (0..nh).collect();
    order.sort_by_key(|&i| strengths[i]);
    let sorted_str: Vec<u16> = order.iter().map(|&i| strengths[i] as u16).collect();
    let sorted_idx: Vec<u16> = order.iter().map(|&i| i as u16).collect();

    Fixture { nh, hand_cards, sorted_str, sorted_idx, tables }
}

/// Deterministic reach values with exact zeros sprinkled in (zeros
/// exercise the skip-on-zero ⇔ branch-skip equivalence).
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
fn assert_bit_exact(
    fx: &Fixture,
    contributions: &[i32],
    fold_mask: u16,
    num_players: u8,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    flop_seen: bool,
    salt: u32,
    label: &str,
) {
    let np = num_players as usize;
    let num_opp = np - 1;
    for traverser in 0..np {
        let reaches = make_reaches(num_opp, fx.nh, salt + traverser as u32);
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
            num_players,
            starting_pot,
            rake_rate,
            rake_cap,
            flop_seen,
        );

        // Singleton bucket reaches ARE the hand reaches (bucket i = hand i).
        let bucketed = bucketed_showdown_cfv(
            &views,
            &fx.tables,
            contributions,
            fold_mask,
            traverser,
            num_players,
            starting_pot,
            rake_rate,
            rake_cap,
            flop_seen,
        );

        assert_eq!(bucketed.len(), fx.nh);
        for h in 0..fx.nh {
            assert_eq!(
                bucketed[h].to_bits(),
                exact[h].to_bits(),
                "{label}: traverser {traverser}, hand/bucket {h}: \
                 bucketed {} vs exact {}",
                bucketed[h],
                exact[h],
            );
        }
    }
}

#[test]
fn arm1_equal_no_folds_bit_exact() {
    let fx = build_fixture();
    // 6-max, equal contributions, no folds → recurse_eq arm.
    assert_bit_exact(&fx, &[20; 6], 0, 6, 12, 0.05, 3.0, true, 1, "arm1 raked");
    assert_bit_exact(&fx, &[20; 6], 0, 6, 12, 0.0, 0.0, true, 2, "arm1 unraked");
    // Zero contributions check-down (starting pot only).
    assert_bit_exact(&fx, &[0; 6], 0, 6, 18, 0.05, 3.0, true, 3, "arm1 checkdown");
    // 5-player table too (np = 5, K = 4).
    assert_bit_exact(&fx, &[15; 5], 0, 5, 10, 0.05, 3.0, true, 4, "arm1 np5");
}

#[test]
fn arm2_folds_and_side_pots_bit_exact() {
    let fx = build_fixture();
    // Two folds, unequal contributions → per-level arm with side pots.
    assert_bit_exact(
        &fx,
        &[20, 35, 35, 5, 20, 35],
        (1 << 3) | (1 << 4),
        6,
        12,
        0.05,
        3.0,
        true,
        5,
        "arm2 folds+sidepots",
    );
    // Equal contributions WITH a fold (dead money path).
    assert_bit_exact(&fx, &[20; 6], 1 << 2, 6, 12, 0.05, 3.0, true, 6, "arm2 equal+fold");
    // Unequal contributions, no folds (all-in ladder) → per-level arm.
    assert_bit_exact(
        &fx,
        &[10, 25, 40, 40, 25, 10],
        0,
        6,
        12,
        0.05,
        3.0,
        true,
        7,
        "arm2 allin ladder",
    );
    // Lone survivor by folds (everyone but player 5 folded), including a
    // folded player who out-contributed the survivor (side-pot return).
    assert_bit_exact(
        &fx,
        &[8, 8, 30, 8, 8, 16],
        0b011111,
        6,
        12,
        0.05,
        3.0,
        true,
        8,
        "arm2 lone survivor",
    );
    // flop_seen = false: "no flop, no drop" gate.
    assert_bit_exact(
        &fx,
        &[20, 35, 35, 5, 20, 35],
        (1 << 3) | (1 << 4),
        6,
        12,
        0.05,
        3.0,
        false,
        9,
        "arm2 no-flop-no-drop",
    );
}
