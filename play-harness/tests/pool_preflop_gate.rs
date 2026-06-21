//! Validation gate for the NL10 pool preflop model (piece 2): aggregate
//! combos-weighted frequencies must land near the target stats —
//! PFR ≈ 18.3, VPIP ≈ 42.9, 3bet ≈ 8.8, steal (BTN/SB PFR) ≈ 35.
//!
//! Run: cargo test --release -p play-harness --test pool_preflop_gate -- --nocapture

use play_harness::pool_preflop::{PoolPreflop, PreAction, POS_BTN, POS_SB};
use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};

fn combos(c: usize) -> f32 {
    PreflopClass::new(c as u8).num_combos() as f32
}

#[test]
fn pool_preflop_frequencies_match_stats() {
    let pool = PoolPreflop::new();
    let total: f32 = (0..NUM_PREFLOP_CLASSES).map(combos).sum();
    let mut rng = 0xABCD_u64;

    // First-in frequencies averaged over the 5 opening positions (UTG..SB),
    // combos-weighted, sampled enough to average the edge randomization.
    let positions = [0usize, 1, 2, 3, 4];
    let mut pfr = 0.0f32; // raise
    let mut vpip = 0.0f32; // raise or limp
    for &pos in &positions {
        for c in 0..NUM_PREFLOP_CLASSES {
            let w = combos(c) / total;
            let (mut nr, mut nv) = (0.0f32, 0.0f32);
            let n = 40;
            for _ in 0..n {
                match pool.first_in(pos, c, &mut rng) {
                    PreAction::Raise => { nr += 1.0; nv += 1.0; }
                    PreAction::Limp => { nv += 1.0; }
                    _ => {}
                }
            }
            pfr += w * nr / n as f32;
            vpip += w * nv / n as f32;
        }
    }
    pfr /= positions.len() as f32;
    vpip /= positions.len() as f32;

    // Steal = BTN/SB open-raise frequency.
    let mut steal = 0.0f32;
    for &pos in &[POS_BTN, POS_SB] {
        for c in 0..NUM_PREFLOP_CLASSES {
            let w = combos(c) / total;
            let n = 40;
            let mut nr = 0.0;
            for _ in 0..n {
                if pool.first_in(pos, c, &mut rng) == PreAction::Raise { nr += 1.0; }
            }
            steal += w * nr / n as f32;
        }
    }
    steal /= 2.0;

    // 3bet frequency facing a raise.
    let mut tb = 0.0f32;
    for c in 0..NUM_PREFLOP_CLASSES {
        let w = combos(c) / total;
        let n = 40;
        let mut nr = 0.0;
        for _ in 0..n {
            if pool.facing_raise(2, c, &mut rng) == PreAction::Raise { nr += 1.0; }
        }
        tb += w * nr / n as f32;
    }

    eprintln!("PFR  {:.1}%  (target 18.3)", pfr * 100.0);
    eprintln!("VPIP {:.1}%  (target 42.9)", vpip * 100.0);
    eprintln!("steal {:.1}% (target 35.3)", steal * 100.0);
    eprintln!("3bet {:.1}%  (target 8.8)", tb * 100.0);

    // Approximate loose-passive pool — bounds reflect "in the right ballpark",
    // not an exact stat match (the model is a behavioral proxy).
    assert!((0.14..0.26).contains(&pfr), "PFR off: {pfr}");
    assert!((0.38..0.51).contains(&vpip), "VPIP off: {vpip}");
    assert!((0.28..0.42).contains(&steal), "steal off: {steal}");
    assert!((0.055..0.13).contains(&tb), "3bet off: {tb}");
    // The defining loose-PASSIVE signature: a big VPIP−PFR gap (lots of limp/flat).
    assert!(vpip - pfr > 0.18, "pool not passive enough (VPIP−PFR={:.2})", vpip - pfr);
    eprintln!("✓ pool preflop frequencies in range");
}
