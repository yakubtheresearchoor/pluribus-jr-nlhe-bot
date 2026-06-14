//! Harness component 2+3 gate: the variance reducer is an instrument, so
//! anchor it against ground truth before trusting it. The RAW estimator
//! (one seating per sample, no card-luck cancellation) is unbiased by
//! construction; the REDUCED estimator (mirrored duplicate, common cards +
//! board) must agree on the MEAN — a reducer that shifts the mean gives
//! tight bars around the wrong number, worse than none — while delivering a
//! TIGHTER standard error.
//!
//! Two comparisons, because the reducer cancels CARD luck specifically:
//!   (A) Blueprint vs Blueprint — true edge 0, variance is pure card luck
//!       (the regime the duplicate targets, and the regime the real B8-vs-
//!       reach-weighted experiment lives in). Tests tightening: reduced σ
//!       must be well below raw σ. Mean must stay ~0 (unbiased).
//!   (B) Blueprint vs AlwaysAggressive — a large NON-ZERO edge with strategic
//!       (not card) variance. Tests that the reducer doesn't shift a
//!       non-zero mean. (It won't tighten much here — the variance isn't
//!       card luck — which is the point of testing both.)
//! Needs a valid artifact (BP_ARTIFACT).

use play_harness::blueprint::{build_oracle_tree, Blueprint};
use play_harness::experiment::{ab_duplicate, mbb_per_100};
use play_harness::match_play::{MatchEnv, Policy};

fn env_or_skip() -> Option<(Blueprint, solver_core::tree::flat::FlatTree)> {
    let path = std::env::var("BP_ARTIFACT")
        .unwrap_or_else(|_| "../blueprint_out_b10_4x4/flop_0000.bp".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: no banked artifact at {path}");
        return None;
    }
    Some((Blueprint::load(&path).unwrap(), build_oracle_tree()))
}

#[test]
fn variance_reducer_unbiased_and_tighter() {
    let Some((bp, tree)) = env_or_skip() else { return };
    let env = MatchEnv::new(&bp, &tree);
    let n = 20_000;

    // (A) CARD-LUCK regime: Blueprint vs Blueprint (true edge 0).
    let mut rng = 0xC0FFEE_u64;
    let a = ab_duplicate(&env, &Policy::Blueprint(&bp), &Policy::Blueprint(&bp), n, &mut rng);
    let (arm, ars) = (a.raw.mean(), a.raw.stderr());
    let (adm, ads) = (a.reduced.mean(), a.reduced.stderr());
    eprintln!("(A) Blueprint vs Blueprint, {n} decks (true edge 0):");
    eprintln!("    RAW     {arm:+.4} ± {ars:.4} | REDUCED {adm:+.4} ± {ads:.4} chips/hand");
    eprintln!("    variance-reduction (raw σ / reduced σ): {:.2}×", ars / ads);

    // (B) NON-ZERO-edge unbiasedness: Blueprint vs AlwaysAggressive.
    let mut rng = 0x1234_u64;
    let b = ab_duplicate(&env, &Policy::Blueprint(&bp), &Policy::AlwaysAggressive, n, &mut rng);
    let (brm, brs) = (b.raw.mean(), b.raw.stderr());
    let (bdm, bds) = (b.reduced.mean(), b.reduced.stderr());
    eprintln!("(B) Blueprint vs AlwaysAggressive, {n} decks:");
    eprintln!("    RAW     {brm:+.4} ± {brs:.4} | REDUCED {bdm:+.4} ± {bds:.4} chips/hand = {:+.0} mbb/100",
        mbb_per_100(bdm, 2.0));
    eprintln!("    stratified by live-count (the family key):");
    for (live, s) in &b.by_live {
        eprintln!("      live-{live}: {:+.3} ± {:.3} ({} hands)", s.mean(), s.stderr(), s.n);
    }

    // (A) UNBIASED at 0: reduced mean within a few σ of zero.
    assert!(adm.abs() < 4.0 * ads.max(1e-9),
        "(A) reduced mean {adm:+.4} not ~0 (± {ads:.4}) — reducer biased at zero edge");
    // (A) TIGHTER: card-luck cancellation gives a real reduction.
    assert!(ads < ars * 0.7,
        "(A) reduced σ {ads:.4} not meaningfully below raw σ {ars:.4} — card-luck cancellation failed");
    // (B) UNBIASED at non-zero: reduced mean agrees with raw mean.
    let comb = (brs * brs + bds * bds).sqrt();
    assert!((brm - bdm).abs() < 4.0 * comb,
        "(B) reducer SHIFTS a non-zero mean: raw {brm:+.4} vs reduced {bdm:+.4} (± {comb:.4})");
    eprintln!("GATE PASS: reducer unbiased at 0 and non-0; card-luck variance reduced {:.2}×.", ars / ads);
}
