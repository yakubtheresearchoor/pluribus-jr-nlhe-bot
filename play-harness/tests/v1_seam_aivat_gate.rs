//! AIVAT control-variate gate (step 1 of the docket). The runout control
//! variate (exact Rao-Blackwellization over turn+river) must, like the
//! duplicate reducer, be ANCHORED before trust: the AIVAT-corrected estimate
//! must match the RAW mean (it may not shift the number, only tighten bars),
//! and it must reduce variance. Gated on synthetic policies — no blueprint.
//!
//! DEFERRED (named): the real reduction FACTOR depends on how good the
//! blueprint is as a predictor of the residual ACTION variance, and the
//! blueprint doesn't exist yet. So this gates CORRECTNESS (unbiased +
//! reduces); the factor is measured at option 3 with the real value function.
//! Keep the hand-count plan able to absorb a weaker-than-hoped factor.

use play_harness::experiment::Stat;
use play_harness::v1_seam::{SeamGame, SeamPolicy};

fn flop() -> [u8; 3] { [47, 21, 0] }

#[test]
fn aivat_unbiased_and_reduces() {
    // AlwaysAggressive ⇒ showdowns ⇒ all variance is runout luck, which the
    // runout control variate removes exactly. Per-seat-hand edge of seat 0.
    let g = SeamGame::new(3, 2, 12, flop());
    let pols = vec![SeamPolicy::AlwaysAggressive; 3];
    let mut arng = 0xA1u64;
    let mut rrng = 0xB2u64;
    let mut raw = Stat::default();
    let mut aiv = Stat::default();
    let n = 6_000;
    for _ in 0..n {
        let (holes, _) = g.deal(&mut arng);
        let (r, a, _live) = g.play_aivat(&pols, &holes, &mut arng, &mut rrng);
        raw.push(r[0] as f64);
        aiv.push(a[0]);
    }
    let (rm, rs) = (raw.mean(), raw.stderr());
    let (am, as_) = (aiv.mean(), aiv.stderr());
    eprintln!("AlwaysAggressive seat-0, {n} hands:");
    eprintln!("  RAW   {rm:+.4} ± {rs:.4} chips/hand");
    eprintln!("  AIVAT {am:+.4} ± {as_:.4} chips/hand");
    eprintln!("  variance-reduction (raw σ / aivat σ): {:.2}×", rs / as_);

    // (1) UNBIASED: AIVAT mean agrees with raw mean within combined error
    // (AIVAT is E[net | actions, holes], so same expectation — it only
    // removes runout sampling noise).
    let comb = (rs * rs + as_ * as_).sqrt();
    assert!(
        (rm - am).abs() < 4.0 * comb,
        "AIVAT SHIFTS the mean: raw {rm:+.4} vs aivat {am:+.4} (± {comb:.4}) — reject"
    );
    // (2) REDUCES: the runout control variate must tighten meaningfully
    // (showdown-dominated game ⇒ large runout variance removed).
    assert!(as_ < rs * 0.7, "AIVAT not tighter: aivat σ {as_:.4} vs raw σ {rs:.4}");
    eprintln!("AIVAT GATE PASS: unbiased (means agree), reduces {:.2}× (runout variance removed).", rs / as_);
    eprintln!("(factor here is the runout component only; real blueprint adds the action-node");
    eprintln!(" control variate at option 3 — factor measured there, not assumed.)");
}

#[test]
fn aivat_conserves_per_hand() {
    // The runout-expected net must still conserve: Σ(aivat net) = dead − rake
    // is NOT exact per-hand (rake varies with the board), but Σ over runouts
    // of (dead − rake) / count = dead − mean_rake, and Σ_p aivat[p] must equal
    // that. Check Σ aivat ∈ [dead − cap, dead] (rake in [0, cap]).
    for live in 2u8..=5 {
        let g = SeamGame::new(live, 2, 12, flop());
        let pols = vec![SeamPolicy::AlwaysAggressive; live as usize];
        let mut arng = 0xC3u64 ^ live as u64;
        let mut rrng = 0xD4u64;
        for _ in 0..300 {
            let (holes, _) = g.deal(&mut arng);
            let (_r, a, _l) = g.play_aivat(&pols, &holes, &mut arng, &mut rrng);
            let sum: f64 = a.iter().sum();
            assert!(
                sum <= g.dead as f64 + 1e-6 && sum >= g.dead as f64 - g.rake_cap as f64 - 1e-6,
                "live-{live}: Σaivat {sum} outside [dead−cap, dead]=[{},{}]",
                g.dead as f64 - g.rake_cap as f64, g.dead
            );
        }
    }
    eprintln!("AIVAT conserves (Σ runout-expected net = dead − mean_rake, within [dead−cap, dead]).");
}
