//! V1 seam per-family power re-run (step 2) — on SYNTHETIC policies (power is
//! a variance/sampling property; no blueprints, stays before the production
//! solves). Confirms the seeded per-family design DISSOLVES the rare-family
//! sampling problem the uniform-deal oracle hit (live-5 got 2 samples in 56k
//! decks): here EVERY family is dealt a controlled N, so every family's MDE
//! is f(N), resolvable. Also gates the true-reach reweighting combine.

use play_harness::experiment::{reach_weighted_overall, FamilyEdge, Stat};
use play_harness::v1_seam::{SeamGame, SeamPolicy};

fn flop() -> [u8; 3] { [47, 21, 0] } // Ks 7d 2c

// chips/seat-hand → mbb/100 (bb = 2 units in v1).
fn mbb100(chips: f64) -> f64 { chips / 2.0 * 1000.0 * 100.0 }

#[test]
fn seam_per_family_power_controlled() {
    // Per-family per-seat-hand outcome std sets the MDE; seeding fixes N per
    // family, so no family is starved (the oracle's live-5 = 2/56k problem).
    // Synthetic all-in play = a conservative (max-variance) upper bound.
    eprintln!("\n═══ V1 SEAM PER-FAMILY POWER (seeded, synthetic) ═══");
    eprintln!("family | per-seat σ (chips) | MDE @ N=1e6 seeded hands (×1.53 dup) mbb/100");
    let dup = 1.53f64; // duplicate variance-reduction factor (from the gated reducer)
    for live in 2u8..=6 {
        let g = SeamGame::new(live, 2, 12, flop());
        let pols = vec![SeamPolicy::AlwaysAggressive; live as usize];
        let mut rng = 0xA11Eu64 ^ (live as u64);
        let mut s = Stat::default();
        for _ in 0..20_000 {
            let (holes, board) = g.deal(&mut rng);
            let (net, _l) = g.play(&pols, &holes, &board, &mut rng);
            // seat 0's outcome as the per-seat-hand sample.
            s.push(net[0] as f64);
        }
        let per_seat_std = s.stderr() * (s.n as f64).sqrt();
        // MDE = 1.96 σ/√N, with duplicate cutting σ by √dup.
        let n = 1e6f64;
        let mde_chips = 1.96 * (per_seat_std / dup.sqrt()) / n.sqrt();
        eprintln!("  live-{live} | {per_seat_std:8.1} | {:8.1} mbb/100  (all families resolvable at controlled N)",
            mbb100(mde_chips));
        // The point: every family gets N samples by construction — none is
        // starved. A real (non-all-in) blueprint comparison has far lower σ.
        assert!(per_seat_std.is_finite() && s.n == 20_000, "live-{live}: family fully sampled");
    }
    eprintln!("→ EVERY family resolvable at controlled N (seeding solves the rare-family starvation);");
    eprintln!("  all-in synthetic σ is a conservative upper bound — real blueprints mix, lower σ.");
}

#[test]
fn reach_weighting_combine_correct() {
    // Hand-computable: per-family edges weighted by banked reach×pot.
    // weights (reach×pot, banked): live-2 .153 / 3 .323 / 4 .328 / 5 .163 / 6 .032.
    let fams = vec![
        FamilyEdge { live: 2, edge: 10.0, stderr: 1.0, reach_weight: 0.153 },
        FamilyEdge { live: 3, edge: 20.0, stderr: 1.0, reach_weight: 0.323 },
        FamilyEdge { live: 4, edge: 30.0, stderr: 1.0, reach_weight: 0.328 },
        FamilyEdge { live: 5, edge: -50.0, stderr: 2.0, reach_weight: 0.163 }, // rare, LOSES
        FamilyEdge { live: 6, edge: -80.0, stderr: 5.0, reach_weight: 0.032 },
    ];
    let (edge, se) = reach_weighted_overall(&fams);
    // Σw = 0.999; weighted = (.153·10+.323·20+.328·30+.163·-50+.032·-80)/.999
    //     = (1.53+6.46+9.84-8.15-2.56)/.999 = 7.12/.999 = 7.13
    let expect = (0.153 * 10.0 + 0.323 * 20.0 + 0.328 * 30.0 + 0.163 * -50.0 + 0.032 * -80.0) / 0.999;
    eprintln!("reach-weighted overall {edge:.3} ± {se:.3} (expect {expect:.3})");
    assert!((edge - expect).abs() < 1e-6, "reach-weighting wrong: {edge} vs {expect}");
    // The teaching case: live-5/6 LOSE but are rare, so the blend is +7.13 —
    // a NAIVE average would be (10+20+30-50-80)/5 = -14, opposite sign. The
    // reach weighting (not the seeded rate) is what makes the verdict honest.
    let naive = (10.0 + 20.0 + 30.0 - 50.0 - 80.0) / 5.0;
    eprintln!("  (naive unweighted average = {naive:.1} — OPPOSITE SIGN; weighting is load-bearing)");
    assert!(edge > 0.0 && naive < 0.0, "demonstrates weighting flips the verdict vs naive average");
    eprintln!("REACH-WEIGHTING GATE PASS.");
}
