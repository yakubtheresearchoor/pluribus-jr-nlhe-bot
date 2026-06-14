//! STATISTICAL POWER ANALYSIS (before option 3, user-ordered): verified ≠
//! powered. The duplicate reducer is unbiased and cuts card-luck variance
//! 1.53×, but option 3's question — does reach-weighted beat B8 — may be a
//! SMALL edge, and the decision-critical families (live-5/6) are RARE, so
//! their power is the worst. This prices the minimum detectable edge (MDE,
//! in mbb/100) vs hand count, blended AND per-family, so option 3 isn't
//! fired underpowered. Uses Blueprint-vs-Blueprint as the variance proxy
//! (the reach-weighted-vs-B8 comparison has the same card-luck variance
//! structure, two similar strategies). Needs BP_ARTIFACT.

use play_harness::blueprint::{build_oracle_tree, Blueprint};
use play_harness::experiment::ab_duplicate;
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

// chips (3-seat-team paired sample per deck) → mbb/100 per seat-hand.
// per-seat-hand edge = team / 3; mbb/100 = chips/bb × 1000 × 100, bb=2.
fn team_chips_to_mbb100(team_chips: f64) -> f64 {
    (team_chips / 3.0) / 2.0 * 1000.0 * 100.0
}

#[test]
#[ignore = "power analysis (minutes); --ignored --nocapture --release"]
fn power_vs_hand_count() {
    let Some((bp, tree)) = env_or_skip() else { return };
    let env = MatchEnv::new(&bp, &tree);
    let n0 = 100_000usize;
    let mut rng = 0xBEEF_u64;
    let r = ab_duplicate(&env, &Policy::Blueprint(&bp), &Policy::Blueprint(&bp), n0, &mut rng);

    // Per-DECK std of the reduced estimator (stderr × √n).
    let red_per_deck = r.reduced.stderr() * (r.reduced.n as f64).sqrt();
    let raw_per_deck = r.raw.stderr() * (r.raw.n as f64).sqrt();
    eprintln!("\n═══ POWER ANALYSIS (reduced, 1.53× duplicate) ═══");
    eprintln!("per-deck σ: reduced {red_per_deck:.1} chips (raw {raw_per_deck:.1}); calib n={}", r.reduced.n);
    eprintln!("MINIMUM DETECTABLE EDGE (95% CI half-width = 1.96σ/√N), per seat-hand:");
    eprintln!("  decks N | reduced MDE | duplicate-only | with AIVAT(~10×)");
    for &nn in &[1e4f64, 1e5, 1e6, 1e7, 1e8] {
        let stderr = red_per_deck / nn.sqrt();
        let mde = team_chips_to_mbb100(1.96 * stderr);
        let mde_aivat = mde / 10f64.sqrt(); // AIVAT ~10× variance → √10 on MDE
        eprintln!("  {nn:>7.0e} | {mde:8.1} mbb/100 | (above)        | {mde_aivat:8.1} mbb/100");
    }

    // PER-FAMILY: rare families get few samples → far worse power. The
    // live-5/6 rows are where the reach-weighting decision lives.
    let total: u64 = r.by_live.values().map(|s| s.n).sum();
    eprintln!("\nPER-FAMILY sampling + power penalty (the decision lives on live-5/6):");
    eprintln!("  family | freq | per-sample σ | N-multiplier vs blended | MDE @ N=1e7 decks");
    for (live, s) in &r.by_live {
        let freq = s.n as f64 / total as f64;
        let fam_samples = 1e7 * 2.0 * freq; // 2 assignments/deck × freq
        let fam_std = s.stderr() * (s.n as f64).sqrt(); // per-hand std in this family
        let mde_fam = if fam_samples > 1.0 {
            team_chips_to_mbb100(1.96 * fam_std / fam_samples.sqrt())
        } else { f64::INFINITY };
        eprintln!(
            "  live-{live} | {:.4} | {fam_std:7.1} | {:8.0}× | {mde_fam:8.1} mbb/100 ({} samples)",
            freq, 1.0 / freq.max(1e-12), s.n
        );
    }
    eprintln!("\n→ blended/common families: read the MDE table for the feasible-N edge.");
    eprintln!("  rare families (live-5/6): freq tiny ⇒ MDE huge at any feasible N ⇒ UNIFORM random");
    eprintln!("  deal CANNOT resolve them — option 3 must OVERSAMPLE rare families (forced live-5/6");
    eprintln!("  deals) or their quality stays harness-unresolved. AIVAT helps the common edge, not");
    eprintln!("  the rare-family SAMPLING problem (that needs stratified dealing).");
}
