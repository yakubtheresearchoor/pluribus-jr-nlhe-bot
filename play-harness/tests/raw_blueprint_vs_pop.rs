//! TRACK 1 — RAW BLUEPRINT vs the NL10 loose-passive POOL (no search).
//! Establishes the baseline win rate the search-enhanced run must beat. Uses
//! the variance-reduced mirrored duplicate (Blueprint at seats {0,1,2}, the
//! calibrated Population at {3,4,5}, then swapped) so card-luck cancels. The
//! Population policy is calibrated to reference_6max_pool_priors (NL10 row:
//! cbet 52.4 / fold-to-cbet 29.1 / chk-raise 7.3). This is a POSTFLOP
//! (flop-start) sim, so only the pool's postflop tendencies are modelled.
//!
//! Run: BP_ARTIFACT=<path> cargo test --release -p play-harness --test \
//!      raw_blueprint_vs_pop raw_blueprint_vs_population -- --ignored --nocapture

use play_harness::blueprint::{build_oracle_tree, Blueprint};
use play_harness::experiment::ab_duplicate;
use play_harness::match_play::{MatchEnv, Policy};

// chips (3-seat-team paired sample per deck) → mbb/100 per seat-hand. bb = 2 chips.
fn team_chips_to_mbb100(team_chips: f64) -> f64 {
    (team_chips / 3.0) / 2.0 * 1000.0 * 100.0
}

#[test]
#[ignore = "raw blueprint vs NL10 pool baseline; --ignored --nocapture --release"]
fn raw_blueprint_vs_population() {
    let path = std::env::var("BP_ARTIFACT")
        .unwrap_or_else(|_| "../blueprint_out_b10_4x4/flop_0000.bp".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: no banked artifact at {path}");
        return;
    }
    let bp = Blueprint::load(&path).unwrap();
    let tree = build_oracle_tree();
    let env = MatchEnv::new(&bp, &tree);

    let n_decks: usize = std::env::var("BP_HANDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let mut rng = 0xF00D_u64;

    // A = our raw blueprint, B = the NL10 pool. reduced.mean() = blueprint's
    // chip edge over the pool per deck (3-seat team), variance-reduced.
    let r = ab_duplicate(&env, &Policy::Blueprint(&bp), &Policy::Population, n_decks, &mut rng);

    let edge = team_chips_to_mbb100(r.reduced.mean());
    let ci95 = team_chips_to_mbb100(1.96 * r.reduced.stderr());
    eprintln!("\n═══ RAW BLUEPRINT vs NL10 POOL (no search) ═══");
    eprintln!("decks={n_decks}  samples={}", r.reduced.n);
    eprintln!(
        "blueprint edge = {edge:+.1} ± {ci95:.1} mbb/100 (95% CI)  {}",
        if edge - ci95 > 0.0 { "✓ beats pool (significant)" }
        else if edge + ci95 < 0.0 { "✗ LOSES to pool (significant)" }
        else { "≈ inconclusive (CI spans 0)" }
    );
    eprintln!("per-live-count edge:");
    let total: u64 = r.by_live.values().map(|s| s.n).sum();
    for (live, s) in &r.by_live {
        let e = team_chips_to_mbb100(s.mean());
        eprintln!(
            "  live-{live}: {e:+8.1} mbb/100  (n={}, {:.1}%)",
            s.n,
            100.0 * s.n as f64 / total as f64
        );
    }
    eprintln!("→ this is the baseline; the search-enhanced run (correct k=4 leaf values) must beat it.");
}
