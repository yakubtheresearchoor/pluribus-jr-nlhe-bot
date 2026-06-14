//! REALISTIC-VARIANCE BALLPARK (the AIVAT decision) + CANDIDATE-BUILD PRICING
//! — the two pre-option-3 checks. All-in synthetic was a max-variance upper
//! bound; a mixing policy is far closer to real blueprints. The realistic
//! per-family σ sets the MDE at feasible N, which decides duplicate-1.53× vs
//! AIVAT. Treated as a BALLPARK (real mixing unknown until blueprints exist):
//! clearly-low → skip AIVAT, clearly-high or borderline → build AIVAT
//! (under-powering the production experiment costs more than the reducer).

use play_harness::v1_seam::{SeamGame, SeamPolicy};
use play_harness::experiment::Stat;

fn flop() -> [u8; 3] { [47, 21, 0] }
fn mbb100(chips: f64) -> f64 { chips / 2.0 * 1000.0 * 100.0 }

#[test]
#[ignore = "realistic-variance ballpark; --ignored --nocapture --release"]
fn seam_realistic_variance_aivat_call() {
    eprintln!("\n═══ REALISTIC per-family variance (Mixed policy, ballpark) ═══");
    eprintln!("family | per-seat σ chips | MDE mbb/100 @1e6 / @1e7 (dup 1.53×)");
    let dup = 1.53f64;
    let mut worst_mde_1e7 = 0.0f64;
    for live in 2u8..=6 {
        let g = SeamGame::new(live, 2, 12, flop());
        let pols = vec![SeamPolicy::Mixed; live as usize];
        let mut rng = 0x1357u64.wrapping_add(live as u64).wrapping_mul(2654435761);
        let mut s = Stat::default();
        for _ in 0..40_000 {
            let (holes, board) = g.deal(&mut rng);
            let (net, _l) = g.play(&pols, &holes, &board, &mut rng);
            s.push(net[0] as f64);
        }
        let sd = s.stderr() * (s.n as f64).sqrt();
        let mde = |n: f64| mbb100(1.96 * (sd / dup.sqrt()) / n.sqrt());
        worst_mde_1e7 = worst_mde_1e7.max(mde(1e7));
        eprintln!("  live-{live} | {sd:7.1} | {:8.1} / {:8.1}", mde(1e6), mde(1e7));
    }
    eprintln!("worst-family MDE @1e7 hands ≈ {worst_mde_1e7:.0} mbb/100 (duplicate-only)");
    eprintln!("→ AIVAT CALL: B8 is catastrophic multiway, so the fidelity edge is plausibly");
    eprintln!("  LARGE (100s mbb/100) on families where blueprints differ. If worst MDE @1e7 «");
    eprintln!("  that edge, duplicate-only suffices; if comparable/larger, build AIVAT. BALLPARK —");
    eprintln!("  real blueprint mixing unknown; borderline ⇒ default to AIVAT (cheaper than re-run).");
}

/// CANDIDATE-BUILD PRICING (the expensive half of option 3 is BUILDING the
/// three blueprints, not running the harness). Price each candidate against
/// the banked cost ladder + reach weights. Pure arithmetic on banked numbers.
#[test]
#[ignore = "candidate-build pricing; --ignored --nocapture --release"]
fn candidate_blueprint_build_pricing() {
    // Banked GPU per-cell-row hours at B8 (validated prices, per family).
    let gpu_b8_h = [("live-2", 0.05f64), ("live-3", 0.36), ("live-4", 1.6), ("live-5", 16.7), ("live-6", 192.0)];
    // Reach×pot weights (banked).
    let _w = [0.153, 0.323, 0.328, 0.163, 0.032];
    // Cost-ladder super-linearity (per-iter, nh=1176, banked): B8 10.1s →
    // B10 38 → B15 400 → B20 2195 → B25 8453. Multiplier vs B8:
    let ladder = [("B8", 1.0f64), ("B10", 3.76), ("B15", 39.6), ("B20", 217.0), ("B25", 837.0)];
    eprintln!("\n═══ CANDIDATE-BLUEPRINT BUILD PRICING (GPU hours) ═══");
    let b8_total: f64 = gpu_b8_h.iter().map(|(_, h)| h).sum();
    eprintln!("B8-uniform (the floor): Σ = {b8_total:.1} GPU-h  (live-6 alone {:.0}h = {:.0}%)",
        gpu_b8_h[4].1, gpu_b8_h[4].1 / b8_total * 100.0);
    eprintln!("ladder cost of a UNIFORM finer blueprint (× B8 total {b8_total:.0}h):");
    for (name, mult) in &ladder {
        eprintln!("  {name}-uniform: {:.0} GPU-h", b8_total * mult);
    }
    eprintln!("\nREACH-WEIGHTED (full fidelity common live-3/4, COARSE rare live-5/6):");
    // reach-weighted: common families finer (say B15-equiv on live-3/4), rare
    // families at B8 or below. live-5/6 dominate B8 cost (208h of 211h), so
    // keeping them coarse is what makes reach-weighting affordable.
    let rw_common = gpu_b8_h[0].1 + gpu_b8_h[1].1 * 39.6 + gpu_b8_h[2].1 * 39.6; // live-2 B8, live-3/4 B15
    let rw_rare = gpu_b8_h[3].1 + gpu_b8_h[4].1; // live-5/6 at B8
    eprintln!("  reach-weighted (live-3/4 @B15, live-5/6 @B8): ≈ {:.0} GPU-h", rw_common + rw_rare);
    eprintln!("  reach-weighted-PLUS (live-3/4 @B20): ≈ {:.0} GPU-h",
        gpu_b8_h[0].1 + (gpu_b8_h[1].1 + gpu_b8_h[2].1) * 217.0 + rw_rare);
    eprintln!("\n→ 'finer-UNIFORM' is dominated by live-6 on the super-linear ladder (B15-uniform");
    eprintln!("  ≈ {:.0}h, B20 ≈ {:.0}h) — likely UNAFFORDABLE. The buildable+decision-relevant",
        b8_total * 39.6, b8_total * 217.0);
    eprintln!("  three points are: B8 (floor) / reach-weighted (the escape) / reach-weighted-PLUS");
    eprintln!("  (finer on common families only, where it's affordable) — NOT a finer-uniform that");
    eprintln!("  spends the ladder on rare live-6 the bot reaches 3% of the time.");
}
