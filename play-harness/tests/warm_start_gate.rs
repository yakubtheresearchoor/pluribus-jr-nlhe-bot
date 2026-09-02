//! WARM-START GATE (docket item #6): decide ship/no-ship for seeding the
//! real-time search from the blueprint lookup.
//!
//! History: the NAIVE warm-start was previously MEASURED BAD. This variant is
//! the principled one — seed ONLY the cumulative (average) strategy from the
//! blueprint's own per-bucket action dist (regrets untouched, so regret
//! matching / DCFR dynamics are unchanged; the seed acts as `weight` virtual
//! iterations of the blueprint strategy that decay as real iterations
//! accumulate — it cannot move the fixed point, only the early average).
//!
//! Gate: warm exploitability must be ≤ cold at the production iteration count
//! (and not worse anywhere along the curve beyond noise), with both converging
//! to the same floor. If warm does not beat cold, the measured outcome is a
//! documented NO-SHIP (like off-grid-turn and live-6-river).
//!
//! Heavy (blueprint load ~50s + 2× sweep). Run on demand:
//!   cargo test --release -p play-harness --test warm_start_gate -- --ignored --nocapture

use play_harness::api_conn::ConnDecider;

fn ws(p: &str) -> String { format!("{}/../{}", env!("CARGO_MANIFEST_DIR"), p) }

#[test]
#[ignore = "warm-start gate: loads the full blueprint + 2x exploitability sweep. Run on demand."]
fn warm_start_beats_or_matches_cold() {
    let Ok(dec) = ConnDecider::load(
        &std::env::var("CONN_TEST_BP").unwrap_or_else(|_| ws("blueprint_conn_eqr")),
        &ws("gs14_blueprint_cache"), 6, 5, 200, 7,
    ) else { eprintln!("blueprint absent — skipping"); return; };

    // Same representative live-3 flop cell as conn_flop_exploit. Checkpoints
    // bracket the production budget (CONN_ITERS ~ 32-40).
    let (flop_id, live, commit, pot) = (0usize, 3usize, 6i32, 18i32);
    let checkpoints = [8u32, 16, 32, 64, 160];
    let weight: f32 = std::env::var("WARM_W").ok().and_then(|s| s.parse().ok()).unwrap_or(8.0);

    let sweep = dec.warm_start_flop_sweep(flop_id, live, commit, pot, &checkpoints, weight);
    if sweep.is_empty() { eprintln!("no adapter/cell — skipping"); return; }

    eprintln!("warm-start gate (live={live}, pot={pot}, seed weight={weight}):");
    eprintln!("  {:>6} {:>12} {:>12} {:>8}", "iters", "cold %pot", "warm %pot", "Δ");
    for &(it, c, w) in &sweep {
        eprintln!("  {:>6} {:>11.3}% {:>11.3}% {:>+7.3}", it, c, w, w - c);
    }

    // Gate 1: at the production checkpoint (32 iters) warm must not be worse
    // than cold beyond noise (0.5% pot).
    let prod = sweep.iter().find(|&&(it, _, _)| it == 32).unwrap();
    assert!(
        prod.2 <= prod.1 + 0.5,
        "warm start HURTS at production iters: cold {:.3}% vs warm {:.3}%",
        prod.1, prod.2
    );

    // Gate 2: same fixed point — at the deepest checkpoint the two must agree
    // within noise (the seed must decay away, not bias the equilibrium).
    let last = sweep.last().unwrap();
    assert!(
        (last.2 - last.1).abs() <= 1.0,
        "warm start BIASES the fixed point: cold {:.3}% vs warm {:.3}% at {} iters",
        last.1, last.2, last.0
    );

    // Ship signal (informational, the ship decision reads the table): does warm
    // actually HELP early?
    let early = sweep.iter().find(|&&(it, _, _)| it == 8).unwrap();
    if early.2 < early.1 - 0.25 {
        eprintln!("SHIP SIGNAL: warm helps early ({:.3}% -> {:.3}% at 8 iters)", early.1, early.2);
    } else {
        eprintln!("NO-SHIP SIGNAL: warm does not help early ({:.3}% vs {:.3}% at 8 iters)", early.1, early.2);
    }
}
