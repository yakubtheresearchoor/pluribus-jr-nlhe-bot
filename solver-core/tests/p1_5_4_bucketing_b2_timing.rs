//! Phase B2 — measured per-flop costs for the GS14 bucketing pipeline and
//! the W/T/L precompute, at PRODUCTION scale (full nh ≈ 1176, full
//! 49 × 48 runout enumeration, k-means++ × 25 restarts per the spec).
//!
//! Note 3 of the B1 confirmation: "the W/T/L precompute is a cost claim
//! without a measurement … B2 should produce a measured per-flop W/T/L
//! precompute time at production nh as one of its first numbers, and the
//! blueprint wall-clock projection in B4 must include it."
//!
//! This test produces:
//!   - per-stage pipeline wall-clock (equity enumeration, river/turn/flop
//!     clustering at B = 15 each, 25 restarts)
//!   - per-flop W/T/L precompute wall-clock over all runouts
//!   - the ×1755-canonicals projection line for B4
//!
//! Per-stage progress is printed BEFORE each stage starts, so a mis-sized
//! stage is visible in the log within seconds (the 16-hour preflop lesson,
//! encoded).

use std::time::Instant;

use solver_core::abstraction::postflop_buckets::{
    build_postflop_bucketing, compute_wtl_for_runout, enumerate_valid_hands,
    strengths_at_river,
};
use solver_core::card::{card_from_str, Card};

#[test]
#[ignore = "B2 timing: per-flop pipeline + W/T/L at production scale (~minutes, prints stage timings)"]
fn b2_per_flop_pipeline_and_wtl_timing() {
    let flop: [Card; 3] = [
        card_from_str("Th").unwrap(),
        card_from_str("9d").unwrap(),
        card_from_str("8c").unwrap(),
    ];
    eprintln!("\n=== B2 timing: per-flop GS14 pipeline + W/T/L (production scale) ===");
    eprintln!("Flop: Th 9d 8c (wet); buckets B_f=B_t=B_r=15; restarts=25 (GS14 verbatim)");

    // Full runout enumeration: 49 turns × 48 rivers per turn.
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let turns: Vec<Card> = (0u8..52).filter(|&c| board_mask & (1u64 << c) == 0).collect();
    let rivers: Vec<Vec<Card>> = turns.iter().map(|&tc| {
        (0u8..52).filter(|&c| c != tc && board_mask & (1u64 << c) == 0).collect()
    }).collect();
    let n_runouts: usize = rivers.iter().map(|r| r.len()).sum();
    let hands = enumerate_valid_hands(&flop);
    eprintln!("nh = {}, turns = {}, runouts = {}", hands.len(), turns.len(), n_runouts);

    // ── Pipeline (equity + 3-street clustering inside) ──
    eprintln!("\n[stage] full pipeline starting (equity enumeration is the first phase inside)…");
    let t0 = Instant::now();
    let b = build_postflop_bucketing(&flop, &turns, &rivers, 15, 15, 15, 25, 0xB2B2);
    let pipeline_s = t0.elapsed().as_secs_f64();
    eprintln!("[stage] pipeline DONE: {:.1}s  (wcss river={:.4} turn={:.4} flop={:.4})",
        pipeline_s, b.wcss_river, b.wcss_turn, b.wcss_flop);

    // Bucket population sanity (no empty river/turn/flop buckets at B=15
    // on a wet board would be suspicious the other way; just report).
    let mut flop_pop = vec![0usize; 15];
    for &m in &b.flop_map { flop_pop[m as usize] += 1; }
    eprintln!("flop bucket populations: {:?}", flop_pop);

    // ── W/T/L precompute over all runouts (river maps) ──
    eprintln!("\n[stage] W/T/L precompute starting ({} runouts)…", n_runouts);
    let weights = vec![1.0f64; hands.len()];
    let t1 = Instant::now();
    let mut checksum = 0.0f64;
    for (ti, &tc) in turns.iter().enumerate() {
        for (ri, &rc) in rivers[ti].iter().enumerate() {
            let strengths = strengths_at_river(&hands, &flop, tc, rc);
            let m = compute_wtl_for_runout(
                &hands, &strengths, &weights, &b.river_map[ti][ri], 15,
            );
            checksum += m.w[0] + m.t[0] + m.l[0];
        }
        if ti == 0 {
            eprintln!("[stage] first turn column done ({} rivers): {:.2}s — projected total ≈ {:.1}s",
                rivers[0].len(), t1.elapsed().as_secs_f64(),
                t1.elapsed().as_secs_f64() * turns.len() as f64);
        }
    }
    let wtl_s = t1.elapsed().as_secs_f64();
    eprintln!("[stage] W/T/L DONE: {:.1}s over {} runouts ({:.2} ms/runout)  [checksum {:.3e}]",
        wtl_s, n_runouts, wtl_s * 1000.0 / n_runouts as f64, checksum);

    // ── B4 projection line ──
    eprintln!("\n=== B2 measured numbers (for the B4 projection) ===");
    eprintln!("per-flop pipeline:  {:.1}s", pipeline_s);
    eprintln!("per-flop W/T/L:     {:.1}s", wtl_s);
    eprintln!("× 1755 canonicals:  pipeline ≈ {:.1}h, W/T/L ≈ {:.1}h (single-core; embarrassingly parallel across flops)",
        pipeline_s * 1755.0 / 3600.0, wtl_s * 1755.0 / 3600.0);
}
