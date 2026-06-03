// Cost projection: measure K=3 factored TVRP at production nh=50, extrapolate
// K=4 / K=5 recursion costs, compare against a 25s solve budget.
//
// Per the agreed plan: this is the cheap check that gates K=4/K=5 derivation
// behind a "tractable correct" filter. If K=5 doesn't fit even with the
// theoretical maximum throughput, isomorphism / kernel-overhead reduction
// work moves onto the critical path BEFORE further formula derivation.
//
// The numbers here come from microbenchmarking the SAME factored function
// that production code calls, at the SAME nh production code uses. No
// theoretical FLOPS handwaving — measured wall-clock per call.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::showdown::{precompute_opp_masses, total_valid_reach_product};
use std::time::Instant;

fn build_inputs(nh: usize, num_opp: usize) -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();

    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1; hand_cards[i * 2 + 1] = c2;
    }

    let mut hand_strength = vec![0u16; nh];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        hand_strength[i] = h.evaluate_internal() as u16;
    }

    // Non-uniform reach to exercise floating-point arithmetic, not just
    // the integer-reach fast paths.
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.5 + 0.4 * ((h + oi * 7) % 11) as f32 / 11.0).collect()
    }).collect();

    (hand_cards, hand_strength, reach)
}

#[test]
fn project_k3_factored_at_nh50() {
    let nh = 50usize;
    let num_opp = 3usize;  // K=3 = 4-player
    let (hand_cards, hand_strength, reach) = build_inputs(nh, num_opp);
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();

    // Precompute once (this cost is paid per-terminal too).
    let t_pre0 = Instant::now();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let pre_elapsed = t_pre0.elapsed();
    eprintln!("[K=3 nh=50] precompute_opp_masses: {:?}", pre_elapsed);

    // Time many TVRP calls.
    const N_CALLS: usize = 1000;
    let t0 = Instant::now();
    let mut sink = 0.0f64;
    for _ in 0..N_CALLS {
        let out = total_valid_reach_product(&masses, &reach_views);
        sink += out.iter().sum::<f64>();
    }
    let elapsed = t0.elapsed();
    std::hint::black_box(sink);

    let per_call_us = elapsed.as_secs_f64() * 1e6 / N_CALLS as f64;
    let per_h_ns = elapsed.as_secs_f64() * 1e9 / (N_CALLS * nh) as f64;

    eprintln!("[K=3 nh=50] {} TVRP calls in {:?}", N_CALLS, elapsed);
    eprintln!("[K=3 nh=50] per-call (= per-terminal): {:.2} µs", per_call_us);
    eprintln!("[K=3 nh=50] per-h: {:.1} ns", per_h_ns);

    // EXTRAPOLATION to K=4 / K=5 by recursion-depth multiplier.
    // Per the recursive K=2 expansion: K-opp TVRP cost is O(nh^{K-2} · 52).
    // K=3: nh^1 · 52 per h.
    // K=4: nh^2 · 52 per h. = K=3 × nh.
    // K=5: nh^3 · 52 per h. = K=3 × nh^2.
    let per_terminal_k3_us = per_call_us;
    let per_terminal_k4_us = per_call_us * nh as f64;
    let per_terminal_k5_us = per_call_us * (nh * nh) as f64;
    eprintln!("");
    eprintln!("Extrapolated per-terminal cost (recursion-depth multiplier × {}):", nh);
    eprintln!("  K=3 (measured): {:>10.2} µs", per_terminal_k3_us);
    eprintln!("  K=4 (× nh):     {:>10.2} µs = {:.2} ms", per_terminal_k4_us, per_terminal_k4_us / 1000.0);
    eprintln!("  K=5 (× nh²):    {:>10.2} µs = {:.2} ms", per_terminal_k5_us, per_terminal_k5_us / 1000.0);

    // Tree size estimates (rough, from prior smoke testing).
    // The 4p tree at nh=50 has on the order of 5,000 terminals; the
    // 6p tree closer to 10,000. Adjust per-terminal × terminals to get
    // per-iter cost.
    let terminals_4p = 5_000u64;
    let terminals_6p = 10_000u64;
    eprintln!("");
    eprintln!("Per-iter cost (terminals × per-terminal, assumed terminals: 4p=5k, 6p=10k):");
    let per_iter_k3_s = per_terminal_k3_us * 1e-6 * terminals_4p as f64;
    let per_iter_k4_s = per_terminal_k4_us * 1e-6 * terminals_4p as f64;
    let per_iter_k5_s = per_terminal_k5_us * 1e-6 * terminals_6p as f64;
    eprintln!("  K=3 4p: {:.2} s per iter", per_iter_k3_s);
    eprintln!("  K=4 5p: {:.2} s per iter (assuming 5p tree size ≈ 4p)", per_iter_k4_s);
    eprintln!("  K=5 6p: {:.2} s per iter", per_iter_k5_s);

    // Convergence iteration counts. The 3-player nh=50 baseline floors at
    // 0.01% in 2000 iters. Higher player count plausibly needs comparable
    // order — let's project at 2000 iters as a starting estimate.
    let iters_to_floor = 2000u64;
    let total_k3_s = per_iter_k3_s * iters_to_floor as f64;
    let total_k4_s = per_iter_k4_s * iters_to_floor as f64;
    let total_k5_s = per_iter_k5_s * iters_to_floor as f64;
    eprintln!("");
    eprintln!("Full-solve cost (× {} iters to floor):", iters_to_floor);
    eprintln!("  K=3 4p: {:.1} s ({:.2} min)", total_k3_s, total_k3_s / 60.0);
    eprintln!("  K=4 5p: {:.1} s ({:.2} min)", total_k4_s, total_k4_s / 60.0);
    eprintln!("  K=5 6p: {:.1} s ({:.2} min)", total_k5_s, total_k5_s / 60.0);

    let budget_s = 25.0;
    eprintln!("");
    eprintln!("vs 25s solve budget:");
    eprintln!("  K=3 4p: {:.1}× budget", total_k3_s / budget_s);
    eprintln!("  K=4 5p: {:.1}× budget", total_k4_s / budget_s);
    eprintln!("  K=5 6p: {:.1}× budget", total_k5_s / budget_s);

    // Isomorphism reduction estimate. Suit symmetry typically reduces
    // effective nh by 4× (one canonical suit, three permutations). The
    // factored formula's per-h cost scales as nh^{K-2}, so:
    //   K=3: × 1/(4^1) = ×0.25
    //   K=4: × 1/(4^2) = ×0.0625
    //   K=5: × 1/(4^3) = ×0.015625
    eprintln!("");
    eprintln!("With suit-symmetry isomorphism (effective nh × 1/4):");
    eprintln!("  K=3 4p: {:.1}× budget", total_k3_s * 0.25 / budget_s);
    eprintln!("  K=4 5p: {:.1}× budget", total_k4_s * 0.0625 / budget_s);
    eprintln!("  K=5 6p: {:.1}× budget", total_k5_s * 0.015625 / budget_s);

    // This is a soft assertion — the test is for the projection PRINT,
    // not a hard pass/fail. We assert only that the timing did happen
    // (per-call > 0) so the test fails if the bench compiled wrong.
    assert!(per_call_us > 0.0);
}
