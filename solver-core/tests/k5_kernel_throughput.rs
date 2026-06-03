// K=5 brute-force inner-loop GPU microbenchmark.
//
// LOAD-BEARING MEASUREMENT for the corrected K=5 projection:
//   - The "K=5 reaches the 25s budget with optimization" conclusion rests on
//     the assumption that the K=5 kernel runs at 1–30 TFLOPS on the M4 Max,
//     up to 30–100× the 287 MFLOPS measured on the 3p K=2 production path.
//   - The 287 MFLOPS on the K=2 path is overhead-bound: each iter does only
//     17,107 × 50 × 2500 = 2.14B ops total, kernel-launch and tree-traversal
//     overhead dominates.
//   - The K=5 inner kernel is two orders of magnitude more compute-dense per
//     memory access (nh^5 ≈ 312M FMAs per h_player vs nh^2 = 2500), so it
//     SHOULD run closer to peak — but should-arguments do not replace
//     measurements, and the budget conclusion swings entirely on which
//     point in [1 TFLOPS, 30 TFLOPS] the kernel actually hits.
//
// This bench dispatches the K=5 brute-force inner loop at nh=50 with enough
// concurrent threads (batches × nh = 50,000+) to saturate the M4 Max GPU,
// measures wall-clock, and backs out the achieved ops/sec.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::buffer::MetalBuffer;
use std::time::Instant;
use metal::MTLSize;

const NH: usize = 50;
const NUM_OPP: usize = 5;

fn build_inputs() -> (Vec<u8>, Vec<f32>) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let mut hand_cards = vec![0u8; NH * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1; hand_cards[i * 2 + 1] = c2;
    }
    // Reach: non-uniform but all > 0 so almost every (g_0..g_4) tuple is "live"
    // — measure the dense inner-loop throughput, not the early-exit-pruned one.
    let mut reach = vec![0f32; NUM_OPP * NH];
    for oi in 0..NUM_OPP {
        for h in 0..NH {
            reach[oi * NH + h] = 0.3 + 0.6 * ((h + oi * 5) % 11) as f32 / 11.0;
        }
    }
    (hand_cards, reach)
}

/// Count actual valid (g_0, ..., g_{K-1}) tuples for one h_player, given
/// hand cards. Used to compute achieved ops/sec from wall-clock.
fn count_valid_tuples_per_h(hand_cards: &[u8], h: usize, num_opp: usize, nh: usize) -> u64 {
    let g_mask: Vec<u64> = (0..nh)
        .map(|g| (1u64 << hand_cards[g * 2]) | (1u64 << hand_cards[g * 2 + 1]))
        .collect();
    let h_m = g_mask[h];

    fn recurse(depth: usize, num_opp: usize, nh: usize, mask: u64, g_mask: &[u64]) -> u64 {
        if depth == num_opp { return 1; }
        let mut count = 0u64;
        for g in 0..nh {
            if g_mask[g] & mask != 0 { continue; }
            count += recurse(depth + 1, num_opp, nh, mask | g_mask[g], g_mask);
        }
        count
    }
    recurse(0, num_opp, nh, h_m, &g_mask)
}

#[test]
fn k5_brute_force_kernel_throughput() {
    let (hand_cards, reach) = build_inputs();

    // Pre-count valid tuples for ONE h_player (the others are roughly
    // identical by symmetry; this gives the per-h structural compute count
    // without paying 50× the count time).
    let h_probe = NH / 2;
    let t0_count = Instant::now();
    let per_h_tuples = count_valid_tuples_per_h(&hand_cards, h_probe, NUM_OPP, NH);
    let count_elapsed = t0_count.elapsed();
    let tuples_per_batch: u64 = per_h_tuples * NH as u64;
    let avg_per_h = per_h_tuples;
    eprintln!("Probed valid K=5 tuples for h={}: {} ({:?})", h_probe, per_h_tuples, count_elapsed);
    eprintln!("=== K=5 brute-force structural compute count (nh={}) ===", NH);
    eprintln!("Valid K=5 tuples summed across {} h_player slots: {}", NH, tuples_per_batch);
    eprintln!("Average tuples per h: {} (~{:.1}M)", avg_per_h, avg_per_h as f64 / 1e6);

    // Ops per tuple inside the kernel: 1 mask load, ~5 ANDs, 1 reach load,
    // 1 multiply, 1 FMA, plus the outer mask/branch overhead. Count as
    // ~10 "ops" per tuple — a conservative arithmetic-intensity model.
    // (For a strict FMA-only TFLOPS metric we'd count 1 FMA = 2 FLOPs per
    // tuple. Both numbers reported below.)
    let kernel_ops_per_tuple: u64 = 10;     // mixed mask + arithmetic
    let kernel_flops_per_tuple: u64 = 2;    // FMA-only conservative

    let ctx = MetalContext::new().expect("Metal context");
    let pipeline = ctx.create_pipeline("k5_brute_force_microbench")
        .expect("kernel not found");

    let reach_buf = MetalBuffer::from_slice(ctx.device(), &reach);
    let hc_buf = MetalBuffer::from_slice(ctx.device(), &hand_cards);

    // Sweep batch counts to find saturating one.
    let batch_counts = &[1usize, 10, 100, 500, 1000, 2000];

    for &batches in batch_counts {
        let out_len = batches * NH;
        let out_buf: MetalBuffer<f32> = MetalBuffer::zeros(ctx.device(), out_len);

        // Warm-up dispatch (don't time first run; includes pipeline cache hit).
        {
            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(reach_buf.as_ref()), 0);
            enc.set_buffer(1, Some(hc_buf.as_ref()), 0);
            enc.set_buffer(2, Some(out_buf.as_ref()), 0);
            let nh32 = NH as i32;
            let b32 = batches as i32;
            enc.set_bytes(3, 4, &nh32 as *const _ as *const _);
            enc.set_bytes(4, 4, &b32 as *const _ as *const _);
            let grid = MTLSize { width: batches as u64, height: NH as u64, depth: 1 };
            let tg = MTLSize { width: 1, height: 32.min(NH as u64), depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Timed dispatch.
        let t0 = Instant::now();
        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(reach_buf.as_ref()), 0);
        enc.set_buffer(1, Some(hc_buf.as_ref()), 0);
        enc.set_buffer(2, Some(out_buf.as_ref()), 0);
        let nh32 = NH as i32;
        let b32 = batches as i32;
        enc.set_bytes(3, 4, &nh32 as *const _ as *const _);
        enc.set_bytes(4, 4, &b32 as *const _ as *const _);
        let grid = MTLSize { width: batches as u64, height: NH as u64, depth: 1 };
        let tg = MTLSize { width: 1, height: 32.min(NH as u64), depth: 1 };
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let elapsed = t0.elapsed();

        let total_tuples = tuples_per_batch * batches as u64;
        let total_ops = total_tuples * kernel_ops_per_tuple;
        let total_flops = total_tuples * kernel_flops_per_tuple;
        let secs = elapsed.as_secs_f64();
        let gops = total_ops as f64 / secs / 1e9;
        let gflops = total_flops as f64 / secs / 1e9;

        eprintln!("[batches={:>5}] threads={:>7}  elapsed={:>9.2}ms  total_tuples={:>14.3e}  achieved={:>7.2} G-mixed-ops/s ({:>5.2} GFLOPS pure-FMA)",
            batches, batches * NH, elapsed.as_secs_f64() * 1000.0,
            total_tuples as f64, gops, gflops);
    }

    eprintln!("");
    eprintln!("=== Interpretation ===");
    eprintln!("M4 Max f32 peak: 10-30 TFLOPS depending on workload.");
    eprintln!("3p K=2 production path achieved: 287 MFLOPS (kernel-launch dominated).");
    eprintln!("If the K=5 kernel hits >= 1 TFLOPS, the K=5 6p nh=50 corrected projection holds");
    eprintln!("(~minutes-per-solve with recursive exact + isomorphism + 200 iters).");
    eprintln!("If it hits ~30 TFLOPS, the budget is comfortably reached.");
    eprintln!("If it stays at ~287 MFLOPS, the kernel is overhead-bound the same way K=2 is");
    eprintln!("and the corrected projection's assumption fails.");
}
