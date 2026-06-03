// K=5 factored CFV Metal kernel — validation against CPU + throughput
// measurement against the 86 GFLOPS brute-force baseline.
//
// Two phases, in sequence (per the docket):
//   PHASE 1: validate the kernel against CPU factored ground truth to
//            float precision on the conflict-heavy non-uniform-reach
//            cases. Speed numbers on a wrong kernel mislead.
//   PHASE 2: measure achieved throughput at saturating dispatch
//            (sweep batch counts until saturated, report at saturation
//            point, not at thread-starved low end).

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::showdown::{precompute_opp_masses, OppMasses};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::buffer::MetalBuffer;
use metal::MTLSize;
use std::time::Instant;

const NUM_OPP: usize = 5;

// ---- CPU reference (same factored as validated in k_factored_cfv_general) --

fn factored_share_k2_extended(
    masses: &OppMasses,
    opp_reach: &[&[f32]],
    hand_strength: &[u16],
    h: usize,
    h_dead_mask: u64,
    oa: usize, ob: usize,
    tied_offset: u32,
) -> f64 {
    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;
    let h_str = hand_strength[h];

    let mut ba = 0.0f64; let mut ta = 0.0f64;
    let mut bb_tot = 0.0f64; let mut tb_tot = 0.0f64;
    let mut ba_pc = [0.0f64; 52]; let mut ta_pc = [0.0f64; 52];
    let mut bb_pc = [0.0f64; 52]; let mut tb_pc = [0.0f64; 52];
    let mut h_bb = 0.0f64; let mut h_tt = 0.0f64;

    for g in 0..nh {
        let gc1 = hand_cards[g * 2] as usize;
        let gc2 = hand_cards[g * 2 + 1] as usize;
        let g_m = (1u64 << gc1) | (1u64 << gc2);
        if g_m & h_dead_mask != 0 { continue; }
        let s = hand_strength[g];
        let ra = opp_reach[oa][g] as f64;
        let rb = opp_reach[ob][g] as f64;
        if ra == 0.0 && rb == 0.0 { continue; }
        if s < h_str {
            ba += ra; ba_pc[gc1] += ra; ba_pc[gc2] += ra;
            bb_tot += rb; bb_pc[gc1] += rb; bb_pc[gc2] += rb;
            h_bb += ra * rb;
        } else if s == h_str {
            ta += ra; ta_pc[gc1] += ra; ta_pc[gc2] += ra;
            tb_tot += rb; tb_pc[gc1] += rb; tb_pc[gc2] += rb;
            h_tt += ra * rb;
        }
    }

    let mut bb_edge = 0.0f64;
    let mut bt_edge = 0.0f64;
    let mut tb_edge = 0.0f64;
    let mut tt_edge = 0.0f64;
    for c in 0..52usize {
        if (1u64 << c) & h_dead_mask != 0 { continue; }
        bb_edge += ba_pc[c] * bb_pc[c];
        bt_edge += ba_pc[c] * tb_pc[c];
        tb_edge += ta_pc[c] * bb_pc[c];
        tt_edge += ta_pc[c] * tb_pc[c];
    }

    let pair_bb = ba * bb_tot - bb_edge + h_bb;
    let pair_bt = ba * tb_tot - bt_edge;
    let pair_tb = ta * bb_tot - tb_edge;
    let pair_tt = ta * tb_tot - tt_edge + h_tt;

    let t = tied_offset as f64;
    pair_bb / (1.0 + t) + pair_bt / (2.0 + t) + pair_tb / (2.0 + t) + pair_tt / (3.0 + t)
}

fn factored_share_cpu(
    masses: &OppMasses,
    opp_reach: &[&[f32]],
    hand_strength: &[u16],
    h: usize,
    h_dead_mask: u64,
    opp_indices: &[usize],
    tied_so_far: u32,
) -> f64 {
    if opp_indices.len() == 2 {
        return factored_share_k2_extended(
            masses, opp_reach, hand_strength, h, h_dead_mask,
            opp_indices[0], opp_indices[1], tied_so_far,
        );
    }

    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;
    let h_str = hand_strength[h];
    let oi = opp_indices[0];

    let mut sum = 0.0f64;
    for g in 0..nh {
        let g_m = (1u64 << hand_cards[g * 2]) | (1u64 << hand_cards[g * 2 + 1]);
        if g_m & h_dead_mask != 0 { continue; }
        let r = opp_reach[oi][g] as f64;
        if r == 0.0 { continue; }
        let s = hand_strength[g];
        if s > h_str { continue; }
        let new_tied = if s == h_str { tied_so_far + 1 } else { tied_so_far };
        sum += r * factored_share_cpu(
            masses, opp_reach, hand_strength, h, h_dead_mask | g_m,
            &opp_indices[1..], new_tied,
        );
    }
    sum
}

// ---- Test setup with three-way-shared-card structure ----------------------

fn three_way_shared_card_setup(num_opp: usize) -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize) {
    // Same configuration as in k_factored_cfv_general.rs.
    let nh = 10;
    let hand_cards: Vec<u8> = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 0, 8,
        0, 2, 2, 9, 10, 11, 12, 13, 14, 15,
    ];
    let hand_strength: Vec<u16> = vec![100, 90, 70, 80, 70, 60, 50, 50, 40, 30];
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.3 + 0.6 * ((h + oi * 3) % 7) as f32 / 7.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh)
}

// ---- Production-scale setup for throughput phase ---------------------------

fn build_nh50_inputs(num_opp: usize) -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>) {
    let nh = 50;
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
    let mut hand_strength = vec![0u16; nh];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i*2] = c1; hand_cards[i*2+1] = c2;
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        hand_strength[i] = h.evaluate_internal() as u16;
    }
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.3 + 0.6 * ((h + oi * 5) % 11) as f32 / 11.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach)
}

// ---- PHASE 1: CPU↔GPU validation -------------------------------------------

#[test]
fn k5_factored_kernel_phase1_cpu_gpu_parity() {
    let (hand_cards, hand_strength, reach, nh) = three_way_shared_card_setup(NUM_OPP);
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    // Flatten reach to [num_opp * nh].
    let reach_flat: Vec<f32> = (0..NUM_OPP).flat_map(|oi| reach[oi].iter().copied()).collect();

    let ctx = MetalContext::new().expect("Metal");
    let pipeline = ctx.create_pipeline("k5_factored_share_microbench")
        .expect("kernel not found");

    let reach_buf = MetalBuffer::from_slice(ctx.device(), &reach_flat);
    let hc_buf = MetalBuffer::from_slice(ctx.device(), &hand_cards);
    let hs_buf = MetalBuffer::from_slice(ctx.device(), &hand_strength);

    let batches = 1usize;
    let out_buf: MetalBuffer<f32> = MetalBuffer::zeros(ctx.device(), batches * nh);

    let cmd = ctx.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(reach_buf.as_ref()), 0);
    enc.set_buffer(1, Some(hc_buf.as_ref()), 0);
    enc.set_buffer(2, Some(hs_buf.as_ref()), 0);
    enc.set_buffer(3, Some(out_buf.as_ref()), 0);
    let nh32 = nh as i32; let b32 = batches as i32;
    enc.set_bytes(4, 4, &nh32 as *const _ as *const _);
    enc.set_bytes(5, 4, &b32 as *const _ as *const _);
    let grid = MTLSize { width: batches as u64, height: nh as u64, depth: 1 };
    let tg = MTLSize { width: 1, height: (nh as u64).min(32), depth: 1 };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_out = out_buf.to_vec();
    let opp_indices: Vec<usize> = (0..NUM_OPP).collect();

    eprintln!("=== K=5 factored CPU vs GPU on 3-way-shared-card setup ===");
    eprintln!("Hand structure: h=0,4,5 all contain card 0 (3-way conflict on 0)");
    eprintln!("               h=1,5,6 all contain card 2 (3-way conflict on 2)");

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut worst_h = 0;

    for h in 0..nh {
        let hc1 = hand_cards[h * 2] as usize;
        let hc2 = hand_cards[h * 2 + 1] as usize;
        let h_dead = (1u64 << hc1) | (1u64 << hc2);

        let cpu = factored_share_cpu(
            &masses, &reach_views, &hand_strength,
            h, h_dead, &opp_indices, 0,
        );
        let gpu = gpu_out[h] as f64;
        let diff = (gpu - cpu).abs();
        let scale = cpu.abs().max(1e-9);
        let rel = diff / scale;
        if diff > max_abs { max_abs = diff; worst_h = h; }
        if rel > max_rel { max_rel = rel; }

        eprintln!("h={:>2} cards=({:>2},{:>2}) s={:>3}  cpu_f64={:>10.6}  gpu_f32={:>10.6}  diff={:.3e}  rel={:.3e}",
            h, hand_cards[h*2], hand_cards[h*2+1], hand_strength[h],
            cpu, gpu, diff, rel);
    }

    eprintln!("\nK=5 factored CPU↔GPU: max_abs={:.3e} at h={}, max_rel={:.3e}",
        max_abs, worst_h, max_rel);

    // Tolerance: CPU uses f64, GPU uses f32. f32 vs f64 accumulation noise
    // at nh=10 is ~1e-6 relative.
    assert!(max_rel < 1e-5 || max_abs < 1e-5,
        "K=5 factored CPU↔GPU parity fails at h={}: max_abs={:.3e}, max_rel={:.3e}",
        worst_h, max_abs, max_rel);
}

// ---- PHASE 2: throughput measurement at nh=50 -----------------------------

#[test]
fn k5_factored_kernel_phase2_throughput_at_nh50() {
    let nh = 50usize;
    let (hand_cards, hand_strength, reach) = build_nh50_inputs(NUM_OPP);
    let reach_flat: Vec<f32> = (0..NUM_OPP).flat_map(|oi| reach[oi].iter().copied()).collect();

    let ctx = MetalContext::new().expect("Metal");
    let pipeline = ctx.create_pipeline("k5_factored_share_microbench")
        .expect("kernel not found");

    let reach_buf = MetalBuffer::from_slice(ctx.device(), &reach_flat);
    let hc_buf = MetalBuffer::from_slice(ctx.device(), &hand_cards);
    let hs_buf = MetalBuffer::from_slice(ctx.device(), &hand_strength);

    // Count actual valid (g_0..g_4) tuples per h_player for throughput math.
    // (Brute-force counts; same as the brute-force microbench.)
    fn count_tuples(hand_cards: &[u8], h: usize, num_opp: usize, nh: usize) -> u64 {
        let g_mask: Vec<u64> = (0..nh)
            .map(|g| (1u64 << hand_cards[g * 2]) | (1u64 << hand_cards[g * 2 + 1]))
            .collect();
        let h_m = g_mask[h];
        fn rec(depth: usize, k: usize, nh: usize, mask: u64, gm: &[u64]) -> u64 {
            if depth == k { return 1; }
            let mut c = 0u64;
            for g in 0..nh {
                if gm[g] & mask != 0 { continue; }
                c += rec(depth + 1, k, nh, mask | gm[g], gm);
            }
            c
        }
        rec(0, num_opp, nh, h_m, &g_mask)
    }
    let probe_h = nh / 2;
    let t_count = Instant::now();
    let per_h_brute = count_tuples(&hand_cards, probe_h, NUM_OPP, nh);
    let count_t = t_count.elapsed();
    eprintln!("[bench] Brute-force valid K=5 tuples for h={}: {} ({:?})", probe_h, per_h_brute, count_t);

    // For the FACTORED kernel, the actual structural compute is much less
    // than the brute-force tuple count: only 3 outer levels enumerated
    // (g_0, g_1, g_2), then a per-call K=2 PAIR base case in O(nh+52).
    // Expected work per h: nh^3 outer × O(nh + 52) inner ≈ 50^3 × 100 = 12.5M ops.
    let factored_per_h_estimate: u64 = (nh.pow(3) as u64) * (nh as u64 + 52);
    eprintln!("[bench] Factored per-h structural ops (nh^3 × (nh + 52) ≈ {}M)",
        factored_per_h_estimate as f64 / 1e6);

    let batch_counts = &[1usize, 10, 100, 500, 1000, 2000];

    eprintln!("\n=== PHASE 2: K=5 factored kernel throughput at nh=50 ===");
    eprintln!("(Compare against 86 GFLOPS pure-FMA brute-force baseline from k5_kernel_throughput.)");

    for &batches in batch_counts {
        let out_buf: MetalBuffer<f32> = MetalBuffer::zeros(ctx.device(), batches * nh);
        // Warm-up
        {
            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(reach_buf.as_ref()), 0);
            enc.set_buffer(1, Some(hc_buf.as_ref()), 0);
            enc.set_buffer(2, Some(hs_buf.as_ref()), 0);
            enc.set_buffer(3, Some(out_buf.as_ref()), 0);
            let nh32 = nh as i32; let b32 = batches as i32;
            enc.set_bytes(4, 4, &nh32 as *const _ as *const _);
            enc.set_bytes(5, 4, &b32 as *const _ as *const _);
            let grid = MTLSize { width: batches as u64, height: nh as u64, depth: 1 };
            let tg = MTLSize { width: 1, height: 32.min(nh as u64), depth: 1 };
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let t0 = Instant::now();
        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(reach_buf.as_ref()), 0);
        enc.set_buffer(1, Some(hc_buf.as_ref()), 0);
        enc.set_buffer(2, Some(hs_buf.as_ref()), 0);
        enc.set_buffer(3, Some(out_buf.as_ref()), 0);
        let nh32 = nh as i32; let b32 = batches as i32;
        enc.set_bytes(4, 4, &nh32 as *const _ as *const _);
        enc.set_bytes(5, 4, &b32 as *const _ as *const _);
        let grid = MTLSize { width: batches as u64, height: nh as u64, depth: 1 };
        let tg = MTLSize { width: 1, height: 32.min(nh as u64), depth: 1 };
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let elapsed = t0.elapsed();

        // Per-h structural work: ≈ nh^3 outer × (nh + 52) inner per call.
        // Sum: batches × nh × nh^3 × (nh+52) = batches × nh^4 × (nh+52).
        let total_work = (batches as u64) * (nh as u64) * factored_per_h_estimate;
        let secs = elapsed.as_secs_f64();
        let gops = total_work as f64 / secs / 1e9;

        eprintln!("[batches={:>5}] threads={:>7}  elapsed={:>9.2}ms  total_ops≈{:>11.3e}  achieved={:>7.2} G-mixed-ops/s",
            batches, batches * nh, elapsed.as_secs_f64() * 1000.0,
            total_work as f64, gops);
    }

    eprintln!("\n=== K=5 factored vs brute-force throughput comparison ===");
    eprintln!("Brute-force baseline (k5_kernel_throughput): 86 GFLOPS pure-FMA / 430 G-mixed-ops/s");
    eprintln!("Factored target (12.5M ops/h vs brute 1.28e9 ops/h ≈ 100× less compute per h):");
    eprintln!("  → Factored kernel needs >= 4.3 G-mixed-ops/s to match the brute-force RATE-per-cost.");
    eprintln!("  → If factored throughput stays ≥ brute-force baseline rate (430 G-mixed-ops/s),");
    eprintln!("     the K=5 6p nh=50 corrected projection (4 min per solve) holds or improves.");
    eprintln!("  → If factored throughput drops below brute-force (recursion/branch divergence hurts),");
    eprintln!("     the 4-min projection moves up by the ratio.");
}
