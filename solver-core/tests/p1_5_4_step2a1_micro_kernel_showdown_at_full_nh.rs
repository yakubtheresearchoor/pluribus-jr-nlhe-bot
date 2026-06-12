// Step 2.A.1 bug localization (micro-kernel level): direct showdown parity
// at nh=1176 with hand-crafted controlled inputs.
//
// The per-stage parity (p1_5_4_step2a1_per_stage_parity_at_full_nh.rs)
// pinpoints the divergence to bottom_up_zone(River) at terminal nodes.
// The terminal handler calls multiway_brute_force_showdown. This test
// removes ALL the tree/reach/strategy upstream and calls the showdown
// helper directly via the debug_brute_force_showdown kernel, with inputs
// reproducible deterministically. Any divergence here is in the helper
// itself.
//
// Inputs (hand-crafted, deterministic):
//   nh = 1176 (the smallest scale where the bug surfaces)
//   np = 2 (HU)
//   traverser = 0
//   hand_cards: each hand is (2*i % 52, (2*i+1) % 52) — distinct simple
//     pairs covering the deck. Some are self-blocking; we filter.
//   strengths: each hand has unique strength = i + 1 (so no ties).
//   opp_reach: opp_reach[h] = (h + 1) as f32 / nh (smooth non-uniform).
//   contributions: [10, 10] (equal stakes, no side pots).
//   fold_mask = 0 (no folds — pure showdown).
//   starting_pot = 20.
//
// Compare GPU output to CPU side_pot_showdown_cfv. Bit-exact gate.

#![cfg(feature = "metal")]

use metal::MTLSize;
use solver_core::card::index_to_card_pair;
use solver_core::gpu_metal::buffer::MetalBuffer;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::showdown::side_pot_showdown_cfv;

#[repr(C)]
#[derive(Clone, Copy)]
struct DebugBruteForceParams {
    nh: i32,
    np: i32,
    traverser: i32,
    starting_pot: i32,
    fold_mask: u16,
    _pad: u16,
    rake_rate: f32,
    rake_cap: f32,
    flop_seen: i32,
}

fn gpu_brute_force(
    ctx: &MetalContext,
    nh: usize, np: usize, traverser: usize,
    starting_pot: i32, fold_mask: u16,
    opp_reach: &[f32], contributions: &[i32],
    hand_cards: &[u8], pl_str: &[u16], pl_idx: &[u16],
) -> Vec<f32> {
    let device = ctx.device();
    let pipeline = ctx.create_pipeline("debug_brute_force_showdown")
        .expect("debug pipeline");

    let d_output = MetalBuffer::<f32>::zeros(device, nh);
    let d_opp_reach = MetalBuffer::from_slice(device, opp_reach);
    let d_contributions = MetalBuffer::from_slice(device, contributions);
    let d_hand_cards = MetalBuffer::from_slice(device, hand_cards);
    let d_pl_str = MetalBuffer::from_slice(device, pl_str);
    let d_pl_idx = MetalBuffer::from_slice(device, pl_idx);

    let params = DebugBruteForceParams {
        nh: nh as i32, np: np as i32, traverser: traverser as i32,
        starting_pot, fold_mask, _pad: 0,
        rake_rate: 0.0, rake_cap: 0.0, flop_seen: 0,
    };
    let d_params = MetalBuffer::from_slice(device, &[params]);

    let cmd = ctx.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(d_output.as_ref()), 0);
    enc.set_buffer(1, Some(d_opp_reach.as_ref()), 0);
    enc.set_buffer(2, Some(d_contributions.as_ref()), 0);
    enc.set_buffer(3, Some(d_hand_cards.as_ref()), 0);
    enc.set_buffer(4, Some(d_pl_str.as_ref()), 0);
    enc.set_buffer(5, Some(d_pl_idx.as_ref()), 0);
    enc.set_buffer(6, Some(d_params.as_ref()), 0);

    let grid = MTLSize { width: 1, height: 1, depth: 1 };
    let tg = MTLSize { width: 1, height: 1, depth: 1 };
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    d_output.to_vec()
}

fn make_sorted(strengths: &[u16]) -> (Vec<u16>, Vec<u16>) {
    let nh = strengths.len();
    let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (strengths[h], h as u16)).collect();
    items.sort_by_key(|&(s, _)| s);
    let mut s_str = vec![0u16; nh];
    let mut s_idx = vec![0u16; nh];
    for i in 0..nh {
        s_str[i] = items[i].0;
        s_idx[i] = items[i].1;
    }
    (s_str, s_idx)
}

fn micro_parity_at_nh(nh_target: usize) -> (usize, f32, f32) {
    // Use the first nh_target valid hand indices (decoded to (c1, c2)).
    let mut hand_cards = vec![0u8; nh_target * 2];
    for i in 0..nh_target {
        let (c1, c2) = index_to_card_pair(i);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
    }
    let strengths: Vec<u16> = (0..nh_target).map(|i| (i as u16) + 1).collect();
    let (pl_str, pl_idx) = make_sorted(&strengths);

    let np = 2usize;
    let num_opp = np - 1;
    let opp_reach: Vec<f32> = (0..num_opp * nh_target)
        .map(|i| ((i % nh_target) as f32 + 1.0) / nh_target as f32)
        .collect();
    let contributions = vec![10i32, 10];
    let fold_mask = 0u16;
    let starting_pot = 20i32;
    let traverser = 0usize;

    let opp_reach_per_opp: Vec<Vec<f32>> = (0..num_opp)
        .map(|oi| opp_reach[oi * nh_target..(oi + 1) * nh_target].to_vec())
        .collect();
    let opp_reach_views: Vec<&[f32]> = opp_reach_per_opp.iter().map(|v| v.as_slice()).collect();
    let mut sorted_opp_str = Vec::with_capacity(num_opp * nh_target);
    let mut sorted_opp_idx = Vec::with_capacity(num_opp * nh_target);
    for _ in 0..num_opp {
        sorted_opp_str.extend_from_slice(&pl_str);
        sorted_opp_idx.extend_from_slice(&pl_idx);
    }

    let cpu_cfv = side_pot_showdown_cfv(
        &opp_reach_views, &hand_cards, nh_target,
        &sorted_opp_str, &sorted_opp_idx,
        &pl_str, &pl_idx,
        &contributions, fold_mask, traverser, np as u8, starting_pot,
    );

    let ctx = MetalContext::new().expect("Metal");
    let gpu_cfv = gpu_brute_force(
        &ctx, nh_target, np, traverser, starting_pot, fold_mask,
        &opp_reach, &contributions, &hand_cards, &pl_str, &pl_idx,
    );

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut nonzero_diffs = 0usize;
    for h in 0..nh_target {
        let d = (cpu_cfv[h] - gpu_cfv[h]).abs();
        if d > 1e-9 { nonzero_diffs += 1; }
        if d > max_abs { max_abs = d; }
        let scale = cpu_cfv[h].abs().max(gpu_cfv[h].abs());
        if scale > 0.01 {
            let rel = d / scale;
            if rel > max_rel { max_rel = rel; }
        }
    }

    eprintln!("nh={:>4}: max_abs = {:.6e}, max_rel = {:.4}%, nonzero_diffs = {} / {}",
              nh_target, max_abs, max_rel * 100.0, nonzero_diffs, nh_target);
    if max_abs > 1e-3 {
        eprintln!("  CPU[0..4]: {:?}", &cpu_cfv[0..4.min(nh_target)]);
        eprintln!("  GPU[0..4]: {:?}", &gpu_cfv[0..4.min(nh_target)]);
        // First 5 diverging h values
        let mut shown = 0;
        for h in 0..nh_target {
            if (cpu_cfv[h] - gpu_cfv[h]).abs() > 1e-3 && shown < 5 {
                eprintln!("  h={:4}: CPU={:.4} GPU={:.4} d={:.4}",
                          h, cpu_cfv[h], gpu_cfv[h], (cpu_cfv[h] - gpu_cfv[h]).abs());
                shown += 1;
            }
        }
    }
    (nonzero_diffs, max_abs, max_rel)
}

#[test]
#[ignore = "Step 2.A.1 micro-kernel: direct showdown parity scan across nh. Run on demand."]
fn micro_kernel_showdown_parity_scan() {
    eprintln!("\n========================================================================");
    eprintln!("=== Step 2.A.1 micro-kernel: showdown parity scan across nh         ===");
    eprintln!("========================================================================\n");

    // Scan to find at what nh the bug first appears.
    let mut last_pass_nh = 0;
    let mut first_fail_nh: Option<usize> = None;
    for &nh in &[4usize, 16, 64, 128, 256, 512, 1024, 1176] {
        let (_diffs, max_abs, _max_rel) = micro_parity_at_nh(nh);
        if max_abs < 1e-3 {
            last_pass_nh = nh;
        } else if first_fail_nh.is_none() {
            first_fail_nh = Some(nh);
        }
    }

    eprintln!("\n========================================================================");
    eprintln!("=== Boundary localization                                            ===");
    eprintln!("========================================================================");
    eprintln!("  Last nh that passed: {}", last_pass_nh);
    if let Some(fail) = first_fail_nh {
        eprintln!("  First nh that failed: {}", fail);
        eprintln!("  Bug appears in the range ({}, {}].", last_pass_nh, fail);
        // Bracketing tells us:
        // - If fail/last is exactly 2× (e.g. 16 → 32), suggests a power-of-2 limit
        //   in the kernel (register file, threadgroup memory, indexing bound).
        // - Gradual degradation suggests accumulation error.
    } else {
        eprintln!("  All scales passed — the bug is in bottom_up_zone integration, not the helper.");
    }
}
