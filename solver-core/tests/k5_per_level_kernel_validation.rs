// K=5 per-level factored share Metal kernel — validation against CPU.
//
// Validates the GPU kernel against the f64 CPU per-level factored share
// at the multi-eligibility configurations from the K=5 CPU validation:
// six-level-style patterns, multi-fold mixed eligibility, and the
// three-way-shared-card conflict pattern combined with eligibility flags.
// CPU uses f64; GPU uses f32; expect ~1e-5 to 1e-7 relative tolerance.

#![cfg(feature = "metal")]

use solver_core::solver::showdown::{precompute_opp_masses, OppMasses};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::buffer::MetalBuffer;
use metal::MTLSize;

// CPU reference (same as k4_k5 validation).
fn factored_share_at_level(
    masses: &OppMasses,
    opp_reach: &[&[f32]],
    hand_strength: &[u16],
    h: usize,
    h_dead_mask: u64,
    opp_indices: &[usize],
    elig: &[bool],
    tied_so_far: u32,
) -> f64 {
    if opp_indices.len() == 2 {
        return share_at_level_k2_base(
            masses, opp_reach, hand_strength, h, h_dead_mask,
            opp_indices[0], opp_indices[1], elig[0], elig[1], tied_so_far,
        );
    }
    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;
    let h_str = hand_strength[h];
    let oi = opp_indices[0];
    let oi_elig = elig[0];
    let mut sum = 0.0f64;
    for g in 0..nh {
        let g_m = (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]);
        if g_m & h_dead_mask != 0 { continue; }
        let r = opp_reach[oi][g] as f64;
        if r == 0.0 { continue; }
        let s = hand_strength[g];
        let new_tied = if oi_elig {
            if s > h_str { continue; }
            if s == h_str { tied_so_far + 1 } else { tied_so_far }
        } else {
            tied_so_far
        };
        sum += r * factored_share_at_level(
            masses, opp_reach, hand_strength, h, h_dead_mask | g_m,
            &opp_indices[1..], &elig[1..], new_tied,
        );
    }
    sum
}

fn share_at_level_k2_base(
    masses: &OppMasses,
    opp_reach: &[&[f32]],
    hand_strength: &[u16],
    h: usize,
    h_dead_mask: u64,
    oa: usize, ob: usize,
    ea: bool, eb: bool,
    tied_offset: u32,
) -> f64 {
    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;
    let h_str = hand_strength[h];
    let mut b_a = 0.0f64; let mut t_a = 0.0f64; let mut s_a = 0.0f64;
    let mut b_b = 0.0f64; let mut t_b = 0.0f64; let mut s_b = 0.0f64;
    let mut b_a_pc = [0.0f64; 52]; let mut t_a_pc = [0.0f64; 52]; let mut s_a_pc = [0.0f64; 52];
    let mut b_b_pc = [0.0f64; 52]; let mut t_b_pc = [0.0f64; 52]; let mut s_b_pc = [0.0f64; 52];
    let mut h_bb = 0.0f64; let mut h_tt = 0.0f64; let mut h_ss = 0.0f64;
    for g in 0..nh {
        let gc1 = hand_cards[g*2] as usize;
        let gc2 = hand_cards[g*2+1] as usize;
        let g_m = (1u64 << gc1) | (1u64 << gc2);
        if g_m & h_dead_mask != 0 { continue; }
        let r_a = opp_reach[oa][g] as f64;
        let r_b = opp_reach[ob][g] as f64;
        if r_a == 0.0 && r_b == 0.0 { continue; }
        let s = hand_strength[g];
        if s < h_str {
            b_a += r_a; b_a_pc[gc1] += r_a; b_a_pc[gc2] += r_a;
            b_b += r_b; b_b_pc[gc1] += r_b; b_b_pc[gc2] += r_b;
            h_bb += r_a * r_b;
        } else if s == h_str {
            t_a += r_a; t_a_pc[gc1] += r_a; t_a_pc[gc2] += r_a;
            t_b += r_b; t_b_pc[gc1] += r_b; t_b_pc[gc2] += r_b;
            h_tt += r_a * r_b;
        } else {
            s_a += r_a; s_a_pc[gc1] += r_a; s_a_pc[gc2] += r_a;
            s_b += r_b; s_b_pc[gc1] += r_b; s_b_pc[gc2] += r_b;
            h_ss += r_a * r_b;
        }
    }
    let h_tot = h_bb + h_tt + h_ss;
    let r_a = b_a + t_a + s_a;
    let r_b = b_b + t_b + s_b;
    let mut edge_bb = 0.0f64; let mut edge_bt = 0.0f64;
    let mut edge_tb = 0.0f64; let mut edge_tt = 0.0f64;
    let mut edge_be = 0.0f64; let mut edge_te = 0.0f64;
    let mut edge_eb = 0.0f64; let mut edge_et = 0.0f64;
    let mut edge_ee = 0.0f64;
    for c in 0..52usize {
        if (1u64 << c) & h_dead_mask != 0 { continue; }
        let bac = b_a_pc[c]; let tac = t_a_pc[c]; let sac = s_a_pc[c];
        let bbc = b_b_pc[c]; let tbc = t_b_pc[c]; let sbc = s_b_pc[c];
        let rac = bac + tac + sac;
        let rbc = bbc + tbc + sbc;
        edge_bb += bac * bbc; edge_bt += bac * tbc;
        edge_tb += tac * bbc; edge_tt += tac * tbc;
        edge_be += bac * rbc; edge_te += tac * rbc;
        edge_eb += rac * bbc; edge_et += rac * tbc;
        edge_ee += rac * rbc;
    }
    let pair_bb = b_a * b_b - edge_bb + h_bb;
    let pair_bt = b_a * t_b - edge_bt;
    let pair_tb = t_a * b_b - edge_tb;
    let pair_tt = t_a * t_b - edge_tt + h_tt;
    let pair_be = b_a * r_b - edge_be + h_bb;
    let pair_te = t_a * r_b - edge_te + h_tt;
    let pair_eb = r_a * b_b - edge_eb + h_bb;
    let pair_et = r_a * t_b - edge_et + h_tt;
    let pair_ee = r_a * r_b - edge_ee + h_tot;
    let t0 = tied_offset as f64;
    match (ea, eb) {
        (true, true) => {
            pair_bb / (1.0 + t0) + pair_bt / (2.0 + t0) + pair_tb / (2.0 + t0) + pair_tt / (3.0 + t0)
        }
        (true, false) => pair_be / (1.0 + t0) + pair_te / (2.0 + t0),
        (false, true) => pair_eb / (1.0 + t0) + pair_et / (2.0 + t0),
        (false, false) => pair_ee / (1.0 + t0),
    }
}

fn make_6p_setup() -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize) {
    let nh = 8;
    let hand_cards: Vec<u8> = vec![
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    ];
    let hand_strength: Vec<u16> = vec![100, 80, 70, 70, 60, 50, 40, 30];
    let num_opp = 5;
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.4 + 0.5 * ((h + oi * 3) % 7) as f32 / 7.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh)
}

fn make_6p_three_way_setup() -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize) {
    let nh = 10;
    let hand_cards: Vec<u8> = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 0, 8,
        0, 2, 2, 9, 10, 11, 12, 13, 14, 15,
    ];
    let hand_strength: Vec<u16> = vec![100, 90, 70, 80, 70, 60, 50, 50, 40, 30];
    let num_opp = 5;
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.3 + 0.6 * ((h + oi * 3) % 7) as f32 / 7.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh)
}

fn run_test(label: &str, three_way: bool, elig_bits: u32, tied_offset: u32) -> (f64, f64) {
    let (hand_cards, hand_strength, reach, nh) =
        if three_way { make_6p_three_way_setup() } else { make_6p_setup() };
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let reach_flat: Vec<f32> = (0..5).flat_map(|oi| reach[oi].iter().copied()).collect();

    let ctx = MetalContext::new().expect("Metal");
    let pipeline = ctx.create_pipeline("k5_per_level_share_microbench")
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
    enc.set_bytes(6, 4, &elig_bits as *const _ as *const _);
    enc.set_bytes(7, 4, &tied_offset as *const _ as *const _);
    let grid = MTLSize { width: batches as u64, height: nh as u64, depth: 1 };
    let tg = MTLSize { width: 1, height: 32.min(nh as u64), depth: 1 };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu = out_buf.to_vec();
    let opp_indices: Vec<usize> = (0..5).collect();
    let elig: Vec<bool> = (0..5).map(|i| (elig_bits >> i) & 1 != 0).collect();

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut worst_h = 0;
    for h in 0..nh {
        let h_dead = (1u64 << hand_cards[h*2]) | (1u64 << hand_cards[h*2+1]);
        let cpu = factored_share_at_level(
            &masses, &reach_views, &hand_strength,
            h, h_dead, &opp_indices, &elig, tied_offset,
        );
        let gpu_v = gpu[h] as f64;
        let diff = (gpu_v - cpu).abs();
        let scale = cpu.abs().max(1.0);
        let rel = diff / scale;
        if diff > max_abs { max_abs = diff; worst_h = h; }
        if rel > max_rel { max_rel = rel; }
    }
    eprintln!("{}: elig={:05b} max_abs={:.3e} max_rel={:.3e} at h={}",
        label, elig_bits, max_abs, max_rel, worst_h);
    (max_abs, max_rel)
}

#[test]
fn k5_per_level_kernel_matches_cpu_all_eligible() {
    let (_, max_rel) = run_test("all-eligible (11111)", false, 0b11111, 0);
    assert!(max_rel < 1e-5, "K=5 GPU all-eligible: max_rel={:.3e}", max_rel);
}

#[test]
fn k5_per_level_kernel_matches_cpu_progressive_shrink() {
    // The configuration the docket called out: progressively shrinking
    // eligibility from each level. Test eligibility patterns covering
    // (5→4→3→2→1) opps eligible.
    let configs: &[(&str, u32)] = &[
        ("4 elig (11110)", 0b11110),
        ("3 elig (11100)", 0b11100),
        ("2 elig (11000)", 0b11000),
        ("1 elig (10000)", 0b10000),
        ("0 elig (00000)", 0b00000),
    ];
    for (label, bits) in configs {
        let (_, max_rel) = run_test(label, false, *bits, 0);
        assert!(max_rel < 1e-5, "K=5 {}: max_rel={:.3e}", label, max_rel);
    }
}

#[test]
fn k5_per_level_kernel_matches_cpu_mixed_eligibility() {
    // Non-contiguous eligibility (some E, some I interleaved) — exercises
    // the per-opp-index branching that contiguous patterns don't.
    let configs: &[(&str, u32)] = &[
        ("EIIEI (10110)", 0b10110),
        ("IEIEE (11010)", 0b11010),
        ("IIEEI (01100)", 0b01100),
        ("EIEEI (10110)", 0b10110),
    ];
    for (label, bits) in configs {
        let (_, max_rel) = run_test(label, false, *bits, 0);
        assert!(max_rel < 1e-5, "K=5 {}: max_rel={:.3e}", label, max_rel);
    }
}

#[test]
fn k5_per_level_kernel_matches_cpu_three_way_shared_card() {
    // The three-way-shared-card configuration that broke the matching
    // formula — exercised here at K=5 GPU with several eligibility patterns.
    let configs: &[(&str, u32)] = &[
        ("3way all-eligible", 0b11111),
        ("3way 4 elig", 0b11110),
        ("3way 3 elig + 1 mixed", 0b10110),
        ("3way 2 elig", 0b11000),
    ];
    for (label, bits) in configs {
        let (_, max_rel) = run_test(label, true, *bits, 0);
        assert!(max_rel < 1e-5, "K=5 {}: max_rel={:.3e}", label, max_rel);
    }
}

#[test]
fn k5_per_level_kernel_matches_cpu_nonzero_tied_offset() {
    // Non-zero tied_offset_in — exercises the case where a previous level's
    // recursion has already incremented the tied count.
    let (_, max_rel) = run_test("all-elig t0=2", false, 0b11111, 2);
    assert!(max_rel < 1e-5, "K=5 nonzero tied: max_rel={:.3e}", max_rel);
}
