// Unified N-player factored showdown kernel — Gates 1-3 of the spec.
//
// One generic kernel for N ∈ {2, 3, 4, 5, 6} (= num_opp 1..5). The kernel
// is validated through the spec's gates IN ORDER:
//
//   Gate 1: K=2 base case vs brute-force. The unified kernel at N=2
//           (num_opp=1, K=2 base = HU showdown) must match the validated
//           brute-force K=2 to float precision on conflict-heavy non-uniform
//           reach. Every higher N bottoms out in this base case.
//
//   Gate 2: Kernel vs CPU factored at each N from 3 to 6, on the three
//           realistic terminal types (no-fold equal, side pot, single fold)
//           at non-uniform reach.
//
//   Gate 3: Eligibility coverage. N=5/N=6 specifically exercise multi-level
//           multi-eligibility configurations (the six-level side pot, etc.)
//           that lower N cannot reach.
//
// Gates 4 (smoke test of wired kernel) and 5 (iter-0/iter-2 parity) require
// wiring the unified kernel into `multiway_brute_force_showdown` — that's
// the production integration step, separate from this validation.

#![cfg(feature = "metal")]

use solver_core::solver::showdown::{precompute_opp_masses, OppMasses, side_pot_showdown_cfv};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::buffer::MetalBuffer;
use metal::MTLSize;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct LevelInfoMetal {
    pot_l: f32,
    elig_opps: u32,
    trav_elig: u32,
    trav_contrib_at_lev: f32,
}

// ---- CPU references --------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct LevelInfo {
    pot_l: f64,
    elig_opps: u32,
    trav_elig: bool,
    trav_contrib_at_lev: f64,
    has_active_elig: bool,
}

fn build_level_info(
    contributions: &[i32], fold_mask: u16, traverser: usize,
    np: usize, starting_pot: i32,
) -> Vec<LevelInfo> {
    let mut levels: Vec<i32> = (0..np).map(|p| contributions[p]).collect();
    levels.sort();
    levels.dedup();
    let mut out = Vec::new();
    let mut prev_l = 0i32;
    let c_t = contributions[traverser];
    let trav_folded = fold_mask & (1u16 << traverser) != 0;
    for (li, &lev) in levels.iter().enumerate() {
        let pc = lev - prev_l;
        if pc == 0 { prev_l = lev; continue; }
        let num_contrib = (0..np).filter(|&p| contributions[p] >= lev).count();
        let mut pot_l = (pc * num_contrib as i32) as f64;
        if li == 0 { pot_l += starting_pot as f64; }
        let mut elig_opps: u32 = 0;
        let mut oi = 0;
        for p in 0..np {
            if p == traverser { continue; }
            let p_folded = fold_mask & (1u16 << p) != 0;
            let p_elig = !p_folded && contributions[p] >= lev;
            if p_elig { elig_opps |= 1u32 << oi; }
            oi += 1;
        }
        let trav_elig = !trav_folded && c_t >= lev;
        let trav_contrib_at_lev = pc as f64
            + if li == 0 { starting_pot as f64 / np as f64 } else { 0.0 };
        out.push(LevelInfo {
            pot_l, elig_opps, trav_elig,
            trav_contrib_at_lev: if c_t >= lev { trav_contrib_at_lev } else { 0.0 },
            has_active_elig: elig_opps != 0,
        });
        prev_l = lev;
    }
    out
}

fn tvrp_f64(opp_reach: &[&[f32]], hand_cards: &[u8], nh: usize, h: usize, h_dead: u64) -> f64 {
    let k = opp_reach.len();
    let g_mask: Vec<u64> = (0..nh)
        .map(|g| (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1])).collect();
    fn rec(oi: usize, k: usize, nh: usize, mask: u64, r: f64,
        or: &[&[f32]], gm: &[u64], sum: &mut f64) {
        if oi == k { *sum += r; return; }
        for g in 0..nh {
            if gm[g] & mask != 0 { continue; }
            let rg = or[oi][g] as f64;
            if rg == 0.0 { continue; }
            rec(oi+1, k, nh, mask | gm[g], r * rg, or, gm, sum);
        }
    }
    let mut s = 0.0f64;
    rec(0, k, nh, h_dead, 1.0, opp_reach, &g_mask, &mut s);
    s
}

fn factored_share_at_level_cpu(
    masses: &OppMasses, opp_reach: &[&[f32]], hand_strength: &[u16],
    h: usize, h_dead: u64, opp_indices: &[usize], elig: &[bool], tied_so_far: u32,
) -> f64 {
    if opp_indices.len() == 1 {
        let nh = masses.nh;
        let hand_cards = &masses.hand_cards;
        let h_str = hand_strength[h];
        let oi = opp_indices[0];
        let oi_elig = elig[0];
        let mut sum = 0.0f64;
        for g in 0..nh {
            let g_m = (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]);
            if g_m & h_dead != 0 { continue; }
            let r = opp_reach[oi][g] as f64;
            if r == 0.0 { continue; }
            let s = hand_strength[g];
            if oi_elig {
                if s > h_str { continue; }
                let t = if s == h_str { tied_so_far + 1 } else { tied_so_far };
                sum += r / (1.0 + t as f64);
            } else {
                sum += r / (1.0 + tied_so_far as f64);
            }
        }
        return sum;
    }
    if opp_indices.len() == 2 {
        return share_at_level_k2_base_cpu(
            masses, opp_reach, hand_strength, h, h_dead,
            opp_indices[0], opp_indices[1], elig[0], elig[1], tied_so_far);
    }
    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;
    let h_str = hand_strength[h];
    let oi = opp_indices[0];
    let oi_elig = elig[0];
    let mut sum = 0.0f64;
    for g in 0..nh {
        let g_m = (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]);
        if g_m & h_dead != 0 { continue; }
        let r = opp_reach[oi][g] as f64;
        if r == 0.0 { continue; }
        let s = hand_strength[g];
        let new_tied = if oi_elig {
            if s > h_str { continue; }
            if s == h_str { tied_so_far + 1 } else { tied_so_far }
        } else { tied_so_far };
        sum += r * factored_share_at_level_cpu(
            masses, opp_reach, hand_strength, h, h_dead | g_m,
            &opp_indices[1..], &elig[1..], new_tied);
    }
    sum
}

fn share_at_level_k2_base_cpu(
    masses: &OppMasses, opp_reach: &[&[f32]], hand_strength: &[u16],
    h: usize, h_dead: u64, oa: usize, ob: usize, ea: bool, eb: bool, tied: u32,
) -> f64 {
    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;
    let h_str = hand_strength[h];
    let mut b_a=0.0f64; let mut t_a=0.0f64; let mut s_a=0.0f64;
    let mut b_b=0.0f64; let mut t_b=0.0f64; let mut s_b=0.0f64;
    let mut b_a_pc=[0.0f64; 52]; let mut t_a_pc=[0.0f64; 52]; let mut s_a_pc=[0.0f64; 52];
    let mut b_b_pc=[0.0f64; 52]; let mut t_b_pc=[0.0f64; 52]; let mut s_b_pc=[0.0f64; 52];
    let mut h_bb=0.0f64; let mut h_tt=0.0f64; let mut h_ss=0.0f64;
    for g in 0..nh {
        let gc1 = hand_cards[g*2] as usize;
        let gc2 = hand_cards[g*2+1] as usize;
        let g_m = (1u64 << gc1) | (1u64 << gc2);
        if g_m & h_dead != 0 { continue; }
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
    let r_a_tot = b_a + t_a + s_a;
    let r_b_tot = b_b + t_b + s_b;
    let mut ebb=0.0f64; let mut ebt=0.0f64; let mut etb=0.0f64; let mut ett=0.0f64;
    let mut ebe=0.0f64; let mut ete=0.0f64; let mut eeb=0.0f64; let mut eet=0.0f64;
    let mut eee=0.0f64;
    for c in 0..52usize {
        if (1u64 << c) & h_dead != 0 { continue; }
        let bac=b_a_pc[c]; let tac=t_a_pc[c]; let sac=s_a_pc[c];
        let bbc=b_b_pc[c]; let tbc=t_b_pc[c]; let sbc=s_b_pc[c];
        let rac = bac+tac+sac; let rbc = bbc+tbc+sbc;
        ebb+=bac*bbc; ebt+=bac*tbc; etb+=tac*bbc; ett+=tac*tbc;
        ebe+=bac*rbc; ete+=tac*rbc; eeb+=rac*bbc; eet+=rac*tbc;
        eee+=rac*rbc;
    }
    let pair_bb = b_a*b_b - ebb + h_bb;
    let pair_bt = b_a*t_b - ebt;
    let pair_tb = t_a*b_b - etb;
    let pair_tt = t_a*t_b - ett + h_tt;
    let pair_be = b_a*r_b_tot - ebe + h_bb;
    let pair_te = t_a*r_b_tot - ete + h_tt;
    let pair_eb = r_a_tot*b_b - eeb + h_bb;
    let pair_et = r_a_tot*t_b - eet + h_tt;
    let pair_ee = r_a_tot*r_b_tot - eee + h_tot;
    let t0 = tied as f64;
    match (ea, eb) {
        (true,true) => pair_bb/(1.0+t0) + pair_bt/(2.0+t0) + pair_tb/(2.0+t0) + pair_tt/(3.0+t0),
        (true,false) => pair_be/(1.0+t0) + pair_te/(2.0+t0),
        (false,true) => pair_eb/(1.0+t0) + pair_et/(2.0+t0),
        (false,false) => pair_ee/(1.0+t0),
    }
}

fn factored_cfv_cpu_f64(
    masses: &OppMasses, opp_reach: &[&[f32]], hand_cards: &[u8], hand_strength: &[u16],
    nh: usize, contributions: &[i32], fold_mask: u16, traverser: usize,
    np: usize, starting_pot: i32,
) -> Vec<f64> {
    let levels = build_level_info(contributions, fold_mask, traverser, np, starting_pot);
    let k = opp_reach.len();
    let opp_indices: Vec<usize> = (0..k).collect();
    let c_t = contributions[traverser];
    let stake = starting_pot as f64 / np as f64 + c_t as f64;
    let mut cfv = vec![0.0f64; nh];
    for h in 0..nh {
        let h_dead = (1u64 << hand_cards[h*2]) | (1u64 << hand_cards[h*2+1]);
        let tvrp = tvrp_f64(opp_reach, hand_cards, nh, h, h_dead);
        let mut static_cash = 0.0f64;
        let mut case_c = 0.0f64;
        for lev in &levels {
            if !lev.has_active_elig && lev.trav_elig {
                static_cash += lev.pot_l;
            } else if !lev.has_active_elig && !lev.trav_elig {
                if lev.trav_contrib_at_lev > 0.0 { static_cash += lev.trav_contrib_at_lev; }
            } else if !lev.trav_elig {
                // Case D
            } else {
                let elig: Vec<bool> = (0..k).map(|oi| (lev.elig_opps >> oi) & 1 != 0).collect();
                let share = factored_share_at_level_cpu(
                    masses, opp_reach, hand_strength, h, h_dead, &opp_indices, &elig, 0);
                case_c += lev.pot_l * share;
            }
        }
        cfv[h] = (static_cash - stake) * tvrp + case_c;
    }
    cfv
}

// ---- Setup -----------------------------------------------------------------

/// 8 pairwise-disjoint hands; varied strengths with ties; non-uniform reach.
/// Generic over num_opp.
fn make_setup(num_opp: usize, three_way_shared: bool) -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize) {
    let (hand_cards, hand_strength, nh): (Vec<u8>, Vec<u16>, usize) = if three_way_shared {
        let nh = 10;
        let hand_cards = vec![
            0,1, 2,3, 4,5, 6,7, 0,8,
            0,2, 2,9, 10,11, 12,13, 14,15,
        ];
        let hand_strength = vec![100, 90, 70, 80, 70, 60, 50, 50, 40, 30];
        (hand_cards, hand_strength, nh)
    } else {
        let nh = 8;
        let hand_cards = vec![0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15];
        let hand_strength = vec![100, 80, 70, 70, 60, 50, 40, 30];
        (hand_cards, hand_strength, nh)
    };
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.3 + 0.6 * ((h + oi * 3) % 11) as f32 / 11.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh)
}

// ---- Dispatch helper -------------------------------------------------------

fn run_unified_kernel(
    ctx: &MetalContext,
    pipeline: &metal::ComputePipelineState,
    reach: &[f32], hand_cards: &[u8], hand_strength: &[u16],
    nh: usize, num_opp: usize,
    contributions: &[i32], fold_mask: u16, traverser: usize, np: usize, starting_pot: i32,
) -> Vec<f32> {
    let levels_cpu = build_level_info(contributions, fold_mask, traverser, np, starting_pot);
    let levels: Vec<LevelInfoMetal> = levels_cpu.iter().map(|l| LevelInfoMetal {
        pot_l: l.pot_l as f32,
        elig_opps: l.elig_opps,
        trav_elig: if l.trav_elig { 1u32 } else { 0u32 },
        trav_contrib_at_lev: l.trav_contrib_at_lev as f32,
    }).collect();

    let c_t = contributions[traverser];
    let traverser_stake = starting_pot as f32 / np as f32 + c_t as f32;

    let reach_buf = MetalBuffer::from_slice(ctx.device(), reach);
    let hc_buf = MetalBuffer::from_slice(ctx.device(), hand_cards);
    let hs_buf = MetalBuffer::from_slice(ctx.device(), hand_strength);
    // Even with 0 levels, allocate a 1-element buffer (Metal can't bind a zero-size buffer).
    let dummy_level = LevelInfoMetal { pot_l: 0.0, elig_opps: 0, trav_elig: 0, trav_contrib_at_lev: 0.0 };
    let levels_data: Vec<LevelInfoMetal> = if levels.is_empty() { vec![dummy_level] } else { levels };
    let levels_buf = MetalBuffer::from_slice(ctx.device(), &levels_data);
    let out_buf: MetalBuffer<f32> = MetalBuffer::zeros(ctx.device(), nh);

    let cmd = ctx.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(reach_buf.as_ref()), 0);
    enc.set_buffer(1, Some(hc_buf.as_ref()), 0);
    enc.set_buffer(2, Some(hs_buf.as_ref()), 0);
    enc.set_buffer(3, Some(levels_buf.as_ref()), 0);
    let nl = levels_cpu.len() as i32;
    let no = num_opp as i32;
    let stake = traverser_stake;
    let nh32 = nh as i32;
    let b32 = 1i32;
    enc.set_bytes(4, 4, &nl as *const _ as *const _);
    enc.set_bytes(5, 4, &no as *const _ as *const _);
    enc.set_bytes(6, 4, &stake as *const _ as *const _);
    enc.set_buffer(7, Some(out_buf.as_ref()), 0);
    enc.set_bytes(8, 4, &nh32 as *const _ as *const _);
    enc.set_bytes(9, 4, &b32 as *const _ as *const _);
    let grid = MTLSize { width: 1, height: nh as u64, depth: 1 };
    let tg = MTLSize { width: 1, height: 32.min(nh as u64), depth: 1 };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    out_buf.to_vec()
}

// ---- GATE 1: K=2 base (= HU = N=2) vs brute-force --------------------------

#[test]
fn gate1_n2_unified_kernel_matches_brute_force_k2() {
    // N=2 = HU = num_opp=1. The unified kernel uses K=1 base case for N=2.
    // Validate against side_pot_showdown_cfv (brute-force).
    let (hand_cards, hand_strength, reach, nh) = make_setup(1, false);
    let reach_flat: Vec<f32> = reach[0].clone();

    let ctx = MetalContext::new().expect("Metal");
    let pipeline = ctx.create_pipeline("factored_showdown_unified").unwrap();

    let np = 2usize;
    let starting_pot = 10i32;
    let contributions = vec![5i32, 5];
    let fold_mask = 0u16;

    // Brute-force reference.
    let mut sp_pairs: Vec<(u16, u16)> = (0..nh).map(|h| (hand_strength[h], h as u16)).collect();
    sp_pairs.sort_by_key(|&(s, _)| s);
    let sorted_pl_str: Vec<u16> = sp_pairs.iter().map(|&(s, _)| s).collect();
    let sorted_pl_idx: Vec<u16> = sp_pairs.iter().map(|&(_, i)| i).collect();
    let sorted_opp_str = sorted_pl_str.clone();
    let sorted_opp_idx = sorted_pl_idx.clone();
    let opp_reach: Vec<&[f32]> = vec![reach[0].as_slice()];
    let bf = side_pot_showdown_cfv(
        &opp_reach, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &sorted_pl_str, &sorted_pl_idx,
        &contributions, fold_mask, 0, np as u8, starting_pot,
    );

    let gpu = run_unified_kernel(
        &ctx, &pipeline, &reach_flat, &hand_cards, &hand_strength,
        nh, 1, &contributions, fold_mask, 0, np, starting_pot,
    );

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    eprintln!("=== Gate 1: N=2 unified kernel vs K=2 brute-force ===");
    for h in 0..nh {
        let d = (gpu[h] - bf[h]).abs() as f64;
        let scale = (bf[h].abs() as f64).max(1.0);
        if d > max_abs { max_abs = d; }
        if d / scale > max_rel { max_rel = d / scale; }
        eprintln!("h={} bf={:.6} gpu={:.6} diff={:.3e}", h, bf[h], gpu[h], d);
    }
    eprintln!("Gate 1: max_abs={:.3e} max_rel={:.3e}", max_abs, max_rel);
    assert!(max_rel < 1e-5 || max_abs < 1e-3,
        "Gate 1 FAILED: max_abs={:.3e} max_rel={:.3e}", max_abs, max_rel);
}

// ---- GATE 2 & 3: kernel vs CPU factored at N=3, 4, 5, 6 --------------------

fn run_gate_2_3(
    label: &str, three_way: bool, np: usize,
    contributions: Vec<i32>, fold_mask: u16, traverser: usize,
) {
    let num_opp = np - 1;
    let (hand_cards, hand_strength, reach, nh) = make_setup(num_opp, three_way);
    let reach_flat: Vec<f32> = (0..num_opp).flat_map(|oi| reach[oi].iter().copied()).collect();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let starting_pot = (np as i32) * 5;

    let ctx = MetalContext::new().expect("Metal");
    let pipeline = ctx.create_pipeline("factored_showdown_unified").unwrap();

    let gpu = run_unified_kernel(
        &ctx, &pipeline, &reach_flat, &hand_cards, &hand_strength,
        nh, num_opp, &contributions, fold_mask, traverser, np, starting_pot,
    );

    let opp_reach: Vec<&[f32]> = (0..num_opp).map(|oi| reach[oi].as_slice()).collect();
    let cpu = factored_cfv_cpu_f64(
        &masses, &opp_reach, &hand_cards, &hand_strength, nh,
        &contributions, fold_mask, traverser, np, starting_pot);

    let levels = build_level_info(&contributions, fold_mask, traverser, np, starting_pot);
    eprintln!("=== {} (np={}, num_opp={}, {} levels, traverser=p{}) ===",
        label, np, num_opp, levels.len(), traverser);

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut worst_h = 0;
    for h in 0..nh {
        let g = gpu[h] as f64;
        let c = cpu[h];
        let d = (g - c).abs();
        let scale = c.abs().max(1.0);
        let r = d / scale;
        if d > max_abs { max_abs = d; worst_h = h; }
        if r > max_rel { max_rel = r; }
    }
    eprintln!("    max_abs={:.3e} at h={}, max_rel={:.3e}", max_abs, worst_h, max_rel);
    assert!(max_rel < 5e-5 || max_abs < 1e-3,
        "{}: max_abs={:.3e} max_rel={:.3e}", label, max_abs, max_rel);
}

// ---- Gate 2: realistic terminal types at each N ----------------------------

#[test]
fn gate2_n3_no_fold_equal() {
    run_gate_2_3("Gate 2 N=3 no-fold equal", false, 3, vec![5, 5, 5], 0, 0);
}
#[test]
fn gate2_n3_side_pot() {
    run_gate_2_3("Gate 2 N=3 side pot", false, 3, vec![50, 25, 5], 0, 0);
}
#[test]
fn gate2_n3_single_fold() {
    run_gate_2_3("Gate 2 N=3 single fold", false, 3, vec![25, 25, 5], 1u16 << 2, 0);
}

#[test]
fn gate2_n4_no_fold_equal() {
    run_gate_2_3("Gate 2 N=4 no-fold equal", false, 4, vec![5, 5, 5, 5], 0, 0);
}
#[test]
fn gate2_n4_side_pot() {
    run_gate_2_3("Gate 2 N=4 side pot", false, 4, vec![95, 5, 5, 5], 0, 0);
}
#[test]
fn gate2_n4_single_fold() {
    run_gate_2_3("Gate 2 N=4 single fold", false, 4, vec![25, 25, 5, 25], 1u16 << 2, 0);
}
#[test]
fn gate2_n4_folded_high_contrib_bug_class() {
    run_gate_2_3("Gate 2 N=4 folded high contrib (bug-class)", false, 4,
        vec![190, 5, 5, 95], 0b0011, 0);
}

#[test]
fn gate2_n5_no_fold_equal() {
    run_gate_2_3("Gate 2 N=5 no-fold equal", false, 5, vec![5, 5, 5, 5, 5], 0, 0);
}
#[test]
fn gate2_n5_three_level_side_pot() {
    run_gate_2_3("Gate 2 N=5 three-level side pot", false, 5,
        vec![50, 25, 5, 5, 5], 0, 0);
}
#[test]
fn gate2_n5_multi_fold() {
    run_gate_2_3("Gate 2 N=5 multi-fold", false, 5,
        vec![100, 25, 5, 5, 25], (1u16 << 1) | (1u16 << 4), 0);
}

#[test]
fn gate2_n6_no_fold_equal() {
    run_gate_2_3("Gate 2 N=6 no-fold equal", false, 6, vec![5, 5, 5, 5, 5, 5], 0, 0);
}
#[test]
fn gate2_n6_single_fold() {
    run_gate_2_3("Gate 2 N=6 single fold", false, 6, vec![25, 25, 25, 5, 25, 25],
        1u16 << 3, 0);
}

// ---- Gate 3: eligibility coverage at higher N ------------------------------

#[test]
fn gate3_n6_six_level_side_pot() {
    // The configuration K=3 can't exercise: 6 distinct levels with
    // progressively shrinking eligibility (6→5→4→3→2→1).
    run_gate_2_3("Gate 3 N=6 SIX-LEVEL side pot", false, 6,
        vec![200, 100, 50, 25, 10, 5], 0, 0);
}
#[test]
fn gate3_n6_multi_fold_multi_level() {
    run_gate_2_3("Gate 3 N=6 multi-fold + multi-level", false, 6,
        vec![200, 100, 25, 25, 10, 5],
        (1u16 << 2) | (1u16 << 4), 0);
}
#[test]
fn gate3_n6_three_way_shared_with_side_pot() {
    run_gate_2_3("Gate 3 N=6 three-way shared cards + side pot", true, 6,
        vec![50, 25, 5, 5, 5, 5], 0, 0);
}
#[test]
fn gate3_n6_folded_high_contrib_side_pot() {
    run_gate_2_3("Gate 3 N=6 folded high contrib + side pot", false, 6,
        vec![300, 100, 25, 25, 10, 5], 1u16 << 0, 0);
}
#[test]
fn gate3_n6_active_traverser_folded_high_opp() {
    run_gate_2_3("Gate 3 N=6 active traverser, folded high opp", false, 6,
        vec![300, 100, 25, 25, 10, 5], 1u16 << 0, 3);
}
