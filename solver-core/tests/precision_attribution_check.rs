// Precision attribution: confirm the 4.29e-6 worst-case from the unified
// kernel gate 3 is f32 single-precision rounding, not a kernel-specific bug.
//
// Method: same factored CFV algorithm, three variants:
//   - CPU f64 (math reference; matches brute force at f64 noise floor)
//   - CPU f32 (same operation order as GPU, in f32)
//   - GPU f32 (the unified kernel)
//
// If CPU-f32 and GPU-f32 both deviate from CPU-f64 by similar amounts,
// the gap is inherent f32 precision in the cancellation-heavy PAIR
// formulas, not a kernel bug. If CPU-f32 stays near CPU-f64 but GPU-f32
// is far off, there IS a kernel-specific issue.
//
// Apple Silicon GPUs lack native fp64 (Metal `double` is emulated or
// silently downcast), so f64-on-GPU isn't a real comparison option.
// The CPU-f32 version is the proper proxy.

#![cfg(feature = "metal")]

use solver_core::solver::showdown::{precompute_opp_masses, OppMasses};
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
    levels.sort(); levels.dedup();
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

// ---- f64 versions (math reference) -----------------------------------------

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

fn factored_share_at_level_f64(
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
        return share_k2_base_f64(
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
        sum += r * factored_share_at_level_f64(
            masses, opp_reach, hand_strength, h, h_dead | g_m,
            &opp_indices[1..], &elig[1..], new_tied);
    }
    sum
}

fn share_k2_base_f64(
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

fn factored_cfv_f64(
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
                let share = factored_share_at_level_f64(
                    masses, opp_reach, hand_strength, h, h_dead, &opp_indices, &elig, 0);
                case_c += lev.pot_l * share;
            }
        }
        cfv[h] = (static_cash - stake) * tvrp + case_c;
    }
    cfv
}

// ---- f32 versions (matching GPU operation order) ---------------------------

fn tvrp_f32(opp_reach: &[&[f32]], hand_cards: &[u8], nh: usize, h: usize, h_dead: u64) -> f32 {
    let k = opp_reach.len();
    let g_mask: Vec<u64> = (0..nh)
        .map(|g| (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1])).collect();
    fn rec(oi: usize, k: usize, nh: usize, mask: u64, r: f32,
        or: &[&[f32]], gm: &[u64], sum: &mut f32) {
        if oi == k { *sum += r; return; }
        for g in 0..nh {
            if gm[g] & mask != 0 { continue; }
            let rg = or[oi][g];
            if rg == 0.0 { continue; }
            rec(oi+1, k, nh, mask | gm[g], r * rg, or, gm, sum);
        }
    }
    let mut s = 0.0f32;
    rec(0, k, nh, h_dead, 1.0, opp_reach, &g_mask, &mut s);
    s
}

fn factored_share_at_level_f32(
    masses: &OppMasses, opp_reach: &[&[f32]], hand_strength: &[u16],
    h: usize, h_dead: u64, opp_indices: &[usize], elig: &[bool], tied_so_far: u32,
) -> f32 {
    if opp_indices.len() == 1 {
        let nh = masses.nh;
        let hand_cards = &masses.hand_cards;
        let h_str = hand_strength[h];
        let oi = opp_indices[0];
        let oi_elig = elig[0];
        let mut sum = 0.0f32;
        for g in 0..nh {
            let g_m = (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]);
            if g_m & h_dead != 0 { continue; }
            let r = opp_reach[oi][g];
            if r == 0.0 { continue; }
            let s = hand_strength[g];
            if oi_elig {
                if s > h_str { continue; }
                let t = if s == h_str { tied_so_far + 1 } else { tied_so_far };
                sum += r / (1.0 + t as f32);
            } else {
                sum += r / (1.0 + tied_so_far as f32);
            }
        }
        return sum;
    }
    if opp_indices.len() == 2 {
        return share_k2_base_f32(
            masses, opp_reach, hand_strength, h, h_dead,
            opp_indices[0], opp_indices[1], elig[0], elig[1], tied_so_far);
    }
    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;
    let h_str = hand_strength[h];
    let oi = opp_indices[0];
    let oi_elig = elig[0];
    let mut sum = 0.0f32;
    for g in 0..nh {
        let g_m = (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]);
        if g_m & h_dead != 0 { continue; }
        let r = opp_reach[oi][g];
        if r == 0.0 { continue; }
        let s = hand_strength[g];
        let new_tied = if oi_elig {
            if s > h_str { continue; }
            if s == h_str { tied_so_far + 1 } else { tied_so_far }
        } else { tied_so_far };
        sum += r * factored_share_at_level_f32(
            masses, opp_reach, hand_strength, h, h_dead | g_m,
            &opp_indices[1..], &elig[1..], new_tied);
    }
    sum
}

fn share_k2_base_f32(
    masses: &OppMasses, opp_reach: &[&[f32]], hand_strength: &[u16],
    h: usize, h_dead: u64, oa: usize, ob: usize, ea: bool, eb: bool, tied: u32,
) -> f32 {
    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;
    let h_str = hand_strength[h];
    let mut b_a=0.0f32; let mut t_a=0.0f32; let mut s_a=0.0f32;
    let mut b_b=0.0f32; let mut t_b=0.0f32; let mut s_b=0.0f32;
    let mut b_a_pc=[0.0f32; 52]; let mut t_a_pc=[0.0f32; 52]; let mut s_a_pc=[0.0f32; 52];
    let mut b_b_pc=[0.0f32; 52]; let mut t_b_pc=[0.0f32; 52]; let mut s_b_pc=[0.0f32; 52];
    let mut h_bb=0.0f32; let mut h_tt=0.0f32; let mut h_ss=0.0f32;
    for g in 0..nh {
        let gc1 = hand_cards[g*2] as usize;
        let gc2 = hand_cards[g*2+1] as usize;
        let g_m = (1u64 << gc1) | (1u64 << gc2);
        if g_m & h_dead != 0 { continue; }
        let r_a = opp_reach[oa][g];
        let r_b = opp_reach[ob][g];
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
    let mut ebb=0.0f32; let mut ebt=0.0f32; let mut etb=0.0f32; let mut ett=0.0f32;
    let mut ebe=0.0f32; let mut ete=0.0f32; let mut eeb=0.0f32; let mut eet=0.0f32;
    let mut eee=0.0f32;
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
    let t0 = tied as f32;
    match (ea, eb) {
        (true,true) => pair_bb/(1.0+t0) + pair_bt/(2.0+t0) + pair_tb/(2.0+t0) + pair_tt/(3.0+t0),
        (true,false) => pair_be/(1.0+t0) + pair_te/(2.0+t0),
        (false,true) => pair_eb/(1.0+t0) + pair_et/(2.0+t0),
        (false,false) => pair_ee/(1.0+t0),
    }
}

fn factored_cfv_f32(
    masses: &OppMasses, opp_reach: &[&[f32]], hand_cards: &[u8], hand_strength: &[u16],
    nh: usize, contributions: &[i32], fold_mask: u16, traverser: usize,
    np: usize, starting_pot: i32,
) -> Vec<f32> {
    let levels = build_level_info(contributions, fold_mask, traverser, np, starting_pot);
    let k = opp_reach.len();
    let opp_indices: Vec<usize> = (0..k).collect();
    let c_t = contributions[traverser];
    let stake = starting_pot as f32 / np as f32 + c_t as f32;
    let mut cfv = vec![0.0f32; nh];
    for h in 0..nh {
        let h_dead = (1u64 << hand_cards[h*2]) | (1u64 << hand_cards[h*2+1]);
        let tvrp = tvrp_f32(opp_reach, hand_cards, nh, h, h_dead);
        let mut static_cash = 0.0f32;
        let mut case_c = 0.0f32;
        for lev in &levels {
            if !lev.has_active_elig && lev.trav_elig {
                static_cash += lev.pot_l as f32;
            } else if !lev.has_active_elig && !lev.trav_elig {
                if lev.trav_contrib_at_lev > 0.0 { static_cash += lev.trav_contrib_at_lev as f32; }
            } else if !lev.trav_elig {
                // Case D
            } else {
                let elig: Vec<bool> = (0..k).map(|oi| (lev.elig_opps >> oi) & 1 != 0).collect();
                let share = factored_share_at_level_f32(
                    masses, opp_reach, hand_strength, h, h_dead, &opp_indices, &elig, 0);
                case_c += lev.pot_l as f32 * share;
            }
        }
        cfv[h] = (static_cash - stake) * tvrp + case_c;
    }
    cfv
}

// ---- GPU dispatch ---------------------------------------------------------

fn run_unified_gpu(
    ctx: &MetalContext, pipeline: &metal::ComputePipelineState,
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
    let stake = starting_pot as f32 / np as f32 + c_t as f32;
    let reach_buf = MetalBuffer::from_slice(ctx.device(), reach);
    let hc_buf = MetalBuffer::from_slice(ctx.device(), hand_cards);
    let hs_buf = MetalBuffer::from_slice(ctx.device(), hand_strength);
    let dummy = LevelInfoMetal { pot_l: 0.0, elig_opps: 0, trav_elig: 0, trav_contrib_at_lev: 0.0 };
    let data: Vec<LevelInfoMetal> = if levels.is_empty() { vec![dummy] } else { levels };
    let levels_buf = MetalBuffer::from_slice(ctx.device(), &data);
    let out_buf: MetalBuffer<f32> = MetalBuffer::zeros(ctx.device(), nh);
    let cmd = ctx.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(reach_buf.as_ref()), 0);
    enc.set_buffer(1, Some(hc_buf.as_ref()), 0);
    enc.set_buffer(2, Some(hs_buf.as_ref()), 0);
    enc.set_buffer(3, Some(levels_buf.as_ref()), 0);
    let nl = levels_cpu.len() as i32; let no = num_opp as i32;
    let nh32 = nh as i32; let b32 = 1i32;
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

fn make_6p_setup() -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize) {
    let nh = 8;
    let hand_cards = vec![0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15];
    let hand_strength = vec![100u16, 80, 70, 70, 60, 50, 40, 30];
    let reach: Vec<Vec<f32>> = (0..5).map(|oi| {
        (0..nh).map(|h| 0.3 + 0.6 * ((h + oi * 3) % 11) as f32 / 11.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh)
}

#[test]
fn precision_attribution_worst_case() {
    // The worst-case configuration from Gate 3: 6p, contribs [300,100,25,25,10,5],
    // p0 folded, traverser=p3. Hit 4.29e-6 max_rel.
    let (hand_cards, hand_strength, reach, nh) = make_6p_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let opp_reach: Vec<&[f32]> = (0..5).map(|oi| reach[oi].as_slice()).collect();
    let reach_flat: Vec<f32> = (0..5).flat_map(|oi| reach[oi].iter().copied()).collect();

    let np = 6usize;
    let contributions = vec![300i32, 100, 25, 25, 10, 5];
    let fold_mask = 1u16 << 0;
    let traverser = 3usize;
    let starting_pot = (np as i32) * 5;

    let cpu_f64 = factored_cfv_f64(
        &masses, &opp_reach, &hand_cards, &hand_strength, nh,
        &contributions, fold_mask, traverser, np, starting_pot);

    let cpu_f32 = factored_cfv_f32(
        &masses, &opp_reach, &hand_cards, &hand_strength, nh,
        &contributions, fold_mask, traverser, np, starting_pot);

    let ctx = MetalContext::new().expect("Metal");
    let pipeline = ctx.create_pipeline("factored_showdown_unified").unwrap();
    let gpu_f32 = run_unified_gpu(
        &ctx, &pipeline, &reach_flat, &hand_cards, &hand_strength,
        nh, 5, &contributions, fold_mask, traverser, np, starting_pot);

    eprintln!("=== Precision attribution: 6p Gate 3 worst-case ===");
    eprintln!("contribs={:?}, fold_mask={:#b}, traverser=p{}", contributions, fold_mask, traverser);
    eprintln!("h    cpu_f64           cpu_f32           gpu_f32         |f64-f32cpu|  |f64-f32gpu|  |f32cpu-f32gpu|");
    let mut max_f64_f32cpu = 0.0f64;
    let mut max_f64_f32gpu = 0.0f64;
    let mut max_f32cpu_f32gpu = 0.0f64;
    for h in 0..nh {
        let f64v = cpu_f64[h];
        let cpu32 = cpu_f32[h] as f64;
        let gpu32 = gpu_f32[h] as f64;
        let d1 = (f64v - cpu32).abs();
        let d2 = (f64v - gpu32).abs();
        let d3 = (cpu32 - gpu32).abs();
        if d1 > max_f64_f32cpu { max_f64_f32cpu = d1; }
        if d2 > max_f64_f32gpu { max_f64_f32gpu = d2; }
        if d3 > max_f32cpu_f32gpu { max_f32cpu_f32gpu = d3; }
        eprintln!("h={}  {:>14.6}  {:>14.6}  {:>14.6}  {:>10.3e}  {:>10.3e}  {:>10.3e}",
            h, f64v, cpu32, gpu32, d1, d2, d3);
    }
    eprintln!("\nmax_abs[f64 vs f32-cpu]: {:.3e}  (the inherent f32 rounding floor)",
        max_f64_f32cpu);
    eprintln!("max_abs[f64 vs f32-gpu]: {:.3e}  (what the GPU achieves)", max_f64_f32gpu);
    eprintln!("max_abs[f32-cpu vs f32-gpu]: {:.3e}  (kernel correctness — should be tiny)",
        max_f32cpu_f32gpu);

    // The verdict:
    //   If f64-vs-f32-gpu ≈ f64-vs-f32-cpu, GPU achieves the same f32 floor
    //     as the CPU. The kernel is correct; the 4.29e-6 was inherent f32
    //     rounding, not a kernel-specific bug.
    //   If f64-vs-f32-gpu >> f64-vs-f32-cpu, GPU has more error than the
    //     algorithm necessarily incurs. Real kernel issue.
    let ratio = max_f64_f32gpu / max_f64_f32cpu.max(1e-9);
    eprintln!("\nRatio gpu_error / cpu_f32_floor: {:.2}x", ratio);
    eprintln!("  ratio ≈ 1: GPU achieves the f32 algorithm floor, no kernel-specific issue");
    eprintln!("  ratio >> 1: GPU has extra error beyond f32 inherent — kernel issue");

    // Production decision: GPU-f32 is at the f32 algorithm precision floor
    // (ratio gpu_error/cpu_f32_floor < 1). CPU and GPU at f32 differ by
    // accumulator ordering only, not by algorithm correctness — neither
    // is "more right"; both are valid f32 realizations of the same math.
    // CFR convergence tolerates f32 noise at this scale (single ULPs
    // per iteration, smoothing across thousands of iterations).
    //
    // Use RELATIVE tolerance (the CFV magnitudes range from hundreds to
    // tens of thousands; absolute thresholds don't scale meaningfully).
    let mut max_rel = 0.0f64;
    for h in 0..nh {
        let f64v = cpu_f64[h];
        let gpu32 = gpu_f32[h] as f64;
        let rel = (f64v - gpu32).abs() / f64v.abs().max(1.0);
        if rel > max_rel { max_rel = rel; }
    }
    eprintln!("\nmax_rel[f64 vs f32-gpu]: {:.3e}", max_rel);
    assert!(max_rel < 1e-5,
        "GPU f32 max_rel exceeds 1e-5 (the f32 algorithm floor for this problem): {:.3e}",
        max_rel);
}
