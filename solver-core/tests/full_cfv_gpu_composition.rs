// Full per-terminal CFV composed on GPU via the validated K=5 per-level
// share kernel. Uses the SAME kernel dispatched at multiple elig_opps
// values: elig_opps=0 gives TVRP; per-level elig_opps gives the Case C
// shares. Host orchestrates the composition.
//
// Validation: full CFV (GPU f32, composed) vs CPU full CFV (f64) on
// realistic terminal types:
//   - 4-player (K=3 inner): no-fold equal, side pot, single fold,
//     folded-high-contrib
//   - 6-player (K=5 inner): six-level side pot, multi-fold mixed
//     eligibility, three-way-shared-card + side pot
//
// Tolerance: f32 vs f64 noise — expect 1e-4 to 1e-6 max-abs scaled
// against the CFV magnitudes (which can be hundreds-to-thousands).

#![cfg(feature = "metal")]

use solver_core::solver::showdown::{precompute_opp_masses, OppMasses};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::buffer::MetalBuffer;
use metal::MTLSize;

// ---- CPU reference (full CFV in f64) ---------------------------------------

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

fn factored_share_at_level(
    masses: &OppMasses, opp_reach: &[&[f32]], hand_strength: &[u16],
    h: usize, h_dead: u64, opp_indices: &[usize], elig: &[bool], tied_so_far: u32,
) -> f64 {
    if opp_indices.len() == 2 {
        return share_at_level_k2_base(
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
        sum += r * factored_share_at_level(
            masses, opp_reach, hand_strength, h, h_dead | g_m,
            &opp_indices[1..], &elig[1..], new_tied);
    }
    sum
}

fn share_at_level_k2_base(
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
                let share = factored_share_at_level(
                    masses, opp_reach, hand_strength, h, h_dead, &opp_indices, &elig, 0);
                case_c += lev.pot_l * share;
            }
        }
        cfv[h] = (static_cash - stake) * tvrp + case_c;
    }
    cfv
}

// ---- GPU host-orchestrated full CFV composition ----------------------------

/// Dispatch the K=5 per-level share kernel once with a given elig_opps and
/// tied_offset, return the per-h share vector.
fn dispatch_share_kernel(
    ctx: &MetalContext,
    pipeline: &metal::ComputePipelineState,
    reach_buf: &MetalBuffer<f32>,
    hc_buf: &MetalBuffer<u8>,
    hs_buf: &MetalBuffer<u16>,
    nh: usize, elig_opps: u32, tied_offset: u32,
) -> Vec<f32> {
    let out: MetalBuffer<f32> = MetalBuffer::zeros(ctx.device(), nh);
    let cmd = ctx.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(reach_buf.as_ref()), 0);
    enc.set_buffer(1, Some(hc_buf.as_ref()), 0);
    enc.set_buffer(2, Some(hs_buf.as_ref()), 0);
    enc.set_buffer(3, Some(out.as_ref()), 0);
    let nh32 = nh as i32; let b32 = 1i32;
    enc.set_bytes(4, 4, &nh32 as *const _ as *const _);
    enc.set_bytes(5, 4, &b32 as *const _ as *const _);
    enc.set_bytes(6, 4, &elig_opps as *const _ as *const _);
    enc.set_bytes(7, 4, &tied_offset as *const _ as *const _);
    let grid = MTLSize { width: 1, height: nh as u64, depth: 1 };
    let tg = MTLSize { width: 1, height: 32.min(nh as u64), depth: 1 };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    out.to_vec()
}

/// Compose the full CFV on GPU via host-orchestrated dispatches of the
/// per-level share kernel.
fn full_cfv_gpu_composed(
    ctx: &MetalContext,
    pipeline: &metal::ComputePipelineState,
    reach_buf: &MetalBuffer<f32>,
    hc_buf: &MetalBuffer<u8>,
    hs_buf: &MetalBuffer<u16>,
    nh: usize,
    contributions: &[i32], fold_mask: u16, traverser: usize,
    np: usize, starting_pot: i32,
) -> Vec<f32> {
    let levels = build_level_info(contributions, fold_mask, traverser, np, starting_pot);
    let c_t = contributions[traverser];
    let stake = starting_pot as f32 / np as f32 + c_t as f32;

    // Dispatch elig_opps=0 → TVRP for each h.
    let tvrp = dispatch_share_kernel(ctx, pipeline, reach_buf, hc_buf, hs_buf, nh, 0u32, 0u32);

    let mut static_cash: f32 = 0.0;
    let mut case_c = vec![0.0f32; nh];

    for lev in &levels {
        if !lev.has_active_elig && lev.trav_elig {
            static_cash += lev.pot_l as f32;
        } else if !lev.has_active_elig && !lev.trav_elig {
            if lev.trav_contrib_at_lev > 0.0 {
                static_cash += lev.trav_contrib_at_lev as f32;
            }
        } else if !lev.trav_elig {
            // Case D: no contribution to cfv.
        } else {
            // Case C: dispatch share kernel for this level.
            let share = dispatch_share_kernel(
                ctx, pipeline, reach_buf, hc_buf, hs_buf,
                nh, lev.elig_opps, 0u32);
            let pot = lev.pot_l as f32;
            for h in 0..nh {
                case_c[h] += pot * share[h];
            }
        }
    }

    let mut cfv = vec![0.0f32; nh];
    for h in 0..nh {
        cfv[h] = (static_cash - stake) * tvrp[h] + case_c[h];
    }
    cfv
}

// ---- Setups ---------------------------------------------------------------

fn setup_6p(num_opp: usize) -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize) {
    let nh = 8;
    let hand_cards: Vec<u8> = vec![0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15];
    let hand_strength: Vec<u16> = vec![100, 80, 70, 70, 60, 50, 40, 30];
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.4 + 0.5 * ((h + oi * 3) % 7) as f32 / 7.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh)
}

fn setup_6p_three_way(num_opp: usize) -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize) {
    let nh = 10;
    let hand_cards: Vec<u8> = vec![
        0,1, 2,3, 4,5, 6,7, 0,8,
        0,2, 2,9, 10,11, 12,13, 14,15,
    ];
    let hand_strength: Vec<u16> = vec![100, 90, 70, 80, 70, 60, 50, 50, 40, 30];
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.3 + 0.6 * ((h + oi * 3) % 7) as f32 / 7.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh)
}

fn run_full_cfv_test(
    label: &str,
    three_way: bool,
    contributions: Vec<i32>,
    fold_mask: u16,
    traverser: usize,
    np: usize,
) {
    let num_opp = np - 1;
    let (hand_cards, hand_strength, reach, nh) =
        if three_way { setup_6p_three_way(num_opp) }
        else { setup_6p(num_opp) };
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let starting_pot = (np as i32) * 5;
    let reach_flat: Vec<f32> = (0..num_opp).flat_map(|oi| reach[oi].iter().copied()).collect();

    let ctx = MetalContext::new().expect("Metal");
    let kernel_name = if num_opp == 3 { "k3_per_level_share_microbench" }
                      else if num_opp == 5 { "k5_per_level_share_microbench" }
                      else { panic!("no factored kernel for num_opp={}", num_opp); };
    let pipeline = ctx.create_pipeline(kernel_name).unwrap();
    let reach_buf = MetalBuffer::from_slice(ctx.device(), &reach_flat);
    let hc_buf = MetalBuffer::from_slice(ctx.device(), &hand_cards);
    let hs_buf = MetalBuffer::from_slice(ctx.device(), &hand_strength);

    // GPU full CFV (host-orchestrated composition over kernel dispatches).
    let gpu = full_cfv_gpu_composed(
        &ctx, &pipeline, &reach_buf, &hc_buf, &hs_buf,
        nh, &contributions, fold_mask, traverser, np, starting_pot);

    // CPU full CFV (f64).
    let opp_reach: Vec<&[f32]> = (0..num_opp).map(|oi| reach[oi].as_slice()).collect();
    let cpu = factored_cfv_cpu_f64(
        &masses, &opp_reach, &hand_cards, &hand_strength, nh,
        &contributions, fold_mask, traverser, np, starting_pot);

    let levels = build_level_info(&contributions, fold_mask, traverser, np, starting_pot);
    eprintln!("=== {} (np={}, {} levels, traverser=p{}) ===",
        label, np, levels.len(), traverser);
    eprintln!("    contribs={:?}, fold_mask={:#b}", contributions, fold_mask);
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut worst_h = 0;
    for h in 0..nh {
        let g = gpu[h] as f64;
        let c = cpu[h];
        let diff = (g - c).abs();
        let scale = c.abs().max(1.0);
        let rel = diff / scale;
        if diff > max_abs { max_abs = diff; worst_h = h; }
        if rel > max_rel { max_rel = rel; }
        eprintln!("    h={} cpu_f64={:>14.6} gpu_f32={:>14.6} diff={:.3e} rel={:.3e}",
            h, c, g, diff, rel);
    }
    eprintln!("    max_abs={:.3e} at h={}, max_rel={:.3e}\n", max_abs, worst_h, max_rel);
    // Threshold: f32 noise floor for multi-level composition. Each per-level
    // share kernel dispatch validates at ~6e-7 max_rel against CPU f64;
    // composing N levels accumulates ~N × f32-op noise. For up to 6 levels
    // plus TVRP, expected max_rel is on the order of 1e-5.
    assert!(max_rel < 5e-5,
        "{}: max_abs={:.3e} max_rel={:.3e}", label, max_abs, max_rel);
}

#[test]
fn full_cfv_4p_no_fold_equal() {
    run_full_cfv_test("4p no-fold equal", false,
        vec![5, 5, 5, 5], 0, 0, 4);
}

#[test]
fn full_cfv_4p_side_pot() {
    run_full_cfv_test("4p side pot", false,
        vec![95, 5, 5, 5], 0, 0, 4);
}

#[test]
fn full_cfv_4p_single_fold() {
    run_full_cfv_test("4p single fold", false,
        vec![25, 25, 5, 25], 1u16 << 2, 0, 4);
}

#[test]
fn full_cfv_4p_folded_high_contrib() {
    run_full_cfv_test("4p folded high contrib (bug-class)", false,
        vec![190, 5, 5, 95], 0b0011, 0, 4);
}

#[test]
fn full_cfv_6p_all_eligible() {
    run_full_cfv_test("6p all eligible", false,
        vec![5, 5, 5, 5, 5, 5], 0, 0, 6);
}

#[test]
fn full_cfv_6p_six_level_side_pot() {
    run_full_cfv_test("6p SIX-LEVEL side pot", false,
        vec![200, 100, 50, 25, 10, 5], 0, 0, 6);
}

#[test]
fn full_cfv_6p_multi_fold_multi_level() {
    run_full_cfv_test("6p multi-fold + multi-level", false,
        vec![200, 100, 25, 25, 10, 5],
        (1u16 << 2) | (1u16 << 4), 0, 6);
}

#[test]
fn full_cfv_6p_three_way_side_pot() {
    run_full_cfv_test("6p three-way shared + side pot", true,
        vec![50, 25, 5, 5, 5, 5], 0, 0, 6);
}

#[test]
fn full_cfv_6p_folded_high_contrib_side_pot() {
    run_full_cfv_test("6p folded high contrib + side pot", false,
        vec![300, 100, 25, 25, 10, 5], 1u16 << 0, 0, 6);
}

#[test]
fn full_cfv_6p_active_traverser_with_folded_high_contrib_opp() {
    run_full_cfv_test("6p active traverser, folded high contrib opp", false,
        vec![300, 100, 25, 25, 10, 5], 1u16 << 0, 3, 6);
}
