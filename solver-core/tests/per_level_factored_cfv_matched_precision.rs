// Matched-precision validation of the K=3 per-level factored CFV.
//
// The previous validation (per_level_factored_cfv.rs) compared factored
// (f64) against side_pot_showdown_cfv (f32) and got 1e-3 to 1e-4 max-abs
// on the three realistic terminal types (no-fold equal, side pot, single
// fold). The attribution was "f32 vs f64 precision noise" — but that
// pattern of attribution has been wrong before in this project, and the
// single-fold case (the largest 1e-3 gap) is exactly the path where the
// folded-traverser side-pot bug was found earlier. So before treating
// the per-level factored as foundation, confirm the gap is precision,
// not a real bug, by running brute-force in f64 too and comparing
// like-to-like.
//
// If the factored-f64 vs brute-f64 gap is at the f64 noise floor
// (~1e-14), the attribution was correct and the per-level factored is
// validated as foundation. If the gap stays at 1e-3, there's a real
// discrepancy in the fold/side-pot composition and it gets differential-
// test treatment, the same way the multiway showdown bugs were found.

use solver_core::card::index_to_card_pair;
use solver_core::solver::showdown::{precompute_opp_masses, OppMasses, side_pot_showdown_cfv};

// ---- f64 brute-force CFV reference -----------------------------------------

/// Brute-force per-level CFV in f64 throughout. Mirrors the structure of
/// side_pot_showdown_cfv but uses f64 accumulators so the comparison
/// against the f64 factored is like-to-like (no f32 precision noise).
fn brute_force_cfv_f64(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    hand_strength: &[u16],
    nh: usize,
    contributions: &[i32],
    fold_mask: u16,
    traverser: usize,
    np: usize,
    starting_pot: i32,
) -> Vec<f64> {
    let k = opp_reach.len();
    let g_mask: Vec<u64> = (0..nh)
        .map(|g| (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]))
        .collect();
    let c_t = contributions[traverser];
    let traverser_stake = starting_pot as f64 / np as f64 + c_t as f64;
    let traverser_folded = fold_mask & (1u16 << traverser) != 0;

    let opp_player: Vec<usize> = (0..k).map(|oi| {
        if oi < traverser { oi } else { oi + 1 }
    }).collect();
    let opp_contrib: Vec<i32> = opp_player.iter().map(|&p| contributions[p]).collect();
    let opp_folded: Vec<bool> = opp_player.iter().map(|&p| fold_mask & (1u16 << p) != 0).collect();

    let mut levels: Vec<i32> = (0..np).map(|p| contributions[p]).collect();
    levels.sort();
    levels.dedup();

    let mut cfv = vec![0.0f64; nh];

    fn rec(
        oi: usize, k: usize, nh: usize, mask: u64, r_so_far: f64,
        g_str: &mut [u16],
        opp_reach: &[&[f32]], g_mask: &[u64], hand_strength: &[u16],
        callback: &mut dyn FnMut(f64, &[u16]),
    ) {
        if oi == k { callback(r_so_far, g_str); return; }
        for g in 0..nh {
            if g_mask[g] & mask != 0 { continue; }
            let r = opp_reach[oi][g] as f64;
            if r == 0.0 { continue; }
            g_str[oi] = hand_strength[g];
            rec(oi+1, k, nh, mask | g_mask[g], r_so_far * r,
                g_str, opp_reach, g_mask, hand_strength, callback);
        }
    }

    for h in 0..nh {
        let h_m = g_mask[h];
        let h_str = hand_strength[h];
        let mut accum = 0.0f64;
        let mut g_str = vec![0u16; k];

        rec(0, k, nh, h_m, 1.0, &mut g_str,
            opp_reach, &g_mask, hand_strength,
            &mut |r_prod: f64, g_str: &[u16]| {
                // Walk levels in f64.
                let mut cash = 0.0f64;
                let mut prev_l = 0i32;
                for (li, &lev) in levels.iter().enumerate() {
                    let pc = lev - prev_l;
                    if pc == 0 { prev_l = lev; continue; }
                    let num_contrib = (0..np).filter(|&p| contributions[p] >= lev).count();
                    let mut pot_l = (pc * num_contrib as i32) as f64;
                    if li == 0 { pot_l += starting_pot as f64; }

                    let trav_elig = !traverser_folded && c_t >= lev;
                    let mut elig_count = if trav_elig { 1u32 } else { 0 };
                    let mut max_str = if trav_elig { h_str } else { 0u16 };
                    for oi in 0..k {
                        if opp_folded[oi] { continue; }
                        if opp_contrib[oi] < lev { continue; }
                        elig_count += 1;
                        if g_str[oi] > max_str { max_str = g_str[oi]; }
                    }

                    if elig_count == 0 {
                        if contributions[traverser] >= lev {
                            let trav_contrib = pc as f64 + if li == 0 { starting_pot as f64 / np as f64 } else { 0.0 };
                            cash += trav_contrib;
                        }
                        prev_l = lev;
                        continue;
                    }

                    if !trav_elig {
                        prev_l = lev;
                        continue;
                    }

                    let mut tied = 0u32;
                    if h_str == max_str { tied += 1; }
                    for oi in 0..k {
                        if opp_folded[oi] { continue; }
                        if opp_contrib[oi] < lev { continue; }
                        if g_str[oi] == max_str { tied += 1; }
                    }

                    if h_str == max_str {
                        cash += pot_l / tied as f64;
                    }
                    prev_l = lev;
                }
                let net = cash - traverser_stake;
                accum += r_prod * net;
            });

        cfv[h] = accum;
    }
    cfv
}

// ---- Per-level factored CFV (same as per_level_factored_cfv.rs) -----------

#[derive(Clone, Copy, Debug)]
struct LevelInfo {
    pot_l: f64,
    elig_opps: u32,
    trav_elig: bool,
    trav_contrib_at_lev: f64,
    has_active_elig: bool,
}

fn build_level_info(
    contributions: &[i32],
    fold_mask: u16,
    traverser: usize,
    np: usize,
    starting_pot: i32,
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
            pot_l,
            elig_opps,
            trav_elig,
            trav_contrib_at_lev: if c_t >= lev { trav_contrib_at_lev } else { 0.0 },
            has_active_elig: elig_opps != 0,
        });
        prev_l = lev;
    }
    out
}

fn tvrp(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    nh: usize,
    h: usize,
    h_dead_mask: u64,
) -> f64 {
    let k = opp_reach.len();
    let g_mask: Vec<u64> = (0..nh)
        .map(|g| (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]))
        .collect();
    fn rec(
        oi: usize, k: usize, nh: usize, mask: u64, r_so_far: f64,
        opp_reach: &[&[f32]], g_mask: &[u64], sum: &mut f64,
    ) {
        if oi == k { *sum += r_so_far; return; }
        for g in 0..nh {
            if g_mask[g] & mask != 0 { continue; }
            let r = opp_reach[oi][g] as f64;
            if r == 0.0 { continue; }
            rec(oi+1, k, nh, mask | g_mask[g], r_so_far * r,
                opp_reach, g_mask, sum);
        }
    }
    let mut s = 0.0f64;
    rec(0, k, nh, h_dead_mask, 1.0, opp_reach, &g_mask, &mut s);
    s
}

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
            opp_indices[0], opp_indices[1],
            elig[0], elig[1],
            tied_so_far,
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

    let mut edge_bb = 0.0f64;
    let mut edge_bt = 0.0f64;
    let mut edge_tb = 0.0f64;
    let mut edge_tt = 0.0f64;
    let mut edge_be = 0.0f64;
    let mut edge_te = 0.0f64;
    let mut edge_eb = 0.0f64;
    let mut edge_et = 0.0f64;
    let mut edge_ee = 0.0f64;
    for c in 0..52usize {
        if (1u64 << c) & h_dead_mask != 0 { continue; }
        let bac = b_a_pc[c]; let tac = t_a_pc[c]; let sac = s_a_pc[c];
        let bbc = b_b_pc[c]; let tbc = t_b_pc[c]; let sbc = s_b_pc[c];
        let rac = bac + tac + sac;
        let rbc = bbc + tbc + sbc;
        edge_bb += bac * bbc;
        edge_bt += bac * tbc;
        edge_tb += tac * bbc;
        edge_tt += tac * tbc;
        edge_be += bac * rbc;
        edge_te += tac * rbc;
        edge_eb += rac * bbc;
        edge_et += rac * tbc;
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
            pair_bb / (1.0 + t0)
                + pair_bt / (2.0 + t0)
                + pair_tb / (2.0 + t0)
                + pair_tt / (3.0 + t0)
        }
        (true, false) => pair_be / (1.0 + t0) + pair_te / (2.0 + t0),
        (false, true) => pair_eb / (1.0 + t0) + pair_et / (2.0 + t0),
        (false, false) => pair_ee / (1.0 + t0),
    }
}

fn factored_cfv_f64(
    masses: &OppMasses,
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    hand_strength: &[u16],
    nh: usize,
    contributions: &[i32],
    fold_mask: u16,
    traverser: usize,
    np: usize,
    starting_pot: i32,
) -> Vec<f64> {
    let levels = build_level_info(contributions, fold_mask, traverser, np, starting_pot);
    let k = opp_reach.len();
    let opp_indices: Vec<usize> = (0..k).collect();
    let c_t = contributions[traverser];
    let traverser_stake = starting_pot as f64 / np as f64 + c_t as f64;

    let mut cfv = vec![0.0f64; nh];
    for h in 0..nh {
        let h_dead = (1u64 << hand_cards[h*2]) | (1u64 << hand_cards[h*2+1]);
        let tvrp_h = tvrp(opp_reach, hand_cards, nh, h, h_dead);
        let mut static_cash = 0.0f64;
        let mut case_c_total = 0.0f64;
        for lev in &levels {
            if !lev.has_active_elig && lev.trav_elig {
                static_cash += lev.pot_l;
            } else if !lev.has_active_elig && !lev.trav_elig {
                if lev.trav_contrib_at_lev > 0.0 {
                    static_cash += lev.trav_contrib_at_lev;
                }
            } else if !lev.trav_elig {
                // Case D: no cash
            } else {
                let elig: Vec<bool> = (0..k).map(|oi| (lev.elig_opps >> oi) & 1 != 0).collect();
                let share = factored_share_at_level(
                    masses, opp_reach, hand_strength,
                    h, h_dead, &opp_indices, &elig, 0,
                );
                case_c_total += lev.pot_l * share;
            }
        }
        cfv[h] = (static_cash - traverser_stake) * tvrp_h + case_c_total;
    }
    cfv
}

// ---- Tests ----------------------------------------------------------------

fn make_4p_setup() -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize, usize) {
    let nh = 8;
    let hand_cards: Vec<u8> = vec![
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    ];
    let hand_strength: Vec<u16> = vec![100, 80, 70, 70, 60, 50, 40, 30];
    let np = 4usize;
    let num_opp = np - 1;
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.4 + 0.5 * ((h + oi * 3) % 7) as f32 / 7.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh, np)
}

fn run_matched_test(
    label: &str,
    contributions: Vec<i32>,
    fold_mask: u16,
    traverser: usize,
) -> (f64, f64) {
    let (hand_cards, hand_strength, reach, nh, np_u) = make_4p_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let np = np_u as u8;
    let starting_pot = (np_u as i32) * 5;
    let num_opp = np_u - 1;

    // f64 brute force
    let opp_reach: Vec<&[f32]> = (0..num_opp).map(|oi| reach[oi].as_slice()).collect();
    let bf_f64 = brute_force_cfv_f64(
        &opp_reach, &hand_cards, &hand_strength, nh,
        &contributions, fold_mask, traverser, np_u, starting_pot,
    );

    // f64 factored
    let fac_f64 = factored_cfv_f64(
        &masses, &opp_reach, &hand_cards, &hand_strength, nh,
        &contributions, fold_mask, traverser, np_u, starting_pot,
    );

    // f32 brute force (existing side_pot_showdown_cfv)
    let mut sp_pairs: Vec<(u16, u16)> = (0..nh).map(|h| (hand_strength[h], h as u16)).collect();
    sp_pairs.sort_by_key(|&(s, _)| s);
    let sorted_pl_str: Vec<u16> = sp_pairs.iter().map(|&(s, _)| s).collect();
    let sorted_pl_idx: Vec<u16> = sp_pairs.iter().map(|&(_, i)| i).collect();
    let mut sorted_opp_str = vec![0u16; num_opp * nh];
    let mut sorted_opp_idx = vec![0u16; num_opp * nh];
    for oi in 0..num_opp {
        for h in 0..nh {
            sorted_opp_str[oi * nh + h] = sorted_pl_str[h];
            sorted_opp_idx[oi * nh + h] = sorted_pl_idx[h];
        }
    }
    let bf_f32 = side_pot_showdown_cfv(
        &opp_reach, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &sorted_pl_str, &sorted_pl_idx,
        &contributions, fold_mask, traverser, np, starting_pot,
    );

    let mut max_abs_fac_vs_bf_f64 = 0.0f64;
    let mut max_abs_bf_f64_vs_f32 = 0.0f64;
    eprintln!("=== {} ===", label);
    eprintln!("traverser={}, contribs={:?}, fold_mask={:#b}", traverser, contributions, fold_mask);
    for h in 0..nh {
        let d_fac_f64 = (fac_f64[h] - bf_f64[h]).abs();
        let d_bf = (bf_f64[h] - bf_f32[h] as f64).abs();
        if d_fac_f64 > max_abs_fac_vs_bf_f64 { max_abs_fac_vs_bf_f64 = d_fac_f64; }
        if d_bf > max_abs_bf_f64_vs_f32 { max_abs_bf_f64_vs_f32 = d_bf; }
        eprintln!("  h={} bf_f64={:.10} fac_f64={:.10} bf_f32={:.10}  fac-bf_f64={:.3e}  bf_f64-bf_f32={:.3e}",
            h, bf_f64[h], fac_f64[h], bf_f32[h], d_fac_f64, d_bf);
    }
    eprintln!("  max_abs[factored_f64 vs brute_f64]:  {:.3e}", max_abs_fac_vs_bf_f64);
    eprintln!("  max_abs[brute_f64 vs brute_f32]:      {:.3e}  (= existing-impl f32 precision)", max_abs_bf_f64_vs_f32);
    eprintln!();
    (max_abs_fac_vs_bf_f64, max_abs_bf_f64_vs_f32)
}

#[test]
fn matched_precision_no_fold_equal() {
    let (max_fac, max_f32) = run_matched_test(
        "NO-FOLD EQUAL contribs [5,5,5,5]",
        vec![5, 5, 5, 5], 0, 0,
    );
    // If the gap is precision noise, the f32 vs f64 diff should be comparable
    // to or larger than the factored vs f64 diff.
    eprintln!("Verdict: factored-vs-brute(f64) = {:.3e}, brute-precision(f64-vs-f32) = {:.3e}", max_fac, max_f32);
    // The gate: matched-precision factored vs brute must be at f64 noise floor.
    assert!(max_fac < 1e-10,
        "NO-FOLD-EQUAL: factored f64 should match brute f64 to f64 noise floor; got {:.3e}",
        max_fac);
}

#[test]
fn matched_precision_side_pot() {
    let (max_fac, max_f32) = run_matched_test(
        "SIDE POT contribs [95, 5, 5, 5]",
        vec![95, 5, 5, 5], 0, 0,
    );
    eprintln!("Verdict: factored-vs-brute(f64) = {:.3e}, brute-precision(f64-vs-f32) = {:.3e}", max_fac, max_f32);
    assert!(max_fac < 1e-10,
        "SIDE-POT: factored f64 should match brute f64 to f64 noise floor; got {:.3e}",
        max_fac);
}

#[test]
fn matched_precision_single_fold() {
    let (max_fac, max_f32) = run_matched_test(
        "SINGLE FOLD contribs [25, 25, 5, 25], p2 folded",
        vec![25, 25, 5, 25], 1u16 << 2, 0,
    );
    eprintln!("Verdict: factored-vs-brute(f64) = {:.3e}, brute-precision(f64-vs-f32) = {:.3e}", max_fac, max_f32);
    assert!(max_fac < 1e-10,
        "SINGLE-FOLD (the path where bugs hide): factored f64 should match brute f64 to f64 noise floor; got {:.3e}",
        max_fac);
}

#[test]
fn matched_precision_high_contrib_folded() {
    // The path that originally hid the folded-traverser side-pot bug.
    let (max_fac, max_f32) = run_matched_test(
        "FOLDED HIGH CONTRIB contribs [190, 5, 5, 95], p0+p1 folded",
        vec![190, 5, 5, 95], 0b0011, 0,
    );
    eprintln!("Verdict: factored-vs-brute(f64) = {:.3e}, brute-precision(f64-vs-f32) = {:.3e}", max_fac, max_f32);
    assert!(max_fac < 1e-10,
        "FOLDED-HIGH-CONTRIB: factored f64 should match brute f64 to f64 noise floor; got {:.3e}",
        max_fac);
}

#[test]
fn matched_precision_two_folds_active_traverser() {
    // Two folded players, active traverser computing CFV.
    let (max_fac, max_f32) = run_matched_test(
        "TWO FOLDS, ACTIVE TRAVERSER contribs [190, 5, 5, 95], p0+p1 folded, traverser=p3",
        vec![190, 5, 5, 95], 0b0011, 3,
    );
    eprintln!("Verdict: factored-vs-brute(f64) = {:.3e}, brute-precision(f64-vs-f32) = {:.3e}", max_fac, max_f32);
    assert!(max_fac < 1e-10,
        "TWO-FOLDS-ACTIVE-TRAVERSER: factored f64 should match brute f64 to f64 noise floor; got {:.3e}",
        max_fac);
}

#[allow(dead_code)]
fn unused() { let _ = index_to_card_pair; }
