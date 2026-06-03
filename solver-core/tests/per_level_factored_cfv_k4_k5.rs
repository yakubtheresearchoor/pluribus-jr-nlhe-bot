// K=4 and K=5 per-level factored CFV — CPU extension + matched-precision
// validation against f64 brute-force.
//
// The docket's gate: K=4/K=5 CPU validated to f64 precision against brute-
// force on small nh before any Metal kernel work. Validation must
// deliberately exercise eligibility patterns that only appear at higher
// player count — multi-level side-pot terminals with several different
// eligibility sets, the configuration space K=3 cannot fully exercise.
// History: bugs in this project (the folded-traverser side-pot bug; bug
// #17) appeared only at higher player counts, so K=5 validation must
// include the multi-level + multi-eligibility patterns that K=3 cannot
// produce (max 4 levels at K=3).
//
// All tests compare f64 factored against f64 brute-force, both running
// in matched precision so the gap is the math discrepancy, not the
// existing-impl's f32 noise.

use solver_core::card::index_to_card_pair;
use solver_core::solver::showdown::{precompute_opp_masses, OppMasses};

// ---- Helpers (copy of the K=3 file's working code, generalized for K) -----

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

fn tvrp_f64(
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
        let tvrp_h = tvrp_f64(opp_reach, hand_cards, nh, h, h_dead);
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
                // Case D
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

// ---- Setup helpers ---------------------------------------------------------

/// 8 pairwise-disjoint hands; strengths chosen with diversity and ties.
fn make_setup_8hands(num_opp: usize) -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize) {
    let nh = 8;
    let hand_cards: Vec<u8> = vec![
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    ];
    let hand_strength: Vec<u16> = vec![100, 80, 70, 70, 60, 50, 40, 30];
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.4 + 0.5 * ((h + oi * 3) % 7) as f32 / 7.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh)
}

/// Setup with deliberate three-way card sharing for K=5 oracle gate:
///   h=0 {0,1}  ← shares card 0 with h=4 and h=5
///   h=1 {2,3}  ← shares card 2 with h=5 and h=6
///   h=2 {4,5}
///   h=3 {6,7}
///   h=4 {0,8}  ← 3-way on card 0 with h=0, h=5
///   h=5 {0,2}  ← 3-way on card 0 (h=0, h=4) and card 2 (h=1, h=6)
///   h=6 {2,9}  ← 3-way on card 2 with h=1, h=5
///   h=7 {10,11}
///   h=8 {12,13}
///   h=9 {14,15}
fn make_setup_three_way_shared(num_opp: usize) -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize) {
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

fn run_matched(
    label: &str,
    np: usize,
    contributions: Vec<i32>,
    fold_mask: u16,
    traverser: usize,
    setup_three_way: bool,
) -> f64 {
    let num_opp = np - 1;
    let (hand_cards, hand_strength, reach, nh) =
        if setup_three_way { make_setup_three_way_shared(num_opp) }
        else { make_setup_8hands(num_opp) };
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let starting_pot = (np as i32) * 5;

    let opp_reach: Vec<&[f32]> = (0..num_opp).map(|oi| reach[oi].as_slice()).collect();

    let bf = brute_force_cfv_f64(
        &opp_reach, &hand_cards, &hand_strength, nh,
        &contributions, fold_mask, traverser, np, starting_pot,
    );
    let fac = factored_cfv_f64(
        &masses, &opp_reach, &hand_cards, &hand_strength, nh,
        &contributions, fold_mask, traverser, np, starting_pot,
    );

    let mut max_abs = 0.0f64;
    let mut worst_h = 0;
    let levels = build_level_info(&contributions, fold_mask, traverser, np, starting_pot);
    let n_levels = levels.len();
    eprintln!("=== {} (np={}, {} levels, traverser=p{}) ===",
        label, np, n_levels, traverser);
    eprintln!("    contribs={:?}, fold_mask={:#b}", contributions, fold_mask);
    for (li, lev) in levels.iter().enumerate() {
        eprintln!("    level {}: pot={:.1}, elig_opps={:#b}, trav_elig={}",
            li, lev.pot_l, lev.elig_opps, lev.trav_elig);
    }
    for h in 0..nh {
        let d = (fac[h] - bf[h]).abs();
        if d > max_abs { max_abs = d; worst_h = h; }
    }
    eprintln!("    factored vs brute (matched f64): max_abs = {:.3e} at h={}\n",
        max_abs, worst_h);
    max_abs
}

// ---- K=4 (5-player) tests --------------------------------------------------

#[test]
fn k4_all_eligible_no_fold() {
    let max = run_matched("K=4 all-eligible no-fold",
        5, vec![5, 5, 5, 5, 5], 0, 0, false);
    assert!(max < 1e-9, "K=4 all-eligible: {:.3e}", max);
}

#[test]
fn k4_single_fold_one_level() {
    // p4 folded; 4 active, single level.
    let max = run_matched("K=4 single fold one level",
        5, vec![5, 5, 5, 5, 5], 1u16 << 4, 0, false);
    assert!(max < 1e-9, "K=4 single fold: {:.3e}", max);
}

#[test]
fn k4_three_level_side_pot_no_fold() {
    // contribs [50, 25, 5, 5, 5]: 3 levels (5, 25, 50)
    // Level 5: 5 elig. Level 25: 2 elig (p0, p1). Level 50: 1 elig (p0).
    let max = run_matched("K=4 three-level side pot",
        5, vec![50, 25, 5, 5, 5], 0, 0, false);
    assert!(max < 1e-9, "K=4 three-level side pot: {:.3e}", max);
}

#[test]
fn k4_side_pot_with_fold() {
    // contribs [50, 25, 5, 5, 5], p2 folded
    let max = run_matched("K=4 side pot + fold",
        5, vec![50, 25, 5, 5, 5], 1u16 << 2, 0, false);
    assert!(max < 1e-9, "K=4 side pot + fold: {:.3e}", max);
}

#[test]
fn k4_traverser_folded_with_side_pot_return() {
    // traverser=p0 folded with high contrib (the bug-suspicion path)
    let max = run_matched("K=4 folded traverser high contrib",
        5, vec![100, 25, 5, 5, 25], 1u16 << 0, 0, false);
    assert!(max < 1e-9, "K=4 folded high contrib: {:.3e}", max);
}

// ---- K=5 (6-player) tests --------------------------------------------------

#[test]
fn k5_all_eligible_no_fold() {
    let max = run_matched("K=5 all-eligible no-fold",
        6, vec![5, 5, 5, 5, 5, 5], 0, 0, false);
    assert!(max < 1e-8, "K=5 all-eligible: {:.3e}", max);
}

#[test]
fn k5_single_fold_one_level() {
    let max = run_matched("K=5 single fold one level",
        6, vec![5, 5, 5, 5, 5, 5], 1u16 << 5, 0, false);
    assert!(max < 1e-8, "K=5 single fold: {:.3e}", max);
}

#[test]
fn k5_four_level_side_pot_no_fold() {
    // contribs [200, 100, 50, 25, 10, 5]: 6 distinct levels.
    // Level 5: 6 elig. Level 10: 5 elig. Level 25: 4 elig. Level 50: 3 elig.
    // Level 100: 2 elig. Level 200: 1 elig.
    // ← MULTI-LEVEL SIDE POT with PROGRESSIVELY SHRINKING ELIGIBILITY — the
    //   configuration space K=3 cannot fully exercise.
    let max = run_matched("K=5 SIX-LEVEL side pot no-fold (the multi-eligibility test)",
        6, vec![200, 100, 50, 25, 10, 5], 0, 0, false);
    assert!(max < 1e-8, "K=5 six-level side pot: {:.3e}", max);
}

#[test]
fn k5_complex_side_pot_with_multiple_folds() {
    // contribs [200, 100, 25, 25, 10, 5], p2 and p4 folded.
    // Active: p0 (200), p1 (100), p3 (25), p5 (5).
    // Levels: 5, 10, 25, 100, 200.
    // Multi-level + multi-fold — the K=5-specific configuration space.
    let max = run_matched("K=5 multi-level side pot + multi-fold",
        6, vec![200, 100, 25, 25, 10, 5], (1u16 << 2) | (1u16 << 4), 0, false);
    assert!(max < 1e-8, "K=5 multi-fold + multi-level side pot: {:.3e}", max);
}

#[test]
fn k5_folded_traverser_high_contrib_side_pot() {
    // traverser folded with the highest contribution; gets side-pot return
    // at the orphan level. Same bug-class as the K=3 folded-high-contrib
    // case, exercised at K=5 with deeper recursion.
    let max = run_matched("K=5 folded traverser high contrib + side pot",
        6, vec![300, 100, 25, 25, 10, 5], 1u16 << 0, 0, false);
    assert!(max < 1e-8, "K=5 folded high contrib: {:.3e}", max);
}

#[test]
fn k5_three_way_shared_cards_no_fold() {
    // The three-way-shared-card configuration that broke the matching
    // formula earlier — exercised at K=5 with the per-level factored.
    // Tests the recursion handles 3-way card conflicts correctly via the
    // growing dead mask even with per-level eligibility logic threaded
    // through.
    let max = run_matched("K=5 three-way shared cards no-fold",
        6, vec![5, 5, 5, 5, 5, 5], 0, 0, true);
    assert!(max < 1e-8, "K=5 three-way shared cards: {:.3e}", max);
}

#[test]
fn k5_three_way_shared_cards_with_side_pot() {
    // Three-way card conflict configuration + side pots — combined stress
    // of the conflict-handling and the per-level eligibility decomposition.
    let max = run_matched("K=5 three-way shared cards + side pot",
        6, vec![50, 25, 5, 5, 5, 5], 0, 0, true);
    assert!(max < 1e-8, "K=5 three-way + side pot: {:.3e}", max);
}

#[test]
fn k5_active_traverser_against_folded_high_contrib() {
    // traverser=p3 (active, contrib 25); p0 folded with contrib 300.
    // p0's folded contribution at level 100-300 has nobody eligible →
    // returns to p0 (a folded player), which previously was the bug.
    // For an ACTIVE traverser, this exercises the per-level walk's
    // handling of side-pot returns to other (folded) players via Case A.
    let max = run_matched("K=5 active traverser, folded high-contrib opp",
        6, vec![300, 100, 25, 25, 10, 5], 1u16 << 0, 3, false);
    assert!(max < 1e-8, "K=5 active vs folded high contrib: {:.3e}", max);
}

#[allow(dead_code)]
fn unused() { let _ = index_to_card_pair; }
