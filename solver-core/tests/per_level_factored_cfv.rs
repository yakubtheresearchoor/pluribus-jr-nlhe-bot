// Per-level factored CFV at K=3: derivation + validation against the
// brute-force per-level evaluator already in side_pot_showdown_cfv.
//
// The factored share function previously validated only handles the
// no-fold equal-contribution single-pot case. Real terminals have:
//   - Multiple pot levels (side pots from unequal contributions)
//   - Folded players (eligibility ≠ activity)
//   - Mixed eligibility per level (some opps active+contributed, others not)
//
// For each pot level l, the FACTORED share-at-l uses eligibility-restricted
// strength comparison: eligible opps use B/T/S categorization vs h_str,
// ineligible opps contribute reach without strength filter (their hand
// doesn't affect winner determination at this level).
//
// The K=2 base case expands from 4 PAIRs (BB, BT, TB, TT) to handle four
// eligibility configurations: (E,E), (E,I), (I,E), (I,I). For (E,I) cases,
// the ineligible opp's "any-strength" reach decomposes as B+T+S without
// strength constraint; same-hand corrections still apply with the
// eligibility-determined category binding.
//
// Validation: K=3 against the existing brute-force K=3 per-level evaluator
// (in side_pot_showdown_cfv) on terminals with realistic contribution
// patterns: all-equal-no-fold, single fold, side pot.

use solver_core::card::index_to_card_pair;
use solver_core::solver::showdown::{precompute_opp_masses, OppMasses, side_pot_showdown_cfv};

// ---- Helpers ---------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct LevelInfo {
    pot_l: f64,
    /// Bitmask of OPP indices (0..K) eligible at this level (active + contrib ≥ lev).
    elig_opps: u32,
    /// Whether the traverser is eligible at this level.
    trav_elig: bool,
    /// Traverser's contribution-at-this-slice, used in the "no eligible
    /// active → return to contributors" path (Case A).
    trav_contrib_at_lev: f64,
    /// Whether ANY active opp is eligible (= elig_opps != 0).
    has_active_elig: bool,
}

/// Compute the per-level breakdown of a terminal: pot levels, eligibility
/// sets per level, traverser contribution at each slice.
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

        // Build opp eligibility mask: bit i set iff opp at slot i (in
        // the OPP indexing 0..K, NOT player index) is eligible.
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

/// TVRP(h) = Σ over valid (g_0, ..., g_{K-1}) of Π r_i[g_i].
/// Used to scale the static (no-strength-comparison) cash components.
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

/// Per-level share function for one level l (Case C): traverser eligible
/// AND at least one active opp eligible. Uses eligibility-restricted
/// strength comparison.
///
/// share_at_l(h) = Σ over (g_0, ..., g_{K-1}) valid of
///   r_product × indicator(traverser at top among eligible ∪ {traverser})
///                 / (1 + tied_at_l)
///
/// where for eligible opps we enforce s ≤ h_str (skip S), and for
/// ineligible opps we don't filter (any strength).
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

/// K=2 base case for the per-level share. Four eligibility configurations:
///   (E,E), (E,I), (I,E), (I,I) — each with its own PAIR formula.
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

    // Compute base K=2 quantities at the EXTENDED dead mask in one pass.
    // For each opp we track:
    //   B_total, T_total, S_total, and per-card B, T, S.
    // Plus same-hand sums H_BB, H_TT, H_SS (over h_x with s_x in the
    // respective category, sum of r_a[h_x]·r_b[h_x]).
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

    // Total-reach versions for ineligible opps.
    let r_a = b_a + t_a + s_a;
    let r_b = b_b + t_b + s_b;

    // Compute edge sums for whichever PAIR formulas we need.
    // For (E, E): edge_bb, edge_bt, edge_tb, edge_tt
    // For (E, I): edge_bE = Σ b_a^(c) · r_b^(c); edge_tE = Σ t_a^(c) · r_b^(c)
    //            where r_b^(c) = b_b^(c) + t_b^(c) + s_b^(c)
    // For (I, E): edge_Eb = Σ r_a^(c) · b_b^(c); edge_Et similarly
    // For (I, I): edge_EE = Σ r_a^(c) · r_b^(c)
    //
    // To avoid recomputing, precompute the four edge sums we need based on
    // the (ea, eb) combination.
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

    // PAIR formulas (with same-hand corrections).
    let pair_bb = b_a * b_b - edge_bb + h_bb;
    let pair_bt = b_a * t_b - edge_bt;
    let pair_tb = t_a * b_b - edge_tb;
    let pair_tt = t_a * t_b - edge_tt + h_tt;
    let pair_be = b_a * r_b - edge_be + h_bb; // same-hand binding from B side
    let pair_te = t_a * r_b - edge_te + h_tt;
    let pair_eb = r_a * b_b - edge_eb + h_bb;
    let pair_et = r_a * t_b - edge_et + h_tt;
    let pair_ee = r_a * r_b - edge_ee + h_tot;

    // Tied-count factors.
    let t0 = tied_offset as f64;

    // Dispatch by eligibility configuration.
    match (ea, eb) {
        (true, true) => {
            // share = PAIR_BB / (1+t0) + (PAIR_BT + PAIR_TB) / (2+t0) + PAIR_TT / (3+t0)
            pair_bb / (1.0 + t0)
                + pair_bt / (2.0 + t0)
                + pair_tb / (2.0 + t0)
                + pair_tt / (3.0 + t0)
        }
        (true, false) => {
            // (E, I): opp_a uses B/T, opp_b any. tied from opp_b is 0.
            // share = PAIR_BE / (1+t0) + PAIR_TE / (2+t0)
            pair_be / (1.0 + t0) + pair_te / (2.0 + t0)
        }
        (false, true) => {
            // (I, E): symmetric.
            pair_eb / (1.0 + t0) + pair_et / (2.0 + t0)
        }
        (false, false) => {
            // (I, I): neither contributes to tied.
            // share = PAIR_EE / (1+t0)
            pair_ee / (1.0 + t0)
        }
    }
}

/// Brute-force per-level share at level l. Reference for the factored
/// implementation. Same shape as the brute-force enumeration in
/// side_pot_showdown_cfv but isolated for one level so the per-level
/// validation is targeted.
fn brute_force_share_at_level(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    hand_strength: &[u16],
    nh: usize,
    h: usize,
    h_dead_mask: u64,
    elig: &[bool],
    tied_offset: u32,
) -> f64 {
    let k = opp_reach.len();
    let g_mask: Vec<u64> = (0..nh)
        .map(|g| (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]))
        .collect();
    let h_str = hand_strength[h];

    fn rec(
        oi: usize, k: usize, nh: usize, mask: u64, r_so_far: f64, tied: u32, max_so_far: u16,
        h_str: u16, opp_reach: &[&[f32]], g_mask: &[u64], hand_strength: &[u16],
        elig: &[bool], accum: &mut f64,
    ) {
        if oi == k {
            // Traverser at top among eligible iff max_so_far ≤ h_str.
            if max_so_far > h_str { return; }
            *accum += r_so_far / (1.0 + tied as f64);
            return;
        }
        for g in 0..nh {
            if g_mask[g] & mask != 0 { continue; }
            let r = opp_reach[oi][g] as f64;
            if r == 0.0 { continue; }
            let s = hand_strength[g];
            let (nm, nt) = if elig[oi] {
                if s > h_str { continue; }
                let nm = max_so_far.max(s);
                let nt = if s == h_str { tied + 1 } else { tied };
                (nm, nt)
            } else {
                // Ineligible: strength doesn't count toward max or tied.
                (max_so_far, tied)
            };
            rec(oi+1, k, nh, mask | g_mask[g], r_so_far * r, nt, nm,
                h_str, opp_reach, g_mask, hand_strength, elig, accum);
        }
    }

    let mut sum = 0.0f64;
    rec(0, k, nh, h_dead_mask, 1.0, tied_offset, 0,
        h_str, opp_reach, &g_mask, hand_strength, elig, &mut sum);
    sum
}

// ---- Tests ----------------------------------------------------------------

fn make_4p_setup() -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize, usize) {
    let nh = 8;
    // Pairwise-disjoint hands with diverse strengths including ties.
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

#[test]
fn per_level_factored_share_all_eligible_matches_brute_force() {
    // (E, E, E): all 3 opps eligible at this level. Should match the
    // already-validated factored_share function (in k_factored_cfv_general).
    let (hand_cards, hand_strength, reach, nh, _np) = make_4p_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    let opp_indices: Vec<usize> = (0..3).collect();
    let elig = [true, true, true];
    let mut max_abs = 0.0f64;

    for h in 0..nh {
        let h_dead = (1u64 << hand_cards[h*2]) | (1u64 << hand_cards[h*2+1]);
        let bf = brute_force_share_at_level(
            &reach_views, &hand_cards, &hand_strength, nh, h, h_dead, &elig, 0);
        let fac = factored_share_at_level(
            &masses, &reach_views, &hand_strength, h, h_dead, &opp_indices, &elig, 0);
        let diff = (fac - bf).abs();
        if diff > max_abs { max_abs = diff; }
        eprintln!("(E,E,E) h={} s={}  bf={:.6} fac={:.6} diff={:.3e}",
            h, hand_strength[h], bf, fac, diff);
    }
    eprintln!("(E,E,E) max_abs={:.3e}", max_abs);
    assert!(max_abs < 1e-10);
}

#[test]
fn per_level_factored_share_mixed_eligibility_k3() {
    let (hand_cards, hand_strength, reach, nh, _np) = make_4p_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let opp_indices: Vec<usize> = (0..3).collect();

    // Test all 7 non-trivial eligibility configurations (excluding all-I
    // which is essentially TVRP / 1).
    let configs: &[(&str, [bool; 3])] = &[
        ("(E,E,I)", [true, true, false]),
        ("(E,I,E)", [true, false, true]),
        ("(I,E,E)", [false, true, true]),
        ("(E,I,I)", [true, false, false]),
        ("(I,E,I)", [false, true, false]),
        ("(I,I,E)", [false, false, true]),
        ("(I,I,I)", [false, false, false]),
    ];

    for (label, elig_arr) in configs {
        let elig: &[bool] = elig_arr;
        let mut max_abs = 0.0f64;
        let mut worst_h = 0;
        for h in 0..nh {
            let h_dead = (1u64 << hand_cards[h*2]) | (1u64 << hand_cards[h*2+1]);
            let bf = brute_force_share_at_level(
                &reach_views, &hand_cards, &hand_strength, nh, h, h_dead, elig, 0);
            let fac = factored_share_at_level(
                &masses, &reach_views, &hand_strength, h, h_dead, &opp_indices, elig, 0);
            let diff = (fac - bf).abs();
            if diff > max_abs { max_abs = diff; worst_h = h; }
        }
        eprintln!("{}: max_abs={:.3e} at h={}", label, max_abs, worst_h);
        assert!(max_abs < 1e-10,
            "Per-level factored share with eligibility {} fails: max_abs={:.3e} at h={}",
            label, max_abs, worst_h);
    }
}

#[test]
fn per_level_factored_cfv_matches_brute_force_no_fold_equal() {
    // Compose the full per-level CFV from the per-level shares and the
    // static (no-strength) returns, and compare against the existing
    // brute-force side_pot_showdown_cfv on a no-fold equal-contribution
    // 4-player showdown.
    let (hand_cards, hand_strength, reach, nh, np_u) = make_4p_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let np = np_u as u8;

    let contributions = vec![5i32; np_u];
    let fold_mask = 0u16;
    let starting_pot = (np_u as i32) * 5;

    let opp_indices: Vec<usize> = (0..3).collect();
    let traverser = 0usize;
    let levels = build_level_info(&contributions, fold_mask, traverser, np_u, starting_pot);
    eprintln!("Levels: {:?}", levels);
    assert_eq!(levels.len(), 1, "No-fold equal-contrib should produce 1 level");

    // Build sorted_pl arrays for side_pot_showdown_cfv.
    let mut sp_pairs: Vec<(u16, u16)> = (0..nh).map(|h| (hand_strength[h], h as u16)).collect();
    sp_pairs.sort_by_key(|&(s, _)| s);
    let sorted_pl_str: Vec<u16> = sp_pairs.iter().map(|&(s, _)| s).collect();
    let sorted_pl_idx: Vec<u16> = sp_pairs.iter().map(|&(_, i)| i).collect();
    let num_opp = np_u - 1;
    let mut sorted_opp_str = vec![0u16; num_opp * nh];
    let mut sorted_opp_idx = vec![0u16; num_opp * nh];
    for oi in 0..num_opp {
        for h in 0..nh {
            sorted_opp_str[oi * nh + h] = sorted_pl_str[h];
            sorted_opp_idx[oi * nh + h] = sorted_pl_idx[h];
        }
    }
    let opp_reach_bf: Vec<&[f32]> = (0..num_opp).map(|oi| reach[oi].as_slice()).collect();
    let bf_cfv = side_pot_showdown_cfv(
        &opp_reach_bf, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &sorted_pl_str, &sorted_pl_idx,
        &contributions, fold_mask, traverser, np, starting_pot,
    );

    // Factored: CFV(h) = (Σ_A return + Σ_B pot_l*trav_elig_only + Σ_C pot_l*share - stake) × TVRP scaling
    // For no-fold equal-contribs (single level), only Case C applies.
    let c_t = contributions[traverser];
    let traverser_stake = starting_pot as f64 / np_u as f64 + c_t as f64;
    let mut max_abs = 0.0f64;
    let mut worst_h = 0;
    for h in 0..nh {
        let h_dead = (1u64 << hand_cards[h*2]) | (1u64 << hand_cards[h*2+1]);
        let tvrp_h = tvrp(&reach_views, &hand_cards, nh, h, h_dead);

        let mut static_cash = 0.0f64;
        let mut case_c_total = 0.0f64;
        for lev in &levels {
            if !lev.has_active_elig && lev.trav_elig {
                static_cash += lev.pot_l;
            } else if !lev.has_active_elig && !lev.trav_elig {
                if lev.trav_contrib_at_lev > 0.0 {
                    static_cash += lev.trav_contrib_at_lev;
                }
            } else if lev.has_active_elig && !lev.trav_elig {
                // Case D: no cash
            } else {
                // Case C
                let elig: Vec<bool> = (0..num_opp).map(|oi| (lev.elig_opps >> oi) & 1 != 0).collect();
                let share = factored_share_at_level(
                    &masses, &reach_views, &hand_strength,
                    h, h_dead, &opp_indices, &elig, 0,
                );
                case_c_total += lev.pot_l * share;
            }
        }
        let fac_cfv = (static_cash - traverser_stake) * tvrp_h + case_c_total;
        let diff = (fac_cfv as f32 - bf_cfv[h]).abs() as f64;
        if diff > max_abs { max_abs = diff; worst_h = h; }
        eprintln!("h={} bf={:.6} fac={:.6} diff={:.3e}",
            h, bf_cfv[h], fac_cfv, diff);
    }
    eprintln!("no-fold-equal max_abs={:.3e} at h={}", max_abs, worst_h);
    assert!(max_abs < 1e-3, "no-fold equal CFV factored vs brute: max_abs={:.3e}", max_abs);
}

#[test]
fn per_level_factored_cfv_matches_brute_force_with_side_pot() {
    // 4-player with side pot: contribs [95, 5, 5, 5], no fold. Generates
    // two pot levels: level 5 (all 4 eligible) and level 95 (only p0).
    // At level 95, only p0 is eligible (traverser when traverser=0) and
    // since traverser is the only eligible, no strength comparison needed.
    let (hand_cards, hand_strength, reach, nh, np_u) = make_4p_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let np = np_u as u8;

    let contributions = vec![95i32, 5, 5, 5];
    let fold_mask = 0u16;
    let starting_pot = (np_u as i32) * 5;

    let opp_indices: Vec<usize> = (0..3).collect();

    // Test with traverser = 0 (the high-contrib player who is sole-eligible at level 95).
    let traverser = 0usize;
    let levels = build_level_info(&contributions, fold_mask, traverser, np_u, starting_pot);
    eprintln!("Traverser p0 levels: {:?}", levels);

    let mut sp_pairs: Vec<(u16, u16)> = (0..nh).map(|h| (hand_strength[h], h as u16)).collect();
    sp_pairs.sort_by_key(|&(s, _)| s);
    let sorted_pl_str: Vec<u16> = sp_pairs.iter().map(|&(s, _)| s).collect();
    let sorted_pl_idx: Vec<u16> = sp_pairs.iter().map(|&(_, i)| i).collect();
    let num_opp = np_u - 1;
    let mut sorted_opp_str = vec![0u16; num_opp * nh];
    let mut sorted_opp_idx = vec![0u16; num_opp * nh];
    for oi in 0..num_opp {
        for h in 0..nh {
            sorted_opp_str[oi * nh + h] = sorted_pl_str[h];
            sorted_opp_idx[oi * nh + h] = sorted_pl_idx[h];
        }
    }
    let opp_reach_bf: Vec<&[f32]> = (0..num_opp).map(|oi| reach[oi].as_slice()).collect();
    let bf_cfv = side_pot_showdown_cfv(
        &opp_reach_bf, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &sorted_pl_str, &sorted_pl_idx,
        &contributions, fold_mask, traverser, np, starting_pot,
    );

    let c_t = contributions[traverser];
    let traverser_stake = starting_pot as f64 / np_u as f64 + c_t as f64;
    let mut max_abs = 0.0f64;
    let mut worst_h = 0;
    for h in 0..nh {
        let h_dead = (1u64 << hand_cards[h*2]) | (1u64 << hand_cards[h*2+1]);
        let tvrp_h = tvrp(&reach_views, &hand_cards, nh, h, h_dead);
        let mut static_cash = 0.0f64;
        let mut case_c_total = 0.0f64;
        for lev in &levels {
            if !lev.has_active_elig && lev.trav_elig {
                // Traverser sole-eligible (no other active eligible): wins level
                static_cash += lev.pot_l;
            } else if !lev.has_active_elig && !lev.trav_elig {
                if lev.trav_contrib_at_lev > 0.0 {
                    static_cash += lev.trav_contrib_at_lev;
                }
            } else if !lev.trav_elig {
                // Case D
            } else {
                let elig: Vec<bool> = (0..num_opp).map(|oi| (lev.elig_opps >> oi) & 1 != 0).collect();
                let share = factored_share_at_level(
                    &masses, &reach_views, &hand_strength,
                    h, h_dead, &opp_indices, &elig, 0,
                );
                case_c_total += lev.pot_l * share;
            }
        }
        let fac_cfv = (static_cash - traverser_stake) * tvrp_h + case_c_total;
        let diff = (fac_cfv as f32 - bf_cfv[h]).abs() as f64;
        if diff > max_abs { max_abs = diff; worst_h = h; }
        eprintln!("p0 h={} bf={:.6} fac={:.6} diff={:.3e}",
            h, bf_cfv[h], fac_cfv, diff);
    }
    eprintln!("p0 max_abs={:.3e}", max_abs);
    assert!(max_abs < 1e-3,
        "Side-pot CFV factored vs brute: max_abs={:.3e} at h={}", max_abs, worst_h);
}

#[test]
fn per_level_factored_cfv_matches_brute_force_with_fold() {
    // 4-player with 1 fold: p2 folded, contribs [25, 25, 5, 25], single level.
    let (hand_cards, hand_strength, reach, nh, np_u) = make_4p_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);
    let np = np_u as u8;

    let contributions = vec![25i32, 25, 5, 25];
    let fold_mask: u16 = 1u16 << 2; // p2 folded
    let starting_pot = (np_u as i32) * 5;

    let opp_indices: Vec<usize> = (0..3).collect();
    let traverser = 0usize;
    let levels = build_level_info(&contributions, fold_mask, traverser, np_u, starting_pot);
    eprintln!("Fold scenario p0 levels: {:?}", levels);

    let mut sp_pairs: Vec<(u16, u16)> = (0..nh).map(|h| (hand_strength[h], h as u16)).collect();
    sp_pairs.sort_by_key(|&(s, _)| s);
    let sorted_pl_str: Vec<u16> = sp_pairs.iter().map(|&(s, _)| s).collect();
    let sorted_pl_idx: Vec<u16> = sp_pairs.iter().map(|&(_, i)| i).collect();
    let num_opp = np_u - 1;
    let mut sorted_opp_str = vec![0u16; num_opp * nh];
    let mut sorted_opp_idx = vec![0u16; num_opp * nh];
    for oi in 0..num_opp {
        for h in 0..nh {
            sorted_opp_str[oi * nh + h] = sorted_pl_str[h];
            sorted_opp_idx[oi * nh + h] = sorted_pl_idx[h];
        }
    }
    let opp_reach_bf: Vec<&[f32]> = (0..num_opp).map(|oi| reach[oi].as_slice()).collect();
    let bf_cfv = side_pot_showdown_cfv(
        &opp_reach_bf, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &sorted_pl_str, &sorted_pl_idx,
        &contributions, fold_mask, traverser, np, starting_pot,
    );

    let c_t = contributions[traverser];
    let traverser_stake = starting_pot as f64 / np_u as f64 + c_t as f64;
    let mut max_abs = 0.0f64;
    let mut worst_h = 0;
    for h in 0..nh {
        let h_dead = (1u64 << hand_cards[h*2]) | (1u64 << hand_cards[h*2+1]);
        let tvrp_h = tvrp(&reach_views, &hand_cards, nh, h, h_dead);
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
                let elig: Vec<bool> = (0..num_opp).map(|oi| (lev.elig_opps >> oi) & 1 != 0).collect();
                let share = factored_share_at_level(
                    &masses, &reach_views, &hand_strength,
                    h, h_dead, &opp_indices, &elig, 0,
                );
                case_c_total += lev.pot_l * share;
            }
        }
        let fac_cfv = (static_cash - traverser_stake) * tvrp_h + case_c_total;
        let diff = (fac_cfv as f32 - bf_cfv[h]).abs() as f64;
        if diff > max_abs { max_abs = diff; worst_h = h; }
        eprintln!("fold p0 h={} bf={:.6} fac={:.6} diff={:.3e}",
            h, bf_cfv[h], fac_cfv, diff);
    }
    eprintln!("fold max_abs={:.3e}", max_abs);
    assert!(max_abs < 1e-3,
        "Fold scenario CFV factored vs brute: max_abs={:.3e} at h={}", max_abs, worst_h);
}

#[allow(dead_code)]
fn unused() { let _ = index_to_card_pair; }
