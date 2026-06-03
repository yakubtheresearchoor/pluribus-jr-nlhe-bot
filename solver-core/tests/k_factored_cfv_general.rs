// Generic K-opp factored CFV via recursive K-1 expansion.
//
// The recursion: fix one opponent's hand, add its cards to the dead mask,
// recurse on K-1 opponents. Base case at K=2 uses the PAIR-decomposition
// formula (BB, BT, TB, TT with extended-mask I-E). Validated at K=3 in
// k3_factored_cfv.rs; this file extends to K=4 and K=5 and validates
// against brute-force on conflict-heavy small-nh games.
//
// Critical-path docket item 1+2:
//   #1: extend the proven K=3 pattern to K=4 and K=5
//   #2: validate K=5 factored against brute-force on small games
//       including the three-way-shared-card configuration that broke
//       the earlier matching formula
//
// Lower risk than #1 sounds because the K=3 pattern is validated.
// Real value is in #2: the K=5 factored formula is genuinely different
// code from K=5 brute-force, so they actually test each other now
// (the earlier ie_k3_tvrp_matches_bruteforce test was tautological —
// brute vs brute — because tvrp_brute was used as both sides).

use solver_core::card::index_to_card_pair;
use solver_core::solver::showdown::{precompute_opp_masses, OppMasses};

// ---- Brute-force reference for arbitrary K ---------------------------------

/// Brute-force K-opp showdown CFV share at one h_player.
/// Returns Σ over valid (g_0, ..., g_{K-1}) of Π r_i[g_i] · 1/(1+tied_count)
/// where tied_count = #(g_i with s_i == h_str), provided all s_i ≤ h_str.
fn brute_force_share(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    hand_strength: &[u16],
    nh: usize,
    h: usize,
    h_dead_mask: u64,
) -> f64 {
    let k = opp_reach.len();
    let g_mask: Vec<u64> = (0..nh)
        .map(|g| (1u64 << hand_cards[g * 2]) | (1u64 << hand_cards[g * 2 + 1]))
        .collect();
    let h_str = hand_strength[h];

    fn recurse(
        oi: usize, k: usize, nh: usize,
        mask_so_far: u64, reach_so_far: f64, tied: u32, max_so_far: u16,
        h_str: u16,
        opp_reach: &[&[f32]], g_mask: &[u64], hand_strength: &[u16],
        accum: &mut f64,
    ) {
        if oi == k {
            if max_so_far > h_str { return; }
            *accum += reach_so_far / (1.0 + tied as f64);
            return;
        }
        for g in 0..nh {
            if g_mask[g] & mask_so_far != 0 { continue; }
            let r = opp_reach[oi][g] as f64;
            if r == 0.0 { continue; }
            let s = hand_strength[g];
            if s > h_str { continue; }
            let (nm, nt) = if s == h_str { (h_str.max(max_so_far), tied + 1) } else { (max_so_far, tied) };
            recurse(oi + 1, k, nh,
                mask_so_far | g_mask[g], reach_so_far * r, nt, nm,
                h_str, opp_reach, g_mask, hand_strength,
                accum);
        }
    }

    let mut sum = 0.0f64;
    recurse(0, k, nh, h_dead_mask, 1.0, 0, 0,
        h_str, opp_reach, &g_mask, hand_strength,
        &mut sum);
    sum
}

// ---- Generic recursive factored CFV (K ≥ 3) --------------------------------

/// K=2 base case: PAIR-decomposition with extended-mask I-E.
/// Recomputes B/T masses at the extended mask on the fly (O(nh + 52) per
/// call). This is simpler than I-E expansion from base masses; sufficient
/// for the correctness validation at small nh. Production GPU will use
/// precomputed base masses + I-E expansion for speed.
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

    // Recompute B/T totals + per-card + same-hand sums at the extended mask.
    let mut ba = 0.0f64; let mut ta = 0.0f64;
    let mut bb_tot = 0.0f64; let mut tb_tot = 0.0f64;
    let mut ba_pc = [0.0f64; 52]; let mut ta_pc = [0.0f64; 52];
    let mut bb_pc = [0.0f64; 52]; let mut tb_pc = [0.0f64; 52];
    let mut h_bb = 0.0f64; let mut h_tt = 0.0f64;
    let mut h_bb_pc = [0.0f64; 52]; let mut h_tt_pc = [0.0f64; 52];

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
            let prod = ra * rb;
            h_bb += prod; h_bb_pc[gc1] += prod; h_bb_pc[gc2] += prod;
        } else if s == h_str {
            ta += ra; ta_pc[gc1] += ra; ta_pc[gc2] += ra;
            tb_tot += rb; tb_pc[gc1] += rb; tb_pc[gc2] += rb;
            let prod = ra * rb;
            h_tt += prod; h_tt_pc[gc1] += prod; h_tt_pc[gc2] += prod;
        }
        // s > h_str: contributes to none (traverser loses).
    }
    let _ = h_bb_pc; let _ = h_tt_pc;

    // Edge sums.
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

/// Generic K-opp factored share, recursive K-1 expansion.
///
/// For K = 2: delegate to the PAIR-decomposition base case.
/// For K ≥ 3: enumerate first opp (in B or T category, skip S), recurse
/// on remaining opps with extended dead mask and incremented tied count.
fn factored_share(
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
        sum += r * factored_share(
            masses, opp_reach, hand_strength, h, h_dead_mask | g_m,
            &opp_indices[1..], new_tied,
        );
    }
    sum
}

// ---- Test setups -----------------------------------------------------------

/// 10 hands with deliberate three-way-shared-card structure:
///   h=0 {0,1}  ← shares card 0 with h=4 AND h=5  (3-way on card 0)
///   h=1 {2,3}  ← shares card 2 with h=5 AND h=6  (3-way on card 2)
///   h=2 {4,5}
///   h=3 {6,7}
///   h=4 {0,8}  ← shares 0 with h=0, h=5
///   h=5 {0,2}  ← shares 0 with h=0, h=4; shares 2 with h=1, h=6
///   h=6 {2,9}  ← shares 2 with h=1, h=5
///   h=7 {10,11}
///   h=8 {12,13}
///   h=9 {14,15}
///
/// Strengths chosen so the B/T/S categories all see action at multiple
/// h_player positions and so the tied-at-top branch is exercised.
fn make_three_way_shared_card_setup(num_opp: usize) -> (Vec<u8>, Vec<u16>, Vec<Vec<f32>>, usize) {
    let nh = 10;
    let hand_cards: Vec<u8> = vec![
        0, 1,
        2, 3,
        4, 5,
        6, 7,
        0, 8,
        0, 2,
        2, 9,
        10, 11,
        12, 13,
        14, 15,
    ];
    // Two pairs of tied hands: (h=2, h=4) tied at 70, (h=6, h=7) tied at 50.
    let hand_strength: Vec<u16> = vec![100, 90, 70, 80, 70, 60, 50, 50, 40, 30];
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.3 + 0.6 * ((h + oi * 3) % 7) as f32 / 7.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh)
}

// ---- Tests -----------------------------------------------------------------

#[test]
fn k3_general_matches_bruteforce_on_triangle() {
    let (hand_cards, hand_strength, reach, nh) = make_three_way_shared_card_setup(3);
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    let opp_indices: Vec<usize> = (0..3).collect();
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;

    for h in 0..nh {
        let hc1 = hand_cards[h * 2] as usize;
        let hc2 = hand_cards[h * 2 + 1] as usize;
        let h_dead = (1u64 << hc1) | (1u64 << hc2);

        let bf = brute_force_share(&reach_views, &hand_cards, &hand_strength, nh, h, h_dead);
        let fac = factored_share(&masses, &reach_views, &hand_strength, h, h_dead, &opp_indices, 0);

        let diff = (fac - bf).abs();
        let scale = bf.abs().max(1e-9);
        let rel = diff / scale;
        if diff > max_abs { max_abs = diff; }
        if rel > max_rel { max_rel = rel; }

        eprintln!("K=3 h={} cards=({},{}) s={}  bf={:.6} fac={:.6} diff={:.3e}",
            h, hand_cards[h*2], hand_cards[h*2+1], hand_strength[h], bf, fac, diff);
    }

    eprintln!("K=3 max_abs={:.3e}, max_rel={:.3e}", max_abs, max_rel);
    assert!(max_abs < 1e-5 || max_rel < 1e-5,
        "K=3 factored does not match brute-force on three-way-shared-card setup");
}

#[test]
fn k4_factored_matches_bruteforce_on_three_way_shared_cards() {
    let (hand_cards, hand_strength, reach, nh) = make_three_way_shared_card_setup(4);
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    let opp_indices: Vec<usize> = (0..4).collect();
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut worst_h = 0usize;

    eprintln!("=== K=4 factored vs brute-force on 3-way-shared-card setup ===");
    for h in 0..nh {
        let hc1 = hand_cards[h * 2] as usize;
        let hc2 = hand_cards[h * 2 + 1] as usize;
        let h_dead = (1u64 << hc1) | (1u64 << hc2);

        let bf = brute_force_share(&reach_views, &hand_cards, &hand_strength, nh, h, h_dead);
        let fac = factored_share(&masses, &reach_views, &hand_strength, h, h_dead, &opp_indices, 0);

        let diff = (fac - bf).abs();
        let scale = bf.abs().max(1e-9);
        let rel = diff / scale;
        if diff > max_abs { max_abs = diff; worst_h = h; }
        if rel > max_rel { max_rel = rel; }

        eprintln!("K=4 h={} cards=({},{}) s={}  bf={:.6} fac={:.6} diff={:.3e}",
            h, hand_cards[h*2], hand_cards[h*2+1], hand_strength[h], bf, fac, diff);
    }
    eprintln!("K=4 max_abs={:.3e} at h={}, max_rel={:.3e}", max_abs, worst_h, max_rel);
    assert!(max_abs < 1e-5 || max_rel < 1e-5,
        "K=4 factored does not match brute-force: max_abs={:.3e} at h={}, max_rel={:.3e}",
        max_abs, worst_h, max_rel);
}

#[test]
fn k5_factored_matches_bruteforce_on_three_way_shared_cards() {
    let (hand_cards, hand_strength, reach, nh) = make_three_way_shared_card_setup(5);
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    let opp_indices: Vec<usize> = (0..5).collect();
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut worst_h = 0usize;

    eprintln!("=== K=5 factored vs brute-force on 3-way-shared-card setup ===");
    eprintln!("Hand structure: h=0,4,5 all contain card 0 (3-way); h=1,5,6 all contain card 2 (3-way)");

    for h in 0..nh {
        let hc1 = hand_cards[h * 2] as usize;
        let hc2 = hand_cards[h * 2 + 1] as usize;
        let h_dead = (1u64 << hc1) | (1u64 << hc2);

        let bf = brute_force_share(&reach_views, &hand_cards, &hand_strength, nh, h, h_dead);
        let fac = factored_share(&masses, &reach_views, &hand_strength, h, h_dead, &opp_indices, 0);

        let diff = (fac - bf).abs();
        let scale = bf.abs().max(1e-9);
        let rel = diff / scale;
        if diff > max_abs { max_abs = diff; worst_h = h; }
        if rel > max_rel { max_rel = rel; }

        eprintln!("K=5 h={} cards=({},{}) s={}  bf={:.6} fac={:.6} diff={:.3e}",
            h, hand_cards[h*2], hand_cards[h*2+1], hand_strength[h], bf, fac, diff);
    }
    eprintln!("K=5 max_abs={:.3e} at h={}, max_rel={:.3e}", max_abs, worst_h, max_rel);
    assert!(max_abs < 1e-5 || max_rel < 1e-5,
        "K=5 factored does not match brute-force: max_abs={:.3e} at h={}, max_rel={:.3e}",
        max_abs, worst_h, max_rel);
}

#[test]
fn k5_strongest_hand_boundary() {
    // h=0 is unique strongest (strength 100). T_i=0, S_i=0, B_i=R_i for all opps.
    // Recursive S-branches at every opp must drop cleanly without producing NaN.
    let (hand_cards, hand_strength, reach, nh) = make_three_way_shared_card_setup(5);
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    let opp_indices: Vec<usize> = (0..5).collect();
    let h = 0;
    let hc1 = hand_cards[h * 2] as usize;
    let hc2 = hand_cards[h * 2 + 1] as usize;
    let h_dead = (1u64 << hc1) | (1u64 << hc2);

    let bf = brute_force_share(&reach_views, &hand_cards, &hand_strength, nh, h, h_dead);
    let fac = factored_share(&masses, &reach_views, &hand_strength, h, h_dead, &opp_indices, 0);

    eprintln!("K=5 STRONGEST h=0: bf={:.6} fac={:.6} diff={:.3e}", bf, fac, (fac - bf).abs());
    assert!(fac.is_finite(), "K=5 strongest must not be NaN/Inf");
    assert!((fac - bf).abs() < 1e-5, "K=5 strongest boundary fails");
}

#[test]
fn k5_weakest_hand_b_zero_boundary() {
    // h=9 is unique weakest (strength 30). B_i=0, T_i=0, S_i=R_i for all opps.
    // Recursion must terminate at exactly 0 — every g_0 enumerated is in S
    // and contributes nothing. Tests the most-likely subtle-error site.
    let (hand_cards, hand_strength, reach, nh) = make_three_way_shared_card_setup(5);
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    let opp_indices: Vec<usize> = (0..5).collect();
    let h = 9;
    let hc1 = hand_cards[h * 2] as usize;
    let hc2 = hand_cards[h * 2 + 1] as usize;
    let h_dead = (1u64 << hc1) | (1u64 << hc2);

    let bf = brute_force_share(&reach_views, &hand_cards, &hand_strength, nh, h, h_dead);
    let fac = factored_share(&masses, &reach_views, &hand_strength, h, h_dead, &opp_indices, 0);

    eprintln!("K=5 WEAKEST h=9 (B_i=0 boundary): bf={:.6} fac={:.6}", bf, fac);
    assert_eq!(bf, 0.0, "K=5 brute-force at weakest h must be 0");
    assert!(fac.abs() < 1e-9, "K=5 factored at B_i=0 boundary must be 0, got {}", fac);
    assert!(fac.is_finite());
}

#[test]
fn k5_tied_at_top_boundary() {
    // h=2 strength 70, tied with h=4. T_i and B_i both non-zero.
    let (hand_cards, hand_strength, reach, nh) = make_three_way_shared_card_setup(5);
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    let opp_indices: Vec<usize> = (0..5).collect();
    let h = 2;
    let hc1 = hand_cards[h * 2] as usize;
    let hc2 = hand_cards[h * 2 + 1] as usize;
    let h_dead = (1u64 << hc1) | (1u64 << hc2);

    let bf = brute_force_share(&reach_views, &hand_cards, &hand_strength, nh, h, h_dead);
    let fac = factored_share(&masses, &reach_views, &hand_strength, h, h_dead, &opp_indices, 0);

    let diff = (fac - bf).abs();
    eprintln!("K=5 TIED-AT-TOP h=2 (tied with h=4): bf={:.6} fac={:.6} diff={:.3e}", bf, fac, diff);
    assert!(diff < 1e-5, "K=5 tied-at-top boundary fails: bf={} fac={} diff={}", bf, fac, diff);
}

#[allow(dead_code)]
fn unused() { let _ = index_to_card_pair; }
