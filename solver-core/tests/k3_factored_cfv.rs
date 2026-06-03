// K=3 factored showdown CFV via recursive K=2 expansion with B/T/S
// stratification — the leading exact candidate for K≥3 production.
//
// Spec from the spec (verbatim):
//   share_K(h, dead, tied) = Σ over g_K of {
//     S_K: 0,
//     T_K: r · share_{K-1}(h, dead ∪ g_K_cards, tied+1),
//     B_K: r · share_{K-1}(h, dead ∪ g_K_cards, tied)
//   }
//   base case: share_0(tied) = 1 / (1 + tied)
//
// Validation: brute-force reference on a 4p toy with hand strengths
// chosen to exercise three boundary cases:
//   (1) STRONGEST hand h (B_i large, T_i possibly 0, S_i = 0)
//   (2) WEAKEST hand h (B_i = 0 — the boundary the spec called out)
//   (3) Mixed middle case (B/T/S all non-zero)
//
// Tolerance: factored vs brute-force on small nh should match to ~1e-5
// relative (f32 floor + accumulation noise). Exact would be float-precision
// zero; any visible drift means the formula or its bookkeeping is off.

use solver_core::card::{card_pair_to_index, index_to_card_pair};
use solver_core::solver::showdown::{precompute_opp_masses, OppMasses};

// ---- Brute-force reference -------------------------------------------------

/// Brute-force K=3 showdown CFV at a single h_player.
///
/// CFV(h) = Σ over (g_0, g_1, g_2) ordered, pairwise card-disjoint, no
/// conflict with h, no conflict with dead_mask of:
///     r_0[g_0] r_1[g_1] r_2[g_2] · share(h, g_0, g_1, g_2)
///
/// where share = 1/(1 + tied_count) if max(s_0, s_1, s_2) ≤ h_str else 0,
/// and tied_count = # of g_i with s_i == h_str.
///
/// (Caller multiplies by pot and subtracts traverser_stake · TVRP. This
/// function returns just the share-weighted reach product, the part the
/// factored formula computes.)
fn brute_force_share(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    hand_strength: &[u16],
    nh: usize,
    h: usize,
    h_dead_mask: u64, // includes h's cards + base dead mask
) -> f64 {
    let g_mask: Vec<u64> = (0..nh)
        .map(|g| (1u64 << hand_cards[g * 2]) | (1u64 << hand_cards[g * 2 + 1]))
        .collect();
    let h_str = hand_strength[h];

    fn recurse(
        oi: usize,
        nh: usize,
        mask_so_far: u64,
        reach_so_far: f64,
        tied: u32,
        max_str_so_far: u16,
        h_str: u16,
        opp_reach: &[&[f32]],
        g_mask: &[u64],
        hand_strength: &[u16],
        accum: &mut f64,
    ) {
        if oi == 3 {
            if max_str_so_far > h_str { return; }
            // tied_count for share = tied opps + 1 (the traverser himself).
            *accum += reach_so_far / (1.0 + tied as f64);
            return;
        }
        for g in 0..nh {
            if g_mask[g] & mask_so_far != 0 { continue; }
            let r = opp_reach[oi][g] as f64;
            if r == 0.0 { continue; }
            let s = hand_strength[g];
            if s > h_str { continue; } // strictly stronger → share=0 globally
            let (new_max, new_tied) = if s == h_str {
                (h_str.max(max_str_so_far), tied + 1)
            } else {
                (max_str_so_far, tied)
            };
            recurse(
                oi + 1, nh,
                mask_so_far | g_mask[g],
                reach_so_far * r, new_tied, new_max,
                h_str, opp_reach, g_mask, hand_strength,
                accum,
            );
        }
    }

    let mut sum = 0.0f64;
    recurse(
        0, nh, h_dead_mask, 1.0, 0u32, 0u16,
        h_str, opp_reach, &g_mask, hand_strength,
        &mut sum,
    );
    sum
}

// ---- Factored reference ----------------------------------------------------

/// Look up r_i[h_{c1,c2}] AND check that h_{c1,c2}'s strength is in the
/// requested category (B = strictly weaker than h_str, T = equal to h_str).
fn lookup_pair_reach_in_category(
    opp_reach: &[f32],
    hand_index: &[i32],
    hand_strength: &[u16],
    h_str: u16,
    c1: usize,
    c2: usize,
    want_b: bool, // true = want B category, false = want T
) -> f64 {
    if c1 == c2 { return 0.0; }
    let idx = hand_index[c1 * 52 + c2];
    if idx < 0 { return 0.0; }
    let g = idx as usize;
    let s = hand_strength[g];
    let in_b = s < h_str;
    let in_t = s == h_str;
    if (want_b && in_b) || (!want_b && in_t) {
        opp_reach[g] as f64
    } else {
        0.0
    }
}

/// K=3 factored CFV (share part) via recursive K=2 expansion.
///
/// share_K=3(h) = Σ over g_0 valid, s_0 ≤ h_str of
///                  r_0[g_0] · X_inner(h, g_0, tied_from_0)
///
/// X_inner uses extended-mask PAIRs:
///   X_inner(t_0) = PAIR_BB / (1+t_0) + (PAIR_BT + PAIR_TB) / (2+t_0)
///                 + PAIR_TT / (3+t_0)
///
/// where each PAIR_XY = X_1_ext · Y_2_ext − Σ_c X_1_ext^(c) · Y_2_ext^(c)
///                    + (X==Y == B: H_BB_ext;  X==Y == T: H_TT_ext;  else 0)
///
/// Same-hand correction for cross-category PAIRs (BT, TB) is identically
/// zero — a single hand has one strength, can't be in both B and T.
fn factored_k3_share(
    masses: &OppMasses,
    opp_reach: &[&[f32]],
    hand_strength: &[u16],
    h: usize,
    h_dead_mask: u64,
) -> f64 {
    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;
    let h_str = hand_strength[h];

    // Base K=2 masses for opps 1 and 2 at this h_player (already stratified).
    let b1 = masses.b[1 * nh + h] as f64;
    let t1 = masses.t[1 * nh + h] as f64;
    let b2 = masses.b[2 * nh + h] as f64;
    let t2 = masses.t[2 * nh + h] as f64;

    // Per-card B/T masses for opps 1 and 2.
    let b1_pc = |c: usize| masses.b_per_card[(1 * nh + h) * 52 + c] as f64;
    let t1_pc = |c: usize| masses.t_per_card[(1 * nh + h) * 52 + c] as f64;
    let b2_pc = |c: usize| masses.b_per_card[(2 * nh + h) * 52 + c] as f64;
    let t2_pc = |c: usize| masses.t_per_card[(2 * nh + h) * 52 + c] as f64;

    // H_BB and H_TT (and per-card variants), computed on the fly once per
    // h_player. K-quadratic in opp pairs so not stored in OppMasses.
    let mut h_bb = 0.0f64;
    let mut h_tt = 0.0f64;
    let mut h_bb_pc = [0.0f64; 52];
    let mut h_tt_pc = [0.0f64; 52];
    for g in 0..nh {
        let gc1 = hand_cards[g * 2] as usize;
        let gc2 = hand_cards[g * 2 + 1] as usize;
        let g_m = (1u64 << gc1) | (1u64 << gc2);
        if g_m & h_dead_mask != 0 { continue; }
        let s = hand_strength[g];
        let r1 = opp_reach[1][g] as f64;
        let r2 = opp_reach[2][g] as f64;
        let prod = r1 * r2;
        if prod == 0.0 { continue; }
        if s < h_str {
            h_bb += prod;
            h_bb_pc[gc1] += prod;
            h_bb_pc[gc2] += prod;
        } else if s == h_str {
            h_tt += prod;
            h_tt_pc[gc1] += prod;
            h_tt_pc[gc2] += prod;
        }
        // s > h_str: contributes to neither.
    }

    // Outer loop over g_0.
    let mut share = 0.0f64;

    for g0 in 0..nh {
        let g0c1 = hand_cards[g0 * 2] as usize;
        let g0c2 = hand_cards[g0 * 2 + 1] as usize;
        let g0_m = (1u64 << g0c1) | (1u64 << g0c2);
        if g0_m & h_dead_mask != 0 { continue; }
        let r0_g0 = opp_reach[0][g0] as f64;
        if r0_g0 == 0.0 { continue; }
        let s0 = hand_strength[g0];
        if s0 > h_str { continue; } // S category → contributes 0
        let t0: u32 = if s0 == h_str { 1 } else { 0 };
        let g0_in_b = s0 < h_str;
        let g0_in_t = s0 == h_str;

        // g_0-aware same-hand corrections for B/T per-opp at the {g0c1,g0c2} pair.
        let r1_g0 = opp_reach[1][g0] as f64;
        let r2_g0 = opp_reach[2][g0] as f64;
        let b1_pair_g0 = if g0_in_b { r1_g0 } else { 0.0 };
        let b2_pair_g0 = if g0_in_b { r2_g0 } else { 0.0 };
        let t1_pair_g0 = if g0_in_t { r1_g0 } else { 0.0 };
        let t2_pair_g0 = if g0_in_t { r2_g0 } else { 0.0 };

        // Extended-mask totals for opps 1 and 2 in each strength category.
        let b1_ext = b1 - b1_pc(g0c1) - b1_pc(g0c2) + b1_pair_g0;
        let b2_ext = b2 - b2_pc(g0c1) - b2_pc(g0c2) + b2_pair_g0;
        let t1_ext = t1 - t1_pc(g0c1) - t1_pc(g0c2) + t1_pair_g0;
        let t2_ext = t2 - t2_pc(g0c1) - t2_pc(g0c2) + t2_pair_g0;

        // Edge sums over c ∉ extended dead.
        let ext_mask = h_dead_mask | g0_m;
        let mut bb_edge = 0.0f64;
        let mut bt_edge = 0.0f64;
        let mut tb_edge = 0.0f64;
        let mut tt_edge = 0.0f64;

        for c in 0..52usize {
            if (1u64 << c) & ext_mask != 0 { continue; }
            // M_i_ext^(c) = M_i^(c) - {pair reach with g0c1 AND c, in this category}
            //                       - {pair reach with g0c2 AND c, in this category}
            let b1_g0c1_c = lookup_pair_reach_in_category(
                opp_reach[1], &masses.hand_index, hand_strength, h_str, g0c1, c, true);
            let b1_g0c2_c = lookup_pair_reach_in_category(
                opp_reach[1], &masses.hand_index, hand_strength, h_str, g0c2, c, true);
            let b2_g0c1_c = lookup_pair_reach_in_category(
                opp_reach[2], &masses.hand_index, hand_strength, h_str, g0c1, c, true);
            let b2_g0c2_c = lookup_pair_reach_in_category(
                opp_reach[2], &masses.hand_index, hand_strength, h_str, g0c2, c, true);
            let t1_g0c1_c = lookup_pair_reach_in_category(
                opp_reach[1], &masses.hand_index, hand_strength, h_str, g0c1, c, false);
            let t1_g0c2_c = lookup_pair_reach_in_category(
                opp_reach[1], &masses.hand_index, hand_strength, h_str, g0c2, c, false);
            let t2_g0c1_c = lookup_pair_reach_in_category(
                opp_reach[2], &masses.hand_index, hand_strength, h_str, g0c1, c, false);
            let t2_g0c2_c = lookup_pair_reach_in_category(
                opp_reach[2], &masses.hand_index, hand_strength, h_str, g0c2, c, false);

            let b1c = b1_pc(c) - b1_g0c1_c - b1_g0c2_c;
            let b2c = b2_pc(c) - b2_g0c1_c - b2_g0c2_c;
            let t1c = t1_pc(c) - t1_g0c1_c - t1_g0c2_c;
            let t2c = t2_pc(c) - t2_g0c1_c - t2_g0c2_c;

            bb_edge += b1c * b2c;
            bt_edge += b1c * t2c;
            tb_edge += t1c * b2c;
            tt_edge += t1c * t2c;
        }

        // Extended-mask H_BB and H_TT.
        // Same-hand correction at {g0c1, g0c2} = r_1[g_0]·r_2[g_0] · 1[g_0 ∈ category].
        let h_bb_g0g0 = if g0_in_b { r1_g0 * r2_g0 } else { 0.0 };
        let h_tt_g0g0 = if g0_in_t { r1_g0 * r2_g0 } else { 0.0 };
        let h_bb_ext = h_bb - h_bb_pc[g0c1] - h_bb_pc[g0c2] + h_bb_g0g0;
        let h_tt_ext = h_tt - h_tt_pc[g0c1] - h_tt_pc[g0c2] + h_tt_g0g0;

        let pair_bb = b1_ext * b2_ext - bb_edge + h_bb_ext;
        let pair_bt = b1_ext * t2_ext - bt_edge; // no same-hand correction
        let pair_tb = t1_ext * b2_ext - tb_edge;
        let pair_tt = t1_ext * t2_ext - tt_edge + h_tt_ext;

        // X_inner per category factor:
        let f_bb = 1.0 / (1.0 + t0 as f64);
        let f_bt = 1.0 / (2.0 + t0 as f64);
        let f_tb = 1.0 / (2.0 + t0 as f64);
        let f_tt = 1.0 / (3.0 + t0 as f64);

        let x_inner = pair_bb * f_bb + pair_bt * f_bt + pair_tb * f_tb + pair_tt * f_tt;
        share += r0_g0 * x_inner;
    }

    share
}

// ---- Test helpers ----------------------------------------------------------

fn make_toy_setup() -> (
    Vec<u8>,        // hand_cards [nh*2]
    Vec<u16>,       // hand_strength [nh]
    Vec<Vec<f32>>,  // opp_reach [K][nh]
    usize,          // nh
) {
    // 8 hands; pairwise mostly disjoint to ensure brute-force is non-trivial.
    // Strengths chosen with a unique max and a unique min, plus an interior
    // tie pair so the T-branch is exercised.
    //   h=0 cards {0,1}  strength 100   (strongest, unique)
    //   h=1 cards {2,3}  strength  80
    //   h=2 cards {4,5}  strength  70   ← tied with h=3
    //   h=3 cards {6,7}  strength  70   ← tied with h=2
    //   h=4 cards {8,9}  strength  60
    //   h=5 cards {10,11} strength 50
    //   h=6 cards {12,13} strength 40
    //   h=7 cards {14,15} strength 30  (weakest, unique)
    //
    // All 8 hands are pairwise disjoint (use distinct card pairs); ensures
    // every K-tuple over distinct h is valid (no card-sharing rejection
    // hides the share-formula errors).
    let nh = 8;
    let hand_cards: Vec<u8> = vec![
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    ];
    let hand_strength: Vec<u16> = vec![100, 80, 70, 70, 60, 50, 40, 30];
    let reach: Vec<Vec<f32>> = (0..3).map(|oi| {
        (0..nh).map(|h| 0.4 + 0.5 * ((h + oi * 3) % 7) as f32 / 7.0).collect()
    }).collect();
    (hand_cards, hand_strength, reach, nh)
}

#[test]
fn k3_factored_cfv_matches_brute_force() {
    let (hand_cards, hand_strength, reach, nh) = make_toy_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    let h_dead_mask_base = 0u64;

    eprintln!("=== K=3 factored CFV share vs brute-force ===");
    eprintln!("Hand strengths: {:?}", hand_strength);
    eprintln!("Reach (opp 0): {:?}", reach[0]);

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut worst_h = 0;

    for h in 0..nh {
        let hc1 = hand_cards[h * 2] as usize;
        let hc2 = hand_cards[h * 2 + 1] as usize;
        let h_dead = h_dead_mask_base | (1u64 << hc1) | (1u64 << hc2);

        let bf = brute_force_share(&reach_views, &hand_cards, &hand_strength, nh, h, h_dead);
        let factored = factored_k3_share(&masses, &reach_views, &hand_strength, h, h_dead);

        let diff = (factored - bf).abs();
        let scale = bf.abs().max(1e-9);
        let rel = diff / scale;
        if diff > max_abs { max_abs = diff; worst_h = h; }
        if rel > max_rel { max_rel = rel; }

        // Strength-category boundary labels for read-out.
        let strongest = hand_strength.iter().max().copied().unwrap_or(0);
        let weakest = hand_strength.iter().min().copied().unwrap_or(0);
        let label = if hand_strength[h] == strongest { " [STRONGEST]" }
                    else if hand_strength[h] == weakest { " [WEAKEST — B_i=0 boundary]" }
                    else { "" };

        eprintln!(
            "h={} cards=({},{}) s={}{:30}  bf={:.9} factored={:.9} diff={:.3e} rel={:.3e}",
            h, hand_cards[h*2], hand_cards[h*2+1], hand_strength[h], label, bf, factored, diff, rel
        );
    }

    eprintln!("\nmax_abs={:.3e} at h={}, max_rel={:.3e}", max_abs, worst_h, max_rel);

    assert!(
        max_abs < 1e-5 || max_rel < 1e-5,
        "K=3 factored CFV does not match brute-force: max_abs={:.3e}, max_rel={:.3e}, worst_h={}",
        max_abs, max_rel, worst_h
    );
}

#[test]
fn k3_factored_cfv_strongest_hand_boundary() {
    // When h is the unique strongest hand, B_i = total reach over weaker
    // hands (large), T_i = 0 (no ties with the strongest), S_i = 0. The
    // recursion's S-branch is exercised on 0; only the B-branch contributes.
    let (hand_cards, hand_strength, reach, nh) = make_toy_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    // h=0 is strongest (strength 100, unique).
    let h = 0;
    let hc1 = hand_cards[h * 2] as usize;
    let hc2 = hand_cards[h * 2 + 1] as usize;
    let h_dead = (1u64 << hc1) | (1u64 << hc2);

    // Confirm boundary: T_i = 0 for both opps at this h.
    assert_eq!(masses.t[1 * nh + h], 0.0, "T_1 must be 0 at strongest unique h");
    assert_eq!(masses.t[2 * nh + h], 0.0, "T_2 must be 0 at strongest unique h");
    // B_i = total reach over valid opps (= R_i since S_i = 0 too).
    let total_r1: f32 = (0..nh).map(|g| {
        let g_m = (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]);
        if g_m & h_dead != 0 { 0.0 } else { reach[1][g] }
    }).sum();
    let total_r2: f32 = (0..nh).map(|g| {
        let g_m = (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]);
        if g_m & h_dead != 0 { 0.0 } else { reach[2][g] }
    }).sum();
    eprintln!("STRONGEST boundary: B_1={}, B_2={}, T_1={}, T_2={}, R_1={}, R_2={}",
        masses.b[1*nh+h], masses.b[2*nh+h], masses.t[1*nh+h], masses.t[2*nh+h], total_r1, total_r2);
    assert!((masses.b[1*nh+h] - total_r1).abs() < 1e-5);
    assert!((masses.b[2*nh+h] - total_r2).abs() < 1e-5);

    let bf = brute_force_share(&reach_views, &hand_cards, &hand_strength, nh, h, h_dead);
    let factored = factored_k3_share(&masses, &reach_views, &hand_strength, h, h_dead);
    let diff = (factored - bf).abs();
    eprintln!("STRONGEST h={}: bf={:.9} factored={:.9} diff={:.3e}", h, bf, factored, diff);
    assert!(diff < 1e-5, "STRONGEST boundary fails: bf={}, factored={}, diff={}",
        bf, factored, diff);
}

#[test]
fn k3_factored_cfv_weakest_hand_boundary() {
    // When h is the unique weakest hand, B_i = 0 (nothing weaker), T_i = 0
    // (no ties — unique weakest), S_i = total reach. The share recursion's
    // S branch returns 0 for every g_0, so the whole share collapses to 0.
    // Verifying both brute-force AND factored return 0 ensures the recursion
    // handles the all-S case without producing NaN, Inf, or accumulator
    // garbage from extended-mask cancellations.
    let (hand_cards, hand_strength, reach, nh) = make_toy_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    let h = 7; // weakest (strength 30, unique)
    let hc1 = hand_cards[h * 2] as usize;
    let hc2 = hand_cards[h * 2 + 1] as usize;
    let h_dead = (1u64 << hc1) | (1u64 << hc2);

    assert_eq!(masses.b[1*nh+h], 0.0, "B_1 must be 0 at weakest unique h");
    assert_eq!(masses.b[2*nh+h], 0.0, "B_2 must be 0 at weakest unique h");
    assert_eq!(masses.t[1*nh+h], 0.0, "T_1 must be 0 at weakest unique h");
    assert_eq!(masses.t[2*nh+h], 0.0, "T_2 must be 0 at weakest unique h");

    let bf = brute_force_share(&reach_views, &hand_cards, &hand_strength, nh, h, h_dead);
    let factored = factored_k3_share(&masses, &reach_views, &hand_strength, h, h_dead);
    eprintln!("WEAKEST h={} (B_i=0 boundary): bf={:.9} factored={:.9}", h, bf, factored);
    assert_eq!(bf, 0.0, "brute-force at weakest unique h must be 0");
    assert!(factored.abs() < 1e-9, "factored at weakest unique h must be 0, got {}", factored);
    assert!(factored.is_finite(), "factored at B_i=0 boundary must not be NaN/Inf");
}

#[test]
fn k3_factored_cfv_tied_at_top_boundary() {
    // When h is tied with one other hand at the top (T_i > 0 for that one
    // tied opponent, S_i = 0), the share has substantial contributions from
    // T-branches at various tied counts. Exercises the (T,T), (B,T), (T,B)
    // PAIR formulas with non-zero values simultaneously.
    let (hand_cards, hand_strength, reach, nh) = make_toy_setup();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
    let masses = precompute_opp_masses(&reach_views, &hand_cards, &hand_strength, 0u64);

    // h=2 strength 70, tied with h=3.
    let h = 2;
    let hc1 = hand_cards[h * 2] as usize;
    let hc2 = hand_cards[h * 2 + 1] as usize;
    let h_dead = (1u64 << hc1) | (1u64 << hc2);

    assert!(masses.t[1*nh+h] > 0.0, "T_1 should include h=3 (tied)");
    assert!(masses.b[1*nh+h] > 0.0, "B_1 should include weaker hands");

    let bf = brute_force_share(&reach_views, &hand_cards, &hand_strength, nh, h, h_dead);
    let factored = factored_k3_share(&masses, &reach_views, &hand_strength, h, h_dead);
    let diff = (factored - bf).abs();
    eprintln!("TIED-AT-TOP h={}: bf={:.9} factored={:.9} diff={:.3e}", h, bf, factored, diff);
    assert!(diff < 1e-5, "tied-at-top boundary fails: bf={}, factored={}, diff={}",
        bf, factored, diff);
}

#[allow(dead_code)]
fn unused() { let _ = card_pair_to_index; let _ = index_to_card_pair; }
