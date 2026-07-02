//! First-order cluster correction to the multiway fold-terminal reach-product
//! mass. The EXACT mass is `Σ over mutually-disjoint (g_1..g_K) of Π r_i[g_i]`
//! (O(nh^K)); `factored_total_reach_product` multiplies per-opponent totals
//! independently, IGNORING inter-opponent card removal (two opponents can't
//! hold the same card) — an over-count that grows with K.
//!
//! The connected-cluster expansion in the pairwise collision probability is
//!   M_K(h) = Π S_i − Σ_{i<j} C_ij Π_{k≠i,j} S_k + O(triple collisions)
//! where S_i is opponent i's ⊥h total reach and C_ij is the ⊥h COLLIDING-pair
//! reach `Σ_c Pm_i^h[c]·Pm_j^h[c] − Σ_{g⊥h} r_i r_j` (D-form; the exact 2-body
//! term, validated in tests/cluster_expansion.rs). Keeping only κ2 corrects the
//! DOMINANT collision (measured: 6.5× closer to the full expansion than the
//! factored product at deck-49) at O(nh) prep + O(52·K²) per hand — cheap
//! enough for the real-time CPU re-solve. Triple/quad terms (the remaining ~1%)
//! need per-terminal card-form prep and are deferred to the GPU port.

/// Pairwise-corrected mass for K opponents (K≥2). O(nh·K + nh·K²) prep, then
/// O(52·K² + 2^0) per hand. Falls back to the exact factored product for K<2.
pub fn mass_cluster_pairs(opp_reach: &[&[f32]], hand_cards: &[u8], nh: usize) -> Vec<f32> {
    let k = opp_reach.len();
    if k < 2 {
        // K=1: exact total-reach (card removal vs h) — the factored product is
        // already exact here.
        return crate::solver::showdown::factored_total_reach_product(opp_reach, hand_cards, nh);
    }
    let npair = k * (k - 1) / 2;
    let mut pm = vec![0.0f32; k * 52];
    let mut s = vec![0.0f32; k];
    // per-hand r_i needed for the ⊥h corrections; keep the reach slices.
    let mut cc_full = vec![0.0f32; npair];
    let mut cc_row = vec![0.0f32; npair * 52];
    // O(1) pair→hand-index table (built once; the per-h correction needs
    // r_i(c, h) lookups K²·52 times per hand — a scan there is O(nh²) death).
    let mut p2h = vec![u32::MAX; 52 * 52];
    for g in 0..nh {
        let (a, b) = (hand_cards[g * 2] as usize, hand_cards[g * 2 + 1] as usize);
        p2h[a * 52 + b] = g as u32; p2h[b * 52 + a] = g as u32;
    }
    for g in 0..nh {
        let (a, b) = (hand_cards[g * 2] as usize, hand_cards[g * 2 + 1] as usize);
        for i in 0..k {
            let r = opp_reach[i][g];
            if r != 0.0 {
                pm[i * 52 + a] += r; pm[i * 52 + b] += r; s[i] += r;
            }
        }
        let mut p = 0;
        for i in 0..k { for j in (i + 1)..k {
            let v = opp_reach[i][g] * opp_reach[j][g];
            if v != 0.0 { cc_full[p] += v; cc_row[p * 52 + a] += v; cc_row[p * 52 + b] += v; }
            p += 1;
        }}
    }

    let mut out = vec![0.0f32; nh];
    for h in 0..nh {
        let (h1, h2) = (hand_cards[h * 2] as usize, hand_cards[h * 2 + 1] as usize);
        // ⊥h per-role total: S_i^h = s_i − pm_i[h1] − pm_i[h2] + r_i(h)
        let mut sh = vec![0.0f32; k];
        for i in 0..k {
            sh[i] = s[i] - pm[i * 52 + h1] - pm[i * 52 + h2] + opp_reach[i][h];
        }
        let prod_all: f32 = sh.iter().product();
        // Σ_{i<j} C_ij^h · Π_{k≠i,j} S_k^h
        let mut corr = 0.0f32;
        let mut p = 0;
        for i in 0..k { for j in (i + 1)..k {
            // C_ij^h = Σ_{c≠h} Pm_i^h[c]·Pm_j^h[c] − Σ_{g⊥h} r_i r_j
            let mut dot = 0.0f32;
            for c in 0..52 {
                if c == h1 || c == h2 { continue; }
                let ri_c_h1 = { let g = p2h[c * 52 + h1]; if g == u32::MAX { 0.0 } else { opp_reach[i][g as usize] } };
                let ri_c_h2 = { let g = p2h[c * 52 + h2]; if g == u32::MAX { 0.0 } else { opp_reach[i][g as usize] } };
                let rj_c_h1 = { let g = p2h[c * 52 + h1]; if g == u32::MAX { 0.0 } else { opp_reach[j][g as usize] } };
                let rj_c_h2 = { let g = p2h[c * 52 + h2]; if g == u32::MAX { 0.0 } else { opp_reach[j][g as usize] } };
                let pmi = pm[i * 52 + c] - ri_c_h1 - ri_c_h2;
                let pmj = pm[j * 52 + c] - rj_c_h1 - rj_c_h2;
                dot += pmi * pmj;
            }
            let hp = opp_reach[i][h] * opp_reach[j][h];
            let cc_perp = cc_full[p] - cc_row[p * 52 + h1] - cc_row[p * 52 + h2] + hp;
            let cij = dot - cc_perp;
            // Π_{other roles} S^h
            let mut rest = 1.0f32;
            for m in 0..k { if m != i && m != j { rest *= sh[m]; } }
            corr += cij * rest;
            p += 1;
        }}
        out[h] = (prod_all - corr).max(0.0);
    }
    out
}

/// O(1)-per-hand form of `mass_cluster_pairs` (same result, bit-close). The
/// per-hand `Σ_c Pm_i^h[c]·Pm_j^h[c]` dot product is expanded into per-terminal
/// tables so the hand loop is a handful of lookups instead of O(52·K²):
///   Σ_c Pm_i^h·Pm_j^h = DOT_ij − A_ij[h1] − A_ij[h2] − A_ji[h1] − A_ji[h2]
///                       + B_ij[h1,h1] + B_ij[h1,h2] + B_ij[h2,h1] + B_ij[h2,h2]
/// with DOT_ij=Σ_c pm_i·pm_j, A_ij[d]=Σ_c pm_i[c]·r_j(c,d),
///      B_ij[d,e]=Σ_c r_i(c,d)·r_j(c,e); then the c∈{h1,h2} diagonal terms are
/// removed to restrict to c∉h. Prep ≈ factored's; per-hand ≈ O(K²).
pub fn mass_cluster_pairs_fast(opp_reach: &[&[f32]], hand_cards: &[u8], nh: usize) -> Vec<f32> {
    let k = opp_reach.len();
    if k < 2 {
        return crate::solver::showdown::factored_total_reach_product(opp_reach, hand_cards, nh);
    }
    let npair = k * (k - 1) / 2;
    // pair2hand + per-card reach + collision aggregates (as in the base fn).
    let mut p2h = vec![u32::MAX; 52 * 52];
    let mut pm = vec![0.0f32; k * 52];
    let mut s = vec![0.0f32; k];
    let mut cc_full = vec![0.0f32; npair];
    let mut cc_row = vec![0.0f32; npair * 52];
    // dense per-role pair-reach matrices [K][52*52] (for the B_ij matmul).
    let mut pair = vec![0.0f32; k * 52 * 52];
    for g in 0..nh {
        let (a, b) = (hand_cards[g * 2] as usize, hand_cards[g * 2 + 1] as usize);
        p2h[a * 52 + b] = g as u32; p2h[b * 52 + a] = g as u32;
        for i in 0..k {
            let r = opp_reach[i][g];
            if r != 0.0 {
                pm[i * 52 + a] += r; pm[i * 52 + b] += r; s[i] += r;
                pair[i * 2704 + a * 52 + b] = r; pair[i * 2704 + b * 52 + a] = r;
            }
        }
        let mut p = 0;
        for i in 0..k { for j in (i + 1)..k {
            let v = opp_reach[i][g] * opp_reach[j][g];
            if v != 0.0 { cc_full[p] += v; cc_row[p * 52 + a] += v; cc_row[p * 52 + b] += v; }
            p += 1;
        }}
    }
    // per-pair tables DOT / A_ij / A_ji / B_ij
    let mut dot = vec![0.0f32; npair];
    let mut a_ij = vec![0.0f32; npair * 52];
    let mut a_ji = vec![0.0f32; npair * 52];
    let mut b_ij = vec![0.0f32; npair * 52 * 52];
    let mut p = 0;
    for i in 0..k { for j in (i + 1)..k {
        for c in 0..52 { dot[p] += pm[i * 52 + c] * pm[j * 52 + c]; }
        // A_ij[d]=Σ_c pm_i[c]·r_j(c,d);  A_ji[d]=Σ_c pm_j[c]·r_i(c,d)
        for g in 0..nh {
            let (c, d) = (hand_cards[g * 2] as usize, hand_cards[g * 2 + 1] as usize);
            let (rj, ri) = (opp_reach[j][g], opp_reach[i][g]);
            if rj != 0.0 { a_ij[p * 52 + d] += pm[i * 52 + c] * rj; a_ij[p * 52 + c] += pm[i * 52 + d] * rj; }
            if ri != 0.0 { a_ji[p * 52 + d] += pm[j * 52 + c] * ri; a_ji[p * 52 + c] += pm[j * 52 + d] * ri; }
        }
        // B_ij[d][e]=Σ_c pair_i(c,d)·pair_j(c,e) = (pair_i^T · pair_j)[d][e].
        // Dense 52^3 matmul on the pre-built pair matrices.
        for c in 0..52 {
            let bi = i * 2704 + c * 52;
            let bj = j * 2704 + c * 52;
            for d in 0..52 {
                let rid = pair[bi + d];
                if rid == 0.0 { continue; }
                let row = p * 2704 + d * 52;
                for e in 0..52 { b_ij[row + e] += rid * pair[bj + e]; }
            }
        }
        p += 1;
    }}

    let mut out = vec![0.0f32; nh];
    for h in 0..nh {
        let (h1, h2) = (hand_cards[h * 2] as usize, hand_cards[h * 2 + 1] as usize);
        let rh = |i: usize| { let g = p2h[h1 * 52 + h2]; if g == u32::MAX { 0.0 } else { opp_reach[i][g as usize] } };
        let mut sh = vec![0.0f32; k];
        for i in 0..k { sh[i] = s[i] - pm[i * 52 + h1] - pm[i * 52 + h2] + opp_reach[i][h]; }
        let prod_all: f32 = sh.iter().product();
        let mut corr = 0.0f32;
        let mut p = 0;
        for i in 0..k { for j in (i + 1)..k {
            let rih = rh(i); let rjh = rh(j);
            // Σ_all_c Pm_i^h·Pm_j^h
            let mut dotp = dot[p]
                - a_ij[p * 52 + h1] - a_ij[p * 52 + h2]
                - a_ji[p * 52 + h1] - a_ji[p * 52 + h2]
                + b_ij[p * 2704 + h1 * 52 + h1] + b_ij[p * 2704 + h1 * 52 + h2]
                + b_ij[p * 2704 + h2 * 52 + h1] + b_ij[p * 2704 + h2 * 52 + h2];
            // remove c=h1,h2:  Pm_i^h[h1]=pm_i[h1]−r_i(h1h2), etc.
            let pmi_h1 = pm[i * 52 + h1] - rih; let pmj_h1 = pm[j * 52 + h1] - rjh;
            let pmi_h2 = pm[i * 52 + h2] - rih; let pmj_h2 = pm[j * 52 + h2] - rjh;
            dotp -= pmi_h1 * pmj_h1 + pmi_h2 * pmj_h2;
            let cc_perp = cc_full[p] - cc_row[p * 52 + h1] - cc_row[p * 52 + h2] + rih * rjh;
            let cij = dotp - cc_perp;
            let mut rest = 1.0f32;
            for m in 0..k { if m != i && m != j { rest *= sh[m]; } }
            corr += cij * rest;
            p += 1;
        }}
        out[h] = (prod_all - corr).max(0.0);
    }
    out
}
