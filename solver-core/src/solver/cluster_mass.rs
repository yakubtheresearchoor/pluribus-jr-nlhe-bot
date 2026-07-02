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
