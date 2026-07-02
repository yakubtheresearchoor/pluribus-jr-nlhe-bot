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

/// Pairs + TRIPLES cluster mass (κ2 exact + κ3), O(1)-per-hand via per-terminal
/// tables. κ3's h-corrections use an ORDER-CONSISTENT truncation: correction
/// pieces of collision-order p³ (the same order as the dropped κ4) are omitted,
/// which keeps every table ≤52² (exact κ3 rows would need 52³ ≈ GBs). K must
/// be 4. Accuracy target ≈ the drop-κ4 ladder point (~1e-2 scale-rel vs exact,
/// ~69× better than factored) — gated in tests.
pub fn mass_cluster_k23_fast(opp_reach: &[&[f32]], hand_cards: &[u8], nh: usize) -> Vec<f32> {
    let k = opp_reach.len();
    assert_eq!(k, 4, "k23 mass is built for K=4 (np=5)");
    // ---------- shared prep (as in pairs_fast) ----------
    let mut p2h = vec![u32::MAX; 52 * 52];
    let mut pm = vec![0.0f32; k * 52];
    let mut s = vec![0.0f32; k];
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
    }
    let npair = 6usize;
    let pi = [0usize, 0, 0, 1, 1, 2];
    let pj = [1usize, 2, 3, 2, 3, 3];
    let mut cc_full = vec![0.0f32; npair];
    let mut cc_row = vec![0.0f32; npair * 52];
    for g in 0..nh {
        let (a, b) = (hand_cards[g * 2] as usize, hand_cards[g * 2 + 1] as usize);
        for q in 0..npair {
            let v = opp_reach[pi[q]][g] * opp_reach[pj[q]][g];
            if v != 0.0 { cc_full[q] += v; cc_row[q * 52 + a] += v; cc_row[q * 52 + b] += v; }
        }
    }
    let mut dot = vec![0.0f32; npair];
    let mut a_ij = vec![0.0f32; npair * 52];
    let mut a_ji = vec![0.0f32; npair * 52];
    let mut b_ij = vec![0.0f32; npair * 2704];
    for q in 0..npair {
        let (i, j) = (pi[q], pj[q]);
        for c in 0..52 { dot[q] += pm[i * 52 + c] * pm[j * 52 + c]; }
        for g in 0..nh {
            let (c, d) = (hand_cards[g * 2] as usize, hand_cards[g * 2 + 1] as usize);
            let (rj, ri) = (opp_reach[j][g], opp_reach[i][g]);
            if rj != 0.0 { a_ij[q * 52 + d] += pm[i * 52 + c] * rj; a_ij[q * 52 + c] += pm[i * 52 + d] * rj; }
            if ri != 0.0 { a_ji[q * 52 + d] += pm[j * 52 + c] * ri; a_ji[q * 52 + c] += pm[j * 52 + d] * ri; }
        }
        for c in 0..52 {
            let bi = i * 2704 + c * 52; let bj = j * 2704 + c * 52;
            for d in 0..52 {
                let rid = pair[bi + d];
                if rid == 0.0 { continue; }
                let row = q * 2704 + d * 52;
                for e in 0..52 { b_ij[row + e] += rid * pair[bj + e]; }
            }
        }
    }
    // ---------- κ3 prep: per ordered path (ctr; i, k) ----------
    // T0 = Σ_g r_c·X_i·X_k with X_i(g)=pm_i[a]+pm_i[b]−r_i[g]; + per-card rows.
    // T1 (h-correction, order p²): T1a_ik[e] = Σ_g r_c·X_i·y_k[e]  (and i↔k),
    //   y_k[e](g) = pair_k(a,e)+pair_k(b,e). Rows for T1 omitted (order p³).
    // W_tri per unordered triple: T_AAA via the 5 card-coincidence cases from
    //   pm/pair tables (O(52³) prep, h-corrected at leading order), T_AAB via
    //   per-hand scans (+rows), 5Σrrr scalar (+rows).
    let triples: [[usize; 3]; 4] = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]];
    // ordered paths within each triple: (ctr, i, k) with i<k
    let mut t0 = Vec::new();      // per path: scalar
    let mut t0_row = Vec::new();  // per path: [52]
    let mut t1a = Vec::new();     // per path: [52] (correction on the k side)
    let mut t1b = Vec::new();     // per path: [52] (correction on the i side)
    let mut paths: Vec<(usize, usize, usize)> = Vec::new();
    for t in &triples {
        for c in 0..3 {
            let ctr = t[c];
            let rest: Vec<usize> = (0..3).filter(|&x| x != c).map(|x| t[x]).collect();
            paths.push((ctr, rest[0], rest[1]));
        }
    }
    for &(ctr, i, kk) in &paths {
        let mut v0 = 0.0f32;
        let mut row = vec![0.0f32; 52];
        let mut c1 = vec![0.0f32; 52];
        let mut c2 = vec![0.0f32; 52];
        for g in 0..nh {
            let rc = opp_reach[ctr][g];
            if rc == 0.0 { continue; }
            let (a, b) = (hand_cards[g * 2] as usize, hand_cards[g * 2 + 1] as usize);
            let xi = pm[i * 52 + a] + pm[i * 52 + b] - opp_reach[i][g];
            let xk = pm[kk * 52 + a] + pm[kk * 52 + b] - opp_reach[kk][g];
            let v = rc * xi * xk;
            v0 += v; row[a] += v; row[b] += v;
            // T1: Σ r_c·X_i·y_k[e] — accumulate at e = every card that pairs
            // with g in role k's pair matrix. y_k[e](g) = pair_k(a,e)+pair_k(b,e):
            // scatter over the two rows of pair_k.
            for e in 0..52 {
                let yk = pair[kk * 2704 + a * 52 + e] + pair[kk * 2704 + b * 52 + e];
                if yk != 0.0 { c1[e] += rc * xi * yk; }
                let yi = pair[i * 2704 + a * 52 + e] + pair[i * 2704 + b * 52 + e];
                if yi != 0.0 { c2[e] += rc * xk * yi; }
            }
        }
        t0.push(v0); t0_row.push(row); t1a.push(c1); t1b.push(c2);
    }
    // W_tri tables per triple: T_AAA cases + T_AAB (+rows) + Σrrr (+rows)
    let mut tri_aaa = vec![0.0f32; 4];          // all-distinct + coincidence cases, full deck
    let mut tri_aaa_row = vec![[0.0f32; 52]; 4]; // leading-order per-card row (for ⊥h)
    let mut tri_aab = vec![0.0f32; 4];
    let mut tri_aab_row = vec![[0.0f32; 52]; 4];
    let mut tri_rrr = vec![0.0f32; 4];
    let mut tri_rrr_row = vec![[0.0f32; 52]; 4];
    for (ti, t) in triples.iter().enumerate() {
        let (i, j, kk) = (t[0], t[1], t[2]);
        // T_AAA all-distinct + cases — computed via card sums on pm/pair.
        // all-distinct: Σ_{c1≠c2≠c3} pr_i(c1,c3)pr_j(c1,c2)pr_k(c2,c3)
        let mut aaa = 0.0f32;
        for c1 in 0..52 {
            for c2 in 0..52 {
                if c2 == c1 { continue; }
                let prj = &pair[j * 2704 + c1 * 52..j * 2704 + c1 * 52 + 52];
                if prj[c2] == 0.0 && pm[j * 52 + c1] == 0.0 { }
                let pj12 = prj[c2];
                if pj12 == 0.0 { continue; }
                for c3 in 0..52 {
                    if c3 == c1 || c3 == c2 { continue; }
                    aaa += pair[i * 2704 + c1 * 52 + c3] * pj12 * pair[kk * 2704 + c2 * 52 + c3];
                }
            }
        }
        // coincidence cases (ii)-(v) from the validated closed_w_tri:
        let mut cases = 0.0f32;
        for a in 0..52 { for b in 0..52 {
            if a == b { continue; }
            cases += pair[i * 2704 + a * 52 + b] * pm[j * 52 + a] * pair[kk * 2704 + a * 52 + b];
            cases += pair[i * 2704 + b * 52 + a] * pair[j * 2704 + b * 52 + a] * pm[kk * 52 + a];
            cases += pm[i * 52 + a] * pair[j * 2704 + a * 52 + b] * pair[kk * 2704 + a * 52 + b];
        }}
        for c in 0..52 { cases += pm[i * 52 + c] * pm[j * 52 + c] * pm[kk * 52 + c]; }
        tri_aaa[ti] = aaa + cases;
        // leading-order rows for ⊥h: d/d(exclude card c) of the dominant
        // all-equal case only (others are higher order): pm-products row.
        for c in 0..52 {
            tri_aaa_row[ti][c] = pm[i * 52 + c] * pm[j * 52 + c] * pm[kk * 52 + c];
        }
        // T_AAB + rows and Σrrr + rows (per-hand scans)
        for g in 0..nh {
            let (a, b) = (hand_cards[g * 2] as usize, hand_cards[g * 2 + 1] as usize);
            let (ri, rj, rk) = (opp_reach[i][g], opp_reach[j][g], opp_reach[kk][g]);
            let x_j = pm[j * 52 + a] + pm[j * 52 + b] + 2.0 * rj;
            let x_k = pm[kk * 52 + a] + pm[kk * 52 + b] + 2.0 * rk;
            let x_i = pm[i * 52 + a] + pm[i * 52 + b] + 2.0 * ri;
            let v = ri * rk * x_j + ri * rj * x_k + rj * rk * x_i;
            tri_aab[ti] += v; tri_aab_row[ti][a] += v; tri_aab_row[ti][b] += v;
            let w = ri * rj * rk;
            tri_rrr[ti] += w; tri_rrr_row[ti][a] += w; tri_rrr_row[ti][b] += w;
        }
    }

    // ---------- per-hand eval ----------
    let mut out = vec![0.0f32; nh];
    for h in 0..nh {
        let (h1, h2) = (hand_cards[h * 2] as usize, hand_cards[h * 2 + 1] as usize);
        let rh = |i: usize| opp_reach[i][h];
        let mut sh = [0.0f32; 4];
        for i in 0..k { sh[i] = s[i] - pm[i * 52 + h1] - pm[i * 52 + h2] + rh(i); }
        // κ2 per pair (exact, as pairs_fast)
        let mut c2v = [0.0f32; 6];
        for q in 0..npair {
            let (i, j) = (pi[q], pj[q]);
            let (rih, rjh) = (rh(i), rh(j));
            let mut dotp = dot[q]
                - a_ij[q * 52 + h1] - a_ij[q * 52 + h2]
                - a_ji[q * 52 + h1] - a_ji[q * 52 + h2]
                + b_ij[q * 2704 + h1 * 52 + h1] + b_ij[q * 2704 + h1 * 52 + h2]
                + b_ij[q * 2704 + h2 * 52 + h1] + b_ij[q * 2704 + h2 * 52 + h2];
            let pmi_h1 = pm[i * 52 + h1] - rih; let pmj_h1 = pm[j * 52 + h1] - rjh;
            let pmi_h2 = pm[i * 52 + h2] - rih; let pmj_h2 = pm[j * 52 + h2] - rjh;
            dotp -= pmi_h1 * pmj_h1 + pmi_h2 * pmj_h2;
            let ccp = cc_full[q] - cc_row[q * 52 + h1] - cc_row[q * 52 + h2] + rih * rjh;
            c2v[q] = dotp - ccp;
        }
        let cpair = |i: usize, j: usize| -> f32 {
            for q in 0..npair { if pi[q] == i && pj[q] == j { return c2v[q]; } }
            0.0
        };
        // κ3 per triple: Σ paths [T0⊥h − T1 corrections] − W_tri⊥h(leading order)
        let mut k3v = [0.0f32; 4];
        for (ti, t) in triples.iter().enumerate() {
            let mut acc = 0.0f32;
            for c in 0..3 {
                let pidx = ti * 3 + c;
                let (_ctr, _i, _kk) = paths[pidx];
                let t0h = t0[pidx] - t0_row[pidx][h1] - t0_row[pidx][h2];
                let corr = t1a[pidx][h1] + t1a[pidx][h2] + t1b[pidx][h1] + t1b[pidx][h2];
                acc += t0h - corr;
            }
            let tri = (tri_aaa[ti] - tri_aaa_row[ti][h1] - tri_aaa_row[ti][h2])
                - (tri_aab[ti] - tri_aab_row[ti][h1] - tri_aab_row[ti][h2])
                + 5.0 * (tri_rrr[ti] - tri_rrr_row[ti][h1] - tri_rrr_row[ti][h2]);
            k3v[ti] = acc - tri;
            let _ = t;
        }
        // assemble the 14 partitions (κ4 dropped)
        let s0 = sh[0]; let s1 = sh[1]; let s2 = sh[2]; let s3 = sh[3];
        let mut m = s0 * s1 * s2 * s3;
        m += -cpair(0,1) * s2 * s3 - cpair(0,2) * s1 * s3 - cpair(0,3) * s1 * s2
           - cpair(1,2) * s0 * s3 - cpair(1,3) * s0 * s2 - cpair(2,3) * s0 * s1;
        m += cpair(0,1) * cpair(2,3) + cpair(0,2) * cpair(1,3) + cpair(0,3) * cpair(1,2);
        m += k3v[0] * s3 + k3v[1] * s2 + k3v[2] * s1 + k3v[3] * s0;
        out[h] = m.max(0.0);
    }
    out
}
