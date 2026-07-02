//! Shared closed-form mass3/mass4 REFERENCE implementation (f64) — the
//! validated math (k4/closed_form gates in np4_lone_mass_algebra.rs). Used by
//! that harness AND the GPU np5 parity test via `#[path] mod`.
#![allow(dead_code)]

    pub const D: usize = 52; // card index space (deck may be smaller)

    pub struct Roles {
        pub ra: Vec<f64>,
        pub rb: Vec<f64>,
        pub rc: Vec<f64>,
        pub pb: Vec<f64>,   // per-card B mass
        pub pc: Vec<f64>,
        pub sb: f64,
        pub sc: f64,
        pub pairb: Vec<f64>, // pairb[c*D+d] = rB of hand {c,d}
        pub pairc: Vec<f64>,
        pub p2h: Vec<usize>,
        pub g0c: Vec<f64>,  // PB[c]·PC[c]
        pub g0: f64,
        pub g1t: Vec<f64>,  // G1[e]
        pub g2t: Vec<f64>,  // G2[e*D+e']
        pub tbc: f64,
        pub bcc: Vec<f64>,
    }

    pub fn roles(hands: &[(u8, u8)], ra: &[f64], rb: &[f64], rc: &[f64]) -> Roles {
        let mut pb = vec![0.0; D]; let mut pc = vec![0.0; D];
        let mut pairb = vec![0.0; D * D]; let mut pairc = vec![0.0; D * D];
        let mut p2h = vec![usize::MAX; D * D];
        let (mut sb, mut sc, mut tbc) = (0.0, 0.0, 0.0);
        let mut bcc = vec![0.0; D];
        for (g, &(a, b)) in hands.iter().enumerate() {
            let (a, b) = (a as usize, b as usize);
            pb[a] += rb[g]; pb[b] += rb[g]; sb += rb[g];
            pc[a] += rc[g]; pc[b] += rc[g]; sc += rc[g];
            pairb[a * D + b] = rb[g]; pairb[b * D + a] = rb[g];
            pairc[a * D + b] = rc[g]; pairc[b * D + a] = rc[g];
            p2h[a * D + b] = g; p2h[b * D + a] = g;
            let v = rb[g] * rc[g];
            tbc += v; bcc[a] += v; bcc[b] += v;
        }
        let mut g0c = vec![0.0; D]; let mut g0 = 0.0;
        for c in 0..D { g0c[c] = pb[c] * pc[c]; g0 += g0c[c]; }
        let mut g1t = vec![0.0; D];
        let mut g2t = vec![0.0; D * D];
        for e in 0..D {
            let mut s = 0.0;
            for c in 0..D { s += pb[c] * pairc[c * D + e] + pc[c] * pairb[c * D + e]; }
            g1t[e] = s;
            for ep in 0..D {
                let mut s2 = 0.0;
                for c in 0..D { s2 += pairb[c * D + e] * pairc[c * D + ep]; }
                g2t[e * D + ep] = s2;
            }
        }
        Roles { ra: ra.to_vec(), rb: rb.to_vec(), rc: rc.to_vec(), pb, pc, sb, sc,
                pairb, pairc, p2h, g0c, g0, g1t, g2t, tbc, bcc }
    }

    /// X-free per-hand features (g's cards a,b).
    pub struct Feat {
        pub qb: f64, pub qc: f64, pub ub: f64, pub uc: f64,
        pub dp: f64, pub g1f: f64, pub g2g: f64, pub vbc: f64,
        pub yb: Vec<f64>, pub yc: Vec<f64>, pub wbc: Vec<f64>,
        pub kap: Vec<f64>, pub row: Vec<f64>, pub col: Vec<f64>,
    }
    pub fn feat(r: &Roles, hands: &[(u8, u8)], g: usize) -> Feat {
        let (a, b) = (hands[g].0 as usize, hands[g].1 as usize);
        let mut yb = vec![0.0; D]; let mut yc = vec![0.0; D];
        let mut wbc = vec![0.0; D]; let mut kap = vec![0.0; D];
        let mut row = vec![0.0; D]; let mut col = vec![0.0; D];
        for e in 0..D {
            let pba = r.pairb[a * D + e]; let pbb = r.pairb[b * D + e];
            let pca = r.pairc[a * D + e]; let pcb = r.pairc[b * D + e];
            yb[e] = pba + pbb;
            yc[e] = pca + pcb;
            wbc[e] = pba * pca + pbb * pcb;
            kap[e] = r.pb[a] * pca + r.pc[a] * pba + r.pb[b] * pcb + r.pc[b] * pbb;
            row[e] = r.g2t[e * D + a] + r.g2t[e * D + b];
            col[e] = r.g2t[a * D + e] + r.g2t[b * D + e];
        }
        Feat {
            qb: r.rb[g], qc: r.rc[g],
            ub: r.pb[a] + r.pb[b], uc: r.pc[a] + r.pc[b],
            dp: r.g0c[a] + r.g0c[b],
            g1f: r.g1t[a] + r.g1t[b],
            g2g: r.g2t[a * D + a] + r.g2t[a * D + b] + r.g2t[b * D + a] + r.g2t[b * D + b],
            vbc: r.bcc[a] + r.bcc[b],
            yb, yc, wbc, kap, row, col,
        }
    }

    /// Aggregate = (total, per-card rows). Scalar kinds indexed by SK, vector
    /// kinds by VK (52 aggregates each), matrix kinds by MK (52² each).
    pub const SK_N: usize = 0; pub const SK_QB: usize = 1; pub const SK_QC: usize = 2;
    pub const SK_UB: usize = 3; pub const SK_UC: usize = 4; pub const SK_DP: usize = 5;
    pub const SK_G1: usize = 6; pub const SK_G2G: usize = 7; pub const SK_VBC: usize = 8;
    pub const SK_QBQC: usize = 9; pub const SK_QBUC: usize = 10; pub const SK_UBQC: usize = 11;
    pub const SK_UBUC: usize = 12; pub const N_SK: usize = 13;
    pub const VK_YB: usize = 0; pub const VK_YC: usize = 1; pub const VK_WBC: usize = 2;
    pub const VK_KAP: usize = 3; pub const VK_ROW: usize = 4; pub const VK_COL: usize = 5;
    pub const VK_QBYC: usize = 6; pub const VK_YBQC: usize = 7; pub const VK_UBYC: usize = 8;
    pub const VK_YBUC: usize = 9; pub const N_VK: usize = 10;
    pub const MK_TYY: usize = 0; pub const MK_ZSAME: usize = 1; pub const N_MK: usize = 2;

    pub fn sk_of(f: &Feat, k: usize) -> f64 {
        match k {
            SK_N => 1.0, SK_QB => f.qb, SK_QC => f.qc, SK_UB => f.ub, SK_UC => f.uc,
            SK_DP => f.dp, SK_G1 => f.g1f, SK_G2G => f.g2g, SK_VBC => f.vbc,
            SK_QBQC => f.qb * f.qc, SK_QBUC => f.qb * f.uc,
            SK_UBQC => f.ub * f.qc, SK_UBUC => f.ub * f.uc,
            _ => unreachable!(),
        }
    }
    pub fn vk_of(f: &Feat, k: usize, e: usize) -> f64 {
        match k {
            VK_YB => f.yb[e], VK_YC => f.yc[e], VK_WBC => f.wbc[e], VK_KAP => f.kap[e],
            VK_ROW => f.row[e], VK_COL => f.col[e],
            VK_QBYC => f.qb * f.yc[e], VK_YBQC => f.yb[e] * f.qc,
            VK_UBYC => f.ub * f.yc[e], VK_YBUC => f.yb[e] * f.uc,
            _ => unreachable!(),
        }
    }
    /// mk features need the per-card split (zsame), so take roles+hand directly.
    pub fn mk_of(r: &Roles, hands: &[(u8, u8)], g: usize, k: usize, e: usize, ep: usize) -> f64 {
        let (a, b) = (hands[g].0 as usize, hands[g].1 as usize);
        match k {
            MK_TYY => (r.pairb[a * D + e] + r.pairb[b * D + e]) * (r.pairc[a * D + ep] + r.pairc[b * D + ep]),
            MK_ZSAME => r.pairb[a * D + e] * r.pairc[a * D + ep] + r.pairb[b * D + e] * r.pairc[b * D + ep],
            _ => unreachable!(),
        }
    }

    pub struct Agg {
        pub sk: Vec<(f64, Vec<f64>)>,          // [N_SK] (total, rows[D])
        pub vk: Vec<(Vec<f64>, Vec<f64>)>,     // [N_VK] (tot[D], rows[D*D])
        pub mk: Vec<(Vec<f64>, Vec<f64>)>,     // [N_MK] (tot[D*D], rows[D*D*D])
    }
    pub fn build(r: &Roles, hands: &[(u8, u8)]) -> Agg {
        let mut sk = vec![(0.0, vec![0.0; D]); N_SK];
        let mut vk = vec![(vec![0.0; D], vec![0.0; D * D]); N_VK];
        let mut mk = vec![(vec![0.0; D * D], vec![0.0; D * D * D]); N_MK];
        for (g, &(a, b)) in hands.iter().enumerate() {
            let w = r.ra[g];
            if w == 0.0 { continue; }
            let (a, b) = (a as usize, b as usize);
            let f = feat(r, hands, g);
            for k in 0..N_SK {
                let v = w * sk_of(&f, k);
                sk[k].0 += v; sk[k].1[a] += v; sk[k].1[b] += v;
            }
            for k in 0..N_VK {
                for e in 0..D {
                    let v = w * vk_of(&f, k, e);
                    if v == 0.0 { continue; }
                    vk[k].0[e] += v; vk[k].1[a * D + e] += v; vk[k].1[b * D + e] += v;
                }
            }
            for k in 0..N_MK {
                for e in 0..D {
                    for ep in 0..D {
                        let v = w * mk_of(r, hands, g, k, e, ep);
                        if v == 0.0 { continue; }
                        mk[k].0[e * D + ep] += v;
                        mk[k].1[a * D * D + e * D + ep] += v;
                        mk[k].1[b * D * D + e * D + ep] += v;
                    }
                }
            }
        }
        Agg { sk, vk, mk }
    }

    /// Σ_{g⊥X} rA·feature: total − Σ_{x∈X} rows[x] + Σ_{pairs⊆X} rA[pair]·feature(pair).
    fn addback_pairs(r: &Roles, x: &[u8], mut f: impl FnMut(usize) -> f64) -> f64 {
        let mut s = 0.0;
        for i in 0..x.len() {
            for j in (i + 1)..x.len() {
                let idx = r.p2h[x[i] as usize * D + x[j] as usize];
                if idx != usize::MAX && r.ra[idx] != 0.0 {
                    s += r.ra[idx] * f(idx);
                }
            }
        }
        s
    }
    pub fn rsk(r: &Roles, hands: &[(u8, u8)], a: &Agg, k: usize, x: &[u8]) -> f64 {
        let mut v = a.sk[k].0;
        for &c in x { v -= a.sk[k].1[c as usize]; }
        v + addback_pairs(r, x, |g| sk_of(&feat(r, hands, g), k))
    }
    pub fn rvk(r: &Roles, hands: &[(u8, u8)], a: &Agg, k: usize, e: usize, x: &[u8]) -> f64 {
        let mut v = a.vk[k].0[e];
        for &c in x { v -= a.vk[k].1[c as usize * D + e]; }
        v + addback_pairs(r, x, |g| vk_of(&feat(r, hands, g), k, e))
    }
    pub fn rmk(r: &Roles, hands: &[(u8, u8)], a: &Agg, k: usize, e: usize, ep: usize, x: &[u8]) -> f64 {
        let mut v = a.mk[k].0[e * D + ep];
        for &c in x { v -= a.mk[k].1[c as usize * D * D + e * D + ep]; }
        v + addback_pairs(r, x, |g| mk_of(r, hands, g, k, e, ep))
    }

    /// The three closed sub-parts (each = Σ_{g⊥X} rA·part(X∪g)).
    pub fn s_part(r: &Roles, hands: &[(u8, u8)], a: &Agg, x: &[u8]) -> f64 {
        // αB = S_B(X) etc.
        let mut ab = r.sb; let mut ac = r.sc;
        for &c in x { ab -= r.pb[c as usize]; ac -= r.pc[c as usize]; }
        for i in 0..x.len() { for j in (i + 1)..x.len() {
            ab += r.pairb[x[i] as usize * D + x[j] as usize];
            ac += r.pairc[x[i] as usize * D + x[j] as usize];
        }}
        let n = rsk(r, hands, a, SK_N, x);
        // Σ⊥ rA·sB and ·sC
        let mut sb_sum = rsk(r, hands, a, SK_QB, x) - rsk(r, hands, a, SK_UB, x);
        let mut sc_sum = rsk(r, hands, a, SK_QC, x) - rsk(r, hands, a, SK_UC, x);
        for &e in x {
            sb_sum += rvk(r, hands, a, VK_YB, e as usize, x);
            sc_sum += rvk(r, hands, a, VK_YC, e as usize, x);
        }
        // Σ⊥ rA·sB·sC (9 groups)
        let mut cross = rsk(r, hands, a, SK_QBQC, x) - rsk(r, hands, a, SK_QBUC, x)
            - rsk(r, hands, a, SK_UBQC, x) + rsk(r, hands, a, SK_UBUC, x);
        for &e in x {
            let e = e as usize;
            cross += rvk(r, hands, a, VK_QBYC, e, x);
            cross -= rvk(r, hands, a, VK_UBYC, e, x);
            cross += rvk(r, hands, a, VK_YBQC, e, x);
            cross -= rvk(r, hands, a, VK_YBUC, e, x);
        }
        for &e in x { for &ep in x {
            cross += rmk(r, hands, a, MK_TYY, e as usize, ep as usize, x);
        }}
        ab * ac * n + ab * sc_sum + ac * sb_sum + cross
    }

    pub fn b_part(r: &Roles, hands: &[(u8, u8)], a: &Agg, x: &[u8]) -> f64 {
        let mut tbcx = r.tbc;
        for &c in x { tbcx -= r.bcc[c as usize]; }
        for i in 0..x.len() { for j in (i + 1)..x.len() {
            let idx = r.p2h[x[i] as usize * D + x[j] as usize];
            if idx != usize::MAX { tbcx += r.rb[idx] * r.rc[idx]; }
        }}
        let n = rsk(r, hands, a, SK_N, x);
        let mut v = tbcx * n + rsk(r, hands, a, SK_QBQC, x) - rsk(r, hands, a, SK_VBC, x);
        for &e in x { v += rvk(r, hands, a, VK_WBC, e as usize, x); }
        v
    }

    pub fn d_part(r: &Roles, hands: &[(u8, u8)], a: &Agg, x: &[u8]) -> f64 {
        // ΦBX(c) = Σ_{e∈X} pB(c,e)
        let phib = |c: usize| -> f64 { x.iter().map(|&e| r.pairb[c * D + e as usize]).sum() };
        let phic = |c: usize| -> f64 { x.iter().map(|&e| r.pairc[c * D + e as usize]).sum() };
        // D-const(X): the all-X-slots value of the D-formula.
        let mut dx = r.g0;
        for &c in x { dx -= r.g0c[c as usize]; }
        for &e in x { dx -= r.g1t[e as usize]; }
        for &e in x { for &c in x {
            dx += r.pb[c as usize] * r.pairc[c as usize * D + e as usize]
                + r.pc[c as usize] * r.pairb[c as usize * D + e as usize];
        }}
        for &e in x { for &ep in x { dx += r.g2t[e as usize * D + ep as usize]; } }
        for &c in x { dx -= phib(c as usize) * phic(c as usize); }

        let n = rsk(r, hands, a, SK_N, x);
        let mut v = dx * n;
        // term1,2 g-parts
        v -= rsk(r, hands, a, SK_DP, x);
        v -= rsk(r, hands, a, SK_G1, x);
        // term3 g-parts
        for &e in x { v += rvk(r, hands, a, VK_KAP, e as usize, x); }
        for &c in x {
            let c = c as usize;
            v += r.pb[c] * rvk(r, hands, a, VK_YC, c, x)
               + r.pc[c] * rvk(r, hands, a, VK_YB, c, x);
        }
        v += rsk(r, hands, a, SK_UBQC, x) + rsk(r, hands, a, SK_QBUC, x);
        // term4 g-parts
        for &e in x {
            v += rvk(r, hands, a, VK_ROW, e as usize, x);
            v += rvk(r, hands, a, VK_COL, e as usize, x);
        }
        v += rsk(r, hands, a, SK_G2G, x);
        // term5 (−ΦΦ): c∈X part
        for &c in x {
            let c = c as usize;
            v -= phib(c) * rvk(r, hands, a, VK_YC, c, x);
            v -= phic(c) * rvk(r, hands, a, VK_YB, c, x);
            v -= rmk(r, hands, a, MK_TYY, c, c, x);
        }
        // term5: c∈g part
        for &e in x { for &ep in x {
            v -= rmk(r, hands, a, MK_ZSAME, e as usize, ep as usize, x);
        }}
        for &e in x {
            let e = e as usize;
            v -= rvk(r, hands, a, VK_YBQC, e, x);
            v -= rvk(r, hands, a, VK_QBYC, e, x);
        }
        v -= 2.0 * rsk(r, hands, a, SK_QBQC, x);
        v
    }

    pub fn mass3_closed(r: &Roles, hands: &[(u8, u8)], a: &Agg, x: &[u8]) -> f64 {
        s_part(r, hands, a, x) + b_part(r, hands, a, x) - d_part(r, hands, a, x)
    }


pub fn disjoint2(x: (u8, u8), y: (u8, u8)) -> bool {
    x.0 != y.0 && x.0 != y.1 && x.1 != y.0 && x.1 != y.1
}

/// mass4(h) = Σ_{g0⊥h} r0[g0] · closed-mass3(h∪g0) — the np5 kernel algorithm.
pub fn mass4_closed_inner(
    hands: &[(u8, u8)], h: usize,
    r0: &[f64], roles: &Roles, agg: &Agg,
) -> f64 {
    let hh = hands[h];
    let mut m = 0.0;
    for (g0, &(a, b)) in hands.iter().enumerate() {
        if r0[g0] == 0.0 || !disjoint2((a, b), hh) { continue; }
        m += r0[g0] * mass3_closed(roles, hands, agg, &[hh.0, hh.1, a, b]);
    }
    m
}
