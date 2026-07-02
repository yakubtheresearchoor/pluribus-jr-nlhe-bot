//! K=3 (np=4) lone-survivor compatible-mass ALGEBRA validation — the math gate
//! before the Metal port. A np=4 lone-survivor fold terminal's cfv is
//!   payoff × mass3(h) / num_combinations,
//! mass3(h) = Σ over MUTUALLY DISJOINT (g0,g1,g2), all avoiding h, of
//!            r0[g0]·r1[g1]·r2[g2]      (full joint card removal — EXACT).
//!
//! Brute is O(nh³) per hand. The factored form loops g0 with an O(1) inner
//! M2(E), E = h∪g0 (4 cards), built from per-terminal tables:
//!   S_i, P_i[c], W[g]=P2[g.a]+P2[g.b], TA1=Σ r1·W, A1c[c]=Σ_{g∋c} r1·W,
//!   TB=Σ r1·r2, Bc[c]=Σ_{g∋c} r1·r2, V[d]=Σ_g r1[g]·U_d[g],
//!   Vc[c][d]=Σ_{g∋c} r1[g]·U_d[g], with U_d[g]=r2({g.a,d})+r2({g.b,d}).
//! Inclusion–exclusion identities (no approximation):
//!   S(E)      = S − Σ_{c∈E} P[c] + Σ_{c<d∈E} r(cd)
//!   P2^E[c]   = P2[c] − Σ_{d∈E} r2({c,d})
//!   Σ_{g⊥E} f = T − Σ_{c∈E} Fc[c] + Σ_{c<d∈E} f(cd-hand)
//!   M2(E)     = S1(E)·S2(E) − [A1(E) − A2(E)] + B(E)
//! where A1(E)=Σ_{g1⊥E} r1·W, A2(E)=Σ_{g1⊥E} r1·Σ_{d∈E}U_d, B(E)=Σ_{g1⊥E} r1·r2.
//!
//! Uses a reduced deck (12 cards → 66 hands) so brute O(nh³) is instant, with
//! randomized reaches over many trials. Gate: |factored − brute| tiny relative.

/// hands = all 2-card combos of a `deck_n`-card deck.
fn make_hands(deck_n: u8) -> Vec<(u8, u8)> {
    let mut v = Vec::new();
    for a in 0..deck_n {
        for b in (a + 1)..deck_n {
            v.push((a, b));
        }
    }
    v
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / (1u64 << 31) as f64
    }
}

fn disjoint(x: (u8, u8), y: (u8, u8)) -> bool {
    x.0 != y.0 && x.0 != y.1 && x.1 != y.0 && x.1 != y.1
}
fn avoids(g: (u8, u8), e: &[u8]) -> bool {
    !e.contains(&g.0) && !e.contains(&g.1)
}

/// Brute: full triple enumeration with mutual disjointness (the parity target —
/// identical in structure to multiway_brute_force_showdown's mass at np=4).
fn brute_mass3(hands: &[(u8, u8)], h: usize, r0: &[f64], r1: &[f64], r2: &[f64]) -> f64 {
    let hh = hands[h];
    let mut m = 0.0;
    for (g0, &rr0) in hands.iter().zip(r0) {
        if rr0 == 0.0 || !disjoint(*g0, hh) { continue; }
        for (g1, &rr1) in hands.iter().zip(r1) {
            if rr1 == 0.0 || !disjoint(*g1, hh) || !disjoint(*g1, *g0) { continue; }
            for (g2, &rr2) in hands.iter().zip(r2) {
                if rr2 == 0.0 || !disjoint(*g2, hh) || !disjoint(*g2, *g0) || !disjoint(*g2, *g1) { continue; }
                m += rr0 * rr1 * rr2;
            }
        }
    }
    m
}

/// Factored: O(nh) outer g0 loop with O(1) inner M2(E) from precomputed tables.
/// This is EXACTLY the per-(terminal,hand) thread algorithm the Metal kernel runs.
fn factored_mass3(
    hands: &[(u8, u8)], deck_n: usize, h: usize,
    r0: &[f64], r1: &[f64], r2: &[f64],
) -> f64 {
    let nh = hands.len();
    // pair2hand
    let mut p2h = vec![usize::MAX; deck_n * deck_n];
    for (i, &(a, b)) in hands.iter().enumerate() {
        p2h[a as usize * deck_n + b as usize] = i;
        p2h[b as usize * deck_n + a as usize] = i;
    }
    let rp = |r: &[f64], c: u8, d: u8| -> f64 {
        if c == d { return 0.0; }
        let i = p2h[c as usize * deck_n + d as usize];
        if i == usize::MAX { 0.0 } else { r[i] }
    };

    // ── per-terminal tables (O(nh·deck) build) ──
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    let mut pc1 = vec![0.0f64; deck_n];
    let mut pc2 = vec![0.0f64; deck_n];
    for (g, &(a, b)) in hands.iter().enumerate() {
        s1 += r1[g]; pc1[a as usize] += r1[g]; pc1[b as usize] += r1[g];
        s2 += r2[g]; pc2[a as usize] += r2[g]; pc2[b as usize] += r2[g];
    }
    // W[g] = P2[g.a] + P2[g.b]
    let w: Vec<f64> = hands.iter().map(|&(a, b)| pc2[a as usize] + pc2[b as usize]).collect();
    // TA1 / A1c   and   TB / Bc
    let mut ta1 = 0.0f64;
    let mut a1c = vec![0.0f64; deck_n];
    let mut tb = 0.0f64;
    let mut bc = vec![0.0f64; deck_n];
    for (g, &(a, b)) in hands.iter().enumerate() {
        let x = r1[g] * w[g];
        ta1 += x; a1c[a as usize] += x; a1c[b as usize] += x;
        let y = r1[g] * r2[g];
        tb += y; bc[a as usize] += y; bc[b as usize] += y;
    }
    // V[d] = Σ_g r1[g]·U_d[g],  Vc[c][d] = Σ_{g∋c} r1[g]·U_d[g]
    let mut v = vec![0.0f64; deck_n];
    let mut vc = vec![0.0f64; deck_n * deck_n];
    for (g, &(a, b)) in hands.iter().enumerate() {
        if r1[g] == 0.0 { continue; }
        for d in 0..deck_n as u8 {
            let u = rp(r2, a, d) + rp(r2, b, d);
            if u == 0.0 { continue; }
            let x = r1[g] * u;
            v[d as usize] += x;
            vc[a as usize * deck_n + d as usize] += x;
            vc[b as usize * deck_n + d as usize] += x;
        }
    }

    let hh = hands[h];
    // restrict helper: Σ_{g⊥E} f  =  T − Σ_{c∈E} Fc[c] + Σ_{c<d∈E} f(cd)
    // (E always has 4 DISTINCT cards here: h ∪ g0, with g0 ⊥ h.)
    let e_pairs = |e: &[u8; 4]| -> [(u8, u8); 6] {
        [(e[0],e[1]),(e[0],e[2]),(e[0],e[3]),(e[1],e[2]),(e[1],e[3]),(e[2],e[3])]
    };

    let mut mass = 0.0f64;
    for (g0i, &(a, b)) in hands.iter().enumerate() {
        let rr0 = r0[g0i];
        if rr0 == 0.0 || !disjoint((a, b), hh) { continue; }
        let e = [hh.0, hh.1, a, b];

        // S1(E), S2(E)
        let mut s1e = s1; let mut s2e = s2;
        for &c in &e { s1e -= pc1[c as usize]; s2e -= pc2[c as usize]; }
        for &(c, d) in &e_pairs(&e) { s1e += rp(r1, c, d); s2e += rp(r2, c, d); }

        // A1(E) = Σ_{g1⊥E} r1·W   (f(cd) = r1(cd)·W[cd-hand])
        let mut a1e = ta1;
        for &c in &e { a1e -= a1c[c as usize]; }
        for &(c, d) in &e_pairs(&e) {
            let i = p2h[c as usize * deck_n + d as usize];
            if i != usize::MAX { a1e += r1[i] * w[i]; }
        }

        // A2(E) = Σ_{d∈E} Σ_{g1⊥E} r1·U_d   (f(cd) for card-pair (c,c') = r1(cc')·U_d(cc'))
        let mut a2e = 0.0f64;
        for &dcard in &e {
            let d = dcard as usize;
            let mut t = v[d];
            for &c in &e { t -= vc[c as usize * deck_n + d]; }
            for &(c, cp) in &e_pairs(&e) {
                let i = p2h[c as usize * deck_n + cp as usize];
                if i != usize::MAX {
                    t += r1[i] * (rp(r2, hands[i].0, dcard) + rp(r2, hands[i].1, dcard));
                }
            }
            a2e += t;
        }

        // B(E) = Σ_{g1⊥E} r1·r2
        let mut be = tb;
        for &c in &e { be -= bc[c as usize]; }
        for &(c, d) in &e_pairs(&e) {
            let i = p2h[c as usize * deck_n + d as usize];
            if i != usize::MAX { be += r1[i] * r2[i]; }
        }

        // M2(E): note P2^E already folds the “−Σ_{d∈E} r2({c,d})” correction into
        // A2; and Compat2's "+r2[g1]" term is B(E).
        let m2 = s1e * s2e - (a1e - a2e) + be;
        mass += rr0 * m2;
    }
    mass
}

#[test]
fn k3_inclusion_exclusion_matches_brute() {
    let deck_n = 12usize; // 66 hands — brute O(nh³) ≈ 287k triples/hand: instant
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let mut rng = Lcg(0xD15C0_C0DE);

    let mut worst_rel = 0.0f64;
    for trial in 0..40 {
        // Random reaches incl. zeros (folded-hand sparsity like real CFR reach).
        let mk = |rng: &mut Lcg| -> Vec<f64> {
            (0..nh).map(|_| if rng.f() < 0.25 { 0.0 } else { rng.f() }).collect()
        };
        let (r0, r1, r2) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
        for h in 0..nh {
            let b = brute_mass3(&hands, h, &r0, &r1, &r2);
            let f = factored_mass3(&hands, deck_n, h, &r0, &r1, &r2);
            let rel = (b - f).abs() / b.abs().max(1e-12);
            if rel > worst_rel { worst_rel = rel; }
            assert!(
                rel < 1e-9,
                "trial {trial} hand {h}: brute={b:.12} factored={f:.12} rel={rel:.3e}"
            );
        }
    }
    eprintln!("K=3 inclusion-exclusion == brute over 40 trials × {nh} hands; worst rel err {worst_rel:.3e}");
}

// ═══════════════════════════════════════════════════════════════════════════
// CLOSED FORM: O(1)-per-eval mass3(X) after a one-pass aggregate build.
// Eliminates the per-eval g0 loop entirely. Derivation (banked in memory):
//   inner D-FORM  M2(Y) = S_B(Y)·S_C(Y) + B(Y) − D(Y),
//                 D(Y) = Σ_{c∉Y} P_B^Y[c]·P_C^Y[c]
//   outer         mass3(X) = Σ_{g⊥X} rA[g]·M2(X∪g)
// expanded into X-free per-hand FEATURES whose rA-weighted aggregates (totals +
// per-card-restricted rows) are prebuilt; ⊥X restriction = total − rows[x∈X]
// + direct pair add-backs. X is GENERAL (|X|=2 here; |X|=4 = the K=4 inner).
// Validated by SUB-PART (S/B/D each vs brute) then total vs brute enumeration.
// ═══════════════════════════════════════════════════════════════════════════

mod cf {
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
}

// ── brute sub-part references (direct scans; the sub-validator targets) ──

fn brute_sy(hands: &[(u8, u8)], r: &[f64], y: &[u8]) -> f64 {
    hands.iter().zip(r).filter(|(&g, _)| avoids(g, y)).map(|(_, &v)| v).sum()
}
fn brute_by(hands: &[(u8, u8)], rb: &[f64], rc: &[f64], y: &[u8]) -> f64 {
    hands.iter().enumerate().filter(|(_, &g)| avoids(g, y)).map(|(i, _)| rb[i] * rc[i]).sum()
}
fn brute_dy(hands: &[(u8, u8)], deck_n: usize, rb: &[f64], rc: &[f64], y: &[u8]) -> f64 {
    let mut pby = vec![0.0; deck_n]; let mut pcy = vec![0.0; deck_n];
    for (i, &(a, b)) in hands.iter().enumerate() {
        if !avoids((a, b), y) { continue; }
        pby[a as usize] += rb[i]; pby[b as usize] += rb[i];
        pcy[a as usize] += rc[i]; pcy[b as usize] += rc[i];
    }
    (0..deck_n).filter(|c| !y.contains(&(*c as u8))).map(|c| pby[c] * pcy[c]).sum()
}
/// 2-opp disjoint mass avoiding Y (direct enumeration).
fn brute_m2(hands: &[(u8, u8)], rb: &[f64], rc: &[f64], y: &[u8]) -> f64 {
    let mut m = 0.0;
    for (i, &g1) in hands.iter().enumerate() {
        if rb[i] == 0.0 || !avoids(g1, y) { continue; }
        for (j, &g2) in hands.iter().enumerate() {
            if rc[j] == 0.0 || !avoids(g2, y) || !disjoint(g1, g2) { continue; }
            m += rb[i] * rc[j];
        }
    }
    m
}
/// mass3 with an EXPLICIT exclusion set X (generalizes brute_mass3's h).
fn brute_mass3_excl(hands: &[(u8, u8)], x: &[u8], r0: &[f64], r1: &[f64], r2: &[f64]) -> f64 {
    let mut m = 0.0;
    for (g0, &ga) in hands.iter().enumerate() {
        if r0[g0] == 0.0 || !avoids(ga, x) { continue; }
        let mut y: Vec<u8> = x.to_vec();
        y.push(ga.0); y.push(ga.1);
        m += r0[g0] * brute_m2(hands, r1, r2, &y);
    }
    m
}

#[test]
fn d_form_identity() {
    // M2(Y) = S_B·S_C + B − D  must equal the direct 2-opp disjoint enumeration.
    let deck_n = 12usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let mut rng = Lcg(0xDF04);
    for _ in 0..30 {
        let mk = |rng: &mut Lcg| -> Vec<f64> {
            (0..nh).map(|_| if rng.f() < 0.25 { 0.0 } else { rng.f() }).collect()
        };
        let (rb, rc) = (mk(&mut rng), mk(&mut rng));
        // random Y of size 4 (distinct cards)
        let mut y = vec![];
        while y.len() < 4 {
            let c = (rng.f() * deck_n as f64) as u8;
            if !y.contains(&c) { y.push(c); }
        }
        let s = brute_sy(&hands, &rb, &y) * brute_sy(&hands, &rc, &y);
        let b = brute_by(&hands, &rb, &rc, &y);
        let d = brute_dy(&hands, deck_n, &rb, &rc, &y);
        let m2 = brute_m2(&hands, &rb, &rc, &y);
        let rel = ((s + b - d) - m2).abs() / m2.abs().max(1e-12);
        assert!(rel < 1e-10, "D-form identity broken: {} vs {m2}", s + b - d);
    }
}

#[test]
fn closed_form_sub_parts_and_total() {
    let deck_n = 12usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let mut rng = Lcg(0xC105_ED);

    let mut worst = [0.0f64; 4]; // s, b, d, total
    for trial in 0..25 {
        let mk = |rng: &mut Lcg| -> Vec<f64> {
            (0..nh).map(|_| if rng.f() < 0.25 { 0.0 } else { rng.f() }).collect()
        };
        let (ra, rb, rc) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
        let roles = cf::roles(&hands, &ra, &rb, &rc);
        let agg = cf::build(&roles, &hands);

        // |X| = 2 (every hand as X) and |X| = 4 (random sets) in the same trial.
        let mut xs: Vec<Vec<u8>> = hands.iter().map(|&(a, b)| vec![a, b]).collect();
        for _ in 0..6 {
            let mut x = vec![];
            while x.len() < 4 {
                let c = (rng.f() * deck_n as f64) as u8;
                if !x.contains(&c) { x.push(c); }
            }
            xs.push(x);
        }

        for x in &xs {
            // brute sub-parts: scan g ⊥ X directly.
            let (mut bs, mut bb, mut bd) = (0.0, 0.0, 0.0);
            for (g, &gc) in hands.iter().enumerate() {
                if ra[g] == 0.0 || !avoids(gc, x) { continue; }
                let mut y = x.clone();
                y.push(gc.0); y.push(gc.1);
                bs += ra[g] * brute_sy(&hands, &rb, &y) * brute_sy(&hands, &rc, &y);
                bb += ra[g] * brute_by(&hands, &rb, &rc, &y);
                bd += ra[g] * brute_dy(&hands, deck_n, &rb, &rc, &y);
            }
            let cs = cf::s_part(&roles, &hands, &agg, x);
            let cb = cf::b_part(&roles, &hands, &agg, x);
            let cd = cf::d_part(&roles, &hands, &agg, x);
            let bt = brute_mass3_excl(&hands, x, &ra, &rb, &rc);
            let ct = cf::mass3_closed(&roles, &hands, &agg, x);
            let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
            worst[0] = worst[0].max(rel(cs, bs));
            worst[1] = worst[1].max(rel(cb, bb));
            worst[2] = worst[2].max(rel(cd, bd));
            worst[3] = worst[3].max(rel(ct, bt));
            assert!(rel(cs, bs) < 1e-9, "trial {trial} X={x:?}: S closed={cs} brute={bs}");
            assert!(rel(cb, bb) < 1e-9, "trial {trial} X={x:?}: B closed={cb} brute={bb}");
            assert!(rel(cd, bd) < 1e-9, "trial {trial} X={x:?}: D closed={cd} brute={bd}");
            assert!(rel(ct, bt) < 1e-9, "trial {trial} X={x:?}: TOTAL closed={ct} brute={bt}");
        }
    }
    eprintln!("closed-form worst rel: S={:.2e} B={:.2e} D={:.2e} TOTAL={:.2e} (|X|=2 all hands + |X|=4 random, 25 trials)",
        worst[0], worst[1], worst[2], worst[3]);
}

// ═══════════════════════════════════════════════════════════════════════════
// K=4 (np=5) lone-survivor mass — the live-5 GPU terminal math gate.
//   mass4(h) = Σ over MUTUALLY DISJOINT (g0,g1,g2,g3) ⊥ h of r0·r1·r2·r3
//            = Σ_{g0⊥h} r0[g0] · mass3_{r1,r2,r3}(h ∪ g0)
// The inner mass3 with |X|=4 is the ALREADY-VALIDATED closed form (mod cf,
// 2.1e-14). Also gates the MC-SAMPLED outer estimator (the budget lever):
//   mass4 ≈ (W/M)·Σ_{s=1..M, g_s~r0/W} [g_s⊥h]·mass3(h∪g_s),  W = Σ r0
// (unbiased; samples shared across h within a terminal — the kernel design).
// ═══════════════════════════════════════════════════════════════════════════

fn brute_mass4(hands: &[(u8, u8)], h: usize, rs: [&[f64]; 4]) -> f64 {
    let hh = hands[h];
    let mut m = 0.0;
    for (g0, &c0) in hands.iter().enumerate() {
        if rs[0][g0] == 0.0 || !disjoint(c0, hh) { continue; }
        for (g1, &c1) in hands.iter().enumerate() {
            if rs[1][g1] == 0.0 || !disjoint(c1, hh) || !disjoint(c1, c0) { continue; }
            for (g2, &c2) in hands.iter().enumerate() {
                if rs[2][g2] == 0.0 || !disjoint(c2, hh) || !disjoint(c2, c0) || !disjoint(c2, c1) { continue; }
                for (g3, &c3) in hands.iter().enumerate() {
                    if rs[3][g3] == 0.0 || !disjoint(c3, hh) || !disjoint(c3, c0)
                        || !disjoint(c3, c1) || !disjoint(c3, c2) { continue; }
                    m += rs[0][g0] * rs[1][g1] * rs[2][g2] * rs[3][g3];
                }
            }
        }
    }
    m
}

/// The kernel algorithm: outer g0 loop × closed-form mass3 with X = h∪g0.
fn mass4_closed_inner(
    hands: &[(u8, u8)], h: usize,
    r0: &[f64], roles: &cf::Roles, agg: &cf::Agg,
) -> f64 {
    let hh = hands[h];
    let mut m = 0.0;
    for (g0, &(a, b)) in hands.iter().enumerate() {
        if r0[g0] == 0.0 || !disjoint((a, b), hh) { continue; }
        m += r0[g0] * cf::mass3_closed(roles, hands, agg, &[hh.0, hh.1, a, b]);
    }
    m
}

/// MC outer: M draws ∝ r0 (shared across h), importance weight W/M.
fn mass4_mc(
    hands: &[(u8, u8)], h: usize,
    r0: &[f64], roles: &cf::Roles, agg: &cf::Agg,
    m_draws: usize, seed: u64,
) -> f64 {
    let hh = hands[h];
    let w: f64 = r0.iter().sum();
    if w <= 0.0 { return 0.0; }
    // CDF sample (the kernel will binary-search a per-terminal CDF).
    let cdf: Vec<f64> = r0.iter().scan(0.0, |s, &v| { *s += v; Some(*s) }).collect();
    let mut rng = Lcg(seed);
    let mut acc = 0.0;
    for _ in 0..m_draws {
        let u = rng.f() * w;
        let g0 = cdf.partition_point(|&c| c < u).min(hands.len() - 1);
        let (a, b) = hands[g0];
        if !disjoint((a, b), hh) { continue; }
        acc += cf::mass3_closed(roles, hands, agg, &[hh.0, hh.1, a, b]);
    }
    acc * w / m_draws as f64
}

#[test]
fn k4_mass4_closed_inner_matches_brute() {
    let deck_n = 12usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let mut rng = Lcg(0x4A55);
    let mut worst = 0.0f64;
    let mut mc_worst = 0.0f64;
    let mut mc_cv_max = 0.0f64;
    for trial in 0..6 {
        let mk = |rng: &mut Lcg| -> Vec<f64> {
            (0..nh).map(|_| if rng.f() < 0.25 { 0.0 } else { rng.f() }).collect()
        };
        let (r0, r1, r2, r3) = (mk(&mut rng), mk(&mut rng), mk(&mut rng), mk(&mut rng));
        let roles = cf::roles(&hands, &r1, &r2, &r3);
        let agg = cf::build(&roles, &hands);
        for h in (trial % 8..nh).step_by(9) {
            let bt = brute_mass4(&hands, h, [&r0, &r1, &r2, &r3]);
            let ct = mass4_closed_inner(&hands, h, &r0, &roles, &agg);
            let rel = (ct - bt).abs() / bt.abs().max(1e-12);
            worst = worst.max(rel);
            assert!(rel < 1e-9, "trial {trial} h={h}: closed {ct} vs brute {bt}");
            // MC estimator: 256-seed mean at M=128 must converge to exact
            // (noise floor of the mean ≈ single-draw σ/16 ≈ 1%; gate 1.5×).
            let draws: Vec<f64> = (0..256)
                .map(|s| mass4_mc(&hands, h, &r0, &roles, &agg, 128, 0x9E3779B9 + s))
                .collect();
            let mc_mean: f64 = draws.iter().sum::<f64>() / draws.len() as f64;
            let var: f64 = draws.iter().map(|d| (d - mc_mean).powi(2)).sum::<f64>() / draws.len() as f64;
            let cv = var.sqrt() / bt.abs().max(1e-12); // single-estimate σ / truth
            mc_cv_max = mc_cv_max.max(cv);
            let mc_rel = (mc_mean - bt).abs() / bt.abs().max(1e-12);
            mc_worst = mc_worst.max(mc_rel);
        }
    }
    eprintln!("mass4 closed-inner worst rel={worst:.2e}; MC(M=128): 256-seed-mean worst rel={mc_worst:.2e}, single-estimate CV max={mc_cv_max:.3}");
    assert!(mc_worst < 0.015, "MC outer biased? {mc_worst}");
}
