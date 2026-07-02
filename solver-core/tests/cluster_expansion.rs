//! CLUSTER-EXPANSION warm-up (task #3 phase 1): validate the connected-cluster
//! framework on mass3 before the κ4 (38-graph) build. Formula:
//!   M3(h) = S1S2S3 − C12·S3 − C13·S2 − C23·S1 + [W_p1 + W_p2 + W_p3] − W_tri
//! where all quantities are ⊥h:
//!   S_i        = solo mass
//!   C_ij       = colliding-pair mass       = Σ_c Pm_i[c]Pm_j[c] − Σ_g r_i r_j
//!   W_path(j)  = edges (i,j),(j,k) collide = Σ_{g_j} r_j·X_i(g_j)·X_k(g_j),
//!                X_i(g) = Pm_i[a]+Pm_i[b]−r_i[g]
//!   W_tri      = all three edges collide, via the exact (A−B)³ expansion
//!                (A = #shared cards per edge, B = same-hand; 1[collide]=A−B):
//!                W_tri = T_AAA − ΣT_AAB + 5·Σ_g r1 r2 r3
//! Every term is gated against its own BRUTE scan (isolate-sign-errors), then
//! the assembled M3 against brute triple enumeration. The K=2 case of this
//! framework is the D-form already validated at 1e-14 (M2 = S1S2 − C12).

fn make_hands(deck_n: u8) -> Vec<(u8, u8)> {
    let mut v = Vec::new();
    for a in 0..deck_n { for b in (a + 1)..deck_n { v.push((a, b)); } }
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
fn collide(x: (u8, u8), y: (u8, u8)) -> bool { !disjoint(x, y) }

/// Per-h primitives for one role: ⊥h per-card mass + total.
struct Prim { pm: Vec<f64>, s: f64 }
fn prim(hands: &[(u8, u8)], r: &[f64], hh: (u8, u8), deck_n: usize) -> Prim {
    let mut pm = vec![0.0; deck_n];
    let mut s = 0.0;
    for (g, &(a, b)) in hands.iter().enumerate() {
        if !disjoint((a, b), hh) { continue; }
        pm[a as usize] += r[g]; pm[b as usize] += r[g]; s += r[g];
    }
    Prim { pm, s }
}

fn closed_c(hands: &[(u8, u8)], ri: &[f64], rj: &[f64], pi: &Prim, pj: &Prim, hh: (u8, u8), deck_n: usize) -> f64 {
    let mut v = 0.0;
    for c in 0..deck_n { v += pi.pm[c] * pj.pm[c]; }
    for (g, &gc) in hands.iter().enumerate() {
        if disjoint(gc, hh) { v -= ri[g] * rj[g]; }
    }
    v
}

fn closed_w_path(hands: &[(u8, u8)], rj: &[f64], pi: &Prim, pk: &Prim, ri: &[f64], rk: &[f64], hh: (u8, u8)) -> f64 {
    let mut v = 0.0;
    for (g, &(a, b)) in hands.iter().enumerate() {
        if rj[g] == 0.0 || !disjoint((a, b), hh) { continue; }
        let xi = pi.pm[a as usize] + pi.pm[b as usize] - ri[g];
        let xk = pk.pm[a as usize] + pk.pm[b as usize] - rk[g];
        v += rj[g] * xi * xk;
    }
    v
}

/// W_tri closed: T_AAA (5 card-equality cases) − ΣT_AAB + 5·Σ r1r2r3.
fn closed_w_tri(
    hands: &[(u8, u8)], rs: [&[f64]; 3], ps: [&Prim; 3], hh: (u8, u8), deck_n: usize,
) -> f64 {
    // pair-reach lookup ⊥h
    let mut p2h = vec![usize::MAX; deck_n * deck_n];
    for (i, &(a, b)) in hands.iter().enumerate() {
        if !disjoint((a, b), hh) { continue; }
        p2h[a as usize * deck_n + b as usize] = i;
        p2h[b as usize * deck_n + a as usize] = i;
    }
    let pr = |r: &[f64], c: usize, d: usize| -> f64 {
        if c == d { return 0.0; }
        let i = p2h[c * deck_n + d];
        if i == usize::MAX { 0.0 } else { r[i] }
    };
    let (r1, r2, r3) = (rs[0], rs[1], rs[2]);
    let (p1, p2, p3) = (ps[0], ps[1], ps[2]);

    // T_AAA: edges (1,2)=c1, (2,3)=c2, (1,3)=c3; g1⊇{c1,c3}, g2⊇{c1,c2}, g3⊇{c2,c3}
    let mut t_aaa = 0.0;
    // (i) all distinct
    for c1 in 0..deck_n { for c2 in 0..deck_n { if c2 == c1 { continue; } for c3 in 0..deck_n {
        if c3 == c1 || c3 == c2 { continue; }
        t_aaa += pr(r1, c1, c3) * pr(r2, c1, c2) * pr(r3, c2, c3);
    }}}
    for a in 0..deck_n { for b in 0..deck_n { if a == b { continue; }
        // (ii) c1=c2=a, c3=b:  g2∋a any; g1={a,b}; g3={a,b}
        t_aaa += pr(r1, a, b) * p2.pm[a] * pr(r3, a, b);
        // (iii) c2=c3=a, c1=b:  g3∋a any; g1={b,a}; g2={b,a}
        t_aaa += pr(r1, b, a) * pr(r2, b, a) * p3.pm[a];
        // (iv) c1=c3=a, c2=b:  g1∋a any; g2={a,b}; g3={b,a}
        t_aaa += p1.pm[a] * pr(r2, a, b) * pr(r3, a, b);
    }}
    // (v) all equal
    for c in 0..deck_n { t_aaa += p1.pm[c] * p2.pm[c] * p3.pm[c]; }

    // T_AAB (one same-hand edge) + the scalar tails
    let mut t_aab = 0.0;
    let mut t_rrr = 0.0;
    for (g, &(a, b)) in hands.iter().enumerate() {
        if !disjoint((a, b), hh) { continue; }
        let (a, b) = (a as usize, b as usize);
        // B13 (g1=g3=g): × (Pm2[a]+Pm2[b]+2r2[g])
        t_aab += r1[g] * r3[g] * (p2.pm[a] + p2.pm[b] + 2.0 * r2[g]);
        // B12 (g1=g2=g): × (Pm3[a]+Pm3[b]+2r3[g])
        t_aab += r1[g] * r2[g] * (p3.pm[a] + p3.pm[b] + 2.0 * r3[g]);
        // B23 (g2=g3=g): × (Pm1[a]+Pm1[b]+2r1[g])
        t_aab += r2[g] * r3[g] * (p1.pm[a] + p1.pm[b] + 2.0 * r1[g]);
        t_rrr += r1[g] * r2[g] * r3[g];
    }
    t_aaa - t_aab + 5.0 * t_rrr
}

#[test]
fn cluster_expansion_mass3() {
    let deck_n = 12usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let mut rng = Lcg(0xC1_057E);
    let mut worst = [0.0f64; 4]; // C, W_path, W_tri, M3
    for trial in 0..12 {
        let mk = |rng: &mut Lcg| -> Vec<f64> {
            (0..nh).map(|_| if rng.f() < 0.25 { 0.0 } else { rng.f() }).collect()
        };
        let (r1, r2, r3) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
        for h in (trial % 5..nh).step_by(17) {
            let hh = hands[h];
            let ok = |g: (u8, u8)| disjoint(g, hh);
            let p1 = prim(&hands, &r1, hh, deck_n);
            let p2 = prim(&hands, &r2, hh, deck_n);
            let p3 = prim(&hands, &r3, hh, deck_n);
            let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);

            // C12 gate
            let mut bc = 0.0;
            for (gi, &ci) in hands.iter().enumerate() {
                if r1[gi] == 0.0 || !ok(ci) { continue; }
                for (gj, &cj) in hands.iter().enumerate() {
                    if ok(cj) && collide(ci, cj) { bc += r1[gi] * r2[gj]; }
                }
            }
            let cc = closed_c(&hands, &r1, &r2, &p1, &p2, hh, deck_n);
            worst[0] = worst[0].max(rel(cc, bc));
            assert!(rel(cc, bc) < 1e-10, "C: {cc} vs {bc}");

            // W_path gate (center 2: edges (1,2),(2,3))
            let mut bw = 0.0;
            for (gj, &cj) in hands.iter().enumerate() {
                if r2[gj] == 0.0 || !ok(cj) { continue; }
                let mut xi = 0.0; let mut xk = 0.0;
                for (gi, &ci) in hands.iter().enumerate() {
                    if ok(ci) && collide(ci, cj) { xi += r1[gi]; xk += r3[gi]; }
                }
                bw += r2[gj] * xi * xk;
            }
            let cw = closed_w_path(&hands, &r2, &p1, &p3, &r1, &r3, hh);
            worst[1] = worst[1].max(rel(cw, bw));
            assert!(rel(cw, bw) < 1e-10, "W_path: {cw} vs {bw}");

            // W_tri gate
            let mut bt = 0.0;
            for (g1, &c1) in hands.iter().enumerate() {
                if r1[g1] == 0.0 || !ok(c1) { continue; }
                for (g2, &c2) in hands.iter().enumerate() {
                    if r2[g2] == 0.0 || !ok(c2) || !collide(c1, c2) { continue; }
                    for (g3, &c3) in hands.iter().enumerate() {
                        if r3[g3] == 0.0 || !ok(c3) || !collide(c2, c3) || !collide(c1, c3) { continue; }
                        bt += r1[g1] * r2[g2] * r3[g3];
                    }
                }
            }
            let ct = closed_w_tri(&hands, [&r1, &r2, &r3], [&p1, &p2, &p3], hh, deck_n);
            worst[2] = worst[2].max(rel(ct, bt));
            assert!(rel(ct, bt) < 1e-10, "W_tri: h={h} {ct} vs {bt}");

            // Assembled M3 vs brute triple enumeration
            let mut bm = 0.0;
            for (g1, &c1) in hands.iter().enumerate() {
                if r1[g1] == 0.0 || !ok(c1) { continue; }
                for (g2, &c2) in hands.iter().enumerate() {
                    if r2[g2] == 0.0 || !ok(c2) || !disjoint(c1, c2) { continue; }
                    for (g3, &c3) in hands.iter().enumerate() {
                        if r3[g3] == 0.0 || !ok(c3) || !disjoint(c1, c3) || !disjoint(c2, c3) { continue; }
                        bm += r1[g1] * r2[g2] * r3[g3];
                    }
                }
            }
            let c12 = cc;
            let c13 = closed_c(&hands, &r1, &r3, &p1, &p3, hh, deck_n);
            let c23 = closed_c(&hands, &r2, &r3, &p2, &p3, hh, deck_n);
            let wp1 = closed_w_path(&hands, &r1, &p2, &p3, &r2, &r3, hh); // center 1
            let wp2 = cw;                                                  // center 2
            let wp3 = closed_w_path(&hands, &r3, &p1, &p2, &r1, &r2, hh); // center 3
            let m3 = p1.s * p2.s * p3.s - c12 * p3.s - c13 * p2.s - c23 * p1.s
                + (wp1 + wp2 + wp3) - ct;
            worst[3] = worst[3].max(rel(m3, bm));
            assert!(rel(m3, bm) < 1e-9, "M3: h={h} {m3} vs {bm}");
        }
    }
    eprintln!("cluster-expansion mass3 worst rel: C={:.2e} W_path={:.2e} W_tri={:.2e} M3={:.2e}",
        worst[0], worst[1], worst[2], worst[3]);
}
