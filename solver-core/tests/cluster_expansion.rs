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

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2 — K=4: the GENERIC W_H evaluator. For a labeled collision graph H:
//   W_H = Σ_{B ⊆ E(H)} (−1)^{|B|} Σ_{partitions P of A-edges} val(B, P)
// B-edges contract hands (same-hand); each A-edge carries a card VARIABLE;
// blocks of P share one distinct card value. A hand-GROUP's factor by its
// incident distinct blocks: 0 → Scalar(group), 1 → PerCard(group, c),
// 2 → PairReach(group, {c,d}), ≥3 → ZERO (a 2-card hand can't hold 3 cards).
// Sums run over DISTINCT card tuples per partition. All ⊥h.
// Gated: (a) generic == manual κ3 forms on 3-vertex graphs; (b) per-graph
// brute scans at K=4; (c) assembled M4 vs brute quadruple enumeration.
// ═══════════════════════════════════════════════════════════════════════════

/// Per-h group primitives for every role-subset mask: Σ Π r_i over ⊥h hands.
struct GroupPrim {
    scalar: Vec<f64>,        // [mask]
    per_card: Vec<Vec<f64>>, // [mask][card]
    pair: Vec<Vec<f64>>,     // [mask][c*deck+d] product-reach at hand {c,d}
}
fn group_prim(hands: &[(u8, u8)], rs: &[&[f64]], hh: (u8, u8), deck_n: usize) -> GroupPrim {
    let k = rs.len();
    let nmask = 1usize << k;
    let mut scalar = vec![0.0; nmask];
    let mut per_card = vec![vec![0.0; deck_n]; nmask];
    let mut pair = vec![vec![0.0; deck_n * deck_n]; nmask];
    for (g, &(a, b)) in hands.iter().enumerate() {
        if !disjoint((a, b), hh) { continue; }
        for mask in 1..nmask {
            let mut w = 1.0;
            for i in 0..k { if mask & (1 << i) != 0 { w *= rs[i][g]; } }
            if w == 0.0 { continue; }
            scalar[mask] += w;
            per_card[mask][a as usize] += w;
            per_card[mask][b as usize] += w;
            pair[mask][a as usize * deck_n + b as usize] += w;
            pair[mask][b as usize * deck_n + a as usize] += w;
        }
    }
    GroupPrim { scalar, per_card, pair }
}

/// Set partitions of 0..n via restricted-growth strings.
fn set_partitions(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut rgs = vec![0usize; n];
    loop {
        out.push(rgs.clone());
        // next RGS
        let mut i = n as isize - 1;
        loop {
            if i <= 0 { return out; }
            let maxp = rgs[..i as usize].iter().cloned().max().unwrap_or(0);
            if rgs[i as usize] <= maxp { rgs[i as usize] += 1; break; }
            i -= 1;
        }
        for j in (i as usize + 1)..n { rgs[j] = 0; }
    }
}

/// Generic W_H for a labeled graph on `k` vertices (roles rs[0..k]).
/// edges: list of (u, v). All ⊥h via `gp` primitives.
fn w_h_generic(gp: &GroupPrim, k: usize, edges: &[(usize, usize)], deck_n: usize) -> f64 {
    let ne = edges.len();
    let mut total = 0.0;
    for bmask in 0..(1u32 << ne) {
        // contract B-edges: union-find over vertices
        let mut parent: Vec<usize> = (0..k).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            if p[x] != x { let r = find(p, p[x]); p[x] = r; }
            p[x]
        }
        let mut a_edges: Vec<(usize, usize)> = Vec::new();
        for (ei, &(u, v)) in edges.iter().enumerate() {
            if bmask & (1 << ei) != 0 {
                let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
                if ru != rv { parent[ru] = rv; }
            } else {
                a_edges.push((u, v));
            }
        }
        // group masks
        let mut group_of = vec![0usize; k];
        for v in 0..k { group_of[v] = find(&mut parent, v); }
        let roots: Vec<usize> = { let mut r: Vec<usize> = group_of.clone(); r.sort_unstable(); r.dedup(); r };
        let gmask = |root: usize| -> usize {
            (0..k).filter(|&v| group_of[v] == root).fold(0usize, |m, v| m | (1 << v))
        };
        let sign = if bmask.count_ones() % 2 == 0 { 1.0 } else { -1.0 };
        let na = a_edges.len();
        for part in set_partitions(na) {
            let nblocks = part.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
            // group -> set of incident distinct blocks
            let mut ok = true;
            let mut inc: Vec<Vec<usize>> = vec![Vec::new(); roots.len()];
            for (ei, &(u, v)) in a_edges.iter().enumerate() {
                let b = part[ei];
                for vv in [u, v] {
                    let gi = roots.iter().position(|&r| r == group_of[vv]).unwrap();
                    if !inc[gi].contains(&b) { inc[gi].push(b); }
                }
            }
            for i in &inc { if i.len() > 2 { ok = false; break; } }
            if !ok { continue; }
            // sum over DISTINCT card assignments to blocks (recursive)
            let mut cards = vec![usize::MAX; nblocks];
            fn rec(
                bi: usize, nblocks: usize, cards: &mut Vec<usize>, deck_n: usize,
                gp: &GroupPrim, roots: &[usize], inc: &[Vec<usize>],
                gmask: &dyn Fn(usize) -> usize,
            ) -> f64 {
                if bi == nblocks {
                    let mut v = 1.0;
                    for (gi, &root) in roots.iter().enumerate() {
                        let m = gmask(root);
                        v *= match inc[gi].len() {
                            0 => gp.scalar[m],
                            1 => gp.per_card[m][cards[inc[gi][0]]],
                            _ => gp.pair[m][cards[inc[gi][0]] * deck_n + cards[inc[gi][1]]],
                        };
                        if v == 0.0 { return 0.0; }
                    }
                    return v;
                }
                let mut s = 0.0;
                for c in 0..deck_n {
                    if cards[..bi].contains(&c) { continue; }
                    cards[bi] = c;
                    s += rec(bi + 1, nblocks, cards, deck_n, gp, roots, inc, gmask);
                }
                cards[bi] = usize::MAX;
                s
            }
            total += sign * rec(0, nblocks, &mut cards, deck_n, gp, &roots, &inc, &gmask);
        }
    }
    total
}

#[test]
fn generic_evaluator_matches_manual_k3() {
    let deck_n = 12usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let mut rng = Lcg(0x6E4E_71C);
    let mk = |rng: &mut Lcg| -> Vec<f64> {
        (0..nh).map(|_| if rng.f() < 0.25 { 0.0 } else { rng.f() }).collect()
    };
    let (r1, r2, r3) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
    let mut worst = 0.0f64;
    for h in (0..nh).step_by(23) {
        let hh = hands[h];
        let p1 = prim(&hands, &r1, hh, deck_n);
        let p2 = prim(&hands, &r2, hh, deck_n);
        let p3 = prim(&hands, &r3, hh, deck_n);
        let gp = group_prim(&hands, &[&r1, &r2, &r3], hh, deck_n);
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
        // path centered at 2 = edges (0,1),(1,2)
        let g_path = w_h_generic(&gp, 3, &[(0, 1), (1, 2)], deck_n);
        let m_path = closed_w_path(&hands, &r2, &p1, &p3, &r1, &r3, hh);
        worst = worst.max(rel(g_path, m_path));
        assert!(rel(g_path, m_path) < 1e-10, "generic path {g_path} vs manual {m_path}");
        // triangle
        let g_tri = w_h_generic(&gp, 3, &[(0, 1), (1, 2), (0, 2)], deck_n);
        let m_tri = closed_w_tri(&hands, [&r1, &r2, &r3], [&p1, &p2, &p3], hh, deck_n);
        worst = worst.max(rel(g_tri, m_tri));
        assert!(rel(g_tri, m_tri) < 1e-10, "generic tri {g_tri} vs manual {m_tri}");
    }
    eprintln!("generic-vs-manual κ3 worst rel = {worst:.2e}");
}

/// All connected labeled graphs on `k` vertices (as edge lists over pairs).
fn connected_graphs(k: usize) -> Vec<Vec<(usize, usize)>> {
    let mut pairs = Vec::new();
    for u in 0..k { for v in (u + 1)..k { pairs.push((u, v)); } }
    let ne = pairs.len();
    let mut out = Vec::new();
    for mask in 1u32..(1 << ne) {
        let edges: Vec<(usize, usize)> = (0..ne).filter(|&i| mask & (1 << i) != 0).map(|i| pairs[i]).collect();
        // spanning-connected over ALL k vertices
        let mut parent: Vec<usize> = (0..k).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            if p[x] != x { let r = find(p, p[x]); p[x] = r; } p[x]
        }
        for &(u, v) in &edges {
            let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
            if ru != rv { parent[ru] = rv; }
        }
        let r0 = find(&mut parent, 0);
        if (1..k).all(|v| find(&mut parent, v) == r0) { out.push(edges); }
    }
    out
}

#[test]
fn cluster_expansion_mass4() {
    let deck_n = 12usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let graphs4 = connected_graphs(4);
    assert_eq!(graphs4.len(), 38, "connected labeled graphs on 4 vertices");
    let graphs3 = connected_graphs(3);
    assert_eq!(graphs3.len(), 4);

    let mut rng = Lcg(0x4444_C1);
    let mut worst_wh = 0.0f64;
    let mut worst_m4 = 0.0f64;
    for trial in 0..2 {
        let mk = |rng: &mut Lcg| -> Vec<f64> {
            (0..nh).map(|_| if rng.f() < 0.3 { 0.0 } else { rng.f() }).collect()
        };
        let rs: Vec<Vec<f64>> = (0..4).map(|_| mk(&mut rng)).collect();
        let rr: Vec<&[f64]> = rs.iter().map(|r| r.as_slice()).collect();
        for h in (7 + trial * 13..nh).step_by(29) {
            let hh = hands[h];
            let ok = |g: (u8, u8)| disjoint(g, hh);
            let gp4 = group_prim(&hands, &rr, hh, deck_n);
            let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-9);

            // (b) per-graph gates: generic W_H vs brute constrained scan, all 38
            for edges in &graphs4 {
                let gw = w_h_generic(&gp4, 4, edges, deck_n);
                let mut bw = 0.0;
                for (g0, &c0) in hands.iter().enumerate() {
                    if rs[0][g0] == 0.0 || !ok(c0) { continue; }
                    for (g1, &c1) in hands.iter().enumerate() {
                        if rs[1][g1] == 0.0 || !ok(c1) { continue; }
                        if edges.contains(&(0, 1)) && !collide(c0, c1) { continue; }
                        for (g2, &c2) in hands.iter().enumerate() {
                            if rs[2][g2] == 0.0 || !ok(c2) { continue; }
                            if edges.contains(&(0, 2)) && !collide(c0, c2) { continue; }
                            if edges.contains(&(1, 2)) && !collide(c1, c2) { continue; }
                            for (g3, &c3) in hands.iter().enumerate() {
                                if rs[3][g3] == 0.0 || !ok(c3) { continue; }
                                if edges.contains(&(0, 3)) && !collide(c0, c3) { continue; }
                                if edges.contains(&(1, 3)) && !collide(c1, c3) { continue; }
                                if edges.contains(&(2, 3)) && !collide(c2, c3) { continue; }
                                bw += rs[0][g0] * rs[1][g1] * rs[2][g2] * rs[3][g3];
                            }
                        }
                    }
                }
                worst_wh = worst_wh.max(rel(gw, bw));
                assert!(rel(gw, bw) < 1e-9, "W_H {edges:?}: generic {gw} vs brute {bw}");
            }

            // (c) assembled M4 vs brute quadruple disjoint enumeration
            let kappa = |block: &[usize]| -> f64 {
                let brs: Vec<&[f64]> = block.iter().map(|&i| rs[i].as_slice()).collect();
                let gp = group_prim(&hands, &brs, hh, deck_n);
                match block.len() {
                    1 => gp.scalar[1],
                    2 => -w_h_generic(&gp, 2, &[(0, 1)], deck_n),
                    3 => graphs3.iter().map(|e| {
                        let s = if e.len() % 2 == 0 { 1.0 } else { -1.0 };
                        s * w_h_generic(&gp, 3, e, deck_n)
                    }).sum(),
                    _ => graphs4.iter().map(|e| {
                        let s = if e.len() % 2 == 0 { 1.0 } else { -1.0 };
                        s * w_h_generic(&gp, 4, e, deck_n)
                    }).sum(),
                }
            };
            // 15 partitions of {0,1,2,3}
            let parts: Vec<Vec<Vec<usize>>> = {
                let mut out = Vec::new();
                for rgs in set_partitions(4) {
                    let nb = rgs.iter().cloned().max().unwrap() + 1;
                    let mut blocks = vec![Vec::new(); nb];
                    for (i, &b) in rgs.iter().enumerate() { blocks[b].push(i); }
                    out.push(blocks);
                }
                out
            };
            assert_eq!(parts.len(), 15);
            let m4: f64 = parts.iter().map(|blocks| blocks.iter().map(|b| kappa(b)).product::<f64>()).sum();

            let mut bm = 0.0;
            for (g0, &c0) in hands.iter().enumerate() {
                if rs[0][g0] == 0.0 || !ok(c0) { continue; }
                for (g1, &c1) in hands.iter().enumerate() {
                    if rs[1][g1] == 0.0 || !ok(c1) || !disjoint(c0, c1) { continue; }
                    for (g2, &c2) in hands.iter().enumerate() {
                        if rs[2][g2] == 0.0 || !ok(c2) || !disjoint(c0, c2) || !disjoint(c1, c2) { continue; }
                        for (g3, &c3) in hands.iter().enumerate() {
                            if rs[3][g3] == 0.0 || !ok(c3) || !disjoint(c0, c3) || !disjoint(c1, c3) || !disjoint(c2, c3) { continue; }
                            bm += rs[0][g0] * rs[1][g1] * rs[2][g2] * rs[3][g3];
                        }
                    }
                }
            }
            worst_m4 = worst_m4.max(rel(m4, bm));
            assert!(rel(m4, bm) < 1e-8, "M4 h={h}: cluster {m4} vs brute {bm}");
        }
    }
    eprintln!("K=4 cluster expansion: 38 graphs each gated (worst W_H rel={worst_wh:.2e}); assembled M4 worst rel={worst_m4:.2e}");
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 3a — the H-CORRECTION LAYER (what the Metal kernel evaluates):
// primitives built ONCE on the FULL deck; per-h evaluation via O(1) corrections:
//   Scalar_m^h    = Scalar_m − PC_m[h1] − PC_m[h2] + PR_m(h1,h2)
//   PerCard_m^h[c]= PC_m[c] − PR_m(c,h1) − PR_m(c,h2)   (c ∉ {h1,h2})
//   Pair_m^h(c,d) = PR_m(c,d), 0 if it touches h
// and the card-tuple sums SKIP h1,h2. Gated against the per-h-rebuilt
// evaluator (phases 1-2, brute-proven) — this is the kernel's exact math.
// ═══════════════════════════════════════════════════════════════════════════

fn w_h_generic_fast(
    gp_full: &GroupPrim, hh: (u8, u8), k: usize, edges: &[(usize, usize)], deck_n: usize,
) -> f64 {
    let (h1, h2) = (hh.0 as usize, hh.1 as usize);
    let pairq = |m: usize, c: usize, d: usize| -> f64 {
        if c == h1 || c == h2 || d == h1 || d == h2 { 0.0 } else { gp_full.pair[m][c * deck_n + d] }
    };
    let ne = edges.len();
    let mut total = 0.0;
    for bmask in 0..(1u32 << ne) {
        let mut parent: Vec<usize> = (0..k).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            if p[x] != x { let r = find(p, p[x]); p[x] = r; } p[x]
        }
        let mut a_edges: Vec<(usize, usize)> = Vec::new();
        for (ei, &(u, v)) in edges.iter().enumerate() {
            if bmask & (1 << ei) != 0 {
                let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
                if ru != rv { parent[ru] = rv; }
            } else { a_edges.push((u, v)); }
        }
        let mut group_of = vec![0usize; k];
        for v in 0..k { group_of[v] = find(&mut parent, v); }
        let roots: Vec<usize> = { let mut r = group_of.clone(); r.sort_unstable(); r.dedup(); r };
        let gmask = |root: usize| -> usize {
            (0..k).filter(|&v| group_of[v] == root).fold(0usize, |m, v| m | (1 << v))
        };
        let sign = if bmask.count_ones() % 2 == 0 { 1.0 } else { -1.0 };
        for part in set_partitions(a_edges.len()) {
            let nblocks = part.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
            let mut inc: Vec<Vec<usize>> = vec![Vec::new(); roots.len()];
            let mut ok = true;
            for (ei, &(u, v)) in a_edges.iter().enumerate() {
                let b = part[ei];
                for vv in [u, v] {
                    let gi = roots.iter().position(|&r| r == group_of[vv]).unwrap();
                    if !inc[gi].contains(&b) { inc[gi].push(b); }
                }
            }
            for i in &inc { if i.len() > 2 { ok = false; break; } }
            if !ok { continue; }
            let mut cards = vec![usize::MAX; nblocks];
            #[allow(clippy::too_many_arguments)]
            fn rec(
                bi: usize, nblocks: usize, cards: &mut Vec<usize>, deck_n: usize,
                gp: &GroupPrim, roots: &[usize], inc: &[Vec<usize>],
                gmask: &dyn Fn(usize) -> usize, h1: usize, h2: usize,
                pairq: &dyn Fn(usize, usize, usize) -> f64,
            ) -> f64 {
                if bi == nblocks {
                    let mut v = 1.0;
                    for (gi, &root) in roots.iter().enumerate() {
                        let m = gmask(root);
                        v *= match inc[gi].len() {
                            0 => gp.scalar[m] - gp.per_card[m][h1] - gp.per_card[m][h2]
                                 + gp.pair[m][h1 * deck_n + h2],
                            1 => {
                                let c = cards[inc[gi][0]];
                                gp.per_card[m][c] - gp.pair[m][c * deck_n + h1] - gp.pair[m][c * deck_n + h2]
                            }
                            _ => pairq(m, cards[inc[gi][0]], cards[inc[gi][1]]),
                        };
                        if v == 0.0 { return 0.0; }
                    }
                    return v;
                }
                let mut s = 0.0;
                for c in 0..deck_n {
                    if c == h1 || c == h2 || cards[..bi].contains(&c) { continue; }
                    cards[bi] = c;
                    s += rec(bi + 1, nblocks, cards, deck_n, gp, roots, inc, gmask, h1, h2, pairq);
                }
                cards[bi] = usize::MAX;
                s
            }
            total += sign * rec(0, nblocks, &mut cards, deck_n, gp_full, &roots, &inc, &gmask, h1, h2, &pairq);
        }
    }
    total
}

#[test]
fn h_correction_layer_matches_perh() {
    let deck_n = 12usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let graphs4 = connected_graphs(4);
    let graphs3 = connected_graphs(3);
    let mut rng = Lcg(0xFA57_C0);
    let mk = |rng: &mut Lcg| -> Vec<f64> {
        (0..nh).map(|_| if rng.f() < 0.3 { 0.0 } else { rng.f() }).collect()
    };
    let rs: Vec<Vec<f64>> = (0..4).map(|_| mk(&mut rng)).collect();
    let rr: Vec<&[f64]> = rs.iter().map(|r| r.as_slice()).collect();
    // FULL-deck primitives built ONCE (hh = sentinel that blocks nothing).
    let gp_full = group_prim(&hands, &rr, (255, 254), deck_n);

    let mut worst = 0.0f64;
    for h in (0..nh).step_by(11) {
        let hh = hands[h];
        let gp_perh = group_prim(&hands, &rr, hh, deck_n);
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
        for edges in graphs4.iter().chain(graphs3.iter()) {
            let kk = edges.iter().flat_map(|&(u, v)| [u, v]).max().unwrap() + 1;
            let slow = w_h_generic(&gp_perh, kk, edges, deck_n);
            let fast = w_h_generic_fast(&gp_full, hh, kk, edges, deck_n);
            worst = worst.max(rel(fast, slow));
            assert!(rel(fast, slow) < 1e-10, "h={h} {edges:?}: fast {fast} vs perh {slow}");
        }
    }
    eprintln!("h-correction layer vs per-h rebuild: worst rel = {worst:.2e} (all 42 graphs × sampled h)");
}

/// PHASE 3b prelude — TERM CENSUS at production deck (52): enumerate every
/// surviving (partition-block, graph, B-mask, A-partition) term of the full M4
/// assembly, bucket by card-sum dimension, and estimate direct per-hand eval
/// cost. Decides the kernel split: dim ≤ D_direct evaluated per-thread,
/// dim > D_direct precomputed per terminal with h-expansion.
#[test]
fn m4_term_census() {
    let deck = 52usize;
    let graphs: [Vec<Vec<(usize, usize)>>; 3] = [connected_graphs(2), connected_graphs(3), connected_graphs(4)];
    // count over ALL blocks of the 15 role-partitions: each block of size k>=2
    // contributes its connected-graph expansion once per occurrence.
    let mut block_counts = [0usize; 5]; // how many blocks of each size appear across the 15 partitions
    for rgs in set_partitions(4) {
        let nb = rgs.iter().cloned().max().unwrap() + 1;
        let mut sizes = vec![0usize; nb];
        for &b in &rgs { sizes[b] += 1; }
        for &s in &sizes { block_counts[s] += 1; }
    }
    eprintln!("blocks by size across 15 partitions: 1s={} 2s={} 3s={} 4s={}",
        block_counts[1], block_counts[2], block_counts[3], block_counts[4]);

    let mut terms_by_dim = [0usize; 7];
    let mut ops_by_dim = [0f64; 7];
    for (ki, k) in [(0usize, 2usize), (1, 3), (2, 4)] {
        let occurrences = block_counts[k];
        for edges in &graphs[ki] {
            let ne = edges.len();
            for bmask in 0..(1u32 << ne) {
                let mut parent: Vec<usize> = (0..k).collect();
                fn find(p: &mut Vec<usize>, x: usize) -> usize {
                    if p[x] != x { let r = find(p, p[x]); p[x] = r; } p[x]
                }
                let mut a_edges: Vec<(usize, usize)> = Vec::new();
                for (ei, &(u, v)) in edges.iter().enumerate() {
                    if bmask & (1 << ei) != 0 {
                        let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
                        if ru != rv { parent[ru] = rv; }
                    } else { a_edges.push((u, v)); }
                }
                let mut group_of = vec![0usize; k];
                for v in 0..k { group_of[v] = find(&mut parent, v); }
                let roots: Vec<usize> = { let mut r = group_of.clone(); r.sort_unstable(); r.dedup(); r };
                for part in set_partitions(a_edges.len()) {
                    let nblocks = part.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
                    let mut inc: Vec<Vec<usize>> = vec![Vec::new(); roots.len()];
                    let mut ok = true;
                    for (ei, &(u, v)) in a_edges.iter().enumerate() {
                        for vv in [u, v] {
                            let gi = roots.iter().position(|&r| r == group_of[vv]).unwrap();
                            if !inc[gi].contains(&part[ei]) { inc[gi].push(part[ei]); }
                        }
                    }
                    for i in &inc { if i.len() > 2 { ok = false; break; } }
                    if !ok { continue; }
                    let d = nblocks.min(6);
                    terms_by_dim[d] += occurrences;
                    // direct-eval cost: distinct-tuple sum × ~(4 lookups + 4 mults) per group
                    ops_by_dim[d] += occurrences as f64 * (deck as f64).powi(d as i32) * (roots.len() as f64 * 6.0);
                }
            }
        }
    }
    let total_terms: usize = terms_by_dim.iter().sum();
    eprintln!("M4 term census (× partition occurrences): {terms_by_dim:?}  total={total_terms}");
    for d in 0..7 {
        if terms_by_dim[d] > 0 {
            eprintln!("  dim {d}: {} terms, direct-eval ≈ {:.2e} ops/hand", terms_by_dim[d], ops_by_dim[d]);
        }
    }
    let direct_le2: f64 = ops_by_dim[..3].iter().sum();
    let over: f64 = ops_by_dim[3..].iter().sum();
    eprintln!("dim≤2 direct total ≈ {direct_le2:.2e} ops/hand (budget ~3e5); dim≥3 (must precompute) ≈ {over:.2e}");
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 3b — EDGE-VARIABLE (contraction) BASIS. The partition device was only
// an evaluation trick; the ORIGINAL object is already unrestricted:
//   W_H(B) = Σ over card vars c_e (one per A-edge, full h-excluded deck,
//            NO ≠ constraints) of Π_groups F_group(assigned incident cards)
// with the group factor determined by the DISTINCT SET S of incident values:
//   |S|=0 → Scalar^h, 1 → PerCard^h, 2 → Pair^h, ≥3 → 0.
// Because F(c,c)=PerCard[c] and F(c,d)=Pair(c,d) form ONE fixed diagonal-
// extended table, this sum is a PURE TENSOR CONTRACTION — the prep-kernel
// form. Gated vs the distinct-partition evaluator (brute-proven chain).
// ═══════════════════════════════════════════════════════════════════════════

fn w_h_edgevar(
    gp_full: &GroupPrim, hh: (u8, u8), k: usize, edges: &[(usize, usize)], deck_n: usize,
) -> f64 {
    let (h1, h2) = (hh.0 as usize, hh.1 as usize);
    let ne = edges.len();
    let mut total = 0.0;
    for bmask in 0..(1u32 << ne) {
        let mut parent: Vec<usize> = (0..k).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            if p[x] != x { let r = find(p, p[x]); p[x] = r; } p[x]
        }
        let mut a_edges: Vec<(usize, usize)> = Vec::new();
        for (ei, &(u, v)) in edges.iter().enumerate() {
            if bmask & (1 << ei) != 0 {
                let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
                if ru != rv { parent[ru] = rv; }
            } else { a_edges.push((u, v)); }
        }
        let mut group_of = vec![0usize; k];
        for v in 0..k { group_of[v] = find(&mut parent, v); }
        let roots: Vec<usize> = { let mut r = group_of.clone(); r.sort_unstable(); r.dedup(); r };
        let sign = if bmask.count_ones() % 2 == 0 { 1.0 } else { -1.0 };
        // per-group incident A-edge indices
        let mut inc: Vec<Vec<usize>> = vec![Vec::new(); roots.len()];
        for (ei, &(u, v)) in a_edges.iter().enumerate() {
            for vv in [u, v] {
                let gi = roots.iter().position(|&r| r == group_of[vv]).unwrap();
                if !inc[gi].contains(&ei) { inc[gi].push(ei); }
            }
        }
        let masks: Vec<usize> = roots.iter().map(|&root| {
            (0..k).filter(|&v| group_of[v] == root).fold(0usize, |m, v| m | (1 << v))
        }).collect();
        // recursion over A-edge card variables (unrestricted, h-excluded)
        let mut cards = vec![usize::MAX; a_edges.len()];
        #[allow(clippy::too_many_arguments)]
        fn rec(
            ei: usize, ne: usize, cards: &mut Vec<usize>, deck_n: usize,
            gp: &GroupPrim, masks: &[usize], inc: &[Vec<usize>], h1: usize, h2: usize,
        ) -> f64 {
            if ei == ne {
                let mut v = 1.0;
                for (gi, m) in masks.iter().enumerate() {
                    // distinct incident value set
                    let mut s: Vec<usize> = inc[gi].iter().map(|&e| cards[e]).collect();
                    s.sort_unstable(); s.dedup();
                    v *= match s.len() {
                        0 => gp.scalar[*m] - gp.per_card[*m][h1] - gp.per_card[*m][h2]
                             + gp.pair[*m][h1 * deck_n + h2],
                        1 => gp.per_card[*m][s[0]] - gp.pair[*m][s[0] * deck_n + h1]
                             - gp.pair[*m][s[0] * deck_n + h2],
                        2 => gp.pair[*m][s[0] * deck_n + s[1]],
                        _ => 0.0,
                    };
                    if v == 0.0 { return 0.0; }
                }
                v
            } else {
                let mut acc = 0.0;
                for c in 0..deck_n {
                    if c == h1 || c == h2 { continue; }
                    cards[ei] = c;
                    acc += rec(ei + 1, ne, cards, deck_n, gp, masks, inc, h1, h2);
                }
                cards[ei] = usize::MAX;
                acc
            }
        }
        total += sign * rec(0, a_edges.len(), &mut cards, deck_n, gp_full, &masks, &inc, h1, h2);
    }
    total
}

#[test]
fn edgevar_basis_matches_distinct() {
    let deck_n = 10usize; // 45 hands — keeps the 6-var unrestricted sums fast
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let graphs4 = connected_graphs(4);
    let graphs3 = connected_graphs(3);
    let mut rng = Lcg(0xED6E);
    let mk = |rng: &mut Lcg| -> Vec<f64> {
        (0..nh).map(|_| if rng.f() < 0.3 { 0.0 } else { rng.f() }).collect()
    };
    let rs: Vec<Vec<f64>> = (0..4).map(|_| mk(&mut rng)).collect();
    let rr: Vec<&[f64]> = rs.iter().map(|r| r.as_slice()).collect();
    let gp_full = group_prim(&hands, &rr, (255, 254), deck_n);
    let mut worst = 0.0f64;
    for h in (0..nh).step_by(16) {
        let hh = hands[h];
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
        for edges in graphs4.iter().chain(graphs3.iter()) {
            let kk = edges.iter().flat_map(|&(u, v)| [u, v]).max().unwrap() + 1;
            let a = w_h_edgevar(&gp_full, hh, kk, edges, deck_n);
            let b = w_h_generic_fast(&gp_full, hh, kk, edges, deck_n);
            worst = worst.max(rel(a, b));
            assert!(rel(a, b) < 1e-10, "h={h} {edges:?}: edgevar {a} vs distinct {b}");
        }
    }
    eprintln!("edge-variable (pure-contraction) basis vs distinct evaluator: worst rel = {worst:.2e}");
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 4a — TERM PLANS: the tree-independent, kernel-consumable form. Each
// (connected graph, B-mask) becomes one TermPlan {sign, per-group role masks,
// per-group incident A-edge variable ids}. Generated ONCE on the CPU (this is
// the const buffer the Metal kernels interpret). The plan-driven evaluator is
// gated == w_h_edgevar (itself gated to brute through the full chain).
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
struct TermPlan {
    sign: f64,
    nvars: usize,
    /// per group: (role-subset mask over the BLOCK's roles, incident var ids)
    groups: Vec<(usize, Vec<usize>)>,
}

fn gen_term_plans(k: usize, edges: &[(usize, usize)]) -> Vec<TermPlan> {
    let ne = edges.len();
    let mut plans = Vec::new();
    for bmask in 0..(1u32 << ne) {
        let mut parent: Vec<usize> = (0..k).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            if p[x] != x { let r = find(p, p[x]); p[x] = r; } p[x]
        }
        let mut a_edges: Vec<(usize, usize)> = Vec::new();
        for (ei, &(u, v)) in edges.iter().enumerate() {
            if bmask & (1 << ei) != 0 {
                let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
                if ru != rv { parent[ru] = rv; }
            } else { a_edges.push((u, v)); }
        }
        let mut group_of = vec![0usize; k];
        for v in 0..k { group_of[v] = find(&mut parent, v); }
        let roots: Vec<usize> = { let mut r = group_of.clone(); r.sort_unstable(); r.dedup(); r };
        let groups: Vec<(usize, Vec<usize>)> = roots.iter().map(|&root| {
            let mask = (0..k).filter(|&v| group_of[v] == root).fold(0usize, |m, v| m | (1 << v));
            let inc: Vec<usize> = a_edges.iter().enumerate()
                .filter(|(_, &(u, v))| group_of[u] == root || group_of[v] == root)
                .map(|(ei, _)| ei).collect();
            (mask, inc)
        }).collect();
        plans.push(TermPlan {
            sign: if bmask.count_ones() % 2 == 0 { 1.0 } else { -1.0 },
            nvars: a_edges.len(),
            groups,
        });
    }
    plans
}

/// Plan-driven W_H eval — EXACTLY the arithmetic the Metal main/prep kernels
/// run: per plan, sum over per-var cards (h-excluded, unrestricted) of the
/// product of group factors read via O(1) h-corrected accessors.
fn w_h_from_plans(
    gp_full: &GroupPrim, hh: (u8, u8), plans: &[TermPlan], deck_n: usize,
) -> f64 {
    let (h1, h2) = (hh.0 as usize, hh.1 as usize);
    let mut total = 0.0;
    for plan in plans {
        let mut cards = vec![usize::MAX; plan.nvars];
        fn rec(
            vi: usize, plan: &TermPlan, cards: &mut Vec<usize>, deck_n: usize,
            gp: &GroupPrim, h1: usize, h2: usize,
        ) -> f64 {
            if vi == plan.nvars {
                let mut v = 1.0;
                for (mask, inc) in &plan.groups {
                    let mut s: Vec<usize> = inc.iter().map(|&e| cards[e]).collect();
                    s.sort_unstable(); s.dedup();
                    v *= match s.len() {
                        0 => gp.scalar[*mask] - gp.per_card[*mask][h1] - gp.per_card[*mask][h2]
                             + gp.pair[*mask][h1 * deck_n + h2],
                        1 => gp.per_card[*mask][s[0]] - gp.pair[*mask][s[0] * deck_n + h1]
                             - gp.pair[*mask][s[0] * deck_n + h2],
                        2 => gp.pair[*mask][s[0] * deck_n + s[1]],
                        _ => 0.0,
                    };
                    if v == 0.0 { return 0.0; }
                }
                v
            } else {
                let mut acc = 0.0;
                for c in 0..deck_n {
                    if c == h1 || c == h2 { continue; }
                    cards[vi] = c;
                    acc += rec(vi + 1, plan, cards, deck_n, gp, h1, h2);
                }
                cards[vi] = usize::MAX;
                acc
            }
        }
        total += plan.sign * rec(0, plan, &mut cards, deck_n, gp_full, h1, h2);
    }
    total
}

#[test]
fn term_plans_match_edgevar() {
    let deck_n = 10usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let graphs4 = connected_graphs(4);
    let graphs3 = connected_graphs(3);
    let mut rng = Lcg(0x9147);
    let mk = |rng: &mut Lcg| -> Vec<f64> {
        (0..nh).map(|_| if rng.f() < 0.3 { 0.0 } else { rng.f() }).collect()
    };
    let rs: Vec<Vec<f64>> = (0..4).map(|_| mk(&mut rng)).collect();
    let rr: Vec<&[f64]> = rs.iter().map(|r| r.as_slice()).collect();
    let gp_full = group_prim(&hands, &rr, (255, 254), deck_n);
    let mut worst = 0.0f64;
    let mut nplans = 0usize;
    for h in (0..nh).step_by(13) {
        let hh = hands[h];
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
        for edges in graphs4.iter().chain(graphs3.iter()) {
            let kk = edges.iter().flat_map(|&(u, v)| [u, v]).max().unwrap() + 1;
            let plans = gen_term_plans(kk, edges);
            nplans += plans.len();
            let a = w_h_from_plans(&gp_full, hh, &plans, deck_n);
            let b = w_h_edgevar(&gp_full, hh, kk, edges, deck_n);
            worst = worst.max(rel(a, b));
            assert!(rel(a, b) < 1e-12, "h={h} {edges:?}");
        }
    }
    eprintln!("term-plan evaluator == edgevar: worst rel = {worst:.2e}; plans/pass = {}", nplans / ((nh + 12) / 13));
}

/// TRUNCATION MEASUREMENT: drop κ4 (the 38-graph quad cluster) — error vs the
/// full expansion. The cluster series is in collision probability (~8%/pair at
/// production deck, ~3x larger on the 12-card test deck ⇒ this OVER-estimates).
#[test]
fn mass4_truncation_error() {
    let deck_n = 12usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let graphs4 = connected_graphs(4);
    let graphs3 = connected_graphs(3);
    let mut rng = Lcg(0x7241C);
    let mk = |rng: &mut Lcg| -> Vec<f64> {
        (0..nh).map(|_| if rng.f() < 0.3 { 0.0 } else { rng.f() }).collect()
    };
    let mut worst_noq = 0.0f64;   // drop κ4 entirely
    let mut worst_tree = 0.0f64;  // κ4 ≈ trees only (|E|=3)
    for _trial in 0..3 {
        let rs: Vec<Vec<f64>> = (0..4).map(|_| mk(&mut rng)).collect();
        for h in (5..nh).step_by(23) {
            let hh = hands[h];
            let kappa = |block: &[usize], mode: u8| -> f64 {
                let brs: Vec<&[f64]> = block.iter().map(|&i| rs[i].as_slice()).collect();
                let gp = group_prim(&hands, &brs, hh, deck_n);
                match block.len() {
                    1 => gp.scalar[1],
                    2 => -w_h_generic(&gp, 2, &[(0, 1)], deck_n),
                    3 => graphs3.iter().map(|e| {
                        let s = if e.len() % 2 == 0 { 1.0 } else { -1.0 };
                        s * w_h_generic(&gp, 3, e, deck_n)
                    }).sum(),
                    _ => match mode {
                        0 => 0.0, // dropped
                        1 => graphs4.iter().filter(|e| e.len() == 3).map(|e| -w_h_generic(&gp, 4, e, deck_n)).sum(),
                        _ => graphs4.iter().map(|e| {
                            let s = if e.len() % 2 == 0 { 1.0 } else { -1.0 };
                            s * w_h_generic(&gp, 4, e, deck_n)
                        }).sum(),
                    },
                }
            };
            let m4 = |mode: u8| -> f64 {
                set_partitions(4).iter().map(|rgs| {
                    let nb = rgs.iter().cloned().max().unwrap() + 1;
                    let mut blocks = vec![Vec::new(); nb];
                    for (i, &b) in rgs.iter().enumerate() { blocks[b].push(i); }
                    blocks.iter().map(|b| kappa(b, mode)).product::<f64>()
                }).sum()
            };
            let full = m4(2);
            if full.abs() < 1e-9 { continue; }
            worst_noq = worst_noq.max(((m4(0) - full) / full).abs());
            worst_tree = worst_tree.max(((m4(1) - full) / full).abs());
        }
    }
    eprintln!("mass4 truncation @deck-12 (over-estimates production): drop-κ4 worst rel = {worst_noq:.4}; κ4≈trees-only worst rel = {worst_tree:.4}");
}

/// PRODUCTION-SCALE truncation: deck 49 (river-ish unseen count), SCALE-relative
/// metric (|err| / max_h |full| — the repo's parity convention; per-hand
/// relative explodes at cancellation points and is not what CFR feels).
#[test]
#[ignore = "slow: full 38-graph eval at deck-49 for sampled hands"]
fn mass4_truncation_production_deck() {
    let deck_n = 49usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let graphs4 = connected_graphs(4);
    let graphs3 = connected_graphs(3);
    let mut rng = Lcg(0x9210D);
    let mk = |rng: &mut Lcg| -> Vec<f64> {
        (0..nh).map(|_| if rng.f() < 0.3 { 0.0 } else { rng.f() }).collect()
    };
    let rs: Vec<Vec<f64>> = (0..4).map(|_| mk(&mut rng)).collect();
    let hs: Vec<usize> = (7..nh).step_by(nh / 6).collect();
    let mut fulls = Vec::new();
    let mut noqs = Vec::new();
    let mut trees = Vec::new();
    for &h in &hs {
        let hh = hands[h];
        let kappa = |block: &[usize], mode: u8| -> f64 {
            let brs: Vec<&[f64]> = block.iter().map(|&i| rs[i].as_slice()).collect();
            let gp = group_prim(&hands, &brs, hh, deck_n);
            match block.len() {
                1 => gp.scalar[1],
                2 => -w_h_generic(&gp, 2, &[(0, 1)], deck_n),
                3 => graphs3.iter().map(|e| {
                    let s = if e.len() % 2 == 0 { 1.0 } else { -1.0 };
                    s * w_h_generic(&gp, 3, e, deck_n)
                }).sum(),
                _ => match mode {
                    0 => 0.0,
                    1 => graphs4.iter().filter(|e| e.len() == 3).map(|e| -w_h_generic(&gp, 4, e, deck_n)).sum(),
                    _ => graphs4.iter().map(|e| {
                        let s = if e.len() % 2 == 0 { 1.0 } else { -1.0 };
                        s * w_h_generic(&gp, 4, e, deck_n)
                    }).sum(),
                },
            }
        };
        let m4 = |mode: u8| -> f64 {
            set_partitions(4).iter().map(|rgs| {
                let nb = rgs.iter().cloned().max().unwrap() + 1;
                let mut blocks = vec![Vec::new(); nb];
                for (i, &b) in rgs.iter().enumerate() { blocks[b].push(i); }
                blocks.iter().map(|b| kappa(b, mode)).product::<f64>()
            }).sum()
        };
        fulls.push(m4(2)); noqs.push(m4(0)); trees.push(m4(1));
    }
    let scale = fulls.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
    let w = |xs: &[f64]| xs.iter().zip(&fulls).map(|(x, f)| (x - f).abs() / scale).fold(0.0f64, f64::max);
    eprintln!("deck-49 SCALE-relative truncation: drop-κ4 = {:.2e}; κ4≈trees = {:.2e}  ({} hands sampled)",
        w(&noqs), w(&trees), hs.len());
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL-SHAPE reference for the TRUNCATED (κ4 ≈ trees) mass4 — the exact
// per-(terminal,hand) algorithm the Metal kernel ports literally:
//   mass4_trees(h) = Σ_{15 partitions} Π κ,  κ4 = −Σ_{16 trees} W_tree
// New shapes beyond κ3 (both gated vs the generic evaluator):
//   W_star(j; i,k,l)   = Σ_{g⊥h} r_j[g]·X_i(g)·X_k(g)·X_l(g)      (O(nh) loop)
//   W_path4(i-j-k-l)   = Σ_c U_{j,i}(c)·U_{k,l}(c) − Σ_{g⊥h} r_j r_k X_i X_l
//     where U_{j,i}(c) = Σ_{g∋c,g⊥h} r_j[g]·X_i(g)   (the A−B edge pattern
//     applied to the middle edge (j,k); share-2 hands corrected by the loop).
// ═══════════════════════════════════════════════════════════════════════════

fn xval(p: &Prim, r: &[f64], g: usize, a: usize, b: usize) -> f64 {
    p.pm[a] + p.pm[b] - r[g]
}

fn w_star(hands: &[(u8, u8)], hh: (u8, u8), rj: &[f64], pi: &Prim, pk: &Prim, pl: &Prim,
          ri: &[f64], rk: &[f64], rl: &[f64]) -> f64 {
    let mut v = 0.0;
    for (g, &(a, b)) in hands.iter().enumerate() {
        if rj[g] == 0.0 || !disjoint((a, b), hh) { continue; }
        let (a, b) = (a as usize, b as usize);
        v += rj[g] * xval(pi, ri, g, a, b) * xval(pk, rk, g, a, b) * xval(pl, rl, g, a, b);
    }
    v
}

fn w_path4(hands: &[(u8, u8)], hh: (u8, u8), deck_n: usize,
           ri: &[f64], rj: &[f64], rk: &[f64], rl: &[f64],
           pi: &Prim, pl: &Prim) -> f64 {
    // U_{j,i}[c] = Σ_{g∋c, g⊥h} r_j[g]·X_i(g);  V analog for (k,l).
    let mut u = vec![0.0f64; deck_n];
    let mut w = vec![0.0f64; deck_n];
    let mut share2 = 0.0f64;
    for (g, &(a, b)) in hands.iter().enumerate() {
        if !disjoint((a, b), hh) { continue; }
        let (a, b) = (a as usize, b as usize);
        let xi = xval(pi, ri, g, a, b);
        let xl = xval(pl, rl, g, a, b);
        if rj[g] != 0.0 {
            u[a] += rj[g] * xi;
            u[b] += rj[g] * xi;
        }
        if rk[g] != 0.0 {
            w[a] += rk[g] * xl;
            w[b] += rk[g] * xl;
        }
        share2 += rj[g] * rk[g] * xi * xl;
    }
    let mut v = 0.0;
    for c in 0..deck_n { v += u[c] * w[c]; }
    v - share2
}

/// The truncated assembly, kernel-shaped (per-h Prim tables + closed forms).
fn mass4_trees_kernel(
    hands: &[(u8, u8)], deck_n: usize, h: usize, rs: &[Vec<f64>],
) -> f64 {
    let hh = hands[h];
    let prims: Vec<Prim> = (0..4).map(|i| prim(hands, &rs[i], hh, deck_n)).collect();
    let s = |i: usize| prims[i].s;
    let c_pair = |i: usize, j: usize| closed_c(hands, &rs[i], &rs[j], &prims[i], &prims[j], hh, deck_n);
    let k3 = |i: usize, j: usize, k: usize| -> f64 {
        let wp = |ctr: usize, x: usize, y: usize| closed_w_path(hands, &rs[ctr], &prims[x], &prims[y], &rs[x], &rs[y], hh);
        let wt = closed_w_tri(hands, [&rs[i], &rs[j], &rs[k]], [&prims[i], &prims[j], &prims[k]], hh, deck_n);
        wp(i, j, k) + wp(j, i, k) + wp(k, i, j) - wt
    };
    // κ4 ≈ −Σ trees: 4 stars (center m) + 12 labeled paths (i-j-k-l, unordered ends)
    let mut k4 = 0.0f64;
    for ctr in 0..4usize {
        let leaves: Vec<usize> = (0..4).filter(|&x| x != ctr).collect();
        k4 -= w_star(hands, hh, &rs[ctr], &prims[leaves[0]], &prims[leaves[1]], &prims[leaves[2]],
                     &rs[leaves[0]], &rs[leaves[1]], &rs[leaves[2]]);
    }
    // paths: choose the middle EDGE (j,k) ordered pair j<k? Labeled paths i-j-k-l:
    // middle pair {j,k} (6 choices) × assignment of the remaining two as ends (2) = 12.
    for j in 0..4usize {
        for k in (j + 1)..4 {
            let rest: Vec<usize> = (0..4).filter(|&x| x != j && x != k).collect();
            for (i, l) in [(rest[0], rest[1]), (rest[1], rest[0])] {
                // path i - j - k - l (edges (i,j),(j,k),(k,l))
                k4 -= w_path4(hands, hh, deck_n, &rs[i], &rs[j], &rs[k], &rs[l], &prims[i], &prims[l]);
            }
        }
    }
    // 15-partition assembly
    let kappa = |block: &[usize]| -> f64 {
        match block.len() {
            1 => s(block[0]),
            2 => -c_pair(block[0], block[1]),
            3 => k3(block[0], block[1], block[2]),
            _ => k4,
        }
    };
    set_partitions(4).iter().map(|rgs| {
        let nb = rgs.iter().cloned().max().unwrap() + 1;
        let mut blocks = vec![Vec::new(); nb];
        for (i, &b) in rgs.iter().enumerate() { blocks[b].push(i); }
        blocks.iter().map(|b| kappa(b)).product::<f64>()
    }).sum()
}

#[test]
fn kernel_shape_matches_generic_trees() {
    let deck_n = 12usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let graphs4 = connected_graphs(4);
    let graphs3 = connected_graphs(3);
    let mut rng = Lcg(0x5487E);
    let mk = |rng: &mut Lcg| -> Vec<f64> {
        (0..nh).map(|_| if rng.f() < 0.3 { 0.0 } else { rng.f() }).collect()
    };
    let rs: Vec<Vec<f64>> = (0..4).map(|_| mk(&mut rng)).collect();
    let mut worst_shape = 0.0f64;
    for h in (0..nh).step_by(9) {
        let hh = hands[h];
        // per-shape gates: W_star and W_path4 vs generic single-graph evals
        let gp4 = group_prim(&hands, &[&rs[0], &rs[1], &rs[2], &rs[3]], hh, deck_n);
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-9);
        // star center 1: edges (0,1),(1,2),(1,3)
        let prims: Vec<Prim> = (0..4).map(|i| prim(&hands, &rs[i], hh, deck_n)).collect();
        let gs = w_h_generic(&gp4, 4, &[(0, 1), (1, 2), (1, 3)], deck_n);
        let ks = w_star(&hands, hh, &rs[1], &prims[0], &prims[2], &prims[3], &rs[0], &rs[2], &rs[3]);
        worst_shape = worst_shape.max(rel(ks, gs));
        assert!(rel(ks, gs) < 1e-10, "star: {ks} vs {gs}");
        // path 0-1-2-3: edges (0,1),(1,2),(2,3)
        let gpp = w_h_generic(&gp4, 4, &[(0, 1), (1, 2), (2, 3)], deck_n);
        let kp = w_path4(&hands, hh, deck_n, &rs[0], &rs[1], &rs[2], &rs[3], &prims[0], &prims[3]);
        worst_shape = worst_shape.max(rel(kp, gpp));
        assert!(rel(kp, gpp) < 1e-10, "path4: {kp} vs {gpp}");
        // full truncated assembly vs generic trees-mode
        let kappa_g = |block: &[usize]| -> f64 {
            let brs: Vec<&[f64]> = block.iter().map(|&i| rs[i].as_slice()).collect();
            let gp = group_prim(&hands, &brs, hh, deck_n);
            match block.len() {
                1 => gp.scalar[1],
                2 => -w_h_generic(&gp, 2, &[(0, 1)], deck_n),
                3 => graphs3.iter().map(|e| {
                    let sg = if e.len() % 2 == 0 { 1.0 } else { -1.0 };
                    sg * w_h_generic(&gp, 3, e, deck_n)
                }).sum(),
                _ => graphs4.iter().filter(|e| e.len() == 3).map(|e| -w_h_generic(&gp, 4, e, deck_n)).sum(),
            }
        };
        let m_g: f64 = set_partitions(4).iter().map(|rgs| {
            let nb = rgs.iter().cloned().max().unwrap() + 1;
            let mut blocks = vec![Vec::new(); nb];
            for (i, &b) in rgs.iter().enumerate() { blocks[b].push(i); }
            blocks.iter().map(|b| kappa_g(b)).product::<f64>()
        }).sum();
        let m_k = mass4_trees_kernel(&hands, deck_n, h, &rs);
        worst_shape = worst_shape.max(rel(m_k, m_g));
        assert!(rel(m_k, m_g) < 1e-9, "assembly h={h}: {m_k} vs {m_g}");
    }
    eprintln!("kernel-shape (star, path4, truncated assembly) vs generic: worst rel = {worst_shape:.2e}");
}

/// TRUNCATION LADDER @deck-49: pairs-only (κ2, O(52)/h, FAST) vs trees vs full.
#[test]
#[ignore = "slow deck-49"]
fn mass4_truncation_ladder() {
    let deck_n = 49usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let graphs4 = connected_graphs(4);
    let graphs3 = connected_graphs(3);
    let mut rng = Lcg(0x1ADDE1);
    let mk = |rng: &mut Lcg| -> Vec<f64> { (0..nh).map(|_| if rng.f()<0.3 {0.0} else {rng.f()}).collect() };
    let rs: Vec<Vec<f64>> = (0..4).map(|_| mk(&mut rng)).collect();
    let hs: Vec<usize> = (7..nh).step_by(nh/6).collect();
    let mut fulls=Vec::new(); let mut pairs=Vec::new(); let mut nok3=Vec::new();
    for &h in &hs {
        let hh = hands[h];
        let kappa = |block: &[usize], mode: u8| -> f64 {
            let brs: Vec<&[f64]> = block.iter().map(|&i| rs[i].as_slice()).collect();
            let gp = group_prim(&hands, &brs, hh, deck_n);
            match block.len() {
                1 => gp.scalar[1],
                2 => -w_h_generic(&gp, 2, &[(0,1)], deck_n),
                3 => if mode==0 {0.0} else { graphs3.iter().map(|e| { let s=if e.len()%2==0 {1.0} else {-1.0}; s*w_h_generic(&gp,3,e,deck_n)}).sum() },
                _ => if mode<=1 {0.0} else { graphs4.iter().map(|e| { let s=if e.len()%2==0 {1.0} else {-1.0}; s*w_h_generic(&gp,4,e,deck_n)}).sum() },
            }
        };
        let m4 = |mode: u8| -> f64 {
            set_partitions(4).iter().map(|rgs| {
                let nb=rgs.iter().cloned().max().unwrap()+1; let mut b=vec![Vec::new();nb];
                for (i,&x) in rgs.iter().enumerate(){b[x].push(i);}
                b.iter().map(|bl| kappa(bl,mode)).product::<f64>()
            }).sum()
        };
        fulls.push(m4(2)); pairs.push(m4(0)); nok3.push(m4(1));
    }
    // factored = product of singleton totals with NO collision correction
    let fact: Vec<f64> = hs.iter().map(|&h| {
        let hh = hands[h];
        (0..4).map(|i| { let gp=group_prim(&hands,&[rs[i].as_slice()],hh,deck_n); gp.scalar[1] }).product()
    }).collect();
    let scale = fulls.iter().cloned().fold(0.0,|a:f64,b| a.max(b.abs()));
    let w = |xs: &[f64]| xs.iter().zip(&fulls).map(|(x,f)| (x-f).abs()/scale).fold(0.0,f64::max);
    eprintln!("deck-49 scale-rel vs FULL: factored={:.2e}  pairs-only(κ2)={:.2e}  trees(no-κ4)={:.2e}",
        w(&fact), w(&pairs), w(&nok3));
}

/// Gate the PRODUCTION k23_fast (order-consistent truncated κ3 corrections)
/// at deck-49 vs the EXACT κ2+κ3 assembly (drop-κ4 config) and vs FULL.
#[test]
#[ignore = "slow deck-49"]
fn k23_fast_production_accuracy() {
    use solver_core::solver::cluster_mass::{mass_cluster_k23_fast, mass_cluster_pairs_fast};
    let deck_n = 49usize;
    let hands = make_hands(deck_n as u8);
    let nh = hands.len();
    let hc: Vec<u8> = hands.iter().flat_map(|&(a, b)| [a, b]).collect();
    let graphs3 = connected_graphs(3);
    let mut rng = Lcg(0x23F457);
    let mk = |rng: &mut Lcg| -> Vec<f64> { (0..nh).map(|_| if rng.f()<0.3 {0.0} else {rng.f()}).collect() };
    let rs64: Vec<Vec<f64>> = (0..4).map(|_| mk(&mut rng)).collect();
    let rs32: Vec<Vec<f32>> = rs64.iter().map(|r| r.iter().map(|&x| x as f32).collect()).collect();
    let refs: Vec<&[f32]> = rs32.iter().map(|r| r.as_slice()).collect();
    let k23 = mass_cluster_k23_fast(&refs, &hc, nh);
    let k2 = mass_cluster_pairs_fast(&refs, &hc, nh);
    let hs: Vec<usize> = (7..nh).step_by(nh / 6).collect();
    let mut exact = Vec::new(); let mut full = Vec::new();
    for &h in &hs {
        let hh = hands[h];
        let kappa = |block: &[usize], with_k4: bool| -> f64 {
            let brs: Vec<&[f64]> = block.iter().map(|&i| rs64[i].as_slice()).collect();
            let gp = group_prim(&hands, &brs, hh, deck_n);
            match block.len() {
                1 => gp.scalar[1],
                2 => -w_h_generic(&gp, 2, &[(0,1)], deck_n),
                3 => graphs3.iter().map(|e| { let s=if e.len()%2==0 {1.0} else {-1.0}; s*w_h_generic(&gp,3,e,deck_n)}).sum(),
                _ => if with_k4 { connected_graphs(4).iter().map(|e| { let s=if e.len()%2==0 {1.0} else {-1.0}; s*w_h_generic(&gp,4,e,deck_n)}).sum() } else { 0.0 },
            }
        };
        let m4 = |with_k4: bool| -> f64 {
            set_partitions(4).iter().map(|rgs| {
                let nb=rgs.iter().cloned().max().unwrap()+1; let mut b=vec![Vec::new();nb];
                for (i,&x) in rgs.iter().enumerate(){b[x].push(i);}
                b.iter().map(|bl| kappa(bl, with_k4)).product::<f64>()
            }).sum()
        };
        exact.push(m4(false)); full.push(m4(true));
    }
    let scale = full.iter().cloned().fold(0.0f64, |a,b| a.max(b.abs()));
    let mut w23f = 0.0f64; let mut w2f = 0.0f64; let mut wex = 0.0f64;
    for (n, &h) in hs.iter().enumerate() {
        w23f = w23f.max((k23[h] as f64 - full[n]).abs()/scale);
        w2f = w2f.max((k2[h] as f64 - full[n]).abs()/scale);
        wex = wex.max((exact[n] - full[n]).abs()/scale);
    }
    eprintln!("deck-49 scale-rel vs FULL: k23_fast={:.3e}  pairs_fast={:.3e}  exact-κ2κ3={:.3e}", w23f, w2f, wex);
}
