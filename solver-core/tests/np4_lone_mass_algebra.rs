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
