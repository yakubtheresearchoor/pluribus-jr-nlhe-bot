//! Gate mass_cluster_pairs: (1) EXACT at K=2 (D-form, no truncation) vs brute;
//! (2) at K=4 beats the factored product vs brute; (3) per-iter cost.
use solver_core::solver::cluster_mass::mass_cluster_pairs;
use solver_core::solver::showdown::factored_total_reach_product;

fn hands_deck(deck_n: u8) -> Vec<u8> {
    let mut v = Vec::new();
    for a in 0..deck_n { for b in (a+1)..deck_n { v.push(a); v.push(b); } }
    v
}
struct Lcg(u64);
impl Lcg { fn f(&mut self)->f32 { self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1); ((self.0>>33) as f32)/(1u64<<31) as f32 } }
fn disj(hc:&[u8],g:usize,h:usize)->bool{ let(a,b)=(hc[g*2],hc[g*2+1]);let(c,d)=(hc[h*2],hc[h*2+1]);a!=c&&a!=d&&b!=c&&b!=d }

#[test]
fn k2_exact_vs_brute() {
    let deck=16u8; let hc=hands_deck(deck); let nh=hc.len()/2;
    let mut rng=Lcg(0xD1); 
    let mut worst=0.0f64; let mut scale=1e-9f64;
    for _ in 0..4 {
        let rs:Vec<Vec<f32>>=(0..2).map(|_|(0..nh).map(|_| if rng.f()<0.3{0.0}else{rng.f()}).collect()).collect();
        let refs:Vec<&[f32]>=rs.iter().map(|r|r.as_slice()).collect();
        let m=mass_cluster_pairs(&refs,&hc,nh);
        for h in 0..nh {
            let mut b=0.0f64;
            for g0 in 0..nh { if rs[0][g0]==0.0||!disj(&hc,g0,h){continue;}
              for g1 in 0..nh { if rs[1][g1]==0.0||!disj(&hc,g1,h)||!disj(&hc,g1,g0){continue;}
                b += (rs[0][g0]*rs[1][g1]) as f64; }}
            scale=scale.max(b.abs());
            worst=worst.max((m[h] as f64 - b).abs());
        }
    }
    eprintln!("K=2 cluster (D-form) vs brute: worst abs/scale = {:.2e}", worst/scale);
    assert!(worst/scale < 1e-5, "K=2 must be EXACT: {}", worst/scale);
}

#[test]
fn k4_beats_factored() {
    let deck=15u8; let hc=hands_deck(deck); let nh=hc.len()/2;
    let mut rng=Lcg(0xC4);
    let rs:Vec<Vec<f32>>=(0..4).map(|_|(0..nh).map(|_| if rng.f()<0.3{0.0}else{rng.f()}).collect()).collect();
    let refs:Vec<&[f32]>=rs.iter().map(|r|r.as_slice()).collect();
    let clu=mass_cluster_pairs(&refs,&hc,nh);
    let fac=factored_total_reach_product(&refs,&hc,nh);
    let mut wc=0.0f64; let mut wf=0.0f64; let mut scale=1e-9f64;
    for h in (0..nh).step_by(3) {
        let mut b=0.0f64;
        for g0 in 0..nh { if rs[0][g0]==0.0||!disj(&hc,g0,h){continue;}
          for g1 in 0..nh { if rs[1][g1]==0.0||!disj(&hc,g1,h)||!disj(&hc,g1,g0){continue;}
            for g2 in 0..nh { if rs[2][g2]==0.0||!disj(&hc,g2,h)||!disj(&hc,g2,g0)||!disj(&hc,g2,g1){continue;}
              for g3 in 0..nh { if rs[3][g3]==0.0||!disj(&hc,g3,h)||!disj(&hc,g3,g0)||!disj(&hc,g3,g1)||!disj(&hc,g3,g2){continue;}
                b+=(rs[0][g0]*rs[1][g1]*rs[2][g2]*rs[3][g3]) as f64; }}}}
        scale=scale.max(b.abs());
        wc=wc.max((clu[h] as f64 - b).abs()); wf=wf.max((fac[h] as f64 - b).abs());
    }
    eprintln!("K=4 vs brute (scale-rel): cluster-pairs={:.3e}  factored={:.3e}", wc/scale, wf/scale);
    assert!(wc < wf, "cluster must beat factored");
}

#[test]
#[ignore = "cost bench at production nh"]
fn cost_at_production_nh() {
    // river nh ~ 1081; 4 opponents. one terminal = one mass call.
    let nh = 1081usize;
    // synthetic hand_cards: valid 2-card combos off a 5-card board (deck 47).
    let mut hc = Vec::new();
    let deck: Vec<u8> = (0..47).collect();
    'outer: for i in 0..deck.len() { for j in (i+1)..deck.len() {
        hc.push(deck[i]); hc.push(deck[j]);
        if hc.len()/2 >= nh { break 'outer; }
    }}
    let nh = hc.len()/2;
    let mut rng = Lcg(0xBEEF);
    let rs: Vec<Vec<f32>> = (0..4).map(|_| (0..nh).map(|_| if rng.f()<0.3 {0.0} else {rng.f()}).collect()).collect();
    let refs: Vec<&[f32]> = rs.iter().map(|r| r.as_slice()).collect();
    // time N mass calls (≈ N fold terminals)
    let n = 200;
    let t0 = std::time::Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..n { let m = mass_cluster_pairs(&refs, &hc, nh); acc += m[0]; }
    let per = t0.elapsed().as_secs_f64()/n as f64*1000.0;
    let t1 = std::time::Instant::now();
    for _ in 0..n { let m = factored_total_reach_product(&refs, &hc, nh); acc += m[0]; }
    let perf = t1.elapsed().as_secs_f64()/n as f64*1000.0;
    eprintln!("per-terminal (nh={nh}): cluster-pairs={per:.2}ms  factored={perf:.3}ms  ratio={:.0}x  (acc {acc:.0})", per/perf.max(1e-6));
}

#[test]
fn fast_matches_base() {
    use solver_core::solver::cluster_mass::mass_cluster_pairs_fast;
    let deck=18u8; let hc=hands_deck(deck); let nh=hc.len()/2;
    let mut rng=Lcg(0xFA57);
    let mut worst=0.0f64;
    for kk in [2usize,3,4] {
        for _ in 0..3 {
            let rs:Vec<Vec<f32>>=(0..kk).map(|_|(0..nh).map(|_| if rng.f()<0.3{0.0}else{rng.f()}).collect()).collect();
            let refs:Vec<&[f32]>=rs.iter().map(|r|r.as_slice()).collect();
            let base=mass_cluster_pairs(&refs,&hc,nh);
            let fast=mass_cluster_pairs_fast(&refs,&hc,nh);
            let scale=base.iter().fold(1e-9f32,|a,&b| a.max(b.abs())) as f64;
            for h in 0..nh { worst=worst.max((fast[h] as f64 - base[h] as f64).abs()/scale); }
        }
    }
    eprintln!("fast vs base (K=2,3,4): worst scale-rel = {worst:.2e}");
    assert!(worst < 1e-4, "fast must match base: {worst}");
}

#[test]
#[ignore = "cost"]
fn fast_cost() {
    use solver_core::solver::cluster_mass::mass_cluster_pairs_fast;
    let mut hc=Vec::new(); let deck:Vec<u8>=(0..47).collect();
    'o: for i in 0..deck.len(){for j in (i+1)..deck.len(){hc.push(deck[i]);hc.push(deck[j]); if hc.len()/2>=1081{break 'o;}}}
    let nh=hc.len()/2; let mut rng=Lcg(0xBEE);
    let rs:Vec<Vec<f32>>=(0..4).map(|_|(0..nh).map(|_| if rng.f()<0.3{0.0}else{rng.f()}).collect()).collect();
    let refs:Vec<&[f32]>=rs.iter().map(|r|r.as_slice()).collect();
    let n=200; let mut acc=0.0f32;
    let t=std::time::Instant::now();
    for _ in 0..n { acc+=mass_cluster_pairs_fast(&refs,&hc,nh)[0]; }
    let per=t.elapsed().as_secs_f64()/n as f64*1000.0;
    let tf=std::time::Instant::now();
    for _ in 0..n { acc+=factored_total_reach_product(&refs,&hc,nh)[0]; }
    let perf=tf.elapsed().as_secs_f64()/n as f64*1000.0;
    eprintln!("fast per-terminal (nh={nh}): {per:.3}ms  vs factored {perf:.3}ms  ratio={:.1}x  (acc {acc:.0})", per/perf.max(1e-6));
}

#[test]
fn k23_accuracy_and_cost() {
    use solver_core::solver::cluster_mass::{mass_cluster_k23_fast, mass_cluster_pairs_fast};
    // accuracy vs brute at deck-15 (K=4)
    let deck=15u8; let hc=hands_deck(deck); let nh=hc.len()/2;
    let mut rng=Lcg(0x23);
    let rs:Vec<Vec<f32>>=(0..4).map(|_|(0..nh).map(|_| if rng.f()<0.3{0.0}else{rng.f()}).collect()).collect();
    let refs:Vec<&[f32]>=rs.iter().map(|r|r.as_slice()).collect();
    let k23=mass_cluster_k23_fast(&refs,&hc,nh);
    let k2=mass_cluster_pairs_fast(&refs,&hc,nh);
    let fac=factored_total_reach_product(&refs,&hc,nh);
    let mut scale=1e-9f64; let mut w23=0.0f64; let mut w2=0.0f64; let mut wf=0.0f64;
    for h in (0..nh).step_by(4) {
        let mut b=0.0f64;
        for g0 in 0..nh { if rs[0][g0]==0.0||!disj(&hc,g0,h){continue;}
          for g1 in 0..nh { if rs[1][g1]==0.0||!disj(&hc,g1,h)||!disj(&hc,g1,g0){continue;}
            for g2 in 0..nh { if rs[2][g2]==0.0||!disj(&hc,g2,h)||!disj(&hc,g2,g0)||!disj(&hc,g2,g1){continue;}
              for g3 in 0..nh { if rs[3][g3]==0.0||!disj(&hc,g3,h)||!disj(&hc,g3,g0)||!disj(&hc,g3,g1)||!disj(&hc,g3,g2){continue;}
                b+=(rs[0][g0]*rs[1][g1]*rs[2][g2]*rs[3][g3]) as f64; }}}}
        scale=scale.max(b.abs());
        w23=w23.max((k23[h] as f64-b).abs()); w2=w2.max((k2[h] as f64-b).abs()); wf=wf.max((fac[h] as f64-b).abs());
    }
    eprintln!("deck-15 vs brute (scale-rel): k23={:.3e}  pairs={:.3e}  factored={:.3e}", w23/scale, w2/scale, wf/scale);
    assert!(w23 < w2, "k23 must beat pairs-only");
    // cost at production nh
    let mut hc2=Vec::new(); let dk:Vec<u8>=(0..47).collect();
    'o: for i in 0..dk.len(){for j in (i+1)..dk.len(){hc2.push(dk[i]);hc2.push(dk[j]); if hc2.len()/2>=1081{break 'o;}}}
    let nh2=hc2.len()/2; let mut rng2=Lcg(0xC057);
    let rs2:Vec<Vec<f32>>=(0..4).map(|_|(0..nh2).map(|_| if rng2.f()<0.3{0.0}else{rng2.f()}).collect()).collect();
    let refs2:Vec<&[f32]>=rs2.iter().map(|r|r.as_slice()).collect();
    let n=50; let mut acc=0.0f32;
    let t=std::time::Instant::now();
    for _ in 0..n { acc+=mass_cluster_k23_fast(&refs2,&hc2,nh2)[0]; }
    let per=t.elapsed().as_secs_f64()/n as f64*1000.0;
    eprintln!("k23 per-terminal (nh={nh2}): {per:.3}ms (acc {acc:.0})");
}
