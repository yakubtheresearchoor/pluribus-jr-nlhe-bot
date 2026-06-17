//! RANGE-DEPENDENCE PROBE (2026-06-16): the decisive test for whether option 1
//! (re-value postflop vs realistic ranges) would fix the uniform preflop
//! looseness. Compute a weak hand's (72o) and a premium's (AA) postflop value
//! when the OPPONENT range is UNIFORM vs TIGHT (top ~15%). If 72o's value
//! collapses vs tight (≪ vs uniform), the looseness IS a range-assumption
//! problem → option 1 fixes it. If 72o's value is ~the same vs tight as vs
//! uniform, the looseness is NOT about ranges → option 1 won't help (structural).

use clean_rules::eval::best5;
use solver_core::abstraction::preflop_class::{class_combos, PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::{card_pair_to_index, NUM_POSSIBLE_HANDS, Card};
use solver_core::solver::preflop_start_game::compute_v_flop_at_root_converged;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

fn sm(x:&mut u64)->u64{*x=x.wrapping_add(0x9E3779B97F4A7C15);let mut z=*x;z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9);z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);z^(z>>31)}

fn eq_vs_random(c: usize, rng: &mut u64, n: usize) -> f64 {
    let (a, b) = class_combos(PreflopClass(c as u8))[0];
    let mut w = 0.0;
    for _ in 0..n {
        let mut d: Vec<u8> = (0..52u8).filter(|&x| x != a && x != b).collect();
        for i in 0..7 { let j = i + (sm(rng) % (d.len()-i) as u64) as usize; d.swap(i,j); }
        let (o1,o2,bd)=(d[0],d[1],[d[2],d[3],d[4],d[5],d[6]]);
        let me=best5(&[a,b,bd[0],bd[1],bd[2],bd[3],bd[4]]).0;
        let op=best5(&[o1,o2,bd[0],bd[1],bd[2],bd[3],bd[4]]).0;
        w += if me>op {1.0} else if me==op {0.5} else {0.0};
    }
    w / n as f64
}

#[test]
#[ignore = "diagnostic; --ignored --nocapture --release"]
fn range_dependence_probe() {
    let spec = production_game_v1();
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let tree = build_tree(&spec.flop_seam_config(2, 20, 40, bets)).unwrap();
    // a CANONICAL dry rainbow broadway flop (must be a canonical representative
    // or the chance table builds empty → all-zero values).
    use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES as NPC;
    let canon = solver_core::solver::preflop_start_game::PreflopChanceTable::new(6, vec![vec![1.0f32/NPC as f32; NPC]; 6]).canonical_flops.clone();
    let board: [Card; 3] = *canon.iter().find(|b| {
        let r: Vec<u8> = b.iter().map(|&c| c>>2).collect();
        let s: Vec<u8> = b.iter().map(|&c| c&3).collect();
        r[0]!=r[1]&&r[1]!=r[2]&&r[0]!=r[2] && s[0]!=s[1]&&s[1]!=s[2]&&s[0]!=s[2] && *r.iter().max().unwrap()>=10
    }).unwrap();

    // tight range = top 25 classes by equity-vs-random
    let mut rng = 0xABCDu64;
    let mut eqs: Vec<(usize,f64)> = (0..NUM_PREFLOP_CLASSES).map(|c|(c, eq_vs_random(c,&mut rng,1200))).collect();
    eqs.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    let tight: std::collections::HashSet<usize> = eqs[..25].iter().map(|x|x.0).collect();

    // build per-combo ranges (length NUM_POSSIBLE_HANDS)
    let uniform = vec![1.0f32; NUM_POSSIBLE_HANDS];
    let mut tight_r = vec![0.0f32; NUM_POSSIBLE_HANDS];
    for c1 in 0..52u8 { for c2 in (c1+1)..52u8 {
        if tight.contains(&PreflopClass::from_combo(c1,c2).index()) { tight_r[card_pair_to_index(c1,c2)] = 1.0; }
    }}

    let per_class = |v: &[f32], lay: &[(Card,Card)], c: usize| -> f64 {
        let mut s=0.0; let mut n=0;
        for (i,&(a,b)) in lay.iter().enumerate() {
            if PreflopClass::from_combo(a,b).index()==c { s+=v[i] as f64; n+=1; }
        }
        if n>0 { s/n as f64 } else { 0.0 }
    };
    let aa = PreflopClass::from_combo(solver_core::card::card_from_str("Ac").unwrap(), solver_core::card::card_from_str("Ad").unwrap()).index();
    let t72 = PreflopClass::from_combo(solver_core::card::card_from_str("7c").unwrap(), solver_core::card::card_from_str("2d").unwrap()).index();
    let t98 = PreflopClass::from_combo(solver_core::card::card_from_str("9c").unwrap(), solver_core::card::card_from_str("8c").unwrap()).index();

    // traverser=0 uniform; opponent=1 uniform vs tight
    let (vu, layu) = compute_v_flop_at_root_converged(board, &tree, &[uniform.clone(), uniform.clone()], 0, 16);
    // ANCHOR GATE: the probe must reproduce a KNOWN value (AA vs uniform = sane
    // high) before we trust the unknown (72o vs tight). If this is 0/nonsense,
    // the probe is still bugged → stop, go to in-solver instrumentation.
    let vmin = vu.iter().cloned().fold(f32::INFINITY, f32::min);
    let vmax = vu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let aa_n = layu.iter().filter(|&&(a,b)| PreflopClass::from_combo(a,b).index()==aa).count();
    eprintln!("ANCHOR: v.len={} min={vmin:.3} max={vmax:.3} | AA combos in layout={aa_n} | AA-vs-uniform={:.3}",
        vu.len(), { let mut s=0.0; let mut n=0; for (i,&(a,b)) in layu.iter().enumerate(){ if PreflopClass::from_combo(a,b).index()==aa {s+=vu[i] as f64; n+=1;}} if n>0 {s/n as f64} else {0.0} });
    let (vt, layt) = compute_v_flop_at_root_converged(board, &tree, &[uniform.clone(), tight_r.clone()], 0, 16);

    eprintln!("\n=== traverser-0 postflop value: opponent UNIFORM vs TIGHT (board Qc6d2h) ===");
    eprintln!("{:>5} {:>12} {:>12} {:>10}", "hand", "vs UNIFORM", "vs TIGHT", "Δ(drop)");
    for (nm,c) in [("AA",aa),("98s",t98),("72o",t72)] {
        let u=per_class(&vu,&layu,c); let t=per_class(&vt,&layt,c);
        eprintln!("{:>5} {:>12.3} {:>12.3} {:>10.3}", nm, u, t, u-t);
    }
    eprintln!("\nIf 72o drops a lot vs TIGHT (and AA stays high) → looseness IS range-assumption → option 1 fixes.");
    eprintln!("If 72o ~unchanged → NOT range-driven → option 1 won't fix (structural).");
}
