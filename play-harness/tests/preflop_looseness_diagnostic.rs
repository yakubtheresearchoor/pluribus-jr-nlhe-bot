//! PREFLOP LOOSENESS DIAGNOSTIC (2026-06-16): is the UTG over-opening SELECTIVE
//! or UNIFORM? The "frozen postflop valued vs average caller" (seam) hypothesis
//! predicts SELECTIVE over-raising — concentrated in hands that do OK vs a
//! random hand but POORLY vs a strong continuing range (the divergent band:
//! suited junk, weak aces). Under-convergence / missing-rake predict UNIFORM
//! looseness — even hands bad vs EVERYTHING (32o-type) raise a lot.
//!
//! Test: per class, raise-freq at the UTG open vs (equity-vs-random) and
//! (equity-vs-tight-range). If over-raised hands all have decent eq-vs-random
//! and big (eq_random − eq_tight) divergence → SEAM (option 1 fixes it). If
//! the bottom (low eq-vs-random) hands also raise → UNIFORM (option 1 won't).

use clean_rules::eval::best5;
use solver_core::abstraction::preflop_class::{class_combos, PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree_preflop_only;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

fn cap3() -> FlatTree {
    let spec = production_game_v1();
    let mrc = MAX_NA_PREFLOP.saturating_sub(2);
    let mut cfg = spec.preflop_tree_config(BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect(),
    });
    cfg.max_bets_per_street = BetCap::all(3);
    build_tree_preflop_only(&cfg).expect("tree")
}

fn sm(x: &mut u64) -> u64 { *x = x.wrapping_add(0x9E3779B97F4A7C15); let mut z=*x; z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB); z^(z>>31) }

fn name(c: usize) -> String {
    let pc = PreflopClass(c as u8);
    let (a, b) = class_combos(pc)[0];
    let r = |x: u8| b"23456789TJQKA"[(x >> 2) as usize] as char;
    let (h, l) = if (a >> 2) >= (b >> 2) { (a, b) } else { (b, a) };
    if pc.is_pair() { format!("{}{}", r(h), r(l)) }
    else { format!("{}{}{}", r(h), r(l), if pc.is_suited() { 's' } else { 'o' }) }
}

/// MC equity of class `c` vs an opponent drawn from `opp_classes` (None = any),
/// over a full random 5-card board. win + tie/2.
fn equity(c: usize, opp_pool: &Option<Vec<usize>>, rng: &mut u64, samples: usize) -> f64 {
    let (a, b) = class_combos(PreflopClass(c as u8))[0];
    let mut win = 0.0;
    for _ in 0..samples {
        // draw opponent (optionally from pool), then board
        let (o1, o2) = loop {
            let mut deck: Vec<u8> = (0..52u8).filter(|&x| x != a && x != b).collect();
            for i in 0..2 { let j = i + (sm(rng) % (deck.len() - i) as u64) as usize; deck.swap(i, j); }
            let (o1, o2) = (deck[0], deck[1]);
            if let Some(pool) = opp_pool {
                if !pool.contains(&(PreflopClass::from_combo(o1, o2).index())) { continue; }
            }
            break (o1, o2);
        };
        let mut deck: Vec<u8> = (0..52u8).filter(|&x| x != a && x != b && x != o1 && x != o2).collect();
        for i in 0..5 { let j = i + (sm(rng) % (deck.len() - i) as u64) as usize; deck.swap(i, j); }
        let board = [deck[0], deck[1], deck[2], deck[3], deck[4]];
        let me = best5(&[a, b, board[0], board[1], board[2], board[3], board[4]]).0;
        let op = best5(&[o1, o2, board[0], board[1], board[2], board[3], board[4]]).0;
        win += if me > op { 1.0 } else if me == op { 0.5 } else { 0.0 };
    }
    win / samples as f64
}

#[test]
#[ignore = "diagnostic; --ignored --nocapture --release"]
fn preflop_looseness_diagnostic() {
    let tree = cap3();
    let mut solver = PreflopVectorCfr::new(&tree);
    let nc = NUM_PREFLOP_CLASSES;
    let path = format!("{}/../preflop_out_v1/preflop.blob", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap();
    let nl1 = bytes.iter().position(|&b| b == b'\n').unwrap();
    let nl2 = nl1 + 1 + bytes[nl1 + 1..].iter().position(|&b| b == b'\n').unwrap();
    let f: Vec<f32> = bytes[nl2 + 1..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    solver.cum_strategy.copy_from_slice(&f);
    let avg = solver.average_strategy(&tree);

    // UTG open node = first decision node, seat 2.
    let root = *tree.decision_node_ids.iter().find(|&&nid| {
        let i = nid as usize;
        solver.local_offset[i] != usize::MAX && tree.nodes[i].num_children as usize >= 2
    }).unwrap() as usize;
    let na = tree.nodes[root].num_children as usize;
    let labels: Vec<u8> = tree.node_children(root).iter().map(|&c| tree.nodes[c as usize].action_label).collect();
    let off = solver.local_offset[root] * MAX_NA_PREFLOP * nc;
    let raise_freq = |c: usize| -> f64 {
        let tot: f32 = (0..na).map(|a| avg[off + a * nc + c].max(0.0)).sum();
        if tot <= 0.0 { return 0.0; }
        let r: f32 = (0..na).filter(|&a| labels[a] == 4).map(|a| avg[off + a * nc + c]).sum();
        (r / tot) as f64
    };

    let mut rng = 0x1234_5678_9abc_def0u64;
    const S: usize = 1500;
    let eq_rand: Vec<f64> = (0..nc).map(|c| equity(c, &None, &mut rng, S)).collect();
    // tight range = top 25 classes by eq-vs-random (~roughly a 15% range by combos)
    let mut order: Vec<usize> = (0..nc).collect();
    order.sort_by(|&x, &y| eq_rand[y].partial_cmp(&eq_rand[x]).unwrap());
    let tight: Vec<usize> = order[..25].to_vec();
    let eq_tight: Vec<f64> = (0..nc).map(|c| equity(c, &Some(tight.clone()), &mut rng, S)).collect();

    // ── selective vs uniform readout ──
    let mut by_rand = order.clone(); // already sorted desc by eq_rand
    eprintln!("\n=== UTG open: raise-freq vs strength (sorted weakest→strongest by eq-vs-random) ===");
    eprintln!("{:>5} {:>8} {:>8} {:>9} {:>8}", "hand", "eqRand", "eqTight", "diverge", "raise");
    by_rand.reverse();
    for &c in by_rand.iter().take(20) {
        eprintln!("{:>5} {:>8.3} {:>8.3} {:>9.3} {:>8.2}", name(c), eq_rand[c], eq_tight[c], eq_rand[c]-eq_tight[c], raise_freq(c));
    }
    eprintln!("... (strongest 8) ...");
    for &c in order.iter().take(8) {
        eprintln!("{:>5} {:>8.3} {:>8.3} {:>9.3} {:>8.2}", name(c), eq_rand[c], eq_tight[c], eq_rand[c]-eq_tight[c], raise_freq(c));
    }

    // KEY SUMMARY
    let med = { let mut v = eq_rand.clone(); v.sort_by(|a,b| a.partial_cmp(b).unwrap()); v[nc/2] };
    let over: Vec<usize> = (0..nc).filter(|&c| raise_freq(c) > 0.40).collect();
    let over_badvsrand = over.iter().filter(|&&c| eq_rand[c] < med).count();
    let n_raise_any = (0..nc).filter(|&c| raise_freq(c) > 0.40).count();
    // bottom-20 hands by eq_rand: their avg raise freq
    let bottom20: f64 = by_rand.iter().take(20).map(|&c| raise_freq(c)).sum::<f64>() / 20.0;
    let top20: f64 = order.iter().take(20).map(|&c| raise_freq(c)).sum::<f64>() / 20.0;
    eprintln!("\n=== VERDICT INPUTS ===");
    eprintln!("classes raising >40%: {n_raise_any}/169   (sane UTG ≈ 25-45)");
    eprintln!("  of those, BAD-vs-random (eqRand<median): {over_badvsrand}  (seam→~0; uniform→many)");
    eprintln!("avg raise-freq: weakest-20 hands = {bottom20:.2}  |  strongest-20 = {top20:.2}");
    eprintln!("  SELECTIVE(seam): weakest≈0, strongest≈high.  UNIFORM(conv/rake): weakest also high.");
}
