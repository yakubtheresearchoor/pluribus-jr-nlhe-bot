//! STEAL-THROUGH RATE (2026-06-16): P(all 5 players behind fold to a UTG open),
//! computed PURELY from the average-strategy fold frequencies — the instrument
//! that's proven reliable (three CFV reconstructions failed; every strategy
//! read worked). Tests the defenders-over-fold hypothesis for the uniform
//! preflop looseness. PRE-REGISTERED: >65% confirms over-folding, <55% refutes,
//! 55-65% ambiguous. Walk the all-fold line from the UTG open, multiplying each
//! seat's combo-weighted fold probability.

use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree_preflop_only;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

fn cap3() -> FlatTree {
    let spec = production_game_v1();
    let mrc = MAX_NA_PREFLOP.saturating_sub(2);
    let mut cfg = spec.preflop_tree_config(BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect() });
    cfg.max_bets_per_street = BetCap::all(3);
    build_tree_preflop_only(&cfg).unwrap()
}

#[test]
#[ignore = "diagnostic; --ignored --nocapture --release"]
fn steal_through_probe() {
    let tree = cap3();
    let nc = NUM_PREFLOP_CLASSES;
    let mut solver = PreflopVectorCfr::new(&tree);
    // PF_BLOB overrides the default, so the FIXED strategy can be measured
    // without clobbering the old broken-strategy blob.
    let p = std::env::var("PF_BLOB")
        .unwrap_or_else(|_| format!("{}/../preflop_out_v1/preflop.blob", env!("CARGO_MANIFEST_DIR")));
    let b = std::fs::read(&p).unwrap();
    let n1 = b.iter().position(|&x| x==b'\n').unwrap();
    let n2 = n1+1+b[n1+1..].iter().position(|&x| x==b'\n').unwrap();
    let f: Vec<f32> = b[n2+1..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    solver.cum_strategy.copy_from_slice(&f);
    let avg = solver.average_strategy(&tree);
    let combos: Vec<f64> = (0..nc).map(|c| PreflopClass(c as u8).num_combos() as f64).collect();
    let tot_combos: f64 = combos.iter().sum();

    // combo-weighted fold probability at a player node (fold = action_label 0)
    let fold_prob = |node: usize| -> f64 {
        let local = solver.local_offset[node];
        let na = tree.nodes[node].num_children as usize;
        let off = local * MAX_NA_PREFLOP * nc;
        let labels: Vec<u8> = tree.node_children(node).iter().map(|&c| tree.nodes[c as usize].action_label).collect();
        let fa = labels.iter().position(|&l| l == 0);
        let fa = match fa { Some(a) => a, None => return 0.0 }; // no fold action
        let mut num = 0.0; let mut den = 0.0;
        for c in 0..nc {
            // normalize over actions for this class
            let s: f32 = (0..na).map(|a| avg[off + a*nc + c].max(0.0)).sum();
            if s > 0.0 {
                num += combos[c] * (avg[off + fa*nc + c].max(0.0) / s) as f64;
            }
            den += combos[c];
        }
        let _ = (tot_combos,);
        num / den
    };
    let fold_child = |node: usize| -> Option<usize> {
        tree.node_children(node).iter().copied().map(|x| x as usize)
            .find(|&ch| tree.nodes[ch].action_label == 0)
    };
    let smallest_raise_child = |node: usize| -> usize {
        // first raise (label 4) child = smallest size
        tree.node_children(node).iter().copied().find(|&c| tree.nodes[c as usize].action_label == 4).unwrap() as usize
    };

    // UTG = node 0. Take the smallest open, then walk the all-fold line.
    let open = smallest_raise_child(0);
    eprintln!("\n=== STEAL-THROUGH: P(all behind fold to UTG smallest open) ===");
    eprintln!("(pre-registered: >65% confirms over-folding, <55% refutes, 55-65% ambiguous)");
    let mut node = open;
    let mut steal = 1.0f64;
    let mut steps = 0;
    loop {
        if !tree.nodes[node].is_player() || solver.local_offset[node] == usize::MAX { break; }
        let pl = tree.nodes[node].player_id;
        let labels: Vec<u8> = tree.node_children(node).iter().map(|&c| tree.nodes[c as usize].action_label).collect();
        let contribs: Vec<i32> = (0..6).map(|q| tree.get_contribution(node, q as u8)).collect();
        let fp = fold_prob(node);
        eprintln!("  node {node} seat {pl} contribs {contribs:?} labels {labels:?} → fold = {fp:.3}");
        steal *= fp;
        steps += 1;
        match fold_child(node) { Some(ch) => node = ch, None => break }
        if steps > 6 { break; }
    }
    eprintln!("\nSTEAL-THROUGH RATE = {:.1}%  ({steps} seats folded around)", steal*100.0);
    eprintln!("{}", if steal > 0.65 { "→ >65%: DEFENSE HYPOTHESIS CONFIRMED (defenders over-fold)" }
        else if steal < 0.55 { "→ <55%: defense roughly sane → REFUTED, look opener-side" }
        else { "→ 55-65%: AMBIGUOUS → option 2 (in-solver instrumentation)" });
}
