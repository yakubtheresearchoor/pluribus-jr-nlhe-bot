//! UTG ACTION-EV DECOMPOSITION (2026-06-16): read the cause of the preflop
//! over-looseness straight off the converged solve. At the UTG open, compute
//! each action's value for 72o and AA under the deployed (average) profile,
//! TWICE: with the real postflop oracle, and with the oracle ZEROED. Then
//!   raise-EV(zeroed)        = pure STEAL value (win blinds when all fold)
//!   raise-EV(real) − zeroed = postflop CONTINUATION value (when called)
//! If 72o's raise beats fold on the STEAL component alone → defenders
//! over-fold (defense-side bug). If it needs the continuation → postflop is
//! over-valuing the hand (seam/range, option-1 territory).

use play_harness::preflop_oracle::banked_read_source;
use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::card_from_str;
use solver_core::solver::postflop_oracle::{BucketKeyedOracle, SeamCell};
use solver_core::solver::preflop_cfr::{make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr};
use solver_core::solver::flop_start_vector_cfr::DcfrParams;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
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
fn load_cells(root: &str) -> Vec<(u8, i32, i32, usize)> {
    std::fs::read_to_string(format!("{root}/cells.txt")).unwrap().lines().filter(|l| l.starts_with("CELL live=")).map(|l| {
        let g = |k: &str| -> i64 { l[l.find(&format!("{k}=")).unwrap()+k.len()+1..].split_whitespace().next().unwrap().parse().unwrap() };
        (g("live") as u8, g("commit") as i32, g("pot") as i32, g("b") as usize)
    }).collect()
}

#[test]
#[ignore = "diagnostic; --ignored --nocapture --release"]
fn utg_action_ev_probe() {
    let spec = production_game_v1();
    let bp_root = format!("{}/../blueprint_out_v1", env!("CARGO_MANIFEST_DIR"));
    let tree = cap3();
    let nc = NUM_PREFLOP_CLASSES;
    let mut solver = PreflopVectorCfr::new(&tree);
    // load converged blob
    let p = format!("{}/../preflop_out_v1/preflop.blob", env!("CARGO_MANIFEST_DIR"));
    let b = std::fs::read(&p).unwrap();
    let n1 = b.iter().position(|&x| x==b'\n').unwrap();
    let n2 = n1+1+b[n1+1..].iter().position(|&x| x==b'\n').unwrap();
    let f: Vec<f32> = b[n2+1..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    solver.cum_strategy.copy_from_slice(&f);
    // everyone plays the DEPLOYED average
    solver.strategy = solver.average_strategy(&tree);

    let table = PreflopChanceTable::new(6, vec![vec![1.0f32/nc as f32; nc]; 6]);
    let canon = table.canonical_flops.clone();
    let cells = load_cells(&bp_root);
    let source = banked_read_source(bp_root.clone(), cells, canon, spec.stack);
    let mut oracle = BucketKeyedOracle::new(spec.stack, 6, 0, source);
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);

    let reach = solver.compute_preflop_reach(&tree, None);
    let chance_nodes = solver.preflop_chance_node_indices(&tree);
    let np = 6usize;
    let t = 2u8; // UTG = seat 2 (node 0)

    // chance-node CFV for UTG, real and zeroed. COLLAPSE by bucket key (the
    // live-UTG expansion depends only on (flop,t,key) for a frozen oracle), so
    // compute once per key and broadcast — same as the production iteration.
    let mut cc_real: Vec<Vec<f32>> = vec![vec![0.0; nc]; tree.num_nodes()];
    let mut by_key: std::collections::HashMap<(u8, i64), Vec<f32>> = std::collections::HashMap::new();
    for &c in &chance_nodes {
        let mask = tree.get_folded_mask(c);
        if (mask >> t) & 1 == 1 {
            let base = c * nc;
            let reach_at: Vec<Vec<f32>> = (0..np).map(|q| reach[q][base..base+nc].to_vec()).collect();
            cc_real[c] = term_fn(c, t, &reach_at);
        } else {
            let cell = SeamCell::at_chance_node(&tree, c, np);
            let key = cell.bucket_key(spec.stack);
            cc_real[c] = match by_key.get(&key) {
                Some(v) => v.clone(),
                None => {
                    let v = solver.compute_chance_node_cfv_with_expansion_for_cell(c, t, &reach, &table, &mut oracle, cell, mask);
                    by_key.insert(key, v.clone());
                    v
                }
            };
        }
    }
    // zeroed: postflop (live-UTG) chance nodes → 0; folded-UTG keep term_fn.
    let mut cc_zero = cc_real.clone();
    for &c in &chance_nodes {
        if (tree.get_folded_mask(c) >> t) & 1 == 0 { cc_zero[c] = vec![0.0; nc]; }
    }

    let params = DcfrParams::new(0);
    let run = |solver: &mut PreflopVectorCfr, seed: &[Vec<f32>]| -> Vec<Vec<f32>> {
        let mut cfv = seed.to_vec();
        solver.bottom_up_preflop_for_traverser(&tree, t, &chance_nodes, &reach, &term_fn, &mut cfv, &params);
        cfv
    };
    let cfv_real = run(&mut solver, &cc_real);
    let cfv_zero = run(&mut solver, &cc_zero);

    let aa = PreflopClass::from_combo(card_from_str("Ac").unwrap(), card_from_str("Ad").unwrap()).index();
    let t72 = PreflopClass::from_combo(card_from_str("7c").unwrap(), card_from_str("2d").unwrap()).index();
    let kids = tree.node_children(0);
    let labels: Vec<u8> = kids.iter().map(|&c| tree.nodes[c as usize].action_label).collect();
    let lab = |l: u8| match l { 0=>"fold",1=>"check",2=>"call",3=>"bet",4=>"raise",5=>"allin",_=>"?" };

    for (nm, cls) in [("72o", t72), ("AA", aa)] {
        eprintln!("\n=== UTG action EVs for {nm} (fold baseline) ===");
        eprintln!("{:>6} {:>10} {:>10} {:>10}", "action", "EV(real)", "EV(steal)", "EV(postfl)");
        for (a, &kid) in kids.iter().enumerate() {
            let real = cfv_real[kid as usize][cls];
            let steal = cfv_zero[kid as usize][cls];
            eprintln!("{:>6} {:>10.3} {:>10.3} {:>10.3}", lab(labels[a]), real, steal, real - steal);
        }
        let fold_ev = cfv_real[kids[0] as usize][cls];
        let best_raise_steal = kids.iter().enumerate().filter(|(a,_)| labels[*a]==4).map(|(_,&k)| cfv_zero[k as usize][cls]).fold(f32::MIN, f32::max);
        eprintln!("  → best raise STEAL-only EV {best_raise_steal:.3}  vs  fold EV {fold_ev:.3}");
        eprintln!("    {}", if best_raise_steal > fold_ev { "STEAL alone beats fold → DEFENSE-side (defenders over-fold)" } else { "steal alone < fold → needs CONTINUATION → seam/range" });
    }
}
