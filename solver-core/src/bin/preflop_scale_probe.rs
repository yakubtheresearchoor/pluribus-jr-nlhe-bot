//! PREFLOP SCALE PROBE (2026-06-17): after the reach-init fix (sum-1 default),
//! the converged preflop LIMPS everything (raise 0%) — the residual terminal
//! scale still dwarfs the chip-scale continuation, so the solver minimizes
//! investment. The blocking matrix M is already a [0,1] fraction, so the
//! residual is NOT simple combo-weighting; trace it, don't guess.
//!
//! This isolates the TERMINAL/factored-sum scale. Feed a KNOWN chip-scale
//! synthetic continuation (+C for AA / −C for 72o at EVERY preflop chance leaf)
//! through the real bottom_up + the production (bootstrap-pairwise) terminal fn,
//! with the sum-1 reach, and read node-0's action values. Sweep C ∈ {10, 0, 1e6}:
//!   - if node-0 ≈ ±C (tracks the continuation), the terminals are chip-scale;
//!   - if node-0 ≈ 1e6 even at C=10 or C=0, the TERMINALS/factored-sum produce a
//!     hand-scale mass that swamps the continuation → the residual that limps.

use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::card_from_str;
use solver_core::solver::flop_start_vector_cfr::DcfrParams;
use solver_core::solver::preflop_cfr::{
    make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr,
};
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
    build_tree_preflop_only(&cfg).expect("cap-3 preflop tree")
}

fn main() {
    let tree = cap3();
    let nn = tree.num_nodes();
    let nc = NUM_PREFLOP_CLASSES;
    let aa = PreflopClass::from_combo(card_from_str("Ac").unwrap(), card_from_str("Ad").unwrap()).index();
    let t72 = PreflopClass::from_combo(card_from_str("7c").unwrap(), card_from_str("2d").unwrap()).index();

    let mut solver = PreflopVectorCfr::new(&tree);
    solver.compute_preflop_strategy(&tree); // uniform at iter 0
    let reach = solver.compute_preflop_reach(&tree, None);
    let chance_nodes = solver.preflop_chance_node_indices(&tree);
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    let params = DcfrParams::new(0);
    let trav = tree.nodes[0].player_id;

    // Root opponent reach mass (sanity: should be ~1.0 after the sum-1 fix).
    let opp = ((trav as usize) + 1) % tree.num_players as usize;
    let root_mass: f32 = (0..nc).map(|c| reach[opp][c]).sum();
    let n_chance = chance_nodes.len();
    let labels: Vec<u8> =
        tree.node_children(0).iter().map(|&k| tree.nodes[k as usize].action_label).collect();
    println!("cap-3 preflop: {nn} nodes, {n_chance} preflop chance leaves, traverser(node0)={trav}");
    println!("root opp reach mass = {root_mass:.4} (≈1.0 expected) | node-0 labels {labels:?}\n");

    // Inspect chance-node reach structure: per-player reach mass + folded mask.
    let np = tree.num_players as usize;
    println!("chance-node reach masses (first 4, per player; folded mask in []):");
    for &ci in chance_nodes.iter().take(4) {
        let fm = tree.get_folded_mask(ci);
        let masses: Vec<String> = (0..np)
            .map(|p| format!("{:.3}", (0..nc).map(|c| reach[p][ci * nc + c]).sum::<f32>()))
            .collect();
        println!("  node {ci} [fold {fm:06b}]: {}", masses.join(" "));
    }
    println!();

    // Player nodes owned by the traverser, sampled at increasing depth.
    let mut owned: Vec<usize> = (0..nn)
        .filter(|&i| tree.nodes[i].is_player() && tree.nodes[i].player_id == trav
            && tree.nodes[i].board_state == solver_core::tree::action::BoardState::Preflop as u8)
        .collect();
    owned.sort();
    let sample: Vec<usize> = owned.iter().take(2).copied().collect();

    // DERIVED reach weight per chance leaf, EXACT: route the opp reaches at the
    // chance node through the SAME multiway machinery the fold terminals use
    // (preflop_fold_terminal_cfv_multiway_pairwise with unit chip_delta) → the
    // per-class reach weight w[c]. The continuation is then oracle[c] × w[c],
    // element-wise — i.e. "treat the chance leaf as a terminal whose (per-class)
    // chip_delta is the oracle EV." Same-node-same-weight holds BY CONSTRUCTION:
    // identical function, identical opp-reach inputs (all p≠traverser). The
    // oracle's own card-removal is consistent — M here weights the opp-reach MASS
    // to the non-blocking range the oracle averaged over (range vs mass, not a
    // double count).
    use solver_core::solver::preflop_terminal::{
        build_class_blocking_matrix, preflop_fold_terminal_cfv_multiway_pairwise,
    };
    let mblock = build_class_blocking_matrix();
    let wvec = |ci: usize| -> Vec<f32> {
        let opp: Vec<&[f32]> = (0..np)
            .filter(|&p| p != trav as usize)
            .map(|p| &reach[p][ci * nc..ci * nc + nc])
            .collect();
        preflop_fold_terminal_cfv_multiway_pairwise(&opp, 1.0, &mblock)
    };

    let build = |weighted: bool, c: f32| -> Vec<Vec<f32>> {
        let mut cfv = vec![vec![0.0f32; nc]; nn];
        for &ci in &chance_nodes {
            if weighted {
                let w = wvec(ci);
                cfv[ci][aa] = c * w[aa];
                cfv[ci][t72] = -c * w[t72];
            } else {
                cfv[ci][aa] = c;
                cfv[ci][t72] = -c;
            }
        }
        cfv
    };
    let runs: [(&str, bool, f32); 3] = [
        ("terminals-only (C=0)", false, 0.0),
        ("UNWEIGHTED continuation (C=10) — the bug", false, 10.0),
        ("DERIVED-WEIGHTED continuation (C=10) — candidate", true, 10.0),
    ];
    for &node in &sample {
        let ch = tree.node_children(node);
        let labs: Vec<u8> = ch.iter().map(|&k| tree.nodes[k as usize].action_label).collect();
        println!("── node {node} labels {labs:?} (traverser {trav}) ──");
        for (label, weighted, c) in runs {
            let mut cfv = build(weighted, c);
            solver.bottom_up_preflop_for_traverser(&tree, trav, &chance_nodes, &reach, &term_fn, &mut cfv, &params);
            let row: Vec<String> = ch.iter().map(|&k| format!("{:+.3e}", cfv[k as usize][aa])).collect();
            println!("   AA  {label:<48} [{}]", row.join(", "));
        }
        println!();
    }
    println!("(SUCCESS = derived-weighted continuation lands at the SAME scale as terminals-only at");
    println!(" EACH node — not a uniform ±C — so postflop payoff is comparable to investment, not ×1e5 over it.)");
}
