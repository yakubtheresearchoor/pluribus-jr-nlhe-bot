//! LIVE-COUNT CENSUS (2026-06-17): sanity-check the joint-solve finding that
//! live-4/5 subgames are NEVER solved (count = 0). Two opposite explanations:
//!  (1) REAL structural zero — the cap-3 preflop tree never has 4+ players
//!      reaching a flop together (betting narrows the field to 2-3), so live-4/5
//!      flop-entry chance nodes don't exist / are unreachable; OR
//!  (2) BUG — live-4/5 chance nodes exist and are reachable but were miscounted.
//!
//! Decisive structural test: enumerate the preflop tree's flop-entry chance
//! nodes, bucket by live-player count (= np − folded), and report node count,
//! distinct SPR bucket-keys, and reach mass (under the initial ~uniform
//! strategy) per live count. If live-4/5 have 0 nodes ⇒ structural (real); if
//! they have nodes with nonzero reach but the joint solve counted 0 ⇒ bug.

use std::collections::{HashMap, HashSet};

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::postflop_oracle::SeamCell;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree_preflop_only;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

fn cap3_preflop_tree() -> FlatTree {
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
    let spec = production_game_v1();
    let tree = cap3_preflop_tree();
    let np = 6usize;
    let nc = NUM_PREFLOP_CLASSES;
    let solver = PreflopVectorCfr::new(&tree);
    let chance_nodes = solver.preflop_chance_node_indices(&tree);

    // reach under the INITIAL (regret-zero ⇒ uniform) strategy — every
    // structurally-present node has nonzero reach here, so reach≈0 means the
    // line is structurally near-impossible, not strategy-pruned.
    let reach = solver.compute_preflop_reach(&tree, None);

    let mut node_by_live = [0u64; 8];
    let mut keys_by_live: HashMap<u8, HashSet<(u8, i64)>> = HashMap::new();
    let mut reach_by_live = [0.0f64; 8];

    for &c in &chance_nodes {
        let cell = SeamCell::at_chance_node(&tree, c, np);
        let live = cell.live as usize;
        node_by_live[live] += 1;
        keys_by_live.entry(cell.live).or_default().insert(cell.bucket_key(spec.stack));

        // node reach proxy: product over LIVE seats of (their reach summed over
        // classes). Zero if any live seat can't arrive here ⇒ node unreachable.
        let mask = tree.get_folded_mask(c);
        let base = c * nc;
        let mut prod = 1.0f64;
        for p in 0..np {
            if (mask >> p) & 1 == 0 {
                let s: f64 = (0..nc).map(|cl| reach[p][base + cl] as f64).sum();
                prod *= s;
            }
        }
        reach_by_live[live] += prod;
    }

    println!("=== flop-entry chance-node census (na=16 cap-3, np=6) ===");
    println!("{} flop-entry chance nodes total\n", chance_nodes.len());
    println!("{:<8} {:>10} {:>12} {:>16}", "live", "nodes", "SPR-keys", "reach-mass(unif)");
    for live in 2..=6usize {
        let nkeys = keys_by_live.get(&(live as u8)).map(|s| s.len()).unwrap_or(0);
        println!(
            "{:<8} {:>10} {:>12} {:>16.4e}",
            format!("live-{live}"),
            node_by_live[live],
            nkeys,
            reach_by_live[live]
        );
    }
    println!();
    let l45_nodes = node_by_live[4] + node_by_live[5];
    let l45_reach = reach_by_live[4] + reach_by_live[5];
    if l45_nodes == 0 {
        println!("VERDICT: ZERO live-4/5 flop-entry nodes EXIST → REAL structural zero.");
        println!("         The cap-3 tree never sends 4+ players to a flop together. The");
        println!("         multiway-fidelity work on live-4/5 priced families that don't occur.");
    } else if l45_reach < 1e-9 {
        println!("VERDICT: live-4/5 nodes EXIST ({l45_nodes}) but reach≈0 ({l45_reach:.2e}) →");
        println!("         structurally unreachable in practice (real zero, not a bug).");
    } else {
        println!("VERDICT: live-4/5 nodes EXIST ({l45_nodes}) WITH nonzero reach ({l45_reach:.2e})");
        println!("         but the joint solve counted 0 → CLASSIFICATION/ROUTING BUG. Investigate.");
    }
}
