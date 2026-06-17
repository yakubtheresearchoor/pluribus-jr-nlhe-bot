//! FLOP LIVE-COUNT CENSUS (2026-06-17): sanity-check the joint_solve finding
//! that live-4/5 are NEVER reached (count = 0). Fork: either (1) REAL — the
//! cap-3 preflop tree structurally never produces a 4/5-way flop (then the whole
//! multiway-fidelity investigation priced families that don't occur), or (2) a
//! classification BUG (live-4/5 nodes exist and are silently unsolved → the bot
//! would play those real spots with garbage). Walk the SAME tree joint_solve
//! uses and ask: how many flop-entry chance nodes of each live count EXIST, and
//! what's their reach mass.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::postflop_oracle::SeamCell;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree_preflop_only;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

fn main() {
    let spec = production_game_v1();
    let np = 6usize;

    // EXACT same tree as joint_solve (na=16, cap-3).
    let tree: FlatTree = {
        let mrc = MAX_NA_PREFLOP.saturating_sub(2);
        let mut cfg = spec.preflop_tree_config(BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect(),
        });
        cfg.max_bets_per_street = BetCap::all(3);
        build_tree_preflop_only(&cfg).expect("cap-3 preflop tree")
    };

    let solver = PreflopVectorCfr::new(&tree);
    let chance_nodes = solver.preflop_chance_node_indices(&tree);
    let nc = NUM_PREFLOP_CLASSES;

    eprintln!(
        "tree: {} nodes, {} flop-entry chance nodes\n",
        tree.num_nodes(),
        chance_nodes.len()
    );

    // (A) EXISTENCE: count flop-entry chance nodes by live count, two ways
    //     (folded_mask popcount AND SeamCell::at_chance_node) — cross-checked.
    let mut count_mask = [0u64; 8];
    let mut count_cell = [0u64; 8];
    let mut mismatch = 0u64;
    for &c in &chance_nodes {
        let mask = tree.get_folded_mask(c);
        let live_mask = np - mask.count_ones() as usize;
        let cell = SeamCell::at_chance_node(&tree, c, np);
        count_mask[live_mask] += 1;
        count_cell[(cell.live as usize).min(7)] += 1;
        if cell.live as usize != live_mask {
            mismatch += 1;
        }
    }

    // (B) REACH (uniform strategy): total reach mass at each live count — are
    //     they even reachable? (uniform play reaches every structurally-live node)
    let reach = solver.compute_preflop_reach(&tree, None);
    let mut reach_by_live = [0.0f64; 8];
    for &c in &chance_nodes {
        let live = (np - tree.get_folded_mask(c).count_ones() as usize).min(7);
        let m: f64 = (0..nc).map(|cl| reach[0][c * nc + cl] as f64).sum();
        reach_by_live[live] += m;
    }

    eprintln!("{:<8} {:>14} {:>14} {:>16}", "live", "nodes(mask)", "nodes(cell)", "reach mass(P0,unif)");
    for live in 0..8 {
        if count_mask[live] > 0 || count_cell[live] > 0 || reach_by_live[live] > 0.0 {
            eprintln!(
                "live-{live:<3} {:>14} {:>14} {:>16.4e}",
                count_mask[live], count_cell[live], reach_by_live[live]
            );
        }
    }
    eprintln!("\nmask-vs-cell live-count mismatches: {mismatch}");

    // (C) CORRECTED per-path solve breakdown, from the tree (no slow walk).
    // Distinct SPR bucket-keys per path × 1755 canonical flops = unique solves
    // (validated: live-2 = 25 keys × 1755 = 43,875, matching joint_solve).
    use std::collections::HashSet;
    let stack = spec.stack;
    let mut k_exact: HashSet<(u8, i64)> = HashSet::new();
    let mut k_roll: HashSet<(u8, i64)> = HashSet::new();
    let mut k_buck: [HashSet<(u8, i64)>; 8] = Default::default();
    for &c in &chance_nodes {
        let cell = SeamCell::at_chance_node(&tree, c, np);
        let key = cell.bucket_key(stack);
        if cell.live == 2 {
            k_exact.insert(key);
        } else if key.1 == i64::MIN || cell.live == 6 {
            k_roll.insert(key);
        } else {
            k_buck[cell.live as usize].insert(key);
        }
    }
    let flops = 1755u64;
    // (path label, distinct keys, s/subgame)
    let rows: Vec<(String, u64, f64)> = vec![
        ("live-2 exact".into(), k_exact.len() as u64, 11.4),
        ("live-3 B15".into(), k_buck[3].len() as u64, 0.15),
        ("live-4 B15".into(), k_buck[4].len() as u64, 1.46),
        ("live-5 B8".into(), k_buck[5].len() as u64, 4.70),
        ("live-6/allin".into(), k_roll.len() as u64, 0.05),
    ];
    eprintln!("\n=== CORRECTED per-iteration breakdown (keys × 1755 flops × s/subgame) ===");
    eprintln!("{:<14} {:>6} {:>11} {:>11} {:>14}", "path", "keys", "solves", "s/subgame", "wall h (serial)");
    let (mut tot_s, mut tot_n) = (0.0f64, 0u64);
    for (label, keys, t) in &rows {
        let solves = keys * flops;
        let wall = solves as f64 * t;
        tot_s += wall;
        tot_n += solves;
        eprintln!("{:<14} {:>6} {:>11} {:>11.2} {:>14.2}", label, keys, solves, t, wall / 3600.0);
    }
    eprintln!("{:<14} {:>6} {:>11} {:>11} {:>14.2}", "TOTAL", "", tot_n, "", tot_s / 3600.0);
    eprintln!("(live-6/allin s/subgame is a rollout estimate; live-2 11.4s is the exact-HU cost)");
    let exist45 = count_mask[4] + count_mask[5];
    if exist45 == 0 {
        eprintln!("VERDICT: NO live-4/5 flop-entry chance nodes EXIST in the tree → joint_solve's");
        eprintln!("         zero is REAL/structural. The cap-3 tree never produces a 4/5-way flop.");
    } else if reach_by_live[4] + reach_by_live[5] < 1e-9 {
        eprintln!("VERDICT: live-4/5 nodes EXIST ({exist45}) but uniform reach ~0 → unreachable (odd).");
    } else {
        eprintln!("VERDICT: live-4/5 nodes EXIST ({exist45}) AND are reachable → REAL multiway spots");
        eprintln!("         (the multiway-fidelity work priced families that genuinely occur).");
        eprintln!("         joint_solve's live-4/5=0 was an INCOMPLETE walk (killed during live-3),");
        eprintln!("         NOT structural and NOT a counter bug — see CORRECTED breakdown above.");
    }
}
