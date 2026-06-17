//! CFV SCALE PROBE (2026-06-15): confirm the seam chance-node CFV magnitude
//! scales with LIVE-PLAYER COUNT (~nh^num_opp), not as a per-hand EV — the
//! suspected cause of the preflop's never-raise collapse (call/limp keeps more
//! players in → numerically huge CFV → always wins the magnitude contest).
//! Prints AA's per-class chance-node CFV at one chance node of each live count.

use play_harness::preflop_oracle::banked_read_source;
use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::card_from_str;
use solver_core::solver::postflop_oracle::{BucketKeyedOracle, SeamCell};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::build_tree_preflop_only;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP, NODE_TYPE_CHANCE};

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

fn load_cells(root: &str) -> Vec<(u8, i32, i32, usize)> {
    std::fs::read_to_string(format!("{root}/cells.txt")).unwrap().lines()
        .filter(|l| l.starts_with("CELL live=")).map(|l| {
            let g = |k: &str| -> i64 { l[l.find(&format!("{k}=")).unwrap()+k.len()+1..].split_whitespace().next().unwrap().parse().unwrap() };
            (g("live") as u8, g("commit") as i32, g("pot") as i32, g("b") as usize)
        }).collect()
}

#[test]
fn cfv_scale_probe() {
    let spec = production_game_v1();
    let bp_root = format!("{}/../blueprint_out_v1", env!("CARGO_MANIFEST_DIR"));
    let tree = cap3();
    let table = PreflopChanceTable::new(6, vec![vec![1.0f32/NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6]);
    let canon = table.canonical_flops.clone();
    let cells = load_cells(&bp_root);
    let source = banked_read_source(bp_root.clone(), cells, canon, spec.stack);
    let mut oracle = BucketKeyedOracle::new(spec.stack, 6, 0, source);
    let mut solver = solver_core::solver::preflop_cfr::PreflopVectorCfr::new(&tree);
    solver.compute_preflop_strategy(&tree);
    let reach = solver.compute_preflop_reach(&tree, None);
    let aa = PreflopClass::from_combo(card_from_str("Ac").unwrap(), card_from_str("Ad").unwrap()).index();
    let trash = PreflopClass::from_combo(card_from_str("7c").unwrap(), card_from_str("2d").unwrap()).index();
    let mid = PreflopClass::from_combo(card_from_str("9c").unwrap(), card_from_str("8c").unwrap()).index();

    // one representative non-all-in chance node per live count
    let mut seen: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    for idx in 0..tree.num_nodes() {
        let n = &tree.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.board_state != BoardState::Flop as u8 { continue; }
        let cell = SeamCell::at_chance_node(&tree, idx, 6);
        if spec.stack - cell.commit <= 0 { continue; }
        seen.entry(cell.live).or_insert(idx);
    }
    let mut lives: Vec<u8> = seen.keys().copied().collect();
    lives.sort();
    eprintln!("\n=== AA chance-node CFV (v_class[AA]) by live count ===");
    eprintln!("(if these scale ~nh^num_opp instead of being comparable EVs, that's the never-raise bias)");
    for live in lives {
        let idx = seen[&live];
        let cell = SeamCell::at_chance_node(&tree, idx, 6);
        let mask = tree.get_folded_mask(idx);
        let v = solver.compute_chance_node_cfv_with_expansion_for_cell(idx, 0, &reach, &table, &mut oracle, cell, mask);
        eprintln!("  live-{live} (pot {}): AA={:+.3e}  98s={:+.3e}  72o={:+.3e}", cell.pot, v[aa], v[mid], v[trash]);
    }
}
