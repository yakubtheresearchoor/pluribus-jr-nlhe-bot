//! STANDING bb-units gate (2026-06-12, user directive: "our units
//! should be in bb not chips, i was under the understanding that that
//! was enforced").
//!
//! The convention it enforces:
//!   - ALL integer money in TreeConfig / FlatTree is in chip units,
//!     1 bb = UNITS_PER_BB units (single authoritative constant).
//!   - starting_pot is DEAD money; live blinds go in
//!     initial_contributions (additive terminal math, pinned 242d4c6).
//!   - The tree's recorded amounts must round-trip through bb exactly
//!     for the production configs (no fractional-bb money anywhere).
//!
//! This is the gate the user assumed existed. It pins the ORACLE flop
//! game (the family every current blueprint is solved on) numerically
//! in bb, so any config drift or sizing regression (e.g., the pot-blind
//! 1-unit bet bug) fails loudly here.

use solver_core::tree::action::{
    BetSize, BetSizeOptions, BoardState, TreeConfig, UNITS_PER_BB,
};
use solver_core::tree::builder::build_tree;

fn oracle_cfg() -> TreeConfig {
    TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 12,
        starting_stacks: vec![94; 6],
        initial_contributions: vec![0; 6],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    }
}

/// PRODUCTION GAME CLASS v1 — numeric pins in bb (the behavioral pins
/// run through clean-rules in the harness `production_game_gate`).
#[test]
fn production_game_v1_bb_pins() {
    use solver_core::tree::action::production_game_v1;
    let g = production_game_v1();
    assert_eq!(g.num_players, 6);
    assert_eq!(g.ante, 0, "v1 is a no-ante game");
    assert_eq!(g.sb * 2, g.bb, "SB = 0.5 bb");
    assert_eq!(g.bb, UNITS_PER_BB, "BB = 1 bb by definition");
    assert_eq!(g.stack / UNITS_PER_BB, 100, "100 bb starting stacks");
    assert_eq!(g.stack % UNITS_PER_BB, 0);
    assert!((g.rake_rate - 0.05).abs() < 1e-12, "5% rake");
    assert_eq!(g.rake_cap / UNITS_PER_BB, 10, "10 bb rake cap");
    assert_eq!(g.rake_cap % UNITS_PER_BB, 0);
    assert!(g.no_flop_no_drop, "no flop, no drop");
    // Rake-rate units sanity: the cap binds at pot = cap/rate = 400
    // units = 200 bb — exactly the two-stacks-all-in pot. Any pot
    // below all-in is raked strictly by percentage.
    assert_eq!((g.rake_cap as f64 / g.rake_rate) as i32, 2 * g.stack);
}

#[test]
fn bb_units_gate() {
    assert_eq!(UNITS_PER_BB, 2, "conversion constant is pinned; changing it re-prices every report");

    let cfg = oracle_cfg();

    // 1. Config money in bb (exact, no remainders).
    assert_eq!(cfg.starting_pot % UNITS_PER_BB, 0);
    assert_eq!(cfg.starting_pot / UNITS_PER_BB, 6, "oracle dead pot = 6 bb (6 antes of 1 bb)");
    for &s in &cfg.starting_stacks {
        assert_eq!(s % UNITS_PER_BB, 0);
        assert_eq!(s / UNITS_PER_BB, 47, "oracle stacks = 47 bb behind");
    }

    // 2. The built tree's recorded money is in the same units and on
    //    the bb grid everywhere (every contribution at every node).
    let tree = build_tree(&cfg).expect("oracle tree");
    for idx in 0..tree.nodes.len() {
        for p in 0..6u8 {
            let c = tree.get_contribution(idx, p);
            assert!(
                c % UNITS_PER_BB == 0,
                "node {idx} p{p}: contribution {c} units is fractional in bb \
                 (oracle game money moves in whole bb)"
            );
            assert!(c >= 0 && c <= 94, "node {idx} p{p}: contribution {c} outside [0, stack]");
        }
    }

    // 3. Sizing regression pin: the root 1.0x-pot bet records exactly
    //    6 bb (12 units) — the pot-blind bug recorded 1 unit here.
    let bet = *tree
        .node_children(0)
        .iter()
        .find(|&&c| tree.nodes[c as usize].action_label == 3)
        .expect("root bet child") as usize;
    let bettor = tree.nodes[0].player_id;
    assert_eq!(
        tree.get_contribution(bet, bettor) / UNITS_PER_BB,
        6,
        "flop 1.0x-pot bet must be 6 bb"
    );
}
