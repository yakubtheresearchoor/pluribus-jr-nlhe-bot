//! LIVE-2 BANK STORAGE (2026-06-15): size the on-disk blob for a full-nh,
//! 1×1-runout live-2 strategy (what we'd bank if live-2 goes in the BP). The
//! HU solver (FlopStartVectorCfr) holds the averaged strategy in
//! cum_strategy_{flop,turn,river}; on a 1×1 table (n_turn=1, n_river=1) those
//! buffers ARE the bankable blob. Project to the full family (25 SPR buckets ×
//! 1755 flops). No solve needed — buffer sizes are fixed at construction.

use solver_core::card::{card_from_str, Card};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

#[test]
#[ignore = "live-2 bank storage probe; --ignored --nocapture --release"]
fn live2_bank_storage() {
    let spec = production_game_v1();
    // The widest seam tree (smallest commit / biggest SPR) is the upper bound
    // on infoset count → upper-bound the per-flop blob.
    for (commit, pot) in [(2i32, 12i32), (10, 32), (40, 120)] {
        let tree = build_tree(&spec.flop_seam_config(
            2, commit, pot,
            BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        )).unwrap();
        let board: [Card; 3] = ["Ks", "7d", "2c"].map(|s| card_from_str(s).unwrap());
        let turn = card_from_str("9h").unwrap();
        let river = card_from_str("4s").unwrap();
        let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
        rd[turn as usize] = vec![river];
        let table = FlopChanceTable::build_full_nh_sampled(board, 2, &[turn], &rd);
        let nh = table.num_valid;
        let game = FlopStartGame::new(table);
        let s = FlopStartVectorCfr::new(&tree, game.table());

        let flop_len = s.cum_strategy_flop().len();
        let per_inf = (flop_len / s.flop_infosets().max(1)) as f64; // MAX_NA × nh
        // 1×1 ⇒ turn/river cum are 1 outcome each; their blob = infosets × per_inf.
        let turn_f = s.turn_infosets() as f64 * per_inf;
        let river_f = s.river_infosets() as f64 * per_inf;
        let total_f = flop_len as f64 + turn_f + river_f;
        let per_flop_bytes = total_f * 4.0;
        let family_gb = per_flop_bytes * 25.0 * 1755.0 / 1e9;
        eprintln!(
            "commit {commit:>3} pot {pot:>3} | {} nodes, nh {nh} | infosets f/t/r = {}/{}/{} \
             | per-flop blob {:.2} MB → family(25×1755) {:.1} GB",
            tree.num_nodes(), s.flop_infosets(), s.turn_infosets(), s.river_infosets(),
            per_flop_bytes / 1e6, family_gb,
        );
    }
}
