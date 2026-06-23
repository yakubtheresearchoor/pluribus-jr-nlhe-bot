//! LIVE-2 TURN/RIVER REAL-TIME SOLVE PROBE: measure the per-decision latency of an
//! EXACT heads-up turn-rooted and river-rooted solve (rich M2 menu), to decide
//! whether live-2 turn/river decisions can be searched live (vs the 1×1 bank that
//! only covers the flop). River root = no chance (board complete); turn root =
//! river chance (47) + river betting. Both use the generic VectorCfr + the (reused)
//! TurnStartGame GameSpec over a ChanceTable.
//!
//! Run: cargo run --release -p play-harness --bin live2_streets_probe

use std::time::Instant;

use play_harness::live2_bank::live2_bet_menu;
use solver_core::card::{card_from_str, Card};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::{build_tree, build_tree_with_bet_override};

fn main() {
    let spec = production_game_v1();
    let iters: u32 = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let ranges = vec![vec![1.0f32; 1326]; 2]; // uniform full range, both players
    let commit: i32 = std::env::var("COMMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let pot: i32 = std::env::var("POT").ok().and_then(|s| s.parse().ok()).unwrap_or(60);

    let cards = |ss: &[&str]| -> Vec<Card> { ss.iter().map(|s| card_from_str(s).unwrap()).collect() };

    // ── RIVER ROOT: 5-card board, no chance, betting → exact showdown ──
    {
        let board = cards(&["Ks", "9d", "4c", "2h", "7s"]);
        let t_tab = Instant::now();
        let table = ChanceTable::compute_river_start(&board, &ranges, 2);
        let tab_ms = t_tab.elapsed().as_millis();
        let nh = table.num_valid;
        let game = TurnStartGame::new(table);
        let tree = build_tree(&spec.street_seam_config(BoardState::River, 2, commit, pot, live2_bet_menu())).unwrap();
        let t = Instant::now();
        let mut cfr = VectorCfr::new(&tree, vec![nh; 2]);
        cfr.run(&tree, &game, iters);
        let run_ms = t.elapsed().as_millis();
        let na = tree.nodes[0].num_children as usize;
        let strat = cfr.get_average_strategy(0, na, nh);
        let probs: Vec<f32> = (0..na).map(|a| strat[a][0]).collect(); // hand 0's dist
        let sum: f32 = probs.iter().sum();
        println!(
            "RIVER  nh={nh} nodes={} na={na} | table {}ms + solve {}ms ({iters} it) | hand0 Σp={:.3} {:?}",
            tree.num_nodes(), tab_ms, run_ms, sum,
            probs.iter().map(|p| (p * 100.0).round() / 100.0).collect::<Vec<_>>()
        );
    }

    // ── TURN ROOT: FULL river betting vs NESTED (coarse river continuation) ──
    {
        let board = cards(&["Ks", "9d", "4c", "2h"]);
        let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
        let nh = table.num_valid;
        let game = TurnStartGame::new(table);
        let cfg = spec.street_seam_config(BoardState::Turn, 2, commit, pot, live2_bet_menu());

        let full = build_tree(&cfg).unwrap();
        let one_bet = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
        let nested_1b = build_tree_with_bet_override(&cfg, &[(BoardState::River, one_bet)]).unwrap();
        let check_only = BetSizeOptions { bet: vec![], raise: vec![] };
        let nested_ck = build_tree_with_bet_override(&cfg, &[(BoardState::River, check_only)]).unwrap();

        for (label, tree) in [("FULL   ", &full), ("NEST-1b", &nested_1b), ("NEST-ck", &nested_ck)] {
            let t = Instant::now();
            let mut cfr = VectorCfr::new(tree, vec![nh; 2]);
            cfr.run(tree, &game, iters);
            let run_ms = t.elapsed().as_millis();
            let na = tree.nodes[0].num_children as usize;
            let strat = cfr.get_average_strategy(0, na, nh);
            let probs: Vec<f32> = (0..na).map(|a| strat[a][0]).collect();
            println!(
                "TURN {label} nh={nh} nodes={} na={na} | solve {}ms ({iters} it) | hand0 {:?}",
                tree.num_nodes(), run_ms,
                probs.iter().map(|p| (p * 100.0).round() / 100.0).collect::<Vec<_>>()
            );
        }
    }
}
