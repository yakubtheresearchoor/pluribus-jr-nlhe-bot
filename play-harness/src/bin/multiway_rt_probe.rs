//! MULTIWAY REAL-TIME TURN/RIVER LATENCY PROBE: can the exact HU turn/river search
//! generalize to live-3/4/5 to fill the blueprint's multiway runout hole? The
//! machinery already supports np (compute_turn/river_start take a player count,
//! side_pot_showdown is multiway, VectorCfr is np-agnostic). The only question is
//! cost — the multiway showdown is the same wall that made the blueprint 1×1. This
//! times a RIVER re-solve (board known, no chance) and a nested TURN re-solve
//! (check-only river continuation) per live count, so we can see which fit budget.
//!
//! Run: cargo run --release -p play-harness --bin multiway_rt_probe

use std::time::Instant;

use play_harness::live2_bank::live2_bet_menu;
use solver_core::card::{card_from_str, Card};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{production_game_v1, BetSizeOptions, BoardState};
use solver_core::tree::builder::{build_tree, build_tree_with_bet_override};

fn time_solve(state: BoardState, live: u8, commit: i32, pot: i32, board: &[u8], iters: u32) -> (usize, usize, u128) {
    let spec = production_game_v1();
    let ranges = vec![vec![1.0f32; 1326]; live as usize];
    let board_c: Vec<Card> = board.to_vec();
    let table = match state {
        BoardState::Turn => ChanceTable::compute_turn_start(&board_c, &ranges, live),
        BoardState::River => ChanceTable::compute_river_start(&board_c, &ranges, live),
        _ => unreachable!(),
    };
    let nh = table.num_valid;
    let game = TurnStartGame::new(table);
    let cfg = spec.street_seam_config(state, live, commit, pot, live2_bet_menu());
    // Turn uses the nested check-only river continuation (as the HU path does).
    let tree = if state == BoardState::Turn {
        let ck = BetSizeOptions { bet: vec![], raise: vec![] };
        build_tree_with_bet_override(&cfg, &[(BoardState::River, ck)]).unwrap()
    } else {
        build_tree(&cfg).unwrap()
    };
    let mut cfr = VectorCfr::new(&tree, vec![nh; live as usize]);
    let t0 = Instant::now();
    cfr.run(&tree, &game, iters);
    (nh, tree.num_nodes(), t0.elapsed().as_millis())
}

fn main() {
    let iters: u32 = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let card = |s: &str| card_from_str(s).unwrap();
    let flop = [card("Ks"), card("9d"), card("4c")];
    let turn_board: Vec<u8> = vec![flop[0], flop[1], flop[2], card("2h")];
    let river_board: Vec<u8> = vec![flop[0], flop[1], flop[2], card("2h"), card("7s")];

    let only: Option<u8> = std::env::var("LIVE").ok().and_then(|s| s.parse().ok());
    let streets = std::env::var("STREET").unwrap_or_else(|_| "both".into());
    eprintln!("multiway real-time turn/river latency, {iters} iters, mid-SPR");
    for live in [2u8, 3, 4, 5] {
        if let Some(o) = only {
            if o != live {
                continue;
            }
        }
        let commit = 20;
        let pot = 20 * live as i32 + 40; // covers live commits + dead money
        let fit = |ms: u128| if ms < 14_000 { "✓budget" } else { "✗OVER" };
        if streets != "turn" {
            let (nh, nd, ms) = time_solve(BoardState::River, live, commit, pot, &river_board, iters);
            eprintln!("live-{live} RIVER  {ms}ms {}  (nh~{nh}, {nd} nodes)", fit(ms));
        }
        if streets != "river" {
            let (nh, nd, ms) = time_solve(BoardState::Turn, live, commit, pot, &turn_board, iters);
            eprintln!("live-{live} TURN   {ms}ms {}  (nh~{nh}, {nd} nodes)", fit(ms));
        }
    }
}
