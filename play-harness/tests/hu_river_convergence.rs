//! Is the deep-SPR HU river nuts-check a real trap or UNDER-CONVERGENCE? The live
//! server (600it) returns ~99% check on quads at SPR~9.5 (commit=10,pot=20) but
//! value-bets at shallow SPR. The river is single-street (exact showdown, no
//! continuation) so we can over-solve it cheaply and watch the nuts bet vs iters.
//! If it climbs toward a value bet, the 600-it budget is the culprit (like the
//! turn); if it stays checked, it's a genuine deep-stack check-raise trap.
//!
//! Ignored (sweeps high iters). Run on demand: EXACT_ITERS unused; sweeps inline.

use solver_core::card::{card_from_str, Card};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BoardState, production_game_v1};
use solver_core::tree::builder::build_tree;

use play_harness::live2_bank::live2_bet_menu;

fn card(s: &str) -> u8 { card_from_str(s).unwrap() as u8 }

#[test]
#[ignore = "river convergence sweep (~mins). Run on demand."]
fn hu_river_nuts_convergence_vs_iters() {
    // Quad aces on the river; deep-SPR spot where the live server checks ~99%.
    let board = [card("As"), card("Ah"), card("7c"), card("2d"), card("9s")];
    let (commit, pot) = (10i32, 20i32);
    let spec = production_game_v1();
    let cfg = spec.street_seam_config(BoardState::River, 2, commit, pot, live2_bet_menu());

    let board_c: Vec<Card> = board.iter().map(|&c| c as Card).collect();
    let ranges = vec![vec![1.0f32; 1326]; 2];
    let table = ChanceTable::compute_river_start(&board_c, &ranges, 2);
    let nh = table.num_valid;
    let hand_cards = table.hand_cards.clone();
    let game = TurnStartGame::new(table);
    let tree = build_tree(&cfg).expect("river tree"); // full river, exact showdown

    let (a, b) = (card("Ac").min(card("Ad")), card("Ac").max(card("Ad")));
    let h = (0..nh).find(|&i| hand_cards[i * 2] == a && hand_cards[i * 2 + 1] == b).expect("quads");
    let root = (0..tree.num_nodes())
        .find(|&n| tree.nodes[n].is_player() && tree.nodes[n].board_state == BoardState::River as u8)
        .expect("river player node");

    for &iters in &[300u32, 600, 1500, 4000, 10000] {
        let mut cfr = VectorCfr::new(&tree, vec![nh; 2]);
        cfr.run(&tree, &game, iters);
        let na = tree.nodes[root].num_children as usize;
        let s = cfr.get_average_strategy(root, na, nh);
        let ch = tree.node_children(root);
        let bet: f32 = (0..na)
            .filter(|&a| { let l = tree.nodes[ch[a] as usize].action_label; l != 0 && l != 1 })
            .map(|a| s[a][h]).sum();
        eprintln!("river quads (commit={commit},pot={pot}) it={iters:5}: bet={bet:.4}");
    }
}
