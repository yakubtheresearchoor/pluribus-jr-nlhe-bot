//! P0 probe: factored vs EXACT showdown on a dry river (K7249), np=3, uniform
//! ranges — 62o's strategy at the root decides where the bug lives.
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetCap, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::build_tree;

#[test]
fn factored_vs_exact_river_np3() {
    let board: Vec<solver_core::card::Card> = vec![47, 20, 3, 9, 30]; // K 7 2 4 9
    let np = 3u8;
    let ranges = vec![vec![1.0f32; 1326]; np as usize];
    let spec = solver_core::tree::action::production_game_v1();
    for factored in [true] {
        let table = ChanceTable::compute_river_start(&board, &ranges, np);
        let nh = table.num_valid;
        let hc = table.hand_cards.clone();
        let game = if factored { TurnStartGame::new(table).with_factored() } else { TurnStartGame::new(table) };
        let mut cfg = spec.street_seam_config(BoardState::River, np, 6, 18,
            BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![BetSize::PotRelative(1.0)] });
        cfg.max_bets_per_street = BetCap::all(2);
        let tree = build_tree(&cfg).expect("tree");
        let mut cfr = VectorCfr::new(&tree, vec![nh; np as usize]);
        cfr.run(&tree, &game, 120);
        let rank = |c: u8| c >> 2;
        // TRUE AIR: 3h + 5d — no pair with K7249, 5-high. (62o pairs the board 2!)
        let trash = (0..nh).find(|&h| {
            let (a, b) = (hc[h*2], hc[h*2+1]);
            (rank(a) == 1 && rank(b) == 3) || (rank(a) == 3 && rank(b) == 1)
        }).unwrap();
        let nuts = (0..nh).find(|&h| rank(hc[h*2]) == 11 && rank(hc[h*2+1]) == 11).unwrap(); // KK = top set
        let na = tree.nodes[0].num_children as usize;
        let s = cfr.get_average_strategy(0, na, nh);
        eprintln!("factored={factored}: root na={na}  trash({},{}) dist={:?}  KK dist={:?}",
            hc[trash*2], hc[trash*2+1],
            (0..na).map(|a| (s[a][trash]*100.0).round()/100.0).collect::<Vec<_>>(),
            (0..na).map(|a| (s[a][nuts]*100.0).round()/100.0).collect::<Vec<_>>());
    }
}
