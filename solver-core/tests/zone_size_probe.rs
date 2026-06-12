//! Probe: per-zone node counts for the v1 live-6 seam cells — sizes
//! the zone-local mega-buffer layout would allocate (the SIGSEGV
//! unlock math). One river walk per outcome runs concurrently, so the
//! dominant term is n_river_walks × river_zone_size.

use solver_core::card::{card_from_str, Card};
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::Zone;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

#[test]
#[ignore = "measurement probe"]
fn zone_sizes_live6() {
    let spec = production_game_v1();
    let flop_bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    for &(live, commit, pot, label) in
        &[(6u8, 2i32, 12i32, "6-way limp"), (6, 7, 42, "6-way raised"), (5, 2, 10, "5-way limp")]
    {
        let cfg = spec.flop_seam_config(live, commit, pot, flop_bets.clone());
        let tree = build_tree(&cfg).expect("tree");
        let flop: [Card; 3] = [
            card_from_str("2h").unwrap(),
            card_from_str("7d").unwrap(),
            card_from_str("Ks").unwrap(),
        ];
        let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
        let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
        let turn_cards: Vec<u8> = [6usize, 18, 30, 42].iter().map(|&p| deck[p]).collect();
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
        for &tc in &turn_cards {
            let rdeck: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
            river_decks[tc as usize] =
                [8usize, 20, 32, 44].iter().map(|&p| rdeck[p]).collect();
        }
        let table = FlopChanceTable::build_full_nh_sampled(flop, live, &turn_cards, &river_decks);
        let game = FlopStartGame::new(table);
        let bk = FlopBucketing::identity(game.table());
        let solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
        let layout = solver.gpu_layout(&bk);
        let nn = tree.nodes.len();
        let (mut f, mut t, mut r) = (0usize, 0usize, 0usize);
        for idx in 0..nn {
            match layout.zone_of(idx) {
                Zone::Flop => f += 1,
                Zone::Turn => t += 1,
                Zone::River => r += 1,
                Zone::Preflop => {}
            }
        }
        let np = live as usize;
        let nh = game.table().num_valid;
        let n_walks_river = 16usize;
        let n_walks_turn = 4usize;
        let cur = 21 * nn * np * nh;
        let zl = (f + n_walks_turn * t + n_walks_river * r) * np * nh;
        eprintln!(
            "{label}: nn {nn} | flop {f} turn {t} river {r} | nh {nh} | reach floats: \
             current {cur} ({:.1} GB) -> zone-local {zl} ({:.2} GB) | u32-ok: {}",
            cur as f64 * 4.0 / 1e9,
            zl as f64 * 4.0 / 1e9,
            zl <= u32::MAX as usize
        );
    }
}
