//! VALIDATE `root_cfv_from_avg` (the read-banked oracle's CFV extractor)
//! against `run_all_root_cfv` (re-solve) at RESEARCH nh, both on the bucketed
//! collapsed terminal. If the CFV computed from a CONVERGED cum-average
//! matches the re-solve's root CFV, then reading a banked .bp's strategy and
//! extracting its root CFV is equivalent to re-solving the cell — the basis
//! for the preflop oracle reading the fill instead of redoing ~25.8 h of it.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn clean_table(np: u8, nh: usize) -> FlopChanceTable {
    let board: [Card; 3] = ["Ks", "7d", "2c"].map(|s| card_from_str(s).unwrap());
    let bmask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if bmask & ((1u64 << c1) | (1u64 << c2)) == 0 { all_valid.push(idx as u16); }
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..np as usize { for &hi in &chosen { ranges[p][hi as usize] = 1.0; } }
    let turn_cards: Vec<u8> = ["9h", "4s"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let river_strs: [&[&str]; 2] = [&["Qd", "3c"], &["Td", "Js"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] = river_strs[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(&board, &ranges, np, &chosen, &turn_cards, &river_decks)
}

fn seam_tree(live: u8, commit: i32, pot: i32) -> FlatTree {
    let spec = production_game_v1();
    build_tree(&spec.flop_seam_config(
        live, commit, pot,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
    )).expect("seam tree")
}

#[test]
#[ignore = "root_cfv_from_avg vs run_all_root_cfv at research nh; --ignored --nocapture --release"]
fn root_cfv_from_avg_matches_resolve() {
    const ITERS: u32 = 600;
    eprintln!("\n═══ root_cfv_from_avg (read-banked) vs run_all_root_cfv (re-solve), B=8 ═══");
    let mut worst = 0.0f64;
    for (live, commit, pot, nh) in [(3u8, 7i32, 25i32, 24usize), (4, 7, 29, 16)] {
        let tree = seam_tree(live, commit, pot);
        let table = clean_table(live, nh);
        let bk = FlopBucketing::quantile(&table, 8);
        let game = FlopStartGame::new(table);

        // READ-BANKED: solve to converge, then extract root CFV from the avg.
        let mut a = BucketedFlopCfr::new(&tree, game.table(), &bk);
        a.set_terminal_design(TerminalDesign::Design1Collapsed);
        a.run(&tree, &game, &bk, ITERS);
        let cfv_read = a.root_cfv_from_avg(&tree, &game, &bk);

        // RE-SOLVE: fresh, run_all_root_cfv.
        let mut b = BucketedFlopCfr::new(&tree, game.table(), &bk);
        b.set_terminal_design(TerminalDesign::Design1Collapsed);
        let cfv_resolve = b.run_all_root_cfv(&tree, &game, &bk, ITERS);

        for t in 0..live as usize {
            let n = cfv_read[t].len().min(cfv_resolve[t].len());
            let mad: f64 = (0..n)
                .map(|i| (cfv_read[t][i] as f64 - cfv_resolve[t][i] as f64).abs())
                .sum::<f64>() / n as f64;
            let scale = cfv_resolve[t][..n].iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6) as f64;
            let rel = mad / scale;
            worst = worst.max(rel);
            eprintln!("  live-{live} trav {t}: read vs resolve mean|Δ| {mad:.4} (rel {:.2}%, scale {scale:.3})", rel * 100.0);
        }
    }
    eprintln!("→ {} (worst rel {:.2}%)", if worst < 0.02 { "MATCH" } else { "DIVERGE" }, worst * 100.0);
    assert!(worst < 0.02, "root_cfv_from_avg diverges from re-solve by {:.2}% — extractor bug", worst * 100.0);
}
