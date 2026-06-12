//! V1 CELL PRICING (measurement instrument, 2026-06-12): CPU s/iter
//! for representative v1 SEAM CELLS at production fidelity (full-nh
//! table, quantile B=8, 4×4 sampled runouts, Design1Collapsed) — the
//! re-pricing the blueprint relaunch decision needs. Cells chosen from
//! the v1_seam_census: the common raised-pot families per live count
//! plus the worst-case limp cells (live 5/6) that dominate node counts.
//!
//! All-in cells (commit = 200) are NOT priced: no postflop decisions —
//! the oracle for those is a pure equity rollout, not a CFR solve.

use solver_core::card::{card_from_str, Card};
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use std::time::Instant;

fn quantile_maps(
    table: &FlopChanceTable,
    nb: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8]| -> Vec<u16> {
        let alive: Vec<usize> = pl_idx[..nh]
            .iter()
            .map(|&i| i as usize)
            .filter(|&h| !conflicts(h, dead))
            .collect();
        let n = alive.len();
        assert!(n >= nb);
        let mut map = vec![NO_BUCKET; nh];
        for (pos, &h) in alive.iter().enumerate() {
            map[h] = ((pos * nb) / n) as u16;
        }
        map
    };
    let (_, _, _, base_pi, _) = table.sorted_opp_arrays_base();
    let flop_map = map_for(&base_pi, &[]);
    let mut turn_maps = Vec::new();
    let mut river_maps = Vec::new();
    for &tc_card in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
        turn_maps.push(map_for(pi, &[tc_card]));
        let mut rms = Vec::new();
        for &rc_card in &table.river_decks[tc_card as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
            rms.push(map_for(pi, &[tc_card, rc_card]));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
}

/// Production-nh table at 4×4 runouts (the G4/G5 ladder's deck-position
/// convention) for `np` live players.
fn build_table(np: u8) -> FlopChanceTable {
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
        river_decks[tc as usize] = [8usize, 20, 32, 44].iter().map(|&p| rdeck[p]).collect();
    }
    FlopChanceTable::build_full_nh_sampled(flop, np, &turn_cards, &river_decks)
}

#[test]
#[ignore = "v1 cell pricing (CPU, minutes); run with --ignored --nocapture --release"]
fn v1_cell_pricing_cpu() {
    let spec = production_game_v1();
    let flop_bets =
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    const NB: usize = 8;
    const ITERS: u32 = 5;

    // (live, commit, pot, label)
    let cells: &[(u8, i32, i32, &str)] = &[
        (2, 7, 15, "HU single-raised (open 3.5bb, sb dead)"),
        (3, 7, 29, "3-way single-raised + dead"),
        (2, 24, 49, "HU 3-bet pot"),
        (4, 7, 29, "4-way raised"),
        (5, 2, 10, "5-way limp (worst live-5 shape)"),
        (6, 2, 12, "6-way limp (the old oracle's shape)"),
        (6, 7, 42, "6-way raised (largest census tree)"),
    ];

    eprintln!("\n════ v1 cell pricing: CPU, 4×4 runouts, {ITERS} iters ════");
    eprintln!("(live ≥ 4: bucketed quantile B={NB} Design1Collapsed — bucketing's np ≥ 4 scope;");
    eprintln!(" live 2-3: EXACT vector CFR — small-np terminals don't need the abstraction)");
    for &(live, commit, pot, label) in cells {
        let cfg = spec.flop_seam_config(live, commit, pot, flop_bets.clone());
        let tree = build_tree(&cfg).expect("seam tree");
        let table = build_table(live);
        let per_iter = if live >= 4 {
            let (fm, tm, rm) = quantile_maps(&table, NB);
            let game = FlopStartGame::new(table);
            let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
            let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
            solver.set_terminal_design(TerminalDesign::Design1Collapsed);
            let t0 = Instant::now();
            let _root = solver.run(&tree, &game, &bk, ITERS);
            t0.elapsed().as_secs_f64() / ITERS as f64
        } else {
            let game = FlopStartGame::new(table);
            let mut solver = FlopStartVectorCfr::new(&tree, game.table());
            let t0 = Instant::now();
            let _root = solver.run(&tree, &game, ITERS);
            t0.elapsed().as_secs_f64() / ITERS as f64
        };
        eprintln!(
            "live {live} commit {commit:>3} pot {pot:>4} ({label}): {} nodes, {per_iter:.3}s/iter \
             | 34 iters × 1755 flops ≈ {:.1}h/cell-row single-core",
            tree.nodes.len(),
            per_iter * 34.0 * 1755.0 / 3600.0
        );
    }
}
