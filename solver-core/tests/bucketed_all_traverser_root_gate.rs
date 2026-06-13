//! Gate for `BucketedFlopCfr::run_all_root_cfv` (2026-06-13, the
//! bucketed-oracle keystone primitive): the all-traverser root-CFV
//! variant must reproduce `run` EXACTLY on the traverser-0 slice — the
//! per-traverser-0 arithmetic is byte-identical; the new method only
//! additionally captures the other traversers' flop-root vectors that
//! `run` computed and discarded. Bit-exact (to_bits) or it's a
//! divergent copy, not a faithful one.
//!
//! Small NH=6 fixture (the g3-gate shape); independent of GPU.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

const NP: u8 = 4;
const NH: usize = 6;

fn build_table() -> FlopChanceTable {
    let board: Vec<Card> =
        ["Th", "9d", "8c"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
            continue;
        }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> =
        (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            ranges[p][hi as usize] = 1.0;
        }
    }
    let turn_cards: Vec<u8> =
        ["2c", "Jd"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let river_strs: [&[&str]; 2] = [&["4s", "7h"], &["3s", "Qc"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] =
            river_strs[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

#[test]
fn run_all_root_cfv_traverser0_bit_exact_vs_run() {
    // A live-4 seam tree (the family the oracle most needs all-traverser
    // values for).
    let spec = production_game_v1();
    let tree = build_tree(&spec.flop_seam_config(
        4,
        7,
        29,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
    ))
    .expect("seam tree");
    const NB: usize = 4;
    const ITERS: u32 = 6;

    // Arm A: run() — traverser-0 root CFV.
    let game_a = FlopStartGame::new(build_table());
    let bk_a = FlopBucketing::identity(game_a.table());
    let mut a = BucketedFlopCfr::new(&tree, game_a.table(), &bk_a);
    a.set_terminal_design(TerminalDesign::Design1Collapsed);
    let run0 = a.run(&tree, &game_a, &bk_a, ITERS);

    // Arm B: run_all_root_cfv() — all traversers; compare [0].
    let game_b = FlopStartGame::new(build_table());
    let bk_b = FlopBucketing::identity(game_b.table());
    let mut b = BucketedFlopCfr::new(&tree, game_b.table(), &bk_b);
    b.set_terminal_design(TerminalDesign::Design1Collapsed);
    let all = b.run_all_root_cfv(&tree, &game_b, &bk_b, ITERS);

    assert_eq!(all.len(), NP as usize, "one root vector per player");
    assert_eq!(all[0].len(), run0.len());
    for (i, (x, y)) in all[0].iter().zip(&run0).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "traverser-0 hand {i}: run_all {x} vs run {y} — copy must be faithful"
        );
    }
    // Sanity: other traversers produced finite, not-all-zero vectors.
    for t in 1..NP as usize {
        assert!(all[t].iter().all(|v| v.is_finite()), "traverser {t} finite");
        assert!(all[t].iter().any(|&v| v != 0.0), "traverser {t} nonzero");
    }
    let _ = NB;
    eprintln!("run_all_root_cfv: traverser-0 slice bit-exact vs run ({ITERS} iters, {NP} players)");
}

// ── Reconciliation gate (the keystone seam): table-hand-order →
// flop_combo_layout-order must be a BIJECTION over the same combo set. ──
#[test]
fn bucketed_oracle_reconciliation_gate() {
    use solver_core::solver::preflop_start_game::{flop_combo_layout, table_hand_to_layout_perm};
    let table = build_table(); // NH=6 chosen hands, board Th9d8c
    // NOTE: build_table uses a SUBSET (NH=6) so layout(full) != num_valid;
    // the perm gate needs the FULL board-compatible table. Use a full-nh
    // table on the same board for the bijection check.
    let board = ["Th", "9d", "8c"].map(|s| card_from_str(s).unwrap());
    let turn_cards: Vec<u8> =
        ["2c", "Jd"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] =
        ["4s", "7h"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    river_decks[turn_cards[1] as usize] =
        ["3s", "Qc"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let full = FlopChanceTable::build_full_nh_sampled(board, NP, &turn_cards, &river_decks);

    let perm = table_hand_to_layout_perm(&full.hand_cards, full.num_valid, board);
    let layout = flop_combo_layout(board);
    assert_eq!(perm.len(), layout.len());
    // Bijection: every table index hit exactly once.
    let mut seen = vec![false; full.num_valid];
    for &t in &perm {
        assert!(!seen[t], "table index {t} hit twice — not injective");
        seen[t] = true;
    }
    assert!(seen.iter().all(|&s| s), "not surjective — some table hand unmapped");
    // Correctness: perm maps layout combo to the table hand with the
    // SAME cards.
    for (li, &(c1, c2)) in layout.iter().enumerate() {
        let h = perm[li];
        let (a, b) = (full.hand_cards[h * 2], full.hand_cards[h * 2 + 1]);
        assert_eq!((c1.min(c2), c1.max(c2)), (a.min(b), a.max(b)),
            "layout {li} ({c1},{c2}) mapped to table hand {h} ({a},{b}) — wrong combo");
    }
    let _ = table;
    eprintln!("reconciliation: table↔layout bijection over {} combos, cards match", perm.len());
}
