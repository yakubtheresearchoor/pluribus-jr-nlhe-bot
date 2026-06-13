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

// ════════════════════════════════════════════════════════════════════
// WRAPPER GATES (2026-06-13): the assembled bucketed_live_subset_source.
// Per the resume refinements, all on a MULTIWAY-WITH-FOLDS live-subset
// cell (live-3 of a 4-handed deal: ≥2 live traversers AND ≥1 folded
// opponent) — NOT a traverser-0/HU path (already proven).

use solver_core::solver::postflop_oracle::{bucketed_live_subset_source, SeamCell};
use solver_core::tree::action::production_game_v1 as pgv1;

// A live-3 cell out of a 6-max game (3 folded preflop). mask: seats
// 3,4,5 folded; live seats 0,1,2. commit/pot of a 3-way single-raised
// pot. The traverser we query is a LIVE NON-ZERO seat (seat 2), so the
// gate exercises the multi-traverser selection the traverser-0 path
// can't.
const LIVE3_MASK: u16 = 0b111000; // seats 3,4,5 folded
// live, commit, pot — pot ≥ live×commit (3×7=21 ≤ 25; 4 units dead from
// folded blinds), a 3-way single-raised pot.
const LIVE3_CELL: (u8, i32, i32) = (3, 7, 25);

fn flopv(s: [&str; 3]) -> [Card; 3] {
    [card_from_str(s[0]).unwrap(), card_from_str(s[1]).unwrap(), card_from_str(s[2]).unwrap()]
}

#[test]
fn bucketed_oracle_range_convention_gate() {
    // REACH-INDEPENDENCE (load-bearing: all three consumers inherit it).
    // Two DIFFERENT reach inputs must give byte-identical output, or the
    // shared-chance collapse contract is violated.
    let spec = pgv1();
    let cell = SeamCell { live: LIVE3_CELL.0, commit: LIVE3_CELL.1, pot: LIVE3_CELL.2 };
    let flop = flopv(["2h", "7d", "Ks"]);
    let mut src = bucketed_live_subset_source(spec, 8, 2, 2, 4);

    let np = 6usize;
    let nlay = {
        use solver_core::solver::preflop_start_game::flop_combo_layout;
        flop_combo_layout(flop).len()
    };
    let reach_a: Vec<Vec<f32>> = vec![vec![1.0f32; nlay]; np];
    let mut reach_b: Vec<Vec<f32>> = vec![vec![0.3f32; nlay]; np];
    for p in 0..np {
        for (i, v) in reach_b[p].iter_mut().enumerate() {
            *v = ((i % 7) as f32) * 0.11 + 0.01; // wildly different, nonuniform
        }
    }
    // Query a live non-zero traverser (seat 2).
    let va = src(cell, LIVE3_MASK, flop, &reach_a, 2);
    let vb = src(cell, LIVE3_MASK, flop, &reach_b, 2);
    // NOTE: same cell+flop+traverser ⇒ cache hit on the 2nd call, which
    // already proves reuse; to prove TRUE reach-independence (not just
    // caching) use a FRESH source for arm B.
    let mut src_b = bucketed_live_subset_source(pgv1(), 8, 2, 2, 4);
    let vb_fresh = src_b(cell, LIVE3_MASK, flop, &reach_b, 2);
    assert_eq!(va.len(), vb.len());
    for (i, (x, y)) in va.iter().zip(&vb_fresh).enumerate() {
        assert_eq!(
            x.to_bits(), y.to_bits(),
            "reach-independence VIOLATED at combo {i}: uniform {x} vs nonuniform {y} — \
             the frozen-oracle collapse contract requires identical output regardless of reach"
        );
    }
    let _ = vb;
    eprintln!("range-convention: reach-independent (2 reach inputs → bit-identical), {} combos", va.len());
}

#[test]
fn bucketed_oracle_wiring_gate() {
    // The wrapper must assemble the two gated pieces (run_all_root_cfv +
    // table_hand_to_layout_perm) correctly for a LIVE NON-ZERO traverser
    // in a multiway-with-folds cell. Compare wrapper ≡ direct
    // composition, bit-exact. (Proves traverser-selection + per-traverser
    // reconciliation application — the plumbing the traverser-0 gate
    // can't reach.)
    use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
    use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
    use solver_core::solver::preflop_start_game::{flop_combo_layout, table_hand_to_layout_perm};
    use solver_core::tree::action::BetSizeOptions;
    use solver_core::tree::action::BetSize;

    let spec = pgv1();
    let cell = SeamCell { live: LIVE3_CELL.0, commit: LIVE3_CELL.1, pot: LIVE3_CELL.2 };
    let flop = flopv(["2h", "7d", "Ks"]);
    const NB: usize = 8;
    const IT: u32 = 4;

    // Wrapper, query each live traverser (live indices 0,1,2 ↔ seats 0,1,2).
    let mut src = bucketed_live_subset_source(spec.clone(), NB, 2, 2, IT);
    let reach = vec![vec![1.0f32; flop_combo_layout(flop).len()]; 6];
    let wrap_seat2 = src(cell, LIVE3_MASK, flop, &reach, 2);

    // Direct: build the same live-3 game, run_all_root_cfv, reconcile.
    let tree = build_tree(&spec.flop_seam_config(
        3, LIVE3_CELL.1, LIVE3_CELL.2,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
    )).unwrap();
    let bm: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let turn_cards: Vec<u8> = [12usize, 36].iter().map(|&p| deck[p]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let rd: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        river_decks[tc as usize] = [10usize, 30].iter().map(|&p| rd[p]).collect();
    }
    let table = FlopChanceTable::build_full_nh_sampled(flop, 3, &turn_cards, &river_decks);
    let bk = FlopBucketing::quantile(&table, NB);
    let perm = table_hand_to_layout_perm(&table.hand_cards, table.num_valid, flop);
    let game = FlopStartGame::new(table);
    let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
    solver.set_terminal_design(TerminalDesign::Design1Collapsed);
    let root_all = solver.run_all_root_cfv(&tree, &game, &bk, IT);
    // live seat 2 = live index 2 (seats 0,1,2 all live in this mask).
    let direct_seat2: Vec<f32> = perm.iter().map(|&h| root_all[2][h]).collect();

    assert_eq!(wrap_seat2.len(), direct_seat2.len());
    for (i, (x, y)) in wrap_seat2.iter().zip(&direct_seat2).enumerate() {
        assert_eq!(
            x.to_bits(), y.to_bits(),
            "wiring combo {i}: wrapper {x} vs direct {y} — traverser-selection or \
             reconciliation-application is wrong for a live non-zero traverser"
        );
    }
    eprintln!("wiring: wrapper ≡ direct (run_all+perm) for live seat 2, multiway-with-folds, bit-exact");
}

/// VALUE gate: close the last sliver — the bucketed keystone's
/// traverser≥1 VALUES vs an INDEPENDENT EXACT reference, on a
/// multiway-with-folds live-3 cell, at NB-IDENTITY (no bucketing
/// lossiness — isolates the multi-traverser solve from the abstraction
/// v1_cell_parity already covers). REFERENCE = FlopStartVectorCfr
/// (exact, non-bucketed) — NOT compute_v_flop_at_root_converged, which
/// is the np≥3 combinatorial wall (measured non-terminating at nh=1176;
/// the exact-VECTOR solver on a SMALL NH=6 subset is fast). At
/// NB-identity the bucketed and exact-vector solvers are the SAME
/// arithmetic up to reduction order, so ALL traversers must agree to
/// accumulated rounding.
///
/// First proves the exact-vector all-traverser variant is itself
/// faithful (its [0] ≡ run() bit-exact), then checks every traverser.
#[test]
fn bucketed_oracle_value_gate_vs_exact_vector_multiway() {
    use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
    use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
    use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
    use solver_core::tree::action::{BetSize, BetSizeOptions};

    let spec = pgv1();
    let flop = flopv(["Th", "9d", "8c"]); // the NH=6 fixture board
    const IT: u32 = 8;
    // live-3 seam tree (multiway-with-folds: 3 live of a 6-handed deal).
    let tree = build_tree(&spec.flop_seam_config(
        3, LIVE3_CELL.1, LIVE3_CELL.2,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
    )).unwrap();

    // Build an NH=6 subset table on this board (fast exact reference).
    let board: Vec<Card> = flop.to_vec();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & ((1u64 << c1) | (1u64 << c2)) == 0 { all_valid.push(idx as u16); }
    }
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..3 { for &hi in &chosen { ranges[p][hi as usize] = 1.0; } }
    let turn_cards: Vec<u8> = ["2c", "Jd"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let river_strs: [&[&str]; 2] = [&["4s", "7h"], &["3s", "Qc"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] = river_strs[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    let mk = || FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, 3, &chosen, &turn_cards, &river_decks);

    // Bucketed @ identity, all traversers.
    let gb = FlopStartGame::new(mk());
    let bk = FlopBucketing::identity(gb.table());
    let mut bsolver = BucketedFlopCfr::new(&tree, gb.table(), &bk);
    bsolver.set_terminal_design(TerminalDesign::Design1Collapsed);
    let bucketed_all = bsolver.run_all_root_cfv(&tree, &gb, &bk, IT);

    // Exact-vector reference: prove its all-traverser variant faithful
    // ([0] ≡ run), then use all traversers.
    let g0 = FlopStartGame::new(mk());
    let mut v0 = FlopStartVectorCfr::new(&tree, g0.table());
    let exact_run0 = v0.run(&tree, &g0, IT);
    let g1 = FlopStartGame::new(mk());
    let mut v1s = FlopStartVectorCfr::new(&tree, g1.table());
    let exact_all = v1s.run_all_root_cfv(&tree, &g1, IT);
    for (i, (x, y)) in exact_all[0].iter().zip(&exact_run0).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(),
            "exact-vector run_all[0] hand {i}: {x} vs run {y} — reference copy not faithful");
    }

    // EVERY traverser: bucketed-identity ≡ exact-vector, drift-bounded.
    let mut worst = 0.0f64;
    for t in 0..3usize {
        let scale = exact_all[t].iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6) as f64;
        let d = bucketed_all[t].iter().zip(&exact_all[t])
            .map(|(a, b)| (*a as f64 - *b as f64).abs() / scale)
            .fold(0.0f64, f64::max);
        eprintln!("value gate t={t}: max rel drift {d:.2e} (bucketed-identity vs exact-vector)");
        worst = worst.max(d);
    }
    assert!(worst < 1e-3, "multi-traverser values diverge from exact ({worst:.2e}) — solve bug");
    eprintln!("value gate PASSED: keystone multi-traverser values match the exact-vector reference (all 3 live traversers)");
}
