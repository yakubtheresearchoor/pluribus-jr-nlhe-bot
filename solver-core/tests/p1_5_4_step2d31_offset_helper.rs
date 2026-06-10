// Phase 2 / Phase 3: Unit test the consolidated inter-zone offset helper.
//
// The helper (MetalFlopStartSolver::infoset_float_offset /
// MetalFlopStartSolver::infoset_byte_offset, see flop_solver.rs) consolidates
// the inter-zone offset arithmetic that used to be inlined at every dispatch
// site. With per-stage MAX_NA in Phase 2, the strides for each zone become
// per-stage values (postflop in this case), and stride math has been a known
// bug source in this project, so the consolidated helper is exercised here
// against hand-computed reference offsets PER ZONE.
//
// The cross-stage stride DIFFERENCE (preflop vs postflop) is the per-stage
// bug surface; the cross-OUTCOME stride DIFFERENCE within postflop (turn ti
// vs river outcome_idx) is the historical bug surface (disk-backed load
// previously miscounted river outcome offsets). Both are exercised here.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::{BufferZone, MetalFlopStartSolver};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};

fn build_6p_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let np = 6u8;
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..np as usize {
        for &hi in &chosen {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
            ranges[p][pair_idx] = 1.0;
        }
    }
    // Use 2 turn cards × 2 river cards = 4 (ti, ri) pairs so we can test
    // multiple distinct outcome_idx values in the River zone.
    let turn_cards = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("9s").unwrap() as u8,
    ];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("5s").unwrap() as u8,
        card_from_str("6h").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("4d").unwrap() as u8,
        card_from_str("Tc").unwrap() as u8,
    ];
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, np, &chosen, &turn_cards, &river_decks,
    );
    let config = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![100; np as usize],
        initial_contributions: vec![5; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

#[test]
fn offset_helper_flop_returns_zero() {
    let (tree, table) = build_6p_table(8);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Flop is always at the start of the postflop buffer.
    assert_eq!(gpu.infoset_float_offset(BufferZone::Flop), 0,
        "Flop zone must start at float offset 0 (it's the first stacked zone in d_regrets)");
    assert_eq!(gpu.infoset_byte_offset(BufferZone::Flop), 0,
        "Flop zone byte offset must be 0");
}

#[test]
fn offset_helper_turn_matches_hand_computation() {
    let (tree, table) = build_6p_table(8);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let turn_off = gpu.turn_offset();
    let turn_stride = gpu.turn_stride();
    assert!(turn_stride > 0, "turn_stride should be nonzero for a valid tree");
    assert!(turn_off > 0, "turn_offset should be after the flop zone");

    // The relationship that must hold: turn_stride = turn_infosets * MAX_NA_POSTFLOP * nh.
    // We don't have direct access to turn_infosets but we can sanity-check that
    // turn_stride is divisible by (MAX_NA_POSTFLOP * nh).
    let cell = MAX_NA_POSTFLOP * cpu.num_hands();
    assert_eq!(turn_stride % cell, 0,
        "turn_stride ({}) must be divisible by MAX_NA_POSTFLOP * nh ({}) — \
         the stride bookkeeping ties them together",
        turn_stride, cell);

    // Test turn outcomes ti = 0..n_turn against hand-computed offsets.
    for ti in 0..gpu.n_turn() {
        let expected_float = turn_off + ti * turn_stride;
        let expected_byte = (expected_float * std::mem::size_of::<f32>()) as u64;
        assert_eq!(
            gpu.infoset_float_offset(BufferZone::Turn { ti }),
            expected_float,
            "Turn ti={} float offset mismatch", ti
        );
        assert_eq!(
            gpu.infoset_byte_offset(BufferZone::Turn { ti }),
            expected_byte,
            "Turn ti={} byte offset mismatch", ti
        );
    }
}

#[test]
fn offset_helper_river_matches_hand_computation() {
    let (tree, table) = build_6p_table(8);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let river_off = gpu.river_offset();
    let river_stride = gpu.river_stride();
    assert!(river_stride > 0, "river_stride should be nonzero");
    assert!(river_off > 0, "river_offset should be after flop+turn zones");

    // Per-stride sanity: divisible by MAX_NA_POSTFLOP * nh.
    let cell = MAX_NA_POSTFLOP * cpu.num_hands();
    assert_eq!(river_stride % cell, 0,
        "river_stride ({}) must be divisible by MAX_NA_POSTFLOP * nh ({})",
        river_stride, cell);

    // Test multiple distinct outcome_idx values. We use river_outcome_idx
    // for each (ti, ri) and verify the offset helper matches the
    // hand-computed formula.
    let n_turn = gpu.n_turn();
    let max_river = gpu.max_river();
    let mut seen_outcomes = std::collections::HashSet::new();
    let outcomes_per_turn = gpu.river_outcomes_per_turn();
    for ti in 0..n_turn {
        let n_river = outcomes_per_turn[ti];
        for ri in 0..n_river {
            let outcome_idx = gpu.river_outcome_idx(ti, ri);
            seen_outcomes.insert(outcome_idx);
            let expected_float = river_off + outcome_idx * river_stride;
            let expected_byte = (expected_float * std::mem::size_of::<f32>()) as u64;
            assert_eq!(
                gpu.infoset_float_offset(BufferZone::River { outcome_idx }),
                expected_float,
                "River outcome_idx={} (ti={} ri={}) float offset mismatch",
                outcome_idx, ti, ri
            );
            assert_eq!(
                gpu.infoset_byte_offset(BufferZone::River { outcome_idx }),
                expected_byte,
                "River outcome_idx={} (ti={} ri={}) byte offset mismatch",
                outcome_idx, ti, ri
            );
        }
    }
    // Make sure we tested at least 2 distinct outcomes (the cross-outcome
    // stride is the bug-prone slot).
    assert!(seen_outcomes.len() >= 2,
        "Test config must produce at least 2 distinct River outcomes to exercise \
         the cross-outcome stride — got {}", seen_outcomes.len());
    let _ = max_river;
}

/// Phase B3 cross-source check: the GPU solver's offsets must agree with
/// an INDEPENDENTLY-constructed ZoneDims built from the same logical
/// inputs. The solver's helper now delegates to its internal ZoneDims;
/// this test rebuilds the dims from scratch (different code route to the
/// same numbers) so a constructor-wiring bug can't self-certify.
#[test]
fn offset_helper_matches_independent_zone_dims() {
    use solver_core::solver::zone_dims::{ZoneDims, ZoneRef};
    use solver_core::tree::flat::MAX_NA_POSTFLOP;

    let (tree, table) = build_6p_table(8);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let dims = ZoneDims::uniform(
        MAX_NA_POSTFLOP,
        cpu.num_hands(),
        cpu.flop_infosets(),
        cpu.turn_infosets(),
        cpu.river_infosets(),
        gpu.n_turn(),
        gpu.max_river(),
    );

    assert_eq!(gpu.infoset_float_offset(BufferZone::Flop),
        dims.zone_float_offset(ZoneRef::Flop));
    for ti in 0..gpu.n_turn() {
        assert_eq!(gpu.infoset_float_offset(BufferZone::Turn { ti }),
            dims.zone_float_offset(ZoneRef::Turn { ti }),
            "turn ti={ti} offset diverges from independent ZoneDims");
    }
    for o in 0..gpu.n_turn() * gpu.max_river() {
        assert_eq!(gpu.infoset_float_offset(BufferZone::River { outcome_idx: o }),
            dims.zone_float_offset(ZoneRef::River { outcome_idx: o }),
            "river outcome={o} offset diverges from independent ZoneDims");
    }
    assert_eq!(gpu.turn_stride(), dims.turn_stride());
}

#[test]
fn offset_helper_zones_are_strictly_ordered() {
    // Cross-stage gate: Flop < Turn < River and Turn outcomes lie strictly
    // inside the turn-zone range. A bug that made any zone's offsets bleed
    // into another zone's would show up here as a non-strict ordering.
    let (tree, table) = build_6p_table(8);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let flop = gpu.infoset_float_offset(BufferZone::Flop);
    let turn_first = gpu.infoset_float_offset(BufferZone::Turn { ti: 0 });
    let river_first = gpu.infoset_float_offset(BufferZone::River { outcome_idx: 0 });
    assert_eq!(flop, 0);
    assert!(turn_first > flop,
        "Turn must start after Flop: flop={} turn={}", flop, turn_first);
    assert!(river_first > turn_first,
        "River must start after Turn: turn={} river={}", turn_first, river_first);

    // Last turn outcome must lie strictly before river_first.
    let n_turn = gpu.n_turn();
    if n_turn >= 2 {
        let turn_last = gpu.infoset_float_offset(BufferZone::Turn { ti: n_turn - 1 });
        let turn_stride = gpu.turn_stride();
        assert!(turn_last + turn_stride <= river_first,
            "Last turn outcome's end ({}) must not overflow into the river zone (river_first={})",
            turn_last + turn_stride, river_first);
    }
}
