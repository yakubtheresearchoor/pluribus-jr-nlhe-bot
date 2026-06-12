//! B3 step 4 gate: the bucketed walk at B = nh must reproduce
//! `FlopStartVectorCfr` BIT-EXACTLY — root CFVs and every persistent
//! buffer (regrets + cum_strategy, all three zones) after multiple full
//! DCFR iterations. No tolerance ("if it comes back merely close,
//! that's a finding to chase, not a fallback to accept").
//!
//! Config: the Phase 4 wet-deep shape (6-max, Th9d8c, stacks 500, pot
//! 30, 2 turns × 2 rivers sampled chance, NH=6 stride-sampled hands) —
//! the same family the quality gate (step 5) reuses. Bet set fits the
//! production MAX_NA_POSTFLOP = 4 bank.
//!
//! What this gate certifies: stride math (divergent-nh ZoneDims path),
//! map indirection, per-bucket regret aggregation, terminal
//! reduce→Design-1→expand seam, DCFR discounting, accumulation order.
//! What it STRUCTURALLY CANNOT see: within-bucket reach drift across
//! imperfect-recall boundaries (vacuous at singletons) — that error
//! class is certified only by the step-5 quality gate. Documented at
//! postflop_buckets.rs module docs (B1 note 2).

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const NP: u8 = 6;
const NH: usize = 6;
const STACKS: i32 = 500;
const STARTING_POT: i32 = 30;
const STARTING_CONTRIB: i32 = 5;
const ITERS: u32 = 5;

/// Phase 4 wet-deep chance table (Th9d8c, 2×2 sampled runouts,
/// stride-sampled NH-hand universe) — same construction as
/// p1_5_4_phase4_redo_measurement.rs.
fn build_wet_deep_table() -> FlopChanceTable {
    let board: Vec<Card> = ["Th", "9d", "8c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
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
    let mut ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            ranges[p][hi as usize] = 1.0;
        }
    }
    let turn_cards: Vec<u8> = ["2c", "Jd"]
        .iter()
        .map(|s| card_from_str(s).unwrap() as u8)
        .collect();
    let river_strs_per_turn: [&[&str]; 2] = [&["4s", "7h"], &["3s", "Qc"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] = river_strs_per_turn[ti]
            .iter()
            .map(|s| card_from_str(s).unwrap() as u8)
            .collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

fn build_wet_deep_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: STARTING_POT,
        starting_stacks: vec![STACKS; NP as usize],
        initial_contributions: vec![STARTING_CONTRIB; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.33), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&config).unwrap()
}

fn assert_buffers_bit_exact(label: &str, exact: &[f32], bucketed: &[f32]) {
    assert_eq!(exact.len(), bucketed.len(), "{label}: length mismatch");
    let mut first_diff = None;
    let mut diff_count = 0usize;
    for i in 0..exact.len() {
        if exact[i].to_bits() != bucketed[i].to_bits() {
            diff_count += 1;
            if first_diff.is_none() {
                first_diff = Some(i);
            }
        }
    }
    assert!(
        diff_count == 0,
        "{label}: {diff_count}/{} slots differ; first at [{}]: exact {} ({:#010x}) vs bucketed {} ({:#010x})",
        exact.len(),
        first_diff.unwrap(),
        exact[first_diff.unwrap()],
        exact[first_diff.unwrap()].to_bits(),
        bucketed[first_diff.unwrap()],
        bucketed[first_diff.unwrap()].to_bits(),
    );
}

#[test]
fn identity_gate_wet_deep_bit_exact() {
    let tree = build_wet_deep_tree();
    eprintln!(
        "wet-deep tree: {} nodes, {} decision nodes",
        tree.num_nodes(),
        tree.decision_node_ids.len()
    );

    // Exact reference.
    let game_a = FlopStartGame::new(build_wet_deep_table());
    let mut exact = FlopStartVectorCfr::new(&tree, game_a.table());
    let root_exact = exact.run(&tree, &game_a, ITERS);

    // Bucketed at B = nh through the bucketed code path.
    let game_b = FlopStartGame::new(build_wet_deep_table());
    let bk = FlopBucketing::identity(game_b.table());
    assert_eq!(bk.nb_flop, NH);
    let mut bucketed = BucketedFlopCfr::new(&tree, game_b.table(), &bk);
    let root_bucketed = bucketed.run(&tree, &game_b, &bk, ITERS);

    // Design1Collapsed through the SAME walk: at singletons the
    // collapsed arm-2's DP is point-mass selection, so the full walk
    // must also be bit-exact (the B4 collapse gate's walk-level link).
    let game_c = FlopStartGame::new(build_wet_deep_table());
    let bk_c = FlopBucketing::identity(game_c.table());
    let mut collapsed = BucketedFlopCfr::new(&tree, game_c.table(), &bk_c);
    collapsed.set_terminal_design(
        solver_core::solver::bucketed_flop_cfr::TerminalDesign::Design1Collapsed,
    );
    let root_collapsed = collapsed.run(&tree, &game_c, &bk_c, ITERS);

    // Stride sanity: at B = nh the bucketed strides must equal the
    // exact solver's (same infoset counts, same nh).
    assert_eq!(bucketed.flop_stride() * 1, exact.regrets_flop().len());
    assert_eq!(exact.turn_stride(), bucketed.turn_stride());
    assert_eq!(exact.river_stride(), bucketed.river_stride());

    assert_buffers_bit_exact("root_cfv", &root_exact, &root_bucketed);
    assert_buffers_bit_exact("regrets_flop", exact.regrets_flop(), bucketed.regrets_flop());
    assert_buffers_bit_exact(
        "cum_strategy_flop",
        exact.cum_strategy_flop(),
        bucketed.cum_strategy_flop(),
    );
    assert_buffers_bit_exact("regrets_turn", exact.regrets_turn(), bucketed.regrets_turn());
    assert_buffers_bit_exact(
        "cum_strategy_turn",
        exact.cum_strategy_turn(),
        bucketed.cum_strategy_turn(),
    );
    assert_buffers_bit_exact("regrets_river", exact.regrets_river(), bucketed.regrets_river());
    assert_buffers_bit_exact(
        "cum_strategy_river",
        exact.cum_strategy_river(),
        bucketed.cum_strategy_river(),
    );

    // Collapsed terminal, same standard.
    assert_buffers_bit_exact("collapsed root_cfv", &root_exact, &root_collapsed);
    assert_buffers_bit_exact(
        "collapsed regrets_flop",
        exact.regrets_flop(),
        collapsed.regrets_flop(),
    );
    assert_buffers_bit_exact(
        "collapsed cum_strategy_flop",
        exact.cum_strategy_flop(),
        collapsed.cum_strategy_flop(),
    );
    assert_buffers_bit_exact(
        "collapsed regrets_turn",
        exact.regrets_turn(),
        collapsed.regrets_turn(),
    );
    assert_buffers_bit_exact(
        "collapsed cum_strategy_turn",
        exact.cum_strategy_turn(),
        collapsed.cum_strategy_turn(),
    );
    assert_buffers_bit_exact(
        "collapsed regrets_river",
        exact.regrets_river(),
        collapsed.regrets_river(),
    );
    assert_buffers_bit_exact(
        "collapsed cum_strategy_river",
        exact.cum_strategy_river(),
        collapsed.cum_strategy_river(),
    );

    eprintln!(
        "identity gate PASSED: {} iters bit-exact across root cfv + 6 persistent buffers \
         ({} + {} + {} floats)",
        ITERS,
        exact.regrets_flop().len() * 2,
        exact.regrets_turn().len() * 2,
        exact.regrets_river().len() * 2,
    );
}
