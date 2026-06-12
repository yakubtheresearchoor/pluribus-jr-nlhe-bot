// Step 2.B.2 helper unit test: `river_outcome_idx(ti, ri)` is the single
// source of truth for river kernel buffer addressing. Per the consolidation
// principle, a bug here surfaces loud-everywhere (every river dispatch
// site at once) rather than quiet-at-one-site, so testing it directly is
// cheap insurance for the load-bearing piece.
//
// What this gates:
//   1. InMemory mode returns `ti * max_river + ri` — matches the legacy
//      hard-coded computation that the audit-era code used at every site.
//   2. DiskBacked mode returns 0 — the kernel addresses the scratch slot
//      regardless of which logical (ti, ri) pair is loaded.
//   3. The mode transition (into_disk_backed_gpu) flips the return value
//      for the same (ti, ri) input.
//
// What this does NOT gate (covered by 2.B.2 end-to-end validation):
//   - Whether kernel dispatches actually USE the helper.
//   - Whether load/save correctly cycle state through the scratch.
//   - End-to-end bit-exact GPU-InMemory vs GPU-DiskBacked.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::{GpuRiverMode, MetalFlopStartSolver};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// Use a 6p game with MULTIPLE turn cards and MULTIPLE river cards per
/// turn so (ti=0, ri=0), (ti=0, ri=1), (ti=1, ri=0), etc. all
/// exercise distinct InMemory outcome indices (catches max_river vs
/// max(n_river_per_turn) confusion).
fn build_6p_multi_pair_game() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 6u8;
    let nh = 6usize;

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();

    let mut ranges: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..num_players as usize {
        for &hi in &chosen {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
            ranges[p][pair_idx] = 1.0;
        }
    }
    // 2 turn cards × 2 river cards per turn = 4 distinct (ti, ri) pairs.
    let tc_3c = card_from_str("3c").unwrap() as u8;
    let tc_4c = card_from_str("4c").unwrap() as u8;
    let turn_cards = vec![tc_3c, tc_4c];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[tc_3c as usize] = vec![
        card_from_str("5s").unwrap() as u8,
        card_from_str("6c").unwrap() as u8,
    ];
    river_decks[tc_4c as usize] = vec![
        card_from_str("8d").unwrap() as u8,
        card_from_str("9h").unwrap() as u8,
    ];

    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, num_players, &chosen, &turn_cards, &river_decks,
    );

    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![100; 6], initial_contributions: vec![5; 6],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = FlopStartGame::new(table);
    (tree, game)
}

struct ScratchFile(std::path::PathBuf);
impl Drop for ScratchFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}

/// GATE 1: InMemory mode returns `ti * max_river + ri` for all valid
/// (ti, ri) pairs — matches the legacy hard-coded computation.
#[test]
fn river_outcome_idx_in_memory_matches_ti_times_max_river_plus_ri() {
    let (tree, game) = build_6p_multi_pair_game();
    let ctx = MetalContext::new().expect("Metal");
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let max_river = gpu.max_river();
    let n_turn = gpu.n_turn();
    assert!(matches!(gpu.gpu_river_mode(), GpuRiverMode::InMemory));

    // Sanity on the test setup itself.
    assert!(max_river >= 2, "test wants at least 2 river cards per turn");
    assert!(n_turn >= 2, "test wants at least 2 turn cards");

    for ti in 0..n_turn {
        for ri in 0..max_river {
            let expected = ti * max_river + ri;
            let got = gpu.river_outcome_idx(ti, ri);
            assert_eq!(got, expected,
                "InMemory river_outcome_idx({}, {}) = {} (expected {} = {}*max_river+{})",
                ti, ri, got, expected, ti, ri);
        }
    }
}

/// GATE 2: DiskBacked mode returns 0 for ALL (ti, ri) pairs — the
/// kernel always addresses the scratch slot, regardless of which
/// logical pair is loaded.
#[test]
fn river_outcome_idx_disk_backed_is_zero_for_all_pairs() {
    let (tree, game) = build_6p_multi_pair_game();
    let ctx = MetalContext::new().expect("Metal");
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let regrets_guard = ScratchFile(std::env::temp_dir()
        .join("step2b2_helper_regrets.bin"));
    let cum_guard = ScratchFile(std::env::temp_dir()
        .join("step2b2_helper_cum.bin"));
    let _ = std::fs::remove_file(&regrets_guard.0);
    let _ = std::fs::remove_file(&cum_guard.0);

    gpu.into_disk_backed_gpu(&ctx, &regrets_guard.0, &cum_guard.0)
        .expect("into_disk_backed_gpu");
    assert!(matches!(gpu.gpu_river_mode(), GpuRiverMode::DiskBacked { .. }));

    let max_river = gpu.max_river();
    let n_turn = gpu.n_turn();
    assert!(max_river >= 2 && n_turn >= 2);

    for ti in 0..n_turn {
        for ri in 0..max_river {
            let got = gpu.river_outcome_idx(ti, ri);
            assert_eq!(got, 0,
                "DiskBacked river_outcome_idx({}, {}) = {} (expected 0 — scratch addressing)",
                ti, ri, got);
        }
    }
}

/// GATE 3: The mode transition (into_disk_backed_gpu) flips the helper
/// output for the same (ti, ri) input. Catches a stale-mode bug where the
/// helper reads a cached mode value instead of the live one.
#[test]
fn river_outcome_idx_flips_on_mode_transition() {
    let (tree, game) = build_6p_multi_pair_game();
    let ctx = MetalContext::new().expect("Metal");
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let max_river = gpu.max_river();
    let n_turn = gpu.n_turn();
    let (ti, ri) = (1, 1);  // non-zero pair so the expected value isn't trivially 0
    assert!(ti < n_turn && ri < max_river);

    // PRE-transition: InMemory → expect ti*max_river+ri.
    let pre = gpu.river_outcome_idx(ti, ri);
    let expected_pre = ti * max_river + ri;
    assert_eq!(pre, expected_pre);
    assert_ne!(pre, 0,
        "test sanity: chosen (ti, ri) = (1, 1) should give non-zero in InMemory \
         (otherwise the flip-to-zero is trivially observed)");

    let regrets_guard = ScratchFile(std::env::temp_dir()
        .join("step2b2_helper_flip_regrets.bin"));
    let cum_guard = ScratchFile(std::env::temp_dir()
        .join("step2b2_helper_flip_cum.bin"));
    let _ = std::fs::remove_file(&regrets_guard.0);
    let _ = std::fs::remove_file(&cum_guard.0);

    gpu.into_disk_backed_gpu(&ctx, &regrets_guard.0, &cum_guard.0)
        .expect("into_disk_backed_gpu");

    // POST-transition: DiskBacked → expect 0.
    let post = gpu.river_outcome_idx(ti, ri);
    assert_eq!(post, 0,
        "POST-transition river_outcome_idx({}, {}) = {} (expected 0)",
        ti, ri, post);
    assert_ne!(pre, post,
        "mode transition did not flip the helper output (pre = {}, post = {})",
        pre, post);
}
