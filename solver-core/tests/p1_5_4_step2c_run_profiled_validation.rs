// Step 2.C validation: `MetalFlopStartSolver::run_profiled` must produce
// bit-identical state to `run()`, with the StageProfile populated.
//
// Validation gates:
//   1. Regrets / cum_strategy after N iters: bit-identical between run()
//      and run_profiled(). The profile collection must not perturb state.
//   2. The profile's total > 0 and attributed (sum of stages) ≤ total.
//      Host overhead is the gap; should be small but positive.
//   3. Every stage in the profile that's expected to be exercised has
//      time > 0 (compute_strategies, compute_reach_flop, etc.).
//
// Why this matters for 2.B: when DiskBacked GPU lands, the I/O stages
// (load_river_pair, save_river_pair) will be added to StageProfile.
// Without the formalized API, every diagnostic test would have to
// re-implement the loop and add manual `Instant::now()` brackets around
// the new I/O calls — exactly the fragility 2.C eliminates.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// Build the same small K=6-hand 6-player game as six_player_iter0_parity.
/// Production-API construction; substantively equivalent to the post-audit
/// canonical 6p table.
fn build_6p_game() -> (FlatTree, FlopStartGame) {
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
    let turn_cards = vec![card_from_str("3c").unwrap() as u8];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![card_from_str("5s").unwrap() as u8];

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

/// GATE 1: bit-identical state between `run()` and `run_profiled()` after
/// N iters. If profile collection perturbs state, this fails.
#[test]
fn run_profiled_produces_bit_identical_state_to_run() {
    let (tree, game) = build_6p_game();
    let ctx = MetalContext::new().expect("Metal");

    // Two solvers, identical initial state.
    let cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let mut gpu_a = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);
    let mut gpu_b = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);

    let n_iters = 5;
    gpu_a.run(&ctx, &tree, &game, n_iters);
    let _profile = gpu_b.run_profiled(&ctx, &tree, &game, n_iters);

    // Compare every byte of regrets and cum_strategy.
    let reg_a = gpu_a.download_regrets();
    let reg_b = gpu_b.download_regrets();
    assert_eq!(reg_a.len(), reg_b.len(),
        "regret buffer size mismatch: run()={} run_profiled()={}",
        reg_a.len(), reg_b.len());
    for i in 0..reg_a.len() {
        assert_eq!(reg_a[i].to_bits(), reg_b[i].to_bits(),
            "regret[{}] differs: run()={} run_profiled()={}",
            i, reg_a[i], reg_b[i]);
    }

    let cum_a = gpu_a.download_cum_strategy();
    let cum_b = gpu_b.download_cum_strategy();
    assert_eq!(cum_a.len(), cum_b.len(),
        "cum_strategy buffer size mismatch");
    for i in 0..cum_a.len() {
        assert_eq!(cum_a[i].to_bits(), cum_b[i].to_bits(),
            "cum_strategy[{}] differs: run()={} run_profiled()={}",
            i, cum_a[i], cum_b[i]);
    }

    // Iteration counter must also be identical.
    assert_eq!(gpu_a.iteration(), gpu_b.iteration(),
        "iteration counter diverged: run()={} run_profiled()={}",
        gpu_a.iteration(), gpu_b.iteration());
}

/// GATE 2 & 3: the profile is populated sensibly. total > 0, attributed
/// fits inside total, and every expected stage has time > 0.
#[test]
fn run_profiled_returns_populated_stage_profile() {
    let (tree, game) = build_6p_game();
    let ctx = MetalContext::new().expect("Metal");
    let cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);

    let profile = gpu.run_profiled(&ctx, &tree, &game, 3);

    // Total must be > 0 — we ran 3 iters on a 6p game.
    assert!(profile.total.as_nanos() > 0,
        "StageProfile.total is zero, but we ran 3 iters");

    // Attributed (sum of stages) must be ≤ total. The gap is host overhead
    // (command-buffer setup outside any time_stage! bracket, etc.).
    assert!(profile.attributed() <= profile.total,
        "attributed ({}ns) > total ({}ns) — accounting bug",
        profile.attributed().as_nanos(), profile.total.as_nanos());

    // Stages that MUST be exercised in any 6p iter:
    assert!(profile.compute_strategies.as_nanos() > 0,
        "compute_strategies stage never timed");
    assert!(profile.compute_reach_flop.as_nanos() > 0,
        "compute_reach_flop stage never timed");
    assert!(profile.compute_reach_turn.as_nanos() > 0,
        "compute_reach_turn stage never timed");
    assert!(profile.compute_reach_river.as_nanos() > 0,
        "compute_reach_river stage never timed");
    assert!(profile.bottom_up_river.as_nanos() > 0,
        "bottom_up_river stage never timed");
    assert!(profile.bottom_up_turn.as_nanos() > 0,
        "bottom_up_turn stage never timed");
    assert!(profile.bottom_up_flop.as_nanos() > 0,
        "bottom_up_flop stage never timed");
    assert!(profile.chance_accumulate_river.as_nanos() > 0,
        "chance_accumulate_river stage never timed");
    assert!(profile.chance_finalize_river.as_nanos() > 0,
        "chance_finalize_river stage never timed");
    assert!(profile.chance_accumulate_turn.as_nanos() > 0,
        "chance_accumulate_turn stage never timed");
    assert!(profile.chance_finalize_turn.as_nanos() > 0,
        "chance_finalize_turn stage never timed");
    assert!(profile.zero_buffer_total.as_nanos() > 0,
        "zero_buffer_total stage never timed");

    eprintln!("\n{}", profile.report());
}
