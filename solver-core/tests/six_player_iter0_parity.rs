// Gate 5: CPU↔GPU iter-0 parity at full 6-player.
//
// Per the spec: "run gate 5 (full-6-player parity) on the complex multi-
// level multi-eligibility terminals specifically, because the host-side
// LevelInfo construction is the new wiring code and its bugs only show
// on complex terminals, not the simple ones."
//
// In this implementation, the level construction is done INSIDE the
// Metal kernel (replicating the K=2 path's level extraction). The
// principle still applies: the new K≥3 code path is only fully exercised
// by terminals with multiple pot levels and folds, so the gate must
// run on a game that produces complex terminals.
//
// The existing FlopStartGame at our betting structure produces 1-3 level
// terminals (multi-street betting with folds). The full 6p tree contains
// 485k terminals spanning the full range of (np, fold_mask, contributions)
// configurations the kernel must handle. A 1-iter CFR pass touches all
// of them, so this parity test is the broad-coverage gate.
//
// Method: same protocol as the 3-player iter-0 parity test:
//   1. Build the same 6p game and tree on CPU and GPU sides.
//   2. Run 1 CFR iter on each (CPU uses validated brute-force per-level
//      side_pot_showdown_cfv; GPU uses the unified factored kernel).
//   3. Compare regrets element-wise across all infosets and zones.
//
// Tolerance: f32 single-precision noise (~1e-5 to 1e-4 absolute on regret
// values that range up to ~5 at iter-1). The 3-player parity test passes
// at max_diff < 0.1; the 6-player factored has more accumulator depth so
// we set a slightly looser threshold for f32 noise compounding.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// AUDIT MIGRATION 2026-06: build_6p_table now uses the production API
/// `compute_flop_start_subset_with_decks`. This test anchors the
/// "6-player GPU correctness" claim that's load-bearing for eventual
/// 6-max production. It was tightened from 1e-3 to 1e-5 in commit
/// adb0d6c (#38) on a hand-rolled table; the audit confirms whether
/// that tightening was on solid ground.
///
/// FINDING (2026-06): This test's hand-rolled pattern was structurally
/// "done correctly" — it used the +1 anti-zero-padding rule, exact
/// recursive num_combinations enumeration, and uniform initial_weights
/// (matches what production gives for ranges=1.0 at chosen hands).
/// Substantively equivalent to production. Migration expected to
/// reproduce same max_diff.
fn build_6p_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 6u8;

    // Same hand selection as pre-migration: even step across non-board-blocking hands.
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();

    // Ranges: weight 1.0 at each chosen hand, 0.0 elsewhere.
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
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

#[test]
fn gate5_six_player_iter0_parity_nh6() {
    let nh = 6;
    let (tree, table) = build_6p_table(nh);
    let game = FlopStartGame::new(table);

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.set_iteration(0);

    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    eprintln!("\n=== Gate 5: 6-player iter-0 parity, nh={} ===", nh);
    eprintln!("Tree: {} nodes total", tree.num_nodes());

    // Note: CPU iter at 6p with K=5 brute-force per-level is slow. For nh=6
    // this is tractable but takes minutes.
    let t_cpu = std::time::Instant::now();
    cpu.run(&tree, &game, 1);
    let cpu_elapsed = t_cpu.elapsed();
    eprintln!("CPU 1 iter: {:?}", cpu_elapsed);

    let t_gpu = std::time::Instant::now();
    gpu.run(&ctx, &tree, &game, 1);
    let gpu_elapsed = t_gpu.elapsed();
    eprintln!("GPU 1 iter: {:?}", gpu_elapsed);

    let cpu_flop = cpu.regrets_flop();
    let cpu_turn = cpu.regrets_turn();
    let cpu_river = cpu.regrets_river();
    let gpu_all = gpu.download_regrets();

    let fl = cpu_flop.len();
    let tl = cpu_turn.len();
    let rl = cpu_river.len();
    eprintln!("Zone lengths: flop={} turn={} river={}", fl, tl, rl);

    let mut zone_max = [0.0f32; 3];
    let mut zone_count = [0usize; 3];
    let mut zone_sum_sq = [0.0f64; 3];

    for zone_idx in 0..3 {
        let (cpu_slice, gpu_start) = match zone_idx {
            0 => (cpu_flop, 0),
            1 => (cpu_turn, fl),
            _ => (cpu_river, fl + tl),
        };
        for i in 0..cpu_slice.len() {
            let gi = gpu_start + i;
            if gi >= gpu_all.len() { break; }
            let diff = (cpu_slice[i] - gpu_all[gi]).abs();
            if cpu_slice[i] != 0.0 || gpu_all[gi] != 0.0 {
                zone_max[zone_idx] = zone_max[zone_idx].max(diff);
                zone_sum_sq[zone_idx] += (diff as f64).powi(2);
                zone_count[zone_idx] += 1;
            }
        }
    }
    let flop_rmse = (zone_sum_sq[0] / zone_count[0].max(1) as f64).sqrt();
    let turn_rmse = (zone_sum_sq[1] / zone_count[1].max(1) as f64).sqrt();
    let river_rmse = (zone_sum_sq[2] / zone_count[2].max(1) as f64).sqrt();

    eprintln!("\nFlop zone:  max_diff={:.6e} rmse={:.6e} ({} nonzero entries)",
        zone_max[0], flop_rmse, zone_count[0]);
    eprintln!("Turn zone:  max_diff={:.6e} rmse={:.6e} ({} nonzero entries)",
        zone_max[1], turn_rmse, zone_count[1]);
    eprintln!("River zone: max_diff={:.6e} rmse={:.6e} ({} nonzero entries)",
        zone_max[2], river_rmse, zone_count[2]);

    let max_diff = zone_max[0].max(zone_max[1]).max(zone_max[2]);
    eprintln!("\nOverall max_diff = {:.6e}", max_diff);

    // Threshold: f32 noise compounded through factored multi-level
    // composition. The 3-player parity passes at ~2e-6 max_diff; the
    // 6-player factored has deeper accumulation. Loose initial threshold
    // for diagnostic; tighten as the integration matures.
    // #38 final tightening: from 1e-3 → 1e-5 after #39's orchestration oracle
    // anchored bottom_up_zone against textbook CFR formula. With both CPU and
    // GPU anchored against independent references (showdown oracle + CFR
    // formula oracle), tight CPU-GPU tolerance now means precision around
    // truth, not just precision around mutual agreement. Empirical: 1.9e-6.
    assert!(max_diff < 1e-5,
        "Gate 5: 6-player iter-0 parity failed. Overall max_diff = {} (threshold 1e-5)",
        max_diff);
}
