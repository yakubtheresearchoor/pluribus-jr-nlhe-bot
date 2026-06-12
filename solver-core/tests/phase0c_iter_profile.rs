// Phase 0.C: per-iteration wall-clock breakdown by phase.
//
// Time each phase of the GPU iter loop independently to identify
// where the per-iter cost goes. Categories per the spec:
//   (a) GPU kernel compute for showdown (fused inside bottom_up_*)
//   (b) GPU kernel for tree traversal and regret updates (compute_reach_*,
//       and the non-showdown portion of bottom_up_*)
//   (c) Host-side work (orchestration loop overhead)
//   (d) Synchronization/launch overhead (wait_until_completed cost)
//
// The unified factored kernel is FUSED with regret update inside
// `vcfr_bottom_up_batched`, so (a) and (b) can't be cleanly separated
// at the dispatch level — they're reported as one combined cost
// "backward + showdown + regret" with a note for Phase 2 follow-up.
//
// What we can cleanly measure:
//   - compute_all_strategies time
//   - compute_reach_flop time
//   - per-turn loop: compute_reach_turn + per-river inner + chance_accumulate/finalize + bottom_up_turn
//   - bottom_up_flop time
//   - Total wall-clock minus sum of measured phases = host orchestration overhead
//
// Methodology: same iter structure as MetalFlopStartSolver::run() but
// with Instant::now() around each phase. Run several warmup iters
// (transition through iter-2 spike to settle), then profile iter ~6
// which is in steady-state.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::io::Write;
use std::time::{Duration, Instant};

/// AUDIT MIGRATION + 2.C reuse 2026-06: hand-rolling replaced by production
/// API. Pattern was structurally correct; bit-identical to production
/// (verified in aggregate via six_player). The Phase 0.C deliverable
/// (per-stage breakdown) is now produced by `MetalFlopStartSolver::run_profiled`
/// — see end of file.
fn build_6p_asymmetric_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 6u8;

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
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

// PhaseBudget + one_iter_with_phase_timing — REPLACED 2026-06 by the
// formalized `MetalFlopStartSolver::run_profiled` / `StageProfile` API
// (Step 2.C). The manual loop duplication was a Step 2.B blocker: every
// new GPU stage (DiskBacked I/O) would require updating both run() and
// every diagnostic test independently. The formalized timing surface is
// the single point of stage addition.

fn fmt_phase(name: &str, d: Duration, total: Duration) -> String {
    let s = d.as_secs_f64();
    let pct = s / total.as_secs_f64() * 100.0;
    format!("  {:30} {:>10.2} s  ({:>5.1}%)", name, s, pct)
}

#[test]
#[ignore = "Phase 0.C profile, ~30 min at nh=12"]
fn phase0c_per_iter_breakdown_nh12() {
    let nh = 12usize;

    eprintln!("\n=== Phase 0.C: per-iter wall-clock breakdown at nh={} ===\n", nh);
    let (tree, table) = build_6p_asymmetric_table(nh);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    eprintln!("Tree: {} nodes, {} MB regrets at nh={}",
        tree.num_nodes(), tree.num_nodes() * nh * 4 / 1024 / 1024, nh);
    std::io::stderr().flush().ok();

    // Warmup: settle JIT, get past iter-1 (uniform reach) and iter-2 (transition spike).
    eprintln!("Warmup pass: 1 iter to absorb kernel JIT (~14 min expected)...");
    let t_warm = Instant::now();
    gpu.run(&ctx, &tree, &game, 1);
    eprintln!("Warmup done: {:.1} s", t_warm.elapsed().as_secs_f64());

    eprintln!("Throwaway iter 2 (transition spike, ~12 min expected)...");
    let t = Instant::now();
    gpu.run(&ctx, &tree, &game, 1);
    eprintln!("  iter 2 done: {:.1} s (skipped from profile)", t.elapsed().as_secs_f64());

    eprintln!("Throwaway iters 3-5 (settling)...");
    let t = Instant::now();
    gpu.run(&ctx, &tree, &game, 3);
    eprintln!("  iters 3-5 done: {:.1} s (skipped from profile)", t.elapsed().as_secs_f64());

    eprintln!();
    eprintln!("Profiled iter 6 (steady state) via run_profiled:");
    std::io::stderr().flush().ok();
    let prof = gpu.run_profiled(&ctx, &tree, &game, 1);

    eprintln!();
    eprintln!("=== Phase budget for iter 6 (total {:.2} s) ===", prof.total.as_secs_f64());
    let host_overhead = prof.total.saturating_sub(prof.attributed());

    println!("{}", fmt_phase("Strategy compute (flop)", prof.compute_strategies, prof.total));
    println!("{}", fmt_phase("Reach: flop forward", prof.compute_reach_flop, prof.total));
    println!("{}", fmt_phase("Reach: turn (per-turn)", prof.compute_reach_turn, prof.total));
    println!("{}", fmt_phase("Reach: river (per-river)", prof.compute_reach_river, prof.total));
    println!("{}", fmt_phase("Bottom-up: RIVER (showdown+regret)", prof.bottom_up_river, prof.total));
    println!("{}", fmt_phase("Bottom-up: TURN", prof.bottom_up_turn, prof.total));
    println!("{}", fmt_phase("Bottom-up: FLOP", prof.bottom_up_flop, prof.total));
    println!("{}", fmt_phase("Chance accumulate (river)", prof.chance_accumulate_river, prof.total));
    println!("{}", fmt_phase("Chance finalize (river)", prof.chance_finalize_river, prof.total));
    println!("{}", fmt_phase("Chance accumulate (turn)", prof.chance_accumulate_turn, prof.total));
    println!("{}", fmt_phase("Chance finalize (turn)", prof.chance_finalize_turn, prof.total));
    println!("{}", fmt_phase("Zero buffer (small ops)", prof.zero_buffer_total, prof.total));
    println!("{}", fmt_phase("HOST/SYNC OVERHEAD (residual)", host_overhead, prof.total));
    println!();
    println!("=== Aggregated categories (Phase 0 deliverable) ===");
    let category_traversal = prof.compute_strategies + prof.compute_reach_flop +
                              prof.compute_reach_turn + prof.compute_reach_river +
                              prof.chance_accumulate_river + prof.chance_finalize_river +
                              prof.chance_accumulate_turn + prof.chance_finalize_turn;
    let category_backward_showdown_regret = prof.bottom_up_river + prof.bottom_up_turn + prof.bottom_up_flop;
    println!("{}", fmt_phase("Tree traversal + reach + chance integration", category_traversal, prof.total));
    println!("{}", fmt_phase("Backward + SHOWDOWN + regret update", category_backward_showdown_regret, prof.total));
    println!("{}", fmt_phase("Zero buffers", prof.zero_buffer_total, prof.total));
    println!("{}", fmt_phase("Host orchestration + sync overhead", host_overhead, prof.total));
    println!();
    println!("Note: showdown CFV is FUSED with regret update inside vcfr_bottom_up_batched");
    println!("kernel, so cannot be cleanly separated at the dispatch level. The 'Backward +");
    println!("SHOWDOWN + regret' category combines all three. If this category dominates,");
    println!("Phase 2 should add kernel-level timestamps to isolate the showdown specifically.");
}
