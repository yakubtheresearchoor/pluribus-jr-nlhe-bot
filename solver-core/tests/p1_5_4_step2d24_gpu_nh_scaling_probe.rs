// Step 2.D.24 (#116): nh-scaling probe at fixed n_pairs=1.
//
// 2.D.8c measured 27 ms/iter at nh=8, n_pairs=1 (1 turn × 1 river).
// 2.D.23 estimated 12 min/iter at nh=1176, n_pairs=2162 (full deck).
// Naive linear-in-both: 27ms × 147 × 2162 = 2.4 hours/iter. Observed
// 12 min is faster than that prediction, but still 5-6 orders of
// magnitude above what the GPU FLOP count justifies.
//
// This probe isolates per-pair work's nh-dependence by varying nh at
// fixed n_pairs=1. Output: per-iter time vs nh. Scaling exponent tells
// us which kernel has nh-dependent cost.
//
// nh ∈ {8, 50, 200, 500, 1176}
//
// If per-iter ∝ nh: per-pair work is O(nh). Expected for vector-CFR.
// If per-iter ∝ nh²: O(nh²) somewhere — brute-force showdown? Quadratic
//   indexing in bottom_up?
// If per-iter constant: parallelism amortizes nh perfectly (would mean
//   the small-scale 27ms is mostly dispatch overhead).

#![cfg(feature = "metal")]

use solver_core::card::{card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::StageProfile;
use solver_core::gpu_metal::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::flop_combo_layout;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_tiny_flop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 4,
        starting_stacks: vec![10, 10],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&cfg).expect("flop tree builds")
}

fn make_combo_ranges(canonical: [Card; 3]) -> Vec<Vec<f32>> {
    let layout = flop_combo_layout(canonical);
    let np = 2usize;
    let mut full: Vec<Vec<f32>> = vec![vec![0.0f32; NUM_POSSIBLE_HANDS]; np];
    for p in 0..np {
        for (li, &(c1, c2)) in layout.iter().enumerate() {
            let w = 0.5 + (li as f32 * 0.01).sin() * 0.3 + (p as f32) * 0.05;
            full[p][card_pair_to_index(c1, c2)] = w.max(0.05).min(1.0);
        }
    }
    full
}

/// Pick subset of `n_hands` non-board-conflicting hands (mutual blocking
/// OK — handled at showdown terminal via conflict matrix). + 1 turn × 1
/// river per turn to fix n_pairs=1.
///
/// Turn/river cards are chosen to not conflict with the board. They MAY
/// conflict with some chosen hands; the conflict matrix handles this at
/// showdown (those hands contribute 0 to terminal CFV at that turn/river).
/// For per-iter cost measurement this is fine — cost depends on nh, not
/// whether some hands are blocked at terminals.
fn pick_subset_n_hands(canonical: [Card; 3], n_hands: usize) -> (Vec<u16>, Vec<u8>, Vec<Vec<u8>>) {
    let board_mask: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut chosen: Vec<u16> = Vec::new();
    // No used_cards tracking — allow overlapping hands.
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = solver_core::card::index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        chosen.push(idx as u16);
        if chosen.len() == n_hands { break; }
    }
    // Turn/river: just pick first 2 non-board cards (their interaction with
    // chosen hands is handled by conflict matrix at terminals).
    let mut turn_cards: Vec<u8> = Vec::new();
    for c in 0u8..52u8 {
        if board_mask & (1u64 << c) != 0 { continue; }
        turn_cards.push(c);
        if turn_cards.len() == 1 { break; }
    }
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let mut rivers: Vec<u8> = Vec::new();
        for c in 0u8..52u8 {
            if board_mask & (1u64 << c) != 0 { continue; }
            if c == tc { continue; }
            rivers.push(c);
            if rivers.len() == 1 { break; }
        }
        river_decks[tc as usize] = rivers;
    }
    (chosen, turn_cards, river_decks)
}

fn measure_at_nh(
    ctx: &MetalContext,
    flop_tree: &FlatTree,
    canonical: [Card; 3],
    n_hands_target: usize,
    n_iters: u32,
) -> (StageProfile, usize, usize) {
    let full = make_combo_ranges(canonical);
    let board: Vec<Card> = canonical.iter().copied().collect();

    let table = if n_hands_target >= 1000 {
        FlopChanceTable::compute_flop_start(&board, &full, 2)
    } else {
        let (chosen, turn_cards, river_decks) =
            pick_subset_n_hands(canonical, n_hands_target);
        FlopChanceTable::compute_flop_start_subset_with_decks(
            &board, &full, 2, &chosen, &turn_cards, &river_decks)
    };
    let actual_nh = table.num_valid;
    let n_pairs: usize = table.river_decks.iter().map(|d| d.len()).sum();
    let game = FlopStartGame::new(table);
    let cpu_solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let mut gpu_solver = MetalFlopStartSolver::new(ctx, flop_tree, &game, &cpu_solver);

    let _ = gpu_solver.run_profiled(ctx, flop_tree, &game, 2);
    let prof = gpu_solver.run_profiled(ctx, flop_tree, &game, n_iters);
    (prof, actual_nh, n_pairs)
}

/// Time the ASYNC-mode run() path (Fix C orchestration). run_profiled
/// forces sync mode for per-stage timing; this measures the actual
/// production hot-path wall-clock.
fn measure_async_at_nh(
    ctx: &MetalContext,
    flop_tree: &FlatTree,
    canonical: [Card; 3],
    n_hands_target: usize,
    n_iters: u32,
) -> (f64, usize, usize) {
    use std::time::Instant;
    let full = make_combo_ranges(canonical);
    let board: Vec<Card> = canonical.iter().copied().collect();

    let table = if n_hands_target >= 1000 {
        FlopChanceTable::compute_flop_start(&board, &full, 2)
    } else {
        let (chosen, turn_cards, river_decks) =
            pick_subset_n_hands(canonical, n_hands_target);
        FlopChanceTable::compute_flop_start_subset_with_decks(
            &board, &full, 2, &chosen, &turn_cards, &river_decks)
    };
    let actual_nh = table.num_valid;
    let n_pairs: usize = table.river_decks.iter().map(|d| d.len()).sum();
    let game = FlopStartGame::new(table);
    let cpu_solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let mut gpu_solver = MetalFlopStartSolver::new(ctx, flop_tree, &game, &cpu_solver);

    // Warmup (cold pipelines).
    gpu_solver.run(ctx, flop_tree, &game, 2);
    let t = Instant::now();
    gpu_solver.run(ctx, flop_tree, &game, n_iters);
    let secs = t.elapsed().as_secs_f64();
    (secs * 1000.0 / n_iters as f64, actual_nh, n_pairs)
}

#[test]
#[ignore = "Step 2.D.24: nh-scaling probe at fixed n_pairs=1, find scaling exponent"]
fn step2d24_gpu_nh_scaling_probe() {
    let flop_tree = build_tiny_flop_tree();
    eprintln!("\n=== Step 2.D.24: GPU per-iter time vs nh ===");
    eprintln!("Tree: {} nodes (HU 1+1 stacks=10). n_pairs=1 fixed.", flop_tree.num_nodes());
    eprintln!("Goal: find scaling exponent (linear, quadratic, worse).");
    eprintln!();

    let ctx = MetalContext::new().expect("Metal");
    let canonical: [Card; 3] = [12, 16, 20];

    // Note: n_hands targets are aspirational — actual nh = number of
    // mutually-non-blocking hands available (max ~12 since each pair uses 2 cards).
    // We'll just sweep with non-subset compute_flop_start at the high end to
    // get full nh, and use a different approach for intermediate nh.
    //
    // Subset path is limited by mutual-non-blocking: each hand uses 2 unique
    // cards, so max ~22 hands fit (50 non-board / 2 = 25, minus overlap).
    // For larger nh we need non-mutual-blocking subset OR full.
    //
    // To get a clean nh scaling, use FULL compute_flop_start with reduced
    // turn/river deck (1 turn × 1 river per turn) to fix n_pairs=1.
    //
    // But subset_with_decks requires `chosen` list which limits hands.
    // Workaround: use full nh + small deck → can't, deck is full when using
    // compute_flop_start.
    //
    // Easier path for nh sweep: use subset_with_decks with VARYING chosen
    // hand counts, allowing OVERLAPPING hands (relax the mutual-block).
    // The CFR will still compute correctly; just the conflict matrix will
    // have many blocking entries (which is realistic).

    let nh_targets: Vec<usize> = vec![8, 16, 24, 50, 100, 200];
    let mut results: Vec<(usize, usize, f64, StageProfile, f64)> = Vec::new();
    eprintln!("\n{:>5}  {:>10}  {:>10}  {:>14}", "nh", "sync ms/it", "async ms/it", "speedup (×)");
    for &nh_target in &nh_targets {
        let (prof, actual_nh, n_pairs) = measure_at_nh(&ctx, &flop_tree, canonical, nh_target, 10);
        let sync_ms = prof.total.as_secs_f64() * 1000.0 / 10.0;
        let (async_ms, _, _) = measure_async_at_nh(&ctx, &flop_tree, canonical, nh_target, 10);
        let speedup = sync_ms / async_ms.max(1e-9);
        eprintln!("{:>5}  {:>10.3}  {:>10.3}  {:>14.2}",
            actual_nh, sync_ms, async_ms, speedup);
        let _ = n_pairs;
        results.push((actual_nh, n_pairs, sync_ms, prof, async_ms));
    }

    // Also measure full nh (no subset) at 1 turn × 1 river — fall back to
    // separate config since the subset path may cap nh.
    eprintln!("\n── async-mode scaling (Fix C orchestration) ──");
    for i in 1..results.len() {
        let nh1 = results[i-1].0;
        let nh2 = results[i].0;
        let t1 = results[i-1].4;
        let t2 = results[i].4;
        if nh1 == nh2 { continue; }
        let slope = (t2.ln() - t1.ln()) / ((nh2 as f64).ln() - (nh1 as f64).ln());
        eprintln!("  nh {} → {}: per-iter {:.2}ms → {:.2}ms, slope = {:.2}",
            nh1, nh2, t1, t2, slope);
    }

    // Per-stage breakdown for largest nh — shows which stage scales.
    let largest = &results.last().unwrap().clone();
    let total = largest.3.total.as_secs_f64();
    eprintln!("\n── per-stage breakdown at largest nh={} ──", largest.0);
    let stages: Vec<(&str, f64)> = vec![
        ("compute_strategies",       largest.3.compute_strategies.as_secs_f64()),
        ("compute_reach_flop",       largest.3.compute_reach_flop.as_secs_f64()),
        ("compute_reach_turn",       largest.3.compute_reach_turn.as_secs_f64()),
        ("compute_reach_river",      largest.3.compute_reach_river.as_secs_f64()),
        ("bottom_up_river",          largest.3.bottom_up_river.as_secs_f64()),
        ("bottom_up_turn",           largest.3.bottom_up_turn.as_secs_f64()),
        ("bottom_up_flop",           largest.3.bottom_up_flop.as_secs_f64()),
        ("chance_accum_river",       largest.3.chance_accumulate_river.as_secs_f64()),
        ("chance_finalize_river",    largest.3.chance_finalize_river.as_secs_f64()),
        ("chance_accum_turn",        largest.3.chance_accumulate_turn.as_secs_f64()),
        ("zero_buffer_total",        largest.3.zero_buffer_total.as_secs_f64()),
    ];
    let mut sorted = stages.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, s) in sorted.iter().take(8) {
        let pct = s / total.max(1e-12) * 100.0;
        eprintln!("  {:30} {:>10.4} s ({:>5.1}%)", name, s, pct);
    }
}
