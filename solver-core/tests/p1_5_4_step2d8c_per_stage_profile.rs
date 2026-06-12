// Step 2.D.8c: profile what's dominating the 25 ms-per-iter cost.
//
// Use the existing MetalFlopStartSolver::run_profiled + StageProfile to
// break the per-iter time down by stage. Two questions:
//
//   (i)  Is the 25 ms launch-overhead-bound or compute-bound? If
//        launch-overhead-bound, production-scale workload won't be
//        proportionally worse (more math fits inside fewer dispatches).
//   (ii) Which stages dominate? That points where the optimization
//        would have to land if we wanted to compress wall-clock without
//        the abstraction.

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

fn pick_subset(canonical: [Card; 3]) -> (Vec<u16>, Vec<u8>, Vec<Vec<u8>>) {
    let board_mask: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut chosen: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = solver_core::card::index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        chosen.push(idx as u16);
        if chosen.len() == 8 { break; }
    }
    let mut hand_mask = board_mask;
    for &i in &chosen {
        let (c1, c2) = solver_core::card::index_to_card_pair(i as usize);
        hand_mask |= 1u64 << c1;
        hand_mask |= 1u64 << c2;
    }
    let mut turn_cards: Vec<u8> = Vec::new();
    for c in 0u8..52u8 {
        if hand_mask & (1u64 << c) != 0 { continue; }
        turn_cards.push(c);
        if turn_cards.len() == 2 { break; }
    }
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let mut rivers: Vec<u8> = Vec::new();
        for c in 0u8..52u8 {
            if hand_mask & (1u64 << c) != 0 { continue; }
            if c == tc { continue; }
            rivers.push(c);
            if rivers.len() == 2 { break; }
        }
        river_decks[tc as usize] = rivers;
    }
    (chosen, turn_cards, river_decks)
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

fn build_solver(
    ctx: &MetalContext, flop_tree: &FlatTree, canonical: [Card; 3],
) -> MetalFlopStartSolver {
    let full = make_combo_ranges(canonical);
    let board: Vec<Card> = canonical.iter().copied().collect();
    let (chosen, turn_cards, river_decks) = pick_subset(canonical);
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &full, 2, &chosen, &turn_cards, &river_decks);
    let game = FlopStartGame::new(table);
    let cpu_solver = FlopStartVectorCfr::new(flop_tree, &game.table());
    MetalFlopStartSolver::new(ctx, flop_tree, &game, &cpu_solver)
}

fn make_game(canonical: [Card; 3]) -> FlopStartGame {
    let full = make_combo_ranges(canonical);
    let board: Vec<Card> = canonical.iter().copied().collect();
    let (chosen, turn_cards, river_decks) = pick_subset(canonical);
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &full, 2, &chosen, &turn_cards, &river_decks);
    FlopStartGame::new(table)
}

#[test]
#[ignore = "Step 2.D.8c: per-stage profile of the 25 ms-per-iter cost"]
fn step2d8c_per_stage_profile() {
    let flop_tree = build_tiny_flop_tree();
    eprintln!("\n=== Step 2.D.8c: per-stage profile ===");
    eprintln!("Tiny postflop config: 257-node tree, 8 hands × 4 pairs subset.");

    let ctx = MetalContext::new().expect("Metal");
    let canonical: [Card; 3] = [12, 16, 20];

    // Warmup run (first run includes pipeline-compile + cold caches).
    {
        let mut s = build_solver(&ctx, &flop_tree, canonical);
        let g = make_game(canonical);
        let _ = s.run_profiled(&ctx, &flop_tree, &g, 5);
    }

    // Measure at three iter counts so we can separate per-iter cost from
    // fixed per-run overhead within the GPU solver itself.
    for &n_iters in &[10u32, 50, 200] {
        let mut s = build_solver(&ctx, &flop_tree, canonical);
        let g = make_game(canonical);
        let prof: StageProfile = s.run_profiled(&ctx, &flop_tree, &g, n_iters);
        eprintln!("\n──── n_iters = {} ────", n_iters);
        eprintln!("{}", prof.report());
        let per_iter = prof.total.as_secs_f64() / n_iters as f64;
        eprintln!("  per-iter (total/n_iters): {:.4} s ({:.2} ms)",
            per_iter, per_iter * 1000.0);
    }

    // The launch-overhead-bound signal: if per-iter cost is roughly
    // CONSTANT across n_iters ∈ {10, 50, 200}, the work is compute-bound
    // and production scale will pay the same ms-per-iter.
    // If per-iter DROPS sharply as n_iters increases, there's a fixed
    // per-run overhead amortizing — production scale would benefit.
    //
    // Per-stage percentages tell where the time goes: if compute_strategies
    // / reach / bottom_up are each a few ms with HUGE unattributed
    // remainder, the cost is host-side orchestration (command-buffer setup,
    // dispatch latency), and the GPU is bored. If stages saturate ~total,
    // the GPU is doing real work — fixable only by tree-size reduction
    // (which is what the abstraction provides).
    eprintln!("\nDIAGNOSTIC NOTES:");
    eprintln!("  - If per-iter is flat across n_iters → compute-bound (real GPU work).");
    eprintln!("  - If per-iter drops with n_iters → fixed overhead amortizes.");
    eprintln!("  - If 'unattributed' >> stages → host-orchestration-bound, not GPU-bound.");
    eprintln!("    Larger production workload would not make this worse proportionally;");
    eprintln!("    real per-iter at production scale could be similar or only modestly higher.");
}
