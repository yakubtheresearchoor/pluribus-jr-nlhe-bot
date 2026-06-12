// Step 2.D.8b: isolate the cost components.
//
// The original 2.D.8 measurement was going to conflate two different
// costs:
//   (a) Per-canonical SETUP (build FlopChanceTable, build MetalFlopStartSolver,
//       allocate GPU buffers).
//   (b) Per-postflop-iter WORK (one solver.run iteration).
//
// (a) is architectural overhead that a real production blueprint would
// hoist (pool solvers, cache tables). (b) is the irreducible GPU work.
// Reporting their sum at K=10/20/40 gives a wall-clock number dominated
// by (a) — which is "garbage" as the user correctly flagged. The number
// that matters for #96 abstraction sizing is (a) amortized + (b) per
// preflop-iter × chance × traverser × canonical.
//
// THIS TEST measures (a) and (b) separately on the production GPU
// artifact (MetalFlopStartSolver), so the projection is honest about
// what's fixable architecture vs irreducible work.

#![cfg(feature = "metal")]

use std::time::Instant;

use solver_core::card::{card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
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
    // Asymmetric per-combo ranges. Same shape both players for simplicity;
    // the cost is range-shape-independent.
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

#[test]
#[ignore = "Step 2.D.8b: isolate per-canonical setup from per-iter work"]
fn step2d8b_isolated_costs() {
    let flop_tree = build_tiny_flop_tree();
    eprintln!("\n=== Step 2.D.8b: isolated cost components ===");
    eprintln!("Flop tree: {} nodes (tiny postflop: stacks=10, 1 bet size)", flop_tree.num_nodes());
    eprintln!("Subset: 8 hands × 2 turn × 2 river per canonical");

    let ctx = MetalContext::new().expect("Metal");

    // Use 3 distinct canonicals — average across them so we don't fixate
    // on one canonical's specific cost.
    let canonicals: Vec<[Card; 3]> = vec![
        [0, 4, 8],   // 2c, 3c, 4c
        [12, 16, 20], // 5c, 6c, 7c
        [24, 28, 32], // 8c, 9c, Tc
    ];

    // ── (a) Per-canonical SETUP ──
    eprintln!("\n--- (a) Per-canonical SETUP (one-time work; hoistable) ---");
    let mut setup_costs: Vec<f64> = Vec::new();
    let mut precomputed: Vec<(MetalFlopStartSolver, FlopStartGame, FlatTree)> = Vec::new();
    for &canonical in &canonicals {
        let full = make_combo_ranges(canonical);
        let board: Vec<Card> = canonical.iter().copied().collect();
        let (chosen, turn_cards, river_decks) = pick_subset(canonical);

        // Setup = table build + game + solver + GPU buffer alloc.
        let t = Instant::now();
        let table = FlopChanceTable::compute_flop_start_subset_with_decks(
            &board, &full, 2, &chosen, &turn_cards, &river_decks);
        let game = FlopStartGame::new(table);
        let cpu_solver = FlopStartVectorCfr::new(&flop_tree, game.table());
        let gpu_solver = MetalFlopStartSolver::new(&ctx, &flop_tree, &game, &cpu_solver);
        let setup_secs = t.elapsed().as_secs_f64();
        setup_costs.push(setup_secs);
        eprintln!("  canonical {:?}: setup = {:.3} s ({:.0} ms)", canonical, setup_secs, setup_secs * 1000.0);
        precomputed.push((gpu_solver, game, flop_tree.clone()));
    }
    let avg_setup = setup_costs.iter().sum::<f64>() / setup_costs.len() as f64;
    eprintln!("  → avg per-canonical setup: {:.3} s ({:.0} ms)", avg_setup, avg_setup * 1000.0);

    // ── (b) Per postflop-iter WORK on pre-built solver ──
    eprintln!("\n--- (b) Per postflop-iter WORK (irreducible GPU work) ---");
    // For each pre-built solver, time various postflop iter counts and
    // compute slope.
    for (ci, (gpu_solver, game, ft)) in precomputed.iter_mut().enumerate() {
        let canonical = canonicals[ci];

        // Time N postflop iters. Use N1 vs N2 to compute per-iter cost
        // separately from any per-run overhead.
        let n1 = 10u32;
        let n2 = 50u32;

        let t = Instant::now();
        gpu_solver.run(&ctx, ft, game, n1);
        let secs_n1 = t.elapsed().as_secs_f64();

        // Reconstruct solver for clean N2 measurement (run() mutates state).
        let game2 = FlopStartGame::new(FlopChanceTable::compute_flop_start_subset_with_decks(
            &canonical.iter().copied().collect::<Vec<_>>(),
            &make_combo_ranges(canonical),
            2,
            &pick_subset(canonical).0,
            &pick_subset(canonical).1,
            &pick_subset(canonical).2,
        ));
        let cpu_solver2 = FlopStartVectorCfr::new(ft, game2.table());
        let mut gpu_solver2 = MetalFlopStartSolver::new(&ctx, ft, &game2, &cpu_solver2);

        let t = Instant::now();
        gpu_solver2.run(&ctx, ft, &game2, n2);
        let secs_n2 = t.elapsed().as_secs_f64();

        let per_iter_secs = (secs_n2 - secs_n1) / (n2 - n1) as f64;
        let run_overhead = secs_n1 - per_iter_secs * n1 as f64;
        eprintln!("  canonical {:?}: run({}) = {:.3}s, run({}) = {:.3}s",
            canonical, n1, secs_n1, n2, secs_n2);
        eprintln!("    → per postflop-iter: {:.4} s ({:.1} ms), per-run overhead: {:.4}s ({:.1} ms)",
            per_iter_secs, per_iter_secs * 1000.0, run_overhead, run_overhead * 1000.0);
    }

    // ── Projection. ──
    eprintln!("\n--- Projection ---");
    eprintln!("Per-canonical SETUP (avg): {:.0} ms", avg_setup * 1000.0);
    eprintln!();
    eprintln!("If setup is hoisted (build 1755 solvers once at startup, reuse them");
    eprintln!("via run()), the production-blueprint cost is:");
    eprintln!("  startup_cost     = 1755 × per_canonical_setup");
    eprintln!("  per_iter_cost    = 1755 × n_chance × n_traverser × per_run({} postflop iters)", 50);
    eprintln!("  total            = startup + n_preflop_iters × per_iter_cost");
    eprintln!();
    eprintln!("Without hoisting (rebuild solver every call as the current architecture does):");
    eprintln!("  per_call_cost    = setup + per_run({} postflop iters)", 50);
    eprintln!("  per_preflop_iter = 1755 × n_chance × n_traverser × per_call_cost");
    eprintln!("  total            = n_preflop_iters × per_preflop_iter");
    eprintln!();
    eprintln!("Per-canonical SETUP is the ENTIRE difference between hoisted and naive.");
    eprintln!("If setup >> per-iter, naive measurement is meaningless for sizing the abstraction;");
    eprintln!("hoisted measurement is the real cost framing.");
}
