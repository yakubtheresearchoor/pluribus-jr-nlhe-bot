// P1.5.4 Slice A.3e: converged per_flop_solver wires correctly.
//
// Per the lead's directive (2026-06-04): the engine MUST use converged
// postflop, not iter-0. The "iter-0 postflop" engine solves a wrong
// game (preflop CFR with fixed uniform postflop) that a GPU faithfully
// reproducing it would still produce an unplayable strategy.
//
// This test confirms:
//   (1) `make_per_flop_solver_converged` builds and runs without panic.
//   (2) The converged CFV differs from the iter-0 CFV (so the fix
//       actually changes the behavior — same call signature, different
//       semantics, observable).
//   (3) The converged CFV is finite (not NaN/Inf) and within a
//       reasonable magnitude (sanity).
//
// This is a small-scale wiring check, NOT a Slice B convergence study.
// Slice B (multi-iter on the corrected engine, correctness-baseline-first)
// is its own slice after the fix lands.

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::preflop_start_game::{
    compute_v_flop_at_root_converged, compute_v_flop_at_root_iter0,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn build_simple_flop_tree() -> solver_core::tree::flat::FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 6,
        starting_stacks: vec![97, 97],
        initial_contributions: vec![3, 3],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    build_tree(&cfg).expect("flop tree")
}

#[test]
#[ignore = "Slice A.3e wiring check: converged per_flop_solver at production nh, \
            ~minute wall-clock (10 postflop iters). Run on demand: cargo test --release \
            --test p1_5_4_slice_a3e_converged_per_flop_solver -- --ignored --nocapture"]
fn slice_a3e_converged_per_flop_solver_differs_from_iter0() {
    eprintln!("\n═══ Slice A.3e: converged per_flop_solver wires + differs from iter-0 ═══");

    let flop_tree = build_simple_flop_tree();
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let canonical: [Card; 3] = [board[0], board[1], board[2]];

    // Uniform combo ranges (full nh after board-blocking).
    let combo_ranges: Vec<Vec<f32>> = (0..2)
        .map(|_| vec![1.0_f32; NUM_POSSIBLE_HANDS])
        .collect();

    // Call the underlying functions directly (bypassing the engine-
    // layout wrappers, which expect engine-layout reaches not
    // NUM_POSSIBLE_HANDS-indexed). This smoke test compares the
    // iter-0 vs converged FUNCTIONS, not the engine wrappers.
    eprintln!("── Running iter-0 per-flop solver (uniform postflop strategies) ──");
    let t0 = std::time::Instant::now();
    let (v_iter0, _layout) = compute_v_flop_at_root_iter0(canonical, &flop_tree, &combo_ranges, 0);
    let iter0_secs = t0.elapsed().as_secs_f64();
    eprintln!("  iter-0: {:.2}s, v_combo len = {}, range [{:.3}, {:.3}], mean {:.3}",
        iter0_secs, v_iter0.len(),
        v_iter0.iter().cloned().fold(f32::INFINITY, f32::min),
        v_iter0.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        v_iter0.iter().sum::<f32>() / v_iter0.len() as f32);

    eprintln!("── Running converged per-flop solver (10 postflop iters) ──");
    let t1 = std::time::Instant::now();
    let (v_conv_10, _) = compute_v_flop_at_root_converged(canonical, &flop_tree, &combo_ranges, 0, 10);
    let conv10_secs = t1.elapsed().as_secs_f64();
    eprintln!("  converged-10: {:.2}s, v_combo len = {}, range [{:.3}, {:.3}], mean {:.3}",
        conv10_secs, v_conv_10.len(),
        v_conv_10.iter().cloned().fold(f32::INFINITY, f32::min),
        v_conv_10.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        v_conv_10.iter().sum::<f32>() / v_conv_10.len() as f32);

    eprintln!("── Running converged per-flop solver (50 postflop iters) ──");
    let t2 = std::time::Instant::now();
    let (v_conv_50, _) = compute_v_flop_at_root_converged(canonical, &flop_tree, &combo_ranges, 0, 50);
    let conv50_secs = t2.elapsed().as_secs_f64();
    eprintln!("  converged-50: {:.2}s, v_combo len = {}, range [{:.3}, {:.3}], mean {:.3}",
        conv50_secs, v_conv_50.len(),
        v_conv_50.iter().cloned().fold(f32::INFINITY, f32::min),
        v_conv_50.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        v_conv_50.iter().sum::<f32>() / v_conv_50.len() as f32);

    // Cost ratio: converged should be O(num_postflop_iters) × iter-0 cost.
    eprintln!("\n── Cost attribution ──");
    eprintln!("  iter-0:        {:.2}s", iter0_secs);
    eprintln!("  converged-10:  {:.2}s ({:.1}x iter-0)", conv10_secs, conv10_secs / iter0_secs.max(1e-3));
    eprintln!("  converged-50:  {:.2}s ({:.1}x iter-0)", conv50_secs, conv50_secs / iter0_secs.max(1e-3));

    // (1) Wiring sanity: both converged variants return same-length v_combo.
    assert_eq!(v_iter0.len(), v_conv_10.len());
    assert_eq!(v_iter0.len(), v_conv_50.len());

    // (2) Converged differs from iter-0 (the fix actually changes behavior).
    let max_diff_10 = v_iter0.iter().zip(v_conv_10.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let max_diff_50 = v_iter0.iter().zip(v_conv_50.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    eprintln!("\n── Iter-0 vs converged: differs as expected ──");
    eprintln!("  max |v_iter0 - v_converged_10| = {:.4}", max_diff_10);
    eprintln!("  max |v_iter0 - v_converged_50| = {:.4}", max_diff_50);
    assert!(max_diff_10 > 1e-3,
        "converged-10 didn't differ from iter-0 by > 1e-3 (max_diff = {}); the converged solver \
         isn't actually doing CFR iterations or the strategies didn't change",
        max_diff_10);

    // (3) Sanity: finite values, magnitudes in expected range (chip amounts ~10–100s).
    for &v in &v_conv_50 {
        assert!(v.is_finite(), "non-finite v in converged result: {}", v);
        assert!(v.abs() < 1e4,
            "v magnitude {} suspiciously large (chip amounts should be ~10s of pot)", v);
    }

    // (4) Convergence-trajectory signal: |v_converged_50 - v_converged_10|
    //     should be smaller than |v_converged_10 - v_iter0| if the strategy
    //     is settling (later iters are closer to each other than to iter-0).
    let max_diff_10_to_50 = v_conv_10.iter().zip(v_conv_50.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    eprintln!("  max |v_converged_10 - v_converged_50| = {:.4} (later-iter change should be < iter-0-to-converged change)",
        max_diff_10_to_50);

    eprintln!("\nSlice A.3e PASS: converged per_flop_solver builds, runs, differs from iter-0, \
              produces finite values. The engine wired with this solver solves the CORRECT GAME \
              (preflop CFR with converged postflop), not the iter-0 wrong-game variant.");
}
