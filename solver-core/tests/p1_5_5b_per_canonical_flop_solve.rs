//! P1.5.5b Slice 1 smoke test: per-canonical real V_flop computation.
//!
//! This exercises `compute_v_flop_at_root_iter0` on a single canonical
//! flop, verifying the wiring works end-to-end:
//!   - FlopChanceTable builds from the canonical flop
//!   - FlopStartVectorCfr instantiates on the flop-rooted tree + table
//!   - compute_reach_flop + bottom_up_zone(River/Turn/Flop) run without
//!     panic
//!   - Root CFV is returned with the expected shape (length = per-flop nh)
//!   - CFV values are finite and non-trivial (not all zero, not NaN/Inf)
//!
//! What this does NOT test (deferred to later slices):
//!   - Scaling to 1,755 canonicals (Slice 2)
//!   - Orbit-weighted aggregation against an independent reference
//!     (Slice 3 = P2.5b)
//!   - Correctness of CFV values against an independent computation
//!     (the per-flop solver is anchored elsewhere; this slice trusts
//!     it and just confirms we call it correctly)

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::preflop_start_game::compute_v_flop_at_root_iter0;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn build_simple_flop_tree() -> solver_core::tree::flat::FlatTree {
    // HU flop-rooted tree: pot 5+5 = 10, both players have 100 chips
    // remaining. Simple bet sizing for a manageable tree size.
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 10,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    };
    build_tree(&cfg).expect("flop tree builds")
}

#[test]
fn per_canonical_v_flop_wires_through_existing_solver() {
    // Pick a representative canonical flop: 2c, 7d, Ks (rainbow,
    // non-paired, non-connected — generic "dry" flop).
    let flop: [Card; 3] = [
        card_from_str("2c").unwrap() as Card,
        card_from_str("7d").unwrap() as Card,
        card_from_str("Ks").unwrap() as Card,
    ];

    eprintln!("\n=== P1.5.5b Slice 1: per-canonical V_flop wiring smoke test ===");
    eprintln!("Canonical flop: 2c 7d Ks");

    let flop_tree = build_simple_flop_tree();
    eprintln!("Flop tree: {} nodes", flop_tree.num_nodes());

    // Uniform combo ranges for both players (full range, weight 1.0 per
    // combo). The expand step at this canonical would normally produce
    // these from class-level uniform reach; for the slice 1 smoke test
    // we just use uniform combo ranges directly.
    let ranges_per_player = vec![
        vec![1.0f32; NUM_POSSIBLE_HANDS],
        vec![1.0f32; NUM_POSSIBLE_HANDS],
    ];

    let t0 = std::time::Instant::now();
    let (v_combo, layout) = compute_v_flop_at_root_iter0(
        flop, &flop_tree, &ranges_per_player, 0,
    );
    let elapsed = t0.elapsed();

    eprintln!("Per-canonical solve: {} combos × {} traverser=0, took {:?}",
        layout.len(), v_combo.len(), elapsed);

    // Sanity: output shape matches per-flop nh (combos compatible with flop).
    // For a typical 3-distinct-card flop, per-flop nh = 1326 - 150 = 1176.
    let expected_nh = 1176;
    assert_eq!(layout.len(), expected_nh,
        "layout length should be 1176 for a generic 3-card flop, got {}", layout.len());
    assert_eq!(v_combo.len(), expected_nh,
        "v_combo length should match layout length");

    // Sanity: combos in layout do NOT include any flop card.
    for &(c1, c2) in &layout {
        assert!(!flop.contains(&c1), "layout combo ({}, {}) uses flop card {}", c1, c2, c1);
        assert!(!flop.contains(&c2), "layout combo ({}, {}) uses flop card {}", c1, c2, c2);
    }

    // Sanity: CFV values are finite (no NaN, no Inf).
    let mut n_nan = 0;
    let mut n_inf = 0;
    let mut n_zero = 0;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in &v_combo {
        if v.is_nan() { n_nan += 1; }
        else if v.is_infinite() { n_inf += 1; }
        else {
            if v == 0.0 { n_zero += 1; }
            if v < min_v { min_v = v; }
            if v > max_v { max_v = v; }
        }
    }
    eprintln!("CFV stats: min={}, max={}, n_nan={}, n_inf={}, n_zero={}/{}",
        min_v, max_v, n_nan, n_inf, n_zero, v_combo.len());
    assert_eq!(n_nan, 0, "found {} NaN in V_combo — solver produced invalid values", n_nan);
    assert_eq!(n_inf, 0, "found {} Inf in V_combo — solver produced invalid values", n_inf);

    // Sanity: not all values are zero (would indicate broken solver).
    assert!(n_zero < v_combo.len(),
        "all V_combo values are zero — solver is returning trivial output");

    // Sanity: CFV magnitudes are bounded (the game is HU with pot ≤ 210
    // chips, so per-hand CFV should be ≤ a few hundred in magnitude).
    let max_abs = min_v.abs().max(max_v.abs());
    assert!(max_abs < 10_000.0,
        "max |V_combo| = {} is unreasonably large (game pot ≤ 210)", max_abs);

    eprintln!("✓ Per-canonical V_flop wiring works end-to-end");
}

#[test]
fn per_canonical_v_flop_is_deterministic() {
    // Two calls with the same inputs must return bit-identical outputs.
    let flop: [Card; 3] = [
        card_from_str("2c").unwrap() as Card,
        card_from_str("7d").unwrap() as Card,
        card_from_str("Ks").unwrap() as Card,
    ];
    let flop_tree = build_simple_flop_tree();
    let ranges = vec![
        vec![1.0f32; NUM_POSSIBLE_HANDS],
        vec![1.0f32; NUM_POSSIBLE_HANDS],
    ];

    let (v1, l1) = compute_v_flop_at_root_iter0(flop, &flop_tree, &ranges, 0);
    let (v2, l2) = compute_v_flop_at_root_iter0(flop, &flop_tree, &ranges, 0);

    assert_eq!(l1, l2, "layout must be deterministic");
    assert_eq!(v1.len(), v2.len(), "v_combo length must be deterministic");
    for i in 0..v1.len() {
        assert_eq!(v1[i].to_bits(), v2[i].to_bits(),
            "v_combo[{}] differs across deterministic calls: {} vs {}",
            i, v1[i], v2[i]);
    }
    eprintln!("✓ Per-canonical V_flop is deterministic across repeated calls");
}
