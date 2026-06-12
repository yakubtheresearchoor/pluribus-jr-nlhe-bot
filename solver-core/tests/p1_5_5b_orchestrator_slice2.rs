//! P1.5.5b Slice 2: stratified-subset orchestrator smoke test.
//!
//! Composes Slice 1's per-canonical V_flop with P1.5.5a's reduce_cfv +
//! P1.5.4's aggregate_preflop_chance to compute the preflop class CFV
//! at the chance node — the full orchestration shape, but over a
//! STRATIFIED SUBSET of canonical flops rather than all 1,755, to keep
//! wall-clock tractable for CI.
//!
//! Stratification: covers all (orbit_size × rank_shape) cells,
//! n_per_cell = 3, giving ~9-cell × 3 = ~27 canonicals. Includes trip
//! flops (KKK shape) and paired flops (KK7 shape) where the per-
//! (class, F) survivor count is most stressed (drives some class
//! expansions to 0, exercises blocked-class handling in reduce/expand).
//!
//! This is a SMOKE test (Slice 2 wiring), not a max_diff anchor —
//! Slice 3 (P2.5b) does the un-canonicalized reference comparison.
//! Slice 2 verifies the orchestration runs end-to-end and produces
//! sane output.

use solver_core::abstraction::preflop_class::{class_combos, PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::preflop_start_game::{
    aggregate_preflop_chance_subset, compute_v_flop_at_root_iter0,
    flop_rank_shape, reduce_cfv_combo_to_class, stratified_canonical_subset,
    FlopRankShape, PreflopChanceTable,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn build_flop_tree() -> solver_core::tree::flat::FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 10,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![0, 0],
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
    build_tree(&cfg).expect("flop tree builds")
}

/// Map class-level reach to combo-level reach for one canonical, in the
/// full 1326-indexed layout that FlopChanceTable::compute_flop_start
/// expects. This uses the lossless 169 expand factor 1/|expansion|
/// (each class's reach is uniformly distributed among its surviving
/// combos).
fn expand_class_reach_to_full_combo_layout(
    canonical_flop: [Card; 3],
    class_reach: &[f32],
) -> Vec<f32> {
    use solver_core::card::index_to_card_pair;
    let mut combo_reach = vec![0.0f32; NUM_POSSIBLE_HANDS];
    // For each class, find its surviving combos at this flop, distribute reach
    for c_idx in 0..NUM_PREFLOP_CLASSES {
        let class = PreflopClass(c_idx as u8);
        let combos = class_combos(class);
        let surviving: Vec<(Card, Card)> = combos.into_iter()
            .filter(|&(c1, c2)| !canonical_flop.contains(&c1) && !canonical_flop.contains(&c2))
            .collect();
        if surviving.is_empty() { continue; }
        let per_combo = class_reach[c_idx] / surviving.len() as f32;
        for (c1, c2) in surviving {
            // Find the index of (c1, c2) in the 1326-combo layout.
            // The standard pair_idx formula matches index_to_card_pair's inverse.
            let pair_idx = card_pair_to_idx(c1, c2);
            combo_reach[pair_idx] = per_combo;
        }
    }
    let _ = index_to_card_pair; // suppress unused-import warning if any
    combo_reach
}

/// Inverse of `index_to_card_pair`. The pairing formula (with c1 < c2)
/// is `c1 * (101 - c1) / 2 + c2 - 1`.
fn card_pair_to_idx(a: Card, b: Card) -> usize {
    let (c1, c2) = if a < b { (a, b) } else { (b, a) };
    (c1 as usize) * (101 - c1 as usize) / 2 + (c2 as usize) - 1
}

#[test]
fn slice2_orchestration_runs_on_stratified_subset() {
    let table = PreflopChanceTable::new(
        2, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );

    // Stratified subset: 3 per (orbit_size × rank_shape) cell.
    // Expected ~9 cells × 3 = ~27 canonicals.
    let subset = stratified_canonical_subset(&table, 3);

    eprintln!("\n=== P1.5.5b Slice 2: stratified-subset orchestrator smoke ===");
    eprintln!("Subset size: {} canonicals", subset.len());

    // Report stratification coverage
    use std::collections::BTreeMap;
    let mut cell_counts: BTreeMap<(u32, FlopRankShape), usize> = BTreeMap::new();
    for &idx in &subset {
        let orbit = table.orbit_sizes[idx];
        let shape = flop_rank_shape(table.canonical_flops[idx]);
        *cell_counts.entry((orbit, shape)).or_insert(0) += 1;
    }
    eprintln!("Stratification coverage:");
    for ((orbit, shape), count) in &cell_counts {
        eprintln!("  orbit={} shape={:?} → {} canonicals", orbit, shape, count);
    }

    // Verify coverage: must include all three rank shapes
    let shapes_present: std::collections::BTreeSet<FlopRankShape> = subset.iter()
        .map(|&i| flop_rank_shape(table.canonical_flops[i]))
        .collect();
    assert!(shapes_present.contains(&FlopRankShape::Rainbow),
        "stratified subset must include Rainbow flops");
    assert!(shapes_present.contains(&FlopRankShape::Paired),
        "stratified subset must include Paired flops (stress survivor count)");
    assert!(shapes_present.contains(&FlopRankShape::Trip),
        "stratified subset must include Trip flops (drive class survivors to 0)");

    let orbit_sizes_present: std::collections::BTreeSet<u32> = subset.iter()
        .map(|&i| table.orbit_sizes[i])
        .collect();
    for expected_orbit in [4u32, 12, 24] {
        assert!(orbit_sizes_present.contains(&expected_orbit),
            "stratified subset must include orbit size {} canonicals", expected_orbit);
    }

    // Build a HU flop tree
    let flop_tree = build_flop_tree();

    // Uniform class-level reach for both players
    let class_reach = vec![1.0f32; NUM_PREFLOP_CLASSES];

    // For each canonical in subset: expand class reach → combo reach
    // (in full 1326 layout) → run per-flop solver → get V_combo at
    // flop root → reduce to V_class
    let t0 = std::time::Instant::now();
    let mut flop_cfvs_subset: Vec<Vec<f32>> = Vec::with_capacity(subset.len());
    for &canonical_idx in &subset {
        let f_canon = table.canonical_flops[canonical_idx];
        let combo_reach = expand_class_reach_to_full_combo_layout(f_canon, &class_reach);
        let ranges = vec![combo_reach.clone(), combo_reach];

        let (v_combo, layout) = compute_v_flop_at_root_iter0(
            f_canon, &flop_tree, &ranges, 0,
        );
        // Reduce V_combo to V_class using the per-flop layout
        let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
        flop_cfvs_subset.push(v_class);
    }
    let solve_time = t0.elapsed();
    eprintln!("Per-canonical solves complete: {:?} for {} canonicals",
        solve_time, subset.len());

    // Aggregate via orbit-weighted partial sum
    let preflop_class_cfv = aggregate_preflop_chance_subset(
        &table, &subset, &flop_cfvs_subset,
    );

    // Sanity: output is per-class, all finite, magnitudes bounded
    assert_eq!(preflop_class_cfv.len(), NUM_PREFLOP_CLASSES);
    let mut n_nan = 0;
    let mut n_inf = 0;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for (c, &v) in preflop_class_cfv.iter().enumerate() {
        if v.is_nan() {
            n_nan += 1;
            eprintln!("  class {} V = NaN", c);
        } else if v.is_infinite() {
            n_inf += 1;
        } else {
            if v < min_v { min_v = v; }
            if v > max_v { max_v = v; }
        }
    }
    eprintln!("Preflop class CFV stats (partial-sum over {} canonicals):", subset.len());
    eprintln!("  min={}, max={}, n_nan={}, n_inf={}", min_v, max_v, n_nan, n_inf);

    assert_eq!(n_nan, 0, "found {} NaN in preflop class CFV", n_nan);
    assert_eq!(n_inf, 0, "found {} Inf in preflop class CFV", n_inf);

    // The partial sum should be roughly proportional to the subset's
    // total chance probability. Each class's full sum (over all 1,755
    // canonicals) would equal "average V over all flops weighted by
    // class probability." On the subset, it's the partial.
    //
    // Bound: |V_class[c]| ≤ max_per_canonical_V × max_chance_prob ×
    // subset_size. With game pot ≤ 210, V per canonical bounded by
    // ~210, and Σ_F P(F|c) over subset ≤ Σ_F P(F|c) over all ≤ 1.0,
    // so |V_class[c]| ≤ 210. Use 1000 as generous bound.
    let max_abs = min_v.abs().max(max_v.abs());
    assert!(max_abs < 1000.0,
        "max |preflop class CFV| = {} too large for HU game with pot ≤ 210", max_abs);

    eprintln!("✓ Slice 2: orchestration runs end-to-end on stratified subset");
}
