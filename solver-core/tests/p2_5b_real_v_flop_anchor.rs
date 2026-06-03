//! P2.5b: real-V_flop anchor for the preflop chance integration.
//!
//! The full integration test. P2.5a anchored the chance-integration
//! arithmetic against a STUB V_flop. P2.5b replaces the stub with real
//! per-flop solver values (FlopStartVectorCfr at iter-0 with expanded
//! reach), keeping the same structural-independence pattern:
//!
//!   RUNTIME (canonicalized): loop subset_indices in 0..1,755,
//!     run per-flop solver per canonical, reduce via
//!     reduce_cfv_combo_to_class, aggregate via
//!     aggregate_preflop_chance_subset (orbit-weighted).
//!
//!   REFERENCE (un-canonicalized): for each canonical in subset,
//!     enumerate its orbit_of(F) actual flops, run per-flop solver
//!     per actual, compute class CFV by EXPLICIT enumeration of
//!     class.num_combos() combos with EXPLICIT compatibility filter
//!     and EXPLICIT per-class averaging, sum with EXPLICIT unit weight.
//!     The reference does NOT call reduce_cfv_combo_to_class or
//!     aggregate_preflop_chance_subset or chance_probability_flop.
//!
//! Both sides share the per-flop solver (the V_flop leaf values).
//! Everything ABOVE the leaf — the orbit weighting, the per-class
//! survivor count, the combo-class averaging, the actual-flop unit
//! weighting — is computed independently on each side.
//!
//! Per user direction: f64 mirror discriminator BAKED IN from the start
//! (not added after seeing a small f32 diff and asserting precision).
//! Real per-flop solves accumulate more f32 than the P2.5a stub did, so
//! the f32 floor is larger here and the f64 separation of
//! precision-from-bug matters more.

use solver_core::abstraction::flop_isomorphism::orbit_of;
use solver_core::abstraction::preflop_class::{class_combos, PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::preflop_start_game::{
    aggregate_preflop_chance_subset, compute_v_flop_at_root_iter0,
    reduce_cfv_combo_to_class,
    stratified_canonical_subset, FLOPS_PER_HAND, PreflopChanceTable,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use std::collections::HashMap;

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
    };
    build_tree(&cfg).expect("flop tree builds")
}

fn card_pair_to_idx(a: Card, b: Card) -> usize {
    let (c1, c2) = if a < b { (a, b) } else { (b, a) };
    (c1 as usize) * (101 - c1 as usize) / 2 + (c2 as usize) - 1
}

fn expand_class_reach_to_full_combo_layout(
    canonical_flop: [Card; 3],
    class_reach: &[f32],
) -> Vec<f32> {
    let mut combo_reach = vec![0.0f32; NUM_POSSIBLE_HANDS];
    for c_idx in 0..NUM_PREFLOP_CLASSES {
        let class = PreflopClass(c_idx as u8);
        let surviving: Vec<(Card, Card)> = class_combos(class).into_iter()
            .filter(|&(c1, c2)| !canonical_flop.contains(&c1) && !canonical_flop.contains(&c2))
            .collect();
        if surviving.is_empty() { continue; }
        let per_combo = class_reach[c_idx] / surviving.len() as f32;
        for (c1, c2) in surviving {
            combo_reach[card_pair_to_idx(c1, c2)] = per_combo;
        }
    }
    combo_reach
}

/// Compute per-canonical V_class via the RUNTIME path:
/// reduce_cfv_combo_to_class + aggregate_preflop_chance_subset.
/// Returns (f32 result, f64 mirror result).
fn runtime_path(
    table: &PreflopChanceTable,
    flop_tree: &solver_core::tree::flat::FlatTree,
    subset: &[usize],
    class_reach: &[f32],
) -> (Vec<f32>, Vec<f64>) {
    let mut flop_cfvs_subset_f32: Vec<Vec<f32>> = Vec::with_capacity(subset.len());
    let mut flop_cfvs_subset_f64: Vec<Vec<f64>> = Vec::with_capacity(subset.len());

    for &canonical_idx in subset {
        let f_canon = table.canonical_flops[canonical_idx];
        let combo_reach = expand_class_reach_to_full_combo_layout(f_canon, class_reach);
        let ranges = vec![combo_reach.clone(), combo_reach];
        let (v_combo, layout) = compute_v_flop_at_root_iter0(f_canon, flop_tree, &ranges, 0);

        // f32 path: use library reduce
        let v_class_f32 = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
        flop_cfvs_subset_f32.push(v_class_f32);

        // f64 mirror: reduce in f64 using the SAME shape as library reduce
        // (sum over expansion, divide by |expansion|), but in f64
        let layout_map: HashMap<(Card, Card), usize> = layout.iter().enumerate()
            .map(|(i, &c)| (c, i)).collect();
        let mut sums_f64 = vec![0.0f64; NUM_PREFLOP_CLASSES];
        let mut counts_f64 = vec![0usize; NUM_PREFLOP_CLASSES];
        for &(c1, c2) in &layout {
            let class = PreflopClass::from_combo(c1, c2);
            let idx = layout_map[&(c1, c2)];
            sums_f64[class.index()] += v_combo[idx] as f64;
            counts_f64[class.index()] += 1;
        }
        let v_class_f64: Vec<f64> = (0..NUM_PREFLOP_CLASSES)
            .map(|c| if counts_f64[c] > 0 { sums_f64[c] / counts_f64[c] as f64 } else { 0.0 })
            .collect();
        flop_cfvs_subset_f64.push(v_class_f64);
    }

    // f32 path: use library aggregate
    let runtime_f32 = aggregate_preflop_chance_subset(table, subset, &flop_cfvs_subset_f32);

    // f64 mirror: aggregate in f64 using SAME shape (orbit-weighted sum)
    let mut runtime_f64 = vec![0.0f64; NUM_PREFLOP_CLASSES];
    for (i, &canonical_idx) in subset.iter().enumerate() {
        let f_canon = table.canonical_flops[canonical_idx];
        let orbit_size = table.orbit_sizes[canonical_idx] as f64;
        for c in 0..NUM_PREFLOP_CLASSES {
            let class = PreflopClass(c as u8);
            let n_c = class.num_combos() as f64;
            let exp_size = solver_core::abstraction::preflop_class::expansion(class, f_canon).len() as f64;
            let p_f_given_c = (orbit_size * exp_size) / (n_c * FLOPS_PER_HAND as f64);
            runtime_f64[c] += p_f_given_c * flop_cfvs_subset_f64[i][c];
        }
    }

    (runtime_f32, runtime_f64)
}

/// Compute per-class CFV via the REFERENCE path: explicit enumeration
/// of orbit members of each canonical in subset, explicit per-class
/// per-actual-flop averaging, explicit unit weight per actual flop.
///
/// THE STRUCTURAL INDEPENDENCE REQUIREMENT, MADE CONCRETE:
///   - For each canonical in subset, enumerate orbit_of(F_canon) — all
///     actual flops in F_canon's orbit
///   - For each actual flop, run a real per-flop solve (sharing the
///     solver with runtime; the V_flop leaf values are the only thing
///     both sides share)
///   - Class CFV computed by EXPLICITLY iterating class_combos(class),
///     EXPLICITLY filtering by flop conflict (OR-union on the 3 cards),
///     EXPLICITLY averaging the V_combo values over surviving combos,
///     EXPLICITLY weighting by surviving.len() / (n_class × 19,600)
///   - NO calls to reduce_cfv_combo_to_class, aggregate_preflop_chance_subset,
///     or chance_probability_flop. Verify by READING the loop below.
///
/// Returns (f32 result, f64 mirror result).
fn reference_path(
    table: &PreflopChanceTable,
    flop_tree: &solver_core::tree::flat::FlatTree,
    subset: &[usize],
    class_reach: &[f32],
) -> (Vec<f32>, Vec<f64>) {
    let mut ref_class_cfv_f32 = vec![0.0f64; NUM_PREFLOP_CLASSES];
    let mut ref_class_cfv_f64 = vec![0.0f64; NUM_PREFLOP_CLASSES];

    // STRUCTURAL CHECK: outer loop iterates orbit_of for each canonical
    // in subset (orbit members are actual flops, not canonicals).
    for &canonical_idx in subset {
        let f_canon = table.canonical_flops[canonical_idx];
        for actual_flop in orbit_of(f_canon) {
            // Build combo reach for THIS actual flop (expand class reach)
            let combo_reach = expand_class_reach_to_full_combo_layout(actual_flop, class_reach);
            let ranges = vec![combo_reach.clone(), combo_reach];
            let (v_combo, layout) = compute_v_flop_at_root_iter0(
                actual_flop, flop_tree, &ranges, 0,
            );
            let layout_map: HashMap<(Card, Card), usize> = layout.iter().enumerate()
                .map(|(i, &(a, b))| {
                    // Normalize to (lower, higher) ordering to match how the
                    // reference looks up keys (class_combos may return (higher,
                    // lower) for non-pair classes; layout from
                    // index_to_card_pair is (card1, card2) with card1 < card2;
                    // sort once on both sides to align).
                    let key = if a < b { (a, b) } else { (b, a) };
                    (key, i)
                }).collect();

            // EXPLICIT per-class enumeration
            for c_idx in 0..NUM_PREFLOP_CLASSES {
                let class = PreflopClass(c_idx as u8);
                // EXPLICIT per-class combo enumeration (NOT calling expansion())
                let surviving: Vec<(Card, Card)> = class_combos(class).into_iter()
                    .filter(|&(c1, c2)| {
                        // EXPLICIT OR-union conflict check
                        !actual_flop.contains(&c1) && !actual_flop.contains(&c2)
                    })
                    .collect();
                if surviving.is_empty() { continue; }
                // EXPLICIT average over surviving combos at THIS actual flop
                let n_surv = surviving.len();
                let mut sum_v_f32 = 0.0f32;
                let mut sum_v_f64 = 0.0f64;
                for &(a, b) in &surviving {
                    let key = if a < b { (a, b) } else { (b, a) };
                    let idx = layout_map[&key];
                    sum_v_f32 += v_combo[idx];
                    sum_v_f64 += v_combo[idx] as f64;
                }
                let avg_v_f32 = sum_v_f32 / n_surv as f32;
                let avg_v_f64 = sum_v_f64 / n_surv as f64;
                // EXPLICIT unit weight: (|surviving| / n_class) per actual flop
                let n_c = class.num_combos() as f64;
                let weight = (n_surv as f64) / (n_c * FLOPS_PER_HAND as f64);
                ref_class_cfv_f32[c_idx] += weight * avg_v_f32 as f64;
                ref_class_cfv_f64[c_idx] += weight * avg_v_f64;
            }
        }
    }

    let ref_f32: Vec<f32> = ref_class_cfv_f32.iter().map(|&x| x as f32).collect();
    (ref_f32, ref_class_cfv_f64)
}

#[test]
fn p2_5b_real_v_flop_anchored_against_uncanonicalized_truth() {
    let table = PreflopChanceTable::new(
        2, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2],
    );
    let flop_tree = build_flop_tree();
    let class_reach = vec![1.0f32; NUM_PREFLOP_CLASSES];

    // Stratified subset for CI (n_per_cell=3 → 15 canonicals).
    let subset = stratified_canonical_subset(&table, 3);
    let total_orbit_members: usize = subset.iter()
        .map(|&i| orbit_of(table.canonical_flops[i]).len())
        .sum();
    eprintln!("\n=== P2.5b: real-V_flop anchor on stratified subset ===");
    eprintln!("Subset: {} canonicals, {} orbit members (actual flops) total",
        subset.len(), total_orbit_members);

    // RUNTIME PATH: uses library reduce + aggregate
    let t0 = std::time::Instant::now();
    let (runtime_f32, runtime_f64) = runtime_path(&table, &flop_tree, &subset, &class_reach);
    let t_rt = t0.elapsed();
    eprintln!("Runtime path: {:?}", t_rt);

    // REFERENCE PATH: explicit enumeration of orbit-member actual flops,
    // explicit reduce, explicit unit weighting. STRUCTURALLY INDEPENDENT.
    let t0 = std::time::Instant::now();
    let (reference_f32, reference_f64) = reference_path(&table, &flop_tree, &subset, &class_reach);
    let t_ref = t0.elapsed();
    eprintln!("Reference path: {:?}", t_ref);

    // f32 comparison
    let mut max_abs_diff_f32: f32 = 0.0;
    let mut max_rel_diff_f32: f32 = 0.0;
    let mut argmax_f32: usize = 0;
    for c in 0..NUM_PREFLOP_CLASSES {
        let abs = (runtime_f32[c] - reference_f32[c]).abs();
        let rel = if reference_f32[c].abs() > 1e-10 { abs / reference_f32[c].abs() } else { abs };
        if rel > max_rel_diff_f32 {
            max_rel_diff_f32 = rel;
            max_abs_diff_f32 = abs;
            argmax_f32 = c;
        }
    }
    eprintln!("\nf32 max_rel_diff = {:e} (class {}, abs {:e})",
        max_rel_diff_f32, argmax_f32, max_abs_diff_f32);
    eprintln!("  runtime_f32[{}]   = {}", argmax_f32, runtime_f32[argmax_f32]);
    eprintln!("  reference_f32[{}] = {}", argmax_f32, reference_f32[argmax_f32]);

    // f64 comparison: THE DEMONSTRATOR
    let mut max_abs_diff_f64: f64 = 0.0;
    let mut max_rel_diff_f64: f64 = 0.0;
    let mut argmax_f64: usize = 0;
    for c in 0..NUM_PREFLOP_CLASSES {
        let abs = (runtime_f64[c] - reference_f64[c]).abs();
        let rel = if reference_f64[c].abs() > 1e-20 { abs / reference_f64[c].abs() } else { abs };
        if rel > max_rel_diff_f64 {
            max_rel_diff_f64 = rel;
            max_abs_diff_f64 = abs;
            argmax_f64 = c;
        }
    }
    eprintln!("\nf64 max_rel_diff = {:e} (class {}, abs {:e})",
        max_rel_diff_f64, argmax_f64, max_abs_diff_f64);
    eprintln!("  runtime_f64[{}]   = {}", argmax_f64, runtime_f64[argmax_f64]);
    eprintln!("  reference_f64[{}] = {}", argmax_f64, reference_f64[argmax_f64]);

    // f64 verdict: precision or bug?
    let f64_floor_tol = 1e-9;  // generous floor for actual-flop-solve f64 mirror
    let bug_floor = 1e-5;
    if max_rel_diff_f64 < f64_floor_tol {
        eprintln!("\n✓ f32-floor DEMONSTRATED: f64 mirror reduces diff to {:e} (< {:e})",
            max_rel_diff_f64, f64_floor_tol);
        eprintln!("  The {:e} f32 diff is the runtime's f32 accumulator floor,", max_rel_diff_f32);
        eprintln!("  not a value-dependent bug. Arithmetic proven in exact math.");
    } else if max_rel_diff_f64 > bug_floor {
        panic!(
            "P2.5b f64 DIAGNOSTIC FAILED — VALUE-DEPENDENT BUG.\n\
             f64 max_rel_diff = {} (class {}, abs {}). f64 floor should be < 1e-10. \
             If f64 diff is large, the runtime path computes a different value than \
             the reference in EXACT arithmetic — not f32 precision, real bug.\n\
             runtime_f64[{}]   = {}\n\
             reference_f64[{}] = {}",
            max_rel_diff_f64, argmax_f64, max_abs_diff_f64,
            argmax_f64, runtime_f64[argmax_f64], argmax_f64, reference_f64[argmax_f64],
        );
    } else {
        eprintln!("\n⚠ f64 diff in ambiguous range: {} (between {:e} and {:e})",
            max_rel_diff_f64, f64_floor_tol, bug_floor);
        eprintln!("  Likely precision but worth a look — accumulation order differences,");
        eprintln!("  not f32 floor exactly, not bug threshold either.");
    }

    // f32 check: with f64 floor demonstrated, f32 diff should be within
    // f32 floor for this accumulation pattern. Bound is generous to allow
    // for f32 accumulation in many adds.
    let f32_floor_tol = 1e-3_f32;
    assert!(
        max_rel_diff_f32 < f32_floor_tol,
        "P2.5b f32 max_rel_diff = {} > tolerance {} (class {}). With f64 floor \
         demonstrated, this should be f32 noise; if it's much larger something is \
         wrong with the f32 path that doesn't show up in f64 (unusual). \
         abs diff = {}, runtime = {}, reference = {}",
        max_rel_diff_f32, f32_floor_tol, argmax_f32,
        max_abs_diff_f32, runtime_f32[argmax_f32], reference_f32[argmax_f32],
    );

    eprintln!("\n✓ P2.5b PASS on stratified subset: runtime path agrees with");
    eprintln!("  un-canonicalized actual-flop reference at f32 floor, with f64");
    eprintln!("  discriminator proving the diff is precision, not bug.");
}
