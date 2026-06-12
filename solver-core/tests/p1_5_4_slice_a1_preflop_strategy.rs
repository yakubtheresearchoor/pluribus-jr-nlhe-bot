// P1.5.4 Slice A.1: PreflopVectorCfr initialization + compute_preflop_strategy.
//
// The narrowest sub-slice of the preflop CFR loop: per-infoset regret-
// matching. The composition risks here are tiny but real: stride must
// match the production storage convention (`MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES`,
// indexed as `[a * NUM_PREFLOP_CLASSES + c]`), per-class independence
// must hold (changing class 0's regret must not affect class 5's
// strategy), the REGRET_MATCH_EPS threshold must drop near-zero noise to
// the uniform fallback, and the action range must respect each infoset's
// actual na (not MAX_NA_PREFLOP).
//
// Reference: textbook regret-matching at f32 floor.
//
// Once A.1 is anchored, A.2 (compute_preflop_reach) can compose against
// this without re-validating strategy correctness.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

/// Build a minimal preflop-rooted HU config. Simple action set so the
/// preflop tree is small and inspectable.
fn build_hu_preflop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        // HU blinds: SB=1, BB=2, pot=3 chips after antes.
        starting_pot: 3,
        starting_stacks: vec![99, 98], // post-blind stacks
        initial_contributions: vec![1, 2],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            // Preflop open at 2.5x BB (i.e., bet to 5 total from BB=2).
            bet: vec![BetSize::PotRelative(0.5)], // matches a moderate open
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    build_tree(&cfg).expect("preflop tree builds")
}

/// Recompute the expected strategy for one infoset, one class, using the
/// textbook regret-matching formula directly. Used as the validation
/// reference.
fn textbook_regret_matching(regrets: &[f32]) -> Vec<f32> {
    let pos_sum: f32 = regrets.iter()
        .map(|&r| if r > 1e-5 { r } else { 0.0 })
        .sum();
    if pos_sum > 0.0 {
        regrets.iter().map(|&r| if r > 1e-5 { r / pos_sum } else { 0.0 }).collect()
    } else {
        let n = regrets.len();
        vec![1.0_f32 / n as f32; n]
    }
}

#[test]
fn slice_a1_initial_strategy_is_uniform_per_infoset() {
    let tree = build_hu_preflop_tree();
    let solver = PreflopVectorCfr::new(&tree);

    eprintln!("preflop infoset_count: {}", solver.infoset_count());
    assert!(solver.infoset_count() > 0,
        "preflop-rooted tree should have at least one preflop player infoset");

    // For every preflop player node, the initial strategy across all
    // classes is uniform 1/na.
    let mut checked = 0usize;
    for idx in 0..tree.num_nodes() {
        let local = solver.local_offset[idx];
        if local == usize::MAX { continue; }  // not a preflop player node
        let na = tree.nodes[idx].num_children as usize;
        assert!(na >= 2, "preflop player node {} has na={} < 2", idx, na);
        let uniform = 1.0_f32 / na as f32;
        let off = local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
        for c in 0..NUM_PREFLOP_CLASSES {
            for a in 0..na {
                let s = solver.strategy[off + a * NUM_PREFLOP_CLASSES + c];
                assert!((s - uniform).abs() < 1e-7,
                    "node {} class {} action {}: strategy {} != uniform {}",
                    idx, c, a, s, uniform);
            }
            // Actions beyond na are zeroed (padded slots in the MAX_NA_PREFLOP stride).
            for a in na..MAX_NA_PREFLOP {
                let s = solver.strategy[off + a * NUM_PREFLOP_CLASSES + c];
                assert_eq!(s, 0.0,
                    "node {} class {} padded action {}: strategy {} != 0 (uniform_init left it 0)",
                    idx, c, a, s);
            }
        }
        checked += 1;
    }
    eprintln!("checked uniform strategy on {} preflop infosets across {} classes",
        checked, NUM_PREFLOP_CLASSES);
}

#[test]
fn slice_a1_compute_preflop_strategy_matches_textbook_regret_matching() {
    let tree = build_hu_preflop_tree();
    let mut solver = PreflopVectorCfr::new(&tree);

    // Find an infoset to perturb.
    let target_idx = (0..tree.num_nodes())
        .find(|&idx| solver.local_offset[idx] != usize::MAX)
        .expect("at least one preflop player infoset");
    let target_local = solver.local_offset[target_idx];
    let na = tree.nodes[target_idx].num_children as usize;
    let off = target_local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;

    eprintln!("perturbing infoset at node {} (local {}), na={}", target_idx, target_local, na);

    // Test pattern (per-class regret vectors of length na):
    //   class 0: positive-dominant, all > eps  → regret-matching to proportions
    //   class 1: mixed sign                    → positive-only normalization
    //   class 2: all near zero (below eps)     → uniform fallback
    //   class 3: all exact zero                → uniform fallback
    //   class 4+: leave at zero (uniform expected)
    let class0_regrets: Vec<f32> = (0..na).map(|a| 1.0_f32 + a as f32).collect();
    let mut class1_regrets: Vec<f32> = (0..na).map(|a| if a % 2 == 0 { 2.0_f32 } else { -1.0_f32 }).collect();
    if na >= 1 { class1_regrets[0] = 3.0; }  // ensure non-trivial
    let class2_regrets: Vec<f32> = vec![1e-7_f32; na];  // all below REGRET_MATCH_EPS = 1e-5
    let class3_regrets: Vec<f32> = vec![0.0_f32; na];

    for a in 0..na {
        solver.regrets[off + a * NUM_PREFLOP_CLASSES + 0] = class0_regrets[a];
        solver.regrets[off + a * NUM_PREFLOP_CLASSES + 1] = class1_regrets[a];
        solver.regrets[off + a * NUM_PREFLOP_CLASSES + 2] = class2_regrets[a];
        solver.regrets[off + a * NUM_PREFLOP_CLASSES + 3] = class3_regrets[a];
    }

    solver.compute_preflop_strategy(&tree);

    // Validate against textbook regret-matching, per class.
    for (class_idx, regrets) in [
        (0usize, class0_regrets),
        (1, class1_regrets),
        (2, class2_regrets),
        (3, class3_regrets),
    ] {
        let expected = textbook_regret_matching(&regrets);
        for a in 0..na {
            let got = solver.strategy[off + a * NUM_PREFLOP_CLASSES + class_idx];
            let exp = expected[a];
            assert!((got - exp).abs() < 1e-6,
                "class {} action {}: got {:.6} expected {:.6} (regrets={:?})",
                class_idx, a, got, exp, regrets);
        }
    }

    // Classes 4..NUM_PREFLOP_CLASSES: regrets still all-zero, strategy
    // should be uniform 1/na (uniform fallback).
    let uniform = 1.0_f32 / na as f32;
    for c in 4..NUM_PREFLOP_CLASSES {
        for a in 0..na {
            let got = solver.strategy[off + a * NUM_PREFLOP_CLASSES + c];
            assert!((got - uniform).abs() < 1e-7,
                "class {} action {}: got {:.6} expected uniform {:.6} (regrets zero, expect uniform)",
                c, a, got, uniform);
        }
    }

    // Other infosets: unchanged from uniform-init.
    for idx in 0..tree.num_nodes() {
        let local = solver.local_offset[idx];
        if local == usize::MAX { continue; }
        if idx == target_idx { continue; }
        let other_na = tree.nodes[idx].num_children as usize;
        let other_uniform = 1.0_f32 / other_na as f32;
        let other_off = local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
        for c in 0..NUM_PREFLOP_CLASSES {
            for a in 0..other_na {
                let got = solver.strategy[other_off + a * NUM_PREFLOP_CLASSES + c];
                assert!((got - other_uniform).abs() < 1e-7,
                    "untouched infoset (node {}) class {} action {}: got {:.6} expected uniform {:.6} \
                     (cross-infoset contamination)",
                    idx, c, a, got, other_uniform);
            }
        }
    }

    eprintln!("Slice A.1 PASS: regret-matching matches textbook formula at f32 floor; \
              per-class independence holds; cross-infoset isolation holds; uniform \
              fallback triggers below REGRET_MATCH_EPS and at exact zero.");
}
