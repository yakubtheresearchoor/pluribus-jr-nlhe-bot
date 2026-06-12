// P1.5.4 Slice A.2: PreflopVectorCfr::compute_preflop_reach.
//
// Top-down reach propagation through the preflop zone. The composition
// risks here are: acting-vs-non-acting player distinction at each edge
// (acting player's reach scales by strategy, others pass through), chance
// nodes pass reach through unchanged (chance probability applied
// externally at aggregate_preflop_chance), and the zone boundary
// (preflop→flop chance) stops propagation so the per-canonical solver
// receives the chance-node reach as its input.
//
// Reference: an independent recursive top-down walk implemented in this
// test file (no shared code with the production path). For each node,
// the reference visits each path from root and accumulates the product
// of strategy weights at the acting player's decisions. Per-class
// because the strategy is per-class.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

fn build_hu_preflop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![99, 98],
        initial_contributions: vec![1, 2],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
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

/// Independent recursive top-down reference reach computation.
///
/// For each node `n`, computes `reach[p][n * NUM_PREFLOP_CLASSES + c]`
/// as the product over edges from root to n of `strategy[edge, c]` if
/// edge's parent is acting player `p`, else 1.0.
///
/// Implementation: depth-first walk, tracking the current per-player
/// per-class reach in a mutable accumulator. Strategy lookup goes
/// through the production storage convention (mirrors the production
/// layout so the test catches storage-convention bugs).
fn reference_preflop_reach(
    tree: &FlatTree,
    solver: &PreflopVectorCfr,
    initial_reach: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let nn = tree.num_nodes();
    let np = solver.num_players as usize;
    let n_classes = NUM_PREFLOP_CLASSES;
    let mut reach: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![0.0_f32; nn * n_classes])
        .collect();
    for p in 0..np {
        reach[p][0..n_classes].copy_from_slice(&initial_reach[p]);
    }
    walk(tree, solver, 0, &mut reach, np, n_classes);
    reach
}

fn walk(
    tree: &FlatTree,
    solver: &PreflopVectorCfr,
    node_idx: usize,
    reach: &mut [Vec<f32>],
    np: usize,
    n_classes: usize,
) {
    let node = &tree.nodes[node_idx];
    if node.board_state != BoardState::Preflop as u8 { return; }
    let children: Vec<u32> = tree.node_children(node_idx).to_vec();
    if children.is_empty() { return; }

    let parent_base = node_idx * n_classes;

    if node.is_player() {
        let pid = node.player_id as usize;
        let local = solver.local_offset[node_idx];
        let na = node.num_children as usize;
        let off = local * MAX_NA_PREFLOP * n_classes;
        for (a, &child_u32) in children.iter().enumerate() {
            let child = child_u32 as usize;
            let child_base = child * n_classes;
            for p in 0..np {
                if p == pid {
                    for c in 0..n_classes {
                        reach[p][child_base + c] =
                            reach[p][parent_base + c]
                            * solver.strategy[off + a * n_classes + c];
                    }
                } else {
                    for c in 0..n_classes {
                        reach[p][child_base + c] = reach[p][parent_base + c];
                    }
                }
            }
            walk(tree, solver, child, reach, np, n_classes);
        }
    } else {
        // Chance: passes reach through unchanged.
        for &child_u32 in &children {
            let child = child_u32 as usize;
            let child_base = child * n_classes;
            for p in 0..np {
                for c in 0..n_classes {
                    reach[p][child_base + c] = reach[p][parent_base + c];
                }
            }
            walk(tree, solver, child, reach, np, n_classes);
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn slice_a2_uniform_strategy_reach_at_root() {
    let tree = build_hu_preflop_tree();
    let solver = PreflopVectorCfr::new(&tree);
    let reach = solver.compute_preflop_reach(&tree, None);
    // At root: reach should be 1.0 for all (player, class).
    for p in 0..solver.num_players as usize {
        for c in 0..NUM_PREFLOP_CLASSES {
            assert_eq!(reach[p][c], 1.0,
                "root reach[{}][{}] = {} != 1.0", p, c, reach[p][c]);
        }
    }
}

#[test]
fn slice_a2_matches_independent_reference_uniform_strategy() {
    let tree = build_hu_preflop_tree();
    let solver = PreflopVectorCfr::new(&tree);

    let np = solver.num_players as usize;
    let initial_reach: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32; NUM_PREFLOP_CLASSES])
        .collect();

    let production = solver.compute_preflop_reach(&tree, Some(&initial_reach));
    let reference = reference_preflop_reach(&tree, &solver, &initial_reach);

    for p in 0..np {
        let d = max_abs_diff(&production[p], &reference[p]);
        eprintln!("uniform strategy: player {} max_abs_diff = {:.4e}", p, d);
        assert_eq!(d, 0.0,
            "player {} reach disagrees with reference at uniform strategy (max_abs_diff {})",
            p, d);
    }
}

#[test]
fn slice_a2_matches_independent_reference_perturbed_strategy() {
    let tree = build_hu_preflop_tree();
    let mut solver = PreflopVectorCfr::new(&tree);

    // Set non-trivial regrets at the root (which is a player node), so
    // compute_preflop_strategy produces a non-uniform per-class strategy.
    let root_local = solver.local_offset[0];
    assert!(root_local != usize::MAX, "expected root to be a preflop player node");
    let na = tree.nodes[0].num_children as usize;
    let off = root_local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
    // Per-class varying regrets: class c gets regret a → (c % 7 + a + 1) for action a.
    for c in 0..NUM_PREFLOP_CLASSES {
        for a in 0..na {
            solver.regrets[off + a * NUM_PREFLOP_CLASSES + c] = (c % 7 + a + 1) as f32;
        }
    }
    solver.compute_preflop_strategy(&tree);

    let np = solver.num_players as usize;
    let initial_reach: Vec<Vec<f32>> = (0..np)
        .map(|p| (0..NUM_PREFLOP_CLASSES).map(|c| 0.5 + 0.01 * ((p + c) % 5) as f32).collect())
        .collect();

    let production = solver.compute_preflop_reach(&tree, Some(&initial_reach));
    let reference = reference_preflop_reach(&tree, &solver, &initial_reach);

    let mut total_max_diff = 0.0_f32;
    for p in 0..np {
        let d = max_abs_diff(&production[p], &reference[p]);
        eprintln!("perturbed strategy: player {} max_abs_diff = {:.4e}", p, d);
        if d > total_max_diff { total_max_diff = d; }
    }
    assert_eq!(total_max_diff, 0.0,
        "perturbed strategy: reach disagrees with reference (max_abs_diff {})", total_max_diff);
}

#[test]
fn slice_a2_acting_vs_non_acting_distinction_holds() {
    // Discriminating check: when player p acts at node n with non-uniform
    // strategy, reach[p] must scale at children but reach[other] must NOT.
    // A bug that scales both players' reach would still pass the uniform-
    // strategy test (everything is 0.5 either way for 2-action root) but
    // fail this perturbed strategy check.
    let tree = build_hu_preflop_tree();
    let mut solver = PreflopVectorCfr::new(&tree);
    let na = tree.nodes[0].num_children as usize;
    assert_eq!(na, 2, "test assumes 2-action root for the symmetry-breaking signal");

    // Set strategy at root such that for class 0, action 0 is preferred (0.9 vs 0.1).
    let root_local = solver.local_offset[0];
    let off = root_local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
    // Equivalent to setting regrets [9, 1] which normalizes to [0.9, 0.1].
    solver.regrets[off + 0 * NUM_PREFLOP_CLASSES + 0] = 9.0;
    solver.regrets[off + 1 * NUM_PREFLOP_CLASSES + 0] = 1.0;
    solver.compute_preflop_strategy(&tree);
    // Sanity:
    let s_a0 = solver.strategy[off + 0 * NUM_PREFLOP_CLASSES + 0];
    let s_a1 = solver.strategy[off + 1 * NUM_PREFLOP_CLASSES + 0];
    assert!((s_a0 - 0.9).abs() < 1e-6 && (s_a1 - 0.1).abs() < 1e-6,
        "expected [0.9, 0.1] at root class 0; got [{}, {}]", s_a0, s_a1);

    // Initial reach: all-ones.
    let reach = solver.compute_preflop_reach(&tree, None);

    // First child of root is action 0. Find its node id.
    let children = tree.node_children(0);
    let child_a0 = children[0] as usize;
    let acting_p = tree.nodes[0].player_id as usize;
    let other_p = 1 - acting_p;

    // For acting player at child of action 0: reach should be 0.9 (= 1.0 * 0.9).
    let r_acting_class0 = reach[acting_p][child_a0 * NUM_PREFLOP_CLASSES + 0];
    let r_other_class0 = reach[other_p][child_a0 * NUM_PREFLOP_CLASSES + 0];
    assert!((r_acting_class0 - 0.9).abs() < 1e-6,
        "after acting player chose action 0 (strategy=0.9): acting reach = {} != 0.9", r_acting_class0);
    assert!((r_other_class0 - 1.0).abs() < 1e-6,
        "non-acting player reach should pass through unchanged at 1.0; got {}", r_other_class0);

    // For acting player at child of action 1: reach should be 0.1 (= 1.0 * 0.1).
    let child_a1 = children[1] as usize;
    let r_acting_a1 = reach[acting_p][child_a1 * NUM_PREFLOP_CLASSES + 0];
    let r_other_a1 = reach[other_p][child_a1 * NUM_PREFLOP_CLASSES + 0];
    assert!((r_acting_a1 - 0.1).abs() < 1e-6,
        "after acting player chose action 1 (strategy=0.1): acting reach = {} != 0.1", r_acting_a1);
    assert!((r_other_a1 - 1.0).abs() < 1e-6,
        "non-acting player reach should pass through unchanged at 1.0; got {}", r_other_a1);

    // Sanity: across all 169 classes at the (acting-player) action-0 child,
    // reach values for other_p stay at 1.0 (unchanged).
    for c in 0..NUM_PREFLOP_CLASSES {
        let r_other = reach[other_p][child_a0 * NUM_PREFLOP_CLASSES + c];
        assert!((r_other - 1.0).abs() < 1e-6,
            "class {}: non-acting reach drifted from 1.0 to {} (cross-player contamination)",
            c, r_other);
    }
    eprintln!("Slice A.2 PASS: acting player reach scales by strategy, non-acting stays unchanged, \
              per-class independence holds, perturbed-strategy reference match at exact 0 diff.");
}
