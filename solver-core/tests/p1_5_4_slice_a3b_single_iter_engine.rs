// P1.5.4 Slice A.3b: PreflopVectorCfr::run_one_iteration end-to-end.
//
// Engine that composes A.1 (strategy), A.2 (reach), A.3a (chance CFV
// composition), bottom-up walk + regret update, into one full preflop
// CFR iteration. Terminal values + per-flop CFV are passed as opaque
// callbacks (the test provides synthetic implementations; A.3c will
// provide a production terminal_value_fn from the class×class blocking
// matrix; the production per_flop_solver wraps compute_v_flop_at_root_iter0).
//
// Validation strategy:
//
//   1. Build a small HU preflop config + PreflopVectorCfr at uniform init.
//   2. Define synthetic per_flop_solver returning constant v per combo
//      (so v_at_chance simplifies to a known constant per class).
//   3. Define synthetic terminal_value_fn returning a hand-picked
//      per-class CFV per terminal (test owns these values).
//   4. Snapshot regrets pre-run, run_one_iteration, snapshot regrets post.
//   5. Independently in the test, recompute the expected post-iteration
//      regret at one specific traverser infoset using the standard
//      factored-CFR formula:
//        cfv_avg[c] = Σ_a strategy[a, c] × cfv_child[a, c]
//        inst_regret[a, c] = cfv_child[a, c] - cfv_avg[c]
//        new_regret[a, c] = coef × old_regret + inst_regret
//      where coef = alpha_t (if old_regret ≥ 0) else beta_t per DCFR.
//   6. Compare the engine's post-iteration regrets to the manual values
//      at f32 floor.
//
// The discriminating piece is step 5: it walks the same formula the
// engine uses but in the test, applied to ONE infoset with cfv_child
// values the test KNOWS (because they're either constant from the
// synthetic per_flop_solver propagated unchanged through opp's sum, or
// from the synthetic terminal_value_fn). If the engine's bottom-up
// composition is correct, the formulas agree at f32 floor.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::solver::flop_start_vector_cfr::DcfrParams;
use solver_core::solver::postflop_oracle::ClosureOracle;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
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

#[test]
fn slice_a3b_single_iteration_runs_without_panic_and_advances_iteration() {
    let tree = build_hu_preflop_tree();
    let mut solver = PreflopVectorCfr::new(&tree);
    let table = PreflopChanceTable::new(2, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2]);

    assert_eq!(solver.iteration(), 0);

    // Both callbacks return constants for this smoke test.
    let mut oracle = ClosureOracle::new(
        |_canonical: [Card; 3], combo_ranges: &[Vec<f32>], _trav: u8| -> Vec<f32> {
            vec![2.0_f32; combo_ranges[0].len()]
        }
    );
    let terminal_value_fn = |_term_idx: usize, _trav: u8, _reach: &[Vec<f32>]| -> Vec<f32> {
        vec![-0.5_f32; NUM_PREFLOP_CLASSES]
    };

    solver.run_one_iteration(&tree, &table, &mut oracle, terminal_value_fn);

    assert_eq!(solver.iteration(), 1, "iteration counter should advance");

    // After one iteration, some regrets should be non-zero (the test would
    // be uninformative if the engine didn't actually touch regrets).
    let max_abs_regret = solver.regrets.iter().cloned().map(f32::abs).fold(0.0_f32, f32::max);
    assert!(max_abs_regret > 0.0,
        "after one iteration, max|regret| = 0; engine isn't applying the update");
    eprintln!("after iter 1: max |regret| = {:.4e}", max_abs_regret);
    let max_abs_cum_strat = solver.cum_strategy.iter().cloned().map(f32::abs).fold(0.0_f32, f32::max);
    assert!(max_abs_cum_strat > 0.0,
        "after one iteration, max|cum_strategy| = 0; engine isn't accumulating strategy");
    eprintln!("after iter 1: max |cum_strategy| = {:.4e}", max_abs_cum_strat);
}

#[test]
fn slice_a3b_regret_update_at_root_matches_textbook_formula() {
    let tree = build_hu_preflop_tree();
    let mut solver = PreflopVectorCfr::new(&tree);
    let table = PreflopChanceTable::new(2, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2]);

    // Find root traverser (player at node 0; HU button-first preflop = SB = player 1).
    let root_pid = tree.nodes[0].player_id;
    let na = tree.nodes[0].num_children as usize;
    eprintln!("root node player_id={} (traverser when computing this update), na={}", root_pid, na);

    // Snapshot the initial uniform strategy (regrets all zero pre-iter).
    let root_local = solver.local_offset[0];
    let off = root_local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
    let uniform = 1.0_f32 / na as f32;
    let mut strat_before: Vec<f32> = Vec::with_capacity(na * NUM_PREFLOP_CLASSES);
    for a in 0..na {
        for c in 0..NUM_PREFLOP_CLASSES {
            strat_before.push(solver.strategy[off + a * NUM_PREFLOP_CLASSES + c]);
        }
    }
    // Verify uniform init (sanity).
    for &s in &strat_before {
        assert!((s - uniform).abs() < 1e-7);
    }

    // Synthetic per_flop_solver: returns a constant per combo so v_at_chance
    // is a (near) constant per class. We can predict v_at_chance ≈ K_chance
    // for all classes.
    let k_chance = 1.25_f32;
    let mut oracle = ClosureOracle::new(
        move |_canonical: [Card; 3], combo_ranges: &[Vec<f32>], _trav: u8| -> Vec<f32> {
            vec![k_chance; combo_ranges[0].len()]
        }
    );
    // Synthetic terminal_value_fn: returns a constant per class per terminal.
    // Make different terminals have different constants to exercise the
    // bottom-up summation.
    let terminal_value_fn = |term_idx: usize, _trav: u8, _reach: &[Vec<f32>]| -> Vec<f32> {
        let v = -0.3_f32 + 0.001_f32 * (term_idx % 17) as f32;
        vec![v; NUM_PREFLOP_CLASSES]
    };

    // Run iteration.
    solver.run_one_iteration(&tree, &table, &mut oracle, terminal_value_fn);

    // Engine's post-iteration regrets at root.
    let regrets_after: Vec<f32> = (0..na * NUM_PREFLOP_CLASSES)
        .map(|i| solver.regrets[off + i])
        .collect();

    // ── Independent computation: replicate the engine's bottom-up walk
    // for traverser = root_pid, recover cfv at root via the same logic, and
    // apply the same regret update formula. The test trusts:
    //   - compute_chance_node_cfv_with_expansion (A.3a validated)
    //   - the textbook regret update formula
    // The engine's bottom-up walk + regret update at the root is what's
    // being checked.
    let chance_nodes = solver.preflop_chance_node_indices(&tree);
    let reach = solver.compute_preflop_reach(&tree, None);

    // Reconstruct cfv at every preflop chance node (same as engine did).
    let mut cfv_ref: Vec<Vec<f32>> = vec![vec![0.0_f32; NUM_PREFLOP_CLASSES]; tree.num_nodes()];
    let mut ref_oracle = ClosureOracle::new(
        move |_canonical: [Card; 3], combo_ranges: &[Vec<f32>], _trav: u8| -> Vec<f32> {
            vec![k_chance; combo_ranges[0].len()]
        }
    );
    for &c_idx in &chance_nodes {
        cfv_ref[c_idx] = solver.compute_chance_node_cfv_with_expansion(
            c_idx, root_pid, &reach, &table, &mut ref_oracle,
        );
    }

    // Recursive bottom-up reference (test-side): mirror the engine's logic
    // but without touching solver.regrets / solver.cum_strategy.
    fn walk(
        tree: &FlatTree,
        node_idx: usize,
        traverser: u8,
        strategy: &[f32],
        local_offset: &[usize],
        chance_nodes: &[usize],
        cfv: &mut [Vec<f32>],
        term_val: &impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
    ) {
        if chance_nodes.binary_search(&node_idx).is_ok() { return; }
        let node = &tree.nodes[node_idx];
        if node.is_terminal() {
            // The engine passes per-terminal per-player reach; in this
            // synthetic-callback test, the callback ignores it.
            let dummy_reach: Vec<Vec<f32>> = vec![vec![0.0; NUM_PREFLOP_CLASSES]; 2];
            cfv[node_idx] = term_val(node_idx, traverser, &dummy_reach);
            return;
        }
        let children: Vec<u32> = tree.node_children(node_idx).to_vec();
        for &ch in &children {
            walk(tree, ch as usize, traverser, strategy, local_offset, chance_nodes, cfv, term_val);
        }
        let local = local_offset[node_idx];
        let na = node.num_children as usize;
        let off = local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
        let mut cfv_avg = vec![0.0_f32; NUM_PREFLOP_CLASSES];
        if node.player_id == traverser {
            for (a, &ch) in children.iter().enumerate() {
                let child = ch as usize;
                for c in 0..NUM_PREFLOP_CLASSES {
                    cfv_avg[c] += strategy[off + a * NUM_PREFLOP_CLASSES + c] * cfv[child][c];
                }
            }
        } else {
            for &ch in &children {
                let child = ch as usize;
                for c in 0..NUM_PREFLOP_CLASSES {
                    cfv_avg[c] += cfv[child][c];
                }
            }
        }
        cfv[node_idx] = cfv_avg;
    }
    // Snapshot strategy BEFORE the iteration (engine has since written
    // the same uniform values back; we keep our snapshot for clarity).
    // Note: solver.strategy was overwritten by compute_preflop_strategy
    // inside run_one_iteration, but since regrets were all zero pre-iter,
    // it stayed uniform. Confirm:
    for (a, c) in (0..na).flat_map(|a| (0..NUM_PREFLOP_CLASSES).map(move |c| (a, c))) {
        let s = solver.strategy[off + a * NUM_PREFLOP_CLASSES + c];
        assert!((s - uniform).abs() < 1e-7,
            "strategy after iter at root [a={}, c={}] = {} != uniform {} (regrets were zero pre-iter, expect unchanged)",
            a, c, s, uniform);
    }
    let snapshot_strategy: Vec<f32> = (0..solver.strategy.len()).map(|i| solver.strategy[i]).collect();
    walk(&tree, 0, root_pid, &snapshot_strategy, &solver.local_offset, &chance_nodes, &mut cfv_ref, &terminal_value_fn);

    // cfv_ref[children of root, c] is now the per-action child CFV for the root infoset.
    // Compute expected regret update at root via the textbook formula.
    let params = DcfrParams::new(0); // pre-iter iteration was 0
    let root_children = tree.node_children(0);
    let mut cfv_avg_root = vec![0.0_f32; NUM_PREFLOP_CLASSES];
    for (a, &ch) in root_children.iter().enumerate() {
        let child = ch as usize;
        for c in 0..NUM_PREFLOP_CLASSES {
            cfv_avg_root[c] += snapshot_strategy[off + a * NUM_PREFLOP_CLASSES + c] * cfv_ref[child][c];
        }
    }
    let mut expected_regrets = vec![0.0_f32; na * NUM_PREFLOP_CLASSES];
    for (a, &ch) in root_children.iter().enumerate() {
        let child = ch as usize;
        for c in 0..NUM_PREFLOP_CLASSES {
            let inst_regret = cfv_ref[child][c] - cfv_avg_root[c];
            // old_regret = 0 → coef = alpha_t
            let coef = params.alpha_t();  // 0 ≥ 0 so this branch
            expected_regrets[a * NUM_PREFLOP_CLASSES + c] = coef * 0.0 + inst_regret;
        }
    }

    // Compare.
    let mut max_diff = 0.0_f32;
    let mut max_loc = (0_usize, 0_usize);
    for a in 0..na {
        for c in 0..NUM_PREFLOP_CLASSES {
            let i = a * NUM_PREFLOP_CLASSES + c;
            let d = (regrets_after[i] - expected_regrets[i]).abs();
            if d > max_diff { max_diff = d; max_loc = (a, c); }
        }
    }
    eprintln!("root regret update: max_diff = {:.4e} at action {}, class {} \
              (engine={:.6}, expected={:.6})",
        max_diff, max_loc.0, max_loc.1,
        regrets_after[max_loc.0 * NUM_PREFLOP_CLASSES + max_loc.1],
        expected_regrets[max_loc.0 * NUM_PREFLOP_CLASSES + max_loc.1]);
    assert!(max_diff < 1e-5,
        "root regret update diverges from textbook formula (max_diff = {:.4e}); \
         engine's bottom-up walk or regret update is wrong",
        max_diff);

    // Sanity: instantaneous regret is nontrivial (otherwise the test is uninformative).
    let max_abs_inst = expected_regrets.iter().cloned().map(f32::abs).fold(0.0_f32, f32::max);
    assert!(max_abs_inst > 1e-4,
        "expected regrets are near-zero ({:.4e}); synthetic v values too symmetric to discriminate",
        max_abs_inst);

    eprintln!("Slice A.3b PASS: single-iteration engine matches textbook factored-CFR \
              regret update at root infoset across {} actions × {} classes; \
              max_diff {:.4e} ≤ 1e-5.", na, NUM_PREFLOP_CLASSES, max_diff);
}
