// P1.5.4 Slice A.3a: PreflopVectorCfr::compute_chance_node_cfv_with_expansion.
//
// The integration wiring at the preflop→flop boundary: extract reach at
// a preflop chance node, expand class-reach to combo-reach via P5b for
// each canonical, hand off to a per_flop_solver closure, reduce per-class
// via P5b, aggregate per-canonical via P5a (orbit-weighted).
//
// What's NEW in A.3a:
//   - chance-node identification (preflop_chance_node_indices)
//   - per-chance-node reach extraction from the full reach array
//   - per-canonical expansion + per-flop solver call wiring
//
// What's anchored already:
//   - expand_reach_class_to_combo / reduce_cfv_combo_to_class (P5b)
//   - aggregate_preflop_chance (P5a)
//   - compute_preflop_cfv_per_canonical_pass (P5c composition)
//
// Validation: two tests.
//
//   1. CONSTANT v_flop: per_flop_solver returns a class-uniform constant
//      K per combo. By the reduce identity (P5b), reduce_cfv_combo_to_class
//      gives K per class; aggregate over canonicals weighted by chance
//      probabilities sums to K × Σ P(F|class) = K × 1.0 per class. Catches
//      orbit-weight-doubling or normalization-missed bugs without
//      requiring the test to redo any non-trivial math.
//
//   2. REACH-AWARE v_flop: per_flop_solver returns the opponent's
//      expanded combo reach as v[combo]. The production
//      `compute_chance_node_cfv_with_expansion` must produce the same
//      output as an independent path through `compute_preflop_cfv_per_
//      canonical_pass` (P5c-anchored) when its v_flop_fn captures the
//      equivalent expansion. Catches wrong-canonical, wrong-layout,
//      double-expansion bugs.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::solver::postflop_oracle::ClosureOracle;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::solver::preflop_start_game::{
    compute_preflop_cfv_per_canonical_pass, expand_reach_class_to_combo,
    flop_combo_layout, PreflopChanceTable,
};
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn slice_a3a_preflop_chance_nodes_exist_in_hu_config() {
    let tree = build_hu_preflop_tree();
    let solver = PreflopVectorCfr::new(&tree);
    let chance_nodes = solver.preflop_chance_node_indices(&tree);
    eprintln!("HU preflop config has {} preflop chance nodes", chance_nodes.len());
    assert!(!chance_nodes.is_empty(),
        "any preflop-rooted config that reaches the flop must have at least one preflop chance node");

    // Each preflop→flop chance node has board_state == Flop (the
    // tree-builder's destination convention: the chance node carries the
    // board_state of the zone it transitions TO). Its PARENT must be in
    // the preflop zone — that's the discriminator.
    let mut parents_of = vec![None::<u32>; tree.num_nodes()];
    for parent_idx in 0..tree.num_nodes() {
        for &child in tree.node_children(parent_idx) {
            parents_of[child as usize] = Some(parent_idx as u32);
        }
    }
    for &idx in &chance_nodes {
        let n = &tree.nodes[idx];
        assert!(n.is_chance(),
            "node {} reported as preflop→flop chance but is_chance()=false", idx);
        assert_eq!(n.board_state, BoardState::Flop as u8,
            "node {}: preflop→flop chance should carry destination state Flop (0), got {}",
            idx, n.board_state);
        let pp = parents_of[idx];
        let parent_state = pp.map(|p| tree.nodes[p as usize].board_state).unwrap_or(255);
        assert_eq!(parent_state, BoardState::Preflop as u8,
            "node {}: preflop→flop chance must have a preflop parent; got parent={:?} (state {})",
            idx, pp, parent_state);
    }
}

#[test]
fn slice_a3a_constant_v_flop_aggregates_to_constant_per_class() {
    let tree = build_hu_preflop_tree();
    let solver = PreflopVectorCfr::new(&tree);
    let table = PreflopChanceTable::new(2, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2]);

    let chance_nodes = solver.preflop_chance_node_indices(&tree);
    let chance_idx = chance_nodes[0];

    let reach = solver.compute_preflop_reach(&tree, None);

    // CONSTANT v_flop: return K per combo for every (canonical, combo).
    let k = 1.75_f32;
    let mut oracle = ClosureOracle::new(
        |_canonical: [Card; 3], combo_ranges: &[Vec<f32>], _traverser: u8| -> Vec<f32> {
            vec![k; combo_ranges[0].len()]
        }
    );
    let v_at_chance = solver.compute_chance_node_cfv_with_expansion(
        chance_idx, 0, &reach, &table, &mut oracle,
    );

    // Expected: K per class (the reduce identity: class-uniform combo
    // values reduce to that constant per class; aggregate with
    // chance_probability_flop summing to 1 per class gives K).
    //
    // Tolerance: f32 accumulation over 1755 canonicals × 169 classes,
    // weighted by orbit sizes (×4 / ×12 / ×24). The P5a anchor measured
    // ~4.66e-7 PER CLASS at the linear-N×ULP regime with input scale ~1
    // and small sums; with K=1.75 and orbit-weighted summation over 1755
    // canonicals, the per-class drift scales linearly with K and with N
    // — so 1.75 × ~1755 × eps ≈ 1755 × 1e-7 ≈ 2e-4 worst-case bound, and
    // observed drift around 2e-5 falls comfortably inside that. We use
    // 1e-4 as the test tolerance — discriminating enough to catch a
    // sign-flip or factor-of-2 bug, loose enough not to flag the
    // legitimate f32 floor.
    let mut max_diff = 0.0_f32;
    for c in 0..NUM_PREFLOP_CLASSES {
        let got = v_at_chance[c];
        let diff = (got - k).abs();
        if diff > max_diff { max_diff = diff; }
    }
    eprintln!("constant v_flop K={}: max |v[c] - K| across 169 classes = {:.4e}", k, max_diff);
    assert!(max_diff < 1e-4,
        "constant K composition: max_diff {:.4e} exceeds f32 floor tolerance 1e-4. \
         A diff at orders of K (e.g., 0.5K) would indicate orbit-weight or normalization bug.",
        max_diff);
}

#[test]
fn slice_a3a_reach_aware_v_flop_matches_p5c_anchored_path() {
    let tree = build_hu_preflop_tree();
    let mut solver = PreflopVectorCfr::new(&tree);

    // Perturb strategy at root to make the reach non-uniform across classes,
    // so the test exercises the expansion code on varied inputs.
    let root_local = solver.local_offset[0];
    let na = tree.nodes[0].num_children as usize;
    let off = root_local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
    for c in 0..NUM_PREFLOP_CLASSES {
        for a in 0..na {
            solver.regrets[off + a * NUM_PREFLOP_CLASSES + c] = (c % 5 + a + 1) as f32;
        }
    }
    solver.compute_preflop_strategy(&tree);

    let table = PreflopChanceTable::new(2, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 2]);
    let chance_nodes = solver.preflop_chance_node_indices(&tree);
    let chance_idx = chance_nodes[0];
    let reach = solver.compute_preflop_reach(&tree, None);

    let traverser: u8 = 0;
    let opp: u8 = 1;
    let chance_base = chance_idx * NUM_PREFLOP_CLASSES;

    // Production path: PreflopVectorCfr's wired composition through
    // the oracle interface. Synthetic oracle returns combo_ranges[opp]
    // as v_combo — a deterministic function of the EXPANDED reach (not
    // the class reach).
    let mut prod_oracle = ClosureOracle::new(
        |_canonical: [Card; 3], combo_ranges: &[Vec<f32>], _trav: u8| -> Vec<f32> {
            combo_ranges[opp as usize].clone()
        }
    );
    let production = solver.compute_chance_node_cfv_with_expansion(
        chance_idx, traverser, &reach, &table, &mut prod_oracle,
    );

    // Reference path: compute_preflop_cfv_per_canonical_pass (P5c-anchored)
    // with a v_flop_fn that performs the EQUIVALENT expansion lookup itself.
    // This independently exercises the expand+reduce+aggregate chain via the
    // anchored primitive.
    let reference = compute_preflop_cfv_per_canonical_pass(&table,
        |canonical: [Card; 3], combo: (Card, Card)| -> f32 {
            let layout = flop_combo_layout(canonical);
            let class_reach_opp = &reach[opp as usize][chance_base..chance_base + NUM_PREFLOP_CLASSES];
            let combo_reach_opp = expand_reach_class_to_combo(canonical, class_reach_opp, &layout);
            // Locate combo's index in layout (linear scan; layout is at most ~1176 entries).
            for (i, &c) in layout.iter().enumerate() {
                if c == combo {
                    return combo_reach_opp[i];
                }
            }
            panic!("combo {:?} not in layout of canonical {:?}", combo, canonical);
        }
    );

    let d = max_abs_diff(&production, &reference);
    eprintln!("reach-aware v_flop: max_abs_diff (production vs P5c-anchored) = {:.4e}", d);
    assert!(d < 1e-5,
        "production composition diverges from P5c-anchored compute_preflop_cfv_per_canonical_pass: \
         max_abs_diff = {:.4e}", d);

    // Sanity: result is non-trivial (not all-zero, not all-equal).
    let max_v = production.iter().cloned().fold(0.0_f32, f32::max);
    let min_v = production.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!("v_at_chance value range: [{:.4e}, {:.4e}] across 169 classes", min_v, max_v);
    assert!(max_v > 1e-6, "v_at_chance is all near-zero; reach is suspect");
    assert!((max_v - min_v).abs() > 1e-6,
        "v_at_chance is constant across classes; reach-dependence not actually exercised");

    eprintln!("Slice A.3a PASS: chance-node CFV composition matches P5c-anchored direct path at \
              f32 floor; reach extraction at chance node, per-canonical expansion, reduce, \
              and orbit-weighted aggregation are wired correctly.");
}
