//! V1 PREFLOP RUNTIME GATE (2026-06-12): the four-zone preflop layer
//! with a SEAM-CELL/BUCKET-KEYED oracle — the v1 wiring (the validated
//! SPR cell-set policy supplies the postflop oracle per bucket; the
//! shared-chance collapse keys on `seam_bucket_chance_key`).
//!
//! Gates:
//!   1. ROUTING: the engine must call the cell-aware oracle method at
//!      every chance CFV (the stub's cell-blind method PANICS).
//!   2. COLLAPSE EXACTNESS at bucket keys: shared-chance(seam key) ≡
//!      uncollapsed, BIT-exact across regrets/cum/strategy, for an
//!      oracle whose answer depends on (flop, traverser, BUCKET) —
//!      the v1 oracle contract. (The bootstrap gate proved this for a
//!      single-key oracle; v1 keys ~125 buckets.)
//!   3. KEY SANITY: distinct buckets at the flop boundary get distinct
//!      oracle answers (the stub varies by bucket), and the number of
//!      unique keys equals the number of unique (live, SPR-bin) cells.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::solver::postflop_oracle::{PostflopValueOracle, SeamCell};
use solver_core::solver::preflop_cfr::{
    make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr,
};
use solver_core::solver::preflop_start_game::{flop_combo_layout, PreflopChanceTable};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree_preflop_only;
use std::collections::HashSet;

const STACK: i32 = 200;

/// Bucket-keyed stub oracle: deterministic synthetic values varying by
/// (flop card 0, traverser, bucket). Cell-blind entry PANICS — gate 1.
struct BucketStubOracle {
    keys_seen: HashSet<(u8, i64)>,
}

impl PostflopValueOracle for BucketStubOracle {
    fn flop_root_cfv(
        &mut self,
        _canonical_flop: [Card; 3],
        _combo_ranges: &[Vec<f32>],
        _traverser: u8,
    ) -> Vec<f32> {
        panic!("cell-blind oracle path reached — the v1 engine must route through flop_root_cfv_for_cell");
    }

    fn flop_root_cfv_for_cell(
        &mut self,
        canonical_flop: [Card; 3],
        _combo_ranges: &[Vec<f32>],
        traverser: u8,
        cell: SeamCell,
        _folded_mask: u16,
    ) -> Vec<f32> {
        let (live, bin) = cell.bucket_key(STACK);
        self.keys_seen.insert((live, bin));
        let n = flop_combo_layout(canonical_flop).len();
        let seed = (canonical_flop[0] as i64) * 7919
            + (traverser as i64) * 104729
            + (live as i64) * 1299709
            + bin.rem_euclid(1 << 20) * 15485863;
        (0..n)
            .map(|i| (((seed + i as i64 * 31) % 1000) as f32 / 1000.0 - 0.5))
            .collect()
    }
}

fn hu_preflop_tree() -> solver_core::tree::flat::FlatTree {
    // HU preflop, blinds [2,1] (gate-I convention), one bet + one
    // raise size so multiple flop-entry pots (limp pot, raised pots)
    // produce MULTIPLE seam buckets.
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 0,
        starting_stacks: vec![STACK - 2, STACK - 1],
        initial_contributions: vec![2, 1],
        rake_rate: 0.0,
        rake_cap: 0.0,
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
    build_tree_preflop_only(&cfg).expect("HU preflop-only tree")
}

#[test]
#[ignore = "v1 preflop runtime gate (minutes: 1755-canonical expansions × 2 arms); --ignored --nocapture --release"]
fn v1_preflop_runtime_collapse_gate() {
    let tree = hu_preflop_tree();
    let np = 2u8;
    let table = PreflopChanceTable::new(
        np,
        vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; np as usize],
    );
    let terminal_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);

    // Key census at the flop boundary.
    let chance_nodes: Vec<usize> = (0..tree.nodes.len())
        .filter(|&i| {
            tree.nodes[i].node_type == solver_core::tree::flat::NODE_TYPE_CHANCE
                && tree.nodes[i].num_children == 0
        })
        .collect();
    let key_fn = PreflopVectorCfr::seam_bucket_chance_key(&tree, np as usize, STACK);
    let distinct_keys: HashSet<u64> = chance_nodes.iter().map(|&i| key_fn(i)).collect();
    let distinct_cells: HashSet<(u8, i64)> = chance_nodes
        .iter()
        .map(|&i| SeamCell::at_chance_node(&tree, i, np as usize).bucket_key(STACK))
        .collect();
    eprintln!(
        "tree {} nodes, {} flop entries, {} distinct bucket keys",
        tree.nodes.len(),
        chance_nodes.len(),
        distinct_keys.len()
    );
    assert_eq!(distinct_keys.len(), distinct_cells.len(), "key packing must be injective");
    assert!(distinct_keys.len() >= 3, "fixture must exercise multiple buckets");

    const ITERS: u32 = 3;

    // Arm A: uncollapsed.
    let mut a = PreflopVectorCfr::new(&tree);
    let mut oracle_a = BucketStubOracle { keys_seen: HashSet::new() };
    for _ in 0..ITERS {
        a.run_one_iteration(&tree, &table, &mut oracle_a, &terminal_fn);
    }

    // Arm B: shared-chance collapse with the v1 seam-bucket key.
    let mut b = PreflopVectorCfr::new(&tree);
    let mut oracle_b = BucketStubOracle { keys_seen: HashSet::new() };
    for _ in 0..ITERS {
        let key_fn = PreflopVectorCfr::seam_bucket_chance_key(&tree, np as usize, STACK);
        b.run_one_iteration_shared_chance(&tree, &table, &mut oracle_b, &terminal_fn, key_fn);
    }

    // Gate 3: both arms saw the same bucket set.
    assert_eq!(oracle_a.keys_seen, oracle_b.keys_seen, "bucket sets must agree");
    eprintln!("buckets exercised: {:?}", {
        let mut v: Vec<_> = oracle_b.keys_seen.iter().collect();
        v.sort();
        v
    });

    // Gate 2: bit-exact state equality.
    for (name, va, vb) in [
        ("regrets", &a.regrets, &b.regrets),
        ("cum_strategy", &a.cum_strategy, &b.cum_strategy),
        ("strategy", &a.strategy, &b.strategy),
    ] {
        assert_eq!(va.len(), vb.len());
        for i in 0..va.len() {
            assert_eq!(
                va[i].to_bits(),
                vb[i].to_bits(),
                "{name}[{i}]: uncollapsed {} vs collapsed {} — the bucket-keyed \
                 collapse must be EXACT for a (flop, traverser, bucket) oracle",
                va[i],
                vb[i]
            );
        }
    }
    eprintln!(
        "v1 preflop runtime gate PASSED: cell routing enforced, bucket-keyed \
         collapse bit-exact over {ITERS} iters, {} buckets",
        distinct_keys.len()
    );
}

/// PRODUCTION-SCALE measurement: the v1 preflop layer on the CAP-3
/// FULL-LADDER tree (1.90M nodes / 883k infosets). Measures buffer
/// allocation, the bucket-key census, per-unit costs, and projects the
/// shared-chance iteration; runs ONE real iteration only if the
/// projection is sane (the bootstrap probe's stop-and-report
/// protocol).
#[test]
#[ignore = "production-scale preflop measurement; --ignored --nocapture --release"]
fn v1_preflop_runtime_production_measure() {
    use solver_core::tree::action::{production_game_v1, BetCap};
    use solver_core::tree::flat::MAX_NA_PREFLOP;
    use std::time::Instant;

    let spec = production_game_v1();
    let max_raise_count = MAX_NA_PREFLOP.saturating_sub(2);
    let mut cfg = spec.preflop_tree_config(BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..max_raise_count)
            .map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64))
            .collect(),
    });
    cfg.max_bets_per_street = BetCap::all(3);
    let t0 = Instant::now();
    let tree = build_tree_preflop_only(&cfg).expect("cap-3 tree");
    eprintln!("tree: {} nodes in {:.1?}", tree.nodes.len(), t0.elapsed());

    let t0 = Instant::now();
    let solver = PreflopVectorCfr::new(&tree);
    let buf_gb = solver.regrets.len() as f64 * 4.0 * 3.0 / 1e9;
    eprintln!(
        "solver: {} infosets, 3 buffers × {} floats = {:.1} GB total, alloc {:.1?}",
        solver.infoset_count,
        solver.regrets.len(),
        buf_gb,
        t0.elapsed()
    );

    // Bucket-key census at the flop boundary.
    let key_fn = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, spec.stack);
    let mut keys: HashSet<u64> = HashSet::new();
    let mut n_chance = 0usize;
    for i in 0..tree.nodes.len() {
        let n = &tree.nodes[i];
        if n.node_type == solver_core::tree::flat::NODE_TYPE_CHANCE && n.num_children == 0 {
            n_chance += 1;
            keys.insert(key_fn(i));
        }
    }
    let allin_keys = keys.iter().filter(|&&k| (k as u32) == (i32::MIN as u32)).count();
    eprintln!(
        "flop entries {n_chance} | distinct bucket keys {} ({} all-in sentinels = free)",
        keys.len(),
        allin_keys
    );

    // Unit probes → projection (the v1 cost model: per-key chance CFV
    // ≈ 0.13s at np=6 measured; re-measure here with the stub).
    let table =
        PreflopChanceTable::new(6, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 6]);
    let mut oracle = BucketStubOracle { keys_seen: HashSet::new() };
    let reach_probe = {
        // one chance-CFV at production: pick the first boundary node.
        let idx = (0..tree.nodes.len())
            .find(|&i| {
                tree.nodes[i].node_type == solver_core::tree::flat::NODE_TYPE_CHANCE
                    && tree.nodes[i].num_children == 0
            })
            .unwrap();
        let mut s2 = PreflopVectorCfr::new(&tree);
        s2.compute_preflop_strategy(&tree);
        let reach = s2.compute_preflop_reach(&tree, None);
        let cell = SeamCell::at_chance_node(&tree, idx, 6);
        let fmask = tree.get_folded_mask(idx);
        let t0 = Instant::now();
        let _ = s2.compute_chance_node_cfv_with_expansion_for_cell(
            idx, 0, &reach, &table, &mut oracle, cell, fmask,
        );
        t0.elapsed().as_secs_f64()
    };
    let solve_keys = keys.len() - allin_keys;
    // All-in keys still cost a chance-CFV call (stub answers instantly;
    // production all-in oracle = cached equity rollout, also cheap).
    let chance_s = keys.len() as f64 * reach_probe * 6.0;
    eprintln!(
        "per-key chance CFV: {reach_probe:.3}s → {} keys × 6 traversers ≈ {chance_s:.0}s/iter \
         ({} solve keys + {} all-in)",
        keys.len(),
        solve_keys,
        allin_keys
    );

    const BUDGET_S: f64 = 1200.0;
    let n_term = (0..tree.nodes.len())
        .filter(|&i| tree.nodes[i].node_type == 0)
        .count();
    let term_s = n_term as f64 * 0.063e-3 * 6.0; // bootstrap-measured per-terminal
    let projected = chance_s + term_s;
    eprintln!(
        "projected shared-chance iteration: chance {chance_s:.0}s + terminals \
         ({n_term} × 0.063ms × 6) {term_s:.0}s ≈ {projected:.0}s single-core"
    );
    if projected > BUDGET_S {
        eprintln!("⛔ STOP-AND-REPORT: projection {projected:.0}s exceeds budget {BUDGET_S:.0}s");
        return;
    }
    let terminal_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    let mut s = solver;
    let t0 = Instant::now();
    let key_fn = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, spec.stack);
    s.run_one_iteration_shared_chance(&tree, &table, &mut oracle, &terminal_fn, key_fn);
    eprintln!("MEASURED one shared-chance iteration: {:.1?}", t0.elapsed());
}
