//! V1 EQUILIBRIUM REACH (2026-06-12): sharpen the reach×pot allocation
//! lens from the conservative uniform-strategy upper bound to a real
//! bootstrap equilibrium. Slice-2 unblocks this (6-max flop entries
//! have folded players). The postflop value source is a FROZEN iter0
//! live-subset oracle (the production frozen-CFV design): one cache
//! fill, then cheap preflop iterations converge the strategy against it.
//!
//! Measure-before-launch: `eq_reach_cost_probe` times ONE iteration
//! (fill-dominated) and ONE cached iteration before any long run.
//! `eq_reach_bootstrap` runs the solve and reports the equilibrium
//! reach×pot table (only sharpens the uniform bound — equilibrium folds
//! more under 5% rake, so live-5/6 shares can only shrink).
//!
//! COST FINDING (2026-06-12, measure-before-launch paid off): the fill
//! iteration with this EXACT iter0 source ran > 41 min without
//! finishing — hot path `sorted_sweep_with_rake_components` inside
//! `compute_v_flop_at_root_iter0`. The source does FULL EXACT multiway
//! showdowns (no bucketing) × 1755 canonicals × ~124 buckets × 6
//! traversers; at np≥3 the exact showdown is combinatorial (the same
//! wall that made live-3 exact unpriceable earlier). VERDICT: the
//! equilibrium solve here is NOT a casual verification at exact
//! fidelity. Per the allocation framing it VERIFIES a known-direction
//! bound (equilibrium folds more ⇒ live-5/6 shares only shrink ⇒ the
//! degrade-the-tail call only strengthens), already triply-supported
//! (uniform bound + reach×pot re-derivation + L1 lever agreement), so
//! it is BACKGROUNDABLE-or-SKIP, not load-bearing. To background it
//! cheaply, swap in a BUCKETED (B=8) live-subset source — the same
//! abstraction the production fill uses (reusable, not throwaway) —
//! which collapses the showdown; left as a named follow-up.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::postflop_oracle::{BucketKeyedOracle, SeamCell};
use solver_core::solver::preflop_cfr::{
    make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr,
};
use solver_core::solver::preflop_start_game::{
    compute_v_flop_at_root_iter0, flop_combo_layout, PreflopChanceTable,
};
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, GameSpec};
use solver_core::tree::builder::{build_tree, build_tree_preflop_only};
use solver_core::tree::flat::{FlatTree, NODE_TYPE_CHANCE, MAX_NA_PREFLOP};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

/// Frozen iter0 live-subset value source for the bootstrap solve.
/// Caches the per-cell seam tree; solves iter0 on the LIVE-subset game.
fn iter0_live_subset_source(
    spec: GameSpec,
) -> impl FnMut(SeamCell, u16, [Card; 3], &[Vec<f32>], u8) -> Vec<f32> {
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let mut trees: HashMap<(u8, i32, i32), Arc<FlatTree>> = HashMap::new();
    move |cell, folded_mask, canonical, combo_reaches, traverser| {
        let tree = trees
            .entry((cell.live, cell.commit, cell.pot))
            .or_insert_with(|| {
                Arc::new(
                    build_tree(&spec.flop_seam_config(
                        cell.live,
                        cell.commit.min(spec.stack),
                        cell.pot,
                        bets.clone(),
                    ))
                    .expect("seam tree"),
                )
            })
            .clone();
        let layout = flop_combo_layout(canonical);
        let np = combo_reaches.len();
        let live_seats: Vec<usize> =
            (0..np).filter(|&p| (folded_mask >> p) & 1 == 0).collect();
        let trav_live =
            live_seats.iter().position(|&p| p == traverser as usize).expect("trav live");
        let mut full: Vec<Vec<f32>> =
            vec![vec![0.0f32; NUM_POSSIBLE_HANDS]; live_seats.len()];
        for (lp, &seat) in live_seats.iter().enumerate() {
            for (li, &(c1, c2)) in layout.iter().enumerate() {
                full[lp][card_pair_to_index(c1, c2)] = combo_reaches[seat][li];
            }
        }
        let (v_table, layout_table) =
            compute_v_flop_at_root_iter0(canonical, &tree, &full, trav_live as u8);
        let mut pos: HashMap<(Card, Card), usize> = HashMap::new();
        for (i, &(a, b)) in layout_table.iter().enumerate() {
            pos.insert((a, b), i);
        }
        layout.iter().map(|&(a, b)| v_table[pos[&(a, b)]]).collect()
    }
}

fn cap3_preflop_tree() -> FlatTree {
    let spec = production_game_v1();
    let max_raise_count = MAX_NA_PREFLOP.saturating_sub(2);
    let mut cfg = spec.preflop_tree_config(BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..max_raise_count)
            .map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64))
            .collect(),
    });
    cfg.max_bets_per_street = BetCap::all(3);
    build_tree_preflop_only(&cfg).expect("cap-3 tree")
}

/// PRODUCTION PER-ITER (2026-06-14): time the SHARED-CHANCE COLLAPSE
/// iteration on the cap-3 tree at MAX_NA_PREFLOP=8, using a CHEAP STUB
/// postflop source so we measure the preflop traversal cost (chance
/// collapse + fold terminals + infoset updates) WITHOUT the >41 min
/// exact-fill trap. The real fill is the postflop seam fill, priced
/// separately. Buffer ~10.6 GB after the 16→8 stride cut (was 28.6 GB).
#[test]
#[ignore = "production preflop per-iter (collapsed, stub source, ~10.6 GB); --ignored --nocapture --release"]
fn eq_reach_collapsed_periter_stub() {
    let spec = production_game_v1();
    let tree = cap3_preflop_tree();
    let table = PreflopChanceTable::new(6, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 6]);
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    // Cheap synthetic CFVs (cost is traversal+lookup, not the values).
    let stub = |_cell: SeamCell, _mask: u16, canonical: [Card; 3], _r: &[Vec<f32>], _t: u8| -> Vec<f32> {
        flop_combo_layout(canonical).iter().enumerate().map(|(i, _)| (i as f32 * 1e-4).sin()).collect()
    };
    let mut oracle = BucketKeyedOracle::new(spec.stack, 6, 0, stub);
    let mut solver = PreflopVectorCfr::new(&tree);
    eprintln!(
        "cap-3 tree: {} infosets; buffer ≈ {:.1} GB (3 × infosets × {} × {} × 4B)",
        solver.infoset_count(),
        3.0 * solver.infoset_count() as f64 * 8.0 * NUM_PREFLOP_CLASSES as f64 * 4.0 / 1e9,
        8, NUM_PREFLOP_CLASSES,
    );
    let t0 = Instant::now();
    let kf = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, spec.stack);
    solver.run_one_iteration_shared_chance(&tree, &table, &mut oracle, &term_fn, kf);
    let fill = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let kf = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, spec.stack);
    solver.run_one_iteration_shared_chance(&tree, &table, &mut oracle, &term_fn, kf);
    let cached = t0.elapsed().as_secs_f64();
    eprintln!("COLLAPSED per-iter (stub): fill {fill:.1}s, cached {cached:.1}s, {} distinct chance keys", oracle.cache_len());
    for n in [50u32, 100, 200] {
        eprintln!("  {n}-iter preflop bootstrap ≈ {:.1} min (cached-dominated)", n as f64 * cached / 60.0);
    }
}

#[test]
#[ignore = "equilibrium-reach COST PROBE (times 1 fill iter + 1 cached); --ignored --nocapture --release"]
fn eq_reach_cost_probe() {
    let spec = production_game_v1();
    let tree = cap3_preflop_tree();
    let table =
        PreflopChanceTable::new(6, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; 6]);
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    let mut oracle =
        BucketKeyedOracle::new(spec.stack, 6, 0, iter0_live_subset_source(spec.clone()));
    let mut solver = PreflopVectorCfr::new(&tree);

    let t0 = Instant::now();
    let key_fn = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, spec.stack);
    solver.run_one_iteration_shared_chance(&tree, &table, &mut oracle, &term_fn, key_fn);
    let fill = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let key_fn = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, spec.stack);
    solver.run_one_iteration_shared_chance(&tree, &table, &mut oracle, &term_fn, key_fn);
    let cached = t0.elapsed().as_secs_f64();

    eprintln!(
        "EQ-REACH COST: fill iter {fill:.1}s (oracle cache {} entries), cached iter {cached:.1}s",
        oracle.cache_len()
    );
    eprintln!(
        "PROJECT 50-iter bootstrap ≈ {:.1} min (fill once + 49 cached)",
        (fill + 49.0 * cached) / 60.0
    );
}

/// The bootstrap solve + equilibrium reach×pot table. ITERS chosen
/// after the cost probe; frozen oracle (refresh_every=0).
#[test]
#[ignore = "equilibrium-reach BOOTSTRAP (run after the cost probe); --ignored --nocapture --release"]
fn eq_reach_bootstrap() {
    const ITERS: u32 = 60;
    let spec = production_game_v1();
    let tree = cap3_preflop_tree();
    let np = 6usize;
    let nc = NUM_PREFLOP_CLASSES;
    let table = PreflopChanceTable::new(6, vec![vec![1.0f32; nc]; 6]);
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    let mut oracle =
        BucketKeyedOracle::new(spec.stack, 6, 0, iter0_live_subset_source(spec.clone()));
    let mut solver = PreflopVectorCfr::new(&tree);

    let t0 = Instant::now();
    for it in 0..ITERS {
        let key_fn = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, spec.stack);
        solver.run_one_iteration_shared_chance(&tree, &table, &mut oracle, &term_fn, key_fn);
        if it == 0 || (it + 1) % 20 == 0 {
            eprintln!("  iter {} ({:.1?})", it + 1, t0.elapsed());
        }
    }
    eprintln!("bootstrap {ITERS} iters in {:.1?}", t0.elapsed());

    // Reach from the converged AVERAGE strategy.
    solver.compute_preflop_strategy(&tree); // refresh strategy from regrets
    let reach = solver.compute_preflop_reach(&tree, None);

    let mut by_live: BTreeMap<u8, (f64, f64)> = BTreeMap::new(); // reach, reach×pot
    let mut total_mass = 0.0f64;
    let mut total_rp = 0.0f64;
    for idx in 0..tree.nodes.len() {
        let n = &tree.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.num_children != 0 {
            continue;
        }
        let base = idx * nc;
        let mut joint = 1.0f64;
        for p in 0..np {
            joint *= reach[p][base..base + nc].iter().map(|&x| x as f64).sum::<f64>()
                / nc as f64;
        }
        let cell = SeamCell::at_chance_node(&tree, idx, np);
        let pot_bb = cell.pot as f64 / 2.0;
        let e = by_live.entry(cell.live).or_default();
        e.0 += joint;
        e.1 += joint * pot_bb;
        total_mass += joint;
        total_rp += joint * pot_bb;
    }

    let row_h: BTreeMap<u8, f64> =
        [(2u8, 0.5f64), (3, 3.0), (4, 13.0), (5, 36.0), (6, 58.0)].into_iter().collect();
    eprintln!("\n═══ EQUILIBRIUM REACH×POT (sharpened vs uniform bound) ═══");
    eprintln!("live | reach%% | reach×pot%% | bill h");
    for (&live, &(mass, rp)) in &by_live {
        eprintln!(
            "  {live}  | {:5.2}% | {:9.2}% | {:5.0}",
            100.0 * mass / total_mass,
            100.0 * rp / total_rp,
            row_h[&live]
        );
    }
    eprintln!(
        "\nCompare to the uniform upper bound (live-5 16.3%%, live-6 3.2%% reach×pot): \
         equilibrium folds more, so these should be ≤ those."
    );
}
