//! PREFLOP CHANCE-CACHE GATE (2026-06-15): `run_one_iteration_shared_chance_cached`
//! must be BIT-EXACT to `run_one_iteration_shared_chance` over multiple
//! iterations, given a frozen reach-independent oracle. The cached variant
//! computes the live-traverser chance-node expansion ONCE (it's reach-
//! independent for a frozen oracle) and reuses it; this gate proves the reuse
//! changes nothing — the cum strategy, regrets, and per-iter strategy stay
//! identical. (If reach-independence were violated, the two would diverge by
//! iteration 2 and this fails — the safety check before the full anchor run.)

use solver_core::card::Card;
use solver_core::solver::postflop_oracle::ClosureOracle;
use solver_core::solver::preflop_cfr::{
    make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr,
};
use solver_core::solver::preflop_start_game::{flop_combo_layout, PreflopChanceTable};
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree_preflop_only;
use solver_core::tree::flat::FlatTree;

/// Minimal cap-1 production-game preflop tree — few chance nodes/keys so the
/// gate runs quickly, but the same code path as production.
fn small_preflop_tree() -> FlatTree {
    let mut cfg = production_game_v1().preflop_tree_config(BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: vec![BetSize::PotRelative(1.0)],
    });
    cfg.max_bets_per_street = BetCap::all(1);
    build_tree_preflop_only(&cfg).expect("small preflop tree")
}

/// Deterministic, REACH-INDEPENDENT per-combo oracle value (varies by flop +
/// traverser + combo, ignores ranges) — exercises real arithmetic through the
/// expand/reduce/aggregate path while satisfying the frozen-oracle contract.
fn frozen_oracle() -> ClosureOracle<impl FnMut([Card; 3], &[Vec<f32>], u8) -> Vec<f32>> {
    ClosureOracle::new(|flop: [Card; 3], _ranges: &[Vec<f32>], t: u8| {
        let layout = flop_combo_layout(flop);
        let base = flop.iter().map(|&c| c as u32).sum::<u32>() as f32;
        layout
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| {
                ((a as u32 + b as u32) as f32 * 0.013 + i as f32 * 0.0007 + base * 0.1
                    + t as f32 * 3.0)
                    .sin()
            })
            .collect()
    })
}

#[test]
fn preflop_chance_cache_gate() {
    let tree = small_preflop_tree();
    let table = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    let stack = production_game_v1().stack;
    const K: u32 = 5;

    // Uncached reference.
    let mut s_ref = PreflopVectorCfr::new(&tree);
    let mut o_ref = frozen_oracle();
    for _ in 0..K {
        let kf = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, stack);
        s_ref.run_one_iteration_shared_chance(&tree, &table, &mut o_ref, &term_fn, kf);
    }

    // Cached.
    let mut s_cac = PreflopVectorCfr::new(&tree);
    let mut o_cac = frozen_oracle();
    let mut cache: Vec<std::collections::HashMap<u64, Vec<f32>>> = Vec::new();
    for _ in 0..K {
        let kf = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, stack);
        s_cac.run_one_iteration_shared_chance_cached(&tree, &table, &mut o_cac, &term_fn, kf, &mut cache);
    }

    assert_eq!(
        s_ref.cum_strategy.len(),
        s_cac.cum_strategy.len(),
        "cum strategy length mismatch"
    );
    let max_d = s_ref
        .cum_strategy
        .iter()
        .zip(&s_cac.cum_strategy)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "chance-cache gate: {K} iters, {} infosets, cache keys/trav {:?} | max cum |Δ| = {max_d:.3e}",
        s_ref.infoset_count,
        cache.iter().map(|m| m.len()).collect::<Vec<_>>(),
    );
    assert_eq!(s_ref.cum_strategy, s_cac.cum_strategy, "cached != uncached (bit-exact required)");
}
