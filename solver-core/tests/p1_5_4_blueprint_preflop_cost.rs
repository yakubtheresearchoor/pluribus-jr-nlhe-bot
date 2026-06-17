//! Blueprint close-out: the PREFLOP PRODUCTION LAYER — the one unpriced
//! line item in the B4 end-to-end projection
//! (p1_5_4_bucketing_b4_cost_measurement.rs, item 4).
//!
//! === What the blueprint loop actually is (composition) ===
//!   ORACLE FILL: postflop solves per (canonical flop) — the B4 ladder
//!   line (B=8 full-set/1%: 10.5h at 16 cores). Frozen-CFV design:
//!   CFVs cached per (flop, traverser), NOT multiplied by preflop
//!   iterations. Refresh cadence is a design input (each refresh costs
//!   one more fill).
//!   PREFLOP ITERATIONS (this test prices them): per iteration, per
//!   traverser (6), per preflop chance node: expand 169-class reach to
//!   per-combo (per canonical, ~1326 combos × 6 players), frozen-
//!   oracle lookup, reduce combo→class, orbit-weighted aggregate over
//!   1755 canonicals — plus the bottom-up walk with pairwise fold
//!   terminals (5 matvecs of 169×169 per terminal per traverser).
//!
//! === Gate axis (named, not re-run) ===
//! The production path is the SAME code path the Phase 4 bootstrap
//! validated: pairwise terminal bias direction unit-tested
//! (preflop_terminal.rs pairwise_tests), expand/reduce seam anchored
//! (P5a/P5b), frozen-CFV reach-independence certified by the
//! two-fidelity band-stability check (research scale). Nothing
//! structurally new executes at production scale — only larger inputs
//! through gated arithmetic. The reach-independence approximation
//! remains research-scale-certified only (same instrument wall as the
//! count curve; production confirmation belongs to head-to-head).
//!
//! Protocol: census + unit probes → printed projection → budget guard
//! → full measured iteration only if projected sane.
//!
//! ═══ MEASURED 2026-06-10 (rich preflop tree, MAX_NA_PREFLOP=16,
//! frozen stub oracle, single core) ═══
//!   census: 239,467 nodes, 7,466 preflop infosets, 1,493 chance
//!     nodes (flop transitions), 7,465 fold terminals;
//!   one chance-node CFV over 1755 canonicals: 0.10s;
//!   one pairwise fold terminal (dense 169): 0.058ms;
//!   UNCOLLAPSED full iteration: 897.1s (projection said 892 — 1%),
//!     15,721,290 oracle lookups — the cost is 6 real chance-CFV
//!     computations plus 8,952 bit-identical recomputations;
//!   SHARED-CHANCE COLLAPSE (run_one_iteration_shared_chance, gated
//!     bit-exact across regrets+cum+strategy): steady-state iteration
//!     3.4s — 264×. THE PRODUCTION PREFLOP LINE: 3.4s/iter single
//!     core; even 1,000 preflop iterations < 1h. The layer is priced
//!     and NEGLIGIBLE next to the postflop fill.
//!   Caveat: collapse validity = oracle reach-independence within an
//!   epoch (the frozen-CFV design's existing approximation, research-
//!   scale-certified by the Phase 4 band-stability check); pot-bucket
//!   keyed oracles use chance_key = pot bucket (n_keys × 0.10s × 6
//!   per iteration — still seconds for any sane bucket count).

use solver_core::card::Card;
use solver_core::solver::postflop_oracle::PostflopValueOracle;
use solver_core::solver::preflop_cfr::{
    make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr,
};
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::preflop_start_game::{flop_combo_layout, PreflopChanceTable};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};
use std::collections::HashMap;
use std::time::Instant;

const PROJECTION_BUDGET_S: f64 = 900.0;

/// Rich 6-max preflop tree — RETARGETED 2026-06-12 to PRODUCTION GAME
/// v1 (GameSpec-derived: no ante, blinds 1/2 units, equal-total 200
/// stacks, button 5) with the MAX_NA_PREFLOP=16 raise ladder, built
/// PREFLOP-ONLY (flop-entry chance nodes are oracle leaves — the
/// production layer never descends past them, and the preflop ladder
/// is not a legal postflop abstraction).
///
/// HISTORY: the bootstrap config (starting_pot 3 + blinds in
/// contributions, flat [100;6] stacks) double-counted the blinds under
/// the pinned additive semantics and gave the blinds deeper totals;
/// its 2026-06-10 numbers (239,467 nodes / 7,466 infosets / 3.4s
/// shared-chance iter) also predate the fold-continuation tree fix.
fn build_rich_preflop_tree() -> FlatTree {
    use solver_core::tree::action::{production_game_v1, BetCap};
    use solver_core::tree::builder::build_tree_preflop_only;
    let max_raise_count = MAX_NA_PREFLOP.saturating_sub(2);
    let raises: Vec<BetSize> = (0..max_raise_count)
        .map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64))
        .collect();
    let mut pre_cfg = production_game_v1().preflop_tree_config(BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: raises,
    });
    // PRODUCTION cap = 3 (v1_depth_cap_ab: smallest defense-complete). The
    // earlier 3.4s number was UNCAPPED + pre-fold-fix (7,466 infosets); the
    // real production tree is cap-3 (~654k infosets at MAX_NA_PREFLOP=8).
    pre_cfg.max_bets_per_street = BetCap::all(3);
    build_tree_preflop_only(&pre_cfg).expect("v1 preflop tree builds")
}

/// Frozen-CFV stub oracle: cache per canonical flop, lookup + clone per
/// call — the cost shape of the production frozen oracle (CFVs cached
/// per (flop, traverser), reach-independent within an oracle epoch).
/// Values are synthetic (cost doesn't depend on them).
struct FrozenStubOracle {
    cache: HashMap<[Card; 3], Vec<f32>>,
    calls: usize,
}

impl PostflopValueOracle for FrozenStubOracle {
    fn flop_root_cfv(
        &mut self,
        canonical_flop: [Card; 3],
        _combo_ranges: &[Vec<f32>],
        _traverser: u8,
    ) -> Vec<f32> {
        self.calls += 1;
        self.cache
            .entry(canonical_flop)
            .or_insert_with(|| {
                let n = flop_combo_layout(canonical_flop).len();
                (0..n).map(|i| (i as f32 * 0.001).sin()).collect()
            })
            .clone()
    }
}

#[test]
#[ignore = "blueprint preflop-layer cost probe (~minutes if projection sane); run with --ignored --nocapture"]
fn blueprint_preflop_iteration_cost() {
    eprintln!("\n════ Blueprint preflop production layer: cost probe ════");

    let tree = build_rich_preflop_tree();
    let np = 6usize;

    // Dense production ranges: every class reachable.
    let ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let t0 = Instant::now();
    let table = PreflopChanceTable::new(np as u8, ranges);
    eprintln!(
        "PreflopChanceTable: {} canonical flops, built in {:.1}s",
        table.num_canonical_flops(),
        t0.elapsed().as_secs_f64()
    );

    let mut cfr = PreflopVectorCfr::new(&tree);
    let chance_nodes = cfr.preflop_chance_node_indices(&tree);

    // Census.
    let n_terminals = (0..tree.num_nodes())
        .filter(|&i| {
            tree.nodes[i].is_terminal()
                && tree.nodes[i].board_state == BoardState::Preflop as u8
        })
        .count();
    eprintln!(
        "census: {} nodes, {} preflop infosets, {} chance nodes (flop transitions), \
         {} preflop fold terminals",
        tree.num_nodes(),
        cfr.infoset_count(),
        chance_nodes.len(),
        n_terminals,
    );

    // ── Unit probes ──
    let mut oracle = FrozenStubOracle { cache: HashMap::new(), calls: 0 };

    // Warm the oracle cache OUTSIDE the timed probe (production fill is
    // the B4 postflop ladder line, priced separately).
    cfr.compute_preflop_strategy(&tree);
    let reach = cfr.compute_preflop_reach(&tree, None);
    let t0 = Instant::now();
    let _ = cfr.compute_chance_node_cfv_with_expansion(
        chance_nodes[0], 0, &reach, &table, &mut oracle,
    );
    let warm_s = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let _ = cfr.compute_chance_node_cfv_with_expansion(
        chance_nodes[0], 0, &reach, &table, &mut oracle,
    );
    let chance_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "one chance-node CFV over {} canonicals: cold {:.2}s, warm {:.2}s \
         (expand + lookup + reduce + aggregate)",
        table.num_canonical_flops(),
        warm_s,
        chance_s
    );

    // Pairwise fold-terminal probe.
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    let term_idx = (0..tree.num_nodes())
        .find(|&i| {
            tree.nodes[i].is_terminal()
                && tree.nodes[i].board_state == BoardState::Preflop as u8
        })
        .expect("at least one preflop fold terminal");
    let dense: Vec<Vec<f32>> =
        (0..np).map(|_| vec![1.0 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]).collect();
    let reps = 100u32;
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = term_fn(term_idx, 0, &dense);
    }
    let term_s = t0.elapsed().as_secs_f64() / reps as f64;
    eprintln!("one pairwise fold terminal (dense 169): {:.3}ms", term_s * 1e3);

    // ── Projection ──
    let projected = 6.0 * chance_nodes.len() as f64 * chance_s
        + 6.0 * n_terminals as f64 * term_s;
    eprintln!(
        "\nprojected full iteration: 6 × {} chance × {:.2}s + 6 × {} terminals × {:.3}ms \
         ≈ {:.0}s",
        chance_nodes.len(),
        chance_s,
        n_terminals,
        term_s * 1e3,
        projected
    );

    if projected > PROJECTION_BUDGET_S {
        eprintln!(
            "⛔ STOP-AND-REPORT: projection {:.0}s exceeds budget {:.0}s — full \
             iteration not run; the projection is the finding.",
            projected, PROJECTION_BUDGET_S
        );
        return;
    }

    // ── Full measured iteration (frozen oracle) ──
    oracle.calls = 0;
    let t0 = Instant::now();
    cfr.run_one_iteration(&tree, &table, &mut oracle, term_fn);
    let iter_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "\n══ measured preflop iteration (frozen oracle): {:.1}s ({} oracle lookups) ══",
        iter_s, oracle.calls
    );
    eprintln!(
        "composition reminder: oracle FILL is the B4 postflop line (B=8 full-set/1%: \
         10.5h @ 16 cores), paid once per refresh epoch; preflop iterations ride on \
         frozen CFVs at the cost above."
    );
}

/// Gate + collapsed cost: `run_one_iteration_shared_chance` must be
/// BIT-EXACT to `run_one_iteration` with a reach-independent frozen
/// oracle (the collapse is control flow — identical float ops on
/// identical inputs, per the method docs), and its measured iteration
/// cost replaces the 897.1s uncollapsed line in the blueprint
/// projection.
///
/// Runs at a stride-sampled canonical SUBSET via the same engine path?
/// No — bit-exactness demands the same inputs, so the gate runs the
/// full 1755 canonicals for the uncollapsed arm ONCE (≈15 min, the
/// price of the gate) and the collapsed arm twice (gate iter +
/// timed iter).
#[test]
#[ignore = "collapse gate + measurement (~35 min: one uncollapsed full iteration); run with --ignored --nocapture"]
fn blueprint_preflop_shared_chance_gate_and_cost() {
    eprintln!("\n════ preflop shared-chance collapse: gate + cost ════");
    let tree = build_rich_preflop_tree();
    let np = 6usize;
    let ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let table = PreflopChanceTable::new(np as u8, ranges);

    // Arm A: uncollapsed (reference).
    let mut cfr_a = PreflopVectorCfr::new(&tree);
    let mut oracle_a = FrozenStubOracle { cache: HashMap::new(), calls: 0 };
    let term_a = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    let t0 = Instant::now();
    cfr_a.run_one_iteration(&tree, &table, &mut oracle_a, term_a);
    eprintln!("uncollapsed iteration: {:.1}s", t0.elapsed().as_secs_f64());

    // Arm B: collapsed (constant key — fully reach-independent oracle).
    let mut cfr_b = PreflopVectorCfr::new(&tree);
    let mut oracle_b = FrozenStubOracle { cache: HashMap::new(), calls: 0 };
    let term_b = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    let t0 = Instant::now();
    cfr_b.run_one_iteration_shared_chance(&tree, &table, &mut oracle_b, term_b, |_| 0);
    let collapsed_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "collapsed iteration: {:.1}s ({} oracle lookups vs {} uncollapsed)",
        collapsed_s, oracle_b.calls, oracle_a.calls
    );

    // Bit-exact gate across all persistent state.
    for (label, a, b) in [
        ("regrets", &cfr_a.regrets, &cfr_b.regrets),
        ("cum_strategy", &cfr_a.cum_strategy, &cfr_b.cum_strategy),
        ("strategy", &cfr_a.strategy, &cfr_b.strategy),
    ] {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "{label}[{i}]: uncollapsed {} vs collapsed {} — the collapse \
                 must be bit-exact by construction; drift is a bug",
                a[i], b[i]
            );
        }
    }
    eprintln!("collapse gate PASSED: regrets + cum_strategy + strategy bit-exact");

    // Timed steady-state collapsed iteration (oracle cache warm).
    let t0 = Instant::now();
    let term_b2 = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    cfr_b.run_one_iteration_shared_chance(&tree, &table, &mut oracle_b, term_b2, |_| 0);
    eprintln!(
        "steady-state collapsed iteration: {:.1}s — the production preflop line",
        t0.elapsed().as_secs_f64()
    );
}
