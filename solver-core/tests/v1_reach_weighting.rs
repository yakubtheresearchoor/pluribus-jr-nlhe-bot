//! V1 REACH MEASUREMENT (lever L5, 2026-06-12) — the free,
//! quality-PRESERVING cut. The honest bill (110h GPU) concentrates 96%
//! of cost in live-5/6 (limp families). Reach-weighting spends compute
//! where equilibrium play actually goes, which is categorically unlike
//! the rejected coverage lever (that traded quality on reached spots).
//!
//! CONSERVATIVE FIRST CUT: reach under the UNIFORM preflop strategy.
//! A 5%-raked equilibrium FOLDS MORE than uniform (rake taxes
//! marginal continues), so uniform-strategy reach is an UPPER BOUND on
//! every multiway family's equilibrium mass: if live-5/6 is already
//! small under uniform, it is smaller at equilibrium. No expensive
//! preflop solve required for the bound.
//!
//! Per-bucket reach proxy: uniform-class-prior joint arrival =
//! Π_player mean_class reach[p][node]. NAMED approximation — assumes
//! card-removal independence across players (the same assumption the
//! preflop CLASS abstraction already makes) and a uniform class prior.
//! Enough to answer "is the live-6 row reachable mass or rare limp
//! subtrees", which needs order-of-magnitude, not precision.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::postflop_oracle::SeamCell;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree_preflop_only;
use solver_core::tree::flat::{NODE_TYPE_CHANCE, MAX_NA_PREFLOP};
use std::collections::BTreeMap;

#[test]
#[ignore = "L5 reach measurement (~minutes: reach DFS over the cap-3 tree); --ignored --nocapture --release"]
fn v1_reach_weighting_uniform_bound() {
    let spec = production_game_v1();
    let max_raise_count = MAX_NA_PREFLOP.saturating_sub(2);
    let mut cfg = spec.preflop_tree_config(BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..max_raise_count)
            .map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64))
            .collect(),
    });
    cfg.max_bets_per_street = BetCap::all(3);
    let t0 = std::time::Instant::now();
    let tree = build_tree_preflop_only(&cfg).expect("cap-3 tree");
    let np = spec.num_players as usize;
    let nc = NUM_PREFLOP_CLASSES;

    // Fresh solver = uniform strategy; reach DFS under it.
    let mut solver = PreflopVectorCfr::new(&tree);
    solver.compute_preflop_strategy(&tree); // uniform from zero regrets
    let reach = solver.compute_preflop_reach(&tree, None);
    eprintln!("reach computed in {:.1?} ({} nodes)", t0.elapsed(), tree.nodes.len());

    // Per flop-entry chance node: uniform-class-prior joint arrival.
    // by_live[live] = (summed reach mass, bucket-key set, node count)
    let mut by_live: BTreeMap<u8, (f64, std::collections::HashSet<i64>, usize)> =
        BTreeMap::new();
    let mut total_mass = 0.0f64;
    // Also accumulate per (live, SPR-bin) so we can see WHICH live-5/6
    // buckets carry the mass.
    let mut by_bucket: BTreeMap<(u8, i64), f64> = BTreeMap::new();
    let mut rp_by_bucket: BTreeMap<(u8, i64), f64> = BTreeMap::new();
    let mut rp_by_live: BTreeMap<u8, f64> = BTreeMap::new();
    for idx in 0..tree.nodes.len() {
        let n = &tree.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.num_children != 0 {
            continue;
        }
        let base = idx * nc;
        let mut joint = 1.0f64;
        for p in 0..np {
            let mean_c: f64 =
                reach[p][base..base + nc].iter().map(|&x| x as f64).sum::<f64>() / nc as f64;
            joint *= mean_c;
        }
        let cell = SeamCell::at_chance_node(&tree, idx, np);
        let (live, bin) = cell.bucket_key(spec.stack);
        total_mass += joint;
        let e = by_live.entry(live).or_default();
        e.0 += joint;
        e.1.insert(bin);
        e.2 += 1;
        *by_bucket.entry((live, bin)).or_default() += joint;
        // reach × pot (in bb) — the EV-error weight: a strategy mistake
        // in a big pot costs more chips, so degrade-tolerance ∝ stakes.
        let pot_bb = cell.pot as f64 / 2.0;
        *rp_by_bucket.entry((live, bin)).or_default() += joint * pot_bb;
        *rp_by_live.entry(live).or_default() += joint * pot_bb;
    }

    eprintln!("\n═══ UNIFORM-STRATEGY REACH (upper bound on equilibrium mass) ═══");
    eprintln!("live | reach %% | buckets | flop-entry nodes");
    for (&live, (mass, bins, nodes)) in &by_live {
        eprintln!(
            "  {live}  | {:6.2}% | {:5} | {nodes}",
            100.0 * mass / total_mass,
            bins.len()
        );
    }

    // The bill rows (honest, measured-only levers).
    let row_h: BTreeMap<u8, f64> =
        [(2u8, 0.5f64), (3, 3.0), (4, 13.0), (5, 36.0), (6, 58.0)].into_iter().collect();
    let bill: f64 = row_h.values().sum();

    // ── REACH × POT (the EV-error / allocation weight) ──
    // Degrading a bucket injects a strategy error; its EV cost ∝ how
    // OFTEN the bucket occurs (reach) × how much MONEY is at stake
    // (pot). This is the weight that decides fidelity, not reach alone.
    // Cross it with the per-family GPU cost: is the expensive tail also
    // the high-stakes tail (must-keep), or low-stakes (safe to degrade)?
    let total_rp: f64 = rp_by_live.values().sum();
    eprintln!("\n═══ REACH×POT IMPORTANCE vs COST (the allocation lens) ═══");
    eprintln!("live | reach%% | reach×pot%% | bill h | cost-per-importance");
    for (&live, &rp) in &rp_by_live {
        let reach_pct = 100.0 * by_live[&live].0 / total_mass;
        let rp_pct = 100.0 * rp / total_rp;
        let h = row_h[&live];
        eprintln!(
            "  {live}  | {reach_pct:5.2}% | {rp_pct:9.2}% | {h:5.0} | {:.2} h/%imp",
            h / rp_pct.max(1e-9)
        );
    }
    eprintln!(
        "\nREADING: the family whose reach×pot share is FAR below its bill-hour\n\
         share is the safe-to-degrade tail; a family whose reach×pot is\n\
         comparable to its cost must keep fidelity. (Low-reach ≠ safe if the\n\
         pot is big — the user's flag: the tail can be high-stakes.)"
    );
    // ILLUSTRATIVE ONLY (see caveat below): a delete-below-threshold
    // model. The REAL policy degrades low-reach buckets (min runouts/
    // iters or an equity-rollout fallback), it does NOT omit them —
    // omitting would leave the preflop solve blind on lines that route
    // to the dropped bucket. The honest signal is the reach
    // distribution above (live-5/6 = ~14% of mass), not these numbers.
    eprintln!("\n═══ DELETE-THRESHOLD PROJECTION (illustrative, NOT the policy) ═══");
    for thresh_pct in [0.0f64, 0.01, 0.1, 0.5] {
        // A bucket survives if its reach fraction ≥ threshold. Cost is
        // charged per-FAMILY proportionally to surviving buckets in
        // that family (the family row is the sum over its buckets).
        let mut kept_h = 0.0f64;
        for (&live, row) in &row_h {
            let fam_buckets: Vec<f64> = by_bucket
                .iter()
                .filter(|((l, _), _)| *l == live)
                .map(|(_, &m)| m)
                .collect();
            let fam_mass: f64 = fam_buckets.iter().sum();
            if fam_mass == 0.0 {
                continue;
            }
            let kept: usize = fam_buckets
                .iter()
                .filter(|&&m| 100.0 * m / total_mass >= thresh_pct)
                .count();
            let frac = kept as f64 / fam_buckets.len().max(1) as f64;
            kept_h += row * frac;
        }
        eprintln!(
            "  threshold {thresh_pct:.2}%: bill {bill:.0}h → {kept_h:.1}h GPU",
        );
    }
    eprintln!(
        "\nNAMED: uniform-strategy upper bound + uniform-class-prior arrival proxy. \
         If live-5/6 is already thin here, equilibrium (more folding under 5%% rake) \
         makes it thinner — the cut is conservative. Equilibrium refinement = a real \
         preflop solve (135s/iter), run only if this bound leaves a budget problem."
    );
}
