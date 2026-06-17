//! V1 SEAM CENSUS (measurement instrument, 2026-06-12): enumerate the
//! production-game-v1 preflop tree's flop-entry chance nodes and bin
//! them into SEAM CELLS (live players, per-live commit, pot). Each
//! distinct cell is a flop-start game the postflop oracle must cover
//! (pot-bucket-keyed frozen-CFV design); the census tells us how many
//! cells exist, their multiplicities, and (for representative cells)
//! the flop-tree node counts that drive blueprint pricing.
//!
//! Also gates the v1 preflop config itself: builds, poker-rules clean
//! (the standing tree_correctness_gate covers legality on its own
//! configs; here we assert structural seam invariants — every flop
//! entry has equal live commits and pot = Σ contribs).

use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::{build_tree, build_tree_preflop_only};
use solver_core::tree::flat::{NODE_TYPE_CHANCE, MAX_NA_PREFLOP};
use std::collections::BTreeMap;

fn preflop_bets() -> BetSizeOptions {
    // The bootstrap's production preflop abstraction (verbatim shape):
    // one open size + a pot-relative raise ladder filling MAX_NA_PREFLOP.
    let max_raise_count = MAX_NA_PREFLOP.saturating_sub(2);
    BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..max_raise_count)
            .map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64))
            .collect(),
    }
}

#[test]
#[ignore = "census instrument (~minutes on the fold-continuation tree); run with --ignored --nocapture"]
fn v1_seam_census() {
    let spec = production_game_v1();
    let cfg = spec.preflop_tree_config(preflop_bets());
    let t0 = std::time::Instant::now();
    let tree = build_tree_preflop_only(&cfg).expect("v1 preflop tree");
    let nn = tree.nodes.len();
    eprintln!("v1 preflop tree: {} nodes (built in {:.1?})", nn, t0.elapsed());

    let np = spec.num_players as usize;
    // Seam cells: (live, commit, pot) → count of flop-entry chance nodes.
    let mut cells: BTreeMap<(u8, i32, i32), usize> = BTreeMap::new();
    let mut n_chance = 0usize;
    for idx in 0..nn {
        let n = &tree.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.board_state != BoardState::Flop as u8 {
            continue;
        }
        n_chance += 1;
        let mask = tree.get_folded_mask(idx);
        let contribs: Vec<i32> =
            (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let live: Vec<usize> = (0..np).filter(|&p| mask & (1 << p) == 0).collect();
        let pot: i32 = cfg.starting_pot + contribs.iter().sum::<i32>();
        // SEAM INVARIANT: all live players have equal commits at the
        // flop, each at least the dead-money max, unless all-in short.
        let c0 = contribs[live[0]];
        for &p in &live {
            assert!(
                contribs[p] == c0 || contribs[p] >= spec.stack,
                "node {idx}: unequal live commits {:?} (live {:?})",
                contribs,
                live
            );
        }
        *cells.entry((live.len() as u8, c0, pot)).or_default() += 1;
    }
    eprintln!("flop-entry chance nodes: {n_chance}");
    let mut by_live: BTreeMap<u8, usize> = BTreeMap::new();
    for (&(live, _, _), &count) in &cells {
        *by_live.entry(live).or_default() += count;
    }
    eprintln!("cells: {} distinct (live, commit, pot) | entries by live count:", cells.len());
    for (live, count) in &by_live {
        let ncells = cells.keys().filter(|k| k.0 == *live).count();
        eprintln!("  {live} players: {count} flop entries across {ncells} cells");
    }
    eprintln!("top 20 cells by multiplicity:");
    let mut ranked: Vec<_> = cells.iter().collect();
    ranked.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    for (&(live, commit, pot), &count) in ranked.iter().take(20) {
        eprintln!(
            "  live {live}  commit {commit:>3} ({:>5.1} bb)  pot {pot:>4} ({:>6.1} bb)  ×{count}",
            commit as f64 / 2.0,
            pot as f64 / 2.0
        );
    }

    // Flop-tree sizes (drives pricing): oracle-shape abstraction
    // (1.0x pot bet, no raises). Build every distinct cell, report
    // per-live aggregate node counts + the largest cells.
    let flop_bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let mut sizes: Vec<((u8, i32, i32), usize)> = Vec::new();
    for &(live, commit, pot) in cells.keys() {
        let fcfg = spec.flop_seam_config(live, commit, pot, flop_bets.clone());
        let ft = build_tree(&fcfg).expect("flop seam tree");
        sizes.push(((live, commit, pot), ft.nodes.len()));
    }
    eprintln!("flop-start tree sizes (oracle-shape abstraction), per live count:");
    for (&live, _) in &by_live {
        let s: Vec<usize> =
            sizes.iter().filter(|((l, _, _), _)| *l == live).map(|&(_, n)| n).collect();
        let (min, max) = (s.iter().min().unwrap(), s.iter().max().unwrap());
        let sum: usize = s.iter().sum();
        eprintln!(
            "  live {live}: {} cells, nodes min {min} / max {max} / mean {:.0}",
            s.len(),
            sum as f64 / s.len() as f64
        );
    }
    let mut top: Vec<_> = sizes.iter().collect();
    top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    eprintln!("largest flop trees:");
    for ((live, commit, pot), n) in top.iter().take(8) {
        eprintln!("  live {live} commit {commit} pot {pot}: {n} nodes");
    }
}

/// Census at candidate raise-depth caps (the knob's payoff): the
/// uncapped v1 preflop tree measured 4.33M nodes / 1.99M infosets /
/// 9,923 seam cells — buffers ~21.5GB each, fold terminals alone
/// ~132s/iter. The cap prunes the deep raise-war subtrees.
#[test]
#[ignore = "census instrument; run with --ignored --nocapture"]
fn v1_seam_census_depth_capped() {
    use solver_core::tree::action::BetCap;
    let spec = production_game_v1();
    for cap in [3u8, 4] {
        let mut cfg = spec.preflop_tree_config(preflop_bets());
        cfg.max_bets_per_street = BetCap::all(cap);
        let t0 = std::time::Instant::now();
        let tree = build_tree_preflop_only(&cfg).expect("v1 preflop tree");
        let nn = tree.nodes.len();
        let np = spec.num_players as usize;
        let (mut n_player, mut n_chance, mut n_term) = (0usize, 0usize, 0usize);
        let mut cells = std::collections::BTreeSet::new();
        for idx in 0..nn {
            let n = &tree.nodes[idx];
            match n.node_type {
                2 => n_player += 1,
                0 => n_term += 1,
                1 => {
                    n_chance += 1;
                    let mask = tree.get_folded_mask(idx);
                    let contribs: Vec<i32> =
                        (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
                    let live: Vec<usize> =
                        (0..np).filter(|&p| mask & (1 << p) == 0).collect();
                    let pot: i32 = contribs.iter().sum();
                    cells.insert((live.len() as u8, contribs[live[0]], pot));
                }
                _ => {}
            }
        }
        eprintln!(
            "cap {cap}: {} nodes ({} player / {} flop-entry chance / {} fold terminals), \
             {} seam cells, built in {:.1?}",
            nn, n_player, n_chance, n_term, cells.len(), t0.elapsed()
        );
    }
}

/// CELL-WIDTH ANALYSIS (2026-06-12): the depth cap shrank the preflop
/// tree 2.3× but the seam-cell count barely moved (9,923 → 9,521) —
/// production cost is driven by cell WIDTH. This instrument asks what
/// width actually is. A flop-start cell is the game (live, behind,
/// pot) with behind = stack − commit; with pot-relative sizing the
/// tree and (scaled) equilibrium depend on the cell essentially
/// through (live, SPR = behind/pot) — thousands of distinct integer
/// (commit, pot) pairs collapse onto a far smaller strategic surface.
/// Quantified here: log2-SPR binning at several widths, weighted by
/// flop-entry multiplicity, on the CAP-3 production preflop tree.
/// (Reach-weighting needs the preflop solve; multiplicity is the
/// structural proxy. The bucketing residual is a harness A/B.)
#[test]
#[ignore = "analysis instrument; run with --ignored --nocapture"]
fn v1_cell_width_analysis() {
    use solver_core::tree::action::BetCap;
    use std::collections::BTreeMap;
    let spec = production_game_v1();
    let mut cfg = spec.preflop_tree_config(preflop_bets());
    cfg.max_bets_per_street = BetCap::all(3);
    let tree = build_tree_preflop_only(&cfg).expect("v1 preflop tree (cap 3)");
    let np = spec.num_players as usize;

    // (live, commit, pot) → multiplicity.
    let mut cells: BTreeMap<(u8, i32, i32), usize> = BTreeMap::new();
    for idx in 0..tree.nodes.len() {
        let n = &tree.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.board_state != BoardState::Flop as u8 {
            continue;
        }
        let mask = tree.get_folded_mask(idx);
        let contribs: Vec<i32> =
            (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let live: Vec<usize> = (0..np).filter(|&p| mask & (1 << p) == 0).collect();
        let pot: i32 = contribs.iter().sum();
        *cells.entry((live.len() as u8, contribs[live[0]], pot)).or_default() += 1;
    }
    let total_entries: usize = cells.values().sum();
    eprintln!("cap-3 cells: {} distinct, {} flop entries", cells.len(), total_entries);

    // Decompose width: how many distinct commits / pots / SPRs per live.
    for live in 2..=6u8 {
        let sub: Vec<(&(u8, i32, i32), &usize)> =
            cells.iter().filter(|((l, _, _), _)| *l == live).collect();
        if sub.is_empty() {
            continue;
        }
        let commits: std::collections::BTreeSet<i32> =
            sub.iter().map(|((_, c, _), _)| *c).collect();
        let pots: std::collections::BTreeSet<i32> =
            sub.iter().map(|((_, _, p), _)| *p).collect();
        let allin = sub
            .iter()
            .filter(|((_, c, _), _)| *c >= spec.stack)
            .map(|(_, &m)| m)
            .sum::<usize>();
        eprintln!(
            "live {live}: {} cells | {} distinct commits, {} distinct pots | \
             all-in entries {} ({:.1}% of live-{live} entries)",
            sub.len(),
            commits.len(),
            pots.len(),
            allin,
            100.0 * allin as f64 / sub.iter().map(|(_, &m)| m).sum::<usize>() as f64
        );
    }

    // SPR bucketing: bins of log2(SPR) at width w; all-in cells (SPR 0)
    // get their own bucket per live (they need no solve at all).
    for w in [0.5f64, 0.25, 0.1] {
        let mut buckets: BTreeMap<(u8, i64), usize> = BTreeMap::new();
        let mut allin_buckets = 0usize;
        for (&(live, commit, pot), &mult) in &cells {
            let behind = spec.stack - commit;
            let key = if behind <= 0 {
                allin_buckets = 1;
                (live, i64::MIN)
            } else {
                let spr = behind as f64 / pot as f64;
                (live, (spr.log2() / w).floor() as i64)
            };
            *buckets.entry(key).or_default() += mult;
        }
        let solvable: Vec<(&(u8, i64), &usize)> =
            buckets.iter().filter(|((_, b), _)| *b != i64::MIN).collect();
        let n_solve = solvable.len();
        // Coverage: entries in the top-k solvable buckets.
        let mut by_mult: Vec<usize> = solvable.iter().map(|(_, &m)| m).collect();
        by_mult.sort_unstable_by(|a, b| b.cmp(a));
        let covered: usize = by_mult.iter().take(50).sum();
        let solvable_total: usize = by_mult.iter().sum();
        eprintln!(
            "log2-SPR width {w}: {} SOLVE buckets (+ all-in free) | top-50 buckets \
             cover {:.1}% of non-allin entries",
            n_solve,
            100.0 * covered as f64 / solvable_total.max(1) as f64
        );
        let _ = allin_buckets;
    }

    // Tree-shape collapse check: within a (live, log2-SPR/0.25) bucket,
    // how many distinct ORACLE-SHAPE tree node counts appear? If ~1,
    // the bucket members are structurally the same game (the residual
    // is payoff scaling + integer rounding — harness-measurable).
    let flop_bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let mut shape: BTreeMap<(u8, i64), std::collections::BTreeSet<usize>> = BTreeMap::new();
    for &(live, commit, pot) in cells.keys() {
        let behind = spec.stack - commit;
        if behind <= 0 {
            continue;
        }
        let spr = behind as f64 / pot as f64;
        let b = (spr.log2() / 0.25).floor() as i64;
        let fcfg = spec.flop_seam_config(live, commit, pot, flop_bets.clone());
        let ft = build_tree(&fcfg).expect("flop tree");
        shape.entry((live, b)).or_default().insert(ft.nodes.len());
    }
    let multi = shape.values().filter(|s| s.len() > 1).count();
    let maxv = shape.values().map(|s| s.len()).max().unwrap_or(0);
    eprintln!(
        "tree-shape collapse at width 0.25: {} buckets, {} with >1 distinct node count \
         (max variants in a bucket: {maxv})",
        shape.len(),
        multi
    );
}

/// PRODUCTION FILL PRICING (2026-06-14): price the reach-weighted per-family
/// blueprint over the ACTUAL SPR-cell set the v1 preflop reaches. For each
/// (live, log2-SPR/w) SOLVE bucket (all-in cells are free rollout), the
/// production fill solves ONE representative cell (the tree-shape collapse
/// above justifies one per bucket) across 1755 flops. Per-flop GPU solve time
/// is modelled t(nodes) = a + b·nodes, calibrated per family from MEASURED
/// 3-flop probes on this exact runner (live-3/4 @ B15, live-5 @ B8, the
/// reach-weighted fidelity allocation; live-6 = rollout/no-solve, live-2 = HU
/// exact path priced elsewhere):
///   live-3: (102n,0.2s)(597n,0.3s)   live-4: (279n,0.5s)(2635n,1.0s)
///   live-5: (708n,0.7s)(11153n,2.5s)
/// Reports hours per family and the matrix total at SPR widths 0.5 and 0.25.
#[test]
#[ignore = "pricing instrument (builds rep trees, ~seconds); run with --ignored --nocapture"]
fn v1_production_fill_pricing() {
    use solver_core::tree::action::BetCap;
    use std::collections::BTreeMap;
    let spec = production_game_v1();
    let mut cfg = spec.preflop_tree_config(preflop_bets());
    cfg.max_bets_per_street = BetCap::all(3);
    let tree = build_tree_preflop_only(&cfg).expect("v1 preflop tree (cap 3)");
    let np = spec.num_players as usize;
    let flop_bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    const FLOPS: f64 = 1755.0;

    // Per-family per-flop GPU time model t(B, n) = a(B) + slope(B)·n (sec),
    // calibrated from runner probes on THIS GPU path (deep/shallow cells at
    // B8/12/16). live-3 is overhead-bound (flat ~0.10s, B-independent);
    // live-4 scales ~B^1.3 (a∝B^1.58, slope∝B^0.72 from the B8/12/16 fit);
    // live-5 is held at B8 (B15 would be gigantic — its tree is 10× live-4's).
    // B>16 is a PROJECTION: the GPU bucketed kernel caps at MAX_BUCKETS_GPU=16,
    // so B20/24/32 require raising that cap (shader recompile + bit-exact
    // re-gate; threadgroup memory bounds it near B32).
    let model = |live: u8, b: usize, n: usize| -> Option<f64> {
        match live {
            3 => Some(0.103),                              // flat, B-independent
            4 => {
                let bf = b as f64;
                let a = 0.525 * (bf / 16.0).powf(1.58);
                let slope = 1.42e-4 * (bf / 16.0).powf(0.72);
                Some(a + slope * n as f64)
            }
            5 => Some(0.578 + 1.72e-4 * n as f64),         // B8 fixed
            _ => None, // live-2 HU exact, live-6 rollout
        }
    };
    // Bucket levels to price live-3/4 at (16 = GPU max today; >16 = cap-raised).
    let blevels = [16usize, 20, 24, 32];

    // (live, commit, pot) -> multiplicity (structural reach proxy).
    let mut cells: BTreeMap<(u8, i32, i32), usize> = BTreeMap::new();
    for idx in 0..tree.nodes.len() {
        let n = &tree.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.board_state != BoardState::Flop as u8 {
            continue;
        }
        let mask = tree.get_folded_mask(idx);
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let live: Vec<usize> = (0..np).filter(|&p| mask & (1 << p) == 0).collect();
        let pot: i32 = contribs.iter().sum();
        *cells.entry((live.len() as u8, contribs[live[0]], pot)).or_default() += 1;
    }

    // Per (width, live): SPR-bucket representatives (highest-multiplicity), tree built once.
    let mut reps: BTreeMap<(u64, u8), Vec<(u8, i32, i32)>> = BTreeMap::new();
    let mut allin: BTreeMap<(u64, u8), usize> = BTreeMap::new();
    let widths = [0.5f64, 0.25];
    for (wi, &w) in widths.iter().enumerate() {
        let mut rep: BTreeMap<(u8, i64), (usize, (u8, i32, i32))> = BTreeMap::new();
        for (&(live, commit, pot), &mult) in &cells {
            let behind = spec.stack - commit;
            if behind <= 0 {
                *allin.entry((wi as u64, live)).or_default() += 1;
                continue;
            }
            let bkey = (live, ((behind as f64 / pot as f64).log2() / w).floor() as i64);
            let e = rep.entry(bkey).or_insert((0, (live, commit, pot)));
            if mult > e.0 { *e = (mult, (live, commit, pot)); }
        }
        for (&(live, _), &(_, cell)) in &rep {
            reps.entry((wi as u64, live)).or_default().push(cell);
        }
    }
    // Sum hours for one family at one bucket level over its SPR-bucket reps.
    let fam_hours = |wi: usize, live: u8, b: usize| -> f64 {
        reps.get(&(wi as u64, live)).map(|cells| {
            cells.iter().map(|&(l, c, p)| {
                let ft = build_tree(&spec.flop_seam_config(l, c, p, flop_bets.clone())).unwrap();
                model(live, b, ft.nodes.len()).unwrap() * FLOPS / 3600.0
            }).sum()
        }).unwrap_or(0.0)
    };

    eprintln!("\n══════ PRODUCTION FILL: live-3/4 finer buckets × SPR width (live-5 @ B8 fixed) ══════");
    for (wi, &w) in widths.iter().enumerate() {
        let n3 = reps.get(&(wi as u64, 3)).map(|v| v.len()).unwrap_or(0);
        let n4 = reps.get(&(wi as u64, 4)).map(|v| v.len()).unwrap_or(0);
        let n5 = reps.get(&(wi as u64, 5)).map(|v| v.len()).unwrap_or(0);
        let h5 = fam_hours(wi, 5, 8); // live-5 fixed at B8
        eprintln!(
            "\n── SPR width {w}  (live-3: {n3} cells, live-4: {n4} cells, live-5: {n5} cells) ──"
        );
        eprintln!("  live-5 @ B8 (fixed): {h5:.2} h");
        eprintln!("  B(3&4) | live-3 h | live-4 h | 3+4 h | +live-5 = TOTAL h (days) | GPU?");
        for &b in &blevels {
            let h3 = fam_hours(wi, 3, b);
            let h4 = fam_hours(wi, 4, b);
            let total = h3 + h4 + h5;
            let gpu = if b <= 16 { "GPU now" } else { "cap-raise" };
            eprintln!(
                "  B{b:<5}| {h3:7.2}  | {h4:7.2}  | {:5.2} | {total:6.2} h ({:.2} d)        | {gpu}",
                h3 + h4, total / 24.0
            );
        }
    }
    let _ = &allin;

    // ── PRODUCTION CELL LIST (width 0.25) for the fill runner: one line per
    // (live, SPR-bucket) representative, machine-parseable. live-3/4 @ B15,
    // live-5 @ B8 (the reach-weighted allocation). live-2 = HU/exact and
    // live-6 = rollout are deployment paths, NOT solved here.
    eprintln!("\n══ PRODUCTION CELL LIST (SPR width 0.25, reach-weighted B) ══");
    let mut n_cells = 0usize;
    for live in 3..=5u8 {
        if let Some(cells) = reps.get(&(1u64, live)) {
            let b = if live == 5 { 8 } else { 15 };
            let mut sorted = cells.clone();
            sorted.sort_by_key(|&(_, c, p)| (c, p));
            for &(l, c, p) in &sorted {
                eprintln!("CELL live={l} commit={c} pot={p} b={b}");
                n_cells += 1;
            }
        }
    }
    eprintln!("CELLS_TOTAL {n_cells}");
}
