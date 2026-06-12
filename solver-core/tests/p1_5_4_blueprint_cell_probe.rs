//! Blueprint cell decision probe — putting BOTH axes of the
//! (flop set × runout policy) choice in the same currency (mean |Δσ|
//! of the banked flop strategy) before the run that anchors everything
//! downstream.
//!
//! The asymmetry being fixed: the runout axis is measured (1×1 costs
//! 0.105 mean flop-σ movement vs the 4×4 reference; 2×2 costs 0.034),
//! but the flop-set axis was priced only in hours. Its true cost is
//! COVERAGE ERROR: a dealt flop outside the solved set borrows its
//! nearest in-sample neighbor's solution. Orbit-weighted, a 100-flop
//! plan leaves ~94% of deals borrowing (300-plan ~83%) — the
//! substitution cost multiplies into nearly everything.
//!
//! === Probe 1: neighbor-substitution cost ===
//! Texture-feature-stratified plans of 100 and 300 canonicals (sorted
//! stride over (paired, suits, ranks) — deterministic). Held-out
//! flops (outside both plans) are each solved and compared against
//! their nearest in-plan neighbor's solve: per-(node, action,
//! 169-class) flop σ_avg, mean |Δσ|. All solves share one tree (tree
//! structure is board-independent), B=8 quantile maps,
//! Design1Collapsed, 2×2 deterministic runout draw, 15 iters.
//!
//! CONTROL (the noise floor this measurement rides on): same flop
//! solved at two DIFFERENT 2×2 runout draws — substitution cost is
//! read as the EXCESS over this floor, since cross-board comparisons
//! inherit runout-draw noise by construction.
//!
//! === Probe 2: preflop integration error (the case for option A) ===
//! The preflop blueprint integrates per-flop CFVs over the whole flop
//! set; sampling replaces the 1755-term sum with a re-weighted subset
//! sum. Functional: iter-1 root class-CFV (uniform-strategy values —
//! real, texture-dependent, cheap at 1 iter × 1×1). Nested plans
//! (plan-100 = every 4th of plan-400, both stratified): per-class
//! orbit-weighted means compared, max/mean |Δ| as % of pot.
//!
//! Named limits: probe-scale tree (1-bet M2 shape — substitution is a
//! texture question, measurable on any fixed tree); 15-iter solves
//! (same-count comparisons); the neighbor metric is the candidate
//! deployment metric, documented below; production confirmation of
//! the chosen cell belongs to head-to-head, like every quality
//! verdict at production scale.
//!
//! ═══ MEASURED 2026-06-11 ═══
//!   coverage mass: plan-100 borrows on 94.3% of deals, plan-300 82.9%
//!   control noise floor (same flop, different 2×2 draw):
//!     mean |Δσ| 0.0706 / 0.0608 (avg 0.0657)
//!   substitution, plan-100: mean |Δσ| 0.1646 (excess +0.0989)
//!   substitution, plan-300: mean |Δσ| 0.1578 (excess +0.0921)
//!   preflop integration (normalized class-CFV profiles):
//!     plan-100 vs plan-400: mean 3.06%, max 9.75% of mean magnitude
//!
//!   READING — the probe flipped the tentative recommendation:
//!   substitution cost is LARGE (≈ the entire 1×1 runout penalty of
//!   0.105) and nearly FLAT in plan size (tripling flops bought
//!   0.007). Flop strategies are sharply board-specific under this
//!   neighbor metric; coverage cannot be bought cheaply at these plan
//!   sizes. Cell arithmetic (mean flop-σ error, deal-mass weighted):
//!     A: 1755×1×1 (10.5h) ≈ 0.105 (runout only; coverage exact,
//!        preflop integration exact)
//!     B: 300×2×2 (7.2h)  ≈ 0.034 + 0.83×0.092 ≈ 0.110 (+3% mean
//!        integration error)
//!     B': 100×4×4 (9.6h) ≈ 0.94×0.099 ≈ 0.093 (+integration)
//!     C: 1755×2×2 via GPU port (est ~8h) ≈ 0.034 — the 3× unlock
//!   A and B are within noise of each other; A wins the tie-break
//!   (exact coverage, exact integration on the street the blueprint
//!   plays directly, no neighbor-metric risk). C is the challenger
//!   cell once the GPU port exists. Caveats: substitution excess
//!   measured at the flop street on the probe tree under the feature
//!   neighbor metric — a better metric could lower it (research
//!   lever, not assumed); both axes' floors are 2×2-draw noise.

use solver_core::abstraction::preflop_class::PreflopClass;
use solver_core::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const NP: u8 = 6;
const NB: usize = 8;
const ITERS: u32 = 15;
const N_HELD_OUT: usize = 8;
const N_CONTROL: usize = 2;
const THREADS: usize = 8;
const N_CLASSES: usize = 169;

// ── Texture features + plans ──

/// (paired, n_suits, r_hi, r_mid, r_lo) — sort key for stratification
/// and basis of the neighbor metric.
fn features(flop: [Card; 3]) -> (u8, u8, u8, u8, u8) {
    let mut ranks: Vec<u8> = flop.iter().map(|&c| c >> 2).collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a));
    let paired = if ranks[0] == ranks[2] {
        2
    } else if ranks[0] == ranks[1] || ranks[1] == ranks[2] {
        1
    } else {
        0
    };
    let mut suits: Vec<u8> = flop.iter().map(|&c| c & 3).collect();
    suits.sort_unstable();
    suits.dedup();
    (paired, suits.len() as u8, ranks[0], ranks[1], ranks[2])
}

/// Candidate deployment neighbor metric: weighted L1 over features
/// (structure dominates ranks).
fn feature_dist(a: (u8, u8, u8, u8, u8), b: (u8, u8, u8, u8, u8)) -> i32 {
    6 * (a.0 as i32 - b.0 as i32).abs()
        + 4 * (a.1 as i32 - b.1 as i32).abs()
        + (a.2 as i32 - b.2 as i32).abs()
        + (a.3 as i32 - b.3 as i32).abs()
        + (a.4 as i32 - b.4 as i32).abs()
}

/// Feature-sorted stride plan of size n over the canonical list.
fn stride_plan(sorted_idx: &[usize], n: usize) -> Vec<usize> {
    let step = sorted_idx.len() as f64 / n as f64;
    let mut out = Vec::with_capacity(n);
    let mut x = 0.0f64;
    while out.len() < n && (x as usize) < sorted_idx.len() {
        out.push(sorted_idx[x as usize]);
        x += step;
    }
    out
}

// ── Probe-scale tree (M2 shape — fixed across all solves) ──

fn build_probe_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&config).unwrap()
}

// ── Per-flop full-nh table at a deterministic runout draw ──

/// draw 0: turns at deck positions {12, 36}, rivers at {10, 30};
/// draw 1 (control): turns {6, 24}, rivers {16, 40}. 1×1 (probe 2):
/// turn {12}, river {10}.
///
/// HAND-ROLLED (the build_m2_table pattern, num_combinations = 1.0):
/// `compute_flop_start_subset_with_decks` computes num_combinations by
/// recursive JOINT ENUMERATION — O(nh^6), frozen forever at nh=1176
/// (first probe launch hung exactly there; the B4 cost test dodged the
/// same trap the same way). nc only scales cfv readout uniformly: σ is
/// row-normalized (probe 1 unaffected) and probe 2 compares the same
/// nc=1.0 functional across flops (named).
fn build_flop_table(flop: [Card; 3], n_turn: usize, n_river: usize, draw: usize) -> FlopChanceTable {
    use solver_core::hand::eval::Hand;
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let num_opp = NP as usize - 1;

    let mut chosen: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & ((1u64 << c1) | (1u64 << c2)) == 0 {
            chosen.push(idx as u16);
        }
    }
    let nh = chosen.len();
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
    }
    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh {
        for j in 0..nh {
            if i == j {
                conflict[i * nh + j] = 1;
                continue;
            }
            let (a1, a2) = (hand_cards[i * 2], hand_cards[i * 2 + 1]);
            let (b1, b2) = (hand_cards[j * 2], hand_cards[j * 2 + 1]);
            if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 {
                conflict[i * nh + j] = 1;
            }
        }
    }
    let mut hr = vec![0u16; nh];
    for i in 0..nh {
        let mut h = Hand::new()
            .add_card(hand_cards[i * 2] as usize)
            .add_card(hand_cards[i * 2 + 1] as usize);
        for &bc in &flop {
            h = h.add_card(bc as usize);
        }
        hr[i] = h.evaluate_internal() as u16;
    }

    let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let turn_pos: &[usize] = match (n_turn, draw) {
        (1, _) => &[12],
        (2, 0) => &[12, 36],
        (2, 1) => &[6, 24],
        _ => unreachable!(),
    };
    let river_pos: &[usize] = match (n_river, draw) {
        (1, _) => &[10],
        (2, 0) => &[10, 30],
        (2, 1) => &[16, 40],
        _ => unreachable!(),
    };
    let turn_cards: Vec<u8> = turn_pos.iter().map(|&p| deck[p]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let rdeck: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        river_decks[tc as usize] = river_pos.iter().map(|&p| rdeck[p]).collect();
    }

    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &t in &turn_cards {
        let tm = board_mask | (1u64 << t);
        for i in 0..nh {
            let (c1, c2) = (hand_cards[i * 2], hand_cards[i * 2 + 1]);
            if tm & ((1u64 << c1) | (1u64 << c2)) != 0 {
                continue;
            }
            let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
            for &bc in &flop {
                h = h.add_card(bc as usize);
            }
            h = h.add_card(t as usize);
            turn_ranks[t as usize * nh + i] = h.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> =
            (0..nh).map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            let off = t as usize * num_opp * nh + oi * nh;
            for h in 0..nh {
                turn_sorted_str[off + h] = items[h].0;
                turn_sorted_idx[off + h] = items[h].1;
            }
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &t in &turn_cards {
        let tm = board_mask | (1u64 << t);
        for &r in &river_decks[t as usize] {
            let fm = tm | (1u64 << r);
            for i in 0..nh {
                let (c1, c2) = (hand_cards[i * 2], hand_cards[i * 2 + 1]);
                if fm & ((1u64 << c1) | (1u64 << c2)) != 0 {
                    continue;
                }
                let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
                for &bc in &flop {
                    h = h.add_card(bc as usize);
                }
                h = h.add_card(t as usize).add_card(r as usize);
                river_ranks[t as usize * 52 * nh + r as usize * nh + i] =
                    h.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off =
                    t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh {
                    river_sorted_str[off + h] = items[h].0;
                    river_sorted_idx[off + h] = items[h].1;
                }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; NP as usize];
    FlopChanceTable {
        hand_ranks_base: hr,
        valid_hand_indices: chosen,
        num_valid: nh,
        conflict,
        hand_cards,
        remaining_deck: turn_cards,
        turn_ranks,
        turn_sorted_str,
        turn_sorted_idx,
        river_ranks,
        river_sorted_str,
        river_sorted_idx,
        initial_weights: iw,
        num_players: NP,
        num_combinations: 1.0,
        river_decks,
    }
}

fn quantile_maps(
    table: &FlopChanceTable,
    nb: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8]| -> Vec<u16> {
        let alive: Vec<usize> = pl_idx[..nh]
            .iter()
            .map(|&i| i as usize)
            .filter(|&h| !conflicts(h, dead))
            .collect();
        let n = alive.len();
        assert!(n >= nb);
        let mut map = vec![NO_BUCKET; nh];
        for (pos, &h) in alive.iter().enumerate() {
            map[h] = ((pos * nb) / n) as u16;
        }
        map
    };
    let (_, _, _, base_pi, _) = table.sorted_opp_arrays_base();
    let flop_map = map_for(&base_pi, &[]);
    let mut turn_maps = Vec::new();
    let mut river_maps = Vec::new();
    for &tc_card in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
        turn_maps.push(map_for(pi, &[tc_card]));
        let mut rms = Vec::new();
        for &rc_card in &table.river_decks[tc_card as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
            rms.push(map_for(pi, &[tc_card, rc_card]));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
}

/// Solve one flop; return per-(flop-zone node, action, class) σ_avg
/// rows, flat, in deterministic node order — comparable ACROSS flops
/// because the tree is shared and classes are board-independent.
fn solve_flop_class_sigma(tree: &FlatTree, flop: [Card; 3], draw: usize) -> Vec<f32> {
    let table = build_flop_table(flop, 2, 2, draw);
    let nh = table.num_valid;
    let (fm, tm, rm) = quantile_maps(&table, NB);
    let game = FlopStartGame::new(table);
    let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
    let mut bucketed = BucketedFlopCfr::new(tree, game.table(), &bk);
    bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
    bucketed.run(tree, &game, &bk, ITERS);

    let cum = bucketed.cum_strategy_flop();
    let table = game.table();
    // hand → class
    let h_class: Vec<usize> = (0..nh)
        .map(|h| {
            PreflopClass::from_combo(table.hand_cards[h * 2], table.hand_cards[h * 2 + 1]).index()
        })
        .collect();

    let mut out = Vec::new();
    for &nid in &tree.decision_node_ids {
        let idx = nid as usize;
        let Some(local) = bucketed.flop_local_offset_at(idx) else { continue };
        let na = tree.nodes[idx].num_children as usize;
        let off = local * MAX_NA_POSTFLOP * NB;
        // per-class accumulators
        let mut cls_sum = vec![0.0f64; na * N_CLASSES];
        let mut cls_n = vec![0.0f64; N_CLASSES];
        for h in 0..nh {
            let b = bk.flop_map[h];
            if b == NO_BUCKET {
                continue;
            }
            let mut row_sum = 0.0f32;
            for a in 0..na {
                row_sum += cum[off + a * NB + b as usize];
            }
            let c = h_class[h];
            cls_n[c] += 1.0;
            for a in 0..na {
                let s = if row_sum > 0.0 {
                    cum[off + a * NB + b as usize] / row_sum
                } else {
                    1.0 / na as f32
                };
                cls_sum[a * N_CLASSES + c] += s as f64;
            }
        }
        for a in 0..na {
            for c in 0..N_CLASSES {
                out.push(if cls_n[c] > 0.0 {
                    (cls_sum[a * N_CLASSES + c] / cls_n[c]) as f32
                } else {
                    -1.0 // class dead on this board (blocked); skip in compare
                });
            }
        }
    }
    out
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (x, y) in a.iter().zip(b) {
        if *x < 0.0 || *y < 0.0 {
            continue; // class dead on either board
        }
        sum += (*x as f64 - *y as f64).abs();
        n += 1;
    }
    sum / n as f64
}

/// Parallel map over jobs with a shared work index.
fn par_solve<J: Sync, R: Send>(jobs: &[J], f: impl Fn(&J) -> R + Sync) -> Vec<R>
where
    R: Default,
{
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<R>> = (0..jobs.len()).map(|_| Mutex::new(R::default())).collect();
    std::thread::scope(|s| {
        for _ in 0..THREADS.min(jobs.len()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                *results[i].lock().unwrap() = f(&jobs[i]);
            });
        }
    });
    results.into_iter().map(|m| m.into_inner().unwrap()).collect()
}

#[test]
#[ignore = "blueprint cell probe (~40-60 min at 8 threads); run with --ignored --nocapture"]
fn cell_probe_substitution_and_integration() {
    eprintln!("\n════ blueprint cell probe ════");
    let t_all = Instant::now();

    // Canonical flop list + orbit weights.
    let ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![1.0 / N_CLASSES as f32; N_CLASSES]).collect();
    let ptable = PreflopChanceTable::new(NP, ranges);
    let canon = &ptable.canonical_flops;
    let orbit = &ptable.orbit_sizes;
    let n_canon = canon.len();

    // Feature-sorted order; nested stratified plans.
    let mut sorted_idx: Vec<usize> = (0..n_canon).collect();
    sorted_idx.sort_by_key(|&i| features(canon[i]));
    let plan400 = stride_plan(&sorted_idx, 400);
    let plan100: Vec<usize> = plan400.iter().step_by(4).copied().collect();
    let plan300 = stride_plan(&sorted_idx, 300);

    // Coverage mass (orbit-weighted fraction of deals borrowing).
    let total_w: f64 = orbit.iter().map(|&w| w as f64).sum();
    for (name, plan) in [("plan-100", &plan100), ("plan-300", &plan300)] {
        let in_w: f64 = plan.iter().map(|&i| orbit[i] as f64).sum();
        eprintln!("{name}: {} flops, in-sample deal mass {:.1}% (borrowing {:.1}%)",
            plan.len(), 100.0 * in_w / total_w, 100.0 * (1.0 - in_w / total_w));
    }

    // Held-out flops: stride over the complement of plan100 ∪ plan300.
    let in_any: std::collections::HashSet<usize> =
        plan100.iter().chain(plan300.iter()).copied().collect();
    let complement: Vec<usize> =
        sorted_idx.iter().copied().filter(|i| !in_any.contains(i)).collect();
    let held_out = stride_plan(&complement, N_HELD_OUT);

    // Nearest in-plan neighbors.
    let nearest = |f_idx: usize, plan: &[usize]| -> usize {
        let ff = features(canon[f_idx]);
        *plan
            .iter()
            .min_by_key(|&&j| feature_dist(ff, features(canon[j])))
            .unwrap()
    };
    let n100: Vec<usize> = held_out.iter().map(|&f| nearest(f, &plan100)).collect();
    let n300: Vec<usize> = held_out.iter().map(|&f| nearest(f, &plan300)).collect();

    let tree = build_probe_tree();

    // Job list: (canonical index, draw). Dedup.
    let mut jobs: Vec<(usize, usize)> = Vec::new();
    for &f in held_out.iter().chain(n100.iter()).chain(n300.iter()) {
        if !jobs.contains(&(f, 0)) {
            jobs.push((f, 0));
        }
    }
    // Controls: first N_CONTROL plan-100 members at both draws.
    for &f in plan100.iter().take(N_CONTROL) {
        for d in [0, 1] {
            if !jobs.contains(&(f, d)) {
                jobs.push((f, d));
            }
        }
    }

    // Probe one solve first (measure-before-launch).
    let t0 = Instant::now();
    let first = solve_flop_class_sigma(&tree, canon[jobs[0].0], jobs[0].1);
    let one_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "one probe solve (2×2, B={NB}, {ITERS} iters): {:.0}s → {} solves ≈ {:.0} min at {THREADS} threads",
        one_s,
        jobs.len(),
        (jobs.len() - 1) as f64 * one_s / THREADS as f64 / 60.0
    );

    let results: Vec<Vec<f32>> = {
        let rest = &jobs[1..];
        let mut r = par_solve(rest, |&(fi, d)| solve_flop_class_sigma(&tree, canon[fi], d));
        r.insert(0, first);
        r
    };
    let sigma_of = |fi: usize, d: usize| -> &Vec<f32> {
        &results[jobs.iter().position(|&j| j == (fi, d)).unwrap()]
    };

    // Control floor.
    let mut floor_sum = 0.0f64;
    for &f in plan100.iter().take(N_CONTROL) {
        let d = mean_abs_diff(sigma_of(f, 0), sigma_of(f, 1));
        eprintln!("control (same flop, different 2×2 draw): mean |Δσ| {:.4}", d);
        floor_sum += d;
    }
    let floor = floor_sum / N_CONTROL as f64;

    // Substitution.
    for (name, nbrs) in [("plan-100", &n100), ("plan-300", &n300)] {
        let mut sum = 0.0f64;
        for (k, &f) in held_out.iter().enumerate() {
            sum += mean_abs_diff(sigma_of(f, 0), sigma_of(nbrs[k], 0));
        }
        let m = sum / held_out.len() as f64;
        eprintln!(
            "substitution {name}: mean |Δσ| {:.4} (excess over {:.4} floor: {:+.4})",
            m,
            floor,
            m - floor
        );
    }

    eprintln!("\ntotal probe-1 wall-clock: {:.0} min", t_all.elapsed().as_secs_f64() / 60.0);
}

/// Probe 2 (separate test — probe 1's results stand alone): preflop
/// integration error of stratified sampling, on NORMALIZED per-flop
/// class-CFV profiles. With nc = 1.0 (the hand-rolled table's joint-
/// count dodge) raw class CFVs carry a per-flop ~1e15 scale; the first
/// combined run reported a meaningless "% of pot". Fix: scale each
/// flop's class vector to unit mean |v| (dimensionless profile), then
/// compare orbit-weighted profile means, |Δ| in % of mean magnitude.
#[test]
#[ignore = "blueprint cell probe 2: integration (~15 min at 8 threads); run with --ignored --nocapture"]
fn cell_probe_integration() {
    eprintln!("\n════ blueprint cell probe 2: preflop integration ════");
    let ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![1.0 / N_CLASSES as f32; N_CLASSES]).collect();
    let ptable = PreflopChanceTable::new(NP, ranges);
    let canon = &ptable.canonical_flops;
    let orbit = &ptable.orbit_sizes;
    let mut sorted_idx: Vec<usize> = (0..canon.len()).collect();
    sorted_idx.sort_by_key(|&i| features(canon[i]));
    let plan400 = stride_plan(&sorted_idx, 400);
    let plan100: Vec<usize> = plan400.iter().step_by(4).copied().collect();

    let tree = build_probe_tree();
    let int_jobs: Vec<(usize, usize)> = plan400.iter().map(|&i| (i, 0)).collect();
    let class_cfvs: Vec<Vec<f32>> = par_solve(&int_jobs, |&(fi, _)| {
        let table = build_flop_table(canon[fi], 1, 1, 0);
        let nh = table.num_valid;
        let (fm, tm, rm) = quantile_maps(&table, NB);
        let game = FlopStartGame::new(table);
        let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
        let mut bucketed = BucketedFlopCfr::new(&tree, game.table(), &bk);
        bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
        let root = bucketed.run(&tree, &game, &bk, 1);
        let table = game.table();
        let mut sum = vec![0.0f64; N_CLASSES];
        let mut n = vec![0.0f64; N_CLASSES];
        for h in 0..nh {
            let c = PreflopClass::from_combo(table.hand_cards[h * 2], table.hand_cards[h * 2 + 1])
                .index();
            sum[c] += root[h] as f64;
            n[c] += 1.0;
        }
        let raw: Vec<f64> = (0..N_CLASSES)
            .map(|c| if n[c] > 0.0 { sum[c] / n[c] } else { f64::NAN })
            .collect();
        // Normalize to unit mean |v| (kills the per-flop nc scale).
        let finite: Vec<f64> = raw.iter().copied().filter(|v| v.is_finite()).collect();
        let scale = finite.iter().map(|v| v.abs()).sum::<f64>() / finite.len() as f64;
        raw.iter().map(|&v| (v / scale) as f32).collect()
    });

    let weighted_mean = |plan: &[usize]| -> Vec<f64> {
        let mut acc = vec![0.0f64; N_CLASSES];
        let mut wsum = vec![0.0f64; N_CLASSES];
        for &fi in plan {
            let pos = plan400.iter().position(|&j| j == fi).unwrap();
            let w = orbit[fi] as f64;
            for c in 0..N_CLASSES {
                let v = class_cfvs[pos][c];
                if v.is_finite() {
                    acc[c] += w * v as f64;
                    wsum[c] += w;
                }
            }
        }
        (0..N_CLASSES).map(|c| if wsum[c] > 0.0 { acc[c] / wsum[c] } else { 0.0 }).collect()
    };
    let v100 = weighted_mean(&plan100);
    let v400 = weighted_mean(&plan400);
    let mut max_d = 0.0f64;
    let mut sum_d = 0.0f64;
    for c in 0..N_CLASSES {
        let d = (v100[c] - v400[c]).abs() * 100.0; // % of mean |v|
        if d > max_d {
            max_d = d;
        }
        sum_d += d;
    }
    eprintln!(
        "integration plan-100 vs plan-400 (normalized profiles): per-class |Δ| \
         mean {:.2}% max {:.2}% of mean magnitude",
        sum_d / N_CLASSES as f64,
        max_d
    );
}

/// 6×-surprise characterization (standing rule: convenient-direction
/// surprises get their mechanism confirmed before banking). The B4
/// ladder row (B=8: 10.14s/iter) was an ITERATION-1 measurement with
/// dense uniform reaches; the runner measures 60-95s per 34-iter
/// solve (~1.8-2.8s/iter average). Hypothesis: reaches sparsify as
/// CFR converges (folded-out buckets → zero reach → terminal skips),
/// so iter-1 is expensive and later iters cheap. Counter-hypotheses
/// to rule out: tree-shape difference (oracle tree 4161 nodes, equal
/// contribs → arm-1-heavy census vs M2's asymmetric-contrib arm-2-
/// heavy census), early convergence cutoff (none exists), terminal
/// K lower than expected.
///
/// Probe: one canonical flop on the RUNNER's exact config, run(1)×34
/// with per-iteration wall-clock (parameter-equivalent to run(34):
/// DcfrParams reads the persisted iteration counter). PLUS the same
/// curve on the M2-shaped ladder tree, so the tree-shape factor and
/// the sparsification factor separate.
#[test]
#[ignore = "6x-surprise characterization (~10 min); run with --ignored --nocapture"]
fn characterize_runner_cost_curve() {
    use solver_core::solver::flop_start_game::FlopChanceTable as FCT;
    let ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![1.0 / N_CLASSES as f32; N_CLASSES]).collect();
    let ptable = PreflopChanceTable::new(NP, ranges);
    let flop = ptable.canonical_flops[0];

    // Runner-config tree (oracle shape: 1-bet, stacks 94, pot 12).
    let oracle_tree = {
        let cfg = TreeConfig {
            num_players: NP,
            initial_state: BoardState::Flop,
            starting_pot: 12,
            starting_stacks: vec![94; 6],
            initial_contributions: vec![0; 6],
            rake_rate: 0.0,
            rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
            add_allin_threshold: 1.0,
            force_allin_threshold: 1.0,
            merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,
        };
        build_tree(&cfg).unwrap()
    };
    // Ladder-config tree (M2 shape: asymmetric contribs).
    let m2_tree = build_probe_tree();

    for (name, tree) in [("oracle/runner", &oracle_tree), ("M2/ladder", &m2_tree)] {
        // Same runout draw scheme as the runner (fi=0).
        let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
        let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
        let tc = deck[19 % deck.len()];
        let rdeck: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        let rc = rdeck[10 % rdeck.len()];
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
        river_decks[tc as usize] = vec![rc];
        let table = FCT::build_full_nh_sampled(flop, 6, &[tc], &river_decks);
        let (fm, tm, rm) = quantile_maps(&table, NB);
        let game = FlopStartGame::new(table);
        let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
        let mut solver = BucketedFlopCfr::new(tree, game.table(), &bk);
        solver.set_terminal_design(TerminalDesign::Design1Collapsed);
        let mut times = Vec::new();
        for _ in 0..34 {
            let t0 = std::time::Instant::now();
            solver.run(tree, &game, &bk, 1);
            times.push(t0.elapsed().as_secs_f64());
        }
        let total: f64 = times.iter().sum();
        eprintln!(
            "{name} ({} nodes): total {total:.1}s | iters 1-5: {:.2} {:.2} {:.2} {:.2} {:.2} | \
             10: {:.2} | 20: {:.2} | 34: {:.2} | iter1/iter34 ratio {:.1}×",
            tree.num_nodes(),
            times[0], times[1], times[2], times[3], times[4],
            times[9], times[19], times[33],
            times[0] / times[33]
        );
    }
}
