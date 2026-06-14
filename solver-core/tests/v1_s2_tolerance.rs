//! S2 — SEARCH-TOLERANCE-TO-BLUEPRINT-ROUGHNESS, 2-D measurement
//! (2026-06-13). The depth-limited-search arc's headline deliverable:
//! how ROUGH can the multiway blueprint be and still have real-time
//! search recover acceptable play from it? (Pluribus: rough blueprint +
//! search carries quality.)
//!
//! INSTRUMENT (validated path): S0 proved QRE→Nash (HU+multiway); S1
//! proved depth-limited search on the real CFR mechanism anchors to the
//! fine blueprint and CORRECTS a rough one (after the convergence-axis
//! misdiagnosis was caught). The convergence map proved the real seam
//! games CONVERGE (live-3/4 by exact exploitability; live-5/6 by
//! strategy-delta), so a clean reference EXISTS to measure search output
//! against — S2 is WELL-POSED.
//!
//! MECHANISM: `FlopStartVectorCfr::run_flop_search` searches the FLOP
//! while Turn+River play a FROZEN blueprint continuation (the
//! `continuation_frozen` flag gates off their regret/cum update; CFV
//! still propagates under the blueprint strategy, so the flop sees
//! "continue with blueprint" leaf values, range-correctly). Roughness
//! knob = the bucket count the blueprint was solved at (lifted in via the
//! keystone lift). Iterations axis = search iters. Output = scored
//! exploitability (% pot) of searched-flop + blueprint-turn/river.
//!
//! SCOPE (the load-bearing caveat, 2026-06-13): this is exploitability-
//! GROUNDED for live-3/4 (exact scorer feasible at research nh). At
//! live-5/6 exploitability is the O(nh^np) wall, so S2's live-5/6 verdict
//! is a TRANSFER of the live-3/4 tolerance up the player-count axis —
//! stability is mapped (convergence map) but QUALITY is not directly
//! measurable there. That live-5/6 quality transfer is the THIRD named
//! harness residual (with SPR texture-transfer and slice-2 card-removal):
//! validated at the measurable scale, transferred to the unmeasurable
//! scale, harness-confirmed. The harness multiway games are the sole
//! instrument that ever confirms it directly. This file measures
//! live-3/4 (where the curve MEANS exploitability) and names live-5/6 as
//! transfer — it does NOT manufacture a live-5/6 quality number it can't
//! measure.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{
    lift_cum_to_exact, BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

// ─── helpers (duplicated per the per-test-crate convention) ──────────────

/// Clean dry board (Ks7d2c, no straight/flush on any sampled runout) —
/// the convergence map's "dry" texture, chosen to avoid the chop-board
/// confound that earlier inflated the bucketing cliff ~2×.
fn clean_table(np: u8, nh: usize) -> FlopChanceTable {
    let board: [Card; 3] = ["Ks", "7d", "2c"].map(|s| card_from_str(s).unwrap());
    let bmask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if bmask & ((1u64 << c1) | (1u64 << c2)) == 0 { all_valid.push(idx as u16); }
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..np as usize { for &hi in &chosen { ranges[p][hi as usize] = 1.0; } }
    let turn_cards: Vec<u8> = ["9h", "4s"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let river_strs: [&[&str]; 2] = [&["Qd", "3c"], &["Td", "Js"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] = river_strs[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(&board, &ranges, np, &chosen, &turn_cards, &river_decks)
}

fn quantile_maps(
    table: &FlopChanceTable, nb: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8]| -> Vec<u16> {
        let alive: Vec<usize> =
            pl_idx[..nh].iter().map(|&i| i as usize).filter(|&h| !conflicts(h, dead)).collect();
        let n = alive.len();
        assert!(n >= nb, "fewer alive hands ({n}) than buckets ({nb})");
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
    for &tc in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc);
        turn_maps.push(map_for(pi, &[tc]));
        let mut rms = Vec::new();
        for &rc in &table.river_decks[tc as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc, rc);
            rms.push(map_for(pi, &[tc, rc]));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
}

fn expl_pct(cpu: &FlopStartVectorCfr, tree: &FlatTree, game: &FlopStartGame, pot: i32) -> f32 {
    let np = tree.num_players as usize;
    let mut total = 0.0f32;
    for p in 0..np {
        let br = cpu.best_response_value_debug(tree, game, p as u8);
        let sv = cpu.strategy_value_debug(tree, game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total += (br[h] - sv[h]).max(0.0);
        }
    }
    total / pot as f32 * 100.0
}

fn seam_tree(live: u8, commit: i32, pot: i32) -> FlatTree {
    let spec = production_game_v1();
    build_tree(&spec.flop_seam_config(
        live, commit, pot,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
    )).expect("seam tree")
}

/// Exact converged solver (the reference / fine continuation source).
fn exact_solver(tree: &FlatTree, np: u8, nh: usize, iters: u32) -> FlopStartVectorCfr {
    let game = FlopStartGame::new(clean_table(np, nh));
    let mut s = FlopStartVectorCfr::new(tree, game.table());
    s.run(tree, &game, iters);
    s
}

/// A blueprint solver whose cum_strategy holds the blueprint at roughness
/// `nb` (None = exact/fine), lifted to the exact-nh game so its turn/river
/// cum can be frozen as the search continuation. Solved to `iters` (the
/// blueprint must itself be CONVERGED — else roughness is conflated with
/// undertraining, the convergence reflex).
fn blueprint_solver(tree: &FlatTree, np: u8, nh: usize, nb: Option<usize>, iters: u32) -> FlopStartVectorCfr {
    match nb {
        None => exact_solver(tree, np, nh, iters),
        Some(b) => {
            let table = clean_table(np, nh);
            let (fm, tm, rm) = quantile_maps(&table, b);
            let bk = FlopBucketing::from_maps(&table, b, b, b, fm, tm, rm);
            let game = FlopStartGame::new(table);
            let mut bucketed = BucketedFlopCfr::new(tree, game.table(), &bk);
            bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
            bucketed.run(tree, &game, &bk, iters);
            let gs = FlopStartGame::new(clean_table(np, nh));
            let mut scorer = FlopStartVectorCfr::new(tree, gs.table());
            lift_cum_to_exact(tree, &bucketed, &bk, &mut scorer);
            scorer
        }
    }
}

/// DEPTH-LIMITED SEARCH OUTPUT: freeze `bp`'s turn+river as the
/// continuation, search the flop for `search_iters`, score exploitability.
fn search_output_expl(
    tree: &FlatTree, np: u8, nh: usize, bp: &FlopStartVectorCfr, search_iters: u32, pot: i32,
) -> f32 {
    let game = FlopStartGame::new(clean_table(np, nh));
    let mut s = FlopStartVectorCfr::new(tree, game.table());
    // Inject the blueprint continuation (turn+river); flop stays fresh
    // (zeroed) so it is re-solved by the search.
    s.cum_strategy_turn_mut().copy_from_slice(bp.cum_strategy_turn());
    s.cum_strategy_river_mut().copy_from_slice(bp.cum_strategy_river());
    s.run_flop_search(tree, &game, search_iters);
    expl_pct(&s, tree, &game, pot)
}

/// Score a blueprint solver directly (no search) — the RAW roughness.
fn raw_expl(tree: &FlatTree, np: u8, nh: usize, bp: &FlopStartVectorCfr, pot: i32) -> f32 {
    let game = FlopStartGame::new(clean_table(np, nh));
    expl_pct(bp, tree, &game, pot)
}

// ─── S3 COST ARITHMETIC (before building nesting) ─────────────────────────

/// Rule out (or in) nesting BY ARITHMETIC before the build (user-ordered):
/// measure the per-iter wall cost of a gadget flop-search vs nh, fit the
/// scaling, project to production nh=1176. The flop-search un-abstracts to
/// FULL hand resolution, and the showdown terminal is O(nh^np) — the same
/// wall that forced bucketing. If a single full-nh flop-search iter is
/// already too slow, nesting is moot regardless of scheduling. Also reports
/// n_boundaries (flop→turn transitions) for the naive-vs-once-per-boundary
/// multiplication. Times the EXISTING two-sided gadget (the real cost unit).
#[test]
#[ignore = "S3 cost arithmetic; --ignored --nocapture --release"]
fn s3_cost_arithmetic() {
    use std::time::Instant;
    let (live, commit, pot) = (3u8, 7i32, 29i32);
    let tree = seam_tree(live, commit, pot);
    let n_boundaries = {
        // turn_chance_children count = flop→turn transitions. Build a solver
        // to read it (it's derived at construction).
        let g = FlopStartGame::new(clean_table(live, 12));
        let s = FlopStartVectorCfr::new(&tree, g.table());
        s.turn_chance_children().len()
    };
    eprintln!("\n═══ S3 COST ARITHMETIC (live-3, gadget flop-search) ═══");
    eprintln!("n_boundaries (flop→turn transitions) = {n_boundaries}");
    let k3: [(usize, f32); 3] = [(0, 0.0), (0, 0.6), (usize::MAX, 0.6)];
    const TIMED_ITERS: u32 = 20;
    let mut pts: Vec<(usize, f64)> = Vec::new();
    for &nh in &[16usize, 24, 32, 48] {
        let g = FlopStartGame::new(clean_table(live, nh));
        let mut s = FlopStartVectorCfr::new(&tree, g.table());
        // warm the continuation so the search does real work
        let fine = exact_solver(&tree, live, nh, 100);
        s.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
        s.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
        let t0 = Instant::now();
        s.run_flop_search_two_sided(&tree, &g, TIMED_ITERS, &k3);
        let ms_per_iter = t0.elapsed().as_secs_f64() * 1000.0 / TIMED_ITERS as f64;
        eprintln!("  nh={nh:>3}: {ms_per_iter:8.3} ms/iter (gadget, K=3)");
        pts.push((nh, ms_per_iter));
    }
    // Fit power law ms = a·nh^p from first and last point.
    let (n0, t0) = pts[0];
    let (n1, t1) = pts[pts.len() - 1];
    let p = (t1 / t0).ln() / (n1 as f64 / n0 as f64).ln();
    let a = t1 / (n1 as f64).powf(p);
    let proj_1176 = a * 1176f64.powf(p);
    eprintln!("  fitted scaling: ms/iter ≈ {a:.3e} · nh^{p:.2}");
    eprintln!("  PROJECTED nh=1176: {proj_1176:.1} ms/iter = {:.2} s/iter", proj_1176 / 1000.0);
    eprintln!("\n  ARITHMETIC vs 13s budget (per-iter × iters):");
    let per = proj_1176 / 1000.0;
    for &fi in &[100u32, 500, 1000] {
        eprintln!("    flop-search alone, {fi} iters: {:.1} s", per * fi as f64);
    }
    eprintln!("  NAIVE nesting (×): flop_it × n_bound × turn_it × extra-K — multiplies, see report.");
    eprintln!("  ONCE-PER-BOUNDARY (+): n_bound×turn_it + flop_it, added — the deployable form.");
    eprintln!("→ if full-nh flop-search per-iter alone blows 13s, search un-abstraction is wall-limited");
    eprintln!("  REGARDLESS of nesting; routes to option A or a resolution-bounded search.");

    // AFFORDABLE-RESOLUTION CROSSOVER (the interior the endpoint skipped):
    // invert the scaling — what search resolution nh_s fits 13s, including
    // iters-to-converge? CPU and GPU (GPU = measured-bucketed ~100× factor,
    // STATED ASSUMPTION for the exact-terminal path). B8 = 8 buckets is the
    // blueprint baseline search must beat to be useful.
    eprintln!("\n  AFFORDABLE search resolution nh_s (fits 13s budget):");
    eprintln!("  iters | CPU nh_s | GPU(×100) nh_s | vs B8=8");
    let budget_ms = 13000.0;
    for &iters in &[200u32, 500, 1000, 1500] {
        let solve = |speedup: f64| -> f64 {
            // a·nh^p · iters / speedup ≤ budget  →  nh ≤ (budget·speedup/(a·iters))^(1/p)
            (budget_ms * speedup / (a * iters as f64)).powf(1.0 / p)
        };
        let cpu = solve(1.0);
        let gpu = solve(100.0);
        eprintln!("  {iters:>5} | {cpu:7.1} | {gpu:11.1} | CPU {:.1}× / GPU {:.1}× B8",
            cpu / 8.0, gpu / 8.0);
    }
    eprintln!("→ CPU nh_s ≈ B8 (search useless multiway); GPU nh_s ≈ 8-14× B8 (the useful lever).");
    eprintln!("  QUALITY at nh_s (search-output vs B8 scored on full) = gadget-on-bucketed build OR harness");
    eprintln!("  (scorer wall O(nh^np)); cost side already shows the knob's range is NOT zero on GPU.");
}

// ─── diagnostic ──────────────────────────────────────────────────────────

/// Warm-start the search solver with the FULL fine equilibrium (flop +
/// turn + river), then search the flop with the (same) frozen continuation.
/// At 0 search iters this IS the exact solver → must score baseline. If it
/// DRIFTS UP from baseline as iters rise, the search DYNAMICS actively push
/// the flop away from equilibrium (a mechanism bug); if it STAYS, cold-start
/// convergence is the only issue.
#[test]
#[ignore = "S2 anchor diagnostic; --ignored --nocapture --release"]
fn s2_anchor_warmstart_diagnose() {
    let (live, commit, pot, nh) = (3u8, 7i32, 25i32, 18usize);
    let tree = seam_tree(live, commit, pot);
    let fine = exact_solver(&tree, live, nh, 600);
    let game = FlopStartGame::new(clean_table(live, nh));
    let base = expl_pct(&fine, &tree, &game, pot);
    eprintln!("baseline (exact) {base:.4}%");
    for &it in &[0u32, 50, 150, 400, 1000] {
        let g = FlopStartGame::new(clean_table(live, nh));
        let mut s = FlopStartVectorCfr::new(&tree, g.table());
        // WARM full equilibrium.
        s.cum_strategy_flop_mut().copy_from_slice(fine.cum_strategy_flop());
        s.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
        s.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
        // Seed regrets too so the searched strategy doesn't restart from
        // uniform — warm regrets reproduce the equilibrium current strategy.
        s.regrets_flop_mut().copy_from_slice(fine.regrets_flop());
        s.set_iteration(fine.iteration_count());
        s.run_flop_search(&tree, &g, it);
        let e = expl_pct(&s, &tree, &g, pot);
        eprintln!("  FROZEN-flop-search warm + {it:>4} iters: {e:.4}%");
    }
    // CONTROL: warm-start identically but run the FULL game (all zones
    // update). If this STAYS ≈baseline while frozen-search DRIFTS, the
    // FREEZING is the cause (range problem). If this ALSO drifts, it's
    // multiplayer-CFR instability the frozen version merely inherits.
    eprintln!("  --- control: full run() (no freeze), same warm start ---");
    for &it in &[0u32, 50, 150, 400, 1000] {
        let g = FlopStartGame::new(clean_table(live, nh));
        let mut s = FlopStartVectorCfr::new(&tree, g.table());
        s.cum_strategy_flop_mut().copy_from_slice(fine.cum_strategy_flop());
        s.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
        s.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
        s.regrets_flop_mut().copy_from_slice(fine.regrets_flop());
        s.regrets_turn_mut().copy_from_slice(fine.regrets_turn());
        s.set_iteration(fine.iteration_count());
        s.run(&tree, &g, it);
        let e = expl_pct(&s, &tree, &g, pot);
        eprintln!("  FULL run() warm + {it:>4} iters: {e:.4}%");
    }
    // GADGET GATE: the multi-continuation ROBUST search on the SAME warm
    // start + fine continuation. If this STAYS ≈baseline where naive
    // frozen search DIVERGES, the gadget fixes the multiway range problem
    // → S2 resumes with it. If it ALSO diverges, depth-limited search may
    // not work multiway at all (reflex armed: last divergence was a mirage).
    eprintln!("  --- gadget: run_flop_search_robust (multi-continuation), same warm start ---");
    for &it in &[0u32, 50, 150, 400, 1000] {
        let g = FlopStartGame::new(clean_table(live, nh));
        let mut s = FlopStartVectorCfr::new(&tree, g.table());
        s.cum_strategy_flop_mut().copy_from_slice(fine.cum_strategy_flop());
        s.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
        s.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
        s.regrets_flop_mut().copy_from_slice(fine.regrets_flop());
        s.set_iteration(fine.iteration_count());
        s.run_flop_search_robust(&tree, &g, it);
        let e = expl_pct(&s, &tree, &g, pot);
        eprintln!("  GADGET-robust(1-sided proxy) warm + {it:>4} iters: {e:.4}%");
    }
    // CONTROL GATE: two-sided gadget with the variant set COLLAPSED to
    // blueprint-only must reproduce run_flop_search (the k-choice is
    // trivial, opponents play pure blueprint) — proves the gadget plumbing
    // adds nothing spurious before trusting the K=3 result.
    let blueprint_only: [(usize, f32); 1] = [(0, 0.0)];
    let variants3: [(usize, f32); 3] = [(0, 0.0), (0, 0.6), (usize::MAX, 0.6)];
    for &it in &[150u32] {
        let g = FlopStartGame::new(clean_table(live, nh));
        let mut a = FlopStartVectorCfr::new(&tree, g.table());
        a.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
        a.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
        a.run_flop_search(&tree, &g, it);
        let ea = expl_pct(&a, &tree, &g, pot);
        let mut b = FlopStartVectorCfr::new(&tree, g.table());
        b.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
        b.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
        b.run_flop_search_two_sided(&tree, &g, it, &blueprint_only);
        let eb = expl_pct(&b, &tree, &g, pot);
        eprintln!("  CONTROL @ {it}it: run_flop_search {ea:.5}% vs two_sided[blueprint-only] {eb:.5}%");
        assert!((ea - eb).abs() < 1e-3,
            "two-sided[blueprint-only] {eb:.5}% != run_flop_search {ea:.5}% — gadget plumbing is \
             spurious; the K=3 result cannot be trusted until this control is bit-clean");
    }
    // GADGET GATE (faithful, two-sided): same warm start + fine continuation.
    // If THIS holds ≈baseline where naive frozen search diverges, the gadget
    // fixes the multiway range problem → S2 resumes with it. If it ALSO
    // diverges, depth-limited search may not work multiway at all.
    eprintln!("  --- gadget: run_flop_search_two_sided (FAITHFUL K=3), same warm start ---");
    for &it in &[0u32, 50, 150, 400, 1000] {
        let g = FlopStartGame::new(clean_table(live, nh));
        let mut s = FlopStartVectorCfr::new(&tree, g.table());
        s.cum_strategy_flop_mut().copy_from_slice(fine.cum_strategy_flop());
        s.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
        s.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
        s.regrets_flop_mut().copy_from_slice(fine.regrets_flop());
        s.set_iteration(fine.iteration_count());
        s.run_flop_search_two_sided(&tree, &g, it, &variants3);
        let e = expl_pct(&s, &tree, &g, pot);
        eprintln!("  GADGET-2sided warm + {it:>4} iters: {e:.4}%");
    }
}

/// CONTINUATION TOLERANCE — FINE SHAPE NEAR EXACT (2026-06-13): the coarse
/// map showed tight-in-COMPRESSION-RATIO, but nesting's question is
/// error-MAGNITUDE — where does the flop-search-delta cross zero in turn-
/// error pt? Two points interpolate to ~15pt (loose: a ~1% searched-turn
/// clears), but the near-exact shape is unknown and the tight ratio hints
/// superlinearity. Map fine turn buckets near exact; report turn-error
/// (raw(B)−raw(exact)) so the crossover is in the same pt units a searched
/// turn lives in. nh=32 permits B up to 26 (alive-hand guard).
#[test]
#[ignore = "S2 continuation tolerance fine shape; --ignored --nocapture --release"]
fn s2_continuation_tolerance_fine() {
    let (live, commit, pot, nh) = (3u8, 7i32, 29i32, 32usize);
    let tree = seam_tree(live, commit, pot);
    const REF: u32 = 600;
    const SEARCH_IT: u32 = 1500;
    let fine = exact_solver(&tree, live, nh, REF);
    let b8 = blueprint_solver(&tree, live, nh, Some(8), REF);
    let k3: [(usize, f32); 3] = [(0, 0.0), (0, 0.6), (usize::MAX, 0.6)];
    // raw(exact turn) reference for turn-error magnitude.
    let raw_exact = {
        let g2 = FlopStartGame::new(clean_table(live, nh));
        let mut m = FlopStartVectorCfr::new(&tree, g2.table());
        m.cum_strategy_flop_mut().copy_from_slice(b8.cum_strategy_flop());
        m.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
        m.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
        expl_pct(&m, &tree, &g2, pot)
    };
    eprintln!("\n═══ S2 CONTINUATION TOLERANCE — FINE SHAPE (live-3 nh={nh}, river EXACT) ═══");
    eprintln!("raw(exact turn)={raw_exact:.3}% | turn-err = raw(B)−raw(exact); crossover in pt = nesting margin");
    eprintln!("turn (ratio) | turn-err pt | raw | searched | flop-Δ (− helps)");
    for &bt in &[None, Some(26usize), Some(20), Some(16)] {
        let turn_src = match bt {
            None => exact_solver(&tree, live, nh, REF),
            Some(b) => blueprint_solver(&tree, live, nh, Some(b), REF),
        };
        let raw = {
            let g2 = FlopStartGame::new(clean_table(live, nh));
            let mut m = FlopStartVectorCfr::new(&tree, g2.table());
            m.cum_strategy_flop_mut().copy_from_slice(b8.cum_strategy_flop());
            m.cum_strategy_turn_mut().copy_from_slice(turn_src.cum_strategy_turn());
            m.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
            expl_pct(&m, &tree, &g2, pot)
        };
        let searched = {
            let g2 = FlopStartGame::new(clean_table(live, nh));
            let mut s = FlopStartVectorCfr::new(&tree, g2.table());
            s.cum_strategy_turn_mut().copy_from_slice(turn_src.cum_strategy_turn());
            s.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
            s.run_flop_search_two_sided(&tree, &g2, SEARCH_IT, &k3);
            expl_pct(&s, &tree, &g2, pot)
        };
        let ratio = nh as f32 / bt.unwrap_or(nh) as f32;
        let terr = raw - raw_exact;
        let d = searched - raw;
        let label = match bt { None => "exact".to_string(), Some(b) => format!("B={b}") };
        eprintln!("  {label:>6} ({ratio:>4.2}:1) | {terr:6.2}pt | {raw:7.3}% | {searched:7.3}% | {d:+6.2} {}",
            if d < 0.0 { "HELPS" } else { "HURTS" });
    }
    eprintln!("→ crossover turn-err (pt) vs searched-turn ~1pt: margin >> 1 ⇒ nesting clears (structure = anchor-gate);");
    eprintln!("  margin ≲ 1 ⇒ build turn-search, test the SEARCHED-turn object directly before nesting.");
}

/// CONTINUATION-VALUE TOLERANCE MAP (2026-06-13, the corrected S2 question).
/// Sizes the ONE real requirement the fork-reversal leaves: how accurate
/// must the multiway blueprint's CONTINUATION VALUES be for flop-search to
/// NET-IMPROVE over playing the bucketed flop raw? Three sharpenings:
///   (1) METRIC = crossover, not absolute bar: per continuation fidelity,
///       compare searched-output vs RAW bucketed-flop (same continuation).
///       Crossover = fidelity where searched == raw; finer ⇒ search helps,
///       rougher ⇒ search HURTS (continuation-error amplification beats the
///       un-abstraction benefit). The crossover IS the minimum continuation
///       fidelity = the blueprint spec.
///   (2) RIVER HELD EXACT (it is search-solved/showdown-exact at runtime);
///       vary ONLY the TURN continuation — the only place the burden lands.
///   (3) report in COMPRESSION-RATIO units (nh/B_turn). Production bucket
///       count is a TRANSFER of the crossover ratio to 147:1 scale — named
///       caveat, harness-confirmed; this does NOT emit a production-B directly.
#[test]
#[ignore = "S2 continuation-value tolerance map; --ignored --nocapture --release"]
fn s2_continuation_tolerance() {
    let (live, commit, pot, nh) = (3u8, 7i32, 29i32, 24usize);
    let tree = seam_tree(live, commit, pot);
    const REF: u32 = 600;
    const SEARCH_IT: u32 = 2000;
    let fine = exact_solver(&tree, live, nh, REF);
    let g = FlopStartGame::new(clean_table(live, nh));
    let baseline = expl_pct(&fine, &tree, &g, pot);
    // The raw flop = production-candidate B8 bucketing (the un-abstraction
    // opportunity is B8-flop → full-res-flop).
    let b8 = blueprint_solver(&tree, live, nh, Some(8), REF);
    let k3: [(usize, f32); 3] = [(0, 0.0), (0, 0.6), (usize::MAX, 0.6)];

    eprintln!("\n═══ S2 CONTINUATION-VALUE TOLERANCE (live-3 nh={nh}, RIVER held EXACT) ═══");
    eprintln!("baseline (all exact) {baseline:.3}% | flop raw = B8 | metric = searched vs raw (crossover)");
    eprintln!("turn fidelity (ratio) | raw: B8flop+turn+exactRiver | searched: fullflop+turn+exactRiver | search?");
    for &bt in &[None, Some(16usize), Some(8), Some(4)] {
        // Turn-continuation source at fidelity `bt` (None = exact).
        let turn_src = match bt {
            None => exact_solver(&tree, live, nh, REF),
            Some(b) => blueprint_solver(&tree, live, nh, Some(b), REF),
        };
        // RAW: bucketed (B8) flop + this turn + EXACT river — play blueprint raw.
        let raw = {
            let g2 = FlopStartGame::new(clean_table(live, nh));
            let mut m = FlopStartVectorCfr::new(&tree, g2.table());
            m.cum_strategy_flop_mut().copy_from_slice(b8.cum_strategy_flop());
            m.cum_strategy_turn_mut().copy_from_slice(turn_src.cum_strategy_turn());
            m.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
            expl_pct(&m, &tree, &g2, pot)
        };
        // SEARCHED: full-res flop search + this turn + EXACT river.
        let searched = {
            let g2 = FlopStartGame::new(clean_table(live, nh));
            let mut s = FlopStartVectorCfr::new(&tree, g2.table());
            s.cum_strategy_turn_mut().copy_from_slice(turn_src.cum_strategy_turn());
            s.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
            s.run_flop_search_two_sided(&tree, &g2, SEARCH_IT, &k3);
            expl_pct(&s, &tree, &g2, pot)
        };
        let ratio = nh as f32 / bt.unwrap_or(nh) as f32;
        let label = match bt { None => "exact".to_string(), Some(b) => format!("B={b}") };
        let verdict = if searched < raw { format!("HELPS (−{:.1}pt)", raw - searched) }
            else { format!("HURTS (+{:.1}pt)", searched - raw) };
        eprintln!("  {label:>6} ({ratio:>4.1}:1) | {raw:8.3}% | {searched:8.3}% | {verdict}");
    }
    eprintln!("→ crossover ratio (HELPS→HURTS) = min continuation fidelity for search to pay.");
    eprintln!("  LOOSE (helps to high ratio) ⇒ cheap-ish multiway blueprint, no nesting needed.");
    eprintln!("  TIGHT (only helps near exact) ⇒ nesting (searched-turn values) is the justified build.");
    eprintln!("  CAVEAT: research-scale ratios; production-B is a TRANSFER of the crossover to 147:1, harness-confirmed.");
}

/// THE ARCHITECTURAL FORK (2026-06-13): does search IMPROVE a multiway
/// blueprint AT ALL, given an ACCURATE continuation? Isolates the two costs
/// the corrects test conflated — un-abstraction BENEFIT vs continuation-VALUE
/// error — by holding the continuation EXACT and varying only the flop:
///   baseline      = exact flop + exact continuation (search's ideal target)
///   B8-flop+exact = bucketed flop + EXACT continuation (the un-abstraction
///                   opportunity — what a fine blueprint's flop-abstraction
///                   costs WITHOUT any continuation error)
///   gadget search = full-res flop search + exact continuation (~1% floor)
/// If B8-flop+exact > gadget ⇒ search un-abstracts usefully (corrects-test
/// failure was the rough CONTINUATION, not search). If B8-flop+exact <
/// gadget ⇒ search HURTS even with a perfect continuation ⇒ net-negative
/// multiway ⇒ play raw, search is heads-up-only. Research-scale; harness is
/// the production verdict, but the sign of the inequality is the fork.
#[test]
#[ignore = "S2 architectural fork: does search improve a multiway blueprint; --ignored --nocapture --release"]
fn s2_does_search_help_multiway() {
    let (live, commit, pot, nh) = (3u8, 7i32, 25i32, 18usize);
    let tree = seam_tree(live, commit, pot);
    const REF: u32 = 600;
    let fine = exact_solver(&tree, live, nh, REF);
    let g = FlopStartGame::new(clean_table(live, nh));
    let baseline = expl_pct(&fine, &tree, &g, pot);
    let b8 = blueprint_solver(&tree, live, nh, Some(8), REF);
    let raw_b8_full = raw_expl(&tree, live, nh, &b8, pot);

    // B8 flop + EXACT continuation: exact everywhere, flop replaced by B8.
    let mixed_expl = {
        let g2 = FlopStartGame::new(clean_table(live, nh));
        let mut m = FlopStartVectorCfr::new(&tree, g2.table());
        m.cum_strategy_flop_mut().copy_from_slice(b8.cum_strategy_flop());
        m.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
        m.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
        expl_pct(&m, &tree, &g2, pot)
    };

    let k3: [(usize, f32); 3] = [(0, 0.0), (0, 0.6), (usize::MAX, 0.6)];
    eprintln!("\n═══ S2 FORK: does search improve a multiway blueprint? (live-3 nh={nh}) ═══");
    eprintln!("baseline (exact flop + exact cont)     : {baseline:.4}%");
    eprintln!("B8 full (B8 flop + B8 cont, raw)       : {raw_b8_full:.4}%");
    eprintln!("B8 flop + EXACT cont (raw, no search)  : {mixed_expl:.4}%   ← un-abstraction opportunity");
    eprint!("gadget search (full flop + EXACT cont) :");
    let mut best = f32::INFINITY;
    for &it in &[300u32, 800, 2000] {
        let gg = FlopStartGame::new(clean_table(live, nh));
        let mut s = FlopStartVectorCfr::new(&tree, gg.table());
        s.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
        s.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
        s.run_flop_search_two_sided(&tree, &gg, it, &k3);
        let e = expl_pct(&s, &tree, &gg, pot);
        eprint!(" {it}→{e:.3}%");
        best = best.min(e);
    }
    eprintln!("  (best {best:.3}%)");
    eprintln!("→ if B8-flop+exact ({mixed_expl:.3}%) > gadget best ({best:.3}%): search UN-ABSTRACTS usefully");
    eprintln!("  (corrects failure = rough CONTINUATION). If <: search HURTS w/ perfect cont = net-neg multiway.");
}

/// Depth-limited search output with the FAITHFUL two-sided gadget: freeze
/// `bp`'s turn+river as the continuation, search the flop FRESH (zeroed)
/// for `iters`, score. The flop is re-solved against the (possibly rough)
/// frozen continuation via the multi-continuation choice.
fn search_output_two_sided(
    tree: &FlatTree, np: u8, nh: usize, bp: &FlopStartVectorCfr,
    iters: u32, variants: &[(usize, f32)], pot: i32,
) -> f32 {
    let game = FlopStartGame::new(clean_table(np, nh));
    let mut s = FlopStartVectorCfr::new(tree, game.table());
    s.cum_strategy_turn_mut().copy_from_slice(bp.cum_strategy_turn());
    s.cum_strategy_river_mut().copy_from_slice(bp.cum_strategy_river());
    s.run_flop_search_two_sided(tree, &game, iters, variants);
    expl_pct(&s, tree, &game, pot)
}

/// S2 CORRECTS TEST (2026-06-13): does the two-sided gadget search RESCUE a
/// ROUGH blueprint toward the ~1% band? Three lenses: (1) capability —
/// output < raw rough blueprint? (2) structural floor — do rough AND fine
/// seeds converge to the SAME residual? if so ~1% is the multiway floor,
/// not a continuation-FORM weakness (re-solved continuations would be
/// futile). (3) research-scale — production roughness rescue rides to the
/// harness (the transfer wall). Cold flop search (the real depth-limited
/// scenario: rough blueprint in hand, re-solve the flop against its frozen
/// continuation).
#[test]
#[ignore = "S2 two-sided corrects test; --ignored --nocapture --release"]
fn s2_two_sided_corrects() {
    let (live, commit, pot, nh) = (3u8, 7i32, 25i32, 18usize);
    let tree = seam_tree(live, commit, pot);
    const REF: u32 = 600;
    let baseline = {
        let s = exact_solver(&tree, live, nh, REF);
        let g = FlopStartGame::new(clean_table(live, nh));
        expl_pct(&s, &tree, &g, pot)
    };
    let k3: [(usize, f32); 3] = [(0, 0.0), (0, 0.6), (usize::MAX, 0.6)];
    eprintln!("\n═══ S2 CORRECTS: two-sided gadget rescue of rough blueprints (live-3 nh={nh}) ═══");
    eprintln!("baseline (exact) {baseline:.4}% | RESEARCH-SCALE; production roughness rescue → harness");
    eprintln!("roughness | raw blueprint | gadget-search out @500 @1500 | capability");
    for &nb in &[None, Some(8usize), Some(4usize)] {
        let bp = blueprint_solver(&tree, live, nh, nb, REF);
        let raw = raw_expl(&tree, live, nh, &bp, pot);
        let o500 = search_output_two_sided(&tree, live, nh, &bp, 500, &k3, pot);
        let o1500 = search_output_two_sided(&tree, live, nh, &bp, 1500, &k3, pot);
        let label = match nb { None => "fine  ".into(), Some(b) => format!("B={b:<4}") };
        let cap = if nb.is_none() { "(anchor)".to_string() }
            else if o1500 < raw * 0.85 { format!("CORRECTS ({:.1}×→band)", raw / o1500) }
            else { "no rescue".to_string() };
        eprintln!("  {label} | {raw:7.3}% | {o500:7.3}%  {o1500:7.3}% | {cap}");
    }
    eprintln!("→ rough out → fine band ⇒ search rescues, branch-2 viable, bar drops to seed-quality.");
    eprintln!("  rough out ≈ fine out ⇒ ~1% is the STRUCTURAL multiway floor (form-upgrade futile).");
}

/// Tail of the two-sided gadget warm-start anchor: does it converge to
/// baseline (gadget fully cures) or plateau above it (continuation set too
/// thin → enrich K)? Convergence-by-iter-sweep reflex before calling 0.75%
/// structural. Also tests a richer K=5 continuation set.
#[test]
#[ignore = "S2 two-sided gadget anchor tail; --ignored --nocapture --release"]
fn s2_two_sided_anchor_tail() {
    let (live, commit, pot, nh) = (3u8, 7i32, 25i32, 18usize);
    let tree = seam_tree(live, commit, pot);
    let fine = exact_solver(&tree, live, nh, 600);
    let game = FlopStartGame::new(clean_table(live, nh));
    let base = expl_pct(&fine, &tree, &game, pot);
    eprintln!("baseline (exact) {base:.4}%");
    let k3: [(usize, f32); 3] = [(0, 0.0), (0, 0.6), (usize::MAX, 0.6)];
    // Richer set: blueprint + mild/strong passive + mild/strong aggressive.
    let k5: [(usize, f32); 5] =
        [(0, 0.0), (0, 0.35), (0, 0.7), (usize::MAX, 0.35), (usize::MAX, 0.7)];
    for (label, vs) in [("K=3", &k3[..]), ("K=5", &k5[..])] {
        eprintln!("  -- two-sided {label} warm-start tail --");
        for &it in &[500u32, 1500, 4000, 9000] {
            let g = FlopStartGame::new(clean_table(live, nh));
            let mut s = FlopStartVectorCfr::new(&tree, g.table());
            s.cum_strategy_flop_mut().copy_from_slice(fine.cum_strategy_flop());
            s.cum_strategy_turn_mut().copy_from_slice(fine.cum_strategy_turn());
            s.cum_strategy_river_mut().copy_from_slice(fine.cum_strategy_river());
            s.regrets_flop_mut().copy_from_slice(fine.regrets_flop());
            s.set_iteration(fine.iteration_count());
            s.run_flop_search_two_sided(&tree, &g, it, vs);
            let e = expl_pct(&s, &tree, &g, pot);
            eprintln!("    {label} warm + {it:>4} iters: {e:.4}%  ({:.1}× baseline)", e / base);
        }
    }
    eprintln!("→ falls toward baseline ⇒ gadget fully cures; plateaus ⇒ enrich continuations.");
}

// ─── S2 measurement ──────────────────────────────────────────────────────

#[test]
#[ignore = "S2 tolerance 2-D (research nh, minutes); --ignored --nocapture --release"]
fn s2_search_tolerance_to_roughness() {
    // (live, commit, pot, nh) — live-3/4, the exploitability-GROUNDED
    // families. nh kept feasible for the O(nh^np) scorer.
    let fams: &[(u8, i32, i32, usize)] = &[(3, 7, 25, 24), (4, 7, 29, 14)];
    const REF_ITERS: u32 = 600;     // baseline + blueprint convergence
    // The iterations axis is LOAD-BEARING: the warm-equilibrium diagnostic
    // showed naive frozen-continuation search DIVERGES multiway (the range
    // problem, non-mirage), so "more search" is not monotone-better — there
    // is an OPTIMAL stop. Sweep it and report the per-roughness MINIMUM.
    let search_pts = [25u32, 50, 100, 200, 400, 800];

    eprintln!("\n═══ S2: SEARCH TOLERANCE TO BLUEPRINT ROUGHNESS (clean board, live-3/4) ═══");
    eprintln!("exploitability-GROUNDED (live-3/4); live-5/6 = stability+transfer (3rd harness residual)");
    eprintln!("FINDING (diagnostic-confirmed): naive single-continuation depth-limited search is");
    eprintln!("UNSAFE multiway — even a FINE continuation diverges (range problem, non-mirage). So");
    eprintln!("the grid reports search-output across the ITERS axis and the per-roughness MINIMUM");
    eprintln!("(best search can do) — NOT a converged tail (which diverges).");

    for &(live, commit, pot, nh) in fams {
        let tree = seam_tree(live, commit, pot);
        let baseline = {
            let s = exact_solver(&tree, live, nh, REF_ITERS);
            let game = FlopStartGame::new(clean_table(live, nh));
            expl_pct(&s, &tree, &game, pot)
        };
        // Roughness ladder respecting per-runout alive-hand count (river
        // can drop ~nb+6 below nh after card-removal; guard keeps
        // quantile_maps well-posed).
        let roughness: Vec<Option<usize>> = std::iter::once(None)
            .chain([16usize, 8, 4].into_iter().filter(|&b| b + 6 <= nh).map(Some))
            .collect();
        eprintln!("\n── live-{live} (np={live}, nh={nh}) | baseline (exact, {REF_ITERS} it) {baseline:.3}% ──");
        eprint!("  roughness (ratio) | raw  ");
        for it in search_pts { eprint!("| {it:>5}it "); }
        eprintln!("| BEST(@it)");

        for &nb in &roughness {
            let bp = blueprint_solver(&tree, live, nh, nb, REF_ITERS);
            let raw = raw_expl(&tree, live, nh, &bp, pot);
            let label = match nb { None => "fine ".to_string(), Some(b) => format!("B={b:<3}") };
            let ratio = nh as f32 / nb.unwrap_or(nh) as f32;
            eprint!("  {label}({ratio:>4.1}:1) | {raw:5.2}");
            let mut best = f32::INFINITY;
            let mut best_it = 0u32;
            for &it in &search_pts {
                let e = search_output_expl(&tree, live, nh, &bp, it, pot);
                eprint!("| {e:6.2} ");
                if e < best { best = e; best_it = it; }
            }
            // Does search's BEST point beat the raw blueprint (the
            // capability S2 credits — correction), and how close to baseline?
            let helps = if best < raw * 0.85 { "corrects" }
                else if best <= raw + 0.05 { "≈raw" } else { "WORSE" };
            eprintln!("| {best:5.2}(@{best_it}) {helps}");
        }
    }
    eprintln!("\nTOLERANCE READ (honest, naive-search): per roughness, BEST = the lowest exploitability");
    eprintln!("search reaches before the range-problem drift dominates. Search CORRECTS a rough");
    eprintln!("blueprint transiently but does NOT recover equilibrium quality multiway with a frozen");
    eprintln!("single continuation → the multi-continuation gadget (S1-built) is REQUIRED for");
    eprintln!("equilibrium-grade refinement. live-5/6: TRANSFER (stability mapped, quality walled) =");
    eprintln!("3rd named harness residual. S3: the live iters budget must STOP before drift (the BEST@it");
    eprintln!("column is the operating point); S4 re-prices at the fidelity this minimum implies.");
}
