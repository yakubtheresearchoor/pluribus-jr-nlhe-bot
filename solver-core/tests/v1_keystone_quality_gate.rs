//! V1 KEYSTONE QUALITY GATE (2026-06-13, user: "does the keystone pass
//! the lift-and-score quality gate at production B=8, not just the NH=6
//! machinery gates?"). The five keystone gates this turn are all
//! MACHINERY/identity gates (bit-exact, bijection, reach-independence,
//! wiring, small-nh values) — they prove correctness-of-COMPUTATION.
//! NONE proves quality-of-STRATEGY: that the B=8 compression THROUGH
//! this object plays acceptably. The project has repeatedly shown those
//! two diverge (a strategy can be computed correctly yet be a poor
//! compression). The earlier bucketing arc cleared the quality gate —
//! but on the 6p ORACLE-shape game. The keystone compresses NEW game
//! shapes (the live-SUBSET seam trees: live-3/4 with folded players),
//! so the quality gate runs once HERE on those shapes before banking
//! the keystone fill-ready.
//!
//! Method = the B3 lift-and-score (p1_5_4_bucketing_b3_quality_gate):
//! solve bucketed, LIFT the bucketed strategy to the exact full-nh game
//! (`lift_cum_to_exact`), score exploitability (% pot). Production B=8 =
//! the production bucket count; nh is RESEARCH-SCALE because the
//! exploitability scorer is O(nh^(np+1)) — infeasible at production
//! nh=1176 for ANYONE (the standing reason the harness head-to-head is
//! the high-np verdict). live-3/4 are scorable here; live-5/6 quality
//! rides the harness (named, consistent with the established wall).
//!
//! Gate per shape: B=8 lifted exploitability must be (a) FAR below the
//! B=2 crude-compression damage (the harness-can-rank check) and (b)
//! within a sane multiple of the exact baseline floor. Counts measured
//! independently per shape — never interpolated (the standing
//! bucket-count discipline).

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{
    lift_cum_to_exact, BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// Research-nh subset table on a fixed board for `np` live players.
fn research_table(np: u8, nh: usize) -> (FlopChanceTable, [Card; 3]) {
    let board: [Card; 3] = ["Th", "9d", "8c"].map(|s| card_from_str(s).unwrap());
    let bmask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if bmask & ((1u64 << c1) | (1u64 << c2)) == 0 {
            all_valid.push(idx as u16);
        }
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> =
        (0..np).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..np as usize {
        for &hi in &chosen { ranges[p][hi as usize] = 1.0; }
    }
    // 2×2 runout (sane sampled chance; quality is about the bucketing).
    let turn_cards: Vec<u8> =
        ["2c", "Jd"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let river_strs: [&[&str]; 2] = [&["4s", "7h"], &["3s", "Qc"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] =
            river_strs[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    (
        FlopChanceTable::compute_flop_start_subset_with_decks(
            &board, &ranges, np, &chosen, &turn_cards, &river_decks,
        ),
        board,
    )
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

/// Solve bucketed at nb, lift to exact, score exploitability % pot.
fn lifted_expl(tree: &FlatTree, np: u8, nh: usize, nb: usize, iters: u32, pot: i32) -> f32 {
    let (table, _) = research_table(np, nh);
    let (fm, tm, rm) = quantile_maps(&table, nb);
    let bk = FlopBucketing::from_maps(&table, nb, nb, nb, fm, tm, rm);
    let game = FlopStartGame::new(table);
    let mut bucketed = BucketedFlopCfr::new(tree, game.table(), &bk);
    bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
    bucketed.run(tree, &game, &bk, iters);

    let (t2, _) = research_table(np, nh);
    let gs = FlopStartGame::new(t2);
    let mut scorer = FlopStartVectorCfr::new(tree, gs.table());
    lift_cum_to_exact(tree, &bucketed, &bk, &mut scorer);
    expl_pct(&scorer, tree, &gs, pot)
}

/// Lift the IDENTITY (B=nh) bucketing — the lift-instrument anchor.
/// FlopBucketing::identity maps each hand to itself (incl. runout-
/// conflicting hands), so it reproduces the exact solve; lifting it must
/// ≈ the exact baseline, validating the lift indexing on this shape.
fn identity_lifted_expl(tree: &FlatTree, np: u8, nh: usize, iters: u32, pot: i32) -> f32 {
    let (table, _) = research_table(np, nh);
    let bk = FlopBucketing::identity(&table);
    let game = FlopStartGame::new(table);
    let mut bucketed = BucketedFlopCfr::new(tree, game.table(), &bk);
    bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
    bucketed.run(tree, &game, &bk, iters);
    let (t2, _) = research_table(np, nh);
    let gs = FlopStartGame::new(t2);
    let mut scorer = FlopStartVectorCfr::new(tree, gs.table());
    lift_cum_to_exact(tree, &bucketed, &bk, &mut scorer);
    expl_pct(&scorer, tree, &gs, pot)
}

/// Exact (unbucketed) converged exploitability — the floor.
fn exact_expl(tree: &FlatTree, np: u8, nh: usize, iters: u32, pot: i32) -> f32 {
    let (table, _) = research_table(np, nh);
    let game = FlopStartGame::new(table);
    let mut exact = FlopStartVectorCfr::new(tree, game.table());
    exact.run(tree, &game, iters);
    expl_pct(&exact, tree, &game, pot)
}

#[test]
#[ignore = "keystone B=8 lift-and-score quality (research nh, minutes); --ignored --nocapture --release"]
fn v1_keystone_quality_b8_live_subset() {
    const NH: usize = 24; // B=8 over 24 hands = 3 hands/bucket: real compression
    const ITERS: u32 = 60;
    // (live, commit, pot) — multiway-with-folds seam cells.
    let shapes: &[(u8, i32, i32, &str)] = &[
        (3, 7, 25, "live-3 single-raised (1 folder)"),
        (4, 7, 29, "live-4 raised (the priced family)"),
    ];
    eprintln!("\n═══ KEYSTONE QUALITY: B=8 lift-and-score on live-subset shapes (NH={NH}) ═══");
    eprintln!("(production B=8; research nh — exploitability is O(nh^(np+1)), infeasible at nh=1176)");
    for &(live, commit, pot, label) in shapes {
        let tree = seam_tree(live, commit, pot);
        let base = exact_expl(&tree, live, NH, ITERS, pot);
        // LIFT ANCHOR (B3 precedent): lifting the B=nh IDENTITY bucketing
        // must reproduce ~the exact result — validates the lift indexing
        // ITSELF on this tree shape before any B=8 verdict. If this is
        // far from baseline, the lift/scorer is broken on the live-subset
        // shape (instrument bug), NOT a B=8 quality regression.
        let lift_anchor = identity_lifted_expl(&tree, live, NH, ITERS, pot);
        eprintln!("{label}: LIFT ANCHOR (B=nh) {lift_anchor:.3}% vs baseline {base:.3}%");
        assert!(
            (lift_anchor - base).abs() < 0.5 || lift_anchor < base * 3.0,
            "{label}: lift anchor {lift_anchor:.3}% diverges from baseline {base:.3}% — \
             the LIFT INSTRUMENT is broken on this tree shape; B=8 numbers are meaningless \
             until this is fixed (NOT a keystone quality verdict)"
        );
        let b8 = lifted_expl(&tree, live, NH, 8, ITERS, pot);
        let b2 = lifted_expl(&tree, live, NH, 2, ITERS, pot);
        eprintln!(
            "{label}: baseline {base:.3}% | B=8 {b8:.3}% | B=2 {b2:.3}% (damage sanity)"
        );
        // (a) harness-can-rank: crude B=2 must be MEASURABLY worse than B=8.
        assert!(b2 > b8 * 1.5, "{label}: B=2 ({b2:.3}%) not measurably worse than B=8 ({b8:.3}%) — \
            the lift-and-score instrument can't rank compression here; verdict untrustworthy");
        // (b) B=8 in a sane regime vs the floor (banked B=8 ≈ 2-8% pot;
        // bar: within the established multiwa​y B=8 envelope, generously
        // 12% absolute, and not pathological vs baseline).
        assert!(b8 < 12.0, "{label}: B=8 lifted exploitability {b8:.3}% out of the banked B=8 regime — \
            compression-quality REGRESSION the identity gates are blind to");
    }
    eprintln!("KEYSTONE QUALITY PASSED: B=8 compression on the live-subset shapes plays in the banked regime");
}

/// DIAGNOSTIC (2026-06-13): is the live-3 B=8 79% real bucketing badness
/// or an undertraining / research-scale artifact? Sweep iters × nh,
/// lift anchor each, on the live-3 cell.
#[test]
#[ignore = "diagnostic sweep; --ignored --nocapture --release"]
fn keystone_quality_diagnose_live3() {
    let tree = seam_tree(3, 7, 25);
    let pot = 25;
    for &nh in &[16usize, 24] {
        for &iters in &[60u32, 300] {
            let base = exact_expl(&tree, 3, nh, iters, pot);
            let anchor = identity_lifted_expl(&tree, 3, nh, iters, pot);
            let b8 = lifted_expl(&tree, 3, nh, 8, iters, pot);
            eprintln!(
                "live-3 nh={nh} iters={iters}: baseline {base:.2}% | identity-lift {anchor:.2}% | B=8 {b8:.2}%"
            );
        }
    }
}

/// Inspect the np=3 B=8 flop bucketing distribution — is it degenerate
/// (hands collapsing into few buckets) or healthy (8 even buckets)?
#[test]
#[ignore = "bucketing inspection; --ignored --nocapture --release"]
fn inspect_live3_bucketing() {
    for &nh in &[16usize, 24] {
        let (table, _) = research_table(3, nh);
        let (fm, _tm, _rm) = quantile_maps(&table, 8);
        let mut counts = std::collections::BTreeMap::new();
        for &b in &fm {
            *counts.entry(b).or_insert(0usize) += 1;
        }
        eprintln!("np=3 nh={nh} B=8 flop map: {} entries, bucket→count {:?}", fm.len(), counts);
    }
    // Compare np=6 (the calibrated-good case) at nh=16.
    let (table6, _) = research_table(6, 16);
    let (fm6, _, _) = quantile_maps(&table6, 8);
    let mut c6 = std::collections::BTreeMap::new();
    for &b in &fm6 { *c6.entry(b).or_insert(0usize) += 1; }
    eprintln!("np=6 nh=16 B=8 flop map: {} entries, bucket→count {:?}", fm6.len(), c6);
}

/// live-4 + live-3 B=8 lift-score (np trend for the finding).
#[test]
#[ignore = "np trend; --ignored --nocapture --release"]
fn keystone_quality_np_trend() {
    const NH: usize = 16;
    const IT: u32 = 80;
    for &(live, commit, pot, label) in &[
        (3u8, 7i32, 25i32, "live-3"),
        (4u8, 7i32, 29i32, "live-4"),
    ] {
        let tree = seam_tree(live, commit, pot);
        let base = exact_expl(&tree, live, NH, IT, pot);
        let anchor = identity_lifted_expl(&tree, live, NH, IT, pot);
        let b8 = lifted_expl(&tree, live, NH, 8, IT, pot);
        eprintln!("{label} (np={live}) nh={NH}: baseline {base:.2}% | identity-lift {anchor:.2}% | B=8 {b8:.2}%");
    }
}

// ════════════════════════════════════════════════════════════════════
// FOLD-NODE HYPOTHESIS (2026-06-13, user): the leading hypothesis is
// NOT "B=8 too coarse" but "the fold-CONTINUATION subtrees are
// mishandled" — two fingers point there: the banked B=8 calibration
// went stale EXACTLY when the fold-fix added those subtrees, and the
// one un-gated keystone residual (slice-2 folded-opponent card-removal)
// lives EXACTLY there. A plain B-curve can't separate the two (identity
// has no compression to mishandle, so it's clean under BOTH). The
// discriminator: REGION-SWAP — lift B=8, then overwrite the strategy
// ONLY at fold-continuation nodes with the EXACT equilibrium, re-score.
//   - if exploitability collapses to baseline → the fold region IS the
//     culprit (correctness; slice-2 residual is an ACTIVE bug)
//   - if it stays high → spread (genuine compression-granularity; raise
//     B, a budget knob)
// Plus the complement (exact everywhere EXCEPT fold region) as control.

use solver_core::tree::flat::MAX_NA_POSTFLOP;
use solver_core::tree::action::BoardState;

/// Decision nodes whose path from the root includes a FOLD edge — the
/// fold-continuation subtrees the multiway fold-fix added.
fn post_fold_nodes(tree: &FlatTree) -> Vec<bool> {
    let nn = tree.nodes.len();
    let mut post = vec![false; nn];
    // DFS carrying `seen` = a FOLD edge is on the path INTO this node.
    // The root's action_label is spuriously 0 (== ACTION_LABEL_FOLD) but
    // no edge led to it, so we seed root with seen=false and test each
    // CHILD's own incoming label — NOT the node's own label (that bug
    // marked the whole tree post-fold; the 216/216 count caught it).
    let mut stack = vec![(0usize, false)];
    while let Some((idx, seen)) = stack.pop() {
        post[idx] = seen && tree.nodes[idx].is_player();
        for &c in tree.node_children(idx) {
            let child_fold = seen || tree.nodes[c as usize].action_label == 0; // FOLD=0
            stack.push((c as usize, child_fold));
        }
    }
    post
}

/// Copy `exact`'s cum_strategy into `b8` for nodes where pred(idx),
/// across all zones and outcome indices.
fn swap_region(
    b8: &mut FlopStartVectorCfr,
    exact: &FlopStartVectorCfr,
    tree: &FlatTree,
    nh: usize,
    pred: &dyn Fn(usize) -> bool,
) {
    let blk = MAX_NA_POSTFLOP * nh;
    let n_turn = exact.n_turn_outcomes();
    let ts = exact.turn_stride();
    let rs = exact.river_stride();
    let mnr = exact.max_n_river();
    for idx in 0..tree.nodes.len() {
        if !tree.nodes[idx].is_player() || !pred(idx) {
            continue;
        }
        match tree.nodes[idx].board_state {
            x if x == BoardState::Flop as u8 => {
                if let Some(local) = exact.flop_local_offset_at(idx) {
                    let o = local * blk;
                    let src = exact.cum_strategy_flop()[o..o + blk].to_vec();
                    b8.cum_strategy_flop_mut()[o..o + blk].copy_from_slice(&src);
                }
            }
            x if x == BoardState::Turn as u8 => {
                if let Some(local) = exact.turn_local_offset_at(idx) {
                    for tc in 0..n_turn {
                        let o = tc * ts + local * blk;
                        let src = exact.cum_strategy_turn()[o..o + blk].to_vec();
                        b8.cum_strategy_turn_mut()[o..o + blk].copy_from_slice(&src);
                    }
                }
            }
            x if x == BoardState::River as u8 => {
                if let Some(local) = exact.river_local_offset_at(idx) {
                    for tc in 0..n_turn {
                        for rc in 0..mnr {
                            let o = (tc * mnr + rc) * rs + local * blk;
                            if o + blk <= b8.cum_strategy_river().len() {
                                let src = exact.cum_strategy_river()[o..o + blk].to_vec();
                                b8.cum_strategy_river_mut()[o..o + blk].copy_from_slice(&src);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Build a B=8-lifted scorer, optionally swap a region to exact, score.
fn region_swap_expl(
    tree: &FlatTree, np: u8, nh: usize, iters: u32, pot: i32,
    swap_pred: Option<&dyn Fn(usize) -> bool>,
) -> f32 {
    // exact equilibrium (for both baseline strategy AND the swap source).
    let (te, _) = research_table(np, nh);
    let ge = FlopStartGame::new(te);
    let mut exact = FlopStartVectorCfr::new(tree, ge.table());
    exact.run(tree, &ge, iters);

    // B=8 lifted into a fresh scorer.
    let (tb, _) = research_table(np, nh);
    let (fm, tm, rm) = quantile_maps(&tb, 8);
    let bk = FlopBucketing::from_maps(&tb, 8, 8, 8, fm, tm, rm);
    let gb = FlopStartGame::new(tb);
    let mut bucketed = BucketedFlopCfr::new(tree, gb.table(), &bk);
    bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
    bucketed.run(tree, &gb, &bk, iters);

    let (ts2, _) = research_table(np, nh);
    let gs = FlopStartGame::new(ts2);
    let mut b8 = FlopStartVectorCfr::new(tree, gs.table());
    lift_cum_to_exact(tree, &bucketed, &bk, &mut b8);

    if let Some(pred) = swap_pred {
        swap_region(&mut b8, &exact, tree, nh, pred);
    }
    expl_pct(&b8, tree, &gs, pot)
}

#[test]
#[ignore = "fold-node hypothesis discriminator; --ignored --nocapture --release"]
fn keystone_fold_node_hypothesis() {
    const NH: usize = 24;
    const IT: u32 = 80;
    for &(live, commit, pot, label) in &[
        (3u8, 7i32, 25i32, "live-3"),
        (4u8, 7i32, 29i32, "live-4"),
    ] {
        let tree = seam_tree(live, commit, pot);
        let post = post_fold_nodes(&tree);
        let n_pf = post.iter().filter(|&&b| b).count();
        let n_dec = (0..tree.nodes.len()).filter(|&i| tree.nodes[i].is_player()).count();
        let base = exact_expl(&tree, live, NH, IT, pot);
        let full = region_swap_expl(&tree, live, NH, IT, pot, None);
        let fold_to_exact = region_swap_expl(&tree, live, NH, IT, pot,
            Some(&|i| post[i]));
        let nonfold_to_exact = region_swap_expl(&tree, live, NH, IT, pot,
            Some(&|i| !post[i]));
        eprintln!(
            "{label}: {n_pf}/{n_dec} decision nodes post-fold | baseline {base:.2}% | \
             B=8 full {full:.2}% | B=8(fold→exact) {fold_to_exact:.2}% | \
             B=8(nonfold→exact) {nonfold_to_exact:.2}%"
        );
        eprintln!(
            "  → if (fold→exact) ≈ baseline: FOLD REGION is the culprit (correctness/slice-2). \
             if (nonfold→exact) stays high: confirms. if (fold→exact) stays high: spread (budget)."
        );
    }
}

/// B-curve: with the fold-node hypothesis REJECTED by the region-swap
/// (exploitability is spread, not fold-concentrated), a falling curve
/// now cleanly means GRANULARITY (budget: raise B), not a fold bug.
#[test]
#[ignore = "B-curve; --ignored --nocapture --release"]
fn keystone_b_curve_live3() {
    const NH: usize = 32;
    const IT: u32 = 100;
    let tree = seam_tree(3, 7, 25);
    let pot = 25;
    let base = exact_expl(&tree, 3, NH, IT, pot);
    let ident = identity_lifted_expl(&tree, 3, NH, IT, pot);
    eprintln!("live-3 nh={NH}: baseline {base:.2}% | identity-lift {ident:.2}%");
    for &nb in &[8usize, 12, 16, 24] {
        let e = lifted_expl(&tree, 3, NH, nb, IT, pot);
        eprintln!("  B={nb}: lifted {e:.2}% pot ({:.1} hands/bucket)", NH as f32 / nb as f32);
    }
}

// ════════════════════════════════════════════════════════════════════
// CONFOUND CONTROL (2026-06-13): the fixed board Th9d8c + Jd + Qc is a
// T-9-8-J-Q BOARD STRAIGHT runout (~903/1081 hands chop), where
// strength-ordering is degenerate and quantile bucketing is meaningless
// ON THAT RUNOUT. Could inflate the cliff. Re-run the B-curve on a CLEAN
// board (no straight/flush on any runout) before trusting the cliff or
// running GS14 (which would inherit the same poisoned board).

fn clean_table(np: u8, nh: usize) -> FlopChanceTable {
    // Ks 7d 2c dry rainbow; turns 9h/4s; rivers chosen so no runout makes
    // a straight or 3-flush.
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

/// Solve+lift+score with a caller-supplied table builder + bucketing.
fn lifted_expl_tbl(
    tree: &FlatTree, mk: &dyn Fn() -> FlopChanceTable, nb: Option<usize>, iters: u32, pot: i32,
) -> f32 {
    let table = mk();
    let bk = match nb {
        Some(b) => { let (fm, tm, rm) = quantile_maps(&table, b); FlopBucketing::from_maps(&table, b, b, b, fm, tm, rm) }
        None => FlopBucketing::identity(&table),
    };
    let game = FlopStartGame::new(table);
    let mut bucketed = BucketedFlopCfr::new(tree, game.table(), &bk);
    bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
    bucketed.run(tree, &game, &bk, iters);
    let gs = FlopStartGame::new(mk());
    let mut scorer = FlopStartVectorCfr::new(tree, gs.table());
    lift_cum_to_exact(tree, &bucketed, &bk, &mut scorer);
    expl_pct(&scorer, tree, &gs, pot)
}
fn exact_expl_tbl(tree: &FlatTree, mk: &dyn Fn() -> FlopChanceTable, iters: u32, pot: i32) -> f32 {
    let game = FlopStartGame::new(mk());
    let mut exact = FlopStartVectorCfr::new(tree, game.table());
    exact.run(tree, &game, iters);
    expl_pct(&exact, tree, &game, pot)
}

#[test]
#[ignore = "clean-board B-curve control; --ignored --nocapture --release"]
fn keystone_b_curve_clean_board() {
    const NH: usize = 32;
    const IT: u32 = 100;
    let tree = seam_tree(3, 7, 25);
    let pot = 25;
    let mk = |np: u8, nh: usize| move || clean_table(np, nh);
    let mk3 = mk(3, NH);
    let base = exact_expl_tbl(&tree, &mk3, IT, pot);
    let ident = lifted_expl_tbl(&tree, &mk3, None, IT, pot);
    eprintln!("CLEAN BOARD Ks7d2c live-3 nh={NH}: baseline {base:.2}% | identity-lift {ident:.2}%");
    for &nb in &[8usize, 12, 16, 24] {
        let e = lifted_expl_tbl(&tree, &mk3, Some(nb), IT, pot);
        eprintln!("  B={nb}: lifted {e:.2}% ({:.1} h/bucket)", NH as f32 / nb as f32);
    }
}

/// GS14 (potential-aware EMD-kmeans) vs quantile on the CLEAN board,
/// live-3, across seeds (GS14 fit-at-research-scale is seed noise per
/// the banked b4_sweep finding, so report the distribution). The
/// discriminator: does the potential-aware COORDINATE beat strength-rank
/// quantile where quantile is poor?
#[test]
#[ignore = "GS14 vs quantile clean-board; --ignored --nocapture --release"]
fn keystone_gs14_vs_quantile_clean() {
    use solver_core::abstraction::postflop_buckets::build_postflop_bucketing_for_hands;
    use solver_core::solver::bucketed_flop_cfr::NO_BUCKET as NOB;
    const NH: usize = 32;
    const IT: u32 = 100;
    let tree = seam_tree(3, 7, 25);
    let pot = 25;
    let mk = || clean_table(3, NH);

    let base = exact_expl_tbl(&tree, &mk, IT, pot);
    eprintln!("CLEAN live-3 nh={NH}: baseline {base:.2}%");

    let board: [Card; 3] = ["Ks", "7d", "2c"].map(|s| card_from_str(s).unwrap());
    let turn_cards: Vec<Card> = ["9h", "4s"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let rivers: Vec<Vec<Card>> = [["Qd", "3c"], ["Td", "Js"]].iter()
        .map(|rs| rs.iter().map(|s| card_from_str(s).unwrap()).collect()).collect();

    for &nb in &[8usize, 16] {
        let q = lifted_expl_tbl(&tree, &mk, Some(nb), IT, pot);
        let mut gs_vals = Vec::new();
        for seed in [1u64, 7, 42, 99] {
            let mk_bk = {
                let board = board; let turn_cards = turn_cards.clone(); let rivers = rivers.clone();
                move |table: &FlopChanceTable| {
                    let nh = table.num_valid;
                    let hands: Vec<(Card, Card)> = (0..nh)
                        .map(|h| (table.hand_cards[h*2], table.hand_cards[h*2+1])).collect();
                    let pb = build_postflop_bucketing_for_hands(
                        hands.clone(), &board, &turn_cards, &rivers, nb, nb, nb, 10, seed);
                    assert_eq!(pb.hands, hands);
                    FlopBucketing::from_maps(table, nb, nb, nb,
                        pb.flop_map.clone(), pb.turn_map.clone(), pb.river_map.clone())
                }
            };
            // solve+lift+score with GS14 bk
            let table = mk();
            let bk = mk_bk(&table);
            let game = FlopStartGame::new(table);
            let mut b = BucketedFlopCfr::new(&tree, game.table(), &bk);
            b.set_terminal_design(TerminalDesign::Design1Collapsed);
            b.run(&tree, &game, &bk, IT);
            let gs = FlopStartGame::new(mk());
            let mut sc = FlopStartVectorCfr::new(&tree, gs.table());
            lift_cum_to_exact(&tree, &b, &bk, &mut sc);
            gs_vals.push(expl_pct(&sc, &tree, &gs, pot));
            let _ = NOB;
        }
        let gmin = gs_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let gmax = gs_vals.iter().cloned().fold(0.0f32, f32::max);
        let gmean = gs_vals.iter().sum::<f32>() / gs_vals.len() as f32;
        eprintln!(
            "B={nb}: QUANTILE {q:.2}% | GS14 seeds {:?} → min {gmin:.2}% mean {gmean:.2}% max {gmax:.2}%",
            gs_vals.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>()
        );
    }
    eprintln!("→ GS14 min << quantile ⇒ coordinate fix exists; GS14 ≈ quantile ⇒ deeper (budget/more buckets).");
}

/// GS14 lifted exploitability on the clean board at (nh, nb, seed).
fn gs14_lifted_clean(tree: &FlatTree, np: u8, nh: usize, nb: usize, seed: u64, iters: u32, pot: i32) -> f32 {
    use solver_core::abstraction::postflop_buckets::build_postflop_bucketing_for_hands;
    let board: [Card; 3] = ["Ks", "7d", "2c"].map(|s| card_from_str(s).unwrap());
    let turn_cards: Vec<Card> = ["9h", "4s"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let rivers: Vec<Vec<Card>> = [["Qd", "3c"], ["Td", "Js"]].iter()
        .map(|rs| rs.iter().map(|s| card_from_str(s).unwrap()).collect()).collect();
    let table = clean_table(np, nh);
    let hands: Vec<(Card, Card)> = (0..table.num_valid)
        .map(|h| (table.hand_cards[h*2], table.hand_cards[h*2+1])).collect();
    let pb = build_postflop_bucketing_for_hands(
        hands.clone(), &board, &turn_cards, &rivers, nb, nb, nb, 10, seed);
    assert_eq!(pb.hands, hands);
    let bk = FlopBucketing::from_maps(&table, nb, nb, nb,
        pb.flop_map.clone(), pb.turn_map.clone(), pb.river_map.clone());
    let game = FlopStartGame::new(table);
    let mut b = BucketedFlopCfr::new(tree, game.table(), &bk);
    b.set_terminal_design(TerminalDesign::Design1Collapsed);
    b.run(tree, &game, &bk, iters);
    let gs = FlopStartGame::new(clean_table(np, nh));
    let mut sc = FlopStartVectorCfr::new(tree, gs.table());
    lift_cum_to_exact(tree, &b, &bk, &mut sc);
    expl_pct(&sc, tree, &gs, pot)
}

/// COMPRESSION-SCALING: fix nb=8, increase nh → increase compression
/// ratio (nh/8) toward production-like. If GS14's EDGE over quantile
/// GROWS with compression, it's a production answer (potential-aware
/// preserves info whose loss matters more under heavier compression);
/// if the edge is flat, it's a research-scale artifact. The available
/// proxy for the unmeasurable production (nh=1176, 147:1) question.
#[test]
#[ignore = "compression-scaling GS14 vs quantile; --ignored --nocapture --release"]
fn keystone_compression_scaling() {
    const NB: usize = 8;
    const IT: u32 = 100;
    let tree = seam_tree(3, 7, 25);
    let pot = 25;
    eprintln!("\n═══ COMPRESSION SCALING (clean board, np=3, nb={NB}) ═══");
    eprintln!("ratio | baseline | quantile | GS14(min over seeds) | gap(q−g) | ratio(q/g)");
    for &nh in &[16usize, 24, 32, 48] {
        let mk = move || clean_table(3, nh);
        let base = exact_expl_tbl(&tree, &mk, IT, pot);
        let q = lifted_expl_tbl(&tree, &mk, Some(NB), IT, pot);
        let g = [7u64, 42].iter().map(|&s| gs14_lifted_clean(&tree, 3, nh, NB, s, IT, pot))
            .fold(f32::INFINITY, f32::min);
        eprintln!(
            "{:>4}:1 | {base:7.2}% | {q:7.2}% | {g:7.2}% | {:+6.2} | {:.2}×",
            nh / NB, q - g, q / g.max(1e-3)
        );
    }
    eprintln!("→ gap/ratio GROWS with compression ⇒ GS14 is the production answer; FLAT ⇒ research artifact.");
}

/// COMPRESSION-SCALING v2 (2026-06-13): the v1 confounded compression
/// with subset-resampling (each nh = a different hand set; the
/// non-monotonic baseline 0.97→0.58 was the tell). FIX: hold the game
/// FIXED (one nh), increase compression by LOWERING nb. Same game,
/// constant baseline, only the ratio varies — toward production-like.
#[test]
#[ignore = "compression-scaling v2 (fixed game); --ignored --nocapture --release"]
fn keystone_compression_scaling_v2() {
    const NH: usize = 48;
    const IT: u32 = 120;
    let tree = seam_tree(3, 7, 25);
    let pot = 25;
    let mk = || clean_table(3, NH);
    let base = exact_expl_tbl(&tree, &mk, IT, pot);
    eprintln!("\n═══ COMPRESSION SCALING v2 (clean board, np=3, FIXED nh={NH}) ═══");
    eprintln!("baseline (constant, same game): {base:.2}%");
    eprintln!("ratio | nb | quantile | GS14(min/mean over 3 seeds) | gap(q−g)");
    for &nb in &[24usize, 16, 8, 4, 2] {
        let q = lifted_expl_tbl(&tree, &mk, Some(nb), IT, pot);
        let gv: Vec<f32> = [7u64, 42, 123].iter()
            .map(|&s| gs14_lifted_clean(&tree, 3, NH, nb, s, IT, pot)).collect();
        let gmin = gv.iter().cloned().fold(f32::INFINITY, f32::min);
        let gmean = gv.iter().sum::<f32>() / gv.len() as f32;
        eprintln!("{:>3}:1 | {nb:2} | {q:7.2}% | {gmin:6.2}/{gmean:6.2}% | {:+6.2}", NH / nb, q - gmin);
    }
    eprintln!("→ gap GROWS as ratio rises (nb falls) ⇒ GS14 edge is real at scale; FLAT/noisy ⇒ artifact.");
}

// ════════════════════════════════════════════════════════════════════
// CONVERGENCE MAP (2026-06-13): the S1-multiway toy revealed np=3 CFR
// convergence is GAME-DEPENDENT — blueprint-quality and search-tolerance
// are only well-defined where the reference CONVERGES. So MAP it across
// the real multiway seam families × textures on the converging machinery
// (FlopStartVectorCfr, exact). This gates S2 (tolerance is only askable
// where exploitability converges), informs S3 (non-converging spots =
// least-predictable live search), and is a direct blueprint-decision
// input (non-converging spots are a SEPARATE bucket — maybe "play rough,
// nothing converges there anyway"). The toy thrashed (0.17→0.27→0.24);
// do the real games converge?
#[test]
#[ignore = "convergence map (real machinery, minutes); --ignored --nocapture --release"]
fn convergence_map_multiway() {
    // (live, commit, pot, nh) — nh kept so exact exploitability (O(nh^np))
    // stays feasible (np=4 at nh=16 = 65k; np=3 at nh=24 = 14k).
    let fams: &[(u8, i32, i32, usize)] = &[(3, 7, 25, 18), (4, 7, 29, 12)];
    let textures: &[(&str, fn(u8, usize) -> FlopChanceTable)] =
        &[("dry ", clean_table), ("wet ", research_table_tbl)];
    let pts = [34u32, 120, 400];
    eprintln!("\n═══ CONVERGENCE MAP — exact exploitability vs iters (real seam games) ═══");
    for &(live, commit, pot, nh) in fams {
        let tree = seam_tree(live, commit, pot);
        for &(tex, mk) in textures {
            let mk_box = move || mk(live, nh);
            let mut series = Vec::new();
            for it in pts {
                let e = exact_expl_tbl(&tree, &mk_box, it, pot);
                eprintln!("  ROW live {live} {tex} @ {it:>3} iters: expl {e:.4}");
                series.push(e);
            }
            // CONVERGES = monotone DECREASING (the real distinction from
            // the toy, which bounced UP 0.17→0.27→0.24). Threshold-on-tail
            // alone mislabels a still-falling series; trend is the signal.
            let monotone = series.windows(2).all(|w| w[1] <= w[0] + 1e-4);
            let tail = series[series.len() - 1];
            let verdict = if monotone && tail < 0.15 { "CONVERGES (monotone↓)" }
                else if monotone { "converging (slow)" } else { "THRASHES (non-monotone)" };
            eprintln!("  >> live {live} {tex}: {:.3}→{:.3}→{:.3} → {verdict}",
                series[0], series[1], series[2]);
        }
    }
    eprintln!("MAP: CONVERGES ⇒ S2 tolerance askable there; THRASHES ⇒ separate bucket \
        (no clean reference — 'play rough, nothing converges there anyway').");
}

// research_table returns (table, board); adapt to the (np,nh)->table shape.
fn research_table_tbl(np: u8, nh: usize) -> FlopChanceTable { research_table(np, nh).0 }
