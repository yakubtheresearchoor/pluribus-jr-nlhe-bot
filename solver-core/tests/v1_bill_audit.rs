//! V1 BILL AUDIT (2026-06-12, user directive: "whenever something said
//! 1200 hours, it was usually a bug"). The bill (v1_postflop_bill) is
//! MODEL arithmetic: per-iter anchors × 34 iters × 1755 flops, scaled
//! linear-in-nodes. Three suspected biases, all with project history:
//!
//!   1. DENSE-ITERATION OVERPRICING: anchors were measured at iters
//!      1-5 from cold start; the G5 sparsity-decay lesson showed dense
//!      early iters overprice the 34-iter average (~1.7× on the old
//!      tree — the 12.0h→7.0h repricing).
//!   2. UNPRICED OVERHEAD: the bill is iteration-only. Table builds,
//!      quantile maps, solver/GPU construction, readback are omitted —
//!      at 125 buckets × 1755 flops = 219k pairs, 1s/pair = 61h.
//!   3. 34-ITER CONVENTION: validated on the old wrong-game tree;
//!      unrevalidated on fold-continuation trees.
//!
//! This audit runs an END-TO-END MINI-ROW — one real bucket × N real
//! flops × 34 iters, wall-clock INCLUDING table+maps+construction —
//! and decomposes per-flop cost into (overhead, iteration) parts plus
//! the per-iter decay curve. The honest bill correction = measured
//! mini-row extrapolated vs the bill's model prediction.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, Card};
use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use std::time::Instant;

const NB: usize = 8;

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

/// 4×4 runout policy for an arbitrary flop (the pricing convention's
/// deck positions).
fn table_for_flop(flop: [Card; 3], np: u8) -> FlopChanceTable {
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let turn_cards: Vec<u8> = [6usize, 18, 30, 42].iter().map(|&p| deck[p]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let rdeck: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        river_decks[tc as usize] = [8usize, 20, 32, 44].iter().map(|&p| rdeck[p]).collect();
    }
    FlopChanceTable::build_full_nh_sampled(flop, np, &turn_cards, &river_decks)
}

#[test]
#[ignore = "bill audit mini-row (~20-40 min); --ignored --nocapture --release --features metal"]
fn v1_bill_audit_mini_row() {
    let ctx = MetalContext::new().expect("Metal");
    let spec = production_game_v1();
    let flop_bets =
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };

    // The audited bucket: live-4 raised (the bill prices its row at
    // 1.6h GPU from the 0.099 s/iter anchor). Tree is bucket-constant.
    let (live, commit, pot) = (4u8, 7i32, 29i32);
    let cfg = spec.flop_seam_config(live, commit, pot, flop_bets);
    let tree = build_tree(&cfg).expect("seam tree");
    eprintln!("bucket: live-{live} c{commit} p{pot}, {} nodes", tree.nodes.len());

    // N real flops (varied textures).
    let flops: Vec<[Card; 3]> = [
        ["2h", "7d", "Ks"],
        ["Ah", "Kh", "Qh"],
        ["9c", "9d", "2s"],
        ["6h", "7h", "8s"],
        ["As", "2d", "7c"],
        ["Td", "Jd", "Qc"],
        ["3c", "3d", "3h"],
        ["Kc", "8h", "2d"],
        ["5s", "6d", "9h"],
        ["Qd", "Td", "4s"],
    ]
    .iter()
    .map(|f| {
        [
            card_from_str(f[0]).unwrap(),
            card_from_str(f[1]).unwrap(),
            card_from_str(f[2]).unwrap(),
        ]
    })
    .collect();

    const ITERS: u32 = 34;
    let (mut t_table, mut t_maps, mut t_build, mut t_iter, mut busy) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut first_flop_curve: Vec<f64> = Vec::new();
    for (fi, &flop) in flops.iter().enumerate() {
        let t0 = Instant::now();
        let table = table_for_flop(flop, live);
        t_table += t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let (fm, tm, rm) = quantile_maps(&table, NB);
        t_maps += t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let game = FlopStartGame::new(table);
        let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
        let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
        solver.set_terminal_design(TerminalDesign::Design1Collapsed);
        let mut native =
            BucketedNativeGpu::new(&ctx, &tree, game.table(), &bk, &solver, (32 / NB) as u32)
                .expect("native");
        t_build += t0.elapsed().as_secs_f64();

        let b0 = native.gpu_busy_seconds();
        let t0 = Instant::now();
        if fi == 0 {
            // Per-iter decay curve on the first flop (the dense-iter
            // bias measurement).
            for _ in 0..ITERS {
                let ti = Instant::now();
                native.run(1);
                first_flop_curve.push(ti.elapsed().as_secs_f64());
            }
        } else {
            native.run(ITERS);
        }
        t_iter += t0.elapsed().as_secs_f64();
        busy += native.gpu_busy_seconds() - b0;
    }

    let n = flops.len() as f64;
    let per_flop_overhead = (t_table + t_maps + t_build) / n;
    let per_flop_iter = t_iter / n;
    eprintln!(
        "per-flop decomposition over {n} flops: table {:.2}s + maps {:.2}s + \
         construct {:.2}s = OVERHEAD {per_flop_overhead:.2}s | 34 iters {per_flop_iter:.2}s \
         (busy {:.0}%)",
        t_table / n,
        t_maps / n,
        t_build / n,
        100.0 * busy / t_iter
    );
    let head: f64 = first_flop_curve.iter().take(5).sum::<f64>() / 5.0;
    let tail: f64 =
        first_flop_curve.iter().skip(24).sum::<f64>() / (first_flop_curve.len() - 24) as f64;
    eprintln!(
        "decay curve (flop 0): iters 1-5 avg {head:.3}s | iters 25-34 avg {tail:.3}s | \
         dense/converged ratio {:.2}",
        head / tail
    );

    // Bill comparison.
    let bill_row_h = 0.099 * 34.0 * 1755.0 / 3600.0; // the bill's live-4 anchor row
    let measured_row_h = (per_flop_overhead + per_flop_iter) * 1755.0 / 3600.0;
    eprintln!(
        "BILL says this bucket-row = {bill_row_h:.2}h | MEASURED end-to-end \
         extrapolation = {measured_row_h:.2}h | correction ×{:.2}",
        measured_row_h / bill_row_h
    );
    eprintln!(
        "(overhead share of true cost: {:.0}% — the bill omitted it entirely)",
        100.0 * per_flop_overhead / (per_flop_overhead + per_flop_iter)
    );
}

/// Spot checks on the bill's two remaining exposure points:
///   A. live-6 (the 1,031h dominator): overhead share at 45k nodes +
///      price flop-independence (the anchor was one flop).
///   B. dispatch floor: GPU s/iter on the SMALLEST live-3 bucket tree —
///      the bill's linear-in-nodes model underestimates below the
///      fixed per-pass dispatch cost.
#[test]
#[ignore = "bill audit spot checks (~15 min); --ignored --nocapture --release --features metal"]
fn v1_bill_audit_spot_checks() {
    let ctx = MetalContext::new().expect("Metal");
    let spec = production_game_v1();
    let flop_bets =
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };

    // A. live-6 limp on TWO flops (neither is required to match the
    // anchor flop "2h7dKs" exactly — flop-independence check).
    let cfg = spec.flop_seam_config(6, 2, 12, flop_bets.clone());
    let tree = build_tree(&cfg).expect("tree");
    for f in [["Ah", "Kh", "Qh"], ["9c", "9d", "2s"]] {
        let flop = [
            card_from_str(f[0]).unwrap(),
            card_from_str(f[1]).unwrap(),
            card_from_str(f[2]).unwrap(),
        ];
        let t0 = Instant::now();
        let table = table_for_flop(flop, 6);
        let (fm, tm, rm) = quantile_maps(&table, NB);
        let game = FlopStartGame::new(table);
        let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
        let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
        solver.set_terminal_design(TerminalDesign::Design1Collapsed);
        let mut native =
            BucketedNativeGpu::new(&ctx, &tree, game.table(), &bk, &solver, (32 / NB) as u32)
                .expect("native (shared mode)");
        let overhead = t0.elapsed().as_secs_f64();
        native.run(1); // warm
        let t0 = Instant::now();
        native.run(3);
        let per_iter = t0.elapsed().as_secs_f64() / 3.0;
        eprintln!(
            "live-6 limp @ {f:?}: overhead {overhead:.1}s | {per_iter:.2}s/iter \
             (anchor 10.25; overhead share of a 34-iter flop: {:.1}%)",
            100.0 * overhead / (overhead + per_iter * 34.0)
        );
    }

    // B. dispatch floor: smallest live-3 cell (deep commit → tiny tree).
    // Find a small non-allin live-3 cell: commit 80, pot 244 (SPR 0.49).
    let cfg = spec.flop_seam_config(3, 80, 244, flop_bets);
    let tree = build_tree(&cfg).expect("tree");
    let flop = [
        card_from_str("2h").unwrap(),
        card_from_str("7d").unwrap(),
        card_from_str("Ks").unwrap(),
    ];
    let table = table_for_flop(flop, 3);
    let (fm, tm, rm) = quantile_maps(&table, NB);
    let game = FlopStartGame::new(table);
    let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
    let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
    solver.set_terminal_design(TerminalDesign::Design1Collapsed);
    let mut native =
        BucketedNativeGpu::new(&ctx, &tree, game.table(), &bk, &solver, (32 / NB) as u32)
            .expect("native");
    native.run(1);
    let t0 = Instant::now();
    native.run(10);
    let per_iter = t0.elapsed().as_secs_f64() / 10.0;
    let linear_model = 0.022 * tree.nodes.len() as f64 / 501.0;
    eprintln!(
        "DISPATCH FLOOR: live-3 c80 p244, {} nodes: {per_iter:.4}s/iter \
         (bill's linear model would say {linear_model:.4}) — floor ratio ×{:.1}",
        tree.nodes.len(),
        per_iter / linear_model
    );
}
