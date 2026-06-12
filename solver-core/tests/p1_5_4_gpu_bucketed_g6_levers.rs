//! G6 lever measurements (implement → measure → keep or discard).
//! Each lever runs the SAME cells as the decay baseline (34-iter
//! curves, M2 shape, production nh) with the lever toggled on/off in
//! fresh solver instances.
//!
//! Lever 1: function-constant specialization of the batched terminal
//! kernel (FC_NB/NP/NUM_OPP/STRIPES → constant-folding + unrolling).
//! Bit-exact by construction (same ops, same order); identity gates
//! run through the specialized pipeline.
//!
//! ═══ MEASURED 2026-06-11 (quiet box, release, 34 iters/arm) ═══
//!   Lever 1 — KEPT. Generic arms reproduce the decay baseline
//!   (0.423 vs 0.424, 1.268 vs 1.274 — clean control):
//!   B=8  4×4: 0.423 → 0.355 s/iter (1.19×) — full set 7.0h → 5.9h
//!   B=10 4×4: 1.268 → 0.969 s/iter (1.31×) — full set 21.1h → 16.1h
//!   Gates: N1 + 1b bit-exact THROUGH the specialized pipeline
//!   (identity = uniform dims); gate 2 / N2 drift unchanged (N2
//!   divergent dims exercises the generic fallback).
//!
//! Lever 4 (cross-walk batched reach) — DISCARDED 2026-06-11. Two
//! multi-walk kernels (seed + reach level, per-walk offsets via
//! descriptor) cut the reach stage's dispatch count ~8× (≈170 →
//! ≈20/pass): 34-avg 0.354→0.350 (B=8 4×4) and 0.971→0.966 (B=10
//! 4×4) — 1.01×, noise. FINDING: the flat 0.083s reach stage is
//! BANDWIDTH-bound (mega-zero + per-edge row traffic is real work),
//! not dispatch-serialization-bound. Remaining reach headroom =
//! traffic reduction (zero-skip / row-sparse propagation), ceiling
//! ~15% of the B=8 cell, semantics-risky — not taken. Reverted.
//!
//! G6 CLOSE: kept = lever 1 only. Final 34-avg: B=8 4×4 0.355 s/iter
//! (full set 5.9h GPU-only; GPU+CPU split ≈ 4.0h), B=10 4×4 0.969
//! (16.1h). The native path is now within noise of its measured
//! structural floors on every axis we probed: odometer latency-bound
//! (FMA 1.00×), reach bandwidth-bound (batching 1.01×), enumeration
//! already sparse (compaction −9%), occupancy saturated (wall=busy).
//!
//! Lever 2 (FMA contraction) — DISCARDED 2026-06-11. Mechanism: the
//! batched kernel stamped twice in one TU via a body include, _fma
//! copy under `#pragma METAL fp contract(fast)`. Measured: contraction
//! ACTIVE (4148/105984 river-regret bits diverged from the exact arm
//! at 3 iters), trajectory gate clean (2.2e-6..9.3e-6, root 7.6e-7,
//! 8 iters vs CPU) — and gain 1.00×/1.01× at both cells. The odometer
//! is LATENCY/DIVERGENCE-bound, not FP-throughput-bound; contraction
//! buys nothing and costs the exact-arithmetic property. Reverted
//! (host plumbing + body include removed; this header is the record).

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, Card};
use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::time::Instant;

const NP: u8 = 6;

fn build_m2_tree() -> FlatTree {
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
    };
    build_tree(&config).unwrap()
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

fn build_table(n_turn: usize, n_river: usize) -> FlopChanceTable {
    let flop: [Card; 3] = [
        card_from_str("2h").unwrap(),
        card_from_str("7d").unwrap(),
        card_from_str("Ks").unwrap(),
    ];
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let turn_pos: &[usize] = match n_turn {
        1 => &[12],
        2 => &[12, 36],
        4 => &[6, 18, 30, 42],
        _ => unreachable!(),
    };
    let river_pos: &[usize] = match n_river {
        1 => &[10],
        2 => &[10, 30],
        4 => &[8, 20, 32, 44],
        _ => unreachable!(),
    };
    let turn_cards: Vec<u8> = turn_pos.iter().map(|&p| deck[p]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let rdeck: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        river_decks[tc as usize] = river_pos.iter().map(|&p| rdeck[p]).collect();
    }
    FlopChanceTable::build_full_nh_sampled(flop, NP, &turn_cards, &river_decks)
}

/// 34-iter run; returns (it2 wall, 34-avg wall).
fn run_cell(ctx: &MetalContext, tree: &FlatTree, nb: usize, nt: usize, nr: usize, specialized: bool) -> (f64, f64) {
    let table = build_table(nt, nr);
    let (fm, tm, rm) = quantile_maps(&table, nb);
    let game = FlopStartGame::new(table);
    let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
    let mut solver = BucketedFlopCfr::new(tree, game.table(), &bk);
    solver.set_terminal_design(TerminalDesign::Design1Collapsed);
    let mut native =
        BucketedNativeGpu::new(ctx, tree, game.table(), &bk, &solver, (32 / nb) as u32)
            .expect("native gpu");
    native.set_use_specialized(specialized);
    let mut it2 = 0.0f64;
    let mut total = 0.0f64;
    for i in 0..34u32 {
        let t0 = Instant::now();
        native.run(1);
        let w = t0.elapsed().as_secs_f64();
        total += w;
        if i == 1 {
            it2 = w;
        }
    }
    (it2, total / 34.0)
}

#[test]
#[ignore = "G6 lever 1 A/B: FC specialization on/off; run with --ignored --nocapture --release"]
fn g6_lever1_function_constants() {
    eprintln!("\n════ G6 lever 1: function-constant specialization (quiet box) ════");
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_m2_tree();
    for (nb, nt, nr) in [(8usize, 4usize, 4usize), (10, 4, 4)] {
        let (it2_g, avg_g) = run_cell(&ctx, &tree, nb, nt, nr, false);
        let (it2_s, avg_s) = run_cell(&ctx, &tree, nb, nt, nr, true);
        eprintln!(
            "B={nb} {nt}×{nr}: generic it2 {it2_g:.3}s / 34-avg {avg_g:.3}s  |  \
             SPECIALIZED it2 {it2_s:.3}s / 34-avg {avg_s:.3}s  |  \
             gain it2 {:.2}× / avg {:.2}×",
            it2_g / it2_s,
            avg_g / avg_s
        );
    }
}
