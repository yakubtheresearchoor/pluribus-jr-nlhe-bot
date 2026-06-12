//! G5-4: the GPU ladder re-measured on the fully NATIVE path
//! (BucketedNativeGpu — zero CPU walk work, Fix-C async, resident
//! terminals), same cells as the G4 ladder (M2 shape, production
//! nh=1176, B ∈ {8, 10} × runouts {1×1, 2×2, 4×4}).
//!
//! Comparison rows (G4, batched HYBRID, measured 2026-06-11):
//!   B=8:  1×1 0.30 | 2×2 0.54 | 4×4 1.41   s/iter (busy 70/56/43%)
//!   B=10: 1×1 1.00 | 2×2 1.53 | 4×4 3.69   s/iter (busy 91/84/78%)
//!
//! G5-1 projection (the stop-and-report line is >2× off): eliminable
//! hybrid budget = reach prepass + packing + per-walk readback; B=10
//! 4×4 3.69 → ~2.9s floor from traffic alone, async compounding
//! beyond.
//!
//! ═══ MEASURED 2026-06-11 (quiet box, release) ═══
//!
//!   cell       | hybrid s/iter (busy) | NATIVE s/iter (busy) | gain
//!   B=8  1×1   | 0.30 (70%)           | 0.22 (100%)          | 1.36×
//!   B=8  2×2   | 0.54 (56%)           | 0.34 (100%)          | 1.59×
//!   B=8  4×4   | 1.41 (43%)           | 0.72 (100%)          | 1.96×
//!   B=10 1×1   | 1.00 (91%)           | 0.93 (100%)          | 1.08×
//!   B=10 2×2   | 1.53 (84%)           | 1.33 (100%)          | 1.15×
//!   B=10 4×4   | 3.69 (78%)           | 2.97 (100%)          | 1.24×
//!
//!   READINGS:
//!   - ON projection: B=10 4×4 measured 2.97 vs the ~2.9 floor (3%);
//!     no stop-and-report. Every cell lands at hybrid_s × busy% —
//!     the native path removed ALL non-GPU wall time; wall ≈ busy.
//!   - Wall = busy at N=1 ⇒ the G4 concurrency multiplier (~1.7×, an
//!     idle-gap effect) does NOT apply to the native path; these
//!     single-stream numbers are final for one chip.
//!   - 34×1755 single-stream: B=8 4×4 12.0h (parity with CPU-14-thread
//!     12.4h, CPU left 100% FREE); B=10 2×2 22.0h — the B=10
//!     challenger blueprint now fits 24h directly; B=10 4×4 49.2h.
//!   - GPU+CPU split at the chosen cell (B=8 4×4): combined rate
//!     1/12.0 + 1/12.4 ⇒ ≈ 6.1h full set — the production run halves.
//!   - Remaining headroom is GPU-work reduction only (sparsity
//!     compaction, function constants), not orchestration.
//!
//!   ERRATUM (same day — see p1_5_4_gpu_bucketed_g5_sparsity_decay):
//!   these rows are DENSE-ITERATION costs (iters 2-4), not flat
//!   per-iter costs. The native path decays 3.8-6.8× to a converged
//!   plateau by ~iter 5 (the odometer's r==0 skip always pruned).
//!   True 34-avg full-set hours: B=8 4×4 7.0h, B=10 2×2 16.4h,
//!   B=10 4×4 21.1h. The compaction lever itself measured a ~9%
//!   regression and was reverted.

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

/// Production-nh table at the given runout policy (deterministic deck
/// positions — the G4 ladder's exact convention).
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

#[test]
#[ignore = "G5-4 native ladder: B{8,10} × runouts{1×1,2×2,4×4}; run with --ignored --nocapture --release"]
fn g5_ladder_native() {
    eprintln!("\n════ G5-4 native ladder (fully GPU-resident, quiet box) ════");
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_m2_tree();
    for nb in [8usize, 10] {
        for (nt, nr) in [(1usize, 1usize), (2, 2), (4, 4)] {
            let table = build_table(nt, nr);
            let (fm, tm, rm) = quantile_maps(&table, nb);
            let game = FlopStartGame::new(table);
            let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
            let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
            solver.set_terminal_design(TerminalDesign::Design1Collapsed);
            let mut native = BucketedNativeGpu::new(
                &ctx, &tree, game.table(), &bk, &solver, (32 / nb) as u32,
            )
            .expect("native gpu");
            native.run(1); // warm (pipelines, first-touch, residency)
            let busy0 = native.gpu_busy_seconds();
            let t0 = Instant::now();
            native.run(3);
            let per_iter = t0.elapsed().as_secs_f64() / 3.0;
            let busy = (native.gpu_busy_seconds() - busy0) / 3.0;
            let full_1755_h = per_iter * 34.0 * 1755.0 / 3600.0;
            eprintln!(
                "NATIVE B={nb} {nt}×{nr}: {per_iter:.2}s/iter | busy {:.0}% | \
                 34×1755 single-stream ≈ {full_1755_h:.1}h",
                100.0 * busy / per_iter
            );
        }
    }
}
