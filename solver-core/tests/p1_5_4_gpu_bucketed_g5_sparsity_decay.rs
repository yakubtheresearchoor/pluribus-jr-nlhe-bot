//! G5-4 sparsity measurement: per-iteration NATIVE GPU cost over a
//! full 34-iter solve. Built to evaluate a compaction lever
//! (nonzero-bucket lists + any-empty early-out in the terminal
//! odometers); the lever was built, gated bit-identical (all 7 gates,
//! drift numbers unchanged), measured against a same-day control —
//! and REVERTED on the control's verdict.
//!
//! ═══ MEASURED 2026-06-11 (quiet box, release) ═══
//!
//!   PRODUCTION KERNEL (the control — this is what's in tree):
//!   B=8  4×4: 34-avg 0.424 s/iter (it1 1.39 → it34 0.36) → 7.0h set
//!   B=10 2×2: 34-avg 0.990 s/iter (it1 2.15 → it34 0.95) → 16.4h
//!   B=10 4×4: 34-avg 1.274 s/iter (it1 5.90 → it34 0.87) → 21.1h
//!
//!   +COMPACTION (reverted): 0.467 / 1.091 / 1.380 — a flat ~9%
//!   REGRESSION at every iteration (threadgroup-list indirection +
//!   the depth-0 stripe filter scanning out-of-range entries), zero
//!   sparsity gain.
//!
//!   THE FINDING: the odometer's `r == 0 → continue` was ALREADY
//!   skipping zero-reach subtrees — the G4 ladder's "GPU cost is
//!   flat" framing was a dense-iteration measurement artifact (it
//!   measured iters 2-4 only; it2-4 here ≈ 0.76 reproduces the G4
//!   0.72 row). The GPU decays like the CPU does, and the 34-avg
//!   re-prices every cell DOWN ~40-60%:
//!     B=8 4×4  12.0h → 7.0h  (GPU now beats CPU's 12.4h outright);
//!     B=10 2×2 22.0h → 16.4h;
//!     B=10 4×4 49.2h → 21.1h (the B=10 reference-fidelity challenger
//!     FITS 24h single-stream).
//!   GPU+CPU split at B=8 4×4: 1/7.0 + 1/12.4 ⇒ ≈ 4.5h full set.

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
            max_bets_per_street: None,
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

#[test]
#[ignore = "G5-4 sparsity decay: 34-iter per-iter curves; run with --ignored --nocapture --release"]
fn g5_sparsity_decay_curves() {
    eprintln!("\n════ G5-4 post-compaction decay curves (quiet box) ════");
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_m2_tree();
    for (nb, nt, nr) in [(8usize, 4usize, 4usize), (10, 2, 2), (10, 4, 4)] {
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

        let mut per_iter = Vec::with_capacity(34);
        let mut total_busy0 = native.gpu_busy_seconds();
        let mut total_wall = 0.0f64;
        for _ in 0..34u32 {
            let t0 = Instant::now();
            native.run(1);
            let w = t0.elapsed().as_secs_f64();
            let b = native.gpu_busy_seconds() - total_busy0;
            total_busy0 = native.gpu_busy_seconds();
            total_wall += w;
            per_iter.push((w, b));
        }
        let avg = total_wall / 34.0;
        let full_1755_h = avg * 34.0 * 1755.0 / 3600.0;
        let curve: Vec<String> = per_iter
            .iter()
            .enumerate()
            .filter(|(i, _)| [0usize, 1, 2, 4, 8, 16, 24, 33].contains(i))
            .map(|(i, (w, b))| format!("it{} {w:.2}/{b:.2}", i + 1))
            .collect();
        eprintln!(
            "NATIVE B={nb} {nt}×{nr}: 34-avg {avg:.3}s/iter | \
             decay {:.1}× (it1 {:.2} → it34 {:.2}) | full set ≈ {full_1755_h:.1}h\n\
             curve (wall/busy): {}",
            per_iter[0].0 / per_iter[33].0.max(1e-9),
            per_iter[0].0,
            per_iter[33].0,
            curve.join(" | ")
        );
    }
}
