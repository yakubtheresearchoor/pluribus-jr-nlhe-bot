//! SHARED-MODE SEAM-SHAPE GATE (2026-06-12; was the failure
//! bisection). HISTORY: live-6 at-size parity failed with drift ~1.0
//! while N1-SHARED passed bit-exact on the g5 gate-tree fixture. This
//! bisection reproduced the failure at nh=6 IDENTITY on the seam
//! trees (so: tree shape, not nh/bucketing/stripes) and pinned the
//! bug — ONE shared accumulator served both the per-ti river chance
//! accumulation and the cross-ti turn accumulation, which overlap in
//! time on the same entries; the gate-tree fixture's shape happened
//! not to expose it. Fix: split acc_river / acc_turn. Post-fix all
//! three cases sit at striped-rounding scale (≤ 8.5e-7; stripes > 1
//! here, so bit-exactness is not expected — that's N1-SHARED's job at
//! stripes 1).
//!
//! STANDING ROLE: the cheap (~2 min) shared-mode gate on the tree
//! shapes that actually exposed the bug. Asserts drift < 1e-3.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const ITERS: u32 = 2;

fn build_small_table(np: u8, nh: usize) -> FlopChanceTable {
    let flop: [Card; 3] = [
        card_from_str("2h").unwrap(),
        card_from_str("7d").unwrap(),
        card_from_str("Ks").unwrap(),
    ];
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let turn_cards: Vec<u8> = [6usize, 18, 30, 42].iter().map(|&p| deck[p]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let rdeck: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        river_decks[tc as usize] = [8usize, 20, 32, 44].iter().map(|&p| rdeck[p]).collect();
    }
    // Subset table at small nh (board-conflict-free at the flop).
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
            continue;
        }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> =
        (0..np).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..np as usize {
        for &hi in &chosen {
            ranges[p][hi as usize] = 1.0;
        }
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &flop, &ranges, np, &chosen, &turn_cards, &river_decks,
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

/// Run forced-shared GPU vs CPU; return max rel drift across the six
/// buffers + root.
fn shared_vs_cpu_drift(
    ctx: &MetalContext,
    tree: &FlatTree,
    np: u8,
    nh: usize,
    identity: bool,
    nb: usize,
) -> f64 {
    let make_bk = |t: &FlopChanceTable| -> FlopBucketing {
        if identity {
            FlopBucketing::identity(t)
        } else {
            let (fm, tm, rm) = quantile_maps(t, nb);
            FlopBucketing::from_maps(t, nb, nb, nb, fm, tm, rm)
        }
    };
    let table_a = build_small_table(np, nh);
    let game_a = FlopStartGame::new(table_a);
    let bk_a = make_bk(game_a.table());
    let stripes = (32 / bk_a.nb_flop.max(bk_a.nb_turn).max(bk_a.nb_river)).max(1) as u32;
    let solver_a = {
        let mut s = BucketedFlopCfr::new(tree, game_a.table(), &bk_a);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        s
    };
    let mut native =
        BucketedNativeGpu::new_forced_shared(ctx, tree, game_a.table(), &bk_a, &solver_a, stripes)
            .expect("forced shared");
    let root_gpu = native.run(ITERS);

    let table_b = build_small_table(np, nh);
    let game_b = FlopStartGame::new(table_b);
    let bk_b = make_bk(game_b.table());
    let mut cpu = BucketedFlopCfr::new(tree, game_b.table(), &bk_b);
    cpu.set_terminal_design(TerminalDesign::Design1Collapsed);
    let root_cpu = cpu.run(tree, &game_b, &bk_b, ITERS);

    let scale = |xs: &[f32]| xs.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;
    let pairs: [(&str, &[f32], &[f32]); 6] = [
        ("regrets_flop", native.regrets_flop(), cpu.regrets_flop()),
        ("cum_flop", native.cum_strategy_flop(), cpu.cum_strategy_flop()),
        ("regrets_turn", native.regrets_turn(), cpu.regrets_turn()),
        ("cum_turn", native.cum_strategy_turn(), cpu.cum_strategy_turn()),
        ("regrets_river", native.regrets_river(), cpu.regrets_river()),
        ("cum_river", native.cum_strategy_river(), cpu.cum_strategy_river()),
    ];
    let mut max_d = 0.0f64;
    for (_, ga, ca) in pairs {
        let s = scale(ca).max(1e-30);
        for (a, b) in ga.iter().zip(ca.iter()) {
            let d = (*a as f64 - *b as f64).abs() / s;
            if d > max_d {
                max_d = d;
            }
        }
    }
    let rs = scale(&root_cpu).max(1e-30);
    for (a, b) in root_gpu.iter().zip(&root_cpu) {
        let d = (*a as f64 - *b as f64).abs() / rs;
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

#[test]
#[ignore = "bisection probe; --ignored --nocapture --release --features metal"]
fn shared_mode_bisect() {
    let ctx = MetalContext::new().expect("Metal");
    let spec = production_game_v1();
    let flop_bets =
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };

    // Case 1: live-6 seam TREE (the shape that exposed the acc bug),
    // identity maps, nh=6.
    let t6 = build_tree(&spec.flop_seam_config(6, 2, 12, flop_bets.clone())).unwrap();
    let d = shared_vs_cpu_drift(&ctx, &t6, 6, 6, true, 0);
    eprintln!("live-6 tree | identity nh=6: drift {d:.2e}");
    assert!(d < 1e-3, "shared-mode drift {d:.2e} on the live-6 seam shape");

    // Case 2: live-6 tree, quantile B=4 (NO_BUCKET + stripes>1), nh=24.
    let d = shared_vs_cpu_drift(&ctx, &t6, 6, 24, false, 4);
    eprintln!("live-6 tree | quantile B=4 nh=24: drift {d:.2e}");
    assert!(d < 1e-3, "shared-mode drift {d:.2e} (quantile case)");

    // Case 3: live-4 seam tree forced shared (mega is its production
    // mode; this keeps the shared path covered on a second shape).
    let t4 = build_tree(&spec.flop_seam_config(4, 7, 29, flop_bets)).unwrap();
    let d = shared_vs_cpu_drift(&ctx, &t4, 4, 24, false, 4);
    eprintln!("live-4 tree | quantile B=4 nh=24 (forced shared): drift {d:.2e}");
    assert!(d < 1e-3, "shared-mode drift {d:.2e} (live-4 forced shared)");
}
