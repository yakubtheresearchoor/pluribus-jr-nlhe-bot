//! AT-SIZE PARITY GATE for the native GPU at the v1 seam-cell shapes
//! (2026-06-12, closing the validation gap the SIGSEGV opened): the
//! crash proved the native path had SIZE-DEPENDENT memory bugs, so
//! identity bit-exactness on the NH=6 fixture (N1 / N1-SHARED) is
//! necessary but NOT sufficient — it confirms the plumbing, not the
//! computation at production size. This gate runs GPU-native vs
//! bucketed-CPU (SAME B=8 quantile maps, SAME Design1Collapsed) at the
//! exact shapes the cell prices were measured on:
//!   live-4 raised  (2,443 nodes,  mega layout)
//!   live-5 limp    (14,753 nodes, mega layout)
//!   live-6 limp    (45,711 nodes, SHARED-buffer layout — the shape
//!                   that crashed)
//! nh = 1176 production hands, 4×4 runouts.
//!
//! Drift methodology = the G3 gate-2 standard: regrets + root cfv are
//! continuous accumulations and must sit at accumulated-rounding scale
//! (< 1e-3 of buffer scale); cum-strategy outliers beyond that must be
//! CERTIFIED knife-edges (regret matching is discontinuous at EPS —
//! every outlier's infoset must have max |regret| ≤ 2×EPS) and rare.

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

const NB: usize = 8;
const ITERS: u32 = 2;
const EPS: f32 = 1e-5; // the solver's regret-match epsilon

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

fn build_table(np: u8) -> FlopChanceTable {
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
    FlopChanceTable::build_full_nh_sampled(flop, np, &turn_cards, &river_decks)
}

fn cpu_buffers<'a>(s: &'a BucketedFlopCfr) -> [(&'static str, &'a [f32]); 6] {
    [
        ("regrets_flop", s.regrets_flop()),
        ("cum_flop", s.cum_strategy_flop()),
        ("regrets_turn", s.regrets_turn()),
        ("cum_turn", s.cum_strategy_turn()),
        ("regrets_river", s.regrets_river()),
        ("cum_river", s.cum_strategy_river()),
    ]
}

fn gpu_buffers<'a>(g: &'a BucketedNativeGpu) -> [(&'static str, &'a [f32]); 6] {
    [
        ("regrets_flop", g.regrets_flop()),
        ("cum_flop", g.cum_strategy_flop()),
        ("regrets_turn", g.regrets_turn()),
        ("cum_turn", g.cum_strategy_turn()),
        ("regrets_river", g.regrets_river()),
        ("cum_river", g.cum_strategy_river()),
    ]
}

fn run_cell_parity(ctx: &MetalContext, live: u8, commit: i32, pot: i32, label: &str) {
    use solver_core::tree::flat::MAX_NA_POSTFLOP;
    let spec = production_game_v1();
    let flop_bets =
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let cfg = spec.flop_seam_config(live, commit, pot, flop_bets);
    let tree = build_tree(&cfg).expect("seam tree");

    // GPU-native arm.
    let table_a = build_table(live);
    let (fm, tm, rm) = quantile_maps(&table_a, NB);
    let game_a = FlopStartGame::new(table_a);
    let bk_a =
        FlopBucketing::from_maps(game_a.table(), NB, NB, NB, fm.clone(), tm.clone(), rm.clone());
    let mut solver_a = BucketedFlopCfr::new(&tree, game_a.table(), &bk_a);
    solver_a.set_terminal_design(TerminalDesign::Design1Collapsed);
    let mut native =
        BucketedNativeGpu::new(ctx, &tree, game_a.table(), &bk_a, &solver_a, (32 / NB) as u32)
            .expect("native");
    let mode = if native.is_shared_mode() { "SHARED" } else { "mega" };
    let root_gpu = native.run(ITERS);

    // Bucketed-CPU arm (identical maps).
    let table_b = build_table(live);
    let game_b = FlopStartGame::new(table_b);
    let bk_b = FlopBucketing::from_maps(game_b.table(), NB, NB, NB, fm, tm, rm);
    let mut cpu = BucketedFlopCfr::new(&tree, game_b.table(), &bk_b);
    cpu.set_terminal_design(TerminalDesign::Design1Collapsed);
    let root_cpu = cpu.run(&tree, &game_b, &bk_b, ITERS);

    let scale = |xs: &[f32]| xs.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;

    let _ = &solver_a; // layout donor for the native constructor
    // Regrets + root: accumulated-rounding scale.
    let mut max_drift = 0.0f64;
    let bufs_g = gpu_buffers(&native);
    let bufs_c = cpu_buffers(&cpu);
    for ((lbl, ga), (_, ca)) in bufs_g.iter().zip(bufs_c.iter()) {
        assert_eq!(ga.len(), ca.len(), "{lbl} length");
        let s = scale(ca).max(1e-30);
        let d = ga
            .iter()
            .zip(ca.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).abs() / s)
            .fold(0.0, f64::max);
        eprintln!("  [{label}/{mode}] {lbl}: max rel drift {d:.2e}");
        if lbl.starts_with("regrets") && d > max_drift {
            max_drift = d;
        }
    }
    let root_d = root_gpu
        .iter()
        .zip(&root_cpu)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0, f64::max)
        / scale(&root_cpu).max(1e-30);
    eprintln!("  [{label}/{mode}] root cfv: max rel drift {root_d:.2e}");
    assert!(
        max_drift.max(root_d) < 1e-3,
        "[{label}] regret/root drift {max_drift:.2e}/{root_d:.2e} beyond \
         accumulated-rounding scale — kernel breakage AT SIZE"
    );

    // Cum: knife-edge certification (gate-2 standard).
    let mut outliers = 0usize;
    for (((lbl, ga), (_, ca)), (_, rc)) in bufs_g
        .iter()
        .zip(bufs_c.iter())
        .filter(|((l, _), _)| l.starts_with("cum"))
        .zip(bufs_c.iter().filter(|(l, _)| l.starts_with("regrets")))
    {
        let s = scale(ca).max(1e-30);
        for i in 0..ga.len() {
            let d = (ga[i] as f64 - ca[i] as f64).abs() / s;
            if d <= 1e-3 {
                continue;
            }
            outliers += 1;
            let block = i - (i % (MAX_NA_POSTFLOP * NB));
            let b = i % NB;
            let max_r = (0..MAX_NA_POSTFLOP)
                .map(|a| rc[block + a * NB + b].abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_r <= 2.0 * EPS,
                "[{label}] {lbl}[{i}] drift {d:.2e} at infoset with max |regret| \
                 {max_r:.2e} > 2×EPS — NOT a knife-edge: breakage AT SIZE"
            );
        }
    }
    eprintln!(
        "  [{label}/{mode}] PASS at size: {} nodes, nh {}, cum knife-edge outliers {outliers}",
        tree.nodes.len(),
        game_b.table().num_valid
    );
    assert!(outliers <= 32, "[{label}] {outliers} cum outliers — too many for knife-edges");
}

#[test]
#[ignore = "at-size parity, live-3 (np=3 scope extension); --ignored --nocapture --release --features metal"]
fn cell_parity_live3_mega() {
    let ctx = MetalContext::new().expect("Metal");
    run_cell_parity(&ctx, 3, 7, 29, "live-3 raised");
}

#[test]
#[ignore = "at-size parity, live-4 (~minutes); --ignored --nocapture --release --features metal"]
fn cell_parity_live4_mega() {
    let ctx = MetalContext::new().expect("Metal");
    run_cell_parity(&ctx, 4, 7, 29, "live-4 raised");
}

#[test]
#[ignore = "at-size parity, live-5 (~10 min CPU arm); --ignored --nocapture --release --features metal"]
fn cell_parity_live5_mega() {
    let ctx = MetalContext::new().expect("Metal");
    run_cell_parity(&ctx, 5, 2, 10, "live-5 limp");
}

#[test]
#[ignore = "at-size parity, live-6 SHARED (~2h CPU arm); --ignored --nocapture --release --features metal"]
fn cell_parity_live6_shared() {
    let ctx = MetalContext::new().expect("Metal");
    run_cell_parity(&ctx, 6, 2, 12, "live-6 limp");
}
