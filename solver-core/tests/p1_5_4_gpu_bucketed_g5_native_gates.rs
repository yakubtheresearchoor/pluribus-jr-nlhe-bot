//! G5-3 gates for the fully GPU-NATIVE bucketed iteration
//! (`BucketedNativeGpu`: strategies, reach, terminals, bottom-up cfv,
//! regret/cum updates, chance accumulation — all on-device, Fix-C
//! async, zero per-walk CPU work). Fourth link of the gate chain:
//!
//!   Gate N1 (load-bearing): UNSTRIPED config at B = nh identity maps
//!   — root cfv + all 6 persistent buffers bit-exact vs the pure-CPU
//!   bucketed walk across full DCFR iterations. The CPU walk is
//!   anchored to the exact evaluator; the hybrid GPU paths are
//!   anchored at G3/G4; this closes the native path onto the same
//!   chain.
//!
//!   Gate N2: STRIPED config at DIVERGENT per-street dims (4/3/5,
//!   rake + side-pot tree) through 8 DCFR iterations — drift vs the
//!   pure CPU walk pinned at accumulated-rounding scale (the
//!   established trajectory standard; uniform-B identity cannot
//!   exercise the stride divergence).
//!
//! ═══ MEASURED 2026-06-11 ═══
//!   Gate N1: bit-exact ✓ (root cfv + 6 persistent buffers, 3 iters,
//!     NH=6 identity maps, unstriped, every stage native: strategies,
//!     reach, batched resident terminals, cfv levels, regret/cum,
//!     chance accumulation, root accumulation).
//!   Gate N2: striped S=6 @ 4/3/5 divergent dims, 8 iters: per-buffer
//!     max rel drift 2.1e-6 .. 7.7e-5 (root 8.9e-7) — numerically the
//!     SAME drift profile as the hybrid G3 gate-4 run (2.1e-6 ..
//!     7.7e-5, root 8.9e-7): the native walk stages add zero drift
//!     beyond the already-characterized striped terminal. Bug line
//!     1e-3.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const NP: u8 = 6;

fn build_table(nh: usize) -> FlopChanceTable {
    let board: Vec<Card> =
        ["Th", "9d", "8c"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
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
        (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            ranges[p][hi as usize] = 1.0;
        }
    }
    let turn_cards: Vec<u8> =
        ["2c", "Jd"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let river_strs: [&[&str]; 2] = [&["4s", "7h"], &["3s", "Qc"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] =
            river_strs[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

/// The G3 gate-1 tree (no rake, deeper bets — identity fixture).
fn build_identity_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![500; NP as usize],
        initial_contributions: vec![5; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.33), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&config).unwrap()
}

/// The G3 gate-3/4 tree (rake + uneven contributions → fold and
/// side-pot terminals — the divergent-dims fixture).
fn build_divergent_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![200; NP as usize],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.05,
        rake_cap: 3.0,
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
    nb_f: usize,
    nb_t: usize,
    nb_r: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8], nb: usize| -> Vec<u16> {
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
    let flop_map = map_for(&base_pi, &[], nb_f);
    let mut turn_maps = Vec::new();
    let mut river_maps = Vec::new();
    for &tc_card in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
        turn_maps.push(map_for(pi, &[tc_card], nb_t));
        let mut rms = Vec::new();
        for &rc_card in &table.river_decks[tc_card as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
            rms.push(map_for(pi, &[tc_card, rc_card], nb_r));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
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

/// Gate N1: identity maps (B = nh), unstriped terminal, 3 DCFR
/// iterations — the fully native iteration must be bit-exact against
/// the pure-CPU bucketed walk on root cfv and all 6 persistent
/// buffers.
#[test]
fn gate_n1_native_identity_bit_exact() {
    const NH: usize = 6;
    const ITERS: u32 = 3;
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_identity_tree();

    // Native GPU arm.
    let game_a = FlopStartGame::new(build_table(NH));
    let bk_a = FlopBucketing::identity(game_a.table());
    let solver_a = BucketedFlopCfr::new(&tree, game_a.table(), &bk_a);
    let mut native =
        BucketedNativeGpu::new(&ctx, &tree, game_a.table(), &bk_a, &solver_a, 1)
            .expect("native gpu");
    let root_gpu = native.run(ITERS);

    // Pure-CPU arm.
    let game_b = FlopStartGame::new(build_table(NH));
    let bk_b = FlopBucketing::identity(game_b.table());
    let mut cpu = BucketedFlopCfr::new(&tree, game_b.table(), &bk_b);
    cpu.set_terminal_design(TerminalDesign::Design1Collapsed);
    let root_cpu = cpu.run(&tree, &game_b, &bk_b, ITERS);

    for (i, (a, b)) in root_gpu.iter().zip(&root_cpu).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "root cfv[{i}]: native {a} vs cpu {b}"
        );
    }
    for ((label, ga), (_, ca)) in gpu_buffers(&native).iter().zip(cpu_buffers(&cpu).iter()) {
        assert_eq!(ga.len(), ca.len(), "{label} length");
        for i in 0..ga.len() {
            assert_eq!(
                ga[i].to_bits(),
                ca[i].to_bits(),
                "{label}[{i}]: native {} vs cpu {} — the native iteration at \
                 singletons must coincide with the CPU graph element-for-element",
                ga[i],
                ca[i]
            );
        }
    }
    eprintln!(
        "gate N1 PASSED: fully native iteration bit-exact at identity \
         ({ITERS} iters, root + 6 buffers; gpu busy {:.3}s)",
        native.gpu_busy_seconds()
    );
}

/// Gate N2: divergent per-street dims (4/3/5) on the rake/side-pot
/// tree, striped terminal, 8 DCFR iterations — drift vs pure CPU at
/// accumulated-rounding scale (bug line 1e-3, the G3 gate-4 standard).
#[test]
fn gate_n2_native_divergent_dims_trajectory() {
    const NH: usize = 16;
    const ITERS: u32 = 8;
    const S: u32 = 6; // 32 / max(nb) = 32 / 5
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_divergent_tree();

    // Native GPU arm.
    let game_a = FlopStartGame::new(build_table(NH));
    let (fm, tm, rm) = quantile_maps(game_a.table(), 4, 3, 5);
    let bk_a = FlopBucketing::from_maps(game_a.table(), 4, 3, 5, fm, tm, rm);
    let solver_a = BucketedFlopCfr::new(&tree, game_a.table(), &bk_a);
    let mut native =
        BucketedNativeGpu::new(&ctx, &tree, game_a.table(), &bk_a, &solver_a, S)
            .expect("native gpu");
    let root_gpu = native.run(ITERS);

    // Pure-CPU arm.
    let game_b = FlopStartGame::new(build_table(NH));
    let (fm, tm, rm) = quantile_maps(game_b.table(), 4, 3, 5);
    let bk_b = FlopBucketing::from_maps(game_b.table(), 4, 3, 5, fm, tm, rm);
    let mut cpu = BucketedFlopCfr::new(&tree, game_b.table(), &bk_b);
    cpu.set_terminal_design(TerminalDesign::Design1Collapsed);
    let root_cpu = cpu.run(&tree, &game_b, &bk_b, ITERS);

    let scale = |xs: &[f32]| xs.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;
    let mut max_drift = 0.0f64;
    for ((label, ga), (_, ca)) in gpu_buffers(&native).iter().zip(cpu_buffers(&cpu).iter()) {
        assert_eq!(ga.len(), ca.len(), "{label} length");
        let s = scale(ca).max(1e-30);
        let d = ga
            .iter()
            .zip(ca.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).abs() / s)
            .fold(0.0, f64::max);
        eprintln!("gate N2 {label}: max rel drift {d:.2e}");
        max_drift = max_drift.max(d);
    }
    let root_d = root_gpu
        .iter()
        .zip(&root_cpu)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0, f64::max)
        / scale(&root_cpu).max(1e-30);
    eprintln!("gate N2 root: {root_d:.2e}");
    assert!(
        max_drift.max(root_d) < 1e-3,
        "native divergent-dims trajectory drift {max_drift:.2e} beyond \
         accumulated-rounding scale — breakage"
    );
}

/// Gate N3 (2026-06-14, cap-raise 16→32): the terminal kernel's
/// threadgroup arrays grew to 32×32; validate the NEW B>16 range is
/// still GPU==CPU within the N2 accumulated-rounding standard (1e-3).
/// Uniform B=32 on the np=6 divergent tree (S=1 since 32/32=1), nh=48
/// so every runout has ≥32 alive hands for the quantile maps.
///
/// DO NOT RUN AS-IS: this exact configuration HUNG THE GPU and crashed
/// the machine (2026-06-14). #[ignore]'d. A B>16 re-attempt must step up
/// cautiously from B20 with a TINY fixture and low iters — not jump to
/// the B32 occupancy cliff with nh=48/8-iter. Requires MAX_BUCKETS_GPU>16.
#[test]
#[ignore = "crashed the GPU at B32/nh48/8it — re-attempt cautiously from B20, tiny fixture"]
fn gate_n3_native_b32_drift() {
    const NH: usize = 48;
    const ITERS: u32 = 8;
    const B: usize = 32;
    const S: u32 = 1; // 32 / max(nb)=32
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_divergent_tree();

    let game_a = FlopStartGame::new(build_table(NH));
    let (fm, tm, rm) = quantile_maps(game_a.table(), B, B, B);
    let bk_a = FlopBucketing::from_maps(game_a.table(), B, B, B, fm, tm, rm);
    let solver_a = BucketedFlopCfr::new(&tree, game_a.table(), &bk_a);
    let mut native =
        BucketedNativeGpu::new(&ctx, &tree, game_a.table(), &bk_a, &solver_a, S)
            .expect("native gpu");
    let root_gpu = native.run(ITERS);

    let game_b = FlopStartGame::new(build_table(NH));
    let (fm, tm, rm) = quantile_maps(game_b.table(), B, B, B);
    let bk_b = FlopBucketing::from_maps(game_b.table(), B, B, B, fm, tm, rm);
    let mut cpu = BucketedFlopCfr::new(&tree, game_b.table(), &bk_b);
    cpu.set_terminal_design(TerminalDesign::Design1Collapsed);
    let root_cpu = cpu.run(&tree, &game_b, &bk_b, ITERS);

    let scale = |xs: &[f32]| xs.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;
    let mut max_drift = 0.0f64;
    for ((label, ga), (_, ca)) in gpu_buffers(&native).iter().zip(cpu_buffers(&cpu).iter()) {
        assert_eq!(ga.len(), ca.len(), "{label} length");
        let s = scale(ca).max(1e-30);
        let d = ga.iter().zip(ca.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).abs() / s)
            .fold(0.0, f64::max);
        eprintln!("gate N3 (B32) {label}: max rel drift {d:.2e}");
        max_drift = max_drift.max(d);
    }
    let root_d = root_gpu.iter().zip(&root_cpu)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0, f64::max) / scale(&root_cpu).max(1e-30);
    eprintln!("gate N3 (B32) root: {root_d:.2e}");
    assert!(
        max_drift.max(root_d) < 1e-3,
        "B32 native trajectory drift {max_drift:.2e} beyond accumulated-rounding scale — \
         the cap-raise broke the terminal kernel"
    );
}

/// Gate N1-SHARED (2026-06-12, big-tree unlock): the SHARED-BUFFER
/// SEQUENTIAL pass (one nn-sized reach/cfv buffer, walks serialized by
/// encode order, per-outcome incremental chance accumulation) must be
/// bit-exact vs the pure CPU at identity — same standard as N1. Small
/// trees auto-select mega mode, so this forces shared on the same
/// fixture; if it matches CPU bit-for-bit, it also matches mega mode.
#[test]
fn gate_n1_shared_sequential_identity_bit_exact() {
    const NH: usize = 6;
    const ITERS: u32 = 3;
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_identity_tree();

    let game_a = FlopStartGame::new(build_table(NH));
    let bk_a = FlopBucketing::identity(game_a.table());
    let solver_a = BucketedFlopCfr::new(&tree, game_a.table(), &bk_a);
    let mut native =
        BucketedNativeGpu::new_forced_shared(&ctx, &tree, game_a.table(), &bk_a, &solver_a, 1)
            .expect("native gpu (shared)");
    assert!(native.is_shared_mode());
    let root_gpu = native.run(ITERS);

    let game_b = FlopStartGame::new(build_table(NH));
    let bk_b = FlopBucketing::identity(game_b.table());
    let mut cpu = BucketedFlopCfr::new(&tree, game_b.table(), &bk_b);
    cpu.set_terminal_design(TerminalDesign::Design1Collapsed);
    let root_cpu = cpu.run(&tree, &game_b, &bk_b, ITERS);

    for (i, (a, b)) in root_gpu.iter().zip(&root_cpu).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "root cfv[{i}]: shared {a} vs cpu {b}");
    }
    for ((label, ga), (_, ca)) in gpu_buffers(&native).iter().zip(cpu_buffers(&cpu).iter()) {
        assert_eq!(ga.len(), ca.len(), "{label} length");
        for i in 0..ga.len() {
            assert_eq!(
                ga[i].to_bits(),
                ca[i].to_bits(),
                "{label}[{i}]: shared {} vs cpu {}",
                ga[i],
                ca[i]
            );
        }
    }
    eprintln!("gate N1-SHARED PASSED: sequential shared-buffer pass bit-exact at identity");
}
