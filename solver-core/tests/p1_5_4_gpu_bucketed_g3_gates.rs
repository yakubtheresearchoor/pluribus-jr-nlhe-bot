//! G3 gates for the GPU bucketed terminal offload, at the standard
//! declared in G1 (and amended per directive):
//!
//!   Gate 1 (this file, load-bearing): UNSTRIPED config (stripes=1),
//!   full identity walk at B = nh — every persistent buffer bit-exact
//!   vs the pure-CPU bucketed walk across full DCFR iterations. The
//!   CPU walk is itself bit-exact-anchored to the exact evaluator, so
//!   this is the middle link of the three-point chain.
//!
//!   Gate 2: STRIPED config (production) at general B — drift vs the
//!   CPU walk pinned at accumulated-rounding scale (the f64-reference
//!   same-quantity proof and trajectory parity run in the companion
//!   gates; this gate catches gross breakage early and pins the
//!   measured number).
//!
//! ═══ MEASURED 2026-06-11 ═══
//!   Gate 1: bit-exact ✓ (root cfv + 6 persistent buffers, 3 iters,
//!     wet-deep NH=6 identity maps, every zone walk through the GPU).
//!   Gate 2: striped S=8 @ B=4 quantile maps, 3 iters: per-buffer max
//!     rel drift 7.5e-7 .. 5.7e-6 (root cfv 2.9e-7) — accumulated f32
//!     rounding, consistent with the collapse-gate precedent
//!     (~1e-5/terminal reorder, CFR-compounded). Bug line 1e-3.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::bucketed_terminal::BucketedTerminalGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const NP: u8 = 6;
const NH: usize = 6;
const ITERS: u32 = 3;

fn build_wet_deep_table() -> FlopChanceTable {
    let board: Vec<Card> = ["Th", "9d", "8c"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
            continue;
        }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
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

fn build_gate_tree() -> FlatTree {
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

/// Run ITERS iterations with and without the GPU hook; return both
/// solvers (hooked first).
fn run_pair(
    ctx: &MetalContext,
    tree: &FlatTree,
    bk_builder: impl Fn(&FlopChanceTable) -> FlopBucketing,
    stripes: u32,
) -> (BucketedFlopCfr, BucketedFlopCfr, Vec<f32>, Vec<f32>) {
    // GPU-hooked arm.
    let game_a = FlopStartGame::new(build_wet_deep_table());
    let bk_a = bk_builder(game_a.table());
    let mut gpu_arm = BucketedFlopCfr::new(tree, game_a.table(), &bk_a);
    gpu_arm.set_terminal_design(TerminalDesign::Design1Collapsed);
    let term_gpu =
        BucketedTerminalGpu::new(ctx, tree, game_a.table(), &bk_a, &gpu_arm, stripes)
            .expect("gpu terminal");
    gpu_arm.set_terminal_offload_hook(Some(term_gpu.into_hook()));
    let root_gpu = gpu_arm.run(tree, &game_a, &bk_a, ITERS);

    // Pure-CPU arm.
    let game_b = FlopStartGame::new(build_wet_deep_table());
    let bk_b = bk_builder(game_b.table());
    let mut cpu_arm = BucketedFlopCfr::new(tree, game_b.table(), &bk_b);
    cpu_arm.set_terminal_design(TerminalDesign::Design1Collapsed);
    let root_cpu = cpu_arm.run(tree, &game_b, &bk_b, ITERS);

    (gpu_arm, cpu_arm, root_gpu, root_cpu)
}

fn buffers<'a>(s: &'a BucketedFlopCfr) -> [(&'static str, &'a [f32]); 6] {
    [
        ("regrets_flop", s.regrets_flop()),
        ("cum_flop", s.cum_strategy_flop()),
        ("regrets_turn", s.regrets_turn()),
        ("cum_turn", s.cum_strategy_turn()),
        ("regrets_river", s.regrets_river()),
        ("cum_river", s.cum_strategy_river()),
    ]
}

#[test]
fn gate1_unstriped_identity_walk_bit_exact() {
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_gate_tree();
    let (gpu_arm, cpu_arm, root_gpu, root_cpu) =
        run_pair(&ctx, &tree, |t| FlopBucketing::identity(t), 1);

    for (a, b) in root_gpu.iter().zip(&root_cpu) {
        assert_eq!(a.to_bits(), b.to_bits(), "root cfv: gpu {a} vs cpu {b}");
    }
    for ((label, ga), (_, ca)) in buffers(&gpu_arm).iter().zip(buffers(&cpu_arm).iter()) {
        assert_eq!(ga.len(), ca.len());
        for i in 0..ga.len() {
            assert_eq!(
                ga[i].to_bits(),
                ca[i].to_bits(),
                "{label}[{i}]: gpu {} vs cpu {} — unstriped GPU at singletons \
                 must coincide with the CPU graph; drift is a bug",
                ga[i],
                ca[i]
            );
        }
    }
    eprintln!(
        "gate 1 PASSED: identity walk bit-exact through GPU-unstriped terminals \
         ({ITERS} iters, root + 6 buffers)"
    );
}

#[test]
fn gate2_striped_general_b_drift_pinned() {
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_gate_tree();
    const NB: usize = 4;
    let (gpu_arm, cpu_arm, root_gpu, root_cpu) = run_pair(
        &ctx,
        &tree,
        |t| {
            let (fm, tm, rm) = quantile_maps(t, NB);
            FlopBucketing::from_maps(t, NB, NB, NB, fm, tm, rm)
        },
        (32 / NB) as u32, // S = 8 at B = 4: full 32-lane striping
    );

    let mut max_drift = 0.0f64;
    let scale = |xs: &[f32]| xs.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;
    for ((label, ga), (_, ca)) in buffers(&gpu_arm).iter().zip(buffers(&cpu_arm).iter()) {
        let s = scale(ca).max(1e-30);
        let d = ga
            .iter()
            .zip(ca.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).abs() / s)
            .fold(0.0, f64::max);
        eprintln!("gate 2 {label}: max rel drift {d:.2e}");
        if d > max_drift {
            max_drift = d;
        }
    }
    let root_d = root_gpu
        .iter()
        .zip(&root_cpu)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0, f64::max)
        / scale(&root_cpu).max(1e-30);
    eprintln!("gate 2 root cfv: max rel drift {root_d:.2e}");
    // Accumulated f32 rounding through 3 DCFR iterations of reordered
    // terminal sums. Bug line set from the collapse-gate precedent
    // (per-terminal reorder ≈ 1e-5; the CFR loop compounds ~2.8×/iter
    // → ≤ ~1e-4 at 3 iters). Beyond 1e-3 is breakage, not rounding.
    assert!(
        max_drift.max(root_d) < 1e-3,
        "striped drift {max_drift:.2e} beyond accumulated-rounding scale — breakage"
    );
}
