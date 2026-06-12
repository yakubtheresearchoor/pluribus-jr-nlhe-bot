//! G4 final measurements (quiet box — baseline completed): the GPU
//! ladder, clean CPU reference rows, and the concurrency re-measure
//! on the batched path. The cell re-pricing arithmetic prints at the
//! end of each test; the deliverable table is assembled from these
//! rows in the close-out report.
//!
//! Cells: B ∈ {8, 10} × runouts {1×1, 2×2, 4×4} on the M2-shaped tree
//! at production nh=1176 (the shape every prior rung used — re-prices
//! linearly to other tree shapes by terminal count, as established).
//!
//! ═══ MEASURED 2026-06-11 (quiet box, baseline complete) ═══
//!
//!   Full-blueprint hours = s/iter × 34 × 1755 (CPU ÷14 threads;
//!   GPU single stream; concurrency adds ~1.7× at N=4-8):
//!
//!   cell       | CPU s/iter (34-avg) → hours | GPU s/iter (busy) → hours
//!   B=8  1×1   |  0.93 →  1.1h               | 0.30 (70%) →  4.9h
//!   B=8  2×2   |  3.02 →  3.6h               | 0.54 (56%) →  9.0h
//!   B=8  4×4   | 10.48 → 12.4h  ← FITS 24h   | 1.41 (43%) → 23.4h
//!   B=10 1×1   |  3.01 →  3.6h               | 1.00 (91%) → 16.6h
//!   B=10 2×2   | 50.1 first-5 partial (≤59h) | 1.53 (84%) → 25.4h (~15h w/ conc.)
//!   B=10 4×4   | 183  first-5 partial (≤218h)| 3.69 (78%) → 61.1h (~36h w/ conc.)
//!
//!   Concurrency re-measure (batched, B=10 1×1): N=1 92% busy already
//!   (batching saturated the chip per-flop); N=8 ratio 1.78× at 165%
//!   summed busy — concurrency is a ~1.7× multiplier, not N×.
//!
//!   READINGS:
//!   - CPU wins every B=8 cell: converged-run sparsification (zero-
//!     reach skips) beats the GPU's flat full-B^K enumeration at small
//!     B. The GPU crossover is B=10 × 2×2+ — fidelity territory.
//!   - THE RE-PRICED FRONTIER: B=8 × 4×4 × full 1755 canonicals fits
//!     the 24h budget ON CPU (12.4h) — 4×4 was the runout-stability
//!     REFERENCE fidelity, so the measured runout penalty collapses
//!     into the named-to-head-to-head residual (4×4 vs full 47×46).
//!   - GPU per-board solve latency: 34 iters × 1.41s = 48s vs CPU
//!     single-thread 356s (7.4×) — the banked role for the search
//!     layer's re-solves, plus B≥10 challenger blueprints and refresh
//!     epochs.
//!   Tree-shape caveat unchanged: M2 1-bet shape; the lean MAX_NA=4
//!   action set re-prices by terminal count and needs its own row
//!   before that tree is banked.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, Card};
use solver_core::gpu_metal::bucketed_terminal::BucketedTerminalGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::Zone;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::sync::{Arc, Mutex};
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

/// Production-nh table at the given runout policy (deterministic deck
/// positions — the runout-policy probe's convention).
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

struct Cell {
    solver: BucketedFlopCfr,
    game: FlopStartGame,
    bk: FlopBucketing,
}

fn build_cell(tree: &FlatTree, nb: usize, nt: usize, nr: usize) -> Cell {
    let table = build_table(nt, nr);
    let (fm, tm, rm) = quantile_maps(&table, nb);
    let game = FlopStartGame::new(table);
    let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
    let mut solver = BucketedFlopCfr::new(tree, game.table(), &bk);
    solver.set_terminal_design(TerminalDesign::Design1Collapsed);
    Cell { solver, game, bk }
}

fn attach_gpu(ctx: &MetalContext, tree: &FlatTree, cell: &mut Cell) -> Arc<Mutex<BucketedTerminalGpu>> {
    let nb = cell.bk.nb_flop;
    let gpu = Arc::new(Mutex::new(
        BucketedTerminalGpu::new(ctx, tree, cell.game.table(), &cell.bk, &cell.solver, (32 / nb) as u32)
            .expect("gpu"),
    ));
    let gpu_h = gpu.clone();
    cell.solver.set_terminal_offload_hook(Some(Box::new(
        move |zone, tc, rc, trav, reach: &[f32], cfv: &mut [f32]| {
            gpu_h.lock().unwrap().fill_terminals(zone, tc, rc, trav, reach, cfv)
        },
    )));
    gpu
}

fn run_batched_iters(
    cell: &mut Cell,
    tree: &FlatTree,
    gpu: &Arc<Mutex<BucketedTerminalGpu>>,
    iters: u32,
) {
    let gpu_f = gpu.clone();
    let mut fill =
        move |walks: &[(Zone, Option<usize>, Option<usize>)], trav: u8, reaches: &[&[f32]]| {
            gpu_f.lock().unwrap().fill_walks(walks, trav, reaches);
        };
    cell.solver.run_batched(tree, &cell.game, &cell.bk, iters, &mut fill);
}

#[test]
#[ignore = "G4 GPU ladder: B{8,10} × runouts{1×1,2×2,4×4} (~10 min); run with --ignored --nocapture"]
fn g4_ladder_gpu() {
    eprintln!("\n════ G4 GPU ladder (batched hybrid, quiet box) ════");
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_m2_tree();
    for nb in [8usize, 10] {
        for (nt, nr) in [(1usize, 1usize), (2, 2), (4, 4)] {
            let mut cell = build_cell(&tree, nb, nt, nr);
            let gpu = attach_gpu(&ctx, &tree, &mut cell);
            run_batched_iters(&mut cell, &tree, &gpu, 1); // warm
            let busy0 = gpu.lock().unwrap().gpu_busy_seconds();
            let t0 = Instant::now();
            run_batched_iters(&mut cell, &tree, &gpu, 3);
            let per_iter = t0.elapsed().as_secs_f64() / 3.0;
            let busy = (gpu.lock().unwrap().gpu_busy_seconds() - busy0) / 3.0;
            let full_1755_h = per_iter * 34.0 * 1755.0 / 3600.0;
            eprintln!(
                "GPU B={nb} {nt}×{nr}: {per_iter:.2}s/iter | busy {:.0}% | \
                 34×1755 single-stream ≈ {full_1755_h:.1}h",
                100.0 * busy / per_iter
            );
        }
    }
}

#[test]
#[ignore = "G4 CPU reference rows (~40 min); run with --ignored --nocapture"]
fn g4_ladder_cpu() {
    eprintln!("\n════ G4 CPU reference rows (quiet box) ════");
    let tree = build_m2_tree();
    // Full 34-iter solves where affordable; labeled partials otherwise.
    for (nb, nt, nr, iters, label) in [
        (8usize, 1usize, 1usize, 34u32, "full"),
        (8, 2, 2, 34, "full"),
        (8, 4, 4, 34, "full"),
        (10, 1, 1, 34, "full"),
        (10, 2, 2, 5, "first-5 (labeled partial)"),
        (10, 4, 4, 5, "first-5 (labeled partial)"),
    ] {
        let mut cell = build_cell(&tree, nb, nt, nr);
        let t0 = Instant::now();
        cell.solver.run(&tree, &cell.game, &cell.bk, iters);
        let total = t0.elapsed().as_secs_f64();
        let full_1755_h = if iters == 34 {
            total * 1755.0 / 14.0 / 3600.0
        } else {
            f64::NAN
        };
        eprintln!(
            "CPU B={nb} {nt}×{nr} [{label}]: {total:.0}s/{iters} iters \
             ({:.2}s/iter avg){}",
            total / iters as f64,
            if iters == 34 {
                format!(" | 34×1755 @14 threads ≈ {full_1755_h:.1}h")
            } else {
                String::new()
            }
        );
    }
}

#[test]
#[ignore = "G4 concurrency re-measure, batched path (~10 min); run with --ignored --nocapture"]
fn g4_concurrency_batched() {
    eprintln!("\n════ G4 concurrency re-measure (batched, quiet box) ════");
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_m2_tree();
    let ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![1.0 / 169.0; 169]).collect();
    let ptable = PreflopChanceTable::new(NP, ranges);
    const NB: usize = 10;
    const ITERS: u32 = 3;

    let mut t1 = 0.0f64;
    for n in [1usize, 2, 4, 8] {
        let busy_total = Arc::new(Mutex::new(0.0f64));
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for w in 0..n {
                let flop = ptable.canonical_flops[w * 200];
                let tree = &tree;
                let ctx = &ctx;
                let busy_total = busy_total.clone();
                s.spawn(move || {
                    let board_mask: u64 =
                        flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
                    let deck: Vec<u8> =
                        (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
                    let tc = deck[12];
                    let rdeck: Vec<u8> =
                        deck.iter().copied().filter(|&c| c != tc).collect();
                    let rc = rdeck[10];
                    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
                    river_decks[tc as usize] = vec![rc];
                    let table =
                        FlopChanceTable::build_full_nh_sampled(flop, NP, &[tc], &river_decks);
                    let (fm, tm, rm) = quantile_maps(&table, NB);
                    let game = FlopStartGame::new(table);
                    let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
                    let mut solver = BucketedFlopCfr::new(tree, game.table(), &bk);
                    solver.set_terminal_design(TerminalDesign::Design1Collapsed);
                    let mut gpu = BucketedTerminalGpu::new(
                        ctx, tree, game.table(), &bk, &solver, (32 / NB) as u32,
                    )
                    .expect("gpu");
                    gpu.set_queue(ctx.device().new_command_queue());
                    let gpu = Arc::new(Mutex::new(gpu));
                    let gpu_h = gpu.clone();
                    solver.set_terminal_offload_hook(Some(Box::new(
                        move |zone, tc, rc, trav, reach: &[f32], cfv: &mut [f32]| {
                            gpu_h.lock().unwrap().fill_terminals(zone, tc, rc, trav, reach, cfv)
                        },
                    )));
                    let gpu_f = gpu.clone();
                    let mut fill = move |walks: &[(Zone, Option<usize>, Option<usize>)],
                                         trav: u8,
                                         reaches: &[&[f32]]| {
                        gpu_f.lock().unwrap().fill_walks(walks, trav, reaches);
                    };
                    solver.run_batched(tree, &game, &bk, ITERS, &mut fill);
                    *busy_total.lock().unwrap() += gpu.lock().unwrap().gpu_busy_seconds();
                });
            }
        });
        let wall = t0.elapsed().as_secs_f64();
        let busy = *busy_total.lock().unwrap();
        let throughput = (n as u32 * ITERS) as f64 / wall;
        if n == 1 {
            t1 = throughput;
        }
        eprintln!(
            "N={n}: wall {wall:.1}s | {throughput:.2} flop-iters/s | ratio {:.2} | \
             summed GPU busy {:.0}% of wall",
            throughput / t1,
            100.0 * busy / wall
        );
    }
}
