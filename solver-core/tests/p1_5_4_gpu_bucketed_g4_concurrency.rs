//! G4 step 2: multi-flop concurrency probe WITH the attribution
//! channel (per directive — a ratio alone cannot distinguish chip
//! saturation from orchestration serialization; the Fix-B failure
//! mode one level up).
//!
//! Design: N fully independent worker threads, each with its OWN
//! command queue (set_queue — cross-flop submission parallel by
//! construction), its own flop, table, bucketing, GPU object, and
//! hybrid solver. Each runs ITERS hybrid iterations; aggregate
//! throughput (flop-iterations/sec) at N ∈ {1, 2, 4, 8} plus the
//! summed per-command-buffer GPU-busy time.
//!
//! Attribution rule:
//!   ratio ≈ N                      → chip was idle; concurrency is
//!                                    the unlock (expected, per the
//!                                    confirmed launch-latency
//!                                    signature).
//!   ratio ≈ 1, busy-fraction high  → chip saturated; bank it, skip
//!                                    concurrency.
//!   ratio ≈ 1, busy-fraction low   → orchestration serialized; FIX
//!                                    THE DRIVER LOOP and re-measure.
//!
//! CPU-contention caveat (named): each worker also burns a core on
//! the CPU walk; at N=8 with the baseline running the box is
//! oversubscribed. The GPU-busy channel is immune; the throughput
//! ratio at high N is a lower bound until the baseline completes.
//!
//! ═══ MEASURED 2026-06-11 (B=8, S=4, 3 iters/worker; baseline
//! occupying 14 cores — N≥4 rows are CPU-confounded lower bounds) ═══
//!   N=1: 1.22 flop-iters/s | GPU busy 76% of wall
//!   N=2: 2.32 (ratio 1.90)  | summed busy 141% of wall
//!   N=4: 2.77 (ratio 2.27)  | 164%
//!   N=8: 2.56 (ratio 2.09)  | 150%  ← oversubscribed CPU, re-measure
//!
//!   ATTRIBUTION (the channel earned its keep): neither pre-named
//!   outcome. Summed busy > wall proves kernels OVERLAP across queues
//!   (not orchestration-serialized), but scaling stops at ~2.3× —
//!   per-dispatch OCCUPANCY is the binding constraint: each fill
//!   launches only ~100-550 threadgroups (one per terminal), far
//!   below what the chip needs in flight; two streams interleave into
//!   each other's gaps, beyond that the small dispatches contend
//!   rather than stack. The unlock is therefore BIGGER DISPATCHES,
//!   not more queues: batch all (traverser × outcome) walks of an
//!   iteration into one grid (the terminal count × 6 × n_outcomes —
//!   the Fix-C lesson at the iteration level), with multi-flop
//!   concurrency layered on top afterward. N≥4 re-measure after the
//!   baseline frees cores; batching redesign is the measured next
//!   step before any ladder row is banked.

#![cfg(feature = "metal")]

use solver_core::card::Card;
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const NP: u8 = 6;
const NB: usize = 8;
const ITERS: u32 = 3;

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

fn table_for_flop(flop: [Card; 3]) -> FlopChanceTable {
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let tc = deck[12];
    let rdeck: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
    let rc = rdeck[10];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[tc as usize] = vec![rc];
    FlopChanceTable::build_full_nh_sampled(flop, NP, &[tc], &river_decks)
}

#[test]
#[ignore = "G4 multi-flop concurrency probe (~10-15 min); run with --ignored --nocapture"]
fn g4_multiflop_concurrency() {
    eprintln!("\n════ G4 step 2: multi-flop concurrency (independent queues) ════");
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_m2_tree();
    let ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![1.0 / 169.0; 169]).collect();
    let ptable = PreflopChanceTable::new(NP, ranges);
    let flops: Vec<[Card; 3]> =
        (0..8).map(|i| ptable.canonical_flops[i * 200]).collect();

    let mut t1 = 0.0f64;
    for n in [1usize, 2, 4, 8] {
        let busy_ns = AtomicU64::new(0);
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for w in 0..n {
                let flop = flops[w];
                let tree = &tree;
                let ctx = &ctx;
                let busy_ns = &busy_ns;
                s.spawn(move || {
                    let table = table_for_flop(flop);
                    let (fm, tm, rm) = quantile_maps(&table, NB);
                    let game = FlopStartGame::new(table);
                    let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
                    let mut solver = BucketedFlopCfr::new(tree, game.table(), &bk);
                    solver.set_terminal_design(TerminalDesign::Design1Collapsed);
                    let mut gpu = BucketedTerminalGpu::new(
                        ctx, tree, game.table(), &bk, &solver, (32 / NB) as u32,
                    )
                    .expect("gpu");
                    // Independent queue per worker — the directive's
                    // by-construction parallel orchestration.
                    gpu.set_queue(ctx.device().new_command_queue());
                    // Warm one walk, then run ITERS full iterations
                    // through the hook. The GPU object must outlive the
                    // hook to read busy time: track via a raw pointer
                    // pattern is unsafe — instead reconstruct: hook owns
                    // gpu; busy time accumulated inside and reported by
                    // a final probe call. Simpler: don't use the hook;
                    // drive run() with the hook installed from a Boxed
                    // gpu we can't read back... so instead run the walk
                    // manually? No — wrap: read busy AFTER by keeping
                    // the gpu in an Arc<Mutex<>> closure.
                    let gpu = std::sync::Arc::new(std::sync::Mutex::new(gpu));
                    let gpu_for_hook = gpu.clone();
                    solver.set_terminal_offload_hook(Some(Box::new(
                        move |zone: Zone, tc, rc, trav, reach: &[f32], cfv: &mut [f32]| {
                            gpu_for_hook
                                .lock()
                                .unwrap()
                                .fill_terminals(zone, tc, rc, trav, reach, cfv)
                        },
                    )));
                    solver.run(tree, &game, &bk, ITERS);
                    let busy = gpu.lock().unwrap().gpu_busy_seconds();
                    busy_ns.fetch_add((busy * 1e9) as u64, Ordering::Relaxed);
                });
            }
        });
        let wall = t0.elapsed().as_secs_f64();
        let busy = busy_ns.load(Ordering::Relaxed) as f64 / 1e9;
        let throughput = (n as u32 * ITERS) as f64 / wall;
        if n == 1 {
            t1 = throughput;
        }
        eprintln!(
            "N={n}: wall {wall:.1}s | {throughput:.2} flop-iters/s | ratio vs N=1: {:.2} | \
             GPU busy {busy:.1}s = {:.0}% of wall",
            throughput / t1,
            100.0 * busy / wall
        );
    }
    eprintln!(
        "\nattribution rule: ratio≈N → idle chip, concurrency is the unlock; \
         ratio≈1 + busy high → saturated; ratio≈1 + busy low → orchestration \
         bug, fix and re-measure."
    );
}
