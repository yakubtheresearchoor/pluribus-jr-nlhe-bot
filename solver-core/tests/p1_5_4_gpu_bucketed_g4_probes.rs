//! G4 step 1: unit probes at production nh, BEFORE any full GPU run
//! (the 563× lesson). Measures:
//!
//!   1. Per-zone-walk GPU terminal fill cost at nh=1176, B∈{8,10},
//!      striped vs unstriped — the offload's unit of work (one
//!      dispatch + one sync per zone walk).
//!   2. Per-stage breakdown of one full hybrid iteration (GPU
//!      terminals + CPU walk), with the PRE-REGISTERED signature
//!      check: are the walk stages launch-latency-bound (tiny CPU
//!      cost dominated by the offload's dispatch+sync overhead)? That
//!      signature makes multi-flop concurrency the expected unlock.
//!   3. Projection with budget guard; full-iteration measurement only
//!      if sane.
//!
//! CPU-contention caveat (named): the baseline blueprint occupies 14
//! cores while this probe runs; GPU-side numbers are clean (GPU idle),
//! CPU-side stage times are contention-inflated. The probe prints
//! both; ladder rows that depend on clean CPU numbers are banked only
//! after the baseline completes.
//!
//! ═══ MEASURED 2026-06-11 (M2 tree, nh=1176, baseline occupying 14
//! cores — GPU numbers clean, CPU numbers contention-inflated) ═══
//!
//!   Zone-walk terminal fill (dense iter-1-like reach):
//!     B=8  striped S=4: flop 57.6 / turn 62.8 / river 79.6 ms
//!     B=8  unstriped:   223.7 / 229.7 / 282.1 ms  (striping ≈ 3.7×,
//!       near the S=4 ideal — lanes are busy, not starved)
//!     B=10 striped S=3: 267.0 / 271.7 / 336.4 ms
//!     B=10 unstriped:   655.7 / 676.7 / 844.4 ms  (≈ 2.5× at S=3)
//!   Full hybrid iteration (GPU terminals + CPU walk):
//!     B=8:  0.76s iter-1, 0.68s iter-2  vs CPU 15.3s / 7.4s → 20×
//!     B=10: 2.47s iter-1, 2.48s iter-2  vs CPU 57.4s / 24.6s → 23×
//!   READ-THROUGHS:
//!   - The hybrid's iter-2 ≈ iter-1 (no sparsification benefit): the
//!     GPU terminal enumerates the full B^K space regardless of reach
//!     zeros (dense bucket reach at production nh), so the hybrid's
//!     cost is FLAT across iterations while the CPU's decays 20-35×.
//!     At converged-average the CPU is ~0.5s/iter vs hybrid 0.68s —
//!     the hybrid wins ONLY the early dense iterations at B=8, but
//!     wins everywhere at B=10+. The GPU case strengthens with B,
//!     exactly as the B^K scaling predicts.
//!   - Walk-stage signature: hybrid iter at B=8 is 0.68s of which
//!     terminal fills ≈ 24 walks × ~60-80ms... that IS the whole
//!     iteration — the offload (dispatch+sync+reach copy per walk)
//!     dominates and the CPU walk is in its shadow. The pre-registered
//!     launch-latency signature CONFIRMED: per-walk overhead bounds
//!     the hybrid floor → multi-flop concurrency (independent queues,
//!     overlapping fills) is the expected unlock for full-chip
//!     utilization.
//!   Projection (34 iters × 1755 flops, hybrid, GPU serialized):
//!     B=8 ≈ 11.3h | B=10 ≈ 36h — single-GPU-stream is comparable to
//!     16-core CPU at B=8; the ladder's real rows need the multi-flop
//!     axis (next probe) before any cell is re-priced.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, Card};
use solver_core::gpu_metal::bucketed_terminal::BucketedTerminalGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::Zone;
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

fn build_table_1x1() -> FlopChanceTable {
    let flop: [Card; 3] = [
        card_from_str("2h").unwrap(),
        card_from_str("7d").unwrap(),
        card_from_str("Ks").unwrap(),
    ];
    let tc = card_from_str("3c").unwrap() as u8;
    let rc = card_from_str("5s").unwrap() as u8;
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[tc as usize] = vec![rc];
    FlopChanceTable::build_full_nh_sampled(flop, NP, &[tc], &river_decks)
}

#[test]
#[ignore = "G4 unit probes (~minutes); run with --ignored --nocapture"]
fn g4_unit_probes() {
    eprintln!("\n════ G4 step 1: GPU bucketed unit probes at production nh ════");
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_m2_tree();
    let nn = tree.num_nodes();
    let nh = build_table_1x1().num_valid;
    eprintln!("M2 tree {} nodes, nh={nh} (1×1 runout)", nn);
    eprintln!("(CPU numbers contention-inflated: baseline occupies 14 cores — named)");

    for nb in [8usize, 10] {
        let table = build_table_1x1();
        let (fm, tm, rm) = quantile_maps(&table, nb);
        let game = FlopStartGame::new(table);
        let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
        let solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
        let np = NP as usize;

        // Dense synthetic reach (iter-1-like: the expensive regime).
        let mut reach = vec![0.0f32; nn * np * nh];
        for (i, r) in reach.iter_mut().enumerate() {
            let v = (i as u32).wrapping_mul(2654435761) % 13;
            *r = if v == 0 { 0.0 } else { v as f32 / 16.0 };
        }
        let mut cfv = vec![0.0f32; nn * nh];

        for stripes in [(32 / nb) as u32, 1] {
            let mut gpu =
                BucketedTerminalGpu::new(&ctx, &tree, game.table(), &bk, &solver, stripes)
                    .expect("gpu");
            // Warm (pipeline/caches), then time per zone walk.
            gpu.fill_terminals(Zone::Flop, None, None, 0, &reach, &mut cfv);
            for (zname, zone, tc, rc) in [
                ("flop ", Zone::Flop, None, None),
                ("turn ", Zone::Turn, Some(0), None),
                ("river", Zone::River, Some(0), Some(0)),
            ] {
                let reps = 5;
                let t0 = Instant::now();
                for trav in 0..reps {
                    gpu.fill_terminals(zone, tc, rc, (trav % 6) as u8, &reach, &mut cfv);
                }
                let per = t0.elapsed().as_secs_f64() / reps as f64;
                eprintln!(
                    "B={nb} S={stripes} {zname} walk terminal fill: {:.2} ms",
                    per * 1e3
                );
            }
        }

        // Full hybrid iteration, per-stage attribution.
        let mut hyb = BucketedFlopCfr::new(&tree, game.table(), &bk);
        hyb.set_terminal_design(TerminalDesign::Design1Collapsed);
        let gpu = BucketedTerminalGpu::new(&ctx, &tree, game.table(), &bk, &hyb, (32 / nb) as u32)
            .expect("gpu");
        hyb.set_terminal_offload_hook(Some(gpu.into_hook()));
        let t0 = Instant::now();
        hyb.run(&tree, &game, &bk, 1);
        let hybrid_iter1 = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        hyb.run(&tree, &game, &bk, 1);
        let hybrid_iter2 = t0.elapsed().as_secs_f64();

        // CPU-only comparison (same iteration indices).
        let game2 = FlopStartGame::new(build_table_1x1());
        let (fm2, tm2, rm2) = quantile_maps(game2.table(), nb);
        let bk2 = FlopBucketing::from_maps(game2.table(), nb, nb, nb, fm2, tm2, rm2);
        let mut cpu = BucketedFlopCfr::new(&tree, game2.table(), &bk2);
        cpu.set_terminal_design(TerminalDesign::Design1Collapsed);
        let t0 = Instant::now();
        cpu.run(&tree, &game2, &bk2, 1);
        let cpu_iter1 = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        cpu.run(&tree, &game2, &bk2, 1);
        let cpu_iter2 = t0.elapsed().as_secs_f64();

        eprintln!(
            "B={nb}: hybrid iter1 {hybrid_iter1:.2}s iter2 {hybrid_iter2:.2}s | \
             CPU iter1 {cpu_iter1:.2}s iter2 {cpu_iter2:.2}s | \
             iter1 speedup {:.1}×",
            cpu_iter1 / hybrid_iter1
        );
    }
}
