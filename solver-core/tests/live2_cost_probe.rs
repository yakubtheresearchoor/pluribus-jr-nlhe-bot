//! LIVE-2 EXACT COST PROBE (2026-06-14): full-fidelity live-2 = per-hand
//! exact (`FlopStartVectorCfr`, exact terminal — np=2 showdown is O(nh²), no
//! wall). Measure per-flop solve + CFV-extract cost at production nh=1176, and
//! project the full live-2 fill, to decide: exact-CPU is fine, vs a bucketed-
//! np2 fallback, vs GPU. CFV extraction uses `strategy_value_debug` — which is
//! O(nh²) and FEASIBLE at np=2 (the same call hangs at np=4: O(nh⁴)).

use solver_core::card::{card_from_str, Card};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use std::time::Instant;

#[test]
#[ignore = "live-2 exact cost probe; --ignored --nocapture --release"]
fn live2_cost_probe() {
    let spec = production_game_v1();
    let (commit, pot) = (2i32, 12i32);
    let tree = build_tree(&spec.flop_seam_config(
        2, commit, pot,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
    )).unwrap();
    eprintln!("\n═══ LIVE-2 EXACT COST PROBE (commit {commit}, pot {pot}) ═══");
    eprintln!("seam tree: {} nodes", tree.num_nodes());

    // Full nh=1176, 1×1 runout.
    let board: [Card; 3] = ["Ks", "7d", "2c"].map(|s| card_from_str(s).unwrap());
    let turn = card_from_str("9h").unwrap();
    let river = card_from_str("4s").unwrap();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn as usize] = vec![river];
    let table = FlopChanceTable::build_full_nh_sampled(board, 2, &[turn], &river_decks);
    let nh = table.num_valid;
    let game = FlopStartGame::new(table);
    eprintln!("nh = {nh}");

    const ITERS: u32 = 34;
    let mut s = FlopStartVectorCfr::new(&tree, game.table());
    let t0 = Instant::now();
    s.run(&tree, &game, ITERS);
    let solve = t0.elapsed().as_secs_f64();

    // CFV extraction (both traversers, what the preflop oracle needs).
    let t0 = Instant::now();
    let _c0 = s.strategy_value_debug(&tree, &game, 0);
    let _c1 = s.strategy_value_debug(&tree, &game, 1);
    let cfv = t0.elapsed().as_secs_f64();

    let per_flop = solve + cfv;
    eprintln!("solve {solve:.2}s ({ITERS} it) + CFV-extract(2 trav) {cfv:.3}s = {per_flop:.2}s / flop");

    let (cells, flops) = (25.0f64, 1755.0f64);
    let core_h = cells * flops * per_flop / 3600.0;
    eprintln!(
        "PROJECT full live-2 fill ({cells:.0} cells × {flops:.0} flops):\n  \
         {core_h:.1} core-hours → ~{:.1} h @14 threads / ~{:.1} h @8 threads",
        core_h / 14.0, core_h / 8.0
    );
    eprintln!("→ if @14thr ≲ 2h: exact-CPU fine. If much more: bucketed-np2 fallback or GPU.");
}
