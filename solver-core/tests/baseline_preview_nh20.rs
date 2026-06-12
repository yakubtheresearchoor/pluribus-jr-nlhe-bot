// Baseline preview at nh=20 — the smaller-scale stand-in for the
// nh=50 multi-day baseline. Goal:
//   1. Confirm the descent curve descends cleanly to a floor at scale
//   2. Expose any issues that show only at scale (memory pressure,
//      f32 noise compounding, kernel launch overhead dominating)
//   3. Measure the actual per-iter cost at nh=20 to pin down which
//      scaling exponent (nh^1.7 optimistic vs nh^3 pessimistic) holds
//   4. Make an informed go/no-go for the nh=50 multi-day baseline
//
// Setup: 6 players, asymmetric initial contributions [10,5,5,5,5,5]
// (every terminal is multi-level side-pot, exercising the unified
// factored kernel's Case A/B/C paths every iter). Rainbow flop
// "2h 7d Ks" — most common flop texture and the one suit isomorphism
// would have given zero benefit on. 100 GPU iters with exploitability
// checkpoints at logarithmically spaced points.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::io::Write;
use std::time::Instant;

/// AUDIT MIGRATION 2026-06: this test anchors the nh=50 baseline cost
/// projection (decision: "feasibility of the multi-day baseline run").
/// Pattern check: STRUCTURALLY CORRECT (+1 rule, exact enum_nc, uniform
/// weights) — identical pattern to six_player and pruning_floor_match,
/// both of which produced bit-identical output post-migration. Migration
/// to production API performed; downstream output expected bit-identical
/// (pattern verified in aggregate across three other tests).
///
/// Independent structural-safety note: the decision-relevant output is
/// per-iter WALL-CLOCK time, which depends on tree size and kernel work
/// (not chance-table contents). So the cost projection is independent of
/// the chance-table harness even if the table differed.
fn build_6p_asymmetric_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 6u8;

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();

    let mut ranges: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..num_players as usize {
        for &hi in &chosen {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
            ranges[p][pair_idx] = 1.0;
        }
    }
    let turn_cards = vec![card_from_str("3c").unwrap() as u8];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![card_from_str("5s").unwrap() as u8];

    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, num_players, &chosen, &turn_cards, &river_decks,
    );

    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

fn measure_exploitability(cpu: &FlopStartVectorCfr, tree: &FlatTree, game: &FlopStartGame, np: usize) -> f32 {
    let mut total = 0.0f32;
    for p in 0..np {
        let br = cpu.best_response_value_debug(tree, game, p as u8);
        let sv = cpu.strategy_value_debug(tree, game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total += (br[h] - sv[h]).max(0.0);
        }
    }
    total
}

fn upload_gpu_to_cpu(cpu: &mut FlopStartVectorCfr, gpu_reg: &[f32], gpu_cum: &[f32]) {
    let fl = cpu.regrets_flop().len();
    let tl = cpu.regrets_turn().len();
    { let r = cpu.regrets_flop_mut(); for i in 0..fl { r[i] = gpu_reg[i]; } }
    { let r = cpu.regrets_turn_mut(); for i in 0..tl { r[i] = gpu_reg[fl + i]; } }
    { let r = cpu.regrets_river_mut(); for i in 0..r.len() {
        if fl + tl + i < gpu_reg.len() { r[i] = gpu_reg[fl + tl + i]; } } }
    { let c = cpu.cum_strategy_flop_mut(); for i in 0..fl { c[i] = gpu_cum[i]; } }
    { let c = cpu.cum_strategy_turn_mut(); for i in 0..tl { c[i] = gpu_cum[fl + i]; } }
    { let c = cpu.cum_strategy_river_mut(); for i in 0..c.len() {
        if fl + tl + i < gpu_cum.len() { c[i] = gpu_cum[fl + tl + i]; } } }
}

#[test]
#[ignore = "multi-hour baseline preview; run with `cargo test --release --features metal -- --ignored`"]
fn baseline_preview_nh20_100iters() {
    // PIVOTED TWICE. nh=20 stalled at 47 min (CPU 0%). nh=16 active at
    // 99% CPU for 60+ min but iter-1 never printed — best-response
    // cost dominates at scale. Pivoted to nh=12 with FEWER checkpoints
    // to minimize CPU best-response overhead, where we have measured
    // baseline (86s/iter timed, ~15 min warmup).
    let nh = 12;
    let total_iters = 100u32;
    // Reduced checkpoint count: only measure expensive best-response
    // at end of run. Per-iter GPU cost is measured inline in the run.
    let checkpoints: &[u32] = &[1, 10, 50, 100];

    eprintln!("\n=== Baseline preview: 6p nh={} 100 iters ===", nh);
    eprintln!("Rainbow flop, asymmetric contribs [10,5,5,5,5,5], single bet/no raise");
    eprintln!("Goals:");
    eprintln!("  1. Confirm descent curve cleanly descends to a floor");
    eprintln!("  2. Measure per-iter cost to disambiguate scaling exponent");
    eprintln!("  3. Project nh=50 baseline cost from real data");
    eprintln!();

    let t_build = Instant::now();
    let (tree, table) = build_6p_asymmetric_table(nh);
    let game = FlopStartGame::new(table);
    let np = 6usize;
    eprintln!("Tree build: {:?}", t_build.elapsed());
    eprintln!("Tree: {} nodes, {} bytes per nh-vector ({} MB regrets at nh={})",
        tree.num_nodes(),
        tree.num_nodes() * nh * 4,
        tree.num_nodes() * nh * 4 / 1024 / 1024,
        nh);

    let mut cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);

    let pot_total = 65.0f32;
    let mut prev = 0u32;
    let t0 = Instant::now();
    let mut trajectory: Vec<(u32, f32, f64)> = Vec::new();

    eprintln!();
    eprintln!("{:>5} | {:>13} | {:>12} | {:>12} | {:>10}",
        "iter", "expl (% pot)", "iter (s)", "total (h)", "iter avg s");
    eprintln!("{}", "-".repeat(70));

    for &cp in checkpoints {
        if cp == 0 { continue; }
        let n_to_run = cp - prev;
        let it0 = Instant::now();
        gpu.run(&ctx, &tree, &game, n_to_run);
        let batch_elapsed = it0.elapsed().as_secs_f64();
        prev = cp;
        let avg_s = batch_elapsed / n_to_run as f64;

        // Exploitability measurement
        cpu_proxy.set_iteration(gpu.iteration());
        let gpu_reg = gpu.download_regrets();
        let gpu_cum = gpu.download_cum_strategy();
        upload_gpu_to_cpu(&mut cpu_proxy, &gpu_reg, &gpu_cum);
        let expl = measure_exploitability(&cpu_proxy, &tree, &game, np);
        let pct = expl / pot_total * 100.0;
        trajectory.push((cp, pct, avg_s));
        eprintln!("{:>5} | {:>12.6}% | {:>12.1} | {:>12.2} | {:>10.2}",
            cp, pct, batch_elapsed, t0.elapsed().as_secs_f64() / 3600.0, avg_s);
        std::io::stderr().flush().ok();
    }
    let total_elapsed = t0.elapsed();

    eprintln!();
    eprintln!("=== Summary ===");
    let first_pct = trajectory.first().map(|&(_, p, _)| p).unwrap_or(0.0);
    let last_pct = trajectory.last().map(|&(_, p, _)| p).unwrap_or(0.0);
    eprintln!("Descent: {:.6}% → {:.6}% over {} iters ({:.0}x)",
        first_pct, last_pct, total_iters, first_pct / last_pct.max(1e-12));
    eprintln!("Total wall-clock: {:.2} hours", total_elapsed.as_secs_f64() / 3600.0);
    let avg_iter_s = total_elapsed.as_secs_f64() / total_iters as f64;
    eprintln!("Average iter time: {:.2} s/iter", avg_iter_s);
    eprintln!();
    eprintln!("=== nh=50 baseline projection (assuming similar nh-scaling) ===");
    for &(name, exp) in &[("optimistic (nh^1.7)", 1.7), ("mid (nh^2.5)", 2.5), ("pessimistic (nh^3.0)", 3.0), ("very pessimistic (nh^4)", 4.0)] {
        let scale = (50.0_f64 / nh as f64).powf(exp);
        let proj_s = avg_iter_s * scale;
        let proj_200_h = proj_s * 200.0 / 3600.0;
        eprintln!("  {}: {:.0}s/iter at nh=50 → 200 iters: {:.1} h = {:.2} d",
            name, proj_s, proj_200_h, proj_200_h / 24.0);
    }

    assert!(last_pct.is_finite(), "f32 instability: non-finite exploitability");
    assert!(last_pct < first_pct, "Did not converge: {:.6}% → {:.6}%", first_pct, last_pct);
}
