// Phase 1.A validation: the pruned solve must reach the same exploitability
// floor as the unpruned solve in the FULL GAME (not the pruned solve's own
// reduced world). Pruning that changes the floor is changing the equilibrium,
// not skipping idle work. The Pluribus carve-outs (never prune river, never
// prune terminal-leading, stochastic re-enable, regret floor for recovery)
// are what make pruning behavior-preserving — this test confirms they are.
//
// Method: small game (3-player nh=8 corrected asymmetric tree, fast to
// converge), run twice — pruning OFF then pruning ON — and compare final
// exploitability. They must match within tolerance.

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

/// AUDIT MIGRATION 2026-06: build_3p_asymmetric_table now uses production API.
/// This test anchors the Phase 1.A negative-regret pruning decision ("Pluribus
/// carve-outs make pruning behavior-preserving"). Pre-migration measurement:
/// unpruned floor 0.017238% of pot, pruned floor 0.017238% — bit-identical
/// (0.00% drift against a 10% gate). The named rationalization "f32 noise +
/// stochastic re-enable timing" predicted some drift; reality shows none.
/// Hand-rolling pattern was structurally correct (+1 rule, exact recursive nc,
/// uniform weights), so migration expected bit-identical (per the
/// pattern-check-as-hypothesis discipline from the six_player audit).
fn build_3p_asymmetric_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 3u8;
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
        num_players, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![200; 3],
        initial_contributions: vec![10, 5, 5],
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

fn run_and_get_floor(
    n_iters: u32, pruning: Option<(f32, u32)>,
    nh: usize, np: usize,
) -> (f32, f64) {
    let (tree, table) = build_3p_asymmetric_table(nh);
    let game = FlopStartGame::new(table);
    let mut cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);

    if let Some((thr, stride)) = pruning {
        gpu.set_pruning(true, thr, stride);
        eprintln!("  Pruning ENABLED: threshold={}, stride={}", thr, stride);
    } else {
        eprintln!("  Pruning DISABLED (baseline)");
    }

    let t0 = Instant::now();
    gpu.run(&ctx, &tree, &game, n_iters);
    let elapsed_s = t0.elapsed().as_secs_f64();

    cpu_proxy.set_iteration(gpu.iteration());
    let gpu_reg = gpu.download_regrets();
    let gpu_cum = gpu.download_cum_strategy();
    upload_gpu_to_cpu(&mut cpu_proxy, &gpu_reg, &gpu_cum);
    let expl = measure_exploitability(&cpu_proxy, &tree, &game, np);
    (expl, elapsed_s)
}

#[test]
#[ignore = "Phase 1.A validation: pruned-vs-unpruned floor match in full game"]
fn pruning_floor_match_3p_nh8() {
    let nh = 8usize;
    let np = 3usize;
    let n_iters = 100u32;
    let pot_total = 30.0f32;

    eprintln!("\n=== Phase 1.A validation: pruned floor must match unpruned in FULL game ===\n");
    eprintln!("Setup: 3-player nh={}, asymmetric contribs [10,5,5], {} iters", nh, n_iters);
    std::io::stderr().flush().ok();

    eprintln!("\nRun 1: UNPRUNED baseline");
    let (expl_off, t_off) = run_and_get_floor(n_iters, None, nh, np);
    let pct_off = expl_off / pot_total * 100.0;
    eprintln!("  Final exploitability: {:.6}% of pot (= {:.4})", pct_off, expl_off);
    eprintln!("  Wall-clock: {:.2}s", t_off);

    eprintln!("\nRun 2: PRUNED (threshold=-1000, stride=20)");
    let (expl_on, t_on) = run_and_get_floor(n_iters, Some((-1000.0, 20)), nh, np);
    let pct_on = expl_on / pot_total * 100.0;
    eprintln!("  Final exploitability: {:.6}% of pot (= {:.4})", pct_on, expl_on);
    eprintln!("  Wall-clock: {:.2}s (vs unpruned {:.2}s → speedup {:.2}x)",
        t_on, t_off, t_off / t_on);

    eprintln!();
    eprintln!("=== FLOOR MATCH ANALYSIS ===");
    let abs_diff = (expl_on - expl_off).abs();
    let rel_diff_pct = if expl_off.abs() > 1e-6 { abs_diff / expl_off.abs() * 100.0 } else { 0.0 };
    eprintln!("  Unpruned floor: {:.6}% of pot", pct_off);
    eprintln!("  Pruned floor  : {:.6}% of pot", pct_on);
    eprintln!("  Absolute diff : {:.6}", abs_diff);
    eprintln!("  Relative diff : {:.2}%", rel_diff_pct);
    eprintln!();

    // The gate: floor difference must be small. Some difference is expected
    // from f32 noise + stochastic re-enable timing. Pluribus reports their
    // pruning is empirically behavior-preserving within solver noise.
    // Threshold: allow up to 10% relative drift, OR if floor is essentially
    // zero (< 0.01% of pot), absolute diff < 0.05% of pot.
    let absolute_pct_diff = (pct_on - pct_off).abs();
    if pct_off < 0.01 {
        assert!(absolute_pct_diff < 0.05,
            "Pruned floor drifted: unpruned {:.6}%, pruned {:.6}%, abs diff {:.6}%",
            pct_off, pct_on, absolute_pct_diff);
        eprintln!("  ✓ Both floors near zero, absolute drift {:.6}% < 0.05% gate", absolute_pct_diff);
    } else {
        assert!(rel_diff_pct < 10.0,
            "Pruned floor drifted: unpruned {:.6}%, pruned {:.6}%, rel diff {:.2}%",
            pct_off, pct_on, rel_diff_pct);
        eprintln!("  ✓ Relative drift {:.2}% < 10% gate", rel_diff_pct);
    }
    eprintln!("  PASS: pruning is behavior-preserving on this small game.");
}
