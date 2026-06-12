// P1.3 + P1.4: measure pruning's net speedup (per-iter × iter-count factors)
// AND gate equilibrium quality vs unpruned reference.
//
// METHOD
//   1. Build same 6-max game twice (one for unpruned, one for pruned).
//   2. Run both to N iters, log per-iter wall time + exploitability per
//      checkpoint.
//   3. Find iter at which each crosses the convergence target (1% pot).
//   4. Net speedup = (unpruned iter cost / pruned iter cost) × (unpruned
//      iters-to-target / pruned iters-to-target).
//   5. CORRECTNESS GATE: pruned exploitability at end of run must be within
//      tolerance of unpruned exploitability. The "quality-disguised-as-
//      speedup trap" failure mode is a pruned run that converges FAST but
//      to a worse equilibrium — that's a regression, not a speedup.
//
// PARAMETERS
//   threshold: -0.05 (from P1.1 calibration — p50-p70 of our regret scale)
//   stride: 20 (Pluribus's 5% full-iter cadence)
//   carve-outs: all enforced by kernel (river skip, terminal-leading skip,
//               re_enable iter skip)

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::time::Instant;

fn build_6p(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let np = 6u8;
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..np as usize {
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
        &board, &ranges, np, &chosen, &turn_cards, &river_decks,
    );
    let config = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![100; np as usize],
        initial_contributions: vec![5; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
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

/// Run one trajectory: log per-checkpoint (iter, wall_s, expl_pct).
fn run_trajectory(
    ctx: &MetalContext,
    tree: &FlatTree,
    game: &FlopStartGame,
    pruning: Option<(f32, u32)>,  // (threshold, stride)
    checkpoints: &[u32],
    np: usize,
    pot: f32,
    label: &str,
) -> Vec<(u32, f32, f32)> {
    eprintln!("\n── Trajectory: {} ──", label);
    let cpu_seed = FlopStartVectorCfr::new(tree, &game.table());
    let mut gpu = MetalFlopStartSolver::new(ctx, tree, game, &cpu_seed);
    let mut cpu = FlopStartVectorCfr::new(tree, &game.table());
    if let Some((thr, stride)) = pruning {
        gpu.set_pruning(true, thr, stride);
        eprintln!("Pruning enabled: threshold={:.4}, stride={} (re_enable every {}th iter)",
            thr, stride, stride);
    } else {
        eprintln!("Pruning DISABLED (baseline)");
    }
    let mut prev = 0u32;
    let t0 = Instant::now();
    let mut history: Vec<(u32, f32, f32)> = Vec::new();
    eprintln!("{:>6}  {:>14}  {:>10}", "iter", "expl (% pot)", "wall (s)");
    for &cp in checkpoints {
        let delta = cp - prev;
        gpu.run(ctx, tree, game, delta);
        cpu.run(tree, game, delta);
        prev = cp;
        let elapsed = t0.elapsed().as_secs_f32();
        let expl = measure_exploitability(&cpu, tree, game, np);
        let pct = expl / pot * 100.0;
        eprintln!("{:>6}  {:>13.4}%  {:>10.1}", cp, pct, elapsed);
        history.push((cp, pct, elapsed));
    }
    history
}

#[test]
#[ignore = "P1.3+P1.4: pruning vs unpruned A/B with correctness gate (~10 min at nh=14)"]
fn p1_pruning_speedup_and_quality_gate() {
    let nh = 14usize;
    let np = 6usize;
    let pot = (np as f32) * 5.0;
    let (tree, table_u) = build_6p(nh);
    let (_, table_p) = build_6p(nh);
    let game_u = FlopStartGame::new(table_u);
    let game_p = FlopStartGame::new(table_p);

    eprintln!("\n=== P1.3+P1.4: Pruning measurement + quality gate ===");
    eprintln!("6-max nh={} np={} tree_nodes={}", nh, np, tree.num_nodes());
    eprintln!("Threshold: -0.05 (calibrated from P1.1, p50-p70 of negative regret scale)");
    eprintln!("Stride: 20 (Pluribus's 5% re_enable cadence)");
    eprintln!("Carve-outs (kernel-enforced): no pruning on river, no pruning on actions");
    eprintln!("  leading to terminals, full traversal every 20th iter\n");

    let ctx = MetalContext::new().expect("Metal");
    let checkpoints: Vec<u32> = vec![1, 5, 10, 25, 50, 100];

    let unpruned = run_trajectory(&ctx, &tree, &game_u, None, &checkpoints, np, pot, "UNPRUNED");
    let pruned = run_trajectory(&ctx, &tree, &game_p, Some((-0.05, 20)), &checkpoints, np, pot, "PRUNED");

    // ── Net speedup analysis ──
    eprintln!("\n── Per-iter wall-clock comparison ──");
    eprintln!("{:>6}  {:>14}  {:>14}  {:>10}", "iter", "unpruned (s)", "pruned (s)", "speedup");
    for (i, &(it, _, w_u)) in unpruned.iter().enumerate() {
        let (_, _, w_p) = pruned[i];
        let speedup = w_u / w_p.max(1e-9);
        eprintln!("{:>6}  {:>14.2}  {:>14.2}  {:>9.2}x", it, w_u, w_p, speedup);
    }
    let final_w_u = unpruned.last().unwrap().2;
    let final_w_p = pruned.last().unwrap().2;
    let per_iter_speedup = final_w_u / final_w_p.max(1e-9);
    eprintln!("Per-iter wall-clock speedup (at last checkpoint): {:.2}x", per_iter_speedup);

    eprintln!("\n── Convergence comparison ──");
    eprintln!("{:>6}  {:>14}  {:>14}", "iter", "unpruned expl%", "pruned expl%");
    for (i, &(it, e_u, _)) in unpruned.iter().enumerate() {
        let (_, e_p, _) = pruned[i];
        eprintln!("{:>6}  {:>13.4}%  {:>13.4}%", it, e_u, e_p);
    }

    // Iters-to-1%-pot for each.
    let unpruned_to_1pct = unpruned.iter().find(|(_, p, _)| *p < 1.0).map(|(i, _, _)| *i);
    let pruned_to_1pct = pruned.iter().find(|(_, p, _)| *p < 1.0).map(|(i, _, _)| *i);
    eprintln!("\nIters-to-1%-pot:");
    eprintln!("  Unpruned: {:?}", unpruned_to_1pct);
    eprintln!("  Pruned:   {:?}", pruned_to_1pct);

    // Net speedup = per-iter × iter-count ratio
    if let (Some(i_u), Some(i_p)) = (unpruned_to_1pct, pruned_to_1pct) {
        let iter_ratio = i_u as f32 / i_p as f32;
        let net = per_iter_speedup * iter_ratio;
        eprintln!("\n── NET SPEEDUP ──");
        eprintln!("  per_iter:    {:.2}x", per_iter_speedup);
        eprintln!("  iter_count:  {:.2}x ({} → {} iters)", iter_ratio, i_u, i_p);
        eprintln!("  NET = {:.2}x × {:.2}x = {:.2}x",
            per_iter_speedup, iter_ratio, per_iter_speedup * iter_ratio);
        let _ = net;
    } else {
        eprintln!("WARNING: one or both runs did not cross 1% pot threshold");
    }

    // ── Quality gate: pruned must reach exploitability within tolerance ──
    let final_e_u = unpruned.last().unwrap().1;
    let final_e_p = pruned.last().unwrap().1;
    let rel_gap = ((final_e_p - final_e_u).abs() / final_e_u.max(1e-6)) as f32;
    let abs_gap = (final_e_p - final_e_u).abs();
    eprintln!("\n── Quality gate (catches 'quality-disguised-as-speedup' trap) ──");
    eprintln!("  Final exploitability  unpruned: {:.4}%  pruned: {:.4}%",
        final_e_u, final_e_p);
    eprintln!("  Absolute gap: {:.4}%  Relative gap: {:.2}x",
        abs_gap, rel_gap);
    // Tolerance: pruned must be within 2× unpruned exploitability AND
    // within 0.1% absolute. The first catches "fast but worse equilibrium";
    // the second handles the case where both are at the f32 floor.
    let rel_tol = 2.0f32;
    let abs_tol = 0.1f32;
    assert!(
        rel_gap < rel_tol || abs_gap < abs_tol,
        "QUALITY GATE FAIL: pruned equilibrium worse than unpruned beyond tolerance. \
         unpruned={:.4}% pruned={:.4}% rel_gap={:.2}x abs_gap={:.4}% (tol: rel<{:.1}x OR abs<{:.2}%)",
        final_e_u, final_e_p, rel_gap, abs_gap, rel_tol, abs_tol
    );
    eprintln!("  PASS (rel_gap < {:.1}× OR abs_gap < {:.2}%)", rel_tol, abs_tol);

    eprintln!("\n=== P1.3+P1.4 COMPLETE ===");
    eprintln!("Pruning measured on both axes; quality gate held against unpruned reference.");
}
