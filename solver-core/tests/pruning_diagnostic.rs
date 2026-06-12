// Phase 1.A diagnostic — the FIRST-CLASS output of Option A.
//
// Per user spec: "implement Option A with the regret-distribution measurement
// as a first-class output, not just the skip... Without that instrumentation,
// Option A is just a small win and you still do not know Option B's value.
// With it, Option A answers the question that decides Option B."
//
// This test:
// 1. Runs CFR with pruning OFF and ON, measures per-iter wall-clock + speedup
// 2. At checkpoints, downloads regret buffers and computes per-street
//    distribution stats (min, p25, p50, p75, p90, max, fraction-below-threshold)
// 3. Reports: how many actions would Option B prune, broken down by street and
//    convergence stage. THIS is the data that prices Option B.

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

fn build_6p_asymmetric_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 6u8;
    let num_opp = 5usize;
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i*2] = c1; hand_cards[i*2+1] = c2;
    }
    let mut conflict = vec![0u8; nh*nh];
    for i in 0..nh { for j in 0..nh {
        if i == j { conflict[i*nh+j] = 1; continue; }
        let (a1,a2) = index_to_card_pair(chosen[i] as usize);
        let (b1,b2) = index_to_card_pair(chosen[j] as usize);
        if a1==b1||a1==b2||a2==b1||a2==b2 { conflict[i*nh+j] = 1; }
    }}
    let mut hr = vec![0u16; nh];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        hr[i] = h.evaluate_internal() as u16;
    }
    let tc = vec![card_from_str("3c").unwrap() as u8];
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    rd[tc[0] as usize] = vec![card_from_str("5s").unwrap() as u8];
    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &t in &tc {
        for (i, &hi) in chosen.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let tm = board_mask | (1u64 << t);
            if tm & (1u64 << c1) != 0 || tm & (1u64 << c2) != 0 { continue; }
            let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
            for &bc in &board { h = h.add_card(bc as usize); }
            h = h.add_card(t as usize);
            turn_ranks[t as usize * nh + i] = h.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            let off = t as usize * num_opp * nh + oi * nh;
            for h in 0..nh { turn_sorted_str[off + h] = items[h].0; turn_sorted_idx[off + h] = items[h].1; }
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &t in &tc {
        let tm = board_mask | (1u64 << t);
        for &r in &rd[t as usize] {
            let fm = tm | (1u64 << r);
            for (i, &hi) in chosen.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if fm & (1u64 << c1) != 0 || fm & (1u64 << c2) != 0 { continue; }
                let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
                for &bc in &board { h = h.add_card(bc as usize); }
                h = h.add_card(t as usize).add_card(r as usize);
                river_ranks[t as usize * 52 * nh + r as usize * nh + i] = h.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16)).collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; num_players as usize];
    fn enum_nc(player: usize, np: usize, nh: usize, combined: u64, hand_cards: &[u8], weight: f64) -> f64 {
        if player == np { return weight; }
        let mut total = 0.0;
        for h in 0..nh {
            let m = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
            if combined & m != 0 { continue; }
            total += enum_nc(player + 1, np, nh, combined | m, hand_cards, weight);
        }
        total
    }
    let nc = enum_nc(0, num_players as usize, nh, 0, &hand_cards[..], 1.0);
    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players, num_combinations: nc, river_decks: rd,
    };
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

#[derive(Debug, Default, Clone)]
struct ZoneStats {
    name: String,
    n: usize,
    nonzero: usize,
    min: f32,
    p25: f32,
    p50: f32,
    p75: f32,
    p90: f32,
    max: f32,
    mean: f32,
    fraction_below_threshold: f64,
    fraction_negative: f64,
}

fn percentile(sorted: &[f32], pct: f64) -> f32 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() - 1) as f64 * pct / 100.0).round() as usize;
    sorted[idx]
}

fn compute_zone_stats(name: &str, regrets: &[f32], threshold: f32) -> ZoneStats {
    let mut sorted: Vec<f32> = regrets.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let nonzero = sorted.iter().filter(|&&v| v != 0.0).count();
    let mean = if n > 0 { sorted.iter().sum::<f32>() / n as f32 } else { 0.0 };
    let frac_below = if n > 0 {
        sorted.iter().filter(|&&v| v < threshold).count() as f64 / n as f64
    } else { 0.0 };
    let frac_neg = if n > 0 {
        sorted.iter().filter(|&&v| v < 0.0).count() as f64 / n as f64
    } else { 0.0 };
    ZoneStats {
        name: name.to_string(), n, nonzero,
        min: sorted.first().copied().unwrap_or(0.0),
        p25: percentile(&sorted, 25.0),
        p50: percentile(&sorted, 50.0),
        p75: percentile(&sorted, 75.0),
        p90: percentile(&sorted, 90.0),
        max: sorted.last().copied().unwrap_or(0.0),
        mean,
        fraction_below_threshold: frac_below,
        fraction_negative: frac_neg,
    }
}

fn print_zone_stats(s: &ZoneStats) {
    eprintln!("    {:6} n={:>10} nonzero={:>10} | min/p25/p50/p75/p90/max: {:>10.1}/{:>9.2}/{:>9.2}/{:>9.2}/{:>9.2}/{:>10.1} | mean: {:>9.2}",
        s.name, s.n, s.nonzero, s.min, s.p25, s.p50, s.p75, s.p90, s.max, s.mean);
    eprintln!("    {:6}   below_threshold: {:.3}%   negative: {:.3}%",
        "", s.fraction_below_threshold * 100.0, s.fraction_negative * 100.0);
}

#[test]
#[ignore = "Phase 1.A diagnostic: measures prunable fraction (prices Option B)"]
fn pruning_diagnostic_6p_nh8() {
    let nh = 8usize;
    let n_iters = 30u32;
    let checkpoints: &[u32] = &[1, 3, 10, 30];
    let threshold: f32 = -1000.0;
    let pruning_stride: u32 = 20;

    eprintln!("\n=== Phase 1.A diagnostic: regret distribution + prunable fraction ===");
    eprintln!("This is the FIRST-CLASS OUTPUT — the data that prices Option B.\n");
    eprintln!("Setup: 6p nh={}, asymmetric contribs [10,5,5,5,5,5], {} iters", nh, n_iters);
    eprintln!("Pruning threshold: {} (per-action,hand regret below this skips update)", threshold);
    eprintln!("Pruning stride: every {}th iter re-enables all (Pluribus-style)\n", pruning_stride);
    std::io::stderr().flush().ok();

    let (tree, table) = build_6p_asymmetric_table(nh);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");

    // ─── Run 1: pruning OFF (baseline timing) ───
    eprintln!("=== Run 1: pruning OFF (baseline timing reference) ===");
    let mut gpu_off = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    let t0 = Instant::now();
    gpu_off.run(&ctx, &tree, &game, n_iters);
    let t_off_total = t0.elapsed().as_secs_f64();
    eprintln!("  {} iters: {:.2}s total ({:.2}s/iter avg)\n", n_iters, t_off_total, t_off_total / n_iters as f64);

    // ─── Run 2: pruning ON, with regret-distribution measurement at checkpoints ───
    eprintln!("=== Run 2: pruning ON, regret distribution at checkpoints ===");
    let mut gpu_on = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu_on.set_pruning(true, threshold, pruning_stride);

    let fl_len = cpu.regrets_flop().len();
    let tl_len = cpu.regrets_turn().len();
    let rv_len = cpu.regrets_river().len();
    eprintln!("Zone sizes: flop={}, turn={}, river={}", fl_len, tl_len, rv_len);
    eprintln!();

    let mut prev = 0u32;
    let t_total = Instant::now();
    let mut per_checkpoint_times: Vec<(u32, f64)> = Vec::new();

    for &cp in checkpoints {
        let t_batch = Instant::now();
        gpu_on.run(&ctx, &tree, &game, cp - prev);
        let batch_s = t_batch.elapsed().as_secs_f64();
        per_checkpoint_times.push((cp, batch_s));
        prev = cp;
        eprintln!("--- After iter {} ({:.2}s batch, {:.1}min total) ---",
            cp, batch_s, t_total.elapsed().as_secs_f64() / 60.0);

        let regs_all = gpu_on.download_regrets();
        let stats_flop = compute_zone_stats("FLOP",
            &regs_all[0..fl_len], threshold);
        let stats_turn = compute_zone_stats("TURN",
            &regs_all[fl_len..fl_len + tl_len], threshold);
        let stats_river = compute_zone_stats("RIVER",
            &regs_all[fl_len + tl_len..fl_len + tl_len + rv_len.min(regs_all.len() - fl_len - tl_len)],
            threshold);

        print_zone_stats(&stats_flop);
        print_zone_stats(&stats_turn);
        print_zone_stats(&stats_river);

        let total_n = stats_flop.n + stats_turn.n + stats_river.n;
        let total_below = (stats_flop.fraction_below_threshold * stats_flop.n as f64
            + stats_turn.fraction_below_threshold * stats_turn.n as f64
            + stats_river.fraction_below_threshold * stats_river.n as f64) / total_n as f64;
        eprintln!("    OVERALL prunable-fraction: {:.3}% of (action, hand) pairs below threshold", total_below * 100.0);
        std::io::stderr().flush().ok();
    }
    let t_on_total = t_total.elapsed().as_secs_f64();

    // ─── Speedup measurement ───
    eprintln!();
    eprintln!("=== Speedup measurement (Option A) ===");
    eprintln!("  Unpruned total: {:.2}s ({:.2}s/iter avg)", t_off_total, t_off_total / n_iters as f64);
    eprintln!("  Pruned total:   {:.2}s ({:.2}s/iter avg)", t_on_total, t_on_total / n_iters as f64);
    eprintln!("  Speedup: {:.2}x", t_off_total / t_on_total);
    eprintln!();
    eprintln!("=== What this tells us about Option B ===");
    eprintln!("  Option B (subtree-skip) gains scale with the prunable-fraction:");
    eprintln!("  if X% of (action, hand) pairs are below threshold at steady state,");
    eprintln!("  Option B saves roughly X% of river bottom-up cost = X% × 84% total speedup.");
    eprintln!();
    eprintln!("  If steady-state prunable >50%, Option B would deliver >2× total speedup");
    eprintln!("  → worth building (multi-day implementation, full Pluribus class).");
    eprintln!();
    eprintln!("  If steady-state prunable <10%, Option B would deliver <1.1× total speedup");
    eprintln!("  → not worth building, baseline already feasible.");
    eprintln!();
    eprintln!("  In between (10-50%): judgment call based on baseline runtime + budget.");
}
