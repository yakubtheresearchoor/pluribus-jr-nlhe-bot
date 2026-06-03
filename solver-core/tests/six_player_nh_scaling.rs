// Empirical throughput scaling at varying nh on the SAME tree structure.
//
// The baseline projection from nh=8 to nh=50 has a wide bracket (5.5 hours
// to 1.1 days) depending on whether tree work or showdown dominates the
// current 25s/iter. This test pins down the scaling exponent by measuring
// per-iter time at nh ∈ {4, 8, 12, 16} with identical tree config, then
// extrapolating to nh=50.
//
// Method: build the asymmetric-contrib 6p tree at each nh, run 1 GPU iter
// (after one warmup iter to settle compile/cache effects), report time.
// Tree node count and terminal count are independent of nh, so this
// isolates the per-hand and per-terminal scaling.

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
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

#[test]
#[ignore = "runs ~5-15 min; use with --ignored"]
fn nh_scaling_for_baseline_projection() {
    let ctx = MetalContext::new().expect("Metal");

    eprintln!("\n=== nh scaling on 6p asymmetric-contrib tree ===");
    eprintln!("Tree structure independent of nh; only per-node + per-terminal work scales.");
    eprintln!();
    eprintln!("{:>4} | {:>10} | {:>15} | {:>15} | {:>10}", "nh", "tree_nodes", "warmup_iter (s)", "timed_iter (s)", "ratio_vs_8");
    eprintln!("{}", "-".repeat(75));

    let nh_values = [4usize, 8, 12];  // 16 omitted; tight on per-iter time
    let mut per_iter_times: Vec<(usize, f64)> = Vec::new();

    for &nh in &nh_values {
        let (tree, table) = build_6p_asymmetric_table(nh);
        let game = FlopStartGame::new(table);
        let cpu = FlopStartVectorCfr::new(&tree, &game.table());
        let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

        // Warmup (kernel compile, cache warm)
        let t_warm = Instant::now();
        gpu.run(&ctx, &tree, &game, 1);
        let warm_s = t_warm.elapsed().as_secs_f64();

        // Timed iter
        let t_iter = Instant::now();
        gpu.run(&ctx, &tree, &game, 1);
        let iter_s = t_iter.elapsed().as_secs_f64();

        per_iter_times.push((nh, iter_s));
        eprintln!("{:>4} | {:>10} | {:>15.3} | {:>15.3} |",
            nh, tree.num_nodes(), warm_s, iter_s);
    }

    // Recompute with ratios
    let baseline_8 = per_iter_times.iter().find(|&&(n, _)| n == 8).map(|&(_, t)| t).unwrap_or(1.0);
    eprintln!();
    eprintln!("{:>4} | {:>15} | {:>15} | {:>15}", "nh", "per_iter (s)", "ratio_vs_8", "nh^3_predicted");
    eprintln!("{}", "-".repeat(60));
    for &(nh, t) in &per_iter_times {
        let ratio = t / baseline_8;
        let nh3_pred = (nh as f64 / 8.0).powi(3);
        eprintln!("{:>4} | {:>15.3} | {:>15.3} | {:>15.3}", nh, t, ratio, nh3_pred);
    }

    // Project to nh=50
    eprintln!();
    eprintln!("=== Projection to nh=50 ===");
    // Linear scaling vs nh
    let t_8 = baseline_8;
    let lin_50 = t_8 * (50.0 / 8.0);
    let cubic_50 = t_8 * (50.0_f64 / 8.0).powi(3);

    // Fit a + b*nh^3 model to the measured points
    // t(nh) = a + b*nh^3; use two points (nh=4, nh=12) to fit a, b
    if per_iter_times.len() >= 2 {
        let mut sorted = per_iter_times.clone();
        sorted.sort_by_key(|&(n, _)| n);
        let (nh_lo, t_lo) = sorted[0];
        let (nh_hi, t_hi) = sorted[sorted.len() - 1];
        let x_lo = (nh_lo as f64).powi(3);
        let x_hi = (nh_hi as f64).powi(3);
        let b = (t_hi - t_lo) / (x_hi - x_lo);
        let a = t_lo - b * x_lo;
        let proj_50 = a + b * 50.0_f64.powi(3);
        eprintln!("Fit model: t(nh) = {:.4} + {:.6e} * nh^3", a, b);
        eprintln!("Constants (a, presumably tree work): {:.3} s/iter", a);
        eprintln!("nh^3 coefficient (b, presumably showdown): {:.6e}", b);
        eprintln!();
        eprintln!("Projected per-iter at nh=50: {:.1} s", proj_50);
        let h_per_iter = proj_50 / 3600.0;
        eprintln!("  200 iters: {:.2} h = {:.2} days", proj_50 * 200.0 / 3600.0, proj_50 * 200.0 / 86400.0);
        eprintln!("  with 4x isomorphism: {:.2} h = {:.2} days",
            proj_50 * 200.0 / 4.0 / 3600.0, proj_50 * 200.0 / 4.0 / 86400.0);
        let _ = h_per_iter;
    }
    eprintln!();
    eprintln!("Sanity bounds:");
    eprintln!("  Pure linear scaling nh=50: {:.1} s/iter", lin_50);
    eprintln!("  Pure cubic   scaling nh=50: {:.1} s/iter", cubic_50);
    eprintln!("  → 200 iters lin: {:.2} h ({:.2} h with 4x iso)", lin_50 * 200.0 / 3600.0, lin_50 * 200.0 / 14400.0);
    eprintln!("  → 200 iters cubic: {:.2} days ({:.2} days with 4x iso)", cubic_50 * 200.0 / 86400.0, cubic_50 * 200.0 / 4.0 / 86400.0);
}
