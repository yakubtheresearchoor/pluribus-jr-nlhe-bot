// Steady-state per-iter cost measurement at a single nh, into the
// non-uniform-regret regime where the baseline actually runs.
//
// Prior tests gave iter-1 data (clean and reproducible) at multiple nh
// values, but iter-1 is the uniform-reach iteration — the cheapest and
// least representative case. The nh=12 sample of (iter1=84s, iter2=654s,
// iter3=275s) showed a 7.8x spread inside 3 iters that was filed as
// "variance to characterize later" while the baseline projection was
// computed from iter-1 alone — pricing the multi-day run at the
// cheapest data point.
//
// This test runs many iters at nh=12 to measure ACTUAL steady-state
// per-iter cost, the number that prices the baseline. Reports the full
// per-iter distribution (mean, median, p75, p90, max, min, stddev) so
// the baseline can be priced off a representative central tendency
// rather than the optimistic tail.
//
// nh=12 chosen because: 30 iters in ~2-4 hours wall-clock fits a single
// session, prior data exists for comparison, and it's large enough to
// be in the "real cost" regime (the nh=8 convergence test converges in
// 5 iters to a check-down equilibrium and so never exercises non-
// uniform-regret steady state).

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

fn percentile(times: &[f64], pct: f64) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() - 1) as f64 * pct / 100.0).round() as usize;
    sorted[idx]
}

#[test]
#[ignore = "multi-hour steady-state measurement; run with --ignored"]
fn steady_state_nh12_30iters() {
    let nh = 12usize;
    let n_iters = 30usize;

    eprintln!("\n=== Steady-state per-iter cost at nh={} over {} iters ===", nh, n_iters);
    eprintln!("Measures actual cost of iterations in the non-uniform regret regime");
    eprintln!("(iter-1 alone is the cheapest unrepresentative case).");
    eprintln!();

    let (tree, table) = build_6p_asymmetric_table(nh);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    eprintln!("Tree: {} nodes, {} MB regrets at nh={}",
        tree.num_nodes(), tree.num_nodes() * nh * 4 / 1024 / 1024, nh);
    std::io::stderr().flush().ok();

    // Warmup
    let t_warm = Instant::now();
    gpu.run(&ctx, &tree, &game, 1);
    let warm_s = t_warm.elapsed().as_secs_f64();
    eprintln!("Warmup iter: {:.2} s", warm_s);
    std::io::stderr().flush().ok();

    // Many iters
    let mut times = Vec::with_capacity(n_iters);
    let t_total_start = Instant::now();
    for i in 0..n_iters {
        let t = Instant::now();
        gpu.run(&ctx, &tree, &game, 1);
        let s = t.elapsed().as_secs_f64();
        times.push(s);
        eprintln!("  iter {:3}: {:.2} s  (total {:.1} min)",
            i + 1, s, t_total_start.elapsed().as_secs_f64() / 60.0);
        std::io::stderr().flush().ok();
    }

    // Distribution stats
    let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
    let var: f64 = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let stddev = var.sqrt();
    let min = *times.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max = *times.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let p25 = percentile(&times, 25.0);
    let p50 = percentile(&times, 50.0);
    let p75 = percentile(&times, 75.0);
    let p90 = percentile(&times, 90.0);
    let p95 = percentile(&times, 95.0);

    eprintln!();
    eprintln!("=== Steady-state distribution (nh={}, {} iters) ===", nh, n_iters);
    eprintln!("  min     : {:7.2} s", min);
    eprintln!("  p25     : {:7.2} s", p25);
    eprintln!("  median  : {:7.2} s", p50);
    eprintln!("  mean    : {:7.2} s", mean);
    eprintln!("  p75     : {:7.2} s", p75);
    eprintln!("  p90     : {:7.2} s", p90);
    eprintln!("  p95     : {:7.2} s", p95);
    eprintln!("  max     : {:7.2} s", max);
    eprintln!("  stddev  : {:7.2} s", stddev);
    eprintln!("  spread (max/min): {:.2}x", max / min);

    // Trend check: is later half slower (rich regime) than early half?
    let half = n_iters / 2;
    let early_mean: f64 = times.iter().take(half).sum::<f64>() / half as f64;
    let late_mean: f64 = times.iter().skip(half).sum::<f64>() / (n_iters - half) as f64;
    eprintln!();
    eprintln!("Trend across run:");
    eprintln!("  early-half mean (iters 1-{}): {:.2} s", half, early_mean);
    eprintln!("  late-half  mean (iters {}-{}): {:.2} s", half + 1, n_iters, late_mean);
    eprintln!("  late/early ratio: {:.2}x", late_mean / early_mean);

    // Project nh=50 baseline from MEDIAN steady-state, not iter-1
    eprintln!();
    eprintln!("=== nh=50 baseline projection from steady-state ===");
    eprintln!("(scaling exponent ~1.84 from iter-1 nh=12→nh=16; using nh=12 steady-state)");
    let exp_optimistic = 1.7;
    let exp_observed = 1.84;
    let exp_pessimistic = 3.0;
    for &(name, per_iter) in &[
        ("median (p50)", p50),
        ("p75 (3/4 iters slower than this)", p75),
        ("p90 (1/10 iters slower than this)", p90),
        ("mean", mean),
        ("max (worst observed)", max),
    ] {
        eprintln!("  Anchored on {}:", name);
        for &(label, e) in &[("optimistic nh^1.7", exp_optimistic), ("observed nh^1.84", exp_observed), ("pessimistic nh^3.0", exp_pessimistic)] {
            let proj = per_iter * (50.0_f64 / nh as f64).powf(e);
            let h_200 = proj * 200.0 / 3600.0;
            eprintln!("    {}: {:.0} s/iter → 200 iters = {:.1} h = {:.2} d",
                label, proj, h_200, h_200 / 24.0);
        }
    }

    // Sanity assert
    assert!(times.iter().all(|t| t.is_finite() && *t > 0.0),
        "Iter timing produced invalid values");
}
