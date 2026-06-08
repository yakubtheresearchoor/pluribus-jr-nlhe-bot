// M2: 6-max per-terminal cost curve.
//
// Measures bottom_up_river time at np=6, three nh values. Divides by terminal
// count to get per-terminal cost. Fits log-log to derive ACTUAL scaling
// exponent (the conversation has been assuming nh^4 from O(nh^(K-1)) factored
// math; empirical exponent may differ due to constant factors, memory layout,
// or thread divergence at small nh).
//
// Measured AT 6-max — no extrapolation from HU/4p/5p. The whole point.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_6p_table(nh: usize) -> (FlatTree, FlopChanceTable) {
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

    let mut ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
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
    let starting_pot: i32 = (np as i32) * 5;
    let stacks: Vec<i32> = vec![100; np as usize];
    let contribs: Vec<i32> = vec![5; np as usize];
    let config = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot,
        starting_stacks: stacks,
        initial_contributions: contribs,
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

fn count_terminals(tree: &FlatTree) -> (usize, usize, usize) {
    let mut showdown = 0;
    let mut fold = 0;
    for (idx, node) in tree.nodes.iter().enumerate() {
        if node.is_terminal() {
            let fm = tree.get_folded_mask(idx);
            if fm == 0 { showdown += 1; } else { fold += 1; }
        }
    }
    (showdown + fold, showdown, fold)
}

#[test]
#[ignore = "M2: 6-max per-terminal cost curve — slow (~3-5 min)"]
fn m2_6max_per_terminal_cost_curve() {
    let nh_points: [usize; 3] = [8, 14, 20];
    let warmup_iters: u32 = 2;
    let measure_iters: u32 = 3;

    let ctx = MetalContext::new().expect("Metal");
    eprintln!("\n=== M2: 6-max per-terminal cost curve ===");
    eprintln!("Measuring bottom_up_river at np=6, nh ∈ {:?}", nh_points);
    eprintln!("warmup={} iters, measure={} iters per nh\n", warmup_iters, measure_iters);

    let mut results: Vec<(usize, usize, f64, f64)> = Vec::new();
    // (nh, terminals, bottom_up_river_total_ms, per_terminal_us_per_iter)

    for &nh in &nh_points {
        let (tree, table) = build_6p_table(nh);
        let (total_terminals, showdown, fold) = count_terminals(&tree);
        eprintln!("── nh = {} ──", nh);
        eprintln!("    tree: {} nodes total, {} terminals ({} showdown, {} fold)",
            tree.num_nodes(), total_terminals, showdown, fold);

        let game = FlopStartGame::new(table);
        let cpu = FlopStartVectorCfr::new(&tree, &game.table());
        let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

        // Warmup
        gpu.run(&ctx, &tree, &game, warmup_iters);

        // Profile
        let prof = gpu.run_profiled(&ctx, &tree, &game, measure_iters);
        let total_ms = prof.total.as_secs_f64() * 1000.0;
        let bu_river_ms = prof.bottom_up_river.as_secs_f64() * 1000.0;
        let bu_turn_ms  = prof.bottom_up_turn.as_secs_f64() * 1000.0;
        let bu_flop_ms  = prof.bottom_up_flop.as_secs_f64() * 1000.0;

        let per_iter_total_ms = total_ms / measure_iters as f64;
        let per_iter_bu_river_ms = bu_river_ms / measure_iters as f64;
        let per_terminal_us = (per_iter_bu_river_ms * 1000.0) / total_terminals as f64;

        eprintln!("    {} iters profiled, total {:.1} ms ({:.1} ms/iter)",
            measure_iters, total_ms, per_iter_total_ms);
        eprintln!("    bottom_up_river:  {:.2} ms/iter  ({:.2} % of iter)",
            per_iter_bu_river_ms, per_iter_bu_river_ms / per_iter_total_ms * 100.0);
        eprintln!("    bottom_up_turn:   {:.2} ms/iter", bu_turn_ms / measure_iters as f64);
        eprintln!("    bottom_up_flop:   {:.2} ms/iter", bu_flop_ms / measure_iters as f64);
        eprintln!("    per_terminal:     {:.3} µs/iter  ({} terminals)",
            per_terminal_us, total_terminals);

        results.push((nh, total_terminals, per_iter_bu_river_ms, per_terminal_us));
        eprintln!();
    }

    // Log-log fit: ln(t) = ln(C) + k · ln(nh) → k = best fit slope across the 3 points.
    eprintln!("── Log-log scaling fit (per-terminal cost vs nh) ──");
    eprintln!("{:>4}  {:>10}  {:>14}", "nh", "ln(nh)", "ln(µs/term)");
    let mut log_pairs: Vec<(f64, f64)> = Vec::new();
    for &(nh, _, _, per_term_us) in &results {
        let ln_nh = (nh as f64).ln();
        let ln_t = per_term_us.ln();
        eprintln!("{:>4}  {:>10.4}  {:>14.4}", nh, ln_nh, ln_t);
        log_pairs.push((ln_nh, ln_t));
    }

    // Pairwise slopes (each consecutive pair).
    eprintln!("\nPairwise slopes:");
    for i in 1..log_pairs.len() {
        let (x1, y1) = log_pairs[i - 1];
        let (x2, y2) = log_pairs[i];
        let slope = (y2 - y1) / (x2 - x1);
        eprintln!("    nh {} → {}: slope = {:.3}",
            results[i - 1].0, results[i].0, slope);
    }

    // OLS slope across all 3 points.
    let n = log_pairs.len() as f64;
    let mx: f64 = log_pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
    let my: f64 = log_pairs.iter().map(|(_, y)| y).sum::<f64>() / n;
    let num: f64 = log_pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = log_pairs.iter().map(|(x, _)| (x - mx) * (x - mx)).sum();
    let k = num / den;
    let c = my - k * mx;
    let c_real = c.exp();
    eprintln!("\n── OLS fit ──");
    eprintln!("    per_terminal_µs ≈ {:.4} · nh^{:.3}", c_real, k);
    eprintln!("    (assumed nh^4 = 4.0; measured exponent = {:.3})", k);
    eprintln!("\n── Prediction (per-terminal cost) ──");
    for &nh_pred in &[8usize, 14, 20, 32, 50, 80, 100, 150, 200, 500, 1000] {
        let pred = c_real * (nh_pred as f64).powf(k);
        eprintln!("    nh={:>4}: {:>12.3} µs/terminal  (=> {:>10.3} µs at nh^4 model)",
            nh_pred, pred, c_real * (nh_pred as f64).powi(4));
    }
}
