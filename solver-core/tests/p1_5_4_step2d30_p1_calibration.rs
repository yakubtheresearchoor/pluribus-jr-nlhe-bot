// P1.1: Calibrate negative-regret pruning threshold to OUR regret scale.
//
// The Pluribus paper uses -300,000,000 because their cumulative regrets
// integrate over millions of iters with their specific magnitudes. Our scale
// will differ — we use DCFR (alpha=1.5/beta=0/gamma=2) and run far fewer
// iters at the per-flop blueprint level. Picking the paper's threshold
// blindly would either prune nothing (too low) or prune everything (too
// high).
//
// METHOD
// Run 6-max CFR for representative number of iters, download d_regrets,
// compute the distribution of negative regrets, and pick threshold as a
// quantile that prunes the bottom ~50-70% of negative regrets per Pluribus
// guidance (low enough that most negative-regret actions are skipped, high
// enough that nearly-zero negative-regrets stay active in case they recover).
//
// OUTPUT
// - Histogram of negative regret values across iter checkpoints
// - Suggested threshold based on distribution percentiles
// - Sanity check: how many regrets would be pruned per-iter at each
//   candidate threshold

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

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
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

#[test]
#[ignore = "P1.1: calibrate pruning threshold to our regret scale (~5 min at nh=14)"]
fn p1_calibrate_pruning_threshold() {
    let nh = 14usize;
    let (tree, table) = build_6p(nh);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    eprintln!("\n=== P1.1: Pruning threshold calibration ===");
    eprintln!("6-max nh={} tree_nodes={}", nh, tree.num_nodes());
    eprintln!("DCFR: alpha=1.5, beta=0, gamma=2. Pluribus threshold is -300M;");
    eprintln!("our scale is different — measuring it directly.\n");

    let checkpoints = [10u32, 50, 100, 200];
    let mut prev = 0u32;

    for &cp in &checkpoints {
        let delta = cp - prev;
        gpu.run(&ctx, &tree, &game, delta);
        prev = cp;

        let regrets = gpu.download_regrets();
        let total = regrets.len();

        // Count distribution: positives, zeros, negatives. For negatives,
        // tabulate percentile breakdown.
        let mut neg: Vec<f32> = regrets.iter().filter(|&&v| v < 0.0).copied().collect();
        let n_pos = regrets.iter().filter(|&&v| v > 0.0).count();
        let n_zero = regrets.iter().filter(|&&v| v == 0.0).count();
        let n_neg = neg.len();
        let neg_frac = n_neg as f32 / total as f32;

        eprintln!("── iter {} ──", cp);
        eprintln!("  regrets: {} total ({} pos, {} zero, {} neg = {:.1}%)",
            total, n_pos, n_zero, n_neg, neg_frac * 100.0);

        if neg.is_empty() {
            eprintln!("  (no negative regrets — pruning would have no effect)");
            continue;
        }

        // Sort ascending (most negative first).
        neg.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f32| -> f32 {
            let i = ((neg.len() - 1) as f32 * q).round() as usize;
            neg[i.min(neg.len() - 1)]
        };
        // Percentiles of negative regrets (0% = most negative, 100% = least negative).
        eprintln!("  neg regret distribution:");
        eprintln!("    min  (0%):  {:>12.4}", neg[0]);
        eprintln!("    p10:        {:>12.4}", p(0.10));
        eprintln!("    p25:        {:>12.4}", p(0.25));
        eprintln!("    median:     {:>12.4}", p(0.50));
        eprintln!("    p75:        {:>12.4}", p(0.75));
        eprintln!("    p90:        {:>12.4}", p(0.90));
        eprintln!("    max (100%): {:>12.4}", neg[neg.len() - 1]);

        // Candidate thresholds: would prune everything below threshold.
        // For each candidate, count how many (action, hand) regrets get pruned.
        let candidates = [-1000.0f32, -100.0, -10.0, -1.0, -0.1, -0.01, -0.001];
        eprintln!("  prune fraction by threshold:");
        for &thr in &candidates {
            let pruned = neg.iter().filter(|&&v| v < thr).count();
            let frac_of_neg = pruned as f32 / neg.len() as f32;
            let frac_of_total = pruned as f32 / total as f32;
            eprintln!("    thr={:>10.4}: prunes {:>8} ({:.1}% of neg, {:.1}% of all)",
                thr, pruned, frac_of_neg * 100.0, frac_of_total * 100.0);
        }
        eprintln!();
    }

    // Recommendation: pick threshold at p50-p70 of the late-iter negative
    // distribution. This prunes the "deeply negative" half — actions CFR
    // has confidently dismissed — while keeping nearly-zero negatives
    // active so they can recover.
    let final_regrets = gpu.download_regrets();
    let mut final_neg: Vec<f32> = final_regrets.iter().filter(|&&v| v < 0.0).copied().collect();
    final_neg.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !final_neg.is_empty() {
        let p50 = final_neg[final_neg.len() / 2];
        let p70 = final_neg[(final_neg.len() as f32 * 0.7).round() as usize];
        eprintln!("=== RECOMMENDED THRESHOLD (post-iter-{}) ===", checkpoints[checkpoints.len() - 1]);
        eprintln!("    p50 (prunes 50% of negative regrets): {:.4}", p50);
        eprintln!("    p70 (prunes 70% of negative regrets): {:.4}", p70);
        eprintln!("    Pluribus equivalent in our scale ≈ p50-p70.");
        eprintln!("    Suggested initial value: {:.4} (between p50 and p70)", (p50 + p70) / 2.0);
    }
}
