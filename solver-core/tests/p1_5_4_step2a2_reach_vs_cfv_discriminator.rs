// Step 2.A.2 trace — reach-vs-CFV discriminator.
//
// The K=4 minimal-asymmetry stratum 1 test fails with regrets diverging
// and the signature `GPU=0.0` at specific entries. Per the lead's sharpened
// trace plan (task #74), the GPU=0.0 signature distinguishes TWO bug
// classes:
//   - Hypothesis A: bug in compute_reach_* under non-uniform initial
//     weights — GPU's reach at the diverging entry IS 0 because reach
//     was computed wrong.
//   - Hypothesis B: bug in CFV computation consuming a correct reach
//     wrongly — GPU's reach at the diverging entry is NON-ZERO but the
//     CFV multiplied it by some wrong term.
//
// This test discriminates: compute reach on both sides, compare zone by
// zone. If reach matches everywhere, the bug is in CFV (Hypothesis B).
// If reach diverges at some zone, the bug is upstream there (Hypothesis A).

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// Same minimal-asymmetry K=4 game that reproduces the bug.
fn build_minimal_asymmetry_game() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["Ah", "Kd", "7c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 2u8;
    let k = 4usize;

    use solver_core::hand::eval::Hand;
    let mut all_with_strength: Vec<(u16, u16)> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        all_with_strength.push((h.evaluate_internal() as u16, idx as u16));
    }
    all_with_strength.sort_by_key(|&(s, _)| s);
    let n = all_with_strength.len();
    let step = n / k;
    let chosen: Vec<u16> = (0..k).map(|i| all_with_strength[i * step].1).collect();

    let mut ranges: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for (rank_idx, &hi) in chosen.iter().enumerate() {
        let strength_frac = rank_idx as f32 / k as f32;
        // MINIMAL ASYMMETRY: P0=uniform 1.0, P1=half 1.0/half 0.5.
        let p0_weight = 1.0_f32;
        let p1_weight = if strength_frac >= 0.5 { 1.0_f32 } else { 0.5_f32 };
        let (c1, c2) = index_to_card_pair(hi as usize);
        let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
        let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
        ranges[0][pair_idx] = p0_weight;
        ranges[1][pair_idx] = p1_weight;
    }
    let turn_cards: Vec<u8> = vec![
        card_from_str("Td").unwrap() as u8,
        card_from_str("3s").unwrap() as u8,
    ];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("4h").unwrap() as u8,
        card_from_str("Qc").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("2s").unwrap() as u8,
        card_from_str("Js").unwrap() as u8,
    ];

    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, num_players, &chosen, &turn_cards, &river_decks,
    );
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 6,
        starting_stacks: vec![50, 50], initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = FlopStartGame::new(table);
    (tree, game)
}

fn max_abs(a: &[f32], b: &[f32], label: &str) -> (f32, usize) {
    let mut max_d = 0.0f32;
    let mut worst_idx = 0usize;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        if d > max_d { max_d = d; worst_idx = i; }
    }
    eprintln!("  {}: len {}/{}  max_abs={:.6e} at idx {} (CPU={:.6} GPU={:.6})",
        label, a.len(), b.len(), max_d, worst_idx,
        a.get(worst_idx).copied().unwrap_or(0.0),
        b.get(worst_idx).copied().unwrap_or(0.0));
    (max_d, worst_idx)
}

#[test]
#[ignore = "2.A.2 trace: reach-vs-CFV discriminator on K=4 minimal-asymmetry game"]
fn reach_per_zone_cpu_vs_gpu_at_iter1_under_non_uniform() {
    let (tree, game) = build_minimal_asymmetry_game();
    let ctx = MetalContext::new().expect("Metal");
    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    eprintln!("\n=== REACH DISCRIMINATOR (K=4 minimal-asymmetry, iter 1) ===");
    eprintln!("Tree: {} nodes, n_turn={} max_river={}",
        tree.num_nodes(), gpu.n_turn(), gpu.max_river());

    // Strategies — both start with all-zero regrets so both produce
    // uniform 0.5. Compute on BOTH sides (test-bug fix 2026-06: CPU's
    // compute_reach_flop multiplies by strategy_flop, so the strategy
    // must be populated first or reach degenerates to zero).
    cpu.compute_all_strategies(&tree);
    gpu.compute_all_strategies(&ctx);

    // ════════════════════════════════════════════════════════════════
    // FLOP REACH
    // ════════════════════════════════════════════════════════════════
    eprintln!("\nFLOP reach:");
    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);
    gpu.compute_reach_flop(&ctx);
    let gpu_full_reach = gpu.download_reach();
    let flop_len = cpu_flop_reach.len();
    let (flop_diff, _) = max_abs(&cpu_flop_reach, &gpu_full_reach[..flop_len], "flop_reach");

    // ════════════════════════════════════════════════════════════════
    // TURN REACH (for tc=0 only — sufficient to discriminate)
    // ════════════════════════════════════════════════════════════════
    eprintln!("\nTURN reach (ti=0):");
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 0, &cpu_flop_reach);
    gpu.compute_reach_turn(&ctx, 0);
    let gpu_turn_reach = gpu.download_turn_reach();
    let turn_len = cpu_turn_reach.len();
    let (turn_diff, _) = max_abs(&cpu_turn_reach, &gpu_turn_reach[..turn_len], "turn_reach");

    // ════════════════════════════════════════════════════════════════
    // RIVER REACH (ti=0, ri=0)
    // ════════════════════════════════════════════════════════════════
    eprintln!("\nRIVER reach (ti=0, ri=0):");
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 0, &cpu_turn_reach);
    gpu.compute_reach_river(&ctx, 0, 0);
    let gpu_river_reach = gpu.download_river_reach();
    let river_len = cpu_river_reach.len();
    let (river_diff, _) = max_abs(&cpu_river_reach, &gpu_river_reach[..river_len], "river_reach");

    eprintln!("\n=== VERDICT ===");
    let tol = 1e-5_f32;
    let flop_match = flop_diff < tol;
    let turn_match = turn_diff < tol;
    let river_match = river_diff < tol;

    if flop_match && turn_match && river_match {
        eprintln!("  ✓ Reach MATCHES at ALL zones. Bug is in CFV computation (Hypothesis B).");
        eprintln!("  Next: trace showdown CFV for the 'right-at-uniform-wrong-at-non-uniform'");
        eprintln!("        reach-weighting term in vcfr.metal HU brute force (lines 473-552).");
    } else if !flop_match {
        eprintln!("  ✗ Reach diverges at FLOP. Bug in compute_reach_flop (Hypothesis A).");
    } else if !turn_match {
        eprintln!("  ✗ Reach diverges at TURN. Bug in compute_reach_turn (Hypothesis A).");
    } else {
        eprintln!("  ✗ Reach diverges at RIVER. Bug in compute_reach_river (Hypothesis A).");
    }
}
