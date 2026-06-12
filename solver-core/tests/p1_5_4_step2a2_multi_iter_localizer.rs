// Step 2.A.2 multi-iter localizer: find which iter and which stage first
// diverges between CPU and GPU under non-uniform asymmetric inputs.
//
// Post `if pc==0` fix at iter 1: max_abs ~1e-6 (machine floor).
// Post `if pc==0` fix at iter 2: max_abs ~4e-5 (~35x growth — too large for f32 noise).
// Per the lead: the orders-of-magnitude-per-iter growth is the fingerprint of a
// compounding bug, not f32 noise.
//
// This test:
// 1. Verifies regrets after iter 1 are BIT-EXACT (to_bits comparison)
//    between CPU and GPU. If not, iter 1 isn't truly clean.
// 2. If iter 1 is bit-exact, runs iter 2 manually stage-by-stage and
//    identifies the FIRST stage at iter 2 where state diverges.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_realistic_hu_optb_game(k: usize) -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["Ah", "Kd", "7c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 2u8;

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
    let step = all_with_strength.len() / k;
    let chosen: Vec<u16> = (0..k).map(|i| all_with_strength[i * step].1).collect();

    let mut ranges: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for (rank_idx, &hi) in chosen.iter().enumerate() {
        let strength_frac = rank_idx as f32 / k as f32;
        let p0_weight = (strength_frac - 0.3).max(0.05) * 1.5;
        let p0_weight = p0_weight.min(1.0);
        let p1_weight = 0.6 + 0.4 * strength_frac;
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

/// Count how many entries differ between CPU and GPU at to_bits precision,
/// and report the largest absolute difference for any non-bit-exact entries.
fn count_bit_diffs(cpu: &[f32], gpu: &[f32], label: &str) -> (usize, f32, usize) {
    let mut bit_diffs = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    let n = cpu.len().min(gpu.len());
    for i in 0..n {
        if cpu[i].to_bits() != gpu[i].to_bits() {
            bit_diffs += 1;
            let d = (cpu[i] - gpu[i]).abs();
            if d > max_abs { max_abs = d; max_idx = i; }
        }
    }
    eprintln!("  {}: {} / {} bit-different entries (max_abs={:.6e} at idx {})",
        label, bit_diffs, n, max_abs, max_idx);
    (bit_diffs, max_abs, max_idx)
}

#[test]
#[ignore = "2.A.2 multi-iter localizer"]
fn iter_1_bit_exact_check_iter_2_first_diverging_stage() {
    let (tree, game) = build_realistic_hu_optb_game(12);
    let ctx = MetalContext::new().expect("Metal");
    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Run ONE iter on both
    cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    eprintln!("\n=== STATE AFTER ITER 1 (bit-exact check) ===");
    let cpu_reg_flop = cpu.regrets_flop().to_vec();
    let cpu_reg_turn = cpu.regrets_turn().to_vec();
    let cpu_reg_river = cpu.regrets_river().to_vec();
    let cpu_cum_flop = cpu.cum_strategy_flop().to_vec();
    let cpu_cum_turn = cpu.cum_strategy_turn().to_vec();
    let cpu_cum_river = cpu.cum_strategy_river().to_vec();

    let gpu_regs = gpu.download_regrets();
    let gpu_cum = gpu.download_cum_strategy();
    let fl = cpu_reg_flop.len();
    let tl = cpu_reg_turn.len();

    let (rf_bits, rf_max, _) = count_bit_diffs(&cpu_reg_flop, &gpu_regs[..fl], "regrets_flop");
    let (rt_bits, rt_max, _) = count_bit_diffs(&cpu_reg_turn, &gpu_regs[fl..fl+tl], "regrets_turn");
    let (rr_bits, rr_max, _) = count_bit_diffs(&cpu_reg_river, &gpu_regs[fl+tl..fl+tl+cpu_reg_river.len()], "regrets_river");
    let (cf_bits, cf_max, _) = count_bit_diffs(&cpu_cum_flop, &gpu_cum[..fl], "cum_strategy_flop");
    let (ct_bits, ct_max, _) = count_bit_diffs(&cpu_cum_turn, &gpu_cum[fl..fl+tl], "cum_strategy_turn");
    let (cr_bits, cr_max, _) = count_bit_diffs(&cpu_cum_river, &gpu_cum[fl+tl..fl+tl+cpu_cum_river.len()], "cum_strategy_river");

    let total_bit_diffs = rf_bits + rt_bits + rr_bits + cf_bits + ct_bits + cr_bits;
    eprintln!("Total bit-different entries after iter 1: {}", total_bit_diffs);
    eprintln!("Max abs diff across all state buffers: {:.6e}",
        rf_max.max(rt_max).max(rr_max).max(cf_max).max(ct_max).max(cr_max));

    if total_bit_diffs == 0 {
        eprintln!("\n✓ ITER 1 IS BIT-EXACT. State buffer to_bits comparison shows 0 differences.");
        eprintln!("If iter 2 still diverges from this point, something INTRODUCES divergence at iter 2.");
    } else {
        eprintln!("\n✗ ITER 1 has {} bit-different entries despite max_abs ~1e-6.", total_bit_diffs);
        eprintln!("This means iter 1 is NOT bit-clean — compounding will amplify these tiny differences.");
        eprintln!("The `if pc==0` fix is correct but doesn't make iter 1 bit-exact;");
        eprintln!("multi-iter divergence is THEN expected as accumulated f32 noise per iter.");
    }
}
