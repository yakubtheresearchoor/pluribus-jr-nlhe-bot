// Step 2.A.2 stratum 2: GPU vs CPU parity at INTERMEDIATE nh under
// realistic asymmetric non-uniform inputs.
//
// THE FRAMING (banked from 2026-06 post-fix conversation):
// - Stratum 1 (K=12) passing post-fix establishes the CPU↔GPU REPLICATION
//   link at small nh. CPU↔GPU is now a replication check, not a correctness
//   check (the sweep-vs-brute fix engineered bit-exactness).
// - The correctness signal lives in standing_showdown_oracle (CPU vs the
//   independent rules-derived enumerator).
// - Strata 2/3 are SCALE-MEASUREMENT gates: do not predict from stratum 1,
//   measure. The arc's lesson: scale-dependent behavior must be observed.
//
// This stratum measures whether the bit-exactness holds at K=100 (an
// intermediate nh) on the same realistic asymmetric harness as stratum 1.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_realistic_hu_optb_game_k(k: usize) -> (FlatTree, FlopStartGame) {
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

fn measure_parity(k: usize, n_iters: u32) -> (f32, f32, f32, f32, f32, f32) {
    eprintln!("\n--- K={} {} iters ---", k, n_iters);
    let (tree, game) = build_realistic_hu_optb_game_k(k);
    eprintln!("Tree: {} nodes, nh={}", tree.num_nodes(), game.table().num_valid);
    let ctx = MetalContext::new().expect("Metal");
    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    let _ = cpu.run(&tree, &game, n_iters);
    gpu.run(&ctx, &tree, &game, n_iters);
    let cpu_flop = cpu.regrets_flop().to_vec();
    let cpu_turn = cpu.regrets_turn().to_vec();
    let cpu_river = cpu.regrets_river().to_vec();
    let cpu_cum_flop = cpu.cum_strategy_flop().to_vec();
    let cpu_cum_turn = cpu.cum_strategy_turn().to_vec();
    let cpu_cum_river = cpu.cum_strategy_river().to_vec();
    let gpu_regs = gpu.download_regrets();
    let gpu_cum = gpu.download_cum_strategy();
    let fl = cpu_flop.len();
    let tl = cpu_turn.len();
    let rl = cpu_river.len();

    let max_abs = |a: &[f32], b: &[f32]| -> f32 {
        let mut m = 0.0f32;
        for i in 0..a.len().min(b.len()) {
            let d = (a[i] - b[i]).abs();
            if d > m { m = d; }
        }
        m
    };
    let rf = max_abs(&cpu_flop, &gpu_regs[..fl]);
    let rt = max_abs(&cpu_turn, &gpu_regs[fl..fl+tl]);
    let rr = max_abs(&cpu_river, &gpu_regs[fl+tl..fl+tl+rl]);
    let cf = max_abs(&cpu_cum_flop, &gpu_cum[..fl]);
    let ct = max_abs(&cpu_cum_turn, &gpu_cum[fl..fl+tl]);
    let cr = max_abs(&cpu_cum_river, &gpu_cum[fl+tl..fl+tl+rl]);
    eprintln!("  regrets:        flop {:.3e}  turn {:.3e}  river {:.3e}", rf, rt, rr);
    eprintln!("  cum_strategy:   flop {:.3e}  turn {:.3e}  river {:.3e}", cf, ct, cr);
    (rf, rt, rr, cf, ct, cr)
}

#[test]
#[ignore = "Step 2.A.2 stratum 2: intermediate-nh GPU↔CPU parity at K=50, 100, 200 (measurement, not prediction)"]
fn step2a2_stratum2_intermediate_nh_gpu_vs_cpu() {
    eprintln!("\n=== Step 2.A.2 stratum 2 (HU OptB, intermediate nh, realistic asymmetric) ===");
    eprintln!("Scale-axis measurement: stratum 1 (K=12) bit-exact post-fix.");
    eprintln!("Do strata 2 results hold the same property at intermediate nh?\n");

    let tol = 1e-5_f32;
    for &k in &[50usize, 100, 200] {
        let (rf, rt, rr, cf, ct, cr) = measure_parity(k, 50);
        assert!(rf < tol, "K={}: regrets_flop diff {:.3e} > {}", k, rf, tol);
        assert!(rt < tol, "K={}: regrets_turn diff {:.3e} > {}", k, rt, tol);
        assert!(rr < tol, "K={}: regrets_river diff {:.3e} > {}", k, rr, tol);
        assert!(cf < tol, "K={}: cum_flop diff {:.3e} > {}", k, cf, tol);
        assert!(ct < tol, "K={}: cum_turn diff {:.3e} > {}", k, ct, tol);
        assert!(cr < tol, "K={}: cum_river diff {:.3e} > {}", k, cr, tol);
        eprintln!("  ✓ K={}: all 6 buffers within {:.3e}", k, tol);
    }
    eprintln!("\nSTRATUM 2 PASS: intermediate-nh CPU↔GPU REPLICATION holds at f32 floor.");
    eprintln!("Reminder: this is a replication check. Correctness signal lives in standing_showdown_oracle.");
}
