// Step 2.A.2 stratum 3: GPU vs CPU parity at FULL nh=1176 under realistic
// asymmetric non-uniform inputs — the mandatory blueprint-scale gate.
//
// THE FRAMING (banked from 2026-06 post-fix conversation):
// - Strata 1 and 2 passing post-fix DOES NOT IMPLY stratum 3 will pass.
//   The arc's lesson is explicit: scale-dependent behavior must be measured,
//   not predicted from smaller scales.
// - Stratum 3 is the actual unblock condition for Step 2.D.
// - This test is a CPU↔GPU REPLICATION check. Correctness signal lives
//   in standing_showdown_oracle (CPU vs independent enumerator).
//
// nh=1176 = C(49,2) = all valid hand pairs not conflicting with the 3-card
// board (Ah, Kd, 7c). Range shape: asymmetric non-uniform sigmoid,
// matching strata 1/2 input pattern.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_full_nh_realistic_hu_optb_game() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["Ah", "Kd", "7c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 2u8;

    // FULL nh = every valid hand pair on this board, sorted by hand strength.
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
    let chosen: Vec<u16> = all_with_strength.iter().map(|&(_, i)| i).collect();
    let k = chosen.len();
    assert_eq!(k, 1176, "expected nh=1176 on a 3-card board, got {}", k);

    // Same sigmoid asymmetric shape as stratum 1/2.
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

    // Keep subset deck (2 turn × 2 river) so total wall-clock is tractable;
    // the nh-scale axis is exercised by nh=1176, the realistic-input axis
    // by the sigmoid asymmetric ranges. (Full chance space is a separate
    // axis that the existing full-geometry parity test exercises.)
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

#[test]
#[ignore = "Step 2.A.2 stratum 3: full nh=1176 GPU↔CPU parity at f32 floor — the blueprint-scale measurement"]
fn step2a2_stratum3_full_nh_gpu_vs_cpu() {
    eprintln!("\n=== Step 2.A.2 stratum 3 (HU OptB, FULL nh=1176, realistic asymmetric) ===");
    eprintln!("Measurement, not prediction. Strata 1/2 passing does not imply stratum 3 will.");

    let (tree, game) = build_full_nh_realistic_hu_optb_game();
    let nh = game.table().num_valid;
    eprintln!("\nTree: {} nodes, nh = {}", tree.num_nodes(), nh);
    eprintln!("n_turn = {}, max_river = {}",
        game.table().remaining_deck.len(),
        game.table().river_decks.iter().filter(|d| !d.is_empty())
            .map(|d| d.len()).max().unwrap_or(0));

    let ctx = MetalContext::new().expect("Metal");
    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let n_iters = 20u32;
    eprintln!("\nRunning {} iters on both solvers (this takes a while at nh=1176)...", n_iters);
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

    let max_abs = |a: &[f32], b: &[f32], label: &str| -> f32 {
        let mut m = 0.0f32;
        let mut idx = 0usize;
        for i in 0..a.len().min(b.len()) {
            let d = (a[i] - b[i]).abs();
            if d > m { m = d; idx = i; }
        }
        eprintln!("  {:24}  max_abs = {:.3e}  at idx {}  (CPU={:.6}, GPU={:.6})",
            label, m, idx,
            a.get(idx).copied().unwrap_or(0.0),
            b.get(idx).copied().unwrap_or(0.0));
        m
    };

    eprintln!("\nAfter {} iters:", n_iters);
    let rf = max_abs(&cpu_flop,      &gpu_regs[..fl],                 "regrets_flop");
    let rt = max_abs(&cpu_turn,      &gpu_regs[fl..fl+tl],            "regrets_turn");
    let rr = max_abs(&cpu_river,     &gpu_regs[fl+tl..fl+tl+rl],      "regrets_river");
    let cf = max_abs(&cpu_cum_flop,  &gpu_cum[..fl],                  "cum_strategy_flop");
    let ct = max_abs(&cpu_cum_turn,  &gpu_cum[fl..fl+tl],             "cum_strategy_turn");
    let cr = max_abs(&cpu_cum_river, &gpu_cum[fl+tl..fl+tl+rl],       "cum_strategy_river");

    // f32 floor tolerance — should be bit-exact (0.0) post sweep-vs-brute fix.
    let tol = 1e-4_f32;
    assert!(rf < tol, "STRATUM 3 BUG: regrets_flop diff {:.3e} > {} — scale-dependent divergence at nh=1176", rf, tol);
    assert!(rt < tol, "STRATUM 3 BUG: regrets_turn diff {:.3e} > {} — scale-dependent divergence at nh=1176", rt, tol);
    assert!(rr < tol, "STRATUM 3 BUG: regrets_river diff {:.3e} > {} — scale-dependent divergence at nh=1176", rr, tol);
    assert!(cf < tol, "STRATUM 3 BUG: cum_flop diff {:.3e} > {} — scale-dependent divergence at nh=1176", cf, tol);
    assert!(ct < tol, "STRATUM 3 BUG: cum_turn diff {:.3e} > {} — scale-dependent divergence at nh=1176", ct, tol);
    assert!(cr < tol, "STRATUM 3 BUG: cum_river diff {:.3e} > {} — scale-dependent divergence at nh=1176", cr, tol);

    eprintln!("\nSTRATUM 3 PASS: full nh=1176 CPU↔GPU REPLICATION holds at f32 floor.");
    eprintln!("Reminder: this is a replication check (engineered post sweep-vs-brute fix).");
    eprintln!("The correctness signal lives in standing_showdown_oracle (CPU vs independent enumerator).");
    eprintln!("Step 2.D (unified preflop+postflop GPU port) is now unblocked at the small-deck subset.");
}
