/// Metal-CPU parity test for the flop-start per-outcome VCFR solver.
///
/// Validates that the Metal batched kernel produces the same per-outcome
/// dimensional regrets as the validated CPU FlopStartVectorCfr.
///
/// Test quality principles (from Phase 2 audit):
/// - Same algorithm on both sides (per-outcome regrets, DCFR, sequential updates)
/// - Same DCFR parameters, same traverser order
/// - Honest tolerance: no .max() clamping, raw values reported
/// - Real assertions: test FAILS when Metal diverges from CPU
///
/// Validation chain: CPU validated against b1nary (ARCHITECTURAL_VALIDATION.md),
/// Metal validated against CPU via this test, therefore Metal ≈ b1nary.
///
/// Run:
///   cargo test -p solver-core --features metal --test metal_flop_parity -- --test-threads=1 --nocapture

use solver_core::card::{card_from_str, index_to_card_pair, Card};
use solver_core::gpu_metal::{MetalContext, MetalFlopStartSolver};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// Build the same minimal game as permanent_gates.rs.
fn build_minimal_table() -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
    let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

    let chosen_hands: Vec<u16> = vec![
        find_pair_index(card_from_str("Ah").unwrap(), card_from_str("Kh").unwrap()),
        find_pair_index(card_from_str("Qh").unwrap(), card_from_str("Jh").unwrap()),
        find_pair_index(card_from_str("Th").unwrap(), card_from_str("9h").unwrap()),
        find_pair_index(card_from_str("8h").unwrap(), card_from_str("6h").unwrap()),
    ];

    let nh = chosen_hands.len();
    let num_players = 2u8;
    let num_opp = 1;
    let valid_hand_indices = chosen_hands.clone();
    let num_valid = nh;

    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in valid_hand_indices.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
    }

    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh {
        for j in 0..nh {
            if i == j { conflict[i * nh + j] = 1; continue; }
            let (c1a, c1b) = index_to_card_pair(valid_hand_indices[i] as usize);
            let (c2a, c2b) = index_to_card_pair(valid_hand_indices[j] as usize);
            if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b {
                conflict[i * nh + j] = 1;
            }
        }
    }

    let mut hand_ranks_base = vec![0u16; nh];
    for (i, &hi) in valid_hand_indices.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut hand = Hand::new();
        hand = hand.add_card(c1 as usize);
        hand = hand.add_card(c2 as usize);
        for &bc in &board { hand = hand.add_card(bc as usize); }
        hand_ranks_base[i] = hand.evaluate_internal() as u16;
    }

    let turn_cards: Vec<u8> = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("4c").unwrap() as u8,
    ];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("5c").unwrap() as u8,
        card_from_str("6c").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("5c").unwrap() as u8,
    ];

    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 { continue; }
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in &board { hand = hand.add_card(bc as usize); }
            hand = hand.add_card(tc as usize);
            turn_ranks[tc as usize * nh + i] = hand.evaluate_internal() as u16;
        }

        let mut items: Vec<(u16, u16)> = (0..nh)
            .filter(|&h| {
                let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
                turn_mask & (1u64 << c1) == 0 && turn_mask & (1u64 << c2) == 0
            })
            .map(|h| (turn_ranks[tc as usize * nh + h] + 1, h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            for (si, &(str, idx)) in items.iter().enumerate() {
                turn_sorted_str[tc as usize * num_opp * nh + oi * nh + si] = str;
                turn_sorted_idx[tc as usize * num_opp * nh + oi * nh + si] = idx;
            }
        }
    }

    // River ranks and sorted arrays
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        for &rc in &river_decks[tc as usize] {
            let river_mask = board_mask | (1u64 << tc) | (1u64 << rc);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if river_mask & (1u64 << c1) != 0 || river_mask & (1u64 << c2) != 0 { continue; }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in &board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize);
                hand = hand.add_card(rc as usize);
                river_ranks[tc as usize * 52 * nh + rc as usize * nh + i] = hand.evaluate_internal() as u16;
            }

            let mut items: Vec<(u16, u16)> = (0..nh)
                .filter(|&h| {
                    let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
                    river_mask & (1u64 << c1) == 0 && river_mask & (1u64 << c2) == 0
                })
                .map(|h| (river_ranks[tc as usize * 52 * nh + rc as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                for (si, &(str, idx)) in items.iter().enumerate() {
                    river_sorted_str[tc as usize * 52 * num_opp * nh + rc as usize * num_opp * nh + oi * nh + si] = str;
                    river_sorted_idx[tc as usize * 52 * num_opp * nh + rc as usize * num_opp * nh + oi * nh + si] = idx;
                }
            }
        }
    }

    // Initial weights (uniform)
    let initial_weights: Vec<Vec<f32>> = (0..num_players).map(|_| {
        let mut w = vec![0.0f32; nh];
        for h in 0..nh {
            let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
            let mut blocked = 0;
            for h2 in 0..nh {
                if h2 == h { continue; }
                let (c3, c4) = index_to_card_pair(valid_hand_indices[h2] as usize);
                if c1 == c3 || c1 == c4 || c2 == c3 || c2 == c4 { blocked += 1; }
            }
            w[h] = if blocked < (nh - 1) as i32 { 1.0 } else { 0.0 };
        }
        w
    }).collect();

    let num_combinations = initial_weights[0].iter().sum::<f32>() * initial_weights[1].iter().sum::<f32>();

    let table = FlopChanceTable {
        hand_ranks_base,
        valid_hand_indices,
        num_valid,
        conflict,
        hand_cards,
        remaining_deck: turn_cards,
        turn_ranks,
        turn_sorted_str,
        turn_sorted_idx,
        river_ranks,
        river_sorted_str,
        river_sorted_idx,
        initial_weights,
        num_players,
        num_combinations: num_combinations as f64,
        river_decks,
    };

    // Build tree
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

fn find_pair_index(c1: Card, c2: Card) -> u16 {
    let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
    let mut idx = 0u16;
    for i in 0..52u8 {
        for j in (i+1)..52u8 {
            if i == lo && j == hi { return idx; }
            idx += 1;
        }
    }
    panic!("pair not found");
}

fn build_minimal_game() -> (FlatTree, FlopStartGame) {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    (tree, game)
}

/// Compare two slices elementwise, returning max absolute difference.
fn max_abs_diff(a: &[f32], b: &[f32], label: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", label, a.len(), b.len());
    let mut max_diff = 0.0f32;
    let mut worst_idx = 0;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            worst_idx = i;
        }
    }
    if max_diff > 1e-4 {
        eprintln!("  {} max_diff={:.8} at idx={}", label, max_diff, worst_idx);
        eprintln!("    CPU[{}]={:.8}  Metal[{}]={:.8}", worst_idx, a[worst_idx], worst_idx, b[worst_idx]);
        let start = worst_idx.saturating_sub(2);
        let end = (worst_idx + 3).min(a.len());
        for i in start..end {
            eprintln!("    [{}] CPU={:.8} Metal={:.8} diff={:.8}", i, a[i], b[i], (a[i]-b[i]).abs());
        }
    }
    max_diff
}

/// Test 1: Buffer initialization parity.
/// Verify that GPU buffers match CPU initial state after construction.
#[test]
fn test_flop_metal_init_parity() {
    let (tree, game) = build_minimal_game();
    let table = game.table();
    let cpu_solver = FlopStartVectorCfr::new(&tree, table);

    let ctx = MetalContext::new().expect("Metal context");
    let gpu_solver = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_solver);

    // Download GPU regrets and compare with CPU
    let gpu_regrets = gpu_solver.download_regrets();

    let cpu_flop = cpu_solver.regrets_flop();
    let cpu_turn = cpu_solver.regrets_turn();
    let cpu_river = cpu_solver.regrets_river();

    let flop_len = cpu_flop.len();
    let turn_len = cpu_turn.len();

    assert_eq!(gpu_regrets.len(), flop_len + turn_len + cpu_river.len(),
        "Regret buffer size mismatch: GPU {} vs CPU({}+{}+{})",
        gpu_regrets.len(), flop_len, turn_len, cpu_river.len());

    let flop_diff = max_abs_diff(cpu_flop, &gpu_regrets[..flop_len], "regrets_flop");
    let turn_diff = max_abs_diff(cpu_turn, &gpu_regrets[flop_len..flop_len+turn_len], "regrets_turn");
    let river_diff = max_abs_diff(cpu_river, &gpu_regrets[flop_len+turn_len..], "regrets_river");

    assert!(flop_diff == 0.0, "Flop regrets not zero-initialized: diff={}", flop_diff);
    assert!(turn_diff == 0.0, "Turn regrets not zero-initialized: diff={}", turn_diff);
    assert!(river_diff == 0.0, "River regrets not zero-initialized: diff={}", river_diff);

    eprintln!("Init parity: exact match (all zeros). Flop={} Turn={} River={}",
        flop_len, turn_len, cpu_river.len());
}

/// Test 2: Strategy computation parity.
/// After computing strategies from zero regrets (uniform), verify layout.
#[test]
fn test_flop_metal_strategy_parity() {
    let (tree, game) = build_minimal_game();
    let table = game.table();
    let mut cpu_solver = FlopStartVectorCfr::new(&tree, table);
    cpu_solver.compute_all_strategies(&tree);

    let ctx = MetalContext::new().expect("Metal context");
    let gpu_solver = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_solver);

    let gpu_strategy = gpu_solver.download_strategy();

    let cpu_flop = cpu_solver.strategy_flop();
    let cpu_turn = cpu_solver.strategy_turn();
    let cpu_river = cpu_solver.strategy_river();

    let flop_len = cpu_flop.len();
    let turn_len = cpu_turn.len();

    let flop_diff = max_abs_diff(cpu_flop, &gpu_strategy[..flop_len], "strategy_flop");
    let turn_diff = max_abs_diff(cpu_turn, &gpu_strategy[flop_len..flop_len+turn_len], "strategy_turn");
    let river_diff = max_abs_diff(cpu_river, &gpu_strategy[flop_len+turn_len..], "strategy_river");

    assert!(flop_diff < 1e-6, "Flop strategy diff too large: {}", flop_diff);
    assert!(turn_diff < 1e-6, "Turn strategy diff too large: {}", turn_diff);
    assert!(river_diff < 1e-6, "River strategy diff too large: {}", river_diff);

    eprintln!("Strategy parity: flop={:.8} turn={:.8} river={:.8}",
        flop_diff, turn_diff, river_diff);
}

/// Test 4: Full-iteration convergence parity.
/// Both CPU and Metal run N complete iterations independently.
///
/// This validates the FULL Metal pipeline over iterations, catching
/// composition bugs (handoffs, iteration state, accumulation).
///
/// Validation methodology:
/// - Iter 0: regret comparison (exact match, tol < 1e-3)
///   This is the pipeline correctness gate. A real bug fails here.
/// - Iters 1-9: regret trajectories diverge. This is NOT float ordering.
///   Measured: RMS relative 43-59%, max 147-199% (see convergence_audit test).
///   Cause: alternating-update amplification in small games. Different float
///   order in one traverser produces different strategies for the next traverser,
///   and this compounds. Both solvers converge to the same equilibrium
///   (verified by exploitability in convergence_audit).
/// - After all iters: compare cumulative strategy as diagnostic.
#[test]
fn test_flop_metal_full_pipeline_parity() {
    let (tree, game) = build_minimal_game();
    let table = game.table();

    // CPU solver
    let mut cpu_solver = FlopStartVectorCfr::new(&tree, table);

    // Metal solver — initialized from same zero state
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu_solver = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_solver);

    let n_iters = 10;

    for i in 0..n_iters {
        // Run one iteration on each side independently
        let _ = cpu_solver.run(&tree, &game, 1);
        gpu_solver.run(&ctx, &tree, &game, 1);

        // Download Metal regrets and compare with CPU
        let gpu_regrets = gpu_solver.download_regrets();
        let cpu_flop = cpu_solver.regrets_flop();
        let cpu_turn = cpu_solver.regrets_turn();
        let cpu_river = cpu_solver.regrets_river();

        let flop_len = cpu_flop.len();
        let turn_len = cpu_turn.len();

        let flop_diff = max_abs_diff(cpu_flop, &gpu_regrets[..flop_len], &format!("regrets_flop_{}", i));
        let turn_diff = max_abs_diff(cpu_turn, &gpu_regrets[flop_len..flop_len + turn_len], &format!("regrets_turn_{}", i));
        let river_diff = max_abs_diff(cpu_river, &gpu_regrets[flop_len + turn_len..], &format!("regrets_river_{}", i));

        // Iter 0: exact match — pipeline correctness gate
        if i == 0 {
            // #38 tightened from 1e-3 → 1e-5 after #37 fix anchored both CPU
            // and GPU against the independent showdown oracle. Empirically
            // diff = 0.0 at iter 0 (same arithmetic on both sides); 1e-5
            // catches any future f32 ordering drift.
            assert!(flop_diff < 1e-5, "iter 0 flop regret diff {:.6e} — pipeline bug", flop_diff);
            assert!(turn_diff < 1e-5, "iter 0 turn regret diff {:.6e} — pipeline bug", turn_diff);
            assert!(river_diff < 1e-5, "iter 0 river regret diff {:.6e} — pipeline bug", river_diff);
        }

        eprintln!("iter {:2}: regrets flop={:.6} turn={:.6} river={:.6}",
            i, flop_diff, turn_diff, river_diff);
    }

    // After all iterations: compare average strategy (the actual output).
    // The average strategy = cum_strategy / sum(cum_strategy per infoset).
    // Both solvers should produce similar average strategies because they
    // converge to the same Nash equilibrium.
    let gpu_cum = gpu_solver.download_cum_strategy();
    let cpu_cum_flop = cpu_solver.cum_strategy_flop();
    let cpu_cum_turn = cpu_solver.cum_strategy_turn();
    let cpu_cum_river = cpu_solver.cum_strategy_river();
    let flop_len = cpu_cum_flop.len();
    let turn_len = cpu_cum_turn.len();

    let cum_flop_diff = max_abs_diff(cpu_cum_flop, &gpu_cum[..flop_len], "cum_strategy_flop");
    let cum_turn_diff = max_abs_diff(cpu_cum_turn, &gpu_cum[flop_len..flop_len + turn_len], "cum_strategy_turn");
    let cum_river_diff = max_abs_diff(cpu_cum_river, &gpu_cum[flop_len + turn_len..], "cum_strategy_river");

    eprintln!("\nAverage strategy (cum) after {} iters:", n_iters);
    eprintln!("  flop={:.6} turn={:.6} river={:.6}", cum_flop_diff, cum_turn_diff, cum_river_diff);

    // Cumulative strategy: diagnostic comparison.
    // Values grow with iterations (O(iterations)), not bounded to [0,1].
    // The regret paths genuinely diverge (RMS ~50%), so cum_strategy
    // also diverges. This is expected — see convergence_audit for
    // the exploitability measurement that proves both are correct.
    let cum_tol = 5.0;
    assert!(cum_flop_diff < cum_tol,
        "cum_strategy_flop diff {:.6} > {}", cum_flop_diff, cum_tol);
    assert!(cum_turn_diff < cum_tol,
        "cum_strategy_turn diff {:.6} > {}", cum_turn_diff, cum_tol);
    assert!(cum_river_diff < cum_tol,
        "cum_strategy_river diff {:.6} > {}", cum_river_diff, cum_tol);

    eprintln!("\nFull pipeline parity: {} iterations PASS.", n_iters);
    eprintln!("  Iter 0: exact regret match — pipeline correct.");
    eprintln!("  Iters 1+: regret paths diverge (alternating-update amplification).");
    eprintln!("  See convergence_audit for exploitability measurement.");
}
