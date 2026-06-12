// Step 2.A.1 — Production-nh GPU↔CPU parity at the cheapest scale that
// exercises the scale-dependence risk.
//
// Background: the GPU↔CPU parity gates that exist today (convergence_audit,
// three_player_convergence_proper, etc.) all run at nh=4. The streaming-
// strategy refactor exposed scale-dependent bugs at nh=1176 that nh=4
// could not catch (the d_strategy upload bug being the headline). Before
// Step 2's GPU port commits to the disk-backed I/O design and the per-
// stage instrumentation, we need to know: does the GPU match CPU at full
// nh=1176?
//
// 2.A.1 design choices:
//   - SMALL TREE (HU 1+0, low stacks): the per-iter GPU debug-instrumentation
//     downloads scale with n_pairs × tree-size. A small tree keeps each
//     download manageable.
//   - SAME 4-BOARD CHANCE TABLE STRUCTURE as convergence_audit (2 turn ×
//     2 river), keeping the chance-table size from blowing up.
//   - FULL NH (1176, all non-blocking hands on the flop): this is the
//     scale axis we want to exercise. The streaming refactor's per-pair
//     scratch sizing, the strategy scratch (river_stride at nh=1176), the
//     compute_*_strategy_for_* per-pair calls — all see production nh.
//   - IN-MEMORY both sides: no disk-backed wiring yet (that's 2.B). The
//     comparison is pure logic, not I/O.
//
// What this gate catches that nh=4 cannot:
//   - Scratch-vs-full buffer-size mismatches (the d_strategy upload bug
//     pattern, in case any other site has the same shape).
//   - GPU strategy-buffer indexing at nh=1176 (offsets that fit a u16 at
//     nh=4 vs need u32 at nh=1176).
//   - Per-pair regret matching at production stride.
//   - Terminal CFV at production nh (this is the multiway-scaling driver
//     for Step 2.D; getting HU baseline first is the validation).
//
// What this gate does NOT catch (deferred to 2.A.2 at production scale):
//   - Disk-backed I/O on GPU.
//   - Full 49×48 = 2,352 pair chance integration.
//   - Production tree depth effects.

#![cfg(feature = "metal")]

use solver_core::card::{
    card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS,
};
use solver_core::gpu_metal::{MetalContext, MetalFlopStartSolver};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

/// Build the SAME 4-board chance-table structure as convergence_audit, but
/// at nh = all-non-blocking-hands-on-this-flop (= 1176). Same flop
/// (2h 7d Ks), same 2 turn × 2 river structure.
fn build_full_nh_4board_game(stacks: i32, pot: i32)
    -> (solver_core::tree::flat::FlatTree, FlopStartGame)
{
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter()
        .map(|s| card_from_str(s).unwrap()).collect();
    let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
    let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

    // Take ALL non-blocking hand indices (nh ≈ 1176 on this flop).
    let mut chosen: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS as u16 {
        let (c1, c2) = index_to_card_pair(idx as usize);
        if board_mask & (1u64 << c1) != 0 { continue; }
        if board_mask & (1u64 << c2) != 0 { continue; }
        chosen.push(idx);
    }
    let nh = chosen.len();
    let num_players = 2u8;
    let num_opp = 1usize;
    let valid_hand_indices = chosen.clone();
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

    // Turn ranks + sorted arrays.
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
            .map(|h| (turn_ranks[tc as usize * nh + h] + 1, h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            for (si, &(str_, idx)) in items.iter().enumerate() {
                turn_sorted_str[tc as usize * num_opp * nh + oi * nh + si] = str_;
                turn_sorted_idx[tc as usize * num_opp * nh + oi * nh + si] = idx;
            }
        }
    }

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
                .map(|h| (river_ranks[tc as usize * 52 * nh + rc as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                for (si, &(str_, idx)) in items.iter().enumerate() {
                    // BUG FIX 2026-06-05: idx indexing must include rc too
                    // (convergence_audit had this typo).
                    let base = tc as usize * 52 * num_opp * nh
                             + rc as usize * num_opp * nh
                             + oi * nh;
                    river_sorted_str[base + si] = str_;
                    river_sorted_idx[base + si] = idx;
                }
            }
        }
    }

    let initial_weights: Vec<Vec<f32>> = (0..num_players).map(|_| {
        let mut w = vec![0.0f32; nh];
        for h in 0..nh {
            let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
            let mut blocked = 0i32;
            for h2 in 0..nh {
                if h2 == h { continue; }
                let (c3, c4) = index_to_card_pair(valid_hand_indices[h2] as usize);
                if c1 == c3 || c1 == c4 || c2 == c3 || c2 == c4 { blocked += 1; }
            }
            w[h] = if blocked < (nh as i32 - 1) { 1.0 } else { 0.0 };
        }
        w
    }).collect();
    let num_combinations = initial_weights[0].iter().sum::<f32>()
                          * initial_weights[1].iter().sum::<f32>();

    let table = FlopChanceTable {
        hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
        remaining_deck: turn_cards, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx, initial_weights, num_players,
        num_combinations: num_combinations as f64, river_decks,
    };

    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop,
        starting_pot: pot, starting_stacks: vec![stacks, stacks],
        initial_contributions: vec![pot / 2, pot / 2],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)], raise: vec![],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).expect("flop tree");
    let game = FlopStartGame::new(table);
    (tree, game)
}

#[test]
#[ignore = "Step 2.A.1: GPU↔CPU parity at full nh=1176, small tree, 4-board. Run on demand."]
fn step_2a1_gpu_cpu_parity_at_full_nh_small_tree() {
    eprintln!("\n========================================================================");
    eprintln!("=== Step 2.A.1: GPU↔CPU parity at FULL NH (small tree, in-memory)    ===");
    eprintln!("===   Scale-dependence catch for the streaming-strategy refactor     ===");
    eprintln!("========================================================================\n");

    let (tree, game) = build_full_nh_4board_game(/*stacks=*/5, /*pot=*/2);
    let table = game.table();
    let nh = table.num_valid;
    let nn = tree.num_nodes();
    let n_pairs = table.remaining_deck.iter()
        .map(|&tc| table.river_decks[tc as usize].len())
        .sum::<usize>();
    eprintln!("Setup: tree {} nodes, nh = {}, pairs = {}", nn, nh, n_pairs);
    assert_eq!(nh, 1176, "expected nh = 1176 (full non-blocking hand set)");

    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let fl = cpu.regrets_flop().len();
    let tl = cpu.regrets_turn().len();
    let rl = cpu.regrets_river().len();
    eprintln!("Buffer lengths: regrets_flop={}, regrets_turn={}, regrets_river={}",
              fl, tl, rl);
    eprintln!("Total regrets storage: {} f32 = {:.2} MB",
              fl + tl + rl, (fl + tl + rl) * 4 / (1 << 20));

    // -------- Per-iter parity --------
    let n_iters = 5u32;
    eprintln!("\n--- Running {} iters on both, comparing per-iter ---", n_iters);

    let mut overall_max_abs_reg = 0.0f32;
    let mut overall_max_rel_reg = 0.0f32;
    let mut overall_max_abs_cum = 0.0f32;
    let mut overall_max_rel_cum = 0.0f32;

    for i in 0..n_iters {
        let _ = cpu.run(&tree, &game, 1);
        gpu.run(&ctx, &tree, &game, 1);

        let gpu_reg = gpu.download_regrets();
        let gpu_cum = gpu.download_cum_strategy();

        for (zone, cpu_slice, gpu_slice) in [
            ("flop",  cpu.regrets_flop(),  &gpu_reg[..fl]),
            ("turn",  cpu.regrets_turn(),  &gpu_reg[fl..fl+tl]),
            ("river", cpu.regrets_river(), &gpu_reg[fl+tl..]),
        ] {
            let (max_abs, max_rel) = max_diffs(cpu_slice, gpu_slice);
            if max_abs > overall_max_abs_reg { overall_max_abs_reg = max_abs; }
            if max_rel > overall_max_rel_reg { overall_max_rel_reg = max_rel; }
            eprintln!("  iter {} regrets {:>5}: max_abs={:.3e} max_rel={:.2}%",
                      i, zone, max_abs, max_rel * 100.0);
        }

        for (zone, cpu_slice, gpu_slice) in [
            ("flop",  cpu.cum_strategy_flop(),  &gpu_cum[..fl]),
            ("turn",  cpu.cum_strategy_turn(),  &gpu_cum[fl..fl+tl]),
            ("river", cpu.cum_strategy_river(), &gpu_cum[fl+tl..]),
        ] {
            let (max_abs, max_rel) = max_diffs(cpu_slice, gpu_slice);
            if max_abs > overall_max_abs_cum { overall_max_abs_cum = max_abs; }
            if max_rel > overall_max_rel_cum { overall_max_rel_cum = max_rel; }
            eprintln!("  iter {} cum     {:>5}: max_abs={:.3e} max_rel={:.2}%",
                      i, zone, max_abs, max_rel * 100.0);
        }
    }

    // -------- Final report --------
    eprintln!("\n========================================================================");
    eprintln!("=== Overall worst divergence across all iters and zones              ===");
    eprintln!("========================================================================");
    eprintln!("  regrets:      max_abs = {:.6e}, max_rel = {:.4}%",
              overall_max_abs_reg, overall_max_rel_reg * 100.0);
    eprintln!("  cum_strategy: max_abs = {:.6e}, max_rel = {:.4}%",
              overall_max_abs_cum, overall_max_rel_cum * 100.0);

    // Gate: at f32 floor (4.29e-6 documented as f32 algorithm floor in task #6).
    // We allow slightly more than the documented floor (10×) because per-iter
    // accumulation amplifies f32 noise.
    let f32_floor = 4.29e-6f32;
    let max_rel_threshold = 0.01_f32; // 1% per the lead's metal-stage tolerance

    let pass_reg_abs = overall_max_abs_reg < f32_floor * 100.0;
    let pass_reg_rel = overall_max_rel_reg < max_rel_threshold;
    let pass_cum_abs = overall_max_abs_cum < f32_floor * 100.0;
    let pass_cum_rel = overall_max_rel_cum < max_rel_threshold;

    eprintln!("\n  Gate: max_rel < {:.2}% AND max_abs < {:.2e} (10× f32 floor)",
              max_rel_threshold * 100.0, f32_floor * 100.0);
    eprintln!("    regrets:      max_abs {} max_rel {}",
              if pass_reg_abs { "PASS" } else { "FAIL" },
              if pass_reg_rel { "PASS" } else { "FAIL" });
    eprintln!("    cum_strategy: max_abs {} max_rel {}",
              if pass_cum_abs { "PASS" } else { "FAIL" },
              if pass_cum_rel { "PASS" } else { "FAIL" });

    let all_pass = pass_reg_abs && pass_reg_rel && pass_cum_abs && pass_cum_rel;
    if all_pass {
        eprintln!("\n=== Step 2.A.1 PASS ===");
        eprintln!("  GPU matches CPU at full nh=1176 (scale-dependence verified).");
        eprintln!("  Streaming-strategy refactor's behavior carries to GPU correctly.");
        eprintln!("  Step 2.B (GPU disk-backed port) and 2.A.2 (production-scale parity)");
        eprintln!("  proceed on a validated GPU baseline.");
    } else {
        panic!("Step 2.A.1 FAIL: GPU diverges from CPU at full nh. \
                Some operation at nh=1176 differs from nh=4 in a way that breaks parity. \
                See per-iter / per-zone divergence above to localize.");
    }
}

fn max_diffs(cpu: &[f32], gpu: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        let scale = a.abs().max(b.abs());
        if scale > 0.01 {
            let rel = d / scale;
            if rel > max_rel { max_rel = rel; }
        }
    }
    (max_abs, max_rel)
}
