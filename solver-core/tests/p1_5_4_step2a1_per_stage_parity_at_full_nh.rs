// Step 2.A.1 bug localization: per-stage GPU↔CPU parity at full nh=1176.
//
// The aggregate parity (p1_5_4_step2a1_gpu_cpu_parity_at_full_nh.rs) FAILED
// at iter 0 across all three zones simultaneously. The all-zones-diverge
// signature points at a SHARED CFV computation (most likely bottom_up_zone /
// showdown CFV). This test runs the stages in order at full nh and reports
// the first stage where GPU and CPU diverge. That stage's kernel is the bug.
//
// Stages (mirroring metal_stage_validation but at nh=1176, not nh=4):
//   1. compute_all_strategies (strategy buffer at iter 0 = uniform 1/na)
//   2. compute_reach_flop
//   3. compute_reach_turn(ti=0)
//   4. compute_reach_river(ti=0, ri=0)
//   5. bottom_up_zone(River, ti=0, ri=0)  ← prime suspect
//
// Methodology per the lead's directive: find the kernel by MEASUREMENT, not
// inspection. Each stage's output buffer is downloaded from GPU, compared
// against the CPU's equivalent. The first divergence localizes the bug.

#![cfg(feature = "metal")]

use solver_core::card::{
    card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS,
};
use solver_core::gpu_metal::{MetalContext, MetalFlopStartSolver};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{
    DcfrParams, FlopStartVectorCfr, Zone,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

mod helper {
    use super::*;
    /// Build the SAME 4-board chance-table structure as convergence_audit,
    /// but at full nh=1176 (all non-blocking hands on this flop).
    pub fn build_full_nh_4board_game(stacks: i32, pot: i32)
        -> (FlatTree, FlopStartGame)
    {
        let board: Vec<Card> = ["2h", "7d", "Ks"].iter()
            .map(|s| card_from_str(s).unwrap()).collect();
        let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
        let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

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
                        // BUG FIX 2026-06-05: idx indexing must include rc too (convergence_audit
                        // had this typo; coincidentally OK at nh=4, breaks at nh=1176).
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
}

fn diff_stats(cpu: &[f32], gpu: &[f32], label: &str) -> (f32, f32, usize) {
    let n = cpu.len().min(gpu.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut nonzero_diffs = 0usize;
    for i in 0..n {
        let d = (cpu[i] - gpu[i]).abs();
        if d > 1e-9 { nonzero_diffs += 1; }
        if d > max_abs { max_abs = d; }
        let scale = cpu[i].abs().max(gpu[i].abs());
        if scale > 0.01 {
            let rel = d / scale;
            if rel > max_rel { max_rel = rel; }
        }
    }
    eprintln!("  {} (len={}): max_abs={:.6e} max_rel={:.4}% nonzero_diffs={}",
              label, n, max_abs, max_rel * 100.0, nonzero_diffs);
    (max_abs, max_rel, nonzero_diffs)
}

const F32_FLOOR: f32 = 4.29e-6;
const STAGE_ABS_TOL: f32 = F32_FLOOR * 100.0; // 10× algorithm floor for cumulative noise
const STAGE_REL_TOL: f32 = 0.01;              // 1%

fn check_stage(label: &str, cpu: &[f32], gpu: &[f32]) -> bool {
    let (max_abs, max_rel, _) = diff_stats(cpu, gpu, label);
    let pass = max_abs < STAGE_ABS_TOL && max_rel < STAGE_REL_TOL;
    if pass {
        eprintln!("  ✓ {} PASS", label);
    } else {
        eprintln!("  ✗ {} FAIL (max_abs={:.6e} >= {:.2e}, OR max_rel={:.4}% >= {:.2}%)",
                  label, max_abs, STAGE_ABS_TOL, max_rel * 100.0, STAGE_REL_TOL * 100.0);
    }
    pass
}

#[test]
#[ignore = "Step 2.A.1 stage localization at full nh=1176. Run on demand."]
fn step_2a1_per_stage_parity_at_full_nh() {
    eprintln!("\n========================================================================");
    eprintln!("=== Step 2.A.1 stage localization: per-stage parity at nh=1176       ===");
    eprintln!("===   Find the FIRST kernel where GPU diverges from CPU              ===");
    eprintln!("========================================================================\n");

    let (tree, game) = helper::build_full_nh_4board_game(/*stacks=*/5, /*pot=*/2);
    let table = game.table();
    let nh = table.num_valid;
    let nn = tree.num_nodes();
    eprintln!("Setup: tree {} nodes, nh = {}", nn, nh);
    assert_eq!(nh, 1176, "expected nh=1176");

    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // ─── Stage 1: compute_all_strategies (uniform 1/na from zero regrets) ───
    eprintln!("\n--- Stage 1: compute_all_strategies (CPU vs GPU) ---");
    cpu.compute_all_strategies(&tree);
    gpu.compute_all_strategies(&ctx);
    let gpu_strat = gpu.download_strategy();
    // CPU's strategy_*() getters return per-zone scratches in post-refactor layout.
    // For stage-1 verification we check that BOTH CPU and GPU produce uniform
    // 1/na in the flop strategy buffer (the only fully-materialized one).
    let cpu_strat_flop = cpu.strategy_flop();
    // GPU strategy buffer is [flop_stride | turn_total | river_total]. Flop is at offset 0.
    let stage1_pass = check_stage("Stage 1: strategy_flop", cpu_strat_flop, &gpu_strat[..cpu_strat_flop.len()]);
    if !stage1_pass {
        panic!("Stage 1 FAIL: compute_all_strategies (kernel: vcfr_compute_strategy or equivalent) \
                produces different strategy from CPU at nh=1176.");
    }

    // ─── Stage 2: compute_reach_flop ───
    eprintln!("\n--- Stage 2: compute_reach_flop ---");
    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);
    gpu.compute_reach_flop(&ctx);
    let gpu_flop_reach = gpu.download_reach();
    let stage2_pass = check_stage("Stage 2: reach_flop", &cpu_flop_reach, &gpu_flop_reach);
    if !stage2_pass {
        panic!("Stage 2 FAIL: compute_reach_flop kernel diverges from CPU at nh=1176.");
    }

    // ─── Stage 3: compute_reach_turn(ti=0) ───
    eprintln!("\n--- Stage 3: compute_reach_turn(ti=0) ---");
    let ti = 0;
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_flop_reach);
    gpu.compute_reach_turn(&ctx, ti);
    let gpu_turn_reach = gpu.download_turn_reach();
    let stage3_pass = check_stage("Stage 3: reach_turn(ti=0)", &cpu_turn_reach, &gpu_turn_reach);
    if !stage3_pass {
        panic!("Stage 3 FAIL: compute_reach_turn kernel diverges at nh=1176.");
    }

    // ─── Stage 4: compute_reach_river(ti=0, ri=0) ───
    eprintln!("\n--- Stage 4: compute_reach_river(ti=0, ri=0) ---");
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 0, &cpu_turn_reach);
    gpu.compute_reach_river(&ctx, 0, 0);
    let gpu_river_reach = gpu.download_river_reach();
    let stage4_pass = check_stage("Stage 4: reach_river(0,0)", &cpu_river_reach, &gpu_river_reach);
    if !stage4_pass {
        panic!("Stage 4 FAIL: compute_reach_river kernel diverges at nh=1176.");
    }

    // ─── Stage 5: bottom_up_zone(River, ti=0, ri=0) — the prime suspect ───
    eprintln!("\n--- Stage 5: bottom_up_zone(River, 0, 0) — multiway CFV / showdown ---");
    let mut cpu_cfv = vec![0.0f32; nn * nh];
    let cpu_params = DcfrParams::new(0);
    cpu.bottom_up_zone(&tree, table, 0, &cpu_river_reach, &mut cpu_cfv,
                       Zone::River, Some(0), Some(0), &cpu_params);
    use solver_core::gpu_metal::GpuDcfrParams;
    let gpu_params = GpuDcfrParams::new(0);
    gpu.bottom_up_river(&ctx, 0, 0, 0, &gpu_params);
    let gpu_cfv_batch = gpu.download_river_cfv_batch();
    let gpu_river_cfv = &gpu_cfv_batch[0..nn*nh];
    let stage5_pass = check_stage("Stage 5: bottom_up_zone(River)", &cpu_cfv, gpu_river_cfv);
    if !stage5_pass {
        // Drill into where the divergence lives within the cfv buffer.
        eprintln!("\n  --- Top 10 worst diffs (showing node type) ---");
        let mut diffs: Vec<(usize, f32)> = (0..nn*nh)
            .map(|i| (i, (cpu_cfv[i] - gpu_river_cfv[i]).abs()))
            .filter(|&(_, d)| d > 1e-3)
            .collect();
        diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(idx, d) in diffs.iter().take(10) {
            let node_id = idx / nh;
            let h = idx % nh;
            let ntype = tree.nodes[node_id].node_type;
            let ntype_str = match ntype { 0 => "TERM", 1 => "CHANCE", 2 => "PLAYER", _ => "???" };
            eprintln!("    node={} h={} type={} CPU={:.4} GPU={:.4} diff={:.4}",
                      node_id, h, ntype_str, cpu_cfv[idx], gpu_river_cfv[idx], d);
        }
        panic!("Stage 5 FAIL: bottom_up_zone(River) kernel diverges at nh=1176. \
                The divergence in the topology above indicates whether the bug is at \
                terminal nodes (showdown CFV) or aggregation nodes (player/chance handlers).");
    }

    eprintln!("\n=== ALL STAGES PASS at nh=1176 ===");
    eprintln!("  The aggregate parity failure (regret divergence at iter 0) must be in a");
    eprintln!("  stage AFTER bottom_up_zone(River) — turn bottom-up, flop bottom-up, or the");
    eprintln!("  per-pair regret update kernel. Extend this test with stages 6-9 to localize.");
}
