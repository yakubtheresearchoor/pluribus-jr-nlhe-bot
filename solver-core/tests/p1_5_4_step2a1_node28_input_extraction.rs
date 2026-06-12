// Step 2.A.1: node-28 input extraction.
//
// Findings so far:
//   - Stages 1–4 (strategy, reach_flop/turn/river) PASS bit-exact at nh=1176
//   - Showdown helper with controlled inputs PASSES at nh=1176 (micro-kernel)
//   - Sorted arrays PASS bit-exact at nh=1176 (upload + (tc, rc) slice)
//   - num_combinations divisor PASSES — pre-divide CPU=329,728 vs GPU=-3,465,
//     so the helper produces DIFFERENT output at terminal node 28
//
// Test plan: call the debug helper kernel with the EXACT inputs node 28's
// terminal handler would see (extracted from CPU side, where they match GPU
// for verified components: opp_reach via reach_river, sorted arrays via
// (tc,rc) slice). Then:
//
//   Case A: debug helper result == GPU bottom_up_zone result (-0.0025)
//     → the helper IS producing the same output as bottom_up_zone (the
//       inputs match what bottom_up_zone passes). Then CPU bottom_up_zone
//       must be passing DIFFERENT inputs than I extract (or computing the
//       reference differently). Localizes the bug to "what bottom_up_zone
//       passes the helper at terminals."
//
//   Case B: debug helper result == CPU bottom_up_zone result (0.238)
//     → the helper produces the right answer with these inputs. Then GPU
//       bottom_up_zone's terminal-handler wrapping has the bug (something
//       in the lines 877–935 of vcfr.metal: opp_reach copy, sorted-array
//       pointer arithmetic at the kernel level, contributions read, etc.)
//
//   Case C: debug helper result matches neither
//     → the inputs I'm extracting are wrong; bug is upstream of what I
//       can see from public CPU APIs.

#![cfg(feature = "metal")]

use metal::MTLSize;
use solver_core::card::{
    card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS,
};
use solver_core::gpu_metal::buffer::MetalBuffer;
use solver_core::gpu_metal::{GpuDcfrParams, MetalContext, MetalFlopStartSolver};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{
    DcfrParams, FlopStartVectorCfr, Zone,
};
use solver_core::solver::showdown::side_pot_showdown_cfv;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

#[repr(C)]
#[derive(Clone, Copy)]
struct DebugBruteForceParams {
    nh: i32,
    np: i32,
    traverser: i32,
    starting_pot: i32,
    fold_mask: u16,
    _pad: u16,
    rake_rate: f32,
    rake_cap: f32,
    flop_seen: i32,
}

fn gpu_brute_force(
    ctx: &MetalContext,
    nh: usize, np: usize, traverser: usize,
    starting_pot: i32, fold_mask: u16,
    opp_reach: &[f32], contributions: &[i32],
    hand_cards: &[u8], pl_str: &[u16], pl_idx: &[u16],
) -> Vec<f32> {
    let device = ctx.device();
    let pipeline = ctx.create_pipeline("debug_brute_force_showdown")
        .expect("debug pipeline");

    let d_output = MetalBuffer::<f32>::zeros(device, nh);
    let d_opp_reach = MetalBuffer::from_slice(device, opp_reach);
    let d_contributions = MetalBuffer::from_slice(device, contributions);
    let d_hand_cards = MetalBuffer::from_slice(device, hand_cards);
    let d_pl_str = MetalBuffer::from_slice(device, pl_str);
    let d_pl_idx = MetalBuffer::from_slice(device, pl_idx);

    let params = DebugBruteForceParams {
        nh: nh as i32, np: np as i32, traverser: traverser as i32,
        starting_pot, fold_mask, _pad: 0,
        rake_rate: 0.0, rake_cap: 0.0, flop_seen: 0,
    };
    let d_params = MetalBuffer::from_slice(device, &[params]);

    let cmd = ctx.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(d_output.as_ref()), 0);
    enc.set_buffer(1, Some(d_opp_reach.as_ref()), 0);
    enc.set_buffer(2, Some(d_contributions.as_ref()), 0);
    enc.set_buffer(3, Some(d_hand_cards.as_ref()), 0);
    enc.set_buffer(4, Some(d_pl_str.as_ref()), 0);
    enc.set_buffer(5, Some(d_pl_idx.as_ref()), 0);
    enc.set_buffer(6, Some(d_params.as_ref()), 0);

    let grid = MTLSize { width: 1, height: 1, depth: 1 };
    let tg = MTLSize { width: 1, height: 1, depth: 1 };
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    d_output.to_vec()
}

mod helper {
    use super::*;
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

#[test]
#[ignore = "Step 2.A.1 node-28 input extraction. Run on demand."]
fn step_2a1_node28_input_extraction() {
    eprintln!("\n========================================================================");
    eprintln!("=== Step 2.A.1: node-28 input extraction at nh=1176                  ===");
    eprintln!("========================================================================\n");

    let (tree, game) = helper::build_full_nh_4board_game(/*stacks=*/5, /*pot=*/2);
    let table = game.table();
    let nh = table.num_valid;
    let nn = tree.num_nodes();
    let np = table.num_players as usize;
    let num_opp = np - 1;
    eprintln!("Setup: tree {} nodes, nh = {}, np = {}", nn, nh, np);

    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    cpu.compute_all_strategies(&tree);
    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 0, &cpu_flop_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 0, &cpu_turn_reach);

    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, 0);
    gpu.compute_reach_river(&ctx, 0, 0);
    let gpu_river_reach = gpu.download_river_reach();

    // ─── Extract node 28's inputs ───
    let node_id = 28usize;
    let traverser = 0usize;
    let opp = if traverser == 0 { 1 } else { 0 };
    let node_reach_base = node_id * np * nh;
    let cpu_opp_reach: Vec<f32> = cpu_river_reach[node_reach_base + opp * nh
        ..node_reach_base + opp * nh + nh].to_vec();
    let gpu_opp_reach: Vec<f32> = gpu_river_reach[node_reach_base + opp * nh
        ..node_reach_base + opp * nh + nh].to_vec();

    eprintln!("\n--- Node 28 opp_reach sample ---");
    eprintln!("  CPU opp_reach[0..6] = {:?}", &cpu_opp_reach[0..6]);
    eprintln!("  GPU opp_reach[0..6] = {:?}", &gpu_opp_reach[0..6]);
    let opp_reach_diff = cpu_opp_reach.iter().zip(gpu_opp_reach.iter())
        .filter(|(a, b)| (**a - **b).abs() > 1e-6).count();
    eprintln!("  opp_reach diff count (>1e-6): {} / {}", opp_reach_diff, nh);

    let cpu_opp_sum: f32 = cpu_opp_reach.iter().sum();
    let gpu_opp_sum: f32 = gpu_opp_reach.iter().sum();
    eprintln!("  CPU sum(opp_reach) = {:.4}, GPU sum(opp_reach) = {:.4}",
              cpu_opp_sum, gpu_opp_sum);

    let cpu_opp_nonzero = cpu_opp_reach.iter().filter(|x| x.abs() > 1e-9).count();
    let gpu_opp_nonzero = gpu_opp_reach.iter().filter(|x| x.abs() > 1e-9).count();
    eprintln!("  CPU opp_reach nonzero count: {}", cpu_opp_nonzero);
    eprintln!("  GPU opp_reach nonzero count: {}", gpu_opp_nonzero);

    let contributions: Vec<i32> = (0..np)
        .map(|p| tree.contributions[node_id * np + p]).collect();
    let fold_mask = tree.get_folded_mask(node_id);
    eprintln!("\n--- Node 28 other inputs ---");
    eprintln!("  contributions: {:?}", contributions);
    eprintln!("  fold_mask: {}", fold_mask);
    eprintln!("  starting_pot: {}", tree.starting_pot);

    // ─── CPU showdown reference ───
    let mut sorted_opp_str = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx = Vec::with_capacity(num_opp * nh);
    let tc_card = table.remaining_deck[0] as usize;
    let rc_card = table.river_decks[tc_card][0] as usize;
    let sorted_offset = (tc_card * 52 + rc_card) * num_opp * nh;
    for oi in 0..num_opp {
        sorted_opp_str.extend_from_slice(
            &table.river_sorted_str[sorted_offset + oi * nh..sorted_offset + (oi + 1) * nh]);
        sorted_opp_idx.extend_from_slice(
            &table.river_sorted_idx[sorted_offset + oi * nh..sorted_offset + (oi + 1) * nh]);
    }
    let pl_str: Vec<u16> = table.river_sorted_str[sorted_offset..sorted_offset + nh].to_vec();
    let pl_idx: Vec<u16> = table.river_sorted_idx[sorted_offset..sorted_offset + nh].to_vec();

    let opp_reach_views = vec![cpu_opp_reach.as_slice()];
    let cpu_cfv_ref = side_pot_showdown_cfv(
        &opp_reach_views, &table.hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &pl_str, &pl_idx,
        &contributions, fold_mask, traverser, np as u8, tree.starting_pot,
    );

    // ─── GPU debug helper with the extracted inputs ───
    let gpu_helper_out = gpu_brute_force(
        &ctx, nh, np, traverser, tree.starting_pot, fold_mask,
        &cpu_opp_reach, &contributions, &table.hand_cards, &pl_str, &pl_idx,
    );

    // ─── Compare ───
    let nc = table.num_combinations as f32;
    eprintln!("\n--- Comparison at h=0 (pre-divide) ---");
    eprintln!("  CPU showdown ref (helper input):   {:.4}", cpu_cfv_ref[0]);
    eprintln!("  GPU debug helper (helper input):   {:.4}", gpu_helper_out[0]);
    eprintln!("  Helper-vs-helper diff: {:.4}", (cpu_cfv_ref[0] - gpu_helper_out[0]).abs());

    eprintln!("\n  (For reference, the failing bottom_up_zone shows:");
    eprintln!("    CPU bottom_up_zone × nc = {:.4} (computed earlier as 329,728)", cpu_cfv_ref[0]);
    eprintln!("    GPU bottom_up_zone × nc = -3465 (-0.0025 × 1,382,976)");
    eprintln!("  )");

    eprintln!("\n--- Bigger-picture diff summary ---");
    let mut max_abs = 0.0f32;
    let mut nonzero_diffs = 0usize;
    for h in 0..nh {
        let d = (cpu_cfv_ref[h] - gpu_helper_out[h]).abs();
        if d > 1e-6 { nonzero_diffs += 1; }
        if d > max_abs { max_abs = d; }
    }
    eprintln!("  CPU showdown_cfv vs GPU debug helper:");
    eprintln!("    max_abs across all h: {:.4e}", max_abs);
    eprintln!("    nonzero_diffs:        {} / {}", nonzero_diffs, nh);

    eprintln!("\n--- Verdict ---");
    if max_abs < 1e-2 {
        eprintln!("  ✓ Debug helper with extracted inputs MATCHES CPU reference.");
        eprintln!("    → The helper produces the right answer for the extracted inputs.");
        eprintln!("    → bottom_up_zone(River) terminal-handler wrapping logic has the bug.");
        eprintln!("    → Bug is in vcfr.metal lines 877-935 specifically — the GPU is");
        eprintln!("      passing DIFFERENT inputs to the helper than what's in the buffers.");
        eprintln!("    → Next: inspect the per-line kernel code for what's modified at");
        eprintln!("      terminal handling vs the debug kernel's invocation.");
    } else {
        eprintln!("  ✗ Debug helper with extracted inputs ALSO diverges from CPU.");
        eprintln!("    max_abs = {:.4} >> 1e-2", max_abs);
        eprintln!("    → My extracted inputs differ from what the CPU bottom_up_zone uses.");
        eprintln!("      OR the helper has a different bug path triggered by sparse reach.");
        eprintln!("    → Need to instrument CPU bottom_up_zone to capture its exact inputs.");
    }
}
