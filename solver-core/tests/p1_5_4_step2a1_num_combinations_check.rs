// Step 2.A.1 continuation: check num_combinations divisor at nh=1176.
//
// the lead's handoff: the symptom (GPU values smaller than CPU by ~100x) +
// scale-dependence (works at nh=4, fails at nh=1176) + scale-axis (num_
// combinations is the ONE operation in the wrapping window whose value
// depends on nh) converges on the divisor as the prime suspect.
//
// Concrete checks (in order, cheapest first):
//   1. Print table.num_combinations at nh=1176 in this test setup.
//   2. Run bottom_up_zone(River, 0, 0) on both CPU and GPU; at node 28,
//      extract pre-divide CFV (cpu_value × num_combinations and
//      gpu_value × num_combinations). If they MATCH, the divisor is the
//      ONLY difference → bug is the divide-by-num_combinations.
//   3. If pre-divide values DIFFER, the helper itself produces different
//      output at terminal nodes when called from bottom_up_zone (despite
//      the controlled-input micro test passing). That means some input
//      to the helper is different at terminal nodes — fall through to
//      node-28 input extraction (next step).

#![cfg(feature = "metal")]

use solver_core::card::{
    card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS,
};
use solver_core::gpu_metal::{GpuDcfrParams, MetalContext, MetalFlopStartSolver};
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
#[ignore = "Step 2.A.1 divisor check at nh=1176. Run on demand."]
fn step_2a1_num_combinations_divisor_check() {
    eprintln!("\n========================================================================");
    eprintln!("=== Step 2.A.1: num_combinations divisor check at nh=1176            ===");
    eprintln!("===   the lead's prime suspect: scale-dependent divide                  ===");
    eprintln!("========================================================================\n");

    let (tree, game) = helper::build_full_nh_4board_game(/*stacks=*/5, /*pot=*/2);
    let table = game.table();
    let nh = table.num_valid;
    let nn = tree.num_nodes();
    eprintln!("Setup: tree {} nodes, nh = {}", nn, nh);

    // ─── Check 1: num_combinations value ───
    let nc_f64 = table.num_combinations;
    let nc_f32 = nc_f64 as f32;
    eprintln!("\n--- Check 1: num_combinations value ---");
    eprintln!("  table.num_combinations (f64): {:.6}", nc_f64);
    eprintln!("  cast to f32:                  {:.6}", nc_f32);
    eprintln!("  f64-to-f32 precision loss:    {:.6e}", nc_f64 - nc_f32 as f64);
    eprintln!("  CPU bottom_up_zone divides by this f32 cast (same on GPU).");

    // ─── Check 2: run bottom_up_zone(River, 0, 0) on both, extract node 28 ───
    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    cpu.compute_all_strategies(&tree);
    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 0, &cpu_flop_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 0, &cpu_turn_reach);
    let cpu_params = DcfrParams::new(0);
    let mut cpu_cfv = vec![0.0f32; nn * nh];
    cpu.bottom_up_zone(&tree, table, 0, &cpu_river_reach, &mut cpu_cfv,
                       Zone::River, Some(0), Some(0), &cpu_params);

    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, 0);
    gpu.compute_reach_river(&ctx, 0, 0);
    let gpu_params = GpuDcfrParams::new(0);
    gpu.bottom_up_river(&ctx, 0, 0, 0, &gpu_params);
    let gpu_cfv_batch = gpu.download_river_cfv_batch();
    let gpu_cfv = &gpu_cfv_batch[0..nn*nh];

    eprintln!("\n--- Check 2: node 28 CFV pre- and post-divide ---");
    let node_id = 28;
    let h = 0;
    let idx = node_id * nh + h;
    let cpu_val = cpu_cfv[idx];
    let gpu_val = gpu_cfv[idx];
    let cpu_pre_divide = cpu_val * nc_f32;
    let gpu_pre_divide = gpu_val * nc_f32;

    let node_28 = &tree.nodes[node_id];
    let ntype = match node_28.node_type { 0 => "TERMINAL", 1 => "CHANCE", 2 => "PLAYER", _ => "???" };
    eprintln!("  node {} type: {} ({})", node_id, node_28.node_type, ntype);
    eprintln!("  num_children: {}", node_28.num_children);
    eprintln!("  contributions: {:?}", &tree.contributions[node_id*2..node_id*2+2]);
    eprintln!("  folded_mask:   {}", tree.get_folded_mask(node_id));

    eprintln!("\n  num_combinations divisor = {:.6}", nc_f32);
    eprintln!();
    eprintln!("  CPU cfv[node=28, h=0] = {:.6}  (post-divide)", cpu_val);
    eprintln!("  GPU cfv[node=28, h=0] = {:.6}  (post-divide)", gpu_val);
    eprintln!();
    eprintln!("  CPU × num_combinations = {:.6}  (pre-divide)", cpu_pre_divide);
    eprintln!("  GPU × num_combinations = {:.6}  (pre-divide)", gpu_pre_divide);

    let post_diff = (cpu_val - gpu_val).abs();
    let pre_diff = (cpu_pre_divide - gpu_pre_divide).abs();
    eprintln!();
    eprintln!("  post-divide diff: {:.6}", post_diff);
    eprintln!("  pre-divide diff:  {:.6}", pre_diff);

    eprintln!("\n--- Verdict ---");
    if pre_diff < 1e-3 && post_diff > 1e-3 {
        eprintln!("  ✗ Pre-divide MATCHES, post-divide DIFFERS.");
        eprintln!("    → Bug is in the DIVIDE-BY-NUM_COMBINATIONS, not the helper output.");
        eprintln!("    → the lead's prime suspect confirmed. Inspect divisor in GPU vs CPU.");
    } else if pre_diff > 1e-3 {
        eprintln!("  ✗ Pre-divide DIFFERS (diff = {:.4}).", pre_diff);
        eprintln!("    → Helper output itself differs between CPU and GPU at this terminal.");
        eprintln!("    → Despite controlled-input micro-test passing. Some input to the");
        eprintln!("      helper at node {} differs between CPU and GPU.", node_id);
        eprintln!("    → Fall through to node-28 input extraction (next step).");
        eprintln!();
        // Ratios to hint at the cause.
        let ratio = if gpu_val.abs() > 1e-9 { cpu_val / gpu_val } else { 0.0 };
        eprintln!("  Diagnostic ratio (CPU/GPU post-divide): {:.4}", ratio);
        eprintln!("  Diagnostic ratio (CPU/GPU pre-divide):  {:.4}",
                  if gpu_pre_divide.abs() > 1e-9 { cpu_pre_divide / gpu_pre_divide } else { 0.0 });
        eprintln!("  If ratio is integer-like (1, 2, 100), suggests missing factor.");
        eprintln!("  If ratio is ~num_combinations, suggests double-divide or no-divide.");
    } else {
        eprintln!("  ✓ Pre- AND post-divide both match. Node 28 is fine, must look at");
        eprintln!("    a DIFFERENT diverging node (per-stage test showed 12,870 entries).");
    }
}
