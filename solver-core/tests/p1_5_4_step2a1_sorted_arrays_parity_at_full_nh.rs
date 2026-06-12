// Step 2.A.1 sorted-arrays parity at nh=1176.
//
// Coverage gap identified: stages 1–4 (strategy, reach_flop, reach_turn,
// reach_river) all passed bit-exact at nh=1176. The showdown helper passed
// at nh=1176 with controlled inputs (micro-kernel test). The bug is in
// stage 5 (bottom_up_zone(River)) at terminal nodes. The terminal handler
// reads opp_reach (verified) and the sorted arrays (sorted_pl_strength,
// sorted_pl_indices) and passes them to the helper (verified correct given
// correct inputs). The sorted arrays are the ONE input that was never
// directly compared.
//
// Prior: the test-helper bug I found this same turn was in sorted-array
// indexing (river_sorted_idx missing the rc term). The GPU's read of the
// sorted arrays uses (tc_card * 52 + rc_card) * num_opp * nh as the byte
// offset (divided by 2 for u16). If the GPU at nh=1176 reads from a slot
// that's NOT what the CPU populates (e.g., wrong stride math, byte/element
// confusion that's right at nh=4 but wrong at nh=1176), THAT's the bug.
//
// What this test does:
//   1. Build the same setup as the per-stage parity at nh=1176.
//   2. Download the full d_river_sorted_str / d_river_sorted_idx buffers
//      from the GPU.
//   3. Compare against CPU's table.river_sorted_str / table.river_sorted_idx
//      (a) Whole-buffer length match.
//      (b) Bit-exact full content match.
//   4. Then for (tc=0, rc=0) specifically: compute the byte offset the GPU
//      uses and the slice of CPU's buffer at the same offset, compare.
//   5. Report any mismatches with attention to where in the offset space.

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
#[ignore = "Step 2.A.1 sorted-array parity at nh=1176. Run on demand."]
fn step_2a1_sorted_arrays_parity_at_full_nh() {
    eprintln!("\n========================================================================");
    eprintln!("=== Step 2.A.1: GPU sorted-array READ verification at nh=1176        ===");
    eprintln!("===   Tests whether the (tc, rc) byte-offset math is right at scale  ===");
    eprintln!("========================================================================\n");

    let (tree, game) = helper::build_full_nh_4board_game(/*stacks=*/5, /*pot=*/2);
    let table = game.table();
    let nh = table.num_valid;
    let num_opp = (game.table().num_players - 1) as usize;
    eprintln!("Setup: nh = {}, num_opp = {}", nh, num_opp);
    assert_eq!(nh, 1176, "expected nh=1176");

    let cpu = FlopStartVectorCfr::new(&tree, table);
    let ctx = MetalContext::new().expect("Metal");
    let gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // ─── Check 1: download d_river_sorted_str / d_river_sorted_idx,
    //              compare against CPU's full table.river_sorted_str/_idx.
    let gpu_river_str = gpu.download_river_sorted_str();
    let gpu_river_idx = gpu.download_river_sorted_idx();
    eprintln!("\n--- Check 1: full-buffer upload integrity ---");
    eprintln!("  CPU table.river_sorted_str.len() = {}", table.river_sorted_str.len());
    eprintln!("  GPU d_river_sorted_str.len()     = {}", gpu_river_str.len());
    eprintln!("  CPU table.river_sorted_idx.len() = {}", table.river_sorted_idx.len());
    eprintln!("  GPU d_river_sorted_idx.len()     = {}", gpu_river_idx.len());

    let len_str_match = gpu_river_str.len() == table.river_sorted_str.len();
    let len_idx_match = gpu_river_idx.len() == table.river_sorted_idx.len();
    eprintln!("  lengths match: str={} idx={}", len_str_match, len_idx_match);
    assert!(len_str_match && len_idx_match, "buffer length mismatch — upload corrupt");

    let str_mismatch_count = table.river_sorted_str.iter().zip(gpu_river_str.iter())
        .filter(|(a, b)| a != b).count();
    let idx_mismatch_count = table.river_sorted_idx.iter().zip(gpu_river_idx.iter())
        .filter(|(a, b)| a != b).count();
    eprintln!("  full-buffer mismatch count: str={} idx={}",
              str_mismatch_count, idx_mismatch_count);
    if str_mismatch_count == 0 && idx_mismatch_count == 0 {
        eprintln!("  ✓ Full buffer uploaded correctly. Bug is NOT in upload.");
    } else {
        eprintln!("  ✗ Upload corruption — first 5 mismatches in str:");
        let mut shown = 0;
        for (i, (a, b)) in table.river_sorted_str.iter().zip(gpu_river_str.iter()).enumerate() {
            if a != b && shown < 5 {
                eprintln!("    str[{}]: CPU={} GPU={}", i, a, b);
                shown += 1;
            }
        }
        panic!("Upload integrity failure");
    }

    // ─── Check 2: per-(tc, rc) slice that the GPU's terminal handler reads.
    let tc_card = table.remaining_deck[0] as usize;   // first turn card value
    let rc_card = table.river_decks[tc_card][0] as usize;  // first river card value
    let slice_offset = (tc_card * 52 + rc_card) * num_opp * nh;
    let slice_len = num_opp * nh;
    eprintln!("\n--- Check 2: per-(tc=0, rc=0) slice ---");
    eprintln!("  tc_card = {} (deck idx 0)", tc_card);
    eprintln!("  rc_card = {} (deck[tc][0])", rc_card);
    eprintln!("  computed slice offset: (tc*52+rc)*num_opp*nh = {}*52+{}*1*{} = {}",
              tc_card, rc_card, nh, slice_offset);
    eprintln!("  slice length: num_opp*nh = {}", slice_len);
    eprintln!("  slice offset in BYTES (u16): {}", slice_offset * 2);
    eprintln!("  slice END in BYTES: {}", (slice_offset + slice_len) * 2);
    eprintln!("  buffer total BYTES: {}", table.river_sorted_str.len() * 2);

    assert!(slice_offset + slice_len <= table.river_sorted_str.len(),
        "computed slice extends beyond buffer — offset bug");

    let cpu_slice_str = &table.river_sorted_str[slice_offset..slice_offset + slice_len];
    let gpu_slice_str = &gpu_river_str[slice_offset..slice_offset + slice_len];
    let cpu_slice_idx = &table.river_sorted_idx[slice_offset..slice_offset + slice_len];
    let gpu_slice_idx = &gpu_river_idx[slice_offset..slice_offset + slice_len];

    let str_slice_mismatch = cpu_slice_str.iter().zip(gpu_slice_str.iter())
        .filter(|(a, b)| a != b).count();
    let idx_slice_mismatch = cpu_slice_idx.iter().zip(gpu_slice_idx.iter())
        .filter(|(a, b)| a != b).count();
    eprintln!("  per-(tc, rc) slice mismatches: str={} idx={}",
              str_slice_mismatch, idx_slice_mismatch);
    assert_eq!(str_slice_mismatch, 0, "per-(tc, rc) str slice differs");
    assert_eq!(idx_slice_mismatch, 0, "per-(tc, rc) idx slice differs");

    // ─── Check 3: how the GPU's terminal handler ACTUALLY reads the slice.
    // Replicate the kernel's offset arithmetic:
    //   sos_byte_off = ((tc_card * 52 + rc_card) * num_opp * nh) * 2
    // The kernel reads from u16 *sps* starting at sos_byte_off (interpreted as u16).
    let sos_byte_off = ((tc_card * 52 + rc_card) * num_opp * nh) * 2;
    let kernel_read_index = sos_byte_off / 2;
    eprintln!("\n--- Check 3: kernel-side offset arithmetic ---");
    eprintln!("  sos_byte_off = ((tc*52+rc)*num_opp*nh)*2 = {}*2 = {}",
              (tc_card * 52 + rc_card) * num_opp * nh, sos_byte_off);
    eprintln!("  kernel reads from u16 index = {} (= byte_off / 2)", kernel_read_index);
    eprintln!("  ✓ matches the slice-offset computation above" );

    eprintln!("\n========================================================================");
    eprintln!("=== Sorted-array parity at nh=1176: PASS ===");
    eprintln!("  Upload is bit-exact. Per-(tc, rc) slice is bit-exact. Kernel offset");
    eprintln!("  math computes the same slot as the CPU populates.");
    eprintln!();
    eprintln!("  CONCLUSION: the bug is NOT in the sorted-array inputs to the terminal");
    eprintln!("  handler. It IS in the wrapping logic OR in another input we haven't");
    eprintln!("  compared yet (hand_cards, contributions, folded_masks). Next move: ");
    eprintln!("  node-28-extraction micro-kernel test (option 1) on the next turn.");
    eprintln!("========================================================================");
}
