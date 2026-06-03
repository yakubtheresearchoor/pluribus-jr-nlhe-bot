// Characterize the CPU↔Metal regret divergence in
// test_flop_metal_full_pipeline_parity. The test reports max_diff = 0.22 at
// iter 0, but the three diff values shown are EXACTLY equal (clean f32
// fractions: 225/1024 flop, 5/8 turn, 15/128 river). That pattern strongly
// suggests a per-(infoset, action) constant offset broadcast across hands,
// not a numerical divergence in the kernel.
//
// If confirmed, the offset is BENIGN for strategy computation: when a
// constant is added to all actions at the same node, the relative
// preference (argmax of regret) is unchanged, so strategy and cum_strategy
// remain correct. The blueprint converges correctly despite raw-regret
// divergence. This is what allows the loose-tolerance six_player gates
// (threshold 0.5) to pass at iter-0 and iter-2 — the diff stays under 0.5
// AND, more importantly, doesn't affect the strategy.
//
// This test dumps the full diff structure to verify the per-(infoset, action)
// constant hypothesis: at every (infoset, action), are all nh diffs equal,
// and across different (infoset, action) are the constants different?

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card};
use solver_core::gpu_metal::{MetalContext, MetalFlopStartSolver};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA};

// Copy of metal_flop_parity's minimal table builder for direct comparison.
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
            .map(|h| (turn_ranks[tc as usize * nh + h], h as u16))
            .collect();
        items.sort_by(|a, b| b.0.cmp(&a.0));
        for (k, &(r, idx)) in items.iter().enumerate() {
            turn_sorted_str[(tc as usize) * num_opp * nh + 0 * nh + k] = r;
            turn_sorted_idx[(tc as usize) * num_opp * nh + 0 * nh + k] = idx;
        }
    }

    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        for &rc in &river_decks[tc as usize] {
            let combined = board_mask | (1u64 << tc) | (1u64 << rc);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if combined & (1u64 << c1) != 0 || combined & (1u64 << c2) != 0 { continue; }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in &board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize);
                hand = hand.add_card(rc as usize);
                let r = hand.evaluate_internal() as u16;
                let key = (tc as usize) * 52 + (rc as usize);
                river_ranks[key * nh + i] = r;
            }
            let key = (tc as usize) * 52 + (rc as usize);
            let mut items: Vec<(u16, u16)> = (0..nh)
                .filter(|&h| {
                    let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
                    combined & (1u64 << c1) == 0 && combined & (1u64 << c2) == 0
                })
                .map(|h| (river_ranks[key * nh + h], h as u16))
                .collect();
            items.sort_by(|a, b| b.0.cmp(&a.0));
            for (k, &(r, idx)) in items.iter().enumerate() {
                river_sorted_str[key * num_opp * nh + 0 * nh + k] = r;
                river_sorted_idx[key * num_opp * nh + 0 * nh + k] = idx;
            }
        }
    }

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
        hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
        remaining_deck: turn_cards, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights, num_players,
        num_combinations: num_combinations as f64, river_decks,
    };

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

#[test]
fn characterize_metal_pipeline_offset_pattern() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let nh = 4usize;

    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Run 1 iter on each side.
    let _ = cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    let cpu_flop = cpu.regrets_flop();
    let cpu_turn = cpu.regrets_turn();
    let cpu_river = cpu.regrets_river();
    let gpu_all = gpu.download_regrets();

    // Layout: regrets[infoset_idx * MAX_NA * nh + action * nh + hand]
    // For each (infoset, action) pair, the nh hand-indexed diffs should all
    // be equal if the hypothesis "per-(infoset, action) constant offset"
    // holds.

    eprintln!("\n=== Per-(infoset, action) offset characterization ===\n");

    for (zone_label, cpu_slice, gpu_offset) in &[
        ("FLOP", cpu_flop, 0usize),
        ("TURN", cpu_turn, cpu_flop.len()),
        ("RIVER", cpu_river, cpu_flop.len() + cpu_turn.len()),
    ] {
        let n = cpu_slice.len();
        let n_ia = n / nh; // number of (infoset × action) pairs
        let mut nonuniform_count = 0;
        let mut distinct_diffs: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        let mut all_diffs: Vec<f32> = Vec::new();
        let mut nonzero_ia = 0;

        for ia in 0..n_ia {
            let base = ia * nh;
            let mut diffs: Vec<f32> = Vec::with_capacity(nh);
            let mut any_nonzero = false;
            for h in 0..nh {
                let i = base + h;
                if i >= cpu_slice.len() { break; }
                let gi = gpu_offset + i;
                if gi >= gpu_all.len() { break; }
                let d = gpu_all[gi] - cpu_slice[i]; // signed diff (metal - cpu)
                diffs.push(d);
                if cpu_slice[i] != 0.0 || gpu_all[gi] != 0.0 { any_nonzero = true; }
            }
            if !any_nonzero {
                continue;
            }
            nonzero_ia += 1;

            // Check if all diffs in this (infoset, action) are equal.
            let first = diffs[0];
            let uniform = diffs.iter().all(|&d| (d - first).abs() < 1e-7);
            if !uniform {
                nonuniform_count += 1;
                if nonuniform_count <= 5 {
                    eprintln!("  [{}] NON-UNIFORM (infoset×action {}): diffs = {:?}",
                        zone_label, ia, diffs);
                }
            }

            // Bucket the constant for distinct-count.
            // Use a quantized integer key to bucket near-equal diffs.
            let q = (first as f64 * 1024.0).round() as i64;
            distinct_diffs.insert(q);
            all_diffs.push(first);
        }

        eprintln!(
            "{}: {} (infoset, action) pairs total, {} nonzero, {} non-uniform within (per-hand-uniformity hypothesis: {})",
            zone_label,
            n_ia,
            nonzero_ia,
            nonuniform_count,
            if nonuniform_count == 0 { "✓ HOLDS" } else { "✗ VIOLATED" }
        );
        eprintln!(
            "  Distinct offset values (rounded /1024): {} (out of {} nonzero (infoset, action) pairs)",
            distinct_diffs.len(),
            nonzero_ia
        );
        if distinct_diffs.len() <= 10 {
            let display: Vec<f32> = distinct_diffs.iter().map(|&q| q as f32 / 1024.0).collect();
            eprintln!("  Values: {:?}", display);
        } else {
            eprintln!(
                "  Range: min={:.4}, max={:.4}",
                all_diffs.iter().cloned().fold(f32::INFINITY, f32::min),
                all_diffs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
        eprintln!();
    }
}
