#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::hand::eval::Hand;
use solver_core::solver::chance_table::ChanceTable;
use solver_core::tree::action::BoardState;
use solver_core::tree::flat::{FlatNode, FlatTree};

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

fn compute_cpu_showdown_cfv_base_ranks(
    table: &ChanceTable,
    traverser: usize,
    contribution: i32,
) -> Vec<f32> {
    let nh = table.num_valid;
    let num_opp = 1;
    let opp = 1 - traverser;

    let hand_ranks = &table.hand_ranks_base;

    let mut opp_reach = vec![0.0f32; nh];
    for h in 0..nh {
        opp_reach[h] = table.initial_weights[opp][h];
    }

    let mut items: Vec<(u16, usize)> = (0..nh)
        .map(|h| (hand_ranks[h] + 1, h))
        .collect();
    items.sort_by_key(|&(s, _)| s);

    let sorted_strength: Vec<u16> = items.iter().map(|&(s, _)| s).collect();
    let sorted_indices: Vec<usize> = items.iter().map(|&(_, i)| i).collect();

    let mut cfv = vec![0.0f32; nh];

    for _oi in 0..num_opp {
        let opp_str = &sorted_strength;
        let opp_idx = &sorted_indices;

        let mut cfreach_sum = 0.0f32;
        let mut cfreach_minus = vec![0.0f32; 52];

        let mut i = 0;
        for si in 0..nh {
            let str_h = sorted_strength[si];
            let h = sorted_indices[si];
            while i < nh && opp_str[i] < str_h {
                let ho = opp_idx[i];
                let r = opp_reach[ho];
                if r != 0.0 {
                    cfreach_sum += r;
                    cfreach_minus[table.hand_cards[ho * 2] as usize] += r;
                    cfreach_minus[table.hand_cards[ho * 2 + 1] as usize] += r;
                }
                i += 1;
            }
            let cfreach = cfreach_sum
                - cfreach_minus[table.hand_cards[h * 2] as usize]
                - cfreach_minus[table.hand_cards[h * 2 + 1] as usize];
            cfv[h] += contribution as f32 * cfreach;
        }

        cfreach_sum = 0.0;
        for c in 0..52 { cfreach_minus[c] = 0.0; }

        i = nh;
        for si in (0..nh).rev() {
            let str_h = sorted_strength[si];
            let h = sorted_indices[si];
            while i > 0 && opp_str[i - 1] > str_h {
                i -= 1;
                let ho = opp_idx[i];
                let r = opp_reach[ho];
                if r != 0.0 {
                    cfreach_sum += r;
                    cfreach_minus[table.hand_cards[ho * 2] as usize] += r;
                    cfreach_minus[table.hand_cards[ho * 2 + 1] as usize] += r;
                }
            }
            let cfreach = cfreach_sum
                - cfreach_minus[table.hand_cards[h * 2] as usize]
                - cfreach_minus[table.hand_cards[h * 2 + 1] as usize];
            cfv[h] += (-contribution as f32) * cfreach;
        }
    }

    cfv
}

fn compute_cpu_showdown_cfv(
    board: &[Card],
    river_card: Card,
    traverser: usize,
    contribution: i32,
    table: &ChanceTable,
) -> Vec<f32> {
    let nh = table.num_valid;
    let num_opp = 1;

    let mut hand_ranks = vec![0u16; nh];
    for (i, &vi) in table.valid_hand_indices.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(vi as usize);
        let mut hand = Hand::new();
        hand = hand.add_card(c1 as usize);
        hand = hand.add_card(c2 as usize);
        for &bc in board {
            hand = hand.add_card(bc as usize);
        }
        hand = hand.add_card(river_card as usize);
        hand_ranks[i] = hand.evaluate();
    }

    let opp = 1 - traverser;
    let mut opp_reach = vec![0.0f32; nh];
    for h in 0..nh {
        let vi = table.valid_hand_indices[h] as usize;
        opp_reach[h] = table.initial_weights[opp][h];
    }

    let mut items: Vec<(u16, usize)> = (0..nh)
        .map(|h| (hand_ranks[h] + 1, h))
        .collect();
    items.sort_by_key(|&(s, _)| s);

    let sorted_strength: Vec<u16> = items.iter().map(|&(s, _)| s).collect();
    let sorted_indices: Vec<usize> = items.iter().map(|&(_, i)| i).collect();

    let mut cfv = vec![0.0f32; nh];

    for oi in 0..num_opp {
        let opp_str = &sorted_strength;
        let opp_idx = &sorted_indices;

        let mut cfreach_sum = 0.0f32;
        let mut cfreach_minus = vec![0.0f32; 52];

        let mut i = 0;
        for si in 0..nh {
            let str_h = sorted_strength[si];
            let h = sorted_indices[si];
            while i < nh && opp_str[i] < str_h {
                let ho = opp_idx[i];
                let r = opp_reach[ho];
                if r != 0.0 {
                    cfreach_sum += r;
                    cfreach_minus[table.hand_cards[ho * 2] as usize] += r;
                    cfreach_minus[table.hand_cards[ho * 2 + 1] as usize] += r;
                }
                i += 1;
            }
            let cfreach = cfreach_sum
                - cfreach_minus[table.hand_cards[h * 2] as usize]
                - cfreach_minus[table.hand_cards[h * 2 + 1] as usize];
            cfv[h] += contribution as f32 * cfreach;
        }

        cfreach_sum = 0.0;
        for c in 0..52 { cfreach_minus[c] = 0.0; }

        i = nh;
        for si in (0..nh).rev() {
            let str_h = sorted_strength[si];
            let h = sorted_indices[si];
            while i > 0 && opp_str[i - 1] > str_h {
                i -= 1;
                let ho = opp_idx[i];
                let r = opp_reach[ho];
                if r != 0.0 {
                    cfreach_sum += r;
                    cfreach_minus[table.hand_cards[ho * 2] as usize] += r;
                    cfreach_minus[table.hand_cards[ho * 2 + 1] as usize] += r;
                }
            }
            let cfreach = cfreach_sum
                - cfreach_minus[table.hand_cards[h * 2] as usize]
                - cfreach_minus[table.hand_cards[h * 2 + 1] as usize];
            cfv[h] += (-contribution as f32) * cfreach;
        }
    }

    cfv
}

fn build_big_bet_tree() -> FlatTree {
    let mut tree = FlatTree::new(2, 200, vec![200, 200], 0.0, 0.0);

    let n_root = tree.alloc_node(FlatNode::player(0, BoardState::Turn, 0));
    tree.set_contribution(n_root, 0, 5);
    tree.set_contribution(n_root, 1, 100);

    let n_showdown = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n_showdown, 0, 100);
    tree.set_contribution(n_showdown, 1, 100);

    let n_fold = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n_fold, 0, 5);
    tree.set_contribution(n_fold, 1, 100);

    tree.set_children(n_root, vec![n_showdown as u32, n_fold as u32]);

    tree.set_folded_mask(n_fold, 0b01);

    tree
}

fn find_hand_idx(table: &ChanceTable, c1: Card, c2: Card) -> Option<usize> {
    let target = solver_core::card::card_pair_to_index(c1, c2);
    table.valid_hand_indices.iter().position(|&vi| vi as usize == target)
}

#[test]
fn gpu_verify_pure_showdown_no_fold() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();

    let cpu_cfv = compute_cpu_showdown_cfv_base_ranks(&table, 0, 5);

    // Tree: root(player 0, 1 action: showdown) → terminal
    let mut tree = FlatTree::new(2, 10, vec![95, 95], 0.0, 0.0);
    let n_root = tree.alloc_node(FlatNode::player(0, BoardState::Turn, 0));
    tree.set_contribution(n_root, 0, 5);
    tree.set_contribution(n_root, 1, 5);

    let n_showdown = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n_showdown, 0, 5);
    tree.set_contribution(n_showdown, 1, 5);

    let n_check = tree.alloc_node(FlatNode::player(1, BoardState::Turn, 0));
    tree.set_contribution(n_check, 0, 5);
    tree.set_contribution(n_check, 1, 5);

    let n_showdown2 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n_showdown2, 0, 5);
    tree.set_contribution(n_showdown2, 1, 5);

    tree.set_children(n_root, vec![n_check as u32]);
    tree.set_children(n_check, vec![n_showdown as u32]);

    let (sorted_opp_strength, sorted_opp_indices, sorted_player_strength, sorted_player_indices, same_hand_idx) =
        table.sorted_opp_arrays();

    let mut solver = gpu
        .create_nplayer_solver(
            &tree,
            nh,
            &table.hand_ranks_gpu(),
            &sorted_opp_strength,
            &sorted_opp_indices,
            &sorted_player_strength,
            &sorted_player_indices,
            &same_hand_idx,
            &table.hand_cards_gpu(),
            &table.initial_weight_flat(),
            None,
            &[],
            None,
            None,
        )
        .expect("solver creation failed");

    solver.run(32, 100).expect("GPU run failed");

    // With 1 action each, regrets should be ~0 and cum_strategy = reach
    let regrets = solver.download_regrets().expect("download failed");
    let cum_strategy = solver.download_cum_strategy().expect("download cum_strategy failed");

    // Node 0 (player 0): 1 action, offset 0
    // Node 2 (player 1): 1 action, offset nh
    let node0_cum: Vec<f32> = (0..nh).map(|h| cum_strategy[0 * nh + h]).collect();
    let node2_cum: Vec<f32> = (0..nh).map(|h| cum_strategy[1 * nh + h]).collect();

    println!("Pure showdown (no fold):");
    println!("Node 0 cum_strategy (first 5): {:?}", &node0_cum[..5.min(nh)]);
    println!("Node 2 cum_strategy (first 5): {:?}", &node2_cum[..5.min(nh)]);
    println!("Node 0 regrets (first 5): {:?}", &(0..5.min(nh)).map(|h| regrets[0*nh+h]).collect::<Vec<_>>());
    println!("Node 2 regrets (first 5): {:?}", &(0..5.min(nh)).map(|h| regrets[1*nh+h]).collect::<Vec<_>>());

    let cpu_sum: f32 = cpu_cfv.iter().sum();
    let cpu_pos = cpu_cfv.iter().filter(|&&v| v > 0.0).count();
    let cpu_neg = cpu_cfv.iter().filter(|&&v| v < 0.0).count();
    println!("CPU base ranks: sum={:.2}, positive={}, negative={}", cpu_sum, cpu_pos, cpu_neg);
}

#[test]
fn gpu_standalone_showdown_with_chance_ranks() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let river_card = card_from_str("9s").unwrap();

    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();

    let (chance_sorted_str, chance_sorted_idx) = table.chance_sorted_arrays_gpu();

    let ch_stride = (table.num_players as usize - 1) * nh;
    let chance_str_slice = &chance_sorted_str[river_card as usize * ch_stride..river_card as usize * ch_stride + ch_stride];
    let chance_idx_slice = &chance_sorted_idx[river_card as usize * ch_stride..river_card as usize * ch_stride + ch_stride];

    let hand_cards = table.hand_cards_gpu();
    let opp_reach = vec![1.0f32; nh];

    let gpu_output = gpu.run_test_showdown(
        chance_str_slice, chance_idx_slice,
        chance_str_slice, chance_idx_slice,
        &hand_cards, &opp_reach,
        nh, 5.0,
    ).expect("test showdown failed");

    let cpu_cfv = compute_cpu_showdown_cfv(&board, river_card, 0, 5, &table);

    let mut sign_agree = 0;
    let mut sign_disagree = 0;
    let mut max_diff = 0.0f32;

    for h in 0..nh {
        let gpu_v = gpu_output[h];
        let cpu_v = cpu_cfv[h];
        let diff = (gpu_v - cpu_v).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if (gpu_v > 0.0) == (cpu_v > 0.0) {
            sign_agree += 1;
        } else {
            sign_disagree += 1;
        }
    }

    let gpu_pos = gpu_output.iter().filter(|&&v| v > 0.0).count();
    let gpu_neg = gpu_output.iter().filter(|&&v| v < 0.0).count();
    let gpu_sum: f32 = gpu_output.iter().sum();
    let cpu_pos = cpu_cfv.iter().filter(|&&v| v > 0.0).count();
    let cpu_neg = cpu_cfv.iter().filter(|&&v| v < 0.0).count();
    let cpu_sum: f32 = cpu_cfv.iter().sum();

    println!("Standalone showdown with chance-sorted arrays (river 9s):");
    println!("GPU: positive={}, negative={}, sum={:.2}", gpu_pos, gpu_neg, gpu_sum);
    println!("CPU: positive={}, negative={}, sum={:.2}", cpu_pos, cpu_neg, cpu_sum);
    println!("Sign agreement: {}/{} ({:.1}%)", sign_agree, nh, 100.0 * sign_agree as f32 / nh as f32);
    println!("Max abs diff: {:.4}", max_diff);

    assert!(sign_disagree < nh / 20,
        "Chance-sorted showdown: too many sign disagreements: {}/{}", sign_disagree, nh);
    assert!(max_diff < 1.0, "Max diff too large: {:.4}", max_diff);
}

#[test]
fn gpu_standalone_showdown_kernel() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();

    let (s_opp_str, s_opp_idx, s_pl_str, s_pl_idx, _) = table.sorted_opp_arrays();
    let hand_cards = table.hand_cards_gpu();
    let opp_reach = vec![1.0f32; nh];

    let gpu_output = gpu.run_test_showdown(
        &s_opp_str, &s_opp_idx,
        &s_pl_str, &s_pl_idx,
        &hand_cards, &opp_reach,
        nh, 5.0,
    ).expect("test showdown failed");

    let cpu_cfv = compute_cpu_showdown_cfv_base_ranks(&table, 0, 5);

    let mut sign_agree = 0;
    let mut sign_disagree = 0;
    let mut max_diff = 0.0f32;
    let mut worst_h = 0;

    for h in 0..nh {
        let gpu_v = gpu_output[h];
        let cpu_v = cpu_cfv[h];
        let diff = (gpu_v - cpu_v).abs();
        if diff > max_diff {
            max_diff = diff;
            worst_h = h;
        }
        if (gpu_v > 0.0) == (cpu_v > 0.0) {
            sign_agree += 1;
        } else {
            sign_disagree += 1;
        }
    }

    let gpu_pos = gpu_output.iter().filter(|&&v| v > 0.0).count();
    let gpu_neg = gpu_output.iter().filter(|&&v| v < 0.0).count();
    let gpu_zero = gpu_output.iter().filter(|&&v| v == 0.0).count();
    let gpu_sum: f32 = gpu_output.iter().sum();

    println!("Standalone showdown kernel:");
    println!("GPU: positive={}, negative={}, zero={}, sum={:.2}", gpu_pos, gpu_neg, gpu_zero, gpu_sum);
    println!("CPU: positive={}, negative={}, zero=0, sum=0.00",
        cpu_cfv.iter().filter(|&&v| v > 0.0).count(),
        cpu_cfv.iter().filter(|&&v| v < 0.0).count());
    println!("Sign agreement: {}/{} ({:.1}%)", sign_agree, nh, 100.0 * sign_agree as f32 / nh as f32);
    println!("Max abs diff: {:.4} at hand {}", max_diff, worst_h);

    if sign_disagree > 0 {
        let vi = table.valid_hand_indices[worst_h] as usize;
        let (c1, c2) = index_to_card_pair(vi);
        let s1 = solver_core::card::card_to_string(c1).unwrap();
        let s2 = solver_core::card::card_to_string(c2).unwrap();
        println!("Worst hand: {}{} gpu={:.2} cpu={:.2}", s1, s2, gpu_output[worst_h], cpu_cfv[worst_h]);
    }

    // Print first 20 GPU vs CPU
    for h in 0..20.min(nh) {
        let vi = table.valid_hand_indices[h] as usize;
        let (c1, c2) = index_to_card_pair(vi);
        let s1 = solver_core::card::card_to_string(c1).unwrap();
        let s2 = solver_core::card::card_to_string(c2).unwrap();
        if h < 5 || (gpu_output[h] - cpu_cfv[h]).abs() > 1.0 {
            println!("  h={}: {}{} gpu={:.2} cpu={:.2} diff={:.2}",
                h, s1, s2, gpu_output[h], cpu_cfv[h], gpu_output[h] - cpu_cfv[h]);
        }
    }

    assert!(sign_disagree < nh / 20,
        "Standalone showdown: too many sign disagreements: {}/{}", sign_disagree, nh);
}

#[test]
fn gpu_verify_river_no_chance_baseline() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let full_board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let board4 = &full_board[..4];

    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(board4, &ranges, 2);
    let nh = table.num_valid_hands();

    let cpu_showdown_cfv = compute_cpu_showdown_cfv_base_ranks(&table, 0, 95);

    let unique_ranks: std::collections::HashSet<u16> = table.hand_ranks_base.iter().copied().collect();
    println!("Hand ranks: {} unique out of {}", unique_ranks.len(), nh);

    let tree = build_big_bet_tree();

    let (sorted_opp_strength, sorted_opp_indices, sorted_player_strength, sorted_player_indices, same_hand_idx) =
        table.sorted_opp_arrays();

    let mut solver = gpu
        .create_nplayer_solver(
            &tree,
            nh,
            &table.hand_ranks_gpu(),
            &sorted_opp_strength,
            &sorted_opp_indices,
            &sorted_player_strength,
            &sorted_player_indices,
            &same_hand_idx,
            &table.hand_cards_gpu(),
            &table.initial_weight_flat(),
            None,
            &[],
            None,
            None,
        )
        .expect("solver creation failed");

    solver.run(32, 200).expect("GPU run failed");

    let regrets = solver.download_regrets().expect("download failed");
    let regret_check: Vec<f32> = (0..nh).map(|h| regrets[0 * nh + h]).collect();
    let regret_fold: Vec<f32> = (0..nh).map(|h| regrets[1 * nh + h]).collect();

    let mut sign_agree = 0;
    let mut sign_disagree = 0;

    for h in 0..nh {
        let gpu_prefers_check = regret_check[h] > regret_fold[h];
        let cpu_positive_cfv = cpu_showdown_cfv[h] > -5.0 * (nh as f32 - 2.0);
        if gpu_prefers_check == cpu_positive_cfv {
            sign_agree += 1;
        } else {
            sign_disagree += 1;
        }
    }

    let gpu_fold = (0..nh).filter(|&h| regret_fold[h] > regret_check[h]).count();
    let gpu_check = nh - gpu_fold;
    println!("Big-bet no-chance: check={}/{}, fold={}/{}", gpu_check, nh, gpu_fold, nh);

    assert!(gpu_fold > nh / 4, "Expected >25% folds, got {}/{}", gpu_fold, nh);
    assert!(gpu_check > nh / 4, "Expected >25% checks, got {}/{}", gpu_check, nh);
}

#[test]
fn gpu_verify_known_hand_signs() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();

    let tree = build_big_bet_tree();

    let (sorted_opp_strength, sorted_opp_indices, sorted_player_strength, sorted_player_indices, same_hand_idx) =
        table.sorted_opp_arrays();

    let mut solver = gpu
        .create_nplayer_solver(
            &tree,
            nh,
            &table.hand_ranks_gpu(),
            &sorted_opp_strength,
            &sorted_opp_indices,
            &sorted_player_strength,
            &sorted_player_indices,
            &same_hand_idx,
            &table.hand_cards_gpu(),
            &table.initial_weight_flat(),
            None,
            &[],
            None,
            None,
        )
        .expect("solver creation failed");

    solver.run(32, 200).expect("GPU run failed");

    let regrets = solver.download_regrets().expect("download failed");
    let regret_check: Vec<f32> = (0..nh).map(|h| regrets[0 * nh + h]).collect();
    let regret_fold: Vec<f32> = (0..nh).map(|h| regrets[1 * nh + h]).collect();

    let as_kh = find_hand_idx(&table, card_from_str("As").unwrap(), card_from_str("Kh").unwrap());
    let tc_3c = find_hand_idx(&table, card_from_str("Tc").unwrap(), card_from_str("3c").unwrap());
    let kd_7c = find_hand_idx(&table, card_from_str("Kd").unwrap(), card_from_str("7c").unwrap());

    if let Some(i) = as_kh {
        println!("AsKh: regret_check={:.2}, regret_fold={:.2}", regret_check[i], regret_fold[i]);
        assert!(regret_check[i] > regret_fold[i],
            "AsKh should prefer check (showdown) over fold: check_regret={:.2}, fold_regret={:.2}",
            regret_check[i], regret_fold[i]);
    }

    if let Some(i) = tc_3c {
        println!("Tc3c: regret_check={:.2}, regret_fold={:.2}", regret_check[i], regret_fold[i]);
        assert!(regret_fold[i] > regret_check[i],
            "Tc3c (weak hand facing big bet) should prefer fold: check_regret={:.2}, fold_regret={:.2}",
            regret_check[i], regret_fold[i]);
    }

    if let Some(i) = kd_7c {
        println!("Kd7c (two pair): regret_check={:.2}, regret_fold={:.2}", regret_check[i], regret_fold[i]);
        assert!(regret_check[i] > regret_fold[i],
            "Kd7c (two pair) should prefer check over fold: check_regret={:.2}, fold_regret={:.2}",
            regret_check[i], regret_fold[i]);
    }

    let gpu_fold = (0..nh).filter(|&h| regret_fold[h] > regret_check[h]).count();
    let gpu_check = nh - gpu_fold;
    println!("Check preferred: {}/{} ({:.1}%), Fold preferred: {}/{} ({:.1}%)",
        gpu_check, nh, 100.0 * gpu_check as f32 / nh as f32,
        gpu_fold, nh, 100.0 * gpu_fold as f32 / nh as f32);
}

#[test]
fn gpu_verify_exact_cfv_vs_cpu() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let river_card = card_from_str("9s").unwrap();

    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();

    let cpu_cfv = compute_cpu_showdown_cfv(&board, river_card, 0, 95, &table);

    let cpu_positive = cpu_cfv.iter().filter(|&&v| v > 0.0).count();
    let cpu_negative = cpu_cfv.iter().filter(|&&v| v < 0.0).count();
    println!("CPU showdown CFV (contribution=95): positive={}, negative={}", cpu_positive, cpu_negative);

    let tree = build_big_bet_tree();
    let remaining_deck = vec![river_card];

    let (chance_sorted_str, chance_sorted_idx) = table.chance_sorted_arrays_gpu();
    let (sorted_opp_strength, sorted_opp_indices, sorted_player_strength, sorted_player_indices, same_hand_idx) =
        table.sorted_opp_arrays();

    let mut solver = gpu
        .create_nplayer_solver(
            &tree,
            nh,
            &table.hand_ranks_gpu(),
            &sorted_opp_strength,
            &sorted_opp_indices,
            &sorted_player_strength,
            &sorted_player_indices,
            &same_hand_idx,
            &table.hand_cards_gpu(),
            &table.initial_weight_flat(),
            Some(&table.chance_ranks_gpu()),
            &remaining_deck,
            Some(&chance_sorted_str),
            Some(&chance_sorted_idx),
        )
        .expect("solver creation failed");

    solver.run(32, 200).expect("GPU run failed");

    let regrets = solver.download_regrets().expect("download failed");
    let regret_check: Vec<f32> = (0..nh).map(|h| regrets[0 * nh + h]).collect();
    let regret_fold: Vec<f32> = (0..nh).map(|h| regrets[1 * nh + h]).collect();

    let mut sign_agree = 0;
    let mut sign_disagree = 0;

    for h in 0..nh {
        let gpu_prefers_check = regret_check[h] > regret_fold[h];
        let cpu_positive_cfv = cpu_cfv[h] > 0.0;
        if gpu_prefers_check == cpu_positive_cfv {
            sign_agree += 1;
        } else {
            sign_disagree += 1;
        }
    }

    let gpu_fold = (0..nh).filter(|&h| regret_fold[h] > regret_check[h]).count();
    let gpu_check = nh - gpu_fold;
    println!("GPU MCCFR (big-bet): check={}/{}, fold={}/{}", gpu_check, nh, gpu_fold, nh);
    println!("Sign agreement: {}/{} ({:.1}%)", sign_agree, nh, 100.0 * sign_agree as f32 / nh as f32);

    assert!(gpu_fold > nh / 4, "Expected >25% folds, got {}/{}", gpu_fold, nh);
    assert!(gpu_check > nh / 4, "Expected >25% checks, got {}/{}", gpu_check, nh);
}

#[test]
fn gpu_verify_multiple_rivers_sign_agreement() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let test_rivers: Vec<Card> = ["Ac", "Td", "5s", "Qh", "9c", "2s", "Kd", "3h"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let tree = build_big_bet_tree();

    let mut all_pass = true;
    for &river_card in &test_rivers {
        let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
        let nh = table.num_valid_hands();

        let remaining_deck = vec![river_card];
        let (chance_sorted_str, chance_sorted_idx) = table.chance_sorted_arrays_gpu();
        let (sorted_opp_strength, sorted_opp_indices, sorted_player_strength, sorted_player_indices, same_hand_idx) =
            table.sorted_opp_arrays();

        let mut solver = gpu
            .create_nplayer_solver(
                &tree,
                nh,
                &table.hand_ranks_gpu(),
                &sorted_opp_strength,
                &sorted_opp_indices,
                &sorted_player_strength,
                &sorted_player_indices,
                &same_hand_idx,
                &table.hand_cards_gpu(),
                &table.initial_weight_flat(),
                Some(&table.chance_ranks_gpu()),
                &remaining_deck,
                Some(&chance_sorted_str),
                Some(&chance_sorted_idx),
            )
            .expect("solver creation failed");

        solver.run(32, 200).expect("GPU run failed");

        let regrets = solver.download_regrets().expect("download failed");
        let regret_check: Vec<f32> = (0..nh).map(|h| regrets[0 * nh + h]).collect();
        let regret_fold: Vec<f32> = (0..nh).map(|h| regrets[1 * nh + h]).collect();

        let gpu_fold = (0..nh).filter(|&h| regret_fold[h] > regret_check[h]).count();
        let gpu_check = nh - gpu_fold;

        let river_str = solver_core::card::card_to_string(river_card).unwrap();
        let pass = gpu_fold > nh / 4 && gpu_check > nh / 4;
        println!("River {}: check={}/{}, fold={}/{} {}",
            river_str, gpu_check, nh, gpu_fold, nh, if pass { "PASS" } else { "FAIL" });

        if !pass {
            all_pass = false;
        }
    }

    assert!(all_pass, "One or more river cards failed sign agreement test");
}
