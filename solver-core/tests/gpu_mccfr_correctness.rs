#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::chance_table::ChanceTable;
use solver_core::tree::action::BoardState;
use solver_core::tree::flat::{FlatNode, FlatTree};

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

fn build_big_bet_tree() -> FlatTree {
    let mut tree = FlatTree::new(2, 200, vec![200, 200], 0.0, 0.0);

    let n_root = tree.alloc_node(FlatNode::player(0, BoardState::Turn, 0));
    tree.set_contribution(n_root, 0, 5);
    tree.set_contribution(n_root, 1, 100);

    let n_chance = tree.alloc_node(FlatNode::chance(BoardState::River));
    tree.set_contribution(n_chance, 0, 100);
    tree.set_contribution(n_chance, 1, 100);

    let n_showdown = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n_showdown, 0, 100);
    tree.set_contribution(n_showdown, 1, 100);

    let n_fold = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n_fold, 0, 5);
    tree.set_contribution(n_fold, 1, 100);

    tree.set_children(n_root, vec![n_chance as u32, n_fold as u32]);
    tree.set_children(n_chance, vec![n_showdown as u32]);

    tree.set_folded_mask(n_fold, 0b01);

    tree
}

#[test]
fn gpu_mccfr_fold_correctness() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let test_river = card_from_str("9s").unwrap();
    let remaining_deck = vec![test_river];

    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();

    let tree = build_big_bet_tree();

    let (s_opp_str, s_opp_idx, s_pl_str, s_pl_idx, _) = table.sorted_opp_arrays();
    let (chance_sorted_str, chance_sorted_idx) = table.chance_sorted_arrays_gpu();
    let hand_cards = table.hand_cards_gpu();
    let initial_weight = table.initial_weight_flat();

    // Verify standalone showdown with contribution=95
    let ch_stride = (table.num_players as usize - 1) * nh;
    let river_off = test_river as usize * ch_stride;
    let chance_str = &chance_sorted_str[river_off..river_off + ch_stride];
    let chance_idx = &chance_sorted_idx[river_off..river_off + ch_stride];
    let opp_reach = vec![1.0f32; nh];

    let gpu_showdown = gpu.run_test_showdown(
        chance_str, chance_idx, chance_str, chance_idx,
        &hand_cards, &opp_reach, nh, 95.0,
    ).expect("standalone showdown failed");

    let showdown_pos = gpu_showdown.iter().filter(|&&v| v > 0.0).count();
    let showdown_neg = gpu_showdown.iter().filter(|&&v| v < 0.0).count();
    let showdown_min = gpu_showdown.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("Standlone showdown (contribution=95): pos={}, neg={}, min={:.0}", 
        showdown_pos, showdown_neg, showdown_min);

    // Fold: player 0 has contrib 5, opp has 100 → payoff = -5
    // With ~1000 unblocked opponents: fold_cfv ≈ -5 * 1000 = -5000
    // Showdown: contribution=95, min cfv ≈ -95 * 1000 = -95000
    // Weak hands: fold (-5000) >> showdown (-95000) → should fold
    println!("Fold CFV ≈ -5 * ~1000 = -5000 (per hand, with card blocking)");

    // Run GPU MCCFR with meaningful fold
    let mut gpu_solver = gpu.create_nplayer_solver(
        &tree, nh,
        &table.hand_ranks_gpu(),
        &s_opp_str, &s_opp_idx,
        &s_pl_str, &s_pl_idx,
        &vec![u16::MAX; nh],
        &hand_cards,
        &initial_weight,
        Some(&table.chance_ranks_gpu()),
        &remaining_deck,
        Some(&chance_sorted_str),
        Some(&chance_sorted_idx),
    ).expect("solver creation failed");

    gpu_solver.run(32, 200).expect("GPU run failed");

    let gpu_regrets = gpu_solver.download_regrets().expect("download failed");
    let gpu_check = (0..nh).filter(|&h| gpu_regrets[0 * nh + h] > gpu_regrets[1 * nh + h]).count();
    let gpu_fold = nh - gpu_check;

    // Print specific hands
    let as_kh_idx = solver_core::card::card_pair_to_index(card_from_str("As").unwrap(), card_from_str("Kh").unwrap());
    let qs_jd_idx = solver_core::card::card_pair_to_index(card_from_str("Qs").unwrap(), card_from_str("Jd").unwrap());
    let tc_3c_idx = solver_core::card::card_pair_to_index(card_from_str("Tc").unwrap(), card_from_str("3c").unwrap());

    for &(label, idx) in &[("AsKh", as_kh_idx), ("QsJd", qs_jd_idx), ("Tc3c", tc_3c_idx)] {
        if let Some(hi) = table.valid_hand_indices.iter().position(|&vi| vi as usize == idx) {
            let r_check = gpu_regrets[0 * nh + hi];
            let r_fold = gpu_regrets[1 * nh + hi];
            let pref = if r_check > r_fold { "check" } else { "fold" };
            println!("{}: r_check={:.0}, r_fold={:.0} → prefer {}", label, r_check, r_fold, pref);
        }
    }

    println!("GPU MCCFR: check={}/{} ({:.1}%), fold={}/{} ({:.1}%)",
        gpu_check, nh, 100.0 * gpu_check as f32 / nh as f32,
        gpu_fold, nh, 100.0 * gpu_fold as f32 / nh as f32);

    assert!(gpu_fold > nh / 4,
        "Expected >25% hands to prefer fold, got {}/{}", gpu_fold, nh);
    assert!(gpu_check > nh / 4,
        "Expected >25% hands to prefer check, got {}/{}", gpu_check, nh);
}
