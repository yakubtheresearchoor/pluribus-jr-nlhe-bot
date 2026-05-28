#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::{ChanceGpuData, GpuContext};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;
use solver_core::card::index_to_card_pair;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

#[test]
fn turn_tree_vcfr_gpu_runs() {
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Turn,
        starting_pot: 200,
        starting_stacks: vec![400, 400],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };

    let tree = build_tree(&config).expect("tree build failed");
    let mut terminals = 0; let mut chance_n = 0; let mut player_n = 0;
    for node in &tree.nodes {
        if node.is_terminal() { terminals += 1; }
        else if node.is_chance() { chance_n += 1; }
        else { player_n += 1; }
    }
    println!("Turn tree: {} nodes (T:{} C:{} P:{}), depth={}",
        tree.num_nodes(), terminals, chance_n, player_n, tree.max_depth);

    let board_turn: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];

    let turn_table = ChanceTable::compute_turn_start(&board_turn, &ranges, 2);
    let nh = turn_table.num_valid;
    println!("Turn nh={}", nh);

    let (opp_str, opp_idx, pl_str, pl_idx, _) = turn_table.sorted_opp_arrays();
    let hand_cards = turn_table.hand_cards_gpu();
    let mut initial_weight = turn_table.initial_weight_flat();

    let chance_probs = compute_chance_probabilities(&turn_table);
    let (chance_sorted_str, chance_sorted_idx) = turn_table.chance_sorted_arrays_gpu();

    let gpu = GpuContext::new().expect("GPU init failed");
    let mut vcfr = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight,
        Some(ChanceGpuData {
            chance_sorted_strength: chance_sorted_str,
            chance_sorted_indices: chance_sorted_idx,
            chance_probabilities: chance_probs,
            remaining_deck: turn_table.remaining_deck.clone(),
        }),
    ).expect("vcfr create failed");

    let t0 = std::time::Instant::now();
    vcfr.run(25).expect("vcfr run failed");
    let elapsed = t0.elapsed().as_secs_f64();
    println!("Turn VCFR: 25 iters in {:.1}s ({:.0}ms/iter)", elapsed, elapsed / 25.0 * 1000.0);

    let cum = vcfr.download_cum_strategy().expect("download failed");
    let offsets: Vec<usize> = (0..tree.num_nodes()).map(|i| {
        let is = tree.infoset_offsets[i];
        if is == u32::MAX { usize::MAX } else { is as usize * MAX_NA * nh }
    }).collect();

    let off = offsets[0];
    let na = tree.nodes[0].num_children as usize;
    println!("Root node: offset={}, na={}", off, na);
    
    let mut sum_check = 0.0f32;
    let mut sum_bet = 0.0f32;
    for h in 0..nh.min(10) {
        let check = cum[off + 0 * nh + h];
        let bet = if na > 1 { cum[off + 1 * nh + h] } else { 0.0 };
        println!("  h={}: check={:.4} bet={:.4}", h, check, bet);
        sum_check += check;
        sum_bet += bet;
    }
    println!("First 10 hands: sum_check={:.4} sum_bet={:.4}", sum_check, sum_bet);
    
    let mut total_check = 0.0f32;
    let mut total_bet = 0.0f32;
    for h in 0..nh {
        total_check += cum[off + 0 * nh + h];
        if na > 1 { total_bet += cum[off + 1 * nh + h]; }
    }
    let avg_bet = if total_check + total_bet > 0.0 { total_bet / (total_check + total_bet) } else { 0.0 };
    println!("Turn P0 avg bet prob (25 iters): {:.4}", avg_bet);
    assert!(avg_bet > 0.001 && avg_bet < 0.999, "bet prob degenerate: {}", avg_bet);
}

fn compute_chance_probabilities(table: &ChanceTable) -> Vec<f32> {
    let nh = table.num_valid;
    let num_outcomes = table.remaining_deck.len();
    let mut probs = vec![0.0f32; num_outcomes * nh];
    for o in 0..num_outcomes {
        let card = table.remaining_deck[o];
        for h in 0..nh {
            let (c1, c2) = index_to_card_pair(table.valid_hand_indices[h] as usize);
            if card == c1 || card == c2 { continue; }
            let blocked = table.remaining_deck.iter().filter(|&&rc| rc == c1 || rc == c2).count();
            probs[o * nh + h] = 1.0 / (num_outcomes as f32 - blocked as f32);
        }
    }
    probs
}
