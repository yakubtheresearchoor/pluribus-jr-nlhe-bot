#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::{ChanceGpuData, GpuContext};
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }
fn make_board() -> Vec<Card> {
    ["2h", "7d", "Ks", "4c"].iter().map(|s| card_from_str(s).unwrap()).collect()
}

fn build_turn_tree() -> (solver_core::tree::flat::FlatTree, ChanceTable, TurnStartGame) {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let game = TurnStartGame::new(ChanceTable::compute_turn_start(&board, &ranges, 2));
    let nh = table.num_valid;

    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Turn,
        starting_pot: 200,
        starting_stacks: vec![400, 400],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build failed");
    (tree, table, game)
}

fn compute_chance_probabilities(table: &ChanceTable) -> Vec<f32> {
    let nh = table.num_valid;
    let num_outcomes = table.remaining_deck.len();
    let mut probs = vec![0.0f32; num_outcomes * nh];

    for o in 0..num_outcomes {
        let card = table.remaining_deck[o];
        for h in 0..nh {
            let (c1, c2) = index_to_card_pair(table.valid_hand_indices[h] as usize);
            if card == c1 || card == c2 {
                probs[o * nh + h] = 0.0;
                continue;
            }
            let blocked = table.remaining_deck.iter()
                .filter(|&&rc| rc == c1 || rc == c2)
                .count();
            probs[o * nh + h] = 1.0 / (num_outcomes as f32 - blocked as f32);
        }
    }
    probs
}

#[test]
fn v3_multistreet_turn_start_cpu_gpu_parity() {
    let (tree, table, game) = build_turn_tree();
    let nh = table.num_valid;
    let np = 2;

    let num_chance = tree.nodes.iter().filter(|n| n.is_chance()).count();
    println!("Turn tree: {} nodes, {} chance nodes, {} infosets, nh={}",
        tree.num_nodes(), num_chance, tree.num_infosets, nh);

    let checkpoints = [100, 500];

    let mut cpu_mccfr = CpuMccfr::new(&tree, vec![nh, nh]);
    let mut cpu_iter = 0u32;

    let (opp_str, opp_idx, pl_str, pl_idx, _) = table.sorted_opp_arrays();
    let hand_cards = table.hand_cards_gpu();
    let initial_weight = table.initial_weight_flat();

    let chance_probs = compute_chance_probabilities(&table);
    let (chance_sorted_str, chance_sorted_idx) = table.chance_sorted_arrays_gpu();

    let gpu = GpuContext::new().expect("GPU init failed");
    let mut gpu_solver = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight,
        Some(ChanceGpuData {
            chance_sorted_strength: chance_sorted_str,
            chance_sorted_indices: chance_sorted_idx,
            chance_probabilities: chance_probs,
            remaining_deck: table.remaining_deck.clone(),
        }),
    ).expect("vcfr creation failed");

    println!("\n=== V3 Multi-street: Turn-start 2bet ({} nodes, {} chance, {} infosets, nh={}) ===",
        tree.num_nodes(), num_chance, tree.num_infosets, nh);
    println!(" iters   cpu_mccfr      gpu_vcfr      ratio     winner");

    for &target in &checkpoints {
        let delta = target - cpu_iter;
        cpu_mccfr.run(&tree, &game, delta);
        cpu_iter = target;
        let cpu_cum = cpu_mccfr.cum_strategy_slice().to_vec();
        let cpu_offsets: Vec<usize> = cpu_mccfr.node_offsets().to_vec();
        let cpu_profile = StrategyProfile::from_usize_offsets(&cpu_cum, &cpu_offsets, nh);
        let cpu_exp = exploitability(&tree, &game, &cpu_profile);

        gpu_solver.run(delta).expect("GPU run failed");
        let gpu_cum = gpu_solver.download_cum_strategy().expect("download failed");
        let gpu_offsets: Vec<usize> = (0..tree.num_nodes()).map(|i| {
            let infoset = tree.infoset_offsets[i];
            if infoset == u32::MAX { usize::MAX } else { infoset as usize * MAX_NA * nh }
        }).collect();
        let gpu_profile = StrategyProfile::from_usize_offsets(&gpu_cum, &gpu_offsets, nh);
        let gpu_exp = exploitability(&tree, &game, &gpu_profile);

        let ratio = gpu_exp / cpu_exp;
        let winner = if ratio < 0.95 { "GPU" } else if ratio > 1.05 { "CPU" } else { "~" };
        println!("{:>6} {:>12.4} {:>12.4} {:>10.4} {:>10}", target, cpu_exp, gpu_exp, ratio, winner);
    }

    let cpu_cum = cpu_mccfr.cum_strategy_slice().to_vec();
    let cpu_offsets: Vec<usize> = cpu_mccfr.node_offsets().to_vec();
    let cpu_profile = StrategyProfile::from_usize_offsets(&cpu_cum, &cpu_offsets, nh);
    let cpu_exp = exploitability(&tree, &game, &cpu_profile);
    let gpu_cum = gpu_solver.download_cum_strategy().expect("download failed");
    let gpu_offsets: Vec<usize> = (0..tree.num_nodes()).map(|i| {
        let infoset = tree.infoset_offsets[i];
        if infoset == u32::MAX { usize::MAX } else { infoset as usize * MAX_NA * nh }
    }).collect();
    let gpu_profile = StrategyProfile::from_usize_offsets(&gpu_cum, &gpu_offsets, nh);
    let gpu_exp = exploitability(&tree, &game, &gpu_profile);
    let ratio = gpu_exp / cpu_exp;

    assert!(ratio > 0.80 && ratio < 1.20,
        "GPU/CPU ratio {} outside 0.80-1.20 range. CPU={:.4} GPU={:.4}",
        ratio, cpu_exp, gpu_exp);
}
