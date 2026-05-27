#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::{ChanceGpuData, GpuContext};
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::game::GameSpec;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::card::index_to_card_pair;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;
use postflop_solver::*;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

use postflop_solver::{
    BetSizeOptions as ExtBetSizeOptions, TreeConfig as ExtTreeConfig,
    BoardState as ExtBoardState, NOT_DEALT,
    flop_from_str, card_from_str as ext_card_from_str, solve, solve_step,
};

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

#[test]
fn sc1_production_25s() {
    let time_budget = 25.0f64;
    let full_range = "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+,J2s+,J2o+,T2s+,T2o+,92s+,92o+,82s+,82o+,72s+,72o+,62s+,62o+,52s+,52o+,42s+,42o+,32s,32o";

    println!("\n========================================");
    println!("  SC1 Production Comparison (25s budget)");
    println!("========================================\n");

    // === Config 1: T2D HU Turn (production bet sizes) ===
    println!("--- Config 1: HU Turn, pot=200, stacks=9500, bet=[0.5p,1.0p], raise=[0.5p] ---");
    
    // Our tree
    let board_turn: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    
    let turn_config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Turn,
        starting_pot: 200,
        starting_stacks: vec![9500, 9500],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5)],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    let turn_tree = build_tree(&turn_config).expect("tree build failed");
    let turn_nh = {
        let t = ChanceTable::compute_turn_start(&board_turn, &ranges, 2);
        t.num_valid
    };
    
    let mut terminals = 0; let mut chance_n = 0; let mut player_n = 0;
    for node in &turn_tree.nodes {
        if node.is_terminal() { terminals += 1; }
        else if node.is_chance() { chance_n += 1; }
        else { player_n += 1; }
    }
    println!("Our tree: {} nodes (T:{} C:{} P:{}), nh={}, depth={}", 
        turn_tree.num_nodes(), terminals, chance_n, player_n, turn_nh, turn_tree.max_depth);

    // External solver — same config
    let card_config_turn = CardConfig {
        range: [full_range.parse().unwrap(), full_range.parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(),
        turn: card_from_str("4c").unwrap(),
        river: NOT_DEALT,
    };
    let ext_bet_turn = ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap();
    let ext_turn_config = ExtTreeConfig {
        initial_state: ExtBoardState::Turn,
        starting_pot: 200,
        effective_stack: 9500,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [ext_bet_turn.clone(), ext_bet_turn.clone()],
        turn_bet_sizes: [ext_bet_turn.clone(), ext_bet_turn.clone()],
        river_bet_sizes: [ext_bet_turn.clone(), ext_bet_turn],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    let ext_turn_tree = ActionTree::new(ext_turn_config).unwrap();
    let mut ext_turn_game = PostFlopGame::with_config(card_config_turn, ext_turn_tree).unwrap();
    ext_turn_game.allocate_memory(false);
    println!("External hands: {}/{}", ext_turn_game.num_private_hands(0), ext_turn_game.num_private_hands(1));
    println!("External root actions: {:?}", ext_turn_game.available_actions());

    // Run external solver for 25s
    println!("\nExternal solver (25s budget)...");
    let ext_start = std::time::Instant::now();
    let mut ext_turn_iters = 0u32;
    loop {
        solve_step(&ext_turn_game, ext_turn_iters);
        ext_turn_iters += 1;
        if ext_start.elapsed().as_secs_f64() >= time_budget { break; }
    }
    let ext_turn_time = ext_start.elapsed().as_secs_f64();
    let ext_turn_exp = {
        let (mem, _) = ext_turn_game.memory_usage();
        println!("External memory: {:.0} MB", mem as f64 / 1e6);
        // External solver computes its own exploitability internally
        // We need to use compute_exploitability from their library
        // For now, just report iterations
        -1.0f32 // placeholder
    };
    println!("External: {} iters in {:.1}s ({:.0}ms/iter)", 
        ext_turn_iters, ext_turn_time, ext_turn_time / ext_turn_iters as f64 * 1000.0);

    // Run our VCFR for 25s
    let turn_table = ChanceTable::compute_turn_start(&board_turn, &ranges, 2);
    let turn_nh = turn_table.num_valid;
    let turn_game = TurnStartGame::new(ChanceTable::compute_turn_start(&board_turn, &ranges, 2));
    let (opp_str, opp_idx, pl_str, pl_idx, _) = turn_table.sorted_opp_arrays();
    let hand_cards = turn_table.hand_cards_gpu();
    let mut initial_weight = turn_table.initial_weight_flat();
    let chance_probs = compute_chance_probabilities(&turn_table);
    let (chance_sorted_str, chance_sorted_idx) = turn_table.chance_sorted_arrays_gpu();

    let gpu = GpuContext::new().expect("GPU init failed");

    // Calibrate
    let calib_iters = 5u32;
    let mut calib = gpu.create_vcfr_solver(
        &turn_tree, turn_nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight,
        Some(ChanceGpuData {
            chance_sorted_strength: chance_sorted_str.clone(),
            chance_sorted_indices: chance_sorted_idx.clone(),
            chance_probabilities: chance_probs.clone(),
            remaining_deck: turn_table.remaining_deck.clone(),
        }),
    ).expect("vcfr create failed");
    let t0 = std::time::Instant::now();
    calib.run(calib_iters).expect("vcfr run failed");
    let vcfr_time_per_iter = (t0.elapsed().as_secs_f64() / calib_iters as f64) as f64;
    let vcfr_target_iters = (time_budget / vcfr_time_per_iter) as u32;
    println!("\nVCFR calibration: {:.0}ms/iter → {} iters in 25s", vcfr_time_per_iter * 1000.0, vcfr_target_iters);

    let mut vcfr = gpu.create_vcfr_solver(
        &turn_tree, turn_nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight,
        Some(ChanceGpuData {
            chance_sorted_strength: chance_sorted_str,
            chance_sorted_indices: chance_sorted_idx,
            chance_probabilities: chance_probs,
            remaining_deck: turn_table.remaining_deck.clone(),
        }),
    ).expect("vcfr create failed");
    let t1 = std::time::Instant::now();
    vcfr.run(vcfr_target_iters).expect("vcfr run failed");
    let vcfr_time = t1.elapsed().as_secs_f64();
    println!("VCFR: {} iters in {:.1}s ({:.0}ms/iter)", 
        vcfr_target_iters, vcfr_time, vcfr_time / vcfr_target_iters as f64 * 1000.0);

    // === Config 2: T2D HU River (fast exploitability check) ===
    println!("\n--- Config 2: HU River, pot=200, stacks=9500, bet=[0.5p,1.0p], raise=[0.5p] ---");
    
    let board_river: Vec<Card> = ["2h", "7d", "Ks", "4c", "Qs"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    
    let river_config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::River,
        starting_pot: 200,
        starting_stacks: vec![9500, 9500],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5)],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    let river_tree = build_tree(&river_config).expect("tree build failed");
    let river_game = RiverPokerGame::new(&board_river, &ranges, 2);
    let river_nh = river_game.num_hands(0);
    let river_hand_cards = river_game.hand_cards_gpu();
    let mut river_init_weight = river_game.initial_weight(0);
    river_init_weight.extend_from_slice(&river_game.initial_weight(1));
    let (r_opp_str, r_opp_idx, r_pl_str, r_pl_idx, _) = river_game.sorted_opp_arrays();
    
    let river_offsets: Vec<usize> = (0..river_tree.num_nodes()).map(|i| {
        let is = river_tree.infoset_offsets[i];
        if is == u32::MAX { usize::MAX } else { is as usize * MAX_NA * river_nh }
    }).collect();

    let mut r_terminals = 0; let mut r_chance = 0; let mut r_player = 0;
    for node in &river_tree.nodes {
        if node.is_terminal() { r_terminals += 1; }
        else if node.is_chance() { r_chance += 1; }
        else { r_player += 1; }
    }
    println!("Our tree: {} nodes (T:{} C:{} P:{}), nh={}", 
        river_tree.num_nodes(), r_terminals, r_chance, r_player, river_nh);

    // External solver river
    let card_config_river = CardConfig {
        range: [full_range.parse().unwrap(), full_range.parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(),
        turn: card_from_str("Qs").unwrap(),
        river: card_from_str("4c").unwrap(),
    };
    let ext_bet_river = ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap();
    let ext_river_config = ExtTreeConfig {
        initial_state: ExtBoardState::River,
        starting_pot: 200,
        effective_stack: 9500,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [ext_bet_river.clone(), ext_bet_river.clone()],
        turn_bet_sizes: [ext_bet_river.clone(), ext_bet_river.clone()],
        river_bet_sizes: [ext_bet_river.clone(), ext_bet_river],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    let ext_river_tree = ActionTree::new(ext_river_config).unwrap();
    let mut ext_river_game = PostFlopGame::with_config(card_config_river, ext_river_tree).unwrap();
    ext_river_game.allocate_memory(false);
    println!("External root actions: {:?}", ext_river_game.available_actions());

    // Run external for 25s
    println!("\nExternal solver (25s budget)...");
    let ext_r_start = std::time::Instant::now();
    let mut ext_r_iters = 0u32;
    loop {
        solve_step(&ext_river_game, ext_r_iters);
        ext_r_iters += 1;
        if ext_r_start.elapsed().as_secs_f64() >= time_budget { break; }
    }
    let ext_r_time = ext_r_start.elapsed().as_secs_f64();
    // compute exploitability from external solver
    let ext_r_exp = {
        // We can't call compute_exploitability on their game easily
        // Instead, run a fresh solve() for the same iters and get exp
        let mut fresh_game = PostFlopGame::with_config(
            CardConfig {
                range: [full_range.parse().unwrap(), full_range.parse().unwrap()],
                flop: flop_from_str("2h7dKs").unwrap(),
                turn: card_from_str("Qs").unwrap(),
                river: card_from_str("4c").unwrap(),
            },
            ActionTree::new(ExtTreeConfig {
                initial_state: ExtBoardState::River,
                starting_pot: 200,
                effective_stack: 9500,
                rake_rate: 0.0, rake_cap: 0.0,
                flop_bet_sizes: [ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap(), ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap()],
                turn_bet_sizes: [ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap(), ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap()],
                river_bet_sizes: [ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap(), ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap()],
                turn_donk_sizes: None,
                river_donk_sizes: None,
                add_allin_threshold: 1.5,
                force_allin_threshold: 0.15,
                merging_threshold: 0.0,
            }).unwrap()
        ).unwrap();
        fresh_game.allocate_memory(false);
        solve(&mut fresh_game, ext_r_iters, 0.0, false)
    };
    println!("External: {} iters in {:.1}s ({:.0}ms/iter), exp={:.6}", 
        ext_r_iters, ext_r_time, ext_r_time / ext_r_iters as f64 * 1000.0, ext_r_exp);

    // Run VCFR for 25s on river tree
    let r_calib_iters = 20u32;
    let mut r_calib_solver = gpu.create_vcfr_solver(
        &river_tree, river_nh, &r_opp_str, &r_opp_idx, &r_pl_str, &r_pl_idx, 
        &river_hand_cards, &river_init_weight, None,
    ).expect("vcfr create failed");
    let tr0 = std::time::Instant::now();
    r_calib_solver.run(r_calib_iters).expect("vcfr run failed");
    let r_time_per_iter = tr0.elapsed().as_secs_f64() / r_calib_iters as f64;
    let r_target = (time_budget / r_time_per_iter) as u32;
    
    let mut r_vcfr = gpu.create_vcfr_solver(
        &river_tree, river_nh, &r_opp_str, &r_opp_idx, &r_pl_str, &r_pl_idx, 
        &river_hand_cards, &river_init_weight, None,
    ).expect("vcfr create failed");
    let tr1 = std::time::Instant::now();
    r_vcfr.run(r_target).expect("vcfr run failed");
    let r_vcfr_time = tr1.elapsed().as_secs_f64();
    let r_cum = r_vcfr.download_cum_strategy().expect("download failed");
    let r_profile = StrategyProfile::from_usize_offsets(&r_cum, &river_offsets, river_nh);
    let r_vcfr_exp = exploitability(&river_tree, &river_game, &r_profile);
    println!("VCFR:     {} iters in {:.1}s ({:.0}ms/iter), exp={:.6}", 
        r_target, r_vcfr_time, r_vcfr_time / r_target as f64 * 1000.0, r_vcfr_exp);

    println!("\n========================================");
    println!("  SUMMARY (25s budget)");
    println!("========================================");
    println!("\nConfig 1: HU Turn (219 nodes, bet=[0.5p,1.0p], raise=[0.5p])");
    println!("  External: {} iters ({:.0}ms/iter)", ext_turn_iters, ext_turn_time / ext_turn_iters as f64 * 1000.0);
    println!("  VCFR:     {} iters ({:.0}ms/iter)", vcfr_target_iters, vcfr_time / vcfr_target_iters as f64 * 1000.0);
    println!("  Throughput ratio: {:.1}x", (vcfr_target_iters as f64 / vcfr_time) / (ext_turn_iters as f64 / ext_turn_time));
    
    println!("\nConfig 2: HU River (27 nodes, bet=[0.5p,1.0p], raise=[0.5p])");
    println!("  External: {} iters, exp={:.6}", ext_r_iters, ext_r_exp);
    println!("  VCFR:     {} iters, exp={:.6}", r_target, r_vcfr_exp);
    let river_ratio = if ext_r_exp > 0.0 { r_vcfr_exp / ext_r_exp } else { 0.0 };
    println!("  Exploitability ratio: {:.2}x", river_ratio);
}
