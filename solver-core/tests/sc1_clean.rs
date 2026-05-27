#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::game::GameSpec;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;
use postflop_solver::*;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

use postflop_solver::{
    BetSizeOptions as ExtBetSizeOptions, TreeConfig as ExtTreeConfig,
    BoardState as ExtBoardState, NOT_DEALT,
    flop_from_str, card_from_str as ext_card_from_str, solve,
};

#[test]
fn sc1_clean_river() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "Qs"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();

    // Single bet size only — both solvers should produce identical trees
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::River,
        starting_pot: 200,
        starting_stacks: vec![9500, 9500],
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
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_hands(0);
    let hand_cards = game.hand_cards_gpu();
    let mut initial_weight = game.initial_weight(0);
    initial_weight.extend_from_slice(&game.initial_weight(1));
    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();

    let gpu = GpuContext::new().expect("GPU init failed");
    let offsets: Vec<usize> = (0..tree.num_nodes()).map(|i| {
        let is = tree.infoset_offsets[i];
        if is == u32::MAX { usize::MAX } else { is as usize * MAX_NA * nh }
    }).collect();

    // Print our tree structure
    let mut terminals = 0; let mut chance_n = 0; let mut player_n = 0;
    for node in &tree.nodes {
        if node.is_terminal() { terminals += 1; }
        else if node.is_chance() { chance_n += 1; }
        else { player_n += 1; }
    }
    let root = &tree.nodes[0];
    let root_children = tree.node_children(0);
    println!("\n=== Our Tree ===");
    println!("Nodes: {} (T:{} C:{} P:{}), Infosets: {}, nh: {}, Max depth: {}",
        tree.num_nodes(), terminals, chance_n, player_n, tree.num_infosets, nh, tree.max_depth);
    println!("Root na={}, children: {:?}", root.num_children, root_children);

    // External solver — single bet size "50%"
    let full_range = "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+,J2s+,J2o+,T2s+,T2o+,92s+,92o+,82s+,82o+,72s+,72o+,62s+,62o+,52s+,52o+,42s+,42o+,32s,32o";

    let card_config = CardConfig {
        range: [full_range.parse().unwrap(), full_range.parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(),
        turn: card_from_str("Qs").unwrap(),
        river: card_from_str("4c").unwrap(),
    };

    let ext_bet = ExtBetSizeOptions::try_from(("50%", "50%")).unwrap();
    let ext_tree_config = ExtTreeConfig {
        initial_state: ExtBoardState::River,
        starting_pot: 200,
        effective_stack: 9500,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [ext_bet.clone(), ext_bet.clone()],
        turn_bet_sizes: [ext_bet.clone(), ext_bet.clone()],
        river_bet_sizes: [ext_bet.clone(), ext_bet],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };

    let action_tree = ActionTree::new(ext_tree_config).unwrap();
    let mut ext_game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    ext_game.allocate_memory(false);

    println!("\n=== External Solver ===");
    println!("Hands: {}/{}", ext_game.num_private_hands(0), ext_game.num_private_hands(1));
    println!("Root actions: {:?}", ext_game.available_actions());

    // Calibrate our solver time per iteration
    let calib_iters = 50u32;
    let mut calib = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
    ).expect("vcfr create failed");
    let t0 = std::time::Instant::now();
    calib.run(calib_iters).expect("vcfr run failed");
    let calib_time = t0.elapsed().as_secs_f64();
    let time_per_iter = calib_time / calib_iters as f64;
    println!("\nVCFR: {:.1}ms/iter", time_per_iter * 1000.0);

    // Phase 1: Run external solver for fixed iterations, measure time
    let ext_iters = 500u32;
    println!("\nRunning external solver for {} iterations...", ext_iters);
    let ext_start = std::time::Instant::now();
    let ext_exp = solve(&mut ext_game, ext_iters, 0.0, false);
    let ext_time = ext_start.elapsed().as_secs_f64();
    let ext_time_per_iter = ext_time / ext_iters as f64;
    println!("External: {} iters in {:.2}s ({:.1}ms/iter), exp={:.6}",
        ext_iters, ext_time, ext_time_per_iter * 1000.0, ext_exp);

    // Phase 2: Run VCFR for exactly the SAME iteration count
    let vcfr_iters = ext_iters;
    let mut vcfr = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
    ).expect("vcfr create failed");
    let t1 = std::time::Instant::now();
    vcfr.run(vcfr_iters).expect("vcfr run failed");
    let vcfr_time = t1.elapsed().as_secs_f64();
    let cum = vcfr.download_cum_strategy().expect("download failed");
    let profile = StrategyProfile::from_usize_offsets(&cum, &offsets, nh);
    let vcfr_exp = exploitability(&tree, &game, &profile);
    println!("VCFR:     {} iters in {:.2}s ({:.1}ms/iter), exp={:.6}",
        vcfr_iters, vcfr_time, (vcfr_time / vcfr_iters as f64) * 1000.0, vcfr_exp);

    // Phase 3: Run VCFR for matched wall-clock time
    let vcfr_matched_iters = (ext_time / time_per_iter) as u32;
    let mut vcfr2 = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
    ).expect("vcfr create failed");
    let t2 = std::time::Instant::now();
    vcfr2.run(vcfr_matched_iters).expect("vcfr run failed");
    let vcfr2_time = t2.elapsed().as_secs_f64();
    let cum2 = vcfr2.download_cum_strategy().expect("download failed");
    let profile2 = StrategyProfile::from_usize_offsets(&cum2, &offsets, nh);
    let vcfr2_exp = exploitability(&tree, &game, &profile2);
    println!("VCFR:     {} iters in {:.2}s (matched time), exp={:.6}",
        vcfr_matched_iters, vcfr2_time, vcfr2_exp);

    println!("\n=== SC1 Clean Comparison ===");
    println!("{:<30} {:>8} {:>8} {:>12}", "Solver", "Iters", "Time(s)", "Exploitability");
    println!("{:<30} {:>8} {:>8.2} {:>12.6}", "External (DCFR)", ext_iters, ext_time, ext_exp);
    println!("{:<30} {:>8} {:>8.2} {:>12.6}", "VCFR (matched iters)", vcfr_iters, vcfr_time, vcfr_exp);
    println!("{:<30} {:>8} {:>8.2} {:>12.6}", "VCFR (matched time)", vcfr_matched_iters, vcfr2_time, vcfr2_exp);
    
    let ratio_iters = vcfr_exp / ext_exp;
    let ratio_time = vcfr2_exp / ext_exp;
    println!("\nVCFR/External (matched iters): {:.4}x", ratio_iters);
    println!("VCFR/External (matched time):  {:.4}x", ratio_time);
    
    if ratio_time <= 1.05 {
        println!("\nSC1: PASS (within 5% at matched time)");
    } else if ratio_time <= 2.0 {
        println!("\nSC1: CLOSE ({:.1}x at matched time — investigate tree differences)", ratio_time);
    } else {
        println!("\nSC1: FAIL ({:.1}x gap at matched time)", ratio_time);
    }
}
