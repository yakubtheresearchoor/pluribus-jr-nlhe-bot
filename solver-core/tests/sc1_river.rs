#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;
use postflop_solver::*;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

#[test]
fn sc1_river_only() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "Qs"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];

    // Our river tree
    let config = TreeConfig {
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
    let tree = build_tree(&config).expect("tree build failed");
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_hands(0);
    let hand_cards = game.hand_cards_gpu();
    let mut initial_weight = game.initial_weight(0);
    initial_weight.extend_from_slice(&game.initial_weight(1));
    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();

    println!("\n=== SC1 River-Only ===");
    println!("Our tree: {} nodes, nh={}", tree.num_nodes(), nh);

    // External solver - same board, river only
    let full_range = "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+,J2s+,J2o+,T2s+,T2o+,92s+,92o+,82s+,82o+,72s+,72o+,62s+,62o+,52s+,52o+,42s+,42o+,32s,32o";

    let card_config = CardConfig {
        range: [full_range.parse().unwrap(), full_range.parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(),
        turn: card_from_str("Qs").unwrap(),
        river: card_from_str("4c").unwrap(),
    };

    let ext_bet_sizes = ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap();
    let ext_tree_config = ExtTreeConfig {
        initial_state: ExtBoardState::River,
        starting_pot: 200,
        effective_stack: 9500,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [ext_bet_sizes.clone(), ext_bet_sizes.clone()],
        turn_bet_sizes: [ext_bet_sizes.clone(), ext_bet_sizes.clone()],
        river_bet_sizes: [ext_bet_sizes.clone(), ext_bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };

    let action_tree = ActionTree::new(ext_tree_config).unwrap();
    let mut ext_game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    ext_game.allocate_memory(false);

    let gpu = GpuContext::new().expect("GPU init failed");
    let offsets: Vec<usize> = (0..tree.num_nodes()).map(|i| {
        let is = tree.infoset_offsets[i];
        if is == u32::MAX { usize::MAX } else { is as usize * MAX_NA * nh }
    }).collect();

    // Run VCFR at fixed iteration counts to see convergence
    println!("\nGPU VCFR convergence:");
    for iters in [100, 500, 1000, 5000, 10000] {
        let mut v = gpu.create_vcfr_solver(
            &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
        ).expect("vcfr create failed");
        let t = std::time::Instant::now();
        v.run(iters).expect("vcfr run failed");
        let elapsed = t.elapsed().as_secs_f64();
        let c = v.download_cum_strategy().expect("download failed");
        let p = StrategyProfile::from_usize_offsets(&c, &offsets, nh);
        let e = exploitability(&tree, &game, &p);
        println!("  iters={:6} time={:.2}s exp={:.4}", iters, elapsed, e);
    }

    // Run external solver for 500 iterations
    let ext_iters = 500u32;
    println!("\nRunning external solver for {} iterations...", ext_iters);
    let ext_start = std::time::Instant::now();
    let ext_exp = solve(&mut ext_game, ext_iters, 0.0, false);
    let ext_time = ext_start.elapsed().as_secs_f64();
    println!("External: {} iters in {:.2}s, exp={:.4}", ext_iters, ext_time, ext_exp);

    // Matched-time VCFR
    // VCFR converges ~10x slower than Discounted CFR per iteration. At matched time
    // we expect VCFR to be within 10x of external solver's exploitability.
    // Run VCFR for enough iterations to match external time.
    let calib_iters = 100u32;
    let mut calib = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
    ).expect("vcfr create failed");
    let t0 = std::time::Instant::now();
    calib.run(calib_iters).expect("vcfr run failed");
    let calib_time = t0.elapsed().as_secs_f64();
    let vcfr_target = ((ext_time / calib_time) * calib_iters as f64).max(100.0) as u32;

    let mut vcfr = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
    ).expect("vcfr create failed");
    let t1 = std::time::Instant::now();
    vcfr.run(vcfr_target).expect("vcfr run failed");
    let vcfr_time = t1.elapsed().as_secs_f64();
    let cum = vcfr.download_cum_strategy().expect("download failed");
    let profile = StrategyProfile::from_usize_offsets(&cum, &offsets, nh);
    let vcfr_exp = exploitability(&tree, &game, &profile);

    println!("\n=== Summary ===");
    println!("External (Discounted CFR): {:.2}s, {} iters, exp={:.4}", ext_time, ext_iters, ext_exp);
    println!("GPU VCFR (matched time):   {:.2}s, {} iters, exp={:.4}", vcfr_time, vcfr_target, vcfr_exp);
    println!("VCFR/External ratio: {:.4}", vcfr_exp / ext_exp);

    // VCFR should converge at matched time (exp < 100 at ~900 iters)
    assert!(vcfr_exp < 100.0,
        "VCFR exp {:.4} not converging at matched time", vcfr_exp);
}

#[test]
fn sc1_river_convergence() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "Qs"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];

    let config = TreeConfig {
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
    let tree = build_tree(&config).expect("tree build failed");
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

    println!("\n=== SC1 Convergence Test (River, 27 nodes) ===");
    let mut prev_exp = f32::MAX;
    let mut vcfr = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
    ).expect("vcfr create failed");
    for iters in [100, 500, 1000, 5000, 10000] {
        let start = std::time::Instant::now();
        vcfr.run(iters).expect("vcfr run failed");
        let elapsed = start.elapsed().as_secs_f64();
        let c = vcfr.download_cum_strategy().expect("download failed");
        let p = StrategyProfile::from_usize_offsets(&c, &offsets, nh);
        let exp = exploitability(&tree, &game, &p);
        println!("  iters={:6} time={:.1}s exp={:.4}", iters, elapsed, exp);
        assert!(exp < prev_exp * 1.05,
            "Exploitability increased at {} iters: {:.4} > {:.4}", iters, exp, prev_exp);
        prev_exp = exp;
    }
    assert!(prev_exp < 1.0,
        "VCFR should converge below 1.0 (pot/200) at 10000 iters, got {:.4}", prev_exp);
    println!("Convergence PASS: exp={:.4} at 10000 iters", prev_exp);
}

use postflop_solver::{
    BetSizeOptions as ExtBetSizeOptions, TreeConfig as ExtTreeConfig,
    BoardState as ExtBoardState, NOT_DEALT,
    flop_from_str, card_from_str as ext_card_from_str, solve,
};
