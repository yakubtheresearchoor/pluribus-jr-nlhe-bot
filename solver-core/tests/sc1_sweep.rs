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
fn sc1_convergence_sweep() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "Qs"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
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

    println!("\n=== VCFR Convergence (DCFR, river, single bet 0.5p) ===");
    for iters in [10, 50, 100, 200, 500, 1000, 2000, 5000] {
        let mut v = gpu.create_vcfr_solver(
            &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
        ).expect("vcfr create failed");
        let t = std::time::Instant::now();
        v.run(iters).expect("vcfr run failed");
        let elapsed = t.elapsed().as_secs_f64();
        let c = v.download_cum_strategy().expect("download failed");
        let p = StrategyProfile::from_usize_offsets(&c, &offsets, nh);
        let e = exploitability(&tree, &game, &p);
        println!("  iters={:5} time={:.2}s exp={:.6}", iters, elapsed, e);
    }

    // External solver sweep
    let full_range = "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+,J2s+,J2o+,T2s+,T2o+,92s+,92o+,82s+,82o+,72s+,72o+,62s+,62o+,52s+,52o+,42s+,42o+,32s,32o";
    println!("\n=== External Solver Convergence ===");
    for ext_iters in [10, 50, 100, 200, 500, 1000, 2000, 5000] {
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
        let t = std::time::Instant::now();
        let e = solve(&mut ext_game, ext_iters, 0.0, false);
        let elapsed = t.elapsed().as_secs_f64();
        println!("  iters={:5} time={:.2}s exp={:.6}", ext_iters, elapsed, e);
    }
}
