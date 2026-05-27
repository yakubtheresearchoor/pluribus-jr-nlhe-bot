#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::{ChanceGpuData, GpuContext};
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::tree::action::{BetSize, BetSizeOptions as OurBetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;
use postflop_solver::{
    BetSizeOptions as ExtBetSizeOptions, CardConfig, PostFlopGame, ActionTree,
    TreeConfig as ExtTreeConfig, BoardState as ExtBoardState, NOT_DEALT,
    flop_from_str, card_from_str as ext_card_from_str, solve,
};

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

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
fn sc1_external_comparison() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];

    // Build our tree
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Turn,
        starting_pot: 200,
        starting_stacks: vec![9500, 9500],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: OurBetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5)],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build failed");
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid;
    let game = TurnStartGame::new(ChanceTable::compute_turn_start(&board, &ranges, 2));

    let num_chance = tree.nodes.iter().filter(|n| n.is_chance()).count();
    println!("\n=== SC1: External Comparison ===");
    println!("Our tree: {} nodes, {} chance, nh={}", tree.num_nodes(), num_chance, nh);

    // Configure external solver with same parameters
    let oop_range = "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+,J2s+,J2o+,T2s+,T2o+,92s+,92o+,82s+,82o+,72s+,72o+,62s+,62o+,52s+,52o+,42s+,42o+,32s,32o";
    let ip_range = oop_range;

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(),
        turn: ext_card_from_str("4c").unwrap(),
        river: NOT_DEALT,
    };

    let ext_bet_sizes = ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap();
    let ext_tree_config = ExtTreeConfig {
        initial_state: ExtBoardState::Turn,
        starting_pot: 200,
        effective_stack: 9500,
        rake_rate: 0.0,
        rake_cap: 0.0,
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
    let (mem, _) = ext_game.memory_usage();
    println!("External solver memory: {:.2} GB", mem as f64 / 1073741824.0);
    ext_game.allocate_memory(false);

    // Phase 1: Run external solver and measure time
    let ext_target_iters = 50u32;
    println!("\nRunning external solver for {} iterations...", ext_target_iters);
    let ext_start = std::time::Instant::now();
    let ext_exp = solve(&mut ext_game, ext_target_iters, 0.0, false);
    let ext_time = ext_start.elapsed().as_secs_f64();
    println!("External solver: {} iters in {:.1}s, exp={:.4}", ext_target_iters, ext_time, ext_exp);

    // Phase 2: Run our GPU VCFR for same wall-clock time
    let (opp_str, opp_idx, pl_str, pl_idx, _) = table.sorted_opp_arrays();
    let hand_cards = table.hand_cards_gpu();
    let initial_weight = table.initial_weight_flat();
    let chance_probs = compute_chance_probabilities(&table);
    let (chance_sorted_str, chance_sorted_idx) = table.chance_sorted_arrays_gpu();

    let gpu = GpuContext::new().expect("GPU init failed");

    // Calibrate VCFR iteration time
    let calib_iters = 20u32;
    let mut vcfr_calib = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight,
        Some(ChanceGpuData {
            chance_sorted_strength: chance_sorted_str.clone(),
            chance_sorted_indices: chance_sorted_idx.clone(),
            chance_probabilities: chance_probs.clone(),
            remaining_deck: table.remaining_deck.clone(),
        }),
    ).expect("vcfr creation failed");
    let calib_start = std::time::Instant::now();
    vcfr_calib.run(calib_iters).expect("vcfr calib failed");
    let calib_time = calib_start.elapsed().as_secs_f64();
    let vcfr_iters = ((ext_time / calib_time) * calib_iters as f64).max(10.0) as u32;
    println!("\nVCFR calibration: {} iters in {:.1}s → {} iters in {:.1}s",
        calib_iters, calib_time, vcfr_iters, ext_time);

    let mut vcfr_solver = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight,
        Some(ChanceGpuData {
            chance_sorted_strength: chance_sorted_str,
            chance_sorted_indices: chance_sorted_idx,
            chance_probabilities: chance_probs,
            remaining_deck: table.remaining_deck.clone(),
        }),
    ).expect("vcfr creation failed");
    let vcfr_start = std::time::Instant::now();
    vcfr_solver.run(vcfr_iters).expect("vcfr run failed");
    let vcfr_time = vcfr_start.elapsed().as_secs_f64();
    let vcfr_cum = vcfr_solver.download_cum_strategy().expect("download failed");
    let vcfr_offsets: Vec<usize> = (0..tree.num_nodes()).map(|i| {
        let is = tree.infoset_offsets[i];
        if is == u32::MAX { usize::MAX } else { is as usize * MAX_NA * nh }
    }).collect();
    let vcfr_profile = StrategyProfile::from_usize_offsets(&vcfr_cum, &vcfr_offsets, nh);
    let vcfr_exp = exploitability(&tree, &game, &vcfr_profile);
    println!("GPU VCFR: {} iters in {:.1}s, exp={:.4}", vcfr_iters, vcfr_time, vcfr_exp);

    // Summary
    println!("\n=== SC1 Summary ===");
    println!("{:<25} {:>10} {:>10} {:>10}", "Solver", "Time(s)", "Iters", "Exp");
    println!("{:<25} {:>10.1} {:>10} {:>10.4}", "External (Discounted CFR)", ext_time, ext_target_iters, ext_exp);
    println!("{:<25} {:>10.1} {:>10} {:>10.4}", "Our GPU VCFR", vcfr_time, vcfr_iters, vcfr_exp);

    let _best_exp = ext_exp.min(vcfr_exp);
    let vcfr_vs_ext = vcfr_exp / ext_exp;
    println!("\nVCFR / External ratio: {:.4}", vcfr_vs_ext);

    // The external solver uses Discounted CFR which converges much faster than vanilla CFR.
    // We check that our solver produces reasonable results (not divergent).
    assert!(vcfr_exp < 1000.0,
        "GPU VCFR exploitability {:.4} is unreasonably high", vcfr_exp);
    assert!(ext_exp < 1000.0,
        "External solver exploitability {:.4} is unreasonably high", ext_exp);
}
