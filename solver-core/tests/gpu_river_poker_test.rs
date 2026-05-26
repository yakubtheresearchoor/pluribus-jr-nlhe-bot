#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::game::GameSpec;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatNode, FlatTree};

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

fn make_board() -> Vec<Card> {
    ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect()
}

fn build_river_tree() -> FlatTree {
    let mut tree = FlatTree::new(2, 10, vec![95, 95], 0.0, 0.0);

    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n0, 0, 5);
    tree.set_contribution(n0, 1, 5);

    let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n1, 0, 5);
    tree.set_contribution(n1, 1, 5);

    let n2 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n2, 0, 10);
    tree.set_contribution(n2, 1, 5);

    let n3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n3, 0, 5);
    tree.set_contribution(n3, 1, 5);

    let n4 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n4, 0, 5);
    tree.set_contribution(n4, 1, 10);

    let n5 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n5, 0, 10);
    tree.set_contribution(n5, 1, 5);

    let n6 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n6, 0, 10);
    tree.set_contribution(n6, 1, 10);

    let n7 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n7, 0, 5);
    tree.set_contribution(n7, 1, 10);

    let n8 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n8, 0, 10);
    tree.set_contribution(n8, 1, 10);

    tree.set_children(n0, vec![1, 2]);
    tree.set_children(n1, vec![3, 4]);
    tree.set_children(n2, vec![5, 6]);
    tree.set_children(n4, vec![7, 8]);

    // n5: P1 folded (contrib [10,5], P1 didn't match)
    tree.set_folded_mask(n5, 0b10);
    // n7: P0 folded (contrib [5,10], P0 didn't match)
    tree.set_folded_mask(n7, 0b01);

    tree
}

#[test]
fn gpu_river_poker_convergence() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    let unique_ranks: std::collections::HashSet<u16> = game.hand_ranks_gpu().iter().copied().collect();
    println!("Valid hands: {}, unique ranks: {}", nh, unique_ranks.len());

    let tree = build_river_tree();

    let hand_ranks = game.hand_ranks_gpu();
    let (sorted_opp_strength, sorted_opp_indices, sorted_player_strength, sorted_player_indices, same_hand_idx) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let mut solver = gpu
        .create_nplayer_solver(
            &tree,
            nh,
            &hand_ranks,
            &sorted_opp_strength,
            &sorted_opp_indices,
            &sorted_player_strength,
            &sorted_player_indices,
            &same_hand_idx,
            &hand_cards,
            &initial_weight,
            None,
            &[],
            None,
            None,
        )
        .expect("nplayer solver creation failed");

    solver.run(32, 100).expect("GPU run failed");

    let avg_strat = solver
        .get_average_strategy_at(0, 2)
        .expect("download failed");

    let bet_probs: Vec<f32> = avg_strat[1].clone();
    let avg_bet: f32 = bet_probs.iter().sum::<f32>() / nh as f32;

    println!("GPU poker: {} valid hands, avg bet prob: {:.4}", nh, avg_bet);

    let mut ranked: Vec<(u16, f32)> = (0..nh)
        .map(|i| (hand_ranks[i], bet_probs[i]))
        .collect();
    ranked.sort_by_key(|&(r, _)| std::cmp::Reverse(r));

    let top_q = nh / 4;
    let bot_q = nh / 4;
    let top_bet: f32 = ranked[..top_q].iter().map(|(_, b)| *b).sum::<f32>() / top_q as f32;
    let bot_bet: f32 = ranked[nh - bot_q..].iter().map(|(_, b)| *b).sum::<f32>() / bot_q as f32;

    println!("Top 25% bet: {:.4}, Bottom 25% bet: {:.4}", top_bet, bot_bet);

    assert!(avg_bet > 0.01 && avg_bet < 0.99, "avg bet {} seems degenerate", avg_bet);
    assert!(top_bet > bot_bet + 0.01, "strong hands should bet more: top={:.4} bot={:.4}", top_bet, bot_bet);
}

#[test]
fn gpu_vs_cpu_single_iter_regrets() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let hand_ranks = game.hand_ranks_gpu();
    let (s_opp_str, s_opp_idx, s_pl_str, s_pl_idx, same_hand_idx) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    // CPU: 1 iteration (both traversers)
    let mut cpu = CpuMccfr::new(&tree, vec![nh, nh]);
    cpu.run(&tree, &game, 1);
    let cpu_reg = cpu.regrets_slice().to_vec();
    let cpu_cum = cpu.cum_strategy_slice().to_vec();

    // GPU: 1 iteration, batch=2 (expect ~1 thread per traverser)
    let gpu = GpuContext::new().expect("GPU init failed");
    let mut gpu_solver = gpu.create_nplayer_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    gpu_solver.run(2, 1).expect("GPU run failed");
    let gpu_reg = gpu_solver.download_regrets().expect("download failed");
    let gpu_cum = gpu_solver.download_cum_strategy().expect("download failed");

    // Compare regret magnitudes
    let cpu_reg_max = cpu_reg.iter().cloned().map(|v| v.abs()).fold(0.0f32, f32::max);
    let gpu_reg_max = gpu_reg.iter().cloned().map(|v| v.abs()).fold(0.0f32, f32::max);
    let cpu_cum_max = cpu_cum.iter().cloned().map(|v| v.abs()).fold(0.0f32, f32::max);
    let gpu_cum_max = gpu_cum.iter().cloned().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("After 1 iteration (GPU batch=2):");
    println!("  CPU regrets: max_abs={:.1}", cpu_reg_max);
    println!("  GPU regrets: max_abs={:.1}", gpu_reg_max);
    println!("  ratio: {:.1}x", gpu_reg_max / cpu_reg_max.max(0.001));
    println!("  CPU cum_strat: max_abs={:.1}", cpu_cum_max);
    println!("  GPU cum_strat: max_abs={:.1}", gpu_cum_max);

    // Check regrets at node 0 (P0, actions: check=0, bet=1)
    let cpu_off0 = cpu.node_offsets()[0];
    let gpu_off0 = gpu_solver.node_offsets()[0] as usize;
    println!("  Node 0 offsets: CPU={}, GPU={}", cpu_off0, gpu_off0);
    
    // Check regret sign agreement at node 0
    let mut sign_agree = 0usize;
    let mut sign_disagree = 0usize;
    for a in 0..2 {
        for h in 0..nh.min(20) {
            let cpu_v = cpu_reg[cpu_off0 + a * nh + h];
            let gpu_v = gpu_reg[gpu_off0 + a * nh + h];
            if (cpu_v >= 0.0) == (gpu_v >= 0.0) {
                sign_agree += 1;
            } else {
                sign_disagree += 1;
            }
        }
    }
    println!("  Node 0 regret sign: agree={}, disagree={}", sign_agree, sign_disagree);

    // Check cum_strategy at node 0
    let cpu_cum_check: f32 = (0..nh).map(|h| cpu_cum[cpu_off0 + 0 * nh + h]).sum();
    let cpu_cum_bet: f32 = (0..nh).map(|h| cpu_cum[cpu_off0 + 1 * nh + h]).sum();
    let gpu_cum_check: f32 = (0..nh).map(|h| gpu_cum[gpu_off0 + 0 * nh + h]).sum();
    let gpu_cum_bet: f32 = (0..nh).map(|h| gpu_cum[gpu_off0 + 1 * nh + h]).sum();
    println!("  Node 0 cum_strategy action 0 (check): CPU={:.1} GPU={:.1}", cpu_cum_check, gpu_cum_check);
    println!("  Node 0 cum_strategy action 1 (bet):   CPU={:.1} GPU={:.1}", cpu_cum_bet, gpu_cum_bet);

    // Check ratio between actions (should be similar direction)
    let cpu_ratio = if cpu_cum_check > 0.0 { cpu_cum_bet / cpu_cum_check } else { 0.0 };
    let gpu_ratio = if gpu_cum_check > 0.0 { gpu_cum_bet / gpu_cum_check } else { 0.0 };
    println!("  Node 0 bet/check ratio: CPU={:.4} GPU={:.4}", cpu_ratio, gpu_ratio);

    // GPU vanilla 10k iterations batch=1 — convergence check
    let mut gpu10k = gpu.create_nplayer_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    gpu10k.run(1, 10000).expect("GPU run failed");
    let gpu10k_cum = gpu10k.download_cum_strategy().expect("download failed");
    let gpu10k_reg = gpu10k.download_regrets().expect("download failed");
    let gpu10k_profile = StrategyProfile::from_u32_offsets(&gpu10k_cum, gpu10k.node_offsets(), nh);
    let gpu10k_exp = exploitability(&tree, &game, &gpu10k_profile);
    println!("  GPU vanilla b=1 10k iters: exp={:.4}", gpu10k_exp);

    // Print strategy at node 0 for GPU 10k
    let gpu10k_off0 = gpu10k.node_offsets()[0] as usize;
    for h in 0..5.min(nh) {
        let c0 = gpu10k_cum[gpu10k_off0 + 0 * nh + h];
        let c1 = gpu10k_cum[gpu10k_off0 + 1 * nh + h];
        let r0 = gpu10k_reg[gpu10k_off0 + 0 * nh + h];
        let r1 = gpu10k_reg[gpu10k_off0 + 1 * nh + h];
        let total = c0 + c1;
        let s0 = if total > 0.0 { c0 / total } else { 0.5 };
        let s1 = if total > 0.0 { c1 / total } else { 0.5 };
        println!("  GPU10k node0 hand[{}]: cum=[{:.1},{:.1}] strat=[{:.3},{:.3}] reg=[{:.0},{:.0}]", 
                 h, c0, c1, s0, s1, r0, r1);
    }

    // Compare with CPU 10k strategy at node 0
    let mut cpu10k = CpuMccfr::new(&tree, vec![nh, nh]);
    cpu10k.run(&tree, &game, 10000);
    let cpu10k_cum = cpu10k.cum_strategy_slice();
    let cpu10k_reg = cpu10k.regrets_slice();
    let cpu10k_off0 = cpu10k.node_offsets()[0];
    for h in 0..5.min(nh) {
        let c0 = cpu10k_cum[cpu10k_off0 + 0 * nh + h];
        let c1 = cpu10k_cum[cpu10k_off0 + 1 * nh + h];
        let r0 = cpu10k_reg[cpu10k_off0 + 0 * nh + h];
        let r1 = cpu10k_reg[cpu10k_off0 + 1 * nh + h];
        let total = c0 + c1;
        let s0 = if total > 0.0 { c0 / total } else { 0.5 };
        let s1 = if total > 0.0 { c1 / total } else { 0.5 };
        println!("  CPU10k node0 hand[{}]: cum=[{:.1},{:.1}] strat=[{:.3},{:.3}] reg=[{:.0},{:.0}]",
                 h, c0, c1, s0, s1, r0, r1);
    }
    let cpu10k_profile = StrategyProfile::from_usize_offsets(cpu10k.cum_strategy_slice(), cpu10k.node_offsets(), nh);
    let cpu10k_exp = exploitability(&tree, &game, &cpu10k_profile);
    println!("  CPU 10k iters: exp={:.4}", cpu10k_exp);

    // Traversal-matched comparison: CPU 1000 iters = 2000 traversals (2 per iter)
    // GPU vanilla batch=1 2000 iters = 2000 traversals (1 per iter, random traverser)
    // GPU vanilla batch=32 63 iters ≈ 2016 traversals (32 per iter)
    let mut gpu_tm = gpu.create_nplayer_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    gpu_tm.run(1, 2000).expect("GPU run failed");
    let gpu_tm_cum = gpu_tm.download_cum_strategy().expect("download failed");
    let gpu_tm_profile = StrategyProfile::from_u32_offsets(&gpu_tm_cum, gpu_tm.node_offsets(), nh);
    let gpu_tm_exp = exploitability(&tree, &game, &gpu_tm_profile);

    let mut gpu_tm32 = gpu.create_nplayer_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    gpu_tm32.run(32, 63).expect("GPU run failed");
    let gpu_tm32_cum = gpu_tm32.download_cum_strategy().expect("download failed");
    let gpu_tm32_profile = StrategyProfile::from_u32_offsets(&gpu_tm32_cum, gpu_tm32.node_offsets(), nh);
    let gpu_tm32_exp = exploitability(&tree, &game, &gpu_tm32_profile);

    let cpu1k = CpuMccfr::new(&tree, vec![nh, nh]);
    drop(cpu1k);
    let mut cpu1k = CpuMccfr::new(&tree, vec![nh, nh]);
    cpu1k.run(&tree, &game, 1000);
    let cpu1k_profile = StrategyProfile::from_usize_offsets(cpu1k.cum_strategy_slice(), cpu1k.node_offsets(), nh);
    let cpu1k_exp = exploitability(&tree, &game, &cpu1k_profile);

    println!("  Traversal-matched (~2000 total):");
    println!("    CPU 1000 iters (2000 trav): exp={:.4}", cpu1k_exp);
    println!("    GPU vanilla b=1 2000 iters (2000 trav): exp={:.4}", gpu_tm_exp);
    println!("    GPU vanilla b=32 63 iters (2016 trav): exp={:.4}", gpu_tm32_exp);
}

#[test]
fn gpu_tree_builder_river_pipeline() {
    let gpu = GpuContext::new().expect("GPU init failed");
    let board = make_board();

    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::River,
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
    let num_chance = tree.nodes.iter().filter(|n| n.is_chance()).count();
    assert_eq!(num_chance, 0, "river tree should have no chance nodes");

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    let hand_ranks = game.hand_ranks_gpu();
    let (sorted_opp_strength, sorted_opp_indices, sorted_player_strength, sorted_player_indices, same_hand_idx) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let mut solver = gpu
        .create_nplayer_solver(
            &tree,
            nh,
            &hand_ranks,
            &sorted_opp_strength,
            &sorted_opp_indices,
            &sorted_player_strength,
            &sorted_player_indices,
            &same_hand_idx,
            &hand_cards,
            &initial_weight,
            None,
            &[],
            None,
            None,
        )
        .expect("solver creation failed");

    solver.run(32, 10).expect("GPU run failed");

    let regrets = solver.download_regrets().expect("download failed");
    let nonzero = regrets.iter().filter(|&&r| r != 0.0).count();
    println!(
        "Pipeline OK: {} nodes, {} valid hands, {}/{} non-zero regrets",
        tree.num_nodes(), nh, nonzero, regrets.len()
    );
    assert!(nonzero > 0, "should have non-zero regrets after running");
}

#[test]
fn river_exploitability_cpu_convergence() {
    use solver_core::solver::mccfr::CpuMccfr;
    use solver_core::solver::game::GameSpec;
    use solver_core::solver::best_response::{StrategyProfile, exploitability};

    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    let tree = build_river_tree();
    let mut solver = CpuMccfr::new(&tree, vec![nh, nh]);

    let mut exp_prev = f32::MAX;
    let checkpoints: Vec<(u32, u32)> = vec![(100, 100), (500, 400), (1000, 500)];

    for &(total_iters, batch_iters) in &checkpoints {
        solver.run(&tree, &game, batch_iters);

        let profile = StrategyProfile::from_usize_offsets(
            solver.cum_strategy_slice(),
            solver.node_offsets(),
            nh,
        );
        let exp = exploitability(&tree, &game, &profile);
        println!("River exploitability at {} iters: {:.4}", total_iters, exp);

        if total_iters > 100 {
            assert!(exp < exp_prev,
                "Exploitability should decrease: {} iters ({:.4}) < prev ({:.4})", total_iters, exp, exp_prev);
        }
        exp_prev = exp;
    }

    assert!(exp_prev < 5.0,
        "After 1000 iters, exploitability should be reasonable, got {:.4}", exp_prev);

    println!("river_exploitability_cpu_convergence PASSED");
}

#[test]
fn sc1_hu_river_cpu_vs_gpu_exploitability() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let hand_ranks = game.hand_ranks_gpu();
    let (s_opp_str, s_opp_idx, s_pl_str, s_pl_idx, same_hand_idx) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    // CPU vanilla CFR — 1000 iterations
    let mut cpu_solver = CpuMccfr::new(&tree, vec![nh, nh]);
    let cpu_start = std::time::Instant::now();
    cpu_solver.run(&tree, &game, 1000);
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_profile = StrategyProfile::from_usize_offsets(
        cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh,
    );
    let cpu_exp = exploitability(&tree, &game, &cpu_profile);

    // GPU vanilla CFR (not extsamp) — batch=1, 1000 iterations (sequential, no races)
    let gpu = GpuContext::new().expect("GPU init failed");
    let mut gpu_v1 = gpu.create_nplayer_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    let gpu_v1_start = std::time::Instant::now();
    gpu_v1.run(1, 1000).expect("GPU run failed");
    let gpu_v1_elapsed = gpu_v1_start.elapsed();
    let gpu_v1_cum = gpu_v1.download_cum_strategy().expect("download failed");
    let gpu_v1_offsets = gpu_v1.node_offsets().to_vec();
    let gpu_v1_profile = StrategyProfile::from_u32_offsets(&gpu_v1_cum, &gpu_v1_offsets, nh);
    let gpu_v1_exp = exploitability(&tree, &game, &gpu_v1_profile);

    // GPU vanilla CFR — batch=32, 1000 iterations
    let mut gpu_v32 = gpu.create_nplayer_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    let gpu_v32_start = std::time::Instant::now();
    gpu_v32.run(32, 1000).expect("GPU run failed");
    let gpu_v32_elapsed = gpu_v32_start.elapsed();
    let gpu_v32_cum = gpu_v32.download_cum_strategy().expect("download failed");
    let gpu_v32_profile = StrategyProfile::from_u32_offsets(&gpu_v32_cum, gpu_v32.node_offsets(), nh);
    let gpu_v32_exp = exploitability(&tree, &game, &gpu_v32_profile);

    // GPU extsamp compact — 1000 iterations, batch=32
    let mut gpu_ext = gpu.create_nplayer_extsamp_compact_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    let gpu_e_start = std::time::Instant::now();
    gpu_ext.run(32, 1000).expect("GPU run failed");
    let gpu_e_elapsed = gpu_e_start.elapsed();
    let gpu_e_cum = gpu_ext.download_cum_strategy().expect("download failed");
    let gpu_e_offsets = gpu_ext.node_offsets().to_vec();
    let gpu_e_profile = StrategyProfile::from_u32_offsets(&gpu_e_cum, &gpu_e_offsets, nh);
    let gpu_e_exp = exploitability(&tree, &game, &gpu_e_profile);

    // GPU extsamp compact — 10000 iterations
    let mut gpu_ext10k = gpu.create_nplayer_extsamp_compact_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    let gpu_e10k_start = std::time::Instant::now();
    gpu_ext10k.run(32, 10000).expect("GPU run failed");
    let gpu_e10k_elapsed = gpu_e10k_start.elapsed();
    let gpu_e10k_cum = gpu_ext10k.download_cum_strategy().expect("download failed");
    let gpu_e10k_profile = StrategyProfile::from_u32_offsets(&gpu_e10k_cum, gpu_ext10k.node_offsets(), nh);
    let gpu_e10k_exp = exploitability(&tree, &game, &gpu_e10k_profile);

    println!("SC1 HU River ({} hands, {} nodes):", nh, tree.num_nodes());
    println!("  CPU vanilla CFR     1000 iters: exp={:.4}, time={:.3}s", cpu_exp, cpu_elapsed.as_secs_f64());
    println!("  GPU vanilla b=1     1000 iters: exp={:.4}, time={:.3}s", gpu_v1_exp, gpu_v1_elapsed.as_secs_f64());
    println!("  GPU vanilla b=32    1000 iters: exp={:.4}, time={:.3}s", gpu_v32_exp, gpu_v32_elapsed.as_secs_f64());
    println!("  GPU extsamp b=32    1000 iters: exp={:.4}, time={:.3}s", gpu_e_exp, gpu_e_elapsed.as_secs_f64());
    println!("  GPU extsamp b=32   10k iters:  exp={:.4}, time={:.3}s", gpu_e10k_exp, gpu_e10k_elapsed.as_secs_f64());

    // Convergence checks: strategies point in the right direction and improve
    assert!(cpu_exp < 10.0, "CPU should converge, got {:.4}", cpu_exp);
    assert!(gpu_v1_exp < gpu_v1_exp.max(f32::MAX),
        "GPU vanilla b=1 should produce finite exploitability");
    assert!(gpu_v32_exp < 60.0,
        "GPU vanilla b=32 should converge, got {:.4}", gpu_v32_exp);
    assert!(gpu_e10k_exp < gpu_e_exp,
        "10k iters should improve on 1k iters: {:.4} < {:.4}", gpu_e10k_exp, gpu_e_exp);
    assert!(gpu_e10k_exp < 25.0,
        "GPU extsamp 10k should converge below 25, got {:.4}", gpu_e10k_exp);

    // Batch=32 should help over batch=1 (more parallel traversals)
    assert!(gpu_v32_exp <= gpu_v1_exp * 1.5,
        "b=32 should not be much worse than b=1: {:.4} vs {:.4}", gpu_v32_exp, gpu_v1_exp);
}

fn sc1_build_turn_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Turn,
        starting_pot: 200,
        starting_stacks: vec![9500, 9500],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5)],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    build_tree(&config).expect("tree build failed")
}

#[test]
fn sc1_wall_clock_river() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree();

    let hand_ranks = game.hand_ranks_gpu();
    let (s_opp_str, s_opp_idx, s_pl_str, s_pl_idx, same_hand_idx) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);
    let gpu = GpuContext::new().expect("GPU init failed");

    let cpu_iters = 1000;
    let gpu_iters = 1000;

    // CPU vanilla CFR
    let mut cpu_solver = CpuMccfr::new(&tree, vec![nh, nh]);
    let cpu_start = std::time::Instant::now();
    cpu_solver.run(&tree, &game, cpu_iters);
    let cpu_time = cpu_start.elapsed();
    let cpu_profile = StrategyProfile::from_usize_offsets(
        cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh,
    );
    let cpu_exp = exploitability(&tree, &game, &cpu_profile);
    let cpu_ips = cpu_iters as f64 / cpu_time.as_secs_f64();

    // GPU vanilla batch=2 (minimum correct: each player gets 1 traversal)
    let mut gpu_v2 = gpu.create_nplayer_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    let gpu_v2_start = std::time::Instant::now();
    gpu_v2.run(2, gpu_iters).expect("GPU run failed");
    let gpu_v2_time = gpu_v2_start.elapsed();
    let gpu_v2_cum = gpu_v2.download_cum_strategy().expect("download failed");
    let gpu_v2_profile = StrategyProfile::from_u32_offsets(&gpu_v2_cum, gpu_v2.node_offsets(), nh);
    let gpu_v2_exp = exploitability(&tree, &game, &gpu_v2_profile);
    let gpu_v2_ips = (gpu_iters * 2) as f64 / gpu_v2_time.as_secs_f64();

    // GPU vanilla batch=2, 10000 iters (match CPU total traversals)
    let mut gpu_v2_10k = gpu.create_nplayer_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    let gpu_v2_10k_start = std::time::Instant::now();
    gpu_v2_10k.run(2, 10000).expect("GPU run failed");
    let gpu_v2_10k_time = gpu_v2_10k_start.elapsed();
    let gpu_v2_10k_cum = gpu_v2_10k.download_cum_strategy().expect("download failed");
    let gpu_v2_10k_profile = StrategyProfile::from_u32_offsets(&gpu_v2_10k_cum, gpu_v2_10k.node_offsets(), nh);
    let gpu_v2_10k_exp = exploitability(&tree, &game, &gpu_v2_10k_profile);

    // GPU vanilla batch=32
    let mut gpu_v32 = gpu.create_nplayer_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    let gpu_v32_start = std::time::Instant::now();
    gpu_v32.run(32, gpu_iters).expect("GPU run failed");
    let gpu_v32_time = gpu_v32_start.elapsed();
    let gpu_v32_cum = gpu_v32.download_cum_strategy().expect("download failed");
    let gpu_v32_profile = StrategyProfile::from_u32_offsets(&gpu_v32_cum, gpu_v32.node_offsets(), nh);
    let gpu_v32_exp = exploitability(&tree, &game, &gpu_v32_profile);
    let gpu_v32_ips = (gpu_iters * 32) as f64 / gpu_v32_time.as_secs_f64();

    // GPU extsamp compact batch=32
    let mut gpu_ext = gpu.create_nplayer_extsamp_compact_solver(
        &tree, nh,
        &hand_ranks, &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        None, &[], None, None,
    ).expect("solver creation failed");
    let gpu_ext_start = std::time::Instant::now();
    gpu_ext.run(32, gpu_iters).expect("GPU run failed");
    let gpu_ext_time = gpu_ext_start.elapsed();
    let gpu_ext_cum = gpu_ext.download_cum_strategy().expect("download failed");
    let gpu_ext_profile = StrategyProfile::from_u32_offsets(&gpu_ext_cum, gpu_ext.node_offsets(), nh);
    let gpu_ext_exp = exploitability(&tree, &game, &gpu_ext_profile);
    let gpu_ext_ips = (gpu_iters * 32) as f64 / gpu_ext_time.as_secs_f64();

    println!("\n=== SC1 River Wall-Clock (release, {} nodes, {} hands) ===", tree.num_nodes(), nh);
    println!("  CPU vanilla       {} iters: {:.0}ms ({:.0} traj/s) exp={:.4}",
        cpu_iters, cpu_time.as_secs_f64() * 1000.0, cpu_ips, cpu_exp);
    println!("  GPU vanilla b2    {} iters: {:.0}ms ({:.0} traj/s) exp={:.4}",
        gpu_iters, gpu_v2_time.as_secs_f64() * 1000.0, gpu_v2_ips, gpu_v2_exp);
    println!("  GPU vanilla b2   10k iters: {:.0}ms exp={:.4}",
        gpu_v2_10k_time.as_secs_f64() * 1000.0, gpu_v2_10k_exp);
    println!("  GPU vanilla b32   {} iters: {:.0}ms ({:.0} traj/s) exp={:.4}",
        gpu_iters, gpu_v32_time.as_secs_f64() * 1000.0, gpu_v32_ips, gpu_v32_exp);
    println!("  GPU extsamp b32   {} iters: {:.0}ms ({:.0} traj/s) exp={:.4}",
        gpu_iters, gpu_ext_time.as_secs_f64() * 1000.0, gpu_ext_ips, gpu_ext_exp);
    println!("  Throughput: v2={:.1}x, v32={:.1}x, extsamp={:.1}x vs CPU",
        gpu_v2_ips / cpu_ips, gpu_v32_ips / cpu_ips, gpu_ext_ips / cpu_ips);

    assert!(cpu_exp < 1.0, "CPU should converge below 1.0, got {:.4}", cpu_exp);
}

#[test]
fn sc1_wall_clock_turn() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();
    let tree = sc1_build_turn_tree();
    let num_chance = tree.nodes.iter().filter(|n| n.is_chance()).count();
    println!("Turn tree: {} nodes, {} chance, {} hands", tree.num_nodes(), num_chance, nh);

    let (s_opp_str, s_opp_idx, s_pl_str, s_pl_idx, same_hand_idx) = table.sorted_opp_arrays();
    let (ch_str, ch_idx) = table.chance_sorted_arrays_gpu();
    let hand_cards = table.hand_cards_gpu();
    let initial_weight = table.initial_weight_flat();

    let gpu = GpuContext::new().expect("GPU init failed");

    let gpu_iters = 1000;

    // GPU extsamp compact batch=32 with chance
    let mut gpu_ext = gpu.create_nplayer_extsamp_compact_solver(
        &tree, nh,
        &table.hand_ranks_gpu(), &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        Some(&table.chance_ranks_gpu()), &table.remaining_deck_gpu(),
        Some(&ch_str), Some(&ch_idx),
    ).expect("solver creation failed");
    let gpu_ext_start = std::time::Instant::now();
    gpu_ext.run(32, gpu_iters).expect("GPU run failed");
    let gpu_ext_time = gpu_ext_start.elapsed();
    let gpu_ext_ips = (gpu_iters * 32) as f64 / gpu_ext_time.as_secs_f64();

    // GPU vanilla batch=32 with chance
    let mut gpu_v32 = gpu.create_nplayer_solver(
        &tree, nh,
        &table.hand_ranks_gpu(), &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
        &same_hand_idx, &hand_cards, &initial_weight,
        Some(&table.chance_ranks_gpu()), &table.remaining_deck_gpu(),
        Some(&ch_str), Some(&ch_idx),
    ).expect("solver creation failed");
    let gpu_v32_start = std::time::Instant::now();
    gpu_v32.run(32, gpu_iters).expect("GPU run failed");
    let gpu_v32_time = gpu_v32_start.elapsed();
    let gpu_v32_ips = (gpu_iters * 32) as f64 / gpu_v32_time.as_secs_f64();

    println!("\n=== SC1 Turn Wall-Clock (release, {} nodes, {} chance, {} hands) ===",
        tree.num_nodes(), num_chance, nh);
    println!("  GPU vanilla b32 {} iters: {:.0}ms ({:.0} traj/s)",
        gpu_iters, gpu_v32_time.as_secs_f64() * 1000.0, gpu_v32_ips);
    println!("  GPU extsamp b32 {} iters: {:.0}ms ({:.0} traj/s)",
        gpu_iters, gpu_ext_time.as_secs_f64() * 1000.0, gpu_ext_ips);
    println!("  Extsamp/vanilla throughput: {:.1}x", gpu_ext_ips / gpu_v32_ips);
    println!("  Note: exploitability requires TurnStartGame+best_response integration (future work)");
}
