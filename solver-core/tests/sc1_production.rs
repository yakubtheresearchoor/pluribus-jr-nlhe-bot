use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::game::GameSpec;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

fn build_turn_tree(bet_sizes: &[f64], raise_sizes: &[f64]) -> solver_core::tree::flat::FlatTree {
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Turn,
        starting_pot: 200,
        starting_stacks: vec![9500, 9500],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: bet_sizes.iter().map(|&f| BetSize::PotRelative(f)).collect(),
            raise: raise_sizes.iter().map(|&f| BetSize::PotRelative(f)).collect(),
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    build_tree(&config).expect("tree build failed")
}

fn build_flop_tree(np: u8, bet_sizes: &[f64], raise_sizes: &[f64], stacks: Vec<i32>) -> solver_core::tree::flat::FlatTree {
    let config = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 100 * np as i32,
        starting_stacks: stacks,
        initial_contributions: vec![0; np as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: bet_sizes.iter().map(|&f| BetSize::PotRelative(f)).collect(),
            raise: raise_sizes.iter().map(|&f| BetSize::PotRelative(f)).collect(),
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    build_tree(&config).expect("tree build failed")
}

#[test]
fn sc1_production_trees_survey() {
    let configs: Vec<(&str, solver_core::tree::flat::FlatTree)> = vec![
        ("HU turn 1bet", build_turn_tree(&[0.5], &[])),
        ("HU turn 2bet+raise", build_turn_tree(&[0.5, 1.0], &[0.5])),
        ("HU flop 1bet+raise", build_flop_tree(2, &[0.5], &[1.0], vec![500, 500])),
        ("HU flop 2bet+raise", build_flop_tree(2, &[0.33, 0.5, 0.75], &[0.5, 1.0], vec![500, 500])),
        ("HU flop deep 2bet", build_flop_tree(2, &[0.5, 1.0], &[0.5, 1.0], vec![2000, 2000])),
        ("3p flop 1bet", build_flop_tree(3, &[0.5], &[], vec![500, 500, 500])),
    ];

    println!("\n=== Production Tree Survey ===");
    println!("{:<25} {:>8} {:>8} {:>8}", "Config", "Nodes", "Chance", "Player");

    for (name, tree) in &configs {
        let chance = tree.nodes.iter().filter(|n| n.is_chance()).count();
        let player = tree.nodes.iter().filter(|n| n.is_player()).count();
        println!("{:<25} {:>8} {:>8} {:>8}", name, tree.num_nodes(), chance, player);
    }
}

#[test]
fn sc1_production_gpu_throughput() {
    let gpu = GpuContext::new().expect("GPU init failed");
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();

    let (s_opp_str, s_opp_idx, s_pl_str, s_pl_idx, same_hand_idx) = table.sorted_opp_arrays();
    let (ch_str, ch_idx) = table.chance_sorted_arrays_gpu();
    let hand_cards = table.hand_cards_gpu();
    let initial_weight = table.initial_weight_flat();

    let trees: Vec<(&str, solver_core::tree::flat::FlatTree, bool)> = vec![
        ("HU turn 2bet+raise", build_turn_tree(&[0.5, 1.0], &[0.5]), true),
        ("HU flop 1bet+raise", build_flop_tree(2, &[0.5], &[1.0], vec![500, 500]), true),
        ("HU flop 2bet+raise", build_flop_tree(2, &[0.33, 0.5, 0.75], &[0.5, 1.0], vec![500, 500]), true),
    ];

    let iters = 500;

    println!("\n=== SC1 GPU Throughput (release, {} iters, batch=32) ===", iters);
    println!("{:<25} {:>8} {:>8} {:>12} {:>10}", "Config", "Nodes", "Chance", "traj/s", "ms");

    for (name, tree, _has_chance) in &trees {
        let chance = tree.nodes.iter().filter(|n| n.is_chance()).count();

        let mut solver = gpu.create_nplayer_extsamp_compact_solver(
            tree, nh,
            &table.hand_ranks_gpu(), &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
            &same_hand_idx, &hand_cards, &initial_weight,
            Some(&table.chance_ranks_gpu()), &table.remaining_deck_gpu(),
            Some(&ch_str), Some(&ch_idx),
        ).expect("solver creation failed");

        let start = std::time::Instant::now();
        solver.run(32, iters).expect("GPU run failed");
        let elapsed = start.elapsed();
        let traj_s = (iters * 32) as f64 / elapsed.as_secs_f64();

        println!("{:<25} {:>8} {:>8} {:>12.0} {:>8.0}ms",
            name, tree.num_nodes(), chance, traj_s, elapsed.as_secs_f64() * 1000.0);
    }
}

#[test]
fn sc1_production_turn_exploitability() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();
    let tree = build_turn_tree(&[0.5, 1.0], &[0.5]);
    let chance = tree.nodes.iter().filter(|n| n.is_chance()).count();
    println!("Turn tree: {} nodes, {} chance, {} hands", tree.num_nodes(), chance, nh);

    let game = TurnStartGame::new(ChanceTable::compute_turn_start(&board, &ranges, 2));

    // CPU convergence
    let mut cpu_solver = CpuMccfr::new(&tree, vec![nh, nh]);
    for &target in &[100, 500] {
        cpu_solver.run(&tree, &game, target - cpu_solver.iteration_count());
        let profile = StrategyProfile::from_usize_offsets(
            cpu_solver.cum_strategy_slice(), cpu_solver.node_offsets(), nh,
        );
        let exp = exploitability(&tree, &game, &profile);
        println!("CPU vanilla {} iters: exp={:.4}", target, exp);
    }

    // GPU extsamp
    let gpu = GpuContext::new().expect("GPU init failed");
    let (s_opp_str, s_opp_idx, s_pl_str, s_pl_idx, same_hand_idx) = table.sorted_opp_arrays();
    let (ch_str, ch_idx) = table.chance_sorted_arrays_gpu();

    for &gpu_iters in &[1000, 5000] {
        let mut gpu_solver = gpu.create_nplayer_extsamp_compact_solver(
            &tree, nh,
            &table.hand_ranks_gpu(), &s_opp_str, &s_opp_idx, &s_pl_str, &s_pl_idx,
            &same_hand_idx, &table.hand_cards_gpu(), &table.initial_weight_flat(),
            Some(&table.chance_ranks_gpu()), &table.remaining_deck_gpu(),
            Some(&ch_str), Some(&ch_idx),
        ).expect("solver creation failed");

        let start = std::time::Instant::now();
        gpu_solver.run(32, gpu_iters).expect("GPU run failed");
        let elapsed = start.elapsed();

        let cum = gpu_solver.download_cum_strategy().expect("download failed");
        let profile = StrategyProfile::from_u32_offsets(&cum, gpu_solver.node_offsets(), nh);
        let exp = exploitability(&tree, &game, &profile);
        println!("GPU extsamp {} iters: {:.0}ms exp={:.4}", gpu_iters, elapsed.as_secs_f64() * 1000.0, exp);
    }
}
