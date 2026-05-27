#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::game::GameSpec;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::flat::{FlatNode, FlatTree, MAX_NA};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

fn make_board() -> Vec<Card> {
    ["2h", "7d", "Ks", "4c", "9s"]
        .iter().map(|s| card_from_str(s).unwrap()).collect()
}

fn build_river_tree_2bet() -> FlatTree {
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::River,
        starting_pot: 200,
        starting_stacks: vec![9500, 9500],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    build_tree(&config).expect("tree build failed")
}

#[test]
fn diagnose_production_tree() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree_2bet();

    println!("Production river 2bet tree:");
    println!("  num_nodes={}, num_infosets={}, max_depth={}", tree.num_nodes(), tree.num_infosets, tree.max_depth);

    for i in 0..tree.num_nodes() {
        let node = &tree.nodes[i];
        let c0 = tree.get_contribution(i, 0);
        let c1 = tree.get_contribution(i, 1);
        let fm = tree.get_folded_mask(i);
        let is = tree.infoset_offsets[i];
        match node.node_type {
            0 => println!("  node {}: TERMINAL contrib=[{},{}] fold={:02b}", i, c0, c1, fm),
            1 => println!("  node {}: CHANCE children={:?} contrib=[{},{}]", i, tree.node_children(i), c0, c1),
            2 => println!("  node {}: PLAYER{} children={:?} contrib=[{},{}] infoset={}",
                i, node.player_id, tree.node_children(i), c0, c1, is),
            _ => println!("  node {}: UNKNOWN type={}", i, node.node_type),
        }
    }

    // Print level structure
    for level in 0..=tree.max_depth {
        let nodes = tree.nodes_at_level(level as u32);
        println!("  level {}: {:?}", level, nodes);
    }

    // Run trajectory MCCFR for 100 iters and show exploitability
    let mut traj = CpuMccfr::new(&tree, vec![nh, nh]);
    traj.run(&tree, &game, 100);
    let traj_profile = StrategyProfile::from_usize_offsets(
        traj.cum_strategy_slice(), traj.node_offsets(), nh,
    );
    let traj_exp = exploitability(&tree, &game, &traj_profile);
    println!("\nTraj MCCFR 100 iters: exp={:.4}", traj_exp);

    // Run GPU vcfr for 100 iters
    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let gpu = GpuContext::new().expect("GPU init failed");
    let mut vcfr = gpu.create_vcfr_solver(
        &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight, None,
    ).expect("vcfr creation failed");
    vcfr.run(100).expect("GPU run failed");

    let regrets = vcfr.download_regrets().expect("download failed");
    let cum = vcfr.download_cum_strategy().expect("download failed");

    let data_per_infoset = MAX_NA * nh;
    let node_offsets: Vec<usize> = (0..tree.num_nodes())
        .map(|i| {
            let is = tree.infoset_offsets[i];
            if is == u32::MAX { usize::MAX } else { is as usize * data_per_infoset }
        })
        .collect();

    let vcfr_profile = StrategyProfile::from_usize_offsets(&cum, &node_offsets, nh);
    let vcfr_exp = exploitability(&tree, &game, &vcfr_profile);
    println!("GPU VCFR 100 iters: exp={:.4}", vcfr_exp);

    // Print regret stats per infoset
    println!("\nRegret stats:");
    for infoset in 0..tree.num_infosets as usize {
        let node_id = tree.decision_node_ids[infoset];
        let na = tree.nodes[node_id as usize].num_children as usize;

        let mut min_r = f32::MAX;
        let mut max_r = f32::MIN;
        for a in 0..na {
            for h in 0..nh {
                let v = regrets[infoset * data_per_infoset + a * nh + h];
                min_r = min_r.min(v);
                max_r = max_r.max(v);
            }
        }
        println!("  infoset={} (node={} na={}): min_reg={:.1} max_reg={:.1}", infoset, node_id, na, min_r, max_r);
    }

    assert!(false, "Diagnostic");
}
