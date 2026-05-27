#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::flat::{FlatNode, FlatTree, MAX_NA};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

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

fn build_toy_river_tree() -> FlatTree {
    let mut tree = FlatTree::new(2, 10, vec![95, 95], 0.0, 0.0);
    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n0, 0, 5); tree.set_contribution(n0, 1, 5);
    let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n1, 0, 5); tree.set_contribution(n1, 1, 5);
    let n2 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n2, 0, 10); tree.set_contribution(n2, 1, 5);
    let n3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n3, 0, 5); tree.set_contribution(n3, 1, 5);
    let n4 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n4, 0, 5); tree.set_contribution(n4, 1, 10);
    let n5 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n5, 0, 10); tree.set_contribution(n5, 1, 5);
    let n6 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n6, 0, 10); tree.set_contribution(n6, 1, 10);
    let n7 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n7, 0, 5); tree.set_contribution(n7, 1, 10);
    let n8 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n8, 0, 10); tree.set_contribution(n8, 1, 10);
    tree.set_children(n0, vec![1, 2]);
    tree.set_children(n1, vec![3, 4]);
    tree.set_children(n2, vec![5, 6]);
    tree.set_children(n4, vec![7, 8]);
    tree.set_folded_mask(n5, 0b10);
    tree.set_folded_mask(n7, 0b01);
    tree.compute_levels();
    tree
}

fn run_comparison(
    label: &str,
    tree: &FlatTree,
    game: &dyn GameSpec,
    nh: usize,
    opp_str: &[u16], opp_idx: &[u16],
    pl_str: &[u16], pl_idx: &[u16],
    hand_cards: &[u8],
    initial_weight: &[f32],
    gpu: &GpuContext,
) {
    let num_infosets = tree.num_infosets as usize;
    let data_per_infoset = MAX_NA * nh;
    let node_offsets_vcfr: Vec<usize> = (0..tree.num_nodes())
        .map(|i| {
            let is = tree.infoset_offsets[i];
            if is == u32::MAX { usize::MAX } else { is as usize * data_per_infoset }
        })
        .collect();

    println!("\n=== V3: {} ({} nodes, {} infosets, {} hands) ===",
        label, tree.num_nodes(), num_infosets, nh);

    println!("{:>6} {:>12} {:>12} {:>10} {:>10}",
        "iters", "traj_mccfr", "vec_cfr", "ratio", "winner");

    for &n_iters in &[100, 500, 1000, 2000, 5000] {
        // Trajectory MCCFR (CPU)
        let mut traj = CpuMccfr::new(tree, vec![nh, nh]);
        traj.run(tree, game, n_iters);
        let traj_profile = StrategyProfile::from_usize_offsets(
            traj.cum_strategy_slice(), traj.node_offsets(), nh,
        );
        let traj_exp = exploitability(tree, game, &traj_profile);

        // Vector CFR (GPU)
        let mut vcfr = gpu.create_vcfr_solver(
            tree, nh, opp_str, opp_idx, pl_str, pl_idx, hand_cards, initial_weight,
        ).expect("vcfr solver creation failed");
        vcfr.run(n_iters).expect("GPU run failed");
        let vcfr_cum = vcfr.download_cum_strategy().expect("download failed");
        let vcfr_profile = StrategyProfile::from_usize_offsets(&vcfr_cum, &node_offsets_vcfr, nh);
        let vcfr_exp = exploitability(tree, game, &vcfr_profile);

        let ratio = vcfr_exp / traj_exp;
        let winner = if ratio < 0.95 { "VECTOR" } else if ratio > 1.05 { "TRAJ" } else { "~" };

        println!("{:>6} {:>12.4} {:>12.4} {:>10.4} {:>10}",
            n_iters, traj_exp, vcfr_exp, ratio, winner);
    }
}

#[test]
fn v3_toy_river_comparison() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_toy_river_tree();

    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let gpu = GpuContext::new().expect("GPU init failed");

    run_comparison(
        "Toy River (9 nodes, 2bet)",
        &tree, &game, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &hand_cards, &initial_weight,
        &gpu,
    );
}

#[test]
fn v3_production_river_2bet_comparison() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree_2bet();

    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    let gpu = GpuContext::new().expect("GPU init failed");

    run_comparison(
        "HU River 2bet (production)",
        &tree, &game, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &hand_cards, &initial_weight,
        &gpu,
    );
}
