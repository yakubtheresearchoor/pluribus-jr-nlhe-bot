#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }
fn make_board() -> Vec<Card> {
    ["2h", "7d", "Ks", "4c", "9s"].iter().map(|s| card_from_str(s).unwrap()).collect()
}
fn build_river_tree_2bet() -> solver_core::tree::flat::FlatTree {
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::River,
        starting_pot: 200, starting_stacks: vec![9500, 9500],
        initial_contributions: vec![0, 0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    };
    build_tree(&config).unwrap()
}

#[test]
fn cpu_vcfr_vs_mccfr_production_tree() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let tree = build_river_tree_2bet();

    println!("Production river 2bet: {} nodes, {} infosets, {} hands", tree.num_nodes(), tree.num_infosets, nh);

    for &n_iters in &[100, 500, 1000] {
        let mut mccfr = CpuMccfr::new(&tree, vec![nh, nh]);
        mccfr.run(&tree, &game, n_iters);
        let m_profile = StrategyProfile::from_usize_offsets(
            mccfr.cum_strategy_slice(), mccfr.node_offsets(), nh,
        );
        let m_exp = exploitability(&tree, &game, &m_profile);

        let mut vcfr = VectorCfr::new(&tree, vec![nh, nh]);
        vcfr.run_sequential(&tree, &game, n_iters);
        let v_profile = StrategyProfile::from_usize_offsets(
            vcfr.cum_strategy_slice(), vcfr.node_offsets(), nh,
        );
        let v_exp = exploitability(&tree, &game, &v_profile);

        let ratio = v_exp / m_exp;
        println!("{:>5} iters: mccfr={:.4} vcfr={:.4} ratio={:.4}", n_iters, m_exp, v_exp, ratio);
    }
}
