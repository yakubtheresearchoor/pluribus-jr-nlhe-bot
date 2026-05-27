use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::action::BoardState;
use solver_core::tree::flat::{FlatNode, FlatTree};
use solver_core::tree::builder::build_tree;
use solver_core::tree::action::TreeConfig;
use solver_core::tree::action::BetSize;
use solver_core::tree::action::BetSizeOptions;

const NUM_HANDS: usize = 3;

struct KuhnGame;

impl GameSpec for KuhnGame {
    fn num_hands(&self, _player: u8) -> usize { NUM_HANDS }
    fn initial_weight(&self, _player: u8) -> Vec<f32> { vec![1.0; NUM_HANDS] }

    fn evaluate_terminal(
        &self,
        traverser: u8,
        node_idx: usize,
        tree: &FlatTree,
        cfreach: &[Vec<f32>],
    ) -> Vec<f32> {
        let opp = 1 - traverser as usize;
        let c_t = tree.get_contribution(node_idx, traverser) as f32;
        let c_o = tree.get_contribution(node_idx, opp as u8) as f32;
        let mut cfv = vec![0.0f32; NUM_HANDS];
        let is_showdown = (c_t - c_o).abs() < 0.5;

        if is_showdown {
            for h in 0..NUM_HANDS {
                for h_o in 0..NUM_HANDS {
                    if h != h_o {
                        let sign = if h > h_o { 1.0f32 } else { -1.0f32 };
                        cfv[h] += cfreach[opp][h_o] * c_o * sign;
                    }
                }
            }
        } else {
            let traverser_folded = c_t < c_o;
            for h in 0..NUM_HANDS {
                for h_o in 0..NUM_HANDS {
                    if h != h_o {
                        let payoff = if traverser_folded { -c_t } else { c_o };
                        cfv[h] += cfreach[opp][h_o] * payoff;
                    }
                }
            }
        }
        cfv
    }

    fn chance_probability(&self, _outcome: usize, _hand: usize) -> f32 { 0.0 }
}

fn build_kuhn_tree() -> FlatTree {
    let mut tree = FlatTree::new(2, 2, vec![0, 0], 0.0, 0.0);

    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n0, 0, 1);
    tree.set_contribution(n0, 1, 1);

    let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n1, 0, 1);
    tree.set_contribution(n1, 1, 1);

    let n2 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n2, 0, 2);
    tree.set_contribution(n2, 1, 1);

    let n3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n3, 0, 1);
    tree.set_contribution(n3, 1, 1);

    let n4 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n4, 0, 1);
    tree.set_contribution(n4, 1, 2);

    let n5 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n5, 0, 2);
    tree.set_contribution(n5, 1, 1);

    let n6 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n6, 0, 2);
    tree.set_contribution(n6, 1, 2);

    let n7 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n7, 0, 1);
    tree.set_contribution(n7, 1, 2);

    let n8 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n8, 0, 2);
    tree.set_contribution(n8, 1, 2);

    tree.set_children(n0, vec![1, 2]);
    tree.set_children(n1, vec![3, 4]);
    tree.set_children(n2, vec![5, 6]);
    tree.set_children(n4, vec![7, 8]);

    tree.set_folded_mask(n5, 0b10);
    tree.set_folded_mask(n7, 0b01);

    tree.compute_levels();
    tree
}

#[test]
fn vector_cfr_kuhn_convergence() {
    let tree = build_kuhn_tree();
    let game = KuhnGame;
    let mut solver = VectorCfr::new(&tree, vec![NUM_HANDS, NUM_HANDS]);

    let root_cfv = solver.run(&tree, &game, 10000);

    let avg = solver.get_average_strategy(0, 2, NUM_HANDS);
    let bet_j = avg[1][0];
    let bet_q = avg[1][1];
    let bet_k = avg[1][2];

    let game_value: f32 = root_cfv.iter().sum::<f32>() / (NUM_HANDS as f32 * (NUM_HANDS - 1) as f32);
    let expected = -1.0 / 18.0;

    println!("Vector CFR Kuhn game value: {:.6} (expected {:.6})", game_value, expected);
    println!("P0 bet prob (avg): J={:.3} Q={:.3} K={:.3}", bet_j, bet_q, bet_k);

    assert!(
        (game_value - expected).abs() < 0.01,
        "game value = {}, expected {}",
        game_value,
        expected
    );
    assert!(bet_k > 0.50, "K should mostly bet, got {}", bet_k);
    assert!(bet_q < 0.10, "Q should rarely bet, got {}", bet_q);
    assert!(
        bet_j > 0.05 && bet_j < 0.55,
        "J should bluff sometimes, got {}",
        bet_j
    );
}

#[test]
fn vector_cfr_kuhn_parity_with_mccfr() {
    let tree = build_kuhn_tree();
    let game = KuhnGame;

    let mut vcfr = VectorCfr::new(&tree, vec![NUM_HANDS, NUM_HANDS]);
    let mut mccfr = CpuMccfr::new(&tree, vec![NUM_HANDS, NUM_HANDS]);

    vcfr.run_sequential(&tree, &game, 10000);
    mccfr.run(&tree, &game, 10000);

    let v_strat = vcfr.get_average_strategy(0, 2, NUM_HANDS);
    let m_strat = mccfr.get_average_strategy(0, 2, NUM_HANDS);

    println!("Vector CFR bet[0..2]: {:?}", v_strat[1]);
    println!("CPU MCCFR  bet[0..2]: {:?}", m_strat[1]);

    for a in 0..2 {
        for h in 0..NUM_HANDS {
            let diff = (v_strat[a][h] - m_strat[a][h]).abs();
            assert!(
                diff < 0.05,
                "strategies differ at action {} hand {}: vector={:.4} mccfr={:.4}",
                a, h, v_strat[a][h], m_strat[a][h]
            );
        }
    }

    let v_profile = StrategyProfile::from_usize_offsets(
        vcfr.cum_strategy_slice(), vcfr.node_offsets(), NUM_HANDS,
    );
    let m_profile = StrategyProfile::from_usize_offsets(
        mccfr.cum_strategy_slice(), mccfr.node_offsets(), NUM_HANDS,
    );
    let v_exp = exploitability(&tree, &game, &v_profile);
    let m_exp = exploitability(&tree, &game, &m_profile);
    println!("Vector CFR exploitability: {:.6}", v_exp);
    println!("CPU MCCFR  exploitability: {:.6}", m_exp);
    assert!(v_exp < 0.02, "vector CFR exploitability too high: {:.6}", v_exp);
    assert!(m_exp < 0.02, "CPU MCCFR exploitability too high: {:.6}", m_exp);
}

#[test]
fn vector_cfr_kuhn_exploitability() {
    let tree = build_kuhn_tree();
    let game = KuhnGame;
    let mut solver = VectorCfr::new(&tree, vec![NUM_HANDS, NUM_HANDS]);

    let mut exp_100 = 0.0f32;
    let mut exp_1000 = 0.0f32;
    let mut exp_10000 = 0.0f32;

    solver.run(&tree, &game, 100);
    {
        let profile = StrategyProfile::from_usize_offsets(
            solver.cum_strategy_slice(),
            solver.node_offsets(),
            NUM_HANDS,
        );
        exp_100 = exploitability(&tree, &game, &profile);
    }

    solver.run(&tree, &game, 900);
    {
        let profile = StrategyProfile::from_usize_offsets(
            solver.cum_strategy_slice(),
            solver.node_offsets(),
            NUM_HANDS,
        );
        exp_1000 = exploitability(&tree, &game, &profile);
    }

    solver.run(&tree, &game, 9000);
    {
        let profile = StrategyProfile::from_usize_offsets(
            solver.cum_strategy_slice(),
            solver.node_offsets(),
            NUM_HANDS,
        );
        exp_10000 = exploitability(&tree, &game, &profile);
    }

    println!("Vector CFR Kuhn exploitability: 100={:.6}, 1000={:.6}, 10000={:.6}",
        exp_100, exp_1000, exp_10000);

    assert!(exp_10000 < 0.05,
        "After 10000 iterations, exploitability should be near zero, got {:.6}", exp_10000);
}

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
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

    tree.set_folded_mask(n5, 0b10);
    tree.set_folded_mask(n7, 0b01);

    tree.compute_levels();
    tree
}

#[test]
fn vector_cfr_river_poker_smoke() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    let tree = build_river_tree();
    let mut solver = VectorCfr::new(&tree, vec![nh, nh]);

    let root_cfv = solver.run(&tree, &game, 100);

    let game_value: f32 = root_cfv.iter().sum::<f32>() / nh as f32;
    println!("Vector CFR river game value: {:.4}", game_value);

    assert!(game_value.abs() < 500.0, "game value {} out of range", game_value);

    let strat = solver.get_average_strategy(0, 2, nh);
    let avg_bet: f32 = strat[1].iter().sum::<f32>() / nh as f32;
    assert!(avg_bet > 0.01 && avg_bet < 0.99, "bet prob {} degenerate", avg_bet);
}

#[test]
fn vector_cfr_river_poker_parity() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    let tree = build_river_tree();

    let mut vcfr_seq = VectorCfr::new(&tree, vec![nh, nh]);
    let mut vcfr_batch = VectorCfr::new(&tree, vec![nh, nh]);
    let mut mccfr = CpuMccfr::new(&tree, vec![nh, nh]);

    vcfr_seq.run_sequential(&tree, &game, 2000);
    vcfr_batch.run(&tree, &game, 2000);
    mccfr.run(&tree, &game, 2000);

    let seq_profile = StrategyProfile::from_usize_offsets(
        vcfr_seq.cum_strategy_slice(), vcfr_seq.node_offsets(), nh,
    );
    let batch_profile = StrategyProfile::from_usize_offsets(
        vcfr_batch.cum_strategy_slice(), vcfr_batch.node_offsets(), nh,
    );
    let m_profile = StrategyProfile::from_usize_offsets(
        mccfr.cum_strategy_slice(), mccfr.node_offsets(), nh,
    );
    let seq_exp = exploitability(&tree, &game, &seq_profile);
    let batch_exp = exploitability(&tree, &game, &batch_profile);
    let m_exp = exploitability(&tree, &game, &m_profile);
    println!("Vector CFR sequential exploitability (2000 iters): {:.6}", seq_exp);
    println!("Vector CFR batch exploitability (2000 iters): {:.6}", batch_exp);
    println!("CPU MCCFR exploitability (2000 iters): {:.6}", m_exp);

    assert!(seq_exp < 1.0, "sequential exploitability too high: {:.6}", seq_exp);
    assert!(batch_exp < 30.0, "batch exploitability too high: {:.6}", batch_exp);
}

#[test]
fn vector_cfr_river_poker_exploitability() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();

    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    let tree = build_river_tree();
    let mut solver = VectorCfr::new(&tree, vec![nh, nh]);

    solver.run(&tree, &game, 1000);
    let profile = StrategyProfile::from_usize_offsets(
        solver.cum_strategy_slice(),
        solver.node_offsets(),
        nh,
    );
    let exp = exploitability(&tree, &game, &profile);

    println!("Vector CFR river exploitability (1000 iters): {:.6}", exp);
    assert!(exp < 50.0, "exploitability too high after 1000 iters: {:.6}", exp);
}

#[test]
fn vector_cfr_builder_tree() {
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::River,
        starting_pot: 100,
        starting_stacks: vec![950, 950],
        initial_contributions: vec![50, 50],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };

    let tree = build_tree(&config).unwrap();
    assert!(tree.num_nodes() > 10);

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();

    let mut vcfr = VectorCfr::new(&tree, vec![nh, nh]);
    let mut mccfr = CpuMccfr::new(&tree, vec![nh, nh]);

    vcfr.run_sequential(&tree, &game, 500);
    mccfr.run(&tree, &game, 500);

    let v_profile = StrategyProfile::from_usize_offsets(
        vcfr.cum_strategy_slice(), vcfr.node_offsets(), nh,
    );
    let m_profile = StrategyProfile::from_usize_offsets(
        mccfr.cum_strategy_slice(), mccfr.node_offsets(), nh,
    );
    let v_exp = exploitability(&tree, &game, &v_profile);
    let m_exp = exploitability(&tree, &game, &m_profile);
    println!("Builder tree vcfr exploitability (500): {:.6}", v_exp);
    println!("Builder tree mccfr exploitability (500): {:.6}", m_exp);

    let diff = (v_exp - m_exp).abs();
    assert!(diff < 1.0, "sequential vcfr should match mccfr: vcfr={:.6} mccfr={:.6}", v_exp, m_exp);

    vcfr.run_sequential(&tree, &game, 4500);
    let v_profile = StrategyProfile::from_usize_offsets(
        vcfr.cum_strategy_slice(), vcfr.node_offsets(), nh,
    );
    let v_exp_5k = exploitability(&tree, &game, &v_profile);
    println!("Builder tree vcfr exploitability (5000): {:.6}", v_exp_5k);
    assert!(v_exp_5k < v_exp * 0.5, "should converge: {} iters={:.6} vs {} iters={:.6}",
        5000, v_exp_5k, 500, v_exp);
}
