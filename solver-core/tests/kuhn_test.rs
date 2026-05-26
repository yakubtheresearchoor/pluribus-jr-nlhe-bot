use solver_core::solver::game::GameSpec;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::best_response::{StrategyProfile, exploitability, best_response_value, strategy_value};
use solver_core::tree::action::BoardState;
use solver_core::tree::flat::{FlatNode, FlatTree};

const NUM_HANDS: usize = 3;

struct KuhnGame;

impl GameSpec for KuhnGame {
    fn num_hands(&self, _player: u8) -> usize {
        NUM_HANDS
    }

    fn initial_weight(&self, _player: u8) -> Vec<f32> {
        vec![1.0; NUM_HANDS]
    }

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

    fn chance_probability(&self, _outcome: usize, _hand: usize) -> f32 {
        0.0
    }
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

    assert_eq!(tree.num_nodes(), 9);
    tree
}

#[test]
fn kuhn_cfr_convergence() {
    let tree = build_kuhn_tree();
    let game = KuhnGame;
    let mut solver = CpuMccfr::new(&tree, vec![NUM_HANDS, NUM_HANDS]);

    let root_cfv = solver.run(&tree, &game, 10000);

    let avg = solver.get_average_strategy(0, 2, NUM_HANDS);
    let cur = solver.get_current_strategy(0, 2, NUM_HANDS);
    let bet_j = avg[1][0];
    let bet_q = avg[1][1];
    let bet_k = avg[1][2];
    let cur_bet_j = cur[1][0];
    let cur_bet_q = cur[1][1];
    let cur_bet_k = cur[1][2];

    let game_value: f32 = root_cfv.iter().sum::<f32>() / (NUM_HANDS as f32 * (NUM_HANDS - 1) as f32);
    let expected = -1.0 / 18.0;

    println!("Game value: {:.6} (expected {:.6})", game_value, expected);
    println!("Per-hand cfv: {:?}", root_cfv);
    println!("P0 bet prob (avg): J={:.3} Q={:.3} K={:.3}", bet_j, bet_q, bet_k);
    println!("P0 bet prob (cur): J={:.3} Q={:.3} K={:.3}", cur_bet_j, cur_bet_q, cur_bet_k);

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
    assert!(
        bet_k > bet_j,
        "K should bet more than J (value > bluff), got K={} J={}",
        bet_k, bet_j
    );
}

#[test]
fn kuhn_p1_strategy() {
    let tree = build_kuhn_tree();
    let game = KuhnGame;
    let mut solver = CpuMccfr::new(&tree, vec![NUM_HANDS, NUM_HANDS]);
    solver.run(&tree, &game, 10000);

    let node2_strat = solver.get_average_strategy(2, 2, NUM_HANDS);
    let fold_j = node2_strat[0][0];
    let call_q = node2_strat[1][1];
    let call_k = node2_strat[1][2];

    println!("P1 after P0 bet:");
    println!("  J fold: {:.3}", fold_j);
    println!("  Q call: {:.3}", call_q);
    println!("  K call: {:.3}", call_k);

    assert!(fold_j > 0.90, "P1 J should fold after bet, got fold={}", fold_j);
    assert!(call_k > 0.90, "P1 K should call after bet, got call={}", call_k);
    assert!(
        call_q > 0.10 && call_q < 0.60,
        "P1 Q should mixed call after bet, got {}",
        call_q
    );
}

#[test]
fn kuhn_exploitability_convergence() {
    let tree = build_kuhn_tree();
    let game = KuhnGame;

    let mut solver = CpuMccfr::new(&tree, vec![NUM_HANDS, NUM_HANDS]);

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

    println!("Kuhn exploitability: 100 iters={:.6}, 1000={:.6}, 10000={:.6}", exp_100, exp_1000, exp_10000);

    assert!(exp_1000 < exp_100 * 0.8,
        "Exploitability should decrease: 1000 iters ({:.6}) < 80% of 100 iters ({:.6})", exp_1000, exp_100);
    assert!(exp_10000 < exp_1000 * 0.5,
        "Exploitability should decrease: 10000 iters ({:.6}) < 50% of 1000 iters ({:.6})", exp_10000, exp_1000);
    assert!(exp_10000 < 0.05,
        "After 10000 iterations, exploitability should be near zero, got {:.6}", exp_10000);
}

#[test]
fn kuhn_best_response_symmetry() {
    let tree = build_kuhn_tree();
    let game = KuhnGame;
    let mut solver = CpuMccfr::new(&tree, vec![NUM_HANDS, NUM_HANDS]);
    solver.run(&tree, &game, 5000);

    let profile = StrategyProfile::from_usize_offsets(
        solver.cum_strategy_slice(),
        solver.node_offsets(),
        NUM_HANDS,
    );

    let br_p0 = best_response_value(&tree, &game, &profile, 0);
    let br_p1 = best_response_value(&tree, &game, &profile, 1);
    let sv_p0 = strategy_value(&tree, &game, &profile, 0);
    let sv_p1 = strategy_value(&tree, &game, &profile, 1);

    let reach0 = game.initial_weight(0);
    let reach1 = game.initial_weight(1);
    let r0_sum: f32 = reach0.iter().sum();
    let r1_sum: f32 = reach1.iter().sum();

    let avg_br0: f32 = br_p0.iter().enumerate().map(|(h, &v)| reach0[h] / r0_sum * v).sum();
    let avg_br1: f32 = br_p1.iter().enumerate().map(|(h, &v)| reach1[h] / r1_sum * v).sum();
    let avg_sv0: f32 = sv_p0.iter().enumerate().map(|(h, &v)| reach0[h] / r0_sum * v).sum();
    let avg_sv1: f32 = sv_p1.iter().enumerate().map(|(h, &v)| reach1[h] / r1_sum * v).sum();

    println!("P0: BR={:.6}, σ_value={:.6}, gap={:.6}", avg_br0, avg_sv0, avg_br0 - avg_sv0);
    println!("P1: BR={:.6}, σ_value={:.6}, gap={:.6}", avg_br1, avg_sv1, avg_br1 - avg_sv1);

    assert!((avg_sv0 + avg_sv1).abs() < 0.01,
        "2-player zero-sum: EV should sum to ~0, got {:.6}", avg_sv0 + avg_sv1);
    assert!(avg_br0 >= avg_sv0 - 0.01,
        "BR value should >= strategy value for P0");
    assert!(avg_br1 >= avg_sv1 - 0.01,
        "BR value should >= strategy value for P1");
}
