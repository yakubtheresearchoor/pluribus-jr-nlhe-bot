use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::game::GameSpec;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::best_response::{StrategyProfile, exploitability};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

#[test]
fn cpu_turn_start_runs() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();

    let game = TurnStartGame::new(table);
    assert_eq!(game.num_valid_hands(), nh);
    assert!(game.num_chance_outcomes() > 0, "should have chance outcomes");

    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Turn,
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
    button_player: None,
            max_bets_per_street: None,

    };

    let tree = build_tree(&config).expect("tree build failed");
    let num_chance = tree.nodes.iter().filter(|n| n.is_chance()).count();
    println!("Turn tree: {} nodes, {} chance nodes", tree.num_nodes(), num_chance);

    let mut solver = CpuMccfr::new(&tree, vec![nh, nh]);

    solver.run(&tree, &game, 100);

    let regrets = solver.regrets_slice();
    let nonzero = regrets.iter().filter(|&&r| r != 0.0).count();
    println!("CPU turn start: {}/{} non-zero regrets after 100 iters", nonzero, regrets.len());
    assert!(nonzero > 0, "should have non-zero regrets");
}

#[test]
fn cpu_turn_start_chance_probability() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);

    let game = TurnStartGame::new(table);
    let nh = game.num_valid_hands();
    let n_outcomes = game.num_chance_outcomes();

    println!("Chance outcomes: {}", n_outcomes);

    for h in 0..nh.min(5) {
        let mut total_prob = 0.0f32;
        let mut zero_count = 0usize;
        for outcome in 0..n_outcomes {
            let p = game.chance_probability(outcome, h);
            total_prob += p;
            if p == 0.0 { zero_count += 1; }
        }
        println!("Hand {}: total_prob={:.6}, zeros={}/{}", h, total_prob, zero_count, n_outcomes);
        assert!((total_prob - 1.0).abs() < 0.01, "chance probs should sum to ~1.0, got {}", total_prob);
    }
}

#[test]
fn cpu_turn_start_exploitability_convergence() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();

    let game = TurnStartGame::new(table);

    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Turn,
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
    button_player: None,
            max_bets_per_street: None,

    };

    let tree = build_tree(&config).expect("tree build failed");

    let mut solver = CpuMccfr::new(&tree, vec![nh, nh]);

    let mut prev_exp = f32::MAX;
    let checkpoints = [100, 500, 1000];

    for &target in &checkpoints {
        solver.run(&tree, &game, target - solver.iteration_count());
        let profile = StrategyProfile::from_usize_offsets(
            solver.cum_strategy_slice(), solver.node_offsets(), nh,
        );
        let exp = exploitability(&tree, &game, &profile);
        println!("Turn exploitability at {} iters: {:.4}", target, exp);
        assert!(exp < prev_exp + 0.01,
            "Exploitability should not increase: {:.4} -> {:.4} at {} iters", prev_exp, exp, target);
        prev_exp = exp;
    }

    assert!(prev_exp < 5.0, "After 1000 iters, exploitability should be reasonable, got {:.4}", prev_exp);
}
