/// Phase 3: Flop-start CPU solver validation.
/// Tests FlopStartVectorCfr with per-outcome regrets on a tree with chance nodes.

use solver_core::card::{card_from_str, Card};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn uniform_range_2p() -> Vec<Vec<f32>> {
    // 1326 possible hands, all weight 1.0
    vec![vec![1.0f32; 1326], vec![1.0; 1326]]
}

#[test]
fn flop_start_tree_builds_with_chance_nodes() {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = uniform_range_2p();

    // Build a simple flop-start tree: check/bet → chance(turn) → check/bet → chance(river) → check/bet → terminal
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 10,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)], // pot-sized bet
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    };

    let tree = build_tree(&config).expect("tree build should succeed");
    println!("Tree: {} nodes, {} decision nodes, max_depth={}, num_infosets={}",
        tree.num_nodes(), tree.decision_node_ids.len(), tree.max_depth, tree.num_infosets);

    // Print the tree structure
    for level in 0..=tree.max_depth {
        let nodes = tree.nodes_at_level(level);
        for &nid in nodes {
            let n = &tree.nodes[nid as usize];
            let node_type = if n.is_terminal() { "T" } else if n.is_chance() { "C" } else { "P" };
            let children: Vec<u32> = tree.node_children(nid as usize).to_vec();
            println!("  Level {}: node {} type={} player={} board={} children={:?}",
                level, nid, node_type, n.player_id, n.board_state, children);
        }
    }

    // Verify there are chance nodes
    let chance_nodes: Vec<_> = (0..tree.num_nodes())
        .filter(|&i| tree.nodes[i].is_chance())
        .collect();
    assert!(!chance_nodes.is_empty(), "Tree should have chance nodes");
    println!("Chance nodes: {:?}", chance_nodes);

    // Check board states
    for &cn in &chance_nodes {
        let bs = tree.nodes[cn].board_state;
        println!("  Chance node {} board_state={}", cn, bs);
        assert!(bs == 1 || bs == 2, "Chance node should be Turn(1) or River(2)");
    }
}

#[test]
fn flop_start_solver_initializes() {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = uniform_range_2p();

    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 10,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    };

    let tree = build_tree(&config).expect("tree build should succeed");
    let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    let game = FlopStartGame::new(table);

    let solver = FlopStartVectorCfr::new(&tree, game.table());

    println!("Solver initialized:");
    println!("  nh = {}", solver.num_hands());
    println!("  n_turn = {}", solver.n_turn_outcomes());
    println!("  max_n_river = {}", solver.max_river_outcomes());
    println!("  flop_infosets = {}", solver.flop_infosets());
    println!("  turn_infosets = {}", solver.turn_infosets());
    println!("  river_infosets = {}", solver.river_infosets());
    println!("  memory = {:.1} MB", solver.memory_usage() as f64 / 1e6);

    assert!(solver.flop_infosets() > 0, "Should have flop zone infosets");
    assert!(solver.turn_infosets() > 0, "Should have turn zone infosets");
    assert!(solver.river_infosets() > 0, "Should have river zone infosets");
    assert!(solver.memory_usage() > 0, "Should have nonzero memory");
}

#[test]
fn flop_start_solver_runs_iterations() {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = uniform_range_2p();

    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 10,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    };

    let tree = build_tree(&config).expect("tree build should succeed");
    let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    let game = FlopStartGame::new(table);

    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    // Run just 1 iteration first to check correctness
    let cfv = solver.run(&tree, &game, 1);

    assert_eq!(cfv.len(), solver.num_hands(), "CFV should have nh entries");
    let nan_count = cfv.iter().filter(|v| v.is_nan()).count();
    let inf_count = cfv.iter().filter(|v| v.is_infinite()).count();
    println!("After 1 iter: nan={}, inf={}, min={:.4}, max={:.4}",
        nan_count, inf_count,
        cfv.iter().cloned().fold(f32::INFINITY, f32::min),
        cfv.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    assert_eq!(nan_count, 0, "No NaN in CFV");
    assert_eq!(inf_count, 0, "No Inf in CFV");
    // 1 iteration × 2 traversers = 2 half-iterations
    assert!(solver.iteration_count() == 1, "Should have done 1 iteration, got {}", solver.iteration_count());
}
