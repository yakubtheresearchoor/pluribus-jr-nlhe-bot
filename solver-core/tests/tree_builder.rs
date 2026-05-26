use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
fn test_2player_tree_builds() {
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 100,
        starting_stacks: vec![500, 500],
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

    println!("2-player tree: {} nodes", tree.num_nodes());

    assert!(tree.num_nodes() > 10, "tree should have more than 10 nodes");
    assert!(tree.nodes[0].is_player(), "root should be a player node");

    let terminal_count = tree.nodes.iter().filter(|n| n.is_terminal()).count();
    assert!(terminal_count > 0, "should have terminal nodes");
}

#[test]
fn test_3player_tree_builds() {
    let config = TreeConfig {
        num_players: 3,
        initial_state: BoardState::Flop,
        starting_pot: 150,
        starting_stacks: vec![500, 500, 500],
        initial_contributions: vec![50, 100, 0],
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

    println!("3-player tree: {} nodes", tree.num_nodes());

    assert!(tree.num_nodes() > 20, "3-player tree should be larger than 2-player");
    assert!(tree.nodes[0].is_player(), "root should be a player node");

    let terminal_count = tree.nodes.iter().filter(|n| n.is_terminal()).count();
    assert!(terminal_count > 0, "should have terminal nodes");

    for node in &tree.nodes {
        if node.is_player() {
            assert!(
                (node.player_id as usize) < 3,
                "player_id {} exceeds num_players",
                node.player_id
            );
        }
    }
}

#[test]
fn test_3player_contributions_track_correctly() {
    let config = TreeConfig {
        num_players: 3,
        initial_state: BoardState::Flop,
        starting_pot: 150,
        starting_stacks: vec![500, 500, 500],
        initial_contributions: vec![50, 100, 0],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![],
            raise: vec![],
        },
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
    };

    let tree = build_tree(&config).unwrap();

    let root_contrib_0 = tree.get_contribution(0, 0);
    let root_contrib_1 = tree.get_contribution(0, 1);
    let root_contrib_2 = tree.get_contribution(0, 2);

    assert_eq!(root_contrib_0, 50, "player 0 (SB) contribution should be 50");
    assert_eq!(root_contrib_1, 100, "player 1 (BB) contribution should be 100");
    assert_eq!(root_contrib_2, 0, "player 2 (BTN) contribution should be 0");
}

#[test]
fn test_single_bet_size_tree() {
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
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
    };

    let tree = build_tree(&config).unwrap();

    println!("River-only tree: {} nodes", tree.num_nodes());

    assert!(tree.num_nodes() > 5, "river tree should have nodes");

    let has_chance = tree.nodes.iter().any(|n| n.is_chance());
    assert!(!has_chance, "river-only tree should not have chance nodes");
}

#[test]
fn test_6player_tree_builds() {
    let ante = 10i32;
    let sb = 50i32;
    let bb = 100i32;
    let initial_stack = 5000i32;

    let initial_contributions = vec![sb, bb, ante, ante, ante, ante];
    let starting_stacks: Vec<i32> = initial_contributions
        .iter()
        .map(|&c| initial_stack - c)
        .collect();
    let starting_pot: i32 = initial_contributions.iter().sum();

    let config = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot,
        starting_stacks,
        initial_contributions,
        rake_rate: 0.05,
        rake_cap: 30.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };

    let tree = build_tree(&config).unwrap();

    println!("6-player tree: {} nodes", tree.num_nodes());

    assert!(tree.num_nodes() > 100, "6-player tree should be large");
    assert_eq!(tree.num_players, 6);
    assert_eq!(tree.rake_rate, 0.05);
    assert_eq!(tree.rake_cap, 30.0);

    assert_eq!(tree.get_contribution(0, 0), sb, "SB contribution");
    assert_eq!(tree.get_contribution(0, 1), bb, "BB contribution");
    for p in 2..6 {
        assert_eq!(
            tree.get_contribution(0, p),
            ante,
            "player {} ante contribution",
            p
        );
    }

    for node in &tree.nodes {
        if node.is_player() {
            assert!(
                (node.player_id as usize) < 6,
                "player_id {} exceeds num_players",
                node.player_id
            );
        }
    }
}

#[test]
fn test_6player_river_tree_reasonable_size() {
    let initial_contributions = vec![0, 0, 0, 0, 0, 0];
    let starting_stacks = vec![400, 400, 400, 400, 400, 400];

    let config = TreeConfig {
        num_players: 6,
        initial_state: BoardState::River,
        starting_pot: 600,
        starting_stacks,
        initial_contributions,
        rake_rate: 0.05,
        rake_cap: 30.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };

    let tree = build_tree(&config).unwrap();

    println!("6-player river tree: {} nodes", tree.num_nodes());
    println!("6-player river GPU bytes: {}", tree.gpu_bytes());

    assert!(tree.num_nodes() > 10, "should have nodes");
    assert!(tree.num_nodes() < 1_000_000, "river tree with 1 bet size should be tractable");

    let has_chance = tree.nodes.iter().any(|n| n.is_chance());
    assert!(!has_chance, "river-only tree should not have chance nodes");
}
