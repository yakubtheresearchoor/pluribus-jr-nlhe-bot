#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::game::GameSpec;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;
use postflop_solver::*;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

#[test]
fn sc1_tree_diagnostic() {
    // Our tree
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "Qs"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::River,
        starting_pot: 200,
        starting_stacks: vec![9500, 9500],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5)],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build failed");
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_hands(0);

    println!("\n=== Our Tree ===");
    println!("Nodes: {}", tree.num_nodes());
    println!("Infosets: {}", tree.num_infosets);
    println!("nh: {}", nh);
    println!("Max depth: {}", tree.max_depth);
    
    let mut terminals = 0; let mut chance = 0; let mut player = 0;
    for node in &tree.nodes {
        if node.is_terminal() { terminals += 1; }
        else if node.is_chance() { chance += 1; }
        else { player += 1; }
    }
    println!("Terminals: {}, Chance: {}, Player: {}", terminals, chance, player);
    
    // Print action tree structure (first 3 levels)
    use solver_core::tree::flat::FlatTree;
    println!("\nAction tree (top levels):");
    for level in 0..=tree.max_depth.min(2) {
        let nodes = tree.nodes_at_level(level as u32);
        for &nid in nodes {
            let n = &tree.nodes[nid as usize];
            let children = tree.node_children(nid as usize);
            let action_labels: Vec<String> = children.iter().map(|&c| {
                let cn = &tree.nodes[c as usize];
                format!("{}(amt={})", if cn.is_terminal() { "T" } else if cn.is_chance() { "Ch" } else { "P" }, cn.amount)
            }).collect();
            println!("  L{} N{}: type={} player={} na={} actions=[{}]",
                level, nid, n.node_type, n.player_id, n.num_children, action_labels.join(", "));
        }
    }

    // External solver tree
    let full_range = "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+,J2s+,J2o+,T2s+,T2o+,92s+,92o+,82s+,82o+,72s+,72o+,62s+,62o+,52s+,52o+,42s+,42o+,32s,32o";

    let card_config = CardConfig {
        range: [full_range.parse().unwrap(), full_range.parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(),
        turn: card_from_str("Qs").unwrap(),
        river: card_from_str("4c").unwrap(),
    };

    let ext_bet_sizes = ExtBetSizeOptions::try_from(("50%,100%", "50%")).unwrap();
    let ext_tree_config = ExtTreeConfig {
        initial_state: ExtBoardState::River,
        starting_pot: 200,
        effective_stack: 9500,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [ext_bet_sizes.clone(), ext_bet_sizes.clone()],
        turn_bet_sizes: [ext_bet_sizes.clone(), ext_bet_sizes.clone()],
        river_bet_sizes: [ext_bet_sizes.clone(), ext_bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    };

    let action_tree = ActionTree::new(ext_tree_config).unwrap();
    let ext_game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    
    println!("\n=== External Solver ===");
    println!("Num private hands P0: {}", ext_game.num_private_hands(0));
    println!("Num private hands P1: {}", ext_game.num_private_hands(1));
    let (mem, _) = ext_game.memory_usage();
    println!("Memory (32-bit): {:.2} MB", mem as f64 / 1048576.0);
    
    // Print external solver's available actions at root
    let actions = ext_game.available_actions();
    println!("Root actions: {:?}", actions);
}

use postflop_solver::{
    BetSizeOptions as ExtBetSizeOptions, TreeConfig as ExtTreeConfig,
    BoardState as ExtBoardState, NOT_DEALT,
    flop_from_str, card_from_str as ext_card_from_str,
};
