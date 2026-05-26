#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::chance_table::ChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatNode;

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

#[test]
fn gpu_chance_kernel_river_fallback() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board[..4], &ranges, 2);
    let nh = table.num_valid_hands();
    println!("River fallback: {} valid hands", nh);

    let mut tree = solver_core::tree::flat::FlatTree::new(2, 10, vec![95, 95], 0.0, 0.0);
    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n0, 0, 5); tree.set_contribution(n0, 1, 5);
    let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n1, 0, 5); tree.set_contribution(n1, 1, 5);
    let n2 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n2, 0, 5); tree.set_contribution(n2, 1, 5);
    let n3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n3, 0, 10); tree.set_contribution(n3, 1, 5);
    let n4 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n4, 0, 10); tree.set_contribution(n4, 1, 10);
    tree.set_children(n0, vec![n1 as u32, n3 as u32]);
    tree.set_children(n1, vec![n2 as u32, n4 as u32]);
    tree.set_folded_mask(n3, 0b10);

    let (sorted_opp_strength, sorted_opp_indices, sorted_player_strength, sorted_player_indices, same_hand_idx) = table.sorted_opp_arrays();
    let (chance_sorted_str, chance_sorted_idx) = table.chance_sorted_arrays_gpu();

    let mut solver = gpu
        .create_nplayer_solver(
            &tree, nh,
            &table.hand_ranks_gpu(),
            &sorted_opp_strength, &sorted_opp_indices,
            &sorted_player_strength, &sorted_player_indices,
            &same_hand_idx,
            &table.hand_cards_gpu(),
            &table.initial_weight_flat(),
            Some(&table.chance_ranks_gpu()),
            &table.remaining_deck_gpu(),
            Some(&chance_sorted_str), Some(&chance_sorted_idx),
        )
        .expect("solver creation failed");

    solver.run(32, 50).expect("GPU run failed");
    let regrets = solver.download_regrets().expect("download failed");
    let nonzero = regrets.iter().filter(|&&r| r != 0.0).count();
    println!("River fallback: {}/{} non-zero regrets", nonzero, regrets.len());
    assert!(nonzero > 0, "should have non-zero regrets");
}

#[test]
fn gpu_turn_start_smoke() {
    let gpu = GpuContext::new().expect("GPU init failed");

    let board: Vec<Card> = ["2h", "7d", "Ks", "4c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid_hands();
    println!("Turn start: {} valid hands, {} remaining deck", nh, table.remaining_deck.len());

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
    };

    let tree = build_tree(&config).expect("tree build failed");
    let num_chance = tree.nodes.iter().filter(|n| n.is_chance()).count();
    println!("Turn tree: {} nodes, {} chance", tree.num_nodes(), num_chance);

    let (sorted_opp_strength, sorted_opp_indices, sorted_player_strength, sorted_player_indices, same_hand_idx) = table.sorted_opp_arrays();
    let (chance_sorted_str, chance_sorted_idx) = table.chance_sorted_arrays_gpu();

    let mut solver = gpu
        .create_nplayer_solver(
            &tree, nh,
            &table.hand_ranks_gpu(),
            &sorted_opp_strength, &sorted_opp_indices,
            &sorted_player_strength, &sorted_player_indices,
            &same_hand_idx,
            &table.hand_cards_gpu(),
            &table.initial_weight_flat(),
            Some(&table.chance_ranks_gpu()),
            &table.remaining_deck_gpu(),
            Some(&chance_sorted_str), Some(&chance_sorted_idx),
        )
        .expect("solver creation failed");

    solver.run(32, 10).expect("GPU run failed");

    let regrets = solver.download_regrets().expect("download failed");
    let nonzero = regrets.iter().filter(|&&r| r != 0.0).count();
    println!("Turn start: {}/{} non-zero regrets", nonzero, regrets.len());
    assert!(nonzero > 0, "should have non-zero regrets");
}
