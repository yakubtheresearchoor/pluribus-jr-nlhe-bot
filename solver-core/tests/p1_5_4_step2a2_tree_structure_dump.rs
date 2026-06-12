// Step 2.A.2 trace continuation: dump root + node 1 structure to determine
// which side (CPU or GPU) is correct at compute_reach_flop.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::game::GameSpec;
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_minimal_asymmetry_game() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["Ah", "Kd", "7c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 2u8;
    let k = 4usize;

    use solver_core::hand::eval::Hand;
    let mut all_with_strength: Vec<(u16, u16)> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        all_with_strength.push((h.evaluate_internal() as u16, idx as u16));
    }
    all_with_strength.sort_by_key(|&(s, _)| s);
    let step = all_with_strength.len() / k;
    let chosen: Vec<u16> = (0..k).map(|i| all_with_strength[i * step].1).collect();

    let mut ranges: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for (rank_idx, &hi) in chosen.iter().enumerate() {
        let strength_frac = rank_idx as f32 / k as f32;
        let p0_weight = 1.0_f32;
        let p1_weight = if strength_frac >= 0.5 { 1.0_f32 } else { 0.5_f32 };
        let (c1, c2) = index_to_card_pair(hi as usize);
        let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
        let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
        ranges[0][pair_idx] = p0_weight;
        ranges[1][pair_idx] = p1_weight;
    }
    let turn_cards: Vec<u8> = vec![
        card_from_str("Td").unwrap() as u8,
        card_from_str("3s").unwrap() as u8,
    ];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("4h").unwrap() as u8,
        card_from_str("Qc").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("2s").unwrap() as u8,
        card_from_str("Js").unwrap() as u8,
    ];
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, num_players, &chosen, &turn_cards, &river_decks,
    );
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 6,
        starting_stacks: vec![50, 50], initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = FlopStartGame::new(table);
    (tree, game)
}

#[test]
#[ignore = "2.A.2 trace: tree structure dump for manual reach derivation"]
fn dump_root_and_node1_structure() {
    let (tree, game) = build_minimal_asymmetry_game();
    let cpu = FlopStartVectorCfr::new(&tree, game.table());

    eprintln!("\n=== TREE STRUCTURE DUMP ===");
    eprintln!("nh={}, num_players={}", cpu.num_hands(), 2);
    eprintln!("Tree: {} total nodes", tree.num_nodes());

    let root = &tree.nodes[0];
    eprintln!("\nNode 0 (root):");
    eprintln!("  node_type: {:?}", root.node_type);
    eprintln!("  player_id: {}", root.player_id);
    eprintln!("  num_children: {}", root.num_children);
    eprintln!("  children: {:?}", tree.node_children(0));

    let n1 = &tree.nodes[1];
    eprintln!("\nNode 1:");
    eprintln!("  node_type: {:?}", n1.node_type);
    eprintln!("  player_id: {}", n1.player_id);
    eprintln!("  num_children: {}", n1.num_children);
    eprintln!("  children: {:?}", tree.node_children(1));

    // What zone is node 0?
    eprintln!("\nP0 initial_weight[0..4]: {:?}", &game.initial_weight(0)[..4]);
    eprintln!("P1 initial_weight[0..4]: {:?}", &game.initial_weight(1)[..4]);

    // The discriminator showed:
    //   reach[idx=8] = reach[node 1, P0, hand 0]
    //   CPU = 0.0, GPU = 0.5
    eprintln!("\n=== EXPECTED REACH MANUAL DERIVATION ===");
    eprintln!("If root (node 0) owned by P0:");
    eprintln!("  reach[node 0, P0, h0] = initial_weight[P0, h0] = 1.0");
    eprintln!("  reach[child, P0, h0] = 1.0 × strategy[a, h0]");
    eprintln!("  With uniform strategy at iter 0 (1/num_children): {}",
        1.0 / root.num_children as f32);
    eprintln!("  → expected reach at first child for P0,h0 = {}",
        1.0 / root.num_children as f32);
    eprintln!();
    eprintln!("If root (node 0) owned by P1:");
    eprintln!("  reach[node 0, P0, h0] = 1.0 (P0 not acting)");
    eprintln!("  reach[child, P0, h0] = 1.0 (copy through, P0 reach unchanged)");
    eprintln!();
    eprintln!("Observed at idx 8 (node 1, P0, h0):");
    eprintln!("  CPU = 0.0");
    eprintln!("  GPU = 0.5");
    eprintln!();
    eprintln!("Note: node 1 may or may not be root's direct child — depends on tree builder.");
    eprintln!("If node 1 IS root's child and root owned by P0 with 2 actions: expected = 0.5 → GPU CORRECT, CPU WRONG.");
    eprintln!("If node 1 IS root's child and root owned by P1: expected = 1.0 → BOTH WRONG.");
    eprintln!("If node 1 NOT root's child: need to trace path to root for verdict.");
}
