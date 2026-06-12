// Step 2.D.14 (#105) PROBE: find a hand-traceable tiny config for the
// freeze+extract rules anchor.
//
// Goal: identify the smallest HU flop config that
//   1. exercises all three zones (Flop, Turn, River)
//   2. has few enough player infosets to hand-derive σ_avg at K=1 (uniform)
//      and K=2 (non-uniform)
//   3. supports a subset deck (1 turn × 1 river per turn) and nh=2
//
// This is throw-away scaffolding — it prints tree structure and per-zone
// player-infoset counts so we can pick the right TreeConfig before
// writing the main anchor test.

use solver_core::card::{card_from_str, card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{FlopStartVectorCfr, Zone};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

use solver_core::tree::flat::{NODE_TYPE_PLAYER, NODE_TYPE_CHANCE, NODE_TYPE_TERMINAL};

fn build_tree_with_stacks_and_bets(stacks: i32, with_bets: bool) -> FlatTree {
    let (bet, raise) = if with_bets {
        (vec![BetSize::PotRelative(1.0)], vec![BetSize::PotRelative(1.0)])
    } else {
        (vec![], vec![])
    };
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 2,
        starting_stacks: vec![stacks, stacks],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet, raise },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&cfg).expect("tree builds")
}

fn build_tree_with_stacks(stacks: i32) -> FlatTree {
    build_tree_with_stacks_and_bets(stacks, true)
}

#[test]
#[ignore = "Step 2.D.14 probe: print tiny tree structures to pick a hand-traceable config"]
fn step2d14_probe_tree_sizes() {
    eprintln!("\n=== Step 2.D.14 PROBE: tree sizes vs stacks ===\n");
    eprintln!("--- WITH BETS (PotRelative 1.0) ---");
    for stacks in &[1, 2, 4, 6, 10] {
        let tree = build_tree_with_stacks_and_bets(*stacks, true);
        let mut np = 0; let mut nc = 0; let mut nt = 0;
        for n in &tree.nodes {
            match n.node_type {
                NODE_TYPE_PLAYER => np += 1,
                NODE_TYPE_CHANCE => nc += 1,
                NODE_TYPE_TERMINAL => nt += 1,
                _ => {}
            }
        }
        eprintln!("  stacks={:3}  → {:4} nodes ({} player, {} chance, {} terminal), max_depth={}",
            stacks, tree.num_nodes(), np, nc, nt, tree.max_depth);
    }
    eprintln!("\n--- NO BETS (check-only) ---");
    for stacks in &[1, 2, 10] {
        let tree = build_tree_with_stacks_and_bets(*stacks, false);
        let mut np = 0; let mut nc = 0; let mut nt = 0;
        for n in &tree.nodes {
            match n.node_type {
                NODE_TYPE_PLAYER => np += 1,
                NODE_TYPE_CHANCE => nc += 1,
                NODE_TYPE_TERMINAL => nt += 1,
                _ => {}
            }
        }
        eprintln!("  stacks={:3}  → {:4} nodes ({} player, {} chance, {} terminal), max_depth={}",
            stacks, tree.num_nodes(), np, nc, nt, tree.max_depth);
    }

    eprintln!("\n=== legacy block (now no-op) ===\n");
    for stacks in &[1] {
        let tree = build_tree_with_stacks(*stacks);
        let mut n_player = 0;
        let mut n_chance = 0;
        let mut n_terminal = 0;
        for n in &tree.nodes {
            match n.node_type {
                NODE_TYPE_PLAYER => n_player += 1,
                NODE_TYPE_CHANCE => n_chance += 1,
                NODE_TYPE_TERMINAL => n_terminal += 1,
                _ => {}
            }
        }
        eprintln!("  stacks={:3}  → {:4} nodes ({} player, {} chance, {} terminal), max_depth={}",
            stacks, tree.num_nodes(), n_player, n_chance, n_terminal, tree.max_depth);
    }

    // Build the smallest one and inspect player-infoset counts per zone.
    eprintln!("\n=== Smallest tree (stacks=1) — per-zone player infoset counts ===");
    let tree = build_tree_with_stacks(1);

    // Build a tiny FlopChanceTable to seed FlopStartVectorCfr so we can read its
    // zone classification.
    let canonical: [Card; 3] = [
        card_from_str("As").unwrap(),
        card_from_str("Kh").unwrap(),
        card_from_str("Qd").unwrap(),
    ];
    let board: Vec<Card> = canonical.iter().copied().collect();
    let mut ranges: Vec<Vec<f32>> = vec![vec![0.0f32; NUM_POSSIBLE_HANDS]; 2];
    // Two non-blocking hands: 2c2d and 3c3d (no board overlap with AKQ).
    let hands = [
        (card_from_str("2c").unwrap(), card_from_str("2d").unwrap()),
        (card_from_str("3c").unwrap(), card_from_str("3d").unwrap()),
    ];
    for &(c1, c2) in &hands {
        for p in 0..2 { ranges[p][card_pair_to_index(c1, c2)] = 1.0; }
    }
    // Subset deck: 1 turn × 1 river.
    let chosen: Vec<u16> = hands.iter()
        .map(|&(c1, c2)| card_pair_to_index(c1, c2) as u16)
        .collect();
    let turn_cards: Vec<u8> = vec![card_from_str("4c").unwrap() as u8];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![card_from_str("5c").unwrap() as u8];

    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, 2, &chosen, &turn_cards, &river_decks);
    let nh = table.num_valid;
    eprintln!("  nh = {} (after subset construction)", nh);
    eprintln!("  hand_cards: {:?}", &table.hand_cards[..nh * 2]);

    let game = FlopStartGame::new(table);
    let solver = FlopStartVectorCfr::new(&tree, game.table());
    let np = 2;

    let mut flop_player = 0; let mut flop_chance = 0; let mut flop_term = 0;
    let mut turn_player = 0; let mut turn_chance = 0; let mut turn_term = 0;
    let mut river_player = 0; let mut river_chance = 0; let mut river_term = 0;
    for (idx, n) in tree.nodes.iter().enumerate() {
        let zone_label = match solver.zones()[idx] {
            Zone::Flop => "Flop",
            Zone::Turn => "Turn",
            Zone::River => "River",
            Zone::Preflop => "Preflop",
        };
        let type_label = match n.node_type {
            NODE_TYPE_PLAYER => "P",
            NODE_TYPE_CHANCE => "C",
            NODE_TYPE_TERMINAL => "T",
            _ => "?",
        };
        match (solver.zones()[idx], n.node_type) {
            (Zone::Flop, NODE_TYPE_PLAYER) => flop_player += 1,
            (Zone::Flop, NODE_TYPE_CHANCE) => flop_chance += 1,
            (Zone::Flop, NODE_TYPE_TERMINAL) => flop_term += 1,
            (Zone::Turn, NODE_TYPE_PLAYER) => turn_player += 1,
            (Zone::Turn, NODE_TYPE_CHANCE) => turn_chance += 1,
            (Zone::Turn, NODE_TYPE_TERMINAL) => turn_term += 1,
            (Zone::River, NODE_TYPE_PLAYER) => river_player += 1,
            (Zone::River, NODE_TYPE_CHANCE) => river_chance += 1,
            (Zone::River, NODE_TYPE_TERMINAL) => river_term += 1,
            _ => {}
        };
        if idx < 30 {  // print first 30 nodes
            let na = n.num_children as usize;
            eprintln!("    node {:3} [{} / {}] player={} na={} children_start={}",
                idx, zone_label, type_label, n.player_id, na, n.children_start);
        }
    }
    eprintln!("\n  Zone breakdown:");
    eprintln!("    Flop:  {} player, {} chance, {} term", flop_player, flop_chance, flop_term);
    eprintln!("    Turn:  {} player, {} chance, {} term", turn_player, turn_chance, turn_term);
    eprintln!("    River: {} player, {} chance, {} term", river_player, river_chance, river_term);
    eprintln!("    Total decision infosets: {}", flop_player + turn_player + river_player);
    let _ = np;
}
