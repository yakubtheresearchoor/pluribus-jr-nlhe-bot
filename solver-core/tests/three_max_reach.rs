/// 3-player reach comparison: compare GPU vs CPU reach after top-down pass.
use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{FlopStartVectorCfr, Zone};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn find_pair_index(c1: Card, c2: Card) -> u16 {
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (a, b) = index_to_card_pair(idx);
        if (a == c1 as u8 && b == c2 as u8) || (a == c2 as u8 && b == c1 as u8) {
            return idx as u16;
        }
    }
    panic!("pair not found")
}

fn build_3player_table() -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
    let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

    let chosen_hands: Vec<u16> = vec![
        find_pair_index(card_from_str("Ah").unwrap(), card_from_str("Qc").unwrap()),
        find_pair_index(card_from_str("Jd").unwrap(), card_from_str("Ts").unwrap()),
        find_pair_index(card_from_str("9h").unwrap(), card_from_str("8c").unwrap()),
        find_pair_index(card_from_str("As").unwrap(), card_from_str("Qd").unwrap()),
        find_pair_index(card_from_str("Jc").unwrap(), card_from_str("Th").unwrap()),
        find_pair_index(card_from_str("9s").unwrap(), card_from_str("8d").unwrap()),
    ];

    let nh = chosen_hands.len();
    let num_players = 3u8;
    let num_opp = 2;
    let valid_hand_indices = chosen_hands.clone();
    let num_valid = nh;

    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in valid_hand_indices.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
    }

    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh {
        for j in 0..nh {
            if i == j { conflict[i * nh + j] = 1; continue; }
            let (c1a, c1b) = index_to_card_pair(valid_hand_indices[i] as usize);
            let (c2a, c2b) = index_to_card_pair(valid_hand_indices[j] as usize);
            if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b {
                conflict[i * nh + j] = 1;
            }
        }
    }

    let mut hand_ranks_base = vec![0u16; nh];
    for (i, &hi) in valid_hand_indices.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut hand = Hand::new();
        hand = hand.add_card(c1 as usize);
        hand = hand.add_card(c2 as usize);
        for &bc in &board { hand = hand.add_card(bc as usize); }
        hand_ranks_base[i] = hand.evaluate_internal() as u16;
    }

    let turn_cards: Vec<u8> = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("4c").unwrap() as u8,
    ];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("5c").unwrap() as u8,
        card_from_str("6c").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("5c").unwrap() as u8,
    ];

    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 { continue; }
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in &board { hand = hand.add_card(bc as usize); }
            hand = hand.add_card(tc as usize);
            turn_ranks[tc as usize * nh + i] = hand.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh)
            .map(|h| (turn_ranks[tc as usize * nh + h] + 1, h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            let off = tc as usize * num_opp * nh + oi * nh;
            for h in 0..nh {
                turn_sorted_str[off + h] = items[h].0;
                turn_sorted_idx[off + h] = items[h].1;
            }
        }
    }

    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for &rc in &river_decks[tc as usize] {
            let full_mask = turn_mask | (1u64 << rc);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if full_mask & (1u64 << c1) != 0 || full_mask & (1u64 << c2) != 0 { continue; }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in &board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize);
                hand = hand.add_card(rc as usize);
                river_ranks[tc as usize * 52 * nh + rc as usize * nh + i] =
                    hand.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| (river_ranks[tc as usize * 52 * nh + rc as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = tc as usize * 52 * num_opp * nh + rc as usize * num_opp * nh + oi * nh;
                for h in 0..nh {
                    river_sorted_str[off + h] = items[h].0;
                    river_sorted_idx[off + h] = items[h].1;
                }
            }
        }
    }

    let initial_weights = vec![vec![1.0f32; nh]; num_players as usize];
    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let mask0: u64 = (1u64 << hand_cards[h0 * 2]) | (1u64 << hand_cards[h0 * 2 + 1]);
        for h1 in 0..nh {
            let mask1: u64 = (1u64 << hand_cards[h1 * 2]) | (1u64 << hand_cards[h1 * 2 + 1]);
            if mask0 & mask1 != 0 { continue; }
            for h2 in 0..nh {
                let mask2: u64 = (1u64 << hand_cards[h2 * 2]) | (1u64 << hand_cards[h2 * 2 + 1]);
                if mask0 & mask2 != 0 || mask1 & mask2 != 0 { continue; }
                nc += 1.0;
            }
        }
    }

    let table = FlopChanceTable {
        hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
        remaining_deck: turn_cards.clone(), turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx, initial_weights, num_players,
        num_combinations: nc, river_decks,
    };
    let config = TreeConfig {
        num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![100, 100, 100], initial_contributions: vec![5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

#[test]
fn test_3player_flop_reach() {
    let (tree, table) = build_3player_table();
    let game = FlopStartGame::new(FlopChanceTable {
        hand_ranks_base: table.hand_ranks_base.clone(),
        valid_hand_indices: table.valid_hand_indices.clone(),
        num_valid: table.num_valid,
        conflict: table.conflict.clone(),
        hand_cards: table.hand_cards.clone(),
        remaining_deck: table.remaining_deck.clone(),
        turn_ranks: table.turn_ranks.clone(),
        turn_sorted_str: table.turn_sorted_str.clone(),
        turn_sorted_idx: table.turn_sorted_idx.clone(),
        river_ranks: table.river_ranks.clone(),
        river_sorted_str: table.river_sorted_str.clone(),
        river_sorted_idx: table.river_sorted_idx.clone(),
        initial_weights: table.initial_weights.clone(),
        num_players: table.num_players,
        num_combinations: table.num_combinations,
        river_decks: table.river_decks.clone(),
    });
    let nh = table.num_valid;
    let np = 3usize;
    let nn = tree.nodes.len();

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.set_iteration(0);
    
    // Compute strategies from zero regrets (should be uniform)
    cpu.compute_all_strategies(&tree);
    
    // Compute CPU flop reach
    let cpu_reach = cpu.compute_reach_flop(&tree, &game);
    
    // Create GPU solver
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    
    // Compute GPU strategies and reach
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    
    // Download GPU reach
    let gpu_reach = gpu.download_reach();
    
    // Compare reach values at specific nodes
    // Root node = node 0: reach should be initial weights
    // Layout: [node_id * np * nh + player * nh + hand]
    eprintln!("CPU reach at root (P0): {:?}", &cpu_reach[0..nh]);
    eprintln!("GPU reach at root (P0): {:?}", &gpu_reach[0..nh]);
    
    // The CPU's reach at root should be initial weights: [1,1,1,1,1,1]
    // The GPU's reach at root should also be [1,1,1,1,1,1]
    
    // Compare at node 0 for all 3 players
    for p in 0..np {
        let mut max_diff = 0.0f32;
        for h in 0..nh {
            let cpu_off = p * nh + h;
            let gpu_off = p * nh + h;
            let diff = (cpu_reach[cpu_off] - gpu_reach[gpu_off]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("P{} root reach max_diff: {:.6}", p, max_diff);
    }
    
    // Check child of root (node 1 = check, node 2 = bet)
    // After root's uniform strategy [0.5, 0.5], each child's reach for player 0 should be 0.5 * initial
    // For player 1 and 2, reach stays at initial (they don't act yet)
    let child0 = tree.nodes[0].children_start as usize; // first child (check)
    let child1 = child0 + 1; // second child (bet)
    
    eprintln!("Root children: {} and {}", child0, child1);
    
    // Check a few nodes for comparison
    let mut max_diff_overall = 0.0f32;
    let mut max_diff_node = 0;
    let mut max_diff_details = String::new();
    
    for node_id in 0..nn.min(30) {
        for p in 0..np {
            for h in 0..nh {
                let cpu_off = node_id * np * nh + p * nh + h;
                let gpu_off = node_id * np * nh + p * nh + h;
                let diff = (cpu_reach[cpu_off] - gpu_reach[gpu_off]).abs();
                if diff > max_diff_overall {
                    max_diff_overall = diff;
                    max_diff_node = node_id;
                    max_diff_details = format!("P{} H{}: CPU={:.6} GPU={:.6}", p, h, cpu_reach[cpu_off], gpu_reach[gpu_off]);
                }
            }
        }
    }
    
    eprintln!("Max reach diff in first {} nodes: {:.6} at node {} ({})", 
              nn.min(30), max_diff_overall, max_diff_node, max_diff_details);
    
    // Also check turn reach for ti=0
    gpu.compute_reach_turn(&ctx, 0);
    let gpu_turn_reach = gpu.download_turn_reach();
    
    // CPU turn reach
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 0, &cpu_reach);
    
    // Compare turn reach for first few nodes
    let mut turn_max_diff = 0.0f32;
    let mut turn_diff_count = 0usize;
    for node_id in 0..nn {
        for p in 0..np {
            for h in 0..nh {
                let off = node_id * np * nh + p * nh + h;
                let diff = (cpu_turn_reach[off] - gpu_turn_reach[off]).abs();
                if diff > 0.0001 {
                    turn_max_diff = turn_max_diff.max(diff);
                    turn_diff_count += 1;
                }
            }
        }
    }
    eprintln!("Turn reach: {} nodes differ, max_diff={:.6}", turn_diff_count, turn_max_diff);
    
    // Debug: check what the turn chance children are and their reach values
    // Turn chance children are the first nodes of the turn zone
    let turn_cc = cpu.turn_chance_children();
    eprintln!("Turn chance children: {} nodes", turn_cc.len());
    for (i, &cc) in turn_cc.iter().enumerate().take(5) {
        let cc = cc as usize;
        // Check reach at this node in d_reach (flop reach)
        let flop_p0: Vec<f32> = (0..nh).map(|h| cpu_reach[cc * np * nh + 0 * nh + h]).collect();
        let gpu_p0: Vec<f32> = (0..nh).map(|h| gpu_reach[cc * np * nh + 0 * nh + h]).collect();
        // Find parent
        let mut parent = None;
        for nid in 0..nn {
            let n = &tree.nodes[nid];
            if n.is_chance() {
                for &ch in tree.node_children(nid) {
                    if ch as usize == cc { parent = Some(nid); break; }
                }
            }
            if parent.is_some() { break; }
        }
        let zone_str = match cpu.zones()[cc] { Zone::Flop => "Flop", Zone::Turn => "Turn", Zone::River => "River", Zone::Preflop => "Preflop" };
        eprintln!("  Turn CC[{}] = node {}: zone={}, parent={}, CPU P0 reach = {:?}, GPU P0 reach = {:?}", 
                  i, cc, zone_str, parent.unwrap_or(999), 
                  &flop_p0[..nh.min(3)], &gpu_p0[..nh.min(3)]);
        if let Some(p) = parent {
            let parent_zone_str = match cpu.zones()[p] { Zone::Flop => "Flop", Zone::Turn => "Turn", Zone::River => "River", Zone::Preflop => "Preflop" };
            let parent_p0: Vec<f32> = (0..nh).map(|h| cpu_reach[p * np * nh + 0 * nh + h]).collect();
            let gpu_parent_p0: Vec<f32> = (0..nh).map(|h| gpu_reach[p * np * nh + 0 * nh + h]).collect();
            eprintln!("    parent node {} zone={} CPU P0 reach = {:?} GPU P0 reach = {:?}", p, parent_zone_str, &parent_p0[..nh.min(3)], &gpu_parent_p0[..nh.min(3)]);
            
            // Find grandparent (parent of the turn chance node)
            let mut gp = None;
            for nid in 0..nn {
                let n = &tree.nodes[nid];
                for &ch in tree.node_children(nid) {
                    if ch as usize == p { gp = Some(nid); break; }
                }
                if gp.is_some() { break; }
            }
            if let Some(g) = gp {
                let gp_zone_str = match cpu.zones()[g] { Zone::Flop => "Flop", Zone::Turn => "Turn", Zone::River => "River", Zone::Preflop => "Preflop" };
                let gp_p0: Vec<f32> = (0..nh).map(|h| cpu_reach[g * np * nh + 0 * nh + h]).collect();
                eprintln!("    grandparent node {} zone={} CPU P0 reach = {:?}", g, gp_zone_str, &gp_p0[..nh.min(3)]);
            }
        }
    }
    
    // Check river reach for ti=0, ri=0
    gpu.compute_reach_river(&ctx, 0, 0);
    let gpu_river_reach = gpu.download_river_reach();
    
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 0, &cpu_turn_reach);
    
    let mut river_max_diff = 0.0f32;
    let mut river_diff_count = 0usize;
    for node_id in 0..nn {
        for p in 0..np {
            for h in 0..nh {
                let off = node_id * np * nh + p * nh + h;
                let diff = (cpu_river_reach[off] - gpu_river_reach[off]).abs();
                if diff > 0.0001 {
                    river_max_diff = river_max_diff.max(diff);
                    river_diff_count += 1;
                }
            }
        }
    }
    eprintln!("River reach: {} nodes differ, max_diff={:.6}", river_diff_count, river_max_diff);

    // CPU↔Metal reach parity assertions on all three zones.
    //
    // Prior version of this test had ~130 lines of debug instrumentation
    // (hardcoded tree.nodes[1241] etc.) after this point that was stale
    // from when the tree was larger; the test was broken by my Zone::Preflop
    // addition exposing non-exhaustive match arms in those debug prints,
    // and the hardcoded node indices then panicked on the smaller current
    // tree. The actual test logic (the three reach comparisons above) was
    // never reaching the assertion. Replaced with proper assertions on all
    // three reach diffs (flop / turn / river) at the f32 floor tolerance
    // appropriate for iter-0 reach (~uniform-strategy product propagation).
    //
    // Tolerance rationale: max_diff_overall threshold was 0.01 in the
    // original assertion. Reach values at iter-0 are products of uniform
    // strategy probs × initial weight 1.0, so per-node reach is bounded
    // by 1.0; a diff > 0.01 would represent a 1% reach divergence which
    // is far above f32 floor for the small number of multiplications
    // involved. Keeping the 0.01 tolerance for flop, and applying the
    // same tolerance to turn/river.
    let flop_tol = 0.01_f32;
    let turn_tol = 0.01_f32;
    let river_tol = 0.01_f32;
    assert!(max_diff_overall < flop_tol,
        "CPU↔Metal FLOP reach diff = {:.6} > {} tolerance. \
         3-player flop-reach computation diverges between CPU and Metal — \
         likely a bug in compute_reach_flop on either side, or a tree-builder \
         issue causing different reach propagation. This sits in an anchored \
         multiway layer next to active rake/preflop work, so investigate \
         before assuming benign.",
        max_diff_overall, flop_tol);
    assert!(turn_max_diff < turn_tol,
        "CPU↔Metal TURN reach diff = {:.6} > {} tolerance (ti=0). Similar \
         risk profile to the flop assertion above.",
        turn_max_diff, turn_tol);
    assert!(river_max_diff < river_tol,
        "CPU↔Metal RIVER reach diff = {:.6} > {} tolerance (ti=0, ri=0). \
         Similar risk profile to the flop assertion above.",
        river_max_diff, river_tol);

    eprintln!("✓ 3p CPU↔Metal reach parity: flop {:.6} | turn {:.6} | river {:.6} \
        (all < 0.01)",
        max_diff_overall, turn_max_diff, river_max_diff);
}
