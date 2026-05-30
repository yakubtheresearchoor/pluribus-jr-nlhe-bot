/// 3-player Metal vs CPU parity test.
use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::showdown::side_pot_showdown_cfv;
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
    panic!("pair not found");
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
    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

#[test]
fn test_3player_sidepot_cfv_reference() {
    // Directly test side_pot_showdown_cfv for 3 players with known inputs.
    // This isolates the N>2 CFV formula from the pipeline.
    let (tree, table) = build_3player_table();
    let nh = table.num_valid;
    let hand_cards = &table.hand_cards;
    
    // Get base sorted arrays
    let (os, oi, ps, pi, _) = table.sorted_opp_arrays_base();

    // Test 1: 3-player equal contribution, no folds, uniform reach
    let opp_reach_0: Vec<f32> = vec![1.0; nh];
    let opp_reach_1: Vec<f32> = vec![1.0; nh];
    let opp_reach: Vec<&[f32]> = vec![&opp_reach_0, &opp_reach_1];
    let contributions = vec![5i32, 5, 5];
    let fold_mask: u16 = 0;
    let traverser: usize = 0;
    let starting_pot: i32 = 15;

    let cfv = side_pot_showdown_cfv(
        &opp_reach, hand_cards, nh,
        &os, &oi, &ps, &pi,
        &contributions, fold_mask, traverser, 3u8,
        starting_pot,
    );

    eprintln!("3-player equal contrib CFV (not normalized):");
    for h in 0..nh {
        eprintln!("  hand {}: cfv = {:.4}", h, cfv[h]);
    }

    // With uniform reach and pairwise non-conflicting hands:
    // half_pot = 15/3 + 5 = 10
    // For each hand h, both opponents have:
    //   cfreach_sum = 6, no cards conflict between any two hands in the pool
    //   So for opponent oi: cfreach_minus[c] = 0 for all c (no card shared between different hands)
    //   Wait, that's wrong. cfreach_minus accumulates the reach of hands containing card c.
    //   Since hand 0 has cards [12,36], cfreach_minus[12] = reach[0] = 1 for opponent oi.
    //   But this is for the TRAVERSER's cards, not the opponent's.
    //   Actually no: cfreach_minus is computed during the sweep. It tracks the total reach
    //   of hands already processed that contain each card. The cfreach formula is:
    //     cfreach = cfreach_sum - cfreach_minus[h_c1] - cfreach_minus[h_c2]
    //   where h_c1, h_c2 are the CARDS of the current hand h (the TRAVERSER's hand).
    //   So it subtracts the reach of hands that share a card with the traverser's hand.
    
    // Since no two hands in the pool share a card, cfreach_minus only has non-zero
    // entries for cards that appear in hand h itself. So for hand h:
    //   cfreach_minus[h_c1] = reach[h] (the one hand containing card h_c1)
    //   cfreach_minus[h_c2] = reach[h] (the one hand containing card h_c2)
    //   BUT WAIT: h_c1 and h_c2 are in the SAME hand h, so cfreach_minus counts reach[h] twice!
    //   cfreach = cfreach_sum - reach[h] - reach[h] = 6 - 1 - 1 = 4
    //   eff = cfreach + reach[h] = 4 + 1 = 5
    
    // For cum_weaker: the sweep processes weaker hands. Each weaker hand has no shared
    // cards with hand h (since all hands are pairwise non-conflicting). So:
    //   cum_weaker[h] = number of strictly weaker hands
    
    // Sorted order (by rank ascending): 5,2,4,1,3,0 (weakest to strongest)
    // cum_weaker for each hand:
    //   hand 0 (strongest): 5 weaker hands -> cum_weaker = 5
    //   hand 3: 4 weaker -> cum_weaker = 4  
    //   hand 1: 3 weaker -> cum_weaker = 3
    //   hand 4: 2 weaker -> cum_weaker = 2
    //   hand 2: 1 weaker -> cum_weaker = 1
    //   hand 5 (weakest): 0 weaker -> cum_weaker = 0
    
    // With 2 opponents (same reach), product formula:
    //   cfv[h] = 10 * (3 * cum_weaker[h]^2 - 5^2)
    //         = 10 * (3 * cum_weaker[h]^2 - 25)
    
    // hand 0: cfv = 10 * (3*25 - 25) = 10 * 50 = 500
    // hand 3: cfv = 10 * (3*16 - 25) = 10 * 23 = 230
    // hand 1: cfv = 10 * (3*9 - 25) = 10 * 2 = 20
    // hand 4: cfv = 10 * (3*4 - 25) = 10 * (-13) = -130
    // hand 2: cfv = 10 * (3*1 - 25) = 10 * (-22) = -220
    // hand 5: cfv = 10 * (3*0 - 25) = 10 * (-25) = -250
    
    // But these are NOT divided by num_combinations yet.
    // num_combinations = 120 (6*5*4)
    
    let nc = table.num_combinations as f32;
    eprintln!("num_combinations = {}", nc);
    
    // Verify CFVs are non-trivial and have the right sign structure
    // (best hand positive, worst hand negative)
    let raw: Vec<f32> = cfv.iter().map(|v| v * nc).collect();
    eprintln!("  raw CFVs: {:?}", raw);
    assert!(raw[0] > 0.0, "hand 0 (AhQc) should have positive CFV, got {}", raw[0]);
    assert!(raw[5] < 0.0, "hand 5 (9s8d) should have negative CFV, got {}", raw[5]);
}

#[test]
fn test_3player_iter0_parity() {
    let (tree, table) = build_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;

    // Create CPU solver FIRST (GPU solver needs it for construction)
    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.set_iteration(0);

    // Create GPU solver
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Run one iteration on CPU
    cpu.run(&tree, &game, 1);

    // Run one iteration on GPU
    gpu.run(&ctx, &tree, &game, 1);

    // Compare regrets per zone
    let cpu_flop = cpu.regrets_flop();
    let cpu_turn = cpu.regrets_turn();
    let cpu_river = cpu.regrets_river();
    let gpu_all = gpu.download_regrets();

    let fl = cpu_flop.len();
    let tl = cpu_turn.len();
    let rl = cpu_river.len();
    eprintln!("Zone lengths: flop={} turn={} river={}", fl, tl, rl);

    let mut zone_max = [0.0f32; 3];
    let mut zone_count = [0usize; 3];
    for zone_idx in 0..3 {
        let (cpu_slice, gpu_start) = match zone_idx {
            0 => (cpu_flop, 0),
            1 => (cpu_turn, fl),
            _ => (cpu_river, fl + tl),
        };
        for i in 0..cpu_slice.len() {
            let gi = gpu_start + i;
            if gi >= gpu_all.len() { break; }
            let diff = (cpu_slice[i] - gpu_all[gi]).abs();
            if cpu_slice[i] != 0.0 || gpu_all[gi] != 0.0 {
                zone_max[zone_idx] = zone_max[zone_idx].max(diff);
                zone_count[zone_idx] += 1;
            }
        }
    }
    eprintln!("Flop zone: max_diff={:.6} ({} nonzero)", zone_max[0], zone_count[0]);
    eprintln!("Turn zone: max_diff={:.6} ({} nonzero)", zone_max[1], zone_count[1]);
    eprintln!("River zone: max_diff={:.6} ({} nonzero)", zone_max[2], zone_count[2]);

    let max_diff = zone_max[0].max(zone_max[1]).max(zone_max[2]);
    eprintln!("Overall max_diff={:.6}", max_diff);

    assert!(max_diff < 0.1, "3-player iter-0 max regret diff = {:.4} (expected < 0.1)", max_diff);
}

#[test]
fn test_3player_convergence() {
    let (tree, table) = build_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;

    let mut cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());

    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);

    // Run 10 iterations on GPU (start small)
    gpu.run(&ctx, &tree, &game, 10);

    // Measure GPU exploitability via CPU proxy
    cpu_proxy.set_iteration(gpu.iteration());
    let gpu_reg = gpu.download_regrets();
    let gpu_cum = gpu.download_cum_strategy();
    let fl = cpu_proxy.regrets_flop().len();
    let tl = cpu_proxy.regrets_turn().len();
    {
        let cpu_reg_flop = cpu_proxy.regrets_flop_mut();
        for i in 0..fl { cpu_reg_flop[i] = gpu_reg[i]; }
    }
    {
        let cpu_reg_turn = cpu_proxy.regrets_turn_mut();
        for i in 0..tl { cpu_reg_turn[i] = gpu_reg[fl + i]; }
    }
    {
        let cpu_reg_river = cpu_proxy.regrets_river_mut();
        for i in 0..cpu_reg_river.len() {
            if fl + tl + i < gpu_reg.len() {
                cpu_reg_river[i] = gpu_reg[fl + tl + i];
            }
        }
    }
    // Also upload cum_strategy
    {
        let cpu_cum_flop = cpu_proxy.cum_strategy_flop_mut();
        for i in 0..fl { cpu_cum_flop[i] = gpu_cum[i]; }
    }
    {
        let cpu_cum_turn = cpu_proxy.cum_strategy_turn_mut();
        for i in 0..tl { cpu_cum_turn[i] = gpu_cum[fl + i]; }
    }
    {
        let cpu_cum_river = cpu_proxy.cum_strategy_river_mut();
        for i in 0..cpu_cum_river.len() {
            if fl + tl + i < gpu_cum.len() {
                cpu_cum_river[i] = gpu_cum[fl + tl + i];
            }
        }
    }

    let mut total_expl = 0.0f32;
    for p in 0..np {
        let br = cpu_proxy.best_response_value_debug(&tree, &game, p as u8);
        let sv = cpu_proxy.strategy_value_debug(&tree, &game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total_expl += (br[h] - sv[h]).max(0.0);
        }
    }
    let pot_size = 15.0f32;
    let expl_pct = total_expl / pot_size * 100.0;
    eprintln!("GPU after 100 iters: expl={:.4} ({:.2}% of pot)", total_expl, expl_pct);

    assert!(expl_pct < 1000.0, "3-player GPU exploitability = {:.2}% of pot (expected < 1000% at 10 iters)", expl_pct);
}
