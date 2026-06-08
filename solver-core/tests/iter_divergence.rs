/// Iter-by-iter parity: find the first iteration where GPU and CPU diverge.
use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::gpu_metal::flop_solver::DcfrParams;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

const NUM_HANDS: usize = 10;

fn build_table() -> (solver_core::tree::flat::FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 3u8;
    let num_opp = 2;
    let nh = NUM_HANDS;

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen_hands: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();

    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen_hands.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
    }

    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh {
        for j in 0..nh {
            if i == j { conflict[i * nh + j] = 1; continue; }
            let (c1a, c1b) = index_to_card_pair(chosen_hands[i] as usize);
            let (c2a, c2b) = index_to_card_pair(chosen_hands[j] as usize);
            if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b { conflict[i * nh + j] = 1; }
        }
    }

    let mut hand_ranks_base = vec![0u16; nh];
    for (i, &hi) in chosen_hands.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut hand = Hand::new();
        hand = hand.add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { hand = hand.add_card(bc as usize); }
        hand_ranks_base[i] = hand.evaluate_internal() as u16;
    }

    let turn_cards = vec![card_from_str("3c").unwrap() as u8];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![card_from_str("5s").unwrap() as u8];

    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &tc in &turn_cards {
        for (i, &hi) in chosen_hands.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let tm = board_mask | (1u64 << tc);
            if tm & (1u64 << c1) != 0 || tm & (1u64 << c2) != 0 { continue; }
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize).add_card(c2 as usize);
            for &bc in &board { hand = hand.add_card(bc as usize); }
            hand = hand.add_card(tc as usize);
            turn_ranks[tc as usize * nh + i] = hand.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (turn_ranks[tc as usize * nh + h] + 1, h as u16)).collect();
        items.sort_by_key(|&(s,_)| s);
        for oi in 0..num_opp {
            let off = tc as usize * num_opp * nh + oi * nh;
            for h in 0..nh { turn_sorted_str[off + h] = items[h].0; turn_sorted_idx[off + h] = items[h].1; }
        }
    }

    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        let tm = board_mask | (1u64 << tc);
        for &rc in &river_decks[tc as usize] {
            let fm = tm | (1u64 << rc);
            for (i, &hi) in chosen_hands.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if fm & (1u64 << c1) != 0 || fm & (1u64 << c2) != 0 { continue; }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize).add_card(c2 as usize);
                for &bc in &board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize).add_card(rc as usize);
                river_ranks[tc as usize * 52 * nh + rc as usize * nh + i] = hand.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (river_ranks[tc as usize * 52 * nh + rc as usize * nh + h] + 1, h as u16)).collect();
            items.sort_by_key(|&(s,_)| s);
            for oi in 0..num_opp {
                let off = tc as usize * 52 * num_opp * nh + rc as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }

    let initial_weights = vec![vec![1.0f32; nh]; num_players as usize];
    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let m0: u64 = (1u64 << hand_cards[h0*2]) | (1u64 << hand_cards[h0*2+1]);
        for h1 in 0..nh {
            if h0 == h1 { continue; }
            let m1: u64 = (1u64 << hand_cards[h1*2]) | (1u64 << hand_cards[h1*2+1]);
            if m0 & m1 != 0 { continue; }
            for h2 in 0..nh {
                if h2 == h0 || h2 == h1 { continue; }
                let m2: u64 = (1u64 << hand_cards[h2*2]) | (1u64 << hand_cards[h2*2+1]);
                if m0 & m2 != 0 || m1 & m2 != 0 { continue; }
                nc += 1.0;
            }
        }
    }

    let table = FlopChanceTable {
        hand_ranks_base, valid_hand_indices: chosen_hands, num_valid: nh, conflict, hand_cards,
        remaining_deck: turn_cards, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights, num_players, num_combinations: nc, river_decks,
    };
    let config = TreeConfig {
        num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![100, 100, 100], initial_contributions: vec![5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,

    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

#[test]
fn iter_by_iter_divergence() {
    let (tree, table) = build_table();
    let game = FlopStartGame::new(table);
    let nh = NUM_HANDS;

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().unwrap();
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    for iter in 0..10 {
        // Before running, compute strategies on both and compare
        if iter > 0 {
            cpu.compute_all_strategies(&tree);
            gpu.compute_all_strategies(&ctx);
            let cpu_sf = cpu.strategy_flop();
            let gpu_s = gpu.download_strategy();
            let mut strat_max = 0.0f32;
            let mut strat_diverged = 0;
            for i in 0..cpu_sf.len() {
                let d = (cpu_sf[i] - gpu_s[i]).abs();
                strat_max = strat_max.max(d);
                if d > 0.001 { strat_diverged += 1; }
            }
            eprintln!("  PRE-ITER {} strategy flop: max_diff={:.8} ({} diverged)", iter + 1, strat_max, strat_diverged);
            if strat_max > 0.001 {
                for i in 0..cpu_sf.len().min(40) {
                    let d = (cpu_sf[i] - gpu_s[i]).abs();
                    if d > 0.001 {
                        let hand = i % NUM_HANDS;
                        let action = (i / NUM_HANDS) % 4;
                        let node_local = i / (4 * NUM_HANDS);
                        eprintln!("    strat[{}] (n={} a={} h={}): cpu={:.6} gpu={:.6}", i, node_local, action, hand, cpu_sf[i], gpu_s[i]);
                    }
                }
            }
        }

        // Run one iteration on CPU, then one on GPU
        // But on iter 2, compare CFVs mid-computation to find where divergence starts
        cpu.run(&tree, &game, 1);
        gpu.run(&ctx, &tree, &game, 1);

        let cpu_flop = cpu.regrets_flop();
        let cpu_turn = cpu.regrets_turn();
        let cpu_river = cpu.regrets_river();
        let gpu_all = gpu.download_regrets();

        // Also compare strategies
        let cpu_strat_flop = cpu.strategy_flop();
        let gpu_strat = gpu.download_strategy();
        let mut strat_max = 0.0f32;
        for i in 0..cpu_strat_flop.len() {
            strat_max = strat_max.max((cpu_strat_flop[i] - gpu_strat[i]).abs());
        }
        eprintln!("  strategy flop max_diff = {:.8}", strat_max);

        let fl = cpu_flop.len();
        let tl = cpu_turn.len();

        let mut flop_max = 0.0f32;
        for i in 0..fl { flop_max = flop_max.max((cpu_flop[i] - gpu_all[i]).abs()); }
        let mut turn_max = 0.0f32;
        for i in 0..tl { turn_max = turn_max.max((cpu_turn[i] - gpu_all[fl + i]).abs()); }
        let mut river_max = 0.0f32;
        for i in 0..cpu_river.len() {
            if fl + tl + i < gpu_all.len() {
                river_max = river_max.max((cpu_river[i] - gpu_all[fl + tl + i]).abs());
            }
        }

        let max_diff = flop_max.max(turn_max).max(river_max);
        eprintln!("iter {:2}: max_diff = {:.8}  (flop={:.8} turn={:.8} river={:.8})",
            iter + 1, max_diff, flop_max, turn_max, river_max);

        // Print some regret values for diagnostics
        if iter <= 1 {
            eprintln!("  First 20 flop regrets:");
            for i in 0..20.min(fl) {
                eprintln!("    [{}] cpu={:12.8} gpu={:12.8} diff={:.10}", i, cpu_flop[i], gpu_all[i], cpu_flop[i] - gpu_all[i]);
            }
            // Count near-zero regrets (where strategy flip could occur)
            let mut near_zero = 0;
            for i in 0..fl {
                if cpu_flop[i].abs() < 0.01 { near_zero += 1; }
            }
            eprintln!("  Near-zero flop regrets (<0.01): {} / {}", near_zero, fl);
        }

        if max_diff > 0.001 {
            // Find top 5 diverging regret entries
            let mut diffs: Vec<(usize, f32, f32, f32, &str)> = Vec::new();
            for i in 0..fl {
                let d = (cpu_flop[i] - gpu_all[i]).abs();
                if d > 0.001 {
                    diffs.push((i, cpu_flop[i], gpu_all[i], d, "FLOP"));
                }
            }
            for i in 0..tl {
                let d = (cpu_turn[i] - gpu_all[fl + i]).abs();
                if d > 0.001 {
                    diffs.push((i, cpu_turn[i], gpu_all[fl + i], d, "TURN"));
                }
            }
            for i in 0..cpu_river.len() {
                if fl + tl + i < gpu_all.len() {
                    let d = (cpu_river[i] - gpu_all[fl + tl + i]).abs();
                    if d > 0.001 {
                        diffs.push((i, cpu_river[i], gpu_all[fl + tl + i], d, "RIVER"));
                    }
                }
            }
            diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
            eprintln!("  Top divergences ({} total > 0.001):", diffs.len());
            for (i, cpu, gpu, d, zone) in diffs.iter().take(10) {
                // Decode: nh=NUM_HANDS, MAX_NA_POSTFLOP=4, index = node_local * MAX_NA_POSTFLOP * nh + action * nh + hand
                let hand = i % NUM_HANDS;
                let action = (i / NUM_HANDS) % 4;
                let node_local = i / (4 * NUM_HANDS);
                eprintln!("    {} idx={} (node_local={} a={} h={}): cpu={:.6} gpu={:.6} diff={:.6}",
                    zone, i, node_local, action, hand, cpu, gpu, d);
            }
            panic!("Divergence detected at iter {}", iter + 1);
        }
    }
    eprintln!("All 10 iterations match!");
}
