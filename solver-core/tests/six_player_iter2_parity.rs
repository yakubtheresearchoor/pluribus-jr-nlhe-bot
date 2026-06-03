// Iter-2 (non-uniform reach) 6-player parity — the gate that catches
// indexing bugs that iter-0's uniform reach data masks. This was the
// specific gap that caught the multiply-ordering bug in the 3-player
// port; running it on 6-player closes the same gap for the new K≥3
// factored path.
//
// Method: same as gate 5, but run 2 CFR iterations on each side before
// comparing regrets. At iter-2 the reach state is non-uniform (derived
// from iter-1's regret-matched strategy), so position-dependent bugs
// that don't show on uniform-reach data DO show.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_6p_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 6u8;
    let num_opp = 5usize;

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i*2] = c1; hand_cards[i*2+1] = c2;
    }
    let mut conflict = vec![0u8; nh*nh];
    for i in 0..nh { for j in 0..nh {
        if i == j { conflict[i*nh+j] = 1; continue; }
        let (a1,a2) = index_to_card_pair(chosen[i] as usize);
        let (b1,b2) = index_to_card_pair(chosen[j] as usize);
        if a1==b1||a1==b2||a2==b1||a2==b2 { conflict[i*nh+j] = 1; }
    }}
    let mut hr = vec![0u16; nh];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        hr[i] = h.evaluate_internal() as u16;
    }
    let tc = vec![card_from_str("3c").unwrap() as u8];
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    rd[tc[0] as usize] = vec![card_from_str("5s").unwrap() as u8];
    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &t in &tc {
        for (i, &hi) in chosen.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let tm = board_mask | (1u64 << t);
            if tm & (1u64 << c1) != 0 || tm & (1u64 << c2) != 0 { continue; }
            let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
            for &bc in &board { h = h.add_card(bc as usize); }
            h = h.add_card(t as usize);
            turn_ranks[t as usize * nh + i] = h.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh)
            .map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            let off = t as usize * num_opp * nh + oi * nh;
            for h in 0..nh { turn_sorted_str[off + h] = items[h].0; turn_sorted_idx[off + h] = items[h].1; }
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &t in &tc {
        let tm = board_mask | (1u64 << t);
        for &r in &rd[t as usize] {
            let fm = tm | (1u64 << r);
            for (i, &hi) in chosen.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if fm & (1u64 << c1) != 0 || fm & (1u64 << c2) != 0 { continue; }
                let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
                for &bc in &board { h = h.add_card(bc as usize); }
                h = h.add_card(t as usize).add_card(r as usize);
                river_ranks[t as usize * 52 * nh + r as usize * nh + i] = h.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16)).collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; num_players as usize];
    fn enum_nc(player: usize, np: usize, nh: usize, combined: u64,
               hand_cards: &[u8], weight: f64) -> f64 {
        if player == np { return weight; }
        let mut total = 0.0;
        for h in 0..nh {
            let m = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
            if combined & m != 0 { continue; }
            total += enum_nc(player + 1, np, nh, combined | m, hand_cards, weight);
        }
        total
    }
    let nc = enum_nc(0, num_players as usize, nh, 0, &hand_cards[..], 1.0);

    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players, num_combinations: nc, river_decks: rd,
    };
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![100; 6], initial_contributions: vec![5; 6],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

#[test]
fn six_player_iter2_parity_non_uniform_reach() {
    let nh = 6;
    let (tree, table) = build_6p_table(nh);
    let game = FlopStartGame::new(table);

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.set_iteration(0);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    eprintln!("\n=== iter-2 non-uniform-reach parity, 6p nh={} ===", nh);
    eprintln!("Tree: {} nodes total", tree.num_nodes());

    // Run TWO iterations on each side.
    // After iter-1, regrets are non-zero → regret-matched strategy is
    // non-uniform → iter-2's reach state is non-uniform. Indexing bugs
    // that uniform-reach data masks DO show at this iter.
    let t_cpu = std::time::Instant::now();
    cpu.run(&tree, &game, 2);
    let cpu_elapsed = t_cpu.elapsed();
    eprintln!("CPU 2 iters: {:?}", cpu_elapsed);

    let t_gpu = std::time::Instant::now();
    gpu.run(&ctx, &tree, &game, 2);
    let gpu_elapsed = t_gpu.elapsed();
    eprintln!("GPU 2 iters: {:?}", gpu_elapsed);

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
    let mut zone_sum_sq = [0.0f64; 3];
    let mut zone_max_value = [0.0f32; 3];

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
            let v = cpu_slice[i].abs().max(gpu_all[gi].abs());
            if cpu_slice[i] != 0.0 || gpu_all[gi] != 0.0 {
                zone_max[zone_idx] = zone_max[zone_idx].max(diff);
                zone_sum_sq[zone_idx] += (diff as f64).powi(2);
                zone_count[zone_idx] += 1;
                zone_max_value[zone_idx] = zone_max_value[zone_idx].max(v);
            }
        }
    }
    let flop_rmse = (zone_sum_sq[0] / zone_count[0].max(1) as f64).sqrt();
    let turn_rmse = (zone_sum_sq[1] / zone_count[1].max(1) as f64).sqrt();
    let river_rmse = (zone_sum_sq[2] / zone_count[2].max(1) as f64).sqrt();

    eprintln!("\nFlop zone:  max_diff={:.6e} rmse={:.6e} max_value={:.3e} ({} nonzero entries)",
        zone_max[0], flop_rmse, zone_max_value[0], zone_count[0]);
    eprintln!("Turn zone:  max_diff={:.6e} rmse={:.6e} max_value={:.3e} ({} nonzero entries)",
        zone_max[1], turn_rmse, zone_max_value[1], zone_count[1]);
    eprintln!("River zone: max_diff={:.6e} rmse={:.6e} max_value={:.3e} ({} nonzero entries)",
        zone_max[2], river_rmse, zone_max_value[2], zone_count[2]);

    let max_diff = zone_max[0].max(zone_max[1]).max(zone_max[2]);
    eprintln!("\nOverall max_diff (iter-2, non-uniform reach) = {:.6e}", max_diff);

    // #38 final tightening: from 1e-3 → 1e-5 after #39's orchestration oracle
    // anchored bottom_up_zone against textbook CFR formula. Tight CPU-GPU
    // tolerance now means precision around truth, not mutual agreement.
    // Empirical: 1.9e-6.
    assert!(max_diff < 1e-5,
        "iter-2 6-player parity failed: max_diff = {} (threshold 1e-5)", max_diff);
}
