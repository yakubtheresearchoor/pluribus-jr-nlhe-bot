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
                // Decode: nh=NUM_HANDS, MAX_NA=4, index = node_local * MAX_NA * nh + action * nh + hand
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

/// Stage-by-stage CFV comparison at iter 2 to find the first diverging stage.
#[test]
fn stage_by_stage_iter2() {
    let (tree, table) = build_table();
    let game = FlopStartGame::new(table);
    let nh = NUM_HANDS;
    let nn = tree.num_nodes();

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().unwrap();
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Run 1 iteration on both (after which regrets match to float precision)
    cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    eprintln!("Tree: nn={}, nh={}", nn, nh);

    // Verify iter-1 regrets match
    let cpu_reg_f = cpu.regrets_flop();
    let cpu_reg_t = cpu.regrets_turn();
    let cpu_reg_r = cpu.regrets_river();
    let gpu_reg = gpu.download_regrets();
    let fl = cpu_reg_f.len();
    let tl = cpu_reg_t.len();
    let rl = cpu_reg_r.len();
    eprintln!("Regret sizes: flop={} turn={} river={} gpu_total={}", fl, tl, rl, gpu_reg.len());
    eprintln!("Expected gpu_total = {}", fl + tl + rl);
    let mut reg_max = 0.0f32;
    for i in 0..fl { reg_max = reg_max.max((cpu_reg_f[i] - gpu_reg[i]).abs()); }
    for i in 0..tl { reg_max = reg_max.max((cpu_reg_t[i] - gpu_reg[fl + i]).abs()); }
    for i in 0..rl {
        if fl + tl + i < gpu_reg.len() {
            reg_max = reg_max.max((cpu_reg_r[i] - gpu_reg[fl + tl + i]).abs());
        }
    }
    eprintln!("After iter 1, regret max_diff = {:.8} (all zones)", reg_max);
    assert!(reg_max < 0.001, "Regrets should match after iter 1");

    // KEY TEST: Upload CPU regrets to GPU to eliminate float precision differences.
    // If divergence disappears, the bug is float sensitivity (not structural).
    // If divergence persists, there's a structural bug.
    {
        let mut exact_regrets = Vec::new();
        exact_regrets.extend_from_slice(cpu.regrets_flop());
        exact_regrets.extend_from_slice(cpu.regrets_turn());
        exact_regrets.extend_from_slice(cpu.regrets_river());
        gpu.upload_regrets(&exact_regrets);
        eprintln!("Uploaded CPU regrets to GPU (forcing exact match)");

        // Also sync cum_strategy
        let mut exact_cum = Vec::new();
        exact_cum.extend_from_slice(cpu.cum_strategy_flop());
        exact_cum.extend_from_slice(cpu.cum_strategy_turn());
        exact_cum.extend_from_slice(cpu.cum_strategy_river());
        gpu.upload_cum_strategy(&exact_cum);
        gpu.set_iteration(cpu.iteration_count());
        eprintln!("Uploaded CPU cum_strategy to GPU, synced iteration to {}", cpu.iteration_count());
    }

    // Before running, verify strategies match after sync
    {
        cpu.compute_all_strategies(&tree);
        gpu.compute_all_strategies(&ctx);
        let cs = cpu.strategy_flop();
        let gs = gpu.download_strategy();
        let mut smax = 0.0f32;
        let mut sdiv = 0;
        for i in 0..cs.len() {
            let d = (cs[i] - gs[i]).abs();
            smax = smax.max(d);
            if d > 0.001 { sdiv += 1; }
        }
        eprintln!("After sync, flop strategy max_diff = {:.8} ({} diverged / {})", smax, sdiv, cs.len());
        if smax > 0.001 {
            // Print first divergences
            for i in 0..cs.len() {
                let d = (cs[i] - gs[i]).abs();
                if d > 0.001 {
                    let h = i % NUM_HANDS;
                    let a = (i / NUM_HANDS) % 4;
                    let node_local = i / (4 * NUM_HANDS);
                    eprintln!("  strat[{}] (n={} a={} h={}): cpu={:.8} gpu={:.8} diff={:.8}", i, node_local, a, h, cs[i], gs[i], d);
                    if node_local < 3 {
                        // Print the regrets that produced this strategy
                        let cf = cpu.regrets_flop();
                        let off = node_local * 4 * NUM_HANDS;
                        eprintln!("    cpu regrets: {:?}", &cf[off..off + 4 * NUM_HANDS]);
                        let gr = gpu.download_regrets();
                        eprintln!("    gpu regrets: {:?}", &gr[off..off + 4 * NUM_HANDS]);
                    }
                }
            }
        }
    }

    // Run iter 2 on both (with exactly matching start state)
    cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    // Compare iter-2 regrets
    let cpu_reg_f2 = cpu.regrets_flop();
    let cpu_reg_t2 = cpu.regrets_turn();
    let cpu_reg_r2 = cpu.regrets_river();
    let gpu_reg2 = gpu.download_regrets();
    let mut reg_max2 = 0.0f32;
    for i in 0..fl { reg_max2 = reg_max2.max((cpu_reg_f2[i] - gpu_reg2[i]).abs()); }
    for i in 0..tl { reg_max2 = reg_max2.max((cpu_reg_t2[i] - gpu_reg2[fl + i]).abs()); }
    for i in 0..rl {
        if fl + tl + i < gpu_reg2.len() {
            reg_max2 = reg_max2.max((cpu_reg_r2[i] - gpu_reg2[fl + tl + i]).abs());
        }
    }
    eprintln!("After iter 2 (synced start): regret max_diff = {:.8}", reg_max2);
    if reg_max2 > 0.01 {
        eprintln!("  STRUCTURAL BUG: divergence persists even with exact regret sync");
    } else {
        eprintln!("  Float sensitivity confirmed: no structural divergence");
    }

    // Now do a debug traverser pass for iter 2, traverser 0
    // IMPORTANT: This uses the SYNCED state (CPU regrets uploaded to GPU)
    // Re-sync to undo the effects of the run(1) above
    {
        let mut exact_regrets = Vec::new();
        exact_regrets.extend_from_slice(cpu.regrets_flop());
        exact_regrets.extend_from_slice(cpu.regrets_turn());
        exact_regrets.extend_from_slice(cpu.regrets_river());
        // Wait - cpu.run(1) already changed CPU regrets. We need to re-sync FROM
        // the state BEFORE run(1). But we've already run(1) on both.
        // Instead, let me create fresh solvers from the saved iter-1 state.
    }

    // Actually, let's restart from scratch for the debug pass:
    // Create fresh CPU and GPU, run 1 iter each, sync, then debug
    let mut cpu2 = FlopStartVectorCfr::new(&tree, &game.table());
    let mut gpu2 = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu2);
    cpu2.run(&tree, &game, 1);
    gpu2.run(&ctx, &tree, &game, 1);
    // Upload CPU regrets to GPU
    {
        let mut er = Vec::new();
        er.extend_from_slice(cpu2.regrets_flop());
        er.extend_from_slice(cpu2.regrets_turn());
        er.extend_from_slice(cpu2.regrets_river());
        gpu2.upload_regrets(&er);
        let mut ec = Vec::new();
        ec.extend_from_slice(cpu2.cum_strategy_flop());
        ec.extend_from_slice(cpu2.cum_strategy_turn());
        ec.extend_from_slice(cpu2.cum_strategy_river());
        gpu2.upload_cum_strategy(&ec);
        gpu2.set_iteration(cpu2.iteration_count());
    }

    eprintln!("\n=== Debug traverser 0, iter 2 (SYNCED) ===\n");

    let (cpu_river_cfvs, cpu_turn_cfvs, cpu_flop_cfv, cpu_reach) =
        cpu2.debug_traverser_cfvs(&tree, &game, 0);
    let (gpu_river_cfvs, gpu_flop_reach, gpu_flop_cfv) =
        gpu2.debug_traverser_cfvs(&ctx, &tree, &game, 0);

    // Compare reach
    let mut reach_max = 0.0f32;
    for i in 0..cpu_reach.len().min(gpu_flop_reach.len()) {
        reach_max = reach_max.max((cpu_reach[i] - gpu_flop_reach[i]).abs());
    }
    eprintln!("Flop reach max_diff = {:.8}", reach_max);

    // Compare river CFVs per outcome
    eprintln!("\nRiver CFV comparison:");
    for (idx, (cpu_entry, gpu_entry)) in cpu_river_cfvs.iter().zip(gpu_river_cfvs.iter()).enumerate() {
        let (cti, cri, ref ccfv) = *cpu_entry;
        let (gti, gri, ref gcfv) = *gpu_entry;
        assert_eq!(cti, gti);
        assert_eq!(cri, gri);
        let mut max_d = 0.0f32;
        let mut max_i = 0;
        for i in 0..ccfv.len().min(gcfv.len()) {
            let d = (ccfv[i] - gcfv[i]).abs();
            if d > max_d { max_d = d; max_i = i; }
        }
        let node_id = max_i / nh;
        let hand = max_i % nh;
        eprintln!("  ti={} ri={}: max_diff={:.8} at node {} hand {} (cpu={:.6} gpu={:.6})",
            cti, cri, max_d, node_id, hand,
            ccfv.get(max_i).unwrap_or(&0.0), gcfv.get(max_i).unwrap_or(&0.0));

        if max_d > 0.001 {
            // Print first few divergences
            let mut count = 0;
            for i in 0..ccfv.len().min(gcfv.len()) {
                let d = (ccfv[i] - gcfv[i]).abs();
                if d > 0.001 {
                    let nid = i / nh;
                    let h = i % nh;
                    let node = &tree.nodes[nid];
                    eprintln!("    node {} (type={} player={}) hand {}: cpu={:.6} gpu={:.6} diff={:.6}",
                        nid, node.node_type, node.player_id, h, ccfv[i], gcfv[i], d);
                    count += 1;
                    if count >= 5 { break; }
                }
            }
        }
    }

    // Compare turn CFVs
    eprintln!("\nTurn CFV comparison:");
    // GPU turn CFVs need to be extracted from turn_cfv_batch
    let gpu_turn_batch = gpu2.download_turn_cfv_batch();
    for (cpu_entry, ti) in cpu_turn_cfvs.iter().zip(0..) {
        let (cti, ref ccfv) = *cpu_entry;
        let start = ti * nn * nh;
        let end = start + nn * nh;
        let gcfv = &gpu_turn_batch[start..end];
        let mut max_d = 0.0f32;
        let mut max_i = 0;
        for i in 0..ccfv.len().min(gcfv.len()) {
            let d = (ccfv[i] - gcfv[i]).abs();
            if d > max_d { max_d = d; max_i = i; }
        }
        eprintln!("  ti={}: max_diff={:.8} at node {} hand {}",
            cti, max_d, max_i / nh, max_i % nh);

        if max_d > 0.001 {
            let mut count = 0;
            for i in 0..ccfv.len().min(gcfv.len()) {
                let d = (ccfv[i] - gcfv[i]).abs();
                if d > 0.001 {
                    let nid = i / nh;
                    let h = i % nh;
                    let node = &tree.nodes[nid];
                    eprintln!("    node {} (type={} player={}) hand {}: cpu={:.6} gpu={:.6} diff={:.6}",
                        nid, node.node_type, node.player_id, h, ccfv[i], gcfv[i], d);
                    count += 1;
                    if count >= 5 { break; }
                }
            }
        }
    }

    // Compare flop CFV
    eprintln!("\nFlop CFV comparison:");
    let mut flop_max = 0.0f32;
    let mut flop_max_i = 0;
    for i in 0..cpu_flop_cfv.len().min(gpu_flop_cfv.len()) {
        let d = (cpu_flop_cfv[i] - gpu_flop_cfv[i]).abs();
        if d > flop_max { flop_max = d; flop_max_i = i; }
    }
    eprintln!("  max_diff={:.8} at node {} hand {}",
        flop_max, flop_max_i / nh, flop_max_i % nh);

    if flop_max > 0.001 {
        let mut count = 0;
        for i in 0..cpu_flop_cfv.len().min(gpu_flop_cfv.len()) {
            let d = (cpu_flop_cfv[i] - gpu_flop_cfv[i]).abs();
            if d > 0.001 {
                let nid = i / nh;
                let h = i % nh;
                let node = &tree.nodes[nid];
                eprintln!("    node {} (type={} player={}) hand {}: cpu={:.6} gpu={:.6} diff={:.6}",
                    nid, node.node_type, node.player_id, h, cpu_flop_cfv[i], gpu_flop_cfv[i], d);
                count += 1;
                if count >= 5 { break; }
            }
        }
    }

    // Don't assert on flop_max in debug pass (uses vanilla params, not real)

    // === Targeted check: river accum comparison ===
    // Run the GPU pipeline step by step for ti=0, compare river_accum
    eprintln!("\n=== Targeted chance_accumulate check ===");
    {
        // Re-create fresh synced solvers
        let mut cpu3 = FlopStartVectorCfr::new(&tree, &game.table());
        let mut gpu3 = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu3);
        cpu3.run(&tree, &game, 1);
        gpu3.run(&ctx, &tree, &game, 1);
        let mut er = Vec::new();
        er.extend_from_slice(cpu3.regrets_flop());
        er.extend_from_slice(cpu3.regrets_turn());
        er.extend_from_slice(cpu3.regrets_river());
        gpu3.upload_regrets(&er);
        let mut ec = Vec::new();
        ec.extend_from_slice(cpu3.cum_strategy_flop());
        ec.extend_from_slice(cpu3.cum_strategy_turn());
        ec.extend_from_slice(cpu3.cum_strategy_river());
        gpu3.upload_cum_strategy(&ec);
        gpu3.set_iteration(cpu3.iteration_count());

        // Step through GPU for traverser 0
        let params = DcfrParams { alpha_t: 1.0, beta_t: 1.0, gamma_t: 1.0 };
        gpu3.compute_all_strategies(&ctx);
        gpu3.compute_reach_flop(&ctx);
        gpu3.zero_buffer_name(&ctx, 100);
        gpu3.zero_buffer_name(&ctx, 2);

        // ti=0
        let ti = 0;
        let n_river = gpu3.river_outcomes_per_turn()[ti];
        gpu3.zero_buffer_name(&ctx, 0);
        gpu3.zero_buffer_name(&ctx, 1);
        gpu3.compute_reach_turn(&ctx, ti);

        for ri in 0..n_river {
            gpu3.compute_reach_river(&ctx, ti, ri);
            gpu3.bottom_up_river(&ctx, ti, ri, 0, &params);
        }

        // Download river_accum BEFORE chance_accumulate
        let pre_accum = gpu3.download_river_accum();
        eprintln!("  GPU river_accum before chance_accumulate: first 20 = {:?}", &pre_accum[..20.min(pre_accum.len())]);

        gpu3.chance_accumulate_river(&ctx, ti, n_river);

        // Download river_accum AFTER chance_accumulate
        let post_accum = gpu3.download_river_accum();
        eprintln!("  GPU river_accum after chance_accumulate: first 20 = {:?}", &post_accum[..20.min(post_accum.len())]);

        // Compare with CPU's river_cfv_accum
        // The CPU debug pass already computed this for us in cpu_river_cfvs
        // But we need to recompute it manually
        let table = game.table();
        let tc = table.remaining_deck[0];
        let mut cpu_accum = vec![0.0f32; nn * nh];
        for (_, (cti, cri, ref ccfv)) in cpu_river_cfvs.iter().enumerate() {
            if *cti != 0 { continue; }
            for &child_id in cpu3.river_chance_children() {
                for h in 0..nh {
                    let cp = table.chance_probability_river(tc, *cri, h);
                    cpu_accum[child_id as usize * nh + h] += cp * ccfv[child_id as usize * nh + h];
                }
            }
        }

        // Compare at chance children
        eprintln!("\n  River accum comparison at chance children:");
        let mut accum_max = 0.0f32;
        for &child_id in cpu3.river_chance_children() {
            for h in 0..nh {
                let ci = child_id as usize * nh + h;
                let d = (cpu_accum[ci] - post_accum[ci]).abs();
                accum_max = accum_max.max(d);
            }
        }
        eprintln!("  max_diff = {:.8}", accum_max);

        // Now check chance_finalize_river and bottom_up_turn
        gpu3.chance_finalize_river(&ctx, ti);
        let turn_cfv_before = gpu3.download_turn_cfv_batch();
        eprintln!("\n  GPU turn_cfv_batch after finalize (before bottom_up_turn):");
        // Check the first few entries at the chance children nodes
        for &child_id in cpu3.river_chance_children().iter().take(2) {
            let start = child_id as usize * nh;
            eprintln!("    node {} (ti=0 offset): {:?}", child_id, &turn_cfv_before[start..start + nh]);
            eprintln!("    cpu accum at node {}:    {:?}", child_id, &cpu_accum[start..start + nh]);
        }

        gpu3.bottom_up_turn(&ctx, ti, 0, &params);
        let turn_cfv_after = gpu3.download_turn_cfv_batch();
        // Compare with CPU turn CFV
        if let Some((_, ref cpu_tcfv)) = cpu_turn_cfvs.iter().find(|(t, _)| *t == 0) {
            let mut tmax = 0.0f32;
            for i in 0..cpu_tcfv.len().min(turn_cfv_after.len()) {
                tmax = tmax.max((cpu_tcfv[i] - turn_cfv_after[i]).abs());
            }
            eprintln!("\n  Turn CFV after bottom_up (ti=0): max_diff = {:.8}", tmax);
        }
    }
}

/// Per-traverser divergence: run traversers one at a time, compare regrets after each.
#[test]
fn per_traverser_divergence() {
    let (tree, table) = build_table();
    let game = FlopStartGame::new(table);
    let nh = NUM_HANDS;

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().unwrap();
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Run 1 full iteration on both
    cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    // Sync GPU to CPU state
    let mut er = Vec::new();
    er.extend_from_slice(cpu.regrets_flop());
    er.extend_from_slice(cpu.regrets_turn());
    er.extend_from_slice(cpu.regrets_river());
    gpu.upload_regrets(&er);
    let mut ec = Vec::new();
    ec.extend_from_slice(cpu.cum_strategy_flop());
    ec.extend_from_slice(cpu.cum_strategy_turn());
    ec.extend_from_slice(cpu.cum_strategy_river());
    gpu.upload_cum_strategy(&ec);
    gpu.set_iteration(cpu.iteration_count());
    eprintln!("Synced. iteration = {}", cpu.iteration_count());

    let fl = cpu.regrets_flop().len();
    let tl = cpu.regrets_turn().len();

    // Run iter 2 traverser by traverser
    for t in 0..3 {
        // Compare strategies BEFORE this traverser runs
        cpu.compute_all_strategies(&tree);
        gpu.compute_all_strategies(&ctx);
        let cs = cpu.strategy_flop();
        let gs_all = gpu.download_strategy();
        let gs = &gs_all[..cs.len()]; // flop portion
        let mut smax = 0.0f32;
        let mut sdiv = 0;
        for i in 0..cs.len() {
            let d = (cs[i] - gs[i]).abs();
            smax = smax.max(d);
            if d > 0.001 { sdiv += 1; }
        }
        eprintln!("Before traverser {}: flop strategy max_diff={:.8} ({} diverged)", t, smax, sdiv);
        if sdiv > 0 {
            for i in 0..cs.len() {
                let d = (cs[i] - gs[i]).abs();
                if d > 0.001 {
                    let h = i % nh;
                    let a = (i / nh) % 4;
                    let nl = i / (4 * nh);
                    eprintln!("  strat[{}] (n={} a={} h={}): cpu={:.8} gpu={:.8}", i, nl, a, h, cs[i], gs[i]);
                    // Also print regrets that produce this strategy
                    let cf = cpu.regrets_flop();
                    let gf = &gpu.download_regrets()[..fl];
                    let base = nl * 4 * nh;
                    eprintln!("    cpu_reg[n={}]: {:?}", nl, &cf[base..base + 4 * nh]);
                    eprintln!("    gpu_reg[n={}]: {:?}", nl, &gf[base..base + 4 * nh]);
                    break;
                }
            }
        }

        // Also check turn strategies
        let cts = cpu.strategy_turn();
        let gts = &gs_all[fl..fl + cts.len()]; // turn portion
        let mut tsmax = 0.0f32;
        let mut tsdiv = 0;
        for i in 0..cts.len() {
            let d = (cts[i] - gts[i]).abs();
            tsmax = tsmax.max(d);
            if d > 0.001 { tsdiv += 1; }
        }
        eprintln!("Before traverser {}: turn strategy max_diff={:.8} ({} diverged)", t, tsmax, tsdiv);

        let inc = t == 0; // increment iteration on first traverser
        cpu.run_single_traverser(&tree, &game, t, inc);
        gpu.run_single_traverser(&ctx, t, inc);

        // Compare regrets after this traverser
        let cf = cpu.regrets_flop();
        let ct = cpu.regrets_turn();
        let cr = cpu.regrets_river();
        let g = gpu.download_regrets();

        let mut fmax = 0.0f32;
        for i in 0..fl { fmax = fmax.max((cf[i] - g[i]).abs()); }
        let mut tmax = 0.0f32;
        for i in 0..tl { tmax = tmax.max((ct[i] - g[fl + i]).abs()); }
        let mut rmax = 0.0f32;
        for i in 0..cr.len() {
            if fl + tl + i < g.len() {
                rmax = rmax.max((cr[i] - g[fl + tl + i]).abs());
            }
        }
        let max_diff = fmax.max(tmax).max(rmax);
        eprintln!("After traverser {}: max_diff = {:.8} (flop={:.8} turn={:.8} river={:.8})",
            t, max_diff, fmax, tmax, rmax);

        if max_diff > 0.01 {
            // Find where
            let mut count = 0;
            for i in 0..fl {
                let d = (cf[i] - g[i]).abs();
                if d > 0.01 {
                    let h = i % nh;
                    let a = (i / nh) % 4;
                    let nl = i / (4 * nh);
                    eprintln!("  FLOP regret[{}] (n={} a={} h={}): cpu={:.6} gpu={:.6} diff={:.6}",
                        i, nl, a, h, cf[i], g[i], d);
                    count += 1;
                    if count >= 5 { break; }
                }
            }
            count = 0;
            for i in 0..tl {
                let d = (ct[i] - g[fl + i]).abs();
                if d > 0.01 {
                    eprintln!("  TURN regret[{}]: cpu={:.6} gpu={:.6} diff={:.6}", i, ct[i], g[fl + i], d);
                    count += 1;
                    if count >= 3 { break; }
                }
            }
            count = 0;
            for i in 0..cr.len() {
                if fl + tl + i < g.len() {
                    let d = (cr[i] - g[fl + tl + i]).abs();
                    if d > 0.01 {
                        eprintln!("  RIVER regret[{}]: cpu={:.6} gpu={:.6} diff={:.6}", i, cr[i], g[fl + tl + i], d);
                        count += 1;
                        if count >= 3 { break; }
                    }
                }
            }
        }
    }
}
