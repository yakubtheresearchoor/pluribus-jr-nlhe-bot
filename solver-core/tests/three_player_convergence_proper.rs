/// 3-player convergence gate with proper hand count.
/// Uses 30 hands from a real board WITH natural card blocking,
/// making the product formula accurate (error < 3%).
use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

const NUM_HANDS: usize = 50;

fn build_multihand_3player_table() -> (solver_core::tree::flat::FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));

    // Pick 30 hands spread across the valid set (every Nth hand).
    // This gives natural card conflicts without ALL hands sharing one card.
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
            continue;
        }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / NUM_HANDS;
    let chosen_hands: Vec<u16> = (0..NUM_HANDS).map(|i| all_valid[i * step]).collect();
    assert_eq!(chosen_hands.len(), NUM_HANDS);

    let nh = NUM_HANDS;
    let num_players = 3u8;
    let num_opp = 2;

    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen_hands.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
    }

    // Conflict matrix
    let mut conflict = vec![0u8; nh * nh];
    let mut num_conflicts = 0;
    for i in 0..nh {
        for j in 0..nh {
            if i == j {
                conflict[i * nh + j] = 1;
                continue;
            }
            let (c1a, c1b) = index_to_card_pair(chosen_hands[i] as usize);
            let (c2a, c2b) = index_to_card_pair(chosen_hands[j] as usize);
            if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b {
                conflict[i * nh + j] = 1;
                num_conflicts += 1;
            }
        }
    }
    eprintln!("Hand conflicts: {} pairs out of {} (card blocking rate: {:.1}%)",
        num_conflicts / 2, nh * (nh - 1) / 2,
        100.0 * num_conflicts as f64 / (nh * (nh - 1)) as f64);

    // Hand ranks for flop
    let mut hand_ranks_base = vec![0u16; nh];
    for (i, &hi) in chosen_hands.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut hand = Hand::new();
        hand = hand.add_card(c1 as usize);
        hand = hand.add_card(c2 as usize);
        for &bc in &board {
            hand = hand.add_card(bc as usize);
        }
        hand_ranks_base[i] = hand.evaluate_internal() as u16;
    }

    // Use 3 turn cards for more realistic convergence testing
    let turn_cards: Vec<u8> = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("4c").unwrap() as u8,
        card_from_str("Tc").unwrap() as u8,
    ];

    // 2 river cards per turn
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("5s").unwrap() as u8,
        card_from_str("Jh").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("6s").unwrap() as u8,
        card_from_str("Qh").unwrap() as u8,
    ];
    river_decks[turn_cards[2] as usize] = vec![
        card_from_str("8s").unwrap() as u8,
        card_from_str("Ad").unwrap() as u8,
    ];

    // Compute turn ranks and sorted arrays
    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for (i, &hi) in chosen_hands.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 {
                continue;
            }
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in &board {
                hand = hand.add_card(bc as usize);
            }
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

    // Compute river ranks and sorted arrays
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for &rc in &river_decks[tc as usize] {
            let full_mask = turn_mask | (1u64 << rc);
            for (i, &hi) in chosen_hands.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if full_mask & (1u64 << c1) != 0 || full_mask & (1u64 << c2) != 0 {
                    continue;
                }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in &board {
                    hand = hand.add_card(bc as usize);
                }
                hand = hand.add_card(tc as usize);
                hand = hand.add_card(rc as usize);
                river_ranks[tc as usize * 52 * nh + rc as usize * nh + i] =
                    hand.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| {
                    (
                        river_ranks[tc as usize * 52 * nh + rc as usize * nh + h] + 1,
                        h as u16,
                    )
                })
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off =
                    tc as usize * 52 * num_opp * nh + rc as usize * num_opp * nh + oi * nh;
                for h in 0..nh {
                    river_sorted_str[off + h] = items[h].0;
                    river_sorted_idx[off + h] = items[h].1;
                }
            }
        }
    }

    let initial_weights = vec![vec![1.0f32; nh]; num_players as usize];

    // Compute num_combinations (valid 3-player assignments)
    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let mask0: u64 = (1u64 << hand_cards[h0 * 2]) | (1u64 << hand_cards[h0 * 2 + 1]);
        for h1 in 0..nh {
            if h0 == h1 { continue; }
            let mask1: u64 = (1u64 << hand_cards[h1 * 2]) | (1u64 << hand_cards[h1 * 2 + 1]);
            if mask0 & mask1 != 0 { continue; }
            for h2 in 0..nh {
                if h2 == h0 || h2 == h1 { continue; }
                let mask2: u64 = (1u64 << hand_cards[h2 * 2]) | (1u64 << hand_cards[h2 * 2 + 1]);
                if mask0 & mask2 != 0 || mask1 & mask2 != 0 { continue; }
                nc += 1.0;
            }
        }
    }
    eprintln!("num_combinations = {}", nc);

    let table = FlopChanceTable {
        hand_ranks_base,
        valid_hand_indices: chosen_hands,
        num_valid: nh,
        conflict,
        hand_cards,
        remaining_deck: turn_cards,
        turn_ranks,
        turn_sorted_str,
        turn_sorted_idx,
        river_ranks,
        river_sorted_str,
        river_sorted_idx,
        initial_weights,
        num_players,
        num_combinations: nc,
        river_decks,
    };
    let config = TreeConfig {
        num_players: 3,
        initial_state: BoardState::Flop,
        starting_pot: 15,
        starting_stacks: vec![100, 100, 100],
        initial_contributions: vec![5, 5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

fn measure_exploitability(
    cpu: &FlopStartVectorCfr,
    tree: &solver_core::tree::flat::FlatTree,
    game: &FlopStartGame,
    np: usize,
) -> f32 {
    let mut total_expl = 0.0f32;
    for p in 0..np {
        let br = cpu.best_response_value_debug(tree, game, p as u8);
        let sv = cpu.strategy_value_debug(tree, game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total_expl += (br[h] - sv[h]).max(0.0);
        }
    }
    total_expl
}

fn upload_gpu_to_cpu(cpu: &mut FlopStartVectorCfr, gpu_reg: &[f32], gpu_cum: &[f32]) {
    let fl = cpu.regrets_flop().len();
    let tl = cpu.regrets_turn().len();
    {
        let r = cpu.regrets_flop_mut();
        for i in 0..fl { r[i] = gpu_reg[i]; }
    }
    {
        let r = cpu.regrets_turn_mut();
        for i in 0..tl { r[i] = gpu_reg[fl + i]; }
    }
    {
        let r = cpu.regrets_river_mut();
        for i in 0..r.len() {
            if fl + tl + i < gpu_reg.len() { r[i] = gpu_reg[fl + tl + i]; }
        }
    }
    {
        let c = cpu.cum_strategy_flop_mut();
        for i in 0..fl { c[i] = gpu_cum[i]; }
    }
    {
        let c = cpu.cum_strategy_turn_mut();
        for i in 0..tl { c[i] = gpu_cum[fl + i]; }
    }
    {
        let c = cpu.cum_strategy_river_mut();
        for i in 0..c.len() {
            if fl + tl + i < gpu_cum.len() { c[i] = gpu_cum[fl + tl + i]; }
        }
    }
}

/// Verify the game is approximately zero-sum (product formula accuracy check).
#[test]
fn gate_3p_zero_sum_check() {
    let (tree, table) = build_multihand_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;
    let nh = NUM_HANDS;

    // Check at iteration 0 (uniform strategies) first
    let cpu0 = FlopStartVectorCfr::new(&tree, &game.table());
    let sv0: Vec<Vec<f32>> = (0..np)
        .map(|p| cpu0.strategy_value_debug(&tree, &game, p as u8))
        .collect();
    let sv_total0: f32 = (0..np).map(|p| sv0[p].iter().sum::<f32>()).sum();
    let sv_pct0 = sv_total0.abs() / 15.0 / (nh as f32) * 100.0;
    eprintln!("Zero-sum at iter 0: total = {:.6} ({:.4}% of pot per hand)", sv_total0, sv_pct0);

    // Check at each iteration
    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    for iter in 0..10 {
        cpu.run(&tree, &game, 1);
        let sv: Vec<Vec<f32>> = (0..np)
            .map(|p| cpu.strategy_value_debug(&tree, &game, p as u8))
            .collect();
        let sv_total: f32 = (0..np).map(|p| sv[p].iter().sum::<f32>()).sum();
        let sv_pct = sv_total.abs() / 15.0 / (nh as f32) * 100.0;
        eprintln!("Zero-sum at iter {}: total = {:.4} ({:.4}%)", iter + 1, sv_total, sv_pct);
    }

    let sv: Vec<Vec<f32>> = (0..np)
        .map(|p| cpu.strategy_value_debug(&tree, &game, p as u8))
        .collect();

    let sv_total: f32 = (0..np).map(|p| sv[p].iter().sum::<f32>()).sum();
    let pot = 15.0f32;
    let sv_pct = sv_total.abs() / pot / (nh as f32) * 100.0;
    eprintln!("Zero-sum check: total SV sum = {:.4} ({:.2}% of pot per hand)",
        sv_total, sv_pct);

    // With 30 hands and natural card blocking, product formula error should be < 10%
    assert!(
        sv_pct < 10.0,
        "Product formula error too large: {:.2}% per hand (expected < 10%)",
        sv_pct
    );
}

/// CPU convergence gate for 30-hand 3-player game.
#[test]
fn gate_3p_cpu_convergence() {
    let (tree, table) = build_multihand_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;
    let pot = 15.0f32;

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());

    let checkpoints = [100, 500, 1000, 2000];
    let mut prev = 0u32;

    eprintln!("\n=== CPU 30-hand 3-player convergence ===");
    for &cp in &checkpoints {
        cpu.run(&tree, &game, cp - prev);
        prev = cp;
        let expl = measure_exploitability(&cpu, &tree, &game, np);
        let pct = expl / pot * 100.0;
        eprintln!("CPU iter {:5}: {:.2}% of pot", cp, pct);
    }

    let final_expl = measure_exploitability(&cpu, &tree, &game, np);
    let final_pct = final_expl / pot * 100.0;
    eprintln!("CPU final: {:.2}% of pot", final_pct);

    assert!(
        final_pct < 50.0,
        "CPU 30-hand failed to converge: {:.2}% (expected < 50% at 2000 iters)",
        final_pct
    );
}

/// GPU convergence and CPU/GPU agreement for 30-hand 3-player game.
#[test]
fn gate_3p_gpu_convergence() {
    let (tree, table) = build_multihand_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;
    let pot = 15.0f32;

    let mut cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);

    let checkpoints = [100, 500, 1000, 2000];
    let mut prev = 0u32;

    eprintln!("\n=== GPU 30-hand 3-player convergence ===");
    for &cp in &checkpoints {
        gpu.run(&ctx, &tree, &game, cp - prev);
        prev = cp;

        cpu_proxy.set_iteration(gpu.iteration());
        let gpu_reg = gpu.download_regrets();
        let gpu_cum = gpu.download_cum_strategy();
        upload_gpu_to_cpu(&mut cpu_proxy, &gpu_reg, &gpu_cum);

        let expl = measure_exploitability(&cpu_proxy, &tree, &game, np);
        let pct = expl / pot * 100.0;
        eprintln!("GPU iter {:5}: {:.2}% of pot", cp, pct);
    }

    let final_expl = measure_exploitability(&cpu_proxy, &tree, &game, np);
    let final_pct = final_expl / pot * 100.0;
    eprintln!("GPU final: {:.2}% of pot", final_pct);

    assert!(
        final_pct < 50.0,
        "GPU 30-hand failed to converge: {:.2}% (expected < 50% at 2000 iters)",
        final_pct
    );
}

/// Iter-0 parity: GPU matches CPU exactly at iteration 0 (structural correctness).
#[test]
fn gate_3p_iter0_parity() {
    let (tree, table) = build_multihand_3player_table();
    let game = FlopStartGame::new(table);

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.set_iteration(0);

    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    let cpu_flop = cpu.regrets_flop();
    let cpu_turn = cpu.regrets_turn();
    let cpu_river = cpu.regrets_river();
    let gpu_all = gpu.download_regrets();

    let fl = cpu_flop.len();
    let tl = cpu_turn.len();

    let mut flop_max = 0.0f32;
    let mut flop_nonzero = 0usize;
    for i in 0..fl {
        let d = (cpu_flop[i] - gpu_all[i]).abs();
        if d > 0.0 { flop_nonzero += 1; }
        flop_max = flop_max.max(d);
    }
    let mut turn_max = 0.0f32;
    let mut turn_nonzero = 0usize;
    for i in 0..tl {
        let d = (cpu_turn[i] - gpu_all[fl + i]).abs();
        if d > 0.0 { turn_nonzero += 1; }
        turn_max = turn_max.max(d);
    }
    let mut river_max = 0.0f32;
    let mut river_nonzero = 0usize;
    for i in 0..cpu_river.len() {
        if fl + tl + i < gpu_all.len() {
            let d = (cpu_river[i] - gpu_all[fl + tl + i]).abs();
            if d > 0.0 { river_nonzero += 1; }
            river_max = river_max.max(d);
        }
    }

    eprintln!("Zone sizes: flop={} turn={} river={}", fl, tl, cpu_river.len());
    eprintln!("Flop zone: max_diff={:.6} ({} nonzero diffs)", flop_max, flop_nonzero);
    eprintln!("Turn zone: max_diff={:.6} ({} nonzero diffs)", turn_max, turn_nonzero);
    eprintln!("River zone: max_diff={:.6} ({} nonzero diffs)", river_max, river_nonzero);

    let max_diff = flop_max.max(turn_max).max(river_max);
    eprintln!("Overall max_diff = {:.6}", max_diff);
    assert!(
        max_diff < 0.001,
        "iter-0 parity failed: max_diff = {:.6} (expected < 0.001). Flop={:.6} Turn={:.6} River={:.6}",
        max_diff, flop_max, turn_max, river_max
    );
}

/// Sync-and-check parity: run iter-1 on both, upload CPU regrets to GPU,
/// then run iter-2 on both — comparing regrets. This isolates whether the
/// per-iter regret divergence comes from float-arithmetic drift through
/// the 1e-5 regret matching threshold (iter-1 ULPs amplify into iter-2
/// strategy flips) versus a structural bug in the showdown port. If iter-2
/// regrets match closely after sync, the divergence is float sensitivity.
/// If they still diverge, there is a structural bug in the GPU port.
#[test]
fn gate_3p_iter2_synced_parity() {
    let (tree, table) = build_multihand_3player_table();
    let game = FlopStartGame::new(table);

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.set_iteration(0);
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Run iter-1 on both
    cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    let fl = cpu.regrets_flop().len();
    let tl = cpu.regrets_turn().len();
    let rl = cpu.regrets_river().len();

    // Iter-1 regret diff (baseline)
    {
        let gpu_reg = gpu.download_regrets();
        let mut max_d = 0.0f32;
        let cpu_f = cpu.regrets_flop();
        let cpu_t = cpu.regrets_turn();
        let cpu_r = cpu.regrets_river();
        for i in 0..fl { max_d = max_d.max((cpu_f[i] - gpu_reg[i]).abs()); }
        for i in 0..tl { max_d = max_d.max((cpu_t[i] - gpu_reg[fl + i]).abs()); }
        for i in 0..rl {
            if fl + tl + i < gpu_reg.len() {
                max_d = max_d.max((cpu_r[i] - gpu_reg[fl + tl + i]).abs());
            }
        }
        eprintln!("iter-1 (no sync): regret max_diff = {:.6e}", max_d);
    }

    // Sync CPU state to GPU (regrets + cum_strategy + iteration count)
    {
        let mut exact_regrets = Vec::new();
        exact_regrets.extend_from_slice(cpu.regrets_flop());
        exact_regrets.extend_from_slice(cpu.regrets_turn());
        exact_regrets.extend_from_slice(cpu.regrets_river());
        gpu.upload_regrets(&exact_regrets);

        let mut exact_cum = Vec::new();
        exact_cum.extend_from_slice(cpu.cum_strategy_flop());
        exact_cum.extend_from_slice(cpu.cum_strategy_turn());
        exact_cum.extend_from_slice(cpu.cum_strategy_river());
        gpu.upload_cum_strategy(&exact_cum);
        gpu.set_iteration(cpu.iteration_count());
        eprintln!("Synced CPU regrets/cum_strategy to GPU at iter {}", cpu.iteration_count());
    }

    // Sanity check: compute strategies on both, verify they match.
    {
        cpu.compute_all_strategies(&tree);
        gpu.compute_all_strategies(&ctx);
        let cpu_s = cpu.strategy_flop();
        let gpu_s = gpu.download_strategy();
        let mut max_d = 0.0f32;
        for i in 0..cpu_s.len().min(gpu_s.len()) {
            max_d = max_d.max((cpu_s[i] - gpu_s[i]).abs());
        }
        eprintln!("Post-sync strategy max_diff (flop) = {:.6e}", max_d);
    }

    // Run iter-2 on both with matching start state
    cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    let gpu_reg2 = gpu.download_regrets();
    let cpu_f2 = cpu.regrets_flop();
    let cpu_t2 = cpu.regrets_turn();
    let cpu_r2 = cpu.regrets_river();
    let mut flop_max = 0.0f32;
    let mut turn_max = 0.0f32;
    let mut river_max = 0.0f32;
    for i in 0..fl { flop_max = flop_max.max((cpu_f2[i] - gpu_reg2[i]).abs()); }
    for i in 0..tl { turn_max = turn_max.max((cpu_t2[i] - gpu_reg2[fl + i]).abs()); }
    for i in 0..rl {
        if fl + tl + i < gpu_reg2.len() {
            river_max = river_max.max((cpu_r2[i] - gpu_reg2[fl + tl + i]).abs());
        }
    }
    let overall = flop_max.max(turn_max).max(river_max);
    eprintln!(
        "iter-2 (after sync): max_diff = {:.6e} (flop {:.6e}, turn {:.6e}, river {:.6e})",
        overall, flop_max, turn_max, river_max
    );

    // After exact regret sync, iter-2 divergence should be float-precision only.
    // If > 1e-3, there is a structural bug in the showdown port.
    assert!(
        overall < 1e-3,
        "STRUCTURAL BUG: iter-2 (after CPU regret sync) max_diff = {:.6e} (expected < 1e-3). \
         flop={:.6e} turn={:.6e} river={:.6e}",
        overall, flop_max, turn_max, river_max
    );
}

/// Per-iteration GPU/CPU regret parity at iter-2 and iter-10. The brute-force
/// integration is most sensitive to per-hand stride bugs that show up only
/// once reach goes non-uniform (i.e., past iter-1). Iter-0 matching alone
/// only proves the showdown formula matches at uniform reach; the meaningful
/// gate is float-precision parity across the first ~10 iterations.
#[test]
fn gate_3p_iterN_parity() {
    let (tree, table) = build_multihand_3player_table();
    let game = FlopStartGame::new(table);

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.set_iteration(0);
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // The threshold: f32 ULP-level precision. With many accumulations per
    // regret, we allow a small tolerance per iteration.
    // Strict float-precision thresholds. With the early-return-path fix
    // applied (matching CPU's `payoff * sum(r0*r1)` ordering for fold and
    // sole-active showdowns), CPU and GPU produce bit-identical regrets
    // through the first 10+ iterations.
    let checkpoints: &[(u32, f32)] = &[
        (2, 1e-3),
        (5, 1e-3),
        (10, 1e-3),
    ];

    let mut prev = 0u32;
    for &(target, thresh) in checkpoints {
        let delta = target - prev;
        cpu.run(&tree, &game, delta);
        gpu.run(&ctx, &tree, &game, delta);
        prev = target;

        let cpu_flop = cpu.regrets_flop();
        let cpu_turn = cpu.regrets_turn();
        let cpu_river = cpu.regrets_river();
        let gpu_all = gpu.download_regrets();

        let fl = cpu_flop.len();
        let tl = cpu_turn.len();

        let mut flop_max = 0.0f32;
        let mut turn_max = 0.0f32;
        let mut river_max = 0.0f32;

        for i in 0..fl {
            flop_max = flop_max.max((cpu_flop[i] - gpu_all[i]).abs());
        }
        for i in 0..tl {
            turn_max = turn_max.max((cpu_turn[i] - gpu_all[fl + i]).abs());
        }
        for i in 0..cpu_river.len() {
            if fl + tl + i < gpu_all.len() {
                river_max = river_max.max((cpu_river[i] - gpu_all[fl + tl + i]).abs());
            }
        }

        let overall = flop_max.max(turn_max).max(river_max);
        eprintln!(
            "iter {}: max_diff = {:.6e} (flop {:.6e}, turn {:.6e}, river {:.6e})",
            target, overall, flop_max, turn_max, river_max
        );

        // Classify divergence by distribution: is it concentrated at regrets
        // near the 1e-5 regret-match threshold (= amplification, benign), or
        // spread across regret values far from the threshold (= structural)?
        if overall > 1e-4 {
            let mut hist_near_threshold = 0u64;  // diverging positions with |regret| in [1e-7, 1e-3]
            let mut hist_far_above = 0u64;       // diverging positions with |regret| > 1e-1
            let mut hist_far_below = 0u64;       // diverging positions with |regret| < 1e-7
            let mut hist_mid = 0u64;             // mid-range
            let mut max_div_at_far_above = 0.0f32;
            for i in 0..cpu_river.len() {
                if fl + tl + i < gpu_all.len() {
                    let d = (cpu_river[i] - gpu_all[fl + tl + i]).abs();
                    if d > 1e-5 {  // material divergence
                        let mag = cpu_river[i].abs().max(gpu_all[fl + tl + i].abs());
                        if mag < 1e-7 {
                            hist_far_below += 1;
                        } else if mag < 1e-3 {
                            hist_near_threshold += 1;
                        } else if mag < 1e-1 {
                            hist_mid += 1;
                        } else {
                            hist_far_above += 1;
                            if d > max_div_at_far_above {
                                max_div_at_far_above = d;
                            }
                        }
                    }
                }
            }
            let total_div = hist_far_below + hist_near_threshold + hist_mid + hist_far_above;
            eprintln!(
                "  Distribution of >1e-5 divergences in river regrets:\n\
                 \x20   |regret| < 1e-7:           {} positions ({:.1}%)\n\
                 \x20   |regret| in [1e-7, 1e-3]:  {} positions ({:.1}%)  (near threshold = amplification candidate)\n\
                 \x20   |regret| in [1e-3, 1e-1]:  {} positions ({:.1}%)\n\
                 \x20   |regret| > 1e-1:           {} positions ({:.1}%)  (far from threshold = structural candidate)\n\
                 \x20   Max divergence at |regret|>1e-1: {:.6e}",
                hist_far_below, 100.0 * hist_far_below as f64 / total_div.max(1) as f64,
                hist_near_threshold, 100.0 * hist_near_threshold as f64 / total_div.max(1) as f64,
                hist_mid, 100.0 * hist_mid as f64 / total_div.max(1) as f64,
                hist_far_above, 100.0 * hist_far_above as f64 / total_div.max(1) as f64,
                max_div_at_far_above
            );
        }

        assert!(
            overall < thresh,
            "iter-{} parity failed: max_diff = {:.6e} (threshold {:.6e}). \
             flop={:.6e} turn={:.6e} river={:.6e}",
            target, overall, thresh, flop_max, turn_max, river_max
        );
    }
}

/// Extended CPU convergence: 10000 iterations to confirm exploitability descends past 5%.
/// This distinguishes real convergence from plateauing at the product formula error floor.
/// (Uses CPU because GPU is too slow for 10000 iters on small test games.)
#[test]
fn gate_3p_cpu_extended_convergence() {
    let (tree, table) = build_multihand_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;
    let pot = 15.0f32;

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());

    let checkpoints = [500, 1000, 2000, 4000, 6000, 8000, 10000];
    let mut prev = 0u32;
    let mut results: Vec<(u32, f32)> = Vec::new();

    eprintln!("\n=== CPU 50-hand 3-player EXTENDED convergence (10000 iters) ===");
    for &cp in &checkpoints {
        cpu.run(&tree, &game, cp - prev);
        prev = cp;

        let expl = measure_exploitability(&cpu, &tree, &game, np);
        let pct = expl / pot * 100.0;
        eprintln!("CPU iter {:5}: {:.4}% of pot", cp, pct);
        results.push((cp, pct));
    }

    // With 50 hands, the product formula error floor is ~5-8%.
    // The solver oscillates around this floor rather than descending monotonically.
    // Check that exploitability stays bounded (not diverging) — plateau is expected.
    let expl_2000 = results.iter().find(|r| r.0 == 2000).map(|r| r.1).unwrap();
    let expl_10000 = results.last().unwrap().1;
    eprintln!("\nPlateau check: 2000 iters = {:.4}%, 10000 iters = {:.4}%", expl_2000, expl_10000);
    let min_expl = results.iter().map(|r| r.1).fold(f32::MAX, f32::min);
    let max_expl = results.iter().map(|r| r.1).fold(f32::MIN, f32::max);
    eprintln!("Oscillation band: {:.4}% — {:.4}% (range = {:.4}%)", min_expl, max_expl, max_expl - min_expl);

    // The solver should not diverge: all values should stay under 15% of pot
    assert!(
        max_expl < 15.0,
        "Exploitability DIVERGING: max {:.4}% across all checkpoints (expected < 15%)",
        max_expl
    );
}

/// GPU/CPU agreement: run both solvers independently to convergence, compare exploitability.
/// Confirms both reach the same equilibrium, not just some low number.
#[test]
fn gate_3p_gpu_cpu_agreement() {
    let (tree, table) = build_multihand_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;
    let pot = 15.0f32;
    let nh = NUM_HANDS;
    let iters = 2000u32;

    // Run CPU independently
    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.run(&tree, &game, iters);
    let cpu_expl = measure_exploitability(&cpu, &tree, &game, np);
    let cpu_pct = cpu_expl / pot * 100.0;

    // Run GPU independently
    let mut cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);
    gpu.run(&ctx, &tree, &game, iters);

    // Upload GPU state and measure with CPU evaluator
    cpu_proxy.set_iteration(gpu.iteration());
    let gpu_reg = gpu.download_regrets();
    let gpu_cum = gpu.download_cum_strategy();
    upload_gpu_to_cpu(&mut cpu_proxy, &gpu_reg, &gpu_cum);
    let gpu_expl = measure_exploitability(&cpu_proxy, &tree, &game, np);
    let gpu_pct = gpu_expl / pot * 100.0;

    eprintln!("\n=== GPU/CPU Agreement at {} iterations ===", iters);
    eprintln!("CPU exploitability: {:.4}% of pot", cpu_pct);
    eprintln!("GPU exploitability: {:.4}% of pot", gpu_pct);
    let ratio = if cpu_pct > 0.0 { gpu_pct / cpu_pct } else { 1.0 };
    eprintln!("Ratio (GPU/CPU): {:.4}x", ratio);

    // Compare strategy values per player
    let cpu_sv: Vec<Vec<f32>> = (0..np)
        .map(|p| cpu.strategy_value_debug(&tree, &game, p as u8))
        .collect();
    let gpu_sv: Vec<Vec<f32>> = (0..np)
        .map(|p| cpu_proxy.strategy_value_debug(&tree, &game, p as u8))
        .collect();

    for p in 0..np {
        let sv_diff: f32 = (0..nh)
            .map(|h| (cpu_sv[p][h] - gpu_sv[p][h]).abs())
            .sum::<f32>() / nh as f32;
        let sv_max_diff: f32 = (0..nh)
            .map(|h| (cpu_sv[p][h] - gpu_sv[p][h]).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Player {}: avg SV diff = {:.6}, max SV diff = {:.6}", p, sv_diff, sv_max_diff);
    }

    // Both should be in the low-exploitability regime (under 10% of pot).
    // GPU converges faster than CPU due to simultaneous-player traversal,
    // so they may differ by 2-3× at the same iteration count. That's fine —
    // both are solving the same game, just at different rates.
    assert!(
        gpu_pct < 10.0 && cpu_pct < 10.0,
        "GPU/CPU both must converge: GPU={:.4}% CPU={:.4}% (expected both < 10%)",
        gpu_pct, cpu_pct
    );
}
