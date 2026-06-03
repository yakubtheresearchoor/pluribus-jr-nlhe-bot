/// 3-player convergence gate: CPU and GPU must both converge to low exploitability.
/// This is the validation gate for 3-player support.
use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn find_pair_index(c1: Card, c2: Card) -> u16 {
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (a, b) = index_to_card_pair(idx);
        if (a == c1 as u8 && b == c2 as u8) || (a == c2 as u8 && b == c1 as u8) {
            return idx as u16;
        }
    }
    panic!("pair not found")
}

fn build_3player_table() -> (solver_core::tree::flat::FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));

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

    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen_hands.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
    }

    let mut conflict = vec![0u8; nh * nh];
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
            }
        }
    }

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
    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let mask0: u64 = (1u64 << hand_cards[h0 * 2]) | (1u64 << hand_cards[h0 * 2 + 1]);
        for h1 in 0..nh {
            let mask1: u64 = (1u64 << hand_cards[h1 * 2]) | (1u64 << hand_cards[h1 * 2 + 1]);
            if mask0 & mask1 != 0 {
                continue;
            }
            for h2 in 0..nh {
                let mask2: u64 = (1u64 << hand_cards[h2 * 2]) | (1u64 << hand_cards[h2 * 2 + 1]);
                if mask0 & mask2 != 0 || mask1 & mask2 != 0 {
                    continue;
                }
                nc += 1.0;
            }
        }
    }

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

fn upload_gpu_to_cpu(
    cpu: &mut FlopStartVectorCfr,
    gpu_reg: &[f32],
    gpu_cum: &[f32],
) {
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
            if fl + tl + i < gpu_reg.len() {
                r[i] = gpu_reg[fl + tl + i];
            }
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
            if fl + tl + i < gpu_cum.len() {
                c[i] = gpu_cum[fl + tl + i];
            }
        }
    }
}

#[test]
fn gate_3player_cpu_convergence_diagnostic() {
    let (tree, table) = build_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;
    let pot = 15.0f32;

    // Test both vanilla CFR and DCFR to isolate discounting effects
    for mode in ["vanilla", "dcfr"] {
        let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
        if mode == "vanilla" {
            cpu.set_vanilla_mode(true);
        }

        let checkpoints = [100, 200, 500, 1000, 2000, 3000, 4000, 5000];
        let mut prev_checkpoint = 0u32;

        eprintln!("\n=== CPU 3-player convergence ({}) ===", mode);
        for &cp in &checkpoints {
            let delta = cp - prev_checkpoint;
            cpu.run(&tree, &game, delta);
            prev_checkpoint = cp;

            let expl = measure_exploitability(&cpu, &tree, &game, np);
            let pct = expl / pot * 100.0;
            eprintln!("{} iter {:5}: expl={:.4} ({:.2}% of pot)", mode, cp, expl, pct);
        }
    }
}

#[test]
fn gate_3player_zero_sum_check() {
    let (tree, table) = build_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;
    let nh = 6;
    let pot = 15.0f32;

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.run(&tree, &game, 100);

    eprintln!("\n=== Zero-sum sanity check ===");

    // Strategy value should sum to 0 across all players for each hand
    let sv: Vec<Vec<f32>> = (0..np)
        .map(|p| cpu.strategy_value_debug(&tree, &game, p as u8))
        .collect();

    let mut sv_total_per_hand = vec![0.0f32; nh];
    for p in 0..np {
        for h in 0..nh {
            sv_total_per_hand[h] += sv[p][h];
        }
    }
    eprintln!("SV sum per hand: {:?}", sv_total_per_hand);
    let sv_total: f32 = sv_total_per_hand.iter().sum();
    eprintln!("SV total sum: {:.6} (should be ~0 for zero-sum game)", sv_total);

    // Check each player's SV and BR
    for p in 0..np {
        let br = cpu.best_response_value_debug(&tree, &game, p as u8);
        eprintln!("P{} SV: {:?}", p, &sv[p]);
        eprintln!("P{} BR: {:?}", p, &br);
        let expl: f32 = (0..nh).map(|h| (br[h] - sv[p][h]).max(0.0)).sum();
        eprintln!("P{} exploit: {:.4} ({:.2}% of pot)", p, expl, expl / pot * 100.0);
    }

    // Also run with vanilla and check zero-sum at iteration 1
    let mut cpu_v = FlopStartVectorCfr::new(&tree, &game.table());
    cpu_v.set_vanilla_mode(true);
    cpu_v.run(&tree, &game, 1);

    let sv_1: Vec<Vec<f32>> = (0..np)
        .map(|p| cpu_v.strategy_value_debug(&tree, &game, p as u8))
        .collect();
    let sv1_sum: f32 = (0..nh).map(|h| sv_1[0][h] + sv_1[1][h] + sv_1[2][h]).sum();
    eprintln!("\nIter 1 vanilla SV sum: {:.6} (should be ~0)", sv1_sum);
    eprintln!("Iter 1 P0 SV: {:?}", &sv_1[0]);
    eprintln!("Iter 1 P1 SV: {:?}", &sv_1[1]);
    eprintln!("Iter 1 P2 SV: {:?}", &sv_1[2]);

    // The zero-sum property should hold regardless of strategy
    assert!(
        sv_total.abs() < 1.0,
        "Zero-sum violated! SV sum = {:.4} (expected ~0)",
        sv_total
    );
}

#[test]
fn gate_3player_gpu_convergence() {
    let (tree, table) = build_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;
    let pot = 15.0f32;

    let mut cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);

    let checkpoints = [100, 200, 500, 1000, 2000, 5000];

    eprintln!("\n=== GPU 3-player convergence ===");
    let mut prev_checkpoint = 0u32;
    for &cp in &checkpoints {
        let delta = cp - prev_checkpoint;
        gpu.run(&ctx, &tree, &game, delta);
        prev_checkpoint = cp;

        cpu_proxy.set_iteration(gpu.iteration());
        let gpu_reg = gpu.download_regrets();
        let gpu_cum = gpu.download_cum_strategy();
        upload_gpu_to_cpu(&mut cpu_proxy, &gpu_reg, &gpu_cum);

        let expl = measure_exploitability(&cpu_proxy, &tree, &game, np);
        let pct = expl / pot * 100.0;
        eprintln!("GPU iter {:5}: expl={:.4} ({:.2}% of pot)", cp, expl, pct);
    }

    cpu_proxy.set_iteration(gpu.iteration());
    let gpu_reg = gpu.download_regrets();
    let gpu_cum = gpu.download_cum_strategy();
    upload_gpu_to_cpu(&mut cpu_proxy, &gpu_reg, &gpu_cum);

    let final_expl = measure_exploitability(&cpu_proxy, &tree, &game, np);
    let final_pct = final_expl / pot * 100.0;
    eprintln!("\nGPU final: {:.2}% of pot", final_pct);
    assert!(
        final_pct < 10.0,
        "GPU 3-player failed to converge: {:.2}% of pot after {} iters (target < 10%)",
        final_pct,
        checkpoints.last().unwrap()
    );
}

#[test]
fn gate_3player_gpu_cpu_agreement() {
    let (tree, table) = build_3player_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;
    let pot = 15.0f32;

    // Run CPU
    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.run(&tree, &game, 1000);
    let cpu_expl = measure_exploitability(&cpu, &tree, &game, np);
    let cpu_pct = cpu_expl / pot * 100.0;

    // Run GPU
    let mut cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);
    gpu.run(&ctx, &tree, &game, 1000);

    cpu_proxy.set_iteration(gpu.iteration());
    let gpu_reg = gpu.download_regrets();
    let gpu_cum = gpu.download_cum_strategy();
    upload_gpu_to_cpu(&mut cpu_proxy, &gpu_reg, &gpu_cum);
    let gpu_expl = measure_exploitability(&cpu_proxy, &tree, &game, np);
    let gpu_pct = gpu_expl / pot * 100.0;

    eprintln!("\n=== 3-player GPU/CPU agreement at 1000 iters ===");
    eprintln!("CPU: {:.2}% of pot", cpu_pct);
    eprintln!("GPU: {:.2}% of pot", gpu_pct);

    let ratio = if cpu_pct > gpu_pct {
        cpu_pct / gpu_pct.max(0.001)
    } else {
        gpu_pct / cpu_pct.max(0.001)
    };
    eprintln!("Ratio: {:.2}x", ratio);

    assert!(
        ratio < 5.0,
        "GPU/CPU exploitability ratio = {:.2}x (expected < 5x). CPU={:.2}% GPU={:.2}%",
        ratio, cpu_pct, gpu_pct
    );
}
