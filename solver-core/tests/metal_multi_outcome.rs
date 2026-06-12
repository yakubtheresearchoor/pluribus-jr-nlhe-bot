/// Multi-outcome stage validation.
///
/// The existing metal_stage_validation tests only check ti=0, ri=0.
/// Bug #13 (sorted array pointers in batched kernel np==2 sweep using
/// global buffer instead of per-outcome offset) was latent at ti=0 because
/// the offset is zero. These tests check ti>0 and ri>0 to catch such bugs.
///
/// Coverage gap closed: bugs that manifest only at non-zero outcome indices
/// now fail these tests automatically.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::gpu_metal::GpuDcfrParams;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
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

fn build_minimal_table() -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
    let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

    let chosen_hands: Vec<u16> = vec![
        find_pair_index(card_from_str("Ah").unwrap(), card_from_str("Kh").unwrap()),
        find_pair_index(card_from_str("Qh").unwrap(), card_from_str("Jh").unwrap()),
        find_pair_index(card_from_str("Th").unwrap(), card_from_str("9h").unwrap()),
        find_pair_index(card_from_str("8h").unwrap(), card_from_str("6h").unwrap()),
    ];

    let nh = chosen_hands.len();
    let num_players = 2u8;
    let num_opp = 1;
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

    let initial_weights = vec![vec![1.0f32; nh], vec![1.0f32; nh]];
    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let mask0: u64 = (1u64 << hand_cards[h0 * 2]) | (1u64 << hand_cards[h0 * 2 + 1]);
        for h1 in 0..nh {
            let mask1: u64 = (1u64 << hand_cards[h1 * 2]) | (1u64 << hand_cards[h1 * 2 + 1]);
            if mask0 & mask1 == 0 { nc += 1.0; }
        }
    }

    let table = FlopChanceTable {
        hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
        remaining_deck: turn_cards.clone(), turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx, initial_weights, num_players,
        num_combinations: nc, river_decks,
    };
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

fn build_game() -> (FlatTree, FlopStartGame) {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    (tree, game)
}

fn max_abs_diff(a: &[f32], b: &[f32], label: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", label, a.len(), b.len());
    let mut max_diff = 0.0f32;
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let d = (av - bv).abs();
        if d > max_diff { max_diff = d; }
    }
    if max_diff > 1e-6 {
        eprintln!("  {} max_diff={:.8}", label, max_diff);
    } else {
        eprintln!("  {} max_diff={:.8} OK", label, max_diff);
    }
    max_diff
}

/// River bottom-up for ti=0, ri=1 (second river outcome for first turn card).
/// Bug #13 would cause wrong sorted arrays here.
#[test]
fn test_stage5_river_ti0_ri1() {
    let (tree, game) = build_game();
    let table = game.table();
    let nn = tree.num_nodes();
    let nh = FlopStartVectorCfr::new(&tree, table).num_hands();

    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    cpu.compute_all_strategies(&tree);
    let cpu_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 0, &cpu_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 1, &cpu_turn_reach);
    let cpu_params = DcfrParams::new(0);
    let gpu_params = GpuDcfrParams::new(0);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, 0);
    gpu.compute_reach_river(&ctx, 0, 1);
    gpu.bottom_up_river(&ctx, 0, 1, 0, &gpu_params);

    let mut cpu_cfv = vec![0.0f32; nn * nh];
    cpu.bottom_up_zone(&tree, table, 0, &cpu_river_reach, &mut cpu_cfv,
                       Zone::River, Some(0), Some(1), &cpu_params);

    let gpu_cfv_batch = gpu.download_river_cfv_batch();
    // ri=1: offset = 1 * nn * nh
    let gpu_river_cfv = &gpu_cfv_batch[nn * nh..2 * nn * nh];

    let diff = max_abs_diff(&cpu_cfv, gpu_river_cfv, "river_cfv_ti0_ri1");
    assert!(diff < 1e-3, "river CFV ti=0 ri=1 diff {:.8} > 1e-3", diff);
    eprintln!("Stage 5 (ti=0, ri=1) PASS");
}

/// River bottom-up for ti=1, ri=0 (first river outcome for second turn card).
#[test]
fn test_stage5_river_ti1_ri0() {
    let (tree, game) = build_game();
    let table = game.table();
    let nn = tree.num_nodes();
    let nh = FlopStartVectorCfr::new(&tree, table).num_hands();

    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    cpu.compute_all_strategies(&tree);
    let cpu_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 1, &cpu_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, 1, 0, &cpu_turn_reach);
    let cpu_params = DcfrParams::new(0);
    let gpu_params = GpuDcfrParams::new(0);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, 1);
    gpu.compute_reach_river(&ctx, 1, 0);
    gpu.bottom_up_river(&ctx, 1, 0, 0, &gpu_params);

    let mut cpu_cfv = vec![0.0f32; nn * nh];
    cpu.bottom_up_zone(&tree, table, 0, &cpu_river_reach, &mut cpu_cfv,
                       Zone::River, Some(1), Some(0), &cpu_params);

    let gpu_cfv_batch = gpu.download_river_cfv_batch();
    // ti=1, ri=0: the batched kernel writes to offset ri*nn*nh = 0 within the turn's slice
    // But the GPU dispatch uses ti=1 for sorted array offsets
    let gpu_river_cfv = &gpu_cfv_batch[0..nn * nh];

    let diff = max_abs_diff(&cpu_cfv, gpu_river_cfv, "river_cfv_ti1_ri0");
    assert!(diff < 1e-3, "river CFV ti=1 ri=0 diff {:.8} > 1e-3", diff);
    eprintln!("Stage 5 (ti=1, ri=0) PASS");
}

/// River bottom-up for ti=1, ri=1 (second river outcome for second turn card).
#[test]
fn test_stage5_river_ti1_ri1() {
    let (tree, game) = build_game();
    let table = game.table();
    let nn = tree.num_nodes();
    let nh = FlopStartVectorCfr::new(&tree, table).num_hands();

    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    cpu.compute_all_strategies(&tree);
    let cpu_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 1, &cpu_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, 1, 1, &cpu_turn_reach);
    let cpu_params = DcfrParams::new(0);
    let gpu_params = GpuDcfrParams::new(0);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, 1);
    gpu.compute_reach_river(&ctx, 1, 1);
    gpu.bottom_up_river(&ctx, 1, 1, 0, &gpu_params);

    let mut cpu_cfv = vec![0.0f32; nn * nh];
    cpu.bottom_up_zone(&tree, table, 0, &cpu_river_reach, &mut cpu_cfv,
                       Zone::River, Some(1), Some(1), &cpu_params);

    let gpu_cfv_batch = gpu.download_river_cfv_batch();
    let gpu_river_cfv = &gpu_cfv_batch[nn * nh..2 * nn * nh];

    let diff = max_abs_diff(&cpu_cfv, gpu_river_cfv, "river_cfv_ti1_ri1");
    assert!(diff < 1e-3, "river CFV ti=1 ri=1 diff {:.8} > 1e-3", diff);
    eprintln!("Stage 5 (ti=1, ri=1) PASS");
}

/// River chance accumulation for ti=1 (second turn card).
/// Tests all river outcomes for ti=1 accumulated together.
#[test]
fn test_stage6_river_accum_ti1() {
    let (tree, game) = build_game();
    let table = game.table();
    let nn = tree.num_nodes();
    let nh = FlopStartVectorCfr::new(&tree, table).num_hands();
    let turn_deck = &table.remaining_deck;

    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    cpu.compute_all_strategies(&tree);
    let cpu_reach = cpu.compute_reach_flop(&tree, &game);
    let gpu_params = GpuDcfrParams::new(0);
    let cpu_params = DcfrParams::new(0);

    let ti = 1;
    let tc = turn_deck[ti];
    let n_river = table.river_decks[tc as usize].len();

    let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_reach);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, ti);
    for ri in 0..n_river {
        gpu.compute_reach_river(&ctx, ti, ri);
        gpu.bottom_up_river(&ctx, ti, ri, 0, &gpu_params);
    }

    // CPU: accumulate river CFVs
    let mut cpu_river_accum = vec![0.0f32; nn * nh];
    for ri in 0..n_river {
        let cpu_river_reach = cpu.compute_reach_river(&tree, ti, ri, &cpu_turn_reach);
        let mut cfv = vec![0.0f32; nn * nh];
        cpu.bottom_up_zone(&tree, table, 0, &cpu_river_reach, &mut cfv,
                           Zone::River, Some(ti), Some(ri), &cpu_params);
        for &child_id in cpu.river_chance_children() {
            for h in 0..nh {
                let cp = table.chance_probability_river(tc, ri, h);
                cpu_river_accum[child_id as usize * nh + h] +=
                    cp * cfv[child_id as usize * nh + h];
            }
        }
    }

    gpu.chance_accumulate_river(&ctx, ti, n_river);
    let gpu_river_accum = gpu.download_river_accum();

    let diff = max_abs_diff(&cpu_river_accum, &gpu_river_accum, "river_accum_ti1");
    assert!(diff < 1e-3, "river accum ti=1 diff {:.8} > 1e-3", diff);
    eprintln!("Stage 6 (ti=1) PASS");
}

/// Turn bottom-up for ti=1 (second turn card).
/// Full river→chance→turn pipeline for ti=1.
#[test]
fn test_stage8_turn_ti1() {
    let (tree, game) = build_game();
    let table = game.table();
    let nn = tree.num_nodes();
    let nh = FlopStartVectorCfr::new(&tree, table).num_hands();
    let turn_deck = &table.remaining_deck;

    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    cpu.compute_all_strategies(&tree);
    let cpu_reach = cpu.compute_reach_flop(&tree, &game);
    let gpu_params = GpuDcfrParams::new(0);
    let cpu_params = DcfrParams::new(0);

    let ti = 1;
    let tc = turn_deck[ti];
    let n_river = table.river_decks[tc as usize].len();

    let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_reach);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, ti);
    for ri in 0..n_river {
        gpu.compute_reach_river(&ctx, ti, ri);
        gpu.bottom_up_river(&ctx, ti, ri, 0, &gpu_params);
    }

    // CPU river accumulation
    let mut cpu_river_accum = vec![0.0f32; nn * nh];
    for ri in 0..n_river {
        let cpu_river_reach = cpu.compute_reach_river(&tree, ti, ri, &cpu_turn_reach);
        let mut cfv = vec![0.0f32; nn * nh];
        cpu.bottom_up_zone(&tree, table, 0, &cpu_river_reach, &mut cfv,
                           Zone::River, Some(ti), Some(ri), &cpu_params);
        for &child_id in cpu.river_chance_children() {
            for h in 0..nh {
                let cp = table.chance_probability_river(tc, ri, h);
                cpu_river_accum[child_id as usize * nh + h] +=
                    cp * cfv[child_id as usize * nh + h];
            }
        }
    }
    let mut cpu_turn_cfv = vec![0.0f32; nn * nh];
    for &child_id in cpu.river_chance_children() {
        for h in 0..nh {
            cpu_turn_cfv[child_id as usize * nh + h] =
                cpu_river_accum[child_id as usize * nh + h];
        }
    }
    cpu.bottom_up_zone(&tree, table, 0, &cpu_turn_reach, &mut cpu_turn_cfv,
                       Zone::Turn, Some(ti), None, &cpu_params);

    // GPU: chance transitions + turn bottom-up
    gpu.chance_accumulate_river(&ctx, ti, n_river);
    gpu.chance_finalize_river(&ctx, ti);
    gpu.bottom_up_turn(&ctx, ti, 0, &gpu_params);

    let gpu_turn_cfv_batch = gpu.download_turn_cfv_batch();
    // ti=1: offset = ti * nn * nh
    let offset = ti * nn * nh;
    let gpu_turn_cfv = &gpu_turn_cfv_batch[offset..offset + nn * nh];

    let diff = max_abs_diff(&cpu_turn_cfv, gpu_turn_cfv, "turn_cfv_ti1");
    assert!(diff < 1e-3, "turn CFV ti=1 diff {:.8} > 1e-3", diff);
    eprintln!("Stage 8 (ti=1) PASS");
}
