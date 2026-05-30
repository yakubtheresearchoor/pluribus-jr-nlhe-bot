/// Stage-by-stage validation of the Metal pipeline against CPU reference.
///
/// Stages:
/// 1. Strategy computation from zero regrets (uniform)
/// 2. Flop reach computation
/// 3. Turn reach computation
/// 5. River bottom-up CFVs
/// 6. River chance accumulation
/// 7. River chance finalize → turn CFV seeding
/// 8. Turn bottom-up CFVs + regret updates

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
    let mut worst_idx = 0;
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let d = (av - bv).abs();
        if d > max_diff { max_diff = d; worst_idx = i; }
    }
    if max_diff > 1e-6 {
        eprintln!("  {} max_diff={:.8} at idx={}", label, max_diff, worst_idx);
        eprintln!("    A[{}]={:.8}  B[{}]={:.8}", worst_idx, a[worst_idx], worst_idx, b[worst_idx]);
        let lo = worst_idx.saturating_sub(2);
        let hi = (worst_idx + 2).min(a.len() - 1);
        for j in lo..=hi {
            eprintln!("    [{}] A={:.8} B={:.8} d={:.8}", j, a[j], b[j], (a[j]-b[j]).abs());
        }
    } else {
        eprintln!("  {} max_diff={:.8} OK", label, max_diff);
    }
    max_diff
}

/// Stage 1: Strategy from zero regrets should be uniform
#[test]
fn test_stage1_strategy_zero_regrets() {
    let (tree, game) = build_game();
    let table = game.table();
    let cpu = FlopStartVectorCfr::new(&tree, table);
    let ctx = MetalContext::new().expect("Metal");
    let gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    gpu.compute_all_strategies(&ctx);
    let gpu_strat = gpu.download_strategy();

    let na = tree.nodes[0].num_children as usize;
    let nh = cpu.num_hands();

    for a in 0..na {
        for h in 0..nh {
            let idx = a * nh + h;
            let val = gpu_strat[idx];
            assert!((val - 0.5).abs() < 1e-6,
                "flop strat a={} h={} = {} (expected 0.5)", a, h, val);
        }
    }
    eprintln!("Stage 1 PASS: uniform strategies from zero regrets");
}

/// Stage 2: Flop reach computation
#[test]
fn test_stage2_flop_reach() {
    let (tree, game) = build_game();
    let table = game.table();
    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    cpu.compute_all_strategies(&tree);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);

    let cpu_reach = cpu.compute_reach_flop(&tree, &game);
    gpu.compute_reach_flop(&ctx);
    let gpu_reach = gpu.download_reach();

    eprintln!("  CPU reach root[0..4] = {:?}", &cpu_reach[0..4]);
    eprintln!("  GPU reach root[0..4] = {:?}", &gpu_reach[0..4]);

    let diff = max_abs_diff(&cpu_reach, &gpu_reach, "reach_flop");
    assert!(diff < 1e-5, "flop reach diff {:.8} > 1e-5", diff);
    eprintln!("Stage 2 PASS: flop reach matches CPU");
}

/// Stage 3: Turn reach computation (ti=0)
#[test]
fn test_stage3_turn_reach() {
    let (tree, game) = build_game();
    let table = game.table();
    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    cpu.compute_all_strategies(&tree);
    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);

    let ti = 0;
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_flop_reach);
    gpu.compute_reach_turn(&ctx, ti);
    let gpu_turn_reach = gpu.download_turn_reach();

    eprintln!("  CPU turn_reach[0..4] = {:?}", &cpu_turn_reach[0..4]);
    eprintln!("  GPU turn_reach[0..4] = {:?}", &gpu_turn_reach[0..4]);

    let diff = max_abs_diff(&cpu_turn_reach, &gpu_turn_reach, "turn_reach");
    assert!(diff < 1e-5, "turn reach diff {:.8} > 1e-5", diff);
    eprintln!("Stage 3 PASS: turn reach matches CPU");
}

/// Stage 5: River bottom-up CFVs for one outcome (ti=0, ri=0)
#[test]
fn test_stage5_river_bottom_up() {
    let (tree, game) = build_game();
    let table = game.table();
    let nn = tree.num_nodes();
    let nh = FlopStartVectorCfr::new(&tree, table).num_hands();

    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    cpu.compute_all_strategies(&tree);
    let cpu_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 0, &cpu_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 0, &cpu_turn_reach);
    let gpu_params = GpuDcfrParams::new(0);
    let cpu_params = DcfrParams::new(0);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let mut cpu_cfv = vec![0.0f32; nn * nh];
    cpu.bottom_up_zone(&tree, table, 0, &cpu_river_reach, &mut cpu_cfv,
                       Zone::River, Some(0), Some(0), &cpu_params);

    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, 0);
    gpu.compute_reach_river(&ctx, 0, 0);
    gpu.bottom_up_river(&ctx, 0, 0, 0, &gpu_params);

    let gpu_cfv_batch = gpu.download_river_cfv_batch();
    let gpu_river_cfv = &gpu_cfv_batch[0..nn*nh];

    eprintln!("  CPU river cfv[0..4] = {:?}", &cpu_cfv[0..4]);
    eprintln!("  GPU river cfv[0..4] = {:?}", &gpu_river_cfv[0..4]);

    // Find first terminal node and show its CFV
    let (rz, _, _) = cpu.zone_nodes_per_level();
    eprintln!("  River zone nodes per level:");
    for level in 0..=tree.max_depth as usize {
        let count = rz.get(level).map(|v| v.len()).unwrap_or(0);
        if count > 0 { eprintln!("    level {}: {} nodes", level, count); }
    }

    eprintln!("  ti=0: tc_card={}, river_cards={:?}",
        table.remaining_deck[0], &table.river_decks[table.remaining_deck[0] as usize]);
    // Check sorted array at offset 0
    let tc_card = table.remaining_deck[0] as usize;
    let rc_card = table.river_decks[tc_card][0] as usize;
    eprintln!("  tc_card={}, rc_card={}", tc_card, rc_card);
    eprintln!("  river_sorted_str[0..8] = {:?}", &table.river_sorted_str[0..8]);
    let off = (tc_card * 52 + rc_card) * 1 * 4; // num_opp=1, nh=4
    eprintln!("  river_sorted_str at off {}: {:?}", off, &table.river_sorted_str[off..off+8]);
    for node_id in [18, 21, 23] {
        if node_id < nn && tree.nodes[node_id].node_type == 0 {
            let n = &tree.nodes[node_id];
            eprintln!("  Node {}: type={} na={} contrib={:?} folded={}",
                node_id, n.node_type, n.num_children,
                &tree.contributions[node_id*2..node_id*2+2],
                tree.get_folded_mask(node_id));
        }
    }
    let mut diffs: Vec<(usize, f32)> = (0..nn*nh)
        .map(|i| (i, (cpu_cfv[i] - gpu_river_cfv[i]).abs()))
        .filter(|&(_, d)| d > 0.01)
        .collect();
    diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("  Top 5 diffs:");
    for &(idx, d) in diffs.iter().take(5) {
        let node_id = idx / nh;
        let h = idx % nh;
        let ntype = tree.nodes[node_id].node_type;
        let ntype_str = match ntype { 0 => "TERM", 1 => "CHANCE", 2 => "PLAYER", _ => "???" };
        eprintln!("    node={}[{}] h={} type={} CPU={:.4} GPU={:.4} diff={:.4}",
            node_id, idx, h, ntype_str, cpu_cfv[idx], gpu_river_cfv[idx], d);
    }

    // Verify reach at first river chance child
    {
        let cc = cpu.river_chance_children();
        if let Some(&cc_id) = cc.first() {
            let cpu_r = &cpu_river_reach[cc_id as usize * 2 * nh..cc_id as usize * 2 * nh + 2 * nh];
            let gpu_r = gpu.download_river_reach();
            let gpu_r_at = &gpu_r[cc_id as usize * 2 * nh..cc_id as usize * 2 * nh + 2 * nh];
            eprintln!("  CPU river reach at cc[0]={}: {:?}", cc_id, cpu_r);
            eprintln!("  GPU river reach at cc[0]={}: {:?}", cc_id, gpu_r_at);
        }
    }
    {
        let gpu_strat = gpu.download_strategy();
        let gpu_reg = gpu.download_regrets();
        let (flop_stride, turn_stride, _, turn_total, _, _) = gpu.layout();
        let river_off = flop_stride + turn_total;
        eprintln!("  GPU river strat[0..8] = {:?}", &gpu_strat[river_off..river_off+8.min(gpu_strat.len()-river_off)]);
        eprintln!("  GPU river regret[0..8] = {:?}", &gpu_reg[river_off..river_off+8.min(gpu_reg.len()-river_off)]);
        // Check if regrets are truly zero
        let max_r = gpu_reg[river_off..river_off+8].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("  Max |regret| at river[0..8] = {}", max_r);
    }

    // Show reach at node 190 for both CPU and GPU
    if nn > 190 {
        let np = 2;
        eprintln!("  CPU reach at 190: {:?}", &cpu_river_reach[190*np*nh..190*np*nh+np*nh]);
        let gpu_river_reach = gpu.download_river_reach();
        eprintln!("  GPU reach at 190: {:?}", &gpu_river_reach[190*np*nh..190*np*nh+np*nh]);
    }
    for &(idx, _) in diffs.iter() {
        let node_id = idx / nh;
        if tree.nodes[node_id].node_type == 0 {
            eprintln!("    TERM node={}: CPU={:?} GPU={:?}",
                node_id, &cpu_cfv[node_id*nh..node_id*nh+nh], &gpu_river_cfv[node_id*nh..node_id*nh+nh]);
        }
    }
    // Show all terminal CFVs that differ
    for node_id in 0..nn {
        if tree.nodes[node_id].node_type != 0 { continue; }
        let cpu_h0 = cpu_cfv[node_id * nh];
        let gpu_h0 = gpu_river_cfv[node_id * nh];
        if (cpu_h0 - gpu_h0).abs() > 0.01 {
            eprintln!("    DIFF TERM node={}: CPU={:?} GPU={:?}",
                node_id, &cpu_cfv[node_id*nh..node_id*nh+nh], &gpu_river_cfv[node_id*nh..node_id*nh+nh]);
        }
    }

    let diff = max_abs_diff(&cpu_cfv, gpu_river_cfv, "river_cfv");
    assert!(diff < 1e-3, "river CFV diff {:.8} > 1e-3", diff);
    eprintln!("Stage 5 PASS: river bottom-up CFVs match CPU");
}

/// Stage 6: River chance accumulation for one turn card
#[test]
fn test_stage6_river_chance_accumulation() {
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

    let ti = 0;
    let tc = turn_deck[ti];
    let n_river = table.river_decks[tc as usize].len();

    // CPU: all river outcomes for ti=0
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_reach);

    // Create GPU solver BEFORE any bottom_up_zone calls
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, ti);
    for ri in 0..n_river {
        gpu.compute_reach_river(&ctx, ti, ri);
        gpu.bottom_up_river(&ctx, ti, ri, 0, &gpu_params);
    }

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

    // GPU already has river bottom-up results from above
    gpu.chance_accumulate_river(&ctx, ti, n_river);
    let gpu_river_accum = gpu.download_river_accum();

    eprintln!("  CPU river_accum[0..4] = {:?}", &cpu_river_accum[0..4]);
    eprintln!("  GPU river_accum[0..4] = {:?}", &gpu_river_accum[0..4]);

    let diff = max_abs_diff(&cpu_river_accum, &gpu_river_accum, "river_accum");
    assert!(diff < 1e-3, "river accum diff {:.8} > 1e-3", diff);
    eprintln!("Stage 6 PASS: river chance accumulation matches CPU");
}

/// Stage 7: Chance finalize river → turn CFV batch
#[test]
fn test_stage7_chance_finalize() {
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

    let ti = 0;
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
    gpu.chance_accumulate_river(&ctx, ti, n_river);
    gpu.chance_finalize_river(&ctx, ti);

    // CPU: compute river accum and seed turn CFV
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

    let gpu_turn_cfv = gpu.download_turn_cfv_batch();
    // The turn CFV batch is indexed by ti, so offset = ti * nn * nh
    let gpu_offset = ti * nn * nh;

    let diff: f32 = (0..nn * nh)
        .map(|i| (cpu_turn_cfv[i] - gpu_turn_cfv[gpu_offset + i]).abs())
        .fold(0.0f32, f32::max);

    assert!(diff < 1e-3, "turn CFV batch after chance finalize diff {:.8} > 1e-3", diff);
    eprintln!("Stage 7 PASS: chance finalize river → turn CFV batch matches CPU");
}

/// Stage 8: Turn bottom-up for one turn card
#[test]
fn test_stage8_turn_bottom_up() {
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

    let ti = 0;
    let tc = turn_deck[ti];
    let n_river = table.river_decks[tc as usize].len();

    // CPU: full river → accumulate → seed → turn bottom-up
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_reach);

    // Create GPU solver BEFORE any bottom_up_zone calls
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, ti);
    for ri in 0..n_river {
        gpu.compute_reach_river(&ctx, ti, ri);
        gpu.bottom_up_river(&ctx, ti, ri, 0, &gpu_params);
    }

    // Now run CPU bottom_up (which modifies CPU state)
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
    let gpu_turn_cfv = &gpu_turn_cfv_batch[0..nn*nh];

    eprintln!("  CPU turn_cfv after bottom_up[0..4] = {:?}", &cpu_turn_cfv[0..4]);
    eprintln!("  GPU turn_cfv after bottom_up[0..4] = {:?}", &gpu_turn_cfv[0..4]);

    let diff = max_abs_diff(&cpu_turn_cfv, gpu_turn_cfv, "turn_cfv_after_bu");
    assert!(diff < 1e-3, "turn CFV after bottom-up diff {:.8} > 1e-3", diff);
    eprintln!("Stage 8 PASS: turn bottom-up CFVs match CPU");
}

/// Stage 9: Chance accumulate turn → main CFV
#[test]
fn test_stage9_chance_accumulate_turn() {
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

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);

    // Run full river→turn pipeline for all turn cards
    let n_turn = gpu.n_turn();
    for ti in 0..n_turn {
        let tc = turn_deck[ti];
        let n_river = table.river_decks[tc as usize].len();
        let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_reach);
        gpu.compute_reach_turn(&ctx, ti);
        for ri in 0..n_river {
            gpu.compute_reach_river(&ctx, ti, ri);
            gpu.bottom_up_river(&ctx, ti, ri, 0, &gpu_params);
        }
        // CPU: accumulate river → turn CFV for this ti
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

        // GPU: same pipeline
        gpu.chance_accumulate_river(&ctx, ti, n_river);
        gpu.chance_finalize_river(&ctx, ti);
        gpu.bottom_up_turn(&ctx, ti, 0, &gpu_params);
    }

    // Now chance_accumulate_turn for all turn cards
    gpu.chance_accumulate_turn(&ctx);

    // CPU: accumulate turn CFVs into main CFV
    let mut cpu_flop_cfv = vec![0.0f32; nn * nh];
    for ti in 0..n_turn {
        let tc = turn_deck[ti];
        let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_reach);
        // Re-compute CPU turn CFV for this ti
        let mut cpu_river_accum = vec![0.0f32; nn * nh];
        for ri in 0..n_river_for_turn(&table, tc) {
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
        for &child_id in cpu.turn_chance_children() {
            for h in 0..nh {
                let cp = table.chance_probability_turn(ti, h);
                cpu_flop_cfv[child_id as usize * nh + h] +=
                    cp * cpu_turn_cfv[child_id as usize * nh + h];
            }
        }
    }

    let gpu_cfv = gpu.download_cfv();
    let diff = max_abs_diff(&cpu_flop_cfv, &gpu_cfv, "main_cfv_after_turn_accum");
    assert!(diff < 1e-3, "main CFV after turn accumulation diff {:.8} > 1e-3", diff);
    eprintln!("Stage 9 PASS: chance accumulate turn → main CFV matches CPU");
}

fn n_river_for_turn(table: &solver_core::solver::flop_start_game::FlopChanceTable, tc: u8) -> usize {
    table.river_decks[tc as usize].len()
}

/// Stage 10: Flop bottom-up (full pipeline minus regret update)
#[test]
fn test_stage10_flop_bottom_up() {
    let (tree, game) = build_game();
    let table = game.table();
    let nn = tree.num_nodes();
    let nh = FlopStartVectorCfr::new(&tree, table).num_hands();
    let turn_deck = &table.remaining_deck;

    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    cpu.compute_all_strategies(&tree);
    let cpu_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_params = DcfrParams::new(0);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);

    // Run the entire pipeline (river → turn → flop) for traverser=0
    let gpu_params = GpuDcfrParams::new(0);
    let n_turn = gpu.n_turn();
    for ti in 0..n_turn {
        let tc = turn_deck[ti];
        let n_river = table.river_decks[tc as usize].len();
        gpu.compute_reach_turn(&ctx, ti);
        for ri in 0..n_river {
            gpu.compute_reach_river(&ctx, ti, ri);
            gpu.bottom_up_river(&ctx, ti, ri, 0, &gpu_params);
        }
        gpu.chance_accumulate_river(&ctx, ti, n_river);
        gpu.chance_finalize_river(&ctx, ti);
        gpu.bottom_up_turn(&ctx, ti, 0, &gpu_params);
    }
    gpu.chance_accumulate_turn(&ctx);
    gpu.bottom_up_flop(&ctx, 0, &gpu_params);

    // CPU: full pipeline
    let mut cpu_cfv = vec![0.0f32; nn * nh];
    for ti in 0..n_turn {
        let tc = turn_deck[ti];
        let n_river = table.river_decks[tc as usize].len();
        let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_reach);
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
        for &child_id in cpu.turn_chance_children() {
            for h in 0..nh {
                let cp = table.chance_probability_turn(ti, h);
                cpu_cfv[child_id as usize * nh + h] +=
                    cp * cpu_turn_cfv[child_id as usize * nh + h];
            }
        }
    }
    cpu.bottom_up_zone(&tree, table, 0, &cpu_reach, &mut cpu_cfv,
                       Zone::Flop, None, None, &cpu_params);

    let gpu_cfv = gpu.download_cfv();
    let diff = max_abs_diff(&cpu_cfv, &gpu_cfv, "flop_cfv");
    assert!(diff < 1e-3, "flop CFV diff {:.8} > 1e-3", diff);
    eprintln!("Stage 10 PASS: flop bottom-up CFVs match CPU");

    // Stage 10b: Also compare regrets after the pipeline
    let gpu_regrets = gpu.download_regrets();
    let cpu_regrets_flop = cpu.regrets_flop();
    let cpu_regrets_turn = cpu.regrets_turn();
    let cpu_regrets_river = cpu.regrets_river();
    let flop_len = cpu_regrets_flop.len();
    let turn_len = cpu_regrets_turn.len();
    let regret_diff = max_abs_diff(&cpu_regrets_flop, &gpu_regrets[..flop_len], "regret_flop");
    eprintln!("  Regret diff (flop): {:.8}", regret_diff);
    if regret_diff > 1e-3 {
        // Show worst diffs
        for i in 0..flop_len {
            let d = (cpu_regrets_flop[i] - gpu_regrets[i]).abs();
            if d > 0.01 {
                eprintln!("    regrets_flop[{}] CPU={:.8} GPU={:.8} diff={:.8}", i, cpu_regrets_flop[i], gpu_regrets[i], d);
            }
        }
    }
    eprintln!("  regrets_flop[0..12] CPU={:?}", &cpu_regrets_flop[0..12.min(cpu_regrets_flop.len())]);
    eprintln!("  regrets_flop[0..12] GPU={:?}", &gpu_regrets[0..12.min(flop_len)]);
    assert!(regret_diff < 1e-3, "flop regret diff {:.8} > 1e-3", regret_diff);
    eprintln!("Stage 10b PASS: flop regrets match CPU");
}
