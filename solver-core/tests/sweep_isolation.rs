/// Isolation test for the showdown sweep in the batched kernel.
///
/// Strategy: set known inputs, run CPU sorted_sweep_showdown, then run the
/// batched kernel's terminal CFV computation on the same inputs, compare.
///
/// The test creates a minimal terminal node in a river-zone bottom-up dispatch,
/// with known reach values, sorted arrays, and hand cards. It compares the
/// sweep output (before * pot_size / nc) directly.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::gpu_metal::GpuDcfrParams;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr};
use solver_core::solver::showdown::sorted_sweep_showdown;
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

/// Test: run CPU sorted_sweep_showdown with known inputs from the river outcome (ti=0, ri=0)
/// and compare against the GPU batched kernel's terminal CFV at a simple showdown terminal.
///
/// This isolates whether the sweep function itself is wrong, or if the data
/// preparation (sorted array offsets, reach values) is wrong.
#[test]
fn test_sweep_isolation() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let table = game.table();
    let nh = 4usize;
    let num_opp = 1usize;

    let tc_card = table.remaining_deck[0] as usize;  // 3c = 4
    let rc_card = table.river_decks[tc_card][0] as usize;  // 5c = 12
    eprintln!("  tc_card={}, rc_card={}", tc_card, rc_card);

    // Get sorted arrays for this (tc, rc) from the table
    let elem_off = (tc_card * 52 + rc_card) * num_opp * nh;
    let opp_str = &table.river_sorted_str[elem_off..elem_off + num_opp * nh];
    let opp_idx = &table.river_sorted_idx[elem_off..elem_off + num_opp * nh];
    let pl_str = &table.river_sorted_str[elem_off..elem_off + nh]; // same as opp for 2p
    let pl_idx = &table.river_sorted_idx[elem_off..elem_off + nh];

    eprintln!("  opp_str: {:?}", opp_str);
    eprintln!("  opp_idx: {:?}", opp_idx);
    eprintln!("  pl_str:  {:?}", pl_str);
    eprintln!("  pl_idx:  {:?}", pl_idx);
    eprintln!("  hand_cards: {:?}", &table.hand_cards);

    // Set known reach: all hands have reach 1.0 for both players
    let reach_p0 = vec![0.25f32; nh]; // traverser
    let reach_p1 = vec![0.25f32; nh]; // opponent

    // CPU sweep
    let cpu_sweep = sorted_sweep_showdown(
        &[&reach_p1], &table.hand_cards, nh,
        opp_str, opp_idx, pl_str, pl_idx,
    );
    eprintln!("  CPU sweep: {:?}", cpu_sweep);

    // Now find a terminal node with equal contributions (simple showdown)
    // and run the full pipeline to get the GPU's CFV there
    let nn = tree.num_nodes();
    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    cpu.compute_all_strategies(&tree);

    // Find a terminal with equal contributions and no folds
    let mut target_node = None;
    for nid in 0..nn {
        if tree.nodes[nid].node_type != 0 { continue; }
        let c0 = tree.contributions[nid * 2];
        let c1 = tree.contributions[nid * 2 + 1];
        let fm = tree.get_folded_mask(nid);
        if c0 == c1 && c0 > 0 && fm == 0 && c0 < 100 {
            // Check it's in the river zone
            if cpu.zones()[nid] == solver_core::solver::flop_start_vector_cfr::Zone::River {
                target_node = Some(nid);
                break;
            }
        }
    }

    if let Some(nid) = target_node {
        let c0 = tree.contributions[nid * 2];
        let c1 = tree.contributions[nid * 2 + 1];
        eprintln!("  Target terminal: node {} contrib=[{}, {}]", nid, c0, c1);

        // Compute expected CFV: half_pot * sweep / nc
        let half_pot = tree.starting_pot as f32 / 2.0 + c0.min(c1) as f32;
        let nc = table.num_combinations as f32;
        let expected_cfv: Vec<f32> = cpu_sweep.iter().map(|&s| half_pot * s / nc).collect();
        eprintln!("  half_pot={}, nc={}", half_pot, nc);
        eprintln!("  Expected CFV (half_pot * sweep / nc): {:?}", expected_cfv);
    } else {
        eprintln!("  No simple equal-contribution terminal found in river zone");
        // Try with unequal contributions
        for nid in 0..nn {
            if tree.nodes[nid].node_type != 0 { continue; }
            let fm = tree.get_folded_mask(nid);
            if fm != 0 { continue; }
            if cpu.zones()[nid] == solver_core::solver::flop_start_vector_cfr::Zone::River {
                let c0 = tree.contributions[nid * 2];
                let c1 = tree.contributions[nid * 2 + 1];
                eprintln!("  River terminal node {}: contrib=[{}, {}]", nid, c0, c1);
            }
        }
    }

    // Now run the GPU pipeline for ti=0, ri=0 and check terminal CFVs
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    gpu.compute_all_strategies(&ctx);
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, 0);
    gpu.compute_reach_river(&ctx, 0, 0);

    let gpu_params = GpuDcfrParams::new(0);
    gpu.bottom_up_river(&ctx, 0, 0, 0, &gpu_params);

    // Check params AFTER bottom_up_river uploaded them
    {
        let params_out = gpu.debug_params(&ctx);
        let field_names = ["level_count", "num_outcomes", "cfv_batch_stride", "sorted_opp_stride",
                          "num_players", "nh", "traverser", "alpha_t", "beta_t", "gamma_t",
                          "regret_floor", "starting_pot", "num_combinations",
                          "regret_outcome_stride", "cum_outcome_stride"];
        eprintln!("\n  === GPU Params (after bottom_up_river) ===");
        for (i, name) in field_names.iter().enumerate() {
            eprintln!("    {}: {:.4}", name, params_out[i]);
        }
    }

    let gpu_cfv = gpu.download_river_cfv_batch();

    // Read debug output
    {
        let dbg = gpu.download_debug();
        eprintln!("\n  === Batched kernel diagnostic (node 18) ===");
        eprintln!("    half_pot: {:.6}", dbg[0]);
        eprintln!("    min_active: {:.6}", dbg[1]);
        eprintln!("    starting_pot: {:.6}", dbg[2]);
        eprintln!("    np: {:.6}", dbg[3]);
        eprintln!("    num_combinations: {:.6}", dbg[4]);
        eprintln!("    opp_reach: {:?}", &dbg[5..5+nh]);
        eprintln!("    sweep_result: {:?}", &dbg[5+nh..5+2*nh]);
        eprintln!("    out (before nc): {:?}", &dbg[5+2*nh..5+3*nh]);
    }

    // Show terminal CFVs
    // Show reach at key terminals (after we have cpu_river_reach)
    // (moved to after line 312)

    eprintln!("\n  GPU terminal CFVs for river zone (ti=0, ri=0):");
    // Show diagnostics for node 18
    {
        let nid = 18;
        let base = nid * nh;
        let raw_cfv = &gpu_cfv[base..base+nh];
        let raw_sweep = &gpu_cfv[base+nh..base+2*nh];
        let half_pot_val = gpu_cfv[base + 2 * nh];
        eprintln!("  DIAG node 18: raw_cfv={:?}", raw_cfv);
        eprintln!("  DIAG node 18: raw_sweep={:?}", raw_sweep);
        eprintln!("  DIAG node 18: half_pot={}", half_pot_val);
        let min_active = gpu_cfv[base + 2*nh + 1];
        let starting_pot_val = gpu_cfv[base + 2*nh + 2];
        let np_val = gpu_cfv[base + 2*nh + 3];
        eprintln!("  DIAG node 18: min_active={} starting_pot={} np={}", min_active, starting_pot_val, np_val);
    }
    for nid in 0..nn {
        if tree.nodes[nid].node_type != 0 { continue; }
        if cpu.zones()[nid] != solver_core::solver::flop_start_vector_cfr::Zone::River { continue; }
        let fm = tree.get_folded_mask(nid);
        if fm != 0 { continue; }
        let c0 = tree.contributions[nid * 2];
        let c1 = tree.contributions[nid * 2 + 1];
        let gpu_vals = &gpu_cfv[nid * nh..nid * nh + nh];
        eprintln!("    node {}: contrib=[{}, {}] CFV={:?}", nid, c0, c1, gpu_vals);
    }

    // Also compute CPU CFV using the full pipeline for comparison
    let cpu_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 0, &cpu_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 0, &cpu_turn_reach);

    // Show reach at key terminals
    let gpu_reach = gpu.download_river_reach();
    for &nid in &[15, 18, 21, 23] {
        if nid < nn && tree.nodes[nid].node_type == 0 {
            let base = nid * 2 * nh;
            eprintln!("  Reach node {}: CPU_p0={:?} CPU_p1={:?} GPU_p0={:?} GPU_p1={:?}",
                nid,
                &cpu_river_reach[base..base+nh], &cpu_river_reach[base+nh..base+2*nh],
                &gpu_reach[base..base+nh], &gpu_reach[base+nh..base+2*nh]);
        }
    }

    let cpu_params = DcfrParams::new(0);

    // Save a copy of regrets before bottom_up modifies them
    let cpu_regrets_before = cpu.regrets_river().to_vec();

    let mut cpu_cfv = vec![0.0f32; nn * nh];
    cpu.bottom_up_zone(&tree, game.table(), 0, &cpu_river_reach, &mut cpu_cfv,
                       solver_core::solver::flop_start_vector_cfr::Zone::River,
                       Some(0), Some(0), &cpu_params);

    eprintln!("\n  CPU terminal CFVs for river zone (ti=0, ri=0):");
    for nid in 0..nn {
        if tree.nodes[nid].node_type != 0 { continue; }
        if cpu.zones()[nid] != solver_core::solver::flop_start_vector_cfr::Zone::River { continue; }
        let fm = tree.get_folded_mask(nid);
        if fm != 0 { continue; }
        let c0 = tree.contributions[nid * 2];
        let c1 = tree.contributions[nid * 2 + 1];
        let cpu_vals = &cpu_cfv[nid * nh..nid * nh + nh];
        let gpu_vals = &gpu_cfv[nid * nh..nid * nh + nh];
        let diff: f32 = cpu_vals.iter().zip(gpu_vals.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("    node {}: contrib=[{}, {}] CPU={:?} GPU={:?} diff={:.4}", nid, c0, c1, cpu_vals, gpu_vals, diff);
    }

    // Now the key test: compute what the CPU sweep would give for specific nodes
    // and compare directly with GPU's terminal CFV (before nc division)
    // Compare GPU sorted arrays with CPU at the same offset
    {
        let gpu_sorted_str = gpu.download_river_sorted_str();
        let gpu_sorted_idx = gpu.download_river_sorted_idx();
        let elem_off = (tc_card * 52 + rc_card) * nh; // num_opp=1
        eprintln!("\n  === Sorted array comparison at offset {} ===", elem_off);
        eprintln!("  CPU opp_str: {:?}", opp_str);
        eprintln!("  GPU opp_str: {:?}", &gpu_sorted_str[elem_off..elem_off+nh]);
        eprintln!("  CPU opp_idx: {:?}", opp_idx);
        eprintln!("  GPU opp_idx: {:?}", &gpu_sorted_idx[elem_off..elem_off+nh]);

        // Also check pl arrays (same data for 2p)
        let gpu_pl_str = gpu_sorted_str.clone(); // same buffer
        eprintln!("  CPU pl_str:  {:?}", pl_str);
        eprintln!("  GPU pl_str:  {:?}", &gpu_pl_str[elem_off..elem_off+nh]);
    }

    // Now compare Metal sweep using actual GPU buffers
    {
        let opp_reach_18 = vec![0.125f32; nh]; // actual reach at node 18
        let gpu_sweep = gpu.debug_sweep(
            &ctx, &opp_reach_18, nh,
            opp_str, opp_idx, pl_str, pl_idx,
        );
        eprintln!("\n  === Sweep comparison ===");
        eprintln!("  CPU sweep: {:?}", cpu_sweep);
        eprintln!("  GPU sweep (manual sorted): {:?}", gpu_sweep);

        // CPU sweep with reach=[0.125]
        let cpu_sweep_125 = sorted_sweep_showdown(
            &[&opp_reach_18], &table.hand_cards, nh,
            opp_str, opp_idx, pl_str, pl_idx,
        );
        eprintln!("  CPU sweep (reach=0.125): {:?}", cpu_sweep_125);

        // Direct comparison
        for h in 0..nh {
            let d = (cpu_sweep_125[h] - gpu_sweep[h]).abs();
            eprintln!("    h={}: cpu={:.6} gpu={:.6} diff={:.6}", h, cpu_sweep_125[h], gpu_sweep[h], d);
        }
    }

    if let Some(nid) = target_node {
        let c0 = tree.contributions[nid * 2];
        let half_pot = tree.starting_pot as f32 / 2.0 + c0.min(c0) as f32;
        let nc = table.num_combinations as f32;
        let expected = cpu_sweep.iter().map(|&s| half_pot * s / nc).collect::<Vec<_>>();

        let cpu_vals = &cpu_cfv[nid * nh..nid * nh + nh];
        let gpu_vals = &gpu_cfv[nid * nh..nid * nh + nh];

        eprintln!("\n  === Detailed comparison at node {} ===", nid);
        eprintln!("  CPU sweep output: {:?}", cpu_sweep);
        eprintln!("  Expected (sweep * half_pot / nc): {:?}", expected);
        eprintln!("  CPU bottom_up result: {:?}", cpu_vals);
        eprintln!("  GPU bottom_up result: {:?}", gpu_vals);

        // Check: does CPU bottom_up match expected?
        let cpu_expected_diff: f32 = cpu_vals.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  CPU vs expected max diff: {:.6}", cpu_expected_diff);
    }
}
