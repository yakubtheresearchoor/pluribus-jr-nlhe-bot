// Step 2.A.2 trace: localize the source of iter-1 1.1e-6 ULP noise on
// K=12 sigmoid game (same as stratum 1).
//
// The bug is NOT in the `if pc==0` fix (both sides have it).
// The bug is NOT random f32 noise (4 orders-of-magnitude amplification
// in 9 iters proves it's a compounding bug).
//
// This test:
// 1. Builds the K=12 sigmoid game (same as stratum 1)
// 2. Runs CPU bottom_up_zone for River at (ti=0, ri=0) manually
// 3. Runs GPU bottom_up_river at (ti=0, ri=0) manually
// 4. Compares CFV per node-type at (count of differing entries, max_abs)
//    to find WHICH stage produces the iter-1 noise.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::{DcfrParams as GpuDcfrParams, MetalFlopStartSolver};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// K=12 sigmoid game — same as stratum 1.
fn build_k12_sigmoid_game() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["Ah", "Kd", "7c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 2u8;
    let k = 12usize;

    use solver_core::hand::eval::Hand;
    let mut all_with_strength: Vec<(u16, u16)> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        all_with_strength.push((h.evaluate_internal() as u16, idx as u16));
    }
    all_with_strength.sort_by_key(|&(s, _)| s);
    let step = all_with_strength.len() / k;
    let chosen: Vec<u16> = (0..k).map(|i| all_with_strength[i * step].1).collect();

    let mut ranges: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for (rank_idx, &hi) in chosen.iter().enumerate() {
        let strength_frac = rank_idx as f32 / k as f32;
        let p0_weight = (strength_frac - 0.3).max(0.05) * 1.5;
        let p0_weight = p0_weight.min(1.0);
        let p1_weight = 0.6 + 0.4 * strength_frac;
        let (c1, c2) = index_to_card_pair(hi as usize);
        let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
        let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
        ranges[0][pair_idx] = p0_weight;
        ranges[1][pair_idx] = p1_weight;
    }
    let turn_cards: Vec<u8> = vec![
        card_from_str("Td").unwrap() as u8,
        card_from_str("3s").unwrap() as u8,
    ];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("4h").unwrap() as u8,
        card_from_str("Qc").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("2s").unwrap() as u8,
        card_from_str("Js").unwrap() as u8,
    ];
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, num_players, &chosen, &turn_cards, &river_decks,
    );
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 6,
        starting_stacks: vec![50, 50], initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = FlopStartGame::new(table);
    (tree, game)
}

#[test]
#[ignore = "2.A.2 noise source localizer"]
fn iter1_noise_source_per_node_type_k12() {
    let (tree, game) = build_k12_sigmoid_game();
    let ctx = MetalContext::new().expect("Metal");
    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let nh = 12usize;
    let nn = tree.num_nodes();
    let ti = 0usize;
    let ri = 0usize;
    let traverser = 0u8;

    cpu.compute_all_strategies(&tree);
    gpu.compute_all_strategies(&ctx);

    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_flop_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, ti, ri, &cpu_turn_reach);

    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, ti);
    gpu.compute_reach_river(&ctx, ti, ri);

    // Compare reach at river bit-exactly
    let gpu_river_reach = gpu.download_river_reach();
    let mut reach_bit_diffs = 0usize;
    let mut reach_max = 0.0f32;
    let n_reach = cpu_river_reach.len().min(gpu_river_reach.len());
    for i in 0..n_reach {
        if cpu_river_reach[i].to_bits() != gpu_river_reach[i].to_bits() {
            reach_bit_diffs += 1;
            let d = (cpu_river_reach[i] - gpu_river_reach[i]).abs();
            if d > reach_max { reach_max = d; }
        }
    }
    eprintln!("\nREACH river at (ti={}, ri={}): {} / {} bit-different (max_abs={:.6e})",
        ti, ri, reach_bit_diffs, n_reach, reach_max);

    // Compute CFV
    let mut cpu_cfv = vec![0.0f32; nn * nh];
    let cpu_params = DcfrParams::new(0);
    cpu.bottom_up_zone(
        &tree, game.table(), traverser, &cpu_river_reach, &mut cpu_cfv,
        Zone::River, Some(ti), Some(ri), &cpu_params,
    );

    let gpu_params = GpuDcfrParams::new(0);
    gpu.bottom_up_river(&ctx, ti, ri, traverser as u32, &gpu_params);
    let gpu_river_cfv_batch = gpu.download_river_cfv_batch();
    let gpu_cfv_slot = &gpu_river_cfv_batch[ri * nn * nh..(ri + 1) * nn * nh];

    // Categorize by node type
    let mut terminal_bit_diffs = 0usize;
    let mut chance_bit_diffs = 0usize;
    let mut player_bit_diffs = 0usize;
    let mut terminal_max = 0.0f32;
    let mut chance_max = 0.0f32;
    let mut player_max = 0.0f32;
    let mut terminal_count = 0usize;
    let mut chance_count = 0usize;
    let mut player_count = 0usize;
    let mut term_first_diff = None;
    for node_id in 0..nn {
        let t = tree.nodes[node_id].node_type;
        for h in 0..nh {
            let cv = cpu_cfv[node_id * nh + h];
            let gv = gpu_cfv_slot[node_id * nh + h];
            let bit_diff = cv.to_bits() != gv.to_bits();
            let d = (cv - gv).abs();
            match t {
                0 => {
                    terminal_count += 1;
                    if bit_diff {
                        terminal_bit_diffs += 1;
                        if d > terminal_max { terminal_max = d; }
                        if term_first_diff.is_none() {
                            term_first_diff = Some((node_id, h, cv, gv));
                        }
                    }
                }
                1 => {
                    chance_count += 1;
                    if bit_diff {
                        chance_bit_diffs += 1;
                        if d > chance_max { chance_max = d; }
                    }
                }
                2 => {
                    player_count += 1;
                    if bit_diff {
                        player_bit_diffs += 1;
                        if d > player_max { player_max = d; }
                    }
                }
                _ => {}
            }
        }
    }

    eprintln!("\nCFV after bottom_up_river(ti={}, ri={}, traverser={}) on K=12 sigmoid game:", ti, ri, traverser);
    eprintln!("  TERMINAL (type 0): {} / {} bit-different entries, max_abs={:.6e}",
        terminal_bit_diffs, terminal_count, terminal_max);
    eprintln!("  CHANCE   (type 1): {} / {} bit-different entries, max_abs={:.6e}",
        chance_bit_diffs, chance_count, chance_max);
    eprintln!("  PLAYER   (type 2): {} / {} bit-different entries, max_abs={:.6e}",
        player_bit_diffs, player_count, player_max);

    if let Some((node, h, cv, gv)) = term_first_diff {
        eprintln!("\nFirst diverging TERMINAL: node {}, h{}: CPU={:.10}, GPU={:.10}",
            node, h, cv, gv);
        eprintln!("  CPU bits: {:032b}", cv.to_bits());
        eprintln!("  GPU bits: {:032b}", gv.to_bits());
        eprintln!("  ULP diff: {}", (cv.to_bits() as i64 - gv.to_bits() as i64).abs());
        let np = tree.num_players as usize;
        let contribs: Vec<i32> = (0..np).map(|p| tree.contributions[node * np + p]).collect();
        let n = &tree.nodes[node];
        eprintln!("  Node {}: type={}, contributions={:?}, num_children={}",
            node, n.node_type, contribs, n.num_children);
    }

    // Dump up to 5 diverging terminals with contributions
    let mut shown = 0usize;
    eprintln!("\nFirst few diverging terminals (node_id, contributions, fold_mask):");
    for node_id in 0..nn {
        if tree.nodes[node_id].node_type != 0 { continue; }
        let mut node_diverges = false;
        for h in 0..nh {
            let cv = cpu_cfv[node_id * nh + h];
            let gv = gpu_cfv_slot[node_id * nh + h];
            if cv.to_bits() != gv.to_bits() { node_diverges = true; break; }
        }
        if node_diverges {
            let np = tree.num_players as usize;
            let contribs: Vec<i32> = (0..np).map(|p| tree.contributions[node_id * np + p]).collect();
            eprintln!("  node {}: contributions={:?}", node_id, contribs);
            shown += 1;
            if shown >= 5 { break; }
        }
    }

    if terminal_bit_diffs > 0 {
        eprintln!("\n→ Bug is in SHOWDOWN (terminal CFV not bit-exact under non-uniform reach).");
    } else if player_bit_diffs > 0 {
        eprintln!("\n→ Terminals bit-exact, but internal player nodes diverge.");
        eprintln!("  Bug is in CFV AGGREGATION (cfv_avg = sum strategy × cfv[child]) under non-uniform inputs.");
    } else {
        eprintln!("\n→ No CFV divergence found at this terminal. Bug is elsewhere.");
    }
}
