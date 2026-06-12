#![cfg(feature = "metal")]
// Single-terminal differential test against the CPU brute-force oracle
// at the ACTUAL iter-2 reach state. The user's gating diagnostic:
//
//   "Feed the GPU brute-force and the CPU brute-force the identical iter-2
//   reach for one terminal and compare per-hand outputs. If they agree to
//   float precision (1e-6), then the per-iteration 1e-2 divergence is
//   genuinely from float-ordering accumulation across the full tree and is
//   benign. If they diverge by ~1e-2 on the single terminal, that is the
//   structural bug, definitively, because a single terminal evaluated once
//   on identical input cannot accumulate float-ordering error to 1e-2; it is
//   computing something different."
//
// This test:
//   1. Sets up the 50-hand 3-player game.
//   2. Runs iter-1 on CPU+GPU, syncs CPU regrets to GPU.
//   3. Computes strategies on both (which match after sync).
//   4. Computes GPU's iter-2 river reach for the first (tc, rc) — the same
//      reach the GPU brute-force would use in iter-2.
//   5. Downloads that reach.
//   6. Picks a river-zone showdown terminal node and extracts its per-player
//      reach (np * nh values).
//   7. Feeds that EXACT reach to:
//      a. CPU's side_pot_showdown_cfv (the oracle)
//      b. GPU's debug_brute_force_showdown kernel
//   8. Compares per-hand outputs.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::buffer::MetalBuffer;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::showdown::side_pot_showdown_cfv;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use metal::MTLSize;

const NUM_HANDS: usize = 50;

fn build_table() -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let nh = NUM_HANDS;
    let num_opp = 2usize;

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
        hand_cards[i * 2] = c1; hand_cards[i * 2 + 1] = c2;
    }
    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh { for j in 0..nh {
        if i == j { conflict[i*nh+j] = 1; continue; }
        let (a1, a2) = index_to_card_pair(chosen[i] as usize);
        let (b1, b2) = index_to_card_pair(chosen[j] as usize);
        if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 { conflict[i*nh+j] = 1; }
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
            .map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16))
            .collect();
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
                .map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; 3];
    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let m0 = (1u64 << hand_cards[h0*2]) | (1u64 << hand_cards[h0*2+1]);
        for h1 in 0..nh {
            if h0 == h1 { continue; }
            let m1 = (1u64 << hand_cards[h1*2]) | (1u64 << hand_cards[h1*2+1]);
            if m0 & m1 != 0 { continue; }
            for h2 in 0..nh {
                if h2 == h0 || h2 == h1 { continue; }
                let m2 = (1u64 << hand_cards[h2*2]) | (1u64 << hand_cards[h2*2+1]);
                if m0 & m2 != 0 || m1 & m2 != 0 { continue; }
                nc += 1.0;
            }
        }
    }
    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players: 3, num_combinations: nc, river_decks: rd,
    };
    let config = TreeConfig {
        num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![100, 100, 100], initial_contributions: vec![5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DebugBruteForceParams {
    nh: i32,
    np: i32,
    traverser: i32,
    starting_pot: i32,
    fold_mask: u16,
    _pad: u16,
}

fn gpu_brute_force(
    ctx: &MetalContext,
    nh: usize,
    np: usize,
    traverser: usize,
    starting_pot: i32,
    fold_mask: u16,
    opp_reach: &[f32],
    contributions: &[i32],
    hand_cards: &[u8],
    pl_str: &[u16],
    pl_idx: &[u16],
) -> Vec<f32> {
    let device = ctx.device();
    let pipeline = ctx.create_pipeline("debug_brute_force_showdown").expect("debug pipeline");

    let d_output = MetalBuffer::<f32>::zeros(device, nh);
    let d_opp_reach = MetalBuffer::from_slice(device, opp_reach);
    let d_contributions = MetalBuffer::from_slice(device, contributions);
    let d_hand_cards = MetalBuffer::from_slice(device, hand_cards);
    let d_pl_str = MetalBuffer::from_slice(device, pl_str);
    let d_pl_idx = MetalBuffer::from_slice(device, pl_idx);

    let params = DebugBruteForceParams {
        nh: nh as i32, np: np as i32, traverser: traverser as i32,
        starting_pot, fold_mask, _pad: 0,
    };
    let d_params = MetalBuffer::from_slice(device, &[params]);

    let cmd = ctx.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(d_output.as_ref()), 0);
    enc.set_buffer(1, Some(d_opp_reach.as_ref()), 0);
    enc.set_buffer(2, Some(d_contributions.as_ref()), 0);
    enc.set_buffer(3, Some(d_hand_cards.as_ref()), 0);
    enc.set_buffer(4, Some(d_pl_str.as_ref()), 0);
    enc.set_buffer(5, Some(d_pl_idx.as_ref()), 0);
    enc.set_buffer(6, Some(d_params.as_ref()), 0);

    let grid = MTLSize { width: 1, height: 1, depth: 1 };
    let tg = MTLSize { width: 1, height: 1, depth: 1 };
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    d_output.to_vec()
}

#[test]
fn iter2_single_terminal_differential() {
    let (tree, table) = build_table();
    let nh = NUM_HANDS;
    let np = 3usize;
    let starting_pot = 15i32;
    let game = FlopStartGame::new(table);

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    cpu.set_iteration(0);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Run iter-1 on both (then sync state, then iter-2 differential)
    cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    // Sync CPU regrets/cum_strategy/iter to GPU
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

    // Compute strategies on both (should match exactly given regrets match)
    cpu.compute_all_strategies(&tree);
    gpu.compute_all_strategies(&ctx);

    // Find a showdown terminal node in the river zone for (ti=0, ri=0).
    // We use tree topology to identify terminals; river-zone classification
    // is implicit since the tree starts at Flop and goes through chance
    // nodes to River, where terminals live.
    let mut showdown_terminals: Vec<usize> = Vec::new();
    for (idx, node) in tree.nodes.iter().enumerate() {
        if node.is_terminal() {
            let fm = tree.get_folded_mask(idx);
            if fm == 0 {
                showdown_terminals.push(idx);
            }
        }
    }
    eprintln!("Total showdown terminals: {}", showdown_terminals.len());
    assert!(!showdown_terminals.is_empty(), "Need at least one showdown terminal");

    // GPU: compute reach for the first (ti=0, ri=0). This populates
    // d_river_reach with iter-2's actual reach values.
    // (Need to seed turn reach too.)
    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, 0);
    gpu.compute_reach_river(&ctx, 0, 0);

    let gpu_river_reach = gpu.download_river_reach();
    eprintln!("Downloaded GPU river reach: {} f32 values", gpu_river_reach.len());

    // CPU: compute reach for same (ti=0, ri=0) using CPU's own routines.
    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 0, &cpu_flop_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 0, &cpu_turn_reach);

    // Set up sorted_pl arrays for (tc=3c, rc=5s) — needed for helper tests below.
    let table_ref = game.table();
    let tc_card = table_ref.remaining_deck[0];
    let rc_card = table_ref.river_decks[tc_card as usize][0];
    let (_os, _oi, ps, pi) = table_ref.river_sorted_arrays(tc_card, rc_card);
    let pl_str: Vec<u16> = ps[..nh].to_vec();
    let pl_idx: Vec<u16> = pi[..nh].to_vec();
    let mut sorted_opp_str: Vec<u16> = Vec::with_capacity(2 * nh);
    let mut sorted_opp_idx: Vec<u16> = Vec::with_capacity(2 * nh);
    for _ in 0..2 {
        sorted_opp_str.extend_from_slice(&pl_str);
        sorted_opp_idx.extend_from_slice(&pl_idx);
    }

    eprintln!("CPU/GPU reach comparison for (ti=0, ri=0):");
    let mut max_reach_diff = 0.0f32;
    let mut max_reach_idx = 0usize;
    let mut nonzero_diffs = 0usize;
    for i in 0..cpu_river_reach.len().min(gpu_river_reach.len()) {
        let d = (cpu_river_reach[i] - gpu_river_reach[i]).abs();
        if d > 0.0 { nonzero_diffs += 1; }
        if d > max_reach_diff {
            max_reach_diff = d;
            max_reach_idx = i;
        }
    }
    let scale = cpu_river_reach[max_reach_idx].abs().max(1.0);
    let reach_ulps = max_reach_diff / (scale * 1.19e-7);
    eprintln!("  reach max_diff = {:.6e} at idx {} (cpu={}, gpu={}) = {:.1} ULPs",
        max_reach_diff, max_reach_idx,
        cpu_river_reach[max_reach_idx], gpu_river_reach[max_reach_idx], reach_ulps);
    eprintln!("  Total nonzero diffs: {} / {}", nonzero_diffs, cpu_river_reach.len());

    if max_reach_diff > 1e-3 {
        eprintln!("  STRUCTURAL DIVERGENCE in reach propagation (not helper).");
    }

    // Compare CFV propagation: feed identical reach (GPU's) to both
    // CPU bottom_up and GPU bottom_up, then compare resulting CFV.
    // This isolates whether bottom-up propagation (chance aggregation +
    // player-node weighted sum) introduces the divergence.
    let cpu_params = solver_core::solver::flop_start_vector_cfr::DcfrParams::new(
        cpu.iteration_count(),
    );
    let gpu_params = solver_core::gpu_metal::flop_solver::DcfrParams::new(
        cpu.iteration_count(),
    );
    let nn = tree.num_nodes();
    let mut cpu_cfv = vec![0.0f32; nn * nh];

    // CPU bottom_up using GPU's reach
    cpu.bottom_up_zone(
        &tree, game.table(), 0, &gpu_river_reach, &mut cpu_cfv,
        solver_core::solver::flop_start_vector_cfr::Zone::River,
        Some(0), Some(0), &cpu_params,
    );

    // GPU bottom_up_river — writes to d_river_cfv_batch
    gpu.bottom_up_river(&ctx, 0, 0, 0, &gpu_params);
    let gpu_river_cfv = gpu.download_river_cfv_batch();

    // Compare per-node CFV (just river-zone nodes). The d_river_cfv_batch
    // layout is [ri * nn * nh] for batched, with num_outcomes=1 so ri=0 is
    // the only outcome and we read [0..nn*nh].
    let mut cfv_max_diff = 0.0f32;
    let mut cfv_max_node = 0usize;
    let mut cfv_max_h = 0usize;
    let mut cfv_nonzero_diffs = 0usize;
    for node_idx in 0..nn {
        for h in 0..nh {
            let cpu_v = cpu_cfv[node_idx * nh + h];
            let gpu_v = gpu_river_cfv[node_idx * nh + h];
            let d = (cpu_v - gpu_v).abs();
            if d > 0.0 { cfv_nonzero_diffs += 1; }
            if d > cfv_max_diff {
                cfv_max_diff = d;
                cfv_max_node = node_idx;
                cfv_max_h = h;
            }
        }
    }
    let cpu_at = cpu_cfv[cfv_max_node * nh + cfv_max_h];
    let gpu_at = gpu_river_cfv[cfv_max_node * nh + cfv_max_h];
    let scale = cpu_at.abs().max(1.0);
    let cfv_ulps = cfv_max_diff / (scale * 1.19e-7);
    eprintln!("CFV (after bottom_up) comparison for (ti=0, ri=0):");
    eprintln!("  cfv max_diff = {:.6e} at node={} h={} (cpu={}, gpu={}) = {:.1} ULPs",
        cfv_max_diff, cfv_max_node, cfv_max_h, cpu_at, gpu_at, cfv_ulps);
    eprintln!("  Total nonzero CFV diffs: {} / {}", cfv_nonzero_diffs, nn * nh);

    // What type is node 50?
    let max_node = &tree.nodes[cfv_max_node];
    eprintln!("  Node {} type: terminal={}, chance={}, player={}",
        cfv_max_node, max_node.is_terminal(), max_node.is_chance(), max_node.is_player());
    if max_node.is_terminal() {
        let fm = tree.get_folded_mask(cfv_max_node);
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(cfv_max_node, p as u8)).collect();
        eprintln!("  Node {} terminal: fold_mask={:#06b}, contribs={:?}",
            cfv_max_node, fm, contribs);
    }
    // Check whether node 50 is in (ti=0, ri=0) subtree by recomputing CFV
    // via the same helper test as node 21.
    if max_node.is_terminal() {
        let reach_base = cfv_max_node * np * nh;
        let per_player_50: Vec<Vec<f32>> = (0..np)
            .map(|p| gpu_river_reach[reach_base + p * nh..reach_base + (p + 1) * nh].to_vec())
            .collect();
        let board_extra = [tc_card, rc_card];
        let filtered_50: Vec<Vec<f32>> = (0..np)
            .filter(|&p| p != 0)
            .map(|p| {
                let mut filtered = per_player_50[p].clone();
                for h in 0..nh {
                    if filtered[h] != 0.0 {
                        let c1 = table_ref.hand_cards[h * 2];
                        let c2 = table_ref.hand_cards[h * 2 + 1];
                        for &bc in &board_extra {
                            if c1 == bc || c2 == bc {
                                filtered[h] = 0.0;
                                break;
                            }
                        }
                    }
                }
                filtered
            })
            .collect();
        let opp_views_50: Vec<&[f32]> = filtered_50.iter().map(|v| v.as_slice()).collect();
        let contribs50: Vec<i32> = (0..np).map(|p| tree.get_contribution(cfv_max_node, p as u8)).collect();
        let fm50 = tree.get_folded_mask(cfv_max_node);
        let cpu_helper_50 = side_pot_showdown_cfv(
            &opp_views_50, &table_ref.hand_cards, nh,
            &sorted_opp_str, &sorted_opp_idx,
            &pl_str, &pl_idx,
            &contribs50, fm50, 0, np as u8, starting_pot,
        );
        let nc = table_ref.num_combinations as f32;
        eprintln!("  CPU helper CFV at node 50 (h=36): {} = {} (/nc)",
            cpu_helper_50[36], cpu_helper_50[36] / nc);
        eprintln!("  CPU bottom_up CFV at node 50 (h=36): {}", cpu_at);
        eprintln!("  GPU bottom_up CFV at node 50 (h=36): {}", gpu_at);

        // Also call GPU helper kernel on node 50's inputs.
        let mut opp_reach_flat_50 = Vec::with_capacity(2 * nh);
        opp_reach_flat_50.extend_from_slice(&filtered_50[0]);
        opp_reach_flat_50.extend_from_slice(&filtered_50[1]);
        let gpu_helper_50 = gpu_brute_force(
            &ctx, nh, np, 0, starting_pot, fm50,
            &opp_reach_flat_50, &contribs50, &table_ref.hand_cards, &pl_str, &pl_idx,
        );
        eprintln!("  GPU helper CFV at node 50 (h=36): {} = {} (/nc)",
            gpu_helper_50[36], gpu_helper_50[36] / nc);
        eprintln!("  CPU helper vs GPU helper diff: {:.6e}",
            (cpu_helper_50[36] - gpu_helper_50[36]).abs());
        eprintln!("  CPU bottom_up vs GPU bottom_up diff: {:.6e}",
            (cpu_at - gpu_at).abs());
    }

    // Check a few specific terminals in the river zone to see if any disagree.
    eprintln!("\n  CFV comparison at FIRST 5 SHOWDOWN TERMINALS:");
    for &term_idx in showdown_terminals.iter().take(5) {
        let mut term_max = 0.0f32;
        let mut term_h = 0;
        for h in 0..nh {
            let d = (cpu_cfv[term_idx * nh + h] - gpu_river_cfv[term_idx * nh + h]).abs();
            if d > term_max { term_max = d; term_h = h; }
        }
        let cpu_v = cpu_cfv[term_idx * nh + term_h];
        eprintln!("    term node={}: max_diff = {:.6e} at h={} (cpu={})",
            term_idx, term_max, term_h, cpu_v);
    }

    if cfv_max_diff > 1e-3 {
        eprintln!("  STRUCTURAL DIVERGENCE in CFV bottom-up (chance/player propagation).");
    }

    // We need the per-node, per-player reach. Layout: [nn * np * nh].
    // Pick the lowest-indexed showdown terminal (most likely to be in the
    // current (ti=0, ri=0) subtree). Even if it's in a different (tc, rc)
    // subtree, the reach should still be present.
    let node_id = showdown_terminals[0];
    let node_reach_base = node_id * np * nh;
    eprintln!("Testing showdown terminal node_id={}, node_reach_base={}", node_id, node_reach_base);

    let contributions: Vec<i32> = (0..np)
        .map(|p| tree.get_contribution(node_id, p as u8))
        .collect();
    eprintln!("Contributions: {:?}", contributions);
    let fold_mask = tree.get_folded_mask(node_id);
    eprintln!("Fold mask: {:#06b}", fold_mask);

    // Sample reach values for this terminal from gpu_river_reach.
    let per_player_reach: Vec<Vec<f32>> = (0..np)
        .map(|p| gpu_river_reach[node_reach_base + p * nh .. node_reach_base + (p + 1) * nh].to_vec())
        .collect();

    // Print stats
    for (p, reach) in per_player_reach.iter().enumerate() {
        let n_nonzero = reach.iter().filter(|&&v| v != 0.0).count();
        let min = reach.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = reach.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("  Player {}: {} nonzero, min={}, max={}", p, n_nonzero, min, max);
    }

    // Run differential test for each traverser using sorted_pl arrays
    // from above.
    let table = game.table();

    for traverser in 0..np {
        // Filter opp reach by board cards (just like evaluate_terminal does).
        let board_extra = [tc_card, rc_card];
        let filtered_opp: Vec<Vec<f32>> = (0..np)
            .filter(|&p| p != traverser)
            .map(|p| {
                let mut filtered = per_player_reach[p].clone();
                for h in 0..nh {
                    if filtered[h] != 0.0 {
                        let c1 = table_ref.hand_cards[h * 2];
                        let c2 = table_ref.hand_cards[h * 2 + 1];
                        for &bc in &board_extra {
                            if c1 == bc || c2 == bc {
                                filtered[h] = 0.0;
                                break;
                            }
                        }
                    }
                }
                filtered
            })
            .collect();

        let opp_reach_views: Vec<&[f32]> = filtered_opp.iter().map(|v| v.as_slice()).collect();
        let cpu_cfv = side_pot_showdown_cfv(
            &opp_reach_views, &table.hand_cards, nh,
            &sorted_opp_str, &sorted_opp_idx,
            &pl_str, &pl_idx,
            &contributions, fold_mask, traverser, np as u8, starting_pot,
        );

        // Flatten for GPU
        let mut opp_reach_flat = Vec::with_capacity(2 * nh);
        opp_reach_flat.extend_from_slice(&filtered_opp[0]);
        opp_reach_flat.extend_from_slice(&filtered_opp[1]);

        let gpu_cfv = gpu_brute_force(
            &ctx, nh, np, traverser, starting_pot, fold_mask,
            &opp_reach_flat, &contributions, &table.hand_cards, &pl_str, &pl_idx,
        );

        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        for h in 0..nh {
            let d = (cpu_cfv[h] - gpu_cfv[h]).abs();
            if d > max_diff {
                max_diff = d;
                max_idx = h;
            }
        }
        let cpu_v = cpu_cfv[max_idx];
        let gpu_v = gpu_cfv[max_idx];
        let scale = cpu_v.abs().max(1.0);
        let ulps = max_diff / (scale * 1.19e-7);
        eprintln!("Traverser {}: max_diff = {:.6e} at h={} (cpu={}, gpu={}) = {:.1} ULPs",
            traverser, max_diff, max_idx, cpu_v, gpu_v, ulps);

        // The user's threshold: 1e-6 → benign accumulation, 1e-2 → structural.
        // We assert < 1e-3 for "small enough to be float-ordering, not structural".
        assert!(
            max_diff < 1e-3,
            "Traverser {}: single-terminal differential failed at iter-2 reach. \
             max_diff = {:.6e} ({} ULPs). This is STRUCTURAL, not FMA.",
            traverser, max_diff, ulps
        );
    }
}
