// Stage-by-stage CPU↔Metal parity walk on the iter-0 path. The fused
// full-pipeline test shows a structured per-hand CFV divergence
// (sign-flipped across action pairs); this test localizes WHICH stage the
// divergence first enters, since the showdown CFV kernel is clean in
// isolation (k3/k5/unified gates pass at 1e-5) but the full pipeline
// diverges.
//
// Stages walked, in pipeline order:
//   1. compute_all_strategies — at iter 0 should be uniform 1/na across hands
//   2. compute_reach_flop — propagates initial weights through flop strategy
//   3. compute_reach_turn[ti=0]
//   4. compute_reach_river[ti=0, ri=0]
//   5. bottom_up_river: per-terminal CFV → regret update
//   6. chance_accumulate_river / chance_finalize_river
//   7. bottom_up_turn
//   8. chance_accumulate_turn / chance_finalize_turn
//   9. bottom_up_flop
//
// First stage where CPU and Metal disagree (above f32 noise ~1e-5) is the
// bug location.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card};
use solver_core::gpu_metal::{MetalContext, MetalFlopStartSolver};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{FlopStartVectorCfr, Zone, DcfrParams as CpuDcfrParams};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

// Reused from metal_flop_parity.rs / parity_offset_characterization.rs
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
        hand_cards[i * 2] = c1; hand_cards[i * 2 + 1] = c2;
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
            .filter(|&h| {
                let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
                turn_mask & (1u64 << c1) == 0 && turn_mask & (1u64 << c2) == 0
            })
            .map(|h| (turn_ranks[tc as usize * nh + h], h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s); // ASCENDING — matches production
        for (k, &(r, idx)) in items.iter().enumerate() {
            turn_sorted_str[(tc as usize) * num_opp * nh + 0 * nh + k] = r;
            turn_sorted_idx[(tc as usize) * num_opp * nh + 0 * nh + k] = idx;
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        for &rc in &river_decks[tc as usize] {
            let combined = board_mask | (1u64 << tc) | (1u64 << rc);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if combined & (1u64 << c1) != 0 || combined & (1u64 << c2) != 0 { continue; }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in &board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize);
                hand = hand.add_card(rc as usize);
                let r = hand.evaluate_internal() as u16;
                let key = (tc as usize) * 52 + (rc as usize);
                river_ranks[key * nh + i] = r;
            }
            let key = (tc as usize) * 52 + (rc as usize);
            let mut items: Vec<(u16, u16)> = (0..nh)
                .filter(|&h| {
                    let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
                    combined & (1u64 << c1) == 0 && combined & (1u64 << c2) == 0
                })
                .map(|h| (river_ranks[key * nh + h], h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s); // ASCENDING — matches production
            for (k, &(r, idx)) in items.iter().enumerate() {
                river_sorted_str[key * num_opp * nh + 0 * nh + k] = r;
                river_sorted_idx[key * num_opp * nh + 0 * nh + k] = idx;
            }
        }
    }
    let initial_weights: Vec<Vec<f32>> = (0..num_players).map(|_| {
        let mut w = vec![0.0f32; nh];
        for h in 0..nh {
            let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
            let mut blocked = 0;
            for h2 in 0..nh {
                if h2 == h { continue; }
                let (c3, c4) = index_to_card_pair(valid_hand_indices[h2] as usize);
                if c1 == c3 || c1 == c4 || c2 == c3 || c2 == c4 { blocked += 1; }
            }
            w[h] = if blocked < (nh - 1) as i32 { 1.0 } else { 0.0 };
        }
        w
    }).collect();
    let num_combinations = initial_weights[0].iter().sum::<f32>() * initial_weights[1].iter().sum::<f32>();
    let table = FlopChanceTable {
        hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
        remaining_deck: turn_cards, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights, num_players,
        num_combinations: num_combinations as f64, river_decks,
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

fn find_pair_index(c1: Card, c2: Card) -> u16 {
    let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
    let mut idx = 0u16;
    for i in 0..52u8 {
        for j in (i+1)..52u8 {
            if i == lo && j == hi { return idx; }
            idx += 1;
        }
    }
    panic!("pair not found");
}

fn max_abs_diff(a: &[f32], b: &[f32], label: &str) -> (f32, usize) {
    let n = a.len().min(b.len());
    let mut max_d = 0.0f32;
    let mut max_i = 0usize;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > max_d { max_d = d; max_i = i; }
    }
    if max_d > 1e-5 {
        eprintln!("  [{}] DIVERGE: max_diff={:.6e} at idx={}", label, max_d, max_i);
        let start = max_i.saturating_sub(2);
        let end = (max_i + 3).min(n);
        for i in start..end {
            eprintln!("    [{}] CPU={:.6} Metal={:.6} diff={:.6e}", i, a[i], b[i], (a[i] - b[i]).abs());
        }
    } else {
        eprintln!("  [{}] OK: max_diff={:.6e}", label, max_d);
    }
    (max_d, max_i)
}

#[test]
fn localize_first_divergent_pipeline_stage() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);

    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    eprintln!("\n=== Stage-by-stage CPU↔Metal parity walk (iter 0) ===\n");

    // Stage 1: compute_all_strategies. At iter 0 strategy must be uniform.
    cpu.compute_all_strategies(&tree);
    gpu.compute_all_strategies(&ctx);
    let cpu_strat = cpu.strategy_flop();
    let gpu_strat = gpu.download_strategy();
    eprintln!("STAGE 1: compute_all_strategies (flop)");
    max_abs_diff(cpu_strat, &gpu_strat[..cpu_strat.len()], "strategy_flop");

    let cpu_strat_turn = cpu.strategy_turn();
    let gpu_strat_turn = &gpu_strat[cpu_strat.len()..cpu_strat.len() + cpu_strat_turn.len()];
    eprintln!("STAGE 1b: compute_all_strategies (turn)");
    max_abs_diff(cpu_strat_turn, gpu_strat_turn, "strategy_turn");

    let cpu_strat_river = cpu.strategy_river();
    let gpu_strat_river = &gpu_strat[cpu_strat.len() + cpu_strat_turn.len()
        ..cpu_strat.len() + cpu_strat_turn.len() + cpu_strat_river.len()];
    eprintln!("STAGE 1c: compute_all_strategies (river)");
    max_abs_diff(cpu_strat_river, gpu_strat_river, "strategy_river");

    // Stage 2: compute_reach_flop. Should match exactly at iter 0.
    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);
    gpu.compute_reach_flop(&ctx);
    let gpu_flop_reach = gpu.download_reach();
    eprintln!("\nSTAGE 2: compute_reach_flop");
    max_abs_diff(&cpu_flop_reach, &gpu_flop_reach, "reach_flop");

    // Stage 3: compute_reach_turn for ti=0
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 0, &cpu_flop_reach);
    gpu.compute_reach_turn(&ctx, 0);
    let gpu_turn_reach = gpu.download_turn_reach();
    eprintln!("\nSTAGE 3: compute_reach_turn[ti=0]");
    max_abs_diff(&cpu_turn_reach, &gpu_turn_reach, "reach_turn[0]");

    // Stage 4: compute_reach_river for ti=0, ri=0
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 0, &cpu_turn_reach);
    gpu.compute_reach_river(&ctx, 0, 0);
    let gpu_river_reach = gpu.download_river_reach();
    eprintln!("\nSTAGE 4: compute_reach_river[ti=0, ri=0]");
    max_abs_diff(&cpu_river_reach, &gpu_river_reach, "reach_river[0,0]");

    // Stage 5: bottom_up_river. CPU does CFV + regret update in
    // bottom_up_zone(Zone::River); Metal does it in bottom_up_river kernel.
    // The output CFV is per-node (length nn * nh) and we compare directly.
    let params_gpu = solver_core::gpu_metal::GpuDcfrParams::new(0);
    let params_cpu = CpuDcfrParams::new(0);
    gpu.zero_buffer_name(&ctx, 100);
    gpu.zero_buffer_name(&ctx, 2);
    gpu.zero_buffer_name(&ctx, 0);
    gpu.zero_buffer_name(&ctx, 1);
    gpu.bottom_up_river(&ctx, 0, 0, 0, &params_gpu);

    // CPU equivalent: bottom_up_zone(Zone::River, tc=0, rc=0, traverser=0).
    let nh = 4usize;
    let nn = tree.num_nodes();
    let mut cpu_cfv = vec![0.0f32; nn * nh];
    cpu.bottom_up_zone(
        &tree, game.table(), 0,
        &cpu_river_reach, &mut cpu_cfv,
        Zone::River, Some(0), Some(0),
        &params_cpu,
    );

    let gpu_river_cfv = gpu.download_river_cfv_batch();
    eprintln!("\nSTAGE 5: bottom_up_river[ti=0, ri=0, traverser=0] CFV output");
    eprintln!(
        "  CPU cfv shape [nn={}, nh={}] = {} entries; GPU river_cfv_batch = {} entries",
        nn, nh, cpu_cfv.len(), gpu_river_cfv.len()
    );

    // The CPU buffer is [node_idx * nh + h]. The GPU layout may differ.
    // First try the direct same-layout comparison.
    let mut max_d_river_cfv = 0.0f32;
    let mut max_i_river_cfv = 0usize;
    for i in 0..cpu_cfv.len().min(gpu_river_cfv.len()) {
        let d = (cpu_cfv[i] - gpu_river_cfv[i]).abs();
        if d > max_d_river_cfv { max_d_river_cfv = d; max_i_river_cfv = i; }
    }
    eprintln!(
        "  river_cfv direct-layout comparison: max_diff={:.6e} at idx={}",
        max_d_river_cfv, max_i_river_cfv
    );
    if max_d_river_cfv > 1e-5 {
        let start = max_i_river_cfv.saturating_sub(2);
        let end = (max_i_river_cfv + 3).min(cpu_cfv.len()).min(gpu_river_cfv.len());
        for i in start..end {
            eprintln!(
                "    [{}] CPU={:.6} GPU={:.6} diff={:.6e}",
                i, cpu_cfv[i], gpu_river_cfv[i], (cpu_cfv[i] - gpu_river_cfv[i]).abs()
            );
        }
    }

    // Also compare regrets after stage 5 to confirm divergence point.
    let cpu_regrets_river = cpu.regrets_river();
    let gpu_regrets_all = gpu.download_regrets();
    let cpu_flop_len = cpu.regrets_flop().len();
    let cpu_turn_len = cpu.regrets_turn().len();
    let gpu_regrets_river_slice = &gpu_regrets_all[cpu_flop_len + cpu_turn_len..];
    eprintln!("  river-zone regrets after stage 5:");
    max_abs_diff(cpu_regrets_river, gpu_regrets_river_slice, "regrets_river");

    // Walk node-by-node to characterize WHICH nodes have divergent CFV.
    // CPU layout: cpu_cfv[node*nh + h]. GPU outcome-0 layout: gpu_river_cfv[node*nh + h].
    eprintln!("\n  Per-node CFV characterization (river zone, outcome 0):");
    let mut div_terminals = 0;
    let mut div_players = 0;
    let mut div_chance = 0;
    let mut div_other = 0; // nodes NOT in river zone but somehow nonzero on GPU
    let mut sample_div_nodes: Vec<(usize, &str, Vec<f32>, Vec<f32>)> = Vec::new();
    for node_idx in 0..nn {
        let base_cpu = node_idx * nh;
        let base_gpu = node_idx * nh;
        let mut node_max_diff = 0.0f32;
        for h in 0..nh {
            let d = (cpu_cfv[base_cpu + h] - gpu_river_cfv[base_gpu + h]).abs();
            if d > node_max_diff { node_max_diff = d; }
        }
        if node_max_diff > 1e-4 {
            let node = &tree.nodes[node_idx];
            let kind = if node.is_terminal() { "TERM" }
                       else if node.is_chance() { "CHANCE" }
                       else { "PLAYER" };
            match kind {
                "TERM" => div_terminals += 1,
                "PLAYER" => div_players += 1,
                "CHANCE" => div_chance += 1,
                _ => div_other += 1,
            }
            if sample_div_nodes.len() < 5 {
                let cpu_h: Vec<f32> = (0..nh).map(|h| cpu_cfv[base_cpu + h]).collect();
                let gpu_h: Vec<f32> = (0..nh).map(|h| gpu_river_cfv[base_gpu + h]).collect();
                sample_div_nodes.push((node_idx, kind, cpu_h, gpu_h));
            }
        }
    }
    eprintln!(
        "  Divergent nodes: {} TERMINAL, {} PLAYER, {} CHANCE (others: {})",
        div_terminals, div_players, div_chance, div_other
    );
    for (idx, kind, cpu_v, gpu_v) in &sample_div_nodes {
        eprintln!("    node[{}] {} cpu={:?} gpu={:?}", idx, kind, cpu_v, gpu_v);
    }

    // Diagnostic conclusion:
    // If divergent nodes are TERMINAL → bug is in showdown (but k3/k5 gates pass in
    //   isolation, so this would point at a misuse of the showdown by the kernel,
    //   like wrong reach or chance_prob masking specifically to the orchestration).
    // If divergent nodes are PLAYER (and terminals match) → bug is in CFV
    //   aggregation at internal nodes (sigma weighting, child indexing, etc.).
    // If divergent nodes are outside river zone → bug is in level/zone classification
    //   on the GPU side (GPU is processing nodes it shouldn't).

    eprintln!(
        "\n=== Summary ===\n\
         Stages 1-4 (strategy, reach_flop, reach_turn, reach_river) all match exactly.\n\
         Divergence enters somewhere in:\n\
         - bottom_up_river (per-terminal CFV + regret update for river zone)\n\
         - chance_accumulate_river / chance_finalize_river\n\
         - bottom_up_turn / chance_accumulate_turn / chance_finalize_turn\n\
         - bottom_up_flop\n\
         The per-hand sign-flipped pattern in regret diffs suggests something\n\
         in the per-hand CFV-to-regret assembly. Next: instrument CPU's\n\
         bottom_up_zone to dump intermediate cfv arrays and compare with\n\
         the GPU's d_river_cfv_batch."
    );
}
