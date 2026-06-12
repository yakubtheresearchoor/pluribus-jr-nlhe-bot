// Step 2.A.1: faithful Rust translation of GPU's HU showdown branch.
//
// Plan: implement the HU branch of multiway_brute_force_showdown (lines
// 142-552 of vcfr.metal) in pure Rust, run it with node-28's inputs at
// nh=1176, compare to:
//   - CPU side_pot_showdown_cfv (returns 329,728)
//   - GPU debug helper (returns -3,465)
//
// Outcomes:
//   A. Rust translation matches GPU (-3,465): bug is in the LOGIC and pure-
//      Rust → I can debug freely with per-g_a trace.
//   B. Rust translation matches CPU (329,728): bug is GPU-specific (memory
//      layout, register spilling, thread-local arrays at nh=1176).
//   C. Rust translation matches NEITHER: I misread the kernel; need to
//      re-read carefully.
//
// If outcome A, the test ALSO dumps per-g_a survival count and sign at
// h=0 so the bug location is visible without further iteration.

#![cfg(feature = "metal")]

use metal::MTLSize;
use solver_core::card::{
    card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS,
};
use solver_core::gpu_metal::buffer::MetalBuffer;
use solver_core::gpu_metal::{MetalContext, MetalFlopStartSolver};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::showdown::side_pot_showdown_cfv;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

#[repr(C)]
#[derive(Clone, Copy)]
struct DebugBruteForceParams {
    nh: i32, np: i32, traverser: i32, starting_pot: i32,
    fold_mask: u16, _pad: u16,
    rake_rate: f32, rake_cap: f32, flop_seen: i32,
}

fn gpu_brute_force(
    ctx: &MetalContext,
    nh: usize, np: usize, traverser: usize,
    starting_pot: i32, fold_mask: u16,
    opp_reach: &[f32], contributions: &[i32],
    hand_cards: &[u8], pl_str: &[u16], pl_idx: &[u16],
) -> Vec<f32> {
    let device = ctx.device();
    let pipeline = ctx.create_pipeline("debug_brute_force_showdown")
        .expect("debug pipeline");
    let d_output = MetalBuffer::<f32>::zeros(device, nh);
    let d_opp_reach = MetalBuffer::from_slice(device, opp_reach);
    let d_contributions = MetalBuffer::from_slice(device, contributions);
    let d_hand_cards = MetalBuffer::from_slice(device, hand_cards);
    let d_pl_str = MetalBuffer::from_slice(device, pl_str);
    let d_pl_idx = MetalBuffer::from_slice(device, pl_idx);
    let params = DebugBruteForceParams {
        nh: nh as i32, np: np as i32, traverser: traverser as i32,
        starting_pot, fold_mask, _pad: 0,
        rake_rate: 0.0, rake_cap: 0.0, flop_seen: 0,
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

/// Faithful Rust translation of multiway_brute_force_showdown HU branch
/// (num_opp==1) — vcfr.metal lines 142-552.
///
/// Returns (cfv[nh], traces_at_h0[nh]) where traces_at_h0[g_a] records
/// the per-g_a state for h=0 (survival flags, strength comparison, term
/// contribution, accumulator after).
#[derive(Clone, Debug, Default)]
struct GaTrace {
    survived_blocking: bool,
    ra: f32,
    survived_reach: bool,
    h_str: u16,
    s_a: u16,
    /// +1 = traverser wins (h_str > s_a), 0 = tie, -1 = loses
    win_lose_tie: i32,
    cash: f32,
    net: f32,
    term: f32,       // ra * net
    accum_after: f32,
}

fn rust_translation_of_gpu_hu(
    nh: usize, np: usize, traverser: usize,
    starting_pot: i32, fold_mask: u16,
    contributions: &[i32], opp_reach_local: &[f32],  // [num_opp * nh]
    hand_cards: &[u8],
    sorted_pl_strength: &[u16], sorted_pl_indices: &[u16],
    rake_rate: f32, rake_cap: f32, flop_seen: bool,
    trace_h: Option<usize>,
) -> (Vec<f32>, Vec<GaTrace>) {
    let num_opp = np - 1;
    let c_t = contributions[traverser];
    let eff_rake_rate = if flop_seen { rake_rate } else { 0.0 };
    let eff_rake_cap  = if flop_seen { rake_cap  } else { 0.0 };

    // hand_strength[1326] uninitialized (per kernel); writes via sorted_pl_indices.
    let mut hand_strength = vec![0u16; 1326];
    for si in 0..nh {
        hand_strength[sorted_pl_indices[si] as usize] = sorted_pl_strength[si];
    }

    // Sorted+deduped contribution levels (max 8).
    let mut levels = [0i32; 8];
    let mut num_levels = 0usize;
    for p in 0..np {
        let c = contributions[p];
        let mut found = false;
        for l in 0..num_levels {
            if levels[l] == c { found = true; break; }
        }
        if !found && num_levels < 8 {
            levels[num_levels] = c;
            num_levels += 1;
        }
    }
    // Insertion sort the levels (mirror Metal's bubble sort).
    for i in 0..num_levels.saturating_sub(1) {
        for j in (i + 1)..num_levels {
            if levels[j] < levels[i] {
                levels.swap(i, j);
            }
        }
    }

    let traverser_stake = starting_pot as f32 / np as f32 + c_t as f32;
    let traverser_folded = (fold_mask & (1u16 << traverser)) != 0;

    // HU branch: num_opp == 1.
    assert_eq!(num_opp, 1, "this translation handles HU only");
    let opp_a = if traverser == 0 { 1 } else { 0 };
    let reach_a = &opp_reach_local[0 * nh..(0 + 1) * nh];
    let c_opp_a = contributions[opp_a];
    let a_folded = (fold_mask & (1u16 << opp_a)) != 0;

    // K=1 HU main-pot rake (computed once before h-loop).
    let hu_main_pot_amount = if num_levels == 0 {
        starting_pot
    } else {
        let mut num_main_contributors = 0;
        for p in 0..np {
            if contributions[p] >= levels[0] {
                num_main_contributors += 1;
            }
        }
        levels[0] * num_main_contributors + starting_pot
    };
    let hu_main_pot_rake = (hu_main_pot_amount as f32 * eff_rake_rate)
        .min(eff_rake_cap).max(0.0);

    let mut out = vec![0.0f32; nh];
    let mut traces = vec![GaTrace::default(); nh];

    for h in 0..nh {
        let hc1 = hand_cards[h * 2] as i32;
        let hc2 = hand_cards[h * 2 + 1] as i32;
        let h_str = hand_strength[h];
        let mut accum = 0.0f32;

        for g_a in 0..nh {
            let g_ac1 = hand_cards[g_a * 2] as i32;
            let g_ac2 = hand_cards[g_a * 2 + 1] as i32;

            // Blocking check.
            let blocked = g_ac1 == hc1 || g_ac1 == hc2 || g_ac2 == hc1 || g_ac2 == hc2;
            let survived_blocking = !blocked;
            let mut trace = GaTrace::default();
            trace.survived_blocking = survived_blocking;

            if blocked {
                if trace_h == Some(h) { traces[g_a] = trace; }
                continue;
            }

            let ra = reach_a[g_a];
            trace.ra = ra;
            trace.survived_reach = ra != 0.0;
            if ra == 0.0 {
                if trace_h == Some(h) { traces[g_a] = trace; }
                continue;
            }
            let s_a = hand_strength[g_a];
            trace.h_str = h_str;
            trace.s_a = s_a;
            trace.win_lose_tie = match h_str.cmp(&s_a) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Less => -1,
            };

            let net = if traverser_folded {
                -traverser_stake
            } else {
                let mut cash = 0.0f32;
                let mut prev_l = 0i32;
                for li in 0..num_levels {
                    let lev = levels[li];
                    let pc = lev - prev_l;
                    if pc == 0 { prev_l = lev; continue; }
                    let mut num_contrib = 0;
                    for p in 0..np {
                        if contributions[p] >= lev { num_contrib += 1; }
                    }
                    let mut pot_l = (pc * num_contrib) as f32;
                    if li == 0 { pot_l += starting_pot as f32; }
                    let trav_elig = c_t >= lev;
                    let a_elig = !a_folded && c_opp_a >= lev;
                    if !trav_elig { prev_l = lev; continue; }
                    let pot_after_rake = if li == 0 {
                        pot_l - hu_main_pot_rake
                    } else {
                        pot_l
                    };
                    if !a_elig {
                        cash += pot_after_rake;
                    } else {
                        let max_str = h_str.max(s_a);
                        let mut tied = 0i32;
                        if h_str == max_str { tied += 1; }
                        if s_a == max_str { tied += 1; }
                        if h_str == max_str {
                            cash += pot_after_rake / tied as f32;
                        }
                    }
                    prev_l = lev;
                }
                trace.cash = cash;
                cash - traverser_stake
            };
            trace.net = net;
            trace.term = ra * net;
            accum += trace.term;
            trace.accum_after = accum;
            if trace_h == Some(h) { traces[g_a] = trace; }
        }
        out[h] = accum;
    }
    (out, traces)
}

mod helper {
    use super::*;
    pub fn build_full_nh_4board_game(stacks: i32, pot: i32)
        -> (FlatTree, FlopStartGame)
    {
        let board: Vec<Card> = ["2h", "7d", "Ks"].iter()
            .map(|s| card_from_str(s).unwrap()).collect();
        let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
        let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));
        let mut chosen: Vec<u16> = Vec::new();
        for idx in 0..NUM_POSSIBLE_HANDS as u16 {
            let (c1, c2) = index_to_card_pair(idx as usize);
            if board_mask & (1u64 << c1) != 0 { continue; }
            if board_mask & (1u64 << c2) != 0 { continue; }
            chosen.push(idx);
        }
        let nh = chosen.len();
        let num_players = 2u8;
        let num_opp = 1usize;
        let valid_hand_indices = chosen.clone();
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
            // BUG FIX 2026-06-05: same as river — include all nh hands, no filter.
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| (turn_ranks[tc as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                for (si, &(str_, idx)) in items.iter().enumerate() {
                    turn_sorted_str[tc as usize * num_opp * nh + oi * nh + si] = str_;
                    turn_sorted_idx[tc as usize * num_opp * nh + oi * nh + si] = idx;
                }
            }
        }
        let mut river_ranks = vec![0u16; 52 * 52 * nh];
        let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
        let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
        for &tc in &turn_cards {
            for &rc in &river_decks[tc as usize] {
                let river_mask = board_mask | (1u64 << tc) | (1u64 << rc);
                for (i, &hi) in valid_hand_indices.iter().enumerate() {
                    let (c1, c2) = index_to_card_pair(hi as usize);
                    if river_mask & (1u64 << c1) != 0 || river_mask & (1u64 << c2) != 0 { continue; }
                    let mut hand = Hand::new();
                    hand = hand.add_card(c1 as usize);
                    hand = hand.add_card(c2 as usize);
                    for &bc in &board { hand = hand.add_card(bc as usize); }
                    hand = hand.add_card(tc as usize);
                    hand = hand.add_card(rc as usize);
                    river_ranks[tc as usize * 52 * nh + rc as usize * nh + i] = hand.evaluate_internal() as u16;
                }
                // BUG FIX 2026-06-05: match production compute_flop_start
                // (flop_start_game.rs line 175). Include ALL nh hands (with
                // strength 0 → 1 for blocked) instead of filtering. Otherwise
                // sorted_pl_idx has zero-padding that overwrites hand_strength[0]
                // in the GPU showdown helper.
                let mut items: Vec<(u16, u16)> = (0..nh)
                    .map(|h| (river_ranks[tc as usize * 52 * nh + rc as usize * nh + h] + 1, h as u16))
                    .collect();
                items.sort_by_key(|&(s, _)| s);
                for oi in 0..num_opp {
                    let base = tc as usize * 52 * num_opp * nh
                             + rc as usize * num_opp * nh
                             + oi * nh;
                    for (si, &(str_, idx)) in items.iter().enumerate() {
                        river_sorted_str[base + si] = str_;
                        river_sorted_idx[base + si] = idx;
                    }
                }
            }
        }
        let initial_weights: Vec<Vec<f32>> = (0..num_players).map(|_| {
            let mut w = vec![0.0f32; nh];
            for h in 0..nh {
                let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
                let mut blocked = 0i32;
                for h2 in 0..nh {
                    if h2 == h { continue; }
                    let (c3, c4) = index_to_card_pair(valid_hand_indices[h2] as usize);
                    if c1 == c3 || c1 == c4 || c2 == c3 || c2 == c4 { blocked += 1; }
                }
                w[h] = if blocked < (nh as i32 - 1) { 1.0 } else { 0.0 };
            }
            w
        }).collect();
        let num_combinations = initial_weights[0].iter().sum::<f32>()
                              * initial_weights[1].iter().sum::<f32>();
        let table = FlopChanceTable {
            hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
            remaining_deck: turn_cards, turn_ranks, turn_sorted_str, turn_sorted_idx,
            river_ranks, river_sorted_str, river_sorted_idx, initial_weights, num_players,
            num_combinations: num_combinations as f64, river_decks,
        };
        let config = TreeConfig {
            num_players: 2, initial_state: BoardState::Flop,
            starting_pot: pot, starting_stacks: vec![stacks, stacks],
            initial_contributions: vec![pot / 2, pot / 2],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions {
                bet: vec![BetSize::PotRelative(1.0)], raise: vec![],
            },
            add_allin_threshold: 1.0, force_allin_threshold: 1.0,
            merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
        };
        let tree = build_tree(&config).expect("flop tree");
        let game = FlopStartGame::new(table);
        (tree, game)
    }
}

#[test]
#[ignore = "Step 2.A.1 Rust-translation diagnostic. Run on demand."]
fn step_2a1_rust_translation_diagnostic() {
    eprintln!("\n========================================================================");
    eprintln!("=== Step 2.A.1: Rust translation of GPU HU showdown, run with        ===");
    eprintln!("===   node-28 inputs, compare to both CPU showdown_cfv and GPU       ===");
    eprintln!("========================================================================\n");

    let (tree, game) = helper::build_full_nh_4board_game(/*stacks=*/5, /*pot=*/2);
    let table = game.table();
    let nh = table.num_valid;
    let np = table.num_players as usize;
    let num_opp = np - 1;
    let nn = tree.num_nodes();
    eprintln!("Setup: nh = {}, np = {}", nh, np);

    // ─── Run setup to get the reach at node 28 ───
    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    let ctx = MetalContext::new().expect("Metal");
    let gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    cpu.compute_all_strategies(&tree);
    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, 0, &cpu_flop_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, 0, 0, &cpu_turn_reach);
    let _ = gpu;

    let node_id = 28usize;
    let traverser = 0usize;
    let opp = if traverser == 0 { 1 } else { 0 };
    let node_reach_base = node_id * np * nh;
    let opp_reach: Vec<f32> = cpu_river_reach[node_reach_base + opp * nh
        ..node_reach_base + opp * nh + nh].to_vec();
    let contributions: Vec<i32> = (0..np).map(|p|
        tree.contributions[node_id * np + p]).collect();
    let fold_mask = tree.get_folded_mask(node_id);

    let tc_card = table.remaining_deck[0] as usize;
    let rc_card = table.river_decks[tc_card][0] as usize;
    let sorted_offset = (tc_card * 52 + rc_card) * num_opp * nh;
    let pl_str: Vec<u16> = table.river_sorted_str[sorted_offset..sorted_offset + nh].to_vec();
    let pl_idx: Vec<u16> = table.river_sorted_idx[sorted_offset..sorted_offset + nh].to_vec();
    let mut sorted_opp_str = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx = Vec::with_capacity(num_opp * nh);
    for oi in 0..num_opp {
        sorted_opp_str.extend_from_slice(
            &table.river_sorted_str[sorted_offset + oi * nh..sorted_offset + (oi + 1) * nh]);
        sorted_opp_idx.extend_from_slice(
            &table.river_sorted_idx[sorted_offset + oi * nh..sorted_offset + (oi + 1) * nh]);
    }

    let _ = nn;
    // ─── 1. CPU side_pot_showdown_cfv ───
    let opp_reach_views = vec![opp_reach.as_slice()];
    let cpu_cfv = side_pot_showdown_cfv(
        &opp_reach_views, &table.hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &pl_str, &pl_idx,
        &contributions, fold_mask, traverser, np as u8, tree.starting_pot,
    );

    // ─── 2. GPU debug helper ───
    let gpu_cfv = gpu_brute_force(
        &ctx, nh, np, traverser, tree.starting_pot, fold_mask,
        &opp_reach, &contributions, &table.hand_cards, &pl_str, &pl_idx,
    );

    // ─── 3. Rust translation of GPU HU branch ───
    let (rust_cfv, traces) = rust_translation_of_gpu_hu(
        nh, np, traverser, tree.starting_pot, fold_mask,
        &contributions, &opp_reach, &table.hand_cards,
        &pl_str, &pl_idx,
        0.0, 0.0, false,
        Some(0),  // trace h=0
    );

    // ─── Compare h=0 values ───
    eprintln!("\n--- h=0 values ---");
    eprintln!("  CPU side_pot_showdown_cfv: {:>12.4}", cpu_cfv[0]);
    eprintln!("  GPU debug helper:          {:>12.4}", gpu_cfv[0]);
    eprintln!("  Rust translation:          {:>12.4}", rust_cfv[0]);

    let cpu_v = cpu_cfv[0];
    let gpu_v = gpu_cfv[0];
    let rust_v = rust_cfv[0];

    let rust_matches_cpu  = (rust_v - cpu_v).abs() < 1.0;
    let rust_matches_gpu  = (rust_v - gpu_v).abs() < 1.0;

    eprintln!("\n--- Verdict ---");
    if rust_matches_cpu && !rust_matches_gpu {
        eprintln!("  ✓ Rust matches CPU (within 1.0). Bug is GPU-SPECIFIC.");
        eprintln!("    The logic translated to Rust gives the right answer.");
        eprintln!("    The bug is in Metal-specific behavior (register spilling,");
        eprintln!("    thread-local array layout, alignment, etc.) at nh=1176.");
        eprintln!("    Next: instrument the actual Metal kernel.");
    } else if rust_matches_gpu && !rust_matches_cpu {
        eprintln!("  ✓ Rust matches GPU (within 1.0). Bug is LOGIC-LEVEL.");
        eprintln!("    The faithful translation reproduces the GPU's wrong answer.");
        eprintln!("    The bug is in the logic I translated — and I can debug it");
        eprintln!("    directly using the per-g_a trace below.");
    } else if rust_matches_cpu && rust_matches_gpu {
        eprintln!("  ? Rust matches BOTH — that's impossible given CPU≠GPU.");
        eprintln!("    Check unit / sign conventions.");
    } else {
        eprintln!("  ✗ Rust matches NEITHER. I misread the kernel; recheck translation.");
        eprintln!("    diff Rust vs CPU: {}", (rust_v - cpu_v).abs());
        eprintln!("    diff Rust vs GPU: {}", (rust_v - gpu_v).abs());
    }

    // ─── Trace summary at h=0 ───
    eprintln!("\n--- Per-g_a survival + sign summary at h=0 ---");
    let surv_blocking = traces.iter().filter(|t| t.survived_blocking).count();
    let surv_both     = traces.iter().filter(|t| t.survived_blocking && t.survived_reach).count();
    let wins  = traces.iter().filter(|t| t.survived_blocking && t.survived_reach && t.win_lose_tie > 0).count();
    let ties  = traces.iter().filter(|t| t.survived_blocking && t.survived_reach && t.win_lose_tie == 0).count();
    let losses = traces.iter().filter(|t| t.survived_blocking && t.survived_reach && t.win_lose_tie < 0).count();
    eprintln!("  survived blocking:        {} / {}", surv_blocking, nh);
    eprintln!("  survived blocking+reach:  {}",       surv_both);
    eprintln!("  wins (h_str > s_a):       {}", wins);
    eprintln!("  ties (h_str == s_a):      {}", ties);
    eprintln!("  losses (h_str < s_a):     {}", losses);

    let sum_terms: f32 = traces.iter().map(|t| t.term).sum();
    let positive_term_sum: f32 = traces.iter().map(|t| t.term.max(0.0)).sum();
    let negative_term_sum: f32 = traces.iter().map(|t| t.term.min(0.0)).sum();
    eprintln!("  sum of all terms (=accum, =Rust h=0 result): {:.4}", sum_terms);
    eprintln!("  positive term sum: {:.4}", positive_term_sum);
    eprintln!("  negative term sum: {:.4}", negative_term_sum);
}
