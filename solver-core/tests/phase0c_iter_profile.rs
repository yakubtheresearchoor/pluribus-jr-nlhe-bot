// Phase 0.C: per-iteration wall-clock breakdown by phase.
//
// Time each phase of the GPU iter loop independently to identify
// where the per-iter cost goes. Categories per the spec:
//   (a) GPU kernel compute for showdown (fused inside bottom_up_*)
//   (b) GPU kernel for tree traversal and regret updates (compute_reach_*,
//       and the non-showdown portion of bottom_up_*)
//   (c) Host-side work (orchestration loop overhead)
//   (d) Synchronization/launch overhead (wait_until_completed cost)
//
// The unified factored kernel is FUSED with regret update inside
// `vcfr_bottom_up_batched`, so (a) and (b) can't be cleanly separated
// at the dispatch level — they're reported as one combined cost
// "backward + showdown + regret" with a note for Phase 2 follow-up.
//
// What we can cleanly measure:
//   - compute_all_strategies time
//   - compute_reach_flop time
//   - per-turn loop: compute_reach_turn + per-river inner + chance_accumulate/finalize + bottom_up_turn
//   - bottom_up_flop time
//   - Total wall-clock minus sum of measured phases = host orchestration overhead
//
// Methodology: same iter structure as MetalFlopStartSolver::run() but
// with Instant::now() around each phase. Run several warmup iters
// (transition through iter-2 spike to settle), then profile iter ~6
// which is in steady-state.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::io::Write;
use std::time::{Duration, Instant};

fn build_6p_asymmetric_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 6u8;
    let num_opp = 5usize;

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
        hand_cards[i*2] = c1; hand_cards[i*2+1] = c2;
    }
    let mut conflict = vec![0u8; nh*nh];
    for i in 0..nh { for j in 0..nh {
        if i == j { conflict[i*nh+j] = 1; continue; }
        let (a1,a2) = index_to_card_pair(chosen[i] as usize);
        let (b1,b2) = index_to_card_pair(chosen[j] as usize);
        if a1==b1||a1==b2||a2==b1||a2==b2 { conflict[i*nh+j] = 1; }
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
        let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
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
            let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16)).collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; num_players as usize];
    fn enum_nc(player: usize, np: usize, nh: usize, combined: u64, hand_cards: &[u8], weight: f64) -> f64 {
        if player == np { return weight; }
        let mut total = 0.0;
        for h in 0..nh {
            let m = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
            if combined & m != 0 { continue; }
            total += enum_nc(player + 1, np, nh, combined | m, hand_cards, weight);
        }
        total
    }
    let nc = enum_nc(0, num_players as usize, nh, 0, &hand_cards[..], 1.0);
    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players, num_combinations: nc, river_decks: rd,
    };
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

#[derive(Default, Clone, Debug)]
struct PhaseBudget {
    total: Duration,
    compute_strategies: Duration,
    compute_reach_flop: Duration,
    compute_reach_turn: Duration,
    compute_reach_river: Duration,
    bottom_up_river: Duration,  // includes showdown CFV
    bottom_up_turn: Duration,
    bottom_up_flop: Duration,
    chance_accumulate_river: Duration,
    chance_finalize_river: Duration,
    chance_accumulate_turn: Duration,
    chance_finalize_turn: Duration,
    zero_buffer_total: Duration,
}

/// Run one iter while measuring per-phase wall-clock; mirrors run() structure.
fn one_iter_with_phase_timing(
    gpu: &mut MetalFlopStartSolver,
    ctx: &MetalContext,
    _tree: &FlatTree,
    _game: &FlopStartGame,
) -> PhaseBudget {
    let mut b = PhaseBudget::default();
    let np = 6usize;
    let params = solver_core::gpu_metal::flop_solver::DcfrParams::new(gpu.iteration());
    gpu.set_iteration(gpu.iteration() + 1);
    let t_total = Instant::now();

    for traverser in 0..np {
        let t = Instant::now();
        gpu.compute_all_strategies(ctx);
        b.compute_strategies += t.elapsed();

        let t = Instant::now();
        gpu.compute_reach_flop(ctx);
        b.compute_reach_flop += t.elapsed();

        let t = Instant::now();
        gpu.zero_buffer_name(ctx, 100);
        gpu.zero_buffer_name(ctx, 2);
        b.zero_buffer_total += t.elapsed();

        for ti in 0..gpu.n_turn() {
            let n_river = gpu.river_outcomes_per_turn()[ti];
            let t = Instant::now();
            gpu.zero_buffer_name(ctx, 0);
            gpu.zero_buffer_name(ctx, 1);
            b.zero_buffer_total += t.elapsed();

            let t = Instant::now();
            gpu.compute_reach_turn(ctx, ti);
            b.compute_reach_turn += t.elapsed();

            for ri in 0..n_river {
                let t = Instant::now();
                gpu.compute_reach_river(ctx, ti, ri);
                b.compute_reach_river += t.elapsed();

                let t = Instant::now();
                gpu.bottom_up_river(ctx, ti, ri, traverser as u32, &params);
                b.bottom_up_river += t.elapsed();
            }

            let t = Instant::now();
            gpu.chance_accumulate_river(ctx, ti, n_river);
            b.chance_accumulate_river += t.elapsed();

            let t = Instant::now();
            gpu.chance_finalize_river(ctx, ti);
            b.chance_finalize_river += t.elapsed();

            let t = Instant::now();
            gpu.bottom_up_turn(ctx, ti, traverser as u32, &params);
            b.bottom_up_turn += t.elapsed();
        }

        let t = Instant::now();
        gpu.chance_accumulate_turn(ctx);
        b.chance_accumulate_turn += t.elapsed();

        let t = Instant::now();
        gpu.chance_finalize_turn(ctx);
        b.chance_finalize_turn += t.elapsed();

        let t = Instant::now();
        gpu.bottom_up_flop(ctx, traverser as u32, &params);
        b.bottom_up_flop += t.elapsed();
    }
    b.total = t_total.elapsed();
    b
}

fn fmt_phase(name: &str, d: Duration, total: Duration) -> String {
    let s = d.as_secs_f64();
    let pct = s / total.as_secs_f64() * 100.0;
    format!("  {:30} {:>10.2} s  ({:>5.1}%)", name, s, pct)
}

#[test]
#[ignore = "Phase 0.C profile, ~30 min at nh=12"]
fn phase0c_per_iter_breakdown_nh12() {
    let nh = 12usize;

    eprintln!("\n=== Phase 0.C: per-iter wall-clock breakdown at nh={} ===\n", nh);
    let (tree, table) = build_6p_asymmetric_table(nh);
    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    eprintln!("Tree: {} nodes, {} MB regrets at nh={}",
        tree.num_nodes(), tree.num_nodes() * nh * 4 / 1024 / 1024, nh);
    std::io::stderr().flush().ok();

    // Warmup: settle JIT, get past iter-1 (uniform reach) and iter-2 (transition spike).
    eprintln!("Warmup pass: 1 iter to absorb kernel JIT (~14 min expected)...");
    let t_warm = Instant::now();
    gpu.run(&ctx, &tree, &game, 1);
    eprintln!("Warmup done: {:.1} s", t_warm.elapsed().as_secs_f64());

    eprintln!("Throwaway iter 2 (transition spike, ~12 min expected)...");
    let t = Instant::now();
    gpu.run(&ctx, &tree, &game, 1);
    eprintln!("  iter 2 done: {:.1} s (skipped from profile)", t.elapsed().as_secs_f64());

    eprintln!("Throwaway iters 3-5 (settling)...");
    let t = Instant::now();
    gpu.run(&ctx, &tree, &game, 3);
    eprintln!("  iters 3-5 done: {:.1} s (skipped from profile)", t.elapsed().as_secs_f64());

    eprintln!();
    eprintln!("Profiled iter 6 (steady state):");
    std::io::stderr().flush().ok();
    let prof = one_iter_with_phase_timing(&mut gpu, &ctx, &tree, &game);

    eprintln!();
    eprintln!("=== Phase budget for iter 6 (total {:.2} s) ===", prof.total.as_secs_f64());
    let measured = prof.compute_strategies + prof.compute_reach_flop +
                   prof.compute_reach_turn + prof.compute_reach_river +
                   prof.bottom_up_river + prof.bottom_up_turn + prof.bottom_up_flop +
                   prof.chance_accumulate_river + prof.chance_finalize_river +
                   prof.chance_accumulate_turn + prof.chance_finalize_turn +
                   prof.zero_buffer_total;
    let host_overhead = prof.total.checked_sub(measured).unwrap_or_default();

    println!("{}", fmt_phase("Strategy compute (flop)", prof.compute_strategies, prof.total));
    println!("{}", fmt_phase("Reach: flop forward", prof.compute_reach_flop, prof.total));
    println!("{}", fmt_phase("Reach: turn (per-turn)", prof.compute_reach_turn, prof.total));
    println!("{}", fmt_phase("Reach: river (per-river)", prof.compute_reach_river, prof.total));
    println!("{}", fmt_phase("Bottom-up: RIVER (showdown+regret)", prof.bottom_up_river, prof.total));
    println!("{}", fmt_phase("Bottom-up: TURN", prof.bottom_up_turn, prof.total));
    println!("{}", fmt_phase("Bottom-up: FLOP", prof.bottom_up_flop, prof.total));
    println!("{}", fmt_phase("Chance accumulate (river)", prof.chance_accumulate_river, prof.total));
    println!("{}", fmt_phase("Chance finalize (river)", prof.chance_finalize_river, prof.total));
    println!("{}", fmt_phase("Chance accumulate (turn)", prof.chance_accumulate_turn, prof.total));
    println!("{}", fmt_phase("Chance finalize (turn)", prof.chance_finalize_turn, prof.total));
    println!("{}", fmt_phase("Zero buffer (small ops)", prof.zero_buffer_total, prof.total));
    println!("{}", fmt_phase("HOST/SYNC OVERHEAD (residual)", host_overhead, prof.total));
    println!();
    println!("=== Aggregated categories (Phase 0 deliverable) ===");
    let category_traversal = prof.compute_strategies + prof.compute_reach_flop +
                              prof.compute_reach_turn + prof.compute_reach_river +
                              prof.chance_accumulate_river + prof.chance_finalize_river +
                              prof.chance_accumulate_turn + prof.chance_finalize_turn;
    let category_backward_showdown_regret = prof.bottom_up_river + prof.bottom_up_turn + prof.bottom_up_flop;
    println!("{}", fmt_phase("Tree traversal + reach + chance integration", category_traversal, prof.total));
    println!("{}", fmt_phase("Backward + SHOWDOWN + regret update", category_backward_showdown_regret, prof.total));
    println!("{}", fmt_phase("Zero buffers", prof.zero_buffer_total, prof.total));
    println!("{}", fmt_phase("Host orchestration + sync overhead", host_overhead, prof.total));
    println!();
    println!("Note: showdown CFV is FUSED with regret update inside vcfr_bottom_up_batched");
    println!("kernel, so cannot be cleanly separated at the dispatch level. The 'Backward +");
    println!("SHOWDOWN + regret' category combines all three. If this category dominates,");
    println!("Phase 2 should add kernel-level timestamps to isolate the showdown specifically.");
}
