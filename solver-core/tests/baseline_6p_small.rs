// Baseline measurement on the validated K=5 6-player solver — small scale.
//
// The full nh=50 6p baseline on the validated CPU brute-force solver is
// ~years-per-solve (485k terminals × 50 h × 64M tuples per h × 10 ops, even
// at multi-threaded CPU rates). That measurement requires the GPU-integrated
// factored K=5 kernel (next session's work; the standalone kernel from this
// session is the foundation but not yet wired into evaluate_terminal).
//
// THIS BASELINE: nh=8 6p K=5 on the CPU brute-force showdown path that's
// already wired in via the line-822 fix. Establishes:
//   - convergence curve descent (how fast exploitability falls)
//   - per-iter wall time at this scale
//   - memory footprint (observed as side effect)
//   - methodology for the eventual nh=50 baseline
//
// Scaling extrapolation to nh=50: per-iter cost scales nh^5 per terminal in
// brute-force × constant in terminal count. (50/8)^5 ≈ 9,537× per-iter cost
// at nh=50 vs nh=8. The factored-kernel measurement (174 µs/h, 70 min/iter
// for nh=50 6p on GPU saturated) is the actual scale anchor for the
// integrated production baseline.
//
// Discipline: this is the REFERENCE. Every optimization that follows is
// gated on convergence matching this baseline. One change at a time.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::io::Write;
use std::time::Instant;

fn build_game(np: u8, nh: usize) -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_opp = np as usize - 1;

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
    let iw = vec![vec![1.0f32; nh]; np as usize];
    fn enum_nc(player: usize, np: usize, nh: usize, combined: u64,
               hand_cards: &[u8], weight: f64) -> f64 {
        if player == np { return weight; }
        let mut total = 0.0;
        for h in 0..nh {
            let m = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
            if combined & m != 0 { continue; }
            total += enum_nc(player + 1, np, nh, combined | m, hand_cards, weight);
        }
        total
    }
    let nc = enum_nc(0, np as usize, nh, 0, &hand_cards[..], 1.0);
    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players: np, num_combinations: nc, river_decks: rd,
    };
    let config = TreeConfig {
        num_players: np, initial_state: BoardState::Flop, starting_pot: np as i32 * 5,
        starting_stacks: vec![100; np as usize], initial_contributions: vec![5; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,

    };
    let tree = build_tree(&config).unwrap();
    let game = FlopStartGame::new(table);
    (tree, game)
}

fn measure_exploitability(cpu: &FlopStartVectorCfr, tree: &FlatTree, game: &FlopStartGame, np: usize) -> f32 {
    let mut total = 0.0f32;
    for p in 0..np {
        let br = cpu.best_response_value_debug(tree, game, p as u8);
        let sv = cpu.strategy_value_debug(tree, game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total += (br[h] - sv[h]).max(0.0);
        }
    }
    total
}

/// Baseline at nh=8 6p K=5 on the validated CPU brute-force solver.
///
/// Establishes the reference for everything that follows.
/// - Convergence curve: descend exploitability over iters, identify floor.
/// - Per-iter timing.
/// - Memory footprint: observed as a side effect.
/// - Methodology for the eventual nh=50 baseline (after GPU integration).
///
/// Marked #[ignore] so the default test run doesn't trigger this ~2 hour run.
/// Run with: `cargo test --release --test baseline_6p_small -- --ignored --nocapture`
#[test]
#[ignore = "slow ~2 hr; baseline measurement, run explicitly"]
fn baseline_6p_nh8_validated_solver() {
    let np = 6u8;
    let nh = 8usize;
    let (tree, game) = build_game(np, nh);
    let np_us = np as usize;
    let pot = (np as i32 * 5) as f32;

    eprintln!("\n=== BASELINE: nh={} 6p K=5 on validated CPU brute-force solver ===", nh);
    eprintln!("Tree: {} total nodes, terminals counted at first iter", tree.nodes.len());
    let total_terminals = tree.nodes.iter().filter(|n| n.is_terminal()).count();
    eprintln!("Terminals: {} ({:.1}% of nodes)", total_terminals,
        100.0 * total_terminals as f64 / tree.nodes.len() as f64);
    eprintln!("Pot: {}", pot);

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());

    // Memory observation: capture RSS estimate via resident-size APIs would
    // require an external dep. The solver's main allocations are
    // regrets/cum_strategy/strategy at sizes nh × MAX_NA_POSTFLOP × infoset_count
    // per zone. Print sizes so we can read out actual memory.
    let nh_bytes_per_slot = nh * 4; // f32
    let flop_slot = cpu.flop_infosets() * solver_core::tree::flat::MAX_NA_POSTFLOP * nh_bytes_per_slot;
    let turn_slot = cpu.turn_infosets() * solver_core::tree::flat::MAX_NA_POSTFLOP * nh_bytes_per_slot
                    * cpu.n_turn_outcomes();
    let river_slot = cpu.river_infosets() * solver_core::tree::flat::MAX_NA_POSTFLOP * nh_bytes_per_slot
                    * cpu.n_turn_outcomes() * cpu.max_river_outcomes();
    // Three arrays each (regrets, strategy, cum_strategy).
    let total_solver_arrays = 3 * (flop_slot + turn_slot + river_slot);
    eprintln!("Solver memory (3× regrets+strategy+cum, per-zone):");
    eprintln!("  Flop arrays:  {:.1} MB", flop_slot as f64 / 1e6);
    eprintln!("  Turn arrays:  {:.1} MB", turn_slot as f64 / 1e6);
    eprintln!("  River arrays: {:.1} MB", river_slot as f64 / 1e6);
    eprintln!("  Total:        {:.1} MB ({:.3}% of 36GB)",
        total_solver_arrays as f64 / 1e6,
        100.0 * total_solver_arrays as f64 / 36e9);

    // Checkpoints: log dense for the early-descent visible portion, sparser
    // as iters grow.
    let checkpoints: &[u32] = &[1, 2, 5, 10, 20, 30, 50];
    let mut prev: u32 = 0;
    let t0 = Instant::now();
    let mut trajectory: Vec<(u32, f32, f64)> = Vec::new();

    eprintln!("\nIter   Exploit%  Per-iter time   Total elapsed");
    eprintln!("----   --------  -------------   -------------");

    for &cp in checkpoints {
        let batch = cp - prev;
        let t_batch = Instant::now();
        cpu.run(&tree, &game, batch);
        let batch_elapsed = t_batch.elapsed();
        prev = cp;

        let expl = measure_exploitability(&cpu, &tree, &game, np_us);
        let pct = (expl / pot * 100.0) as f64;
        let per_iter = batch_elapsed.as_secs_f64() / batch as f64;
        trajectory.push((cp, expl, per_iter));

        eprintln!("{:>4}   {:>6.3}%  {:>10.2}s   {:>11.1}s",
            cp, pct, per_iter, t0.elapsed().as_secs_f64());
        std::io::stderr().flush().ok();
    }

    eprintln!("\n--- Convergence curve summary ---");
    eprintln!("First-iter exploit%: {:.3}%",
        (trajectory.first().unwrap().1 / pot * 100.0) as f64);
    eprintln!("Last-iter  exploit%: {:.3}%",
        (trajectory.last().unwrap().1 / pot * 100.0) as f64);
    let drop_ratio = trajectory.first().unwrap().1 / trajectory.last().unwrap().1.max(1e-6);
    eprintln!("Drop ratio: {:.1}x over {} iters", drop_ratio, trajectory.last().unwrap().0);

    let avg_per_iter: f64 = trajectory.iter().map(|&(_, _, t)| t).sum::<f64>()
        / trajectory.len() as f64;
    eprintln!("Average per-iter at nh={}: {:.2}s", nh, avg_per_iter);

    eprintln!("\n--- Projection to nh=50 ---");
    let scale = (50f64 / nh as f64).powi(5); // K=5 brute-force per-h ∝ nh^5
    let per_iter_nh50_brute_cpu = avg_per_iter * scale;
    eprintln!("CPU brute-force scaling factor (nh^5): {:.0}x", scale);
    eprintln!("Projected nh=50 6p CPU brute-force per-iter: {:.0} s = {:.1} hr",
        per_iter_nh50_brute_cpu, per_iter_nh50_brute_cpu / 3600.0);
    eprintln!("For 200 iters at this rate: {:.1} days",
        per_iter_nh50_brute_cpu * 200.0 / 86400.0);
    eprintln!("(Confirms: full nh=50 baseline requires GPU integration of");
    eprintln!(" the factored K=5 kernel — measured at 174 µs/h, ≈ 70 min/iter)");

    // Sanity assertions.
    assert!(trajectory.last().unwrap().1 < trajectory.first().unwrap().1,
        "Baseline did not descend over {} iters", trajectory.last().unwrap().0);
    assert!(trajectory.iter().all(|&(_, e, _)| e.is_finite()),
        "Non-finite exploitability");
}
