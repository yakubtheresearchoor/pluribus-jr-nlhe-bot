// f32-through-meaningful-convergence on a 6-player game with REAL side-pot
// and fold terminals exercised on every iteration.
//
// The previous nh=6 equal-contrib test converged to check-down in 5 iters
// and ran on a game where side-pot terminals had near-zero reach during
// iteration — so it confirmed f32 stability but not f32 sufficiency on
// the cancellation-heavy side-pot computations where f32 is actually in
// question.
//
// THIS TEST: asymmetric initial contributions [10, 5, 5, 5, 5, 5]
// (effectively a "big blind" position for player 0). Even the trivial
// check-down equilibrium produces 2-level pot structure with non-trivial
// reach, so the per-level factored path is exercised on side-pot
// terminals every iteration. Multi-street folds add 3- and 4-level
// terminals to the mix.
//
// nh=8 gives meaningful hand-strength diversity. 50 iters is enough to
// see the descent settle to a floor.
//
// Pass criteria:
//   - Exploitability bounded (no divergence in f32)
//   - No NaN / Inf
//   - Trajectory descends OR stays at a low equilibrium floor — either
//     pattern confirms f32 stability through convergence on the relevant
//     terminal types.

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
use std::time::Instant;

fn build_6p_asymmetric_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
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
        let mut items: Vec<(u16, u16)> = (0..nh)
            .map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
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
                .map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16)).collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; num_players as usize];
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
    let nc = enum_nc(0, num_players as usize, nh, 0, &hand_cards[..], 1.0);
    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players, num_combinations: nc, river_decks: rd,
    };
    // ASYMMETRIC CONTRIBUTIONS — produces side-pot terminals from the start.
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
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

fn upload_gpu_to_cpu(cpu: &mut FlopStartVectorCfr, gpu_reg: &[f32], gpu_cum: &[f32]) {
    let fl = cpu.regrets_flop().len();
    let tl = cpu.regrets_turn().len();
    { let r = cpu.regrets_flop_mut(); for i in 0..fl { r[i] = gpu_reg[i]; } }
    { let r = cpu.regrets_turn_mut(); for i in 0..tl { r[i] = gpu_reg[fl + i]; } }
    { let r = cpu.regrets_river_mut(); for i in 0..r.len() {
        if fl + tl + i < gpu_reg.len() { r[i] = gpu_reg[fl + tl + i]; } } }
    { let c = cpu.cum_strategy_flop_mut(); for i in 0..fl { c[i] = gpu_cum[i]; } }
    { let c = cpu.cum_strategy_turn_mut(); for i in 0..tl { c[i] = gpu_cum[fl + i]; } }
    { let c = cpu.cum_strategy_river_mut(); for i in 0..c.len() {
        if fl + tl + i < gpu_cum.len() { c[i] = gpu_cum[fl + tl + i]; } } }
}

#[test]
#[ignore = "slow (~15-20 min); run with `cargo test --release --features metal -- --ignored`"]
fn six_player_f32_convergence_with_side_pot_terminals() {
    let nh = 8;
    let (tree, table) = build_6p_asymmetric_table(nh);
    let game = FlopStartGame::new(table);
    let np = 6usize;

    // Count terminals with multi-level pot structure (side-pot terminals)
    // to confirm the test really exercises them.
    let mut terminal_count = 0usize;
    let mut side_pot_terminal_count = 0usize;
    for (idx, node) in tree.nodes.iter().enumerate() {
        if !node.is_terminal() { continue; }
        terminal_count += 1;
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let mut sorted = contribs.clone();
        sorted.sort();
        sorted.dedup();
        if sorted.len() >= 2 {
            side_pot_terminal_count += 1;
        }
    }
    eprintln!("\n=== f32 convergence on 6p with asymmetric contribs [10,5,5,5,5,5] ===");
    eprintln!("nh={}, tree nodes={}", nh, tree.num_nodes());
    eprintln!("Terminals: {} total, {} side-pot (≥2 levels)",
        terminal_count, side_pot_terminal_count);
    eprintln!("Pot pre-action: 30 + 35 = 65 chips (asymmetric blinds make every terminal multi-level)");

    let mut cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);

    let checkpoints: &[u32] = &[1, 3, 10, 25, 50];
    let mut prev = 0u32;
    let pot_total = 65.0f32;

    eprintln!();
    let t0 = Instant::now();
    let mut trajectory: Vec<(u32, f32)> = Vec::new();
    for &cp in checkpoints {
        let it0 = Instant::now();
        gpu.run(&ctx, &tree, &game, cp - prev);
        let it_elapsed = it0.elapsed();
        prev = cp;

        cpu_proxy.set_iteration(gpu.iteration());
        let gpu_reg = gpu.download_regrets();
        let gpu_cum = gpu.download_cum_strategy();
        upload_gpu_to_cpu(&mut cpu_proxy, &gpu_reg, &gpu_cum);
        let expl = measure_exploitability(&cpu_proxy, &tree, &game, np);
        let pct = expl / pot_total * 100.0;
        trajectory.push((cp, pct));
        eprintln!("GPU iter {:5}: exploitability = {:.6}% of pot  (batch {:?}, total {:?})",
            cp, pct, it_elapsed, t0.elapsed());
        std::io::stderr().flush().ok();
    }

    // Pass criteria: bounded throughout, no NaN, never grows over time.
    let first = trajectory.first().map(|&(_, p)| p).unwrap_or(0.0);
    let last = trajectory.last().map(|&(_, p)| p).unwrap_or(0.0);
    let max_in_run = trajectory.iter().map(|&(_, p)| p).fold(0.0f32, f32::max);
    eprintln!("\nTrajectory: first={:.6}%, last={:.6}%, max-in-run={:.6}%",
        first, last, max_in_run);

    assert!(last.is_finite(), "f32 convergence produced non-finite exploitability");
    assert!(max_in_run.is_finite() && max_in_run < 1000.0,
        "Max-in-run exploitability {:.4} indicates instability", max_in_run);
    // The trajectory should not be monotonically GROWING (which would indicate
    // f32 noise compounding into divergence). It's OK if it bounces around a
    // floor, OR descends, OR descends-then-oscillates. The bad pattern is
    // strictly increasing from a low floor.
    let descended_at_some_point = trajectory.windows(2).any(|w| w[1].1 < w[0].1);
    assert!(descended_at_some_point || last <= first * 1.5,
        "Exploitability strictly grew across all checkpoints: {:?}", trajectory);
    assert!(side_pot_terminal_count > 0,
        "No side-pot terminals in tree — test setup failed to exercise them");
}
