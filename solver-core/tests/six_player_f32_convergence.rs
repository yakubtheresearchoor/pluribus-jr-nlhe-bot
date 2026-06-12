// f32-through-convergence: confirm the 6-player GPU solver (using f32
// throughout, with the unified factored kernel for K=5 showdown)
// converges cleanly to a floor.
//
// Per-terminal f32 precision does not by itself establish f32 sufficiency
// over a multi-iter convergence run. Single-iter parity (gate 5) shows
// the kernel computes the right CFV; iter-2 parity (the indexing-bug
// gate) shows non-uniform reach doesn't introduce position-dependent
// drift; this test confirms the solver actually CONVERGES in f32 over
// many iterations — that the f32 noise per iter doesn't compound into
// divergence or stalled progress.
//
// Method: run GPU CFR on a 6p nh=6 game for N iterations. At checkpoints,
// download regrets, upload to a CPU proxy, compute exploitability via
// best_response vs strategy_value. Print the exploitability curve.
//
// Pass criterion: exploitability descends from a high initial value to a
// substantially lower floor. The specific floor depends on the game
// structure; what matters for this gate is that the curve descends and
// stays descended (no divergence after some iterations, no stagnation
// at a high value).

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

fn build_6p_table(nh: usize) -> (FlatTree, FlopChanceTable) {
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
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![100; 6], initial_contributions: vec![5; 6],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

fn measure_exploitability(
    cpu: &FlopStartVectorCfr, tree: &FlatTree, game: &FlopStartGame, np: usize,
) -> f32 {
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
#[ignore = "slow (minutes); run with `cargo test --release --features metal -- --ignored`"]
fn six_player_f32_convergence_nh6() {
    let nh = 6;
    let (tree, table) = build_6p_table(nh);
    let game = FlopStartGame::new(table);
    let np = 6usize;

    let mut cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);

    // Checkpoint schedule: descending, with logarithmic-ish spacing.
    let checkpoints: &[u32] = &[5, 10, 25, 50, 100];
    let mut prev = 0u32;
    let pot = 30.0f32; // starting_pot for 6p

    eprintln!("\n=== f32-through-convergence: 6p nh={} GPU ===", nh);
    eprintln!("Checkpoints: {:?}", checkpoints);
    eprintln!();
    let t0 = Instant::now();
    let mut prev_pct = f32::INFINITY;
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
        let pct = expl / pot * 100.0;
        trajectory.push((cp, pct));
        eprintln!("GPU iter {:5}: exploitability = {:.4}% of pot  (batch {:?}, total {:?})",
            cp, pct, it_elapsed, t0.elapsed());
        std::io::stderr().flush().ok();
        prev_pct = pct;
    }
    let _ = prev_pct;

    // Pass criterion: exploitability descended from the first checkpoint to
    // the last by a meaningful factor (at least 2x), AND is finite (no
    // divergence in f32).
    let first = trajectory.first().map(|&(_, p)| p).unwrap_or(0.0);
    let last = trajectory.last().map(|&(_, p)| p).unwrap_or(0.0);
    eprintln!("\nDescent: {:.4}% → {:.4}% ({:.2}x)", first, last, first / last.max(1e-6));
    assert!(last.is_finite(),
        "f32 convergence produced non-finite exploitability: {}", last);
    assert!(last < first,
        "Exploitability did not descend over iters: first={:.4}%, last={:.4}%",
        first, last);
    assert!(first / last >= 2.0,
        "Insufficient descent: first/last = {:.2}x (need >= 2.0x)", first / last);
}
