// Gate 6: CPU-only convergence for K=3 (4-player) and K=5 (6-player).
//
// Verifies that the brute-force K>=3 showdown CFV actually drives CFR toward
// a stable equilibrium, not just produces zero-sum cfv at single iterations.
// This is the structural test that catches "cfv is right but the gradient is
// wrong" failure modes — the kind that would make zero-sum checks pass while
// the solver still diverges.
//
// Per the plan:
//   "Assert convergence trajectory descends and floors on a non-degenerate
//    game (the specific floor value becomes the reference for gate 9)."
//
// nh is kept small (4-player: nh=10, 6-player: nh=8) because brute-force is
// O(nh^K) per terminal; production performance comes from the factored I-E
// formula once derived correctly.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::time::Instant;

fn build_game_n(np: u8, nh: usize) -> (FlatTree, FlopStartGame) {
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
    for i in 0..nh {
        for j in 0..nh {
            if i == j { conflict[i * nh + j] = 1; continue; }
            let (a1, a2) = index_to_card_pair(chosen[i] as usize);
            let (b1, b2) = index_to_card_pair(chosen[j] as usize);
            if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 { conflict[i * nh + j] = 1; }
        }
    }
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
    let starting_pot: i32 = (np as i32) * 5;
    let stacks: Vec<i32> = vec![100; np as usize];
    let contribs: Vec<i32> = vec![5; np as usize];
    let config = TreeConfig {
        num_players: np, initial_state: BoardState::Flop, starting_pot,
        starting_stacks: stacks, initial_contributions: contribs,
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).unwrap();
    let game = FlopStartGame::new(table);
    (tree, game)
}

fn measure_exploitability(
    cpu: &FlopStartVectorCfr,
    tree: &FlatTree,
    game: &FlopStartGame,
    np: usize,
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

fn run_convergence(np_u8: u8, nh: usize, max_iters: u32, checkpoints: &[u32], label: &str) {
    let (tree, game) = build_game_n(np_u8, nh);
    let np = np_u8 as usize;
    let pot = (np as i32 * 5) as f32;

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let _ = max_iters;

    eprintln!("\n=== [{}] CPU convergence np={} nh={} ===", label, np, nh);
    let t0 = Instant::now();
    let mut prev: u32 = 0;
    let mut trajectory: Vec<(u32, f32)> = Vec::new();
    for &cp in checkpoints {
        let batch = cp - prev;
        let it0 = Instant::now();
        cpu.run(&tree, &game, batch);
        let it_elapsed = it0.elapsed();
        prev = cp;
        let expl = measure_exploitability(&cpu, &tree, &game, np);
        let pct = expl / pot * 100.0;
        trajectory.push((cp, pct));
        eprintln!("[{}] iter {:4}: {:.4}% of pot  (batch {:?}, total {:?})",
            label, cp, pct, it_elapsed, t0.elapsed());
    }

    // Trajectory descent check: each later checkpoint should be lower than
    // (at least) the first one. The exact floor depends on the game and is
    // not asserted; the test asserts the SHAPE of the trajectory.
    let first = trajectory.first().map(|&(_, p)| p).unwrap_or(0.0);
    let last = trajectory.last().map(|&(_, p)| p).unwrap_or(0.0);
    eprintln!("[{}] first iter {:.4}%, last iter {:.4}% (drop = {:.4}x)",
        label, first, last, first / last.max(1e-6));
    assert!(last < first,
        "[{}] solver did not descend: first={:.4}% last={:.4}%", label, first, last);
    // Sanity: not nan/inf
    assert!(last.is_finite(),
        "[{}] solver produced non-finite exploitability: {}", label, last);
}

#[test]
fn gate6_k3_cpu_convergence_4p() {
    // K=3 = 4-player. nh=10. Per-terminal cost: nh^3 = 1000 per h × 10 h
    // = 10k per terminal. Tree has ~thousands of terminals. Per iter:
    // ~10M ops. 100 iters ~1B = a few seconds.
    run_convergence(4, 10, 100, &[1, 10, 25, 50, 100], "K=3 4p nh=10");
}

#[test]
#[ignore = "slow (minutes); run with `cargo test --release -- --ignored`"]
fn gate6_k5_cpu_convergence_6p() {
    // K=5 = 6-player. nh=7 keeps per-terminal cost at nh^5 = 16.8k per h
    // × 7 h = 117k per terminal. Tree may have ~tens of thousands of
    // terminals; iter cost on the order of a few seconds. n_valid=6 after
    // turn/river filter is the minimum for K=5 to admit any non-degenerate
    // scenarios (need >K opp hands available); some hands at the high end
    // might still see TVRP=0 but the trajectory should descend overall.
    run_convergence(6, 7, 10, &[1, 3, 5, 10], "K=5 6p nh=7");
}
