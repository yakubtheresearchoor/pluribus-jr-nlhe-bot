//! Slice 2 definition-of-done gate: CPU↔Metal parity at rake ≠ 0.
//!
//! Per user direction: "The definition of done for the Metal rake is the
//! rake≠0 parity gate passing against the CPU reference at f32 floor.
//! The rake=0 gates validate nothing about rake (the rake term vanishes
//! on both sides at rake=0), so the rake≠0 gate is the only real
//! validation, and the CPU side is the proven-against-hand-computation
//! reference, so a rake≠0 parity failure is unambiguously the Metal side."
//!
//! With CUDA deleted (commit d7e0a8b), Slice 2 collapses to a SINGLE
//! GPU site: vcfr.metal. This file is the bar.
//!
//! ## Gate coverage (the lesson from the discipline check)
//!
//! Per user direction: "The gate must exercise every site you implement,
//! so add gate scenarios to cover all five payoff paths (fold-win,
//! equal-contributions showdown, HU per-level, unequal-contributions
//! side-pot, K≥3 factored), because a single-scenario gate validates
//! only the sites that scenario reaches and could go green while an
//! unexercised site is wrong."
//!
//! Site map (each maps to one branch in `multiway_brute_force_showdown`
//! in vcfr.metal):
//!
//! | Site | Branch | Where the scenario exercises it |
//! |------|--------|--------------------------------|
//! | (a)  | K=2 all-equal brute (line ~187)        | 3p showdown, equal contribs |
//! | (b)  | K=2 fold-win fast (line ~243)          | 3p with fold → num_active≤1 |
//! | (c)  | K=2 unequal/side-pot (line ~281)       | 3p with allin → side pots |
//! | (d)  | K=1 HU (line ~376)                     | HU (2-player) showdown |
//! | (e)  | K≥3 factored (line ~447, ~3113)        | 4p showdown |
//!
//! All scenarios run with `rake_rate=0.05, rake_cap=1000` (cap unbinding
//! so the gate is sensitive to rake_rate proportionality; a separate
//! variant could exercise cap-binding behavior).
//!
//! ## States
//!
//! BEFORE Phase B math lands (current):
//!   - CPU applies rake (Slice 1.x hand-anchored; threaded through
//!     evaluate_terminal at Slice 1.6, commit 62cc5bc)
//!   - Metal does NOT apply rake (vcfr.metal sites unchanged from rake=0)
//!   - rake≠0 tests should FAIL (CPU CFV diverges by the rake amount)
//!   - All rake≠0 tests `#[ignore]` so CI stays green; each is runnable
//!     manually as the diagnostic that confirms its site's gap exists
//!
//! AFTER Phase B math lands:
//!   - Both CPU and Metal apply rake (matching the rake spec: main pot
//!     only, single cap per hand, no-flop-no-drop)
//!   - All rake≠0 tests pass at f32 floor with CPU-reference f64-confirmed
//!   - Remove `#[ignore]` to make each a permanent regression gate
//!
//! ## f64 discriminator (Metal-appropriate form)
//!
//! Per user direction: "The Metal kernel runs f32, so a CPU-Metal parity
//! comparison is f32-vs-f32, and you cannot run the Metal side in f64.
//! So the f64 discriminator for the Metal rake has to be structured
//! carefully: confirm the CPU rake at the same scenario, computed in
//! f64, matches the CPU rake in f32 to f64-floor (proving the CPU rake
//! is exact), and then the Metal-vs-CPU f32 parity diff should be at
//! the f32-accumulation floor, and if it is above that floor it is a
//! Metal rake bug. So the discriminator is: CPU-rake-f64 confirms the
//! reference is exact, then Metal-vs-CPU-f32 should be at f32 floor,
//! and a diff above f32 floor (when the reference is confirmed exact
//! in f64) is unambiguously a Metal arithmetic error."
//!
//! Do NOT expect the P2.5a-style clean collapse to 2e-13. Metal is
//! f32; the floor is the expected resolution. The discipline is the
//! CPU-reference-is-f64-exact framing, not a magnitude argument.
//!
//! ## Preflop discriminant (deferred validation)
//!
//! Per user direction: "Verify the Preflop board_state value on the
//! Metal side matches the hardcoded check, and explicitly note the
//! flop_seen path as dormant-untested-until-preflop, so it is a
//! known-deferred-validation item rather than assumed-correct."
//!
//! Verified at commit time: `BoardState::Preflop = 3` with `#[repr(u8)]`
//! in solver-core/src/tree/action.rs:27. So the Metal kernel's
//! `flop_seen = (node.board_state != 3)` is correct against the current
//! enum encoding. A static-assert anchor lives in the kernel-host
//! plumbing (see flop_solver.rs).
//!
//! HOWEVER: the no-flop-no-drop path is dormant-untested in current
//! flop-onward trees because no preflop terminals exist (Task #41,
//! preflop integration, is in-progress). The flop_seen logic ships
//! correct-by-construction but un-exercised by these gate scenarios
//! until preflop terminals exist. This is a known-deferred-validation
//! item; the preflop integration milestone (P2.5/P5) must include a
//! gate scenario with a preflop-fold terminal at rake≠0 to convert
//! "correct by construction" into "validated".

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

// ─────────────────────────────────────────────────────────────────────
// Preflop-discriminant static anchor.
//
// The Metal kernel uses `node.board_state != 3` to detect "not preflop"
// for the no-flop-no-drop gate. If the BoardState repr ever changes,
// this assertion fires at test-binary load — the magic-3 in the kernel
// stops being correct silently. Catching the drift here (cheap) is
// better than discovering it only at preflop integration.
// ─────────────────────────────────────────────────────────────────────
const _BOARD_STATE_PREFLOP_ANCHOR: () = {
    assert!(BoardState::Preflop as u8 == 3,
        "BoardState::Preflop repr changed; update vcfr.metal `flop_seen` check");
};

fn find_pair_index(c1: Card, c2: Card) -> u16 {
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (a, b) = index_to_card_pair(idx);
        if (a == c1 as u8 && b == c2 as u8) || (a == c2 as u8 && b == c1 as u8) {
            return idx as u16;
        }
    }
    panic!("pair not found");
}

/// Generalized manual chance-table + tree builder for N-player parity
/// gates. Mirrors `six_player_iter0_parity::build_6p_table`'s structure
/// (recursive `enum_nc` for num_combinations; sorted-by-rank turn/river
/// items for arbitrary num_opp).
///
/// Parameters:
/// - `num_players`: 2..=6
/// - `nh`: number of distinct hands in the per-flop table
/// - `rake_rate`, `rake_cap`: passed through to TreeConfig
/// - `bet_sizes`: TreeConfig bet/raise options
/// - `initial_contributions`: per-player contributions at the flop-start root
/// - `starting_pot`: TreeConfig starting_pot
/// - `starting_stacks`: per-player remaining stacks
fn build_np_with_rake(
    num_players: u8,
    nh: usize,
    rake_rate: f64,
    rake_cap: f64,
    bet_sizes: BetSizeOptions,
    initial_contributions: Vec<i32>,
    starting_pot: i32,
    starting_stacks: Vec<i32>,
) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_opp = (num_players - 1) as usize;

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = (all_valid.len() / nh).max(1);
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1; hand_cards[i * 2 + 1] = c2;
    }
    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh { for j in 0..nh {
        if i == j { conflict[i*nh+j] = 1; continue; }
        let (a1,a2) = index_to_card_pair(chosen[i] as usize);
        let (b1,b2) = index_to_card_pair(chosen[j] as usize);
        if a1==b1 || a1==b2 || a2==b1 || a2==b2 { conflict[i*nh+j] = 1; }
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
                .map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; num_players as usize];

    // num_combinations exact via per-player non-conflicting hand enumeration.
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
        hand_ranks_base: hr,
        valid_hand_indices: chosen,
        num_valid: nh,
        conflict,
        hand_cards,
        remaining_deck: tc,
        turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw,
        num_players,
        num_combinations: nc,
        river_decks: rd,
    };
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot,
        starting_stacks, initial_contributions,
        rake_rate, rake_cap,
        bet_sizes,
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

/// LEGACY (kept for compatibility with the existing baseline test): the
/// 6-chosen-hand 3p scenario. Identical shape to three_max_parity's
/// known-good pattern.
fn build_3player_with_rake(rake_rate: f64, rake_cap: f64)
    -> (FlatTree, FlopChanceTable)
{
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
    let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

    let chosen_hands: Vec<u16> = vec![
        find_pair_index(card_from_str("Ah").unwrap(), card_from_str("Qc").unwrap()),
        find_pair_index(card_from_str("Jd").unwrap(), card_from_str("Ts").unwrap()),
        find_pair_index(card_from_str("9h").unwrap(), card_from_str("8c").unwrap()),
        find_pair_index(card_from_str("As").unwrap(), card_from_str("Qd").unwrap()),
        find_pair_index(card_from_str("Jc").unwrap(), card_from_str("Th").unwrap()),
        find_pair_index(card_from_str("9s").unwrap(), card_from_str("8d").unwrap()),
    ];

    let nh = chosen_hands.len();
    let num_players = 3u8;
    let num_opp = 2;
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

    let initial_weights = vec![vec![1.0f32; nh]; num_players as usize];
    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let mask0: u64 = (1u64 << hand_cards[h0 * 2]) | (1u64 << hand_cards[h0 * 2 + 1]);
        for h1 in 0..nh {
            let mask1: u64 = (1u64 << hand_cards[h1 * 2]) | (1u64 << hand_cards[h1 * 2 + 1]);
            if mask0 & mask1 != 0 { continue; }
            for h2 in 0..nh {
                let mask2: u64 = (1u64 << hand_cards[h2 * 2]) | (1u64 << hand_cards[h2 * 2 + 1]);
                if mask0 & mask2 != 0 || mask1 & mask2 != 0 { continue; }
                nc += 1.0;
            }
        }
    }

    let table = FlopChanceTable {
        hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
        remaining_deck: turn_cards.clone(), turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx, initial_weights, num_players,
        num_combinations: nc, river_decks,
    };
    let config = TreeConfig {
        num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![100, 100, 100], initial_contributions: vec![5, 5, 5],
        rake_rate, rake_cap,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

/// Compare CPU per-zone regrets against GPU's concatenated buffer.
/// GPU layout: [flop | turn | river]. Returns max abs diff over all
/// zones plus the worst-diff zone label and within-zone index.
fn compare_regrets_per_zone(
    cpu: &FlopStartVectorCfr,
    gpu_all: &[f32],
) -> (f32, &'static str, usize) {
    let cpu_flop = cpu.regrets_flop();
    let cpu_turn = cpu.regrets_turn();
    let cpu_river = cpu.regrets_river();
    let fl = cpu_flop.len();
    let tl = cpu_turn.len();

    let mut overall_max = 0.0f32;
    let mut overall_label: &'static str = "flop";
    let mut overall_idx: usize = 0;

    for (cpu_slice, gpu_start, label) in [
        (cpu_flop,  0,        "flop"),
        (cpu_turn,  fl,       "turn"),
        (cpu_river, fl + tl,  "river"),
    ].iter() {
        for i in 0..cpu_slice.len() {
            let gi = gpu_start + i;
            if gi >= gpu_all.len() { break; }
            let diff = (cpu_slice[i] - gpu_all[gi]).abs();
            if diff > overall_max {
                overall_max = diff;
                overall_label = label;
                overall_idx = i;
            }
        }
    }
    (overall_max, overall_label, overall_idx)
}

/// Run CPU+Metal for one iteration and report max regret diff across
/// all zones. Returns the same triple as compare_regrets_per_zone.
fn run_parity(tree: &FlatTree, table: FlopChanceTable, label: &str)
    -> (f32, &'static str, usize)
{
    let game = FlopStartGame::new(table);
    let mut cpu = FlopStartVectorCfr::new(tree, game.table());
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, tree, &game, &cpu);

    cpu.run(tree, &game, 1);
    gpu.run(&ctx, tree, &game, 1);
    let gpu_reg = gpu.download_regrets();
    let (max_diff, zone, argmax) = compare_regrets_per_zone(&cpu, &gpu_reg);
    eprintln!("[{}] CPU↔Metal max regret diff = {} (zone={}, idx={})",
        label, max_diff, zone, argmax);
    (max_diff, zone, argmax)
}

// ═════════════════════════════════════════════════════════════════════
// rake=0 baseline (must always pass; foundation under every rake≠0 gate)
// ═════════════════════════════════════════════════════════════════════

#[test]
fn cpu_metal_parity_at_rake_0_baseline() {
    // SANITY CHECK: at rake=0, CPU↔Metal MUST agree. If this fails,
    // infrastructure is broken — not rake. This guards against
    // regressions in the data path (the Phase A plumbing must not
    // disturb rake=0 behavior).
    let (tree, table) = build_3player_with_rake(0.0, 0.0);
    let (max_diff, zone, argmax) = run_parity(&tree, table, "rake=0 3p baseline");

    let f32_floor_tol = 1e-4_f32;
    assert!(
        max_diff < f32_floor_tol,
        "rake=0 baseline FAILED with max_diff = {} > {} (zone={}, idx={}). \
         CPU↔Metal infrastructure broken — not a rake issue.",
        max_diff, f32_floor_tol, zone, argmax,
    );
    eprintln!("✓ rake=0 3p baseline OK ({}); test infrastructure works", max_diff);
}

// ═════════════════════════════════════════════════════════════════════
// Phase B targets: one gate per Metal showdown payoff site.
// Each is #[ignore] until Phase B closes its corresponding site.
// ═════════════════════════════════════════════════════════════════════

/// Site (a) — K=2 all-equal brute (vcfr.metal line ~187)
/// 3p equal-contributions showdown (existing baseline scenario at rake≠0).
#[test]
#[ignore = "Slice 2 Phase B Site (a): K=2 all-equal brute. Enable when Metal rake \
            lands at this site. Run: cargo test --release --features metal --test \
            gpu_rake_parity_gate site_a_3p_equal_showdown_rake -- --ignored"]
fn site_a_3p_equal_showdown_rake() {
    let (tree, table) = build_3player_with_rake(0.05, 1000.0);
    let (max_diff, zone, argmax) = run_parity(&tree, table, "Site (a) 3p equal-showdown rake=5%");
    assert!(max_diff < 1e-4, "Site (a) FAILED: max_diff = {} (zone={}, idx={})",
        max_diff, zone, argmax);
}

/// Site (b) — K=2 fold-win fast path (vcfr.metal line ~243)
/// 3p with bet/fold sequence creating fold terminals (num_active<=1).
/// Reuses the 3p baseline tree which contains fold terminals naturally.
/// The fold-win path is exercised when the CFR walk hits a terminal
/// with fold_mask covering 2 of 3 players (one survivor).
///
/// EMPIRICAL OBSERVATION (today, pre-Phase-B): this gate produces
/// max_diff = 0.21875 at flop idx=96 — IDENTICAL to Site (a)'s. The
/// max-regret position is dominated by the all-active showdown path
/// (Site a), not the fold-win path. Fold terminals are exercised but
/// not numerically dominant in this uniform-strategy iter-1 setup.
///
/// COVERAGE STATUS: this gate exercises the Site (b) kernel branch
/// (fold terminals exist in the tree; the K=2 fold-win fast path
/// runs at them) but does not isolate it from Site (a) in the max-
/// diff signal. The discriminating test is:
///
/// 1. Phase B closes both Site (a) and Site (b) branches together
///    (they share the eff_rake math at function entry).
/// 2. If only Site (a)'s math lands and Site (b) is left wrong, the
///    diffs at this gate vs Site (a)'s gate will DIVERGE — which is
///    itself the discriminating signal.
///
/// So coverage is achieved by joint convergence: Site (b)'s separate
/// kernel branch is exercised, but its math must be implemented in
/// the same Phase B pass as Site (a)'s. A separate Site (b)-dominant
/// scenario (where fold-terminal payoff overwhelms showdown payoff)
/// would require a strategy-biased tree which iter-1 uniform
/// strategy doesn't naturally produce.
#[test]
#[ignore = "Slice 2 Phase B Site (b): K=2 fold-win fast path. Enable when Metal \
            rake lands at the fold-win site. Run: cargo test --release \
            --features metal --test gpu_rake_parity_gate site_b -- --ignored"]
fn site_b_3p_fold_terminal_rake() {
    // ISOLATION STRATEGY — cap-binding rake:
    //
    // At rake_cap = 1.0 (small, binds at EVERY terminal in this tree
    // because all pots exceed 20 chips × 5% = 1.0), the rake amount
    // is the same CONSTANT (=cap) at every fold AND every showdown
    // terminal. This eliminates the "showdown has bigger pot → bigger
    // rake error → dominates" asymmetry that made the previous
    // version of this gate (using rake_cap=1000) measure site (a)'s
    // error bleeding through site (b)'s scenario.
    //
    // With uniform per-terminal rake error (=cap), the max diff is
    // now driven by REACH MASS through each terminal type, not by
    // pot size. Fold-terminal reach and showdown-terminal reach are
    // both meaningful at iter-1 uniform strategy, so site (b)'s
    // contribution is now arithmetically comparable to site (a)'s
    // rather than dominated by it.
    //
    // This is the achievable improvement WITHOUT per-terminal CFV
    // instrumentation. It is not perfect isolation (the gate still
    // sees a mix of site (a) + site (b) error at decision nodes
    // whose actions lead to both terminal types), but it is
    // strictly BETTER than the prior overlap because the dominant-
    // error terminal type is no longer pre-determined by pot size.
    //
    // EMPIRICAL FINDING (after Phase B site (b) closure):
    //
    // This gate did NOT drop when site (b)'s kernel math landed and
    // the unit test went to f32 floor. The diff stayed at 0.20833 at
    // flop idx=96 even with site (b) demonstrably correct (unit
    // test = 0.0).
    //
    // Interpretation: this gate's max-diff position (idx 96) is
    // dominated by sites (a) and (c) errors (rake-free showdowns and
    // 1-folded terminals), NOT by site (b)'s lone-survivor
    // contribution. Even with cap binding shifting per-terminal rake
    // magnitudes to uniform, the dominant error term is from
    // terminal types that don't route to site (b)'s branch.
    //
    // So this gate is NOT actually measuring site (b)'s correctness.
    // It measures the OVERALL CFR-regret rake error across all
    // K=2-cluster terminals (sites a + b + c collectively). It will
    // converge to f32 floor only after ALL of sites (a), (b), (c)
    // are closed — making it a JOINT-K=2 convergence gate, not a
    // site-(b)-specific one.
    //
    // The REAL site (b) validation is `site_b_isolated_kernel_unit_test`
    // (defined below), which directly invokes site (b)'s branch via
    // the debug kernel and compares against CPU truth. That test
    // went from max_diff=2.0 (today, before Phase B) to max_diff=0.0
    // (after Phase B Site (b) closure) with the kernel rake math.
    //
    // This gate is kept under the "site (b)" name because (i) it
    // sits in the K=2-cluster joint signal and (ii) when Phase B
    // closes sites (a) and (c) too, this gate becomes the f32-floor
    // gate for the K=2 cluster as a whole. But the per-SITE site (b)
    // DoD is the unit test, not this regret-level gate.
    let (tree, table) = build_3player_with_rake(0.05, 1.0);
    let (max_diff, zone, argmax) = run_parity(&tree, table, "Site (b) 3p fold-term rake_cap=1.0 (binding)");
    assert!(max_diff < 1e-4, "Site (b) FAILED: max_diff = {} (zone={}, idx={})",
        max_diff, zone, argmax);
}

/// Site (c) — K=2 unequal-contributions / side-pot path (vcfr.metal line ~281)
/// 3p with AllIn bet enabled, generating unequal contributions and
/// thus side-pots at terminals.
#[test]
#[ignore = "Slice 2 Phase B Site (c): K=2 unequal/side-pot path. Enable when Metal \
            rake lands at this site. Run: cargo test --release --features metal \
            --test gpu_rake_parity_gate site_c -- --ignored"]
fn site_c_3p_sidepot_rake() {
    let bet_sizes = BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0), BetSize::AllIn],
        raise: vec![BetSize::AllIn],
    };
    // Stacks unequal so an allin call short of opponent creates side pot.
    let (tree, table) = build_np_with_rake(
        3, 6, 0.05, 1000.0,
        bet_sizes,
        vec![5, 5, 5],
        15,
        vec![50, 100, 200],
    );
    let (max_diff, zone, argmax) = run_parity(&tree, table, "Site (c) 3p side-pot rake=5%");
    assert!(max_diff < 1e-4, "Site (c) FAILED: max_diff = {} (zone={}, idx={})",
        max_diff, zone, argmax);
}

/// Site (d) — K=1 HU per-level (vcfr.metal line ~376)
/// 2-player tree. The HU showdown path uses per-level cash
/// accumulation (single-opponent case in multiway_brute_force_showdown).
#[test]
#[ignore = "Slice 2 Phase B Site (d): K=1 HU per-level. Enable when Metal rake \
            lands at the HU site. Run: cargo test --release --features metal \
            --test gpu_rake_parity_gate site_d -- --ignored"]
fn site_d_hu_rake() {
    let (tree, table) = build_np_with_rake(
        2, 6, 0.05, 1000.0,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        vec![5, 5],
        10,
        vec![100, 100],
    );
    let (max_diff, zone, argmax) = run_parity(&tree, table, "Site (d) HU rake=5%");
    assert!(max_diff < 1e-4, "Site (d) FAILED: max_diff = {} (zone={}, idx={})",
        max_diff, zone, argmax);
}

/// Site (e) — K≥3 factored per-level (vcfr.metal line ~447, factored_showdown_unified line ~3113)
/// 4-player tree. K=3 (num_opp=3) triggers the factored share path in
/// multiway_brute_force_showdown, which dispatches to
/// factored_share_for_level_thread for per-level eligibility math.
#[test]
#[ignore = "Slice 2 Phase B Site (e): K≥3 factored per-level. Enable when Metal \
            rake lands at the factored site. Run: cargo test --release \
            --features metal --test gpu_rake_parity_gate site_e -- --ignored"]
fn site_e_4p_factored_rake() {
    let (tree, table) = build_np_with_rake(
        4, 6, 0.05, 1000.0,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        vec![5, 5, 5, 5],
        20,
        vec![100, 100, 100, 100],
    );
    let (max_diff, zone, argmax) = run_parity(&tree, table, "Site (e) 4p factored rake=5%");
    assert!(max_diff < 1e-4, "Site (e) FAILED: max_diff = {} (zone={}, idx={})",
        max_diff, zone, argmax);
}

// ═════════════════════════════════════════════════════════════════════
// Site (e) PRIMARY ISOLATION TEST — direct CPU↔Metal kernel unit check
//
// Routing to site (e)'s K≥3 factored branch (line ~528 in
// multiway_brute_force_showdown post-deletion):
//   - np = 4 (num_opp = 3 → K≥3 → enters factored path)
//   - fold_mask = 0 (no folds, all-equal contribs)
//   - traverser = 0
//   - → kernel falls through K=1/K=2 branches, enters K≥3 factored
//
// CPU reference: side_pot_showdown_cfv_with_rake at the K≥3
// per-level brute-force path (showdown.rs ~870-910, main-pot-only,
// cap-once). The factored Metal implementation produces the same
// CFV via the K-1 recursive expansion with eligibility-restricted
// strength comparison.
//
// Today (pre-Phase-B-Step-4): Metal site (e) is rake-free; CPU
// applies main-pot rake. Diff = main-pot rake fraction × reach.
// After Step 4 lands: diff → f32 floor.
// ═════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "Slice 2 Phase B Site (e) ISOLATION (primary): direct CPU↔Metal \
            kernel-level check of site (e)'s K≥3 factored branch rake math. \
            np=4, fold_mask=0, equal contribs → routes through factored \
            per-level. Run: cargo test --release --features metal --test \
            gpu_rake_parity_gate site_e_isolated -- --ignored"]
fn site_e_isolated_kernel_unit_test() {
    use solver_core::solver::showdown::side_pot_showdown_cfv_with_rake;

    let nh = 4;
    let np = 4;
    let num_opp = 3;
    // 4 hands using 8 distinct cards (no conflicts).
    let hand_cards: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let strengths: Vec<u16> = vec![10, 20, 30, 40];
    let (pl_str, pl_idx) = debug_kernel::make_sorted(&strengths);

    let opp_reach: Vec<f32> = vec![1.0; num_opp * nh];
    let contributions: Vec<i32> = vec![5, 5, 5, 5];  // all equal → K≥3 factored
    let starting_pot: i32 = 20;
    let fold_mask: u16 = 0;
    let traverser = 0;

    let rake_rate = 0.05_f32;
    let rake_cap = 1.0_f32;
    let flop_seen = true;

    let opp_reach_per_opp: Vec<Vec<f32>> = (0..num_opp)
        .map(|oi| opp_reach[oi * nh..(oi + 1) * nh].to_vec())
        .collect();
    let opp_reach_views: Vec<&[f32]> = opp_reach_per_opp.iter().map(|v| v.as_slice()).collect();
    let mut sorted_opp_str = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx = Vec::with_capacity(num_opp * nh);
    for _ in 0..num_opp {
        sorted_opp_str.extend_from_slice(&pl_str);
        sorted_opp_idx.extend_from_slice(&pl_idx);
    }
    let cpu_cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &pl_str, &pl_idx,
        &contributions, fold_mask, traverser, np as u8, starting_pot,
        rake_rate, rake_cap, flop_seen,
    );

    let ctx = MetalContext::new().expect("Metal context");
    let gpu_cfv = debug_kernel::gpu_brute_force_with_rake(
        &ctx, nh, np, traverser, starting_pot, fold_mask,
        &opp_reach, &contributions, &hand_cards, &pl_str, &pl_idx,
        rake_rate, rake_cap, flop_seen,
    );

    eprintln!("Site (e) K≥3 factored isolated kernel unit test:");
    eprintln!("  CPU (with rake): {:?}", cpu_cfv);
    eprintln!("  Metal (with rake): {:?}", gpu_cfv);

    let mut max_diff = 0.0_f32;
    let mut max_h = 0;
    for h in 0..nh {
        let d = (cpu_cfv[h] - gpu_cfv[h]).abs();
        if d > max_diff { max_diff = d; max_h = h; }
    }
    eprintln!("  max_diff = {} at h={}", max_diff, max_h);

    assert!(max_diff < 1e-4,
        "Site (e) isolated unit test FAILED: max_diff = {} at h={}. \
         CPU computes K≥3 factored with main-pot-only rake; Metal \
         computes it. Phase B Step 4 must add rake math at K≥3 \
         factored branch in multiway_brute_force_showdown line ~528.",
        max_diff, max_h);
    eprintln!("✓ Site (e) isolated unit test PASSED");
}

// ═════════════════════════════════════════════════════════════════════
// Site (d) DIAGNOSTIC ISOLATION TEST — discriminates kernel-math
// correctness from gate signal coverage
//
// Per user direction added during Phase B: "(d) HU is okay with
// gate-primary plus per-site f64 (genuinely isolated, no num_opp=1
// sibling) ... if either's gate behaves unexpectedly when closed
// (doesn't drop, or moves another), that's the trigger to add their
// unit test too."
//
// Phase B Site (d) kernel rake was applied to `multiway_brute_force_
// showdown`'s K=1 branch (vcfr.metal ~376). Result: HU gate dropped
// 0.73333 → 0.09375 — significantly but NOT to f32 floor. That's
// the user-predicted unexpected behavior, and the trigger for this
// unit test.
//
// What it discriminates:
//   - If THIS test → 0.0 (multiway K=1 correct in isolation) but
//     gate stays at 0.09: site (d) has a SECOND code location not
//     yet mirrored. Likely candidate: `sorted_sweep_showdown_vcfr_
//     local` (lines ~1155, ~1600, ~1706, ~2132) which is a separate
//     helper called at HU showdown sites in production kernels and
//     is currently rake-free.
//   - If THIS test ALSO fails: the multiway K=1 rake math itself
//     is wrong; the gate residual is honest.
//
// Either result is informative. The user's discipline applied:
// "The principle 'trigger is the measurement, not the assumption'
// is right, and applied correctly it says..." — measurement
// triggered, so we measure.
// ═════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "Slice 2 Phase B Site (d) DIAGNOSTIC: HU multiway K=1 isolation. \
            Discriminates kernel-math correctness from gate signal coverage \
            after the HU gate behaved unexpectedly (dropped substantially \
            but not to f32 floor) post site (d) closure. Run: cargo test \
            --release --features metal --test gpu_rake_parity_gate \
            site_d_isolated -- --ignored"]
fn site_d_isolated_kernel_unit_test() {
    use solver_core::solver::showdown::side_pot_showdown_cfv_with_rake;

    // np=2 → num_opp=1 → routes to multiway K=1 branch (site d).
    let nh = 3;
    let np = 2;
    let num_opp = 1;
    let hand_cards: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
    let strengths: Vec<u16> = vec![10, 20, 30];
    let (pl_str, pl_idx) = debug_kernel::make_sorted(&strengths);

    let opp_reach: Vec<f32> = vec![1.0; num_opp * nh];
    let contributions: Vec<i32> = vec![5, 5];  // HU equal
    let starting_pot: i32 = 10;
    let fold_mask: u16 = 0;
    let traverser = 0;

    let rake_rate = 0.05_f32;
    let rake_cap = 1.0_f32;
    let flop_seen = true;

    let opp_reach_per_opp: Vec<Vec<f32>> = (0..num_opp)
        .map(|oi| opp_reach[oi * nh..(oi + 1) * nh].to_vec())
        .collect();
    let opp_reach_views: Vec<&[f32]> = opp_reach_per_opp.iter().map(|v| v.as_slice()).collect();
    let mut sorted_opp_str = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx = Vec::with_capacity(num_opp * nh);
    for _ in 0..num_opp {
        sorted_opp_str.extend_from_slice(&pl_str);
        sorted_opp_idx.extend_from_slice(&pl_idx);
    }
    let cpu_cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &pl_str, &pl_idx,
        &contributions, fold_mask, traverser, np as u8, starting_pot,
        rake_rate, rake_cap, flop_seen,
    );

    let ctx = MetalContext::new().expect("Metal context");
    let gpu_cfv = debug_kernel::gpu_brute_force_with_rake(
        &ctx, nh, np, traverser, starting_pot, fold_mask,
        &opp_reach, &contributions, &hand_cards, &pl_str, &pl_idx,
        rake_rate, rake_cap, flop_seen,
    );

    eprintln!("Site (d) HU multiway K=1 isolated kernel unit test:");
    eprintln!("  CPU (with rake): {:?}", cpu_cfv);
    eprintln!("  Metal (multiway K=1 with rake): {:?}", gpu_cfv);

    let mut max_diff = 0.0_f32;
    let mut max_h = 0;
    for h in 0..nh {
        let d = (cpu_cfv[h] - gpu_cfv[h]).abs();
        if d > max_diff { max_diff = d; max_h = h; }
    }
    eprintln!("  max_diff = {} at h={}", max_diff, max_h);
    eprintln!("  → If 0.0: multiway K=1 math correct. Gate residual is from sorted_sweep_showdown_vcfr_local (separate HU helper, currently rake-free).");
    eprintln!("  → If non-zero: multiway K=1 math itself is wrong.");

    assert!(max_diff < 1e-4,
        "Site (d) HU multiway K=1 isolated unit test FAILED: max_diff = {} \
         at h={}. If gate dropped substantially but not to floor AND this \
         test fails, multiway K=1 has its own math bug. If gate dropped \
         but this test PASSES, the residual is from sorted_sweep_showdown_\
         vcfr_local (separate HU helper that's not yet rake-mirrored).",
        max_diff, max_h);
    eprintln!("✓ Site (d) HU multiway K=1 isolated unit test PASSED");
}

// ═════════════════════════════════════════════════════════════════════
// Site (d) TIE-BAND ISOLATION TEST — catches the +reach correction
//
// Per user direction for Phase B Site (d) part 2: "include a tie-band
// scenario in the site (d) unit test, because the tie correction is
// the part most likely to be subtly wrong and the part that the
// non-tie cases would not catch."
//
// The +reach inclusion-exclusion correction (audit-fix #37) only
// fires when there are TIES in opp_str == pl_str. The distinct-
// strengths unit test above will NEVER exercise it. A tie-band test
// is therefore the targeted defense for the sorted-sweep rake mirror
// (Phase B Site (d) part 2), since that's the location where the
// 3-component decomposition (sweep_net, win_reach, tie_reach) carries
// the +reach correction in tie_reach.
//
// This test is needed BEFORE site (d) part 2 lands. Today it runs
// against the multiway K=1 branch (which doesn't have the tie-band
// subtlety in the same form — it does brute-force per-(g_a)
// enumeration), so it's also a useful coverage extension for the
// multiway K=1 site's tie handling.
//
// SCENARIO: nh=3 with TWO hands tied at top strength. Strengths
// [10, 30, 30] mean h=1 and h=2 are tied for top. Site (d) lone-
// survivor path runs (no fold variance in HU showdown), so the test
// validates the (currently-correct, but-with-this-new-scenario)
// tie handling at the multiway K=1 branch. After site (d) part 2,
// it ALSO validates the sorted_sweep tie-band rake correction.
// ═════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "Slice 2 Phase B Site (d) TIE-BAND ISOLATION: validates HU tie-band \
            arithmetic at multiway K=1 (currently passes, but extends coverage \
            beyond distinct-strengths case). Also serves as the targeted defense \
            for site (d) part 2 (sorted_sweep rake mirror) where the +reach \
            inclusion-exclusion correction lives — that correction only fires \
            in tie bands and the non-tie unit test would not catch errors in it. \
            Run: cargo test --release --features metal --test \
            gpu_rake_parity_gate site_d_tie -- --ignored"]
fn site_d_tie_band_isolated_kernel_unit_test() {
    use solver_core::solver::showdown::side_pot_showdown_cfv_with_rake;

    let nh = 3;
    let np = 2;
    let num_opp = 1;
    let hand_cards: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
    // TIE BAND: h=1 and h=2 share strength 30 (top). h=0 has strength 10.
    // This exercises the tied-at-top accounting that the distinct-
    // strengths unit test never reaches.
    let strengths: Vec<u16> = vec![10, 30, 30];
    let (pl_str, pl_idx) = debug_kernel::make_sorted(&strengths);

    let opp_reach: Vec<f32> = vec![1.0; num_opp * nh];
    let contributions: Vec<i32> = vec![5, 5];
    let starting_pot: i32 = 10;
    let fold_mask: u16 = 0;
    let traverser = 0;

    let rake_rate = 0.05_f32;
    let rake_cap = 1.0_f32;
    let flop_seen = true;

    let opp_reach_per_opp: Vec<Vec<f32>> = (0..num_opp)
        .map(|oi| opp_reach[oi * nh..(oi + 1) * nh].to_vec())
        .collect();
    let opp_reach_views: Vec<&[f32]> = opp_reach_per_opp.iter().map(|v| v.as_slice()).collect();
    let mut sorted_opp_str = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx = Vec::with_capacity(num_opp * nh);
    for _ in 0..num_opp {
        sorted_opp_str.extend_from_slice(&pl_str);
        sorted_opp_idx.extend_from_slice(&pl_idx);
    }
    let cpu_cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &pl_str, &pl_idx,
        &contributions, fold_mask, traverser, np as u8, starting_pot,
        rake_rate, rake_cap, flop_seen,
    );

    let ctx = MetalContext::new().expect("Metal context");
    let gpu_cfv = debug_kernel::gpu_brute_force_with_rake(
        &ctx, nh, np, traverser, starting_pot, fold_mask,
        &opp_reach, &contributions, &hand_cards, &pl_str, &pl_idx,
        rake_rate, rake_cap, flop_seen,
    );

    eprintln!("Site (d) HU TIE-BAND isolated kernel unit test:");
    eprintln!("  CPU (with rake):    {:?}", cpu_cfv);
    eprintln!("  Metal (with rake):  {:?}", gpu_cfv);

    let mut max_diff = 0.0_f32;
    let mut max_h = 0;
    for h in 0..nh {
        let d = (cpu_cfv[h] - gpu_cfv[h]).abs();
        if d > max_diff { max_diff = d; max_h = h; }
    }
    eprintln!("  max_diff = {} at h={}", max_diff, max_h);

    assert!(max_diff < 1e-4,
        "Site (d) HU tie-band isolated unit test FAILED: max_diff = {} \
         at h={}. With tied strengths at top (h=1, h=2 both at strength \
         30), this scenario exercises the +reach inclusion-exclusion \
         correction in the tie band — the most likely place for sorted-\
         sweep rake arithmetic to be subtly wrong. CPU formula uses \
         self_correction = if tie_includes_self {{ reach[h] }} else {{ 0 }}; \
         Metal mirror must produce the same.",
        max_diff, max_h);
    eprintln!("✓ Site (d) HU tie-band isolated unit test PASSED");
}

// ═════════════════════════════════════════════════════════════════════
// HU RESIDUAL DIAGNOSTIC — fold-win-after-bet (unequal contribs)
//
// Investigation per the lead's elevated priority: the HU gate residual at
// 0.09375 with rake fully ruled out for the helper branches. Hypothesis:
// HU K=1 lone-survivor with UNEQUAL contribs (fold-after-bet) is the
// site that diverges from the K=2 fast path's rake formula.
//
// Scenario: HU (np=2), traverser=0 is lone survivor (P1 folded),
// contributions=[15, 5] (P0 bet 10 chips after blinds, P1 folded).
//
//   CPU fast path (showdown.rs ~484-548 — `num_active<=1 && num_opp<=2`):
//     total_pot = 10 + 15 + 5 = 30
//     rake = min(30 * 0.05, 1000) = 1.5
//     payoff = (30 - 1.5) - 20 = 8.5
//
//   Metal K=1 per-level (multiway helper line ~376, post Site (d) part 1):
//     main_pot_amount = 5 * 2 + 10 = 20 (called portion only)
//     main_pot_rake = min(20 * 0.05, 1000) = 1.0
//     li=0: pot_after_rake = 19, cash += 19
//     li=1 (P0 uncalled extra): pot_after_rake = 10 (no rake on side), cash += 10
//     net = 29 - 20 = 9.0
//
//   Diff per terminal = 0.5 = exactly (1.5 - 1.0) rake difference.
//
// If this test shows diff ≈ 0.5 (× reach factor), it CONFIRMS:
//   1. CPU rakes the full pot (including uncalled bets) for HU fold-win.
//   2. Metal K=1 rakes only the called portion (main_pot_only).
//   3. They disagree on this case, which is what shows up at the
//      HU gate residual 0.09375.
//
// Internal inconsistency in Metal: K=2 lone-survivor uses total-pot
// rake (mirroring CPU fast path) — that was Phase B Site (b) closure.
// K=1 lone-survivor uses main-pot-only rake (Phase B Site (d) part 1).
// They use DIFFERENT formulas for the same conceptual case (lone
// survivor with unequal contributions), one at K=1 and one at K=2.
//
// Which formula is right per the the rake spec ("main pot only") is
// for the lead to decide; this test demonstrates the divergence exists
// and localizes the source unambiguously.
// ═════════════════════════════════════════════════════════════════════

#[test]
fn hu_residual_fold_after_bet_diagnostic() {
    // POST-FIX (2026-06-04): the divergence is now GONE. This test was
    // originally written to DEMONSTRATE the bug exists (asserting
    // max_diff > 0.1). After the lead confirmed main-pot-only is the
    // correct spec, both CPU fast path and Metal K=2 fast path were
    // fixed to use main_pot_amount instead of total_pot for rake. The
    // diff is now at f32 floor.
    //
    // Test repurposed as a PERMANENT REGRESSION GATE: if either CPU or
    // Metal regresses to total-pot rake for fold-win-after-bet (which
    // would over-rake the uncalled bet), this test fails immediately.
    //
    // Setup: HU, P0 bet 15, P1 had 5 then folded. P0 is lone survivor.
    //   Per the rake spec (main-pot-only):
    //     main_pot = min(15,5) × 2 + starting_pot(10) = 20
    //     main_pot_rake = 20 × 0.05 = 1.0
    //     payoff = (total_pot=30 - 1.0) - traverser_investment(20) = 9.0
    //     CFV per hand = 9.0 × reach(2) = 18.0
    //   Buggy total-pot version would have given CFV = 17.0 (diff 1.0).
    use solver_core::solver::showdown::side_pot_showdown_cfv_with_rake;

    let nh = 3;
    let np = 2;
    let num_opp = 1;
    let hand_cards: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
    let strengths: Vec<u16> = vec![10, 20, 30];
    let (pl_str, pl_idx) = debug_kernel::make_sorted(&strengths);

    let opp_reach: Vec<f32> = vec![1.0; num_opp * nh];

    // Fold-after-bet: P0 bet, P1 folded. Contribs UNEQUAL.
    let contributions: Vec<i32> = vec![15, 5];
    let starting_pot: i32 = 10;
    // fold_mask = 0b10: P1 folded, traverser P0 is lone survivor.
    let fold_mask: u16 = 0b10;
    let traverser = 0;

    let rake_rate = 0.05_f32;
    let rake_cap = 1000.0_f32;  // uncapped — test pure rate formula
    let flop_seen = true;

    let opp_reach_per_opp: Vec<Vec<f32>> = (0..num_opp)
        .map(|oi| opp_reach[oi * nh..(oi + 1) * nh].to_vec())
        .collect();
    let opp_reach_views: Vec<&[f32]> = opp_reach_per_opp.iter().map(|v| v.as_slice()).collect();
    let mut sorted_opp_str = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx = Vec::with_capacity(num_opp * nh);
    for _ in 0..num_opp {
        sorted_opp_str.extend_from_slice(&pl_str);
        sorted_opp_idx.extend_from_slice(&pl_idx);
    }
    let cpu_cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &pl_str, &pl_idx,
        &contributions, fold_mask, traverser, np as u8, starting_pot,
        rake_rate, rake_cap, flop_seen,
    );

    let ctx = MetalContext::new().expect("Metal context");
    let gpu_cfv = debug_kernel::gpu_brute_force_with_rake(
        &ctx, nh, np, traverser, starting_pot, fold_mask,
        &opp_reach, &contributions, &hand_cards, &pl_str, &pl_idx,
        rake_rate, rake_cap, flop_seen,
    );

    eprintln!("HU residual diagnostic: fold-after-bet (unequal contribs)");
    eprintln!("  Setup: np=2, contribs=[15,5], fold_mask=0b10, traverser=0 (lone survivor)");
    eprintln!("  CPU (fast path, rake on total_pot=30): {:?}", cpu_cfv);
    eprintln!("  Metal (K=1 per-level, main_pot_rake): {:?}", gpu_cfv);

    let mut max_diff = 0.0_f32;
    for h in 0..nh {
        let d = (cpu_cfv[h] - gpu_cfv[h]).abs();
        if d > max_diff { max_diff = d; }
    }
    eprintln!("  max_diff = {} (post-fix: should be ≤ f32 floor)", max_diff);

    // POST-FIX assertion: CPU and Metal now both use main_pot_only rake.
    // The diff at this scenario should be at f32 floor (effectively 0).
    // If it's not, either CPU or Metal regressed to total-pot rake.
    assert!(
        max_diff < 1e-4,
        "Regression: HU fold-win-after-bet shows max_diff = {} > 1e-4. \
         Expected: CPU and Metal both apply main_pot_only rake (uncalled \
         bets returned un-raked per the rake spec). If CPU returns ≈17.0 \
         and Metal returns ≈18.0, CPU regressed to total_pot rake (the \
         buggy version from before 2026-06-04). If both return ≈17.0, \
         both regressed.",
        max_diff,
    );
    eprintln!("✓ HU fold-win-after-bet: CPU and Metal agree on main_pot_only rake \
        (uncalled bet returned un-raked).");
}

// ═════════════════════════════════════════════════════════════════════
// Site (b) PRIMARY ISOLATION TEST — direct CPU↔Metal kernel unit check
//
// Per user direction after Phase A.6: "The divergence-assertion fallback
// catches the case where site b is wrong and site a is right, but it
// does not catch the case where site a and site b are wrong in the same
// way (both have the same rake-formula error, say the cap applied wrong
// in a way common to both K=2 paths), because then they converge
// together, their ratio stays ~1, and the divergence assertion never
// fires while both are wrong. So the trade isn't 'perfect vs
// good-enough,' it's 'catches the less-likely failure, misses the
// more-likely one.' Make site (b) measure site (b) against truth, then
// Phase B proceeds with all five sites genuinely covered."
//
// This test directly invokes the Metal `multiway_brute_force_showdown`
// helper via the `debug_brute_force_showdown` kernel (existing
// infrastructure, used by gpu_brute_force_unit.rs) with inputs that
// route the kernel EXCLUSIVELY through site (b)'s lone-survivor branch:
//
//   - np = 3 (num_opp = 2 → K=2 paths)
//   - fold_mask = 0b110 (players 1, 2 folded)
//   - traverser = 0 (the lone survivor; NOT folded)
//   - → kernel checks num_opp==2 && (num_active<=1) → site (b) fast path
//
// CPU reference: `side_pot_showdown_cfv_with_rake` with the same inputs,
// rake_rate=0.05, rake_cap=1.0, flop_seen=true. Returns rake-applied
// per-hand CFV.
//
// Today (pre-Phase B): Metal's debug kernel calls the helper WITHOUT
// rake params (the helper signature hasn't been extended yet), so
// Metal's output is rake-free. Diff = rake amount per surviving hand.
// Test FAILS today.
//
// After Phase B closes site (b) AND extends DebugBruteForceParams +
// the debug kernel to forward rake params to the helper: Metal's site
// (b) computes the same rake math as CPU. Diff → f32 floor. Test PASSES.
//
// THIS TEST IS THE PRIMARY DEFENSE because:
//   - It validates site (b)'s arithmetic against CPU GROUND TRUTH, not
//     against site (a)'s convergence. Correlated failure (both K=2
//     paths sharing a wrong eff_rake helper) is caught: even if site
//     (a)'s gate also fails the same way, this test independently
//     catches site (b) being wrong against the CPU rake reference.
//   - It directly invokes site (b)'s code path via fold_mask routing,
//     so the measured diff IS site (b)'s error (zero contribution
//     from any other kernel branch).
//
// The cap-binding scenario (site_b_3p_fold_terminal_rake) and the
// divergence-assertion test (site_ab_divergence_check_post_phase_b)
// remain as SECOND and THIRD lines of defense respectively. But this
// kernel unit test is the primary site-(b)-against-truth validation
// the user asked for.
// ═════════════════════════════════════════════════════════════════════

// Local copy of the debug-kernel calling pattern from
// gpu_brute_force_unit.rs (cross-test-file imports aren't supported
// in Rust integration tests; minor duplication is the path).
mod debug_kernel {
    use metal::MTLSize;
    use solver_core::gpu_metal::buffer::MetalBuffer;
    use solver_core::gpu_metal::context::MetalContext;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct DebugBruteForceParams {
        nh: i32,
        np: i32,
        traverser: i32,
        starting_pot: i32,
        fold_mask: u16,
        _pad: u16,
        rake_rate: f32,
        rake_cap: f32,
        flop_seen: i32,
    }

    /// Invoke Metal's `multiway_brute_force_showdown` via the debug
    /// kernel, with explicit rake params for Slice 2 site isolation
    /// tests. Pass `rake_rate=0.0, rake_cap=0.0, flop_seen=false` for
    /// rake-free invocation (matches pre-Slice-2 behavior).
    pub fn gpu_brute_force_with_rake(
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
        rake_rate: f32,
        rake_cap: f32,
        flop_seen: bool,
    ) -> Vec<f32> {
        let device = ctx.device();
        let pipeline = ctx.create_pipeline("debug_brute_force_showdown").expect("pipeline");

        let d_output = MetalBuffer::<f32>::zeros(device, nh);
        let d_opp_reach = MetalBuffer::from_slice(device, opp_reach);
        let d_contributions = MetalBuffer::from_slice(device, contributions);
        let d_hand_cards = MetalBuffer::from_slice(device, hand_cards);
        let d_pl_str = MetalBuffer::from_slice(device, pl_str);
        let d_pl_idx = MetalBuffer::from_slice(device, pl_idx);

        let params = DebugBruteForceParams {
            nh: nh as i32,
            np: np as i32,
            traverser: traverser as i32,
            starting_pot,
            fold_mask,
            _pad: 0,
            rake_rate,
            rake_cap,
            flop_seen: if flop_seen { 1 } else { 0 },
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

    pub fn make_sorted(strengths: &[u16]) -> (Vec<u16>, Vec<u16>) {
        let nh = strengths.len();
        let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (strengths[h], h as u16)).collect();
        items.sort_by_key(|&(s, _)| s);
        let mut s_str = vec![0u16; nh];
        let mut s_idx = vec![0u16; nh];
        for i in 0..nh {
            s_str[i] = items[i].0;
            s_idx[i] = items[i].1;
        }
        (s_str, s_idx)
    }
}

#[test]
#[ignore = "Slice 2 Phase B Site (b) ISOLATION (primary): direct CPU↔Metal \
            kernel-level check of site (b)'s lone-survivor rake math. \
            Catches BOTH asymmetric (site b wrong, a right) AND correlated \
            (both K=2 sites wrong via shared helper) failures because it \
            validates site (b)'s output against CPU GROUND TRUTH at the \
            kernel-helper level, not against another site's convergence. \
            Enable when Phase B (i) closes site (b)'s rake math in \
            multiway_brute_force_showdown AND (ii) extends \
            DebugBruteForceParams + debug_brute_force_showdown kernel to \
            forward rake_rate/rake_cap/flop_seen to the helper. Run: \
            cargo test --release --features metal --test \
            gpu_rake_parity_gate site_b_isolated -- --ignored"]
fn site_b_isolated_kernel_unit_test() {
    use solver_core::solver::showdown::side_pot_showdown_cfv_with_rake;

    // 3p (num_opp=2) — K=2 paths.
    let nh = 3;
    let np = 3;
    let num_opp = 2;
    let hand_cards: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
    let strengths: Vec<u16> = vec![10, 20, 30];
    let (pl_str, pl_idx) = debug_kernel::make_sorted(&strengths);

    // Reach: uniform 1.0 for both opponents. Site (b) lone-survivor
    // formula is independent of opp_reach distribution (the survivor
    // wins regardless of opp hands), but we need a nonzero value so
    // the kernel's reach-weighted accumulator produces a measurable
    // CFV. With uniform reach=1.0, the per-hand CFV =
    //   (num_opp_hand_combos_compatible_with_h) * payoff(h)
    // and payoff(h) is constant per h (= total_pot - rake -
    // traverser_stake for the surviving traverser), so CFV scales by
    // the compatible-pair count.
    let opp_reach: Vec<f32> = vec![1.0; num_opp * nh];

    // Equal contributions; total_pot = starting_pot + sum(contribs) = 30.
    let contributions: Vec<i32> = vec![5, 5, 5];
    let starting_pot: i32 = 15;

    // fold_mask = 0b110: players 1 and 2 folded. traverser=0 is the
    // sole active player. The kernel's check
    //   `num_opp == 2 && (num_active <= 1 || traverser_folded)`
    // triggers site (b)'s fast path. num_active = 1 ≤ 1 → enters
    // this branch with traverser as the lone survivor.
    let fold_mask: u16 = 0b110;
    let traverser = 0;

    // Rake params: rake_rate=0.05 (5%), rake_cap=1.0 (binds at this
    // pot since total_pot * 0.05 = 1.5 > 1.0). flop_seen=true (current
    // flop-onward tree, no preflop terminals).
    let rake_rate = 0.05_f32;
    let rake_cap = 1.0_f32;
    let flop_seen = true;

    // CPU reference: side_pot_showdown_cfv_with_rake. This is the
    // proven-against-hand-computation reference; per Slice 1.x it
    // has been anchored at every path including the lone-survivor
    // fast path (showdown.rs ~484-548).
    let opp_reach_per_opp: Vec<Vec<f32>> = (0..num_opp)
        .map(|oi| opp_reach[oi * nh..(oi + 1) * nh].to_vec())
        .collect();
    let opp_reach_views: Vec<&[f32]> = opp_reach_per_opp.iter().map(|v| v.as_slice()).collect();
    let mut sorted_opp_str = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx = Vec::with_capacity(num_opp * nh);
    for _ in 0..num_opp {
        sorted_opp_str.extend_from_slice(&pl_str);
        sorted_opp_idx.extend_from_slice(&pl_idx);
    }
    let cpu_cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &pl_str, &pl_idx,
        &contributions, fold_mask, traverser, np as u8, starting_pot,
        rake_rate, rake_cap, flop_seen,
    );

    // Metal helper: invoke via debug kernel. TODAY (pre-Phase B):
    // the debug kernel does not pass rake params to the helper, so
    // Metal computes the rake-FREE site (b) CFV. After Phase B
    // extends DebugBruteForceParams + the debug kernel to forward
    // rake params, Metal will apply the same rake math as CPU.
    let ctx = MetalContext::new().expect("Metal context");
    let gpu_cfv = debug_kernel::gpu_brute_force_with_rake(
        &ctx, nh, np, traverser, starting_pot, fold_mask,
        &opp_reach, &contributions, &hand_cards, &pl_str, &pl_idx,
        rake_rate, rake_cap, flop_seen,
    );

    eprintln!("Site (b) isolated kernel unit test:");
    eprintln!("  CPU (with rake): {:?}", cpu_cfv);
    eprintln!("  Metal (today, no rake): {:?}", gpu_cfv);

    let mut max_diff = 0.0_f32;
    let mut max_h = 0;
    for h in 0..nh {
        let d = (cpu_cfv[h] - gpu_cfv[h]).abs();
        if d > max_diff { max_diff = d; max_h = h; }
    }
    eprintln!("  max_diff = {} at h={} (today: ≈ rake amount per surviving hand)",
        max_diff, max_h);

    // After Phase B (kernel + debug-params updated to forward rake):
    // diff at f32 floor. CPU reference for this scenario is hand-
    // anchored and small (3-hand uniform), so f64 confirmation of the
    // CPU reference is trivial (the formula is `(total_pot * rate)
    // .min(cap)` = min(30 * 0.05, 1.0) = 1.0 per surviving hand
    // exactly, no accumulation drift).
    assert!(max_diff < 1e-4,
        "Site (b) isolated unit test FAILED: max_diff = {} at h={}. \
         CPU computes site (b) lone-survivor with rake; Metal computes \
         it without. Phase B must (i) add rake math at the lone-survivor \
         branch in multiway_brute_force_showdown (vcfr.metal line ~243) \
         AND (ii) extend DebugBruteForceParams + debug kernel to forward \
         rake_rate/rake_cap/flop_seen. Until BOTH land, this test fails. \
         This test is the PRIMARY site (b) defense: it catches correlated \
         K=2 failures the divergence assertion misses, because it \
         compares against CPU GROUND TRUTH not against site (a).",
        max_diff, max_h);
    eprintln!("✓ Site (b) isolated unit test PASSED: Metal site (b) matches CPU rake reference");
}

// ═════════════════════════════════════════════════════════════════════
// Site (a) PRIMARY ISOLATION TEST — direct CPU↔Metal kernel unit check
//
// Per user direction after Phase B Site (b): "Closing (b) didn't move
// other gates" proves (b) wasn't contaminating them, but does NOT prove
// (a) and (c) are self-isolated, because (a) and (c) are both still
// open and could be contaminating each other's gates. The gate-level
// signal was proven unreliable for the K=2 cluster (b's gate didn't
// measure b), so (a) and (c) get the same debug-kernel unit test that
// worked for (b) — not gate-reliance.
//
// Routing to site (a)'s K=2 all-equal brute (vcfr.metal line ~187):
//   - np = 3 (num_opp = 2 → K=2 paths)
//   - fold_mask = 0 (no folds → fails (b)'s num_active≤1 check)
//   - contributions all equal → passes (a)'s all_active_equal check
//   - → kernel routes to site (a)'s branch exclusively
//
// CPU reference: side_pot_showdown_cfv_with_rake. With all-equal
// contribs and no folds, CPU takes the K≥2 equal-contributions path
// (showdown.rs ~699-779) which uses rake_per_unit_stake:
//   rake = min(total_pot * eff_rake_rate, eff_rake_cap)
//   rake_per_unit_stake = rake / half_pot
//   payoff_unit = K - rake_per_unit_stake [for wins]
//                = (K+1-T)/T - rake_per_unit_stake/T [for ties]
// ═════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "Slice 2 Phase B Site (a) ISOLATION (primary): direct CPU↔Metal \
            kernel-level check of site (a)'s K=2 all-equal brute rake math. \
            Catches asymmetric AND correlated K=2-cluster failures (sibling \
            of site b's unit test, same pattern). Run: cargo test --release \
            --features metal --test gpu_rake_parity_gate site_a_isolated \
            -- --ignored"]
fn site_a_isolated_kernel_unit_test() {
    use solver_core::solver::showdown::side_pot_showdown_cfv_with_rake;

    let nh = 3;
    let np = 3;
    let num_opp = 2;
    let hand_cards: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
    let strengths: Vec<u16> = vec![10, 20, 30];
    let (pl_str, pl_idx) = debug_kernel::make_sorted(&strengths);

    let opp_reach: Vec<f32> = vec![1.0; num_opp * nh];
    let contributions: Vec<i32> = vec![5, 5, 5];  // all equal → site (a)
    let starting_pot: i32 = 15;
    let fold_mask: u16 = 0;  // no folds → site (a)
    let traverser = 0;

    let rake_rate = 0.05_f32;
    let rake_cap = 1.0_f32;
    let flop_seen = true;

    let opp_reach_per_opp: Vec<Vec<f32>> = (0..num_opp)
        .map(|oi| opp_reach[oi * nh..(oi + 1) * nh].to_vec())
        .collect();
    let opp_reach_views: Vec<&[f32]> = opp_reach_per_opp.iter().map(|v| v.as_slice()).collect();
    let mut sorted_opp_str = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx = Vec::with_capacity(num_opp * nh);
    for _ in 0..num_opp {
        sorted_opp_str.extend_from_slice(&pl_str);
        sorted_opp_idx.extend_from_slice(&pl_idx);
    }
    let cpu_cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &pl_str, &pl_idx,
        &contributions, fold_mask, traverser, np as u8, starting_pot,
        rake_rate, rake_cap, flop_seen,
    );

    let ctx = MetalContext::new().expect("Metal context");
    let gpu_cfv = debug_kernel::gpu_brute_force_with_rake(
        &ctx, nh, np, traverser, starting_pot, fold_mask,
        &opp_reach, &contributions, &hand_cards, &pl_str, &pl_idx,
        rake_rate, rake_cap, flop_seen,
    );

    eprintln!("Site (a) isolated kernel unit test:");
    eprintln!("  CPU (with rake): {:?}", cpu_cfv);
    eprintln!("  Metal (with rake? — today: NO, pre-Phase-B-site-a): {:?}", gpu_cfv);

    let mut max_diff = 0.0_f32;
    let mut max_h = 0;
    for h in 0..nh {
        let d = (cpu_cfv[h] - gpu_cfv[h]).abs();
        if d > max_diff { max_diff = d; max_h = h; }
    }
    eprintln!("  max_diff = {} at h={} (predicted today: ≈ rake_per_unit_stake × reach)",
        max_diff, max_h);

    assert!(max_diff < 1e-4,
        "Site (a) isolated unit test FAILED: max_diff = {} at h={}. \
         CPU computes site (a) K=2 all-equal brute with rake; Metal \
         computes it without. Phase B must add rake math at vcfr.metal \
         line ~187 mirroring CPU showdown.rs lines 699-779: \
         rake_per_unit_stake = rake / half_pot; payoff_unit adjustments \
         for win/tie cases.",
        max_diff, max_h);
    eprintln!("✓ Site (a) isolated unit test PASSED");
}

// ═════════════════════════════════════════════════════════════════════
// Site (c) PRIMARY ISOLATION TEST — direct CPU↔Metal kernel unit check
//
// Per user direction (same Phase A.7 lesson applied to site (c)):
// (c) is the K=2 path most sensitive to rake-formula errors because
// the main-pot-only and cap-once rules interact with per-level cash
// accumulation. A per-level-rake or per-level-cap error would be
// invisible at the gate level (correlated K=2 errors) but caught
// immediately by a direct CPU-truth comparison at the (c) branch.
//
// Routing to site (c)'s K=2 general path (vcfr.metal line ~281):
//   - np = 3 (num_opp = 2)
//   - contributions UNEQUAL (e.g., [10, 5, 5]) → fails (a)'s
//     all_active_equal check
//   - fold_mask = 0 (no folds) → fails (b)'s num_active≤1 check
//   - → kernel falls through to (c)'s per-level brute-force loop
//
// CPU reference: side_pot_showdown_cfv_with_rake at the unequal-
// contributions path (showdown.rs ~870-910 the main-pot-only Slice 1.5
// implementation). Site (c) is the trickiest mirror because of the
// per-level cash logic + main-pot-only-rake spec.
// ═════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "Slice 2 Phase B Site (c) ISOLATION (primary): direct CPU↔Metal \
            kernel-level check of site (c)'s K=2 unequal-contributions / \
            side-pot rake math. The main-pot-only and cap-once rules make \
            this the most rake-formula-sensitive site; isolated unit test \
            catches per-level-rake or per-level-cap errors that would be \
            invisible at the gate level. Run: cargo test --release \
            --features metal --test gpu_rake_parity_gate site_c_isolated \
            -- --ignored"]
fn site_c_isolated_kernel_unit_test() {
    use solver_core::solver::showdown::side_pot_showdown_cfv_with_rake;

    let nh = 3;
    let np = 3;
    let num_opp = 2;
    let hand_cards: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
    let strengths: Vec<u16> = vec![10, 20, 30];
    let (pl_str, pl_idx) = debug_kernel::make_sorted(&strengths);

    let opp_reach: Vec<f32> = vec![1.0; num_opp * nh];

    // Unequal contributions to route to (c)'s side-pot branch. Player
    // 0 has 10, others have 5. This creates a 2-level pot structure:
    //   - Level 1 (li=0, main pot): all 3 contributed 5 each → 15 + starting_pot
    //   - Level 2 (li=1, side pot): only player 0 contributed extra 5 → 5
    // Per the rake spec (Slice 1.5): rake applies to MAIN POT ONLY,
    // side-pot level un-raked, cap applied ONCE.
    let contributions: Vec<i32> = vec![10, 5, 5];
    let starting_pot: i32 = 15;
    let fold_mask: u16 = 0;
    let traverser = 0;

    let rake_rate = 0.05_f32;
    let rake_cap = 1.0_f32;
    let flop_seen = true;

    let opp_reach_per_opp: Vec<Vec<f32>> = (0..num_opp)
        .map(|oi| opp_reach[oi * nh..(oi + 1) * nh].to_vec())
        .collect();
    let opp_reach_views: Vec<&[f32]> = opp_reach_per_opp.iter().map(|v| v.as_slice()).collect();
    let mut sorted_opp_str = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx = Vec::with_capacity(num_opp * nh);
    for _ in 0..num_opp {
        sorted_opp_str.extend_from_slice(&pl_str);
        sorted_opp_idx.extend_from_slice(&pl_idx);
    }
    let cpu_cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &sorted_opp_str, &sorted_opp_idx,
        &pl_str, &pl_idx,
        &contributions, fold_mask, traverser, np as u8, starting_pot,
        rake_rate, rake_cap, flop_seen,
    );

    let ctx = MetalContext::new().expect("Metal context");
    let gpu_cfv = debug_kernel::gpu_brute_force_with_rake(
        &ctx, nh, np, traverser, starting_pot, fold_mask,
        &opp_reach, &contributions, &hand_cards, &pl_str, &pl_idx,
        rake_rate, rake_cap, flop_seen,
    );

    eprintln!("Site (c) isolated kernel unit test:");
    eprintln!("  CPU (with rake, main-pot-only): {:?}", cpu_cfv);
    eprintln!("  Metal (today, no rake): {:?}", gpu_cfv);

    let mut max_diff = 0.0_f32;
    let mut max_h = 0;
    for h in 0..nh {
        let d = (cpu_cfv[h] - gpu_cfv[h]).abs();
        if d > max_diff { max_diff = d; max_h = h; }
    }
    eprintln!("  max_diff = {} at h={} (today: ≈ main-pot rake × reach)",
        max_diff, max_h);

    assert!(max_diff < 1e-4,
        "Site (c) isolated unit test FAILED: max_diff = {} at h={}. \
         CPU computes site (c) K=2 unequal-contributions with main-pot-\
         only rake; Metal computes it without. Phase B must add rake \
         math at vcfr.metal line ~281 mirroring CPU showdown.rs \
         ~870-910: rake applies ONLY at li==0 (main pot), side-pot \
         levels (li≥1) un-raked, cap applied ONCE per hand. The most \
         rake-formula-sensitive site; isolated test catches per-level \
         errors invisible at gate level.",
        max_diff, max_h);
    eprintln!("✓ Site (c) isolated unit test PASSED");
}

// ═════════════════════════════════════════════════════════════════════
// Site (b) SECONDARY DEFENSE — divergence assertion
//
// (Kept as a secondary line of defense. After the primary isolation
// test passes, the divergence assertion catches the remaining failure
// mode: site (a) closed, site (b) closed but with a math bug that
// happens to converge to f32 floor in the isolation test but diverges
// from site (a) in the gate scenario. Unlikely, but cheap to keep.)
// ═════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "Slice 2 Phase B DISCRIMINATOR: site (a)/(b) divergence check. \
            Enable when Phase B closes both K=2 sites. Fails if site (b)'s \
            diff is meaningfully larger than site (a)'s (which would mean \
            site (b)'s kernel branch was left wrong while site (a)'s went \
            green via the joint eff_rake math). Per-site gates fire first \
            if either fails outright; this gate catches asymmetric \
            convergence between them."]
fn site_ab_divergence_check_post_phase_b() {
    // Run both scenarios; both must be converged to f32 floor.
    let (tree_a, table_a) = build_3player_with_rake(0.05, 1000.0);
    let (diff_a, _, _) = run_parity(&tree_a, table_a, "[discriminator] site (a)");

    let (tree_b, table_b) = build_3player_with_rake(0.05, 1.0);
    let (diff_b, _, _) = run_parity(&tree_b, table_b, "[discriminator] site (b)");

    // Per-site convergence check (each must individually be at f32 floor).
    let f32_floor_tol = 1e-4_f32;
    assert!(diff_a < f32_floor_tol,
        "Site (a) not converged: diff_a = {} >= {}. Per-site gate should fire first.",
        diff_a, f32_floor_tol);
    assert!(diff_b < f32_floor_tol,
        "Site (b) not converged: diff_b = {} >= {}. Per-site gate should fire first.",
        diff_b, f32_floor_tol);

    // Divergence check: after both pass, they should both be at the SAME
    // floor (the f32 accumulation precision). A meaningful asymmetry
    // would indicate one site's math is fundamentally different from
    // the other's — which it shouldn't be, since both branches share the
    // eff_rake (rate, cap) computation at function entry.
    //
    // If diff_a converged to ~1e-6 and diff_b stays at ~1e-2, that's
    // a clear sign site (b)'s kernel branch has a bug that site (a)'s
    // branch doesn't (e.g., wrong rake formula in the fold-win path).
    //
    // 10x is a generous discrimination threshold; tighter could be set
    // post-Phase-B once empirical floor magnitudes are known.
    let divergence_ratio = if diff_a > 0.0 { diff_b / diff_a } else { diff_b };
    let divergence_threshold = 10.0_f32;
    assert!(
        divergence_ratio < divergence_threshold,
        "Site (a)/(b) DIVERGENCE: diff_b / diff_a = {} > {}x. \
         Both scenarios passed per-site f32-floor checks, but site (b)'s \
         diff is {}x larger than site (a)'s. This is the discriminator \
         the user warned about: site (a)'s K=2 all-equal brute closed, \
         but site (b)'s K=2 fold-win fast path was left wrong (or vice \
         versa). Inspect both kernel branches — they should share the \
         eff_rake math at function entry. \
         diff_a={}, diff_b={}",
        divergence_ratio, divergence_threshold,
        divergence_ratio, diff_a, diff_b,
    );
    eprintln!("✓ Site (a)/(b) divergence check OK: ratio={} (both converged \
        to comparable f32 floor)", divergence_ratio);
}

// ═════════════════════════════════════════════════════════════════════
// Legacy entry point (pre-expansion). Kept for git-blame continuity with
// the original Slice 2 gate commit (f13e2fd); now redundant with
// site_a_3p_equal_showdown_rake.
// ═════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "Superseded by site_a_3p_equal_showdown_rake (same scenario). \
            Kept under its original name for git-blame continuity with the \
            initial Slice 2 gate commit (f13e2fd)."]
fn cpu_metal_parity_at_rake_5pct() {
    let (tree, table) = build_3player_with_rake(0.05, 1000.0);
    let (max_diff, zone, argmax) = run_parity(&tree, table, "[legacy] 3p rake=5%");
    assert!(max_diff < 1e-4, "FAILED: max_diff = {} (zone={}, idx={})",
        max_diff, zone, argmax);
}
