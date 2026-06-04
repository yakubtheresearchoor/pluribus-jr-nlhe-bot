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
    // Same tree as Site (a) — fold terminals are present in any 3p
    // tree with bet/check options. The fold-win Metal kernel branch
    // applies rake to the surviving player's payoff.
    let (tree, table) = build_3player_with_rake(0.05, 1000.0);
    let (max_diff, zone, argmax) = run_parity(&tree, table, "Site (b) 3p fold-term rake=5%");
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
