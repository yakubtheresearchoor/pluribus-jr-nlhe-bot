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
//! GPU site: vcfr.metal. This gate is the bar.
//!
//! ## States
//!
//! BEFORE Slice 2 lands (current state):
//!   - CPU applies rake via side_pot_showdown_cfv_with_rake threaded
//!     through evaluate_terminal (Slice 1.6, commit 62cc5bc)
//!   - Metal does NOT apply rake (vcfr.metal showdown sites unchanged
//!     from rake=0 formula)
//!   - rake≠0 test should FAIL (CPU CFV diverges from Metal by the rake
//!     amount at terminals; the diff cascades through CFR iterations)
//!   - Marked #[ignore] so CI stays green; runnable manually as the
//!     diagnostic that confirms the gap exists and is the gate
//!
//! AFTER Slice 2 lands:
//!   - Both CPU and Metal apply rake (matching the rake spec: main
//!     pot only, single cap per hand, no-flop-no-drop)
//!   - rake≠0 test passes at f32 floor
//!   - Remove #[ignore] to make it a permanent regression gate
//!
//! ## Test infrastructure
//!
//! Uses the manual-table-construction pattern from three_max_parity.rs
//! with 6 chosen hands. Full FlopChanceTable::compute_flop_start (full
//! 1326-hand deck) at production nh caused the Metal solver init to
//! hang in an earlier draft (likely buffer allocation / kernel compile
//! overhead). Manual table with chosen hands keeps setup fast.
//!
//! Setup is 3-player (mirroring three_max_parity's known-good pattern)
//! with equal contributions, all active, no folds. Rake at this terminal
//! type exercises the brute-force per-(h, g_a, g_b) path (Slice 1.4's
//! anchor). The CPU-side anchor proves rake is correctly applied at
//! this path; the Metal-side test confirms it matches.
//!
//! ## f64 discriminator
//!
//! Per user direction: "When the gate shows a small diff as Metal rake
//! converges, do not accept it as f32 floor from the magnitude, confirm
//! it with the f64 discriminator the same way the P2.5a diff was
//! confirmed, because a Metal rake bug (wrong buffer slot for the rake
//! value, cap applied per-level instead of once per hand, rake attached
//! to the wrong eligibility level, no-flop-no-drop condition mishandled)
//! could produce a small diff that looks like precision, and the kernel
//! is the worst place for a silent rake error."
//!
//! When the gate produces a small diff that LOOKS like f32 floor
//! (e.g., ~1e-5 relative), the discipline is to NOT accept the magnitude
//! argument. The path forward — applied when Slice 2 lands — is to
//! match the kernel computation against a CPU f64-mirror of the same
//! shape and confirm collapse to f64 floor (the P2.5a precision
//! demonstrator pattern). For now the gate's pass/fail is at f32 floor
//! with a TODO note for the f64 discriminator when needed.

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

fn find_pair_index(c1: Card, c2: Card) -> u16 {
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (a, b) = index_to_card_pair(idx);
        if (a == c1 as u8 && b == c2 as u8) || (a == c2 as u8 && b == c1 as u8) {
            return idx as u16;
        }
    }
    panic!("pair not found");
}

/// Build a 3-player chance table + tree with parameterized rake. The 6
/// chosen hands are pairwise non-conflicting (no shared cards across
/// any pair, no shared cards with the board). Mirrors
/// three_max_parity's build_3player_table exactly except for the rake
/// values in TreeConfig.
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

/// Compare CPU per-zone regrets (flop/turn/river) against GPU's
/// concatenated regret buffer. GPU layout: [flop | turn | river].
/// Returns (max_abs_diff_overall, (zone_label, idx_in_zone)).
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

    for (zone_idx, (cpu_slice, gpu_start, label)) in [
        (cpu_flop,  0,        "flop"),
        (cpu_turn,  fl,       "turn"),
        (cpu_river, fl + tl,  "river"),
    ].iter().enumerate() {
        let _ = zone_idx;
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

#[test]
fn cpu_metal_parity_at_rake_0_baseline() {
    // SANITY CHECK: at rake=0, CPU↔Metal MUST agree. This is what every
    // existing parity gate validates and is the foundation the rake≠0
    // gate sits on. If this fails, infrastructure is broken — not rake.
    let (tree, table) = build_3player_with_rake(0.0, 0.0);
    let game = FlopStartGame::new(table);

    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    let gpu_reg = gpu.download_regrets();
    let (max_diff, zone, argmax) = compare_regrets_per_zone(&cpu, &gpu_reg);

    eprintln!("rake=0 baseline: CPU↔Metal max regret diff = {} (idx {})", max_diff, argmax);
    eprintln!("  worst-diff zone: {}, idx-within-zone: {}", zone, argmax);

    // At rake=0, existing parity is established; f32 floor for this small
    // setup is well under 1e-4.
    let f32_floor_tol = 1e-4_f32;
    assert!(
        max_diff < f32_floor_tol,
        "rake=0 baseline FAILED with max_diff = {} > {}. CPU↔Metal \
         infrastructure broken — not a rake issue.",
        max_diff, f32_floor_tol,
    );
    eprintln!("✓ rake=0 baseline OK ({}); test infrastructure works", max_diff);
}

#[test]
#[ignore = "Slice 2 gate: CPU↔Metal parity at rake≠0. FAILS today \
            because vcfr.metal does not apply rake. Enable (remove \
            #[ignore]) when Metal rake lands — at that point this becomes \
            the permanent regression gate. The CPU side is the proven \
            reference (Slice 1.x rake math hand-anchored, threaded \
            through evaluate_terminal at Slice 1.6); a rake≠0 parity \
            failure here is unambiguously the Metal side. Run manually: \
            cargo test --release --features metal --test \
            gpu_rake_parity_gate -- --ignored"]
fn cpu_metal_parity_at_rake_5pct() {
    let (tree, table) = build_3player_with_rake(0.05, 1000.0);
    let game = FlopStartGame::new(table);

    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    eprintln!("\n=== Slice 2 gate: CPU↔Metal parity at rake_rate=0.05, rake_cap=1000 ===");
    eprintln!("Tree: {} nodes; per-flop nh: {}", tree.num_nodes(), game.table().num_valid);

    cpu.run(&tree, &game, 1);
    gpu.run(&ctx, &tree, &game, 1);

    let gpu_reg = gpu.download_regrets();
    let (max_diff, zone, argmax) = compare_regrets_per_zone(&cpu, &gpu_reg);

    eprintln!("CPU↔Metal max regret diff = {} (idx {})", max_diff, argmax);
    eprintln!("  worst-diff zone: {}, idx-within-zone: {}", zone, argmax);

    // f32 floor tolerance for iter-1 regrets on this small setup. Below
    // 1e-4 is comfortably within f32 floor for the few-hundred-add
    // accumulation at this scale.
    let f32_floor_tol = 1e-4_f32;
    assert!(
        max_diff < f32_floor_tol,
        "CPU↔Metal parity at rake_rate=0.05 FAILED: max_diff = {} > {}. \
         CPU applies rake via Slice 1.6 evaluate_terminal threading; \
         if Metal hasn't been updated to apply rake (the Slice 2 work), \
         regrets diverge by the rake amount at terminals and the diff \
         cascades through CFR iterations. \
         \
         TODO when Slice 2 lands: if max_diff is small but non-zero \
         (looks like f32 floor), apply the f64 discriminator per user \
         direction — kernel rake bugs can masquerade as f32 noise. \
         Specifically: confirm the diff collapses to f64 floor when both \
         sides use f64 accumulators (P2.5a precision-demonstrator \
         pattern). The kernel is the worst place for a silent rake error.",
        max_diff, f32_floor_tol,
    );
    eprintln!("✓ Slice 2 gate PASSED at f32 floor — Metal rake matches CPU reference");
}
