// Phase 4 REDO — P0 config probe.
//
// PRECONDITION GATE: The Phase 4 redo requires a config where the RICH action
// set produces NONZERO rich exploitability at our iter budget. Every config
// tried in v1-v4 hit 0% pot (the f32 floor) by iter 200, which made the
// lean-vs-rich comparison vacuous — both reached the floor and the comparison
// couldn't distinguish "lean is fine" from "lean is bad but we can't see it."
//
// This probe sweeps candidate configs along the three knobs most likely to
// produce nonzero exploitability:
//   1. STACK DEPTH — deeper stacks mean more bet-sizing decisions at depth
//      (overbets and small bets become strategically distinct, instead of
//      converging because the all-in cap is right there).
//   2. BOARD TEXTURE — wet boards (straight + flush draws) make bet sizing
//      genuinely matter (overbets to charge draws, blocks to deny equity).
//      A dry KING-high rainbow lets sizing collapse to a single optimal.
//   3. ITER COUNT — at iter 25 of M3 (nh=12 6p) we measured 0.087% pot, and
//      at iter 50 it was 0.005%. So shorter iter budgets keep CFR mid-
//      trajectory; the lean-vs-rich gap is most visible BEFORE the floor.
//
// The acceptance criterion for moving to P1 is:
//   rich exploitability > 0.1% pot at the chosen (config, iter_count)
//
// 0.1% is two production targets above the f32 floor and the tight (0.05%)
// production target; this gap is what the lean cost has to fit inside.
//
// Each candidate is evaluated on a SHORT iter budget (≤100) on nh=6 6p so
// the probe is fast — full Phase 4 measurement at the chosen config will
// be its own test in P3.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::time::Instant;

struct Candidate {
    name: &'static str,
    board: &'static [&'static str; 3],
    stacks: i32,
    turn_cards: &'static [&'static str],
    river_cards_per_turn: &'static [&'static [&'static str]],
}

fn build_table_for(c: &Candidate, nh: usize, np: u8) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = c.board.iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..np as usize {
        for &hi in &chosen {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
            ranges[p][pair_idx] = 1.0;
        }
    }
    let turn_cards: Vec<u8> = c.turn_cards.iter()
        .map(|s| card_from_str(s).unwrap() as u8).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        let rcs: Vec<u8> = c.river_cards_per_turn[ti].iter()
            .map(|s| card_from_str(s).unwrap() as u8).collect();
        river_decks[tc as usize] = rcs;
    }
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, np, &chosen, &turn_cards, &river_decks,
    );
    // Rich action set — Pluribus-flavored postflop blueprint.
    let config = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![c.stacks; np as usize],
        initial_contributions: vec![5; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![
                BetSize::PotRelative(0.33),
                BetSize::PotRelative(0.66),
                BetSize::PotRelative(1.0),
                BetSize::PotRelative(1.5),
            ],
            raise: vec![
                BetSize::PotRelative(1.0),
                BetSize::PotRelative(2.0),
            ],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
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

#[test]
#[ignore = "Phase 4 P0 config probe — finds a config with nonzero rich exploitability (~10-20 min per candidate)"]
fn phase4_p0_config_probe() {
    let nh = 6usize;
    let np = 6u8;
    let pot = 30.0f32;
    let max_iters = 100u32;
    let checkpoints: Vec<u32> = vec![5, 10, 25, 50, 100];

    // Probe candidates. Start with the MOST PROMISING (deep + wet) first;
    // if that hits the >0.1% gate, the rest are unnecessary.
    let candidates: Vec<Candidate> = vec![
        Candidate {
            name: "deep_wet_th9d8c_stacks500",
            board: &["Th", "9d", "8c"],
            stacks: 500,
            turn_cards: &["2c", "Jd"],
            river_cards_per_turn: &[
                &["4s", "7h"],  // after 2c
                &["3s", "Qc"],  // after Jd (completes straight, useful texture)
            ],
        },
        // Fallback candidates if deep+wet hits the floor too fast:
        Candidate {
            name: "deepest_wet_th9d8c_stacks1000",
            board: &["Th", "9d", "8c"],
            stacks: 1000,
            turn_cards: &["2c", "Jd"],
            river_cards_per_turn: &[
                &["4s", "7h"],
                &["3s", "Qc"],
            ],
        },
        Candidate {
            name: "deep_wet_jh10d9c_stacks500",
            board: &["Jh", "Td", "9c"],
            stacks: 500,
            turn_cards: &["2c", "Qd"],
            river_cards_per_turn: &[
                &["4s", "7h"],
                &["8s", "Kc"],  // multiple straight completions
            ],
        },
    ];

    eprintln!("\n=== Phase 4 P0 — Config probe for nonzero rich exploitability ===");
    eprintln!("Goal: find config where rich expl > 0.1% pot at some iter checkpoint.");
    eprintln!("nh={} np={} max_iters={} checkpoints={:?}\n", nh, np, max_iters, checkpoints);

    let mut passing_candidate: Option<&Candidate> = None;
    let mut passing_iter: Option<u32> = None;
    let mut passing_expl_pct: f32 = 0.0;

    for cand in &candidates {
        eprintln!("── Candidate: {} ──", cand.name);
        eprintln!("  board={:?} stacks={} chance={}×{}={} outcomes",
            cand.board, cand.stacks,
            cand.turn_cards.len(),
            cand.river_cards_per_turn[0].len(),
            cand.turn_cards.len() * cand.river_cards_per_turn[0].len());

        let (tree, table) = build_table_for(cand, nh, np);
        let game = FlopStartGame::new(table);
        eprintln!("  tree_nodes={}", tree.num_nodes());

        let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
        let ctx = MetalContext::new().expect("Metal");
        let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

        eprintln!("  {:>6}  {:>14}  {:>14}", "iter", "expl (% pot)", "wall (s)");
        let t0 = Instant::now();
        let mut prev = 0u32;
        let mut max_expl_pct = 0.0f32;
        let mut max_expl_iter = 0u32;
        for &cp in &checkpoints {
            if cp > max_iters { break; }
            gpu.run(&ctx, &tree, &game, cp - prev);
            cpu.run(&tree, &game, cp - prev);
            prev = cp;
            let expl = measure_exploitability(&cpu, &tree, &game, np as usize);
            let pct = expl / pot * 100.0;
            eprintln!("  {:>6}  {:>13.4}%  {:>14.1}", cp, pct, t0.elapsed().as_secs_f32());
            if pct > max_expl_pct {
                max_expl_pct = pct;
                max_expl_iter = cp;
            }
        }

        eprintln!("  → max rich expl: {:.4}% pot at iter {}", max_expl_pct, max_expl_iter);
        if max_expl_pct > 0.1 {
            eprintln!("  ✓ GATE PASS (>0.1% pot) — this config is viable for Phase 4 redo.");
            if passing_candidate.is_none() {
                passing_candidate = Some(cand);
                passing_iter = Some(max_expl_iter);
                passing_expl_pct = max_expl_pct;
            }
        } else {
            eprintln!("  ✗ GATE FAIL — rich exploitability never above 0.1% pot.");
        }
        eprintln!();
    }

    match passing_candidate {
        Some(c) => {
            eprintln!("\n=== P0 RESULT ===");
            eprintln!("Selected config: {}", c.name);
            eprintln!("  board={:?} stacks={}", c.board, c.stacks);
            eprintln!("  iter checkpoint: {}", passing_iter.unwrap());
            eprintln!("  rich exploitability at that checkpoint: {:.4}% pot", passing_expl_pct);
            eprintln!("\nP1 (empirical lean selection) can now proceed against this config.");
        }
        None => {
            eprintln!("\n=== P0 FAIL ===");
            eprintln!("No candidate produced rich exploitability > 0.1% pot.");
            eprintln!("Phase 4 redo cannot proceed — the comparison would be vacuous.");
            eprintln!("Next knobs to try: shorter iter budget, denser chance, larger nh.");
            panic!("P0 gate not met by any candidate");
        }
    }
}
