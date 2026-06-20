//! Milestone 1 (CORRECTED per the verified Pluribus architecture): time ONE
//! real-time depth-limited search the RIGHT way —
//!   • search only the CURRENT round (flop); turn+river are the frozen
//!     blueprint continuation (k=4 multi-continuation), NOT re-searched;
//!   • run on a BUCKET-SIZED state space (a small representative hand set),
//!     so the showdown is O(nh_small^np), not the O(1176^2) that hung the
//!     naive first attempt;
//!   • QRE λ sharp for our seat.
//! This validates the corrected loop shape and gives the per-decision wall
//! time (the prod budget). HU (live-2) — the most common postflop spot;
//! multiway needs the bucketed-showdown adapter next.
//!
//! NOTE vs full Pluribus: warm-start (σ←blueprint) and the Bayesian reach
//! prior are NOT yet wired (noted refinements) — cold + uniform reach is a
//! conservative timing upper bound.
//!
//! Run: cargo test --release -p solver-core --test live_search_timing -- --ignored --nocapture

use std::time::Instant;

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};

const NP: u8 = 2; // HU — most common postflop spot
const NH: usize = 40; // bucket-sized representative hand set (≈ a B≈40 abstraction)

fn build_hu_flop_subgame() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = {
        let pt = PreflopChanceTable::new(
            6,
            vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
        );
        pt.canonical_flops[0].to_vec()
    };
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
            continue;
        }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            ranges[p][hi as usize] = 1.0;
        }
    }
    // 1×1 runout (one turn, one river) — the depth-limit continuation only
    // needs a representative runout; this keeps the frozen subtree small.
    let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let turn = deck[12];
    let river = *deck.iter().find(|&&c| c != turn).unwrap();
    let turn_cards = vec![turn];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn as usize] = vec![river];

    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    );

    let spec = production_game_v1();
    let mrc = MAX_NA_POSTFLOP.saturating_sub(2);
    let bets = BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect(),
    };
    let mut cfg = spec.flop_seam_config(2, 6, 20, bets);
    cfg.max_bets_per_street = BetCap::all(3);
    let tree = build_tree(&cfg).unwrap();
    (tree, FlopStartGame::new(table))
}

/// Build the searcher: freeze turn+river (the continuation) to `bp`, with k=4
/// Pluribus continuations + sharp QRE. This is the prod decision setup.
fn build_searcher(tree: &FlatTree, bp: &CpuMccfr, deeper: &[usize], nh: usize, np: usize) -> CpuMccfr {
    let mut s = CpuMccfr::new(tree, vec![nh; np]);
    for &nid in deeper {
        let na = tree.nodes[nid].num_children as usize;
        let st = bp.get_average_strategy(nid, na, nh);
        let flat: Vec<f32> = (0..na).flat_map(|a| (0..nh).map(move |h| (a, h)))
            .map(|(a, h)| st[a][h]).collect();
        s.freeze_node(nid, &flat);
    }
    s.setup_pluribus_continuations(tree, 4, 5.0);
    s.set_lambda(vec![300.0; np]); // sharp (≈ Nash) for our seat
    s
}

#[test]
#[ignore = "live-search timing probe; run with --ignored --nocapture --release"]
fn live_search_hu_flop_timing() {
    let (tree, game) = build_hu_flop_subgame();
    let np = tree.num_players as usize;
    let nh = game.table().num_valid;
    let flop_nodes = (0..tree.num_nodes())
        .filter(|&i| tree.nodes[i].is_player() && tree.nodes[i].board_state == BoardState::Flop as u8)
        .count();
    eprintln!(
        "HU flop subgame: {} nodes ({flop_nodes} flop player nodes), np={np}, nh={nh}",
        tree.num_nodes()
    );

    // Blueprint stand-in: a self-solve of the full subgame (prod loads the
    // banked blueprint here). Its turn+river become the frozen continuation.
    let t_bp = Instant::now();
    let mut bp = CpuMccfr::new(&tree, vec![nh; np]);
    bp.run(&tree, &game, 500);
    eprintln!("blueprint self-solve (500 it): {:.2}s", t_bp.elapsed().as_secs_f64());

    let deeper: Vec<usize> = (0..tree.num_nodes())
        .filter(|&i| tree.nodes[i].is_player()
            && tree.nodes[i].board_state != BoardState::Flop as u8)
        .collect();

    // THE LIVE DECISION: search the FLOP only (turn/river frozen), timed at
    // real-time iter budgets.
    for iters in [100u32, 250, 500] {
        let mut s = build_searcher(&tree, &bp, &deeper, nh, np);
        let t = Instant::now();
        s.run(&tree, &game, iters);
        let secs = t.elapsed().as_secs_f64();
        let na = tree.nodes[0].num_children as usize;
        let strat = s.get_average_strategy(0, na, nh);
        let probs: Vec<f32> = (0..na).map(|a| strat[a][0]).collect();
        let sum: f32 = probs.iter().sum();
        eprintln!(
            "  flop search {iters:>3} it: {secs:7.3}s  ({:.1} ms/it) | root dist(h0) {:?} Σ{sum:.3}",
            secs * 1000.0 / iters as f64,
            probs.iter().map(|p| (p * 1000.0).round() / 1000.0).collect::<Vec<_>>()
        );
        assert!((sum - 1.0).abs() < 1e-2 || sum == 0.0, "root strategy not normalized: {sum}");
    }
    eprintln!("→ per-decision budget = (iters to converge) × (ms/it). Warm-start (σ←blueprint) \
               cuts the iters; that + the Bayesian reach prior are the next refinements.");
}
