//! B4 sweep: bucket counts {5, 8, 10} through the B3 harness with
//! GS14 maps and the Design-1-collapsed terminal (collapse gate green:
//! singleton bit-exact, identity walk bit-exact, f64-ref same-quantity).
//!
//! Maps: the PRODUCTION GS14 pipeline (`build_postflop_bucketing` —
//! equity histograms, EMD k-means, backward recursion) run on the full
//! 1176-hand universe of each config's flop with the config's sampled
//! runouts, then RESTRICTED to the research hand subset. Effective
//! (used) bucket counts are reported alongside nominal — restriction
//! can leave buckets empty, and the curve is honest only against what
//! was actually used.
//!
//! Scale: lean 1-bet tree, NH=16, 15 iters — the instrument-wall
//! ceiling for the exact scorer (named limitation 1,
//! postflop_buckets.rs). Quality numbers are RESEARCH-SCALE; under the
//! exact collapsed terminal the production residual is only how
//! abstraction loss scales with nh. Each point measured independently;
//! non-monotonicity reported, never smoothed.
//!
//! Calibration (separate #[ignore] test): NH=32, B ∈ {10, 20} + exact
//! baseline — one point above the research ceiling for curve shape,
//! plus an nh-scaling pair (B=10 at NH=16 vs NH=32) that probes the
//! named-limitation residual directionally.
//!
//! ═══ MEASURED 2026-06-10 — TWO FINDINGS, ONE USABLE CURVE ═══
//!
//! FINDING A (restriction coarseness): maps fit on the full 1176-hand
//! universe and restricted to the 16-hand subset are far coarser than
//! nominal (wet turn used 3/5 buckets; the Jd→Qc runout on Th9d8c is a
//! BOARD STRAIGHT — 903/1081 hands chop into one cluster). Restriction
//! measures transfer, not the abstraction → switched to fit-on-subset
//! (`build_postflop_bucketing_for_hands`).
//!
//! FINDING B (GS14 at research scale is seed noise): fit-on-subset
//! GS14 lifted exploitability swings wildly with the k-means seed —
//!   dry B=8:  74.77 / 25.49 / 25.64 % pot (seeds 42/43/44)
//!   dry B=10:  4.87 / 13.06 / 14.27 % pot
//! k-means over ~14 alive hands with 2-river histograms has no mass to
//! fit. GS14-fit-at-research-scale CANNOT rank bucket counts; GS14 is
//! a production-scale fit (1176 hands, full runouts) — exactly where
//! the quality verdict is unmeasurable (named limitation 1). The
//! count-quality curve below therefore uses QUANTILE maps (stable
//! control, same harness):
//!
//!   lifted expl % pot (NH=16 lean, 15 iters; baselines 0.65-0.81%):
//!   config  |  B=5   |  B=8   |  B=10  | (GS14 seed-42, for contrast)
//!   wet-16  | 10.29  |  7.14  |  8.41  | 74.98 / 32.55 / 24.24
//!   dry-16  |  7.79  |  7.70  |  7.02  | 40.61 / 74.77 /  4.87
//!   The quantile curve is FLAT-ISH from B=5→10 at ~7-10% pot at this
//!   research scale; non-monotonicity (wet B=10 > B=8) reported, not
//!   smoothed. Production count selection from this scale is NOT
//!   banked — see the session report for the instrument-wall framing.

use solver_core::abstraction::postflop_buckets::{
    build_postflop_bucketing, build_postflop_bucketing_for_hands,
};
use solver_core::card::{card_from_str, card_pair_to_index, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{
    lift_cum_to_exact, BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::time::Instant;

const NP: u8 = 6;
const STARTING_POT: i32 = 30;
const STARTING_CONTRIB: i32 = 5;
const ITERS: u32 = 15;
const GS14_RESTARTS: usize = 10;
const GS14_SEED: u64 = 42;

struct Config {
    name: &'static str,
    board: [&'static str; 3],
    turns: [&'static str; 2],
    rivers: [[&'static str; 2]; 2],
    stacks: i32,
}

const WET: Config = Config {
    name: "wet-16",
    board: ["Th", "9d", "8c"],
    turns: ["2c", "Jd"],
    rivers: [["4s", "7h"], ["3s", "Qc"]],
    stacks: 500,
};
const DRY: Config = Config {
    name: "dry-16",
    board: ["Ks", "7d", "2h"],
    turns: ["Jc", "5d"],
    rivers: [["4s", "Th"], ["3s", "Ac"]],
    stacks: 500,
};
const SHORT: Config = Config {
    name: "short-16",
    board: ["Th", "9d", "8c"],
    turns: ["2c", "Jd"],
    rivers: [["4s", "7h"], ["3s", "Qc"]],
    stacks: 200,
};

fn build_table(cfg: &Config, nh: usize) -> FlopChanceTable {
    let board: Vec<Card> = cfg.board.iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
            continue;
        }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            ranges[p][hi as usize] = 1.0;
        }
    }
    let turn_cards: Vec<u8> = cfg.turns.iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] =
            cfg.rivers[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

fn build_lean_tree(stacks: i32) -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: STARTING_POT,
        starting_stacks: vec![stacks; NP as usize],
        initial_contributions: vec![STARTING_CONTRIB; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    build_tree(&config).unwrap()
}

/// GS14 maps FIT TO the research table's hand subset (the game being
/// solved), via the production pipeline's for_hands entry. Returns
/// (flop_map, turn_maps, river_maps, min effective counts (f, t, r)).
///
/// FINDING (kept for the record): the first sweep attempt fit maps on
/// the full 1176-hand universe and restricted them to the subset —
/// effectively far coarser than nominal (wet turn used 3 of 5 buckets;
/// the Jd→Qc runout on Th9d8c is a BOARD STRAIGHT where 903/1081 hands
/// chop, collapsing every subset hand into the one chop cluster), and
/// wet/short B=5 scored ~59% pot. That measured a restriction
/// transfer, not the abstraction. Diagnostic: diag_wet_river_clusters.
#[allow(clippy::type_complexity)]
fn gs14_maps_subset(
    cfg: &Config,
    table: &FlopChanceTable,
    nb: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>, (usize, usize, usize)) {
    let board: Vec<Card> = cfg.board.iter().map(|s| card_from_str(s).unwrap()).collect();
    let flop: [Card; 3] = [board[0], board[1], board[2]];
    let turn_cards: Vec<Card> = cfg.turns.iter().map(|s| card_from_str(s).unwrap()).collect();
    let river_per_turn: Vec<Vec<Card>> = cfg
        .rivers
        .iter()
        .map(|rs| rs.iter().map(|s| card_from_str(s).unwrap()).collect())
        .collect();

    let nh = table.num_valid;
    let subset_hands: Vec<(Card, Card)> = (0..nh)
        .map(|h| (table.hand_cards[h * 2], table.hand_cards[h * 2 + 1]))
        .collect();
    let pb = build_postflop_bucketing_for_hands(
        subset_hands.clone(),
        &flop,
        &turn_cards,
        &river_per_turn,
        nb,
        nb,
        nb,
        GS14_RESTARTS,
        GS14_SEED,
    );
    // pb.hands == subset_hands in order; maps are direct.
    assert_eq!(pb.hands, subset_hands);
    let flop_map = pb.flop_map.clone();
    let turn_maps = pb.turn_map.clone();
    let river_maps = pb.river_map.clone();

    let used = |m: &[u16]| -> usize {
        let mut seen = vec![false; nb];
        for &b in m {
            if b != NO_BUCKET {
                seen[b as usize] = true;
            }
        }
        seen.iter().filter(|&&s| s).count()
    };
    let eff_flop = used(&flop_map);
    let eff_turn = turn_maps.iter().map(|m| used(m)).min().unwrap();
    let eff_river = river_maps
        .iter()
        .flat_map(|tm| tm.iter().map(|m| used(m)))
        .min()
        .unwrap();
    (flop_map, turn_maps, river_maps, (eff_flop, eff_turn, eff_river))
}

fn expl_pct(cpu: &FlopStartVectorCfr, tree: &FlatTree, game: &FlopStartGame) -> f32 {
    let np = tree.num_players as usize;
    let mut total = 0.0f32;
    for p in 0..np {
        let br = cpu.best_response_value_debug(tree, game, p as u8);
        let sv = cpu.strategy_value_debug(tree, game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total += (br[h] - sv[h]).max(0.0);
        }
    }
    total / STARTING_POT as f32 * 100.0
}

fn sweep_point(cfg: &Config, tree: &FlatTree, nh: usize, nb: usize) -> (f32, (usize, usize, usize)) {
    let game = FlopStartGame::new(build_table(cfg, nh));
    let t0 = Instant::now();
    let (fm, tm, rm, eff) = gs14_maps_subset(cfg, game.table(), nb);
    let map_s = t0.elapsed().as_secs_f64();
    let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
    let mut bucketed = BucketedFlopCfr::new(tree, game.table(), &bk);
    bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
    let t0 = Instant::now();
    bucketed.run(tree, &game, &bk, ITERS);

    let game_score = FlopStartGame::new(build_table(cfg, nh));
    let mut scorer = FlopStartVectorCfr::new(tree, game_score.table());
    lift_cum_to_exact(tree, &bucketed, &bk, &mut scorer);
    let e = expl_pct(&scorer, tree, &game_score);
    eprintln!(
        "  {} B={nb}: lifted {e:.4}% pot | effective (f/t/r) {:?} | maps {map_s:.2}s, solve+score {:.1}s",
        cfg.name,
        eff,
        t0.elapsed().as_secs_f64()
    );
    (e, eff)
}

fn baseline(cfg: &Config, tree: &FlatTree, nh: usize) -> f32 {
    let game = FlopStartGame::new(build_table(cfg, nh));
    let mut exact = FlopStartVectorCfr::new(tree, game.table());
    exact.run(tree, &game, ITERS);
    let e = expl_pct(&exact, tree, &game);
    eprintln!("  {} baseline (exact): {e:.4}% pot", cfg.name);
    e
}

#[test]
#[ignore = "B4 sweep {5,8,10} × 3 configs at NH=16 (~3-10 min); run with --ignored --nocapture"]
fn b4_sweep_counts() {
    const NH: usize = 16;
    for cfg in [&WET, &DRY, &SHORT] {
        let tree = build_lean_tree(cfg.stacks);
        eprintln!("\n=== {} === ({} nodes, NH={NH}, {ITERS} iters)", cfg.name, tree.num_nodes());
        let base = baseline(cfg, &tree, NH);
        let mut results = Vec::new();
        for nb in [5usize, 8, 10] {
            let (e, eff) = sweep_point(cfg, &tree, NH, nb);
            results.push((nb, e, eff));
        }
        eprintln!("  {} curve: baseline {base:.4}% | {}", cfg.name,
            results.iter().map(|(nb, e, _)| format!("B={nb}: {e:.4}%")).collect::<Vec<_>>().join(" | "));
        // Sanity only — counts are ranked by the printed numbers, never
        // asserted against each other (non-monotonicity is reported,
        // not failed).
        for (nb, e, _) in &results {
            assert!(e.is_finite(), "{} B={nb}: non-finite exploitability", cfg.name);
        }
    }
}

#[test]
#[ignore = "B4 calibration: NH=32, B∈{10,20} + baseline (~30-60 min); run with --ignored --nocapture"]
fn b4_calibration_nh32() {
    const NH: usize = 32;
    let cfg = &WET;
    let tree = build_lean_tree(cfg.stacks);
    eprintln!("\n=== calibration {} NH={NH} === ({} nodes, {ITERS} iters)", cfg.name, tree.num_nodes());
    let base = baseline(cfg, &tree, NH);
    for nb in [10usize, 20] {
        let (e, _) = sweep_point(cfg, &tree, NH, nb);
        eprintln!("  calibration B={nb}: cost over baseline {:+.4}% pot", e - base);
    }
    eprintln!("  (B=10 here vs B=10 at NH=16 in the sweep = the nh-scaling pair");
    eprintln!("   for named-limitation 1 — directional only, never extrapolated)");
}

/// Diagnostic for the wet/short effective-river-1 degeneracy: print
/// per-runout subset bucket assignments and full-universe cluster
/// occupancy for the river layer.
#[test]
#[ignore = "diagnostic; run with --ignored --nocapture"]
fn diag_wet_river_clusters() {
    const NH: usize = 16;
    let cfg = &WET;
    let nb = 8usize;
    let game = FlopStartGame::new(build_table(cfg, NH));
    let table = game.table();

    let board: Vec<Card> = cfg.board.iter().map(|s| card_from_str(s).unwrap()).collect();
    let flop: [Card; 3] = [board[0], board[1], board[2]];
    let turn_cards: Vec<Card> = cfg.turns.iter().map(|s| card_from_str(s).unwrap()).collect();
    let river_per_turn: Vec<Vec<Card>> = cfg
        .rivers
        .iter()
        .map(|rs| rs.iter().map(|s| card_from_str(s).unwrap()).collect())
        .collect();
    let pb = build_postflop_bucketing(
        &flop, &turn_cards, &river_per_turn, nb, nb, nb, GS14_RESTARTS, GS14_SEED,
    );

    let mut full_idx = vec![usize::MAX; NUM_POSSIBLE_HANDS];
    for (li, &(c1, c2)) in pb.hands.iter().enumerate() {
        full_idx[card_pair_to_index(c1, c2)] = li;
    }
    let nh = table.num_valid;
    for ti in 0..turn_cards.len() {
        for ri in 0..river_per_turn[ti].len() {
            // Full-universe cluster occupancy at this runout.
            let mut occ = vec![0usize; nb];
            for li in 0..pb.hands.len() {
                let b = pb.river_map[ti][ri][li];
                if b != u16::MAX {
                    occ[b as usize] += 1;
                }
            }
            // Subset assignments.
            let subset: Vec<i32> = (0..nh)
                .map(|h| {
                    let li = full_idx[card_pair_to_index(
                        table.hand_cards[h * 2],
                        table.hand_cards[h * 2 + 1],
                    )];
                    let b = pb.river_map[ti][ri][li];
                    if b == u16::MAX { -1 } else { b as i32 }
                })
                .collect();
            eprintln!("runout (ti={ti}, ri={ri}): full-universe occupancy {occ:?}");
            eprintln!("  subset buckets: {subset:?}");
        }
    }
    eprintln!("river cluster means (equity): {:?}", pb.river_cluster_means);
}

/// Control: strength-quantile maps at the same {5,8,10} on the same
/// configs. Separates "regime is brutal for any abstraction" from
/// "GS14-on-subset produces pathological maps".
fn quantile_maps_ctl(
    table: &FlopChanceTable,
    nb: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8]| -> Vec<u16> {
        let alive: Vec<usize> = pl_idx[..nh]
            .iter()
            .map(|&i| i as usize)
            .filter(|&h| !conflicts(h, dead))
            .collect();
        let n = alive.len();
        assert!(n >= nb);
        let mut map = vec![NO_BUCKET; nh];
        for (pos, &h) in alive.iter().enumerate() {
            map[h] = ((pos * nb) / n) as u16;
        }
        map
    };
    let (_, _, _, base_pi, _) = table.sorted_opp_arrays_base();
    let flop_map = map_for(&base_pi, &[]);
    let mut turn_maps = Vec::new();
    let mut river_maps = Vec::new();
    for &tc_card in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
        turn_maps.push(map_for(pi, &[tc_card]));
        let mut rms = Vec::new();
        for &rc_card in &table.river_decks[tc_card as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
            rms.push(map_for(pi, &[tc_card, rc_card]));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
}

#[test]
#[ignore = "control: quantile maps at {5,8,10} on wet/dry NH=16 (~4 min); run with --ignored --nocapture"]
fn b4_sweep_quantile_control() {
    const NH: usize = 16;
    for cfg in [&WET, &DRY] {
        let tree = build_lean_tree(cfg.stacks);
        eprintln!("\n=== {} QUANTILE CONTROL === (NH={NH}, {ITERS} iters)", cfg.name);
        for nb in [5usize, 8, 10] {
            let game = FlopStartGame::new(build_table(cfg, NH));
            let (fm, tm, rm) = quantile_maps_ctl(game.table(), nb);
            let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
            let mut bucketed = BucketedFlopCfr::new(&tree, game.table(), &bk);
            bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
            bucketed.run(&tree, &game, &bk, ITERS);
            let game_score = FlopStartGame::new(build_table(cfg, NH));
            let mut scorer = FlopStartVectorCfr::new(&tree, game_score.table());
            lift_cum_to_exact(&tree, &bucketed, &bk, &mut scorer);
            let e = expl_pct(&scorer, &tree, &game_score);
            eprintln!("  {} quantile B={nb}: lifted {e:.4}% pot", cfg.name);
        }
    }
}

#[test]
#[ignore = "control: GS14 seed sensitivity, dry B in {8,10} seeds {42,43,44} (~5 min); run with --ignored --nocapture"]
fn b4_sweep_seed_sensitivity() {
    const NH: usize = 16;
    let cfg = &DRY;
    let tree = build_lean_tree(cfg.stacks);
    eprintln!("\n=== {} GS14 SEED SENSITIVITY ===", cfg.name);
    for nb in [8usize, 10] {
        for seed in [42u64, 43, 44] {
            let game = FlopStartGame::new(build_table(cfg, NH));
            let table = game.table();
            let board: Vec<Card> = cfg.board.iter().map(|s| card_from_str(s).unwrap()).collect();
            let flop: [Card; 3] = [board[0], board[1], board[2]];
            let turn_cards: Vec<Card> =
                cfg.turns.iter().map(|s| card_from_str(s).unwrap()).collect();
            let river_per_turn: Vec<Vec<Card>> = cfg
                .rivers
                .iter()
                .map(|rs| rs.iter().map(|s| card_from_str(s).unwrap()).collect())
                .collect();
            let subset_hands: Vec<(Card, Card)> = (0..table.num_valid)
                .map(|h| (table.hand_cards[h * 2], table.hand_cards[h * 2 + 1]))
                .collect();
            let pb = build_postflop_bucketing_for_hands(
                subset_hands, &flop, &turn_cards, &river_per_turn, nb, nb, nb,
                GS14_RESTARTS, seed,
            );
            let bk = FlopBucketing::from_maps(
                game.table(), nb, nb, nb,
                pb.flop_map.clone(), pb.turn_map.clone(), pb.river_map.clone(),
            );
            let mut bucketed = BucketedFlopCfr::new(&tree, game.table(), &bk);
            bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
            bucketed.run(&tree, &game, &bk, ITERS);
            let game_score = FlopStartGame::new(build_table(cfg, NH));
            let mut scorer = FlopStartVectorCfr::new(&tree, game_score.table());
            lift_cum_to_exact(&tree, &bucketed, &bk, &mut scorer);
            let e = expl_pct(&scorer, &tree, &game_score);
            eprintln!("  {} GS14 B={nb} seed={seed}: lifted {e:.4}% pot", cfg.name);
        }
    }
}
