//! B3 step 5: the quality gate — the bucketed strategy lifted to
//! per-hand granularity and scored in the UNBUCKETED game with the
//! exact terminal (the cross_tree lift precedent at hand granularity).
//!
//! This is the gate that certifies what the identity gate structurally
//! cannot: within-bucket reach drift across imperfect-recall boundaries
//! and the named terminal approximations (opp-opp pairwise blocking,
//! within-bucket relation mixing) — all of it lands in one measured
//! number, the lifted strategy's exploitability in the real game.
//!
//! Three checks, in order:
//!   1. LIFT ANCHOR — lifting the B = nh identity bucketing must
//!      reproduce the exact solver's cum_strategy buffers bit-for-bit
//!      (validates the lift indexing itself; piggybacks on the step-4
//!      identity gate).
//!   2. CONFIG VALIDITY — mixed equilibrium (some infosets genuinely
//!      mix, so abstraction damage has somewhere to show) and a
//!      CONVERGED baseline (so the lifted strategy's exploitability is
//!      attributable to the abstraction, not residual non-convergence).
//!      NOTE on "nonzero headroom": the Phase 4 configs' verified
//!      nonzero self-exploitability (~0.87% pot at iter 25) was with
//!      the RICH action set; this harness runs the lean production set
//!      (MAX_NA_POSTFLOP = 4 bank), where self-play converges to the
//!      floor in 30 iters. That is NOT the Phase-4-v4 vacuousness trap
//!      (0-vs-0, instrument blind): here the B=2 damage signal lands
//!      five orders of magnitude above the baseline, demonstrating the
//!      instrument registers damage — which is what headroom is FOR.
//!   3. 2-BUCKET SANITY — B = 2 strength-quantile bucketing must be
//!      MEASURABLY bad. If collapsing each street to two buckets does
//!      not hurt, the harness cannot be trusted to rank bucket counts
//!      in B4.
//!
//! Bucket counts are NOT swept here (that is B4, each count measured
//! independently — never interpolated; B=4 is printed for information
//! only, never asserted against B=2). The quantile maps used for the
//! sanity check are deliberately crude; B4 feeds GS14 maps through this
//! same harness.
//!
//! Measured (2026-06-10, this harness, 30 iters, NH=6 sampled hands,
//! lean action set):
//!   wet-deep   baseline 0.0001% | mixed 63.3% | B=2 56.79% | B=4 2.09%
//!   dry-deep   baseline 0.0048% | mixed 60.5% | B=2 59.31% | B=4 5.99%
//!   wet-short  baseline 0.0000% | mixed 55.5% | B=2 57.87% | B=4 1.89%
//! The B=4 spread across configs (1.9–6.0% pot) is exactly the kind of
//! config-dependence the B4 cross-config check exists to measure —
//! never extrapolate one config's count-quality curve to another.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{
    lift_cum_to_exact, BucketedFlopCfr, FlopBucketing, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};

const NP: u8 = 6;
const NH: usize = 6;
const STARTING_POT: i32 = 30;
const STARTING_CONTRIB: i32 = 5;
const ITERS: u32 = 30;

struct Config {
    name: &'static str,
    board: [&'static str; 3],
    turns: [&'static str; 2],
    rivers: [[&'static str; 2]; 2],
    stacks: i32,
}

const WET_DEEP: Config = Config {
    name: "wet-deep",
    board: ["Th", "9d", "8c"],
    turns: ["2c", "Jd"],
    rivers: [["4s", "7h"], ["3s", "Qc"]],
    stacks: 500,
};
const DRY_DEEP: Config = Config {
    name: "dry-deep",
    board: ["Ks", "7d", "2h"],
    turns: ["Jc", "5d"],
    rivers: [["4s", "Th"], ["3s", "Ac"]],
    stacks: 500,
};
const WET_SHORT: Config = Config {
    name: "wet-short",
    board: ["Th", "9d", "8c"],
    turns: ["2c", "Jd"],
    rivers: [["4s", "7h"], ["3s", "Qc"]],
    stacks: 200,
};

fn build_table(cfg: &Config) -> FlopChanceTable {
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
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            ranges[p][hi as usize] = 1.0;
        }
    }
    let turn_cards: Vec<u8> = cfg.turns.iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] = cfg.rivers[ti]
            .iter()
            .map(|s| card_from_str(s).unwrap() as u8)
            .collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

fn build_cfg_tree(cfg: &Config) -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: STARTING_POT,
        starting_stacks: vec![cfg.stacks; NP as usize],
        initial_contributions: vec![STARTING_CONTRIB; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.33), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&config).unwrap()
}

/// Phase 4's exploitability metric: Σ_p Σ_h max(br − sv, 0), as % pot.
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

/// Strength-quantile maps: per runout, hands sorted by that runout's
/// strength (the table's pl_idx order IS that sort) are split into nb
/// contiguous groups; runout-conflicting hands get NO_BUCKET.
fn quantile_maps(
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
        assert!(n >= nb, "fewer alive hands than buckets");
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

/// Run the bucketed solver and score the lifted strategy in the exact
/// game. Returns (lifted exploitability % pot).
fn bucketed_lifted_expl(
    cfg: &Config,
    tree: &FlatTree,
    nb: usize,
    iters: u32,
) -> f32 {
    let game = FlopStartGame::new(build_table(cfg));
    let (fm, tm, rm) = quantile_maps(game.table(), nb);
    let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
    let mut bucketed = BucketedFlopCfr::new(tree, game.table(), &bk);
    bucketed.run(tree, &game, &bk, iters);

    let game_score = FlopStartGame::new(build_table(cfg));
    let mut scorer = FlopStartVectorCfr::new(tree, game_score.table());
    lift_cum_to_exact(tree, &bucketed, &bk, &mut scorer);
    expl_pct(&scorer, tree, &game_score)
}

/// Mixed-equilibrium probe: fraction of flop-zone (infoset, hand) rows
/// whose normalized average strategy has max action prob < 0.9.
fn mixed_fraction_flop(exact: &FlopStartVectorCfr, tree: &FlatTree) -> f32 {
    let nh = NH;
    let cum = exact.cum_strategy_flop();
    let mut rows = 0usize;
    let mut mixed = 0usize;
    for &nid in &tree.decision_node_ids {
        let idx = nid as usize;
        let Some(local) = exact.flop_local_offset_at(idx) else { continue };
        let na = tree.nodes[idx].num_children as usize;
        let off = local * MAX_NA_POSTFLOP * nh;
        for h in 0..nh {
            let mut sum = 0.0f32;
            let mut maxv = 0.0f32;
            for a in 0..na {
                let v = cum[off + a * nh + h];
                sum += v;
                if v > maxv {
                    maxv = v;
                }
            }
            if sum > 0.0 {
                rows += 1;
                if maxv / sum < 0.9 {
                    mixed += 1;
                }
            }
        }
    }
    if rows == 0 { 0.0 } else { mixed as f32 / rows as f32 }
}

fn run_config(cfg: &Config) -> (f32, f32, f32, f32) {
    let tree = build_cfg_tree(cfg);
    eprintln!("\n=== {} === ({} nodes)", cfg.name, tree.num_nodes());

    // 2. Config validity: exact baseline.
    let game = FlopStartGame::new(build_table(cfg));
    let mut exact = FlopStartVectorCfr::new(&tree, game.table());
    exact.run(&tree, &game, ITERS);
    let baseline = expl_pct(&exact, &tree, &game);
    let mixed = mixed_fraction_flop(&exact, &tree);
    eprintln!(
        "  baseline (exact, {ITERS} iters): {baseline:.4}% pot | mixed flop rows: {:.1}%",
        mixed * 100.0
    );

    // 3. 2-bucket sanity + a B=4 informational point.
    let b2 = bucketed_lifted_expl(cfg, &tree, 2, ITERS);
    let b4 = bucketed_lifted_expl(cfg, &tree, 4, ITERS);
    eprintln!("  lifted B=2: {b2:.4}% pot (cost {:+.4}%)", b2 - baseline);
    eprintln!("  lifted B=4: {b4:.4}% pot (cost {:+.4}%)", b4 - baseline);

    (baseline, mixed, b2, b4)
}

/// Lift anchor: lifting the identity (B = nh) bucketing reproduces the
/// exact solver's cum buffers bit-for-bit.
#[test]
fn lift_anchor_identity_bit_exact() {
    let cfg = &WET_DEEP;
    let tree = build_cfg_tree(cfg);

    let game_a = FlopStartGame::new(build_table(cfg));
    let mut exact = FlopStartVectorCfr::new(&tree, game_a.table());
    exact.run(&tree, &game_a, 5);

    let game_b = FlopStartGame::new(build_table(cfg));
    let bk = FlopBucketing::identity(game_b.table());
    let mut bucketed = BucketedFlopCfr::new(&tree, game_b.table(), &bk);
    bucketed.run(&tree, &game_b, &bk, 5);

    let game_c = FlopStartGame::new(build_table(cfg));
    let mut lifted = FlopStartVectorCfr::new(&tree, game_c.table());
    lift_cum_to_exact(&tree, &bucketed, &bk, &mut lifted);

    for (label, a, b) in [
        ("flop", exact.cum_strategy_flop(), lifted.cum_strategy_flop()),
        ("turn", exact.cum_strategy_turn(), lifted.cum_strategy_turn()),
        ("river", exact.cum_strategy_river(), lifted.cum_strategy_river()),
    ] {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "lift anchor: cum_{label}[{i}] exact {} vs lifted {}",
                a[i],
                b[i]
            );
        }
    }
    eprintln!("lift anchor PASSED: identity lift reproduces exact cum buffers bit-for-bit");
}

#[test]
fn quality_gate_wet_deep() {
    let (baseline, mixed, b2, _b4) = run_config(&WET_DEEP);

    // Config validity: converged baseline (clean attribution) and a
    // genuinely mixed equilibrium. See header for why the baseline is
    // at the floor here (lean action set) and why that is not the
    // Phase-4-v4 vacuousness trap.
    assert!(
        baseline < 0.05,
        "wet-deep baseline {baseline:.4}% pot — not converged enough to \
         attribute lifted-strategy exploitability to the abstraction"
    );
    assert!(
        mixed > 0.05,
        "wet-deep equilibrium not mixed (only {:.1}% of flop rows mix) — \
         a near-pure equilibrium can hide abstraction damage",
        mixed * 100.0
    );

    // 2-bucket sanity: must be measurably bad (measured 56.79% pot).
    assert!(
        b2 > baseline + 1.0,
        "B=2 sanity FAILED: lifted B=2 exploitability {b2:.4}% pot is not \
         measurably worse than baseline {baseline:.4}% — harness cannot be \
         trusted to rank bucket counts"
    );
}

#[test]
#[ignore = "cross-config quality gate (dry-deep + wet-short), ~4 min; run with --ignored"]
fn quality_gate_cross_configs() {
    for cfg in [&DRY_DEEP, &WET_SHORT] {
        let (baseline, mixed, b2, _b4) = run_config(cfg);
        assert!(
            baseline < 0.05,
            "{}: baseline {baseline:.4}% pot not converged",
            cfg.name
        );
        assert!(mixed > 0.05, "{}: equilibrium not mixed", cfg.name);
        assert!(
            b2 > baseline + 1.0,
            "{}: B=2 sanity FAILED ({b2:.4}% vs baseline {baseline:.4}%)",
            cfg.name
        );
    }
}
