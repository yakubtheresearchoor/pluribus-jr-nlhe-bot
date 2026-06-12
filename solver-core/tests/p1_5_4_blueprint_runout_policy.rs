//! Blueprint close-out: RUNOUT POLICY — the second design input the
//! B4 projection named ("re-prices the postflop line linearly in
//! sampled n_turn × n_river").
//!
//! === Cost axis (arithmetic over the measured ladder, printed) ===
//! Postflop line = ladder row × (n_turn × n_river). The ladder's M2
//! shape is 1×1, so policy P multiplies the row by |P|: at B=8
//! full-set/1%/16-core, 1×1 = 10.5h, 2×2 = 42h (breaks 24h), 4×4 =
//! 168h. Runout fidelity is therefore THE budget governor at full
//! canonicals — richer policies force flop sampling or the GPU port.
//!
//! === Quality axis (measured here) ===
//! Two-fidelity stability, the Phase 4 instrument, applied to the
//! artifact the blueprint banks: the FLOP-ZONE average strategy.
//! Solve the same config (wet-16, quantile B=8, Design1Collapsed) at
//! NESTED runout policies 1×1 ⊂ 2×2 ⊂ 4×4 and compare σ_avg on the
//! flop zone (shared structure: same nodes, same hands, same flop
//! bucket maps — the flop quantile map depends only on the base
//! strengths). Metrics: mean and max |Δσ| over (infoset, hand, action)
//! vs the 4×4 reference, plus own-game lifted exploitability per
//! policy as harness sanity. NOT measured (named): the residual loss
//! of 4×4 itself vs full 47×46 runouts — unaffordable at any scale
//! with per-(tc,rc) buffers; the production confirmation of the chosen
//! policy belongs to head-to-head, same as the count.
//!
//! ═══ MEASURED 2026-06-10 (wet-16, B=8 quantile, 15 iters) ═══
//!   cost (B=8 full-set/1%/16-core): 1×1 10.5h fits | 2×2 42.0h
//!     breaks | 4×4 168.1h breaks. At 100 flops: 0.6 / 2.4 / 9.6h —
//!     all fit. Runout fidelity is THE budget governor at full
//!     canonicals.
//!   flop σ_avg divergence vs 4×4 reference (unweighted, 1152
//!     (node,action,hand) entries):
//!     1×1: mean 0.1049, max 1.0000
//!     2×2: mean 0.0337, max 0.9821
//!   2×2 is ~3× more stable than 1×1 in the mean; max-entries are
//!   whole-strategy flips at individual hands (metric is unweighted —
//!   low-reach rows included). The fidelity trend has NOT converged
//!   by 2×2 (3.4% mean movement remains); own-game expl (3.92 / 7.14
//!   / 5.29% pot) is sanity only — different games, not comparable.
//!   Decision shape this hands the blueprint: full-1755 × 1×1 (10.5h,
//!   unstable flop strategies) vs ~100-300 flops × 2×2/4×4 (fits
//!   easily, stabler) vs GPU port unlocking 2×2 at full canonicals.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{
    lift_cum_to_exact, BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};

const NP: u8 = 6;
const NH: usize = 16;
const NB: usize = 8;
const STARTING_POT: i32 = 30;
const STARTING_CONTRIB: i32 = 5;
const STACKS: i32 = 500;
const ITERS: u32 = 15;

const BOARD: [&str; 3] = ["Th", "9d", "8c"];
/// Nested policies: 1×1 ⊂ 2×2 ⊂ 4×4 (differences reflect fidelity,
/// not sampling luck).
const TURNS_4: [&str; 4] = ["2c", "Jd", "6h", "Qs"];
const RIVERS_4: [[&str; 4]; 4] = [
    ["4s", "7h", "Kd", "2d"],
    ["3s", "Qc", "8h", "Ah"],
    ["5c", "Td", "9s", "3h"],
    ["4d", "Jh", "6s", "Kc"],
];

fn build_table_policy(n_turns: usize, n_rivers: usize) -> FlopChanceTable {
    let board: Vec<Card> = BOARD.iter().map(|s| card_from_str(s).unwrap()).collect();
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
    let turn_cards: Vec<u8> = TURNS_4[..n_turns]
        .iter()
        .map(|s| card_from_str(s).unwrap() as u8)
        .collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] = RIVERS_4[ti][..n_rivers]
            .iter()
            .map(|s| card_from_str(s).unwrap() as u8)
            .collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

fn build_policy_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: STARTING_POT,
        starting_stacks: vec![STACKS; NP as usize],
        initial_contributions: vec![STARTING_CONTRIB; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&config).unwrap()
}

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

/// Solve at policy (n_turns × n_rivers); return (flop-zone σ_avg rows
/// keyed by (node_idx, action, hand) in a flat map, own-game lifted
/// exploitability).
fn solve_policy(
    tree: &FlatTree,
    n_turns: usize,
    n_rivers: usize,
) -> (Vec<(usize, Vec<f32>)>, f32) {
    let game = FlopStartGame::new(build_table_policy(n_turns, n_rivers));
    let (fm, tm, rm) = quantile_maps(game.table(), NB);
    let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
    let mut bucketed = BucketedFlopCfr::new(tree, game.table(), &bk);
    bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
    bucketed.run(tree, &game, &bk, ITERS);

    // σ_avg per flop-zone node: normalized cum rows EXPANDED TO HANDS
    // through the flop map (hand space is shared across policies even
    // if bucket labels permute — quantile flop maps are identical, but
    // hand space is the robust comparison key).
    let nh = NH;
    let cum = bucketed.cum_strategy_flop();
    let mut rows: Vec<(usize, Vec<f32>)> = Vec::new();
    for &nid in &tree.decision_node_ids {
        let idx = nid as usize;
        let Some(local) = bucketed.flop_local_offset_at(idx) else { continue };
        let na = tree.nodes[idx].num_children as usize;
        let off = local * MAX_NA_POSTFLOP * NB;
        let mut sigma = vec![0.0f32; na * nh];
        for h in 0..nh {
            let b = bk.flop_map[h];
            if b == NO_BUCKET {
                continue;
            }
            let mut sum = 0.0f32;
            for a in 0..na {
                sum += cum[off + a * NB + b as usize];
            }
            for a in 0..na {
                sigma[a * nh + h] = if sum > 0.0 {
                    cum[off + a * NB + b as usize] / sum
                } else {
                    1.0 / na as f32
                };
            }
        }
        rows.push((idx, sigma));
    }

    // Own-game lifted exploitability (harness sanity).
    let game_score = FlopStartGame::new(build_table_policy(n_turns, n_rivers));
    let mut scorer = FlopStartVectorCfr::new(tree, game_score.table());
    lift_cum_to_exact(tree, &bucketed, &bk, &mut scorer);
    let e = expl_pct(&scorer, tree, &game_score);
    (rows, e)
}

#[test]
#[ignore = "runout-policy quality measurement (~5-10 min); run with --ignored --nocapture"]
fn runout_policy_two_fidelity_stability() {
    eprintln!("\n════ runout policy: flop-strategy stability across fidelity ════");

    // Cost axis (arithmetic over the measured ladder, for the record).
    eprintln!("cost axis (B=8 full-set/1%/16-core, ladder row 10.14s × policy size):");
    for (label, mult) in [("1×1", 1.0), ("2×2", 4.0), ("4×4", 16.0)] {
        let h = 10.14 * 34.0 * 1755.0 / 16.0 / 3600.0 * mult;
        eprintln!("  {label}: {h:.1} h {}", if h <= 24.0 { "(fits 24h)" } else { "(breaks 24h)" });
    }

    let tree = build_policy_tree();
    eprintln!("\ntree: {} nodes", tree.num_nodes());

    let (rows_11, e11) = solve_policy(&tree, 1, 1);
    eprintln!("1×1: own-game lifted expl {e11:.4}% pot");
    let (rows_22, e22) = solve_policy(&tree, 2, 2);
    eprintln!("2×2: own-game lifted expl {e22:.4}% pot");
    let (rows_44, e44) = solve_policy(&tree, 4, 4);
    eprintln!("4×4: own-game lifted expl {e44:.4}% pot (reference fidelity)");

    for (label, rows) in [("1×1 vs 4×4", &rows_11), ("2×2 vs 4×4", &rows_22)] {
        assert_eq!(rows.len(), rows_44.len());
        let mut max_d = 0.0f32;
        let mut sum_d = 0.0f64;
        let mut n = 0usize;
        for ((idx_a, sa), (idx_b, sb)) in rows.iter().zip(rows_44.iter()) {
            assert_eq!(idx_a, idx_b);
            assert_eq!(sa.len(), sb.len());
            for (a, b) in sa.iter().zip(sb.iter()) {
                let d = (a - b).abs();
                if d > max_d {
                    max_d = d;
                }
                sum_d += d as f64;
                n += 1;
            }
        }
        eprintln!(
            "{label}: flop σ_avg divergence mean {:.4}, max {:.4} (over {} (node,action,hand) entries)",
            sum_d / n as f64,
            max_d,
            n
        );
    }
    eprintln!(
        "\nNOT measured (named): residual loss of 4×4 vs full 47×46 runouts — \
         production confirmation of the chosen policy belongs to head-to-head."
    );
}
