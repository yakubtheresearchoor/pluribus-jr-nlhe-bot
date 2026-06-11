//! B4 step 1: bucketed-iteration cost at PRODUCTION nh — measured,
//! per-stage, BEFORE any quality sweep compute is spent.
//!
//! THE GAP BEING MEASURED: the M4 feasibility table ("bucket count
//! 15-25 fits 24h") was priced from the measured GPU cost formula
//!     iter_ms(nh, np=6) ≈ 335 · (nh/8)^4.84
//! evaluated AT nh = B — i.e. a game where EVERYTHING (reach, cfv,
//! tree walk, terminals) runs at bucket granularity. The architecture
//! B3 actually built is a deliberate hybrid: per-hand reach/cfv (card
//! removal verbatim), bucket-granular storage and terminals via
//! reduce→Design-1→expand. Its cost is
//!     per-hand stages at REAL nh (≈1176)
//!   + bucketed terminals (arm 1 ~B^(K+1); arm 2 ~(3B)^K_active —
//!     relation branching, a term the B^(K+1) framing hides)
//!   + reduce/expand seams at O(np · nh) per terminal
//! and NONE of those is iter_ms(B). This test measures the real thing.
//!
//! Protocol (measure-one-unit-first discipline):
//!   1. Unit probes: single bucketed terminal (arm 1 and arm 2 at
//!      production fold counts), single per-hand reach pass at nh=1176,
//!      terminal census of the tree.
//!   2. Projection printed from probes; the full iteration runs only if
//!      projected < PROJECTION_BUDGET_S (else that IS the finding).
//!   3. One full bucketed iteration at (nh=1176, B=15) with per-stage
//!      wall-clock attribution (public-API replication of run()'s loop;
//!      terminal share attributed arithmetically from probes × census).
//!   4. Reference: one EXACT CPU iteration at nh = B = 15 on the same
//!      tree — the "everything at B" game the M4 table priced, on the
//!      SAME device, so the hybrid-overhead ratio is device-free.
//!
//! Decision rule (from the B4 go-order): if the device-free hybrid
//! ratio (or the absolute cost vs the formula's intent) lands > ~2× the
//! everything-at-B prediction, STOP AND REPORT — that triggers the
//! design fork (bucket the K≤3 arms / fully bucket-granular chance)
//! rather than the sweep.
//!
//! Tree: the M2 measurement tree (six_player_nh_scaling's asymmetric
//! 6p config: 1 bet size, no raises, 1 turn × 1 river, stacks 200,
//! contribs [10,5,5,5,5,5]) so the comparison hits the SAME game the
//! formula was fit on. num_combinations is set to 1.0 — the exact
//! enumeration is O(nh^6) at nh=1176 and nc only scales cfv readout,
//! not cost.
//!
//! Maps: strength-quantile at B=15. Map QUALITY is irrelevant to cost
//! (cost depends on B and the dispatch arms, not on which hands share
//! a bucket); GS14 maps go through the sweep, not through this gate.
//!
//! ═══ MEASURED 2026-06-10 — STOP-AND-REPORT FIRED ═══
//! M2 tree (2431 nodes, 1033 terminals: 138 arm-1, 895 arm-2), nh=1176:
//!   per-hand reach pass:            0.001 s   (68 MB buffer) — FREE
//!   arm-1 terminal (K=5):           0.128 s
//!   arm-2 terminal (5 active):      0.716 s   ((3B)^5 relation branching)
//!   arm-2 terminal (4 fold/1 act):  0.288 s   (folded opps still cost B-depth)
//!   projected iteration:            ≈ 3953 s  vs M4 prediction 7.0 s → 563×
//! Full iteration NOT run (projection >> budget); the projection is the
//! finding. Neither pre-named fork branch fired: per-hand stages are
//! negligible and there is no K≤2-at-full-nh term at np=6 (num_opp is
//! always 5; fold terminals route through the 5-deep arms). The entire
//! overrun is the Design-1 terminal itself: bucket-tuple enumeration
//! has NO pruning (bucket reach is dense, f_n > 0 between almost all
//! bucket pairs — the conflict/zero-reach pruning that makes hand-tuple
//! enumeration at nh=B cheap simply vanishes in bucket space), arm 2
//! multiplies by ~3^K_active relation branching the exact arm never
//! pays (it reads relations from strengths in O(1) per tuple), and
//! folded opponents still cost a full B-wide recursion level each.
//! → Design fork: Design 2 (factored-over-buckets, O(K·B²)) per the
//! B1 report's pre-logged road, with its factorization error measured
//! against Design 1 at small B; Design 1 stays the validation anchor
//! (identity gate bit-exact) — see session report.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::bucketed_showdown::{bucketed_showdown_cfv, bucketed_showdown_cfv_factored};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::time::Instant;

const NP: u8 = 6;
const B: usize = 15;
/// If the probe-based projection exceeds this, skip the full iteration:
/// the projection itself is the stop-and-report finding.
const PROJECTION_BUDGET_S: f64 = 900.0;

// ── M2 tree + full-nh table (six_player_nh_scaling construction,
//    parameterized over the chosen hand set, nc fixed at 1.0) ──

fn build_m2_table(nh: usize) -> FlopChanceTable {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_opp = NP as usize - 1;

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let chosen: Vec<u16> = if nh >= all_valid.len() {
        all_valid.clone()
    } else {
        let step = all_valid.len() / nh;
        (0..nh).map(|i| all_valid[i * step]).collect()
    };
    let nh = chosen.len();

    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
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
        let mut items: Vec<(u16, u16)> =
            (0..nh).map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            let off = t as usize * num_opp * nh + oi * nh;
            for h in 0..nh {
                turn_sorted_str[off + h] = items[h].0;
                turn_sorted_idx[off + h] = items[h].1;
            }
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
                river_ranks[t as usize * 52 * nh + r as usize * nh + i] =
                    h.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh {
                    river_sorted_str[off + h] = items[h].0;
                    river_sorted_idx[off + h] = items[h].1;
                }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; NP as usize];
    FlopChanceTable {
        hand_ranks_base: hr,
        valid_hand_indices: chosen,
        num_valid: nh,
        conflict,
        hand_cards,
        remaining_deck: tc,
        turn_ranks,
        turn_sorted_str,
        turn_sorted_idx,
        river_ranks,
        river_sorted_str,
        river_sorted_idx,
        initial_weights: iw,
        num_players: NP,
        // Exact enumeration is O(nh^6) at nh=1176; nc scales cfv readout
        // only, never cost. 1.0 keeps the measurement honest and cheap.
        num_combinations: 1.0,
        river_decks: rd,
    }
}

fn build_m2_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
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

/// Strength-quantile maps (same construction as the B3 quality gate).
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

/// Terminal census: per zone, terminals split by dispatch arm
/// (arm 1 = equal contribs & no folds; arm 2 = everything else),
/// with arm-2 active/folded opponent counts.
struct Census {
    arm1: usize,
    arm2: Vec<(usize, usize)>, // (n_active_opp, n_folded_opp) per terminal
}

fn census_zone(tree: &FlatTree, zones: &[Zone], zone: Zone) -> Census {
    let np = NP as usize;
    let mut arm1 = 0usize;
    let mut arm2 = Vec::new();
    for idx in 0..tree.num_nodes() {
        if zones[idx] != zone || !tree.nodes[idx].is_terminal() { continue; }
        let fold_mask = tree.get_folded_mask(idx);
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let active: Vec<i32> = (0..np)
            .filter(|&p| fold_mask & (1u16 << p) == 0)
            .map(|p| contribs[p])
            .collect();
        let all_equal = active.windows(2).all(|w| w[0] == w[1]);
        if all_equal && fold_mask == 0 {
            arm1 += 1;
        } else {
            // Census from traverser-0 perspective (counts differ by ±1
            // across traversers; fine for projection purposes).
            let n_folded = (1..np).filter(|&p| fold_mask & (1u16 << p) != 0).count();
            arm2.push((np - 1 - n_folded, n_folded));
        }
    }
    Census { arm1, arm2 }
}

#[test]
#[ignore = "B4 step-1 cost gate (~minutes, prints measurement); run with --ignored --nocapture"]
fn b4_cost_measurement_production_nh() {
    let m4_pred_ms = 335.0 * (B as f64 / 8.0).powf(4.84);
    eprintln!("\n════ B4 step 1: bucketed iteration cost at production nh ════");
    eprintln!("M4 prediction (everything-at-B GPU formula) at B={B}: {:.1} s/iter", m4_pred_ms / 1000.0);

    let tree = build_m2_tree();
    eprintln!("M2 tree: {} nodes, {} decision, {} terminals",
        tree.num_nodes(), tree.decision_node_ids.len(),
        (0..tree.num_nodes()).filter(|&i| tree.nodes[i].is_terminal()).count());

    // ── Full-nh table + bucketing ──
    let t0 = Instant::now();
    let table = build_m2_table(usize::MAX);
    let nh = table.num_valid;
    eprintln!("full-nh table: nh={nh}, built in {:.1}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let (fm, tm, rm) = quantile_maps(&table, B);
    let game = FlopStartGame::new(table);
    let bk = FlopBucketing::from_maps(game.table(), B, B, B, fm, tm, rm);
    eprintln!("maps + W/T/L tables (3 runouts): {:.2}s", t0.elapsed().as_secs_f64());

    let mut bucketed = BucketedFlopCfr::new(&tree, game.table(), &bk);

    // ── Census (zones via a throwaway exact solver at tiny nh) ──
    let small_table = build_m2_table(B);
    let zones_probe = FlopStartVectorCfr::new(&tree, &small_table);
    let zones = zones_probe.zones().to_vec();
    let census_f = census_zone(&tree, &zones, Zone::Flop);
    let census_t = census_zone(&tree, &zones, Zone::Turn);
    let census_r = census_zone(&tree, &zones, Zone::River);
    for (name, c) in [("flop", &census_f), ("turn", &census_t), ("river", &census_r)] {
        eprintln!("  {name} zone: {} arm-1 terminals, {} arm-2 (active/folded sample: {:?})",
            c.arm1, c.arm2.len(),
            &c.arm2.iter().take(4).collect::<Vec<_>>());
    }

    // ── Unit probes ──
    eprintln!("\n── unit probes (B={B}) ──");
    let uniform_reach: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0 / B as f32; B]).collect();
    let views: Vec<&[f32]> = uniform_reach.iter().map(|v| v.as_slice()).collect();

    // Arm 1: equal contributions, no folds (B^(K+1)-class).
    let t0 = Instant::now();
    let _ = bucketed_showdown_cfv(
        &views, &bk.flop_tables, &[20; 6], 0, 0, NP, 30, 0.0, 0.0, true,
    );
    let arm1_s = t0.elapsed().as_secs_f64();
    eprintln!("arm-1 terminal (equal, no folds, K=5): {:.3}s", arm1_s);

    // Arm 2 worst case: unequal contributions, NO folds → 5 active
    // opponents × 3-way relation branching ≈ (3B)^5 paths.
    let t0 = Instant::now();
    let _ = bucketed_showdown_cfv(
        &views, &bk.flop_tables, &[10, 20, 30, 30, 20, 10], 0, 0, NP, 30, 0.0, 0.0, true,
    );
    let arm2_active5_s = t0.elapsed().as_secs_f64();
    eprintln!("arm-2 terminal (unequal, 5 active): {:.3}s  (~(3B)^5 relation branching)", arm2_active5_s);

    // Arm 2 typical fold terminal: 4 folded, 1 active.
    let t0 = Instant::now();
    let _ = bucketed_showdown_cfv(
        &views, &bk.flop_tables, &[10, 20, 5, 5, 5, 5], 0b111100, 0, NP, 30, 0.0, 0.0, true,
    );
    let arm2_fold4_s = t0.elapsed().as_secs_f64();
    eprintln!("arm-2 terminal (4 folded, 1 active): {:.3}s", arm2_fold4_s);

    // Per-hand reach pass at full nh.
    bucketed.compute_flop_strategy(&tree, &bk);
    let t0 = Instant::now();
    let flop_reach = bucketed.compute_reach_flop(&tree, &game, &bk);
    let reach_s = t0.elapsed().as_secs_f64();
    eprintln!("compute_reach_flop at nh={nh}: {:.3}s ({} MB buffer)",
        reach_s, tree.num_nodes() * NP as usize * nh * 4 / 1_000_000);
    drop(flop_reach);

    // ── Projection from probes ──
    let arm2_cost = |c: &Census| -> f64 {
        c.arm2.iter().map(|&(act, _)| match act {
            a if a >= 4 => arm2_active5_s,      // conservative: 5-active probe
            _ => arm2_fold4_s.max(arm2_active5_s * 0.05),
        }).sum()
    };
    let zone_terminal_s = |c: &Census| c.arm1 as f64 * arm1_s + arm2_cost(c);
    // Zone walks per iteration: river 6 traversers × 1 (tc,rc); turn 6 × 1 tc;
    // flop 6. Reach passes: flop 6, turn 6, river 6.
    let terminal_total = 6.0 * (zone_terminal_s(&census_f) + zone_terminal_s(&census_t) + zone_terminal_s(&census_r));
    let reach_total = 18.0 * reach_s; // flop+turn+river reach per traverser, ~equal cost
    let projected = terminal_total + reach_total;
    eprintln!("\nprojected iteration: terminals ≈ {:.0}s + per-hand reach ≈ {:.0}s ≈ {:.0}s total",
        terminal_total, reach_total, projected);
    eprintln!("  vs M4 prediction {:.1}s → projected ratio ≈ {:.1}×",
        m4_pred_ms / 1000.0, projected / (m4_pred_ms / 1000.0));

    if projected > PROJECTION_BUDGET_S {
        eprintln!("\n⛔ STOP-AND-REPORT: projection {:.0}s exceeds budget {:.0}s — \
                   full iteration NOT run; the projection is the finding. \
                   Design fork triggered before sweep.", projected, PROJECTION_BUDGET_S);
        return;
    }

    // ── Full bucketed iteration with per-stage timers ──
    eprintln!("\n── full bucketed iteration (per-stage) ──");
    let np = NP as usize;
    let nn = tree.num_nodes();
    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();
    let params = DcfrParams::new(0);

    let mut t_strategy = 0.0f64;
    let mut t_reach = 0.0f64;
    let mut t_buz_river = 0.0f64;
    let mut t_buz_turn = 0.0f64;
    let mut t_buz_flop = 0.0f64;
    let mut t_chance = 0.0f64;

    let mut flop_cfv = vec![0.0f32; nn * nh];
    let mut river_cfv_accum = vec![0.0f32; nn * nh];
    let mut cfv = vec![0.0f32; nn * nh];
    let mut turn_cfv = vec![0.0f32; nn * nh];

    let iter_t0 = Instant::now();
    for traverser in 0..np {
        let t = Instant::now();
        bucketed.compute_flop_strategy(&tree, &bk);
        t_strategy += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let flop_reach = bucketed.compute_reach_flop(&tree, &game, &bk);
        t_reach += t.elapsed().as_secs_f64();

        for &child_id in zones_probe.turn_chance_children() {
            let off = child_id as usize * nh;
            for h in 0..nh { flop_cfv[off + h] = 0.0; }
        }

        for (ti, &tc) in turn_deck.iter().enumerate() {
            let t = Instant::now();
            bucketed.compute_turn_strategy_for_tc(&tree, &bk, ti);
            t_strategy += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let turn_reach = bucketed.compute_reach_turn(&tree, &bk, ti, &flop_reach);
            t_reach += t.elapsed().as_secs_f64();
            let n_river = table_ref.river_decks[tc as usize].len();

            for &child_id in zones_probe.river_chance_children() {
                let off = child_id as usize * nh;
                for h in 0..nh { river_cfv_accum[off + h] = 0.0; }
            }

            for ri in 0..n_river {
                let t = Instant::now();
                bucketed.compute_river_strategy_for_pair(&tree, &bk, ti, ri);
                t_strategy += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let river_reach = bucketed.compute_reach_river(&tree, &bk, ti, ri, &turn_reach);
                t_reach += t.elapsed().as_secs_f64();

                let t = Instant::now();
                bucketed.bottom_up_zone(
                    &tree, table_ref, &bk, traverser as u8, &river_reach, &mut cfv,
                    Zone::River, Some(ti), Some(ri), &params,
                );
                t_buz_river += t.elapsed().as_secs_f64();

                let t = Instant::now();
                for &child_id in zones_probe.river_chance_children() {
                    for h in 0..nh {
                        let cp = table_ref.chance_probability_river(tc, ri, h);
                        river_cfv_accum[child_id as usize * nh + h] +=
                            cp * cfv[child_id as usize * nh + h];
                    }
                }
                t_chance += t.elapsed().as_secs_f64();
            }

            for &child_id in zones_probe.river_chance_children() {
                for h in 0..nh {
                    turn_cfv[child_id as usize * nh + h] =
                        river_cfv_accum[child_id as usize * nh + h];
                }
            }

            let t = Instant::now();
            bucketed.bottom_up_zone(
                &tree, table_ref, &bk, traverser as u8, &turn_reach, &mut turn_cfv,
                Zone::Turn, Some(ti), None, &params,
            );
            t_buz_turn += t.elapsed().as_secs_f64();

            let t = Instant::now();
            for &child_id in zones_probe.turn_chance_children() {
                for h in 0..nh {
                    let cp = table_ref.chance_probability_turn(ti, h);
                    flop_cfv[child_id as usize * nh + h] +=
                        cp * turn_cfv[child_id as usize * nh + h];
                }
            }
            t_chance += t.elapsed().as_secs_f64();
        }

        let t = Instant::now();
        bucketed.bottom_up_zone(
            &tree, table_ref, &bk, traverser as u8, &flop_reach, &mut flop_cfv,
            Zone::Flop, None, None, &params,
        );
        t_buz_flop += t.elapsed().as_secs_f64();
        eprintln!("  traverser {traverser} done at {:.0}s", iter_t0.elapsed().as_secs_f64());
    }
    let iter_s = iter_t0.elapsed().as_secs_f64();

    // Terminal share (arithmetic attribution from probes × census).
    let term_river = 6.0 * zone_terminal_s(&census_r);
    let term_turn = 6.0 * zone_terminal_s(&census_t);
    let term_flop = 6.0 * zone_terminal_s(&census_f);

    eprintln!("\n══ measured bucketed iteration at nh={nh}, B={B}: {:.1}s ══", iter_s);
    eprintln!("  strategy (bucket-granular):       {:>8.2}s", t_strategy);
    eprintln!("  reach (PER-HAND, nh={nh}):        {:>8.2}s", t_reach);
    eprintln!("  bottom_up river:                  {:>8.2}s  (terminals ≈ {:.1}s by probe attribution)", t_buz_river, term_river);
    eprintln!("  bottom_up turn:                   {:>8.2}s  (terminals ≈ {:.1}s)", t_buz_turn, term_turn);
    eprintln!("  bottom_up flop:                   {:>8.2}s  (terminals ≈ {:.1}s)", t_buz_flop, term_flop);
    eprintln!("  chance accumulation (per-hand):   {:>8.2}s", t_chance);

    // ── Reference: exact CPU at nh = B (the everything-at-B game) ──
    eprintln!("\n── reference: exact CPU iteration at nh=B={B} (same tree, same device) ──");
    let small_game = FlopStartGame::new(build_m2_table(B));
    let mut exact_small = FlopStartVectorCfr::new(&tree, small_game.table());
    let t0 = Instant::now();
    exact_small.run(&tree, &small_game, 1);
    let exact_b_s = t0.elapsed().as_secs_f64();
    eprintln!("exact CPU at nh={B}: {:.2}s/iter", exact_b_s);

    let ratio_devicefree = iter_s / exact_b_s;
    let ratio_formula = iter_s / (m4_pred_ms / 1000.0);
    eprintln!("\n══ VERDICT INPUTS ══");
    eprintln!("  hybrid / everything-at-B (device-free): {:.1}×", ratio_devicefree);
    eprintln!("  hybrid / M4 GPU formula at B={B}:        {:.1}×", ratio_formula);
    eprintln!("  (decision rule: > ~2× on the device-free ratio triggers the design fork)");
}

/// Design-2 production cost: full bucketed iteration at nh=1176, B=15
/// with the factored terminal, vs the M4 everything-at-B prediction
/// (7.0 s/iter) — including the explicit GPU-port-mootness check: if
/// the CPU-only Design-2 iteration lands at or under the prediction,
/// the carried GPU-bucketed-port item may be DELETABLE (removing an
/// entire desync-surface class and its parity gates), not deferred.
///
/// NOTE: Design 2's equilibrium-quality gate is OPEN at this writing
/// (A/B failures, see p1_5_4_bucketing_b4_design2_equilibrium_ab) —
/// this measurement prices the candidate, it does not bless it.
///
/// ═══ MEASURED 2026-06-10 ═══
///   Design-2 unit probes (B=15): arm-1 63.8 µs, arm-2 5-active
///   137.3 µs, arm-2 4-folded 44.4 µs (vs Design 1's 0.128 s /
///   0.716 s / 0.288 s — 3-4 orders of magnitude).
///   Full CPU iteration (nh=1176, B=15, M2 tree): 0.30 s
///   vs M4 prediction 7.0 s → 0.04× — 23× UNDER, CPU-only.
///   GPU-mootness: HOLDS decisively on this tree. To be re-confirmed
///   at the production tree shape before deleting the port item.
#[test]
#[ignore = "B4 Design-2 cost measurement (~minutes); run with --ignored --nocapture"]
fn b4_cost_measurement_design2() {
    let m4_pred_ms = 335.0 * (B as f64 / 8.0).powf(4.84);
    eprintln!("\n════ B4: Design-2 (factored) cost at production nh ════");
    eprintln!("M4 prediction at B={B}: {:.1} s/iter", m4_pred_ms / 1000.0);

    let tree = build_m2_tree();
    let table = build_m2_table(usize::MAX);
    let nh = table.num_valid;
    let (fm, tm, rm) = quantile_maps(&table, B);
    let game = FlopStartGame::new(table);
    let bk = FlopBucketing::from_maps(game.table(), B, B, B, fm, tm, rm);
    let mut bucketed = BucketedFlopCfr::new(&tree, game.table(), &bk);
    bucketed.set_terminal_design(TerminalDesign::Design2Factored);

    // Unit probes (same scenarios as the Design-1 gate).
    let uniform_reach: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0 / B as f32; B]).collect();
    let views: Vec<&[f32]> = uniform_reach.iter().map(|v| v.as_slice()).collect();
    for (name, contribs, fold_mask) in [
        ("arm-1 (equal, no folds)", [20i32; 6].to_vec(), 0u16),
        ("arm-2 (unequal, 5 active)", vec![10, 20, 30, 30, 20, 10], 0),
        ("arm-2 (4 folded, 1 active)", vec![10, 20, 5, 5, 5, 5], 0b111100),
    ] {
        // Warm + time over many reps (single call is sub-µs-noisy).
        let reps = 200u32;
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = bucketed_showdown_cfv_factored(
                &views, &bk.flop_tables, &contribs, fold_mask, 0, NP, 30, 0.0, 0.0, true,
            );
        }
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        eprintln!("D2 {name}: {:.1} µs/terminal", per * 1e6);
    }

    // Full iteration, timed.
    let t0 = Instant::now();
    bucketed.run(&tree, &game, &bk, 1);
    let iter_s = t0.elapsed().as_secs_f64();
    eprintln!("\n══ Design-2 full CPU iteration at nh={nh}, B={B}: {:.2}s ══", iter_s);
    eprintln!("  vs M4 prediction {:.1}s → {:.2}×", m4_pred_ms / 1000.0, iter_s / (m4_pred_ms / 1000.0));
    eprintln!("  GPU-mootness check: CPU-only {} the everything-at-B prediction on this tree",
        if iter_s <= m4_pred_ms / 1000.0 { "is AT/UNDER" } else { "EXCEEDS" });
}

/// Design-1-COLLAPSED production cost ladder + the B4 wall-clock
/// projection, with the GPU-port-vs-flop-sampling fork priced
/// EXPLICITLY (not assumed). All numbers below are measured on this
/// machine (M4 Max, single core); the projection arithmetic is printed
/// so the fork can be re-priced as inputs change.
///
/// ═══ MEASURED 2026-06-10 (M2 tree shape: 109 terminals, 1×1 sampled
/// chance — the SAME shape the M4 formula was fit on; nh=1176, single
/// core, Design1Collapsed) ═══
///
///   B  | s/iter   | 34 iters (1% pot) ×1755 flops ÷16 | ×100 flops
///    5 |    0.85  |   0.88 h                          |  0.05 h
///    8 |   10.14  |  10.50 h  ← fits 24h CPU-only     |  0.60 h
///   10 |   37.94  |  39.30 h  ← does NOT fit          |  2.24 h
///   15 |  399.66  | 414 h                             | 23.6 h
///   20 | 2195.21  | 2274 h                            | 130 h
///   25 | 8453.34  | 8757 h                            | 499 h
///   (0.1%-pot target = 158 iters = ×4.65: B=8 full-set 48.8h does
///    NOT fit; B=8 @100 flops 2.8h fits; B=5 full-set 4.1h fits.)
///
/// The 18× line item: the unpinned B=10 guess was 2.1 s/iter; the
/// measurement says 37.94 — the multiplication beat the estimate
/// twice this arc, which is why no rung was extrapolated.
///
/// ═══ END-TO-END BLUEPRINT PROJECTION (the 24h budget governs the
/// whole loop, not the per-flop solve) ═══
///   1. Postflop solves: the table above — the dominant line.
///   2. Clustering: 36.4 s/flop single-core (B2 measured), pipelined
///      against solves: hidden whenever 34·iter_s ≥ 36.4s (true for
///      B ≥ 8; at B=5 clustering IS the bottleneck, adding ≤ 0.4h at
///      1755/16). Parallel-contention wall-clock UNMEASURED — the one
///      open cost flag on this line.
///   3. W/T/L precompute: 0.1 s/flop (B2 measured) → ~11 s total at
///      1755/16. Negligible.
///   4. Preflop layer: frozen-CFV oracle design means postflop solves
///      are NOT multiplied by preflop iterations (cached per flop ×
///      traverser); the preflop walk + pairwise terminals add a line
///      that is measured only at research scale — production preflop
///      wall-clock is the named UNPRICED line item.
///   5. Runout-sampling caveat: rows are per-flop at the M2 1×1
///      sampled-chance shape (same as the M4 formula). The blueprint's
///      runout-sampling policy multiplies the postflop line linearly
///      in sampled (n_turn × n_river) — a design input, priced when
///      chosen, not assumed here.
///
/// ═══ FORK PRICING (signed-off wording) ═══
///   GPU bucketed port: MOOT at B=8 full-canonical/1%-pot (10.5h CPU);
///   the UNLOCK for B=10+ at full canonicals (39.3h → est. ~8h at an
///   unmeasured ~5×) or for 0.1%-pot targets; one of two priced
///   options against flop sampling (100-flop set: B=10 2.2h, B=15
///   23.6h — sampling's quality cost is measurable by the harness).
///   B ≥ 20 is calibration-only territory on any path.
#[test]
#[ignore = "B4 collapsed cost ladder (~10 min at the B=25 point); run with --ignored --nocapture"]
fn b4_cost_ladder_design1_collapsed() {
    use solver_core::solver::bucketed_showdown::bucketed_showdown_cfv_design1_collapsed;

    eprintln!("\n════ B4: Design-1-collapsed cost ladder at production nh ════");
    let tree = build_m2_tree();
    let table = build_m2_table(usize::MAX);
    let nh = table.num_valid;
    let game = FlopStartGame::new(table);

    let mut iter_costs: Vec<(usize, f64)> = Vec::new();
    for b in [5usize, 8, 10, 15, 20, 25] {
        let (fm, tm, rm) = quantile_maps(game.table(), b);
        let bk = FlopBucketing::from_maps(game.table(), b, b, b, fm, tm, rm);

        // Per-terminal probe (arm-2 worst case: unequal, 5 active).
        let ur: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0 / b as f32; b]).collect();
        let views: Vec<&[f32]> = ur.iter().map(|v| v.as_slice()).collect();
        let reps = if b <= 10 { 50u32 } else { 5 };
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = bucketed_showdown_cfv_design1_collapsed(
                &views, &bk.flop_tables, &[10, 20, 30, 30, 20, 10], 0, 0, NP, 30, 0.0, 0.0,
                true,
            );
        }
        let per_term = t0.elapsed().as_secs_f64() / reps as f64;

        let mut bucketed = BucketedFlopCfr::new(&tree, game.table(), &bk);
        bucketed.set_terminal_design(TerminalDesign::Design1Collapsed);
        let t0 = Instant::now();
        bucketed.run(&tree, &game, &bk, 1);
        let iter_s = t0.elapsed().as_secs_f64();
        iter_costs.push((b, iter_s));
        eprintln!("B={b:>2}: iter {iter_s:>8.2}s | arm-2 terminal {:.2}ms", per_term * 1e3);
    }

    eprintln!("\n── projection: 34 iters (1% pot target), 16-way flop parallelism ──");
    eprintln!("{:>4} | {:>14} | {:>14}", "B", "1755 flops", "100 flops");
    for &(b, s) in &iter_costs {
        let full = s * 34.0 * 1755.0 / 16.0 / 3600.0;
        let sampled = s * 34.0 * 100.0 / 16.0 / 3600.0;
        eprintln!("{b:>4} | {full:>12.2} h | {sampled:>12.2} h");
    }
    eprintln!("\nFork pricing (24h budget): GPU port needed only where the CPU");
    eprintln!("row exceeds 24h at the chosen flop set; flop sampling trades a");
    eprintln!("measurable quality cost (harness can score sampled-set configs).");
    eprintln!("Clustering adds 36.4s/flop single-core (B2 measurement), fully");
    eprintln!("pipelineable against solves (~17.8h/1755 unpipelined, ~1h/100).");
}
