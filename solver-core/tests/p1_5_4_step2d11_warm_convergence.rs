// Step 2.D.11: warm-K convergence vs converged reference.
//
// PER USER (banked 2026-06-07): warm-start COST is measured (50× at K=1)
// but warm-start VALIDITY at small K is unmeasured. The actual lever size
// is determined by smallest K where warm converges to a CONVERGED
// reference at acceptable tolerance. Cost framing has a free parameter
// (which K) that convergence determines.
//
// THIS TEST:
//   REFERENCE: cold K=large (converged oracle). Blueprint loop's final
//     preflop state = ground truth.
//   VARIANTS:  cold K=50 + warm K∈{1, 5, 10}.
//   COMPARE:   state deviation (regrets / cum_strategy / strategy) from
//     converged reference. Smallest warm K with deviation ≤ cold K=50's
//     deviation = the lever size for #96.
//
// DISCIPLINE: anchored against the converged reference, NOT self-consistency.
//
// SUBSET (for tractability):
//   - 10 canonicals subset (instead of 1755) — convergence dynamics scale
//     similarly per-canonical; this measures whether warm catches up to
//     cold per-canonical, which is the load-bearing property.
//   - Postflop subset: 8 hands × 2 turn × 2 river (matches 2.D.6 / 2.D.7).
//   - 5 preflop iters.

use std::collections::HashMap;
use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::solver::preflop_start_game::{
    aggregate_preflop_chance_subset, expand_reach_class_to_combo,
    flop_combo_layout, reduce_cfv_combo_to_class, PreflopChanceTable,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const PREFLOP_ITERS: u32 = 5;
const CANONICAL_SUBSET_SIZE: usize = 10;

fn build_minimal_hu_preflop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![20, 19],
        initial_contributions: vec![1, 2],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&cfg).expect("preflop tree builds")
}

fn build_tiny_flop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 4,
        starting_stacks: vec![10, 10],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&cfg).expect("flop tree builds")
}

fn pick_subset(canonical: [Card; 3]) -> (Vec<u16>, Vec<u8>, Vec<Vec<u8>>) {
    let board_mask: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    // CRITICAL: pick non-mutually-blocking hands. Naive "first 8 non-blocking by
    // index" picks pairs like (3,4), (3,5), (3,6)... all sharing card 3 — these
    // are all mutually blocking → opp_reach[g] = 0 for all g != h in subset →
    // showdown CFV = 0 by construction. Instead, walk pairs and accept only if
    // it doesn't conflict with any previously chosen hand.
    let mut chosen: Vec<u16> = Vec::new();
    let mut used_cards: u64 = board_mask;
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = solver_core::card::index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        if used_cards & (1u64 << c1) != 0 || used_cards & (1u64 << c2) != 0 { continue; }
        chosen.push(idx as u16);
        used_cards |= 1u64 << c1;
        used_cards |= 1u64 << c2;
        if chosen.len() == 8 { break; }
    }
    let mut hand_mask = board_mask;
    for &i in &chosen {
        let (c1, c2) = solver_core::card::index_to_card_pair(i as usize);
        hand_mask |= 1u64 << c1;
        hand_mask |= 1u64 << c2;
    }
    let mut turn_cards: Vec<u8> = Vec::new();
    for c in 0u8..52u8 {
        if hand_mask & (1u64 << c) != 0 { continue; }
        turn_cards.push(c);
        if turn_cards.len() == 2 { break; }
    }
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let mut rivers: Vec<u8> = Vec::new();
        for c in 0u8..52u8 {
            if hand_mask & (1u64 << c) != 0 { continue; }
            if c == tc { continue; }
            rivers.push(c);
            if rivers.len() == 2 { break; }
        }
        river_decks[tc as usize] = rivers;
    }
    (chosen, turn_cards, river_decks)
}

struct WarmState {
    regrets_flop: Vec<f32>,
    regrets_turn: Vec<f32>,
    regrets_river: Vec<f32>,
    cum_strategy_flop: Vec<f32>,
    cum_strategy_turn: Vec<f32>,
    cum_strategy_river: Vec<f32>,
    iteration: u32,
}

/// Per-canonical postflop subsolve. If `state_map` already has an entry
/// for (canonical, traverser), load it into the solver before run() —
/// that's the warm-start mechanic. Per the cold/warm switch:
///   - Cold: caller passes a state_map that's cleared before each call.
///   - Warm: caller passes a state_map that persists across calls.
fn flop_root_cfv_subset(
    flop_tree: &FlatTree,
    canonical: [Card; 3],
    combo_ranges_per_player: &[Vec<f32>],
    traverser: u8,
    iters_per_call: u32,
    state_map: &mut HashMap<([Card; 3], u8), WarmState>,
) -> Vec<f32> {
    let np = combo_ranges_per_player.len();
    assert!(np >= 2);

    let layout_engine = flop_combo_layout(canonical);
    let mut combo_ranges_full: Vec<Vec<f32>> =
        vec![vec![0.0f32; NUM_POSSIBLE_HANDS]; np];
    for p in 0..np {
        assert_eq!(combo_ranges_per_player[p].len(), layout_engine.len());
        for (li, &(c1, c2)) in layout_engine.iter().enumerate() {
            combo_ranges_full[p][card_pair_to_index(c1, c2)] = combo_ranges_per_player[p][li];
        }
    }

    let board: Vec<Card> = canonical.iter().copied().collect();
    let (chosen, turn_cards, river_decks) = pick_subset(canonical);
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &combo_ranges_full, np as u8, &chosen, &turn_cards, &river_decks,
    );
    let nh = table.num_valid;
    let layout_table: Vec<(Card, Card)> = (0..nh)
        .map(|i| (table.hand_cards[i * 2], table.hand_cards[i * 2 + 1]))
        .collect();
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(flop_tree, game.table());

    let key = (canonical, traverser);
    if let Some(state) = state_map.get(&key) {
        let rf = solver.regrets_flop_mut();
        if rf.len() == state.regrets_flop.len() {
            rf.copy_from_slice(&state.regrets_flop);
        }
        let rt = solver.regrets_turn_mut();
        if rt.len() == state.regrets_turn.len() {
            rt.copy_from_slice(&state.regrets_turn);
        }
        let rr = solver.regrets_river_mut();
        if rr.len() == state.regrets_river.len() {
            rr.copy_from_slice(&state.regrets_river);
        }
        let cf = solver.cum_strategy_flop_mut();
        if cf.len() == state.cum_strategy_flop.len() {
            cf.copy_from_slice(&state.cum_strategy_flop);
        }
        let ct = solver.cum_strategy_turn_mut();
        if ct.len() == state.cum_strategy_turn.len() {
            ct.copy_from_slice(&state.cum_strategy_turn);
        }
        let cr = solver.cum_strategy_river_mut();
        if cr.len() == state.cum_strategy_river.len() {
            cr.copy_from_slice(&state.cum_strategy_river);
        }
        solver.set_iteration(state.iteration);
    }

    let _ = solver.run(flop_tree, &game, iters_per_call);

    // Save state BEFORE the freeze+extract pass (freeze mutates strategy
    // buffers; we want pure regret/cum_strategy state preserved for the
    // next warm call).
    let saved = WarmState {
        regrets_flop: solver.regrets_flop().to_vec(),
        regrets_turn: solver.regrets_turn().to_vec(),
        regrets_river: solver.regrets_river().to_vec(),
        cum_strategy_flop: solver.cum_strategy_flop().to_vec(),
        cum_strategy_turn: solver.cum_strategy_turn().to_vec(),
        cum_strategy_river: solver.cum_strategy_river().to_vec(),
        iteration: solver.iteration_count(),
    };
    state_map.insert(key, saved);

    // BUGFIX 2026-06-07 (#107 reopen + #105 fix): the prior extraction used
    // single flop_reach for all bottom_up_zone calls, which silently zeroed
    // the showdown contribution at turn/river terminals. Now mirrors
    // compute_v_flop_at_root_converged's fixed shape: per-zone reach
    // (compute_reach_turn per tc, compute_reach_river per (tc, ri)) +
    // chance-prob weighted bubble-up via separate accumulator buffers.
    //
    // Impact: warm-K convergence numbers from the pre-fix runs were measured
    // against a buggy converged reference AND with buggy warm-K extractions.
    // This re-run produces the corrected measurements.
    solver.freeze_average_strategy_flop(flop_tree);
    let flop_reach = solver.compute_reach_flop(flop_tree, &game);
    let nn = flop_tree.num_nodes();
    let mut cfv = vec![0.0f32; nn * nh];
    let mut river_cfv_accum = vec![0.0f32; nn * nh];
    let mut turn_cfv = vec![0.0f32; nn * nh];
    let mut flop_cfv = vec![0.0f32; nn * nh];
    let params = DcfrParams::new(0);
    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();

    for &child_id in solver.turn_chance_children() {
        let off = child_id as usize * nh;
        for h in 0..nh { flop_cfv[off + h] = 0.0; }
    }

    for (ti, &tc_card) in turn_deck.iter().enumerate() {
        solver.freeze_average_strategy_for_turn(flop_tree, ti);
        let turn_reach = solver.compute_reach_turn(flop_tree, ti, &flop_reach);
        let river_deck = &table_ref.river_decks[tc_card as usize];

        for &child_id in solver.river_chance_children() {
            let off = child_id as usize * nh;
            for h in 0..nh { river_cfv_accum[off + h] = 0.0; }
        }

        for ri in 0..river_deck.len() {
            solver.load_river_pair(ti, ri).unwrap();
            solver.freeze_average_strategy_for_river_pair(flop_tree, ti, ri);
            let river_reach = solver.compute_reach_river(flop_tree, ti, ri, &turn_reach);
            solver.bottom_up_zone(
                flop_tree, table_ref, traverser, &river_reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params,
            );
            solver.save_river_pair(ti, ri).unwrap();

            for &child_id in solver.river_chance_children() {
                for h in 0..nh {
                    let cp = table_ref.chance_probability_river(tc_card, ri, h);
                    river_cfv_accum[child_id as usize * nh + h] +=
                        cp * cfv[child_id as usize * nh + h];
                }
            }
        }

        for &child_id in solver.river_chance_children() {
            for h in 0..nh {
                turn_cfv[child_id as usize * nh + h] =
                    river_cfv_accum[child_id as usize * nh + h];
            }
        }

        solver.bottom_up_zone(
            flop_tree, table_ref, traverser, &turn_reach, &mut turn_cfv,
            Zone::Turn, Some(ti), None, &params,
        );

        for &child_id in solver.turn_chance_children() {
            for h in 0..nh {
                let cp = table_ref.chance_probability_turn(ti, h);
                flop_cfv[child_id as usize * nh + h] +=
                    cp * turn_cfv[child_id as usize * nh + h];
            }
        }
    }

    for &child_id in solver.turn_chance_children() {
        for h in 0..nh {
            cfv[child_id as usize * nh + h] = flop_cfv[child_id as usize * nh + h];
        }
    }

    solver.bottom_up_zone(
        flop_tree, table_ref, traverser, &flop_reach, &mut cfv,
        Zone::Flop, None, None, &params,
    );

    let v_table = cfv[0..nh].to_vec();
    let mut v_engine = vec![0.0f32; layout_engine.len()];
    for (li, &combo) in layout_engine.iter().enumerate() {
        if let Some(pos) = layout_table.iter().position(|&c| c == combo) {
            v_engine[li] = v_table[pos];
        }
    }
    v_engine
}

/// One preflop iter using canonical_subset (matches 2.D.6 pattern).
fn run_one_preflop_iter_subset<F: FnMut([Card; 3], &[Vec<f32>], u8) -> Vec<f32>>(
    tree: &FlatTree,
    table: &PreflopChanceTable,
    chance_node_indices: &[usize],
    canonical_subset: &[usize],
    np: usize,
    n_classes: usize,
    iter: u32,
    terminal_value_fn: &dyn Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
    solver: &mut PreflopVectorCfr,
    mut per_canonical: F,
) {
    let nn = tree.num_nodes();
    solver.compute_preflop_strategy(tree);
    let reach = solver.compute_preflop_reach(tree, None);
    let params = DcfrParams::new(iter);
    for t in 0..np as u8 {
        let mut cfv: Vec<Vec<f32>> = vec![vec![0.0f32; n_classes]; nn];
        for &chance_idx in chance_node_indices {
            let chance_base = chance_idx * n_classes;
            let mut per_canon_v: Vec<Vec<f32>> = Vec::with_capacity(canonical_subset.len());
            for &canonical_idx in canonical_subset {
                let f_canon = table.canonical_flops[canonical_idx];
                let layout = flop_combo_layout(f_canon);
                let mut combo_reaches: Vec<Vec<f32>> = Vec::with_capacity(np);
                for p in 0..np {
                    let class_reach = &reach[p][chance_base..chance_base + n_classes];
                    combo_reaches.push(expand_reach_class_to_combo(f_canon, class_reach, &layout));
                }
                let v_combo = per_canonical(f_canon, &combo_reaches, t);
                let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
                per_canon_v.push(v_class);
            }
            cfv[chance_idx] = aggregate_preflop_chance_subset(table, canonical_subset, &per_canon_v);
        }
        solver.bottom_up_preflop_for_traverser(
            tree, t, chance_node_indices, &reach,
            |term_idx, tr, r| terminal_value_fn(term_idx, tr, r),
            &mut cfv, &params,
        );
    }
}

fn max_abs_and_rms(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    let n = a.len().min(b.len());
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > max_abs { max_abs = d; }
        sum_sq += (d as f64) * (d as f64);
    }
    let rms = if n > 0 { (sum_sq / n as f64).sqrt() as f32 } else { 0.0 };
    (max_abs, rms)
}

#[test]
#[ignore = "Step 2.D.11: warm-K convergence measurement"]
fn step2d11_warm_k_convergence_vs_converged_reference() {
    let tree = build_minimal_hu_preflop_tree();
    let flop_tree = build_tiny_flop_tree();
    let np = tree.num_players as usize;
    eprintln!("\n=== Step 2.D.11: warm-K convergence vs converged reference ===");
    eprintln!("Preflop tree: {} nodes, flop tree: {} nodes", tree.num_nodes(), flop_tree.num_nodes());
    eprintln!("Subset: {} canonicals, 8 hands × 2 turn × 2 river per canonical.",
        CANONICAL_SUBSET_SIZE);
    eprintln!("Blueprint loop: {} preflop iters per variant.", PREFLOP_ITERS);

    let mut class_weights: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();
    for k in 0..NUM_PREFLOP_CLASSES {
        let s = k as f32 / NUM_PREFLOP_CLASSES as f32;
        class_weights[0][k] = ((s - 0.3).max(0.05) * 1.5).min(1.0);
        class_weights[1][k] = 0.6 + 0.4 * s;
    }
    eprintln!("Building PreflopChanceTable...");
    let table = PreflopChanceTable::new(np as u8, class_weights);
    let canonical_subset: Vec<usize> = (0..CANONICAL_SUBSET_SIZE).collect();

    let terminal_value_fn = |term_idx: usize, traverser: u8, _r: &[Vec<f32>]| -> Vec<f32> {
        (0..NUM_PREFLOP_CLASSES).map(|c| {
            let seed: u64 = (term_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (traverser as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                ^ (c as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            let bits = ((seed >> 32) & 0xFFFFFF) as i64 - (1 << 23);
            (bits as f32) / ((1 << 23) as f32)
        }).collect()
    };
    let n_classes = NUM_PREFLOP_CLASSES;

    // Variants to measure: (label, iters_per_call, warm_persistent)
    // POST #107 RE-OPEN: warm K=5 produced anomalous 19× cold K=50 noise floor
    // deviation while K=1=6× and K=10=4× — non-monotonic spike at K=5 with no
    // obvious mechanism. Filed as #110 candidate artifact. Finer K sweep to
    // localize whether the spike is sharp at K=5 or part of a broader hump.
    let variants = vec![
        ("ref cold K=200", 200u32, false),
        ("cold K=50",      50u32,  false),
        ("warm K=1",        1u32,  true),
        ("warm K=2",        2u32,  true),
        ("warm K=3",        3u32,  true),
        ("warm K=4",        4u32,  true),
        ("warm K=5",        5u32,  true),
        ("warm K=6",        6u32,  true),
        ("warm K=7",        7u32,  true),
        ("warm K=8",        8u32,  true),
        ("warm K=9",        9u32,  true),
        ("warm K=10",      10u32,  true),
        ("warm K=12",      12u32,  true),
        ("warm K=14",      14u32,  true),
        ("warm K=16",      16u32,  true),
        ("warm K=20",      20u32,  true),
    ];

    let mut results: Vec<(String, f64, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();

    for (label, iters_per_call, warm) in &variants {
        eprintln!("\n── {} (iters/call = {}, warm = {}) ──", label, iters_per_call, warm);
        let mut prod = PreflopVectorCfr::new(&tree);
        let chance_node_indices = prod.preflop_chance_node_indices(&tree);
        let mut state_map: HashMap<([Card; 3], u8), WarmState> = HashMap::new();

        let t = Instant::now();
        for iter in 0..PREFLOP_ITERS {
            if !*warm { state_map.clear(); }  // cold mode: drop state between preflop iters
            let iters = *iters_per_call;
            let ft = &flop_tree;
            run_one_preflop_iter_subset(
                &tree, &table, &chance_node_indices, &canonical_subset,
                np, n_classes, iter, &terminal_value_fn, &mut prod,
                |canonical, ranges, traverser| {
                    // For COLD: clear state before EACH call. For WARM: leave it.
                    if !*warm { state_map.clear(); }
                    flop_root_cfv_subset(ft, canonical, ranges, traverser, iters, &mut state_map)
                },
            );
        }
        let secs = t.elapsed().as_secs_f64();
        eprintln!("  done in {:.1} s", secs);
        results.push((label.to_string(), secs, prod.regrets.clone(),
            prod.strategy.clone(), prod.cum_strategy.clone()));
    }

    // Reference = first variant (cold K=200).
    let (_ref_label, ref_secs, ref_regrets, ref_strategy, ref_cum) = &results[0];

    eprintln!("\n=== Deviation from REFERENCE (cold K=200) ===");
    eprintln!("{:>16}  {:>10}  {:>20}  {:>20}  {:>20}",
        "variant", "wall (s)", "regrets max/rms", "strategy max/rms", "cum_strat max/rms");
    eprintln!("{:>16}  {:>10.1}  {:>20}  {:>20}  {:>20}",
        "ref cold K=200", ref_secs, "(reference)", "(reference)", "(reference)");

    let mut deviation_table: Vec<(String, f32, f32, f32, f32, f32, f32)> = Vec::new();
    for (label, secs, r, s, c) in results.iter().skip(1) {
        let (r_max, r_rms) = max_abs_and_rms(ref_regrets, r);
        let (s_max, s_rms) = max_abs_and_rms(ref_strategy, s);
        let (c_max, c_rms) = max_abs_and_rms(ref_cum, c);
        eprintln!("{:>16}  {:>10.1}  {:>9.3e}/{:>9.3e}  {:>9.3e}/{:>9.3e}  {:>9.3e}/{:>9.3e}",
            label, secs, r_max, r_rms, s_max, s_rms, c_max, c_rms);
        deviation_table.push((label.clone(), r_max, r_rms, s_max, s_rms, c_max, c_rms));
    }

    // ── Interpretation ──
    eprintln!("\n=== Interpretation ===");
    // Find cold K=50 deviation as the "production-cold noise floor".
    let cold_k50 = &deviation_table[0]; // first in iter, which is cold K=50
    let cold_k50_strategy_max = cold_k50.3;
    eprintln!("Cold K=50 strategy deviation from converged ref: {:.3e}", cold_k50_strategy_max);
    eprintln!("(this is the noise floor the production cold-K=50 oracle already accepts)");
    eprintln!();
    for (label, _r_max, _r_rms, s_max, _s_rms, _c_max, _c_rms) in &deviation_table[1..] {
        let factor = s_max / cold_k50_strategy_max.max(1e-12);
        let verdict = if factor <= 1.0 {
            "≤ cold K=50 noise floor → lever HOLDS at this K"
        } else if factor <= 3.0 {
            "moderately above floor → marginal lever at this K"
        } else {
            "significantly above floor → lever DOES NOT HOLD at this K; need larger K"
        };
        eprintln!("  {}: strategy_max = {:.3e} ({:.2}× cold K=50). {}",
            label, s_max, factor, verdict);
    }

    eprintln!("\nThe smallest warm K within the cold-K=50 noise floor = the lever for #96.");
    eprintln!("(Bigger lever = less abstraction compression required.)");
    eprintln!();
    eprintln!("CAVEATS:");
    eprintln!("- Subset config (10 canonicals, 8 hands × 4 pairs subset, 5 preflop iters).");
    eprintln!("  Convergence dynamics at full 1755 canonicals × full deck × more preflop iters");
    eprintln!("  could differ — drift across preflop iters interacts with warm-start.");
    eprintln!("- Cold K=200 is the reference. If cold K=200 is itself under-converged (deviation");
    eprintln!("  from cold K=500 noticeable), the noise-floor calibration shifts. Future follow-up.");
}

// ─────────────────────────────────────────────────────────────────────
// Step 2.D.20 (#107 follow-up): variance-vs-preflop-iter-count.
//
// The K-sweep at preflop_iters=5 showed a wide warm-K spread driven by
// DCFR-reset-vs-call-boundary alignment (#110 trace). The hypothesis: at
// higher preflop iter counts, DCFR resets are rarer (t=4, 16, 64, 256, ...)
// AND the alignment-variance averages out across many calls. If the
// best-K-vs-worst-K spread SHRINKS as preflop iters grow, the production
// regime is favorable for warm-start (lever real, bucketing relaxes for
// #96). If spread persists, cold baseline carries.
//
// This is the test #96's cost denominator is gated on.
// ─────────────────────────────────────────────────────────────────────

fn run_variant_with_iters(
    tree: &FlatTree,
    flop_tree: &FlatTree,
    table: &PreflopChanceTable,
    canonical_subset: &[usize],
    terminal_value_fn: &dyn Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
    iters_per_call: u32,
    warm: bool,
    preflop_iters: u32,
) -> (f64, Vec<f32>, Vec<f32>) {
    let np = tree.num_players as usize;
    let n_classes = NUM_PREFLOP_CLASSES;
    let mut prod = PreflopVectorCfr::new(tree);
    let chance_node_indices = prod.preflop_chance_node_indices(tree);
    let mut state_map: HashMap<([Card; 3], u8), WarmState> = HashMap::new();
    let t = Instant::now();
    for iter in 0..preflop_iters {
        if !warm { state_map.clear(); }
        let iters = iters_per_call;
        let ft = flop_tree;
        run_one_preflop_iter_subset(
            tree, table, &chance_node_indices, canonical_subset,
            np, n_classes, iter, terminal_value_fn, &mut prod,
            |canonical, ranges, traverser| {
                if !warm { state_map.clear(); }
                flop_root_cfv_subset(ft, canonical, ranges, traverser, iters, &mut state_map)
            },
        );
    }
    let secs = t.elapsed().as_secs_f64();
    (secs, prod.strategy.clone(), prod.cum_strategy.clone())
}

#[test]
#[ignore = "Step 2.D.20: variance-vs-preflop-iter-count — test the favorable-regime hypothesis"]
fn step2d20_warm_variance_vs_preflop_iters() {
    let tree = build_minimal_hu_preflop_tree();
    let flop_tree = build_tiny_flop_tree();
    let np = tree.num_players as usize;
    eprintln!("\n=== Step 2.D.20: warm-K spread vs preflop iter count ===");
    eprintln!("Hypothesis: at higher preflop iters, DCFR-reset alignment variance averages out");
    eprintln!("→ best-K vs worst-K spread should SHRINK with growing preflop iters.");
    eprintln!();
    eprintln!("Forks #96's cost denominator:");
    eprintln!("  spread SHRINKS → production regime favorable → warm-start is real lever → bucketing relaxes");
    eprintln!("  spread PERSISTS → cold baseline carries → bucketing carries the load");
    eprintln!();

    let mut class_weights: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();
    for k in 0..NUM_PREFLOP_CLASSES {
        let s = k as f32 / NUM_PREFLOP_CLASSES as f32;
        class_weights[0][k] = ((s - 0.3).max(0.05) * 1.5).min(1.0);
        class_weights[1][k] = 0.6 + 0.4 * s;
    }
    let table = PreflopChanceTable::new(np as u8, class_weights);
    let canonical_subset: Vec<usize> = (0..CANONICAL_SUBSET_SIZE).collect();

    // POST #113 CONFIG FIX: use realistic terminal_value_fn (production
    // fold-CFV with blocking matrix) instead of random hash-derived TVF.
    // Step 2.D.21 probe confirmed random TVF → pure dirac equilibrium
    // (0% mixed) at this scale; realistic TVF → 16.67% mixed. Without
    // mixed equilibria the warm-K variance measurement is contaminated
    // by dirac-convergence robustness (#112 finding).
    use solver_core::solver::preflop_cfr::make_production_terminal_value_fn_hu;
    use solver_core::solver::preflop_terminal::build_class_blocking_matrix;
    let blocking = build_class_blocking_matrix();
    let terminal_value_fn = make_production_terminal_value_fn_hu(&tree, &blocking);

    // 3 preflop iter levels: small, medium, large.
    // K sweep at each: cover misaligned (5, 14) and aligned (8, 16) cases.
    // Always include cold K=200 ref and cold K=50 noise floor at the SAME preflop_iters.
    let preflop_levels = vec![5u32, 15, 30];
    let warm_ks = vec![5u32, 8, 14, 16];

    let mut per_level: Vec<(u32, f32, Vec<(u32, f32, f32)>)> = Vec::new();
    // per_level[i] = (preflop_iters, cold_k50_noise, [(K, dev, ratio_to_floor), ...])

    for &preflop_iters in &preflop_levels {
        eprintln!("\n────────── preflop_iters = {} ──────────", preflop_iters);

        let (ref_secs, ref_strategy, ref_cum) = run_variant_with_iters(
            &tree, &flop_tree, &table, &canonical_subset, &terminal_value_fn,
            200, false, preflop_iters,
        );
        eprintln!("  cold K=200 ref:   {:.1}s", ref_secs);
        // DIAGNOSTIC: characterize the strategy so bit-exact zero isn't blindly trusted.
        let n_zero: usize = ref_strategy.iter().filter(|&&v| v == 0.0).count();
        let n_one: usize = ref_strategy.iter().filter(|&&v| v == 1.0).count();
        let n_mixed: usize = ref_strategy.iter().filter(|&&v| v != 0.0 && v != 1.0).count();
        let cum_max: f32 = ref_cum.iter().cloned().fold(0.0f32, f32::max);
        let cum_min: f32 = ref_cum.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("    strategy stats: total={}, zero={}, one={}, mixed={}; cum range [{:.3e}, {:.3e}]",
            ref_strategy.len(), n_zero, n_one, n_mixed, cum_min, cum_max);

        let (cold50_secs, cold50_strategy, _) = run_variant_with_iters(
            &tree, &flop_tree, &table, &canonical_subset, &terminal_value_fn,
            50, false, preflop_iters,
        );
        let (cold50_dev, _) = max_abs_and_rms(&cold50_strategy, &ref_strategy);
        // DIAGNOSTIC: also compare against itself (run cold K=50 twice, check determinism).
        let (_, cold50_strategy2, _) = run_variant_with_iters(
            &tree, &flop_tree, &table, &canonical_subset, &terminal_value_fn,
            50, false, preflop_iters,
        );
        let (self_dev, _) = max_abs_and_rms(&cold50_strategy, &cold50_strategy2);
        eprintln!("  cold K=50 floor:  {:.1}s  dev = {:.3e}  (self-dev: {:.3e})",
            cold50_secs, cold50_dev, self_dev);

        let mut warm_rows: Vec<(u32, f32, f32)> = Vec::new();
        for &k in &warm_ks {
            let (w_secs, w_strategy, _) = run_variant_with_iters(
                &tree, &flop_tree, &table, &canonical_subset, &terminal_value_fn,
                k, true, preflop_iters,
            );
            let (w_dev, _) = max_abs_and_rms(&w_strategy, &ref_strategy);
            let ratio = w_dev / cold50_dev.max(1e-12);
            eprintln!("  warm K={:>3}:        {:.1}s  dev = {:.3e}  ({:.2}× cold-K50)",
                k, w_secs, w_dev, ratio);
            warm_rows.push((k, w_dev, ratio));
        }

        per_level.push((preflop_iters, cold50_dev, warm_rows));
    }

    eprintln!("\n══════════ SPREAD vs preflop iters ══════════");
    eprintln!("{:>14}  {:>12}  {:>16}  {:>16}  {:>12}",
        "preflop_iters", "cold-K50 dev", "best-K (× floor)", "worst-K (× floor)", "spread");
    let mut prior_spread: Option<f32> = None;
    for (pi, _floor, rows) in &per_level {
        let mut best_ratio = f32::INFINITY;
        let mut worst_ratio = 0.0f32;
        let mut best_k = 0;
        let mut worst_k = 0;
        for &(k, _dev, ratio) in rows {
            if ratio < best_ratio { best_ratio = ratio; best_k = k; }
            if ratio > worst_ratio { worst_ratio = ratio; worst_k = k; }
        }
        let spread = worst_ratio - best_ratio;
        let trend_str = if let Some(prev) = prior_spread {
            let trend_pct = (spread - prev) / prev.max(1e-12) * 100.0;
            format!("  (Δ {:+.0}%)", trend_pct)
        } else {
            String::new()
        };
        eprintln!("{:>14}  {:>12.3e}  K={} {:>10.2}×  K={} {:>10.2}×  {:>10.2}×{}",
            pi, per_level[per_level.iter().position(|(p,_,_)| p==pi).unwrap()].1,
            best_k, best_ratio, worst_k, worst_ratio, spread, trend_str);
        prior_spread = Some(spread);
    }

    let first_spread = per_level[0].2.iter()
        .map(|(_,_,r)| *r).fold(0.0f32, f32::max)
        - per_level[0].2.iter().map(|(_,_,r)| *r).fold(f32::INFINITY, f32::min);
    let last_spread = per_level[per_level.len()-1].2.iter()
        .map(|(_,_,r)| *r).fold(0.0f32, f32::max)
        - per_level[per_level.len()-1].2.iter().map(|(_,_,r)| *r).fold(f32::INFINITY, f32::min);
    let spread_shrink_ratio = last_spread / first_spread.max(1e-12);

    eprintln!();
    eprintln!("══════════ VERDICT ══════════");
    eprintln!("Spread at preflop_iters={}: {:.2}× (best-vs-worst K, in cold-K50 floors)",
        preflop_levels[0], first_spread);
    eprintln!("Spread at preflop_iters={}: {:.2}× (best-vs-worst K, in cold-K50 floors)",
        preflop_levels[preflop_levels.len()-1], last_spread);
    eprintln!("Spread ratio (last/first): {:.2}×", spread_shrink_ratio);
    eprintln!();
    if spread_shrink_ratio < 0.5 {
        eprintln!("→ Spread SHRINKS substantially (< 50% of original). Production regime is FAVORABLE.");
        eprintln!("  Warm-start is a real lever at production scale; bucketing pressure for #96 RELAXES.");
        eprintln!("  Recommended #96 cost denominator: warm-start (regime to be selected).");
    } else if spread_shrink_ratio < 1.0 {
        eprintln!("→ Spread shrinks modestly. Production regime is MILDLY favorable.");
        eprintln!("  Project the trend at larger preflop_iters before committing #96 denominator.");
    } else {
        eprintln!("→ Spread PERSISTS or grows. Production regime is UNFAVORABLE for simple-fixed-K warm-start.");
        eprintln!("  Cold baseline carries; bucketing must carry the load for #96.");
    }
    eprintln!();
    eprintln!("CAVEATS:");
    eprintln!("- Still subset config ({} canonicals). Production has 1755.", CANONICAL_SUBSET_SIZE);
    eprintln!("- Cold K=200 reference at each preflop_iters may itself be under-converged;");
    eprintln!("  spread measurement is robust to this since both warm and cold compare to same ref.");
    eprintln!("- Three preflop iter levels is a trend, not a converged extrapolation.");

    // DISCRIMINATING DIAGNOSTIC: is cold K=200 itself converged at the
    // highest preflop_iters? Compare cold K=200 at preflop_iters=last vs
    // preflop_iters=last+20. If they differ substantially, the "noise
    // floor" calibration shifts and the warm-K deviation interpretation
    // needs reframing.
    let pi_last = *preflop_levels.last().unwrap();
    let pi_check = pi_last + 20;
    eprintln!();
    eprintln!("══════════ REFERENCE-CONVERGENCE DIAGNOSTIC ══════════");
    eprintln!("Is cold K=200 itself converged at preflop_iters={}?", pi_last);
    eprintln!("Comparing cold K=200 at preflop_iters={} vs {}.", pi_last, pi_check);
    eprintln!("(If similar → cold K=200 converged → warm-K deviations are real error.");
    eprintln!(" If different → reference itself still moving → metric needs reframing.)");
    let (s1_secs, s1_strategy, _) = run_variant_with_iters(
        &tree, &flop_tree, &table, &canonical_subset, &terminal_value_fn,
        200, false, pi_last,
    );
    let (s2_secs, s2_strategy, _) = run_variant_with_iters(
        &tree, &flop_tree, &table, &canonical_subset, &terminal_value_fn,
        200, false, pi_check,
    );
    let (ref_drift, ref_drift_rms) = max_abs_and_rms(&s1_strategy, &s2_strategy);
    let last_cold_floor = per_level[per_level.len()-1].1;
    let drift_ratio = ref_drift / last_cold_floor.max(1e-12);
    eprintln!();
    eprintln!("  cold K=200 @ preflop={}: {:.1}s", pi_last, s1_secs);
    eprintln!("  cold K=200 @ preflop={}: {:.1}s", pi_check, s2_secs);
    eprintln!("  reference drift: max_abs={:.3e} rms={:.3e}", ref_drift, ref_drift_rms);
    eprintln!("  drift vs cold K=50 floor at preflop={}: {:.2}×", pi_last, drift_ratio);
    eprintln!();
    if drift_ratio < 0.1 {
        eprintln!("→ cold K=200 reference IS converged at preflop_iters={} (drift < 10% of floor).", pi_last);
        eprintln!("  Warm-K deviations from cold K=200 are REAL equilibrium-distance.");
        eprintln!("  Lever measurement is on solid ground.");
    } else if drift_ratio < 1.0 {
        eprintln!("→ cold K=200 reference is APPROXIMATELY converged (drift ~ floor magnitude).");
        eprintln!("  Warm-K deviation interpretation is partial — true equilibrium is closer than");
        eprintln!("  cold K=200, so all variants likely under-estimate true distance.");
    } else {
        eprintln!("→ cold K=200 reference is NOT converged — drift > cold K=50 floor.");
        eprintln!("  The 'noise floor' has been calibrated against a moving reference. Warm-K");
        eprintln!("  deviations are not directly interpretable. Need higher K (K=500+) or more");
        eprintln!("  preflop iters as reference before warm-K lever can be sized.");
    }
}
