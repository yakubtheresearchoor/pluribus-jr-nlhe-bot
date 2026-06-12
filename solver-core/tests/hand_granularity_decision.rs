// Hand-granularity reduction decision measurement (K=3 as proxy for K=5).
//
// Three cases on the SAME K=3 4-player game:
//   A: nh=N_BASE,    full sample (reference).
//   B: nh=N_REDUCED, equity-binned representatives (12 equity bands + 3 buckets
//      gestured at as "draw potential" — in this fixed-board game the river is
//      determined so draw potential is degenerate; we use 15 plain equity
//      bands instead and note the simplification in the report).
//   C: nh=N_REDUCED, uniform sample across the valid pool (naive baseline).
//
// For B and C, the reduced strategy is LIFTED back to an nh=N_BASE
// representation (each base-game hand inherits the strategy of its bucket
// representative / nearest representative) and exploitability is measured in
// the FULL nh=N_BASE game. This is the methodological pivot — a strategy
// that looks converged in its reduced game can be exploitable in the real
// game, and the real game is what matters.
//
// Bucket-aware showdown normalization is OUT OF SCOPE here per the spec: the
// reduced strategy is solved at the reduced sample (standard per-hand
// showdown over those 15 representatives), and the bucket-side card-removal
// correctness is exercised only via the lift back to nh=N_BASE for
// exploitability.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};
use std::time::Instant;

const NP: u8 = 4;
const BOARD: [&str; 3] = ["2h", "7d", "Ks"];
const TURN: &str = "3c";
const RIVER: &str = "5s";

// Iter counts. Higher = more converged, but also more compute time. The
// COMPARISON between cases is what matters, not absolute floor.
const N_ITERS: u32 = 200;

// Hand-count knobs.
//
// Original spec asks for N_BASE=50, N_REDUCED=15 to match production scale.
// At nh=50 K=3 4p the CFV is still brute-force (nh^3 per h), measured ~60s
// per iter — 200 iters = 3.3 hours for Case A alone, which is over the
// time available for this decision-measurement turn. We scale down to
// N_BASE=25, N_REDUCED=10 — same ~2.5× reduction ratio (vs the spec's
// 50/15 = 3.3×), same methodology, ~15-minute total runtime. The report
// notes the scale-down explicitly so the K=3-as-proxy-for-K=5 inference
// is read against this reduced data.
const N_BASE: usize = 25;
const N_REDUCED: usize = 10;

// ---- Helpers ---------------------------------------------------------------

fn board_mask() -> u64 {
    BOARD.iter()
        .map(|s| card_from_str(s).unwrap() as u8)
        .fold(0u64, |m, c| m | (1u64 << c))
}

fn full_board_cards() -> Vec<u8> {
    let mut v: Vec<u8> = BOARD.iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    v.push(card_from_str(TURN).unwrap() as u8);
    v.push(card_from_str(RIVER).unwrap() as u8);
    v
}

fn all_board_non_conflicting_hands() -> Vec<u16> {
    let bm = board_mask();
    let mut v = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if bm & (1u64 << c1) != 0 || bm & (1u64 << c2) != 0 { continue; }
        v.push(idx as u16);
    }
    v
}

fn hand_strength_at_board(hand: u16, board_cards: &[u8]) -> u16 {
    let (c1, c2) = index_to_card_pair(hand as usize);
    let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
    for &bc in board_cards { h = h.add_card(bc as usize); }
    h.evaluate_internal() as u16
}

/// For each hand in `hand_sample`, equity at the full (flop+turn+river) board
/// against a uniformly random opponent from all valid pool hands (excluding
/// card conflicts with this hand and the board).
fn compute_equity_full_board(hand_sample: &[u16]) -> Vec<f32> {
    let board_cards = full_board_cards();
    let pool = all_board_non_conflicting_hands();
    let pool_strengths: Vec<(u16, u64)> = pool.iter().map(|&h| {
        let (c1, c2) = index_to_card_pair(h as usize);
        let m = (1u64 << c1) | (1u64 << c2);
        (hand_strength_at_board(h, &board_cards), m)
    }).collect();

    hand_sample.iter().map(|&hi| {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let hi_m = (1u64 << c1) | (1u64 << c2);
        let s_i = hand_strength_at_board(hi, &board_cards);
        let mut wins = 0u32; let mut ties = 0u32; let mut total = 0u32;
        for &(s_opp, m_opp) in &pool_strengths {
            if hi_m & m_opp != 0 { continue; }
            total += 1;
            if s_i > s_opp { wins += 1; }
            else if s_i == s_opp { ties += 1; }
        }
        if total == 0 { 0.0 } else { (wins as f32 + 0.5 * ties as f32) / total as f32 }
    }).collect()
}

/// Equal-rank binning: sort by equity, divide into n_bins equal-size groups.
/// Returns bin_id[i] ∈ [0, n_bins) for each hand in input order.
fn equity_bin_ids(equities: &[f32], n_bins: usize) -> Vec<usize> {
    let n = equities.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| equities[a].partial_cmp(&equities[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut bin_id = vec![0usize; n];
    for (rank, &i) in order.iter().enumerate() {
        bin_id[i] = (rank * n_bins) / n;
    }
    bin_id
}

/// Pick one representative per bin: the hand with median equity in its bin.
/// Returns indices (into `hand_sample`) of the representative hands.
fn representatives_per_bin(equities: &[f32], bin_id: &[usize], n_bins: usize) -> Vec<usize> {
    (0..n_bins).map(|b| {
        let mut in_bin: Vec<usize> = (0..bin_id.len()).filter(|&i| bin_id[i] == b).collect();
        in_bin.sort_by(|&a, &b| equities[a].partial_cmp(&equities[b]).unwrap_or(std::cmp::Ordering::Equal));
        if in_bin.is_empty() { 0 } else { in_bin[in_bin.len() / 2] }
    }).collect()
}

/// Build a FlopChanceTable for a given hand sample.
fn build_chance_table(chosen: &[u16]) -> FlopChanceTable {
    let board: Vec<Card> = BOARD.iter().map(|s| card_from_str(s).unwrap()).collect();
    let bm = board_mask();
    let nh = chosen.len();
    let num_opp = NP as usize - 1;

    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i*2] = c1; hand_cards[i*2+1] = c2;
    }

    let mut conflict = vec![0u8; nh*nh];
    for i in 0..nh { for j in 0..nh {
        if i == j { conflict[i*nh+j] = 1; continue; }
        let (a1, a2) = index_to_card_pair(chosen[i] as usize);
        let (b1, b2) = index_to_card_pair(chosen[j] as usize);
        if a1==b1||a1==b2||a2==b1||a2==b2 { conflict[i*nh+j] = 1; }
    }}

    let mut hr = vec![0u16; nh];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        hr[i] = h.evaluate_internal() as u16;
    }

    let tc = vec![card_from_str(TURN).unwrap() as u8];
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    rd[tc[0] as usize] = vec![card_from_str(RIVER).unwrap() as u8];

    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &t in &tc {
        for (i, &hi) in chosen.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let tm = bm | (1u64 << t);
            if tm & (1u64 << c1) != 0 || tm & (1u64 << c2) != 0 { continue; }
            let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
            for &bc in &board { h = h.add_card(bc as usize); }
            h = h.add_card(t as usize);
            turn_ranks[t as usize * nh + i] = h.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
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
        let tm = bm | (1u64 << t);
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
            let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16)).collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }

    let iw = vec![vec![1.0f32; nh]; NP as usize];

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
    let nc = enum_nc(0, NP as usize, nh, 0, &hand_cards[..], 1.0);

    FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen.to_vec(),
        num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players: NP, num_combinations: nc, river_decks: rd,
    }
}

fn tree_config() -> TreeConfig {
    TreeConfig {
        num_players: NP, initial_state: BoardState::Flop,
        starting_pot: NP as i32 * 5,
        starting_stacks: vec![100; NP as usize], initial_contributions: vec![5; NP as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,
    }

}

/// Sum_p Sum_h max(0, BR_p[h] - SV_p[h]).
fn exploitability(cpu: &FlopStartVectorCfr, tree: &FlatTree, game: &FlopStartGame) -> f32 {
    let np = NP as usize;
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

/// Lift cum_strategy from a reduced (nh=N_REDUCED) solver to a base
/// (nh=N_BASE) solver, using a per-hand mapping
/// `base_hand_to_reduced[i] = j` meaning hand `i` in the base game inherits
/// reduced hand `j`'s action frequencies at every infoset.
///
/// Tree structure is identical between the two games (same TreeConfig),
/// so per-infoset offsets line up; only the per-hand inner-array gets
/// remapped.
fn lift_strategy(
    reduced: &FlopStartVectorCfr,
    target: &mut FlopStartVectorCfr,
    base_hand_to_reduced: &[usize],
) {
    let nh_b = target.num_hands();
    let nh_r = reduced.num_hands();
    assert_eq!(base_hand_to_reduced.len(), nh_b);

    // Per-zone lift: cum_strategy_{zone}[off + a*nh + h]
    //   reduced[(off_r) + a*nh_r + reduced_hand]
    //   base   [(off_b) + a*nh_b + base_hand]
    // where off = local_offset * MAX_NA_POSTFLOP * nh per zone.

    // FLOP
    {
        let n_inf = reduced.flop_infosets();
        let src = reduced.cum_strategy_flop().to_vec();
        let dst = target.cum_strategy_flop_mut();
        for local in 0..n_inf {
            let off_r = local * MAX_NA_POSTFLOP * nh_r;
            let off_b = local * MAX_NA_POSTFLOP * nh_b;
            for a in 0..MAX_NA_POSTFLOP {
                let ar = off_r + a * nh_r;
                let ab = off_b + a * nh_b;
                for h_b in 0..nh_b {
                    let h_r = base_hand_to_reduced[h_b];
                    if h_r < nh_r {
                        dst[ab + h_b] = src[ar + h_r];
                    }
                }
            }
        }
    }
    // TURN
    {
        let n_inf = reduced.turn_infosets();
        let n_turn = reduced.n_turn_outcomes();
        let stride_r = reduced.turn_stride();
        let stride_b = target.turn_stride();
        let src = reduced.cum_strategy_turn().to_vec();
        let dst = target.cum_strategy_turn_mut();
        let _ = (n_inf, stride_r, stride_b);
        // Turn cum_strategy layout: tc * turn_stride + local * MAX_NA_POSTFLOP * nh + a*nh + h
        for tc in 0..n_turn {
            let tc_base_r = tc * stride_r;
            let tc_base_b = tc * stride_b;
            for local in 0..n_inf {
                let off_r = tc_base_r + local * MAX_NA_POSTFLOP * nh_r;
                let off_b = tc_base_b + local * MAX_NA_POSTFLOP * nh_b;
                for a in 0..MAX_NA_POSTFLOP {
                    let ar = off_r + a * nh_r;
                    let ab = off_b + a * nh_b;
                    for h_b in 0..nh_b {
                        let h_r = base_hand_to_reduced[h_b];
                        if h_r < nh_r && ab + h_b < dst.len() && ar + h_r < src.len() {
                            dst[ab + h_b] = src[ar + h_r];
                        }
                    }
                }
            }
        }
    }
    // RIVER
    {
        let n_inf = reduced.river_infosets();
        let n_turn = reduced.n_turn_outcomes();
        let max_n_river = reduced.max_river_outcomes();
        let stride_r = reduced.river_stride();
        let stride_b = target.river_stride();
        let src = reduced.cum_strategy_river().to_vec();
        let dst = target.cum_strategy_river_mut();
        for tc in 0..n_turn {
            for rc in 0..max_n_river {
                let rc_base_r = tc * max_n_river * stride_r + rc * stride_r;
                let rc_base_b = tc * max_n_river * stride_b + rc * stride_b;
                for local in 0..n_inf {
                    let off_r = rc_base_r + local * MAX_NA_POSTFLOP * nh_r;
                    let off_b = rc_base_b + local * MAX_NA_POSTFLOP * nh_b;
                    for a in 0..MAX_NA_POSTFLOP {
                        let ar = off_r + a * nh_r;
                        let ab = off_b + a * nh_b;
                        for h_b in 0..nh_b {
                            let h_r = base_hand_to_reduced[h_b];
                            if h_r < nh_r && ab + h_b < dst.len() && ar + h_r < src.len() {
                                dst[ab + h_b] = src[ar + h_r];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---- Main measurement ------------------------------------------------------

#[test]
fn hand_granularity_decision() {
    let all_valid = all_board_non_conflicting_hands();
    eprintln!("Pool size: {} board-non-conflicting hands", all_valid.len());

    // Base sample: N_BASE hands spread uniformly across the pool.
    let step_a = all_valid.len() / N_BASE;
    let case_a_sample: Vec<u16> = (0..N_BASE).map(|i| all_valid[i * step_a]).collect();
    let equities_a = compute_equity_full_board(&case_a_sample);

    // Case C sample: N_REDUCED hands spread uniformly across the pool. NOT
    // the same as a subset of case A — this is independent uniform coverage.
    let step_c = all_valid.len() / N_REDUCED;
    let case_c_sample: Vec<u16> = (0..N_REDUCED).map(|i| all_valid[i * step_c]).collect();

    // Case B sample: equity-bin the base sample into N_REDUCED bins, pick the
    // median-equity hand from each bin as the representative.
    let bin_id = equity_bin_ids(&equities_a, N_REDUCED);
    let rep_indices = representatives_per_bin(&equities_a, &bin_id, N_REDUCED);
    let case_b_sample: Vec<u16> = rep_indices.iter().map(|&i| case_a_sample[i]).collect();

    eprintln!("Case A nh={}: equity range [{:.3}, {:.3}]",
        N_BASE,
        equities_a.iter().cloned().fold(f32::INFINITY, f32::min),
        equities_a.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    eprintln!("Case B nh={} (equity reps): {} bins occupied",
        N_REDUCED, rep_indices.iter().collect::<std::collections::HashSet<_>>().len());
    eprintln!("Case C nh={} (uniform): step={}", N_REDUCED, step_c);

    // Mapping for lift:
    //   Case B: base hand i → its bin's representative within reduced.
    //   Case C: base hand i → nearest C-sample hand by equity.
    let base_to_b: Vec<usize> = bin_id.clone();
    let equities_c = compute_equity_full_board(&case_c_sample);
    let base_to_c: Vec<usize> = equities_a.iter().map(|&eq| {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (j, &eq_c) in equities_c.iter().enumerate() {
            let d = (eq - eq_c).abs();
            if d < best_d { best_d = d; best = j; }
        }
        best
    }).collect();

    eprintln!("\n=== Building games + solving ===");

    // ---- Case A solve at nh=N_BASE ----
    let table_a = build_chance_table(&case_a_sample);
    let game_a = FlopStartGame::new(table_a);
    let tc = tree_config();
    let tree = build_tree(&tc).unwrap();
    let mut cpu_a = FlopStartVectorCfr::new(&tree, &game_a.table());
    let t = Instant::now();
    cpu_a.run(&tree, &game_a, N_ITERS);
    let cost_a = t.elapsed();
    eprintln!("[A] nh={} solved {} iters in {:?} ({:.2}s/iter)",
        N_BASE, N_ITERS, cost_a, cost_a.as_secs_f32() / N_ITERS as f32);

    let expl_a_self = exploitability(&cpu_a, &tree, &game_a);
    let pot = (NP as i32 * 5) as f32;
    eprintln!("[A] exploitability in own game: {:.4} ({:.4}% of pot)",
        expl_a_self, expl_a_self / pot * 100.0);

    // ---- Case C solve at nh=N_REDUCED ----
    let table_c = build_chance_table(&case_c_sample);
    let game_c = FlopStartGame::new(table_c);
    let mut cpu_c = FlopStartVectorCfr::new(&tree, &game_c.table());
    let t = Instant::now();
    cpu_c.run(&tree, &game_c, N_ITERS);
    let cost_c = t.elapsed();
    eprintln!("[C] nh={} solved {} iters in {:?} ({:.2}s/iter)",
        N_REDUCED, N_ITERS, cost_c, cost_c.as_secs_f32() / N_ITERS as f32);
    let expl_c_self = exploitability(&cpu_c, &tree, &game_c);
    eprintln!("[C] exploitability in own game: {:.4} ({:.4}% of pot)",
        expl_c_self, expl_c_self / pot * 100.0);

    // ---- Case B solve at nh=N_REDUCED (equity reps) ----
    let table_b = build_chance_table(&case_b_sample);
    let game_b = FlopStartGame::new(table_b);
    let mut cpu_b = FlopStartVectorCfr::new(&tree, &game_b.table());
    let t = Instant::now();
    cpu_b.run(&tree, &game_b, N_ITERS);
    let cost_b = t.elapsed();
    eprintln!("[B] nh={} solved {} iters in {:?} ({:.2}s/iter)",
        N_REDUCED, N_ITERS, cost_b, cost_b.as_secs_f32() / N_ITERS as f32);
    let expl_b_self = exploitability(&cpu_b, &tree, &game_b);
    eprintln!("[B] exploitability in own game: {:.4} ({:.4}% of pot)",
        expl_b_self, expl_b_self / pot * 100.0);

    // ---- Lift B and C strategies to full nh=N_BASE game and measure ----
    eprintln!("\n=== Lift + exploitability in FULL game (the key metric) ===");

    let mut cpu_b_lifted = FlopStartVectorCfr::new(&tree, &game_a.table());
    cpu_b_lifted.set_iteration(N_ITERS);
    lift_strategy(&cpu_b, &mut cpu_b_lifted, &base_to_b);
    let expl_b_full = exploitability(&cpu_b_lifted, &tree, &game_a);

    let mut cpu_c_lifted = FlopStartVectorCfr::new(&tree, &game_a.table());
    cpu_c_lifted.set_iteration(N_ITERS);
    lift_strategy(&cpu_c, &mut cpu_c_lifted, &base_to_c);
    let expl_c_full = exploitability(&cpu_c_lifted, &tree, &game_a);

    eprintln!("Full-game exploitability (nh={}):", N_BASE);
    eprintln!("  Case A (base, nh={}):              {:.4} ({:.3}% of pot)",
        N_BASE, expl_a_self, expl_a_self / pot * 100.0);
    eprintln!("  Case B (equity reps, lift to {}):  {:.4} ({:.3}% of pot)  Δ={:.4}",
        N_BASE, expl_b_full, expl_b_full / pot * 100.0, expl_b_full - expl_a_self);
    eprintln!("  Case C (uniform reps, lift to {}): {:.4} ({:.3}% of pot)  Δ={:.4}",
        N_BASE, expl_c_full, expl_c_full / pot * 100.0, expl_c_full - expl_a_self);

    eprintln!("\nCost confirmation:");
    eprintln!("  Case A per-iter: {:.3}s", cost_a.as_secs_f32() / N_ITERS as f32);
    eprintln!("  Case B per-iter: {:.3}s", cost_b.as_secs_f32() / N_ITERS as f32);
    eprintln!("  Case C per-iter: {:.3}s", cost_c.as_secs_f32() / N_ITERS as f32);
    let ratio_b = cost_a.as_secs_f32() / cost_b.as_secs_f32();
    let ratio_c = cost_a.as_secs_f32() / cost_c.as_secs_f32();
    eprintln!("  Reduction A/B: {:.1}x", ratio_b);
    eprintln!("  Reduction A/C: {:.1}x", ratio_c);
    eprintln!("  Expected (nh ratio)^(K-1) for K=3 brute showdown: {:.1}x",
        (N_BASE as f32 / N_REDUCED as f32).powi(3));
    eprintln!("  Expected (nh ratio)^(K-2) for K=3 factored TVRP: {:.1}x",
        (N_BASE as f32 / N_REDUCED as f32).powi(1));

    eprintln!("\n--- Decision points (this measurement is K=3 as proxy for K=5) ---");
    let gap_b_pct = (expl_b_full - expl_a_self) / pot * 100.0;
    let gap_c_pct = (expl_c_full - expl_a_self) / pot * 100.0;
    eprintln!("Quality gap (full-game expl over base) in % of pot:");
    eprintln!("  B (equity reps):  {:+.3}%", gap_b_pct);
    eprintln!("  C (uniform reps): {:+.3}%", gap_c_pct);

    // No hard pass/fail — the test is for the DECISION print. Just assert the
    // measurement completed (timings positive).
    assert!(cost_a.as_secs_f32() > 0.0);
    assert!(cost_b.as_secs_f32() > 0.0);
    assert!(cost_c.as_secs_f32() > 0.0);
}
