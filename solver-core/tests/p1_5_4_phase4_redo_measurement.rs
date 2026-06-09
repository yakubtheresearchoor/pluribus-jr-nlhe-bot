// Phase 4 REDO — the actual measurement.
//
// Composes P0 (harder config that produces nonzero rich exploitability),
// P1 (empirical bet-size observer that drives lean selection), P2 (cross-
// action-space best-response evaluator via cross_tree::lift_strategy +
// best_response::exploitability).
//
// THE QUESTION: at what K does the empirical top-K lean action set incur
// cross-action-space exploit ≤ 0.05% pot above rich's own exploit?
//
// Production target = 0.05% pot (tight). Above the production target, lean
// is strictly worse than rich and we'd be sacrificing solution quality for
// a cheaper postflop tree. At-or-below the target, the lean set is sound.
//
// The smallest K that passes IS MAX_NA_POSTFLOP (+1 for fold and +1 for
// check/call, since both must always be included: na_total = K + 2).
//
// COMPARED TO PHASE 4 v4 (reverted):
//   - v4 hit 0% vs 0% — both at the f32 floor. Vacuous.
//   - v4 lean set came from hand-rolled bands (workaround for top-K-by-mass
//     triggering tree explosion). Not empirical.
//   - v4 used same-action-space exploitability — couldn't see dropped sizes.
//
// THIS TEST FIXES ALL THREE:
//   - Config has rich exploitability ~0.87% pot at iter 25 (measured in P0)
//   - Lean set is the empirical top-K with a structural min-chip floor (no
//     band hand-picking)
//   - Cost is the cross-action-space BR-exploit measurement (lifted lean
//     strategy played against best-response opponent in the rich tree)

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::cross_tree::{
    build_action_map, build_pseudo_harmonic_map, lift_into_rich_solver_with_lean,
    lift_into_rich_solver_pseudo_harmonic,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{
    FlatTree, ACTION_LABEL_BET, ACTION_LABEL_RAISE,
};
use std::collections::BTreeMap;
use std::time::Instant;

// === Config (from P0 probe) ====================================================

const STACKS: i32 = 500;
const STARTING_POT: i32 = 30;
const STARTING_CONTRIB: i32 = 5;
const NH: usize = 6;
const NP: u8 = 6;
const N_ITERS: u32 = 50; // baseline iters for rich (always run to this)
// Lean is run until self-expl drops below this threshold, or LEAN_MAX_ITERS,
// whichever comes first. Eliminates the "K=3 noise spike" problem in the
// first-pass measurement where K=3's lean self-expl was 0.0279% vs K=4's
// 0.0018% at iter 50, producing a non-monotonic sweep (K=3 cost > K=4 cost).
const LEAN_SELF_EXPL_FLOOR_PCT: f32 = 0.005;
const LEAN_MAX_ITERS: u32 = 300;
const LEAN_ITER_CHUNK: u32 = 25;

// Min chip increment to count a bet as a "real" size choice. Drops tiny
// bets that would otherwise top the conditional-mass ranking and trigger
// tree explosion when lean tries to include them. 10 chips = 33% of pot
// at the start; smaller bets are uncommon in real play and structurally
// expensive to support in the postflop tree.
const MIN_CHIP_INCREMENT: i32 = 10;

fn rich_bet_sizes() -> Vec<BetSize> {
    vec![
        BetSize::PotRelative(0.33),
        BetSize::PotRelative(0.66),
        BetSize::PotRelative(1.0),
        BetSize::PotRelative(1.5),
    ]
}
fn rich_raise_sizes() -> Vec<BetSize> {
    vec![
        BetSize::PotRelative(1.0),
        BetSize::PotRelative(2.0),
    ]
}

fn build_chance_table() -> FlopChanceTable {
    let board: Vec<Card> = ["Th", "9d", "8c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
            ranges[p][pair_idx] = 1.0;
        }
    }
    let turn_cards: Vec<u8> = ["2c", "Jd"].iter()
        .map(|s| card_from_str(s).unwrap() as u8).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = ["4s", "7h"].iter()
        .map(|s| card_from_str(s).unwrap() as u8).collect();
    river_decks[turn_cards[1] as usize] = ["3s", "Qc"].iter()
        .map(|s| card_from_str(s).unwrap() as u8).collect();
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

fn build_tree_with(bets: Vec<BetSize>, raises: Vec<BetSize>) -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: STARTING_POT,
        starting_stacks: vec![STACKS; NP as usize],
        initial_contributions: vec![STARTING_CONTRIB; NP as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: bets, raise: raises },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    build_tree(&config).unwrap()
}

// === P1: Empirical bet-size observer ============================================
//
// At every PLAYER node, for every hand h:
//   1. Normalize cum_strategy → σ_avg[a, h]
//   2. bet_mass(h) = sum over actions a where action is BET or RAISE of σ_avg[a, h]
//   3. If bet_mass(h) is meaningfully nonzero, then for each such action a:
//        conditional_prob(size, h) = σ_avg[a, h] / bet_mass(h)
//        ↑ "given the player chose to bet here with this hand, P(size = this)"
//   4. Bucket by pot_fraction (rounded to 0.01) and accumulate conditional probs
//   5. Normalize across buckets → rank by aggregate conditional mass
//
// This is what the v4 banded heuristic was hand-rolling around. With the
// MIN_CHIP_INCREMENT floor, we don't need the bands — tiny-bet pathologies
// are filtered structurally, not by hand-picked ranges.

#[derive(Debug, Clone)]
struct BetSizeObservation {
    pot_fraction: f32,      // representative (mean of bucketed observations)
    conditional_mass: f64,  // sum of conditional probability across all infosets×hands
    occurrence_count: usize,
}

fn observe_bet_sizes(
    tree: &FlatTree,
    cum_strategy: &[f32],
    offsets: &[usize],
) -> Vec<BetSizeObservation> {
    let np = tree.num_players as usize;
    let nh = NH;

    // Bucket pot_fraction by 0.01 granularity (rounded).
    let mut bucket_mass: BTreeMap<i32, f64> = BTreeMap::new();
    let mut bucket_frac_sum: BTreeMap<i32, f64> = BTreeMap::new();
    let mut bucket_count: BTreeMap<i32, usize> = BTreeMap::new();

    for &nid in &tree.decision_node_ids {
        let idx = nid as usize;
        let node = &tree.nodes[idx];
        let p = node.player_id as usize;
        let na = node.num_children as usize;
        let off = offsets[idx];
        let children = tree.node_children(idx).to_vec();

        // Pot at this node = sum of all contributions. `tree.starting_pot`
        // is NOT added: the builder's bet-size math uses contributions-only
        // (verified empirically — adding starting_pot causes declared 0.66p
        // bets to bucket at 0.55, etc.). showdown.rs's formula adds
        // starting_pot for FINAL terminal pot only.
        let mut pot: i64 = 0;
        for pp in 0..np {
            pot += tree.contributions[idx * np + pp] as i64;
        }
        if pot <= 0 { continue; }

        let parent_p_contrib = tree.contributions[idx * np + p];

        for h in 0..nh {
            // Normalize σ_avg over actions for this hand
            let mut total = 0.0f32;
            for a in 0..na {
                total += cum_strategy[off + a * nh + h].max(0.0);
            }
            if total <= 1e-12 { continue; }

            // Identify bet actions and accumulate bet_mass
            let mut bet_mass = 0.0f32;
            for a in 0..na {
                let ac = &tree.nodes[children[a] as usize];
                if ac.action_label != ACTION_LABEL_BET && ac.action_label != ACTION_LABEL_RAISE {
                    continue;
                }
                bet_mass += cum_strategy[off + a * nh + h].max(0.0) / total;
            }
            if bet_mass < 1e-6 { continue; }

            // For each bet action, accumulate its conditional probability
            for a in 0..na {
                let ac = &tree.nodes[children[a] as usize];
                if ac.action_label != ACTION_LABEL_BET && ac.action_label != ACTION_LABEL_RAISE {
                    continue;
                }
                let child_p_contrib = tree.contributions[children[a] as usize * np + p];
                let inc = child_p_contrib - parent_p_contrib;
                if inc < MIN_CHIP_INCREMENT { continue; }
                let prob = cum_strategy[off + a * nh + h].max(0.0) / total;
                let cond = (prob / bet_mass) as f64;
                let pot_frac = inc as f32 / pot as f32;
                let bucket = (pot_frac * 100.0).round() as i32;
                *bucket_mass.entry(bucket).or_insert(0.0) += cond;
                *bucket_frac_sum.entry(bucket).or_insert(0.0) += pot_frac as f64;
                *bucket_count.entry(bucket).or_insert(0) += 1;
            }
        }
    }

    let total_mass: f64 = bucket_mass.values().sum();
    let mut out: Vec<BetSizeObservation> = bucket_mass.iter()
        .map(|(&bucket, &m)| {
            let count = bucket_count[&bucket];
            let frac = (bucket_frac_sum[&bucket] / count as f64) as f32;
            BetSizeObservation {
                pot_fraction: frac,
                conditional_mass: if total_mass > 0.0 { m / total_mass } else { 0.0 },
                occurrence_count: count,
            }
        })
        .collect();
    out.sort_by(|a, b| b.conditional_mass.partial_cmp(&a.conditional_mass).unwrap());
    out
}

// Snap an observed pot fraction to one of the DECLARED rich bet sizes. The
// lean tree builder takes BetSize::PotRelative(f) so we have to feed it the
// exact rich values, not the bucketed averages (which might be 0.34 vs the
// declared 0.33 due to integer chip math).
fn snap_to_rich_size(pot_frac: f32) -> Option<f32> {
    let candidates = [0.33f32, 0.66, 1.0, 1.5];
    let mut best = None;
    let mut best_diff = f32::INFINITY;
    for &c in &candidates {
        let d = (pot_frac - c).abs();
        if d < best_diff && d < 0.10 {
            best = Some(c);
            best_diff = d;
        }
    }
    best
}

// === P2: Run the cross-action-space measurement =================================

fn solve(tree: &FlatTree, game: &FlopStartGame, n_iters: u32) -> FlopStartVectorCfr {
    let mut cpu = FlopStartVectorCfr::new(tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, tree, game, &cpu);
    gpu.run(&ctx, tree, game, n_iters);
    cpu.run(tree, game, n_iters);
    cpu
}

/// Walk rich tree's PLAYER nodes; for each with a single PH-map entry
/// (weight 1.0), compare lifted cum_strategy slot to lean's source slot
/// for every action with a corresponding lean action. Reports the count
/// of mismatches (should be 0 for a correct lift) and the worst-case
/// absolute diff. Bounded sample-size early-stop keeps cost low.
fn diagnose_single_entry_lift_fidelity(
    rich_tree: &FlatTree,
    lean_tree: &FlatTree,
    ph_map: &solver_core::solver::cross_tree::PseudoHarmonicCrossTreeMap,
    lean_cpu: &FlopStartVectorCfr,
    rich_cpu_lifted: &FlopStartVectorCfr,
) {
    use solver_core::tree::flat::{
        MAX_NA_POSTFLOP, ACTION_LABEL_BET, ACTION_LABEL_RAISE,
    };
    let nh = lean_cpu.num_hands();
    let lean_zones = lean_cpu.zones();
    let rich_zones = rich_cpu_lifted.zones();
    let lean_flop_local = lean_cpu.flop_local_offset();
    let rich_flop_local = rich_cpu_lifted.flop_local_offset();

    // Helper to compute pot-fraction for a rich/lean action.
    let pf = |tree: &FlatTree, parent: usize, child: usize, player: usize| -> f64 {
        let np = tree.num_players as usize;
        let pot_p: i64 = (0..np).map(|p| tree.contributions[parent * np + p] as i64).sum();
        if pot_p <= 0 { return 0.0; }
        let pc = tree.contributions[parent * np + player] as i64;
        let cc = tree.contributions[child * np + player] as i64;
        (cc - pc) as f64 / pot_p as f64
    };

    let mut single_entry_nodes = 0usize;
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut sample_mismatch_subtree: Option<(usize, &'static str)> = None;
    let _ = sample_mismatch_subtree;

    let mut first_mismatch: Option<(usize, usize, u8, f32, f32)> = None;

    let lean_csf = lean_cpu.cum_strategy_flop();
    let lean_cst = lean_cpu.cum_strategy_turn();
    let lean_csr = lean_cpu.cum_strategy_river();
    let rich_csf_lifted = rich_cpu_lifted.cum_strategy_flop();
    let rich_cst_lifted = rich_cpu_lifted.cum_strategy_turn();
    let rich_csr_lifted = rich_cpu_lifted.cum_strategy_river();

    let lean_turn_local = lean_cpu.turn_local_offset();
    let rich_turn_local = rich_cpu_lifted.turn_local_offset();
    let lean_river_local = lean_cpu.river_local_offset();
    let rich_river_local = rich_cpu_lifted.river_local_offset();
    let lean_turn_stride = lean_cpu.turn_stride();
    let rich_turn_stride = rich_cpu_lifted.turn_stride();
    let lean_river_stride = lean_cpu.river_stride();
    let rich_river_stride = rich_cpu_lifted.river_stride();
    let lean_n_turn = if lean_turn_stride > 0 { lean_cst.len() / lean_turn_stride } else { 0 };
    let lean_max_n_river = if lean_river_stride > 0 && lean_n_turn > 0 {
        lean_csr.len() / (lean_n_turn * lean_river_stride)
    } else { 0 };
    let rich_max_n_river = if rich_river_stride > 0 && lean_n_turn > 0 {
        rich_csr_lifted.len() / (lean_n_turn * rich_river_stride)
    } else { 0 };

    // Per-zone fidelity check. Compares lifted slot to lean source slot
    // for every action with an exact (label, pot-fraction) match, for
    // single-entry-weight-1.0 PH map nodes only (where lifted = lean copy
    // is the expected invariant).
    let mut check_zone = |zone: solver_core::solver::flop_start_vector_cfr::Zone,
                          stride_iters: Box<dyn Fn() -> Box<dyn Iterator<Item = (usize, usize)>>>| {
        let _ = (zone, stride_iters);
    };
    let _ = check_zone;

    // FLOP zone
    for &nid in &rich_tree.decision_node_ids {
        let r_idx = nid as usize;
        if !matches!(rich_zones[r_idx],
            solver_core::solver::flop_start_vector_cfr::Zone::Flop) { continue; }
        let entries = &ph_map.rich_to_lean_nodes[r_idx];
        if entries.len() != 1 { continue; }
        let (l_idx, w) = entries[0];
        if (w - 1.0).abs() > 1e-6 { continue; }
        if !matches!(lean_zones[l_idx],
            solver_core::solver::flop_start_vector_cfr::Zone::Flop) { continue; }
        let r_local = rich_flop_local[r_idx];
        let l_local = lean_flop_local[l_idx];
        if r_local == usize::MAX || l_local == usize::MAX { continue; }
        single_entry_nodes += 1;
        let r_base = r_local * MAX_NA_POSTFLOP * nh;
        let l_base = l_local * MAX_NA_POSTFLOP * nh;
        let r_children = rich_tree.node_children(r_idx);
        let l_children = lean_tree.node_children(l_idx);
        let r_player = rich_tree.nodes[r_idx].player_id as usize;
        let l_player = lean_tree.nodes[l_idx].player_id as usize;
        let mut used = vec![false; l_children.len()];
        for (a_r, &rc) in r_children.iter().enumerate() {
            let rc_node = &rich_tree.nodes[rc as usize];
            let label = rc_node.action_label;
            let a_l = find_exact_match(
                rich_tree, lean_tree, r_idx, l_idx, rc as usize, l_children, r_player, l_player, label, &mut used, pf,
            );
            let a_l = match a_l { Some(i) => i, None => continue };
            for h in 0..nh {
                let lifted_v = rich_csf_lifted[r_base + a_r * nh + h];
                let source_v = lean_csf[l_base + a_l * nh + h];
                let d = (lifted_v - source_v).abs();
                if d > max_abs { max_abs = d; }
                if d > 1e-6 {
                    mismatches += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((r_idx, l_idx, label, lifted_v, source_v));
                    }
                }
            }
        }
    }
    eprintln!(
        "  Lift fidelity (flop, single-entry only):  {:>5} nodes checked, {:>5} mismatches, max_abs={:.3e}",
        single_entry_nodes, mismatches, max_abs);
    if let Some((r, l, lbl, lv, sv)) = first_mismatch {
        eprintln!("    First mismatch (flop): r_idx={} l_idx={} label={} lifted={:.3e} source={:.3e}", r, l, lbl, lv, sv);
    }

    // TURN zone — sum across tc since the per-(tc) breakdown isn't useful here
    let mut turn_nodes = 0usize;
    let mut turn_mism = 0usize;
    let mut turn_max = 0.0f32;
    for &nid in &rich_tree.decision_node_ids {
        let r_idx = nid as usize;
        if !matches!(rich_zones[r_idx],
            solver_core::solver::flop_start_vector_cfr::Zone::Turn) { continue; }
        let entries = &ph_map.rich_to_lean_nodes[r_idx];
        if entries.len() != 1 { continue; }
        let (l_idx, w) = entries[0];
        if (w - 1.0).abs() > 1e-6 { continue; }
        if !matches!(lean_zones[l_idx],
            solver_core::solver::flop_start_vector_cfr::Zone::Turn) { continue; }
        let r_local = rich_turn_local[r_idx];
        let l_local = lean_turn_local[l_idx];
        if r_local == usize::MAX || l_local == usize::MAX { continue; }
        turn_nodes += 1;
        let r_children = rich_tree.node_children(r_idx);
        let l_children = lean_tree.node_children(l_idx);
        let r_player = rich_tree.nodes[r_idx].player_id as usize;
        let l_player = lean_tree.nodes[l_idx].player_id as usize;
        for tc in 0..lean_n_turn {
            let r_base = tc * rich_turn_stride + r_local * MAX_NA_POSTFLOP * nh;
            let l_base = tc * lean_turn_stride + l_local * MAX_NA_POSTFLOP * nh;
            let mut used = vec![false; l_children.len()];
            for (a_r, &rc) in r_children.iter().enumerate() {
                let rc_node = &rich_tree.nodes[rc as usize];
                let label = rc_node.action_label;
                let a_l = find_exact_match(
                    rich_tree, lean_tree, r_idx, l_idx, rc as usize, l_children, r_player, l_player, label, &mut used, pf,
                );
                let a_l = match a_l { Some(i) => i, None => continue };
                for h in 0..nh {
                    let lifted_v = rich_cst_lifted[r_base + a_r * nh + h];
                    let source_v = lean_cst[l_base + a_l * nh + h];
                    let d = (lifted_v - source_v).abs();
                    if d > turn_max { turn_max = d; }
                    if d > 1e-6 { turn_mism += 1; }
                }
            }
        }
    }
    eprintln!(
        "  Lift fidelity (turn, single-entry only):  {:>5} nodes checked, {:>5} mismatches, max_abs={:.3e}",
        turn_nodes, turn_mism, turn_max);

    // RIVER zone
    let mut river_nodes = 0usize;
    let mut river_mism = 0usize;
    let mut river_max = 0.0f32;
    for &nid in &rich_tree.decision_node_ids {
        let r_idx = nid as usize;
        if !matches!(rich_zones[r_idx],
            solver_core::solver::flop_start_vector_cfr::Zone::River) { continue; }
        let entries = &ph_map.rich_to_lean_nodes[r_idx];
        if entries.len() != 1 { continue; }
        let (l_idx, w) = entries[0];
        if (w - 1.0).abs() > 1e-6 { continue; }
        if !matches!(lean_zones[l_idx],
            solver_core::solver::flop_start_vector_cfr::Zone::River) { continue; }
        let r_local = rich_river_local[r_idx];
        let l_local = lean_river_local[l_idx];
        if r_local == usize::MAX || l_local == usize::MAX { continue; }
        river_nodes += 1;
        let r_children = rich_tree.node_children(r_idx);
        let l_children = lean_tree.node_children(l_idx);
        let r_player = rich_tree.nodes[r_idx].player_id as usize;
        let l_player = lean_tree.nodes[l_idx].player_id as usize;
        for tc in 0..lean_n_turn {
            for rc_idx in 0..lean_max_n_river {
                let r_base = (tc * rich_max_n_river + rc_idx) * rich_river_stride
                             + r_local * MAX_NA_POSTFLOP * nh;
                let l_base = (tc * lean_max_n_river + rc_idx) * lean_river_stride
                             + l_local * MAX_NA_POSTFLOP * nh;
                let mut used = vec![false; l_children.len()];
                for (a_r, &rc) in r_children.iter().enumerate() {
                    let rc_node = &rich_tree.nodes[rc as usize];
                    let label = rc_node.action_label;
                    let a_l = find_exact_match(
                        rich_tree, lean_tree, r_idx, l_idx, rc as usize, l_children, r_player, l_player, label, &mut used, pf,
                    );
                    let a_l = match a_l { Some(i) => i, None => continue };
                    for h in 0..nh {
                        let lifted_v = rich_csr_lifted[r_base + a_r * nh + h];
                        let source_v = lean_csr[l_base + a_l * nh + h];
                        let d = (lifted_v - source_v).abs();
                        if d > river_max { river_max = d; }
                        if d > 1e-6 { river_mism += 1; }
                    }
                }
            }
        }
    }
    eprintln!(
        "  Lift fidelity (river, single-entry only): {:>5} nodes checked, {:>5} mismatches, max_abs={:.3e}",
        river_nodes, river_mism, river_max);

    // Also count UNPAIRED nodes per zone (rich nodes with empty PH map).
    let mut unpaired_flop = 0usize;
    let mut unpaired_turn = 0usize;
    let mut unpaired_river = 0usize;
    for &nid in &rich_tree.decision_node_ids {
        let r_idx = nid as usize;
        if !ph_map.rich_to_lean_nodes[r_idx].is_empty() { continue; }
        match rich_zones[r_idx] {
            solver_core::solver::flop_start_vector_cfr::Zone::Flop => unpaired_flop += 1,
            solver_core::solver::flop_start_vector_cfr::Zone::Turn => unpaired_turn += 1,
            solver_core::solver::flop_start_vector_cfr::Zone::River => unpaired_river += 1,
            _ => {}
        }
    }
    eprintln!(
        "  Unpaired rich PLAYER nodes (uniform-default in lift): flop={}  turn={}  river={}",
        unpaired_flop, unpaired_turn, unpaired_river);
}

fn find_exact_match<F>(
    rich_tree: &FlatTree,
    lean_tree: &FlatTree,
    r_parent: usize,
    l_parent: usize,
    r_child: usize,
    l_children: &[u32],
    r_player: usize,
    l_player: usize,
    label: u8,
    used: &mut [bool],
    pf: F,
) -> Option<usize>
where F: Fn(&FlatTree, usize, usize, usize) -> f64 + Copy,
{
    use solver_core::tree::flat::{ACTION_LABEL_BET, ACTION_LABEL_RAISE};
    if label != ACTION_LABEL_BET && label != ACTION_LABEL_RAISE {
        for (a_l, &lc) in l_children.iter().enumerate() {
            if used[a_l] { continue; }
            if lean_tree.nodes[lc as usize].action_label == label {
                used[a_l] = true;
                return Some(a_l);
            }
        }
        return None;
    }
    let pf_r = pf(rich_tree, r_parent, r_child, r_player);
    for (a_l, &lc) in l_children.iter().enumerate() {
        if used[a_l] { continue; }
        let lc_node = &lean_tree.nodes[lc as usize];
        if lc_node.action_label != label { continue; }
        let pf_l = pf(lean_tree, l_parent, lc as usize, l_player);
        if (pf_l - pf_r).abs() < 1e-3 {
            used[a_l] = true;
            return Some(a_l);
        }
    }
    None
}

/// Solve lean to a self-expl floor instead of a fixed iter count. K=3
/// in the first-pass sweep was less converged than K=4 (0.0279% vs 0.0018%)
/// because of variation in how fast different action abstractions reach
/// equilibrium; running to a common self-expl threshold normalizes the
/// comparison.
fn solve_lean_to_floor(
    tree: &FlatTree,
    game: &FlopStartGame,
) -> (FlopStartVectorCfr, u32, f32) {
    let np = tree.num_players as usize;
    let mut cpu = FlopStartVectorCfr::new(tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, tree, game, &cpu);
    let mut total_iters = 0u32;
    let mut last_pct = f32::INFINITY;
    while total_iters < LEAN_MAX_ITERS {
        gpu.run(&ctx, tree, game, LEAN_ITER_CHUNK);
        cpu.run(tree, game, LEAN_ITER_CHUNK);
        total_iters += LEAN_ITER_CHUNK;
        let mut total = 0.0f32;
        for p in 0..np {
            let br = cpu.best_response_value_debug(tree, game, p as u8);
            let sv = cpu.strategy_value_debug(tree, game, p as u8);
            for h in 0..br.len().min(sv.len()) {
                total += (br[h] - sv[h]).max(0.0);
            }
        }
        last_pct = total / STARTING_POT as f32 * 100.0;
        if last_pct < LEAN_SELF_EXPL_FLOOR_PCT { break; }
    }
    (cpu, total_iters, last_pct)
}

/// Read flattened cum_strategy from a solved cpu (for the bet-size observer only).
fn get_flattened(cpu: &FlopStartVectorCfr, tree: &FlatTree) -> (Vec<f32>, Vec<usize>) {
    cpu.flattened_cum_strategy(tree)
}

fn measure_rich_exploitability_pct(
    cpu: &FlopStartVectorCfr,
    tree: &FlatTree,
    game: &FlopStartGame,
) -> f32 {
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

#[test]
#[ignore = "Phase 4 redo measurement — solves rich + lean, lifts, computes cross-action-space exploit (~30 min)"]
fn phase4_redo_measurement() {
    eprintln!("\n=== Phase 4 REDO measurement ===");
    eprintln!("Config: deep wet stacks={} board=Th9d8c chance=2x2 nh={} np={} iters={}",
        STACKS, NH, NP, N_ITERS);

    // ── Build chance table & rich tree ──
    let table = build_chance_table();
    let rich_tree = build_tree_with(rich_bet_sizes(), rich_raise_sizes());
    eprintln!("\nRich tree: {} nodes, {} infosets",
        rich_tree.num_nodes(), rich_tree.num_infosets);

    // ── Solve rich to the SAME self-expl floor as lean. The first-pass
    // measurement ran rich for a fixed 50 iters (self-expl 0.13%) while
    // lean was run to a floor of 0.005%, so the lifted-lean strategies
    // could appear more converged than rich itself, producing negative
    // costs at the identity boundary (K_FULL lifted-rich showed xt = 0%
    // but rich_self_expl = 0.13%, giving cost = −0.13% which is
    // measurement noise, not a real result). Running rich to the same
    // floor drives the identity boundary to exactly 0%.
    eprintln!("\n── Solve RICH (to self-expl floor < {:.4}%) ──", LEAN_SELF_EXPL_FLOOR_PCT);
    let t0 = Instant::now();
    let rich_game = FlopStartGame::new(table);
    let (rich_cpu, rich_iters, rich_pct) = solve_lean_to_floor(&rich_tree, &rich_game);
    eprintln!("Rich solve: {} iters, {:.1}s; rich self-expl: {:.4}% pot",
        rich_iters, t0.elapsed().as_secs_f32(), rich_pct);
    assert!(rich_pct < LEAN_SELF_EXPL_FLOOR_PCT * 5.0,
        "Rich didn't converge to a comparable floor: {:.4}% vs lean target {:.4}%",
        rich_pct, LEAN_SELF_EXPL_FLOOR_PCT);

    // Sanity check: lift rich's strategy back into a fresh rich solver and
    // measure exploit. Should equal rich_pct because rich → rich is identity
    // (every node pairs with itself, every action with itself).
    {
        let rich_self_map = build_action_map(&rich_tree, &rich_tree);
        let table_id = build_chance_table();
        let rich_game_id = FlopStartGame::new(table_id);
        let mut rich_cpu_id = FlopStartVectorCfr::new(&rich_tree, &rich_game_id.table());
        lift_into_rich_solver_with_lean(&rich_tree, &rich_tree, &rich_self_map, &rich_cpu, &mut rich_cpu_id);

        // ── Layout sanity: are source/dest local-offset tables identical? ──
        let lo_match = rich_cpu.flop_local_offset() == rich_cpu_id.flop_local_offset();
        let to_match = rich_cpu.turn_local_offset() == rich_cpu_id.turn_local_offset();
        let ro_match = rich_cpu.river_local_offset() == rich_cpu_id.river_local_offset();
        let z_match = rich_cpu.zones() == rich_cpu_id.zones();
        eprintln!("Layout: flop_local match={} turn_local match={} river_local match={} zones match={}",
            lo_match, to_match, ro_match, z_match);
        if !lo_match || !to_match || !ro_match || !z_match {
            // Find first divergent index per table for triage.
            for (i, (a, b)) in rich_cpu.flop_local_offset().iter()
                .zip(rich_cpu_id.flop_local_offset().iter()).enumerate() {
                if a != b { eprintln!("  first flop_local divergence: idx={} src={} dst={}", i, a, b); break; }
            }
        }

        // ── Diagnostic: are the cum_strategy buffers actually equal after identity lift? ──
        let f1 = rich_cpu.cum_strategy_flop();
        let f2 = rich_cpu_id.cum_strategy_flop();
        let t1 = rich_cpu.cum_strategy_turn();
        let t2 = rich_cpu_id.cum_strategy_turn();
        let r1 = rich_cpu.cum_strategy_river();
        let r2 = rich_cpu_id.cum_strategy_river();
        eprintln!("Buffer-equality diagnostic after identity lift:");
        eprintln!("  flop:  rich.len={}, id.len={}", f1.len(), f2.len());
        eprintln!("  turn:  rich.len={}, id.len={}", t1.len(), t2.len());
        eprintln!("  river: rich.len={}, id.len={}", r1.len(), r2.len());

        fn buf_diff(name: &str, a: &[f32], b: &[f32]) -> (usize, f64, f64) {
            let mut diffs = 0usize;
            let mut max_abs = 0.0f64;
            let mut sum_a = 0.0f64;
            let mut sum_b = 0.0f64;
            let n = a.len().min(b.len());
            for i in 0..n {
                sum_a += a[i] as f64;
                sum_b += b[i] as f64;
                if a[i] != b[i] {
                    diffs += 1;
                    let d = (a[i] as f64 - b[i] as f64).abs();
                    if d > max_abs { max_abs = d; }
                }
            }
            eprintln!("  {} sum_rich={:.3e} sum_id={:.3e} diffs={}/{} max_abs={:.3e}",
                name, sum_a, sum_b, diffs, n, max_abs);
            (diffs, max_abs, sum_a)
        }
        let (fd, _, fs) = buf_diff("flop", f1, f2);
        let (td, _, ts) = buf_diff("turn", t1, t2);
        let (rd, _, rs) = buf_diff("river", r1, r2);
        let total_diffs = fd + td + rd;
        eprintln!("  Total nonzero entries in rich (proxy for accumulation): flop_sum={:.3e} turn_sum={:.3e} river_sum={:.3e}", fs, ts, rs);

        let id_pct = measure_rich_exploitability_pct(&rich_cpu_id, &rich_tree, &rich_game_id);
        eprintln!("Identity-lift sanity check: id_pct={:.4}% rich_pct={:.4}% (should be ≈ equal)", id_pct, rich_pct);

        // ── Brute-copy bypass: overwrite the entire buffers verbatim, bypassing
        // my iteration logic. If BR matches now, my lift iteration misses
        // entries. If BR still doesn't match, something other than cum_strategy
        // matters for BR. ──
        {
            let src_f = rich_cpu.cum_strategy_flop().to_vec();
            let src_t = rich_cpu.cum_strategy_turn().to_vec();
            let src_r = rich_cpu.cum_strategy_river().to_vec();
            rich_cpu_id.cum_strategy_flop_mut().copy_from_slice(&src_f);
            rich_cpu_id.cum_strategy_turn_mut().copy_from_slice(&src_t);
            rich_cpu_id.cum_strategy_river_mut().copy_from_slice(&src_r);
            let brute_pct = measure_rich_exploitability_pct(&rich_cpu_id, &rich_tree, &rich_game_id);
            eprintln!("BRUTE-COPY BR result: {:.4}% pot (rich self-pct = {:.4}%)", brute_pct, rich_pct);
            if (brute_pct - rich_pct).abs() / rich_pct.max(1e-6) < 0.05 {
                eprintln!("  → BRUTE-COPY MATCHES — bug is in my lift iteration");
            } else {
                eprintln!("  → BRUTE-COPY STILL OFF — bug is downstream (game state, regrets, or strategy buffer)");
            }
        }
        eprintln!("  Buffer-equality verdict: {}",
            if total_diffs == 0 { "BUFFERS IDENTICAL — bug is downstream of lift" }
            else { "BUFFERS DIFFER — bug is in lift" });

        let gap = (id_pct - rich_pct).abs() / rich_pct.max(1e-6);
        assert!(gap < 0.05,
            "Identity-lift sanity FAILED: {:.4}% vs {:.4}% ({:.1}% relative gap, > 5%) — \
             per-outcome lift is broken, do not trust K-sweep numbers. \
             total_diffs={}", id_pct, rich_pct, gap * 100.0, total_diffs);
    }

    // ── Observe rich's bet-size usage ──
    eprintln!("\n── P1: Empirical bet-size observation ──");
    let (rich_buf, rich_buf_offsets) = get_flattened(&rich_cpu, &rich_tree);
    let observations = observe_bet_sizes(&rich_tree, &rich_buf, &rich_buf_offsets);
    eprintln!("Bucketed bet-size usage (sorted by conditional mass):");
    eprintln!("  {:>10}  {:>8}  {:>12}  {:>10}", "pot_frac", "snap", "cond_mass", "count");
    for o in &observations {
        let snap = snap_to_rich_size(o.pot_fraction);
        let snap_str = snap.map(|s| format!("{:.2}", s)).unwrap_or_else(|| "—".to_string());
        eprintln!("  {:>10.3}  {:>8}  {:>11.2}%  {:>10}",
            o.pot_fraction, snap_str,
            o.conditional_mass * 100.0,
            o.occurrence_count);
    }

    // Build ordered list of unique rich sizes ranked by observed usage.
    let mut ranked_sizes: Vec<f32> = Vec::new();
    let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for o in &observations {
        if let Some(s) = snap_to_rich_size(o.pot_fraction) {
            let key = (s * 100.0).round() as i32;
            if seen.insert(key) {
                ranked_sizes.push(s);
            }
        }
    }
    eprintln!("\nRanked rich bet sizes by observed usage: {:?}", ranked_sizes);

    // ── P3: K sweep ──
    eprintln!("\n── P3: K sweep (each K builds + solves a lean tree, then lifts + BRs in rich) ──");
    eprintln!("Lean set always includes raise [1.0]; bet sizes = top-K of ranked_sizes.\n");

    let max_k = ranked_sizes.len().min(4);
    let mut k_results: Vec<(usize, Vec<f32>, f32, f32, f32, usize)> = Vec::new(); // (K, sizes, lean_self_pct, xt_pct, cost_pct, nodes)
    // K_FULL = K=4 with both raises [1.0, 2.0] — should give ~0% cost (lean
    // is effectively rich); sanity check that the measurement converges at
    // the identity boundary.
    let configs: Vec<(usize, Vec<BetSize>, Vec<BetSize>, &str)> = (1..=max_k).map(|k| {
        let bets: Vec<BetSize> = ranked_sizes[..k].iter().map(|&f| BetSize::PotRelative(f as f64)).collect();
        let raises = vec![BetSize::PotRelative(1.0)];
        (k, bets, raises, "1 raise")
    }).chain(std::iter::once((
        max_k + 1,
        ranked_sizes[..max_k].iter().map(|&f| BetSize::PotRelative(f as f64)).collect(),
        vec![BetSize::PotRelative(1.0), BetSize::PotRelative(2.0)],
        "K_FULL (rich raises)",
    ))).collect();

    for (k, lean_bets, lean_raises, raise_label) in configs {
        eprintln!("── K={}: bets={:?} raises={} ──", k, lean_bets, raise_label);

        let lean_tree = build_tree_with(lean_bets.clone(), lean_raises.clone());
        eprintln!("  Lean tree: {} nodes, {} infosets ({:.2}× rich)",
            lean_tree.num_nodes(),
            lean_tree.num_infosets,
            lean_tree.num_nodes() as f32 / rich_tree.num_nodes() as f32);
        let lean_bets_for_record: Vec<f32> = lean_bets.iter().map(|b| match b {
            BetSize::PotRelative(f) => *f as f32,
            _ => 0.0,
        }).collect();

        // Sanity: validate cross-tree pairing on this lean.
        let map = build_action_map(&rich_tree, &lean_tree);
        let lean_paired = map.paired_player_count(&rich_tree);
        eprintln!("  Cross-tree pair: {}/{} rich PLAYER nodes paired with a lean node",
            lean_paired, rich_tree.decision_node_ids.len());

        // Solve lean to a self-expl FLOOR (not fixed iter count) so every K
        // is compared at the same convergence quality.
        let t_lean = Instant::now();
        let table_lean = build_chance_table();
        let lean_game = FlopStartGame::new(table_lean);
        let (lean_cpu, lean_iters, lean_self_pct) = solve_lean_to_floor(&lean_tree, &lean_game);
        let lean_wall = t_lean.elapsed().as_secs_f32();
        eprintln!("  Lean solve: {} iters, {:.1}s; lean self-expl (lean-space): {:.4}% pot",
            lean_iters, lean_wall, lean_self_pct);

        // ── Cross-action-space cost with PSEUDO-HARMONIC translation (Pluribus
        // mapping, Ganzfried-Sandholm 2013). Pot-fraction matching at every
        // node + probabilistic split at unmatched bets/raises. ──
        let table_lift_ph = build_chance_table();
        let rich_game_lift_ph = FlopStartGame::new(table_lift_ph);
        let mut rich_cpu_lifted_ph = FlopStartVectorCfr::new(&rich_tree, &rich_game_lift_ph.table());
        let ph_map = build_pseudo_harmonic_map(&rich_tree, &lean_tree);
        eprintln!("  PH map: {} paired rich PLAYER nodes, max {} entries/node, {} total entries",
            ph_map.paired_player_count(&rich_tree),
            ph_map.max_entries_per_node(),
            ph_map.total_entries());
        lift_into_rich_solver_pseudo_harmonic(
            &rich_tree, &lean_tree, &ph_map, &lean_cpu, &mut rich_cpu_lifted_ph);

        // ── Bug-hunt diagnostic: for every rich PLAYER node whose PH map
        // has a single entry (weight 1.0), the lifted cum_strategy slot
        // values should be bit-exact equal to lean's source slot values
        // for actions that exact-match between the trees. Divergence here
        // localizes a lift bug to the affected subtree (e.g. K=3 vs K=4
        // inversion's residual-bug-at-bet-1.5p-subtree hypothesis). ──
        diagnose_single_entry_lift_fidelity(
            &rich_tree, &lean_tree, &ph_map, &lean_cpu, &rich_cpu_lifted_ph,
        );

        let xt_ph_pct = measure_rich_exploitability_pct(&rich_cpu_lifted_ph, &rich_tree, &rich_game_lift_ph);
        eprintln!("  xt (pseudo-harmonic): {:.4}% pot", xt_ph_pct);

        // ── Compare to NEAREST translation (the previous baseline, kept for
        // separating translation-error attribution from structural cost). ──
        let table_lift_n = build_chance_table();
        let rich_game_lift_n = FlopStartGame::new(table_lift_n);
        let mut rich_cpu_lifted_n = FlopStartVectorCfr::new(&rich_tree, &rich_game_lift_n.table());
        lift_into_rich_solver_with_lean(&rich_tree, &lean_tree, &map, &lean_cpu, &mut rich_cpu_lifted_n);
        let xt_pct = measure_rich_exploitability_pct(&rich_cpu_lifted_n, &rich_tree, &rich_game_lift_n);
        eprintln!("  xt (nearest):         {:.4}% pot", xt_pct);

        // Cost-of-leaning uses the pseudo-harmonic number (production-grade
        // action translation) as the canonical bank.
        let cost_pct = xt_ph_pct - rich_pct;
        eprintln!("  Cross-action-space expl (lifted lean in RICH): {:.4}% pot", xt_pct);
        eprintln!("  COST of leaning = xt - rich = {:.4}% pot", cost_pct);

        k_results.push((k, lean_bets_for_record, lean_self_pct, xt_pct, cost_pct, lean_tree.num_nodes()));
        eprintln!();
    }

    // ── Verdict ──
    eprintln!("\n=== Phase 4 REDO summary ===");
    eprintln!("Rich self-exploitability: {:.4}% pot at iter {}", rich_pct, N_ITERS);
    eprintln!();
    eprintln!("{:>3} {:>4} {:>30} {:>14} {:>14} {:>14}",
        "K", "MAX_NA",  "lean_bets", "lean_self_%", "xt_%", "cost_%");
    for (k, sizes, lean_self, xt, cost, _) in &k_results {
        // na_total = K bets + fold + check/call + raise (always 1) = K + 3
        let max_na = *k + 3;
        eprintln!("{:>3} {:>4} {:>30} {:>13.4}% {:>13.4}% {:>13.4}%",
            k, max_na, format!("{:?}", sizes), lean_self, xt, cost);
    }

    let target_pct = 0.05f32; // tight production target above rich
    let mut chosen: Option<&(usize, Vec<f32>, f32, f32, f32, usize)> = None;
    for r in &k_results {
        if r.4 <= target_pct {
            chosen = Some(r);
            break;
        }
    }
    eprintln!();
    match chosen {
        Some(r) => {
            let max_na = r.0 + 3;
            eprintln!("VERDICT: Smallest K with cost ≤ {:.2}% pot is K={} → MAX_NA_POSTFLOP = {}",
                target_pct, r.0, max_na);
            eprintln!("Empirical sizes: {:?}", r.1);
        }
        None => {
            eprintln!("VERDICT: No K within tested range hit the {:.2}% pot cost gate.",
                target_pct);
            eprintln!("Either widen K (more bet sizes) or loosen target. MAX_NA_POSTFLOP > 4+3 = 7.");
        }
    }
}
