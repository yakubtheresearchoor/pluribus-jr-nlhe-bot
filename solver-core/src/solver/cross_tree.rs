//! Cross-tree action-space mapping for evaluating a strategy from one
//! action abstraction against a different (typically richer) one.
//!
//! Motivating use case: Phase 4 redo of MAX_NA_POSTFLOP. The lean action
//! set's quality cannot be evaluated by same-action-space exploitability —
//! by construction the dropped bet sizes don't exist in the lean game, so
//! "lean strategy exploits in lean game" can't see the cost of those drops.
//! The right metric is "lean strategy exploits in RICH game, where the
//! best-response opponent can use the dropped sizes" — i.e. how much does
//! a rich-space adversary punish the lean strategy.
//!
//! This module provides three pieces:
//!
//! 1. [`build_action_map`] — pair every node in the rich tree with its
//!    counterpart in the lean tree by walking both in lockstep. Matching
//!    is by (action_label, amount) at PLAYER nodes and by outcome index
//!    at CHANCE nodes. Lean's actions must be a strict subset of rich's
//!    available actions at every paired player node; un-paired rich nodes
//!    sit on subtrees the lean strategy never reaches.
//!
//! 2. [`lift_strategy`] — take a flattened cum_strategy buffer from the
//!    lean solver and lay it out against the RICH tree's infoset offsets.
//!    Lean actions copy their values into the matching rich action slot;
//!    rich-only actions get zero (lean strategy never plays them, so the
//!    normalization in `StrategyProfile::get_strategy` will assign them
//!    probability 0).
//!
//! 3. [`compute_rich_offsets`] — replicate the offset layout used by
//!    `FlopStartVectorCfr::flattened_cum_strategy` so that `StrategyProfile`
//!    reads the lifted buffer correctly: `offset(N) = info_idx(N) * MAX_NA_POSTFLOP * nh`.
//!
//! Together: the lifted buffer + rich offsets feed into
//! `StrategyProfile::from_usize_offsets(...)`, and `best_response::exploitability`
//! gives the cross-action-space cost-of-leaning number.

use crate::tree::flat::{
    FlatTree, MAX_NA_POSTFLOP, ACTION_LABEL_BET, ACTION_LABEL_RAISE,
};
use crate::solver::flop_start_vector_cfr::{FlopStartVectorCfr, Zone};

const NO_LOCAL_OFFSET: usize = usize::MAX;

/// Result of pairing rich-tree nodes to lean-tree nodes.
///
/// `rich_to_lean_node[r_idx] = Some(l_idx)` when rich node r_idx has a
/// paired lean node l_idx (reachable in lean's tree). `None` means lean
/// never reaches this subtree (rich-only action above it).
///
/// `rich_action_to_lean[r_idx] = vec![Some(lean_action_idx); num_rich_actions]`
/// for paired PLAYER nodes. None entries are rich actions with no lean
/// counterpart at this infoset.
pub struct CrossTreeMap {
    pub rich_to_lean_node: Vec<Option<usize>>,
    pub rich_action_to_lean: Vec<Vec<Option<usize>>>,
}

impl CrossTreeMap {
    /// Number of rich PLAYER nodes that paired with a lean node.
    pub fn paired_player_count(&self, rich: &FlatTree) -> usize {
        let mut n = 0;
        for (idx, slot) in self.rich_to_lean_node.iter().enumerate() {
            if slot.is_some() && rich.nodes[idx].is_player() { n += 1; }
        }
        n
    }

    /// Number of rich PLAYER actions that paired with a lean action,
    /// summed over all paired player nodes.
    pub fn paired_action_count(&self, rich: &FlatTree) -> usize {
        let mut n = 0;
        for (idx, slot) in self.rich_to_lean_node.iter().enumerate() {
            if slot.is_none() || !rich.nodes[idx].is_player() { continue; }
            for a in &self.rich_action_to_lean[idx] {
                if a.is_some() { n += 1; }
            }
        }
        n
    }
}

/// Pair every reachable lean node with its rich counterpart by walking both
/// trees from the root in lockstep.
///
/// PRECONDITION: rich and lean were built from the same `FlopChanceTable`
/// with the same `num_players`, `starting_pot`, `starting_stacks`,
/// `initial_contributions`, and `initial_state`. Otherwise CHANCE-node
/// pairing breaks (different outcome counts) and the resulting map is
/// unsound.
///
/// PRECONDITION: lean's action set is a SUBSET of rich's at every paired
/// player node. The pairing matches by exact (action_label, amount); a
/// lean action with no rich match is a config bug and will be returned
/// as a None on the lean side (counted as un-paired). Callers should
/// `assert!(map.unpaired_lean_actions(lean) == 0)` for a clean lift.
pub fn build_action_map(rich: &FlatTree, lean: &FlatTree) -> CrossTreeMap {
    assert_eq!(rich.num_players, lean.num_players,
        "rich and lean must have same num_players for cross-tree map");
    let n_rich = rich.num_nodes();
    let mut rich_to_lean_node = vec![None; n_rich];
    let mut rich_action_to_lean: Vec<Vec<Option<usize>>> = (0..n_rich)
        .map(|i| {
            let n = rich.nodes[i].num_children as usize;
            vec![None; n]
        })
        .collect();
    rich_to_lean_node[0] = Some(0);
    walk_pair(rich, lean, 0, 0, &mut rich_to_lean_node, &mut rich_action_to_lean);
    CrossTreeMap { rich_to_lean_node, rich_action_to_lean }
}

fn walk_pair(
    rich: &FlatTree,
    lean: &FlatTree,
    r_idx: usize,
    l_idx: usize,
    node_map: &mut [Option<usize>],
    action_map: &mut [Vec<Option<usize>>],
) {
    let r_node = &rich.nodes[r_idx];
    let l_node = &lean.nodes[l_idx];

    if r_node.is_terminal() || l_node.is_terminal() {
        return;
    }

    let r_children = rich.node_children(r_idx).to_vec();
    let l_children = lean.node_children(l_idx).to_vec();

    if r_node.is_chance() {
        // CHANCE nodes pair 1:1 by outcome index (same chance table).
        // If the tree builders produced different child counts, that's a
        // config-precondition violation; abort rather than silently mismap.
        assert_eq!(r_children.len(), l_children.len(),
            "CHANCE node child count mismatch: rich node {} has {} children, \
             lean node {} has {} — chance table mismatch?",
            r_idx, r_children.len(), l_idx, l_children.len());
        for (i, (&rc, &lc)) in r_children.iter().zip(l_children.iter()).enumerate() {
            let _ = i;
            node_map[rc as usize] = Some(lc as usize);
            walk_pair(rich, lean, rc as usize, lc as usize, node_map, action_map);
        }
        return;
    }

    // PLAYER node: pair rich actions to lean actions.
    //
    // Pass 1 — exact (action_label, amount) match, with claimed-lean tracking
    // so duplicate-key children (e.g. dead-allin chains with amount=0) pair
    // in order rather than collapsing onto the first match.
    //
    // Pass 2 — ACTION TRANSLATION: rich-only BET/RAISE actions that didn't
    // exact-match get redirected to the nearest-by-amount lean action with
    // the SAME label. This is the production deployment model (Pluribus-
    // style): when the deployed lean strategy faces an opponent action it
    // didn't train on, it interprets it as the closest lean action and uses
    // the corresponding lean response. Without this, the lift fills rich-
    // only subtrees with uniform, which catastrophically overestimates the
    // cost-of-leaning (measured 65%+ pot at K=4 before fix).
    //
    // Translation does NOT consume a lean slot — multiple rich actions can
    // redirect to the same lean action, which is the whole point.
    let mut used = vec![false; l_children.len()];
    for (a_r, &rc) in r_children.iter().enumerate() {
        let rc_node = &rich.nodes[rc as usize];
        let key = (rc_node.action_label, rc_node.amount);
        let mut found: Option<usize> = None;
        for (a_l, &lc) in l_children.iter().enumerate() {
            if used[a_l] { continue; }
            let lc_node = &lean.nodes[lc as usize];
            if (lc_node.action_label, lc_node.amount) == key {
                found = Some(a_l);
                used[a_l] = true;
                node_map[rc as usize] = Some(lc as usize);
                walk_pair(rich, lean, rc as usize, lc as usize, node_map, action_map);
                break;
            }
        }
        action_map[r_idx][a_r] = found;
    }
    // Pass 2: action translation for rich-only BET/RAISE.
    for (a_r, &rc) in r_children.iter().enumerate() {
        if action_map[r_idx][a_r].is_some() { continue; }
        let rc_node = &rich.nodes[rc as usize];
        if rc_node.action_label != ACTION_LABEL_BET && rc_node.action_label != ACTION_LABEL_RAISE {
            continue;
        }
        let mut best: Option<(usize, i64)> = None;
        for (a_l, &lc) in l_children.iter().enumerate() {
            let lc_node = &lean.nodes[lc as usize];
            if lc_node.action_label != rc_node.action_label { continue; }
            let d = (lc_node.amount as i64 - rc_node.amount as i64).abs();
            if best.map(|(_, b)| d < b).unwrap_or(true) {
                best = Some((a_l, d));
            }
        }
        if let Some((a_l, _)) = best {
            let lc = l_children[a_l];
            action_map[r_idx][a_r] = Some(a_l);
            // Only set node_map if not already set by a deeper exact-match
            // recursion (shouldn't happen here but defensive).
            if node_map[rc as usize].is_none() {
                node_map[rc as usize] = Some(lc as usize);
                walk_pair(rich, lean, rc as usize, lc as usize, node_map, action_map);
            }
        }
    }
}

/// Replicate the offset layout used by
/// `FlopStartVectorCfr::flattened_cum_strategy`: a per-rich-node table where
/// `offsets[N] = infoset_idx(N) * MAX_NA_POSTFLOP * nh` for every player
/// node, and 0 for non-player nodes (unused by `StrategyProfile`).
///
/// `nh` is the number of hands per player (must match the value the
/// solver was run with).
pub fn compute_rich_offsets(rich: &FlatTree, nh: usize) -> Vec<usize> {
    let n = rich.num_nodes();
    let mut offsets = vec![0usize; n];
    for &nid in &rich.decision_node_ids {
        let info_idx = rich.infoset_offsets[nid as usize] as usize;
        offsets[nid as usize] = info_idx * MAX_NA_POSTFLOP * nh;
    }
    offsets
}

/// Lift a lean cum_strategy buffer into the rich action space.
///
/// `lean_buffer` and `lean_offsets` come from
/// `FlopStartVectorCfr::flattened_cum_strategy(&lean_tree)`. The result is
/// a fresh buffer indexed by `rich_offsets` (from `compute_rich_offsets`)
/// so it can be wrapped in a `StrategyProfile::from_usize_offsets`.
///
/// At every paired rich player node, each rich action a_R either:
///   - has a matching lean action a_L → its slot copies from lean's slot
///   - has no match → its slot stays at 0 (lean strategy never picks this)
///
/// Un-paired rich nodes (rich-only subtrees) have all-zero strategy slots.
/// `StrategyProfile::get_strategy` will return uniform there because the
/// total is zero; that's fine — lean's lifted reach at those nodes is zero,
/// so the choice of fallback strategy doesn't affect EV.
pub fn lift_strategy(
    rich: &FlatTree,
    lean: &FlatTree,
    map: &CrossTreeMap,
    lean_buffer: &[f32],
    lean_offsets: &[usize],
    rich_offsets: &[usize],
    nh: usize,
) -> Vec<f32> {
    let buf_size = (rich.num_infosets as usize) * MAX_NA_POSTFLOP * nh;
    let mut lifted = vec![0.0f32; buf_size];

    for &nid in &rich.decision_node_ids {
        let r_idx = nid as usize;
        let l_idx = match map.rich_to_lean_node[r_idx] {
            Some(i) => i,
            None => continue, // rich-only subtree, leave zero
        };
        if !lean.nodes[l_idx].is_player() { continue; }
        let r_off = rich_offsets[r_idx];
        let l_off = lean_offsets[l_idx];
        let actions_at_r = rich.nodes[r_idx].num_children as usize;
        for a_r in 0..actions_at_r {
            let a_l = match map.rich_action_to_lean[r_idx][a_r] {
                Some(i) => i,
                None => continue, // rich-only action; lean has no equivalent
            };
            for h in 0..nh {
                lifted[r_off + a_r * nh + h] = lean_buffer[l_off + a_l * nh + h];
            }
        }
    }

    lifted
}

/// Lift lean's PER-OUTCOME cum_strategy buffers directly into a fresh
/// rich-tree `FlopStartVectorCfr`'s internal per-outcome buffers.
///
/// Why this exists instead of using the flattened lift + `best_response.rs`
/// walker: the public `best_response::walk_value` averages strategies
/// across chance outcomes and has no notion of (tc, rc) — so for a
/// `FlopStartGame` (which carries distinct per-(turn_card, river_card)
/// strategy state), the public walker is fundamentally wrong (measured
/// 58× mismatch on the identity sanity check). The solver-internal
/// `walk_br`/`walk_sv` correctly track (tc, rc) and read per-outcome
/// strategies, so the right move is to populate a fresh rich solver's
/// per-outcome buffers with the lifted lean strategy and use ITS BR walker.
///
/// PRECONDITION: rich and lean were built from the same chance table
/// (so `n_turn` and `max_n_river` match) and the cross-tree map was
/// produced from these exact trees.
///
/// Layout reference (from flop_start_vector_cfr.rs get_strategy_for_outcome):
///   Flop:  cum_strategy_flop[local * MAX_NA * nh + a*nh + h]
///   Turn:  cum_strategy_turn[tc * turn_stride + local * MAX_NA * nh + a*nh + h]
///   River: cum_strategy_river[(tc * max_n_river + rc) * river_stride
///                            + local * MAX_NA * nh + a*nh + h]
/// Find the lean action index whose (action_label, amount) exact-matches
/// the rich action `r_action_node`. Returns None if no exact match at this
/// lean node (rich-only action).
fn find_exact_action_in_lean(
    lean: &FlatTree,
    lean_children: &[u32],
    rich_action_node: &crate::tree::flat::FlatNode,
) -> Option<usize> {
    let key = (rich_action_node.action_label, rich_action_node.amount);
    for (a_l, &lc) in lean_children.iter().enumerate() {
        let lc_node = &lean.nodes[lc as usize];
        if (lc_node.action_label, lc_node.amount) == key {
            return Some(a_l);
        }
    }
    None
}

pub fn lift_into_rich_solver(
    rich_tree: &FlatTree,
    map: &CrossTreeMap,
    lean_cpu: &FlopStartVectorCfr,
    rich_cpu_lifted: &mut FlopStartVectorCfr,
) {
    lift_into_rich_solver_with_lean_tree(rich_tree, map, lean_cpu, rich_cpu_lifted, None)
}

/// Lift, but with the lean FlatTree threaded through so we can do per-node
/// exact-match action lookup. If `lean_tree` is None, falls back to the
/// (incorrect) action_map-based lift kept only for back-compat — DO NOT USE
/// for measurements, it inflates probabilities at translated subtrees.
pub fn lift_into_rich_solver_with_lean(
    rich_tree: &FlatTree,
    lean_tree: &FlatTree,
    map: &CrossTreeMap,
    lean_cpu: &FlopStartVectorCfr,
    rich_cpu_lifted: &mut FlopStartVectorCfr,
) {
    lift_into_rich_solver_with_lean_tree(rich_tree, map, lean_cpu, rich_cpu_lifted, Some(lean_tree))
}

fn lift_into_rich_solver_with_lean_tree(
    rich_tree: &FlatTree,
    map: &CrossTreeMap,
    lean_cpu: &FlopStartVectorCfr,
    rich_cpu_lifted: &mut FlopStartVectorCfr,
    lean_tree_opt: Option<&FlatTree>,
) {
    let nh = lean_cpu.num_hands();
    assert_eq!(nh, rich_cpu_lifted.num_hands(),
        "rich and lean must use same nh for cross-tree lift");

    // Derive n_turn and max_n_river from buffer sizes (no public accessor).
    // turn buffer = n_turn * turn_stride; river buffer = n_turn * max_n_river * river_stride.
    let lean_turn_stride = lean_cpu.turn_stride();
    let lean_river_stride = lean_cpu.river_stride();
    let lean_n_turn = if lean_turn_stride > 0 {
        lean_cpu.cum_strategy_turn().len() / lean_turn_stride
    } else { 0 };
    let lean_max_n_river = if lean_river_stride > 0 && lean_n_turn > 0 {
        lean_cpu.cum_strategy_river().len() / (lean_n_turn * lean_river_stride)
    } else { 0 };

    let rich_turn_stride = rich_cpu_lifted.turn_stride();
    let rich_river_stride = rich_cpu_lifted.river_stride();
    let rich_n_turn = if rich_turn_stride > 0 {
        rich_cpu_lifted.cum_strategy_turn().len() / rich_turn_stride
    } else { 0 };
    let rich_max_n_river = if rich_river_stride > 0 && rich_n_turn > 0 {
        rich_cpu_lifted.cum_strategy_river().len() / (rich_n_turn * rich_river_stride)
    } else { 0 };

    // Trees from same chance table → these must match.
    assert_eq!(lean_n_turn, rich_n_turn,
        "n_turn mismatch: lean={} rich={} — chance tables differ?",
        lean_n_turn, rich_n_turn);
    assert_eq!(lean_max_n_river, rich_max_n_river,
        "max_n_river mismatch: lean={} rich={}", lean_max_n_river, rich_max_n_river);

    let n_turn = rich_n_turn;
    let max_n_river = rich_max_n_river;

    // Snapshot the read-only metadata BEFORE the mut-borrow phase so we
    // don't tangle aliases.
    let lean_zones: Vec<Zone> = lean_cpu.zones().to_vec();
    let lean_flop_local: Vec<usize> = lean_cpu.flop_local_offset().to_vec();
    let lean_turn_local: Vec<usize> = lean_cpu.turn_local_offset().to_vec();
    let lean_river_local: Vec<usize> = lean_cpu.river_local_offset().to_vec();
    let lean_csf: Vec<f32> = lean_cpu.cum_strategy_flop().to_vec();
    let lean_cst: Vec<f32> = lean_cpu.cum_strategy_turn().to_vec();
    let lean_csr: Vec<f32> = lean_cpu.cum_strategy_river().to_vec();

    let rich_zones: Vec<Zone> = rich_cpu_lifted.zones().to_vec();
    let rich_flop_local: Vec<usize> = rich_cpu_lifted.flop_local_offset().to_vec();
    let rich_turn_local: Vec<usize> = rich_cpu_lifted.turn_local_offset().to_vec();
    let rich_river_local: Vec<usize> = rich_cpu_lifted.river_local_offset().to_vec();

    // ── FLOP zone ──
    {
        let csf = rich_cpu_lifted.cum_strategy_flop_mut();
        for v in csf.iter_mut() { *v = 0.0; }
        for &nid in &rich_tree.decision_node_ids {
            let r_idx = nid as usize;
            if rich_zones[r_idx] != Zone::Flop { continue; }
            let l_idx = match map.rich_to_lean_node[r_idx] {
                Some(i) => i,
                None => continue,
            };
            if lean_zones[l_idx] != Zone::Flop { continue; }
            let r_local = rich_flop_local[r_idx];
            let l_local = lean_flop_local[l_idx];
            if r_local == NO_LOCAL_OFFSET || l_local == NO_LOCAL_OFFSET { continue; }
            let r_base = r_local * MAX_NA_POSTFLOP * nh;
            let l_base = l_local * MAX_NA_POSTFLOP * nh;
            let na_r = rich_tree.nodes[r_idx].num_children as usize;
            let r_children = rich_tree.node_children(r_idx);
            let l_children: Vec<u32> = if let Some(lt) = lean_tree_opt {
                lt.node_children(l_idx).to_vec()
            } else {
                Vec::new()
            };
            // Per-node EXACT action lookup with claim-tracking. Same problem
            // as walk_pair had: when a parent's children include multiple
            // (ALLIN, 0) entries (dead-stack chains), naive first-match
            // collapses them onto one lean slot. Track which lean a_l has
            // been used by an earlier rich action so duplicates pair in
            // order.
            let mut used_lean_action = vec![false; l_children.len()];
            for a_r in 0..na_r {
                let a_l = if let Some(lt) = lean_tree_opt {
                    let rc_node = &rich_tree.nodes[r_children[a_r] as usize];
                    let key = (rc_node.action_label, rc_node.amount);
                    let mut found: Option<usize> = None;
                    for (a_l_candidate, &lc) in l_children.iter().enumerate() {
                        if used_lean_action[a_l_candidate] { continue; }
                        let lc_node = &lt.nodes[lc as usize];
                        if (lc_node.action_label, lc_node.amount) == key {
                            found = Some(a_l_candidate);
                            used_lean_action[a_l_candidate] = true;
                            break;
                        }
                    }
                    match found {
                        Some(i) => i,
                        None => continue,
                    }
                } else {
                    match map.rich_action_to_lean[r_idx][a_r] {
                        Some(i) => i,
                        None => continue,
                    }
                };
                for h in 0..nh {
                    csf[r_base + a_r * nh + h] = lean_csf[l_base + a_l * nh + h];
                }
            }
        }
    }

    // ── TURN zone (per turn_card_idx) ──
    {
        let cst = rich_cpu_lifted.cum_strategy_turn_mut();
        for v in cst.iter_mut() { *v = 0.0; }
        for &nid in &rich_tree.decision_node_ids {
            let r_idx = nid as usize;
            if rich_zones[r_idx] != Zone::Turn { continue; }
            let l_idx = match map.rich_to_lean_node[r_idx] {
                Some(i) => i,
                None => continue,
            };
            if lean_zones[l_idx] != Zone::Turn { continue; }
            let r_local = rich_turn_local[r_idx];
            let l_local = lean_turn_local[l_idx];
            if r_local == NO_LOCAL_OFFSET || l_local == NO_LOCAL_OFFSET { continue; }
            let na_r = rich_tree.nodes[r_idx].num_children as usize;
            for tc in 0..n_turn {
                let r_base = tc * rich_turn_stride + r_local * MAX_NA_POSTFLOP * nh;
                let l_base = tc * lean_turn_stride + l_local * MAX_NA_POSTFLOP * nh;
                for a_r in 0..na_r {
                    let a_l = match map.rich_action_to_lean[r_idx][a_r] {
                        Some(i) => i,
                        None => continue,
                    };
                    for h in 0..nh {
                        cst[r_base + a_r * nh + h] = lean_cst[l_base + a_l * nh + h];
                    }
                }
            }
        }
    }

    // ── RIVER zone (per (tc, rc)) ──
    {
        let csr = rich_cpu_lifted.cum_strategy_river_mut();
        for v in csr.iter_mut() { *v = 0.0; }
        for &nid in &rich_tree.decision_node_ids {
            let r_idx = nid as usize;
            if rich_zones[r_idx] != Zone::River { continue; }
            let l_idx = match map.rich_to_lean_node[r_idx] {
                Some(i) => i,
                None => continue,
            };
            if lean_zones[l_idx] != Zone::River { continue; }
            let r_local = rich_river_local[r_idx];
            let l_local = lean_river_local[l_idx];
            if r_local == NO_LOCAL_OFFSET || l_local == NO_LOCAL_OFFSET { continue; }
            let na_r = rich_tree.nodes[r_idx].num_children as usize;
            for tc in 0..n_turn {
                for rc in 0..max_n_river {
                    let r_base = (tc * rich_max_n_river + rc) * rich_river_stride
                                 + r_local * MAX_NA_POSTFLOP * nh;
                    let l_base = (tc * lean_max_n_river + rc) * lean_river_stride
                                 + l_local * MAX_NA_POSTFLOP * nh;
                    for a_r in 0..na_r {
                        let a_l = match map.rich_action_to_lean[r_idx][a_r] {
                            Some(i) => i,
                            None => continue,
                        };
                        for h in 0..nh {
                            csr[r_base + a_r * nh + h] = lean_csr[l_base + a_l * nh + h];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
    use crate::tree::builder::build_tree;

    fn config_with(bets: Vec<BetSize>, raises: Vec<BetSize>) -> TreeConfig {
        TreeConfig {
            num_players: 2,
            initial_state: BoardState::Flop,
            starting_pot: 30,
            starting_stacks: vec![100; 2],
            initial_contributions: vec![5; 2],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: bets, raise: raises },
            add_allin_threshold: 1.0,
            force_allin_threshold: 1.0,
            merging_threshold: 0.0,
            button_player: None,
        }
    }

    #[test]
    fn identity_pairing_when_trees_are_equal() {
        // Lean = rich: every node and every action should pair.
        let cfg = config_with(
            vec![BetSize::PotRelative(1.0)],
            vec![],
        );
        let t = build_tree(&cfg).unwrap();
        let map = build_action_map(&t, &t);
        // Every player node paired with itself.
        for &nid in &t.decision_node_ids {
            assert_eq!(map.rich_to_lean_node[nid as usize], Some(nid as usize),
                "identity: node {} should pair with itself", nid);
            for (a, slot) in map.rich_action_to_lean[nid as usize].iter().enumerate() {
                assert_eq!(*slot, Some(a),
                    "identity: action {} at node {} should pair with itself", a, nid);
            }
        }
    }

    #[test]
    fn lean_subset_pairs_subset_of_actions() {
        // Rich has bets [0.5, 1.0]; lean has bets [1.0]. At every player
        // node, rich's bet-1.0 should pair, bet-0.5 should NOT, and
        // fold/check should always pair.
        let rich_cfg = config_with(
            vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            vec![],
        );
        let lean_cfg = config_with(
            vec![BetSize::PotRelative(1.0)],
            vec![],
        );
        let rich = build_tree(&rich_cfg).unwrap();
        let lean = build_tree(&lean_cfg).unwrap();
        let map = build_action_map(&rich, &lean);

        // At least the root should pair (both are PLAYER at root with at
        // least fold/check + the call/bet structure overlapping).
        let mut paired_actions = 0;
        let mut unpaired_actions = 0;
        for &nid in &rich.decision_node_ids {
            if map.rich_to_lean_node[nid as usize].is_none() { continue; }
            for slot in &map.rich_action_to_lean[nid as usize] {
                if slot.is_some() { paired_actions += 1; }
                else { unpaired_actions += 1; }
            }
        }
        assert!(paired_actions > 0, "expected some actions to pair in lean-subset case");
        assert!(unpaired_actions > 0, "expected lean's missing bet-0.5 to leave some actions un-paired");
    }

    #[test]
    fn rich_offsets_match_flattened_layout() {
        // The offsets we compute must match what flattened_cum_strategy does:
        // offset(N) = infoset_offsets[N] * MAX_NA_POSTFLOP * nh.
        let cfg = config_with(vec![BetSize::PotRelative(1.0)], vec![]);
        let t = build_tree(&cfg).unwrap();
        let nh = 10usize;
        let offsets = compute_rich_offsets(&t, nh);
        for &nid in &t.decision_node_ids {
            let info_idx = t.infoset_offsets[nid as usize] as usize;
            let expected = info_idx * MAX_NA_POSTFLOP * nh;
            assert_eq!(offsets[nid as usize], expected,
                "node {} offset: expected {} got {}", nid, expected, offsets[nid as usize]);
        }
    }
}
