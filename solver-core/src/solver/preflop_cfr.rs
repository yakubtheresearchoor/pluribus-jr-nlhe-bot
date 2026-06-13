// Preflop CFR loop (P1.5.4): the iterating preflop-rooted solver that
// composes the anchored components (P5a chance integration, P5b boundary
// expand/reduce, P5c composition) into a multi-iteration preflop solve.
//
// This module is the integration piece. the lead's directive (2026-06-04):
// slice it because integration is where attribution gets lost. The
// preflop CFR iteration has distinct pieces:
//   - preflop strategy computation (regret-matching at preflop infosets,
//     button-first action order) — THIS FILE, Slice A.1
//   - preflop reach update (propagating reach forward through preflop
//     tree) — Slice A.2
//   - per-iteration composition (preflop reach → expand to combos →
//     per-flop solve → reduce CFVs → aggregate over canonicals → preflop
//     regret update) — Slice A.3
//
// Each sub-slice gets its own validation:
//   - A.1: strategy = uniform when regrets zero; matches textbook
//     regret-matching when regrets hand-set.
//   - A.2: reach at terminal = product of opponent strategy probabilities
//     along the path, validated against manual top-down.
//   - A.3: CFV computation matches `compute_preflop_cfv_per_canonical_pass`
//     fed the same expanded reaches (P5c anchored); regret update matches
//     textbook regret-matching formula node-by-node.
//
// The preflop "hand" dimension is the lossless 169-class layout
// (`NUM_PREFLOP_CLASSES`), NOT the per-combo nh=1176 used by postflop.
// Combo multiplicity (pair=6, suited=4, offsuit=12) enters at the
// preflop→flop chance boundary via `expand_reach_class_to_combo` (P5b),
// not in preflop-side regret-matching.

use crate::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use crate::card::{card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use crate::solver::flop_start_vector_cfr::DcfrParams;
use crate::solver::postflop_oracle::PostflopValueOracle;
use crate::solver::preflop_start_game::{
    aggregate_preflop_chance, aggregate_preflop_chance_subset,
    compute_v_flop_at_root_converged, compute_v_flop_at_root_iter0,
    expand_reach_class_to_combo, flop_combo_layout,
    reduce_cfv_combo_to_class, PreflopChanceTable,
};
use crate::tree::flat::{FlatTree, MAX_NA_PREFLOP};

/// Sentinel for "this node is not a preflop-zone player decision node."
const UNUSED: usize = usize::MAX;

/// Small positive-regret threshold matching `flop_start_vector_cfr`'s
/// `REGRET_MATCH_EPS`. Below this, regrets are treated as zero in
/// regret-matching so numerical noise near zero doesn't dominate the
/// uniform-fallback decision.
const REGRET_MATCH_EPS: f32 = 1e-5;

/// Preflop CFR solver state. Holds the preflop-side strategy / regret /
/// cum_strategy buffers for a preflop-rooted tree.
///
/// Storage convention: each preflop player decision node gets `MAX_NA_PREFLOP *
/// NUM_PREFLOP_CLASSES` floats per buffer. The byte offset for infoset i
/// is `local_offset[i] * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES`. Within an
/// infoset, indexing is `[a * NUM_PREFLOP_CLASSES + c]` (same convention
/// as `flop_start_vector_cfr`'s per-infoset stride, with class taking
/// the place of hand).
pub struct PreflopVectorCfr {
    /// Number of players in the tree.
    pub num_players: u8,
    /// Number of preflop player decision infosets.
    pub infoset_count: usize,
    /// `local_offset[idx]` = preflop infoset index for node idx, or
    /// `UNUSED` if the node is not a preflop player decision node.
    pub local_offset: Vec<usize>,
    /// Iteration counter; advanced once per `run_one_iteration` call.
    pub iteration: u32,

    /// Current strategy. `infoset_count * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES`.
    /// Initialized to uniform `1/na` per (class, action) on `new`.
    pub strategy: Vec<f32>,
    /// Cumulative regrets. Same shape. Initialized to zero.
    pub regrets: Vec<f32>,
    /// Cumulative strategy (for time-averaging at the end). Same shape.
    /// Initialized to zero.
    pub cum_strategy: Vec<f32>,
}

impl PreflopVectorCfr {
    /// Construct a fresh preflop solver state for the given tree.
    ///
    /// Requires the tree to have `BoardState::Preflop` somewhere — i.e.,
    /// `initial_state: BoardState::Preflop` in the config. If no preflop
    /// player decision nodes exist (e.g., flop-start tree), the solver
    /// allocates empty buffers and reports `infoset_count == 0`.
    pub fn new(tree: &FlatTree) -> Self {
        use crate::tree::action::BoardState;
        let nn = tree.num_nodes();
        let mut local_offset = vec![UNUSED; nn];
        let mut infoset_count = 0usize;
        let mut max_player_id_seen: i32 = -1;

        for idx in 0..nn {
            let node = &tree.nodes[idx];
            // Preflop zone: board_state field == Preflop (repr=3).
            // Decision (player) node, not chance / terminal.
            if node.board_state != BoardState::Preflop as u8 { continue; }
            if !node.is_player() { continue; }
            local_offset[idx] = infoset_count;
            infoset_count += 1;
            if (node.player_id as i32) > max_player_id_seen {
                max_player_id_seen = node.player_id as i32;
            }
        }

        // num_players from the tree (set on FlatTree, not per-node).
        // We could also derive from max observed player_id + 1 over
        // preflop player nodes, but trees built via the standard
        // builder thread num_players through reliably, so prefer that.
        // Cross-check: if both are computable, they should agree.
        let tree_num_players = tree.num_players;
        if max_player_id_seen >= 0 {
            let derived = (max_player_id_seen + 1) as u8;
            debug_assert!(
                derived <= tree_num_players,
                "max preflop player_id {} >= tree.num_players {}",
                max_player_id_seen, tree_num_players
            );
        }
        let num_players = tree_num_players;

        let stride = MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
        let total = infoset_count * stride;

        // Initial strategy: uniform per node's actual na. We DON'T know
        // na without inspecting the tree, so initialize via a helper
        // that fills uniform per-infoset using the tree's actual na.
        let mut strategy = vec![0.0f32; total];
        Self::init_uniform_strategy(tree, &local_offset, &mut strategy);

        Self {
            num_players,
            infoset_count,
            local_offset,
            iteration: 0,
            strategy,
            regrets: vec![0.0f32; total],
            cum_strategy: vec![0.0f32; total],
        }
    }

    /// Initialize `strategy` to uniform `1/na` per (class, action) for
    /// every preflop player decision node. Helper used by `new`.
    fn init_uniform_strategy(
        tree: &FlatTree,
        local_offset: &[usize],
        strategy: &mut [f32],
    ) {
        let nn = tree.num_nodes();
        for idx in 0..nn {
            let local = local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            if na == 0 { continue; }
            let off = local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
            let uniform = 1.0_f32 / na as f32;
            for a in 0..na {
                for c in 0..NUM_PREFLOP_CLASSES {
                    strategy[off + a * NUM_PREFLOP_CLASSES + c] = uniform;
                }
            }
        }
    }

    /// Recompute the current strategy at every preflop player decision
    /// node from accumulated regrets, via standard regret-matching:
    ///   strategy[a, c] = max(0, regret[a, c]) / Σ_a max(0, regret[a, c])
    ///   if the denominator > 0; else uniform 1/na.
    ///
    /// Per-class independent. Mirrors `flop_start_vector_cfr`'s
    /// `regret_matching_into` semantics with class taking the place of
    /// hand.
    pub fn compute_preflop_strategy(&mut self, tree: &FlatTree) {
        let n_classes = NUM_PREFLOP_CLASSES;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            let local = self.local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            if na == 0 { continue; }
            let off = local * MAX_NA_PREFLOP * n_classes;
            regret_matching_into(
                &self.regrets[off..off + na * n_classes],
                na, n_classes,
                &mut self.strategy[off..off + na * n_classes],
            );
        }
    }

    /// Number of preflop player decision infosets in this tree.
    pub fn infoset_count(&self) -> usize { self.infoset_count }

    /// Current iteration counter.
    pub fn iteration(&self) -> u32 { self.iteration }

    /// Compute per-player, per-class reach top-down through the preflop
    /// zone of the tree, starting from `initial_reach` at the root.
    ///
    /// Returns: `vec[player_idx]` of length `tree.num_nodes() *
    /// NUM_PREFLOP_CLASSES`, indexed as `[node_idx * NUM_PREFLOP_CLASSES
    /// + class]`. Reach is set for every preflop-zone node (player,
    /// chance, and terminal). For nodes outside the preflop zone (e.g.,
    /// children of the preflop→flop chance node, which live in the flop
    /// zone), reach is left as 0.0; downstream code (per-flop solvers)
    /// receives reach via `expand_reach_class_to_combo` reading from
    /// the preflop chance node specifically.
    ///
    /// Top-down rule (standard CFR):
    ///   - At a player node where `node.player_id == p`: scale `reach[p]`
    ///     by the strategy at that node for the action leading to each
    ///     child. Other players' reaches are unchanged across the edge.
    ///   - At a chance node: reach is propagated unchanged across each
    ///     child edge. The chance PROBABILITY is applied externally at
    ///     the chance-integration step (`aggregate_preflop_chance`),
    ///     consistent with how postflop reach is computed in
    ///     `compute_reach_flop`/`compute_reach_turn`/`compute_reach_river`.
    ///   - At a terminal: no children, no propagation.
    ///
    /// `initial_reach`:
    ///   - `None` → all-ones (uniform-1.0 starting reach across classes,
    ///     suitable for tests with constant initial ranges).
    ///   - `Some(&[Vec<f32>])` → `[num_players]` vectors of length
    ///     `NUM_PREFLOP_CLASSES` giving each player's starting per-class
    ///     reach (e.g., from initial preflop ranges).
    pub fn compute_preflop_reach(
        &self,
        tree: &FlatTree,
        initial_reach: Option<&[Vec<f32>]>,
    ) -> Vec<Vec<f32>> {
        use crate::tree::action::BoardState;
        let nn = tree.num_nodes();
        let np = self.num_players as usize;
        let n_classes = NUM_PREFLOP_CLASSES;
        let mut reach: Vec<Vec<f32>> = (0..np)
            .map(|_| vec![0.0_f32; nn * n_classes])
            .collect();

        // Sanity-check + initialize reach at root.
        if let Some(ir) = initial_reach {
            assert_eq!(ir.len(), np,
                "initial_reach has {} player slots; expected {}", ir.len(), np);
            for p in 0..np {
                assert_eq!(ir[p].len(), n_classes,
                    "initial_reach[{}].len() = {}; expected {}", p, ir[p].len(), n_classes);
                reach[p][0..n_classes].copy_from_slice(&ir[p]);
            }
        } else {
            for p in 0..np {
                for c in 0..n_classes {
                    reach[p][c] = 1.0;
                }
            }
        }

        // DFS from root. The preflop tree is small (tens of nodes), so
        // recursion depth is not a concern.
        self.propagate_reach_from(tree, 0, &mut reach, np, n_classes,
            BoardState::Preflop as u8);

        reach
    }

    /// All preflop→flop chance nodes (chance nodes whose PARENT is in
    /// the preflop zone). Per the tree builder's destination convention,
    /// the chance node itself has `board_state == Flop` (it transitions
    /// TO the flop); only its parent has `board_state == Preflop`. Each
    /// preflop→flop chance node corresponds to a preflop action line
    /// that completed betting and reached the flop deal.
    pub fn preflop_chance_node_indices(&self, tree: &FlatTree) -> Vec<usize> {
        use crate::tree::action::BoardState;
        let nn = tree.num_nodes();
        let mut out = Vec::new();
        // Identify the parent of each chance node by scanning children
        // lists. The tree is small for preflop configs so this O(nn)
        // scan is cheap.
        for parent_idx in 0..nn {
            if tree.nodes[parent_idx].board_state != BoardState::Preflop as u8 { continue; }
            for &child_u32 in tree.node_children(parent_idx) {
                let child = child_u32 as usize;
                if tree.nodes[child].is_chance() {
                    out.push(child);
                }
            }
        }
        // Stable + dedup (a chance node has exactly one parent so dedup
        // shouldn't matter, but be defensive).
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Compute traverser's per-class CFV at a preflop chance node by
    /// composing: (1) extract reach[player][chance_node, class] for each
    /// player, (2) for each canonical flop expand each player's reach to
    /// per-combo via `expand_reach_class_to_combo` (P5b), (3) hand the
    /// expanded combo reaches to `per_flop_solver` which returns v[combo]
    /// for traverser, (4) reduce per-combo → per-class via
    /// `reduce_cfv_combo_to_class` (P5b), (5) aggregate per-canonical
    /// via `aggregate_preflop_chance` (P5a, orbit-weighted).
    ///
    /// This is the engine wiring for the preflop→flop boundary. The
    /// arithmetic at each step is already anchored by P5a/P5b/P5c; this
    /// method's responsibility is to compose them in the right order
    /// with the right inputs from PreflopVectorCfr's reach output.
    ///
    /// **`per_flop_solver` signature:**
    /// `(canonical_flop, traverser, combo_reaches) -> v_combo[layout_size]`
    /// where `combo_reaches[player]` is the EXPANDED per-combo reach for
    /// that player on this canonical, in `flop_combo_layout(canonical)`
    /// order. Return is traverser's per-combo CFV at the flop root in
    /// the same order.
    ///
    /// The caller's `per_flop_solver` is opaque: in tests it is a
    /// synthetic closure; in A.3b's production engine it will wrap
    /// `compute_v_flop_at_root_iter0` (or a multi-iter equivalent) on the
    /// passed-in `flop_tree`.
    pub fn compute_chance_node_cfv_with_expansion(
        &self,
        chance_node_idx: usize,
        traverser: u8,
        reach: &[Vec<f32>],
        table: &PreflopChanceTable,
        oracle: &mut impl PostflopValueOracle,
    ) -> Vec<f32> {
        self.chance_cfv_expansion_inner(chance_node_idx, traverser, reach, table, oracle, None)
    }

    /// SLICE-2 ROUTER (2026-06-12): the per-chance-node CFV for one
    /// traverser at the preflop→flop seam, handling folded players.
    ///   - FOLDED TRAVERSER: value is postflop-independent — exactly the
    ///     fold chip-delta × pairwise blocking that `terminal_value_fn`
    ///     computes at any node (it reads contributions + fold mask).
    ///     Routed there, reusing validated machinery with zero new value
    ///     logic. (The bottom-up walk treats this chance node like any
    ///     fold leaf below the traverser's own fold.)
    ///   - LIVE TRAVERSER: the cell-aware oracle, given the folded_mask
    ///     so a live-subset value source can pick the live ranges.
    pub fn chance_cfv_for_traverser(
        &self,
        tree: &FlatTree,
        chance_idx: usize,
        traverser: u8,
        reach: &[Vec<f32>],
        table: &PreflopChanceTable,
        oracle: &mut impl PostflopValueOracle,
        terminal_value_fn: &impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        let n_classes = NUM_PREFLOP_CLASSES;
        let mask = tree.get_folded_mask(chance_idx);
        if (mask >> traverser) & 1 == 1 {
            // Folded traverser: same value a fold leaf at this node gives.
            let base = chance_idx * n_classes;
            let reach_at: Vec<Vec<f32>> =
                (0..np).map(|p| reach[p][base..base + n_classes].to_vec()).collect();
            return terminal_value_fn(chance_idx, traverser, &reach_at);
        }
        let cell =
            crate::solver::postflop_oracle::SeamCell::at_chance_node(tree, chance_idx, np);
        self.compute_chance_node_cfv_with_expansion_for_cell(
            chance_idx, traverser, reach, table, oracle, cell, mask,
        )
    }

    /// SEAM-CELL-AWARE variant (2026-06-12, v1 pot-bucket-keyed
    /// oracles): identical composition, but the oracle is called via
    /// `flop_root_cfv_for_cell` with the flop-entry chance node's seam
    /// cell + `folded_mask` (slice 2: which seats folded preflop, so a
    /// live-subset oracle can select the live ranges). Cell-blind
    /// oracles behave identically (the trait default forwards), so this
    /// is a strict generalization. CONTRACT: the traverser must be LIVE
    /// (the caller routes folded-traverser chance nodes through the
    /// fold machinery).
    pub fn compute_chance_node_cfv_with_expansion_for_cell(
        &self,
        chance_node_idx: usize,
        traverser: u8,
        reach: &[Vec<f32>],
        table: &PreflopChanceTable,
        oracle: &mut impl PostflopValueOracle,
        cell: crate::solver::postflop_oracle::SeamCell,
        folded_mask: u16,
    ) -> Vec<f32> {
        self.chance_cfv_expansion_inner(
            chance_node_idx,
            traverser,
            reach,
            table,
            oracle,
            Some((cell, folded_mask)),
        )
    }

    fn chance_cfv_expansion_inner(
        &self,
        chance_node_idx: usize,
        traverser: u8,
        reach: &[Vec<f32>],
        table: &PreflopChanceTable,
        oracle: &mut impl PostflopValueOracle,
        cell: Option<(crate::solver::postflop_oracle::SeamCell, u16)>,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        assert_eq!(reach.len(), np,
            "reach has {} player slots; expected {}", reach.len(), np);
        let n_classes = NUM_PREFLOP_CLASSES;
        let n_canon = table.num_canonical_flops();
        let chance_base = chance_node_idx * n_classes;

        let mut per_canonical_v_class: Vec<Vec<f32>> = Vec::with_capacity(n_canon);
        for canonical_idx in 0..n_canon {
            let f_canon = table.canonical_flops[canonical_idx];
            let layout = flop_combo_layout(f_canon);

            // Expand each player's class-reach to per-combo via the
            // anchored P5b primitive.
            let mut combo_reaches: Vec<Vec<f32>> = Vec::with_capacity(np);
            for p in 0..np {
                let class_reach = &reach[p][chance_base..chance_base + n_classes];
                combo_reaches.push(expand_reach_class_to_combo(f_canon, class_reach, &layout));
            }

            // Run the postflop oracle for traverser. The oracle decides
            // how to answer (fresh converged solve, warm-start, or
            // bucketed lookup); the engine just consumes the per-combo
            // CFV in flop_combo_layout order.
            let v_combo = match cell {
                Some((c, mask)) => {
                    oracle.flop_root_cfv_for_cell(f_canon, &combo_reaches, traverser, c, mask)
                }
                None => oracle.flop_root_cfv(f_canon, &combo_reaches, traverser),
            };
            assert_eq!(v_combo.len(), layout.len(),
                "oracle returned v_combo of length {}, layout has {} entries \
                 (canonical {:?})", v_combo.len(), layout.len(), f_canon);

            // Reduce per-combo → per-class via anchored P5b.
            let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
            per_canonical_v_class.push(v_class);
        }

        // Aggregate per-canonical → per-class via anchored P5a
        // (orbit-weighted).
        aggregate_preflop_chance(table, &per_canonical_v_class)
    }

    /// Subset variant of `compute_chance_node_cfv_with_expansion`:
    /// runs the per-flop solver on a SUBSET of canonical-flop indices
    /// (Slice B uses this for cheap N measurement; Slice C runs the full
    /// 1755). Aggregates via `aggregate_preflop_chance_subset` (P5a-
    /// anchored subset variant).
    ///
    /// Note: this returns the partial preflop CFV over the subset only.
    /// For convergence baseline purposes (Slice B), both the run-loop
    /// and the best-response calculation use this same subset, so the
    /// engine converges to the SUBSET's equilibrium — a well-defined
    /// reduced-scale problem that preserves the preflop CFR convergence
    /// dynamics per the lead's "cheap dimension" directive.
    /// Cell-aware subset variant (2026-06-12): subset of canonicals,
    /// oracle called via `flop_root_cfv_for_cell`. Used by the seam
    /// gates (cheap canonical subsets) and the bootstrap reach run.
    pub fn compute_chance_node_cfv_with_expansion_subset_for_cell(
        &self,
        chance_node_idx: usize,
        traverser: u8,
        reach: &[Vec<f32>],
        table: &PreflopChanceTable,
        subset_indices: &[usize],
        oracle: &mut impl PostflopValueOracle,
        cell: crate::solver::postflop_oracle::SeamCell,
        folded_mask: u16,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        let n_classes = NUM_PREFLOP_CLASSES;
        let chance_base = chance_node_idx * n_classes;
        let mut per_canonical_v_class: Vec<Vec<f32>> = Vec::with_capacity(subset_indices.len());
        for &canonical_idx in subset_indices {
            let f_canon = table.canonical_flops[canonical_idx];
            let layout = flop_combo_layout(f_canon);
            let mut combo_reaches: Vec<Vec<f32>> = Vec::with_capacity(np);
            for p in 0..np {
                let class_reach = &reach[p][chance_base..chance_base + n_classes];
                combo_reaches.push(expand_reach_class_to_combo(f_canon, class_reach, &layout));
            }
            let v_combo = oracle
                .flop_root_cfv_for_cell(f_canon, &combo_reaches, traverser, cell, folded_mask);
            assert_eq!(v_combo.len(), layout.len());
            per_canonical_v_class.push(reduce_cfv_combo_to_class(f_canon, &v_combo, &layout));
        }
        aggregate_preflop_chance_subset(table, subset_indices, &per_canonical_v_class)
    }

    pub fn compute_chance_node_cfv_with_expansion_subset(
        &self,
        chance_node_idx: usize,
        traverser: u8,
        reach: &[Vec<f32>],
        table: &PreflopChanceTable,
        subset_indices: &[usize],
        oracle: &mut impl PostflopValueOracle,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        assert_eq!(reach.len(), np);
        let n_classes = NUM_PREFLOP_CLASSES;
        let chance_base = chance_node_idx * n_classes;

        let mut per_subset_v_class: Vec<Vec<f32>> = Vec::with_capacity(subset_indices.len());
        for &canonical_idx in subset_indices {
            let f_canon = table.canonical_flops[canonical_idx];
            let layout = flop_combo_layout(f_canon);

            let mut combo_reaches: Vec<Vec<f32>> = Vec::with_capacity(np);
            for p in 0..np {
                let class_reach = &reach[p][chance_base..chance_base + n_classes];
                combo_reaches.push(expand_reach_class_to_combo(f_canon, class_reach, &layout));
            }

            let v_combo = oracle.flop_root_cfv(f_canon, &combo_reaches, traverser);
            assert_eq!(v_combo.len(), layout.len());
            let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
            per_subset_v_class.push(v_class);
        }

        aggregate_preflop_chance_subset(table, subset_indices, &per_subset_v_class)
    }

    /// Subset-aware variant of `run_one_iteration`. Runs per-flop
    /// solves only on the given canonical-flop indices, aggregating
    /// orbit-weighted via the subset-aware aggregation.
    ///
    /// Slice B uses this for the multi-iter convergence study: a small
    /// subset (say 10 canonicals) keeps each iteration's wall-clock
    /// bounded so we can run 50–100 iters and measure when the time-
    /// averaged strategy stabilizes (correctness baseline first, then
    /// N readout).
    pub fn run_one_iteration_subset(
        &mut self,
        tree: &FlatTree,
        table: &PreflopChanceTable,
        subset_indices: &[usize],
        oracle: &mut impl PostflopValueOracle,
        terminal_value_fn: impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
    ) {
        let n_classes = NUM_PREFLOP_CLASSES;
        let nn = tree.num_nodes();
        let np = self.num_players;

        self.compute_preflop_strategy(tree);
        let reach = self.compute_preflop_reach(tree, None);
        let chance_nodes = self.preflop_chance_node_indices(tree);
        let params = DcfrParams::new(self.iteration);

        oracle.begin_preflop_iter(self.iteration);

        for t in 0..np {
            let mut cfv: Vec<Vec<f32>> = vec![vec![0.0_f32; n_classes]; nn];
            for &chance_idx in &chance_nodes {
                cfv[chance_idx] = self.compute_chance_node_cfv_with_expansion_subset(
                    chance_idx, t, &reach, table, subset_indices, oracle,
                );
            }
            self.bottom_up_preflop_for_traverser(
                tree, t, &chance_nodes, &reach, &terminal_value_fn, &mut cfv, &params,
            );
        }

        oracle.end_preflop_iter(self.iteration);
        self.iteration += 1;
    }

    /// Compute traverser's best-response value at the root, given the
    /// engine's current state (strategy, regrets, cum_strategy).
    ///
    /// Best-response semantics: at each TRAVERSER infoset, replace
    /// strategy with the argmax-per-class (one-hot on the action
    /// maximizing v[child[a], c]). At opp infosets, use the standard
    /// factored-CFR plain-sum aggregation (opp's strategy absorbed into
    /// reach via the per-flop solver's expansion).
    ///
    /// Caller pre-computes `chance_cfv[chance_node_idx]` for each
    /// preflop chance node using the engine's current reach (so opp's
    /// strategy at the boundary is whatever the caller chose — for
    /// correctness baseline, pass cfv from compute_chance_node_cfv_with_
    /// expansion_subset using opp's current strategy reach).
    ///
    /// Returns traverser's per-class BR value at the root. For an
    /// exploitability-trend signal, compute this periodically and
    /// observe |sum across players| / 2 decreasing.
    pub fn compute_traverser_br_value(
        &self,
        tree: &FlatTree,
        traverser: u8,
        chance_node_set: &[usize],
        reach: &[Vec<f32>],
        chance_cfv: &[Vec<f32>],
        terminal_value_fn: impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
    ) -> Vec<f32> {
        let n_classes = NUM_PREFLOP_CLASSES;
        let nn = tree.num_nodes();
        let mut cfv: Vec<Vec<f32>> = vec![vec![0.0_f32; n_classes]; nn];
        // Pre-populate chance nodes.
        for &c_idx in chance_node_set {
            cfv[c_idx] = chance_cfv[c_idx].clone();
        }
        let is_chance_leaf = |idx: usize| chance_node_set.binary_search(&idx).is_ok();
        self.br_recursive(
            tree, 0, traverser, &is_chance_leaf, reach,
            &terminal_value_fn, &mut cfv,
        );
        cfv[0].clone()
    }

    fn br_recursive(
        &self,
        tree: &FlatTree,
        node_idx: usize,
        traverser: u8,
        is_chance_leaf: &impl Fn(usize) -> bool,
        reach: &[Vec<f32>],
        terminal_value_fn: &impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
        cfv: &mut [Vec<f32>],
    ) {
        if is_chance_leaf(node_idx) { return; }
        let node = &tree.nodes[node_idx];
        let n_classes = NUM_PREFLOP_CLASSES;
        if node.is_terminal() {
            let np = self.num_players as usize;
            let base = node_idx * n_classes;
            let mut reach_at_terminal: Vec<Vec<f32>> = Vec::with_capacity(np);
            for p in 0..np {
                reach_at_terminal.push(reach[p][base..base + n_classes].to_vec());
            }
            cfv[node_idx] = terminal_value_fn(node_idx, traverser, &reach_at_terminal);
            return;
        }
        let children: Vec<u32> = tree.node_children(node_idx).to_vec();
        for &ch in &children {
            self.br_recursive(
                tree, ch as usize, traverser, is_chance_leaf, reach,
                terminal_value_fn, cfv,
            );
        }
        let mut v = vec![0.0_f32; n_classes];
        if node.player_id == traverser {
            // BR: per class, pick argmax across actions.
            for c in 0..n_classes {
                let mut best = f32::NEG_INFINITY;
                for &ch in &children {
                    let child = ch as usize;
                    if cfv[child][c] > best { best = cfv[child][c]; }
                }
                v[c] = best;
            }
        } else {
            // Opp: factored convention, plain sum (opp's strategy already
            // absorbed into reach via per-flop solver expansion).
            for &ch in &children {
                let child = ch as usize;
                for c in 0..n_classes {
                    v[c] += cfv[child][c];
                }
            }
        }
        cfv[node_idx] = v;
    }

    /// Bottom-up CFV computation + regret update for one traverser,
    /// composing v_at_chance (the chance-node boundary values from A.3a)
    /// with terminal values (provided opaque per node) plus standard
    /// factored-CFR aggregation at preflop player nodes.
    ///
    /// Mirrors `flop_start_vector_cfr::bottom_up_zone`'s pattern:
    ///   - At traverser's OWN decision node: strategy-weighted average of
    ///     children's CFVs, plus DCFR regret update (alpha/beta on positive
    ///     vs negative regrets, gamma on cum_strategy).
    ///   - At opp's decision node: PLAIN sum across actions. Factored
    ///     CFR convention: opp's strategy is already absorbed into the
    ///     reach factor at children, so summing across actions integrates
    ///     correctly.
    ///
    /// Inputs:
    ///   - `cfv`: caller-supplied storage `[nn]` of `Vec<f32; NUM_PREFLOP_CLASSES>`,
    ///     pre-populated with `v_at_chance[chance_node_idx]` for each
    ///     preflop chance node (so the walk treats chance nodes as
    ///     leaves), and with terminal values OR via `terminal_value_fn`.
    ///   - `terminal_value_fn`: called on each preflop terminal to get
    ///     traverser's per-class CFV at that terminal. Opaque to the
    ///     engine; A.3c will wrap the class×class blocking matrix in
    ///     a production implementation.
    ///
    /// On completion: `cfv[root]` holds traverser's per-class CFV at the
    /// root, AND `self.regrets` / `self.cum_strategy` have been updated
    /// for every traverser-owned preflop player node along the way.
    pub fn bottom_up_preflop_for_traverser(
        &mut self,
        tree: &FlatTree,
        traverser: u8,
        chance_node_set: &[usize],
        reach: &[Vec<f32>],
        terminal_value_fn: impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
        cfv: &mut [Vec<f32>],
        params: &DcfrParams,
    ) {
        // Build a quick lookup set of chance-node indices (HU has 1–10).
        let is_chance_leaf = |idx: usize| chance_node_set.binary_search(&idx).is_ok();
        self.bottom_up_recursive(
            tree, 0, traverser, &is_chance_leaf, reach,
            &terminal_value_fn, cfv, params,
        );
    }

    fn bottom_up_recursive(
        &mut self,
        tree: &FlatTree,
        node_idx: usize,
        traverser: u8,
        is_chance_leaf: &impl Fn(usize) -> bool,
        reach: &[Vec<f32>],
        terminal_value_fn: &impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
        cfv: &mut [Vec<f32>],
        params: &DcfrParams,
    ) {
        // Preflop chance node = leaf for the preflop bottom-up walk.
        // The cfv at this node was populated by the caller via
        // compute_chance_node_cfv_with_expansion. Nothing more to do.
        if is_chance_leaf(node_idx) { return; }

        let node = &tree.nodes[node_idx];
        let n_classes = NUM_PREFLOP_CLASSES;

        if node.is_terminal() {
            // Build per-terminal per-player reach slice for the callback.
            // The callback can pick out opp reaches as needed (HU: one opp).
            let np = self.num_players as usize;
            let mut reach_at_terminal: Vec<Vec<f32>> = Vec::with_capacity(np);
            let base = node_idx * n_classes;
            for p in 0..np {
                reach_at_terminal.push(reach[p][base..base + n_classes].to_vec());
            }
            cfv[node_idx] = terminal_value_fn(node_idx, traverser, &reach_at_terminal);
            return;
        }

        // Preflop player node. Recurse into all children first.
        let children: Vec<u32> = tree.node_children(node_idx).to_vec();
        for &child_u32 in &children {
            self.bottom_up_recursive(
                tree, child_u32 as usize, traverser,
                is_chance_leaf, reach, terminal_value_fn, cfv, params,
            );
        }

        let local = self.local_offset[node_idx];
        debug_assert!(local != UNUSED,
            "node {} reached in preflop bottom-up walk but has no preflop local_offset \
             (not a preflop player node and not a chance/terminal leaf — tree shape unexpected)",
            node_idx);
        let na = node.num_children as usize;
        let off = local * MAX_NA_PREFLOP * n_classes;

        let mut cfv_avg = vec![0.0_f32; n_classes];

        if node.player_id == traverser {
            // Traverser's own decision: strategy-weighted aggregation.
            for (a, &child_u32) in children.iter().enumerate() {
                let child = child_u32 as usize;
                let s_base = off + a * n_classes;
                for c in 0..n_classes {
                    cfv_avg[c] += self.strategy[s_base + c] * cfv[child][c];
                }
            }
            // Regret + cum_strategy update with DCFR discount.
            for (a, &child_u32) in children.iter().enumerate() {
                let child = child_u32 as usize;
                for c in 0..n_classes {
                    let inst_regret = cfv[child][c] - cfv_avg[c];
                    let ridx = off + a * n_classes + c;
                    let old_r = self.regrets[ridx];
                    let coef = if old_r >= 0.0 { params.alpha_t() } else { params.beta_t() };
                    self.regrets[ridx] = coef * old_r + inst_regret;
                    self.cum_strategy[ridx] = params.gamma_t() * self.cum_strategy[ridx]
                        + self.strategy[ridx];
                }
            }
        } else {
            // Opp's decision: plain sum across actions (factored convention).
            for &child_u32 in &children {
                let child = child_u32 as usize;
                for c in 0..n_classes {
                    cfv_avg[c] += cfv[child][c];
                }
            }
        }

        cfv[node_idx] = cfv_avg;
    }

    /// Run one full preflop CFR iteration:
    ///   1. Recompute strategy from current regrets.
    ///   2. Compute reach top-down.
    ///   3. For each traverser:
    ///        a. Compute v_at_chance at each preflop chance node via
    ///           A.3a composition (per_flop_solver gives per-flop CFV;
    ///           expand/reduce/aggregate wrap it).
    ///        b. Bottom-up walk + regret update for traverser-owned nodes.
    ///   4. Advance iteration counter.
    ///
    /// `per_flop_solver`: (canonical, traverser, expanded_combo_reaches_per_player) → v_combo
    /// `terminal_value_fn`: (terminal_node_idx, traverser) → per-class CFV at that terminal
    pub fn run_one_iteration(
        &mut self,
        tree: &FlatTree,
        table: &PreflopChanceTable,
        oracle: &mut impl PostflopValueOracle,
        terminal_value_fn: impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
    ) {
        let n_classes = NUM_PREFLOP_CLASSES;
        let nn = tree.num_nodes();
        let np = self.num_players;

        // Step 1: update strategy from regrets.
        self.compute_preflop_strategy(tree);

        // Step 2: reach (uniform initial reach for now; non-uniform can
        // be threaded in via a separate parameter later).
        let reach = self.compute_preflop_reach(tree, None);

        let chance_nodes = self.preflop_chance_node_indices(tree);
        let params = DcfrParams::new(self.iteration);

        // Oracle hook: begin iter. Lets stateful oracles (warm-start /
        // Fix B's outer-vs-inner cadence; bucketed periodic refresh)
        // decide what to do this iter. The simple oracle is no-op.
        oracle.begin_preflop_iter(self.iteration);

        // Step 3: per-traverser CFV + regret update.
        for t in 0..np {
            // 3a. v_at_chance per preflop chance node (cell-aware since
            // 2026-06-12 — strict generalization: cell-blind oracles
            // ignore the cell via the trait default, and the collapsed
            // path's bit-exact gate vs this loop demands both arms make
            // the same oracle calls).
            let mut cfv: Vec<Vec<f32>> = vec![vec![0.0_f32; n_classes]; nn];
            for &chance_idx in &chance_nodes {
                cfv[chance_idx] = self.chance_cfv_for_traverser(
                    tree, chance_idx, t, &reach, table, oracle, &terminal_value_fn,
                );
            }

            // 3b. Bottom-up walk + regret update.
            self.bottom_up_preflop_for_traverser(
                tree, t, &chance_nodes, &reach, &terminal_value_fn, &mut cfv, &params,
            );
        }

        // Oracle hook: end iter. Commit per-canonical state, decide
        // next-iter cadence.
        oracle.end_preflop_iter(self.iteration);

        // Step 4: advance iteration.
        self.iteration += 1;
    }

    /// The V1 collapse key: the validated SPR cell-set policy's bucket
    /// id at each flop-entry chance node — (live, log2-SPR bin at
    /// width 0.25; all-in sentinel). Use as `chance_key` for
    /// `run_one_iteration_shared_chance` with a bucket-keyed oracle:
    /// n_keys ≈ the ~125 solve units + per-live all-in bins.
    pub fn seam_bucket_chance_key<'a>(
        tree: &'a FlatTree,
        np: usize,
        stack: i32,
    ) -> impl Fn(usize) -> u64 + 'a {
        move |chance_idx| {
            let cell = crate::solver::postflop_oracle::SeamCell::at_chance_node(
                tree, chance_idx, np,
            );
            let (live, bin) = cell.bucket_key(stack);
            // Pack (live, bin) — bin is small (±~30) or the i64::MIN
            // all-in sentinel; clamp to i32 for packing.
            let bin32 = bin.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            ((live as u64) << 32) | (bin32 as u32 as u64)
        }
    }

    /// `run_one_iteration` with the chance-node CFV computed ONCE per
    /// (chance_key, traverser) and broadcast to every chance node
    /// sharing that key — the frozen-oracle collapse (B5 preflop
    /// pricing, 2026-06-10).
    ///
    /// WHY THIS IS EXACT (not an approximation), given a reach-
    /// independent oracle: `compute_chance_node_cfv_with_expansion`'s
    /// output flows reach ONLY into the oracle call's `combo_reaches`;
    /// `reduce_cfv_combo_to_class` is an unweighted per-class average
    /// of `v_combo` and `aggregate_preflop_chance` weights only by
    /// chance probability. A frozen oracle (CFVs cached per (flop,
    /// traverser), ignoring reach) therefore yields BIT-IDENTICAL
    /// v_class at every chance node with the same key — the per-node
    /// recomputation is pure control flow, same species as the 3^K
    /// collapse. Measured: 897.1s/iter → collapse target ~3s (1493
    /// chance nodes × 6 traversers × 1755 canonicals of redundant
    /// expand/lookup/reduce removed).
    ///
    /// `chance_key(chance_idx)`: epoch-sharing key. Constant for a
    /// fully reach-independent frozen oracle; pot-bucket id for a
    /// pot-bucketed frozen oracle. CALLER CONTRACT: the oracle's
    /// answer must depend only on (canonical_flop, traverser, key) —
    /// an oracle that reads `combo_ranges` voids the collapse (use
    /// `run_one_iteration`). Gated bit-exact vs `run_one_iteration`
    /// by p1_5_4_blueprint_preflop_cost.
    pub fn run_one_iteration_shared_chance(
        &mut self,
        tree: &FlatTree,
        table: &PreflopChanceTable,
        oracle: &mut impl PostflopValueOracle,
        terminal_value_fn: impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
        chance_key: impl Fn(usize) -> u64,
    ) {
        let n_classes = NUM_PREFLOP_CLASSES;
        let nn = tree.num_nodes();
        let np = self.num_players;

        self.compute_preflop_strategy(tree);
        let reach = self.compute_preflop_reach(tree, None);
        let chance_nodes = self.preflop_chance_node_indices(tree);
        let params = DcfrParams::new(self.iteration);

        oracle.begin_preflop_iter(self.iteration);

        for t in 0..np {
            let mut cfv: Vec<Vec<f32>> = vec![vec![0.0_f32; n_classes]; nn];
            let mut by_key: std::collections::HashMap<u64, Vec<f32>> =
                std::collections::HashMap::new();
            for &chance_idx in &chance_nodes {
                // SLICE-2 (2026-06-12): a FOLDED traverser's value is
                // postflop-independent and NODE-SPECIFIC (its own
                // contributions + reaches), so it is NOT collapsed by
                // key — computed per node via the fold machinery. Only
                // LIVE-traverser oracle calls are collapsed.
                let mask = tree.get_folded_mask(chance_idx);
                if (mask >> t) & 1 == 1 {
                    let base = chance_idx * n_classes;
                    let reach_at: Vec<Vec<f32>> = (0..np as usize)
                        .map(|p| reach[p][base..base + n_classes].to_vec())
                        .collect();
                    cfv[chance_idx] = terminal_value_fn(chance_idx, t, &reach_at);
                    continue;
                }
                let key = chance_key(chance_idx);
                let v = match by_key.get(&key) {
                    Some(v) => v.clone(),
                    None => {
                        // Cell-aware call (2026-06-12): the oracle sees
                        // the seam cell + folded_mask of the FIRST chance
                        // node bearing this key — consistent with the
                        // caller contract (answer depends only on
                        // (flop, t, key); any key-mate selects the same
                        // representative blueprint).
                        let cell = crate::solver::postflop_oracle::SeamCell::at_chance_node(
                            tree, chance_idx, np as usize,
                        );
                        let v = self.compute_chance_node_cfv_with_expansion_for_cell(
                            chance_idx, t, &reach, table, oracle, cell, mask,
                        );
                        by_key.insert(key, v.clone());
                        v
                    }
                };
                cfv[chance_idx] = v;
            }

            self.bottom_up_preflop_for_traverser(
                tree, t, &chance_nodes, &reach, &terminal_value_fn, &mut cfv, &params,
            );
        }

        oracle.end_preflop_iter(self.iteration);
        self.iteration += 1;
    }

    /// Helper: propagate reach from `node_idx` to its children, then
    /// recurse. Stops propagation if a child is NOT in the preflop zone
    /// (zone boundary is the preflop→flop chance edge; children of that
    /// chance node belong to the flop zone and are handled by the
    /// per-canonical postflop solves).
    fn propagate_reach_from(
        &self,
        tree: &FlatTree,
        node_idx: usize,
        reach: &mut [Vec<f32>],
        np: usize,
        n_classes: usize,
        preflop_state: u8,
    ) {
        let node = &tree.nodes[node_idx];
        if node.board_state != preflop_state { return; }
        let children: Vec<u32> = tree.node_children(node_idx).to_vec();
        if children.is_empty() { return; }

        if node.is_player() {
            let pid = node.player_id as usize;
            let local = self.local_offset[node_idx];
            assert!(local != UNUSED,
                "preflop player node {} has no local_offset (should be impossible after new())",
                node_idx);
            let na = node.num_children as usize;
            assert_eq!(na, children.len(),
                "node {}: num_children {} disagrees with node_children.len() {}",
                node_idx, na, children.len());
            let off = local * MAX_NA_PREFLOP * n_classes;
            let parent_base = node_idx * n_classes;
            for (a, &child_u32) in children.iter().enumerate() {
                let child = child_u32 as usize;
                let child_base = child * n_classes;
                // Acting player: scale by strategy[node, a, c].
                for c in 0..n_classes {
                    reach[pid][child_base + c] =
                        reach[pid][parent_base + c]
                        * self.strategy[off + a * n_classes + c];
                }
                // Other players: pass through unchanged.
                for p in 0..np {
                    if p == pid { continue; }
                    for c in 0..n_classes {
                        reach[p][child_base + c] = reach[p][parent_base + c];
                    }
                }
                self.propagate_reach_from(tree, child, reach, np, n_classes, preflop_state);
            }
        } else {
            // Chance or terminal. Terminals have no children (already
            // returned above). Chance: propagate unchanged; chance
            // PROBABILITY applied at the integration step, not here.
            let parent_base = node_idx * n_classes;
            for &child_u32 in &children {
                let child = child_u32 as usize;
                let child_base = child * n_classes;
                for p in 0..np {
                    for c in 0..n_classes {
                        reach[p][child_base + c] = reach[p][parent_base + c];
                    }
                }
                self.propagate_reach_from(tree, child, reach, np, n_classes, preflop_state);
            }
        }
    }
}

/// Build a `per_flop_solver` closure that wraps
/// `compute_v_flop_at_root_converged`. This is the CORRECT-GAME
/// per-flop solver (per the lead's directive 2026-06-04): postflop is
/// CFR-converged per call, not iter-0 uniform. The engine using this
/// solver solves the REAL preflop game, not the iter-0-postflop hybrid
/// game the iter-0 wrapper produces.
///
/// `num_postflop_iters`: convergence budget per per-flop solve.
/// Convergence_audit reference: 100 iters at nh=4 → 0.37% pot
/// exploitability. At larger nh the convergence rate is its own
/// measurement; start with 100 and tune. Each per-flop solve becomes a
/// ~100×-cost operation versus iter-0, which is the order-of-magnitude
/// increase the GPU exists to make feasible.
///
/// Layout reconciliation (same as iter-0 wrapper): converts engine-
/// layout combo_reaches to NUM_POSSIBLE_HANDS indexing, calls the
/// converged solver, re-orders returned v_combo from table layout to
/// engine layout.
pub fn make_per_flop_solver_converged<'a>(
    flop_tree: &'a FlatTree,
    num_postflop_iters: u32,
) -> impl Fn([Card; 3], u8, &[Vec<f32>]) -> Vec<f32> + 'a {
    move |canonical: [Card; 3], traverser: u8, combo_reaches: &[Vec<f32>]| -> Vec<f32> {
        let layout_engine = flop_combo_layout(canonical);
        let np = combo_reaches.len();
        assert!(np >= 2);
        let mut combo_ranges_full: Vec<Vec<f32>> =
            vec![vec![0.0_f32; NUM_POSSIBLE_HANDS]; np];
        for p in 0..np {
            assert_eq!(combo_reaches[p].len(), layout_engine.len());
            for (li, &(c1, c2)) in layout_engine.iter().enumerate() {
                combo_ranges_full[p][card_pair_to_index(c1, c2)] = combo_reaches[p][li];
            }
        }
        let (v_table_order, layout_table) = compute_v_flop_at_root_converged(
            canonical, flop_tree, &combo_ranges_full, traverser, num_postflop_iters,
        );
        assert_eq!(layout_engine.len(), layout_table.len());
        let mut v_engine_order = Vec::with_capacity(layout_engine.len());
        for &combo in &layout_engine {
            let pos = layout_table.iter().position(|&c| c == combo)
                .unwrap_or_else(|| panic!("combo {:?} not in table layout", combo));
            v_engine_order.push(v_table_order[pos]);
        }
        v_engine_order
    }
}

/// Build a `per_flop_solver` closure that wraps
/// `compute_v_flop_at_root_iter0` for use with
/// `PreflopVectorCfr::run_one_iteration`.
///
/// **DEPRECATED for production use** (2026-06-04): this solver runs
/// postflop at iter-0 uniform strategies, which makes the engine solve
/// the wrong game ("preflop CFR with fixed iter-0 postflop", not
/// NLHE). Kept for the Slice A unit-test scenarios where the per-flop
/// CFV semantics don't matter (e.g., synthetic v_flop_fn validations
/// of the chance-boundary composition). Production callers use
/// `make_per_flop_solver_converged`.
///
/// Per the lead's decisions confirmed via the record:
///   - **No caching across canonicals.** Per slice 7a cost attribution,
///     table construction = 2% of per-flop cost (solve = 95%); caching
///     saves ~140ms/call for the memory cost of holding 1755 tables.
///     Ephemeral build-use-discard per canonical.
///   - **No special memory layout.** Slice 7a measured ~10MB peak
///     working set with the build-discard pattern; sufficient.
///   - **Canonical set is parameterized at the run-loop level**, not
///     hardcoded here. Slice B runs a subset for cheap N measurement;
///     Slice C runs full 1755. This wrapper is canonical-agnostic; the
///     caller iterates over whatever canonical set is appropriate.
///
/// Layout reconciliation: `compute_v_flop_at_root_iter0` builds a
/// `FlopChanceTable` whose internal `hand_cards` ordering may differ
/// from `flop_combo_layout(canonical)` (the layout the engine uses
/// throughout `compute_chance_node_cfv_with_expansion`). This wrapper:
///
///   1. Maps engine-layout-indexed `combo_reaches` to `NUM_POSSIBLE_HANDS`-
///      indexed `combo_ranges` (the format `compute_v_flop_at_root_iter0`
///      expects) via `card_pair_to_index`.
///   2. Calls `compute_v_flop_at_root_iter0`, which returns
///      `(v_combo_table_order, layout_table)`.
///   3. Re-orders `v_combo` from table layout back to engine layout
///      via a per-canonical position map. Both layouts contain the
///      same SET of non-board-blocked combos, so a permutation suffices.
pub fn make_per_flop_solver_iter0<'a>(
    flop_tree: &'a FlatTree,
) -> impl Fn([Card; 3], u8, &[Vec<f32>]) -> Vec<f32> + 'a {
    move |canonical: [Card; 3], traverser: u8, combo_reaches: &[Vec<f32>]| -> Vec<f32> {
        let layout_engine = flop_combo_layout(canonical);
        let np = combo_reaches.len();
        assert!(np >= 2, "need at least 2 players");

        // Convert engine-layout combo_reaches → NUM_POSSIBLE_HANDS-indexed.
        let mut combo_ranges_full: Vec<Vec<f32>> =
            vec![vec![0.0_f32; NUM_POSSIBLE_HANDS]; np];
        for p in 0..np {
            assert_eq!(combo_reaches[p].len(), layout_engine.len(),
                "combo_reaches[{}] length {} != engine layout length {} for canonical {:?}",
                p, combo_reaches[p].len(), layout_engine.len(), canonical);
            for (li, &(c1, c2)) in layout_engine.iter().enumerate() {
                let idx = card_pair_to_index(c1, c2);
                combo_ranges_full[p][idx] = combo_reaches[p][li];
            }
        }

        // Solve.
        let (v_table_order, layout_table) = compute_v_flop_at_root_iter0(
            canonical, flop_tree, &combo_ranges_full, traverser,
        );
        assert_eq!(layout_engine.len(), layout_table.len(),
            "layout size mismatch for canonical {:?}: engine={}, table={}",
            canonical, layout_engine.len(), layout_table.len());

        // Re-order v_combo from table layout to engine layout.
        // Linear scan suffices (layouts are small, ~1176 entries).
        let mut v_engine_order: Vec<f32> = Vec::with_capacity(layout_engine.len());
        for &combo in &layout_engine {
            let pos = layout_table.iter().position(|&c| c == combo)
                .unwrap_or_else(|| panic!(
                    "engine combo {:?} not in table layout for canonical {:?}",
                    combo, canonical));
            v_engine_order.push(v_table_order[pos]);
        }
        v_engine_order
    }
}

/// Build the production `terminal_value_fn` closure that wraps A.3c's
/// `preflop_fold_terminal_cfv_hu` + A.3d's chip_delta extraction, for
/// HU. The closure is suitable for passing as the
/// `terminal_value_fn` argument to
/// `PreflopVectorCfr::run_one_iteration`.
///
/// DEPRECATED in favor of `make_production_terminal_value_fn_multiway`,
/// which is np-agnostic and handles HU as a special case. Kept for the
/// existing HU Slice B test compatibility; new callers use the
/// multiway version.
pub fn make_production_terminal_value_fn_hu<'a>(
    tree: &'a FlatTree,
    blocking_matrix: &'a [f32],
) -> impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32> + 'a {
    use crate::solver::preflop_terminal::{
        preflop_fold_terminal_chip_delta, preflop_fold_terminal_cfv_hu,
    };
    move |term_idx: usize, traverser: u8, reach_at_term: &[Vec<f32>]| -> Vec<f32> {
        let chip_delta = preflop_fold_terminal_chip_delta(tree, term_idx, traverser);
        // HU: one opponent.
        let opp = (1 - traverser) as usize;
        preflop_fold_terminal_cfv_hu(&reach_at_term[opp], blocking_matrix, chip_delta)
    }
}

/// Build the production `terminal_value_fn` closure for ARBITRARY
/// player count. The np-agnostic version: composes A.3d's chip_delta
/// (already np-aware via the `starting_pot/np + c_t` formula) with
/// B.2's multiway fold-terminal CFV (which handles N-1 opps via
/// joint blocking).
///
/// At HU (np=2), produces output identical to
/// `make_production_terminal_value_fn_hu` (validated by B.3 test).
///
/// At 6-max (np=6), iterates over the 5 opps' per-class reaches at
/// the terminal node and feeds them to the multiway primitive. The
/// caller (PreflopVectorCfr) supplies `reach_at_term` for all np
/// players; this closure filters to the N-1 opps (everyone except
/// traverser) and dispatches.
pub fn make_production_terminal_value_fn_multiway<'a>(
    tree: &'a FlatTree,
) -> impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32> + 'a {
    use crate::solver::preflop_terminal::{
        preflop_fold_terminal_chip_delta, preflop_fold_terminal_cfv_multiway,
    };
    move |term_idx: usize, traverser: u8, reach_at_term: &[Vec<f32>]| -> Vec<f32> {
        let chip_delta = preflop_fold_terminal_chip_delta(tree, term_idx, traverser);
        let np = tree.num_players as usize;
        let t = traverser as usize;
        assert_eq!(reach_at_term.len(), np,
            "reach_at_term has {} players; expected {}", reach_at_term.len(), np);
        // Collect opp reaches: all players except traverser.
        let opp_reaches: Vec<&[f32]> = (0..np)
            .filter(|&p| p != t)
            .map(|p| reach_at_term[p].as_slice())
            .collect();
        preflop_fold_terminal_cfv_multiway(&opp_reaches, chip_delta)
    }
}

/// Pairwise-approximate variant of
/// `make_production_terminal_value_fn_multiway` for dense-reach callers
/// (the P8 preflop bootstrap). Same shape, but fold-terminal CFVs use
/// `preflop_fold_terminal_cfv_multiway_pairwise` (each opp's blocking
/// vs traverser independent; opp-opp cross-blocking ignored — a smooth
/// over-count, exact at N=2). The exact joint enumeration is O(Π_i
/// nnz(opp_i)) per traverser class and measured intractable at 6-max
/// dense reaches (≈169^5 tuples per terminal); the pairwise form is 5
/// matvecs of the fixed 169×169 blocking matrix, precomputed here once.
///
/// BOOTSTRAP-GRADE: use for size observation and other approximation-
/// tolerant measurements, NOT for production blueprint training without
/// first characterizing the bias against the exact primitive on sparse
/// cases.
pub fn make_bootstrap_terminal_value_fn_multiway_pairwise<'a>(
    tree: &'a FlatTree,
) -> impl Fn(usize, u8, &[Vec<f32>]) -> Vec<f32> + 'a {
    use crate::solver::preflop_terminal::{
        build_class_blocking_matrix, preflop_fold_terminal_chip_delta,
        preflop_fold_terminal_cfv_multiway_pairwise,
    };
    let blocking_matrix = build_class_blocking_matrix();
    move |term_idx: usize, traverser: u8, reach_at_term: &[Vec<f32>]| -> Vec<f32> {
        let chip_delta = preflop_fold_terminal_chip_delta(tree, term_idx, traverser);
        let np = tree.num_players as usize;
        let t = traverser as usize;
        assert_eq!(reach_at_term.len(), np,
            "reach_at_term has {} players; expected {}", reach_at_term.len(), np);
        let opp_reaches: Vec<&[f32]> = (0..np)
            .filter(|&p| p != t)
            .map(|p| reach_at_term[p].as_slice())
            .collect();
        preflop_fold_terminal_cfv_multiway_pairwise(&opp_reaches, chip_delta, &blocking_matrix)
    }
}

/// Regret-matching: convert regrets to a per-class probability
/// distribution over actions. Local helper, mirrors the postflop
/// `regret_matching_into` shape but operates on per-class strides.
fn regret_matching_into(regrets: &[f32], na: usize, n_classes: usize, strategy: &mut [f32]) {
    for c in 0..n_classes {
        let mut pos_sum = 0.0f32;
        for a in 0..na {
            let r = regrets[a * n_classes + c];
            if r > REGRET_MATCH_EPS { pos_sum += r; }
        }
        if pos_sum > 0.0 {
            for a in 0..na {
                let r = regrets[a * n_classes + c];
                strategy[a * n_classes + c] = if r > REGRET_MATCH_EPS { r / pos_sum } else { 0.0 };
            }
        } else {
            let uniform = 1.0 / na as f32;
            for a in 0..na {
                strategy[a * n_classes + c] = uniform;
            }
        }
    }
}
