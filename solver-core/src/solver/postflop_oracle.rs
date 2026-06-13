// Postflop value oracle: the seam between preflop CFR and the postflop
// layer. Per the lead's directive (2026-06-04): design this seam BEFORE
// building n-player composition across it, because abstraction (#42) is
// load-bearing for 6-max feasibility, not late-stage polish. Building
// composition against a concrete unabstracted solver would mean reworking
// composition when abstraction lands; building against the trait means
// abstraction is a drop-in.
//
// The seam:
//
//   The preflop engine asks: "at canonical flop F, given per-player
//   combo ranges (per-combo reach at the flop start), what is the per-
//   combo CFV at the flop root for traverser?" The implementation
//   decides how to answer.
//
// Implementations:
//
//   - `UnabstractedPostflopOracle`: runs converged postflop CFR per
//     call. The slow ground-truth reference. Maintains per-canonical
//     persistent CFR state for warm-start across preflop iterations
//     (Fix B's outer/inner cadence). Used directly for validation and
//     as the ground-truth anchor that any abstraction is checked
//     against.
//
//   - `BucketedPostflopOracle` (future, satisfies #42): uses postflop
//     hand-strength bucketing for speed. Per-combo CFV is a bucket-
//     membership lookup plus low-dim solve. Lossy; the lossy-ness is
//     characterized against UnabstractedPostflopOracle as ground truth.
//
//   - `ClosureOracle`: thin compatibility shim that wraps an FnMut
//     closure satisfying the same signature. Used by Slice A's
//     synthetic-callback unit tests (where the per-flop CFV semantics
//     don't matter — only the composition wiring does). Keeps the
//     engine API as the trait (single path) rather than dual-pathing
//     trait-and-closure.
//
// Hook design (begin/end_preflop_iter):
//
//   Per the lead: include now (not after Fix B lands) because Fix B is
//   mandatory at 6-max, not speculative. Defined with the warm-start
//   oracle's usage in mind, not blank no-ops:
//
//     - `begin_preflop_iter(iter)`: called by the engine before its
//       per-traverser loop. Lets the oracle decide refresh cadence:
//       full re-converge (outer cycle) vs warm-start-only (inner
//       cycles). The simple oracle ignores; the warm-start oracle uses
//       this to schedule outer vs inner iterations.
//
//     - `end_preflop_iter(iter)`: called after the traverser loop.
//       Lets the oracle commit per-canonical state, log convergence
//       metrics, decide if next iter is an outer or inner cycle, etc.

use crate::card::Card;
use crate::tree::flat::FlatTree;

/// A preflop→flop SEAM CELL: the flop-entry game context (live player
/// count, the equal per-live commitment, total pot — dead money
/// included). Derived from the flop-entry chance node's folded mask +
/// contributions (the v1_seam_census convention). The strategic
/// coordinate is (live, SPR = (stack − commit)/pot); the validated
/// cell-set policy buckets on it at log2 width 0.25.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeamCell {
    pub live: u8,
    pub commit: i32,
    pub pot: i32,
}

impl SeamCell {
    /// The collapse / bucket key under the VALIDATED policy (4f1dabe):
    /// (live, floor(log2(SPR)/0.25)); all-in cells (behind ≤ 0) get a
    /// per-live sentinel bin (they need no solve — equity rollouts).
    /// `stack` = the game class's total per-player stack.
    pub fn bucket_key(&self, stack: i32) -> (u8, i64) {
        let behind = stack - self.commit;
        if behind <= 0 {
            return (self.live, i64::MIN);
        }
        let spr = behind as f64 / self.pot as f64;
        (self.live, (spr.log2() / 0.25).floor() as i64)
    }

    /// Derive the cell at a flop-entry chance node (the v1_seam_census
    /// convention: live set from the folded mask, equal live commits,
    /// pot = Σ contributions — preflop trees carry no dead
    /// starting_pot; blinds are live contributions).
    pub fn at_chance_node(tree: &FlatTree, chance_idx: usize, np: usize) -> SeamCell {
        let mask = tree.get_folded_mask(chance_idx);
        let mut live = 0u8;
        let mut commit = 0i32;
        let mut pot = 0i32;
        for p in 0..np {
            let c = tree.get_contribution(chance_idx, p as u8);
            pot += c;
            if mask & (1 << p) == 0 {
                live += 1;
                commit = commit.max(c);
            }
        }
        SeamCell { live, commit, pot }
    }
}

/// The interface between the preflop engine and the postflop layer.
///
/// The preflop engine asks for per-combo CFV at the flop root for one
/// traverser, given per-player combo-level reaches at the flop start.
/// Implementations decide whether to compute fresh, warm-start from
/// persisted state, return cached values, or do a bucketed lookup.
///
/// Combo indexing convention: the returned `Vec<f32>` is in
/// `flop_combo_layout(canonical_flop)` order. Oracle implementations
/// that use a different internal layout (e.g., `FlopChanceTable`'s
/// `hand_cards`) must re-order before returning.
///
/// Stateful: implementations may maintain state across calls. The
/// engine doesn't dictate caching, warm-start, or refresh policy;
/// those decisions live inside the oracle. The engine signals
/// iteration boundaries via `begin_preflop_iter` / `end_preflop_iter`
/// so stateful oracles can act on cadence.
pub trait PostflopValueOracle {
    /// Compute per-combo CFV at the flop root for `traverser`, given
    /// per-player combo-level reaches at the flop start.
    ///
    /// `canonical_flop`: the canonical flop board.
    /// `combo_ranges`: `[num_players]` × `[layout_size]` per-player
    ///   per-combo reach at the flop start. Layout matches
    ///   `flop_combo_layout(canonical_flop)`.
    /// `traverser`: which player's CFV to return.
    ///
    /// Returns: `Vec<f32>` of length `layout_size`, per-combo CFV at
    /// the flop root for traverser, indexed in `flop_combo_layout`
    /// order.
    fn flop_root_cfv(
        &mut self,
        canonical_flop: [Card; 3],
        combo_ranges: &[Vec<f32>],
        traverser: u8,
    ) -> Vec<f32>;

    /// SEAM-CELL-AWARE variant (2026-06-12, v1 pot-bucket-keyed
    /// oracles): same contract as `flop_root_cfv`, plus the seam cell
    /// of the flop-entry chance node — (live players, per-live commit,
    /// pot) — so bucket-keyed oracles can dispatch to the right
    /// representative blueprint (the validated SPR cell-set policy:
    /// log2-SPR width 0.25, bin-center reps). Default forwards to the
    /// cell-blind method so existing oracles are unaffected.
    ///
    /// CALLER CONTRACT (mirrors `run_one_iteration_shared_chance`'s):
    /// under the shared-chance collapse, the answer must depend only
    /// on (canonical_flop, traverser, the collapse key) — and the
    /// collapse key must be a function of the cell for cell-keyed
    /// oracles.
    /// `folded_mask`: which of the `combo_ranges.len()` seats folded
    /// PREFLOP (slice 2, 2026-06-12). The traverser is guaranteed LIVE
    /// here — the engine routes folded-traverser chance nodes through
    /// the validated fold machinery (chip-delta × pairwise blocking),
    /// never the oracle. Folded OPPONENTS enter the postflop game only
    /// as dead money (already in `cell.pot`) and card-removal; a
    /// live-subset oracle selects the live seats' ranges via this mask.
    fn flop_root_cfv_for_cell(
        &mut self,
        canonical_flop: [Card; 3],
        combo_ranges: &[Vec<f32>],
        traverser: u8,
        _cell: SeamCell,
        _folded_mask: u16,
    ) -> Vec<f32> {
        self.flop_root_cfv(canonical_flop, combo_ranges, traverser)
    }

    /// Called by the engine BEFORE its per-traverser loop at preflop
    /// iteration `iter`.
    ///
    /// Stateful oracles use this to decide refresh cadence:
    ///   - Simple unabstracted (no warm-start): ignore (default no-op).
    ///   - Warm-start unabstracted: decide whether this iter is an
    ///     OUTER cycle (full re-converge of all canonicals' postflop
    ///     CFR from current ranges) or an INNER cycle (just warm-start
    ///     a few iters per canonical from persisted state). Refresh
    ///     cadence is the oracle's call, not the engine's.
    ///   - Bucketed: may trigger periodic bucket-value refresh based
    ///     on whether preflop ranges have shifted enough.
    ///
    /// Default no-op for implementations that don't need iter
    /// boundaries.
    fn begin_preflop_iter(&mut self, _iter: u32) {}

    /// Called by the engine AFTER its per-traverser loop at preflop
    /// iteration `iter`.
    ///
    /// Stateful oracles use this to:
    ///   - Commit per-canonical CFR state changes (warm-start oracle).
    ///   - Log convergence metrics per canonical (sum of regrets,
    ///     cum_strategy delta, etc.) for diagnostics.
    ///   - Decide whether the next iter is an outer or inner cycle
    ///     based on observed range shifts since the last outer cycle.
    ///
    /// Default no-op.
    fn end_preflop_iter(&mut self, _iter: u32) {}
}

/// Compatibility adapter that wraps an `FnMut` closure satisfying the
/// same call signature into a `PostflopValueOracle`.
///
/// Used by Slice A's synthetic-callback unit tests, where the per-flop
/// CFV's semantics (iter-0 vs converged vs synthetic constant) don't
/// matter and the test wants to drive the composition with a
/// hand-written closure. The engine API stays as the trait; this
/// V1 BUCKET-KEYED ORACLE (2026-06-12, slice 1 of the four-zone
/// runtime): answers per (bucket key, canonical flop, traverser) from
/// the bucket's OWN seam-cell game — the v1 fix over the bootstrap's
/// single-game seam (limp pots and raised pots get DIFFERENT postflop
/// games). The value source is pluggable (`src`): the bootstrap reach
/// run plugs a cheap iter0/equity source; the production fill plugs
/// banked representative lookups. Frozen per epoch: answers cached by
/// (key, flop, traverser) and reused until `begin_preflop_iter`
/// invalidates per the refresh cadence — satisfying the shared-chance
/// collapse contract (answer depends only on (flop, traverser, key)).
///
/// SCOPE (slice 2, 2026-06-12): the traverser is always LIVE here — the
/// engine routes folded-TRAVERSER chance nodes through the validated
/// fold machinery (chip-delta × pairwise blocking), never the oracle,
/// so a folded-traverser call is a wiring bug and the guard fires.
/// Folded OPPONENTS are handled by the value `src`, which receives the
/// `folded_mask` to select the live seats' ranges and build the
/// live-subset postflop game; folded opponents enter only as dead money
/// (already in `cell.pot`) and card-removal (named pairwise-blocking
/// residual, the same approximation the bucketed terminal already
/// makes). The collapse key keys ONLY on (bucket, flop, traverser) —
/// NOT on which seats folded — because the cell-set policy buckets on
/// (live, SPR) and members with the same (live, commit, pot) are the
/// same game regardless of seat identity.
pub struct BucketKeyedOracle<S>
where
    S: FnMut(SeamCell, u16, [Card; 3], &[Vec<f32>], u8) -> Vec<f32>,
{
    pub stack: i32,
    pub num_players: u8,
    src: S,
    cache: std::collections::HashMap<((u8, i64), [Card; 3], u8), Vec<f32>>,
    /// Refresh cadence: cache cleared when iter % refresh_every == 0
    /// at begin_preflop_iter (0 = never refresh: fully frozen).
    pub refresh_every: u32,
}

impl<S> BucketKeyedOracle<S>
where
    S: FnMut(SeamCell, u16, [Card; 3], &[Vec<f32>], u8) -> Vec<f32>,
{
    pub fn new(stack: i32, num_players: u8, refresh_every: u32, src: S) -> Self {
        Self { stack, num_players, src, cache: std::collections::HashMap::new(), refresh_every }
    }
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }
}

impl<S> PostflopValueOracle for BucketKeyedOracle<S>
where
    S: FnMut(SeamCell, u16, [Card; 3], &[Vec<f32>], u8) -> Vec<f32>,
{
    fn flop_root_cfv(
        &mut self,
        _canonical_flop: [Card; 3],
        _combo_ranges: &[Vec<f32>],
        _traverser: u8,
    ) -> Vec<f32> {
        panic!(
            "BucketKeyedOracle requires the cell-aware path (flop_root_cfv_for_cell); \
             a cell-blind call indicates an engine wiring regression"
        );
    }

    fn flop_root_cfv_for_cell(
        &mut self,
        canonical_flop: [Card; 3],
        combo_ranges: &[Vec<f32>],
        traverser: u8,
        cell: SeamCell,
        folded_mask: u16,
    ) -> Vec<f32> {
        // The traverser must be LIVE — folded traversers are handled by
        // the engine, never reach the oracle.
        assert_eq!(
            (folded_mask >> traverser) & 1,
            0,
            "BucketKeyedOracle: folded traverser {traverser} reached the oracle — \
             the engine must route folded-traverser chance nodes through the fold \
             machinery, not the postflop value source"
        );
        let key = (cell.bucket_key(self.stack), canonical_flop, traverser);
        if let Some(v) = self.cache.get(&key) {
            return v.clone();
        }
        let v = (self.src)(cell, folded_mask, canonical_flop, combo_ranges, traverser);
        self.cache.insert(key, v.clone());
        v
    }

    fn begin_preflop_iter(&mut self, iter: u32) {
        if self.refresh_every > 0 && iter > 0 && iter % self.refresh_every == 0 {
            self.cache.clear();
        }
    }
}

/// THE BUCKETED LIVE-SUBSET VALUE SOURCE (2026-06-13, the keystone): a
/// reach-independent frozen postflop oracle source, shared by all three
/// consumers (equilibrium-reach verification, production fill, preflop
/// runtime). Plug into `BucketKeyedOracle::new(.., src)`.
///
/// For a flop-entry cell it solves the LIVE-SUBSET game ONCE (all live
/// traversers) and caches the per-traverser per-combo (flop_combo_layout
/// order) root CFV. live≥3 uses the bucketed solver
/// (`run_all_root_cfv` + `table_hand_to_layout_perm`); live==2 uses the
/// exact converged solver (the bucketed terminal is np≥3 scope). Both
/// solve against the table's structural UNIFORM range, so the answer is
/// REACH-INDEPENDENT — it ignores the passed `combo_reaches` (gated:
/// `bucketed_oracle_range_convention_gate`). That reach-independence is
/// the assumption the whole three-consumer economy rests on: one solved
/// cell answers every chance node routing to it (the shared-chance
/// collapse contract).
///
/// `nt`/`nr` = runout fidelity (turn/river samples). NAMED residual:
/// folded-opponent card-removal is NOT modeled by the live-subset game
/// (slice-2 residual, rides to the harness).
pub fn bucketed_live_subset_source(
    spec: crate::tree::action::GameSpec,
    nb: usize,
    nt: usize,
    nr: usize,
    iters: u32,
) -> impl FnMut(SeamCell, u16, [Card; 3], &[Vec<f32>], u8) -> Vec<f32> {
    use crate::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
    use crate::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
    use crate::solver::preflop_start_game::{
        compute_v_flop_at_root_converged, flop_combo_layout, table_hand_to_layout_perm,
    };
    use crate::tree::action::{BetSize, BetSizeOptions};
    use crate::tree::builder::build_tree;
    use std::collections::HashMap;

    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    // Deterministic nt×nr runout deck positions (the pricing convention).
    let runout = move |flop: [Card; 3]| -> (Vec<u8>, Vec<Vec<u8>>) {
        let bm: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
        let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
        let tp: &[usize] = match nt { 1 => &[12], 2 => &[12, 36], 4 => &[6, 18, 30, 42], _ => &[12] };
        let rp: &[usize] = match nr { 1 => &[10], 2 => &[10, 30], 4 => &[8, 20, 32, 44], _ => &[10] };
        let turn_cards: Vec<u8> = tp.iter().map(|&p| deck[p]).collect();
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
        for &tc in &turn_cards {
            let rd: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
            river_decks[tc as usize] = rp.iter().map(|&p| rd[p]).collect();
        }
        (turn_cards, river_decks)
    };

    let mut trees: HashMap<(u8, i32, i32), std::sync::Arc<FlatTree>> = HashMap::new();
    // (cell-id, flop) -> [live_index][layout-ordered combo CFV].
    let mut cache: HashMap<((u8, i32, i32), [Card; 3]), Vec<Vec<f32>>> = HashMap::new();

    move |cell, folded_mask, canonical, combo_reaches, traverser| {
        let np = combo_reaches.len();
        let live_seats: Vec<usize> =
            (0..np).filter(|&p| (folded_mask >> p) & 1 == 0).collect();
        debug_assert_eq!(live_seats.len(), cell.live as usize);
        let trav_live = live_seats
            .iter()
            .position(|&p| p == traverser as usize)
            .expect("traverser must be live");

        let cell_id = (cell.live, cell.commit, cell.pot);
        if !cache.contains_key(&(cell_id, canonical)) {
            let tree = trees
                .entry(cell_id)
                .or_insert_with(|| {
                    std::sync::Arc::new(
                        build_tree(&spec.flop_seam_config(
                            cell.live,
                            cell.commit.min(spec.stack),
                            cell.pot,
                            bets.clone(),
                        ))
                        .expect("seam tree"),
                    )
                })
                .clone();
            let (turn_cards, river_decks) = runout(canonical);
            let layout = flop_combo_layout(canonical);
            let nlay = layout.len();

            let per_live: Vec<Vec<f32>> = if cell.live >= 3 {
                // Bucketed path: one self-play solve, all live traversers.
                let table =
                    FlopChanceTable::build_full_nh_sampled(canonical, cell.live, &turn_cards, &river_decks);
                let bk = FlopBucketing::quantile(&table, nb);
                let perm = table_hand_to_layout_perm(&table.hand_cards, table.num_valid, canonical);
                let game = FlopStartGame::new(table);
                let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
                solver.set_terminal_design(TerminalDesign::Design1Collapsed);
                let root_all = solver.run_all_root_cfv(&tree, &game, &bk, iters);
                root_all
                    .iter()
                    .map(|per_hand| perm.iter().map(|&h| per_hand[h]).collect())
                    .collect()
            } else {
                // live == 2: exact converged, per traverser, reach-
                // independent (uniform full ranges). Returns table order
                // + its layout; reorder to flop_combo_layout.
                let uni: Vec<Vec<f32>> =
                    vec![vec![1.0f32; crate::card::NUM_POSSIBLE_HANDS]; cell.live as usize];
                (0..cell.live as usize)
                    .map(|t| {
                        let (v_tbl, lay_tbl) = compute_v_flop_at_root_converged(
                            canonical, &tree, &uni, t as u8, iters,
                        );
                        let mut pos: HashMap<(Card, Card), usize> = HashMap::new();
                        for (i, &(a, b)) in lay_tbl.iter().enumerate() {
                            pos.insert((a.min(b), a.max(b)), i);
                        }
                        layout
                            .iter()
                            .map(|&(a, b)| v_tbl[pos[&(a.min(b), a.max(b))]])
                            .collect::<Vec<f32>>()
                    })
                    .collect()
            };
            debug_assert!(per_live.iter().all(|v| v.len() == nlay));
            cache.insert((cell_id, canonical), per_live);
        }
        cache[&(cell_id, canonical)][trav_live].clone()
    }
}

/// adapter keeps tests on the same code path without forcing them to
/// declare a full oracle type.
///
/// Iteration hooks are no-ops: synthetic-callback tests don't have
/// state to manage across iterations.
pub struct ClosureOracle<F>
where
    F: FnMut([Card; 3], &[Vec<f32>], u8) -> Vec<f32>,
{
    f: F,
}

impl<F> ClosureOracle<F>
where
    F: FnMut([Card; 3], &[Vec<f32>], u8) -> Vec<f32>,
{
    pub fn new(f: F) -> Self { Self { f } }
}

impl<F> PostflopValueOracle for ClosureOracle<F>
where
    F: FnMut([Card; 3], &[Vec<f32>], u8) -> Vec<f32>,
{
    fn flop_root_cfv(
        &mut self,
        canonical_flop: [Card; 3],
        combo_ranges: &[Vec<f32>],
        traverser: u8,
    ) -> Vec<f32> {
        (self.f)(canonical_flop, combo_ranges, traverser)
    }
}

/// Unabstracted postflop oracle: the slow ground-truth reference.
///
/// SIMPLE VERSION (v1): every call runs a fresh converged postflop
/// CFR. No warm-start, no persistent per-canonical state. This is the
/// stateless replacement for `make_per_flop_solver_converged` and
/// gives identical results (same DCFR, same num_postflop_iters, same
/// freeze_average_strategy + final pass).
///
/// WARM-START VERSION (Fix B, future): will maintain per-canonical
/// `(FlopChanceTable, FlopStartVectorCfr)` pairs persistent across
/// preflop iterations, with `begin_preflop_iter` deciding outer
/// (full re-converge) vs inner (warm-start few iters) cadence. The
/// trait stays the same; the implementation behind the trait
/// changes.
pub struct UnabstractedPostflopOracle<'a> {
    flop_tree: &'a FlatTree,
    num_postflop_iters: u32,
}

impl<'a> UnabstractedPostflopOracle<'a> {
    pub fn new(flop_tree: &'a FlatTree, num_postflop_iters: u32) -> Self {
        Self { flop_tree, num_postflop_iters }
    }
}

impl<'a> PostflopValueOracle for UnabstractedPostflopOracle<'a> {
    fn flop_root_cfv(
        &mut self,
        canonical_flop: [Card; 3],
        combo_ranges: &[Vec<f32>],
        traverser: u8,
    ) -> Vec<f32> {
        use crate::card::{card_pair_to_index, NUM_POSSIBLE_HANDS};
        use crate::solver::preflop_start_game::{
            compute_v_flop_at_root_converged, flop_combo_layout,
        };
        let layout_engine = flop_combo_layout(canonical_flop);
        let np = combo_ranges.len();
        assert!(np >= 2);
        let mut combo_ranges_full: Vec<Vec<f32>> =
            vec![vec![0.0_f32; NUM_POSSIBLE_HANDS]; np];
        for p in 0..np {
            assert_eq!(combo_ranges[p].len(), layout_engine.len());
            for (li, &(c1, c2)) in layout_engine.iter().enumerate() {
                combo_ranges_full[p][card_pair_to_index(c1, c2)] = combo_ranges[p][li];
            }
        }
        let (v_table_order, layout_table) = compute_v_flop_at_root_converged(
            canonical_flop, self.flop_tree, &combo_ranges_full, traverser,
            self.num_postflop_iters,
        );
        assert_eq!(layout_engine.len(), layout_table.len());
        let mut v_engine_order = Vec::with_capacity(layout_engine.len());
        for &combo in &layout_engine {
            let pos = layout_table.iter().position(|&c| c == combo)
                .unwrap_or_else(|| panic!(
                    "combo {:?} not in table layout for canonical {:?}",
                    combo, canonical_flop));
            v_engine_order.push(v_table_order[pos]);
        }
        v_engine_order
    }

    // begin/end_preflop_iter: no-ops in v1 (stateless). Fix B's warm-
    // start version will use them to decide outer/inner cadence.
}
