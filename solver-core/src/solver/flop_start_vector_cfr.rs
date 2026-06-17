use crate::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use crate::solver::game::GameSpec;
use crate::solver::showdown::side_pot_showdown_cfv_with_rake;
use crate::tree::flat::{FlatTree, MAX_NA_POSTFLOP, VCFR_NO_INFOSET};

const UNUSED: usize = usize::MAX;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Zone { Flop, Turn, River, Preflop }

/// Storage strategy for `regrets_river` and `cum_strategy_river`. At full
/// nh on real-game configs the full per-(tc, rc) buffers are 175 GB each
/// (HU OptB nh=1176), which doesn't fit physical RAM. `DiskBacked` swaps
/// per-pair via load_river_pair/save_river_pair.
///
/// Architecture choice: the access pattern in run() and freeze is per-pair
/// sequential — never all (tc, rc) simultaneously — so disk-backed
/// streaming is feasible. Verified by measurement-C-redux 2026-06-05.
#[derive(Clone)]
pub enum RiverPersistenceMode {
    /// All [n_turn × max_n_river × river_stride] f32 stays in RAM. Default,
    /// fast, but caps the achievable nh by physical memory.
    InMemory,
    /// Two binary files (regrets, cum_strategy) hold the full per-(tc, rc)
    /// state. In-RAM scratch is [river_stride] f32 each. `load_river_pair`
    /// reads from file, `save_river_pair` writes back. Caller (run, freeze)
    /// is responsible for invoking load before any per-pair op and save
    /// after any per-pair mutation.
    DiskBacked {
        regrets_path: std::path::PathBuf,
        cum_strategy_path: std::path::PathBuf,
    },
}

pub struct DcfrParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DcfrParams {
    pub fn new(current_iteration: u32) -> Self {
        let nearest_lower_power_of_4 = match current_iteration {
            0 => 0u32,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };
        let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
        let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;
        let pow_alpha = t_alpha * t_alpha.sqrt();
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);
        Self {
            alpha_t: (pow_alpha / (pow_alpha + 1.0)) as f32,
            beta_t: 0.5,
            gamma_t: pow_gamma as f32,
        }
    }

    /// Accessors for cross-module callers (e.g., preflop_cfr's
    /// bottom-up regret update). Fields stay private to avoid silent
    /// mutation; values are computed at `new()` from the iteration.
    pub fn alpha_t(&self) -> f32 { self.alpha_t }
    pub fn beta_t(&self) -> f32 { self.beta_t }
    pub fn gamma_t(&self) -> f32 { self.gamma_t }
}

/// Per-outcome regrets for flop-start games.
///
/// Each chance outcome gets its own regret/strategy/cum_strategy storage for
/// nodes below that chance event. This matches b1nary's architecture where each
/// expanded subtree node has independent regret storage.
///
/// Layout:
///   Flop zone: one copy (shared across all outcomes)
///   Turn zone: one copy per turn card (n_turn copies)
///   River zone: one copy per (turn_card, river_card) pair (n_turn × n_river copies)
pub struct FlopStartVectorCfr {
    num_players: u8,
    nh: usize,

    // ── Zone classification ──
    zones: Vec<Zone>,                          // [nn]
    flop_zone_nodes: Vec<Vec<u32>>,            // per level
    turn_zone_nodes: Vec<Vec<u32>>,            // per level (template)
    river_zone_nodes: Vec<Vec<u32>>,           // per level (template)
    /// P1.5.2: preflop zone nodes per level. Empty for flop-start trees
    /// (no preflop nodes exist). Non-empty only when the tree's root is
    /// at BoardState::Preflop. Consumed by `zone_nodes_per_level_with_preflop`
    /// for the four-zone walk; the existing 3-tuple
    /// `zone_nodes_per_level` accessor stays unchanged to keep the
    /// blast radius of P1.5.2 to zero existing-consumer churn.
    preflop_zone_nodes: Vec<Vec<u32>>,         // per level

    // ── Per-zone infoset bookkeeping ──
    // For each template decision node, its local infoset offset within its zone.
    // UNUSED if the node is not in this zone or not a decision node.
    flop_local_offset: Vec<usize>,             // [nn]
    turn_local_offset: Vec<usize>,             // [nn]
    river_local_offset: Vec<usize>,            // [nn]
    /// P1.5.2: preflop local offsets. UNUSED everywhere for flop-start trees.
    preflop_local_offset: Vec<usize>,          // [nn]

    flop_infoset_count: usize,
    turn_infoset_count: usize,
    river_infoset_count: usize,
    /// P1.5.2: preflop infoset count. 0 for flop-start trees.
    preflop_infoset_count: usize,

    // ── Per-zone strides ──
    flop_stride: usize,    // flop_infoset_count × MAX_NA × nh
    turn_stride: usize,    // turn_infoset_count × MAX_NA × nh
    river_stride: usize,   // river_infoset_count × MAX_NA × nh

    // ── Per-outcome regret storage ──
    //
    // Memory layout (post 2026-06-05 streaming-strategy refactor):
    //
    //   regrets_* and cum_strategy_*: PERSISTENT across iterations, sized
    //   per-outcome (turn carries n_turn copies; river carries
    //   n_turn × max_n_river copies). These accumulate over iters and
    //   cannot be discarded.
    //
    //   strategy_*: SCRATCH, sized per-stride (one outcome's worth). Each
    //   iter, the strategy for ONE (tc, rc) is regret-matched into this
    //   scratch just before the bottom_up_zone call that reads it. Strategy
    //   does not need to persist across iters or across outcomes — it's
    //   always a function of the current regrets for a specific outcome.
    //
    //   This eliminates 175 GB (river) + 671 MB (turn) of strategy buffer
    //   at HU OptB nh=1176, replacing it with 76 MB + 14 MB scratches.
    //   No semantic change: strategy values for any (tc, rc) are identical
    //   to the pre-refactor full-buffer values, just computed on-demand.

    // Flop: [flop_stride] (small enough to keep full-size for strategy too)
    regrets_flop: Vec<f32>,
    strategy_flop: Vec<f32>,
    cum_strategy_flop: Vec<f32>,

    // Turn: regrets/cum_strategy [n_turn × turn_stride], strategy scratch [turn_stride]
    regrets_turn: Vec<f32>,
    strategy_turn: Vec<f32>,        // SCRATCH: holds current tc's strategy only
    cum_strategy_turn: Vec<f32>,

    // River: regrets/cum_strategy storage depends on river_mode.
    //   - InMemory (default): [n_turn × max_n_river × river_stride] f32 each,
    //     ~175 GB at HU OptB nh=1176. Fits only at small effective nh or with
    //     enough physical RAM.
    //   - DiskBacked: scratch buffer of [river_stride] f32 each, ~76 MB at
    //     HU OptB nh=1176. The full per-(tc, rc) state lives in two binary
    //     files; load_river_pair(tc, rc) reads from file into scratch,
    //     save_river_pair(tc, rc) writes scratch back to file. Total in-RAM
    //     working set: ~76 MB × 3 (regrets + cum_strategy + strategy) per
    //     active pair, swapped one (tc, rc) at a time.
    //
    // Strategy scratch is always per-pair-sized [river_stride] (strategy is
    // derived from regrets per-iter, never persisted across pairs/iters).
    regrets_river: Vec<f32>,
    strategy_river: Vec<f32>,       // SCRATCH: holds current (tc, rc)'s strategy only
    cum_strategy_river: Vec<f32>,

    // River persistence mode + machinery (post 2026-06-05 hybrid A: disk-
    // backed per-board persistence for the 175 GB river buffers).
    river_mode: RiverPersistenceMode,
    /// DiskBacked only: open file handles for the two persistent buffers.
    river_files: Option<(std::fs::File, std::fs::File)>,
    /// DiskBacked only: which (tc, rc) is currently loaded into the scratch
    /// buffers. None = nothing loaded yet (a load is required before the
    /// scratches contain valid data for any pair).
    current_river_pair: Option<(usize, usize)>,

    // ── Chance node bookkeeping ──
    river_chance_children: Vec<u32>,
    turn_chance_children: Vec<u32>,

    // ── Chance outcome counts ──
    n_turn: usize,
    max_n_river: usize,
    // Per-turn-card river deck sizes (for indexing)
    river_deck_sizes: Vec<usize>,  // [52]

    // ── Per-turn-chance-subtree node lists ──
    // For each turn chance child, the list of turn-zone nodes below it.
    // This prevents off-path regret updates during bottom_up_zone.
    turn_subtree_nodes: Vec<Vec<u32>>,
    // For each river chance child, the list of river-zone nodes below it.
    river_subtree_nodes: Vec<Vec<u32>>,
    // Mapping from turn-zone node to which turn chance subtree it belongs to
    // (index into turn_chance_children)
    turn_node_to_subtree: Vec<usize>,  // [nn]
    // Mapping from river-zone node to which river chance subtree it belongs to
    river_node_to_subtree: Vec<usize>, // [nn]

    iteration: u32,
    regret_floor: f32,
    debug: bool,
    vanilla_mode: bool,  // If true, use alpha=beta=gamma=1
    // DEPTH-LIMITED SEARCH (S2, 2026-06-13): when true, bottom_up_zone
    // SKIPS the regret/cum update for Turn+River zones — those zones play
    // a FROZEN blueprint strategy (loaded into strategy_turn/river scratch
    // from the blueprint cum), while the Flop zone is SEARCHED normally.
    // The CFV still propagates through turn/river under the frozen
    // strategy, so the flop search sees "continue with blueprint" leaf
    // values, range-correctly (CFV uses the searched flop reach). Default
    // false → production run()/run_all_root_cfv untouched.
    continuation_frozen: bool,
}

pub(crate) fn mark_descendants(tree: &FlatTree, node_idx: usize, below: &mut [bool]) {
    below[node_idx] = true;
    for &child in tree.node_children(node_idx) {
        mark_descendants(tree, child as usize, below);
    }
}

impl FlopStartVectorCfr {
    pub fn new(tree: &FlatTree, table: &FlopChanceTable) -> Self {
        let nn = tree.num_nodes();
        let max_depth = tree.max_depth as usize;

        // ── Classify zones ──
        // P1.5.2: four-zone classification (Preflop / Flop / Turn / River).
        // For flop-start trees (initial_state = Flop), no Preflop-to-Flop
        // chance node exists, so below_flop stays all-false and the
        // classification collapses to the original three-zone behavior
        // (Preflop bucket stays empty). For preflop-start trees, below_flop
        // is populated by marking descendants of the Preflop-to-Flop
        // chance node (which has board_state == Flop == 0 by the destination
        // convention).
        let mut below_river = vec![false; nn];
        let mut below_turn = vec![false; nn];
        let mut below_flop = vec![false; nn];
        let mut river_cc = Vec::new();
        let mut turn_cc = Vec::new();
        let mut flop_cc = Vec::new();

        for i in 0..nn {
            let n = &tree.nodes[i];
            if n.is_chance() && n.board_state == 2 {
                // River chance node (destination = River)
                for &c in tree.node_children(i) {
                    river_cc.push(c);
                    mark_descendants(tree, c as usize, &mut below_river);
                }
            }
        }
        for i in 0..nn {
            let n = &tree.nodes[i];
            if n.is_chance() && n.board_state == 1 {
                // Turn chance node (destination = Turn)
                for &c in tree.node_children(i) {
                    turn_cc.push(c);
                    mark_descendants(tree, c as usize, &mut below_turn);
                }
            }
        }
        // Preflop-to-Flop chance nodes have board_state == Flop == 0
        // (destination convention). Only present in preflop-start trees.
        // For flop-start trees this loop finds nothing.
        for i in 0..nn {
            let n = &tree.nodes[i];
            if n.is_chance() && n.board_state == 0 {
                for &c in tree.node_children(i) {
                    flop_cc.push(c);
                    mark_descendants(tree, c as usize, &mut below_flop);
                }
            }
        }

        // Root board_state determines the "else" zone: preflop-start trees
        // default unclassified nodes to Preflop; flop-start trees default
        // to Flop (preserving the original behavior). This is the dispatch
        // that lets the same classification handle both tree shapes.
        let root_is_preflop = tree.nodes[0].board_state == 3;

        let zones: Vec<Zone> = (0..nn)
            .map(|i| {
                if below_river[i] { Zone::River }
                else if below_turn[i] { Zone::Turn }
                else if below_flop[i] { Zone::Flop }
                else if root_is_preflop { Zone::Preflop }
                else { Zone::Flop }
            })
            .collect();

        // ── Per-zone node lists ──
        let mut river_zn = vec![vec![]; max_depth + 1];
        let mut turn_zn = vec![vec![]; max_depth + 1];
        let mut flop_zn = vec![vec![]; max_depth + 1];
        let mut preflop_zn = vec![vec![]; max_depth + 1];

        for level in 0..=max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                match zones[nid as usize] {
                    Zone::River => river_zn[level].push(nid),
                    Zone::Turn => turn_zn[level].push(nid),
                    Zone::Flop => flop_zn[level].push(nid),
                    Zone::Preflop => preflop_zn[level].push(nid),
                }
            }
        }

        // ── Per-zone infoset offsets ──
        let mut flop_local_offset = vec![UNUSED; nn];
        let mut turn_local_offset = vec![UNUSED; nn];
        let mut river_local_offset = vec![UNUSED; nn];
        let mut preflop_local_offset = vec![UNUSED; nn];
        let mut flop_count = 0usize;
        let mut turn_count = 0usize;
        let mut river_count = 0usize;
        let mut preflop_count = 0usize;

        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            match zones[idx] {
                Zone::Flop => {
                    flop_local_offset[idx] = flop_count;
                    flop_count += 1;
                }
                Zone::Turn => {
                    turn_local_offset[idx] = turn_count;
                    turn_count += 1;
                }
                Zone::River => {
                    river_local_offset[idx] = river_count;
                    river_count += 1;
                }
                Zone::Preflop => {
                    preflop_local_offset[idx] = preflop_count;
                    preflop_count += 1;
                }
            }
        }

        let nh = table.num_valid;
        let n_turn = table.remaining_deck.len();
        let max_n_river = table.remaining_deck.iter()
            .map(|&tc| table.river_decks[tc as usize].len())
            .max()
            .unwrap_or(0);

        // Strides sourced from ZoneDims — the single stride/offset source
        // shared with the GPU side (Phase B3 dims-threading). Uniform nh
        // here reproduces the pre-bucketing formulas exactly (pinned by
        // zone_dims unit tests).
        let stride_dims = crate::solver::zone_dims::ZoneDims::uniform(
            MAX_NA_POSTFLOP, nh, flop_count, turn_count, river_count,
            n_turn, max_n_river,
        );
        let flop_stride = stride_dims.flop_stride();
        let turn_stride = stride_dims.turn_stride();
        let river_stride = stride_dims.river_stride();

        let mut river_deck_sizes = vec![0usize; 52];
        for &tc in &table.remaining_deck {
            river_deck_sizes[tc as usize] = table.river_decks[tc as usize].len();
        }

        // Compute per-turn-chance-subtree node lists
        let mut turn_subtree_nodes = Vec::new();
        let mut turn_node_to_subtree = vec![UNUSED; nn];
        for (sti, &cc) in turn_cc.iter().enumerate() {
            let mut nodes = Vec::new();
            collect_zone_nodes(tree, cc as usize, &zones, Zone::Turn, &mut nodes);
            for &n in &nodes { turn_node_to_subtree[n as usize] = sti; }
            turn_subtree_nodes.push(nodes);
        }

        // Compute per-river-chance-subtree node lists
        let mut river_subtree_nodes = Vec::new();
        let mut river_node_to_subtree = vec![UNUSED; nn];
        for (sti, &cc) in river_cc.iter().enumerate() {
            let mut nodes = Vec::new();
            collect_zone_nodes(tree, cc as usize, &zones, Zone::River, &mut nodes);
            for &n in &nodes { river_node_to_subtree[n as usize] = sti; }
            river_subtree_nodes.push(nodes);
        }

        FlopStartVectorCfr {
            num_players: tree.num_players,
            nh,
            zones,
            flop_zone_nodes: flop_zn,
            turn_zone_nodes: turn_zn,
            river_zone_nodes: river_zn,
            flop_local_offset,
            turn_local_offset,
            river_local_offset,
            preflop_local_offset,
            preflop_zone_nodes: preflop_zn,
            flop_infoset_count: flop_count,
            turn_infoset_count: turn_count,
            river_infoset_count: river_count,
            preflop_infoset_count: preflop_count,
            flop_stride,
            turn_stride,
            river_stride,
            regrets_flop: vec![0.0; flop_stride],
            strategy_flop: vec![0.0; flop_stride],
            cum_strategy_flop: vec![0.0; flop_stride],
            regrets_turn: vec![0.0; n_turn * turn_stride],
            strategy_turn: vec![0.0; turn_stride],   // SCRATCH: current tc's strategy
            cum_strategy_turn: vec![0.0; n_turn * turn_stride],
            regrets_river: vec![0.0; n_turn * max_n_river * river_stride],
            strategy_river: vec![0.0; river_stride],   // SCRATCH: current (tc, rc)'s strategy
            cum_strategy_river: vec![0.0; n_turn * max_n_river * river_stride],
            river_mode: RiverPersistenceMode::InMemory,
            river_files: None,
            current_river_pair: None,
            river_chance_children: river_cc,
            turn_chance_children: turn_cc,
            n_turn,
            max_n_river,
            river_deck_sizes,
            turn_subtree_nodes,
            river_subtree_nodes,
            turn_node_to_subtree,
            river_node_to_subtree,
            iteration: 0,
            regret_floor: -1e30,
            debug: false,
            vanilla_mode: false,
            continuation_frozen: false,
        }
    }

    // ══════════════════════════════════════════════
    // River persistence (hybrid A: disk-backed per-board state)
    // ══════════════════════════════════════════════

    /// Total length of one persistent river buffer = n_turn × max_n_river × river_stride.
    pub fn river_persistent_len(&self) -> usize {
        self.n_turn * self.max_n_river * self.river_stride
    }

    /// File byte offset for the (tc, rc) slice.
    fn river_pair_byte_offset(&self, tc: usize, rc: usize) -> u64 {
        debug_assert!(tc < self.n_turn && rc < self.max_n_river);
        let pair_index = (tc * self.max_n_river + rc) as u64;
        let stride_bytes = (self.river_stride * std::mem::size_of::<f32>()) as u64;
        pair_index * stride_bytes
    }

    /// Convert this solver from `InMemory` to `DiskBacked` river persistence.
    /// Writes the current full regrets_river / cum_strategy_river content
    /// (initially all zeros) to two binary files, then replaces the in-memory
    /// buffers with per-pair scratch (size river_stride). After this:
    ///   - `regrets_river` and `cum_strategy_river` are SCRATCH for the
    ///     currently-loaded pair only.
    ///   - All per-pair callers (run, freeze, compute_*_strategy_for_pair)
    ///     must invoke `load_river_pair(tc, rc)` before reading and
    ///     `save_river_pair(tc, rc)` after writing.
    ///
    /// In-RAM working set per pair drops from 175 GB (full buffers) to
    /// 76 MB (regrets scratch) + 76 MB (cum_strategy scratch) + 76 MB
    /// (strategy scratch, already there) = ~228 MB at HU OptB nh=1176.
    pub fn into_disk_backed<P: AsRef<std::path::Path>>(
        mut self,
        regrets_path: P,
        cum_strategy_path: P,
    ) -> std::io::Result<Self> {
        use std::io::Write;
        let regrets_path = regrets_path.as_ref().to_path_buf();
        let cum_strategy_path = cum_strategy_path.as_ref().to_path_buf();

        assert!(
            matches!(self.river_mode, RiverPersistenceMode::InMemory),
            "into_disk_backed called on a solver already in DiskBacked mode"
        );

        // Sanity: the full buffers exist at the expected length.
        let full_len = self.river_persistent_len();
        assert_eq!(self.regrets_river.len(), full_len,
            "regrets_river length {} != expected full length {} — InMemory mode invariant violated",
            self.regrets_river.len(), full_len);
        assert_eq!(self.cum_strategy_river.len(), full_len,
            "cum_strategy_river length {} != expected full length {}",
            self.cum_strategy_river.len(), full_len);

        // Persist the current full content to disk.
        {
            let mut f = std::fs::OpenOptions::new()
                .read(true).write(true).create(true).truncate(true)
                .open(&regrets_path)?;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    self.regrets_river.as_ptr() as *const u8,
                    self.regrets_river.len() * std::mem::size_of::<f32>(),
                )
            };
            f.write_all(bytes)?;
            f.flush()?;
        }
        {
            let mut f = std::fs::OpenOptions::new()
                .read(true).write(true).create(true).truncate(true)
                .open(&cum_strategy_path)?;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    self.cum_strategy_river.as_ptr() as *const u8,
                    self.cum_strategy_river.len() * std::mem::size_of::<f32>(),
                )
            };
            f.write_all(bytes)?;
            f.flush()?;
        }

        // Re-open files for read+write (no truncate this time) and store handles.
        let regrets_file = std::fs::OpenOptions::new()
            .read(true).write(true).open(&regrets_path)?;
        let cum_strategy_file = std::fs::OpenOptions::new()
            .read(true).write(true).open(&cum_strategy_path)?;

        // Replace full buffers with per-pair scratches.
        self.regrets_river = vec![0.0; self.river_stride];
        self.cum_strategy_river = vec![0.0; self.river_stride];
        self.river_mode = RiverPersistenceMode::DiskBacked {
            regrets_path, cum_strategy_path,
        };
        self.river_files = Some((regrets_file, cum_strategy_file));
        self.current_river_pair = None;

        Ok(self)
    }

    /// Load the (tc, rc) slice of regrets and cum_strategy into the scratches.
    /// No-op in InMemory mode (data is always "loaded"). In DiskBacked mode,
    /// seeks to the per-pair offset in each file and reads river_stride×4
    /// bytes into the corresponding scratch.
    pub fn load_river_pair(&mut self, tc: usize, rc: usize) -> std::io::Result<()> {
        match &self.river_mode {
            RiverPersistenceMode::InMemory => Ok(()),
            RiverPersistenceMode::DiskBacked { .. } => {
                use std::io::{Read, Seek, SeekFrom};
                let off = self.river_pair_byte_offset(tc, rc);
                let stride = self.river_stride;
                let (rf, cf) = self.river_files.as_mut()
                    .expect("DiskBacked mode requires river_files");
                rf.seek(SeekFrom::Start(off))?;
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.regrets_river.as_mut_ptr() as *mut u8,
                        stride * std::mem::size_of::<f32>(),
                    )
                };
                rf.read_exact(bytes)?;

                cf.seek(SeekFrom::Start(off))?;
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.cum_strategy_river.as_mut_ptr() as *mut u8,
                        stride * std::mem::size_of::<f32>(),
                    )
                };
                cf.read_exact(bytes)?;

                self.current_river_pair = Some((tc, rc));
                Ok(())
            }
        }
    }

    /// Write the scratches back to disk at the (tc, rc) offset.
    /// No-op in InMemory mode. In DiskBacked mode, must be called after
    /// any mutation to regrets_river / cum_strategy_river to persist.
    pub fn save_river_pair(&mut self, tc: usize, rc: usize) -> std::io::Result<()> {
        match &self.river_mode {
            RiverPersistenceMode::InMemory => Ok(()),
            RiverPersistenceMode::DiskBacked { .. } => {
                use std::io::{Seek, SeekFrom, Write};
                debug_assert_eq!(self.current_river_pair, Some((tc, rc)),
                    "save_river_pair({}, {}) called but current loaded pair is {:?}",
                    tc, rc, self.current_river_pair);
                let off = self.river_pair_byte_offset(tc, rc);
                let stride = self.river_stride;
                let (rf, cf) = self.river_files.as_mut()
                    .expect("DiskBacked mode requires river_files");
                rf.seek(SeekFrom::Start(off))?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        self.regrets_river.as_ptr() as *const u8,
                        stride * std::mem::size_of::<f32>(),
                    )
                };
                rf.write_all(bytes)?;

                cf.seek(SeekFrom::Start(off))?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        self.cum_strategy_river.as_ptr() as *const u8,
                        stride * std::mem::size_of::<f32>(),
                    )
                };
                cf.write_all(bytes)?;
                Ok(())
            }
        }
    }

    pub fn river_mode(&self) -> &RiverPersistenceMode { &self.river_mode }

    // ══════════════════════════════════════════════
    // Storage accessors
    // ══════════════════════════════════════════════

    // ══════════════════════════════════════════════
    // Strategy computation (regret matching)
    // ══════════════════════════════════════════════

    /// Compute the flop-zone strategy from current regrets.
    /// Flop strategy is small (flop_stride = flop_infosets × MAX_NA × nh)
    /// and used by every traverser walk, so it's kept fully materialized.
    pub fn compute_flop_strategy(&mut self, tree: &FlatTree) {
        let nh = self.nh;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Flop { continue; }
            let local = self.flop_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let off = local * MAX_NA_POSTFLOP * nh;
            regret_matching_into(
                &self.regrets_flop[off..off + na * nh],
                na, nh,
                &mut self.strategy_flop[off..off + na * nh],
            );
        }
    }

    /// Compute the turn-zone strategy for ONE turn card (tc) from its
    /// per-tc regrets, writing into the strategy_turn SCRATCH (size
    /// turn_stride). Call before any code path that reads turn strategy
    /// for THIS tc (compute_reach_turn, bottom_up_zone(Turn, tc)).
    pub fn compute_turn_strategy_for_tc(&mut self, tree: &FlatTree, tc: usize) {
        let nh = self.nh;
        let base = tc * self.turn_stride;   // regrets are per-tc
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Turn { continue; }
            let local = self.turn_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let off_persist = base + local * MAX_NA_POSTFLOP * nh;
            let off_scratch = local * MAX_NA_POSTFLOP * nh;  // strategy scratch: no tc base
            regret_matching_into(
                &self.regrets_turn[off_persist..off_persist + na * nh],
                na, nh,
                &mut self.strategy_turn[off_scratch..off_scratch + na * nh],
            );
        }
    }

    /// Compute the river-zone strategy for ONE (tc, rc) pair from its
    /// per-(tc,rc) regrets, writing into the strategy_river SCRATCH (size
    /// river_stride). Call before any code path that reads river strategy
    /// for THIS pair (compute_reach_river, bottom_up_zone(River, tc, rc)).
    pub fn compute_river_strategy_for_pair(&mut self, tree: &FlatTree, tc: usize, rc: usize) {
        let nh = self.nh;
        // In InMemory mode, regrets_river holds all (tc, rc) at once; index
        // with the per-pair base. In DiskBacked mode, regrets_river is a
        // scratch holding the currently-loaded pair only; index from 0.
        let base = match self.river_mode {
            RiverPersistenceMode::InMemory => (tc * self.max_n_river + rc) * self.river_stride,
            RiverPersistenceMode::DiskBacked { .. } => {
                debug_assert_eq!(self.current_river_pair, Some((tc, rc)),
                    "compute_river_strategy_for_pair({}, {}) requires matching load_river_pair; \
                     currently loaded: {:?}",
                    tc, rc, self.current_river_pair);
                0
            }
        };
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::River { continue; }
            let local = self.river_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let off_persist = base + local * MAX_NA_POSTFLOP * nh;
            let off_scratch = local * MAX_NA_POSTFLOP * nh;  // strategy scratch: no (tc,rc) base
            regret_matching_into(
                &self.regrets_river[off_persist..off_persist + na * nh],
                na, nh,
                &mut self.strategy_river[off_scratch..off_scratch + na * nh],
            );
        }
    }

    /// DEPRECATED in favor of per-zone / per-pair strategy compute. Kept
    /// for back-compat with existing tests that call this directly. Calling
    /// this populates only the flop strategy (which fits) and the FIRST
    /// turn/river outcome's scratch (the rest of the scratch is undefined
    /// at the start of an iter — callers should call compute_*_strategy_*
    /// per-outcome just-in-time).
    pub fn compute_all_strategies(&mut self, tree: &FlatTree) {
        self.compute_flop_strategy(tree);
        if self.n_turn > 0 {
            self.compute_turn_strategy_for_tc(tree, 0);
            if self.max_n_river > 0 {
                self.compute_river_strategy_for_pair(tree, 0, 0);
            }
        }
    }

    // ══════════════════════════════════════════════
    // Reach probability computation (top-down)
    // ══════════════════════════════════════════════

    /// Compute reach for flop zone (single copy, no outcome dimension).
    pub fn compute_reach_flop(
        &self,
        tree: &FlatTree,
        game: &FlopStartGame,
    ) -> Vec<f32> {
        let nn = tree.num_nodes();
        let np = self.num_players as usize;
        let nh = self.nh;
        let mut reach = vec![0.0f32; nn * np * nh];

        for p in 0..np {
            let w = game.initial_weight(p as u8);
            for h in 0..nh { reach[p * nh + h] = w[h]; }
        }

        if self.debug { eprintln!("  [DEBUG] reach at root: P0={:?} P1={:?}", &reach[0..nh], &reach[nh..2*nh]); }

        for level in 0..=tree.max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                let idx = nid as usize;
                if self.zones[idx] != Zone::Flop { continue; }
                let node = &tree.nodes[idx];

                if node.is_player() {
                    let player = node.player_id as usize;
                    let na = node.num_children as usize;
                    let local = self.flop_local_offset[idx];
                    let sigma = if local != UNUSED {
                        let off = local * MAX_NA_POSTFLOP * nh;
                        &self.strategy_flop[off..off + na * nh]
                    } else {
                        continue;
                    };

                    if self.debug { eprintln!("  [DEBUG] reach P{} node {} (level {}): P0={:?} P1={:?}",
                        player, idx, level,
                        &reach[idx * np * nh..idx * np * nh + nh],
                        &reach[idx * np * nh + nh..idx * np * nh + 2*nh]); }
                    if self.debug {
                        let ch: Vec<u32> = tree.node_children(idx).to_vec();
                        eprintln!("    node {} has {} children: {:?}", idx, ch.len(), ch);
                        for (a, &child) in ch.iter().enumerate() {
                            eprintln!("      sigma[{}] = {:?}", a, &sigma[a * nh..(a+1) * nh]);
                        }
                    }

                    for (a, &child) in tree.node_children(idx).iter().enumerate() {
                        let src = idx * np * nh;
                        let dst = child as usize * np * nh;
                        for p in 0..np {
                            for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                        }
                        for h in 0..nh { reach[dst + player * nh + h] *= sigma[a * nh + h]; }
                    }
                } else if node.is_chance() {
                    if self.debug { eprintln!("  [DEBUG] CHANCE node {} (level {}): P0={:?} P1={:?}",
                        idx, level,
                        &reach[idx * np * nh..idx * np * nh + nh],
                        &reach[idx * np * nh + nh..idx * np * nh + 2*nh]); }
                    // Propagate reach to all chance children (they'll be in turn zone)
                    for &child in tree.node_children(idx) {
                        let src = idx * np * nh;
                        let dst = child as usize * np * nh;
                        for p in 0..np {
                            for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                        }
                    }
                }
            }
        }
        reach
    }

    /// Compute reach for turn zone, for a specific turn card.
    /// Starts from the turn chance children (which received reach from the flop zone).
    pub fn compute_reach_turn(
        &self,
        tree: &FlatTree,
        tc: usize,
        flop_reach: &[f32],  // reach from flop zone (used to seed turn chance children)
    ) -> Vec<f32> {
        let nn = tree.num_nodes();
        let np = self.num_players as usize;
        let nh = self.nh;
        let mut reach = vec![0.0f32; nn * np * nh];

        // Seed from flop zone reach at turn chance children.
        // compute_reach_flop propagates through chance nodes, so the turn chance children
        // already have reach set in flop_reach.
        for &child_id in &self.turn_chance_children {
            let src = child_id as usize * np * nh;
            for p in 0..np {
                for h in 0..nh { reach[src + p * nh + h] = flop_reach[src + p * nh + h]; }
            }
        }

        // Top-down through turn zone using per-tc strategy.
        // POST 2026-06-05 REFACTOR: strategy_turn is now SCRATCH (size turn_stride),
        // holding the strategy for the CURRENT tc only. Caller is responsible for
        // populating it via compute_turn_strategy_for_tc(tc) before calling this.
        let _ = tc; // tc no longer used to index strategy_turn (kept for regrets indexing in callers)
        for level in 0..=tree.max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                let idx = nid as usize;
                if self.zones[idx] != Zone::Turn { continue; }
                let node = &tree.nodes[idx];

                if node.is_player() {
                    let player = node.player_id as usize;
                    let na = node.num_children as usize;
                    let local = self.turn_local_offset[idx];
                    let sigma = if local != UNUSED {
                        let off = local * MAX_NA_POSTFLOP * nh;   // scratch: no tc base
                        &self.strategy_turn[off..off + na * nh]
                    } else {
                        continue;
                    };
                    for (a, &child) in tree.node_children(idx).iter().enumerate() {
                        let src = idx * np * nh;
                        let dst = child as usize * np * nh;
                        for p in 0..np {
                            for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                        }
                        for h in 0..nh { reach[dst + player * nh + h] *= sigma[a * nh + h]; }
                    }
                } else if node.is_chance() {
                    // River chance — propagate to children (they'll be in river zone)
                    for &child in tree.node_children(idx) {
                        let src = idx * np * nh;
                        let dst = child as usize * np * nh;
                        for p in 0..np {
                            for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                        }
                    }
                }
            }
        }
        reach
    }

    /// Compute reach for river zone, for a specific (tc, rc) pair.
    pub fn compute_reach_river(
        &self,
        tree: &FlatTree,
        tc: usize,
        rc: usize,
        turn_reach: &[f32],  // reach from turn zone for this tc
    ) -> Vec<f32> {
        let nn = tree.num_nodes();
        let np = self.num_players as usize;
        let nh = self.nh;
        let mut reach = vec![0.0f32; nn * np * nh];

        // Seed from turn zone reach at river chance children
        for &child_id in &self.river_chance_children {
            let src = child_id as usize * np * nh;
            for p in 0..np {
                for h in 0..nh { reach[src + p * nh + h] = turn_reach[src + p * nh + h]; }
            }
        }

        // Top-down through river zone using per-(tc,rc) strategy.
        // POST 2026-06-05 REFACTOR: strategy_river is now SCRATCH (size river_stride),
        // holding the strategy for the CURRENT (tc, rc) pair only. Caller is responsible
        // for populating it via compute_river_strategy_for_pair(tc, rc) before this.
        let _ = (tc, rc); // tc, rc no longer used to index strategy_river (kept for regrets in callers)
        for level in 0..=tree.max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                let idx = nid as usize;
                if self.zones[idx] != Zone::River { continue; }
                let node = &tree.nodes[idx];

                if node.is_player() {
                    let player = node.player_id as usize;
                    let na = node.num_children as usize;
                    let local = self.river_local_offset[idx];
                    let sigma = if local != UNUSED {
                        let off = local * MAX_NA_POSTFLOP * nh;   // scratch: no (tc,rc) base
                        &self.strategy_river[off..off + na * nh]
                    } else {
                        continue;
                    };
                    for (a, &child) in tree.node_children(idx).iter().enumerate() {
                        let src = idx * np * nh;
                        let dst = child as usize * np * nh;
                        for p in 0..np {
                            for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                        }
                        for h in 0..nh { reach[dst + player * nh + h] *= sigma[a * nh + h]; }
                    }
                }
            }
        }
        reach
    }

    // ══════════════════════════════════════════════
    // Main solver loop
    // ══════════════════════════════════════════════

    pub fn run(
        &mut self,
        tree: &FlatTree,
        game: &FlopStartGame,
        num_iterations: u32,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = self.nh;
        let nn = tree.num_nodes();
        let table = game.table();
        let turn_deck = &table.remaining_deck;

        let mut root_cfv_sum = vec![0.0f32; nh];
        let mut count = 0u32;

        // HOISTED BUFFERS (perf fix, 2026-06-05): previously allocated
        // fresh inside the river/turn loops, producing 13.5k allocations
        // of nn*nh*4 bytes per iter at 6-max (~97 TB of allocator traffic
        // per iter). Each is now reused across all iterations × traversers
        // × turns × rivers. Slots that subsequent code reads as
        // accumulator-zero (river_chance_children / turn_chance_children
        // indices) are explicitly zeroed at the right scope; all other
        // slots are overwritten by bottom_up_zone's deeper-first walk
        // before being read, so stale data is unreachable.
        //
        // Correctness invariant: bottom_up_zone(Z, ...) writes every
        // zone-Z node slot in cfv as a function of cfv at children
        // (which are zone-Z deeper levels, written by this same call,
        // or boundary nodes (the chance-children) seeded just-prior
        // by the explicit accumulator/seed loops below). No stale-read
        // path exists. Validated by p1_5_4_step1_run_alloc_fix_identity
        // bit-exact pre/post and flop_cfv_crosscheck identity.
        let mut flop_cfv = vec![0.0f32; nn * nh];
        let mut river_cfv_accum = vec![0.0f32; nn * nh];
        let mut cfv = vec![0.0f32; nn * nh];
        let mut turn_cfv = vec![0.0f32; nn * nh];

        for _ in 0..num_iterations {
            let params = if self.vanilla_mode {
                DcfrParams { alpha_t: 1.0, beta_t: 1.0, gamma_t: 1.0 }
            } else {
                DcfrParams::new(self.iteration)
            };
            self.iteration += 1;

            for traverser in 0..np {
                // Sequential: recompute strategies before each traverser.
                // POST 2026-06-05 STREAMING REFACTOR: only flop is fully
                // materialized here. Turn strategy is computed per-tc and
                // river strategy is computed per-(tc,rc) just-in-time below.
                self.compute_flop_strategy(tree);

                if self.debug && traverser == 0 {
                    // Print strategy at root node (node 0)
                    let na = tree.nodes[0].num_children as usize;
                    let off = self.flop_local_offset[0] * MAX_NA_POSTFLOP * self.nh;
                    let strat = &self.strategy_flop[off..off + na * self.nh];
                    let mut avg = vec![0.0f32; na];
                    for a in 0..na {
                        for h in 0..self.nh { avg[a] += strat[a * self.nh + h]; }
                        avg[a] /= self.nh as f32;
                    }
                    let regret = &self.regrets_flop[off..off + na * self.nh];
                    let mut r_avg = vec![0.0f32; na];
                    for a in 0..na {
                        for h in 0..self.nh { r_avg[a] += regret[a * self.nh + h]; }
                        r_avg[a] /= self.nh as f32;
                    }
                    eprintln!("  [DEBUG] iter={} root strategy avg: {:?}", self.iteration, avg);
                    eprintln!("  [DEBUG] iter={} root regret avg: {:?}", self.iteration, r_avg);
                }

                // Flop zone reach (shared)
                let flop_reach = self.compute_reach_flop(tree, game);

                // Reset flop_cfv accumulator slots (turn_chance_children) for
                // this traverser. Other flop_cfv slots are overwritten by the
                // bottom_up_zone(Flop) call below (deeper-first walk → fresh
                // writes before reads).
                for &child_id in &self.turn_chance_children {
                    let off = child_id as usize * nh;
                    for h in 0..nh { flop_cfv[off + h] = 0.0; }
                }

                for (ti, &tc) in turn_deck.iter().enumerate() {
                    // STREAMING-STRATEGY: populate turn scratch for this tc
                    // before compute_reach_turn or bottom_up_zone(Turn) read it.
                    self.compute_turn_strategy_for_tc(tree, ti);
                    let turn_reach = self.compute_reach_turn(tree, ti, &flop_reach);
                    let n_river = self.river_deck_sizes[tc as usize];

                    // Reset river_cfv_accum accumulator slots (river_chance_
                    // children) for this turn. Other slots are not read.
                    for &child_id in &self.river_chance_children {
                        let off = child_id as usize * nh;
                        for h in 0..nh { river_cfv_accum[off + h] = 0.0; }
                    }

                    for ri in 0..n_river {
                        // HYBRID A: in DiskBacked mode, read this pair's
                        // regrets+cum_strategy from disk into scratch. No-op in
                        // InMemory mode.
                        self.load_river_pair(ti, ri)
                            .expect("load_river_pair failed (DiskBacked I/O)");
                        // STREAMING-STRATEGY: populate river scratch for this
                        // (ti, ri) before compute_reach_river or bottom_up_zone read it.
                        self.compute_river_strategy_for_pair(tree, ti, ri);
                        let river_reach = self.compute_reach_river(tree, ti, ri, &turn_reach);
                        // cfv: bottom_up_zone(River, ti, ri) overwrites every
                        // river_zone_nodes slot; the read below only accesses
                        // river_chance_children, which are river-zone nodes
                        // written by this call. No reset required.

                        self.bottom_up_zone(
                            tree, table, traverser as u8, &river_reach, &mut cfv,
                            Zone::River, Some(ti), Some(ri),
                            &params,
                        );

                        // HYBRID A: bottom_up_zone wrote into regrets/cum_strategy
                        // scratches. Persist to disk before moving to next pair.
                        self.save_river_pair(ti, ri)
                            .expect("save_river_pair failed (DiskBacked I/O)");

                        // Weight by river chance probability and accumulate
                        for &child_id in &self.river_chance_children {
                            for h in 0..nh {
                                let cp = table.chance_probability_river(tc, ri, h);
                                river_cfv_accum[child_id as usize * nh + h] +=
                                    cp * cfv[child_id as usize * nh + h];
                            }
                        }
                    }

                    // Turn zone seed: overwrite river_chance_children slots
                    // from the just-accumulated river_cfv_accum. Other turn_cfv
                    // slots will be overwritten by bottom_up_zone(Turn) walk.
                    for &child_id in &self.river_chance_children {
                        for h in 0..nh {
                            turn_cfv[child_id as usize * nh + h] =
                                river_cfv_accum[child_id as usize * nh + h];
                        }
                    }

                    self.bottom_up_zone(
                        tree, table, traverser as u8, &turn_reach, &mut turn_cfv,
                        Zone::Turn, Some(ti), None,
                        &params,
                    );

                    // Accumulate at turn chance children weighted by turn chance probability
                    for &child_id in &self.turn_chance_children {
                        for h in 0..nh {
                            let cp = table.chance_probability_turn(ti, h);
                            flop_cfv[child_id as usize * nh + h] +=
                                cp * turn_cfv[child_id as usize * nh + h];
                        }
                    }
                }

                // Flop zone: direct regret update (no accumulation)
                self.bottom_up_zone(
                    tree, table, traverser as u8, &flop_reach, &mut flop_cfv,
                    Zone::Flop, None, None,
                    &params,
                );

                if traverser == 0 {
                    for h in 0..nh { root_cfv_sum[h] += flop_cfv[h]; }
                    count += 1;
                }

                if self.debug {
                    let fc_min = flop_cfv[0..nh].iter().cloned().fold(f32::INFINITY, f32::min);
                    let fc_max = flop_cfv[0..nh].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!("  [DEBUG] iter={} trav={} root_cfv min={:.4} max={:.4}",
                        self.iteration, traverser, fc_min, fc_max);
                }
            }
        }

        for h in 0..nh { root_cfv_sum[h] /= count as f32; }
        root_cfv_sum
    }

    /// ALL-TRAVERSER root CFV (2026-06-13): the EXACT-vector mirror of
    /// `BucketedFlopCfr::run_all_root_cfv`, returning `[np][nh]`. Serves
    /// as the cheap (small-nh) independent reference for the bucketed
    /// keystone's multi-traverser values — the exact CONVERGED source is
    /// the np≥3 combinatorial wall, but the exact VECTOR solver on an
    /// NH=6 subset is fast. Gated bit-exact: `run_all_root_cfv(..)[0]` ≡
    /// `run(..)` (per-traverser-0 arithmetic unchanged; also captures the
    /// others). DiskBacked load/save preserved for faithfulness.
    pub fn run_all_root_cfv(
        &mut self,
        tree: &FlatTree,
        game: &FlopStartGame,
        num_iterations: u32,
    ) -> Vec<Vec<f32>> {
        let np = self.num_players as usize;
        let nh = self.nh;
        let nn = tree.num_nodes();
        let table = game.table();
        let turn_deck = &table.remaining_deck;

        let mut root_all = vec![vec![0.0f32; nh]; np];
        let mut count = 0u32;
        let mut flop_cfv = vec![0.0f32; nn * nh];
        let mut river_cfv_accum = vec![0.0f32; nn * nh];
        let mut cfv = vec![0.0f32; nn * nh];
        let mut turn_cfv = vec![0.0f32; nn * nh];

        for _ in 0..num_iterations {
            let params = if self.vanilla_mode {
                DcfrParams { alpha_t: 1.0, beta_t: 1.0, gamma_t: 1.0 }
            } else {
                DcfrParams::new(self.iteration)
            };
            self.iteration += 1;

            for traverser in 0..np {
                self.compute_flop_strategy(tree);
                let flop_reach = self.compute_reach_flop(tree, game);
                for &child_id in &self.turn_chance_children {
                    let off = child_id as usize * nh;
                    for h in 0..nh { flop_cfv[off + h] = 0.0; }
                }
                for (ti, &tc) in turn_deck.iter().enumerate() {
                    self.compute_turn_strategy_for_tc(tree, ti);
                    let turn_reach = self.compute_reach_turn(tree, ti, &flop_reach);
                    let n_river = self.river_deck_sizes[tc as usize];
                    for &child_id in &self.river_chance_children {
                        let off = child_id as usize * nh;
                        for h in 0..nh { river_cfv_accum[off + h] = 0.0; }
                    }
                    for ri in 0..n_river {
                        self.load_river_pair(ti, ri).expect("load_river_pair");
                        self.compute_river_strategy_for_pair(tree, ti, ri);
                        let river_reach = self.compute_reach_river(tree, ti, ri, &turn_reach);
                        self.bottom_up_zone(
                            tree, table, traverser as u8, &river_reach, &mut cfv,
                            Zone::River, Some(ti), Some(ri), &params,
                        );
                        self.save_river_pair(ti, ri).expect("save_river_pair");
                        for &child_id in &self.river_chance_children {
                            for h in 0..nh {
                                let cp = table.chance_probability_river(tc, ri, h);
                                river_cfv_accum[child_id as usize * nh + h] +=
                                    cp * cfv[child_id as usize * nh + h];
                            }
                        }
                    }
                    for &child_id in &self.river_chance_children {
                        for h in 0..nh {
                            turn_cfv[child_id as usize * nh + h] =
                                river_cfv_accum[child_id as usize * nh + h];
                        }
                    }
                    self.bottom_up_zone(
                        tree, table, traverser as u8, &turn_reach, &mut turn_cfv,
                        Zone::Turn, Some(ti), None, &params,
                    );
                    for &child_id in &self.turn_chance_children {
                        for h in 0..nh {
                            let cp = table.chance_probability_turn(ti, h);
                            flop_cfv[child_id as usize * nh + h] +=
                                cp * turn_cfv[child_id as usize * nh + h];
                        }
                    }
                }
                self.bottom_up_zone(
                    tree, table, traverser as u8, &flop_reach, &mut flop_cfv,
                    Zone::Flop, None, None, &params,
                );
                for h in 0..nh { root_all[traverser][h] += flop_cfv[h]; }
                if traverser == 0 { count += 1; }
            }
        }
        for t in 0..np {
            for h in 0..nh { root_all[t][h] /= count as f32; }
        }
        root_all
    }

    // ══════════════════════════════════════════════
    // Bottom-up zone processing
    // ══════════════════════════════════════════════

    #[allow(clippy::too_many_arguments)]
    pub fn bottom_up_zone(
        &mut self,
        tree: &FlatTree,
        table: &FlopChanceTable,
        traverser: u8,
        reach: &[f32],
        cfv: &mut [f32],
        zone: Zone,
        tc: Option<usize>,
        rc: Option<usize>,
        params: &DcfrParams,
    ) {
        let nh = self.nh;
        let np = self.num_players as usize;
        let max_depth = tree.max_depth as usize;
        let num_opp = np - 1;
        let mut traced = false; // TRACE_SD: one-shot showdown data trace

        // Collect node IDs for this zone/level before borrowing self mutably
        // (zone_nodes borrows self, so we must extract the data before get_mut_slices)
        let zone_node_ids: Vec<Vec<u32>> = match zone {
            Zone::Flop => self.flop_zone_nodes.clone(),
            Zone::Turn => self.turn_zone_nodes.clone(),
            Zone::River => self.river_zone_nodes.clone(),
            Zone::Preflop => unreachable!(
                "Zone::Preflop in flop-start bottom_up_zone; preflop processing \
                 lives in P1.5.4 (#44), this code path handles flop-start trees only"
            ),
        };
        let regret_floor = self.regret_floor;

        for level in (0..=max_depth).rev() {
            for &nid in zone_node_ids.get(level).into_iter().flatten() {
                let idx = nid as usize;
                let node = &tree.nodes[idx];

                if node.is_terminal() {
                    // Terminal evaluation
                    let node_reach_base = idx * np * nh;
                    let c_t = tree.get_contribution(idx, traverser);
                    let fold_mask = tree.get_folded_mask(idx);

                    // Determine board cards for this zone
                    let board_cards: Vec<u8> = match zone {
                        Zone::River => {
                            let ti = tc.unwrap();
                            let ri = rc.unwrap();
                            let tc_card = table.remaining_deck[ti];
                            let rc_card = table.river_decks[tc_card as usize][ri];
                            vec![tc_card, rc_card]
                        }
                        Zone::Turn => {
                            let ti = tc.unwrap();
                            vec![table.remaining_deck[ti]]
                        }
                        Zone::Flop => vec![],
                        Zone::Preflop => unreachable!(
                            "Zone::Preflop in flop-start board_cards selection; \
                             preflop processing lives in P1.5.4 (#44)"
                        ),
                    };

                    // Filter opponent reach to exclude hands conflicting with turn/river cards
                    let filtered_opp: Vec<Vec<f32>> = (0..np)
                        .filter(|&p| p != traverser as usize)
                        .map(|p| {
                            let raw = &reach[node_reach_base + p * nh..node_reach_base + (p + 1) * nh];
                            if board_cards.is_empty() {
                                raw.to_vec()
                            } else {
                                let mut filtered = raw.to_vec();
                                for h in 0..nh {
                                    if filtered[h] != 0.0 {
                                        let c1 = table.hand_cards[h * 2];
                                        let c2 = table.hand_cards[h * 2 + 1];
                                        for &bc in &board_cards {
                                            if c1 == bc || c2 == bc {
                                                filtered[h] = 0.0;
                                                break;
                                            }
                                        }
                                    }
                                }
                                filtered
                            }
                        })
                        .collect();
                    let opp_reach: Vec<&[f32]> = filtered_opp.iter().map(|v| v.as_slice()).collect();

                    let contributions: Vec<i32> = (0..np)
                        .map(|p| tree.get_contribution(idx, p as u8))
                        .collect();

                    // Select sorted arrays based on zone
                    let (opp_str, opp_idx, pl_str, pl_idx) = match zone {
                        Zone::River => {
                            let ti = tc.unwrap();
                            let ri = rc.unwrap();
                            let tc_card = table.remaining_deck[ti];
                            let rc_card = table.river_decks[tc_card as usize][ri];
                            table.river_sorted_arrays(tc_card, rc_card)
                        }
                        Zone::Turn => {
                            let ti = tc.unwrap();
                            let tc_card = table.remaining_deck[ti];
                            table.turn_sorted_arrays(tc_card)
                        }
                        Zone::Flop => {
                            let (os, oi, ps, pi, _) = table.sorted_opp_arrays_base();
                            // Slice 1.6: rake threaded. flop_seen=true
                            // (flop-start solver runs on flop-onward trees).
                            let cfv_out = side_pot_showdown_cfv_with_rake(
                                &opp_reach, &table.hand_cards, nh,
                                &os, &oi, &ps, &pi,
                                &contributions, fold_mask, traverser as usize, self.num_players,
                                tree.starting_pot,
                                tree.rake_rate as f32, tree.rake_cap as f32, true,
                            );
                            let nc = table.num_combinations as f32;
                            if nc > 0.0 {
                                for h in 0..nh { cfv[idx * nh + h] = cfv_out[h] / nc; }
                            } else {
                                for h in 0..nh { cfv[idx * nh + h] = cfv_out[h]; }
                            }
                            continue;
                        }
                        Zone::Preflop => unreachable!(
                            "Zone::Preflop in flop-start sorted-array selection; \
                             preflop processing lives in P1.5.4 (#44)"
                        ),
                    };

                    let cfv_out = side_pot_showdown_cfv_with_rake(
                        &opp_reach, &table.hand_cards, nh,
                        opp_str, opp_idx, pl_str, pl_idx,
                        &contributions, fold_mask, traverser as usize, self.num_players,
                        tree.starting_pot,
                        tree.rake_rate as f32, tree.rake_cap as f32, true,
                    );

                    let nc = table.num_combinations as f32;
                    if nc > 0.0 {
                        for h in 0..nh { cfv[idx * nh + h] = cfv_out[h] / nc; }
                    } else {
                        for h in 0..nh { cfv[idx * nh + h] = cfv_out[h]; }
                    }

                    if !traced && matches!(zone, Zone::River) && std::env::var("TRACE_SD").is_ok() {
                        traced = true;
                        let rsum: f32 = opp_reach[0].iter().sum();
                        let rsq: f32 = opp_reach[0].iter().map(|x| x * x).sum();
                        let coutsq: f32 = cfv_out.iter().map(|x| x * x).sum();
                        let csq: f32 = (0..nh).map(|h| cfv[idx * nh + h] * cfv[idx * nh + h]).sum();
                        eprintln!(
                            "TRACE_SD river-term {idx}: opp_reach[sum {rsum:.5} sumsq {rsq:.7}] | cfv_out sumsq {coutsq:.5} | cfv(/nc) sumsq {csq:.7} | nc {nc:.1}"
                        );
                    }

                } else if node.is_chance() {
                    // Chance node: sum child CFVs
                    for h in 0..nh { cfv[idx * nh + h] = 0.0; }
                    for &child in tree.node_children(idx) {
                        for h in 0..nh {
                            cfv[idx * nh + h] += cfv[child as usize * nh + h];
                        }
                    }

                } else {
                    // Player decision node
                    let owner = node.player_id as usize;
                    let na = node.num_children as usize;
                    // DEPTH-LIMITED SEARCH: Turn+River play a frozen
                    // blueprint here — compute CFV under the frozen strategy
                    // but DO NOT update regrets/cum (no learning in the
                    // continuation). Flop is always searched (frozen=false).
                    // Computed before the mutable get_mut_slices borrow.
                    let frozen = self.continuation_frozen
                        && (zone == Zone::Turn || zone == Zone::River);
                    let (regret_slice, strategy_slice, cum_slice) = self.get_mut_slices(zone, idx, tc, rc, na);

                    let mut cfv_avg = vec![0.0f32; nh];

                    if owner == traverser as usize {
                        // Traverser node: compute CFV and update regrets
                        for a in 0..na {
                            let child = tree.node_children(idx)[a] as usize;
                            for h in 0..nh {
                                cfv_avg[h] += strategy_slice[a * nh + h] * cfv[child * nh + h];
                            }
                        }

                        // Regret update with DCFR discount (skipped for
                        // frozen continuation zones).
                        for a in 0..na {
                            if frozen { break; }
                            let child = tree.node_children(idx)[a] as usize;
                            for h in 0..nh {
                                let inst_regret = cfv[child * nh + h] - cfv_avg[h];
                                let ridx = a * nh + h;

                                let old_r = regret_slice[ridx];
                                let coef = if old_r >= 0.0 { params.alpha_t } else { params.beta_t };
                                regret_slice[ridx] = coef * old_r + inst_regret;
                                if regret_slice[ridx] < regret_floor {
                                    regret_slice[ridx] = regret_floor;
                                }

                                cum_slice[ridx] = params.gamma_t * cum_slice[ridx]
                                    + strategy_slice[a * nh + h];
                            }
                        }
                    } else {
                        // Non-traverser: just aggregate CFV
                        for a in 0..na {
                            let child = tree.node_children(idx)[a] as usize;
                            for h in 0..nh {
                                cfv_avg[h] += cfv[child * nh + h];
                            }
                        }
                    }

                    cfv[idx * nh..idx * nh + nh].copy_from_slice(&cfv_avg);

                    if self.debug && idx == 0 {
                        let cfv_min = cfv[0..nh].iter().cloned().fold(f32::INFINITY, f32::min);
                        let cfv_max = cfv[0..nh].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let cfv_sum: f32 = cfv[0..nh].iter().sum();
                        eprintln!("    [DEBUG] root CFV after bottom_up: min={:.4} max={:.4} avg={:.4}",
                            cfv_min, cfv_max, cfv_sum / nh as f32);
                    }
                }
            }
        }
    }

    /// Get mutable slices for regrets, strategy, and cum_strategy for a given node/outcome.
    /// Returns (regret_slice, strategy_slice, cum_strategy_slice), each of length na × nh.
    fn get_mut_slices(
        &mut self,
        zone: Zone,
        node_id: usize,
        tc: Option<usize>,
        rc: Option<usize>,
        na: usize,
    ) -> (&mut [f32], &[f32], &mut [f32]) {
        let len = na * self.nh;
        match zone {
            Zone::Flop => {
                let off = self.flop_local_offset[node_id] * MAX_NA_POSTFLOP * self.nh;
                let r = &mut self.regrets_flop[off..off + len];
                let s = &self.strategy_flop[off..off + len];
                let c = &mut self.cum_strategy_flop[off..off + len];
                (r, s, c)
            }
            Zone::Turn => {
                let tc = tc.unwrap();
                let local_off = self.turn_local_offset[node_id] * MAX_NA_POSTFLOP * self.nh;
                let off_persist = tc * self.turn_stride + local_off;
                let off_scratch = local_off;  // strategy is SCRATCH (no tc multiplier)
                let r = &mut self.regrets_turn[off_persist..off_persist + len];
                let s = &self.strategy_turn[off_scratch..off_scratch + len];
                let c = &mut self.cum_strategy_turn[off_persist..off_persist + len];
                (r, s, c)
            }
            Zone::River => {
                let tc = tc.unwrap();
                let rc = rc.unwrap();
                let local_off = self.river_local_offset[node_id] * MAX_NA_POSTFLOP * self.nh;
                // In InMemory mode, regrets_river and cum_strategy_river are full
                // [n_turn × max_n_river × river_stride] buffers indexed per-pair.
                // In DiskBacked mode they are river_stride-sized scratches; the
                // currently-loaded pair must match (tc, rc) — load_river_pair(tc, rc)
                // must have been called before this access.
                let off_persist = match self.river_mode {
                    RiverPersistenceMode::InMemory => {
                        (tc * self.max_n_river + rc) * self.river_stride + local_off
                    }
                    RiverPersistenceMode::DiskBacked { .. } => {
                        debug_assert_eq!(self.current_river_pair, Some((tc, rc)),
                            "DiskBacked get_mut_slices(River, tc={}, rc={}) without matching \
                             load_river_pair — currently loaded: {:?}",
                            tc, rc, self.current_river_pair);
                        local_off
                    }
                };
                let off_scratch = local_off;  // strategy is SCRATCH (no (tc,rc) multiplier)
                let r = &mut self.regrets_river[off_persist..off_persist + len];
                let s = &self.strategy_river[off_scratch..off_scratch + len];
                let c = &mut self.cum_strategy_river[off_persist..off_persist + len];
                (r, s, c)
            }
            Zone::Preflop => unreachable!(
                "Zone::Preflop in flop-start get_mut_slices; preflop \
                 regret/strategy storage lives in P1.5.4 (#44)"
            ),
        }
    }

    pub fn iteration_count(&self) -> u32 { self.iteration }
    pub fn set_iteration(&mut self, i: u32) { self.iteration = i; }
    pub fn num_hands(&self) -> usize { self.nh }
    pub fn flop_infosets(&self) -> usize { self.flop_infoset_count }
    pub fn turn_infosets(&self) -> usize { self.turn_infoset_count }
    pub fn river_infosets(&self) -> usize { self.river_infoset_count }
    pub fn n_turn_outcomes(&self) -> usize { self.n_turn }
    pub fn max_river_outcomes(&self) -> usize { self.max_n_river }
    pub fn set_debug(&mut self, d: bool) { self.debug = d; }
    pub fn set_vanilla_mode(&mut self, v: bool) { self.vanilla_mode = v; }
    pub fn set_continuation_frozen(&mut self, v: bool) { self.continuation_frozen = v; }

    /// DEPTH-LIMITED FLOP SEARCH (S2, 2026-06-13). Searches the FLOP zone
    /// while Turn+River play a FROZEN blueprint continuation. Preconditions:
    /// `cum_strategy_turn` / `cum_strategy_river` already hold the blueprint
    /// (e.g. copied from a lifted bucketed solve), and the flop regrets/cum
    /// are at their fresh (zeroed) state so the flop is re-solved.
    ///
    /// Mirrors `run()` exactly EXCEPT: (1) turn/river strategy scratch is
    /// populated by `freeze_average_strategy_for_{turn,river_pair}` (the
    /// blueprint average) instead of regret-matched `compute_*_strategy`,
    /// and (2) `continuation_frozen` gates off the turn/river regret/cum
    /// update inside `bottom_up_zone`. The flop is searched normally.
    ///
    /// This is the S2 instrument: the blueprint's ROUGHNESS (the bucket
    /// count it was solved at, lifted in here) is the tolerance knob;
    /// `num_iterations` is the search-iteration axis; the scored
    /// exploitability of the resulting profile (searched flop + frozen
    /// blueprint turn/river) is the search-output quality. InMemory river
    /// only (research-scale S2); asserts otherwise.
    pub fn run_flop_search(
        &mut self,
        tree: &FlatTree,
        game: &FlopStartGame,
        num_iterations: u32,
    ) {
        assert!(matches!(self.river_mode, RiverPersistenceMode::InMemory),
            "run_flop_search is research-scale (InMemory river) only");
        let np = self.num_players as usize;
        let nh = self.nh;
        let nn = tree.num_nodes();
        let table = game.table();
        let turn_deck = &table.remaining_deck;
        let prev_frozen = self.continuation_frozen;
        self.continuation_frozen = true;

        let mut flop_cfv = vec![0.0f32; nn * nh];
        let mut river_cfv_accum = vec![0.0f32; nn * nh];
        let mut cfv = vec![0.0f32; nn * nh];
        let mut turn_cfv = vec![0.0f32; nn * nh];

        for _ in 0..num_iterations {
            let params = if self.vanilla_mode {
                DcfrParams { alpha_t: 1.0, beta_t: 1.0, gamma_t: 1.0 }
            } else {
                DcfrParams::new(self.iteration)
            };
            self.iteration += 1;

            for traverser in 0..np {
                // Flop is SEARCHED: regret-match its current strategy.
                self.compute_flop_strategy(tree);
                let flop_reach = self.compute_reach_flop(tree, game);

                for &child_id in &self.turn_chance_children {
                    let off = child_id as usize * nh;
                    for h in 0..nh { flop_cfv[off + h] = 0.0; }
                }

                for (ti, &tc) in turn_deck.iter().enumerate() {
                    // FROZEN continuation: load the blueprint turn average
                    // into the strategy scratch (NOT regret-matched).
                    self.freeze_average_strategy_for_turn(tree, ti);
                    let turn_reach = self.compute_reach_turn(tree, ti, &flop_reach);
                    let n_river = self.river_deck_sizes[tc as usize];

                    for &child_id in &self.river_chance_children {
                        let off = child_id as usize * nh;
                        for h in 0..nh { river_cfv_accum[off + h] = 0.0; }
                    }

                    for ri in 0..n_river {
                        // FROZEN continuation: blueprint river average.
                        self.freeze_average_strategy_for_river_pair(tree, ti, ri);
                        let river_reach = self.compute_reach_river(tree, ti, ri, &turn_reach);
                        self.bottom_up_zone(
                            tree, table, traverser as u8, &river_reach, &mut cfv,
                            Zone::River, Some(ti), Some(ri), &params,
                        );
                        for &child_id in &self.river_chance_children {
                            for h in 0..nh {
                                let cp = table.chance_probability_river(tc, ri, h);
                                river_cfv_accum[child_id as usize * nh + h] +=
                                    cp * cfv[child_id as usize * nh + h];
                            }
                        }
                    }

                    for &child_id in &self.river_chance_children {
                        for h in 0..nh {
                            turn_cfv[child_id as usize * nh + h] =
                                river_cfv_accum[child_id as usize * nh + h];
                        }
                    }
                    self.bottom_up_zone(
                        tree, table, traverser as u8, &turn_reach, &mut turn_cfv,
                        Zone::Turn, Some(ti), None, &params,
                    );
                    for &child_id in &self.turn_chance_children {
                        for h in 0..nh {
                            let cp = table.chance_probability_turn(ti, h);
                            flop_cfv[child_id as usize * nh + h] +=
                                cp * turn_cfv[child_id as usize * nh + h];
                        }
                    }
                }

                // Flop zone: SEARCHED (regret/cum update).
                self.bottom_up_zone(
                    tree, table, traverser as u8, &flop_reach, &mut flop_cfv,
                    Zone::Flop, None, None, &params,
                );
            }
        }
        self.continuation_frozen = prev_frozen;
    }

    /// Per-OWNER continuation-variant freeze for the turn scratch (robust
    /// gadget, S2). Opponent-owned turn nodes get the blueprint average
    /// shifted toward `target_action` by `eps`; the traverser's own nodes
    /// keep the pure blueprint. eps==0 ⇒ blueprint everywhere (≡
    /// freeze_average_strategy_for_turn). target_action usize::MAX ⇒
    /// aggressive (last child); else min(target_action, na-1).
    fn freeze_variant_turn(
        &mut self, tree: &FlatTree, tc: usize, traverser: usize, target_action: usize, eps: f32,
    ) {
        let nh = self.nh;
        let base = tc * self.turn_stride;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Turn { continue; }
            let local = self.turn_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let owner = tree.nodes[idx].player_id as usize;
            let off_persist = base + local * MAX_NA_POSTFLOP * nh;
            let off_scratch = local * MAX_NA_POSTFLOP * nh;
            normalize_cum_into_strategy(
                &self.cum_strategy_turn[off_persist..off_persist + na * nh],
                na, nh,
                &mut self.strategy_turn[off_scratch..off_scratch + na * nh],
            );
            if eps > 0.0 && owner != traverser {
                let ta = if target_action == usize::MAX { na - 1 } else { target_action.min(na - 1) };
                let s = &mut self.strategy_turn[off_scratch..off_scratch + na * nh];
                for h in 0..nh {
                    for a in 0..na {
                        let one = if a == ta { 1.0 } else { 0.0 };
                        s[a * nh + h] = (1.0 - eps) * s[a * nh + h] + eps * one;
                    }
                }
            }
        }
    }

    /// Per-OWNER continuation-variant freeze for the river scratch (robust
    /// gadget, S2). Same semantics as `freeze_variant_turn`. InMemory only.
    fn freeze_variant_river(
        &mut self, tree: &FlatTree, tc: usize, rc: usize, traverser: usize,
        target_action: usize, eps: f32,
    ) {
        let nh = self.nh;
        let base = (tc * self.max_n_river + rc) * self.river_stride;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::River { continue; }
            let local = self.river_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let owner = tree.nodes[idx].player_id as usize;
            let off_persist = base + local * MAX_NA_POSTFLOP * nh;
            let off_scratch = local * MAX_NA_POSTFLOP * nh;
            normalize_cum_into_strategy(
                &self.cum_strategy_river[off_persist..off_persist + na * nh],
                na, nh,
                &mut self.strategy_river[off_scratch..off_scratch + na * nh],
            );
            if eps > 0.0 && owner != traverser {
                let ta = if target_action == usize::MAX { na - 1 } else { target_action.min(na - 1) };
                let s = &mut self.strategy_river[off_scratch..off_scratch + na * nh];
                for h in 0..nh {
                    for a in 0..na {
                        let one = if a == ta { 1.0 } else { 0.0 };
                        s[a * nh + h] = (1.0 - eps) * s[a * nh + h] + eps * one;
                    }
                }
            }
        }
    }

    /// MULTI-CONTINUATION robust depth-limited flop search — a ONE-SIDED
    /// PROXY for the gadget (S2, 2026-06-13). At the flop→turn boundary the
    /// OPPONENT commits, per boundary node, to whichever of K precomputed
    /// continuation variants (blueprint + per-owner biased) MINIMIZES the
    /// searcher's aggregate value; the traverser's own continuation stays
    /// the blueprint.
    ///
    /// INCONCLUSIVE / NOT THE FAITHFUL GADGET — DO NOT read its result as
    /// "the gadget fails." This is ONE-SIDED: only the opponent gets a
    /// continuation choice, so the searcher plays a strictly HARDER game
    /// than the real one and the leaf is over-pessimistic — it diverges on
    /// the warm-start diagnostic (per-hand min → 113%, per-node agg → 40%)
    /// for that reason, NOT necessarily because depth-limited search is
    /// broken multiway. The FAITHFUL test is the TWO-SIDED, CFR-solved
    /// continuation-choice gadget (the S1 V_{k0,k1} construction, which
    /// currently exists only on the toy CpuMccfr engine): both players'
    /// per-boundary k-choice co-adapts under CFR, restoring balance. That
    /// port (K^np continuation combos + a per-boundary k-choice regret
    /// layer on this vectorized multi-street engine) is the real gate and
    /// is a substantial build — deliberately NOT taken on faith. Kept here
    /// as the banked one-sided probe + the per-owner variant-freeze
    /// machinery the two-sided build will reuse. InMemory only.
    pub fn run_flop_search_robust(
        &mut self,
        tree: &FlatTree,
        game: &FlopStartGame,
        num_iterations: u32,
    ) {
        assert!(matches!(self.river_mode, RiverPersistenceMode::InMemory),
            "run_flop_search_robust is research-scale (InMemory river) only");
        let np = self.num_players as usize;
        let nh = self.nh;
        let nn = tree.num_nodes();
        let table = game.table();
        let turn_deck = &table.remaining_deck;
        // K continuation variants: (target_action, eps). (_, 0.0)=blueprint.
        let variants: [(usize, f32); 3] =
            [(0, 0.0), (0, 0.6), (usize::MAX, 0.6)]; // blueprint, passive, aggressive
        let prev_frozen = self.continuation_frozen;
        self.continuation_frozen = true;

        let mut flop_cfv_min = vec![0.0f32; nn * nh];
        let mut flop_cfv_k = vec![0.0f32; nn * nh];
        let mut river_cfv_accum = vec![0.0f32; nn * nh];
        let mut cfv = vec![0.0f32; nn * nh];
        let mut turn_cfv = vec![0.0f32; nn * nh];

        for _ in 0..num_iterations {
            let params = if self.vanilla_mode {
                DcfrParams { alpha_t: 1.0, beta_t: 1.0, gamma_t: 1.0 }
            } else {
                DcfrParams::new(self.iteration)
            };
            self.iteration += 1;

            for traverser in 0..np {
                self.compute_flop_strategy(tree);
                let flop_reach = self.compute_reach_flop(tree, game);

                // Robust boundary: opponent commits to ONE continuation
                // variant PER BOUNDARY NODE (aggregate best-response — NOT
                // per-hand omniscient, which over-punishes in multiplayer).
                // best_agg[child] tracks the lowest reach-summed searcher
                // value seen; the winning variant's full per-hand vector is
                // copied in.
                let mut best_agg = vec![f32::INFINITY; nn];
                for &child_id in &self.turn_chance_children {
                    let off = child_id as usize * nh;
                    for h in 0..nh { flop_cfv_min[off + h] = 0.0; }
                }

                for &(ta, eps) in &variants {
                    for &child_id in &self.turn_chance_children {
                        let off = child_id as usize * nh;
                        for h in 0..nh { flop_cfv_k[off + h] = 0.0; }
                    }
                    for (ti, &tc) in turn_deck.iter().enumerate() {
                        self.freeze_variant_turn(tree, ti, traverser, ta, eps);
                        let turn_reach = self.compute_reach_turn(tree, ti, &flop_reach);
                        let n_river = self.river_deck_sizes[tc as usize];
                        for &child_id in &self.river_chance_children {
                            let off = child_id as usize * nh;
                            for h in 0..nh { river_cfv_accum[off + h] = 0.0; }
                        }
                        for ri in 0..n_river {
                            self.freeze_variant_river(tree, ti, ri, traverser, ta, eps);
                            let river_reach = self.compute_reach_river(tree, ti, ri, &turn_reach);
                            self.bottom_up_zone(
                                tree, table, traverser as u8, &river_reach, &mut cfv,
                                Zone::River, Some(ti), Some(ri), &params,
                            );
                            for &child_id in &self.river_chance_children {
                                for h in 0..nh {
                                    let cp = table.chance_probability_river(tc, ri, h);
                                    river_cfv_accum[child_id as usize * nh + h] +=
                                        cp * cfv[child_id as usize * nh + h];
                                }
                            }
                        }
                        for &child_id in &self.river_chance_children {
                            for h in 0..nh {
                                turn_cfv[child_id as usize * nh + h] =
                                    river_cfv_accum[child_id as usize * nh + h];
                            }
                        }
                        self.bottom_up_zone(
                            tree, table, traverser as u8, &turn_reach, &mut turn_cfv,
                            Zone::Turn, Some(ti), None, &params,
                        );
                        for &child_id in &self.turn_chance_children {
                            for h in 0..nh {
                                let cp = table.chance_probability_turn(ti, h);
                                flop_cfv_k[child_id as usize * nh + h] +=
                                    cp * turn_cfv[child_id as usize * nh + h];
                            }
                        }
                    }
                    // Opponent picks the variant minimizing the AGGREGATE
                    // (reach-summed) searcher value at each boundary node.
                    for &child_id in &self.turn_chance_children {
                        let off = child_id as usize * nh;
                        let agg: f32 = (0..nh).map(|h| flop_cfv_k[off + h]).sum();
                        if agg < best_agg[child_id as usize] {
                            best_agg[child_id as usize] = agg;
                            for h in 0..nh { flop_cfv_min[off + h] = flop_cfv_k[off + h]; }
                        }
                    }
                }

                // Flop is SEARCHED against the robust (min) boundary value.
                self.bottom_up_zone(
                    tree, table, traverser as u8, &flop_reach, &mut flop_cfv_min,
                    Zone::Flop, None, None, &params,
                );
            }
        }
        self.continuation_frozen = prev_frozen;
    }

    // ════ TWO-SIDED MULTI-CONTINUATION GADGET (S2 fixability gate) ════
    // The faithful Brown-style gadget on the seam engine: at each
    // flop→turn boundary EVERY player co-adapts a continuation-variant
    // choice k under CFR (regret-matched per boundary × hand). Because the
    // traverser's CFV is LINEAR in each opponent's reach, the opponents'
    // k-choices marginalize by BLENDING their variant strategies (weighted
    // by their current k-strategy) → one blended frozen opponent
    // continuation, so a traverser pass costs only K continuation passes
    // (one per the traverser's own pure variant), NOT K^np. Two-sided ⇒ no
    // one-sided over-pessimism (the flaw that sank the proxy).

    /// Map every Turn/River node to the dense index of the flop→turn
    /// boundary (turn_chance_child) it descends from; MAX for non-cont
    /// nodes. Boundaries are the turn_chance_children (flop's leaves).
    fn build_node_boundary(&self, tree: &FlatTree) -> Vec<usize> {
        let nn = tree.num_nodes();
        let mut nb = vec![usize::MAX; nn];
        for (dense, &broot) in self.turn_chance_children.iter().enumerate() {
            // DFS the continuation subtree rooted at this boundary.
            let mut stack = vec![broot as usize];
            while let Some(idx) = stack.pop() {
                if nb[idx] != usize::MAX { continue; }
                if self.zones[idx] != Zone::Turn && self.zones[idx] != Zone::River { continue; }
                nb[idx] = dense;
                for &c in tree.node_children(idx) { stack.push(c as usize); }
            }
        }
        nb
    }

    /// Two-sided continuation freeze for one zone scratch (turn or river).
    /// Traverser-owned nodes play the PURE variant `k_p`; opponent-owned
    /// nodes play the per-hand BLEND of variants weighted by their current
    /// k-strategy `k_strat` at the node's boundary. k_strat index:
    /// ((b*np + player)*K + k)*nh + h.
    #[allow(clippy::too_many_arguments)]
    fn freeze_two_sided_zone(
        &mut self, tree: &FlatTree, zone: Zone, tc: usize, rc: usize,
        traverser: usize, k_p: usize, k_strat: &[f32], node_boundary: &[usize],
        variants: &[(usize, f32)],
    ) {
        let nh = self.nh;
        let kk = variants.len();
        let np = self.num_players as usize;
        let base = match zone {
            Zone::Turn => tc * self.turn_stride,
            Zone::River => (tc * self.max_n_river + rc) * self.river_stride,
            _ => return,
        };
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != zone { continue; }
            let local = match zone {
                Zone::Turn => self.turn_local_offset[idx],
                Zone::River => self.river_local_offset[idx],
                _ => UNUSED,
            };
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let owner = tree.nodes[idx].player_id as usize;
            let off_persist = base + local * MAX_NA_POSTFLOP * nh;
            let off_scratch = local * MAX_NA_POSTFLOP * nh;
            let b = node_boundary[idx];
            // Blueprint average for this node into a local (immutable
            // borrow of cum, dropped before the mutable strategy write).
            let mut bp = vec![0.0f32; na * nh];
            {
                let cum = match zone {
                    Zone::Turn => &self.cum_strategy_turn,
                    _ => &self.cum_strategy_river,
                };
                normalize_cum_into_strategy(&cum[off_persist..off_persist + na * nh], na, nh, &mut bp);
            }
            // Blend toward variant targets: traverser → pure variant k_p;
            // opponent → per-hand mix by its k-strategy at boundary b.
            let mut out = bp.clone();
            if !(owner == traverser && k_p == 0) {
                for h in 0..nh {
                    let mut shift_total = 0.0f32;
                    let mut add = [0.0f32; 8];
                    for k in 1..kk {
                        let (target, eps) = variants[k];
                        if eps == 0.0 { continue; }
                        let w = if owner == traverser {
                            if k == k_p { 1.0 } else { 0.0 }
                        } else {
                            k_strat[((b * np + owner) * kk + k) * nh + h]
                        };
                        if w == 0.0 { continue; }
                        let ta = if target == usize::MAX { na - 1 } else { target.min(na - 1) };
                        add[ta] += w * eps;
                        shift_total += w * eps;
                    }
                    if shift_total > 0.0 {
                        for a in 0..na {
                            out[a * nh + h] = (1.0 - shift_total) * bp[a * nh + h] + add[a];
                        }
                    }
                }
            }
            let s = match zone {
                Zone::Turn => &mut self.strategy_turn,
                _ => &mut self.strategy_river,
            };
            s[off_scratch..off_scratch + na * nh].copy_from_slice(&out);
        }
    }

    /// FAITHFUL two-sided multi-continuation depth-limited flop search
    /// (S2 fixability gate, 2026-06-13). See module gadget comment. With
    /// `variants` collapsed to a single blueprint entry it reproduces
    /// `run_flop_search` (the control gate). InMemory only.
    pub fn run_flop_search_two_sided(
        &mut self,
        tree: &FlatTree,
        game: &FlopStartGame,
        num_iterations: u32,
        variants: &[(usize, f32)],
    ) {
        assert!(matches!(self.river_mode, RiverPersistenceMode::InMemory),
            "run_flop_search_two_sided is research-scale (InMemory river) only");
        let np = self.num_players as usize;
        let nh = self.nh;
        let nn = tree.num_nodes();
        let kk = variants.len();
        let table = game.table();
        let turn_deck = &table.remaining_deck;
        let node_boundary = self.build_node_boundary(tree);
        let n_b = self.turn_chance_children.len();
        // Boundary node id per dense index, for reading/writing flop_cfv.
        let b_node: Vec<usize> = self.turn_chance_children.iter().map(|&c| c as usize).collect();

        // Per-boundary, per-player, per-k, per-hand continuation-choice
        // regret + cum (the gadget's k-choice CFR layer).
        let kidx = |b: usize, p: usize, k: usize, h: usize| ((b * np + p) * kk + k) * nh + h;
        let mut k_regret = vec![0.0f32; n_b * np * kk * nh];
        let mut k_cum = vec![0.0f32; n_b * np * kk * nh];
        let mut k_strat = vec![0.0f32; n_b * np * kk * nh];

        let prev_frozen = self.continuation_frozen;
        self.continuation_frozen = true;

        let mut flop_cfv = vec![0.0f32; nn * nh];
        let mut river_cfv_accum = vec![0.0f32; nn * nh];
        let mut cfv = vec![0.0f32; nn * nh];
        let mut turn_cfv = vec![0.0f32; nn * nh];
        // V[b][k][h] = traverser boundary value under its pure variant k.
        let mut vval = vec![0.0f32; n_b * kk * nh];

        for _ in 0..num_iterations {
            let params = if self.vanilla_mode {
                DcfrParams { alpha_t: 1.0, beta_t: 1.0, gamma_t: 1.0 }
            } else {
                DcfrParams::new(self.iteration)
            };
            self.iteration += 1;

            // Regret-match the k-choice strategy for ALL (b,p,k,h).
            for b in 0..n_b {
                for p in 0..np {
                    for h in 0..nh {
                        let mut pos = 0.0f32;
                        for k in 0..kk { pos += k_regret[kidx(b, p, k, h)].max(0.0); }
                        for k in 0..kk {
                            k_strat[kidx(b, p, k, h)] = if pos > 0.0 {
                                k_regret[kidx(b, p, k, h)].max(0.0) / pos
                            } else { 1.0 / kk as f32 };
                        }
                    }
                }
            }

            for traverser in 0..np {
                self.compute_flop_strategy(tree);
                let flop_reach = self.compute_reach_flop(tree, game);

                // K continuation passes: traverser plays pure variant k_p,
                // opponents play their blended (k_strat-weighted) variants.
                for k_p in 0..kk {
                    for &child_id in &self.turn_chance_children {
                        let off = child_id as usize * nh;
                        for h in 0..nh { flop_cfv[off + h] = 0.0; }
                    }
                    for (ti, &tc) in turn_deck.iter().enumerate() {
                        self.freeze_two_sided_zone(tree, Zone::Turn, ti, 0, traverser, k_p,
                            &k_strat, &node_boundary, variants);
                        let turn_reach = self.compute_reach_turn(tree, ti, &flop_reach);
                        let n_river = self.river_deck_sizes[tc as usize];
                        for &child_id in &self.river_chance_children {
                            let off = child_id as usize * nh;
                            for h in 0..nh { river_cfv_accum[off + h] = 0.0; }
                        }
                        for ri in 0..n_river {
                            self.freeze_two_sided_zone(tree, Zone::River, ti, ri, traverser, k_p,
                                &k_strat, &node_boundary, variants);
                            let river_reach = self.compute_reach_river(tree, ti, ri, &turn_reach);
                            self.bottom_up_zone(
                                tree, table, traverser as u8, &river_reach, &mut cfv,
                                Zone::River, Some(ti), Some(ri), &params,
                            );
                            for &child_id in &self.river_chance_children {
                                for h in 0..nh {
                                    let cp = table.chance_probability_river(tc, ri, h);
                                    river_cfv_accum[child_id as usize * nh + h] +=
                                        cp * cfv[child_id as usize * nh + h];
                                }
                            }
                        }
                        for &child_id in &self.river_chance_children {
                            for h in 0..nh {
                                turn_cfv[child_id as usize * nh + h] =
                                    river_cfv_accum[child_id as usize * nh + h];
                            }
                        }
                        self.bottom_up_zone(
                            tree, table, traverser as u8, &turn_reach, &mut turn_cfv,
                            Zone::Turn, Some(ti), None, &params,
                        );
                        for &child_id in &self.turn_chance_children {
                            for h in 0..nh {
                                let cp = table.chance_probability_turn(ti, h);
                                flop_cfv[child_id as usize * nh + h] +=
                                    cp * turn_cfv[child_id as usize * nh + h];
                            }
                        }
                    }
                    // Stash this pure-variant boundary value.
                    for b in 0..n_b {
                        let off = b_node[b] * nh;
                        for h in 0..nh { vval[(b * kk + k_p) * nh + h] = flop_cfv[off + h]; }
                    }
                }

                // k-choice CFR at each boundary: mix V by the traverser's
                // k-strategy → the boundary value the flop searches against;
                // update the traverser's k-regret.
                for b in 0..n_b {
                    let off = b_node[b] * nh;
                    for h in 0..nh {
                        let mut mixed = 0.0f32;
                        for k in 0..kk {
                            mixed += k_strat[kidx(b, traverser, k, h)] * vval[(b * kk + k) * nh + h];
                        }
                        flop_cfv[off + h] = mixed;
                        for k in 0..kk {
                            let inst = vval[(b * kk + k) * nh + h] - mixed;
                            let ri = kidx(b, traverser, k, h);
                            let coef = if k_regret[ri] >= 0.0 { params.alpha_t } else { params.beta_t };
                            k_regret[ri] = coef * k_regret[ri] + inst;
                            k_cum[ri] = params.gamma_t * k_cum[ri]
                                + k_strat[kidx(b, traverser, k, h)];
                        }
                    }
                }

                // Search the flop against the gadget boundary values.
                self.bottom_up_zone(
                    tree, table, traverser as u8, &flop_reach, &mut flop_cfv,
                    Zone::Flop, None, None, &params,
                );
            }
        }
        let _ = &k_cum;
        self.continuation_frozen = prev_frozen;
    }

    pub fn best_response_value_debug(&self, tree: &FlatTree, game: &FlopStartGame, p: u8) -> Vec<f32> {
        self.best_response_value(tree, game, p)
    }
    pub fn strategy_value_debug(&self, tree: &FlatTree, game: &FlopStartGame, p: u8) -> Vec<f32> {
        self.strategy_value(tree, game, p)
    }

    pub fn zones(&self) -> &[Zone] { &self.zones }
    pub fn strategy_flop(&self) -> &[f32] { &self.strategy_flop }
    pub fn cum_strategy_flop(&self) -> &[f32] { &self.cum_strategy_flop }
    pub fn regrets_flop(&self) -> &[f32] { &self.regrets_flop }
    pub fn regrets_flop_mut(&mut self) -> &mut [f32] { &mut self.regrets_flop }
    pub fn cum_strategy_flop_mut(&mut self) -> &mut [f32] { &mut self.cum_strategy_flop }
    pub fn turn_local_offset(&self) -> &[usize] { &self.turn_local_offset }
    pub fn turn_stride(&self) -> usize { self.turn_stride }
    pub fn regrets_turn(&self) -> &[f32] { &self.regrets_turn }
    pub fn regrets_turn_mut(&mut self) -> &mut [f32] { &mut self.regrets_turn }
    pub fn cum_strategy_turn(&self) -> &[f32] { &self.cum_strategy_turn }
    pub fn cum_strategy_turn_mut(&mut self) -> &mut [f32] { &mut self.cum_strategy_turn }
    pub fn river_chance_children(&self) -> &[u32] { &self.river_chance_children }
    pub fn turn_chance_children(&self) -> &[u32] { &self.turn_chance_children }
    pub fn flop_stride(&self) -> usize { self.flop_stride }

    /// Local infoset offset for a player decision node within its zone's
    /// strategy/regret/cum_strategy buffers. Returns None if the node has
    /// no infoset slot (UNUSED — typically na ≤ 1 nodes). Caller computes
    /// byte offset as `local * MAX_NA_POSTFLOP * nh`.
    ///
    /// Added 2026-06-07 for the freeze+extract rules anchor (#108) — the
    /// K≥2 walker needs to read σ_avg per (idx, action, hand) from the
    /// frozen strategy buffers, which requires knowing the per-zone local
    /// offset assigned at solver construction.
    pub fn flop_local_offset_at(&self, idx: usize) -> Option<usize> {
        let v = self.flop_local_offset[idx];
        if v == UNUSED { None } else { Some(v) }
    }
    pub fn turn_local_offset_at(&self, idx: usize) -> Option<usize> {
        let v = self.turn_local_offset[idx];
        if v == UNUSED { None } else { Some(v) }
    }
    pub fn river_local_offset_at(&self, idx: usize) -> Option<usize> {
        let v = self.river_local_offset[idx];
        if v == UNUSED { None } else { Some(v) }
    }
    pub fn flop_local_offset(&self) -> &[usize] { &self.flop_local_offset }
    pub fn river_local_offset(&self) -> &[usize] { &self.river_local_offset }
    pub fn num_players(&self) -> u8 { self.num_players }
    pub fn strategy_turn(&self) -> &[f32] { &self.strategy_turn }
    pub fn strategy_river(&self) -> &[f32] { &self.strategy_river }
    pub fn regrets_river(&self) -> &[f32] { &self.regrets_river }
    pub fn regrets_river_mut(&mut self) -> &mut [f32] { &mut self.regrets_river }
    pub fn cum_strategy_river(&self) -> &[f32] { &self.cum_strategy_river }
    pub fn cum_strategy_river_mut(&mut self) -> &mut [f32] { &mut self.cum_strategy_river }
    pub fn river_stride(&self) -> usize { self.river_stride }
    pub fn max_n_river(&self) -> usize { self.max_n_river }

    /// Mutable accessors for strategy buffers. Needed by callers (e.g.,
    /// `compute_v_flop_at_root_converged`) that want to freeze the
    /// time-averaged strategy into the current-strategy slot before
    /// running a final CFV pass.
    pub fn strategy_flop_mut(&mut self) -> &mut [f32] { &mut self.strategy_flop }
    pub fn strategy_turn_mut(&mut self) -> &mut [f32] { &mut self.strategy_turn }
    pub fn strategy_river_mut(&mut self) -> &mut [f32] { &mut self.strategy_river }

    /// After running CFR for N iterations, replace `strategy_{flop,turn,river}`
    /// with the time-averaged strategy (normalized per-infoset per-hand from
    /// cum_strategy). Subsequent `compute_reach_*` + `bottom_up_zone` calls
    /// then compute CFV using the Nash-converging average, not the
    /// oscillating current strategy.
    ///
    /// Per-infoset, per-hand normalization: for each (infoset, hand),
    /// strategy[a, hand] = cum_strategy[a, hand] / Σ_a cum_strategy[a, hand]
    /// if the denominator > 0; else uniform 1/na.
    ///
    /// Stride convention: each infoset gets `MAX_NA_POSTFLOP * nh` floats; indexed
    /// `[a * nh + h]` within an infoset. Outside-na slots are zeroed
    /// (matches uniform-init convention).
    /// Freeze the flop-zone averaged strategy. Flop strategy is fully
    /// materialized (small), so this populates the entire strategy_flop
    /// buffer.
    pub fn freeze_average_strategy_flop(&mut self, tree: &FlatTree) {
        let nh = self.nh;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Flop { continue; }
            let local = self.flop_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let off = local * MAX_NA_POSTFLOP * nh;
            normalize_cum_into_strategy(
                &self.cum_strategy_flop[off..off + na * nh],
                na, nh,
                &mut self.strategy_flop[off..off + na * nh],
            );
        }
    }

    /// Freeze the time-averaged turn-zone strategy for ONE turn card (tc)
    /// into the strategy_turn SCRATCH. Call before any code path that
    /// uses the FROZEN turn strategy for THIS tc (replacing the
    /// regret-matched current strategy for Nash-converging CFV).
    pub fn freeze_average_strategy_for_turn(&mut self, tree: &FlatTree, tc: usize) {
        let nh = self.nh;
        let base = tc * self.turn_stride;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Turn { continue; }
            let local = self.turn_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let off_persist = base + local * MAX_NA_POSTFLOP * nh;
            let off_scratch = local * MAX_NA_POSTFLOP * nh;
            normalize_cum_into_strategy(
                &self.cum_strategy_turn[off_persist..off_persist + na * nh],
                na, nh,
                &mut self.strategy_turn[off_scratch..off_scratch + na * nh],
            );
        }
    }

    /// Freeze the time-averaged river-zone strategy for ONE (tc, rc) pair
    /// into the strategy_river SCRATCH. Call before any code path that
    /// uses the FROZEN river strategy for THIS pair.
    pub fn freeze_average_strategy_for_river_pair(
        &mut self, tree: &FlatTree, tc: usize, rc: usize,
    ) {
        let nh = self.nh;
        let base = match self.river_mode {
            RiverPersistenceMode::InMemory => (tc * self.max_n_river + rc) * self.river_stride,
            RiverPersistenceMode::DiskBacked { .. } => {
                debug_assert_eq!(self.current_river_pair, Some((tc, rc)),
                    "freeze_average_strategy_for_river_pair({}, {}) requires matching \
                     load_river_pair; currently loaded: {:?}",
                    tc, rc, self.current_river_pair);
                0
            }
        };
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::River { continue; }
            let local = self.river_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let off_persist = base + local * MAX_NA_POSTFLOP * nh;
            let off_scratch = local * MAX_NA_POSTFLOP * nh;
            normalize_cum_into_strategy(
                &self.cum_strategy_river[off_persist..off_persist + na * nh],
                na, nh,
                &mut self.strategy_river[off_scratch..off_scratch + na * nh],
            );
        }
    }

    /// DEPRECATED in favor of per-zone/per-pair freeze. Pre 2026-06-05
    /// this iterated all (tc, rc) and wrote the full strategy buffers.
    /// Post-refactor strategy_turn/river are SCRATCH (one outcome's
    /// worth each), so this method now only freezes the flop zone fully
    /// and the first turn/river outcome's scratch. Callers MUST call
    /// freeze_average_strategy_for_turn(tc) / for_river_pair(tc, rc)
    /// just before reading the corresponding strategy slice.
    pub fn freeze_average_strategy(&mut self, tree: &FlatTree) {
        self.freeze_average_strategy_flop(tree);
        if self.n_turn > 0 {
            self.freeze_average_strategy_for_turn(tree, 0);
            if self.max_n_river > 0 {
                self.freeze_average_strategy_for_river_pair(tree, 0, 0);
            }
        }
    }

    /// Return zone nodes organized by level for GPU dispatch.
    /// Returns (river, turn, flop) each as Vec<Vec<u32>> (one Vec per level).
    ///
    /// 3-zone accessor preserved unchanged for the P1.5.2 sub-step's
    /// zero-blast-radius rule: 16 existing files consume this signature,
    /// they continue to work. Preflop-zone consumers use the 4-zone
    /// accessor below.
    pub fn zone_nodes_per_level(&self) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<u32>>) {
        (
            self.river_zone_nodes.clone(),
            self.turn_zone_nodes.clone(),
            self.flop_zone_nodes.clone(),
        )
    }

    /// P1.5.2: four-zone accessor for the preflop integration walk.
    /// Returns (preflop, river, turn, flop), each Vec<Vec<u32>> per level.
    /// Empty preflop vec for flop-start trees. Consumed by P1.5.4
    /// (bottom_up_zone preflop extension) and by P2.5 (orchestration
    /// oracle four-zone walk).
    ///
    /// Note the (preflop, river, turn, flop) ordering: the existing
    /// 3-tuple kept its (river, turn, flop) bottom-up ordering, and
    /// preflop slots at the front because in the bottom-up walk it
    /// is the last zone visited (above flop). Pick whichever ordering
    /// is clearer to consumers at the time; the data is the same.
    pub fn zone_nodes_per_level_with_preflop(&self) -> (
        Vec<Vec<u32>>,
        Vec<Vec<u32>>,
        Vec<Vec<u32>>,
        Vec<Vec<u32>>,
    ) {
        (
            self.preflop_zone_nodes.clone(),
            self.river_zone_nodes.clone(),
            self.turn_zone_nodes.clone(),
            self.flop_zone_nodes.clone(),
        )
    }

    /// P1.5.2: preflop-zone decision node IDs + local infoset offsets.
    /// Empty for flop-start trees. Mirrors `per_zone_decision_nodes` for
    /// the preflop zone only; that 3-zone function is preserved unchanged
    /// to keep blast radius zero.
    pub fn preflop_decision_nodes(&self, tree: &FlatTree) -> (Vec<u32>, Vec<u32>) {
        let mut ids = Vec::new();
        let mut offs = Vec::new();
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] == Zone::Preflop {
                let off = self.preflop_local_offset[idx];
                if off != UNUSED {
                    ids.push(nid);
                    offs.push(off as u32);
                }
            }
        }
        (ids, offs)
    }

    /// Number of preflop-zone decision (infoset) nodes. 0 for flop-start
    /// trees.
    pub fn preflop_infoset_count(&self) -> usize { self.preflop_infoset_count }

    /// Per-zone decision node IDs (for strategy computation on GPU).
    /// Returns (flop_decision_ids, flop_infoset_offsets, turn_decision_ids, turn_infoset_offsets,
    ///          river_decision_ids, river_infoset_offsets).
    pub fn per_zone_decision_nodes(&self, tree: &FlatTree) -> (
        Vec<u32>, Vec<u32>,  // flop
        Vec<u32>, Vec<u32>,  // turn
        Vec<u32>, Vec<u32>,  // river
    ) {
        let mut flop_ids = Vec::new();
        let mut flop_offs = Vec::new();
        let mut turn_ids = Vec::new();
        let mut turn_offs = Vec::new();
        let mut river_ids = Vec::new();
        let mut river_offs = Vec::new();

        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            match self.zones[idx] {
                Zone::Flop => {
                    let off = self.flop_local_offset[idx];
                    if off != UNUSED {
                        flop_ids.push(nid);
                        flop_offs.push(off as u32);
                    }
                }
                Zone::Turn => {
                    let off = self.turn_local_offset[idx];
                    if off != UNUSED {
                        turn_ids.push(nid);
                        turn_offs.push(off as u32);
                    }
                }
                Zone::River => {
                    let off = self.river_local_offset[idx];
                    if off != UNUSED {
                        river_ids.push(nid);
                        river_offs.push(off as u32);
                    }
                }
                Zone::Preflop => {
                    // 3-zone accessor signature is preserved (16 consumers).
                    // Preflop decision nodes are surfaced via the separate
                    // `preflop_decision_nodes()` accessor added in P1.5.2.
                    // For flop-start trees this branch is unreachable; for
                    // preflop-start trees, preflop nodes are emitted on the
                    // dedicated accessor instead, not this 3-zone one.
                }
            }
        }
        (flop_ids, flop_offs, turn_ids, turn_offs, river_ids, river_offs)
    }

    /// Dump regrets and strategies for ALL decision nodes across ALL outcomes.
    /// Only practical for small trees.
    pub fn dump_all_regrets(&self, tree: &FlatTree, game: &FlopStartGame) {
        let table = game.table();
        let nh = self.nh;
        let na_max = MAX_NA_POSTFLOP;

        println!("  ── Flop zone regrets/strategy ──");
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Flop { continue; }
            let local = self.flop_local_offset[idx];
            let na = tree.nodes[idx].num_children as usize;
            let off = local * MAX_NA_POSTFLOP * nh;
            println!("    Node {} (P{}, {} actions):", idx, tree.nodes[idx].player_id, na);
            for a in 0..na {
                let r: Vec<f32> = (0..nh).map(|h| self.regrets_flop[off + a * nh + h]).collect();
                let s: Vec<f32> = (0..nh).map(|h| self.strategy_flop[off + a * nh + h]).collect();
                let c: Vec<f32> = (0..nh).map(|h| self.cum_strategy_flop[off + a * nh + h]).collect();
                println!("      Action {}: regret={:?}", a, r);
                println!("      Action {}: strat ={:?}", a, s);
                println!("      Action {}: cum   ={:?}", a, c);
            }
        }

        println!("  ── Turn zone regrets/strategy (per turn card) ──");
        for ti in 0..self.n_turn {
            let tc_card = table.remaining_deck[ti];
            let base = ti * self.turn_stride;
            println!("    Turn card index {} (card {}):", ti, tc_card);
            for &nid in &tree.decision_node_ids {
                let idx = nid as usize;
                if self.zones[idx] != Zone::Turn { continue; }
                let local = self.turn_local_offset[idx];
                if local == UNUSED { continue; }
                let na = tree.nodes[idx].num_children as usize;
                let off = base + local * MAX_NA_POSTFLOP * nh;
                println!("      Node {} (P{}, {} actions):", idx, tree.nodes[idx].player_id, na);
                for a in 0..na {
                    let r: Vec<f32> = (0..nh).map(|h| self.regrets_turn[off + a * nh + h]).collect();
                    let s: Vec<f32> = (0..nh).map(|h| self.strategy_turn[off + a * nh + h]).collect();
                    println!("        Action {}: regret={:?}", a, r);
                    println!("        Action {}: strat ={:?}", a, s);
                }
            }
        }

        println!("  ── River zone regrets/strategy (per outcome) ──");
        for ti in 0..self.n_turn {
            let tc_card = table.remaining_deck[ti];
            for ri in 0..self.river_deck_sizes[tc_card as usize] {
                let rc_card = table.river_decks[tc_card as usize][ri];
                let base = (ti * self.max_n_river + ri) * self.river_stride;
                println!("    Outcome tc={} (card {}) rc={} (card {}):",
                    ti, tc_card, ri, rc_card);
                for &nid in &tree.decision_node_ids {
                    let idx = nid as usize;
                    if self.zones[idx] != Zone::River { continue; }
                    let local = self.river_local_offset[idx];
                    if local == UNUSED { continue; }
                    let na = tree.nodes[idx].num_children as usize;
                    let off = base + local * MAX_NA_POSTFLOP * nh;
                    println!("        Node {} (P{}, {} actions):", idx, tree.nodes[idx].player_id, na);
                    for a in 0..na {
                        let r: Vec<f32> = (0..nh).map(|h| self.regrets_river[off + a * nh + h]).collect();
                        let s: Vec<f32> = (0..nh).map(|h| self.strategy_river[off + a * nh + h]).collect();
                        println!("          Action {}: regret={:?}", a, r);
                        println!("          Action {}: strat ={:?}", a, s);
                    }
                }
            }
        }
    }

    /// Build a flattened cum_strategy array suitable for the existing exploitability() function.
    ///
    /// The layout maps each template decision node to its per-outcome cum_strategy slice.
    /// For flop zone nodes: use the single shared cum_strategy_flop.
    /// For turn zone nodes: average cum_strategy across all turn cards.
    /// For river zone nodes: average cum_strategy across all (turn, river) pairs.
    ///
    /// This produces a "marginal" average strategy that the tree walker can use.
    pub fn flattened_cum_strategy(&self, tree: &FlatTree) -> (Vec<f32>, Vec<usize>) {
        let nn = tree.num_nodes();
        let nh = self.nh;
        let num_infosets = tree.num_infosets as usize;
        let mut flat_cum = vec![0.0f32; num_infosets * MAX_NA_POSTFLOP * nh];
        let mut offsets = vec![UNUSED; nn];

        // Copy tree's infoset offsets
        for &nid in &tree.decision_node_ids {
            offsets[nid as usize] = tree.infoset_offsets[nid as usize] as usize * MAX_NA_POSTFLOP * nh;
        }

        // Flop zone: direct copy
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Flop { continue; }
            let local = self.flop_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let src_off = local * MAX_NA_POSTFLOP * nh;
            let dst_off = offsets[idx];
            flat_cum[dst_off..dst_off + na * nh]
                .copy_from_slice(&self.cum_strategy_flop[src_off..src_off + na * nh]);
        }

        // Turn zone: average across all turn cards
        let n_turn = self.n_turn;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Turn { continue; }
            let local = self.turn_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let dst_off = offsets[idx];

            for tc in 0..n_turn {
                let src_off = tc * self.turn_stride + local * MAX_NA_POSTFLOP * nh;
                for i in 0..na * nh {
                    flat_cum[dst_off + i] += self.cum_strategy_turn[src_off + i];
                }
            }
            for i in 0..na * nh {
                flat_cum[dst_off + i] /= n_turn as f32;
            }
        }

        // River zone: average across all (turn, river) pairs
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::River { continue; }
            let local = self.river_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let dst_off = offsets[idx];

            let mut count = 0usize;
            for tc in 0..n_turn {
                let n_river = self.river_deck_sizes[
                    /* need turn card, but we don't have the mapping here */
                    0 // placeholder
                ];
                for rc in 0..n_river {
                    let src_off = (tc * self.max_n_river + rc) * self.river_stride + local * MAX_NA_POSTFLOP * nh;
                    for i in 0..na * nh {
                        flat_cum[dst_off + i] += self.cum_strategy_river[src_off + i];
                    }
                    count += 1;
                }
            }
            if count > 0 {
                for i in 0..na * nh {
                    flat_cum[dst_off + i] /= count as f32;
                }
            }
        }

        (flat_cum, offsets)
    }

    /// Compute exploitability using per-outcome strategy.
    /// This walks the tree, and at each decision node selects the strategy
    /// from the correct per-outcome slot based on the game's current chance state.
    pub fn compute_exploitability(
        &self,
        tree: &FlatTree,
        game: &FlopStartGame,
    ) -> f32 {
        let np = self.num_players as usize;
        let mut total_exploit = 0.0f32;

        for p in 0..np {
            let br_val = self.best_response_value(tree, game, p as u8);
            let sv_val = self.strategy_value(tree, game, p as u8);
            let reach = game.initial_weight(p as u8);
            for h in 0..br_val.len() {
                total_exploit += reach[h] * (br_val[h] - sv_val[h]);
            }
        }

        total_exploit / np as f32
    }


    /// Get strategy for a specific outcome context.
    /// Average action distribution at `node_idx` for one `hand` (the
    /// blueprint's normalized average strategy, length `na`). For harness
    /// blueprint policies — the swappable value source SeamGame reads.
    pub fn avg_action_dist(
        &self, node_idx: usize, na: usize, tc: Option<usize>, rc: Option<usize>, hand: usize,
    ) -> Vec<f32> {
        let s = self.get_strategy_for_outcome(node_idx, na, tc, rc);
        (0..na).map(|a| s[a * self.nh + hand]).collect()
    }

    fn get_strategy_for_outcome(
        &self, node_idx: usize, na: usize,
        tc_idx: Option<usize>, rc_idx: Option<usize>,
    ) -> Vec<f32> {
        let nh = self.nh;
        let zone = self.zones[node_idx];
        let len = na * nh;
        let mut strategy = match zone {
            Zone::Flop => {
                let local = self.flop_local_offset[node_idx];
                if local == UNUSED { return vec![1.0 / na as f32; len]; }
                let off = local * MAX_NA_POSTFLOP * nh;
                self.cum_strategy_flop[off..off + len].to_vec()
            }
            Zone::Turn => {
                let tc = tc_idx.unwrap_or(0);
                let local = self.turn_local_offset[node_idx];
                if local == UNUSED { return vec![1.0 / na as f32; len]; }
                let off = tc * self.turn_stride + local * MAX_NA_POSTFLOP * nh;
                self.cum_strategy_turn[off..off + len].to_vec()
            }
            Zone::River => {
                let tc = tc_idx.unwrap_or(0);
                let rc = rc_idx.unwrap_or(0);
                let local = self.river_local_offset[node_idx];
                if local == UNUSED { return vec![1.0 / na as f32; len]; }
                let off = (tc * self.max_n_river + rc) * self.river_stride + local * MAX_NA_POSTFLOP * nh;
                self.cum_strategy_river[off..off + len].to_vec()
            }
            Zone::Preflop => unreachable!(
                "Zone::Preflop in flop-start get_strategy_for_outcome; \
                 preflop cum-strategy storage lives in P1.5.4 (#44)"
            ),
        };
        normalize_strategy(&mut strategy, na, nh);
        strategy
    }

    pub fn best_response_value(&self, tree: &FlatTree, game: &FlopStartGame, target_player: u8) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = game.num_hands(target_player);
        let mut cfreach: Vec<Vec<f32>> = (0..np)
            .map(|p| {
                if p == target_player as usize { vec![1.0; nh] }
                else { game.initial_weight(p as u8) }
            })
            .collect();
        self.walk_br(tree, game, target_player, 0, &mut cfreach, None, None)
    }

    pub fn strategy_value(&self, tree: &FlatTree, game: &FlopStartGame, target_player: u8) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = game.num_hands(target_player);
        let mut cfreach: Vec<Vec<f32>> = (0..np)
            .map(|p| {
                if p == target_player as usize { vec![1.0; nh] }
                else { game.initial_weight(p as u8) }
            })
            .collect();
        self.walk_sv(tree, game, target_player, 0, &mut cfreach, None, None)
    }

    /// Best response walk: at target player nodes, pick the best action.
    fn walk_br(
        &self, tree: &FlatTree, game: &FlopStartGame,
        target_player: u8, node_idx: usize,
        cfreach: &mut [Vec<f32>],
        tc_idx: Option<usize>, rc_idx: Option<usize>,
    ) -> Vec<f32> {
        let node = &tree.nodes[node_idx];

        if node.is_terminal() {
            return game.evaluate_terminal(target_player, node_idx, tree, cfreach);
        }

        if node.is_chance() {
            let children = tree.node_children(node_idx);
            let nh_t = game.num_hands(target_player);
            let mut cfv = vec![0.0f32; nh_t];
            let n_outcomes = game.num_chance_outcomes();

            let child = if !children.is_empty() { children[0] as usize } else { return cfv; };

            for outcome in 0..n_outcomes {
                if outcome > 0 { game.clear_chance_outcome(); }
                let probs: Vec<f32> = (0..nh_t)
                    .map(|h| game.chance_probability(outcome, h)).collect();
                game.set_chance_outcome(outcome);

                let (new_tc, new_rc) = if node.board_state == 1 {
                    (Some(outcome), rc_idx)
                } else if node.board_state == 2 {
                    (tc_idx, Some(outcome))
                } else {
                    (tc_idx, rc_idx)
                };

                let child_cfv = self.walk_br(tree, game, target_player, child, cfreach, new_tc, new_rc);
                for h in 0..nh_t { cfv[h] += probs[h] * child_cfv[h]; }
            }
            game.clear_chance_outcome();
            return cfv;
        }

        let player = node.player_id;
        let num_actions = node.num_children as usize;
        if num_actions == 0 {
            let nh_t = game.num_hands(target_player);
            return vec![0.0f32; nh_t];
        }
        let nh = game.num_hands(player);
        let children = tree.node_children(node_idx);
        let sigma = self.get_strategy_for_outcome(node_idx, num_actions, tc_idx, rc_idx);

        let mut cfv_all: Vec<Vec<f32>> = Vec::with_capacity(num_actions);
        for (a, &child) in children.iter().enumerate() {
            let saved = cfreach[player as usize].clone();
            for h in 0..nh { cfreach[player as usize][h] = saved[h] * sigma[a * nh + h]; }
            cfv_all.push(self.walk_br(tree, game, target_player, child as usize, cfreach, tc_idx, rc_idx));
            cfreach[player as usize] = saved;
        }

        let nh_t = cfv_all[0].len();
        let mut result = vec![0.0f32; nh_t];
        if player == target_player {
            for h in 0..nh_t {
                let mut best = f32::NEG_INFINITY;
                for a in 0..num_actions { best = best.max(cfv_all[a][h]); }
                result[h] = best;
            }
        } else {
            // Non-target player: unweighted sum (sigma already in cfreach)
            for a in 0..num_actions {
                for h in 0..nh_t { result[h] += cfv_all[a][h]; }
            }
        }
        result
    }

    /// Strategy value walk: follow the strategy at all player nodes.
    fn walk_sv(
        &self, tree: &FlatTree, game: &FlopStartGame,
        target_player: u8, node_idx: usize,
        cfreach: &mut [Vec<f32>],
        tc_idx: Option<usize>, rc_idx: Option<usize>,
    ) -> Vec<f32> {
        let node = &tree.nodes[node_idx];

        if node.is_terminal() {
            return game.evaluate_terminal(target_player, node_idx, tree, cfreach);
        }

        if node.is_chance() {
            let children = tree.node_children(node_idx);
            let nh_t = game.num_hands(target_player);
            let mut cfv = vec![0.0f32; nh_t];
            let n_outcomes = game.num_chance_outcomes();

            let child = if !children.is_empty() { children[0] as usize } else { return cfv; };

            for outcome in 0..n_outcomes {
                if outcome > 0 { game.clear_chance_outcome(); }
                let probs: Vec<f32> = (0..nh_t)
                    .map(|h| game.chance_probability(outcome, h)).collect();
                game.set_chance_outcome(outcome);

                let (new_tc, new_rc) = if node.board_state == 1 {
                    (Some(outcome), rc_idx)
                } else if node.board_state == 2 {
                    (tc_idx, Some(outcome))
                } else {
                    (tc_idx, rc_idx)
                };

                let child_cfv = self.walk_sv(tree, game, target_player, child, cfreach, new_tc, new_rc);
                for h in 0..nh_t { cfv[h] += probs[h] * child_cfv[h]; }
            }
            game.clear_chance_outcome();
            return cfv;
        }

        let player = node.player_id;
        let num_actions = node.num_children as usize;
        if num_actions == 0 {
            let nh_t = game.num_hands(target_player);
            return vec![0.0f32; nh_t];
        }
        let nh = game.num_hands(player);
        let children = tree.node_children(node_idx);
        let sigma = self.get_strategy_for_outcome(node_idx, num_actions, tc_idx, rc_idx);

        let mut cfv_all: Vec<Vec<f32>> = Vec::with_capacity(num_actions);
        for (a, &child) in children.iter().enumerate() {
            let saved = cfreach[player as usize].clone();
            for h in 0..nh { cfreach[player as usize][h] = saved[h] * sigma[a * nh + h]; }
            cfv_all.push(self.walk_sv(tree, game, target_player, child as usize, cfreach, tc_idx, rc_idx));
            cfreach[player as usize] = saved;
        }

        let nh_t = cfv_all[0].len();
        let mut cfv = vec![0.0f32; nh_t];
        if player == target_player {
            // Target player: sigma-weighted sum (strategy value)
            for (a, cfv_a) in cfv_all.iter().enumerate() {
                for h in 0..nh_t { cfv[h] += sigma[a * nh + h] * cfv_a[h]; }
            }
        } else {
            // Non-target player: unweighted sum (sigma already in cfreach)
            for a in 0..num_actions {
                for h in 0..nh_t { cfv[h] += cfv_all[a][h]; }
            }
        }
        cfv
    }
}

fn normalize_strategy(strategy: &mut [f32], na: usize, nh: usize) {
    for h in 0..nh {
        let mut total = 0.0f32;
        for a in 0..na { total += strategy[a * nh + h]; }
        if total > 0.0 {
            for a in 0..na { strategy[a * nh + h] /= total; }
        } else {
            let uniform = 1.0 / na as f32;
            for a in 0..na { strategy[a * nh + h] = uniform; }
        }
    }
}

/// Note: regret_matching_into is defined above (it was in the head section).

/// Regret matching: compute strategy from regrets.
/// Bug B fix: regret matching epsilon to prevent ULP-level strategy flips.
const REGRET_MATCH_EPS: f32 = 1e-5;

/// Normalize cumulative-strategy into a per-(infoset, hand) probability
/// distribution over actions. Same layout convention as
/// `regret_matching_into` (per-infoset slice of `na * nh`, indexed
/// `[a * nh + h]`).
///
/// Identity property: at convergence the average strategy is the Nash
/// strategy; this function exposes it for callers (e.g. preflop CFR's
/// converged per-flop solver) that need to compute CFV under the
/// averaged strategy rather than the oscillating current-iter strategy.
fn normalize_cum_into_strategy(cum: &[f32], na: usize, nh: usize, strategy: &mut [f32]) {
    for h in 0..nh {
        let mut sum = 0.0f32;
        for a in 0..na {
            let v = cum[a * nh + h];
            if v > 0.0 { sum += v; }
        }
        if sum > 0.0 {
            for a in 0..na {
                let v = cum[a * nh + h];
                strategy[a * nh + h] = if v > 0.0 { v / sum } else { 0.0 };
            }
        } else {
            let uniform = 1.0 / na as f32;
            for a in 0..na {
                strategy[a * nh + h] = uniform;
            }
        }
    }
}

pub(crate) fn regret_matching_into(regrets: &[f32], na: usize, nh: usize, strategy: &mut [f32]) {
    for h in 0..nh {
        let mut pos_sum = 0.0f32;
        for a in 0..na {
            let r = regrets[a * nh + h];
            if r > REGRET_MATCH_EPS { pos_sum += r; }
        }
        if pos_sum > 0.0 {
            for a in 0..na {
                let r = regrets[a * nh + h];
                strategy[a * nh + h] = if r > REGRET_MATCH_EPS { r / pos_sum } else { 0.0 };
            }
        } else {
            let uniform = 1.0 / na as f32;
            for a in 0..na {
                strategy[a * nh + h] = uniform;
            }
        }
    }
}

/// Collect nodes in a specific zone that are descendants of a given root,
/// stopping at chance nodes that lead into a different zone.
pub(crate) fn collect_zone_nodes(
    tree: &FlatTree, node_idx: usize,
    zones: &[Zone], zone: Zone,
    nodes: &mut Vec<u32>,
) {
    if zones[node_idx] != zone { return; }
    let node = &tree.nodes[node_idx];
    if node.is_player() {
        nodes.push(node_idx as u32);
    }
    for &child in tree.node_children(node_idx) {
        collect_zone_nodes(tree, child as usize, zones, zone, nodes);
    }
}
