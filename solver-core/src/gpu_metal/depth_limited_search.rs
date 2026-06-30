//! GPU depth-limited street search — the runtime entry point that assembles the
//! validated pieces (MetalVectorCfr async loop + bucketed continuation leaf +
//! fast lone-survivor terminal) into one call mirroring the CPU
//! `play_harness::pluribus_play::search_street_strat`.
//!
//! Inputs are the same the CPU search consumes: the chance-free depth-limited
//! tree (continuation leaves = childless chance), the flop `FlopChanceTable`
//! (sorted arrays / hand_cards / nc), the `FlopBucketing` (continuation tables),
//! and the per-player reach prior. Returns the average strategy at every
//! current-street player node, keyed by node id — same shape as the CPU search.

use std::collections::HashMap;

use crate::gpu_metal::{MetalContext, MetalVectorCfr};
use crate::solver::bucketed_flop_cfr::FlopBucketing;
use crate::solver::bucketed_search::ContStreet;
use crate::solver::flop_start_game::FlopChanceTable;
use crate::tree::action::BoardState;
use crate::tree::flat::FlatTree;

/// Configuration for the GPU street search.
pub struct GpuSearchCfg {
    pub iters: u32,
    /// MC samples for the multiway (np>=3) continuation leaf. Ignored for HU
    /// (np=2, exact closed form).
    pub sample_m: u32,
    pub seed: u64,
    /// Use the factored O(nh) lone-survivor terminal (vs the bit-exact brute
    /// O(nh^2)). Both give the same value; factored is far faster at large nh.
    pub factored_terminals: bool,
    /// QRE inverse-temperature λ (0 = off / DCFR). Matches the CPU search's
    /// `set_lambda(cfg.lambda)` — the production runtime uses QRE smoothing.
    pub lambda: f32,
}

impl Default for GpuSearchCfg {
    fn default() -> Self {
        Self { iters: 300, sample_m: 500, seed: 0x5EED, factored_terminals: true, lambda: 0.0 }
    }
}

/// Flop convenience wrapper for [`gpu_search_street_strat`].
pub fn gpu_search_flop_strat(
    ctx: &MetalContext,
    tree: &FlatTree,
    table: &FlopChanceTable,
    bk: &FlopBucketing,
    reach: &[Vec<f32>],
    cfg: &GpuSearchCfg,
) -> HashMap<usize, Vec<Vec<f32>>> {
    gpu_search_street_strat(ctx, tree, table, bk, ContStreet::Flop, reach, cfg)
}

/// Run the GPU depth-limited search on a `build_tree_depth_limited` tree rooted
/// on `cont`'s street (Flop / Turn(ti) / River(ti,ri)) and return the average
/// strategy at every CURRENT-street player node. The depth-limit continuation
/// leaves integrate that street's runout tables:
///   Flop      → bk.flop_tables  / bk.flop_map
///   Turn(ti)  → bk.turn_tables[ti] / bk.turn_map[ti]   (river runout for ti)
///   River(ti,ri) → bk.river_tables[ti][ri] / bk.river_map[ti][ri]
/// (River-rooted trees have no continuation leaf — the river is the last street,
/// so its terminals are real showdowns; set_continuation is then a no-op.)
///
/// `reach[p]` is player p's reach prior at the root ([np][nh]).
pub fn gpu_search_street_strat(
    ctx: &MetalContext,
    tree: &FlatTree,
    table: &FlopChanceTable,
    bk: &FlopBucketing,
    cont: ContStreet,
    reach: &[Vec<f32>],
    cfg: &GpuSearchCfg,
) -> HashMap<usize, Vec<Vec<f32>>> {
    let np = tree.num_players as usize;
    let nh = table.num_valid;
    assert_eq!(reach.len(), np, "reach must have np rows");
    for r in reach { assert_eq!(r.len(), nh, "reach row must be nh long"); }

    // Select the street's bucketing (nb / map / runout tables) + the player-node
    // street tag to read the strategy off.
    let (nb, map, ftab, street_u8) = match cont {
        ContStreet::Flop => (bk.nb_flop, &bk.flop_map, &bk.flop_tables, BoardState::Flop as u8),
        ContStreet::Turn(ti) => (bk.nb_turn, &bk.turn_map[ti], &bk.turn_tables[ti], BoardState::Turn as u8),
        ContStreet::River(ti, ri) => (bk.nb_river, &bk.river_map[ti][ri], &bk.river_tables[ti][ri], BoardState::River as u8),
    };

    let (sos, soi, sps, spi, _) = table.sorted_opp_arrays_base();
    let mut gpu = MetalVectorCfr::new(
        ctx, tree, nh, reach, &sos, &soi, &sps, &spi, &table.hand_cards, table.num_combinations,
    );
    if cfg.lambda > 0.0 {
        gpu.set_lambda(cfg.lambda);
    }

    // Continuation leaves = childless chance nodes; value them with this street's
    // runout tables. (None for a river-rooted tree ⇒ no-op.)
    let leaf_nodes: Vec<u32> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_chance() && tree.node_children(n).is_empty())
        .map(|n| n as u32)
        .collect();
    if !leaf_nodes.is_empty() {
        gpu.set_continuation(
            ctx, &leaf_nodes, map, nb,
            &ftab.f_w, &ftab.f_t, &ftab.f_l, &ftab.f_n,
            tree.rake_rate as f32, tree.rake_cap as f32,
            cfg.sample_m, cfg.seed,
        );
    }

    // Fast lone-survivor terminals: ONLY np==3 (num_opp==2). The kernel reads
    // exactly two opponents; np>=4 would silently drop opponents 3+ and produce
    // wrong cfv, so np>=4 uses the base bottom-up (its K>=3 path is already
    // factored). HU (np==2) folds are already O(nh) inclusion-exclusion.
    if np == 3 {
        let lone: Vec<u32> = (0..tree.num_nodes())
            .filter(|&n| tree.nodes[n].is_terminal())
            .filter(|&n| {
                let fm = tree.get_folded_mask(n);
                (0..np).filter(|&p| fm & (1 << p) == 0).count() <= 1
            })
            .map(|n| n as u32)
            .collect();
        if !lone.is_empty() {
            gpu.set_fast_lone_terminals_ex(ctx, &lone, cfg.factored_terminals);
        }
    }

    gpu.run_batched(ctx, tree, cfg.iters);

    let mut out = HashMap::new();
    for n in 0..tree.num_nodes() {
        if tree.nodes[n].is_player() && tree.nodes[n].board_state == street_u8 {
            let na = tree.nodes[n].num_children as usize;
            out.insert(n, gpu.get_average_strategy(n, na, nh));
        }
    }
    out
}
