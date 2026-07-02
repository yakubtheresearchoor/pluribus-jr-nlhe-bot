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
    /// HARD wall-clock budget for the solve (ms). The GPU loop stops committing
    /// iterations once spent (worst-case overrun ≈ one 25-iter chunk) — the
    /// runaway guard: without it a deep multiway tree can pin the GPU for
    /// minutes, and a client disconnect leaves that work grinding orphaned.
    /// Mirrors the CPU path's adaptive LIVE2_RT_BUDGET_MS.
    pub budget_ms: u64,
}

impl Default for GpuSearchCfg {
    fn default() -> Self {
        Self { iters: 300, sample_m: 500, seed: 0x5EED, factored_terminals: true, lambda: 0.0, budget_ms: 9_000 }
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

    let ran = gpu.run_batched_budget(ctx, tree, cfg.iters, cfg.budget_ms);
    if ran < cfg.iters {
        eprintln!("[gpu-search] budget hit: ran {ran}/{} iters in {}ms budget", cfg.iters, cfg.budget_ms);
    }

    let mut out = HashMap::new();
    for n in 0..tree.num_nodes() {
        if tree.nodes[n].is_player() && tree.nodes[n].board_state == street_u8 {
            let na = tree.nodes[n].num_children as usize;
            out.insert(n, gpu.get_average_strategy(n, na, nh));
        }
    }
    out
}

/// GPU HU (np=2) TURN search on the ACTUAL board — the fast, fully-converged
/// alternative to the CPU `solve_live2_street` (which is budget-limited to ~43
/// iters at 208ms/iter). Builds the turn continuation tables directly from the
/// real board (per-hand best-5-of-6 strength → nb strength-quantile buckets →
/// `compute_wtl_for_runout`), the SAME single-strength continuation model the
/// validated `gpu_search_street_strat` uses. `board` = 4 cards (flop+turn);
/// `tree` = a depth-limited TURN tree (truncates at the river deal); `reach` =
/// [2][nh] ranges. Returns (hand_cards, strategy at every turn player node).
#[cfg(feature = "metal")]
pub fn gpu_hu_turn_strat(
    ctx: &MetalContext,
    board: &[u8],
    tree: &FlatTree,
    reach: &[Vec<f32>],
    nb: usize,
    river_integrated: bool,
    cfg: &GpuSearchCfg,
) -> (Vec<u8>, HashMap<usize, Vec<Vec<f32>>>) {
    use crate::abstraction::postflop_buckets::{compute_river_integrated_wtl, compute_wtl_for_runout};
    use crate::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
    use crate::hand::eval::Hand;
    use crate::solver::bucketed_showdown::BucketedRunoutTables;

    assert_eq!(tree.num_players, 2, "gpu_hu_turn_strat is HU only");
    let bmask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));

    // Valid hands + per-hand turn strength (best 5 of the 6 = hole+board).
    let mut hand_cards: Vec<u8> = Vec::new();
    let mut hands: Vec<(Card, Card)> = Vec::new();
    let mut strengths: Vec<i32> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if bmask & (1u64 << c1) != 0 || bmask & (1u64 << c2) != 0 { continue; }
        let mut hand = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in board { hand = hand.add_card(bc as usize); }
        strengths.push(hand.evaluate_full() as i32);
        hands.push((c1, c2));
        hand_cards.push(c1); hand_cards.push(c2);
    }
    let nh = hands.len();

    // nb strength-quantile bucket map.
    let mut order: Vec<usize> = (0..nh).collect();
    order.sort_by_key(|&i| strengths[i]);
    let mut map = vec![0u16; nh];
    for (rank, &i) in order.iter().enumerate() {
        map[i] = ((rank * nb) / nh).min(nb - 1) as u16;
    }
    // Continuation tables (win/tie/lose per bucket pair). `river_integrated` uses
    // the exact check-to-showdown model (avg over every river card) so the turn
    // strategy matches the exact solve; otherwise the cheaper single-strength turn
    // proxy (over-values immediate showdown ⇒ over-bets the tail).
    let weights = vec![1.0f64; nh];
    let wtl = if river_integrated {
        compute_river_integrated_wtl(&hands, board, &map, nb)
    } else {
        compute_wtl_for_runout(&hands, &strengths, &weights, &map, nb)
    };
    let mut sums = vec![0.0f64; nb];
    for h in 0..nh { sums[map[h] as usize] += 1.0; }
    let tables = BucketedRunoutTables::from_wtl(&wtl, &sums);

    // Sorted-strength arrays (needed by MetalVectorCfr::new; HU fold terminals use
    // reach inclusion-exclusion, not strengths, so these only need to be valid).
    let mut sps = vec![0u16; nh];
    let mut spi = vec![0u16; nh];
    for (si, &i) in order.iter().enumerate() {
        sps[si] = strengths[i] as u16;
        spi[si] = i as u16;
    }
    let (sos, soi) = (sps.clone(), spi.clone());

    // num_combinations = compatible (hero,opp) hand-pair mass (matches ChanceTable).
    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let m0 = (1u64 << hand_cards[h0 * 2]) | (1u64 << hand_cards[h0 * 2 + 1]);
        for h1 in 0..nh {
            let m1 = (1u64 << hand_cards[h1 * 2]) | (1u64 << hand_cards[h1 * 2 + 1]);
            if m0 & m1 == 0 { nc += (reach[0][h0] as f64) * (reach[1][h1] as f64); }
        }
    }

    let mut gpu = MetalVectorCfr::new(ctx, tree, nh, reach, &sos, &soi, &sps, &spi, &hand_cards, nc);
    if cfg.lambda > 0.0 { gpu.set_lambda(cfg.lambda); }

    let leaf_nodes: Vec<u32> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_chance() && tree.node_children(n).is_empty())
        .map(|n| n as u32).collect();
    if !leaf_nodes.is_empty() {
        gpu.set_continuation(ctx, &leaf_nodes, &map, nb,
            &tables.f_w, &tables.f_t, &tables.f_l, &tables.f_n,
            tree.rake_rate as f32, tree.rake_cap as f32, cfg.sample_m, cfg.seed);
    }

    let ran = gpu.run_batched_budget(ctx, tree, cfg.iters, cfg.budget_ms);
    if ran < cfg.iters {
        eprintln!("[gpu-search] budget hit: ran {ran}/{} iters in {}ms budget", cfg.iters, cfg.budget_ms);
    }

    let turn = BoardState::Turn as u8;
    let mut out = HashMap::new();
    for n in 0..tree.num_nodes() {
        if tree.nodes[n].is_player() && tree.nodes[n].board_state == turn {
            let na = tree.nodes[n].num_children as usize;
            out.insert(n, gpu.get_average_strategy(n, na, nh));
        }
    }
    (hand_cards, out)
}
