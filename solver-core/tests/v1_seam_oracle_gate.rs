//! V1 SEAM-ORACLE GATE (2026-06-12, four-zone runtime slice 1): the
//! load-bearing join — the preflop layer pulling postflop values from
//! the BUCKET'S OWN seam-cell game at the right keys.
//!
//! Gates (HU: every flop-entry cell is all-live, so slice 1's scope is
//! complete here):
//!   1. ROUTING EXACTNESS: the engine's cell-aware chance-CFV through
//!      BucketKeyedOracle ≡ a DIRECT composition that hand-picks the
//!      cell's game — bit-exact (same float path).
//!   2. THE V1 DISTINGUISHING FACT: the limp-pot cell and the 3-bet
//!      cell produce DIFFERENT chance values (the bootstrap's
//!      single-game seam could not represent this — its values were
//!      cell-blind). Hand-checkable direction: deeper pots ⇒ larger
//!      value magnitudes at the same classes.
//!   3. FROZEN-CACHE contract: within an epoch the oracle computes
//!      once per (key, flop, traverser); cache size = keys × flops ×
//!      traversers touched.
//!
//! NAMED RESIDUAL (rides to the head-to-head): the frozen-values
//! approximation, confirmed at the PREFLOP layer specifically — the
//! one layer the bot plays unrefined (search starts postflop).

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::solver::postflop_oracle::{BucketKeyedOracle, PostflopValueOracle, SeamCell};
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::solver::preflop_start_game::{
    compute_v_flop_at_root_iter0, flop_combo_layout, PreflopChanceTable,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::{build_tree, build_tree_preflop_only};
use solver_core::tree::flat::{FlatTree, NODE_TYPE_CHANCE};
use std::collections::HashMap;

const STACK: i32 = 200;

fn hu_preflop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 0,
        starting_stacks: vec![STACK - 2, STACK - 1],
        initial_contributions: vec![2, 1],
        rake_rate: 0.0,
        rake_cap: 0.0,
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
    build_tree_preflop_only(&cfg).expect("HU preflop-only tree")
}

/// The cell's seam game tree (HU: live = 2). Cached by cell.
fn cell_tree(cell: SeamCell, cache: &mut HashMap<(u8, i32, i32), std::sync::Arc<FlatTree>>) -> std::sync::Arc<FlatTree> {
    cache
        .entry((cell.live, cell.commit, cell.pot))
        .or_insert_with(|| {
            let spec_bets =
                BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
            // GameSpec-equivalent seam config at HU gate scale: behind =
            // STACK - commit, dead pot = cell.pot (mirrors
            // GameSpec::flop_seam_config with stack = STACK).
            let cfg = TreeConfig {
                num_players: cell.live,
                initial_state: BoardState::Flop,
                starting_pot: cell.pot,
                starting_stacks: vec![STACK - cell.commit; cell.live as usize],
                initial_contributions: vec![0; cell.live as usize],
                rake_rate: 0.0,
                rake_cap: 0.0,
                bet_sizes: spec_bets,
                add_allin_threshold: 1.0,
                force_allin_threshold: 1.0,
                merging_threshold: 0.0,
                button_player: None,
                max_bets_per_street: None,
            };
            std::sync::Arc::new(build_tree(&cfg).expect("cell tree"))
        })
        .clone()
}

/// The slice-2 value source: iter0 root CFV on the CELL'S LIVE-SUBSET
/// game. `folded_mask` selects the live seats — the live-player game is
/// solved with only the live seats' ranges; folded opponents enter only
/// as dead money (in cell.pot) and card-removal (named residual: not
/// modeled by the live-subset game). At HU all-live (mask 0) this is the
/// identity over seats — the slice-1 behavior.
fn iter0_source(
) -> impl FnMut(SeamCell, u16, [Card; 3], &[Vec<f32>], u8) -> Vec<f32> {
    let mut trees: HashMap<(u8, i32, i32), std::sync::Arc<FlatTree>> = HashMap::new();
    move |cell, folded_mask, canonical, combo_reaches, traverser| {
        let tree = cell_tree(cell, &mut trees);
        let layout = flop_combo_layout(canonical);
        let np = combo_reaches.len();
        // Live seats in seat order; the cell tree has `cell.live` players
        // in this order. Traverser (guaranteed live) maps to its index.
        let live_seats: Vec<usize> =
            (0..np).filter(|&p| (folded_mask >> p) & 1 == 0).collect();
        assert_eq!(live_seats.len(), cell.live as usize, "live-seat count vs cell.live");
        let trav_live = live_seats
            .iter()
            .position(|&p| p == traverser as usize)
            .expect("traverser must be live in the slice-2 source");
        let mut full: Vec<Vec<f32>> =
            vec![vec![0.0f32; solver_core::card::NUM_POSSIBLE_HANDS]; live_seats.len()];
        for (lp, &seat) in live_seats.iter().enumerate() {
            for (li, &(c1, c2)) in layout.iter().enumerate() {
                full[lp][solver_core::card::card_pair_to_index(c1, c2)] =
                    combo_reaches[seat][li];
            }
        }
        let (v_table, layout_table) =
            compute_v_flop_at_root_iter0(canonical, &tree, &full, trav_live as u8);
        // Re-order table layout → engine layout.
        let mut pos: HashMap<(Card, Card), usize> = HashMap::new();
        for (i, &(a, b)) in layout_table.iter().enumerate() {
            pos.insert((a, b), i);
        }
        layout
            .iter()
            .map(|&(a, b)| v_table[pos[&(a, b)]])
            .collect()
    }
}

#[test]
#[ignore = "seam oracle gate (~minutes); --ignored --nocapture --release"]
fn v1_seam_oracle_gate() {
    let tree = hu_preflop_tree();
    let np = 2u8;
    let table =
        PreflopChanceTable::new(np, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; np as usize]);
    let subset: Vec<usize> = (0..table.num_canonical_flops()).step_by(70).collect(); // ~25 canonicals

    // Pick two structurally different cells: the LIMP pot and the
    // deepest raised pot among flop entries.
    let chance_nodes: Vec<usize> = (0..tree.nodes.len())
        .filter(|&i| {
            tree.nodes[i].node_type == NODE_TYPE_CHANCE && tree.nodes[i].num_children == 0
        })
        .collect();
    let cells: Vec<(usize, SeamCell)> = chance_nodes
        .iter()
        .map(|&i| (i, SeamCell::at_chance_node(&tree, i, np as usize)))
        .collect();
    let (limp_idx, limp_cell) =
        *cells.iter().min_by_key(|(_, c)| c.pot).expect("limp cell");
    let (raise_idx, raise_cell) =
        *cells.iter().max_by_key(|(_, c)| c.pot).expect("raised cell");
    eprintln!("limp cell {limp_cell:?} @node {limp_idx} | raised cell {raise_cell:?} @node {raise_idx}");
    assert_ne!(limp_cell.pot, raise_cell.pot, "fixture must span pots");

    let solver = {
        let mut s = PreflopVectorCfr::new(&tree);
        s.compute_preflop_strategy(&tree);
        s
    };
    let reach = solver.compute_preflop_reach(&tree, None);

    // ── Gate 1+3: engine-through-oracle ≡ direct, per cell; cache. ──
    let mut oracle = BucketKeyedOracle::new(STACK, np, 0, iter0_source());
    let mut direct_src = iter0_source();
    for (idx, cell) in [(limp_idx, limp_cell), (raise_idx, raise_cell)] {
        let fmask = tree.get_folded_mask(idx);
        for t in 0..np {
            let via_engine = solver.compute_chance_node_cfv_with_expansion_subset_for_cell(
                idx, t, &reach, &table, &subset, &mut oracle, cell, fmask,
            );
            // Direct composition: same expand → src on the cell's game
            // → reduce → aggregate, hand-built.
            let direct = {
                use solver_core::solver::preflop_start_game::{
                    aggregate_preflop_chance_subset, expand_reach_class_to_combo,
                    reduce_cfv_combo_to_class,
                };
                let n_classes = NUM_PREFLOP_CLASSES;
                let base = idx * n_classes;
                let mut per_canon = Vec::new();
                for &ci in &subset {
                    let f = table.canonical_flops[ci];
                    let layout = flop_combo_layout(f);
                    let mut cr = Vec::new();
                    for p in 0..np as usize {
                        cr.push(expand_reach_class_to_combo(
                            f,
                            &reach[p][base..base + n_classes],
                            &layout,
                        ));
                    }
                    let v = direct_src(cell, fmask, f, &cr, t);
                    per_canon.push(reduce_cfv_combo_to_class(f, &v, &layout));
                }
                aggregate_preflop_chance_subset(&table, &subset, &per_canon)
            };
            assert_eq!(via_engine.len(), direct.len());
            for (i, (a, b)) in via_engine.iter().zip(&direct).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "cell {cell:?} t{t} class {i}: engine {a} vs direct {b} — \
                     routing must be exact"
                );
            }
        }
    }
    let expected_cache = 2 * subset.len() * np as usize;
    assert_eq!(
        oracle.cache_len(),
        expected_cache,
        "frozen contract: one entry per (key, flop, traverser)"
    );
    eprintln!("gate 1+3: routing bit-exact; cache {} = 2 cells × {} flops × {np} travs", oracle.cache_len(), subset.len());

    // ── Gate 2: cell-distinct values (the v1 fix). ──
    let v_limp = solver.compute_chance_node_cfv_with_expansion_subset_for_cell(
        limp_idx, 0, &reach, &table, &subset, &mut oracle, limp_cell,
        tree.get_folded_mask(limp_idx),
    );
    let v_raise = solver.compute_chance_node_cfv_with_expansion_subset_for_cell(
        raise_idx, 0, &reach, &table, &subset, &mut oracle, raise_cell,
        tree.get_folded_mask(raise_idx),
    );
    let mut diff = 0usize;
    for (a, b) in v_limp.iter().zip(&v_raise) {
        if a.to_bits() != b.to_bits() {
            diff += 1;
        }
    }
    eprintln!("gate 2a: {diff}/{} classes differ across cells", v_limp.len());
    assert!(diff > NUM_PREFLOP_CLASSES / 2, "cells must produce distinct value surfaces");

    // 2b STRUCTURAL hand-check (theorem-grade): the cell determines
    // the GAME, not just the values. The limp cell (pot 4, deep behind)
    // has postflop betting decisions; the deepest raised cell here is
    // the all-in pot (commit 200 = STACK), which by definition has NO
    // postflop player nodes — straight to showdown. Build both cell
    // trees and count player nodes.
    //
    // (This also explains why an absolute-magnitude hand-check is
    // ill-posed at this value source: iter0 + uniform ranges + HU
    // symmetry makes the traverser's root CFV ≈ 0 by symmetry at BOTH
    // cells, and the all-in cell has no play to optimize at all. The
    // distinguishing fact is structural — gate 2a's 169/169 bit-level
    // divergence plus this node-count split — not a magnitude.)
    let mut trees: HashMap<(u8, i32, i32), std::sync::Arc<FlatTree>> = HashMap::new();
    let limp_tree = cell_tree(limp_cell, &mut trees);
    let raise_tree = cell_tree(raise_cell, &mut trees);
    // GENUINE decisions = player nodes with > 1 child. The all-in cell
    // still has player nodes, but they are all FORCED CHECK (one child,
    // zero chips behind — the `player_remaining <= 0` path); the limp
    // cell has real betting decisions.
    let count_decisions = |t: &FlatTree| {
        (0..t.nodes.len())
            .filter(|&i| t.nodes[i].is_player() && t.nodes[i].num_children > 1)
            .count()
    };
    let limp_dec = count_decisions(&limp_tree);
    let raise_dec = count_decisions(&raise_tree);
    eprintln!(
        "gate 2b structural: limp cell (pot {}) has {} genuine postflop decisions; \
         all-in raised cell (commit {} = stack) has {} (forced-check only)",
        limp_cell.pot, limp_dec, raise_cell.commit, raise_dec
    );
    assert!(limp_dec > 0, "the playable cell must have real betting decisions");
    assert_eq!(
        raise_cell.commit, STACK,
        "the deepest HU flop-entry cell is the all-in pot"
    );
    assert_eq!(
        raise_dec, 0,
        "the all-in cell has no GENUINE decisions (only forced checks) — the seam \
         routes to a structurally different game, not just different numbers"
    );
    eprintln!("v1 seam-oracle gate PASSED");
}

// ════════════════════════════════════════════════════════════════════
// SLICE 2 (2026-06-12): folded players at the seam. A 3-handed preflop
// tree HAS flop-entry cells where one player folded preflop (live=2 of
// 3) — the case slice 1 asserted against. Two paths, both gated:
//   A. FOLDED TRAVERSER: the engine routes to the fold machinery
//      (terminal_value_fn), NOT the oracle. Gate: router output ≡
//      terminal_value_fn at that node, bit-exact; and the oracle's
//      folded-traverser guard FIRES if mis-called.
//   B. LIVE TRAVERSER, FOLDED OPPONENT: the oracle solves the 2-player
//      LIVE-SUBSET game. Gate: router-through-oracle ≡ a direct
//      2-player composition on the live seats, bit-exact.

use solver_core::solver::preflop_cfr::make_bootstrap_terminal_value_fn_multiway_pairwise;

fn three_handed_preflop_tree() -> FlatTree {
    // 3-max, blinds [SB 1, BB 2, button 0 contributes 0]; one bet + one
    // raise so a player can fold preflop and the other two reach a flop.
    let cfg = TreeConfig {
        num_players: 3,
        initial_state: BoardState::Preflop,
        starting_pot: 0,
        starting_stacks: vec![STACK - 1, STACK - 2, STACK],
        initial_contributions: vec![1, 2, 0],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(2),
        max_bets_per_street: None,
    };
    build_tree_preflop_only(&cfg).expect("3-handed preflop-only tree")
}

#[test]
#[ignore = "slice-2 seam oracle gate (~minutes); --ignored --nocapture --release"]
fn v1_seam_oracle_gate_slice2_folded() {
    let tree = three_handed_preflop_tree();
    let np = 3u8;
    let table =
        PreflopChanceTable::new(np, vec![vec![1.0f32; NUM_PREFLOP_CLASSES]; np as usize]);
    let subset: Vec<usize> = (0..table.num_canonical_flops()).step_by(120).collect(); // ~15
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);

    // Find a live-2 flop-entry chance node (one preflop folder).
    let mut found: Option<(usize, u8, Vec<usize>)> = None;
    for idx in 0..tree.nodes.len() {
        let n = &tree.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.num_children != 0 {
            continue;
        }
        let mask = tree.get_folded_mask(idx);
        let live: Vec<usize> = (0..np as usize).filter(|&p| (mask >> p) & 1 == 0).collect();
        if live.len() == 2 {
            let folded = (0..np as usize).find(|&p| (mask >> p) & 1 == 1).unwrap() as u8;
            found = Some((idx, folded, live));
            break;
        }
    }
    let (idx, folded_t, live_seats) = found.expect("a live-2 flop entry must exist at 3-handed");
    let cell = SeamCell::at_chance_node(&tree, idx, np as usize);
    let mask = tree.get_folded_mask(idx);
    eprintln!(
        "slice-2 node {idx}: cell {cell:?}, folded seat {folded_t}, live seats {live_seats:?}"
    );
    assert_eq!(cell.live, 2);

    let solver = {
        let mut s = PreflopVectorCfr::new(&tree);
        s.compute_preflop_strategy(&tree);
        s
    };
    let reach = solver.compute_preflop_reach(&tree, None);
    let nc = NUM_PREFLOP_CLASSES;
    let base = idx * nc;
    let reach_at: Vec<Vec<f32>> =
        (0..np as usize).map(|p| reach[p][base..base + nc].to_vec()).collect();

    // ── A. FOLDED TRAVERSER routes to the fold machinery, bit-exact. ──
    // (full-canonical subset doesn't matter: folded value is postflop-
    // independent; the router ignores the oracle entirely.)
    let mut guard_oracle = BucketKeyedOracle::new(STACK, np, 0, iter0_source());
    let routed =
        solver.chance_cfv_for_traverser(&tree, idx, folded_t, &reach, &table, &mut guard_oracle, &term_fn);
    let direct_fold = term_fn(idx, folded_t, &reach_at);
    assert_eq!(routed.len(), direct_fold.len());
    for (i, (a, b)) in routed.iter().zip(&direct_fold).enumerate() {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "folded traverser class {i}: router {a} vs fold-machinery {b} — must be exact"
        );
    }
    // The folded value is a CONSTANT chip-delta × blocking: every class
    // shares the sign (folding loses, so ≤ 0). Hand-check.
    assert!(routed.iter().all(|&v| v <= 1e-6), "folded traverser value must be ≤ 0 (loses contribution)");
    eprintln!("slice-2 A: folded traverser routed to fold machinery, bit-exact, all ≤ 0");

    // ── B. LIVE TRAVERSER, folded opponent: oracle solves the 2-player
    // live-subset game; router-through-oracle ≡ direct composition. ──
    let live_t = live_seats[0] as u8;
    let mut oracle = BucketKeyedOracle::new(STACK, np, 0, iter0_source());
    let via_engine = solver.compute_chance_node_cfv_with_expansion_subset_for_cell(
        idx, live_t, &reach, &table, &subset, &mut oracle, cell, mask,
    );
    let mut direct_src = iter0_source();
    let direct = {
        use solver_core::solver::preflop_start_game::{
            aggregate_preflop_chance_subset, expand_reach_class_to_combo,
            reduce_cfv_combo_to_class,
        };
        let mut per_canon = Vec::new();
        for &ci in &subset {
            let f = table.canonical_flops[ci];
            let layout = flop_combo_layout(f);
            let cr: Vec<Vec<f32>> = (0..np as usize)
                .map(|p| expand_reach_class_to_combo(f, &reach[p][base..base + nc], &layout))
                .collect();
            let v = direct_src(cell, mask, f, &cr, live_t);
            per_canon.push(reduce_cfv_combo_to_class(f, &v, &layout));
        }
        aggregate_preflop_chance_subset(&table, &subset, &per_canon)
    };
    assert_eq!(via_engine.len(), direct.len());
    for (i, (a, b)) in via_engine.iter().zip(&direct).enumerate() {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "live traverser (folded opp) class {i}: engine {a} vs direct 2-player {b}"
        );
    }
    eprintln!("slice-2 B: live traverser w/ folded opponent — 2-player live-subset routing bit-exact");

    // ── Negative check: the oracle's folded-traverser guard FIRES. ──
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut o = BucketKeyedOracle::new(STACK, np, 0, iter0_source());
        let c = SeamCell { live: 2, commit: cell.commit, pot: cell.pot };
        o.flop_root_cfv_for_cell(
            table.canonical_flops[subset[0]],
            &vec![vec![1.0f32; flop_combo_layout(table.canonical_flops[subset[0]]).len()]; np as usize],
            folded_t,
            c,
            mask,
        )
    }));
    assert!(caught.is_err(), "oracle must reject a folded traverser");
    eprintln!("slice-2 negative: oracle folded-traverser guard fires ✓");
    eprintln!("v1 slice-2 seam-oracle gate PASSED");
}
