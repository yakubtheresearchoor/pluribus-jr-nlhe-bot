// Step 2.D.8: production-iter wall-clock baseline via K∈{10,20,40} linear
// scaling, then project to K=1755.
//
// PER USER FRAMING (banked 2026-06):
//
// The cost is dominated by the per-canonical postflop solve, which is the
// same work per canonical, so it's close to linear in canonical count.
// Measure a few small points, confirm linearity, project. DO NOT run 1755.
//
// (a) Three K points (10, 20, 40) at PRODUCTION iter counts (~10 preflop ×
//     ~50 postflop), not the sweep's compounding-exposing counts. Two
//     points minimum separates fixed overhead from per-canonical cost via
//     slope = (cost(K2) − cost(K1)) / (K2 − K1). The third point confirms
//     linearity rather than assuming it.
//
// (b) Check the slope between K=10→20 and K=20→40. If flat, linearity
//     holds and projection to K=1755 is trustworthy. If creeping up,
//     project with the trend and flag the super-linear term.
//
// (c) Report the projection as a RATIO to compute budget — the absolute
//     number alone doesn't size the abstraction. If budget unknown,
//     surface absolute + flag.
//
// (d) GPU side only (MetalFlopStartSolver). The GPU is the production
//     artifact — CPU is the bit-exact reference, not the production
//     wall-clock baseline. Bit-exactness (already established by 2.D.6 +
//     2.D.7) means GPU and CPU produce the same OUTPUT, not the same
//     throughput; measuring CPU here would give the wrong baseline number
//     entirely.

#![cfg(feature = "metal")]

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr};
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::solver::preflop_start_game::{
    aggregate_preflop_chance_subset, expand_reach_class_to_combo,
    flop_combo_layout, reduce_cfv_combo_to_class, PreflopChanceTable,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const POSTFLOP_ITERS: u32 = 50;
const PREFLOP_ITERS: u32 = 10;

/// K points for the linear-scaling sweep. Three is the minimum to
/// separate intercept, slope, and slope drift.
const K_VALUES: &[usize] = &[10, 20, 40];

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
    let mut chosen: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = solver_core::card::index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        chosen.push(idx as u16);
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

fn expand_combo_ranges_to_full(
    canonical: [Card; 3],
    combo_ranges_per_player: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let layout = flop_combo_layout(canonical);
    let np = combo_ranges_per_player.len();
    let mut full: Vec<Vec<f32>> = vec![vec![0.0f32; NUM_POSSIBLE_HANDS]; np];
    for p in 0..np {
        for (li, &(c1, c2)) in layout.iter().enumerate() {
            full[p][card_pair_to_index(c1, c2)] = combo_ranges_per_player[p][li];
        }
    }
    full
}

/// Production per-canonical postflop subsolve on GPU (MetalFlopStartSolver).
/// This is what wall-clock measurement at production iter counts is
/// measuring — the actual production artifact.
fn gpu_per_canonical_v_combo(
    ctx: &MetalContext,
    flop_tree: &FlatTree,
    canonical: [Card; 3],
    combo_ranges_per_player: &[Vec<f32>],
    _traverser: u8,
) -> Vec<f32> {
    let np = combo_ranges_per_player.len() as u8;
    let full = expand_combo_ranges_to_full(canonical, combo_ranges_per_player);
    let board: Vec<Card> = canonical.iter().copied().collect();
    let (chosen, turn_cards, river_decks) = pick_subset(canonical);
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &full, np, &chosen, &turn_cards, &river_decks);
    let nh = table.num_valid;
    let layout_table: Vec<(Card, Card)> = (0..nh)
        .map(|i| (table.hand_cards[i * 2], table.hand_cards[i * 2 + 1])).collect();
    let game = FlopStartGame::new(table);
    let cpu_solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let mut gpu_solver = MetalFlopStartSolver::new(ctx, flop_tree, &game, &cpu_solver);
    gpu_solver.run(ctx, flop_tree, &game, POSTFLOP_ITERS);
    let gpu_cfv = gpu_solver.download_cfv();
    let v_table = gpu_cfv[0..nh].to_vec();
    let layout_engine = flop_combo_layout(canonical);
    let mut v_engine = vec![0.0f32; layout_engine.len()];
    for (li, &combo) in layout_engine.iter().enumerate() {
        if let Some(pos) = layout_table.iter().position(|&c| c == combo) {
            v_engine[li] = v_table[pos];
        }
    }
    v_engine
}

fn run_one_preflop_iter(
    ctx: &MetalContext,
    tree: &FlatTree,
    table: &PreflopChanceTable,
    flop_tree: &FlatTree,
    chance_node_indices: &[usize],
    canonical_subset: &[usize],
    np: usize,
    n_classes: usize,
    iter: u32,
    terminal_value_fn: &dyn Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
    solver: &mut PreflopVectorCfr,
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
                let v_combo = gpu_per_canonical_v_combo(ctx, flop_tree, f_canon, &combo_reaches, t);
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

#[test]
#[ignore = "Step 2.D.8: production-iter wall-clock baseline at K∈{10,20,40} for projection to K=1755"]
fn step2d8_production_iter_baseline() {
    let tree = build_minimal_hu_preflop_tree();
    let flop_tree = build_tiny_flop_tree();
    let np = tree.num_players as usize;
    eprintln!("\n=== Step 2.D.8: production-iter wall-clock baseline ===");
    eprintln!("Preflop tree: {} nodes, flop tree: {} nodes", tree.num_nodes(), flop_tree.num_nodes());
    eprintln!("Production iter counts: {} preflop × {} postflop", PREFLOP_ITERS, POSTFLOP_ITERS);
    eprintln!("Side: GPU (MetalFlopStartSolver) — the production artifact.");
    eprintln!("Operates on the isomorphism-reduced 1755-class space (#95 verified).");
    let ctx = MetalContext::new().expect("Metal");

    let mut class_weights: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();
    for k in 0..NUM_PREFLOP_CLASSES {
        let s = k as f32 / NUM_PREFLOP_CLASSES as f32;
        class_weights[0][k] = ((s - 0.3).max(0.05) * 1.5).min(1.0);
        class_weights[1][k] = 0.6 + 0.4 * s;
    }
    eprintln!("Building PreflopChanceTable...");
    let table = PreflopChanceTable::new(np as u8, class_weights);

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

    // ── Measurement table. ──
    let mut measurements: Vec<(usize, f64)> = Vec::new();
    eprintln!("\n{:>6} {:>10} {:>14} {:>14}",
        "K", "wall (s)", "per-K (s)", "per-canon (s)");

    for &k in K_VALUES {
        let canonical_subset: Vec<usize> = (0..k).collect();
        let mut prod = PreflopVectorCfr::new(&tree);
        let chance_node_indices = prod.preflop_chance_node_indices(&tree);

        let t0 = Instant::now();
        for iter in 0..PREFLOP_ITERS {
            run_one_preflop_iter(
                &ctx, &tree, &table, &flop_tree, &chance_node_indices,
                &canonical_subset, np, n_classes, iter,
                &terminal_value_fn, &mut prod,
            );
        }
        let secs = t0.elapsed().as_secs_f64();
        measurements.push((k, secs));
        eprintln!("{:>6} {:>10.1} {:>14.2} {:>14.3}",
            k, secs, secs, secs / k as f64);
    }

    // ── Linear-scaling analysis. ──
    eprintln!("\n=== Linear-scaling analysis ===");
    // Slope between consecutive K's.
    // wall(K) = intercept + slope × K
    // slope(K1→K2) = (wall(K2) − wall(K1)) / (K2 − K1)
    let (k1, w1) = measurements[0];
    let (k2, w2) = measurements[1];
    let (k3, w3) = measurements[2];
    let slope_12 = (w2 - w1) / (k2 - k1) as f64;
    let slope_23 = (w3 - w2) / (k3 - k2) as f64;
    let intercept_12 = w1 - slope_12 * k1 as f64;
    let intercept_23 = w2 - slope_23 * k2 as f64;
    eprintln!("Slope K={}→K={}: {:.3} s/canonical (intercept inferred {:.2} s)",
        k1, k2, slope_12, intercept_12);
    eprintln!("Slope K={}→K={}: {:.3} s/canonical (intercept inferred {:.2} s)",
        k2, k3, slope_23, intercept_23);
    let slope_drift = (slope_23 - slope_12).abs();
    let drift_ratio = if slope_12 > 0.0 { slope_drift / slope_12 } else { 0.0 };
    eprintln!("Slope drift K=10→20 vs K=20→40: {:+.3} s/canonical ({:.1}% of slope_12)",
        slope_23 - slope_12, drift_ratio * 100.0);

    let linearity_holds = drift_ratio < 0.15;
    if linearity_holds {
        eprintln!("→ LINEAR SCALING CONFIRMED (slope drift {:.1}% < 15% tolerance).",
            drift_ratio * 100.0);
    } else {
        eprintln!("→ SUPER-LINEAR SCALING DETECTED (slope drift {:.1}% > 15% tolerance).",
            drift_ratio * 100.0);
        eprintln!("  Project with the trend, not the constant slope.");
    }

    // ── Projection to K=1755. ──
    // Use the K=20→40 slope (the most representative) and the
    // intercept inferred from it.
    let slope = slope_23;
    let intercept = intercept_23;
    let projected_secs = intercept + slope * 1755.0;
    let projected_hours = projected_secs / 3600.0;
    let projected_days = projected_hours / 24.0;
    eprintln!("\n=== Projection to K=1755 ===");
    eprintln!("  intercept (fixed overhead): {:.2} s", intercept);
    eprintln!("  per-canonical slope:        {:.3} s/canonical", slope);
    eprintln!("  projected wall-clock at K=1755: {:.0} s = {:.2} h = {:.3} days",
        projected_secs, projected_hours, projected_days);

    // ── Cost ratio framing. ──
    eprintln!("\n=== Cost framing for #96 abstraction design ===");
    eprintln!("The naive unabstracted production blueprint at K=1755 with {} preflop × {} postflop iters",
        PREFLOP_ITERS, POSTFLOP_ITERS);
    eprintln!("(after the verified isomorphism reduction, on this Mac Studio):");
    eprintln!("  ~ {:.2} hours wall-clock (projected from K≤{} linear scaling).", projected_hours, K_VALUES[2]);
    eprintln!();
    eprintln!("the lead's compute budget for a blueprint run is NOT yet specified in the");
    eprintln!("task context. Surfacing absolute number; the budget ratio that sizes the");
    eprintln!("abstraction's bucketing aggressiveness needs the lead's budget number as input.");
    eprintln!();
    eprintln!("Ratio framings the budget needs to resolve:");
    eprintln!("  If budget =  1 hour:   ratio = {:.1}x → aggressive bucketing required.",
        projected_hours / 1.0);
    eprintln!("  If budget =  4 hours:  ratio = {:.1}x → moderate bucketing required.",
        projected_hours / 4.0);
    eprintln!("  If budget = 24 hours:  ratio = {:.2}x → gentle bucketing may suffice.",
        projected_hours / 24.0);

    // ── Honesty banner. ──
    eprintln!("\n=== HONESTY ===");
    eprintln!("This is a PROJECTION from K∈{{{},{},{}}} measurements, NOT a measured",
        K_VALUES[0], K_VALUES[1], K_VALUES[2]);
    eprintln!("production-scale run. If #96 ends up at a feasibility boundary where this");
    eprintln!("projection's error matters, measure a larger K to tighten. Until then the");
    eprintln!("small-K projection is sufficient input to size the abstraction.");
    eprintln!();
    eprintln!("Also note: the SMALL postflop config (stacks=10, 1 bet size, subset 8 hands");
    eprintln!("× 2 turn × 2 river per canonical) is a per-canonical cost LOWER BOUND. Production");
    eprintln!("postflop config (larger stacks, OptB bet sizes, full nh=1176 × full deck) is");
    eprintln!("orders of magnitude more expensive per canonical. The projection above is the");
    eprintln!("\"with-subset\" floor; the real production blueprint cost is this × the");
    eprintln!("postflop-config cost factor. Budget framing should account for that.");
}
