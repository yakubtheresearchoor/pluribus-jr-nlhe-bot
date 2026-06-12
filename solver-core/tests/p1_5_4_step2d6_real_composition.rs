// Step 2.D.6 (LOAD-BEARING): unified preflop+postflop REAL composition
// CPU↔GPU multi-iter replication gate.
//
// THE CORRECTION (banked 2026-06): 2.D.5 validated the preflop loop
// mechanics with a STUB leaf. The unified port's defining feature —
// preflop and postflop composed in one loop — is precisely the piece the
// stub replaced. Composition-by-transitivity (postflop validated alone +
// preflop validated alone → composition validated) is the decomposition
// argument that the missing-CFV-seeding bug REFUTED. So the composition
// must be measured, not asserted.
//
// THIS TEST: swap the stub for the real per-canonical postflop solver.
// CPU side uses CPU FlopStartVectorCfr per canonical. GPU side uses
// MetalFlopStartSolver per canonical (same flop game; the only thing that
// differs between sides is which solver computes the per-canonical CFV).
//
// Bit-exactness expected: postflop CPU↔GPU established by 2.A.2 production
// cell. If the unified port wires the per-canonical solver correctly,
// each canonical's v_combo bit-exact, hence aggregated v_class bit-exact,
// hence preflop regret update bit-exact, hence multi-iter state bit-exact.
//
// SCALE: small for tractability — small canonical subset, tiny postflop
// (stacks=10, 1 bet size), few preflop iters. Scaling to full production
// cell at this composition is a separate follow-up; this gate validates
// the wiring shape.

#![cfg(feature = "metal")]

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::{MetalBuffer, MetalFlopStartSolver};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr};
use solver_core::solver::postflop_oracle::{ClosureOracle, PostflopValueOracle};
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::solver::preflop_start_game::{
    aggregate_preflop_chance_subset, expand_reach_class_to_combo,
    flop_combo_layout, reduce_cfv_combo_to_class, PreflopChanceTable,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const POSTFLOP_ITERS: u32 = 1;
const PREFLOP_ITERS: u32 = 3;
const CANONICAL_SUBSET_SIZE: usize = 2;

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

/// Tiny postflop tree: HU, small stacks, single bet size. Keeps per-
/// canonical solve cheap so we can afford the full subset × multi-iter run.
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

/// Convert layout-engine combo reach to NUM_POSSIBLE_HANDS-indexed reach,
/// same as UnabstractedPostflopOracle does.
fn expand_combo_ranges_to_full(
    canonical: [Card; 3],
    combo_ranges_per_player: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let layout = flop_combo_layout(canonical);
    let np = combo_ranges_per_player.len();
    let mut full: Vec<Vec<f32>> = vec![vec![0.0f32; NUM_POSSIBLE_HANDS]; np];
    for p in 0..np {
        assert_eq!(combo_ranges_per_player[p].len(), layout.len());
        for (li, &(c1, c2)) in layout.iter().enumerate() {
            full[p][card_pair_to_index(c1, c2)] = combo_ranges_per_player[p][li];
        }
    }
    full
}

/// Pick a small subset of valid hands + a small deck subset for the
/// per-canonical postflop. Same subset is used on both CPU and GPU sides
/// per call so bit-exact comparison holds. Subset is chosen to avoid
/// blocking the canonical's cards.
fn pick_subset(canonical: [Card; 3]) -> (Vec<u16>, Vec<u8>, Vec<Vec<u8>>) {
    let board_mask: u64 = canonical.iter()
        .fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    // BUGFIX 2026-06-07 (#103): naive "first 8 non-blocking by index" picks
    // pairs all sharing the smallest non-board card → mutually blocking →
    // showdown CFV is zero by construction. Track used_cards bitmask and
    // only accept pairs that don't conflict with previously chosen hands,
    // so the subset is mutually compatible (opp can hold any g != h in
    // subset, opp_reach[g] > 0, showdown CFV is non-degenerate).
    let mut chosen: Vec<u16> = Vec::new();
    let mut used_cards: u64 = board_mask;
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = solver_core::card::index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        if used_cards & (1u64 << c1) != 0 || used_cards & (1u64 << c2) != 0 { continue; }
        chosen.push(idx as u16);
        used_cards |= 1u64 << c1;
        used_cards |= 1u64 << c2;
        if chosen.len() == 8 { break; }
    }
    // Pick 2 turn cards that don't conflict with the board or chosen hands.
    let hand_mask = used_cards;
    let mut turn_cards: Vec<u8> = Vec::new();
    for c in 0u8..52u8 {
        if hand_mask & (1u64 << c) != 0 { continue; }
        turn_cards.push(c);
        if turn_cards.len() == 2 { break; }
    }
    // Per turn card, pick 2 river cards that don't conflict.
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

/// CPU per-canonical postflop subsolve. Uses compute_flop_start_subset_with_decks
/// with a tiny subset for tractable wall-clock.
/// Fixed extraction shape (post #105) — mirrors compute_v_flop_at_root_converged
/// in src/solver/preflop_start_game.rs (per-zone reach + chance-prob bubble-up
/// with frozen averaged strategies). Used by both cpu/gpu helpers so the
/// composition test exercises the correct extraction semantics, not the
/// pre-#105 buggy shape (single flop_reach for all bottom_up calls).
fn fixed_extraction_from_solver(
    solver: &mut FlopStartVectorCfr,
    flop_tree: &FlatTree,
    game: &FlopStartGame,
    traverser: u8,
) -> Vec<f32> {
    use solver_core::solver::flop_start_vector_cfr::Zone;

    solver.freeze_average_strategy_flop(flop_tree);

    let nh = solver.num_hands();
    let nn = flop_tree.num_nodes();
    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();
    let params = DcfrParams::new(0);

    let flop_reach = solver.compute_reach_flop(flop_tree, game);
    let mut cfv = vec![0.0f32; nn * nh];
    let mut river_cfv_accum = vec![0.0f32; nn * nh];
    let mut turn_cfv = vec![0.0f32; nn * nh];
    let mut flop_cfv = vec![0.0f32; nn * nh];

    for &child_id in solver.turn_chance_children() {
        let off = child_id as usize * nh;
        for h in 0..nh { flop_cfv[off + h] = 0.0; }
    }

    for (ti, &tc_card) in turn_deck.iter().enumerate() {
        solver.freeze_average_strategy_for_turn(flop_tree, ti);
        let turn_reach = solver.compute_reach_turn(flop_tree, ti, &flop_reach);
        let river_deck = &table_ref.river_decks[tc_card as usize];

        for &child_id in solver.river_chance_children() {
            let off = child_id as usize * nh;
            for h in 0..nh { river_cfv_accum[off + h] = 0.0; }
        }

        for ri in 0..river_deck.len() {
            solver.load_river_pair(ti, ri).unwrap();
            solver.freeze_average_strategy_for_river_pair(flop_tree, ti, ri);
            let river_reach = solver.compute_reach_river(flop_tree, ti, ri, &turn_reach);
            solver.bottom_up_zone(
                flop_tree, table_ref, traverser, &river_reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params,
            );
            solver.save_river_pair(ti, ri).unwrap();

            for &child_id in solver.river_chance_children() {
                for h in 0..nh {
                    let cp = table_ref.chance_probability_river(tc_card, ri, h);
                    river_cfv_accum[child_id as usize * nh + h] +=
                        cp * cfv[child_id as usize * nh + h];
                }
            }
        }

        for &child_id in solver.river_chance_children() {
            for h in 0..nh {
                turn_cfv[child_id as usize * nh + h] =
                    river_cfv_accum[child_id as usize * nh + h];
            }
        }

        solver.bottom_up_zone(
            flop_tree, table_ref, traverser, &turn_reach, &mut turn_cfv,
            Zone::Turn, Some(ti), None, &params,
        );

        for &child_id in solver.turn_chance_children() {
            for h in 0..nh {
                let cp = table_ref.chance_probability_turn(ti, h);
                flop_cfv[child_id as usize * nh + h] +=
                    cp * turn_cfv[child_id as usize * nh + h];
            }
        }
    }

    for &child_id in solver.turn_chance_children() {
        for h in 0..nh {
            cfv[child_id as usize * nh + h] = flop_cfv[child_id as usize * nh + h];
        }
    }

    solver.bottom_up_zone(
        flop_tree, table_ref, traverser, &flop_reach, &mut cfv,
        Zone::Flop, None, None, &params,
    );

    cfv[0..nh].to_vec()
}

fn cpu_per_canonical_v_combo(
    flop_tree: &FlatTree,
    canonical: [Card; 3],
    combo_ranges_per_player: &[Vec<f32>],
    traverser: u8,
) -> Vec<f32> {
    let np = combo_ranges_per_player.len() as u8;
    let full = expand_combo_ranges_to_full(canonical, combo_ranges_per_player);
    let board: Vec<Card> = canonical.iter().copied().collect();
    let (chosen, turn_cards, river_decks) = pick_subset(canonical);
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &full, np, &chosen, &turn_cards, &river_decks,
    );
    let nh = table.num_valid;
    let layout_table: Vec<(Card, Card)> = (0..nh)
        .map(|i| (table.hand_cards[i * 2], table.hand_cards[i * 2 + 1]))
        .collect();
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let _ = solver.run(flop_tree, &game, POSTFLOP_ITERS);
    // POST #105 FIX: use the fixed extraction shape (frozen averaged σ +
    // per-zone reach + chance-prob bubble-up). The pre-#105 single-flop-reach
    // pattern was discovered to silently zero out the showdown contribution.
    let v_table = fixed_extraction_from_solver(&mut solver, flop_tree, &game, traverser);
    let layout_engine = flop_combo_layout(canonical);
    let mut v_engine = vec![0.0f32; layout_engine.len()];
    for (li, &combo) in layout_engine.iter().enumerate() {
        if let Some(pos) = layout_table.iter().position(|&c| c == combo) {
            v_engine[li] = v_table[pos];
        }
    }
    v_engine
}

/// GPU equivalent: same shape but uses MetalFlopStartSolver for the per-
/// iter pipeline. After run(), do the same bottom_up extraction passes on
/// CPU using FlopStartVectorCfr fed with the GPU's regrets+cum_strategy
/// state — this matches CPU production semantics while ensuring the
/// PER-ITER solver state was computed on GPU.
///
/// (A fully GPU-side bottom_up extraction would need new MetalFlopStartSolver
/// API surface; that's mechanical follow-up. For THIS gate the key claim
/// is that the GPU postflop solver's state, threaded through the unified
/// preflop loop, produces bit-exact results.)
fn gpu_per_canonical_v_combo(
    ctx: &MetalContext,
    flop_tree: &FlatTree,
    canonical: [Card; 3],
    combo_ranges_per_player: &[Vec<f32>],
    traverser: u8,
) -> Vec<f32> {
    let np = combo_ranges_per_player.len() as u8;
    let full = expand_combo_ranges_to_full(canonical, combo_ranges_per_player);
    let board: Vec<Card> = canonical.iter().copied().collect();
    let (chosen, turn_cards, river_decks) = pick_subset(canonical);
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &full, np, &chosen, &turn_cards, &river_decks,
    );
    let nh = table.num_valid;
    let layout_table: Vec<(Card, Card)> = (0..nh)
        .map(|i| (table.hand_cards[i * 2], table.hand_cards[i * 2 + 1]))
        .collect();
    let game = FlopStartGame::new(table);
    let cpu_solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let mut gpu_solver = MetalFlopStartSolver::new(ctx, flop_tree, &game, &cpu_solver);
    gpu_solver.run(ctx, flop_tree, &game, POSTFLOP_ITERS);
    // BUGFIX 2026-06-07 (#104): the old code returned gpu.download_cfv()[..nh]
    // directly. That cfv reflects the postflop solver's LAST-ITER perspective
    // (player iter % np), NOT the preflop traverser passed in here. When
    // showdown was zero (pick_subset bug), both perspectives returned zero
    // so the bug was hidden. With real showdown, the perspectives differ
    // and CPU↔GPU diverge.
    //
    // Fix: download GPU's regrets+cum_strategy, install in a CPU
    // FlopStartVectorCfr, then run the SAME bottom_up_zone extraction pass
    // as the CPU side. This honors the comment block above (which said
    // "do the same bottom_up extraction passes on CPU using FlopStartVectorCfr
    // fed with the GPU's regrets+cum_strategy state") that the code never
    // actually implemented.
    let gpu_regrets = gpu_solver.download_regrets();
    let gpu_cum_strategy = gpu_solver.download_cum_strategy();
    let mut solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let fl = solver.regrets_flop().len();
    let tl = solver.regrets_turn().len();
    let rl = solver.regrets_river().len();
    assert_eq!(gpu_regrets.len(), fl + tl + rl, "regrets shape mismatch");
    assert_eq!(gpu_cum_strategy.len(), fl + tl + rl, "cum_strategy shape mismatch");
    solver.regrets_flop_mut().copy_from_slice(&gpu_regrets[..fl]);
    solver.regrets_turn_mut().copy_from_slice(&gpu_regrets[fl..fl + tl]);
    solver.regrets_river_mut().copy_from_slice(&gpu_regrets[fl + tl..]);
    solver.cum_strategy_flop_mut().copy_from_slice(&gpu_cum_strategy[..fl]);
    solver.cum_strategy_turn_mut().copy_from_slice(&gpu_cum_strategy[fl..fl + tl]);
    solver.cum_strategy_river_mut().copy_from_slice(&gpu_cum_strategy[fl + tl..]);
    solver.set_iteration(POSTFLOP_ITERS);
    // POST #105 FIX: strategy_flop will be OVERWRITTEN by freeze inside the
    // fixed extraction helper (freeze recomputes σ_avg from cum_strategy).
    // No need to install GPU's last-iter strategy snapshot — the FROZEN
    // strategy is what production uses for averaged-σ CFV extraction.
    let v_table = fixed_extraction_from_solver(&mut solver, flop_tree, &game, traverser);
    let layout_engine = flop_combo_layout(canonical);
    let mut v_engine = vec![0.0f32; layout_engine.len()];
    for (li, &combo) in layout_engine.iter().enumerate() {
        if let Some(pos) = layout_table.iter().position(|&c| c == combo) {
            v_engine[li] = v_table[pos];
        }
    }
    v_engine
}

#[test]
#[ignore = "Step 2.D.6: real preflop+postflop composition CPU↔GPU multi-iter gate"]
fn step2d6_real_composition_cpu_gpu_multi_iter() {
    let tree = build_minimal_hu_preflop_tree();
    let flop_tree = build_tiny_flop_tree();
    let np = tree.num_players as usize;
    eprintln!("\n=== Step 2.D.6: REAL preflop+postflop composition gate ===");
    eprintln!("Preflop tree: {} nodes", tree.num_nodes());
    eprintln!("Flop tree:    {} nodes", flop_tree.num_nodes());

    // Asymmetric class weights.
    let mut class_weights: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();
    for k in 0..NUM_PREFLOP_CLASSES {
        let s = k as f32 / NUM_PREFLOP_CLASSES as f32;
        class_weights[0][k] = ((s - 0.3).max(0.05) * 1.5).min(1.0);
        class_weights[1][k] = 0.6 + 0.4 * s;
    }
    eprintln!("Building PreflopChanceTable...");
    let table = PreflopChanceTable::new(np as u8, class_weights);

    // Canonical subset: first CANONICAL_SUBSET_SIZE canonical flops.
    let canonical_subset: Vec<usize> = (0..CANONICAL_SUBSET_SIZE).collect();
    eprintln!("Canonical subset size: {} of {}",
        canonical_subset.len(), table.num_canonical_flops());

    let mut prod_cpu = PreflopVectorCfr::new(&tree);
    let mut prod_cpu_gpu_side = PreflopVectorCfr::new(&tree);
    let chance_node_indices = prod_cpu.preflop_chance_node_indices(&tree);

    // Terminal value fn: simple deterministic asymmetric.
    let terminal_value_fn = |term_idx: usize, traverser: u8, _r: &[Vec<f32>]| -> Vec<f32> {
        (0..NUM_PREFLOP_CLASSES).map(|c| {
            let seed: u64 = (term_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (traverser as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                ^ (c as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            let bits = ((seed >> 32) & 0xFFFFFF) as i64 - (1 << 23);
            (bits as f32) / ((1 << 23) as f32)
        }).collect()
    };

    let ctx = MetalContext::new().expect("Metal");

    // Custom oracle for both sides: uses {cpu,gpu}_per_canonical_v_combo
    // for the SUBSET only; off-subset canonicals contribute 0.
    // We run the SAME compute_chance_node_cfv_with_expansion_subset logic
    // manually here, calling the appropriate per-canonical solver.

    // For each preflop iter, do a full run_one_iteration analog manually
    // for both CPU and GPU sides, so the only differing piece is the
    // per-canonical postflop solver.

    let n_classes = NUM_PREFLOP_CLASSES;
    for iter in 0..PREFLOP_ITERS {
        eprintln!("\n--- preflop iter {} ---", iter);
        // CPU side.
        run_one_preflop_iter(
            &tree, &table, &flop_tree, &chance_node_indices, &canonical_subset,
            np, n_classes, iter, &terminal_value_fn,
            &mut prod_cpu,
            |canonical, combo_ranges, traverser| {
                cpu_per_canonical_v_combo(&flop_tree, canonical, combo_ranges, traverser)
            },
        );
        // GPU side.
        run_one_preflop_iter(
            &tree, &table, &flop_tree, &chance_node_indices, &canonical_subset,
            np, n_classes, iter, &terminal_value_fn,
            &mut prod_cpu_gpu_side,
            |canonical, combo_ranges, traverser| {
                gpu_per_canonical_v_combo(&ctx, &flop_tree, canonical, combo_ranges, traverser)
            },
        );

        // Compare state.
        let max_abs = |a: &[f32], b: &[f32]| -> (usize, f32) {
            let mut bc = 0usize;
            let mut m = 0.0f32;
            for i in 0..a.len().min(b.len()) {
                if a[i].to_bits() != b[i].to_bits() {
                    bc += 1;
                    let d = (a[i] - b[i]).abs();
                    if d > m { m = d; }
                }
            }
            (bc, m)
        };
        let (rd, rm) = max_abs(&prod_cpu.regrets, &prod_cpu_gpu_side.regrets);
        let (sd, sm) = max_abs(&prod_cpu.strategy, &prod_cpu_gpu_side.strategy);
        let (cd, cm) = max_abs(&prod_cpu.cum_strategy, &prod_cpu_gpu_side.cum_strategy);
        eprintln!("  regrets:      {} bit-diff, max_abs {:.3e}", rd, rm);
        eprintln!("  strategy:     {} bit-diff, max_abs {:.3e}", sd, sm);
        eprintln!("  cum_strategy: {} bit-diff, max_abs {:.3e}", cd, cm);
        assert_eq!(rd, 0, "STEP 2.D.6: regrets diverge at preflop iter {}: max_abs {:.3e}", iter, rm);
        assert_eq!(sd, 0, "STEP 2.D.6: strategy diverges at preflop iter {}", iter);
        assert_eq!(cd, 0, "STEP 2.D.6: cum_strategy diverges at preflop iter {}", iter);
    }

    eprintln!("\n=== STEP 2.D.6 PASS — REAL COMPOSITION GATE GREEN ===");
    eprintln!("{} preflop iters × {} canonicals (subset) × {} postflop iters per call.",
        PREFLOP_ITERS, canonical_subset.len(), POSTFLOP_ITERS);
    eprintln!("CPU preflop + CPU postflop  ==  CPU preflop + GPU postflop  bit-exactly.");
    eprintln!("The unified port composition holds end-to-end.");
}

/// Run one preflop iter manually using the supplied per-canonical solver
/// closure. Mirrors PreflopVectorCfr::run_one_iteration but with explicit
/// hooks for the per-canonical step.
fn run_one_preflop_iter(
    tree: &FlatTree,
    table: &PreflopChanceTable,
    _flop_tree: &FlatTree,
    chance_node_indices: &[usize],
    canonical_subset: &[usize],
    np: usize,
    n_classes: usize,
    iter: u32,
    terminal_value_fn: &dyn Fn(usize, u8, &[Vec<f32>]) -> Vec<f32>,
    solver: &mut PreflopVectorCfr,
    per_canonical: impl Fn([Card; 3], &[Vec<f32>], u8) -> Vec<f32>,
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
                let v_combo = per_canonical(f_canon, &combo_reaches, t);
                let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
                per_canon_v.push(v_class);
            }
            cfv[chance_idx] = aggregate_preflop_chance_subset(table, canonical_subset, &per_canon_v);
        }
        solver.bottom_up_preflop_for_traverser(
            tree, t, chance_node_indices, &reach,
            |term_idx, traverser, _r| terminal_value_fn(term_idx, traverser, _r),
            &mut cfv, &params,
        );
    }
}
