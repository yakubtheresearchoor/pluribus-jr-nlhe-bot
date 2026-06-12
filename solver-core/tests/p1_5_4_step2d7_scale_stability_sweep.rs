// Step 2.D.7 (REFRAMED): incremental scale-stability sweep for the unified
// preflop+postflop composition. Measure-until-confident, NOT run-to-
// completion.
//
// PER USER FRAMING (banked 2026-06):
//
// (a) Compounding-exposing iter counts, not production counts. Composition
//     bugs surface early — the missing-CFV-seeding was visible at iter 0,
//     the postflop ULP compounder was visible by iter 10. So ~10 preflop
//     iters × ~10 postflop iters is enough to expose compounding.
//
// (b) Watch memory at each step. The killed run died swap-bound at 14.6 GB.
//     If memory growth per canonical projects past the 128 GB ceiling
//     before reaching 1755, that's a finding — surface it, don't push
//     through into swap.
//
// (c) Stopping rule: scale canonicals 10 → 50 → 100 → 250 → 1755. At each
//     step measure CPU↔GPU divergence + peak RAM + wall-clock. If
//     divergence stays 0 AND memory stays bounded, the composition is
//     scale-stable and we STOP at whatever step establishes the trend.
//     If any step shows divergence, localize with hand-derivation against
//     rules-oracle (NOT CPU↔GPU comparison — they're engineered-identical).
//
// REPORT: the divergence trend and memory trend, not a completed full-
// scale run.

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

const POSTFLOP_ITERS: u32 = 10;
const PREFLOP_ITERS: u32 = 10;

// Canonical-count steps for the sweep. We stop at whatever step establishes
// the trend (0 divergence + bounded memory).
const STEPS: &[usize] = &[10, 50, 100, 250, 1755];

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
    // BUGFIX 2026-06-07 (#103): see step2d6 — naive "first 8 by index" gives
    // mutually blocking subset → zero showdown CFV. Use used_cards bitmask
    // to pick mutually compatible hands.
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
    let hand_mask = used_cards;
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

/// Fixed extraction shape (post #105) — mirrors compute_v_flop_at_root_converged
/// in src/solver/preflop_start_game.rs (per-zone reach + chance-prob bubble-up
/// with frozen averaged strategies). See 2.D.6 for the same helper.
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
            solver.bottom_up_zone(flop_tree, table_ref, traverser, &river_reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params);
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

        solver.bottom_up_zone(flop_tree, table_ref, traverser, &turn_reach, &mut turn_cfv,
            Zone::Turn, Some(ti), None, &params);

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

    solver.bottom_up_zone(flop_tree, table_ref, traverser, &flop_reach, &mut cfv,
        Zone::Flop, None, None, &params);

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
    // POST #105 FIX: fixed extraction shape (frozen averaged σ + per-zone reach).
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
    let cpu_solver_init = FlopStartVectorCfr::new(flop_tree, game.table());
    let mut gpu_solver = MetalFlopStartSolver::new(ctx, flop_tree, &game, &cpu_solver_init);
    gpu_solver.run(ctx, flop_tree, &game, POSTFLOP_ITERS);
    // BUGFIX 2026-06-07 (#104): same as 2.D.6 — download_cfv returns the
    // GPU's last-postflop-iter CFV (traverser=iter%np perspective), NOT
    // the preflop traverser's extraction. With zero showdown (the buggy
    // pick_subset path) this was hidden; with real showdown the perspectives
    // differ. Mirror CPU side's explicit bottom_up_zone extraction.
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
    // POST #105 FIX: strategy_flop is overwritten by freeze inside the helper.
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

/// Read current process RSS (resident set size) in bytes via mach task_info.
/// Returns 0 on failure (best-effort; the sweep gates on the trend so a
/// missing reading at one step is recoverable).
fn current_rss_bytes() -> u64 {
    use std::mem::MaybeUninit;
    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut std::ffi::c_void,
            task_info_count: *mut u32,
        ) -> i32;
    }
    const MACH_TASK_BASIC_INFO: u32 = 20;
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [u32; 2],
        system_time: [u32; 2],
        policy: u32,
        suspend_count: u32,
    }
    let mut info = MaybeUninit::<MachTaskBasicInfo>::uninit();
    let mut count: u32 = (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
    unsafe {
        let r = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            info.as_mut_ptr() as *mut std::ffi::c_void,
            &mut count as *mut u32,
        );
        if r != 0 { return 0; }
        info.assume_init().resident_size
    }
}

fn run_one_preflop_iter(
    tree: &FlatTree,
    table: &PreflopChanceTable,
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
            |term_idx, traverser, r| terminal_value_fn(term_idx, traverser, r),
            &mut cfv, &params,
        );
    }
}

#[test]
#[ignore = "Step 2.D.7 sweep: measure-until-confident scale-stability of unified composition"]
fn step2d7_scale_stability_sweep() {
    let tree = build_minimal_hu_preflop_tree();
    let flop_tree = build_tiny_flop_tree();
    let np = tree.num_players as usize;
    eprintln!("\n=== Step 2.D.7: scale-stability sweep ===");
    eprintln!("Preflop tree: {} nodes; flop tree: {} nodes", tree.num_nodes(), flop_tree.num_nodes());
    eprintln!("Per-canonical postflop: subset (8 hands × 2 turn × 2 river), {} postflop iters.",
        POSTFLOP_ITERS);
    eprintln!("Per scale step: {} preflop iters", PREFLOP_ITERS);
    eprintln!("Compounding window: 10 preflop × 10 postflop is enough to expose ULP-compounding.");

    let mut class_weights: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();
    for k in 0..NUM_PREFLOP_CLASSES {
        let s = k as f32 / NUM_PREFLOP_CLASSES as f32;
        class_weights[0][k] = ((s - 0.3).max(0.05) * 1.5).min(1.0);
        class_weights[1][k] = 0.6 + 0.4 * s;
    }
    eprintln!("Building PreflopChanceTable (1755 canonical orbits)...");
    let t0 = Instant::now();
    let table = PreflopChanceTable::new(np as u8, class_weights);
    eprintln!("  Built in {:.1}s. Initial RSS = {:.2} GB",
        t0.elapsed().as_secs_f64(),
        current_rss_bytes() as f64 / 1e9);

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
    let n_classes = NUM_PREFLOP_CLASSES;

    // The sweep.
    eprintln!("\n{:>6} {:>10} {:>10} {:>10} {:>16}", "K", "wall (s)", "peak RAM", "ΔRAM/canon", "max diff");
    let mut prior_rss = current_rss_bytes();
    let mut prior_k = 0usize;
    let memory_ceiling_bytes: u64 = 100 * 1024 * 1024 * 1024; // 100 GB watch ceiling
    for &k in STEPS {
        let canonical_subset: Vec<usize> = (0..k).collect();

        let mut prod_cpu = PreflopVectorCfr::new(&tree);
        let mut prod_cpu_gpu_side = PreflopVectorCfr::new(&tree);
        let chance_node_indices = prod_cpu.preflop_chance_node_indices(&tree);

        let t_step = Instant::now();
        let mut peak_rss = current_rss_bytes();
        let mut step_max_diff = 0.0f32;
        let mut step_max_bit_diff = 0usize;

        for iter in 0..PREFLOP_ITERS {
            run_one_preflop_iter(
                &tree, &table, &chance_node_indices, &canonical_subset,
                np, n_classes, iter, &terminal_value_fn,
                &mut prod_cpu,
                |canonical, combo_ranges, traverser| {
                    cpu_per_canonical_v_combo(&flop_tree, canonical, combo_ranges, traverser)
                },
            );
            run_one_preflop_iter(
                &tree, &table, &chance_node_indices, &canonical_subset,
                np, n_classes, iter, &terminal_value_fn,
                &mut prod_cpu_gpu_side,
                |canonical, combo_ranges, traverser| {
                    gpu_per_canonical_v_combo(&ctx, &flop_tree, canonical, combo_ranges, traverser)
                },
            );
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
            let step_diff_count = rd + sd + cd;
            let iter_max = rm.max(sm).max(cm);
            if iter_max > step_max_diff { step_max_diff = iter_max; }
            if step_diff_count > step_max_bit_diff { step_max_bit_diff = step_diff_count; }

            let cur_rss = current_rss_bytes();
            if cur_rss > peak_rss { peak_rss = cur_rss; }
            if cur_rss > memory_ceiling_bytes {
                eprintln!("\nABORT — RSS {:.2} GB exceeds {:.0} GB ceiling at K={}, iter={}.",
                    cur_rss as f64 / 1e9, memory_ceiling_bytes as f64 / 1e9, k, iter);
                eprintln!("Finding: per-canonical FlopChanceTable allocation is not bounded.");
                eprintln!("Surface this as the scale failure mode; do not push into swap.");
                panic!("memory ceiling exceeded — refusing to swap");
            }
        }
        let step_secs = t_step.elapsed().as_secs_f64();
        let delta_rss = (peak_rss as i64) - (prior_rss as i64);
        let delta_per_canon = if k > prior_k && delta_rss > 0 {
            (delta_rss / (k - prior_k) as i64) as f64
        } else { 0.0 };
        eprintln!("{:>6} {:>10.1} {:>9.2}G {:>9.2}M {:>16} max_abs={:.3e}",
            k, step_secs, peak_rss as f64 / 1e9,
            delta_per_canon / 1e6,
            step_max_bit_diff, step_max_diff);

        // Gate.
        assert_eq!(step_max_bit_diff, 0,
            "STEP 2.D.7 DIVERGENCE at K={}: {} bit-different entries (max_abs {:.3e}). \
             Localize with hand-derivation against rules-oracle, NOT CPU↔GPU comparison.",
            k, step_max_bit_diff, step_max_diff);

        // Memory projection: if per-canonical growth × remaining canonicals > ceiling,
        // surface the finding even if current step is OK.
        let remaining = 1755usize.saturating_sub(k);
        let projected = peak_rss as f64 + delta_per_canon * remaining as f64;
        if projected > memory_ceiling_bytes as f64 {
            eprintln!("  PROJECTION: scaling per-canonical growth to K=1755 would reach {:.1} GB > ceiling {:.0} GB.",
                projected / 1e9, memory_ceiling_bytes as f64 / 1e9);
            eprintln!("  FINDING (banked): the unified composition is bit-stable at K={} but the per-",
                k);
            eprintln!("  canonical allocation pattern does not scale to K=1755 at current implementation.");
            eprintln!("  Refusing to push past projected ceiling; this is the planned 'surface, not swap' outcome.");
            eprintln!("  Per-canonical hoisting/streaming is the optimization the projection makes feasibility-mandatory.");
            break;
        }
        prior_rss = peak_rss;
        prior_k = k;
    }

    eprintln!("\n=== STEP 2.D.7 SWEEP REPORT ===");
    eprintln!("Divergence trend: bit-exact (0) at every K step measured.");
    eprintln!("Composition is scale-stable across the measured range; full-1755 either reached or projected to require memory optimization.");
    eprintln!("Per the framing: stop-when-stable. The trend establishes the composition holds.");
}
