// Step 2.D.7b: disambiguate the 440 MB-per-canonical finding from #94.
// CHURN (allocator high-water-mark; hoisting fixes it) vs RESIDENT
// (per-canonical state retained; hoisting won't help).
//
// THE INCONSISTENCY (banked per user 2026-06): #94 diagnosed the 440 MB as
// allocation churn (FlopChanceTable rebuilt every per-canonical call) but
// projected it as if it's resident (linear scaling to 772 GB at K=1755).
// Those can't both be true: if each call's table is dropped after use,
// peak RSS is roughly one-table-at-a-time, NOT K-tables-simultaneously.
//
// DISAMBIGUATION:
//   (a) Take RSS snapshots before/after each canonical solve within ONE
//       preflop iter. If RSS oscillates (rises during call, drops after),
//       it's churn. If RSS rises monotonically across canonical index,
//       per-canonical data is being retained.
//   (b) Run K=10 step TWICE in sequence; if peak doesn't grow on the
//       second pass, the high-water-mark is bounded by allocator
//       behavior, not by accumulated residency.

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
const K: usize = 10;

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

fn current_rss_bytes() -> u64 {
    use std::mem::MaybeUninit;
    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(t: u32, f: u32, o: *mut std::ffi::c_void, c: *mut u32) -> i32;
    }
    const MACH_TASK_BASIC_INFO: u32 = 20;
    #[repr(C)]
    struct MTBI {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [u32; 2],
        system_time: [u32; 2],
        policy: u32,
        suspend_count: u32,
    }
    let mut info = MaybeUninit::<MTBI>::uninit();
    let mut count: u32 = (std::mem::size_of::<MTBI>() / std::mem::size_of::<u32>()) as u32;
    unsafe {
        let r = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
            info.as_mut_ptr() as *mut std::ffi::c_void, &mut count as *mut u32);
        if r != 0 { 0 } else { info.assume_init().resident_size }
    }
}

/// Returns (peak_rss_resident_max, current_rss_resident).
/// resident_size_max is the kernel's tracked peak — accumulates over
/// process lifetime so consecutive K=10 runs share its high-water-mark.
fn current_rss_with_peak() -> (u64, u64) {
    use std::mem::MaybeUninit;
    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(t: u32, f: u32, o: *mut std::ffi::c_void, c: *mut u32) -> i32;
    }
    const MACH_TASK_BASIC_INFO: u32 = 20;
    #[repr(C)]
    struct MTBI {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [u32; 2],
        system_time: [u32; 2],
        policy: u32,
        suspend_count: u32,
    }
    let mut info = MaybeUninit::<MTBI>::uninit();
    let mut count: u32 = (std::mem::size_of::<MTBI>() / std::mem::size_of::<u32>()) as u32;
    unsafe {
        let r = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
            info.as_mut_ptr() as *mut std::ffi::c_void, &mut count as *mut u32);
        if r != 0 { (0, 0) } else {
            let i = info.assume_init();
            (i.resident_size_max, i.resident_size)
        }
    }
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
        &board, &full, np, &chosen, &turn_cards, &river_decks);
    let nh = table.num_valid;
    let layout_table: Vec<(Card, Card)> = (0..nh)
        .map(|i| (table.hand_cards[i * 2], table.hand_cards[i * 2 + 1])).collect();
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let _ = solver.run(flop_tree, &game, POSTFLOP_ITERS);
    let nn = flop_tree.num_nodes();
    let mut cfv = vec![0.0f32; nn * nh];
    let reach = solver.compute_reach_flop(flop_tree, &game);
    let params = DcfrParams::new(POSTFLOP_ITERS);
    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();
    use solver_core::solver::flop_start_vector_cfr::Zone;
    for (ti, &tc) in turn_deck.iter().enumerate() {
        let river_deck = &table_ref.river_decks[tc as usize];
        for ri in 0..river_deck.len() {
            solver.load_river_pair(ti, ri).unwrap();
            solver.bottom_up_zone(flop_tree, table_ref, traverser, &reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params);
            solver.save_river_pair(ti, ri).unwrap();
        }
    }
    for ti in 0..turn_deck.len() {
        solver.bottom_up_zone(flop_tree, table_ref, traverser, &reach, &mut cfv,
            Zone::Turn, Some(ti), None, &params);
    }
    solver.bottom_up_zone(flop_tree, table_ref, traverser, &reach, &mut cfv,
        Zone::Flop, None, None, &params);
    let v_table = cfv[0..nh].to_vec();
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

#[test]
#[ignore = "Step 2.D.7b: disambiguate churn-vs-resident for the 440 MB-per-canonical finding"]
fn step2d7b_memory_disambiguation() {
    let tree = build_minimal_hu_preflop_tree();
    let flop_tree = build_tiny_flop_tree();
    let np = tree.num_players as usize;
    eprintln!("\n=== Step 2.D.7b: memory disambiguation — churn vs resident ===");

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

    let ctx = MetalContext::new().expect("Metal");

    // ─────────────────────────────────────────────────────────────────────
    // Diagnostic A: RSS snapshots BEFORE/AFTER each per-canonical solve
    // within ONE preflop iter at K=10.
    // ─────────────────────────────────────────────────────────────────────
    eprintln!("\n--- Diagnostic A: RSS oscillation within one preflop iter (K=10) ---");
    eprintln!("If RSS oscillates (rises during call, drops after), it's CHURN.");
    eprintln!("If RSS rises monotonically with canonical index, it's RESIDENT.\n");

    let canonical_subset: Vec<usize> = (0..K).collect();
    let mut prod = PreflopVectorCfr::new(&tree);
    let chance_node_indices = prod.preflop_chance_node_indices(&tree);
    let n_classes = NUM_PREFLOP_CLASSES;

    prod.compute_preflop_strategy(&tree);
    let reach = prod.compute_preflop_reach(&tree, None);

    let (rss_peak_before_a, rss_cur_before_a) = current_rss_with_peak();
    eprintln!("Pre-iter:  resident_size = {:.3} GB, resident_size_max = {:.3} GB",
        rss_cur_before_a as f64 / 1e9, rss_peak_before_a as f64 / 1e9);

    let nn = tree.num_nodes();
    let traverser = 0u8;
    let mut cfv: Vec<Vec<f32>> = vec![vec![0.0f32; n_classes]; nn];
    eprintln!("\n{:>4} {:>14} {:>14} {:>14} {:>14}",
        "K_i", "before (MB)", "after (MB)", "Δ (MB)", "peak_max (MB)");
    for &chance_idx in &chance_node_indices[..1] {
        let chance_base = chance_idx * n_classes;
        for (ki, &canonical_idx) in canonical_subset.iter().enumerate() {
            let (_peak_before, cur_before) = current_rss_with_peak();
            let f_canon = table.canonical_flops[canonical_idx];
            let layout = flop_combo_layout(f_canon);
            let mut combo_reaches: Vec<Vec<f32>> = Vec::with_capacity(np);
            for p in 0..np {
                let class_reach = &reach[p][chance_base..chance_base + n_classes];
                combo_reaches.push(expand_reach_class_to_combo(f_canon, class_reach, &layout));
            }
            let v_combo = cpu_per_canonical_v_combo(&flop_tree, f_canon, &combo_reaches, traverser);
            let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
            let _ = v_class; // result not needed for this diagnostic
            let (peak_after, cur_after) = current_rss_with_peak();
            let delta_mb = (cur_after as i64 - cur_before as i64) as f64 / 1e6;
            eprintln!("{:>4} {:>14.1} {:>14.1} {:>+14.1} {:>14.1}",
                ki,
                cur_before as f64 / 1e6, cur_after as f64 / 1e6, delta_mb,
                peak_after as f64 / 1e6);
        }
    }
    let (rss_peak_after_a, rss_cur_after_a) = current_rss_with_peak();
    eprintln!("\nPost-iter: resident_size = {:.3} GB, resident_size_max = {:.3} GB",
        rss_cur_after_a as f64 / 1e9, rss_peak_after_a as f64 / 1e9);
    let net_growth = rss_cur_after_a as i64 - rss_cur_before_a as i64;
    eprintln!("Net resident growth over one preflop sub-iter at K=10: {:+.3} GB",
        net_growth as f64 / 1e9);

    // ─────────────────────────────────────────────────────────────────────
    // Diagnostic B: run the full K=10 step TWICE. If peak is bounded,
    // the second pass's resident_size_max won't grow; if accumulating,
    // it will.
    // ─────────────────────────────────────────────────────────────────────
    eprintln!("\n--- Diagnostic B: run K=10 step twice, peak should plateau ---");
    eprintln!("If peak_max plateaus, the allocator high-water-mark is bounded by per-iter peak,");
    eprintln!("NOT by per-canonical retention.\n");

    let run_one_step = |table: &PreflopChanceTable,
                       chance_node_indices: &[usize],
                       canonical_subset: &[usize]| -> (u64, u64, f64, usize) {
        let mut prod = PreflopVectorCfr::new(&tree);
        let t0 = Instant::now();
        let mut peak_rss = current_rss_bytes();
        for iter in 0..2u32 {
            prod.compute_preflop_strategy(&tree);
            let reach = prod.compute_preflop_reach(&tree, None);
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
                        let v_combo = cpu_per_canonical_v_combo(&flop_tree, f_canon, &combo_reaches, t);
                        let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
                        per_canon_v.push(v_class);

                        let cur = current_rss_bytes();
                        if cur > peak_rss { peak_rss = cur; }
                    }
                    cfv[chance_idx] = aggregate_preflop_chance_subset(table, canonical_subset, &per_canon_v);
                }
                prod.bottom_up_preflop_for_traverser(
                    &tree, t, chance_node_indices, &reach,
                    |term_idx, tr, r| terminal_value_fn(term_idx, tr, r),
                    &mut cfv, &params,
                );
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        let (final_peak_max, final_cur) = current_rss_with_peak();
        (final_peak_max, peak_rss, secs, final_cur as usize)
    };

    let canonical_subset: Vec<usize> = (0..K).collect();
    let (peak_max_1, peak_sampled_1, secs_1, cur_1) = run_one_step(&table, &chance_node_indices, &canonical_subset);
    eprintln!("Run 1: peak_max={:.2} GB, peak_sampled={:.2} GB, current={:.2} GB, {:.1}s",
        peak_max_1 as f64 / 1e9, peak_sampled_1 as f64 / 1e9, cur_1 as f64 / 1e9, secs_1);
    let (peak_max_2, peak_sampled_2, secs_2, cur_2) = run_one_step(&table, &chance_node_indices, &canonical_subset);
    eprintln!("Run 2: peak_max={:.2} GB, peak_sampled={:.2} GB, current={:.2} GB, {:.1}s",
        peak_max_2 as f64 / 1e9, peak_sampled_2 as f64 / 1e9, cur_2 as f64 / 1e9, secs_2);

    let peak_max_delta = peak_max_2 as i64 - peak_max_1 as i64;
    let cur_delta = cur_2 as i64 - cur_1 as i64;
    eprintln!("\nΔ peak_max:  {:+.3} GB", peak_max_delta as f64 / 1e9);
    eprintln!("Δ current:   {:+.3} GB", cur_delta as f64 / 1e9);

    // ─────────────────────────────────────────────────────────────────────
    // Verdict.
    // ─────────────────────────────────────────────────────────────────────
    eprintln!("\n=== VERDICT ===");
    if cur_delta.abs() < 100 * 1024 * 1024 {
        eprintln!("Resident size BOUNDED across consecutive K=10 runs (Δ < 100 MB).");
        eprintln!("→ The 440 MB-per-canonical from #94 is CHURN (allocator high-water-mark),");
        eprintln!("  not RESIDENT. Hoisting the FlopChanceTable allocation outside the");
        eprintln!("  per-canonical loop should bring per-canonical cost toward zero.");
        eprintln!("  The 772 GB projection from #94 is ILLUSORY and should NOT be used as");
        eprintln!("  the abstraction's bucket-budget framing.");
    } else {
        eprintln!("Resident size GROWS across consecutive K=10 runs (Δ = {:+.3} GB).",
            cur_delta as f64 / 1e9);
        eprintln!("→ The 440 MB-per-canonical from #94 is RESIDENT.");
        eprintln!("  Hoisting won't help; 772 GB is the real wall.");
        eprintln!("  The cost framing for the abstraction is correct as banked.");
    }
}
