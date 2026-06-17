//! JOINT COLD-CFR DRIVER (2026-06-16) — the real factory test, not a toy.
//!
//! One cold run: the production preflop CFR loop (`run_one_iteration`) driven by
//! a LIVE postflop value source that solves each flop-entry cell on the
//! production GPU bucketed solver (`BucketedNativeGpu`) against the CURRENT
//! preflop reach, every iteration (`refresh_every=1` ⇒ the oracle cache clears
//! each iter, so values are range-dependent / not frozen). No banked read, no
//! warm-start (each subgame solved cold), no coarsening.
//!
//! This is the FULL-SCALE workload: na=16 preflop (≈28.6 GB cap-3 buffer) +
//! every reached flop-entry cell × canonical flop solved on the GPU. It is RUN
//! UNDER the RSS memory guard (`mem_guard`) so an overshoot aborts cleanly
//! instead of OOM-killing the box — the point is to MEASURE whether it fits, at
//! the real scale, not to extrapolate from a baby.
//!
//! STAGE A (this file): the GPU postflop subgames are solved COLD per visit and
//! NOT persisted across preflop iters — so this measures the real peak RSS
//! (na=16 preflop + the live GPU working set) and the real per-iter throughput.
//! Range-injection into the GPU solve (true coherence) and cross-iter subgame
//! persistence are Stage B; neither changes the GPU buffer footprint, so the
//! memory verdict here is representative.
//!
//! Env: JOINT_PF_CAP (preflop iters to run, default 3 — enough to measure peak
//! RSS + s/iter), JOINT_POST_ITERS (postflop DCFR iters per subgame, default
//! 34), JOINT_B (bucket count, default 8). MEM_GUARD_SOFT_GB / MEM_GUARD_HARD_GB
//! configure the guard (defaults 26 / 31 on a 36 GB box).

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("joint_solve requires --features metal");
    std::process::exit(2);
}

#[cfg(feature = "metal")]
fn main() {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
    use solver_core::card::Card;
    use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
    use solver_core::gpu_metal::context::MetalContext;
    use solver_core::mem_guard::MemGuard;
    use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
    use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
    use solver_core::solver::postflop_oracle::{BucketKeyedOracle, SeamCell};
    use solver_core::solver::preflop_cfr::{
        make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr,
    };
    use solver_core::solver::preflop_start_game::{
        compute_v_flop_root_gpu_per_traverser, flop_combo_layout, table_hand_to_layout_perm,
        PreflopChanceTable,
    };
    use solver_core::tree::action::{
        production_game_v1, BetCap, BetSize, BetSizeOptions, GameSpec,
    };
    use solver_core::tree::builder::{build_tree, build_tree_preflop_only};
    use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

    // ---- knobs -------------------------------------------------------------
    let pf_cap: u32 = std::env::var("JOINT_PF_CAP").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let post_iters: u32 =
        std::env::var("JOINT_POST_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(34);
    let nb: usize = std::env::var("JOINT_B").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    // count-only: skip the GPU solve, return zeros — measures the per-iteration
    // unique (key,canonical) solve COUNT (the work scale) without the wall.
    let count_only = std::env::var("JOINT_COUNT_ONLY").is_ok();
    let spec = production_game_v1();

    // ---- guard ARMED before any big allocation -----------------------------
    let guard = MemGuard::from_env();

    // ---- the production cap-3 preflop tree at na=16 ------------------------
    let tree: FlatTree = {
        let mrc = MAX_NA_PREFLOP.saturating_sub(2);
        let mut cfg = spec.preflop_tree_config(BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect(),
        });
        cfg.max_bets_per_street = BetCap::all(3);
        build_tree_preflop_only(&cfg).expect("cap-3 preflop tree")
    };
    let table = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);

    eprintln!(
        "joint_solve: na={MAX_NA_PREFLOP} preflop tree {} nodes, B={nb}, {post_iters} post-iters/subgame, {pf_cap} preflop iters",
        tree.num_nodes()
    );
    eprintln!("  RSS after preflop tree build: {:.2} GB", guard.peak_gb());

    // ---- LIVE GPU value source --------------------------------------------
    // Same routing as banked_read_source, but SOLVED LIVE: live-3/4/5 on the
    // GPU bucketed solver, live-2 exact (CPU), live-6 / all-in equity rollout.
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let stack = spec.stack;
    let canon = table.canonical_flops.clone();
    let mut flop_index: HashMap<[Card; 3], usize> = HashMap::new();
    for (fi, f) in canon.iter().enumerate() {
        flop_index.insert(*f, fi);
    }

    let ctx = MetalContext::new().expect("Metal context");
    let mut betting_trees: HashMap<(u8, i32, i32), Arc<FlatTree>> = HashMap::new();
    let mut rollout_trees: HashMap<(u8, i32, i32), Arc<FlatTree>> = HashMap::new();
    let gpu_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let gpu_count_src = gpu_count.clone();
    let guard_src = guard.clone(); // live progress readout
    // Per-(key,canonical) cache: solve a subgame ONCE per preflop iteration and
    // serve all traversers (the oracle's per-traverser cache would re-solve
    // np×). Cleared each iter via `epoch` — range-dependent values can't
    // persist across iters. `gpu_count` = cache misses = actual subgame solves.
    let epoch = std::rc::Rc::new(std::cell::Cell::new(0u32));
    let epoch_src = epoch.clone();
    let mut src_cache: std::collections::HashMap<((u8, i64), [Card; 3]), Vec<Vec<f32>>> =
        std::collections::HashMap::new();
    let mut last_epoch = u32::MAX;

    // seeded 1×1 runout (matches the fill convention so the live solve uses the
    // same single-runout subgame the banked path priced).
    let splitmix = |mut x: u64| -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let seeded_1x1 = move |canonical: [Card; 3], fi: usize| -> (Vec<u8>, Vec<Vec<u8>>) {
        let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
        let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
        let mut pool = deck.clone();
        let x = splitmix(fi as u64);
        let tc = pool.remove((x % pool.len() as u64) as usize);
        let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
        let mut rpool: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        let rx = splitmix((fi as u64) ^ 0xDEAD_BEEF ^ ((tc as u64) << 40));
        let rc = rpool.remove((rx % rpool.len() as u64) as usize);
        rd[tc as usize] = vec![rc];
        (vec![tc], rd)
    };

    let rollout_tree = |spec: &GameSpec, live: u8, commit: i32, pot: i32| -> FlatTree {
        let mut cfg = spec.flop_seam_config(
            live,
            commit.min(spec.stack),
            pot,
            BetSizeOptions { bet: vec![], raise: vec![] },
        );
        cfg.add_allin_threshold = 0.0;
        cfg.force_allin_threshold = 0.0;
        build_tree(&cfg).expect("rollout seam tree")
    };

    let mut source = move |cell: SeamCell,
                           folded_mask: u16,
                           canonical: [Card; 3],
                           combo_reaches: &[Vec<f32>],
                           traverser: u8|
          -> Vec<f32> {
        let np = combo_reaches.len();
        let live_seats: Vec<usize> = (0..np).filter(|&p| (folded_mask >> p) & 1 == 0).collect();
        let trav_live =
            live_seats.iter().position(|&p| p == traverser as usize).expect("traverser live");
        let key = cell.bucket_key(stack);

        // per-iteration cache: clear on epoch change, then serve all traversers
        // of this (key, canonical) from one solve.
        let ep = epoch_src.get();
        if ep != last_epoch {
            src_cache.clear();
            last_epoch = ep;
        }
        if let Some(pl) = src_cache.get(&(key, canonical)) {
            return pl[trav_live].clone();
        }
        let n_so_far = gpu_count_src.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n_so_far % 200 == 0 {
            eprintln!("    [{n_so_far} solves so far | live {:.1} GB]", guard_src.peak_gb());
        }
        if count_only {
            let pl = vec![vec![0.0f32; flop_combo_layout(canonical).len()]; live_seats.len()];
            src_cache.insert((key, canonical), pl.clone());
            return pl[trav_live].clone();
        }

        let fi = *flop_index.get(&canonical).expect("canonical flop in list");
        let (turns, river_decks) = seeded_1x1(canonical, fi);
        let is_allin = key.1 == i64::MIN;

        // per-live layout-ordered CFV (one vec per live seat)
        let per_live: Vec<Vec<f32>> = if cell.live == 2 {
            // exact HU on GPU (MetalFlopStartSolver) — the measured bottleneck,
            // now off CPU. ONE GPU solve, both traversers extracted.
            let layout = flop_combo_layout(canonical);
            let bt = betting_trees
                .entry((cell.live, cell.commit, cell.pot))
                .or_insert_with(|| {
                    Arc::new(
                        build_tree(&spec.flop_seam_config(cell.live, cell.commit, cell.pot, bets.clone()))
                            .expect("live-2 seam tree"),
                    )
                })
                .clone();
            let tbl = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
            let (per_trav, lay_tbl) =
                compute_v_flop_root_gpu_per_traverser(&ctx, tbl, &bt, 2, post_iters);
            let mut pos: HashMap<(Card, Card), usize> = HashMap::new();
            for (i, &(a, b)) in lay_tbl.iter().enumerate() {
                pos.insert((a.min(b), a.max(b)), i);
            }
            per_trav
                .iter()
                .map(|v_tbl| layout.iter().map(|&(a, b)| v_tbl[pos[&(a.min(b), a.max(b))]]).collect())
                .collect()
        } else if is_allin || cell.live == 6 {
            // equity rollout (check-only, bucketed) — faithful for the cells the
            // fill never solves
            let rt = rollout_trees
                .entry((cell.live, cell.commit, cell.pot))
                .or_insert_with(|| Arc::new(rollout_tree(&spec, cell.live, cell.commit, cell.pot)))
                .clone();
            let tbl = FlopChanceTable::build_full_nh_sampled(canonical, cell.live, &turns, &river_decks);
            let bk = FlopBucketing::quantile(&tbl, nb);
            let perm = table_hand_to_layout_perm(&tbl.hand_cards, tbl.num_valid, canonical);
            let game = FlopStartGame::new(tbl);
            let mut solver = BucketedFlopCfr::new(&rt, game.table(), &bk);
            solver.set_terminal_design(TerminalDesign::Design1Collapsed);
            solver.run(&rt, &game, &bk, 2);
            let root_all = solver.root_cfv_from_avg(&rt, &game, &bk);
            root_all.iter().map(|ph| perm.iter().map(|&h| ph[h]).collect()).collect()
        } else {
            // live-3/4/5: the PRODUCTION GPU bucketed solve (the bulk; the memory
            // driver). Solve cold, load the avg back, extract the root CFV.
            let bt = betting_trees
                .entry((cell.live, cell.commit, cell.pot))
                .or_insert_with(|| {
                    Arc::new(
                        build_tree(&spec.flop_seam_config(cell.live, cell.commit, cell.pot, bets.clone()))
                            .expect("seam tree"),
                    )
                })
                .clone();
            // blueprint per-live fidelity: live-3/4 @ B15, live-5 @ B8.
            let cell_nb: usize = if cell.live <= 4 { 15 } else { 8 };
            let tbl = FlopChanceTable::build_full_nh_sampled(canonical, cell.live, &turns, &river_decks);
            let bk = FlopBucketing::quantile(&tbl, cell_nb);
            let perm = table_hand_to_layout_perm(&tbl.hand_cards, tbl.num_valid, canonical);
            let game = FlopStartGame::new(tbl);
            let mut solver = BucketedFlopCfr::new(&bt, game.table(), &bk);
            solver.set_terminal_design(TerminalDesign::Design1Collapsed);
            let mut n =
                BucketedNativeGpu::new(&ctx, &bt, game.table(), &bk, &solver, (32 / cell_nb) as u32)
                    .expect("native gpu");
            n.run(post_iters);
            solver.cum_strategy_flop_mut().copy_from_slice(n.cum_strategy_flop());
            solver.cum_strategy_turn_mut().copy_from_slice(n.cum_strategy_turn());
            solver.cum_strategy_river_mut().copy_from_slice(n.cum_strategy_river());
            let root_all = solver.root_cfv_from_avg(&bt, &game, &bk);
            root_all.iter().map(|ph| perm.iter().map(|&h| ph[h]).collect()).collect()
        };
        // garbage-catcher: a non-finite CFV would silently corrupt the preflop.
        assert!(
            per_live.iter().all(|v| v.iter().all(|x| x.is_finite())),
            "non-finite CFV — cell {cell:?} flop {canonical:?}"
        );
        src_cache.insert((key, canonical), per_live.clone());
        per_live[trav_live].clone()
    };

    // ---- the joint cold loop ----------------------------------------------
    let mut oracle = BucketKeyedOracle::new(stack, 6, 1, &mut source);
    let mut solver = PreflopVectorCfr::new(&tree);
    eprintln!("  infosets {} | RSS after preflop CFR alloc: {:.2} GB", solver.infoset_count, guard.peak_gb());

    let t_all = Instant::now();
    for it in 0..pf_cap {
        if guard.should_abort() {
            eprintln!("MEM_GUARD soft abort at preflop iter {it} — peak RSS {:.2} GB; exiting clean", guard.peak_gb());
            break;
        }
        epoch.set(it + 1); // new iteration ⇒ source clears its per-iter cache
        let prev = gpu_count.load(std::sync::atomic::Ordering::Relaxed);
        let t0 = Instant::now();
        solver.run_one_iteration(&tree, &table, &mut oracle, &term_fn);
        let this_iter = gpu_count.load(std::sync::atomic::Ordering::Relaxed) - prev;
        eprintln!(
            "  preflop iter {it}: {:.1}s | {this_iter} unique subgame solves | peak RSS {:.2} GB",
            t0.elapsed().as_secs_f64(),
            guard.peak_gb(),
        );
    }
    let avg = solver.average_strategy(&tree);
    let finite = avg.iter().all(|x| x.is_finite());
    let (mn, mx) = avg.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    eprintln!(
        "DIAGNOSTIC: preflop avg strategy finite={finite}, range [{mn:.4}, {mx:.4}], {} entries",
        avg.len()
    );
    assert!(finite, "preflop avg strategy NON-FINITE — garbage produced");
    eprintln!(
        "DONE: {} preflop iters in {:.1} min | PEAK RSS {:.2} GB (soft {:.0} / hard {:.0})",
        pf_cap,
        t_all.elapsed().as_secs_f64() / 60.0,
        guard.peak_gb(),
        guard.soft_gb(),
        guard.hard_gb(),
    );
}
