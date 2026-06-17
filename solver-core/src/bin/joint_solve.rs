//! JOINT COLD-CFR FACTORY (parallel, 2026-06-17).
//!
//! PHASE 1 — solve every flop-entry subgame ONCE, in parallel, into a persistent
//! cache: live-2 on CPU (exact HU, bit-exact to the GPU path but 28× faster for
//! these tiny 1×1 subgames — live2_gpu_gate), live-3/4/5 on GPU (one
//! MetalContext per worker thread, the fill's pattern), live-6/all-in CPU equity
//! rollout. CPU work (rayon, all cores) and GPU work (N threads) run concurrently.
//!
//! PHASE 2 — run the production preflop CFR against the cached postflop values
//! (frozen oracle, refresh_every=0 — the postflop is solved once, not re-solved
//! per preflop iter). No warm-start, no architecture change; just the parallel
//! fill + the correct engine per family.
//!
//! Env: JOINT_PF_CAP (preflop iters, default 60), JOINT_POST_ITERS (DCFR
//! iters/subgame, default 34), JOINT_GPU_THREADS (default 4), JOINT_KEY_LIMIT
//! (debug: solve only the first N distinct cell-keys). MEM_GUARD_* as before.

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("joint_solve requires --features metal");
    std::process::exit(2);
}

#[cfg(feature = "metal")]
use solver_core::card::Card;
#[cfg(feature = "metal")]
use solver_core::gpu_metal::context::MetalContext;
#[cfg(feature = "metal")]
use solver_core::solver::postflop_oracle::SeamCell;
#[cfg(feature = "metal")]
use solver_core::tree::action::GameSpec;

/// Seeded 1×1 runout (matches the fill convention).
#[cfg(feature = "metal")]
fn seeded_1x1(canonical: [Card; 3], fi: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let sm = |mut x: u64| -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let mut pool = deck.clone();
    let tc = pool.remove((sm(fi as u64) % pool.len() as u64) as usize);
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    let mut rpool: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
    let rc = rpool.remove((sm((fi as u64) ^ 0xDEAD_BEEF ^ ((tc as u64) << 40)) % rpool.len() as u64) as usize);
    rd[tc as usize] = vec![rc];
    (vec![tc], rd)
}

/// Solve one (cell, canonical) subgame → per-live-seat, layout-ordered CFV.
#[cfg(feature = "metal")]
fn solve_cell(
    cell: SeamCell,
    canonical: [Card; 3],
    fi: usize,
    ctx: Option<&MetalContext>,
    spec: &GameSpec,
    post_iters: u32,
) -> Vec<Vec<f32>> {
    use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
    use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
    use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
    use solver_core::solver::preflop_start_game::{
        compute_v_flop_at_root_converged_with_table, flop_combo_layout, table_hand_to_layout_perm,
    };
    use solver_core::tree::action::{BetSize, BetSizeOptions};
    use solver_core::tree::builder::build_tree;
    use std::collections::HashMap;

    let (turns, river_decks) = seeded_1x1(canonical, fi);
    let key = cell.bucket_key(spec.stack);
    let is_allin = key.1 == i64::MIN;
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };

    if cell.live == 2 {
        // live-2: CPU exact HU (bit-exact to GPU, 28× faster on tiny subgames).
        let tree = build_tree(&spec.flop_seam_config(2, cell.commit, cell.pot, bets)).expect("live-2 tree");
        let layout = flop_combo_layout(canonical);
        (0..2u8)
            .map(|t| {
                let tbl = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
                let (v_tbl, lay) = compute_v_flop_at_root_converged_with_table(tbl, &tree, t, post_iters);
                let mut pos: HashMap<(Card, Card), usize> = HashMap::new();
                for (i, &(a, b)) in lay.iter().enumerate() {
                    pos.insert((a.min(b), a.max(b)), i);
                }
                layout.iter().map(|&(a, b)| v_tbl[pos[&(a.min(b), a.max(b))]]).collect()
            })
            .collect()
    } else if is_allin || cell.live == 6 {
        // live-6 / all-in: CPU equity rollout (check-only).
        let mut cfg = spec.flop_seam_config(
            cell.live,
            cell.commit.min(spec.stack),
            cell.pot,
            BetSizeOptions { bet: vec![], raise: vec![] },
        );
        cfg.add_allin_threshold = 0.0;
        cfg.force_allin_threshold = 0.0;
        let tree = build_tree(&cfg).expect("rollout tree");
        let tbl = FlopChanceTable::build_full_nh_sampled(canonical, cell.live, &turns, &river_decks);
        let bk = FlopBucketing::quantile(&tbl, 8);
        let perm = table_hand_to_layout_perm(&tbl.hand_cards, tbl.num_valid, canonical);
        let game = FlopStartGame::new(tbl);
        let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
        solver.set_terminal_design(TerminalDesign::Design1Collapsed);
        solver.run(&tree, &game, &bk, 2);
        let root = solver.root_cfv_from_avg(&tree, &game, &bk);
        root.iter().map(|ph| perm.iter().map(|&h| ph[h]).collect()).collect()
    } else {
        // live-3/4/5: GPU bucketed (B15 / B8) — GPU is 5-13× faster than CPU here.
        let ctx = ctx.expect("multiway needs a GPU context");
        let cell_nb: usize = if cell.live <= 4 { 15 } else { 8 };
        let tree = build_tree(&spec.flop_seam_config(cell.live, cell.commit, cell.pot, bets)).expect("seam tree");
        let tbl = FlopChanceTable::build_full_nh_sampled(canonical, cell.live, &turns, &river_decks);
        let bk = FlopBucketing::quantile(&tbl, cell_nb);
        let perm = table_hand_to_layout_perm(&tbl.hand_cards, tbl.num_valid, canonical);
        let game = FlopStartGame::new(tbl);
        let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
        solver.set_terminal_design(TerminalDesign::Design1Collapsed);
        let mut n = BucketedNativeGpu::new(ctx, &tree, game.table(), &bk, &solver, (32 / cell_nb) as u32)
            .expect("native gpu");
        n.run(post_iters);
        solver.cum_strategy_flop_mut().copy_from_slice(n.cum_strategy_flop());
        solver.cum_strategy_turn_mut().copy_from_slice(n.cum_strategy_turn());
        solver.cum_strategy_river_mut().copy_from_slice(n.cum_strategy_river());
        let root = solver.root_cfv_from_avg(&tree, &game, &bk);
        root.iter().map(|ph| perm.iter().map(|&h| ph[h]).collect()).collect()
    }
}

#[cfg(feature = "metal")]
fn main() {
    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
    use solver_core::mem_guard::MemGuard;
    use solver_core::solver::postflop_oracle::BucketKeyedOracle;
    use solver_core::solver::preflop_cfr::{
        make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr,
    };
    use solver_core::solver::preflop_start_game::PreflopChanceTable;
    use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
    use solver_core::tree::builder::build_tree_preflop_only;
    use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

    let env = |k: &str, d: u64| -> u64 { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) };
    let pf_cap = env("JOINT_PF_CAP", 60) as u32;
    let post_iters = env("JOINT_POST_ITERS", 34) as u32;
    let gpu_threads = env("JOINT_GPU_THREADS", 4) as usize;
    let key_limit = std::env::var("JOINT_KEY_LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let spec = production_game_v1();
    let stack = spec.stack;
    let guard = MemGuard::from_env();

    // production cap-3 preflop tree at na=16
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
    let canon = table.canonical_flops.clone();
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);
    let mut solver = PreflopVectorCfr::new(&tree);
    eprintln!(
        "na={MAX_NA_PREFLOP} preflop tree {} nodes, {} infosets | RSS {:.2} GB",
        tree.num_nodes(),
        solver.infoset_count,
        guard.peak_gb()
    );

    // distinct representative cell per SPR bucket-key (first seen, matching the
    // on-demand cache's behavior), then × all canonical flops.
    let chance_nodes = solver.preflop_chance_node_indices(&tree);
    let mut rep: HashMap<(u8, i64), SeamCell> = HashMap::new();
    for &c in &chance_nodes {
        let cell = SeamCell::at_chance_node(&tree, c, 6);
        rep.entry(cell.bucket_key(stack)).or_insert(cell);
    }
    let mut keys: Vec<((u8, i64), SeamCell)> = rep.iter().map(|(&k, &c)| (k, c)).collect();
    keys.sort_by_key(|(k, _)| *k);
    if keys.len() > key_limit {
        keys.truncate(key_limit);
    }

    let mut cpu_work: Vec<((u8, i64), SeamCell, [Card; 3], usize)> = Vec::new();
    let mut gpu_work: Vec<((u8, i64), SeamCell, [Card; 3], usize)> = Vec::new();
    for (key, cell) in &keys {
        let is_allin = key.1 == i64::MIN;
        for (fi, &canonical) in canon.iter().enumerate() {
            let item = (*key, *cell, canonical, fi);
            if cell.live == 2 || is_allin || cell.live == 6 {
                cpu_work.push(item);
            } else {
                gpu_work.push(item);
            }
        }
    }
    eprintln!(
        "PHASE 1: {} cell-keys × {} flops = {} subgames | CPU(live-2/rollout) {} + GPU(multiway) {} | {gpu_threads} GPU threads",
        keys.len(), canon.len(), cpu_work.len() + gpu_work.len(), cpu_work.len(), gpu_work.len()
    );

    // ---- parallel pre-solve: CPU (rayon) + GPU (threads) concurrently ----
    let cache: Mutex<HashMap<((u8, i64), [Card; 3]), Vec<Vec<f32>>>> =
        Mutex::new(HashMap::with_capacity(cpu_work.len() + gpu_work.len()));
    let done = AtomicUsize::new(0);
    let gpu_next = AtomicUsize::new(0);
    let total = cpu_work.len() + gpu_work.len();
    let t_fill = Instant::now();
    let report = |d: usize| {
        if d % 5000 == 0 || d == total {
            let el = t_fill.elapsed().as_secs_f64();
            eprintln!(
                "    fill {d}/{total} | {:.1} min | {:.2} GB | ETA {:.1} min",
                el / 60.0,
                guard.peak_gb(),
                if d > 0 { el / d as f64 * (total - d) as f64 / 60.0 } else { 0.0 }
            );
        }
    };
    std::thread::scope(|s| {
        // GPU workers (one MetalContext each), pull from a shared index.
        for _ in 0..gpu_threads {
            let (cache, gpu_next, gpu_work, done, spec, guard) =
                (&cache, &gpu_next, &gpu_work, &done, &spec, &guard);
            s.spawn(move || {
                let ctx = MetalContext::new().expect("Metal context");
                loop {
                    if guard.should_abort() {
                        break;
                    }
                    let k = gpu_next.fetch_add(1, Ordering::Relaxed);
                    if k >= gpu_work.len() {
                        break;
                    }
                    let (key, cell, canonical, fi) = gpu_work[k];
                    let pl = solve_cell(cell, canonical, fi, Some(&ctx), spec, post_iters);
                    cache.lock().unwrap().insert((key, canonical), pl);
                    report(done.fetch_add(1, Ordering::Relaxed) + 1);
                }
            });
        }
        // CPU workers (rayon over all cores), concurrent with the GPU.
        {
            let (cache, cpu_work, done, spec, guard) = (&cache, &cpu_work, &done, &spec, &guard);
            s.spawn(move || {
                cpu_work.par_iter().for_each(|&(key, cell, canonical, fi)| {
                    if guard.should_abort() {
                        return;
                    }
                    let pl = solve_cell(cell, canonical, fi, None, spec, post_iters);
                    cache.lock().unwrap().insert((key, canonical), pl);
                    report(done.fetch_add(1, Ordering::Relaxed) + 1);
                });
            });
        }
    });
    let cache_read = cache.into_inner().unwrap();
    eprintln!(
        "PHASE 1 done: {} subgames in {:.1} min | peak {:.2} GB",
        cache_read.len(),
        t_fill.elapsed().as_secs_f64() / 60.0,
        guard.peak_gb()
    );
    if guard.should_abort() {
        eprintln!("MEM_GUARD soft abort during fill — exiting");
        return;
    }

    // ---- PHASE 2: preflop CFR against the frozen pre-solved cache ----
    let tolerant = key_limit < usize::MAX; // debug: zeros for un-pre-solved keys
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
        match cache_read.get(&(key, canonical)) {
            Some(pl) => pl[trav_live].clone(),
            None if tolerant => {
                vec![0.0f32; solver_core::solver::preflop_start_game::flop_combo_layout(canonical).len()]
            }
            None => panic!("pre-solved cell missing: key {key:?} flop {canonical:?}"),
        }
    };
    let mut oracle = BucketKeyedOracle::new(stack, 6, 0, &mut source);
    eprintln!("PHASE 2: preflop CFR, {pf_cap} iters (frozen oracle)");
    let t_pf = Instant::now();
    for it in 0..pf_cap {
        if guard.should_abort() {
            eprintln!("MEM_GUARD soft abort at preflop iter {it}");
            break;
        }
        let t0 = Instant::now();
        solver.run_one_iteration(&tree, &table, &mut oracle, &term_fn);
        eprintln!("  pf iter {it}: {:.1}s | {:.2} GB", t0.elapsed().as_secs_f64(), guard.peak_gb());
    }
    let avg = solver.average_strategy(&tree);
    assert!(avg.iter().all(|x| x.is_finite()), "preflop avg strategy NON-FINITE");
    eprintln!(
        "DONE: PHASE1 {:.1} min + PHASE2 {:.1} min = total {:.1} min | peak {:.2} GB",
        (t_pf.duration_since(t_fill).as_secs_f64()) / 60.0,
        t_pf.elapsed().as_secs_f64() / 60.0,
        t_fill.elapsed().as_secs_f64() / 60.0,
        guard.peak_gb()
    );
}
