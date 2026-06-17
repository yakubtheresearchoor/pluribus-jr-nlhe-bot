//! JOINT PER-SUBGAME PROBE (2026-06-16): measure where the per-subgame GPU time
//! goes — construction vs run(34) vs extract — for each fidelity (live-2 exact,
//! live-3/4 @ B15, live-5 @ B8). Tells us whether to amortize construction or
//! parallelize dispatch before optimizing the joint driver. REPS each, warm.
//!
//! Env: PROBE_REPS (default 8), PROBE_ITERS (default 34).
//! OUT_DIR must point at the dir holding solver.metallib.

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("joint_probe requires --features metal");
}

#[cfg(feature = "metal")]
fn main() {
    use std::time::Instant;

    use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
    use solver_core::card::Card;
    use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
    use solver_core::gpu_metal::context::MetalContext;
    use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
    use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
    use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
    use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
    use solver_core::solver::preflop_start_game::{
        extract_v_flop_root_from_cum, PreflopChanceTable,
    };
    use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
    use solver_core::tree::builder::build_tree;

    let reps: usize = std::env::var("PROBE_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: u32 = std::env::var("PROBE_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(34);
    let spec = production_game_v1();
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };

    let ptable = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let canonical = ptable.canonical_flops[0];
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let turns = vec![deck[0]];
    let mut river_decks = vec![vec![]; 52];
    river_decks[deck[0] as usize] = vec![deck[1]];

    let ctx = MetalContext::new().expect("Metal context (set OUT_DIR)");
    eprintln!("joint_probe: {reps} reps × {iters} iters/subgame\n");

    // ---- bucketed (live-3/4/5) ----
    let probe_bucketed = |live: u8, nb: usize, label: &str| {
        let tree = build_tree(&spec.flop_seam_config(live, 2, 12, bets.clone())).unwrap();
        let (mut t_con, mut t_run, mut t_ext) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..reps {
            let t0 = Instant::now();
            let tbl = FlopChanceTable::build_full_nh_sampled(canonical, live, &turns, &river_decks);
            let bk = FlopBucketing::quantile(&tbl, nb);
            let game = FlopStartGame::new(tbl);
            let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
            solver.set_terminal_design(TerminalDesign::Design1Collapsed);
            let mut n = BucketedNativeGpu::new(&ctx, &tree, game.table(), &bk, &solver, (32 / nb) as u32)
                .expect("native gpu");
            t_con += t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            n.run(iters);
            t_run += t1.elapsed().as_secs_f64();
            let t2 = Instant::now();
            solver.cum_strategy_flop_mut().copy_from_slice(n.cum_strategy_flop());
            solver.cum_strategy_turn_mut().copy_from_slice(n.cum_strategy_turn());
            solver.cum_strategy_river_mut().copy_from_slice(n.cum_strategy_river());
            let _root = solver.root_cfv_from_avg(&tree, &game, &bk);
            t_ext += t2.elapsed().as_secs_f64();
        }
        let r = reps as f64;
        eprintln!(
            "{label:<16} nh~{:>4} | construct {:>7.1}ms | run({iters}) {:>7.1}ms | extract {:>7.1}ms | TOTAL {:>7.1}ms",
            FlopChanceTable::build_full_nh_sampled(canonical, live, &turns, &river_decks).num_valid,
            t_con / r * 1e3, t_run / r * 1e3, t_ext / r * 1e3, (t_con + t_run + t_ext) / r * 1e3
        );
    };
    probe_bucketed(3, 15, "live-3 @B15");
    probe_bucketed(4, 15, "live-4 @B15");
    probe_bucketed(5, 8, "live-5 @B8");

    // ---- exact live-2 (MetalFlopStartSolver) ----
    {
        let tree = build_tree(&spec.flop_seam_config(2, 2, 12, bets.clone())).unwrap();
        let (mut t_con, mut t_run, mut t_ext) = (0.0f64, 0.0f64, 0.0f64);
        let mut nh = 0;
        for _ in 0..reps {
            let t0 = Instant::now();
            let tbl = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
            nh = tbl.num_valid;
            let game = FlopStartGame::new(tbl);
            let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
            let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
            t_con += t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            gpu.run(&ctx, &tree, &game, iters);
            t_run += t1.elapsed().as_secs_f64();
            let t2 = Instant::now();
            let cum = gpu.download_cum_strategy();
            let (nf, nt) = (cpu.cum_strategy_flop().len(), cpu.cum_strategy_turn().len());
            cpu.cum_strategy_flop_mut().copy_from_slice(&cum[0..nf]);
            cpu.cum_strategy_turn_mut().copy_from_slice(&cum[nf..nf + nt]);
            cpu.cum_strategy_river_mut().copy_from_slice(&cum[nf + nt..]);
            let _ = extract_v_flop_root_from_cum(&mut cpu, &tree, &game, 0);
            t_ext += t2.elapsed().as_secs_f64();
        }
        let r = reps as f64;
        eprintln!(
            "{:<16} nh~{nh:>4} | construct {:>7.1}ms | run({iters}) {:>7.1}ms | extract {:>7.1}ms | TOTAL {:>7.1}ms",
            "live-2 exact", t_con / r * 1e3, t_run / r * 1e3, t_ext / r * 1e3,
            (t_con + t_run + t_ext) / r * 1e3
        );
    }
    eprintln!("\nDominant phase ⇒ optimization target: construct=amortize/reuse, run=parallelize dispatch, extract=CPU.");
}
