//! GPU full-nh flop-solve benchmark — the Path-A determinant.
//!
//! Question: can a GPU `MetalFlopStartSolver` solve a REAL flop (all 1326
//! valid hero hands) fast enough to BE the real-time search, replacing the
//! 200-bucket depth-limited CPU search with an exact per-hand solve?
//!
//! The full-river-enumeration flop game is the 175 GB monster (see
//! end_to_end_real_cost). For a real-time solve we Monte-Carlo the runouts:
//! all hero hands (lossless current street) but a SAMPLED turn/river deck
//! (NT turn cards × NR rivers each), which keeps GPU memory feasible and is
//! exactly the depth-limited-by-sampling shape a real-time solve wants.
//!
//! Env knobs (all optional):
//!   GPU_BENCH_NT   turn cards sampled               (default 8)
//!   GPU_BENCH_NR   rivers per turn card             (default 8)
//!   GPU_BENCH_IT   GPU iterations to time           (default 100)
//!   GPU_BENCH_BET  betting richness: "rich"|"pot"   (default "rich")
//!
//! Reports: nh, buffer memory, ms/iter, and the projected iters that fit in
//! the 14 s budget. This is a COST measurement (no correctness assertion;
//! GPU↔CPU parity is established by the convergence_audit gate).

#![cfg(feature = "metal")]

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::{MetalContext, MetalFlopStartSolver, MetalVectorCfr};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::{build_tree, build_tree_depth_limited};
use solver_core::tree::flat::NODE_TYPE_TERMINAL;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn fmt_ms(ms: f64) -> String {
    if ms < 1000.0 { format!("{ms:.1} ms") } else { format!("{:.2} s", ms / 1000.0) }
}

#[test]
#[ignore = "GPU cost benchmark — run on demand with --features metal --release"]
fn gpu_full_nh_flop_solve_cost() {
    let nt = env_usize("GPU_BENCH_NT", 8);
    let nr = env_usize("GPU_BENCH_NR", 8);
    let n_iters = env_usize("GPU_BENCH_IT", 100) as u32;
    let rich = std::env::var("GPU_BENCH_BET").map(|v| v != "pot").unwrap_or(true);

    eprintln!("\n══════════════════════════════════════════════════════════════════");
    eprintln!("=== GPU full-nh flop-solve benchmark (Path-A determinant)       ===");
    eprintln!("===   all hero hands, sampled runout {nt} turns × {nr} rivers          ===");
    eprintln!("══════════════════════════════════════════════════════════════════\n");

    // ---- canonical flop ----
    let np = 2u8;
    let class_weights: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let pre_table = PreflopChanceTable::new(np, class_weights);
    let canonical: [Card; 3] = pre_table.canonical_flops[0];
    let board: Vec<Card> = canonical.iter().copied().collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));

    // ---- all valid hero hands (lossless current street) ----
    let hand_indices: Vec<u16> = (0..NUM_POSSIBLE_HANDS)
        .filter(|&hi| {
            let (c1, c2) = index_to_card_pair(hi);
            board_mask & (1u64 << c1) == 0 && board_mask & (1u64 << c2) == 0
        })
        .map(|hi| hi as u16)
        .collect();

    // ---- sampled runout: NT turn cards, each with NR rivers ----
    let non_board: Vec<u8> = (0..52u8).filter(|&c| board_mask & (1u64 << c) == 0).collect();
    let turn_cards: Vec<u8> = non_board.iter().copied().take(nt).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        river_decks[tc as usize] = non_board
            .iter()
            .copied()
            .filter(|&c| c != tc)
            .take(nr)
            .collect();
    }

    let ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS])
        .collect();

    let t = Instant::now();
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, np, &hand_indices, &turn_cards, &river_decks,
    );
    let nh = table.num_valid;
    eprintln!("table build:   {} (nh = {nh}, turns = {}, rivers/turn ≈ {nr})",
              fmt_ms(t.elapsed().as_secs_f64() * 1000.0), turn_cards.len());

    // ---- production-ish betting tree ----
    let bet_sizes = if rich {
        BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.75)],
        }
    } else {
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] }
    };
    let cfg = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 20,
        starting_stacks: vec![100; np as usize],
        initial_contributions: vec![0; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes,
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
        max_bets_per_street: None,
        no_open_limp: false, threebet_or_fold: false,
    };
    let tree = build_tree(&cfg).expect("tree");
    eprintln!("tree:          {} nodes ({})", tree.num_nodes(), if rich { "rich bets" } else { "pot-only" });

    let game = FlopStartGame::new(table);
    let cpu = FlopStartVectorCfr::new(&tree, game.table());
    let buf_len = cpu.river_persistent_len();
    // d_regrets + d_strategy + d_cum_strategy each ≈ buf_len f32; plus reach
    // buffers (nn·np·nh ×3) and cfv (nn·nh). Report the dominant solver-state.
    let solver_state_gb = 3.0 * buf_len as f64 * 4.0 / 1e9;
    eprintln!("solver state:  {:.2} GB (3 × {} f32 regret/strategy/cum)", solver_state_gb, buf_len);

    // ---- GPU solve, timed ----
    let ctx = MetalContext::new().expect("Metal");
    let t = Instant::now();
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);
    eprintln!("gpu alloc:     {}", fmt_ms(t.elapsed().as_secs_f64() * 1000.0));

    // warm-up (1 iter — first dispatch pays pipeline/JIT cost)
    gpu.run(&ctx, &tree, &game, 1);

    let t = Instant::now();
    gpu.run(&ctx, &tree, &game, n_iters);
    let total_ms = t.elapsed().as_secs_f64() * 1000.0;
    let per_iter = total_ms / n_iters as f64;

    let strat = gpu.download_strategy();
    let any_nan = strat.iter().any(|x| x.is_nan());
    let any_nz = strat.iter().any(|x| x.abs() > 1e-9);

    eprintln!("\n── GPU solve ─────────────────────────────────────────────────────");
    eprintln!("  {n_iters} iters:     {}", fmt_ms(total_ms));
    eprintln!("  per iter:      {}", fmt_ms(per_iter));
    eprintln!("  strategy:      non-zero={any_nz}, NaN={any_nan}");
    assert!(!any_nan, "GPU strategy contains NaN — solver state corrupt");
    assert!(any_nz, "GPU strategy all-zero — likely OOM silent failure (>27 GB?)");

    let budget_ms = 14_000.0;
    let fit_iters = (budget_ms / per_iter).floor() as u64;
    eprintln!("\n── 14 s budget ───────────────────────────────────────────────────");
    eprintln!("  iters that fit: {fit_iters}");
    eprintln!("  verdict: {}", if fit_iters >= 200 {
        "✓ comfortably enough iters for convergence — Path A VIABLE at this runout"
    } else if fit_iters >= 60 {
        "~ marginal — enough for a coarse solve; check convergence quality"
    } else {
        "✗ too few iters — exact GPU solve too slow at this runout"
    });
    eprintln!("══════════════════════════════════════════════════════════════════\n");
}

/// DEPTH-LIMITED GPU CFR-loop benchmark — the Path-A/B determinant.
///
/// The full-game solver is 73 s/iter because it expands the whole river. The
/// real-time search is DEPTH-LIMITED: solve only the current street's betting
/// tree (per-hand, lossless) and read a continuation value at the leaf — no
/// river expansion. This times the GPU CFR loop (`MetalVectorCfr`) over a
/// production depth-limited flop tree at full nh.
///
/// To isolate the CFR-loop cost with existing validated code, the depth-limit
/// chance leaves are flipped to SHOWDOWN terminals (an UPPER BOUND on the leaf
/// cost: a per-hand O(nh²) showdown is strictly heavier than the real per-bucket
/// MC-sampled continuation). If THIS fits the budget, the real solver — with the
/// cheaper bucketed leaf — fits comfortably.
///
/// Env: GPU_DL_IT (iters, default 500).
#[test]
#[ignore = "GPU cost benchmark — run with --features metal --release"]
fn gpu_depth_limited_cfr_loop_cost() {
    let n_iters = env_usize("GPU_DL_IT", 500) as u32;

    eprintln!("\n══════════════════════════════════════════════════════════════════");
    eprintln!("=== GPU depth-limited CFR-loop benchmark (Path-A/B determinant)  ===");
    eprintln!("===   flop betting tree, per-hand, leaf = showdown (UB on cont)  ===");
    eprintln!("══════════════════════════════════════════════════════════════════\n");

    // ---- canonical flop + full-nh showdown arrays (production nh) ----
    let np = env_usize("GPU_DL_NP", 2) as u8;
    let class_weights: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let pre_table = PreflopChanceTable::new(np, class_weights);
    let canonical: [Card; 3] = pre_table.canonical_flops[0];
    let board: Vec<Card> = canonical.iter().copied().collect();
    let ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS])
        .collect();
    // The leaf is a continuation/showdown, so flop-level sorted arrays suffice.
    // Always use the subset-with-decks path (full hands + a MINIMAL 1-turn×1-river
    // deck) — light enough for np≥3, and the continuation leaf integrates the
    // runout in bucket space anyway. GPU_DL_NH=k subsamples hero hands.
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let valid: Vec<u16> = (0..NUM_POSSIBLE_HANDS)
        .filter(|&hi| { let (c1, c2) = index_to_card_pair(hi);
            board_mask & (1u64 << c1) == 0 && board_mask & (1u64 << c2) == 0 })
        .map(|hi| hi as u16).collect();
    let sub_nh = env_usize("GPU_DL_NH", 0);
    let hands: Vec<u16> = if sub_nh == 0 { valid.clone() } else {
        let step = (valid.len() / sub_nh).max(1);
        valid.iter().step_by(step).copied().take(sub_nh).collect()
    };
    let nbc: Vec<u8> = (0..52u8).filter(|&c| board_mask & (1u64 << c) == 0).collect();
    let turn = nbc[0];
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    rd[turn as usize] = vec![nbc[1]];
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(&board, &ranges, np, &hands, &[turn], &rd);
    let nh = table.num_valid;
    let (sos, soi, sps, spi, _) = table.sorted_opp_arrays_base();
    let iw: Vec<Vec<f32>> = (0..np as usize).map(|p| table.initial_weights[p].clone()).collect();
    let hc = table.hand_cards.clone();
    let nc = table.num_combinations;
    eprintln!("nh = {nh}  (full lossless current street)");

    // ---- production depth-limited flop tree (chance leaves → terminals) ----
    let cfg = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 20,
        starting_stacks: vec![100; np as usize],
        initial_contributions: vec![0; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.75)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
        max_bets_per_street: None,
        no_open_limp: false, threebet_or_fold: false,
    };
    // GPU_DL_CONT=1 → measure the REAL bucketed continuation leaf (keep leaves
    // as childless chance, install set_continuation). Else the showdown proxy
    // (flip leaves → terminals).
    let use_cont = std::env::var("GPU_DL_CONT").is_ok();
    // GPU_DL_BARE: leave continuation leaves as childless chance (cheap CHANCE
    // branch keeps cfv=0), NO continuation install, NO showdown flip → isolates
    // the BASE CFR loop + fold-terminal cost.
    let bare = std::env::var("GPU_DL_BARE").is_ok();
    let mut tree = build_tree_depth_limited(&cfg).expect("depth-limited tree");
    let leaf_nodes: Vec<u32> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_chance() && tree.node_children(n).is_empty())
        .map(|n| n as u32).collect();
    if bare {
        eprintln!("tree: {} nodes, {} leaves → BARE (no leaf valuation)", tree.num_nodes(), leaf_nodes.len());
    } else if !use_cont {
        for &n in &leaf_nodes { tree.nodes[n as usize].node_type = NODE_TYPE_TERMINAL; }
        tree.compute_levels();
        eprintln!("tree: {} nodes, {} continuation leaves → showdown terminals (proxy)",
                  tree.num_nodes(), leaf_nodes.len());
    } else {
        eprintln!("tree: {} nodes, {} continuation leaves → BUCKETED continuation (nb=200)",
                  tree.num_nodes(), leaf_nodes.len());
    }

    // ---- GPU CFR loop, timed ----
    let ctx = MetalContext::new().expect("Metal");
    let t = Instant::now();
    let mut gpu = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &hc, nc);
    eprintln!("gpu alloc: {}", fmt_ms(t.elapsed().as_secs_f64() * 1000.0));

    // Fast lone-survivor terminals (np≥3): GPU_DL_FAST=1 installs the parallel
    // terminal kernel for terminals with <=1 live player (every terminal of a
    // no-all-in depth-limited tree). Removes the O(nh³)/node base bottom-up cost.
    if std::env::var("GPU_DL_FAST").is_ok() && np == 3 {
        let lone: Vec<u32> = (0..tree.num_nodes())
            .filter(|&n| tree.nodes[n].is_terminal())
            .filter(|&n| {
                let fm = tree.get_folded_mask(n);
                let live = (0..np).filter(|&p| fm & (1 << p) == 0).count();
                live <= 1
            })
            .map(|n| n as u32).collect();
        let factored = std::env::var("GPU_DL_FACTORED").is_ok();
        eprintln!("fast lone-survivor terminals: {} ({})", lone.len(), if factored { "factored O(nh)" } else { "brute O(nh²)" });
        gpu.set_fast_lone_terminals_ex(&ctx, &lone, factored);
    }

    if use_cont && !bare {
        // Synthetic valid tables at production nb=200 (timing is independent of
        // table values; correctness is pinned by gpu_continuation_leaf_parity).
        let nb = 200usize;
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rf = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed >> 33) as f32) / (1u64 << 31) as f32 };
        let (mut fw, mut ft, mut fl, mut fnn) = (vec![0.0f32; nb*nb], vec![0.0f32; nb*nb], vec![0.0f32; nb*nb], vec![0.0f32; nb*nb]);
        for i in 0..nb*nb { let cm = 0.3 + 0.7*rf(); let (a,b,c)=(rf(),rf(),rf()); let s=a+b+c;
            fw[i]=cm*a/s; ft[i]=cm*b/s; fl[i]=cm*c/s; fnn[i]=fw[i]+ft[i]+fl[i]; }
        let map: Vec<u16> = (0..nh).map(|h| (h % nb) as u16).collect();
        gpu.set_continuation(&ctx, &leaf_nodes, &map, nb, &fw, &ft, &fl, &fnn, 0.0, 0.0, 500, 7);
    }

    let async_mode = std::env::var("GPU_DL_SYNC").is_err(); // default: async latency path
    gpu.run_batched(&ctx, &tree, 1); // warm-up

    let t = Instant::now();
    if async_mode {
        gpu.run_batched(&ctx, &tree, n_iters);
    } else {
        gpu.run(&ctx, &tree, n_iters);
    }
    let total_ms = t.elapsed().as_secs_f64() * 1000.0;
    let per_iter = total_ms / n_iters as f64;
    eprintln!("mode: {}", if async_mode { "run_batched (async, no per-iter readback)" } else { "run (sync)" });

    let cum = gpu.cum_strategy_slice();
    let any_nan = cum.iter().any(|x| x.is_nan());
    let any_nz = cum.iter().any(|x| x.abs() > 1e-9);

    eprintln!("\n── GPU depth-limited CFR ──────────────────────────────────────────");
    eprintln!("  {n_iters} iters: {}", fmt_ms(total_ms));
    eprintln!("  per iter:  {}", fmt_ms(per_iter));
    eprintln!("  cum_strategy: non-zero={any_nz}, NaN={any_nan}");
    assert!(!any_nan, "NaN in cum_strategy");
    assert!(any_nz, "cum_strategy all-zero");

    let budget_ms = 14_000.0;
    let fit = (budget_ms / per_iter).floor() as u64;
    eprintln!("\n── 14 s budget ───────────────────────────────────────────────────");
    eprintln!("  betting-tree CFR iters that fit: {fit}");
    eprintln!("  (real solver adds a per-bucket MC continuation leaf — CHEAPER than");
    eprintln!("   this per-hand showdown proxy; ~30 ms/full-zone at B=32 from the");
    eprintln!("   terminal-sampling gate, amortized across the leaves)");
    eprintln!("  verdict: {}", if fit >= 1000 {
        "✓✓ CFR loop is ms-scale — depth-limited GPU search is VIABLE with huge headroom"
    } else if fit >= 200 {
        "✓ enough iters for convergence — VIABLE"
    } else {
        "~ check: CFR loop heavier than expected"
    });
    eprintln!("══════════════════════════════════════════════════════════════════\n");
}
