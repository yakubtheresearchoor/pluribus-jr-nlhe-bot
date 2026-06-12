// Disturbance test: prove the d_strategy upload at GPU constructor is dead.
//
// Background: post the 2026-06-05 streaming-strategy refactor, CPU's
// strategy_*() getters return scratch buffers (76 MB river / 14 MB turn)
// instead of the full per-(tc, rc) 175 GB / 671 MB they used to. The GPU
// constructor at gpu_metal/flop_solver.rs previously read those getters
// into a full-sized d_strategy buffer — leaving bytes past the scratch
// uninitialized. The fix was to initialize d_strategy with zeros.
//
// CLAIM made when applying the zeros fix: "GPU's run loop calls
// compute_all_strategies(ctx) at the start of every iter (line 507),
// which overwrites d_strategy from d_regrets BEFORE any code reads it,
// so the init value of d_strategy is dead."
//
// This test is the DISCRIMINATING evidence for that claim, per the lead's
// directive. If the claim is true, then poisoning d_strategy with NaN
// after construction must still produce NaN-free output: the GPU
// overwrites the NaN before reading. If the claim is false (some read
// path uses d_strategy before compute_all_strategies overwrites it), the
// NaN will propagate and download_regrets/cum_strategy will contain NaN.
//
// Why NaN, not 1.0: NaN is an unmistakable leak signal. Any float-arith
// path touching a NaN produces NaN, which propagates. Non-NaN garbage
// could be silently absorbed by normalization or convergence; NaN cannot.
//
// Two checks:
//   1. Output contains no NaN (the leak signal).
//   2. Output is bit-identical to a baseline run with zero-init d_strategy.
//      If identical, the upload is provably dead (init value doesn't
//      affect any downstream computation). If different, the upload is
//      live and the architectural cleanup needs more care.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::{MetalContext, MetalFlopStartSolver};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn build_tiny_hu_setup() -> (solver_core::tree::flat::FlatTree, FlopStartGame) {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 10,
        starting_stacks: vec![95, 95],
        initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("HU 1-bet flop tree");

    let np = 2u8;
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter()
        .map(|s| card_from_str(s).unwrap()).collect();
    let combo_ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS])
        .collect();
    let table = FlopChanceTable::compute_flop_start(&board, &combo_ranges, np);
    let game = FlopStartGame::new(table);
    (tree, game)
}

fn run_n_iters_and_download(
    n_iters: u32,
    poison_with_nan: bool,
) -> (Vec<f32>, Vec<f32>) {
    let (tree, game) = build_tiny_hu_setup();
    let table = game.table();
    let cpu = FlopStartVectorCfr::new(&tree, table);

    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    if poison_with_nan {
        let len = gpu.strategy_buffer_len();
        let nan_buf = vec![f32::NAN; len];
        gpu.poison_strategy(&nan_buf);
        eprintln!("  poisoned d_strategy with NaN ({} f32 = {} MB)",
                  len, len * 4 / (1 << 20));
    }

    for _ in 0..n_iters {
        gpu.run(&ctx, &tree, &game, 1);
    }

    (gpu.download_regrets(), gpu.download_cum_strategy())
}

/// Part 1 of the disturbance: prove that `compute_all_strategies(ctx)` —
/// the kernel the run loop dispatches at the START of every iter — does
/// overwrite d_strategy with regret-matched values, regardless of init.
/// This is the smallest unit version: poison d_strategy with NaN, call
/// compute_all_strategies(ctx) directly (no run loop), download d_strategy,
/// confirm no NaN.
#[test]
fn d_strategy_overwritten_by_compute_all_strategies() {
    let (tree, game) = build_tiny_hu_setup();
    let table = game.table();
    let cpu = FlopStartVectorCfr::new(&tree, table);
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let len = gpu.strategy_buffer_len();
    eprintln!("\nd_strategy buffer: {} f32 = {} MB", len, len * 4 / (1 << 20));

    // 1. Poison with NaN.
    gpu.poison_strategy(&vec![f32::NAN; len]);
    let after_poison = gpu.download_strategy();
    let poison_nan = after_poison.iter().filter(|x| x.is_nan()).count();
    eprintln!("  after poison: {} / {} NaN values (should be all)", poison_nan, len);
    assert_eq!(poison_nan, len, "poison failed to write NaN to all slots");

    // 2. Call compute_all_strategies(ctx) — the kernel run() dispatches.
    gpu.compute_all_strategies(&ctx);
    let after_recompute = gpu.download_strategy();
    let post_nan = after_recompute.iter().filter(|x| x.is_nan()).count();
    eprintln!("  after compute_all_strategies(ctx): {} / {} NaN values (should be 0)",
              post_nan, len);

    assert_eq!(post_nan, 0,
        "NaN survives compute_all_strategies(ctx) — the kernel does NOT overwrite \
         all d_strategy slots. The init-upload-is-dead claim is FALSE.");
    eprintln!("  → compute_all_strategies(ctx) overwrites every slot of d_strategy.\n");
}

#[test]
fn d_strategy_upload_is_dead_nan_disturbance() {
    eprintln!("\n=== Disturbance test: d_strategy upload is dead (NaN-leak check) ===\n");

    // Baseline: standard zero-init (what the current code does).
    eprintln!("Run 1: BASELINE (zero-init d_strategy)");
    let (baseline_reg, baseline_cum) = run_n_iters_and_download(1, false);
    let baseline_nan_count = baseline_reg.iter().filter(|x| x.is_nan()).count()
        + baseline_cum.iter().filter(|x| x.is_nan()).count();
    eprintln!("  NaN count in regrets+cum_strategy: {}", baseline_nan_count);
    assert_eq!(baseline_nan_count, 0,
        "BASELINE run produced NaN — bug independent of disturbance");

    // Disturbance: NaN-poison d_strategy at init, then run identically.
    eprintln!("\nRun 2: DISTURBED (NaN-poison d_strategy at init)");
    let (disturbed_reg, disturbed_cum) = run_n_iters_and_download(1, true);
    let disturbed_reg_nan = disturbed_reg.iter().filter(|x| x.is_nan()).count();
    let disturbed_cum_nan = disturbed_cum.iter().filter(|x| x.is_nan()).count();
    eprintln!("  NaN count in regrets: {} / {}", disturbed_reg_nan, disturbed_reg.len());
    eprintln!("  NaN count in cum_strategy: {} / {}", disturbed_cum_nan, disturbed_cum.len());

    // Check 1: no NaN leak.
    assert_eq!(disturbed_reg_nan, 0,
        "NaN leaked into regrets — d_strategy upload IS live (GPU reads init values \
         before compute_all_strategies overwrites). Architecture cleanup cannot proceed \
         without auditing the read paths.");
    assert_eq!(disturbed_cum_nan, 0,
        "NaN leaked into cum_strategy — same diagnosis as regrets check.");

    // Check 2: bit-identical to baseline.
    assert_eq!(baseline_reg.len(), disturbed_reg.len(), "len mismatch");
    assert_eq!(baseline_cum.len(), disturbed_cum.len(), "len mismatch");

    let mut reg_diffs = 0usize;
    let mut max_reg_diff = 0.0f32;
    for i in 0..baseline_reg.len() {
        if baseline_reg[i].to_bits() != disturbed_reg[i].to_bits() {
            reg_diffs += 1;
            let d = (baseline_reg[i] - disturbed_reg[i]).abs();
            if d > max_reg_diff { max_reg_diff = d; }
        }
    }
    let mut cum_diffs = 0usize;
    let mut max_cum_diff = 0.0f32;
    for i in 0..baseline_cum.len() {
        if baseline_cum[i].to_bits() != disturbed_cum[i].to_bits() {
            cum_diffs += 1;
            let d = (baseline_cum[i] - disturbed_cum[i]).abs();
            if d > max_cum_diff { max_cum_diff = d; }
        }
    }
    eprintln!("\n  Bit-identical check:");
    eprintln!("    regrets:       {} / {} diffs (max abs = {:.6e})",
              reg_diffs, baseline_reg.len(), max_reg_diff);
    eprintln!("    cum_strategy:  {} / {} diffs (max abs = {:.6e})",
              cum_diffs, baseline_cum.len(), max_cum_diff);

    assert_eq!(reg_diffs, 0,
        "regrets output differs between zero-init and NaN-init d_strategy — \
         upload value affects computation, so the upload is NOT dead. \
         Constructor cleanup cannot proceed.");
    assert_eq!(cum_diffs, 0,
        "cum_strategy output differs — same diagnosis as regrets.");

    eprintln!("\n=== VERDICT: d_strategy upload at GPU constructor IS DEAD ===");
    eprintln!("  - NaN init produces no NaN output");
    eprintln!("  - NaN init produces bit-identical output to zero init");
    eprintln!("  → GPU compute_all_strategies(ctx) at run() start overwrites d_strategy");
    eprintln!("    before any read. The init value is unreachable.");
    eprintln!("  → Architectural cleanup (remove &cpu from constructor, zero-init internally,");
    eprintln!("    or skip the init upload entirely) is safe.");
}
