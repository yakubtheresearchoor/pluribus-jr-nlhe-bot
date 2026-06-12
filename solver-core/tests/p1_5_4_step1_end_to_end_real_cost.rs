// End-to-end real-cost measurement: HU OptB nh=1176, DiskBacked mode,
// ONE preflop-equivalent iter, all stages timed. Replaces the 2-3 hr
// projection (which was bottom_up_zone-only) with an honest measured number.
//
// Per the lead's directive: measure each stage including I/O, don't validate
// against the projection. The measured number answers whether full-nh
// unabstracted HU is feasible on CPU as the lossless reference for C
// (abstraction, #42).
//
// Setup: HU OptB stacks=97 (matches convergence_audit baseline). At HU
// OptB nh=1176 the full per-pair geometry is 49×48=2352 pairs at river_count
// ≈ 4244, giving river_stride ~76 MB per pair and total ~175 GB per buffer.
// In-memory mode cannot run this; DiskBacked is the only feasible mode.
//
// What this test does NOT do: validate correctness (no comparison reference
// at this scale; correctness is established by the A parity gate at the
// largest scale that runs both modes). This test measures cost only.

use std::io::{Seek, SeekFrom, Write};
use std::time::Instant;

/// RAII guard: removes the file when dropped (incl. on panic). Pre-cleans
/// at construction so re-runs after a prior SIGKILL'd run start fresh.
/// (Does not protect against SIGKILL on THIS run — only Drop is hooked.)
struct ScratchFile(std::path::PathBuf);
impl Drop for ScratchFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}
impl ScratchFile {
    fn new(path: std::path::PathBuf) -> Self {
        let _ = std::fs::remove_file(&path);
        Self(path)
    }
    fn path(&self) -> &std::path::Path { &self.0 }
}

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn fmt_dur(ms: u128) -> String {
    if ms < 1000 { format!("{} ms", ms) }
    else if ms < 60_000 { format!("{:.2} s", ms as f64 / 1000.0) }
    else { format!("{:.2} min", ms as f64 / 60_000.0) }
}

#[test]
#[ignore = "End-to-end real-cost: ~2 hr wall-clock at HU OptB nh=1176 in DiskBacked. Run on demand."]
fn end_to_end_real_cost_hu_optb_nh1176_diskbacked() {
    eprintln!("\n========================================================================");
    eprintln!("=== End-to-end real-cost measurement (1 iter, DiskBacked mode)       ===");
    eprintln!("===   HU OptB, stacks=50, nh=1176, full geometry (2352 pairs)        ===");
    eprintln!("========================================================================");
    eprintln!("=== NOTE: stacks=50 instead of production stacks=97 because the      ===");
    eprintln!("=== production size (~350 GB disk) exceeds available SSD space       ===");
    eprintln!("=== (~274 GB free). At stacks=50 the tree is ~7k nodes (vs ~14k at   ===");
    eprintln!("=== stacks=97). Per-iter cost is sub-production by a small factor    ===");
    eprintln!("=== (tree-size proportional); the FEASIBILITY answer scales linearly ===");
    eprintln!("=== and is still discriminating for the abstraction question.        ===");
    eprintln!("========================================================================\n");

    // -------- Setup --------
    let np = 2u8;
    let cfg = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 6,
        starting_stacks: vec![50; np as usize],
        initial_contributions: vec![0; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    };

    let t = Instant::now();
    let tree = build_tree(&cfg).expect("HU OptB tree");
    let setup_tree_ms = t.elapsed().as_millis();
    eprintln!("Stage S1 (tree build):              {} ({} nodes)",
              fmt_dur(setup_tree_ms), tree.num_nodes());

    let t = Instant::now();
    let class_weights: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let pre_table = PreflopChanceTable::new(np, class_weights);
    let canonical: [Card; 3] = pre_table.canonical_flops[0];
    let combo_ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS])
        .collect();
    let board: Vec<Card> = canonical.iter().copied().collect();
    let table = FlopChanceTable::compute_flop_start(&board, &combo_ranges, np);
    let setup_table_ms = t.elapsed().as_millis();
    let nh = table.num_valid;
    eprintln!("Stage S2 (FlopChanceTable build):   {} (nh = {})",
              fmt_dur(setup_table_ms), nh);

    let game = FlopStartGame::new(table);

    let t = Instant::now();
    let solver = FlopStartVectorCfr::new(&tree, game.table());
    let setup_solver_ms = t.elapsed().as_millis();
    let river_persistent_len = solver.river_persistent_len();
    let total_bytes_per_buffer = river_persistent_len * std::mem::size_of::<f32>();
    eprintln!("Stage S3 (solver::new):             {}", fmt_dur(setup_solver_ms));
    eprintln!("  river_persistent_len:  {} f32 = {:.2} GB per buffer",
              river_persistent_len, total_bytes_per_buffer as f64 / 1e9);

    // Build disk paths and convert to DiskBacked. This writes 2× the full
    // buffer to disk as initial-zero state. RAII guards remove the files
    // on Drop (incl. panic), so a panicked test doesn't leak the multi-GB
    // files on macOS where temp_dir() is on the real SSD (not tmpfs).
    let regrets_guard = ScratchFile::new(std::env::temp_dir().join("e2e_cost_regrets.bin"));
    let cum_guard = ScratchFile::new(std::env::temp_dir().join("e2e_cost_cum.bin"));

    let t = Instant::now();
    let mut solver = solver.into_disk_backed(regrets_guard.path(), cum_guard.path())
        .expect("into_disk_backed");
    let into_disk_ms = t.elapsed().as_millis();
    eprintln!("Stage S4 (into_disk_backed):        {} ({:.2} GB × 2 written)",
              fmt_dur(into_disk_ms), total_bytes_per_buffer as f64 / 1e9);

    eprintln!("\n--- Setup total: {} ---",
              fmt_dur(setup_tree_ms + setup_table_ms + setup_solver_ms + into_disk_ms));

    // -------- ONE iter: end-to-end timing via production solver.run() --------
    eprintln!("\n--- Running 1 iter via solver.run() (production path) ---");
    let t = Instant::now();
    let root_cfv = solver.run(&tree, &game, 1);
    let iter_ms = t.elapsed().as_millis();
    eprintln!("\nStage R1 (solver.run(1 iter)):      {} ({:.2} min)",
              fmt_dur(iter_ms), iter_ms as f64 / 60_000.0);
    eprintln!("  root_cfv[0..4]: {:?}", root_cfv.iter().take(4).collect::<Vec<_>>());
    let any_nonzero = root_cfv.iter().any(|x| x.abs() > 1e-9);
    let any_nan = root_cfv.iter().any(|x| x.is_nan());
    eprintln!("  any non-zero: {}, any NaN: {}", any_nonzero, any_nan);
    assert!(!any_nan, "root_cfv contains NaN — solver state corrupt");

    // -------- I/O size accounting --------
    let n_turn = solver.n_turn_outcomes();
    let max_n_river = solver.max_river_outcomes();
    let pairs = n_turn * max_n_river;
    let stride_bytes = (river_persistent_len / pairs) * std::mem::size_of::<f32>();
    let traversers = np as usize;
    let io_per_iter_bytes =
        2 * traversers * pairs * stride_bytes * 2; // 2 buffers (regrets+cum), 2 ops (read+write)
    eprintln!("\nStage R2 (I/O accounting):");
    eprintln!("  pairs:                    {} ({} turn × {} max_river)", pairs, n_turn, max_n_river);
    eprintln!("  stride per pair:          {} bytes ({:.2} MB)",
              stride_bytes, stride_bytes as f64 / (1u64 << 20) as f64);
    let rw_per_pair_bytes = 2usize * stride_bytes * 2;
    eprintln!("  read+write per pair:      {} bytes ({:.2} MB)",
              rw_per_pair_bytes, rw_per_pair_bytes as f64 / (1u64 << 20) as f64);
    eprintln!("  per-iter I/O total:       {} bytes ({:.2} GB)",
              io_per_iter_bytes, io_per_iter_bytes as f64 / 1e9);
    let io_throughput = (io_per_iter_bytes as f64) / (iter_ms as f64 / 1000.0) / 1e9;
    eprintln!("  effective I/O throughput: {:.2} GB/s (over wall-clock; actual I/O may overlap with compute)",
              io_throughput);

    // -------- Final report --------
    eprintln!("\n========================================================================");
    eprintln!("=== End-to-end real-cost SUMMARY                                    ===");
    eprintln!("========================================================================");
    eprintln!("  Setup (tree + table + solver + into_disk):     {}",
              fmt_dur(setup_tree_ms + setup_table_ms + setup_solver_ms + into_disk_ms));
    eprintln!("  1 iter solver.run() wall-clock:                 {} ({:.2} min)",
              fmt_dur(iter_ms), iter_ms as f64 / 60_000.0);
    eprintln!("  Per-iter I/O budget:                            {:.2} GB", io_per_iter_bytes as f64 / 1e9);
    eprintln!();
    eprintln!("  This is THE honest end-to-end per-iter cost for full-nh unabstracted");
    eprintln!("  HU on CPU in DiskBacked mode. Includes compute + I/O + freeze + reach +");
    eprintln!("  per-pair regret_matching across both traversers. No projection adjustment.");
    eprintln!();
    eprintln!("  Determines whether A suffices for HU baseline:");
    eprintln!("    - per-iter < 1 hr  → full-nh HU is comfortably feasible");
    eprintln!("    - per-iter 1-3 hr  → feasible but tight; ~hundreds of iters → days/weeks");
    eprintln!("    - per-iter > 5 hr  → HU also needs C (abstraction) to be practical");

    // Cleanup: drop solver to close file handles, then ScratchFile guards
    // remove the files on their own Drop at end of function scope. The
    // guards also fire on panic, so a partial-iter crash doesn't leak.
    drop(solver);
    eprintln!("\n  (will clean up {:.0} GB of disk state via Drop guards)",
              2.0 * total_bytes_per_buffer as f64 / 1e9);
}
