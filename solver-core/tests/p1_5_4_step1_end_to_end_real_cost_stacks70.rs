// Second data point for the end-to-end cost projection (#59 follow-up).
//
// The stacks=50 measurement gave 41.45 min/iter for a 5453-node tree.
// Projecting to production stacks=97 (14213 nodes) by linear scaling
// gave ~108 min/iter. That's a SINGLE-POINT linear extrapolation — the
// same pattern that has been wrong repeatedly in this project (the
// per-board single-point projection was 4× off when the variance probe
// added a second sample).
//
// This test runs the same end-to-end measurement at stacks=70, giving a
// second data point. From {stacks=50 → 41.45 min, stacks=70 → X min} we
// can compute the actual scaling exponent:
//
//   cost(stacks) ∝ nodes(stacks)^β
//
// β = 1: linear (the projection is good)
// β > 1: super-linear (HU production is tighter than projected)
// β < 1: sub-linear (HU production is more comfortable than projected)
//
// Disk: stacks=70 expected to need ~200 GB on disk (vs 140 GB at stacks=50).
// Available: ~274 GB after the stacks=50 cleanup.

use std::io::{Seek, SeekFrom, Write};
use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

/// RAII guard: removes the file on Drop (incl. panic). On macOS
/// std::env::temp_dir() is on the real SSD (not tmpfs).
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

fn fmt_dur(ms: u128) -> String {
    if ms < 1000 { format!("{} ms", ms) }
    else if ms < 60_000 { format!("{:.2} s", ms as f64 / 1000.0) }
    else { format!("{:.2} min", ms as f64 / 60_000.0) }
}

#[test]
#[ignore = "End-to-end second data point at stacks=70 (~1.5 hr wall-clock). Run on demand."]
fn end_to_end_real_cost_hu_optb_stacks70() {
    eprintln!("\n========================================================================");
    eprintln!("=== End-to-end real-cost: SECOND DATA POINT (stacks=70)              ===");
    eprintln!("===   Verifies the stacks=50 → stacks=97 linear-scaling projection   ===");
    eprintln!("========================================================================\n");

    let np = 2u8;
    let cfg = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 6,
        starting_stacks: vec![70; np as usize],
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
    let tree = build_tree(&cfg).expect("HU OptB stacks=70 tree");
    eprintln!("Tree build: {} ({} nodes)", fmt_dur(t.elapsed().as_millis()), tree.num_nodes());

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
    eprintln!("FlopChanceTable: {} (nh = {})", fmt_dur(setup_table_ms), nh);

    let game = FlopStartGame::new(table);
    let solver = FlopStartVectorCfr::new(&tree, game.table());
    let river_persistent_len = solver.river_persistent_len();
    let total_bytes = river_persistent_len * std::mem::size_of::<f32>();
    eprintln!("river_persistent_len: {} f32 = {:.2} GB per buffer ({:.2} GB total disk needed)",
              river_persistent_len, total_bytes as f64 / 1e9, 2.0 * total_bytes as f64 / 1e9);

    // Disk-space pre-flight: refuse to start if not enough free space, so
    // we don't fill the SSD and panic mid-write.
    let needed_bytes = 2 * total_bytes as u64 + (5 * (1u64 << 30)); // 5 GB safety buffer
    let stats = std::fs::metadata(std::env::temp_dir()).ok();
    if let Some(_meta) = stats {
        // Best-effort free-space check via df. If it fails, just proceed.
        if let Ok(df_out) = std::process::Command::new("df").arg("-k").arg("/").output() {
            if let Ok(s) = String::from_utf8(df_out.stdout) {
                if let Some(line) = s.lines().nth(1) {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 4 {
                        if let Ok(avail_kb) = parts[3].parse::<u64>() {
                            let avail_bytes = avail_kb * 1024;
                            eprintln!("Free disk: {:.2} GB. Need: {:.2} GB (incl 5 GB safety).",
                                      avail_bytes as f64 / 1e9, needed_bytes as f64 / 1e9);
                            if avail_bytes < needed_bytes {
                                panic!("INSUFFICIENT DISK: {:.2} GB free, need {:.2} GB. \
                                        Free up space or reduce stacks to a feasible size.",
                                       avail_bytes as f64 / 1e9, needed_bytes as f64 / 1e9);
                            }
                        }
                    }
                }
            }
        }
    }

    let regrets_guard = ScratchFile::new(std::env::temp_dir().join("e2e_s70_regrets.bin"));
    let cum_guard = ScratchFile::new(std::env::temp_dir().join("e2e_s70_cum.bin"));

    let t = Instant::now();
    let mut solver = solver.into_disk_backed(regrets_guard.path(), cum_guard.path())
        .expect("into_disk_backed");
    eprintln!("into_disk_backed: {} (wrote {:.2} GB × 2)",
              fmt_dur(t.elapsed().as_millis()), total_bytes as f64 / 1e9);

    eprintln!("\n--- Running 1 iter via solver.run() ---");
    let t = Instant::now();
    let root_cfv = solver.run(&tree, &game, 1);
    let iter_ms = t.elapsed().as_millis();
    eprintln!("solver.run(1 iter): {} ({:.2} min)", fmt_dur(iter_ms), iter_ms as f64 / 60_000.0);

    let any_nan = root_cfv.iter().any(|x| x.is_nan());
    eprintln!("root_cfv[0..4]: {:?}, any NaN: {}",
              root_cfv.iter().take(4).collect::<Vec<_>>(), any_nan);
    assert!(!any_nan, "root_cfv contains NaN");

    // -------- Scaling analysis vs the stacks=50 baseline --------
    eprintln!("\n========================================================================");
    eprintln!("=== Scaling analysis (vs stacks=50 baseline = 41.45 min, 5453 nodes) ===");
    eprintln!("========================================================================");
    let nodes_50 = 5453.0;
    let cost_50 = 41.45 * 60_000.0; // ms

    let nodes_70 = tree.num_nodes() as f64;
    let cost_70 = iter_ms as f64;

    let node_ratio = nodes_70 / nodes_50;
    let cost_ratio = cost_70 / cost_50;
    let beta = cost_ratio.ln() / node_ratio.ln();   // cost ∝ nodes^β

    eprintln!("  stacks=50: {} nodes, {} min/iter", nodes_50 as u32, 41.45);
    eprintln!("  stacks=70: {} nodes, {:.2} min/iter", nodes_70 as u32, cost_70 / 60_000.0);
    eprintln!("  node_ratio (70/50):  {:.3}", node_ratio);
    eprintln!("  cost_ratio (70/50):  {:.3}", cost_ratio);
    eprintln!("  scaling β (cost ∝ nodes^β): {:.3}", beta);
    eprintln!();
    if (beta - 1.0).abs() < 0.10 {
        eprintln!("  → β ≈ 1.0 (linear): the single-point projection was reliable.");
    } else if beta > 1.0 {
        eprintln!("  → β > 1 (super-linear): HU production is TIGHTER than the linear projection.");
    } else {
        eprintln!("  → β < 1 (sub-linear): HU production is MORE COMFORTABLE than projected.");
    }

    // -------- Projection to production stacks=97 with the measured β --------
    let nodes_97 = 14213.0;
    let projected_ms_linear = cost_50 * (nodes_97 / nodes_50);
    let projected_ms_measured = cost_50 * (nodes_97 / nodes_50).powf(beta);
    eprintln!("\n  Projection to production stacks=97 ({} nodes):", nodes_97 as u32);
    eprintln!("    LINEAR projection (β=1):      {:.1} min ({:.2} hr)",
              projected_ms_linear / 60_000.0, projected_ms_linear / 3_600_000.0);
    eprintln!("    MEASURED-β projection (β={:.2}): {:.1} min ({:.2} hr)",
              beta, projected_ms_measured / 60_000.0, projected_ms_measured / 3_600_000.0);

    eprintln!("\n  Cleanup: ScratchFile guards drop at function end.");
    drop(solver);
}
