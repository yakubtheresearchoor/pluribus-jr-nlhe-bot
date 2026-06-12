// A parity gate at full HU pair geometry, sized to exceed 2 GB file offsets.
// Targeted at the failure modes the nh=100/4-pair test cannot reach:
//
//   1. Integer-overflow in offset computation past 2 GB (if any intermediate
//      were 32-bit; ruled out by code inspection but verified empirically here
//      by reaching file offsets > 2 GB and confirming correct read/write).
//
//   2. Large-seek + partial-read I/O edge cases at multi-GB file sizes.
//
//   3. Pair-index offset bugs that 4 pairs cannot exercise (e.g. swapped
//      tc/rc, mixed-up max_n_river vs n_turn).
//
// Test structure:
//   A. Geometry sanity: print n_turn, max_n_river, river_stride, total file
//      size, and (last_pair_index × stride_bytes) to confirm the test does
//      reach offsets > 2 GB.
//
//   B. Sentinel offset-isolation: directly write a unique f32 to a specific
//      pair's first slot, load DIFFERENT pairs, confirm the sentinel is
//      ONLY visible when loading that pair. Pinpoints offset bugs separately
//      from full parity divergence (the sentinel is a structured oracle).
//
//   C. Full-geometry parity: run 1 iter on the InMemory solver, capture
//      regrets_river + cum_strategy_river. Run 1 iter on the DiskBacked
//      solver, read both files back from disk. Bit-exact compare.
//
// nh and tree sizing: we use HU 1+1 with small stacks to keep river_count
// modest while still giving large enough river_stride at nh=1176 to push
// the total file past 2 GB. The compute_flop_start API constructs the full
// 49 turn × 47-48 river chance table for us — no manual 100+ line helper.

use std::io::{Read, Seek, SeekFrom, Write};
use std::time::Instant;

/// RAII guard: removes the file on Drop (incl. panic). On macOS,
/// std::env::temp_dir() is on the real SSD (not tmpfs), so leaks of
/// multi-GB test files are real disk loss until manual cleanup.
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
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn build_full_geometry_game(
    bet_options: BetSizeOptions,
    stacks: i32,
    pot: i32,
) -> (solver_core::tree::flat::FlatTree, FlopStartGame) {
    let np = 2u8;
    let cfg = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: pot,
        starting_stacks: vec![stacks; np as usize],
        initial_contributions: vec![0; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: bet_options,
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("flop tree builds");

    // Full deck via compute_flop_start. nh ends up = 1176 at HU on this board.
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
    let game = FlopStartGame::new(table);
    (tree, game)
}

#[test]
#[ignore = "A full-geometry parity gate: ~minutes wall-clock at HU 1+1 stacks=20 nh=1176. Run on demand."]
fn a_full_geometry_parity_offsets_exceed_2gb() {
    let bet_options = BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: vec![BetSize::PotRelative(1.0)],   // 1+1 to push river_count up
    };
    let (tree, game) = build_full_geometry_game(bet_options, 20, 4);
    let table = game.table();
    let nh = table.num_valid;
    let nn = tree.num_nodes();
    let n_turn = table.remaining_deck.len();
    let max_n_river = table.remaining_deck.iter()
        .map(|&tc| table.river_decks[tc as usize].len())
        .max().unwrap();
    let n_pairs = n_turn * max_n_river;
    eprintln!("\n=== A full-geometry parity gate ===");
    eprintln!("  tree: {} nodes, nh = {}, n_turn = {}, max_n_river = {}, pairs = {}",
              nn, nh, n_turn, max_n_river, n_pairs);

    // -------- A. Geometry sanity --------
    let solver = FlopStartVectorCfr::new(&tree, table);
    let river_persistent_len = solver.river_persistent_len();
    let total_bytes = river_persistent_len * std::mem::size_of::<f32>();
    let last_pair_offset = ((n_turn * max_n_river - 1) as u64)
        * (river_persistent_len / n_pairs * std::mem::size_of::<f32>()) as u64;

    eprintln!("\n  --- A. Geometry sanity ---");
    eprintln!("  river_persistent_len:  {} f32", river_persistent_len);
    eprintln!("  total file size:       {} bytes ({:.2} GB)",
              total_bytes, total_bytes as f64 / 1e9);
    eprintln!("  last_pair file offset: {} bytes ({:.2} GB)",
              last_pair_offset, last_pair_offset as f64 / 1e9);
    assert!(last_pair_offset > (1u64 << 31),
        "TEST PRECONDITION FAILED: last_pair_offset {} <= 2 GB. Increase nh or river_count \
         (use larger tree / richer abstraction) so offsets exceed 2 GB and the test can \
         actually exercise the >2 GB seek path.",
        last_pair_offset);
    eprintln!("  ✓ last_pair_offset > 2 GB — large-seek path is exercised");

    let regrets_guard = ScratchFile::new(std::env::temp_dir().join("a_full_geom_regrets.bin"));
    let cum_guard = ScratchFile::new(std::env::temp_dir().join("a_full_geom_cum.bin"));
    let regrets_path = regrets_guard.path().to_path_buf();
    let cum_path = cum_guard.path().to_path_buf();

    // -------- B. Sentinel offset-isolation --------
    eprintln!("\n  --- B. Sentinel offset-isolation ---");
    {
        let sentinel = f32::from_bits(0xC0FFEE42); // distinct, easy-to-spot

        // Create a DiskBacked solver. After into_disk_backed, both files exist
        // and are filled with zeros (initial regrets/cum_strategy state).
        let mut solver = FlopStartVectorCfr::new(&tree, table)
            .into_disk_backed(&regrets_path, &cum_path)
            .expect("into_disk_backed");

        // Pick the LAST pair (highest offset). Compute the byte offset by
        // recomputing the same arithmetic as solver internally does (we can't
        // call the private method, so we recompute here and compare against
        // what load actually reads).
        let n_pairs_minus_1 = n_pairs - 1;
        let last_tc = n_pairs_minus_1 / max_n_river;
        let last_rc = n_pairs_minus_1 % max_n_river;
        let stride_f32 = river_persistent_len / n_pairs;
        let last_pair_byte_offset = (n_pairs_minus_1 as u64)
            * (stride_f32 * std::mem::size_of::<f32>()) as u64;

        eprintln!("  sentinel = {:#x} written at last pair (tc={}, rc={})",
                  sentinel.to_bits(), last_tc, last_rc);
        eprintln!("  byte offset of last pair: {} ({:.2} GB)",
                  last_pair_byte_offset, last_pair_byte_offset as f64 / 1e9);

        // Write the sentinel as the first f32 of the last pair's slot in
        // the regrets file.
        {
            let mut f = std::fs::OpenOptions::new()
                .read(true).write(true).open(&regrets_path)
                .expect("reopen regrets for sentinel write");
            f.seek(SeekFrom::Start(last_pair_byte_offset)).expect("seek");
            f.write_all(&sentinel.to_le_bytes()).expect("write sentinel");
            f.flush().expect("flush");
        }

        // Load (tc=0, rc=0) — sentinel must NOT appear in the scratch.
        solver.load_river_pair(0, 0).expect("load (0,0)");
        let regrets_at_first_pair = solver.regrets_river()[0];
        assert_eq!(regrets_at_first_pair.to_bits(), 0u32,
            "Sentinel leak: loaded (tc=0, rc=0) but regrets_river[0] = {:#x} \
             (expected 0). Offset computation is collapsing to the last pair's slot.",
            regrets_at_first_pair.to_bits());
        eprintln!("  ✓ load(tc=0, rc=0): regrets_river[0] = 0 (sentinel not leaked)");

        // Load (last_tc, last_rc) — sentinel MUST appear in the scratch.
        solver.load_river_pair(last_tc, last_rc).expect("load last");
        let regrets_at_last_pair = solver.regrets_river()[0];
        assert_eq!(regrets_at_last_pair.to_bits(), sentinel.to_bits(),
            "Sentinel not found: loaded (tc={}, rc={}) but regrets_river[0] = {:#x} \
             (expected {:#x}). Offset computation is wrong for the last pair.",
            last_tc, last_rc, regrets_at_last_pair.to_bits(), sentinel.to_bits());
        eprintln!("  ✓ load(tc={}, rc={}): regrets_river[0] = {:#x} (sentinel correctly read)",
                  last_tc, last_rc, regrets_at_last_pair.to_bits());

        // Also probe a middle pair: it should NOT see the sentinel.
        let mid = n_pairs / 2;
        let mid_tc = mid / max_n_river;
        let mid_rc = mid % max_n_river;
        solver.load_river_pair(mid_tc, mid_rc).expect("load mid");
        let regrets_at_mid = solver.regrets_river()[0];
        assert_eq!(regrets_at_mid.to_bits(), 0u32,
            "Sentinel leak at middle pair: load(tc={}, rc={}) saw {:#x} (expected 0).",
            mid_tc, mid_rc, regrets_at_mid.to_bits());
        eprintln!("  ✓ load(tc={}, rc={}) mid: regrets_river[0] = 0 (no sentinel leak)",
                  mid_tc, mid_rc);

        // Cleanup the corrupted disk state before C.
        drop(solver);
        let _ = std::fs::remove_file(&regrets_path);
        let _ = std::fs::remove_file(&cum_path);
    }

    // -------- C. Full-geometry parity (1 iter) --------
    eprintln!("\n  --- C. Full-geometry parity (1 iter) ---");

    let t0 = Instant::now();
    let mut mem_solver = FlopStartVectorCfr::new(&tree, table);
    let _ = mem_solver.run(&tree, &game, 1);
    let mem_regrets = mem_solver.regrets_river().to_vec();
    let mem_cum = mem_solver.cum_strategy_river().to_vec();
    let t_mem = t0.elapsed().as_millis();
    eprintln!("  InMemory   1 iter: {} ms ({:.2} s) — {} f32 captured per buffer",
              t_mem, t_mem as f64 / 1000.0, mem_regrets.len());

    let t0 = Instant::now();
    let disk_solver = FlopStartVectorCfr::new(&tree, table)
        .into_disk_backed(&regrets_path, &cum_path)
        .expect("into_disk_backed");
    let mut disk_solver = disk_solver;
    let _ = disk_solver.run(&tree, &game, 1);
    let t_disk = t0.elapsed().as_millis();
    eprintln!("  DiskBacked 1 iter: {} ms ({:.2} s)", t_disk, t_disk as f64 / 1000.0);
    eprintln!("  DiskBacked / InMemory ratio: {:.2}×", t_disk as f64 / t_mem as f64);

    // Read full state back from files.
    let mut disk_regrets = vec![0.0f32; mem_regrets.len()];
    let mut disk_cum = vec![0.0f32; mem_cum.len()];
    {
        let mut f = std::fs::OpenOptions::new().read(true).open(&regrets_path).unwrap();
        f.seek(SeekFrom::Start(0)).unwrap();
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                disk_regrets.as_mut_ptr() as *mut u8,
                disk_regrets.len() * std::mem::size_of::<f32>(),
            )
        };
        f.read_exact(bytes).unwrap();
    }
    {
        let mut f = std::fs::OpenOptions::new().read(true).open(&cum_path).unwrap();
        f.seek(SeekFrom::Start(0)).unwrap();
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                disk_cum.as_mut_ptr() as *mut u8,
                disk_cum.len() * std::mem::size_of::<f32>(),
            )
        };
        f.read_exact(bytes).unwrap();
    }

    // Bit-exact compare.
    let mut reg_diffs = 0usize;
    let mut max_reg_diff = 0.0f32;
    for i in 0..mem_regrets.len() {
        if mem_regrets[i].to_bits() != disk_regrets[i].to_bits() {
            reg_diffs += 1;
            let d = (mem_regrets[i] - disk_regrets[i]).abs();
            if d > max_reg_diff { max_reg_diff = d; }
        }
    }
    let mut cum_diffs = 0usize;
    let mut max_cum_diff = 0.0f32;
    for i in 0..mem_cum.len() {
        if mem_cum[i].to_bits() != disk_cum[i].to_bits() {
            cum_diffs += 1;
            let d = (mem_cum[i] - disk_cum[i]).abs();
            if d > max_cum_diff { max_cum_diff = d; }
        }
    }
    eprintln!("\n  regrets:      {} / {} diffs (max abs = {:.6e})",
              reg_diffs, mem_regrets.len(), max_reg_diff);
    eprintln!("  cum_strategy: {} / {} diffs (max abs = {:.6e})",
              cum_diffs, mem_cum.len(), max_cum_diff);

    // Files cleaned by ScratchFile guards on Drop (incl. panic from asserts).
    assert_eq!(reg_diffs, 0, "regrets bit-exact match failed");
    assert_eq!(cum_diffs, 0, "cum_strategy bit-exact match failed");

    eprintln!("\n=== A full-geometry parity gate PASS ===");
    eprintln!("  - Sentinel offset-isolation: 3 probes (first, mid, last) correct");
    eprintln!("  - File offsets reach {:.2} GB (> 2 GB)", last_pair_offset as f64 / 1e9);
    eprintln!("  - InMemory vs DiskBacked: bit-identical across {} f32 in each buffer",
              mem_regrets.len());
}
