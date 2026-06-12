// Step 2.B.1 validation: DiskBacked GPU river persistence — file I/O
// round-trip foundation.
//
// What this gates:
//   1. into_disk_backed_gpu transitions the solver from InMemory to
//      DiskBacked and seeds the files with the current (zero) river
//      state, sized correctly.
//   2. save_river_pair_gpu writes the in-buffer river slot to the file
//      at the correct byte offset.
//   3. load_river_pair_gpu reads the file back into the in-buffer slot,
//      recovering bit-exact f32 values.
//   4. The buf-offset and file-offset math agree (writing pattern A to
//      slot (ti, ri), overwriting with pattern B, loading (ti, ri),
//      reading back gives A — not B, not zero, not corrupted).
//
// What this does NOT gate (deferred to 2.B.2):
//   - DiskBacked kernel dispatch (offsets stay full-buffer here).
//   - run_one_iter mode-aware flow.
//   - End-to-end bit-exact InMemory vs DiskBacked.
//
// Step 2.B.1 is the foundation: file ↔ GPU-buffer transport works
// at the byte level. 2.B.2 layers the integration on top.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::{GpuRiverMode, MetalFlopStartSolver};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// Same canonical 6p game as six_player_iter0_parity (post-audit).
fn build_6p_game() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 6u8;
    let nh = 6usize;

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();

    let mut ranges: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..num_players as usize {
        for &hi in &chosen {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
            ranges[p][pair_idx] = 1.0;
        }
    }
    let turn_cards = vec![card_from_str("3c").unwrap() as u8];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![card_from_str("5s").unwrap() as u8];

    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, num_players, &chosen, &turn_cards, &river_decks,
    );

    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![100; 6], initial_contributions: vec![5; 6],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = FlopStartGame::new(table);
    (tree, game)
}

/// RAII scratch file — removed on Drop (incl. panic). std::env::temp_dir
/// on macOS is on the real SSD, but the files are tiny (a few MB at most
/// on these small games) and removed promptly.
struct ScratchFile(std::path::PathBuf);
impl Drop for ScratchFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}

/// GATE: write pattern A → save → overwrite in-buffer with pattern B
/// → load → in-buffer must hold pattern A bit-exact.
#[test]
fn disk_backed_gpu_round_trips_river_pair_bit_exact() {
    let (tree, game) = build_6p_game();
    let ctx = MetalContext::new().expect("Metal");
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // PRE: solver is InMemory.
    assert!(matches!(gpu.gpu_river_mode(), GpuRiverMode::InMemory),
        "newly-built solver must default to InMemory");

    // RAII-guarded file paths.
    let regrets_guard = ScratchFile(std::env::temp_dir()
        .join("step2b1_gpu_regrets_river.bin"));
    let cum_guard = ScratchFile(std::env::temp_dir()
        .join("step2b1_gpu_cum_strategy_river.bin"));
    let _ = std::fs::remove_file(&regrets_guard.0);
    let _ = std::fs::remove_file(&cum_guard.0);

    // Transition to DiskBacked.
    gpu.into_disk_backed_gpu(&ctx, &regrets_guard.0, &cum_guard.0)
        .expect("into_disk_backed_gpu");
    assert!(matches!(gpu.gpu_river_mode(), GpuRiverMode::DiskBacked { .. }),
        "post-transition mode must be DiskBacked");

    // File should now exist and be sized for the full river state.
    let stride = gpu.river_stride();
    let n_turn = gpu.n_turn();
    let max_river = gpu.max_river();
    let expected_bytes = (n_turn * max_river * stride
        * std::mem::size_of::<f32>()) as u64;
    let regrets_size = std::fs::metadata(&regrets_guard.0).unwrap().len();
    let cum_size = std::fs::metadata(&cum_guard.0).unwrap().len();
    assert_eq!(regrets_size, expected_bytes,
        "regrets file size: got {} expected {}", regrets_size, expected_bytes);
    assert_eq!(cum_size, expected_bytes,
        "cum_strategy file size: got {} expected {}", cum_size, expected_bytes);

    // The slot under test: (ti=0, ri=0) — the only non-empty slot in
    // this single-turn × single-river game.
    let ti = 0usize;
    let ri = 0usize;
    let buf_off = gpu.river_offset() + (ti * max_river + ri) * stride;

    // Pattern A: distinct ramps for regrets (positive) and cum (negative).
    let regrets_a: Vec<f32> = (0..stride).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let cum_a: Vec<f32> = (0..stride).map(|i| (i as f32) * -0.25 - 2.0).collect();

    // Build full-buffer image by downloading, splicing in pattern A,
    // and uploading. Other slots stay zero.
    let mut regrets_image = gpu.download_regrets();
    let mut cum_image = gpu.download_cum_strategy();
    for i in 0..stride {
        regrets_image[buf_off + i] = regrets_a[i];
        cum_image[buf_off + i] = cum_a[i];
    }
    gpu.upload_regrets(&regrets_image);
    gpu.upload_cum_strategy(&cum_image);

    // Save A to disk.
    gpu.save_river_pair_gpu(ti, ri).expect("save_river_pair_gpu");

    // Overwrite in-buffer slot with pattern B.
    let regrets_b: Vec<f32> = (0..stride).map(|i| -(i as f32) * 0.75 + 99.0).collect();
    let cum_b: Vec<f32> = (0..stride).map(|i| (i as f32) * 0.125 + 50.0).collect();
    let mut regrets_image_b = gpu.download_regrets();
    let mut cum_image_b = gpu.download_cum_strategy();
    for i in 0..stride {
        regrets_image_b[buf_off + i] = regrets_b[i];
        cum_image_b[buf_off + i] = cum_b[i];
    }
    gpu.upload_regrets(&regrets_image_b);
    gpu.upload_cum_strategy(&cum_image_b);

    // Sanity: in-buffer now holds B (not A).
    {
        let r = gpu.download_regrets();
        let c = gpu.download_cum_strategy();
        assert_eq!(r[buf_off].to_bits(), regrets_b[0].to_bits(),
            "in-buffer regrets should hold pattern B after overwrite");
        assert_eq!(c[buf_off].to_bits(), cum_b[0].to_bits(),
            "in-buffer cum should hold pattern B after overwrite");
    }

    // Load from disk — should restore pattern A.
    gpu.load_river_pair_gpu(ti, ri).expect("load_river_pair_gpu");

    // Verify in-buffer slot now holds A bit-exact.
    let r = gpu.download_regrets();
    let c = gpu.download_cum_strategy();
    for i in 0..stride {
        assert_eq!(r[buf_off + i].to_bits(), regrets_a[i].to_bits(),
            "regrets[{}] not recovered: got {} expected {}",
            i, r[buf_off + i], regrets_a[i]);
        assert_eq!(c[buf_off + i].to_bits(), cum_a[i].to_bits(),
            "cum_strategy[{}] not recovered: got {} expected {}",
            i, c[buf_off + i], cum_a[i]);
    }

    eprintln!("2.B.1 round-trip PASS: {} f32 entries × 2 buffers recovered bit-exact",
        stride);
}

/// SANITY: in InMemory mode, load and save are no-ops.
/// (Both save and load just return Ok without touching the buffer.)
#[test]
fn in_memory_mode_load_save_are_noops() {
    let (tree, game) = build_6p_game();
    let ctx = MetalContext::new().expect("Metal");
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Default mode = InMemory.
    assert!(matches!(gpu.gpu_river_mode(), GpuRiverMode::InMemory));

    // Set a known pattern in the buffer.
    let mut regrets_image = gpu.download_regrets();
    let buf_off = gpu.river_offset();
    regrets_image[buf_off] = 42.0;
    gpu.upload_regrets(&regrets_image);

    // load and save should both succeed without touching anything.
    gpu.load_river_pair_gpu(0, 0).expect("load (no-op) should succeed");
    gpu.save_river_pair_gpu(0, 0).expect("save (no-op) should succeed");

    // Buffer should still hold 42.0.
    let r = gpu.download_regrets();
    assert_eq!(r[buf_off].to_bits(), 42.0_f32.to_bits(),
        "InMemory load/save must be no-ops; got {}", r[buf_off]);
}

/// 2.B SHRINK GATE: `into_disk_backed_gpu` must SHRINK the GPU buffers
/// from `flop_stride + turn_total + river_total` to
/// `flop_stride + turn_total + river_stride` (one scratch slot for
/// river). This is the load-bearing memory optimization for production
/// scale: at HU OptB nh=1176 the river region drops from 175 GB to
/// 76 MB per buffer.
///
/// This test gates the OPTIMIZATION itself (that the shrink happened),
/// not just that the code still works post-shrink. The bit-exact parity
/// test gates correctness under shrink.
#[test]
fn into_disk_backed_gpu_shrinks_buffers_to_scratch_size() {
    let (tree, game) = build_6p_game();
    let ctx = MetalContext::new().expect("Metal");
    let cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // PRE-transition: InMemory; buffers sized for full river region.
    let pre_regrets_len = gpu.download_regrets().len();
    let pre_cum_len = gpu.download_cum_strategy().len();
    let pre_strategy_len = gpu.download_strategy().len();
    let river_offset = gpu.river_offset();
    let river_stride = gpu.river_stride();
    let n_turn = gpu.n_turn();
    let max_river = gpu.max_river();
    let expected_pre = river_offset + n_turn * max_river * river_stride;
    assert_eq!(pre_regrets_len, expected_pre,
        "PRE regrets buffer size = {} (expected {})", pre_regrets_len, expected_pre);
    assert_eq!(pre_cum_len, expected_pre);
    assert_eq!(pre_strategy_len, expected_pre);

    // Transition.
    let regrets_guard = ScratchFile(std::env::temp_dir()
        .join("step2b_shrink_regrets.bin"));
    let cum_guard = ScratchFile(std::env::temp_dir()
        .join("step2b_shrink_cum.bin"));
    let _ = std::fs::remove_file(&regrets_guard.0);
    let _ = std::fs::remove_file(&cum_guard.0);
    gpu.into_disk_backed_gpu(&ctx, &regrets_guard.0, &cum_guard.0)
        .expect("into_disk_backed_gpu");

    // POST-transition: DiskBacked; buffers shrunk to one scratch slot.
    let post_regrets_len = gpu.download_regrets().len();
    let post_cum_len = gpu.download_cum_strategy().len();
    let post_strategy_len = gpu.download_strategy().len();
    let expected_post = river_offset + river_stride;
    assert_eq!(post_regrets_len, expected_post,
        "POST regrets buffer size = {} (expected {} = river_offset + river_stride)",
        post_regrets_len, expected_post);
    assert_eq!(post_cum_len, expected_post);
    assert_eq!(post_strategy_len, expected_post);

    // Sanity: the shrink actually reduced size (or at least didn't grow).
    assert!(post_regrets_len < pre_regrets_len || n_turn * max_river <= 1,
        "shrink should reduce buffer size: pre={} post={} (n_turn*max_river={})",
        pre_regrets_len, post_regrets_len, n_turn * max_river);

    let saved_per_buffer = (pre_regrets_len - post_regrets_len) * 4;
    eprintln!("2.B SHRINK: each of 3 buffers reduced by {} bytes ({} f32 entries)",
        saved_per_buffer, pre_regrets_len - post_regrets_len);
}
