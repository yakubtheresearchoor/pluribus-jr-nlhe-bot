// Step 2.B.2 end-to-end validation: GPU-InMemory vs GPU-DiskBacked
// bit-exact parity after N iters on the same small game.
//
// What this gates:
//   1. flop + turn regions of d_regrets, d_cum_strategy match BIT-EXACT
//      between modes — they're unchanged from before 2.B.2 landed.
//   2. EVERY (ti, ri) river slot matches BIT-EXACT between modes. In
//      InMemory, the slot lives in the in-buffer river region. In
//      DiskBacked, the slot lives in the persistence file at the
//      canonical offset. Same kernel, same arithmetic, same inputs →
//      same f32 bits regardless of where the result is stored.
//
// IMPORTANT SCOPE NOTE (per the lead's carry on 2.B.2):
//   This is a BEHAVIORAL-PRESERVATION gate, NOT a production-correctness
//   gate. It proves DiskBacked doesn't break InMemory's behavior. The
//   outstanding production-correctness gate is 2.A.2 (full-nh GPU vs CPU
//   parity on the production-API harness), which 2.B.2 enables (DiskBacked
//   makes production scale runnable) but does NOT substitute for.
//
// Why this game: 6p nh=6 with 2 turn cards × 2 river cards per turn = 4
// distinct (ti, ri) pairs. The pair-cycling mode in DiskBacked is
// non-trivially exercised (4 load → save cycles per traverser per iter).

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::{GpuRiverMode, MetalFlopStartSolver};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::io::{Read, Seek, SeekFrom};

fn build_6p_multi_pair_game() -> (FlatTree, FlopStartGame) {
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
    let tc_3c = card_from_str("3c").unwrap() as u8;
    let tc_4c = card_from_str("4c").unwrap() as u8;
    let turn_cards = vec![tc_3c, tc_4c];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[tc_3c as usize] = vec![
        card_from_str("5s").unwrap() as u8,
        card_from_str("6c").unwrap() as u8,
    ];
    river_decks[tc_4c as usize] = vec![
        card_from_str("8d").unwrap() as u8,
        card_from_str("9h").unwrap() as u8,
    ];

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

struct ScratchFile(std::path::PathBuf);
impl Drop for ScratchFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}

/// Read one (ti, ri) slot of `river_stride` f32 from a disk-backed file
/// at the canonical byte offset.
fn read_pair_from_file(
    path: &std::path::Path,
    ti: usize, ri: usize, max_river: usize, stride: usize,
) -> Vec<f32> {
    let off = ((ti * max_river + ri) * stride * std::mem::size_of::<f32>()) as u64;
    let mut f = std::fs::File::open(path).expect("open file for reading");
    f.seek(SeekFrom::Start(off)).expect("seek");
    let byte_len = stride * std::mem::size_of::<f32>();
    let mut buf = vec![0u8; byte_len];
    f.read_exact(&mut buf).expect("read_exact");
    let mut out = vec![0.0f32; stride];
    for i in 0..stride {
        out[i] = f32::from_le_bytes([
            buf[i*4], buf[i*4+1], buf[i*4+2], buf[i*4+3],
        ]);
    }
    out
}

/// GATE: GPU-InMemory and GPU-DiskBacked produce BIT-EXACT identical
/// state after N iters across flop, turn, AND every river (ti, ri) slot.
#[test]
fn disk_backed_gpu_bit_exact_to_in_memory_gpu_after_n_iters() {
    let (tree, game) = build_6p_multi_pair_game();
    let ctx = MetalContext::new().expect("Metal");
    let cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());

    // ────────────────────────────────────────────────────────────
    // Solver A: InMemory (control). Run N iters with run().
    // ────────────────────────────────────────────────────────────
    let mut gpu_inmem = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);
    assert!(matches!(gpu_inmem.gpu_river_mode(), GpuRiverMode::InMemory));

    let n_iters = 3;
    gpu_inmem.run(&ctx, &tree, &game, n_iters);

    // Download final InMemory state.
    let inmem_regrets = gpu_inmem.download_regrets();
    let inmem_cum = gpu_inmem.download_cum_strategy();

    // ────────────────────────────────────────────────────────────
    // Solver B: DiskBacked. Same initial state, same n_iters.
    // ────────────────────────────────────────────────────────────
    let mut gpu_disk = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);

    let regrets_guard = ScratchFile(std::env::temp_dir()
        .join("step2b2_parity_regrets.bin"));
    let cum_guard = ScratchFile(std::env::temp_dir()
        .join("step2b2_parity_cum.bin"));
    let _ = std::fs::remove_file(&regrets_guard.0);
    let _ = std::fs::remove_file(&cum_guard.0);
    // PRE-transition buffer sizes (will be checked against POST below).
    let pre_disk_regrets_len = gpu_disk.download_regrets().len();

    gpu_disk.into_disk_backed_gpu(&ctx, &regrets_guard.0, &cum_guard.0)
        .expect("into_disk_backed_gpu");
    assert!(matches!(gpu_disk.gpu_river_mode(), GpuRiverMode::DiskBacked { .. }));

    // ────────────────────────────────────────────────────────────
    // SHRINK gate (validates the memory optimization actually
    // happened on this multi-pair game where savings are non-zero).
    // ────────────────────────────────────────────────────────────
    let post_disk_regrets_len = gpu_disk.download_regrets().len();
    let river_offset_check = gpu_disk.river_offset();
    let river_stride_check = gpu_disk.river_stride();
    assert_eq!(post_disk_regrets_len, river_offset_check + river_stride_check,
        "POST-shrink regrets size = {} (expected {} = river_offset + river_stride)",
        post_disk_regrets_len, river_offset_check + river_stride_check);
    assert!(post_disk_regrets_len < pre_disk_regrets_len,
        "shrink failed to reduce buffer size: pre = {}, post = {} (multi-pair game; expected reduction)",
        pre_disk_regrets_len, post_disk_regrets_len);
    let saved_f32 = pre_disk_regrets_len - post_disk_regrets_len;
    eprintln!("2.B SHRINK gate: each of 3 buffers reduced by {} f32 ({} bytes); now downstream parity gate confirms correctness post-shrink",
        saved_f32, saved_f32 * 4);

    gpu_disk.run(&ctx, &tree, &game, n_iters);

    let disk_regrets = gpu_disk.download_regrets();
    let disk_cum = gpu_disk.download_cum_strategy();

    let max_river = gpu_disk.max_river();
    let n_turn = gpu_disk.n_turn();
    let stride = gpu_disk.river_stride();
    let river_offset = gpu_disk.river_offset();

    // ────────────────────────────────────────────────────────────
    // GATE 1: flop + turn regions bit-exact between modes.
    // (Both modes share the same kernel for flop+turn; their state
    // lives in [0..river_offset) of each buffer.)
    // ────────────────────────────────────────────────────────────
    let mut flop_turn_mismatches = 0usize;
    for i in 0..river_offset {
        if inmem_regrets[i].to_bits() != disk_regrets[i].to_bits() {
            if flop_turn_mismatches < 5 {
                eprintln!("regrets[{}] flop/turn mismatch: InMemory={} DiskBacked={}",
                    i, inmem_regrets[i], disk_regrets[i]);
            }
            flop_turn_mismatches += 1;
        }
        if inmem_cum[i].to_bits() != disk_cum[i].to_bits() {
            if flop_turn_mismatches < 5 {
                eprintln!("cum_strategy[{}] flop/turn mismatch: InMemory={} DiskBacked={}",
                    i, inmem_cum[i], disk_cum[i]);
            }
            flop_turn_mismatches += 1;
        }
    }
    assert_eq!(flop_turn_mismatches, 0,
        "{} bit-mismatches in flop+turn region — kernels for non-river zones drifted",
        flop_turn_mismatches);

    // ────────────────────────────────────────────────────────────
    // GATE 2: every (ti, ri) river slot bit-exact between modes.
    // InMemory: read from in-buffer at slot offset.
    // DiskBacked: read from file at canonical offset (in-buffer
    // scratch only holds the last-processed pair).
    // ────────────────────────────────────────────────────────────
    let mut total_river_mismatches = 0usize;
    let mut total_river_compared = 0usize;
    for ti in 0..n_turn {
        // Only the (ti, ri) pairs with ri < n_river for this ti carry data.
        // For pairs with ri >= n_river, both modes have zeros there
        // (InMemory was never written, DiskBacked file was seeded with
        // zeros at construction and never overwritten).
        let n_river = if ti == 0 { 2 } else { 2 };  // both turns have 2 rivers in this game
        for ri in 0..n_river {
            let inmem_slot_start = river_offset + (ti * max_river + ri) * stride;
            let inmem_regrets_slot = &inmem_regrets[inmem_slot_start..inmem_slot_start + stride];
            let inmem_cum_slot = &inmem_cum[inmem_slot_start..inmem_slot_start + stride];

            let disk_regrets_slot = read_pair_from_file(
                &regrets_guard.0, ti, ri, max_river, stride);
            let disk_cum_slot = read_pair_from_file(
                &cum_guard.0, ti, ri, max_river, stride);

            for i in 0..stride {
                total_river_compared += 1;
                if inmem_regrets_slot[i].to_bits() != disk_regrets_slot[i].to_bits() {
                    if total_river_mismatches < 5 {
                        eprintln!("regrets[ti={}, ri={}, off={}]: InMemory={} DiskBacked(file)={}",
                            ti, ri, i, inmem_regrets_slot[i], disk_regrets_slot[i]);
                    }
                    total_river_mismatches += 1;
                }
                if inmem_cum_slot[i].to_bits() != disk_cum_slot[i].to_bits() {
                    if total_river_mismatches < 5 {
                        eprintln!("cum[ti={}, ri={}, off={}]: InMemory={} DiskBacked(file)={}",
                            ti, ri, i, inmem_cum_slot[i], disk_cum_slot[i]);
                    }
                    total_river_mismatches += 1;
                }
            }
        }
    }
    assert_eq!(total_river_mismatches, 0,
        "{} bit-mismatches across {} river f32 entries — DiskBacked diverges from InMemory",
        total_river_mismatches, total_river_compared);

    eprintln!("2.B.2 PARITY PASS: {} iters, flop+turn ({} f32 × 2 buf) + river ({} f32 × 2 buf) BIT-EXACT",
        n_iters, river_offset, total_river_compared / 2);
}
