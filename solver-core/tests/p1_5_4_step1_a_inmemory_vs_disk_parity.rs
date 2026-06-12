// A parity gate: bit-exact InMemory vs DiskBacked at the largest
// in-memory-feasible nh (the lead's discipline: small-scale passes must not
// be assumed to cover large-scale; the on-disk offset math could be right
// at nh=4 and wrong at nh=200 due to stride or offset overflow).
//
// Test structure:
//   1. Build a tiny HU flop tree at restricted ranges that yield a moderate
//      nh (using a custom FlopChanceTable with manually-chosen hands).
//   2. Run solver.run(N) in InMemory mode, capture final regrets_river and
//      cum_strategy_river contents (the full per-(tc, rc) buffers).
//   3. Run the SAME setup in DiskBacked mode (same tree, same table, same
//      iter count), read the persisted regrets and cum_strategy back from
//      disk, compare bit-exact f32 against the InMemory results.
//   4. Repeat for N ∈ {1, 3, 10} to catch any per-iter accumulation drift
//      between modes (the subtle failure mode where modes match at iter 1
//      but diverge by iter 10 due to a precision or ordering difference).
//
// Failure criteria:
//   - Any single f32 differs between InMemory and DiskBacked → file offset
//     math is wrong, or load/save corrupts data.
//   - Difference grows with N → iter-ordering or accumulation discipline
//     differs between modes (the most insidious failure: looks fine at
//     short runs, wrong at long runs).
//
// nh choice: pushed to ~100 by using a custom chance table with 100 hands.
// At nh=100, river_stride = river_count × 4 × 100 ≈ small_river_count × 400
// f32 per pair. n_turn × max_n_river × river_stride is comfortably in-RAM.

use solver_core::card::{card_from_str, index_to_card_pair, Card};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn find_pair(c1: Card, c2: Card) -> u16 {
    use solver_core::card::NUM_POSSIBLE_HANDS;
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (a, b) = index_to_card_pair(idx);
        if (a == c1 as u8 && b == c2 as u8) || (a == c2 as u8 && b == c1 as u8) {
            return idx as u16;
        }
    }
    panic!("pair not found");
}

/// Build a custom FlopChanceTable with `nh_target` hands (or as many as
/// fit non-blocking on this flop). Returns (tree, game) with nh ≈ nh_target.
fn build_custom_nh_game(nh_target: usize) -> (solver_core::tree::flat::FlatTree, FlopStartGame) {
    // Board: 2h 7d Ks (same as convergence_audit).
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter()
        .map(|s| card_from_str(s).unwrap()).collect();
    let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
    let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

    // Pick the first `nh_target` non-blocking hand indices.
    use solver_core::card::NUM_POSSIBLE_HANDS;
    let mut chosen: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS as u16 {
        if chosen.len() >= nh_target { break; }
        let (c1, c2) = index_to_card_pair(idx as usize);
        if board_mask & (1u64 << c1) != 0 { continue; }
        if board_mask & (1u64 << c2) != 0 { continue; }
        chosen.push(idx);
    }
    let nh = chosen.len();
    let num_players = 2u8;
    let num_opp = 1usize;

    // hand_cards layout: 2 bytes per hand.
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
    }

    // hand_ranks_base + valid_hand_indices.
    let valid_hand_indices = chosen.clone();
    let num_valid = nh;

    // conflict matrix: 1 if hands share any card, 0 otherwise.
    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh {
        let (c1, c2) = index_to_card_pair(chosen[i] as usize);
        for j in 0..nh {
            if i == j { continue; }
            let (c3, c4) = index_to_card_pair(chosen[j] as usize);
            if c1 == c3 || c1 == c4 || c2 == c3 || c2 == c4 {
                conflict[i * nh + j] = 1;
            }
        }
    }

    // Two turn cards × two river cards each (4 boards total).
    let turn_cards: Vec<u8> = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("4c").unwrap() as u8,
    ];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("5c").unwrap() as u8,
        card_from_str("6c").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("5c").unwrap() as u8,
    ];

    // Compute turn and river ranks + sorted arrays.
    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 { continue; }
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in &board_set { hand = hand.add_card(bc as usize); }
            hand = hand.add_card(tc as usize);
            turn_ranks[tc as usize * nh + i] = hand.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh)
            .filter(|&h| {
                let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
                turn_mask & (1u64 << c1) == 0 && turn_mask & (1u64 << c2) == 0
            })
            .map(|h| (turn_ranks[tc as usize * nh + h] + 1, h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            for (si, &(str, idx)) in items.iter().enumerate() {
                turn_sorted_str[tc as usize * num_opp * nh + oi * nh + si] = str;
                turn_sorted_idx[tc as usize * num_opp * nh + oi * nh + si] = idx;
            }
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        for &rc in &river_decks[tc as usize] {
            let river_mask = board_mask | (1u64 << tc) | (1u64 << rc);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if river_mask & (1u64 << c1) != 0 || river_mask & (1u64 << c2) != 0 { continue; }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in &board_set { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize);
                hand = hand.add_card(rc as usize);
                river_ranks[tc as usize * 52 * nh + rc as usize * nh + i] = hand.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .filter(|&h| {
                    let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
                    river_mask & (1u64 << c1) == 0 && river_mask & (1u64 << c2) == 0
                })
                .map(|h| (river_ranks[tc as usize * 52 * nh + rc as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                for (si, &(str, idx)) in items.iter().enumerate() {
                    river_sorted_str[tc as usize * 52 * num_opp * nh + rc as usize * num_opp * nh + oi * nh + si] = str;
                    river_sorted_idx[tc as usize * 52 * num_opp * nh + rc as usize * num_opp * nh + oi * nh + si] = idx;
                }
            }
        }
    }

    let initial_weights: Vec<Vec<f32>> = (0..num_players).map(|_| {
        let mut w = vec![0.0f32; nh];
        for h in 0..nh {
            let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
            let mut blocked = 0i32;
            for h2 in 0..nh {
                if h2 == h { continue; }
                let (c3, c4) = index_to_card_pair(valid_hand_indices[h2] as usize);
                if c1 == c3 || c1 == c4 || c2 == c3 || c2 == c4 { blocked += 1; }
            }
            w[h] = if blocked < (nh as i32 - 1) { 1.0 } else { 0.0 };
        }
        w
    }).collect();
    let num_combinations = initial_weights[0].iter().sum::<f32>() * initial_weights[1].iter().sum::<f32>();

    let hand_ranks_base = vec![0u16; nh];

    let table = FlopChanceTable {
        hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
        remaining_deck: turn_cards, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx, initial_weights, num_players,
        num_combinations: num_combinations as f64, river_decks,
    };

    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = FlopStartGame::new(table);
    (tree, game)
}

fn run_inmemory_capture(
    tree: &solver_core::tree::flat::FlatTree,
    game: &FlopStartGame,
    n_iters: u32,
) -> (Vec<f32>, Vec<f32>) {
    let mut solver = FlopStartVectorCfr::new(tree, game.table());
    let _ = solver.run(tree, game, n_iters);
    (solver.regrets_river().to_vec(), solver.cum_strategy_river().to_vec())
}

fn run_disk_capture(
    tree: &solver_core::tree::flat::FlatTree,
    game: &FlopStartGame,
    n_iters: u32,
    label: &str,
) -> (Vec<f32>, Vec<f32>) {
    // RAII guards: remove files on Drop (incl. panic). On macOS,
    // std::env::temp_dir() is on the real SSD, not tmpfs.
    struct ScratchFile(std::path::PathBuf);
    impl Drop for ScratchFile {
        fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
    }
    let regrets_guard = ScratchFile(std::env::temp_dir().join(format!("a_parity_regrets_{}.bin", label)));
    let cum_guard = ScratchFile(std::env::temp_dir().join(format!("a_parity_cum_{}.bin", label)));
    let regrets_path = regrets_guard.0.clone();
    let cum_path = cum_guard.0.clone();
    let _ = std::fs::remove_file(&regrets_path); // pre-cleanup if prior leftover
    let _ = std::fs::remove_file(&cum_path);
    let solver = FlopStartVectorCfr::new(tree, game.table());
    let mut solver = solver
        .into_disk_backed(&regrets_path, &cum_path)
        .expect("into_disk_backed");
    let _ = solver.run(tree, game, n_iters);

    // After run, drain remaining loaded pair to disk (run() saves after each
    // bottom_up_zone, but the final pair's state is what matters; the save is
    // already done by the loop).
    //
    // Read the full persistent buffers back from disk and return them.
    let full_len = solver.river_persistent_len();
    let mut regrets = vec![0.0f32; full_len];
    let mut cum = vec![0.0f32; full_len];

    // Read raw bytes from each file.
    use std::io::{Read, Seek, SeekFrom};
    let mut rf = std::fs::OpenOptions::new().read(true).open(&regrets_path).unwrap();
    rf.seek(SeekFrom::Start(0)).unwrap();
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            regrets.as_mut_ptr() as *mut u8,
            regrets.len() * std::mem::size_of::<f32>(),
        )
    };
    rf.read_exact(bytes).unwrap();

    let mut cf = std::fs::OpenOptions::new().read(true).open(&cum_path).unwrap();
    cf.seek(SeekFrom::Start(0)).unwrap();
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            cum.as_mut_ptr() as *mut u8,
            cum.len() * std::mem::size_of::<f32>(),
        )
    };
    cf.read_exact(bytes).unwrap();

    // Cleanup.
    // Files removed by ScratchFile guards on Drop at end of function scope
    // (incl. panic from the bit-exact assertion in the caller).
    drop((regrets_guard, cum_guard));

    (regrets, cum)
}

fn assert_bit_identical(label: &str, mem: &[f32], disk: &[f32]) {
    assert_eq!(mem.len(), disk.len(), "{}: length mismatch ({} vs {})",
               label, mem.len(), disk.len());
    let mut diffs = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..mem.len() {
        if mem[i].to_bits() != disk[i].to_bits() {
            diffs += 1;
            let d = (mem[i] - disk[i]).abs();
            if d > max_abs { max_abs = d; }
        }
    }
    assert_eq!(diffs, 0,
        "{}: {} f32 differ between InMemory and DiskBacked (max abs = {:.6e})",
        label, diffs, max_abs);
}

#[test]
#[ignore = "A parity gate: bit-exact InMemory vs DiskBacked at moderate nh. Run on demand."]
fn a_parity_inmemory_vs_disk_iters_1_3_10() {
    let nh_target = 100;
    eprintln!("\n=== A parity gate: InMemory vs DiskBacked, nh_target = {} ===", nh_target);
    let (tree, game) = build_custom_nh_game(nh_target);
    let nh = game.table().num_valid;
    let n_turn = game.table().remaining_deck.len();
    let max_n_river = game.table().remaining_deck.iter()
        .map(|&tc| game.table().river_decks[tc as usize].len())
        .max().unwrap();
    eprintln!("  tree: {} nodes, nh = {}, n_turn × max_n_river = {} × {} = {} pairs",
              tree.num_nodes(), nh, n_turn, max_n_river, n_turn * max_n_river);

    for &n_iters in &[1u32, 3, 10] {
        eprintln!("\n  --- n_iters = {} ---", n_iters);
        let label = format!("n{}", n_iters);
        let t0 = std::time::Instant::now();
        let (mem_reg, mem_cum) = run_inmemory_capture(&tree, &game, n_iters);
        let t_mem = t0.elapsed().as_millis();

        let t0 = std::time::Instant::now();
        let (disk_reg, disk_cum) = run_disk_capture(&tree, &game, n_iters, &label);
        let t_disk = t0.elapsed().as_millis();

        eprintln!("    InMemory:    {:>6} ms  (regrets len {}, cum_strategy len {})",
                  t_mem, mem_reg.len(), mem_cum.len());
        eprintln!("    DiskBacked:  {:>6} ms  (regrets len {}, cum_strategy len {})",
                  t_disk, disk_reg.len(), disk_cum.len());
        eprintln!("    DiskBacked / InMemory wall-clock ratio: {:.2}×",
                  t_disk as f64 / t_mem as f64);

        assert_bit_identical("regrets_river", &mem_reg, &disk_reg);
        assert_bit_identical("cum_strategy_river", &mem_cum, &disk_cum);
        eprintln!("    ✓ both buffers bit-identical at n_iters={}", n_iters);
    }

    eprintln!("\n=== A parity gate PASS: InMemory and DiskBacked produce bit-identical state ===");
    eprintln!("  All three iter counts (1, 3, 10) match every f32 across both buffers.");
}
