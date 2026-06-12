// Step 2.D.13 (#104 localization): single-canonical CPU↔GPU per-canonical
// CFV comparison at the subset config.
//
// 2.D.6 / 2.D.7 used the buggy mutually-blocking pick_subset → per-canonical
// showdown CFV was zero by construction → CPU↔GPU bit-exactness was trivial
// (0 == 0). After #103's fix, 2.D.6 fails with 135 bit-diff regret entries
// (max_abs 2.6e-3). The divergence comes from per-canonical postflop CFV
// differing CPU↔GPU at the subset config.
//
// This test isolates the divergence to ONE per-canonical postflop call:
// build the same FlopChanceTable, run K=1 iter on CPU and GPU, compare
// v_combo at flop root bit-exact. Localizes whether the bug is:
//   - SUBSET-SPECIFIC: full-nh × full-deck CPU↔GPU bit-exact (2.A.2 cell)
//     but subset has a CPU↔GPU mismatch (maybe a showdown special case for
//     small nh)
//   - REAL CPU↔GPU regression introduced by recent changes (would also
//     affect full-config tests, would need broader investigation)

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::flop_combo_layout;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_tiny_flop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 4,
        starting_stacks: vec![10, 10],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&cfg).expect("flop tree builds")
}

fn pick_subset_fixed(canonical: [Card; 3]) -> (Vec<u16>, Vec<u8>, Vec<Vec<u8>>) {
    let board_mask: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut chosen: Vec<u16> = Vec::new();
    let mut used_cards: u64 = board_mask;
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = solver_core::card::index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        if used_cards & (1u64 << c1) != 0 || used_cards & (1u64 << c2) != 0 { continue; }
        chosen.push(idx as u16);
        used_cards |= 1u64 << c1;
        used_cards |= 1u64 << c2;
        if chosen.len() == 8 { break; }
    }
    let hand_mask = used_cards;
    let mut turn_cards: Vec<u8> = Vec::new();
    for c in 0u8..52u8 {
        if hand_mask & (1u64 << c) != 0 { continue; }
        turn_cards.push(c);
        if turn_cards.len() == 2 { break; }
    }
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let mut rivers: Vec<u8> = Vec::new();
        for c in 0u8..52u8 {
            if hand_mask & (1u64 << c) != 0 { continue; }
            if c == tc { continue; }
            rivers.push(c);
            if rivers.len() == 2 { break; }
        }
        river_decks[tc as usize] = rivers;
    }
    (chosen, turn_cards, river_decks)
}

#[test]
#[ignore = "Step 2.D.13: CPU↔GPU per-canonical postflop bit-exact gate at subset config"]
fn step2d13_subset_cpu_gpu_per_canonical_bit_exact() {
    let flop_tree = build_tiny_flop_tree();
    let ctx = MetalContext::new().expect("Metal");
    eprintln!("\n=== Step 2.D.13: CPU↔GPU per-canonical postflop at subset config ===");
    eprintln!("Tree: {} nodes (HU 1+1 stacks=10).", flop_tree.num_nodes());
    eprintln!("Subset: 8 hands (non-mutually-blocking) × 2 turn × 2 river.\n");

    let canonical: [Card; 3] = [
        card_from_str("Ah").unwrap(),
        card_from_str("Kd").unwrap(),
        card_from_str("7c").unwrap(),
    ];
    let np = 2;
    let layout = flop_combo_layout(canonical);
    let mut full_ranges: Vec<Vec<f32>> = vec![vec![0.0f32; NUM_POSSIBLE_HANDS]; np];
    for p in 0..np {
        for (li, &(c1, c2)) in layout.iter().enumerate() {
            let w = 0.5 + (li as f32 * 0.01).sin() * 0.3 + (p as f32) * 0.05;
            full_ranges[p][card_pair_to_index(c1, c2)] = w.max(0.05).min(1.0);
        }
    }

    let board: Vec<Card> = canonical.iter().copied().collect();
    let (chosen, turn_cards, river_decks) = pick_subset_fixed(canonical);
    eprintln!("Chosen hands (8): {:?}", chosen);
    eprintln!("Turn cards: {:?}, river_decks: {:?}",
        turn_cards,
        turn_cards.iter().map(|&tc| river_decks[tc as usize].clone()).collect::<Vec<_>>());

    // CPU path.
    let table_cpu = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &full_ranges, np as u8, &chosen, &turn_cards, &river_decks);
    let nh = table_cpu.num_valid;
    let game_cpu = FlopStartGame::new(table_cpu);
    let mut cpu_solver = FlopStartVectorCfr::new(&flop_tree, game_cpu.table());
    cpu_solver.run(&flop_tree, &game_cpu, 1);

    // GPU path (same table+game shape).
    let table_gpu = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &full_ranges, np as u8, &chosen, &turn_cards, &river_decks);
    let game_gpu = FlopStartGame::new(table_gpu);
    let cpu_solver_for_gpu = FlopStartVectorCfr::new(&flop_tree, game_gpu.table());
    let mut gpu_solver = MetalFlopStartSolver::new(&ctx, &flop_tree, &game_gpu, &cpu_solver_for_gpu);
    gpu_solver.run(&ctx, &flop_tree, &game_gpu, 1);

    // Compare state buffers bit-exact.
    let cpu_regrets_flop = cpu_solver.regrets_flop().to_vec();
    let cpu_regrets_turn = cpu_solver.regrets_turn().to_vec();
    let cpu_regrets_river = cpu_solver.regrets_river().to_vec();
    let gpu_regrets_all = gpu_solver.download_regrets();
    let fl = cpu_regrets_flop.len();
    let tl = cpu_regrets_turn.len();
    let rl = cpu_regrets_river.len();

    let max_abs = |a: &[f32], b: &[f32], label: &str| -> (usize, f32) {
        let mut bd = 0usize;
        let mut m = 0.0f32;
        let mut first_diff_idx = None;
        for i in 0..a.len().min(b.len()) {
            if a[i].to_bits() != b[i].to_bits() {
                bd += 1;
                let d = (a[i] - b[i]).abs();
                if d > m { m = d; }
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
            }
        }
        eprintln!("  {:14} {:>6} bit-diff / {:>6}, max_abs={:.3e}",
            label, bd, a.len(), m);
        if let Some(i) = first_diff_idx {
            eprintln!("    first diff at idx {}: CPU={:.9} GPU={:.9}",
                i, a[i], b[i]);
        }
        (bd, m)
    };

    eprintln!("\n=== State after 1 postflop iter ===");
    let (rf, mf) = max_abs(&cpu_regrets_flop, &gpu_regrets_all[..fl], "regrets_flop");
    let (rt, mt) = max_abs(&cpu_regrets_turn, &gpu_regrets_all[fl..fl+tl], "regrets_turn");
    let (rr, mr) = max_abs(&cpu_regrets_river, &gpu_regrets_all[fl+tl..fl+tl+rl], "regrets_river");

    eprintln!("\nnh = {}, total state buffer sizes: flop {} / turn {} / river {}",
        nh, fl, tl, rl);

    if rf + rt + rr == 0 {
        eprintln!("\n✓ CPU↔GPU subset postflop is BIT-EXACT after 1 iter. The 2.D.6 divergence");
        eprintln!("  must come from something else in the unified blueprint loop integration.");
    } else {
        let max_overall = mf.max(mt).max(mr);
        eprintln!("\n✗ CPU↔GPU subset postflop DIVERGES after 1 iter (max_abs {:.3e}).", max_overall);
        eprintln!("  Real bug: subset showdown helper or related path has a CPU↔GPU mismatch");
        eprintln!("  that the degenerate pick_subset was hiding (all-zero showdown = trivially");
        eprintln!("  bit-exact). 2.A.2 production cell at full nh × full deck is bit-exact;");
        eprintln!("  the subset path specifically has divergence — investigate showdown helper");
        eprintln!("  special cases at small nh.");
    }
}
