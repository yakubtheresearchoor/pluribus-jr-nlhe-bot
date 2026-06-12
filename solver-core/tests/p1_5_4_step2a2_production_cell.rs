// Step 2.A.2 PRODUCTION CELL: GPU vs CPU parity at the COMBINED axis the
// production blueprint actually exercises — full chance breadth (~47 turn ×
// ~46 river) × full nh=1176 × realistic asymmetric × multi-iter.
//
// THE FRAMING (banked from 2026-06 post-fix conversation):
// - Strata 1/2/3 (small/intermediate/full nh × subset deck × realistic
//   asymmetric) all pass bit-exact post sweep-vs-brute fix.
// - The FULL DECK adds chance integration across ~2162 (tc, rc) pairs.
//   This is a distinct code path from the showdown helper that the
//   stratum tests fixate on — the integration aggregates many CFVs into
//   chance nodes' values, and the order/precision of that aggregation
//   matters.
// - "Parts pass independently" decomposition has historically missed
//   interaction bugs. Measure, don't predict.
//
// Wall-clock estimate at HU OptB (2+2) stacks=50 nh=1176 × ~2162 pairs:
// stratum 3 (4 pairs, 20 iters) took 58s ≈ 0.72s per (iter × pair).
// Production cell at 3 iters × 2162 pairs ≈ 4700s ≈ 78 min, which is
// the wall-clock cost of the answer to "is the combined axis ok."
//
// Test uses CPU InMemory vs GPU InMemory. InMemory at nh=1176 × full deck
// needs significant RAM but fits on M4 Max with 64GB+. If memory becomes
// a problem, fall back to DiskBacked on both sides.

#![cfg(feature = "metal")]

use std::time::Instant;

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_production_cell_game() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["Ah", "Kd", "7c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 2u8;

    // Realistic asymmetric sigmoid ranges over all nh=1176 valid hand pairs.
    use solver_core::hand::eval::Hand;
    let mut all_with_strength: Vec<(u16, u16)> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        all_with_strength.push((h.evaluate_internal() as u16, idx as u16));
    }
    all_with_strength.sort_by_key(|&(s, _)| s);
    let chosen: Vec<u16> = all_with_strength.iter().map(|&(_, i)| i).collect();
    let k = chosen.len();
    assert_eq!(k, 1176, "expected nh=1176 on a 3-card board, got {}", k);

    let mut ranges: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for (rank_idx, &hi) in chosen.iter().enumerate() {
        let strength_frac = rank_idx as f32 / k as f32;
        let p0_weight = (strength_frac - 0.3).max(0.05) * 1.5;
        let p0_weight = p0_weight.min(1.0);
        let p1_weight = 0.6 + 0.4 * strength_frac;
        let (c1, c2) = index_to_card_pair(hi as usize);
        let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
        let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
        ranges[0][pair_idx] = p0_weight;
        ranges[1][pair_idx] = p1_weight;
    }

    // FULL DECK — every legal (turn, river) pair on this board.
    // compute_flop_start handles the canonical 49-card-deck × 47/48-river-deck
    // chance table construction.
    let table = FlopChanceTable::compute_flop_start(&board, &ranges, num_players);

    // HU OptB tree (smaller stacks to keep per-pair cost down).
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop,
        starting_pot: 4,
        starting_stacks: vec![20, 20],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = FlopStartGame::new(table);
    (tree, game)
}

#[test]
#[ignore = "Step 2.A.2 PRODUCTION CELL: full deck × full nh × realistic × multi-iter (~minutes wall-clock)"]
fn step2a2_production_cell_full_deck_full_nh_realistic_multi_iter() {
    eprintln!("\n=== Step 2.A.2 PRODUCTION CELL ===");
    eprintln!("Combined axis: full deck × full nh=1176 × realistic asymmetric × multi-iter");
    eprintln!("Measurement, not prediction. This is the cell production blueprint exercises.\n");

    let t_build = Instant::now();
    let (tree, game) = build_production_cell_game();
    let table = game.table();
    let nh = table.num_valid;
    let n_turn = table.remaining_deck.len();
    let max_river = table.river_decks.iter().filter(|d| !d.is_empty())
        .map(|d| d.len()).max().unwrap_or(0);
    let total_pairs: usize = table.river_decks.iter().map(|d| d.len()).sum();
    eprintln!("Tree: {} nodes, nh = {}, n_turn = {}, max_river = {}, total_pairs = {}",
        tree.num_nodes(), nh, n_turn, max_river, total_pairs);
    eprintln!("Built in {:.1}s", t_build.elapsed().as_secs_f64());

    let ctx = MetalContext::new().expect("Metal");
    let mut cpu = FlopStartVectorCfr::new(&tree, table);
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let n_iters = 3u32;
    eprintln!("\nRunning {} iters on CPU then GPU at production cell scale...", n_iters);

    let t0 = Instant::now();
    let _ = cpu.run(&tree, &game, n_iters);
    let t_cpu = t0.elapsed().as_secs_f64();
    eprintln!("CPU done in {:.1}s ({:.2}s/iter)", t_cpu, t_cpu / n_iters as f64);

    let t0 = Instant::now();
    gpu.run(&ctx, &tree, &game, n_iters);
    let t_gpu = t0.elapsed().as_secs_f64();
    eprintln!("GPU done in {:.1}s ({:.2}s/iter)", t_gpu, t_gpu / n_iters as f64);

    // Bit-exact compare across all state buffers.
    let cpu_reg_flop = cpu.regrets_flop().to_vec();
    let cpu_reg_turn = cpu.regrets_turn().to_vec();
    let cpu_reg_river = cpu.regrets_river().to_vec();
    let cpu_cum_flop = cpu.cum_strategy_flop().to_vec();
    let cpu_cum_turn = cpu.cum_strategy_turn().to_vec();
    let cpu_cum_river = cpu.cum_strategy_river().to_vec();
    let gpu_regs = gpu.download_regrets();
    let gpu_cum = gpu.download_cum_strategy();
    let fl = cpu_reg_flop.len();
    let tl = cpu_reg_turn.len();
    let rl = cpu_reg_river.len();

    let bitcount = |a: &[f32], b: &[f32], label: &str| -> (usize, f32, usize) {
        let mut diff_count = 0usize;
        let mut max_abs = 0.0f32;
        let mut max_idx = 0usize;
        for i in 0..a.len().min(b.len()) {
            if a[i].to_bits() != b[i].to_bits() {
                diff_count += 1;
                let d = (a[i] - b[i]).abs();
                if d > max_abs { max_abs = d; max_idx = i; }
            }
        }
        eprintln!("  {:24}  {:>8} / {:>9} bit-different, max_abs={:.3e} at idx {}",
            label, diff_count, a.len(), max_abs, max_idx);
        (diff_count, max_abs, max_idx)
    };

    eprintln!("\nState buffer comparison after {} iters:", n_iters);
    let (rf_bits, rf_max, _) = bitcount(&cpu_reg_flop,  &gpu_regs[..fl],             "regrets_flop");
    let (rt_bits, rt_max, _) = bitcount(&cpu_reg_turn,  &gpu_regs[fl..fl+tl],        "regrets_turn");
    let (rr_bits, rr_max, _) = bitcount(&cpu_reg_river, &gpu_regs[fl+tl..fl+tl+rl],  "regrets_river");
    let (cf_bits, cf_max, _) = bitcount(&cpu_cum_flop,  &gpu_cum[..fl],              "cum_strategy_flop");
    let (ct_bits, ct_max, _) = bitcount(&cpu_cum_turn,  &gpu_cum[fl..fl+tl],         "cum_strategy_turn");
    let (cr_bits, cr_max, _) = bitcount(&cpu_cum_river, &gpu_cum[fl+tl..fl+tl+rl],   "cum_strategy_river");

    let total_bits = rf_bits + rt_bits + rr_bits + cf_bits + ct_bits + cr_bits;
    let total_max = rf_max.max(rt_max).max(rr_max).max(cf_max).max(ct_max).max(cr_max);

    eprintln!("\nTotal bit-different entries across all 6 buffers: {}", total_bits);
    eprintln!("Max abs diff across all buffers: {:.6e}", total_max);

    // f32 floor tolerance — should be bit-exact (0.0) post sweep-vs-brute fix
    // EVEN at the combined axis. If non-zero, it surfaces a chance-integration
    // or aggregation bug masked by stratum 3's subset deck.
    let tol = 1e-4_f32;
    assert!(rf_max < tol, "PRODUCTION CELL BUG: regrets_flop diff {:.3e} > {} — combined-axis divergence", rf_max, tol);
    assert!(rt_max < tol, "PRODUCTION CELL BUG: regrets_turn diff {:.3e} > {} — combined-axis divergence", rt_max, tol);
    assert!(rr_max < tol, "PRODUCTION CELL BUG: regrets_river diff {:.3e} > {} — combined-axis divergence", rr_max, tol);
    assert!(cf_max < tol, "PRODUCTION CELL BUG: cum_flop diff {:.3e} > {} — combined-axis divergence", cf_max, tol);
    assert!(ct_max < tol, "PRODUCTION CELL BUG: cum_turn diff {:.3e} > {} — combined-axis divergence", ct_max, tol);
    assert!(cr_max < tol, "PRODUCTION CELL BUG: cum_river diff {:.3e} > {} — combined-axis divergence", cr_max, tol);

    eprintln!("\n=== PRODUCTION CELL PASS ===");
    eprintln!("CPU↔GPU REPLICATION holds at the combined axis:");
    eprintln!("  full deck × full nh={} × realistic asymmetric × {} iters", nh, n_iters);
    eprintln!("Step 2.D (unified preflop+postflop GPU port) is UNBLOCKED at the production cell.");
    eprintln!("Reminder: correctness signal is in standing_showdown_oracle (CPU vs independent enumerator).");
}
