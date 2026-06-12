// Step 1 — Per-board variance check on bottom_up_zone(River, tc, rc).
//
// Single-board projection (board zero = 382 ms × 2352 → 15 min) was the
// kind of single-point extrapolation that has been wrong repeatedly in
// this project. This probe samples bottom_up_zone(River, tc, rc) at a
// strided set of (tc, rc) pairs and reports min/max/mean/stddev so the
// per-iter projection has bounded uncertainty.
//
// What this can reveal:
//   - per-board cost is uniform (low variance) → projection of 15 min holds
//   - per-board cost varies 10-100× → projection is wrong; some boards
//     cost much more, some less, and the iter cost is sum-not-mean × n
//   - some boards are pathological (much slower than mean) → indicates
//     board-structure-dependent compute (e.g., monotone boards have more
//     side-pot terminals to compute, or sets of high cards have different
//     hand-cover statistics)

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
#[ignore = "Step 1 per-board variance: sample bottom_up_zone(River, tc, rc) across boards. Run on demand."]
fn per_board_variance_bottom_up_zone_river() {
    let np = 2u8;
    let flop_cfg = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 6,
        starting_stacks: vec![97; np as usize],
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
    let tree = build_tree(&flop_cfg).expect("HU OptB flop builds");

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

    let nh = table.num_valid;
    let nn = tree.num_nodes();
    eprintln!("\n=== Per-board variance for bottom_up_zone(River, tc, rc) ===");
    eprintln!("  HU OptB flop tree: {} nodes, nh={}", nn, nh);

    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    let n_turn = solver.n_turn_outcomes();
    let max_n_river = solver.max_river_outcomes();
    eprintln!("  n_turn = {}, max_n_river = {}, n_pairs = {}", n_turn, max_n_river, n_turn * max_n_river);

    // Skip compute_all_strategies (the >5 min hang). Strategy buffer at zero
    // is OK for timing the call shape.

    let flop_reach = solver.compute_reach_flop(&tree, &game);
    let params = DcfrParams::new(0);
    let mut cfv = vec![0.0f32; nn * nh];

    // Sample boards: stride across the (tc, rc) grid for diversity.
    // 16 samples gives a reasonable spread without hammering the test.
    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();
    let mut samples: Vec<((usize, usize), u128)> = Vec::new();

    // Sample boards strided across the (tc, rc) grid.
    let target_samples = 16;
    let n_pairs = n_turn * max_n_river;
    let stride = (n_pairs / target_samples).max(1);

    // Pre-compute turn reach per sampled tc (reuse where possible).
    let mut current_tc: Option<usize> = None;
    let mut turn_reach: Vec<f32> = Vec::new();

    let mut linear_idx = 0usize;
    let t_total = Instant::now();
    while linear_idx < n_pairs && samples.len() < target_samples {
        let ti = linear_idx / max_n_river;
        let ri = linear_idx % max_n_river;
        if ti >= n_turn { break; }
        // Skip (ti, ri) pairs where ri exceeds this tc's actual river deck.
        let tc = turn_deck[ti];
        let n_river_for_this_tc = table_ref.river_decks[tc as usize].len();
        if ri >= n_river_for_this_tc {
            linear_idx += stride;
            continue;
        }
        // Compute turn reach if needed.
        if current_tc != Some(ti) {
            turn_reach = solver.compute_reach_turn(&tree, ti, &flop_reach);
            current_tc = Some(ti);
        }
        let river_reach = solver.compute_reach_river(&tree, ti, ri, &turn_reach);

        let t = Instant::now();
        solver.bottom_up_zone(
            &tree, table_ref, 0, &river_reach, &mut cfv,
            Zone::River, Some(ti), Some(ri), &params,
        );
        let elapsed = t.elapsed().as_millis();
        samples.push(((ti, ri), elapsed));
        eprintln!("  sample {:2}: (tc={:2}, rc={:2})  {:5} ms",
                  samples.len(), ti, ri, elapsed);
        linear_idx += stride;
    }
    let total_ms = t_total.elapsed().as_millis();
    eprintln!("\n  Total measurement time: {} ms ({:.2} s)", total_ms, total_ms as f64 / 1000.0);

    // Statistics.
    let times: Vec<u128> = samples.iter().map(|(_, t)| *t).collect();
    let n = times.len() as f64;
    let sum: u128 = times.iter().sum();
    let mean = sum as f64 / n;
    let min = *times.iter().min().unwrap();
    let max = *times.iter().max().unwrap();
    let variance: f64 = times.iter().map(|t| {
        let d = *t as f64 - mean;
        d * d
    }).sum::<f64>() / n;
    let stddev = variance.sqrt();

    eprintln!("\n=== Per-board statistics ({} samples) ===", samples.len());
    eprintln!("  min:    {:5} ms", min);
    eprintln!("  mean:   {:5.0} ms", mean);
    eprintln!("  max:    {:5} ms", max);
    eprintln!("  stddev: {:5.0} ms (CV = {:.1}%)", stddev, 100.0 * stddev / mean);
    eprintln!("  max/min ratio: {:.2}×", (max as f64) / (min as f64));

    let projected_mean_iter_ms = mean * n_pairs as f64;
    let projected_max_iter_ms = max as f64 * n_pairs as f64;
    eprintln!("\n=== Per-iter projection (HU, single traverser, bottom_up_zone work only) ===");
    eprintln!("  mean-based:  {:.0} ms = {:.2} min", projected_mean_iter_ms, projected_mean_iter_ms / 60_000.0);
    eprintln!("  max-based:   {:.0} ms = {:.2} min (worst-case upper bound)",
              projected_max_iter_ms, projected_max_iter_ms / 60_000.0);
    eprintln!("  previous single-point projection (board 0 × n_pairs): see prior probe");

    eprintln!("\n=== Variance interpretation ===");
    if (max as f64) / (min as f64) < 2.0 {
        eprintln!("  → Per-board cost is UNIFORM (max/min < 2×). Single-point projection is reliable.");
    } else if (max as f64) / (min as f64) < 5.0 {
        eprintln!("  → Per-board cost shows MODERATE variance ({:.1}×). Use mean-based projection.",
                  (max as f64) / (min as f64));
    } else {
        eprintln!("  → Per-board cost varies WIDELY ({:.1}×). Some boards are pathological;",
                  (max as f64) / (min as f64));
        eprintln!("    investigate which boards are slow and why before trusting the projection.");
    }
}
