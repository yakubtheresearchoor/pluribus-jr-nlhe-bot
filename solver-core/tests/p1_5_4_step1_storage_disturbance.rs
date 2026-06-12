// Step 1 — Storage/compute disturbance test (the lead-directed).
//
// The "600 GB structural memory layout" diagnosis from the previous turn
// was inspection-plus-arithmetic with an *estimated* river_count and a
// page-fault hypothesis. That's the same shape of confident-jump that
// the first ("allocation idiom") diagnosis was; the first turned out
// incomplete. Before resetting the plan to "abstraction first," this
// probe replaces both unknowns with measurements:
//
//   A. ACTUAL buffer sizes from the constructor — read regrets_*.len(),
//      strategy_*.len(), cum_strategy_*.len(); read the per-zone
//      decision-node counts. No estimates.
//
//   B. DISTURBANCE TEST separating compute_all_strategies from a single
//      bottom_up_zone(River, 0, 0) call. The storage-materialization
//      hypothesis predicts compute_all_strategies dominates (writes
//      sparsely across all (tc,rc) board pairs' strategies up-front)
//      while one bottom_up_zone(River, 0, 0) is fast (touches only one
//      (tc,rc) slice). If both are slow, the cost is per-call compute,
//      not up-front materialization, and the streaming-layout fix
//      wouldn't help.
//
// What this probe does NOT do: it does not measure FULL run() cost
// (which we already know SIGKILLs at >5 min). It separates two of run()'s
// constituent operations so the dominant cost can be attributed.

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 30 { format!("{:.2} GB", b as f64 / (1u64 << 30) as f64) }
    else if b >= 1 << 20 { format!("{:.1} MB", b as f64 / (1u64 << 20) as f64) }
    else { format!("{} KB", b / 1024) }
}

#[test]
#[ignore = "Step 1 disturbance: A=actual sizes, B=separate compute_all_strategies from one bottom_up_zone. Run on demand."]
fn storage_vs_compute_hu_optb_nh1176() {
    // ---------- HU OptB setup ----------
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
    eprintln!("\n=== Setup ===");
    eprintln!("  HU OptB flop tree: {} nodes", nn);
    eprintln!("  nh (num_valid hands at this flop): {}", nh);

    let game = FlopStartGame::new(table);

    // ---------- A. Actual buffer sizes from constructor ----------
    eprintln!("\n=== A. Actual buffer sizes (MEASURED, not estimated) ===");
    let t_ctor = Instant::now();
    let solver = FlopStartVectorCfr::new(&tree, game.table());
    let ctor_ms = t_ctor.elapsed().as_millis();
    eprintln!("  FlopStartVectorCfr::new(): {} ms", ctor_ms);

    // Count decision nodes per zone via the public zones() vector and the
    // tree's player-node set.
    use solver_core::solver::flop_start_vector_cfr::Zone as Z;
    let mut zone_decision_counts = [0u32; 4]; // Preflop, Flop, Turn, River
    let zones = solver.zones();
    for &nid in &tree.decision_node_ids {
        let idx = nid as usize;
        let z = zones[idx];
        let bucket = match z { Z::Preflop => 0, Z::Flop => 1, Z::Turn => 2, Z::River => 3 };
        zone_decision_counts[bucket] += 1;
    }
    eprintln!("  decision nodes per zone: flop={}, turn={}, river={}",
              zone_decision_counts[1], zone_decision_counts[2], zone_decision_counts[3]);

    let n_turn = solver.n_turn_outcomes();
    let max_n_river = solver.max_river_outcomes();
    eprintln!("  n_turn = {}, max_n_river = {}", n_turn, max_n_river);

    // Read actual buffer lengths via the existing public getters.
    let regrets_river_len = solver.regrets_river().len();
    let strategy_river_len = solver.strategy_river().len();
    // No cum_strategy_river getter currently exists; size should match.
    eprintln!("\n  regrets_river:      {:>13} f32 = {}",
              regrets_river_len, fmt_bytes(regrets_river_len * 4));
    eprintln!("  strategy_river:     {:>13} f32 = {}",
              strategy_river_len, fmt_bytes(strategy_river_len * 4));
    // Sanity: derive expected size from constructor formula.
    let river_count = zone_decision_counts[3] as usize;
    use solver_core::tree::flat::MAX_NA_POSTFLOP;
    let expected_river_stride = river_count * MAX_NA_POSTFLOP * nh;
    let expected_total = n_turn * max_n_river * expected_river_stride;
    eprintln!("  derived river_stride = river_count×MAX_NA_POSTFLOP×nh = {}×{}×{} = {} f32 ({})",
              river_count, MAX_NA_POSTFLOP, nh, expected_river_stride,
              fmt_bytes(expected_river_stride * 4));
    eprintln!("  expected total len   = n_turn×max_n_river×river_stride = {}",
              expected_total);
    eprintln!("  matches?             {}", if expected_total == regrets_river_len { "YES" } else { "NO" });

    // Same for turn and flop zones.
    let regrets_turn_len = solver.regrets_turn().len();
    let regrets_flop_len = solver.regrets_flop().len();
    eprintln!("  regrets_turn:       {:>13} f32 = {}",
              regrets_turn_len, fmt_bytes(regrets_turn_len * 4));
    eprintln!("  regrets_flop:       {:>13} f32 = {}",
              regrets_flop_len, fmt_bytes(regrets_flop_len * 4));

    // Total persistent storage: 3 buffers per zone (regrets, strategy, cum_strategy).
    let total = 3 * (regrets_river_len + regrets_turn_len + regrets_flop_len);
    eprintln!("\n  TOTAL PERSISTENT STORAGE (3 buffers × 3 zones):");
    eprintln!("    {} f32 = {}", total, fmt_bytes(total * 4));

    // ---------- B. Disturbance test ----------
    eprintln!("\n=== B. Disturbance test: compute_all_strategies vs ONE bottom_up_zone(River, 0, 0) ===");
    let mut solver = solver; // re-bind as mut

    // B.1: time compute_all_strategies alone.
    eprintln!("\n  B.1 compute_all_strategies (writes ALL strategy_river/turn/flop buffers)");
    let t = Instant::now();
    solver.compute_all_strategies(&tree);
    let cas_ms = t.elapsed().as_millis();
    eprintln!("      time: {} ms ({:.2} s)", cas_ms, cas_ms as f64 / 1000.0);

    // B.2: time the data setup for ONE bottom_up_zone(River, 0, 0) call.
    let flop_reach = solver.compute_reach_flop(&tree, &game);
    let turn_reach = solver.compute_reach_turn(&tree, 0, &flop_reach);
    let river_reach = solver.compute_reach_river(&tree, 0, 0, &turn_reach);
    let mut cfv = vec![0.0f32; nn * nh];
    let params = DcfrParams::new(0);

    // B.3: time ONE bottom_up_zone(River, 0, 0) call.
    eprintln!("\n  B.2 ONE bottom_up_zone(River, ti=0, ri=0) — reads only (ti=0, ri=0)'s strategy slice");
    let t = Instant::now();
    solver.bottom_up_zone(
        &tree, game.table(), 0, &river_reach, &mut cfv,
        Zone::River, Some(0), Some(0), &params,
    );
    let buz_ms = t.elapsed().as_millis();
    eprintln!("      time: {} ms ({:.2} s)", buz_ms, buz_ms as f64 / 1000.0);

    // ---------- Discrimination ----------
    eprintln!("\n=== Diagnosis from the disturbance test ===");
    let cas_total_pairs = (n_turn * max_n_river) as f64;
    let buz_per_pair = buz_ms as f64; // bottom_up_zone is per (ti, ri)
    let projected_full_run_buz = buz_per_pair * cas_total_pairs;
    eprintln!("  compute_all_strategies (one-shot, all (tc,rc)):  {} ms", cas_ms);
    eprintln!("  bottom_up_zone(River, 0, 0) (one (tc,rc)):       {} ms", buz_ms);
    eprintln!("  projected full per-iter bottom_up_zone cost (×{}): {:.0} ms = {:.2} s",
              cas_total_pairs as u64, projected_full_run_buz, projected_full_run_buz / 1000.0);
    eprintln!();
    if cas_ms as f64 > projected_full_run_buz * 0.5 {
        eprintln!("  → compute_all_strategies is comparable to OR larger than the full bottom_up_zone");
        eprintln!("    cost. Up-front strategy materialization is a SIGNIFICANT cost component.");
        eprintln!("    Streaming-strategy fix (compute strategy per (tc,rc) just-in-time inside the");
        eprintln!("    run loop, eliminate the precompute) would address it.");
    } else {
        eprintln!("  → compute_all_strategies is small relative to projected per-call cost.");
        eprintln!("    Per-call (bottom_up_zone) compute dominates; streaming-strategy wouldn't help.");
        eprintln!("    Either per-call compute is genuinely the cost, or the regrets+cum_strategy");
        eprintln!("    access pattern in bottom_up_zone has its own scaling issue.");
    }
}
