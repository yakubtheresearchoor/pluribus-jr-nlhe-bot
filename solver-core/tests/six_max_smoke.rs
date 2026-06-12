#![cfg(feature = "metal")]
// 6-max smoke test: verify a 6-player tree builds, allocates within the
// 36GB memory budget, computes num_combinations correctly, and runs one
// iteration without crashing or producing garbage. Specifically measure:
//
//   1. num_combinations for 6 players (the recursive enumeration falls
//      back to pairwise for nh^np > 1M, which is wrong; smoke test
//      reveals whether the small-nh case works and how the fallback
//      affects the answer when it triggers).
//   2. Brute-force showdown enumeration cost — current code only
//      implements K=2 (3-player) brute-force; for K=5 (6-player) the
//      naive enumeration is O(nh^5). Smoke test reveals whether the
//      existing fallback returns 0, panics, or otherwise.
//   3. GPU memory footprint at small nh, projected to nh=50.
//
// This test is intentionally small (nh=6, short action lines) so we can
// run quickly and observe what works and what does not.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use std::time::Instant;

fn build_6max_table(nh: usize) -> Option<(solver_core::tree::flat::FlatTree, FlopChanceTable)> {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 6u8;
    let num_opp = 5usize; // num_players - 1

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    if all_valid.len() < nh { return None; }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();

    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1; hand_cards[i * 2 + 1] = c2;
    }
    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh { for j in 0..nh {
        if i == j { conflict[i*nh+j] = 1; continue; }
        let (a1, a2) = index_to_card_pair(chosen[i] as usize);
        let (b1, b2) = index_to_card_pair(chosen[j] as usize);
        if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 { conflict[i*nh+j] = 1; }
    }}
    let mut hr = vec![0u16; nh];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        hr[i] = h.evaluate_internal() as u16;
    }
    let tc = vec![card_from_str("3c").unwrap() as u8];
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    rd[tc[0] as usize] = vec![card_from_str("5s").unwrap() as u8];

    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &t in &tc {
        for (i, &hi) in chosen.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let tm = board_mask | (1u64 << t);
            if tm & (1u64 << c1) != 0 || tm & (1u64 << c2) != 0 { continue; }
            let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
            for &bc in &board { h = h.add_card(bc as usize); }
            h = h.add_card(t as usize);
            turn_ranks[t as usize * nh + i] = h.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh)
            .map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            let off = t as usize * num_opp * nh + oi * nh;
            for h in 0..nh { turn_sorted_str[off + h] = items[h].0; turn_sorted_idx[off + h] = items[h].1; }
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &t in &tc {
        let tm = board_mask | (1u64 << t);
        for &r in &rd[t as usize] {
            let fm = tm | (1u64 << r);
            for (i, &hi) in chosen.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if fm & (1u64 << c1) != 0 || fm & (1u64 << c2) != 0 { continue; }
                let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
                for &bc in &board { h = h.add_card(bc as usize); }
                h = h.add_card(t as usize).add_card(r as usize);
                river_ranks[t as usize * 52 * nh + r as usize * nh + i] = h.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; num_players as usize];

    // num_combinations: time this. For 6 players with nh=6, exact enumeration
    // should work (nh^np = 6^6 = 46656 < 1M). For nh=50, nh^np = 1.5e10 > 1M
    // so it falls back to pairwise — WRONG for 6 players.
    eprintln!("Computing num_combinations for {} players, nh={}...", num_players, nh);
    let t0 = Instant::now();
    // Replicate the FlopChanceTable logic for instrumentation
    let nc = {
        let np = num_players as usize;
        let max_enumerate = 1_000_000usize;
        let projected = (nh as u64).pow(np as u32);
        eprintln!("  Projected nh^np = {} (max_enumerate cap = {})", projected, max_enumerate);
        if (nh as usize).pow(np as u32) <= max_enumerate {
            fn enumerate(
                player: usize, np: usize, nh: usize,
                combined_mask: u64,
                hand_cards: &[u8],
                weights: &[Vec<f32>],
                current_weight: f64,
            ) -> f64 {
                if player == np { return current_weight; }
                let mut total = 0.0f64;
                for h in 0..nh {
                    let mask_h: u64 = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
                    if combined_mask & mask_h != 0 { continue; }
                    let w = weights[player][h] as f64;
                    if w == 0.0 { continue; }
                    total += enumerate(
                        player + 1, np, nh,
                        combined_mask | mask_h,
                        hand_cards, weights,
                        current_weight * w,
                    );
                }
                total
            }
            enumerate(0, np, nh, 0, &hand_cards[..], &iw, 1.0)
        } else {
            eprintln!("  TOO LARGE — would fall back to pairwise (WRONG for 6 players)");
            0.0
        }
    };
    let nc_elapsed = t0.elapsed();
    eprintln!("  num_combinations = {} (took {:?})", nc, nc_elapsed);

    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players, num_combinations: nc, river_decks: rd,
    };
    let config = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 30,                                  // 6 × 5 blinds
        starting_stacks: vec![100; 6],
        initial_contributions: vec![5; 6],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    eprintln!("Building 6-player tree...");
    let tb_t0 = Instant::now();
    let tree = match build_tree(&config) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  TREE BUILD FAILED: {}", e);
            return None;
        }
    };
    let tb_elapsed = tb_t0.elapsed();
    eprintln!("  Tree built: {} nodes (took {:?})", tree.num_nodes(), tb_elapsed);
    Some((tree, table))
}

#[test]
fn six_max_smoke_nh6() {
    eprintln!("\n=== 6-max smoke test, nh=6 ===");
    let result = build_6max_table(6);
    if result.is_none() {
        eprintln!("Smoke test stopped at table build.");
        return;
    }
    let (tree, table) = result.unwrap();
    let game = FlopStartGame::new(table);

    // Memory budget estimate. d_river_reach is [nn * np * nh] f32 = 4 bytes.
    let nn = tree.num_nodes();
    let np = 6usize;
    let nh = 6usize;
    let bytes_per_zone = (nn * np * nh * 4) as u64;
    eprintln!("Memory estimates:");
    eprintln!("  d_reach (flop):    {} bytes ({:.2} MB)", bytes_per_zone, bytes_per_zone as f64 / 1e6);
    eprintln!("  d_turn_reach:      {} bytes ({:.2} MB)", bytes_per_zone, bytes_per_zone as f64 / 1e6);
    eprintln!("  d_river_reach:     {} bytes ({:.2} MB)", bytes_per_zone, bytes_per_zone as f64 / 1e6);
    let cfv_bytes = (nn * nh * 4) as u64;
    eprintln!("  d_cfv:             {} bytes ({:.2} MB)", cfv_bytes, cfv_bytes as f64 / 1e6);
    eprintln!("  Total reach+cfv:   {:.2} MB", (bytes_per_zone * 3 + cfv_bytes) as f64 / 1e6);
    eprintln!("  vs 36 GB budget:   {:.4}%", ((bytes_per_zone * 3 + cfv_bytes) as f64 / 36e9) * 100.0);

    eprintln!("\nTrying GPU solver construction...");
    let cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Metal context creation FAILED: {:?}", e);
            return;
        }
    };
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);
    eprintln!("  GPU solver constructed successfully.");

    eprintln!("\nRunning 1 iter on GPU...");
    let t0 = Instant::now();
    gpu.run(&ctx, &tree, &game, 1);
    let iter_elapsed = t0.elapsed();
    eprintln!("  1 iter took: {:?}", iter_elapsed);

    // Sanity check the regrets — they should be finite, not NaN/Inf, and
    // non-trivially-distributed (not all zero).
    let regrets = gpu.download_regrets();
    let n_nan = regrets.iter().filter(|&&v| v.is_nan()).count();
    let n_inf = regrets.iter().filter(|&&v| v.is_infinite()).count();
    let n_nonzero = regrets.iter().filter(|&&v| v != 0.0).count();
    let min_r = regrets.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_r = regrets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("  Regret sanity: NaN={}, Inf={}, nonzero={}/{}",
        n_nan, n_inf, n_nonzero, regrets.len());
    eprintln!("  Regret range: [{:.6e}, {:.6e}]", min_r, max_r);
    assert_eq!(n_nan, 0, "regrets contain NaN");
    assert_eq!(n_inf, 0, "regrets contain Inf");
    if n_nonzero == 0 {
        eprintln!("  ⚠ All regrets are zero. Confirmed: vcfr.metal:420 has");
        eprintln!("    `// Fallback for np >= 4: leave at zero (not supported).`");
        eprintln!("    The 6-player showdown returns 0 CFV, so cfv_avg = 0, so");
        eprintln!("    regret updates produce 0. Iter runs without crashing or");
        eprintln!("    NaN/Inf, but no CFR work happens. This is the structural");
        eprintln!("    blocker for 6max: a K>=3 brute-force (or a tractable");
        eprintln!("    approximation) must be implemented in the helper.");
    }

    // CFV at a sample of terminal nodes — verify they're finite and
    // not all zero. The brute-force for num_opp=5 is currently
    // unimplemented (falls into the num_opp >= 3 fallback path which
    // leaves cfv at 0), so this reveals whether the showdown returns
    // garbage or just zeros for 6-player.
    let cfv = gpu.download_cfv();
    let n_cfv_nan = cfv.iter().filter(|&&v| v.is_nan()).count();
    let n_cfv_inf = cfv.iter().filter(|&&v| v.is_infinite()).count();
    let n_cfv_nonzero = cfv.iter().filter(|&&v| v != 0.0).count();
    eprintln!("  Flop CFV sanity: NaN={}, Inf={}, nonzero={}/{}",
        n_cfv_nan, n_cfv_inf, n_cfv_nonzero, cfv.len());
    assert_eq!(n_cfv_nan, 0, "cfv contains NaN");
    assert_eq!(n_cfv_inf, 0, "cfv contains Inf");

    eprintln!("\nSmoke check: tree built ✓, table built ✓, GPU allocated ✓,");
    eprintln!("  1 iter ran in {:?} ✓, no NaN/Inf ✓", iter_elapsed);
    if n_cfv_nonzero == 0 {
        eprintln!("  WARNING: ALL CFV ZERO — showdown brute-force for num_opp=5 is");
        eprintln!("  unimplemented (current code path leaves cfv at 0).");
    } else {
        eprintln!("  CFV nonzero ({}/{}), showdown produced values.", n_cfv_nonzero, cfv.len());
    }
}

/// Project memory and brute-force cost to nh=50 (production scale).
#[test]
fn six_max_projection_nh50() {
    eprintln!("\n=== 6-max cost projection at nh=50 ===");
    let nh = 50usize;
    let np = 6usize;
    let num_opp = np - 1;

    // num_combinations cost
    let nh_pow_np = (nh as u64).pow(np as u32);
    eprintln!("num_combinations cost:");
    eprintln!("  nh^np = {}^{} = {}", nh, np, nh_pow_np);
    eprintln!("  current FlopChanceTable cap: 1,000,000 → would FALL BACK to pairwise");
    eprintln!("  Pairwise fallback is WRONG for 6 players (counts only h0×h1).");

    // Brute-force showdown cost per terminal
    let bf_per_terminal_per_hand = (nh as u64).pow(num_opp as u32);
    let bf_per_terminal = bf_per_terminal_per_hand * nh as u64;
    eprintln!("Brute-force showdown cost (per terminal):");
    eprintln!("  per-hand inner enumeration = nh^{} = {}", num_opp, bf_per_terminal_per_hand);
    eprintln!("  per-terminal total = nh × that = {}", bf_per_terminal);
    eprintln!("  At 3-player nh=50: 50^2 × 50 = 125,000 ops/terminal");
    eprintln!("  At 6-player nh=50: 50^5 × 50 = {} ops/terminal", bf_per_terminal);
    eprintln!("  Scale factor: {}x slower than 3-player", bf_per_terminal / 125_000);

    // Memory at nh=50, np=6 — full 6-max tree size needs an actual build,
    // but a back-of-envelope: 3-player nn was ~36k. 6-player tree will be
    // MUCH larger due to longer betting sequences (6 players × actions).
    eprintln!("\nMemory projection (3-player tree was nn≈36k at nh=50):");
    let est_nn_3p = 36_000usize;
    // 6-player tree depth ≈ doubles, branching ≈ same, so nn ~ depth*branching ≈ 36k × 2 = 72k? Conservative 100k.
    let est_nn_6p = 100_000usize;
    let est_reach_6p = (est_nn_6p * np * nh * 4) as u64;
    let est_total = est_reach_6p * 3 + (est_nn_6p * nh * 4) as u64;
    eprintln!("  Est. 6-player nn ≈ {} (vs 3-player {})", est_nn_6p, est_nn_3p);
    eprintln!("  Est. d_reach (one zone): {} bytes ({:.2} GB)", est_reach_6p, est_reach_6p as f64 / 1e9);
    eprintln!("  Est. total reach+cfv:    {} bytes ({:.2} GB)", est_total, est_total as f64 / 1e9);
    eprintln!("  vs 36 GB budget:         {:.2}%", (est_total as f64 / 36e9) * 100.0);

    // Strict findings
    eprintln!("\nFindings to act on:");
    eprintln!("  (1) num_combinations: fix the N>2 path so it doesn't silently fall");
    eprintln!("      back to pairwise. Options: smarter enumeration (use the");
    eprintln!("      independence-corrected formula like Cepheus uses), or accept");
    eprintln!("      that the pairwise fallback gives a wrong normalization.");
    eprintln!("  (2) Showdown brute-force: O(nh^5) per terminal will not run in");
    eprintln!("      reasonable time. Need either (a) isomorphism-aware enumeration,");
    eprintln!("      (b) cached partial sums across hands, or (c) the product");
    eprintln!("      formula approximation as fallback (accepting some error).");
    eprintln!("  (3) Memory: at nh=50 it likely fits in the 36 GB budget, but");
    eprintln!("      depends on actual tree size which the smoke test does not yet");
    eprintln!("      measure (the tree builder can run, but builds may be huge).");
}
