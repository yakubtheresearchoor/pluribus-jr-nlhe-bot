// Phase 4 REDO — the actual measurement.
//
// Composes P0 (harder config that produces nonzero rich exploitability),
// P1 (empirical bet-size observer that drives lean selection), P2 (cross-
// action-space best-response evaluator via cross_tree::lift_strategy +
// best_response::exploitability).
//
// THE QUESTION: at what K does the empirical top-K lean action set incur
// cross-action-space exploit ≤ 0.05% pot above rich's own exploit?
//
// Production target = 0.05% pot (tight). Above the production target, lean
// is strictly worse than rich and we'd be sacrificing solution quality for
// a cheaper postflop tree. At-or-below the target, the lean set is sound.
//
// The smallest K that passes IS MAX_NA_POSTFLOP (+1 for fold and +1 for
// check/call, since both must always be included: na_total = K + 2).
//
// COMPARED TO PHASE 4 v4 (reverted):
//   - v4 hit 0% vs 0% — both at the f32 floor. Vacuous.
//   - v4 lean set came from hand-rolled bands (workaround for top-K-by-mass
//     triggering tree explosion). Not empirical.
//   - v4 used same-action-space exploitability — couldn't see dropped sizes.
//
// THIS TEST FIXES ALL THREE:
//   - Config has rich exploitability ~0.87% pot at iter 25 (measured in P0)
//   - Lean set is the empirical top-K with a structural min-chip floor (no
//     band hand-picking)
//   - Cost is the cross-action-space BR-exploit measurement (lifted lean
//     strategy played against best-response opponent in the rich tree)

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::cross_tree::{build_action_map, lift_into_rich_solver_with_lean};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{
    FlatTree, ACTION_LABEL_BET, ACTION_LABEL_RAISE,
};
use std::collections::BTreeMap;
use std::time::Instant;

// === Config (from P0 probe) ====================================================

const STACKS: i32 = 500;
const STARTING_POT: i32 = 30;
const STARTING_CONTRIB: i32 = 5;
const NH: usize = 6;
const NP: u8 = 6;
const N_ITERS: u32 = 50; // longer than P0 probe — give all K levels time to converge in lean game

// Min chip increment to count a bet as a "real" size choice. Drops tiny
// bets that would otherwise top the conditional-mass ranking and trigger
// tree explosion when lean tries to include them. 10 chips = 33% of pot
// at the start; smaller bets are uncommon in real play and structurally
// expensive to support in the postflop tree.
const MIN_CHIP_INCREMENT: i32 = 10;

fn rich_bet_sizes() -> Vec<BetSize> {
    vec![
        BetSize::PotRelative(0.33),
        BetSize::PotRelative(0.66),
        BetSize::PotRelative(1.0),
        BetSize::PotRelative(1.5),
    ]
}
fn rich_raise_sizes() -> Vec<BetSize> {
    vec![
        BetSize::PotRelative(1.0),
        BetSize::PotRelative(2.0),
    ]
}

fn build_chance_table() -> FlopChanceTable {
    let board: Vec<Card> = ["Th", "9d", "8c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
            ranges[p][pair_idx] = 1.0;
        }
    }
    let turn_cards: Vec<u8> = ["2c", "Jd"].iter()
        .map(|s| card_from_str(s).unwrap() as u8).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = ["4s", "7h"].iter()
        .map(|s| card_from_str(s).unwrap() as u8).collect();
    river_decks[turn_cards[1] as usize] = ["3s", "Qc"].iter()
        .map(|s| card_from_str(s).unwrap() as u8).collect();
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

fn build_tree_with(bets: Vec<BetSize>, raises: Vec<BetSize>) -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: STARTING_POT,
        starting_stacks: vec![STACKS; NP as usize],
        initial_contributions: vec![STARTING_CONTRIB; NP as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: bets, raise: raises },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    build_tree(&config).unwrap()
}

// === P1: Empirical bet-size observer ============================================
//
// At every PLAYER node, for every hand h:
//   1. Normalize cum_strategy → σ_avg[a, h]
//   2. bet_mass(h) = sum over actions a where action is BET or RAISE of σ_avg[a, h]
//   3. If bet_mass(h) is meaningfully nonzero, then for each such action a:
//        conditional_prob(size, h) = σ_avg[a, h] / bet_mass(h)
//        ↑ "given the player chose to bet here with this hand, P(size = this)"
//   4. Bucket by pot_fraction (rounded to 0.01) and accumulate conditional probs
//   5. Normalize across buckets → rank by aggregate conditional mass
//
// This is what the v4 banded heuristic was hand-rolling around. With the
// MIN_CHIP_INCREMENT floor, we don't need the bands — tiny-bet pathologies
// are filtered structurally, not by hand-picked ranges.

#[derive(Debug, Clone)]
struct BetSizeObservation {
    pot_fraction: f32,      // representative (mean of bucketed observations)
    conditional_mass: f64,  // sum of conditional probability across all infosets×hands
    occurrence_count: usize,
}

fn observe_bet_sizes(
    tree: &FlatTree,
    cum_strategy: &[f32],
    offsets: &[usize],
) -> Vec<BetSizeObservation> {
    let np = tree.num_players as usize;
    let nh = NH;

    // Bucket pot_fraction by 0.01 granularity (rounded).
    let mut bucket_mass: BTreeMap<i32, f64> = BTreeMap::new();
    let mut bucket_frac_sum: BTreeMap<i32, f64> = BTreeMap::new();
    let mut bucket_count: BTreeMap<i32, usize> = BTreeMap::new();

    for &nid in &tree.decision_node_ids {
        let idx = nid as usize;
        let node = &tree.nodes[idx];
        let p = node.player_id as usize;
        let na = node.num_children as usize;
        let off = offsets[idx];
        let children = tree.node_children(idx).to_vec();

        // Pot at this node = sum of all contributions. `tree.starting_pot`
        // is NOT added: the builder's bet-size math uses contributions-only
        // (verified empirically — adding starting_pot causes declared 0.66p
        // bets to bucket at 0.55, etc.). showdown.rs's formula adds
        // starting_pot for FINAL terminal pot only.
        let mut pot: i64 = 0;
        for pp in 0..np {
            pot += tree.contributions[idx * np + pp] as i64;
        }
        if pot <= 0 { continue; }

        let parent_p_contrib = tree.contributions[idx * np + p];

        for h in 0..nh {
            // Normalize σ_avg over actions for this hand
            let mut total = 0.0f32;
            for a in 0..na {
                total += cum_strategy[off + a * nh + h].max(0.0);
            }
            if total <= 1e-12 { continue; }

            // Identify bet actions and accumulate bet_mass
            let mut bet_mass = 0.0f32;
            for a in 0..na {
                let ac = &tree.nodes[children[a] as usize];
                if ac.action_label != ACTION_LABEL_BET && ac.action_label != ACTION_LABEL_RAISE {
                    continue;
                }
                bet_mass += cum_strategy[off + a * nh + h].max(0.0) / total;
            }
            if bet_mass < 1e-6 { continue; }

            // For each bet action, accumulate its conditional probability
            for a in 0..na {
                let ac = &tree.nodes[children[a] as usize];
                if ac.action_label != ACTION_LABEL_BET && ac.action_label != ACTION_LABEL_RAISE {
                    continue;
                }
                let child_p_contrib = tree.contributions[children[a] as usize * np + p];
                let inc = child_p_contrib - parent_p_contrib;
                if inc < MIN_CHIP_INCREMENT { continue; }
                let prob = cum_strategy[off + a * nh + h].max(0.0) / total;
                let cond = (prob / bet_mass) as f64;
                let pot_frac = inc as f32 / pot as f32;
                let bucket = (pot_frac * 100.0).round() as i32;
                *bucket_mass.entry(bucket).or_insert(0.0) += cond;
                *bucket_frac_sum.entry(bucket).or_insert(0.0) += pot_frac as f64;
                *bucket_count.entry(bucket).or_insert(0) += 1;
            }
        }
    }

    let total_mass: f64 = bucket_mass.values().sum();
    let mut out: Vec<BetSizeObservation> = bucket_mass.iter()
        .map(|(&bucket, &m)| {
            let count = bucket_count[&bucket];
            let frac = (bucket_frac_sum[&bucket] / count as f64) as f32;
            BetSizeObservation {
                pot_fraction: frac,
                conditional_mass: if total_mass > 0.0 { m / total_mass } else { 0.0 },
                occurrence_count: count,
            }
        })
        .collect();
    out.sort_by(|a, b| b.conditional_mass.partial_cmp(&a.conditional_mass).unwrap());
    out
}

// Snap an observed pot fraction to one of the DECLARED rich bet sizes. The
// lean tree builder takes BetSize::PotRelative(f) so we have to feed it the
// exact rich values, not the bucketed averages (which might be 0.34 vs the
// declared 0.33 due to integer chip math).
fn snap_to_rich_size(pot_frac: f32) -> Option<f32> {
    let candidates = [0.33f32, 0.66, 1.0, 1.5];
    let mut best = None;
    let mut best_diff = f32::INFINITY;
    for &c in &candidates {
        let d = (pot_frac - c).abs();
        if d < best_diff && d < 0.10 {
            best = Some(c);
            best_diff = d;
        }
    }
    best
}

// === P2: Run the cross-action-space measurement =================================

fn solve(tree: &FlatTree, game: &FlopStartGame, n_iters: u32) -> FlopStartVectorCfr {
    let mut cpu = FlopStartVectorCfr::new(tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, tree, game, &cpu);
    gpu.run(&ctx, tree, game, n_iters);
    cpu.run(tree, game, n_iters);
    cpu
}

/// Read flattened cum_strategy from a solved cpu (for the bet-size observer only).
fn get_flattened(cpu: &FlopStartVectorCfr, tree: &FlatTree) -> (Vec<f32>, Vec<usize>) {
    cpu.flattened_cum_strategy(tree)
}

fn measure_rich_exploitability_pct(
    cpu: &FlopStartVectorCfr,
    tree: &FlatTree,
    game: &FlopStartGame,
) -> f32 {
    let np = tree.num_players as usize;
    let mut total = 0.0f32;
    for p in 0..np {
        let br = cpu.best_response_value_debug(tree, game, p as u8);
        let sv = cpu.strategy_value_debug(tree, game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total += (br[h] - sv[h]).max(0.0);
        }
    }
    total / STARTING_POT as f32 * 100.0
}

#[test]
#[ignore = "Phase 4 redo measurement — solves rich + lean, lifts, computes cross-action-space exploit (~30 min)"]
fn phase4_redo_measurement() {
    eprintln!("\n=== Phase 4 REDO measurement ===");
    eprintln!("Config: deep wet stacks={} board=Th9d8c chance=2x2 nh={} np={} iters={}",
        STACKS, NH, NP, N_ITERS);

    // ── Build chance table & rich tree ──
    let table = build_chance_table();
    let rich_tree = build_tree_with(rich_bet_sizes(), rich_raise_sizes());
    eprintln!("\nRich tree: {} nodes, {} infosets",
        rich_tree.num_nodes(), rich_tree.num_infosets);

    // ── Solve rich ──
    eprintln!("\n── Solve RICH ({} iters) ──", N_ITERS);
    let t0 = Instant::now();
    let rich_game = FlopStartGame::new(table);
    let rich_cpu = solve(&rich_tree, &rich_game, N_ITERS);
    eprintln!("Rich solve wall: {:.1}s", t0.elapsed().as_secs_f32());

    // Rich exploitability via solver-internal API (same-action-space).
    let rich_pct = measure_rich_exploitability_pct(&rich_cpu, &rich_tree, &rich_game);
    eprintln!("Rich self-exploitability: {:.4}% pot", rich_pct);
    assert!(rich_pct > 0.1,
        "P0 gate violated: rich expl {:.4}% pot below 0.1% — comparison is vacuous", rich_pct);

    // Sanity check: lift rich's strategy back into a fresh rich solver and
    // measure exploit. Should equal rich_pct because rich → rich is identity
    // (every node pairs with itself, every action with itself).
    {
        let rich_self_map = build_action_map(&rich_tree, &rich_tree);
        let table_id = build_chance_table();
        let rich_game_id = FlopStartGame::new(table_id);
        let mut rich_cpu_id = FlopStartVectorCfr::new(&rich_tree, &rich_game_id.table());
        lift_into_rich_solver_with_lean(&rich_tree, &rich_tree, &rich_self_map, &rich_cpu, &mut rich_cpu_id);

        // ── Layout sanity: are source/dest local-offset tables identical? ──
        let lo_match = rich_cpu.flop_local_offset() == rich_cpu_id.flop_local_offset();
        let to_match = rich_cpu.turn_local_offset() == rich_cpu_id.turn_local_offset();
        let ro_match = rich_cpu.river_local_offset() == rich_cpu_id.river_local_offset();
        let z_match = rich_cpu.zones() == rich_cpu_id.zones();
        eprintln!("Layout: flop_local match={} turn_local match={} river_local match={} zones match={}",
            lo_match, to_match, ro_match, z_match);
        if !lo_match || !to_match || !ro_match || !z_match {
            // Find first divergent index per table for triage.
            for (i, (a, b)) in rich_cpu.flop_local_offset().iter()
                .zip(rich_cpu_id.flop_local_offset().iter()).enumerate() {
                if a != b { eprintln!("  first flop_local divergence: idx={} src={} dst={}", i, a, b); break; }
            }
        }

        // ── Diagnostic: are the cum_strategy buffers actually equal after identity lift? ──
        let f1 = rich_cpu.cum_strategy_flop();
        let f2 = rich_cpu_id.cum_strategy_flop();
        let t1 = rich_cpu.cum_strategy_turn();
        let t2 = rich_cpu_id.cum_strategy_turn();
        let r1 = rich_cpu.cum_strategy_river();
        let r2 = rich_cpu_id.cum_strategy_river();
        eprintln!("Buffer-equality diagnostic after identity lift:");
        eprintln!("  flop:  rich.len={}, id.len={}", f1.len(), f2.len());
        eprintln!("  turn:  rich.len={}, id.len={}", t1.len(), t2.len());
        eprintln!("  river: rich.len={}, id.len={}", r1.len(), r2.len());

        fn buf_diff(name: &str, a: &[f32], b: &[f32]) -> (usize, f64, f64) {
            let mut diffs = 0usize;
            let mut max_abs = 0.0f64;
            let mut sum_a = 0.0f64;
            let mut sum_b = 0.0f64;
            let n = a.len().min(b.len());
            for i in 0..n {
                sum_a += a[i] as f64;
                sum_b += b[i] as f64;
                if a[i] != b[i] {
                    diffs += 1;
                    let d = (a[i] as f64 - b[i] as f64).abs();
                    if d > max_abs { max_abs = d; }
                }
            }
            eprintln!("  {} sum_rich={:.3e} sum_id={:.3e} diffs={}/{} max_abs={:.3e}",
                name, sum_a, sum_b, diffs, n, max_abs);
            (diffs, max_abs, sum_a)
        }
        let (fd, _, fs) = buf_diff("flop", f1, f2);
        let (td, _, ts) = buf_diff("turn", t1, t2);
        let (rd, _, rs) = buf_diff("river", r1, r2);
        let total_diffs = fd + td + rd;
        eprintln!("  Total nonzero entries in rich (proxy for accumulation): flop_sum={:.3e} turn_sum={:.3e} river_sum={:.3e}", fs, ts, rs);

        let id_pct = measure_rich_exploitability_pct(&rich_cpu_id, &rich_tree, &rich_game_id);
        eprintln!("Identity-lift sanity check: id_pct={:.4}% rich_pct={:.4}% (should be ≈ equal)", id_pct, rich_pct);

        // ── Brute-copy bypass: overwrite the entire buffers verbatim, bypassing
        // my iteration logic. If BR matches now, my lift iteration misses
        // entries. If BR still doesn't match, something other than cum_strategy
        // matters for BR. ──
        {
            let src_f = rich_cpu.cum_strategy_flop().to_vec();
            let src_t = rich_cpu.cum_strategy_turn().to_vec();
            let src_r = rich_cpu.cum_strategy_river().to_vec();
            rich_cpu_id.cum_strategy_flop_mut().copy_from_slice(&src_f);
            rich_cpu_id.cum_strategy_turn_mut().copy_from_slice(&src_t);
            rich_cpu_id.cum_strategy_river_mut().copy_from_slice(&src_r);
            let brute_pct = measure_rich_exploitability_pct(&rich_cpu_id, &rich_tree, &rich_game_id);
            eprintln!("BRUTE-COPY BR result: {:.4}% pot (rich self-pct = {:.4}%)", brute_pct, rich_pct);
            if (brute_pct - rich_pct).abs() / rich_pct.max(1e-6) < 0.05 {
                eprintln!("  → BRUTE-COPY MATCHES — bug is in my lift iteration");
            } else {
                eprintln!("  → BRUTE-COPY STILL OFF — bug is downstream (game state, regrets, or strategy buffer)");
            }
        }
        eprintln!("  Buffer-equality verdict: {}",
            if total_diffs == 0 { "BUFFERS IDENTICAL — bug is downstream of lift" }
            else { "BUFFERS DIFFER — bug is in lift" });

        let gap = (id_pct - rich_pct).abs() / rich_pct.max(1e-6);
        assert!(gap < 0.05,
            "Identity-lift sanity FAILED: {:.4}% vs {:.4}% ({:.1}% relative gap, > 5%) — \
             per-outcome lift is broken, do not trust K-sweep numbers. \
             total_diffs={}", id_pct, rich_pct, gap * 100.0, total_diffs);
    }

    // ── Observe rich's bet-size usage ──
    eprintln!("\n── P1: Empirical bet-size observation ──");
    let (rich_buf, rich_buf_offsets) = get_flattened(&rich_cpu, &rich_tree);
    let observations = observe_bet_sizes(&rich_tree, &rich_buf, &rich_buf_offsets);
    eprintln!("Bucketed bet-size usage (sorted by conditional mass):");
    eprintln!("  {:>10}  {:>8}  {:>12}  {:>10}", "pot_frac", "snap", "cond_mass", "count");
    for o in &observations {
        let snap = snap_to_rich_size(o.pot_fraction);
        let snap_str = snap.map(|s| format!("{:.2}", s)).unwrap_or_else(|| "—".to_string());
        eprintln!("  {:>10.3}  {:>8}  {:>11.2}%  {:>10}",
            o.pot_fraction, snap_str,
            o.conditional_mass * 100.0,
            o.occurrence_count);
    }

    // Build ordered list of unique rich sizes ranked by observed usage.
    let mut ranked_sizes: Vec<f32> = Vec::new();
    let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for o in &observations {
        if let Some(s) = snap_to_rich_size(o.pot_fraction) {
            let key = (s * 100.0).round() as i32;
            if seen.insert(key) {
                ranked_sizes.push(s);
            }
        }
    }
    eprintln!("\nRanked rich bet sizes by observed usage: {:?}", ranked_sizes);

    // ── P3: K sweep ──
    eprintln!("\n── P3: K sweep (each K builds + solves a lean tree, then lifts + BRs in rich) ──");
    eprintln!("Lean set always includes raise [1.0]; bet sizes = top-K of ranked_sizes.\n");

    let max_k = ranked_sizes.len().min(4);
    let mut k_results: Vec<(usize, Vec<f32>, f32, f32, f32, usize)> = Vec::new(); // (K, sizes, lean_self_pct, xt_pct, cost_pct, nodes)
    // K_FULL = K=4 with both raises [1.0, 2.0] — should give ~0% cost (lean
    // is effectively rich); sanity check that the measurement converges at
    // the identity boundary.
    let configs: Vec<(usize, Vec<BetSize>, Vec<BetSize>, &str)> = (1..=max_k).map(|k| {
        let bets: Vec<BetSize> = ranked_sizes[..k].iter().map(|&f| BetSize::PotRelative(f as f64)).collect();
        let raises = vec![BetSize::PotRelative(1.0)];
        (k, bets, raises, "1 raise")
    }).chain(std::iter::once((
        max_k + 1,
        ranked_sizes[..max_k].iter().map(|&f| BetSize::PotRelative(f as f64)).collect(),
        vec![BetSize::PotRelative(1.0), BetSize::PotRelative(2.0)],
        "K_FULL (rich raises)",
    ))).collect();

    for (k, lean_bets, lean_raises, raise_label) in configs {
        eprintln!("── K={}: bets={:?} raises={} ──", k, lean_bets, raise_label);

        let lean_tree = build_tree_with(lean_bets.clone(), lean_raises.clone());
        eprintln!("  Lean tree: {} nodes, {} infosets ({:.2}× rich)",
            lean_tree.num_nodes(),
            lean_tree.num_infosets,
            lean_tree.num_nodes() as f32 / rich_tree.num_nodes() as f32);
        let lean_bets_for_record: Vec<f32> = lean_bets.iter().map(|b| match b {
            BetSize::PotRelative(f) => *f as f32,
            _ => 0.0,
        }).collect();

        // Sanity: validate cross-tree pairing on this lean.
        let map = build_action_map(&rich_tree, &lean_tree);
        let lean_paired = map.paired_player_count(&rich_tree);
        eprintln!("  Cross-tree pair: {}/{} rich PLAYER nodes paired with a lean node",
            lean_paired, rich_tree.decision_node_ids.len());

        // Solve lean.
        let t_lean = Instant::now();
        let table_lean = build_chance_table();
        let lean_game = FlopStartGame::new(table_lean);
        let lean_cpu = solve(&lean_tree, &lean_game, N_ITERS);
        let lean_wall = t_lean.elapsed().as_secs_f32();
        let lean_self_pct = measure_rich_exploitability_pct(&lean_cpu, &lean_tree, &lean_game);
        eprintln!("  Lean solve: {:.1}s; lean self-expl (lean-space): {:.4}% pot",
            lean_wall, lean_self_pct);

        // Lift lean's per-outcome cum_strategy → rich-solver's per-outcome
        // buffers, then use rich's internal BR walker (which correctly
        // tracks tc/rc — the public best_response.rs walker can't).
        let table_lift = build_chance_table();
        let rich_game_lift = FlopStartGame::new(table_lift);
        let mut rich_cpu_lifted = FlopStartVectorCfr::new(&rich_tree, &rich_game_lift.table());
        lift_into_rich_solver_with_lean(&rich_tree, &lean_tree, &map, &lean_cpu, &mut rich_cpu_lifted);
        let xt_pct = measure_rich_exploitability_pct(&rich_cpu_lifted, &rich_tree, &rich_game_lift);
        let cost_pct = xt_pct - rich_pct;
        eprintln!("  Cross-action-space expl (lifted lean in RICH): {:.4}% pot", xt_pct);
        eprintln!("  COST of leaning = xt - rich = {:.4}% pot", cost_pct);

        k_results.push((k, lean_bets_for_record, lean_self_pct, xt_pct, cost_pct, lean_tree.num_nodes()));
        eprintln!();
    }

    // ── Verdict ──
    eprintln!("\n=== Phase 4 REDO summary ===");
    eprintln!("Rich self-exploitability: {:.4}% pot at iter {}", rich_pct, N_ITERS);
    eprintln!();
    eprintln!("{:>3} {:>4} {:>30} {:>14} {:>14} {:>14}",
        "K", "MAX_NA",  "lean_bets", "lean_self_%", "xt_%", "cost_%");
    for (k, sizes, lean_self, xt, cost, _) in &k_results {
        // na_total = K bets + fold + check/call + raise (always 1) = K + 3
        let max_na = *k + 3;
        eprintln!("{:>3} {:>4} {:>30} {:>13.4}% {:>13.4}% {:>13.4}%",
            k, max_na, format!("{:?}", sizes), lean_self, xt, cost);
    }

    let target_pct = 0.05f32; // tight production target above rich
    let mut chosen: Option<&(usize, Vec<f32>, f32, f32, f32, usize)> = None;
    for r in &k_results {
        if r.4 <= target_pct {
            chosen = Some(r);
            break;
        }
    }
    eprintln!();
    match chosen {
        Some(r) => {
            let max_na = r.0 + 3;
            eprintln!("VERDICT: Smallest K with cost ≤ {:.2}% pot is K={} → MAX_NA_POSTFLOP = {}",
                target_pct, r.0, max_na);
            eprintln!("Empirical sizes: {:?}", r.1);
        }
        None => {
            eprintln!("VERDICT: No K within tested range hit the {:.2}% pot cost gate.",
                target_pct);
            eprintln!("Either widen K (more bet sizes) or loosen target. MAX_NA_POSTFLOP > 4+3 = 7.");
        }
    }
}
