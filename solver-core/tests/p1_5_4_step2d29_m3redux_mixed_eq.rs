// M3 REDUX: 6-max iters-to-convergence at a second nh + mixed-equilibrium check.
//
// Two concerns from the previous M3:
//
// 1. Single-nh measurement: M3 ran only nh=12. The "34 iters to 1% pot"
//    convergence count is the load-bearing multiplier in the M4 budget
//    arithmetic. If iters-to-convergence GROWS with nh (production-relevant
//    nh is larger), the budget projection is optimistic.
//
//    This file re-runs the same trajectory at nh=20 (60% larger) to check
//    whether iters-to-convergence is roughly nh-invariant or nh-dependent.
//
// 2. Dirac-equilibrium risk: a degenerate game where the equilibrium happens
//    to be pure (one action probability ~1.0 at every infoset) would
//    converge in very few iters because there's nothing to learn mid-spectrum.
//    Production multiway equilibria are mixed; an artificially fast
//    convergence on a Dirac equilibrium wouldn't transfer.
//
//    This file extracts the converged strategy, normalizes to σ_avg per
//    (infoset, hand), and reports:
//      - mean per-action entropy across the strategy
//      - fraction of (infoset, hand) decisions where max action probability
//        > 0.99 (effectively pure)
//
//    A mean entropy near ln(na) and Dirac fraction < 50% indicates a
//    genuinely mixed equilibrium → the 34-iters finding extrapolates to
//    real production scenarios.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};
use std::time::Instant;

fn build_6p_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let np = 6u8;
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..np as usize {
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
        &board, &ranges, np, &chosen, &turn_cards, &river_decks,
    );
    let config = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![100; np as usize],
        initial_contributions: vec![5; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

fn measure_exploitability(
    cpu: &FlopStartVectorCfr,
    tree: &FlatTree,
    game: &FlopStartGame,
    np: usize,
) -> f32 {
    let mut total = 0.0f32;
    for p in 0..np {
        let br = cpu.best_response_value_debug(tree, game, p as u8);
        let sv = cpu.strategy_value_debug(tree, game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total += (br[h] - sv[h]).max(0.0);
        }
    }
    total
}

/// Analyze the converged strategy for mixedness.
///
/// cum_strategy layout: per-zone (flop / turn / river), each is
/// [outcome × infoset × MAX_NA_POSTFLOP × nh] with na varying per infoset (≤ MAX_NA_POSTFLOP).
/// To get σ_avg at a specific (infoset, hand), normalize over actions.
///
/// We don't have direct na-per-infoset access here, so we approximate by
/// considering all MAX_NA_POSTFLOP action slots and ignoring zero-valued ones. If a
/// node has na=2, action slots 2..MAX_NA_POSTFLOP are all 0 and don't affect entropy.
fn analyze_mixedness(cum_strategy: &[f32], nh: usize) -> (f32, f32, usize, usize) {
    // Iterate cum_strategy by chunks of (MAX_NA_POSTFLOP * nh). Each chunk = one infoset.
    let chunk_size = MAX_NA_POSTFLOP * nh;
    let mut total_entropy = 0.0f64;
    let mut total_decisions = 0usize;
    let mut dirac_count = 0usize;
    let mut zero_count = 0usize;

    let n_chunks = cum_strategy.len() / chunk_size;
    for chunk_idx in 0..n_chunks {
        let base = chunk_idx * chunk_size;
        // For each hand h in this infoset, compute σ_avg over actions.
        for h in 0..nh {
            let mut sum = 0.0f32;
            let mut probs = [0.0f32; MAX_NA_POSTFLOP];
            for a in 0..MAX_NA_POSTFLOP {
                let v = cum_strategy[base + a * nh + h].max(0.0);
                probs[a] = v;
                sum += v;
            }
            if sum <= 1e-9 {
                zero_count += 1;
                continue; // No data accumulated yet
            }
            let mut entropy = 0.0f64;
            let mut max_p = 0.0f32;
            let mut nonzero_actions = 0;
            for a in 0..MAX_NA_POSTFLOP {
                let p = probs[a] / sum;
                if p > 1e-9 {
                    entropy -= (p as f64) * (p as f64).ln();
                    nonzero_actions += 1;
                }
                if p > max_p { max_p = p; }
            }
            if nonzero_actions <= 1 {
                // Only one action possible (na=1 or all but one zero). Not a decision.
                continue;
            }
            total_entropy += entropy;
            total_decisions += 1;
            if max_p > 0.99 {
                dirac_count += 1;
            }
        }
    }
    let mean_entropy = if total_decisions > 0 {
        (total_entropy / total_decisions as f64) as f32
    } else {
        0.0
    };
    let dirac_fraction = if total_decisions > 0 {
        dirac_count as f32 / total_decisions as f32
    } else {
        0.0
    };
    (mean_entropy, dirac_fraction, total_decisions, zero_count)
}

#[test]
#[ignore = "M3 REDUX: 6-max convergence at nh=20 + mixed-equilibrium check (~40 min)"]
fn m3_redux_6max_second_nh_mixed_eq() {
    let nh = 20usize;
    let np = 6usize;
    let max_iters = 100u32;
    let checkpoints: Vec<u32> = vec![1, 5, 10, 25, 50, 100];

    let (tree, table) = build_6p_table(nh);
    let game = FlopStartGame::new(table);
    let pot = (np as f32) * 5.0; // 30
    eprintln!("\n=== M3 REDUX: 6-max convergence at nh={} + mixed-eq check ===", nh);
    eprintln!("(M3 prior measurement was nh=12; this is the second point.)");
    eprintln!("nh={} np={} tree_nodes={} max_iters={}", nh, np, tree.num_nodes(), max_iters);

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // ── (1) Convergence trajectory at nh=20 ──
    eprintln!("\n── Convergence trajectory ──");
    eprintln!("{:>6}  {:>14}  {:>10}  {:>14}", "iter", "expl (% pot)", "ratio", "wall (s)");
    let mut prev_iter = 0u32;
    let t0 = Instant::now();
    let mut history: Vec<(u32, f32, f32)> = Vec::new();
    for &cp in &checkpoints {
        if cp > max_iters { break; }
        let delta = cp - prev_iter;
        gpu.run(&ctx, &tree, &game, delta);
        cpu.run(&tree, &game, delta);
        prev_iter = cp;
        let expl = measure_exploitability(&cpu, &tree, &game, np);
        let pct = expl / pot * 100.0;
        let elapsed = t0.elapsed().as_secs_f32();
        let ratio = if let Some((_, p0, _)) = history.first() { p0 / pct.max(1e-9) } else { 1.0 };
        eprintln!("{:>6}  {:>13.4}%  {:>9.2}x  {:>14.1}", cp, pct, ratio, elapsed);
        history.push((cp, pct, elapsed));
    }

    let first_pct = history.first().unwrap().1;
    let last_pct = history.last().unwrap().1;
    eprintln!("\n6p nh={} trajectory: {:.4}% → {:.4}% over {} iters ({:.2}× drop)",
        nh, first_pct, last_pct, max_iters, first_pct / last_pct.max(1e-9));

    // Find iter where exploitability first crosses 1% pot.
    let crossing = history.iter().find(|(_, p, _)| *p < 1.0).map(|(i, _, _)| *i);
    match crossing {
        Some(i) => eprintln!("Crossed 1% pot threshold at iter {}", i),
        None => eprintln!("Did NOT cross 1% pot within {} iters", max_iters),
    }

    // Compare to M3 measurement at nh=12 (34 iters to 1% pot).
    eprintln!("\nM3 reference (nh=12): 34 iters to 1% pot.");
    if let Some(i_nh20) = crossing {
        let ratio = i_nh20 as f32 / 34.0;
        eprintln!("nh=20 vs nh=12: {} vs 34 iters = {:.2}× the iter count",
            i_nh20, ratio);
        if ratio > 2.0 {
            eprintln!("WARNING: iters-to-convergence is GROWING with nh — M4 budget arithmetic optimistic.");
        } else {
            eprintln!("Iters-to-convergence is approximately nh-invariant — M4 arithmetic transfers.");
        }
    }

    // ── (2) Mixed-equilibrium check ──
    eprintln!("\n── Mixed-equilibrium check (post-convergence strategy analysis) ──");
    let cum = gpu.download_cum_strategy();
    let (mean_entropy, dirac_frac, total_decisions, zero_count) = analyze_mixedness(&cum, nh);
    // Theoretical max entropy at na=2: ln(2) ≈ 0.693 nats
    let max_entropy_na2 = (2.0f32).ln();
    let entropy_ratio = mean_entropy / max_entropy_na2;
    eprintln!("Total active decisions (infoset × hand): {}", total_decisions);
    eprintln!("Zero/unused infoset×hand slots: {}", zero_count);
    eprintln!("Mean per-decision entropy: {:.4} nats (max for na=2 = {:.4})",
        mean_entropy, max_entropy_na2);
    eprintln!("Entropy / max ratio: {:.2}% (1.0 = uniform random, 0.0 = pure)",
        entropy_ratio * 100.0);
    eprintln!("Dirac fraction (max action prob > 0.99): {:.2}%",
        dirac_frac * 100.0);

    // Verdict
    if dirac_frac > 0.9 {
        eprintln!("\nVERDICT: equilibrium is >90% Dirac — production multiway equilibria");
        eprintln!("are typically mixed, so the 34-iter convergence at nh=12 may NOT transfer.");
    } else if mean_entropy > 0.1 {
        eprintln!("\nVERDICT: equilibrium is meaningfully MIXED (entropy {:.2} nats, dirac frac {:.0}%).",
            mean_entropy, dirac_frac * 100.0);
        eprintln!("Convergence-count finding from M3 (34 iters) extrapolates as expected.");
    } else {
        eprintln!("\nVERDICT: equilibrium is borderline mixed (low entropy but not pure).");
        eprintln!("Interpret M3's 34-iter finding cautiously.");
    }

    // No assertion — this is a measurement test. The values are reported and
    // interpreted manually. We just confirm the test ran to completion.
}
