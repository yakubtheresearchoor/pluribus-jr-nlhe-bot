// Phase 1 Slice 7a follow-up: multi-flop convergence check.
//
// Context: the single-flop convergence study on canonical_flops[0]
// showed an anomalous trajectory at production nh=1176:
//   iter 10: avg=[0.382, 0.618]  (action-1 preferred)
//   iter 30: avg=[0.752, 0.248]  (action-0 preferred, flipped)
// with iter-delta still drifting at 0.024 by iter 30.
//
// Per the lead (2026-06-04): "A time-averaged strategy flipping its
// preferred action between iter 10 and 30 and still drifting on a
// 2-action single-flop case is slow/unstable for how simple that
// problem is. Before sizing a 16-32 hour subset study on the
// assumption N is just large, run the corrected metric on a few
// different canonical flops (cheap, ~30 min) to distinguish flop-
// specific close-decision (benign, N fine elsewhere) from a CFR
// convergence problem (not benign, would mean the CPU reference is
// itself questionable, which defeats the gate's purpose)."
//
// This test runs the corrected cum_strategy convergence metric on
// SIX canonical flops spanning the canonical list (indices 0, 350,
// 700, 1050, 1400, 1754) at 30 iters each, ~34 min total. The
// indices are chosen to span the canonical-flop enumeration without
// cherry-picking archetype labels.
//
// Distinguishing reads of the results:
//
//   1. ALL flops show the same flippy/drifting pattern as flop[0].
//      → CFR convergence problem, not flop-specific. Investigate
//        FlopStartVectorCfr (DCFR params, strategy update, regret
//        accumulation) BEFORE sizing the bounded slice 7c run. The
//        CPU is not yet the hyper-confident reference the gate needs.
//
//   2. MOST flops settle to a stable preference by iter 30, only a
//      few flip / drift.
//      → Benign: those flops are close-decision (mixed equilibrium,
//        legitimately near-50/50). Slice 7c sizing proceeds with the
//        N estimate based on the STABLE flops, not the close-decision
//        ones. The slice 7c subset study at proper N would then
//        confirm.
//
//   3. SOMETHING ELSE: e.g., per-flop wall-clock varies wildly,
//      some flops error, the avg snapshot is identical across flops
//      (= bug in the helper reading the wrong infoset).
//      → Investigate the specific anomaly before sizing.

use solver_core::abstraction::flop_isomorphism::enumerate_canonical_flops;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA_POSTFLOP;

/// Snapshot the time-averaged strategy at the root infoset.
/// Mirrors slice7a_convergence_study.rs helper of the same name.
fn snapshot_root_cum_strategy_normalized_avg(
    solver: &FlopStartVectorCfr,
    tree: &solver_core::tree::flat::FlatTree,
) -> Vec<f32> {
    let na = tree.nodes[0].num_children as usize;
    if na == 0 { return vec![]; }
    let nh = solver.num_hands();
    let cum = solver.cum_strategy_flop();
    let local = solver.flop_local_offset()[0];
    let off = local * MAX_NA_POSTFLOP * nh;

    let mut avg = vec![0.0f32; na];
    let mut hands_with_mass = 0usize;
    for h in 0..nh {
        let mut hand_sum = 0.0f32;
        for a in 0..na { hand_sum += cum[off + a * nh + h]; }
        if hand_sum > 0.0 {
            for a in 0..na {
                avg[a] += cum[off + a * nh + h] / hand_sum;
            }
            hands_with_mass += 1;
        }
    }
    if hands_with_mass > 0 {
        for a in 0..na { avg[a] /= hands_with_mass as f32; }
    } else {
        for a in 0..na { avg[a] = 1.0 / na as f32; }
    }
    avg
}

fn l1_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

struct FlopTrajectory {
    flop_index: usize,
    flop_cards: [Card; 3],
    avgs_at_milestones: Vec<(u32, Vec<f32>, f32)>,  // (iter, avg, iter_delta)
    flipped_preferred_action: bool,
    final_iter_delta: f32,
    wall_clock: std::time::Duration,
}

fn run_one_flop_trajectory(
    flop_index: usize,
    flop: [Card; 3],
    iters: u32,
) -> FlopTrajectory {
    let t_total = std::time::Instant::now();

    let combo_ranges: Vec<Vec<f32>> = (0..2)
        .map(|_| vec![1.0f32; NUM_POSSIBLE_HANDS])
        .collect();
    let table = FlopChanceTable::compute_flop_start(&flop, &combo_ranges, 2);

    let flop_cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 6,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![3, 3],
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
    let tree = build_tree(&flop_cfg).expect("flop tree");
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    let na = tree.nodes[0].num_children as usize;
    let mut prev_avg: Option<Vec<f32>> = None;
    let mut avgs_at_milestones = Vec::new();
    let milestones = [1u32, 5, 10, 15, 20, 25, 30];
    let mut preferred_at_10: Option<usize> = None;
    let mut preferred_at_30: Option<usize> = None;
    let mut final_iter_delta = 0.0f32;

    for iter in 0..iters {
        let _ = solver.run(&tree, &game, 1);
        let cur_avg = snapshot_root_cum_strategy_normalized_avg(&solver, &tree);
        let l1_iter_delta = prev_avg.as_ref()
            .map(|p| l1_delta(&cur_avg, p))
            .unwrap_or(f32::INFINITY);

        if milestones.contains(&(iter + 1)) {
            avgs_at_milestones.push((iter + 1, cur_avg.clone(), l1_iter_delta));
        }

        if iter + 1 == 10 {
            preferred_at_10 = Some(cur_avg.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0);
        }
        if iter + 1 == iters {
            preferred_at_30 = Some(cur_avg.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0);
            final_iter_delta = l1_iter_delta;
        }

        prev_avg = Some(cur_avg);
    }
    assert_eq!(na, 2, "test assumes 2-action root (check/bet) for this config");

    FlopTrajectory {
        flop_index,
        flop_cards: flop,
        avgs_at_milestones,
        flipped_preferred_action: preferred_at_10 != preferred_at_30,
        final_iter_delta,
        wall_clock: t_total.elapsed(),
    }
}

#[test]
#[ignore = "Slice 7a follow-up: 6-flop convergence check, ~35 min wall-clock. \
            Run on demand: cargo test --release --test \
            slice7a_multi_flop_convergence_check -- --ignored --nocapture"]
fn slice7a_multi_flop_convergence_distinguishes_close_decision_from_cfr_problem() {
    eprintln!("\n═══ Slice 7a follow-up: multi-flop convergence check ═══");
    eprintln!("Question: is the iter-10→iter-30 preference flip on canonical[0]");
    eprintln!("a flop-specific close-decision (benign) or a CFR convergence");
    eprintln!("problem (not benign, defeats the gate's purpose)?\n");

    let canonicals = enumerate_canonical_flops();
    assert_eq!(canonicals.len(), 1755);
    // Six indices spanning the canonical list. Not archetype-cherry-picked.
    let test_indices = [0usize, 350, 700, 1050, 1400, 1754];

    let mut all_traj = Vec::new();
    for (k, &idx) in test_indices.iter().enumerate() {
        eprintln!("── Flop {}/{}: canonical[{}] = {:?} ──",
            k + 1, test_indices.len(), idx, canonicals[idx]);
        let traj = run_one_flop_trajectory(idx, canonicals[idx], 30);
        eprintln!("  wall-clock: {:?}", traj.wall_clock);
        for (iter, avg, delta) in &traj.avgs_at_milestones {
            eprintln!("    iter {:>2}: avg=[{}], iter-delta={:.4e}",
                iter,
                avg.iter().map(|x| format!("{:.3}", x))
                    .collect::<Vec<_>>().join(", "),
                delta);
        }
        eprintln!("  preference flipped iter-10 to iter-30: {}", traj.flipped_preferred_action);
        eprintln!("  final iter-delta:                       {:.4e}", traj.final_iter_delta);
        eprintln!("");
        all_traj.push(traj);
    }

    // ──────────────────────────────────────────────────────────────
    // Summary table
    // ──────────────────────────────────────────────────────────────
    eprintln!("══ Summary ══");
    eprintln!("  {:<10} {:<10} {:<15} {:<15} {:<10}",
        "flop idx", "flipped?", "iter-10 avg", "iter-30 avg", "final Δ");
    for t in &all_traj {
        let iter10 = t.avgs_at_milestones.iter().find(|(i, _, _)| *i == 10).unwrap();
        let iter30 = t.avgs_at_milestones.iter().find(|(i, _, _)| *i == 30).unwrap();
        eprintln!("  {:<10} {:<10} [{:.3}, {:.3}]  [{:.3}, {:.3}]  {:.4e}",
            t.flop_index,
            if t.flipped_preferred_action { "YES" } else { "no" },
            iter10.1[0], iter10.1[1],
            iter30.1[0], iter30.1[1],
            t.final_iter_delta);
    }

    let n_flipped = all_traj.iter().filter(|t| t.flipped_preferred_action).count();
    let n_drifting = all_traj.iter().filter(|t| t.final_iter_delta > 0.01).count();
    eprintln!("\n  Flops where preferred action flipped iter-10 → iter-30: {}/{}",
        n_flipped, all_traj.len());
    eprintln!("  Flops where iter-delta still > 0.01 at iter 30:          {}/{}",
        n_drifting, all_traj.len());

    eprintln!("\n══ Interpretation ══");
    if n_flipped == all_traj.len() && n_drifting == all_traj.len() {
        eprintln!("  ALL flops show the iter-10→iter-30 flip and still-drifting pattern.");
        eprintln!("  This is NOT a flop-specific close-decision; this is a CFR");
        eprintln!("  convergence signal worth investigating BEFORE sizing slice 7c.");
        eprintln!("  Investigate: DCFR params, strategy update, regret accumulation,");
        eprintln!("  cum_strategy weighting in FlopStartVectorCfr. The CPU is not yet");
        eprintln!("  the hyper-confident reference the gate requires.");
    } else if n_flipped <= 2 {
        eprintln!("  Only {}/{} flops show the flip pattern. Likely benign:", n_flipped, all_traj.len());
        eprintln!("  those flops have close-decision equilibria (~mixed). Slice 7c");
        eprintln!("  sizing should use N estimated from the STABLE flops, not the");
        eprintln!("  flippy ones. Re-run convergence study on a STABLE flop to size N.");
    } else {
        eprintln!("  {}/{} flops show the flip pattern. Inconclusive; warrants a", n_flipped, all_traj.len());
        eprintln!("  longer-iters run (50-100 iters) on the flipping flops to see if");
        eprintln!("  they stabilize at higher N, plus inspection of which flops are");
        eprintln!("  in each bucket (are flippers a recognizable archetype?).");
    }

    // No assertion on n_flipped / n_drifting: this is a diagnostic run.
    // The test passes if it completes; the interpretation above is the deliverable.
}
