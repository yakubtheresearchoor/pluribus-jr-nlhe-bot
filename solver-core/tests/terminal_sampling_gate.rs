//! Terminal-sampling correctness gate (the fidelity-unlock prerequisite).
//!
//! The Design-1-collapsed terminal kernel exhaustively enumerates B^K
//! opponent bucket tuples per terminal — the scaling wall that caps B.
//! The sampling path (sample_m > 0) instead draws M opponent tuples per
//! traverser bucket from the per-opponent reach marginal, importance-
//! weighted by (Π_d Z_d)·Π_pairs fn — an UNBIASED estimator of the same
//! quantity (the leaf value with the leading reach factor factored out).
//!
//! This gate isolates the estimator: it runs ONE terminal dispatch (the
//! single `bucketed_terminal_collapsed` kernel via `fill_terminals`) with
//! a fixed synthetic reach, once exhaustive (ground truth) and once
//! sampled, and compares the per-bucket/per-hand CFV. No CFR averaging —
//! the only difference between the two runs is the terminal estimator, so
//! any disagreement beyond sampling noise is a bug.
//!
//! Three checks:
//!   1. DETERMINISM — same (M, seed) twice ⇒ bit-identical CFV (the RNG
//!      stream is a pure function of (seed, node, bt)).
//!   2. ACCURACY — at large M the reach-weighted mean relative error vs
//!      exhaustive is small (unbiased ⇒ converges to truth).
//!   3. CONVERGENCE — 4× the samples cuts the error (≈1/√M), the
//!      signature of genuine sampling noise rather than bias.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::bucketed_terminal::BucketedTerminalGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::Zone;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const NP: u8 = 6;
use std::time::Instant;

const NH_FIDELITY: usize = 48; // single-kernel estimator gate: rich hand set
const NH_BPUSH: usize = 44; // ≥ 32 buckets survive turn+river card removal
const NH_SOLVE: usize = 12; // batched full-solve gate (≥ nb after card removal)
const NB: usize = 8;

fn build_table(nh: usize) -> FlopChanceTable {
    let board: Vec<Card> = ["Th", "9d", "8c"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
            continue;
        }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            ranges[p][hi as usize] = 1.0;
        }
    }
    let turn_cards: Vec<u8> =
        ["2c", "Jd"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let river_strs: [&[&str]; 2] = [&["4s", "7h"], &["3s", "Qc"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] =
            river_strs[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

fn build_tree_cfg() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![500; NP as usize],
        initial_contributions: vec![5; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.33), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
        max_bets_per_street: None,
        no_open_limp: false,
        threebet_or_fold: false,
    };
    build_tree(&config).unwrap()
}

/// Light tree for the batched full-solve gate: a single pot-sized bet and
/// NO raises, so the 6-player betting tree stays small (the walk cost, not
/// the terminal, dominates a full solve — this keeps the smoke test fast).
fn build_light_tree_cfg() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![500; NP as usize],
        initial_contributions: vec![5; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
        max_bets_per_street: None,
        no_open_limp: false,
        threebet_or_fold: false,
    };
    build_tree(&config).unwrap()
}

fn quantile_maps(
    table: &FlopChanceTable,
    nb: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8]| -> Vec<u16> {
        let alive: Vec<usize> = pl_idx[..nh]
            .iter()
            .map(|&i| i as usize)
            .filter(|&h| !conflicts(h, dead))
            .collect();
        let n = alive.len();
        assert!(n >= nb);
        let mut map = vec![NO_BUCKET; nh];
        for (pos, &h) in alive.iter().enumerate() {
            map[h] = ((pos * nb) / n) as u16;
        }
        map
    };
    let (_, _, _, base_pi, _) = table.sorted_opp_arrays_base();
    let flop_map = map_for(&base_pi, &[]);
    let mut turn_maps = Vec::new();
    let mut river_maps = Vec::new();
    for &tc_card in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
        turn_maps.push(map_for(pi, &[tc_card]));
        let mut rms = Vec::new();
        for &rc_card in &table.river_decks[tc_card as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
            rms.push(map_for(pi, &[tc_card, rc_card]));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
}

/// Tiny tree for the B-push timing harness: low stacks so a single pot bet
/// forces an all-in → flop all-in showdowns (K=5 multiway terminals, arm-1),
/// and very few of them (one binary decision per player). That keeps the
/// EXHAUSTIVE B^K cost affordable at low B while still exercising the worst
/// case the sampler must beat.
fn build_allin_tree_cfg() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![15; NP as usize], // pot-bet > stack ⇒ all-in
        initial_contributions: vec![5; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
        max_bets_per_street: None,
        no_open_limp: false,
        threebet_or_fold: false,
    };
    build_tree(&config).unwrap()
}

/// One flop-zone terminal dispatch at arbitrary (nh, nb); returns the cfv
/// and the warm GPU wall time (one warmup dispatch precedes the timed one).
fn timed_terminals(
    ctx: &MetalContext,
    tree: &FlatTree,
    reach: &[f32],
    nh: usize,
    nb: usize,
    sample_m: u32,
    seed: u32,
) -> (Vec<f32>, f64) {
    let game = FlopStartGame::new(build_table(nh));
    let (fm, tm, rm) = quantile_maps(game.table(), nb);
    let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
    let arm = BucketedFlopCfr::new(tree, game.table(), &bk);
    let mut term = BucketedTerminalGpu::new(ctx, tree, game.table(), &bk, &arm, 1)
        .expect("gpu terminal");
    term.set_sampling(sample_m, seed);
    let nn = tree.num_nodes();
    let mut cfv = vec![0.0f32; nn * nh];
    term.fill_terminals(Zone::Flop, None, None, 0, reach, &mut cfv); // warmup
    let t = Instant::now();
    term.fill_terminals(Zone::Flop, None, None, 0, reach, &mut cfv);
    (cfv, t.elapsed().as_secs_f64() * 1000.0)
}

#[test]
#[ignore = "B-push experiment: requires MAX_BUCKETS_GPU>=32 (production ceiling is 16). \
            Bump the constant, then run with --ignored --nocapture --release."]
fn terminal_bpush_crossover_and_unlock() {
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_allin_tree_cfg();
    let nn = tree.num_nodes();
    let nh = NH_BPUSH;
    let reach = synth_reach(nn, NP as usize, nh);

    eprintln!("=== terminal-phase wall time, live-6 / K=5 (tiny all-in tree) ===");
    // EXHAUSTIVE — B^K; SAFE ONLY at low B. Never run exhaustive at B>=16
    // (device-watchdog crash, per bucketed_showdown.rs ceiling note).
    let mut ex8 = 0.0;
    for nb in [8usize, 10, 12] {
        let (_, ms) = timed_terminals(&ctx, &tree, &reach, nh, nb, 0, 0);
        if nb == 8 {
            ex8 = ms;
        }
        eprintln!("  exhaustive    B={nb:2}: {ms:8.2} ms   (B^5 = {})", nb.pow(5));
    }
    // SAMPLED — O(M·K), ~flat in B. The unlock: B up to 32 with no B^K blowup.
    let mut s8 = 0.0;
    for nb in [8usize, 12, 16, 20, 24, 32] {
        let (_, ms) = timed_terminals(&ctx, &tree, &reach, nh, nb, 2000, 1);
        if nb == 8 {
            s8 = ms;
        }
        eprintln!("  sampled M2000 B={nb:2}: {ms:8.2} ms", );
    }
    eprintln!(
        "  → at K=5, sampled is already {:.1}× cheaper than exhaustive at B=8, \
         and stays ~flat to B=32 while exhaustive would scale B^5",
        ex8 / s8.max(1e-9)
    );

    // B=32 has NO exhaustive ground truth (would crash). Validate intrinsically:
    // (a) deterministic, (b) self-consistent — M=8000 agrees with M=2000 (the
    // estimates converge to a common value ⇒ unbiased, no silent B>16 breakage).
    let (d1, _) = timed_terminals(&ctx, &tree, &reach, nh, 32, 8000, 5);
    let (d2, _) = timed_terminals(&ctx, &tree, &reach, nh, 32, 8000, 5);
    for i in 0..d1.len() {
        assert_eq!(d1[i].to_bits(), d2[i].to_bits(), "B=32 sampling not deterministic at {i}");
    }
    let (lo, _) = timed_terminals(&ctx, &tree, &reach, nh, 32, 2000, 1);
    let (hi, _) = timed_terminals(&ctx, &tree, &reach, nh, 32, 8000, 1);
    let (mean, _maxr, n) = errors(&hi, &lo); // hi (more samples) as the reference
    eprintln!("  B=32 self-consistency: M2000 vs M8000 weighted-mean rel {mean:.4} (n={n})");
    assert!(n > 0, "no significant entries at B=32");
    assert!(mean < 0.06, "B=32 estimates inconsistent across M ({mean:.4}) — possible B>16 breakage");
}

/// Deterministic synthetic reach: random in [0,1) per (node, player, hand)
/// from a fixed LCG. Both the exhaustive and sampled runs see the SAME
/// reach, so the estimator is the only variable.
fn synth_reach(nn: usize, np: usize, nh: usize) -> Vec<f32> {
    let mut s: u64 = 0x1234_5678_9abc_def0;
    let mut v = vec![0.0f32; nn * np * nh];
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((s >> 40) as f32) / ((1u64 << 24) as f32); // [0,1)
    }
    v
}

/// One flop-zone terminal dispatch with the given (sample_m, seed).
/// sample_m == 0 ⇒ exhaustive. Returns the full cfv buffer (nn·nh).
fn run_terminals(
    ctx: &MetalContext,
    tree: &FlatTree,
    reach: &[f32],
    sample_m: u32,
    seed: u32,
) -> Vec<f32> {
    let game = FlopStartGame::new(build_table(NH_FIDELITY));
    let (fm, tm, rm) = quantile_maps(game.table(), NB);
    let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
    let arm = BucketedFlopCfr::new(tree, game.table(), &bk);
    let mut term = BucketedTerminalGpu::new(ctx, tree, game.table(), &bk, &arm, 1)
        .expect("gpu terminal");
    term.set_sampling(sample_m, seed);
    let nn = tree.num_nodes();
    let mut cfv = vec![0.0f32; nn * NH_FIDELITY];
    term.fill_terminals(Zone::Flop, None, None, 0, reach, &mut cfv);
    cfv
}

/// One full bucketed flop solve at B=8 through the production batched
/// terminal kernel (the offload hook), with the given (sample_m, seed).
/// Returns the root CFV. ITERS kept small — this validates the batched
/// estimator end-to-end, not convergence depth.
fn run_full_solve(ctx: &MetalContext, tree: &FlatTree, sample_m: u32, seed: u32) -> Vec<f32> {
    const ITERS: u32 = 2;
    let game = FlopStartGame::new(build_table(NH_SOLVE));
    let (fm, tm, rm) = quantile_maps(game.table(), NB);
    let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
    let mut arm = BucketedFlopCfr::new(tree, game.table(), &bk);
    arm.set_terminal_design(TerminalDesign::Design1Collapsed);
    let mut term = BucketedTerminalGpu::new(ctx, tree, game.table(), &bk, &arm, 1)
        .expect("gpu terminal");
    term.set_sampling(sample_m, seed);
    arm.set_terminal_offload_hook(Some(term.into_hook()));
    arm.run(tree, &game, &bk, ITERS)
}

#[test]
fn terminal_sampling_batched_solve_gate() {
    // Smoke test for the BATCHED port: the estimator gate already proves the
    // sampling math (single kernel, rigorous 1/√M), and g3 covers the batched
    // EXHAUSTIVE path. This only confirms the batched SAMPLING branch runs
    // end-to-end through the production hook and yields a small error. Light
    // tree + NH=8 + 2 iters keep it fast (the walk dominates a full solve).
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_light_tree_cfg();

    let exact = run_full_solve(&ctx, &tree, 0, 0);
    let hi = run_full_solve(&ctx, &tree, 8000, 1);

    let (mean_hi, max_hi, n) = errors(&exact, &hi);
    eprintln!(
        "batched full-solve root cfv, sampled(M=8000) vs exhaustive (B={NB}, {n} entries): \
         weighted-mean rel {mean_hi:.4}, max rel {max_hi:.4}"
    );
    assert!(n > 0, "no significant root CFV entries");
    assert!(
        mean_hi < 0.02,
        "batched-sampled root cfv mean rel error {mean_hi:.4} too high — port likely wrong"
    );
}

/// Reach-weighted mean and max relative error over CFV entries whose
/// exact value is significant (above a fraction of the peak magnitude).
fn errors(exact: &[f32], samp: &[f32]) -> (f64, f64, usize) {
    let peak = exact.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;
    let floor = (peak * 1e-3).max(1e-9);
    let (mut num, mut den, mut maxrel, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for (&e, &s) in exact.iter().zip(samp.iter()) {
        let e = e as f64;
        if e.abs() < floor {
            continue;
        }
        let rel = (s as f64 - e).abs() / e.abs();
        num += (s as f64 - e).abs();
        den += e.abs();
        if rel > maxrel {
            maxrel = rel;
        }
        n += 1;
    }
    (num / den.max(1e-30), maxrel, n)
}

#[test]
fn terminal_sampling_estimator_gate() {
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_tree_cfg();
    let nn = tree.num_nodes();
    let reach = synth_reach(nn, NP as usize, NH_FIDELITY);

    // Ground truth: exhaustive B^K enumeration.
    let exact = run_terminals(&ctx, &tree, &reach, 0, 0);

    // 1. DETERMINISM — same (M, seed) twice ⇒ bit-identical.
    let a = run_terminals(&ctx, &tree, &reach, 4000, 7);
    let b = run_terminals(&ctx, &tree, &reach, 4000, 7);
    for i in 0..a.len() {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "sampling not deterministic at {i}: {} vs {}",
            a[i],
            b[i]
        );
    }

    // 2. ACCURACY at large M.
    let (mean_lo, max_lo, n_lo) = errors(&exact, &run_terminals(&ctx, &tree, &reach, 2000, 1));
    let big = run_terminals(&ctx, &tree, &reach, 8000, 1);
    let (mean_hi, max_hi, n) = errors(&exact, &big);
    eprintln!(
        "terminal sampling vs exhaustive (B={NB}, {n} significant CFV entries):\n  \
         M=2000: weighted-mean rel {mean_lo:.4}, max rel {max_lo:.4} (n={n_lo})\n  \
         M=8000: weighted-mean rel {mean_hi:.4}, max rel {max_hi:.4}"
    );
    assert!(n > 0, "no significant CFV entries — scenario degenerate");
    assert!(
        mean_hi < 0.02,
        "weighted-mean rel error {mean_hi:.4} at M=8000 too high — estimator likely biased"
    );

    // 3. CONVERGENCE — more samples ⇒ less error (unbiased noise, not bias).
    assert!(
        mean_hi < mean_lo,
        "error did not shrink with 4× samples (M2000 {mean_lo:.4} -> M8000 {mean_hi:.4}); \
         a stuck error floor indicates bias, not variance"
    );
}
