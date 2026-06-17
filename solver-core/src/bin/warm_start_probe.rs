//! SAME-SUBGAME WARM-START PROBE (2026-06-16): the feasibility lever for the
//! joint solve. The cold solve re-runs each subgame 34 iters from scratch every
//! preflop iteration (~80k solves/iter → days). Warm-start instead carries each
//! subgame's OWN regrets across iterations (~1 iter/subgame/iter → ~30× cheaper).
//! The worry (from L7/d75fbc5): regrets accumulated under drifting ranges might
//! bias the result — but that finding was CROSS-FLOP seeding; same-subgame
//! carry-forward was explicitly left untested. This probe tests it directly.
//!
//! Setup (exact HU, the expensive cell): a subgame whose opponent ranges DRIFT
//! uniform→tight over T_drift iters, then HOLD at tight for T_hold. Solve it:
//!   WARM  — one solver, 1 iter/step, regrets accumulate across the drift.
//!   COLD  — fresh solver, T_drift+T_hold iters all at the FINAL (tight) range.
//! Compare the deployed (avg-strategy) flop-root CFV. CONTROL: cold@uniform vs
//! cold@tight = the "range effect" magnitude — so warm being stuck at the START
//! range can't masquerade as convergence.
//!
//! VERDICT: warm ≪ range-effect from cold@tight ⇒ warm tracked to the final
//! equilibrium (contamination washed out) ⇒ lever VALID. warm ≈ range-effect
//! ⇒ stuck/biased ⇒ garbage like cross-flop.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{card_pair_to_index, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::{extract_v_flop_root_from_cum, PreflopChanceTable};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn main() {
    let t_drift: u32 = std::env::var("WS_DRIFT").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let t_hold: u32 = std::env::var("WS_HOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let np = 2u8;
    let spec = production_game_v1();

    let ptable = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let canonical = ptable.canonical_flops[0];
    let board: Vec<Card> = canonical.to_vec();

    // seeded 1×1 runout (matches the fill convention)
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let tc = deck[(splitmix(0) % deck.len() as u64) as usize];
    let rpool: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
    let rc = rpool[(splitmix(0xDEAD) % rpool.len() as u64) as usize];
    let turns = vec![tc];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[tc as usize] = vec![rc];

    // valid (flop-unblocked) hand indices
    let chosen: Vec<u16> = (0..NUM_POSSIBLE_HANDS)
        .filter(|&idx| {
            let (c1, c2) = index_to_card_pair(idx);
            bm & ((1u64 << c1) | (1u64 << c2)) == 0
        })
        .map(|i| i as u16)
        .collect();

    // ranges over all 1326 combos (subset builder selects via `chosen`).
    let uniform_r: Vec<f32> = vec![1.0; NUM_POSSIBLE_HANDS];
    // TIGHT (extreme): pairs or both-broadway (T+) only, else near-zero —
    // indexed via card_pair_to_index (the scheme the table builder reads), like
    // the working range_dependence_probe, so the weights aren't scrambled.
    let mut tight_r: Vec<f32> = vec![0.0; NUM_POSSIBLE_HANDS]; // pure: trash = 0
    for c1 in 0..52u8 {
        for c2 in (c1 + 1)..52u8 {
            let (r1, r2) = (c1 >> 2, c2 >> 2); // rank index 0=2 .. 8=T .. 12=A
            if r1 == r2 || (r1 >= 8 && r2 >= 8) {
                tight_r[card_pair_to_index(c1, c2)] = 1.0;
            }
        }
    }
    let _ = index_to_card_pair; // (still used below for `chosen`)
    let lerp = |a: &[f32], b: &[f32], t: f32| -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x * (1.0 - t) + y * t).collect()
    };
    // traverser (player 0) stays uniform; only the OPPONENT (player 1) drifts —
    // isolates the range effect on the traverser's value.
    let opp_range = |r1: &[f32]| -> Vec<Vec<f32>> { vec![uniform_r.clone(), r1.to_vec()] };

    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let tree = build_tree(&spec.flop_seam_config(np, 2, 12, bets)).unwrap();

    let build_game = |ranges: &[Vec<f32>]| -> FlopStartGame {
        let table = FlopChanceTable::compute_flop_start_subset_with_decks(
            &board, ranges, np, &chosen, &turns, &river_decks,
        );
        FlopStartGame::new(table)
    };
    let solve_cold = |ranges: &[Vec<f32>], iters: u32| -> Vec<f32> {
        let game = build_game(ranges);
        let mut s = FlopStartVectorCfr::new(&tree, game.table());
        s.run(&tree, &game, iters);
        extract_v_flop_root_from_cum(&mut s, &tree, &game, 0).0
    };

    let total = t_drift + t_hold;
    eprintln!("warm-start probe: HU exact, drift {t_drift} + hold {t_hold} = {total} iters\n");

    // references (1×1 sampled — what the fill/joint solve use)
    let v_cold_final = solve_cold(&opp_range(&tight_r), total);
    let v_cold_start = solve_cold(&opp_range(&uniform_r), total);

    // CONTROL: FULL-BOARD range effect on the SAME flop (range_dependence_probe
    // found this large). If full-board moves but 1×1 doesn't, 1×1 sampling is
    // killing range-sensitivity; if both flat, it's an extraction artifact.
    let solve_cold_full = |ranges: &[Vec<f32>], iters: u32| -> Vec<f32> {
        let table = FlopChanceTable::compute_flop_start(&board, ranges, np);
        let game = FlopStartGame::new(table);
        let mut s = FlopStartVectorCfr::new(&tree, game.table());
        s.run(&tree, &game, iters);
        extract_v_flop_root_from_cum(&mut s, &tree, &game, 0).0
    };
    let fb_iters: u32 = std::env::var("WS_FB_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let v_fb_final = solve_cold_full(&opp_range(&tight_r), fb_iters);
    let v_fb_start = solve_cold_full(&opp_range(&uniform_r), fb_iters);

    // WARM: one solver, drift then hold, regrets accumulating.
    let game0 = build_game(&opp_range(&uniform_r));
    let mut warm = FlopStartVectorCfr::new(&tree, game0.table());
    let mut last_game = game0;
    for t in 1..=t_drift {
        let frac = t as f32 / t_drift as f32;
        let game_t = build_game(&opp_range(&lerp(&uniform_r, &tight_r, frac)));
        warm.run(&tree, &game_t, 1);
        last_game = game_t;
    }
    for _ in 0..t_hold {
        let game_t = build_game(&opp_range(&tight_r));
        warm.run(&tree, &game_t, 1);
        last_game = game_t;
    }
    let v_warm = extract_v_flop_root_from_cum(&mut warm, &tree, &last_game, 0).0;

    let dist = |a: &[f32], b: &[f32]| -> (f64, f64) {
        let n = a.len().max(1);
        let mad = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / n as f64;
        let mx = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max);
        (mad, mx)
    };
    let (range_mad, range_mx) = dist(&v_cold_start, &v_cold_final);
    let (fb_mad, fb_mx) = dist(&v_fb_start, &v_fb_final);
    let (warm_mad, warm_mx) = dist(&v_warm, &v_cold_final);
    let ratio = warm_mad / range_mad.max(1e-9);

    eprintln!("FULL-BOARD range effect (control, {fb_iters} it): mean {fb_mad:.4}  max {fb_mx:.4}");
    eprintln!("RANGE EFFECT  (cold@uniform vs cold@tight): mean {range_mad:.4}  max {range_mx:.4}");
    eprintln!("WARM-START    (warm        vs cold@tight):  mean {warm_mad:.4}  max {warm_mx:.4}");
    eprintln!("ratio warm/range = {ratio:.3}\n");
    if range_mad < 1e-3 {
        eprintln!("INCONCLUSIVE: range effect too small to test (strengthen the tight range).");
    } else if ratio < 0.15 {
        eprintln!("VERDICT: warm tracked to the FINAL equilibrium (drift contamination washed out) → LEVER VALID.");
    } else if ratio > 0.6 {
        eprintln!("VERDICT: warm STUCK/BIASED (did not track to final) → garbage like cross-flop. Cold required.");
    } else {
        eprintln!("VERDICT: AMBIGUOUS partial tracking — tune T_hold / inspect before trusting.");
    }
}
