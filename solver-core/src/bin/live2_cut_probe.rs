//! LIVE-2 CUT PROBE (2026-06-17): size the achievable live-2 cost cut. Bucketing
//! HU is NOT available — bucketed_showdown_cfv asserts np>=3 and BucketedFlopCfr
//! routes all terminals through it, so HU is forced to the EXACT O(nh²) sweep
//! (11.4s/34-iter, the breakdown's 66%). The lever WITHOUT new code is iteration
//! count: the search-backstop argument says the blueprint HU only needs to SEED
//! live search, not converge. So: how cheap can we go, and how degraded is the
//! seed vs a near-converged reference?
//!
//! Measures exact-HU solve time + the flop-root CFV distance from a 100-iter
//! reference, at 34 / 16 / 8 / 4 iters. (The bigger lever — a bucketed HU
//! showdown, O(nb²) — is flagged but unbuilt; it needs the np>=3 gate lifted.)

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::{extract_v_flop_root_from_cum, PreflopChanceTable};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

fn main() {
    let spec = production_game_v1();
    let np = 2u8;
    let ptable = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let canonical = ptable.canonical_flops[0];
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let turns = vec![deck[0]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[deck[0] as usize] = vec![deck[1]];

    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let tree = build_tree(&spec.flop_seam_config(np, 2, 12, bets)).unwrap();

    let solve = |iters: u32| -> (Vec<f32>, f64) {
        let t = FlopChanceTable::build_full_nh_sampled(canonical, np, &turns, &river_decks);
        let game = FlopStartGame::new(t);
        let mut s = FlopStartVectorCfr::new(&tree, game.table());
        let t0 = Instant::now();
        s.run(&tree, &game, iters);
        let secs = t0.elapsed().as_secs_f64();
        let v = extract_v_flop_root_from_cum(&mut s, &tree, &game, 0).0;
        (v, secs)
    };
    let dist = |a: &[f32], b: &[f32]| -> (f64, f64) {
        let n = a.len().max(1);
        let mad = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / n as f64;
        let mx = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max);
        (mad, mx)
    };

    eprintln!("live-2 cut probe: exact HU, flop {:?}, nh-scale\n", canonical);
    let (vref, tref) = solve(100);
    let scale = vref.iter().cloned().fold(0.0f32, |a, x| a.max(x.abs()));
    eprintln!("reference 100 iters: {tref:.2}s | CFV |mag| {scale:.2}\n");
    eprintln!("{:>6} {:>9} {:>14} {:>14} {:>12}", "iters", "time s", "CFV mean|Δ|", "CFV max|Δ|", "vs 11.4s");
    for iters in [34u32, 16, 8, 4] {
        let (v, secs) = solve(iters);
        let (mad, mx) = dist(&v, &vref);
        eprintln!("{iters:>6} {secs:>9.2} {mad:>14.4} {mx:>14.4} {:>11.1}x", 11.4 / secs.max(1e-9));
    }
    eprintln!("\n(quality = distance of the reduced-iter seed from the 100-iter near-equilibrium;");
    eprintln!(" small Δ ⇒ the cheap seed is close, so live search has little to recover.");
    eprintln!(" BIGGER lever, unbuilt: bucketed HU showdown O(nb²) — needs the np>=3 gate lifted.)");
}
