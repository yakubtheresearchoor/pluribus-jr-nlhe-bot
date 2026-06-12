//! V1 FIDELITY-LEVER MEASUREMENTS (2026-06-12, user: "run through all
//! of the levers one at a time — measure, build, test, remeasure, and
//! make sure we're not bleeding fidelity at an unacceptable rate").
//!
//! The bill (audited, 33e13e6): 1,217 GPU-h at uniform full fidelity,
//! 96% in live-5/6. Each lever gets its own test: COST side measured
//! directly, QUALITY side in the project's established currency —
//! mean |Δσ| over normalized FLOP-ZONE strategy rows (bars: blueprint
//! convergence median 0.00122; runout-draw noise floor 0.0657 from
//! the cell probe; substitution residual +0.092) — or exact
//! exploitability where the HU machinery applies.
//!
//! Levers:
//!   L3 iterations (the ×34 convention — also the open bill multiplier)
//!   L1 runout fidelity (4×4 → 2×2 → 1×1) on live-5/6
//!   L4 bucket count (B=8 → B=4) on live-6 — terminal work ~ B^K
//!   L2 flop coverage (neighbor substitution) re-validated on v1 trees
//!   L6 small-tree batching (dispatch-floor recovery — engineering)
//!   L7 warm-start promise (CPU, neighbor-flop seed)
//!   L5 reach-weighting: requires the preflop bootstrap; priced, not
//!      run here.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, Card};
use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};
use std::time::Instant;

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

/// n_turn × n_river runout policy at deterministic deck positions (the
/// pricing convention, generalized to 1/2/4 per street).
fn table_for(flop: [Card; 3], np: u8, nt: usize, nr: usize) -> FlopChanceTable {
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let tp: &[usize] = match nt {
        1 => &[12],
        2 => &[12, 36],
        4 => &[6, 18, 30, 42],
        _ => unreachable!(),
    };
    let rp: &[usize] = match nr {
        1 => &[10],
        2 => &[10, 30],
        4 => &[8, 20, 32, 44],
        _ => unreachable!(),
    };
    let turn_cards: Vec<u8> = tp.iter().map(|&p| deck[p]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let rdeck: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        river_decks[tc as usize] = rp.iter().map(|&p| rdeck[p]).collect();
    }
    FlopChanceTable::build_full_nh_sampled(flop, np, &turn_cards, &river_decks)
}

fn flop(name: [&str; 3]) -> [Card; 3] {
    [
        card_from_str(name[0]).unwrap(),
        card_from_str(name[1]).unwrap(),
        card_from_str(name[2]).unwrap(),
    ]
}

fn seam_tree(live: u8, commit: i32, pot: i32) -> FlatTree {
    let spec = production_game_v1();
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    build_tree(&spec.flop_seam_config(live, commit, pot, bets)).expect("seam tree")
}

/// Solve on GPU-native; return (per-iter seconds, normalized flop-zone
/// σ rows). σ rows are per (flop infoset, bucket): the normalized
/// average strategy over actions — the conv-stat / cell-arc currency.
fn solve_sigma(
    ctx: &MetalContext,
    tree: &FlatTree,
    f: [Card; 3],
    live: u8,
    nt: usize,
    nr: usize,
    nb: usize,
    iters: u32,
) -> (f64, Vec<Vec<f32>>) {
    let table = table_for(f, live, nt, nr);
    let (fm, tm, rm) = quantile_maps(&table, nb);
    let game = FlopStartGame::new(table);
    let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
    let mut solver = BucketedFlopCfr::new(tree, game.table(), &bk);
    solver.set_terminal_design(TerminalDesign::Design1Collapsed);
    let stripes = (32 / nb).max(1) as u32;
    let mut native = BucketedNativeGpu::new(ctx, tree, game.table(), &bk, &solver, stripes)
        .expect("native");
    let t0 = Instant::now();
    native.run(iters);
    let per_iter = t0.elapsed().as_secs_f64() / iters as f64;

    let cum = native.cum_strategy_flop();
    let rows = cum.len() / (MAX_NA_POSTFLOP * nb);
    let mut sigma: Vec<Vec<f32>> = Vec::with_capacity(rows * nb);
    for r in 0..rows {
        let base = r * MAX_NA_POSTFLOP * nb;
        for b in 0..nb {
            let mut v: Vec<f32> =
                (0..MAX_NA_POSTFLOP).map(|a| cum[base + a * nb + b].max(0.0)).collect();
            let s: f32 = v.iter().sum();
            if s > 0.0 {
                for x in v.iter_mut() {
                    *x /= s;
                }
            }
            sigma.push(v);
        }
    }
    (per_iter, sigma)
}

/// Mean |Δσ| over rows where EITHER side has mass (the cell-arc
/// convention).
fn mean_dsigma(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    assert_eq!(a.len(), b.len(), "σ row count must match for comparison");
    let mut tot = 0.0f64;
    let mut n = 0usize;
    for (ra, rb) in a.iter().zip(b.iter()) {
        let ma: f32 = ra.iter().sum();
        let mb: f32 = rb.iter().sum();
        if ma == 0.0 && mb == 0.0 {
            continue;
        }
        for (x, y) in ra.iter().zip(rb.iter()) {
            tot += (*x as f64 - *y as f64).abs();
        }
        n += 1;
    }
    tot / n.max(1) as f64
}

/// ═══ L3: the ×34-iteration convention (open bill multiplier) ═══
/// Δσ(34 vs 136) ≈ how much the strategy still moves past 34 iters.
/// Bar: the blueprint banked at conv median 0.00122 per-iter delta;
/// if σ(34) is within noise of σ(136), 34 is validated on corrected
/// trees; if Δσ(17 vs 136) is also small, rare buckets can take 17.
#[test]
#[ignore = "L3 iterations; --ignored --nocapture --release --features metal"]
fn lever3_iterations() {
    let ctx = MetalContext::new().expect("Metal");
    let f0 = flop(["2h", "7d", "Ks"]);
    for (live, commit, pot, label) in
        [(4u8, 7i32, 29i32, "live-4 raised"), (6, 2, 12, "live-6 limp")]
    {
        let tree = seam_tree(live, commit, pot);
        let (_, s17) = solve_sigma(&ctx, &tree, f0, live, 4, 4, 8, 17);
        let (_, s34) = solve_sigma(&ctx, &tree, f0, live, 4, 4, 8, 34);
        let (_, s68) = solve_sigma(&ctx, &tree, f0, live, 4, 4, 8, 68);
        let (_, s136) = solve_sigma(&ctx, &tree, f0, live, 4, 4, 8, 136);
        eprintln!(
            "L3 {label}: Δσ(17↔136) {:.4} | Δσ(34↔136) {:.4} | Δσ(68↔136) {:.4}",
            mean_dsigma(&s17, &s136),
            mean_dsigma(&s34, &s136),
            mean_dsigma(&s68, &s136)
        );
    }
}

/// ═══ L1: runout fidelity on the expensive families ═══
/// Cost: s/iter at 1×1 / 2×2 / 4×4. Quality: flop-zone Δσ vs the 4×4
/// reference (same flop maps; bar = the 0.0657 runout-draw noise floor
/// and the cell-arc precedent 1×1 0.105 / 2×2 0.034).
#[test]
#[ignore = "L1 runouts; --ignored --nocapture --release --features metal"]
fn lever1_runout_fidelity() {
    let ctx = MetalContext::new().expect("Metal");
    let f0 = flop(["2h", "7d", "Ks"]);
    for (live, commit, pot, label) in
        [(5u8, 2i32, 10i32, "live-5 limp"), (6, 2, 12, "live-6 limp")]
    {
        let tree = seam_tree(live, commit, pot);
        let (c44, s44) = solve_sigma(&ctx, &tree, f0, live, 4, 4, 8, 34);
        let (c22, s22) = solve_sigma(&ctx, &tree, f0, live, 2, 2, 8, 34);
        let (c11, s11) = solve_sigma(&ctx, &tree, f0, live, 1, 1, 8, 34);
        eprintln!(
            "L1 {label}: s/iter 4×4 {c44:.3} | 2×2 {c22:.3} (÷{:.1}) | 1×1 {c11:.3} (÷{:.1})",
            c44 / c22,
            c44 / c11
        );
        eprintln!(
            "L1 {label}: Δσ(2×2↔4×4) {:.4} | Δσ(1×1↔4×4) {:.4}  [floor 0.0657]",
            mean_dsigma(&s22, &s44),
            mean_dsigma(&s11, &s44)
        );
    }
}

/// ═══ L4: bucket count at live-6 (terminal work ~ B^K: (8/4)^5 = 32×
/// on terminals) ═══ Cost measured here; QUALITY cannot be compared by
/// Δσ across B (different infoset granularity) — it rides the banked
/// research curves (B=5 vs B=8: wet 10.29 vs 7.14 %pot, dry 7.79 vs
/// 7.70) and the harness head-to-head, family-stratified.
#[test]
#[ignore = "L4 bucket count; --ignored --nocapture --release --features metal"]
fn lever4_bucket_count_live6() {
    let ctx = MetalContext::new().expect("Metal");
    let f0 = flop(["2h", "7d", "Ks"]);
    let tree = seam_tree(6, 2, 12);
    let (c8, _) = solve_sigma(&ctx, &tree, f0, 6, 4, 4, 8, 6);
    let (c4, _) = solve_sigma(&ctx, &tree, f0, 6, 4, 4, 4, 6);
    eprintln!(
        "L4 live-6 limp: B=8 {c8:.3} s/iter | B=4 {c4:.3} s/iter (÷{:.1}) | \
         row 170h → {:.1}h | quality rides banked B-curves + stratified harness",
        c8 / c4,
        170.0 * c4 / c8
    );
}

/// ═══ L2: flop coverage (neighbor substitution), re-validated on v1
/// trees ═══ The cell probe banked +0.092-0.099 Δσ excess for
/// neighbor substitution on the OLD trees. Re-measure on a v1 cell:
/// Δσ between the cell solved on flop A vs solved on a NEAR-neighbor
/// flop B (one rank apart) and a FAR flop C — using A's strategy for
/// B is the substitution; its residual is Δσ(A↔B), read against the
/// runout noise floor and the far-pair scale.
#[test]
#[ignore = "L2 neighbor substitution; --ignored --nocapture --release --features metal"]
fn lever2_neighbor_substitution() {
    let ctx = MetalContext::new().expect("Metal");
    let tree = seam_tree(4, 7, 29);
    let fa = flop(["2h", "7d", "Ks"]);
    let fb = flop(["2h", "8d", "Ks"]); // near: one rank
    let fc = flop(["Ah", "Kh", "Qh"]); // far: monotone broadway
    let (_, sa) = solve_sigma(&ctx, &tree, fa, 4, 4, 4, 8, 34);
    let (_, sb) = solve_sigma(&ctx, &tree, fb, 4, 4, 4, 8, 34);
    let (_, sc) = solve_sigma(&ctx, &tree, fc, 4, 4, 4, 8, 34);
    eprintln!(
        "L2 live-4: Δσ(near 7d↔8d) {:.4} | Δσ(far 2h7dKs↔AhKhQh) {:.4}  \
         [old-tree banked substitution ≈ 0.092-0.099; noise floor 0.0657]",
        mean_dsigma(&sa, &sb),
        mean_dsigma(&sa, &sc)
    );
    // live-6 (the family the lever is FOR — measured at the 1×1
    // fidelity it will be stacked with).
    let t6 = seam_tree(6, 2, 12);
    let (_, sa6) = solve_sigma(&ctx, &t6, fa, 6, 1, 1, 8, 34);
    let (_, sb6) = solve_sigma(&ctx, &t6, fb, 6, 1, 1, 8, 34);
    eprintln!(
        "L2 live-6 @1×1 (the stack): Δσ(near 7d↔8d) {:.4}  [floor 0.0657]",
        mean_dsigma(&sa6, &sb6)
    );
}

/// ═══ L6: small-tree batching (dispatch-floor recovery) ═══ Tiny
/// buckets are dispatch-bound (floor ~0.0125 s/iter, GPU NOT
/// saturated), so running K flops concurrently (per-thread contexts,
/// the BP_GPU precedent) should multiply throughput where wall=busy
/// said concurrency was dead for big trees. Measure aggregate
/// throughput at K = 1, 4, 8 on the 102-node live-3 cell.
#[test]
#[ignore = "L6 batching; --ignored --nocapture --release --features metal"]
fn lever6_small_tree_batching() {
    use std::sync::Arc;
    let tree = Arc::new(seam_tree(3, 80, 244));
    let flops: Vec<[Card; 3]> = [
        ["2h", "7d", "Ks"],
        ["Ah", "Kh", "Qh"],
        ["9c", "9d", "2s"],
        ["6h", "7h", "8s"],
        ["As", "2d", "7c"],
        ["Td", "Jd", "Qc"],
        ["3c", "3d", "3h"],
        ["Kc", "8h", "2d"],
    ]
    .iter()
    .map(|f| flop(*f))
    .collect();
    const ITERS: u32 = 200;
    for k in [1usize, 4, 8] {
        let t0 = Instant::now();
        let handles: Vec<_> = (0..k)
            .map(|i| {
                let tree = Arc::clone(&tree);
                let f = flops[i % flops.len()];
                std::thread::spawn(move || {
                    let ctx = MetalContext::new().expect("Metal");
                    let table = table_for(f, 3, 4, 4);
                    let (fm, tm, rm) = quantile_maps(&table, 8);
                    let game = FlopStartGame::new(table);
                    let bk = FlopBucketing::from_maps(game.table(), 8, 8, 8, fm, tm, rm);
                    let mut s = BucketedFlopCfr::new(&tree, game.table(), &bk);
                    s.set_terminal_design(TerminalDesign::Design1Collapsed);
                    let mut native =
                        BucketedNativeGpu::new(&ctx, &tree, game.table(), &bk, &s, 4)
                            .expect("native");
                    native.run(ITERS);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let wall = t0.elapsed().as_secs_f64();
        eprintln!(
            "L6 K={k}: wall {wall:.2}s for {k}×{ITERS} iters → aggregate \
             {:.4} s/iter-solve",
            wall / (k as f64 * ITERS as f64),
        );
    }
}

/// ═══ L7: warm-start promise (CPU) ═══ Seed flop B's solve with flop
/// A's converged state; how many iters does B need to reach the σ it
/// reaches cold at 34? Measured as Δσ(warm-k ↔ cold-34) for k ∈
/// {8, 17}; if warm-8 ≈ cold-34, the fill cost ÷4 on iteration count.
#[test]
#[ignore = "L7 warm-start; --ignored --nocapture --release"]
fn lever7_warm_start_cpu() {
    let tree = seam_tree(4, 7, 29);
    let fa = flop(["2h", "7d", "Ks"]);
    let fb = flop(["2h", "8d", "Ks"]);

    let solve_cpu = |f: [Card; 3],
                     iters: u32,
                     seed: Option<(&[f32], &[f32], &[f32], &[f32], &[f32], &[f32])>|
     -> (BucketedFlopCfr, Vec<Vec<f32>>) {
        let table = table_for(f, 4, 4, 4);
        let (fm, tm, rm) = quantile_maps(&table, 8);
        let game = FlopStartGame::new(table);
        let bk = FlopBucketing::from_maps(game.table(), 8, 8, 8, fm, tm, rm);
        let mut s = BucketedFlopCfr::new(&tree, game.table(), &bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        if let Some((rf, cf, rt, ct, rr, cr)) = seed {
            s.regrets_flop_mut().copy_from_slice(rf);
            s.cum_strategy_flop_mut().copy_from_slice(cf);
            s.regrets_turn_mut().copy_from_slice(rt);
            s.cum_strategy_turn_mut().copy_from_slice(ct);
            s.regrets_river_mut().copy_from_slice(rr);
            s.cum_strategy_river_mut().copy_from_slice(cr);
        }
        let _ = s.run(&tree, &game, &bk, iters);
        let nb = 8usize;
        let cum = s.cum_strategy_flop().to_vec();
        let rows = cum.len() / (MAX_NA_POSTFLOP * nb);
        let mut sigma = Vec::with_capacity(rows * nb);
        for r in 0..rows {
            let base = r * MAX_NA_POSTFLOP * nb;
            for b in 0..nb {
                let mut v: Vec<f32> =
                    (0..MAX_NA_POSTFLOP).map(|a| cum[base + a * nb + b].max(0.0)).collect();
                let sm: f32 = v.iter().sum();
                if sm > 0.0 {
                    for x in v.iter_mut() {
                        *x /= sm;
                    }
                }
                sigma.push(v);
            }
        }
        (s, sigma)
    };

    let (sa, _) = solve_cpu(fa, 34, None);
    let seed = (
        sa.regrets_flop().to_vec(),
        sa.cum_strategy_flop().to_vec(),
        sa.regrets_turn().to_vec(),
        sa.cum_strategy_turn().to_vec(),
        sa.regrets_river().to_vec(),
        sa.cum_strategy_river().to_vec(),
    );
    let (_, cold34) = solve_cpu(fb, 34, None);
    let (_, cold136) = solve_cpu(fb, 136, None);
    for k in [8u32, 17] {
        let (_, warm) = solve_cpu(
            fb,
            k,
            Some((&seed.0, &seed.1, &seed.2, &seed.3, &seed.4, &seed.5)),
        );
        eprintln!(
            "L7: Δσ(warm-{k} ↔ cold-136) {:.4} vs Δσ(cold-34 ↔ cold-136) {:.4}",
            mean_dsigma(&warm, &cold136),
            mean_dsigma(&cold34, &cold136)
        );
    }
}
