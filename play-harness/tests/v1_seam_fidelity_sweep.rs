//! STEP 3 — the multiway-fidelity sweep (the number the bucketing crisis has
//! waited on). Per family, distance-from-exact (exploitability % pot of the
//! bucketed lift vs the exact ceiling) as a function of compression RATIO
//! (nh/nb — comparable across families' nh). Anchored to exact (not relative
//! standings) so it forks cleanly: if every candidate is far from exact, the
//! abstraction itself is inadequate; if reach-weighted is close, bucket count
//! suffices. live-6 is the equity ROLLOUT (no solve), excluded here and
//! deferred to the field model. Reach-weighted blend by banked reach×pot.
//!
//! CAVEAT (named): research-scale ratios (1.5–3:1) are far gentler than
//! production (147:1). The sweep gives the per-family RELATIONSHIP (does the
//! gap grow faster in some families?), which transfers as a shape; the absolute
//! production fidelity is the production build's to apply, harness/live-confirmed.

use play_harness::experiment::{reach_weighted_overall, FamilyEdge};
use play_harness::v1_seam::{SeamBlueprint, SeamGame};

fn flop() -> [u8; 3] { [47, 21, 0] }

#[test]
#[ignore = "fidelity sweep (research-scale, minutes); --ignored --nocapture --release"]
fn multiway_fidelity_sweep() {
    // (live, nh) — nh chosen so exact exploitability O(nh^np) stays feasible
    // and nb at each ratio is ≥ a few buckets.
    // Multiway families only (live-2 = HU exact/search path; live-6 = rollout).
    let fams: &[(u8, usize, f64)] = &[
        (3, 21, 0.323), (4, 15, 0.328), (5, 10, 0.163),
    ];
    let ratios = [1.5f64, 3.0];
    const ITERS: u32 = 400;
    eprintln!("\n═══ MULTIWAY FIDELITY: distance-from-exact (% pot) vs compression ratio ═══");
    eprintln!("family (nh) | exact | ratio 1.5 | ratio 3.0   (lower = closer to exact)");
    // dist[fam][ratio] for the candidate blends.
    let mut dist: Vec<Vec<f32>> = Vec::new();
    let mut exacts: Vec<f32> = Vec::new();
    for &(live, nh, _w) in fams {
        let g = SeamGame::new(live, 2, 12, flop());
        let exact = SeamBlueprint::solve_research(&g, nh, ITERS + 200);
        let e0 = exact.exploitability(&g, nh, 12);
        let mut row = Vec::new();
        for &r in &ratios {
            let nb = ((nh as f64 / r).round() as usize).max(2);
            let bp = SeamBlueprint::solve_research_bucketed(&g, nh, nb, ITERS);
            row.push(bp.exploitability(&g, nh, 12));
        }
        eprintln!("  live-{live} ({nh:>2}) | {e0:5.2} | {:8.2} | {:8.2}", row[0], row[1]);
        dist.push(row);
        exacts.push(e0);
    }

    // Per-family GAP above exact, at each ratio → does a family degrade faster?
    eprintln!("\nGAP above exact (distance, % pot):");
    eprintln!("family | ratio 1.5 | ratio 3.0");
    for (i, &(live, _, _)) in fams.iter().enumerate() {
        eprintln!("  live-{live} | {:8.2} | {:8.2}", dist[i][0] - exacts[i], dist[i][1] - exacts[i]);
    }

    // CANDIDATE BLENDS by reach weight (the overall fidelity verdict). live-6
    // (rollout) excluded; weights renormalize over live-2..5.
    let mk = |pick: &dyn Fn(usize) -> f32| -> (f64, f64) {
        let fe: Vec<FamilyEdge> = fams.iter().enumerate().map(|(i, &(live, _, w))| FamilyEdge {
            live, edge: pick(i) as f64, stderr: 0.0, reach_weight: w,
        }).collect();
        reach_weighted_overall(&fe)
    };
    let coarse = mk(&|i| dist[i][1]);                       // ratio-3 everywhere ("B8-like" floor)
    let rw = mk(&|i| if fams[i].0 <= 4 { dist[i][0] } else { dist[i][1] }); // common finer (1.5), rare coarse (3)
    let plus = mk(&|i| dist[i][0]);                          // ratio-1.5 everywhere (PLUS)
    let exact_blend = mk(&|i| exacts[i]);
    eprintln!("\nREACH-WEIGHTED OVERALL distance-from-exact (% pot):");
    eprintln!("  exact ceiling        {:6.2}", exact_blend.0);
    eprintln!("  coarse (ratio-3 all) {:6.2}", coarse.0);
    eprintln!("  reach-weighted       {:6.2}  (common@1.5, rare@3)", rw.0);
    eprintln!("  reach-weighted-PLUS  {:6.2}  (all@1.5)", plus.0);
    eprintln!("\n→ FORK: reach-weighted close to exact ⇒ bucket count suffices (build it). All far");
    eprintln!("  from exact ⇒ abstraction inadequate (better-than-quantile coordinate). PLUS−rw gap");
    eprintln!("  = whether finer-on-common is worth its cost. (Research ratios; transfer caveat.)");
}
