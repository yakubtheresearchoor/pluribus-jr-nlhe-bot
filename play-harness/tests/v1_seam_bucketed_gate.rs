//! Bucketed-blueprint faithfulness gate (step 3, the load-bearing one). The
//! bucketing LIFT (many hands → one bucket → one strategy → lifted back to
//! per-hand) is the one new error surface; a bug there would corrupt the
//! fidelity number the whole experiment exists to produce. So gate it
//! INDEPENDENTLY of the exact wiring: (1) lift uniformity — hands in the same
//! flop bucket play the SAME strategy; (2) decision-faithfulness — the play
//! path reproduces the looked-up distribution; (3) conservation; (4) the
//! distance-from-exact ANCHOR — exact ≈ 0 exploitability, bucketed measures the
//! abstraction gap, monotone in compression.

use play_harness::v1_seam::{SeamBlueprint, SeamGame};

fn flop() -> [u8; 3] { [47, 21, 0] }

#[test]
fn bucketed_lift_faithful_and_anchored() {
    let g = SeamGame::new(3, 2, 12, flop());
    let nh = 12;
    let nb = 4; // 3:1 compression
    let bp = SeamBlueprint::solve_research_bucketed(&g, nh, nb, 300);
    let buckets = SeamBlueprint::flop_buckets(&g, nh, nb);
    let dummy = [flop()[0], flop()[1], flop()[2], bp.runouts[0].0, bp.runouts[0].1];

    // (1) LIFT UNIFORMITY: every pair of hands in the same flop bucket plays an
    // identical root strategy (the bucketed strategy assigned uniformly).
    let mut checked = 0;
    for h1 in 0..nh {
        for h2 in (h1 + 1)..nh {
            if buckets[h1] != buckets[h2] || buckets[h1] == u16::MAX { continue; }
            let d1 = bp.action_dist(g.tree_ref(), 0, [bp.hands[h1].0, bp.hands[h1].1], &dummy);
            let d2 = bp.action_dist(g.tree_ref(), 0, [bp.hands[h2].0, bp.hands[h2].1], &dummy);
            for a in 0..d1.len() {
                assert!((d1[a] - d2[a]).abs() < 1e-4,
                    "lift NOT uniform: hands {h1},{h2} share bucket {} but differ at action {a}: {d1:?} vs {d2:?}",
                    buckets[h1]);
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no same-bucket hand pairs found to check lift uniformity");
    eprintln!("lift uniformity: {checked} same-bucket pairs identical ✓");

    // (2) DECISION-FAITHFULNESS through the play path (bucketed source).
    let th = [bp.hands[nh / 2].0, bp.hands[nh / 2].1];
    let want = bp.action_dist(g.tree_ref(), 0, th, &dummy);
    let mut freq = vec![0u64; want.len()];
    let mut rng = 0xB17u64;
    for _ in 0..40_000 {
        freq[g.sample_blueprint_action(&bp, 0, th, &dummy, &mut rng)] += 1;
    }
    for a in 0..want.len() {
        assert!((freq[a] as f64 / 40_000.0 - want[a] as f64).abs() < 0.02, "bucketed play != dist at {a}");
    }

    // (3) conservation with bucketed seats over audit hands.
    let bps: Vec<&SeamBlueprint> = (0..3).map(|_| &bp).collect();
    let mut rng = 0xC27u64;
    let mut played = 0;
    for _ in 0..1500 {
        let Some((holes, board)) = g.deal_audit(&bp, &mut rng) else { continue };
        let (net, _l) = g.play_blueprints(&bps, &holes, &board, &mut rng);
        let rake = g.dead as i64 - net.iter().sum::<i64>();
        assert!(rake >= 0 && rake <= g.rake_cap as i64);
        played += 1;
    }
    assert!(played > 300);

    // (4) DISTANCE-FROM-EXACT ANCHOR: exact ≈ 0; bucketed grows with compression.
    let exact = SeamBlueprint::solve_research(&g, nh, 400);
    let e_exact = exact.exploitability(&g, nh, 12);
    let e_b4 = bp.exploitability(&g, nh, 12);
    let bp2 = SeamBlueprint::solve_research_bucketed(&g, nh, 6, 300); // 2:1 (finer)
    let e_b6 = bp2.exploitability(&g, nh, 12);
    eprintln!("DISTANCE-FROM-EXACT (% pot): exact {e_exact:.3} | B6(2:1) {e_b6:.3} | B4(3:1) {e_b4:.3}");
    assert!(e_exact < 1.0, "exact blueprint not near-equilibrium ({e_exact:.3}%) — anchor invalid");
    assert!(e_b4 > e_exact, "bucketed B4 not above exact — lift/anchor suspect");
    assert!(e_b6 <= e_b4 + 1.5, "finer (B6) not ≤ coarser (B4) — bucketing non-monotone (suspect)");
    eprintln!("BUCKETED GATE PASS: lift uniform ✓, faithful ✓, conserves ✓, anchored to exact ✓.");
}
