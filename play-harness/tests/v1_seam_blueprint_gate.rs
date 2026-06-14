//! Blueprint-into-SeamGame wiring gate (docket step 2b). Gated against a CHEAP
//! research-scale in-memory solve (no production artifact). Proves the harness
//! reads a blueprint correctly: (1) the card→hand map is a correct bijection,
//! (2) looked-up distributions are valid, (3) SeamGame plays the SAME decisions
//! the blueprint would standalone (empirical action frequencies match the
//! looked-up distribution — no lookup/indexing bug), (4) conservation holds with
//! real (blueprint) policies in the seats.

use play_harness::v1_seam::{SeamBlueprint, SeamGame};

fn flop() -> [u8; 3] { [47, 21, 0] } // Ks 7d 2c

#[test]
fn blueprint_wiring_correct_and_conserves() {
    let g = SeamGame::new(3, 2, 12, flop());
    let bp = SeamBlueprint::solve_research(&g, 12, 300);

    // (1) card→hand map is a correct bijection: hands[h] is the card pair of
    // index h, so looking it up must recover h.
    for h in 0..bp.nh {
        assert_eq!(bp.hand_index(bp.hands[h]), h, "card→hand map not a bijection at hand {h}");
    }

    // (2) looked-up distributions are valid (sum to 1, in [0,1]) at the root.
    let dummy = [flop()[0], flop()[1], flop()[2], bp.runouts[0].0, bp.runouts[0].1];
    for h in 0..bp.nh {
        let d = bp.action_dist(g.tree_ref(), 0, [bp.hands[h].0, bp.hands[h].1], &dummy);
        let s: f32 = d.iter().sum();
        assert!((s - 1.0).abs() < 1e-3 && d.iter().all(|&x| (-1e-6..=1.0 + 1e-6).contains(&x)),
            "invalid action distribution for hand {h}: {d:?} (sum {s})");
    }

    // (3) SAME DECISIONS: at the root, the empirical action frequencies when
    // SeamGame samples seat-0's blueprint must match the looked-up distribution
    // (proves the sampling reads the right hand's strategy — no indexing bug).
    // Pick a hand whose root strategy is non-degenerate if possible.
    let test_hand = [bp.hands[bp.nh / 2].0, bp.hands[bp.nh / 2].1];
    let want = bp.action_dist(g.tree_ref(), 0, test_hand, &dummy);
    let na = want.len();
    let mut freq = vec![0u64; na];
    let mut rng = 0xB1u64;
    let trials = 40_000u64;
    for _ in 0..trials {
        let a = g.sample_blueprint_action(&bp, 0, test_hand, &dummy, &mut rng);
        freq[a] += 1;
    }
    for a in 0..na {
        let emp = freq[a] as f64 / trials as f64;
        assert!((emp - want[a] as f64).abs() < 0.02,
            "action {a}: empirical {emp:.3} != looked-up {:.3} (sampling/indexing bug)", want[a]);
    }
    eprintln!("root action dist hand[{}]: looked-up {want:?} ≈ empirical play", bp.nh / 2);

    // (4) conservation with blueprint seats over audit-mode hands.
    let bps: Vec<&SeamBlueprint> = (0..3).map(|_| &bp).collect();
    let mut rng = 0xC2u64;
    let mut played = 0;
    for _ in 0..2000 {
        let Some((holes, board)) = g.deal_audit(&bp, &mut rng) else { continue };
        let (net, _l) = g.play_blueprints(&bps, &holes, &board, &mut rng);
        let rake = g.dead as i64 - net.iter().sum::<i64>();
        assert!(rake >= 0 && rake <= g.rake_cap as i64, "blueprint play rake {rake} out of range");
        played += 1;
    }
    assert!(played > 500, "too few audit hands played ({played}) — dealer/universe issue");
    eprintln!("BLUEPRINT WIRING GATE PASS: bijection ✓, valid dists ✓, same decisions ✓, conserves ✓ ({played} hands).");
}
