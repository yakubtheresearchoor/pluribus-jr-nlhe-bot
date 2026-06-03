// Independent CPU brute-force enumeration of fold-terminal CFV — no
// inclusion-exclusion shortcuts. Direct O(nh) enumeration with explicit
// card-conflict checks. This is the truly-independent ground truth for
// the fold-terminal CFV computation, the reference against which both
// CPU side_pot_showdown_cfv and GPU multiway_brute_force_showdown should
// be judged.
//
// Why this matters: #37 reversed three times. The current findings are:
//   - CPU side_pot_showdown_cfv fold-fast-path uses `sum - minus[c1] - minus[c2]`
//   - This is missing the inclusion-exclusion term `+ uses_both[c1, c2]`
//   - GPU brute-force enumerates with explicit card-conflict checks (no shortcut)
//
// This test settles which is correct by enumerating from FIRST PRINCIPLES.

#[derive(Debug)]
struct TestCase {
    nh: usize,
    np: usize,
    traverser: usize,
    starting_pot: i32,
    contributions: Vec<i32>,
    fold_mask: u16,
    opp_reach: Vec<f32>, // [nh]
    hand_cards: Vec<u8>, // [nh * 2], encoded as separate card indices
    num_combinations: f32,
}

/// Direct from-first-principles enumeration of fold-terminal CFV.
/// Does NOT use any inclusion-exclusion shortcut. Iterates over all
/// possible opponent hands g, checks card-conflict against my hand h
/// (and against my-hand's individual cards), accumulates reach.
///
/// For HU (np=2) with traverser folded: CFV(h) = payoff * sum_{g not conflicting with h} reach[g]
fn independent_fold_cfv(tc: &TestCase) -> Vec<f32> {
    assert_eq!(tc.np, 2, "this enumerator handles HU only");
    let np = tc.np;
    let nh = tc.nh;
    let c_t = tc.contributions[tc.traverser];
    let traverser_folded = tc.fold_mask & (1u16 << tc.traverser) != 0;
    assert!(traverser_folded, "this test is for fold-terminal case");

    // payoff = -traverser_investment when folded
    let traverser_investment = tc.starting_pot as f32 / np as f32 + c_t as f32;
    let payoff = -traverser_investment;

    let mut cfv = vec![0.0f32; nh];
    for h in 0..nh {
        let hc1 = tc.hand_cards[h * 2];
        let hc2 = tc.hand_cards[h * 2 + 1];
        let mut cfreach = 0.0f32;
        for g in 0..nh {
            let gc1 = tc.hand_cards[g * 2];
            let gc2 = tc.hand_cards[g * 2 + 1];
            // Explicit card-conflict check: skip if any of g's cards
            // match any of h's cards (in either direction).
            if gc1 == hc1 || gc1 == hc2 || gc2 == hc1 || gc2 == hc2 { continue; }
            cfreach += tc.opp_reach[g];
        }
        cfv[h] = payoff * cfreach / tc.num_combinations;
    }
    cfv
}

#[test]
fn settle_fold_terminal_node15_by_direct_enumeration() {
    // Reproduce the exact node[15] case from handtrace_fold_terminal_node15.rs:
    // HU, traverser=0 folded, contributions=[5, 15], opp_reach all 0.125,
    // 4 hands: AhKh, QhJh, Th9h, 8h6h. starting_pot=10, num_combinations=16.
    //
    // Card indices (from index_to_card_pair on the test fixture):
    //   AhKh hand_cards = [Ah_idx, Kh_idx]
    //   QhJh hand_cards = [Qh_idx, Jh_idx]
    //   Th9h hand_cards = [Th_idx, 9h_idx]
    //   8h6h hand_cards = [8h_idx, 6h_idx]
    // From handtrace test we saw hand_cards = [46, 50, 38, 42, 30, 34, 18, 26]
    // (probably: 46=Ah, 50=Kh, 38=Qh, 42=Jh, 30=Th, 34=9h, 18=8h, 26=6h
    //  — none repeat, so no two hands share cards).

    let tc = TestCase {
        nh: 4, np: 2, traverser: 0, starting_pot: 10,
        contributions: vec![5, 15], // p0 contributed 5 then folded; p1 raised to 15
        fold_mask: 0b01,  // p0 folded
        opp_reach: vec![0.125, 0.125, 0.125, 0.125],
        hand_cards: vec![46, 50, 38, 42, 30, 34, 18, 26],
        num_combinations: 16.0,
    };

    let cfv = independent_fold_cfv(&tc);
    eprintln!("\n=== INDEPENDENT brute-force enumeration of node[15] fold CFV ===");
    eprintln!("Inputs:");
    eprintln!("  contributions = {:?}", tc.contributions);
    eprintln!("  fold_mask     = {:b} (p0 folded)", tc.fold_mask);
    eprintln!("  opp_reach     = {:?}", tc.opp_reach);
    eprintln!("  hand_cards    = {:?}", tc.hand_cards);
    eprintln!("    [AhKh, QhJh, Th9h, 8h6h with distinct cards each]");
    eprintln!("  payoff = -traverser_investment = -(10/2 + 5) = -10");
    eprintln!();
    eprintln!("Per-hand enumeration:");
    for h in 0..tc.nh {
        let hc1 = tc.hand_cards[h * 2];
        let hc2 = tc.hand_cards[h * 2 + 1];
        let mut included: Vec<usize> = Vec::new();
        let mut excluded: Vec<usize> = Vec::new();
        for g in 0..tc.nh {
            let gc1 = tc.hand_cards[g * 2];
            let gc2 = tc.hand_cards[g * 2 + 1];
            if gc1 == hc1 || gc1 == hc2 || gc2 == hc1 || gc2 == hc2 {
                excluded.push(g);
            } else {
                included.push(g);
            }
        }
        eprintln!(
            "  h={}: hc=[{},{}] included opp hands: {:?} excluded: {:?} cfreach = {} * 0.125 = {} → CFV/{} = {}",
            h, hc1, hc2, included, excluded,
            included.len(),
            included.len() as f32 * 0.125,
            tc.num_combinations,
            -10.0 * included.len() as f32 * 0.125 / tc.num_combinations
        );
    }
    eprintln!("\nResult: {:?}", cfv);
    eprintln!();
    eprintln!("Comparison with known answers from handtrace_cpu_node13.rs run:");
    eprintln!("  CPU side_pot_showdown_cfv produced: [-0.156, -0.156, -0.156, -0.156]");
    eprintln!("  GPU multiway_brute_force_showdown:  [-0.234, -0.234, -0.234, -0.234]");
    eprintln!("  This independent enumerator:         {:?}", cfv);
    eprintln!();

    let expected_indep = -0.234375f32;
    let cpu_value = -0.15625f32;

    eprintln!("  Independent enumeration ground truth: {}", expected_indep);
    eprintln!("  GPU was always correct (matches truth).");
    eprintln!("  CPU was originally buggy (returned -0.156); fixed in #37 audit.");
    eprintln!("  Post-fix CPU should also match truth — see handtrace_fold_terminal_node15.rs.");

    // Assert the truth value is what direct enumeration gives.
    // This is a standing regression check for the independent enumerator itself.
    assert!(
        (cfv[0] - expected_indep).abs() < 1e-6,
        "Independent enumeration result {} doesn't match expected truth {}. \
         The enumerator itself may have changed; if so, re-verify by hand calculation.",
        cfv[0], expected_indep
    );
    let _ = cpu_value; // suppress unused warning
}
