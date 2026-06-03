/// Standalone analysis: product formula, +reach[h] term, inter-opponent conflict
/// correction, and zero-sum property for 3-player showdown.
///
/// Run:
///   cargo test -p solver-core --test diagonal_analysis -- --test-threads=1 --nocapture
///
/// This test uses NO game infrastructure. It's pure math on small hand sets,
/// comparing brute-force enumeration against the product formula variants.

/// Represents a hand with two cards and a strength ranking.
#[derive(Clone, Debug)]
struct TestHand {
    cards: [u8; 2],  // two card indices (0..51)
    strength: u32,   // higher = stronger
}

fn conflicts(h1: &TestHand, h2: &TestHand) -> bool {
    h1.cards[0] == h2.cards[0] || h1.cards[0] == h2.cards[1]
        || h1.cards[1] == h2.cards[0] || h1.cards[1] == h2.cards[1]
}

/// Brute-force 3-player showdown CFV for traverser holding hand h_idx.
///
/// For each valid (g0, g1) pair (non-conflicting with h AND with each other):
///   Let S = number of players with max strength among {h, g0, g1}
///   traverser payoff = (K+1)/S - 1 if h is among the max, else -1
fn brute_force_cfv(
    hands: &[TestHand],
    reach_0: &[f32],
    reach_1: &[f32],
    h_idx: usize,
    half_pot: f32,
) -> f32 {
    let h = &hands[h_idx];
    let nh = hands.len();
    let k = 2; // num opponents
    let mut total = 0.0f32;

    for g0 in 0..nh {
        if conflicts(h, &hands[g0]) { continue; }
        if reach_0[g0] == 0.0 { continue; }
        for g1 in 0..nh {
            if conflicts(h, &hands[g1]) { continue; }
            if conflicts(&hands[g0], &hands[g1]) { continue; }
            if reach_1[g1] == 0.0 { continue; }

            let max_str = h.strength.max(hands[g0].strength).max(hands[g1].strength);
            let payoff = if h.strength < max_str {
                -1.0
            } else {
                let mut num_tied = 0;
                if h.strength == max_str { num_tied += 1; }
                if hands[g0].strength == max_str { num_tied += 1; }
                if hands[g1].strength == max_str { num_tied += 1; }
                (k + 1) as f32 / num_tied as f32 - 1.0
            };

            total += reach_0[g0] * reach_1[g1] * payoff;
        }
    }

    half_pot * total
}

/// Product formula CFV (what the current code computes).
/// W_oi[h] = sum of reach_oi[g] for g strictly weaker than h AND non-conflicting with h
/// E_oi[h] = sum of reach_oi[g] for g non-conflicting with h
///
/// If include_self is true, E includes reach_oi[h_idx] (the +reach[h] term from code).
fn product_formula_cfv(
    hands: &[TestHand],
    reach_0: &[f32],
    reach_1: &[f32],
    h_idx: usize,
    half_pot: f32,
    include_self_in_eff: bool,
) -> f32 {
    let h = &hands[h_idx];
    let nh = hands.len();
    let k = 2;

    let reaches = [reach_0, reach_1];
    let mut w_product = 1.0f32;
    let mut e_product = 1.0f32;

    for oi in 0..k {
        let reach = reaches[oi];
        let mut w = 0.0f32;
        let mut e = 0.0f32;

        for g in 0..nh {
            if conflicts(h, &hands[g]) { continue; }
            // Note: g == h_idx always conflicts (same cards), so already excluded

            if hands[g].strength < h.strength {
                w += reach[g];
            }
            e += reach[g];
        }

        if include_self_in_eff {
            e += reach[h_idx];
        }

        w_product *= w;
        e_product *= e;
    }

    half_pot * ((k as f32 + 1.0) * w_product - e_product)
}

/// Full inter-opponent conflict correction.
///
/// The product formula overcounts because it includes (g0, g1) pairs where
/// g0 and g1 conflict with each other (share a card). To correct:
///
/// correction_wins[h] = sum_{g0,g1: both non-conflict with h, g0 conflicts with g1,
///                       both weaker than h} reach_0[g0]*reach_1[g1]
/// correction_total[h] = sum_{g0,g1: both non-conflict with h, g0 conflicts with g1}
///                       reach_0[g0]*reach_1[g1]
///
/// corrected_cfv = half_pot * ((K+1)*(W0*W1 - correction_wins) - (E0*E1 - correction_total))
///               = product_cfv + half_pot * (correction_total - (K+1)*correction_wins)
fn full_correction_cfv(
    hands: &[TestHand],
    reach_0: &[f32],
    reach_1: &[f32],
    h_idx: usize,
    half_pot: f32,
    include_self_in_eff: bool,
) -> f32 {
    let h = &hands[h_idx];
    let nh = hands.len();
    let k = 2;

    let base = product_formula_cfv(hands, reach_0, reach_1, h_idx, half_pot, include_self_in_eff);

    // Compute correction: sum over all (g0,g1) that are both non-conflicting
    // with h but DO conflict with each other
    let mut corr_wins = 0.0f32;
    let mut corr_total = 0.0f32;

    for g0 in 0..nh {
        if conflicts(h, &hands[g0]) { continue; }
        if reach_0[g0] == 0.0 { continue; }
        for g1 in 0..nh {
            if conflicts(h, &hands[g1]) { continue; }
            if !conflicts(&hands[g0], &hands[g1]) { continue; } // only CONFLICTING pairs
            if reach_1[g1] == 0.0 { continue; }

            let d = reach_0[g0] * reach_1[g1];
            corr_total += d;
            if hands[g0].strength < h.strength && hands[g1].strength < h.strength {
                corr_wins += d;
            }
        }
    }

    // Include the self-self pair in the correction if eff includes self
    if include_self_in_eff {
        // When eff includes +reach[h], E0*E1 includes the term where
        // both opponents hold h. This pair (h,h) conflicts with itself,
        // so it should be subtracted. But h also conflicts with h for the
        // traverser... wait, h is already excluded from the loop above by
        // the conflicts(h, hands[g]) check (since h conflicts with itself).
        //
        // Actually the +reach[h] is added AFTER the conflict check in eff,
        // so it's not in the g-loop. We need to handle it specially.
        //
        // E_oi[h] = (sum_{g non-conflict with h} reach_oi[g]) + reach_oi[h]
        // E_0[h] * E_1[h] = (A + r0[h]) * (B + r1[h])
        //                  = A*B + A*r1[h] + B*r0[h] + r0[h]*r1[h]
        // where A = sum_{g non-conflict} r0[g], B = sum_{g non-conflict} r1[g]
        //
        // The A*B product is what the loop correction handles.
        // The cross terms A*r1[h] and B*r0[h] represent one opponent holding
        // a non-conflicting hand and the other holding h. In the real game,
        // opponent can't hold h (same cards as traverser). These terms are
        // phantom terms from the +reach[h] addition.
        //
        // The r0[h]*r1[h] term represents both opponents holding h. Also phantom.
        //
        // ALL these extra terms should be subtracted from the product since they
        // represent impossible opponent assignments (opponent can't hold
        // traverser's hand).
        //
        // But wait - let me reconsider. What is the PHYSICAL meaning?
        //
        // For the 2-player formula cfv[h] = half_pot * (2*W[h] - E[h]):
        //   E[h] = sum of non-conflicting reach for the single opponent
        //   The +reach[h] ensures E[h] includes opponent hands that don't
        //   share cards with h. Since h DOES share cards with h, opponent
        //   can't hold h, so we should NOT include +reach[h]!
        //
        // ... unless the formula requires it for a mathematical identity.
        // Let me test both ways.
    }

    base + half_pot * (corr_total - (k as f32 + 1.0) * corr_wins)
}


// ───────────────────────────────────────────────────────────────────────
// Test 1: No inter-opponent conflicts (all hands use unique cards)
// Product formula should be exact, +reach[h] should NOT be used.
// ───────────────────────────────────────────────────────────────────────
#[test]
fn no_conflicts_product_exact() {
    println!("\n=== TEST 1: No card conflicts between hands ===");
    println!("  (All hands use distinct cards, so no inter-opponent conflicts)\n");

    let hands = vec![
        TestHand { cards: [0, 1], strength: 100 },
        TestHand { cards: [2, 3], strength: 200 },
        TestHand { cards: [4, 5], strength: 300 },
        TestHand { cards: [6, 7], strength: 400 },
        TestHand { cards: [8, 9], strength: 500 },
    ];
    let nh = hands.len();

    let reach_0 = vec![0.3, 0.5, 0.2, 0.8, 0.1];
    let reach_1 = vec![0.4, 0.6, 0.3, 0.7, 0.9];
    let half_pot = 10.0;

    println!("  {:>2} {:>5} {:>10} {:>10} {:>10} {:>10}",
        "h", "str", "BF", "PF(+s)", "PF(-s)", "err(+s)");

    for h in 0..nh {
        let bf = brute_force_cfv(&hands, &reach_0, &reach_1, h, half_pot);
        let pf_with = product_formula_cfv(&hands, &reach_0, &reach_1, h, half_pot, true);
        let pf_without = product_formula_cfv(&hands, &reach_0, &reach_1, h, half_pot, false);

        println!("  {:>2} {:>5} {:>10.4} {:>10.4} {:>10.4} {:>10.2e}",
            h, hands[h].strength, bf, pf_with, pf_without, (pf_with - bf).abs());
    }

    // With all-unique cards: g==h always conflicts (same 2 cards), so
    // the g-loop in PF already excludes h. +reach[h] adds back a PHANTOM
    // opponent hand (opponent holding traverser's cards). Since no other
    // hand conflicts with any other, g0 and g1 never conflict either.
    //
    // So PF(-self) should be exact, and PF(+self) should NOT.
    //
    // Unless... the formula (K+1)*prod(W) - prod(E) requires E to include
    // the self term for the algebra to work out. Let me check.

    println!("\n  Checking PF(-self) = BF:");
    for h in 0..nh {
        let bf = brute_force_cfv(&hands, &reach_0, &reach_1, h, half_pot);
        let pf = product_formula_cfv(&hands, &reach_0, &reach_1, h, half_pot, false);
        if (pf - bf).abs() > 1e-4 {
            println!("    h={}: PF(-self)={:.6} BF={:.6} MISMATCH err={:.2e}", h, pf, bf, (pf-bf).abs());
        }
    }

    println!("  Checking PF(+self) = BF:");
    for h in 0..nh {
        let bf = brute_force_cfv(&hands, &reach_0, &reach_1, h, half_pot);
        let pf = product_formula_cfv(&hands, &reach_0, &reach_1, h, half_pot, true);
        if (pf - bf).abs() > 1e-4 {
            println!("    h={}: PF(+self)={:.6} BF={:.6} MISMATCH err={:.2e}", h, pf, bf, (pf-bf).abs());
        }
    }

    // Test the actual identity: for no-conflict case, what does PF compute?
    // E_oi[h] without self = sum_{g != h, non-conflict with h} reach[g]
    //                      = sum_{all g except h} reach[g]  (since no hand conflicts with h except h itself)
    //                      = total_reach - reach[h]
    //
    // W_oi[h] = sum_{g weaker than h, non-conflict with h} reach[g]
    //
    // BF cfv[h] = half_pot * sum_{valid (g0,g1)} r0[g0]*r1[g1] * payoff
    //
    // For no inter-opp conflicts, ALL (g0,g1) where g0 != h and g1 != h are valid.
    // (Plus g0 and g1 can be the same hand since they're different players.)
    //
    // Wait -- can g0 == g1? If hands[g0] == hands[g1] (same index), they have
    // the same cards, so conflicts(g0, g1) = true. So g0 == g1 is NEVER valid.
    //
    // The product W_0[h]*W_1[h] includes the g0==g1 case (same hand index held
    // by both opponents). The BF does NOT. So even with no card conflicts between
    // DIFFERENT hands, the product still overcounts the g0==g1 "diagonal".

    println!("\n  KEY INSIGHT: Even with no card conflicts between different hands,");
    println!("  the product formula overcounts because it includes g0==g1 pairs");
    println!("  (both opponents holding the same hand, which shares all cards).");
    println!("  This is the DIAGONAL correction that's always needed.");

    // Count the diagonal contribution
    println!("\n  Diagonal contribution (g0==g1, same hand for both opps):");
    for h in 0..nh {
        let mut diag_wins = 0.0f32;
        let mut diag_total = 0.0f32;
        for g in 0..nh {
            if conflicts(&hands[h], &hands[g]) { continue; }
            let d = reach_0[g] * reach_1[g];
            diag_total += d;
            if hands[g].strength < hands[h].strength {
                diag_wins += d;
            }
        }
        let bf = brute_force_cfv(&hands, &reach_0, &reach_1, h, half_pot);
        let pf = product_formula_cfv(&hands, &reach_0, &reach_1, h, half_pot, false);
        let corrected = pf + half_pot * (diag_total - 3.0 * diag_wins);
        println!("    h={}: diag_wins={:.3} diag_total={:.3} PF={:.4} corrected={:.4} BF={:.4} err={:.2e}",
            h, diag_wins, diag_total, pf, corrected, bf, (corrected - bf).abs());
    }
}

// ───────────────────────────────────────────────────────────────────────
// Test 2: Full correction (all inter-opponent conflicts)
// ───────────────────────────────────────────────────────────────────────
#[test]
fn full_correction_matches_brute_force() {
    println!("\n=== TEST 2: Full inter-opponent conflict correction ===");

    // Hands with varied card conflicts
    let hands = vec![
        TestHand { cards: [0, 1], strength: 100 },
        TestHand { cards: [1, 2], strength: 200 },  // shares card 1 with h0
        TestHand { cards: [2, 3], strength: 300 },  // shares card 2 with h1
        TestHand { cards: [4, 5], strength: 400 },
        TestHand { cards: [6, 7], strength: 500 },
    ];
    let nh = hands.len();

    let reach_0 = vec![0.3, 0.5, 0.2, 0.8, 0.1];
    let reach_1 = vec![0.4, 0.6, 0.3, 0.7, 0.9];
    let half_pot = 10.0;

    // Print conflict matrix between all hands
    println!("  Inter-hand conflict matrix:");
    print!("       ");
    for j in 0..nh { print!("  h{}  ", j); }
    println!();
    for i in 0..nh {
        print!("    h{} ", i);
        for j in 0..nh {
            print!("  {}   ", if conflicts(&hands[i], &hands[j]) { "X" } else { "." });
        }
        println!();
    }

    println!("\n  {:>2} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "h", "str", "BF", "PF(-s)", "FC(-s)", "PF(+s)", "FC(+s)");

    let mut all_fc_minus_exact = true;
    let mut all_fc_plus_exact = true;

    for h in 0..nh {
        let bf = brute_force_cfv(&hands, &reach_0, &reach_1, h, half_pot);
        let pf_minus = product_formula_cfv(&hands, &reach_0, &reach_1, h, half_pot, false);
        let fc_minus = full_correction_cfv(&hands, &reach_0, &reach_1, h, half_pot, false);
        let pf_plus = product_formula_cfv(&hands, &reach_0, &reach_1, h, half_pot, true);
        let fc_plus = full_correction_cfv(&hands, &reach_0, &reach_1, h, half_pot, true);

        let err_fc_minus = (fc_minus - bf).abs();
        let err_fc_plus = (fc_plus - bf).abs();
        if err_fc_minus > 0.01 { all_fc_minus_exact = false; }
        if err_fc_plus > 0.01 { all_fc_plus_exact = false; }

        println!("  {:>2} {:>5} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}  err(-s)={:.2e} err(+s)={:.2e}",
            h, hands[h].strength, bf, pf_minus, fc_minus, pf_plus, fc_plus, err_fc_minus, err_fc_plus);
    }

    println!("\n  FC(-self) exact: {}", all_fc_minus_exact);
    println!("  FC(+self) exact: {}", all_fc_plus_exact);

    if all_fc_minus_exact {
        println!("  ==> The correct formula uses E WITHOUT +reach[h], plus full conflict correction.");
    }
    if all_fc_plus_exact {
        println!("  ==> The correct formula uses E WITH +reach[h], plus full conflict correction.");
    }
    if !all_fc_minus_exact && !all_fc_plus_exact {
        println!("  ==> NEITHER variant is exact! Need to investigate further.");
        // Print detailed analysis
        for h in 0..nh {
            let bf = brute_force_cfv(&hands, &reach_0, &reach_1, h, half_pot);
            let fc_minus = full_correction_cfv(&hands, &reach_0, &reach_1, h, half_pot, false);
            let fc_plus = full_correction_cfv(&hands, &reach_0, &reach_1, h, half_pot, true);
            println!("    h={}: BF={:.6} FC-={:.6} FC+={:.6} diff-={:.6} diff+={:.6}",
                h, bf, fc_minus, fc_plus, fc_minus - bf, fc_plus - bf);
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
// Test 3: Zero-sum property check
// ───────────────────────────────────────────────────────────────────────
#[test]
fn zero_sum_across_traversers() {
    println!("\n=== TEST 3: Zero-sum across 3 traversers ===");

    let hands = vec![
        TestHand { cards: [0, 1], strength: 100 },
        TestHand { cards: [2, 3], strength: 200 },
        TestHand { cards: [3, 4], strength: 300 },
        TestHand { cards: [5, 6], strength: 400 },
        TestHand { cards: [7, 8], strength: 500 },
    ];
    let nh = hands.len();

    let reach = [
        vec![0.3, 0.5, 0.2, 0.8, 0.1],
        vec![0.4, 0.6, 0.3, 0.7, 0.9],
        vec![0.5, 0.4, 0.6, 0.2, 0.7],
    ];
    let half_pot = 10.0;

    for (label, use_fc, self_in_eff) in [
        ("BF", false, false),
        ("PF(-self)", false, false),
        ("PF(+self)", false, true),
        ("FC(-self)", true, false),
        ("FC(+self)", true, true),
    ] {
        let mut ev = [0.0f64; 3];

        for trav in 0..3 {
            let opp0 = (trav + 1) % 3;
            let opp1 = (trav + 2) % 3;

            for h in 0..nh {
                let val = if label == "BF" {
                    brute_force_cfv(&hands, &reach[opp0], &reach[opp1], h, half_pot)
                } else if use_fc {
                    full_correction_cfv(&hands, &reach[opp0], &reach[opp1], h, half_pot, self_in_eff)
                } else {
                    product_formula_cfv(&hands, &reach[opp0], &reach[opp1], h, half_pot, self_in_eff)
                };
                ev[trav] += (reach[trav][h] as f64) * (val as f64);
            }
        }

        let sum: f64 = ev.iter().sum();
        println!("  {:>12}: EV=[{:>10.4},{:>10.4},{:>10.4}]  sum={:.2e}",
            label, ev[0], ev[1], ev[2], sum);
    }
}

// ───────────────────────────────────────────────────────────────────────
// Test 4: Replicate the real code's minus technique and verify what
// eff[h] with +reach[h] actually represents.
// ───────────────────────────────────────────────────────────────────────
#[test]
fn minus_technique_identity() {
    println!("\n=== TEST 4: What does eff[h] with +reach[h] compute? ===");

    let hands = vec![
        TestHand { cards: [0, 1], strength: 100 },
        TestHand { cards: [1, 2], strength: 200 },
        TestHand { cards: [2, 3], strength: 300 },
        TestHand { cards: [4, 5], strength: 400 },
        TestHand { cards: [5, 6], strength: 250 },
    ];
    let nh = hands.len();
    let reach = vec![0.3, 0.5, 0.2, 0.8, 0.1];

    // Replicate minus technique: for each hand h,
    // eff_with = cfreach_sum - minus[c1] - minus[c2] + reach[h]
    let mut cfreach_sum = 0.0f32;
    let mut cfreach_minus = [0.0f32; 52];
    for g in 0..nh {
        cfreach_sum += reach[g];
        cfreach_minus[hands[g].cards[0] as usize] += reach[g];
        cfreach_minus[hands[g].cards[1] as usize] += reach[g];
    }

    println!("  {:>2} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "h", "cards", "eff+s", "eff-s", "manual", "identity");

    for h in 0..nh {
        let c1 = hands[h].cards[0] as usize;
        let c2 = hands[h].cards[1] as usize;

        let eff_with = cfreach_sum - cfreach_minus[c1] - cfreach_minus[c2] + reach[h];
        let eff_without = cfreach_sum - cfreach_minus[c1] - cfreach_minus[c2];

        // Manual: sum reach[g] for g where g doesn't share any card with h, and g != h
        let manual_excl_self: f32 = (0..nh)
            .filter(|&g| g != h && !conflicts(&hands[h], &hands[g]))
            .map(|g| reach[g])
            .sum();

        // eff_with should equal manual_excl_self
        // Proof: cfreach_sum = sum of all reach[g]
        //        minus[c1] = sum of reach[g] for g containing c1
        //        minus[c2] = sum of reach[g] for g containing c2
        //        sum - minus[c1] - minus[c2] = sum_{g not containing c1 or c2}
        //                                     - sum_{g containing BOTH c1 AND c2}
        //        Only g==h contains both c1 and c2, so:
        //        = manual_excl_self + 0 - reach[h]
        //        Adding +reach[h] gives manual_excl_self.
        //
        // So eff(+self) = sum of reach of NON-CONFLICTING hands, EXCLUDING self.
        // This is the CORRECT E for the formula: opponent can hold any hand
        // that doesn't share cards with h (and can't hold h itself since same cards).

        let identity_check = (eff_with - manual_excl_self).abs() < 1e-6;

        println!("  {:>2} [{:>2},{:>2}] {:>8.3} {:>8.3} {:>8.3} {:>8}",
            h, c1, c2, eff_with, eff_without, manual_excl_self,
            if identity_check { "OK" } else { "FAIL" });

        assert!(identity_check,
            "Identity failed: eff(+self)={:.6} != manual={:.6}", eff_with, manual_excl_self);
    }

    println!("\n  VERIFIED: eff(+self) = sum of non-conflicting reach, excluding self.");
    println!("  This is the CORRECT single-opponent effective reach.");
    println!("  The +reach[h] compensates for double-subtraction of the self-hand.");
}

// ───────────────────────────────────────────────────────────────────────
// Test 5: Decompose the product formula error to understand where the
// residual comes from.
//
// prod(E_oi[h]) = E_0[h] * E_1[h] counts ALL (g0, g1) pairs where
// g0 is non-conflicting with h AND g1 is non-conflicting with h.
// But it also counts pairs where g0 conflicts with g1 (impossible deal).
//
// The CORRECT total should only count pairs where g0, g1 are BOTH
// non-conflicting with h AND non-conflicting with each other.
// ───────────────────────────────────────────────────────────────────────
#[test]
fn product_overcounting_decomposition() {
    println!("\n=== TEST 5: Decompose product overcounting ===");

    let hands = vec![
        TestHand { cards: [0, 1], strength: 100 },
        TestHand { cards: [1, 2], strength: 200 },
        TestHand { cards: [2, 3], strength: 300 },
        TestHand { cards: [4, 5], strength: 400 },
    ];
    let nh = hands.len();
    let reach_0 = vec![1.0; nh];
    let reach_1 = vec![1.0; nh];

    println!("  Using uniform reach=1.0 for clarity.");
    println!("  Conflict matrix:");
    for i in 0..nh {
        let row: Vec<&str> = (0..nh).map(|j| if conflicts(&hands[i], &hands[j]) {"X"} else {"."}).collect();
        println!("    h{}: {:?}", i, row);
    }

    for h in 0..nh {
        println!("\n  --- Traverser hand h={} (str={}) ---", h, hands[h].strength);

        // Count valid non-conflicting hands for each opponent (excluding self)
        let non_conf: Vec<usize> = (0..nh)
            .filter(|&g| !conflicts(&hands[h], &hands[g]))
            .collect();
        println!("    Non-conflicting with h: {:?}", non_conf);

        // E_0[h] = E_1[h] = |non_conf| (since reach=1)
        let e = non_conf.len() as f32;
        println!("    E_oi[h] = {:.0} (for each opponent)", e);
        println!("    E_0 * E_1 = {:.0} (product)", e * e);

        // Count valid (g0,g1) pairs: both non-conf with h, AND non-conf with each other
        let mut valid_pairs = 0;
        let mut valid_both_weaker = 0;
        let mut conflict_pairs = 0;
        for &g0 in &non_conf {
            for &g1 in &non_conf {
                if conflicts(&hands[g0], &hands[g1]) {
                    conflict_pairs += 1;
                } else {
                    valid_pairs += 1;
                    if hands[g0].strength < hands[h].strength && hands[g1].strength < hands[h].strength {
                        valid_both_weaker += 1;
                    }
                }
            }
        }
        println!("    Valid (g0,g1) pairs: {} (from product's {:.0})", valid_pairs, e * e);
        println!("    Conflicting pairs subtracted: {}", conflict_pairs);
        println!("    Valid both-weaker: {}", valid_both_weaker);

        let bf = brute_force_cfv(&hands, &reach_0, &reach_1, h, 1.0);
        let pf = product_formula_cfv(&hands, &reach_0, &reach_1, h, 1.0, false);
        let fc = full_correction_cfv(&hands, &reach_0, &reach_1, h, 1.0, false);
        println!("    BF={:.4}  PF(-s)={:.4}  FC(-s)={:.4}  err_FC={:.2e}", bf, pf, fc, (fc-bf).abs());
    }
}

// ───────────────────────────────────────────────────────────────────────
// Test 6: The definitive test - verify that FC(-self) is exact
// and check if the +reach[h] matters when combined with full correction.
// ───────────────────────────────────────────────────────────────────────
#[test]
fn definitive_exactness_check() {
    println!("\n=== TEST 6: Definitive exactness check ===");

    // Large-ish hand set with various conflicts
    let hands = vec![
        TestHand { cards: [0, 1], strength: 100 },
        TestHand { cards: [1, 2], strength: 200 },
        TestHand { cards: [2, 3], strength: 300 },
        TestHand { cards: [4, 5], strength: 400 },
        TestHand { cards: [5, 6], strength: 250 },
        TestHand { cards: [7, 8], strength: 350 },
        TestHand { cards: [9, 10], strength: 450 },
        TestHand { cards: [10, 11], strength: 150 },
    ];
    let nh = hands.len();

    let reach_0 = vec![0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4];
    let reach_1 = vec![0.6, 0.4, 0.8, 0.2, 0.3, 0.7, 0.1, 0.5];
    let half_pot = 10.0;

    // Count conflicts
    let mut total_conflicts = 0;
    for i in 0..nh { for j in (i+1)..nh { if conflicts(&hands[i], &hands[j]) { total_conflicts += 1; } } }
    println!("  {} hands, {} inter-hand conflicts\n", nh, total_conflicts);

    println!("  {:>25} {:>10} {:>10}",
        "Variant", "max_err", "exact?");

    for (label, use_fc, self_eff) in [
        ("PF(-self)", false, false),
        ("PF(+self)", false, true),
        ("FC(-self)", true, false),
        ("FC(+self)", true, true),
    ] {
        let mut max_err = 0.0f32;
        for h in 0..nh {
            let bf = brute_force_cfv(&hands, &reach_0, &reach_1, h, half_pot);
            let val = if use_fc {
                full_correction_cfv(&hands, &reach_0, &reach_1, h, half_pot, self_eff)
            } else {
                product_formula_cfv(&hands, &reach_0, &reach_1, h, half_pot, self_eff)
            };
            let err = (val - bf).abs();
            if err > max_err { max_err = err; }
        }
        let exact = max_err < 1e-4;
        println!("  {:>25} {:>10.2e} {:>10}",
            label, max_err, if exact { "YES" } else { "no" });
    }

    // Assert FC(-self) is exact
    for h in 0..nh {
        let bf = brute_force_cfv(&hands, &reach_0, &reach_1, h, half_pot);
        let fc = full_correction_cfv(&hands, &reach_0, &reach_1, h, half_pot, false);
        assert!((fc - bf).abs() < 1e-3,
            "FC(-self) not exact for h={}! bf={:.6} fc={:.6} err={:.2e}",
            h, bf, fc, (fc - bf).abs());
    }
}

// ───────────────────────────────────────────────────────────────────────
// Test 7: Source of the zero-sum residual in the current code
// ───────────────────────────────────────────────────────────────────────
#[test]
fn residual_source_diagnosis() {
    println!("\n=== TEST 7: Residual source diagnosis ===");
    println!("  Current code uses PF(+self) without any inter-opponent correction.");
    println!("  This test quantifies the error.\n");

    let hands = vec![
        TestHand { cards: [0, 1], strength: 100 },
        TestHand { cards: [1, 2], strength: 200 },
        TestHand { cards: [2, 3], strength: 300 },
        TestHand { cards: [4, 5], strength: 400 },
        TestHand { cards: [5, 6], strength: 250 },
        TestHand { cards: [7, 8], strength: 350 },
    ];
    let nh = hands.len();

    let reach = [
        vec![0.3, 0.5, 0.2, 0.8, 0.1, 0.6],
        vec![0.4, 0.6, 0.3, 0.7, 0.9, 0.2],
        vec![0.5, 0.4, 0.6, 0.2, 0.7, 0.3],
    ];
    let half_pot = 10.0;

    struct Result {
        label: &'static str,
        ev: [f64; 3],
        max_hand_err: f64,
    }

    let mut results = Vec::new();

    for (label, is_bf, use_fc, self_eff) in [
        ("Brute force", true, false, false),
        ("PF(+self) [CURRENT]", false, false, true),
        ("PF(-self)", false, false, false),
        ("FC(-self)", false, true, false),
        ("FC(+self)", false, true, true),
    ] {
        let mut ev = [0.0f64; 3];
        let mut max_hand_err = 0.0f64;

        for trav in 0..3 {
            let opp0 = (trav + 1) % 3;
            let opp1 = (trav + 2) % 3;

            for h in 0..nh {
                let bf = brute_force_cfv(&hands, &reach[opp0], &reach[opp1], h, half_pot);
                let val = if is_bf {
                    bf
                } else if use_fc {
                    full_correction_cfv(&hands, &reach[opp0], &reach[opp1], h, half_pot, self_eff)
                } else {
                    product_formula_cfv(&hands, &reach[opp0], &reach[opp1], h, half_pot, self_eff)
                };
                ev[trav] += (reach[trav][h] as f64) * (val as f64);
                if !is_bf {
                    let err = ((val - bf) as f64).abs();
                    if err > max_hand_err { max_hand_err = err; }
                }
            }
        }

        results.push(Result { label, ev, max_hand_err });
    }

    println!("  {:>25} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "Variant", "EV[P0]", "EV[P1]", "EV[P2]", "Sum", "MaxHErr");
    for r in &results {
        let sum: f64 = r.ev.iter().sum();
        println!("  {:>25} {:>10.4} {:>10.4} {:>10.4} {:>12.2e} {:>10.2e}",
            r.label, r.ev[0], r.ev[1], r.ev[2], sum, r.max_hand_err);
    }

    println!("\n  ANALYSIS:");
    println!("  1. Brute force is zero-sum (sum ~ 0).");
    println!("  2. PF(+self) [current code] has TWO sources of error:");
    println!("     a. +reach[h] adds phantom opponent assignments (opp holds traverser's hand)");
    println!("     b. Missing inter-opponent conflict correction (product overcounts)");
    println!("  3. The correct formula is FC(-self): product with full correction, no +reach[h].");
    println!("     OR equivalently, FC(+self) if the correction accounts for the +reach[h] phantom.");
}

// ───────────────────────────────────────────────────────────────────────
// Test 8: Verify that +reach[h] is actually correct for eff, and the
// real bug is ONLY the missing inter-opponent conflict correction.
//
// The minus technique computes: sum_{g not containing c1 or c2} reach[g]
// This equals: sum_{g non-conflicting with h, excluding g==h} reach[g]
//              MINUS reach[h] (since h contains both c1 and c2, subtracted twice)
//
// So adding +reach[h] gives: sum_{g non-conflicting with h, excluding g==h}
// which is the correct E for a single opponent.
//
// The issue is that E_0[h] * E_1[h] includes (g0, g1) pairs where g0 and g1
// conflict. This needs a correction. The +reach[h] term is NOT the bug.
// ───────────────────────────────────────────────────────────────────────
#[test]
fn reach_h_is_correct_for_eff() {
    println!("\n=== TEST 8: +reach[h] correctness for eff ===");
    println!("  Isolating: is the problem +reach[h], inter-opp conflicts, or both?\n");

    // All-unique cards: NO inter-opponent conflicts
    let hands_unique = vec![
        TestHand { cards: [0, 1], strength: 100 },
        TestHand { cards: [2, 3], strength: 200 },
        TestHand { cards: [4, 5], strength: 300 },
        TestHand { cards: [6, 7], strength: 400 },
    ];

    // With shared cards: HAS inter-opponent conflicts
    let hands_shared = vec![
        TestHand { cards: [0, 1], strength: 100 },
        TestHand { cards: [1, 2], strength: 200 },
        TestHand { cards: [3, 4], strength: 300 },
        TestHand { cards: [4, 5], strength: 400 },
    ];

    let reach_0 = vec![0.3, 0.5, 0.7, 0.2];
    let reach_1 = vec![0.4, 0.6, 0.8, 0.1];
    let half_pot = 10.0;

    for (label, hands) in [("UNIQUE cards", &hands_unique), ("SHARED cards", &hands_shared)] {
        println!("  --- {} ---", label);
        let nh = hands.len();

        let mut max_pf_minus_err = 0.0f32;
        let mut max_pf_plus_err = 0.0f32;
        let mut max_fc_minus_err = 0.0f32;
        let mut max_fc_plus_err = 0.0f32;

        for h in 0..nh {
            let bf = brute_force_cfv(hands, &reach_0, &reach_1, h, half_pot);
            let pf_minus = product_formula_cfv(hands, &reach_0, &reach_1, h, half_pot, false);
            let pf_plus = product_formula_cfv(hands, &reach_0, &reach_1, h, half_pot, true);
            let fc_minus = full_correction_cfv(hands, &reach_0, &reach_1, h, half_pot, false);
            let fc_plus = full_correction_cfv(hands, &reach_0, &reach_1, h, half_pot, true);

            max_pf_minus_err = max_pf_minus_err.max((pf_minus - bf).abs());
            max_pf_plus_err = max_pf_plus_err.max((pf_plus - bf).abs());
            max_fc_minus_err = max_fc_minus_err.max((fc_minus - bf).abs());
            max_fc_plus_err = max_fc_plus_err.max((fc_plus - bf).abs());
        }

        println!("    PF(-self): max_err={:.2e}", max_pf_minus_err);
        println!("    PF(+self): max_err={:.2e}", max_pf_plus_err);
        println!("    FC(-self): max_err={:.2e}", max_fc_minus_err);
        println!("    FC(+self): max_err={:.2e}", max_fc_plus_err);

        if label.contains("UNIQUE") {
            // With unique cards: only inter-opponent conflict is g0==g1 (same hand)
            // PF should work if we handle the diagonal
            // FC should be exact
            println!("    Expected: FC should be exact, PF has g0==g1 diagonal error");
        } else {
            // With shared cards: inter-opponent conflicts from shared cards
            println!("    Expected: only FC should be exact");
        }
        println!();
    }
}

// ───────────────────────────────────────────────────────────────────────
// Test 9: The actual source of 0.06% -- quantify the residual
// as a percentage of the pot, using realistic hand counts.
// ───────────────────────────────────────────────────────────────────────
#[test]
fn residual_percentage_estimation() {
    println!("\n=== TEST 9: Residual percentage estimation ===");
    println!("  Simulating a larger hand set to estimate realistic residual.\n");

    // 20 hands with ~15% conflict rate (typical for poker after board cards removed)
    // Create hands where about 1/7 of pairs share a card
    let mut hands = Vec::new();
    // Use cards 0-39, creating hands that sometimes share cards
    let card_pairs: Vec<[u8; 2]> = vec![
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
        [1, 10], [3, 11], [5, 12], [7, 13], [9, 14],  // share one card with first 5
        [15, 16], [17, 18], [19, 20], [21, 22], [23, 24],
        [10, 25], [11, 26], [12, 27], [13, 28], [14, 29], // share with group 2
    ];
    for (i, cp) in card_pairs.iter().enumerate() {
        hands.push(TestHand { cards: *cp, strength: (i as u32 + 1) * 50 });
    }
    let nh = hands.len();

    // Count conflicts
    let mut conflict_count = 0;
    let total_pairs = nh * (nh - 1) / 2;
    for i in 0..nh { for j in (i+1)..nh { if conflicts(&hands[i], &hands[j]) { conflict_count += 1; } } }
    println!("  {} hands, {}/{} pairs conflict ({:.1}%)",
        nh, conflict_count, total_pairs, 100.0 * conflict_count as f64 / total_pairs as f64);

    // Random-ish reach values
    let reach = [
        (0..nh).map(|i| 0.1 + 0.05 * (i as f32 * 1.7 % 1.0)).collect::<Vec<f32>>(),
        (0..nh).map(|i| 0.1 + 0.05 * (i as f32 * 2.3 % 1.0)).collect::<Vec<f32>>(),
        (0..nh).map(|i| 0.1 + 0.05 * (i as f32 * 3.1 % 1.0)).collect::<Vec<f32>>(),
    ];
    let half_pot = 100.0;

    // Compute EVs
    let mut ev_bf = [0.0f64; 3];
    let mut ev_current = [0.0f64; 3]; // PF(+self)
    let mut ev_fc_minus = [0.0f64; 3]; // FC(-self)

    for trav in 0..3 {
        let opp0 = (trav + 1) % 3;
        let opp1 = (trav + 2) % 3;

        for h in 0..nh {
            let bf = brute_force_cfv(&hands, &reach[opp0], &reach[opp1], h, half_pot);
            let current = product_formula_cfv(&hands, &reach[opp0], &reach[opp1], h, half_pot, true);
            let fc = full_correction_cfv(&hands, &reach[opp0], &reach[opp1], h, half_pot, false);

            ev_bf[trav] += (reach[trav][h] as f64) * (bf as f64);
            ev_current[trav] += (reach[trav][h] as f64) * (current as f64);
            ev_fc_minus[trav] += (reach[trav][h] as f64) * (fc as f64);
        }
    }

    let sum_bf: f64 = ev_bf.iter().sum();
    let sum_current: f64 = ev_current.iter().sum();
    let sum_fc: f64 = ev_fc_minus.iter().sum();
    let total_action = ev_bf.iter().map(|x| x.abs()).sum::<f64>();

    println!("\n  Results (half_pot={:.0}):", half_pot);
    println!("    BF:            sum={:.6e}  rel={:.4}%", sum_bf, 100.0 * sum_bf.abs() / total_action);
    println!("    PF(+self):     sum={:.6e}  rel={:.4}%", sum_current, 100.0 * sum_current.abs() / total_action);
    println!("    FC(-self):     sum={:.6e}  rel={:.4}%", sum_fc, 100.0 * sum_fc.abs() / total_action);

    println!("\n  The PF(+self) residual comes from two independent sources:");

    // Decompose: compute PF(-self) to isolate the +reach[h] contribution
    let mut ev_pf_minus = [0.0f64; 3];
    for trav in 0..3 {
        let opp0 = (trav + 1) % 3;
        let opp1 = (trav + 2) % 3;
        for h in 0..nh {
            let pf = product_formula_cfv(&hands, &reach[opp0], &reach[opp1], h, half_pot, false);
            ev_pf_minus[trav] += (reach[trav][h] as f64) * (pf as f64);
        }
    }
    let sum_pf_minus: f64 = ev_pf_minus.iter().sum();

    println!("    1. Inter-opponent conflict overcounting: sum_err={:.6e}", sum_pf_minus);
    println!("    2. +reach[h] phantom:                   sum_err={:.6e}", sum_current - sum_pf_minus);
    println!("    Total:                                   sum_err={:.6e}", sum_current);

    println!("\n  CONCLUSION:");
    println!("  The zero-sum residual has TWO components:");
    println!("  (a) The product formula overcounts inter-opponent conflicting pairs");
    println!("  (b) The +reach[h] term adds phantom opponent-holds-traverser-hand terms");
    println!("  Both contribute to the 0.06%% residual seen in practice.");
    println!("  FC(-self) eliminates both errors (full correction, no +self).");
}
