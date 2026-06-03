// Extend the independent enumerator from #37 to 3-player fold terminals.
// Validates that the multiway path (which uses brute-force enumeration
// directly, not the buggy inclusion-exclusion formula) agrees with
// first-principles enumeration. Confirms the fix in #37 doesn't leave
// multiway-specific siblings, as the user explicitly flagged.

use solver_core::solver::showdown::side_pot_showdown_cfv;

#[test]
fn three_player_fold_terminal_against_independent_enumeration() {
    // Synthetic 3-player fold terminal: traverser=0 folded after
    // committing 5, p1 active at 15, p2 active at 15.
    // Hand set: 4 disjoint hands (no card overlap between hands).
    let nh = 4;
    let np = 3;
    let traverser = 0;
    let starting_pot = 5;
    let contributions = vec![5, 15, 15];
    let fold_mask: u16 = 0b001; // p0 folded
    // Distinct card encoding (avoids any inter-hand collision).
    let hand_cards: Vec<u8> = vec![
        46, 50,  // hand 0
        38, 42,  // hand 1
        30, 34,  // hand 2
        18, 26,  // hand 3
    ];
    // Each opponent has reach 0.25 for every hand (uniform).
    let opp_reach_p1: Vec<f32> = vec![0.25; nh];
    let opp_reach_p2: Vec<f32> = vec![0.25; nh];
    let opp_reach_views: Vec<&[f32]> = vec![&opp_reach_p1, &opp_reach_p2];

    // Sorted arrays — not used on fold-fast-path (the brute-force path
    // checks card conflicts directly) but required by the signature.
    let dummy_sorted_str: Vec<u16> = vec![100, 80, 60, 40]; // descending
    let dummy_sorted_idx: Vec<u16> = vec![0, 1, 2, 3];

    // Independent enumeration: for traverser holding h, enumerate (g1, g2)
    // pairs that don't conflict with h or each other; sum joint reach.
    let mut expected = vec![0.0f32; nh];
    let traverser_investment = starting_pot as f32 / np as f32 + 5.0; // = 5/3 + 5 ≈ 6.667
    let payoff = -traverser_investment;
    for h in 0..nh {
        let hc1 = hand_cards[h * 2];
        let hc2 = hand_cards[h * 2 + 1];
        let mut joint = 0.0f32;
        for g1 in 0..nh {
            let g1c1 = hand_cards[g1 * 2];
            let g1c2 = hand_cards[g1 * 2 + 1];
            if g1c1 == hc1 || g1c1 == hc2 || g1c2 == hc1 || g1c2 == hc2 { continue; }
            let r1 = opp_reach_p1[g1];
            for g2 in 0..nh {
                let g2c1 = hand_cards[g2 * 2];
                let g2c2 = hand_cards[g2 * 2 + 1];
                if g2c1 == hc1 || g2c1 == hc2 || g2c2 == hc1 || g2c2 == hc2 { continue; }
                if g2c1 == g1c1 || g2c1 == g1c2 || g2c2 == g1c1 || g2c2 == g1c2 { continue; }
                let r2 = opp_reach_p2[g2];
                joint += r1 * r2;
            }
        }
        expected[h] = payoff * joint;
    }

    eprintln!("\n=== 3p fold terminal: independent joint enumeration ===");
    eprintln!("payoff = -traverser_investment = -({}/3 + 5) ≈ {:.4}", starting_pot, payoff);
    for h in 0..nh {
        eprintln!("  h={}: expected = {:.6}", h, expected[h]);
    }

    // CPU side_pot_showdown_cfv for the same 3p case (goes through the
    // num_opp == 2 brute-force branch at line 313, NOT the num_opp == 1
    // fixed path).
    let cfv_cpu = side_pot_showdown_cfv(
        &opp_reach_views, &hand_cards, nh,
        &dummy_sorted_str, &dummy_sorted_idx,
        &dummy_sorted_str, &dummy_sorted_idx,
        &contributions, fold_mask, traverser, np as u8,
        starting_pot,
    );
    eprintln!("CPU side_pot_showdown_cfv: {:?}", cfv_cpu);

    // Compare element-wise.
    let mut max_diff = 0.0f32;
    for h in 0..nh {
        let d = (expected[h] - cfv_cpu[h]).abs();
        if d > max_diff { max_diff = d; }
    }
    eprintln!("max_diff vs independent enumeration: {:.6e}", max_diff);

    assert!(
        max_diff < 1e-4,
        "3p fold terminal: CPU = {:?} disagrees with independent enumeration {:?} \
         (max_diff = {}). The multiway brute-force path may have a separate bug \
         from the HU inclusion-exclusion bug already fixed in #37.",
        cfv_cpu, expected, max_diff
    );
    eprintln!("✓ 3p multiway fold-terminal CPU agrees with independent enumeration at f32 floor.");
}
