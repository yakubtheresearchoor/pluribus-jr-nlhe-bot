// Regression test for the folded-traverser side-pot return bug.
//
// THE BUG (fixed many sessions ago):
//   When a folded player contributed MORE than the maximum active player's
//   contribution, the folded player should receive back their unmatched
//   excess from the side-pot tier where no active player is eligible. The
//   old fast-path code at side_pot_showdown_cfv's line ~258 used
//   `payoff = -traverser_investment` (full stake lost) for any folded
//   traverser, missing the return.
//
//   LATENT in 3-player games because without raises you cannot reach a
//   terminal where a folded player has contribution > max active player —
//   you only fold if you don't want to call, so your contribution at fold
//   time equals the previous level and active players continue past it.
//
//   FIRST REACHABLE in 4-plus-player games because contribution patterns
//   exist where the player with the highest pot-going-in folds (e.g., a
//   raise, two callers, then the original raiser folds to a re-raise that
//   only one caller can match).
//
// THE ZERO-SUM CANCELLATION TRAP:
//   The bug shows up as a per-player cfv error of e.g. -195 instead of
//   -100 for the folded high-contributor. But by the Σ_p payoff_p = 0
//   identity (total pot redistributed adds to zero), the aggregate
//   Σ_p Σ_h r[h] · cfv_p[h] still came out to zero — meaning a pure
//   zero-sum check at the terminal cannot detect this bug. The aggregate
//   cancels even when the individuals are wrong. This is why the docket
//   explicitly says: "asserts the individual per-player values, not just
//   aggregate zero-sum, because the aggregate cancels even when the
//   individuals are wrong."
//
// THIS TEST:
//   Constructs the exact scenario where the bug fired: 4p, contribs
//   [190, 5, 5, 95], fold_mask = 0b0011 (p0 and p1 folded). p0 contributed
//   190 (max), then folded — so the side pot at the [95, 190] tier has
//   only p0 as a contributor and NO active eligibles, and should return
//   95 to p0 per scenario.
//
//   Computes cfv for each traverser via side_pot_showdown_cfv with
//   uniform reach over a symmetric 5-hand sample, sums Σ_h r·cfv per
//   player, and ASSERTS EACH INDIVIDUAL VALUE against the expected.
//   Under the bug, p0's value would be -195 instead of -100 (and the
//   sum would be -95, not zero). Under the fix, p0 = -100 exactly and
//   the sum is zero.
//
//   IF THIS TEST FAILS WITHOUT OTHER CHANGES, the side-pot return
//   handling has regressed. Do not weaken the assertions.

use solver_core::card::index_to_card_pair;
use solver_core::solver::showdown::side_pot_showdown_cfv;

#[test]
fn folded_traverser_with_higher_contribution_than_max_active_returns_unmatched_side_pot() {
    // 5 pairwise-disjoint hands with distinct strengths.
    // Pairwise-disjoint means every K-tuple from these 5 hands is valid;
    // distinct strengths means there are no ties in main pot.
    let nh = 5;
    let hand_cards: Vec<u8> = vec![
        0, 1,   // h=0
        2, 3,   // h=1
        4, 5,   // h=2
        6, 7,   // h=3
        8, 9,   // h=4
    ];
    let hand_strength: Vec<u16> = vec![100, 80, 60, 40, 20];

    // Sorted-by-strength arrays (ascending strength).
    // sorted_pl: indices sorted by player hand strength ascending.
    // sorted_opp: same (opponents see the same hand set).
    let mut sp_pairs: Vec<(u16, u16)> = (0..nh).map(|h| (hand_strength[h], h as u16)).collect();
    sp_pairs.sort_by_key(|&(s, _)| s);
    let sorted_pl_str: Vec<u16> = sp_pairs.iter().map(|&(s, _)| s).collect();
    let sorted_pl_idx: Vec<u16> = sp_pairs.iter().map(|&(_, i)| i).collect();

    let np = 4u8;
    let num_opp = (np as usize) - 1;
    let starting_pot = 20i32;  // 4p, np × 5

    // sorted_opp_str/idx: per-opponent, same as sorted_pl since all opps
    // see the same hands.
    let mut sorted_opp_str = vec![0u16; num_opp * nh];
    let mut sorted_opp_idx = vec![0u16; num_opp * nh];
    for oi in 0..num_opp {
        for h in 0..nh {
            sorted_opp_str[oi * nh + h] = sorted_pl_str[h];
            sorted_opp_idx[oi * nh + h] = sorted_pl_idx[h];
        }
    }

    // The bug case: p0 contributed 190 then folded; p3 has 95 (max active);
    // p2 has 5; p1 contributed 5 then folded. Fold mask = 0b0011 (p0, p1).
    let contributions = vec![190i32, 5, 5, 95];
    let fold_mask: u16 = 0b0011;

    // Uniform reach over the 5 hands for each opponent.
    let reach_vec: Vec<Vec<f32>> = (0..num_opp).map(|_| vec![1.0f32; nh]).collect();
    let reach_views: Vec<&[f32]> = reach_vec.iter().map(|v| v.as_slice()).collect();

    // Compute num_combinations (= ordered valid 4-tuples for normalization).
    // All 5 hands are pairwise disjoint, so any 4 distinct hands form a
    // valid tuple: 5 * 4 * 3 * 2 = 120.
    let nc = 5.0 * 4.0 * 3.0 * 2.0;

    eprintln!("=== folded_traverser_sidepot_regression ===");
    eprintln!("Setup: 4p, contribs = {:?}, fold_mask = {:#b}", contributions, fold_mask);
    eprintln!("Stakes: starting_pot/np = {} per player + their contribution",
        starting_pot as f64 / np as f64);
    eprintln!("Pot structure:");
    eprintln!("  Level 5  (4 contributors × 5 + sp 20) = pot 40, eligibles: p2, p3 (active)");
    eprintln!("  Level 95 (2 contributors × 90)        = pot 180, eligibles: p3 only");
    eprintln!("  Level 190 (1 contributor × 95)        = pot 95,  eligibles: NONE → return 95 to p0");

    let mut per_player = [0.0f64; 4];

    for traverser in 0..(np as usize) {
        // Opp reach for this traverser (omit traverser's own slot).
        let opp_reach: Vec<&[f32]> = (0..(np as usize))
            .filter(|&p| p != traverser)
            .map(|p| reach_views[if p < traverser { p } else { p - 1 }])
            .collect();

        let cfv = side_pot_showdown_cfv(
            &opp_reach, &hand_cards, nh,
            &sorted_opp_str, &sorted_opp_idx,
            &sorted_pl_str, &sorted_pl_idx,
            &contributions, fold_mask, traverser,
            np, starting_pot,
        );

        // Σ_h r[h] · cfv[h] / nc, with uniform reach.
        let sum: f64 = (0..nh).map(|h| cfv[h] as f64).sum::<f64>() / nc;
        per_player[traverser] = sum;
        eprintln!("  traverser p{} (contrib={}, folded={}): Σ r·cfv/nc = {:.6}",
            traverser, contributions[traverser],
            (fold_mask & (1u16 << traverser)) != 0, sum);
    }

    // Expected per-player net values (per the per-scenario analysis):
    //   p0 (folded, contrib 190): stake = 195; gets 95 back from level 190
    //                              (no eligible active there); net = -100.
    //   p1 (folded, contrib 5):   stake = 10; no returns; net = -10.
    //   p2 (active, contrib 5):   stake = 10; main pot wins by symmetry
    //                              at rate 1/2 (vs p3); E[net] = 0.5·30 + 0.5·(-10) = 10.
    //   p3 (active, contrib 95):  stake = 100; gets side pot 180
    //                              unconditionally + 1/2 share of main 40
    //                              = 180 + 20 - 100 = 100.
    //
    // Σ E[net] = -100 - 10 + 10 + 100 = 0 ✓
    let expected = [-100.0f64, -10.0, 10.0, 100.0];

    eprintln!("\nExpected (post-fix):     {:?}", expected);
    eprintln!("Aggregate Σ per_player:  {:.6}", per_player.iter().sum::<f64>());

    let mut all_close = true;
    let tolerance = 1e-3; // f32 propagation through CFR scenarios
    for p in 0..4 {
        let diff = (per_player[p] - expected[p]).abs();
        let pass = diff < tolerance;
        eprintln!("  p{}: actual={:.6} expected={:.6} diff={:.3e} {}",
            p, per_player[p], expected[p], diff,
            if pass { "✓" } else { "✗" });
        if !pass { all_close = false; }
    }

    // Aggregate zero-sum still must hold (the easier check) — this catches
    // gross regressions on top of the per-player checks.
    let aggregate = per_player.iter().sum::<f64>();
    assert!(aggregate.abs() < tolerance,
        "Aggregate zero-sum failed: Σ per_player = {:.6e}", aggregate);

    // The HARD assertion: each per-player value individually matches the
    // expected. Under the bug, p0 would read -195 instead of -100 (and the
    // sum would be -95). Under the fix, p0 reads -100 exactly.
    assert!(all_close,
        "Per-player cfv values do not match the post-fix expected. \
         Under the OLD bug: p0 would read approximately -195 (full stake \
         lost), and the aggregate Σ would be -95, not 0. Either way, a \
         regression in side-pot-return handling for folded traversers with \
         higher contribution than max active.");

    // Specifically nail the diagnostic value: p0's net is EXACTLY -100
    // (not -195) ↔ the fix is in place.
    assert!((per_player[0] - (-100.0)).abs() < tolerance,
        "p0 (folded, max-contrib) net = {:.4}; expected -100 (post-fix). \
         If value is near -195, the side-pot return at the orphan level \
         is not being credited back to the folded high contributor.",
        per_player[0]);
}

#[allow(dead_code)]
fn unused() { let _ = index_to_card_pair; }
