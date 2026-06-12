//! Pre-blueprint configuration-parameter verification.
//!
//! Required for the blueprint loop (P4, P5): verify in code that the
//! four "throughout" configuration parameters are actually configurable
//! AND actually flow to terminal payoffs / initial state. The same
//! "verify, don't assume from memory" discipline applied to every other
//! piece of the validation arc — requirement-established is not
//! implementation-confirmed.
//!
//! Parameters under test:
//!   1. ante (per-player dead money posted before action)
//!   2. rake_rate (fraction of pot taken by the house)
//!   3. rake_cap (maximum rake amount in chips)
//!   4. stack depth (per-player starting stack constraining max bet)
//!
//! Each parameter gets a verification test that:
//!   - Builds two trees varying ONLY this parameter
//!   - Verifies the parameter affects the expected piece of state
//!   - Failure = the parameter is configurable in storage but not
//!     wired to its intended effect

use solver_core::card::card_from_str;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{FlopStartVectorCfr, Zone, DcfrParams};
use solver_core::solver::game::GameSpec;
use solver_core::solver::showdown::{side_pot_showdown_cfv, side_pot_showdown_cfv_with_rake};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

// ─────────────────────────────────────────────────────────────────────
// Verification 1: ANTE flows to root contributions (initial state)
// ─────────────────────────────────────────────────────────────────────
//
// Convention: ante is folded into `initial_contributions` (no separate
// `ante` field exists). The 6-player test in tree_builder.rs sets
// initial_contributions = [sb, bb, ante, ante, ante, ante]. This test
// confirms the per-player ante values flow through to the tree's root
// contributions.

#[test]
fn verify_ante_flows_to_root_contributions() {
    let ante = 7i32;
    let sb = 50i32;
    let bb = 100i32;
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: sb + bb + 4 * ante,
        starting_stacks: vec![1000; 6],
        initial_contributions: vec![sb, bb, ante, ante, ante, ante],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&cfg).expect("tree builds");

    // ANTE assertions: each ante-paying position (2-5) has contribution == ante
    for p in 2..6 {
        assert_eq!(
            tree.get_contribution(0, p), ante,
            "player {} ante: tree.get_contribution(0, {}) = {} but config sets ante = {}",
            p, p, tree.get_contribution(0, p), ante,
        );
    }
    // Blinds are also configurable via initial_contributions
    assert_eq!(tree.get_contribution(0, 0), sb, "SB initial contribution");
    assert_eq!(tree.get_contribution(0, 1), bb, "BB initial contribution");

    eprintln!("✓ Ante (via initial_contributions) flows to root contributions");
}

// ─────────────────────────────────────────────────────────────────────
// Verification 2: STACK DEPTH bounds max betting (initial state)
// ─────────────────────────────────────────────────────────────────────
//
// max_committable(p) = starting_stacks[p] + initial_contributions[p].
// Bets / raises are clamped to max_committable. Verify by building two
// trees with different stack depths and confirming the deep-stack tree
// allows betting beyond what the short-stack tree allows.

#[test]
fn verify_stack_depth_bounds_max_betting() {
    fn make_cfg(stack: i32) -> TreeConfig {
        TreeConfig {
            num_players: 2,
            initial_state: BoardState::Flop,
            starting_pot: 10,
            starting_stacks: vec![stack, stack],
            initial_contributions: vec![0, 0],
            rake_rate: 0.0,
            rake_cap: 0.0,
            bet_sizes: BetSizeOptions {
                bet: vec![BetSize::PotRelative(2.0)], // 2× pot
                raise: vec![],
            },
            add_allin_threshold: 1.0,
            force_allin_threshold: 1.0,
            merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
        }

    }

    let short = build_tree(&make_cfg(20)).expect("short-stack tree builds");
    let deep = build_tree(&make_cfg(1000)).expect("deep-stack tree builds");

    // The deep-stack tree should have higher max contributions at terminals
    // than the short-stack tree, because betting can go further.
    let short_max: i32 = (0..short.num_nodes())
        .map(|i| (0..2).map(|p| short.get_contribution(i, p)).max().unwrap_or(0))
        .max()
        .unwrap_or(0);
    let deep_max: i32 = (0..deep.num_nodes())
        .map(|i| (0..2).map(|p| deep.get_contribution(i, p)).max().unwrap_or(0))
        .max()
        .unwrap_or(0);

    eprintln!(
        "Short-stack (stack=20) max terminal contribution: {}; \
         deep-stack (stack=1000) max: {}",
        short_max, deep_max
    );
    assert!(
        deep_max > short_max,
        "deep-stack tree should allow larger commitments than short-stack tree, \
         but deep_max = {} <= short_max = {}. Stack depth does not bound max betting.",
        deep_max, short_max
    );
    // Specifically, short-stack max should be bounded by 20 + 0 = 20
    assert!(
        short_max <= 20,
        "short-stack max commitment {} exceeds starting_stack=20 + initial_contribution=0 = 20",
        short_max
    );

    eprintln!("✓ Stack depth bounds max betting (verified end-to-end at tree level)");
}

// ─────────────────────────────────────────────────────────────────────
// Verification 3: STARTING POT flows to terminal payoffs
// ─────────────────────────────────────────────────────────────────────
//
// `side_pot_showdown_cfv` takes `starting_pot: i32` and uses it in the
// pot total: `total_pot = starting_pot + sum(contributions)`. Verify by
// calling the showdown directly with different starting_pot values and
// observing the CFV changes.

#[test]
fn verify_starting_pot_flows_to_terminal_payoffs() {
    // Two non-conflicting combos: combo 0 uses cards (0,1), combo 1
    // uses cards (2,3). With nh=1, the inclusion-exclusion correction
    // forces cfreach=0 (the only opp combo IS the player's combo). With
    // nh>=2 and disjoint card sets, cfreach is nonzero and we can
    // observe the payoff in the CFV.
    let nh = 2usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8];
    let opp_reach_data = vec![1.0f32; nh];
    let opp_reach_slices: Vec<&[f32]> = vec![&opp_reach_data];
    let sorted_str = vec![100u16, 200u16];
    let sorted_idx = vec![0u16, 1u16];

    // Two-player traverser-folded scenario:
    // - traverser=0 folds (fold_mask bit 0 set)
    // - opp is active alone (num_active=1) → fast-path constant payoff branch
    // - traverser_investment = starting_pot/2 + contributions[0]
    // - payoff = -traverser_investment (folded player loses their investment)
    //
    // For starting_pot=10, c_t=5: payoff = -(10/2 + 5) = -10
    // For starting_pot=50, c_t=5: payoff = -(50/2 + 5) = -30
    //
    // CFV[h] = payoff × cfreach. With single hand + uniform opp_reach,
    // cfreach is nonzero, so CFV magnitudes should differ between
    // starting_pot=10 and =50.
    let contributions = vec![5i32, 5i32];
    let fold_mask = 1u16 << 0;

    // Try with starting_pot = 10
    let cfv_small_pot = side_pot_showdown_cfv(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 10,
    );

    // Try with starting_pot = 50
    let cfv_large_pot = side_pot_showdown_cfv(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 50,
    );

    eprintln!(
        "side_pot_showdown_cfv: starting_pot=10 → CFV = {:?}; starting_pot=50 → CFV = {:?}",
        cfv_small_pot, cfv_large_pot
    );

    // The CFV at the showdown is reach × payoff. Reach is 1.0; payoff
    // depends on starting_pot. Larger starting_pot → larger CFV.
    assert!(
        cfv_small_pot.iter().zip(&cfv_large_pot)
            .any(|(s, l)| (s - l).abs() > 1e-3),
        "starting_pot does not flow to CFV: starting_pot=10 gave {:?}, starting_pot=50 gave {:?} \
         — these should differ if starting_pot affects payoffs.",
        cfv_small_pot, cfv_large_pot,
    );

    eprintln!("✓ starting_pot flows to terminal payoffs");
}

// ─────────────────────────────────────────────────────────────────────
// Verification 4: RAKE_RATE and RAKE_CAP affect terminal payoffs
// ─────────────────────────────────────────────────────────────────────
//
// THIS TEST DOCUMENTS A GAP. The TreeConfig and FlatTree both store
// rake_rate and rake_cap, but `side_pot_showdown_cfv` (showdown.rs:232)
// does NOT take rake parameters in its signature — they cannot affect
// terminal payoffs because they're not even passed.
//
// Structural smoking gun: any grep of the solver/GPU code for "rake"
// returns only the storage (TreeConfig, FlatTree, builder pass-through).
// No solver code reads rake from a tree instance.
//
// This test is structured to PASS when the implementation is correct
// (rake affects payoffs) and FAIL when the gap is present (rake has
// no effect). Currently the gap is present, so the test is `#[ignore]`'d
// with explanation. Removing #[ignore] when rake is implemented should
// surface a PASS, confirming the implementation.
//
// Implementation needed:
//   - Add `rake_rate: f32, rake_cap: i32` parameters to
//     side_pot_showdown_cfv
//   - In the payoff calculation: `rake = min(total_pot * rake_rate, rake_cap)`
//   - Winner payoffs scaled by `(total_pot - rake) / total_pot`
//   - Thread rake from FlatTree → game.evaluate_terminal → showdown
//   - Run all existing tests (use rake_rate=0.0, should pass unchanged)
//   - Re-enable this test (removes #[ignore])

// Hand-computed rake reference for the FAST PATH (folded or single-active).
//
// Case: HU, traverser=0 ACTIVE (NOT folded), num_opp=1 OPP IS FOLDED.
// This triggers the fast path (num_active <= 1) with traverser as the
// active winner. With:
//   contributions = [50, 50]  (each player put in 50)
//   starting_pot = 0
//   total_pot = 0 + 100 = 100
//   rake_rate = 0.05, rake_cap = 1000 (cap not binding)
//   rake = min(100 * 0.05, 1000) = 5
// Expected:
//   traverser_investment = 0/2 + 50 = 50
//   payoff (active winner) = (total_pot - rake) - investment = 95 - 50 = 45
// Compare with rake-free baseline:
//   payoff = total_pot - investment = 100 - 50 = 50
// Difference: rake = 5. CFV[h] = payoff × cfreach. With nh=2 non-conflict
// combos and uniform opp_reach=1, cfreach[h]=1, so CFV[h] = payoff.
//   Rake-free CFV: [50, 50]
//   Rake CFV:      [45, 45]

#[test]
fn verify_rake_fast_path_matches_hand_computed_reference() {
    let nh = 2usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8];
    // OPP folded (fold_mask bit 1). Traverser=0 active winner.
    let opp_reach_data = vec![1.0f32; nh];
    let opp_reach_slices: Vec<&[f32]> = vec![&opp_reach_data];
    let sorted_str = vec![100u16, 200u16];
    let sorted_idx = vec![0u16, 1u16];
    let contributions = vec![50i32, 50i32];
    let fold_mask: u16 = 1u16 << 1; // opp folded

    // Rake-free baseline (uses the wrapper, which passes 0.0, 0.0)
    let cfv_no_rake = side_pot_showdown_cfv(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 0,
    );
    // With rake: 5% rate, cap=1000 (not binding)
    let cfv_with_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 0,
        0.05, 1000.0, true,
    );

    eprintln!("fast-path active-winner test:");
    eprintln!("  rake-free CFV: {:?}", cfv_no_rake);
    eprintln!("  rake-5%  CFV: {:?}", cfv_with_rake);

    // Hand-computed: rake-free payoff = 50; rake payoff = 45.
    // CFV[h] = payoff × cfreach. With nh=2 non-conflict, cfreach=1.
    // (Actually cfreach = opp_reach_sum - opp_reach_minus[c1] - opp_reach_minus[c2]
    //  + opp_reach[h] = 2 - 1 - 1 + 1 = 1; same for h=1)
    let expected_no_rake = [50.0f32, 50.0f32];
    let expected_with_rake = [45.0f32, 45.0f32];
    for h in 0..nh {
        assert!((cfv_no_rake[h] - expected_no_rake[h]).abs() < 1e-4,
            "rake-free CFV[{}] = {}, expected {} (hand-computed)",
            h, cfv_no_rake[h], expected_no_rake[h]);
        assert!((cfv_with_rake[h] - expected_with_rake[h]).abs() < 1e-4,
            "rake-5% CFV[{}] = {}, expected {} (hand-computed: \
             (total_pot=100 - rake=5) - investment=50 = 45)",
            h, cfv_with_rake[h], expected_with_rake[h]);
    }

    eprintln!("✓ Fast path (active winner, opp folded): rake applied correctly");
    eprintln!("  Hand-computed: payoff = (pot - rake) - investment = (100-5) - 50 = 45");
}

#[test]
fn verify_rake_cap_binds() {
    // Same setup as above, but rake_cap = 3 (less than total_pot * rake_rate = 5).
    // Expected rake = min(5, 3) = 3.
    // payoff = 100 - 3 - 50 = 47.
    let nh = 2usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8];
    let opp_reach_data = vec![1.0f32; nh];
    let opp_reach_slices: Vec<&[f32]> = vec![&opp_reach_data];
    let sorted_str = vec![100u16, 200u16];
    let sorted_idx = vec![0u16, 1u16];
    let contributions = vec![50i32, 50i32];
    let fold_mask: u16 = 1u16 << 1;

    let cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 0,
        0.05, 3.0, true, // cap = 3 < uncapped rake = 5
    );
    eprintln!("rake-cap-binding test: CFV = {:?}", cfv);
    let expected = [47.0f32, 47.0f32];
    for h in 0..nh {
        assert!((cfv[h] - expected[h]).abs() < 1e-4,
            "CFV[{}] = {}, expected {} (rake capped at 3: payoff = 100-3-50)",
            h, cfv[h], expected[h]);
    }
    eprintln!("✓ rake_cap binds correctly when total_pot × rake_rate exceeds cap");
}

#[test]
fn verify_rake_does_not_apply_to_folded_traverser() {
    // Traverser folded → payoff = −investment, unchanged by rake.
    // Setup: same pot, but traverser=0 folded.
    let nh = 2usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8];
    let opp_reach_data = vec![1.0f32; nh];
    let opp_reach_slices: Vec<&[f32]> = vec![&opp_reach_data];
    let sorted_str = vec![100u16, 200u16];
    let sorted_idx = vec![0u16, 1u16];
    let contributions = vec![50i32, 50i32];
    let fold_mask: u16 = 1u16 << 0; // traverser folded

    let cfv_no_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 0,
        0.0, 0.0, true,
    );
    let cfv_with_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 0,
        0.05, 1000.0, true,
    );
    eprintln!("folded-traverser test:");
    eprintln!("  rake-free CFV: {:?}", cfv_no_rake);
    eprintln!("  rake-5%  CFV: {:?}", cfv_with_rake);
    for h in 0..nh {
        assert!(
            (cfv_no_rake[h] - cfv_with_rake[h]).abs() < 1e-6,
            "folded traverser CFV[{}] should be RAKE-INVARIANT (loser doesn't pay rake); \
             rake-free = {}, rake-5% = {}",
            h, cfv_no_rake[h], cfv_with_rake[h],
        );
    }
    eprintln!("✓ Folded traverser payoff is rake-invariant (rake comes out of winner's claim)");
}

// ─────────────────────────────────────────────────────────────────────
// Slice 2 HU residual fix: fold-win-after-bet (uncalled-bet returned un-raked)
// ─────────────────────────────────────────────────────────────────────
//
// COVERAGE GAP CORRECTION (the lead spec confirmation 2026-06-04): the
// Slice 1.x anchors only covered fold-win with EQUAL contributions
// (e.g., the verify_rake_fast_path_matches_hand_computed_reference
// test uses contributions=[50, 50] where main_pot == total_pot, so
// the rake-on-total-pot vs rake-on-main-pot distinction is invisible).
//
// The UNEQUAL-contributions case (fold-after-bet: the bettor's bet
// went uncalled because the opponent folded) was not anchored. CPU
// happened to use `rake_on_total_pot` which over-raked the uncalled
// portion — a real arithmetic bug that surfaced as the HU gate
// 0.09375 residual after Phase B Site (d) closure.
//
// Per the rake spec: uncalled bets are returned un-raked. Rake is
// applied to the MAIN POT (contested portion) only. The lone-survivor
// winner receives:
//   total_pot - main_pot_rake - traverser_investment
// where main_pot = min_contribution × num_contributors + starting_pot
// (the "called" portion that was contested).
//
// Hand-computed example (the lead's): HU, starting_pot=0, P0 bets 15,
// P1 has 5 then folds. contributions=[15, 5].
//   total_pot = 0 + 15 + 5 = 20
//   main_pot = min(15, 5) × 2 + 0 = 10 (contested portion: P1's 5
//     matched by 5 of P0's 15)
//   rake = min(10 × 0.05, 1000) = 0.5  (rake on main pot only)
//   uncalled excess (returned to P0 un-raked) = 15 - 5 = 10
//   traverser_investment = 0/2 + 15 = 15
//   payoff to P0 = (20 - 0.5) - 15 = 4.5
//     [= uncalled_returned(10) + main_pot_after_rake(9.5) - investment(15)
//      = 10 + 9.5 - 15 = 4.5 ✓ confirms uncalled excess returned un-raked]

#[test]
fn verify_rake_fold_win_after_bet_uncalled_returned_unraked() {
    let nh = 2usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8];
    let opp_reach_data = vec![1.0f32; nh];
    let opp_reach_slices: Vec<&[f32]> = vec![&opp_reach_data];
    let sorted_str = vec![100u16, 200u16];
    let sorted_idx = vec![0u16, 1u16];

    // P0 bets 15, P1 had 5 then folded. UNEQUAL contributions.
    // starting_pot = 0 (simplifies hand computation).
    let contributions = vec![15i32, 5i32];
    let fold_mask: u16 = 1u16 << 1; // P1 folded
    let starting_pot: i32 = 0;

    let rake_rate = 0.05_f32;
    let rake_cap = 1000.0_f32;

    // With rake_rate=0.05, rake_cap=1000:
    //   main_pot = 5 × 2 + 0 = 10
    //   main_pot_rake = 10 × 0.05 = 0.5
    //   total_pot = 20
    //   traverser_investment = 0/2 + 15 = 15
    //   payoff = (total_pot - main_pot_rake) - traverser_investment
    //          = (20 - 0.5) - 15 = 4.5
    //   CFV per hand = payoff × cfreach (cfreach=1 for non-conflict 2-hand case)
    //                = 4.5
    let cfv_with_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, starting_pot,
        rake_rate, rake_cap, true,
    );

    eprintln!("HU fold-win-after-bet (uncalled excess returned un-raked):");
    eprintln!("  Setup: starting_pot=0, contributions=[15,5], P1 folded, traverser=P0");
    eprintln!("  Hand-computed: main_pot=10, rake=0.5, uncalled=10 (returned),");
    eprintln!("                 payoff = (20-0.5) - 15 = 4.5");
    eprintln!("  Actual CFV: {:?}", cfv_with_rake);

    let expected = [4.5_f32, 4.5_f32];
    for h in 0..nh {
        assert!((cfv_with_rake[h] - expected[h]).abs() < 1e-4,
            "fold-win-after-bet CFV[{}] = {}, expected {} \
             (hand-computed: main_pot_only rake, uncalled returned un-raked)",
            h, cfv_with_rake[h], expected[h]);
    }

    // Sanity check: also verify with rake=0 the diff is 5 (the full uncalled-included
    // payoff minus rake-free payoff would be ... actually at rake=0 it should just
    // be the rake-free fold-win, payoff = 20 - 15 = 5).
    let cfv_no_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, starting_pot,
        0.0, 0.0, true,
    );
    let expected_no_rake = [5.0_f32, 5.0_f32];
    for h in 0..nh {
        assert!((cfv_no_rake[h] - expected_no_rake[h]).abs() < 1e-4,
            "rake=0 fold-win CFV[{}] = {}, expected {}",
            h, cfv_no_rake[h], expected_no_rake[h]);
    }
    eprintln!("✓ Fold-win-after-bet correctly applies main-pot-only rake; \
        uncalled bet returned un-raked per the rake spec");

    // OVER-RAKE DEMONSTRATION (documents the bug that was fixed):
    // The previous (buggy) total_pot rake would have given:
    //   rake_buggy = 20 × 0.05 = 1.0
    //   payoff_buggy = (20 - 1.0) - 15 = 4.0  ← over-raked by 0.5
    // The 0.5 per-terminal discrepancy is exactly what surfaced as
    // the HU gate 0.09375 residual after Phase B Site (d) closure.
    let buggy_value = 4.0_f32;
    assert!((cfv_with_rake[0] - buggy_value).abs() > 0.1,
        "Sanity: the buggy total_pot rake would give CFV={}; \
         actual CFV={} confirms the main-pot-only fix is in effect.",
        buggy_value, cfv_with_rake[0]);
}

// ─────────────────────────────────────────────────────────────────────
// Slice 1.3: sorted-sweep rake hand-computed anchor
// ─────────────────────────────────────────────────────────────────────
//
// Sorted-sweep path triggered by "all_active_equal && fold_mask == 0
// && num_active_opp == 1" (line ~640 in showdown.rs). HU showdown,
// both players in for the same amount, no folds.
//
// Case 1: pure-win-no-tie scenario. Player has higher strength for ALL
// nh combos, opp has lower strength for ALL. With ties=0:
//   sweep_net[h] = win_reach[h] (no loss component)
//   tie_reach[h] = 0
//   cfv[h] = half_pot × sweep_net[h] − rake × win_reach[h]
//          = (half_pot − rake) × sweep_net[h]   (since sweep_net = win_reach here)
//
// Case 2: pure-tie scenario. Player and opp have identical strengths
// at every combo. With wins=0, losses=0:
//   sweep_net[h] = 0
//   tie_reach[h] = some positive amount
//   cfv[h] = 0 − rake × (tie_reach/2) = −rake × tie_reach / 2
//   This validates that ties produce a NEGATIVE CFV (player pays half-rake
//   on ties — both players split the post-rake pot, losing half-rake each).

use solver_core::solver::showdown::sorted_sweep_with_rake_components;

// Realistic HU showdown: opp_str = pl_str (same hand-strength evaluation
// on shared board). Player and opp have the SAME nh combos with the SAME
// per-combo strengths. Player's win/tie/loss depends on which combo they
// hold vs which combo opp holds.
//
// Setup: 2 combos with different strengths, no card overlap between
// combos.
//   combo 0: cards (0, 1), strength = 100 (low)
//   combo 1: cards (2, 3), strength = 200 (high)
// Sorted ascending: opp_str = pl_str = [100, 200], opp_idx = pl_idx = [0, 1].
//
// For player holding combo 0 (low):
//   vs opp combo 0 (self-conflict — same cards, can't both hold): tie band
//      includes opp idx 0, BUT card-blocking removes it; the self-correction
//      adds reach[0] back. Net tie contribution from opp combo 0 = reach[0].
//      But the inclusion-exclusion math (after self-correction) gives:
//      tie_reach[0] = tie_sum − tie_minus[0] − tie_minus[1] + reach[0]
//                   = reach[0] − reach[0] − reach[0] + reach[0] = 0.
//   vs opp combo 1: opp_str=200 > pl_str=100 → loss. opp's cards (2,3)
//      don't conflict with h's (0,1). Loss contribution = reach[1] = 1.
//   So for h=0: win_reach=0, tie_reach=0, sweep_net = −1.
//
// For player holding combo 1 (high):
//   vs opp combo 0: opp_str=100 < pl_str=200 → win. opp's cards (0,1)
//      don't conflict with h's (2,3). Win contribution = reach[0] = 1.
//   vs opp combo 1 (self): tie band, after self-correction = 0.
//   So for h=1: win_reach=1, tie_reach=0, sweep_net = 1.

#[test]
fn verify_rake_sorted_sweep_components_hu_realistic() {
    let nh = 2usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8];
    let opp_reach_data = vec![1.0f32; nh];
    let opp_reach_slices: Vec<&[f32]> = vec![&opp_reach_data];
    let pl_str = vec![100u16, 200u16];
    let pl_idx = vec![0u16, 1u16];
    let opp_str = vec![100u16, 200u16]; // HU: same evaluation
    let opp_idx = vec![0u16, 1u16];

    let (sweep_net, win_reach, tie_reach) = sorted_sweep_with_rake_components(
        &opp_reach_slices, &hand_cards, nh, &opp_str, &opp_idx, &pl_str, &pl_idx,
    );
    eprintln!("HU-realistic sorted-sweep components:");
    eprintln!("  sweep_net = {:?}", sweep_net);
    eprintln!("  win_reach = {:?}", win_reach);
    eprintln!("  tie_reach = {:?}", tie_reach);

    // Hand-computed expected:
    //   h=0: win=0, tie=0 (self-corrected), sweep_net=-1
    //   h=1: win=1, tie=0 (self-corrected), sweep_net=+1
    assert!((win_reach[0] - 0.0).abs() < 1e-6, "win_reach[0] = {}, expected 0", win_reach[0]);
    assert!((tie_reach[0] - 0.0).abs() < 1e-6, "tie_reach[0] = {}, expected 0", tie_reach[0]);
    assert!((sweep_net[0] - (-1.0)).abs() < 1e-6, "sweep_net[0] = {}, expected -1", sweep_net[0]);
    assert!((win_reach[1] - 1.0).abs() < 1e-6, "win_reach[1] = {}, expected 1", win_reach[1]);
    assert!((tie_reach[1] - 0.0).abs() < 1e-6, "tie_reach[1] = {}, expected 0", tie_reach[1]);
    assert!((sweep_net[1] - 1.0).abs() < 1e-6, "sweep_net[1] = {}, expected 1", sweep_net[1]);

    eprintln!("✓ Sorted-sweep components match hand-computed HU values \
        (self-correction applied to tie band)");
}

#[test]
fn verify_rake_sorted_sweep_payoffs_hu_realistic() {
    // Same HU setup as above, now applying rake to the full showdown.
    // Contributions [50, 50], starting_pot = 0, total_pot = 100.
    // half_pot = 0/2 + 50 = 50. rake = min(100 × 0.05, 1000) = 5.
    //
    // Hand-computed CFV:
    //   h=0 (loser): half_pot × sweep_net[0] − rake × (win_reach[0] + tie_reach[0]/2)
    //              = 50 × (−1) − 5 × (0 + 0) = −50  (no rake — loser doesn't pay)
    //   h=1 (winner): 50 × 1 − 5 × (1 + 0) = 50 − 5 = 45
    let nh = 2usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8];
    let opp_reach_data = vec![1.0f32; nh];
    let opp_reach_slices: Vec<&[f32]> = vec![&opp_reach_data];
    let pl_str = vec![100u16, 200u16];
    let pl_idx = vec![0u16, 1u16];
    let opp_str = vec![100u16, 200u16];
    let opp_idx = vec![0u16, 1u16];
    let contributions = vec![50i32, 50i32];
    let fold_mask = 0u16;

    let cfv_no_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 0, 2, 0,
        0.0, 0.0, true,
    );
    let cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 0, 2, 0,
        0.05, 1000.0, true,
    );
    eprintln!("HU-realistic showdown:");
    eprintln!("  rake-free CFV: {:?}", cfv_no_rake);
    eprintln!("  rake-5%  CFV: {:?}", cfv);

    let expected_no_rake = [-50.0f32, 50.0f32];
    let expected_with_rake = [-50.0f32, 45.0f32];
    for h in 0..nh {
        assert!((cfv_no_rake[h] - expected_no_rake[h]).abs() < 1e-4,
            "rake-free CFV[{}] = {}, expected {}", h, cfv_no_rake[h], expected_no_rake[h]);
        assert!((cfv[h] - expected_with_rake[h]).abs() < 1e-4,
            "rake CFV[{}] = {}, expected {} (loser unchanged, winner -rake)",
            h, cfv[h], expected_with_rake[h]);
    }
    eprintln!("✓ Sorted-sweep HU rake: winner pays rake (-5), loser unchanged");
}

// ─────────────────────────────────────────────────────────────────────
// Slice 1.4: brute-force per-(h, g_0, g_1) rake anchor (3-player)
// ─────────────────────────────────────────────────────────────────────
//
// 3-player equal-contribution showdown path (num_active_opp == 2,
// all_active_equal, fold_mask == 0). Per-scenario enumeration over
// (g0, g1) opponent assignments with payoff in units of stake:
//   strict win: K
//   tie at top with T tied: (K+1 - T) / T
//   strict loss: -1
// With K = num_active_opp = 2, rake correction subtracts
// rake_per_unit_stake / T from winning/tying payoffs.
//
// Hand-computed for 3p, contributions [50, 50, 50], starting_pot = 0:
//   total_pot = 150, half_pot = stake = 50
//   rake = min(150 × 0.05, 1000) = 7.5
//   rake_per_unit_stake = 7.5 / 50 = 0.15
//
// Setup: nh = 3, combos A=(0,1) str=300, B=(2,3) str=200, C=(4,5) str=100.
// Cards distinct so any two non-A combos can coexist as opp hands.

#[test]
fn verify_rake_brute_force_3p_strict_win_hand_computed() {
    let nh = 3usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8];
    // Sorted ascending by strength: C(100), B(200), A(300)
    let pl_str = vec![100u16, 200u16, 300u16];
    let pl_idx = vec![2u16, 1u16, 0u16];
    let opp_str = pl_str.clone();
    let opp_idx = pl_idx.clone();
    let contributions = vec![50i32, 50i32, 50i32];
    let fold_mask = 0u16;
    let opp_reach = vec![vec![1.0f32; nh]; 2];
    let opp_reach_views: Vec<&[f32]> = opp_reach.iter().map(|v| v.as_slice()).collect();

    let cfv_no_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 0, 3, 0,
        0.0, 0.0, true,
    );
    let cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 0, 3, 0,
        0.05, 1000.0, true,
    );
    eprintln!("3p brute-force strict-win:");
    eprintln!("  rake-free CFV: {:?}", cfv_no_rake);
    eprintln!("  rake-5%  CFV: {:?}", cfv);

    // Hand-computed for h=0 (A, str=300, strict winner):
    //   Valid (g0, g1) pairs: (B, C), (C, B) — both strict wins
    //   payoff_unit = K - rake/stake = 2 - 0.15 = 1.85 each
    //   accum = 2 × 1.85 = 3.7. cfv[0] = 50 × 3.7 = 185.
    //   Rake-free: payoff_unit = 2. accum = 4. cfv[0] = 200.
    assert!((cfv_no_rake[0] - 200.0).abs() < 1e-3,
        "rake-free h=0: {} != 200 (= 2 valid (g0,g1) × payoff_unit=2 × half_pot=50)",
        cfv_no_rake[0]);
    assert!((cfv[0] - 185.0).abs() < 1e-3,
        "rake h=0: {} != 185 (= 2 × (2 - 0.15) × 50, with rake reducing each strict win)",
        cfv[0]);

    // For h=1 (B, str=200, strict loser to A in every valid (g0, g1)):
    //   Valid pairs: (A, C), (C, A) — both strict losses
    //   payoff_unit = -1 each (rake-invariant for losses)
    //   accum = -2. cfv[1] = 50 × -2 = -100, RAKE-FREE EQUALS RAKE.
    assert!((cfv_no_rake[1] - (-100.0)).abs() < 1e-3,
        "rake-free h=1: {} != -100", cfv_no_rake[1]);
    assert!((cfv[1] - (-100.0)).abs() < 1e-3,
        "rake h=1: {} != -100 (loser is rake-invariant)", cfv[1]);

    // h=2 (C, str=100, strict loser): same as h=1 → -100 both with and without rake.
    assert!((cfv_no_rake[2] - (-100.0)).abs() < 1e-3);
    assert!((cfv[2] - (-100.0)).abs() < 1e-3);

    eprintln!("✓ Brute-force 3p strict win: winner pays rake×K_scenarios; losers rake-invariant");
}

#[test]
fn verify_rake_brute_force_3p_tie_at_top_hand_computed() {
    // Tie scenario: player ties with one opp at top.
    // Strengths: A=200, B=200, C=100. Player holds A; opp0=B (tie), opp1=C (loss to top).
    let nh = 3usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8];
    let pl_str = vec![100u16, 200u16, 200u16];  // sorted asc: C, A, B (or B, A — tie)
    let pl_idx = vec![2u16, 0u16, 1u16];        // C at idx 2, A at idx 0, B at idx 1
    let opp_str = pl_str.clone();
    let opp_idx = pl_idx.clone();
    let contributions = vec![50i32, 50i32, 50i32];
    let fold_mask = 0u16;
    let opp_reach = vec![vec![1.0f32; nh]; 2];
    let opp_reach_views: Vec<&[f32]> = opp_reach.iter().map(|v| v.as_slice()).collect();

    let cfv_no_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 0, 3, 0,
        0.0, 0.0, true,
    );
    let cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 0, 3, 0,
        0.05, 1000.0, true,
    );
    eprintln!("3p brute-force tie-at-top:");
    eprintln!("  rake-free CFV: {:?}", cfv_no_rake);
    eprintln!("  rake-5%  CFV: {:?}", cfv);

    // h=0 (A, str=200): valid (g0, g1) ∈ {(B,C), (C,B)}.
    //   (B, C): max=200=h_str. T=2 (A, B tied). payoff_unit = (K+1-T)/T − rake/T
    //          = (2+1-2)/2 − 0.15/2 = 0.5 − 0.075 = 0.425
    //   (C, B): same. 0.425.
    //   accum = 0.85. cfv[0] = 50 × 0.85 = 42.5.
    //   Rake-free: 0.5 each → accum = 1.0 → cfv[0] = 50.
    assert!((cfv_no_rake[0] - 50.0).abs() < 1e-3,
        "rake-free h=0 tie: {} != 50 (= 2 × 0.5 × 50)", cfv_no_rake[0]);
    assert!((cfv[0] - 42.5).abs() < 1e-3,
        "rake h=0 tie: {} != 42.5 (= 2 × 0.425 × 50, T=2 reduces rake share by half)",
        cfv[0]);

    eprintln!("✓ Brute-force 3p tie-at-top T=2: rake shared 50/50 between winners");
}

// ─────────────────────────────────────────────────────────────────────
// Slice 1.5: side-pot rake (site convention: main pot only,
// single per-hand cap, no flop no drop)
// ─────────────────────────────────────────────────────────────────────
//
// Setup for 3p multiway all-in with side pot:
//   contributions = [50, 100, 100]  (p0 short-stack, p1+p2 deep)
//   starting_pot = 0
//   Levels: [50, 100]
//   Main pot (level 0): 50 × 3 = 150  (everyone contributes)
//   Side pot (level 1): 50 × 2 = 100  (only p1, p2 contest)
//   rake_rate = 0.05, rake_cap = 1000 → rake on main = 7.5 (cap not binding)
//   Side pot UN-RAKED.
//
// Combos: A=(0,1)=str200, B=(2,3)=str100, C=(4,5)=str50
//   Sorted asc: pl_str = [50, 100, 200], pl_idx = [2, 1, 0]

#[test]
fn verify_rake_side_pot_short_winner_main_pot_only() {
    // Traverser = p0 (short, c_t=50). Holding A (best). Strict win at main
    // pot (level 0), but NOT eligible for side pot (their contribution 50 <
    // level 100). Expected cash = (150 - 7.5)/1 = 142.5. net = 92.5.
    // 2 valid (g_a, g_b) scenarios. accum = 185.
    let nh = 3usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8];
    let pl_str = vec![50u16, 100u16, 200u16];
    let pl_idx = vec![2u16, 1u16, 0u16];
    let opp_str = pl_str.clone();
    let opp_idx = pl_idx.clone();
    let contributions = vec![50i32, 100i32, 100i32];
    let fold_mask = 0u16;
    let opp_reach = vec![vec![1.0f32; nh]; 2];
    let opp_reach_views: Vec<&[f32]> = opp_reach.iter().map(|v| v.as_slice()).collect();

    let cfv_no_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 0, 3, 0,
        0.0, 0.0, true,
    );
    let cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 0, 3, 0,
        0.05, 1000.0, true,
    );
    eprintln!("3p side-pot, trav=p0 short winner:");
    eprintln!("  rake-free CFV: {:?}", cfv_no_rake);
    eprintln!("  rake-5%  CFV: {:?}", cfv);

    assert!((cfv_no_rake[0] - 200.0).abs() < 1e-3,
        "rake-free cfv[0]={}, expected 200", cfv_no_rake[0]);
    assert!((cfv[0] - 185.0).abs() < 1e-3,
        "rake cfv[0]={}, expected 185 (= 200 − rake×2 scenarios = 200 − 15)",
        cfv[0]);
    // Differences should equal exactly rake × num_scenarios = 7.5 × 2 = 15
    assert!((cfv_no_rake[0] - cfv[0] - 15.0).abs() < 1e-3,
        "main-pot rake reduction should be 15, got {}",
        cfv_no_rake[0] - cfv[0]);

    eprintln!("✓ 3p side-pot: trav-as-short-winner only claims main pot, raked correctly");
}

#[test]
fn verify_rake_side_pot_deep_winner_main_raked_side_not() {
    // Traverser = p1 (deep, c_t=100). Holding A (best). Wins BOTH pots:
    //   main pot cash = (150 - 7.5)/1 = 142.5
    //   side pot cash = 100/1 = 100 (UN-RAKED — this is the key
    //                                discriminator for "main pot only" rule)
    //   total cash = 242.5. net = 142.5.
    //   Rake-free baseline: total cash = 250. net = 150.
    // Valid (g_a, g_b) scenarios (g_a = p0's hand, g_b = p2's hand,
    // both no conflict with h or each other):
    //   (B, C): p0=B, p2=C — valid
    //   (C, B): p0=C, p2=B — valid
    // 2 scenarios. accum = 2 × 142.5 = 285. cfv[0] = 285.
    // Rake-free: accum = 2 × 150 = 300.
    let nh = 3usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8];
    let pl_str = vec![50u16, 100u16, 200u16];
    let pl_idx = vec![2u16, 1u16, 0u16];
    let opp_str = pl_str.clone();
    let opp_idx = pl_idx.clone();
    let contributions = vec![50i32, 100i32, 100i32];
    let fold_mask = 0u16;
    let opp_reach = vec![vec![1.0f32; nh]; 2];
    let opp_reach_views: Vec<&[f32]> = opp_reach.iter().map(|v| v.as_slice()).collect();

    // traverser = 1
    let cfv_no_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 1, 3, 0,
        0.0, 0.0, true,
    );
    let cfv = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 1, 3, 0,
        0.05, 1000.0, true,
    );
    eprintln!("3p side-pot, trav=p1 deep winner:");
    eprintln!("  rake-free CFV: {:?}", cfv_no_rake);
    eprintln!("  rake-5%  CFV: {:?}", cfv);

    assert!((cfv_no_rake[0] - 300.0).abs() < 1e-3,
        "rake-free cfv[0]={}, expected 300 (main pot 150 + side pot 100, × 2 scen − stake 100 = 150 × 2 = 300)",
        cfv_no_rake[0]);
    assert!((cfv[0] - 285.0).abs() < 1e-3,
        "rake cfv[0]={}, expected 285 (main pot raked: (150−7.5)+100 − 100 = 142.5 × 2 = 285). \
         Side pot UN-raked. Critical: if both pots were raked, diff would be larger.",
        cfv[0]);

    // The DISCRIMINATING check: difference is exactly main-pot rake × scenarios
    // = 7.5 × 2 = 15. If side pot were also raked at 5%, the side-pot rake
    // would add 5 × 2 = 10 more, giving cfv = 275 instead of 285.
    let observed_rake_total = cfv_no_rake[0] - cfv[0];
    assert!((observed_rake_total - 15.0).abs() < 1e-3,
        "observed rake reduction = {}, expected 15 (main-only). \
         If side pot were also raked, this would be 25.",
        observed_rake_total);

    eprintln!("✓ 3p side-pot: deep winner gets BOTH pots, ONLY main pot raked");
}

#[test]
fn verify_rake_no_flop_no_drop_zeroes_rake() {
    // Same setup as "trav=short winner" test, but flop_seen=false.
    // Expected: cfv = rake-free baseline regardless of rake_rate.
    let nh = 3usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8, 4u8, 5u8];
    let pl_str = vec![50u16, 100u16, 200u16];
    let pl_idx = vec![2u16, 1u16, 0u16];
    let opp_str = pl_str.clone();
    let opp_idx = pl_idx.clone();
    let contributions = vec![50i32, 100i32, 100i32];
    let fold_mask = 0u16;
    let opp_reach = vec![vec![1.0f32; nh]; 2];
    let opp_reach_views: Vec<&[f32]> = opp_reach.iter().map(|v| v.as_slice()).collect();

    // flop_seen=true with rake → cfv[0] = 185
    let cfv_flop = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 0, 3, 0,
        0.05, 1000.0, true,
    );
    // flop_seen=false (PREFLOP TERMINAL) with same rake → cfv[0] = 200 (no rake)
    let cfv_preflop = side_pot_showdown_cfv_with_rake(
        &opp_reach_views, &hand_cards, nh,
        &opp_str, &opp_idx, &pl_str, &pl_idx,
        &contributions, fold_mask, 0, 3, 0,
        0.05, 1000.0, false,
    );
    eprintln!("no-flop-no-drop test:");
    eprintln!("  flop_seen=true  cfv: {:?}", cfv_flop);
    eprintln!("  flop_seen=false cfv: {:?}", cfv_preflop);

    assert!((cfv_flop[0] - 185.0).abs() < 1e-3, "flop_seen=true gives 185");
    assert!((cfv_preflop[0] - 200.0).abs() < 1e-3,
        "flop_seen=false should be rake-free (200), got {}",
        cfv_preflop[0]);
    // Across all hands, the flop_seen=false result must equal the rake-free
    // result (rake gate to zero on every path).
    for h in 0..nh {
        let rake_free = if h == 0 { 200.0 } else { -100.0 };
        assert!((cfv_preflop[h] - rake_free).abs() < 1e-3,
            "no-flop cfv[{}] = {}, expected rake-free {}", h, cfv_preflop[h], rake_free);
    }

    eprintln!("✓ no flop, no drop: flop_seen=false zeroes rake across all paths");
}

// ─────────────────────────────────────────────────────────────────────
// Slice 1.6: end-to-end — tree.rake_rate flows to SOLVED result
// ─────────────────────────────────────────────────────────────────────
//
// The requirement closure for rake. After Slice 1.6 threads rake from
// FlatTree through evaluate_terminal to side_pot_showdown_cfv_with_rake,
// `tree.rake_rate` and `tree.rake_cap` actually reach the solved-result
// CFV (not just storage). This test runs the flop-start solver iter-0
// with rake=0 and rake=0.05 on the same tree+game setup, and confirms
// the root CFV differs by an amount consistent with the rake.
//
// What this anchors:
//   - The wiring from TreeConfig → FlatTree (Slice 1.6 storage path)
//   - The wiring from FlatTree → FlopStartGame::evaluate_terminal (Slice 1.6
//     evaluate_terminal change)
//   - The wiring from evaluate_terminal → side_pot_showdown_cfv_with_rake
//     (the rake math from Slices 1.1-1.5)
//   - All composed end-to-end via a real solver run
//
// What this does NOT anchor (separate work):
//   - GPU rake (Slice 2.x)
//   - Ante / stack depth end-to-end (separate tests below)

fn build_flop_start_game_with_rake(rake_rate: f64, rake_cap: f64)
    -> (solver_core::tree::flat::FlatTree, FlopStartGame)
{
    use solver_core::card::NUM_POSSIBLE_HANDS;
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 10,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![0, 0],
        rake_rate,
        rake_cap,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&cfg).expect("tree builds");
    let board = vec![
        card_from_str("Ac").unwrap(),
        card_from_str("Kd").unwrap(),
        card_from_str("2h").unwrap(),
    ];
    let ranges = vec![vec![1.0f32; NUM_POSSIBLE_HANDS]; 2];
    let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    let game = FlopStartGame::new(table);
    (tree, game)
}

fn run_iter0_root_cfv(
    tree: &solver_core::tree::flat::FlatTree,
    game: &FlopStartGame,
    traverser: u8,
) -> Vec<f32> {
    let mut solver = FlopStartVectorCfr::new(tree, game.table());
    solver.set_vanilla_mode(true);
    solver.compute_all_strategies(tree);
    let reach = solver.compute_reach_flop(tree, game);
    let table_ref = game.table();
    let nh = solver.num_hands();
    let nn = tree.num_nodes();
    let mut cfv = vec![0.0f32; nn * nh];
    let params = DcfrParams::new(0);
    // Bottom-up walk: River → Turn → Flop
    let turn_deck = table_ref.remaining_deck.clone();
    for (ti, &tc_card) in turn_deck.iter().enumerate() {
        let river_deck = &table_ref.river_decks[tc_card as usize];
        for ri in 0..river_deck.len() {
            solver.bottom_up_zone(
                tree, table_ref, traverser, &reach, &mut cfv,
                Zone::River, Some(ti), Some(ri), &params,
            );
        }
    }
    for ti in 0..turn_deck.len() {
        solver.bottom_up_zone(
            tree, table_ref, traverser, &reach, &mut cfv,
            Zone::Turn, Some(ti), None, &params,
        );
    }
    solver.bottom_up_zone(
        tree, table_ref, traverser, &reach, &mut cfv,
        Zone::Flop, None, None, &params,
    );
    cfv[0..nh].to_vec()
}

#[test]
fn verify_rake_reaches_solver_root_cfv_end_to_end() {
    // SAME tree+game setup with two different rake configs. The only
    // difference: rake_rate / rake_cap. Run iter-0 root CFV. The two
    // results MUST differ (this confirms rake reaches the solver, not
    // just storage).
    let (tree_no_rake, game_no_rake) = build_flop_start_game_with_rake(0.0, 0.0);
    let (tree_rake, game_rake) = build_flop_start_game_with_rake(0.05, 30.0);

    // Storage round-trip first (already verified elsewhere, just confirm
    // here so the test failure mode is clear).
    assert_eq!(tree_no_rake.rake_rate, 0.0);
    assert_eq!(tree_rake.rake_rate, 0.05);

    let cfv_no_rake = run_iter0_root_cfv(&tree_no_rake, &game_no_rake, 0);
    let cfv_rake = run_iter0_root_cfv(&tree_rake, &game_rake, 0);

    let nh = cfv_no_rake.len();
    let mut max_abs_diff = 0.0f32;
    let mut argmax = 0usize;
    for h in 0..nh {
        let d = (cfv_no_rake[h] - cfv_rake[h]).abs();
        if d > max_abs_diff { max_abs_diff = d; argmax = h; }
    }
    eprintln!("\nend-to-end rake reach test (HU flop-start, iter-0):");
    eprintln!("  nh = {}", nh);
    eprintln!("  cfv_no_rake[{}] = {}", argmax, cfv_no_rake[argmax]);
    eprintln!("  cfv_rake[{}]    = {}", argmax, cfv_rake[argmax]);
    eprintln!("  max abs diff   = {}", max_abs_diff);

    // The two root CFVs MUST differ — rake reaches the solver output.
    // The exact magnitude depends on the tree's terminal pot distribution
    // and the solver's normalization (num_combinations), but any non-zero
    // diff at f32 precision confirms wiring.
    assert!(max_abs_diff > 1e-7,
        "rake does NOT reach solver output: cfv unchanged between rake=0 and rake=0.05. \
         The Slice 1.6 thread (evaluate_terminal → side_pot_showdown_cfv_with_rake) is \
         not wired correctly.");

    eprintln!("✓ tree.rake_rate flows end-to-end to solved-result CFV");
}

#[test]
fn verify_stack_depth_reaches_solver_root_cfv_end_to_end() {
    // SAME tree shape with different stack depths. Deep stack allows
    // larger pots at terminals (because betting can go further), so root
    // CFV magnitudes should differ.
    use solver_core::card::NUM_POSSIBLE_HANDS;
    fn build(stack: i32) -> (solver_core::tree::flat::FlatTree, FlopStartGame) {
        // 2×pot bet config + small starting pot. With short stack 20
        // (just bigger than the 2×pot bet of 20), the bet IS allin.
        // With deep stack 1000, the same 2×pot bet is small relative
        // to stack, allowing further action above it. Same shape as
        // verify_stack_depth_bounds_max_betting which confirmed short
        // vs deep produce different tree max-contributions.
        let cfg = TreeConfig {
            num_players: 2,
            initial_state: BoardState::Flop,
            starting_pot: 10,
            starting_stacks: vec![stack, stack],
            initial_contributions: vec![0, 0],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions {
                // Include raise options so the tree can grow deeper at
                // deep stack (each raise compounds, exposing more of the
                // stack via terminal contributions).
                bet: vec![BetSize::PotRelative(1.0)],
                raise: vec![BetSize::PotRelative(1.0), BetSize::AllIn],
            },
            add_allin_threshold: 0.0,
            force_allin_threshold: 0.5,
            merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,

        };
        let tree = build_tree(&cfg).expect("tree builds");
        let board = vec![
            card_from_str("Ac").unwrap(),
            card_from_str("Kd").unwrap(),
            card_from_str("2h").unwrap(),
        ];
        let ranges = vec![vec![1.0f32; NUM_POSSIBLE_HANDS]; 2];
        let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
        let game = FlopStartGame::new(table);
        (tree, game)
    }

    let (tree_short, game_short) = build(20);
    let (tree_deep, game_deep) = build(1000);

    eprintln!("\nstack-depth end-to-end test: short stack vs deep stack");
    eprintln!("  short tree: {} nodes", tree_short.num_nodes());
    eprintln!("  deep tree:  {} nodes", tree_deep.num_nodes());
    let short_max: i32 = (0..tree_short.num_nodes())
        .map(|i| (0..2).map(|p| tree_short.get_contribution(i, p)).max().unwrap_or(0))
        .max().unwrap_or(0);
    let deep_max: i32 = (0..tree_deep.num_nodes())
        .map(|i| (0..2).map(|p| tree_deep.get_contribution(i, p)).max().unwrap_or(0))
        .max().unwrap_or(0);
    eprintln!("  max terminal contribution: short={}, deep={}", short_max, deep_max);

    let cfv_short = run_iter0_root_cfv(&tree_short, &game_short, 0);
    let cfv_deep = run_iter0_root_cfv(&tree_deep, &game_deep, 0);

    let nh = cfv_short.len();
    let mut max_abs_diff = 0.0f32;
    let mut argmax = 0usize;
    for h in 0..nh {
        let d = (cfv_short[h] - cfv_deep[h]).abs();
        if d > max_abs_diff { max_abs_diff = d; argmax = h; }
    }
    eprintln!("  max abs diff = {} (h={})", max_abs_diff, argmax);
    eprintln!("  cfv_short[{}] = {}", argmax, cfv_short[argmax]);
    eprintln!("  cfv_deep[{}]  = {}", argmax, cfv_deep[argmax]);

    assert!(max_abs_diff > 1e-7,
        "stack depth does NOT reach solver output: cfv unchanged between 10 and 200 \
         starting_stack. Stack must affect max-bet → terminal contributions → payoffs. \
         Short tree: {} nodes, max contrib {}; deep tree: {} nodes, max contrib {}.",
        tree_short.num_nodes(), short_max,
        tree_deep.num_nodes(), deep_max);

    eprintln!("✓ tree.starting_stacks flows end-to-end to solved-result CFV");
}

#[test]
fn verify_ante_reaches_solver_root_cfv_end_to_end() {
    // SAME tree shape with different initial_contributions (ante effect).
    // Larger ante → larger initial pot → larger terminal payoffs.
    use solver_core::card::NUM_POSSIBLE_HANDS;
    fn build(ante: i32) -> (solver_core::tree::flat::FlatTree, FlopStartGame) {
        let cfg = TreeConfig {
            num_players: 2,
            initial_state: BoardState::Flop,
            starting_pot: 10,
            starting_stacks: vec![100, 100],
            initial_contributions: vec![ante, ante],
            rake_rate: 0.0, rake_cap: 0.0,
            bet_sizes: BetSizeOptions {
                bet: vec![BetSize::PotRelative(1.0)],
                raise: vec![],
            },
            add_allin_threshold: 1.0,
            force_allin_threshold: 1.0,
            merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,

        };
        let tree = build_tree(&cfg).expect("tree builds");
        let board = vec![
            card_from_str("Ac").unwrap(),
            card_from_str("Kd").unwrap(),
            card_from_str("2h").unwrap(),
        ];
        let ranges = vec![vec![1.0f32; NUM_POSSIBLE_HANDS]; 2];
        let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
        let game = FlopStartGame::new(table);
        (tree, game)
    }

    let (tree_a0, game_a0) = build(0);
    let (tree_a5, game_a5) = build(5);

    let cfv_a0 = run_iter0_root_cfv(&tree_a0, &game_a0, 0);
    let cfv_a5 = run_iter0_root_cfv(&tree_a5, &game_a5, 0);

    let nh = cfv_a0.len();
    let mut max_abs_diff = 0.0f32;
    for h in 0..nh {
        let d = (cfv_a0[h] - cfv_a5[h]).abs();
        if d > max_abs_diff { max_abs_diff = d; }
    }
    eprintln!("\nante end-to-end test: ante=0 vs ante=5");
    eprintln!("  max abs diff = {}", max_abs_diff);

    assert!(max_abs_diff > 1e-7,
        "ante (via initial_contributions) does NOT reach solver output. \
         initial_contributions should affect root contributions → terminal \
         contributions → payoffs.");

    eprintln!("✓ initial_contributions (ante) flows end-to-end to solved-result CFV");
}

#[test]
fn verify_rake_affects_terminal_payoffs() {
    // The originally-#[ignore]'d test from the gap-discovery commit, now
    // unblocked: rake parameter flows to payoffs. Uses fast path so this
    // test passes once Slice 1.1+1.2 lands. Sorted-sweep + side-pot paths
    // get their own anchors in Slice 1.3+1.4.
    let nh = 2usize;
    let hand_cards = vec![0u8, 1u8, 2u8, 3u8];
    let opp_reach_data = vec![1.0f32; nh];
    let opp_reach_slices: Vec<&[f32]> = vec![&opp_reach_data];
    let sorted_str = vec![100u16, 200u16];
    let sorted_idx = vec![0u16, 1u16];
    let contributions = vec![50i32, 50i32];
    let fold_mask: u16 = 1u16 << 1; // active winner, opp folded → fast path

    let cfv_no_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 0,
        0.0, 0.0, true,
    );
    let cfv_with_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 0,
        0.05, 1000.0, true,
    );
    assert!(
        cfv_no_rake.iter().zip(&cfv_with_rake).any(|(a, b)| (a - b).abs() > 1e-4),
        "rake_rate changing from 0 to 0.05 must change CFV. \
         rake-free: {:?}; rake-5%: {:?}",
        cfv_no_rake, cfv_with_rake,
    );
    eprintln!("✓ rake_rate is a live parameter affecting terminal payoffs");
}

// ─────────────────────────────────────────────────────────────────────
// Bonus: storage round-trip (already partially covered by tree_builder
// tests; included here for completeness as part of the pre-blueprint
// requirement audit)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn verify_storage_round_trip_for_all_four_parameters() {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![500, 500],
        initial_contributions: vec![10, 20], // unequal for ante/blind verification
        rake_rate: 0.045,
        rake_cap: 25.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&cfg).expect("tree builds");

    assert_eq!(tree.starting_stacks, vec![500, 500], "starting_stacks round-trip");
    assert_eq!(tree.rake_rate, 0.045, "rake_rate round-trip");
    assert_eq!(tree.rake_cap, 25.0, "rake_cap round-trip");
    assert_eq!(tree.get_contribution(0, 0), 10, "p0 ante/blind round-trip");
    assert_eq!(tree.get_contribution(0, 1), 20, "p1 ante/blind round-trip");
    assert_eq!(tree.starting_pot, 30, "starting_pot round-trip");

    eprintln!("✓ All four parameters round-trip through storage");
    eprintln!("  starting_stacks=[500,500], initial_contributions=[10,20]");
    eprintln!("  rake_rate=0.045, rake_cap=25.0, starting_pot=30");
    eprintln!("  Note: round-trip storage does NOT imply implementation. See");
    eprintln!("  verify_rake_affects_terminal_payoffs (#[ignore]) for the rake gap.");
}

// ─────────────────────────────────────────────────────────────────────
// Bonus: FlopChanceTable does not take rake either
// ─────────────────────────────────────────────────────────────────────
//
// Structural check on the flop-start path: FlopChanceTable's
// compute_flop_start signature takes (known_board, ranges, num_players)
// — no rake. This means rake cannot propagate from flop-start configs
// to per-flop subtree solves.

#[test]
fn verify_flop_chance_table_signature_lacks_rake() {
    // Structural test: we call compute_flop_start with the standard
    // signature. The test passes if the signature does NOT require
    // rake parameters (i.e., the current state). If/when rake is
    // implemented, the signature will gain rake params and this test
    // will fail to compile, surfacing the implementation change.
    //
    // This is a compile-time documentation of the current gap. Like a
    // structural assertion that the gap exists.
    let board = vec![
        card_from_str("Ac").unwrap(),
        card_from_str("Kd").unwrap(),
        card_from_str("2h").unwrap(),
    ];
    let ranges = vec![
        vec![1.0f32; solver_core::card::NUM_POSSIBLE_HANDS],
        vec![1.0f32; solver_core::card::NUM_POSSIBLE_HANDS],
    ];
    let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    let _ = table; // suppress unused-warning

    eprintln!("✓ FlopChanceTable::compute_flop_start signature does NOT take rake.");
    eprintln!("  (If rake gets implemented, this test compiles unchanged unless");
    eprintln!("   compute_flop_start's signature gains rake params, in which case");
    eprintln!("   this test will fail to compile and need updating.)");
}
