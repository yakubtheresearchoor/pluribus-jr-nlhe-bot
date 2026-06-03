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
use solver_core::solver::flop_start_game::FlopChanceTable;
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
        0.05, 1000.0,
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
        0.05, 3.0, // cap = 3 < uncapped rake = 5
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
        0.0, 0.0,
    );
    let cfv_with_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 0,
        0.05, 1000.0,
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
        0.0, 0.0,
    );
    let cfv_with_rake = side_pot_showdown_cfv_with_rake(
        &opp_reach_slices, &hand_cards, nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &contributions, fold_mask, 0, 2, 0,
        0.05, 1000.0,
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
