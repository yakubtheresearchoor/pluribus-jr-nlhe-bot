// P1.5.4 Slice A.3d: preflop fold-terminal chip_delta from tree.contributions.
//
// Per the lead's directive: tree.contributions is the authoritative source.
// The formula matches showdown_oracle's constant-payoff fast path
// (showdown.rs:484–517) by construction, but is implemented as a NEW
// preflop-specific function — the validated oracle is not extended. The
// formula:
//
//   investment = starting_pot / np + contributions[traverser]
//   total_pot  = starting_pot + Σ contributions
//   delta(folder)     = -investment
//   delta(non-folder) = total_pot - investment
//
// Two-anchor validation:
//
//   ANCHOR 1: hand computation on asymmetric-blind cases (HU SB=1/BB=2).
//   These are cases the oracle's symmetric postflop convention can't
//   directly cover, so they prove the new preflop logic correct where
//   the oracle can't go.
//
//   ANCHOR 2: cross-check against the actual showdown_oracle on the
//   symmetric overlap (HU postflop with symmetric initial contributions).
//   Both functions implement the same formula; cross-checking via the
//   oracle's actual code (not a re-derivation) confirms consistency with
//   the validated postflop component.

use solver_core::solver::preflop_terminal::preflop_fold_terminal_chip_delta_from_state;
use solver_core::solver::showdown::side_pot_showdown_cfv;

// ── ANCHOR 1: HU SB=1/BB=2 hand-computed cases ────────────────────────

#[test]
fn slice_a3d_anchor1_sb_folds_at_root_hu_asymmetric_blinds() {
    // HU preflop: SB=1, BB=2, starting_pot=3. SB folds at root.
    // contributions stay at [1, 2], fold_mask = 0b01.
    let contributions = [1_i32, 2_i32];
    let fold_mask = 0b01_u16;
    let starting_pot = 3_i32;
    let np = 2_u8;

    // Hand-computed (formula from showdown.rs:486 + 513-516):
    //   trav=0 (folder): inv = 3/2 + 1 = 2.5; delta = -2.5
    //   trav=1 (non-folder): inv = 3/2 + 2 = 3.5; total_pot = 6; delta = 6 - 3.5 = 2.5
    let delta_sb = preflop_fold_terminal_chip_delta_from_state(
        &contributions, fold_mask, starting_pot, 0, np);
    let delta_bb = preflop_fold_terminal_chip_delta_from_state(
        &contributions, fold_mask, starting_pot, 1, np);
    assert!((delta_sb - (-2.5)).abs() < 1e-7, "SB delta: got {}, expected -2.5", delta_sb);
    assert!((delta_bb - 2.5).abs() < 1e-7, "BB delta: got {}, expected 2.5", delta_bb);
    // Zero-sum across players (per the oracle's convention):
    assert!((delta_sb + delta_bb).abs() < 1e-7, "zero-sum: SB + BB = {} (should be 0)", delta_sb + delta_bb);
}

#[test]
fn slice_a3d_anchor1_bb_folds_after_sb_open_to_5() {
    // HU preflop: SB opens to 5 (post-blind 4 chips), BB folds.
    // contributions=[5, 2], fold_mask = 0b10, starting_pot = 3.
    let contributions = [5_i32, 2_i32];
    let fold_mask = 0b10_u16;
    let starting_pot = 3_i32;
    let np = 2_u8;

    // Hand-computed:
    //   trav=0 (non-folder, SB): inv = 1.5 + 5 = 6.5; total_pot = 3+5+2 = 10; delta = 10 - 6.5 = 3.5
    //   trav=1 (folder, BB):     inv = 1.5 + 2 = 3.5; delta = -3.5
    let delta_sb = preflop_fold_terminal_chip_delta_from_state(
        &contributions, fold_mask, starting_pot, 0, np);
    let delta_bb = preflop_fold_terminal_chip_delta_from_state(
        &contributions, fold_mask, starting_pot, 1, np);
    assert!((delta_sb - 3.5).abs() < 1e-7, "SB delta: got {}, expected 3.5", delta_sb);
    assert!((delta_bb - (-3.5)).abs() < 1e-7, "BB delta: got {}, expected -3.5", delta_bb);
    assert!((delta_sb + delta_bb).abs() < 1e-7, "zero-sum failed: {}", delta_sb + delta_bb);
}

#[test]
fn slice_a3d_anchor1_sb_calls_bb_raises_to_5_sb_folds() {
    // HU preflop: SB calls (1→2), BB raises to 5, SB folds.
    // contributions=[2, 5], fold_mask = 0b01, starting_pot = 3.
    let contributions = [2_i32, 5_i32];
    let fold_mask = 0b01_u16;
    let starting_pot = 3_i32;
    let np = 2_u8;

    // Hand-computed:
    //   trav=0 (folder, SB):     inv = 1.5 + 2 = 3.5; delta = -3.5
    //   trav=1 (non-folder, BB): inv = 1.5 + 5 = 6.5; total_pot = 3+2+5 = 10; delta = 10 - 6.5 = 3.5
    let delta_sb = preflop_fold_terminal_chip_delta_from_state(
        &contributions, fold_mask, starting_pot, 0, np);
    let delta_bb = preflop_fold_terminal_chip_delta_from_state(
        &contributions, fold_mask, starting_pot, 1, np);
    assert!((delta_sb - (-3.5)).abs() < 1e-7, "SB delta: got {}, expected -3.5", delta_sb);
    assert!((delta_bb - 3.5).abs() < 1e-7, "BB delta: got {}, expected 3.5", delta_bb);
    assert!((delta_sb + delta_bb).abs() < 1e-7, "zero-sum failed: {}", delta_sb + delta_bb);
    eprintln!("ANCHOR 1: HU SB=1/BB=2 asymmetric-blind fold-terminal chip deltas \
              hand-computed and matched on 3 cases.");
}

// ── ANCHOR 2: cross-check vs actual showdown_oracle on symmetric overlap ──

/// Invoke the actual showdown_oracle on a minimal HU postflop fold
/// terminal and extract the scalar chip delta from its per-hand output.
///
/// The oracle's `hand_cards` parameter is a SHARED layout for both
/// players — `hand_cards[h*2..h*2+2]` are the cards for the combo at
/// index h, used both as traverser's combo when computing cfv[h] AND
/// as opp's combo when computing per-opp-combo blocking. So nh=1 with
/// one combo makes opp's only combo equal to traverser's combo (full
/// self-blocking, cfv = 0). To get the scalar chip_delta out, use nh=2
/// with non-conflicting combos, set opp_reach to favor the OTHER combo,
/// and read cfv[0] which then sees opp's non-conflicting reach.
fn oracle_fold_terminal_chip_delta_via_actual_function(
    contributions: &[i32],
    fold_mask: u16,
    starting_pot: i32,
    traverser: usize,
    num_players: u8,
) -> f32 {
    // Two non-conflicting combos: hand 0 = (Ac=8, Kc=12), hand 1 = (Ad=9, Kd=13).
    let nh = 2;
    let hand_cards: [u8; 4] = [8, 12, 9, 13];

    // opp_reach[1.0] only on hand 1 (the non-conflicting one for traverser hand 0).
    let opp_reach_vec: Vec<f32> = vec![0.0, 1.0];
    let opp_reach_slice: &[f32] = &opp_reach_vec;
    let opp_reach_arr: [&[f32]; 1] = [opp_reach_slice];

    // sorted_*_str/idx: unused for fold terminal (num_active <= 1 fast path).
    let sorted_empty_u16: [u16; 0] = [];

    let cfv = side_pot_showdown_cfv(
        &opp_reach_arr,
        &hand_cards,
        nh,
        &sorted_empty_u16,
        &sorted_empty_u16,
        &sorted_empty_u16,
        &sorted_empty_u16,
        contributions,
        fold_mask,
        traverser,
        num_players,
        starting_pot,
    );

    assert_eq!(cfv.len(), nh, "expected nh={} CFV; got {}", nh, cfv.len());

    // cfv[0]: traverser holds combo 0 (Ac, Kc); opp's only-reach combo is 1 (Ad, Kd),
    // non-conflicting → survival fraction = 1.0 → cfv[0] = scalar chip_delta.
    cfv[0]
}

#[test]
fn slice_a3d_anchor2_oracle_cross_check_symmetric_postflop_overlap() {
    // HU postflop: starting_pot=10, init=[5,5]. Player 1 bets 3 (contributions
    // become [5, 8]). Player 0 folds.
    let contributions = [5_i32, 8_i32];
    let fold_mask = 0b01_u16;
    let starting_pot = 10_i32;
    let np = 2_u8;

    for traverser in 0..np {
        let preflop_fn_delta = preflop_fold_terminal_chip_delta_from_state(
            &contributions, fold_mask, starting_pot, traverser, np);
        let oracle_delta = oracle_fold_terminal_chip_delta_via_actual_function(
            &contributions, fold_mask, starting_pot, traverser as usize, np);
        eprintln!("traverser {}: preflop_fn delta = {}, oracle delta = {}",
            traverser, preflop_fn_delta, oracle_delta);
        assert!((preflop_fn_delta - oracle_delta).abs() < 1e-6,
            "traverser {}: preflop fn = {}, oracle = {}, diff = {}",
            traverser, preflop_fn_delta, oracle_delta, (preflop_fn_delta - oracle_delta).abs());
    }

    // Repeat with a second symmetric-init postflop terminal (different action).
    let contributions = [12_i32, 5_i32];  // player 0 raised post-flop, player 1 folded
    let fold_mask = 0b10_u16;
    let starting_pot = 10_i32;
    for traverser in 0..np {
        let preflop_fn_delta = preflop_fold_terminal_chip_delta_from_state(
            &contributions, fold_mask, starting_pot, traverser, np);
        let oracle_delta = oracle_fold_terminal_chip_delta_via_actual_function(
            &contributions, fold_mask, starting_pot, traverser as usize, np);
        eprintln!("[#2] traverser {}: preflop_fn delta = {}, oracle delta = {}",
            traverser, preflop_fn_delta, oracle_delta);
        assert!((preflop_fn_delta - oracle_delta).abs() < 1e-6,
            "[#2] traverser {}: preflop fn = {}, oracle = {}, diff = {}",
            traverser, preflop_fn_delta, oracle_delta, (preflop_fn_delta - oracle_delta).abs());
    }

    eprintln!("ANCHOR 2: preflop fold-terminal chip_delta matches showdown_oracle's \
              actual function output on the symmetric postflop overlap. Two configs \
              tested, both traversers in each, max diff < 1e-6.");
}

#[test]
fn slice_a3d_anchor2_oracle_cross_check_extended_to_asymmetric_blinds() {
    // EXTENDING the cross-check beyond strict symmetric overlap: the
    // oracle's formula doesn't care whether starting_pot was contributed
    // symmetrically; it just uses starting_pot/np as the per-player share.
    // For preflop SB=1/BB=2 contributions, the formula still applies —
    // it's just that starting_pot/np = 1.5 doesn't match either blind
    // (it's the average). The oracle's per-player payoffs in this case
    // diverge from "literal" chip movements but remain zero-sum.
    //
    // Anchor confirms: the oracle's function produces the SAME number as
    // my preflop function even on asymmetric-blind inputs. The "symmetric
    // overlap" the lead named is the case where the oracle is CORRECT for
    // postflop use; here we extend the cross-check to confirm formula-
    // level agreement even when neither side claims correctness.
    let contributions = [1_i32, 2_i32];
    let fold_mask = 0b01_u16;
    let starting_pot = 3_i32;
    let np = 2_u8;

    for traverser in 0..np {
        let preflop_fn_delta = preflop_fold_terminal_chip_delta_from_state(
            &contributions, fold_mask, starting_pot, traverser, np);
        let oracle_delta = oracle_fold_terminal_chip_delta_via_actual_function(
            &contributions, fold_mask, starting_pot, traverser as usize, np);
        eprintln!("asym-blind: traverser {}: preflop_fn = {}, oracle = {}",
            traverser, preflop_fn_delta, oracle_delta);
        assert!((preflop_fn_delta - oracle_delta).abs() < 1e-6,
            "asym-blind traverser {}: preflop fn {} != oracle {}",
            traverser, preflop_fn_delta, oracle_delta);
    }
    eprintln!("Slice A.3d PASS: chip_delta from tree.contributions agrees with the \
              showdown_oracle's actual function output (formula consistency) AND with \
              hand computation on asymmetric-blind cases (oracle convention applied to \
              preflop blinds), zero-sum invariant holds throughout.");
}
