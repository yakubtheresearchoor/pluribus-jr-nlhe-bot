//! PRODUCTION GAME CLASS v1 — behavioral gate through the independent
//! rules engine (numeric bb pins live in solver-core `bb_units_gate`).
//!
//! The spec (pinned with user 2026-06-12): 6-max, no ante, blinds
//! 0.5bb/1bb (1/2 units), 100bb stacks (200 units), 5% rake capped at
//! 10bb (20 units), NO FLOP NO DROP.
//!
//! Three hand-computable settlements through clean-rules:
//!   A. fold-around          → rake 0 (no flop, no drop)
//!   B. raised pot, showdown → rake = floor(pot × 5%), percentage region
//!   C. stacks all-in        → rake = cap (the cap binds exactly at the
//!      all-in pot: cap/rate = 400 units = 2 stacks)
//!
//! The TableConfig is DERIVED from `production_game_v1()` — numbers are
//! never restated here.

use clean_rules::table::{Action, HandState, TableConfig};
use solver_core::tree::action::production_game_v1;

fn v1_table() -> TableConfig {
    let g = production_game_v1();
    TableConfig {
        num_players: g.num_players as usize,
        sb: g.sb as u32,
        bb: g.bb as u32,
        stacks: vec![g.stack as u32; g.num_players as usize],
        rake_milli: (g.rake_rate * 1000.0).round() as u32,
        rake_cap: g.rake_cap as u32,
    }
}

/// Deck with pinned (position, card) entries; remaining cards fill the
/// other positions in ascending order. Card encoding: 4·rank + suit.
fn deck_with(pins: &[(usize, u8)]) -> Vec<u8> {
    let mut deck = vec![u8::MAX; 52];
    for &(i, c) in pins {
        deck[i] = c;
    }
    let used: Vec<u8> = pins.iter().map(|&(_, c)| c).collect();
    let mut rest = (0..52u8).filter(|c| !used.contains(c));
    for slot in deck.iter_mut() {
        if *slot == u8::MAX {
            *slot = rest.next().unwrap();
        }
    }
    deck
}

/// Button 5 → sb = seat 0, bb = seat 1, utg = seat 2.
/// Deal order: deck[s] / deck[6+s] to seat s. Board after 12 holes:
/// burn 12, flop 13-15, burn 16, turn 17, burn 18, river 19.
/// BB gets AhAs, UTG 7c2d, board 3c 8d 9h Jc 4d (AA wins, no flush /
/// straight / board-pair complications).
fn pinned_deck() -> Vec<u8> {
    deck_with(&[
        (1, 50),  // bb: Ah
        (7, 51),  // bb: As
        (2, 20),  // utg: 7c
        (8, 1),   // utg: 2d
        (13, 4),  // flop: 3c
        (14, 25), // flop: 8d
        (15, 30), // flop: 9h
        (17, 36), // turn: Jc
        (19, 9),  // river: 4d
    ])
}

/// A: fold-around. Preflop only → NO FLOP NO DROP: rake must be 0
/// even though the rake spec is live. BB wins exactly the SB.
#[test]
fn v1_fold_around_unraked() {
    let mut h = HandState::new(v1_table(), 5, pinned_deck());
    for seat in [2, 3, 4, 5, 0] {
        h.apply(seat, Action::Fold).unwrap();
    }
    assert!(h.is_over());
    let s = h.settle();
    assert_eq!(s.rake, 0, "no flop, no drop");
    assert_eq!(s.net, vec![-1, 1, 0, 0, 0, 0], "BB wins exactly the SB, unraked");
}

/// B: UTG raises to 40 (20bb), BB calls, flop seen, checked down.
/// Pot = 40 + 40 + 1 = 81 → rake = floor(81 × 5%) = 4 (percentage
/// region, far from the cap). BB (AA) nets 81 − 40 − 4 = +37.
#[test]
fn v1_percentage_rake_by_hand() {
    let mut h = HandState::new(v1_table(), 5, pinned_deck());
    h.apply(2, Action::RaiseTo(40)).unwrap();
    for seat in [3, 4, 5, 0] {
        h.apply(seat, Action::Fold).unwrap();
    }
    h.apply(1, Action::Call).unwrap();
    for _street in 0..3 {
        h.apply(1, Action::Check).unwrap();
        h.apply(2, Action::Check).unwrap();
    }
    assert!(h.is_over());
    let s = h.settle();
    assert_eq!(s.rake, 4, "floor(81 × 0.05) = 4 units = 2 bb");
    assert_eq!(s.net, vec![-1, 37, -40, 0, 0, 0]);
    assert_eq!(s.net.iter().sum::<i64>(), -4, "Σnet = −rake");
}

/// C: stacks in — UTG jams 200, BB calls. Pot = 401 → 5% = 20.05,
/// floored AND capped at 20 (the cap binds exactly at the all-in pot).
/// BB (AA) nets 401 − 200 − 20 = +181.
#[test]
fn v1_cap_binds_at_allin() {
    let mut h = HandState::new(v1_table(), 5, pinned_deck());
    h.apply(2, Action::RaiseTo(200)).unwrap();
    for seat in [3, 4, 5, 0] {
        h.apply(seat, Action::Fold).unwrap();
    }
    h.apply(1, Action::Call).unwrap();
    assert!(h.is_over(), "all-in call runs the hand out");
    let s = h.settle();
    assert_eq!(s.rake, 20, "rake cap 10 bb binds at the all-in pot");
    assert_eq!(s.net, vec![-1, 181, -200, 0, 0, 0]);
    assert_eq!(s.net.iter().sum::<i64>(), -20, "Σnet = −rake");
}
