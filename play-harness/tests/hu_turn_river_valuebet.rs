//! Regression for the HU turn/river value-betting defect: heads-up, first-to-act
//! on the turn/river with the nuts, the bot was CHECKING DOWN (it fell through to
//! decide_live6, which checks when to_call==0). The exact HU re-solve
//! (`decide_live2_resolve`) must instead VALUE-BET the nuts.

use play_harness::api::{decide_live2_resolve, DecideRequest};

/// label 1 = check; anything else with a positive amount over commit = a bet/raise.
fn bet_prob(resp: &play_harness::api::DecideResponse) -> f32 {
    resp.actions.iter().filter(|a| a.label != 1 && a.label != 0).map(|a| a.prob).sum()
}

#[test]
fn hu_turn_quads_value_bets_not_check_down() {
    // Quad aces on the turn. board = [As, Ah, 7c, 2d]; hero = [Ac, Ad].
    let mk = |hero_idx: u8| DecideRequest {
        opponent_stats: vec![],
        pool_river_bluff: None,
        eff_stack: None,
        deadline_ms: None,
        budget_ms: None,
        preflop_actions: vec![],
        seat_positions: vec![],
        board: vec![card("As"), card("Ah"), card("7c"), card("2d")],
        hero_cards: [card("Ac"), card("Ad")],
        live: 2, hero_idx,
        commit_entry: 20, pot_entry: 40,
        to_call: Some(0), // first to act, unbet
        seed: Some(7),
        ..Default::default()
    };
    // hero must be the acting player at the root; try both seats.
    let resp = (0..2u8).find_map(|hero_idx| decide_live2_resolve(&mk(hero_idx)))
        .expect("HU turn resolve should serve the first-to-act node for some seat");

    assert_eq!(resp.street, "turn");
    let bp = bet_prob(&resp);
    eprintln!("HU turn quad-aces first-act: {:?}", resp.actions.iter().map(|a| (a.label, (a.prob*100.0).round()/100.0)).collect::<Vec<_>>());
    // The hard defect was 100% check. The turn is CPU-budget-limited (~43 iters,
    // 208ms/iter nested solve) so the value bet under-converges to ~60% — the
    // full fix (≥0.9, like the river) needs the GPU turn solve. Gate: must bet
    // the majority (not check down the nuts).
    assert!(bp > 0.5, "quads on the turn should value-bet most of the time, got bet_prob={bp}");
}

#[test]
fn hu_river_quads_value_bets_not_check_down() {
    // Quad aces on the river. board = [As, Ah, 7c, 2d, 9s]; hero = [Ac, Ad].
    let mk = |hero_idx: u8| DecideRequest {
        opponent_stats: vec![],
        pool_river_bluff: None,
        eff_stack: None,
        deadline_ms: None,
        budget_ms: None,
        preflop_actions: vec![],
        seat_positions: vec![],
        board: vec![card("As"), card("Ah"), card("7c"), card("2d"), card("9s")],
        hero_cards: [card("Ac"), card("Ad")],
        live: 2, hero_idx,
        commit_entry: 20, pot_entry: 40,
        to_call: Some(0), seed: Some(7),
        ..Default::default()
    };
    let resp = (0..2u8).find_map(|hero_idx| decide_live2_resolve(&mk(hero_idx)))
        .expect("HU river resolve should serve the first-to-act node for some seat");
    assert_eq!(resp.street, "river");
    let bp = bet_prob(&resp);
    eprintln!("HU river quad-aces first-act: {:?}", resp.actions.iter().map(|a| (a.label, (a.prob*100.0).round()/100.0)).collect::<Vec<_>>());
    // River is cheap (single street) → runs the full iter ceiling → near-pure
    // value bet. Tight gate locks in the fix (was 100% check, then 57% under the
    // old 96-iter cap, now ~98% at 600).
    assert!(bp > 0.9, "quads on the river should value-bet ~always, got bet_prob={bp}");
}

fn card(s: &str) -> u8 {
    solver_core::card::card_from_str(s).unwrap() as u8
}
