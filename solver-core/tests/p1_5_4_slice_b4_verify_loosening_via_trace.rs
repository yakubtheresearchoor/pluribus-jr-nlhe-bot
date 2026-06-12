// P1.5.4 Slice B.4: verify the third seam-test loosening via trace.
//
// Per the lead (2026-06-04): the third loosening (allow all-in players
// unequal contributions) was asserted from plausibility, not from data.
// The discipline: trace [102, 101] and confirm SB has zero chips
// remaining (genuine all-in, legal uncalled return) vs SB having chips
// left (the round wrongly completed at the all-in boundary, residual
// bug hidden by the loosening).
//
// This test traces node 907 from the HU-with-raise tree (the only
// remaining violation pre-loosening, classified as "both all-in") and
// asserts the all-in condition holds: contributions[p] ==
// max_committable[p] for any p whose contribution isn't the standing
// bet.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

#[test]
fn b4_pre_verify_102_101_is_genuinely_both_all_in() {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![2, 1],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("HU preflop with raise builds");

    // Find every preflop→flop chance node with non-equal contributions
    // among non-folded players. Then for each, check whether the
    // non-matching players are at max_committable (legal all-in) or
    // still have chips remaining (bug — residual wrong-game).
    let mut parent_of = vec![None::<u32>; tree.num_nodes()];
    for p in 0..tree.num_nodes() {
        for &c in tree.node_children(p) {
            parent_of[c as usize] = Some(p as u32);
        }
    }

    let np = tree.num_players as usize;
    let starting_stacks = vec![100_i32; np];
    let initial_contributions: Vec<i32> = (0..np)
        .map(|p| tree.get_contribution(0, p as u8))
        .collect();
    let max_committable: Vec<i32> = (0..np)
        .map(|p| starting_stacks[p] + initial_contributions[p])
        .collect();

    eprintln!("max_committable per player: {:?} (player 0 = BB, player 1 = SB)",
        max_committable);

    let mut unequal_seam_nodes: Vec<(usize, Vec<i32>, u16)> = Vec::new();
    for idx in 0..tree.num_nodes() {
        let n = &tree.nodes[idx];
        if !n.is_chance() { continue; }
        let par = match parent_of[idx] { Some(p) => p as usize, None => continue };
        if tree.nodes[par].board_state != BoardState::Preflop as u8 { continue; }
        let mask = tree.get_folded_mask(idx);
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let active: Vec<usize> = (0..np).filter(|&p| (mask & (1 << p)) == 0).collect();
        if active.len() < 2 { continue; }
        let first = contribs[active[0]];
        if !active.iter().all(|&p| contribs[p] == first) {
            unequal_seam_nodes.push((idx, contribs, mask));
        }
    }

    eprintln!("\nPreflop→flop chance nodes with unequal contributions among active players: {}",
        unequal_seam_nodes.len());

    let mut all_legit = true;
    for (idx, contribs, mask) in &unequal_seam_nodes {
        let active: Vec<usize> = (0..np).filter(|&p| (mask & (1 << p)) == 0).collect();
        let standing_bet = active.iter().map(|&p| contribs[p]).max().unwrap();
        eprintln!("\n── Node {} ──", idx);
        eprintln!("  contributions: {:?}", contribs);
        eprintln!("  fold_mask:     0b{:0b}", mask);
        eprintln!("  active:        {:?}", active);
        eprintln!("  standing bet:  {}", standing_bet);
        for &p in &active {
            let c = contribs[p];
            let m = max_committable[p];
            let remaining = m - c;
            let matches_standing = c == standing_bet;
            let at_max = c >= m;  // genuinely all-in (used >= for safety)
            let label = if matches_standing { "matches standing bet" }
                        else if at_max { "ALL-IN at max_committable (chips remaining = 0)" }
                        else { "✗ NOT MATCHING and NOT ALL-IN — chips remaining > 0 — RESIDUAL BUG" };
            eprintln!("  player {}: contribution={}, max_committable={}, remaining={} → {}",
                p, c, m, remaining, label);
            if !matches_standing && !at_max {
                all_legit = false;
            }
        }
    }

    if all_legit {
        eprintln!("\n✓ ALL unequal-seam cases are legitimate (matched standing bet or all-in).");
        eprintln!("  The third seam-test loosening (allow all-in players unequal contributions)");
        eprintln!("  is verified by the trace: every player whose contribution differs from the");
        eprintln!("  standing bet has zero chips remaining (genuine all-in). The uncalled-bet");
        eprintln!("  return rule applies; the chips are correct per poker rules.");
    } else {
        eprintln!("\n✗ FOUND CASES that are NOT matched-standing AND NOT all-in.");
        eprintln!("  The third seam-test loosening was hiding a residual bug. The fix is");
        eprintln!("  NOT complete — there is wrong-game logic at the all-in boundary that");
        eprintln!("  needs additional investigation.");
    }
    assert!(all_legit, "third loosening hides residual bug — see trace above");
}
