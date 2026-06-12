//! H2 — THE FIRE ALARM: the solver's showdown ordering and the
//! clean-room evaluator score the same river showdowns; any
//! disagreement fails loudly with the exact cards. This breaks the
//! inter-implementation-agreement blind spot: a systematic evaluation
//! bug shared by all six self-play bots is invisible to win rates
//! (money still conserves), but it cannot be shared with an
//! independent implementation of the rules.
//!
//! Surface compared: for sampled flops × runouts, the ORDER (win /
//! tie / lose) of every sampled hand pair under (a) the solver's
//! river rank tables (the same machinery its terminal CFVs are built
//! from) and (b) clean-rules best-5-of-7. Ordering with ties is
//! exactly the relation the terminal payoffs consume.

use clean_rules::eval::best5;
use solver_core::card::{card_from_str, Card};
use solver_core::solver::flop_start_game::FlopChanceTable;

fn crosscheck_flop(flop: [Card; 3], rng: &mut u64) -> usize {
    let mut next = |r: &mut u64| {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *r >> 33
    };
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    // Two seeded turns × two rivers each.
    let tc1 = deck[(next(rng) % deck.len() as u64) as usize];
    let tc2 = {
        let mut c = deck[(next(rng) % deck.len() as u64) as usize];
        while c == tc1 {
            c = deck[(next(rng) % deck.len() as u64) as usize];
        }
        c
    };
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &[tc1, tc2] {
        let pool: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        let r1 = pool[(next(rng) % pool.len() as u64) as usize];
        let mut r2 = pool[(next(rng) % pool.len() as u64) as usize];
        while r2 == r1 {
            r2 = pool[(next(rng) % pool.len() as u64) as usize];
        }
        river_decks[tc as usize] = vec![r1, r2];
    }
    let table = FlopChanceTable::build_full_nh_sampled(flop, 6, &[tc1, tc2], &river_decks);
    let nh = table.num_valid;

    let mut checked = 0usize;
    for &tc in &[tc1, tc2] {
        for &rc in &river_decks[tc as usize].clone() {
            // Solver-side strength per hand at this runout.
            let (s_str, s_idx, _, _) = table.river_sorted_arrays(tc, rc);
            let mut strength = vec![0u16; nh];
            for k in 0..nh {
                strength[s_idx[k] as usize] = s_str[k];
            }
            let conflicts = |h: usize| -> bool {
                let c1 = table.hand_cards[h * 2];
                let c2 = table.hand_cards[h * 2 + 1];
                c1 == tc || c2 == tc || c1 == rc || c2 == rc
            };
            // 2000 sampled pairs per runout.
            for _ in 0..2000 {
                let a = (next(rng) % nh as u64) as usize;
                let b = (next(rng) % nh as u64) as usize;
                if a == b || conflicts(a) || conflicts(b) {
                    continue;
                }
                // Hands must not share a card with each other.
                let (a1, a2) = (table.hand_cards[a * 2], table.hand_cards[a * 2 + 1]);
                let (b1, b2) = (table.hand_cards[b * 2], table.hand_cards[b * 2 + 1]);
                if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 {
                    continue;
                }
                let solver_ord = strength[a].cmp(&strength[b]);
                let ra = best5(&[a1, a2, flop[0], flop[1], flop[2], tc, rc]);
                let rb = best5(&[b1, b2, flop[0], flop[1], flop[2], tc, rc]);
                let rules_ord = ra.cmp(&rb);
                assert_eq!(
                    solver_ord, rules_ord,
                    "FIRE ALARM: showdown disagreement at flop {flop:?} turn {tc} river {rc}: \
                     hand A ({a1},{a2}) vs hand B ({b1},{b2}) — solver {:?}/{:?}, \
                     clean-rules {:?}/{:?}",
                    strength[a], strength[b], ra, rb
                );
                checked += 1;
            }
        }
    }
    checked
}

#[test]
fn fire_alarm_showdown_ordering_agreement() {
    let flops: [[&str; 3]; 4] = [
        ["Th", "9d", "8c"], // wet connected
        ["2h", "7d", "Ks"], // dry rainbow
        ["Ah", "Kh", "Qh"], // monotone broadway (flushes + straights)
        ["6c", "6d", "2s"], // paired (boats/quads boundaries)
    ];
    let mut rng: u64 = 0xF1EE_A1A4;
    let mut total = 0usize;
    for f in flops {
        let flop: Vec<Card> = f.iter().map(|s| card_from_str(s).unwrap()).collect();
        total += crosscheck_flop([flop[0], flop[1], flop[2]], &mut rng);
    }
    eprintln!("fire alarm: {total} cross-implementation showdown pairs agree (4 flops × 4 runouts)");
    assert!(total > 10_000, "too few pairs actually checked: {total}");
}
