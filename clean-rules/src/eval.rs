//! Clean-room hand evaluator. A 5-card hand scores into a single
//! ordered `HandRank`; 6 or 7 cards score as the best 5-card subset
//! (all subsets evaluated — C(7,5) = 21, simple over clever).
//!
//! Score layout (compared as one u32): `category * 13^5 + d4*13^4 +
//! d3*13^3 + d2*13^2 + d1*13 + d0`, where d4..d0 are the category's
//! tiebreak ranks, most significant first, padded with 0. Categories:
//! 8 straight flush, 7 quads, 6 full house, 5 flush, 4 straight,
//! 3 trips, 2 two pair, 1 one pair, 0 high card.

/// Totally ordered hand strength (higher wins).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct HandRank(pub u32);

impl HandRank {
    pub fn category(self) -> u32 {
        self.0 / 371_293 // 13^5
    }
}

pub fn rank_of(card: u8) -> u32 {
    (card / 4) as u32
}
pub fn suit_of(card: u8) -> u32 {
    (card % 4) as u32
}

/// Score exactly five cards.
pub fn score5(c: [u8; 5]) -> HandRank {
    let mut ranks: Vec<u32> = c.iter().map(|&x| rank_of(x)).collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a)); // descending
    let flush = c.iter().all(|&x| suit_of(x) == suit_of(c[0]));

    // Straight: five distinct descending-consecutive ranks, or the
    // wheel (A,5,4,3,2 — ace plays low, top card is the 5).
    let distinct = {
        let mut d = ranks.clone();
        d.dedup();
        d
    };
    let straight_top: Option<u32> = if distinct.len() == 5 {
        if distinct[0] - distinct[4] == 4 {
            Some(distinct[0])
        } else if distinct == [12, 3, 2, 1, 0] {
            Some(3) // wheel: 5-high (rank index 3 = the five)
        } else {
            None
        }
    } else {
        None
    };

    // Rank multiplicities: (count, rank), sorted by count desc then
    // rank desc — the kicker order every category reads.
    let mut counts: Vec<(u32, u32)> = Vec::new();
    for &r in &distinct {
        let n = ranks.iter().filter(|&&x| x == r).count() as u32;
        counts.push((n, r));
    }
    counts.sort_unstable_by(|a, b| b.cmp(a));

    let pack = |cat: u32, digits: &[u32]| -> HandRank {
        let mut v = cat;
        for i in 0..5 {
            v = v * 13 + digits.get(i).copied().unwrap_or(0);
        }
        HandRank(v)
    };

    match (flush, straight_top) {
        (true, Some(top)) => return pack(8, &[top]),
        (true, None) => return pack(5, &ranks),
        (false, Some(top)) => return pack(4, &[top]),
        _ => {}
    }
    match counts[0].0 {
        4 => pack(7, &[counts[0].1, counts[1].1]),
        3 if counts[1].0 == 2 => pack(6, &[counts[0].1, counts[1].1]),
        3 => pack(3, &[counts[0].1, counts[1].1, counts[2].1]),
        2 if counts[1].0 == 2 => pack(2, &[counts[0].1, counts[1].1, counts[2].1]),
        2 => pack(1, &[counts[0].1, counts[1].1, counts[2].1, counts[3].1]),
        _ => pack(0, &ranks),
    }
}

/// Best 5-card hand from 5, 6, or 7 cards.
pub fn best5(cards: &[u8]) -> HandRank {
    let n = cards.len();
    assert!((5..=7).contains(&n), "best5 takes 5..=7 cards");
    let mut best = HandRank(0);
    // Choose 5 of n by skipping (n-5) indices.
    match n {
        5 => best = score5([cards[0], cards[1], cards[2], cards[3], cards[4]]),
        6 => {
            for skip in 0..6 {
                let mut five = [0u8; 5];
                let mut k = 0;
                for (i, &c) in cards.iter().enumerate() {
                    if i != skip {
                        five[k] = c;
                        k += 1;
                    }
                }
                best = best.max(score5(five));
            }
        }
        7 => {
            for s1 in 0..7 {
                for s2 in (s1 + 1)..7 {
                    let mut five = [0u8; 5];
                    let mut k = 0;
                    for (i, &c) in cards.iter().enumerate() {
                        if i != s1 && i != s2 {
                            five[k] = c;
                            k += 1;
                        }
                    }
                    best = best.max(score5(five));
                }
            }
        }
        _ => unreachable!(),
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(rank: u32, suit: u32) -> u8 {
        (rank * 4 + suit) as u8
    }

    /// ANALYTIC ANCHOR: the closed-form 5-card category frequencies
    /// over all C(52,5) = 2,598,960 hands (published since the 19th
    /// century). Any evaluator bug that shifts a single boundary case
    /// moves at least one bucket.
    #[test]
    fn five_card_category_frequencies_exact() {
        let mut freq = [0u64; 9];
        for a in 0..52u8 {
            for b in (a + 1)..52 {
                for c in (b + 1)..52 {
                    for d in (c + 1)..52 {
                        for e in (d + 1)..52 {
                            freq[score5([a, b, c, d, e]).category() as usize] += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(freq[8], 40, "straight flushes");
        assert_eq!(freq[7], 624, "quads");
        assert_eq!(freq[6], 3_744, "full houses");
        assert_eq!(freq[5], 5_108, "flushes");
        assert_eq!(freq[4], 10_200, "straights");
        assert_eq!(freq[3], 54_912, "trips");
        assert_eq!(freq[2], 123_552, "two pair");
        assert_eq!(freq[1], 1_098_240, "one pair");
        assert_eq!(freq[0], 1_302_540, "high card");
    }

    /// Domain-knowledge orderings — always-on sanity invariants.
    #[test]
    fn ordering_invariants() {
        // Wheel is the LOWEST straight.
        let wheel = score5([card(12, 0), card(0, 1), card(1, 2), card(2, 3), card(3, 0)]);
        let six_high = score5([card(0, 0), card(1, 1), card(2, 2), card(3, 3), card(4, 0)]);
        let broadway = score5([card(8, 0), card(9, 1), card(10, 2), card(11, 3), card(12, 0)]);
        assert!(wheel.category() == 4 && six_high.category() == 4 && broadway.category() == 4);
        assert!(wheel < six_high && six_high < broadway);

        // AA never below 72o on any of a sample of boards (the cheap
        // always-on invariant: aces beat seven-deuce unimproved).
        // Board with no pair/straight/flush help for either: K9 4 two
        // tone away from both.
        let board = [card(11, 0), card(7, 1), card(2, 2), card(5, 3), card(10, 1)];
        let aa = {
            let mut v = vec![card(12, 2), card(12, 3)];
            v.extend_from_slice(&board);
            best5(&v)
        };
        let s72 = {
            let mut v = vec![card(5, 0), card(0, 0)];
            v.extend_from_slice(&board);
            best5(&v)
        };
        assert!(aa > s72);

        // Kicker matters: AK > AQ on A-high board.
        let b2 = [card(12, 0), card(7, 1), card(2, 2), card(4, 3), card(9, 1)];
        let ak = {
            let mut v = vec![card(12, 1), card(11, 2)];
            v.extend_from_slice(&b2);
            best5(&v)
        };
        let aq = {
            let mut v = vec![card(12, 2), card(10, 3)];
            v.extend_from_slice(&b2);
            best5(&v)
        };
        assert!(ak > aq);

        // Identical hands tie exactly (split pots depend on this).
        assert_eq!(ak, {
            let mut v = vec![card(12, 3), card(11, 0)];
            v.extend_from_slice(&b2);
            best5(&v)
        });
    }

    /// 7-card sanity: the board can play (all five board cards).
    #[test]
    fn board_plays() {
        // Board is a broadway straight; both hole cards are blanks.
        let cards = [
            card(0, 0),
            card(1, 1), // 2c 3d in hand
            card(8, 0),
            card(9, 1),
            card(10, 2),
            card(11, 3),
            card(12, 0),
        ];
        assert_eq!(best5(&cards).category(), 4);
    }
}
