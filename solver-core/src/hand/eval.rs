use crate::hand::table::HAND_TABLE;

#[derive(Clone, Copy, Default)]
pub struct Hand {
    cards: [usize; 7],
    num_cards: usize,
}

#[inline]
fn keep_n_msb(mut x: i32, n: i32) -> i32 {
    let mut ret = 0;
    for _ in 0..n {
        let bit = 1 << (x.leading_zeros() ^ 31);
        x ^= bit;
        ret |= bit;
    }
    ret
}

#[inline]
fn find_straight(rankset: i32) -> i32 {
    const WHEEL: i32 = 0b1_0000_0000_1111;
    let is_straight = rankset & (rankset << 1) & (rankset << 2) & (rankset << 3) & (rankset << 4);
    if is_straight != 0 {
        keep_n_msb(is_straight, 1)
    } else if (rankset & WHEEL) == WHEEL {
        1 << 3
    } else {
        0
    }
}

impl Hand {
    #[inline]
    pub fn new() -> Hand {
        Hand::default()
    }

    #[inline]
    pub fn add_card(&self, card: usize) -> Hand {
        let mut hand = *self;
        hand.cards[hand.num_cards] = card;
        hand.num_cards += 1;
        hand
    }

    #[inline]
    pub fn contains(&self, card: usize) -> bool {
        self.cards[0..self.num_cards].contains(&card)
    }

    #[inline]
    pub fn evaluate(&self) -> u16 {
        HAND_TABLE.binary_search(&self.evaluate_internal()).unwrap() as u16
    }

    /// Order-preserving u16 index valid for 5-, 6-, AND 7-card hands.
    ///
    /// `evaluate_internal` already collapses any hand to its best-5
    /// value, so the full 5-card value space (7462 classes, built once
    /// at first use by enumerating all C(52,5) hands) indexes every
    /// case; `HAND_TABLE` is the 7-card-REACHABLE subset only and has
    /// gaps for 5/6-card hands (e.g. low high-card hands that cannot
    /// survive best-5-of-7).
    ///
    /// HISTORY (2026-06-11, found by the play-harness fire alarm on
    /// its first run): every FlopChanceTable rank table was built with
    /// `evaluate_internal() as u16`, truncating the category bits
    /// (<<26) and pair/trip sets (<<13) — hands were ordered
    /// essentially by kicker bit-set. Invisible to every solver-vs-
    /// solver gate because all arms consumed the same wrong tables;
    /// caught by the independent clean-room evaluator on the first
    /// cross-check.
    pub fn evaluate_full(&self) -> u16 {
        use std::sync::OnceLock;
        static FULL_TABLE: OnceLock<Vec<i32>> = OnceLock::new();
        let table = FULL_TABLE.get_or_init(|| {
            let mut vals = Vec::with_capacity(2_598_960);
            for a in 0..52usize {
                for b in (a + 1)..52 {
                    for c in (b + 1)..52 {
                        for d in (c + 1)..52 {
                            for e in (d + 1)..52 {
                                let h = Hand::new()
                                    .add_card(a)
                                    .add_card(b)
                                    .add_card(c)
                                    .add_card(d)
                                    .add_card(e);
                                vals.push(h.evaluate_internal());
                            }
                        }
                    }
                }
            }
            vals.sort_unstable();
            vals.dedup();
            vals
        });
        table
            .binary_search(&self.evaluate_internal())
            .expect("hand value outside the 5-card space (fewer than 5 cards?)") as u16
    }

    pub fn evaluate_internal(&self) -> i32 {
        let mut rankset = 0i32;
        let mut rankset_suit = [0i32; 4];
        let mut rankset_of_count = [0i32; 5];
        let mut rank_count = [0i32; 13];

        for &card in &self.cards[..self.num_cards] {
            let rank = card / 4;
            let suit = card % 4;
            rankset |= 1 << rank;
            rankset_suit[suit] |= 1 << rank;
            rank_count[rank] += 1;
        }

        for rank in 0..13 {
            rankset_of_count[rank_count[rank] as usize] |= 1 << rank;
        }

        let mut flush_suit: i32 = -1;
        for suit in 0..4 {
            if rankset_suit[suit as usize].count_ones() >= 5 {
                flush_suit = suit;
            }
        }

        let is_straight = find_straight(rankset);

        if flush_suit >= 0 {
            let is_straight_flush = find_straight(rankset_suit[flush_suit as usize]);
            if is_straight_flush != 0 {
                (8 << 26) | is_straight_flush
            } else {
                (5 << 26) | keep_n_msb(rankset_suit[flush_suit as usize], 5)
            }
        } else if rankset_of_count[4] != 0 {
            let remaining = keep_n_msb(rankset ^ rankset_of_count[4], 1);
            (7 << 26) | (rankset_of_count[4] << 13) | remaining
        } else if rankset_of_count[3].count_ones() == 2 {
            let trips = keep_n_msb(rankset_of_count[3], 1);
            let pair = rankset_of_count[3] ^ trips;
            (6 << 26) | (trips << 13) | pair
        } else if rankset_of_count[3] != 0 && rankset_of_count[2] != 0 {
            let pair = keep_n_msb(rankset_of_count[2], 1);
            (6 << 26) | (rankset_of_count[3] << 13) | pair
        } else if is_straight != 0 {
            (4 << 26) | is_straight
        } else if rankset_of_count[3] != 0 {
            let remaining = keep_n_msb(rankset_of_count[1], 2);
            (3 << 26) | (rankset_of_count[3] << 13) | remaining
        } else if rankset_of_count[2].count_ones() >= 2 {
            let pairs = keep_n_msb(rankset_of_count[2], 2);
            let remaining = keep_n_msb(rankset ^ pairs, 1);
            (2 << 26) | (pairs << 13) | remaining
        } else if rankset_of_count[2] != 0 {
            let remaining = keep_n_msb(rankset_of_count[1], 3);
            (1 << 26) | (rankset_of_count[2] << 13) | remaining
        } else {
            keep_n_msb(rankset, 5)
        }
    }
}
