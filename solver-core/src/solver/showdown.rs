use crate::card::{card_pair_to_index, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use crate::hand::eval::Hand;

pub struct ShowdownTable {
    pub hand_ranks: Vec<u16>,
    pub valid_hands: Vec<u16>,
    pub board: Vec<Card>,
}

impl ShowdownTable {
    pub fn compute(board: &[Card]) -> Self {
        let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
        let mut hand_ranks = vec![0u16; NUM_POSSIBLE_HANDS];
        let mut valid_hands = Vec::new();

        for idx in 0..NUM_POSSIBLE_HANDS {
            let (c1, c2) = index_to_card_pair(idx as usize);

            if board_set.contains(&c1) || board_set.contains(&c2) {
                hand_ranks[idx] = 0;
                continue;
            }

            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in board {
                hand = hand.add_card(bc as usize);
            }

            let rank = hand.evaluate();
            hand_ranks[idx] = rank;
            valid_hands.push(idx as u16);
        }

        ShowdownTable {
            hand_ranks,
            valid_hands,
            board: board.to_vec(),
        }
    }

    pub fn num_valid_hands(&self) -> usize {
        self.valid_hands.len()
    }

    pub fn hands_conflict(h1: usize, h2: usize) -> bool {
        if h1 == h2 {
            return true;
        }
        let (c1a, c1b) = index_to_card_pair(h1);
        let (c2a, c2b) = index_to_card_pair(h2);
        c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b
    }

    pub fn compute_blocker_mask(&self) -> Vec<u8> {
        let n = self.valid_hands.len();
        let mut mask = vec![0u8; n * n];
        for i in 0..n {
            for j in 0..n {
                if Self::hands_conflict(self.valid_hands[i] as usize, self.valid_hands[j] as usize) {
                    mask[i * n + j] = 1;
                }
            }
        }
        mask
    }

    pub fn compute_initial_weights(
        &self,
        ranges: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let n = self.valid_hands.len();
        let num_players = ranges.len();

        let mut weights = vec![vec![0.0f32; n]; num_players];

        for p in 0..num_players {
            for (i, &hand_idx) in self.valid_hands.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hand_idx as usize);
                let idx = card_pair_to_index(c1, c2);
                weights[p][i] = ranges[p][idx];
            }
        }

        weights
    }

    pub fn compute_hand_ranks_gpu(&self) -> Vec<u16> {
        let n = self.valid_hands.len();
        let mut ranks = vec![0u16; n];
        for (i, &hi) in self.valid_hands.iter().enumerate() {
            ranks[i] = self.hand_ranks[hi as usize];
        }
        ranks
    }

    pub fn compute_hand_cards_gpu(&self) -> Vec<u8> {
        let n = self.valid_hands.len();
        let mut cards = vec![0u8; n * 2];
        for (i, &hi) in self.valid_hands.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            cards[i * 2] = c1;
            cards[i * 2 + 1] = c2;
        }
        cards
    }

    pub fn compute_sorted_opp_arrays(&self, num_players: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let nh = self.valid_hands.len();
        let num_opp = num_players - 1;

        let mut strength_items: Vec<(u16, u16)> = (0..nh)
            .map(|h| {
                let rank = self.hand_ranks[self.valid_hands[h] as usize];
                (rank + 1, h as u16)
            })
            .collect();
        strength_items.sort_by_key(|&(s, _)| s);

        let mut sorted_strength = vec![0u16; num_opp * nh];
        let mut sorted_indices = vec![0u16; num_opp * nh];
        for oi in 0..num_opp {
            for h in 0..nh {
                sorted_strength[oi * nh + h] = strength_items[h].0;
                sorted_indices[oi * nh + h] = strength_items[h].1;
            }
        }

        let same_hand_idx = vec![u16::MAX; nh];

        (sorted_strength, sorted_indices, same_hand_idx)
    }

    pub fn compute_initial_weights_flat(&self, ranges: &[Vec<f32>]) -> Vec<f32> {
        let n = self.valid_hands.len();
        let np = ranges.len();
        let mut flat = Vec::with_capacity(np * n);
        for p in 0..np {
            for (i, &hand_idx) in self.valid_hands.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hand_idx as usize);
                let idx = card_pair_to_index(c1, c2);
                flat.push(ranges[p][idx]);
            }
        }
        flat
    }
}

/// O(NH) sorted-sweep showdown. Returns **unscaled** net win/loss reach-weighted
/// fraction per hand. Caller must multiply by the contribution amount (`c_t`)
/// to get the actual counterfactual value: `cfv[h] = c_t * result[h]`.
///
/// For each opponent, two passes (ascending wins, descending losses) compute
/// per-hand counterfactual reach via inclusion-exclusion card blocking.
/// Total complexity: O(num_opp * NH).
#[allow(clippy::too_many_arguments)]
pub fn sorted_sweep_showdown(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    nh: usize,
    opp_str: &[u16],
    opp_idx: &[u16],
    pl_str: &[u16],
    pl_idx: &[u16],
) -> Vec<f32> {
    let num_opp = opp_reach.len();
    let mut cfv = vec![0.0f32; nh];

    for oi in 0..num_opp {
        let reach = opp_reach[oi];
        let mut cfreach_sum = 0.0f32;
        let mut cfreach_minus = vec![0.0f32; 52];

        let mut i = 0;
        for si in 0..nh {
            let str_h = pl_str[si];
            let h = pl_idx[si] as usize;
            while i < nh && opp_str[oi * nh + i] < str_h {
                let ho = opp_idx[oi * nh + i] as usize;
                let r = reach[ho];
                if r != 0.0 {
                    cfreach_sum += r;
                    cfreach_minus[hand_cards[ho * 2] as usize] += r;
                    cfreach_minus[hand_cards[ho * 2 + 1] as usize] += r;
                }
                i += 1;
            }
            let cfreach = cfreach_sum
                - cfreach_minus[hand_cards[h * 2] as usize]
                - cfreach_minus[hand_cards[h * 2 + 1] as usize];
            cfv[h] += cfreach;
        }

        cfreach_sum = 0.0;
        for c in 0..52 { cfreach_minus[c] = 0.0; }

        i = nh;
        for si in (0..nh).rev() {
            let str_h = pl_str[si];
            let h = pl_idx[si] as usize;
            while i > 0 && opp_str[oi * nh + i - 1] > str_h {
                i -= 1;
                let ho = opp_idx[oi * nh + i] as usize;
                let r = reach[ho];
                if r != 0.0 {
                    cfreach_sum += r;
                    cfreach_minus[hand_cards[ho * 2] as usize] += r;
                    cfreach_minus[hand_cards[ho * 2 + 1] as usize] += r;
                }
            }
            let cfreach = cfreach_sum
                - cfreach_minus[hand_cards[h * 2] as usize]
                - cfreach_minus[hand_cards[h * 2 + 1] as usize];
            cfv[h] -= cfreach;
        }
    }
    cfv
}

/// N-player side pot terminal evaluation. Returns **fully scaled** counterfactual
/// values per hand (includes pot_at_level multiplication and `c_t` subtraction).
///
/// Handles all cases: single active player (fold payoff), equal contributions
/// (standard showdown), and unequal contributions (level-by-level side pot).
/// Folded players (identified via `fold_mask`) contribute dead money but are
/// never eligible to contest any pot level.
///
/// `contributions` should be `np` entries for the terminal node (not the full tree).
/// `opp_reach` has N-1 entries indexed by opponent index oi = (p < traverser) ? p : p-1.
#[allow(clippy::too_many_arguments)]
pub fn side_pot_showdown_cfv(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    nh: usize,
    sorted_opp_str: &[u16],
    sorted_opp_idx: &[u16],
    sorted_pl_str: &[u16],
    sorted_pl_idx: &[u16],
    contributions: &[i32],
    fold_mask: u16,
    traverser: usize,
    num_players: u8,
    starting_pot: i32,
) -> Vec<f32> {
    let num_opp = opp_reach.len();
    let np = num_players as usize;
    let c_t = contributions[traverser];
    let mut cfv = vec![0.0f32; nh];

    let mut num_active = 0usize;
    for p in 0..np {
        if fold_mask & (1u16 << p) == 0 {
            num_active += 1;
        }
    }

    if num_active <= 1 || fold_mask & (1u16 << traverser) != 0 {
        let total_pot: i32 = starting_pot + contributions.iter().sum::<i32>();
        let traverser_investment = starting_pot as f32 / np as f32 + c_t as f32;
        let payoff = if fold_mask & (1u16 << traverser) != 0 {
            -traverser_investment
        } else {
            total_pot as f32 - traverser_investment
        };

        let mut opp_reach_sum = 0.0f32;
        let mut opp_reach_minus = vec![0.0f32; 52];
        for oi in 0..num_opp {
            for ho in 0..nh {
                let r = opp_reach[oi][ho];
                if r != 0.0 {
                    opp_reach_sum += r;
                    opp_reach_minus[hand_cards[ho * 2] as usize] += r;
                    opp_reach_minus[hand_cards[ho * 2 + 1] as usize] += r;
                }
            }
        }
        if opp_reach_sum > 0.0 {
            for h in 0..nh {
                let cfreach = opp_reach_sum
                    - opp_reach_minus[hand_cards[h * 2] as usize]
                    - opp_reach_minus[hand_cards[h * 2 + 1] as usize];
                cfv[h] = payoff * cfreach;
            }
        }
        return cfv;
    }

    // Compute min contribution among active (non-folded) players.
    // For 2-player showdowns, the at-risk amount for BOTH players is
    // starting_pot/np + min_active_contrib, regardless of who bet more.
    // The excess contribution is returned to the larger contributor.
    let min_active_contrib: i32 = (0..np)
        .filter(|&p| fold_mask & (1u16 << p) == 0)
        .map(|p| contributions[p])
        .min()
        .unwrap_or(0);

    let mut all_active_equal = true;
    let mut ref_contrib: Option<i32> = None;
    for p in 0..np {
        if fold_mask & (1u16 << p) != 0 { continue; }
        let cp = contributions[p];
        if let Some(r) = ref_contrib {
            if cp != r { all_active_equal = false; break; }
        } else {
            ref_contrib = Some(cp);
        }
    }

    // For 2-player showdowns (including unequal contributions),
    // use the simple sweep formula with min_active_contrib.
    // This correctly handles side pots: each player's at-risk is
    // starting_pot/np + min_active_contrib, because the excess
    // is returned to the larger contributor.
    if np == 2 && fold_mask & (1u16 << traverser) == 0 {
        let num_active_opp = (0..np)
            .filter(|&p| p != traverser && fold_mask & (1u16 << p) == 0)
            .count();

        if num_active_opp == 0 {
            // No active opponent: traverser wins the pot
            let total_pot: i32 = starting_pot + contributions.iter().sum::<i32>();
            let traverser_investment = starting_pot as f32 / np as f32 + c_t as f32;
            let payoff = total_pot as f32 - traverser_investment;
            for h in 0..nh { cfv[h] = payoff; }
            return cfv;
        }

        let mut filtered_opp: Vec<Vec<f32>> = Vec::with_capacity(num_opp);
        for oi in 0..num_opp {
            let p = if oi < traverser { oi } else { oi + 1 };
            if fold_mask & (1u16 << p) != 0 {
                filtered_opp.push(vec![0.0f32; nh]);
            } else {
                filtered_opp.push(opp_reach[oi].to_vec());
            }
        }

        let filtered_views: Vec<&[f32]> = filtered_opp.iter().map(|v| v.as_slice()).collect();
        let sweep = sorted_sweep_showdown(
            &filtered_views, hand_cards, nh,
            sorted_opp_str, sorted_opp_idx,
            sorted_pl_str, sorted_pl_idx,
        );

        let half_pot = starting_pot as f32 / np as f32 + min_active_contrib as f32;
        for h in 0..nh {
            cfv[h] = half_pot * sweep[h];
        }
        return cfv;
    }

    if all_active_equal {
        let num_active_opp = (0..np)
            .filter(|&p| p != traverser && fold_mask & (1u16 << p) == 0)
            .count();

        if num_active_opp == 0 {
            let total_pot: i32 = starting_pot + contributions.iter().sum::<i32>();
            let traverser_investment = starting_pot as f32 / np as f32 + c_t as f32;
            let payoff = total_pot as f32 - traverser_investment;
            let mut opp_reach_sum = 0.0f32;
            let mut opp_reach_minus = vec![0.0f32; 52];
            for oi in 0..num_opp {
                for ho in 0..nh {
                    let r = opp_reach[oi][ho];
                    if r != 0.0 {
                        opp_reach_sum += r;
                        opp_reach_minus[hand_cards[ho * 2] as usize] += r;
                        opp_reach_minus[hand_cards[ho * 2 + 1] as usize] += r;
                    }
                }
            }
            if opp_reach_sum > 0.0 {
                for h in 0..nh {
                    let cfreach = opp_reach_sum
                        - opp_reach_minus[hand_cards[h * 2] as usize]
                        - opp_reach_minus[hand_cards[h * 2 + 1] as usize];
                    cfv[h] = payoff * cfreach;
                }
            }
            return cfv;
        }

        let mut filtered_opp: Vec<Vec<f32>> = Vec::with_capacity(num_opp);
        for oi in 0..num_opp {
            let p = if oi < traverser { oi } else { oi + 1 };
            if fold_mask & (1u16 << p) != 0 {
                filtered_opp.push(vec![0.0f32; nh]);
            } else {
                filtered_opp.push(opp_reach[oi].to_vec());
            }
        }

        if num_active_opp == 1 {
            let filtered_views: Vec<&[f32]> = filtered_opp.iter().map(|v| v.as_slice()).collect();
            let sweep = sorted_sweep_showdown(
                &filtered_views, hand_cards, nh,
                sorted_opp_str, sorted_opp_idx,
                sorted_pl_str, sorted_pl_idx,
            );
            for h in 0..nh {
                cfv[h] = (starting_pot as f32 / np as f32 + c_t as f32) * sweep[h];
            }
            return cfv;
        }

        // N>2 equal-contribution showdown: product-based formula
        // cfv[h] = half_pot * ((num_active_opp + 1) * prod_oi(W_oi(h)) - prod_oi(R_eff_oi(h)))
        // where half_pot = starting_pot/np + c_t (effective per-player pot share)
        // where W_oi(h) = cum weaker reach for opp oi (with card blocking)
        //       R_eff_oi(h) = total reach of non-conflicting hands for opp oi
        let mut cum_weaker: Vec<Vec<f32>> = Vec::with_capacity(num_opp);
        let mut eff_total_reach: Vec<Vec<f32>> = Vec::with_capacity(num_opp);

        for oi in 0..num_opp {
            let reach = &filtered_opp[oi];
            let mut cw = vec![0.0f32; nh];
            let mut cfreach_sum = 0.0f32;
            let mut cfreach_minus = vec![0.0f32; 52];
            let mut i = 0;

            for si in 0..nh {
                let str_h = sorted_pl_str[si];
                let h = sorted_pl_idx[si] as usize;
                while i < nh && sorted_opp_str[oi * nh + i] < str_h {
                    let ho = sorted_opp_idx[oi * nh + i] as usize;
                    let r = reach[ho];
                    if r != 0.0 {
                        cfreach_sum += r;
                        cfreach_minus[hand_cards[ho * 2] as usize] += r;
                        cfreach_minus[hand_cards[ho * 2 + 1] as usize] += r;
                    }
                    i += 1;
                }
                cw[h] = cfreach_sum
                    - cfreach_minus[hand_cards[h * 2] as usize]
                    - cfreach_minus[hand_cards[h * 2 + 1] as usize];
            }

            while i < nh {
                let ho = sorted_opp_idx[oi * nh + i] as usize;
                let r = reach[ho];
                if r != 0.0 {
                    cfreach_sum += r;
                    cfreach_minus[hand_cards[ho * 2] as usize] += r;
                    cfreach_minus[hand_cards[ho * 2 + 1] as usize] += r;
                }
                i += 1;
            }

            let mut eff = vec![0.0f32; nh];
            for h in 0..nh {
                eff[h] = cfreach_sum
                    - cfreach_minus[hand_cards[h * 2] as usize]
                    - cfreach_minus[hand_cards[h * 2 + 1] as usize]
                    + reach[h];
            }

            cum_weaker.push(cw);
            eff_total_reach.push(eff);
        }

        for h in 0..nh {
            let beats_all: f32 = cum_weaker.iter().map(|cw| cw[h]).product();
            let eff_product: f32 = eff_total_reach.iter().map(|er| er[h]).product();
            cfv[h] = (starting_pot as f32 / np as f32 + c_t as f32) * ((num_active_opp as f32 + 1.0) * beats_all - eff_product);
        }
        return cfv;
    }

    let active: Vec<usize> = (0..np)
        .filter(|&p| fold_mask & (1u16 << p) == 0)
        .collect();

    let mut levels: Vec<i32> = (0..np).map(|p| contributions[p]).collect();
    levels.sort();
    levels.dedup();

    let mut prev_level = 0i32;
    for (level_idx, &level) in levels.iter().enumerate() {
        let pot_contribution = level - prev_level;
        if pot_contribution == 0 { continue; }

        let eligible: Vec<usize> = active.iter()
            .copied()
            .filter(|&p| contributions[p] >= level)
            .collect();
        if eligible.is_empty() { continue; }

        let num_eligible = (0..np).filter(|&p| contributions[p] >= level).count() as i32;
        let mut pot_at_level = pot_contribution * num_eligible;
        // Add starting_pot to the first (lowest) level — the main pot shared by all active players
        if level_idx == 0 {
            pot_at_level += starting_pot;
        }
        let traverser_eligible = contributions[traverser] >= level;

        let eligible_opp: Vec<usize> = eligible.iter()
            .copied()
            .filter(|&p| p != traverser)
            .collect();

        if eligible_opp.is_empty() {
            if traverser_eligible {
                for h in 0..nh { cfv[h] += pot_at_level as f32; }
            }
            prev_level = level;
            continue;
        }

        if traverser_eligible {
            for &opp_p in &eligible_opp {
                let oi = if opp_p < traverser { opp_p } else { opp_p - 1 };
                let reach = opp_reach[oi];
                let o_str = &sorted_opp_str[oi * nh..(oi + 1) * nh];
                let o_idx = &sorted_opp_idx[oi * nh..(oi + 1) * nh];

                let mut cfreach_sum = 0.0f32;
                let mut cfreach_minus = vec![0.0f32; 52];

                let mut i = 0;
                for si in 0..nh {
                    let str_h = sorted_pl_str[si];
                    let h = sorted_pl_idx[si] as usize;
                    while i < nh && o_str[i] < str_h {
                        let ho = o_idx[i] as usize;
                        let r = reach[ho];
                        if r != 0.0 {
                            cfreach_sum += r;
                            cfreach_minus[hand_cards[ho * 2] as usize] += r;
                            cfreach_minus[hand_cards[ho * 2 + 1] as usize] += r;
                        }
                        i += 1;
                    }
                    let cfreach = cfreach_sum
                        - cfreach_minus[hand_cards[h * 2] as usize]
                        - cfreach_minus[hand_cards[h * 2 + 1] as usize];
                    cfv[h] += pot_at_level as f32 * cfreach;
                }

                cfreach_sum = 0.0;
                for c in 0..52 { cfreach_minus[c] = 0.0; }
                i = nh;
                for si in (0..nh).rev() {
                    let str_h = sorted_pl_str[si];
                    let h = sorted_pl_idx[si] as usize;
                    while i > 0 && o_str[i - 1] > str_h {
                        i -= 1;
                        let ho = o_idx[i] as usize;
                        let r = reach[ho];
                        if r != 0.0 {
                            cfreach_sum += r;
                            cfreach_minus[hand_cards[ho * 2] as usize] += r;
                            cfreach_minus[hand_cards[ho * 2 + 1] as usize] += r;
                        }
                    }
                    let cfreach = cfreach_sum
                        - cfreach_minus[hand_cards[h * 2] as usize]
                        - cfreach_minus[hand_cards[h * 2 + 1] as usize];
                    cfv[h] -= pot_at_level as f32 * cfreach;
                }
            }
        }

        prev_level = level;
    }

    for h in 0..nh { cfv[h] -= starting_pot as f32 / np as f32 + c_t as f32; }
    cfv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::card_from_str;

    #[test]
    fn test_showdown_table_basic() {
        let board: Vec<Card> = ["2c", "3d", "4h", "5s", "6c"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect();

        let table = ShowdownTable::compute(&board);

        assert!(table.valid_hands.len() > 0);
        assert!(table.valid_hands.len() < NUM_POSSIBLE_HANDS);

        let (c1, c2) = index_to_card_pair(table.valid_hands[0] as usize);
        assert!(!board.contains(&c1));
        assert!(!board.contains(&c2));

        let straight_hand_idx = {
            let mut best = 0u16;
            let mut best_rank = 0u16;
            for &hi in &table.valid_hands {
                let r = table.hand_ranks[hi as usize];
                if r > best_rank {
                    best_rank = r;
                    best = hi;
                }
            }
            best
        };

        let pair_hand_idx = {
            let mut found = 0u16;
            for &hi in &table.valid_hands {
                let (a, b) = index_to_card_pair(hi as usize);
                if a / 4 == b / 4 && a / 4 > 6 {
                    found = hi;
                    break;
                }
            }
            found
        };

        if pair_hand_idx > 0 {
            let straight_rank = table.hand_ranks[straight_hand_idx as usize];
            let pair_rank = table.hand_ranks[pair_hand_idx as usize];
            assert!(
                straight_rank > pair_rank,
                "best hand rank ({}) should beat high pair ({})",
                straight_rank,
                pair_rank
            );
        }
    }

    #[test]
    fn test_blocker_mask() {
        let board: Vec<Card> = ["2c", "3d", "4h", "5s", "6c"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect();

        let table = ShowdownTable::compute(&board);
        let mask = table.compute_blocker_mask();
        let n = table.valid_hands.len();

        for i in 0..n {
            assert_eq!(mask[i * n + i], 1, "hand should conflict with itself");
        }

        let h0 = table.valid_hands[0] as usize;
        let mut conflict_count = 0;
        for j in 0..n {
            if mask[0 * n + j] == 1 {
                conflict_count += 1;
            }
        }
        assert!(conflict_count > 1, "hand should conflict with some others");
    }
}
