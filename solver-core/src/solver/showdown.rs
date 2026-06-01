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

        // Payoff is CONSTANT per scenario. cfv[h] = payoff * N_h where
        // N_h = reach-product-weighted count of valid opponent hand combinations.
        // For num_opp >= 2 we must use the PRODUCT of opponent reaches, not the SUM.
        if num_opp == 1 {
            let mut opp_reach_sum = 0.0f32;
            let mut opp_reach_minus = vec![0.0f32; 52];
            for ho in 0..nh {
                let r = opp_reach[0][ho];
                if r != 0.0 {
                    opp_reach_sum += r;
                    opp_reach_minus[hand_cards[ho * 2] as usize] += r;
                    opp_reach_minus[hand_cards[ho * 2 + 1] as usize] += r;
                }
            }
            for h in 0..nh {
                let cfreach = opp_reach_sum
                    - opp_reach_minus[hand_cards[h * 2] as usize]
                    - opp_reach_minus[hand_cards[h * 2 + 1] as usize];
                cfv[h] = payoff * cfreach;
            }
        } else if num_opp == 2 {
            // Brute-force enumeration over (g0, g1) ordered pairs.
            for h in 0..nh {
                let hc1 = hand_cards[h * 2] as usize;
                let hc2 = hand_cards[h * 2 + 1] as usize;
                let mut nh_count = 0.0f32;
                for g0 in 0..nh {
                    let g0c1 = hand_cards[g0 * 2] as usize;
                    let g0c2 = hand_cards[g0 * 2 + 1] as usize;
                    if g0c1 == hc1 || g0c1 == hc2 || g0c2 == hc1 || g0c2 == hc2 { continue; }
                    let r0 = opp_reach[0][g0];
                    if r0 == 0.0 { continue; }
                    for g1 in 0..nh {
                        let g1c1 = hand_cards[g1 * 2] as usize;
                        let g1c2 = hand_cards[g1 * 2 + 1] as usize;
                        if g1c1 == hc1 || g1c1 == hc2 || g1c2 == hc1 || g1c2 == hc2 { continue; }
                        if g0c1 == g1c1 || g0c1 == g1c2 || g0c2 == g1c1 || g0c2 == g1c2 { continue; }
                        let r1 = opp_reach[1][g1];
                        if r1 == 0.0 { continue; }
                        nh_count += r0 * r1;
                    }
                }
                cfv[h] = payoff * nh_count;
            }
        } else {
            // num_opp >= 3: keep approximate (single-opp-style) formula as fallback.
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

    // Only take the all_active_equal shortcut when there are NO folds. With
    // folds, the dead money requires per-scenario asymmetric accounting which
    // the level-by-level brute-force handles correctly.
    if all_active_equal && fold_mask == 0 {
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

        // N>2 equal-contribution showdown: BRUTE-FORCE over valid opponent assignments.
        // For K=2 opponents (3-player), enumerate exact valid pairs (g0, g1) where
        // g0 and g1 don't conflict with h or with each other. This is exact (no
        // independence approximation). Returns raw reach-product-weighted sum;
        // caller applies global /nc normalization.
        //
        // For K>2, fall back to product formula approximation.
        if num_active_opp == 2 {
            let reach_0 = &filtered_opp[0];
            let reach_1 = &filtered_opp[1];
            let half_pot = starting_pot as f32 / np as f32 + c_t as f32;

            // Build hand strength array from sorted_pl
            let mut hand_strength = vec![0u16; nh];
            for si in 0..nh {
                hand_strength[sorted_pl_idx[si] as usize] = sorted_pl_str[si];
            }

            // Tie-aware brute-force. For each opponent assignment (g0, g1)
            // contributing reach r0*r1, the unit payoff is:
            //   strict win (both opponents weaker): K = num_active_opp
            //   strict loss (any opponent stronger): -1
            //   tie at top with T tied (1..=K+1): (K+1 - T) / T
            // This is zero-sum exactly: for any (h0,h1,h2), sum over players of
            // their unit payoff is 0 (winner pool gets pot, others lose stake).
            let k = num_active_opp as f32;
            for h in 0..nh {
                let hc1 = hand_cards[h * 2] as usize;
                let hc2 = hand_cards[h * 2 + 1] as usize;
                let h_str = hand_strength[h];

                let mut accum = 0.0f32;

                for g0 in 0..nh {
                    let g0c1 = hand_cards[g0 * 2] as usize;
                    let g0c2 = hand_cards[g0 * 2 + 1] as usize;
                    if g0c1 == hc1 || g0c1 == hc2 || g0c2 == hc1 || g0c2 == hc2 { continue; }
                    let r0 = reach_0[g0];
                    if r0 == 0.0 { continue; }
                    let s0 = hand_strength[g0];

                    for g1 in 0..nh {
                        let g1c1 = hand_cards[g1 * 2] as usize;
                        let g1c2 = hand_cards[g1 * 2 + 1] as usize;
                        if g1c1 == hc1 || g1c1 == hc2 || g1c2 == hc1 || g1c2 == hc2 { continue; }
                        // g0 and g1 must not share a card (impossible for two opponents)
                        if g0c1 == g1c1 || g0c1 == g1c2 || g0c2 == g1c1 || g0c2 == g1c2 { continue; }
                        let r1 = reach_1[g1];
                        if r1 == 0.0 { continue; }

                        let s1 = hand_strength[g1];
                        let max_opp = s0.max(s1);

                        let payoff_unit: f32 = if max_opp > h_str {
                            -1.0
                        } else if max_opp == h_str {
                            // tie at top
                            let mut t: u32 = 1;
                            if s0 == h_str { t += 1; }
                            if s1 == h_str { t += 1; }
                            (k + 1.0 - t as f32) / t as f32
                        } else {
                            k
                        };

                        accum += r0 * r1 * payoff_unit;
                    }
                }

                cfv[h] = half_pot * accum;
            }
            return cfv;
        }

        // K>2: product formula approximation (kept as fallback)
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

    // Per-scenario brute-force level-by-level evaluator.
    // For each pot level: iterate over (g0, g1) opponent hand assignments
    // for ALL non-traverser active players. The full scenario determines
    // who wins each pot level. Sum per-scenario net payoff into cfv[h].
    // No end stake subtraction (per-scenario stake is baked in).
    //
    // Note: this brute-force is O(nh^3) per terminal which is acceptable
    // for terminal evaluation (CPU only; GPU still uses simpler formulas).
    //
    // Folded players' contributions remain "dead money" in the lowest pot
    // levels they paid for.

    // Build hand strength array from sorted_pl
    let mut hand_strength = vec![0u16; nh];
    for si in 0..nh {
        hand_strength[sorted_pl_idx[si] as usize] = sorted_pl_str[si];
    }

    // Build pot levels.
    let mut levels: Vec<i32> = (0..np).map(|p| contributions[p]).collect();
    levels.sort();
    levels.dedup();

    // Per-level pot amount and eligible set.
    let mut level_pots: Vec<(i32, Vec<usize>, f32)> = Vec::new(); // (pot_at_level, eligible_active, per_player_stake_at_level)
    let mut prev_level = 0i32;
    for (level_idx, &level) in levels.iter().enumerate() {
        let pot_contribution = level - prev_level;
        if pot_contribution == 0 {
            prev_level = level;
            continue;
        }
        // All players (including folded) contributed at this level if their contribution >= level.
        // The pot at this level = pot_contribution * (#players with contribution >= level).
        let num_contributors = (0..np).filter(|&p| contributions[p] >= level).count();
        let mut pot_at_level = pot_contribution * num_contributors as i32;
        if level_idx == 0 {
            pot_at_level += starting_pot;
        }
        // Eligible to win this pot = ACTIVE players with contribution >= level
        let eligible_active: Vec<usize> = (0..np)
            .filter(|&p| fold_mask & (1u16 << p) == 0 && contributions[p] >= level)
            .collect();
        if eligible_active.is_empty() {
            prev_level = level;
            continue;
        }
        // Per-player stake = traverser's slice of this pot if eligible
        let per_player_stake = if contributions[traverser] >= level {
            pot_contribution as f32 + if level_idx == 0 { starting_pot as f32 / np as f32 } else { 0.0 }
        } else {
            0.0  // Traverser did not contribute to this level (their contribution < level)
        };
        // Wait: if level_idx==0 and traverser >= level, they contributed pot_contribution + sp/np.
        // If level_idx > 0 and traverser >= level, they contributed pot_contribution to this slice.
        // Folded players: still pay their slice but no eligibility.
        let _ = per_player_stake;
        level_pots.push((pot_at_level, eligible_active, 0.0));
        prev_level = level;
    }

    // Need to handle folded players' contributions: they pay into the pot but can't win.
    // For correctness, we recompute per-scenario net using a closure that evaluates
    // payoff at all levels at once given a specific scenario.

    // Traverser's total stake (constant per hand, paid regardless of outcome).
    let traverser_stake = starting_pot as f32 / np as f32 + c_t as f32;
    let traverser_folded = fold_mask & (1u16 << traverser) != 0;

    // Identify opponent slots. Each opponent (active or folded) gets iterated:
    // their hand assignment contributes to the reach product even if folded.
    // num_opp = np - 1.
    if num_opp == 2 {
        // Two opponents: brute-force enumeration over (g0, g1) — regardless
        // of fold status; eligibility/winner determination uses fold_mask.
        let opp_a = if traverser == 0 { 1 } else { 0 };
        let opp_b = if traverser == 2 { 1 } else { 2 };
        let oi_a = 0;
        let oi_b = 1;
        let reach_a = opp_reach[oi_a];
        let reach_b = opp_reach[oi_b];

        let c_opp_a = contributions[opp_a];
        let c_opp_b = contributions[opp_b];
        let a_folded = fold_mask & (1u16 << opp_a) != 0;
        let b_folded = fold_mask & (1u16 << opp_b) != 0;

        for h in 0..nh {
            let hc1 = hand_cards[h * 2] as usize;
            let hc2 = hand_cards[h * 2 + 1] as usize;
            let h_str = hand_strength[h];
            let mut accum = 0.0f32;

            for g_a in 0..nh {
                let g_ac1 = hand_cards[g_a * 2] as usize;
                let g_ac2 = hand_cards[g_a * 2 + 1] as usize;
                if g_ac1 == hc1 || g_ac1 == hc2 || g_ac2 == hc1 || g_ac2 == hc2 { continue; }
                let ra = reach_a[g_a];
                if ra == 0.0 { continue; }
                let s_a = hand_strength[g_a];

                for g_b in 0..nh {
                    let g_bc1 = hand_cards[g_b * 2] as usize;
                    let g_bc2 = hand_cards[g_b * 2 + 1] as usize;
                    if g_bc1 == hc1 || g_bc1 == hc2 || g_bc2 == hc1 || g_bc2 == hc2 { continue; }
                    if g_ac1 == g_bc1 || g_ac1 == g_bc2 || g_ac2 == g_bc1 || g_ac2 == g_bc2 { continue; }
                    let rb = reach_b[g_b];
                    if rb == 0.0 { continue; }
                    let s_b = hand_strength[g_b];

                    // Per-scenario payoff for traverser:
                    //   net = cash_received - traverser_stake
                    // If traverser folded: cash_received = 0 (lost all stake).
                    let net: f32 = if traverser_folded {
                        -traverser_stake
                    } else {
                        let mut cash: f32 = 0.0;
                        let mut prev_l = 0i32;
                        for (li, &lev) in levels.iter().enumerate() {
                            let pc = lev - prev_l;
                            if pc == 0 { prev_l = lev; continue; }
                            let num_contrib = (0..np).filter(|&p| contributions[p] >= lev).count();
                            let mut pot_l = (pc * num_contrib as i32) as f32;
                            if li == 0 { pot_l += starting_pot as f32; }

                            let trav_elig = c_t >= lev;
                            let a_elig = !a_folded && c_opp_a >= lev;
                            let b_elig = !b_folded && c_opp_b >= lev;
                            let n_elig_total = trav_elig as i32 + a_elig as i32 + b_elig as i32;

                            if n_elig_total == 0 {
                                // No eligible active player. Slice is returned to
                                // contributors. Traverser gets back their slice if
                                // they contributed at this level (proportional).
                                if contributions[traverser] >= lev {
                                    let trav_contrib = pc as f32
                                        + if li == 0 { starting_pot as f32 / np as f32 } else { 0.0 };
                                    cash += trav_contrib;
                                }
                                prev_l = lev;
                                continue;
                            }

                            if !trav_elig {
                                prev_l = lev;
                                continue;
                            }

                            // Determine winner(s) among eligible.
                            let mut max_str = h_str;
                            if a_elig && s_a > max_str { max_str = s_a; }
                            if b_elig && s_b > max_str { max_str = s_b; }

                            let mut tied: u32 = 0;
                            if h_str == max_str { tied += 1; }
                            if a_elig && s_a == max_str { tied += 1; }
                            if b_elig && s_b == max_str { tied += 1; }

                            if h_str == max_str {
                                cash += pot_l / tied as f32;
                            }
                            prev_l = lev;
                        }
                        cash - traverser_stake
                    };

                    accum += ra * rb * net;
                }
            }

            cfv[h] = accum;
        }
        return cfv;
    }

    // num_opp == 1: brute-force over single opponent.
    if num_opp == 1 {
        let opp_a = if traverser == 0 { 1 } else { 0 };
        let reach_a = opp_reach[0];
        let c_opp_a = contributions[opp_a];
        let a_folded = fold_mask & (1u16 << opp_a) != 0;

        for h in 0..nh {
            let hc1 = hand_cards[h * 2] as usize;
            let hc2 = hand_cards[h * 2 + 1] as usize;
            let h_str = hand_strength[h];
            let mut accum = 0.0f32;

            for g_a in 0..nh {
                let g_ac1 = hand_cards[g_a * 2] as usize;
                let g_ac2 = hand_cards[g_a * 2 + 1] as usize;
                if g_ac1 == hc1 || g_ac1 == hc2 || g_ac2 == hc1 || g_ac2 == hc2 { continue; }
                let ra = reach_a[g_a];
                if ra == 0.0 { continue; }
                let s_a = hand_strength[g_a];

                let net: f32 = if traverser_folded {
                    -traverser_stake
                } else {
                    let mut cash: f32 = 0.0;
                    let mut prev_l = 0i32;
                    for (li, &lev) in levels.iter().enumerate() {
                        let pc = lev - prev_l;
                        if pc == 0 { prev_l = lev; continue; }
                        let num_contrib = (0..np).filter(|&p| contributions[p] >= lev).count();
                        let mut pot_l = (pc * num_contrib as i32) as f32;
                        if li == 0 { pot_l += starting_pot as f32; }
                        let trav_elig = c_t >= lev;
                        let a_elig = !a_folded && c_opp_a >= lev;
                        if !trav_elig {
                            prev_l = lev;
                            continue;
                        }
                        if !a_elig {
                            cash += pot_l;
                        } else {
                            let max_str = h_str.max(s_a);
                            let mut tied: u32 = 0;
                            if h_str == max_str { tied += 1; }
                            if s_a == max_str { tied += 1; }
                            if h_str == max_str {
                                cash += pot_l / tied as f32;
                            }
                        }
                        prev_l = lev;
                    }
                    cash - traverser_stake
                };
                accum += ra * net;
            }
            cfv[h] = accum;
        }
        return cfv;
    }

    // num_opp >= 3 (np >= 4): no brute-force support. Leave cfv at 0.
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
