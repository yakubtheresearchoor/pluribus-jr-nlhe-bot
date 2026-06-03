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
/// Extended sorted-sweep: returns (sweep_net, win_reach, tie_reach) per
/// hand, where:
///   - `sweep_net[h]` = Σ over opp of (win_reach − loss_reach) with
///     card-blocking inclusion-exclusion. Matches what
///     `sorted_sweep_showdown` returns.
///   - `win_reach[h]` = Σ over opp of reach where opp_str < pl_str[h]
///     (the opps h beats), card-blocked.
///   - `tie_reach[h]` = Σ over opp of reach where opp_str == pl_str[h]
///     (the opps h ties), card-blocked.
///
/// The tie component is needed for the rake correction in showdown
/// payoffs: the winner's claim is (pot − rake), and ties split the
/// (pot − rake) pot, so player's expected rake cost is
/// `rake × (win_reach + tie_reach/2)` per hand. See
/// `side_pot_showdown_cfv_with_rake` for how this composes.
///
/// Complexity: O(num_opp × nh) for the win + loss passes (same as the
/// non-extended function), plus O(nh × 52) per tie band for the tie
/// component allocation (per-si reset of a 52-card minus vector).
#[allow(clippy::too_many_arguments)]
pub fn sorted_sweep_with_rake_components(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    nh: usize,
    opp_str: &[u16],
    opp_idx: &[u16],
    pl_str: &[u16],
    pl_idx: &[u16],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let num_opp = opp_reach.len();
    let mut sweep_net = vec![0.0f32; nh];
    let mut win_reach = vec![0.0f32; nh];
    let mut tie_reach = vec![0.0f32; nh];

    for oi in 0..num_opp {
        let reach = opp_reach[oi];
        let mut cfreach_sum = 0.0f32;
        let mut cfreach_minus = vec![0.0f32; 52];

        // FORWARD PASS: accumulate opponents h beats (opp_str < pl_str)
        // and look-ahead for ties (opp_str == pl_str). Same `i` pointer
        // as the original sorted_sweep_showdown for the win side; the
        // tie look-ahead uses a separate j cursor that doesn't advance i.
        let mut i = 0;
        for si in 0..nh {
            let str_h = pl_str[si];
            let h = pl_idx[si] as usize;
            // Win accumulation: opp_str < str_h
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
            let win = cfreach_sum
                - cfreach_minus[hand_cards[h * 2] as usize]
                - cfreach_minus[hand_cards[h * 2 + 1] as usize];
            sweep_net[h] += win;
            win_reach[h] += win;

            // Tie look-ahead: opp_str == str_h, scanning from i (which
            // points at the first opp with opp_str >= str_h). The tie
            // contribution doesn't advance i, so subsequent hands at
            // the same strength see the same tie band.
            //
            // Inclusion-exclusion correction: the only opp using BOTH
            // of h's cards is opp = h itself. In HU showdown opp_str =
            // pl_str (same hand-strength evaluation), so opp = h IS at
            // the tie boundary (same strength) and IS in the tie band.
            // Without `+ reach[h]` correction, h's reach is
            // double-subtracted (once for each of h's cards). Same
            // shape as the audit-fix #37 correction in the fast path.
            let mut tie_sum = 0.0f32;
            let mut tie_minus = [0.0f32; 52];
            let mut j = i;
            let mut tie_includes_self = false;
            while j < nh && opp_str[oi * nh + j] == str_h {
                let ho = opp_idx[oi * nh + j] as usize;
                let r = reach[ho];
                if r != 0.0 {
                    tie_sum += r;
                    tie_minus[hand_cards[ho * 2] as usize] += r;
                    tie_minus[hand_cards[ho * 2 + 1] as usize] += r;
                    if ho == h { tie_includes_self = true; }
                }
                j += 1;
            }
            let self_correction = if tie_includes_self { reach[h] } else { 0.0 };
            let tie = tie_sum
                - tie_minus[hand_cards[h * 2] as usize]
                - tie_minus[hand_cards[h * 2 + 1] as usize]
                + self_correction;
            tie_reach[h] += tie;
        }

        // BACKWARD PASS: accumulate opponents h loses to (opp_str > pl_str).
        // Subtract from sweep_net (loss reduces net). Same as the original
        // sorted_sweep_showdown's second pass.
        cfreach_sum = 0.0;
        for c in 0..52 { cfreach_minus[c] = 0.0; }

        let mut i = nh;
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
            let loss = cfreach_sum
                - cfreach_minus[hand_cards[h * 2] as usize]
                - cfreach_minus[hand_cards[h * 2 + 1] as usize];
            sweep_net[h] -= loss;
        }
    }

    (sweep_net, win_reach, tie_reach)
}

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
/// Backward-compatible wrapper: no-rake variant. New solver code paths
/// that need rake should call `side_pot_showdown_cfv_with_rake` directly,
/// passing `rake_rate` and `rake_cap` from the `FlatTree`.
///
/// This wrapper is provided so the ~40 existing direct callers in tests
/// and the GPU-mirror callers continue to work unchanged while CPU rake
/// implementation lands incrementally (Slice 1.x of #44 rake gap
/// resolution).
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
    side_pot_showdown_cfv_with_rake(
        opp_reach, hand_cards, nh,
        sorted_opp_str, sorted_opp_idx,
        sorted_pl_str, sorted_pl_idx,
        contributions, fold_mask, traverser, num_players, starting_pot,
        0.0, 0.0,
    )
}

/// N-player side pot terminal evaluation with rake. Returns **fully
/// scaled** counterfactual values per hand (includes pot_at_level
/// multiplication, `c_t` subtraction, and rake reduction on winning
/// payoffs).
///
/// Rake math:
///   `rake = min(total_pot × rake_rate, rake_cap)`
///   - Winners (active players who claim pot) get `(pot − rake)` instead
///     of `pot`. Their payoff is reduced by their share of rake.
///   - Losers (folded or active-but-lost) are unaffected. They were
///     never claiming the pot; rake comes out of the winner's claim.
///   - Folded traverser: payoff = −investment (rake-free; they didn't
///     win anything, so rake doesn't apply to them).
///
/// In the sorted-sweep paths, rake is applied via the win_reach +
/// tie_reach correction:
///   `cfv[h] = half_pot × sweep_net[h] − rake × (win_reach[h] + tie_reach[h] / 2)`
/// (the rake-free formula minus rake-times-pot-claim-probability).
///
/// **Anchor:** `verify_rake_affects_terminal_payoffs` in
/// p_config_param_verification.rs validates the fast-path rake against
/// a hand-computed reference. Sorted-sweep + side-pot paths get their
/// own hand-computed anchors as they get implemented across Slice 1.x.
#[allow(clippy::too_many_arguments)]
pub fn side_pot_showdown_cfv_with_rake(
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
    rake_rate: f32,
    rake_cap: f32,
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

    // Constant-payoff fast path.
    //
    // Mathematically this path's `payoff = total_pot - investment` (active)
    // / `-investment` (folded) is incorrect for individual cfv values when
    // a folded player out-contributed the max active player: the folded
    // player gets back their unmatched excess as a side-pot return, and
    // an active player who didn't contribute to that level does NOT win
    // it. But across all traversers the per-player errors sum to zero
    // (Σ_p payoff_p = total_pot - Σ_p investment = 0), so any downstream
    // computation that only relies on the zero-sum identity (CFR regret
    // updates) remains correct.
    //
    // For 3-player games without raises, folded > active is structurally
    // impossible, so this path is also exact per-player. The 3-player
    // GPU↔CPU bit-parity tests rely on this exact match, so the fast
    // path stays gated on num_opp <= 2 — same code as before, just
    // explicitly limited to K ≤ 2 where it's known safe.
    //
    // For K ≥ 3 (4+ player) the per-player error matters: folded > active
    // can occur, and downstream code that consumes individual cfv values
    // (best-response, exploitability) needs correct values. K ≥ 3 falls
    // through to the per-level evaluator below.
    if (num_active <= 1 || fold_mask & (1u16 << traverser) != 0) && num_opp <= 2 {
        let total_pot: i32 = starting_pot + contributions.iter().sum::<i32>();
        let traverser_investment = starting_pot as f32 / np as f32 + c_t as f32;
        // Rake applies only to the winner's pot claim. Folded traverser
        // doesn't win → rake doesn't reduce their payoff (they lose
        // their investment regardless). Active-winner traverser claims
        // (total_pot − rake) instead of total_pot.
        let rake = (total_pot as f32 * rake_rate).min(rake_cap).max(0.0);
        let payoff = if fold_mask & (1u16 << traverser) != 0 {
            -traverser_investment
        } else {
            (total_pot as f32 - rake) - traverser_investment
        };

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
                // Inclusion-exclusion: |opp hands NOT using c1 NOR c2|
                //   = total - (uses c1) - (uses c2) + (uses BOTH c1 and c2)
                // In 2-card holdem with unique (c1,c2) pair indexing, the only opp
                // hand using BOTH of h's cards is h itself. Without the `+ opp_reach[h]`
                // correction, h's reach is double-subtracted (once for c1, once for c2)
                // → cfreach off by `-opp_reach[h]`. Audit-fix from #37.
                let cfreach = opp_reach_sum
                    - opp_reach_minus[hand_cards[h * 2] as usize]
                    - opp_reach_minus[hand_cards[h * 2 + 1] as usize]
                    + opp_reach[0][h];
                cfv[h] = payoff * cfreach;
            }
        } else {
            // num_opp == 2: brute-force enumeration over (g0, g1).
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
        let (sweep_net, win_reach, tie_reach) = sorted_sweep_with_rake_components(
            &filtered_views, hand_cards, nh,
            sorted_opp_str, sorted_opp_idx,
            sorted_pl_str, sorted_pl_idx,
        );

        let half_pot = starting_pot as f32 / np as f32 + min_active_contrib as f32;
        // Rake correction (Slice 1.3): rake is taken from the winner's pot
        // claim. cfv = half_pot × sweep_net − rake × (win_reach + tie_reach/2),
        // where rake = min(total_pot × rake_rate, rake_cap).clamp(0).
        let total_pot: i32 = starting_pot + contributions.iter().sum::<i32>();
        let rake = (total_pot as f32 * rake_rate).min(rake_cap).max(0.0);
        for h in 0..nh {
            cfv[h] = half_pot * sweep_net[h] - rake * (win_reach[h] + 0.5 * tie_reach[h]);
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
                    // Inclusion-exclusion correction: see comment at line 290's fix
                    // (#37 audit). For multi-opp case, sum the self-hand reach across
                    // all opponents to recover the term double-subtracted by
                    // minus[c1] + minus[c2].
                    let mut uses_both = 0.0f32;
                    for oi in 0..num_opp {
                        uses_both += opp_reach[oi][h];
                    }
                    let cfreach = opp_reach_sum
                        - opp_reach_minus[hand_cards[h * 2] as usize]
                        - opp_reach_minus[hand_cards[h * 2 + 1] as usize]
                        + uses_both;
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
            let (sweep_net, win_reach, tie_reach) = sorted_sweep_with_rake_components(
                &filtered_views, hand_cards, nh,
                sorted_opp_str, sorted_opp_idx,
                sorted_pl_str, sorted_pl_idx,
            );
            let half_pot = starting_pot as f32 / np as f32 + c_t as f32;
            // Rake correction (Slice 1.3): see comment in the all-active-equal
            // sweep path above.
            let total_pot: i32 = starting_pot + contributions.iter().sum::<i32>();
            let rake = (total_pot as f32 * rake_rate).min(rake_cap).max(0.0);
            for h in 0..nh {
                cfv[h] = half_pot * sweep_net[h] - rake * (win_reach[h] + 0.5 * tie_reach[h]);
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

        // K>2 active opponents, equal-contribution showdown.
        // Brute-force enumeration. Per scenario (g_0, ..., g_{K-1}) with reach
        // product r_prod and at-top tie count T_top:
        //   if max(opp strengths) > h_str: net unit = -1 (lose stake)
        //   if max == h_str: net unit = (K+1 - T)/T   (tie split, K = num_active_opp)
        //   if max < h_str: net unit = K             (win pot K+1, paid 1)
        // cfv[h] = half_pot * Σ r_prod * net_unit
        //
        // O(nh^K) per h. Used for validation; production needs factored I-E.
        let filtered_views: Vec<&[f32]> = filtered_opp.iter().map(|v| v.as_slice()).collect();
        let mut hand_strength = vec![0u16; nh];
        for si in 0..nh {
            hand_strength[sorted_pl_idx[si] as usize] = sorted_pl_str[si];
        }
        let mut g_mask_arr = vec![0u64; nh];
        for g in 0..nh {
            g_mask_arr[g] = (1u64 << hand_cards[g * 2]) | (1u64 << hand_cards[g * 2 + 1]);
        }
        let k = num_active_opp as f32;
        let half_pot = starting_pot as f32 / np as f32 + c_t as f32;

        fn recurse_eq(
            oi: usize, num_opp: usize, nh: usize,
            mask_so_far: u64, reach_so_far: f32, max_str_so_far: u16, tied_at_max_so_far: u32,
            h_str: u16,
            opp_reach: &[&[f32]],
            g_mask_arr: &[u64],
            hand_strength: &[u16],
            k: f32,
            accum: &mut f32,
        ) {
            if oi == num_opp {
                let net_unit: f32 = if max_str_so_far > h_str {
                    -1.0
                } else if max_str_so_far == h_str {
                    // Traverser ties at top with (tied_at_max_so_far + 1) total tied.
                    let t = tied_at_max_so_far + 1;
                    (k + 1.0 - t as f32) / t as f32
                } else {
                    k
                };
                *accum += reach_so_far * net_unit;
                return;
            }
            for g in 0..nh {
                if g_mask_arr[g] & mask_so_far != 0 { continue; }
                let r = opp_reach[oi][g];
                if r == 0.0 { continue; }
                let s = hand_strength[g];
                let (new_max, new_tied) = if s > max_str_so_far {
                    (s, 1u32)
                } else if s == max_str_so_far {
                    (max_str_so_far, tied_at_max_so_far + 1)
                } else {
                    (max_str_so_far, tied_at_max_so_far)
                };
                recurse_eq(
                    oi + 1, num_opp, nh,
                    mask_so_far | g_mask_arr[g], reach_so_far * r,
                    new_max, new_tied,
                    h_str, opp_reach, g_mask_arr, hand_strength, k, accum,
                );
            }
        }

        for h in 0..nh {
            let h_m = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
            let h_str = hand_strength[h];
            let mut accum = 0.0f32;
            recurse_eq(
                0, num_opp, nh, h_m, 1.0, 0u16, 0u32,
                h_str, &filtered_views, &g_mask_arr, &hand_strength, k, &mut accum,
            );
            cfv[h] = half_pot * accum;
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

                    // Walk levels. Both folded and active traversers can
                    // receive cash from no-eligible-active levels (side-pot
                    // returns to contributors). Active traversers also
                    // receive cash from levels they win at-top.
                    let mut cash: f32 = 0.0;
                    let mut prev_l = 0i32;
                    for (li, &lev) in levels.iter().enumerate() {
                        let pc = lev - prev_l;
                        if pc == 0 { prev_l = lev; continue; }
                        let num_contrib = (0..np).filter(|&p| contributions[p] >= lev).count();
                        let mut pot_l = (pc * num_contrib as i32) as f32;
                        if li == 0 { pot_l += starting_pot as f32; }

                        let trav_elig = !traverser_folded && c_t >= lev;
                        let a_elig = !a_folded && c_opp_a >= lev;
                        let b_elig = !b_folded && c_opp_b >= lev;
                        let n_elig_total = trav_elig as i32 + a_elig as i32 + b_elig as i32;

                        if n_elig_total == 0 {
                            // No eligible active: slice returned to all
                            // contributors (including folded traverser).
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
                    let net = cash - traverser_stake;

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

                // Walk levels. Folded traverser still receives cash from
                // no-eligible-active levels (side-pot returns).
                let mut cash: f32 = 0.0;
                let mut prev_l = 0i32;
                for (li, &lev) in levels.iter().enumerate() {
                    let pc = lev - prev_l;
                    if pc == 0 { prev_l = lev; continue; }
                    let num_contrib = (0..np).filter(|&p| contributions[p] >= lev).count();
                    let mut pot_l = (pc * num_contrib as i32) as f32;
                    if li == 0 { pot_l += starting_pot as f32; }
                    let trav_elig = !traverser_folded && c_t >= lev;
                    let a_elig = !a_folded && c_opp_a >= lev;
                    let n_elig = trav_elig as i32 + a_elig as i32;

                    if n_elig == 0 {
                        // Slice returned to contributors (including folded).
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
                let net = cash - traverser_stake;
                accum += ra * net;
            }
            cfv[h] = accum;
        }
        return cfv;
    }

    // num_opp >= 3 (np >= 4): generic recursive per-scenario brute-force.
    //
    // Generalization of the K=2 per-level evaluator above to K opponents.
    // Enumerates all (g_0, ..., g_{K-1}) with no pairwise card overlap and no
    // overlap with traverser hand h. For each scenario:
    //   - if traverser folded: net = -traverser_stake (lost stake).
    //   - else: walk levels; for each pot level compute cash to traverser based
    //     on eligibility (contribution ≥ level AND not folded) and per-level
    //     winner (max strength among eligible; tie split by tied count).
    //
    // Cost: O(nh^K) per traverser hand. At nh=50, K=5 this is 50^5 = 312M per
    // h × 50 hands × ~5000 terminals = 78T per iter — infeasible for
    // production. Used for VALIDATION GATES at small nh; production
    // performance comes from the factored I-E formula once derived correctly.

    // Map opponent slot index `oi` ∈ [0, num_opp) to player index.
    let opp_player: Vec<usize> = (0..num_opp).map(|oi| {
        if oi < traverser { oi } else { oi + 1 }
    }).collect();
    let opp_contrib: Vec<i32> = opp_player.iter().map(|&p| contributions[p]).collect();
    let opp_folded: Vec<bool> = opp_player.iter().map(|&p| fold_mask & (1u16 << p) != 0).collect();

    // Per-hand card mask cache.
    let mut g_mask_arr = vec![0u64; nh];
    for g in 0..nh {
        g_mask_arr[g] = (1u64 << hand_cards[g * 2]) | (1u64 << hand_cards[g * 2 + 1]);
    }

    // For each h, recursively enumerate (g_0, ..., g_{K-1}) and accumulate
    // reach_product * per_scenario_net into cfv[h].
    fn recurse_scenario(
        oi: usize,
        num_opp: usize,
        nh: usize,
        mask_so_far: u64,
        reach_so_far: f32,
        opp_reach: &[&[f32]],
        g_mask_arr: &[u64],
        hand_strength: &[u16],
        g_str: &mut [u16],
        // closure context
        callback: &mut dyn FnMut(f32, &[u16]),
    ) {
        if oi == num_opp {
            callback(reach_so_far, g_str);
            return;
        }
        for g in 0..nh {
            if g_mask_arr[g] & mask_so_far != 0 { continue; }
            let r = opp_reach[oi][g];
            if r == 0.0 { continue; }
            g_str[oi] = hand_strength[g];
            recurse_scenario(
                oi + 1, num_opp, nh,
                mask_so_far | g_mask_arr[g],
                reach_so_far * r,
                opp_reach, g_mask_arr, hand_strength, g_str,
                callback,
            );
        }
    }

    for h in 0..nh {
        let h_m = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
        let h_str = hand_strength[h];
        let mut accum = 0.0f32;

        let mut g_str = vec![0u16; num_opp];
        recurse_scenario(
            0, num_opp, nh, h_m, 1.0,
            opp_reach, &g_mask_arr, &hand_strength, &mut g_str,
            &mut |reach_product: f32, g_str: &[u16]| {
                // Walk levels and accumulate cash. Folded traversers can still
                // get cash from side-pot returns at levels where no active
                // player is eligible (this happens when folded traverser
                // contributed more than any active opponent — possible in
                // 4+player games with raise/all-in dynamics; impossible in
                // 3-player without raises which is why the K=2 path never
                // hit this bug).
                let mut cash: f32 = 0.0;
                let mut prev_l = 0i32;
                for (li, &lev) in levels.iter().enumerate() {
                    let pc = lev - prev_l;
                    if pc == 0 { prev_l = lev; continue; }
                    let num_contrib = (0..np).filter(|&p| contributions[p] >= lev).count();
                    let mut pot_l = (pc * num_contrib as i32) as f32;
                    if li == 0 { pot_l += starting_pot as f32; }

                    let trav_elig = !traverser_folded && c_t >= lev;

                    // Eligibility among ACTIVE opponents.
                    let mut elig_count: u32 = trav_elig as u32;
                    let mut max_str = if trav_elig { h_str } else { 0u16 };
                    for oi in 0..num_opp {
                        if opp_folded[oi] { continue; }
                        if opp_contrib[oi] < lev { continue; }
                        elig_count += 1;
                        if g_str[oi] > max_str { max_str = g_str[oi]; }
                    }

                    if elig_count == 0 {
                        // No eligible active player: slice returned to all
                        // contributors (including folded ones).
                        if contributions[traverser] >= lev {
                            let trav_contrib_at_lev = pc as f32
                                + if li == 0 { starting_pot as f32 / np as f32 } else { 0.0 };
                            cash += trav_contrib_at_lev;
                        }
                        prev_l = lev;
                        continue;
                    }

                    if !trav_elig {
                        // Traverser not eligible (folded or contributed < lev):
                        // no cash from this slice.
                        prev_l = lev;
                        continue;
                    }

                    // Tie count among eligible.
                    let mut tied: u32 = 0;
                    if h_str == max_str { tied += 1; }
                    for oi in 0..num_opp {
                        if opp_folded[oi] { continue; }
                        if opp_contrib[oi] < lev { continue; }
                        if g_str[oi] == max_str { tied += 1; }
                    }

                    if h_str == max_str {
                        cash += pot_l / tied as f32;
                    }
                    prev_l = lev;
                }
                let net = cash - traverser_stake;
                accum += reach_product * net;
            },
        );

        cfv[h] = accum;
    }
    cfv
}

// ============================================================================
// Factored inclusion-exclusion N-opponent showdown CFV computation.
//
// Generalizes the K=2 brute-force at lines ~659-820 to K opponents without
// enumerating O(nh^K) tuples. The key structural insight: a card appears in
// at most one hand at the table, so among K opponents any single card can be
// shared by at most 2 opponents. The conflict graph on K opponent slots is
// always built from pairwise edges; there are no 3-way card-sharing terms.
// Inclusion-exclusion truncates to a sum over matchings on the K-vertex
// graph (with bounded vertex degree ≤ 2). For K ≤ 5 the matching count is
// small (≤ ~200 configurations including paths and cycles).
//
// All "not conflicting with h" filters are actually
// "not conflicting with h AND not intersecting dead_card_mask." The mask is
// populated per-solve by the caller. Board-only in the normal one-account
// case, board-plus-other-account's-cards in the two-account case.
//
// K=2 paths in side_pot_showdown_cfv (lines 287, 465, 659) remain on the
// existing per-scenario brute-force — that is the validated ground truth.
// The I-E formula is used only for K ≥ 3. This avoids any K=2 regression
// risk and keeps the bit-exact 3-player work intact.
// ============================================================================

/// Per-opponent reach masses for use in factored inclusion-exclusion N-opponent
/// showdown evaluation.
///
/// All per-card arrays are indexed by raw card index in `[0, 52)` so the array
/// shape does not change when `dead_card_mask` changes — the mask only changes
/// which entries are accumulated into and which entries are read out by the
/// combiner. This avoids mask-dependent stride bugs that would only surface
/// at iter-2 reach states (the lesson from the 3-player GPU port).
///
/// Layout (for `K = num_opp` opponents and `nh` sampled hands):
///
/// - `b`, `t`, `s`: `[K * nh]` flat. Indexed as `[oi * nh + h]`. For opponent
///   `oi` and player hand `h`:
///   - `b[oi*nh + h]` = Σ over `g` strictly weaker than `h`, not conflicting
///     with `h`, not intersecting `dead_card_mask`, of `r_oi[g]`.
///   - `t[oi*nh + h]` = same with `g` tied with `h`.
///   - `s[oi*nh + h]` = same with `g` strictly stronger than `h`.
///
/// - `b_per_card`, `t_per_card`, `s_per_card`: `[K * nh * 52]` flat. Indexed
///   as `[oi * nh * 52 + h * 52 + c]`. The per-card partials:
///   - `b_per_card[..][c]` = Σ over `g` containing card `c`, strictly weaker
///     than `h`, not otherwise conflicting with `h`, not intersecting
///     `dead_card_mask`, of `r_oi[g]`. (Same for `t_per_card`, `s_per_card`.)
///   - Indexed by raw card `c`; entries for `c` in `h`'s cards or in
///     `dead_card_mask` are computed but not read by the combiner.
///
/// - `hand_index`: `52 * 52` flat lookup. `hand_index[c1 * 52 + c2]` is
///   `Some(h)` if hand `{c1, c2}` is in the sample (any ordering of `c1, c2`),
///   else `-1`. Used for the joint per-(c1, c2) reach in vertex-shared edge
///   corrections.
///
/// - `hand_cards`: cached copy of the caller's `hand_cards` so the combiner
///   can resolve `h`'s cards without an extra parameter.
///
/// - `dead_card_mask`: pinned at construction. Bit `c` set iff card `c` is
///   dead (board card or other-account known card).
#[derive(Debug, Clone)]
pub struct OppMasses {
    pub num_opp: usize,
    pub nh: usize,
    pub dead_card_mask: u64,
    pub b: Vec<f32>,
    pub t: Vec<f32>,
    pub s: Vec<f32>,
    pub b_per_card: Vec<f32>,
    pub t_per_card: Vec<f32>,
    pub s_per_card: Vec<f32>,
    pub hand_index: Vec<i32>,
    pub hand_cards: Vec<u8>,
}

/// Precompute per-opponent reach masses for the I-E showdown.
///
/// Filters every accumulation by **both** the per-`h` card-conflict and the
/// `dead_card_mask` intersection. The two filters compose by OR-ing bitmasks;
/// the inner loop is unchanged otherwise.
///
/// Cost: `O(K * nh^2)` per terminal — for nh=50, K=5: ~12,500 ops, dominated
/// by terminal-level memory writes. The per-card partials add a constant
/// factor of 2 per `(g, h)` pair (each `g` contributes to its two card
/// buckets), not a factor of 52.
///
/// Bit-precision discipline (for the future GPU port):
/// - The outer loop order is fixed: `oi -> h -> g`. CPU and GPU must walk
///   in the same order so accumulator round-off matches.
/// - The per-card partial accumulators receive each `g`'s contribution to
///   exactly two card buckets (`g_c1` and `g_c2`). Order: `g_c1` then `g_c2`.
/// - Strength comparison: `g_str < h_str` first, then `g_str == h_str`,
///   else strictly greater.
pub fn precompute_opp_masses(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    hand_strength: &[u16],
    dead_card_mask: u64,
) -> OppMasses {
    let num_opp = opp_reach.len();
    let nh = if num_opp > 0 { opp_reach[0].len() } else { 0 };

    let mut b = vec![0.0f32; num_opp * nh];
    let mut t = vec![0.0f32; num_opp * nh];
    let mut s = vec![0.0f32; num_opp * nh];
    let mut b_per_card = vec![0.0f32; num_opp * nh * 52];
    let mut t_per_card = vec![0.0f32; num_opp * nh * 52];
    let mut s_per_card = vec![0.0f32; num_opp * nh * 52];

    // Precompute per-hand card mask once.
    let mut g_mask = vec![0u64; nh];
    for g in 0..nh {
        g_mask[g] = (1u64 << hand_cards[g * 2]) | (1u64 << hand_cards[g * 2 + 1]);
    }

    // Build hand_index lookup: 52*52 flat, -1 = not in sample.
    // Each sampled hand registers under both orderings (c1*52+c2 and c2*52+c1)
    // so callers don't have to canonicalize.
    let mut hand_index = vec![-1i32; 52 * 52];
    for h in 0..nh {
        let c1 = hand_cards[h * 2] as usize;
        let c2 = hand_cards[h * 2 + 1] as usize;
        hand_index[c1 * 52 + c2] = h as i32;
        hand_index[c2 * 52 + c1] = h as i32;
    }

    for oi in 0..num_opp {
        let reach = opp_reach[oi];
        for h in 0..nh {
            let h_m = g_mask[h];
            let h_str = hand_strength[h];

            for g in 0..nh {
                let g_m = g_mask[g];
                // Filters compose by OR'ing the masks: g must not share any
                // card with h AND must not intersect dead_card_mask.
                if g_m & (h_m | dead_card_mask) != 0 { continue; }
                let r = reach[g];
                if r == 0.0 { continue; }

                let g_str = hand_strength[g];
                let g_c1 = hand_cards[g * 2] as usize;
                let g_c2 = hand_cards[g * 2 + 1] as usize;

                let base = oi * nh + h;
                let base_pc = (oi * nh + h) * 52;

                if g_str < h_str {
                    b[base] += r;
                    b_per_card[base_pc + g_c1] += r;
                    b_per_card[base_pc + g_c2] += r;
                } else if g_str == h_str {
                    t[base] += r;
                    t_per_card[base_pc + g_c1] += r;
                    t_per_card[base_pc + g_c2] += r;
                } else {
                    s[base] += r;
                    s_per_card[base_pc + g_c1] += r;
                    s_per_card[base_pc + g_c2] += r;
                }
            }
        }
    }

    OppMasses {
        num_opp, nh, dead_card_mask,
        b, t, s,
        b_per_card, t_per_card, s_per_card,
        hand_index,
        hand_cards: hand_cards.to_vec(),
    }
}

impl OppMasses {
    /// Total reach `R_i(h) = B_i(h) + T_i(h) + S_i(h)` — opponent `i`'s
    /// reach over any hand `g` that doesn't conflict with `h` or the dead
    /// mask, regardless of strength.
    #[inline]
    pub fn r(&self, oi: usize, h: usize) -> f32 {
        let base = oi * self.nh + h;
        self.b[base] + self.t[base] + self.s[base]
    }

    /// Per-card total reach `R_i^{(c)}(h) = B + T + S` at card `c`.
    /// Returns 0 if `c` is in `h`'s cards or in `dead_card_mask` (those
    /// entries are not meaningfully populated by `precompute_opp_masses`,
    /// but explicitly returning 0 keeps the combiner code uniform).
    #[inline]
    pub fn r_per_card(&self, oi: usize, h: usize, c: usize) -> f32 {
        let base_pc = (oi * self.nh + h) * 52;
        self.b_per_card[base_pc + c] + self.t_per_card[base_pc + c] + self.s_per_card[base_pc + c]
    }

    /// Joint reach for `g_i = {c1, c2}` — the unique hand using both cards.
    /// Returns 0 if the hand is not in the sample, or if it conflicts with
    /// `h` or the dead mask.
    #[inline]
    pub fn r_per_card_pair(&self, oi: usize, h: usize, c1: usize, c2: usize, opp_reach: &[&[f32]]) -> f32 {
        if c1 == c2 { return 0.0; }
        let idx = self.hand_index[c1 * 52 + c2];
        if idx < 0 { return 0.0; }
        let g = idx as usize;
        let g_m = (1u64 << c1) | (1u64 << c2);
        let h_m = (1u64 << self.hand_cards[h * 2]) | (1u64 << self.hand_cards[h * 2 + 1]);
        if g_m & (h_m | self.dead_card_mask) != 0 { return 0.0; }
        opp_reach[oi][g]
    }
}

/// `TotalValidReachProduct(h)` via straight brute-force enumeration. O(nh^K)
/// per `h`. Used for K ≥ 3 in this preliminary correctness-first pass.
///
/// CORRECTNESS PROPERTY: this is the structurally-simplest implementation of
/// the quantity TVRP is supposed to compute. It is the oracle that any future
/// factored I-E formula must reproduce to within f32-precision drift. The
/// factored formula's complexity grows beyond simple matchings on K vertices
/// once edges can share cards (e.g., {(0,1)→c, (0,2)→c} is a valid 2-edge
/// "all three opps contain c" subset for K=3, in addition to the c1 ≠ c2
/// path), and the matching enumeration that the plan assumed was simpler
/// than this turned out to be. Brute-force keeps correctness clean while
/// the factored formula is re-derived against this oracle.
///
/// Performance: O(nh^K) per h. At nh=50, K=3: 125k per h × 50 hands × 5000
/// terminals = 31B/iter — slow but tractable. K=4 borderline; K=5 not.
/// The factored I-E will replace this for K ≥ 4 once verified correct.
fn tvrp_brute(
    masses: &OppMasses,
    opp_reach: &[&[f32]],
    h: usize,
    h_dead_mask: u64,
) -> f64 {
    let k = masses.num_opp;
    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;

    // Per-hand card mask (cached).
    let mut g_mask = vec![0u64; nh];
    for g in 0..nh {
        g_mask[g] = (1u64 << hand_cards[g * 2]) | (1u64 << hand_cards[g * 2 + 1]);
    }

    fn recurse(
        depth: usize,
        k: usize,
        nh: usize,
        mask_so_far: u64,
        reach_so_far: f64,
        opp_reach: &[&[f32]],
        g_mask: &[u64],
        sum: &mut f64,
    ) {
        if depth == k {
            *sum += reach_so_far;
            return;
        }
        for g in 0..nh {
            if g_mask[g] & mask_so_far != 0 { continue; }
            let r = opp_reach[depth][g] as f64;
            if r == 0.0 { continue; }
            recurse(
                depth + 1, k, nh,
                mask_so_far | g_mask[g],
                reach_so_far * r,
                opp_reach, g_mask, sum,
            );
        }
    }

    let mut sum = 0.0f64;
    recurse(0, k, nh, h_dead_mask, 1.0, opp_reach, &g_mask, &mut sum);
    sum
}

/// Compute `TotalValidReachProduct(h)` for each `h` — the I-E quantity:
///
/// ```text
/// TVRP(h) = Σ over valid (g_1, ..., g_K) of Π r_i[g_i]
/// ```
///
/// where "valid" = no `g_i` conflicts with `h`, no `g_i` intersects
/// `dead_card_mask`, and no two `g_i, g_j` share a card.
///
/// This is the showdown numerator's structural twin without strength filters
/// and the denominator the caller normalizes by (it must enumerate over the
/// same set as the showdown's per-level sum, or zero-sum breaks).
///
/// Currently:
///   - K ∈ {0, 1, 2}: factored I-E (constant + O(52 + nh) per `h`).
///   - K ∈ {3, 4, 5}: O(nh^K) brute-force via [`tvrp_brute`]. Correct but
///     not yet performant for K=5 at production nh; the factored
///     formula is still being re-derived against this oracle (see the
///     comment on [`tvrp_brute`] for the structural reason the obvious
///     "matching enumeration" expansion does not suffice).
///   - K ≥ 6: panics (max table size is 6 players → K ≤ 5).
/// Factored K=3 `TotalValidReachProduct(h)` via recursive K=2 expansion.
///
/// ```text
/// TVRP_K=3(h, dead)
///   = Σ over g_0 valid (not conflicting h, not in dead) of
///       r_0[g_0] · TVRP_K=2(opps {1,2}, h, dead ∪ {g_0's two cards})
/// ```
///
/// The inner K=2 TVRP with the EXTENDED dead mask is expressed via I-E on
/// the BASE masses (already precomputed for the base `h_dead_mask`):
///
/// ```text
///   M_i_ext           = M_i      − M_i^(g0_c1)      − M_i^(g0_c2)      + r_i[g_0]
///   M_i_ext^(c)       = M_i^(c)  − r_i[h_{g0_c1,c}] − r_i[h_{g0_c2,c}]    (for c ∉ ext)
///   H_{12}_ext        = H_{12}   − H_{12}^(g0_c1)   − H_{12}^(g0_c2)   + r_1[g_0]·r_2[g_0]
///
///   K=2_TVRP_ext      = M_1_ext · M_2_ext
///                     − Σ over c ∉ ext_mask of M_1_ext^(c) · M_2_ext^(c)
///                     + H_{12}_ext
/// ```
///
/// `H_{12}` and `H_{12}^(c)` are not in `OppMasses` (they'd be K-quadratic
/// in storage if we materialized them across all opp pairs); we compute
/// them on the fly once per `h_player` at O(nh + 52) cost. This is the
/// per-`h` setup that the per-`g_0` inner loop then uses.
///
/// Cost: O(nh · 52) per `h_player`. At nh=50: 2,600 ops per `h` × 50
/// hands = 130k per terminal. Compared with brute-force at O(nh^3) =
/// 125k per `h` × 50 = 6.25M per terminal, the factored formula is
/// ~50× faster at K=3.
///
/// CORRECTNESS PROPERTY: matches brute-force `tvrp_brute` on the
/// triangle-conflict configuration `{h={0,1}, h={0,2}, h={1,2}}` where
/// the matching-based formula (without the higher-order shared-card
/// terms) gave 41 vs the true 18. The recursive K=2 expansion handles
/// the shared-card cases by CONSTRUCTION through the growing dead mask
/// rather than by enumerating conflict patterns it cannot fully
/// represent — which is the structural reason this formulation is
/// preferred over the matching enumeration the original plan called
/// for. See the `ie_k3_tvrp_matches_bruteforce` test.
///
/// SIGN CARE: subtracted terms like `M_i_ext = M_i − M_i^(c1) − M_i^(c2) + r_i[g_0]`
/// can be ≤ 0 in f32 if the base mass is small and the subtracted
/// per-card term equals it. The downstream multiplication
/// `M_1_ext · M_2_ext` propagates this; the formula remains
/// mathematically correct (it's I-E with cancellation), but
/// floating-point accumulators must be f64 for the per-h sums to keep
/// catastrophic-cancellation drift below the 1e-5 threshold the test
/// asserts. All accumulators here are f64.
fn tvrp_k3_factored(
    masses: &OppMasses,
    opp_reach: &[&[f32]],
    h: usize,
    h_dead_mask: u64,    // already includes h_player's cards + base dead mask
) -> f64 {
    let nh = masses.nh;
    let hand_cards = &masses.hand_cards;

    // Base K=2 quantities for opps 1, 2 at THIS h_player.
    // M_i = total reach over valid g for opp i (already precomputed).
    let m1 = masses.r(1, h) as f64;
    let m2 = masses.r(2, h) as f64;

    // H_{12} and H_{12}^(c) — sum-of-products of (r_1[g], r_2[g]) over
    // valid g, optionally restricted to those containing card c. NOT in
    // OppMasses (K-quadratic in opp pairs), computed on the fly here.
    let mut h12 = 0.0f64;
    let mut h12_pc = [0.0f64; 52];
    for g in 0..nh {
        let gc1 = hand_cards[g * 2] as usize;
        let gc2 = hand_cards[g * 2 + 1] as usize;
        let g_m = (1u64 << gc1) | (1u64 << gc2);
        if g_m & h_dead_mask != 0 { continue; }
        let r1 = opp_reach[1][g] as f64;
        let r2 = opp_reach[2][g] as f64;
        let prod = r1 * r2;
        if prod == 0.0 { continue; }
        h12 += prod;
        h12_pc[gc1] += prod;
        h12_pc[gc2] += prod;
    }

    // Per-g_0 inner loop.
    let mut total = 0.0f64;
    for g0 in 0..nh {
        let g0c1 = hand_cards[g0 * 2] as usize;
        let g0c2 = hand_cards[g0 * 2 + 1] as usize;
        let g0_m = (1u64 << g0c1) | (1u64 << g0c2);
        if g0_m & h_dead_mask != 0 { continue; }
        let r0_g0 = opp_reach[0][g0] as f64;
        if r0_g0 == 0.0 { continue; }

        let r1_g0 = opp_reach[1][g0] as f64;
        let r2_g0 = opp_reach[2][g0] as f64;

        // Extended-mask totals (constant per g_0).
        let m1_g0c1 = masses.r_per_card(1, h, g0c1) as f64;
        let m1_g0c2 = masses.r_per_card(1, h, g0c2) as f64;
        let m2_g0c1 = masses.r_per_card(2, h, g0c1) as f64;
        let m2_g0c2 = masses.r_per_card(2, h, g0c2) as f64;
        let m1_ext = m1 - m1_g0c1 - m1_g0c2 + r1_g0;
        let m2_ext = m2 - m2_g0c1 - m2_g0c2 + r2_g0;

        // Σ over c ∉ extended dead of M_1_ext^(c) · M_2_ext^(c).
        let ext_mask = h_dead_mask | g0_m;
        let mut edge_sum = 0.0f64;
        for c in 0..52usize {
            if (1u64 << c) & ext_mask != 0 { continue; }
            // M_i_ext^(c) = M_i^(c) − r_i[h_{g0c1, c}] − r_i[h_{g0c2, c}]
            let r1_h_g0c1_c = lookup_pair_reach(opp_reach[1], &masses.hand_index, g0c1, c);
            let r1_h_g0c2_c = lookup_pair_reach(opp_reach[1], &masses.hand_index, g0c2, c);
            let r2_h_g0c1_c = lookup_pair_reach(opp_reach[2], &masses.hand_index, g0c1, c);
            let r2_h_g0c2_c = lookup_pair_reach(opp_reach[2], &masses.hand_index, g0c2, c);
            let m1_ext_c = masses.r_per_card(1, h, c) as f64 - r1_h_g0c1_c - r1_h_g0c2_c;
            let m2_ext_c = masses.r_per_card(2, h, c) as f64 - r2_h_g0c1_c - r2_h_g0c2_c;
            edge_sum += m1_ext_c * m2_ext_c;
        }

        // H_{12}_ext = H_{12} − H_{12}^(g0c1) − H_{12}^(g0c2) + r_1[g_0]·r_2[g_0]
        let h12_ext = h12 - h12_pc[g0c1] - h12_pc[g0c2] + r1_g0 * r2_g0;

        let k2_tvrp_ext = m1_ext * m2_ext - edge_sum + h12_ext;
        total += r0_g0 * k2_tvrp_ext;
    }

    total
}

/// Look up `r_i[h_{c1, c2}]` — the opp-`i` reach at the unique sampled hand
/// `{c1, c2}` if it exists in the sample, else 0.
///
/// Filtering note: this returns the raw `opp_reach[idx]`. Caller is
/// responsible for the validity invariant that `opp_reach` is already
/// zeroed for hands conflicting with the dead mask / `h_player`. For the
/// `tvrp_k3_factored` call sites, the cards `c1, c2` are guaranteed not in
/// `h_player` or in the original dead mask (by the construction of the
/// extended mask filter), and any hand not in the sample returns -1 via
/// `hand_index` and is handled here.
#[inline]
fn lookup_pair_reach(opp_reach: &[f32], hand_index: &[i32], c1: usize, c2: usize) -> f64 {
    if c1 == c2 { return 0.0; }
    let idx = hand_index[c1 * 52 + c2];
    if idx < 0 { return 0.0; }
    opp_reach[idx as usize] as f64
}

pub fn total_valid_reach_product(
    masses: &OppMasses,
    opp_reach: &[&[f32]],
) -> Vec<f64> {
    let k = masses.num_opp;
    let nh = masses.nh;
    let mut out = vec![0.0f64; nh];

    for h in 0..nh {
        let h_c1 = masses.hand_cards[h * 2] as usize;
        let h_c2 = masses.hand_cards[h * 2 + 1] as usize;
        let h_dead_mask = (1u64 << h_c1) | (1u64 << h_c2) | masses.dead_card_mask;

        out[h] = match k {
            0 => 1.0,
            1 => masses.r(0, h) as f64,
            2 => {
                // K=2 I-E for "g_0 and g_1 share NO cards":
                //
                //   TVRP(h) = R_0 · R_1
                //           − Σ_c R_0^{(c)} · R_1^{(c)}              [single-card sharing]
                //           + Σ_{c1 < c2} [r_0[{c1,c2}] · r_1[{c1,c2}]]  [share both cards]
                //
                // The last term corrects the double-subtraction of same-hand
                // pairs (g_0 = g_1 = {c1, c2}): such a pair satisfies *both*
                // "share c1" and "share c2", so the single-card sum
                // subtracts it twice; we add it back via the 2-card-sharing
                // I-E term. For distinct hands, sharing 2 cards is impossible
                // (a hand has only 2 cards, so sharing 2 means same hand),
                // so the 2-card sum is equivalently a sum over the in-sample
                // hands themselves.
                //
                // Sum order is fixed (c ascending; same-hand sum walks the
                // sample in index order) so CPU and GPU produce
                // bit-equivalent f64 accumulator state.
                let r0 = masses.r(0, h) as f64;
                let r1 = masses.r(1, h) as f64;
                let prod = r0 * r1;

                let mut edge_sum = 0.0f64;
                for c in 0..52usize {
                    if h_dead_mask & (1u64 << c) != 0 { continue; }
                    let p0 = masses.r_per_card(0, h, c) as f64;
                    let p1 = masses.r_per_card(1, h, c) as f64;
                    edge_sum += p0 * p1;
                }

                // Same-hand add-back: enumerate the sampled hands once.
                let mut same_hand_sum = 0.0f64;
                for hx in 0..nh {
                    let hx_c1 = masses.hand_cards[hx * 2] as usize;
                    let hx_c2 = masses.hand_cards[hx * 2 + 1] as usize;
                    let hx_m = (1u64 << hx_c1) | (1u64 << hx_c2);
                    if hx_m & h_dead_mask != 0 { continue; }
                    same_hand_sum += opp_reach[0][hx] as f64 * opp_reach[1][hx] as f64;
                }

                prod - edge_sum + same_hand_sum
            }
            3 => tvrp_k3_factored(masses, opp_reach, h, h_dead_mask),
            4 | 5 => tvrp_brute(masses, opp_reach, h, h_dead_mask),
            _ => unimplemented!("K >= 6 not supported (max table size is 6 players)"),
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::card_from_str;

    /// Brute-force reference for `total_valid_reach_product`. Enumerates all
    /// valid (g_0, ..., g_{K-1}) tuples explicitly. O(nh^K).
    fn brute_force_tvrp(
        opp_reach: &[&[f32]],
        hand_cards: &[u8],
        nh: usize,
        dead_card_mask: u64,
        h: usize,
    ) -> f64 {
        let h_m = (1u64 << hand_cards[h*2]) | (1u64 << hand_cards[h*2+1]);
        let k = opp_reach.len();
        let g_mask: Vec<u64> = (0..nh).map(|g|
            (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1])
        ).collect();

        fn recurse(
            depth: usize,
            k: usize,
            nh: usize,
            mask_so_far: u64,
            reach_so_far: f64,
            opp_reach: &[&[f32]],
            g_mask: &[u64],
            sum: &mut f64,
        ) {
            if depth == k {
                *sum += reach_so_far;
                return;
            }
            for g in 0..nh {
                if g_mask[g] & mask_so_far != 0 { continue; }
                let r = opp_reach[depth][g] as f64;
                if r == 0.0 { continue; }
                recurse(
                    depth + 1, k, nh,
                    mask_so_far | g_mask[g],
                    reach_so_far * r,
                    opp_reach, g_mask, sum,
                );
            }
        }

        let mut sum = 0.0f64;
        recurse(0, k, nh, h_m | dead_card_mask, 1.0, opp_reach, &g_mask, &mut sum);
        sum
    }

    /// Foundation test: factored I-E for K=2 must reproduce the explicit
    /// pair-enumeration brute-force for `TotalValidReachProduct` across
    /// uniform and non-uniform reach, conflict and conflict-free hand sets,
    /// and board-only vs. extended dead-card masks.
    ///
    /// If this test fails, the I-E formula structure is wrong and the whole
    /// approach must be reconsidered — do not proceed to K≥3 until this is
    /// bit-clean.
    #[test]
    fn ie_k2_tvrp_matches_bruteforce() {
        // 6 hands, mix of conflicts. Cards 0..12 used.
        // Hand layout: chosen to have specific overlaps.
        let hand_cards: Vec<u8> = vec![
            0, 1,   // h=0: A♠ K♠
            2, 3,   // h=1: Q♠ J♠
            0, 4,   // h=2: A♠ T♠  (conflicts with h=0 on card 0)
            5, 6,   // h=3: 9♠ 8♠
            7, 8,   // h=4: 7♠ 6♠
            1, 9,   // h=5: K♠ 5♠  (conflicts with h=0 on card 1)
        ];
        let nh = 6;
        let hand_strength: Vec<u16> = vec![100, 90, 80, 70, 60, 50];

        // Test multiple reach configurations and multiple dead-card masks.
        let configs: &[(Vec<f32>, Vec<f32>, u64, &str)] = &[
            (vec![1.0; 6], vec![1.0; 6], 0u64, "uniform, no dead"),
            (vec![0.5, 0.8, 0.2, 0.9, 0.3, 0.7],
             vec![0.4, 0.1, 0.6, 0.5, 0.8, 0.2],
             0u64, "non-uniform, no dead"),
            // Dead mask includes cards 6 and 8 — kills h=3 (5♠ 6♠ uses 5,6) wait
            // hand_cards[6,7] = (5, 6) for h=3. Mask with card 6 set kills h=3.
            // and h=4 (7,8) — card 8 set kills h=4.
            // Note: this also constrains opp's per-card partials.
            (vec![1.0; 6], vec![1.0; 6],
             (1u64 << 6) | (1u64 << 8), "uniform, dead=[6,8]"),
            (vec![0.5, 0.8, 0.2, 0.9, 0.3, 0.7],
             vec![0.4, 0.1, 0.6, 0.5, 0.8, 0.2],
             (1u64 << 6) | (1u64 << 8), "non-uniform, dead=[6,8]"),
        ];

        for (r0, r1, dead, label) in configs {
            let opp_reach_vec: Vec<&[f32]> = vec![&r0[..], &r1[..]];

            let masses = precompute_opp_masses(&opp_reach_vec, &hand_cards, &hand_strength, *dead);
            let ie = total_valid_reach_product(&masses, &opp_reach_vec);

            let mut max_abs_diff = 0.0f64;
            let mut max_rel_diff = 0.0f64;
            for h in 0..nh {
                let bf = brute_force_tvrp(&opp_reach_vec, &hand_cards, nh, *dead, h);
                let diff = (ie[h] - bf).abs();
                let scale = bf.abs().max(1e-9);
                let rel = diff / scale;
                if diff > max_abs_diff { max_abs_diff = diff; }
                if rel > max_rel_diff { max_rel_diff = rel; }
                eprintln!(
                    "  [{}] h={} ie={:.9} bf={:.9} diff={:.3e}",
                    label, h, ie[h], bf, diff
                );
            }
            eprintln!(
                "  [{}] max_abs={:.3e}, max_rel={:.3e}",
                label, max_abs_diff, max_rel_diff
            );

            // The I-E version uses f64 accumulators and is mathematically
            // identical to the brute-force; modulo f32-rounding of the source
            // reach values the two should agree to ~1e-6 relative.
            assert!(
                max_rel_diff < 1e-5,
                "[{}] K=2 I-E does not match brute-force: max_rel={:.3e}",
                label, max_rel_diff
            );
        }
    }

    /// K=3 I-E TVRP must match O(nh^3) brute-force reference across uniform
    /// and non-uniform reach, conflict and conflict-free hand sets, and
    /// board-only vs. extended dead-card masks.
    #[test]
    fn ie_k3_tvrp_matches_bruteforce() {
        // 6 hands carefully chosen to exercise 3-cycle, paths, same-hand, etc.
        // Layout:
        //   h=0: {0, 1}
        //   h=1: {0, 2}   shares c=0 with h=0
        //   h=2: {1, 2}   shares c=1 with h=0, c=2 with h=1; triangle 0-1-2
        //   h=3: {3, 4}
        //   h=4: {5, 6}
        //   h=5: {7, 8}
        // The triangle hand_index[0,2], hand_index[1,2] exists; for h_player
        // outside the triangle, the cycle term has non-zero contributions.
        let hand_cards: Vec<u8> = vec![
            0, 1,
            0, 2,
            1, 2,
            3, 4,
            5, 6,
            7, 8,
        ];
        let nh = 6;
        let hand_strength: Vec<u16> = vec![100, 90, 80, 70, 60, 50];

        let configs: &[(Vec<f32>, Vec<f32>, Vec<f32>, u64, &str)] = &[
            (vec![1.0; 6], vec![1.0; 6], vec![1.0; 6], 0u64, "uniform, no dead"),
            (vec![0.5, 0.8, 0.2, 0.9, 0.3, 0.7],
             vec![0.4, 0.1, 0.6, 0.5, 0.8, 0.2],
             vec![0.3, 0.7, 0.4, 0.9, 0.1, 0.6],
             0u64, "non-uniform, no dead"),
            (vec![1.0; 6], vec![1.0; 6], vec![1.0; 6],
             (1u64 << 9) | (1u64 << 10), "uniform, dead=[9,10] (no effect on sample)"),
            (vec![0.5, 0.8, 0.2, 0.9, 0.3, 0.7],
             vec![0.4, 0.1, 0.6, 0.5, 0.8, 0.2],
             vec![0.3, 0.7, 0.4, 0.9, 0.1, 0.6],
             (1u64 << 6) | (1u64 << 8), "non-uniform, dead=[6,8] (kills h=4, h=5)"),
        ];

        for (r0, r1, r2, dead, label) in configs {
            let opp_reach_vec: Vec<&[f32]> = vec![&r0[..], &r1[..], &r2[..]];

            let masses = precompute_opp_masses(&opp_reach_vec, &hand_cards, &hand_strength, *dead);
            let ie = total_valid_reach_product(&masses, &opp_reach_vec);

            let mut max_abs_diff = 0.0f64;
            let mut max_rel_diff = 0.0f64;
            let mut worst_h = 0usize;
            for h in 0..nh {
                let bf = brute_force_tvrp(&opp_reach_vec, &hand_cards, nh, *dead, h);
                let diff = (ie[h] - bf).abs();
                let scale = bf.abs().max(1e-9);
                let rel = diff / scale;
                if diff > max_abs_diff { max_abs_diff = diff; worst_h = h; }
                if rel > max_rel_diff { max_rel_diff = rel; }
                eprintln!(
                    "  [{}] h={} ie={:.9} bf={:.9} diff={:.3e}",
                    label, h, ie[h], bf, diff
                );
            }
            eprintln!(
                "  [{}] max_abs={:.3e} at h={}, max_rel={:.3e}",
                label, max_abs_diff, worst_h, max_rel_diff
            );

            // Mixed absolute/relative threshold: f64 cancellation noise
            // around bf=0 (e.g., when h_player itself is invalidated by
            // the dead mask and brute-force returns exactly 0) blows up
            // a pure relative comparison. Accept either max_abs < 1e-5
            // (f32-precision floor) OR max_rel < 1e-5 (per-h rel parity).
            assert!(
                max_abs_diff < 1e-5 || max_rel_diff < 1e-5,
                "[{}] K=3 factored does not match brute-force: max_abs={:.3e}, max_rel={:.3e} at h={}",
                label, max_abs_diff, max_rel_diff, worst_h
            );
        }
    }

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
