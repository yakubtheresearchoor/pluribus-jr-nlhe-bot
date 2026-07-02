use std::cell::Cell;

use crate::card::{index_to_card_pair, Card};
use crate::solver::chance_table::ChanceTable;
use crate::solver::game::GameSpec;
use crate::solver::showdown::{factored_showdown_eq_cfv, side_pot_showdown_cfv_with_rake};
use crate::tree::flat::FlatTree;

pub struct TurnStartGame {
    table: ChanceTable,
    current_river_card: Cell<Option<u8>>,
    sorted_opp_strength: Vec<u16>,
    sorted_opp_indices: Vec<u16>,
    sorted_pl_strength: Vec<u16>,
    sorted_pl_indices: Vec<u16>,
    /// When set, multiway (≥2 active opponents) equal-contribution showdowns
    /// use the O(nh·2^K) FACTORED showdown instead of the O(nh^K) exact brute
    /// force — the only way a full-nh multiway turn/river re-solve fits a
    /// real-time budget (exact is ~2.5 s/terminal at live-3, ~1000 s at
    /// live-4). HU and side-pot (unequal active contributions) terminals still
    /// use the exact path. Off by default so blueprint/offline solves stay
    /// bit-exact; turned on only for the real-time multiway fallback.
    use_factored: bool,
}

impl TurnStartGame {
    pub fn new(table: ChanceTable) -> Self {
        let (sorted_opp_strength, sorted_opp_indices, sorted_pl_strength, sorted_pl_indices, _) =
            table.sorted_opp_arrays();
        TurnStartGame {
            table,
            current_river_card: Cell::new(None),
            sorted_opp_strength,
            sorted_opp_indices,
            sorted_pl_strength,
            sorted_pl_indices,
            use_factored: false,
        }
    }

    /// Enable the factored multiway showdown (real-time multiway re-solve).
    pub fn with_factored(mut self) -> Self {
        self.use_factored = true;
        self
    }

    pub fn num_valid_hands(&self) -> usize {
        self.table.num_valid
    }

    pub fn table(&self) -> &ChanceTable {
        &self.table
    }
}

impl GameSpec for TurnStartGame {
    fn num_hands(&self, _player: u8) -> usize {
        self.table.num_valid
    }

    fn initial_weight(&self, player: u8) -> Vec<f32> {
        self.table.initial_weights[player as usize].clone()
    }

    fn evaluate_terminal(
        &self,
        traverser: u8,
        node_idx: usize,
        tree: &FlatTree,
        cfreach: &[Vec<f32>],
    ) -> Vec<f32> {
        let nh = self.table.num_valid;
        let np = self.table.num_players as usize;
        let num_opp = np - 1;
        let fold_mask = tree.get_folded_mask(node_idx);

        let mut contributions = vec![0i32; np];
        for p in 0..np {
            contributions[p] = tree.get_contribution(node_idx, p as u8);
        }

        let opp_reach: Vec<&[f32]> = (0..np)
            .filter(|&p| p != traverser as usize)
            .map(|p| cfreach[p].as_slice())
            .collect();

        // Slice 1.6: rake threaded from FlatTree. flop_seen=true because
        // this is a turn-start game (flop+turn both dealt by construction).
        let rake_rate = tree.rake_rate as f32;
        let rake_cap = tree.rake_cap as f32;

        // Active opponents (non-folded). The factored multiway showdown only
        // applies when ≥2 opponents are active AND every active player matched
        // the same contribution (no side pots) — the dominant showdown shape.
        // HU and side-pot terminals keep the exact path.
        let active_opp: Vec<usize> = (0..num_opp)
            .map(|oi| if oi < traverser as usize { oi } else { oi + 1 })
            .filter(|&p| fold_mask & (1u16 << p) == 0)
            .collect();
        let active_equal_contrib = {
            // traverser is active by construction at a showdown it reaches.
            let mut all_eq = true;
            for &p in &active_opp {
                if contributions[p] != contributions[traverser as usize] {
                    all_eq = false;
                    break;
                }
            }
            all_eq
        };
        // Route every active-traverser, equal-contribution showdown with ≥1
        // active opponent through the factored path. For exactly ONE active
        // opponent (a HU showdown that arose from folds) factored is EXACT (K=1,
        // no independence approximation — it reduces to the sweep); this is the
        // case that otherwise falls through to the exact `side_pot_showdown`'s
        // general O(nh^K) brute force (the fold-path fast sweep is gated on
        // `fold_mask == 0`, so a fold makes it ~6 s/call). ≥2 active opponents is
        // the genuine factored multiway approximation.
        let use_fact = self.use_factored && active_opp.len() >= 1 && active_equal_contrib;

        // CONSTANT-PAYOFF fast path (factored). When the traverser folded, or
        // wins uncontested (≤1 player still in), its value is a constant per
        // hand (− its investment, or the pot net of rake) times the opponents'
        // counterfactual reach product. The exact showdown computes that reach
        // product by O(nh^K) brute-force pair enumeration even though the payoff
        // is constant (showdown.rs lines ~484–565) — the dominant cost of a
        // multiway re-solve. Replace it with the O(nh·K) factored reach product.
        let trav_folded = fold_mask & (1u16 << traverser) != 0;
        let num_active_total = (0..np).filter(|&p| fold_mask & (1u16 << p) == 0).count();
        if self.use_factored && num_opp >= 2 && (trav_folded || num_active_total <= 1) {
            let total_pot: i32 = tree.starting_pot + contributions.iter().sum::<i32>();
            let rake = (total_pot as f32 * rake_rate).min(rake_cap).max(0.0);
            let investment =
                tree.starting_pot as f32 / np as f32 + contributions[traverser as usize] as f32;
            let payoff = if trav_folded {
                -investment // folded loser pays no rake
            } else {
                (total_pot as f32 - rake) - investment
            };
            let rp = crate::solver::showdown::factored_total_reach_product(
                &opp_reach,
                &self.table.hand_cards,
                nh,
            );
            let nc = self.table.num_combinations as f32;
            let inv_nc = if nc > 0.0 { 1.0 / nc } else { 1.0 };
            return (0..nh).map(|h| payoff * rp[h] * inv_nc).collect();
        }

        // Compute the showdown CFV given the sorted strength/index arrays for the
        // relevant outcome (turn root vs a specific river card). Every opponent
        // shares an identical sorted array, so factored can be handed the
        // active-opponent reaches compacted to the front.
        let showdown = |o_str: &[u16], o_idx: &[u16], p_str: &[u16], p_idx: &[u16]| -> Vec<f32> {
            if use_fact {
                let active_reach: Vec<&[f32]> = active_opp
                    .iter()
                    .map(|&p| {
                        // map original player index → its slot in opp_reach
                        let oi = if p < traverser as usize { p } else { p - 1 };
                        opp_reach[oi]
                    })
                    .collect();
                let mut out = factored_showdown_eq_cfv(
                    &active_reach,
                    &self.table.hand_cards,
                    nh,
                    o_str,
                    o_idx,
                    p_str,
                    p_idx,
                    &contributions,
                    traverser as usize,
                    self.table.num_players,
                    tree.starting_pot,
                    rake_rate,
                    rake_cap,
                    true,
                );
                // FOLDED-OPPONENT MASS FACTOR (P0 fix, 2026-07-02): the exact
                // path sums over ALL opponents' hand configs — a folder still
                // contributes its compatible reach MASS as a multiplicative
                // factor. Dropping it left fold-branch showdowns ~mass× (≈10³)
                // smaller than all-live ones — terminal classes in different
                // UNITS, which poisons any solve containing folds (measured:
                // f=21,087 vs e=26,249,340 at a 2-survivor np=3 terminal).
                // Per-folder I-E card removal vs h (independent across folders
                // — the same approximation class as the factored showdown).
                let hc = &self.table.hand_cards;
                for p in 0..np {
                    if p == traverser as usize || fold_mask & (1u16 << p) == 0 {
                        continue;
                    }
                    let oi = if p < traverser as usize { p } else { p - 1 };
                    let r = opp_reach[oi];
                    let mut s = 0.0f32;
                    let mut minus = [0.0f32; 52];
                    for g in 0..nh {
                        let rr = r[g];
                        if rr != 0.0 {
                            s += rr;
                            minus[hc[g * 2] as usize] += rr;
                            minus[hc[g * 2 + 1] as usize] += rr;
                        }
                    }
                    for h in 0..nh {
                        let m = s - minus[hc[h * 2] as usize] - minus[hc[h * 2 + 1] as usize] + r[h];
                        out[h] *= m.max(0.0);
                    }
                }
                out
            } else {
                side_pot_showdown_cfv_with_rake(
                    &opp_reach,
                    &self.table.hand_cards,
                    nh,
                    o_str,
                    o_idx,
                    p_str,
                    p_idx,
                    &contributions,
                    fold_mask,
                    traverser as usize,
                    self.table.num_players,
                    tree.starting_pot,
                    rake_rate,
                    rake_cap,
                    true,
                )
            }
        };

        let cfv = match self.current_river_card.get() {
            Some(card) => {
                let card_idx = card as usize;
                let stride = num_opp * nh;
                let card_opp_str = &self.table.chance_sorted_strength[card_idx * stride..card_idx * stride + stride];
                let card_opp_idx = &self.table.chance_sorted_indices[card_idx * stride..card_idx * stride + stride];
                showdown(card_opp_str, card_opp_idx, card_opp_str, card_opp_idx)
            }
            None => showdown(
                &self.sorted_opp_strength,
                &self.sorted_opp_indices,
                &self.sorted_pl_strength,
                &self.sorted_pl_indices,
            ),
        };

        // Normalize by num_combinations
        let nc = self.table.num_combinations as f32;
        if nc > 0.0 {
            let mut result = cfv;
            for h in 0..nh { result[h] /= nc; }
            result
        } else {
            cfv
        }
    }

    fn chance_probability(&self, outcome: usize, hand: usize) -> f32 {
        let card = self.table.remaining_deck[outcome];
        let (c1, c2) = index_to_card_pair(self.table.valid_hand_indices[hand] as usize);

        if card == c1 || card == c2 {
            return 0.0;
        }

        let mut blocked = 0u32;
        for &rc in &self.table.remaining_deck {
            if rc == c1 || rc == c2 {
                blocked += 1;
            }
        }

        1.0 / (self.table.remaining_deck.len() as f32 - blocked as f32)
    }

    fn num_chance_outcomes(&self) -> usize {
        self.table.remaining_deck.len()
    }

    fn set_chance_outcome(&self, outcome: usize) {
        self.current_river_card.set(Some(self.table.remaining_deck[outcome]));
    }

    fn clear_chance_outcome(&self) {
        self.current_river_card.set(None);
    }
}
