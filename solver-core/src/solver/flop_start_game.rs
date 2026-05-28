use std::cell::Cell;

use crate::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use crate::hand::eval::Hand;
use crate::solver::chance_table::ChanceTable;
use crate::solver::game::GameSpec;
use crate::solver::showdown::side_pot_showdown_cfv;
use crate::tree::flat::FlatTree;

/// Game specification for flop-start games (2 chance transitions: turn + river).
/// Tracks both current turn and river cards for terminal evaluation.
pub struct FlopStartGame {
    table: FlopChanceTable,
    current_turn_card: Cell<Option<u8>>,
    current_river_card: Cell<Option<u8>>,
    // Base sorted arrays (flop-only hand ranks, for flop-level decisions)
    base_sorted_opp_str: Vec<u16>,
    base_sorted_opp_idx: Vec<u16>,
    base_sorted_pl_str: Vec<u16>,
    base_sorted_pl_idx: Vec<u16>,
}

/// Chance table for flop-start games.
/// Contains hand rank data for all (turn, river) combinations.
pub struct FlopChanceTable {
    pub hand_ranks_base: Vec<u16>,       // ranks with just flop (nh)
    pub valid_hand_indices: Vec<u16>,
    pub num_valid: usize,
    pub conflict: Vec<u8>,
    pub hand_cards: Vec<u8>,
    pub remaining_deck: Vec<u8>,          // 52 - 3 = 49 cards (turn candidates)
    pub turn_ranks: Vec<u16>,             // [52 * nh] ranks with (flop + turn)
    pub turn_sorted_str: Vec<u16>,        // [52 * num_opp * nh] sorted str per turn card
    pub turn_sorted_idx: Vec<u16>,        // [52 * num_opp * nh] sorted idx per turn card
    pub river_ranks: Vec<u16>,            // [52 * 52 * nh] ranks with (flop + turn + river)
    pub river_sorted_str: Vec<u16>,       // [52 * 52 * num_opp * nh] sorted str per (turn,river)
    pub river_sorted_idx: Vec<u16>,       // [52 * 52 * num_opp * nh] sorted idx per (turn,river)
    pub initial_weights: Vec<Vec<f32>>,
    pub num_players: u8,
    pub num_combinations: f64,
    // For each turn card, the remaining deck for river
    pub river_decks: Vec<Vec<u8>>,        // [52] -> Vec<u8> of remaining river cards
}

impl FlopChanceTable {
    pub fn compute_flop_start(
        known_board: &[Card],
        ranges: &[Vec<f32>],
        num_players: u8,
    ) -> Self {
        assert_eq!(known_board.len(), 3, "flop start needs exactly 3 known board cards");

        let board_set: Vec<u8> = known_board.iter().map(|&c| c as u8).collect();
        let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

        // Valid hands (don't conflict with flop)
        let mut valid_hand_indices = Vec::new();
        for idx in 0..NUM_POSSIBLE_HANDS {
            let (c1, c2) = index_to_card_pair(idx);
            if board_set.contains(&c1) || board_set.contains(&c2) { continue; }
            valid_hand_indices.push(idx as u16);
        }
        let num_valid = valid_hand_indices.len();

        // Base hand ranks (flop only) — use evaluate_internal for sorting only
        // NOTE: With 5 cards, evaluate() binary search fails; use evaluate_internal() directly
        let mut hand_ranks_base = vec![0u16; num_valid];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in known_board { hand = hand.add_card(bc as usize); }
            hand_ranks_base[i] = hand.evaluate_internal() as u16;
        }

        // Conflict matrix
        let mut conflict = vec![0u8; num_valid * num_valid];
        for i in 0..num_valid {
            let (c1a, c1b) = index_to_card_pair(valid_hand_indices[i] as usize);
            for j in 0..num_valid {
                if i == j { conflict[i * num_valid + j] = 1; continue; }
                let (c2a, c2b) = index_to_card_pair(valid_hand_indices[j] as usize);
                if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b {
                    conflict[i * num_valid + j] = 1;
                }
            }
        }

        // Hand cards
        let mut hand_cards = vec![0u8; num_valid * 2];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            hand_cards[i * 2] = c1;
            hand_cards[i * 2 + 1] = c2;
        }

        // Remaining deck (turn candidates): 52 - 3 = 49 cards
        let remaining_deck: Vec<u8> = (0..52u8).filter(|c| !board_set.contains(c)).collect();

        let num_opp = num_players as usize - 1;
        let nh = num_valid;

        // Turn ranks: for each possible turn card, hand rank with (flop + turn)
        let mut turn_ranks = vec![0u16; 52 * nh];
        // Turn sorted arrays
        let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
        let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];

        // River ranks: for each (turn, river) pair, hand rank with full board
        let mut river_ranks = vec![0u16; 52 * 52 * nh];
        let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
        let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];

        // River decks per turn card
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];

        for &turn_card in &remaining_deck {
            // Compute turn ranks
            let turn_mask = board_mask | (1u64 << turn_card);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 {
                    turn_ranks[turn_card as usize * nh + i] = 0; // hand conflicts with turn
                    continue;
                }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in known_board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(turn_card as usize);
                turn_ranks[turn_card as usize * nh + i] = hand.evaluate_internal() as u16;
            }

            // Sort turn ranks
            {
                let mut items: Vec<(u16, u16)> = (0..nh)
                    .map(|h| (turn_ranks[turn_card as usize * nh + h] + 1, h as u16))
                    .collect();
                items.sort_by_key(|&(s, _)| s);
                for oi in 0..num_opp {
                    let off = turn_card as usize * num_opp * nh + oi * nh;
                    for h in 0..nh {
                        turn_sorted_str[off + h] = items[h].0;
                        turn_sorted_idx[off + h] = items[h].1;
                    }
                }
            }

            // River cards for this turn
            let turn_plus_board: Vec<u8> = known_board.iter().chain(std::iter::once(&turn_card)).cloned().collect();
            let river_cards: Vec<u8> = (0..52u8).filter(|c| !turn_plus_board.contains(c)).collect();
            river_decks[turn_card as usize] = river_cards.clone();

            for &river_card in &river_cards {
                // Full board ranks
                for (i, &hi) in valid_hand_indices.iter().enumerate() {
                    let (c1, c2) = index_to_card_pair(hi as usize);
                    let full_mask = turn_mask | (1u64 << river_card);
                    if full_mask & (1u64 << c1) != 0 || full_mask & (1u64 << c2) != 0 {
                        river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + i] = 0;
                        continue;
                    }
                    let mut hand = Hand::new();
                    hand = hand.add_card(c1 as usize);
                    hand = hand.add_card(c2 as usize);
                    for &bc in known_board { hand = hand.add_card(bc as usize); }
                    hand = hand.add_card(turn_card as usize);
                    hand = hand.add_card(river_card as usize);
                    river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + i] = hand.evaluate_internal() as u16;
                }

                // Sort river ranks for this (turn, river) pair
                {
                    let mut items: Vec<(u16, u16)> = (0..nh)
                        .map(|h| {
                            let r = river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + h];
                            (r + 1, h as u16)
                        })
                        .collect();
                    items.sort_by_key(|&(s, _)| s);
                    for oi in 0..num_opp {
                        let off = turn_card as usize * 52 * num_opp * nh
                            + river_card as usize * num_opp * nh + oi * nh;
                        for h in 0..nh {
                            river_sorted_str[off + h] = items[h].0;
                            river_sorted_idx[off + h] = items[h].1;
                        }
                    }
                }
            }
        }

        // Initial weights
        let mut initial_weights = Vec::with_capacity(num_players as usize);
        for p in 0..num_players as usize {
            let mut w = vec![0.0f32; num_valid];
            for (i, &hand_idx) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hand_idx as usize);
                let pair_idx = if c1 < c2 {
                    c1 as usize * (101 - c1 as usize) / 2 + c2 as usize - 1
                } else {
                    c2 as usize * (101 - c2 as usize) / 2 + c1 as usize - 1
                };
                w[i] = ranges[p][pair_idx];
            }
            initial_weights.push(w);
        }

        // num_combinations
        let nc = if num_players == 2 {
            let w0 = &initial_weights[0];
            let w1 = &initial_weights[1];
            let mut nc = 0.0f64;
            for h0 in 0..num_valid {
                let mask0: u64 = (1u64 << hand_cards[h0 * 2]) | (1u64 << hand_cards[h0 * 2 + 1]);
                for h1 in 0..num_valid {
                    let mask1: u64 = (1u64 << hand_cards[h1 * 2]) | (1u64 << hand_cards[h1 * 2 + 1]);
                    if mask0 & mask1 == 0 {
                        nc += w0[h0] as f64 * w1[h1] as f64;
                    }
                }
            }
            nc
        } else { 1.0 };

        FlopChanceTable {
            hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
            remaining_deck, turn_ranks, turn_sorted_str, turn_sorted_idx,
            river_ranks, river_sorted_str, river_sorted_idx,
            initial_weights, num_players, num_combinations: nc,
            river_decks,
        }
    }

    pub fn num_valid_hands(&self) -> usize { self.num_valid }
    pub fn hand_cards_gpu(&self) -> Vec<u8> { self.hand_cards.clone() }

    pub fn initial_weight_flat(&self) -> Vec<f32> {
        let nh = self.num_valid;
        let np = self.num_players as usize;
        let mut flat = Vec::with_capacity(np * nh);
        for p in 0..np { flat.extend_from_slice(&self.initial_weights[p]); }
        flat
    }

    pub fn sorted_opp_arrays_base(&self) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let nh = self.num_valid;
        let num_opp = self.num_players as usize - 1;
        let mut items: Vec<(u16, u16)> = (0..nh)
            .map(|h| (self.hand_ranks_base[h] + 1, h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        let mut s_str = vec![0u16; num_opp * nh];
        let mut s_idx = vec![0u16; num_opp * nh];
        for oi in 0..num_opp {
            for h in 0..nh {
                s_str[oi * nh + h] = items[h].0;
                s_idx[oi * nh + h] = items[h].1;
            }
        }
        let mut p_str = vec![0u16; nh];
        let mut p_idx = vec![0u16; nh];
        for h in 0..nh { p_str[h] = items[h].0; p_idx[h] = items[h].1; }
        (s_str, s_idx, p_str, p_idx, vec![u16::MAX; nh])
    }

    /// Get turn-card sorted arrays (for turn-level decisions after turn is dealt).
    /// Returns (opp_str, opp_idx, pl_str, pl_idx) for the given turn card.
    pub fn turn_sorted_arrays(&self, turn_card: u8) -> ( &[u16], &[u16], &[u16], &[u16] ) {
        let nh = self.num_valid;
        let num_opp = self.num_players as usize - 1;
        let base = turn_card as usize * num_opp * nh;
        let s_str = &self.turn_sorted_str[base..base + num_opp * nh];
        let s_idx = &self.turn_sorted_idx[base..base + num_opp * nh];
        // Player sorted = same as opponent for 2-player
        (s_str, s_idx, s_str, s_idx)
    }

    /// Get river-card sorted arrays for terminals after both turn and river are dealt.
    pub fn river_sorted_arrays(&self, turn_card: u8, river_card: u8) -> ( &[u16], &[u16], &[u16], &[u16] ) {
        let nh = self.num_valid;
        let num_opp = self.num_players as usize - 1;
        let base = turn_card as usize * 52 * num_opp * nh + river_card as usize * num_opp * nh;
        let s_str = &self.river_sorted_str[base..base + num_opp * nh];
        let s_idx = &self.river_sorted_idx[base..base + num_opp * nh];
        (s_str, s_idx, s_str, s_idx)
    }

    /// Turn chance outcomes (for the first chance transition: flop → turn).
    /// Returns the list of possible turn cards.
    pub fn turn_outcomes(&self) -> &[u8] {
        &self.remaining_deck
    }

    /// River chance outcomes for a given turn card (for the second chance transition: turn → river).
    pub fn river_outcomes(&self, turn_card: u8) -> &[u8] {
        &self.river_decks[turn_card as usize]
    }

    pub fn chance_probability_turn(&self, outcome: usize, hand: usize) -> f32 {
        let card = self.remaining_deck[outcome];
        let (c1, c2) = index_to_card_pair(self.valid_hand_indices[hand] as usize);
        if card == c1 || card == c2 { return 0.0; }
        let blocked = self.remaining_deck.iter().filter(|&&rc| rc == c1 || rc == c2).count();
        1.0 / (self.remaining_deck.len() as f32 - blocked as f32)
    }

    pub fn chance_probability_river(&self, turn_card: u8, outcome: usize, hand: usize) -> f32 {
        let river_deck = &self.river_decks[turn_card as usize];
        let card = river_deck[outcome];
        let (c1, c2) = index_to_card_pair(self.valid_hand_indices[hand] as usize);
        if card == c1 || card == c2 { return 0.0; }
        let blocked = river_deck.iter().filter(|&&rc| rc == c1 || rc == c2).count();
        1.0 / (river_deck.len() as f32 - blocked as f32)
    }
}

impl FlopStartGame {
    pub fn new(table: FlopChanceTable) -> Self {
        let (s_str, s_idx, p_str, p_idx, _) = table.sorted_opp_arrays_base();
        FlopStartGame {
            table,
            current_turn_card: Cell::new(None),
            current_river_card: Cell::new(None),
            base_sorted_opp_str: s_str,
            base_sorted_opp_idx: s_idx,
            base_sorted_pl_str: p_str,
            base_sorted_pl_idx: p_idx,
        }
    }

    pub fn table(&self) -> &FlopChanceTable { &self.table }

    pub fn set_turn_card(&self, card: u8) {
        self.current_turn_card.set(Some(card));
    }

    pub fn set_river_card(&self, card: u8) {
        self.current_river_card.set(Some(card));
    }

    pub fn clear_chance(&self) {
        self.current_turn_card.set(None);
        self.current_river_card.set(None);
    }
}

impl GameSpec for FlopStartGame {
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

        // Select sorted arrays based on current turn + river cards
        let cfv = match (self.current_turn_card.get(), self.current_river_card.get()) {
            (Some(tc), Some(rc)) => {
                // Full board: use (turn, river) sorted arrays
                let (os, oi, ps, pi) = self.table.river_sorted_arrays(tc, rc);
                side_pot_showdown_cfv(
                    &opp_reach, &self.table.hand_cards, nh,
                    os, oi, ps, pi,
                    &contributions, fold_mask, traverser as usize, self.table.num_players,
                    tree.starting_pot,
                )
            }
            (Some(tc), None) => {
                // Turn only (shouldn't happen at terminals, but handle gracefully)
                let (os, oi, ps, pi) = self.table.turn_sorted_arrays(tc);
                side_pot_showdown_cfv(
                    &opp_reach, &self.table.hand_cards, nh,
                    os, oi, ps, pi,
                    &contributions, fold_mask, traverser as usize, self.table.num_players,
                    tree.starting_pot,
                )
            }
            (None, _) => {
                // No chance dealt yet (shouldn't happen at terminals)
                side_pot_showdown_cfv(
                    &opp_reach, &self.table.hand_cards, nh,
                    &self.base_sorted_opp_str, &self.base_sorted_opp_idx,
                    &self.base_sorted_pl_str, &self.base_sorted_pl_idx,
                    &contributions, fold_mask, traverser as usize, self.table.num_players,
                    tree.starting_pot,
                )
            }
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
        match self.current_turn_card.get() {
            Some(tc) => self.table.chance_probability_river(tc, outcome, hand),
            None => self.table.chance_probability_turn(outcome, hand),
        }
    }

    fn num_chance_outcomes(&self) -> usize {
        match self.current_turn_card.get() {
            Some(tc) => self.table.river_decks[tc as usize].len(),
            None => self.table.remaining_deck.len(),
        }
    }

    fn set_chance_outcome(&self, outcome: usize) {
        match self.current_turn_card.get() {
            Some(tc) => {
                // Setting river card for the current turn
                let deck = self.table.river_outcomes(tc);
                let card = deck[outcome];
                self.current_river_card.set(Some(card));
            }
            None => {
                // Setting turn card
                let card = self.table.remaining_deck[outcome];
                self.current_turn_card.set(Some(card));
            }
        }
    }

    fn clear_chance_outcome(&self) {
        if self.current_river_card.get().is_some() {
            self.current_river_card.set(None);
        } else {
            self.current_turn_card.set(None);
        }
    }

    fn num_combinations(&self) -> f64 {
        self.table.num_combinations
    }
}
