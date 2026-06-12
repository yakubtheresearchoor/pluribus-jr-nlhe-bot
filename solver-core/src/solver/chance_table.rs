use crate::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use crate::hand::eval::Hand;

pub struct ChanceTable {
    pub hand_ranks_base: Vec<u16>,
    pub valid_hand_indices: Vec<u16>,
    pub num_valid: usize,
    pub conflict: Vec<u8>,
    pub hand_cards: Vec<u8>,
    pub remaining_deck: Vec<u8>,
    pub chance_ranks_table: Vec<u16>,
    pub chance_sorted_strength: Vec<u16>,
    pub chance_sorted_indices: Vec<u16>,
    pub initial_weights: Vec<Vec<f32>>,
    pub num_players: u8,
    pub num_combinations: f64,
}

impl ChanceTable {
    pub fn compute_turn_start(
        known_board: &[Card],
        ranges: &[Vec<f32>],
        num_players: u8,
    ) -> Self {
        assert_eq!(known_board.len(), 4, "turn start needs exactly 4 known board cards");

        let board_set: Vec<u8> = known_board.iter().map(|&c| c as u8).collect();

        let mut valid_hand_indices = Vec::new();
        for idx in 0..NUM_POSSIBLE_HANDS {
            let (c1, c2) = index_to_card_pair(idx);
            if board_set.contains(&c1) || board_set.contains(&c2) {
                continue;
            }
            valid_hand_indices.push(idx as u16);
        }
        let num_valid = valid_hand_indices.len();

        let mut hand_ranks_base = vec![0u16; num_valid];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in known_board {
                hand = hand.add_card(bc as usize);
            }
            hand_ranks_base[i] = hand.evaluate_full();
        }

        let mut conflict = vec![0u8; num_valid * num_valid];
        for i in 0..num_valid {
            let (c1a, c1b) = index_to_card_pair(valid_hand_indices[i] as usize);
            for j in 0..num_valid {
                if i == j {
                    conflict[i * num_valid + j] = 1;
                    continue;
                }
                let (c2a, c2b) = index_to_card_pair(valid_hand_indices[j] as usize);
                if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b {
                    conflict[i * num_valid + j] = 1;
                }
            }
        }

        let mut hand_cards = vec![0u8; num_valid * 2];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            hand_cards[i * 2] = c1;
            hand_cards[i * 2 + 1] = c2;
        }

        let mut remaining_deck: Vec<u8> = (0..52u8).filter(|c| !board_set.contains(c)).collect();

        let mut chance_ranks_table = vec![0u16; 52 * num_valid];
        for &river_card in &remaining_deck {
            let mut full_board = known_board.to_vec();
            full_board.push(river_card);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in &full_board {
                    hand = hand.add_card(bc as usize);
                }
                chance_ranks_table[river_card as usize * num_valid + i] = hand.evaluate_full();
            }
        }

        let num_opp = num_players as usize - 1;
        let mut chance_sorted_strength = vec![0u16; 52 * num_opp * num_valid];
        let mut chance_sorted_indices = vec![0u16; 52 * num_opp * num_valid];
        for &river_card in &remaining_deck {
            let base_off = river_card as usize * num_opp * num_valid;
            let mut items: Vec<(u16, u16)> = (0..num_valid)
                .map(|h| {
                    let rank = chance_ranks_table[river_card as usize * num_valid + h];
                    (rank + 1, h as u16)
                })
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = base_off + oi * num_valid;
                for h in 0..num_valid {
                    chance_sorted_strength[off + h] = items[h].0;
                    chance_sorted_indices[off + h] = items[h].1;
                }
            }
        }

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

        // Compute num_combinations for CFV normalization
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
        } else {
            1.0 // Multi-player: use pairwise normalization later
        };

        ChanceTable {
            hand_ranks_base,
            valid_hand_indices,
            num_valid,
            conflict,
            hand_cards,
            remaining_deck,
            chance_ranks_table,
            chance_sorted_strength,
            chance_sorted_indices,
            initial_weights,
            num_players,
            num_combinations: nc,
        }
    }

    pub fn num_valid_hands(&self) -> usize {
        self.num_valid
    }

    pub fn hand_ranks_gpu(&self) -> Vec<u16> {
        self.hand_ranks_base.clone()
    }

    pub fn conflict_gpu(&self) -> Vec<u8> {
        self.conflict.clone()
    }

    pub fn hand_cards_gpu(&self) -> Vec<u8> {
        self.hand_cards.clone()
    }

    pub fn remaining_deck_gpu(&self) -> Vec<u8> {
        self.remaining_deck.clone()
    }

    pub fn chance_ranks_gpu(&self) -> Vec<u16> {
        self.chance_ranks_table.clone()
    }

    pub fn chance_sorted_arrays_gpu(&self) -> (Vec<u16>, Vec<u16>) {
        (self.chance_sorted_strength.clone(), self.chance_sorted_indices.clone())
    }

    pub fn initial_weight_flat(&self) -> Vec<f32> {
        let nh = self.num_valid;
        let np = self.num_players as usize;
        let mut flat = Vec::with_capacity(np * nh);
        for p in 0..np {
            flat.extend_from_slice(&self.initial_weights[p]);
        }
        flat
    }

    pub fn sorted_opp_arrays(&self) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let nh = self.num_valid;
        let num_opp = self.num_players as usize - 1;

        let mut items: Vec<(u16, u16)> = (0..nh)
            .map(|h| {
                let rank = self.hand_ranks_base[h];
                (rank + 1, h as u16)
            })
            .collect();
        items.sort_by_key(|&(s, _)| s);

        let mut sorted_strength = vec![0u16; num_opp * nh];
        let mut sorted_indices = vec![0u16; num_opp * nh];
        for oi in 0..num_opp {
            for h in 0..nh {
                sorted_strength[oi * nh + h] = items[h].0;
                sorted_indices[oi * nh + h] = items[h].1;
            }
        }

        let mut player_strength = vec![0u16; nh];
        let mut player_indices = vec![0u16; nh];
        for h in 0..nh {
            player_strength[h] = items[h].0;
            player_indices[h] = items[h].1;
        }

        let same_hand_idx = vec![u16::MAX; nh];

        (sorted_strength, sorted_indices, player_strength, player_indices, same_hand_idx)
    }
}
