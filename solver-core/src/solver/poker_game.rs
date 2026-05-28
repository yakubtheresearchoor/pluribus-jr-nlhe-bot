use crate::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use crate::hand::eval::Hand;
use crate::solver::game::GameSpec;
use crate::tree::flat::FlatTree;

pub struct RiverPokerGame {
    hand_ranks: Vec<u16>,
    valid_hand_indices: Vec<u16>,
    num_valid: usize,
    conflict: Vec<u8>,
    initial_weights: Vec<Vec<f32>>,
    num_players: u8,
    sorted_opp_strength: Vec<u16>,
    sorted_opp_indices: Vec<u16>,
    sorted_pl_strength: Vec<u16>,
    sorted_pl_indices: Vec<u16>,
    hand_cards: Vec<u8>,
    num_combinations: f64,
}

impl RiverPokerGame {
    pub fn new(board: &[Card], ranges: &[Vec<f32>], num_players: u8) -> Self {
        let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();

        let mut hand_ranks = vec![0u16; NUM_POSSIBLE_HANDS];
        let mut valid_hand_indices = Vec::new();

        for idx in 0..NUM_POSSIBLE_HANDS {
            let (c1, c2) = index_to_card_pair(idx);
            if board_set.contains(&c1) || board_set.contains(&c2) {
                continue;
            }

            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in board {
                hand = hand.add_card(bc as usize);
            }

            hand_ranks[idx] = hand.evaluate();
            valid_hand_indices.push(idx as u16);
        }

        let num_valid = valid_hand_indices.len();

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

        let num_opp = num_players as usize - 1;
        let mut items: Vec<(u16, u16)> = (0..num_valid)
            .map(|h| {
                let rank = hand_ranks[valid_hand_indices[h] as usize];
                (rank + 1, h as u16)
            })
            .collect();
        items.sort_by_key(|&(s, _)| s);

        let mut sorted_opp_strength = vec![0u16; num_opp * num_valid];
        let mut sorted_opp_indices = vec![0u16; num_opp * num_valid];
        for oi in 0..num_opp {
            for h in 0..num_valid {
                sorted_opp_strength[oi * num_valid + h] = items[h].0;
                sorted_opp_indices[oi * num_valid + h] = items[h].1;
            }
        }

        let mut sorted_pl_strength = vec![0u16; num_valid];
        let mut sorted_pl_indices = vec![0u16; num_valid];
        for h in 0..num_valid {
            sorted_pl_strength[h] = items[h].0;
            sorted_pl_indices[h] = items[h].1;
        }

        let mut hand_cards = vec![0u8; num_valid * 2];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            hand_cards[i * 2] = c1;
            hand_cards[i * 2 + 1] = c2;
        }

        // Compute num_combinations: sum of w0[h0] × w1[h1] for non-conflicting pairs
        let nc = if num_players == 2 {
            let w0 = &initial_weights[0];
            let w1 = &initial_weights[1];
            let mut nc = 0.0f64;
            for h0 in 0..num_valid {
                let (c1a, c2a) = index_to_card_pair(valid_hand_indices[h0] as usize);
                let mask0: u64 = (1u64 << c1a) | (1u64 << c2a);
                for h1 in 0..num_valid {
                    if mask0 & ((1u64 << hand_cards[h1 * 2]) | (1u64 << hand_cards[h1 * 2 + 1])) == 0 {
                        nc += w0[h0] as f64 * w1[h1] as f64;
                    }
                }
            }
            nc
        } else {
            // For N > 2, approximate with pairwise products
            let w0 = &initial_weights[0];
            let w1 = if initial_weights.len() > 1 { &initial_weights[1] } else { &initial_weights[0] };
            let mut nc = 0.0f64;
            for h0 in 0..num_valid {
                for h1 in 0..w1.len().min(num_valid) {
                    if conflict[h0 * num_valid + h1] == 0 {
                        nc += w0[h0] as f64 * w1[h1] as f64;
                    }
                }
            }
            nc
        };

        RiverPokerGame {
            hand_ranks,
            valid_hand_indices,
            num_valid,
            conflict,
            initial_weights,
            num_players,
            sorted_opp_strength,
            sorted_opp_indices,
            sorted_pl_strength,
            sorted_pl_indices,
            hand_cards,
            num_combinations: nc,
        }
    }

    pub fn num_valid_hands(&self) -> usize {
        self.num_valid
    }

    pub fn valid_hand_indices(&self) -> &[u16] {
        &self.valid_hand_indices
    }

    pub fn sign(&self, h: usize, ho: usize) -> f32 {
        let hr = self.hand_ranks[self.valid_hand_indices[h] as usize];
        let hro = self.hand_ranks[self.valid_hand_indices[ho] as usize];
        if hr > hro {
            1.0
        } else if hr < hro {
            -1.0
        } else {
            0.0
        }
    }

    pub fn hand_ranks_gpu(&self) -> Vec<u16> {
        let mut ranks = vec![0u16; self.num_valid];
        for (i, &hi) in self.valid_hand_indices.iter().enumerate() {
            ranks[i] = self.hand_ranks[hi as usize];
        }
        ranks
    }

    pub fn conflict_gpu(&self) -> Vec<u8> {
        self.conflict.clone()
    }

    pub fn sorted_opp_arrays(&self) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let same_hand_idx = vec![u16::MAX; self.num_valid];
        (
            self.sorted_opp_strength.clone(),
            self.sorted_opp_indices.clone(),
            self.sorted_pl_strength.clone(),
            self.sorted_pl_indices.clone(),
            same_hand_idx,
        )
    }

    pub fn hand_cards_gpu(&self) -> Vec<u8> {
        self.hand_cards.clone()
    }

    pub fn initial_weight_flat(&self, _ranges: &[Vec<f32>]) -> Vec<f32> {
        let nh = self.num_valid;
        let np = self.num_players as usize;
        let mut flat = Vec::with_capacity(np * nh);
        for p in 0..np {
            flat.extend_from_slice(&self.initial_weights[p]);
        }
        flat
    }
}

impl GameSpec for RiverPokerGame {
    fn num_hands(&self, _player: u8) -> usize {
        self.num_valid
    }

    fn initial_weight(&self, player: u8) -> Vec<f32> {
        self.initial_weights[player as usize].clone()
    }

    fn evaluate_terminal(
        &self,
        traverser: u8,
        node_idx: usize,
        tree: &FlatTree,
        cfreach: &[Vec<f32>],
    ) -> Vec<f32> {
        let nh = self.num_valid;
        let np = self.num_players as usize;
        let fold_mask = tree.get_folded_mask(node_idx);

        let mut contributions = vec![0i32; np];
        for p in 0..np {
            contributions[p] = tree.get_contribution(node_idx, p as u8);
        }

        let opp_reach: Vec<&[f32]> = (0..np)
            .filter(|&p| p != traverser as usize)
            .map(|p| cfreach[p].as_slice())
            .collect();

        let mut cfv = crate::solver::showdown::side_pot_showdown_cfv(
            &opp_reach,
            &self.hand_cards,
            nh,
            &self.sorted_opp_strength,
            &self.sorted_opp_indices,
            &self.sorted_pl_strength,
            &self.sorted_pl_indices,
            &contributions,
            fold_mask,
            traverser as usize,
            self.num_players,
            tree.starting_pot,
        );

        // Normalize CFV by num_combinations (matching external solver convention)
        let nc = self.num_combinations as f32;
        if nc > 0.0 {
            for h in 0..nh {
                cfv[h] /= nc;
            }
        }
        cfv
    }

    fn chance_probability(&self, _outcome: usize, _hand: usize) -> f32 {
        0.0
    }

    fn num_combinations(&self) -> f64 {
        self.num_combinations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::card_from_str;

    fn uniform_ranges(num_players: usize) -> Vec<Vec<f32>> {
        let total = (NUM_POSSIBLE_HANDS * (NUM_POSSIBLE_HANDS - 1)) as f32;
        vec![vec![1.0 / total; NUM_POSSIBLE_HANDS]; num_players]
    }

    #[test]
    fn test_river_game_basic() {
        let board: Vec<Card> = ["2c", "3d", "4h", "5s", "6c"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect();

        let game = RiverPokerGame::new(&board, &uniform_ranges(2), 2);

        assert!(game.num_valid > 0);
        assert!(game.num_valid < NUM_POSSIBLE_HANDS);

        // Best hand should be 7x (makes straight)
        // Worst should be something like 8x offsuit that doesn't connect
        let mut best_rank = 0u16;
        let mut worst_rank = u16::MAX;
        for i in 0..game.num_valid {
            let r = game.hand_ranks[game.valid_hand_indices[i] as usize];
            if r > best_rank { best_rank = r; }
            if r < worst_rank { worst_rank = r; }
        }
        assert!(best_rank > worst_rank, "should have hand rank variation");
    }

    #[test]
    fn test_sign_table() {
        let board: Vec<Card> = ["2h", "3h", "4h", "5c", "6d"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect();

        let game = RiverPokerGame::new(&board, &uniform_ranges(2), 2);

        let flush_h = game.valid_hand_indices.iter().position(|&hi| {
            let (c1, c2) = index_to_card_pair(hi as usize);
            c1 % 4 == 2 && c2 % 4 == 2
        });

        let high_pair_h = game.valid_hand_indices.iter().position(|&hi| {
            let (c1, c2) = index_to_card_pair(hi as usize);
            c1 / 4 == c2 / 4 && c1 / 4 >= 10
        });

        if let (Some(fh), Some(ph)) = (flush_h, high_pair_h) {
            assert!(
                game.hand_ranks[game.valid_hand_indices[fh] as usize]
                    > game.hand_ranks[game.valid_hand_indices[ph] as usize],
                "flush ({}) should beat pair ({}) on this board",
                game.hand_ranks[game.valid_hand_indices[fh] as usize],
                game.hand_ranks[game.valid_hand_indices[ph] as usize]
            );
            assert_eq!(game.sign(fh, ph), 1.0);
            assert_eq!(game.sign(ph, fh), -1.0);
        }
    }
}
