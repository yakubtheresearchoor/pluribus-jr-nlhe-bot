use std::cell::Cell;

use crate::card::{index_to_card_pair, Card};
use crate::solver::chance_table::ChanceTable;
use crate::solver::game::GameSpec;
use crate::solver::showdown::side_pot_showdown_cfv_with_rake;
use crate::tree::flat::FlatTree;

pub struct TurnStartGame {
    table: ChanceTable,
    current_river_card: Cell<Option<u8>>,
    sorted_opp_strength: Vec<u16>,
    sorted_opp_indices: Vec<u16>,
    sorted_pl_strength: Vec<u16>,
    sorted_pl_indices: Vec<u16>,
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
        }
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
        let cfv = match self.current_river_card.get() {
            Some(_card) => {
                let card = self.current_river_card.get().unwrap();
                let card_idx = card as usize;
                let stride = num_opp * nh;

                let card_opp_str = &self.table.chance_sorted_strength[card_idx * stride..card_idx * stride + stride];
                let card_opp_idx = &self.table.chance_sorted_indices[card_idx * stride..card_idx * stride + stride];

                side_pot_showdown_cfv_with_rake(
                    &opp_reach,
                    &self.table.hand_cards,
                    nh,
                    card_opp_str,
                    card_opp_idx,
                    card_opp_str,
                    card_opp_idx,
                    &contributions,
                    fold_mask,
                    traverser as usize,
                    self.table.num_players,
                    tree.starting_pot,
                    rake_rate, rake_cap, true,
                )
            }
            None => {
                side_pot_showdown_cfv_with_rake(
                    &opp_reach,
                    &self.table.hand_cards,
                    nh,
                    &self.sorted_opp_strength,
                    &self.sorted_opp_indices,
                    &self.sorted_pl_strength,
                    &self.sorted_pl_indices,
                    &contributions,
                    fold_mask,
                    traverser as usize,
                    self.table.num_players,
                    tree.starting_pot,
                    rake_rate, rake_cap, true,
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
