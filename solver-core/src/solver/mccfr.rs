use crate::tree::flat::{FlatTree, NODE_TYPE_PLAYER};
use super::game::GameSpec;

const UNUSED: usize = usize::MAX;

pub struct CpuMccfr {
    num_players: u8,
    num_hands: Vec<usize>,
    regrets: Vec<f32>,
    cum_strategy: Vec<f32>,
    node_data_offset: Vec<usize>,
    iteration: u32,
    regret_floor: f32,
}

impl CpuMccfr {
    pub fn new(tree: &FlatTree, num_hands: Vec<usize>) -> Self {
        assert_eq!(num_hands.len(), tree.num_players as usize);

        let mut offsets = Vec::with_capacity(tree.num_nodes());
        let mut total = 0usize;

        for node in &tree.nodes {
            if node.node_type == NODE_TYPE_PLAYER {
                offsets.push(total);
                total += node.num_children as usize * num_hands[node.player_id as usize];
            } else {
                offsets.push(UNUSED);
            }
        }

        CpuMccfr {
            num_players: tree.num_players,
            num_hands,
            regrets: vec![0.0; total],
            cum_strategy: vec![0.0; total],
            node_data_offset: offsets,
            iteration: 0,
            regret_floor: -1e7,
        }
    }

    pub fn run(
        &mut self,
        tree: &FlatTree,
        game: &dyn GameSpec,
        num_iterations: u32,
    ) -> Vec<f32> {
        let nh0 = self.num_hands[0];
        let np = self.num_players as usize;

        let mut root_cfv_sum = vec![0.0f32; nh0];
        let mut p0_count = 0u32;

        for _ in 0..num_iterations {
            self.iteration += 1;
            let weight = self.iteration as f32;

            let mut cfreach: Vec<Vec<f32>> = (0..np)
                .map(|p| game.initial_weight(p as u8))
                .collect();

            for traverser in 0..np {
                let mut traverser_reach = game.initial_weight(traverser as u8);
                let cfv = self.walk_cfr(tree, game, traverser as u8, 0, &mut cfreach, &mut traverser_reach, weight);

                if traverser == 0 {
                    for h in 0..nh0 {
                        root_cfv_sum[h] += cfv[h];
                    }
                    p0_count += 1;
                }
            }
        }

        for h in 0..nh0 {
            root_cfv_sum[h] /= p0_count as f32;
        }

        root_cfv_sum
    }

    fn walk_cfr(
        &mut self,
        tree: &FlatTree,
        game: &dyn GameSpec,
        traverser: u8,
        node_idx: usize,
        cfreach: &mut [Vec<f32>],
        traverser_reach: &mut [f32],
        weight: f32,
    ) -> Vec<f32> {
        let node = &tree.nodes[node_idx];

        if node.is_terminal() {
            return game.evaluate_terminal(traverser, node_idx, tree, cfreach);
        }

        if node.is_chance() {
            let children = tree.node_children(node_idx);
            if children.is_empty() {
                return game.evaluate_terminal(traverser, node_idx, tree, cfreach);
            }
            let nh_t = self.num_hands[traverser as usize];
            let mut cfv = vec![0.0f32; nh_t];

            let n_outcomes = game.num_chance_outcomes();
            if n_outcomes > 0 && children.len() == 1 {
                let child = children[0] as usize;
                for outcome in 0..n_outcomes {
                    game.set_chance_outcome(outcome);
                    let child_cfv = self.walk_cfr(tree, game, traverser, child, cfreach, traverser_reach, weight);
                    for h in 0..nh_t {
                        cfv[h] += game.chance_probability(outcome, h) * child_cfv[h];
                    }
                }
                game.clear_chance_outcome();
            } else {
                for (outcome, &child) in children.iter().enumerate() {
                    let child_cfv = self.walk_cfr(tree, game, traverser, child as usize, cfreach, traverser_reach, weight);
                    for h in 0..nh_t {
                        cfv[h] += game.chance_probability(outcome, h) * child_cfv[h];
                    }
                }
            }
            return cfv;
        }

        let player = node.player_id;
        let num_actions = node.num_children as usize;
        let nh = self.num_hands[player as usize];
        let children = tree.node_children(node_idx);

        let strategy = self.compute_strategy(node_idx, num_actions, nh);

        let mut cfv_all: Vec<Vec<f32>> = Vec::with_capacity(num_actions);

        for (a, &child) in children.iter().enumerate() {
            let saved_reach = if player == traverser {
                let saved = traverser_reach.to_vec();
                for h in 0..nh {
                    traverser_reach[h] *= strategy[a * nh + h];
                }
                Some(saved)
            } else {
                let saved = cfreach[player as usize].clone();
                for h in 0..nh {
                    cfreach[player as usize][h] *= strategy[a * nh + h];
                }
                Some(saved)
            };

            cfv_all.push(self.walk_cfr(tree, game, traverser, child as usize, cfreach, traverser_reach, weight));

            if player == traverser {
                traverser_reach.copy_from_slice(saved_reach.as_ref().unwrap());
            } else {
                cfreach[player as usize] = saved_reach.unwrap();
            }
        }

        let nh_traverser = self.num_hands[traverser as usize];
        let mut cfv_avg = vec![0.0f32; nh_traverser];
        if player == traverser {
            for h in 0..nh_traverser {
                for a in 0..num_actions {
                    cfv_avg[h] += strategy[a * nh + h] * cfv_all[a][h];
                }
            }
        } else {
            for h in 0..nh_traverser {
                for a in 0..num_actions {
                    cfv_avg[h] += cfv_all[a][h];
                }
            }
        }

        if player == traverser {
            let offset = self.node_data_offset[node_idx];
            for h in 0..nh {
                for a in 0..num_actions {
                    let idx = offset + a * nh + h;
                    let regret = cfv_all[a][h] - cfv_avg[h];
                    self.regrets[idx] += weight * regret;
                    if self.regrets[idx] < self.regret_floor {
                        self.regrets[idx] = self.regret_floor;
                    }
                }
            }

            for h in 0..nh {
                for a in 0..num_actions {
                    let idx = offset + a * nh + h;
                    self.cum_strategy[idx] += weight * traverser_reach[h] * strategy[a * nh + h];
                }
            }
        }

        cfv_avg
    }

    fn compute_strategy(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<f32> {
        let offset = self.node_data_offset[node_idx];
        let mut strategy = vec![0.0f32; num_actions * nh];

        for h in 0..nh {
            let mut pos_sum = 0.0f32;
            for a in 0..num_actions {
                let r = self.regrets[offset + a * nh + h];
                if r > 0.0 {
                    pos_sum += r;
                }
            }

            if pos_sum > 0.0 {
                for a in 0..num_actions {
                    let r = self.regrets[offset + a * nh + h];
                    strategy[a * nh + h] = if r > 0.0 { r / pos_sum } else { 0.0 };
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    strategy[a * nh + h] = uniform;
                }
            }
        }

        strategy
    }

    pub fn get_regrets(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let offset = self.node_data_offset[node_idx];
        if offset == UNUSED {
            return vec![];
        }
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for a in 0..num_actions {
            for h in 0..nh {
                result[a][h] = self.regrets[offset + a * nh + h];
            }
        }
        result
    }

    pub fn get_cum_strategy(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let offset = self.node_data_offset[node_idx];
        if offset == UNUSED {
            return vec![];
        }
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for a in 0..num_actions {
            for h in 0..nh {
                result[a][h] = self.cum_strategy[offset + a * nh + h];
            }
        }
        result
    }

    pub fn get_current_strategy(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let raw = self.compute_strategy(node_idx, num_actions, nh);
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for a in 0..num_actions {
            for h in 0..nh {
                result[a][h] = raw[a * nh + h];
            }
        }
        result
    }

    pub fn get_average_strategy(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let offset = self.node_data_offset[node_idx];
        if offset == UNUSED {
            return vec![];
        }

        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for h in 0..nh {
            let mut total = 0.0f32;
            for a in 0..num_actions {
                total += self.cum_strategy[offset + a * nh + h];
            }
            if total > 0.0 {
                for a in 0..num_actions {
                    result[a][h] = self.cum_strategy[offset + a * nh + h] / total;
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    result[a][h] = uniform;
                }
            }
        }
        result
    }

    pub fn node_offsets(&self) -> &[usize] {
        &self.node_data_offset
    }

    pub fn iteration_count(&self) -> u32 {
        self.iteration
    }

    pub fn cum_strategy_slice(&self) -> &[f32] {
        &self.cum_strategy
    }

    pub fn regrets_slice(&self) -> &[f32] {
        &self.regrets
    }
}
