use crate::tree::flat::{FlatTree, NODE_TYPE_PLAYER};
use super::game::GameSpec;

const UNUSED: usize = usize::MAX;

pub struct StrategyProfile<'a> {
    cum_strategy: &'a [f32],
    node_offsets_u32: Option<&'a [u32]>,
    node_offsets_usize: Option<&'a [usize]>,
    nh: usize,
}

impl<'a> StrategyProfile<'a> {
    pub fn from_u32_offsets(cum_strategy: &'a [f32], node_offsets: &'a [u32], nh: usize) -> Self {
        StrategyProfile { cum_strategy, node_offsets_u32: Some(node_offsets), node_offsets_usize: None, nh }
    }

    pub fn from_usize_offsets(cum_strategy: &'a [f32], node_offsets: &'a [usize], nh: usize) -> Self {
        StrategyProfile { cum_strategy, node_offsets_u32: None, node_offsets_usize: Some(node_offsets), nh }
    }

    fn node_offset(&self, node_idx: usize) -> usize {
        if let Some(offsets) = self.node_offsets_u32 {
            offsets[node_idx] as usize
        } else if let Some(offsets) = self.node_offsets_usize {
            offsets[node_idx]
        } else {
            UNUSED
        }
    }

    fn get_strategy(&self, node_idx: usize, num_actions: usize) -> Vec<f32> {
        let offset = self.node_offset(node_idx);
        if offset == UNUSED || num_actions == 0 {
            let uniform = if num_actions > 0 { 1.0 / num_actions as f32 } else { 0.0 };
            return vec![uniform; num_actions * self.nh];
        }

        let nh = self.nh;
        let mut strategy = vec![0.0f32; num_actions * nh];
        for h in 0..nh {
            let mut total = 0.0f32;
            for a in 0..num_actions {
                total += self.cum_strategy[offset + a * nh + h];
            }
            if total > 0.0 {
                for a in 0..num_actions {
                    strategy[a * nh + h] = self.cum_strategy[offset + a * nh + h] / total;
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
}

fn walk_value(
    tree: &FlatTree,
    game: &dyn GameSpec,
    strategy: &StrategyProfile,
    target_player: u8,
    node_idx: usize,
    cfreach: &mut [Vec<f32>],
    action_selector: &dyn Fn(&[Vec<f32>], usize, &[f32]) -> Vec<f32>,
) -> Vec<f32> {
    let node = &tree.nodes[node_idx];

    if node.is_terminal() {
        return game.evaluate_terminal(target_player, node_idx, tree, cfreach);
    }

    if node.is_chance() {
        let children = tree.node_children(node_idx);
        if children.is_empty() {
            return game.evaluate_terminal(target_player, node_idx, tree, cfreach);
        }
        let nh_t = game.num_hands(target_player);
        let mut cfv = vec![0.0f32; nh_t];

        let n_outcomes = game.num_chance_outcomes();
        if n_outcomes > 0 && children.len() == 1 {
            let child = children[0] as usize;
            for outcome in 0..n_outcomes {
                game.set_chance_outcome(outcome);
                let child_cfv = walk_value(tree, game, strategy, target_player, child, cfreach, action_selector);
                for h in 0..nh_t {
                    cfv[h] += game.chance_probability(outcome, h) * child_cfv[h];
                }
            }
            game.clear_chance_outcome();
        } else {
            for (outcome, &child) in children.iter().enumerate() {
                let child_cfv = walk_value(tree, game, strategy, target_player, child as usize, cfreach, action_selector);
                for h in 0..nh_t {
                    cfv[h] += game.chance_probability(outcome, h) * child_cfv[h];
                }
            }
        }
        return cfv;
    }

    let player = node.player_id;
    let num_actions = node.num_children as usize;
    let nh = game.num_hands(player);
    let children = tree.node_children(node_idx);

    let sigma = strategy.get_strategy(node_idx, num_actions);

    let mut cfv_all: Vec<Vec<f32>> = Vec::with_capacity(num_actions);
    for (a, &child) in children.iter().enumerate() {
        let saved = cfreach[player as usize].clone();
        for h in 0..nh {
            cfreach[player as usize][h] = saved[h] * sigma[a * nh + h];
        }
        cfv_all.push(walk_value(tree, game, strategy, target_player, child as usize, cfreach, action_selector));
        cfreach[player as usize] = saved;
    }

    if player == target_player {
        action_selector(&cfv_all, nh, &sigma)
    } else {
        let nh_t = game.num_hands(target_player);
        let mut cfv = vec![0.0f32; nh_t];
        for a in 0..num_actions {
            for h in 0..nh_t {
                cfv[h] += cfv_all[a][h];
            }
        }
        cfv
    }
}

pub fn best_response_value(
    tree: &FlatTree,
    game: &dyn GameSpec,
    strategy: &StrategyProfile,
    player: u8,
) -> Vec<f32> {
    let np = tree.num_players as usize;
    let nh = game.num_hands(player);
    let mut cfreach: Vec<Vec<f32>> = (0..np)
        .map(|p| {
            if p == player as usize {
                vec![1.0f32; nh]
            } else {
                game.initial_weight(p as u8)
            }
        })
        .collect();

    walk_value(tree, game, strategy, player, 0, &mut cfreach, &select_best_action)
}

pub fn strategy_value(
    tree: &FlatTree,
    game: &dyn GameSpec,
    strategy: &StrategyProfile,
    player: u8,
) -> Vec<f32> {
    let np = tree.num_players as usize;
    let nh = game.num_hands(player);
    let mut cfreach: Vec<Vec<f32>> = (0..np)
        .map(|p| {
            if p == player as usize {
                vec![1.0f32; nh]
            } else {
                game.initial_weight(p as u8)
            }
        })
        .collect();

    walk_value(tree, game, strategy, player, 0, &mut cfreach, &select_strategy_action)
}

fn select_best_action(cfv_all: &[Vec<f32>], nh: usize, _sigma: &[f32]) -> Vec<f32> {
    let nh_t = cfv_all[0].len();
    let mut result = vec![0.0f32; nh_t];
    for h in 0..nh_t {
        let mut best = f32::NEG_INFINITY;
        for a in 0..cfv_all.len() {
            if cfv_all[a][h] > best {
                best = cfv_all[a][h];
            }
        }
        result[h] = best;
    }
    result
}

fn select_strategy_action(cfv_all: &[Vec<f32>], nh: usize, sigma: &[f32]) -> Vec<f32> {
    let nh_t = cfv_all[0].len();
    let mut result = vec![0.0f32; nh_t];
    for h in 0..nh_t {
        for (a, cfv_a) in cfv_all.iter().enumerate() {
            result[h] += sigma[a * nh + h] * cfv_a[h];
        }
    }
    result
}

pub fn exploitability(
    tree: &FlatTree,
    game: &dyn GameSpec,
    strategy: &StrategyProfile,
) -> f32 {
    let np = tree.num_players as usize;
    let mut total_exploit = 0.0f32;

    for p in 0..np {
        let br_val = best_response_value(tree, game, strategy, p as u8);
        let sv_val = strategy_value(tree, game, strategy, p as u8);
        let reach = game.initial_weight(p as u8);
        let reach_sum: f32 = reach.iter().sum();

        if reach_sum > 0.0 {
            for h in 0..br_val.len() {
                total_exploit += (reach[h] / reach_sum) * (br_val[h] - sv_val[h]);
            }
        }
    }

    total_exploit
}
