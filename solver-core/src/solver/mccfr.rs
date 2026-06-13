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
    // QRE generalization (S0, 2026-06-13): per-seat inverse-temperature
    // λ. `None` = pure Nash (regret matching) — bit-identical to the
    // pre-QRE solver, so existing gates are untouched. `Some(λ)` =
    // QUANTAL response: the acting player's strategy is a LOGIT over the
    // action counterfactual values, σ_a ∝ exp(λ_seat · cfv_a) (per hand).
    // As λ→∞ the logit → argmax(cfv) = best-response-to-last-iterate, so
    // the averaged play is fictitious play → Nash (the reduction the S0
    // gate proves). At finite λ it is a genuine quantal response (λ
    // controls rationality), which S2 uses to model rough opponents.
    // `last_cfv` holds the previous iteration's per-(node,action,hand)
    // counterfactual value the logit responds to.
    lambda: Option<Vec<f32>>,
    last_cfv: Vec<f32>,
    // DEPTH-LIMITED SEARCH (S1, 2026-06-13): nodes flagged `frozen` play
    // a FIXED strategy (the blueprint's) and are NOT updated — the
    // searcher optimizes only the un-frozen (subgame) nodes against the
    // frozen continuation. Freezing the post-boundary street = the
    // depth-limited leaf-continuation valuation (leaf value = continue
    // with the blueprint). The frozen strategy is SWAPPABLE (fine or
    // rough blueprint) — that swappability IS the S2 measurement.
    frozen: Vec<bool>,
    frozen_strategy: Vec<f32>,
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
            regret_floor: -1e30,
            lambda: None,
            last_cfv: vec![0.0; total],
            frozen: vec![false; tree.num_nodes()],
            frozen_strategy: vec![0.0; total],
        }
    }

    /// DEPTH-LIMITED SEARCH: freeze a node to a fixed (blueprint)
    /// strategy. `strat` is `num_actions × nh` in [a*nh + h] order. The
    /// node plays this strategy, is never regret/cum-updated, and its
    /// cum_strategy is set to it so the scored profile reads the
    /// blueprint there. The un-frozen nodes are searched against it.
    pub fn freeze_node(&mut self, node_idx: usize, strat: &[f32]) {
        let off = self.node_data_offset[node_idx];
        assert_ne!(off, UNUSED, "freeze_node on a non-player node {node_idx}");
        assert!(off + strat.len() <= self.frozen_strategy.len(), "frozen strat overruns node block");
        self.frozen[node_idx] = true;
        self.frozen_strategy[off..off + strat.len()].copy_from_slice(strat);
        // Seed cum_strategy so StrategyProfile reads the blueprint here.
        self.cum_strategy[off..off + strat.len()].copy_from_slice(strat);
    }

    /// Enable QUANTAL-response (QRE) mode with per-seat inverse-
    /// temperature λ (length = num_players). High λ → Nash limit; the
    /// S0 reduction gate proves QRE-at-high-λ reproduces the Nash solve.
    pub fn set_lambda(&mut self, lambda: Vec<f32>) {
        assert_eq!(lambda.len(), self.num_players as usize, "lambda per seat");
        self.lambda = Some(lambda);
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
                    if outcome > 0 { game.clear_chance_outcome(); }
                    let probs: Vec<f32> = (0..nh_t).map(|h| game.chance_probability(outcome, h)).collect();
                    game.set_chance_outcome(outcome);
                    let child_cfv = self.walk_cfr(tree, game, traverser, child, cfreach, traverser_reach, weight);
                    for h in 0..nh_t {
                        cfv[h] += probs[h] * child_cfv[h];
                    }
                }
                game.clear_chance_outcome();
            } else {
                for (outcome, &child) in children.iter().enumerate() {
                    if outcome > 0 { game.clear_chance_outcome(); }
                    let probs: Vec<f32> = (0..nh_t).map(|h| game.chance_probability(outcome, h)).collect();
                    game.set_chance_outcome(outcome);
                    let child_cfv = self.walk_cfr(tree, game, traverser, child as usize, cfreach, traverser_reach, weight);
                    for h in 0..nh_t {
                        cfv[h] += probs[h] * child_cfv[h];
                    }
                }
                game.clear_chance_outcome();
            }
            return cfv;
        }

        let player = node.player_id;
        let num_actions = node.num_children as usize;
        let nh = self.num_hands[player as usize];
        let children = tree.node_children(node_idx);

        let strategy = self.compute_strategy(node_idx, num_actions, nh, player);

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

        if player == traverser && !self.frozen[node_idx] {
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

            // QRE: ACCUMULATE action values so the logit responds to the
            // value against the TIME-AVERAGE opponent (smooth fictitious
            // play — convergent in zero-sum), NOT the last iterate
            // (Cournot best-response dynamics — oscillates, does not
            // converge; the S0 v1 bug). cfv is linear in opponent reach,
            // so the average cfv IS the cfv against the average opponent.
            // No-op for `lambda == None` (Nash path ignores last_cfv).
            if self.lambda.is_some() {
                for h in 0..nh {
                    for a in 0..num_actions {
                        self.last_cfv[offset + a * nh + h] += cfv_all[a][h];
                    }
                }
            }
        }

        cfv_avg
    }

    fn compute_strategy(&self, node_idx: usize, num_actions: usize, nh: usize, player: u8) -> Vec<f32> {
        let offset = self.node_data_offset[node_idx];

        // Depth-limited search: a frozen node plays its fixed blueprint
        // strategy (never regret-matched, never updated).
        if self.frozen[node_idx] {
            return self.frozen_strategy[offset..offset + num_actions * nh].to_vec();
        }

        let mut strategy = vec![0.0f32; num_actions * nh];

        // QRE (quantal) mode: logit over action counterfactual values at
        // the acting seat's inverse-temperature λ. σ_a ∝ exp(λ·cfv_a),
        // numerically stabilized by subtracting the per-hand max. First
        // iterate (last_cfv all zero) → uniform, the natural QRE start.
        if let Some(lam) = &self.lambda {
            let l = lam[player as usize];
            // Respond to the AVERAGE action value (accumulated cfv ÷ count
            // of accumulation iterations = value vs the time-average
            // opponent). Iteration was incremented at run-start, and
            // last_cfv accumulates from prior iterations, so the average
            // denominator is (iteration − 1). First iterate: 0 → uniform.
            let denom = (self.iteration as f32 - 1.0).max(1.0);
            for h in 0..nh {
                let mut mx = f32::NEG_INFINITY;
                for a in 0..num_actions {
                    mx = mx.max(self.last_cfv[offset + a * nh + h] / denom);
                }
                let mut z = 0.0f32;
                for a in 0..num_actions {
                    let avg = self.last_cfv[offset + a * nh + h] / denom;
                    let w = (l * (avg - mx)).exp();
                    strategy[a * nh + h] = w;
                    z += w;
                }
                let z = if z > 0.0 { z } else { 1.0 };
                for a in 0..num_actions {
                    strategy[a * nh + h] /= z;
                }
            }
            return strategy;
        }

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
        let player = 0u8; // informational accessor; player unknown here, Nash-mode uses regrets only
        let raw = self.compute_strategy(node_idx, num_actions, nh, player);
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
