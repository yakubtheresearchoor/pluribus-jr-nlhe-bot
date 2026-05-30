use crate::tree::flat::{FlatTree, MAX_NA, VCFR_NO_INFOSET};
use super::game::GameSpec;

const UNUSED: usize = usize::MAX;

struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DiscountParams {
    fn new(current_iteration: u32) -> Self {
        let nearest_lower_power_of_4 = match current_iteration {
            0 => 0u32,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };
        let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
        let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;
        let pow_alpha = t_alpha * t_alpha.sqrt();
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);
        Self {
            alpha_t: (pow_alpha / (pow_alpha + 1.0)) as f32,
            beta_t: 0.5,
            gamma_t: pow_gamma as f32,
        }
    }
}

pub struct VectorCfr {
    num_players: u8,
    nh: usize,
    regrets: Vec<f32>,
    strategy: Vec<f32>,
    cum_strategy: Vec<f32>,
    node_data_offset: Vec<usize>,
    num_infosets: usize,
    iteration: u32,
    regret_floor: f32,
}

impl VectorCfr {
    pub fn new(tree: &FlatTree, num_hands: Vec<usize>) -> Self {
        assert_eq!(num_hands.len(), tree.num_players as usize);
        let nh = num_hands[0];
        for &n in &num_hands {
            assert_eq!(n, nh, "vector CFR requires uniform num_hands across players");
        }

        let num_infosets = tree.num_infosets as usize;
        let data_per_infoset = MAX_NA * nh;

        let mut node_data_offset = vec![UNUSED; tree.num_nodes()];
        for &node_id in &tree.decision_node_ids {
            let infoset_id = tree.infoset_offsets[node_id as usize];
            assert_ne!(infoset_id, VCFR_NO_INFOSET);
            node_data_offset[node_id as usize] = infoset_id as usize * data_per_infoset;
        }

        VectorCfr {
            num_players: tree.num_players,
            nh,
            regrets: vec![0.0; num_infosets * data_per_infoset],
            strategy: vec![0.0; num_infosets * data_per_infoset],
            cum_strategy: vec![0.0; num_infosets * data_per_infoset],
            node_data_offset,
            num_infosets,
            iteration: 0,
            regret_floor: -1e30,
        }
    }

    fn get_strategy_slice(&self, node_idx: usize) -> Option<&[f32]> {
        let offset = self.node_data_offset[node_idx];
        if offset == UNUSED {
            return None;
        }
        Some(&self.strategy[offset..offset + MAX_NA * self.nh])
    }

    fn get_strategy_at(&self, node_idx: usize, num_actions: usize) -> Vec<f32> {
        match self.get_strategy_slice(node_idx) {
            Some(s) => s[..num_actions * self.nh].to_vec(),
            None => vec![1.0 / num_actions as f32; num_actions * self.nh],
        }
    }

    fn compute_all_strategies(&mut self, tree: &FlatTree) {
        let nh = self.nh;
        for &node_id in &tree.decision_node_ids {
            let node = &tree.nodes[node_id as usize];
            let na = node.num_children as usize;
            let offset = self.node_data_offset[node_id as usize];

            for h in 0..nh {
                let mut pos_sum = 0.0f32;
                for a in 0..na {
                    let r = self.regrets[offset + a * nh + h];
                    if r > 0.0 {
                        pos_sum += r;
                    }
                }
                if pos_sum > 0.0 {
                    for a in 0..na {
                        let r = self.regrets[offset + a * nh + h];
                        self.strategy[offset + a * nh + h] = if r > 0.0 { r / pos_sum } else { 0.0 };
                    }
                } else {
                    let uniform = 1.0 / na as f32;
                    for a in 0..na {
                        self.strategy[offset + a * nh + h] = uniform;
                    }
                }
            }
        }
    }

    fn discount_regrets(&mut self, params: &DiscountParams) {
        for r in &mut self.regrets {
            let coef = if *r >= 0.0 { params.alpha_t } else { params.beta_t };
            *r *= coef;
        }
    }

    fn compute_reach(&self, tree: &FlatTree, game: &dyn GameSpec) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = self.nh;
        let num_nodes = tree.num_nodes();
        let mut reach = vec![0.0f32; num_nodes * np * nh];

        for p in 0..np {
            let w = game.initial_weight(p as u8);
            for h in 0..nh {
                reach[p * nh + h] = w[h];
            }
        }

        for level in 0..=tree.max_depth {
            let nodes = tree.nodes_at_level(level);
            for &node_id in nodes {
                let node_id = node_id as usize;
                let node = &tree.nodes[node_id];

                if node.is_player() {
                    let player = node.player_id as usize;
                    let na = node.num_children as usize;
                    let children = tree.node_children(node_id);
                    let sigma = self.get_strategy_at(node_id, na);

                    for (a, &child) in children.iter().enumerate() {
                        let child = child as usize;
                        let src_base = node_id * np * nh;
                        let dst_base = child * np * nh;

                        for p in 0..np {
                            for h in 0..nh {
                                reach[dst_base + p * nh + h] = reach[src_base + p * nh + h];
                            }
                        }
                        for h in 0..nh {
                            reach[dst_base + player * nh + h] *= sigma[a * nh + h];
                        }
                    }
                } else {
                    let children = tree.node_children(node_id);
                    for &child in children {
                        let child = child as usize;
                        let src_base = node_id * np * nh;
                        let dst_base = child * np * nh;
                        for p in 0..np {
                            for h in 0..nh {
                                reach[dst_base + p * nh + h] = reach[src_base + p * nh + h];
                            }
                        }
                    }
                }
            }
        }

        reach
    }

    fn bottom_up_recursive(
        &mut self,
        tree: &FlatTree,
        game: &dyn GameSpec,
        traverser: usize,
        node_idx: usize,
        reach: &[f32],
        alpha_t: f32,
        beta_t: f32,
        gamma: f32,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = self.nh;
        let node = &tree.nodes[node_idx];

        if node.is_terminal() {
            let reach_base = node_idx * np * nh;
            let cfreach: Vec<Vec<f32>> = (0..np)
                .map(|p| (0..nh).map(|h| reach[reach_base + p * nh + h]).collect())
                .collect();
            return game.evaluate_terminal(traverser as u8, node_idx, tree, &cfreach);
        }

        if node.is_chance() {
            let children = tree.node_children(node_idx);
            let mut cfv = vec![0.0f32; nh];
            let n_outcomes = game.num_chance_outcomes();

            if n_outcomes > 0 && children.len() == 1 {
            let child = children[0] as usize;
            for outcome in 0..n_outcomes {
                // Clear previous outcome state first so chance_probability
                // dispatches correctly for nested chance games.
                if outcome > 0 { game.clear_chance_outcome(); }
                // Compute probabilities BEFORE setting the new outcome,
                // using the clean state for correct dispatch.
                let probs: Vec<f32> = (0..nh).map(|h| game.chance_probability(outcome, h)).collect();
                game.set_chance_outcome(outcome);
                let child_cfv = self.bottom_up_recursive(tree, game, traverser, child, reach, alpha_t, beta_t, gamma);
                    for h in 0..nh {
                        cfv[h] += probs[h] * child_cfv[h];
                    }
                }
                game.clear_chance_outcome();
            } else {
            for (outcome, &child) in children.iter().enumerate() {
                if outcome > 0 { game.clear_chance_outcome(); }
                let probs: Vec<f32> = (0..nh).map(|h| game.chance_probability(outcome, h)).collect();
                game.set_chance_outcome(outcome);
                let child_cfv = self.bottom_up_recursive(tree, game, traverser, child as usize, reach, alpha_t, beta_t, gamma);
                    for h in 0..nh {
                        cfv[h] += probs[h] * child_cfv[h];
                    }
                }
                game.clear_chance_outcome();
            }
            return cfv;
        }

        let owner = node.player_id as usize;
        let na = node.num_children as usize;
        let children = tree.node_children(node_idx);
        let sigma = self.get_strategy_at(node_idx, na);

        let mut cfv_all: Vec<Vec<f32>> = Vec::with_capacity(na);
        for &child in children {
            cfv_all.push(self.bottom_up_recursive(tree, game, traverser, child as usize, reach, alpha_t, beta_t, gamma));
        }

        // CRITICAL: At opponent nodes, SUM child CFVs (do NOT weight by strategy).
        // The pre-computed reach at each child already includes the opponent's
        // strategy contribution from the top-down reach pass. Weighting by
        // strategy again would double-count. At traverser nodes, the strategy
        // weighting is correct because the traverser's own reach does NOT
        // include this node's strategy — it only includes strategies from
        // ancestral nodes.
        let mut cfv_avg = vec![0.0f32; nh];
        if owner == traverser {
            for a in 0..na {
                for h in 0..nh {
                    cfv_avg[h] += sigma[a * nh + h] * cfv_all[a][h];
                }
            }
            let offset = self.node_data_offset[node_idx];
            for a in 0..na {
                for h in 0..nh {
                    let idx = offset + a * nh + h;
                    let regret = cfv_all[a][h] - cfv_avg[h];
                    let coef = if self.regrets[idx] >= 0.0 { alpha_t } else { beta_t };
                    self.regrets[idx] = coef * self.regrets[idx] + regret;
                    if self.regrets[idx] < self.regret_floor {
                        self.regrets[idx] = self.regret_floor;
                    }
                }
            }

            let reach_base = node_idx * np * nh;
            for a in 0..na {
                for h in 0..nh {
                    let idx = offset + a * nh + h;
                    self.cum_strategy[idx] = gamma * self.cum_strategy[idx] + sigma[a * nh + h];
                }
            }
        } else {
            for a in 0..na {
                for h in 0..nh {
                    cfv_avg[h] += cfv_all[a][h];
                }
            }
        }

        cfv_avg
    }

    pub fn run(
        &mut self,
        tree: &FlatTree,
        game: &dyn GameSpec,
        num_iterations: u32,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh0 = self.nh;

        let mut root_cfv_sum = vec![0.0f32; nh0];
        let mut count = 0u32;

        for _ in 0..num_iterations {
            let params = DiscountParams::new(self.iteration);
            self.iteration += 1;

            self.compute_all_strategies(tree);

            let reach = self.compute_reach(tree, game);

            for traverser in 0..np {
                let cfv = self.bottom_up_recursive(tree, game, traverser, 0, &reach, params.alpha_t, params.beta_t, params.gamma_t);

                if traverser == 0 {
                    for h in 0..nh0 {
                        root_cfv_sum[h] += cfv[h];
                    }
                    count += 1;
                }
            }
        }

        for h in 0..nh0 {
            root_cfv_sum[h] /= count as f32;
        }
        root_cfv_sum
    }

    pub fn run_sequential(
        &mut self,
        tree: &FlatTree,
        game: &dyn GameSpec,
        num_iterations: u32,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh0 = self.nh;

        let mut root_cfv_sum = vec![0.0f32; nh0];
        let mut count = 0u32;

        for _ in 0..num_iterations {
            let params = DiscountParams::new(self.iteration);
            self.iteration += 1;

            for traverser in 0..np {
                self.compute_all_strategies(tree);
                let reach = self.compute_reach(tree, game);

                let cfv = self.bottom_up_recursive(tree, game, traverser, 0, &reach, params.alpha_t, params.beta_t, params.gamma_t);

                if traverser == 0 {
                    for h in 0..nh0 {
                        root_cfv_sum[h] += cfv[h];
                    }
                    count += 1;
                }
            }
        }

        for h in 0..nh0 {
            root_cfv_sum[h] /= count as f32;
        }
        root_cfv_sum
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

    pub fn get_current_strategy(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let offset = self.node_data_offset[node_idx];
        if offset == UNUSED {
            return vec![];
        }
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for a in 0..num_actions {
            for h in 0..nh {
                result[a][h] = self.strategy[offset + a * nh + h];
            }
        }
        result
    }

    pub fn cum_strategy_slice(&self) -> &[f32] {
        &self.cum_strategy
    }

    pub fn regrets_slice(&self) -> &[f32] {
        &self.regrets
    }

    pub fn node_offsets(&self) -> &[usize] {
        &self.node_data_offset
    }

    pub fn iteration_count(&self) -> u32 {
        self.iteration
    }

    // ---- Diagnostic methods ----

    pub fn strategy_slice(&self) -> &[f32] {
        &self.strategy
    }

    pub fn reach_slice(&self) -> Vec<f32> {
        // Recompute reach from current state
        unreachable!("Use compute_reach directly")
    }

    /// Run one iteration of the `run()` algorithm (not sequential),
    /// returning diagnostic state for a specific traverser.
    /// This matches Metal's `run()` flow: strategies once, reach once, bottom-up per traverser.
    pub fn run_one_iteration_diagnostic(
        &mut self,
        tree: &FlatTree,
        game: &dyn GameSpec,
        traverser: usize,
    ) -> CpuDiagnosticSnapshot {
        let np = self.num_players as usize;
        let nh = self.nh;

        let params = DiscountParams::new(self.iteration);
        self.iteration += 1;

        // Step 1: compute strategies (once)
        self.compute_all_strategies(tree);
        let strategies = self.strategy.clone();

        // Step 2+3: compute reach (once)
        let reach = self.compute_reach(tree, game);

        // Step 4: bottom-up for this traverser
        let cfv = self.bottom_up_recursive(
            tree, game, traverser, 0, &reach,
            params.alpha_t, params.beta_t, params.gamma_t,
        );
        let regrets = self.regrets.clone();
        let cum_strategy = self.cum_strategy.clone();

        CpuDiagnosticSnapshot {
            strategies,
            reach,
            cfv,
            regrets,
            cum_strategy,
            traverser,
            alpha_t: params.alpha_t,
            beta_t: params.beta_t,
            gamma_t: params.gamma_t,
            nh,
            np,
        }
    }
}

pub struct CpuDiagnosticSnapshot {
    pub strategies: Vec<f32>,
    pub reach: Vec<f32>,
    pub cfv: Vec<f32>,
    pub regrets: Vec<f32>,
    pub cum_strategy: Vec<f32>,
    pub traverser: usize,
    pub alpha_t: f32,
    pub beta_t: f32,
    pub gamma_t: f32,
    pub nh: usize,
    pub np: usize,
}
