use crate::gpu_metal::{MetalBuffer, MetalContext};
use crate::tree::flat::{FlatNode, FlatTree, MAX_NA_POSTFLOP};

use metal::{MTLSize, ComputePipelineState};

const UNUSED: u32 = u32::MAX;

struct DcfrParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DcfrParams {
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

/// Metal-backed vector CFR solver.
/// Mirrors the CPU `VectorCfr` but runs all kernels on the GPU via Metal.
pub struct MetalVectorCfr {
    // Pipelines
    strategies_pipeline: ComputePipelineState,
    init_reach_pipeline: ComputePipelineState,
    top_down_pipeline: ComputePipelineState,
    bottom_up_pipeline: ComputePipelineState,
    zero_pipeline: ComputePipelineState,

    // GPU buffers — tree structure (read-only after init)
    d_nodes: MetalBuffer<FlatNode>,
    d_children: MetalBuffer<u32>,
    d_contributions: MetalBuffer<i32>,
    d_folded_masks: MetalBuffer<u16>,
    d_infoset_offsets: MetalBuffer<u32>,
    d_decision_node_ids: MetalBuffer<u32>,

    // GPU buffers — solver state (read-write)
    d_regrets: MetalBuffer<f32>,
    d_strategy: MetalBuffer<f32>,
    d_cum_strategy: MetalBuffer<f32>,
    d_reach: MetalBuffer<f32>,
    d_cfv: MetalBuffer<f32>,

    // GPU buffers — game data (read-only)
    d_initial_weight: MetalBuffer<f32>,
    d_sorted_opp_strength: MetalBuffer<u16>,
    d_sorted_opp_indices: MetalBuffer<u16>,
    d_sorted_pl_strength: MetalBuffer<u16>,
    d_sorted_pl_indices: MetalBuffer<u16>,
    d_hand_cards: MetalBuffer<u8>,

    // Level node buffers (one per depth level)
    d_level_nodes: Vec<MetalBuffer<u32>>,
    level_counts: Vec<i32>,

    // Solver parameters
    num_players: u8,
    num_hands: usize,
    num_infosets: usize,
    max_depth: usize,
    iteration: u32,
    regret_floor: f32,
    starting_pot: i32,
    num_combinations: f32,

    // Scratch buffer for params
    d_params_buf: MetalBuffer<u8>,
}

impl MetalVectorCfr {
    /// Create a new Metal vector CFR solver for the given tree and game data.
    pub fn new(
        ctx: &MetalContext,
        tree: &FlatTree,
        num_hands: usize,
        initial_weight: &[Vec<f32>],  // [np][nh]
        sorted_opp_strength: &[u16],   // [num_opp * nh]
        sorted_opp_indices: &[u16],    // [num_opp * nh]
        sorted_pl_strength: &[u16],    // [nh]
        sorted_pl_indices: &[u16],     // [nh]
        hand_cards: &[u8],             // [nh * 2]
        num_combinations: f64,
    ) -> Self {
        let np = tree.num_players as usize;
        let nn = tree.num_nodes();

        // Determine num_hands
        let nh = num_hands;
        let num_infosets = tree.num_infosets as usize;
        let max_depth = tree.max_depth as usize;
        let infoset_data_size = num_infosets * MAX_NA_POSTFLOP * nh;

        // Create pipelines
        let strategies_pipeline = ctx.create_pipeline("vcfr_compute_strategies")
            .expect("Failed to create strategies pipeline");
        let init_reach_pipeline = ctx.create_pipeline("vcfr_init_reach")
            .expect("Failed to create init_reach pipeline");
        let top_down_pipeline = ctx.create_pipeline("vcfr_top_down_reach")
            .expect("Failed to create top_down pipeline");
        let bottom_up_pipeline = ctx.create_pipeline("vcfr_bottom_up")
            .expect("Failed to create bottom_up pipeline");
        let zero_pipeline = ctx.create_pipeline("vcfr_zero_buffer")
            .expect("Failed to create zero_buffer pipeline");

        // Upload tree structure
        let d_nodes = ctx.upload(&tree.nodes);
        let d_children = ctx.upload(&tree.children);
        let d_contributions = ctx.upload(&tree.contributions);
        let d_folded_masks = ctx.upload(&tree.folded_masks);
        let d_infoset_offsets = ctx.upload(&tree.infoset_offsets);
        let d_decision_node_ids = ctx.upload(&tree.decision_node_ids);

        // Allocate solver state buffers
        let d_regrets = ctx.alloc_zeros(infoset_data_size);
        let d_strategy = ctx.alloc_zeros(infoset_data_size);
        let d_cum_strategy = ctx.alloc_zeros(infoset_data_size);
        let d_reach = ctx.alloc_zeros(nn * np * nh);
        let d_cfv = ctx.alloc_zeros(nn * nh);

        // Flatten initial weights: [p0_nh, p1_nh, ..., pN_nh]
        let initial_weight_flat: Vec<f32> = initial_weight.iter().flat_map(|w| w.iter().copied()).collect();
        let d_initial_weight = ctx.upload(&initial_weight_flat);

        // Upload sorted arrays
        let d_sorted_opp_strength = ctx.upload(sorted_opp_strength);
        let d_sorted_opp_indices = ctx.upload(sorted_opp_indices);
        let d_sorted_pl_strength = ctx.upload(sorted_pl_strength);
        let d_sorted_pl_indices = ctx.upload(sorted_pl_indices);
        let d_hand_cards = ctx.upload(hand_cards);

        // Level node buffers
        let mut d_level_nodes = Vec::with_capacity(max_depth + 1);
        let mut level_counts = Vec::with_capacity(max_depth + 1);
        for level in 0..=max_depth {
            let nodes = tree.nodes_at_level(level as u32).to_vec();
            level_counts.push(nodes.len() as i32);
            d_level_nodes.push(ctx.upload(&nodes));
        }

        // Params scratch buffer (large enough for all param structs)
        let d_params_buf = ctx.alloc_zeros(64);

        MetalVectorCfr {
            strategies_pipeline,
            init_reach_pipeline,
            top_down_pipeline,
            bottom_up_pipeline,
            zero_pipeline,
            d_nodes,
            d_children,
            d_contributions,
            d_folded_masks,
            d_infoset_offsets,
            d_decision_node_ids,
            d_regrets,
            d_strategy,
            d_cum_strategy,
            d_reach,
            d_cfv,
            d_initial_weight,
            d_sorted_opp_strength,
            d_sorted_opp_indices,
            d_sorted_pl_strength,
            d_sorted_pl_indices,
            d_hand_cards,
            d_level_nodes,
            level_counts,
            num_players: tree.num_players,
            num_hands: nh,
            num_infosets,
            max_depth,
            iteration: 0,
            regret_floor: -1e30,
            starting_pot: tree.starting_pot,
            num_combinations: num_combinations as f32,
            d_params_buf,
        }
    }

    /// Run the solver for the specified number of iterations.
    /// Returns root CFV for player 0.
    pub fn run(
        &mut self,
        ctx: &MetalContext,
        tree: &FlatTree,
        num_iterations: u32,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = self.num_hands;
        let ni = self.num_infosets;
        let nn = tree.num_nodes();

        let mut root_cfv_sum = vec![0.0f32; nh];
        let mut count = 0u32;

        for _ in 0..num_iterations {
            let params = DcfrParams::new(self.iteration);
            self.iteration += 1;

            // Sequential (alternating) updates: recompute strategies and reach
            // before each traverser's bottom-up pass. This matches the CUDA
            // GPU solver and Tammelin CFR+ — each traverser sees the most
            // up-to-date regret state, yielding ~8x faster convergence than
            // computing strategies once per iteration.
            for traverser in 0..np {
                self.launch_compute_strategies(ctx, ni, nh);
                self.launch_init_reach(ctx, np, nh);
                self.launch_top_down(ctx, nh, np as u32);

                self.launch_bottom_up(
                    ctx, tree, traverser as u32,
                    params.alpha_t, params.beta_t, params.gamma_t,
                    nh, np as u32,
                );

                if traverser == 0 {
                    let cfv = self.d_cfv.as_slice();
                    for h in 0..nh {
                        root_cfv_sum[h] += cfv[h];
                    }
                    count += 1;
                }
            }
        }

        for h in 0..nh {
            root_cfv_sum[h] /= count as f32;
        }
        root_cfv_sum
    }

    fn launch_compute_strategies(&mut self, ctx: &MetalContext, num_infosets: usize, nh: usize) {
        let params_data: [i32; 2] = [num_infosets as i32, nh as i32];
        self.d_params_buf.as_mut_slice()[..8].copy_from_slice(unsafe {
            std::slice::from_raw_parts(params_data.as_ptr() as *const u8, 8)
        });

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.strategies_pipeline);
        enc.set_buffer(0, Some(self.d_regrets.as_ref()), 0);
        enc.set_buffer(1, Some(self.d_strategy.as_ref()), 0);
        enc.set_buffer(2, Some(self.d_decision_node_ids.as_ref()), 0);
        enc.set_buffer(3, Some(self.d_nodes.as_ref()), 0);
        enc.set_buffer(4, Some(self.d_infoset_offsets.as_ref()), 0);
        enc.set_buffer(5, Some(self.d_params_buf.as_ref()), 0);

        let max_tpg = self.strategies_pipeline.max_total_threads_per_threadgroup() as usize;
        let (grid, tg) = ctx.dispatch_2d(num_infosets, nh, max_tpg);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn launch_init_reach(&mut self, ctx: &MetalContext, np: usize, nh: usize) {
        let nn = self.d_nodes.len();
        let total_reach = nn * np * nh;
        let np_nh = np * nh;

        let params_data: [i32; 2] = [total_reach as i32, np_nh as i32];
        self.d_params_buf.as_mut_slice()[..8].copy_from_slice(unsafe {
            std::slice::from_raw_parts(params_data.as_ptr() as *const u8, 8)
        });

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.init_reach_pipeline);
        enc.set_buffer(0, Some(self.d_reach.as_ref()), 0);
        enc.set_buffer(1, Some(self.d_initial_weight.as_ref()), 0);
        enc.set_buffer(2, Some(self.d_params_buf.as_ref()), 0);

        let (grid, tg) = ctx.dispatch_1d(total_reach, 256);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn launch_top_down(&mut self, ctx: &MetalContext, nh: usize, np: u32) {
        let nh_i32 = nh as i32;

        for level in 0..=self.max_depth {
            let count = self.level_counts[level];
            if count == 0 { continue; }

            let params_data: [i32; 3] = [count, np as i32, nh_i32];
            self.d_params_buf.as_mut_slice()[..12].copy_from_slice(unsafe {
                std::slice::from_raw_parts(params_data.as_ptr() as *const u8, 12)
            });

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.top_down_pipeline);
            enc.set_buffer(0, Some(self.d_level_nodes[level].as_ref()), 0);
            enc.set_buffer(1, Some(self.d_params_buf.as_ref()), 0);
            enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_children.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_strategy.as_ref()), 0);
            enc.set_buffer(5, Some(self.d_infoset_offsets.as_ref()), 0);
            enc.set_buffer(6, Some(self.d_reach.as_ref()), 0);

            let max_tpg = self.top_down_pipeline.max_total_threads_per_threadgroup() as usize;
            let (grid, tg) = ctx.dispatch_2d(count as usize, nh, max_tpg);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    fn launch_bottom_up(
        &mut self,
        ctx: &MetalContext,
        _tree: &FlatTree,
        traverser: u32,
        alpha_t: f32,
        beta_t: f32,
        gamma_t: f32,
        nh: usize,
        np: u32,
    ) {
        let nh_i32 = nh as i32;

        // Zero CFV buffer before bottom-up pass
        {
            let params_data: [i32; 1] = [self.d_cfv.len() as i32];
            self.d_params_buf.as_mut_slice()[..4].copy_from_slice(unsafe {
                std::slice::from_raw_parts(params_data.as_ptr() as *const u8, 4)
            });

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.zero_pipeline);
            enc.set_buffer(0, Some(self.d_cfv.as_ref()), 0);
            enc.set_buffer(1, Some(self.d_params_buf.as_ref()), 0);

            let (grid, tg) = ctx.dispatch_1d(self.d_cfv.len(), 256);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Bottom-up: process levels from deepest to shallowest
        for level in (0..=self.max_depth).rev() {
            let count = self.level_counts[level];
            if count == 0 { continue; }

            // Pack params struct: [level_count, num_players, nh, traverser, alpha_t, beta_t, gamma_t, regret_floor, starting_pot, num_combinations]
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct BuParams {
                level_count: i32,
                num_players: i32,
                nh: i32,
                traverser: u32,
                alpha_t: f32,
                beta_t: f32,
                gamma_t: f32,
                regret_floor: f32,
                starting_pot: i32,
                num_combinations: f32,
            }

            let bu_params = BuParams {
                level_count: count,
                num_players: np as i32,
                nh: nh_i32,
                traverser,
                alpha_t,
                beta_t,
                gamma_t,
                regret_floor: self.regret_floor,
                starting_pot: self.starting_pot,
                num_combinations: self.num_combinations,
            };

            self.d_params_buf.as_mut_slice()[..std::mem::size_of::<BuParams>()]
                .copy_from_slice(unsafe {
                    std::slice::from_raw_parts(
                        &bu_params as *const _ as *const u8,
                        std::mem::size_of::<BuParams>(),
                    )
                });

            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.bottom_up_pipeline);
            enc.set_buffer(0, Some(self.d_level_nodes[level].as_ref()), 0);
            enc.set_buffer(1, Some(self.d_params_buf.as_ref()), 0);
            enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_children.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_contributions.as_ref()), 0);
            enc.set_buffer(5, Some(self.d_folded_masks.as_ref()), 0);
            enc.set_buffer(6, Some(self.d_strategy.as_ref()), 0);
            enc.set_buffer(7, Some(self.d_infoset_offsets.as_ref()), 0);
            enc.set_buffer(8, Some(self.d_reach.as_ref()), 0);
            enc.set_buffer(9, Some(self.d_cfv.as_ref()), 0);
            enc.set_buffer(10, Some(self.d_regrets.as_ref()), 0);
            enc.set_buffer(11, Some(self.d_cum_strategy.as_ref()), 0);
            enc.set_buffer(12, Some(self.d_initial_weight.as_ref()), 0);
            enc.set_buffer(13, Some(self.d_sorted_opp_strength.as_ref()), 0);
            enc.set_buffer(14, Some(self.d_sorted_opp_indices.as_ref()), 0);
            enc.set_buffer(15, Some(self.d_sorted_pl_strength.as_ref()), 0);
            enc.set_buffer(16, Some(self.d_sorted_pl_indices.as_ref()), 0);
            enc.set_buffer(17, Some(self.d_hand_cards.as_ref()), 0);

            let (grid, tg) = ctx.dispatch_1d(count as usize, 1);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    /// Get average strategy at a node.
    pub fn get_average_strategy(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let offset = self.node_data_offset(node_idx);
        if offset == UNUSED as usize {
            return vec![];
        }
        let cum = self.d_cum_strategy.as_slice();
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for h in 0..nh {
            let mut total = 0.0f32;
            for a in 0..num_actions {
                total += cum[offset + a * nh + h];
            }
            if total > 0.0 {
                for a in 0..num_actions {
                    result[a][h] = cum[offset + a * nh + h] / total;
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

    pub fn cum_strategy_slice(&self) -> Vec<f32> {
        self.d_cum_strategy.as_slice().to_vec()
    }

    pub fn node_offsets(&self) -> Vec<usize> {
        let nh = self.num_hands;
        self.d_infoset_offsets.as_slice()
            .iter()
            .map(|&off| {
                if off == UNUSED { usize::MAX }
                else { off as usize * MAX_NA_POSTFLOP * nh }
            })
            .collect()
    }

    pub fn iteration_count(&self) -> u32 {
        self.iteration
    }

    fn node_data_offset(&self, node_idx: usize) -> usize {
        let off = self.d_infoset_offsets.as_slice()[node_idx];
        if off == UNUSED { usize::MAX }
        else { off as usize * MAX_NA_POSTFLOP * self.num_hands }
    }

    // ---- Diagnostic methods for debugging kernel divergence ----

    pub fn strategy_slice(&self) -> Vec<f32> {
        self.d_strategy.as_slice().to_vec()
    }

    pub fn regrets_slice(&self) -> Vec<f32> {
        self.d_regrets.as_slice().to_vec()
    }

    pub fn reach_slice(&self) -> Vec<f32> {
        self.d_reach.as_slice().to_vec()
    }

    pub fn cfv_slice(&self) -> Vec<f32> {
        self.d_cfv.as_slice().to_vec()
    }

    /// Expose internal state for single-iteration diagnostics.
    /// Runs one full iteration: strategies -> init_reach -> top_down -> bottom_up.
    /// After each stage, the current state of the named buffer is returned.
    pub fn run_one_iteration_diagnostic(
        &mut self,
        ctx: &MetalContext,
        tree: &FlatTree,
        traverser: u32,
    ) -> DiagnosticSnapshot {
        let np = self.num_players as usize;
        let nh = self.num_hands;
        let ni = self.num_infosets;
        let nn = tree.num_nodes();
        let params = DcfrParams::new(self.iteration);
        self.iteration += 1;

        // Step 1: compute strategies
        self.launch_compute_strategies(ctx, ni, nh);
        let strategies = self.strategy_slice();

        // Step 2: init reach
        self.launch_init_reach(ctx, np, nh);
        let reach_after_init = self.reach_slice();

        // Step 3: top-down reach
        self.launch_top_down(ctx, nh, np as u32);
        let reach_after_topdown = self.reach_slice();

        // Step 4: bottom-up for this traverser
        self.launch_bottom_up(
            ctx, tree, traverser,
            params.alpha_t, params.beta_t, params.gamma_t,
            nh, np as u32,
        );
        let cfv = self.cfv_slice();
        let regrets = self.regrets_slice();
        let cum_strategy = self.cum_strategy_slice();

        DiagnosticSnapshot {
            strategies,
            reach_after_init,
            reach_after_topdown,
            cfv,
            regrets,
            cum_strategy,
            traverser,
            alpha_t: params.alpha_t,
            beta_t: params.beta_t,
            gamma_t: params.gamma_t,
        }
    }
}

pub struct DiagnosticSnapshot {
    pub strategies: Vec<f32>,
    pub reach_after_init: Vec<f32>,
    pub reach_after_topdown: Vec<f32>,
    pub cfv: Vec<f32>,
    pub regrets: Vec<f32>,
    pub cum_strategy: Vec<f32>,
    pub traverser: u32,
    pub alpha_t: f32,
    pub beta_t: f32,
    pub gamma_t: f32,
}
