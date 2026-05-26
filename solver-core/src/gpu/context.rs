use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::tree::flat::{FlatNode, FlatTree, NODE_TYPE_PLAYER};

const UNUSED: u32 = u32::MAX;

const MAX_NA: usize = 4;
const MAX_NH: usize = 1326;
const MAX_DEPTH: usize = 16;
const FRAME_STRIDE: usize = 2 * MAX_NA * MAX_NH + MAX_NH;

pub struct GpuContext {
    stream: Arc<CudaStream>,
    kuhn_fn: CudaFunction,
    leduc_fn: CudaFunction,
    nplayer_fn: CudaFunction,
    nplayer_extsamp_fn: CudaFunction,
    nplayer_extsamp_compact_fn: CudaFunction,
    test_showdown_fn: CudaFunction,
}

impl GpuContext {
    pub fn new() -> Result<Self, DriverError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
        let ptx_path = PathBuf::from(out_dir.clone()).join("mccfr.ptx");
        let ptx = Ptx::from_file(&ptx_path);

        let module = ctx.load_module(ptx)?;
        let kuhn_fn = module.load_function("mccfr_kuhn")?;
        let leduc_fn = module.load_function("mccfr_leduc")?;
        let nplayer_fn = module.load_function("mccfr_nplayer")?;
        let nplayer_extsamp_fn = module.load_function("mccfr_nplayer_extsamp")?;
        let nplayer_extsamp_compact_fn = module.load_function("mccfr_nplayer_extsamp_compact")?;

        let test_ptx_path = PathBuf::from(out_dir).join("test_showdown.ptx");
        let test_ptx = Ptx::from_file(&test_ptx_path);
        let test_module = ctx.load_module(test_ptx)?;
        let test_showdown_fn = test_module.load_function("test_showdown")?;

        Ok(GpuContext {
            stream,
            kuhn_fn,
            leduc_fn,
            nplayer_fn,
            nplayer_extsamp_fn,
            nplayer_extsamp_compact_fn,
            test_showdown_fn,
        })
    }

    pub fn run_test_showdown(
        &self,
        sorted_opp_strength: &[u16],
        sorted_opp_indices: &[u16],
        sorted_player_strength: &[u16],
        sorted_player_indices: &[u16],
        hand_cards: &[u8],
        opp_reach: &[f32],
        nh: usize,
        contribution: f32,
    ) -> Result<Vec<f32>, DriverError> {
        let d_opp_str = self.stream.clone_htod(sorted_opp_strength)?;
        let d_opp_idx = self.stream.clone_htod(sorted_opp_indices)?;
        let d_pl_str = self.stream.clone_htod(sorted_player_strength)?;
        let d_pl_idx = self.stream.clone_htod(sorted_player_indices)?;
        let d_hand_cards = self.stream.clone_htod(hand_cards)?;
        let d_opp_reach = self.stream.clone_htod(opp_reach)?;
        let d_output: CudaSlice<f32> = self.stream.alloc_zeros(nh)?;

        let nh_i32 = nh as i32;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&self.test_showdown_fn);
            builder
                .arg(&d_opp_str)
                .arg(&d_opp_idx)
                .arg(&d_pl_str)
                .arg(&d_pl_idx)
                .arg(&d_hand_cards)
                .arg(&d_opp_reach)
                .arg(&nh_i32)
                .arg(&contribution)
                .arg(&d_output);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;
        self.stream.clone_dtoh(&d_output)
    }

    pub fn create_solver(
        &self,
        tree: &FlatTree,
        num_hands: Vec<usize>,
        sign_table: &[f32],
        game_type: GpuGameType,
    ) -> Result<GpuMccfr, DriverError> {
        let np = tree.num_players as usize;
        let nn = tree.num_nodes();

        assert_eq!(num_hands.len(), np);

        let mut node_offsets = vec![UNUSED; nn];
        let mut total = 0u32;
        for (i, node) in tree.nodes.iter().enumerate() {
            if node.node_type == NODE_TYPE_PLAYER {
                node_offsets[i] = total;
                total += node.num_children as u32 * num_hands[node.player_id as usize] as u32;
            }
        }

        let d_nodes: CudaSlice<FlatNode> = self.stream.clone_htod(&tree.nodes)?;
        let d_children: CudaSlice<u32> = self.stream.clone_htod(&tree.children)?;
        let d_contributions: CudaSlice<i32> = self.stream.clone_htod(&tree.contributions)?;
        let d_regrets: CudaSlice<f32> = self.stream.alloc_zeros(total as usize)?;
        let d_cum_strategy: CudaSlice<f32> = self.stream.alloc_zeros(total as usize)?;
        let d_node_offsets: CudaSlice<u32> = self.stream.clone_htod(&node_offsets)?;
        let d_sign_table: CudaSlice<f32> = self.stream.clone_htod(sign_table)?;

        let func = match game_type {
            GpuGameType::Kuhn => self.kuhn_fn.clone(),
            GpuGameType::Leduc => self.leduc_fn.clone(),
        };

        Ok(GpuMccfr {
            stream: self.stream.clone(),
            func,
            d_nodes,
            d_children,
            d_contributions,
            d_regrets,
            d_cum_strategy,
            d_node_offsets,
            d_sign_table,
            num_players: tree.num_players,
            num_hands,
            total_data: total as usize,
            node_offsets,
            iteration: 0,
            regret_floor: -1e7f32,
        })
    }
    pub fn create_nplayer_extsamp_solver(
        &self,
        tree: &FlatTree,
        nh: usize,
        hand_ranks: &[u16],
        sorted_opp_strength: &[u16],
        sorted_opp_indices: &[u16],
        sorted_player_strength: &[u16],
        sorted_player_indices: &[u16],
        same_hand_idx: &[u16],
        hand_cards: &[u8],
        initial_weight: &[f32],
        chance_ranks_table: Option<&[u16]>,
        remaining_deck: &[u8],
        chance_sorted_strength: Option<&[u16]>,
        chance_sorted_indices: Option<&[u16]>,
    ) -> Result<GpuNplayerExtSamp, DriverError> {
        let np = tree.num_players as usize;
        let nn = tree.num_nodes();

        let mut node_offsets = vec![UNUSED; nn];
        let mut total = 0u32;
        for (i, node) in tree.nodes.iter().enumerate() {
            if node.node_type == NODE_TYPE_PLAYER {
                node_offsets[i] = total;
                total += node.num_children as u32 * nh as u32;
            }
        }

        let d_nodes: CudaSlice<FlatNode> = self.stream.clone_htod(&tree.nodes)?;
        let d_children: CudaSlice<u32> = self.stream.clone_htod(&tree.children)?;
        let d_contributions: CudaSlice<i32> = self.stream.clone_htod(&tree.contributions)?;
        let d_regrets: CudaSlice<f32> = self.stream.alloc_zeros(total as usize)?;
        let d_cum_strategy: CudaSlice<f32> = self.stream.alloc_zeros(total as usize)?;
        let d_node_offsets: CudaSlice<u32> = self.stream.clone_htod(&node_offsets)?;
        let d_hand_ranks: CudaSlice<u16> = self.stream.clone_htod(hand_ranks)?;
        let d_sorted_opp_strength: CudaSlice<u16> = self.stream.clone_htod(sorted_opp_strength)?;
        let d_sorted_opp_indices: CudaSlice<u16> = self.stream.clone_htod(sorted_opp_indices)?;
        let d_sorted_player_strength: CudaSlice<u16> = self.stream.clone_htod(sorted_player_strength)?;
        let d_sorted_player_indices: CudaSlice<u16> = self.stream.clone_htod(sorted_player_indices)?;
        let d_same_hand_idx: CudaSlice<u16> = self.stream.clone_htod(same_hand_idx)?;
        let d_hand_cards: CudaSlice<u8> = self.stream.clone_htod(hand_cards)?;
        let d_initial_weight: CudaSlice<f32> = self.stream.clone_htod(initial_weight)?;

        let d_chance_ranks: CudaSlice<u16> = if let Some(table) = chance_ranks_table {
            self.stream.clone_htod(table)?
        } else {
            self.stream.alloc_zeros(1)?
        };
        let d_remaining_deck: CudaSlice<u8> = self.stream.clone_htod(remaining_deck)?;

        let num_remaining = remaining_deck.len();

        let d_chance_sorted_strength: CudaSlice<u16> = if let Some(table) = chance_sorted_strength {
            self.stream.clone_htod(table)?
        } else {
            self.stream.alloc_zeros(1)?
        };
        let d_chance_sorted_indices: CudaSlice<u16> = if let Some(table) = chance_sorted_indices {
            self.stream.clone_htod(table)?
        } else {
            self.stream.alloc_zeros(1)?
        };

        Ok(GpuNplayerExtSamp {
            stream: self.stream.clone(),
            func: self.nplayer_extsamp_fn.clone(),
            d_nodes,
            d_children,
            d_contributions,
            d_regrets,
            d_cum_strategy,
            d_node_offsets,
            d_hand_ranks,
            d_sorted_opp_strength,
            d_sorted_opp_indices,
            d_sorted_player_strength,
            d_sorted_player_indices,
            d_same_hand_idx,
            d_hand_cards,
            d_initial_weight,
            d_chance_ranks,
            d_remaining_deck,
            d_chance_sorted_strength,
            d_chance_sorted_indices,
            num_players: tree.num_players,
            num_hands: nh,
            num_remaining,
            total_data: total as usize,
            node_offsets,
            iteration: 0,
            regret_floor: -1e7f32,
        })
    }

    pub fn create_nplayer_extsamp_compact_solver(
        &self,
        tree: &FlatTree,
        nh: usize,
        hand_ranks: &[u16],
        sorted_opp_strength: &[u16],
        sorted_opp_indices: &[u16],
        sorted_player_strength: &[u16],
        sorted_player_indices: &[u16],
        same_hand_idx: &[u16],
        hand_cards: &[u8],
        initial_weight: &[f32],
        chance_ranks_table: Option<&[u16]>,
        remaining_deck: &[u8],
        chance_sorted_strength: Option<&[u16]>,
        chance_sorted_indices: Option<&[u16]>,
    ) -> Result<GpuNplayerExtSampCompact, DriverError> {
        let nn = tree.num_nodes();

        let mut node_offsets = vec![UNUSED; nn];
        let mut total = 0u32;
        for (i, node) in tree.nodes.iter().enumerate() {
            if node.node_type == NODE_TYPE_PLAYER {
                node_offsets[i] = total;
                total += node.num_children as u32 * nh as u32;
            }
        }

        let d_nodes: CudaSlice<FlatNode> = self.stream.clone_htod(&tree.nodes)?;
        let d_children: CudaSlice<u32> = self.stream.clone_htod(&tree.children)?;
        let d_contributions: CudaSlice<i32> = self.stream.clone_htod(&tree.contributions)?;
        let d_folded_masks: CudaSlice<u16> = self.stream.clone_htod(&tree.folded_masks)?;
        let d_regrets: CudaSlice<f32> = self.stream.alloc_zeros(total as usize)?;
        let d_cum_strategy: CudaSlice<f32> = self.stream.alloc_zeros(total as usize)?;
        let d_node_offsets: CudaSlice<u32> = self.stream.clone_htod(&node_offsets)?;
        let d_hand_ranks: CudaSlice<u16> = self.stream.clone_htod(hand_ranks)?;
        let d_sorted_opp_strength: CudaSlice<u16> = self.stream.clone_htod(sorted_opp_strength)?;
        let d_sorted_opp_indices: CudaSlice<u16> = self.stream.clone_htod(sorted_opp_indices)?;
        let d_sorted_player_strength: CudaSlice<u16> = self.stream.clone_htod(sorted_player_strength)?;
        let d_sorted_player_indices: CudaSlice<u16> = self.stream.clone_htod(sorted_player_indices)?;
        let d_same_hand_idx: CudaSlice<u16> = self.stream.clone_htod(same_hand_idx)?;
        let d_hand_cards: CudaSlice<u8> = self.stream.clone_htod(hand_cards)?;
        let d_initial_weight: CudaSlice<f32> = self.stream.clone_htod(initial_weight)?;

        let d_chance_ranks: CudaSlice<u16> = if let Some(table) = chance_ranks_table {
            self.stream.clone_htod(table)?
        } else {
            self.stream.alloc_zeros(1)?
        };
        let d_remaining_deck: CudaSlice<u8> = self.stream.clone_htod(remaining_deck)?;

        let num_remaining = remaining_deck.len();

        let d_chance_sorted_strength: CudaSlice<u16> = if let Some(table) = chance_sorted_strength {
            self.stream.clone_htod(table)?
        } else {
            self.stream.alloc_zeros(1)?
        };
        let d_chance_sorted_indices: CudaSlice<u16> = if let Some(table) = chance_sorted_indices {
            self.stream.clone_htod(table)?
        } else {
            self.stream.alloc_zeros(1)?
        };

        let mut solver = GpuNplayerExtSampCompact {
            stream: self.stream.clone(),
            func: self.nplayer_extsamp_compact_fn.clone(),
            d_nodes,
            d_children,
            d_contributions,
            d_folded_masks,
            d_regrets,
            d_cum_strategy,
            d_node_offsets,
            d_hand_ranks,
            d_sorted_opp_strength,
            d_sorted_opp_indices,
            d_sorted_player_strength,
            d_sorted_player_indices,
            d_same_hand_idx,
            d_hand_cards,
            d_initial_weight,
            d_chance_ranks,
            d_remaining_deck,
            d_chance_sorted_strength,
            d_chance_sorted_indices,
            num_players: tree.num_players,
            num_hands: nh,
            num_remaining,
            total_data: total as usize,
            node_offsets,
            iteration: 0,
            regret_floor: -1e7f32,
            per_traj_stride: MAX_DEPTH * FRAME_STRIDE,
        };

        let measure_batch = 256u32;
        use rand::Rng;
        let measure_seeds: Vec<u32> = (0..measure_batch)
            .map(|_| rand::thread_rng().gen::<u32>() | 1u32)
            .collect();

        let peaks = solver.measure_peak_cursor(measure_batch, &measure_seeds)?;
        let max_peak = *peaks.iter().max().unwrap_or(&0) as usize;
        let tight_stride = ((max_peak as f64 * 1.5).ceil() as usize).max(max_peak + FRAME_STRIDE);
        let tight_stride = ((tight_stride + 63) / 64) * 64; // align to 256 bytes
        solver.per_traj_stride = tight_stride;

        eprintln!("Auto-measured peak cursor: max={}, tight_stride={} (savings: {:.1}%)",
            max_peak, tight_stride,
            100.0 * (1.0 - tight_stride as f64 / (MAX_DEPTH * FRAME_STRIDE) as f64));

        Ok(solver)
    }

    pub fn create_nplayer_solver(
        &self,
        tree: &FlatTree,
        nh: usize,
        hand_ranks: &[u16],
        sorted_opp_strength: &[u16],
        sorted_opp_indices: &[u16],
        sorted_player_strength: &[u16],
        sorted_player_indices: &[u16],
        same_hand_idx: &[u16],
        hand_cards: &[u8],
        initial_weight: &[f32],
        chance_ranks_table: Option<&[u16]>,
        remaining_deck: &[u8],
        chance_sorted_strength: Option<&[u16]>,
        chance_sorted_indices: Option<&[u16]>,
    ) -> Result<GpuNplayerMccfr, DriverError> {
        let np = tree.num_players as usize;
        let nn = tree.num_nodes();

        assert_eq!(hand_ranks.len(), nh);
        assert_eq!(sorted_opp_strength.len(), (np - 1) * nh);
        assert_eq!(sorted_opp_indices.len(), (np - 1) * nh);
        assert_eq!(sorted_player_strength.len(), nh);
        assert_eq!(sorted_player_indices.len(), nh);
        assert_eq!(same_hand_idx.len(), nh);
        assert_eq!(hand_cards.len(), nh * 2);
        assert_eq!(initial_weight.len(), np * nh);

        let mut node_offsets = vec![UNUSED; nn];
        let mut total = 0u32;
        for (i, node) in tree.nodes.iter().enumerate() {
            if node.node_type == NODE_TYPE_PLAYER {
                node_offsets[i] = total;
                total += node.num_children as u32 * nh as u32;
            }
        }

        let d_nodes: CudaSlice<FlatNode> = self.stream.clone_htod(&tree.nodes)?;
        let d_children: CudaSlice<u32> = self.stream.clone_htod(&tree.children)?;
        let d_contributions: CudaSlice<i32> = self.stream.clone_htod(&tree.contributions)?;
        let d_regrets: CudaSlice<f32> = self.stream.alloc_zeros(total as usize)?;
        let d_cum_strategy: CudaSlice<f32> = self.stream.alloc_zeros(total as usize)?;
        let d_node_offsets: CudaSlice<u32> = self.stream.clone_htod(&node_offsets)?;
        let d_hand_ranks: CudaSlice<u16> = self.stream.clone_htod(hand_ranks)?;
        let d_sorted_opp_strength: CudaSlice<u16> = self.stream.clone_htod(sorted_opp_strength)?;
        let d_sorted_opp_indices: CudaSlice<u16> = self.stream.clone_htod(sorted_opp_indices)?;
        let d_sorted_player_strength: CudaSlice<u16> = self.stream.clone_htod(sorted_player_strength)?;
        let d_sorted_player_indices: CudaSlice<u16> = self.stream.clone_htod(sorted_player_indices)?;
        let d_same_hand_idx: CudaSlice<u16> = self.stream.clone_htod(same_hand_idx)?;
        let d_hand_cards: CudaSlice<u8> = self.stream.clone_htod(hand_cards)?;
        let d_initial_weight: CudaSlice<f32> = self.stream.clone_htod(initial_weight)?;

        let has_chance = chance_ranks_table.is_some();
        let d_chance_ranks: CudaSlice<u16> = if let Some(table) = chance_ranks_table {
            assert_eq!(table.len(), 52 * nh);
            self.stream.clone_htod(table)?
        } else {
            self.stream.alloc_zeros(1)?
        };
        let d_remaining_deck: CudaSlice<u8> = self.stream.clone_htod(remaining_deck)?;

        let num_remaining = remaining_deck.len();

        let has_chance = chance_ranks_table.is_some();
        let has_chance_sorted = chance_sorted_strength.is_some();
        let d_chance_sorted_strength: CudaSlice<u16> = if let Some(table) = chance_sorted_strength {
            assert_eq!(table.len(), 52 * nh);
            self.stream.clone_htod(table)?
        } else {
            self.stream.alloc_zeros(1)?
        };
        let d_chance_sorted_indices: CudaSlice<u16> = if let Some(table) = chance_sorted_indices {
            assert_eq!(table.len(), 52 * nh);
            self.stream.clone_htod(table)?
        } else {
            self.stream.alloc_zeros(1)?
        };

        Ok(GpuNplayerMccfr {
            stream: self.stream.clone(),
            func: self.nplayer_fn.clone(),
            d_nodes,
            d_children,
            d_contributions,
            d_regrets,
            d_cum_strategy,
            d_node_offsets,
            d_hand_ranks,
            d_sorted_opp_strength,
            d_sorted_opp_indices,
            d_sorted_player_strength,
            d_sorted_player_indices,
            d_same_hand_idx,
            d_hand_cards,
            d_initial_weight,
            d_chance_ranks,
            d_remaining_deck,
            d_chance_sorted_strength,
            d_chance_sorted_indices,
            num_players: tree.num_players,
            num_hands: nh,
            num_remaining,
            has_chance,
            has_chance_sorted,
            total_data: total as usize,
            node_offsets,
            iteration: 0,
            regret_floor: -1e7f32,
        })
    }
}

pub enum GpuGameType {
    Kuhn,
    Leduc,
}

pub struct GpuMccfr {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    d_nodes: CudaSlice<FlatNode>,
    d_children: CudaSlice<u32>,
    d_contributions: CudaSlice<i32>,
    d_regrets: CudaSlice<f32>,
    d_cum_strategy: CudaSlice<f32>,
    d_node_offsets: CudaSlice<u32>,
    d_sign_table: CudaSlice<f32>,
    num_players: u8,
    #[allow(dead_code)]
    num_hands: Vec<usize>,
    #[allow(dead_code)]
    total_data: usize,
    node_offsets: Vec<u32>,
    iteration: u32,
    regret_floor: f32,
}

impl GpuMccfr {
    pub fn run(&mut self, batch_size: u32, num_iterations: u32) -> Result<(), DriverError> {
        let mut seeds: Vec<u32> = Vec::with_capacity(batch_size as usize);
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..batch_size {
            seeds.push(rng.gen::<u32>() | 1u32);
        }

            for _ in 0..num_iterations {
                self.iteration += 1;

                use rand::Rng;
                let mut rng = rand::thread_rng();
                for s in &mut seeds {
                    *s = rng.gen::<u32>() | 1u32;
                }

                let d_seeds = self.stream.clone_htod(&seeds)?;
                let np = self.num_players as u32;

                let num_blocks = (batch_size + 255) / 256;
                let cfg = LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    let mut builder = self.stream.launch_builder(&self.func);
                    builder
                        .arg(&self.d_nodes)
                        .arg(&self.d_children)
                        .arg(&self.d_contributions)
                        .arg(&mut self.d_regrets)
                        .arg(&mut self.d_cum_strategy)
                        .arg(&self.d_node_offsets)
                        .arg(&self.d_sign_table)
                        .arg(&np)
                        .arg(&batch_size)
                        .arg(&1.0f32)
                        .arg(&self.regret_floor)
                        .arg(&d_seeds);
                    builder.launch(cfg)?;
                }

            self.stream.synchronize()?;
        }

        Ok(())
    }

    pub fn download_regrets(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_regrets)
    }

    pub fn download_cum_strategy(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_cum_strategy)
    }

    pub fn get_regrets_at(
        &self,
        node_idx: usize,
        num_actions: usize,
        nh: usize,
    ) -> Result<Vec<Vec<f32>>, DriverError> {
        let offset = self.node_offsets[node_idx];
        if offset == UNUSED {
            return Ok(vec![]);
        }
        let all = self.download_regrets()?;
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for a in 0..num_actions {
            for h in 0..nh {
                result[a][h] = all[offset as usize + a * nh + h];
            }
        }
        Ok(result)
    }

    pub fn get_average_strategy_at(
        &self,
        node_idx: usize,
        num_actions: usize,
        nh: usize,
    ) -> Result<Vec<Vec<f32>>, DriverError> {
        let offset = self.node_offsets[node_idx];
        if offset == UNUSED {
            return Ok(vec![]);
        }
        let all = self.download_cum_strategy()?;
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for h in 0..nh {
            let mut total = 0.0f32;
            for a in 0..num_actions {
                total += all[offset as usize + a * nh + h];
            }
            if total > 0.0 {
                for a in 0..num_actions {
                    result[a][h] = all[offset as usize + a * nh + h] / total;
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    result[a][h] = uniform;
                }
            }
        }
        Ok(result)
    }

    pub fn get_current_strategy_at(
        &self,
        node_idx: usize,
        num_actions: usize,
        nh: usize,
    ) -> Result<Vec<Vec<f32>>, DriverError> {
        let offset = self.node_offsets[node_idx];
        if offset == UNUSED {
            return Ok(vec![]);
        }
        let regrets = self.download_regrets()?;
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for h in 0..nh {
            let mut pos_sum = 0.0f32;
            for a in 0..num_actions {
                let r = regrets[offset as usize + a * nh + h];
                if r > 0.0 {
                    pos_sum += r;
                }
            }
            if pos_sum > 0.0 {
                for a in 0..num_actions {
                    let r = regrets[offset as usize + a * nh + h];
                    result[a][h] = if r > 0.0 { r / pos_sum } else { 0.0 };
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    result[a][h] = uniform;
                }
            }
        }
        Ok(result)
    }
}

pub struct GpuNplayerExtSamp {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    d_nodes: CudaSlice<FlatNode>,
    d_children: CudaSlice<u32>,
    d_contributions: CudaSlice<i32>,
    d_regrets: CudaSlice<f32>,
    d_cum_strategy: CudaSlice<f32>,
    d_node_offsets: CudaSlice<u32>,
    d_hand_ranks: CudaSlice<u16>,
    d_sorted_opp_strength: CudaSlice<u16>,
    d_sorted_opp_indices: CudaSlice<u16>,
    d_sorted_player_strength: CudaSlice<u16>,
    d_sorted_player_indices: CudaSlice<u16>,
    d_same_hand_idx: CudaSlice<u16>,
    d_hand_cards: CudaSlice<u8>,
    d_initial_weight: CudaSlice<f32>,
    d_chance_ranks: CudaSlice<u16>,
    d_remaining_deck: CudaSlice<u8>,
    d_chance_sorted_strength: CudaSlice<u16>,
    d_chance_sorted_indices: CudaSlice<u16>,
    num_players: u8,
    num_hands: usize,
    num_remaining: usize,
    #[allow(dead_code)]
    total_data: usize,
    node_offsets: Vec<u32>,
    iteration: u32,
    regret_floor: f32,
}

impl GpuNplayerExtSamp {
    pub fn run(&mut self, batch_size: u32, num_iterations: u32) -> Result<(), DriverError> {
        let frame_data_len = batch_size as usize * MAX_DEPTH * FRAME_STRIDE;
        let d_frame_data: CudaSlice<f32> = self.stream.alloc_zeros(frame_data_len)?;

        let mut seeds: Vec<u32> = Vec::with_capacity(batch_size as usize);

        for _ in 0..num_iterations {
            self.iteration += 1;

            let mut rng = rand::thread_rng();
            seeds.clear();
            for _ in 0..batch_size {
                use rand::Rng;
                seeds.push(rng.gen::<u32>() | 1u32);
            }

            let d_seeds = self.stream.clone_htod(&seeds)?;
            let np = self.num_players as u32;
            let nh = self.num_hands as i32;
            let nr = self.num_remaining as i32;

            let num_blocks = (batch_size + 31) / 32;
            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                let mut builder = self.stream.launch_builder(&self.func);
                builder
                    .arg(&self.d_nodes)
                    .arg(&self.d_children)
                    .arg(&self.d_contributions)
                    .arg(&mut self.d_regrets)
                    .arg(&mut self.d_cum_strategy)
                    .arg(&self.d_node_offsets)
                    .arg(&self.d_hand_ranks)
                    .arg(&self.d_sorted_opp_strength)
                    .arg(&self.d_sorted_opp_indices)
                    .arg(&self.d_sorted_player_strength)
                    .arg(&self.d_sorted_player_indices)
                    .arg(&self.d_same_hand_idx)
                    .arg(&self.d_hand_cards)
                    .arg(&self.d_initial_weight)
                    .arg(&self.d_chance_ranks)
                    .arg(&self.d_remaining_deck)
                    .arg(&self.d_chance_sorted_strength)
                    .arg(&self.d_chance_sorted_indices)
                    .arg(&d_frame_data)
                    .arg(&np)
                    .arg(&batch_size)
                    .arg(&nh)
                    .arg(&nr)
                    .arg(&1.0f32)
                    .arg(&self.regret_floor)
                    .arg(&d_seeds);
                builder.launch(cfg)?;
            }

            self.stream.synchronize()?;
        }

        Ok(())
    }

    pub fn run_with_seeds(&mut self, batch_size: u32, num_iterations: u32, seeds: &[u32]) -> Result<(), DriverError> {
        let frame_data_len = batch_size as usize * MAX_DEPTH * FRAME_STRIDE;
        let d_frame_data: CudaSlice<f32> = self.stream.alloc_zeros(frame_data_len)?;

        assert_eq!(seeds.len(), batch_size as usize);

        for _ in 0..num_iterations {
            self.iteration += 1;
            let d_seeds = self.stream.clone_htod(seeds)?;
            let np = self.num_players as u32;
            let nh = self.num_hands as i32;
            let nr = self.num_remaining as i32;

            let num_blocks = (batch_size + 31) / 32;
            let cfg = LaunchConfig { grid_dim: (num_blocks, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };

            unsafe {
                let mut builder = self.stream.launch_builder(&self.func);
                builder
                    .arg(&self.d_nodes).arg(&self.d_children).arg(&self.d_contributions)
                    .arg(&mut self.d_regrets).arg(&mut self.d_cum_strategy).arg(&self.d_node_offsets)
                    .arg(&self.d_hand_ranks).arg(&self.d_sorted_opp_strength).arg(&self.d_sorted_opp_indices)
                    .arg(&self.d_sorted_player_strength).arg(&self.d_sorted_player_indices)
                    .arg(&self.d_same_hand_idx).arg(&self.d_hand_cards).arg(&self.d_initial_weight)
                    .arg(&self.d_chance_ranks).arg(&self.d_remaining_deck)
                    .arg(&self.d_chance_sorted_strength).arg(&self.d_chance_sorted_indices)
                    .arg(&d_frame_data).arg(&np).arg(&batch_size).arg(&nh).arg(&nr)
                    .arg(&1.0f32).arg(&self.regret_floor).arg(&d_seeds);
                builder.launch(cfg)?;
            }
            self.stream.synchronize()?;
        }
        Ok(())
    }

    pub fn download_regrets(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_regrets)
    }

    pub fn download_cum_strategy(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_cum_strategy)
    }
}

pub struct GpuNplayerExtSampCompact {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    d_nodes: CudaSlice<FlatNode>,
    d_children: CudaSlice<u32>,
    d_contributions: CudaSlice<i32>,
    d_folded_masks: CudaSlice<u16>,
    d_regrets: CudaSlice<f32>,
    d_cum_strategy: CudaSlice<f32>,
    d_node_offsets: CudaSlice<u32>,
    d_hand_ranks: CudaSlice<u16>,
    d_sorted_opp_strength: CudaSlice<u16>,
    d_sorted_opp_indices: CudaSlice<u16>,
    d_sorted_player_strength: CudaSlice<u16>,
    d_sorted_player_indices: CudaSlice<u16>,
    d_same_hand_idx: CudaSlice<u16>,
    d_hand_cards: CudaSlice<u8>,
    d_initial_weight: CudaSlice<f32>,
    d_chance_ranks: CudaSlice<u16>,
    d_remaining_deck: CudaSlice<u8>,
    d_chance_sorted_strength: CudaSlice<u16>,
    d_chance_sorted_indices: CudaSlice<u16>,
    num_players: u8,
    num_hands: usize,
    num_remaining: usize,
    #[allow(dead_code)]
    total_data: usize,
    node_offsets: Vec<u32>,
    iteration: u32,
    regret_floor: f32,
    per_traj_stride: usize,
}

impl GpuNplayerExtSampCompact {
    pub fn run(&mut self, batch_size: u32, num_iterations: u32) -> Result<(), DriverError> {
        let frame_data_len = batch_size as usize * self.per_traj_stride;
        let d_frame_data: CudaSlice<f32> = self.stream.alloc_zeros(frame_data_len)?;
        let d_peak: CudaSlice<u32> = self.stream.alloc_zeros(batch_size as usize)?;
        let stride = self.per_traj_stride as u32;

        let mut seeds: Vec<u32> = Vec::with_capacity(batch_size as usize);

        for _ in 0..num_iterations {
            self.iteration += 1;
            let weight = self.iteration as f32;

            let mut rng = rand::thread_rng();
            seeds.clear();
            for _ in 0..batch_size {
                use rand::Rng;
                seeds.push(rng.gen::<u32>() | 1u32);
            }

            let d_seeds = self.stream.clone_htod(&seeds)?;
            let np = self.num_players as u32;
            let nh = self.num_hands as i32;
            let nr = self.num_remaining as i32;

            let num_blocks = (batch_size + 31) / 32;
            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                let mut builder = self.stream.launch_builder(&self.func);
                builder
                    .arg(&self.d_nodes)
                    .arg(&self.d_children)
                    .arg(&self.d_contributions)
                    .arg(&self.d_folded_masks)
                    .arg(&mut self.d_regrets)
                    .arg(&mut self.d_cum_strategy)
                    .arg(&self.d_node_offsets)
                    .arg(&self.d_hand_ranks)
                    .arg(&self.d_sorted_opp_strength)
                    .arg(&self.d_sorted_opp_indices)
                    .arg(&self.d_sorted_player_strength)
                    .arg(&self.d_sorted_player_indices)
                    .arg(&self.d_same_hand_idx)
                    .arg(&self.d_hand_cards)
                    .arg(&self.d_initial_weight)
                    .arg(&self.d_chance_ranks)
                    .arg(&self.d_remaining_deck)
                    .arg(&self.d_chance_sorted_strength)
                    .arg(&self.d_chance_sorted_indices)
                    .arg(&d_frame_data)
                    .arg(&np)
                    .arg(&batch_size)
                    .arg(&nh)
                    .arg(&nr)
                    .arg(&weight)
                    .arg(&self.regret_floor)
                    .arg(&d_seeds)
                    .arg(&d_peak)
                    .arg(&stride);
                builder.launch(cfg)?;
            }

            self.stream.synchronize()?;
        }

        Ok(())
    }

    pub fn run_with_seeds(&mut self, batch_size: u32, num_iterations: u32, seeds: &[u32]) -> Result<(), DriverError> {
        let frame_data_len = batch_size as usize * self.per_traj_stride;
        let d_frame_data: CudaSlice<f32> = self.stream.alloc_zeros(frame_data_len)?;
        let d_peak: CudaSlice<u32> = self.stream.alloc_zeros(batch_size as usize)?;
        let stride = self.per_traj_stride as u32;

        assert_eq!(seeds.len(), batch_size as usize);

        for _ in 0..num_iterations {
            self.iteration += 1;
            let d_seeds = self.stream.clone_htod(seeds)?;
            let np = self.num_players as u32;
            let nh = self.num_hands as i32;
            let nr = self.num_remaining as i32;

            let num_blocks = (batch_size + 31) / 32;
            let cfg = LaunchConfig { grid_dim: (num_blocks, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };

            unsafe {
                let mut builder = self.stream.launch_builder(&self.func);
                builder
                    .arg(&self.d_nodes).arg(&self.d_children).arg(&self.d_contributions)
                    .arg(&self.d_folded_masks)
                    .arg(&mut self.d_regrets).arg(&mut self.d_cum_strategy).arg(&self.d_node_offsets)
                    .arg(&self.d_hand_ranks).arg(&self.d_sorted_opp_strength).arg(&self.d_sorted_opp_indices)
                    .arg(&self.d_sorted_player_strength).arg(&self.d_sorted_player_indices)
                    .arg(&self.d_same_hand_idx).arg(&self.d_hand_cards).arg(&self.d_initial_weight)
                    .arg(&self.d_chance_ranks).arg(&self.d_remaining_deck)
                    .arg(&self.d_chance_sorted_strength).arg(&self.d_chance_sorted_indices)
                    .arg(&d_frame_data).arg(&np).arg(&batch_size).arg(&nh).arg(&nr)
                    .arg(&1.0f32).arg(&self.regret_floor).arg(&d_seeds).arg(&d_peak).arg(&stride);
                builder.launch(cfg)?;
            }
            self.stream.synchronize()?;
        }
        Ok(())
    }

    pub fn measure_peak_cursor(&self, batch_size: u32, seeds: &[u32]) -> Result<Vec<u32>, DriverError> {
        let frame_data_len = batch_size as usize * self.per_traj_stride;
        let d_frame_data: CudaSlice<f32> = self.stream.alloc_zeros(frame_data_len)?;
        let d_peak: CudaSlice<u32> = self.stream.alloc_zeros(batch_size as usize)?;
        let d_seeds = self.stream.clone_htod(seeds)?;
        let stride = self.per_traj_stride as u32;
        let d_dummy_regrets: CudaSlice<f32> = self.stream.clone_htod(&vec![0.0f32; self.total_data])?;
        let d_dummy_cum: CudaSlice<f32> = self.stream.clone_htod(&vec![0.0f32; self.total_data])?;

        let np = self.num_players as u32;
        let nh = self.num_hands as i32;
        let nr = self.num_remaining as i32;

        let num_blocks = (batch_size + 31) / 32;
        let cfg = LaunchConfig { grid_dim: (num_blocks, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            let mut builder = self.stream.launch_builder(&self.func);
            builder
                .arg(&self.d_nodes).arg(&self.d_children).arg(&self.d_contributions)
                .arg(&self.d_folded_masks)
                .arg(&d_dummy_regrets).arg(&d_dummy_cum).arg(&self.d_node_offsets)
                .arg(&self.d_hand_ranks).arg(&self.d_sorted_opp_strength).arg(&self.d_sorted_opp_indices)
                .arg(&self.d_sorted_player_strength).arg(&self.d_sorted_player_indices)
                .arg(&self.d_same_hand_idx).arg(&self.d_hand_cards).arg(&self.d_initial_weight)
                .arg(&self.d_chance_ranks).arg(&self.d_remaining_deck)
                .arg(&self.d_chance_sorted_strength).arg(&self.d_chance_sorted_indices)
                .arg(&d_frame_data).arg(&np).arg(&batch_size).arg(&nh).arg(&nr)
                .arg(&1.0f32).arg(&self.regret_floor).arg(&d_seeds).arg(&d_peak).arg(&stride);
            builder.launch(cfg)?;
        }
        self.stream.synchronize()?;
        self.stream.clone_dtoh(&d_peak)
    }

    pub fn download_regrets(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_regrets)
    }

    pub fn download_cum_strategy(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_cum_strategy)
    }

    pub fn node_offsets(&self) -> &[u32] {
        &self.node_offsets
    }

    pub fn num_hands(&self) -> usize {
        self.num_hands
    }
}

pub struct GpuNplayerMccfr {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    d_nodes: CudaSlice<FlatNode>,
    d_children: CudaSlice<u32>,
    d_contributions: CudaSlice<i32>,
    d_regrets: CudaSlice<f32>,
    d_cum_strategy: CudaSlice<f32>,
    d_node_offsets: CudaSlice<u32>,
    d_hand_ranks: CudaSlice<u16>,
    d_sorted_opp_strength: CudaSlice<u16>,
    d_sorted_opp_indices: CudaSlice<u16>,
    d_sorted_player_strength: CudaSlice<u16>,
    d_sorted_player_indices: CudaSlice<u16>,
    d_same_hand_idx: CudaSlice<u16>,
    d_hand_cards: CudaSlice<u8>,
    d_initial_weight: CudaSlice<f32>,
    d_chance_ranks: CudaSlice<u16>,
    d_remaining_deck: CudaSlice<u8>,
    d_chance_sorted_strength: CudaSlice<u16>,
    d_chance_sorted_indices: CudaSlice<u16>,
    num_players: u8,
    num_hands: usize,
    num_remaining: usize,
    has_chance: bool,
    has_chance_sorted: bool,
    #[allow(dead_code)]
    total_data: usize,
    node_offsets: Vec<u32>,
    iteration: u32,
    regret_floor: f32,
}

impl GpuNplayerMccfr {
    pub fn run(&mut self, batch_size: u32, num_iterations: u32) -> Result<(), DriverError> {
        let frame_data_len = batch_size as usize * MAX_DEPTH * FRAME_STRIDE;
        let d_frame_data: CudaSlice<f32> = self.stream.alloc_zeros(frame_data_len)?;

        let mut seeds: Vec<u32> = Vec::with_capacity(batch_size as usize);

        for _ in 0..num_iterations {
            self.iteration += 1;

            let mut rng = rand::thread_rng();
            seeds.clear();
            for _ in 0..batch_size {
                use rand::Rng;
                seeds.push(rng.gen::<u32>() | 1u32);
            }

            let d_seeds = self.stream.clone_htod(&seeds)?;
            let np = self.num_players as u32;
            let nh = self.num_hands as i32;
            let nr = self.num_remaining as i32;

            let num_blocks = (batch_size + 31) / 32;
            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                let mut builder = self.stream.launch_builder(&self.func);
                builder
                    .arg(&self.d_nodes)
                    .arg(&self.d_children)
                    .arg(&self.d_contributions)
                    .arg(&mut self.d_regrets)
                    .arg(&mut self.d_cum_strategy)
                    .arg(&self.d_node_offsets)
                    .arg(&self.d_hand_ranks)
                    .arg(&self.d_sorted_opp_strength)
                    .arg(&self.d_sorted_opp_indices)
                    .arg(&self.d_sorted_player_strength)
                    .arg(&self.d_sorted_player_indices)
                    .arg(&self.d_same_hand_idx)
                    .arg(&self.d_hand_cards)
                    .arg(&self.d_initial_weight)
                    .arg(&self.d_chance_ranks)
                    .arg(&self.d_remaining_deck)
                    .arg(&self.d_chance_sorted_strength)
                    .arg(&self.d_chance_sorted_indices)
                    .arg(&d_frame_data)
                    .arg(&np)
                    .arg(&batch_size)
                    .arg(&nh)
                    .arg(&nr)
                    .arg(&1.0f32)
                    .arg(&self.regret_floor)
                    .arg(&d_seeds);
                builder.launch(cfg)?;
            }

            self.stream.synchronize()?;
        }

        Ok(())
    }

    pub fn download_regrets(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_regrets)
    }

    pub fn download_cum_strategy(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_cum_strategy)
    }

    pub fn node_offsets(&self) -> &[u32] {
        &self.node_offsets
    }

    pub fn get_average_strategy_at(
        &self,
        node_idx: usize,
        num_actions: usize,
    ) -> Result<Vec<Vec<f32>>, DriverError> {
        let nh = self.num_hands;
        let offset = self.node_offsets[node_idx];
        if offset == UNUSED {
            return Ok(vec![]);
        }
        let all = self.download_cum_strategy()?;
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for h in 0..nh {
            let mut total = 0.0f32;
            for a in 0..num_actions {
                total += all[offset as usize + a * nh + h];
            }
            if total > 0.0 {
                for a in 0..num_actions {
                    result[a][h] = all[offset as usize + a * nh + h] / total;
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    result[a][h] = uniform;
                }
            }
        }
        Ok(result)
    }

    pub fn get_current_strategy_at(
        &self,
        node_idx: usize,
        num_actions: usize,
    ) -> Result<Vec<Vec<f32>>, DriverError> {
        let nh = self.num_hands;
        let offset = self.node_offsets[node_idx];
        if offset == UNUSED {
            return Ok(vec![]);
        }
        let regrets = self.download_regrets()?;
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for h in 0..nh {
            let mut pos_sum = 0.0f32;
            for a in 0..num_actions {
                let r = regrets[offset as usize + a * nh + h];
                if r > 0.0 {
                    pos_sum += r;
                }
            }
            if pos_sum > 0.0 {
                for a in 0..num_actions {
                    let r = regrets[offset as usize + a * nh + h];
                    result[a][h] = if r > 0.0 { r / pos_sum } else { 0.0 };
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    result[a][h] = uniform;
                }
            }
        }
        Ok(result)
    }
}
