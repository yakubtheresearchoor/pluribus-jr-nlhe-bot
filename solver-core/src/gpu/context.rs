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

pub struct ChanceGpuData {
    pub chance_sorted_strength: Vec<u16>,
    pub chance_sorted_indices: Vec<u16>,
    pub chance_probabilities: Vec<f32>,
    pub remaining_deck: Vec<u8>,
}

pub struct ChanceInfo {
    pub num_outcomes: usize,
    pub remaining_deck: Vec<u8>,
    pub num_chance_children: usize,
}

pub struct GpuContext {
    stream: Arc<CudaStream>,
    kuhn_fn: CudaFunction,
    leduc_fn: CudaFunction,
    nplayer_fn: CudaFunction,
    nplayer_extsamp_fn: CudaFunction,
    nplayer_extsamp_compact_fn: CudaFunction,
    test_showdown_fn: CudaFunction,
    vcfr_strategies_fn: CudaFunction,
    vcfr_top_down_fn: CudaFunction,
    vcfr_bottom_up_fn: CudaFunction,
    vcfr_chance_accum_fn: CudaFunction,
    vcfr_chance_final_fn: CudaFunction,
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

        let test_ptx_path = PathBuf::from(out_dir.clone()).join("test_showdown.ptx");
        let test_ptx = Ptx::from_file(&test_ptx_path);
        let test_module = ctx.load_module(test_ptx)?;
        let test_showdown_fn = test_module.load_function("test_showdown")?;

        let vcfr_ptx_path = PathBuf::from(out_dir.clone()).join("vcfr.ptx");
        let vcfr_ptx = Ptx::from_file(&vcfr_ptx_path);
        let vcfr_module = ctx.load_module(vcfr_ptx)?;
        let vcfr_strategies_fn = vcfr_module.load_function("vcfr_compute_strategies")?;
        let vcfr_top_down_fn = vcfr_module.load_function("vcfr_top_down_reach")?;
        let vcfr_bottom_up_fn = vcfr_module.load_function("vcfr_bottom_up")?;
        let vcfr_chance_accum_fn = vcfr_module.load_function("vcfr_chance_accumulate")?;
        let vcfr_chance_final_fn = vcfr_module.load_function("vcfr_chance_finalize")?;

        Ok(GpuContext {
            stream,
            kuhn_fn,
            leduc_fn,
            nplayer_fn,
            nplayer_extsamp_fn,
            nplayer_extsamp_compact_fn,
            test_showdown_fn,
            vcfr_strategies_fn,
            vcfr_top_down_fn,
            vcfr_bottom_up_fn,
            vcfr_chance_accum_fn,
            vcfr_chance_final_fn,
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
            regret_floor: -1e30f32,
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
            regret_floor: -1e30f32,
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
            regret_floor: -1e30f32,
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
            regret_floor: -1e30f32,
        })
    }

    pub fn create_vcfr_solver(
        &self,
        tree: &FlatTree,
        nh: usize,
        sorted_opp_strength: &[u16],
        sorted_opp_indices: &[u16],
        sorted_player_strength: &[u16],
        sorted_player_indices: &[u16],
        hand_cards: &[u8],
        initial_weight: &[f32],
        chance_data: Option<ChanceGpuData>,
    ) -> Result<GpuVectorCfr, DriverError> {
        let np = tree.num_players as usize;
        let nn = tree.num_nodes();
        let num_infosets = tree.num_infosets as usize;
        let max_depth = tree.max_depth as usize;
        let num_opp = np - 1;

        let infoset_data_size = num_infosets * MAX_NA * nh;

        let d_nodes: CudaSlice<FlatNode> = self.stream.clone_htod(&tree.nodes)?;
        let d_children: CudaSlice<u32> = self.stream.clone_htod(&tree.children)?;
        let d_contributions: CudaSlice<i32> = self.stream.clone_htod(&tree.contributions)?;
        let d_folded_masks: CudaSlice<u16> = self.stream.clone_htod(&tree.folded_masks)?;
        let d_infoset_offsets: CudaSlice<u32> = self.stream.clone_htod(&tree.infoset_offsets)?;
        let d_decision_node_ids: CudaSlice<u32> = self.stream.clone_htod(&tree.decision_node_ids)?;
        let d_regrets: CudaSlice<f32> = self.stream.alloc_zeros(infoset_data_size)?;
        let d_strategy: CudaSlice<f32> = self.stream.alloc_zeros(infoset_data_size)?;
        let d_cum_strategy: CudaSlice<f32> = self.stream.alloc_zeros(infoset_data_size)?;
        let d_reach: CudaSlice<f32> = self.stream.alloc_zeros(nn * np * nh)?;
        let d_cfv: CudaSlice<f32> = self.stream.alloc_zeros(nn * nh)?;
        let d_initial_weight: CudaSlice<f32> = self.stream.clone_htod(initial_weight)?;
        let d_sorted_opp_strength: CudaSlice<u16> = self.stream.clone_htod(sorted_opp_strength)?;
        let d_sorted_opp_indices: CudaSlice<u16> = self.stream.clone_htod(sorted_opp_indices)?;
        let d_sorted_player_strength: CudaSlice<u16> = self.stream.clone_htod(sorted_player_strength)?;
        let d_sorted_player_indices: CudaSlice<u16> = self.stream.clone_htod(sorted_player_indices)?;
        let d_hand_cards: CudaSlice<u8> = self.stream.clone_htod(hand_cards)?;

        let mut d_level_nodes: Vec<CudaSlice<u32>> = Vec::with_capacity(max_depth + 1);
        for level in 0..=max_depth {
            let nodes = tree.nodes_at_level(level as u32);
            d_level_nodes.push(self.stream.clone_htod(nodes)?);
        }

        let mut level_counts: Vec<i32> = Vec::with_capacity(max_depth + 1);
        for level in 0..=max_depth {
            level_counts.push(tree.level_size(level as u32) as i32);
        }

        let vcfr_chance_accum_fn = self.vcfr_chance_accum_fn.clone();
        let vcfr_chance_final_fn = self.vcfr_chance_final_fn.clone();

        let (chance_info, d_chance_sorted_strength, d_chance_sorted_indices,
             d_chance_prob, d_chance_child_ids, d_cfv_accum,
             d_below_chance_level_nodes, below_chance_level_counts,
             d_main_level_nodes, main_level_counts) = if let Some(cd) = chance_data {
            let num_outcomes = cd.remaining_deck.len();
            let num_opp = np - 1;

            assert_eq!(cd.chance_sorted_strength.len(), 52 * num_opp * nh);
            assert_eq!(cd.chance_sorted_indices.len(), 52 * num_opp * nh);
            assert_eq!(cd.chance_probabilities.len(), num_outcomes * nh);

            let d_cstr: CudaSlice<u16> = self.stream.clone_htod(&cd.chance_sorted_strength)?;
            let d_cidx: CudaSlice<u16> = self.stream.clone_htod(&cd.chance_sorted_indices)?;
            let d_cprob: CudaSlice<f32> = self.stream.clone_htod(&cd.chance_probabilities)?;

            let mut chance_nodes = Vec::new();
            let mut chance_children = Vec::new();
            let mut below_chance = vec![false; nn];
            for i in 0..nn {
                if tree.nodes[i].is_chance() {
                    chance_nodes.push(i as u32);
                    for &child in tree.node_children(i) {
                        chance_children.push(child);
                        mark_descendants(tree, child as usize, &mut below_chance);
                    }
                }
            }

            let num_chance_children = chance_children.len();
            let d_ccids: CudaSlice<u32> = self.stream.clone_htod(&chance_children)?;
            let d_cacc: CudaSlice<f32> = self.stream.alloc_zeros(nn * nh)?;

            let mut d_bc_nodes: Vec<Option<CudaSlice<u32>>> = Vec::with_capacity(max_depth + 1);
            let mut bc_counts: Vec<i32> = Vec::with_capacity(max_depth + 1);
            let mut d_mn_nodes: Vec<Option<CudaSlice<u32>>> = Vec::with_capacity(max_depth + 1);
            let mut mn_counts: Vec<i32> = Vec::with_capacity(max_depth + 1);

            for level in 0..=max_depth {
                let all_nodes = tree.nodes_at_level(level as u32);
                let mut bc: Vec<u32> = Vec::new();
                let mut mn: Vec<u32> = Vec::new();
                for &nid in all_nodes {
                    if below_chance[nid as usize] {
                        bc.push(nid);
                    } else {
                        mn.push(nid);
                    }
                }
                bc_counts.push(bc.len() as i32);
                mn_counts.push(mn.len() as i32);
                if !bc.is_empty() {
                    d_bc_nodes.push(Some(self.stream.clone_htod(&bc)?));
                } else {
                    d_bc_nodes.push(None);
                }
                if !mn.is_empty() {
                    d_mn_nodes.push(Some(self.stream.clone_htod(&mn)?));
                } else {
                    d_mn_nodes.push(None);
                }
            }

            let info = ChanceInfo {
                num_outcomes,
                remaining_deck: cd.remaining_deck,
                num_chance_children,
            };

            (Some(info), Some(d_cstr), Some(d_cidx), Some(d_cprob),
             Some(d_ccids), Some(d_cacc),
             d_bc_nodes, bc_counts, d_mn_nodes, mn_counts)
        } else {
            let mut d_bc_nodes = Vec::with_capacity(max_depth + 1);
            let mut bc_counts = Vec::with_capacity(max_depth + 1);
            let mut d_mn_nodes = Vec::with_capacity(max_depth + 1);
            let mut mn_counts = Vec::with_capacity(max_depth + 1);
            for level in 0..=max_depth {
                d_bc_nodes.push(None);
                bc_counts.push(0);
                d_mn_nodes.push(None);
                mn_counts.push(0);
            }
            (None, None, None, None, None, None,
             d_bc_nodes, bc_counts, d_mn_nodes, mn_counts)
        };

        Ok(GpuVectorCfr {
            stream: self.stream.clone(),
            vcfr_strategies_fn: self.vcfr_strategies_fn.clone(),
            vcfr_top_down_fn: self.vcfr_top_down_fn.clone(),
            vcfr_bottom_up_fn: self.vcfr_bottom_up_fn.clone(),
            vcfr_chance_accum_fn,
            vcfr_chance_final_fn,
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
            d_sorted_player_strength,
            d_sorted_player_indices,
            d_hand_cards,
            d_level_nodes,
            num_players: tree.num_players,
            num_hands: nh,
            num_infosets,
            max_depth,
            level_counts,
            iteration: 0,
            regret_floor: -1e30f32,
            chance_info,
            d_chance_sorted_strength,
            d_chance_sorted_indices,
            d_chance_prob,
            d_chance_child_ids,
            d_cfv_accum,
            d_below_chance_level_nodes,
            below_chance_level_counts,
            d_main_level_nodes,
            main_level_counts,
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
                let weight = self.iteration as f32;

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
                        .arg(&weight)
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
}

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

fn mark_descendants(tree: &FlatTree, node_idx: usize, below_chance: &mut [bool]) {
    below_chance[node_idx] = true;
    for &child in tree.node_children(node_idx) {
        mark_descendants(tree, child as usize, below_chance);
    }
}

pub struct GpuVectorCfr {
    stream: Arc<CudaStream>,
    vcfr_strategies_fn: CudaFunction,
    vcfr_top_down_fn: CudaFunction,
    vcfr_bottom_up_fn: CudaFunction,
    vcfr_chance_accum_fn: CudaFunction,
    vcfr_chance_final_fn: CudaFunction,
    d_nodes: CudaSlice<FlatNode>,
    d_children: CudaSlice<u32>,
    d_contributions: CudaSlice<i32>,
    d_folded_masks: CudaSlice<u16>,
    d_infoset_offsets: CudaSlice<u32>,
    d_decision_node_ids: CudaSlice<u32>,
    d_regrets: CudaSlice<f32>,
    d_strategy: CudaSlice<f32>,
    d_cum_strategy: CudaSlice<f32>,
    d_reach: CudaSlice<f32>,
    d_cfv: CudaSlice<f32>,
    d_initial_weight: CudaSlice<f32>,
    d_sorted_opp_strength: CudaSlice<u16>,
    d_sorted_opp_indices: CudaSlice<u16>,
    d_sorted_player_strength: CudaSlice<u16>,
    d_sorted_player_indices: CudaSlice<u16>,
    d_hand_cards: CudaSlice<u8>,
    d_level_nodes: Vec<CudaSlice<u32>>,
    num_players: u8,
    num_hands: usize,
    num_infosets: usize,
    max_depth: usize,
    level_counts: Vec<i32>,
    iteration: u32,
    regret_floor: f32,
    chance_info: Option<ChanceInfo>,
    d_chance_sorted_strength: Option<CudaSlice<u16>>,
    d_chance_sorted_indices: Option<CudaSlice<u16>>,
    d_chance_prob: Option<CudaSlice<f32>>,
    d_chance_child_ids: Option<CudaSlice<u32>>,
    d_cfv_accum: Option<CudaSlice<f32>>,
    d_below_chance_level_nodes: Vec<Option<CudaSlice<u32>>>,
    below_chance_level_counts: Vec<i32>,
    d_main_level_nodes: Vec<Option<CudaSlice<u32>>>,
    main_level_counts: Vec<i32>,
}

impl GpuVectorCfr {
    pub fn run(
        &mut self,
        num_iterations: u32,
    ) -> Result<(), DriverError> {
        let np = self.num_players as usize;
        let nh = self.num_hands;
        let ni = self.num_infosets as i32;
        let nh_i32 = nh as i32;
        let np_u32 = np as u32;

        for _ in 0..num_iterations {
            self.iteration += 1;
            let params = DcfrParams::new(self.iteration);

            // First traverser: apply discount + compute strategy
            self.launch_compute_strategies(ni, nh_i32, params.alpha_t, params.beta_t, 1)?;

            for traverser in 0..np {
                if traverser > 0 {
                    // Subsequent traversers: recompute strategy from updated regrets (no re-discount)
                    self.launch_compute_strategies(ni, nh_i32, 0.0, 0.0, 0)?;
                }
                self.launch_init_reach(np, nh)?;
                self.launch_top_down(nh_i32, np_u32)?;
                self.launch_bottom_up(traverser as u32, params.gamma_t, nh_i32, np_u32)?;
            }
        }

        self.stream.synchronize()?;
        Ok(())
    }

    fn launch_compute_strategies(&mut self, num_infosets: i32, nh: i32, alpha_t: f32, beta_t: f32, apply_discount: i32) -> Result<(), DriverError> {
        let block = 256;
        let grid = ((num_infosets + block - 1) / block) as u32;

        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block as u32, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            let mut builder = self.stream.launch_builder(&self.vcfr_strategies_fn);
            builder
                .arg(&self.d_regrets)
                .arg(&self.d_strategy)
                .arg(&self.d_decision_node_ids)
                .arg(&self.d_nodes)
                .arg(&self.d_infoset_offsets)
                .arg(&num_infosets)
                .arg(&nh)
                .arg(&alpha_t)
                .arg(&beta_t)
                .arg(&apply_discount);
            builder.launch(cfg)?;
        }
        Ok(())
    }

    fn launch_init_reach(&mut self, np: usize, nh: usize) -> Result<(), DriverError> {
        let nn = self.d_nodes.len();
        let reach_size = nn * np * nh;

        let init = self.stream.clone_dtoh(&self.d_initial_weight)?;
        let mut reach = vec![0.0f32; reach_size];
        for p in 0..np {
            for h in 0..nh {
                reach[p * nh + h] = init[p * nh + h];
            }
        }
        self.stream.memcpy_htod(&reach, &mut self.d_reach)?;
        Ok(())
    }

    fn launch_top_down(&mut self, nh: i32, np: u32) -> Result<(), DriverError> {
        let block = 256;

        for level in 0..=self.max_depth {
            let count = self.level_counts[level];
            if count == 0 { continue; }
            let grid = ((count + block as i32 - 1) / block as i32) as u32;

            let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };

            unsafe {
                let mut builder = self.stream.launch_builder(&self.vcfr_top_down_fn);
                builder
                    .arg(&self.d_level_nodes[level])
                    .arg(&count)
                    .arg(&self.d_nodes)
                    .arg(&self.d_children)
                    .arg(&self.d_strategy)
                    .arg(&self.d_infoset_offsets)
                    .arg(&self.d_reach)
                    .arg(&np)
                    .arg(&nh);
                builder.launch(cfg)?;
            }
        }
        Ok(())
    }

    fn launch_bottom_up(&mut self, traverser: u32, gamma_t: f32, nh: i32, np: u32) -> Result<(), DriverError> {
        let block = 1;
        let regret_floor = self.regret_floor;
        let nh_usize = nh as usize;
        let num_opp = (np - 1) as usize;

        if self.chance_info.is_some() {
            let chance = self.chance_info.as_ref().unwrap();

            self.stream.memset_zeros(self.d_cfv_accum.as_mut().unwrap())?;

            let d_cfv_accum = self.d_cfv_accum.as_ref().unwrap();
            let d_chance_prob = self.d_chance_prob.as_ref().unwrap();
            let d_chance_child_ids = self.d_chance_child_ids.as_ref().unwrap();
            let d_cstr = self.d_chance_sorted_strength.as_ref().unwrap();
            let d_cidx = self.d_chance_sorted_indices.as_ref().unwrap();

            for outcome in 0..chance.num_outcomes {
                let card = chance.remaining_deck[outcome] as usize;
                let offset = card * num_opp * nh_usize;

                let opp_str_view = d_cstr.slice(offset..offset + num_opp * nh_usize);
                let opp_idx_view = d_cidx.slice(offset..offset + num_opp * nh_usize);
                let pl_str_view = d_cstr.slice(offset..offset + nh_usize);
                let pl_idx_view = d_cidx.slice(offset..offset + nh_usize);

                for level in (0..=self.max_depth).rev() {
                    let count = self.below_chance_level_counts[level];
                    if count == 0 { continue; }
                    let grid = count as u32;
                    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };

                    let level_nodes = self.d_below_chance_level_nodes[level].as_ref().unwrap();
                    unsafe {
                        let mut builder = self.stream.launch_builder(&self.vcfr_bottom_up_fn);
                        builder
                            .arg(level_nodes)
                            .arg(&count)
                            .arg(&self.d_nodes)
                            .arg(&self.d_children)
                            .arg(&self.d_contributions)
                            .arg(&self.d_folded_masks)
                            .arg(&self.d_strategy)
                            .arg(&self.d_infoset_offsets)
                            .arg(&self.d_reach)
                            .arg(&self.d_cfv)
                            .arg(&self.d_regrets)
                            .arg(&self.d_cum_strategy)
                            .arg(&self.d_initial_weight)
                            .arg(&opp_str_view)
                            .arg(&opp_idx_view)
                            .arg(&pl_str_view)
                            .arg(&pl_idx_view)
                            .arg(&self.d_hand_cards)
                            .arg(&np)
                            .arg(&nh)
                            .arg(&traverser)
                            .arg(&gamma_t)
                            .arg(&regret_floor);
                        builder.launch(cfg)?;
                    }
                }
            }
        } else {
            for level in (0..=self.max_depth).rev() {
                let count = self.level_counts[level];
                if count == 0 { continue; }
                let grid = count as u32;
                let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };

                unsafe {
                    let mut builder = self.stream.launch_builder(&self.vcfr_bottom_up_fn);
                    builder
                        .arg(&self.d_level_nodes[level])
                        .arg(&count)
                        .arg(&self.d_nodes)
                        .arg(&self.d_children)
                        .arg(&self.d_contributions)
                        .arg(&self.d_folded_masks)
                        .arg(&self.d_strategy)
                        .arg(&self.d_infoset_offsets)
                        .arg(&self.d_reach)
                        .arg(&self.d_cfv)
                        .arg(&self.d_regrets)
                        .arg(&self.d_cum_strategy)
                        .arg(&self.d_initial_weight)
                        .arg(&self.d_sorted_opp_strength)
                        .arg(&self.d_sorted_opp_indices)
                        .arg(&self.d_sorted_player_strength)
                        .arg(&self.d_sorted_player_indices)
                        .arg(&self.d_hand_cards)
                        .arg(&np)
                        .arg(&nh)
                        .arg(&traverser)
                        .arg(&gamma_t)
                        .arg(&regret_floor);
                    builder.launch(cfg)?;
                }
            }
        }
        Ok(())
    }

    pub fn download_regrets(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_regrets)
    }

    pub fn download_cum_strategy(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_cum_strategy)
    }

    pub fn download_strategy(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_strategy)
    }

    pub fn download_cfv(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_cfv)
    }

    pub fn download_reach(&self) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(&self.d_reach)
    }

    pub fn iteration_count(&self) -> u32 {
        self.iteration
    }

    pub fn num_hands(&self) -> usize {
        self.num_hands
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
                    .arg(&1u32);
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
                    .arg(&1.0f32).arg(&self.regret_floor).arg(&d_seeds)
                    .arg(&0u32);
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
                    .arg(&stride)
                    .arg(&1u32);
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
                    .arg(&1.0f32).arg(&self.regret_floor).arg(&d_seeds).arg(&d_peak).arg(&stride)
                    .arg(&0u32);
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
                .arg(&1.0f32).arg(&self.regret_floor).arg(&d_seeds).arg(&d_peak).arg(&stride)
                .arg(&0u32);
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
