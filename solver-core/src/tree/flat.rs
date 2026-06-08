use crate::tree::action::BoardState;

pub const NODE_TYPE_TERMINAL: u8 = 0;
pub const NODE_TYPE_CHANCE: u8 = 1;
pub const NODE_TYPE_PLAYER: u8 = 2;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FlatNode {
    pub node_type: u8,
    pub player_id: u8,
    pub board_state: u8,
    pub num_children: u16,
    pub children_start: u32,
    pub amount: i32,
    pub action_label: u8,
}

impl FlatNode {
    pub fn terminal() -> Self {
        FlatNode {
            node_type: NODE_TYPE_TERMINAL,
            player_id: 0,
            board_state: 0,
            num_children: 0,
            children_start: 0,
            amount: 0,
            action_label: 0,
        }
    }

    pub fn chance(board_state: BoardState) -> Self {
        FlatNode {
            node_type: NODE_TYPE_CHANCE,
            player_id: 0,
            board_state: board_state as u8,
            num_children: 0,
            children_start: 0,
            amount: 0,
            action_label: 0,
        }
    }

    pub fn player(player_id: u8, board_state: BoardState, amount: i32) -> Self {
        FlatNode {
            node_type: NODE_TYPE_PLAYER,
            player_id,
            board_state: board_state as u8,
            num_children: 0,
            children_start: 0,
            amount,
            action_label: 0,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.node_type == NODE_TYPE_TERMINAL
    }

    pub fn is_chance(&self) -> bool {
        self.node_type == NODE_TYPE_CHANCE
    }

    pub fn is_player(&self) -> bool {
        self.node_type == NODE_TYPE_PLAYER
    }
}

// Per-stage action-slot strides. The single MAX_NA constant was split during
// Phase 2 of the per-stage MAX_NA work so preflop can carry a Pluribus-style
// rich action set (16 = fold + call + up to 14 raise sizes) without inflating
// postflop buffers (which are O(MAX_NA^5) sensitive via the K=5 factored
// showdown — see M2's nh^4.84 measurement). build.rs reads THIS file and
// regenerates the Metal header (max_na_generated.metal) so the constants are
// structurally locked together; updating one and not the other is impossible.
//
// Phase 4 will empirically tune MAX_NA_POSTFLOP on a non-trivial mixed-eq
// 6-max config; the current value of 6 is the starting placeholder
// (fold + call + 4 raise sizes = Pluribus postflop blueprint).
pub const MAX_NA_PREFLOP: usize = 16;
pub const MAX_NA_POSTFLOP: usize = 6;

pub const VCFR_NO_INFOSET: u32 = u32::MAX;

pub const ACTION_LABEL_FOLD: u8 = 0;
pub const ACTION_LABEL_CHECK: u8 = 1;
pub const ACTION_LABEL_CALL: u8 = 2;
pub const ACTION_LABEL_BET: u8 = 3;
pub const ACTION_LABEL_RAISE: u8 = 4;
pub const ACTION_LABEL_ALLIN: u8 = 5;
pub const ACTION_LABEL_CHANCE: u8 = 6;

#[derive(Clone, Debug)]
pub struct FlatTree {
    pub nodes: Vec<FlatNode>,
    pub children: Vec<u32>,
    pub contributions: Vec<i32>,
    pub folded_masks: Vec<u16>,
    pub num_players: u8,
    pub starting_pot: i32,
    pub starting_stacks: Vec<i32>,
    pub rake_rate: f64,
    pub rake_cap: f64,
    pub level_offsets: Vec<u32>,
    pub level_nodes: Vec<u32>,
    pub max_depth: u32,
    pub infoset_offsets: Vec<u32>,
    pub decision_node_ids: Vec<u32>,
    pub num_infosets: u32,
}

impl FlatTree {
    pub fn new(
        num_players: u8,
        starting_pot: i32,
        starting_stacks: Vec<i32>,
        rake_rate: f64,
        rake_cap: f64,
    ) -> Self {
        FlatTree {
            nodes: Vec::new(),
            children: Vec::new(),
            contributions: Vec::new(),
            folded_masks: Vec::new(),
            num_players,
            starting_pot,
            starting_stacks,
            rake_rate,
            rake_cap,
            level_offsets: Vec::new(),
            level_nodes: Vec::new(),
            max_depth: 0,
            infoset_offsets: Vec::new(),
            decision_node_ids: Vec::new(),
            num_infosets: 0,
        }
    }

    pub fn node_children(&self, node_idx: usize) -> &[u32] {
        let node = &self.nodes[node_idx];
        if node.num_children == 0 {
            &[]
        } else {
            let start = node.children_start as usize;
            &self.children[start..start + node.num_children as usize]
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn gpu_bytes(&self) -> usize {
        let nodes_bytes = self.nodes.len() * std::mem::size_of::<FlatNode>();
        let children_bytes = self.children.len() * std::mem::size_of::<u32>();
        let contributions_bytes = self.contributions.len() * std::mem::size_of::<i32>();
        nodes_bytes + children_bytes + contributions_bytes
    }

    pub fn get_contribution(&self, node_idx: usize, player_id: u8) -> i32 {
        self.contributions[node_idx * self.num_players as usize + player_id as usize]
    }

    pub fn set_contribution(&mut self, node_idx: usize, player_id: u8, amount: i32) {
        let idx = node_idx * self.num_players as usize + player_id as usize;
        if idx < self.contributions.len() {
            self.contributions[idx] = amount;
        }
    }

    pub fn alloc_node(&mut self, node: FlatNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        for _ in 0..self.num_players {
            self.contributions.push(0);
        }
        self.folded_masks.push(0);
        idx
    }

    pub fn set_folded_mask(&mut self, node_idx: usize, mask: u16) {
        self.folded_masks[node_idx] = mask;
    }

    pub fn get_folded_mask(&self, node_idx: usize) -> u16 {
        self.folded_masks[node_idx]
    }

    pub fn set_children(&mut self, node_idx: usize, child_indices: Vec<u32>) {
        let start = self.children.len() as u32;
        let count = child_indices.len() as u16;
        self.children.extend(child_indices);
        self.nodes[node_idx].children_start = start;
        self.nodes[node_idx].num_children = count;
    }

    pub fn compute_levels(&mut self) {
        if self.nodes.is_empty() {
            return;
        }

        let n = self.nodes.len();
        let mut depths = vec![0u32; n];
        let mut max_depth = 0u32;

        let mut queue = std::collections::VecDeque::new();
        queue.push_back((0usize, 0u32));
        while let Some((idx, depth)) = queue.pop_front() {
            depths[idx] = depth;
            if depth > max_depth {
                max_depth = depth;
            }
            for &child in self.node_children(idx) {
                queue.push_back((child as usize, depth + 1));
            }
        }

        let num_levels = (max_depth + 1) as usize;
        let mut level_counts = vec![0u32; num_levels];
        for &d in &depths {
            level_counts[d as usize] += 1;
        }

        let mut level_offsets = Vec::with_capacity(num_levels + 1);
        level_offsets.push(0);
        let mut cumulative = 0u32;
        for &count in &level_counts {
            cumulative += count;
            level_offsets.push(cumulative);
        }

        let mut level_nodes = vec![0u32; n];
        let mut write_pos = level_offsets.clone();
        for idx in 0..n {
            let level = depths[idx] as usize;
            level_nodes[write_pos[level] as usize] = idx as u32;
            write_pos[level] += 1;
        }

        let mut infoset_offsets = vec![VCFR_NO_INFOSET; n];
        let mut decision_node_ids = Vec::new();
        let mut infoset_counter = 0u32;
        for idx in 0..n {
            if self.nodes[idx].is_player() {
                infoset_offsets[idx] = infoset_counter;
                decision_node_ids.push(idx as u32);
                infoset_counter += 1;
            }
        }

        self.level_offsets = level_offsets;
        self.level_nodes = level_nodes;
        self.max_depth = max_depth;
        self.infoset_offsets = infoset_offsets;
        self.decision_node_ids = decision_node_ids;
        self.num_infosets = infoset_counter;
    }

    pub fn nodes_at_level(&self, level: u32) -> &[u32] {
        if (level as usize) >= self.level_offsets.len() - 1 {
            return &[];
        }
        let start = self.level_offsets[level as usize] as usize;
        let end = self.level_offsets[level as usize + 1] as usize;
        &self.level_nodes[start..end]
    }

    pub fn level_size(&self, level: u32) -> usize {
        if (level as usize) >= self.level_offsets.len() - 1 {
            return 0;
        }
        (self.level_offsets[level as usize + 1] - self.level_offsets[level as usize]) as usize
    }

    pub fn infoset_offset(&self, node_idx: usize) -> u32 {
        self.infoset_offsets[node_idx]
    }

    pub fn max_level_size(&self) -> usize {
        (0..=self.max_depth)
            .map(|l| self.level_size(l))
            .max()
            .unwrap_or(0)
    }

    pub fn level_of(&self, node_idx: usize) -> u32 {
        for level in 0..=self.max_depth {
            let start = self.level_offsets[level as usize] as usize;
            let end = self.level_offsets[level as usize + 1] as usize;
            if (start..end).any(|i| self.level_nodes[i] as usize == node_idx) {
                return level;
            }
        }
        u32::MAX
    }
}
