use crate::tree::action::BoardState;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

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

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for FlatNode {}

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
}
