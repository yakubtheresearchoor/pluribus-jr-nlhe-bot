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
// Phase 4 redo — BANKED on a VERIFIED SHAPE, not on a fully-established rule.
//
// See p1_5_4_phase4_redo_measurement.rs (commits 239be58 → 3ee750e). The
// cross-action-space measurement framework was built, verified
// bit-exact (identity-lift fidelity per zone, K_FULL identity = 0% cost),
// extended with pseudo-harmonic Pluribus-style translation
// (Ganzfried-Sandholm 2013), a self-expl convergence floor (0.005% pot),
// and pot-fraction action matching. Three distinct configs measured:
//
//   wet-deep Th9d8c, stacks=500 (~100bb effective)
//   wet-short Th9d8c, stacks=200 (~33bb effective)
//   dry-deep Ks7d2h rainbow, stacks=500
//
// Empirical finding — VERIFIED SHAPE for blueprint-only deployment:
//   lean = [0.33] + [1.0, 2.0]  (1 bet, both rich raise sizes)
//   measured cost ≤ 0.001% pot across all three configs
//   max actions at any single node = max(check + 1 bet, fold + call + 2 raises) = 4
//   ⇒ MAX_NA_POSTFLOP = 4
//
// Established rule (verified across all 3 configs, 21+ measurements):
//   DEFENSE-COMPLETENESS ON RAISES — lean's raise set must include all
//   rich raise sizes. Configs missing a rich raise size are catastrophic
//   (60-128% pot exploit) somewhere in the stack-depth × board-texture
//   product. Configs with all rich raises are uniformly safe.
//
// Provisional rule (one config, NOT verified as invariant):
//   INCLUDE RICH'S DOMINANT BET SIZE — lean's bet set should contain the
//   most-used bet size in rich's equilibrium (bet 0.33p in our configs).
//   Config 6 [0.66] + [1.0, 2.0] was borderline-exploitable (3.34%) on
//   the dry board despite having all rich raises; the mechanism is
//   over-translation of rich's frequent 0.33p bets onto lean's 0.66p.
//   This clause is FITTED to the single config that broke it, not
//   independently verified — a config where the dominant bet size isn't
//   0.33p would distinguish "include 0.33p" (config-specific) from
//   "include the dominant size" (general). Treat as provisional.
//
// Deferred: MAX_NA_WITH_SEARCH (depth-limited subgame search at runtime
// refines the blueprint; the cross-action-space metric is then a
// conservative upper bound, and the with-search MAX_NA could be smaller).
// Requires implementing depth-limited search to measure.
//
// MAX_NA_POSTFLOP = 4 is THE CONSERVATIVE BOUND, NOT THE FINAL VALUE.
// The final value lands when search is implemented and a with-search
// measurement of cost-of-leaning replaces the standalone-blueprint
// number. Until then, banking 4 is the right choice (passes ≤ 0.001%
// standalone cost on the verified shape across 3 configs), but treat
// "MAX_NA_POSTFLOP = 4 is final" as an open question that closes only
// after search is in the deployment pipeline.
//
// The previous placeholder (MAX_NA_POSTFLOP = 6 = fold + call + 4 raise
// sizes Pluribus-style) is now superseded by this empirical bank. Existing
// research tests that build trees with > 4 actions at a node (e.g. the
// rich K=5 = 4 bets + 2 raises config in the measurement file) will hit
// the per-stage cap assertion at runtime; they're #[ignore]'d and run
// only on demand for further research, so this is non-blocking for CI.
// MAX_NA_PREFLOP — EMPIRICALLY BOOTSTRAPPED 2026-06-10 (see
// p1_5_4_phase4_redo_preflop_bootstrap.rs, two-fidelity frozen-CFV
// measurement). A rich 14-raise 6-max preflop tree (menu 0.5×p..7.0×p,
// 239k nodes) was solved 80 iterations at two fidelities (nh=12 fully-
// shared CFVs / nh=14 pot-bucketed CFVs); reach-weighted σ_avg raise-size
// usage, stable across both (band-robust check):
//
//   0.5×p ≈ 76-79%   1.0×p ≈ 13-14%   1.5×p ≈ 3.6-4.9%   2.0×p ≈ 1.7-2.9%
//   2.5×p ≈ 0.9-1.1% (marginal)        3.0×p..7.0×p all < 0.5%, mostly ≈ 0
//
// VERDICT: NOT TOO LEAN — the equilibrium concentrates ~98% of raise mass
// in 4-5 of the 14 available sizes; 9-10 slots are dead. 16 is over-
// provisioned ~2× (could shrink to ~8 = 5-6 used sizes + fold + call if
// preflop memory ever mattered), which at preflop's CPU-only cost is free
// and is the safe direction. Caveats: bootstrap-grade approximations
// (pairwise fold-terminal blocking, sampled+shared+frozen postflop CFVs —
// see the bootstrap file docstring); menu floor is 0.5×p, so sub-0.5×p
// raise demand is unobserved.
//
// SHRUNK 16→8 (2026-06-14): preflop memory DID matter — the dense stride
// × NUM_PREFLOP_CLASSES gave a ~28.6 GB cap-3 CFR buffer (3 × 9.55 GB),
// 74% of a 38.7 GB machine and an OOM/crash risk. The production ladder
// derives from this const (`saturating_sub(2)`), so 8 ⇒ 6 raise sizes =
// 0.5× / 1.0× / 1.5× / 2.0× / 2.5× / 3.0× pot, carrying ~99% of raise
// mass (top-4 ≈ 97.6%); the dropped 3.5×–7.0× tail is <0.5% each, ≈0.
// Cascades to the buffer stride (preflop_cfr), the tree-builder action
// cap (builder.rs assert), and the codegen'd Metal header (build.rs).
// Dropping those sizes also prunes raise-war branches, so the tree
// itself shrinks — the buffer lands well under the ~14 GB the stride cut
// alone implied. ACTION-ABSTRACTION change: justified by the banked
// usage above, verified by the gates re-run on the leaner tree.
//
// RE-RAISED 8→16 (2026-06-16) for the joint factory; REVERTED to 8
// (2026-06-17) for the preflop-looseness investigation — the deployed v1
// preflop.blob is na=8, so the analysis tree must match it. (joint_solve
// runs fine at either na; flip back to 16 before the full joint run.)
pub const MAX_NA_PREFLOP: usize = 8;
pub const MAX_NA_POSTFLOP: usize = 4;

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
