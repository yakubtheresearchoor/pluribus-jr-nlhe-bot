use crate::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use crate::solver::game::GameSpec;
use crate::solver::showdown::side_pot_showdown_cfv;
use crate::tree::flat::{FlatTree, MAX_NA, VCFR_NO_INFOSET};

const UNUSED: usize = usize::MAX;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Zone { Flop, Turn, River }

pub struct DcfrParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DcfrParams {
    pub fn new(current_iteration: u32) -> Self {
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

/// Per-outcome regrets for flop-start games.
///
/// Each chance outcome gets its own regret/strategy/cum_strategy storage for
/// nodes below that chance event. This matches b1nary's architecture where each
/// expanded subtree node has independent regret storage.
///
/// Layout:
///   Flop zone: one copy (shared across all outcomes)
///   Turn zone: one copy per turn card (n_turn copies)
///   River zone: one copy per (turn_card, river_card) pair (n_turn × n_river copies)
pub struct FlopStartVectorCfr {
    num_players: u8,
    nh: usize,

    // ── Zone classification ──
    zones: Vec<Zone>,                          // [nn]
    flop_zone_nodes: Vec<Vec<u32>>,            // per level
    turn_zone_nodes: Vec<Vec<u32>>,            // per level (template)
    river_zone_nodes: Vec<Vec<u32>>,           // per level (template)

    // ── Per-zone infoset bookkeeping ──
    // For each template decision node, its local infoset offset within its zone.
    // UNUSED if the node is not in this zone or not a decision node.
    flop_local_offset: Vec<usize>,             // [nn]
    turn_local_offset: Vec<usize>,             // [nn]
    river_local_offset: Vec<usize>,            // [nn]

    flop_infoset_count: usize,
    turn_infoset_count: usize,
    river_infoset_count: usize,

    // ── Per-zone strides ──
    flop_stride: usize,    // flop_infoset_count × MAX_NA × nh
    turn_stride: usize,    // turn_infoset_count × MAX_NA × nh
    river_stride: usize,   // river_infoset_count × MAX_NA × nh

    // ── Per-outcome regret storage ──
    // Flop: [flop_stride]
    regrets_flop: Vec<f32>,
    strategy_flop: Vec<f32>,
    cum_strategy_flop: Vec<f32>,

    // Turn: [n_turn × turn_stride]
    regrets_turn: Vec<f32>,
    strategy_turn: Vec<f32>,
    cum_strategy_turn: Vec<f32>,

    // River: [n_turn × max_n_river × river_stride]
    regrets_river: Vec<f32>,
    strategy_river: Vec<f32>,
    cum_strategy_river: Vec<f32>,

    // ── Chance node bookkeeping ──
    river_chance_children: Vec<u32>,
    turn_chance_children: Vec<u32>,

    // ── Chance outcome counts ──
    n_turn: usize,
    max_n_river: usize,
    // Per-turn-card river deck sizes (for indexing)
    river_deck_sizes: Vec<usize>,  // [52]

    // ── Per-turn-chance-subtree node lists ──
    // For each turn chance child, the list of turn-zone nodes below it.
    // This prevents off-path regret updates during bottom_up_zone.
    turn_subtree_nodes: Vec<Vec<u32>>,
    // For each river chance child, the list of river-zone nodes below it.
    river_subtree_nodes: Vec<Vec<u32>>,
    // Mapping from turn-zone node to which turn chance subtree it belongs to
    // (index into turn_chance_children)
    turn_node_to_subtree: Vec<usize>,  // [nn]
    // Mapping from river-zone node to which river chance subtree it belongs to
    river_node_to_subtree: Vec<usize>, // [nn]

    iteration: u32,
    regret_floor: f32,
    debug: bool,
    vanilla_mode: bool,  // If true, use alpha=beta=gamma=1
}

fn mark_descendants(tree: &FlatTree, node_idx: usize, below: &mut [bool]) {
    below[node_idx] = true;
    for &child in tree.node_children(node_idx) {
        mark_descendants(tree, child as usize, below);
    }
}

impl FlopStartVectorCfr {
    pub fn new(tree: &FlatTree, table: &FlopChanceTable) -> Self {
        let nn = tree.num_nodes();
        let max_depth = tree.max_depth as usize;

        // ── Classify zones ──
        let mut below_river = vec![false; nn];
        let mut below_turn = vec![false; nn];
        let mut river_cc = Vec::new();
        let mut turn_cc = Vec::new();

        for i in 0..nn {
            let n = &tree.nodes[i];
            if n.is_chance() && n.board_state == 2 {
                // River chance node
                for &c in tree.node_children(i) {
                    river_cc.push(c);
                    mark_descendants(tree, c as usize, &mut below_river);
                }
            }
        }
        for i in 0..nn {
            let n = &tree.nodes[i];
            if n.is_chance() && n.board_state == 1 {
                // Turn chance node
                for &c in tree.node_children(i) {
                    turn_cc.push(c);
                    mark_descendants(tree, c as usize, &mut below_turn);
                }
            }
        }

        let zones: Vec<Zone> = (0..nn)
            .map(|i| {
                if below_river[i] { Zone::River }
                else if below_turn[i] { Zone::Turn }
                else { Zone::Flop }
            })
            .collect();

        // ── Per-zone node lists ──
        let mut river_zn = vec![vec![]; max_depth + 1];
        let mut turn_zn = vec![vec![]; max_depth + 1];
        let mut flop_zn = vec![vec![]; max_depth + 1];

        for level in 0..=max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                match zones[nid as usize] {
                    Zone::River => river_zn[level].push(nid),
                    Zone::Turn => turn_zn[level].push(nid),
                    Zone::Flop => flop_zn[level].push(nid),
                }
            }
        }

        // ── Per-zone infoset offsets ──
        let mut flop_local_offset = vec![UNUSED; nn];
        let mut turn_local_offset = vec![UNUSED; nn];
        let mut river_local_offset = vec![UNUSED; nn];
        let mut flop_count = 0usize;
        let mut turn_count = 0usize;
        let mut river_count = 0usize;

        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            match zones[idx] {
                Zone::Flop => {
                    flop_local_offset[idx] = flop_count;
                    flop_count += 1;
                }
                Zone::Turn => {
                    turn_local_offset[idx] = turn_count;
                    turn_count += 1;
                }
                Zone::River => {
                    river_local_offset[idx] = river_count;
                    river_count += 1;
                }
            }
        }

        let nh = table.num_valid;
        let n_turn = table.remaining_deck.len();
        let max_n_river = table.remaining_deck.iter()
            .map(|&tc| table.river_decks[tc as usize].len())
            .max()
            .unwrap_or(0);

        let flop_stride = flop_count * MAX_NA * nh;
        let turn_stride = turn_count * MAX_NA * nh;
        let river_stride = river_count * MAX_NA * nh;

        let mut river_deck_sizes = vec![0usize; 52];
        for &tc in &table.remaining_deck {
            river_deck_sizes[tc as usize] = table.river_decks[tc as usize].len();
        }

        // Compute per-turn-chance-subtree node lists
        let mut turn_subtree_nodes = Vec::new();
        let mut turn_node_to_subtree = vec![UNUSED; nn];
        for (sti, &cc) in turn_cc.iter().enumerate() {
            let mut nodes = Vec::new();
            collect_zone_nodes(tree, cc as usize, &zones, Zone::Turn, &mut nodes);
            for &n in &nodes { turn_node_to_subtree[n as usize] = sti; }
            turn_subtree_nodes.push(nodes);
        }

        // Compute per-river-chance-subtree node lists
        let mut river_subtree_nodes = Vec::new();
        let mut river_node_to_subtree = vec![UNUSED; nn];
        for (sti, &cc) in river_cc.iter().enumerate() {
            let mut nodes = Vec::new();
            collect_zone_nodes(tree, cc as usize, &zones, Zone::River, &mut nodes);
            for &n in &nodes { river_node_to_subtree[n as usize] = sti; }
            river_subtree_nodes.push(nodes);
        }

        FlopStartVectorCfr {
            num_players: tree.num_players,
            nh,
            zones,
            flop_zone_nodes: flop_zn,
            turn_zone_nodes: turn_zn,
            river_zone_nodes: river_zn,
            flop_local_offset,
            turn_local_offset,
            river_local_offset,
            flop_infoset_count: flop_count,
            turn_infoset_count: turn_count,
            river_infoset_count: river_count,
            flop_stride,
            turn_stride,
            river_stride,
            regrets_flop: vec![0.0; flop_stride],
            strategy_flop: vec![0.0; flop_stride],
            cum_strategy_flop: vec![0.0; flop_stride],
            regrets_turn: vec![0.0; n_turn * turn_stride],
            strategy_turn: vec![0.0; n_turn * turn_stride],
            cum_strategy_turn: vec![0.0; n_turn * turn_stride],
            regrets_river: vec![0.0; n_turn * max_n_river * river_stride],
            strategy_river: vec![0.0; n_turn * max_n_river * river_stride],
            cum_strategy_river: vec![0.0; n_turn * max_n_river * river_stride],
            river_chance_children: river_cc,
            turn_chance_children: turn_cc,
            n_turn,
            max_n_river,
            river_deck_sizes,
            turn_subtree_nodes,
            river_subtree_nodes,
            turn_node_to_subtree,
            river_node_to_subtree,
            iteration: 0,
            regret_floor: -1e30,
            debug: false,
            vanilla_mode: false,
        }
    }

    // ══════════════════════════════════════════════
    // Storage accessors
    // ══════════════════════════════════════════════

    // ══════════════════════════════════════════════
    // Strategy computation (regret matching)
    // ══════════════════════════════════════════════

    pub fn compute_all_strategies(&mut self, tree: &FlatTree) {
        let nh = self.nh;

        // Flop zone: one strategy from shared regrets
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Flop { continue; }
            let local = self.flop_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let off = local * MAX_NA * nh;
            regret_matching_into(
                &self.regrets_flop[off..off + na * nh],
                na, nh,
                &mut self.strategy_flop[off..off + na * nh],
            );
        }

        // Turn zone: one strategy per turn card
        for tc in 0..self.n_turn {
            let base = tc * self.turn_stride;
            for &nid in &tree.decision_node_ids {
                let idx = nid as usize;
                if self.zones[idx] != Zone::Turn { continue; }
                let local = self.turn_local_offset[idx];
                if local == UNUSED { continue; }
                let na = tree.nodes[idx].num_children as usize;
                let off = base + local * MAX_NA * nh;
                regret_matching_into(
                    &self.regrets_turn[off..off + na * nh],
                    na, nh,
                    &mut self.strategy_turn[off..off + na * nh],
                );
            }
        }

        // River zone: one strategy per (turn, river) pair
        let turn_deck = &self.river_deck_sizes; // dummy, we iterate all n_turn × max_n_river
        for tc in 0..self.n_turn {
            let n_river = self.max_n_river;
            for rc in 0..n_river {
                let base = (tc * self.max_n_river + rc) * self.river_stride;
                for &nid in &tree.decision_node_ids {
                    let idx = nid as usize;
                    if self.zones[idx] != Zone::River { continue; }
                    let local = self.river_local_offset[idx];
                    if local == UNUSED { continue; }
                    let na = tree.nodes[idx].num_children as usize;
                    let off = base + local * MAX_NA * nh;
                    regret_matching_into(
                        &self.regrets_river[off..off + na * nh],
                        na, nh,
                        &mut self.strategy_river[off..off + na * nh],
                    );
                }
            }
        }
    }

    // ══════════════════════════════════════════════
    // Reach probability computation (top-down)
    // ══════════════════════════════════════════════

    /// Compute reach for flop zone (single copy, no outcome dimension).
    pub fn compute_reach_flop(
        &self,
        tree: &FlatTree,
        game: &FlopStartGame,
    ) -> Vec<f32> {
        let nn = tree.num_nodes();
        let np = self.num_players as usize;
        let nh = self.nh;
        let mut reach = vec![0.0f32; nn * np * nh];

        for p in 0..np {
            let w = game.initial_weight(p as u8);
            for h in 0..nh { reach[p * nh + h] = w[h]; }
        }

        if self.debug { eprintln!("  [DEBUG] reach at root: P0={:?} P1={:?}", &reach[0..nh], &reach[nh..2*nh]); }

        for level in 0..=tree.max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                let idx = nid as usize;
                if self.zones[idx] != Zone::Flop { continue; }
                let node = &tree.nodes[idx];

                if node.is_player() {
                    let player = node.player_id as usize;
                    let na = node.num_children as usize;
                    let local = self.flop_local_offset[idx];
                    let sigma = if local != UNUSED {
                        let off = local * MAX_NA * nh;
                        &self.strategy_flop[off..off + na * nh]
                    } else {
                        continue;
                    };

                    if self.debug { eprintln!("  [DEBUG] reach P{} node {} (level {}): P0={:?} P1={:?}",
                        player, idx, level,
                        &reach[idx * np * nh..idx * np * nh + nh],
                        &reach[idx * np * nh + nh..idx * np * nh + 2*nh]); }
                    if self.debug {
                        let ch: Vec<u32> = tree.node_children(idx).to_vec();
                        eprintln!("    node {} has {} children: {:?}", idx, ch.len(), ch);
                        for (a, &child) in ch.iter().enumerate() {
                            eprintln!("      sigma[{}] = {:?}", a, &sigma[a * nh..(a+1) * nh]);
                        }
                    }

                    for (a, &child) in tree.node_children(idx).iter().enumerate() {
                        let src = idx * np * nh;
                        let dst = child as usize * np * nh;
                        for p in 0..np {
                            for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                        }
                        for h in 0..nh { reach[dst + player * nh + h] *= sigma[a * nh + h]; }
                    }
                } else if node.is_chance() {
                    if self.debug { eprintln!("  [DEBUG] CHANCE node {} (level {}): P0={:?} P1={:?}",
                        idx, level,
                        &reach[idx * np * nh..idx * np * nh + nh],
                        &reach[idx * np * nh + nh..idx * np * nh + 2*nh]); }
                    // Propagate reach to all chance children (they'll be in turn zone)
                    for &child in tree.node_children(idx) {
                        let src = idx * np * nh;
                        let dst = child as usize * np * nh;
                        for p in 0..np {
                            for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                        }
                    }
                }
            }
        }
        reach
    }

    /// Compute reach for turn zone, for a specific turn card.
    /// Starts from the turn chance children (which received reach from the flop zone).
    pub fn compute_reach_turn(
        &self,
        tree: &FlatTree,
        tc: usize,
        flop_reach: &[f32],  // reach from flop zone (used to seed turn chance children)
    ) -> Vec<f32> {
        let nn = tree.num_nodes();
        let np = self.num_players as usize;
        let nh = self.nh;
        let mut reach = vec![0.0f32; nn * np * nh];

        // Seed from flop zone reach at turn chance children.
        // compute_reach_flop propagates through chance nodes, so the turn chance children
        // already have reach set in flop_reach.
        for &child_id in &self.turn_chance_children {
            let src = child_id as usize * np * nh;
            for p in 0..np {
                for h in 0..nh { reach[src + p * nh + h] = flop_reach[src + p * nh + h]; }
            }
        }

        // Top-down through turn zone using per-tc strategy
        let tc_base = tc * self.turn_stride;
        for level in 0..=tree.max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                let idx = nid as usize;
                if self.zones[idx] != Zone::Turn { continue; }
                let node = &tree.nodes[idx];

                if node.is_player() {
                    let player = node.player_id as usize;
                    let na = node.num_children as usize;
                    let local = self.turn_local_offset[idx];
                    let sigma = if local != UNUSED {
                        let off = tc_base + local * MAX_NA * nh;
                        &self.strategy_turn[off..off + na * nh]
                    } else {
                        continue;
                    };
                    for (a, &child) in tree.node_children(idx).iter().enumerate() {
                        let src = idx * np * nh;
                        let dst = child as usize * np * nh;
                        for p in 0..np {
                            for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                        }
                        for h in 0..nh { reach[dst + player * nh + h] *= sigma[a * nh + h]; }
                    }
                } else if node.is_chance() {
                    // River chance — propagate to children (they'll be in river zone)
                    for &child in tree.node_children(idx) {
                        let src = idx * np * nh;
                        let dst = child as usize * np * nh;
                        for p in 0..np {
                            for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                        }
                    }
                }
            }
        }
        reach
    }

    /// Compute reach for river zone, for a specific (tc, rc) pair.
    pub fn compute_reach_river(
        &self,
        tree: &FlatTree,
        tc: usize,
        rc: usize,
        turn_reach: &[f32],  // reach from turn zone for this tc
    ) -> Vec<f32> {
        let nn = tree.num_nodes();
        let np = self.num_players as usize;
        let nh = self.nh;
        let mut reach = vec![0.0f32; nn * np * nh];

        // Seed from turn zone reach at river chance children
        for &child_id in &self.river_chance_children {
            let src = child_id as usize * np * nh;
            for p in 0..np {
                for h in 0..nh { reach[src + p * nh + h] = turn_reach[src + p * nh + h]; }
            }
        }

        // Top-down through river zone using per-(tc,rc) strategy
        let rc_base = (tc * self.max_n_river + rc) * self.river_stride;
        for level in 0..=tree.max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                let idx = nid as usize;
                if self.zones[idx] != Zone::River { continue; }
                let node = &tree.nodes[idx];

                if node.is_player() {
                    let player = node.player_id as usize;
                    let na = node.num_children as usize;
                    let local = self.river_local_offset[idx];
                    let sigma = if local != UNUSED {
                        let off = rc_base + local * MAX_NA * nh;
                        &self.strategy_river[off..off + na * nh]
                    } else {
                        continue;
                    };
                    for (a, &child) in tree.node_children(idx).iter().enumerate() {
                        let src = idx * np * nh;
                        let dst = child as usize * np * nh;
                        for p in 0..np {
                            for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
                        }
                        for h in 0..nh { reach[dst + player * nh + h] *= sigma[a * nh + h]; }
                    }
                }
            }
        }
        reach
    }

    // ══════════════════════════════════════════════
    // Main solver loop
    // ══════════════════════════════════════════════

    pub fn run(
        &mut self,
        tree: &FlatTree,
        game: &FlopStartGame,
        num_iterations: u32,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = self.nh;
        let nn = tree.num_nodes();
        let table = game.table();
        let turn_deck = &table.remaining_deck;

        let mut root_cfv_sum = vec![0.0f32; nh];
        let mut count = 0u32;

        for _ in 0..num_iterations {
            let params = if self.vanilla_mode {
                DcfrParams { alpha_t: 1.0, beta_t: 1.0, gamma_t: 1.0 }
            } else {
                DcfrParams::new(self.iteration)
            };
            self.iteration += 1;

            for traverser in 0..np {
                // Sequential: recompute strategies before each traverser
                self.compute_all_strategies(tree);

                if self.debug && traverser == 0 {
                    // Print strategy at root node (node 0)
                    let na = tree.nodes[0].num_children as usize;
                    let off = self.flop_local_offset[0] * MAX_NA * self.nh;
                    let strat = &self.strategy_flop[off..off + na * self.nh];
                    let mut avg = vec![0.0f32; na];
                    for a in 0..na {
                        for h in 0..self.nh { avg[a] += strat[a * self.nh + h]; }
                        avg[a] /= self.nh as f32;
                    }
                    let regret = &self.regrets_flop[off..off + na * self.nh];
                    let mut r_avg = vec![0.0f32; na];
                    for a in 0..na {
                        for h in 0..self.nh { r_avg[a] += regret[a * self.nh + h]; }
                        r_avg[a] /= self.nh as f32;
                    }
                    eprintln!("  [DEBUG] iter={} root strategy avg: {:?}", self.iteration, avg);
                    eprintln!("  [DEBUG] iter={} root regret avg: {:?}", self.iteration, r_avg);
                }

                // Flop zone reach (shared)
                let flop_reach = self.compute_reach_flop(tree, game);

                // Flop zone CFV: aggregate turn outcomes
                let mut flop_cfv = vec![0.0f32; nn * nh];

                for (ti, &tc) in turn_deck.iter().enumerate() {
                    let turn_reach = self.compute_reach_turn(tree, ti, &flop_reach);
                    let n_river = self.river_deck_sizes[tc as usize];

                    // River zone: process each river outcome
                    let mut river_cfv_accum = vec![0.0f32; nn * nh];
                    for ri in 0..n_river {
                        let river_reach = self.compute_reach_river(tree, ti, ri, &turn_reach);
                        let mut cfv = vec![0.0f32; nn * nh];

                        self.bottom_up_zone(
                            tree, table, traverser as u8, &river_reach, &mut cfv,
                            Zone::River, Some(ti), Some(ri),
                            &params,
                        );

                        // Weight by river chance probability and accumulate
                        for &child_id in &self.river_chance_children {
                            for h in 0..nh {
                                let cp = table.chance_probability_river(tc, ri, h);
                                river_cfv_accum[child_id as usize * nh + h] +=
                                    cp * cfv[child_id as usize * nh + h];
                            }
                        }
                    }

                    // Turn zone: seed CFV from river accumulation
                    let mut turn_cfv = vec![0.0f32; nn * nh];
                    for &child_id in &self.river_chance_children {
                        for h in 0..nh {
                            turn_cfv[child_id as usize * nh + h] =
                                river_cfv_accum[child_id as usize * nh + h];
                        }
                    }

                    self.bottom_up_zone(
                        tree, table, traverser as u8, &turn_reach, &mut turn_cfv,
                        Zone::Turn, Some(ti), None,
                        &params,
                    );

                    // Accumulate at turn chance children weighted by turn chance probability
                    for &child_id in &self.turn_chance_children {
                        for h in 0..nh {
                            let cp = table.chance_probability_turn(ti, h);
                            flop_cfv[child_id as usize * nh + h] +=
                                cp * turn_cfv[child_id as usize * nh + h];
                        }
                    }
                }

                // Flop zone: direct regret update (no accumulation)
                self.bottom_up_zone(
                    tree, table, traverser as u8, &flop_reach, &mut flop_cfv,
                    Zone::Flop, None, None,
                    &params,
                );

                if traverser == 0 {
                    for h in 0..nh { root_cfv_sum[h] += flop_cfv[h]; }
                    count += 1;
                }

                if self.debug {
                    let fc_min = flop_cfv[0..nh].iter().cloned().fold(f32::INFINITY, f32::min);
                    let fc_max = flop_cfv[0..nh].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!("  [DEBUG] iter={} trav={} root_cfv min={:.4} max={:.4}",
                        self.iteration, traverser, fc_min, fc_max);
                }
            }
        }

        for h in 0..nh { root_cfv_sum[h] /= count as f32; }
        root_cfv_sum
    }

    // ══════════════════════════════════════════════
    // Bottom-up zone processing
    // ══════════════════════════════════════════════

    #[allow(clippy::too_many_arguments)]
    pub fn bottom_up_zone(
        &mut self,
        tree: &FlatTree,
        table: &FlopChanceTable,
        traverser: u8,
        reach: &[f32],
        cfv: &mut [f32],
        zone: Zone,
        tc: Option<usize>,
        rc: Option<usize>,
        params: &DcfrParams,
    ) {
        let nh = self.nh;
        let np = self.num_players as usize;
        let max_depth = tree.max_depth as usize;
        let num_opp = np - 1;

        // Collect node IDs for this zone/level before borrowing self mutably
        // (zone_nodes borrows self, so we must extract the data before get_mut_slices)
        let zone_node_ids: Vec<Vec<u32>> = match zone {
            Zone::Flop => self.flop_zone_nodes.clone(),
            Zone::Turn => self.turn_zone_nodes.clone(),
            Zone::River => self.river_zone_nodes.clone(),
        };
        let regret_floor = self.regret_floor;

        for level in (0..=max_depth).rev() {
            for &nid in zone_node_ids.get(level).into_iter().flatten() {
                let idx = nid as usize;
                let node = &tree.nodes[idx];

                if node.is_terminal() {
                    // Terminal evaluation
                    let node_reach_base = idx * np * nh;
                    let c_t = tree.get_contribution(idx, traverser);
                    let fold_mask = tree.get_folded_mask(idx);

                    // Determine board cards for this zone
                    let board_cards: Vec<u8> = match zone {
                        Zone::River => {
                            let ti = tc.unwrap();
                            let ri = rc.unwrap();
                            let tc_card = table.remaining_deck[ti];
                            let rc_card = table.river_decks[tc_card as usize][ri];
                            vec![tc_card, rc_card]
                        }
                        Zone::Turn => {
                            let ti = tc.unwrap();
                            vec![table.remaining_deck[ti]]
                        }
                        Zone::Flop => vec![],
                    };

                    // Filter opponent reach to exclude hands conflicting with turn/river cards
                    let filtered_opp: Vec<Vec<f32>> = (0..np)
                        .filter(|&p| p != traverser as usize)
                        .map(|p| {
                            let raw = &reach[node_reach_base + p * nh..node_reach_base + (p + 1) * nh];
                            if board_cards.is_empty() {
                                raw.to_vec()
                            } else {
                                let mut filtered = raw.to_vec();
                                for h in 0..nh {
                                    if filtered[h] != 0.0 {
                                        let c1 = table.hand_cards[h * 2];
                                        let c2 = table.hand_cards[h * 2 + 1];
                                        for &bc in &board_cards {
                                            if c1 == bc || c2 == bc {
                                                filtered[h] = 0.0;
                                                break;
                                            }
                                        }
                                    }
                                }
                                filtered
                            }
                        })
                        .collect();
                    let opp_reach: Vec<&[f32]> = filtered_opp.iter().map(|v| v.as_slice()).collect();

                    let contributions: Vec<i32> = (0..np)
                        .map(|p| tree.get_contribution(idx, p as u8))
                        .collect();

                    // Select sorted arrays based on zone
                    let (opp_str, opp_idx, pl_str, pl_idx) = match zone {
                        Zone::River => {
                            let ti = tc.unwrap();
                            let ri = rc.unwrap();
                            let tc_card = table.remaining_deck[ti];
                            let rc_card = table.river_decks[tc_card as usize][ri];
                            table.river_sorted_arrays(tc_card, rc_card)
                        }
                        Zone::Turn => {
                            let ti = tc.unwrap();
                            let tc_card = table.remaining_deck[ti];
                            table.turn_sorted_arrays(tc_card)
                        }
                        Zone::Flop => {
                            let (os, oi, ps, pi, _) = table.sorted_opp_arrays_base();
                            let cfv_out = side_pot_showdown_cfv(
                                &opp_reach, &table.hand_cards, nh,
                                &os, &oi, &ps, &pi,
                                &contributions, fold_mask, traverser as usize, self.num_players,
                                tree.starting_pot,
                            );
                            let nc = table.num_combinations as f32;
                            if nc > 0.0 {
                                for h in 0..nh { cfv[idx * nh + h] = cfv_out[h] / nc; }
                            } else {
                                for h in 0..nh { cfv[idx * nh + h] = cfv_out[h]; }
                            }
                            continue;
                        }
                    };

                    let cfv_out = side_pot_showdown_cfv(
                        &opp_reach, &table.hand_cards, nh,
                        opp_str, opp_idx, pl_str, pl_idx,
                        &contributions, fold_mask, traverser as usize, self.num_players,
                        tree.starting_pot,
                    );

                    let nc = table.num_combinations as f32;
                    if nc > 0.0 {
                        for h in 0..nh { cfv[idx * nh + h] = cfv_out[h] / nc; }
                    } else {
                        for h in 0..nh { cfv[idx * nh + h] = cfv_out[h]; }
                    }

                } else if node.is_chance() {
                    // Chance node: sum child CFVs
                    for h in 0..nh { cfv[idx * nh + h] = 0.0; }
                    for &child in tree.node_children(idx) {
                        for h in 0..nh {
                            cfv[idx * nh + h] += cfv[child as usize * nh + h];
                        }
                    }

                } else {
                    // Player decision node
                    let owner = node.player_id as usize;
                    let na = node.num_children as usize;
                    let (regret_slice, strategy_slice, cum_slice) = self.get_mut_slices(zone, idx, tc, rc, na);

                    let mut cfv_avg = vec![0.0f32; nh];

                    if owner == traverser as usize {
                        // Traverser node: compute CFV and update regrets
                        for a in 0..na {
                            let child = tree.node_children(idx)[a] as usize;
                            for h in 0..nh {
                                cfv_avg[h] += strategy_slice[a * nh + h] * cfv[child * nh + h];
                            }
                        }

                        // Regret update with DCFR discount
                        for a in 0..na {
                            let child = tree.node_children(idx)[a] as usize;
                            for h in 0..nh {
                                let inst_regret = cfv[child * nh + h] - cfv_avg[h];
                                let ridx = a * nh + h;

                                let old_r = regret_slice[ridx];
                                let coef = if old_r >= 0.0 { params.alpha_t } else { params.beta_t };
                                regret_slice[ridx] = coef * old_r + inst_regret;
                                if regret_slice[ridx] < regret_floor {
                                    regret_slice[ridx] = regret_floor;
                                }

                                cum_slice[ridx] = params.gamma_t * cum_slice[ridx]
                                    + strategy_slice[a * nh + h];
                            }
                        }
                    } else {
                        // Non-traverser: just aggregate CFV
                        for a in 0..na {
                            let child = tree.node_children(idx)[a] as usize;
                            for h in 0..nh {
                                cfv_avg[h] += cfv[child * nh + h];
                            }
                        }
                    }

                    cfv[idx * nh..idx * nh + nh].copy_from_slice(&cfv_avg);

                    if self.debug && idx == 0 {
                        let cfv_min = cfv[0..nh].iter().cloned().fold(f32::INFINITY, f32::min);
                        let cfv_max = cfv[0..nh].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let cfv_sum: f32 = cfv[0..nh].iter().sum();
                        eprintln!("    [DEBUG] root CFV after bottom_up: min={:.4} max={:.4} avg={:.4}",
                            cfv_min, cfv_max, cfv_sum / nh as f32);
                    }
                }
            }
        }
    }

    /// Get mutable slices for regrets, strategy, and cum_strategy for a given node/outcome.
    /// Returns (regret_slice, strategy_slice, cum_strategy_slice), each of length na × nh.
    fn get_mut_slices(
        &mut self,
        zone: Zone,
        node_id: usize,
        tc: Option<usize>,
        rc: Option<usize>,
        na: usize,
    ) -> (&mut [f32], &[f32], &mut [f32]) {
        let len = na * self.nh;
        match zone {
            Zone::Flop => {
                let off = self.flop_local_offset[node_id] * MAX_NA * self.nh;
                let r = &mut self.regrets_flop[off..off + len];
                let s = &self.strategy_flop[off..off + len];
                let c = &mut self.cum_strategy_flop[off..off + len];
                (r, s, c)
            }
            Zone::Turn => {
                let tc = tc.unwrap();
                let off = tc * self.turn_stride + self.turn_local_offset[node_id] * MAX_NA * self.nh;
                let r = &mut self.regrets_turn[off..off + len];
                let s = &self.strategy_turn[off..off + len];
                let c = &mut self.cum_strategy_turn[off..off + len];
                (r, s, c)
            }
            Zone::River => {
                let tc = tc.unwrap();
                let rc = rc.unwrap();
                let off = (tc * self.max_n_river + rc) * self.river_stride
                    + self.river_local_offset[node_id] * MAX_NA * self.nh;
                let r = &mut self.regrets_river[off..off + len];
                let s = &self.strategy_river[off..off + len];
                let c = &mut self.cum_strategy_river[off..off + len];
                (r, s, c)
            }
        }
    }

    pub fn iteration_count(&self) -> u32 { self.iteration }
    pub fn set_iteration(&mut self, i: u32) { self.iteration = i; }
    pub fn num_hands(&self) -> usize { self.nh }
    pub fn flop_infosets(&self) -> usize { self.flop_infoset_count }
    pub fn turn_infosets(&self) -> usize { self.turn_infoset_count }
    pub fn river_infosets(&self) -> usize { self.river_infoset_count }
    pub fn n_turn_outcomes(&self) -> usize { self.n_turn }
    pub fn max_river_outcomes(&self) -> usize { self.max_n_river }
    pub fn set_debug(&mut self, d: bool) { self.debug = d; }
    pub fn set_vanilla_mode(&mut self, v: bool) { self.vanilla_mode = v; }

    pub fn best_response_value_debug(&self, tree: &FlatTree, game: &FlopStartGame, p: u8) -> Vec<f32> {
        self.best_response_value(tree, game, p)
    }
    pub fn strategy_value_debug(&self, tree: &FlatTree, game: &FlopStartGame, p: u8) -> Vec<f32> {
        self.strategy_value(tree, game, p)
    }

    pub fn zones(&self) -> &[Zone] { &self.zones }
    pub fn strategy_flop(&self) -> &[f32] { &self.strategy_flop }
    pub fn cum_strategy_flop(&self) -> &[f32] { &self.cum_strategy_flop }
    pub fn regrets_flop(&self) -> &[f32] { &self.regrets_flop }
    pub fn regrets_flop_mut(&mut self) -> &mut [f32] { &mut self.regrets_flop }
    pub fn cum_strategy_flop_mut(&mut self) -> &mut [f32] { &mut self.cum_strategy_flop }
    pub fn turn_local_offset(&self) -> &[usize] { &self.turn_local_offset }
    pub fn turn_stride(&self) -> usize { self.turn_stride }
    pub fn regrets_turn(&self) -> &[f32] { &self.regrets_turn }
    pub fn regrets_turn_mut(&mut self) -> &mut [f32] { &mut self.regrets_turn }
    pub fn cum_strategy_turn(&self) -> &[f32] { &self.cum_strategy_turn }
    pub fn cum_strategy_turn_mut(&mut self) -> &mut [f32] { &mut self.cum_strategy_turn }
    pub fn river_chance_children(&self) -> &[u32] { &self.river_chance_children }
    pub fn turn_chance_children(&self) -> &[u32] { &self.turn_chance_children }
    pub fn flop_local_offset(&self) -> &[usize] { &self.flop_local_offset }
    pub fn river_local_offset(&self) -> &[usize] { &self.river_local_offset }
    pub fn num_players(&self) -> u8 { self.num_players }
    pub fn strategy_turn(&self) -> &[f32] { &self.strategy_turn }
    pub fn strategy_river(&self) -> &[f32] { &self.strategy_river }
    pub fn regrets_river(&self) -> &[f32] { &self.regrets_river }
    pub fn regrets_river_mut(&mut self) -> &mut [f32] { &mut self.regrets_river }
    pub fn cum_strategy_river(&self) -> &[f32] { &self.cum_strategy_river }
    pub fn cum_strategy_river_mut(&mut self) -> &mut [f32] { &mut self.cum_strategy_river }
    pub fn river_stride(&self) -> usize { self.river_stride }

    /// Return zone nodes organized by level for GPU dispatch.
    /// Returns (river, turn, flop) each as Vec<Vec<u32>> (one Vec per level).
    pub fn zone_nodes_per_level(&self) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<u32>>) {
        (
            self.river_zone_nodes.clone(),
            self.turn_zone_nodes.clone(),
            self.flop_zone_nodes.clone(),
        )
    }

    /// Per-zone decision node IDs (for strategy computation on GPU).
    /// Returns (flop_decision_ids, flop_infoset_offsets, turn_decision_ids, turn_infoset_offsets,
    ///          river_decision_ids, river_infoset_offsets).
    pub fn per_zone_decision_nodes(&self, tree: &FlatTree) -> (
        Vec<u32>, Vec<u32>,  // flop
        Vec<u32>, Vec<u32>,  // turn
        Vec<u32>, Vec<u32>,  // river
    ) {
        let mut flop_ids = Vec::new();
        let mut flop_offs = Vec::new();
        let mut turn_ids = Vec::new();
        let mut turn_offs = Vec::new();
        let mut river_ids = Vec::new();
        let mut river_offs = Vec::new();

        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            match self.zones[idx] {
                Zone::Flop => {
                    let off = self.flop_local_offset[idx];
                    if off != UNUSED {
                        flop_ids.push(nid);
                        flop_offs.push(off as u32);
                    }
                }
                Zone::Turn => {
                    let off = self.turn_local_offset[idx];
                    if off != UNUSED {
                        turn_ids.push(nid);
                        turn_offs.push(off as u32);
                    }
                }
                Zone::River => {
                    let off = self.river_local_offset[idx];
                    if off != UNUSED {
                        river_ids.push(nid);
                        river_offs.push(off as u32);
                    }
                }
            }
        }
        (flop_ids, flop_offs, turn_ids, turn_offs, river_ids, river_offs)
    }

    /// Dump regrets and strategies for ALL decision nodes across ALL outcomes.
    /// Only practical for small trees.
    pub fn dump_all_regrets(&self, tree: &FlatTree, game: &FlopStartGame) {
        let table = game.table();
        let nh = self.nh;
        let na_max = MAX_NA;

        println!("  ── Flop zone regrets/strategy ──");
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Flop { continue; }
            let local = self.flop_local_offset[idx];
            let na = tree.nodes[idx].num_children as usize;
            let off = local * MAX_NA * nh;
            println!("    Node {} (P{}, {} actions):", idx, tree.nodes[idx].player_id, na);
            for a in 0..na {
                let r: Vec<f32> = (0..nh).map(|h| self.regrets_flop[off + a * nh + h]).collect();
                let s: Vec<f32> = (0..nh).map(|h| self.strategy_flop[off + a * nh + h]).collect();
                let c: Vec<f32> = (0..nh).map(|h| self.cum_strategy_flop[off + a * nh + h]).collect();
                println!("      Action {}: regret={:?}", a, r);
                println!("      Action {}: strat ={:?}", a, s);
                println!("      Action {}: cum   ={:?}", a, c);
            }
        }

        println!("  ── Turn zone regrets/strategy (per turn card) ──");
        for ti in 0..self.n_turn {
            let tc_card = table.remaining_deck[ti];
            let base = ti * self.turn_stride;
            println!("    Turn card index {} (card {}):", ti, tc_card);
            for &nid in &tree.decision_node_ids {
                let idx = nid as usize;
                if self.zones[idx] != Zone::Turn { continue; }
                let local = self.turn_local_offset[idx];
                if local == UNUSED { continue; }
                let na = tree.nodes[idx].num_children as usize;
                let off = base + local * MAX_NA * nh;
                println!("      Node {} (P{}, {} actions):", idx, tree.nodes[idx].player_id, na);
                for a in 0..na {
                    let r: Vec<f32> = (0..nh).map(|h| self.regrets_turn[off + a * nh + h]).collect();
                    let s: Vec<f32> = (0..nh).map(|h| self.strategy_turn[off + a * nh + h]).collect();
                    println!("        Action {}: regret={:?}", a, r);
                    println!("        Action {}: strat ={:?}", a, s);
                }
            }
        }

        println!("  ── River zone regrets/strategy (per outcome) ──");
        for ti in 0..self.n_turn {
            let tc_card = table.remaining_deck[ti];
            for ri in 0..self.river_deck_sizes[tc_card as usize] {
                let rc_card = table.river_decks[tc_card as usize][ri];
                let base = (ti * self.max_n_river + ri) * self.river_stride;
                println!("    Outcome tc={} (card {}) rc={} (card {}):",
                    ti, tc_card, ri, rc_card);
                for &nid in &tree.decision_node_ids {
                    let idx = nid as usize;
                    if self.zones[idx] != Zone::River { continue; }
                    let local = self.river_local_offset[idx];
                    if local == UNUSED { continue; }
                    let na = tree.nodes[idx].num_children as usize;
                    let off = base + local * MAX_NA * nh;
                    println!("        Node {} (P{}, {} actions):", idx, tree.nodes[idx].player_id, na);
                    for a in 0..na {
                        let r: Vec<f32> = (0..nh).map(|h| self.regrets_river[off + a * nh + h]).collect();
                        let s: Vec<f32> = (0..nh).map(|h| self.strategy_river[off + a * nh + h]).collect();
                        println!("          Action {}: regret={:?}", a, r);
                        println!("          Action {}: strat ={:?}", a, s);
                    }
                }
            }
        }
    }

    /// Build a flattened cum_strategy array suitable for the existing exploitability() function.
    ///
    /// The layout maps each template decision node to its per-outcome cum_strategy slice.
    /// For flop zone nodes: use the single shared cum_strategy_flop.
    /// For turn zone nodes: average cum_strategy across all turn cards.
    /// For river zone nodes: average cum_strategy across all (turn, river) pairs.
    ///
    /// This produces a "marginal" average strategy that the tree walker can use.
    pub fn flattened_cum_strategy(&self, tree: &FlatTree) -> (Vec<f32>, Vec<usize>) {
        let nn = tree.num_nodes();
        let nh = self.nh;
        let num_infosets = tree.num_infosets as usize;
        let mut flat_cum = vec![0.0f32; num_infosets * MAX_NA * nh];
        let mut offsets = vec![UNUSED; nn];

        // Copy tree's infoset offsets
        for &nid in &tree.decision_node_ids {
            offsets[nid as usize] = tree.infoset_offsets[nid as usize] as usize * MAX_NA * nh;
        }

        // Flop zone: direct copy
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Flop { continue; }
            let local = self.flop_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let src_off = local * MAX_NA * nh;
            let dst_off = offsets[idx];
            flat_cum[dst_off..dst_off + na * nh]
                .copy_from_slice(&self.cum_strategy_flop[src_off..src_off + na * nh]);
        }

        // Turn zone: average across all turn cards
        let n_turn = self.n_turn;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Turn { continue; }
            let local = self.turn_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let dst_off = offsets[idx];

            for tc in 0..n_turn {
                let src_off = tc * self.turn_stride + local * MAX_NA * nh;
                for i in 0..na * nh {
                    flat_cum[dst_off + i] += self.cum_strategy_turn[src_off + i];
                }
            }
            for i in 0..na * nh {
                flat_cum[dst_off + i] /= n_turn as f32;
            }
        }

        // River zone: average across all (turn, river) pairs
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::River { continue; }
            let local = self.river_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let dst_off = offsets[idx];

            let mut count = 0usize;
            for tc in 0..n_turn {
                let n_river = self.river_deck_sizes[
                    /* need turn card, but we don't have the mapping here */
                    0 // placeholder
                ];
                for rc in 0..n_river {
                    let src_off = (tc * self.max_n_river + rc) * self.river_stride + local * MAX_NA * nh;
                    for i in 0..na * nh {
                        flat_cum[dst_off + i] += self.cum_strategy_river[src_off + i];
                    }
                    count += 1;
                }
            }
            if count > 0 {
                for i in 0..na * nh {
                    flat_cum[dst_off + i] /= count as f32;
                }
            }
        }

        (flat_cum, offsets)
    }

    /// Compute exploitability using per-outcome strategy.
    /// This walks the tree, and at each decision node selects the strategy
    /// from the correct per-outcome slot based on the game's current chance state.
    pub fn compute_exploitability(
        &self,
        tree: &FlatTree,
        game: &FlopStartGame,
    ) -> f32 {
        let np = self.num_players as usize;
        let mut total_exploit = 0.0f32;

        for p in 0..np {
            let br_val = self.best_response_value(tree, game, p as u8);
            let sv_val = self.strategy_value(tree, game, p as u8);
            let reach = game.initial_weight(p as u8);
            for h in 0..br_val.len() {
                total_exploit += reach[h] * (br_val[h] - sv_val[h]);
            }
        }

        total_exploit / np as f32
    }


    /// Get strategy for a specific outcome context.
    fn get_strategy_for_outcome(
        &self, node_idx: usize, na: usize,
        tc_idx: Option<usize>, rc_idx: Option<usize>,
    ) -> Vec<f32> {
        let nh = self.nh;
        let zone = self.zones[node_idx];
        let len = na * nh;
        let mut strategy = match zone {
            Zone::Flop => {
                let local = self.flop_local_offset[node_idx];
                if local == UNUSED { return vec![1.0 / na as f32; len]; }
                let off = local * MAX_NA * nh;
                self.cum_strategy_flop[off..off + len].to_vec()
            }
            Zone::Turn => {
                let tc = tc_idx.unwrap_or(0);
                let local = self.turn_local_offset[node_idx];
                if local == UNUSED { return vec![1.0 / na as f32; len]; }
                let off = tc * self.turn_stride + local * MAX_NA * nh;
                self.cum_strategy_turn[off..off + len].to_vec()
            }
            Zone::River => {
                let tc = tc_idx.unwrap_or(0);
                let rc = rc_idx.unwrap_or(0);
                let local = self.river_local_offset[node_idx];
                if local == UNUSED { return vec![1.0 / na as f32; len]; }
                let off = (tc * self.max_n_river + rc) * self.river_stride + local * MAX_NA * nh;
                self.cum_strategy_river[off..off + len].to_vec()
            }
        };
        normalize_strategy(&mut strategy, na, nh);
        strategy
    }

    pub fn best_response_value(&self, tree: &FlatTree, game: &FlopStartGame, target_player: u8) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = game.num_hands(target_player);
        let mut cfreach: Vec<Vec<f32>> = (0..np)
            .map(|p| {
                if p == target_player as usize { vec![1.0; nh] }
                else { game.initial_weight(p as u8) }
            })
            .collect();
        self.walk_br(tree, game, target_player, 0, &mut cfreach, None, None)
    }

    pub fn strategy_value(&self, tree: &FlatTree, game: &FlopStartGame, target_player: u8) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = game.num_hands(target_player);
        let mut cfreach: Vec<Vec<f32>> = (0..np)
            .map(|p| {
                if p == target_player as usize { vec![1.0; nh] }
                else { game.initial_weight(p as u8) }
            })
            .collect();
        self.walk_sv(tree, game, target_player, 0, &mut cfreach, None, None)
    }

    /// Best response walk: at target player nodes, pick the best action.
    fn walk_br(
        &self, tree: &FlatTree, game: &FlopStartGame,
        target_player: u8, node_idx: usize,
        cfreach: &mut [Vec<f32>],
        tc_idx: Option<usize>, rc_idx: Option<usize>,
    ) -> Vec<f32> {
        let node = &tree.nodes[node_idx];

        if node.is_terminal() {
            return game.evaluate_terminal(target_player, node_idx, tree, cfreach);
        }

        if node.is_chance() {
            let children = tree.node_children(node_idx);
            let nh_t = game.num_hands(target_player);
            let mut cfv = vec![0.0f32; nh_t];
            let n_outcomes = game.num_chance_outcomes();

            let child = if !children.is_empty() { children[0] as usize } else { return cfv; };

            for outcome in 0..n_outcomes {
                if outcome > 0 { game.clear_chance_outcome(); }
                let probs: Vec<f32> = (0..nh_t)
                    .map(|h| game.chance_probability(outcome, h)).collect();
                game.set_chance_outcome(outcome);

                let (new_tc, new_rc) = if node.board_state == 1 {
                    (Some(outcome), rc_idx)
                } else if node.board_state == 2 {
                    (tc_idx, Some(outcome))
                } else {
                    (tc_idx, rc_idx)
                };

                let child_cfv = self.walk_br(tree, game, target_player, child, cfreach, new_tc, new_rc);
                for h in 0..nh_t { cfv[h] += probs[h] * child_cfv[h]; }
            }
            game.clear_chance_outcome();
            return cfv;
        }

        let player = node.player_id;
        let num_actions = node.num_children as usize;
        if num_actions == 0 {
            let nh_t = game.num_hands(target_player);
            return vec![0.0f32; nh_t];
        }
        let nh = game.num_hands(player);
        let children = tree.node_children(node_idx);
        let sigma = self.get_strategy_for_outcome(node_idx, num_actions, tc_idx, rc_idx);

        let mut cfv_all: Vec<Vec<f32>> = Vec::with_capacity(num_actions);
        for (a, &child) in children.iter().enumerate() {
            let saved = cfreach[player as usize].clone();
            for h in 0..nh { cfreach[player as usize][h] = saved[h] * sigma[a * nh + h]; }
            cfv_all.push(self.walk_br(tree, game, target_player, child as usize, cfreach, tc_idx, rc_idx));
            cfreach[player as usize] = saved;
        }

        let nh_t = cfv_all[0].len();
        let mut result = vec![0.0f32; nh_t];
        if player == target_player {
            for h in 0..nh_t {
                let mut best = f32::NEG_INFINITY;
                for a in 0..num_actions { best = best.max(cfv_all[a][h]); }
                result[h] = best;
            }
        } else {
            // Non-target player: unweighted sum (sigma already in cfreach)
            for a in 0..num_actions {
                for h in 0..nh_t { result[h] += cfv_all[a][h]; }
            }
        }
        result
    }

    /// Strategy value walk: follow the strategy at all player nodes.
    fn walk_sv(
        &self, tree: &FlatTree, game: &FlopStartGame,
        target_player: u8, node_idx: usize,
        cfreach: &mut [Vec<f32>],
        tc_idx: Option<usize>, rc_idx: Option<usize>,
    ) -> Vec<f32> {
        let node = &tree.nodes[node_idx];

        if node.is_terminal() {
            return game.evaluate_terminal(target_player, node_idx, tree, cfreach);
        }

        if node.is_chance() {
            let children = tree.node_children(node_idx);
            let nh_t = game.num_hands(target_player);
            let mut cfv = vec![0.0f32; nh_t];
            let n_outcomes = game.num_chance_outcomes();

            let child = if !children.is_empty() { children[0] as usize } else { return cfv; };

            for outcome in 0..n_outcomes {
                if outcome > 0 { game.clear_chance_outcome(); }
                let probs: Vec<f32> = (0..nh_t)
                    .map(|h| game.chance_probability(outcome, h)).collect();
                game.set_chance_outcome(outcome);

                let (new_tc, new_rc) = if node.board_state == 1 {
                    (Some(outcome), rc_idx)
                } else if node.board_state == 2 {
                    (tc_idx, Some(outcome))
                } else {
                    (tc_idx, rc_idx)
                };

                let child_cfv = self.walk_sv(tree, game, target_player, child, cfreach, new_tc, new_rc);
                for h in 0..nh_t { cfv[h] += probs[h] * child_cfv[h]; }
            }
            game.clear_chance_outcome();
            return cfv;
        }

        let player = node.player_id;
        let num_actions = node.num_children as usize;
        if num_actions == 0 {
            let nh_t = game.num_hands(target_player);
            return vec![0.0f32; nh_t];
        }
        let nh = game.num_hands(player);
        let children = tree.node_children(node_idx);
        let sigma = self.get_strategy_for_outcome(node_idx, num_actions, tc_idx, rc_idx);

        let mut cfv_all: Vec<Vec<f32>> = Vec::with_capacity(num_actions);
        for (a, &child) in children.iter().enumerate() {
            let saved = cfreach[player as usize].clone();
            for h in 0..nh { cfreach[player as usize][h] = saved[h] * sigma[a * nh + h]; }
            cfv_all.push(self.walk_sv(tree, game, target_player, child as usize, cfreach, tc_idx, rc_idx));
            cfreach[player as usize] = saved;
        }

        let nh_t = cfv_all[0].len();
        let mut cfv = vec![0.0f32; nh_t];
        if player == target_player {
            // Target player: sigma-weighted sum (strategy value)
            for (a, cfv_a) in cfv_all.iter().enumerate() {
                for h in 0..nh_t { cfv[h] += sigma[a * nh + h] * cfv_a[h]; }
            }
        } else {
            // Non-target player: unweighted sum (sigma already in cfreach)
            for a in 0..num_actions {
                for h in 0..nh_t { cfv[h] += cfv_all[a][h]; }
            }
        }
        cfv
    }
}

fn normalize_strategy(strategy: &mut [f32], na: usize, nh: usize) {
    for h in 0..nh {
        let mut total = 0.0f32;
        for a in 0..na { total += strategy[a * nh + h]; }
        if total > 0.0 {
            for a in 0..na { strategy[a * nh + h] /= total; }
        } else {
            let uniform = 1.0 / na as f32;
            for a in 0..na { strategy[a * nh + h] = uniform; }
        }
    }
}

/// Note: regret_matching_into is defined above (it was in the head section).

/// Regret matching: compute strategy from regrets.
fn regret_matching_into(regrets: &[f32], na: usize, nh: usize, strategy: &mut [f32]) {
    for h in 0..nh {
        let mut pos_sum = 0.0f32;
        for a in 0..na {
            let r = regrets[a * nh + h];
            if r > 0.0 { pos_sum += r; }
        }
        if pos_sum > 0.0 {
            for a in 0..na {
                let r = regrets[a * nh + h];
                strategy[a * nh + h] = if r > 0.0 { r / pos_sum } else { 0.0 };
            }
        } else {
            let uniform = 1.0 / na as f32;
            for a in 0..na {
                strategy[a * nh + h] = uniform;
            }
        }
    }
}

/// Collect nodes in a specific zone that are descendants of a given root,
/// stopping at chance nodes that lead into a different zone.
fn collect_zone_nodes(
    tree: &FlatTree, node_idx: usize,
    zones: &[Zone], zone: Zone,
    nodes: &mut Vec<u32>,
) {
    if zones[node_idx] != zone { return; }
    let node = &tree.nodes[node_idx];
    if node.is_player() {
        nodes.push(node_idx as u32);
    }
    for &child in tree.node_children(node_idx) {
        collect_zone_nodes(tree, child as usize, zones, zone, nodes);
    }
}
