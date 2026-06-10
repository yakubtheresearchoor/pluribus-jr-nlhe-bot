//! Bucketed flop-start CPU CFR (Phase B3, step 4).
//!
//! Mirror of `FlopStartVectorCfr`'s walk — same zone order, same DCFR
//! discounting, same accumulation order — with the A+B hybrid storage:
//!
//!   - regrets / strategy / cum_strategy live at BUCKET granularity
//!     (per-zone, per-outcome bucket counts via the per-runout maps);
//!   - reach and cfv stay at HAND granularity, so per-hand card removal
//!     (chance probabilities, terminal opponent filtering) is verbatim;
//!   - decision nodes read σ through the hand→bucket map and aggregate
//!     instantaneous regrets per bucket with the FIXED within-bucket
//!     weights (the A+B within-bucket-reach statement,
//!     postflop_buckets.rs module docs);
//!   - showdown terminals aggregate per-hand reach into bucket reach,
//!     call the Design-1 `bucketed_showdown_cfv` (step-3 gate), and
//!     expand the per-bucket CFV back to hands.
//!
//! === Identity degeneracy (B = nh) ===
//!
//! With singleton buckets (identity maps over ALL hands, including
//! board-conflicting ones — exactly what `FlopBucketing::identity`
//! builds) every map lookup is `b == h`, every within-bucket weight is
//! exactly 1.0, every per-bucket aggregation is a single `0.0 + x` add,
//! and the terminal degenerates per the step-3 bit-exact gate. The walk
//! then reproduces `FlopStartVectorCfr` BIT-EXACTLY — gated by
//! `p1_5_4_bucketing_b3_identity_gate` on the Phase 4 wet-deep config.
//!
//! At B < nh, hands conflicting with a runout have no bucket
//! (u16::MAX): their reach is zeroed at the strategy step and their
//! expanded cfv is 0.0. Both are invisible at readout (their chance
//! probability is 0) and neither path executes at the identity gate.
//!
//! Production (B4) scope notes: InMemory river persistence only (bucket
//! strides are B-sized, the 175 GB problem does not exist here); GPU
//! port with parity gates is scheduled between B3 and B4.

use crate::abstraction::postflop_buckets::compute_wtl_for_runout;
use crate::card::Card;
use crate::solver::bucketed_showdown::{bucketed_showdown_cfv, BucketedRunoutTables};
use crate::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use crate::solver::game::GameSpec;
use crate::solver::flop_start_vector_cfr::{
    mark_descendants, regret_matching_into, DcfrParams, Zone,
};
use crate::tree::flat::{FlatTree, MAX_NA_POSTFLOP};

const UNUSED: usize = usize::MAX;
pub const NO_BUCKET: u16 = u16::MAX;

/// Per-flop bucketing consumed by the walk: hand→bucket maps per street
/// and per chance outcome (aligned with `table.remaining_deck` /
/// `table.river_decks` order), fixed within-bucket weights, and the
/// per-runout Design-1 fraction tables.
pub struct FlopBucketing {
    pub nh: usize,
    pub nb_flop: usize,
    pub nb_turn: usize,
    pub nb_river: usize,
    /// flop_map[h] — never NO_BUCKET (no board cards dealt yet).
    pub flop_map: Vec<u16>,
    /// turn_map[ti][h] — NO_BUCKET if h conflicts the turn card (B < nh).
    pub turn_map: Vec<Vec<u16>>,
    /// river_map[ti][ri][h].
    pub river_map: Vec<Vec<Vec<u16>>>,
    /// Normalized within-bucket weights ω̂ (Σ over a bucket's members
    /// = 1). Used for the per-bucket regret aggregation. Exactly 1.0 at
    /// singletons regardless of the raw weights (w/w == 1.0 IEEE).
    pub flop_omega: Vec<f32>,
    pub turn_omega: Vec<Vec<f32>>,
    pub river_omega: Vec<Vec<Vec<f32>>>,
    /// Per-runout terminal fraction tables.
    pub flop_tables: BucketedRunoutTables,
    pub turn_tables: Vec<BucketedRunoutTables>,
    pub river_tables: Vec<Vec<BucketedRunoutTables>>,
}

impl FlopBucketing {
    /// The identity-gate bucketing: B = nh, bucket i = hand i for every
    /// street and outcome — INCLUDING hands that conflict with the dealt
    /// turn/river (the exact solver computes deterministic values for
    /// them too; readout-invisible, but the bit-exact gate sees them).
    /// Strengths come from the same per-runout sorted arrays the exact
    /// terminal consumes; weights are unit.
    pub fn identity(table: &FlopChanceTable) -> Self {
        let nh = table.num_valid;
        let id_map: Vec<u16> = (0..nh as u16).collect();
        let n_turn = table.remaining_deck.len();
        let turn_maps: Vec<Vec<u16>> = (0..n_turn).map(|_| id_map.clone()).collect();
        let river_maps: Vec<Vec<Vec<u16>>> = table
            .remaining_deck
            .iter()
            .map(|&tc| {
                (0..table.river_decks[tc as usize].len())
                    .map(|_| id_map.clone())
                    .collect()
            })
            .collect();
        Self::from_maps(table, nh, nh, nh, id_map, turn_maps, river_maps)
    }

    /// General constructor: caller supplies the per-street, per-outcome
    /// hand→bucket maps (NO_BUCKET for runout-conflicting hands at
    /// B < nh); unit initial weights. Builds the per-runout Design-1
    /// fraction tables from the SAME sorted strength arrays the exact
    /// terminal consumes, and ω̂ = 1/|bucket| within-bucket weights.
    ///
    /// At singleton maps this reproduces `identity` exactly (1.0/1.0
    /// omegas, unit weight sums) — the identity gate runs through this
    /// same code path.
    #[allow(clippy::too_many_arguments)]
    pub fn from_maps(
        table: &FlopChanceTable,
        nb_flop: usize,
        nb_turn: usize,
        nb_river: usize,
        flop_map: Vec<u16>,
        turn_map: Vec<Vec<u16>>,
        river_map: Vec<Vec<Vec<u16>>>,
    ) -> Self {
        let nh = table.num_valid;
        let hands: Vec<(Card, Card)> = (0..nh)
            .map(|h| (table.hand_cards[h * 2], table.hand_cards[h * 2 + 1]))
            .collect();
        let unit_w = vec![1.0f64; nh];

        let tables_for = |pl_str: &[u16], pl_idx: &[u16], map: &[u16], nb: usize| {
            // Rebuild hand_strength exactly as the exact terminal does.
            let mut strength = vec![i32::MIN; nh];
            for si in 0..nh {
                strength[pl_idx[si] as usize] = pl_str[si] as i32;
            }
            let wtl = compute_wtl_for_runout(&hands, &strength, &unit_w, map, nb);
            let mut sums = vec![0.0f64; nb];
            for h in 0..nh {
                if map[h] != NO_BUCKET {
                    sums[map[h] as usize] += 1.0;
                }
            }
            BucketedRunoutTables::from_wtl(&wtl, &sums)
        };
        let omega_for = |map: &[u16], nb: usize| -> Vec<f32> {
            let mut counts = vec![0.0f32; nb];
            for h in 0..nh {
                if map[h] != NO_BUCKET {
                    counts[map[h] as usize] += 1.0;
                }
            }
            (0..nh)
                .map(|h| {
                    if map[h] == NO_BUCKET { 0.0 } else { 1.0 / counts[map[h] as usize] }
                })
                .collect()
        };

        let (_, _, base_ps, base_pi, _) = table.sorted_opp_arrays_base();
        let flop_tables = tables_for(&base_ps[..nh], &base_pi[..nh], &flop_map, nb_flop);
        let flop_omega = omega_for(&flop_map, nb_flop);

        let n_turn = table.remaining_deck.len();
        let mut turn_tables = Vec::with_capacity(n_turn);
        let mut river_tables = Vec::with_capacity(n_turn);
        let mut turn_omega = Vec::with_capacity(n_turn);
        let mut river_omega = Vec::with_capacity(n_turn);
        for (ti, &tc_card) in table.remaining_deck.iter().enumerate() {
            let (_, _, ps, pi) = table.turn_sorted_arrays(tc_card);
            turn_tables.push(tables_for(&ps[..nh], &pi[..nh], &turn_map[ti], nb_turn));
            turn_omega.push(omega_for(&turn_map[ti], nb_turn));

            let river_deck = &table.river_decks[tc_card as usize];
            let mut rt = Vec::with_capacity(river_deck.len());
            let mut ro = Vec::with_capacity(river_deck.len());
            for (ri, &rc_card) in river_deck.iter().enumerate() {
                let (_, _, ps, pi) = table.river_sorted_arrays(tc_card, rc_card);
                rt.push(tables_for(&ps[..nh], &pi[..nh], &river_map[ti][ri], nb_river));
                ro.push(omega_for(&river_map[ti][ri], nb_river));
            }
            river_tables.push(rt);
            river_omega.push(ro);
        }

        FlopBucketing {
            nh,
            nb_flop,
            nb_turn,
            nb_river,
            flop_map,
            turn_map,
            river_map,
            flop_omega,
            turn_omega,
            river_omega,
            flop_tables,
            turn_tables,
            river_tables,
        }
    }
}

/// Lift a bucketed solve's cumulative (average) strategy to per-hand
/// granularity inside an EXACT solver's cum_strategy buffers, so the
/// abstraction is scored in the unbucketed game with the exact terminal
/// (the cross_tree lift precedent at hand granularity instead of action
/// granularity). Per-hand σ_avg is the hand's bucket's σ_avg; cum rows
/// are copied unnormalized (normalization happens at readout, and a
/// bucket row normalizes to the same σ_avg as its copies).
///
/// Runout-conflicting hands (NO_BUCKET) get zero rows → uniform σ at
/// readout; they have zero chance probability, so this is invisible to
/// best response.
pub fn lift_cum_to_exact(
    tree: &FlatTree,
    bucketed: &BucketedFlopCfr,
    bk: &FlopBucketing,
    exact: &mut crate::solver::flop_start_vector_cfr::FlopStartVectorCfr,
) {
    let nh = bk.nh;
    let n_turn = bk.turn_map.len();
    let max_n_river = bucketed.max_river_outcomes();
    assert_eq!(max_n_river, exact.max_river_outcomes());

    for &nid in &tree.decision_node_ids {
        let idx = nid as usize;
        let na = tree.nodes[idx].num_children as usize;

        if let (Some(lb), Some(le)) =
            (bucketed.flop_local_offset_at(idx), exact.flop_local_offset_at(idx))
        {
            let nb = bk.nb_flop;
            let off_b = lb * MAX_NA_POSTFLOP * nb;
            let off_e = le * MAX_NA_POSTFLOP * nh;
            let src = bucketed.cum_strategy_flop();
            for a in 0..na {
                for h in 0..nh {
                    let b = bk.flop_map[h];
                    exact.cum_strategy_flop_mut()[off_e + a * nh + h] = if b == NO_BUCKET {
                        0.0
                    } else {
                        src[off_b + a * nb + b as usize]
                    };
                }
            }
        }

        if let (Some(lb), Some(le)) =
            (bucketed.turn_local_offset_at(idx), exact.turn_local_offset_at(idx))
        {
            let nb = bk.nb_turn;
            for ti in 0..n_turn {
                let off_b = ti * bucketed.turn_stride() + lb * MAX_NA_POSTFLOP * nb;
                let off_e = ti * exact.turn_stride() + le * MAX_NA_POSTFLOP * nh;
                for a in 0..na {
                    for h in 0..nh {
                        let b = bk.turn_map[ti][h];
                        let v = if b == NO_BUCKET {
                            0.0
                        } else {
                            bucketed.cum_strategy_turn()[off_b + a * nb + b as usize]
                        };
                        exact.cum_strategy_turn_mut()[off_e + a * nh + h] = v;
                    }
                }
            }
        }

        if let (Some(lb), Some(le)) =
            (bucketed.river_local_offset_at(idx), exact.river_local_offset_at(idx))
        {
            let nb = bk.nb_river;
            for ti in 0..n_turn {
                for ri in 0..bk.river_map[ti].len() {
                    let off_b = (ti * max_n_river + ri) * bucketed.river_stride()
                        + lb * MAX_NA_POSTFLOP * nb;
                    let off_e = (ti * max_n_river + ri) * exact.river_stride()
                        + le * MAX_NA_POSTFLOP * nh;
                    for a in 0..na {
                        for h in 0..nh {
                            let b = bk.river_map[ti][ri][h];
                            let v = if b == NO_BUCKET {
                                0.0
                            } else {
                                bucketed.cum_strategy_river()[off_b + a * nb + b as usize]
                            };
                            exact.cum_strategy_river_mut()[off_e + a * nh + h] = v;
                        }
                    }
                }
            }
        }
    }
}

/// The bucketed walk. Field-by-field mirror of `FlopStartVectorCfr`
/// with bucket-granularity storage (InMemory river only).
pub struct BucketedFlopCfr {
    num_players: u8,
    nh: usize,

    zones: Vec<Zone>,
    flop_zone_nodes: Vec<Vec<u32>>,
    turn_zone_nodes: Vec<Vec<u32>>,
    river_zone_nodes: Vec<Vec<u32>>,

    flop_local_offset: Vec<usize>,
    turn_local_offset: Vec<usize>,
    river_local_offset: Vec<usize>,

    flop_infoset_count: usize,
    turn_infoset_count: usize,
    river_infoset_count: usize,

    // Per-zone strides at BUCKET granularity (infosets × MAX_NA × nb).
    flop_stride: usize,
    turn_stride: usize,
    river_stride: usize,

    regrets_flop: Vec<f32>,
    strategy_flop: Vec<f32>,
    cum_strategy_flop: Vec<f32>,

    regrets_turn: Vec<f32>,
    strategy_turn: Vec<f32>, // SCRATCH: current tc only
    cum_strategy_turn: Vec<f32>,

    regrets_river: Vec<f32>,
    strategy_river: Vec<f32>, // SCRATCH: current (tc, rc) only
    cum_strategy_river: Vec<f32>,

    river_chance_children: Vec<u32>,
    turn_chance_children: Vec<u32>,

    max_n_river: usize,
    river_deck_sizes: Vec<usize>,

    iteration: u32,
    regret_floor: f32,
}

impl BucketedFlopCfr {
    pub fn new(tree: &FlatTree, table: &FlopChanceTable, bucketing: &FlopBucketing) -> Self {
        assert_eq!(bucketing.nh, table.num_valid, "bucketing built for a different hand universe");
        let nn = tree.num_nodes();
        let max_depth = tree.max_depth as usize;

        // ── Zone classification — verbatim from FlopStartVectorCfr ──
        let mut below_river = vec![false; nn];
        let mut below_turn = vec![false; nn];
        let mut river_cc = Vec::new();
        let mut turn_cc = Vec::new();
        for i in 0..nn {
            let n = &tree.nodes[i];
            if n.is_chance() && n.board_state == 2 {
                for &c in tree.node_children(i) {
                    river_cc.push(c);
                    mark_descendants(tree, c as usize, &mut below_river);
                }
            }
        }
        for i in 0..nn {
            let n = &tree.nodes[i];
            if n.is_chance() && n.board_state == 1 {
                for &c in tree.node_children(i) {
                    turn_cc.push(c);
                    mark_descendants(tree, c as usize, &mut below_turn);
                }
            }
        }
        assert!(
            tree.nodes[0].board_state != 3,
            "BucketedFlopCfr handles flop-start trees only (preflop \
             bucketing is lossless-169 and lives in preflop_cfr)"
        );
        let zones: Vec<Zone> = (0..nn)
            .map(|i| {
                if below_river[i] { Zone::River }
                else if below_turn[i] { Zone::Turn }
                else { Zone::Flop }
            })
            .collect();

        let mut river_zn = vec![vec![]; max_depth + 1];
        let mut turn_zn = vec![vec![]; max_depth + 1];
        let mut flop_zn = vec![vec![]; max_depth + 1];
        for level in 0..=max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                match zones[nid as usize] {
                    Zone::River => river_zn[level].push(nid),
                    Zone::Turn => turn_zn[level].push(nid),
                    Zone::Flop => flop_zn[level].push(nid),
                    Zone::Preflop => unreachable!(),
                }
            }
        }

        let mut flop_local_offset = vec![UNUSED; nn];
        let mut turn_local_offset = vec![UNUSED; nn];
        let mut river_local_offset = vec![UNUSED; nn];
        let mut flop_count = 0usize;
        let mut turn_count = 0usize;
        let mut river_count = 0usize;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            match zones[idx] {
                Zone::Flop => { flop_local_offset[idx] = flop_count; flop_count += 1; }
                Zone::Turn => { turn_local_offset[idx] = turn_count; turn_count += 1; }
                Zone::River => { river_local_offset[idx] = river_count; river_count += 1; }
                Zone::Preflop => unreachable!(),
            }
        }

        let nh = table.num_valid;
        let n_turn = table.remaining_deck.len();
        let max_n_river = table.remaining_deck.iter()
            .map(|&tc| table.river_decks[tc as usize].len())
            .max()
            .unwrap_or(0);

        // Strides from ZoneDims with DIVERGENT per-zone nh — the first
        // consumer of the divergence the struct was built for.
        let dims = crate::solver::zone_dims::ZoneDims {
            max_na: MAX_NA_POSTFLOP,
            nh_flop: bucketing.nb_flop,
            nh_turn: bucketing.nb_turn,
            nh_river: bucketing.nb_river,
            flop_infosets: flop_count,
            turn_infosets: turn_count,
            river_infosets: river_count,
            n_turn,
            max_river: max_n_river,
        };
        let flop_stride = dims.flop_stride();
        let turn_stride = dims.turn_stride();
        let river_stride = dims.river_stride();

        let mut river_deck_sizes = vec![0usize; 52];
        for &tc in &table.remaining_deck {
            river_deck_sizes[tc as usize] = table.river_decks[tc as usize].len();
        }

        BucketedFlopCfr {
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
            strategy_turn: vec![0.0; turn_stride],
            cum_strategy_turn: vec![0.0; n_turn * turn_stride],
            regrets_river: vec![0.0; n_turn * max_n_river * river_stride],
            strategy_river: vec![0.0; river_stride],
            cum_strategy_river: vec![0.0; n_turn * max_n_river * river_stride],
            river_chance_children: river_cc,
            turn_chance_children: turn_cc,
            max_n_river,
            river_deck_sizes,
            iteration: 0,
            regret_floor: -1e30,
        }
    }

    // ── Strategy computation (same regret_matching_into, nb-wide) ──

    pub fn compute_flop_strategy(&mut self, tree: &FlatTree, bk: &FlopBucketing) {
        let nb = bk.nb_flop;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Flop { continue; }
            let local = self.flop_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let off = local * MAX_NA_POSTFLOP * nb;
            regret_matching_into(
                &self.regrets_flop[off..off + na * nb],
                na, nb,
                &mut self.strategy_flop[off..off + na * nb],
            );
        }
    }

    pub fn compute_turn_strategy_for_tc(&mut self, tree: &FlatTree, bk: &FlopBucketing, tc: usize) {
        let nb = bk.nb_turn;
        let base = tc * self.turn_stride;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::Turn { continue; }
            let local = self.turn_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let off_persist = base + local * MAX_NA_POSTFLOP * nb;
            let off_scratch = local * MAX_NA_POSTFLOP * nb;
            regret_matching_into(
                &self.regrets_turn[off_persist..off_persist + na * nb],
                na, nb,
                &mut self.strategy_turn[off_scratch..off_scratch + na * nb],
            );
        }
    }

    pub fn compute_river_strategy_for_pair(
        &mut self, tree: &FlatTree, bk: &FlopBucketing, tc: usize, rc: usize,
    ) {
        let nb = bk.nb_river;
        let base = (tc * self.max_n_river + rc) * self.river_stride;
        for &nid in &tree.decision_node_ids {
            let idx = nid as usize;
            if self.zones[idx] != Zone::River { continue; }
            let local = self.river_local_offset[idx];
            if local == UNUSED { continue; }
            let na = tree.nodes[idx].num_children as usize;
            let off_persist = base + local * MAX_NA_POSTFLOP * nb;
            let off_scratch = local * MAX_NA_POSTFLOP * nb;
            regret_matching_into(
                &self.regrets_river[off_persist..off_persist + na * nb],
                na, nb,
                &mut self.strategy_river[off_scratch..off_scratch + na * nb],
            );
        }
    }

    // ── Reach (per-hand, σ read through the bucket map) ──

    fn propagate_player_reach(
        &self,
        tree: &FlatTree,
        idx: usize,
        sigma: &[f32],
        map: &[u16],
        nb: usize,
        reach: &mut [f32],
    ) {
        let nh = self.nh;
        let np = self.num_players as usize;
        let node = &tree.nodes[idx];
        let player = node.player_id as usize;
        for (a, &child) in tree.node_children(idx).iter().enumerate() {
            let src = idx * np * nh;
            let dst = child as usize * np * nh;
            for p in 0..np {
                for h in 0..nh { reach[dst + p * nh + h] = reach[src + p * nh + h]; }
            }
            for h in 0..nh {
                let b = map[h];
                if b == NO_BUCKET {
                    // Hand conflicts the runout: cannot exist here. Never
                    // taken at the identity gate (identity maps are total).
                    reach[dst + player * nh + h] = 0.0;
                } else {
                    reach[dst + player * nh + h] *= sigma[a * nb + b as usize];
                }
            }
        }
    }

    pub fn compute_reach_flop(
        &self, tree: &FlatTree, game: &FlopStartGame, bk: &FlopBucketing,
    ) -> Vec<f32> {
        let nn = tree.num_nodes();
        let np = self.num_players as usize;
        let nh = self.nh;
        let mut reach = vec![0.0f32; nn * np * nh];
        for p in 0..np {
            let w = game.initial_weight(p as u8);
            for h in 0..nh { reach[p * nh + h] = w[h]; }
        }
        for level in 0..=tree.max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                let idx = nid as usize;
                if self.zones[idx] != Zone::Flop { continue; }
                let node = &tree.nodes[idx];
                if node.is_player() {
                    let na = node.num_children as usize;
                    let local = self.flop_local_offset[idx];
                    if local == UNUSED { continue; }
                    let nb = bk.nb_flop;
                    let off = local * MAX_NA_POSTFLOP * nb;
                    let sigma = self.strategy_flop[off..off + na * nb].to_vec();
                    self.propagate_player_reach(tree, idx, &sigma, &bk.flop_map, nb, &mut reach);
                } else if node.is_chance() {
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

    pub fn compute_reach_turn(
        &self, tree: &FlatTree, bk: &FlopBucketing, ti: usize, flop_reach: &[f32],
    ) -> Vec<f32> {
        let nn = tree.num_nodes();
        let np = self.num_players as usize;
        let nh = self.nh;
        let mut reach = vec![0.0f32; nn * np * nh];
        for &child_id in &self.turn_chance_children {
            let src = child_id as usize * np * nh;
            for p in 0..np {
                for h in 0..nh { reach[src + p * nh + h] = flop_reach[src + p * nh + h]; }
            }
        }
        for level in 0..=tree.max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                let idx = nid as usize;
                if self.zones[idx] != Zone::Turn { continue; }
                let node = &tree.nodes[idx];
                if node.is_player() {
                    let na = node.num_children as usize;
                    let local = self.turn_local_offset[idx];
                    if local == UNUSED { continue; }
                    let nb = bk.nb_turn;
                    let off = local * MAX_NA_POSTFLOP * nb;
                    let sigma = self.strategy_turn[off..off + na * nb].to_vec();
                    self.propagate_player_reach(tree, idx, &sigma, &bk.turn_map[ti], nb, &mut reach);
                } else if node.is_chance() {
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

    pub fn compute_reach_river(
        &self, tree: &FlatTree, bk: &FlopBucketing, ti: usize, ri: usize, turn_reach: &[f32],
    ) -> Vec<f32> {
        let nn = tree.num_nodes();
        let np = self.num_players as usize;
        let nh = self.nh;
        let mut reach = vec![0.0f32; nn * np * nh];
        for &child_id in &self.river_chance_children {
            let src = child_id as usize * np * nh;
            for p in 0..np {
                for h in 0..nh { reach[src + p * nh + h] = turn_reach[src + p * nh + h]; }
            }
        }
        for level in 0..=tree.max_depth {
            for &nid in tree.nodes_at_level(level as u32) {
                let idx = nid as usize;
                if self.zones[idx] != Zone::River { continue; }
                let node = &tree.nodes[idx];
                if node.is_player() {
                    let na = node.num_children as usize;
                    let local = self.river_local_offset[idx];
                    if local == UNUSED { continue; }
                    let nb = bk.nb_river;
                    let off = local * MAX_NA_POSTFLOP * nb;
                    let sigma = self.strategy_river[off..off + na * nb].to_vec();
                    self.propagate_player_reach(
                        tree, idx, &sigma, &bk.river_map[ti][ri], nb, &mut reach,
                    );
                }
            }
        }
        reach
    }

    // ── Main loop — verbatim mirror of FlopStartVectorCfr::run ──

    pub fn run(
        &mut self,
        tree: &FlatTree,
        game: &FlopStartGame,
        bk: &FlopBucketing,
        num_iterations: u32,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = self.nh;
        let nn = tree.num_nodes();
        let table = game.table();
        let turn_deck = &table.remaining_deck;

        let mut root_cfv_sum = vec![0.0f32; nh];
        let mut count = 0u32;

        let mut flop_cfv = vec![0.0f32; nn * nh];
        let mut river_cfv_accum = vec![0.0f32; nn * nh];
        let mut cfv = vec![0.0f32; nn * nh];
        let mut turn_cfv = vec![0.0f32; nn * nh];

        for _ in 0..num_iterations {
            let params = DcfrParams::new(self.iteration);
            self.iteration += 1;

            for traverser in 0..np {
                self.compute_flop_strategy(tree, bk);
                let flop_reach = self.compute_reach_flop(tree, game, bk);

                for &child_id in &self.turn_chance_children {
                    let off = child_id as usize * nh;
                    for h in 0..nh { flop_cfv[off + h] = 0.0; }
                }

                for (ti, &tc) in turn_deck.iter().enumerate() {
                    self.compute_turn_strategy_for_tc(tree, bk, ti);
                    let turn_reach = self.compute_reach_turn(tree, bk, ti, &flop_reach);
                    let n_river = self.river_deck_sizes[tc as usize];

                    for &child_id in &self.river_chance_children {
                        let off = child_id as usize * nh;
                        for h in 0..nh { river_cfv_accum[off + h] = 0.0; }
                    }

                    for ri in 0..n_river {
                        self.compute_river_strategy_for_pair(tree, bk, ti, ri);
                        let river_reach = self.compute_reach_river(tree, bk, ti, ri, &turn_reach);

                        self.bottom_up_zone(
                            tree, table, bk, traverser as u8, &river_reach, &mut cfv,
                            Zone::River, Some(ti), Some(ri), &params,
                        );

                        for &child_id in &self.river_chance_children {
                            for h in 0..nh {
                                let cp = table.chance_probability_river(tc, ri, h);
                                river_cfv_accum[child_id as usize * nh + h] +=
                                    cp * cfv[child_id as usize * nh + h];
                            }
                        }
                    }

                    for &child_id in &self.river_chance_children {
                        for h in 0..nh {
                            turn_cfv[child_id as usize * nh + h] =
                                river_cfv_accum[child_id as usize * nh + h];
                        }
                    }

                    self.bottom_up_zone(
                        tree, table, bk, traverser as u8, &turn_reach, &mut turn_cfv,
                        Zone::Turn, Some(ti), None, &params,
                    );

                    for &child_id in &self.turn_chance_children {
                        for h in 0..nh {
                            let cp = table.chance_probability_turn(ti, h);
                            flop_cfv[child_id as usize * nh + h] +=
                                cp * turn_cfv[child_id as usize * nh + h];
                        }
                    }
                }

                self.bottom_up_zone(
                    tree, table, bk, traverser as u8, &flop_reach, &mut flop_cfv,
                    Zone::Flop, None, None, &params,
                );

                if traverser == 0 {
                    for h in 0..nh { root_cfv_sum[h] += flop_cfv[h]; }
                    count += 1;
                }
            }
        }

        for h in 0..nh { root_cfv_sum[h] /= count as f32; }
        root_cfv_sum
    }

    // ── Bottom-up zone walk ──

    #[allow(clippy::too_many_arguments)]
    pub fn bottom_up_zone(
        &mut self,
        tree: &FlatTree,
        table: &FlopChanceTable,
        bk: &FlopBucketing,
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

        let zone_node_ids: Vec<Vec<u32>> = match zone {
            Zone::Flop => self.flop_zone_nodes.clone(),
            Zone::Turn => self.turn_zone_nodes.clone(),
            Zone::River => self.river_zone_nodes.clone(),
            Zone::Preflop => unreachable!(),
        };
        let regret_floor = self.regret_floor;

        // Zone-wide bucketing selectors.
        let (map, omega, tables, nb): (&[u16], &[f32], &BucketedRunoutTables, usize) = match zone {
            Zone::Flop => (&bk.flop_map, &bk.flop_omega, &bk.flop_tables, bk.nb_flop),
            Zone::Turn => {
                let ti = tc.unwrap();
                (&bk.turn_map[ti], &bk.turn_omega[ti], &bk.turn_tables[ti], bk.nb_turn)
            }
            Zone::River => {
                let ti = tc.unwrap();
                let ri = rc.unwrap();
                (
                    &bk.river_map[ti][ri],
                    &bk.river_omega[ti][ri],
                    &bk.river_tables[ti][ri],
                    bk.nb_river,
                )
            }
            Zone::Preflop => unreachable!(),
        };

        for level in (0..=max_depth).rev() {
            for &nid in zone_node_ids.get(level).into_iter().flatten() {
                let idx = nid as usize;
                let node = &tree.nodes[idx];

                if node.is_terminal() {
                    let node_reach_base = idx * np * nh;
                    let fold_mask = tree.get_folded_mask(idx);

                    // Board-card opponent filtering — verbatim from the
                    // exact walk (per-hand card removal stays exact).
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
                        Zone::Preflop => unreachable!(),
                    };

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

                    // Reduce to bucket reaches (single 0.0 + r add per
                    // hand at singletons — bit-preserving).
                    let num_opp = np - 1;
                    let mut bucket_reach: Vec<Vec<f32>> = vec![vec![0.0f32; nb]; num_opp];
                    for oi in 0..num_opp {
                        for h in 0..nh {
                            let b = map[h];
                            if b == NO_BUCKET { continue; }
                            bucket_reach[oi][b as usize] += filtered_opp[oi][h];
                        }
                    }
                    let bucket_views: Vec<&[f32]> =
                        bucket_reach.iter().map(|v| v.as_slice()).collect();

                    let contributions: Vec<i32> = (0..np)
                        .map(|p| tree.get_contribution(idx, p as u8))
                        .collect();

                    let cfv_out = bucketed_showdown_cfv(
                        &bucket_views,
                        tables,
                        &contributions,
                        fold_mask,
                        traverser as usize,
                        self.num_players,
                        tree.starting_pot,
                        tree.rake_rate as f32,
                        tree.rake_cap as f32,
                        true,
                    );

                    // Expand per-bucket CFV back to hands, with the same
                    // /nc normalization as the exact walk.
                    let nc = table.num_combinations as f32;
                    if nc > 0.0 {
                        for h in 0..nh {
                            let b = map[h];
                            cfv[idx * nh + h] = if b == NO_BUCKET {
                                0.0
                            } else {
                                cfv_out[b as usize] / nc
                            };
                        }
                    } else {
                        for h in 0..nh {
                            let b = map[h];
                            cfv[idx * nh + h] = if b == NO_BUCKET {
                                0.0
                            } else {
                                cfv_out[b as usize]
                            };
                        }
                    }
                } else if node.is_chance() {
                    for h in 0..nh { cfv[idx * nh + h] = 0.0; }
                    for &child in tree.node_children(idx) {
                        for h in 0..nh {
                            cfv[idx * nh + h] += cfv[child as usize * nh + h];
                        }
                    }
                } else {
                    let owner = node.player_id as usize;
                    let na = node.num_children as usize;
                    let children: Vec<usize> =
                        tree.node_children(idx).iter().map(|&c| c as usize).collect();
                    let (regret_slice, strategy_slice, cum_slice) =
                        self.get_mut_slices(zone, idx, tc, rc, na, nb);

                    let mut cfv_avg = vec![0.0f32; nh];

                    if owner == traverser as usize {
                        for (a, &child) in children.iter().enumerate() {
                            for h in 0..nh {
                                let b = map[h];
                                if b == NO_BUCKET { continue; }
                                cfv_avg[h] +=
                                    strategy_slice[a * nb + b as usize] * cfv[child * nh + h];
                            }
                        }

                        // Per-bucket regret aggregation with fixed
                        // within-bucket weights ω̂ (1.0 at singletons:
                        // inst[b] = 0.0 + 1.0 × (cfv_a − cfv_σ)).
                        let mut inst = vec![0.0f32; nb];
                        for (a, &child) in children.iter().enumerate() {
                            for v in inst.iter_mut() { *v = 0.0; }
                            for h in 0..nh {
                                let b = map[h];
                                if b == NO_BUCKET { continue; }
                                inst[b as usize] +=
                                    omega[h] * (cfv[child * nh + h] - cfv_avg[h]);
                            }
                            for b in 0..nb {
                                let ridx = a * nb + b;
                                let old_r = regret_slice[ridx];
                                let coef = if old_r >= 0.0 { params.alpha_t() } else { params.beta_t() };
                                regret_slice[ridx] = coef * old_r + inst[b];
                                if regret_slice[ridx] < regret_floor {
                                    regret_slice[ridx] = regret_floor;
                                }
                                cum_slice[ridx] = params.gamma_t() * cum_slice[ridx]
                                    + strategy_slice[ridx];
                            }
                        }
                    } else {
                        for &child in children.iter() {
                            for h in 0..nh {
                                cfv_avg[h] += cfv[child * nh + h];
                            }
                        }
                    }

                    cfv[idx * nh..idx * nh + nh].copy_from_slice(&cfv_avg);
                }
            }
        }
    }

    fn get_mut_slices(
        &mut self,
        zone: Zone,
        node_id: usize,
        tc: Option<usize>,
        rc: Option<usize>,
        na: usize,
        nb: usize,
    ) -> (&mut [f32], &[f32], &mut [f32]) {
        let len = na * nb;
        match zone {
            Zone::Flop => {
                let off = self.flop_local_offset[node_id] * MAX_NA_POSTFLOP * nb;
                (
                    &mut self.regrets_flop[off..off + len],
                    &self.strategy_flop[off..off + len],
                    &mut self.cum_strategy_flop[off..off + len],
                )
            }
            Zone::Turn => {
                let tc = tc.unwrap();
                let local_off = self.turn_local_offset[node_id] * MAX_NA_POSTFLOP * nb;
                let off_persist = tc * self.turn_stride + local_off;
                (
                    &mut self.regrets_turn[off_persist..off_persist + len],
                    &self.strategy_turn[local_off..local_off + len],
                    &mut self.cum_strategy_turn[off_persist..off_persist + len],
                )
            }
            Zone::River => {
                let tc = tc.unwrap();
                let rc = rc.unwrap();
                let local_off = self.river_local_offset[node_id] * MAX_NA_POSTFLOP * nb;
                let off_persist = (tc * self.max_n_river + rc) * self.river_stride + local_off;
                (
                    &mut self.regrets_river[off_persist..off_persist + len],
                    &self.strategy_river[local_off..local_off + len],
                    &mut self.cum_strategy_river[off_persist..off_persist + len],
                )
            }
            Zone::Preflop => unreachable!(),
        }
    }

    // ── Accessors (gate comparison surface) ──

    pub fn iteration_count(&self) -> u32 { self.iteration }
    pub fn num_hands(&self) -> usize { self.nh }
    pub fn flop_infosets(&self) -> usize { self.flop_infoset_count }
    pub fn turn_infosets(&self) -> usize { self.turn_infoset_count }
    pub fn river_infosets(&self) -> usize { self.river_infoset_count }
    pub fn flop_stride(&self) -> usize { self.flop_stride }
    pub fn turn_stride(&self) -> usize { self.turn_stride }
    pub fn river_stride(&self) -> usize { self.river_stride }
    pub fn max_river_outcomes(&self) -> usize { self.max_n_river }
    pub fn regrets_flop(&self) -> &[f32] { &self.regrets_flop }
    pub fn cum_strategy_flop(&self) -> &[f32] { &self.cum_strategy_flop }
    pub fn regrets_turn(&self) -> &[f32] { &self.regrets_turn }
    pub fn cum_strategy_turn(&self) -> &[f32] { &self.cum_strategy_turn }
    pub fn regrets_river(&self) -> &[f32] { &self.regrets_river }
    pub fn cum_strategy_river(&self) -> &[f32] { &self.cum_strategy_river }
    pub fn flop_local_offset_at(&self, idx: usize) -> Option<usize> {
        let v = self.flop_local_offset[idx];
        if v == UNUSED { None } else { Some(v) }
    }
    pub fn turn_local_offset_at(&self, idx: usize) -> Option<usize> {
        let v = self.turn_local_offset[idx];
        if v == UNUSED { None } else { Some(v) }
    }
    pub fn river_local_offset_at(&self, idx: usize) -> Option<usize> {
        let v = self.river_local_offset[idx];
        if v == UNUSED { None } else { Some(v) }
    }
}
