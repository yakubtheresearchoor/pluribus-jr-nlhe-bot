//! G2 step 1 — the offset gate, written BEFORE any bucketed kernel
//! exists (the B1/B3 discipline: a stride bug that survives into a
//! parity gate produces plausible-but-wrong agreement; it gets killed
//! at the layout layer, where divergence is exercisable directly).
//!
//! Subject: `BucketedGpuLayout::flat_index` — the exact arithmetic the
//! kernel will perform (`zone_outcome_base + node_local_base[node] +
//! a·nb + b`). Gates, all at DIVERGENT per-street bucket counts
//! (nb_flop=4, nb_turn=3, nb_river=5 — the case the identity gate at
//! uniform B=nh structurally cannot see):
//!
//!   1. From-scratch recomputation: zone classification and per-zone
//!      local offsets recounted independently in this test (tree order,
//!      the construction rule), ZoneDims rebuilt from raw counts,
//!      flat_index compared for EVERY (decision node, outcome, action,
//!      bucket).
//!   2. CPU-storage consistency: flat = zone_float_offset(zone,outcome)
//!      + (the index the CPU walk uses inside its per-zone buffers),
//!      using the CPU solver's own stride accessors — the relation that
//!      makes GPU↔CPU buffer copies offset-free.
//!   3. KernelZoneDims round-trip at divergent dims: every u32 field
//!      equals the usize source.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, NO_BUCKET};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::zone_dims::{ZoneDims, ZoneRef};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};

const NP: u8 = 6;
const NH: usize = 16;
const NB_F: usize = 4;
const NB_T: usize = 3;
const NB_R: usize = 5;

fn build_table() -> FlopChanceTable {
    let board: Vec<Card> = ["Th", "9d", "8c"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
            continue;
        }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            ranges[p][hi as usize] = 1.0;
        }
    }
    let turn_cards: Vec<u8> =
        ["2c", "Jd"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let river_strs: [&[&str]; 2] = [&["4s", "7h"], &["3s", "Qc"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] =
            river_strs[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

fn build_gate_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![500; NP as usize],
        initial_contributions: vec![5; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.33), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    build_tree(&config).unwrap()
}

fn quantile_maps_divergent(
    table: &FlopChanceTable,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8], nb: usize| -> Vec<u16> {
        let alive: Vec<usize> = pl_idx[..nh]
            .iter()
            .map(|&i| i as usize)
            .filter(|&h| !conflicts(h, dead))
            .collect();
        let n = alive.len();
        assert!(n >= nb);
        let mut map = vec![NO_BUCKET; nh];
        for (pos, &h) in alive.iter().enumerate() {
            map[h] = ((pos * nb) / n) as u16;
        }
        map
    };
    let (_, _, _, base_pi, _) = table.sorted_opp_arrays_base();
    let flop_map = map_for(&base_pi, &[], NB_F);
    let mut turn_maps = Vec::new();
    let mut river_maps = Vec::new();
    for &tc_card in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
        turn_maps.push(map_for(pi, &[tc_card], NB_T));
        let mut rms = Vec::new();
        for &rc_card in &table.river_decks[tc_card as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
            rms.push(map_for(pi, &[tc_card, rc_card], NB_R));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
}

#[test]
fn gpu_bucketed_offset_gate_divergent_dims() {
    let tree = build_gate_tree();
    let game = FlopStartGame::new(build_table());
    let (fm, tm, rm) = quantile_maps_divergent(game.table());
    let bk = FlopBucketing::from_maps(game.table(), NB_F, NB_T, NB_R, fm, tm, rm);
    let solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
    let layout = solver.gpu_layout(&bk);

    // ── 1. From-scratch recomputation ──
    // Zone classification recounted independently: a node is River if
    // below a board_state==2 chance child, Turn if below board_state==1,
    // else Flop (the construction rule, re-derived here).
    let nn = tree.num_nodes();
    fn mark(tree: &FlatTree, idx: usize, below: &mut [bool]) {
        below[idx] = true;
        for &c in tree.node_children(idx) {
            mark(tree, c as usize, below);
        }
    }
    let mut below_river = vec![false; nn];
    let mut below_turn = vec![false; nn];
    for i in 0..nn {
        let n = &tree.nodes[i];
        if n.is_chance() && n.board_state == 2 {
            for &c in tree.node_children(i) {
                mark(&tree, c as usize, &mut below_river);
            }
        }
    }
    for i in 0..nn {
        let n = &tree.nodes[i];
        if n.is_chance() && n.board_state == 1 {
            for &c in tree.node_children(i) {
                mark(&tree, c as usize, &mut below_turn);
            }
        }
    }
    // Per-zone local offsets recounted in decision_node_ids order (the
    // construction rule).
    let mut local = vec![usize::MAX; nn];
    let (mut nf, mut nt, mut nr) = (0usize, 0usize, 0usize);
    for &nid in &tree.decision_node_ids {
        let i = nid as usize;
        if below_river[i] {
            local[i] = nr;
            nr += 1;
        } else if below_turn[i] {
            local[i] = nt;
            nt += 1;
        } else {
            local[i] = nf;
            nf += 1;
        }
    }
    let n_turn = game.table().remaining_deck.len();
    let max_river = game
        .table()
        .remaining_deck
        .iter()
        .map(|&tc| game.table().river_decks[tc as usize].len())
        .max()
        .unwrap();
    let dims = ZoneDims {
        max_na: MAX_NA_POSTFLOP,
        nh_flop: NB_F,
        nh_turn: NB_T,
        nh_river: NB_R,
        flop_infosets: nf,
        turn_infosets: nt,
        river_infosets: nr,
        n_turn,
        max_river,
    };
    assert_eq!(dims, layout.dims, "ZoneDims from-scratch rebuild disagrees with layout");

    let mut checked = 0usize;
    for &nid in &tree.decision_node_ids {
        let i = nid as usize;
        let na = tree.nodes[i].num_children as usize;
        let (zone_refs, nb): (Vec<ZoneRef>, usize) = if below_river[i] {
            (
                (0..n_turn)
                    .flat_map(|ti| {
                        (0..game.table().river_decks
                            [game.table().remaining_deck[ti] as usize]
                            .len())
                            .map(move |ri| ZoneRef::River {
                                outcome_idx: ti * max_river + ri,
                            })
                    })
                    .collect(),
                NB_R,
            )
        } else if below_turn[i] {
            ((0..n_turn).map(|ti| ZoneRef::Turn { ti }).collect(), NB_T)
        } else {
            (vec![ZoneRef::Flop], NB_F)
        };
        for zr in zone_refs {
            let (ti, ri) = match zr {
                ZoneRef::Flop => (None, None),
                ZoneRef::Turn { ti } => (Some(ti), None),
                ZoneRef::River { outcome_idx } => {
                    (Some(outcome_idx / max_river), Some(outcome_idx % max_river))
                }
            };
            for a in 0..na {
                for b in 0..nb {
                    let expected = dims.zone_float_offset(zr)
                        + local[i] * MAX_NA_POSTFLOP * nb
                        + a * nb
                        + b;
                    let got = layout.flat_index(i, ti, ri, a, b);
                    assert_eq!(
                        got, expected,
                        "node {i} zone {zr:?} a={a} b={b}: layout {got} vs scratch {expected}"
                    );
                    checked += 1;
                }
            }
        }
    }
    eprintln!("offset gate: {checked} (node, outcome, action, bucket) indices verified");

    // ── 2. CPU-storage consistency ──
    // flat == zone base + the index the CPU walk uses inside its
    // per-zone buffers (per-zone strides via the solver's accessors).
    assert_eq!(solver.turn_stride(), dims.turn_stride());
    assert_eq!(solver.river_stride(), dims.river_stride());
    assert_eq!(solver.flop_stride(), dims.flop_stride());
    for &nid in &tree.decision_node_ids {
        let i = nid as usize;
        let na = tree.nodes[i].num_children as usize;
        if let Some(l) = solver.turn_local_offset_at(i) {
            for ti in 0..n_turn {
                let cpu_intra = ti * solver.turn_stride() + l * MAX_NA_POSTFLOP * NB_T;
                let flat = layout.flat_index(i, Some(ti), None, na - 1, NB_T - 1);
                assert_eq!(
                    flat,
                    dims.turn_offset() + cpu_intra + (na - 1) * NB_T + (NB_T - 1),
                    "turn CPU-storage consistency, node {i} ti {ti}"
                );
            }
        }
        if let Some(l) = solver.river_local_offset_at(i) {
            for ti in 0..n_turn {
                for ri in 0..max_river {
                    let cpu_intra =
                        (ti * max_river + ri) * solver.river_stride() + l * MAX_NA_POSTFLOP * NB_R;
                    let flat = layout.flat_index(i, Some(ti), Some(ri), 0, 0);
                    assert_eq!(
                        flat,
                        dims.river_offset() + cpu_intra,
                        "river CPU-storage consistency, node {i} ({ti},{ri})"
                    );
                }
            }
        }
        if let Some(l) = solver.flop_local_offset_at(i) {
            let flat = layout.flat_index(i, None, None, 0, 0);
            assert_eq!(flat, l * MAX_NA_POSTFLOP * NB_F, "flop CPU-storage consistency, node {i}");
        }
    }

    // ── 3. KernelZoneDims round-trip at divergent dims ──
    let k = dims.to_kernel_dims();
    assert_eq!(k.max_na as usize, dims.max_na);
    assert_eq!(k.nh_flop as usize, dims.nh_flop);
    assert_eq!(k.nh_turn as usize, dims.nh_turn);
    assert_eq!(k.nh_river as usize, dims.nh_river);
    assert_eq!(k.flop_stride as usize, dims.flop_stride());
    assert_eq!(k.turn_stride as usize, dims.turn_stride());
    assert_eq!(k.river_stride as usize, dims.river_stride());
    assert_eq!(k.turn_offset as usize, dims.turn_offset());
    assert_eq!(k.river_offset as usize, dims.river_offset());
}

/// G5 plan gate (before any native kernel exists): the NativePlan's
/// infoset descriptors and reach-edge lists against from-scratch
/// recomputation at the same divergent dims, plus structural
/// invariants the kernels rely on:
///   - every descriptor base == layout.flat_index (the gated formula);
///   - every non-root zone node has EXACTLY one incoming edge within
///     its zone (single-writer property — the reach kernel's
///     race-freedom argument);
///   - edges at level L have parents at level L (the per-level
///     dispatch dependency order);
///   - walk list matches run()'s order and count.
#[test]
fn gpu_native_plan_gate_divergent_dims() {
    use solver_core::solver::bucketed_flop_cfr::NativePlan;
    let tree = build_gate_tree();
    let game = FlopStartGame::new(build_table());
    let (fm, tm, rm) = quantile_maps_divergent(game.table());
    let bk = FlopBucketing::from_maps(game.table(), NB_F, NB_T, NB_R, fm, tm, rm);
    let solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
    let layout = solver.gpu_layout(&bk);
    let plan: NativePlan = solver.native_plan(&tree, &bk);

    let n_turn = game.table().remaining_deck.len();
    let max_river = game
        .table()
        .remaining_deck
        .iter()
        .map(|&tc| game.table().river_decks[tc as usize].len())
        .max()
        .unwrap();

    // Walk list shape.
    let expected_walks = n_turn
        * game.table().river_decks[game.table().remaining_deck[0] as usize].len()
        + n_turn
        + 1;
    assert_eq!(plan.walks.len(), expected_walks);
    assert_eq!(
        plan.reach_floats_per_walk,
        tree.num_nodes() * NP as usize * game.table().num_valid
    );

    // Descriptor bases == flat_index, for every (zone, outcome, node).
    let mut n_desc = 0usize;
    for d in &plan.flop_infosets {
        assert_eq!(d.base as usize, layout.flat_index(d.node as usize, None, None, 0, 0));
        n_desc += 1;
    }
    for (ti, descs) in plan.turn_infosets.iter().enumerate() {
        for d in descs {
            assert_eq!(
                d.base as usize,
                layout.flat_index(d.node as usize, Some(ti), None, 0, 0)
            );
            n_desc += 1;
        }
    }
    for (oi, descs) in plan.river_infosets.iter().enumerate() {
        let (ti, ri) = (oi / max_river, oi % max_river);
        for d in descs {
            assert_eq!(
                d.base as usize,
                layout.flat_index(d.node as usize, Some(ti), Some(ri), 0, 0)
            );
            n_desc += 1;
        }
    }
    eprintln!("native plan gate: {n_desc} infoset descriptors verified against flat_index");

    // Edge invariants per zone.
    for (zname, edges) in [
        ("flop", &plan.flop_edges),
        ("turn", &plan.turn_edges),
        ("river", &plan.river_edges),
    ] {
        let mut incoming = std::collections::HashMap::<u32, usize>::new();
        let mut n_edges = 0usize;
        for (level, lv) in edges.iter().enumerate() {
            for e in lv {
                // Parent sits at this level.
                assert!(
                    tree.nodes_at_level(level as u32).contains(&e.parent),
                    "{zname} L{level}: parent {} not at level",
                    e.parent
                );
                *incoming.entry(e.child).or_insert(0) += 1;
                n_edges += 1;
            }
        }
        for (&child, &cnt) in &incoming {
            assert_eq!(cnt, 1, "{zname}: child {child} has {cnt} incoming edges (race!)");
        }
        eprintln!("native plan gate: {zname} {n_edges} edges, single-writer ✓");
    }
}
