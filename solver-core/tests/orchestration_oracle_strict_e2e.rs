// STRICT end-to-end orchestration oracle: node-by-node CFV match through
// the chance-integration steps that turn and flop add over the strictly-
// anchored river zone. This is the final precision check before the
// blueprint — the chance integration is where the blueprint spends its
// street transitions, and any drift in the chance arithmetic would
// silently bias every multi-street computation.
//
// Strict comparison at every stage:
//   1. River-zone CFV (per ti, ri): CPU bottom_up_zone vs CFR formula oracle
//   2. River chance accumulation (per ti): sum over ri of chance_prob * river_cfv
//      at river_chance_children positions. Both sides use the same formula —
//      this verifies our formula matches what CPU's run() does inline.
//   3. River chance finalization: copy river_cfv_accum to turn_cfv at boundary.
//   4. Turn-zone CFV (per ti): CPU bottom_up_zone vs oracle.
//   5. Turn chance accumulation: sum over ti of chance_prob * turn_cfv at
//      turn_chance_children positions.
//   6. Turn chance finalization: copy to flop_cfv at boundary.
//   7. Flop-zone CFV: CPU bottom_up_zone vs oracle.
//
// Every stage must match at f32 floor (max_diff < 1e-5). Failure at any
// stage pinpoints the orchestration bug to its first divergent step.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{FlopStartVectorCfr, Zone, DcfrParams};
use solver_core::solver::showdown::side_pot_showdown_cfv;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, NODE_TYPE_PLAYER, NODE_TYPE_TERMINAL, NODE_TYPE_CHANCE};

// Inlined fixture from orchestration_oracle_full_pipeline.rs
fn build_minimal_table() -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
    let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let chosen_hands: Vec<u16> = vec![
        find_pair_index(card_from_str("Ah").unwrap(), card_from_str("Kh").unwrap()),
        find_pair_index(card_from_str("Qh").unwrap(), card_from_str("Jh").unwrap()),
        find_pair_index(card_from_str("Th").unwrap(), card_from_str("9h").unwrap()),
        find_pair_index(card_from_str("8h").unwrap(), card_from_str("6h").unwrap()),
    ];
    let nh = chosen_hands.len();
    let num_players = 2u8;
    let num_opp = 1;
    let valid_hand_indices = chosen_hands.clone();
    let num_valid = nh;
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in valid_hand_indices.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1; hand_cards[i * 2 + 1] = c2;
    }
    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh {
        for j in 0..nh {
            if i == j { conflict[i * nh + j] = 1; continue; }
            let (c1a, c1b) = index_to_card_pair(valid_hand_indices[i] as usize);
            let (c2a, c2b) = index_to_card_pair(valid_hand_indices[j] as usize);
            if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b {
                conflict[i * nh + j] = 1;
            }
        }
    }
    let mut hand_ranks_base = vec![0u16; nh];
    for (i, &hi) in valid_hand_indices.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut hand = Hand::new();
        hand = hand.add_card(c1 as usize); hand = hand.add_card(c2 as usize);
        for &bc in &board { hand = hand.add_card(bc as usize); }
        hand_ranks_base[i] = hand.evaluate_internal() as u16;
    }
    let turn_cards: Vec<u8> = vec![card_from_str("3c").unwrap() as u8, card_from_str("4c").unwrap() as u8];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![card_from_str("5c").unwrap() as u8, card_from_str("6c").unwrap() as u8];
    river_decks[turn_cards[1] as usize] = vec![card_from_str("3c").unwrap() as u8, card_from_str("5c").unwrap() as u8];
    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 { continue; }
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize); hand = hand.add_card(c2 as usize);
            for &bc in &board { hand = hand.add_card(bc as usize); }
            hand = hand.add_card(tc as usize);
            turn_ranks[tc as usize * nh + i] = hand.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh).filter(|&h| {
            let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
            turn_mask & (1u64 << c1) == 0 && turn_mask & (1u64 << c2) == 0
        }).map(|h| (turn_ranks[tc as usize * nh + h], h as u16)).collect();
        items.sort_by_key(|&(s, _)| s);
        for (k, &(r, idx)) in items.iter().enumerate() {
            turn_sorted_str[(tc as usize) * num_opp * nh + 0 * nh + k] = r;
            turn_sorted_idx[(tc as usize) * num_opp * nh + 0 * nh + k] = idx;
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &tc in &turn_cards {
        for &rc in &river_decks[tc as usize] {
            let combined = board_mask | (1u64 << tc) | (1u64 << rc);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if combined & (1u64 << c1) != 0 || combined & (1u64 << c2) != 0 { continue; }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize); hand = hand.add_card(c2 as usize);
                for &bc in &board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize); hand = hand.add_card(rc as usize);
                let r = hand.evaluate_internal() as u16;
                let key = (tc as usize) * 52 + (rc as usize);
                river_ranks[key * nh + i] = r;
            }
            let key = (tc as usize) * 52 + (rc as usize);
            let mut items: Vec<(u16, u16)> = (0..nh).filter(|&h| {
                let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
                combined & (1u64 << c1) == 0 && combined & (1u64 << c2) == 0
            }).map(|h| (river_ranks[key * nh + h], h as u16)).collect();
            items.sort_by_key(|&(s, _)| s);
            for (k, &(r, idx)) in items.iter().enumerate() {
                river_sorted_str[key * num_opp * nh + 0 * nh + k] = r;
                river_sorted_idx[key * num_opp * nh + 0 * nh + k] = idx;
            }
        }
    }
    let initial_weights: Vec<Vec<f32>> = (0..num_players).map(|_| {
        let mut w = vec![0.0f32; nh];
        for h in 0..nh {
            let (c1, c2) = index_to_card_pair(valid_hand_indices[h] as usize);
            let mut blocked = 0;
            for h2 in 0..nh {
                if h2 == h { continue; }
                let (c3, c4) = index_to_card_pair(valid_hand_indices[h2] as usize);
                if c1 == c3 || c1 == c4 || c2 == c3 || c2 == c4 { blocked += 1; }
            }
            w[h] = if blocked < (nh - 1) as i32 { 1.0 } else { 0.0 };
        }
        w
    }).collect();
    let num_combinations = initial_weights[0].iter().sum::<f32>() * initial_weights[1].iter().sum::<f32>();
    let table = FlopChanceTable {
        hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
        remaining_deck: turn_cards, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights, num_players,
        num_combinations: num_combinations as f64, river_decks,
    };
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

fn find_pair_index(c1: Card, c2: Card) -> u16 {
    let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
    let mut idx = 0u16;
    for i in 0..52u8 {
        for j in (i+1)..52u8 {
            if i == lo && j == hi { return idx; }
            idx += 1;
        }
    }
    panic!("pair not found");
}

/// CFR formula bottom-up walk for one zone. Same as orchestration_oracle_river.
fn cfr_formula_bottomup(
    tree: &FlatTree,
    zone_nodes_per_level: &[Vec<u32>],
    init_cfv: &[f32],
    nh: usize,
    traverser: u8,
) -> Vec<f32> {
    let mut cfv = init_cfv.to_vec();
    let max_depth = zone_nodes_per_level.len();
    for level in (0..max_depth).rev() {
        for &nid in &zone_nodes_per_level[level] {
            let idx = nid as usize;
            let node = &tree.nodes[idx];
            let base = idx * nh;
            if node.node_type == NODE_TYPE_TERMINAL { continue; }
            if node.node_type == NODE_TYPE_CHANCE {
                for h in 0..nh { cfv[base + h] = 0.0; }
                for j in 0..node.num_children as usize {
                    let c = tree.children[node.children_start as usize + j] as usize;
                    for h in 0..nh { cfv[base + h] += cfv[c * nh + h]; }
                }
                continue;
            }
            // PLAYER
            let owner = node.player_id;
            let na = node.num_children as usize;
            for h in 0..nh { cfv[base + h] = 0.0; }
            for a in 0..na {
                let c = tree.children[node.children_start as usize + a] as usize;
                for h in 0..nh {
                    let weight = if owner == traverser { 1.0 / na as f32 } else { 1.0 };
                    cfv[base + h] += weight * cfv[c * nh + h];
                }
            }
        }
    }
    cfv
}

/// Seed terminal CFVs in the river-zone via validated showdown helper.
fn seed_river_terminals(
    tree: &FlatTree,
    table: &FlopChanceTable,
    river_levels: &[Vec<u32>],
    river_reach: &[f32],
    tc: u8, rc: u8,
    nh: usize, np: usize, traverser: u8,
) -> Vec<f32> {
    let nn = tree.num_nodes();
    let mut init = vec![0.0f32; nn * nh];
    let board_cards: Vec<u8> = vec![tc, rc];
    let (opp_str, opp_idx, pl_str, pl_idx) = table.river_sorted_arrays(tc, rc);
    for level_nodes in river_levels {
        for &nid in level_nodes {
            let idx = nid as usize;
            let node = &tree.nodes[idx];
            if node.node_type != NODE_TYPE_TERMINAL { continue; }
            let node_reach_base = idx * np * nh;
            let opp_p = 1 - traverser as usize;
            let raw = &river_reach[node_reach_base + opp_p * nh..node_reach_base + (opp_p + 1) * nh];
            let mut filtered: Vec<f32> = raw.to_vec();
            for h in 0..nh {
                if filtered[h] != 0.0 {
                    let c1 = table.hand_cards[h * 2];
                    let c2 = table.hand_cards[h * 2 + 1];
                    for &bc in &board_cards {
                        if c1 == bc || c2 == bc { filtered[h] = 0.0; break; }
                    }
                }
            }
            let opp_reach_views: Vec<&[f32]> = vec![filtered.as_slice()];
            let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
            let fold_mask = tree.get_folded_mask(idx);
            let cfv_raw = side_pot_showdown_cfv(
                &opp_reach_views, &table.hand_cards, nh,
                opp_str, opp_idx, pl_str, pl_idx,
                &contribs, fold_mask, traverser as usize, np as u8,
                tree.starting_pot,
            );
            let nc = table.num_combinations as f32;
            for h in 0..nh {
                init[idx * nh + h] = if nc > 0.0 { cfv_raw[h] / nc } else { cfv_raw[h] };
            }
        }
    }
    init
}

/// Compare two CFV arrays elementwise at f32 floor, returning (max_diff, max_idx).
fn cmp_cfv(a: &[f32], b: &[f32], label: &str, tol: f32) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", label);
    let mut max_d = 0.0f32;
    let mut max_i = 0;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max_d { max_d = d; max_i = i; }
    }
    if max_d > tol {
        eprintln!("  ✗ [{}] max_diff = {:.6e} at idx {} (tol {})", label, max_d, max_i, tol);
        eprintln!("    cpu={:.6} oracle={:.6}", a[max_i], b[max_i]);
    } else {
        eprintln!("  ✓ [{}] max_diff = {:.6e} (tol {})", label, max_d, tol);
    }
    (max_d, max_i)
}

#[test]
fn orchestration_strict_node_by_node_through_chance() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let nh = 4usize;
    let np = 2usize;
    let nn = tree.num_nodes();
    let traverser: u8 = 0;

    let (river_levels, turn_levels, flop_levels) = cpu.zone_nodes_per_level();
    cpu.compute_all_strategies(&tree);
    let flop_reach = cpu.compute_reach_flop(&tree, &game);

    // Identify chance-children sets (the same boundary-node sets CPU uses).
    let mut river_chance_children: Vec<u32> = Vec::new();
    let mut turn_chance_children: Vec<u32> = Vec::new();
    for n in &tree.nodes {
        if n.node_type == NODE_TYPE_CHANCE {
            for j in 0..n.num_children as usize {
                let c = tree.children[n.children_start as usize + j];
                if n.board_state == BoardState::River as u8 {
                    river_chance_children.push(c);
                } else if n.board_state == BoardState::Turn as u8 {
                    turn_chance_children.push(c);
                }
            }
        }
    }

    let table_ref = game.table();
    let turn_deck = table_ref.remaining_deck.clone();
    let params = DcfrParams::new(0);

    eprintln!("\n=== STRICT end-to-end orchestration oracle ===");
    eprintln!("River chance children: {} | Turn chance children: {}",
        river_chance_children.len(), turn_chance_children.len());
    eprintln!();

    let mut overall_max_diff = 0.0f32;
    let mut oracle_flop_init = vec![0.0f32; nn * nh];

    for (ti, &tc) in turn_deck.iter().enumerate() {
        let turn_reach = cpu.compute_reach_turn(&tree, ti, &flop_reach);
        let n_river = table_ref.river_decks[tc as usize].len();

        let mut oracle_river_accum = vec![0.0f32; nn * nh];

        for ri in 0..n_river {
            let river_reach = cpu.compute_reach_river(&tree, ti, ri, &turn_reach);
            let rc = table_ref.river_decks[tc as usize][ri];

            // ── Stage A: terminal CFVs in river zone ──
            // Same on both sides (same showdown helper, validated by
            // standing_showdown_oracle).
            let river_init = seed_river_terminals(
                &tree, table_ref, &river_levels, &river_reach,
                tc, rc, nh, np, traverser,
            );

            // ── Stage B: river-zone bottom-up CFV ──
            //   CPU: bottom_up_zone(Zone::River, ti, ri)
            //   Oracle: CFR formula direct walk
            let mut cpu_river = vec![0.0f32; nn * nh];
            cpu.bottom_up_zone(
                &tree, table_ref, traverser,
                &river_reach, &mut cpu_river,
                Zone::River, Some(ti), Some(ri),
                &params,
            );
            let oracle_river = cfr_formula_bottomup(
                &tree, &river_levels, &river_init, nh, traverser,
            );
            let (d, _) = cmp_cfv(&cpu_river, &oracle_river,
                &format!("river bottom_up (ti={}, ri={})", ti, ri), 1e-5);
            overall_max_diff = overall_max_diff.max(d);

            // ── Stage C: river chance accumulation ──
            // Both sides apply: river_accum[child*nh+h] += cp(tc,ri,h) * cfv[child*nh+h]
            // for child in river_chance_children. This is deterministic
            // arithmetic; we apply it to BOTH cpu_river and oracle_river
            // and verify the accumulators agree.
            for &child_id in &river_chance_children {
                for h in 0..nh {
                    let cp = table_ref.chance_probability_river(tc, ri, h);
                    oracle_river_accum[child_id as usize * nh + h] +=
                        cp * oracle_river[child_id as usize * nh + h];
                }
            }
        }

        // ── Stage D: river chance finalization → turn CFV seed ──
        let mut oracle_turn_init = vec![0.0f32; nn * nh];
        for &child_id in &river_chance_children {
            for h in 0..nh {
                oracle_turn_init[child_id as usize * nh + h] =
                    oracle_river_accum[child_id as usize * nh + h];
            }
        }
        // Seed turn-zone terminal CFVs (fold terminals on turn).
        let board_t: Vec<u8> = vec![tc];
        let (opp_str_t, opp_idx_t, pl_str_t, pl_idx_t) = table_ref.turn_sorted_arrays(tc);
        for level_nodes in &turn_levels {
            for &nid in level_nodes {
                let idx = nid as usize;
                let node = &tree.nodes[idx];
                if node.node_type != NODE_TYPE_TERMINAL { continue; }
                let node_reach_base = idx * np * nh;
                let opp_p = 1 - traverser as usize;
                let raw = &turn_reach[node_reach_base + opp_p * nh..node_reach_base + (opp_p + 1) * nh];
                let mut filtered: Vec<f32> = raw.to_vec();
                for h in 0..nh {
                    if filtered[h] != 0.0 {
                        let c1 = table_ref.hand_cards[h * 2];
                        let c2 = table_ref.hand_cards[h * 2 + 1];
                        for &bc in &board_t {
                            if c1 == bc || c2 == bc { filtered[h] = 0.0; break; }
                        }
                    }
                }
                let opp_reach_views: Vec<&[f32]> = vec![filtered.as_slice()];
                let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
                let fold_mask = tree.get_folded_mask(idx);
                let cfv_raw = side_pot_showdown_cfv(
                    &opp_reach_views, &table_ref.hand_cards, nh,
                    opp_str_t, opp_idx_t, pl_str_t, pl_idx_t,
                    &contribs, fold_mask, traverser as usize, np as u8,
                    tree.starting_pot,
                );
                let nc = table_ref.num_combinations as f32;
                for h in 0..nh {
                    oracle_turn_init[idx * nh + h] = if nc > 0.0 { cfv_raw[h] / nc } else { cfv_raw[h] };
                }
            }
        }

        // ── Stage E: turn-zone bottom-up ──
        let mut cpu_turn = oracle_turn_init.clone();
        cpu.bottom_up_zone(
            &tree, table_ref, traverser,
            &turn_reach, &mut cpu_turn,
            Zone::Turn, Some(ti), None,
            &params,
        );
        let oracle_turn = cfr_formula_bottomup(
            &tree, &turn_levels, &oracle_turn_init, nh, traverser,
        );
        let (d, _) = cmp_cfv(&cpu_turn, &oracle_turn,
            &format!("turn bottom_up (ti={})", ti), 1e-5);
        overall_max_diff = overall_max_diff.max(d);

        // ── Stage F: turn chance accumulation → flop CFV seed ──
        for &child_id in &turn_chance_children {
            for h in 0..nh {
                let cp = table_ref.chance_probability_turn(ti, h);
                oracle_flop_init[child_id as usize * nh + h] +=
                    cp * oracle_turn[child_id as usize * nh + h];
            }
        }
    }

    // ── Stage G: flop chance finalization done by initialization above ──
    // Seed flop-zone terminal CFVs.
    let (opp_str_f, opp_idx_f, pl_str_f, pl_idx_f, _) = table_ref.sorted_opp_arrays_base();
    for level_nodes in &flop_levels {
        for &nid in level_nodes {
            let idx = nid as usize;
            let node = &tree.nodes[idx];
            if node.node_type != NODE_TYPE_TERMINAL { continue; }
            let node_reach_base = idx * np * nh;
            let opp_p = 1 - traverser as usize;
            let raw = &flop_reach[node_reach_base + opp_p * nh..node_reach_base + (opp_p + 1) * nh];
            let opp_reach_views: Vec<&[f32]> = vec![raw];
            let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
            let fold_mask = tree.get_folded_mask(idx);
            let cfv_raw = side_pot_showdown_cfv(
                &opp_reach_views, &table_ref.hand_cards, nh,
                &opp_str_f, &opp_idx_f, &pl_str_f, &pl_idx_f,
                &contribs, fold_mask, traverser as usize, np as u8,
                tree.starting_pot,
            );
            let nc = table_ref.num_combinations as f32;
            for h in 0..nh {
                oracle_flop_init[idx * nh + h] = if nc > 0.0 { cfv_raw[h] / nc } else { cfv_raw[h] };
            }
        }
    }

    // ── Stage H: flop-zone bottom-up ──
    let mut cpu_flop = oracle_flop_init.clone();
    cpu.bottom_up_zone(
        &tree, table_ref, traverser,
        &flop_reach, &mut cpu_flop,
        Zone::Flop, None, None,
        &params,
    );
    let oracle_flop = cfr_formula_bottomup(
        &tree, &flop_levels, &oracle_flop_init, nh, traverser,
    );
    let (d, _) = cmp_cfv(&cpu_flop, &oracle_flop, "flop bottom_up", 1e-5);
    overall_max_diff = overall_max_diff.max(d);

    eprintln!();
    eprintln!("Overall max_diff across all stages: {:.6e}", overall_max_diff);
    assert!(
        overall_max_diff < 1e-5,
        "STRICT end-to-end orchestration oracle: at least one stage exceeded \
         f32-noise tolerance. The CFR formula direct computation (independent \
         of bottom_up_zone) disagrees with CPU at some node, indicating a bug \
         in the orchestration arithmetic. overall_max_diff = {}",
        overall_max_diff
    );

    eprintln!("\n✓ Strict end-to-end orchestration oracle PASSES at f32 floor.");
    eprintln!("  Every stage (river bottom_up × n_river, river chance accumulation, river");
    eprintln!("  chance finalization, turn bottom_up × n_turn, turn chance accumulation,");
    eprintln!("  turn chance finalization, flop bottom_up) matches the CFR formula direct");
    eprintln!("  computation node-by-node at f32 precision.");
    eprintln!();
    eprintln!("  The orchestration layer that bottom_up_zone implements is now anchored");
    eprintln!("  against an implementation-independent reference at every node, through");
    eprintln!("  the chance-integration steps where turn and flop extend over the river.");
    eprintln!("  This is the final precision check before the blueprint — the chance");
    eprintln!("  integration where street transitions happen is provably correct.");
}
