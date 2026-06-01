/// Direct zero-sum check on evaluate_terminal for individual terminal nodes.
/// Tests showdown terminals and fold terminals separately.
use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const NUM_HANDS: usize = 10;

fn build_game() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let np = 3u8;
    let num_opp = 2;
    let nh = NUM_HANDS;

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i*2] = c1; hand_cards[i*2+1] = c2;
    }
    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh { for j in 0..nh {
        if i == j { conflict[i*nh+j] = 1; continue; }
        let (a1,a2) = index_to_card_pair(chosen[i] as usize);
        let (b1,b2) = index_to_card_pair(chosen[j] as usize);
        if a1==b1||a1==b2||a2==b1||a2==b2 { conflict[i*nh+j] = 1; }
    }}
    let mut hr = vec![0u16; nh];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1,c2) = index_to_card_pair(hi as usize);
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        hr[i] = h.evaluate_internal() as u16;
    }
    let tc = vec![card_from_str("3c").unwrap() as u8];
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    rd[tc[0] as usize] = vec![card_from_str("5s").unwrap() as u8];

    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &t in &tc {
        for (i, &hi) in chosen.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let tm = board_mask | (1u64 << t);
            if tm & (1u64 << c1) != 0 || tm & (1u64 << c2) != 0 { continue; }
            let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
            for &bc in &board { h = h.add_card(bc as usize); }
            h = h.add_card(t as usize);
            turn_ranks[t as usize * nh + i] = h.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
        items.sort_by_key(|&(s,_)| s);
        for oi in 0..num_opp {
            let off = t as usize * num_opp * nh + oi * nh;
            for h in 0..nh { turn_sorted_str[off + h] = items[h].0; turn_sorted_idx[off + h] = items[h].1; }
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &t in &tc {
        let tm = board_mask | (1u64 << t);
        for &r in &rd[t as usize] {
            let fm = tm | (1u64 << r);
            for (i, &hi) in chosen.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if fm & (1u64 << c1) != 0 || fm & (1u64 << c2) != 0 { continue; }
                let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
                for &bc in &board { h = h.add_card(bc as usize); }
                h = h.add_card(t as usize).add_card(r as usize);
                river_ranks[t as usize * 52 * nh + r as usize * nh + i] = h.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16)).collect();
            items.sort_by_key(|&(s,_)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; np as usize];
    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let m0: u64 = (1u64 << hand_cards[h0*2]) | (1u64 << hand_cards[h0*2+1]);
        for h1 in 0..nh {
            if h0 == h1 { continue; }
            let m1: u64 = (1u64 << hand_cards[h1*2]) | (1u64 << hand_cards[h1*2+1]);
            if m0 & m1 != 0 { continue; }
            for h2 in 0..nh {
                if h2 == h0 || h2 == h1 { continue; }
                let m2: u64 = (1u64 << hand_cards[h2*2]) | (1u64 << hand_cards[h2*2+1]);
                if m0 & m2 != 0 || m1 & m2 != 0 { continue; }
                nc += 1.0;
            }
        }
    }
    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players: np, num_combinations: nc, river_decks: rd,
    };
    let config = TreeConfig {
        num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![100, 100, 100], initial_contributions: vec![5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).unwrap();
    let game = FlopStartGame::new(table);
    (tree, game)
}

/// Test: call evaluate_terminal on a specific showdown node for all 3 traversers.
/// Check zero-sum BEFORE and AFTER num_combinations division.
#[test]
fn showdown_terminal_zero_sum() {
    let (tree, game) = build_game();
    let nh = NUM_HANDS;
    let np = 3usize;

    // Set up the game for a specific turn + river
    game.set_chance_outcome(0); // turn card 0
    game.set_chance_outcome(0); // river card 0

    // Find a showdown terminal node (no fold)
    let mut showdown_nodes = Vec::new();
    let mut fold_nodes = Vec::new();
    for (idx, node) in tree.nodes.iter().enumerate() {
        if node.is_terminal() {
            let fm = tree.get_folded_mask(idx);
            if fm == 0 {
                showdown_nodes.push(idx);
            } else {
                fold_nodes.push(idx);
            }
        }
    }
    eprintln!("Showdown terminals: {}, Fold terminals: {}", showdown_nodes.len(), fold_nodes.len());

    // Build board-aware validity mask for current turn+river
    let turn_card = game.table().remaining_deck[0]; // 3c
    let river_card = game.table().river_decks[turn_card as usize][0]; // 5s
    let board_extra = [turn_card, river_card];
    let mut valid_mask = vec![1.0f32; nh];
    for h in 0..nh {
        let c1 = game.table().hand_cards[h * 2];
        let c2 = game.table().hand_cards[h * 2 + 1];
        for &bc in &board_extra {
            if c1 == bc || c2 == bc { valid_mask[h] = 0.0; break; }
        }
    }
    let n_valid_hands: usize = valid_mask.iter().filter(|&&v| v > 0.0).count();
    eprintln!("Valid hands (no turn/river conflict): {}/{}", n_valid_hands, nh);

    // Use uniform reach for VALID hands (mimics what a CFR walk would produce
    // at iter 0 after upstream board filtering)
    let uniform_reach: Vec<Vec<f32>> = vec![valid_mask.clone(); np];

    // Sweep many terminals to confirm zero-sum holds at all of them
    eprintln!("\n=== Sweep first 50 showdown terminals ===");
    let mut max_violation_norm: f64 = 0.0;
    let mut max_pct: f64 = 0.0;
    let mut worst_idx = 0usize;
    let mut worst_contribs: Vec<i32> = vec![];
    for &node_idx in showdown_nodes.iter().take(50) {
        let mut cfv_sum = vec![0.0f64; nh];
        for traverser in 0..np {
            let cfv = game.evaluate_terminal(traverser as u8, node_idx, &tree, &uniform_reach);
            for h in 0..nh {
                cfv_sum[h] += (cfv[h] * valid_mask[h]) as f64;
            }
        }
        let total: f64 = cfv_sum.iter().sum();
        let pct = total.abs() / (15.0 / nh as f64) * 100.0;
        if total.abs() > max_violation_norm {
            max_violation_norm = total.abs();
            max_pct = pct;
            worst_idx = node_idx;
            worst_contribs = (0..np).map(|p| tree.get_contribution(node_idx, p as u8)).collect();
        }
    }
    eprintln!("Max showdown violation across 50 terminals: {:.10} ({:.8}% of pot/hand)",
        max_violation_norm, max_pct);
    eprintln!("Worst terminal: node={}, contribs={:?}", worst_idx, worst_contribs);

    // Print a few specific terminals to understand patterns
    eprintln!("\n--- Per-terminal violations (first 20) ---");
    for &node_idx in showdown_nodes.iter().take(20) {
        let mut cfv_sum = vec![0.0f64; nh];
        for traverser in 0..np {
            let cfv = game.evaluate_terminal(traverser as u8, node_idx, &tree, &uniform_reach);
            for h in 0..nh {
                cfv_sum[h] += (cfv[h] * valid_mask[h]) as f64;
            }
        }
        let total: f64 = cfv_sum.iter().sum();
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(node_idx, p as u8)).collect();
        let fm = tree.get_folded_mask(node_idx);
        eprintln!("  node={:5}  fold_mask={:03b}  contribs={:?}  violation={:.6}",
            node_idx, fm, contribs, total);
    }
    assert!(max_violation_norm < 1e-3,
        "Showdown sweep failed zero-sum gate: max = {}", max_violation_norm);

    eprintln!("\n=== Sweep first 50 fold terminals ===");
    let mut max_fold_violation: f64 = 0.0;
    let mut worst_fold_idx = 0usize;
    let mut worst_fold_contribs: Vec<i32> = vec![];
    let mut worst_fold_mask: u16 = 0;
    for &node_idx in fold_nodes.iter().take(50) {
        let mut cfv_sum = vec![0.0f64; nh];
        for traverser in 0..np {
            let cfv = game.evaluate_terminal(traverser as u8, node_idx, &tree, &uniform_reach);
            for h in 0..nh {
                cfv_sum[h] += (cfv[h] * valid_mask[h]) as f64;
            }
        }
        let total: f64 = cfv_sum.iter().sum();
        if total.abs() > max_fold_violation {
            max_fold_violation = total.abs();
            worst_fold_idx = node_idx;
            worst_fold_contribs = (0..np).map(|p| tree.get_contribution(node_idx, p as u8)).collect();
            worst_fold_mask = tree.get_folded_mask(node_idx);
        }
    }
    eprintln!("Max fold violation across 50 terminals: {:.10}", max_fold_violation);
    eprintln!("Worst fold terminal: node={}, fm={:#06b}, contribs={:?}",
        worst_fold_idx, worst_fold_mask, worst_fold_contribs);
    // Print first 10 fold terminals
    for &node_idx in fold_nodes.iter().take(10) {
        let mut cfv_sum = vec![0.0f64; nh];
        for traverser in 0..np {
            let cfv = game.evaluate_terminal(traverser as u8, node_idx, &tree, &uniform_reach);
            for h in 0..nh {
                cfv_sum[h] += (cfv[h] * valid_mask[h]) as f64;
            }
        }
        let total: f64 = cfv_sum.iter().sum();
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(node_idx, p as u8)).collect();
        let fm = tree.get_folded_mask(node_idx);
        eprintln!("  fold node={:5}  fm={:03b}  contribs={:?}  violation={:.6}",
            node_idx, fm, contribs, total);
    }
    assert!(max_fold_violation < 1e-3,
        "Fold sweep failed: max = {}", max_fold_violation);

    // Test showdown terminals
    if let Some(&node_idx) = showdown_nodes.first() {
        eprintln!("\n=== Showdown terminal (node {}) ===", node_idx);
        let contributions: Vec<i32> = (0..np)
            .map(|p| tree.get_contribution(node_idx, p as u8))
            .collect();
        eprintln!("Contributions: {:?}", contributions);

        let mut cfv_sum_norm = vec![0.0f64; nh];

        for traverser in 0..np {
            let cfv = game.evaluate_terminal(traverser as u8, node_idx, &tree, &uniform_reach);
            // Weight CFV by traverser's reach (zero for board-conflicting hands)
            let weighted_sum: f64 = (0..nh).map(|h| (cfv[h] * valid_mask[h]) as f64).sum();
            eprintln!("  Traverser {}: weighted_sum = {:.6}, first 5 = {:?}",
                traverser, weighted_sum, &cfv[..5.min(nh)]);
            for h in 0..nh {
                cfv_sum_norm[h] += (cfv[h] * valid_mask[h]) as f64;
            }
        }

        let total_norm: f64 = cfv_sum_norm.iter().sum();
        eprintln!("  SUM across traversers (reach-weighted, normalized): {:.8}", total_norm);

        // Now compute raw (before num_combinations division) by multiplying back
        let nc = game.table().num_combinations;
        let total_raw = total_norm * nc;
        eprintln!("  SUM across traversers (raw, *nc): {:.8}", total_raw);
        eprintln!("  num_combinations: {}", nc);

        // The critical check: is the raw sum zero to float precision?
        let pct = total_norm.abs() / (15.0 / nh as f64) * 100.0;
        eprintln!("  Zero-sum violation: {:.6}% of pot per hand", pct);
        assert!(total_norm.abs() < 1e-3,
            "Showdown terminal not zero-sum to float precision: total_norm = {}", total_norm);
    }

    // Test fold terminals
    if let Some(&node_idx) = fold_nodes.first() {
        eprintln!("\n=== Fold terminal (node {}) ===", node_idx);
        let fm = tree.get_folded_mask(node_idx);
        let contributions: Vec<i32> = (0..np)
            .map(|p| tree.get_contribution(node_idx, p as u8))
            .collect();
        eprintln!("Fold mask: {:#06b}, Contributions: {:?}", fm, contributions);

        let mut cfv_sum = vec![0.0f64; nh];
        for traverser in 0..np {
            let cfv = game.evaluate_terminal(traverser as u8, node_idx, &tree, &uniform_reach);
            let raw_sum: f64 = cfv.iter().map(|&v| v as f64).sum();
            eprintln!("  Traverser {}: sum = {:.6}", traverser, raw_sum);
            for h in 0..nh { cfv_sum[h] += cfv[h] as f64; }
        }
        let total: f64 = cfv_sum.iter().sum();
        let pct = total.abs() / (15.0 / nh as f64) * 100.0;
        eprintln!("  SUM across traversers: {:.8}", total);
        eprintln!("  Zero-sum violation: {:.6}% of pot per hand", pct);
    }

    game.clear_chance_outcome();
}
