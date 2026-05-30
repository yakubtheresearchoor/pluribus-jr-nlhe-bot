/// Debug the zero-sum and exploitability issue using the minimal tree (4 hands, 2 turn, 2 river each).
///
/// Run:
///   cargo test -p solver-core --features metal --test zero_sum_minimal -- --test-threads=1 --nocapture

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

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

    let valid_hand_indices = chosen_hands;
    let num_valid = nh;
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in valid_hand_indices.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i * 2] = c1;
        hand_cards[i * 2 + 1] = c2;
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
        hand = hand.add_card(c1 as usize);
        hand = hand.add_card(c2 as usize);
        for &bc in &board { hand = hand.add_card(bc as usize); }
        hand_ranks_base[i] = hand.evaluate_internal() as u16;
    }

    let turn_cards: Vec<u8> = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("4c").unwrap() as u8,
    ];

    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("5c").unwrap() as u8,
        card_from_str("6c").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("3c").unwrap() as u8,
        card_from_str("5c").unwrap() as u8,
    ];

    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];

    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 {
                turn_ranks[tc as usize * nh + i] = 0;
                continue;
            }
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in &board { hand = hand.add_card(bc as usize); }
            hand = hand.add_card(tc as usize);
            turn_ranks[tc as usize * nh + i] = hand.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh)
            .map(|h| (turn_ranks[tc as usize * nh + h] + 1, h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            let off = tc as usize * num_opp * nh + oi * nh;
            for h in 0..nh {
                turn_sorted_str[off + h] = items[h].0;
                turn_sorted_idx[off + h] = items[h].1;
            }
        }
    }

    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];

    for &tc in &turn_cards {
        let turn_mask = board_mask | (1u64 << tc);
        for &rc in &river_decks[tc as usize] {
            let full_mask = turn_mask | (1u64 << rc);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if full_mask & (1u64 << c1) != 0 || full_mask & (1u64 << c2) != 0 {
                    river_ranks[tc as usize * 52 * nh + rc as usize * nh + i] = 0;
                    continue;
                }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in &board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(tc as usize);
                hand = hand.add_card(rc as usize);
                river_ranks[tc as usize * 52 * nh + rc as usize * nh + i] =
                    hand.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| {
                    let r = river_ranks[tc as usize * 52 * nh + rc as usize * nh + h];
                    (r + 1, h as u16)
                })
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = tc as usize * 52 * num_opp * nh
                    + rc as usize * num_opp * nh + oi * nh;
                for h in 0..nh {
                    river_sorted_str[off + h] = items[h].0;
                    river_sorted_idx[off + h] = items[h].1;
                }
            }
        }
    }

    let initial_weights = vec![vec![1.0f32; nh], vec![1.0f32; nh]];

    let mut nc = 0.0f64;
    for h0 in 0..nh {
        let mask0: u64 = (1u64 << hand_cards[h0 * 2]) | (1u64 << hand_cards[h0 * 2 + 1]);
        for h1 in 0..nh {
            let mask1: u64 = (1u64 << hand_cards[h1 * 2]) | (1u64 << hand_cards[h1 * 2 + 1]);
            if mask0 & mask1 == 0 { nc += 1.0; }
        }
    }

    let table = FlopChanceTable {
        hand_ranks_base,
        valid_hand_indices,
        num_valid,
        conflict,
        hand_cards,
        remaining_deck: turn_cards.clone(),
        turn_ranks,
        turn_sorted_str,
        turn_sorted_idx,
        river_ranks,
        river_sorted_str,
        river_sorted_idx,
        initial_weights,
        num_players,
        num_combinations: nc,
        river_decks,
    };

    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 10,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

fn find_pair_index(c1: Card, c2: Card) -> u16 {
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (a, b) = index_to_card_pair(idx);
        if (a == c1 as u8 && b == c2 as u8) || (a == c2 as u8 && b == c1 as u8) {
            return idx as u16;
        }
    }
    panic!("pair not found");
}

/// Check zero-sum property at iter 0 (uniform strategy).
#[test]
fn zero_sum_at_iter_0() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    let nh = solver.num_hands();
    let nc = game.table().num_combinations as f32;

    println!("\n=== Zero-sum check at iter 0 (uniform strategy) ===");
    println!("nh = {}, nc = {:.0}", nh, nc);

    // Get strategy values for both players
    let sv0 = solver.strategy_value(&tree, &game, 0);
    let sv1 = solver.strategy_value(&tree, &game, 1);

    // EV = sum_h initial_weight[h] * sv[h] / nc
    let w0 = &game.table().initial_weights[0];
    let w1 = &game.table().initial_weights[1];

    let ev_sv0: f32 = (0..nh).map(|h| w0[h] * sv0[h]).sum::<f32>() / nc;
    let ev_sv1: f32 = (0..nh).map(|h| w1[h] * sv1[h]).sum::<f32>() / nc;

    println!("\n  Strategy values (first {} hands):", nh);
    for h in 0..nh {
        let (c1, c2) = index_to_card_pair(game.table().valid_hand_indices[h] as usize);
        println!("    h={} ({:2},{:2}): SV0={:.6} SV1={:.6}", h, c1, c2, sv0[h], sv1[h]);
    }

    println!("\n  EV[SV,P0] = {:.6}", ev_sv0);
    println!("  EV[SV,P1] = {:.6}", ev_sv1);
    println!("  EV[SV,P0] + EV[SV,P1] = {:.6} (should be 0)", ev_sv0 + ev_sv1);

    // Also check BR values
    let br0 = solver.best_response_value(&tree, &game, 0);
    let br1 = solver.best_response_value(&tree, &game, 1);

    let ev_br0: f32 = (0..nh).map(|h| w0[h] * br0[h]).sum::<f32>() / nc;
    let ev_br1: f32 = (0..nh).map(|h| w1[h] * br1[h]).sum::<f32>() / nc;

    println!("\n  EV[BR,P0] = {:.6}", ev_br0);
    println!("  EV[BR,P1] = {:.6}", ev_br1);
    println!("  EV[BR,P0] + EV[BR,P1] = {:.6} (should be 0)", ev_br0 + ev_br1);

    let expl = solver.compute_exploitability(&tree, &game);
    println!("\n  Exploitability = {:.6e}", expl);
    println!("  = (EV[BR,P0] - EV[SV,P0] + EV[BR,P1] - EV[SV,P1]) / 2");
    println!("  = ({:.6} - {:.6} + {:.6} - {:.6}) / 2 = {:.6}",
        ev_br0, ev_sv0, ev_br1, ev_sv1, (ev_br0 - ev_sv0 + ev_br1 - ev_sv1) / 2.0);

    assert!((ev_sv0 + ev_sv1).abs() < 0.01, "SV not zero-sum: {} + {} = {}", ev_sv0, ev_sv1, ev_sv0 + ev_sv1);
    // Note: BR_sum is NOT expected to be zero. BR_P0 uses P0's optimal strategy
    // against P1's profile, while BR_P1 uses P1's optimal against P0's profile.
    // These are different game situations, so zero-sum doesn't apply.
    println!("  Note: BR_sum = {:.6} is NOT expected to be zero (different strategy pairs)", ev_br0 + ev_br1);
}

/// Trace CFVs at a specific terminal node for each (turn, river) outcome.
#[test]
fn trace_terminal_cfv() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    let nh = solver.num_hands();

    println!("\n=== Terminal CFV trace ===");

    // Find all terminal nodes
    for idx in 0..tree.num_nodes() {
        if !tree.nodes[idx].is_terminal() { continue; }
        let fold_mask = tree.get_folded_mask(idx);
        let contributions: Vec<i32> = (0..2).map(|p| tree.get_contribution(idx, p as u8)).collect();

        println!("\n  Terminal node {}: fold_mask={}, contributions={:?}",
            idx, fold_mask, contributions);

        // For each (turn, river) outcome, compute terminal CFV
        for (ti, &tc) in game.table().remaining_deck.iter().enumerate() {
            for (ri, &rc) in game.table().river_decks[tc as usize].iter().enumerate() {
                // Set the chance outcome
                game.set_turn_card(tc);
                game.set_river_card(rc);

                // Build opp_reach = initial_weight (uniform strategy, no sigma)
                let cfreach = vec![
                    game.table().initial_weights[0].clone(),
                    game.table().initial_weights[1].clone(),
                ];

                let cfv_p0 = game.evaluate_terminal(0, idx, &tree, &cfreach);
                let cfv_p1 = game.evaluate_terminal(1, idx, &tree, &cfreach);

                println!("    turn={} river={}: CFV_P0 = {:?}  CFV_P1 = {:?}", ti, ri, cfv_p0, cfv_p1);

                // Check zero-sum per outcome
                let sum_p0: f32 = cfv_p0.iter().sum();
                let sum_p1: f32 = cfv_p1.iter().sum();
                if sum_p0 + sum_p1 != 0.0 {
                    println!("      ⚠️ NOT ZERO-SUM: sum_P0={:.4} sum_P1={:.4} total={:.4}",
                        sum_p0, sum_p1, sum_p0 + sum_p1);
                }
            }
        }
    }
}

/// Check: for each (turn, river) outcome, does the CFV from evaluate_terminal
/// match what we'd compute by hand?
#[test]
fn manual_terminal_check() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let nh = game.table().num_valid;

    println!("\n=== Manual terminal CFV check ===");

    // Hands: AhKh (h0), QhJh (h1), Th9h (h2), 8h6h (h3)
    // No conflicts between any pair of hands
    // nc = 4*4 = 16 (all pairs non-conflicting)

    // Find showdown terminals (fold_mask = 0)
    for idx in 0..tree.num_nodes() {
        if !tree.nodes[idx].is_terminal() { continue; }
        let fold_mask = tree.get_folded_mask(idx);
        if fold_mask != 0 { continue; } // skip fold terminals

        let contributions: Vec<i32> = (0..2).map(|p| tree.get_contribution(idx, p as u8)).collect();
        println!("\n  Showdown node {}: contributions={:?}", idx, contributions);

        // Uniform opp_reach
        let cfreach = vec![vec![1.0f32; nh], vec![1.0f32; nh]];

        // Turn card 0 (3c), River card 0 (5c)
        let tc = game.table().remaining_deck[0]; // 3c
        let rc = game.table().river_decks[tc as usize][0]; // 5c
        game.set_turn_card(tc);
        game.set_river_card(rc);

        let cfv0 = game.evaluate_terminal(0, idx, &tree, &cfreach);
        let cfv1 = game.evaluate_terminal(1, idx, &tree, &cfreach);

        let nc = game.table().num_combinations as f32;
        let total_pot = tree.starting_pot as f32 + contributions[0] as f32 + contributions[1] as f32;

        println!("    turn=3c river=5c, total_pot={}", total_pot);
        println!("    nc = {:.0}", nc);

        // Manual: for P0 as traverser, all 4 opp hands valid, none block tc or rc
        // cfreach_sum = 4.0 (sum of opp reach for all hands)
        // For h0 (AhKh): conficts with none of the 4 opp hands → cfreach = 4.0
        // cfv[h0] = total_pot * 4.0 / nc (if h0 beats all) or -total_pot * 4.0 / nc (if loses to all)
        // But with showdown, it's based on hand ranks

        println!("    CFV_P0 (traverser=0): {:?}", cfv0);
        println!("    CFV_P1 (traverser=1): {:?}", cfv1);

        // Verify: cfv0[h] + cfv1[h] should NOT be zero per-hand (different traverser views)
        // But EV[cfv0] + EV[cfv1] = 0 when summed over all hands with proper weights

        let raw_sum0: f32 = cfv0.iter().sum();
        let raw_sum1: f32 = cfv1.iter().sum();
        println!("    raw sum CFV_P0 = {:.4}, CFV_P1 = {:.4}, total = {:.4}",
            raw_sum0, raw_sum1, raw_sum0 + raw_sum1);

        // The nc division: side_pot_showdown_cfv returns raw, evaluate_terminal divides by nc
        println!("    After /nc: sum_P0 = {:.4}, sum_P1 = {:.4}", raw_sum0 / nc, raw_sum1 / nc);
    }
}

/// Run a few iterations and check zero-sum at each step.
#[test]
fn zero_sum_across_iters() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(&tree, game.table());

    let nh = solver.num_hands();
    let nc = game.table().num_combinations as f32;
    let w0 = &game.table().initial_weights[0];
    let w1 = &game.table().initial_weights[1];

    println!("\n=== Zero-sum check across iterations ===");
    println!("  nh={} nc={:.0}", nh, nc);
    println!("  {:>4} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}",
        "iter", "SV0", "SV1", "SV_sum", "BR0", "BR1", "BR_sum");

    for i in 0..=10 {
        if i > 0 {
            let _ = solver.run(&tree, &game, 1);
        }

        let sv0 = solver.strategy_value(&tree, &game, 0);
        let sv1 = solver.strategy_value(&tree, &game, 1);
        let br0 = solver.best_response_value(&tree, &game, 0);
        let br1 = solver.best_response_value(&tree, &game, 1);

        let ev_sv0: f32 = (0..nh).map(|h| w0[h] * sv0[h]).sum::<f32>() / nc;
        let ev_sv1: f32 = (0..nh).map(|h| w1[h] * sv1[h]).sum::<f32>() / nc;
        let ev_br0: f32 = (0..nh).map(|h| w0[h] * br0[h]).sum::<f32>() / nc;
        let ev_br1: f32 = (0..nh).map(|h| w1[h] * br1[h]).sum::<f32>() / nc;

        let expl = solver.compute_exploitability(&tree, &game);

        println!("  {:>4} | {:>12.6} | {:>12.6} | {:>12.6} | {:>12.6} | {:>12.6} | {:>12.6}  expl={:.6e}",
            i, ev_sv0, ev_sv1, ev_sv0 + ev_sv1, ev_br0, ev_br1, ev_br0 + ev_br1, expl);
    }
}
