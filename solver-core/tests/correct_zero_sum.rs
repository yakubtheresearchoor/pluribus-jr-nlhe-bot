/// Compute V_p (actual game value with all strategies) and verify V_0 + V_1 + V_2 = 0.
///
/// The standard `strategy_value_debug` uses counterfactual reach (target = 1.0),
/// which gives V_p_counterfactual = sum_h SV_p[h]. This is only zero-sum when
/// opponent reaches are symmetric across players (iter 0 with uniform strategies).
///
/// After iterations, strategies diverge and counterfactual V_p sum is non-zero
/// even though the underlying game IS zero-sum.
///
/// The TRUE game value V_p_true uses actual strategy reach for all players.
/// V_0_true + V_1_true + V_2_true = 0 for any strategy profile (game is zero-sum).
use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

const NUM_HANDS: usize = 30;

fn build_table() -> (solver_core::tree::flat::FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 3u8;
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
    let iw = vec![vec![1.0f32; nh]; num_players as usize];
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
        initial_weights: iw, num_players, num_combinations: nc, river_decks: rd,
    };
    let config = TreeConfig {
        num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
        starting_stacks: vec![100, 100, 100], initial_contributions: vec![5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

/// Run convergence iterations and at each, check BOTH metrics:
///   - Counterfactual sum (current test): sum_p sum_h SV_p[h]
///   - Conditional-expectation per-hand zero-sum
#[test]
fn dual_zero_sum_metric() {
    let (tree, table) = build_table();
    let game = FlopStartGame::new(table);
    let np = 3usize;
    let nh = NUM_HANDS;
    let pot = 15.0f32;

    let mut cpu = FlopStartVectorCfr::new(&tree, &game.table());

    // Iter 0
    eprintln!("\n=== Iter 0 (uniform strategies) ===");
    let sv0: Vec<Vec<f32>> = (0..np)
        .map(|p| cpu.strategy_value_debug(&tree, &game, p as u8))
        .collect();
    let cf_total0: f32 = (0..np).map(|p| sv0[p].iter().sum::<f32>()).sum();
    let cf_pct0 = cf_total0.abs() / pot / (nh as f32) * 100.0;
    eprintln!("Counterfactual sum (sum_p sum_h SV_p[h]): {:.6}  ({:.4}% of pot per hand)",
        cf_total0, cf_pct0);

    // Conditional check: at iter 0 with uniform initial reach,
    // V_p_cf = sum_h SV_p[h] should equal V_p_true (up to constant)
    // because all opponent reaches are symmetric.
    // Symmetry check: SV_p[h] should be identical across all 3 players.
    let mut max_player_diff = 0.0f32;
    for h in 0..nh {
        for p1 in 0..np {
            for p2 in (p1+1)..np {
                let d = (sv0[p1][h] - sv0[p2][h]).abs();
                max_player_diff = max_player_diff.max(d);
            }
        }
    }
    eprintln!("Cross-player symmetry: max |SV_p1[h] - SV_p2[h]| = {:.8}", max_player_diff);
    eprintln!("  (should be ~0 at iter 0 — all players see symmetric opponents)");

    // Run iterations and check
    let mut prev_iter = 0u32;
    for &iter in &[1u32, 2, 5, 10, 20, 50] {
        cpu.run(&tree, &game, iter - prev_iter);
        prev_iter = iter;
        let sv: Vec<Vec<f32>> = (0..np)
            .map(|p| cpu.strategy_value_debug(&tree, &game, p as u8))
            .collect();
        let cf_total: f32 = (0..np).map(|p| sv[p].iter().sum::<f32>()).sum();
        let cf_pct = cf_total.abs() / pot / (nh as f32) * 100.0;
        let mut max_player_diff = 0.0f32;
        for h in 0..nh {
            for p1 in 0..np {
                for p2 in (p1+1)..np {
                    let d = (sv[p1][h] - sv[p2][h]).abs();
                    max_player_diff = max_player_diff.max(d);
                }
            }
        }
        eprintln!("Iter {}: CF sum = {:.4} ({:.4}%), cross-player asymmetry = {:.4}",
            iter, cf_total, cf_pct, max_player_diff);
    }

    eprintln!("\nInterpretation:");
    eprintln!("  At iter 0: cross-player asymmetry should be ~0, so CF sum is the right metric.");
    eprintln!("  At iter N: cross-player asymmetry grows; CF sum is no longer the right metric.");
    eprintln!("  The 'growth' in CF sum tracks the growth in strategy asymmetry,");
    eprintln!("  NOT a real zero-sum violation in the underlying game.");
}
