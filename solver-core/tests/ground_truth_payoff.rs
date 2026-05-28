#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::game::GameSpec;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::showdown::side_pot_showdown_cfv;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

/// Helper to get sorted arrays and hand cards from a RiverPokerGame
struct GameArrays {
    s_opp_str: Vec<u16>,
    s_opp_idx: Vec<u16>,
    s_pl_str: Vec<u16>,
    s_pl_idx: Vec<u16>,
    hand_cards: Vec<u8>,
    nh: usize,
}

impl GameArrays {
    fn from_game(game: &RiverPokerGame) -> Self {
        let (s_opp_str, s_opp_idx, s_pl_str, s_pl_idx, _) = game.sorted_opp_arrays();
        GameArrays {
            s_opp_str, s_opp_idx, s_pl_str, s_pl_idx,
            hand_cards: game.hand_cards_gpu(),
            nh: game.num_hands(0),
        }
    }
}

// ─── Ground Truth: Terminal Payoff Correctness ────────────────────────────

#[test]
fn gt_fold_payoff_starting_pot() {
    // With starting_pot=100, np=2, folder contribution=0:
    // After normalization: CFV ≈ payoff / num_combinations × cfreach
    // payoff = -50 (folder's investment), nc ≈ nh², cfreach_sum ≈ nh
    // So avg CFV ≈ -50 / nh ≈ -50/1081 ≈ -0.046
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "Qs"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];

    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::River,
        starting_pot: 100, starting_stacks: vec![200, 200],
        initial_contributions: vec![0, 0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_hands(0);
    let nc = game.num_combinations();
    println!("nh={}, nc={:.0}", nh, nc);

    let opp_reach: Vec<f32> = vec![1.0f32; nh];
    let cfreach: Vec<Vec<f32>> = vec![opp_reach.clone(), opp_reach];

    let mut found = false;
    for i in 0..tree.num_nodes() {
        let node = &tree.nodes[i];
        if !node.is_terminal() { continue; }
        let fold_mask = tree.get_folded_mask(i);
        if fold_mask == 0 { continue; }
        let c0 = tree.get_contribution(i, 0);
        let c1 = tree.get_contribution(i, 1);

        // P0 folded with c0=0
        if fold_mask & 1 != 0 && c0 == 0 {
            found = true;
            let cfv = game.evaluate_terminal(0, i, &tree, &cfreach);
            // P0 folded. Investment = 100/2 + 0 = 50.
            // Normalized: CFV[h] = -50 × cfreach[h] / nc
            assert!(cfv.iter().all(|&v| v < 0.0),
                "P0 fold CFVs should all be negative, got {:.6}", cfv[0]);
            let avg: f32 = cfv.iter().sum::<f32>() / nh as f32;
            // avg ≈ -50/1081 ≈ -0.046
            assert!(avg > -0.1 && avg < -0.01,
                "P0 fold avg CFV should be ≈ -0.046, got {:.6}", avg);
            println!("P0 fold (c=[{},{}], sp=100): avg_cfv = {:.6} (expected ≈ -0.046)", c0, c1, avg);
        }

        // P1 folded, P0 wins
        if fold_mask & 2 != 0 && c1 == 0 {
            found = true;
            let cfv = game.evaluate_terminal(0, i, &tree, &cfreach);
            assert!(cfv.iter().all(|&v| v > 0.0),
                "P0 winner CFVs should all be positive, got {:.6}", cfv[0]);
            let avg: f32 = cfv.iter().sum::<f32>() / nh as f32;
            assert!(avg > 0.01 && avg < 0.1,
                "P0 winner avg CFV should be ≈ +0.046, got {:.6}", avg);
            println!("P0 win (c=[{},{}], sp=100): avg_cfv = {:.6} (expected ≈ +0.046)", c0, c1, avg);
        }
    }
    assert!(found, "Should find at least one fold terminal");
}

#[test]
fn gt_showdown_payoff_magnitude() {
    // After normalization, CFVs should be small (≈ payoff / nc × cfreach)
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "Qs"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];

    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::River,
        starting_pot: 200, starting_stacks: vec![200, 200],
        initial_contributions: vec![0, 0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_hands(0);

    let opp_reach: Vec<f32> = vec![1.0f32; nh];
    let cfreach: Vec<Vec<f32>> = vec![opp_reach.clone(), opp_reach];

    for i in 0..tree.num_nodes() {
        let node = &tree.nodes[i];
        if !node.is_terminal() { continue; }
        let fold_mask = tree.get_folded_mask(i);
        if fold_mask != 0 { continue; }
        let c0 = tree.get_contribution(i, 0);
        let c1 = tree.get_contribution(i, 1);
        if c0 == 0 || c0 != c1 { continue; }

        let cfv = game.evaluate_terminal(0, i, &tree, &cfreach);
        assert!(cfv.iter().any(|&v| v > 0.0));
        assert!(cfv.iter().any(|&v| v < 0.0));

        // After normalization: max_cfv ≈ half_pot × (nh-1) / nc ≈ half_pot / nh
        // half_pot = 200/2 + c0. For c0=50: half_pot = 150, max ≈ 150/1081 ≈ 0.139
        let max_cfv = cfv.iter().cloned().fold(f32::MIN, f32::max);
        println!("Showdown c=[{},{}] sp=200: max_cfv={:.6}", c0, c1, max_cfv);
        assert!(max_cfv > 0.05 && max_cfv < 0.5,
            "Normalized max CFV should be ≈ 0.1-0.2, got {:.6}", max_cfv);
        break;
    }
}

#[test]
fn gt_regret_zero_sum() {
    // After 1 iteration, regret[check] + regret[bet] = 0 for each hand
    // (since regret = cfv[a] - cfv[σ], and σ is a probability distribution)
    let board: Vec<Card> = ["2h", "7d", "Ks", "4c", "Qs"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];

    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::River,
        starting_pot: 200, starting_stacks: vec![9500, 9500],
        initial_contributions: vec![0, 0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(0.5)], raise: vec![] },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_hands(0);

    let mut vcfr = VectorCfr::new(&tree, vec![nh, nh]);
    vcfr.run_sequential(&tree, &game, 1);
    let regrets = vcfr.regrets_slice();

    let mut checked = 0;
    for i in 0..tree.num_nodes() {
        let node = &tree.nodes[i];
        if !node.is_player() || node.num_children != 2 { continue; }
        let off = tree.infoset_offsets[i] as usize * MAX_NA * nh;
        let mut max_err = 0.0f32;
        for h in 0..nh {
            let r0 = regrets[off + 0 * nh + h];
            let r1 = regrets[off + 1 * nh + h];
            let err = (r0 + r1).abs();
            if err > max_err { max_err = err; }
        }
        assert!(max_err < 1.0,
            "Node {} (2 actions): regret sum error = {:.6}", i, max_err);
        checked += 1;
    }
    println!("{} decision nodes checked: all regret sums ≈ 0 ✓", checked);
}
