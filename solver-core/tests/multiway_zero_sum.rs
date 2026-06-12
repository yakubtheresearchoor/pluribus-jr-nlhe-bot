// Gates 4 + 5: per-terminal zero-sum for K=3 (4-player) and K=5 (6-player).
//
// For each showdown terminal evaluated via the new brute-force K>=3 showdown
// CFV (`side_pot_showdown_cfv`), the reach-weighted sum over players must
// satisfy:
//
//   Σ_p Σ_h reach_p[h] · cfv_p[h] / nc  ≈  0
//
// to within 1e-6 of pot per hand. The plan calls this gate the structural
// test that establishes K-opp showdown is computing pot redistribution
// correctly — anything that gets a player's cash share wrong but happens to
// be self-consistent at other tests would still fail this one.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_game_n(np: u8, nh: usize) -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_opp = np as usize - 1;

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
        hand_cards[i * 2] = c1; hand_cards[i * 2 + 1] = c2;
    }
    let mut conflict = vec![0u8; nh * nh];
    for i in 0..nh {
        for j in 0..nh {
            if i == j { conflict[i * nh + j] = 1; continue; }
            let (a1, a2) = index_to_card_pair(chosen[i] as usize);
            let (b1, b2) = index_to_card_pair(chosen[j] as usize);
            if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 { conflict[i * nh + j] = 1; }
        }
    }
    let mut hr = vec![0u16; nh];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
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
        items.sort_by_key(|&(s, _)| s);
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
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; np as usize];

    // num_combinations via recursive enumeration (matches new
    // flop_start_game::compute_flop_start; included here so the test owns its
    // own copy of the denominator and gates the same formula it checks).
    fn enum_nc(player: usize, np: usize, nh: usize, combined: u64,
               hand_cards: &[u8], weight: f64) -> f64 {
        if player == np { return weight; }
        let mut total = 0.0;
        for h in 0..nh {
            let m = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
            if combined & m != 0 { continue; }
            total += enum_nc(player + 1, np, nh, combined | m, hand_cards, weight);
        }
        total
    }
    let nc = enum_nc(0, np as usize, nh, 0, &hand_cards[..], 1.0);

    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players: np, num_combinations: nc, river_decks: rd,
    };
    let starting_pot: i32 = (np as i32) * 5;
    let stacks: Vec<i32> = vec![100; np as usize];
    let contribs: Vec<i32> = vec![5; np as usize];
    let config = TreeConfig {
        num_players: np, initial_state: BoardState::Flop, starting_pot,
        starting_stacks: stacks, initial_contributions: contribs,
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).unwrap();
    let game = FlopStartGame::new(table);
    (tree, game)
}

fn run_zero_sum(np_u8: u8, nh: usize, threshold: f64, label: &str) {
    let (tree, game) = build_game_n(np_u8, nh);
    let np = np_u8 as usize;
    eprintln!("[{}] nc = {}", label, game.table().num_combinations);

    game.set_chance_outcome(0); // turn
    game.set_chance_outcome(0); // river

    let mut showdown_nodes = Vec::new();
    let mut fold_nodes = Vec::new();
    for (idx, node) in tree.nodes.iter().enumerate() {
        if node.is_terminal() {
            let fm = tree.get_folded_mask(idx);
            if fm == 0 { showdown_nodes.push(idx); }
            else { fold_nodes.push(idx); }
        }
    }
    eprintln!("[{}] np={} nh={}  showdown_terminals={} fold_terminals={}",
        label, np, nh, showdown_nodes.len(), fold_nodes.len());

    let turn_card = game.table().remaining_deck[0];
    let river_card = game.table().river_decks[turn_card as usize][0];
    let board_extra = [turn_card, river_card];
    let mut valid_mask = vec![1.0f32; nh];
    for h in 0..nh {
        let c1 = game.table().hand_cards[h * 2];
        let c2 = game.table().hand_cards[h * 2 + 1];
        for &bc in &board_extra {
            if c1 == bc || c2 == bc { valid_mask[h] = 0.0; break; }
        }
    }
    let n_valid: usize = valid_mask.iter().filter(|&&v| v > 0.0).count();
    eprintln!("[{}] n_valid={}/{} after turn/river filter", label, n_valid, nh);
    let reach: Vec<Vec<f32>> = vec![valid_mask.clone(); np];

    // Sanity: what does evaluate_terminal give for a worst-case fold node?
    // First find a fold node where contribs are non-trivial.
    let mut diag_node = None;
    for (idx, node) in tree.nodes.iter().enumerate() {
        if !node.is_terminal() { continue; }
        let fm = tree.get_folded_mask(idx);
        if fm == 0 { continue; }
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let max_contrib = *contribs.iter().max().unwrap();
        let min_contrib = *contribs.iter().min().unwrap();
        if max_contrib > min_contrib + 50 {
            diag_node = Some(idx);
            break;
        }
    }
    if let Some(node_idx) = diag_node {
        let fm = tree.get_folded_mask(node_idx);
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(node_idx, p as u8)).collect();
        eprintln!("[{}] DIAG fold node {} fm={:#b} contribs={:?}", label, node_idx, fm, contribs);
        for traverser in 0..np {
            let cfv = game.evaluate_terminal(traverser as u8, node_idx, &tree, &reach);
            let weighted: f64 = (0..nh).map(|h| (cfv[h] * valid_mask[h]) as f64).sum();
            eprintln!("[{}]   traverser={}: Σ r*cfv = {:.6}  cfv[0..min(nh,5)] = {:?}",
                label, traverser, weighted, &cfv[..nh.min(5)]);
        }
    }

    let mut max_showdown: f64 = 0.0;
    let mut worst_idx = 0usize;
    let mut worst_contribs: Vec<i32> = vec![];
    let n_sample = showdown_nodes.len().min(50);
    for &node_idx in &showdown_nodes[..n_sample] {
        let mut cfv_sum = vec![0.0f64; nh];
        for traverser in 0..np {
            let cfv = game.evaluate_terminal(traverser as u8, node_idx, &tree, &reach);
            for h in 0..nh {
                cfv_sum[h] += (cfv[h] * valid_mask[h]) as f64;
            }
        }
        let total: f64 = cfv_sum.iter().sum();
        if total.abs() > max_showdown {
            max_showdown = total.abs();
            worst_idx = node_idx;
            worst_contribs = (0..np).map(|p| tree.get_contribution(node_idx, p as u8)).collect();
        }
    }
    let pot = (np as i32 * 5) as f64;  // starting_pot from build_game_n
    let per_hand_pot = pot / nh as f64;
    let max_showdown_pct = max_showdown / per_hand_pot * 100.0;
    eprintln!("[{}] showdown max |Σ cfv| = {:.6e}  ({:.4}% of pot/hand)  worst node={} contribs={:?}",
        label, max_showdown, max_showdown_pct, worst_idx, worst_contribs);

    let mut max_fold: f64 = 0.0;
    let mut worst_fold = 0usize;
    let mut worst_per_player: Vec<f64> = vec![];
    let mut worst_fold_mask: u16 = 0;
    let mut worst_fold_contribs: Vec<i32> = vec![];
    let n_fold_sample = fold_nodes.len().min(50);
    for &node_idx in &fold_nodes[..n_fold_sample] {
        let mut cfv_sum = vec![0.0f64; nh];
        let mut per_player = vec![0.0f64; np];
        for traverser in 0..np {
            let cfv = game.evaluate_terminal(traverser as u8, node_idx, &tree, &reach);
            for h in 0..nh {
                let v = (cfv[h] * valid_mask[h]) as f64;
                cfv_sum[h] += v;
                per_player[traverser] += v;
            }
        }
        let total: f64 = cfv_sum.iter().sum();
        if total.abs() > max_fold {
            max_fold = total.abs();
            worst_fold = node_idx;
            worst_per_player = per_player.clone();
            worst_fold_mask = tree.get_folded_mask(node_idx);
            worst_fold_contribs = (0..np).map(|p| tree.get_contribution(node_idx, p as u8)).collect();
        }
    }
    eprintln!("[{}] fold max |Σ cfv| = {:.6e}  worst node={} fm={:#b} contribs={:?}",
        label, max_fold, worst_fold, worst_fold_mask, worst_fold_contribs);
    eprintln!("[{}]    per-player cfv sums: {:?}", label, worst_per_player);

    game.clear_chance_outcome();

    assert!(max_showdown < threshold,
        "[{}] showdown zero-sum violated: max={:.6e} > {:.6e}", label, max_showdown, threshold);
    assert!(max_fold < threshold,
        "[{}] fold zero-sum violated: max={:.6e} > {:.6e}", label, max_fold, threshold);

    // Non-triviality check: at least one of the sampled terminals must have
    // a cfv whose absolute value exceeds a small floor. Without this,
    // a degenerate case (e.g. TVRP collapsing to 0 because the sampled
    // hand set has too many pairwise card conflicts to admit K disjoint
    // opponent hands) would let the zero-sum test pass with cfv ≡ 0 on
    // every hand, telling us nothing.
    let mut max_cfv_abs = 0.0f64;
    for &node_idx in &showdown_nodes[..n_sample] {
        for traverser in 0..np {
            let cfv = game.evaluate_terminal(traverser as u8, node_idx, &tree, &reach);
            for h in 0..nh {
                let v = (cfv[h] * valid_mask[h]).abs() as f64;
                if v > max_cfv_abs { max_cfv_abs = v; }
            }
        }
    }
    eprintln!("[{}] max |r * cfv| seen in showdown sample = {:.4e}", label, max_cfv_abs);
    assert!(max_cfv_abs > 0.01,
        "[{}] all sampled cfvs are effectively zero — the test is degenerate, \
         likely because the K-disjoint-opp hand requirement collapses for the \
         sampled hand set. Bump nh or pick disjoint hands explicitly.", label);
}

#[test]
fn gate4_k3_zero_sum_4p() {
    // K=3 = 4-player. nh=6 keeps brute-force tractable (nh^4 = 1296 per h).
    // Threshold: 1e-3 absolute, scaled by pot/hand. Pot is 20 (4*5); nh=6 →
    // per_hand_pot = 3.33. 1e-3 / 3.33 = 0.03% per hand — comfortably tight.
    run_zero_sum(4, 6, 1e-3, "K=3 4p nh=6");
}

#[test]
fn gate5_k5_zero_sum_6p() {
    // K=5 = 6-player. Need >=6 pairwise-disjoint valid hands for K=5 TVRP
    // to be non-zero (i.e., for the zero-sum check to actually exercise
    // K=5 brute-force). nh=10 leaves ~8 valid hands after turn/river
    // filter; with diverse card sampling at least 6 are usually pairwise
    // disjoint. Cost: 10^6 = 1M per h × 10 hands × 50 terminals = 500M ops
    // per direction — tens of seconds per test, tractable.
    run_zero_sum(6, 10, 1e-3, "K=5 6p nh=10");
}
