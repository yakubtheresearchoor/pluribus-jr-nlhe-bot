// Hand-trace adjudication at fold terminal node[15] of the HU symmetric
// [5,5] tree. Companion to handtrace_cpu_node13.rs (showdown terminal).
//
// Purpose: determine if Gate 4's brute-force oracle (side_pot_showdown_cfv)
// shares the GPU's fold-terminal bug or not. If oracle returns the
// hand-traced -0.156, Gate 4 just has a coverage gap (the oracle is sound,
// it just doesn't test this case). If oracle returns the GPU's wrong
// -0.234, the oracle is compromised and the "validated against brute-force"
// claim across the project inherits doubt.

use solver_core::card::{card_from_str, index_to_card_pair, Card};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::showdown::side_pot_showdown_cfv;
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
    button_player: None,
            max_bets_per_street: None,

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

#[test]
fn gate4_oracle_at_fold_terminal_node15() {
    let (tree, table) = build_minimal_table();
    let game = FlopStartGame::new(table);
    let table_ref = game.table();
    let nh = 4usize;
    let np = 2usize;

    // node[15] is a fold terminal (traverser=0 folded). Hand-traced:
    //   opp_reach (p1) at this node = [0.125, 0.125, 0.125, 0.125]
    //   payoff = -traverser_investment = -(starting_pot/np + c_t)
    //          = -(10/2 + 5) = -10
    //   For each h, cfreach = opp_reach_sum - cfreach_minus[c1] - cfreach_minus[c2]
    //   Since no opp hand shares cards with any other opp hand here, and the
    //   traverser's hand's 2 cards each block 1 opp hand (the one containing
    //   that card), cfreach = 0.5 - 0.125 - 0.125 = 0.25 for ALL hands.
    //   CFV(h) = -10 * 0.25 / 16 = -0.15625
    //
    // GPU produces -0.234375 (provably wrong by factor 1.5).
    // QUESTION: Does Gate 4's oracle (side_pot_showdown_cfv) produce the
    // hand-traced correct value, or does it agree with the GPU's wrong one?

    let opp_reach_p1: Vec<f32> = vec![0.125, 0.125, 0.125, 0.125];
    let opp_reach_views: Vec<&[f32]> = vec![&opp_reach_p1];
    let contributions: Vec<i32> = (0..np)
        .map(|p| tree.get_contribution(15, p as u8))
        .collect();
    let fold_mask = tree.get_folded_mask(15);

    eprintln!("\n=== Gate 4 oracle adjudication at fold terminal node[15] ===");
    eprintln!("Context:");
    eprintln!("  contributions = {:?}", contributions);
    eprintln!("  fold_mask = {:b} (p0 folded, p1 active)", fold_mask);
    eprintln!("  opp_reach_p1 = {:?}", opp_reach_p1);
    eprintln!("  hand_cards = {:?}", &table_ref.hand_cards);
    eprintln!("  starting_pot = {}", tree.starting_pot);
    eprintln!("  num_combinations = {}\n", table_ref.num_combinations);

    // CORRECTED after independent_fold_enumerator.rs settled the question:
    // direct first-principles enumeration (no inclusion-exclusion shortcut)
    // gives -0.234375 per hand. The GPU is correct. The CPU has the bug.
    // The earlier "hand-trace says -0.156" was mentally executing the same
    // buggy CPU formula, not first-principles enumeration.
    eprintln!("Three values to compare:");
    eprintln!("  HAND-TRACE (direct enumeration, ground truth): -0.234375");
    eprintln!("  GPU multiway_brute_force_showdown:             -0.234375 (matches truth)");
    eprintln!("  Gate 4 oracle (side_pot_showdown_cfv):         TBD (expected: -0.15625 buggy)\n");

    // Pick any sorted arrays — fold-terminal fast path doesn't use them.
    let tc_card = table_ref.remaining_deck[0];
    let rc_card = table_ref.river_decks[tc_card as usize][0];
    let (opp_str, opp_idx, pl_str, pl_idx) = table_ref.river_sorted_arrays(tc_card, rc_card);

    let cfv_out = side_pot_showdown_cfv(
        &opp_reach_views, &table_ref.hand_cards, nh,
        opp_str, opp_idx, pl_str, pl_idx,
        &contributions, fold_mask, 0, 2,
        tree.starting_pot,
    );
    let nc = table_ref.num_combinations as f32;
    let cfv_normalized: Vec<f32> = cfv_out.iter().map(|&v| v / nc).collect();
    eprintln!("Gate 4 oracle output: raw={:?} /num_combinations={:?}", cfv_out, cfv_normalized);

    let oracle_val = cfv_normalized[0];
    let truth = -0.234375f32; // from independent_fold_enumerator.rs
    let oracle_buggy_value = -0.15625f32; // what side_pot_showdown_cfv currently returns

    let oracle_matches_truth = (oracle_val - truth).abs() < 1e-4;
    let oracle_matches_buggy = (oracle_val - oracle_buggy_value).abs() < 1e-4;

    eprintln!("\nADJUDICATION:");
    if oracle_matches_truth {
        eprintln!("  ✓ Oracle ({}) matches direct enumeration ({}). ", oracle_val, truth);
        eprintln!("  → side_pot_showdown_cfv has been FIXED. Gate 4 oracle is now sound.");
        eprintln!("  → If this is the first run after the inclusion-exclusion fix: fix confirmed.");
    } else if oracle_matches_buggy {
        eprintln!("  ✗ Oracle ({}) is WRONG; direct enumeration says {}. ", oracle_val, truth);
        eprintln!("  → Gate 4 oracle (side_pot_showdown_cfv) has the inclusion-exclusion bug.");
        eprintln!("  → Every 'validated against brute-force' claim using this oracle inherits doubt.");
        eprintln!("  → #39 must include independent showdown oracle, not just orchestration.");
        eprintln!("  → FIX: showdown.rs lines 289-305 add `+ opp_reach[0][h]` to cfreach formula");
        eprintln!("     (the self-hand's reach that minus[c1]+minus[c2] double-subtracts).");
    } else {
        eprintln!("  ?? Oracle = {} matches neither truth ({}) nor known-buggy value ({}).",
            oracle_val, truth, oracle_buggy_value);
        eprintln!("  → Three-way disagreement. Need deeper investigation.");
    }

    // Don't assert — this test is a diagnostic. Currently expected to FAIL
    // (oracle returns the buggy value). After the inclusion-exclusion fix
    // lands in showdown.rs, this should match the truth.
    if !oracle_matches_truth {
        eprintln!("\n  NOTE: this test currently expects the bug to be present.");
        eprintln!("  After the fix to side_pot_showdown_cfv lands, oracle should match truth.");
    }
}
