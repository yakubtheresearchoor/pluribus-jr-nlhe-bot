use std::cell::Cell;

use crate::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use crate::hand::eval::Hand;
use crate::solver::chance_table::ChanceTable;
use crate::solver::game::GameSpec;
use crate::solver::showdown::side_pot_showdown_cfv_with_rake;
use crate::tree::flat::FlatTree;

/// Game specification for flop-start games (2 chance transitions: turn + river).
/// Tracks both current turn and river cards for terminal evaluation.
pub struct FlopStartGame {
    table: FlopChanceTable,
    current_turn_card: Cell<Option<u8>>,
    current_river_card: Cell<Option<u8>>,
    // Base sorted arrays (flop-only hand ranks, for flop-level decisions)
    base_sorted_opp_str: Vec<u16>,
    base_sorted_opp_idx: Vec<u16>,
    base_sorted_pl_str: Vec<u16>,
    base_sorted_pl_idx: Vec<u16>,
}

/// Chance table for flop-start games.
/// Contains hand rank data for all (turn, river) combinations.
pub struct FlopChanceTable {
    pub hand_ranks_base: Vec<u16>,       // ranks with just flop (nh)
    pub valid_hand_indices: Vec<u16>,
    pub num_valid: usize,
    pub conflict: Vec<u8>,
    pub hand_cards: Vec<u8>,
    pub remaining_deck: Vec<u8>,          // 52 - 3 = 49 cards (turn candidates)
    pub turn_ranks: Vec<u16>,             // [52 * nh] ranks with (flop + turn)
    pub turn_sorted_str: Vec<u16>,        // [52 * num_opp * nh] sorted str per turn card
    pub turn_sorted_idx: Vec<u16>,        // [52 * num_opp * nh] sorted idx per turn card
    pub river_ranks: Vec<u16>,            // [52 * 52 * nh] ranks with (flop + turn + river)
    pub river_sorted_str: Vec<u16>,       // [52 * 52 * num_opp * nh] sorted str per (turn,river)
    pub river_sorted_idx: Vec<u16>,       // [52 * 52 * num_opp * nh] sorted idx per (turn,river)
    pub initial_weights: Vec<Vec<f32>>,
    pub num_players: u8,
    pub num_combinations: f64,
    // For each turn card, the remaining deck for river
    pub river_decks: Vec<Vec<u8>>,        // [52] -> Vec<u8> of remaining river cards
}

impl FlopChanceTable {
    pub fn compute_flop_start(
        known_board: &[Card],
        ranges: &[Vec<f32>],
        num_players: u8,
    ) -> Self {
        assert_eq!(known_board.len(), 3, "flop start needs exactly 3 known board cards");

        let board_set: Vec<u8> = known_board.iter().map(|&c| c as u8).collect();
        let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

        // Valid hands (don't conflict with flop)
        let mut valid_hand_indices = Vec::new();
        for idx in 0..NUM_POSSIBLE_HANDS {
            let (c1, c2) = index_to_card_pair(idx);
            if board_set.contains(&c1) || board_set.contains(&c2) { continue; }
            valid_hand_indices.push(idx as u16);
        }
        let num_valid = valid_hand_indices.len();

        // Base hand ranks (flop only) — use evaluate_internal for sorting only
        // NOTE: With 5 cards, evaluate() binary search fails; use evaluate_internal() directly
        let mut hand_ranks_base = vec![0u16; num_valid];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in known_board { hand = hand.add_card(bc as usize); }
            hand_ranks_base[i] = hand.evaluate_full();
        }

        // Conflict matrix
        let mut conflict = vec![0u8; num_valid * num_valid];
        for i in 0..num_valid {
            let (c1a, c1b) = index_to_card_pair(valid_hand_indices[i] as usize);
            for j in 0..num_valid {
                if i == j { conflict[i * num_valid + j] = 1; continue; }
                let (c2a, c2b) = index_to_card_pair(valid_hand_indices[j] as usize);
                if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b {
                    conflict[i * num_valid + j] = 1;
                }
            }
        }

        // Hand cards
        let mut hand_cards = vec![0u8; num_valid * 2];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            hand_cards[i * 2] = c1;
            hand_cards[i * 2 + 1] = c2;
        }

        // Remaining deck (turn candidates): 52 - 3 = 49 cards
        let remaining_deck: Vec<u8> = (0..52u8).filter(|c| !board_set.contains(c)).collect();

        let num_opp = num_players as usize - 1;
        let nh = num_valid;

        // Turn ranks: for each possible turn card, hand rank with (flop + turn)
        let mut turn_ranks = vec![0u16; 52 * nh];
        // Turn sorted arrays
        let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
        let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];

        // River ranks: for each (turn, river) pair, hand rank with full board
        let mut river_ranks = vec![0u16; 52 * 52 * nh];
        let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
        let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];

        // River decks per turn card
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];

        for &turn_card in &remaining_deck {
            // Compute turn ranks
            let turn_mask = board_mask | (1u64 << turn_card);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 {
                    turn_ranks[turn_card as usize * nh + i] = 0; // hand conflicts with turn
                    continue;
                }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in known_board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(turn_card as usize);
                turn_ranks[turn_card as usize * nh + i] = hand.evaluate_full();
            }

            // Sort turn ranks
            {
                let mut items: Vec<(u16, u16)> = (0..nh)
                    .map(|h| (turn_ranks[turn_card as usize * nh + h] + 1, h as u16))
                    .collect();
                items.sort_by_key(|&(s, _)| s);
                for oi in 0..num_opp {
                    let off = turn_card as usize * num_opp * nh + oi * nh;
                    for h in 0..nh {
                        turn_sorted_str[off + h] = items[h].0;
                        turn_sorted_idx[off + h] = items[h].1;
                    }
                }
            }

            // River cards for this turn
            let turn_plus_board: Vec<u8> = known_board.iter().chain(std::iter::once(&turn_card)).cloned().collect();
            let river_cards: Vec<u8> = (0..52u8).filter(|c| !turn_plus_board.contains(c)).collect();
            river_decks[turn_card as usize] = river_cards.clone();

            for &river_card in &river_cards {
                // Full board ranks
                for (i, &hi) in valid_hand_indices.iter().enumerate() {
                    let (c1, c2) = index_to_card_pair(hi as usize);
                    let full_mask = turn_mask | (1u64 << river_card);
                    if full_mask & (1u64 << c1) != 0 || full_mask & (1u64 << c2) != 0 {
                        river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + i] = 0;
                        continue;
                    }
                    let mut hand = Hand::new();
                    hand = hand.add_card(c1 as usize);
                    hand = hand.add_card(c2 as usize);
                    for &bc in known_board { hand = hand.add_card(bc as usize); }
                    hand = hand.add_card(turn_card as usize);
                    hand = hand.add_card(river_card as usize);
                    river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + i] = hand.evaluate_full();
                }

                // Sort river ranks for this (turn, river) pair
                {
                    let mut items: Vec<(u16, u16)> = (0..nh)
                        .map(|h| {
                            let r = river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + h];
                            (r + 1, h as u16)
                        })
                        .collect();
                    items.sort_by_key(|&(s, _)| s);
                    for oi in 0..num_opp {
                        let off = turn_card as usize * 52 * num_opp * nh
                            + river_card as usize * num_opp * nh + oi * nh;
                        for h in 0..nh {
                            river_sorted_str[off + h] = items[h].0;
                            river_sorted_idx[off + h] = items[h].1;
                        }
                    }
                }
            }
        }

        // Initial weights
        let mut initial_weights = Vec::with_capacity(num_players as usize);
        for p in 0..num_players as usize {
            let mut w = vec![0.0f32; num_valid];
            for (i, &hand_idx) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hand_idx as usize);
                let pair_idx = if c1 < c2 {
                    c1 as usize * (101 - c1 as usize) / 2 + c2 as usize - 1
                } else {
                    c2 as usize * (101 - c2 as usize) / 2 + c1 as usize - 1
                };
                w[i] = ranges[p][pair_idx];
            }
            initial_weights.push(w);
        }

        // num_combinations: count of non-conflicting hand tuples across all
        // players, reach-weighted by initial weights. MUST enumerate the same
        // set as the showdown numerator, or zero-sum breaks — the same lesson
        // the 3-player port re-established. Done via recursive enumeration
        // (equivalent to total_valid_reach_product at the top-level reach
        // state). No nh^np cap, no pairwise fallback: at production nh=50
        // 6max this is ~15.6B steps once per solve at startup; the same I-E
        // formula that will eventually replace the brute-force showdown will
        // also replace this. For now both share the same O(nh^K) ceiling.
        let nc = {
            let np = num_players as usize;
            fn enumerate(
                player: usize, np: usize, nh: usize,
                combined_mask: u64,
                hand_cards: &[u8],
                weights: &[Vec<f32>],
                current_weight: f64,
            ) -> f64 {
                if player == np {
                    return current_weight;
                }
                let mut total = 0.0f64;
                for h in 0..nh {
                    let mask_h: u64 = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
                    if combined_mask & mask_h != 0 { continue; }
                    let w = weights[player][h] as f64;
                    if w == 0.0 { continue; }
                    total += enumerate(
                        player + 1, np, nh,
                        combined_mask | mask_h,
                        hand_cards, weights,
                        current_weight * w,
                    );
                }
                total
            }
            enumerate(0, np, num_valid, 0, &hand_cards[..], &initial_weights, 1.0)
        };

        FlopChanceTable {
            hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
            remaining_deck, turn_ranks, turn_sorted_str, turn_sorted_idx,
            river_ranks, river_sorted_str, river_sorted_idx,
            initial_weights, num_players, num_combinations: nc,
            river_decks,
        }
    }

    /// Subset variant of `compute_flop_start`: build a `FlopChanceTable`
    /// containing ONLY the caller-provided `hand_indices` (a subset of
    /// the full 1326-hand space). The returned table has
    /// `num_valid == hand_indices.len()`.
    ///
    /// Semantics: this constructs a K-hand poker game where ONLY those K
    /// hands exist (for both players). It is NOT a downsample of the full
    /// nh≈1176 game — poker values depend on which hands are in play (e.g.,
    /// blocking, num_combinations, sorted ranks all change). The subset
    /// variant exists to make GPU parity tests fast at small K while still
    /// going through the production-API path (vs. hand-rolled chance
    /// tables, which have repeatedly drifted from production in scale-
    /// dependent ways).
    ///
    /// Argument contracts:
    ///   - `ranges[p]` is the same 1326-long vector as in `compute_flop_start`,
    ///     indexed by the (c1, c2) → pair_idx mapping used throughout the
    ///     project. Only entries at `hand_indices[i]`'s pair_idx are read.
    ///   - `hand_indices` must contain only non-blocking hands w.r.t. the flop
    ///     and must be unique.
    ///
    /// Validation: `compute_flop_start_subset(board, ranges, np, ALL_NONBLOCKING)`
    /// is asserted to produce a bit-identical `FlopChanceTable` to
    /// `compute_flop_start(board, ranges, np)` (no-regression sanity, in
    /// gpu_brute_force_unit or a dedicated equivalence test). The proper
    /// subset path (K < full) is validated against hand-computed K-hand
    /// poker math (turn_ranks, river_sorted_*, num_combinations all
    /// hand-verifiable at small K).
    pub fn compute_flop_start_subset(
        known_board: &[Card],
        ranges: &[Vec<f32>],
        num_players: u8,
        hand_indices: &[u16],
    ) -> Self {
        assert_eq!(known_board.len(), 3, "flop start needs exactly 3 known board cards");

        let board_set: Vec<u8> = known_board.iter().map(|&c| c as u8).collect();
        let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

        // Validate caller-provided hand indices: unique + non-blocking.
        let mut seen = std::collections::HashSet::new();
        for &hi in hand_indices {
            assert!(seen.insert(hi), "hand_indices contains duplicate {}", hi);
            let (c1, c2) = index_to_card_pair(hi as usize);
            assert!(!board_set.contains(&c1) && !board_set.contains(&c2),
                "hand_indices[{}]={} blocks on flop {:?}", seen.len() - 1, hi, board_set);
        }

        let valid_hand_indices: Vec<u16> = hand_indices.to_vec();
        let num_valid = valid_hand_indices.len();

        // Base hand ranks (flop only).
        let mut hand_ranks_base = vec![0u16; num_valid];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in known_board { hand = hand.add_card(bc as usize); }
            hand_ranks_base[i] = hand.evaluate_full();
        }

        // Conflict matrix (subset).
        let mut conflict = vec![0u8; num_valid * num_valid];
        for i in 0..num_valid {
            let (c1a, c1b) = index_to_card_pair(valid_hand_indices[i] as usize);
            for j in 0..num_valid {
                if i == j { conflict[i * num_valid + j] = 1; continue; }
                let (c2a, c2b) = index_to_card_pair(valid_hand_indices[j] as usize);
                if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b {
                    conflict[i * num_valid + j] = 1;
                }
            }
        }

        // Hand cards.
        let mut hand_cards = vec![0u8; num_valid * 2];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            hand_cards[i * 2] = c1;
            hand_cards[i * 2 + 1] = c2;
        }

        let remaining_deck: Vec<u8> = (0..52u8).filter(|c| !board_set.contains(c)).collect();
        let num_opp = num_players as usize - 1;
        let nh = num_valid;

        let mut turn_ranks = vec![0u16; 52 * nh];
        let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
        let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
        let mut river_ranks = vec![0u16; 52 * 52 * nh];
        let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
        let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];

        for &turn_card in &remaining_deck {
            let turn_mask = board_mask | (1u64 << turn_card);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 {
                    turn_ranks[turn_card as usize * nh + i] = 0;
                    continue;
                }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in known_board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(turn_card as usize);
                turn_ranks[turn_card as usize * nh + i] = hand.evaluate_full();
            }
            // Sorted turn ranks — INCLUDE ALL nh entries (blocked hands get rank 0 → strength 1).
            // This matches the production rule that prevents the GPU helper's hand_strength[0]
            // overwrite bug (three times found in test-harness drift).
            {
                let mut items: Vec<(u16, u16)> = (0..nh)
                    .map(|h| (turn_ranks[turn_card as usize * nh + h] + 1, h as u16))
                    .collect();
                items.sort_by_key(|&(s, _)| s);
                for oi in 0..num_opp {
                    let off = turn_card as usize * num_opp * nh + oi * nh;
                    for h in 0..nh {
                        turn_sorted_str[off + h] = items[h].0;
                        turn_sorted_idx[off + h] = items[h].1;
                    }
                }
            }

            let turn_plus_board: Vec<u8> = known_board.iter()
                .chain(std::iter::once(&turn_card)).cloned().collect();
            let river_cards: Vec<u8> = (0..52u8).filter(|c| !turn_plus_board.contains(c)).collect();
            river_decks[turn_card as usize] = river_cards.clone();

            for &river_card in &river_cards {
                for (i, &hi) in valid_hand_indices.iter().enumerate() {
                    let (c1, c2) = index_to_card_pair(hi as usize);
                    let full_mask = turn_mask | (1u64 << river_card);
                    if full_mask & (1u64 << c1) != 0 || full_mask & (1u64 << c2) != 0 {
                        river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + i] = 0;
                        continue;
                    }
                    let mut hand = Hand::new();
                    hand = hand.add_card(c1 as usize);
                    hand = hand.add_card(c2 as usize);
                    for &bc in known_board { hand = hand.add_card(bc as usize); }
                    hand = hand.add_card(turn_card as usize);
                    hand = hand.add_card(river_card as usize);
                    river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + i] = hand.evaluate_full();
                }
                {
                    let mut items: Vec<(u16, u16)> = (0..nh)
                        .map(|h| {
                            let r = river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + h];
                            (r + 1, h as u16)
                        })
                        .collect();
                    items.sort_by_key(|&(s, _)| s);
                    for oi in 0..num_opp {
                        let off = turn_card as usize * 52 * num_opp * nh
                            + river_card as usize * num_opp * nh + oi * nh;
                        for h in 0..nh {
                            river_sorted_str[off + h] = items[h].0;
                            river_sorted_idx[off + h] = items[h].1;
                        }
                    }
                }
            }
        }

        // Initial weights — read from caller's 1326-indexed ranges via pair_idx
        // (same formula as compute_flop_start).
        let mut initial_weights = Vec::with_capacity(num_players as usize);
        for p in 0..num_players as usize {
            let mut w = vec![0.0f32; num_valid];
            for (i, &hand_idx) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hand_idx as usize);
                let pair_idx = if c1 < c2 {
                    c1 as usize * (101 - c1 as usize) / 2 + c2 as usize - 1
                } else {
                    c2 as usize * (101 - c2 as usize) / 2 + c1 as usize - 1
                };
                w[i] = ranges[p][pair_idx];
            }
            initial_weights.push(w);
        }

        // num_combinations — same recursive enumeration as compute_flop_start,
        // restricted to the subset's hand_cards.
        let nc = {
            let np = num_players as usize;
            fn enumerate(
                player: usize, np: usize, nh: usize,
                combined_mask: u64,
                hand_cards: &[u8],
                weights: &[Vec<f32>],
                current_weight: f64,
            ) -> f64 {
                if player == np {
                    return current_weight;
                }
                let mut total = 0.0f64;
                for h in 0..nh {
                    let mask_h: u64 = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
                    if combined_mask & mask_h != 0 { continue; }
                    let w = weights[player][h] as f64;
                    if w == 0.0 { continue; }
                    total += enumerate(
                        player + 1, np, nh,
                        combined_mask | mask_h,
                        hand_cards, weights,
                        current_weight * w,
                    );
                }
                total
            }
            enumerate(0, np, num_valid, 0, &hand_cards[..], &initial_weights, 1.0)
        };

        FlopChanceTable {
            hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
            remaining_deck, turn_ranks, turn_sorted_str, turn_sorted_idx,
            river_ranks, river_sorted_str, river_sorted_idx,
            initial_weights, num_players, num_combinations: nc,
            river_decks,
        }
    }

    /// Hand-and-deck-subset variant of `compute_flop_start`. Combines the
    /// hand_indices argument from `compute_flop_start_subset` with explicit
    /// turn-card and river-deck restrictions, producing a `FlopChanceTable`
    /// at K hands × (caller-chosen turn cards) × (caller-chosen river decks
    /// per turn).
    ///
    /// Motivation: behavior-test purposes (does the algorithm converge,
    /// does the strategy look right) don't depend on running over all 49×48
    /// boards; restricting to a small board sample makes the test fast
    /// without changing what it validates. This variant lets behavior tests
    /// use the production-API path while keeping the board count small.
    ///
    /// Argument contracts:
    ///   - `hand_indices`: same as `compute_flop_start_subset` (subset of
    ///     non-blocking hands, unique, lengths match `ranges[p].len()`'s
    ///     pair_idx lookups).
    ///   - `turn_cards`: a subset of (0..52) \ board cards. Must contain no
    ///     duplicates and no board cards.
    ///   - `river_decks_subset[tc]`: for each turn card tc that appears in
    ///     `turn_cards`, the list of river cards to use. Each list must
    ///     contain no duplicates, no board cards, and no occurrence of tc
    ///     itself. Entries at indices NOT in `turn_cards` are ignored (and
    ///     conventionally empty).
    ///
    /// Validation: parallel structure to `compute_flop_start_subset`:
    /// (a) degenerate equivalence — calling with full hand set + full turn
    /// list + full river decks produces a `FlopChanceTable` bit-identical
    /// to `compute_flop_start`, (b) K-hand-and-restricted-deck hand
    /// anchor — at small K + 2x2 deck, every field is hand-verifiable.
    /// Full-nh production constructor with sampled runouts and
    /// **num_combinations pinned to 1.0**.
    ///
    /// TRAP SITE (burned twice, 2026-06-10/11): the subset constructor
    /// below computes `num_combinations` by recursive JOINT enumeration
    /// over all players' hands — O(nh^np), frozen forever at nh = 1176,
    /// np = 6. nc only scales cfv readout uniformly (σ is row-
    /// normalized; cross-flop comparisons stay consistent when every
    /// table uses the same nc), so production full-nh tables pin it at
    /// 1.0 instead. Strength/sort arrays are computed only for the
    /// sampled runouts. Unit initial weights.
    pub fn build_full_nh_sampled(
        flop: [Card; 3],
        num_players: u8,
        turn_cards: &[u8],
        river_decks_subset: &[Vec<u8>], // [52]
    ) -> Self {
        use crate::card::{index_to_card_pair, NUM_POSSIBLE_HANDS};
        use crate::hand::eval::Hand;
        let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
        let num_opp = num_players as usize - 1;

        let mut chosen: Vec<u16> = Vec::new();
        for idx in 0..NUM_POSSIBLE_HANDS {
            let (c1, c2) = index_to_card_pair(idx);
            if board_mask & ((1u64 << c1) | (1u64 << c2)) == 0 {
                chosen.push(idx as u16);
            }
        }
        let nh = chosen.len();
        let mut hand_cards = vec![0u8; nh * 2];
        for (i, &hi) in chosen.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            hand_cards[i * 2] = c1;
            hand_cards[i * 2 + 1] = c2;
        }
        let mut conflict = vec![0u8; nh * nh];
        for i in 0..nh {
            for j in 0..nh {
                if i == j {
                    conflict[i * nh + j] = 1;
                    continue;
                }
                let (a1, a2) = (hand_cards[i * 2], hand_cards[i * 2 + 1]);
                let (b1, b2) = (hand_cards[j * 2], hand_cards[j * 2 + 1]);
                if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 {
                    conflict[i * nh + j] = 1;
                }
            }
        }
        let mut hr = vec![0u16; nh];
        for i in 0..nh {
            let mut h = Hand::new()
                .add_card(hand_cards[i * 2] as usize)
                .add_card(hand_cards[i * 2 + 1] as usize);
            for &bc in &flop {
                h = h.add_card(bc as usize);
            }
            hr[i] = h.evaluate_full();
        }

        let mut turn_ranks = vec![0u16; 52 * nh];
        let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
        let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
        for &t in turn_cards {
            let tm = board_mask | (1u64 << t);
            for i in 0..nh {
                let (c1, c2) = (hand_cards[i * 2], hand_cards[i * 2 + 1]);
                if tm & ((1u64 << c1) | (1u64 << c2)) != 0 {
                    continue;
                }
                let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
                for &bc in &flop {
                    h = h.add_card(bc as usize);
                }
                h = h.add_card(t as usize);
                turn_ranks[t as usize * nh + i] = h.evaluate_full();
            }
            let mut items: Vec<(u16, u16)> =
                (0..nh).map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * num_opp * nh + oi * nh;
                for h in 0..nh {
                    turn_sorted_str[off + h] = items[h].0;
                    turn_sorted_idx[off + h] = items[h].1;
                }
            }
        }
        let mut river_ranks = vec![0u16; 52 * 52 * nh];
        let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
        let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
        for &t in turn_cards {
            let tm = board_mask | (1u64 << t);
            for &r in &river_decks_subset[t as usize] {
                let fm = tm | (1u64 << r);
                for i in 0..nh {
                    let (c1, c2) = (hand_cards[i * 2], hand_cards[i * 2 + 1]);
                    if fm & ((1u64 << c1) | (1u64 << c2)) != 0 {
                        continue;
                    }
                    let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
                    for &bc in &flop {
                        h = h.add_card(bc as usize);
                    }
                    h = h.add_card(t as usize).add_card(r as usize);
                    river_ranks[t as usize * 52 * nh + r as usize * nh + i] =
                        h.evaluate_full();
                }
                let mut items: Vec<(u16, u16)> = (0..nh)
                    .map(|h| {
                        (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16)
                    })
                    .collect();
                items.sort_by_key(|&(s, _)| s);
                for oi in 0..num_opp {
                    let off =
                        t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                    for h in 0..nh {
                        river_sorted_str[off + h] = items[h].0;
                        river_sorted_idx[off + h] = items[h].1;
                    }
                }
            }
        }
        let iw = vec![vec![1.0f32; nh]; num_players as usize];
        FlopChanceTable {
            hand_ranks_base: hr,
            valid_hand_indices: chosen,
            num_valid: nh,
            conflict,
            hand_cards,
            remaining_deck: turn_cards.to_vec(),
            turn_ranks,
            turn_sorted_str,
            turn_sorted_idx,
            river_ranks,
            river_sorted_str,
            river_sorted_idx,
            initial_weights: iw,
            num_players,
            num_combinations: 1.0,
            river_decks: river_decks_subset.to_vec(),
        }
    }

    pub fn compute_flop_start_subset_with_decks(
        known_board: &[Card],
        ranges: &[Vec<f32>],
        num_players: u8,
        hand_indices: &[u16],
        turn_cards: &[u8],
        river_decks_subset: &[Vec<u8>],   // [52]
    ) -> Self {
        assert_eq!(known_board.len(), 3, "flop start needs exactly 3 known board cards");
        assert_eq!(river_decks_subset.len(), 52,
            "river_decks_subset must be indexed by card 0..52");

        let board_set: Vec<u8> = known_board.iter().map(|&c| c as u8).collect();
        let board_mask: u64 = board_set.iter().fold(0u64, |m, &c| m | (1u64 << c));

        // Validate hand_indices (subset of non-blocking, unique).
        let mut seen_hands = std::collections::HashSet::new();
        for &hi in hand_indices {
            assert!(seen_hands.insert(hi), "hand_indices contains duplicate {}", hi);
            let (c1, c2) = index_to_card_pair(hi as usize);
            assert!(!board_set.contains(&c1) && !board_set.contains(&c2),
                "hand_indices[{}]={} blocks on flop", seen_hands.len() - 1, hi);
        }
        // Validate turn_cards (subset of non-board cards, unique).
        let mut seen_turns = std::collections::HashSet::new();
        for &tc in turn_cards {
            assert!(seen_turns.insert(tc), "turn_cards contains duplicate {}", tc);
            assert!(!board_set.contains(&tc),
                "turn_cards contains board card {}", tc);
        }
        // Validate river_decks_subset (each entry subset of non-(board ∪ tc)).
        for &tc in turn_cards {
            let rd = &river_decks_subset[tc as usize];
            let mut seen_rivers = std::collections::HashSet::new();
            for &rc in rd {
                assert!(seen_rivers.insert(rc),
                    "river_decks_subset[{}] contains duplicate {}", tc, rc);
                assert!(!board_set.contains(&rc),
                    "river_decks_subset[{}] contains board card {}", tc, rc);
                assert_ne!(rc, tc,
                    "river_decks_subset[{}] contains turn card itself", tc);
            }
        }

        let valid_hand_indices: Vec<u16> = hand_indices.to_vec();
        let num_valid = valid_hand_indices.len();

        // Base hand ranks (flop only) — same as compute_flop_start.
        let mut hand_ranks_base = vec![0u16; num_valid];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let mut hand = Hand::new();
            hand = hand.add_card(c1 as usize);
            hand = hand.add_card(c2 as usize);
            for &bc in known_board { hand = hand.add_card(bc as usize); }
            hand_ranks_base[i] = hand.evaluate_full();
        }

        // Conflict matrix.
        let mut conflict = vec![0u8; num_valid * num_valid];
        for i in 0..num_valid {
            let (c1a, c1b) = index_to_card_pair(valid_hand_indices[i] as usize);
            for j in 0..num_valid {
                if i == j { conflict[i * num_valid + j] = 1; continue; }
                let (c2a, c2b) = index_to_card_pair(valid_hand_indices[j] as usize);
                if c1a == c2a || c1a == c2b || c1b == c2a || c1b == c2b {
                    conflict[i * num_valid + j] = 1;
                }
            }
        }

        // Hand cards.
        let mut hand_cards = vec![0u8; num_valid * 2];
        for (i, &hi) in valid_hand_indices.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            hand_cards[i * 2] = c1;
            hand_cards[i * 2 + 1] = c2;
        }

        // remaining_deck = caller's turn_cards (in caller-provided order — the
        // production semantics is "list of valid turn cards for this hand."
        // We don't re-sort, mirroring how production compute_flop_start
        // builds remaining_deck from a filter of 0..52 (ascending by default).
        let remaining_deck: Vec<u8> = turn_cards.to_vec();
        let num_opp = num_players as usize - 1;
        let nh = num_valid;

        let mut turn_ranks = vec![0u16; 52 * nh];
        let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
        let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
        let mut river_ranks = vec![0u16; 52 * 52 * nh];
        let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
        let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];

        for &turn_card in turn_cards {
            let turn_mask = board_mask | (1u64 << turn_card);
            for (i, &hi) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if turn_mask & (1u64 << c1) != 0 || turn_mask & (1u64 << c2) != 0 {
                    turn_ranks[turn_card as usize * nh + i] = 0;
                    continue;
                }
                let mut hand = Hand::new();
                hand = hand.add_card(c1 as usize);
                hand = hand.add_card(c2 as usize);
                for &bc in known_board { hand = hand.add_card(bc as usize); }
                hand = hand.add_card(turn_card as usize);
                turn_ranks[turn_card as usize * nh + i] = hand.evaluate_full();
            }
            {
                let mut items: Vec<(u16, u16)> = (0..nh)
                    .map(|h| (turn_ranks[turn_card as usize * nh + h] + 1, h as u16))
                    .collect();
                items.sort_by_key(|&(s, _)| s);
                for oi in 0..num_opp {
                    let off = turn_card as usize * num_opp * nh + oi * nh;
                    for h in 0..nh {
                        turn_sorted_str[off + h] = items[h].0;
                        turn_sorted_idx[off + h] = items[h].1;
                    }
                }
            }

            // River cards for THIS turn from the caller-provided subset.
            let river_cards = &river_decks_subset[turn_card as usize];
            river_decks[turn_card as usize] = river_cards.clone();

            for &river_card in river_cards {
                for (i, &hi) in valid_hand_indices.iter().enumerate() {
                    let (c1, c2) = index_to_card_pair(hi as usize);
                    let full_mask = turn_mask | (1u64 << river_card);
                    if full_mask & (1u64 << c1) != 0 || full_mask & (1u64 << c2) != 0 {
                        river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + i] = 0;
                        continue;
                    }
                    let mut hand = Hand::new();
                    hand = hand.add_card(c1 as usize);
                    hand = hand.add_card(c2 as usize);
                    for &bc in known_board { hand = hand.add_card(bc as usize); }
                    hand = hand.add_card(turn_card as usize);
                    hand = hand.add_card(river_card as usize);
                    river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + i] = hand.evaluate_full();
                }
                {
                    let mut items: Vec<(u16, u16)> = (0..nh)
                        .map(|h| {
                            let r = river_ranks[turn_card as usize * 52 * nh + river_card as usize * nh + h];
                            (r + 1, h as u16)
                        })
                        .collect();
                    items.sort_by_key(|&(s, _)| s);
                    for oi in 0..num_opp {
                        let off = turn_card as usize * 52 * num_opp * nh
                            + river_card as usize * num_opp * nh + oi * nh;
                        for h in 0..nh {
                            river_sorted_str[off + h] = items[h].0;
                            river_sorted_idx[off + h] = items[h].1;
                        }
                    }
                }
            }
        }

        // Initial weights — same formula as compute_flop_start.
        let mut initial_weights = Vec::with_capacity(num_players as usize);
        for p in 0..num_players as usize {
            let mut w = vec![0.0f32; num_valid];
            for (i, &hand_idx) in valid_hand_indices.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hand_idx as usize);
                let pair_idx = if c1 < c2 {
                    c1 as usize * (101 - c1 as usize) / 2 + c2 as usize - 1
                } else {
                    c2 as usize * (101 - c2 as usize) / 2 + c1 as usize - 1
                };
                w[i] = ranges[p][pair_idx];
            }
            initial_weights.push(w);
        }

        // num_combinations — same enumeration, restricted to the K hands.
        let nc = {
            let np = num_players as usize;
            fn enumerate(
                player: usize, np: usize, nh: usize,
                combined_mask: u64,
                hand_cards: &[u8],
                weights: &[Vec<f32>],
                current_weight: f64,
            ) -> f64 {
                if player == np { return current_weight; }
                let mut total = 0.0f64;
                for h in 0..nh {
                    let mask_h: u64 = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
                    if combined_mask & mask_h != 0 { continue; }
                    let w = weights[player][h] as f64;
                    if w == 0.0 { continue; }
                    total += enumerate(
                        player + 1, np, nh,
                        combined_mask | mask_h,
                        hand_cards, weights,
                        current_weight * w,
                    );
                }
                total
            }
            enumerate(0, np, num_valid, 0, &hand_cards[..], &initial_weights, 1.0)
        };

        FlopChanceTable {
            hand_ranks_base, valid_hand_indices, num_valid, conflict, hand_cards,
            remaining_deck, turn_ranks, turn_sorted_str, turn_sorted_idx,
            river_ranks, river_sorted_str, river_sorted_idx,
            initial_weights, num_players, num_combinations: nc,
            river_decks,
        }
    }

    pub fn num_valid_hands(&self) -> usize { self.num_valid }
    pub fn hand_cards_gpu(&self) -> Vec<u8> { self.hand_cards.clone() }

    pub fn initial_weight_flat(&self) -> Vec<f32> {
        let nh = self.num_valid;
        let np = self.num_players as usize;
        let mut flat = Vec::with_capacity(np * nh);
        for p in 0..np { flat.extend_from_slice(&self.initial_weights[p]); }
        flat
    }

    pub fn sorted_opp_arrays_base(&self) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let nh = self.num_valid;
        let num_opp = self.num_players as usize - 1;
        let mut items: Vec<(u16, u16)> = (0..nh)
            .map(|h| (self.hand_ranks_base[h] + 1, h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        let mut s_str = vec![0u16; num_opp * nh];
        let mut s_idx = vec![0u16; num_opp * nh];
        for oi in 0..num_opp {
            for h in 0..nh {
                s_str[oi * nh + h] = items[h].0;
                s_idx[oi * nh + h] = items[h].1;
            }
        }
        let mut p_str = vec![0u16; nh];
        let mut p_idx = vec![0u16; nh];
        for h in 0..nh { p_str[h] = items[h].0; p_idx[h] = items[h].1; }
        (s_str, s_idx, p_str, p_idx, vec![u16::MAX; nh])
    }

    /// Get turn-card sorted arrays (for turn-level decisions after turn is dealt).
    /// Returns (opp_str, opp_idx, pl_str, pl_idx) for the given turn card.
    pub fn turn_sorted_arrays(&self, turn_card: u8) -> ( &[u16], &[u16], &[u16], &[u16] ) {
        let nh = self.num_valid;
        let num_opp = self.num_players as usize - 1;
        let base = turn_card as usize * num_opp * nh;
        let s_str = &self.turn_sorted_str[base..base + num_opp * nh];
        let s_idx = &self.turn_sorted_idx[base..base + num_opp * nh];
        // Player sorted = same as opponent for 2-player
        (s_str, s_idx, s_str, s_idx)
    }

    /// Get river-card sorted arrays for terminals after both turn and river are dealt.
    pub fn river_sorted_arrays(&self, turn_card: u8, river_card: u8) -> ( &[u16], &[u16], &[u16], &[u16] ) {
        let nh = self.num_valid;
        let num_opp = self.num_players as usize - 1;
        let base = turn_card as usize * 52 * num_opp * nh + river_card as usize * num_opp * nh;
        let s_str = &self.river_sorted_str[base..base + num_opp * nh];
        let s_idx = &self.river_sorted_idx[base..base + num_opp * nh];
        (s_str, s_idx, s_str, s_idx)
    }

    /// Turn chance outcomes (for the first chance transition: flop → turn).
    /// Returns the list of possible turn cards.
    pub fn turn_outcomes(&self) -> &[u8] {
        &self.remaining_deck
    }

    /// River chance outcomes for a given turn card (for the second chance transition: turn → river).
    pub fn river_outcomes(&self, turn_card: u8) -> &[u8] {
        &self.river_decks[turn_card as usize]
    }

    pub fn chance_probability_turn(&self, outcome: usize, hand: usize) -> f32 {
        let card = self.remaining_deck[outcome];
        let (c1, c2) = index_to_card_pair(self.valid_hand_indices[hand] as usize);
        if card == c1 || card == c2 { return 0.0; }
        let blocked = self.remaining_deck.iter().filter(|&&rc| rc == c1 || rc == c2).count();
        1.0 / (self.remaining_deck.len() as f32 - blocked as f32)
    }

    pub fn chance_probability_river(&self, turn_card: u8, outcome: usize, hand: usize) -> f32 {
        let river_deck = &self.river_decks[turn_card as usize];
        let card = river_deck[outcome];
        let (c1, c2) = index_to_card_pair(self.valid_hand_indices[hand] as usize);
        if card == c1 || card == c2 { return 0.0; }
        let blocked = river_deck.iter().filter(|&&rc| rc == c1 || rc == c2).count();
        1.0 / (river_deck.len() as f32 - blocked as f32)
    }

    /// Pre-compute turn chance probabilities for all (outcome, hand) pairs.
    /// Returns [n_turn * nh] where index = outcome * nh + hand.
    pub fn compute_turn_chance_prob(&self) -> Vec<f32> {
        let n_turn = self.remaining_deck.len();
        let nh = self.num_valid;
        let mut prob = vec![0.0f32; n_turn * nh];
        for ti in 0..n_turn {
            for h in 0..nh {
                prob[ti * nh + h] = self.chance_probability_turn(ti, h);
            }
        }
        prob
    }

    /// Pre-compute river chance probabilities for all (turn, river, hand) triples.
    /// Returns [n_turn * max_n_river * nh] where index = (ti * max_n_river + ri) * nh + h.
    /// Slots for invalid (turn, river) pairs are zero.
    pub fn compute_river_chance_prob(&self) -> Vec<f32> {
        let n_turn = self.remaining_deck.len();
        let nh = self.num_valid;
        let max_n_river = self.river_decks.iter().map(|d| d.len()).max().unwrap_or(0);
        let mut prob = vec![0.0f32; n_turn * max_n_river * nh];
        for (ti, &tc) in self.remaining_deck.iter().enumerate() {
            let n_river = self.river_decks[tc as usize].len();
            for ri in 0..n_river {
                for h in 0..nh {
                    let idx = (ti * max_n_river + ri) * nh + h;
                    prob[idx] = self.chance_probability_river(tc, ri, h);
                }
            }
        }
        prob
    }

    /// Pre-compute player sorted strength/indices per turn card.
    /// For 2-player, same as opponent sorted arrays.
    /// Returns [n_turn * nh].
    pub fn compute_river_board_mask(&self) -> Vec<f32> {
        let n_turn = self.remaining_deck.len();
        let nh = self.num_valid;
        let max_n_river = self.river_decks.iter().map(|d| d.len()).max().unwrap_or(0);
        let mut mask = vec![0.0f32; n_turn * max_n_river * nh];
        for (ti, &tc) in self.remaining_deck.iter().enumerate() {
            let n_river = self.river_decks[tc as usize].len();
            for ri in 0..n_river {
                let rc = self.river_decks[tc as usize][ri];
                for h in 0..nh {
                    let idx = (ti * max_n_river + ri) * nh + h;
                    let (c1, c2) = crate::card::index_to_card_pair(self.valid_hand_indices[h] as usize);
                    if c1 == tc || c2 == tc || c1 == rc || c2 == rc {
                        mask[idx] = 0.0;
                    } else {
                        mask[idx] = 1.0;
                    }
                }
            }
        }
        mask
    }

    pub fn compute_turn_pl_sorted(&self) -> (Vec<u16>, Vec<u16>) {
        // For 2-player, player sorted = opponent sorted (same hand ranking)
        (self.turn_sorted_str.clone(), self.turn_sorted_idx.clone())
    }

    /// Pre-compute player sorted strength/indices per (turn, river) pair.
    /// For 2-player, same as opponent sorted arrays.
    /// Returns [n_turn * max_n_river * nh].
    pub fn compute_river_pl_sorted(&self) -> (Vec<u16>, Vec<u16>) {
        (self.river_sorted_str.clone(), self.river_sorted_idx.clone())
    }
}

impl FlopStartGame {
    pub fn new(table: FlopChanceTable) -> Self {
        let (s_str, s_idx, p_str, p_idx, _) = table.sorted_opp_arrays_base();
        FlopStartGame {
            table,
            current_turn_card: Cell::new(None),
            current_river_card: Cell::new(None),
            base_sorted_opp_str: s_str,
            base_sorted_opp_idx: s_idx,
            base_sorted_pl_str: p_str,
            base_sorted_pl_idx: p_idx,
        }
    }

    pub fn table(&self) -> &FlopChanceTable { &self.table }

    pub fn set_turn_card(&self, card: u8) {
        self.current_turn_card.set(Some(card));
    }

    pub fn set_river_card(&self, card: u8) {
        self.current_river_card.set(Some(card));
    }

    pub fn clear_chance(&self) {
        self.current_turn_card.set(None);
        self.current_river_card.set(None);
    }
}

impl GameSpec for FlopStartGame {
    fn num_hands(&self, _player: u8) -> usize {
        self.table.num_valid
    }

    fn initial_weight(&self, player: u8) -> Vec<f32> {
        self.table.initial_weights[player as usize].clone()
    }

    fn evaluate_terminal(
        &self,
        traverser: u8,
        node_idx: usize,
        tree: &FlatTree,
        cfreach: &[Vec<f32>],
    ) -> Vec<f32> {
        let nh = self.table.num_valid;
        let np = self.table.num_players as usize;
        let num_opp = np - 1;
        let fold_mask = tree.get_folded_mask(node_idx);

        let mut contributions = vec![0i32; np];
        for p in 0..np {
            contributions[p] = tree.get_contribution(node_idx, p as u8);
        }

        // Filter opponent reach to exclude hands conflicting with turn/river cards.
        // The initial_weight already excludes flop-blocking hands, but opponent hands
        // containing the turn or river card are still included and must be zeroed out.
        let board_cards: Vec<u8> = match (self.current_turn_card.get(), self.current_river_card.get()) {
            (Some(tc), Some(rc)) => vec![tc, rc],
            (Some(tc), None) => vec![tc],
            (None, _) => vec![],
        };

        let filtered_opp_reach: Vec<Vec<f32>> = (0..np)
            .filter(|&p| p != traverser as usize)
            .map(|p| {
                let raw = &cfreach[p];
                if board_cards.is_empty() {
                    raw.to_vec()
                } else {
                    let mut filtered = raw.clone();
                    for h in 0..nh {
                        if filtered[h] != 0.0 {
                            let c1 = self.table.hand_cards[h * 2];
                            let c2 = self.table.hand_cards[h * 2 + 1];
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

        let opp_reach: Vec<&[f32]> = filtered_opp_reach.iter().map(|v| v.as_slice()).collect();

        // Select sorted arrays based on current turn + river cards.
        // Slice 1.6: rake threaded from FlatTree. flop_seen=true because
        // this is a flop-start game (the flop has been dealt before this
        // terminal can be reached, by tree construction).
        let rake_rate = tree.rake_rate as f32;
        let rake_cap = tree.rake_cap as f32;
        let cfv = match (self.current_turn_card.get(), self.current_river_card.get()) {
            (Some(tc), Some(rc)) => {
                // Full board: use (turn, river) sorted arrays
                let (os, oi, ps, pi) = self.table.river_sorted_arrays(tc, rc);
                side_pot_showdown_cfv_with_rake(
                    &opp_reach, &self.table.hand_cards, nh,
                    os, oi, ps, pi,
                    &contributions, fold_mask, traverser as usize, self.table.num_players,
                    tree.starting_pot,
                    rake_rate, rake_cap, true,
                )
            }
            (Some(tc), None) => {
                // Turn only (shouldn't happen at terminals, but handle gracefully)
                let (os, oi, ps, pi) = self.table.turn_sorted_arrays(tc);
                side_pot_showdown_cfv_with_rake(
                    &opp_reach, &self.table.hand_cards, nh,
                    os, oi, ps, pi,
                    &contributions, fold_mask, traverser as usize, self.table.num_players,
                    tree.starting_pot,
                    rake_rate, rake_cap, true,
                )
            }
            (None, _) => {
                // No chance dealt yet (shouldn't happen at terminals)
                side_pot_showdown_cfv_with_rake(
                    &opp_reach, &self.table.hand_cards, nh,
                    &self.base_sorted_opp_str, &self.base_sorted_opp_idx,
                    &self.base_sorted_pl_str, &self.base_sorted_pl_idx,
                    &contributions, fold_mask, traverser as usize, self.table.num_players,
                    tree.starting_pot,
                    rake_rate, rake_cap, true,
                )
            }
        };

        // Normalize by num_combinations
        let nc = self.table.num_combinations as f32;
        if nc > 0.0 {
            let mut result = cfv;
            for h in 0..nh { result[h] /= nc; }
            result
        } else {
            cfv
        }
    }

    fn chance_probability(&self, outcome: usize, hand: usize) -> f32 {
        match self.current_turn_card.get() {
            Some(tc) => self.table.chance_probability_river(tc, outcome, hand),
            None => self.table.chance_probability_turn(outcome, hand),
        }
    }

    fn num_chance_outcomes(&self) -> usize {
        match self.current_turn_card.get() {
            Some(tc) => self.table.river_decks[tc as usize].len(),
            None => self.table.remaining_deck.len(),
        }
    }

    fn set_chance_outcome(&self, outcome: usize) {
        match self.current_turn_card.get() {
            Some(tc) => {
                // Setting river card for the current turn
                let deck = self.table.river_outcomes(tc);
                let card = deck[outcome];
                self.current_river_card.set(Some(card));
            }
            None => {
                // Setting turn card
                let card = self.table.remaining_deck[outcome];
                self.current_turn_card.set(Some(card));
            }
        }
    }

    fn clear_chance_outcome(&self) {
        if self.current_river_card.get().is_some() {
            self.current_river_card.set(None);
        } else {
            self.current_turn_card.set(None);
        }
    }

    fn num_combinations(&self) -> f64 {
        self.table.num_combinations
    }
}
