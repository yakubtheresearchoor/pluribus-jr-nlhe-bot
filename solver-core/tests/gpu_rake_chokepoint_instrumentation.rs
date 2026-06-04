//! Slice 2 Phase B Step 5: chokepoint instrumentation (PERMANENT CI guard).
//!
//! Per the lead's directive: "the live path is now a single chokepoint
//! (multiway_brute_force_showdown). The instrumentation's job is to
//! confirm every production terminal routes through that chokepoint and
//! the chokepoint applied rake (or correctly skipped it). That's the
//! completeness proof, and it's the thing that catches the failure mode
//! this whole arc kept hitting: a payoff path that bypasses the rake."
//!
//! ## How the instrumentation works
//!
//! Each production payoff-computing kernel (vcfr_bottom_up at flop,
//! vcfr_bottom_up_batched at turn/river) writes a marker to a per-
//! (terminal-node, hand) u8 buffer RIGHT AFTER calling the rake-correct
//! `multiway_brute_force_showdown` helper:
//!
//!   marker[node, h] = 1  if flop_seen=true  (rake-applied)
//!   marker[node, h] = 2  if flop_seen=false (rake-correctly-skipped
//!                                            per no-flop-no-drop)
//!   marker[node, h] = 0  if untouched       (BUG: terminal bypassed
//!                                            the chokepoint)
//!
//! The buffer is initialized to zero. After a solve, ALL terminal-node-
//! hand cells must be 1 or 2. Any 0 at a terminal indicates a payoff
//! path that does not route through the chokepoint — i.e., a rake-free
//! site reintroduced by a future kernel change. The instrumentation
//! catches it immediately.
//!
//! ## Standing question applied to the instrumentation itself
//!
//! Per the lead: "the instrumentation has to be verified to fire at every
//! terminal the production solve evaluates, which you can check by
//! confirming it fires the expected number of times for a solve whose
//! terminal count you know."
//!
//! The test counts the marker writes and compares to the expected
//! count (= num_terminal_nodes × nh). If the count doesn't match, the
//! instrumentation itself is false-green (present but not firing at
//! every terminal). The seventh false-green pattern (test that exists
//! but doesn't validate) is what this counter-check guards against.
//!
//! ## Permanent CI: NOT #[ignore]
//!
//! Per the lead: "keep it permanent because even with chokepoint, a future
//! change could add a payoff path that bypasses the helper, and the
//! instrumentation is the standing guard."
//!
//! This test runs in every CI pass. If a future change (especially the
//! real-time-search work that will touch these kernels) reintroduces a
//! rake-free site, this test fails immediately.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use solver_core::tree::flat::NODE_TYPE_TERMINAL;

/// Build a minimal HU tree (production-representative for the chokepoint
/// instrumentation check). The instrumentation has to fire at every
/// terminal across all three zones (flop, turn, river).
fn build_hu_minimal_table() -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));

    let num_players = 2u8;
    let num_opp = 1usize;
    let nh = 6usize;

    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = (all_valid.len() / nh).max(1);
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i*2] = c1; hand_cards[i*2+1] = c2;
    }
    let mut conflict = vec![0u8; nh*nh];
    for i in 0..nh { for j in 0..nh {
        if i == j { conflict[i*nh+j] = 1; continue; }
        let (a1,a2) = index_to_card_pair(chosen[i] as usize);
        let (b1,b2) = index_to_card_pair(chosen[j] as usize);
        if a1==b1 || a1==b2 || a2==b1 || a2==b2 { conflict[i*nh+j] = 1; }
    }}
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
        let mut items: Vec<(u16, u16)> = (0..nh)
            .map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
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
                river_ranks[t as usize * 52 * nh + r as usize * nh + i] =
                    h.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh)
                .map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16))
                .collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh {
                    river_sorted_str[off + h] = items[h].0;
                    river_sorted_idx[off + h] = items[h].1;
                }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; num_players as usize];

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
    let nc = enum_nc(0, num_players as usize, nh, 0, &hand_cards[..], 1.0);

    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players, num_combinations: nc, river_decks: rd,
    };
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.05, rake_cap: 1000.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build");
    (tree, table)
}

#[test]
fn chokepoint_instrumentation_every_terminal_marked() {
    // PERMANENT CI guard: every production payoff-computing terminal
    // must route through the chokepoint (multiway_brute_force_showdown)
    // which sets the rake marker. If any terminal-node-hand is unmarked
    // after a solve, a payoff path bypassed the chokepoint — a bug.
    //
    // Also verifies the instrumentation isn't itself false-green
    // (present but not firing) by counting marker writes and asserting
    // count == num_terminals × nh.

    let (tree, table) = build_hu_minimal_table();
    let nh = table.num_valid;
    let game = FlopStartGame::new(table);

    let cpu = FlopStartVectorCfr::new(&tree, game.table());
    let ctx = MetalContext::new().expect("Metal context");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Run one iteration of the production solver. Every terminal in the
    // tree should be visited and write its marker.
    gpu.run(&ctx, &tree, &game, 1);

    let markers = gpu.download_rake_marker();
    let nn = tree.num_nodes();
    assert_eq!(markers.len(), nn * nh, "marker buffer size mismatch");

    // ── Standing question check 1: count of marker writes ──
    // Count terminal nodes in the tree. The instrumentation MUST fire
    // exactly num_terminal_nodes × nh times.
    let mut num_terminals: usize = 0;
    let mut terminal_node_ids: Vec<usize> = Vec::new();
    for (i, node) in tree.nodes.iter().enumerate() {
        if node.node_type == NODE_TYPE_TERMINAL {
            num_terminals += 1;
            terminal_node_ids.push(i);
        }
    }

    let expected_marked_cells = num_terminals * nh;
    let actual_marked_cells: usize = markers.iter().filter(|&&v| v == 1 || v == 2).count();

    eprintln!("Chokepoint instrumentation report:");
    eprintln!("  Tree: {} nodes total, {} terminal nodes, nh={}", nn, num_terminals, nh);
    eprintln!("  Expected marker writes (terminals × nh): {}", expected_marked_cells);
    eprintln!("  Actual marker writes:                   {}", actual_marked_cells);

    assert_eq!(
        actual_marked_cells, expected_marked_cells,
        "Instrumentation firing-count mismatch. Expected {} marker writes \
         (num_terminals={} × nh={}), got {}. If actual < expected, some \
         terminals were not processed (instrumentation false-green: present \
         but not firing at every terminal). If actual > expected, marker \
         was written at non-terminal nodes (kernel bug).",
        expected_marked_cells, num_terminals, nh, actual_marked_cells,
    );

    // ── Standing question check 2: every terminal × hand cell is marked ──
    // For each terminal node, check all nh cells are 1 or 2 (not 0).
    // A 0 means that (node, hand) bypassed the chokepoint.
    let mut unmarked_terminals: Vec<(usize, usize)> = Vec::new();
    for &node_id in &terminal_node_ids {
        for h in 0..nh {
            let cell = markers[node_id * nh + h];
            if cell == 0 {
                unmarked_terminals.push((node_id, h));
            }
        }
    }

    assert!(
        unmarked_terminals.is_empty(),
        "CHOKEPOINT BYPASS DETECTED: {} terminal-node-hand cells are unmarked \
         (first few: {:?}). A payoff path bypassed multiway_brute_force_showdown \
         and did not apply rake. This is the failure mode the instrumentation \
         is designed to catch. Investigate which kernel code path wrote out[h] \
         at these terminals without going through the chokepoint helper.",
        unmarked_terminals.len(),
        &unmarked_terminals[..unmarked_terminals.len().min(5)],
    );

    // ── Standing question check 3: non-terminal cells stay zero ──
    // Sanity: the marker should ONLY be written at terminal nodes. Any
    // non-terminal node with marker != 0 means the kernel is writing
    // markers in the wrong place.
    let mut wrongly_marked_nonterminals: Vec<(usize, usize)> = Vec::new();
    for (i, node) in tree.nodes.iter().enumerate() {
        if node.node_type != NODE_TYPE_TERMINAL {
            for h in 0..nh {
                let cell = markers[i * nh + h];
                if cell != 0 {
                    wrongly_marked_nonterminals.push((i, h));
                }
            }
        }
    }
    assert!(
        wrongly_marked_nonterminals.is_empty(),
        "Marker written at non-terminal nodes ({} cells). First few: {:?}. \
         Kernel bug — marker writes should only fire at terminals.",
        wrongly_marked_nonterminals.len(),
        &wrongly_marked_nonterminals[..wrongly_marked_nonterminals.len().min(5)],
    );

    // ── Marker-state distribution ──
    let count_1 = markers.iter().filter(|&&v| v == 1).count();
    let count_2 = markers.iter().filter(|&&v| v == 2).count();
    let count_0 = markers.iter().filter(|&&v| v == 0).count();
    let count_other = markers.iter().filter(|&&v| v > 2).count();
    eprintln!("  Marker states: 1(rake-applied)={}, 2(rake-skipped)={}, 0(unmarked)={}, other(bug)={}",
        count_1, count_2, count_0, count_other);

    assert_eq!(count_other, 0,
        "Marker has unexpected values (not 0/1/2). Kernel wrote garbage.");

    // For current flop-onward trees: all terminals have board_state >= Flop
    // (i.e., flop_seen=true), so EVERY marked cell should be 1, NONE should
    // be 2. This will change when preflop integration introduces preflop-
    // ending terminals (which would correctly get marker=2).
    assert_eq!(count_2, 0,
        "Expected 0 rake-skipped markers (current trees are flop-onward, \
         so flop_seen is always true). Got {}. This is the dormant \
         no-flop-no-drop path firing — confirm preflop terminals were \
         intentionally introduced, otherwise it's a bug.",
        count_2);
    assert_eq!(count_1, expected_marked_cells,
        "Expected all {} marked cells to be 1 (rake-applied). Got {}.",
        expected_marked_cells, count_1);

    eprintln!("✓ Chokepoint instrumentation: all {} terminal-node-hand cells \
        marked, no bypass, count matches expected.", expected_marked_cells);
}
