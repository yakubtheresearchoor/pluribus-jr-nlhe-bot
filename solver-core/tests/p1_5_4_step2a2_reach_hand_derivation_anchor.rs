// Step 2.A.2 trace anchor: confirm "reach matches" against HAND-DERIVED
// ground truth at specific nodes — not just CPU-equals-GPU.
//
// PRECEDING CONTEXT (task #74):
//   - The reach-vs-CFV discriminator first reported "reach diverges at flop"
//     → "CPU is broken." That was a test setup bug (missing cpu.compute_all_
//     strategies); the discriminator was wrong.
//   - After fix, the same discriminator reports "reach matches at all zones."
//   - But it's the SAME TOOL that was wrong once. "I fixed it and the symptom
//     went away" has the same shape as "I loosened it until it passed."
//     Symptom-quiet ≠ correct.
//
// This test anchors the reach-matches claim against the CFR formula directly
// at specific (node, player, hand) cells. If CPU and GPU both match the
// hand-derived values, the discriminator's tool-fix was correct. If they
// agree with each other but BOTH disagree with hand-derivation, the
// discriminator is missing the bug (inter-implementation agreement trap).
//
// HAND DERIVATION (CFR formula, iter 0 strategy uniform 0.5):
//
//   Tree: Root (node 0, owner=P0, 2 children=[1,2]).
//         Node 1 (owner=P1, 2 children=[3,4]).
//
//   Initial weights: P0 = [1.0, 1.0, 1.0, 1.0]; P1 = [0.5, 0.5, 1.0, 1.0].
//
//   Propagation rule: at a node owned by player p, reach[p, child] =
//   reach[p, parent] × sigma[a]; reach[other_p, child] = reach[other_p, parent].
//
//   reach[node 0, P0, *] = [1.0, 1.0, 1.0, 1.0]   ← initial weights
//   reach[node 0, P1, *] = [0.5, 0.5, 1.0, 1.0]   ← initial weights
//   reach[node 1, P0, *] = [0.5, 0.5, 0.5, 0.5]   ← P0 acts at root, ×0.5
//   reach[node 1, P1, *] = [0.5, 0.5, 1.0, 1.0]   ← P1 copy-through
//   reach[node 3, P0, *] = [0.5, 0.5, 0.5, 0.5]   ← P0 copy-through at node 1
//   reach[node 3, P1, *] = [0.25, 0.25, 0.5, 0.5] ← P1 acts at node 1, ×0.5

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_minimal_asymmetry_game() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["Ah", "Kd", "7c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 2u8;
    let k = 4usize;

    use solver_core::hand::eval::Hand;
    let mut all_with_strength: Vec<(u16, u16)> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        all_with_strength.push((h.evaluate_internal() as u16, idx as u16));
    }
    all_with_strength.sort_by_key(|&(s, _)| s);
    let step = all_with_strength.len() / k;
    let chosen: Vec<u16> = (0..k).map(|i| all_with_strength[i * step].1).collect();

    let mut ranges: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for (rank_idx, &hi) in chosen.iter().enumerate() {
        let strength_frac = rank_idx as f32 / k as f32;
        let p0_weight = 1.0_f32;
        let p1_weight = if strength_frac >= 0.5 { 1.0_f32 } else { 0.5_f32 };
        let (c1, c2) = index_to_card_pair(hi as usize);
        let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
        let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
        ranges[0][pair_idx] = p0_weight;
        ranges[1][pair_idx] = p1_weight;
    }
    let turn_cards: Vec<u8> = vec![
        card_from_str("Td").unwrap() as u8,
        card_from_str("3s").unwrap() as u8,
    ];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("4h").unwrap() as u8,
        card_from_str("Qc").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("2s").unwrap() as u8,
        card_from_str("Js").unwrap() as u8,
    ];
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, num_players, &chosen, &turn_cards, &river_decks,
    );
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 6,
        starting_stacks: vec![50, 50], initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = FlopStartGame::new(table);
    (tree, game)
}

/// Read reach[node, player, hand] from a flat reach buffer.
fn reach_at(buf: &[f32], node: usize, np: usize, nh: usize, p: usize, h: usize) -> f32 {
    buf[node * np * nh + p * nh + h]
}

/// Assert that both CPU and GPU agree with the hand-derived expected value
/// at a specific cell. Returns true if all three agree at f32 floor.
fn check_anchor(
    label: &str,
    cpu_buf: &[f32], gpu_buf: &[f32],
    node: usize, p: usize, h: usize,
    expected: f32,
    np: usize, nh: usize,
) -> bool {
    let cpu_v = reach_at(cpu_buf, node, np, nh, p, h);
    let gpu_v = reach_at(gpu_buf, node, np, nh, p, h);
    let cpu_ok = (cpu_v - expected).abs() < 1e-6;
    let gpu_ok = (gpu_v - expected).abs() < 1e-6;
    let status = if cpu_ok && gpu_ok { "✓" } else { "✗" };
    eprintln!("  {} {} [node {}, P{}, h{}]: expected={}, CPU={}, GPU={}",
        status, label, node, p, h, expected, cpu_v, gpu_v);
    cpu_ok && gpu_ok
}

/// ANCHOR: confirm reach matches hand-derived ground truth at multiple nodes.
/// This is the verification the discriminator OWES before "reach matches"
/// can be trusted as the basis for the CFV trace.
#[test]
#[ignore = "2.A.2 trace: hand-derivation anchor for reach (CPU+GPU vs ground truth)"]
fn reach_anchored_against_hand_derivation_under_asymmetric_weights() {
    let (tree, game) = build_minimal_asymmetry_game();
    let ctx = MetalContext::new().expect("Metal");
    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    // Populate strategies on both sides (uniform 0.5 from zero regrets).
    cpu.compute_all_strategies(&tree);
    gpu.compute_all_strategies(&ctx);

    // Compute reach on both sides.
    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);
    gpu.compute_reach_flop(&ctx);
    let gpu_flop_reach = gpu.download_reach();

    let np = 2usize;
    let nh = 4usize;

    eprintln!("\n=== REACH HAND-DERIVATION ANCHOR ===");
    eprintln!("Initial weights: P0 = [1.0, 1.0, 1.0, 1.0], P1 = [0.5, 0.5, 1.0, 1.0]");
    eprintln!("Iter 0 strategy = uniform 0.5 (zero regrets, 2 actions at each player node)");
    eprintln!();
    eprintln!("Anchored checks (legend: ✓ = both CPU+GPU match hand-derived ground truth):");

    let mut all_pass = true;

    // ─── Node 0 (root) ───
    eprintln!("\nNode 0 (root, P0 owner):");
    all_pass &= check_anchor("initial-P0", &cpu_flop_reach, &gpu_flop_reach, 0, 0, 0, 1.0, np, nh);
    all_pass &= check_anchor("initial-P0", &cpu_flop_reach, &gpu_flop_reach, 0, 0, 2, 1.0, np, nh);
    all_pass &= check_anchor("initial-P1-h0", &cpu_flop_reach, &gpu_flop_reach, 0, 1, 0, 0.5, np, nh);
    all_pass &= check_anchor("initial-P1-h2", &cpu_flop_reach, &gpu_flop_reach, 0, 1, 2, 1.0, np, nh);

    // ─── Node 1 (root child, P1 owner) ───
    eprintln!("\nNode 1 (root child, P1 owner):");
    // P0 acted at root with strategy 0.5 → P0 reach × 0.5 for ALL hands.
    all_pass &= check_anchor("after-root-P0", &cpu_flop_reach, &gpu_flop_reach, 1, 0, 0, 0.5, np, nh);
    all_pass &= check_anchor("after-root-P0", &cpu_flop_reach, &gpu_flop_reach, 1, 0, 3, 0.5, np, nh);
    // P1 unchanged from root (P1 doesn't act at root).
    all_pass &= check_anchor("copy-P1-h0", &cpu_flop_reach, &gpu_flop_reach, 1, 1, 0, 0.5, np, nh);
    all_pass &= check_anchor("copy-P1-h2", &cpu_flop_reach, &gpu_flop_reach, 1, 1, 2, 1.0, np, nh);

    // ─── Node 3 (grandchild via node 1, depth 2) ───
    let n3_owner = tree.nodes[3].player_id;
    let n3_type = tree.nodes[3].node_type;
    let n1_children = tree.node_children(1).to_vec();
    eprintln!("\nNode 3 (grandchild via node 1): owner=P{}, node_type={}, n1 children={:?}",
        n3_owner, n3_type, n1_children);

    // Only assert hand-derivation for grandchildren if they're player/chance
    // nodes (terminal nodes have their reach propagation behavior we need to
    // verify too, but the CFR formula at terminals is also copy-through).
    // P0 was copy-through at node 1 (since P1 acts there), so reach[node 3, P0] = reach[node 1, P0].
    all_pass &= check_anchor("copy-P0-h0", &cpu_flop_reach, &gpu_flop_reach, 3, 0, 0, 0.5, np, nh);
    // P1 acted at node 1 with strategy 0.5 → reach[node 3, P1] = reach[node 1, P1] × 0.5.
    //   P1, h0: 0.5 × 0.5 = 0.25
    //   P1, h2: 1.0 × 0.5 = 0.5
    all_pass &= check_anchor("after-n1-P1-h0", &cpu_flop_reach, &gpu_flop_reach, 3, 1, 0, 0.25, np, nh);
    all_pass &= check_anchor("after-n1-P1-h2", &cpu_flop_reach, &gpu_flop_reach, 3, 1, 2, 0.5, np, nh);

    eprintln!("\n=== VERDICT ===");
    if all_pass {
        eprintln!("  ✓ Both CPU and GPU MATCH hand-derived ground truth at every checked cell.");
        eprintln!("  Reach computation is now anchored, not just agreement-confirmed.");
        eprintln!("  The discriminator's 'reach matches' verdict is TRUSTED.");
        eprintln!("  Next: extend the discriminator to CFV per zone (Hypothesis B trace).");
    } else {
        eprintln!("  ✗ At least one (node, player, hand) cell disagrees with hand-derivation.");
        eprintln!("  Inter-implementation agreement was the trap — both sides agree but the");
        eprintln!("  agreement is on a wrong answer. The reach computation IS broken (in");
        eprintln!("  one or both implementations); the discriminator missed it because");
        eprintln!("  CPU=GPU at the wrong value.");
    }
    assert!(all_pass, "Reach failed to match hand-derived ground truth");
}
