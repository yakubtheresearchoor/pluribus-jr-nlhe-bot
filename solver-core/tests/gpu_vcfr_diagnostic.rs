#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::GpuContext;
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::tree::flat::{FlatNode, FlatTree, MAX_NA};
use solver_core::tree::action::BoardState;

fn uniform_range() -> Vec<f32> {
    vec![1.0; NUM_POSSIBLE_HANDS]
}

fn make_board() -> Vec<Card> {
    ["2h", "7d", "Ks", "4c", "9s"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect()
}

fn build_river_tree() -> FlatTree {
    let mut tree = FlatTree::new(2, 10, vec![95, 95], 0.0, 0.0);

    let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n0, 0, 5);
    tree.set_contribution(n0, 1, 5);

    let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n1, 0, 5);
    tree.set_contribution(n1, 1, 5);

    let n2 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
    tree.set_contribution(n2, 0, 10);
    tree.set_contribution(n2, 1, 5);

    let n3 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n3, 0, 5);
    tree.set_contribution(n3, 1, 5);

    let n4 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
    tree.set_contribution(n4, 0, 5);
    tree.set_contribution(n4, 1, 10);

    let n5 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n5, 0, 10);
    tree.set_contribution(n5, 1, 5);

    let n6 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n6, 0, 10);
    tree.set_contribution(n6, 1, 10);

    let n7 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n7, 0, 5);
    tree.set_contribution(n7, 1, 10);

    let n8 = tree.alloc_node(FlatNode::terminal());
    tree.set_contribution(n8, 0, 10);
    tree.set_contribution(n8, 1, 10);

    tree.set_children(n0, vec![1, 2]);
    tree.set_children(n1, vec![3, 4]);
    tree.set_children(n2, vec![5, 6]);
    tree.set_children(n4, vec![7, 8]);

    tree.set_folded_mask(n5, 0b10);
    tree.set_folded_mask(n7, 0b01);

    tree.compute_levels();
    tree
}

#[test]
fn diagnose_first_divergence_point() {
    let board = make_board();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_valid_hands();
    let np = 2usize;
    let tree = build_river_tree();

    let (opp_str, opp_idx, pl_str, pl_idx, _) = game.sorted_opp_arrays();
    let hand_cards = game.hand_cards_gpu();
    let initial_weight = game.initial_weight_flat(&ranges);

    println!("nh={}", nh);
    println!("num_nodes={}", tree.num_nodes());
    println!("num_infosets={}", tree.num_infosets);
    println!("max_depth={}", tree.max_depth);

    // Print tree structure
    for i in 0..tree.num_nodes() {
        let node = &tree.nodes[i];
        let c0 = tree.get_contribution(i, 0);
        let c1 = tree.get_contribution(i, 1);
        let fm = tree.get_folded_mask(i);
        let is = tree.infoset_offsets[i];
        match node.node_type {
            0 => println!("  node {}: TERMINAL contrib=[{},{}] fold={:02b}", i, c0, c1, fm),
            2 => println!("  node {}: PLAYER{} children={:?} contrib=[{},{}] infoset={}",
                i, node.player_id, tree.node_children(i), c0, c1, is),
            _ => println!("  node {}: OTHER type={}", i, node.node_type),
        }
    }

    // Compute CPU regrets for iteration 1, sequential mode
    // Sequential: compute_strategies → reach → traverser_0_bottom_up → compute_strategies → reach → traverser_1_bottom_up
    let mut cpu = VectorCfr::new(&tree, vec![nh, nh]);
    cpu.run_sequential(&tree, &game, 1);
    let cpu_regrets = cpu.regrets_slice().to_vec();

    // GPU: same thing
    let gpu = GpuContext::new().expect("GPU init failed");
    let mut gpu_solver = gpu
        .create_vcfr_solver(
            &tree, nh, &opp_str, &opp_idx, &pl_str, &pl_idx, &hand_cards, &initial_weight,
        )
        .expect("vcfr solver creation failed");
    gpu_solver.run(1).expect("GPU run failed");
    let gpu_regrets = gpu_solver.download_regrets().expect("download failed");

    let data_per_infoset = MAX_NA * nh;
    let num_infosets = tree.num_infosets as usize;

    // Print infoset-level summary
    println!("\n=== Infoset-level regret comparison ===");
    for infoset in 0..num_infosets {
        let node_id = tree.decision_node_ids[infoset];
        let na = tree.nodes[node_id as usize].num_children as usize;
        let owner = tree.nodes[node_id as usize].player_id;

        let mut max_diff = 0.0f32;
        let mut max_diff_h = 0;
        let mut max_diff_a = 0;
        let mut nonzero_cpu = 0;
        let mut nonzero_gpu = 0;

        for a in 0..na {
            for h in 0..nh {
                let idx = infoset * data_per_infoset + a * nh + h;
                let diff = (cpu_regrets[idx] - gpu_regrets[idx]).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_h = h;
                    max_diff_a = a;
                }
                if cpu_regrets[idx].abs() > 1e-6 { nonzero_cpu += 1; }
                if gpu_regrets[idx].abs() > 1e-6 { nonzero_gpu += 1; }
            }
        }

        let cpu_first5: Vec<f32> = (0..5).map(|a| cpu_regrets[infoset * data_per_infoset + a * nh]).collect();
        let gpu_first5: Vec<f32> = (0..5).map(|a| gpu_regrets[infoset * data_per_infoset + a * nh]).collect();

        println!(
            "  infoset={} (node={} P{} na={}): max_diff={:.6} at a={} h={} | cpu_5={:?} gpu_5={:?} | nonzero: cpu={} gpu={}",
            infoset, node_id, owner, na, max_diff, max_diff_a, max_diff_h,
            cpu_first5, gpu_first5, nonzero_cpu, nonzero_gpu
        );

        if max_diff > 1e-3 && max_diff_h < nh {
            let idx = infoset * data_per_infoset + max_diff_a * nh + max_diff_h;
            println!("    >>> DIVERGENT: cpu={:.8} gpu={:.8}", cpu_regrets[idx], gpu_regrets[idx]);
        }
    }

    // Now check: for P0's infosets (0 and 3), after traverser 0,
    // the regrets should only depend on terminal evaluations at nodes reachable from those infosets.
    // Infoset 0 (node 0, P0): children are nodes 1 and 2
    //   - traverser=0: cfv[check] = CFV at node 1, cfv[bet] = CFV at node 2
    //   - At opponent nodes (1,2): SUM child CFVs
    //   - Node 1 children: 3 (showdown c_t=5), 4 (P0 node)
    //   - Node 2 children: 5 (fold), 6 (showdown c_t=10)

    // Let me compute traverser=0 CFV at each internal node using the CPU evaluate_terminal
    // With uniform strategy (0.5 for each action), initial reach = 1.0

    // For traverser=0:
    // Node 3 (terminal showdown, c_t=5): cfv = evaluate_terminal(0, 3, reach=[0.5, 0.25])
    //   opp is P1, reach_P1 = 0.25

    // But wait: the vector CFR reach at a terminal node for a player p is:
    //   reach_p = initial_weight_p * product(sigma_p[a] at each P node on the path for action taken)
    // For uniform strategy (0.5) and path P0→check→P1→check→node3:
    //   reach_P0 = 1.0 * 0.5 = 0.5
    //   reach_P1 = 1.0 * 0.5 = 0.5

    // Wait, the tree has contributions [95, 95] at root. Let me re-check...
    // No: FlatTree::new(2, 10, vec![95, 95], 0.0, 0.0) creates with initial contributions [95, 95]
    // But set_contribution overrides those. So root has contrib [5, 5] as set.

    // Actually the initial stack sizes don't matter for CFV computation.
    // What matters is the contributions at each node and the strategy.

    println!("\n=== Manual traverser=0 CFV computation ===");

    // With uniform strategy:
    // Path to node 3: P0 checks (prob 0.5) → node 1, P1 checks (prob 0.5) → node 3
    // reach at node 3: P0 = 0.5, P1 = 0.5

    let cfreach_n3: Vec<Vec<f32>> = vec![vec![0.5; nh], vec![0.5; nh]];
    let cfv_n3 = game.evaluate_terminal(0, 3, &tree, &cfreach_n3);

    // Path to node 5: P0 bets (prob 0.5) → node 2, P1 folds (prob 0.5) → node 5
    // reach at node 5: P0 = 0.5, P1 = 0.5
    let cfreach_n5: Vec<Vec<f32>> = vec![vec![0.5; nh], vec![0.5; nh]];
    let cfv_n5 = game.evaluate_terminal(0, 5, &tree, &cfreach_n5);

    // Path to node 6: P0 bets (prob 0.5) → node 2, P1 calls (prob 0.5) → node 6
    // reach at node 6: P0 = 0.5, P1 = 0.5
    let cfreach_n6: Vec<Vec<f32>> = vec![vec![0.5; nh], vec![0.5; nh]];
    let cfv_n6 = game.evaluate_terminal(0, 6, &tree, &cfreach_n6);

    // Path to node 7: P0 checks → node 1, P1 bets → node 4, P0 folds → node 7
    // reach at node 7: P0 = 0.5*0.5 = 0.25, P1 = 0.5
    let cfreach_n7: Vec<Vec<f32>> = vec![vec![0.25; nh], vec![0.5; nh]];
    let cfv_n7 = game.evaluate_terminal(0, 7, &tree, &cfreach_n7);

    // Path to node 8: P0 checks → node 1, P1 bets → node 4, P0 calls → node 8
    // reach at node 8: P0 = 0.5*0.5 = 0.25, P1 = 0.5
    let cfreach_n8: Vec<Vec<f32>> = vec![vec![0.25; nh], vec![0.5; nh]];
    let cfv_n8 = game.evaluate_terminal(0, 8, &tree, &cfreach_n8);

    // CFV at node 1 (P1 node, opponent): SUM of child CFVs
    let mut cfv_n1: Vec<f32> = vec![0.0; nh];
    for h in 0..nh { cfv_n1[h] = cfv_n3[h] + cfv_n4(&tree, &game, nh, 0.25, 0.5, &cfv_n7, &cfv_n8); }

    // Actually wait, node 4 is a P0 node. For traverser=0 at opponent node 1,
    // the CFV is SUM of child CFVs (not weighted by strategy).
    // But node 4 is P0 (traverser!), so at node 4 we weight by strategy.

    // Let me redo this properly:
    // Node 1 (P1, opponent to traverser=0): cfv = cfv_n3 + cfv_at_node4
    // Node 4 (P0, traverser): cfv = 0.5*cfv_n7 + 0.5*cfv_n8

    let mut cfv_n4_weighted: Vec<f32> = vec![0.0; nh];
    for h in 0..nh {
        cfv_n4_weighted[h] = 0.5 * cfv_n7[h] + 0.5 * cfv_n8[h];
    }

    // CFV at node 1: sum of children
    let mut cfv_n1_total: Vec<f32> = vec![0.0; nh];
    for h in 0..nh {
        cfv_n1_total[h] = cfv_n3[h] + cfv_n4_weighted[h];
    }

    // CFV at node 2 (P1, opponent): cfv = cfv_n5 + cfv_n6
    let mut cfv_n2_total: Vec<f32> = vec![0.0; nh];
    for h in 0..nh { cfv_n2_total[h] = cfv_n5[h] + cfv_n6[h]; }

    // CFV at node 0 (P0, traverser): weighted by strategy
    // cfv = 0.5*cfv_n1 + 0.5*cfv_n2
    // regret[check] = cfv_n1 - cfv_avg = cfv_n1 - 0.5*(cfv_n1 + cfv_n2) = 0.5*(cfv_n1 - cfv_n2)
    // regret[bet] = cfv_n2 - cfv_avg = 0.5*(cfv_n2 - cfv_n1)

    println!("Manual regret at infoset 0 (node 0, P0):");
    let mut max_manual_diff = 0.0f32;
    for h in 0..10.min(nh) {
        let regret_check = cfv_n1_total[h] - 0.5 * (cfv_n1_total[h] + cfv_n2_total[h]);
        let regret_bet = cfv_n2_total[h] - 0.5 * (cfv_n1_total[h] + cfv_n2_total[h]);

        let cpu_r_check = cpu_regrets[0 * nh + h];
        let cpu_r_bet = cpu_regrets[1 * nh + h];

        println!("  h={}: manual_check={:.4} manual_bet={:.4} | cpu_check={:.4} cpu_bet={:.4}",
            h, regret_check, regret_bet, cpu_r_check, cpu_r_bet);

        let diff_c = (regret_check - cpu_r_check).abs();
        let diff_b = (regret_bet - cpu_r_bet).abs();
        max_manual_diff = max_manual_diff.max(diff_c).max(diff_b);
    }
    println!("Max manual vs CPU diff: {:.8}", max_manual_diff);

    assert!(false, "Diagnostic — see output");
}

fn cfv_n4(tree: &FlatTree, game: &RiverPokerGame, nh: usize, _r0: f32, _r1: f32, _n7: &[f32], _n8: &[f32]) -> f32 {
    let _ = (tree, game, nh);
    0.0
}
