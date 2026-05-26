#[cfg(feature = "cuda")]
mod tests {
    use solver_core::gpu::{GpuContext, GpuGameType};
    use solver_core::tree::action::BoardState;
    use solver_core::tree::flat::{FlatNode, FlatTree};

    const NUM_HANDS: usize = 3;

    fn build_kuhn_tree() -> FlatTree {
        let mut tree = FlatTree::new(2, 2, vec![0, 0], 0.0, 0.0);

        let n0 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
        tree.set_contribution(n0, 0, 1);
        tree.set_contribution(n0, 1, 1);

        let n1 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
        tree.set_contribution(n1, 0, 1);
        tree.set_contribution(n1, 1, 1);

        let n2 = tree.alloc_node(FlatNode::player(1, BoardState::River, 0));
        tree.set_contribution(n2, 0, 2);
        tree.set_contribution(n2, 1, 1);

        let n3 = tree.alloc_node(FlatNode::terminal());
        tree.set_contribution(n3, 0, 1);
        tree.set_contribution(n3, 1, 1);

        let n4 = tree.alloc_node(FlatNode::player(0, BoardState::River, 0));
        tree.set_contribution(n4, 0, 1);
        tree.set_contribution(n4, 1, 2);

        let n5 = tree.alloc_node(FlatNode::terminal());
        tree.set_contribution(n5, 0, 2);
        tree.set_contribution(n5, 1, 1);

        let n6 = tree.alloc_node(FlatNode::terminal());
        tree.set_contribution(n6, 0, 2);
        tree.set_contribution(n6, 1, 2);

        let n7 = tree.alloc_node(FlatNode::terminal());
        tree.set_contribution(n7, 0, 1);
        tree.set_contribution(n7, 1, 2);

        let n8 = tree.alloc_node(FlatNode::terminal());
        tree.set_contribution(n8, 0, 2);
        tree.set_contribution(n8, 1, 2);

        tree.set_children(n0, vec![1, 2]);
        tree.set_children(n1, vec![3, 4]);
        tree.set_children(n2, vec![5, 6]);
        tree.set_children(n4, vec![7, 8]);

        tree.set_folded_mask(n5, 0b10);
        tree.set_folded_mask(n7, 0b01);

        assert_eq!(tree.num_nodes(), 9);
        tree
    }

    fn kuhn_sign_table() -> Vec<f32> {
        let mut table = vec![0.0f32; NUM_HANDS * NUM_HANDS];
        for h in 0..NUM_HANDS {
            for ho in 0..NUM_HANDS {
                if h != ho {
                    table[h * NUM_HANDS + ho] = if h > ho { 1.0 } else { -1.0 };
                }
            }
        }
        table
    }

    #[test]
    fn gpu_kuhn_convergence() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");
        let tree = build_kuhn_tree();
        let sign_table = kuhn_sign_table();
        let mut solver = ctx
            .create_solver(&tree, vec![NUM_HANDS, NUM_HANDS], &sign_table, GpuGameType::Kuhn)
            .expect("Failed to create GPU solver");

        solver.run(256, 100).expect("GPU solver run failed");

        let regrets = solver.download_regrets().expect("download failed");
        let cum = solver.download_cum_strategy().expect("download failed");

        println!("Raw regrets after 100 iters (24 entries):");
        for (i, r) in regrets.iter().enumerate() {
            println!("  [{:2}] {:.4}", i, r);
        }
        println!("Raw cum_strategy after 100 iters:");
        for (i, c) in cum.iter().enumerate() {
            println!("  [{:2}] {:.4}", i, c);
        }

        solver.run(256, 9900).expect("GPU solver run failed");

        let avg = solver
            .get_average_strategy_at(0, 2, NUM_HANDS)
            .expect("download failed");
        let cur = solver
            .get_current_strategy_at(0, 2, NUM_HANDS)
            .expect("download failed");

        let bet_j = avg[1][0];
        let bet_q = avg[1][1];
        let bet_k = avg[1][2];
        let cur_bet_j = cur[1][0];
        let cur_bet_q = cur[1][1];
        let cur_bet_k = cur[1][2];

        println!("GPU Kuhn P0 bet prob (avg): J={:.3} Q={:.3} K={:.3}", bet_j, bet_q, bet_k);
        println!("GPU Kuhn P0 bet prob (cur): J={:.3} Q={:.3} K={:.3}", cur_bet_j, cur_bet_q, cur_bet_k);

        assert!(bet_k > 0.40, "K should mostly bet, got {}", bet_k);
        assert!(bet_q < 0.25, "Q should rarely bet, got {}", bet_q);
        assert!(
            bet_k > bet_j,
            "K should bet more than J (value > bluff), got K={} J={}",
            bet_k,
            bet_j
        );
    }

    #[test]
    fn gpu_kuhn_p1_strategy() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");
        let tree = build_kuhn_tree();
        let sign_table = kuhn_sign_table();
        let mut solver = ctx
            .create_solver(&tree, vec![NUM_HANDS, NUM_HANDS], &sign_table, GpuGameType::Kuhn)
            .expect("Failed to create GPU solver");

        solver.run(256, 10000).expect("GPU solver run failed");

        let node2_strat = solver
            .get_average_strategy_at(2, 2, NUM_HANDS)
            .expect("download failed");
        let fold_j = node2_strat[0][0];
        let call_q = node2_strat[1][1];
        let call_k = node2_strat[1][2];

        println!("GPU Kuhn P1 after P0 bet:");
        println!("  J fold: {:.3}", fold_j);
        println!("  Q call: {:.3}", call_q);
        println!("  K call: {:.3}", call_k);

        assert!(fold_j > 0.80, "P1 J should fold after bet, got fold={}", fold_j);
        assert!(call_k > 0.80, "P1 K should call after bet, got call={}", call_k);
    }
}
