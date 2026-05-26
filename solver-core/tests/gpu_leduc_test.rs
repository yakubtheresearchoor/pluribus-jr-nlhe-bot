#[cfg(feature = "cuda")]
mod tests {
    use solver_core::gpu::{GpuContext, GpuGameType};
    use solver_core::tree::action::BoardState;
    use solver_core::tree::flat::{FlatNode, FlatTree};

    const NUM_HANDS: usize = 6;

    fn hand_rank(hand: usize) -> usize {
        hand / 2
    }

    fn leduc_compare(h1: usize, h2: usize, board_rank: usize) -> f32 {
        let r1 = hand_rank(h1);
        let r2 = hand_rank(h2);
        if r1 == board_rank && r2 != board_rank {
            return 1.0;
        }
        if r2 == board_rank && r1 != board_rank {
            return -1.0;
        }
        if r1 == r2 {
            return 0.0;
        }
        if r1 > r2 {
            1.0
        } else {
            -1.0
        }
    }

    fn expected_sign(h: usize, h_o: usize) -> f32 {
        let mut total = 0.0f32;
        let mut count = 0;
        for b in 0..NUM_HANDS {
            if b == h || b == h_o {
                continue;
            }
            total += leduc_compare(h, h_o, hand_rank(b));
            count += 1;
        }
        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }

    fn leduc_sign_table() -> Vec<f32> {
        let mut table = vec![0.0f32; NUM_HANDS * NUM_HANDS];
        for h in 0..NUM_HANDS {
            for ho in 0..NUM_HANDS {
                table[h * NUM_HANDS + ho] = expected_sign(h, ho);
            }
        }
        table
    }

    fn alloc(tree: &mut FlatTree, node: FlatNode, c0: i32, c1: i32) -> usize {
        let idx = tree.alloc_node(node);
        tree.set_contribution(idx, 0, c0);
        tree.set_contribution(idx, 1, c1);
        idx
    }

    fn build_round2(tree: &mut FlatTree, c0: i32, c1: i32) -> usize {
        let bet = 4i32;
        let raise_total = 8i32;

        let p0 = alloc(tree, FlatNode::player(0, BoardState::River, 0), c0, c1);
        let p1_after_check = alloc(tree, FlatNode::player(1, BoardState::River, 0), c0, c1);
        let showdown_cc = alloc(tree, FlatNode::terminal(), c0, c1);

        let p0_after_cb = alloc(tree, FlatNode::player(0, BoardState::River, 0), c0, c1 + bet);
        let fold_cb = alloc(tree, FlatNode::terminal(), c0, c1 + bet);
        let call_cb = alloc(tree, FlatNode::terminal(), c1 + bet, c1 + bet);
        let p1_after_cr = alloc(tree, FlatNode::player(1, BoardState::River, 0), c0 + raise_total, c1 + bet);
        let fold_cr = alloc(tree, FlatNode::terminal(), c0 + raise_total, c1 + bet);
        let call_cr = alloc(tree, FlatNode::terminal(), c0 + raise_total, c0 + raise_total);

        tree.set_children(p1_after_check, vec![showdown_cc as u32, p0_after_cb as u32]);
        tree.set_children(p0_after_cb, vec![fold_cb as u32, call_cb as u32, p1_after_cr as u32]);
        tree.set_children(p1_after_cr, vec![fold_cr as u32, call_cr as u32]);

        tree.set_folded_mask(fold_cb, 0b01);
        tree.set_folded_mask(fold_cr, 0b10);

        let p1_after_bet = alloc(tree, FlatNode::player(1, BoardState::River, 0), c0 + bet, c1);
        let fold_b = alloc(tree, FlatNode::terminal(), c0 + bet, c1);
        let call_b = alloc(tree, FlatNode::terminal(), c0 + bet, c0 + bet);
        let p0_after_br = alloc(tree, FlatNode::player(0, BoardState::River, 0), c0 + bet, c1 + raise_total);
        let fold_br = alloc(tree, FlatNode::terminal(), c0 + bet, c1 + raise_total);
        let call_br = alloc(tree, FlatNode::terminal(), c1 + raise_total, c1 + raise_total);

        tree.set_children(p0, vec![p1_after_check as u32, p1_after_bet as u32]);
        tree.set_children(p1_after_bet, vec![fold_b as u32, call_b as u32, p0_after_br as u32]);
        tree.set_children(p0_after_br, vec![fold_br as u32, call_br as u32]);

        tree.set_folded_mask(fold_b, 0b10);
        tree.set_folded_mask(fold_br, 0b01);

        p0
    }

    fn build_leduc_tree() -> FlatTree {
        let mut tree = FlatTree::new(2, 2, vec![0, 0], 0.0, 0.0);

        let bet = 2i32;
        let raise_total = 4i32;

        let n0 = alloc(&mut tree, FlatNode::player(0, BoardState::Flop, 0), 1, 1);
        let n1 = alloc(&mut tree, FlatNode::player(1, BoardState::Flop, 0), 1, 1);
        let r2_cc = build_round2(&mut tree, 1, 1);

        let n3 = alloc(&mut tree, FlatNode::player(0, BoardState::Flop, 0), 1, 1 + bet);
        let t_fold_cb = alloc(&mut tree, FlatNode::terminal(), 1, 1 + bet);
        let r2_call_cb = build_round2(&mut tree, 1 + bet, 1 + bet);

        let n6 = alloc(&mut tree, FlatNode::player(1, BoardState::Flop, 0), 1 + raise_total, 1 + bet);
        let t_fold_cr = alloc(&mut tree, FlatNode::terminal(), 1 + raise_total, 1 + bet);
        let r2_call_cr = build_round2(&mut tree, 1 + raise_total, 1 + raise_total);

        let n9 = alloc(&mut tree, FlatNode::player(1, BoardState::Flop, 0), 1 + bet, 1);
        let t_fold_b = alloc(&mut tree, FlatNode::terminal(), 1 + bet, 1);
        let r2_call_b = build_round2(&mut tree, 1 + bet, 1 + bet);

        let n12 = alloc(&mut tree, FlatNode::player(0, BoardState::Flop, 0), 1 + bet, 1 + raise_total);
        let t_fold_br = alloc(&mut tree, FlatNode::terminal(), 1 + bet, 1 + raise_total);
        let r2_call_br = build_round2(&mut tree, 1 + raise_total, 1 + raise_total);

        tree.set_children(n0, vec![n1 as u32, n9 as u32]);
        tree.set_children(n1, vec![r2_cc as u32, n3 as u32]);
        tree.set_children(n3, vec![t_fold_cb as u32, r2_call_cb as u32, n6 as u32]);
        tree.set_children(n6, vec![t_fold_cr as u32, r2_call_cr as u32]);
        tree.set_children(n9, vec![t_fold_b as u32, r2_call_b as u32, n12 as u32]);
        tree.set_children(n12, vec![t_fold_br as u32, r2_call_br as u32]);

        tree.set_folded_mask(t_fold_cb, 0b01);
        tree.set_folded_mask(t_fold_cr, 0b10);
        tree.set_folded_mask(t_fold_b, 0b10);
        tree.set_folded_mask(t_fold_br, 0b01);

        tree
    }

    #[test]
    fn gpu_leduc_runs() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");
        let tree = build_leduc_tree();
        let sign_table = leduc_sign_table();
        let mut solver = ctx
            .create_solver(&tree, vec![NUM_HANDS, NUM_HANDS], &sign_table, GpuGameType::Leduc)
            .expect("Failed to create GPU solver");

        solver.run(256, 10000).expect("GPU solver run failed");

        let avg = solver
            .get_average_strategy_at(0, 2, NUM_HANDS)
            .expect("download failed");

        let check = avg[0].iter().sum::<f32>() / NUM_HANDS as f32;
        let bet = avg[1].iter().sum::<f32>() / NUM_HANDS as f32;

        println!("GPU Leduc P0 root: avg check={:.3} bet={:.3}", check, bet);

        assert!(
            (check + bet - 1.0).abs() < 0.01,
            "probabilities should sum to ~1, got check={} bet={}",
            check,
            bet
        );
        assert!(check > 0.5, "P0 should mostly check, got {}", check);
        assert!(bet > 0.01, "P0 should sometimes bet, got {}", bet);
    }
}
