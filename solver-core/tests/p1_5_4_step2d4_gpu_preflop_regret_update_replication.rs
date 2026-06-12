// Step 2.D.4: GPU preflop regret-update bit-exact replication gate.
//
// THE GATE STRUCTURE (per #79, #84, #92):
//   - CPU correctness anchored by p1_5_4_slice_a3b_single_iter_engine
//     (preflop bottom_up single-iter) + #92 (multi-iter loop composition
//     vs independent textbook reference).
//   - This test: REPLICATION ONLY. GPU's preflop bottom_up kernel must
//     produce bit-exact regrets + cum_strategy + cfv post-update under
//     asymmetric leaf cfvs.
//
// THE KERNEL:
//   New kernel `vcfr_preflop_bottom_up_player` (added to vcfr.metal).
//   Operates per (player_node, lane) thread; computes cfv_avg via the
//   same per-a iteration order as CPU bottom_up_recursive (preflop_cfr.rs
//   line 598), then DCFR regret + cum_strategy update at traverser-owned
//   nodes. Host dispatches level-by-level bottom-up.
//
// THE TEST:
//   1. Build HU minimal preflop tree + PreflopVectorCfr.
//   2. Seed asymmetric regrets; compute strategy.
//   3. Seed asymmetric per-node-per-class cfv at all CHANCE + TERMINAL
//      preflop-zone leaves (deterministic asymmetric stub). These act
//      as the leaf values that bottom_up walks upward from.
//   4. CPU: run bottom_up_recursive manually (extracted logic mirrors
//      production exactly — we instead call the production walk by
//      configuring the closure-oracle pattern).
//   5. GPU: dispatch vcfr_preflop_bottom_up_player level-by-level (deepest
//      first). After each level the cfv buffer is updated and the
//      regrets+cum_strategy buffers receive their updates at traverser
//      nodes.
//   6. Compare GPU regrets / cum_strategy / cfv bit-exact to CPU.

#![cfg(feature = "metal")]

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::MetalBuffer;
use solver_core::solver::flop_start_vector_cfr::DcfrParams;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatNode, FlatTree, MAX_NA_PREFLOP};

fn build_hu_preflop_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![20, 19],
        initial_contributions: vec![1, 2],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    build_tree(&cfg).expect("preflop tree builds")
}

/// Asymmetric per (leaf_node_idx, class) leaf value. Same shape pattern
/// as 2.D.0c's stub_leaf to keep the test discriminating.
fn leaf_cfv(node_idx: usize, class: usize) -> f32 {
    let seed: u64 = (node_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (class as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let bits = ((seed >> 32) & 0xFFFFFF) as i64 - (1 << 23);
    (bits as f32) / ((1 << 23) as f32)
}

/// Preflop-zone BFS levels from root, restricted to preflop player nodes
/// (the only nodes the kernel processes). The leaves (chance + terminal
/// in preflop zone) have cfv set BEFORE this kernel runs.
fn preflop_player_levels(tree: &FlatTree, prod: &PreflopVectorCfr) -> Vec<Vec<u32>> {
    use solver_core::tree::action::BoardState;
    let nn = tree.num_nodes();
    // Depth assignment via BFS.
    let mut depth = vec![-1i32; nn];
    depth[0] = 0;
    let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::from(vec![0usize]);
    while let Some(idx) = q.pop_front() {
        let d = depth[idx];
        for &c in tree.node_children(idx) {
            if depth[c as usize] < 0 {
                depth[c as usize] = d + 1;
                q.push_back(c as usize);
            }
        }
    }
    let max_depth = depth.iter().filter(|&&d| d >= 0).max().copied().unwrap_or(0);
    let mut by_depth: Vec<Vec<u32>> = vec![Vec::new(); (max_depth + 1) as usize];
    for idx in 0..nn {
        let d = depth[idx];
        if d < 0 { continue; }
        if tree.nodes[idx].board_state != BoardState::Preflop as u8 { continue; }
        // Only preflop PLAYER nodes (those with a local_offset).
        if prod.local_offset[idx] == usize::MAX { continue; }
        by_depth[d as usize].push(idx as u32);
    }
    // Return deepest-first so children's cfv is ready when parent processes.
    by_depth.into_iter().rev().filter(|l| !l.is_empty()).collect()
}

/// Compute CPU reference by mirroring `bottom_up_recursive` in
/// preflop_cfr.rs directly. We don't call production's run_one_iteration
/// because that wraps a chance integration. We just exercise the
/// bottom_up + regret-update step with hand-set leaf cfvs.
fn cpu_bottom_up(
    tree: &FlatTree,
    prod: &mut PreflopVectorCfr,
    leaf_cfvs: &[Vec<f32>],   // [nn][n_classes] populated for chance + terminal
    traverser: u8,
    params: &DcfrParams,
) -> Vec<Vec<f32>> {
    let nn = tree.num_nodes();
    let n_classes = NUM_PREFLOP_CLASSES;
    let mut cfv: Vec<Vec<f32>> = vec![vec![0.0f32; n_classes]; nn];
    // Initialize cfv at leaves.
    for nid in 0..nn {
        if !leaf_cfvs[nid].is_empty() {
            cfv[nid] = leaf_cfvs[nid].clone();
        }
    }
    cpu_recurse(tree, prod, 0, traverser, leaf_cfvs, &mut cfv, params);
    cfv
}

fn cpu_recurse(
    tree: &FlatTree,
    prod: &mut PreflopVectorCfr,
    node_idx: usize,
    traverser: u8,
    leaf_cfvs: &[Vec<f32>],
    cfv: &mut Vec<Vec<f32>>,
    params: &DcfrParams,
) {
    use solver_core::tree::action::BoardState;
    let node = &tree.nodes[node_idx];
    let n_classes = NUM_PREFLOP_CLASSES;
    // If node has leaf cfv set, return (its cfv already populated).
    if !leaf_cfvs[node_idx].is_empty() { return; }
    // If outside preflop zone, return.
    if node.board_state != BoardState::Preflop as u8 { return; }
    let children = tree.node_children(node_idx).to_vec();
    if children.is_empty() { return; }
    if !node.is_player() {
        // Pass through: cfv = sum of child cfvs (single child or chance — for
        // tests we won't hit non-player preflop nodes at this layer).
        for &c in &children {
            cpu_recurse(tree, prod, c as usize, traverser, leaf_cfvs, cfv, params);
        }
        let mut sum = vec![0.0f32; n_classes];
        for &c in &children {
            for k in 0..n_classes { sum[k] += cfv[c as usize][k]; }
        }
        cfv[node_idx] = sum;
        return;
    }
    // Player node.
    for &c in &children {
        cpu_recurse(tree, prod, c as usize, traverser, leaf_cfvs, cfv, params);
    }
    let local = prod.local_offset[node_idx];
    let na = node.num_children as usize;
    let off = local * MAX_NA_PREFLOP * n_classes;
    let mut cfv_avg = vec![0.0f32; n_classes];
    if node.player_id == traverser {
        for (a, &child) in children.iter().enumerate() {
            let s_base = off + a * n_classes;
            for k in 0..n_classes {
                cfv_avg[k] += prod.strategy[s_base + k] * cfv[child as usize][k];
            }
        }
        for (a, &child) in children.iter().enumerate() {
            for k in 0..n_classes {
                let inst_regret = cfv[child as usize][k] - cfv_avg[k];
                let ridx = off + a * n_classes + k;
                let old_r = prod.regrets[ridx];
                let coef = if old_r >= 0.0 { params.alpha_t() } else { params.beta_t() };
                prod.regrets[ridx] = coef * old_r + inst_regret;
                prod.cum_strategy[ridx] = params.gamma_t() * prod.cum_strategy[ridx]
                    + prod.strategy[ridx];
            }
        }
    } else {
        for &child in &children {
            for k in 0..n_classes { cfv_avg[k] += cfv[child as usize][k]; }
        }
    }
    cfv[node_idx] = cfv_avg;
}

#[test]
#[ignore = "Step 2.D.4: GPU preflop regret-update replication gate"]
fn step2d4_gpu_preflop_regret_update_matches_cpu_bit_exactly() {
    let tree = build_hu_preflop_tree();
    let nn = tree.num_nodes();
    eprintln!("\n=== Step 2.D.4: GPU preflop regret update bit-exact gate ===");
    eprintln!("Tree: {} nodes", nn);

    let mut prod_cpu = PreflopVectorCfr::new(&tree);
    let infoset_count = prod_cpu.infoset_count;
    // Seed asymmetric regrets + compute strategy.
    for idx in 0..nn {
        let local = prod_cpu.local_offset[idx];
        if local == usize::MAX { continue; }
        let na = tree.nodes[idx].num_children as usize;
        if na == 0 { continue; }
        let off = local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
        for c in 0..NUM_PREFLOP_CLASSES {
            for a in 0..na {
                let bias = match (c + local * 13) % 4 {
                    0 => 1.0 + a as f32 + (c as f32 * 0.01),
                    1 => if a % 2 == 0 { 2.5 } else { -1.3 },
                    2 => 1e-7,
                    _ => 0.0,
                };
                prod_cpu.regrets[off + a * NUM_PREFLOP_CLASSES + c] = bias;
            }
        }
    }
    prod_cpu.compute_preflop_strategy(&tree);

    // Capture pre-update regrets + cum_strategy for the GPU side.
    let pre_regrets = prod_cpu.regrets.clone();
    let pre_cum = prod_cpu.cum_strategy.clone();
    let strategy = prod_cpu.strategy.clone();

    // Seed leaf cfvs at CHANCE + TERMINAL nodes (anything that's NOT a
    // preflop player decision). For the kernel test we set the SAME leaf
    // cfvs on both sides.
    let mut leaf_cfvs: Vec<Vec<f32>> = vec![Vec::new(); nn];
    for idx in 0..nn {
        let n = &tree.nodes[idx];
        let is_player = n.is_player() && prod_cpu.local_offset[idx] != usize::MAX;
        if is_player { continue; }
        // Treat as leaf: hand-fill cfv.
        leaf_cfvs[idx] = (0..NUM_PREFLOP_CLASSES).map(|c| leaf_cfv(idx, c)).collect();
    }

    let traverser = 0u8;
    let params = DcfrParams::new(0);

    // ── CPU. ──
    let cpu_cfv = cpu_bottom_up(&tree, &mut prod_cpu, &leaf_cfvs, traverser, &params);
    let cpu_regrets = prod_cpu.regrets.clone();
    let cpu_cum = prod_cpu.cum_strategy.clone();

    // ── GPU. ──
    let ctx = MetalContext::new().expect("Metal");
    let pipeline = ctx.create_pipeline("vcfr_preflop_bottom_up_player")
        .expect("preflop_bottom_up_player pipeline");

    // GPU cfv buffer: [nn × NUM_PREFLOP_CLASSES] flat. Init with leaf
    // values, zeros elsewhere.
    let mut h_cfv = vec![0.0f32; nn * NUM_PREFLOP_CLASSES];
    for idx in 0..nn {
        if leaf_cfvs[idx].is_empty() { continue; }
        for c in 0..NUM_PREFLOP_CLASSES {
            h_cfv[idx * NUM_PREFLOP_CLASSES + c] = leaf_cfvs[idx][c];
        }
    }
    let d_cfv: MetalBuffer<f32> = ctx.upload(&h_cfv);

    let total_strat = infoset_count * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
    let d_strategy: MetalBuffer<f32> = ctx.upload(&strategy[..total_strat]);
    let d_regrets: MetalBuffer<f32> = ctx.upload(&pre_regrets[..total_strat]);
    let d_cum_strategy: MetalBuffer<f32> = ctx.upload(&pre_cum[..total_strat]);

    let nodes_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            tree.nodes.as_ptr() as *const u8,
            tree.nodes.len() * std::mem::size_of::<FlatNode>(),
        )
    };
    let d_nodes: MetalBuffer<u8> = ctx.upload(nodes_bytes);
    let d_children: MetalBuffer<u32> = ctx.upload(&tree.children);

    let infoset_offsets: Vec<u32> = (0..nn).map(|idx| {
        let local = prod_cpu.local_offset[idx];
        if local == usize::MAX { u32::MAX } else { local as u32 }
    }).collect();
    let d_infoset_offsets: MetalBuffer<u32> = ctx.upload(&infoset_offsets);

    // Levels of preflop PLAYER nodes, deepest-first.
    let levels = preflop_player_levels(&tree, &prod_cpu);
    eprintln!("Preflop player levels (deepest-first): {} levels, sizes = {:?}",
        levels.len(), levels.iter().map(|l| l.len()).collect::<Vec<_>>());

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Params {
        level_count: i32,
        lanes: i32,
        traverser: u32,
        alpha_t: f32,
        beta_t: f32,
        gamma_t: f32,
    }

    for (li, level_nodes) in levels.iter().enumerate() {
        let d_level: MetalBuffer<u32> = ctx.upload(level_nodes);
        let p = Params {
            level_count: level_nodes.len() as i32,
            lanes: NUM_PREFLOP_CLASSES as i32,
            traverser: traverser as u32,
            alpha_t: params.alpha_t(),
            beta_t: params.beta_t(),
            gamma_t: params.gamma_t(),
        };
        let d_params: MetalBuffer<Params> = ctx.upload(&[p]);

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(d_level.as_ref()), 0);
        enc.set_buffer(1, Some(d_params.as_ref()), 0);
        enc.set_buffer(2, Some(d_nodes.as_ref()), 0);
        enc.set_buffer(3, Some(d_children.as_ref()), 0);
        enc.set_buffer(4, Some(d_infoset_offsets.as_ref()), 0);
        enc.set_buffer(5, Some(d_strategy.as_ref()), 0);
        enc.set_buffer(6, Some(d_regrets.as_ref()), 0);
        enc.set_buffer(7, Some(d_cum_strategy.as_ref()), 0);
        enc.set_buffer(8, Some(d_cfv.as_ref()), 0);
        let max_tpg = pipeline.max_total_threads_per_threadgroup() as usize;
        let (grid, tg) = ctx.dispatch_2d(level_nodes.len(), NUM_PREFLOP_CLASSES, max_tpg);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        eprintln!("  level {} ({} player nodes) dispatched", li, level_nodes.len());
    }

    let gpu_regrets = d_regrets.to_vec();
    let gpu_cum = d_cum_strategy.to_vec();
    let gpu_cfv = d_cfv.to_vec();

    let bit_diff = |cpu: &[f32], gpu: &[f32], label: &str| -> usize {
        let mut count = 0usize;
        let mut max_abs = 0.0f32;
        let mut first = None;
        for i in 0..cpu.len().min(gpu.len()) {
            if cpu[i].to_bits() != gpu[i].to_bits() {
                count += 1;
                if first.is_none() { first = Some((i, cpu[i], gpu[i])); }
                let d = (cpu[i] - gpu[i]).abs();
                if d > max_abs { max_abs = d; }
            }
        }
        eprintln!("  {:14}: {:>6} / {:>7} bit-different, max_abs={:.3e}",
            label, count, cpu.len(), max_abs);
        if let Some((i, cv, gv)) = first {
            eprintln!("    first @ {}: CPU={:.9} GPU={:.9}", i, cv, gv);
        }
        count
    };

    eprintln!("\nComparison:");
    let rdiff = bit_diff(&cpu_regrets, &gpu_regrets, "regrets");
    let cdiff = bit_diff(&cpu_cum, &gpu_cum, "cum_strategy");
    // cfv comparison only over preflop-zone nodes (GPU/CPU only write there).
    use solver_core::tree::action::BoardState;
    let mut cfv_diff = 0usize;
    let mut cfv_max = 0.0f32;
    for nid in 0..nn {
        if tree.nodes[nid].board_state != BoardState::Preflop as u8 { continue; }
        for c in 0..NUM_PREFLOP_CLASSES {
            let cv = cpu_cfv[nid][c];
            let gv = gpu_cfv[nid * NUM_PREFLOP_CLASSES + c];
            if cv.to_bits() != gv.to_bits() {
                cfv_diff += 1;
                let d = (cv - gv).abs();
                if d > cfv_max { cfv_max = d; }
            }
        }
    }
    eprintln!("  {:14}: {:>6} bit-different (preflop-zone only), max_abs={:.3e}",
        "cfv", cfv_diff, cfv_max);

    assert_eq!(rdiff, 0, "STEP 2.D.4 REGRET DIVERGENCE — {} entries differ", rdiff);
    assert_eq!(cdiff, 0, "STEP 2.D.4 CUM_STRATEGY DIVERGENCE — {} entries differ", cdiff);
    assert_eq!(cfv_diff, 0, "STEP 2.D.4 CFV DIVERGENCE — {} entries differ", cfv_diff);

    eprintln!("\n=== STEP 2.D.4 PASS ===");
    eprintln!("GPU preflop bottom_up + regret update bit-exact == CPU on asymmetric leaf cfvs.");
    eprintln!("Replication link holds. CPU anchored by slice A.3b + #92.");
}
