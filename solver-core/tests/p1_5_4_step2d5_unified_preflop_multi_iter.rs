// Step 2.D.5: unified preflop+postflop multi-iter CPU↔GPU REPLICATION gate.
// THE CATCH-THE-COMPOUNDER GATE for the new preflop+postflop composition
// 2.D creates.
//
// THE GATE STRUCTURE (per #79, #84, #92):
//   - CPU correctness for the preflop+postflop multi-iter loop anchored
//     by #92 (textbook-DCFR reference, 10 iters, max_abs = 0.0).
//   - This test: REPLICATION ONLY. GPU's unified run_one_iter (calling
//     the four 2.D.1–2.D.4 kernels in sequence + per-canonical CFV
//     primitive on host) must produce bit-exact strategy/regrets/
//     cum_strategy relative to CPU at every iter checkpoint.
//
// WHY MULTI-ITER IS LOAD-BEARING (the 2.A.2 lesson):
//   The sweep-vs-brute compounder was invisible per-stage at iter 1 (max
//   1e-6 = looks like noise). It compounded 2.8x/iter through CFR feedback
//   to orders-of-magnitude divergence by iter 10. Per-stage replication
//   gates (2.D.1–2.D.4) catch stage-local bugs but STRUCTURALLY CANNOT
//   catch what compounds through the loop. Only this multi-iter gate
//   catches the compounder class for the NEW preflop+postflop wiring.
//
// THE TEST:
//   1. Build the same HU minimal preflop tree + asymmetric class weights
//      + deterministic stub leaf function as #92.
//   2. Initialize CPU PreflopVectorCfr.
//   3. Initialize GPU buffers mirroring CPU's initial state.
//   4. For each of N >= 10 iters:
//        a. CPU::run_one_iteration with ClosureOracle(stub_leaf).
//        b. GPU manual run_one_iter:
//           - vcfr_compute_strategies kernel (2.D.1)
//           - vcfr_top_down_reach kernel per level (2.D.2)
//           - For each traverser:
//             - For each preflop chance node:
//               - For each canonical: expand class→combo + stub_leaf +
//                 reduce combo→class on host (CPU primitives, anchored
//                 P5b/P5a).
//               - vcfr_aggregate_preflop_chance kernel writes directly
//                 into d_cfv slot for that chance node (2.D.3).
//             - vcfr_preflop_bottom_up_player kernel per player level
//               deepest-first (2.D.4).
//        c. Compare strategy/regrets/cum_strategy bit-exact.
//   5. Assert all comparisons pass at every iter.
//
// PER-CANONICAL POSTFLOP SUBSOLVER:
//   The stub_leaf function plays the role of the per-canonical postflop
//   subsolver in this gate. We do NOT exercise the actual MetalFlopStartSolver
//   here because (a) that is its own validated piece (2.A.2 production cell),
//   (b) the load-bearing question at 2.D.5 is the NEW preflop composition,
//   not the postflop solver. Wiring MetalFlopStartSolver into this loop is
//   mechanical once this gate passes.

#![cfg(feature = "metal")]

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::MetalBuffer;
use solver_core::solver::flop_start_vector_cfr::DcfrParams;
use solver_core::solver::postflop_oracle::ClosureOracle;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::solver::preflop_start_game::{
    expand_reach_class_to_combo, flop_combo_layout,
    reduce_cfv_combo_to_class, PreflopChanceTable,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatNode, FlatTree, MAX_NA_PREFLOP};

// ─────────────────────────────────────────────────────────────────────────
// Same setup as #92 (loop-composition anchor): same tree, same class
// weights, same deterministic asymmetric stub leaf.
// ─────────────────────────────────────────────────────────────────────────

fn build_minimal_hu_preflop_tree() -> FlatTree {
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

fn stub_leaf(canonical: [Card; 3], _combo_ranges: &[Vec<f32>], traverser: u8) -> Vec<f32> {
    let layout = flop_combo_layout(canonical);
    let canon_seed: u64 = (canonical[0] as u64) << 16
        | (canonical[1] as u64) << 8
        | (canonical[2] as u64);
    layout.iter().enumerate().map(|(li, &(c1, c2))| {
        let combo_seed: u64 = ((c1 as u64) << 8) | (c2 as u64);
        let trav_seed: u64 = traverser as u64;
        let mix: u64 = canon_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ combo_seed.wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ trav_seed.wrapping_mul(0x94D0_49BB_1331_11EB)
            ^ (li as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
        let bits = ((mix >> 32) & 0xFFFFFF) as i64 - (1 << 23);
        (bits as f32) / ((1 << 23) as f32)
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// Preflop-zone level decomposition for the GPU walk.
// ─────────────────────────────────────────────────────────────────────────

fn preflop_zone_levels_for_reach(tree: &FlatTree) -> Vec<Vec<u32>> {
    let mut levels: Vec<Vec<u32>> = Vec::new();
    let mut current: Vec<u32> = vec![0u32];
    while !current.is_empty() {
        let in_zone: Vec<u32> = current.iter().cloned()
            .filter(|&nid| tree.nodes[nid as usize].board_state == BoardState::Preflop as u8)
            .collect();
        if in_zone.is_empty() { break; }
        levels.push(in_zone.clone());
        let mut next: Vec<u32> = Vec::new();
        for &nid in &in_zone {
            for &c in tree.node_children(nid as usize) {
                next.push(c);
            }
        }
        current = next;
    }
    levels
}

/// Preflop PLAYER levels deepest-first.
fn preflop_player_levels(tree: &FlatTree, prod: &PreflopVectorCfr) -> Vec<Vec<u32>> {
    let nn = tree.num_nodes();
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
        if prod.local_offset[idx] == usize::MAX { continue; }
        by_depth[d as usize].push(idx as u32);
    }
    by_depth.into_iter().rev().filter(|l| !l.is_empty()).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// The unified gate.
// ─────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "Step 2.D.5: CATCH-THE-COMPOUNDER multi-iter CPU↔GPU replication at unified preflop loop"]
fn step2d5_unified_preflop_multi_iter_cpu_gpu_replication() {
    let tree = build_minimal_hu_preflop_tree();
    let np = tree.num_players as usize;
    let nn = tree.num_nodes();
    eprintln!("\n=== Step 2.D.5: unified preflop multi-iter CPU↔GPU replication gate ===");
    eprintln!("Tree: {} nodes, np={}", nn, np);

    // Same asymmetric class weights as #92.
    let mut class_weights: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();
    for k in 0..NUM_PREFLOP_CLASSES {
        let s = k as f32 / NUM_PREFLOP_CLASSES as f32;
        class_weights[0][k] = ((s - 0.3).max(0.05) * 1.5).min(1.0);
        class_weights[1][k] = 0.6 + 0.4 * s;
    }
    eprintln!("Building PreflopChanceTable (1755 canonical orbits)...");
    let table = PreflopChanceTable::new(np as u8, class_weights);
    let n_canon = table.num_canonical_flops();

    // CPU solver state — production reference.
    let mut prod_cpu = PreflopVectorCfr::new(&tree);
    let infoset_count = prod_cpu.infoset_count;
    let stride_per_infoset = MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
    let total_strat = infoset_count * stride_per_infoset;
    let chance_node_indices = prod_cpu.preflop_chance_node_indices(&tree);
    eprintln!("Preflop chance nodes: {}", chance_node_indices.len());
    eprintln!("Preflop infosets: {}", infoset_count);

    let terminal_value_fn = |term_idx: usize, traverser: u8, _r: &[Vec<f32>]| -> Vec<f32> {
        (0..NUM_PREFLOP_CLASSES).map(|c| {
            let seed: u64 = (term_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (traverser as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                ^ (c as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            let bits = ((seed >> 32) & 0xFFFFFF) as i64 - (1 << 23);
            (bits as f32) / ((1 << 23) as f32)
        }).collect()
    };

    // ── GPU one-time setup ──
    let ctx = MetalContext::new().expect("Metal");
    let pipe_strategy = ctx.create_pipeline("vcfr_compute_strategies").unwrap();
    let pipe_top_down = ctx.create_pipeline("vcfr_top_down_reach").unwrap();
    let pipe_aggregate = ctx.create_pipeline("vcfr_aggregate_preflop_chance").unwrap();
    let pipe_bottom_up = ctx.create_pipeline("vcfr_preflop_bottom_up_player").unwrap();

    let nodes_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            tree.nodes.as_ptr() as *const u8,
            tree.nodes.len() * std::mem::size_of::<FlatNode>(),
        )
    };
    let d_nodes: MetalBuffer<u8> = ctx.upload(nodes_bytes);
    let d_children: MetalBuffer<u32> = ctx.upload(&tree.children);

    // decision_ids: preflop player nodes ordered by local_offset.
    let mut by_local: Vec<(usize, u32)> = Vec::new();
    for idx in 0..nn {
        let local = prod_cpu.local_offset[idx];
        if local == usize::MAX { continue; }
        by_local.push((local, idx as u32));
    }
    by_local.sort_by_key(|&(l, _)| l);
    let decision_ids: Vec<u32> = by_local.into_iter().map(|(_, id)| id).collect();
    let d_decision_ids: MetalBuffer<u32> = ctx.upload(&decision_ids);

    let infoset_offsets: Vec<u32> = (0..nn).map(|idx| {
        let local = prod_cpu.local_offset[idx];
        if local == usize::MAX { u32::MAX } else { local as u32 }
    }).collect();
    let d_infoset_offsets: MetalBuffer<u32> = ctx.upload(&infoset_offsets);

    // Initial GPU state mirrors CPU.
    let d_regrets: MetalBuffer<f32> = ctx.upload(&prod_cpu.regrets[..total_strat]);
    let d_strategy: MetalBuffer<f32> = ctx.upload(&prod_cpu.strategy[..total_strat]);
    let d_cum_strategy: MetalBuffer<f32> = ctx.upload(&prod_cpu.cum_strategy[..total_strat]);

    // Precompute prob_table[canonical, class] for the aggregate kernel.
    let mut prob_table = vec![0.0f32; n_canon * NUM_PREFLOP_CLASSES];
    for ci in 0..n_canon {
        for c in 0..NUM_PREFLOP_CLASSES {
            prob_table[ci * NUM_PREFLOP_CLASSES + c] =
                table.chance_probability_flop(ci, c);
        }
    }
    let d_prob_table: MetalBuffer<f32> = ctx.upload(&prob_table);

    // Preflop levels (one-time).
    let reach_levels = preflop_zone_levels_for_reach(&tree);
    let player_levels = preflop_player_levels(&tree, &prod_cpu);

    let n_iters = 10u32;
    let mut all_pass = true;

    for iter in 0..n_iters {
        // ── CPU one iter. ──
        let mut cpu_oracle = ClosureOracle::new(stub_leaf);
        prod_cpu.run_one_iteration(&tree, &table, &mut cpu_oracle, &terminal_value_fn);

        // ── GPU one iter. ──
        let dcfr = DcfrParams::new(iter);

        // 1. compute_preflop_strategy.
        {
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct P { num_infosets: i32, nh: i32, base_offset: i32 }
            let p = P {
                num_infosets: infoset_count as i32,
                nh: NUM_PREFLOP_CLASSES as i32,
                base_offset: 0,
            };
            let d_p: MetalBuffer<P> = ctx.upload(&[p]);
            let d_dummy_offsets: MetalBuffer<u32> = ctx.upload(&[0u32]);
            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe_strategy);
            enc.set_buffer(0, Some(d_regrets.as_ref()), 0);
            enc.set_buffer(1, Some(d_strategy.as_ref()), 0);
            enc.set_buffer(2, Some(d_decision_ids.as_ref()), 0);
            enc.set_buffer(3, Some(d_nodes.as_ref()), 0);
            enc.set_buffer(4, Some(d_dummy_offsets.as_ref()), 0);
            enc.set_buffer(5, Some(d_p.as_ref()), 0);
            let max_tpg = pipe_strategy.max_total_threads_per_threadgroup() as usize;
            let (g, t) = ctx.dispatch_2d(infoset_count, NUM_PREFLOP_CLASSES, max_tpg);
            enc.dispatch_thread_groups(g, t);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // 2. compute_preflop_reach. Initialize d_reach with reach=1.0 at root,
        // 0.0 elsewhere (CPU uses None → all-ones at root).
        let mut h_reach = vec![0.0f32; nn * np * NUM_PREFLOP_CLASSES];
        for p in 0..np {
            for c in 0..NUM_PREFLOP_CLASSES {
                h_reach[0 * np * NUM_PREFLOP_CLASSES + p * NUM_PREFLOP_CLASSES + c] = 1.0;
            }
        }
        let d_reach: MetalBuffer<f32> = ctx.upload(&h_reach);
        for level in &reach_levels {
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct TP { level_count: i32, num_players: i32, nh: i32 }
            let tp = TP {
                level_count: level.len() as i32,
                num_players: np as i32,
                nh: NUM_PREFLOP_CLASSES as i32,
            };
            let d_tp: MetalBuffer<TP> = ctx.upload(&[tp]);
            let d_level: MetalBuffer<u32> = ctx.upload(level);
            let cmd = ctx.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe_top_down);
            enc.set_buffer(0, Some(d_level.as_ref()), 0);
            enc.set_buffer(1, Some(d_tp.as_ref()), 0);
            enc.set_buffer(2, Some(d_nodes.as_ref()), 0);
            enc.set_buffer(3, Some(d_children.as_ref()), 0);
            enc.set_buffer(4, Some(d_strategy.as_ref()), 0);
            enc.set_buffer(5, Some(d_infoset_offsets.as_ref()), 0);
            enc.set_buffer(6, Some(d_reach.as_ref()), 0);
            let max_tpg = pipe_top_down.max_total_threads_per_threadgroup() as usize;
            let (g, t) = ctx.dispatch_2d(level.len(), NUM_PREFLOP_CLASSES, max_tpg);
            enc.dispatch_thread_groups(g, t);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        // Download reach for the chance-node CFV computation (which uses
        // per-class reach per player; we read these from GPU to keep CPU
        // and GPU sides operating on the same reach state).
        let gpu_reach = d_reach.to_vec();

        // 3. Per-traverser.
        for t in 0..np as u8 {
            // d_cfv buffer (per-iter, per-traverser). Zero-init.
            //
            // CRITICAL: populate terminal CFVs from terminal_value_fn BEFORE
            // the bottom-up walk. CPU's bottom_up_recursive does this in-line
            // at terminal leaves (preflop_cfr.rs:617-628); the GPU kernel
            // does not visit terminals, so the host must seed those slots.
            let mut h_cfv = vec![0.0f32; nn * NUM_PREFLOP_CLASSES];
            for nid in 0..nn {
                let n = &tree.nodes[nid];
                if n.board_state != BoardState::Preflop as u8 { continue; }
                if !n.is_terminal() { continue; }
                let term_cfv = terminal_value_fn(nid, t, &Vec::new());
                for c in 0..NUM_PREFLOP_CLASSES {
                    h_cfv[nid * NUM_PREFLOP_CLASSES + c] = term_cfv[c];
                }
            }

            // 3a. For each preflop chance node, compute per-canonical
            // v_class and aggregate via 2.D.3 kernel into d_cfv at the
            // chance_idx slot.
            for &chance_idx in &chance_node_indices {
                // Pull reach slice for this chance node from GPU.
                let chance_base = chance_idx * np * NUM_PREFLOP_CLASSES;
                let mut reach_per_player: Vec<Vec<f32>> = Vec::with_capacity(np);
                for p in 0..np {
                    let start = chance_base + p * NUM_PREFLOP_CLASSES;
                    reach_per_player.push(gpu_reach[start..start + NUM_PREFLOP_CLASSES].to_vec());
                }
                // Compute per-canonical v_class via CPU primitives + stub leaf.
                let mut flat_v = vec![0.0f32; n_canon * NUM_PREFLOP_CLASSES];
                for canonical_idx in 0..n_canon {
                    let f_canon = table.canonical_flops[canonical_idx];
                    let layout = flop_combo_layout(f_canon);
                    let mut combo_reaches: Vec<Vec<f32>> = Vec::with_capacity(np);
                    for p in 0..np {
                        combo_reaches.push(
                            expand_reach_class_to_combo(f_canon, &reach_per_player[p], &layout)
                        );
                    }
                    let v_combo = stub_leaf(f_canon, &combo_reaches, t);
                    let v_class = reduce_cfv_combo_to_class(f_canon, &v_combo, &layout);
                    for c in 0..NUM_PREFLOP_CLASSES {
                        flat_v[canonical_idx * NUM_PREFLOP_CLASSES + c] = v_class[c];
                    }
                }
                let d_flat_v: MetalBuffer<f32> = ctx.upload(&flat_v);

                // 2.D.3 kernel: out → d_cfv at offset chance_idx * NUM_PREFLOP_CLASSES.
                #[repr(C)]
                #[derive(Clone, Copy)]
                struct AP { n_canon: i32, nh: i32 }
                let ap = AP { n_canon: n_canon as i32, nh: NUM_PREFLOP_CLASSES as i32 };
                let d_ap: MetalBuffer<AP> = ctx.upload(&[ap]);
                // Upload d_cfv fresh each chance node call (we'll patch via download/upload).
                let d_cfv: MetalBuffer<f32> = ctx.upload(&h_cfv);
                let out_offset = (chance_idx * NUM_PREFLOP_CLASSES) * std::mem::size_of::<f32>();

                let cmd = ctx.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipe_aggregate);
                enc.set_buffer(0, Some(d_cfv.as_ref()), out_offset as u64);
                enc.set_buffer(1, Some(d_prob_table.as_ref()), 0);
                enc.set_buffer(2, Some(d_flat_v.as_ref()), 0);
                enc.set_buffer(3, Some(d_ap.as_ref()), 0);
                let max_tpg = pipe_aggregate.max_total_threads_per_threadgroup() as usize;
                let (g, tg) = ctx.dispatch_1d(NUM_PREFLOP_CLASSES, max_tpg);
                enc.dispatch_thread_groups(g, tg);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();

                // Download back into h_cfv for the next chance node iter.
                h_cfv = d_cfv.to_vec();
            }

            // Re-upload h_cfv for the bottom-up walk.
            let d_cfv: MetalBuffer<f32> = ctx.upload(&h_cfv);

            // 3b. Bottom-up walk + regret update (2.D.4).
            for level in &player_levels {
                #[repr(C)]
                #[derive(Clone, Copy)]
                struct BP {
                    level_count: i32,
                    lanes: i32,
                    traverser: u32,
                    alpha_t: f32,
                    beta_t: f32,
                    gamma_t: f32,
                }
                let bp = BP {
                    level_count: level.len() as i32,
                    lanes: NUM_PREFLOP_CLASSES as i32,
                    traverser: t as u32,
                    alpha_t: dcfr.alpha_t(),
                    beta_t: dcfr.beta_t(),
                    gamma_t: dcfr.gamma_t(),
                };
                let d_bp: MetalBuffer<BP> = ctx.upload(&[bp]);
                let d_level: MetalBuffer<u32> = ctx.upload(level);
                let cmd = ctx.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipe_bottom_up);
                enc.set_buffer(0, Some(d_level.as_ref()), 0);
                enc.set_buffer(1, Some(d_bp.as_ref()), 0);
                enc.set_buffer(2, Some(d_nodes.as_ref()), 0);
                enc.set_buffer(3, Some(d_children.as_ref()), 0);
                enc.set_buffer(4, Some(d_infoset_offsets.as_ref()), 0);
                enc.set_buffer(5, Some(d_strategy.as_ref()), 0);
                enc.set_buffer(6, Some(d_regrets.as_ref()), 0);
                enc.set_buffer(7, Some(d_cum_strategy.as_ref()), 0);
                enc.set_buffer(8, Some(d_cfv.as_ref()), 0);
                let max_tpg = pipe_bottom_up.max_total_threads_per_threadgroup() as usize;
                let (g, tg) = ctx.dispatch_2d(level.len(), NUM_PREFLOP_CLASSES, max_tpg);
                enc.dispatch_thread_groups(g, tg);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
        }

        // ── Compare state. ──
        let cpu_regrets = &prod_cpu.regrets[..total_strat];
        let cpu_strategy = &prod_cpu.strategy[..total_strat];
        let cpu_cum = &prod_cpu.cum_strategy[..total_strat];
        let gpu_regrets = d_regrets.to_vec();
        let gpu_strategy = d_strategy.to_vec();
        let gpu_cum = d_cum_strategy.to_vec();
        let bit_diff = |cpu: &[f32], gpu: &[f32]| -> (usize, f32) {
            let mut c = 0usize;
            let mut m = 0.0f32;
            for i in 0..cpu.len().min(gpu.len()) {
                if cpu[i].to_bits() != gpu[i].to_bits() {
                    c += 1;
                    let d = (cpu[i] - gpu[i]).abs();
                    if d > m { m = d; }
                }
            }
            (c, m)
        };
        let (rdiff, rmax) = bit_diff(cpu_regrets, &gpu_regrets);
        let (sdiff, smax) = bit_diff(cpu_strategy, &gpu_strategy);
        let (cdiff, cmax) = bit_diff(cpu_cum, &gpu_cum);
        eprintln!("iter {:2}: regrets {:>4} bit-diff (max {:.3e}); strategy {:>4} bit-diff (max {:.3e}); cum_strat {:>4} bit-diff (max {:.3e})",
            iter, rdiff, rmax, sdiff, smax, cdiff, cmax);
        if rdiff > 0 || sdiff > 0 || cdiff > 0 {
            all_pass = false;
            assert_eq!(rdiff, 0, "STEP 2.D.5 COMPOUNDING DIVERGENCE in regrets at iter {}", iter);
            assert_eq!(sdiff, 0, "STEP 2.D.5 COMPOUNDING DIVERGENCE in strategy at iter {}", iter);
            assert_eq!(cdiff, 0, "STEP 2.D.5 COMPOUNDING DIVERGENCE in cum_strategy at iter {}", iter);
        }
    }

    assert!(all_pass, "STEP 2.D.5 unified replication gate failed.");

    eprintln!("\n=== STEP 2.D.5 PASS — CATCH-THE-COMPOUNDER GATE GREEN ===");
    eprintln!("{} iters × full 1755-orbit chance × realistic asymmetric inputs.", n_iters);
    eprintln!("GPU unified preflop loop bit-exact == CPU at every iter.");
    eprintln!("Per #79: replication link holds; CPU correctness anchored by #92.");
    eprintln!("Step 2.D is COMPLETE at the unified preflop multi-iter cell.");
}
