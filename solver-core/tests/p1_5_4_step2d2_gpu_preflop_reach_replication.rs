// Step 2.D.2: GPU compute_preflop_reach bit-exact replication gate.
//
// THE GATE STRUCTURE (per #79, #84, #92):
//   - CPU correctness anchored by p1_5_4_slice_a2_preflop_reach (independent
//     recursive top-down walk) + #92 (multi-iter loop composition).
//   - This test: REPLICATION ONLY. GPU's top-down reach must produce
//     bit-exact output relative to CPU's per-player per-class layout.
//
// THE KERNEL REUSE INSIGHT:
//   The existing `vcfr_top_down_reach` Metal kernel processes one level at
//   a time. At each player node it scales the acting player's reach by
//   strategy[a * lanes + lane], where `lanes` is the per-infoset lane count
//   (postflop: nh; preflop: NUM_PREFLOP_CLASSES). Same arithmetic as CPU
//   `propagate_reach_from`, so substituting lanes = NUM_PREFLOP_CLASSES
//   makes it preflop-correct.
//
// The GPU layout is (node, player, lane) flat; CPU is per-player vectors of
// (node, class). The test converts GPU output to per-player view and
// compares bit-exact.

#![cfg(feature = "metal")]

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::MetalBuffer;
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

/// BFS levels of the preflop zone starting from root. A node is in the
/// preflop zone iff `board_state == Preflop`. We process levels iff the
/// PARENT is in the preflop zone, because the kernel writes reach to
/// CHILDREN. So the level list is "preflop player + chance nodes whose
/// children need their reach set". The kernel walks chance + terminal as
/// pass-through within the preflop zone; we stop emitting levels once no
/// more preflop-zone nodes appear.
fn preflop_zone_levels(tree: &FlatTree) -> Vec<Vec<u32>> {
    use solver_core::tree::action::BoardState;
    let mut levels: Vec<Vec<u32>> = Vec::new();
    let mut current: Vec<u32> = vec![0u32];
    while !current.is_empty() {
        // Only include nodes whose board_state is Preflop in the level
        // (so the kernel knows to process them with the preflop walk).
        let preflop_in_level: Vec<u32> = current.iter().cloned()
            .filter(|&nid| tree.nodes[nid as usize].board_state == BoardState::Preflop as u8)
            .collect();
        if preflop_in_level.is_empty() { break; }
        levels.push(preflop_in_level.clone());
        let mut next: Vec<u32> = Vec::new();
        for &nid in &preflop_in_level {
            for &child in tree.node_children(nid as usize) {
                next.push(child);
            }
        }
        current = next;
    }
    levels
}

#[test]
#[ignore = "Step 2.D.2: GPU preflop reach replication gate"]
fn step2d2_gpu_preflop_reach_matches_cpu_bit_exactly() {
    let tree = build_hu_preflop_tree();
    let np = tree.num_players as usize;
    let nn = tree.num_nodes();
    eprintln!("\n=== Step 2.D.2: GPU compute_preflop_reach bit-exact gate ===");
    eprintln!("Tree: {} nodes, np={}", nn, np);

    let mut prod = PreflopVectorCfr::new(&tree);
    let infoset_count = prod.infoset_count;

    // Seed asymmetric regrets across all infosets, then compute strategy so
    // every per-class strategy entry has realistic non-uniform content.
    for idx in 0..nn {
        let local = prod.local_offset[idx];
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
                prod.regrets[off + a * NUM_PREFLOP_CLASSES + c] = bias;
            }
        }
    }
    prod.compute_preflop_strategy(&tree);

    // Asymmetric per-player per-class initial reach.
    let initial_reach: Vec<Vec<f32>> = (0..np).map(|p| {
        (0..NUM_PREFLOP_CLASSES).map(|c| {
            let strength_frac = c as f32 / NUM_PREFLOP_CLASSES as f32;
            if p == 0 {
                ((strength_frac - 0.3).max(0.05) * 1.5).min(1.0)
            } else {
                0.6 + 0.4 * strength_frac
            }
        }).collect()
    }).collect();

    // ── CPU reach. ──
    let cpu_reach = prod.compute_preflop_reach(&tree, Some(&initial_reach));

    // ── GPU reach. ──
    let ctx = MetalContext::new().expect("Metal");
    let pipeline = ctx.create_pipeline("vcfr_top_down_reach").expect("top_down pipeline");

    // GPU reach buffer: nn * np * NUM_PREFLOP_CLASSES, init to 0 then write
    // initial reach into root slot (node 0).
    let mut h_reach: Vec<f32> = vec![0.0f32; nn * np * NUM_PREFLOP_CLASSES];
    for p in 0..np {
        for c in 0..NUM_PREFLOP_CLASSES {
            h_reach[0 * np * NUM_PREFLOP_CLASSES + p * NUM_PREFLOP_CLASSES + c] = initial_reach[p][c];
        }
    }
    let d_reach: MetalBuffer<f32> = ctx.upload(&h_reach);

    // d_nodes: same byte-level upload as 2.D.1.
    let nodes_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            tree.nodes.as_ptr() as *const u8,
            tree.nodes.len() * std::mem::size_of::<FlatNode>(),
        )
    };
    let d_nodes: MetalBuffer<u8> = ctx.upload(nodes_bytes);
    let d_children: MetalBuffer<u32> = ctx.upload(&tree.children);

    // Strategy in preflop layout: infoset_count * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES.
    let total_strat = infoset_count * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
    let d_strategy: MetalBuffer<f32> = ctx.upload(&prod.strategy[..total_strat]);

    // infoset_offsets[node_id] → local_offset (CPU's preflop infoset id).
    // For non-preflop-player nodes pass u32::MAX (kernel only reads this
    // when node is a player).
    let infoset_offsets: Vec<u32> = (0..nn).map(|idx| {
        let local = prod.local_offset[idx];
        if local == usize::MAX { u32::MAX } else { local as u32 }
    }).collect();
    let d_infoset_offsets: MetalBuffer<u32> = ctx.upload(&infoset_offsets);

    let levels = preflop_zone_levels(&tree);
    eprintln!("Preflop levels: {} levels, sizes = {:?}",
        levels.len(), levels.iter().map(|l| l.len()).collect::<Vec<_>>());

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TopDownParams { level_count: i32, num_players: i32, nh: i32 }

    for (li, level_nodes) in levels.iter().enumerate() {
        let d_level: MetalBuffer<u32> = ctx.upload(level_nodes);
        let params = TopDownParams {
            level_count: level_nodes.len() as i32,
            num_players: np as i32,
            nh: NUM_PREFLOP_CLASSES as i32,
        };
        let d_params: MetalBuffer<TopDownParams> = ctx.upload(&[params]);

        let cmd = ctx.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(d_level.as_ref()), 0);
        enc.set_buffer(1, Some(d_params.as_ref()), 0);
        enc.set_buffer(2, Some(d_nodes.as_ref()), 0);
        enc.set_buffer(3, Some(d_children.as_ref()), 0);
        enc.set_buffer(4, Some(d_strategy.as_ref()), 0);
        enc.set_buffer(5, Some(d_infoset_offsets.as_ref()), 0);
        enc.set_buffer(6, Some(d_reach.as_ref()), 0);

        let max_tpg = pipeline.max_total_threads_per_threadgroup() as usize;
        let (grid, tg) = ctx.dispatch_2d(level_nodes.len(), NUM_PREFLOP_CLASSES, max_tpg);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        eprintln!("  level {} ({} nodes) dispatched", li, level_nodes.len());
    }

    let gpu_reach = d_reach.to_vec();

    // Compare per-player per-(node, class) bit-exactly.
    // CPU layout: cpu_reach[player] is Vec<f32> of length nn * NUM_PREFLOP_CLASSES,
    //   indexed as [node_idx * NUM_PREFLOP_CLASSES + class].
    // GPU layout: gpu_reach[node_idx * np * NUM_PREFLOP_CLASSES + player * NUM_PREFLOP_CLASSES + class].
    //
    // Only compare PREFLOP-ZONE nodes — CPU only writes reach for those.
    // Non-preflop nodes have CPU reach == 0 (initial); GPU also has them
    // as 0 (we initialized to 0 and didn't dispatch on them). But to be
    // safe, restrict comparison to preflop-zone-reachable nodes.
    use solver_core::tree::action::BoardState;
    let mut bit_diff_count = 0usize;
    let mut max_abs = 0.0f32;
    let mut first_diff: Option<(usize, usize, usize, f32, f32)> = None;
    let mut total_compared = 0usize;
    for node_idx in 0..nn {
        // Determine if this node was touched in CPU's preflop walk: any
        // node visited by propagate_reach_from. We include preflop-zone
        // nodes AND children of preflop-zone nodes (the chance/terminal
        // boundary nodes whose reach was written but not propagated past).
        let board_state = tree.nodes[node_idx].board_state;
        let is_preflop_or_boundary_child = if board_state == BoardState::Preflop as u8 {
            true
        } else {
            // Check if any parent is preflop. Tree builder sets children
            // such that we can find parents by linear scan. For test
            // tractability scan once.
            let mut parent_is_preflop = false;
            for parent_idx in 0..nn {
                if tree.nodes[parent_idx].board_state != BoardState::Preflop as u8 { continue; }
                for &c in tree.node_children(parent_idx) {
                    if c as usize == node_idx { parent_is_preflop = true; break; }
                }
                if parent_is_preflop { break; }
            }
            parent_is_preflop
        };
        if !is_preflop_or_boundary_child { continue; }
        for p in 0..np {
            for c in 0..NUM_PREFLOP_CLASSES {
                let cv = cpu_reach[p][node_idx * NUM_PREFLOP_CLASSES + c];
                let gv = gpu_reach[node_idx * np * NUM_PREFLOP_CLASSES + p * NUM_PREFLOP_CLASSES + c];
                total_compared += 1;
                if cv.to_bits() != gv.to_bits() {
                    bit_diff_count += 1;
                    if first_diff.is_none() {
                        first_diff = Some((node_idx, p, c, cv, gv));
                    }
                    let d = (cv - gv).abs();
                    if d > max_abs { max_abs = d; }
                }
            }
        }
    }
    eprintln!("\nReach buffer comparison ({} cells in preflop zone + boundary):", total_compared);
    eprintln!("  {} bit-different", bit_diff_count);
    eprintln!("  max_abs diff: {:.6e}", max_abs);
    if let Some((node, p, c, cv, gv)) = first_diff {
        eprintln!("  first diff at node {}, player {}, class {}: CPU={:.9} GPU={:.9}",
            node, p, c, cv, gv);
        eprintln!("    CPU bits = {:032b}", cv.to_bits());
        eprintln!("    GPU bits = {:032b}", gv.to_bits());
    }

    assert_eq!(bit_diff_count, 0,
        "STEP 2.D.2 REPLICATION GATE BROKEN: {} reach cells differ. \
         Per #84: run slice A.2 oracle on CPU FIRST to disambiguate. \
         If slice A.2 passes, replication drift in GPU top-down kernel; \
         if slice A.2 fails, CPU regression.",
        bit_diff_count);

    eprintln!("\n=== STEP 2.D.2 PASS ===");
    eprintln!("GPU compute_preflop_reach bit-exact == CPU on asymmetric initial reach + strategy.");
    eprintln!("Replication link holds. CPU correctness anchored by slice A.2 + #92.");
}
