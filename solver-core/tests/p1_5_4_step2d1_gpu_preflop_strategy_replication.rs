// Step 2.D.1: GPU compute_preflop_strategy bit-exact replication gate.
//
// THE GATE STRUCTURE (per #79, #84, #92):
//   - CPU side correctness: anchored by p1_5_4_slice_a1_preflop_strategy
//     (textbook regret-matching at f32 floor, per-class asymmetric regret
//     patterns) AND by p1_5_4_step2d0c (multi-iter loop composition vs
//     independent textbook reference).
//   - This test: REPLICATION ONLY. Does GPU's compute_preflop_strategy
//     produce bit-exact output relative to CPU's? If yes, replication
//     holds and GPU inherits CPU's anchored correctness via #79.
//
// THE KERNEL REUSE INSIGHT:
//   The existing `vcfr_compute_strategies` kernel in vcfr.metal:808 is
//   PARAMETRIC OVER `nh` — its arithmetic does regret-matching per
//   (infoset, lane) where lane ∈ [0, nh). For postflop, lane=hand. For
//   preflop, lane=class. The kernel does not assume hand-specific
//   semantics; it just operates on per-infoset strides of
//   `MAX_NA_PREFLOP * lanes` floats with per-lane regret-matching. So preflop
//   reuses the same kernel by passing nh = NUM_PREFLOP_CLASSES.
//
// THE TEST:
//   1. Build a small HU preflop tree.
//   2. CPU: PreflopVectorCfr::new, seed asymmetric per-class regrets at
//      multiple infosets and classes (designed test patterns from slice
//      A.1), call compute_preflop_strategy.
//   3. GPU: upload same regrets, dispatch `vcfr_compute_strategies` with
//      nh = NUM_PREFLOP_CLASSES, download strategy.
//   4. Bit-exact comparison via to_bits().

#![cfg(feature = "metal")]

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

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

/// Build a list of (decision_node_id, local_offset_index) for all preflop
/// player decision nodes, ordered by local_offset_index. The kernel reads
/// decision_node_ids[infoset_id] → node_id, so the ordering must match
/// CPU's local_offset assignment.
fn preflop_decision_node_ids(prod: &PreflopVectorCfr, tree: &FlatTree) -> Vec<u32> {
    let nn = tree.num_nodes();
    let mut by_local: Vec<(usize, u32)> = Vec::new();
    for idx in 0..nn {
        let local = prod.local_offset[idx];
        if local == usize::MAX { continue; }
        by_local.push((local, idx as u32));
    }
    by_local.sort_by_key(|&(local, _)| local);
    // local_offset values run 0..infoset_count contiguously, so this just
    // re-orders to that contiguous sequence.
    by_local.into_iter().map(|(_, id)| id).collect()
}

#[test]
#[ignore = "Step 2.D.1: GPU preflop strategy replication gate"]
fn step2d1_gpu_preflop_strategy_matches_cpu_bit_exactly() {
    let tree = build_hu_preflop_tree();
    eprintln!("\n=== Step 2.D.1: GPU compute_preflop_strategy bit-exact gate ===");
    eprintln!("Tree: {} nodes", tree.num_nodes());

    let mut prod = PreflopVectorCfr::new(&tree);
    let infoset_count = prod.infoset_count;
    eprintln!("Preflop infoset count: {}", infoset_count);

    // ── Seed asymmetric per-class regrets across ALL infosets. ──
    //
    // Per-class pattern (from slice A.1 + the audit-arc discipline of
    // realistic asymmetric inputs that exposed the 2.A.2 compounder):
    //   class index c contributes a class-specific bias
    //   action a contributes a position-specific value
    //   infoset_idx perturbs both so different infosets see different
    //     local minima/uniform-fallback regions
    //
    // Aim: every regret-matching code path is exercised at multiple
    // infosets — positive-only, mixed-sign, near-eps uniform-fallback,
    // exact-zero uniform-fallback.
    {
        let total = infoset_count * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
        for r in prod.regrets.iter_mut().take(total) { *r = 0.0; }
        for idx in 0..tree.num_nodes() {
            let local = prod.local_offset[idx];
            if local == usize::MAX { continue; }
            let na = tree.nodes[idx].num_children as usize;
            if na == 0 { continue; }
            let off = local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
            for c in 0..NUM_PREFLOP_CLASSES {
                let class_phase = (c + local * 13) % 5;
                for a in 0..na {
                    let av = (a as i32) - (na as i32 / 2);
                    let base = match class_phase {
                        0 => 1.0 + a as f32 + (c as f32 * 0.01),  // positive-dominant
                        1 => if a % 2 == 0 { 2.5 } else { -1.3 }, // mixed sign
                        2 => 1e-7,                                 // below eps, uniform fallback
                        3 => 0.0,                                  // exact zero, uniform fallback
                        _ => (av as f32) * 0.5 + (c as f32 * 0.003) + (local as f32) * 0.07,
                    };
                    prod.regrets[off + a * NUM_PREFLOP_CLASSES + c] = base;
                }
            }
        }
    }

    // Capture seed regrets BEFORE running CPU compute (compute_preflop_strategy
    // doesn't mutate regrets, but capture defensively for GPU upload below).
    let cpu_regrets_snapshot = prod.regrets.clone();

    // ── CPU compute. ──
    prod.compute_preflop_strategy(&tree);
    let cpu_strategy = prod.strategy.clone();

    // ── GPU compute. ──
    let ctx = MetalContext::new().expect("Metal");

    // Pipeline.
    let pipeline = ctx.create_pipeline("vcfr_compute_strategies").expect("strategies pipeline");

    // decision_node_ids: ordered by local_offset (matches the kernel's
    // infoset_id → node_id mapping).
    let decision_ids = preflop_decision_node_ids(&prod, &tree);
    assert_eq!(decision_ids.len(), infoset_count);
    let d_decision_ids = ctx.upload(&decision_ids);

    // d_nodes: upload FlatNode slice as-is. The Metal FlatNode struct
    // layout must match the Rust FlatNode layout. Reuse the existing
    // FlatNode upload pattern.
    let nodes_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            tree.nodes.as_ptr() as *const u8,
            tree.nodes.len() * std::mem::size_of::<solver_core::tree::flat::FlatNode>(),
        )
    };
    // The MetalBuffer<T> generic doesn't directly know FlatNode size, but
    // we can construct via raw byte upload. Easier: use upload<u8> and rely
    // on the kernel reading by FlatNode pointer arithmetic.
    use solver_core::gpu_metal::MetalBuffer;
    // Use upload<u8> at the byte level then bind that buffer as device const
    // FlatNode*. MetalBuffer<u8> works since we go through as_ref() which
    // hands the underlying BufferRef.
    let d_nodes_u8: MetalBuffer<u8> = ctx.upload(nodes_bytes);

    // infoset_offsets: kernel slot 4 declared as `device const uint32_t*`
    // but the kernel BODY does not read this buffer for compute_strategies.
    // Pass a 1-element zero buffer to satisfy the binding.
    let d_infoset_offsets: MetalBuffer<u32> = ctx.upload(&[0u32]);

    // params: { num_infosets, nh = NUM_PREFLOP_CLASSES }. The existing Rust
    // Params struct in flop_solver also includes base_offset which the
    // Metal struct does not declare — extra bytes are ignored.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Params { num_infosets: i32, nh: i32, base_offset: i32 }
    let params = Params {
        num_infosets: infoset_count as i32,
        nh: NUM_PREFLOP_CLASSES as i32,
        base_offset: 0,
    };
    let d_params: MetalBuffer<Params> = ctx.upload(&[params]);

    // regrets / strategy buffers. Stride = MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES per infoset.
    let total = infoset_count * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
    let d_regrets = ctx.upload(&cpu_regrets_snapshot[..total]);
    let d_strategy: MetalBuffer<f32> = ctx.alloc_zeros(total);

    // Dispatch.
    let cmd = ctx.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(d_regrets.as_ref()), 0);
    enc.set_buffer(1, Some(d_strategy.as_ref()), 0);
    enc.set_buffer(2, Some(d_decision_ids.as_ref()), 0);
    enc.set_buffer(3, Some(d_nodes_u8.as_ref()), 0);
    enc.set_buffer(4, Some(d_infoset_offsets.as_ref()), 0);
    enc.set_buffer(5, Some(d_params.as_ref()), 0);

    let max_tpg = pipeline.max_total_threads_per_threadgroup() as usize;
    let (grid, tg) = ctx.dispatch_2d(infoset_count, NUM_PREFLOP_CLASSES, max_tpg);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_strategy = d_strategy.to_vec();

    // ── Bit-exact comparison. ──
    let mut bit_diff_count = 0usize;
    let mut max_abs = 0.0f32;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for i in 0..total.min(cpu_strategy.len()).min(gpu_strategy.len()) {
        let cv = cpu_strategy[i];
        let gv = gpu_strategy[i];
        if cv.to_bits() != gv.to_bits() {
            bit_diff_count += 1;
            if first_diff.is_none() { first_diff = Some((i, cv, gv)); }
            let d = (cv - gv).abs();
            if d > max_abs { max_abs = d; }
        }
    }
    eprintln!("\nStrategy buffer comparison:");
    eprintln!("  {} / {} entries bit-different", bit_diff_count, total);
    eprintln!("  max_abs diff: {:.6e}", max_abs);
    if let Some((i, cv, gv)) = first_diff {
        let infoset_id = i / (MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES);
        let within = i % (MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES);
        let a = within / NUM_PREFLOP_CLASSES;
        let c = within % NUM_PREFLOP_CLASSES;
        eprintln!("  first diff at idx {} (infoset {}, a={}, class={}): CPU={:.9} GPU={:.9}",
            i, infoset_id, a, c, cv, gv);
        eprintln!("    CPU bits = {:032b}", cv.to_bits());
        eprintln!("    GPU bits = {:032b}", gv.to_bits());
    }

    assert_eq!(bit_diff_count, 0,
        "STEP 2.D.1 REPLICATION GATE BROKEN: {} entries differ. \
         Per #84: run slice A.1 oracle on CPU FIRST to disambiguate. \
         If slice A.1 still passes, this is replication drift in the GPU kernel. \
         If slice A.1 fails, the regression is on CPU.",
        bit_diff_count);

    eprintln!("\n=== STEP 2.D.1 PASS ===");
    eprintln!("GPU compute_preflop_strategy bit-exact == CPU on asymmetric seeded regrets.");
    eprintln!("Replication link holds. CPU correctness anchored by slice A.1 + #92.");
}
