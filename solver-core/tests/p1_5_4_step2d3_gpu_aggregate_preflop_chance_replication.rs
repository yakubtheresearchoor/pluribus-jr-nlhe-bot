// Step 2.D.3: GPU aggregate_preflop_chance bit-exact replication gate.
//
// THE GATE STRUCTURE (per #79, #84, #92):
//   - CPU correctness anchored by P5a (p5a_preflop_chance_orchestrator_anchor.rs,
//     f64-discrimination) and P2.5a (p2_5a_preflop_chance_anchor.rs,
//     structural independence vs 22100-flop reference).
//   - This test: REPLICATION ONLY. GPU's aggregate kernel must produce
//     bit-exact output relative to CPU's `aggregate_preflop_chance` on
//     realistic asymmetric leaf CFVs.
//
// THE KERNEL:
//   New kernel `vcfr_aggregate_preflop_chance` (added to vcfr.metal):
//   per-class thread sums Σ over canonical of prob_table[canonical, class]
//   × flop_cfvs[canonical, class]. Same iteration order as CPU: outer
//   over canonicals so per-class sums see canonical CFVs in canonical
//   index order, matching CPU's `for (canonical_idx, cfvs_for_flop) in
//   flop_cfvs.iter().enumerate()` order in preflop_start_game.rs:790.
//
// THE INPUTS:
//   - prob_table[1755 × 169] precomputed on CPU from
//     `PreflopChanceTable::chance_probability_flop(canonical, class)`.
//     This is shared with production (it's the same primitive both sides
//     would call); it's already anchored by P5a's f64 check.
//   - flop_cfvs[1755 × 169] is the asymmetric input being aggregated.
//     We construct it as deterministic asymmetric per (canonical, class).

#![cfg(feature = "metal")]

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::MetalBuffer;
use solver_core::solver::preflop_start_game::{aggregate_preflop_chance, PreflopChanceTable};

fn build_table() -> PreflopChanceTable {
    let np = 2u8;
    // Asymmetric class weights matching the 2.A.2 sigmoid harness shape.
    let mut class_weights: Vec<Vec<f32>> = (0..np).map(|_| vec![0.0f32; NUM_PREFLOP_CLASSES]).collect();
    for k in 0..NUM_PREFLOP_CLASSES {
        let s = k as f32 / NUM_PREFLOP_CLASSES as f32;
        let p0 = (s - 0.3).max(0.05) * 1.5;
        let p0 = p0.min(1.0);
        let p1 = 0.6 + 0.4 * s;
        class_weights[0][k] = p0;
        class_weights[1][k] = p1;
    }
    PreflopChanceTable::new(np, class_weights)
}

/// Deterministic asymmetric per-(canonical, class) leaf CFV.
fn make_asymmetric_flop_cfvs(table: &PreflopChanceTable) -> Vec<Vec<f32>> {
    let n_canon = table.num_canonical_flops();
    (0..n_canon).map(|canonical_idx| {
        let f = table.canonical_flops[canonical_idx];
        let canon_seed: u64 = (f[0] as u64) << 16 | (f[1] as u64) << 8 | (f[2] as u64);
        (0..NUM_PREFLOP_CLASSES).map(|c| {
            let mix: u64 = canon_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                ^ (canonical_idx as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            let bits = ((mix >> 32) & 0xFFFFFF) as i64 - (1 << 23);
            (bits as f32) / ((1 << 23) as f32)
        }).collect()
    }).collect()
}

#[test]
#[ignore = "Step 2.D.3: GPU aggregate_preflop_chance replication gate (slow: builds 1755-canonical table)"]
fn step2d3_gpu_aggregate_preflop_chance_matches_cpu_bit_exactly() {
    eprintln!("\n=== Step 2.D.3: GPU aggregate_preflop_chance bit-exact gate ===");
    eprintln!("Building PreflopChanceTable (1755 canonical orbits)...");
    let table = build_table();
    let n_canon = table.num_canonical_flops();
    assert_eq!(n_canon, 1755);

    let flop_cfvs = make_asymmetric_flop_cfvs(&table);

    // ── CPU. ──
    let cpu_out = aggregate_preflop_chance(&table, &flop_cfvs);

    // ── GPU. ──
    let ctx = MetalContext::new().expect("Metal");
    let pipeline = ctx.create_pipeline("vcfr_aggregate_preflop_chance")
        .expect("aggregate_preflop_chance pipeline");

    // Precompute prob_table[canonical * nh + class] on CPU. This is the
    // same primitive CPU production calls (`chance_probability_flop`),
    // anchored at the f64 level by P5a.
    let mut prob_table = vec![0.0f32; n_canon * NUM_PREFLOP_CLASSES];
    for canonical_idx in 0..n_canon {
        for class_idx in 0..NUM_PREFLOP_CLASSES {
            prob_table[canonical_idx * NUM_PREFLOP_CLASSES + class_idx] =
                table.chance_probability_flop(canonical_idx, class_idx);
        }
    }
    let d_prob: MetalBuffer<f32> = ctx.upload(&prob_table);

    // Flatten flop_cfvs in canonical-major layout (canonical × class).
    let mut flat_cfvs = vec![0.0f32; n_canon * NUM_PREFLOP_CLASSES];
    for canonical_idx in 0..n_canon {
        for class_idx in 0..NUM_PREFLOP_CLASSES {
            flat_cfvs[canonical_idx * NUM_PREFLOP_CLASSES + class_idx] =
                flop_cfvs[canonical_idx][class_idx];
        }
    }
    let d_cfvs: MetalBuffer<f32> = ctx.upload(&flat_cfvs);
    let d_out: MetalBuffer<f32> = ctx.alloc_zeros(NUM_PREFLOP_CLASSES);

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Params { n_canon: i32, nh: i32 }
    let params = Params { n_canon: n_canon as i32, nh: NUM_PREFLOP_CLASSES as i32 };
    let d_params: MetalBuffer<Params> = ctx.upload(&[params]);

    let cmd = ctx.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(d_out.as_ref()), 0);
    enc.set_buffer(1, Some(d_prob.as_ref()), 0);
    enc.set_buffer(2, Some(d_cfvs.as_ref()), 0);
    enc.set_buffer(3, Some(d_params.as_ref()), 0);

    let max_tpg = pipeline.max_total_threads_per_threadgroup() as usize;
    let (grid, tg) = ctx.dispatch_1d(NUM_PREFLOP_CLASSES, max_tpg);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_out = d_out.to_vec();

    let mut bit_diff_count = 0usize;
    let mut max_abs = 0.0f32;
    let mut first_diff: Option<(usize, f32, f32)> = None;
    for c in 0..NUM_PREFLOP_CLASSES {
        if cpu_out[c].to_bits() != gpu_out[c].to_bits() {
            bit_diff_count += 1;
            if first_diff.is_none() { first_diff = Some((c, cpu_out[c], gpu_out[c])); }
            let d = (cpu_out[c] - gpu_out[c]).abs();
            if d > max_abs { max_abs = d; }
        }
    }
    eprintln!("\nout buffer comparison ({} classes):", NUM_PREFLOP_CLASSES);
    eprintln!("  {} bit-different", bit_diff_count);
    eprintln!("  max_abs diff: {:.6e}", max_abs);
    if let Some((c, cv, gv)) = first_diff {
        eprintln!("  first diff at class {}: CPU={:.9} GPU={:.9}", c, cv, gv);
        eprintln!("    CPU bits = {:032b}", cv.to_bits());
        eprintln!("    GPU bits = {:032b}", gv.to_bits());
    }

    assert_eq!(bit_diff_count, 0,
        "STEP 2.D.3 REPLICATION GATE BROKEN: {} classes differ. \
         Per #84: run p5a + p2_5a CPU oracles FIRST to disambiguate.",
        bit_diff_count);

    eprintln!("\n=== STEP 2.D.3 PASS ===");
    eprintln!("GPU aggregate_preflop_chance bit-exact == CPU on asymmetric leaf CFVs.");
    eprintln!("Replication link holds. CPU correctness anchored by P5a + P2.5a + #92.");
}
