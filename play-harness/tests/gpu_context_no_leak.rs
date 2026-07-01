//! Validates the shared-MetalContext fix for the wired-GPU-memory leak: creating
//! a fresh `MetalContext` per request (new command queue + metallib load) leaked
//! GPU memory the driver never reclaimed. `shared_context()` loads it ONCE and
//! reuses it for every solve.
//!
//! This test deliberately exercises ONLY the shared pattern — it does NOT loop
//! the per-request `MetalContext::new()` pattern, because doing so re-creates the
//! very leak (each new context leaks driver-side wired memory that only a reboot
//! reclaims). We assert that repeated solves through the shared context do not
//! accumulate GPU allocation (per-solve buffers are freed; the context/metallib
//! is loaded once).

#![cfg(feature = "metal")]

use solver_core::card::card_from_str;
use solver_core::gpu_metal::{gpu_hu_turn_strat, shared_context, GpuSearchCfg};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::build_tree_depth_limited;

fn card(s: &str) -> u8 { card_from_str(s).unwrap() as u8 }

#[test]
fn shared_context_does_not_leak_gpu_memory() {
    let board = [card("As"), card("Ah"), card("7c"), card("2d")];
    let cfg_t = production_game_v1().street_seam_config(
        BoardState::Turn, 2, 20, 40,
        BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
    );
    let tree = build_tree_depth_limited(&cfg_t).expect("turn tree");
    let bmask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let nh = (0..(52 * 51 / 2))
        .filter(|&idx| { let (c1, c2) = solver_core::card::index_to_card_pair(idx);
            bmask & (1u64 << c1) == 0 && bmask & (1u64 << c2) == 0 })
        .count();
    let reach = vec![vec![1.0f32; nh]; 2];
    let gcfg = GpuSearchCfg { iters: 30, sample_m: 0, seed: 7, factored_terminals: false, lambda: 0.0 };

    let ctx = shared_context().expect("Metal");
    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);

    // Warm the context once (metallib load + first buffers), then measure the
    // baseline. Every subsequent solve reuses this SAME context.
    let _ = gpu_hu_turn_strat(ctx, &board, &tree, &reach, 64, true, &gcfg);
    let base = ctx.allocated_size();

    let n = 60usize;
    let mut peak = base;
    for i in 0..n {
        let ctx = shared_context().expect("shared"); // same &'static context each call
        let _ = gpu_hu_turn_strat(ctx, &board, &tree, &reach, 64, true, &gcfg);
        let now = ctx.allocated_size();
        peak = peak.max(now);
        if i % 20 == 19 {
            eprintln!("  after {} shared solves: allocated={:.1}MB", i + 1, mb(now));
        }
    }
    let end = ctx.allocated_size();
    eprintln!("shared context over {n} solves: base={:.1}MB end={:.1}MB peak={:.1}MB",
        mb(base), mb(end), mb(peak));

    // No sustained accumulation: end must not sit far above the warmed baseline
    // (small slack for allocator/driver noise). A per-context leak would grow this
    // monotonically with n.
    assert!(
        end < base + 64 * 1024 * 1024,
        "shared context accumulated {:.1}MB over {n} solves (leak)", mb(end.saturating_sub(base))
    );
}
