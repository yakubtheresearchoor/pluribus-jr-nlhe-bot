# Phase 0: Environment Setup — Status

## Hardware
- **Machine**: Mac Studio (Model Mac16,9)
- **Chip**: Apple M4 Max (not Pro — even better!)
- **Memory**: 36 GB unified memory
- **GPU**: Metal device "Apple M4 Max"
  - Unified memory: true
  - Recommended max working set: 30.15 GB
  - Max buffer length: 22.6 GB
  - Max threads per threadgroup: 1024
  - Thread execution width (SIMD): 32 (same as CUDA warp size)

## Completed
- [x] Rust toolchain installed (rustc 1.96.0, cargo 1.96.0) on aarch64-apple-darwin
- [x] `metal` crate verified: compiles and runs on M4 Max
- [x] Metal compute shader tested: vector_add kernel passes (end-to-end Metal compute works)
- [x] Command Line Tools installed (macOS 26.2 SDK available)
- [x] solver-core compiles CPU-only on macOS (12 warnings, no errors)
- [x] solver-cli compiles CPU-only on macOS (with cuda feature removed)
- [x] solver-core unit tests pass (8/8)
- [x] solver-core tree builder tests pass (6/6) including 3-player and 6-player trees
- [x] postflop-solver (external) compiles and tests pass (38/38, 4 ignored)
- [x] surstromming-solver-cli (external) compiles

## Blocked / Needs Manual Action
- [ ] **Xcode installation**: Full Xcode needed for `xcrun metal` (Metal shader compiler). Currently only Command Line Tools installed. Must install from App Store or developer.apple.com. This is required for Phase 1+.

## Codebase Assessment
- solver-core has evolved significantly beyond the original plan.md — it now has:
  - **Vector CFR (vCFR)**: A full deterministic CFR implementation with GPU kernels (vcfr.cu, ~1400 lines)
  - **External sampling MCCFR**: Multiple GPU kernel variants for stochastic CFR
  - **Flop-start game support**: Multi-street solving with chance node handling
  - **DCFR discounting**: Alpha/beta/gamma parameters
  - **Batched/streaming kernel variants**: For per-outcome processing
  - **Side pot handling**: Full multiway terminal evaluation
  - **GPU context**: ~2500 lines of Rust GPU orchestration (context.rs)

## Key Files to Port (CUDA → Metal)
1. `solver-core/src/gpu/kernels/vcfr.cu` (~1400 lines) — Core CFR kernels:
   - `vcfr_compute_strategies` — Regret matching
   - `vcfr_top_down_reach` — Reach probability propagation
   - `vcfr_bottom_up` — CFV computation + regret update
   - `vcfr_bottom_up_batched` — Batched per-outcome bottom-up
   - `vcfr_streaming_level` — Streaming per-outcome processing
   - `vcfr_chance_accumulate` / `vcfr_chance_finalize` — Chance node CFV accumulation
   - `vcfr_chance_accumulate_grouped` — Grouped chance accumulation
   - `vcfr_regret_apply` / `vcfr_cum_apply` — DCFR discount application
   - `vcfr_init_reach` / `vcfr_zero_buffer` — Buffer management
   - `sorted_sweep_showdown_vcfr` — Sorted sweep terminal evaluation

2. `solver-core/src/gpu/kernels/mccfr.cu` — MCCFR external sampling kernels
3. `solver-core/src/gpu/kernels/test_showdown.cu` — Showdown verification kernel

4. `solver-core/src/gpu/context.rs` (~2500 lines) — GPU orchestration (cudarc → metal-rs)

## Next Steps
1. Install Xcode (manual step)
2. Phase 1: Create Metal compute layer foundation
   - Replace cudarc with metal-rs
   - Create Metal device wrapper
   - Port build.rs from nvcc to metal shader compilation
   - Implement buffer allocation for unified memory
3. Phase 2: Port vcfr.cu kernels to Metal Shading Language
4. Phase 3: Per-outcome regrets implementation
