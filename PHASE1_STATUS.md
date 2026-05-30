# Phase 1: Metal Compute Layer Foundation — COMPLETE

## What Was Built

### 1. Metal Shader Build System (`build.rs`)
- Compiles `.metal` shaders to `.air` then links into `solver.metallib`
- Uses `xcrun -sdk macosx metal` compiler (requires Xcode + Metal Toolchain)
- Supports `#include` via `-I` flag
- Activates only when `--features metal` is set
- Coexists with CUDA build system (which activates on `--features cuda`)

### 2. Metal GPU Module (`src/gpu_metal/`)
- **`buffer.rs`** — `MetalBuffer<T>`: typed buffer wrapper for unified memory
  - `zeros()` — allocate zero-initialized buffer
  - `from_slice()` — allocate and upload data
  - `to_vec()` — download to Vec
  - `as_slice()` / `as_mut_slice()` — direct CPU access to unified memory
  - Uses `StorageModeShared` (zero-copy on Apple Silicon)

- **`context.rs`** — `MetalContext`: device wrapper
  - Device + queue + shader library management
  - `create_pipeline()` — create compute pipeline from kernel name
  - `alloc_zeros()` / `upload()` — buffer helpers
  - `dispatch_1d()` / `dispatch_2d()` — compute grid sizing
  - `new_command_buffer()` — Metal command buffer creation

- **`mod.rs`** — module exports

### 3. Metal Shaders (`src/gpu_metal/shaders/`)
- **`flat_tree.metal`** — Shared type definitions matching Rust's `FlatNode` (20 bytes, repr(C))
  - `FlatNode` struct with explicit padding
  - Node type constants (`NODE_TYPE_TERMINAL`, etc.)
  - `MAX_NA = 4` constant

- **`vcfr.metal`** — Core CFR kernels (4 ported so far):
  - `vcfr_compute_strategies` — Regret matching (2D dispatch)
  - `vcfr_top_down_reach` — Reach probability propagation (2D dispatch)
  - `vcfr_init_reach` — Initialize reach buffer (1D dispatch)
  - `vcfr_zero_buffer` — Zero a buffer (1D dispatch)
  - Each kernel uses a `constant` struct for parameters (Metal best practice)

### 4. Validation Tests (`tests/metal_foundation.rs`)
5 tests, all passing:
- `test_metal_context_creation` — Device, queue, pipeline creation
- `test_metal_buffer_allocation` — Buffer alloc, upload, download, 1M element roundtrip
- `test_metal_vcfr_compute_strategies` — Full kernel execution with known-good values
- `test_metal_zero_buffer_kernel` — Zero-fill kernel
- `test_metal_init_reach_kernel` — Reach initialization kernel

### Key Design Decisions
1. **`MetalBuffer<T>` wraps `metal::Buffer`** with typed access — similar to `CudaSlice<T>` but leverages unified memory for zero-copy CPU access
2. **Parameter structs** — Metal kernels receive scalars via `constant Struct&` rather than individual buffers, which is the idiomatic Metal pattern
3. **`StorageModeShared`** — On Apple Silicon unified memory, GPU and CPU share the same physical memory. No explicit sync needed for coherent access.
4. **Separate feature flag** — `metal` feature coexists with `cuda` feature; CPU code works with neither enabled
5. **Include-based shader organization** — `vcfr.metal` includes `flat_tree.metal` for shared type definitions

### Build System
```bash
cargo build --features metal          # CPU + Metal
cargo build --features cuda           # CPU + CUDA (Windows/Linux)
cargo build                            # CPU only
cargo test --features metal            # Run all tests including Metal
```

### Remaining for Phase 2 (Port All Kernels)
The following kernels from `vcfr.cu` still need to be ported:
- `vcfr_bottom_up` — CFV computation + regret update (the main workhorse)
- `vcfr_bottom_up_batched` — Batched per-outcome bottom-up
- `vcfr_streaming_level` — Streaming per-outcome processing
- `vcfr_chance_accumulate` / `vcfr_chance_finalize` — Chance node CFV
- `vcfr_chance_accumulate_grouped` — Grouped chance accumulation
- `vcfr_regret_apply` / `vcfr_cum_apply` — DCFR discount application
- `sorted_sweep_showdown_vcfr` — Terminal evaluation helper

The `GpuVectorCfr` Rust orchestration (`gpu/context.rs`, ~2500 lines) needs to be ported from `cudarc` to the new `MetalContext`/`MetalBuffer` API.
