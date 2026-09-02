# Surströmming Solver

A from-scratch, production-grade **6-max No-Limit Hold'em poker solver** in Rust with custom **Metal GPU kernels** — blueprint solving via CFR variants, plus a real-time depth-limited search server.

Built to run entirely on one Apple Silicon machine (M4 Max, unified memory): ~130k LOC across four workspace crates, GPU-accelerated solving and search, serving decisions over HTTP.

## Architecture

```
PREFLOP:  standalone EQR chart generator
            (equity-realization terminal, decoupled from postflop)
              ↓ entering ranges
SEAM:     SeamCell (live count, commits, pot) → SPR bin
              ↓
POSTFLOP: bucketed DCFR blueprint
            (1,755 canonical flops, GS14-style card abstraction)
              + depth-limited real-time search at decision time
```

The postflop blueprint is reach-independent by design: entering ranges are
passed at query time, not baked in, and the search boundary freezes to the
blueprint's continuation values (Pluribus-style). The preflop side pivoted to
a decoupled equity-realization terminal after the coupled co-solve proved
unnecessary — real-time search supplies the range-dependence for free.

## Workspace

| Crate | Role |
|---|---|
| `solver-core` | CFR/DCFR/MCCFR engines, tree builder, card abstraction, showdown/rake, Metal GPU kernels (`src/gpu_metal/`) |
| `solver-cli` | Fill/blueprint/solve entry points |
| `bot-server` | Axum HTTP decision server (`/decide`), 1 request per decision point |
| `play-harness` | Client library, runtime API, self-play and gate tests |

## GPU acceleration

Four hand-written Metal shaders (`.metal` sources under
`solver-core/src/gpu_metal/shaders/`): level-parallel blueprint iteration,
bucketed terminals, MCCFR benchmarking, flat-tree utilities. The CPU
reference is the source of truth; every kernel ships with a parity gate
(bit-exact or bounded-error, per path). Eighteen GPU bugs were found and
fixed this way — the gate discipline is documented in
`solver-core/src/gpu_metal/INVENTORY_FINDING.md`.

Notable results:
- Live-5 solving fully on GPU (subgame rooting + k23 cluster-mass kernels)
- Off-grid turn decisions: 28s → 0.8s via warm-started flop sweeps
- Terminal-sampling kernels: unbiased at B=8 vs exhaustive (0.32% mean error
  at M=8000, clean 1/√M scaling), unlocking B=32 bucket counts

## Running

```bash
# decision server (GPU search)
CONN_BP=blueprint_conn_v5 CONN_GS14=gs14_blueprint_cache GPU_SEARCH=1 \
  cargo run --release -p bot-server --features metal

# test suite incl. parity gates
cargo test --release --workspace
```

Blueprint artifacts (multi-GB, regenerable) are not committed; see
`.gitignore` for the generation entry points.

## Validation

The repo's STATUS docs (STATUS.md, HONEST_STATUS.md, ARCHITECTURAL_STATUS.md)
track convergence and parity evidence, including failures. The project's
core discipline: probe before solve, gate before CFR, sanity-check suspicious
numbers before building on them.
