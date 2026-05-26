# MCCFR GPU Solver — North Star

## 1. Lessons from the postflop_solver CUDA Experiment

### What was built

The postflop_solver GPU backend attempted to accelerate an existing CPU CFR solver by offloading individual node operations to CUDA kernels. The architecture was:

1. **CPU walks the game tree recursively** via `solve_recursive()` (identical traversal order to the CPU path)
2. **At each node, one or more GPU kernels are dispatched**: terminal eval kernels, regret-matching kernels, slice-ops (add/mul/weighted-sum) on strategy and regret buffers
3. **GPU buffers mirror CPU storage**: `GpuStorage` holds device pointers corresponding to CPU `storage1`/`storage2`, indexed by byte offsets
4. **Per-node dispatch**: every node in the tree triggers its own set of kernel launches and `memcpy_dtod` transfers

This was implemented across ~2,500 lines of CUDA kernel code (20 sliceops kernels, 3 eval kernels) and ~2,000 lines of Rust dispatch code (`solve_gpu.rs`, `gpu_context.rs`, `slice_dispatch.rs`, `eval_dispatch.rs`, `gpu_storage.rs`, `memory.rs`).

### How it failed

The GPU solver produced **correct results** (converging to the same Nash equilibrium as CPU, within f32 precision) but was **catastrophically slow**:

| Config | CPU (ms/iter) | GPU (ms/iter) | Slowdown |
|--------|--------------|---------------|----------|
| Small (SPR=2, 1 bet) | 331 | 49,064 | **148x** |
| Medium (SPR=5, 1 bet) | 1,298 | >100,000 (est.) | **~77x** |

**Root cause**: kernel launch overhead dominates compute by orders of magnitude.

- **~3.3 million GPU operations per iteration** (1.57M kernel launches + 1.7M `memcpy_dtod` transfers) for the Small config
- Each operation processes only **~1,033 floats (~4 KB)** — far below GPU amortization threshold
- cudarc FFI overhead: **~15 microseconds per operation** (3x worse than the 5 µs planned estimate)
- Total overhead: 3.3M × 15 µs = ~49.5 seconds, matching measured 49 seconds
- Actual GPU compute per operation: **<1 microsecond** — the GPU is idle >99% of the time waiting for dispatch

### Why kernel fusion does not solve it

Fusing all per-node operations into a single kernel per node would reduce 3.3M operations to ~794,000 (one kernel per node). But:

794,000 launches × 15 µs = **11.9 seconds per iteration** — still **36x slower than CPU** (331 ms).

The fundamental problem is that a tree-traversal architecture produces work units (individual nodes) that are ~1,000 floats each. GPUs need ~100,000+ floats per kernel launch to amortize dispatch overhead. The per-node granularity is **100x too small**.

### What this implies

Any GPU-native CFR solver must:
- **Eliminate per-node host-device synchronization entirely** — the GPU must run the full iteration without calling back to the CPU
- **Batch many independent work units into single kernel launches** — thousands of nodes or trajectories processed in parallel
- **Keep all solver state GPU-resident for the entire solve loop** — CPU only sets up the problem and reads final results

---

## 2. Architectural Principles

Each principle below is paired with the failure mode it prevents.

### P1: GPU-resident state for the entire solve loop

The CPU sets up the game tree and initial state, uploads it once, then says "run N iterations." The GPU executes all iterations without host intervention. CPU reads back final strategies.

**Prevents**: The 15 µs per-operation FFI overhead that killed the previous approach. With zero host-device sync per iteration, launch overhead drops from 49 seconds to zero.

### P2: Many parallel trajectories per iteration, not one tree walk

Use External Sampling MCCFR: each GPU thread runs an independent trajectory through the game tree. Thousands of trajectories run simultaneously in a single kernel launch. The traverser explores all actions; opponents/chance are sampled.

**Prevents**: The single-thread-of-control bottleneck that forced sequential per-node dispatch. One trajectory = 1,000 floats of useless GPU work; 10,000 parallel trajectories = tens of millions of floats of useful parallel work.

### P3: Atomic regret updates, not sequential per-node updates

When 10,000 threads update regrets simultaneously, some will write to the same regret entries (same infoset reached via different sampled paths). Use GPU atomic adds (`atomicAdd` on f32) to handle concurrent writes. Small race conditions in MCCFR are acceptable — the algorithm converges in expectation regardless.

**Prevents**: The need for per-infoset locking or serialization, which would re-introduce sequential bottlenecks. Atomic updates are O(1) per write on modern GPUs and are the standard approach for parallel SGD and related algorithms.

### P4: Flat tree representation indexed by integers

The game tree is stored as flat arrays: node types in one array, children indices in another, action labels in another. No recursive Rust types (`enum Node { Terminal, Chance, Player(Box<Node>) }`), no heap pointers, no `Box< dyn >` traits in the solver loop.

**Prevents**: Pointer-chasing on the GPU (which requires 100+ cycle memory latencies per dereference). Flat arrays enable coalesced memory access patterns. Integer indices are directly usable from CUDA kernels without serialization or special handling.

### P5: N-player from day one

The solver handles N players as a first-class concept. Two-player heads-up is N=2, not a separate code path with hardcoded `[i32; 2]` bet arrays and `player ^ 1` opponent lookup. All data structures are indexed by player ID from the start.

**Prevents**: The massive refactoring required to retrofit 2-player assumptions for multiway poker (which is the primary use case — 6-max postflop). Carrying over 2-player assumptions would reproduce the worst architectural mistake of the previous solver.

---

## 3. What to Carry Over from postflop_solver

### Action tree generation — ADAPT

**Source**: `postflop-solver/src/action_tree.rs`
**Key types**: `Action` enum (`Fold`, `Check`, `Call`, `Bet(i32)`, `Raise(i32)`, `AllIn(i32)`, `Chance(Card)`), `ActionTree`, `TreeConfig`, `BetSizeOptions`

The action tree generation logic is well-tested and handles bet size enumeration, all-in capping, and round transitions correctly. It must be extended from 2-player to N-player (currently uses `PLAYER_OOP=0`/`PLAYER_IP=1` constants). The output format should change from recursive node types to a flat array representation, but the core bet-size-to-action enumeration logic transfers directly.

### Hand evaluation tables — TRANSFER DIRECTLY

**Source**: `postflop-solver/src/hand_table.rs`, `src/hand.rs`
**Key types**: `HAND_TABLE` (4,824-entry sorted `i32` array), `Hand::evaluate()` using binary search

The `HAND_TABLE` lookup mechanism is self-contained and proven correct. It can be uploaded to GPU constant memory or texture memory as-is. The binary-search evaluation function maps directly to a GPU kernel. No adaptation needed.

### Terminal evaluation with sorted-sweep showdown — PORT TO GPU

**Source**: `postflop-solver/src/game/evaluation.rs` lines 95-151 (rake=0 path)
**Algorithm**: Two-pass sorted-sweep instead of O(NH²) brute-force comparison

The existing solver pre-sorts valid hands by strength into `StrengthItem { strength: u16, index: u16 }` arrays (one per player). At showdown, two sweeps compute per-hand counterfactual values:

**Pass 1 (wins)**: Walk player strengths ascending. Maintain pointer `i` into opponent strengths. For each player hand, advance `i` past all opponents with strictly lower strength. Accumulate `cfreach_sum` and `cfreach_minus[52]` (per-card opponent reach) as we advance. For each player hand, `cfv_win = (cfreach_sum - cfreach_minus[c1] - cfreach_minus[c2]) * amount_win`. The inclusion-exclusion `cfreach_sum - card_subtracts` removes opponent hands that conflict with the player's own cards.

**Pass 2 (losses)**: Walk player strengths descending. Maintain pointer `i` into opponent strengths from the strong end. Same accumulation pattern but for strictly higher-strength opponents. `cfv_lose = (cfreach_sum - cfreach_minus[c1] - cfreach_minus[c2]) * amount_lose`.

**Result**: `cfv[hand] = cfv_win[hand] + cfv_lose[hand]`. This is O(NH) per pass, O(NH) total — no NH×NH double loop.

**Precomputation required**: For each board configuration (flop/turn/river), precompute per-player sorted strength arrays: `(strength, hand_index)` pairs sorted by strength with sentinel values at boundaries. Also precompute `same_hand_index[player][i]` — the index into opponent's hand list that has the same two cards as player's hand `i` (or u16::MAX if no match), used for tie handling.

**GPU port**: The algorithm maps directly to CUDA. Each thread maintains its own `cfreach_sum` and `cfreach_minus[52]` accumulators (52 floats = 208 bytes, fits in registers). The sorted strength arrays are read-only GPU buffers. The two-pass sweep is sequential per thread but processes all NH hands in O(NH) time instead of O(NH²). This is the critical algorithmic advantage — **PORT THIS ALGORITHM, DO NOT REINVENT IT**.

### Range handling — ADAPT

**Source**: `postflop-solver/src/range.rs`
**Key types**: `Range` struct (1,326 `f32` weights, one per two-card combo), `FromStr` parser for PioSOLVER notation

Range parsing and initialization transfers directly. Must be extended from a single `Range` to `Vec<Range>` (one per player) for N-player support. The weight accessors (`get_weight_pair`, `get_weight_offsuit`, `get_weight_suited`) work as-is.

### Validation test infrastructure — TRANSFER AS PATTERN

**Source**: `postflop-solver/src/game/tests.rs`, `src/cuda/tests.rs`, `tests/kuhn.rs`, `tests/leduc.rs`

The pattern of testing — convergence sweeps at multiple checkpoints (5, 10, 20, 50, 100, 200 iterations), exploitability measurement, GPU vs CPU ratio validation — should be replicated exactly. The existing Kuhn and Leduc test games are also ideal for initial correctness validation of the new solver since they are small enough to compute exact equilibria independently.

### JS-side integration — ADAPT

**Source**: `surstromming-solver-cli/` (the wrapper project)
**Pattern**: JSON-in/JSON-out API via wrapper-client, SQLite result caching, multiway synthesis via per-opponent HU solves

The wrapper-client integration pattern (receive game state as JSON, solve, return strategy as JSON) should be maintained. The current multiway approach (N-1 separate HU solves) will be replaced by the native N-player solver. SQLite caching transfers directly. The API surface should remain compatible.

---

## 4. What NOT to Carry Over

### Recursive `solve_recursive` structure — REJECT

The tree is walked by a recursive function that does one node at a time: compute counterfactual values for children, aggregate, update regrets. This pattern is fundamentally hostile to GPU parallelism because it serializes work that could be done independently.

### 2-player-hardcoded data structures — REJECT

`Game` trait with `player ^ 1` opponent lookup, `[i32; 2]` bet arrays, `PLAYER_OOP`/`PLAYER_IP` constants, `storage1`/`storage2` paired buffers, `OOP`/`IP` naming conventions throughout. These assumptions are embedded in dozens of locations and would require rewriting most of the codebase to support N>2. The new solver uses `Vec` indexed by player ID from the start.

### Discounted CFR (DCFR) weighting — REPLACE

The current solver uses DCFR with alpha/beta/gamma discounting parameters computed from iteration count (alpha = t^(2*sqrt(t)) / (t^(2*sqrt(t)) + 1), beta = 0.5, gamma = (t/(t+1))^3 with power-of-4 stepping). This is replaced by **Linear CFR** weighting (iteration t weighted by t) because:
1. Linear CFR is the standard companion to MCCFR (DCFR can work empirically with sampling but is not the established pairing; Linear CFR has cleaner theoretical properties under sampling)
2. Linear CFR is simpler to implement with atomic updates (multiply accumulated regrets by t at iteration boundaries)
3. Brown & Sandholm (2019) used Linear MCCFR for Pluribus blueprint training

### `GpuSolverContext` per-node dispatch — REJECT

The entire `GpuSolverContext` / `GpuStorage` / `SliceKernelCache` / `EvalKernelCache` layer was built around the per-node dispatch model. It allocates device memory for individual node buffers, maintains cached kernel function pointers for per-node launches, and uses pointer-offset arithmetic to address individual node data. This is the architecture that produced 3.3M kernel launches per iteration.

### Per-node kernel dispatch layer — REJECT

The `slice_dispatch.rs` and `eval_dispatch.rs` modules that launch individual kernels for individual nodes. Each node triggers 1-5 kernel launches and 1-10 `memcpy_dtod` operations. This is replaced by a single kernel launch per iteration that processes all trajectories in parallel.

---

## 5. Algorithm Specification

**Algorithm**: External Sampling Monte Carlo CFR with Linear Weighting (Linear MCCFR)

**Reference**: Brown & Sandholm (2019) "Superhuman AI for Multiplayer Poker" supplementary materials; Lanctot et al. (2009) "Monte Carlo Sampling for Regret Minimization in Extensive Games"; Brown & Sandholm (2019) "Solving Imperfect-Information Games via Discounted Regret Minimization" Section 3.5.

### Per-iteration procedure

1. **Select traverser**: Cycle through players. Player p is the traverser for this iteration.

2. **Launch B parallel trajectories** (B = batch size, e.g., 10,000+). Each trajectory is an independent GPU thread that:
   a. Starts at the root of the game tree
   b. At **traverser decision nodes**: explores ALL actions, computing counterfactual value for each
   c. At **opponent decision nodes**: samples ONE action from the opponent's current strategy (regret matching on positive regrets)
   d. At **chance nodes**: samples ONE outcome (deal board cards) according to the known probability distribution
   e. At **terminal nodes**: evaluates hand strength and computes payoff

3. **Compute instantaneous regret** for the traverser at each visited infoset: regret(I, a) = cfv(I, a) - cfv(I), where cfv is counterfactual value weighted by opponent/chance reach probabilities.

4. **Update regrets atomically**: `atomicAdd(&regrets[infoset_id][action], regret_value * iteration_weight)`. Concurrent writes from different trajectories hitting the same infoset are handled by GPU atomics.

5. **Update average strategy atomically**: `atomicAdd(&avg_strategy[infoset_id][action], strategy_prob * iteration_weight * reach_prob)`.

### Linear weighting

Iteration t has weight t. Implemented by multiplying accumulated regrets and average strategy by t/(t-1) at the start of iteration t (equivalent to weighting each new contribution by t without re-weighting all history). Following Pluribus: apply linear weighting for the first K iterations, then stop discounting to avoid the multiplication overhead. The supplementary materials specify Pluribus used a 400-minute window with discounting every 10 minutes (LCFR_Threshold = 400 minutes, Discount_Interval = 10 minutes). For the new solver, the cutoff should be expressed in iterations rather than wall-clock time since GPU iteration speed will differ dramatically from Pluribus's CPU training. A reasonable starting point is to apply linear weighting for the first 20-30% of planned iterations, then stop.

### Strategy computation from regrets

At each infoset, the current strategy is determined by regret matching:
- If sum of positive regrets > 0: σ(I, a) = R+(I, a) / Σ_a' R+(I, a')
- Otherwise: uniform over all actions

This is computed on-the-fly within each trajectory thread — no separate kernel needed.

### Convergence

Average strategy over all iterations converges to a Nash equilibrium in two-player zero-sum games (Zinkevich et al. 2007). For N > 2, no convergence guarantee exists, but in practice MCCFR produces strong strategies (Brown & Sandholm 2019 demonstrated superhuman play in 6-player poker).

### Terminal evaluation and side pots

The existing 2-player solver assumes a single pot split evenly between two players: `pot = starting_pot + 2 * node.amount()`, `half_pot = 0.5 * pot`. This model breaks for N > 2 where players have different stack depths and all-in situations create side pots.

**Data layout**: Every node in the flat tree stores per-player cumulative contributions. This is a separate array `contributions: Vec<i32>` indexed by `(node_index * max_players + player_id)`. At tree build time, each bet/call/raise/all-in action increments the acting player's contribution, and the value carries forward to children. This array is uploaded to GPU as a read-only buffer alongside the node array.

**VRAM cost**: `num_nodes × max_players × 4 bytes`. For a typical 3-player subgame with 2,000 action sequences and 3 players: 2,000 × 3 × 4 = 24 KB. Negligible relative to the regret/strategy buffers.

**Payoff distribution algorithm**: At terminal nodes, the kernel computes per-player payoffs as follows:

```
compute_payoffs(contributions[N], folded[N], hand_strength[N]):
    active = [p for p in 0..N if !folded[p]]
    levels = sorted unique contribution amounts among active players
    payoffs = [0; N]

    prev_level = 0
    for level in levels:
        // Pot at this level includes ALL players who contributed >= level (including folded)
        pot_this_level = (level - prev_level) * count(contributions[p] >= level for ALL p)
        // But only non-folded players with sufficient contribution contest the pot
        eligible = [p in active where contributions[p] >= level]
        winner = argmax(hand_strength[p] for p in eligible)
        payoffs[winner] += pot_this_level
        prev_level = level

    // Net payoffs: subtract each player's own contribution
    for p in 0..N:
        payoffs[p] -= contributions[p]

    return payoffs
```

This handles all cases:
- **2-player**: One pot level, both eligible, reduces to the existing `half_pot` model
- **3-player, one short all-in**: E.g., A=100, B=60, C=100 → level 60: pot = 60×3 = 180 (A,B,C contest); level 100: pot = 40×2 = 80 (A,C contest)
- **4+ players, multiple all-ins**: Each unique contribution amount creates a pot level with a shrinking eligible set
- **Same player wins multiple levels**: They accumulate (nuts wins everything they're eligible for)
- **Folded players**: Their contributions count toward the pot at the appropriate level (dead money) but they are never eligible to win

**Design decision**: Side pots are computed at terminal evaluation time, not during tree construction. The tree builder does not split nodes for side pots — it simply tracks per-player contributions and marks all-in actions. This keeps the tree structure simple and defers the complexity to the payoff calculation, which is a bounded O(N log N) operation per terminal node per trajectory.

**Tie handling**: If multiple players have equal hand strength at a pot level, the pot for that level is split equally among the tied players (standard poker rules).

**Rake**: Applied to the total pot before distribution. `total_pot = sum(contributions)`, `rake = min(total_pot × rake_rate, rake_cap)`, `net_pot = total_pot - rake`. The rake is deducted from the highest pot level (the one contested by the most invested players).

---

## 6. Open Questions

### Q1: Tree representation — flat array vs. DAG vs. adjacency list?

**Flat array**: Node i has type, children stored as index ranges. Simple, cache-friendly, easy GPU upload. But: no sharing of identical subtrees (wastes memory for large trees).

**DAG with explicit children**: Each node stores a list of child indices. Allows subtree sharing. More complex to traverse on GPU but saves memory for N-player trees where many branches are identical.

**Trade-off**: Memory usage vs. implementation complexity. N-player trees are much larger than 2-player trees (exponentially more action sequences), so subtree sharing may be necessary. However, the flat-array approach is simpler to implement and debug first.

**Pluribus reference**: The supplementary does not describe an explicit tree data structure — Pluribus allocates regret memory lazily per action sequence encountered (664M possible action sequences in blueprint, only 413M ever encountered). During real-time search, subgames have 100–2,000 action sequences (excluding chance nodes). This suggests Pluribus uses a hash-map-like sparse representation keyed by action sequence, not a pre-built tree. For the GPU solver, a flat array is a better fit (dense, coalesced access) since our subgames are small enough to pre-allocate. **Decision: flat array, pre-allocated per solve invocation.** The 100–2,000 action sequence range from Pluribus validates that subgame trees fit comfortably in memory.

### Q2: Card abstraction approach?

Pluribus used lossless abstraction on the preflop, k-means clustering into 200 buckets on each subsequent street for the blueprint, and 500 buckets per street during real-time search. The postflop_solver currently uses no card abstraction (all 1,286 hand combos tracked individually).

**Trade-off**: No abstraction = more precise but higher memory/compute. 200 buckets = 6x reduction in regret/strategy storage per infoset but introduces abstraction error. For GPU, the memory savings may be critical for fitting large N-player trees in VRAM.

**Pluribus reference**: Blueprint uses 200 buckets/street via k-means on domain-specific features (ref 26). Real-time search uses lossless abstraction on the current betting round, then 500 buckets/street for subsequent rounds, with buckets determined per-flop using potential-aware abstraction with earth-mover-distance clustering (ref 28). The supplementary states that on the second betting round there are on average 6,434 infosets per abstract bucket at 200 buckets — so each bucket represents many strategically diverse hands, and pruning partially compensates by reducing effective infosets.

**Decision for initial build**: No card abstraction (lossless, all 1,326 hand combos tracked individually per player). Rationale: (1) Pluribus's abstraction was designed for CPU training over 12,400 core-hours where memory was the bottleneck; our GPU solver with 8GB VRAM can afford lossless for subgame sizes of 100–2,000 action sequences. (2) Avoiding abstraction eliminates a major source of error and simplifies the implementation. (3) If VRAM pressure emerges, 200-bucket k-means is the fallback. Abstraction can be added later without changing the solver architecture.

### Q3: Action abstraction scheme?

Pluribus used 1-14 raise sizes (fractions of pot) depending on the decision point, with different abstraction for blueprint vs. real-time search. The postflop_solver uses configurable bet sizes via `BetSizeOptions`.

**Trade-off**: More bet sizes = larger tree = more GPU memory and compute per iteration. Fewer bet sizes = faster convergence per iteration but possibly weaker strategy. The GPU solver's ability to process many trajectories in parallel may allow finer action abstractions without wallclock penalty.

**Pluribus reference**: Blueprint uses up to 14 raise sizes on preflop (fine-grained since no real-time search there), coarser on flop. Turn and river: at most 3 raise sizes for the first raise in the round (0.5×pot, 1×pot, or all-in), at most 2 for subsequent raises (1×pot or all-in). Real-time search: 1–6 raise sizes. All sizes are fractions of pot, decided by hand based on what earlier Pluribus versions used with significant positive probability. Fold and call always included.

**Decision for initial build**: Use the configurable `BetSizeOptions` system from the existing solver (driven by `solve_config` in the JSON request). Default to a Pluribus-like abstraction for production configs: 3–5 sizes on flop, 2–3 on turn/river. The CLI already exposes `bet_sizes_flop`, `bet_sizes_turn`, `bet_sizes_river` — these map directly. No change needed.

### Q4: GPU memory layout for regret/strategy storage?

**Structure of Arrays (SoA)**: `regrets[infoset_count][max_actions]` — separate flat buffer per data type. Good coalescing when threads access the same infoset.

**Array of Structures (AoS)**: `node_data[infoset_count]` where each entry contains regrets, strategy, cfv arrays. Simpler indexing but poorer coalescing.

**Trade-off**: SoA is standard GPU best practice and enables coalesced memory access. AoS is simpler to reason about. The atomic-update pattern (many threads writing to the same infoset) may favor SoA to reduce bank conflicts.

**Pluribus reference**: No GPU usage — all CPU. Pluribus stored regrets as 4-byte integers, allocated lazily per action sequence. Memory layout was driven by CPU cache behavior, not GPU coalescing. The supplementary does not specify SoA vs. AoS.

**Decision**: SoA. This is standard GPU practice and the atomic-update access pattern (many threads targeting scattered infosets) benefits from coalesced reads of the same array. Separate flat buffers for `regrets`, `strategy`, and `cumulative_strategy`.

### Q5: Batch size per iteration?

**Trade-off**: Larger batch = better GPU utilization (more parallel work per kernel launch) but higher variance per iteration (each trajectory samples fewer unique infosets relative to total). Smaller batch = lower GPU utilization but each iteration is cheaper, allowing more iterations per unit time. Pluribus's external-sampling MCCFR processes one traverser's full walk per iteration on CPU; the GPU version should process B independent walks in parallel. Sweet spot likely in the range B = 1,000 to 100,000 depending on tree size. Note: Pluribus used Strategy_Interval = 10,000 (updating average strategy every 10,000 iterations) and Prune_Threshold = 200 minutes before activating negative-regret pruning. The agent should reference the Pluribus supplementary pseudocode (Algorithm 1) for concrete iteration-level parameters before selecting batch size and related hyperparameters.

**Pluribus reference**: External-sampling MCCFR processes one traverser per iteration on CPU. The supplementary describes two modes for real-time search: (1) Monte Carlo Linear CFR for large subgames or early game, (2) an optimized vector-based Linear CFR that samples one set of public board cards per thread (ref 42) for smaller subgames. The vector-based form is closer to what we want — multiple board samples processed in parallel per iteration. Strategy_Interval = 10,000 iterations (line 10 of Algorithm 1), Prune_Threshold = 200 minutes (line 12), Discount_Interval = 10 minutes, LCFR_Threshold = 400 minutes (line 20). These timing-based thresholds are for blueprint training and do not directly apply to our per-decision real-time mode.

**Decision**: Batch size will be tuned empirically. Starting point: B = 10,000 parallel trajectories per "iteration" (one kernel launch). Each trajectory is an independent external-sampling MCCFR walk for one traverser player, sampling opponent actions and chance outcomes. This maps naturally to GPU parallelism: each CUDA thread processes one trajectory. For a subgame with 100–2,000 action sequences, 10,000 threads should saturate the RTX 4070's 5,888 CUDA cores with enough work to amortize launch overhead.

### Q6: Convergence target and iteration count?

Pluribus blueprint was trained for 12,400 core-hours (8 days on 64 cores). The postflop_solver targets exploitability < 1 mbb/g for HU spots, typically reached in 20-200 iterations depending on config.

**Trade-off**: More iterations = lower exploitability but diminishing returns. For production use in a bot, the solver must return a strategy within seconds. The GPU's parallelism may change the convergence dynamics — more noisy iterations per unit time vs. fewer precise iterations.

**Pluribus reference**: Blueprint training: 800 minutes of Linear MCCFR (400 minutes with discounting at 10-minute intervals, then 400 more without discounting), then snapshots of current strategy taken every 200 minutes and averaged. Real-time search: the supplementary does not give a fixed iteration count — it runs until the time budget is exhausted. Two modes depending on subgame size: Monte Carlo Linear CFR for large subgames, vector-based Linear CFR for small ones. For real-time search, Pluribus uses the **final iteration's strategy** (not the weighted average), because "the final iteration's strategy is sufficiently unpredictable that any exploitation is infeasible" and it helps avoid poor actions not fully eliminated in the average strategy.

**Decision for initial build**: Run MCCFR iterations until the time budget (SC6: <10s for 3-player, <5s for 2-player). No fixed iteration count — the batch size and tree size determine iterations per second, and we run as many as the time budget allows. Exploitability is not measured during the solve (too expensive); convergence quality is validated offline in testing (SC1).

**Output strategy: weighted average (initial), final iteration (option)**. Pluribus uses the final iteration's current strategy for exploitation resistance in their specific adversarial context. For this solver, the weighted average strategy is the safer initial choice: (1) it has lower variance and represents the equilibrium more reliably, especially when running 100–1,000 iterations where early iterations have high noise and the final iteration may be heavily influenced by the most recent few traversals; (2) it is the standard CFR output with cleaner theoretical convergence guarantees. The solver should compute and return the weighted average strategy (cumulative positive-regret-weighted strategy across all iterations). If exploitation resistance becomes a concern in practice, the final iteration's strategy can be exposed as a configurable alternative without architectural change. This is a runtime option, not a design commitment.

**Pruning**: Pluribus's regret-based pruning (Algorithm 1 lines 12–15) skips traverser actions with regret below C = -300M in 95% of iterations, with exceptions for the final betting round and actions leading to terminal nodes. This reportedly contributed a ~2x speedup and effectively increased information abstraction granularity by focusing computation on frequently-reached infosets. For per-decision real-time solving with a bounded iteration budget, pruning is lower priority — the iteration counts are small enough that skipping actions saves less relative time, and the risk of incorrectly pruning a recovering action is higher with fewer total iterations. If blueprint generation is added later (Q8 future enhancement), pruning becomes important for tractable training time. The regret floor at -1e7 (Q7) is a prerequisite for pruning if it is implemented — the floor ensures pruned actions can recover.

### Q7: Regret storage precision?

Pluribus stored regrets as 4-byte integers with a floor at -310M. The postflop_solver uses f32. GPU atomicAdd operates on f32 (or i32 for integer atomics).

**Trade-off**: i32 atomics are faster and have no precision loss from concurrent updates, but require fixed-point scaling. f32 atomics have race-condition-induced precision loss but are simpler. The precision loss from concurrent atomic adds is bounded and acceptable for MCCFR (algorithm already converges in expectation under sampling noise).

**Pluribus reference**: Regrets stored as 4-byte integers with a floor at -310,000,000. This floor serves two purposes: (1) makes it easier to unprune actions that were initially pruned but later could improve, and (2) prevents integer overflow. The pruning threshold C = -300,000,000 (actions with regret below C are skipped in 95% of iterations). The integer approach avoids floating-point issues entirely.

**Decision**: f32 with atomicAdd. Pluribus's integer approach was driven by CPU concerns (memory savings: 4 bytes vs 8 bytes for double). On GPU, f32 atomicAdd is native and well-optimized; i32 atomics would require a fixed-point scaling factor and add complexity with no clear benefit since we are already using f32 for strategy computation. The regret floor from Pluribus (-310M) is a good idea and should be adopted: floor regrets at a large negative value to prevent permanently pruning actions that could recover. Initial floor: -1e7 (tunable). The concurrent-write precision loss from atomicAdd is acceptable — MCCFR converges in expectation under sampling noise anyway.

### Q8: Depth-limited search vs. full subgame solve?

The bot operates in **per-decision real-time mode**: each postflop decision triggers one solver invocation, the solver computes a strategy within the decision time budget (under 10 seconds per SC6), and returns action frequencies. There is no offline blueprint precomputation phase. The new solver implements MCCFR from Pluribus's algorithmic descriptions but uses it in this per-decision real-time context, not in Pluribus's blueprint-then-search operating model.

The initial target is **straightforward MCCFR on the current subgame** — build the tree from the current public state to the end of the hand, run MCCFR iterations until the time budget is exhausted, return the strategy. No depth limiting, no value networks at leaf nodes.

**Possible future enhancement**: If subgame trees become too large to solve within the time budget (deep multiway pots on the flop with many bet sizes), depth-limited search from Brown & Sandholm (2017) could truncate the tree and estimate leaf values. This would require implementing the augmented subgame construction and a leaf-value estimation mechanism (either blueprint-based or neural-network-based as in ReBeL). This is a significant additional component and is not needed for the initial build — the current subgame sizes in production (3-6 players, 2-3 bet sizes, flop/turn/river starting points) should fit within the time budget with GPU-accelerated MCCFR.

Separately, the MCCFR engine could be used to compute offline blueprints for common game configurations (similar to Pluribus's blueprint training). The architecture supports this without modification — blueprint training is the same algorithm run for many more iterations with output cached for repeated lookup. This is not in initial scope. The chart-based preflop system continues to be the primary preflop strategy source. If blueprinting is added later, it would coexist with charts (blueprint for spots the charts do not cover well, charts for high-confidence chart spots) rather than replacing them.

---

## 7. Success Criteria

The new solver is considered complete when all of the following are true:

### SC1: Strategic equivalence on HU spots

For 2-player configurations that the current postflop_solver handles, the new solver converges to within **5% exploitability** of the current solver's result at the same iteration count. Measured by running both solvers on identical game configurations (same tree, same ranges) and comparing exploitability via best-response computation. This validates that the MCCFR implementation is correct despite sampling noise and atomic-update race conditions. Note: the 5% tolerance should be calibrated against the current solver's actual convergence behavior on each test configuration — some configs may converge tighter than 5% naturally, and the tolerance should not be used as an excuse for incorrectness on easy configs.

### SC2: Multiway capability without per-opponent workarounds

The solver natively handles 3-6 player decision points. A single solve call with N ranges produces a strategy for all players simultaneously. No N-1 separate heads-up solves, no "synthesize multiway from HU results" post-processing. The strategy quality in 3-player spots should be at least as good as the current synthesized approach (which is a low bar — the synthesis approach has known weaknesses around opponent interactions).

### SC3: 5x+ speedup over current postflop_solver

On representative configurations (HU and 3+ player), the GPU solver is at least **5x faster** than the current CPU solver in wall-clock time to reach the same exploitability target. This is the minimum speedup that justifies the GPU infrastructure. The speedup should primarily come from GPU parallelism actually working (many trajectories per iteration), not from algorithmic shortcuts. Measured on an RTX 4070 Laptop (8 GB VRAM) as the minimum target GPU.

### SC4: Integration with wrapper-client API

The solver is callable from the existing `surstromming-solver-cli` JSON API. The request/response format is compatible (or backwards-compatible with documented changes). Existing bot infrastructure that calls `wrapper-client.js` works without modification or with minimal documented changes.

### SC5: Memory fits in consumer GPU VRAM

The solver runs within 8 GB of VRAM for typical production configurations (3-6 players, 2-3 bet sizes, no card abstraction on the current street). This is the constraint imposed by the target hardware. Memory usage should be profiled and documented for key configurations.

### SC6: Per-decision solve time

The solver's primary use case is **real-time per-decision solving** during bot gameplay, not offline blueprint generation. A single solve call (tree construction + N iterations + strategy extraction) must complete within the bot's decision timeout. The existing CLI wrapper uses a 30-second timeout. The new solver should aim for **under 10 seconds** for typical 3-player spots and **under 5 seconds** for 2-player spots on the target GPU. This is the wall-clock time the bot experiences per decision, including any CPU-side setup and GPU data transfer overhead.

---

## 8. Integration with Existing Bot Infrastructure

### How the solver gets called

The bot calls the solver via `surstromming-solver-cli`, a Rust binary invoked as:

```
surstromming-solver-cli --input request.json --output result.json --config-dir ./config
```

The binary reads a JSON request, constructs ranges, builds the action tree, runs the CFR solver, and writes a JSON response. SQLite caching (`surstromming-solver-cache.db`) avoids re-solving identical spots.

The new GPU solver replaces this binary. The CLI interface (stdin/stdout JSON, `--input`, `--output`, `--cache-db`, `--config-dir`, `--timeout-seconds` flags) should remain identical.

### JSON request format

The current `SolveRequest` schema (defined in `surstromming-solver-cli/src/config.rs`) accepts:

```json
{
  "schema_version": 1,
  "request_id": "unique-id",
  "stake": "NL100",
  "hero": {
    "position": "BTN",
    "hole_cards": ["Ah", "Kh"],
    "blocker_cards": ["Ah", "Kh"],
    "stack": 10000
  },
  "opponents": [
    { "seat_id": "BB", "position": "BB", "stack": 10000, "vpip": 22, "pfr": 18 }
  ],
  "board": { "flop": ["2s", "7d", "Jc"], "turn": "Td", "river": null },
  "pot": { "starting_pot": 100, "current_pot": 200, "rake_rate": 0.0, "rake_cap": 0 },
  "preflop_action_sequence": [...],
  "postflop_action_sequence": [...],
  "solve_config": {
    "max_iterations": 200,
    "exploitability_target_fraction": 0.005,
    "bet_sizes_flop": ["0.5pot"],
    "bet_sizes_turn": ["0.5pot", "1pot"],
    "bet_sizes_river": ["0.5pot", "1pot"],
    "raise_sizes": ["1pot"],
    "compression": false,
    "synthesis_mode": "SequentialHeadsUp"
  }
}
```

This format must be accepted by the new solver with one change: `synthesis_mode` becomes unnecessary for the native N-player solver (always `null` or removed). The new solver ignores it and solves N-player directly.

### How opponent modeling feeds in

Opponent stats (VPIP, PFR, etc.) arrive in the `opponents` array. The CLI wrapper's `range::construct_opponent_range()` function maps these stats to a `Range` via:
1. Pool average data loaded from `config/pool_averages/` (stake-specific population tendencies)
2. Archetype thresholds loaded from `config/archetype_thresholds/` (classifies opponent into archetype like TAG, LAG, etc.)
3. Range construction from archetype + preflop action sequence + postflop action sequence

The new solver consumes the **output** of this pipeline (a `Range` per opponent), not the raw stats. The range construction pipeline is independent of the solver and transfers as-is. The new solver receives `Vec<Range>` (one per player) and a flat game tree, same as the current solver receives a single opponent `Range`.

### How action history is handled

The current solver accepts `postflop_action_sequence` — an array of `{seat_id, action, amount}` entries describing what happened before the hero's decision point. The solver walks this sequence to navigate to the correct node in the game tree (see `solver.rs` lines 181-264: iterating through actions, matching them to `game.available_actions()`, calling `game.play(idx)` for each).

For the new solver, action history serves a different purpose: it determines **which node in the flat tree is the root of the subgame to solve**. The action history must be consumed to compute the correct starting node index. Pluribus handles this via its belief distribution update (tracking probability over private cards given observed actions). The new solver should:
1. Parse the action history to identify the current public state
2. Map that public state to the correct subtree root in the flat tree
3. Solve only the remaining subtree (not the full game tree from the preflop root)

This is more correct than the current approach (which builds the full tree then navigates to a node) and is how Pluribus actually operates.

The `preflop_action_sequence` is consumed to determine the starting ranges at the flop (which players were involved, which actions they took preflop) and the starting pot/stack state. It does not affect the postflop game tree structure since the solver builds the tree from the flop forward.

### Preflop scope

The bot's preflop strategy comes from the **chart-based system** (GTO Wizard ranges keyed by stake, stack depth, position, and action context). This is existing infrastructure that the new solver does not replace.

The new solver receives postflop game states (flop already dealt) and computes postflop strategy starting from the flop. Preflop action sequence is used to determine the starting ranges and pot/stack state at the flop, but the solver does not compute preflop strategy.

For spots without clean chart matches, the wrapper-client may invoke the solver on preflop subgames as a fallback. This path continues to work with the new solver since the underlying MCCFR algorithm handles any subgame regardless of street. But the primary preflop strategy source remains the charts. This scoping prevents the new solver from expanding into preflop territory unnecessarily and preserves the chart investment.

### Coexistence with old postflop_solver

**Decision: the new solver replaces the old one entirely.** Rationale:
- The old solver's 2-player code path is a special case of N=2 in the new solver
- Maintaining two codepaths doubles maintenance burden and creates divergence risk
- The old solver's multiway synthesis (`synthesize_sequential`, `synthesize_collapse`) is eliminated by native N-player solving
- HU spots solved by the new solver must match the old solver's quality (SC1), so there is no quality regression

If SC1 cannot be met on HU spots initially, a temporary fallback to the old solver for `opponents.len() == 1` is acceptable during development. But the goal is full replacement.

### What changes for the bot operator

From the bot operator's perspective, the change should be transparent:
- Same CLI invocation (`surstromming-solver-cli --input ... --output ...`)
- Same JSON request format (with `synthesis_mode` now ignored)
- Same JSON response format (action frequencies for the hero's hand)
- New fields in response may be added (e.g., `gpu_solve_metadata` with VRAM usage, batch size, iterations completed)
- The `diagnostics.synthesis_path` field will be empty for the native solver (no synthesis steps)
- Solve time should decrease significantly (SC3, SC6)

### Cache compatibility at cutover

The existing SQLite cache (`surstromming-solver-cache.db`) contains entries from the old solver. At cutover, the cache should **not** be used directly — the new solver's output may differ from the old solver in subtle ways (different bet size mixing due to MCCFR sampling noise, different action enumeration order for N-player trees, different exploitability values) making direct cache compatibility unsafe. The recommended approach is to add a `solver_version` field to cache keys so old and new solver entries coexist without collision. Old entries age out naturally via cache eviction. This avoids a flag-day cutover where the cache must be wiped simultaneously with the solver upgrade.
