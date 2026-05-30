# Honest Status Report

## Phase 3: Per-Outcome Regret Architecture — BUG FOUND

### Architecture: Implemented, Compiles, Runs

Dimensional per-outcome regrets, 7.2 GB total memory, fits in 36 GB unified memory.
CPU reference produces valid CFV values (no NaN/Inf).

### Convergence Test: FAILED

| Iter | Exploitability | Status |
|------|---------------|--------|
| 0 | 75.06 | Initial (uniform strategy) |
| 1 | 75.06 | No change (expected for iter 1 with DCFR) |
| 2 | **15.72** | Good drop |
| 3 | **85.98** | **REGRESSION — exploitability increased 5.5x** |
| 4 | 68.27 | Partially recovered |
| 5 | 64.93 | Slowly decreasing |

**Exploitability should decrease monotonically (within DCFR gamma resets).**
The jump from 15.7 to 86.0 at iteration 3 is a bug.

### Possible Root Causes

1. **DCFR gamma calculation bug**: The `nearest_lower_power_of_4` formula gives 4 instead of 1 for iterations 2-3, causing `t_gamma` to wrap. This is present in b1nary's code too, so it's unlikely to be the sole cause (b1nary converges despite it).

2. **Per-outcome strategy/reach mismatch**: The reach computation for turn zone uses per-tc strategy, but the bottom-up CFV computation might be mixing outcomes. If CFVs from river outcome A's processing are read during river outcome B's processing, regrets would be corrupted.

3. **Zone classification error**: If a node is misclassified (e.g., a turn-zone node treated as river-zone), its regrets would be written to the wrong slot.

4. **CFV seeding bug**: After river zone bottom-up, CFVs are accumulated into `river_cfv_accum` and then copied to `turn_cfv`. If this copying is wrong (wrong child IDs, wrong offset), the turn zone gets garbage CFVs.

### What to Debug Next

1. **Single-node trace**: Pick one decision node in the turn zone. Print its regrets before and after each traverser's update for each turn card. Verify the regret values change correctly.

2. **CFV sanity check**: At iteration 2 (where exploitability dropped to 15.7), print the CFVs at key nodes. At iteration 3, print them again. The jump should be traceable to a specific node.

3. **Disable DCFR**: Run with alpha=1, beta=1, gamma=1 (no discounting). If exploitability still regresses, the bug is in the core CFV/regret logic, not in the DCFR discounting.

4. **Reduce outcomes**: Test with a tiny deck (e.g., 2 turn cards, 2 river cards) to make manual tracing tractable.

### Known DCFR Parameter Bug (Cosmetic)

The `nearest_lower_power_of_4` formula gives 4 for iterations 2 and 3 instead of 1.
This causes `t_gamma` to underflow. Present in b1nary's code too — likely benign
because `gamma ≈ 1.0` for these iterations (no discounting).

### Comparison with b1nary (Different Trees)

b1nary's iter-0 exploitability: 13.6 (vs our 75.1). Trees have different structures.
Cannot compare convergence directly until both solvers run on the same tree.
