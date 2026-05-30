# Bug #3: Side Pot Showdown CFV Formula

## Status: FIXED

## Discovery
Found via zero-sum property check on minimal tree (4 hands, 2 turn cards, 2 river cards).

`EV[SV,P0] + EV[SV,P1] = -15.57 ≠ 0` — strategy values violated zero-sum.

## Root Cause
`side_pot_showdown_cfv()` in `showdown.rs` computed incorrect CFVs for **2-player showdowns with unequal contributions** (side pot situations).

The formula was:
```
cfv[h] = pot_at_level * sweep[h] - investment
```

But the correct formula for 2-player showdowns is:
```
cfv[h] = (starting_pot/np + min_active_contrib) * sweep[h]
```

### Why the old formula was wrong
In a 2-player side pot:
- Main pot = starting_pot + 2 × min(c0, c1) — both players contest this
- Side pot = |c0 - c1| — returned to the larger contributor regardless

Each player's **at-risk amount** is `starting_pot/np + min(c0, c1)`, regardless of their actual contribution. The excess is never at risk.

The old code used `pot_at_level` (= main pot = 2 × at_risk) as the payoff multiplier, then subtracted investment once globally. This over-counted because:
1. It multiplied wins/losses by the full pot instead of the at-risk amount
2. It subtracted investment once instead of per-opponent-hand

### Specific example (node 18: contributions=[5,95], starting_pot=10)
- P0 invests 10, P1 invests 100
- At-risk per player = 5 + min(5,95) = 10
- P0 h0 beats 3 opponents: correct cfv = 10 × 3 = 30
- Old code: 20 × 3 - 10 = 50 (wrong by +20 = pot_at_level)

## Fix
Added a dedicated 2-player showdown path in `side_pot_showdown_cfv()` that uses the correct formula:
```rust
if np == 2 && fold_mask & (1u16 << traverser) == 0 {
    let half_pot = starting_pot as f32 / np as f32 + min_active_contrib as f32;
    let sweep = sorted_sweep_showdown(...);
    for h in 0..nh { cfv[h] = half_pot * sweep[h]; }
    return cfv;
}
```

This correctly handles both equal and unequal contributions. For equal contributions, it's equivalent to the existing path. For unequal, it uses `min_active_contrib` instead of the traverser's contribution.

## Verification
- **Zero-sum at terminals**: All showdown terminals now have `sum_P0 + sum_P1 = 0.0000`
- **Zero-sum at root**: `EV[SV,P0] + EV[SV,P1] = 0.000` across all iterations ✓
- **All existing tests pass**: 10/10 vector_cfr_test, 4/4 metal_vcfr_convergence ✓
- **Convergence**: Minimal tree still converges (46.5 → 1.74, 27× reduction)

## Files Changed
- `solver-core/src/solver/showdown.rs`: Fixed `side_pot_showdown_cfv()` for 2-player case
- `solver-core/src/solver/flop_start_game.rs`: Added opp_reach filtering for board cards (turn/river)
- `solver-core/src/solver/flop_start_vector_cfr.rs`: Added opp_reach filtering in `bottom_up_zone()`
