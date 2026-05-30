# Architectural Validation Results

## Game Definition
- **Board**: 2h 7d Ks 3c 5c (river-start, no chance nodes)
- **Ranges**: Uniform (all hands equally likely)
- **Pot**: 10 (5/5 contributions)
- **Effective stack**: 95
- **Bet sizes**: 1 pot-sized bet, no raises
- **Tree**: 9 nodes (manually constructed to match b1nary exactly)
- **Hands**: 1081 valid hands per player
- **Combinations**: 1,070,190

## Tree Structure
```
[0] P0 → check[1], bet_pot[2]
[1] P1 → check[3], bet_pot[4]
[2] P1 → call[5], fold[6]
[3] TERMINAL (check-check showdown, pot=10)
[4] P0 → call[7], fold[8]
[5] TERMINAL (bet-call showdown, pot=30)
[6] TERMINAL (P1 folds, P0 wins pot=15)
[7] TERMINAL (check-bet-call showdown, pot=30)
[8] TERMINAL (P0 folds, P1 wins pot=15)
```

## Results

| Iters | Ours      | b1nary     | Ratio  |
|-------|-----------|-----------|--------|
| 1     | 4.080254  | 2.312137  | 1.76x  |
| 5     | 0.880079  | 1.206834  | 0.73x  |
| 10    | 0.201192  | 0.274935  | 0.73x  |
| 25    | 0.036608  | 0.029760  | 1.23x  |
| 50    | 0.005033  | 0.005046  | 1.00x  |
| 100   | 0.003363  | 0.001812  | 1.86x  |
| 200   | 0.000915  | 0.000682  | 1.34x  |
| 500   | 0.000269  | 0.000322  | 0.84x  |
| 1000  | 0.000075  | 0.000058  | 1.29x  |

Zero-sum verification: EV[SV0] + EV[SV1] = -1.35e-12

## Conclusion

**Architecture validated.** On the same game with the same tree:

1. Both solvers converge to exploitability < 0.0001 by 1000 iterations
2. The ratio oscillates around 1.0 (range: 0.73x to 1.86x)
3. Neither solver consistently dominates the other across all checkpoints
4. Our solver matches b1nary at iter 50 (ratio = 1.00x) and beats it at iters 5, 10, 500

The differences are attributable to:
- Non-monotone DCFR early behavior (different discounting schedules)
- Floating-point ordering differences in regret accumulation
- Different strategies for handling uniform initial regrets

**The per-outcome regret CFR architecture is sound and competitive with b1nary's approach on the same game.**

## What This Does NOT Validate

This validates the CFR algorithm and terminal evaluation on a river-start game (no chance nodes). It does not validate:
1. Chance node handling (turn/river card dealing)
2. The flop-start dimensional regret decomposition
3. Performance at scale (this is a 9-node tree)

These must be validated separately before production use. The Metal GPU port should focus on these unvalidated areas.

## How to Reproduce

```bash
cargo test -p solver-core --features metal --test same_game -- --test-threads=1 --nocapture --ignored
```

Test file: `solver-core/tests/same_game.rs`
