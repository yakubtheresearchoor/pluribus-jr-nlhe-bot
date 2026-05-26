# Implementation Notes

Tactical guidance per phase. Read NORTH_STAR.md first for architectural context.

## Phase 2: N-Player Action Tree Builder

### Tree builder defensive patterns

- Validate `num_players` is 2-10 at construction time
- Validate `starting_stacks.len() == num_players`
- Validate `blinds.len() == num_players` (non-blind positions have blind = 0)
- Validate `positions.len() == num_players` and positions are unique
- Reject negative stacks or blinds
- Reject starting_pot <= 0

### Side pot data structure

The contributions array is populated during tree construction:
- At the root: `contributions[root * N + p] = blinds[p]` for each player p
- On bet/raise/call: increment acting player's contribution by the amount added, carry forward to children
- On fold: carry forward parent's contributions unchanged for the folded player
- On chance node: all contributions carry forward unchanged

Example (3-player, starting_pot = 150 from blinds SB=50, BB=100):
- Root contributions: [50, 100, 0] (BTN/SB/BB)
- BTN calls 100: [100, 100, 0]
- BB raises to 300: [100, 300, 0]
- BTN calls 300: [300, 300, 0]
- BB checks: round ends, chance node with contributions [300, 300, 0]

### Round completion logic

A betting round ends when:
1. All non-folded players have acted at least once this round
2. All non-folded players have equal contributions (or are all-in)

Track per-round state: `has_acted: Vec<bool>` (N elements), reset at each new betting round. `round_complete = has_acted.iter().all(|a| *a) && all_equal_contributions_except_all_in()`

### Acting order

Postflop acting order starts from the first active player left of the dealer (BTN) and proceeds clockwise. For 6-max with positions [BTN, SB, BB, UTG, MP, CO], postflop order is BTN → SB → BB → UTG → MP → CO.

Preflop: starts from UTG (player left of BB), proceeds clockwise. BB acts last preflop.

After a bet/raise: action continues from the next active player clockwise, skipping folded players. Round ends when action returns to the bettor/raiser and all others have called or folded.

## Phase 4: Full Poker Hand Evaluation

### Side pot test suite

Dedicated test module. Each test constructs contributions/folded/hand_strength arrays directly and asserts payoff values. All tests verify `sum(payoffs) == 0` as a sanity check.

**Test 1: Two-player equal stacks, both all-in**
- contributions: [100, 100], folded: [false, false]
- P0 wins: payoffs = [+100, -100]
- P1 wins: payoffs = [-100, +100]
- Tie: payoffs = [0, 0]

**Test 2: Three-player, one short all-in**
- contributions: [100, 60, 100], folded: [false, false, false]
- Levels: 60 (pot = 60×3 = 180, eligible all 3), 100 (pot = 40×2 = 80, eligible P0+P2)
- P0 best overall: wins 180+80=260, net = [+160, -60, -100]. Sum = 0.
- P1 best hand: wins 180, net = [+120 from main pot]. Side pot 80: if P0>P2, P0 gets 80, net P0 = [+80-100, 180-60, -100] = [-20, +120, -100]. Sum = 0.
- P2 best overall: wins 180+80=260, net = [-100, -60, +160]. Sum = 0.

**Test 3: Four-player, two different all-in amounts**
- contributions: [100, 40, 70, 100], folded: [false, false, false, false]
- Levels: 40 (pot=40×4=160, all eligible), 70 (pot=30×3=90, P0+P2+P3 eligible), 100 (pot=30×2=60, P0+P3 eligible)
- P1 has nuts: wins level 40 (160). Level 70: max(P0,P2,P3) wins 90. Level 100: max(P0,P3) wins 60.
- If D has second-best: P1 gets 160-40=+120. Level 70 → P3 wins 90. Level 100 → P3 wins 60. P3 gets 90+60-100=+50. P0: -100. P2: -70. Sum = 120+50-100-70 = 0.

**Test 4: Five-player, one folds (dead money)**
- contributions: [100, 100, 100, 80, 100], folded: [false, false, false, true, false]
- P3 folded after contributing 80. Active: {P0,P1,P2,P4}.
- Levels: 80 (ALL 5 have contribution >= 80, pot=80×5=400, eligible {P0,P1,P2,P4}), 100 (P0,P1,P2,P4 have contribution >= 100, pot=20×4=80, eligible {P0,P1,P2,P4})
- Total pot = 400+80 = 480. Sum contributions = 100+100+100+80+100 = 480.
- P0 wins all: 400+80-100 = +380. P1: -100. P2: -100. P3: -80 (dead money). P4: -100. Sum = 380-100-100-80-100 = 0.

This test validates the critical fix: folded players' contributions count toward the pot size at each level but they are never in the eligible set.

**Test 5: Tie at a pot level**
- contributions: [100, 100], folded: [false, false], hand strengths equal
- Pot = 200. Split: each gets 100. Net: 100-100 = 0 each.

## Phase 5: CLI Integration

### Action history edge cases

1. **Empty action sequence**: Subtree root is node 0 (root of the flat tree). Return immediately, no walking.

2. **Action not in tree**: Two-tier matching, carried over from existing `solver.rs:204-235`:
   - First: exact match within ±0.5 on bet/raise amount
   - Fallback: closest match with `eprintln!` warning
   - Final fallback: match to AllIn if available
   - No match: return error listing available actions

3. **Unknown player**: `seat_id` must map to a player index. If not found, return error immediately. Defensive check that should never fire in practice.

4. **Chance node traversal between actions**: Between action entries, auto-advance through chance nodes (deal turn/river from board config). If chance node expects a card that isn't provided (turn needed but `board.turn` is null), return error with diagnostic (street number, which cards are available). Pattern from existing `solver.rs:182-195`.

5. **Terminal reached before sequence exhausted**: Stop walking. Remaining actions are irrelevant (hand ended). Pattern from existing `solver.rs:196-198`.

### JSON parsing

- `synthesis_mode` field: accept and ignore (always null for new solver). Do not error if present.
- `opponents` array: 1-8 entries, same validation as existing `config.rs:247-252`.
- Board validation: flop requires exactly 3 cards, turn requires flop, river requires turn. Same as existing.
- Card uniqueness: hero hole cards, board cards, and blocker cards must all be unique. Same as existing.

### Cache key generation

- Key inputs: board cards + player ranges (serialized) + bet sizes + action history + solver_version
- Use SHA-256 hash of concatenated inputs (same as existing SHA-256 approach)
- `solver_version` field: start at `"mccfr-gpu-v1"`. Increment on any change that could affect output.
- Old cache entries with no `solver_version` field or different version are never matched — they age out via eviction.

## Phase 6: HU Equivalence Testing

### Best-response verification methodology

1. Build identical game config on both old and new solver (same board, same ranges, same bet sizes).
2. Run old solver for 200 iterations. Measure exploitability via best-response computation.
3. Run new solver for same iteration count. Measure exploitability via same best-response computation.
4. Compare: `new_exploitability <= 1.05 × old_exploitability`.
5. Repeat across 3-5 configs spanning easy (SPR=2, 1 bet) to hard (SPR=10, 3 bets).

Best-response computation:
- For each player p, compute the best response to the other player's fixed strategy.
- Walk the game tree: at opponent nodes, follow their fixed strategy. At player p's nodes, choose the action maximizing expected value.
- Exploitability = (best_response_P0_value + best_response_P1_value) / 2.
- For the new solver: extract the weighted average strategy, then run best-response against it on the flat tree.
- Adapt from existing `game/tests.rs` best-response pattern.
