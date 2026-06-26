# Runtime Decision API

The `bot-server` crate exposes the poker bot as an HTTP service. The live game
(the runtime that tracks tables, seats, chips, and betting) sends one request per
decision point and gets back an action distribution plus a sampled choice. The bot
itself holds no game state — every request carries the derived per-decision state.

- Crate: `bot-server` (axum + tokio)
- Decision logic: `play_harness::api`
- Launch: `BP_ROOT=$PWD/blueprint_out_v1 PF_STRAT=preflop_eqr_bbfix PAR=1 cargo run --release -p bot-server`

---

## 1. Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/`       | Health check (returns a plain-text banner). |
| `POST` | `/decide` | One decision. Body = `DecideRequest` (JSON). Returns `DecideResponse` (JSON). |

Errors are returned as HTTP status + a plain-text reason:
- `400` — malformed state (bad routing, node not the hero's, blocked board).
- `422` — a decision the current build cannot serve (e.g. an SPR bin not banked).
- `500` — internal error.
- `503` — preflop requested but no preflop strategy is loaded (`PF_STRAT` unset).

---

## 2. Card & action encoding

**Cards** are `0..=51`: `card = rank * 4 + suit`, where `rank ∈ 0..=12` (0 = deuce,
12 = ace) and `suit ∈ 0..=3`. Example: A♠ = `12*4 + 0 = 48`.

**Chips** are in **units**, where `1 bb = 2 units` (`UNITS_PER_BB = 2`). So a 100 bb
stack = 200 units, a 0.5 bb small blind = 1 unit. All chip fields (`commit_entry`,
`pot_entry`, `to_total`, `to_call`, response `amount`) are in units.

**Action labels** (stable, used in both directions):

| label | action | meaning |
|-------|--------|---------|
| 0 | fold  | give up the hand |
| 1 | check | pass, no wager (only when not facing a bet) |
| 2 | call  | match the current wager |
| 3 | bet   | first wager on a street |
| 4 | raise | re-raise a bet |
| 5 | allin | shove the remaining stack |

---

## 3. Request schema — `DecideRequest`

The runtime tracks the betting engine; the bot does not. So the request carries the
**derived subgame state** for this one decision. Postflop-only fields may be omitted
on a preflop (empty-board) request.

| field | type | required | meaning |
|-------|------|----------|---------|
| `board` | `[u8]` | yes | Community cards. `[]` = preflop, 3 = flop, 4 = turn, 5 = river. |
| `hero_cards` | `[u8; 2]` | yes | The bot's two hole cards. |
| `live` | `u8` | yes | Players still in the hand (2–6). Selects the decision path (§5). |
| `hero_idx` | `u8` | yes | The hero's index **among the live set** (`0..live`), in action order. Must equal the acting player at the replayed node, or the server returns 400. |
| `commit_entry` | `u32` | postflop | Each live player's matched commitment (units) **entering** this street. |
| `pot_entry` | `u32` | postflop | Total pot (units) entering this street. Must be `≥ live × commit_entry` (dead money is the excess). |
| `street_actions` | `[ActionInput]` | postflop | The ordered betting actions **this street**, replayed to find the hero's node. Empty = hero is first to act. |
| `partner_cards` | `[u8; 2]?` | optional | **Pair mode** (§7): a colluding partner's hole cards, blocked from the pool's range. |
| `partner_idx` | `u8?` | optional | The partner's index among the live set. |
| `to_call` | `u32?` | live-6 | Amount (units) the hero must put in to call. Only the live-6 rollout path uses it; falls back to the max `to_total` in `street_actions`. |
| `cell_dir` | `string` | live-3/4/5 | Blueprint cell directory (§6.2), e.g. `live3_c6_p20_b15`. Omit when `route` derives it, or rely on the runtime's own cell selection. |
| `flop_id` | `u32` | live-3/4/5 | Canonical flop id (`0..1754`). Derived by the server when `route = true`. |
| `route` | `bool` | optional | When `true`, the server canonicalizes the raw `board` (suit isomorphism), derives `flop_id`, and remaps `board`/`hero_cards`/`partner_cards` into the blueprint's canonical suit frame. Lets the runtime send **real cards** and omit `flop_id`. Default `false`. |
| `seed` | `u64?` | optional | RNG seed for the deterministic action sample. |

**`ActionInput`** (one element of `street_actions`):

| field | type | meaning |
|-------|------|---------|
| `label` | `u8` | action label (§2) |
| `to_total` | `u32` | the player's **total** contribution this street **after** the action (units). Disambiguates multiple bet/raise sizes. |

---

## 4. Response schema — `DecideResponse`

| field | type | meaning |
|-------|------|---------|
| `street` | `string` | `"preflop"` / `"flop"` / `"turn"` / `"river"`. |
| `live` | `u8` | echoed live count. |
| `actions` | `[ActionProb]` | the full action distribution at the hero's node (probabilities sum to 1). |
| `chosen` | `ActionProb` | one action sampled from `actions` (deterministic given `seed`). |
| `search_ms` | `u64` | wall-clock the decision took. |
| `paired` | `bool` | whether pair-mode blocking was applied. |

**`ActionProb`**:

| field | type | meaning |
|-------|------|---------|
| `label` | `u8` | action label (§2). |
| `action` | `string` | human-readable name. |
| `amount` | `i32` | the hero's **total** chips in (units) after taking this action — what the runtime should post to (`commit_entry + contribution`). For fold/check it is the current commit. |
| `prob` | `f32` | probability of this action. |

The runtime either plays `chosen`, or samples `actions` itself.

---

## 5. Decision routing (by live count + street)

The server dispatches on `live` and `board.len()`:

| condition | path | mechanism | needs |
|-----------|------|-----------|-------|
| `board == []` | **preflop** | EQR strategy (table lookup, instant) | `PF_STRAT` loaded; `street_actions` (preflop betting), `hero_cards`, `hero_idx`. |
| `live == 2` | **heads-up** | flop = banked exact HU strategy; turn/river = real-time exact HU search | flop needs the live-2 bank (`{BP_ROOT}/live2`); turn/river need nothing extra. |
| `live ∈ {3,4,5}` | **multiway** | per-street depth-limited search over the bucketed blueprint continuation, rich bet sizing | a blueprint cell (`cell_dir` + `flop_id`, or `route = true`). |
| `live ≥ 6` | **full ring** | equity-rollout model: check when unbet, pot-odds call/fold vs Monte-Carlo all-in equity | `to_call` (or derivable from `street_actions`). |

**Connected blueprint (`CONN_BP`, e.g. `blueprint_conn_v4`).** When set, the server
also loads a single self-contained connected blueprint and tries it first for
empty-board and 3–5-card-board requests (`ConnDecider::decide`). It serves
**postflop (flop/turn/river) by lookup** — validated byte-exact and smoke-tested
sane — and this is the recommended postflop path. **It must NOT serve preflop**
(its preflop region is an `MC_NF=1` reach prior, broken for non-pairs — see §11);
preflop must come from `PF_STRAT`. Turn/river via the connected blueprint require
`prior_actions` (the full postflop betting path) + `cell_dir`/`flop_id` (or
`route=true`).

A complete hand is driven by repeated `/decide` calls: one preflop, then one per
postflop street as the board and `street_actions` grow.

---

## 6. Blueprint selection — "snapping" to the right solve

> **CURRENT STATE: only 1 of ~20 blueprints is solved.** The single solved blueprint
> is the production game class: **6-max, 100 bb deep, NL10-style rake (5% capped at
> 10 bb), 0.5/1 bb blinds, no-flop-no-drop.** Until the rest are solved, every request
> snaps to this one blueprint regardless of the table's real stake or depth.
>
> **The solved postflop blueprint is `blueprint_conn_v4`** (connected GPU MCCFR):
> rich preflop menu (5 pot-relative raises + all-in) feeding per-flop solved postflop
> cells — exact heads-up + fine SPR-binned 3-way/4-way + lean 5-6-way, 1755 flops,
> validated. Load with `CONN_BP=blueprint_conn_v4 CONN_GS14=gs14_blueprint_cache`
> (params np=6, nraises=5, nb=200, maxna=7). Serves **postflop**; preflop = `PF_STRAT`
> (§5, §11).

A real deployment needs a **matrix of blueprints** because the equilibrium strategy
depends on two things the cell grid does **not** capture:

1. **Stake** — the rake structure (rake % and the cap *in bb*) materially changes
   marginal calls/folds. NL10 (5% cap 10 bb) plays differently from NL200 (lower
   effective rake in bb). Each stake is a separate solve.
2. **Stack depth** — the buy-in in bb. A 100 bb blueprint mis-prices a 40 bb table
   (every street's SPR is different). Each depth is a separate solve.

~20 blueprints ≈ a few **stakes × a few stack depths** (e.g. 5 stakes ×
4 depths {100, 75, 50, 40 bb}).

### 6.1 The snap-to algorithm (two-key nearest match)

Given the table's `(stake, stack_bb)`, select the blueprint in this order:

```
1. STAKE — match first. Pick the blueprint family whose stake == the table's stake.
            If no exact stake exists, fall back to the NEAREST stake (by rake-in-bb).
2. STACK  — among that stake's blueprints, pick the one whose solved stack depth is
            CLOSEST to the table's effective stack (|solved_bb − table_bb| minimized).
```

Stake is the **primary** key (rake changes strategy more than a moderate depth
mismatch); stack depth is the **secondary** key. Then, *within* the chosen blueprint,
the existing cell router (§6.2) snaps to the right `(live, commit, pot)` cell.

To support this, the request should carry the table context (proposed fields — not
yet consumed, since only one blueprint exists):

| proposed field | type | meaning |
|----------------|------|---------|
| `stake` | `string` | the table's stake id, e.g. `"NL10"`, `"NL50"`. |
| `stack_bb` | `u32` | the table's effective stack depth in big blinds. |

Until the matrix is filled, the server ignores these and uses the one blueprint;
once filled, `BP_ROOT` becomes a directory of blueprints keyed by `(stake, stack_bb)`
and the server resolves them with the algorithm above before cell routing.

### 6.2 Cell routing *within* a blueprint (already implemented)

Inside a single blueprint, the flop-entry cell is chosen by `FlopRouter`
(`play_harness::full_hand`): it keys on `(live, SPR bin)` via `SeamCell::bucket_key`,
with **nearest-SPR-bin fallback**, so any `(live, commit, pot)` maps to the closest
banked cell.

Cell directory naming: **`live{N}_c{commit}_p{pot}_b{buckets}`**, e.g.
`live3_c6_p20_b15` = 3 live, 6-unit matched commit, 20-unit pot, 15 buckets. (Live-5
cells use fewer buckets, `b8`, to fit memory.) The runtime can either:
- send `cell_dir` + `flop_id` directly (it tracks the hand), or
- send the raw `board` with `route = true` and let the server derive `flop_id` (it
  still needs `cell_dir` for the live/commit/pot routing).

The SPR bin = `floor( log2( (stack − commit) / pot ) / 0.25 )` at `stack = 200` units.

---

## 7. Pair mode (range blocking)

Set `partner_cards` (+ `partner_idx`) to model a colluding partner who shares hole-card
info. The partner's two cards are removed from the **pool's** range only (the hero and
partner keep their full ranges; modeling the partner as a point-mass *adversary*
distorts the solve). The decision shifts measurably when blocking is active.
`paired = true` in the response confirms it. Pair mode applies to the multiway
(live-3/4/5) search path; it is a no-op for heads-up and the rollout path.

---

## 8. Board canonicalization (`route = true`)

The blueprint is solved on **canonical** flops (1,755 suit-isomorphism classes). With
`route = true` the server:
1. canonicalizes the raw flop → derives `flop_id`,
2. finds the suit permutation mapping the raw flop onto its canonical form,
3. remaps `board[3..]` (turn/river) and `hero_cards` / `partner_cards` by that
   permutation.

This is suit-iso **invariant** — a permuted board yields an identical decision. For
turn/river the remapped card must land in the bank's sampled runout set
(`bp.turns`/`bp.rivers`); flops with non-trivial suit automorphism (paired/two-tone)
may map a banked runout to an equivalent un-sampled card → `400`.

---

## 9. Server configuration (env vars)

| var | default | meaning |
|-----|---------|---------|
| `BP_ROOT` | `blueprint_out_v1` | blueprint root directory. |
| `BIND` | `127.0.0.1:8080` | listen address. |
| `PF_STRAT` | _(unset)_ | preflop EQR strategy basename; required for preflop decisions. |
| `ITERS` | `160` | base search iterations (per-live schedule applies on top). |
| `PAR` | _(off)_ | enable the parallel search walk. |
| `DCFR` | _(off)_ | enable Discounted-CFR (faster convergence). |
| `L2_SUBDIR` | `live2` | live-2 bank subdirectory (e.g. `live2_m2` for the rich-menu bank). |

Per-live latency is auto-scheduled (parallel + DCFR for heavy multiway counts) to fit
a ~14 s real-time budget: live-3 ~0.3 s, live-4 ~3.5 s, live-5 ~4.5 s; live-2 river
~0.1 s, live-2 turn ~5–8 s; live-6 ~0.5 s.

---

## 10. Examples

**Preflop** (UTG, raise-or-fold tree):
```bash
curl -s localhost:8080/decide -H 'content-type: application/json' -d '{
  "board": [], "hero_cards": [48, 49], "live": 6, "hero_idx": 3,
  "street_actions": [] }'
```

**Multiway flop** (live-3, raw board, server routes):
```bash
curl -s localhost:8080/decide -H 'content-type: application/json' -d '{
  "board": [10,7,0], "hero_cards": [20,33], "live": 3, "hero_idx": 0,
  "commit_entry": 6, "pot_entry": 20, "street_actions": [],
  "cell_dir": "live3_c6_p20_b15", "route": true }'
```

**Heads-up river** (live-2, real-time exact search, no bank needed):
```bash
curl -s localhost:8080/decide -H 'content-type: application/json' -d '{
  "board": [44,33,8,2,19], "hero_cards": [45,30], "live": 2, "hero_idx": 0,
  "commit_entry": 20, "pot_entry": 60, "street_actions": [] }'
```

**Full ring** (live-6, equity rollout, facing a bet):
```bash
curl -s localhost:8080/decide -H 'content-type: application/json' -d '{
  "board": [48,45,42], "hero_cards": [49,50], "live": 6, "hero_idx": 0,
  "commit_entry": 20, "pot_entry": 100, "to_call": 50 }'
```

---

## 11. Model caveats (the honest frontier)

**★ KNOWN BUG (preflop), found 2026-06-26 by HTTP smoke test — NOT a deliberate
choice:** the connected blueprint (`CONN_BP`, e.g. `blueprint_conn_v4`) must serve
**postflop only**. Its *preflop* region is trained in the connected solve's Phase A
over a **single representative flop** (`MC_NF=1`, a dry disconnected rainbow board),
so it is valid as a *reach prior into the postflop cells* but is **NOT a usable
preflop strategy**: on that one dry board, pairs dominate (fold ≈ 0) and non-pairs
over-fold, inversely correlated with strength (measured AKs fold 0.83, KQo 0.91,
72s 0.53). Pairs look fine, which is why the AA-only unit test missed it.
- **The fix:** serve preflop from the DEDICATED all-flops preflop solve (`PF_STRAT`
  = `preflop_out_v1`, solved by `preflop_runner` against the banked oracle over all
  1755 flops), per §5. The connected blueprint's preflop is reach-only.
- **Current `bot-server` routing is WRONG here:** `decide_handler` tries `conn.decide`
  FIRST for empty-board requests, so it serves the broken connected preflop instead of
  `PF_STRAT`. Until fixed, either (a) don't set `CONN_BP` for preflop, or (b) make the
  empty-board branch prefer `PF_STRAT` and fall back to `conn` only postflop.
- **To get a sound preflop WITH all-ins** (the rich-abstraction goal): re-run
  `preflop_runner` with `BetSize::AllIn` in the preflop menu against the banked
  oracle. Postflop (the connected v4 cells) is per-flop solved and validated — no
  preflop bias there.

The remaining items below are deliberate model choices, not bugs — every (street ×
live count) decision is still served:

- **Blueprint matrix is 1/20 solved** — all requests currently snap to the NL10 /
  100 bb blueprint (§6). Real-stake/depth selection is pending the other solves.
- **live-6 is check-down equity** — the full-ring model never value-bets; it checks
  when unbet and calls/folds by pot odds.
- **live-2 turn** uses a nested solve (rich turn menu, check-only river continuation);
  the river is re-solved exactly on arrival, so it plays correctly.
- **live-2 turn/river ranges are uniform** (unconstrained re-solve — no reach
  narrowing from prior betting).
- **The blueprint strategy is u8-quantized on disk** (SSBP2, ~0.1 % on EV-relevant
  mass, money-test-proven play-safe) — the runtime decompresses to f32 at load.
