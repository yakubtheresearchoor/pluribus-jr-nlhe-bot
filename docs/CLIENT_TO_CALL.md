# Client integration: `to_call` (and the fields it depends on)

**Status (2026-07-04, blueprint_conn_v5 deployment):** `to_call` is no longer a
live-6-only nicety. Three server paths now key on it, two of them preflop —
**if your client does not send it, those paths silently fall back to safe
defaults** (blind folds / no jam subgame). Sending it is the single highest-value
integration improvement on the client side.

---

## 1. Definition

`to_call` = the number of **units** the hero must **add right now** to continue
(i.e. call). It is a **delta**, not a total:

```
to_call = (highest to_total among all players this street) − (hero's current total contribution)
```

- **Units**, not currency: `1 bb = 2 units` (see RUNTIME_API.md §2). At NL10
  ($0.05/$0.10), 1 unit = $0.05.
- Preflop, the hero's "current total contribution" **includes the blind already
  posted**. BB facing a raise to 5 units has already posted 2, so
  `to_call = 5 − 2 = 3`.
- Facing no wager (option to check): `to_call = 0` or omit the field. Both mean
  "unbet". Do not send the amount of your own pending blind as `to_call`.
- Never negative; clamp at 0.

**JSON:** optional `u32` field on the `/decide` request body:

```json
{ "board": [], "hero_cards": [33, 9], "live": 2, "hero_idx": 1,
  "commit_entry": 2, "pot_entry": 36, "to_call": 26,
  "street_actions": [ ... ] }
```

## 2. Consistency contract (the part that actually breaks integrations)

`to_call` is cross-checked against two other fields. Get all three from the same
snapshot of table state:

| field | meaning | consistency rule |
|---|---|---|
| `commit_entry` | hero's TOTAL contribution so far (blinds included preflop) | `commit_entry + to_call` = the amount hero would be in for after calling = the current highest `to_total` |
| `pot_entry` | total pot BEFORE hero's pending call (everyone's money in, including the bet hero is facing) | `pot_entry ≥ live * min-contribution`; the server prices calls as `to_call / (pot_entry + 2*to_call)` |
| `street_actions[].to_total` | each action's actor-total-after-action | `to_call` must equal `max(to_total) − commit_entry` |

If `to_call` and `street_actions` disagree, the server does **not** reconcile
them — different paths read different fields, and you get inconsistent
decisions. Derive `to_call` from the same bookkeeping that builds
`street_actions`.

## 3. What the server does with it (per path)

### 3a. Preflop equity guard (untrained facing-aggression nodes) — NEW in v5
When a preflop node has no trained strategy (rare late-position 3-bet/4-bet
defense tails), the server prices the call against the **aggressor's Bayesian
posterior range**:

- `to_call` present and `> 0` → real pot-odds decision:
  `call iff equity_vs_posterior > to_call / (pot_entry + 2*to_call) + 0.04`.
  Verified live: 94♥4♥ facing a 3-bet folds (eq 0.256 vs price 0.295); AA calls
  (eq 0.838).
- `to_call` missing → **blind fold** on every facing-aggression untrained node.
  Safe, but folds some profitable calls. This is what you lose by not sending it.

### 3b. Preflop jam subgame (HU low-SPR) — inert without `to_call`
`decide_preflop_jam` re-solves the fold/call/jam subgame with real equities when
ALL of these hold:

- `live == 2` (heads-up after folds — send the count of players still in, not
  seats dealt),
- `to_call` present and `> 0`,
- `pot_entry > 0`,
- post-call SPR ≤ 2.25 (`CONN_PRE_JAM_SPR`): `(stack − commit_entry − to_call) /
  (pot_entry + to_call) ≤ 2.25`. (Deliberately LOW: jam-or-fold is only sound at
  shallow SPR — a wider gate was measured live pure-jamming 44/Q9/JTs for 100bb
  and reverted 2026-07-04. Deeper facing-aggression spots are served by the
  trained chart or the equity guard's fold/call.)

Without `to_call` this path **never runs** — the whole jam-subgame feature is
dead weight.

### 3c. Live-6 equity rollout (postflop)
The pot-odds fallback path. Here `to_call` is optional in the weaker sense: if
missing, the server derives it as `max(street_actions.to_total) − hero to_total
this street`. Sending it explicitly is still preferred — the derivation cannot
see money committed on earlier streets.

## 4. Worked preflop examples (blinds 1/2, stack 200)

**BB facing a BTN open to 5 after three folds** (hero = BB):
```json
{ "board": [], "live": 3, "hero_idx": 1, "commit_entry": 2, "pot_entry": 8,
  "to_call": 3,
  "street_actions": [
    {"label": 0, "to_total": 0}, {"label": 0, "to_total": 0},
    {"label": 0, "to_total": 0}, {"label": 4, "to_total": 5} ] }
```
`pot_entry` = SB 1 + BB 2 + BTN 5 = 8. `to_call` = 5 − 2 = 3.

**BB facing open-5 then SB 3-bet to 28** (hero = BB, now effectively 3 live):
```json
{ "board": [], "live": 3, "hero_idx": 1, "commit_entry": 2, "pot_entry": 35,
  "to_call": 26,
  "street_actions": [
    {"label": 0, "to_total": 0}, {"label": 0, "to_total": 0},
    {"label": 0, "to_total": 0}, {"label": 4, "to_total": 5},
    {"label": 4, "to_total": 28} ] }
```
`pot_entry` = 2 + 5 + 28 = 35. `to_call` = 28 − 2 = 26. (If BTN then folds to
the 3-bet and hero acts HU vs SB, send `live: 2` — that is what arms the jam
subgame.)

**SB completing (facing no raise)**: `to_call = 1` (2 − 1). **BB with the
option** (limped pot): `to_call = 0` or omit.

## 5. Related: `eff_stack` (NEW 2026-07-05 — short-stack correctness)

Optional `u32`, units: the EFFECTIVE stack for this hand (min of hero/villain
starting stacks). The game spec assumes 200u (100bb); without this field every
SPR computation — the jam-subgame gate above all — runs at phantom 100bb depth.
Measured impact: a 43bb-effective J9o 5-bet-jam spew that the jam solver now
correctly folds once it sees the real depth (and AA/QQ jam for the real 86u,
not 200u). **Send it whenever effective depth ≠ 100bb.**

## 5b. Related: `deadline_ms`

Same request body also accepts `deadline_ms` (default 20000): the server queues
and budget-fits solves to finish inside it, and fast-fails 503 the moment a
useful answer can no longer make it. If your client's internal timeout is not
20 s, send the real one — the queue math is only as good as the deadline it's
given.

## 6. Exploit fields (NEW 2026-07-05)

**`opponent_stats`** — array, one object per live SEAM seat. All stats optional
fractions (0..1) except `af` (ratio) and `sample_size` (hands). The server
blends each stat toward the pool prior by `n/(n+200)` — under ~200 hands the
pool dominates, so low-sample noise cannot swing decisions.

```json
"opponent_stats": [
  {"seat_idx": 1, "user_id": 1634018, "vpip": 0.55, "pfr": 0.15, "af": 1.2,
   "wtsd": 0.42, "fold_to_cbet": 0.25, "three_bet": 0.06, "allin": 0.03,
   "sample_size": 300}
]
```

Live consumers (more added as measured):
- **Maniac β relaxation** (HU): bettor with blended `af > 4` or `allin > 0.10`
  floors the river bluff share at 0.25 — the bot keeps calling down maniacs
  while pool-folding vs honest bettors.
- **Foldy-table bluff exception**: the air-bluff suppressor stands down when
  EVERY live villain has blended `fold_to_cbet ≥ 0.45` and `wtsd ≤ 0.28`.
- **Villain-sized 3-bet range** (HU jam gate): the assumed 3-bet range width =
  the villain's blended `three_bet` (pool 9%) — nits get folded to wider,
  wide 3-bettors get defended wider.

**`pool_river_bluff`** — per-request β override (stake-measured); takes priority
over the server env. Send your mined per-stake value.

`archetype` is accepted per seat but currently informational — behavior derives
from the stats themselves (a label can't be blended by sample size).

## 6b. Client checklist

- [ ] Send `to_call` on EVERY request where hero faces a wager (preflop
      included), derived as `max(to_total) − commit_entry` from the same state
      that builds `street_actions`.
- [ ] Preflop `commit_entry` includes the posted blind.
- [ ] `live` = players still in the hand (drives the HU jam-subgame gate).
- [ ] Chips in units (1 bb = 2), never currency.
- [ ] Omit or send 0 when checking is available — never negative.
- [ ] Send `deadline_ms` if your timeout differs from 20 s.
- [ ] Send `eff_stack` (units) whenever effective depth ≠ 100bb.
