# Showdown-site inventory re-survey: TWO production HU locations confirmed

This file documents an inventory-gap finding during Slice 2 Phase B,
recording the discovery + the lesson + the carry-forward for Phase B
Site (d) part 2 (sorted_sweep rake mirror).

## What triggered the re-survey

Phase B Site (d) part 1 (commit `1cc5884`) closed `multiway_brute_force_
showdown`'s K=1 branch (line ~376) for HU rake. The site (d) isolation
unit test went to 0.0 (multiway K=1 math correct in isolation), but
the HU gate dropped only 0.733 → 0.094 — substantial but NOT to f32
floor. Per user-documented discipline ("gate behaving unexpectedly is
the trigger"), the residual pointed to a SECOND code location.

## The re-survey itself surfaced a meta-lesson

Per user direction: "the showdown-site inventory is demonstrated
incomplete, so before declaring Phase B done, re-survey vcfr.metal for
every showdown payoff location exhaustively."

Delegated to an explore agent. The agent's report claimed:
  - `vcfr_streaming_level` is NOT called from production Rust code (true)
  - `factored_showdown_unified` is NOT called from production (true)
  - All `sorted_sweep_showdown_vcfr_local` calls are in dead/test code
  - All five production sites are in `multiway_brute_force_showdown`
  - Inventory is COMPLETE for production code

But independent verification of the line ranges revealed the agent
MISIDENTIFIED kernel boundaries:

| Line | Helper called             | Agent's claim   | Actual kernel              |
|------|----------------------------|------------------|----------------------------|
| 1263 | sorted_sweep_showdown_vcfr_local | vcfr_streaming_level (dead) | vcfr_streaming_level (dead) ✓ |
| **1716** | sorted_sweep_showdown_vcfr_local | vcfr_streaming_level (dead) ✗ | **vcfr_bottom_up_batched (1504-2149)** PRODUCTION |
| **1822** | sorted_sweep_showdown_vcfr_local | vcfr_streaming_level (dead) ✗ | **vcfr_bottom_up_batched (1504-2149)** PRODUCTION |
| 2248 | sorted_sweep_showdown_vcfr_local | debug_sweep (test) | debug_sweep (test) ✓ |

The agent miscounted line ranges, concluding the inventory was
complete when it WASN'T. The user's lesson hits twice: the original
inventory had gaps, and the re-survey to verify completeness ALSO
had gaps. The only reliable detector is the empirical signal
(gate-not-reaching-f32-floor).

## The two unmapped production sorted_sweep call sites

Both inside `vcfr_bottom_up_batched` (lines 1504-2149), turn/river zones:

**Call site 1 (line ~1716)** — `num_active_opp == 1` (HU after some folds):
```c
if (num_active_opp == 1) {
    float local_cfv[1326];
    sorted_sweep_showdown_vcfr_local(
        opp_reach_local, num_opp, nh,
        opp_str, opp_idx, pl_str, pl_idx,
        hand_cards, local_cfv);
    float pot_size = float(params.starting_pot) / float(np) + float(c_t);
    for (int h = 0; h < nh; h++) out[h] = local_cfv[h] * pot_size;
}
```

**Call site 2 (line ~1822)** — HU (num_players == 2, all_equal):
```c
sorted_sweep_showdown_vcfr_local(
    opp_reach_local, num_opp, nh,
    opp_str, opp_idx, pl_str, pl_idx,
    hand_cards, local_cfv);
for (int h = 0; h < nh; h++) out[h] = half_pot * local_cfv[h];
```

Both are RAKE-FREE. They handle HU showdowns in the batched (turn/river)
kernel. The K=2/3p multi-active path uses different code below them
which is also independent of multiway_brute_force_showdown.

## CPU reference for the mirror

`sorted_sweep_with_rake_components` (showdown.rs ~170-260) returns three
vectors:
  - `sweep_net[h]` — current `local_cfv[h]` output (wins - losses)
  - `win_reach[h]` — cumulative opp reach where h strictly wins
  - `tie_reach[h]` — cumulative opp reach where h ties at top

Caller applies:
```rust
cfv[h] = half_pot * sweep_net[h] - rake * (win_reach[h] + 0.5 * tie_reach[h])
```

With the +reach inclusion-exclusion correction for HU tie band (the
"audit-fix #37" subtlety: when opp_str == pl_str at the boundary,
opp = h is in the tie band, and its reach is double-subtracted by
the two-card minus vector without the +reach correction).

## Phase B Site (d) part 2 plan (NEXT)

The Metal mirror requires extending `sorted_sweep_showdown_vcfr_local`
to compute `win_reach` and `tie_reach` in addition to `sweep_net`. The
two production call sites then apply the rake correction inline:
```c
out[h] = half_pot * sweep_net[h] - rake * (win_reach[h] + 0.5 * tie_reach[h])
```

DoD per site (d):
  1. HU gate (`site_d_hu_rake`) reaches f32 floor (currently 0.09375)
  2. site (d) unit test (multiway K=1 isolation) stays at 0.0
  3. All other gates unchanged (empirical isolation check)

## The standing discipline now codified

The user's framing, validated twice in succession:
  - Site (d) part 1: gate residual 0.094 → unit test discriminator
    (kernel math correct, second location exists)
  - Re-survey itself: agent miscounted line ranges, claimed inventory
    complete when it wasn't

Both iterations of "verify the inventory" had gaps. The only reliable
detector is the empirical signal: gate-not-reaching-f32-floor MUST be
treated as the standing trigger for an unmapped location, until proven
otherwise. The unit test (routed to a specific code location) is the
discriminator: kernel-math-wrong vs location-missing.

## Carry for site (e)

Site (e) factored has two known locations (multiway ~447 and
factored_showdown_unified kernel ~3113). Per the inventory finding,
factored_showdown_unified is NOT called in production (verified
twice — no `create_pipeline("factored_showdown_unified")` in Rust
code). So site (e)'s production location is multiway K≥3 only.

BUT — apply the standing detection method anyway when closing (e):
  1. Add site (e) unit test routed to multiway K≥3
  2. Close multiway K≥3
  3. Run all gates — site (e) gate should reach f32 floor
  4. If not, the residual points to ANOTHER unmapped location (just
     like (d)'s sorted_sweep). Add discriminating unit test.
  5. Continue until the gate reaches f32 floor

Apply same to any future post-Phase-B site that might come from
preflop integration.
