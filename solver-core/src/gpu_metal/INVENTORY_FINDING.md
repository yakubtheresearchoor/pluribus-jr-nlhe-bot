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

## The standing discipline now codified (the deepest version)

The user's framing, validated twice in succession:
  - Site (d) part 1: gate residual 0.094 → unit test discriminator
    (kernel math correct, second location exists)
  - Re-survey itself: agent miscounted line ranges, claimed inventory
    complete when it wasn't

The verification step that was supposed to catch the inventory gap
INTRODUCED its own error. If the agent had trusted the re-survey, the
two production sorted-sweep call sites would have shipped rake-free.
Only the agent's independent check (re-survey says complete, gate
residual says something is still rake-free, chase the tension) caught
the miscounted line ranges. The gate residual caught what TWO rounds
of source-reading inventory missed.

## The hierarchy-of-trust principle (sharpened)

The project's running theme has been: **"agreement between
implementations is not correctness, anchor against truth."** The
inventory finding adds an orthogonal axis, but stated precisely
this time — the first version was too absolute.

WRONG (too absolute): "execution-grounded signals over source-reading
inventories, when they disagree execution wins."

This formulation would mislead later, because this very project is
full of execution-grounded signals that were ALSO wrong:
  - rake=0 parity gates pass while telling you nothing about rake
  - constant-stub test passes while not exercising the value-dependent
    path
  - the loose 0.5 parity gate passed while hiding the inclusion-
    exclusion bug
  - zero-assertion "trace" tests pass while validating nothing
  - the cap-binding site (b) regret-level gate "passes" while not
    actually measuring site (b)

So execution-grounded signals are not automatically authoritative.
The reason the gate residual WON over the inventory in this case is
not that execution always beats reading — it's that the gate residual
was actually exercising production payoff paths while the inventory
wasn't grounded in execution at all.

PRECISE (the deeper principle):

  **A signal that actually exercises the path under test beats one
   that doesn't. Execution-grounded signals earn their authority by
   exercising the thing, not by being execution-grounded per se.**

The failure mode to guard against, in either direction:
  - **The reading not grounded in execution** (inventory says a
    location is covered; it isn't, because the reading miscounted)
  - **The execution-grounded signal not exercising the thing**
    (gate passes; it isn't validating what it appears to, because
    the scenario doesn't reach the relevant code path with the
    relevant input)

Both have happened in this project. The principle covers both:
ask "does this signal actually exercise the path in question?" If
yes, trust it; if no, it's green-but-useless regardless of whether
it's a reading or an execution.

In this arc specifically:
  - The original inventory: reading, not grounded in execution → wrong
  - The re-survey: reading, not grounded in execution → wrong
  - The site (b) gate at rake=0: execution-grounded, not exercising
    rake → useless for rake (the Slice 2 gate was added because of this)
  - The site (b) cap-binding gate: execution-grounded, exercising rake
    but not site (b) specifically → false discriminator (vindicated by
    the kernel unit test)
  - The site (b) kernel unit test: execution-grounded, routed to
    site (b)'s branch via fold_mask → actually exercises site (b),
    authoritative
  - The HU gate residual after site (d) part 1: execution-grounded,
    exercising HU production paths → authoritative, and revealed the
    second unmapped location that two inventories missed

So the right question at every step: "does this signal actually
exercise what I claim it validates?" The gate residual at HU passed
that test. The two inventories did not.

## Consequence for Phase B completion criterion

You cannot declare Phase B complete by checking inventory items off,
because the inventory is demonstrably incomplete and re-surveys are
unreliable. The only declarative criterion is:

  **Every production payoff path is exercised by some gate scenario,
   AND every such gate reaches f32 floor.**

  - The f32-floor-everywhere half is the execution-grounded detector
    (a still-rake-free location holds its gate above floor).
  - The coverage half (every production path exercised by some gate)
    is what the inventory was supposed to ensure and CANNOT be
    trusted to ensure.

So the coverage question must be answered without relying on the
inventory. Two ways:

### Option 1 (logically airtight, more instrumentation)

Instrument the production solve to assert, at each terminal it
evaluates, that the rake path was taken (or correctly skipped per
no-flop-no-drop). Run a representative solve; if every terminal
asserts rake-applied, coverage is proven by execution, not by
enumeration.

This converts coverage from "did I enumerate all locations" (which
keeps being wrong) to "did every terminal the solve actually hit
apply rake" (which the running code can attest to directly).

### Option 2 (less rigorous but cheaper, audited)

Audit that the gate scenarios collectively exercise the terminal
types a production solve produces:
  - HU showdowns at flop / turn / river zones
  - 3p showdowns (all-equal, fold terminals at various depths)
  - 3p side-pot terminals (after allin)
  - 4p+ showdowns (factored path)
  - Rake gating: flop_seen=true terminals (current) and false
    (preflop, dormant)

If the gate scenarios are representative of production play AND every
one reaches f32 floor, the inferred conclusion is "rake correct on
all paths a real solve traverses." A code location no production
solve ever reaches is harmless even if rake-free — it never executes.

So **the coverage that matters is "every path a production solve
traverses," not "every code location in vcfr.metal."** The gate
scenarios should be chosen to be representative of production play;
the inventory becomes an implementation detail rather than a
coverage criterion.

## Recommendation (revised per sharpened principle)

The first version of this recommendation said "apply Option 2
minimally, consider Option 1 as future hardening." That under-applied
the sharpened principle.

The sharpened principle says: a signal earns authority by ACTUALLY
exercising the thing. Option 2 (gate-scenario representativeness of
production play) is coverage-by-reasoning — you have to reason about
which terminal types production produces, then verify the gate
scenarios collectively exercise them. That reasoning is the same
class of activity as the inventory: a reading-based hypothesis about
what production does. The inventory has been wrong twice. Coverage-
by-reasoning could be wrong the same way.

Option 1 (production-solve instrumented assertion that rake was
applied at every terminal it touches) is coverage-by-execution: the
running solve answers "did every terminal I evaluated apply rake?"
directly. No reasoning step that could fail; the live code reports.

Given the inventory has been wrong twice, that's the evidence that
coverage-by-reasoning is the approach that fails on THIS codebase.
Option 1 is the version that doesn't depend on getting coverage right
by reasoning.

**Revised recommendation**: For Phase B done-declaration, lean toward
Option 1 (production-solve instrumentation). It's more work, but the
inventory being wrong twice is exactly the evidence that the cheaper
coverage-by-reasoning approach is the one that fails. Option 2 as
fallback only if instrumentation is genuinely too expensive for the
done-declaration timeframe — and even then, it should be paired with
extra paranoia (audit by multiple methods, accept the residual risk
explicitly).

## Principle to carry beyond Phase B

The inventory is not a finished artifact, it's a hypothesis subject
to empirical detection. This applies to:

  - Future Metal/GPU work (kernel code locations evolve; an inventory
    drawn today is stale tomorrow)
  - Preflop integration (new terminal types → new payoff paths,
    inventory will be partial)
  - Bucketing / postflop abstraction (new code shapes)
  - Any future cross-implementation parity work

The standing method for catching unmapped locations (revised):
  1. Choose signals that actually exercise the path under test
     (not signals that are merely execution-grounded)
  2. f32-floor-everywhere as the declarative completion criterion,
     but only on gate scenarios that exercise production paths
  3. Unit test routed to a specific location as the discriminator
     when a gate doesn't reach floor (kernel-math-wrong vs
     location-missing)
  4. Prefer instrumented assertions in the running solve over
     coverage-by-reasoning, when achievable
  5. For both readings AND execution signals: ask "does this
     actually exercise what I claim it validates?" Skepticism
     applies to both directions.

This is the deepest lesson the rake arc produced and it is bigger than
rake.

## Altitude: where the rake arc sits in the larger plan

The rake arc started as "implement rake, ~1-2 hours" and has become a
multi-session investigation. So far it has produced:

  - CPU: rake was stored-but-unimplemented (the original gap)
  - CPU: rake implemented across all 5 showdown paths with the
    sorted-sweep tie-band subtlety (audit-fix #37 lineage)
  - Validation arc: tie-band oracle coverage gap closed
  - Build hygiene: 3 stale/zero-assertion tests removed; CUDA legacy
    deleted (49 files, 14,811 lines)
  - GPU: showdown-site inventory demonstrated incomplete TWICE
  - GPU: K=2 cluster fully closed (3 unit tests at 0.0)
  - GPU: site (d) part 1 closed; sorted_sweep second location found
  - Sharpened the project's recurring principle (this file)

Every one was a real correctness issue that would have produced
wrong blueprint output or false-confidence validation. The arc has
justified itself many times over in caught bugs.

That said: the question "is the remaining rake work worth the
marginal correctness" is legitimate. Site (d) part 2 and site (e)
are real (a rake-free showdown path produces wrong payoffs, and
rake is strategy-determining), so they SHOULD be finished — this
isn't a place to stop short.

But a meta-point worth keeping visible: **the blueprint is NOT
gated on the remaining GPU rake.** The blueprint is computed on CPU
(via FlopStartVectorCfr); CPU rake is done and validated. The GPU
rake is for the real-time search path (which uses the GPU and must
be rake-correct). So:

  - Blueprint computation: can proceed RIGHT NOW on CPU, rake-correct
  - Real-time search: requires the remaining GPU rake work
  - The two are independent and can run in parallel

The remaining Phase B work (site (d) part 2, site (e), instrumented
coverage check for the done-declaration) is well-scoped and should
be finished while context and detection method are fresh. But if it
extends further (site (e) being more complex and possibly multi-
location like site (d) turned out to be), the blueprint can proceed
in parallel rather than waiting. This is not a redirect; it's a
sequencing option to keep visible.

The sharpening of the principle to "actually exercises the thing"
applies broadly. The remaining sites should be finished, validated
under the sharpened principle, and the blueprint can be computed in
parallel on the rake-correct CPU path whenever convenient.

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
