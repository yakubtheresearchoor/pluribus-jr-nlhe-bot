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

## Fifth iteration: inline payoff sites were dead code, then consolidated

The fourth-iteration finding (inline sites in vcfr_bottom_up_batched
bypass the helper) was itself WRONG. Phase B Site (d) part 2 surgery
mirrored rake into the inline sites; the HU gate stayed at EXACTLY
0.09375006 — unchanged byte-for-byte. The standing question applied
to the surgery itself: "did this code actually execute?" Answer: NO.

The entire ~600-line inline-sites block in vcfr_bottom_up_batched
was inside an `if (false) { ... }` gate at line 1749. Comment at
line 1727 said "Legacy showdown evaluation code removed; replaced by
brute-force above" — the code wasn't removed, just gated. Phase B
Site (d) part 2 edits, F1 edit, and the "5+ inline sites" framing
were ALL on dead code.

The fifth-iteration refinement to the standing method:

  **DISTURBANCE-TEST REACHABILITY BEFORE TRUSTING ANY SITE IS LIVE
  OR DEAD.** Output-write enumeration is necessary but not sufficient
  — each enumerated site must also be verified REACHABLE in
  production code paths. The standing question applies to enumeration
  itself, in both directions:
    - Reading says site is live: insert a deliberate disturbance
      (NaN write, deliberately wrong value). Run production tests.
      If nothing detects it, the site is dead despite the reading.
    - Reading says site is dead: same disturbance test. If anything
      detects it, the site is live despite the reading.

  Both directions of the dead-or-alive claim require execution-
  grounded confirmation, not source-reading.

## Consolidation: per-site-with-scatter collapsed to single-helper chokepoint

the lead's decision after the fifth-iteration finding: "the dead inline
fast paths are abandoned. Clear them out and consolidate on the single
helper. The if(false) finding means production already routes through
multiway_brute_force_showdown, so this is formalizing what's already
true, not changing the live path."

Executed via the lead's deliberate order:

  Step 1 — disturbance-test the if(false) block to PROVE dead before
    deleting (5 NaN writes at different points, including all prior
    surgery sites). Full test suite ran with disturbances in place;
    zero detection, byte-identical results. Reachability proven dead
    by execution.

  Step 2 — delete the dead block as isolated commit (commit 2fc543e):
    627 lines removed (legacy if(false) block + unused
    `sorted_sweep_showdown_vcfr_local_with_components` helper whose
    only callers were inside the deleted block). Forced clean rebuild,
    full test suite confirmed only dead code removed.

  Step 3 — confirm helper is sole production payoff chokepoint
    (this commit). Post-deletion enumeration:
      vcfr_bottom_up: line 863 `out[h] = local_out[h]` (multiway
        helper result), plus chance-integration aggregation writes.
      vcfr_bottom_up_batched: line 1595 `out[h] = local_out[h]`
        (multiway helper result), plus chance-integration writes.
    The chance writes aggregate already-rake-applied terminal CFVs;
    they preserve rake without applying it themselves. So
    `multiway_brute_force_showdown` is the unambiguous payoff
    chokepoint.

    Helper reachability is already proven historically: every
    successful K=2/K=1 rake closure was an edit to the helper that
    DID change test results (positive-form disturbance test), while
    the if(false) block deletion changed NOTHING.

  Step 4 — finish site (e) K≥3 factored (the only remaining
    rake-free branch in the helper). Next session.

  Step 5 — production-solve instrumentation as permanent CI
    completeness check. Next session.

## What the consolidation buys

The per-site-with-scattered-instrumentation plan from before the
fifth-iteration finding collapses to single-helper-chokepoint, which
is simpler and structurally robust:

  - Completeness story: one path to verify (the helper), not N
    scattered sites to police.
  - Future-divergence risk: any future change that adds a payoff
    path bypassing the helper would be caught by the production-solve
    instrumentation (Step 5). The chokepoint structure makes this
    instrumentation cheap and decisive.
  - Maintenance: no scattered rake math to keep in sync across
    inline fast paths.

## Five iterations of inventory/reading failure, all caught by execution

  1. Original inventory missed sorted_sweep call sites entirely
  2. Re-survey (explore agent) miscounted line ranges
  3. My own re-count placed a dead-code call in production
  4. Helper-call search missed inline payoff sites that bypass helpers
  5. Output-write enumeration missed `if (false)` gates → all inline
     sites were dead, surgery on dead code

Plus the build-system bug: the rerun-if-changed path was wrong, so
Metal-only edits weren't reliably recompiled. Validation history had
to be re-verified against a fresh build (it held).

Six false-greens / reading failures in the arc. Each caught by the
same operational question: "does this signal actually exercise what
I claim it validates?" applied to both readings AND execution
signals AND build-system state AND code reachability.

## Fourth iteration: inline payoff sites bypass the helper

Phase B Site (d) part 2 attempted to mirror rake into the production
`sorted_sweep_showdown_vcfr_local` call sites. Two production sites
were correctly edited. But the HU gate stayed at exactly 0.09375006,
unchanged. The standing question revealed: vcfr_bottom_up_batched has
~5+ INLINE payoff computations that bypass the sorted_sweep helper
entirely:

  | Line  | Output                                    | Status     |
  |-------|-------------------------------------------|------------|
  | 1719  | multiway helper call                      | rake-correct |
  | 1772  | out[h] = payoff * cfreach (inline)        | RAKE-FREE  |
  | 1820  | out[h] = payoff * (...) (inline)          | RAKE-FREE  |
  | 1872  | sorted_sweep_with_components              | rake-correct |
  | 1932  | out[h] = (sp/np+c_t) * (...) inline       | RAKE-FREE  |
  | 1959  | out[h] = payoff                           | RAKE-FREE  |
  | 1995  | sorted_sweep_with_components              | rake-correct |
  | 2069+ | per-level inline computations             | RAKE-FREE  |

So searching for the helper's call sites was insufficient: inlined
logic doesn't call the helper. The fourth-iteration refinement:

  **OPERATIONAL METHOD FOR ENUMERATING PAYOFF SITES**:

  Enumerate by the OUTPUT-WRITE PATTERN (`out[h] = ...`,
  `cfv[h] = ...`, `returned_cfv[h] = ...`), NOT by helper-call
  search. A helper-call search is BLIND to code that inlines the
  helper's logic. The output-write pattern catches every assignment
  to a per-hand CFV result, helper-mediated or not. Then classify
  each by which CPU showdown path it mirrors and verify rake-correct.

## Build-system bug: validation history needed re-verification

A related but distinct failure surfaced in the same session:

**The bug**: `cargo:rerun-if-changed` in `solver-core/build.rs` was
watching `src/gpu/metal/shaders/{}` — a non-existent path. The actual
location is `src/gpu_metal/shaders/{}`. So cargo was NOT reliably
recompiling the .metal shader on Metal-only edits.

**Why it didn't fail visibly until now**: earlier Phase B edits all
included adjacent Rust struct changes (extending BParams, DebugBruteForce
Params, etc.). Those Rust changes invalidated other build artifacts,
which transitively re-ran build.rs, which recompiled the metallib.
So edits like "kernel + Rust struct" recompiled correctly. But
edits like "kernel-only" would NOT have triggered a rebuild — they'd
test against stale metallib binary.

**Why this matters**: this is the build-system equivalent of a
false-green signal. The same standing principle applies to validation
history: "the tests passed → the kernel must have rebuilt" was an
ASSUMPTION, exactly the kind the principle says to distrust. The
green test results from before the build fix could have been
validating stale shader binary in any place where the edit was
Metal-only.

**Required action (not optional cleanup)**: with the path fixed,
force a clean rebuild and re-run the entire Phase B validation suite
against a verified-fresh shader. If results hold, the earlier
history was real (Rust changes did trigger rebuilds in practice).
If anything regresses, that result was validated against stale
binary and the closure has to be redone.

**Re-verification result (done before this update was banked)**: all
13 Phase B unit tests still pass at 0.0 against the freshly-built
shader; rake=0 baseline passes; broader parity tests
(three_max_parity, three_max_reach, flop_start_cpu_test,
iter_divergence, gpu_brute_force_unit) all pass. The Phase B
validation history is real — but now VERIFIED, not assumed.

**Standing lesson**: build-system correctness is part of the trust
chain. A green signal whose recompile-on-edit cannot be guaranteed
is the same false-green class as a gate that doesn't exercise the
thing under test. Verify the build is wired correctly before
trusting it as a coverage harness.

## Surgery direction: Option 2 (refactor) over Option 1 (per-site mirror)

The Phase B Site (d) part 2 finding (5+ inline payoff sites) creates
a choice for closing site (d) fully:

  Option 1: mirror rake into each inline site individually.
            Leaves N scattered payoff sites each carrying rake.
            Each is a future-divergence risk and a future
            "missed a site" finding waiting to happen.

  Option 2: refactor vcfr_bottom_up_batched so all showdowns route
            through the rake-correct helpers, removing the inline
            fast paths. Consolidates to ONE rake path.

The discipline lesson of this arc — four inventory failures, all
caused by scattered payoff sites being un-enumerable — argues for
Option 2 specifically because it ELIMINATES the failure mode rather
than mitigating it. Option 1 mitigates (find these 5 sites, fix
them); Option 2 eliminates (there will only ever be one site,
provably).

The cost: larger immediate surgery, possibly performance impact
(the inline fast paths presumably exist for speed). The performance
concern is real for the real-time search path but acceptable for
the blueprint-offline path. Even for real-time, the correctness-
and-maintainability of one rake path likely outweighs the fast-
path speed.

**Caution if Option 2 is taken**: it re-touches validated kernel
paths. The rake=0 baseline is the INVARIANT that guards the refactor
(the refactor changes the path but not the rake-free result, so
rake=0 must stay green throughout). Each formerly-inline case has
to validate via unit test routed to it that the helper-mediated
result matches what the inline used to produce. Same discipline as
everything else, applied to a larger surgery.

## Sixth iteration: even the trusted CPU reference can be silently wrong

After the consolidation (chokepoint = single helper, all 5 branches
unit-test-validated at 0.0), Phase B closure looked done modulo the
HU gate residual at 0.09375. the lead flagged the clean fraction (3/32)
as suspicious: "the last time a clean-fraction parity gap was assumed
benign it was the inclusion-exclusion bug."

Diagnosis via targeted unit test (one shot, predicted divergence
confirmed byte-for-byte):

  CPU fast path:   rake = (total_pot × rate).min(cap)  ← bug
  Metal K=1:       rake = (main_pot × rate).min(cap)   ← spec-correct

For HU fold-win-after-bet (P0 bets 15, P1 has 5 and folds), the two
disagree by exactly (total - main_pot) × rate = 10 × 0.05 = 0.5 per
hand. Reach-weighted: matches the 0.09375 HU gate residual.

**The CPU was the trusted reference, but its anchor for the
fold-win case used EQUAL contributions** (contributions=[50, 50]
where main_pot == total_pot). The unequal-contributions case
(fold-after-bet, where uncalled bets matter) was a COVERAGE GAP in
the anchors. The CPU was silently wrong on that case.

the lead's spec confirmation: main-pot-only is correct (uncalled bets
returned un-raked per the site). Fix applied:
  1. CPU fast path: change to main_pot_amount rake
  2. Metal K=2 fast path: same change (it mirrored buggy CPU)
  3. Metal K=1: unchanged (already main-pot-only by per-level structure)
  4. New CPU anchor: verify_rake_fold_win_after_bet_uncalled_returned_unraked
     (contributions=[15, 5], hand-computed payoff 4.5, validates the spec)

Result: HU gate dropped from 0.09375 → 9.5e-7 (five orders of
magnitude), all 13 CPU anchors pass, all 5 Metal helper-branch unit
tests pass, all gates at f32 floor.

**Sixth-iteration refinement to the standing method**:

  **EVEN THE TRUSTED REFERENCE NEEDS ANCHORS THAT ACTUALLY COVER
  THE CASES.** A case the anchor doesn't exercise is a case the
  reference can be silently wrong on. The disturbance-test
  reachability discipline (fifth iteration, for CODE coverage) has
  its counterpart: ANCHOR COVERAGE for ARITHMETIC. When the
  trusted-reference status of CPU was "validated by hand
  computation," that validation only held for the cases the anchors
  exercised. Coverage gaps in anchors = silent correctness gaps in
  the reference.

## Seventh iteration: tests that use kernels but don't validate them

Surfaced as a side effect of disturbance-testing
factored_showdown_unified (which was confirmed production-dead):
the test files unified_kernel_gates and precision_attribution_check
DO use the kernel, get NaN back, print `gpu=NaN diff=NaN` for every
hand, and still report `test result: ok`. The assertions in those
tests are weak enough to pass on NaN output.

This is the seventh false-green pattern: a test that uses a kernel
but doesn't validate its CFV values. NaN passes through the test as
"not measurably different from CPU within tolerance."

Not blocking Phase B (these are test-only tests on production-dead
code), but the test infrastructure should be hardened in a separate
cleanup to detect NaN as failure.

## Eight false-greens caught in this arc (the operational principle)

  1. Loose 0.5 parity gate (hid inclusion-exclusion bug)
  2. rake=0 gates (validated nothing about rake)
  3. Constant stub (didn't exercise value-dependent path)
  4. Source-reading inventory (missed inline sites)
  5. Output-write enumeration (missed `if (false)` gates)
  6. CPU-Metal parity residual assumed benign (was over-rake bug)
  7. Test that uses a kernel but doesn't validate (NaN passes through)
  Plus the build.rs path bug: recompile not always triggered

All caught by the SAME operational question:
  **"Does this signal actually exercise what I claim it validates?"**

Applied to readings (inventory), execution signals (gates), test
assertions (do they fire on bad input?), build state (does the edit
actually compile in?), and arithmetic anchors (does the anchor
exercise the case?). Eight findings, one question.

## Chokepoint instrumentation as the standing future-proofing guard

Per the lead's Step 5: "Now that the live path is a single chokepoint
(multiway_brute_force_showdown), the instrumentation's job is to
confirm every production terminal routes through that chokepoint
and the chokepoint applied rake (or correctly skipped it)."

Implemented at chokepoint exit (the two production helper-call sites
in vcfr_bottom_up and vcfr_bottom_up_batched). Marker buffer per-
(terminal-node, hand), three states: 1 = rake-applied (flop_seen=true),
2 = rake-correctly-skipped (flop_seen=false per no-flop-no-drop),
0 = unmarked (BUG: bypassed the chokepoint).

The standing question applied to the instrumentation ITSELF (per
the lead): verify it fires at every terminal by counting marker writes
and comparing to num_terminals × nh. Test
`chokepoint_instrumentation_every_terminal_marked` does exactly this
+ asserts non-terminal cells stay zero + asserts all marked cells
have the expected value.

PERMANENT CI test (not #[ignore]). Per the lead: "keep it permanent
because even with the chokepoint, a future change could add a payoff
path that bypasses the helper, and the instrumentation is the
standing guard, especially with the real-time-search kernel work
coming. It should be in place before that work can reintroduce a
bypass."

## Precision-anchor discipline: scale-discrimination as the proof

Added during Phase 1 P5a (preflop chance integration anchor, 2026-06-04).

The standing trap: a precision diff that's "well under tolerance" can
be either the expected f32-accumulation floor (correct anchor) OR a
quiet loosening of an f64 path that should have collapsed to ~1e-13
(broken anchor that the loose tolerance papers over).

The P5a observation: 4.657e-7 diff against an f64 reference. Six
orders looser than P2.5a's ~1e-13 precedent. Was it the expected
f32-accumulation floor or a regression?

The discriminating method that resolved it: SCALE-VARIATION. Run the
same anchor at input scales spanning many orders of magnitude and
measure diff/scale. Three cases:

  - Diff is f32 accumulation floor: diff scales linearly with input,
    diff/scale stays roughly constant.
  - Diff is fixed-magnitude bug: diff stays similar regardless of
    scale, diff/scale ratio diverges with scale.
  - Diff is f64 non-exactness: diff would already be far below f32
    ULP times scale and would NOT scale with input.

P5a's empirical result: diff/scale ratio 1.56 across nine orders
(1e-6 to 1e3). Linearity confirmed. The 4.657e-7 IS the expected
f32 accumulation floor for that input scale, NOT a fixed-magnitude
offset or f64 path non-exactness.

The bonus finding: my original tolerance was 1e-4, loose enough to
pass an order-of-magnitude regression. Tightened to 1e-5 (50x the
empirical floor) so the anchor catches regressions while still
having headroom for cancellation-pattern variation.

THE PATTERN, as standing discipline for future precision anchors:

  1. Define an f64 reference for the production f32 computation.
  2. Anchor the production at f32 floor against the reference.
  3. Validate the floor by scale variation: run the same anchor at
     scales spanning many orders, assert diff/scale stays within
     a small ratio (e.g., 100x range).
  4. Tighten the tolerance to within ~10x to ~50x of the empirical
     floor so the anchor is sensitive to regressions, not just to
     orders-of-magnitude blowups.

The P2.5a "~1e-13" was the STUB anchor where the entire computation
could be in f64. The P5a "~5e-4 per unit scale" is the RUNNING
orchestrator with f32 accumulators by production design. Both are
correctly anchored; they measure different things. Don't compare
absolute magnitudes across anchors without checking what each is
measuring.

CARRY for the full-scale CPU run (Phase 1 slice 7): aggregation
accumulates at the linear-N×ULP bound, not the sqrt-statistical
bound. For 1755 canonical flops at production scale, this is
probably fine but worth remembering when measuring f32 sufficiency
over a long convergence run: the per-iter f32 drift from aggregation
is bounded by the linear bound, not the sqrt one, so accumulated
drift over iterations follows the linear×iters scaling.

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

**ELEVATED to recommended action (Phase B Site (d) part 2 update)**:

The GPU rake has now revealed itself to be substantially bigger than
the five-site plan. Four inventory failures have surfaced, the
sorted_sweep mirror revealed ~5+ inline sites in vcfr_bottom_up_
batched, the build-system bug required re-verifying the whole arc,
and even the cleanest path forward (Option 2 refactor) re-touches
validated kernel paths and is plausibly several more sessions of
careful work.

The blueprint is genuinely unblocked on the CPU rake-correct path.
The GPU rake serves the downstream real-time-search use case, which
sits BELOW having a blueprint at all in the dependency graph.
Sequencing the harder-and-now-deeper downstream-only work BEFORE
the thing it supports has become the more expensive ordering.

This is the moment where "the blueprint can proceed in parallel"
stops being a note and becomes the recommended action: **start the
CPU blueprint while the GPU rake continues**, because:

  1. The GPU rake is no longer a quick finish (build bug, inline
     sites, possible refactor, possible further unmapped sites)
  2. The blueprint is the actual goal and it's ready
  3. The CPU rake is already validated (Slice 1.x hand-anchored,
     end-to-end at Slice 1.6, all CPU tests green)
  4. Running them in parallel surfaces production blueprint
     properties (convergence, exploitability, range outputs) that
     are themselves likely to surface other findings worth
     responding to

The discipline that has made the GPU rake longer than estimated is
the same discipline that should be running on the CPU blueprint
now. The GPU rake completes correctly via the standing detection
method (gate residual + unit tests + Option 2 refactor + build-
verified); the blueprint runs in parallel on the validated CPU
path. Both proceed under the same principle.

This isn't a redirect away from finishing the GPU rake — the
remaining work is real and should be finished. It's the recognition
that the blueprint doesn't have to wait for that completion, and
delaying the blueprint is now the wrong-direction tradeoff.

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

## Phase 1 preflop scope items not to lose behind sizing questions (2026-06-04)

Slice 7a and its follow-up convergence study surfaced two findings
worth keeping above the noise of the slice 7c sizing question:

### Scope: P1.5.4 / task #44 — preflop CFR loop is unbuilt

`FlopStartVectorCfr` has the four-zone classification done (P1.5.2)
with `Zone::Preflop`, `preflop_local_offset`, `preflop_infoset_count`
populated for preflop-rooted trees. BUT the actual preflop processing
in `bottom_up_zone` is explicitly `unreachable!("preflop processing
lives in P1.5.4 (#44)")`. The pieces for the value pass
(`compute_preflop_cfv_per_canonical_pass`) exist, but the CFR update
loop around it (preflop strategy + preflop reach + preflop regret
update around the per-canonical solve) is missing.

The convergence study currently uses `FlopStartVectorCfr::run` at
flop-start as a POSTFLOP-CFR-rate proxy because the real preflop CFR
loop doesn't exist yet. Pinning N to size a slice 7c run is moot
without the loop being built first. P1.5.4 is a prerequisite for both
the actual preflop solve AND the gate — it is not a downstream item
of the sizing question.

### Standing: convergence-trajectory anomaly check before sizing

Slice 7a's first single-flop trajectory at production nh=1176 on
canonical[0] showed the time-averaged strategy flipping its preferred
action between iter 10 and iter 30 with iter-delta still drifting at
0.024 — anomalous-looking for a 2-action single-flop case if it
indicated CFR instability. Per the lead: "An anomalously-converging CPU
isn't the hyper-confident reference the gate requires." The
`slice7a_multi_flop_convergence_check.rs` test ran the corrected
cum_strategy metric on 6 canonical flops; 6 of 6 showed the same flip
pattern. That LOOKED like bucket (2) (CFR convergence problem) but
the actual discriminator turned out to be a direct correctness check
of the solver against its own documented audit, not interpretation of
the early-iter trajectory.

The discriminator: `tests/convergence_audit.rs` documents specific
exploitability numbers after 100 iters on nh=4 (CPU ~8.8% pot, GPU
~6.0% pot, gates assert both < 20% pot and ratio within [0.25, 4.0]).
Running it (2026-06-04): test passes, and currently measures **CPU
0.37% / GPU 0.37%** of pot — about 24x tighter than the docstring's
recorded baseline at the same iter count, with CPU/GPU agreeing to 5
decimal places. The solver has improved since the docstring was
written (the docstring is now stale and should be updated separately).
Solver is converging correctly to a low-exploitability equilibrium on
its documented baseline.

Therefore the 30-iter drift on nh=1176 single canonicals is NOT a
structural CFR problem; it is the expected early-iter regime of a
larger problem (whatever the specific mechanism — gamma-reset,
cum_strategy averaging timescale, generic). Solver correctness is
established by the audit; the proxy's exact convergence timescale on
the bigger problem doesn't need to be diagnosed further to know the
solver is sound.

The methodology lesson is the carry: when an interpretive read of an
early-iter trajectory looks anomalous, the discriminator is an
execution-grounded correctness check (does the solver reproduce a
known result), not a longer or wider interpretive run. The 6/6
multi-flop pattern was consistent with bucket (2) but ALSO consistent
with bucket (1) under a different cum_strategy timescale; only the
correctness check distinguishes them. "Reasoning to a conclusion"
from consistency alone (CFR bug → benign gamma-reset → CFR bug) is
the signal to switch to a falsifiable test.

Standing rule going forward: any convergence-rate measurement that
will size a long bounded run must FIRST pass a solver-correctness
gate against a documented baseline (i.e., the solver does the right
thing when given iters), and SECOND pass an anomaly check across
multiple instances of the held-fixed random dimension. Correctness
before rate. Without the correctness gate, the rate measurement is
interpreting an unanchored signal.

### The convergence-sizing question is blocked on P1.5.4, not on rate

The proxy convergence study uses `FlopStartVectorCfr::run` at
flop-start because the actual preflop CFR loop doesn't exist. Even
with a clean rate estimate from the proxy, the N would be an estimate
for a loop that isn't built. Once P1.5.4 lands, the convergence study
can run on the real preflop solve (which has its own convergence
behavior — preflop signal averaged over 1755 canonicals may converge
differently). The proxy convergence rate is not the bottleneck;
P1.5.4 is.

## P1.5.4 status: Slice A complete + Slice B built (2026-06-04)

### Slice A complete

Six sub-slices, 19 tests passing at f32 floor or exact 0 diff:

  - A.1 PreflopVectorCfr struct + compute_preflop_strategy
  - A.2 compute_preflop_reach
  - A.3a compute_chance_node_cfv_with_expansion
  - A.3b bottom_up_preflop_for_traverser + run_one_iteration
  - A.3c class×class blocking matrix + per-class fold-terminal CFV
  - A.3d chip_delta from tree.contributions (anchored hand-computed
    asymmetric HU SB=1/BB=2 cases + cross-check vs actual
    showdown_oracle on symmetric overlap)

Files: solver-core/src/solver/preflop_cfr.rs (~700 lines),
solver-core/src/solver/preflop_terminal.rs (~180 lines), 6 test files.

Discoveries (carries):
  1. Preflop→flop chance nodes carry `board_state == Flop`
     (destination convention per flop_start_vector_cfr.rs:174-175);
     discriminator is "parent in preflop zone" not own state.
  2. Oracle's `hand_cards` is a SHARED layout for both players' combos,
     so nh=1 cross-check setups have full self-blocking (cfv=0). Use
     nh=2 with opp_reach favoring non-conflicting combo.
  3. Chip-delta convention: tree.contributions is ground truth; oracle's
     `starting_pot/np + c_t` formula is reused in a NEW preflop-specific
     function (validated oracle NOT extended). Symmetric overlap +
     extended asymmetric cross-checks both pass at sub-ULP.

### Slice B built (engine wiring + multi-iter at reduced flop count)

Added to preflop_cfr.rs:
  - make_per_flop_solver_iter0(flop_tree): production per_flop_solver
    closure wrapping compute_v_flop_at_root_iter0 per canonical,
    ephemeral tables (per slice 7a cost attribution: table = 2% of
    per-flop cost, no caching), layout reconciliation between engine's
    flop_combo_layout and FlopChanceTable's hand_cards order.
  - make_production_terminal_value_fn_hu(tree, blocking_matrix):
    composes A.3c + A.3d into the production terminal_value_fn.
  - compute_chance_node_cfv_with_expansion_subset: subset-of-canonicals
    variant aggregating via P5a-anchored aggregate_preflop_chance_subset.
  - run_one_iteration_subset: subset-aware iteration driver.
  - compute_traverser_br_value + br_recursive: best-response value at
    root for one traverser (argmax-per-class at traverser nodes, plain
    sum at opp nodes per factored CFR convention).

Test:
  - p1_5_4_slice_b_multi_iter_correctness_baseline.rs (#[ignore]):
    runs 50 iters on a subset of 10 canonicals (subset = cheap dimension
    preserving convergence dynamics per the lead's directive), with
    periodic BR exploitability proxy checks. Correctness-baseline-first:
    if exploitability proxy decreases over checkpoints, the loop is
    converging correctly toward the subset's equilibrium and N can be
    read from when cum_strategy iter-delta stabilizes over a window.
    If exploitability INCREASES or stays high → bug to investigate
    BEFORE trusting any N readout.

### Carries to Slice C (full-scale gate)

Once Slice B confirms correctness baseline + measures N on the real
engine:
  - Full 1755 canonicals (no subset)
  - Asymmetric-input action-order seam check (button-first preflop to
    postflop-flop mapping must be detectable, not hidden by symmetric
    inputs)
  - Scale measurements at the real 11.5s/iter steady-state
  - F32 drift at the linear-not-sqrt regime (carried from P5a anchor)
  - Bounded run sized from Slice B's N, long enough that developed
    dynamics are present, confirmed converging correctly

Phase 1 completion gate before GPU port (Phase 2).

## Multiway terminal CFV cost is real, not synthetic (2026-06-04, Slice B.3b)

After B.3's dense-reach smoke test measured 94s/traverser at 6-max,
the lead's "measure the real cost before treating synthetic worst-case as
constraint" discipline pushed B.3b: vary opp reach sparsity, measure
the cost curve.

Data (5 opps, 6-max fold terminal, varying n_nonzero classes per opp):

  n=1 : 0.000s (degenerate — 5×AA needs 10 aces, joint = 0)
  n=2 : 0.029s
  n=5 : 29.7s
  n=10: ~16 min (extrapolated; matches B.3's 94s × 169 traverser classes)
  n=20+: hours, combinatorial blowup

Growth: 2→5 was 1024× for (5/2)^5 = 97× expected by tuple count alone.
Per-tuple joint enumeration also grows with class diversity. Net is
roughly O(n_nonzero^5) at 6-max.

The sparsity exploit IS in the code (`preflop_terminal.rs:243,251`:
cumulative-product zero prune + per-class continue). The enumeration
already only visits non-zero-reach opp tuples. The combinatorial
blowup is OVER the non-zero count, not naively over all 169.

At realistic 6-max preflop fold-terminal opp reach density (~30-100
classes per opp depending on position + prior action), 5 opps × 30-50
classes each = 25M-300M tuples per call. Cost is real, not synthetic.

What this implies (carried into the sequencing):

  1. **Postflop abstraction (#42) matters for terminal CFV too**, not
     just per-flop solves. At the terminal computation layer,
     bucketing opp classes into a smaller equivalence set cuts the
     n^5 multiplier directly.
  2. **Memoization** on sorted class-tuple keys (joint blocking is
     symmetric in classes) is a small drop-in that helps independently
     of bucketing.
  3. **The seam already supports optimization**: terminal_value_fn is
     a closure; optimized implementations slot in without composition
     rework, same way PostflopValueOracle accommodates Fix B and #42.

The cost itself is acceptable for small-scale CPU validation (sparse
synthetic reaches give sub-second per call); production-scale runs
need memoization + likely terminal-layer bucketing in addition to
postflop bucketing.

## CRITICAL: per_flop_solver must use CONVERGED postflop, not iter-0 (2026-06-04)

Architectural finding surfaced when Slice B's BR exploitability proxy
asymmetry was investigated: the engine as initially wired used
`compute_v_flop_at_root_iter0` as the per_flop_solver. That function
runs ONE pass of postflop CFR with UNIFORM postflop strategies for
both players, returning the per-combo CFV under that uniform-postflop
assumption.

This means the engine was solving "preflop CFR with fixed iter-0
postflop", NOT NLHE. The preflop equilibrium under iter-0 postflop is
DIFFERENT from the NLHE preflop equilibrium because traverser's best
preflop action depends on what happens postflop, and the optimal
postflop strategy is not uniform.

Per the lead's directive (2026-06-04): the reference MUST solve the real
game, because:
  - (A) "consistent reproduction" is the trap — a perfectly faithful
    GPU reproduction of an iter-0-postflop engine is a faithful
    reproduction of a strategy you can't play.
  - (B) "converged postflop" is the answer because the expensive thing
    is exactly what the GPU exists to make feasible.
  - The fix is localized to the per_flop_solver boundary. The
    P5a/P5b/P5c composition anchoring is unchanged — it validated the
    composition, not the postflop source.

### What landed

`solver-core/src/solver/flop_start_vector_cfr.rs`:
  - `strategy_{flop,turn,river}_mut()` accessors added
  - `freeze_average_strategy(&mut self, &FlatTree)` method added —
    normalizes cum_strategy into the strategy buffer across all 3
    zones, so a subsequent CFV pass uses the time-averaged Nash-
    converging strategy rather than the oscillating current-iter one
  - `normalize_cum_into_strategy` helper added

`solver-core/src/solver/preflop_start_game.rs`:
  - `compute_v_flop_at_root_converged(canonical, flop_tree, ranges,
    traverser, num_postflop_iters)` added — drop-in replacement for
    `compute_v_flop_at_root_iter0` with correct-game semantics: runs
    num_postflop_iters of DCFR, freezes the averaged strategy, then
    does the CFV pass

`solver-core/src/solver/preflop_cfr.rs`:
  - `make_per_flop_solver_converged(flop_tree, num_postflop_iters)`
    builder added — the production per_flop_solver
  - `make_per_flop_solver_iter0` marked DEPRECATED in docstring (kept
    for Slice A unit-test scenarios where the per-flop CFV semantics
    don't matter — synthetic v_flop_fn validations of the
    composition wiring)

### Cost implications

Each per-flop solve becomes ~num_postflop_iters× the iter-0 cost
(measured in slice 7a as ~7.2s/per-flop-iter at production nh with
slice-7a's bet structure). At num_postflop_iters=100, each per-flop
solve is ~720s = 12 min. A single preflop iteration at full scale
(1755 canonicals × 2 traversers) is ~700 hours of CPU. This is the
expensive thing the GPU is for. CPU validates at SMALL SCALE (small
ranges, restricted nh, few canonicals); the GPU port replicates
at small scale (parity validation) and then runs production scale.

The Slice B test was updated to use the converged solver. Its iter
cost is now too expensive even on a 10-canonical subset at production
nh; Slice B sizing requires further scope reduction (smaller flop
tree, restricted ranges, fewer postflop iters per per-flop solve
balanced against postflop convergence). That sizing is its own slice
question after the wiring is confirmed.

### BR asymmetry deferred

The iter-2 BR asymmetry (40.75 vs 0.0097) observed before the fix may
be partly iter-0-game degeneracy; A.3d's chip_delta and the factored-
CFR-convention plain-sum at opp nodes both passed independent
validations (oracle cross-check + matching the postflop solver's
pattern). Per the lead: investigate independently if the asymmetry
persists after the postflop fix, but expect some of it dissolves
because the engine now solves the real game.

### Why this didn't surface earlier

Slice A's tests used synthetic v_flop_fn closures (constant K or
reach-aware lookups) that bypass the per-flop solve entirely. The
P5c-anchored composition tests validated the orchestration, not the
postflop source. The wrong-game choice was invisible until Slice B
tried to read a meaningful BR exploitability, at which point the
"what game is this engine actually solving" question became
load-bearing for the correctness baseline.

The slice discipline still worked: the bug surfaced at the
composition boundary where it could be caught, not silently in a
later phase. The honest framing of "the engine solves preflop-CFR-
with-iter-0-postflop, is that the intended reference" is what
exposed it.

### Standing: methodology self-correction is the convergence-signal
### counterpart to scale discrimination

The slice 7a study's first metric (current-iter `strategy_flop`
deviation from uniform) triggered trivially at iter 1 because
regret-matching produces pure best-responses immediately; the metric
measured the wrong quantity. The corrected metric (iter-over-iter L1
delta of normalized `cum_strategy_flop`) measures CFR's actual
convergence signal. The pattern: when a measurement returns
something inside the trusted range, ask which trusted regime it
landed in (linear-N×ULP accumulation vs sqrt-statistical for f32
floor; or trivial-iter-1 vs developed-dynamics for CFR convergence).
Scale discrimination is the f32-precision instance of this; the
strategy-vs-cum_strategy correction is the convergence-signal
instance. Both belong to the standing "agreement between un-anchored
sides is not correctness" discipline.
