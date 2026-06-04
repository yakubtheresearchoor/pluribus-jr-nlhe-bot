# Phase B: exhaustive output-write site list (production kernels only)

> ## ⚠️ FIFTH-ITERATION INVENTORY FAILURE — this document is INCORRECT
>
> The site list below was constructed by exhaustive output-write
> enumeration in vcfr_bottom_up_batched. After applying rake at F1 and
> observing the HU gate did NOT move (residual stayed at exactly
> 0.09375006), the standing question applied: "does this code actually
> execute?" Answer: NO.
>
> All inline payoff sites F1–G4 (and the sorted_sweep_with_components
> sites 10/13 from Site (d) part 2) are inside an `if (false)` gate at
> vcfr.metal line 1728. Comment at line 1727 says "Legacy showdown
> evaluation code removed; replaced by brute-force above" — but the
> code wasn't deleted, just gated.
>
> **PRODUCTION ROUTING IN vcfr_bottom_up_batched**: every terminal
> goes through the unconditional `multiway_brute_force_showdown` call
> at line 1719, then `return;` at 1725. The entire 600-line block
> below 1728 is DEAD.
>
> **Lesson for the standing method**: output-write enumeration is
> NECESSARY BUT NOT SUFFICIENT. Each enumerated site must ALSO be
> verified REACHABLE in production code paths (the standing question
> applied to enumeration). The simplest verification: insert a deliberate
> compile-break (or NaN write) at the site and confirm tests detect it.
> A site whose disturbance changes nothing is dead.
>
> **What's actually rake-correct in production** (verified by unit tests
> against multiway_brute_force_showdown's 5 branches):
>   - Sites (a) K=2 all-equal brute: unit test 0.0
>   - Sites (b) K=2 fold-win fast: unit test 0.0
>   - Sites (c) K=2 unequal/side-pot: unit test 0.0
>   - Site (d) K=1 HU per-level: unit test 0.0 (incl tie-band)
>   - Site (e) K≥3 factored: STILL OPEN (rake-free)
>
> **Phase B remaining work**: site (e) only (K≥3 factored in
> multiway_brute_force_showdown), plus production-solve instrumentation
> as the standing completeness check. The "5+ inline sites" finding
> dissolved on the standing-question check.
>
> **HU gate residual (0.09375006) still unexplained**: the multiway
> helper is fully rake-correct for K=1, so the residual is NOT a rake
> bug in the K=1 path. Possible sources: site (e) doesn't affect HU
> (K=1 not K≥3); chance integration; CFR propagation through decision
> nodes; numerical artifact. Needs investigation as a separate finding,
> NOT bundled with rake.
>
> The site list below is preserved for reference but should be read
> understanding that everything except line 1719 (multiway helper call)
> is dead code.
>
> ---


Per the lead's directive: per-site rake-mirror keeping the inline fast-path
optimizations, with output-write enumeration as the standing method and
production-solve instrumentation as the permanent CI completeness check.

## Production kernel scope (verified via `create_pipeline` grep)

The only Metal kernels actually instantiated as pipelines in production
Rust code (solver-core/src/gpu_metal/flop_solver.rs lines 222-232):

  - `vcfr_compute_strategies` / `_batched` — strategy compute, not payoff
  - `vcfr_init_reach`, `vcfr_top_down_reach`, `vcfr_seed_reach` — reach
  - `vcfr_zero_buffer` — initialization
  - `vcfr_chance_accumulate`, `vcfr_chance_finalize`,
    `vcfr_chance_accumulate_grouped` — chance integration (aggregates
    already-rake-applied terminal CFVs; no rake needed at this layer)
  - **`vcfr_bottom_up`** (lines 912-1073) — flop zone terminal eval
  - **`vcfr_bottom_up_batched`** (lines 1635-2318) — turn/river zone
    terminal eval

DEAD kernels (defined in vcfr.metal but NOT in any create_pipeline):
`vcfr_streaming_level`, `factored_showdown_unified`, all `debug_*`,
all `k*_microbench`. These do not run in production solves and do
not need rake.

## Exhaustive enumeration by output-write pattern

Searched `out[h] = `, `out[h] +=`, `out[h] -=`, `cfv[*] = `,
`cfv_out[*] = `, `returned_cfv[*] = ` patterns in the two production
payoff-computing kernels. Each row classified by purpose.

### vcfr_bottom_up (flop zone)

| # | Line  | Pattern                          | Purpose                         | Status         |
|---|-------|----------------------------------|---------------------------------|----------------|
| 1 | 987   | `out[h] = local_out[h]`          | multiway helper result          | RAKE-CORRECT   |
| - | 998   | `cfv[...] = 0.0f`                | reset/init                      | non-payoff     |
| - | 1059  | `out[h] = cfv_avg[h]`            | chance integration              | non-payoff (aggregates rake-applied CFVs) |

**Flop zone uses the multiway helper universally** — no inline payoff
sites. Site 1 was closed when site (b)+(c) closed multiway helper rake.

### vcfr_bottom_up_batched (turn/river zone)

| #  | Line  | Pattern                                                        | Purpose                                          | Status         |
|----|-------|----------------------------------------------------------------|--------------------------------------------------|----------------|
| 2  | 1719  | `out[h] = local_out[h]`                                        | multiway helper result                           | RAKE-CORRECT   |
| 3  | 1722  | `out[h] /= num_combinations`                                   | normalization                                    | non-payoff (preserves rake) |
| 4  | 1772  | `out[h] = payoff * cfreach`                                    | **inline fold-win** (general)                    | **RAKE-FREE** |
| 5  | 1775  | `out[h] = 0.0f`                                                | zero-reach path                                  | non-payoff (no opp reach → no CFV) |
| 6  | 1778  | `out[h] /= num_combinations`                                   | normalization                                    | non-payoff   |
| 7  | 1820  | `out[h] = payoff * (opp_reach_sum - ...)`                      | **inline all-equal fold-win** (num_active_opp=0) | **RAKE-FREE** |
| 8  | 1823  | `out[h] = 0.0f`                                                | zero-reach path                                  | non-payoff   |
| 9  | 1826  | `out[h] /= num_combinations`                                   | normalization                                    | non-payoff   |
| 10 | 1872  | `out[h] = pot_size * sweep_net[h] - rake * (...)`              | sorted_sweep with rake (Site (d) part 2 edit 1) | RAKE-CORRECT |
| 11 | 1932  | `out[h] = (sp/np+c_t) * (K * cum_weaker - eff_total)`          | **inline multiway probabilistic** (K≥2 all-equal showdown) | **RAKE-FREE** |
| 12 | 1959  | `out[h] = payoff` (constant)                                   | **inline HU fold-win** (num_active_opp=0, np=2)  | **RAKE-FREE** |
| 13 | 1995  | `out[h] = half_pot * sweep_net[h] - rake * (...)`              | sorted_sweep with rake (Site (d) part 2 edit 2) | RAKE-CORRECT |
| 14 | 2002  | `out[h] = 0.0f`                                                | per-level brute init                             | non-payoff (zero before accumulation) |
| 15 | 2049  | `out[h] += float(pot_at_level)`                                | **per-level dead-money return** (eligible_opp=0) | **RAKE-FREE** (single-traverser-eligible path) |
| 16 | 2091  | `out[h] += float(pot_at_level) * cfreach` (forward)            | **per-level sweep wins**                         | **RAKE-FREE** |
| 17 | 2116  | `out[h] -= float(pot_at_level) * cfreach` (backward)           | **per-level sweep losses**                       | **RAKE-FREE** |
| 18 | 2181  | `out[h] += float(pot_at_level) * (...)`                        | **per-level multiway product**                   | **RAKE-FREE** |
| 19 | 2191  | `out[h] -= (sp/np + c_t)`                                      | **stake subtraction at end of per-level brute**  | non-payoff (final stake adjust; rake should already be in the accumulated sum) |
| 20 | 2196  | `out[h] /= num_combinations`                                   | normalization                                    | non-payoff   |
| 21 | 2204  | `out[h] = 0.0f`                                                | chance accumulator init                          | non-payoff   |
| 22 | 2208  | `out[h] += cfv_o[int(child) * nh + h]`                         | chance integration                               | non-payoff (aggregates rake-applied CFVs) |

## Site rake-status summary

**RAKE-CORRECT payoff sites (4)**:
  - 987 (vcfr_bottom_up multiway)
  - 1719 (vcfr_bottom_up_batched multiway)
  - 1872 (sorted_sweep_with_components, Site (d) part 2)
  - 1995 (sorted_sweep_with_components, Site (d) part 2)

**RAKE-FREE payoff sites that NEED rake (8 logical blocks)**:

| ID  | Lines       | Logical block name                                | CPU reference path                  |
|-----|-------------|---------------------------------------------------|-------------------------------------|
| F1  | 1772        | Inline fold-win (general)                         | showdown.rs ~484-548 (fast path)    |
| F2  | 1820        | Inline all-equal fold-win                         | showdown.rs ~484-548                |
| F3  | 1932        | Inline multiway probabilistic K≥2 all-equal       | showdown.rs ~699-779 (rake_per_unit_stake) |
| F4  | 1959        | Inline HU fold-win (np=2, num_active_opp=0)       | showdown.rs ~484-548                |
| G1  | 2049        | Per-level dead-money return (eligible_opp=0)      | showdown.rs n_elig==0 branch (no rake, just contribs return) |
| G2  | 2091, 2116  | Per-level sweep wins/losses (np=2 unequal path)   | showdown.rs sorted_sweep_with_rake_components |
| G3  | 2181        | Per-level multiway product (np≥3 unequal)         | showdown.rs ~870-910 (main-pot-only) |
| G4  | 2191        | Stake subtraction at end of per-level brute       | (no rake here; rake already in accumulated pot_at_level via G2/G3) |

Note: G4 (line 2191) is not itself a payoff arithmetic but the final
`net = cash - stake` step. Its correctness depends on G2/G3 having
applied rake to the per-level cash accumulation upstream. Listed for
completeness; no separate rake edit needed if G2/G3 are correct.

G1 (line 2049) is the "no eligible opponents at this level" path: dead
money is returned to the contributor, no winner, no rake. CPU does the
same. Listed for completeness; no rake edit needed.

So actual sites requiring rake math: **F1, F2, F3, F4, G2, G3** (6 logical
blocks across 7 line locations).

## Per-site DoD (matches K=2 cluster pattern)

For each of F1–F4, G2, G3:
  1. Add a debug-kernel unit test routed to that block's specific
     scenario (np / fold_mask / contribs combination that routes the
     kernel through THIS block exclusively). Test fails today by the
     predicted rake amount × reach.
  2. Apply rake math at the block, mirroring CPU reference exactly.
  3. Unit test goes to 0.0 (DoD).
  4. site_d_hu_rake gate progresses toward f32 floor as each block
     closes (which blocks contribute is empirically determined).
  5. site_e_4p_factored_rake gate progresses similarly for K≥3 blocks.

## Step 3: production-solve instrumentation (load-bearing per the lead)

After all per-site closures, add permanent CI instrumentation:
  - Per-terminal debug buffer in the kernel that records "rake-applied"
    (or "no-flop-no-drop skip") at every site that produces a CFV
  - Host-side download + assert that every terminal evaluated in a
    representative production solve recorded rake-considered
  - A site the enumeration missed would NOT set the marker → assertion
    fires → gap revealed
  - Standing question applied to the instrumentation: confirm it fires
    the expected number of times for a solve whose terminal count is
    known, so the instrumentation isn't itself false-green

This permanent CI test replaces the structural completeness guarantee
that the helper-consolidation refactor would have provided. The
trade-off (per the lead): keep the inline fast-path optimizations,
accept that any future change can reintroduce a rake-free site, the
standing instrumentation catches it immediately.

## Carry from prior commits

- Site (d) part 2 already closed sites 10 and 13 (sorted_sweep with
  components). Those stay closed; they're in the rake-correct column.
- Site (e) factored is NOT in vcfr.metal at any production output-write
  site I can find — the factored path appears only in
  `factored_showdown_unified` (DEAD). K≥3 production showdowns route
  through the per-level multiway block (G3, line 2181). So site (e)'s
  "two locations" finding from the user's earlier carry resolves to:
  the dead factored_showdown_unified location doesn't matter (not in
  production), the live K≥3 path is G3 here. Will validate this
  empirically as G3 closure approaches f32 floor on site (e) gate.

## Surgery order

Easiest-first (matches prior K=2 cluster pattern):
  1. F4 (constant HU fold-win) — single value
  2. F1, F2 (inline fold-win variants) — single payoff formula
  3. F3 (inline multiway probabilistic K≥2) — rake_per_unit_stake
  4. G2 (per-level sweep, the trickiest — tie-band +reach correction
     in inline form; tie-band unit test defends this)
  5. G3 (per-level multiway product) — main-pot-only at li==0
  6. Production-solve instrumentation as permanent CI gate
