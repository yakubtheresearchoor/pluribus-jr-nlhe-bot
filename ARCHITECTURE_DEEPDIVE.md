# surstromming-solver-cuda — Architecture Deep-Dive

Pluribus-style poker solver (blueprint + depth-limited real-time search), Rust + Metal.
Focus: preflop looseness problem and the per-iteration cost bottleneck.

## 1. Big picture / how the pieces connect

```
                PREFLOP CFR LAYER (169-class infosets)
                ──────────────────────────────────────
   PreflopVectorCfr (preflop_cfr.rs)
     • DCFR over 169 preflop hand classes, button-first action order
     • computes strategy, reach, bottom-up CFV + regret update
                │
                │ at each preflop→flop CHANCE NODE (the "seam"):
                ▼
   PostflopValueOracle trait (postflop_oracle.rs)
     • flop_root_cfv_for_cell(flop, combo_ranges, traverser, SeamCell, folded_mask)
     • implementations:
        - UnabstractedPostflopOracle : fresh converged postflop CFR (ground truth, slow)
        - BucketKeyedOracle          : frozen per-(bucket,flop,trav) cache  ◀── used by joint_solve/blueprint
        - ClosureOracle              : test shim
                │
                │ the per-flop work is:
                ▼
   BucketedFlopCfr (bucketed_flop_cfr.rs) + Metal BucketedNativeGpu (gpu_metal/)
     • GS14/quantile hand-strength buckets, Design1Collapsed multiway terminal
     • DCFR over the bucketed flop→turn→river tree, seeded runout (nt×nr)
```

**Two-layer structure.** Preflop lives in the lossless 169-class space; postflop lives
in per-combo (nh≈1176) space, abstracted into B buckets per street (B=8–15). The
preflop→flop **chance node** is the seam: preflop class-reach is expanded to combo-reach
(`expand_reach_class_to_combo`), handed to the oracle, and the returned per-combo CFV is
reduced back to per-class (`reduce_cfv_combo_to_class`), then orbit-weighted aggregated
over the 1755 canonical flops.

---

## 2. File-by-file

### `solver-core/src/solver/preflop_cfr.rs` (1546 lines) — the preflop CFR engine
**Does:** holds preflop-side strategy/regret/cum_strategy for all preflop player
infosets; runs one DCFR iteration per traverser.
**Key structs:** `PreflopVectorCfr { strategy, regrets, cum_strategy, local_offset,
blocking_matrix, debug_emit }`. Storage stride = `MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES`
per infoset, indexed `[a*nc + c]`.
**Key methods:**
- `compute_preflop_reach` — top-down reach; **the Layer-1 fix lives here** (default
  reach = 1/169 sum-1 probability, NOT 1.0 counts — see §4).
- `chance_cfv_for_traverser` — routes folded traversers to fold machinery, live
  traversers to the oracle, then `weight_continuation` (**the Layer-2 fix**, §4).
- `run_one_iteration` / `run_one_iteration_shared_chance` (frozen-oracle collapse,
  same-key chance nodes share one oracle call) / `run_one_iteration_shared_chance_cached`
  (iter-1 expansion cached, reused for frozen oracles — turns iters 2+ into pure
  bottom-up).
- `bottom_up_preflop_for_traverser` — traverser node: strategy-weighted avg + DCFR
  regret/cum update (α/β/γ from `DcfrParams`); opp node: plain sum (factored convention).
- `make_bootstrap_terminal_value_fn_multiway_pairwise` — production preflop terminal fn
  using the pairwise blocking approximation.
**Connects:** consumes `PreflopChanceTable` + `PostflopValueOracle`; is consumed by
`joint_solve.rs`, `preflop_scale_probe.rs`, blueprint/preflop runners.

### `solver-core/src/solver/preflop_terminal.rs` (518 lines) — preflop fold-terminal valuation
**Does:** per-class CFV at preflop fold terminals; the "frozen/bucketed" postflop
valuation DOES NOT live here (this is preflop-only: uncontested-pot fold terminals).
**Key pieces:**
- `joint_class_tuple_non_blocking_fraction` — brute-force joint card-blocking across N
  classes (exact, O(Π class sizes), the 3-AA-needs-6-aces-impossible trap that kills the
  pairwise approximation at N≥3).
- `preflop_fold_terminal_cfv_multiway` — exact multiway, **169^5 ≈ 1.4e11 tuples at
  6-max dense** → intractable at iter 1.
- `preflop_fold_terminal_cfv_multiway_pairwise` — **the pairwise approximation used in
  production**: `v[c] = chip_delta × Π_i(Σ_c opp_reach[i][c]·M[c,c'])`, ignoring opp-opp
  cross-blocking. ~10^5 FLOPs/terminal (~10^8× cheaper than exact). Over-estimates
  non-blocking tuples; exact at N=2.
- `build_class_blocking_matrix` — the 169×169 non-blocking-fraction matrix (constant).
- `preflop_fold_terminal_chip_delta[_from_state]` — traverser chip gain from
  contributions/fold_mask (zero-sum across players).
**Connects:** the blocking matrix + pairwise fn are used by `PreflopVectorCfr` (fold
terminals AND `weight_continuation`), `make_bootstrap_terminal_value_fn`, probes.

### `solver-core/src/solver/preflop_start_game.rs` (1738 lines) — preflop→flop chance integration
**Does:** the seam math. `PreflopChanceTable` (1755 canonical flops, orbit sizes,
per-player class weights); orbit-weighted `chance_probability_flop` (lossless —
sums to 1 exactly); `expand_reach_class_to_combo` / `reduce_cfv_combo_to_class`
(probability-mass conservation / per-class averaging); `aggregate_preflop_chance`
(orbit-weighted sum over canonicals); stratified canonical subsetting; the real
per-flop CFV computation (`compute_v_flop_at_root_converged`).
**Why lossless:** the 169-class layout + flop suit-isomorphism make the orbit-weighted
canonical sum bit-exactly equal the uncanonicalized 22,100-flop sum (P1.5-pre anchor).
**Connects:** the foundation the oracle + preflop CFR build on.

### `solver-core/src/solver/postflop_oracle.rs` (535 lines) — the preflop↔postflop seam trait
**Does:** defines `PostflopValueOracle` trait + implementations.
**Key:** `SeamCell { live, commit, pot }` — the flop-entry game context; `bucket_key`
= (live, floor(log2(SPR)/0.25)). **This is the "frozen/bucketed postflop state" key.**
- `BucketKeyedOracle<S>` — caches per `((live,bin), flop, traverser)`; `refresh_every=0`
  ⇒ fully frozen (joint_solve), >0 ⇒ periodic refresh. CALLER CONTRACT: answer depends
  only on (flop, traverser, key) — the shared-chance collapse rests on this.
- `bucketed_live_subset_source` — the keystone frozen source: solves the LIVE-SUBSET
  game ONCE against a STRUCTURAL UNIFORM range (reach-independent), live≥3 via
  `BucketedFlopCfr::run_all_root_cfv`, live==2 via exact converged solver.
**Connects:** every preflop iteration asks this trait for per-combo flop-root CFV.

### `solver-core/src/abstraction/postflop_buckets.rs` (1131 lines) — GS14 bucketing
**Does:** the per-canonical-flop information abstraction (potential-aware imperfect-recall).
- River: 50-bin equity histograms, 1-D EMD k-means.
- Turn: histogram over river clusters; Flop: histogram over turn clusters (GS14
  backward recursion, run per-flop where cheap).
- `emd_1d` (linear CDF scan) + `emd_exact_general` (min-cost flow, ≤64 bins);
  `kmeans_emd` (k-means++ , N restarts, lowest WCSS).
- **Counts are COST-driven (B=8–15), NOT Pluribus's 200**, because the multiway terminal
  is O(B^(K+1)).
**Documented limitations:** (1) the instrument wall — exact exploitability scoring is
O(nh^(K+1)), research-scale only; production quality ⇒ head-to-head duplicate play.
(2) Relation-blocking correlation killed Design 2 (factored-over-buckets): pairwise
factorization dropped the opp-opp coupling the regret loop amplifies → tripled
equilibrium damage (+4.89→+13.75% pot). Design1Collapsed (brute-over-buckets,
control-flow-only collapse) is what shipped.

### `solver-core/src/abstraction/preflop_class.rs` (343 lines) — lossless 169-class layout
13 pairs (idx 0..13, AA=0) + 78 suited (13..91) + 78 offsuit (91..169). `from_combo`,
`class_combos`, `expansion` (combos compatible with a flop). Anchored by orbit-weighted
= uncanonicalized tests.

### `solver-core/src/bin/joint_solve.rs` (329 lines) — **WHERE THE 219h COST LIVES**
**Two-phase parallel factory:**
- **PHASE 1 (fill):** solve every (representative cell-key × canonical flop) subgame
  ONCE, in parallel — live-2/all-in/live-6 on CPU (rayon), live-3/4/5 on GPU (one
  MetalContext per worker thread, `BucketedNativeGpu`). Banked into a `Mutex<HashMap>`.
- **PHASE 2:** preflop CFR (`PreflopVectorCfr::run_one_iteration`) against the frozen
  pre-solved cache via `BucketKeyedOracle::new(stack, 6, refresh_every=0, …)`.
**Per-family solve `solve_cell`:** live-2 CPU exact; live-3/4/5 GPU bucketed (cell_nb =
15 for live≤4, 8 for live-5); live-6/all-in CPU equity rollout. 1×1 seeded runout.
**Cost (see §5):** PHASE 1 ≈ one fill ≈ 16–25h; PHASE 2 iters cheap (frozen).

### `solver-core/src/bin/blueprint_runner.rs` (504 lines) — blueprint generation
Banks per-flop `flop_NNNN.bp` artifacts (SSBP1 format: JSON header + cum_flop/turn/river
f32 sections + bucket maps). Per-FAMILY seam arm (`BP_LIVE`) solves
`production_game_v1().flop_seam_config(live, commit, pot, …)` — the deployable
per-player-count blueprint selected at play time by live count. Resumable (.tmp+rename,
skip existing). Quantile B=8, 34 DCFR iters, 1×1 seeded runout. Also emits the CFV bank
(`cfv/L{live}_S{bin}/cfv_NNNN.f32`) alongside strategy.

### `solver-core/src/solver/mccfr.rs` (421 lines) — generic CPU MCCFR (depth-limited search core)
**Does:** vanilla CFR with traverser-reach weighting; `freeze_node` (depth-limited search
— frozen nodes play blueprint strategy, not updated) and `set_lambda` (QRE / quantal
response mode, σ∝exp(λ·cfv)). This is the **real-time depth-limited search engine**, not
the preflop solver.

### `play-harness/src/v1_seam.rs` (624 lines) — the preflop↔postflop match-play seam
**Does:** plays the v1 seam families directly (one family per game, seeded), built ON TOP
of `clean_rules::settle_pots`. Dead money D = pot − live·commit is free to live players
(Σ nets = D − rake). AIVAT runout control-variate (exact Rao-Blackwellization over
turn+river). `SeamBlueprint` = research-scale exact (FlopStartVectorCfr) solve of a seam
family. Synthetic seat policies (CheckFold/Aggressive/Mixed/EquityRollout). This is the
quality-measurement harness (head-to-head duplicate play, the instrument the bucketing
docs say replaces production-scale exploitability).

### `solver-core/src/solver/config.rs` — **stub** (`pub struct SolverConfig;`). No real config object.

### `solver-core/src/bin/mccfr_probe.rs` (1725 lines) — the connected-MCCFR investigation (§6)
The `mccfr-cosolve-probe` branch's main artifact. Contains: `ShrunkGame` (per-cell
postflop subgame), the batched external-sampling `Mccfr` engine (with Pluribus pruning +
VR-MCCFR control variate), and `ConnectedHu` (the preflop→flop→postflop single-trajectory
co-solve). Plus a true-best-response anchor for convergence certification.

### `solver-core/src/bin/preflop_scale_probe.rs` (135 lines) — looseness root-cause localization
The diagnostic that proved the Layer-2 continuation-weighting bug: feeds a known chip-scale
synthetic continuation (±C per AA/72o at every chance leaf) through the real bottom-up +
production terminal fn, sweeping C, reading node-0 action values. Shows terminals-only vs
unweighted-continuation vs derived-weighted-continuation on the same scale.

---

## 3. How the blueprint is built / what "frozen postflop state" means

**Blueprint build (blueprint_runner):** for each of 1755 canonical flops, build a
postflop flop-start tree for the chosen family (`flop_seam_config(live, commit, pot)`),
build the GS14/quantile bucketing, run `BucketedFlopCfr` DCFR 34 iters on GPU, bank
cum_strategy + maps + CFV. Resumable, parallel.

**"Frozen postflop state":** `BucketKeyedOracle` with `refresh_every=0`. The oracle solves
each `(SeamCell.bucket_key = (live, log2-SPR-bin), canonical_flop, traverser)` cell ONCE
against a **structural uniform range** (reach-independent) and never re-solves. The preflop
CFR then runs against this frozen cache. Two consequences:
1. **Cheap preflop iters** — PHASE 1 fill (one-shot) + nearly-free PHASE 2 iters.
2. **The preflop "sees" a uniform, never-folding opponent field** — which is the root of
   the AA-limp / looseness residual (§4). A live co-solve (opponents that actually fold)
   is what fixes AA-limp, but it's the 219h wall (§5).

---

## 4. The preflop looseness problem (commit fb89df8)

The deployed v1 preflop was unusable — raised every hand, ~no differentiation
(AA≈72o). Root cause was **NOT** rake/ante/defense/range-insensitivity/non-convergence
(all refuted) — it was **two compounding CFV-scale bugs**, each traced to the line:

**LAYER 1 — opponent reach init was unnormalized COUNTS** (in `compute_preflop_reach`,
default path): reach[p][c] = 1.0 (sum 169) instead of 1/169 (sum-1). The multiway
fold/showdown terminal CFV is a reach-weighted SUM over joint opp class-tuples, so 1.0
counts scaled it ~`n_classes^num_opp` (169^5 ≈ 1.4e11) — a hand-INDEPENDENT mass that
drowned the chip-scale, hand-DIFFERENTIATED continuation, leaving only the ~0.3%
card-blocking ripple → AA≈72o → all-raise.
**Fix:** seed uniform range 1/n_classes (sum-1) so ΣΠ reach ≈ 1, making the fold terminal
chip-scale and matching the continuation.

**LAYER 2 — the continuation was injected reach-UNWEIGHTED** (in
`chance_cfv_expansion_inner`/the bottom-up). After Layer 1, the solver flipped to
**limp-EVERYTHING**: the oracle's reach-INDEPENDENT per-hand EV was summed by the factored
bottom-up over EVERY opponent-action path → ×(path count), measured ×1 to ×136,700,
action-dependent (limp keeps the most opponents in → most paths → biggest over-count →
limp wins). The fold TERMINALS were reach-weighted; the continuation was not.
**Fix:** `weight_continuation()` routes the continuation through the SAME multiway reach
machinery the terminals use (`preflop_fold_terminal_cfv_multiway_pairwise` with unit
chip_delta), converting the conditional per-hand EV to a counterfactual value on the
terminals' scale. Same-node-same-weight holds by construction. The oracle's frozen
reach-independent contract is unchanged — this is the CONSUMER applying its live reach.

**Converged result:** looseness resolved — trash folds 100%, hands differentiate (T9s
raises ~62%), no all-raise, no limp-everything.

**KNOWN, TRACED, ARCHITECTURAL (NOT fixed — it's not a solver bug):** in the frozen game's
equilibrium **AA limps**. Traced: AA call-vs-raise gap only ~2% (near-indifference, not
domination); steal-through shows BB defends 100% → AA's raise steal value ~0; decomposition
shows limp continuation 76.44 > raise 74–75. It is NOT a Layer-2 residual (reach-weighting
DOWN-weights multiway, pushing AA toward raising, yet AA limps). So vs uniform
never-folding opponents, field-maximization (limp) is genuinely ~optimal.
**A deployable preflop (AA raises) requires the live co-solve** — confirmed by
cosolve_premise_probe (commit dabfa8b): a fold-capable opponent reverses multiway−HU from
+4.66→−9.36 (dry board), flipping AA limp→raise.

---

## 5. The 219h/iter cost bottleneck

**The user's "219h/iter" was an EARLIER, INFLATED extrapolation.** Commit 6b1d3c2
**corrected** it: the probe/bench timed an atypically expensive cell (commit=2,pot=12);
real fill cells are (commit=10,pot=50)-ish and 5–9× cheaper.

**Corrected ground truth (from banked `.bp` headers, 1 thread):**
| family | cost | fill time |
|---|---|---|
| live-3 @B15 | 0.103 s/solve | 1.3 h |
| live-4 @B15 | **0.939 s/solve** ← priciest (bucket count > player count) | 11.0 h |
| live-5 @B8  | 0.611 s/solve | 7.1 h |
| live-2 CPU  | ~0.40 s | 5.1 h |
| **TOTAL** | | **~25h/iter (1 thread), ~13h (2 threads)** |

**So per-iter ≈ the fill (~16–25h).** This is the **frozen-oracle** joint_solve cost:
PHASE 1 = one fill + PHASE 2 = cheap iters. The cost is dominated by the live-3/4/5 GPU
multiway bucketed solves (`nb^num_opp` terminal enumeration, Design1Collapsed), with
live-4@B15 the single priciest path.

**The 219h figure corresponds to the LIVE CO-SOLVE** (re-solving postflop every preflop
iteration, N×fill, to get correct AA behavior). That's the wall the frozen-oracle avoids
at the cost of the AA-limp residual. The whole `mccfr-cosolve-probe` branch (§6) was built
to find a way around this wall.

---

## 6. The bucketing limitations + the connected-MCCFR arc (branch mccfr-cosolve-probe)

**Bucketing limitations (postflop_buckets.rs docs):**
- O(B^(K+1)) multiway terminal → B=8–15 (cost-driven), not Pluribus 200.
- Identity gate (B=nh, singleton buckets) STRUCTURALLY CANNOT see within-bucket recall
  drift — only the head-to-head quality harness can.
- Design 2 (factored-over-buckets, O(K·B²), the road past B=50–100) DIED: dropping
  opp-opp relation-blocking coupling tripled equilibrium damage (+4.89→+13.75% pot).

**The connected-MCCFR investigation arc** (one-thread-per-trajectory, preflop→flop→postflop
co-adapting — structurally NO N×fill). Verdict at every lever, by honest measurement:
- **Wall-escape (external sampling skips nb^num_opp enumeration):** ~2× net — eaten by the
  outcome-sampling variance it introduces.
- **Pluribus pruning:** correct ceiling is the *free* setting (prune only CFR+-floored≈0
  actions) = ~1.64× at live-3, eroding with depth; any positive threshold prunes
  low-but-good actions → converges to a WORSE plateau (the anchor caught it: plateau 6.72
  vs 1.98 with tiny regret bound — fast-but-wrong).
- **VR-MCCFR control variate:** ~0 variance reduction — the dominant variance is the
  OUTCOME sampling (traverser hand + showdown tuple), which is coupled to the wall-escape;
  a baseline that removed it would have to enumerate the showdown = re-incur the wall.
- **Combined:** ~2.9×, **far short of the ~10× rebuild-justifying threshold**.
- **ConnectedHu** (final commits fd5f0f8, bb9893e): threads ONE external-sampling
  trajectory through preflop→flop→postflop, updating regret at BOTH layers; VALIDATED
  (strength gate monotone AA>.982>…>87s .699, AA plateaus) with runout sampling.

**FINAL VERDICT (db7e8bc):** **DON'T REBUILD to batched MCCFR.** Optimize DCFR-in-place:
bucket live-2 with a search backstop, warm-start across preflop iters, reach-weight
multiway. Comparable gain, no rebuild, no new Metal kernel. The current working branch is
`dcfr-inplace-fidelity`.

---

## 7. Summary of the core tension

| | frozen oracle (joint_solve) | live co-solve |
|---|---|---|
| per-iter cost | ~16–25h (1 fill + cheap iters) | ~219h (N×fill) |
| AA behavior | **limps** (uniform never-folding field is genuinely ~optimal) | **raises** (fold-capable opponents) |
| preflop quality | undifferentiated/loose residual | deployable |

The looseness CFV-scale bugs (Layer 1 reach-init, Layer 2 continuation-weighting) are
FIXED. The remaining AA-limp is an architectural property of the frozen game, not a solver
bug, and fixing it (live co-solve) is exactly the 219h wall. The branch concluded
connected MCCFR won't carry the load; the path forward is DCFR-in-place optimization.
