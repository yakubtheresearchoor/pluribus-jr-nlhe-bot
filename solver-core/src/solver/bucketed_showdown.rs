//! Design-1 bucketed showdown terminal (Phase B3, step 3).
//!
//! Brute-over-buckets mirror of `side_pot_showdown_cfv_with_rake` for the
//! two dispatch arms that can fire at np ≥ 4:
//!
//!   1. equal contributions, no folds  → mirror of `recurse_eq`
//!      (showdown.rs:829), bucket-tuple enumeration carrying a
//!      (beaten?, tie-count) state distribution;
//!   2. everything else (folds and/or unequal contributions) → mirror of
//!      `recurse_scenario` + per-level cash walk (showdown.rs:1207+),
//!      bucket-tuple enumeration branching on each ACTIVE opponent's
//!      trinary relation to the traverser.
//!
//! np ≤ 3 is deliberately OUT OF SCOPE (asserted): those dispatch arms
//! (HU sorted-sweep, 3-player constant-payoff fast path) use sweep /
//! inclusion-exclusion arithmetic whose float-add order a bucket-matrix
//! formulation cannot reproduce bit-exactly — and K ≤ 2 terminals are
//! polynomially cheap, so bucketing buys nothing there. The fifth power
//! this phase attacks lives exclusively in the two arms above.
//!
//! === Bit-exact identity argument (B = nh, singleton buckets) ===
//!
//! At B = nh with unit initial weights, every fraction in
//! `BucketedRunoutTables` is EXACTLY 0.0 or 1.0 (IEEE: 1.0/1.0 == 1.0,
//! 0.0/1.0 == 0.0). The degeneracy is then pure control flow:
//!
//!   - skip-on-zero (`if f == 0.0 { continue }`) reproduces the exact
//!     code's card-mask conflict skips — in 2-card hold'em, joint
//!     conflict with a set of hands ⇔ pairwise conflict with at least
//!     one member, so the product of pairwise binary fractions equals
//!     the joint mask check;
//!   - `x * 1.0` is bit-preserving, so the weight chain
//!     `w × r × Π(1.0)` equals the exact `reach_so_far * r` bit-for-bit;
//!   - relation selection by binary fraction reproduces the exact
//!     strength-comparison branches, and the leaf payoff expressions are
//!     copied verbatim, so the same constants are selected;
//!   - the (beaten, ties) state distribution is a point mass, so every
//!     state-vector add is `0.0 + x == x`, and each leaf contributes the
//!     single `accum += weight * net` in the SAME tuple-enumeration
//!     order (buckets iterated in index order == hands in index order).
//!
//! The gate for this argument is `p1_5_4_bucketing_b3_terminal_singleton`
//! (to_bits equality against `side_pot_showdown_cfv_with_rake`), run
//! BEFORE any walk integration.
//!
//! === Named approximations at B < nh (B1 report, verbatim discipline) ===
//!
//!   (i)  opp-opp joint blocking: pairwise conflict-fraction products
//!        ≠ joint non-blocking fraction (the AA-AA-AA trap,
//!        preflop_terminal.rs:62-77). Zero under singletons.
//!   (ii) within-bucket heterogeneity: per-opponent relation fractions
//!        are mixed independently given the traverser BUCKET, but true
//!        relations correlate through the traverser HAND. This is the
//!        abstraction loss itself — measured by the quality gate, never
//!        silently merged with (i).

use crate::abstraction::postflop_buckets::WtlMatrices;

/// Per-runout bucket-pair fraction tables consumed by the terminal.
///
/// All four are nb×nb, indexed `[b_t * nb + b_o]` (traverser bucket
/// major). Fractions are UNCONDITIONAL over the initial-weight pair
/// mass `wsum[b_t] × wsum[b_o]`, so traverser↔opponent card-removal is
/// absorbed: `f_w + f_t + f_l == f_n ≤ 1`, with the deficit being the
/// conflicting-pair mass.
pub struct BucketedRunoutTables {
    pub nb: usize,
    /// traverser bucket beats opponent bucket (h beats g).
    pub f_w: Vec<f32>,
    /// tie.
    pub f_t: Vec<f32>,
    /// opponent bucket beats traverser bucket (g beats h).
    pub f_l: Vec<f32>,
    /// compatible (non-conflicting) pair-mass fraction = f_w + f_t + f_l.
    /// Doubles as the pairwise blocking fraction for folded opponents
    /// and for opp-opp pairs (compat mass is symmetric).
    pub f_n: Vec<f32>,
}

impl BucketedRunoutTables {
    /// Build fraction tables from the per-runout W/T/L mass matrices and
    /// the per-bucket initial-weight sums (over hands ALIVE at this
    /// runout — the same alive set `compute_wtl_for_runout` used).
    ///
    /// At singleton buckets with unit weights every denominator is 1.0
    /// and every mass is 0.0 or 1.0, so the fractions are exactly
    /// binary — the identity-gate degeneracy.
    pub fn from_wtl(wtl: &WtlMatrices, bucket_weight_sums: &[f64]) -> Self {
        let nb = wtl.nb;
        assert_eq!(bucket_weight_sums.len(), nb);
        let mut f_w = vec![0.0f32; nb * nb];
        let mut f_t = vec![0.0f32; nb * nb];
        let mut f_l = vec![0.0f32; nb * nb];
        let mut f_n = vec![0.0f32; nb * nb];
        for bt in 0..nb {
            for bo in 0..nb {
                let denom = bucket_weight_sums[bt] * bucket_weight_sums[bo];
                if denom == 0.0 {
                    continue; // empty bucket at this runout → all-zero row
                }
                let i = bt * nb + bo;
                let w = wtl.w[i] / denom;
                let t = wtl.t[i] / denom;
                let l = wtl.l[i] / denom;
                f_w[i] = w as f32;
                f_t[i] = t as f32;
                f_l[i] = l as f32;
                f_n[i] = (w + t + l) as f32;
            }
        }
        BucketedRunoutTables { nb, f_w, f_t, f_l, f_n }
    }
}

/// Relation codes carried by the per-level arm.
const REL_WIN: u8 = 0; // traverser beats this opponent
const REL_TIE: u8 = 1;
const REL_LOSE: u8 = 2; // this opponent beats the traverser
const REL_FOLDED: u8 = 3; // relation never read (blocking only)

/// Maximum opponents supported by the fixed-size state vector
/// (state = beaten + tie-counts 0..=num_opp).
const MAX_OPP: usize = 9;

/// Design-1 bucketed mirror of `side_pot_showdown_cfv_with_rake` for
/// np ≥ 4. Returns per-traverser-bucket CFVs.
///
/// `bucket_reach[oi][b]` must be the per-runout bucket-aggregated reach
/// of opponent slot `oi` (slot order identical to the exact function:
/// player index `oi` if `oi < traverser` else `oi + 1`), aggregated with
/// the SAME per-runout bucket map the `tables` were built from (hands
/// conflicting with the runout excluded).
#[allow(clippy::too_many_arguments)]
pub fn bucketed_showdown_cfv(
    bucket_reach: &[&[f32]],
    tables: &BucketedRunoutTables,
    contributions: &[i32],
    fold_mask: u16,
    traverser: usize,
    num_players: u8,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    flop_seen: bool,
) -> Vec<f32> {
    let nb = tables.nb;
    let num_opp = bucket_reach.len();
    let np = num_players as usize;
    let c_t = contributions[traverser];
    let mut cfv = vec![0.0f32; nb];

    assert!(
        np >= 4,
        "bucketed_showdown_cfv: np = {np} < 4. The np ≤ 3 dispatch arms \
         (HU sweep / 3p fast paths) are out of scope — no fifth power \
         there, and their sweep/I-E arithmetic is not reproducible from \
         bucket matrices."
    );
    assert!(num_opp == np - 1, "one reach slice per opponent slot");
    assert!(num_opp <= MAX_OPP);

    // "No flop, no drop" gate — verbatim from the exact function.
    let (eff_rake_rate, eff_rake_cap) = if flop_seen {
        (rake_rate, rake_cap)
    } else {
        (0.0, 0.0)
    };

    // Dispatch mirror: at np ≥ 4 the exact function reaches exactly two
    // arms — `all_active_equal && fold_mask == 0` → recurse_eq, else →
    // the per-level scenario evaluator. (The constant-payoff fast path
    // is gated num_opp ≤ 2; the np == 2 sweep arm is gated np == 2.)
    let mut all_active_equal = true;
    let mut ref_contrib: Option<i32> = None;
    for p in 0..np {
        if fold_mask & (1u16 << p) != 0 {
            continue;
        }
        let cp = contributions[p];
        if let Some(r) = ref_contrib {
            if cp != r {
                all_active_equal = false;
                break;
            }
        } else {
            ref_contrib = Some(cp);
        }
    }

    if all_active_equal && fold_mask == 0 {
        // ── Arm 1: mirror of the K > 2 recurse_eq path ──
        let num_active_opp = num_opp; // no folds
        let k = num_active_opp as f32;
        let half_pot = starting_pot as f32 / np as f32 + c_t as f32;
        let total_pot: i32 = starting_pot + contributions.iter().sum::<i32>();
        let rake = (total_pot as f32 * eff_rake_rate).min(eff_rake_cap).max(0.0);
        let rake_per_unit_stake = if half_pot > 0.0 { rake / half_pot } else { 0.0 };

        let mut prefix = vec![0u16; num_opp];
        for bt in 0..nb {
            let mut accum = 0.0f32;
            // state[0] = beaten mass; state[1 + j] = mass with j ties,
            // not beaten. Point mass at "0 ties" to start (the exact
            // recursion starts at reach_so_far = 1.0).
            let mut state = [0.0f32; MAX_OPP + 2];
            state[1] = 1.0;
            recurse_eq_buckets(
                0,
                num_opp,
                nb,
                bt,
                &state,
                &mut prefix,
                bucket_reach,
                tables,
                k,
                rake_per_unit_stake,
                &mut accum,
            );
            cfv[bt] = half_pot * accum;
        }
        return cfv;
    }

    // ── Arm 2: mirror of the per-level scenario evaluator ──

    // Pot levels — verbatim precompute from the exact function.
    let mut levels: Vec<i32> = (0..np).map(|p| contributions[p]).collect();
    levels.sort();
    levels.dedup();

    let main_pot_amount: i32 = if levels.is_empty() {
        starting_pot
    } else {
        let num_main_contributors = (0..np)
            .filter(|&p| contributions[p] >= levels[0])
            .count();
        levels[0] * num_main_contributors as i32 + starting_pot
    };
    let main_pot_rake: f32 = (main_pot_amount as f32 * eff_rake_rate)
        .min(eff_rake_cap)
        .max(0.0);

    let traverser_stake = starting_pot as f32 / np as f32 + c_t as f32;
    let traverser_folded = fold_mask & (1u16 << traverser) != 0;

    let opp_player: Vec<usize> = (0..num_opp)
        .map(|oi| if oi < traverser { oi } else { oi + 1 })
        .collect();
    let opp_contrib: Vec<i32> = opp_player.iter().map(|&p| contributions[p]).collect();
    let opp_folded: Vec<bool> = opp_player
        .iter()
        .map(|&p| fold_mask & (1u16 << p) != 0)
        .collect();

    let ctx = Arm2Ctx {
        levels,
        np,
        contributions: contributions.to_vec(),
        starting_pot,
        main_pot_rake,
        traverser_stake,
        traverser_folded,
        c_t,
        opp_contrib,
        opp_folded,
        traverser,
    };

    let mut prefix = vec![0u16; num_opp];
    let mut rel = vec![0u8; num_opp];
    for bt in 0..nb {
        let mut accum = 0.0f32;
        recurse_levels_buckets(
            0,
            num_opp,
            nb,
            bt,
            1.0,
            &mut prefix,
            &mut rel,
            bucket_reach,
            tables,
            &ctx.opp_folded,
            &mut |weight: f32, rel: &[u8]| {
                accum += weight * ctx.net(rel);
            },
        );
        cfv[bt] = accum;
    }
    cfv
}

/// Shared arm-2 per-scenario payoff context. `net(rel)` is the verbatim
/// mirror of the exact per-level closure (showdown.rs:1248-1318), with
/// strength comparisons replaced by relation codes (the cash depends on
/// opponents' strengths ONLY through their trinary relation to the
/// traverser hand). Both Design 1 (brute-over-buckets) and Design 2
/// (factored-over-buckets) call THIS function — the designs differ only
/// in how scenario probabilities are enumerated, never in payoff
/// arithmetic, so payoff drift between designs is impossible by
/// construction. Extracted 2026-06-10 for the Design-2 fork; the
/// singleton bit-exact gate re-verifies Design 1 through the extraction.
struct Arm2Ctx {
    levels: Vec<i32>,
    np: usize,
    contributions: Vec<i32>,
    starting_pot: i32,
    main_pot_rake: f32,
    traverser_stake: f32,
    traverser_folded: bool,
    c_t: i32,
    opp_contrib: Vec<i32>,
    opp_folded: Vec<bool>,
    traverser: usize,
}

impl Arm2Ctx {
    fn net(&self, rel: &[u8]) -> f32 {
        let num_opp = self.opp_folded.len();
        let mut cash: f32 = 0.0;
        let mut prev_l = 0i32;
        for (li, &lev) in self.levels.iter().enumerate() {
            let pc = lev - prev_l;
            let num_contrib = (0..self.np)
                .filter(|&p| self.contributions[p] >= lev)
                .count();
            let mut pot_l = (pc * num_contrib as i32) as f32;
            if li == 0 {
                pot_l += self.starting_pot as f32;
            }
            if pot_l == 0.0 {
                prev_l = lev;
                continue;
            }

            let trav_elig = !self.traverser_folded && self.c_t >= lev;

            let mut elig_count: u32 = trav_elig as u32;
            let mut opp_beats = false;
            for oi in 0..num_opp {
                if self.opp_folded[oi] {
                    continue;
                }
                if self.opp_contrib[oi] < lev {
                    continue;
                }
                elig_count += 1;
                if rel[oi] == REL_LOSE {
                    opp_beats = true;
                }
            }

            if elig_count == 0 {
                if self.contributions[self.traverser] >= lev {
                    let trav_contrib_at_lev = pc as f32
                        + if li == 0 {
                            self.starting_pot as f32 / self.np as f32
                        } else {
                            0.0
                        };
                    cash += trav_contrib_at_lev;
                }
                prev_l = lev;
                continue;
            }

            if !trav_elig {
                prev_l = lev;
                continue;
            }

            if !opp_beats {
                // Traverser is at the eligible max. Tie count = self +
                // eligible opponents tying the traverser (exact code:
                // tied = [h==max] + Σ [g==max]).
                let mut tied: u32 = 1;
                for oi in 0..num_opp {
                    if self.opp_folded[oi] {
                        continue;
                    }
                    if self.opp_contrib[oi] < lev {
                        continue;
                    }
                    if rel[oi] == REL_TIE {
                        tied += 1;
                    }
                }
                let pot_after_rake = if li == 0 {
                    pot_l - self.main_pot_rake
                } else {
                    pot_l
                };
                cash += pot_after_rake / tied as f32;
            }
            prev_l = lev;
        }
        cash - self.traverser_stake
    }
}

/// ═══ Design 2 — factored-over-buckets, O(K·B²) per terminal ═══
///
/// The B1-logged "road past the M4 ceiling", promoted to the production
/// candidate by the B4 step-1 cost gate (Design 1 measured 563× over
/// the everything-at-B prediction at nh=1176, B=15: bucket-tuple
/// enumeration loses all conflict/zero-reach pruning, and arm 2 pays a
/// ~3^K_active relation-branching multiplier).
///
/// Factorization: opponents are treated as INDEPENDENT given the
/// traverser bucket. Per opponent slot, one matvec over the fraction
/// tables collapses its bucket distribution to scalars
///   w_i = Σ_bo r_i[bo]·f_w[bt][bo]   (traverser beats opp i, compat)
///   t_i, l_i analogous; n_i = Σ_bo r_i[bo]·f_n[bt][bo]  (compat mass)
/// Arm 1 then runs the same (beaten, tie-count) DP over opponents with
/// scalars (O(K²)); arm 2 enumerates relation vectors with scalar
/// branch weights (O(3^K_active), ≤243 at 6-max) and calls the SAME
/// Arm2Ctx::net payoff.
///
/// === Named approximation (the Design-2 error) — two layers ===
///
/// Layer 1 (raw factorization): dropping the opp-opp prefix product
/// Π f_n(b_j, b_i) OVER-COUNTS scenario mass (f_n ≤ 1) — same
/// direction as the preflop pairwise precedent
/// (`preflop_fold_terminal_cfv_multiway_pairwise`). Measured: the
/// over-count is ~Π_{i<j} f̄_n ≈ a per-(terminal, bucket) SCALE on
/// cfv (parity fixture: raw deviation ≈ 1490%, within-terminal shape
/// after per-bucket scale removal 0.2-0.5%). The scale VARIES across
/// terminals/buckets/traversers with the reach distribution, and that
/// variance is what the regret loop amplifies: the raw factored
/// terminal measured +4.89% pot equilibrium cost in the B=4 wet-deep
/// A/B — REJECTED at the 0.25% acceptance line.
///
/// Layer 2 (pairwise mass renormalization, applied below): cfv[bt] is
/// multiplied by C(bt) = Π_{i<j} ĉ_ij(bt), where ĉ_ij is the
/// reach-weighted mean opp-opp compat fraction
///   ĉ_ij(bt) = Σ_{bi,bj} r̂_i(bi)·r̂_j(bj)·f_n(bi,bj),
///   r̂_i(bi) ∝ r_i(bi)·f_n(bt,bi)  (traverser-conditioned reach)
/// — O(K²·B²) per bucket, still microseconds. This restores the
/// per-bucket mass scale up to PAIR-INDEPENDENCE (treating opp-opp
/// blocking events as independent across pairs — the residual is
/// triplet+ blocking correlation, the AA-AA-AA error class proper).
/// The renormalized residual is pinned by the parity test; the
/// equilibrium verdict is the A/B.
#[allow(clippy::too_many_arguments)]
pub fn bucketed_showdown_cfv_factored(
    bucket_reach: &[&[f32]],
    tables: &BucketedRunoutTables,
    contributions: &[i32],
    fold_mask: u16,
    traverser: usize,
    num_players: u8,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    flop_seen: bool,
) -> Vec<f32> {
    let nb = tables.nb;
    let num_opp = bucket_reach.len();
    let np = num_players as usize;
    let c_t = contributions[traverser];
    let mut cfv = vec![0.0f32; nb];

    assert!(np >= 4, "bucketed_showdown_cfv_factored: np ≥ 4 only (same scope as Design 1)");
    assert!(num_opp == np - 1);
    assert!(num_opp <= MAX_OPP);

    let (eff_rake_rate, eff_rake_cap) = if flop_seen {
        (rake_rate, rake_cap)
    } else {
        (0.0, 0.0)
    };

    // Dispatch mirror — identical to Design 1.
    let mut all_active_equal = true;
    let mut ref_contrib: Option<i32> = None;
    for p in 0..np {
        if fold_mask & (1u16 << p) != 0 {
            continue;
        }
        let cp = contributions[p];
        if let Some(r) = ref_contrib {
            if cp != r {
                all_active_equal = false;
                break;
            }
        } else {
            ref_contrib = Some(cp);
        }
    }

    // Per-(bt, oi) scalar collapse: one matvec per opponent.
    // probs[oi] = [w, t, l, n] given traverser bucket bt.
    let collapse = |bt: usize| -> Vec<[f32; 4]> {
        (0..num_opp)
            .map(|oi| {
                let mut acc = [0.0f32; 4];
                for bo in 0..nb {
                    let r = bucket_reach[oi][bo];
                    if r == 0.0 {
                        continue;
                    }
                    let i = bt * nb + bo;
                    acc[0] += r * tables.f_w[i];
                    acc[1] += r * tables.f_t[i];
                    acc[2] += r * tables.f_l[i];
                    acc[3] += r * tables.f_n[i];
                }
                acc
            })
            .collect()
    };

    // Layer-2 pairwise mass renormalization: C(bt) = Π_{i<j} ĉ_ij(bt),
    // the reach-weighted mean opp-opp compat fraction per opponent
    // pair, conditioned on the traverser bucket. f64 accumulation (the
    // product spans C(K,2) = 10 factors at 6-max).
    let renorm = |bt: usize, probs: &[[f32; 4]]| -> f32 {
        let mut c = 1.0f64;
        for i in 0..num_opp {
            let ni = probs[i][3] as f64;
            if ni == 0.0 {
                continue; // zero compat mass → cfv[bt] is zero anyway
            }
            for j in (i + 1)..num_opp {
                let nj = probs[j][3] as f64;
                if nj == 0.0 {
                    continue;
                }
                let mut s = 0.0f64;
                for bi in 0..nb {
                    let wi =
                        bucket_reach[i][bi] as f64 * tables.f_n[bt * nb + bi] as f64;
                    if wi == 0.0 {
                        continue;
                    }
                    for bj in 0..nb {
                        let wj =
                            bucket_reach[j][bj] as f64 * tables.f_n[bt * nb + bj] as f64;
                        if wj == 0.0 {
                            continue;
                        }
                        s += wi * wj * tables.f_n[bi * nb + bj] as f64;
                    }
                }
                c *= s / (ni * nj);
            }
        }
        c as f32
    };

    if all_active_equal && fold_mask == 0 {
        // ── Arm 1: (beaten, tie-count) DP with scalars ──
        let k = num_opp as f32;
        let half_pot = starting_pot as f32 / np as f32 + c_t as f32;
        let total_pot: i32 = starting_pot + contributions.iter().sum::<i32>();
        let rake = (total_pot as f32 * eff_rake_rate).min(eff_rake_cap).max(0.0);
        let rake_per_unit_stake = if half_pot > 0.0 { rake / half_pot } else { 0.0 };

        for bt in 0..nb {
            let probs = collapse(bt);
            let c_bt = renorm(bt, &probs);
            let mut state = [0.0f32; MAX_OPP + 2];
            state[1] = 1.0;
            for (oi, p) in probs.iter().enumerate() {
                let [w, t, l, n] = *p;
                let mut new_state = [0.0f32; MAX_OPP + 2];
                if state[0] != 0.0 {
                    new_state[0] += state[0] * n;
                }
                for j in 0..=oi {
                    let s = state[1 + j];
                    if s == 0.0 {
                        continue;
                    }
                    new_state[0] += s * l;
                    new_state[1 + j + 1] += s * t;
                    new_state[1 + j] += s * w;
                }
                state = new_state;
            }
            // Leaf payoffs — same constants as Design 1's arm-1 leaf.
            let mut accum = 0.0f32;
            if state[0] != 0.0 {
                accum += state[0] * -1.0;
            }
            for j in 0..=num_opp {
                let s = state[1 + j];
                if s == 0.0 {
                    continue;
                }
                let net_unit: f32 = if j == 0 {
                    k - rake_per_unit_stake
                } else {
                    let t_f = (j + 1) as f32;
                    (k + 1.0 - t_f) / t_f - rake_per_unit_stake / t_f
                };
                accum += s * net_unit;
            }
            cfv[bt] = half_pot * accum * c_bt;
        }
        return cfv;
    }

    // ── Arm 2: relation-vector enumeration with scalar weights ──
    let mut levels: Vec<i32> = (0..np).map(|p| contributions[p]).collect();
    levels.sort();
    levels.dedup();
    let main_pot_amount: i32 = if levels.is_empty() {
        starting_pot
    } else {
        let num_main_contributors = (0..np)
            .filter(|&p| contributions[p] >= levels[0])
            .count();
        levels[0] * num_main_contributors as i32 + starting_pot
    };
    let main_pot_rake: f32 = (main_pot_amount as f32 * eff_rake_rate)
        .min(eff_rake_cap)
        .max(0.0);
    let traverser_stake = starting_pot as f32 / np as f32 + c_t as f32;
    let traverser_folded = fold_mask & (1u16 << traverser) != 0;
    let opp_player: Vec<usize> = (0..num_opp)
        .map(|oi| if oi < traverser { oi } else { oi + 1 })
        .collect();
    let opp_contrib: Vec<i32> = opp_player.iter().map(|&p| contributions[p]).collect();
    let opp_folded: Vec<bool> = opp_player
        .iter()
        .map(|&p| fold_mask & (1u16 << p) != 0)
        .collect();

    let ctx = Arm2Ctx {
        levels,
        np,
        contributions: contributions.to_vec(),
        starting_pot,
        main_pot_rake,
        traverser_stake,
        traverser_folded,
        c_t,
        opp_contrib,
        opp_folded: opp_folded.clone(),
        traverser,
    };

    let mut rel = vec![0u8; num_opp];
    for bt in 0..nb {
        let probs = collapse(bt);
        let c_bt = renorm(bt, &probs);
        let mut accum = 0.0f32;
        recurse_rel_scalars(
            0,
            num_opp,
            1.0,
            &mut rel,
            &probs,
            &opp_folded,
            &mut |weight: f32, rel: &[u8]| {
                accum += weight * ctx.net(rel);
            },
        );
        cfv[bt] = accum * c_bt;
    }
    cfv
}

/// Design-2 arm-2 recursion: per-opponent scalar branch weights, no
/// bucket loops, no prefix blocking (the dropped factor IS the named
/// approximation). Folded opponents contribute their compatible mass
/// as a scalar multiplier; active opponents branch 3 ways.
fn recurse_rel_scalars(
    oi: usize,
    num_opp: usize,
    weight: f32,
    rel: &mut [u8],
    probs: &[[f32; 4]],
    opp_folded: &[bool],
    leaf: &mut dyn FnMut(f32, &[u8]),
) {
    if oi == num_opp {
        leaf(weight, rel);
        return;
    }
    if opp_folded[oi] {
        let n = probs[oi][3];
        if n == 0.0 {
            return;
        }
        rel[oi] = REL_FOLDED;
        recurse_rel_scalars(oi + 1, num_opp, weight * n, rel, probs, opp_folded, leaf);
    } else {
        for (code, p) in [
            (REL_WIN, probs[oi][0]),
            (REL_TIE, probs[oi][1]),
            (REL_LOSE, probs[oi][2]),
        ] {
            if p == 0.0 {
                continue;
            }
            rel[oi] = code;
            recurse_rel_scalars(oi + 1, num_opp, weight * p, rel, probs, opp_folded, leaf);
        }
    }
}

/// Arm-1 recursion: bucket-tuple enumeration (same index order as the
/// exact hand-tuple enumeration) carrying the (beaten, tie-count) state
/// distribution. At singletons the state is a point mass and the
/// surviving multiply chain equals `reach_so_far * r` bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn recurse_eq_buckets(
    oi: usize,
    num_opp: usize,
    nb: usize,
    bt: usize,
    state: &[f32; MAX_OPP + 2],
    prefix: &mut [u16],
    bucket_reach: &[&[f32]],
    tables: &BucketedRunoutTables,
    k: f32,
    rake_per_unit_stake: f32,
    accum: &mut f32,
) {
    if oi == num_opp {
        // Leaf payoffs — constants copied verbatim from recurse_eq.
        if state[0] != 0.0 {
            *accum += state[0] * -1.0; // loss, rake-invariant
        }
        for j in 0..=num_opp {
            let s = state[1 + j];
            if s == 0.0 {
                continue;
            }
            let net_unit: f32 = if j == 0 {
                // strict win, T = 1
                k - rake_per_unit_stake
            } else {
                let t_f = (j + 1) as f32;
                (k + 1.0 - t_f) / t_f - rake_per_unit_stake / t_f
            };
            *accum += s * net_unit;
        }
        return;
    }
    for bo in 0..nb {
        let r = bucket_reach[oi][bo];
        if r == 0.0 {
            continue;
        }
        // Pairwise opp-opp blocking fractions (named approximation (i);
        // binary ⇒ exact mask semantics at singletons).
        let mut m = r;
        let mut blocked = false;
        for &pb in prefix[..oi].iter() {
            let f = tables.f_n[pb as usize * nb + bo];
            if f == 0.0 {
                blocked = true;
                break;
            }
            m *= f;
        }
        if blocked {
            continue;
        }
        let i = bt * nb + bo;
        let fn_ = tables.f_n[i];
        if fn_ == 0.0 {
            continue; // fully blocked vs traverser bucket
        }
        let fw = tables.f_w[i];
        let ft = tables.f_t[i];
        let fl = tables.f_l[i];

        let mut new_state = [0.0f32; MAX_OPP + 2];
        // Beaten absorbs: subsequent opponents only contribute their
        // compatible mass.
        if state[0] != 0.0 {
            new_state[0] += state[0] * (m * fn_);
        }
        for j in 0..=oi {
            let s = state[1 + j];
            if s == 0.0 {
                continue;
            }
            if fl != 0.0 {
                new_state[0] += s * (m * fl); // opponent beats traverser
            }
            if ft != 0.0 {
                new_state[1 + j + 1] += s * (m * ft);
            }
            if fw != 0.0 {
                new_state[1 + j] += s * (m * fw);
            }
        }
        prefix[oi] = bo as u16;
        recurse_eq_buckets(
            oi + 1,
            num_opp,
            nb,
            bt,
            &new_state,
            prefix,
            bucket_reach,
            tables,
            k,
            rake_per_unit_stake,
            accum,
        );
    }
}

/// Arm-2 recursion: bucket-tuple enumeration branching on each ACTIVE
/// opponent's trinary relation to the traverser (folded opponents
/// contribute blocking mass only — their strengths are never read,
/// exactly as in the exact evaluator). At singletons exactly one branch
/// per opponent survives skip-on-zero.
#[allow(clippy::too_many_arguments)]
fn recurse_levels_buckets(
    oi: usize,
    num_opp: usize,
    nb: usize,
    bt: usize,
    weight: f32,
    prefix: &mut [u16],
    rel: &mut [u8],
    bucket_reach: &[&[f32]],
    tables: &BucketedRunoutTables,
    opp_folded: &[bool],
    leaf: &mut dyn FnMut(f32, &[u8]),
) {
    if oi == num_opp {
        leaf(weight, rel);
        return;
    }
    for bo in 0..nb {
        let r = bucket_reach[oi][bo];
        if r == 0.0 {
            continue;
        }
        let mut base = weight * r;
        let mut blocked = false;
        for &pb in prefix[..oi].iter() {
            let f = tables.f_n[pb as usize * nb + bo];
            if f == 0.0 {
                blocked = true;
                break;
            }
            base *= f;
        }
        if blocked {
            continue;
        }
        let i = bt * nb + bo;
        prefix[oi] = bo as u16;
        if opp_folded[oi] {
            let fn_ = tables.f_n[i];
            if fn_ == 0.0 {
                continue;
            }
            rel[oi] = REL_FOLDED;
            recurse_levels_buckets(
                oi + 1,
                num_opp,
                nb,
                bt,
                base * fn_,
                prefix,
                rel,
                bucket_reach,
                tables,
                opp_folded,
                leaf,
            );
        } else {
            let fw = tables.f_w[i];
            let ft = tables.f_t[i];
            let fl = tables.f_l[i];
            for (code, f) in [(REL_WIN, fw), (REL_TIE, ft), (REL_LOSE, fl)] {
                if f == 0.0 {
                    continue;
                }
                rel[oi] = code;
                recurse_levels_buckets(
                    oi + 1,
                    num_opp,
                    nb,
                    bt,
                    base * f,
                    prefix,
                    rel,
                    bucket_reach,
                    tables,
                    opp_folded,
                    leaf,
                );
            }
        }
    }
}
