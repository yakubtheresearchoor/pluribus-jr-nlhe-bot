// Core CFR kernels ported from vcfr.cu to Metal Shading Language.
// Phase 1-2: All kernels needed for vector CFR on river/turn-start games.
//
// STEP 2.A.2 BUG FIX 2026-06: disable FMA contraction file-wide so that
// `a * b + c` is NOT folded into a fused multiply-add. The CPU Rust f32
// path NEVER contracts (no implicit FMA), so any contracted GPU op
// produces 1-ULP divergence relative to CPU. These 1-ULP divergences
// compound 2.8x per iteration through the CFR feedback loop into the
// order-of-magnitude multi-iter divergence observed at stratum 1 iter 5+.
// The build.rs also passes -ffp-contract=off and -fno-fast-math, but
// this pragma is the authoritative source-level guarantee.
#pragma METAL fp contract(off)

#include "flat_tree.metal"

// ============================================================================
// Helper: sorted-sweep showdown
//
// Two-pass sorted sweep for HU terminal evaluation.
// Pass 1 (wins): walk player strengths ascending, accumulate opponent reach
//   for strictly weaker hands.
// Pass 2 (losses): walk descending, accumulate for strictly stronger.
// Result: cfv[h] = wins_contribution - losses_contribution
// ============================================================================

// Thread-local sorted sweep (for local arrays in multiway showdown)
inline void sorted_sweep_showdown_vcfr_local(
    thread const float* opp_reach_all, int num_opp, int nh,
    const device uint16_t* opp_strength, const device uint16_t* opp_indices,
    const device uint16_t* player_strength, const device uint16_t* player_indices,
    const device uint8_t* hand_cards,
    thread float* returned_cfv
) {
    for (int h = 0; h < nh; h++) returned_cfv[h] = 0.0f;

    // Pass 1: wins (ascending)
    for (int oi = 0; oi < num_opp; oi++) {
        float cfreach_sum = 0.0f;
        float cfreach_minus[52];
        for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

        int i = 0;
        for (int si = 0; si < nh; si++) {
            uint16_t str_h = player_strength[si];
            uint16_t h = player_indices[si];
            while (i < nh && opp_strength[oi * nh + i] < str_h) {
                uint16_t ho = opp_indices[oi * nh + i];
                float r = opp_reach_all[oi * nh + ho];
                if (r != 0.0f) {
                    cfreach_sum += r;
                    cfreach_minus[hand_cards[ho * 2]] += r;
                    cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                }
                i++;
            }
            float cfreach = cfreach_sum
                - cfreach_minus[hand_cards[h * 2]]
                - cfreach_minus[hand_cards[h * 2 + 1]];
            returned_cfv[h] += cfreach;
        }
    }

    // Pass 2: losses (descending)
    for (int oi = 0; oi < num_opp; oi++) {
        float cfreach_sum = 0.0f;
        float cfreach_minus[52];
        for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

        int i = nh - 1;
        for (int si = nh - 1; si >= 0; si--) {
            uint16_t str_h = player_strength[si];
            uint16_t h = player_indices[si];
            while (i >= 0 && opp_strength[oi * nh + i] > str_h) {
                uint16_t ho = opp_indices[oi * nh + i];
                float r = opp_reach_all[oi * nh + ho];
                if (r != 0.0f) {
                    cfreach_sum += r;
                    cfreach_minus[hand_cards[ho * 2]] += r;
                    cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                }
                i--;
            }
            float cfreach = cfreach_sum
                - cfreach_minus[hand_cards[h * 2]]
                - cfreach_minus[hand_cards[h * 2 + 1]];
            returned_cfv[h] -= cfreach;
        }
    }
}


// ============================================================================
// Helper: sorted-sweep with rake components (mirrors CPU
// sorted_sweep_with_rake_components in solver/showdown.rs:175).
//
// Computes three per-hand outputs in one O(num_opp * nh) sweep:
//   sweep_net[h] = wins_strict - losses_strict (with inclusion-exclusion
//                  card-blocking correction)
//   win_reach[h] = strict wins (same component, kept separate for rake)
//   tie_reach[h] = ties at top (with self-correction for HU since opp==h
//                  is in the tie band)
//
// Then caller computes:
//   cfv[h] = half_pot * sweep_net[h] - rake * (win_reach[h] + 0.5 * tie_reach[h])
//
// STEP 2.A.2 BIT-EXACT-MATCH FIX 2026-06: CPU's HU path uses this sweep
// formulation for ALL non-folded-traverser HU terminals (showdown.rs:598).
// The brute-force path (showdown.rs:1093) and the sweep produce the SAME
// mathematical answer but DIFFERENT float-rounding order. Using sweep
// here makes GPU=CPU bit-exact at every HU terminal under non-uniform
// reach, eliminating the 1-ULP-per-entry compounding source.
// ============================================================================
inline void sorted_sweep_with_rake_components_local(
    thread const float* opp_reach_all, int num_opp, int nh,
    const device uint16_t* opp_strength, const device uint16_t* opp_indices,
    const device uint16_t* player_strength, const device uint16_t* player_indices,
    const device uint8_t* hand_cards,
    thread float* sweep_net,
    thread float* win_reach,
    thread float* tie_reach
) {
    for (int h = 0; h < nh; h++) {
        sweep_net[h] = 0.0f;
        win_reach[h] = 0.0f;
        tie_reach[h] = 0.0f;
    }

    for (int oi = 0; oi < num_opp; oi++) {
        thread const float* reach = opp_reach_all + oi * nh;
        float cfreach_sum = 0.0f;
        float cfreach_minus[52];
        for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

        // FORWARD PASS: wins (opp_str < pl_str), with tie look-ahead.
        int i = 0;
        for (int si = 0; si < nh; si++) {
            uint16_t str_h = player_strength[si];
            uint16_t h = player_indices[si];
            // Win accumulation
            while (i < nh && opp_strength[oi * nh + i] < str_h) {
                uint16_t ho = opp_indices[oi * nh + i];
                float r = reach[ho];
                if (r != 0.0f) {
                    cfreach_sum += r;
                    cfreach_minus[hand_cards[ho * 2]] += r;
                    cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                }
                i++;
            }
            float win_val = cfreach_sum
                - cfreach_minus[hand_cards[h * 2]]
                - cfreach_minus[hand_cards[h * 2 + 1]];
            sweep_net[h] += win_val;
            win_reach[h] += win_val;

            // Tie look-ahead: opp_str == str_h, doesn't advance i.
            // Inclusion-exclusion correction: the only opp using BOTH of
            // h's cards is opp = h itself, and in HU showdown opp_str =
            // pl_str so opp = h IS in the tie band → add self correction.
            float tie_sum = 0.0f;
            float tie_minus[52];
            for (int c = 0; c < 52; c++) tie_minus[c] = 0.0f;
            int j = i;
            bool tie_includes_self = false;
            while (j < nh && opp_strength[oi * nh + j] == str_h) {
                uint16_t ho = opp_indices[oi * nh + j];
                float r = reach[ho];
                if (r != 0.0f) {
                    tie_sum += r;
                    tie_minus[hand_cards[ho * 2]] += r;
                    tie_minus[hand_cards[ho * 2 + 1]] += r;
                    if (ho == h) tie_includes_self = true;
                }
                j++;
            }
            float self_correction = tie_includes_self ? reach[h] : 0.0f;
            float tie_val = tie_sum
                - tie_minus[hand_cards[h * 2]]
                - tie_minus[hand_cards[h * 2 + 1]]
                + self_correction;
            tie_reach[h] += tie_val;
        }

        // BACKWARD PASS: losses (opp_str > pl_str). Subtract from sweep_net.
        cfreach_sum = 0.0f;
        for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

        int i_b = nh;
        for (int si = nh - 1; si >= 0; si--) {
            uint16_t str_h = player_strength[si];
            uint16_t h = player_indices[si];
            while (i_b > 0 && opp_strength[oi * nh + i_b - 1] > str_h) {
                i_b--;
                uint16_t ho = opp_indices[oi * nh + i_b];
                float r = reach[ho];
                if (r != 0.0f) {
                    cfreach_sum += r;
                    cfreach_minus[hand_cards[ho * 2]] += r;
                    cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                }
            }
            float loss_val = cfreach_sum
                - cfreach_minus[hand_cards[h * 2]]
                - cfreach_minus[hand_cards[h * 2 + 1]];
            sweep_net[h] -= loss_val;
        }
    }
}


// ============================================================================
// Helper: per-scenario brute-force showdown for 3-player (K=2 opponents) games.
//
// Mirrors CPU side_pot_showdown_cfv exactly. For each ordered (g_a, g_b)
// opponent hand assignment that doesn't conflict with the player's hand h
// (or with each other), computes the player's per-scenario net payoff
// (cash_received - traverser_stake) and weights by reach_a[g_a]*reach_b[g_b].
//
// Handles all cases via per-pot-level eligibility:
//   - num_active <= 1 (fold-win or all-fold): payoff is constant per scenario.
//   - All-active-equal contributions: single level pot.
//   - Unequal contributions (side pots): multi-level distribution.
//   - Folded players' contributions become dead money distributed by level.
//   - Ties at top: pot is split proportionally among tied players.
//
// Caller's responsibilities:
//   - opp_reach_local is contiguous [num_opp * nh], indexed by oi=opponent_slot.
//   - opp_reach_local must already include card-blocking masking (host filters
//     out hands containing turn/river cards via FlopChanceTable construction).
//   - sorted_pl_strength/indices are device arrays of size [nh].
//   - hand_strength is computed inside (no caller dependency).
//   - out[h] is filled for ALL h in [0, nh); caller is responsible for /nc.
// ============================================================================

// Forward declaration for the K >= 3 factored path (definition near end of
// this file). Allows `multiway_brute_force_showdown` to call it before the
// definition is reached in compilation order.
inline float factored_share_for_level_thread(
    int num_opp,
    uint elig_opps,
    uint tied_offset,
    ulong h_m,
    ushort h_str,
    thread const float* reach,
    const device uchar* hand_cards,
    thread const ushort* hand_strength,
    int nh);

inline void multiway_brute_force_showdown(
    int nh, int np,
    int traverser,
    int32_t starting_pot,
    uint16_t fold_mask,
    const thread int32_t* contributions,           // [np]
    thread const float* opp_reach_local,           // [num_opp * nh] (oi=0..num_opp-1)
    const device uint8_t* hand_cards,              // [nh*2]
    const device uint16_t* sorted_pl_strength,     // [nh]
    const device uint16_t* sorted_pl_indices,      // [nh]
    // ─── Slice 2 rake (CPU↔Metal parity) ───
    // Mirror CPU `side_pot_showdown_cfv_with_rake` interface. The
    // no-flop-no-drop gate is computed once at function entry; per
    // the rake spec, rake is taken from the MAIN POT ONLY (side
    // pots un-raked) with a SINGLE per-hand cap applied once.
    // flop_seen is the no-flop-no-drop gate: when false (preflop-
    // ending terminal), eff_rake_rate and eff_rake_cap are 0 so
    // every downstream path computes rake = 0.
    float rake_rate,
    float rake_cap,
    bool flop_seen,
    thread float* out                              // [nh]
) {
    int num_opp = np - 1;
    int32_t c_t = contributions[traverser];

    // Slice 2 rake gating: no-flop-no-drop. For flop-onward trees
    // (current state), flop_seen is always true and this is a no-op.
    // When preflop integration brings preflop-end terminals, callers
    // pass flop_seen=false at those terminals and rake collapses
    // to zero without revisiting the showdown code.
    float eff_rake_rate = flop_seen ? rake_rate : 0.0f;
    float eff_rake_cap  = flop_seen ? rake_cap  : 0.0f;

    // Build hand_strength array from sorted_pl. CPU equivalent:
    //   for si in 0..nh: hand_strength[sorted_pl_indices[si]] = sorted_pl_strength[si];
    uint16_t hand_strength[1326];
    for (int si = 0; si < nh; si++) {
        hand_strength[sorted_pl_indices[si]] = sorted_pl_strength[si];
    }

    // Build sorted+deduplicated contribution levels.
    int levels[8];
    int num_levels = 0;
    for (int p = 0; p < np; p++) {
        int32_t c = contributions[p];
        bool found = false;
        for (int l = 0; l < num_levels; l++) {
            if (levels[l] == c) { found = true; break; }
        }
        if (!found && num_levels < 8) { levels[num_levels++] = (int)c; }
    }
    for (int i = 0; i < num_levels - 1; i++) {
        for (int j = i + 1; j < num_levels; j++) {
            if (levels[j] < levels[i]) { int tmp = levels[i]; levels[i] = levels[j]; levels[j] = tmp; }
        }
    }

    float traverser_stake = float(starting_pot) / float(np) + float(c_t);
    bool traverser_folded = (fold_mask & (uint16_t)(1u << traverser)) != 0;

    // Count active (non-folded) players. CPU has an early-return path for
    // num_active <= 1 || traverser folded that uses `payoff * sum(r0*r1)`
    // with ONE final multiply. The unified brute-force path below uses
    // `sum(r0*r1*net)` with per-iteration multiplications, producing
    // ~172 ULPs of float drift at iter-2 reach. To match CPU bit-for-bit,
    // take the same early-return path when the conditions hold.
    int num_active = 0;
    for (int p = 0; p < np; p++) {
        if (!(fold_mask & (1u << p))) num_active++;
    }

    // Check all_active_equal && fold_mask == 0 — CPU's K=2 brute-force path
    // does `cfv[h] = half_pot * sum(r0*r1*payoff_unit)` where payoff_unit is
    // K, -1, or partial-tie. The single final multiply by half_pot keeps
    // float ordering distinct from the unified per-iteration multiply path.
    bool all_active_equal = true;
    {
        bool found = false;
        int32_t ref_c = 0;
        for (int p = 0; p < np; p++) {
            if (fold_mask & (1u << p)) continue;
            if (!found) { ref_c = contributions[p]; found = true; }
            else if (contributions[p] != ref_c) { all_active_equal = false; break; }
        }
    }

    if (num_opp == 2 && all_active_equal && fold_mask == 0) {
        int num_active_opp = 0;
        for (int p = 0; p < np; p++) {
            if (p == traverser) continue;
            if (!(fold_mask & (1u << p))) num_active_opp++;
        }
        if (num_active_opp == 2) {
            float half_pot = float(starting_pot) / float(np) + float(c_t);
            thread const float* reach_0 = opp_reach_local + 0 * nh;
            thread const float* reach_1 = opp_reach_local + 1 * nh;
            float k = (float)num_active_opp;

            // ── Slice 2 Phase B Site (a): K=2 all-equal-brute rake ──
            // Mirror CPU `side_pot_showdown_cfv_with_rake` lines 699-779.
            // rake = min(total_pot * eff_rake_rate, eff_rake_cap)
            // rake_per_unit_stake = rake / half_pot
            // payoff_unit adjustments:
            //   loss (max_opp > h_str):  -1                       (rake-invariant)
            //   tie  (max_opp == h_str): (K+1-T)/T - rps/T         (T winners split pot-rake)
            //   win  (max_opp < h_str):  K - rps                   (strict win, T=1)
            // Computed ONCE outside the per-(h,g0,g1) loops since
            // total_pot/rake/rps are all-equal-contribs constants here.
            int32_t total_pot_int = (int)starting_pot;
            for (int p = 0; p < np; p++) total_pot_int += (int)contributions[p];
            float rake = fmax(0.0f, fmin((float)total_pot_int * eff_rake_rate, eff_rake_cap));
            float rake_per_unit_stake = (half_pot > 0.0f) ? (rake / half_pot) : 0.0f;

            for (int h = 0; h < nh; h++) {
                int hc1 = hand_cards[h * 2];
                int hc2 = hand_cards[h * 2 + 1];
                uint16_t h_str = hand_strength[h];
                float accum = 0.0f;

                for (int g0 = 0; g0 < nh; g0++) {
                    int g0c1 = hand_cards[g0 * 2];
                    int g0c2 = hand_cards[g0 * 2 + 1];
                    if (g0c1 == hc1 || g0c1 == hc2 || g0c2 == hc1 || g0c2 == hc2) continue;
                    float r0 = reach_0[g0];
                    if (r0 == 0.0f) continue;
                    uint16_t s0 = hand_strength[g0];

                    for (int g1 = 0; g1 < nh; g1++) {
                        int g1c1 = hand_cards[g1 * 2];
                        int g1c2 = hand_cards[g1 * 2 + 1];
                        if (g1c1 == hc1 || g1c1 == hc2 || g1c2 == hc1 || g1c2 == hc2) continue;
                        if (g0c1 == g1c1 || g0c1 == g1c2 || g0c2 == g1c1 || g0c2 == g1c2) continue;
                        float r1 = reach_1[g1];
                        if (r1 == 0.0f) continue;
                        uint16_t s1 = hand_strength[g1];
                        uint16_t max_opp = (s0 > s1) ? s0 : s1;

                        float payoff_unit;
                        if (max_opp > h_str) {
                            payoff_unit = -1.0f;  // loss: rake-invariant
                        } else if (max_opp == h_str) {
                            uint t = 1;
                            if (s0 == h_str) t++;
                            if (s1 == h_str) t++;
                            float t_f = (float)t;
                            payoff_unit = (k + 1.0f - t_f) / t_f - rake_per_unit_stake / t_f;
                        } else {
                            payoff_unit = k - rake_per_unit_stake;  // strict win
                        }
                        accum += r0 * r1 * payoff_unit;
                    }
                }
                out[h] = half_pot * accum;
            }
            return;
        }
    }

    if (num_opp == 2 && (num_active <= 1 || traverser_folded)) {
        int total_pot = (int)starting_pot;
        for (int p = 0; p < np; p++) total_pot += (int)contributions[p];
        // ── Slice 2 Phase B site (b) — HU residual fix (2026-06-04) ──
        // Mirror CORRECTED CPU `side_pot_showdown_cfv_with_rake` fast
        // path. Per the rake spec: rake applies to
        // MAIN POT ONLY (called portion). Uncalled bets returned to the
        // bettor un-raked.
        //
        // The previous version of this site (and the corresponding CPU
        // fast path) used `total_pot * rake_rate` which over-raked
        // uncalled bets — a real arithmetic bug that surfaced as the
        // HU gate 0.09375 residual once Metal K=1 (already main-pot-only
        // correct) was closed in Phase B Site (d) part 1. The K=1 vs
        // K=2 internal inconsistency was the discriminator.
        //
        // Now consistent across K=1 (per-level, main-pot-only) and
        // K=2 (this fast path, main-pot-only): both formulas give
        // identical results to the per-level computation at equal
        // contributions (main_pot == total_pot), and both correctly
        // refund uncalled bets at unequal contributions.
        //
        // Folded traverser loses their investment regardless of rake.
        // Active lone-survivor claims (total_pot − main_pot_rake), i.e.,
        // gets uncalled excess at face value plus rake-reduced main pot.
        int min_contrib = contributions[0];
        for (int p = 1; p < np; p++) {
            if (contributions[p] < min_contrib) min_contrib = contributions[p];
        }
        int num_main_contributors = 0;
        for (int p = 0; p < np; p++) {
            if (contributions[p] >= min_contrib) num_main_contributors++;
        }
        int main_pot_amount = min_contrib * num_main_contributors + (int)starting_pot;
        float rake = fmax(0.0f, fmin((float)main_pot_amount * eff_rake_rate, eff_rake_cap));
        float payoff;
        if (traverser_folded) {
            payoff = -traverser_stake;
        } else {
            payoff = ((float)total_pot - rake) - traverser_stake;
        }

        thread const float* reach_0 = opp_reach_local + 0 * nh;
        thread const float* reach_1 = opp_reach_local + 1 * nh;

        for (int h = 0; h < nh; h++) {
            int hc1 = hand_cards[h * 2];
            int hc2 = hand_cards[h * 2 + 1];
            float nh_count = 0.0f;
            for (int g0 = 0; g0 < nh; g0++) {
                int g0c1 = hand_cards[g0 * 2];
                int g0c2 = hand_cards[g0 * 2 + 1];
                if (g0c1 == hc1 || g0c1 == hc2 || g0c2 == hc1 || g0c2 == hc2) continue;
                float r0 = reach_0[g0];
                if (r0 == 0.0f) continue;
                for (int g1 = 0; g1 < nh; g1++) {
                    int g1c1 = hand_cards[g1 * 2];
                    int g1c2 = hand_cards[g1 * 2 + 1];
                    if (g1c1 == hc1 || g1c1 == hc2 || g1c2 == hc1 || g1c2 == hc2) continue;
                    if (g0c1 == g1c1 || g0c1 == g1c2 || g0c2 == g1c1 || g0c2 == g1c2) continue;
                    float r1 = reach_1[g1];
                    if (r1 == 0.0f) continue;
                    nh_count += r0 * r1;
                }
            }
            out[h] = payoff * nh_count;
        }
        return;
    }

    if (num_opp == 2) {
        int opp_a = (traverser == 0) ? 1 : 0;
        int opp_b = (traverser == 2) ? 1 : 2;
        thread const float* reach_a = opp_reach_local + 0 * nh;
        thread const float* reach_b = opp_reach_local + 1 * nh;
        int32_t c_opp_a = contributions[opp_a];
        int32_t c_opp_b = contributions[opp_b];
        bool a_folded = (fold_mask & (uint16_t)(1u << opp_a)) != 0;
        bool b_folded = (fold_mask & (uint16_t)(1u << opp_b)) != 0;

        // ── Slice 2 Phase B Site (c): main-pot-only rake (the rake spec) ──
        // Mirror CPU `side_pot_showdown_cfv_with_rake` ~870-910.
        // Rake applies ONLY at li==0 (main pot). Side-pot levels
        // (li>=1) are UN-RAKED per site convention. The cap is
        // applied ONCE here (not per-level), making the unit-test
        // discriminating: a per-level rake or per-level cap error
        // would fail against this CPU reference immediately.
        //
        // main_pot_amount = levels[0] * num_main_contributors + starting_pot
        //   where num_main_contributors = #players (including folded)
        //   whose contribution >= levels[0].
        int32_t main_pot_amount;
        if (num_levels == 0) {
            main_pot_amount = starting_pot;
        } else {
            int num_main_contributors = 0;
            for (int p = 0; p < np; p++) {
                if (contributions[p] >= levels[0]) num_main_contributors++;
            }
            main_pot_amount = levels[0] * num_main_contributors + starting_pot;
        }
        float main_pot_rake = fmax(0.0f, fmin(
            (float)main_pot_amount * eff_rake_rate, eff_rake_cap));

        for (int h = 0; h < nh; h++) {
            int hc1 = hand_cards[h * 2];
            int hc2 = hand_cards[h * 2 + 1];
            uint16_t h_str = hand_strength[h];
            float accum = 0.0f;

            for (int g_a = 0; g_a < nh; g_a++) {
                int g_ac1 = hand_cards[g_a * 2];
                int g_ac2 = hand_cards[g_a * 2 + 1];
                if (g_ac1 == hc1 || g_ac1 == hc2 || g_ac2 == hc1 || g_ac2 == hc2) continue;
                float ra = reach_a[g_a];
                if (ra == 0.0f) continue;
                uint16_t s_a = hand_strength[g_a];

                for (int g_b = 0; g_b < nh; g_b++) {
                    int g_bc1 = hand_cards[g_b * 2];
                    int g_bc2 = hand_cards[g_b * 2 + 1];
                    if (g_bc1 == hc1 || g_bc1 == hc2 || g_bc2 == hc1 || g_bc2 == hc2) continue;
                    if (g_ac1 == g_bc1 || g_ac1 == g_bc2 || g_ac2 == g_bc1 || g_ac2 == g_bc2) continue;
                    float rb = reach_b[g_b];
                    if (rb == 0.0f) continue;
                    uint16_t s_b = hand_strength[g_b];

                    float net;
                    if (traverser_folded) {
                        net = -traverser_stake;
                    } else {
                        float cash = 0.0f;
                        int prev_l = 0;
                        for (int li = 0; li < num_levels; li++) {
                            int lev = levels[li];
                            int pc = lev - prev_l;
                            // 2.A.2 FIX: skip only when total pot at this
                            // level is 0 (after starting_pot addition).
                            int num_contrib = 0;
                            for (int p = 0; p < np; p++) {
                                if (contributions[p] >= lev) num_contrib++;
                            }
                            float pot_l = float(pc * num_contrib);
                            if (li == 0) pot_l += float(starting_pot);
                            if (pot_l == 0.0f) { prev_l = lev; continue; }

                            bool trav_elig = c_t >= lev;
                            bool a_elig = !a_folded && c_opp_a >= lev;
                            bool b_elig = !b_folded && c_opp_b >= lev;
                            int n_elig_total = (trav_elig ? 1 : 0) + (a_elig ? 1 : 0) + (b_elig ? 1 : 0);

                            if (n_elig_total == 0) {
                                if (contributions[traverser] >= lev) {
                                    float trav_contrib = float(pc);
                                    if (li == 0) trav_contrib += float(starting_pot) / float(np);
                                    cash += trav_contrib;
                                }
                                prev_l = lev;
                                continue;
                            }

                            if (!trav_elig) {
                                prev_l = lev;
                                continue;
                            }

                            uint16_t max_str = h_str;
                            if (a_elig && s_a > max_str) max_str = s_a;
                            if (b_elig && s_b > max_str) max_str = s_b;

                            int tied = 0;
                            if (h_str == max_str) tied++;
                            if (a_elig && s_a == max_str) tied++;
                            if (b_elig && s_b == max_str) tied++;

                            if (h_str == max_str) {
                                // Slice 2 Site (c): rake from main pot
                                // only (li == 0); side pots (li > 0) clean.
                                float pot_after_rake = (li == 0)
                                    ? (pot_l - main_pot_rake)
                                    : pot_l;
                                cash += pot_after_rake / float(tied);
                            }
                            prev_l = lev;
                        }
                        net = cash - traverser_stake;
                    }

                    accum += ra * rb * net;
                }
            }
            out[h] = accum;
        }
        return;
    }

    if (num_opp == 1) {
        // ── STEP 2.A.2 BIT-EXACT-MATCH FIX 2026-06 ──
        // HU showdown: mirror CPU `side_pot_showdown_cfv_with_rake` at
        // showdown.rs:598 (np==2 && !traverser_folded → sorted sweep) and
        // showdown.rs:484-568 (lone-survivor / traverser-folded → fast path).
        //
        // The CPU NEVER takes the per-level brute-force path for HU;
        // it always uses either the sorted sweep (active HU showdown)
        // or the inclusion-exclusion fast path (HU fold-end). The GPU
        // was previously running per-level brute-force for HU, which is
        // mathematically equivalent but differs from CPU by 1 ULP in
        // float-rounding order. Those 1-ULP differences then compounded
        // 2.8x per iter through the CFR feedback loop into the
        // order-of-magnitude multi-iter divergence observed at stratum 1.
        //
        // Localized via tests/p1_5_4_step2a2_iter1_noise_source.rs
        // (61% of terminal CFV entries differ at 1 ULP under non-uniform
        // reach; reach itself bit-exact, contributions [0,0] at node 13).
        int opp_a = (traverser == 0) ? 1 : 0;
        thread const float* reach_a = opp_reach_local + 0 * nh;
        int32_t c_opp_a = contributions[opp_a];
        bool a_folded = (fold_mask & (uint16_t)(1u << opp_a)) != 0;

        if (traverser_folded || a_folded) {
            // ── HU fold-end: constant-payoff inclusion-exclusion ──
            // Mirror CPU showdown.rs:484-568 (num_opp == 1 fast path).
            int total_pot = (int)starting_pot;
            for (int p = 0; p < np; p++) total_pot += (int)contributions[p];
            int min_contrib = contributions[0];
            for (int p = 1; p < np; p++) {
                if (contributions[p] < min_contrib) min_contrib = contributions[p];
            }
            int num_main_contributors = 0;
            for (int p = 0; p < np; p++) {
                if (contributions[p] >= min_contrib) num_main_contributors++;
            }
            int main_pot_amount = min_contrib * num_main_contributors + (int)starting_pot;
            float rake = fmax(0.0f, fmin(
                (float)main_pot_amount * eff_rake_rate, eff_rake_cap));
            float payoff = traverser_folded
                ? -traverser_stake
                : ((float)total_pot - rake) - traverser_stake;

            float opp_reach_sum = 0.0f;
            float opp_reach_minus[52];
            for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;
            for (int ho = 0; ho < nh; ho++) {
                float r = reach_a[ho];
                if (r != 0.0f) {
                    opp_reach_sum += r;
                    opp_reach_minus[hand_cards[ho * 2]] += r;
                    opp_reach_minus[hand_cards[ho * 2 + 1]] += r;
                }
            }
            for (int h = 0; h < nh; h++) {
                // Inclusion-exclusion: subtract opp hands using either of
                // h's cards, then ADD BACK reach_a[h] because the only
                // opp using BOTH of h's cards is h itself and we
                // double-subtracted it (audit-fix #37 in CPU).
                float cfreach = opp_reach_sum
                    - opp_reach_minus[hand_cards[h * 2]]
                    - opp_reach_minus[hand_cards[h * 2 + 1]]
                    + reach_a[h];
                out[h] = payoff * cfreach;
            }
            return;
        }

        // ── HU active showdown: sorted-sweep + half-pot scaling ──
        // Mirror CPU showdown.rs:598-639.
        // For HU, opp and player share the same hand-strength evaluation
        // (same board), so sorted_opp arrays equal sorted_pl. We pass
        // sorted_pl for both opp and player arguments.
        int min_active_contrib = (c_t < c_opp_a) ? c_t : c_opp_a;
        float half_pot = float(starting_pot) / float(np) + float(min_active_contrib);
        int total_pot = (int)starting_pot + (int)c_t + (int)c_opp_a;
        float rake = fmax(0.0f, fmin(
            (float)total_pot * eff_rake_rate, eff_rake_cap));

        float sweep_net_arr[1326];
        float win_reach_arr[1326];
        float tie_reach_arr[1326];
        sorted_sweep_with_rake_components_local(
            opp_reach_local, 1, nh,
            sorted_pl_strength, sorted_pl_indices,  // opp = pl in HU
            sorted_pl_strength, sorted_pl_indices,
            hand_cards,
            sweep_net_arr, win_reach_arr, tie_reach_arr
        );

        for (int h = 0; h < nh; h++) {
            out[h] = half_pot * sweep_net_arr[h]
                   - rake * (win_reach_arr[h] + 0.5f * tie_reach_arr[h]);
        }
        return;
    }

    // ========================================================================
    // K >= 3 (np >= 4): UNIFIED FACTORED CFV via the per-level recursive
    // K-1 expansion with eligibility-restricted strength comparison.
    //
    // Replaces the previous K>=3 no-op. Same factored math validated at
    // f32 noise floor across 18 gate tests including the 6-level side
    // pot, three-way-shared-card configurations, and bug-class folded-
    // high-contrib path. K=1 (HU) and K=2 (3-player) brute-force paths
    // remain as-is in this commit (validated production); future cleanup
    // unifies them.
    // ========================================================================

    // ── Slice 2 Phase B Site (e): K≥3 factored main-pot-only rake ──
    // Mirror CPU `side_pot_showdown_cfv_with_rake` main-pot-only spec
    // (showdown.rs ~870-910). Rake applies ONLY at li==0 (main pot);
    // side-pot levels (li>=1) un-raked; cap applied ONCE per hand.
    // Computed ONCE before the h loop since main_pot is determined by
    // contributions and starting_pot, not by h.
    int32_t e_main_pot_amount;
    if (num_levels == 0) {
        e_main_pot_amount = starting_pot;
    } else {
        int num_main_contributors = 0;
        for (int p = 0; p < np; p++) {
            if (contributions[p] >= levels[0]) num_main_contributors++;
        }
        e_main_pot_amount = levels[0] * num_main_contributors + starting_pot;
    }
    float e_main_pot_rake = fmax(0.0f, fmin(
        (float)e_main_pot_amount * eff_rake_rate, eff_rake_cap));

    for (int h = 0; h < nh; h++) {
        ulong h_m = (1ul << hand_cards[h * 2]) | (1ul << hand_cards[h * 2 + 1]);
        ushort h_str = hand_strength[h];

        // TVRP(h) via factored share with all-ineligible (elig_opps=0, tied=0).
        float tvrp = factored_share_for_level_thread(
            num_opp, 0u, 0u, h_m, h_str,
            opp_reach_local, hand_cards, hand_strength, nh);

        // Walk levels: static cash + Case C shares.
        float static_cash = 0.0f;
        float case_c = 0.0f;
        int prev_l = 0;
        for (int li = 0; li < num_levels; li++) {
            int lev = levels[li];
            int pc = lev - prev_l;
            // 2.A.2 FIX: see HU branch lines 520-537 for rationale.
            int num_contrib = 0;
            for (int p = 0; p < np; p++) {
                if (contributions[p] >= lev) num_contrib++;
            }
            float pot_l = (float)(pc * num_contrib);
            if (li == 0) pot_l += (float)starting_pot;
            if (pot_l == 0.0f) { prev_l = lev; continue; }
            // Site (e) rake: at li==0 (main pot) winner-share gets pot
            // reduced by main_pot_rake. Side pots (li>=1) un-raked.
            float pot_after_rake = (li == 0) ? (pot_l - e_main_pot_rake) : pot_l;

            uint elig_opps = 0u;
            int oi = 0;
            for (int p = 0; p < np; p++) {
                if (p == traverser) continue;
                bool p_folded = (fold_mask & (uint16_t)(1u << p)) != 0;
                bool p_elig = !p_folded && (contributions[p] >= lev);
                if (p_elig) elig_opps |= (1u << oi);
                oi++;
            }
            bool trav_elig = !traverser_folded && (c_t >= lev);
            bool has_active_elig = (elig_opps != 0);

            if (!has_active_elig && trav_elig) {
                // Traverser sole eligible — wins (raked) pot at this level.
                static_cash += pot_after_rake;
            } else if (!has_active_elig && !trav_elig) {
                if (contributions[traverser] >= lev) {
                    // Dead-money return — NO RAKE (refund, not a win).
                    // Matches CPU n_elig==0 branch.
                    float trav_contrib = (float)pc;
                    if (li == 0) trav_contrib += (float)starting_pot / (float)np;
                    static_cash += trav_contrib;
                }
            } else if (!trav_elig) {
                // Case D: traverser ineligible at contested level — no cash.
            } else {
                // Case C: contested level. Share of post-rake pot.
                float share = factored_share_for_level_thread(
                    num_opp, elig_opps, 0u, h_m, h_str,
                    opp_reach_local, hand_cards, hand_strength, nh);
                case_c += pot_after_rake * share;
            }
            prev_l = lev;
        }

        out[h] = (static_cash - traverser_stake) * tvrp + case_c;
    }
}

// ============================================================================
// vcfr_compute_strategies
// ============================================================================

struct StrategiesParams {
    int num_infosets;
    int nh;
};

// QRE (quantal-response) strategy: σ_a ∝ exp(λ · last_cfv_a/denom), per-hand
// max-stabilized. Mirrors CPU mccfr.rs:1053-1078. denom = max(iter-1, 1)
// (time-average action value); first iterate (last_cfv=0) → uniform.
struct QreParams {
    int num_infosets;
    int nh;
    float lambda;
    float denom;
};

kernel void vcfr_compute_strategies_qre(
    device const float* last_cfv            [[buffer(0)]],
    device float*       strategy            [[buffer(1)]],
    device const uint32_t* decision_node_ids [[buffer(2)]],
    device const FlatNode* nodes            [[buffer(3)]],
    device const uint32_t* infoset_offsets  [[buffer(4)]],
    constant QreParams& params              [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int infoset_id = int(gid.x);
    int h = int(gid.y);
    if (infoset_id >= params.num_infosets || h >= params.nh) return;
    int nh = params.nh;
    uint node_id = decision_node_ids[infoset_id];
    FlatNode node = nodes[node_id];
    int na = int(node.num_children);
    int stride = MAX_NA * nh;
    const device float* lc = last_cfv + infoset_id * stride;
    device float* s = strategy + infoset_id * stride;
    float denom = params.denom;

    float mx = -INFINITY;
    for (int a = 0; a < na; a++) mx = max(mx, lc[a * nh + h] / denom);
    float z = 0.0f;
    for (int a = 0; a < na; a++) {
        float avg = lc[a * nh + h] / denom;
        float w = exp(params.lambda * (avg - mx));
        s[a * nh + h] = w;
        z += w;
    }
    z = (z > 0.0f) ? z : 1.0f;
    for (int a = 0; a < na; a++) s[a * nh + h] /= z;
    for (int a = na; a < MAX_NA; a++) s[a * nh + h] = 0.0f;
}

kernel void vcfr_compute_strategies(
    device const float* regrets             [[buffer(0)]],
    device float*       strategy            [[buffer(1)]],
    device const uint32_t* decision_node_ids [[buffer(2)]],
    device const FlatNode* nodes            [[buffer(3)]],
    device const uint32_t* infoset_offsets  [[buffer(4)]],
    constant StrategiesParams& params       [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int infoset_id = int(gid.x);
    int h = int(gid.y);
    int num_infosets = params.num_infosets;
    int nh = params.nh;

    if (infoset_id >= num_infosets || h >= nh) return;

    uint node_id = decision_node_ids[infoset_id];
    FlatNode node = nodes[node_id];
    int na = int(node.num_children);
    int stride = MAX_NA * nh;

    const device float* r = regrets + infoset_id * stride;
    device float* s = strategy + infoset_id * stride;

    // Bug B fix: regret matching epsilon to prevent ULP-level strategy flips
    const float REGRET_MATCH_EPS = 1e-5f;

    float pos_sum = 0.0f;
    for (int a = 0; a < na; a++) {
        float rv = r[a * nh + h];
        if (rv > REGRET_MATCH_EPS) pos_sum += rv;
    }

    if (pos_sum > 0.0f) {
        for (int a = 0; a < na; a++) {
            float rv = r[a * nh + h];
            s[a * nh + h] = (rv > REGRET_MATCH_EPS) ? rv / pos_sum : 0.0f;
        }
    } else {
        float u = 1.0f / float(na);
        for (int a = 0; a < na; a++) {
            s[a * nh + h] = u;
        }
    }

    for (int a = na; a < MAX_NA; a++) {
        s[a * nh + h] = 0.0f;
    }
}

// ============================================================================
// vcfr_init_reach
// ============================================================================

struct InitReachParams {
    int total_reach_size;
    int np_nh;
};

kernel void vcfr_init_reach(
    device float*       reach               [[buffer(0)]],
    device const float* initial_weight       [[buffer(1)]],
    constant InitReachParams& params         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    if (idx < params.np_nh) {
        reach[idx] = initial_weight[idx];
    } else if (idx < params.total_reach_size) {
        reach[idx] = 0.0f;
    }
}

// ============================================================================
// vcfr_zero_buffer
// ============================================================================

struct ZeroBufferParams {
    int size;
};

kernel void vcfr_zero_buffer(
    device float* buf                         [[buffer(0)]],
    constant ZeroBufferParams& params         [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    if (idx < params.size) buf[idx] = 0.0f;
}

// ============================================================================
// Depth-limited bucketed CONTINUATION LEAF (HU, Arm-1 closed form).
//
// Continuation chance leaves are reached by check/call closing the street ⇒
// equal contributions, fold_mask=0 ⇒ Arm 1 of the CPU collapsed showdown. For
// HU (num_opp=1) that DP collapses (validated bit-close in
// hu_continuation_closed_form.rs) to, per traverser bucket bt:
//
//   cfv[bt] = half_pot · Σ_bo reach[bo] · ( (f_w-f_l) - rps·(f_w + f_t/2) )
//
// Two kernels run per traverser pass AFTER top_down (reach known) and AFTER the
// d_cfv zero, BEFORE the bottom-up level loop:
//   1. vcfr_continuation_reduce  : per-hand opp reach → per-bucket reach
//   2. vcfr_continuation_fill    : closed-form showdown + expand → cfv[leaf][h]
// 0xFFFF map entries (dead hands) contribute nothing and get cfv=0.
// ============================================================================

struct ContParams {
    int nb;
    int nh;
    int np;
    int traverser;
    int n_leaf;
    int starting_pot;
    float rake_rate;
    float rake_cap;
    float num_combinations;
};

#define CONT_NO_BUCKET 0xFFFFu

// One thread per (leaf, opponent-bucket): sum the traverser's opponent reach
// over the hands that map to that bucket. Grid = (n_leaf, nb).
kernel void vcfr_continuation_reduce(
    device float*          bucket_reach   [[buffer(0)]],  // [n_leaf * nb] out
    device const float*    reach          [[buffer(1)]],  // [nn * np * nh]
    device const uint16_t* map            [[buffer(2)]],  // [nh] hand→bucket
    device const uint32_t* leaf_nodes     [[buffer(3)]],  // [n_leaf] node ids
    constant ContParams&   p              [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int li = int(gid.x);
    int bo = int(gid.y);
    if (li >= p.n_leaf || bo >= p.nb) return;
    uint node_id = leaf_nodes[li];
    int opp = (p.traverser == 0) ? 1 : 0;  // HU
    const device float* opp_r = reach + (uint(node_id) * uint(p.np) + uint(opp)) * uint(p.nh);
    float s = 0.0f;
    for (int h = 0; h < p.nh; h++) {
        if (uint(map[h]) == uint(bo)) s += opp_r[h];
    }
    bucket_reach[li * p.nb + bo] = s;
}

// One thread per (leaf, hand): closed-form HU Arm-1 showdown for that hand's
// bucket, then /num_combinations. Grid = (n_leaf, nh).
kernel void vcfr_continuation_fill(
    device float*          cfv            [[buffer(0)]],  // [nn * nh] (write leaf rows)
    device const float*    bucket_reach   [[buffer(1)]],  // [n_leaf * nb]
    device const uint16_t* map            [[buffer(2)]],  // [nh]
    device const uint32_t* leaf_nodes     [[buffer(3)]],  // [n_leaf]
    device const int32_t*  contributions  [[buffer(4)]],  // [nn * np]
    device const float*    f_w            [[buffer(5)]],  // [nb * nb]
    device const float*    f_t            [[buffer(6)]],
    device const float*    f_l            [[buffer(7)]],
    device const float*    f_n            [[buffer(8)]],
    constant ContParams&   p              [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int li = int(gid.x);
    int h  = int(gid.y);
    if (li >= p.n_leaf || h >= p.nh) return;
    uint node_id = leaf_nodes[li];
    device float* out = cfv + uint(node_id) * uint(p.nh);

    uint bt = uint(map[h]);
    if (bt == CONT_NO_BUCKET) { out[h] = 0.0f; return; }

    int traverser = p.traverser;
    int opp = (traverser == 0) ? 1 : 0;
    int c_t   = contributions[uint(node_id) * uint(p.np) + uint(traverser)];
    int c_opp = contributions[uint(node_id) * uint(p.np) + uint(opp)];
    float half_pot = float(p.starting_pot) / float(p.np) + float(c_t);
    int total_pot = p.starting_pot + c_t + c_opp;
    float rake = max(min(float(total_pot) * p.rake_rate, p.rake_cap), 0.0f);
    float rps = (half_pot > 0.0f) ? (rake / half_pot) : 0.0f;

    float accum = 0.0f;
    int row = int(bt) * p.nb;
    for (int bo = 0; bo < p.nb; bo++) {
        float r = bucket_reach[li * p.nb + bo];
        if (r == 0.0f) continue;
        int i = row + bo;
        if (f_n[i] == 0.0f) continue;
        float fw = f_w[i];
        float ft = f_t[i];
        float fl = f_l[i];
        accum += r * ((fw - fl) - rps * (fw + ft * 0.5f));
    }
    float v = half_pot * accum;
    out[h] = (p.num_combinations > 0.0f) ? (v / p.num_combinations) : v;
}

// ============================================================================
// MULTIWAY continuation leaf (np ≥ 3), Design-1 Arm-1, MC-sampled — faithful to
// the deployed CPU `bucketed_showdown_cfv_design1_collapsed_sampled`. Exact
// bucket enumeration is B^num_opp; sampling draws M opponent bucket-tuples ∝
// reach. Four kernels: reduce_mw (per-opp bucket reach) → cdf_mw (prefix + zsum)
// → showdown_mw (per-bucket MC value) → expand (per-hand /nc).
// ============================================================================

struct ContMwParams {
    int nb;
    int nh;
    int np;
    int traverser;
    int n_leaf;
    int num_opp;
    int starting_pot;
    uint sample_m;
    float rake_rate;
    float rake_cap;
    float num_combinations;
    ulong rng_seed;
};

// map opponent slot oi (0..num_opp) to player id, skipping the traverser.
static inline int mw_opp_player(int oi, int traverser) {
    return (oi < traverser) ? oi : (oi + 1);
}

// Reduce: bucket_reach[(li*num_opp+oi)*nb + bo] = Σ_{h:map==bo} reach[leaf][player(oi)][h].
// Grid (n_leaf*num_opp, nb).
kernel void vcfr_continuation_reduce_mw(
    device float*          bucket_reach   [[buffer(0)]],
    device const float*    reach          [[buffer(1)]],
    device const uint16_t* map            [[buffer(2)]],
    device const uint32_t* leaf_nodes     [[buffer(3)]],
    constant ContMwParams& p              [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int row = int(gid.x);              // li*num_opp + oi
    int bo  = int(gid.y);
    if (row >= p.n_leaf * p.num_opp || bo >= p.nb) return;
    int li = row / p.num_opp;
    int oi = row % p.num_opp;
    uint node_id = leaf_nodes[li];
    int player = mw_opp_player(oi, p.traverser);
    const device float* r = reach + (uint(node_id) * uint(p.np) + uint(player)) * uint(p.nh);
    float s = 0.0f;
    for (int h = 0; h < p.nh; h++) {
        if (uint(map[h]) == uint(bo)) s += r[h];
    }
    bucket_reach[row * p.nb + bo] = s;
}

// CDF: prefix-sum bucket_reach per (leaf,opp) → cdf; zsum = total mass.
// Grid (n_leaf, num_opp).
kernel void vcfr_continuation_cdf_mw(
    device float*          cdf            [[buffer(0)]],  // [n_leaf*num_opp*nb]
    device float*          zsum           [[buffer(1)]],  // [n_leaf*num_opp]
    device const float*    bucket_reach   [[buffer(2)]],
    constant ContMwParams& p              [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int li = int(gid.x);
    int oi = int(gid.y);
    if (li >= p.n_leaf || oi >= p.num_opp) return;
    int row = li * p.num_opp + oi;
    float c = 0.0f;
    for (int b = 0; b < p.nb; b++) {
        c += bucket_reach[row * p.nb + b];
        cdf[row * p.nb + b] = c;
    }
    zsum[row] = c;
}

static inline ulong sm_next(thread ulong& s) {
    s += 0x9E3779B97F4A7C15UL;
    ulong z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}
static inline float sm_unif(thread ulong& s) {
    return (float)(sm_next(s) >> 40) / (float)(1u << 24);
}

// Showdown (MC): per (leaf, bt) draw M opponent bucket-tuples ∝ reach and
// accumulate the per-tuple value. Arm 1 (all active equal, no fold) uses the
// tie-counting state DP; Arm 2 (a player folded during the street ⇒ dead money
// / side pots) uses the level-DP net_expected port. Grid (n_leaf, nb).
kernel void vcfr_continuation_showdown_mw(
    device float*          cfv_bucket     [[buffer(0)]],  // [n_leaf*nb] out
    device const float*    cdf            [[buffer(1)]],
    device const float*    zsum           [[buffer(2)]],
    device const uint32_t* leaf_nodes     [[buffer(3)]],
    device const int32_t*  contributions  [[buffer(4)]],
    device const float*    f_w            [[buffer(5)]],
    device const float*    f_t            [[buffer(6)]],
    device const float*    f_l            [[buffer(7)]],
    device const float*    f_n            [[buffer(8)]],
    device const uint16_t* folded_masks   [[buffer(9)]],
    constant ContMwParams& p              [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int li = int(gid.x);
    int bt = int(gid.y);
    if (li >= p.n_leaf || bt >= p.nb) return;
    uint node_id = leaf_nodes[li];
    int num_opp = p.num_opp;
    int np = p.np;

    float zprod = 1.0f;
    for (int oi = 0; oi < num_opp; oi++) zprod *= zsum[li * num_opp + oi];
    if (zprod <= 0.0f) { cfv_bucket[li * p.nb + bt] = 0.0f; return; }

    // Per-leaf static state.
    int c[6];
    for (int q = 0; q < np; q++) c[q] = contributions[uint(node_id) * uint(np) + uint(q)];
    uint fold_mask = uint(folded_masks[node_id]);
    int traverser = p.traverser;
    int c_t = c[traverser];
    bool trav_folded = ((fold_mask >> uint(traverser)) & 1u) != 0u;
    float traverser_stake = float(p.starting_pot) / float(np) + float(c_t);

    // opponent player ids / contributions / folded
    int opp_pl[6]; int opp_c[6]; bool opp_f[6];
    for (int oi = 0; oi < num_opp; oi++) {
        int pl = (oi < traverser) ? oi : (oi + 1);
        opp_pl[oi] = pl; opp_c[oi] = c[pl];
        opp_f[oi] = ((fold_mask >> uint(pl)) & 1u) != 0u;
    }

    // Arm classification: all ACTIVE players equal contribution AND no fold.
    bool all_eq = true; int ref_c = -1;
    for (int q = 0; q < np; q++) {
        if (((fold_mask >> uint(q)) & 1u) != 0u) continue;
        if (ref_c < 0) ref_c = c[q];
        else if (c[q] != ref_c) { all_eq = false; break; }
    }
    bool arm1 = (all_eq && fold_mask == 0u);

    // Arm-1 rake-per-unit-stake; Arm-2 main-pot rake (computed in net_expected).
    int total = p.starting_pot;
    for (int q = 0; q < np; q++) total += c[q];
    float k = float(num_opp);
    float rpus = 0.0f;
    if (arm1) {
        float rake = max(min(float(total) * p.rake_rate, p.rake_cap), 0.0f);
        rpus = (traverser_stake > 0.0f) ? (rake / traverser_stake) : 0.0f;
    }
    // Arm-2: levels + main pot rake (main pot = lowest level × #contributors @≥ + starting_pot).
    // sorted unique contributions.
    int lv[6]; int nlv = 0;
    for (int q = 0; q < np; q++) {
        int v = c[q]; bool seen = false;
        for (int z = 0; z < nlv; z++) if (lv[z] == v) { seen = true; break; }
        if (!seen) lv[nlv++] = v;
    }
    for (int a = 0; a < nlv; a++) for (int b = a + 1; b < nlv; b++) if (lv[b] < lv[a]) { int t = lv[a]; lv[a] = lv[b]; lv[b] = t; }
    float main_pot_rake = 0.0f;
    if (!arm1 && nlv > 0) {
        int num_main = 0; for (int q = 0; q < np; q++) if (c[q] >= lv[0]) num_main++;
        int main_amt = lv[0] * num_main + p.starting_pot;
        main_pot_rake = max(min(float(main_amt) * p.rake_rate, p.rake_cap), 0.0f);
    }

    ulong seed = p.rng_seed
        ^ (ulong(node_id) * 0x9E3779B97F4A7C15UL)
        ^ (ulong(bt) * 0x9E3779B97F4A7C15UL);

    float sum_g = 0.0f;
    uint bo[9];
    for (uint s = 0; s < p.sample_m; s++) {
        for (int oi = 0; oi < num_opp; oi++) {
            int row = li * num_opp + oi;
            float u = sm_unif(seed) * zsum[row];
            const device float* pref = cdf + row * p.nb;
            int lo = 0, hi = p.nb;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (pref[mid] > u) hi = mid; else lo = mid + 1; }
            bo[oi] = uint(min(lo, p.nb - 1));
        }
        // opp-opp blocking weight (shared by both arms)
        float wpair = 1.0f; bool dead = false;
        for (int oi = 0; oi < num_opp && !dead; oi++) {
            uint b = bo[oi];
            for (int pj = 0; pj < oi; pj++) { float f = f_n[bo[pj] * p.nb + int(b)]; if (f == 0.0f) { dead = true; break; } wpair *= f; }
        }
        if (dead) continue;

        if (arm1) {
            float state[11]; for (int z = 0; z < 11; z++) state[z] = 0.0f; state[1] = 1.0f;
            for (int oi = 0; oi < num_opp && !dead; oi++) {
                uint b = bo[oi];
                int i = bt * p.nb + int(b);
                float fnn = f_n[i]; if (fnn == 0.0f) { dead = true; break; }
                float fw = f_w[i], ft = f_t[i], fl = f_l[i];
                float ns[11]; for (int z = 0; z < 11; z++) ns[z] = 0.0f;
                if (state[0] != 0.0f) ns[0] += state[0] * fnn;
                for (int j = 0; j <= oi; j++) { float sj = state[1 + j]; if (sj == 0.0f) continue;
                    if (fl != 0.0f) ns[0] += sj * fl; if (ft != 0.0f) ns[1 + j + 1] += sj * ft; if (fw != 0.0f) ns[1 + j] += sj * fw; }
                for (int z = 0; z < 11; z++) state[z] = ns[z];
            }
            if (dead) continue;
            float accum = 0.0f;
            if (state[0] != 0.0f) accum += state[0] * -1.0f;
            for (int j = 0; j <= num_opp; j++) { float sj = state[1 + j]; if (sj == 0.0f) continue;
                float nu; if (j == 0) nu = k - rpus; else { float tf = float(j + 1); nu = (k + 1.0f - tf) / tf - rpus / tf; }
                accum += sj * nu; }
            sum_g += wpair * traverser_stake * accum; // half_pot folded in here for arm1
        } else {
            // sc[oi] = [w,t,l, n_if_folded]
            float scw[6], sct[6], scl[6], scn[6];
            for (int oi = 0; oi < num_opp && !dead; oi++) {
                uint b = bo[oi]; int i = bt * p.nb + int(b);
                if (opp_f[oi]) { float n = f_n[i]; if (n == 0.0f) { dead = true; break; } scw[oi]=0; sct[oi]=0; scl[oi]=0; scn[oi]=n; }
                else { float w=f_w[i], t=f_t[i], l=f_l[i]; if (w==0.0f&&t==0.0f&&l==0.0f){dead=true;break;} scw[oi]=w; sct[oi]=t; scl[oi]=l; scn[oi]=0; }
            }
            if (dead) continue;
            // net_expected
            float s_total = 1.0f;
            for (int oi = 0; oi < num_opp; oi++) s_total *= opp_f[oi] ? scn[oi] : (scw[oi] + sct[oi] + scl[oi]);
            float cash = 0.0f; int prev_l = 0;
            for (int liv = 0; liv < nlv; liv++) {
                int lev = lv[liv];
                int pc = lev - prev_l;
                int num_contrib = 0; for (int q = 0; q < np; q++) if (c[q] >= lev) num_contrib++;
                float pot_l = float(pc * num_contrib);
                if (liv == 0) pot_l += float(p.starting_pot);
                if (pot_l == 0.0f) { prev_l = lev; continue; }
                bool trav_elig = (!trav_folded) && (c_t >= lev);
                int elig_count = trav_elig ? 1 : 0;
                for (int oi = 0; oi < num_opp; oi++) { if (opp_f[oi]) continue; if (opp_c[oi] < lev) continue; elig_count++; }
                if (elig_count == 0) {
                    if (c[traverser] >= lev) {
                        float tcl = float(pc) + ((liv == 0) ? (float(p.starting_pot) / float(np)) : 0.0f);
                        cash += tcl * s_total;
                    }
                    prev_l = lev; continue;
                }
                if (!trav_elig) { prev_l = lev; continue; }
                float m_out = 1.0f;
                for (int oi = 0; oi < num_opp; oi++) { if (opp_f[oi]) m_out *= scn[oi]; else if (opp_c[oi] < lev) m_out *= (scw[oi] + sct[oi] + scl[oi]); }
                float dp[7]; for (int z = 0; z < 7; z++) dp[z] = 0.0f; dp[0] = 1.0f;
                int ne = 0;
                for (int oi = 0; oi < num_opp; oi++) {
                    if (opp_f[oi] || opp_c[oi] < lev) continue;
                    float w = scw[oi], t = sct[oi];
                    dp[ne + 1] = 0.0f;
                    for (int j = ne; j >= 0; j--) { float d = dp[j];
                        if (d != 0.0f && t != 0.0f) dp[j + 1] += d * t;
                        dp[j] = (d != 0.0f && w != 0.0f) ? d * w : 0.0f; }
                    ne++;
                }
                float pot_ar = (liv == 0) ? (pot_l - main_pot_rake) : pot_l;
                for (int j = 0; j <= ne; j++) { float d = dp[j]; if (d == 0.0f) continue;
                    cash += m_out * d * (pot_ar / float(j + 1)); }
                prev_l = lev;
            }
            float val = cash - traverser_stake * s_total;
            sum_g += wpair * val;
        }
    }
    // Arm-1 folds half_pot into sum_g; Arm-2 already in absolute chips.
    cfv_bucket[li * p.nb + bt] = (zprod / float(p.sample_m)) * sum_g;
}

// Expand per-bucket cfv → per-hand cfv at the leaf, with /num_combinations.
// Grid (n_leaf, nh).
kernel void vcfr_continuation_expand(
    device float*          cfv            [[buffer(0)]],  // [nn*nh]
    device const float*    cfv_bucket     [[buffer(1)]],  // [n_leaf*nb]
    device const uint16_t* map            [[buffer(2)]],
    device const uint32_t* leaf_nodes     [[buffer(3)]],
    constant ContMwParams& p              [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int li = int(gid.x);
    int h  = int(gid.y);
    if (li >= p.n_leaf || h >= p.nh) return;
    uint node_id = leaf_nodes[li];
    uint bt = uint(map[h]);
    device float* out = cfv + uint(node_id) * uint(p.nh);
    if (bt == CONT_NO_BUCKET) { out[h] = 0.0f; return; }
    float v = cfv_bucket[li * p.nb + int(bt)];
    out[h] = (p.num_combinations > 0.0f) ? (v / p.num_combinations) : v;
}

// ============================================================================
// Fast LONE-SURVIVOR terminal (np=3 / num_opp=2), parallel over (terminal, hand).
//
// In a no-all-in depth-limited tree EVERY terminal is a fold ⇒ num_active<=1 ⇒
// constant payoff; only the opponent-pair compatible reach mass nh_count(h)
// varies per hand. The base `vcfr_bottom_up` runs this with 1 thread/node and
// an inner O(nh²) g0×g1 loop ⇒ O(nh³)/node single-threaded (the np≥3 wall). This
// kernel runs the SAME g0×g1 loop in the SAME order (⇒ bit-exact) but one thread
// per (terminal, hand), turning O(nh³)/node into O(nh²)/thread nh-way parallel.
// Mirrors multiway_brute_force_showdown's num_active<=1||traverser_folded path
// (vcfr.metal ~L408-476) + the /num_combinations normalization.
// ============================================================================

struct LoneTermParams {
    int nh;
    int np;
    int traverser;
    int n_term;
    int starting_pot;
    float rake_rate;
    float rake_cap;
    float num_combinations;
    // np=5 closed-form terminal fields (single Rust owner fills ALL fields —
    // the BottomUpParams under-fill UB class cannot recur here).
    int inner_role_base;   // 0: aggregates over opps 0,1,2; 1 (np5): opps 1,2,3
    int mc_samples;        // 0 = FULL outer enumeration; else M CDF draws
    uint32_t mc_seed;      // per-iteration seed (Rust XORs iteration in)
};

kernel void vcfr_lone_terminal_par(
    device float*          cfv            [[buffer(0)]],  // [nn*nh] write
    device const uint32_t* term_nodes     [[buffer(1)]],  // [n_term] lone-survivor terminals
    device const FlatNode* nodes          [[buffer(2)]],
    device const int32_t*  contributions  [[buffer(3)]],  // [nn*np]
    device const uint16_t* folded_masks   [[buffer(4)]],  // [nn]
    device const float*    reach          [[buffer(5)]],  // [nn*np*nh]
    device const uint8_t*  hand_cards     [[buffer(6)]],  // [nh*2]
    constant LoneTermParams& p            [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x);
    int h  = int(gid.y);
    if (ti >= p.n_term || h >= p.nh) return;
    int nh = p.nh; int np = p.np; int traverser = p.traverser;
    uint node_id = term_nodes[ti];
    uint16_t fold_mask = folded_masks[node_id];
    FlatNode node = nodes[node_id];

    int c_t = contributions[node_id * uint(np) + uint(traverser)];
    float traverser_stake = float(p.starting_pot) / float(np) + float(c_t);
    bool traverser_folded = (fold_mask & (uint16_t)(1u << traverser)) != 0;
    bool flop_seen = (node.board_state != 3);
    float eff_rake_rate = flop_seen ? p.rake_rate : 0.0f;
    float eff_rake_cap  = flop_seen ? p.rake_cap  : 0.0f;

    // payoff (mirror L409-448): main-pot-only rake; folded traverser loses stake.
    int total_pot = p.starting_pot;
    for (int q = 0; q < np; q++) total_pot += contributions[node_id * uint(np) + uint(q)];
    int min_contrib = contributions[node_id * uint(np) + 0u];
    for (int q = 1; q < np; q++) { int c = contributions[node_id * uint(np) + uint(q)]; if (c < min_contrib) min_contrib = c; }
    int num_main = 0;
    for (int q = 0; q < np; q++) if (contributions[node_id * uint(np) + uint(q)] >= min_contrib) num_main++;
    int main_pot_amount = min_contrib * num_main + p.starting_pot;
    float rake = fmax(0.0f, fmin(float(main_pot_amount) * eff_rake_rate, eff_rake_cap));
    float payoff = traverser_folded ? (-traverser_stake) : ((float(total_pot) - rake) - traverser_stake);

    // opponent reaches (the two non-traverser players), oi mapping as in bottom_up.
    int opp0 = (0 < traverser) ? 0 : 1;
    int opp1 = (1 < traverser) ? 1 : 2;
    const device float* reach_0 = reach + (node_id * uint(np) + uint(opp0)) * uint(nh);
    const device float* reach_1 = reach + (node_id * uint(np) + uint(opp1)) * uint(nh);

    int hc1 = hand_cards[h * 2];
    int hc2 = hand_cards[h * 2 + 1];
    float nh_count = 0.0f;
    for (int g0 = 0; g0 < nh; g0++) {
        int g0c1 = hand_cards[g0 * 2];
        int g0c2 = hand_cards[g0 * 2 + 1];
        if (g0c1 == hc1 || g0c1 == hc2 || g0c2 == hc1 || g0c2 == hc2) continue;
        float r0 = reach_0[g0];
        if (r0 == 0.0f) continue;
        for (int g1 = 0; g1 < nh; g1++) {
            int g1c1 = hand_cards[g1 * 2];
            int g1c2 = hand_cards[g1 * 2 + 1];
            if (g1c1 == hc1 || g1c1 == hc2 || g1c2 == hc1 || g1c2 == hc2) continue;
            if (g0c1 == g1c1 || g0c1 == g1c2 || g0c2 == g1c1 || g0c2 == g1c2) continue;
            float r1 = reach_1[g1];
            if (r1 == 0.0f) continue;
            nh_count += r0 * r1;
        }
    }
    float v = payoff * nh_count;
    cfv[node_id * uint(nh) + uint(h)] = (p.num_combinations > 0.0f) ? (v / p.num_combinations) : v;
}

// ============================================================================
// FACTORED lone-survivor terminal (np=3): same value as vcfr_lone_terminal_par
// but the inner O(nh) g1 loop is replaced by single-opponent inclusion-exclusion,
// giving O(nh)/thread (vs O(nh²)). The opp-1 compatible mass for a fixed g0 is
//   CompatR1(g0,h) = Sb(h) − Pb_h[g0.a] − Pb_h[g0.b] + r1[g0]
// where Sb(h)=Σ_{g1∌h} r1, Pb_h[c]=Σ_{g1∌h, c∈g1} r1 = P1[c] − r1[{c,h1}] − r1[{c,h2}].
// pair2hand[x*52+y] = hand index of {x,y} (or -1). NOT bit-exact to the brute
// loop (different float order); validated within tolerance.
// ============================================================================
kernel void vcfr_lone_terminal_factored(
    device float*          cfv            [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const FlatNode* nodes          [[buffer(2)]],
    device const int32_t*  contributions  [[buffer(3)]],
    device const uint16_t* folded_masks   [[buffer(4)]],
    device const float*    reach          [[buffer(5)]],
    device const uint8_t*  hand_cards     [[buffer(6)]],
    device const int32_t*  pair2hand      [[buffer(7)]],  // [52*52] hand idx or -1
    constant LoneTermParams& p            [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x);
    int h  = int(gid.y);
    if (ti >= p.n_term || h >= p.nh) return;
    int nh = p.nh; int np = p.np; int traverser = p.traverser;
    uint node_id = term_nodes[ti];
    uint16_t fold_mask = folded_masks[node_id];
    FlatNode node = nodes[node_id];

    int c_t = contributions[node_id * uint(np) + uint(traverser)];
    float traverser_stake = float(p.starting_pot) / float(np) + float(c_t);
    bool traverser_folded = (fold_mask & (uint16_t)(1u << traverser)) != 0;
    bool flop_seen = (node.board_state != 3);
    float eff_rake_rate = flop_seen ? p.rake_rate : 0.0f;
    float eff_rake_cap  = flop_seen ? p.rake_cap  : 0.0f;
    int total_pot = p.starting_pot;
    for (int q = 0; q < np; q++) total_pot += contributions[node_id * uint(np) + uint(q)];
    int min_contrib = contributions[node_id * uint(np) + 0u];
    for (int q = 1; q < np; q++) { int c = contributions[node_id * uint(np) + uint(q)]; if (c < min_contrib) min_contrib = c; }
    int num_main = 0;
    for (int q = 0; q < np; q++) if (contributions[node_id * uint(np) + uint(q)] >= min_contrib) num_main++;
    int main_pot_amount = min_contrib * num_main + p.starting_pot;
    float rake = fmax(0.0f, fmin(float(main_pot_amount) * eff_rake_rate, eff_rake_cap));
    float payoff = traverser_folded ? (-traverser_stake) : ((float(total_pot) - rake) - traverser_stake);

    int opp0 = (0 < traverser) ? 0 : 1;
    int opp1 = (1 < traverser) ? 1 : 2;
    const device float* reach_0 = reach + (node_id * uint(np) + uint(opp0)) * uint(nh);
    const device float* reach_1 = reach + (node_id * uint(np) + uint(opp1)) * uint(nh);

    int h1 = hand_cards[h * 2];
    int h2 = hand_cards[h * 2 + 1];

    // Per-card opp1 mass P1[c] and total S1 (O(nh)).
    float P1[52];
    for (int c = 0; c < 52; c++) P1[c] = 0.0f;
    float S1 = 0.0f;
    for (int g = 0; g < nh; g++) {
        float r = reach_1[g];
        if (r != 0.0f) { S1 += r; P1[hand_cards[g*2]] += r; P1[hand_cards[g*2+1]] += r; }
    }
    float r1h = reach_1[h];
    float Sb = S1 - P1[h1] - P1[h2] + r1h;

    // g0 loop with O(1) inner compatible mass.
    float nh_count = 0.0f;
    for (int g0 = 0; g0 < nh; g0++) {
        int a = hand_cards[g0*2];
        int b = hand_cards[g0*2+1];
        if (a == h1 || a == h2 || b == h1 || b == h2) continue;
        float r0 = reach_0[g0];
        if (r0 == 0.0f) continue;
        int i_ah1 = pair2hand[a*52 + h1]; float r1_ah1 = (i_ah1 >= 0) ? reach_1[i_ah1] : 0.0f;
        int i_ah2 = pair2hand[a*52 + h2]; float r1_ah2 = (i_ah2 >= 0) ? reach_1[i_ah2] : 0.0f;
        int i_bh1 = pair2hand[b*52 + h1]; float r1_bh1 = (i_bh1 >= 0) ? reach_1[i_bh1] : 0.0f;
        int i_bh2 = pair2hand[b*52 + h2]; float r1_bh2 = (i_bh2 >= 0) ? reach_1[i_bh2] : 0.0f;
        float Pb_a = P1[a] - r1_ah1 - r1_ah2;
        float Pb_b = P1[b] - r1_bh1 - r1_bh2;
        float compat = Sb - Pb_a - Pb_b + reach_1[g0];
        nh_count += r0 * compat;
    }
    float v = payoff * nh_count;
    cfv[node_id * uint(nh) + uint(h)] = (p.num_combinations > 0.0f) ? (v / p.num_combinations) : v;
}

// ============================================================================
// vcfr_aggregate_preflop_chance (Step 2.D.3)
//
// Computes:
//   out[class] = Σ over canonical of prob_table[canonical, class] × flop_cfvs[canonical, class]
//
// Same arithmetic order as CPU `aggregate_preflop_chance` in
// preflop_start_game.rs:779 — outer iteration over canonicals so the
// per-class sum sees canonical CFVs in iteration order. The prob_table
// is precomputed on CPU from PreflopChanceTable::chance_probability_flop
// and uploaded as a flat [n_canon * nh] f32 buffer.
//
// Threading: one thread per class. nh threads, each summing n_canon
// terms. For nh = 169 and n_canon = 1755 this is trivially parallel.
// ============================================================================

struct AggregatePreflopChanceParams {
    int n_canon;
    int nh;
};

kernel void vcfr_aggregate_preflop_chance(
    device float*       out                    [[buffer(0)]],   // [nh]
    device const float* prob_table             [[buffer(1)]],   // [n_canon * nh]
    device const float* flop_cfvs              [[buffer(2)]],   // [n_canon * nh]
    constant AggregatePreflopChanceParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int class_idx = int(gid);
    if (class_idx >= params.nh) return;
    float sum = 0.0f;
    for (int c = 0; c < params.n_canon; c++) {
        sum += prob_table[c * params.nh + class_idx]
             * flop_cfvs[c * params.nh + class_idx];
    }
    out[class_idx] = sum;
}

// ============================================================================
// vcfr_preflop_bottom_up_player (Step 2.D.4)
//
// Processes one level of preflop PLAYER decision nodes. Per (node, lane)
// thread:
//   1. If traverser-owned: cfv_avg = Σ strategy[a, lane] × cfv[child_a][lane].
//      Else (opp): cfv_avg = Σ cfv[child_a][lane] (plain sum).
//   2. Write cfv[node][lane] = cfv_avg.
//   3. Traverser only: for each a, inst_regret = cfv[child_a][lane] - cfv_avg;
//      regrets[a, lane] = (old >= 0 ? alpha_t : beta_t) × old + inst_regret;
//      cum_strategy[a, lane] = gamma_t × old + strategy[a, lane].
//
// Mirrors CPU `bottom_up_recursive` in preflop_cfr.rs:598 — same per-a
// iteration order, same DCFR formula. cfv buffer layout: [node_id × lanes
// + lane] (mirror of CPU `cfv: Vec<Vec<f32>>` with shape [nn][n_classes]).
// strategy/regrets/cum_strategy: [infoset_id × MAX_NA × lanes + a × lanes + lane].
// ============================================================================

struct PreflopBottomUpParams {
    int level_count;
    int lanes;
    uint32_t traverser;
    float alpha_t;
    float beta_t;
    float gamma_t;
};

kernel void vcfr_preflop_bottom_up_player(
    device const uint32_t* level_nodes        [[buffer(0)]],
    constant PreflopBottomUpParams& params    [[buffer(1)]],
    device const FlatNode* nodes              [[buffer(2)]],
    device const uint32_t* children           [[buffer(3)]],
    device const uint32_t* infoset_offsets    [[buffer(4)]],
    device const float* strategy              [[buffer(5)]],
    device float* regrets                     [[buffer(6)]],
    device float* cum_strategy                [[buffer(7)]],
    device float* cfv                         [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int idx = int(gid.x);
    int lane = int(gid.y);
    int lanes = params.lanes;
    if (idx >= params.level_count || lane >= lanes) return;

    uint node_id = level_nodes[idx];
    FlatNode node = nodes[node_id];
    int na = int(node.num_children);
    int stride = MAX_NA * lanes;

    bool is_traverser = (int(node.player_id) == int(params.traverser));

    float cfv_avg = 0.0f;
    if (is_traverser) {
        uint infoset_id = infoset_offsets[node_id];
        const device float* sigma = strategy + infoset_id * stride;
        for (int a = 0; a < na; a++) {
            uint child = children[node.children_start + a];
            float s = sigma[a * lanes + lane];
            float v = cfv[child * lanes + lane];
            cfv_avg += s * v;
        }
    } else {
        for (int a = 0; a < na; a++) {
            uint child = children[node.children_start + a];
            cfv_avg += cfv[child * lanes + lane];
        }
    }
    cfv[node_id * lanes + lane] = cfv_avg;

    if (is_traverser) {
        uint infoset_id = infoset_offsets[node_id];
        const device float* sigma = strategy + infoset_id * stride;
        device float* r = regrets + infoset_id * stride;
        device float* g = cum_strategy + infoset_id * stride;
        for (int a = 0; a < na; a++) {
            uint child = children[node.children_start + a];
            float v_child = cfv[child * lanes + lane];
            float inst_regret = v_child - cfv_avg;
            int ridx = a * lanes + lane;
            float old_r = r[ridx];
            float coef = (old_r >= 0.0f) ? params.alpha_t : params.beta_t;
            r[ridx] = coef * old_r + inst_regret;
            g[ridx] = params.gamma_t * g[ridx] + sigma[ridx];
        }
    }
}

// ============================================================================
// vcfr_top_down_reach
// ============================================================================

struct TopDownParams {
    int level_count;
    int num_players;
    int nh;
};

kernel void vcfr_top_down_reach(
    device const uint32_t* level_nodes       [[buffer(0)]],
    constant TopDownParams& params           [[buffer(1)]],
    device const FlatNode* nodes             [[buffer(2)]],
    device const uint32_t* children          [[buffer(3)]],
    device const float* strategy             [[buffer(4)]],
    device const uint32_t* infoset_offsets   [[buffer(5)]],
    device float* reach                      [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int idx = int(gid.x);
    int h = int(gid.y);

    if (idx >= params.level_count || h >= params.nh) return;

    uint node_id = level_nodes[idx];
    FlatNode node = nodes[node_id];
    int np = params.num_players;
    int nh = params.nh;
    int node_reach_base = node_id * np * nh;

    if (node.node_type == NODE_TYPE_PLAYER) {
        int player = int(node.player_id);
        uint infoset_id = infoset_offsets[node_id];
        int stride = MAX_NA * nh;
        const device float* sigma = strategy + infoset_id * stride;

        for (int a = 0; a < int(node.num_children); a++) {
            uint child = children[node.children_start + a];
            int child_reach_base = child * np * nh;

            for (int p = 0; p < np; p++) {
                reach[child_reach_base + p * nh + h] = reach[node_reach_base + p * nh + h];
            }
            reach[child_reach_base + player * nh + h] *= sigma[a * nh + h];
        }
    } else {
        for (int a = 0; a < int(node.num_children); a++) {
            uint child = children[node.children_start + a];
            int child_reach_base = child * np * nh;
            for (int p = 0; p < np; p++) {
                reach[child_reach_base + p * nh + h] = reach[node_reach_base + p * nh + h];
            }
        }
    }
}

// ============================================================================
// vcfr_bottom_up
//
// Bottom-up CFV computation + regret update.
// Processes one level at a time (called per level from host).
// Each thread processes one node (blockDim=1).
//
// Terminal nodes: evaluate showdown using sorted-sweep.
// Chance nodes: sum child CFVs.
// Player nodes (traverser): compute strategy-weighted avg CFV, update regrets + cum_strategy.
// Player nodes (opponent): sum child CFVs.
// ============================================================================

struct BottomUpParams {
    int level_count;
    int num_players;
    int nh;
    uint32_t traverser;
    float alpha_t;
    float beta_t;
    float gamma_t;
    float regret_floor;
    int32_t starting_pot;
    float num_combinations;
    // When != 0, the terminal branch SKIPS lone-survivor terminals
    // (num_active <= 1): they are pre-filled by vcfr_lone_terminal_par
    // (parallel over hands) before the level loop. Field 11 — kept in
    // sync with the Rust BuParams (solver.rs) which sends 11 fields.
    int32_t skip_lone_terminals;
    // When != 0, accumulate per-action counterfactual values into last_cfv for
    // the QRE strategy (vcfr_compute_strategies_qre). Field 12 — kept in sync
    // with the Rust BuParams (now 12 fields).
    int32_t lambda_active;
    // ─── Slice 2 rake (CPU↔Metal parity) ───
    // Mirror CPU `tree.rake_rate, tree.rake_cap`. Per-terminal gating is
    // applied at evaluation site as: `eff_rate/eff_cap = flop_seen ? rake : 0`.
    float rake_rate;
    float rake_cap;
    // ─── Pluribus-style negative-regret pruning (P1) ───
    // Mirror BatchedParams's pruning fields. Mandatory carve-outs:
    //   1. pruning_enabled master switch (off → no effect)
    //   2. re_enable_iter: every Kth iter we traverse all (5% of iters
    //      when stride=20) so dormant actions get a chance to recover.
    //   3. board_state != 2: NEVER prune on the river (last betting round).
    //   4. action_leads_to_terminal: NEVER prune actions leading directly
    //      to terminal nodes (skipping these can cause CFV mis-estimation).
    int32_t pruning_enabled;
    float pruning_threshold;
    int32_t iteration;
    int32_t pruning_stride;
    int32_t board_state;
};

kernel void vcfr_bottom_up(
    device const uint32_t* level_nodes       [[buffer(0)]],
    constant BottomUpParams& params          [[buffer(1)]],
    device const FlatNode* nodes             [[buffer(2)]],
    device const uint32_t* children          [[buffer(3)]],
    device const int32_t* contributions      [[buffer(4)]],
    device const uint16_t* folded_masks      [[buffer(5)]],
    device const float* strategy             [[buffer(6)]],
    device const uint32_t* infoset_offsets   [[buffer(7)]],
    device const float* reach                [[buffer(8)]],
    device float* cfv                        [[buffer(9)]],
    device float* regrets                    [[buffer(10)]],
    device float* cum_strategy               [[buffer(11)]],
    device const float* initial_weight       [[buffer(12)]],
    device const uint16_t* sorted_opp_strength  [[buffer(13)]],
    device const uint16_t* sorted_opp_indices    [[buffer(14)]],
    device const uint16_t* sorted_pl_strength   [[buffer(15)]],
    device const uint16_t* sorted_pl_indices     [[buffer(16)]],
    device const uint8_t* hand_cards          [[buffer(17)]],
    // ── Step 5: chokepoint instrumentation (Phase B completion guard) ──
    // Per-(terminal-node, hand) marker, sized [nn * nh] u8.
    //   0 = unmarked (BUG: terminal bypassed the chokepoint)
    //   1 = rake-applied (flop_seen=true at this terminal)
    //   2 = rake-correctly-skipped (flop_seen=false, no-flop-no-drop)
    // Written right after multiway_brute_force_showdown returns.
    device uchar* rake_marker                [[buffer(18)]],
    // QRE: accumulated per-(infoset, action, hand) counterfactual values. Only
    // written when params.lambda_active != 0; read by vcfr_compute_strategies_qre.
    device float* last_cfv                   [[buffer(19)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    if (idx >= params.level_count) return;

    uint node_id = level_nodes[idx];
    FlatNode node = nodes[node_id];
    int np = params.num_players;
    int nh = params.nh;
    int num_opp = np - 1;

    // ═══ TERMINAL NODE ═══
    if (node.node_type == NODE_TYPE_TERMINAL) {
        int node_reach_base = node_id * np * nh;
        uint16_t fold_mask = folded_masks[node_id];
        device float* out = cfv + node_id * nh;

        // Fast-terminal skip: lone-survivor terminals (num_active <= 1) are
        // pre-filled by vcfr_lone_terminal_par (parallel over hands) before
        // this level loop. Their cfv is already correct — leave it.
        if (params.skip_lone_terminals != 0) {
            int n_active = 0;
            for (int p = 0; p < np; p++) if (!(fold_mask & (1u << p))) n_active++;
            if (n_active <= 1) return;
        }

        // Copy contributions to thread-local memory for the brute-force helper.
        int32_t contribs_local[8];
        for (int p = 0; p < np; p++) {
            contribs_local[p] = contributions[node_id * np + p];
        }

        // Copy opponent reach to thread-local. CRITICAL: do NOT zero out
        // folded opponents' reach — the brute-force enumerates over their
        // hand assignments (even though they don't affect the winner),
        // and the reach product (r_a * r_b) provides the correct scenario
        // weighting for zero-sum to hold across players.
        float opp_reach_local[5 * 1326];
        for (int oi = 0; oi < num_opp; oi++) {
            int opp = (oi < int(params.traverser)) ? oi : (oi + 1);
            const device float* opp_r = reach + node_reach_base + opp * nh;
            for (int h = 0; h < nh; h++) opp_reach_local[oi * nh + h] = opp_r[h];
        }

        // Per-scenario brute-force showdown evaluation (matches CPU
        // side_pot_showdown_cfv_with_rake). Returns raw CFV with rake
        // applied per CPU spec; we apply /num_combinations below as
        // the caller normalization.
        // flop_seen: per-terminal no-flop-no-drop gate. Today
        // node.board_state ∈ {0=Flop, 1=Turn, 2=River} for all
        // flop-onward terminals → flop_seen=true. When preflop
        // integration brings preflop-end terminals, those have
        // board_state=3 (Preflop) → flop_seen=false → rake collapses.
        // Static-anchored in gpu_rake_parity_gate.rs against the
        // BoardState::Preflop repr.
        bool flop_seen = (node.board_state != 3);
        float local_out[1326];
        multiway_brute_force_showdown(
            nh, np, int(params.traverser),
            params.starting_pot, fold_mask,
            contribs_local, opp_reach_local,
            hand_cards, sorted_pl_strength, sorted_pl_indices,
            params.rake_rate, params.rake_cap, flop_seen,
            local_out
        );
        for (int h = 0; h < nh; h++) out[h] = local_out[h];

        // ── Step 5: chokepoint instrumentation marker write ──
        // Every production terminal evaluation goes through this site,
        // so writing a marker here proves the terminal was rake-processed.
        // 1 = rake-applied (flop_seen=true), 2 = rake-correctly-skipped.
        uchar marker = flop_seen ? (uchar)1 : (uchar)2;
        for (int h = 0; h < nh; h++) {
            rake_marker[node_id * nh + h] = marker;
        }

        if (params.num_combinations > 0.0f) {
            for (int h = 0; h < nh; h++) out[h] /= params.num_combinations;
        }
        return;
    }


    // ═══ CHANCE NODE ═══
    if (node.node_type == NODE_TYPE_CHANCE) {
        // Childless chance node = depth-limit CONTINUATION LEAF: its cfv is
        // pre-filled by vcfr_continuation_fill (run after the d_cfv zero,
        // before this level loop). Keep it — do NOT re-zero. Full trees have
        // no childless chance nodes, so this is byte-exact for existing uses.
        if (node.num_children == 0) return;
        for (int h = 0; h < nh; h++) cfv[node_id * nh + h] = 0.0f;
        for (int a = 0; a < int(node.num_children); a++) {
            uint child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                cfv[node_id * nh + h] += cfv[child * nh + h];
            }
        }
        return;
    }

    // ═══ PLAYER NODE ═══
    int owner = int(node.player_id);
    int na = int(node.num_children);
    uint infoset_id = infoset_offsets[node_id];
    int stride = MAX_NA * nh;
    const device float* sigma = strategy + infoset_id * stride;

    float cfv_avg[1326];
    for (int h = 0; h < nh; h++) cfv_avg[h] = 0.0f;

    if (owner == int(params.traverser)) {
        // Strategy-weighted average CFV
        for (int a = 0; a < na; a++) {
            uint child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                cfv_avg[h] += sigma[a * nh + h] * cfv[child * nh + h];
            }
        }

        // Update regrets with DCFR discount
        int offset = infoset_id * stride;
        for (int a = 0; a < na; a++) {
            uint child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                float inst_regret = cfv[child * nh + h] - cfv_avg[h];
                uint ridx = offset + a * nh + h;
                float coef = (regrets[ridx] >= 0.0f) ? params.alpha_t : params.beta_t;
                regrets[ridx] = coef * regrets[ridx] + inst_regret;
                if (regrets[ridx] < params.regret_floor) regrets[ridx] = params.regret_floor;
            }
        }

        // Update cumulative strategy
        for (int a = 0; a < na; a++) {
            for (int h = 0; h < nh; h++) {
                uint cidx = offset + a * nh + h;
                cum_strategy[cidx] = params.gamma_t * cum_strategy[cidx] + sigma[a * nh + h];
            }
        }

        // QRE: accumulate the per-action counterfactual value (cfv[child]) so the
        // logit (vcfr_compute_strategies_qre) responds to the time-average value.
        if (params.lambda_active != 0) {
            for (int a = 0; a < na; a++) {
                uint child = children[node.children_start + a];
                for (int h = 0; h < nh; h++) {
                    last_cfv[offset + a * nh + h] += cfv[child * nh + h];
                }
            }
        }
    } else {
        // Opponent: sum child CFVs
        for (int a = 0; a < na; a++) {
            uint child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                cfv_avg[h] += cfv[child * nh + h];
            }
        }
    }

    device float* out = cfv + node_id * nh;
    for (int h = 0; h < nh; h++) {
        out[h] = cfv_avg[h];
    }
}

// ============================================================================
// Phase 3: Flop-start kernels
// ============================================================================

// ============================================================================
// vcfr_regret_apply
// Apply batched regret accumulation with DCFR discount.
// regrets[i] = coef * regrets[i] + regret_accum[i]
// regret_accum[i] = 0
// ============================================================================

kernel void vcfr_regret_apply(
    device float* regrets             [[buffer(0)]],
    device float* regret_accum        [[buffer(1)]],
    constant int& total_size          [[buffer(2)]],
    constant float& alpha_t           [[buffer(3)]],
    constant float& beta_t            [[buffer(4)]],
    constant float& regret_floor      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    if (idx >= total_size) return;

    float ir = regret_accum[idx];
    float old_r = regrets[idx];
    float coef = (old_r >= 0.0f) ? alpha_t : beta_t;
    regrets[idx] = coef * old_r + ir;
    if (regrets[idx] < regret_floor) regrets[idx] = regret_floor;
    regret_accum[idx] = 0.0f;
}

// ============================================================================
// vcfr_cum_apply
// Apply cum_strategy accumulation with gamma discount.
// cum_strategy[i] = gamma_t * cum_strategy[i] + cum_accum[i]
// cum_accum[i] = 0
// ============================================================================

kernel void vcfr_cum_apply(
    device float* cum_strategy        [[buffer(0)]],
    device float* cum_accum           [[buffer(1)]],
    constant int& total_size          [[buffer(2)]],
    constant float& gamma_t           [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    if (idx >= total_size) return;

    cum_strategy[idx] = gamma_t * cum_strategy[idx] + cum_accum[idx];
    cum_accum[idx] = 0.0f;
}

// ============================================================================
// vcfr_chance_accumulate
// Accumulate prob * cfv[child] into cfv_accum[child] for a single outcome.
// ============================================================================

kernel void vcfr_chance_accumulate(
    device float* cfv_accum           [[buffer(0)]],
    device const float* cfv           [[buffer(1)]],
    device const float* chance_prob   [[buffer(2)]],
    device const uint32_t* chance_child_ids [[buffer(3)]],
    constant int& num_chance_children [[buffer(4)]],
    constant int& nh                  [[buffer(5)]],
    constant int& outcome             [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    int total = num_chance_children * nh;
    if (idx >= total) return;
    int cn = idx / nh;
    int h = idx % nh;
    uint child_id = chance_child_ids[cn];
    float prob = chance_prob[outcome * nh + h];
    cfv_accum[child_id * nh + h] += prob * cfv[child_id * nh + h];
}

// ============================================================================
// vcfr_chance_finalize
// Copy cfv_accum[child] into cfv[child].
// ============================================================================

kernel void vcfr_chance_finalize(
    device float* cfv                 [[buffer(0)]],
    device const float* cfv_accum     [[buffer(1)]],
    device const uint32_t* chance_child_ids [[buffer(2)]],
    constant int& num_chance_children [[buffer(3)]],
    constant int& nh                  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    int total = num_chance_children * nh;
    if (idx >= total) return;
    int cn = idx / nh;
    int h = idx % nh;
    uint child_id = chance_child_ids[cn];
    cfv[child_id * nh + h] = cfv_accum[child_id * nh + h];
}

// ============================================================================
// vcfr_chance_accumulate_grouped
// Accumulate per-outcome CFVs into per-group accumulators.
// Each outcome belongs to a group (e.g., turn card).
// Uses atomic fetch-and-add for correctness when multiple outcomes
// map to the same group.
// ============================================================================

kernel void vcfr_chance_accumulate_grouped(
    device float* cfv_accum           [[buffer(0)]],
    device const float* cfv_batch     [[buffer(1)]],
    device const float* chance_prob   [[buffer(2)]],
    device const uint32_t* chance_child_ids [[buffer(3)]],
    device const int32_t* outcome_to_group [[buffer(4)]],
    constant int& num_outcomes        [[buffer(5)]],
    constant int& num_chance_children [[buffer(6)]],
    constant int& nn_val              [[buffer(7)]],
    constant int& nh                  [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    int total = num_outcomes * num_chance_children * nh;
    if (idx >= total) return;

    int h = idx % nh;
    int cn = (idx / nh) % num_chance_children;
    int outcome = idx / (num_chance_children * nh);

    uint child_id = chance_child_ids[cn];
    int group = outcome_to_group[outcome];
    float prob = chance_prob[outcome * nh + h];
    float val = prob * cfv_batch[outcome * nn_val * nh + child_id * nh + h];

    // Metal doesn't have atomicAdd for float, but since block=1 and
    // each (outcome, child, h) is unique, multiple threads write to
    // different group locations. Actually, different outcomes in the
    // same group CAN write to the same location. Use a simple race:
    // since Metal on Apple Silicon has coherent L2, we can use
    // atomic_fetch_add_explicit on a float via reinterpret.
    // For safety, use the direct write since each thread handles
    // a unique (outcome, cn, h) triple, and group is determined by outcome.
    // Different outcomes with same group → collision. Use critical section.
    // NOTE: Metal atomics on float require osx 10.14+ / ios 11.0+.
    // We use the workaround: reinterpret as int and use atomic_fetch_add.
    device atomic_int* target = (device atomic_int*)(cfv_accum + group * nn_val * nh + child_id * nh + h);
    union { int i; float f; } u;
    u.f = val;
    atomic_fetch_add_explicit(target, u.i, metal::memory_order_relaxed);
}



// ============================================================================
// vcfr_bottom_up_batched (per-outcome dimensional regrets)
//
// Process multiple outcomes in parallel for a single level of the tree.
// One thread per (outcome, node_in_level) pair.
// Each thread processes all nh hands sequentially.
//
// PER-OUTCOME REGRET ARCHITECTURE:
// Each outcome writes to its own dimensional regret slot — no atomics needed.
// Regret layout: regrets[outcome * regret_outcome_stride + infoset_id * MAX_NA * nh + a * nh + h]
// DCFR discount applied inline at each traverser node.
// ============================================================================

struct BatchedParams {
    int32_t level_count;            // number of nodes at this level
    int32_t num_outcomes;           // number of outcomes (river cards or turn cards)
    int32_t cfv_batch_stride;       // nn * nh
    int32_t sorted_opp_stride;      // num_opp * nh
    int32_t num_players;
    int32_t nh;
    uint32_t traverser;
    float alpha_t;
    float beta_t;
    float gamma_t;
    float regret_floor;
    int32_t starting_pot;
    float num_combinations;
    int32_t regret_outcome_stride;  // stride between outcomes in regrets buffer
    int32_t cum_outcome_stride;     // stride between outcomes in cum_strategy buffer
    // ─── Negative-regret pruning (Phase 1.A, Option A) ───
    int32_t pruning_enabled;        // 0 = off (default), nonzero = on
    float pruning_threshold;        // regret < this & carve-outs ok → skip update
    int32_t iteration;              // for stochastic re-enable
    int32_t pruning_stride;         // re-enable every Kth iter (don't prune that iter)
    int32_t board_state;            // 0=flop, 1=turn, 2=river (never prune river)
    // ─── Slice 2 rake (CPU↔Metal parity) ───
    float rake_rate;
    float rake_cap;
};

kernel void vcfr_bottom_up_batched(
    device const uint32_t* level_nodes       [[buffer(0)]],
    constant BatchedParams& params           [[buffer(1)]],
    device const FlatNode* nodes             [[buffer(2)]],
    device const uint32_t* children          [[buffer(3)]],
    device const int32_t* contributions      [[buffer(4)]],
    device const uint16_t* folded_masks      [[buffer(5)]],
    device const float* strategy             [[buffer(6)]],
    device const uint32_t* infoset_offsets   [[buffer(7)]],
    device const float* reach                [[buffer(8)]],
    device float* cfv                        [[buffer(9)]],
    device float* regrets                    [[buffer(10)]],
    device float* cum_strategy               [[buffer(11)]],
    device const float* initial_weight       [[buffer(12)]],
    device const uint16_t* sorted_opp_str    [[buffer(13)]],
    device const uint16_t* sorted_opp_idx    [[buffer(14)]],
    device const uint16_t* sorted_pl_str     [[buffer(15)]],
    device const uint16_t* sorted_pl_idx     [[buffer(16)]],
    device const uint8_t* hand_cards         [[buffer(17)]],
    device const float* chance_prob          [[buffer(18)]],
    device float* debug_out                  [[buffer(19)]],
    // ── Step 5: chokepoint instrumentation marker buffer ──
    // Same semantics as vcfr_bottom_up's rake_marker (buffer 18 there).
    device uchar* rake_marker                [[buffer(20)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    int outcome = idx / params.level_count;
    int node_in_level = idx % params.level_count;
    if (outcome >= params.num_outcomes) return;

    uint node_id = level_nodes[node_in_level];
    FlatNode node = nodes[node_id];
    int np = params.num_players;
    int nh = params.nh;
    int num_opp = np - 1;

    // Per-outcome offsets for CFV and sorted arrays
    int cfv_off = outcome * params.cfv_batch_stride;
    int sos_off = outcome * params.sorted_opp_stride;
    int sps_off = outcome * params.sorted_opp_stride; // same layout as opp: num_opp * nh per outcome

    const device uint16_t* opp_str = sorted_opp_str + sos_off;
    const device uint16_t* opp_idx = sorted_opp_idx + sos_off;
    const device uint16_t* pl_str = sorted_pl_str + sps_off;
    const device uint16_t* pl_idx = sorted_pl_idx + sps_off;
    device float* cfv_o = cfv + cfv_off;

    // ═══ TERMINAL NODE (brute-force showdown, matches CPU side_pot_showdown_cfv) ═══
    if (node.node_type == NODE_TYPE_TERMINAL) {
        int node_reach_base = int(node_id) * np * nh;
        uint16_t fold_mask = folded_masks[int(node_id)];
        device float* out = cfv_o + int(node_id) * nh;

        // Copy contributions to thread-local for the helper.
        int32_t contribs_local[8];
        for (int p = 0; p < np; p++) {
            contribs_local[p] = contributions[int(node_id) * np + p];
        }

        // Build opp_reach_local. CRITICAL: do NOT zero folded opponents'
        // reach (the brute-force enumerates over their hand assignments and
        // the reach product r_a*r_b carries the correct scenario weighting).
        // DO apply chance_prob masking for board-card-conflicting hands.
        float opp_reach_local[5 * 1326];
        for (int oi = 0; oi < num_opp; oi++) {
            int opp = (oi < int(params.traverser)) ? oi : (oi + 1);
            const device float* opp_r = reach + node_reach_base + opp * nh;
            for (int h = 0; h < nh; h++) {
                opp_reach_local[oi * nh + h] = (chance_prob[h] == 0.0f) ? 0.0f : opp_r[h];
            }
        }

        // Run the brute-force showdown helper. Slice 2: pass rake
        // params + flop_seen (board_state != 3=Preflop) for the
        // no-flop-no-drop gate. See comment at the bottom_up caller
        // for the dormant-but-correct Preflop handling.
        bool flop_seen = (node.board_state != 3);
        float local_out[1326];
        multiway_brute_force_showdown(
            nh, np, int(params.traverser),
            params.starting_pot, fold_mask,
            contribs_local, opp_reach_local,
            hand_cards, pl_str, pl_idx,
            params.rake_rate, params.rake_cap, flop_seen,
            local_out
        );
        for (int h = 0; h < nh; h++) out[h] = local_out[h];

        // ── Step 5: chokepoint instrumentation marker write ──
        // Every production terminal evaluation in vcfr_bottom_up_batched
        // goes through this site, so the marker proves the terminal was
        // rake-processed. 1 = rake-applied, 2 = rake-correctly-skipped.
        uchar marker = flop_seen ? (uchar)1 : (uchar)2;
        for (int h = 0; h < nh; h++) {
            rake_marker[node_id * nh + h] = marker;
        }

        if (params.num_combinations > 0.0f) {
            for (int h = 0; h < nh; h++) out[h] /= params.num_combinations;
        }
        return;
    }


    // ═══ CHANCE NODE ═══
    if (node.node_type == NODE_TYPE_CHANCE) {
        device float* out = cfv_o + int(node_id) * nh;
        for (int h = 0; h < nh; h++) out[h] = 0.0f;
        for (int a = 0; a < int(node.num_children); a++) {
            uint child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                out[h] += cfv_o[int(child) * nh + h];
            }
        }
        return;
    }

    // ═══ PLAYER NODE ═══
    int owner = int(node.player_id);
    int na = int(node.num_children);
    uint infoset_id = infoset_offsets[int(node_id)];
    int stride = MAX_NA * nh;

    // Per-outcome strategy comes from per-outcome regrets (dimensional layout)
    int strat_outcome_off = outcome * params.regret_outcome_stride;
    const device float* sigma = strategy + strat_outcome_off + int(infoset_id) * stride;

    float cfv_avg[1326];
    for (int h = 0; h < nh; h++) cfv_avg[h] = 0.0f;

    if (owner == int(params.traverser)) {
        // Compute weighted CFV
        for (int a = 0; a < na; a++) {
            uint child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                cfv_avg[h] += sigma[a * nh + h] * cfv_o[int(child) * nh + h];
            }
        }

        // Per-outcome regret update with inline DCFR discount
        // Layout: regrets[outcome * regret_outcome_stride + infoset_id * MAX_NA * nh + a * nh + h]
        int regret_base = outcome * params.regret_outcome_stride + int(infoset_id) * stride;
        int cum_base = outcome * params.cum_outcome_stride + int(infoset_id) * stride;
        for (int a = 0; a < na; a++) {
            uint child = children[node.children_start + a];

            // ─── Negative-regret pruning carve-outs (Phase 1.A, per action) ───
            // Apply skip only when ALL conditions hold:
            //   1. pruning_enabled is on
            //   2. NOT on a re-enable iteration (every Kth iter we traverse all)
            //   3. NOT on the last street (river); we always update river regrets
            //   4. action does NOT lead directly to a terminal node
            // The per-hand skip on regret < threshold happens inside the inner
            // loop so we get a fine-grained measurement of prunable fraction.
            bool re_enable_iter = (params.pruning_stride > 0)
                && (params.iteration % params.pruning_stride == 0);
            bool action_leads_to_terminal = (nodes[child].node_type == NODE_TYPE_TERMINAL);
            bool can_prune_this_action = (params.pruning_enabled != 0)
                && !re_enable_iter
                && (params.board_state != 2)
                && !action_leads_to_terminal;

            for (int h = 0; h < nh; h++) {
                float inst_regret = cfv_o[int(child) * nh + h] - cfv_avg[h];
                int ridx = regret_base + a * nh + h;

                // Inline DCFR discount (matches CPU bottom_up_zone)
                float old_r = regrets[ridx];

                // Per-(action, hand) pruning skip: leave regret unchanged.
                // Note Option A skips only the regret-update arithmetic; the
                // CFV computation (cfv_o reads above) already happened in the
                // level below. Option B would skip subtree traversal entirely.
                if (can_prune_this_action && old_r < params.pruning_threshold) {
                    // Skip this regret update (per Pluribus negative-regret pruning).
                    // Cumulative strategy update also skipped — under reset-on-recovery
                    // semantics, the pruned-action strategy stays at last known value.
                    continue;
                }

                float coef = (old_r >= 0.0f) ? params.alpha_t : params.beta_t;
                float new_r = coef * old_r + inst_regret;
                if (new_r < params.regret_floor) new_r = params.regret_floor;
                regrets[ridx] = new_r;

                // Cumulative strategy update (inline gamma discount)
                int cidx = cum_base + a * nh + h;
                cum_strategy[cidx] = params.gamma_t * cum_strategy[cidx] + sigma[a * nh + h];
            }
        }
    } else {
        // Non-traverser: just sum child CFVs (unweighted)
        for (int a = 0; a < na; a++) {
            uint child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                cfv_avg[h] += cfv_o[int(child) * nh + h];
            }
        }
    }

    device float* out_node = cfv_o + int(node_id) * nh;
    for (int h = 0; h < nh; h++) {
        out_node[h] = cfv_avg[h];
    }
}

// ============================================================================
// vcfr_compute_strategies_batched
//
// Compute strategies from regrets for multiple outcomes in parallel.
// One thread per (outcome, infoset, hand) triple.
// Layout: regrets[outcome * outcome_stride + infoset * MAX_NA * nh + a * nh + h]
// ============================================================================

struct BatchedStrategiesParams {
    int32_t num_outcomes;
    int32_t num_infosets;
    int32_t nh;
    int32_t outcome_stride;   // stride between outcomes in regrets/strategy
    int32_t base_offset;      // base offset into regrets/strategy buffer for this zone
};

kernel void vcfr_compute_strategies_batched(
    device const float* regrets             [[buffer(0)]],
    device float*       strategy            [[buffer(1)]],
    device const uint32_t* decision_node_ids [[buffer(2)]],
    device const FlatNode* nodes            [[buffer(3)]],
    device const uint32_t* infoset_offsets  [[buffer(4)]],
    constant BatchedStrategiesParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x encodes (outcome * num_infosets + infoset_id), gid.y = hand
    int flat = int(gid.x);
    int outcome = flat / params.num_infosets;
    int infoset_id = flat % params.num_infosets;
    int h = int(gid.y);
    if (outcome >= params.num_outcomes) return;
    if (infoset_id >= params.num_infosets) return;
    if (h >= params.nh) return;

    uint node_id = decision_node_ids[infoset_id];
    FlatNode node = nodes[node_id];
    int na = int(node.num_children);
    int stride = MAX_NA * params.nh;
    int base = params.base_offset + outcome * params.outcome_stride + infoset_id * stride;

    const device float* r = regrets + base;
    device float* s = strategy + base;

    // Bug B fix: regret matching epsilon
    const float REGRET_MATCH_EPS_B = 1e-5f;

    float pos_sum = 0.0f;
    for (int a = 0; a < na; a++) {
        float rv = r[a * params.nh + h];
        if (rv > REGRET_MATCH_EPS_B) pos_sum += rv;
    }

    if (pos_sum > 0.0f) {
        for (int a = 0; a < na; a++) {
            float rv = r[a * params.nh + h];
            s[a * params.nh + h] = (rv > REGRET_MATCH_EPS_B) ? rv / pos_sum : 0.0f;
        }
    } else {
        float u = 1.0f / float(na);
        for (int a = 0; a < na; a++) {
            s[a * params.nh + h] = u;
        }
    }

    for (int a = na; a < MAX_NA; a++) {
        s[a * params.nh + h] = 0.0f;
    }
}

// ============================================================================
// vcfr_seed_reach
//
// Zero a reach buffer, then copy reach values at specific node positions
// from a source buffer. Used to seed turn/river zone reach from the previous
// zone's reach at chance children.
//
// grid: count * np_nh elements
// ============================================================================

kernel void vcfr_seed_reach(
    device float* dst_reach               [[buffer(0)]],
    device const float* src_reach         [[buffer(1)]],
    device const uint32_t* chance_children [[buffer(2)]],
    constant int& count                   [[buffer(3)]],
    constant int& np_nh                   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int idx = int(gid);
    int total = count * np_nh;
    if (idx >= total) return;

    int child_idx = idx / np_nh;
    int offset_in_node = idx % np_nh;
    uint child_id = chance_children[child_idx];
    int offset = int(child_id) * np_nh + offset_in_node;
    dst_reach[offset] = src_reach[offset];
}

// Diagnostic kernel: run sweep with known inputs
kernel void debug_sweep(
    device float* output [[buffer(0)]],
    device const float* opp_reach_dev [[buffer(1)]],
    device const uint16_t* opp_str [[buffer(2)]],
    device const uint16_t* opp_idx [[buffer(3)]],
    device const uint16_t* pl_str [[buffer(4)]],
    device const uint16_t* pl_idx [[buffer(5)]],
    device const uint8_t* hand_cards [[buffer(6)]],
    constant int& nh_val [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid > 0) return;
    float opp_reach_local[1326];
    for (int h = 0; h < nh_val; h++) opp_reach_local[h] = opp_reach_dev[h];
    float result[1326];
    sorted_sweep_showdown_vcfr_local(
        opp_reach_local, 1, nh_val,
        opp_str, opp_idx,
        pl_str, pl_idx,
        hand_cards, result
    );
    for (int h = 0; h < nh_val; h++) {
        output[h] = result[h];
    }
}

// Debug: read back BatchedParams fields as floats
kernel void debug_params(
    device float* output [[buffer(0)]],
    constant BatchedParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid > 0) return;
    output[0] = float(params.level_count);
    output[1] = float(params.num_outcomes);
    output[2] = float(params.cfv_batch_stride);
    output[3] = float(params.sorted_opp_stride);
    output[4] = float(params.num_players);
    output[5] = float(params.nh);
    output[6] = float(params.traverser);
    output[7] = params.alpha_t;
    output[8] = params.beta_t;
    output[9] = params.gamma_t;
    output[10] = params.regret_floor;
    output[11] = float(params.starting_pot);
    output[12] = params.num_combinations;
    output[13] = float(params.regret_outcome_stride);
    output[14] = float(params.cum_outcome_stride);
}

// Debug: test N>2 equal-contribution product formula
// Given 2 opponent reach vectors and sorted arrays, compute the product-formula CFV
kernel void debug_multiway_sweep(
    device float* output [[buffer(0)]],
    device const float* opp0_reach [[buffer(1)]],
    device const float* opp1_reach [[buffer(2)]],
    device const uint16_t* opp_str [[buffer(3)]],
    device const uint16_t* opp_idx [[buffer(4)]],
    device const uint16_t* pl_str [[buffer(5)]],
    device const uint16_t* pl_idx [[buffer(6)]],
    device const uint8_t* hand_cards [[buffer(7)]],
    constant int& nh_val [[buffer(8)]],
    constant float& half_pot [[buffer(9)]],
    constant int& num_active_opp [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid > 0) return;
    int nh = nh_val;
    int num_opp = 2;
    
    float opp_reach_local[5 * 1326];
    for (int h = 0; h < nh; h++) {
        opp_reach_local[0 * nh + h] = opp0_reach[h];
        opp_reach_local[1 * nh + h] = opp1_reach[h];
    }
    
    // Compute cum_weaker and eff_total using the same code as the batched kernel
    float cum_weaker[5 * 1326];
    float eff_total[5 * 1326];
    for (int h = 0; h < nh; h++) { cum_weaker[h] = 0.0f; eff_total[h] = 0.0f; }
    
    for (int oi = 0; oi < num_opp; oi++) {
        thread const float* opp_r = opp_reach_local + oi * nh;
        float cw[1326];
        float cfreach_sum = 0.0f;
        float cfreach_minus[52];
        for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;
        int i = 0;

        for (int si = 0; si < nh; si++) {
            uint16_t str_h = pl_str[si];
            uint16_t h = pl_idx[si];
            while (i < nh && opp_str[oi * nh + i] < str_h) {
                uint16_t ho = opp_idx[oi * nh + i];
                float r = opp_r[ho];
                if (r != 0.0f) {
                    cfreach_sum += r;
                    cfreach_minus[hand_cards[ho * 2]] += r;
                    cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                }
                i++;
            }
            cw[h] = cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]];
        }

        while (i < nh) {
            uint16_t ho = opp_idx[oi * nh + i];
            float r = opp_r[ho];
            if (r != 0.0f) {
                cfreach_sum += r;
                cfreach_minus[hand_cards[ho * 2]] += r;
                cfreach_minus[hand_cards[ho * 2 + 1]] += r;
            }
            i++;
        }

        for (int h = 0; h < nh; h++) {
            float eff = cfreach_sum
                - cfreach_minus[hand_cards[h * 2]]
                - cfreach_minus[hand_cards[h * 2 + 1]]
                + opp_r[h];
            if (oi == 0) {
                cum_weaker[h] = cw[h];
                eff_total[h] = eff;
            } else {
                cum_weaker[h] *= cw[h];
                eff_total[h] *= eff;
            }
        }
    }
    
    for (int h = 0; h < nh; h++) {
        output[h] = half_pot * (float(num_active_opp + 1) * cum_weaker[h] - eff_total[h]);
    }
}

// ============================================================================
// Debug kernel: invoke multiway_brute_force_showdown with caller-provided
// inputs. Used by Rust tests to verify CPU/GPU parity of the helper directly,
// independent of the bottom-up tree traversal.
// ============================================================================
struct DebugBruteForceParams {
    int nh;
    int np;
    int traverser;
    int32_t starting_pot;
    uint16_t fold_mask;
    uint16_t _pad;          // align rake floats to 4-byte boundary
    // ─── Slice 2 rake forwarding for site (b) isolation unit test ───
    // Per user direction: "The debug-kernel rake-param forwarding is
    // as load-bearing as the kernel math, so both land together and
    // the DoD is the unit test reaching f32 floor (which confirms
    // both the helper applies rake and the test exercises it), not
    // merely changing."
    float rake_rate;
    float rake_cap;
    int32_t flop_seen;      // bool as i32 (Metal struct alignment)
};

kernel void debug_brute_force_showdown(
    device float* output                  [[buffer(0)]],   // [nh]
    device const float* opp_reach_in      [[buffer(1)]],   // [num_opp * nh]
    device const int32_t* contributions_in [[buffer(2)]],  // [np]
    device const uint8_t* hand_cards_dev  [[buffer(3)]],   // [nh*2]
    device const uint16_t* pl_str_dev     [[buffer(4)]],   // [nh]
    device const uint16_t* pl_idx_dev     [[buffer(5)]],   // [nh]
    constant DebugBruteForceParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid > 0) return;
    int nh = params.nh;
    int np = params.np;
    int num_opp = np - 1;

    // Copy inputs to thread-local
    float opp_reach_local[5 * 1326];
    for (int i = 0; i < num_opp * nh; i++) opp_reach_local[i] = opp_reach_in[i];

    int32_t contribs_local[8];
    for (int p = 0; p < np; p++) contribs_local[p] = contributions_in[p];

    float local_out[1326];
    multiway_brute_force_showdown(
        nh, np, params.traverser,
        params.starting_pot, params.fold_mask,
        contribs_local, opp_reach_local,
        hand_cards_dev, pl_str_dev, pl_idx_dev,
        params.rake_rate, params.rake_cap, params.flop_seen != 0,
        local_out
    );

    for (int h = 0; h < nh; h++) output[h] = local_out[h];
}

// ============================================================================
// K=5 brute-force showdown inner-loop MICROBENCHMARK
//
// Purpose: measure the achievable GPU throughput on the K=5 multiway
// showdown inner loop, decoupled from the full CFR solver. The 17-year
// projection in the bake-off rested on assuming the GPU stays at the
// 287 MFLOPS measured on the 3p K=2 path, which is overhead-dominated
// (tiny inner loop). The K=5 inner loop is two orders of magnitude
// more compute-dense per memory access, so it should run closer to peak;
// THIS BENCHMARK MEASURES THAT INSTEAD OF ASSUMING IT.
//
// Dispatch shape: total threads = (batches × nh). Each thread runs one
// h_player slot's full nh^K brute-force enumeration (50^5 ≈ 312M traversals
// per thread, of which ~217M are valid after card-conflict pruning).
//
// Card-mask uses ulong (64-bit) since 52 cards exceeds 32 bits.
// ============================================================================
kernel void k5_brute_force_microbench(
    device const float*    reach           [[buffer(0)]],   // [num_opp * nh]
    device const uchar*    hand_cards      [[buffer(1)]],   // [nh * 2]
    device float*          per_thread_out  [[buffer(2)]],   // [batches * nh]
    constant int&          nh              [[buffer(3)]],
    constant int&          batches         [[buffer(4)]],
    uint3                  tid             [[thread_position_in_grid]]
) {
    int batch_id = tid.x;
    int h = tid.y;

    if (batch_id >= batches || h >= nh) return;

    ulong h_m = (1ul << hand_cards[h * 2]) | (1ul << hand_cards[h * 2 + 1]);

    float accum = 0.0f;

    // K=5 brute-force enumeration: 5 nested loops over opponent hands,
    // each level rejecting conflicts via the running card mask.
    int num_opp = 5;
    int o0 = 0 * nh, o1 = 1 * nh, o2 = 2 * nh, o3 = 3 * nh, o4 = 4 * nh;

    for (int g0 = 0; g0 < nh; g0++) {
        ulong g0_m = (1ul << hand_cards[g0 * 2]) | (1ul << hand_cards[g0 * 2 + 1]);
        if ((g0_m & h_m) != 0) continue;
        float r0 = reach[o0 + g0];
        if (r0 == 0.0f) continue;
        ulong m01 = h_m | g0_m;

        for (int g1 = 0; g1 < nh; g1++) {
            ulong g1_m = (1ul << hand_cards[g1 * 2]) | (1ul << hand_cards[g1 * 2 + 1]);
            if ((g1_m & m01) != 0) continue;
            float r1 = reach[o1 + g1];
            if (r1 == 0.0f) continue;
            float p01 = r0 * r1;
            ulong m12 = m01 | g1_m;

            for (int g2 = 0; g2 < nh; g2++) {
                ulong g2_m = (1ul << hand_cards[g2 * 2]) | (1ul << hand_cards[g2 * 2 + 1]);
                if ((g2_m & m12) != 0) continue;
                float r2 = reach[o2 + g2];
                if (r2 == 0.0f) continue;
                float p012 = p01 * r2;
                ulong m23 = m12 | g2_m;

                for (int g3 = 0; g3 < nh; g3++) {
                    ulong g3_m = (1ul << hand_cards[g3 * 2]) | (1ul << hand_cards[g3 * 2 + 1]);
                    if ((g3_m & m23) != 0) continue;
                    float r3 = reach[o3 + g3];
                    if (r3 == 0.0f) continue;
                    float p0123 = p012 * r3;
                    ulong m34 = m23 | g3_m;

                    for (int g4 = 0; g4 < nh; g4++) {
                        ulong g4_m = (1ul << hand_cards[g4 * 2]) | (1ul << hand_cards[g4 * 2 + 1]);
                        if ((g4_m & m34) != 0) continue;
                        float r4 = reach[o4 + g4];
                        if (r4 == 0.0f) continue;
                        accum += p0123 * r4;  // FMA: net for K=5 brute force scenario
                    }
                }
            }
        }
    }

    // Use batch_id to vary the write slot so the GPU can't dead-code-eliminate.
    per_thread_out[batch_id * nh + h] = accum;
    (void)num_opp;
}

// ============================================================================
// K=5 factored CFV "share" inner-loop MICROBENCHMARK
//
// Implements the recursive K-1 expansion for K=5 share (the per-scenario
// "1/(1+tied)" weighted reach product, weighted by validity-and-strength).
// Recursion: 3 outer loops over g_0, g_1, g_2 with B/T category gating
// (skip g where s > h_str), then K=2 PAIR-decomposition base case for
// (g_3, g_4) using extended-mask masses recomputed on the fly.
//
// Per-thread output is the share (= the part of CFV inside `pot · share`,
// pre-traverser-stake subtraction). Validated against CPU factored_share
// at f32 precision for correctness, then dispatched at production nh=50
// scale for throughput measurement against the 86 GFLOPS brute-force
// baseline.
//
// Thread-local arrays (4 × 52 × float = 832 B per thread) — Metal will
// place these in fast on-chip memory; per-batch grid sizing chosen to
// keep total occupancy within budget.
// ============================================================================
kernel void k5_factored_share_microbench(
    device const float*    reach           [[buffer(0)]],   // [5 * nh]
    device const uchar*    hand_cards      [[buffer(1)]],   // [nh * 2]
    device const ushort*   hand_strength   [[buffer(2)]],   // [nh]
    device float*          share_out       [[buffer(3)]],   // [batches * nh]
    constant int&          nh              [[buffer(4)]],
    constant int&          batches         [[buffer(5)]],
    uint3                  tid             [[thread_position_in_grid]]
) {
    int batch_id = tid.x;
    int h = tid.y;
    if (batch_id >= batches || h >= nh) return;

    ulong h_m = (1ul << hand_cards[h * 2]) | (1ul << hand_cards[h * 2 + 1]);
    ushort h_str = hand_strength[h];

    int o0 = 0 * nh, o1 = 1 * nh, o2 = 2 * nh, o3 = 3 * nh, o4 = 4 * nh;

    float share = 0.0f;

    // Outer K-3 levels: enumerate g_0, g_1, g_2.
    for (int g0 = 0; g0 < nh; g0++) {
        ulong g0_m = (1ul << hand_cards[g0 * 2]) | (1ul << hand_cards[g0 * 2 + 1]);
        if ((g0_m & h_m) != 0) continue;
        float r0 = reach[o0 + g0];
        if (r0 == 0.0f) continue;
        ushort s0 = hand_strength[g0];
        if (s0 > h_str) continue;
        int t0 = (s0 == h_str) ? 1 : 0;
        ulong m1 = h_m | g0_m;

        for (int g1 = 0; g1 < nh; g1++) {
            ulong g1_m = (1ul << hand_cards[g1 * 2]) | (1ul << hand_cards[g1 * 2 + 1]);
            if ((g1_m & m1) != 0) continue;
            float r1 = reach[o1 + g1];
            if (r1 == 0.0f) continue;
            ushort s1 = hand_strength[g1];
            if (s1 > h_str) continue;
            int t01 = t0 + ((s1 == h_str) ? 1 : 0);
            float p01 = r0 * r1;
            ulong m2 = m1 | g1_m;

            for (int g2 = 0; g2 < nh; g2++) {
                ulong g2_m = (1ul << hand_cards[g2 * 2]) | (1ul << hand_cards[g2 * 2 + 1]);
                if ((g2_m & m2) != 0) continue;
                float r2 = reach[o2 + g2];
                if (r2 == 0.0f) continue;
                ushort s2 = hand_strength[g2];
                if (s2 > h_str) continue;
                int t012 = t01 + ((s2 == h_str) ? 1 : 0);
                float p012 = p01 * r2;
                ulong m3 = m2 | g2_m;

                // K=2 PAIR base case for (opp 3, opp 4) at extended mask m3.
                // Recompute B/T totals + per-card + same-hand at extended mask.
                float ba = 0.0f, ta = 0.0f, bb_tot = 0.0f, tb_tot = 0.0f;
                float ba_pc[52]; float ta_pc[52];
                float bb_pc[52]; float tb_pc[52];
                for (int c = 0; c < 52; c++) {
                    ba_pc[c] = 0.0f; ta_pc[c] = 0.0f;
                    bb_pc[c] = 0.0f; tb_pc[c] = 0.0f;
                }
                float h_bb = 0.0f, h_tt = 0.0f;

                for (int g = 0; g < nh; g++) {
                    ulong g_m = (1ul << hand_cards[g * 2]) | (1ul << hand_cards[g * 2 + 1]);
                    if ((g_m & m3) != 0) continue;
                    float ra = reach[o3 + g];
                    float rb = reach[o4 + g];
                    if (ra == 0.0f && rb == 0.0f) continue;
                    ushort s = hand_strength[g];
                    int gc1 = hand_cards[g * 2];
                    int gc2 = hand_cards[g * 2 + 1];
                    if (s < h_str) {
                        ba += ra; ba_pc[gc1] += ra; ba_pc[gc2] += ra;
                        bb_tot += rb; bb_pc[gc1] += rb; bb_pc[gc2] += rb;
                        h_bb += ra * rb;
                    } else if (s == h_str) {
                        ta += ra; ta_pc[gc1] += ra; ta_pc[gc2] += ra;
                        tb_tot += rb; tb_pc[gc1] += rb; tb_pc[gc2] += rb;
                        h_tt += ra * rb;
                    }
                }

                // Edge sums over c ∉ extended dead mask.
                float bb_edge = 0.0f, bt_edge = 0.0f, tb_edge = 0.0f, tt_edge = 0.0f;
                for (int c = 0; c < 52; c++) {
                    if ((m3 & (1ul << c)) != 0) continue;
                    bb_edge += ba_pc[c] * bb_pc[c];
                    bt_edge += ba_pc[c] * tb_pc[c];
                    tb_edge += ta_pc[c] * bb_pc[c];
                    tt_edge += ta_pc[c] * tb_pc[c];
                }

                // PAIR with same-hand corrections only for BB and TT.
                float pair_bb = ba * bb_tot - bb_edge + h_bb;
                float pair_bt = ba * tb_tot - bt_edge;
                float pair_tb = ta * bb_tot - tb_edge;
                float pair_tt = ta * tb_tot - tt_edge + h_tt;

                float tf = (float)t012;
                float pair_share =
                    pair_bb / (1.0f + tf) +
                    pair_bt / (2.0f + tf) +
                    pair_tb / (2.0f + tf) +
                    pair_tt / (3.0f + tf);

                share += p012 * pair_share;
            }
        }
    }

    share_out[batch_id * nh + h] = share;
}

// ============================================================================
// K=3 per-level factored share kernel — for one Case C level with given
// eligibility bitmask. Mirrors CPU `factored_share_at_level` at K=3 with
// the K=2 PAIR-decomposition base case handling all four eligibility
// sub-cases (E,E), (E,I), (I,E), (I,I).
//
// Eligibility: bit `oi` in `elig_opps` set → opp at slot `oi` is eligible
// at this level (use B/T strength filter, skip S). Cleared → ineligible
// (any strength, contribute to reach product only).
// ============================================================================
kernel void k3_per_level_share_microbench(
    device const float*    reach          [[buffer(0)]],   // [3 * nh]
    device const uchar*    hand_cards     [[buffer(1)]],   // [nh * 2]
    device const ushort*   hand_strength  [[buffer(2)]],   // [nh]
    device float*          share_out      [[buffer(3)]],   // [batches * nh]
    constant int&          nh             [[buffer(4)]],
    constant int&          batches        [[buffer(5)]],
    constant uint&         elig_opps      [[buffer(6)]],   // 3 bits
    constant uint&         tied_offset_in [[buffer(7)]],   // 0 at recursion start
    uint3                  tid            [[thread_position_in_grid]]
) {
    int batch_id = tid.x;
    int h = tid.y;
    if (batch_id >= batches || h >= nh) return;

    ulong h_m = (1ul << hand_cards[h * 2]) | (1ul << hand_cards[h * 2 + 1]);
    ushort h_str = hand_strength[h];

    bool e0 = (elig_opps >> 0) & 1u;
    bool e1 = (elig_opps >> 1) & 1u;
    bool e2 = (elig_opps >> 2) & 1u;

    int o0 = 0 * nh, o1 = 1 * nh, o2 = 2 * nh;
    float share = 0.0f;

    // Outer loop over g_0.
    for (int g0 = 0; g0 < nh; g0++) {
        ulong g0_m = (1ul << hand_cards[g0 * 2]) | (1ul << hand_cards[g0 * 2 + 1]);
        if ((g0_m & h_m) != 0) continue;
        float r0 = reach[o0 + g0];
        if (r0 == 0.0f) continue;
        ushort s0 = hand_strength[g0];

        uint t0;
        if (e0) {
            if (s0 > h_str) continue;
            t0 = (s0 == h_str) ? (tied_offset_in + 1) : tied_offset_in;
        } else {
            t0 = tied_offset_in;
        }

        ulong m1 = h_m | g0_m;

        // K=2 base case: compute B/T/S masses for opps 1, 2 at extended mask m1.
        float b1 = 0.0f, t1 = 0.0f, ss1 = 0.0f;
        float b2 = 0.0f, t2 = 0.0f, ss2 = 0.0f;
        float b1_pc[52], t1_pc[52], s1_pc[52];
        float b2_pc[52], t2_pc[52], s2_pc[52];
        for (int c = 0; c < 52; c++) {
            b1_pc[c] = 0.0f; t1_pc[c] = 0.0f; s1_pc[c] = 0.0f;
            b2_pc[c] = 0.0f; t2_pc[c] = 0.0f; s2_pc[c] = 0.0f;
        }
        float h_bb = 0.0f, h_tt = 0.0f, h_ss = 0.0f;

        for (int g = 0; g < nh; g++) {
            ulong g_m = (1ul << hand_cards[g * 2]) | (1ul << hand_cards[g * 2 + 1]);
            if ((g_m & m1) != 0) continue;
            float r1 = reach[o1 + g];
            float r2 = reach[o2 + g];
            if (r1 == 0.0f && r2 == 0.0f) continue;
            ushort s = hand_strength[g];
            int gc1 = hand_cards[g * 2];
            int gc2 = hand_cards[g * 2 + 1];
            if (s < h_str) {
                b1 += r1; b1_pc[gc1] += r1; b1_pc[gc2] += r1;
                b2 += r2; b2_pc[gc1] += r2; b2_pc[gc2] += r2;
                h_bb += r1 * r2;
            } else if (s == h_str) {
                t1 += r1; t1_pc[gc1] += r1; t1_pc[gc2] += r1;
                t2 += r2; t2_pc[gc1] += r2; t2_pc[gc2] += r2;
                h_tt += r1 * r2;
            } else {
                ss1 += r1; s1_pc[gc1] += r1; s1_pc[gc2] += r1;
                ss2 += r2; s2_pc[gc1] += r2; s2_pc[gc2] += r2;
                h_ss += r1 * r2;
            }
        }
        float r1_tot = b1 + t1 + ss1;
        float r2_tot = b2 + t2 + ss2;
        float h_tot = h_bb + h_tt + h_ss;

        // Edge sums over c ∉ extended mask.
        float ebb = 0.0f, ebt = 0.0f, etb = 0.0f, ett = 0.0f;
        float ebe = 0.0f, ete = 0.0f, eeb = 0.0f, eet = 0.0f, eee = 0.0f;
        for (int c = 0; c < 52; c++) {
            if ((m1 & (1ul << c)) != 0) continue;
            float b1c = b1_pc[c], t1c = t1_pc[c], s1c = s1_pc[c];
            float b2c = b2_pc[c], t2c = t2_pc[c], s2c = s2_pc[c];
            float r1c = b1c + t1c + s1c;
            float r2c = b2c + t2c + s2c;
            ebb += b1c * b2c;
            ebt += b1c * t2c;
            etb += t1c * b2c;
            ett += t1c * t2c;
            ebe += b1c * r2c;
            ete += t1c * r2c;
            eeb += r1c * b2c;
            eet += r1c * t2c;
            eee += r1c * r2c;
        }

        // PAIR formulas with eligibility-determined same-hand corrections.
        float pair_bb = b1 * b2 - ebb + h_bb;
        float pair_bt = b1 * t2 - ebt;
        float pair_tb = t1 * b2 - etb;
        float pair_tt = t1 * t2 - ett + h_tt;
        float pair_be = b1 * r2_tot - ebe + h_bb;
        float pair_te = t1 * r2_tot - ete + h_tt;
        float pair_eb = r1_tot * b2 - eeb + h_bb;
        float pair_et = r1_tot * t2 - eet + h_tt;
        float pair_ee = r1_tot * r2_tot - eee + h_tot;

        float tf = (float)t0;
        float inner;
        if (e1 && e2) {
            inner = pair_bb / (1.0f + tf)
                  + pair_bt / (2.0f + tf)
                  + pair_tb / (2.0f + tf)
                  + pair_tt / (3.0f + tf);
        } else if (e1 && !e2) {
            inner = pair_be / (1.0f + tf) + pair_te / (2.0f + tf);
        } else if (!e1 && e2) {
            inner = pair_eb / (1.0f + tf) + pair_et / (2.0f + tf);
        } else {
            inner = pair_ee / (1.0f + tf);
        }

        share += r0 * inner;
    }

    share_out[batch_id * nh + h] = share;
}

// ============================================================================
// K=5 per-level factored share kernel — extends the K=3 version with 2
// more outer loops (g_1, g_2) and per-opp eligibility flags from bits 0..4.
// Same K=2 base case as K=3 for opps 3, 4.
//
// elig_opps bitmask: bit i set ↔ opp at slot i is eligible at this level.
// ============================================================================
kernel void k5_per_level_share_microbench(
    device const float*    reach          [[buffer(0)]],   // [5 * nh]
    device const uchar*    hand_cards     [[buffer(1)]],
    device const ushort*   hand_strength  [[buffer(2)]],
    device float*          share_out      [[buffer(3)]],
    constant int&          nh             [[buffer(4)]],
    constant int&          batches        [[buffer(5)]],
    constant uint&         elig_opps      [[buffer(6)]],   // bits 0..4
    constant uint&         tied_offset_in [[buffer(7)]],
    uint3                  tid            [[thread_position_in_grid]]
) {
    int batch_id = tid.x;
    int h = tid.y;
    if (batch_id >= batches || h >= nh) return;

    ulong h_m = (1ul << hand_cards[h * 2]) | (1ul << hand_cards[h * 2 + 1]);
    ushort h_str = hand_strength[h];

    bool e0 = (elig_opps >> 0) & 1u;
    bool e1 = (elig_opps >> 1) & 1u;
    bool e2 = (elig_opps >> 2) & 1u;
    bool e3 = (elig_opps >> 3) & 1u;
    bool e4 = (elig_opps >> 4) & 1u;

    int o0 = 0 * nh, o1 = 1 * nh, o2 = 2 * nh, o3 = 3 * nh, o4 = 4 * nh;
    float share = 0.0f;

    for (int g0 = 0; g0 < nh; g0++) {
        ulong g0_m = (1ul << hand_cards[g0 * 2]) | (1ul << hand_cards[g0 * 2 + 1]);
        if ((g0_m & h_m) != 0) continue;
        float r0 = reach[o0 + g0];
        if (r0 == 0.0f) continue;
        ushort s0 = hand_strength[g0];
        uint t0;
        if (e0) {
            if (s0 > h_str) continue;
            t0 = (s0 == h_str) ? (tied_offset_in + 1) : tied_offset_in;
        } else {
            t0 = tied_offset_in;
        }
        ulong m1 = h_m | g0_m;

        for (int g1 = 0; g1 < nh; g1++) {
            ulong g1_m = (1ul << hand_cards[g1 * 2]) | (1ul << hand_cards[g1 * 2 + 1]);
            if ((g1_m & m1) != 0) continue;
            float r1 = reach[o1 + g1];
            if (r1 == 0.0f) continue;
            ushort s1 = hand_strength[g1];
            uint t01;
            if (e1) {
                if (s1 > h_str) continue;
                t01 = (s1 == h_str) ? (t0 + 1) : t0;
            } else {
                t01 = t0;
            }
            float p01 = r0 * r1;
            ulong m2 = m1 | g1_m;

            for (int g2 = 0; g2 < nh; g2++) {
                ulong g2_m = (1ul << hand_cards[g2 * 2]) | (1ul << hand_cards[g2 * 2 + 1]);
                if ((g2_m & m2) != 0) continue;
                float r2 = reach[o2 + g2];
                if (r2 == 0.0f) continue;
                ushort s2 = hand_strength[g2];
                uint t012;
                if (e2) {
                    if (s2 > h_str) continue;
                    t012 = (s2 == h_str) ? (t01 + 1) : t01;
                } else {
                    t012 = t01;
                }
                float p012 = p01 * r2;
                ulong m3 = m2 | g2_m;

                // K=2 base case for opps 3, 4 at extended mask m3.
                float b1 = 0.0f, t1 = 0.0f, ss1 = 0.0f;
                float b2 = 0.0f, t2 = 0.0f, ss2 = 0.0f;
                float b1_pc[52], t1_pc[52], s1_pc[52];
                float b2_pc[52], t2_pc[52], s2_pc[52];
                for (int c = 0; c < 52; c++) {
                    b1_pc[c]=0.0f; t1_pc[c]=0.0f; s1_pc[c]=0.0f;
                    b2_pc[c]=0.0f; t2_pc[c]=0.0f; s2_pc[c]=0.0f;
                }
                float h_bb=0.0f, h_tt=0.0f, h_ss=0.0f;

                for (int g = 0; g < nh; g++) {
                    ulong g_m = (1ul << hand_cards[g * 2]) | (1ul << hand_cards[g * 2 + 1]);
                    if ((g_m & m3) != 0) continue;
                    float rA = reach[o3 + g];
                    float rB = reach[o4 + g];
                    if (rA == 0.0f && rB == 0.0f) continue;
                    ushort s = hand_strength[g];
                    int gc1 = hand_cards[g * 2];
                    int gc2 = hand_cards[g * 2 + 1];
                    if (s < h_str) {
                        b1 += rA; b1_pc[gc1] += rA; b1_pc[gc2] += rA;
                        b2 += rB; b2_pc[gc1] += rB; b2_pc[gc2] += rB;
                        h_bb += rA * rB;
                    } else if (s == h_str) {
                        t1 += rA; t1_pc[gc1] += rA; t1_pc[gc2] += rA;
                        t2 += rB; t2_pc[gc1] += rB; t2_pc[gc2] += rB;
                        h_tt += rA * rB;
                    } else {
                        ss1 += rA; s1_pc[gc1] += rA; s1_pc[gc2] += rA;
                        ss2 += rB; s2_pc[gc1] += rB; s2_pc[gc2] += rB;
                        h_ss += rA * rB;
                    }
                }
                float r1_tot = b1 + t1 + ss1;
                float r2_tot = b2 + t2 + ss2;
                float h_tot = h_bb + h_tt + h_ss;

                float ebb=0.0f, ebt=0.0f, etb=0.0f, ett=0.0f;
                float ebe=0.0f, ete=0.0f, eeb=0.0f, eet=0.0f, eee=0.0f;
                for (int c = 0; c < 52; c++) {
                    if ((m3 & (1ul << c)) != 0) continue;
                    float b1c=b1_pc[c], t1c=t1_pc[c], s1c=s1_pc[c];
                    float b2c=b2_pc[c], t2c=t2_pc[c], s2c=s2_pc[c];
                    float r1c = b1c + t1c + s1c;
                    float r2c = b2c + t2c + s2c;
                    ebb += b1c * b2c;
                    ebt += b1c * t2c;
                    etb += t1c * b2c;
                    ett += t1c * t2c;
                    ebe += b1c * r2c;
                    ete += t1c * r2c;
                    eeb += r1c * b2c;
                    eet += r1c * t2c;
                    eee += r1c * r2c;
                }

                float pair_bb = b1 * b2 - ebb + h_bb;
                float pair_bt = b1 * t2 - ebt;
                float pair_tb = t1 * b2 - etb;
                float pair_tt = t1 * t2 - ett + h_tt;
                float pair_be = b1 * r2_tot - ebe + h_bb;
                float pair_te = t1 * r2_tot - ete + h_tt;
                float pair_eb = r1_tot * b2 - eeb + h_bb;
                float pair_et = r1_tot * t2 - eet + h_tt;
                float pair_ee = r1_tot * r2_tot - eee + h_tot;

                float tf = (float)t012;
                float inner;
                if (e3 && e4) {
                    inner = pair_bb / (1.0f + tf)
                          + pair_bt / (2.0f + tf)
                          + pair_tb / (2.0f + tf)
                          + pair_tt / (3.0f + tf);
                } else if (e3 && !e4) {
                    inner = pair_be / (1.0f + tf) + pair_te / (2.0f + tf);
                } else if (!e3 && e4) {
                    inner = pair_eb / (1.0f + tf) + pair_et / (2.0f + tf);
                } else {
                    inner = pair_ee / (1.0f + tf);
                }
                share += p012 * inner;
            }
        }
    }

    share_out[batch_id * nh + h] = share;
}

// ============================================================================
// UNIFIED N-player factored showdown kernel (N = 2 to 6).
//
// One generic Metal kernel for any opponent count K = num_opp ∈ {1, 2, 3, 4, 5},
// replacing the fragmented per-K kernels. Internal structure:
//
//   1. Walk levels of the terminal. For each level:
//      - Case A (no eligible active, traverser eligible OR contributed):
//        add static cash (the slice returned).
//      - Case D (active opps eligible, traverser ineligible): skip.
//      - Case C (active eligible AND traverser eligible): compute factored
//        share via the per-level recursive K-1 expansion with eligibility-
//        determined strength filtering.
//   2. Compose: cfv = (static_cash − stake) × TVRP + Σ_C pot_l × share_l
//      where TVRP = factored share with elig_opps = 0.
//
// Recursion: up to 3 outer loops (max K = 5 → outer over g_0, g_1, g_2) +
// K=2 PAIR-decomposition base case for opps (K-2, K-1). For K = 1, use
// K=1 base directly. For K = 2, use K=2 base directly.
//
// Eligibility-aware: bit i of elig_opps = 1 → opp at slot i eligible at
// this level (B/T strength filter, skip S). Bit cleared → ineligible
// (any strength, contribute to reach product only).
//
// Numerical discipline: f32 throughout for now. f64 inner combiner is
// out-of-scope for the first cut. Same operation order as CPU factored:
// inner accumulators ordered (b_a, t_a, s_a) and edge sums computed in
// (bb, bt, tb, tt, be, te, eb, et, ee) order. CPU and GPU must walk in
// the same order so accumulator round-off matches; pinned here:
//   - Outer loop order: g_0, g_1, g_2 ascending (matches CPU `opp_indices`)
//   - K=2 base: scan hands ascending, compute pc edge sums (c ascending),
//     then PAIR formulas in (BB, BT, TB, TT, BE, TE, EB, ET, EE) order,
//     and share dispatch by (ea, eb).
// ============================================================================

struct LevelInfoMetal {
    float pot_l;
    uint  elig_opps;
    uint  trav_elig;
    float trav_contrib_at_lev;
};

// K=1 base case: enumerate one opponent.
inline float factored_share_k1(
    int oi,
    bool elig,
    ulong dead_mask,
    ushort h_str,
    uint tied_offset,
    device const float* reach,
    device const uchar* hand_cards,
    device const ushort* hand_strength,
    int nh
) {
    float share = 0.0f;
    int o_off = oi * nh;
    for (int g = 0; g < nh; g++) {
        ulong g_m = (1ul << hand_cards[g * 2]) | (1ul << hand_cards[g * 2 + 1]);
        if ((g_m & dead_mask) != 0) continue;
        float r = reach[o_off + g];
        if (r == 0.0f) continue;
        ushort s = hand_strength[g];
        if (elig) {
            if (s > h_str) continue;
            uint t = (s == h_str) ? (tied_offset + 1) : tied_offset;
            share += r / (1.0f + (float)t);
        } else {
            share += r / (1.0f + (float)tied_offset);
        }
    }
    return share;
}

// K=2 base case: 4 PAIR formulas dispatched by (ea, eb) eligibility.
inline float factored_share_k2(
    int oa, int ob,
    bool ea, bool eb,
    ulong dead_mask,
    ushort h_str,
    uint tied_offset,
    device const float* reach,
    device const uchar* hand_cards,
    device const ushort* hand_strength,
    int nh
) {
    int oa_off = oa * nh;
    int ob_off = ob * nh;

    float b_a = 0.0f, t_a = 0.0f, s_a = 0.0f;
    float b_b = 0.0f, t_b = 0.0f, s_b = 0.0f;
    float b_a_pc[52], t_a_pc[52], s_a_pc[52];
    float b_b_pc[52], t_b_pc[52], s_b_pc[52];
    for (int c = 0; c < 52; c++) {
        b_a_pc[c] = 0.0f; t_a_pc[c] = 0.0f; s_a_pc[c] = 0.0f;
        b_b_pc[c] = 0.0f; t_b_pc[c] = 0.0f; s_b_pc[c] = 0.0f;
    }
    float h_bb = 0.0f, h_tt = 0.0f, h_ss = 0.0f;

    for (int g = 0; g < nh; g++) {
        ulong g_m = (1ul << hand_cards[g * 2]) | (1ul << hand_cards[g * 2 + 1]);
        if ((g_m & dead_mask) != 0) continue;
        float r_a = reach[oa_off + g];
        float r_b = reach[ob_off + g];
        if (r_a == 0.0f && r_b == 0.0f) continue;
        ushort s = hand_strength[g];
        int gc1 = hand_cards[g * 2];
        int gc2 = hand_cards[g * 2 + 1];
        if (s < h_str) {
            b_a += r_a; b_a_pc[gc1] += r_a; b_a_pc[gc2] += r_a;
            b_b += r_b; b_b_pc[gc1] += r_b; b_b_pc[gc2] += r_b;
            h_bb += r_a * r_b;
        } else if (s == h_str) {
            t_a += r_a; t_a_pc[gc1] += r_a; t_a_pc[gc2] += r_a;
            t_b += r_b; t_b_pc[gc1] += r_b; t_b_pc[gc2] += r_b;
            h_tt += r_a * r_b;
        } else {
            s_a += r_a; s_a_pc[gc1] += r_a; s_a_pc[gc2] += r_a;
            s_b += r_b; s_b_pc[gc1] += r_b; s_b_pc[gc2] += r_b;
            h_ss += r_a * r_b;
        }
    }
    float r_a_tot = b_a + t_a + s_a;
    float r_b_tot = b_b + t_b + s_b;
    float h_tot = h_bb + h_tt + h_ss;

    float edge_bb = 0.0f, edge_bt = 0.0f, edge_tb = 0.0f, edge_tt = 0.0f;
    float edge_be = 0.0f, edge_te = 0.0f, edge_eb = 0.0f, edge_et = 0.0f;
    float edge_ee = 0.0f;
    for (int c = 0; c < 52; c++) {
        if ((dead_mask & (1ul << c)) != 0) continue;
        float bac = b_a_pc[c], tac = t_a_pc[c], sac = s_a_pc[c];
        float bbc = b_b_pc[c], tbc = t_b_pc[c], sbc = s_b_pc[c];
        float rac = bac + tac + sac;
        float rbc = bbc + tbc + sbc;
        edge_bb += bac * bbc;
        edge_bt += bac * tbc;
        edge_tb += tac * bbc;
        edge_tt += tac * tbc;
        edge_be += bac * rbc;
        edge_te += tac * rbc;
        edge_eb += rac * bbc;
        edge_et += rac * tbc;
        edge_ee += rac * rbc;
    }

    float pair_bb = b_a * b_b - edge_bb + h_bb;
    float pair_bt = b_a * t_b - edge_bt;
    float pair_tb = t_a * b_b - edge_tb;
    float pair_tt = t_a * t_b - edge_tt + h_tt;
    float pair_be = b_a * r_b_tot - edge_be + h_bb;
    float pair_te = t_a * r_b_tot - edge_te + h_tt;
    float pair_eb = r_a_tot * b_b - edge_eb + h_bb;
    float pair_et = r_a_tot * t_b - edge_et + h_tt;
    float pair_ee = r_a_tot * r_b_tot - edge_ee + h_tot;

    float tf = (float)tied_offset;
    if (ea && eb) {
        return pair_bb / (1.0f + tf)
             + pair_bt / (2.0f + tf)
             + pair_tb / (2.0f + tf)
             + pair_tt / (3.0f + tf);
    } else if (ea && !eb) {
        return pair_be / (1.0f + tf) + pair_te / (2.0f + tf);
    } else if (!ea && eb) {
        return pair_eb / (1.0f + tf) + pair_et / (2.0f + tf);
    } else {
        return pair_ee / (1.0f + tf);
    }
}

// Factored share for one level with given eligibility mask & tied_offset.
// Generic over K=1..5 via outer-loop depth determined by num_opp.
inline float factored_share_for_level(
    int num_opp,
    uint elig_opps,
    uint tied_offset,
    ulong h_m,
    ushort h_str,
    device const float* reach,
    device const uchar* hand_cards,
    device const ushort* hand_strength,
    int nh
) {
    if (num_opp == 1) {
        bool e0 = (elig_opps >> 0) & 1u;
        return factored_share_k1(0, e0, h_m, h_str, tied_offset,
                                  reach, hand_cards, hand_strength, nh);
    }
    if (num_opp == 2) {
        bool e0 = (elig_opps >> 0) & 1u;
        bool e1 = (elig_opps >> 1) & 1u;
        return factored_share_k2(0, 1, e0, e1, h_m, h_str, tied_offset,
                                  reach, hand_cards, hand_strength, nh);
    }

    // num_opp >= 3: outer loops to K=2 base.
    int k2a = num_opp - 2;
    int k2b = num_opp - 1;
    bool ek2a = (elig_opps >> k2a) & 1u;
    bool ek2b = (elig_opps >> k2b) & 1u;
    bool e0 = (elig_opps >> 0) & 1u;

    float share = 0.0f;

    for (int g0 = 0; g0 < nh; g0++) {
        ulong g0_m = (1ul << hand_cards[g0 * 2]) | (1ul << hand_cards[g0 * 2 + 1]);
        if ((g0_m & h_m) != 0) continue;
        float r0 = reach[0 * nh + g0];
        if (r0 == 0.0f) continue;
        ushort s0 = hand_strength[g0];
        uint t0;
        if (e0) {
            if (s0 > h_str) continue;
            t0 = (s0 == h_str) ? (tied_offset + 1) : tied_offset;
        } else {
            t0 = tied_offset;
        }
        ulong m1 = h_m | g0_m;

        if (num_opp == 3) {
            share += r0 * factored_share_k2(k2a, k2b, ek2a, ek2b,
                                              m1, h_str, t0,
                                              reach, hand_cards, hand_strength, nh);
            continue;
        }

        bool e1 = (elig_opps >> 1) & 1u;
        for (int g1 = 0; g1 < nh; g1++) {
            ulong g1_m = (1ul << hand_cards[g1 * 2]) | (1ul << hand_cards[g1 * 2 + 1]);
            if ((g1_m & m1) != 0) continue;
            float r1 = reach[1 * nh + g1];
            if (r1 == 0.0f) continue;
            ushort s1 = hand_strength[g1];
            uint t01;
            if (e1) {
                if (s1 > h_str) continue;
                t01 = (s1 == h_str) ? (t0 + 1) : t0;
            } else {
                t01 = t0;
            }
            ulong m2 = m1 | g1_m;
            float p01 = r0 * r1;

            if (num_opp == 4) {
                share += p01 * factored_share_k2(k2a, k2b, ek2a, ek2b,
                                                  m2, h_str, t01,
                                                  reach, hand_cards, hand_strength, nh);
                continue;
            }

            // num_opp == 5
            bool e2 = (elig_opps >> 2) & 1u;
            for (int g2 = 0; g2 < nh; g2++) {
                ulong g2_m = (1ul << hand_cards[g2 * 2]) | (1ul << hand_cards[g2 * 2 + 1]);
                if ((g2_m & m2) != 0) continue;
                float r2 = reach[2 * nh + g2];
                if (r2 == 0.0f) continue;
                ushort s2 = hand_strength[g2];
                uint t012;
                if (e2) {
                    if (s2 > h_str) continue;
                    t012 = (s2 == h_str) ? (t01 + 1) : t01;
                } else {
                    t012 = t01;
                }
                ulong m3 = m2 | g2_m;
                float p012 = p01 * r2;

                share += p012 * factored_share_k2(k2a, k2b, ek2a, ek2b,
                                                   m3, h_str, t012,
                                                   reach, hand_cards, hand_strength, nh);
            }
        }
    }

    return share;
}

// The unified kernel.
kernel void factored_showdown_unified(
    device const float*           reach           [[buffer(0)]],
    device const uchar*           hand_cards      [[buffer(1)]],
    device const ushort*          hand_strength   [[buffer(2)]],
    constant LevelInfoMetal*      levels          [[buffer(3)]],
    constant int&                 num_levels      [[buffer(4)]],
    constant int&                 num_opp         [[buffer(5)]],
    constant float&               traverser_stake [[buffer(6)]],
    device float*                 cfv_out         [[buffer(7)]],
    constant int&                 nh              [[buffer(8)]],
    constant int&                 batches         [[buffer(9)]],
    uint3                         tid             [[thread_position_in_grid]]
) {
    int batch_id = tid.x;
    int h = tid.y;
    if (batch_id >= batches || h >= nh) return;

    ulong h_m = (1ul << hand_cards[h * 2]) | (1ul << hand_cards[h * 2 + 1]);
    ushort h_str = hand_strength[h];

    // 1. TVRP(h) = factored share with all-ineligible (elig_opps = 0, tied = 0).
    float tvrp = factored_share_for_level(
        num_opp, 0u, 0u, h_m, h_str,
        reach, hand_cards, hand_strength, nh);

    // 2. Walk levels, accumulate static cash + Case C shares.
    float static_cash = 0.0f;
    float case_c = 0.0f;
    for (int li = 0; li < num_levels; li++) {
        LevelInfoMetal lev = levels[li];
        bool has_active_elig = (lev.elig_opps != 0);
        bool trav_elig = (lev.trav_elig != 0);
        if (!has_active_elig && trav_elig) {
            static_cash += lev.pot_l;
        } else if (!has_active_elig && !trav_elig) {
            if (lev.trav_contrib_at_lev > 0.0f) {
                static_cash += lev.trav_contrib_at_lev;
            }
        } else if (!trav_elig) {
            // Case D: skip
        } else {
            float share = factored_share_for_level(
                num_opp, lev.elig_opps, 0u, h_m, h_str,
                reach, hand_cards, hand_strength, nh);
            case_c += lev.pot_l * share;
        }
    }

    // PRODUCTION-DEAD (confirmed by disturbance test 2026-06-04):
    // No production create_pipeline binding; NaN write at this output
    // was invisible to ALL production tests (gpu_rake_parity_gate,
    // three_max_parity, three_max_reach, six_player_iter0_parity,
    // flop_start_cpu_test, iter_divergence — all PASS with NaN write).
    //
    // Test-only kernel — used by:
    //   - unified_kernel_gates.rs (K≥3 factored math validation)
    //   - precision_attribution_check.rs (f32 precision analysis)
    //
    // SEPARATE FINDING (recorded for follow-up): the disturbance test
    // ALSO revealed that unified_kernel_gates and precision_attribution_
    // check have WEAK ASSERTIONS that don't catch NaN — those tests print
    // `gpu=NaN diff=NaN` but still report test result OK. This is the
    // seventh false-green pattern surfaced in the rake arc (test that
    // uses a kernel but doesn't validate its CFV values). Not blocking
    // Phase B, but the test infrastructure should be hardened to detect
    // NaN as failures in a separate cleanup.
    cfv_out[batch_id * nh + h] = (static_cash - traverser_stake) * tvrp + case_c;
}

// ============================================================================
// THREAD-LOCAL variants of the factored helpers, for use inside
// `multiway_brute_force_showdown` which keeps opp_reach in thread-local
// memory for cache locality. Same logic as the device-pointer versions
// above, just with `thread const float*` for reach (and thread ushort*
// for hand_strength which is reconstructed locally per-thread).
// ============================================================================

inline float factored_share_k1_thread(
    int oi,
    bool elig,
    ulong dead_mask,
    ushort h_str,
    uint tied_offset,
    thread const float* reach,
    const device uchar* hand_cards,
    thread const ushort* hand_strength,
    int nh
) {
    float share = 0.0f;
    int o_off = oi * nh;
    for (int g = 0; g < nh; g++) {
        ulong g_m = (1ul << hand_cards[g * 2]) | (1ul << hand_cards[g * 2 + 1]);
        if ((g_m & dead_mask) != 0) continue;
        float r = reach[o_off + g];
        if (r == 0.0f) continue;
        ushort s = hand_strength[g];
        if (elig) {
            if (s > h_str) continue;
            uint t = (s == h_str) ? (tied_offset + 1) : tied_offset;
            share += r / (1.0f + (float)t);
        } else {
            share += r / (1.0f + (float)tied_offset);
        }
    }
    return share;
}

// ─── Step 2.D.28 (#1): threadgroup-storage variants for 6-max parallel
// kernel. Same math as _thread variants; input arrays (reach,
// hand_strength) live in threadgroup memory so they can be shared across
// threads in a threadgroup. Internal per-thread accumulators (b_a_pc[52]
// etc) stay in thread storage.
inline float factored_share_k1_tg(
    int oi,
    bool elig,
    ulong dead_mask,
    ushort h_str,
    uint tied_offset,
    threadgroup const float* reach,
    const device uchar* hand_cards,
    threadgroup const ushort* hand_strength,
    int nh
) {
    float share = 0.0f;
    int o_off = oi * nh;
    for (int g = 0; g < nh; g++) {
        ulong g_m = (1ul << hand_cards[g * 2]) | (1ul << hand_cards[g * 2 + 1]);
        if ((g_m & dead_mask) != 0) continue;
        float r = reach[o_off + g];
        if (r == 0.0f) continue;
        ushort s = hand_strength[g];
        if (elig) {
            if (s > h_str) continue;
            uint t = (s == h_str) ? (tied_offset + 1) : tied_offset;
            share += r / (1.0f + (float)t);
        } else {
            share += r / (1.0f + (float)tied_offset);
        }
    }
    return share;
}

inline float factored_share_k2_thread(
    int oa, int ob,
    bool ea, bool eb,
    ulong dead_mask,
    ushort h_str,
    uint tied_offset,
    thread const float* reach,
    const device uchar* hand_cards,
    thread const ushort* hand_strength,
    int nh
) {
    int oa_off = oa * nh;
    int ob_off = ob * nh;
    float b_a = 0.0f, t_a = 0.0f, s_a = 0.0f;
    float b_b = 0.0f, t_b = 0.0f, s_b = 0.0f;
    float b_a_pc[52], t_a_pc[52], s_a_pc[52];
    float b_b_pc[52], t_b_pc[52], s_b_pc[52];
    for (int c = 0; c < 52; c++) {
        b_a_pc[c] = 0.0f; t_a_pc[c] = 0.0f; s_a_pc[c] = 0.0f;
        b_b_pc[c] = 0.0f; t_b_pc[c] = 0.0f; s_b_pc[c] = 0.0f;
    }
    float h_bb = 0.0f, h_tt = 0.0f, h_ss = 0.0f;
    for (int g = 0; g < nh; g++) {
        ulong g_m = (1ul << hand_cards[g * 2]) | (1ul << hand_cards[g * 2 + 1]);
        if ((g_m & dead_mask) != 0) continue;
        float r_a = reach[oa_off + g];
        float r_b = reach[ob_off + g];
        if (r_a == 0.0f && r_b == 0.0f) continue;
        ushort s = hand_strength[g];
        int gc1 = hand_cards[g * 2];
        int gc2 = hand_cards[g * 2 + 1];
        if (s < h_str) {
            b_a += r_a; b_a_pc[gc1] += r_a; b_a_pc[gc2] += r_a;
            b_b += r_b; b_b_pc[gc1] += r_b; b_b_pc[gc2] += r_b;
            h_bb += r_a * r_b;
        } else if (s == h_str) {
            t_a += r_a; t_a_pc[gc1] += r_a; t_a_pc[gc2] += r_a;
            t_b += r_b; t_b_pc[gc1] += r_b; t_b_pc[gc2] += r_b;
            h_tt += r_a * r_b;
        } else {
            s_a += r_a; s_a_pc[gc1] += r_a; s_a_pc[gc2] += r_a;
            s_b += r_b; s_b_pc[gc1] += r_b; s_b_pc[gc2] += r_b;
            h_ss += r_a * r_b;
        }
    }
    float r_a_tot = b_a + t_a + s_a;
    float r_b_tot = b_b + t_b + s_b;
    float h_tot = h_bb + h_tt + h_ss;
    float edge_bb = 0.0f, edge_bt = 0.0f, edge_tb = 0.0f, edge_tt = 0.0f;
    float edge_be = 0.0f, edge_te = 0.0f, edge_eb = 0.0f, edge_et = 0.0f;
    float edge_ee = 0.0f;
    for (int c = 0; c < 52; c++) {
        if ((dead_mask & (1ul << c)) != 0) continue;
        float bac = b_a_pc[c], tac = t_a_pc[c], sac = s_a_pc[c];
        float bbc = b_b_pc[c], tbc = t_b_pc[c], sbc = s_b_pc[c];
        float rac = bac + tac + sac;
        float rbc = bbc + tbc + sbc;
        edge_bb += bac * bbc; edge_bt += bac * tbc;
        edge_tb += tac * bbc; edge_tt += tac * tbc;
        edge_be += bac * rbc; edge_te += tac * rbc;
        edge_eb += rac * bbc; edge_et += rac * tbc;
        edge_ee += rac * rbc;
    }
    float pair_bb = b_a * b_b - edge_bb + h_bb;
    float pair_bt = b_a * t_b - edge_bt;
    float pair_tb = t_a * b_b - edge_tb;
    float pair_tt = t_a * t_b - edge_tt + h_tt;
    float pair_be = b_a * r_b_tot - edge_be + h_bb;
    float pair_te = t_a * r_b_tot - edge_te + h_tt;
    float pair_eb = r_a_tot * b_b - edge_eb + h_bb;
    float pair_et = r_a_tot * t_b - edge_et + h_tt;
    float pair_ee = r_a_tot * r_b_tot - edge_ee + h_tot;
    float tf = (float)tied_offset;
    if (ea && eb) {
        return pair_bb / (1.0f + tf) + pair_bt / (2.0f + tf) + pair_tb / (2.0f + tf) + pair_tt / (3.0f + tf);
    } else if (ea && !eb) {
        return pair_be / (1.0f + tf) + pair_te / (2.0f + tf);
    } else if (!ea && eb) {
        return pair_eb / (1.0f + tf) + pair_et / (2.0f + tf);
    } else {
        return pair_ee / (1.0f + tf);
    }
}

// ─── Step 2.D.28 (#1): threadgroup-storage variant of factored_share_k2.
inline float factored_share_k2_tg(
    int oa, int ob,
    bool ea, bool eb,
    ulong dead_mask,
    ushort h_str,
    uint tied_offset,
    threadgroup const float* reach,
    const device uchar* hand_cards,
    threadgroup const ushort* hand_strength,
    int nh
) {
    int oa_off = oa * nh;
    int ob_off = ob * nh;
    float b_a = 0.0f, t_a = 0.0f, s_a = 0.0f;
    float b_b = 0.0f, t_b = 0.0f, s_b = 0.0f;
    float b_a_pc[52], t_a_pc[52], s_a_pc[52];
    float b_b_pc[52], t_b_pc[52], s_b_pc[52];
    for (int c = 0; c < 52; c++) {
        b_a_pc[c] = 0.0f; t_a_pc[c] = 0.0f; s_a_pc[c] = 0.0f;
        b_b_pc[c] = 0.0f; t_b_pc[c] = 0.0f; s_b_pc[c] = 0.0f;
    }
    float h_bb = 0.0f, h_tt = 0.0f, h_ss = 0.0f;
    for (int g = 0; g < nh; g++) {
        ulong g_m = (1ul << hand_cards[g * 2]) | (1ul << hand_cards[g * 2 + 1]);
        if ((g_m & dead_mask) != 0) continue;
        float r_a = reach[oa_off + g];
        float r_b = reach[ob_off + g];
        if (r_a == 0.0f && r_b == 0.0f) continue;
        ushort s = hand_strength[g];
        int gc1 = hand_cards[g * 2];
        int gc2 = hand_cards[g * 2 + 1];
        if (s < h_str) {
            b_a += r_a; b_a_pc[gc1] += r_a; b_a_pc[gc2] += r_a;
            b_b += r_b; b_b_pc[gc1] += r_b; b_b_pc[gc2] += r_b;
            h_bb += r_a * r_b;
        } else if (s == h_str) {
            t_a += r_a; t_a_pc[gc1] += r_a; t_a_pc[gc2] += r_a;
            t_b += r_b; t_b_pc[gc1] += r_b; t_b_pc[gc2] += r_b;
            h_tt += r_a * r_b;
        } else {
            s_a += r_a; s_a_pc[gc1] += r_a; s_a_pc[gc2] += r_a;
            s_b += r_b; s_b_pc[gc1] += r_b; s_b_pc[gc2] += r_b;
            h_ss += r_a * r_b;
        }
    }
    float r_a_tot = b_a + t_a + s_a;
    float r_b_tot = b_b + t_b + s_b;
    float h_tot = h_bb + h_tt + h_ss;
    float edge_bb = 0.0f, edge_bt = 0.0f, edge_tb = 0.0f, edge_tt = 0.0f;
    float edge_be = 0.0f, edge_te = 0.0f, edge_eb = 0.0f, edge_et = 0.0f;
    float edge_ee = 0.0f;
    for (int c = 0; c < 52; c++) {
        if ((dead_mask & (1ul << c)) != 0) continue;
        float bac = b_a_pc[c], tac = t_a_pc[c], sac = s_a_pc[c];
        float bbc = b_b_pc[c], tbc = t_b_pc[c], sbc = s_b_pc[c];
        float rac = bac + tac + sac;
        float rbc = bbc + tbc + sbc;
        edge_bb += bac * bbc; edge_bt += bac * tbc;
        edge_tb += tac * bbc; edge_tt += tac * tbc;
        edge_be += bac * rbc; edge_te += tac * rbc;
        edge_eb += rac * bbc; edge_et += rac * tbc;
        edge_ee += rac * rbc;
    }
    float pair_bb = b_a * b_b - edge_bb + h_bb;
    float pair_bt = b_a * t_b - edge_bt;
    float pair_tb = t_a * b_b - edge_tb;
    float pair_tt = t_a * t_b - edge_tt + h_tt;
    float pair_be = b_a * r_b_tot - edge_be + h_bb;
    float pair_te = t_a * r_b_tot - edge_te + h_tt;
    float pair_eb = r_a_tot * b_b - edge_eb + h_bb;
    float pair_et = r_a_tot * t_b - edge_et + h_tt;
    float pair_ee = r_a_tot * r_b_tot - edge_ee + h_tot;
    float tf = (float)tied_offset;
    if (ea && eb) {
        return pair_bb / (1.0f + tf) + pair_bt / (2.0f + tf) + pair_tb / (2.0f + tf) + pair_tt / (3.0f + tf);
    } else if (ea && !eb) {
        return pair_be / (1.0f + tf) + pair_te / (2.0f + tf);
    } else if (!ea && eb) {
        return pair_eb / (1.0f + tf) + pair_et / (2.0f + tf);
    } else {
        return pair_ee / (1.0f + tf);
    }
}

inline float factored_share_for_level_thread(
    int num_opp,
    uint elig_opps,
    uint tied_offset,
    ulong h_m,
    ushort h_str,
    thread const float* reach,
    const device uchar* hand_cards,
    thread const ushort* hand_strength,
    int nh
) {
    if (num_opp == 1) {
        bool e0 = (elig_opps >> 0) & 1u;
        return factored_share_k1_thread(0, e0, h_m, h_str, tied_offset,
                                         reach, hand_cards, hand_strength, nh);
    }
    if (num_opp == 2) {
        bool e0 = (elig_opps >> 0) & 1u;
        bool e1 = (elig_opps >> 1) & 1u;
        return factored_share_k2_thread(0, 1, e0, e1, h_m, h_str, tied_offset,
                                         reach, hand_cards, hand_strength, nh);
    }
    int k2a = num_opp - 2;
    int k2b = num_opp - 1;
    bool ek2a = (elig_opps >> k2a) & 1u;
    bool ek2b = (elig_opps >> k2b) & 1u;
    bool e0 = (elig_opps >> 0) & 1u;
    float share = 0.0f;
    for (int g0 = 0; g0 < nh; g0++) {
        ulong g0_m = (1ul << hand_cards[g0 * 2]) | (1ul << hand_cards[g0 * 2 + 1]);
        if ((g0_m & h_m) != 0) continue;
        float r0 = reach[0 * nh + g0];
        if (r0 == 0.0f) continue;
        ushort s0 = hand_strength[g0];
        uint t0;
        if (e0) {
            if (s0 > h_str) continue;
            t0 = (s0 == h_str) ? (tied_offset + 1) : tied_offset;
        } else {
            t0 = tied_offset;
        }
        ulong m1 = h_m | g0_m;
        if (num_opp == 3) {
            share += r0 * factored_share_k2_thread(k2a, k2b, ek2a, ek2b, m1, h_str, t0,
                                                    reach, hand_cards, hand_strength, nh);
            continue;
        }
        bool e1 = (elig_opps >> 1) & 1u;
        for (int g1 = 0; g1 < nh; g1++) {
            ulong g1_m = (1ul << hand_cards[g1 * 2]) | (1ul << hand_cards[g1 * 2 + 1]);
            if ((g1_m & m1) != 0) continue;
            float r1 = reach[1 * nh + g1];
            if (r1 == 0.0f) continue;
            ushort s1 = hand_strength[g1];
            uint t01;
            if (e1) {
                if (s1 > h_str) continue;
                t01 = (s1 == h_str) ? (t0 + 1) : t0;
            } else {
                t01 = t0;
            }
            ulong m2 = m1 | g1_m;
            float p01 = r0 * r1;
            if (num_opp == 4) {
                share += p01 * factored_share_k2_thread(k2a, k2b, ek2a, ek2b, m2, h_str, t01,
                                                         reach, hand_cards, hand_strength, nh);
                continue;
            }
            bool e2 = (elig_opps >> 2) & 1u;
            for (int g2 = 0; g2 < nh; g2++) {
                ulong g2_m = (1ul << hand_cards[g2 * 2]) | (1ul << hand_cards[g2 * 2 + 1]);
                if ((g2_m & m2) != 0) continue;
                float r2 = reach[2 * nh + g2];
                if (r2 == 0.0f) continue;
                ushort s2 = hand_strength[g2];
                uint t012;
                if (e2) {
                    if (s2 > h_str) continue;
                    t012 = (s2 == h_str) ? (t01 + 1) : t01;
                } else {
                    t012 = t01;
                }
                ulong m3 = m2 | g2_m;
                float p012 = p01 * r2;
                share += p012 * factored_share_k2_thread(k2a, k2b, ek2a, ek2b, m3, h_str, t012,
                                                          reach, hand_cards, hand_strength, nh);
            }
        }
    }
    return share;
}

// ─── Step 2.D.28 (#1): threadgroup-storage variant of factored_share_for_level.
// Same recursive K-1 expansion as _thread variant but operates on threadgroup-
// storage `reach` and `hand_strength` arrays so multiple threads in a group
// can call this concurrently for different h values without per-thread copies.
inline float factored_share_for_level_tg(
    int num_opp,
    uint elig_opps,
    uint tied_offset,
    ulong h_m,
    ushort h_str,
    threadgroup const float* reach,
    const device uchar* hand_cards,
    threadgroup const ushort* hand_strength,
    int nh
) {
    if (num_opp == 1) {
        bool e0 = (elig_opps >> 0) & 1u;
        return factored_share_k1_tg(0, e0, h_m, h_str, tied_offset,
                                     reach, hand_cards, hand_strength, nh);
    }
    if (num_opp == 2) {
        bool e0 = (elig_opps >> 0) & 1u;
        bool e1 = (elig_opps >> 1) & 1u;
        return factored_share_k2_tg(0, 1, e0, e1, h_m, h_str, tied_offset,
                                     reach, hand_cards, hand_strength, nh);
    }
    int k2a = num_opp - 2;
    int k2b = num_opp - 1;
    bool ek2a = (elig_opps >> k2a) & 1u;
    bool ek2b = (elig_opps >> k2b) & 1u;
    bool e0 = (elig_opps >> 0) & 1u;
    float share = 0.0f;
    for (int g0 = 0; g0 < nh; g0++) {
        ulong g0_m = (1ul << hand_cards[g0 * 2]) | (1ul << hand_cards[g0 * 2 + 1]);
        if ((g0_m & h_m) != 0) continue;
        float r0 = reach[0 * nh + g0];
        if (r0 == 0.0f) continue;
        ushort s0 = hand_strength[g0];
        uint t0;
        if (e0) {
            if (s0 > h_str) continue;
            t0 = (s0 == h_str) ? (tied_offset + 1) : tied_offset;
        } else {
            t0 = tied_offset;
        }
        ulong m1 = h_m | g0_m;
        if (num_opp == 3) {
            share += r0 * factored_share_k2_tg(k2a, k2b, ek2a, ek2b, m1, h_str, t0,
                                                reach, hand_cards, hand_strength, nh);
            continue;
        }
        bool e1 = (elig_opps >> 1) & 1u;
        for (int g1 = 0; g1 < nh; g1++) {
            ulong g1_m = (1ul << hand_cards[g1 * 2]) | (1ul << hand_cards[g1 * 2 + 1]);
            if ((g1_m & m1) != 0) continue;
            float r1 = reach[1 * nh + g1];
            if (r1 == 0.0f) continue;
            ushort s1 = hand_strength[g1];
            uint t01;
            if (e1) {
                if (s1 > h_str) continue;
                t01 = (s1 == h_str) ? (t0 + 1) : t0;
            } else {
                t01 = t0;
            }
            ulong m2 = m1 | g1_m;
            float p01 = r0 * r1;
            if (num_opp == 4) {
                share += p01 * factored_share_k2_tg(k2a, k2b, ek2a, ek2b, m2, h_str, t01,
                                                     reach, hand_cards, hand_strength, nh);
                continue;
            }
            bool e2 = (elig_opps >> 2) & 1u;
            for (int g2 = 0; g2 < nh; g2++) {
                ulong g2_m = (1ul << hand_cards[g2 * 2]) | (1ul << hand_cards[g2 * 2 + 1]);
                if ((g2_m & m2) != 0) continue;
                float r2 = reach[2 * nh + g2];
                if (r2 == 0.0f) continue;
                ushort s2 = hand_strength[g2];
                uint t012;
                if (e2) {
                    if (s2 > h_str) continue;
                    t012 = (s2 == h_str) ? (t01 + 1) : t01;
                } else {
                    t012 = t01;
                }
                ulong m3 = m2 | g2_m;
                float p012 = p01 * r2;
                share += p012 * factored_share_k2_tg(k2a, k2b, ek2a, ek2b, m3, h_str, t012,
                                                      reach, hand_cards, hand_strength, nh);
            }
        }
    }
    return share;
}

// ============================================================================
// Step 2.D.28: vcfr_bottom_up_batched_tg_parallel (6-max parallel kernel)
//
// 6-max throughput primitive. Each threadgroup processes ONE (outcome,
// node_in_level) pair. Threads within the group cooperatively load
// opp_reach + hand_strength into threadgroup memory, then parallelize
// the outer h loop via tid stride.
//
// Only valid for num_opp >= 3 (np >= 4). The K>=3 factored path has a
// fully-independent outer h loop, so per-h work is embarrassingly
// parallel within the threadgroup. CPU dispatcher branches: HU/3p use
// the existing vcfr_bottom_up_batched kernel (different math paths),
// 4p/5p/6p use this kernel.
//
// Threadgroup memory footprint:
//   opp_reach_tg:      5 * 1326 * 4 = 26,520 B
//   hand_strength_tg:  1326 * 2     =  2,652 B
//   contribs_tg:       8 * 4        =     32 B
//   Total:                          ~= 29,204 B (under 32 KB Apple limit)
//
// All math matches the K>=3 branch of multiway_brute_force_showdown
// (lines 700-797). PLAYER and CHANCE nodes are also parallelized over
// h with tid-stride - same float ordering as the serial kernel since
// each h slot is independent.
// ============================================================================
kernel void vcfr_bottom_up_batched_tg_parallel(
    device const uint32_t* level_nodes       [[buffer(0)]],
    constant BatchedParams& params           [[buffer(1)]],
    device const FlatNode* nodes             [[buffer(2)]],
    device const uint32_t* children          [[buffer(3)]],
    device const int32_t* contributions      [[buffer(4)]],
    device const uint16_t* folded_masks      [[buffer(5)]],
    device const float* strategy             [[buffer(6)]],
    device const uint32_t* infoset_offsets   [[buffer(7)]],
    device const float* reach                [[buffer(8)]],
    device float* cfv                        [[buffer(9)]],
    device float* regrets                    [[buffer(10)]],
    device float* cum_strategy               [[buffer(11)]],
    device const float* initial_weight       [[buffer(12)]],
    device const uint16_t* sorted_opp_str    [[buffer(13)]],
    device const uint16_t* sorted_opp_idx    [[buffer(14)]],
    device const uint16_t* sorted_pl_str     [[buffer(15)]],
    device const uint16_t* sorted_pl_idx     [[buffer(16)]],
    device const uint8_t* hand_cards         [[buffer(17)]],
    device const float* chance_prob          [[buffer(18)]],
    device float* debug_out                  [[buffer(19)]],
    device uchar* rake_marker                [[buffer(20)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    int outcome = int(tgid) / params.level_count;
    int node_in_level = int(tgid) % params.level_count;
    if (outcome >= params.num_outcomes) return;

    uint node_id = level_nodes[node_in_level];
    FlatNode node = nodes[node_id];
    int np = params.num_players;
    int nh = params.nh;
    int num_opp = np - 1;

    int cfv_off = outcome * params.cfv_batch_stride;
    int sps_off = outcome * params.sorted_opp_stride;

    const device uint16_t* pl_str = sorted_pl_str + sps_off;
    const device uint16_t* pl_idx = sorted_pl_idx + sps_off;
    device float* cfv_o = cfv + cfv_off;

    // -- Threadgroup memory (static allocation; ~29 KB total) --
    threadgroup float opp_reach_tg[5 * 1326];
    threadgroup ushort hand_strength_tg[1326];
    threadgroup int32_t contribs_tg[8];

    // === TERMINAL NODE (K>=3 factored showdown, parallel over h) ===
    if (node.node_type == NODE_TYPE_TERMINAL) {
        int node_reach_base = int(node_id) * np * nh;
        uint16_t fold_mask = folded_masks[int(node_id)];
        device float* out = cfv_o + int(node_id) * nh;

        // Cooperative load: contribs (np <= 8 so 1 thread per p).
        if (int(tid) < np) {
            contribs_tg[tid] = contributions[int(node_id) * np + int(tid)];
        }

        // Cooperative build hand_strength_tg from sorted_pl arrays.
        for (int si = int(tid); si < nh; si += int(tg_size)) {
            hand_strength_tg[pl_idx[si]] = pl_str[si];
        }

        // Cooperative build opp_reach_tg with chance_prob masking.
        for (int oi = 0; oi < num_opp; oi++) {
            int opp = (oi < int(params.traverser)) ? oi : (oi + 1);
            const device float* opp_r = reach + node_reach_base + opp * nh;
            for (int h = int(tid); h < nh; h += int(tg_size)) {
                opp_reach_tg[oi * nh + h] = (chance_prob[h] == 0.0f) ? 0.0f : opp_r[h];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // All threads now redundantly compute per-h constants. Cheap
        // (np<=8 loops) and avoids cross-thread state. K>=3 factored
        // math below matches multiway_brute_force_showdown lines 700-797.
        int32_t c_t = contribs_tg[params.traverser];
        bool flop_seen = (node.board_state != 3);
        float eff_rake_rate = flop_seen ? params.rake_rate : 0.0f;
        float eff_rake_cap  = flop_seen ? params.rake_cap  : 0.0f;

        int levels[8];
        int num_levels = 0;
        for (int p = 0; p < np; p++) {
            int32_t c = contribs_tg[p];
            bool found = false;
            for (int l = 0; l < num_levels; l++) {
                if (levels[l] == c) { found = true; break; }
            }
            if (!found && num_levels < 8) { levels[num_levels++] = (int)c; }
        }
        for (int i = 0; i < num_levels - 1; i++) {
            for (int j = i + 1; j < num_levels; j++) {
                if (levels[j] < levels[i]) {
                    int tmp = levels[i]; levels[i] = levels[j]; levels[j] = tmp;
                }
            }
        }

        float traverser_stake = float(params.starting_pot) / float(np) + float(c_t);
        bool traverser_folded = (fold_mask & (uint16_t)(1u << params.traverser)) != 0;

        int32_t e_main_pot_amount;
        if (num_levels == 0) {
            e_main_pot_amount = params.starting_pot;
        } else {
            int num_main_contributors = 0;
            for (int p = 0; p < np; p++) {
                if (contribs_tg[p] >= levels[0]) num_main_contributors++;
            }
            e_main_pot_amount = levels[0] * num_main_contributors + params.starting_pot;
        }
        float e_main_pot_rake = fmax(0.0f, fmin(
            (float)e_main_pot_amount * eff_rake_rate, eff_rake_cap));

        uchar marker = flop_seen ? (uchar)1 : (uchar)2;

        // -- PARALLEL outer h loop --
        for (int h = int(tid); h < nh; h += int(tg_size)) {
            ulong h_m = (1ul << hand_cards[h * 2]) | (1ul << hand_cards[h * 2 + 1]);
            ushort h_str = hand_strength_tg[h];

            float tvrp = factored_share_for_level_tg(
                num_opp, 0u, 0u, h_m, h_str,
                opp_reach_tg, hand_cards, hand_strength_tg, nh);

            float static_cash = 0.0f;
            float case_c = 0.0f;
            int prev_l = 0;
            for (int li = 0; li < num_levels; li++) {
                int lev = levels[li];
                int pc = lev - prev_l;
                int num_contrib = 0;
                for (int p = 0; p < np; p++) {
                    if (contribs_tg[p] >= lev) num_contrib++;
                }
                float pot_l = (float)(pc * num_contrib);
                if (li == 0) pot_l += (float)params.starting_pot;
                if (pot_l == 0.0f) { prev_l = lev; continue; }
                float pot_after_rake = (li == 0) ? (pot_l - e_main_pot_rake) : pot_l;

                uint elig_opps = 0u;
                int oi = 0;
                for (int p = 0; p < np; p++) {
                    if (p == int(params.traverser)) continue;
                    bool p_folded = (fold_mask & (uint16_t)(1u << p)) != 0;
                    bool p_elig = !p_folded && (contribs_tg[p] >= lev);
                    if (p_elig) elig_opps |= (1u << oi);
                    oi++;
                }
                bool trav_elig = !traverser_folded && (c_t >= lev);
                bool has_active_elig = (elig_opps != 0);

                if (!has_active_elig && trav_elig) {
                    static_cash += pot_after_rake;
                } else if (!has_active_elig && !trav_elig) {
                    if (contribs_tg[params.traverser] >= lev) {
                        float trav_contrib = (float)pc;
                        if (li == 0) trav_contrib += (float)params.starting_pot / (float)np;
                        static_cash += trav_contrib;
                    }
                } else if (!trav_elig) {
                    // Case D: traverser ineligible at contested level - no cash.
                } else {
                    float share = factored_share_for_level_tg(
                        num_opp, elig_opps, 0u, h_m, h_str,
                        opp_reach_tg, hand_cards, hand_strength_tg, nh);
                    case_c += pot_after_rake * share;
                }
                prev_l = lev;
            }

            float cfv_val = (static_cash - traverser_stake) * tvrp + case_c;
            if (params.num_combinations > 0.0f) {
                cfv_val /= params.num_combinations;
            }
            out[h] = cfv_val;
            rake_marker[node_id * nh + h] = marker;
        }
        return;
    }

    // === CHANCE NODE (parallel over h) ===
    if (node.node_type == NODE_TYPE_CHANCE) {
        device float* out = cfv_o + int(node_id) * nh;
        int n_children = int(node.num_children);
        uint children_start = node.children_start;
        for (int h = int(tid); h < nh; h += int(tg_size)) {
            float sum = 0.0f;
            for (int a = 0; a < n_children; a++) {
                uint child = children[children_start + a];
                sum += cfv_o[int(child) * nh + h];
            }
            out[h] = sum;
        }
        return;
    }

    // === PLAYER NODE (parallel over h) ===
    int owner = int(node.player_id);
    int na = int(node.num_children);
    uint infoset_id = infoset_offsets[int(node_id)];
    int stride = MAX_NA * nh;
    int strat_outcome_off = outcome * params.regret_outcome_stride;
    const device float* sigma = strategy + strat_outcome_off + int(infoset_id) * stride;
    uint children_start = node.children_start;
    device float* out_node = cfv_o + int(node_id) * nh;

    if (owner == int(params.traverser)) {
        int regret_base = outcome * params.regret_outcome_stride + int(infoset_id) * stride;
        int cum_base = outcome * params.cum_outcome_stride + int(infoset_id) * stride;

        bool re_enable_iter = (params.pruning_stride > 0)
            && (params.iteration % params.pruning_stride == 0);

        for (int h = int(tid); h < nh; h += int(tg_size)) {
            // Compute cfv_avg for this h.
            float cfv_avg_h = 0.0f;
            for (int a = 0; a < na; a++) {
                uint child = children[children_start + a];
                cfv_avg_h += sigma[a * nh + h] * cfv_o[int(child) * nh + h];
            }

            // Per-action regret/cum update.
            for (int a = 0; a < na; a++) {
                uint child = children[children_start + a];
                bool action_leads_to_terminal = (nodes[child].node_type == NODE_TYPE_TERMINAL);
                bool can_prune_this_action = (params.pruning_enabled != 0)
                    && !re_enable_iter
                    && (params.board_state != 2)
                    && !action_leads_to_terminal;

                float inst_regret = cfv_o[int(child) * nh + h] - cfv_avg_h;
                int ridx = regret_base + a * nh + h;
                float old_r = regrets[ridx];

                if (can_prune_this_action && old_r < params.pruning_threshold) {
                    continue;
                }

                float coef = (old_r >= 0.0f) ? params.alpha_t : params.beta_t;
                float new_r = coef * old_r + inst_regret;
                if (new_r < params.regret_floor) new_r = params.regret_floor;
                regrets[ridx] = new_r;

                int cidx = cum_base + a * nh + h;
                cum_strategy[cidx] = params.gamma_t * cum_strategy[cidx] + sigma[a * nh + h];
            }

            out_node[h] = cfv_avg_h;
        }
    } else {
        // Non-traverser: just sum child CFVs (unweighted).
        for (int h = int(tid); h < nh; h += int(tg_size)) {
            float cfv_avg_h = 0.0f;
            for (int a = 0; a < na; a++) {
                uint child = children[children_start + a];
                cfv_avg_h += cfv_o[int(child) * nh + h];
            }
            out_node[h] = cfv_avg_h;
        }
    }
}

// ============================================================================
// vcfr_bottom_up_tg_parallel
//
// Threadgroup-parallel port of vcfr_bottom_up (used by bottom_up_flop). Same
// pattern as vcfr_bottom_up_batched_tg_parallel but for the FLOP zone, which
// uses BottomUpParams (single outcome, no batching) and a different buffer
// layout (no chance_prob, rake_marker at buffer 18 not 20).
//
// M2 measured that bottom_up_flop is 85-89% of 6p iter cost — this kernel is
// the targeted optimization for that bottleneck.
//
// Dispatch: 1 threadgroup per node, TG_SIZE threads cooperating.
//   K>=3 (num_opp>=3): full tg-parallel TERMINAL path via factored_share_for_level_tg
//   K==2 (3p):         tg-parallel TERMINAL path via factored_share_k2_tg
//   PLAYER/CHANCE:     tid-stride over h
// CPU dispatcher routes num_opp >= 2 here; HU stays on serial vcfr_bottom_up.
// ============================================================================
kernel void vcfr_bottom_up_tg_parallel(
    device const uint32_t* level_nodes       [[buffer(0)]],
    constant BottomUpParams& params          [[buffer(1)]],
    device const FlatNode* nodes             [[buffer(2)]],
    device const uint32_t* children          [[buffer(3)]],
    device const int32_t* contributions      [[buffer(4)]],
    device const uint16_t* folded_masks      [[buffer(5)]],
    device const float* strategy             [[buffer(6)]],
    device const uint32_t* infoset_offsets   [[buffer(7)]],
    device const float* reach                [[buffer(8)]],
    device float* cfv                        [[buffer(9)]],
    device float* regrets                    [[buffer(10)]],
    device float* cum_strategy               [[buffer(11)]],
    device const float* initial_weight       [[buffer(12)]],
    device const uint16_t* sorted_opp_strength  [[buffer(13)]],
    device const uint16_t* sorted_opp_indices   [[buffer(14)]],
    device const uint16_t* sorted_pl_strength   [[buffer(15)]],
    device const uint16_t* sorted_pl_indices    [[buffer(16)]],
    device const uint8_t* hand_cards         [[buffer(17)]],
    device uchar* rake_marker                [[buffer(18)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    int idx = int(tgid);
    if (idx >= params.level_count) return;

    uint node_id = level_nodes[idx];
    FlatNode node = nodes[node_id];
    int np = params.num_players;
    int nh = params.nh;
    int num_opp = np - 1;

    threadgroup float opp_reach_tg[5 * 1326];
    threadgroup ushort hand_strength_tg[1326];
    threadgroup int32_t contribs_tg[8];

    // === TERMINAL NODE (K>=2 factored showdown, parallel over h) ===
    if (node.node_type == NODE_TYPE_TERMINAL) {
        uint node_reach_base = node_id * uint(np) * uint(nh);
        uint16_t fold_mask = folded_masks[node_id];
        device float* out = cfv + node_id * uint(nh);

        // Cooperative loads.
        if (int(tid) < np) {
            contribs_tg[tid] = contributions[node_id * uint(np) + tid];
        }
        for (int si = int(tid); si < nh; si += int(tg_size)) {
            hand_strength_tg[sorted_pl_indices[si]] = sorted_pl_strength[si];
        }
        // NOTE: vcfr_bottom_up does NOT apply chance_prob masking — only
        // batched kernels do (river/turn use chance_prob from the chance node
        // child). At the flop level, reach is already validated by upstream
        // chance propagation.
        for (int oi = 0; oi < num_opp; oi++) {
            int opp = (oi < int(params.traverser)) ? oi : (oi + 1);
            const device float* opp_r = reach + node_reach_base + uint(opp * nh);
            for (int h = int(tid); h < nh; h += int(tg_size)) {
                opp_reach_tg[oi * nh + h] = opp_r[h];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int32_t c_t = contribs_tg[params.traverser];
        bool flop_seen = (node.board_state != 3);
        float eff_rake_rate = flop_seen ? params.rake_rate : 0.0f;
        float eff_rake_cap  = flop_seen ? params.rake_cap  : 0.0f;

        int levels[8];
        int num_levels = 0;
        for (int p = 0; p < np; p++) {
            int32_t c = contribs_tg[p];
            bool found = false;
            for (int l = 0; l < num_levels; l++) {
                if (levels[l] == c) { found = true; break; }
            }
            if (!found && num_levels < 8) { levels[num_levels++] = (int)c; }
        }
        for (int i = 0; i < num_levels - 1; i++) {
            for (int j = i + 1; j < num_levels; j++) {
                if (levels[j] < levels[i]) {
                    int tmp = levels[i]; levels[i] = levels[j]; levels[j] = tmp;
                }
            }
        }

        float traverser_stake = float(params.starting_pot) / float(np) + float(c_t);
        bool traverser_folded = (fold_mask & (uint16_t)(1u << params.traverser)) != 0;

        int32_t e_main_pot_amount;
        if (num_levels == 0) {
            e_main_pot_amount = params.starting_pot;
        } else {
            int num_main_contributors = 0;
            for (int p = 0; p < np; p++) {
                if (contribs_tg[p] >= levels[0]) num_main_contributors++;
            }
            e_main_pot_amount = levels[0] * num_main_contributors + params.starting_pot;
        }
        float e_main_pot_rake = fmax(0.0f, fmin(
            (float)e_main_pot_amount * eff_rake_rate, eff_rake_cap));

        uchar marker = flop_seen ? (uchar)1 : (uchar)2;

        // Parallel h loop via tid stride.
        for (int h = int(tid); h < nh; h += int(tg_size)) {
            ulong h_m = (1ul << hand_cards[h * 2]) | (1ul << hand_cards[h * 2 + 1]);
            ushort h_str = hand_strength_tg[h];

            float tvrp = factored_share_for_level_tg(
                num_opp, 0u, 0u, h_m, h_str,
                opp_reach_tg, hand_cards, hand_strength_tg, nh);

            float static_cash = 0.0f;
            float case_c = 0.0f;
            int prev_l = 0;
            for (int li = 0; li < num_levels; li++) {
                int lev = levels[li];
                int pc = lev - prev_l;
                int num_contrib = 0;
                for (int p = 0; p < np; p++) {
                    if (contribs_tg[p] >= lev) num_contrib++;
                }
                float pot_l = (float)(pc * num_contrib);
                if (li == 0) pot_l += (float)params.starting_pot;
                if (pot_l == 0.0f) { prev_l = lev; continue; }
                float pot_after_rake = (li == 0) ? (pot_l - e_main_pot_rake) : pot_l;

                uint elig_opps = 0u;
                int oi = 0;
                for (int p = 0; p < np; p++) {
                    if (p == int(params.traverser)) continue;
                    bool p_folded = (fold_mask & (uint16_t)(1u << p)) != 0;
                    bool p_elig = !p_folded && (contribs_tg[p] >= lev);
                    if (p_elig) elig_opps |= (1u << oi);
                    oi++;
                }
                bool trav_elig = !traverser_folded && (c_t >= lev);
                bool has_active_elig = (elig_opps != 0);

                if (!has_active_elig && trav_elig) {
                    static_cash += pot_after_rake;
                } else if (!has_active_elig && !trav_elig) {
                    if (contribs_tg[params.traverser] >= lev) {
                        float trav_contrib = (float)pc;
                        if (li == 0) trav_contrib += (float)params.starting_pot / (float)np;
                        static_cash += trav_contrib;
                    }
                } else if (!trav_elig) {
                    // Case D: no cash.
                } else {
                    float share = factored_share_for_level_tg(
                        num_opp, elig_opps, 0u, h_m, h_str,
                        opp_reach_tg, hand_cards, hand_strength_tg, nh);
                    case_c += pot_after_rake * share;
                }
                prev_l = lev;
            }

            float cfv_val = (static_cash - traverser_stake) * tvrp + case_c;
            if (params.num_combinations > 0.0f) {
                cfv_val /= params.num_combinations;
            }
            out[h] = cfv_val;
            rake_marker[node_id * nh + h] = marker;
        }
        return;
    }

    // === CHANCE NODE (parallel over h) ===
    if (node.node_type == NODE_TYPE_CHANCE) {
        int n_children = int(node.num_children);
        uint children_start = node.children_start;
        for (int h = int(tid); h < nh; h += int(tg_size)) {
            float sum = 0.0f;
            for (int a = 0; a < n_children; a++) {
                uint child = children[children_start + a];
                sum += cfv[child * nh + h];
            }
            cfv[node_id * nh + h] = sum;
        }
        return;
    }

    // === PLAYER NODE (parallel over h) ===
    int owner = int(node.player_id);
    int na = int(node.num_children);
    uint infoset_id = infoset_offsets[node_id];
    int stride = MAX_NA * nh;
    const device float* sigma = strategy + infoset_id * stride;
    uint children_start = node.children_start;
    device float* out_node = cfv + node_id * nh;

    if (owner == int(params.traverser)) {
        int offset = infoset_id * stride;
        // Pluribus pruning carve-outs (P1):
        //   - re_enable_iter: every Kth iter is a full traversal
        //   - board_state != 2: NEVER prune on the river
        // action_leads_to_terminal is computed per-action below.
        bool re_enable_iter = (params.pruning_stride > 0)
            && (params.iteration % params.pruning_stride == 0);
        for (int h = int(tid); h < nh; h += int(tg_size)) {
            // Compute cfv_avg for this h.
            float cfv_avg_h = 0.0f;
            for (int a = 0; a < na; a++) {
                uint child = children[children_start + a];
                cfv_avg_h += sigma[a * nh + h] * cfv[child * nh + h];
            }
            // Per-action regret + cum_strategy update.
            for (int a = 0; a < na; a++) {
                uint child = children[children_start + a];
                bool action_leads_to_terminal = (nodes[child].node_type == NODE_TYPE_TERMINAL);
                bool can_prune_this_action = (params.pruning_enabled != 0)
                    && !re_enable_iter
                    && (params.board_state != 2)
                    && !action_leads_to_terminal;
                float inst_regret = cfv[child * nh + h] - cfv_avg_h;
                uint ridx = uint(offset + a * nh + h);
                float old_r = regrets[ridx];
                // Pluribus carve-out applies HERE: skip the regret + cum
                // update if this action was confidently dismissed last iter.
                if (can_prune_this_action && old_r < params.pruning_threshold) {
                    continue;
                }
                float coef = (old_r >= 0.0f) ? params.alpha_t : params.beta_t;
                regrets[ridx] = coef * old_r + inst_regret;
                if (regrets[ridx] < params.regret_floor) regrets[ridx] = params.regret_floor;

                uint cidx = uint(offset + a * nh + h);
                cum_strategy[cidx] = params.gamma_t * cum_strategy[cidx] + sigma[a * nh + h];
            }
            out_node[h] = cfv_avg_h;
        }
    } else {
        for (int h = int(tid); h < nh; h += int(tg_size)) {
            float cfv_avg_h = 0.0f;
            for (int a = 0; a < na; a++) {
                uint child = children[children_start + a];
                cfv_avg_h += cfv[child * nh + h];
            }
            out_node[h] = cfv_avg_h;
        }
    }
}

// ============================================================================
// np=4 (K=3 opponents) FACTORED lone-survivor terminal.
//
// ALL depth-limited street-tree terminals are lone-survivor folds (a fold only
// ends the hand at 1 survivor; showdowns live at the continuation leaves), so
// this kernel family is the live-4 GPU enabler: it replaces the base K>=3
// per-NODE single-thread path (per-h loop x g0 loop x O(nh) k2 inner =
// O(nh^3)/node serial — the seconds-long dispatch that starved WindowServer)
// with one thread per (terminal, hand) at O(nh) each.
//
// Math (EXACT, full joint card removal — validated 1e-14 vs brute triple
// enumeration in solver-core/tests/np4_lone_mass_algebra.rs, whose
// factored_mass3() is the locked reference this ports LITERALLY):
//   cfv[h] = payoff * mass3(h) / num_combinations
//   mass3(h) = sum_{g0 disjoint h} r0[g0] * M2(E),  E = h ∪ g0 (4 cards)
//   M2(E) = S1(E)*S2(E) − (A1(E) − A2(E)) + B(E)
// with per-(terminal,traverser) tables built by the prep kernels below:
//   P1[c], P2[c]      per-card reach mass of opp1 / opp2
//   S1, S2            total masses
//   W[g] = P2[g.a]+P2[g.b]                      (recomputed O(1) from P2)
//   TA1, A1c[c]       total / per-card of  r1[g]*W[g]
//   TB,  Bc[c]        total / per-card of  r1[g]*r2[g]
//   V[d], Vc[c][d]    total / per-card of  r1[g]*U_d[g],
//                     U_d[g] = r2({g.a,d}) + r2({g.b,d})
// Restrictions use  sum_{g⊥E} f = T − Σ_{c∈E} Fc[c] + Σ_{c<d∈E} f(cd-hand).
//
// Table layout per terminal (floats), stride NP4_STRIDE:
//   [0..52)      P1        [52..104)   P2
//   [104..156)   A1c       [156..208)  Bc
//   [208..260)   V         [260..2964) Vc (row-major c*52+d)
//   [2964..2968) scalars: S1, S2, TA1, TB
// ============================================================================

constant int NP4_STRIDE = 2968;
#define NP4_P1   0
#define NP4_P2   52
#define NP4_A1C  104
#define NP4_BC   156
#define NP4_V    208
#define NP4_VC   260
#define NP4_SCAL 2964

// Opponent reach rows for np=4: the three non-traverser players in seat order.
static inline const device float* np4_opp_reach(
    const device float* reach, uint node_id, int np, int nh, int traverser, int oi
) {
    int p = 0, seen = 0, opp = -1;
    for (p = 0; p < np; p++) {
        if (p == traverser) continue;
        if (seen == oi) { opp = p; break; }
        seen++;
    }
    return reach + (node_id * uint(np) + uint(opp)) * uint(nh);
}

// ── prep P: P1[c], P2[c] (+ c==0: S1, S2). Grid (n_term, 52). ──
kernel void vcfr_np4_lone_prep_p(
    device float*          tables         [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const float*    reach          [[buffer(2)]],
    device const uint8_t*  hand_cards     [[buffer(3)]],
    constant LoneTermParams& p            [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x);
    int c  = int(gid.y);
    if (ti >= p.n_term || c >= 52) return;
    int nh = p.nh;
    uint node_id = term_nodes[ti];
    const device float* r1 = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, 1);
    const device float* r2 = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, 2);
    device float* t = tables + ti * NP4_STRIDE;

    float p1 = 0.0f, p2 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    for (int g = 0; g < nh; g++) {
        int a = hand_cards[g * 2];
        int b = hand_cards[g * 2 + 1];
        float x1 = r1[g], x2 = r2[g];
        if (a == c || b == c) { p1 += x1; p2 += x2; }
        if (c == 0) { s1 += x1; s2 += x2; }
    }
    t[NP4_P1 + c] = p1;
    t[NP4_P2 + c] = p2;
    if (c == 0) { t[NP4_SCAL + 0] = s1; t[NP4_SCAL + 1] = s2; }
}

// ── prep Q: A1c[c], Bc[c] (+ c==0: TA1, TB). Needs P2 (hazard-ordered). ──
kernel void vcfr_np4_lone_prep_q(
    device float*          tables         [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const float*    reach          [[buffer(2)]],
    device const uint8_t*  hand_cards     [[buffer(3)]],
    constant LoneTermParams& p            [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x);
    int c  = int(gid.y);
    if (ti >= p.n_term || c >= 52) return;
    int nh = p.nh;
    uint node_id = term_nodes[ti];
    const device float* r1 = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, 1);
    const device float* r2 = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, 2);
    device float* t = tables + ti * NP4_STRIDE;

    float a1c = 0.0f, bc = 0.0f, ta1 = 0.0f, tb = 0.0f;
    for (int g = 0; g < nh; g++) {
        float x1 = r1[g];
        if (x1 == 0.0f) continue;
        int a = hand_cards[g * 2];
        int b = hand_cards[g * 2 + 1];
        float w = t[NP4_P2 + a] + t[NP4_P2 + b];
        float xw = x1 * w;
        float xy = x1 * r2[g];
        if (a == c || b == c) { a1c += xw; bc += xy; }
        if (c == 0) { ta1 += xw; tb += xy; }
    }
    t[NP4_A1C + c] = a1c;
    t[NP4_BC + c]  = bc;
    if (c == 0) { t[NP4_SCAL + 2] = ta1; t[NP4_SCAL + 3] = tb; }
}

// ── prep R: V[d] + Vc[·][d] column. Grid (n_term, 52); thread owns column d. ──
kernel void vcfr_np4_lone_prep_r(
    device float*          tables         [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const float*    reach          [[buffer(2)]],
    device const uint8_t*  hand_cards     [[buffer(3)]],
    device const int32_t*  pair2hand      [[buffer(4)]],
    constant LoneTermParams& p            [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x);
    int d  = int(gid.y);
    if (ti >= p.n_term || d >= 52) return;
    int nh = p.nh;
    uint node_id = term_nodes[ti];
    const device float* r1 = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, 1);
    const device float* r2 = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, 2);
    device float* t = tables + ti * NP4_STRIDE;

    float col[52];
    for (int c = 0; c < 52; c++) col[c] = 0.0f;
    float v = 0.0f;
    for (int g = 0; g < nh; g++) {
        float x1 = r1[g];
        if (x1 == 0.0f) continue;
        int a = hand_cards[g * 2];
        int b = hand_cards[g * 2 + 1];
        int iad = pair2hand[a * 52 + d];
        int ibd = pair2hand[b * 52 + d];
        float u = ((iad >= 0) ? r2[iad] : 0.0f) + ((ibd >= 0) ? r2[ibd] : 0.0f);
        if (u == 0.0f) continue;
        float x = x1 * u;
        v += x;
        col[a] += x;
        col[b] += x;
    }
    t[NP4_V + d] = v;
    for (int c = 0; c < 52; c++) t[NP4_VC + c * 52 + d] = col[c];
}

// ── main: cfv[node][h] = payoff * mass3(h) / nc. Grid (n_term, nh). ──
// LITERAL port of np4_lone_mass_algebra.rs::factored_mass3.
// ── main: cfv[node][h] = payoff * mass3(h) / nc.
// OPTIMIZED v2: dispatch with threadgroups of (1, NP4_TG) — one TERMINAL per
// threadgroup — so the terminal's tables (11.9KB), both reach rows (10.6KB),
// and pair2hand (as i16, 5.4KB) are cooperatively staged in THREADGROUP memory
// (27.2KB < 32KB). The per-g0 inner body reads ~50-100 values; from device
// memory that made the kernel memory-bound (516ms/iter at live-4 full scale).
// h-only corrections are HOISTED out of the g0 loop (same algebra as the
// factored_mass3 reference, reassociated — parity-gated).
constant int NP4_TG = 256;

kernel void vcfr_np4_lone_main(
    device float*          cfv            [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const FlatNode* nodes          [[buffer(2)]],
    device const int32_t*  contributions  [[buffer(3)]],
    device const uint16_t* folded_masks   [[buffer(4)]],
    device const float*    reach          [[buffer(5)]],
    device const uint8_t*  hand_cards     [[buffer(6)]],
    device const int32_t*  pair2hand      [[buffer(7)]],
    device const float*    tables         [[buffer(8)]],
    constant LoneTermParams& p            [[buffer(9)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]]
) {
    int ti = int(gid.x);
    int h  = int(gid.y);
    int nh = p.nh; int np = p.np; int traverser = p.traverser;
    if (ti >= p.n_term) return;

    // ── cooperative stage: tables + reach rows + pair2hand into threadgroup ──
    threadgroup float sh_tab[NP4_STRIDE];   // 11,872 B
    threadgroup float sh_r1[1326];          // 5,304 B
    threadgroup float sh_r2[1326];          // 5,304 B
    threadgroup short sh_p2h[52 * 52];      // 5,408 B
    uint node_id = term_nodes[ti];
    {
        const device float* t = tables + ti * NP4_STRIDE;
        const device float* r1d = np4_opp_reach(reach, node_id, np, nh, traverser, 1);
        const device float* r2d = np4_opp_reach(reach, node_id, np, nh, traverser, 2);
        int lane = int(lid.y);
        int stride = int(tptg.y); // ACTUAL threadgroup size (may be < NP4_TG)
        for (int i = lane; i < NP4_STRIDE; i += stride) sh_tab[i] = t[i];
        for (int i = lane; i < nh; i += stride) { sh_r1[i] = r1d[i]; sh_r2[i] = r2d[i]; }
        for (int i = lane; i < 52 * 52; i += stride) sh_p2h[i] = short(pair2hand[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (h >= nh) return;

    uint16_t fold_mask = folded_masks[node_id];
    FlatNode node = nodes[node_id];

    // payoff — identical convention to vcfr_lone_terminal_factored.
    int c_t = contributions[node_id * uint(np) + uint(traverser)];
    float traverser_stake = float(p.starting_pot) / float(np) + float(c_t);
    bool traverser_folded = (fold_mask & (uint16_t)(1u << traverser)) != 0;
    bool flop_seen = (node.board_state != 3);
    float eff_rake_rate = flop_seen ? p.rake_rate : 0.0f;
    float eff_rake_cap  = flop_seen ? p.rake_cap  : 0.0f;
    int total_pot = p.starting_pot;
    for (int q = 0; q < np; q++) total_pot += contributions[node_id * uint(np) + uint(q)];
    int min_contrib = contributions[node_id * uint(np) + 0u];
    for (int q = 1; q < np; q++) { int c = contributions[node_id * uint(np) + uint(q)]; if (c < min_contrib) min_contrib = c; }
    int num_main = 0;
    for (int q = 0; q < np; q++) if (contributions[node_id * uint(np) + uint(q)] >= min_contrib) num_main++;
    int main_pot_amount = min_contrib * num_main + p.starting_pot;
    float rake = fmax(0.0f, fmin(float(main_pot_amount) * eff_rake_rate, eff_rake_cap));
    float payoff = traverser_folded ? (-traverser_stake) : ((float(total_pot) - rake) - traverser_stake);

    const device float* r0 = np4_opp_reach(reach, node_id, np, nh, traverser, 0);

    float S1  = sh_tab[NP4_SCAL + 0];
    float S2  = sh_tab[NP4_SCAL + 1];
    float TA1 = sh_tab[NP4_SCAL + 2];
    float TB  = sh_tab[NP4_SCAL + 3];

    int h1 = hand_cards[h * 2];
    int h2 = hand_cards[h * 2 + 1];

    // pair-lookup helpers on the staged copies
    #define P2H(x, y)  int(sh_p2h[(x) * 52 + (y)])
    #define R1AT(i)    ((i) >= 0 ? sh_r1[i] : 0.0f)
    #define R2AT(i)    ((i) >= 0 ? sh_r2[i] : 0.0f)

    // ── h-only hoisted corrections ──
    int i_hh = P2H(h1, h2);
    float r1_hh = R1AT(i_hh), r2_hh = R2AT(i_hh);
    float Sh1 = S1 - sh_tab[NP4_P1 + h1] - sh_tab[NP4_P1 + h2] + r1_hh;
    float Sh2 = S2 - sh_tab[NP4_P2 + h1] - sh_tab[NP4_P2 + h2] + r2_hh;
    float w_hh = sh_tab[NP4_P2 + h1] + sh_tab[NP4_P2 + h2];
    float A1h = TA1 - sh_tab[NP4_A1C + h1] - sh_tab[NP4_A1C + h2] + r1_hh * w_hh;
    float Bh  = TB  - sh_tab[NP4_BC + h1]  - sh_tab[NP4_BC + h2]  + r1_hh * r2_hh;
    // A2 h-columns: T_d pure-h parts for d in {h1,h2} (pair (h1,h2) folded in).
    float Th[2];
    int hcards[2] = { h1, h2 };
    for (int kd = 0; kd < 2; kd++) {
        int d = hcards[kd];
        float u_hh = R2AT(P2H(h1, d)) + R2AT(P2H(h2, d));
        Th[kd] = sh_tab[NP4_V + d]
               - sh_tab[NP4_VC + h1 * 52 + d] - sh_tab[NP4_VC + h2 * 52 + d]
               + r1_hh * u_hh;
    }

    float mass = 0.0f;
    for (int g0 = 0; g0 < nh; g0++) {
        float rr0 = r0[g0];
        if (rr0 == 0.0f) continue;
        int a = hand_cards[g0 * 2];
        int b = hand_cards[g0 * 2 + 1];
        if (a == h1 || a == h2 || b == h1 || b == h2) continue;

        // 5 g0-involving E-pairs (the 6th, (h1,h2), is hoisted).
        int i_ah1 = P2H(a, h1), i_ah2 = P2H(a, h2);
        int i_bh1 = P2H(b, h1), i_bh2 = P2H(b, h2);
        int i_ab  = P2H(a, b);
        float r1_ah1 = R1AT(i_ah1), r2_ah1 = R2AT(i_ah1);
        float r1_ah2 = R1AT(i_ah2), r2_ah2 = R2AT(i_ah2);
        float r1_bh1 = R1AT(i_bh1), r2_bh1 = R2AT(i_bh1);
        float r1_bh2 = R1AT(i_bh2), r2_bh2 = R2AT(i_bh2);
        float r1_ab  = R1AT(i_ab),  r2_ab  = R2AT(i_ab);

        float P1a = sh_tab[NP4_P1 + a], P1b = sh_tab[NP4_P1 + b];
        float P2a = sh_tab[NP4_P2 + a], P2b = sh_tab[NP4_P2 + b];

        // S1(E), S2(E)
        float s1e = Sh1 - (P1a - r1_ah1 - r1_ah2) - (P1b - r1_bh1 - r1_bh2) + r1_ab;
        float s2e = Sh2 - (P2a - r2_ah1 - r2_ah2) - (P2b - r2_bh1 - r2_bh2) + r2_ab;

        // A1(E): W(xy) = P2[x] + P2[y]
        float a1e = A1h
            - (sh_tab[NP4_A1C + a] - r1_ah1 * (P2a + sh_tab[NP4_P2 + h1])
                                   - r1_ah2 * (P2a + sh_tab[NP4_P2 + h2]))
            - (sh_tab[NP4_A1C + b] - r1_bh1 * (P2b + sh_tab[NP4_P2 + h1])
                                   - r1_bh2 * (P2b + sh_tab[NP4_P2 + h2]))
            + r1_ab * (P2a + P2b);

        // B(E)
        float be = Bh
            - (sh_tab[NP4_BC + a] - r1_ah1 * r2_ah1 - r1_ah2 * r2_ah2)
            - (sh_tab[NP4_BC + b] - r1_bh1 * r2_bh1 - r1_bh2 * r2_bh2)
            + r1_ab * r2_ab;

        // A2(E) = Σ_{d∈E} T_d
        float a2e = 0.0f;
        // d ∈ {h1,h2}: hoisted pure-h part + g0-dependent corrections.
        for (int kd = 0; kd < 2; kd++) {
            int d = hcards[kd];
            float td = Th[kd]
                - sh_tab[NP4_VC + a * 52 + d] - sh_tab[NP4_VC + b * 52 + d];
            // pairs (h1,a),(h1,b),(h2,a),(h2,b),(a,b): + r1(pair)·U_d(pair)
            td += r1_ah1 * (R2AT(P2H(a, d)) + R2AT(P2H(h1, d)));
            td += r1_bh1 * (R2AT(P2H(b, d)) + R2AT(P2H(h1, d)));
            td += r1_ah2 * (R2AT(P2H(a, d)) + R2AT(P2H(h2, d)));
            td += r1_bh2 * (R2AT(P2H(b, d)) + R2AT(P2H(h2, d)));
            td += r1_ab  * (R2AT(P2H(a, d)) + R2AT(P2H(b, d)));
            a2e += td;
        }
        // d ∈ {a,b}: fully per-g0.
        int gcards[2] = { a, b };
        for (int kd = 0; kd < 2; kd++) {
            int d = gcards[kd];
            float td = sh_tab[NP4_V + d]
                - sh_tab[NP4_VC + h1 * 52 + d] - sh_tab[NP4_VC + h2 * 52 + d]
                - sh_tab[NP4_VC + a * 52 + d]  - sh_tab[NP4_VC + b * 52 + d];
            float u_hh_d = R2AT(P2H(h1, d)) + R2AT(P2H(h2, d));
            td += r1_hh  * u_hh_d;
            td += r1_ah1 * (R2AT(P2H(a, d)) + R2AT(P2H(h1, d)));
            td += r1_bh1 * (R2AT(P2H(b, d)) + R2AT(P2H(h1, d)));
            td += r1_ah2 * (R2AT(P2H(a, d)) + R2AT(P2H(h2, d)));
            td += r1_bh2 * (R2AT(P2H(b, d)) + R2AT(P2H(h2, d)));
            td += r1_ab  * (R2AT(P2H(a, d)) + R2AT(P2H(b, d)));
            a2e += td;
        }

        float m2 = s1e * s2e - (a1e - a2e) + be;
        mass += rr0 * m2;
    }
    #undef P2H
    #undef R1AT
    #undef R2AT

    float v = payoff * mass;
    cfv[node_id * uint(nh) + uint(h)] = (p.num_combinations > 0.0f) ? (v / p.num_combinations) : v;
}

// ============================================================================
// np=4 CLOSED-FORM lone-survivor terminal (v3): O(1) eval per (terminal,hand).
// LITERAL port of `mod cf` in solver-core/tests/np4_lone_mass_algebra.rs
// (validated 2.1e-14 vs brute; see commit 26779e1). Replaces the O(nh)-per-
// thread g0 loop of vcfr_np4_lone_main.
//   mass3(X) = S_part + B_part − D_part over aggregates with ⊥X restriction
//   (total − rows[h1] − rows[h2] + rA[hpair]·feature(hpair)).
// Table layout per terminal (floats), STRIDE NP4CF_STRIDE:
//   PB[52] PC[52] G1[52] BCC[52] G2[2704] GS[4]{SB,SC,G0,TBC}
//   SKT[13] SKR[13*52] VKT[10*52] VKR[10*52*52] MKT[2*2704] MKR[2*52*2704]
// ============================================================================

constant int NP4CF_PB  = 0;
constant int NP4CF_PC  = 52;
constant int NP4CF_G1  = 104;
constant int NP4CF_BCC = 156;
constant int NP4CF_G2  = 208;
constant int NP4CF_GS  = 2912;   // SB,SC,G0,TBC
constant int NP4CF_SKT = 2916;
constant int NP4CF_SKR = 2929;   // k*52 + c
constant int NP4CF_VKT = 3605;   // k*52 + e
constant int NP4CF_VKR = 4125;   // k*2704 + c*52 + e
constant int NP4CF_MKT = 31165;  // k*2704 + e*52 + ep
constant int NP4CF_MKR = 36573;  // k*140608 + c*2704 + e*52 + ep
constant int NP4CF_STRIDE = 317792; // 317789 padded

// SK indices (must match cf::SK_*)
#define SKN 0
#define SKQB 1
#define SKQC 2
#define SKUB 3
#define SKUC 4
#define SKDP 5
#define SKG1 6
#define SKG2G 7
#define SKVBC 8
#define SKQBQC 9
#define SKQBUC 10
#define SKUBQC 11
#define SKUBUC 12
// VK indices
#define VKYB 0
#define VKYC 1
#define VKWBC 2
#define VKKAP 3
#define VKROW 4
#define VKCOL 5
#define VKQBYC 6
#define VKYBQC 7
#define VKUBYC 8
#define VKYBUC 9
// MK indices
#define MKTYY 0
#define MKZSAME 1

// pair reach helper: reach of role r's pair-hand {c,e} (0 if invalid).
static inline float np4cf_pr(const device float* r, const device int32_t* p2h, int c, int e) {
    if (c == e) return 0.0f;
    int i = p2h[c * 52 + e];
    return (i >= 0) ? r[i] : 0.0f;
}

// ── prep A: PB/PC/BCC per card (+ c==0: SB,SC,TBC). Grid (n_term, 52). ──
kernel void vcfr_np4cf_prep_a(
    device float*          tab            [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const float*    reach          [[buffer(2)]],
    device const uint8_t*  hand_cards     [[buffer(3)]],
    constant LoneTermParams& p            [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x); int c = int(gid.y);
    if (ti >= p.n_term || c >= 52) return;
    int nh = p.nh;
    uint node_id = term_nodes[ti];
    const device float* rb = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 1);
    const device float* rc = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 2);
    device float* t = tab + size_t(ti) * NP4CF_STRIDE;
    float pb = 0.0f, pc = 0.0f, bcc = 0.0f, sb = 0.0f, sc = 0.0f, tbc = 0.0f;
    for (int g = 0; g < nh; g++) {
        int a = hand_cards[g * 2], b = hand_cards[g * 2 + 1];
        float vb = rb[g], vc = rc[g];
        if (a == c || b == c) { pb += vb; pc += vc; bcc += vb * vc; }
        if (c == 0) { sb += vb; sc += vc; tbc += vb * vc; }
    }
    t[NP4CF_PB + c] = pb; t[NP4CF_PC + c] = pc; t[NP4CF_BCC + c] = bcc;
    if (c == 0) { t[NP4CF_GS + 0] = sb; t[NP4CF_GS + 1] = sc; t[NP4CF_GS + 3] = tbc; }
}

// ── prep B: G1[e], G2[e][*] (+ e==0: G0). Needs PB/PC. Grid (n_term, 52). ──
kernel void vcfr_np4cf_prep_b(
    device float*          tab            [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const float*    reach          [[buffer(2)]],
    device const int32_t*  p2h            [[buffer(3)]],
    constant LoneTermParams& p            [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x); int e = int(gid.y);
    if (ti >= p.n_term || e >= 52) return;
    uint node_id = term_nodes[ti];
    const device float* rb = np4_opp_reach(reach, node_id, p.np, p.nh, p.traverser, p.inner_role_base + 1);
    const device float* rc = np4_opp_reach(reach, node_id, p.np, p.nh, p.traverser, p.inner_role_base + 2);
    device float* t = tab + size_t(ti) * NP4CF_STRIDE;
    float g1 = 0.0f, g0 = 0.0f;
    for (int c = 0; c < 52; c++) {
        float pbce = np4cf_pr(rb, p2h, c, e);
        float pcce = np4cf_pr(rc, p2h, c, e);
        g1 += t[NP4CF_PB + c] * pcce + t[NP4CF_PC + c] * pbce;
        if (e == 0) g0 += t[NP4CF_PB + c] * t[NP4CF_PC + c];
    }
    t[NP4CF_G1 + e] = g1;
    if (e == 0) t[NP4CF_GS + 2] = g0;
    for (int ep = 0; ep < 52; ep++) {
        float s = 0.0f;
        for (int c = 0; c < 52; c++) {
            s += np4cf_pr(rb, p2h, c, e) * np4cf_pr(rc, p2h, c, ep);
        }
        t[NP4CF_G2 + e * 52 + ep] = s;
    }
}

// scalar features of hand g (cards a,b) given built G-tables.
static inline void np4cf_sk_feats(
    const device float* t, const device float* rb, const device float* rc,
    int g, int a, int b, thread float* out /*13*/
) {
    float qb = rb[g], qc = rc[g];
    float ub = t[NP4CF_PB + a] + t[NP4CF_PB + b];
    float uc = t[NP4CF_PC + a] + t[NP4CF_PC + b];
    out[SKN] = 1.0f; out[SKQB] = qb; out[SKQC] = qc; out[SKUB] = ub; out[SKUC] = uc;
    out[SKDP] = t[NP4CF_PB + a] * t[NP4CF_PC + a] + t[NP4CF_PB + b] * t[NP4CF_PC + b];
    out[SKG1] = t[NP4CF_G1 + a] + t[NP4CF_G1 + b];
    out[SKG2G] = t[NP4CF_G2 + a * 52 + a] + t[NP4CF_G2 + a * 52 + b]
               + t[NP4CF_G2 + b * 52 + a] + t[NP4CF_G2 + b * 52 + b];
    out[SKVBC] = t[NP4CF_BCC + a] + t[NP4CF_BCC + b];
    out[SKQBQC] = qb * qc; out[SKQBUC] = qb * uc; out[SKUBQC] = ub * qc; out[SKUBUC] = ub * uc;
}

// ── prep SK: totals (cell 52) + rows (cell c). Grid (n_term, 53). ──
kernel void vcfr_np4cf_prep_sk(
    device float*          tab            [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const float*    reach          [[buffer(2)]],
    device const uint8_t*  hand_cards     [[buffer(3)]],
    constant LoneTermParams& p            [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x); int cell = int(gid.y);
    if (ti >= p.n_term || cell >= 53) return;
    int nh = p.nh;
    uint node_id = term_nodes[ti];
    const device float* ra = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 0);
    const device float* rb = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 1);
    const device float* rc = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 2);
    device float* t = tab + size_t(ti) * NP4CF_STRIDE;
    float acc[13];
    for (int k = 0; k < 13; k++) acc[k] = 0.0f;
    float f[13];
    for (int g = 0; g < nh; g++) {
        float w = ra[g];
        if (w == 0.0f) continue;
        int a = hand_cards[g * 2], b = hand_cards[g * 2 + 1];
        if (cell < 52 && a != cell && b != cell) continue;
        np4cf_sk_feats(t, rb, rc, g, a, b, f);
        for (int k = 0; k < 13; k++) acc[k] += w * f[k];
    }
    if (cell == 52) { for (int k = 0; k < 13; k++) t[NP4CF_SKT + k] = acc[k]; }
    else { for (int k = 0; k < 13; k++) t[NP4CF_SKR + k * 52 + cell] = acc[k]; }
}

// vector features at (g, e).
static inline void np4cf_vk_feats(
    const device float* t, const device float* rb, const device float* rc,
    const device int32_t* p2h, int g, int a, int b, int e, thread float* out /*10*/
) {
    float pba = np4cf_pr(rb, p2h, a, e), pbb = np4cf_pr(rb, p2h, b, e);
    float pca = np4cf_pr(rc, p2h, a, e), pcb = np4cf_pr(rc, p2h, b, e);
    float yb = pba + pbb, yc = pca + pcb;
    float qb = rb[g], qc = rc[g];
    float ub = t[NP4CF_PB + a] + t[NP4CF_PB + b];
    float uc = t[NP4CF_PC + a] + t[NP4CF_PC + b];
    out[VKYB] = yb; out[VKYC] = yc;
    out[VKWBC] = pba * pca + pbb * pcb;
    out[VKKAP] = t[NP4CF_PB + a] * pca + t[NP4CF_PC + a] * pba
               + t[NP4CF_PB + b] * pcb + t[NP4CF_PC + b] * pbb;
    out[VKROW] = t[NP4CF_G2 + e * 52 + a] + t[NP4CF_G2 + e * 52 + b];
    out[VKCOL] = t[NP4CF_G2 + a * 52 + e] + t[NP4CF_G2 + b * 52 + e];
    out[VKQBYC] = qb * yc; out[VKYBQC] = yb * qc;
    out[VKUBYC] = ub * yc; out[VKYBUC] = yb * uc;
}

// ── prep VK: thread (t, e) owns column e: totals + rows[c][e] for all c. ──
kernel void vcfr_np4cf_prep_vk(
    device float*          tab            [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const float*    reach          [[buffer(2)]],
    device const uint8_t*  hand_cards     [[buffer(3)]],
    device const int32_t*  p2h            [[buffer(4)]],
    constant LoneTermParams& p            [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x); int e = int(gid.y);
    if (ti >= p.n_term || e >= 52) return;
    int nh = p.nh;
    uint node_id = term_nodes[ti];
    const device float* ra = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 0);
    const device float* rb = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 1);
    const device float* rc = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 2);
    device float* t = tab + size_t(ti) * NP4CF_STRIDE;
    float tot[10];
    for (int k = 0; k < 10; k++) tot[k] = 0.0f;
    // zero this thread's row column first (rows are accumulated)
    for (int k = 0; k < 10; k++)
        for (int c = 0; c < 52; c++)
            t[NP4CF_VKR + k * 2704 + c * 52 + e] = 0.0f;
    float f[10];
    for (int g = 0; g < nh; g++) {
        float w = ra[g];
        if (w == 0.0f) continue;
        int a = hand_cards[g * 2], b = hand_cards[g * 2 + 1];
        np4cf_vk_feats(t, rb, rc, p2h, g, a, b, e, f);
        for (int k = 0; k < 10; k++) {
            float v = w * f[k];
            if (v == 0.0f) continue;
            tot[k] += v;
            t[NP4CF_VKR + k * 2704 + a * 52 + e] += v;
            t[NP4CF_VKR + k * 2704 + b * 52 + e] += v;
        }
    }
    for (int k = 0; k < 10; k++) t[NP4CF_VKT + k * 52 + e] = tot[k];
}

// ── prep MK: thread (t, e, ep) owns (e,ep): totals + rows. Grid 3D. ──
kernel void vcfr_np4cf_prep_mk(
    device float*          tab            [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const float*    reach          [[buffer(2)]],
    device const uint8_t*  hand_cards     [[buffer(3)]],
    device const int32_t*  p2h            [[buffer(4)]],
    constant LoneTermParams& p            [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x); int e = int(gid.y); int ep = int(gid.z);
    if (ti >= p.n_term || e >= 52 || ep >= 52) return;
    int nh = p.nh;
    uint node_id = term_nodes[ti];
    const device float* ra = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 0);
    const device float* rb = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 1);
    const device float* rc = np4_opp_reach(reach, node_id, p.np, nh, p.traverser, p.inner_role_base + 2);
    device float* t = tab + size_t(ti) * NP4CF_STRIDE;
    float t_tyy = 0.0f, t_zs = 0.0f;
    for (int c = 0; c < 52; c++) {
        t[NP4CF_MKR + 0 * 140608 + c * 2704 + e * 52 + ep] = 0.0f;
        t[NP4CF_MKR + 1 * 140608 + c * 2704 + e * 52 + ep] = 0.0f;
    }
    for (int g = 0; g < nh; g++) {
        float w = ra[g];
        if (w == 0.0f) continue;
        int a = hand_cards[g * 2], b = hand_cards[g * 2 + 1];
        float pba = np4cf_pr(rb, p2h, a, e), pbb = np4cf_pr(rb, p2h, b, e);
        float pca = np4cf_pr(rc, p2h, a, ep), pcb = np4cf_pr(rc, p2h, b, ep);
        float tyy = (pba + pbb) * (pca + pcb);
        float zs = pba * pca + pbb * pcb;
        if (tyy != 0.0f) {
            float v = w * tyy;
            t_tyy += v;
            t[NP4CF_MKR + 0 * 140608 + a * 2704 + e * 52 + ep] += v;
            t[NP4CF_MKR + 0 * 140608 + b * 2704 + e * 52 + ep] += v;
        }
        if (zs != 0.0f) {
            float v = w * zs;
            t_zs += v;
            t[NP4CF_MKR + 1 * 140608 + a * 2704 + e * 52 + ep] += v;
            t[NP4CF_MKR + 1 * 140608 + b * 2704 + e * 52 + ep] += v;
        }
    }
    t[NP4CF_MKT + 0 * 2704 + e * 52 + ep] = t_tyy;
    t[NP4CF_MKT + 1 * 2704 + e * 52 + ep] = t_zs;
}

// ── restricted-lookup helpers for X = {h1,h2} (single pair add-back). ──
// hp = pair2hand index of {h1,h2} (-1 if invalid); wA = rA[hp] (0 if invalid).
struct Np4cfCtx {
    const device float* t;
    const device float* ra;
    const device float* rb;
    const device float* rc;
    const device int32_t* p2h;
    int h1; int h2;
    int hp;        // pair2hand(h1,h2)
    float wa;      // rA[hp] or 0
};

static inline float np4cf_rsk(thread const Np4cfCtx& x, int k, thread const float* hp_sk) {
    float v = x.t[NP4CF_SKT + k] - x.t[NP4CF_SKR + k * 52 + x.h1] - x.t[NP4CF_SKR + k * 52 + x.h2];
    if (x.wa != 0.0f) v += x.wa * hp_sk[k];
    return v;
}
static inline float np4cf_rvk(thread const Np4cfCtx& x, int k, int e) {
    float v = x.t[NP4CF_VKT + k * 52 + e]
        - x.t[NP4CF_VKR + k * 2704 + x.h1 * 52 + e]
        - x.t[NP4CF_VKR + k * 2704 + x.h2 * 52 + e];
    if (x.wa != 0.0f) {
        float f[10];
        np4cf_vk_feats(x.t, x.rb, x.rc, x.p2h, x.hp, x.h1, x.h2, e, f);
        v += x.wa * f[k];
    }
    return v;
}
static inline float np4cf_rmk(thread const Np4cfCtx& x, int k, int e, int ep) {
    float v = x.t[NP4CF_MKT + k * 2704 + e * 52 + ep]
        - x.t[NP4CF_MKR + k * 140608 + x.h1 * 2704 + e * 52 + ep]
        - x.t[NP4CF_MKR + k * 140608 + x.h2 * 2704 + e * 52 + ep];
    if (x.wa != 0.0f) {
        float pba = np4cf_pr(x.rb, x.p2h, x.h1, e), pbb = np4cf_pr(x.rb, x.p2h, x.h2, e);
        float pca = np4cf_pr(x.rc, x.p2h, x.h1, ep), pcb = np4cf_pr(x.rc, x.p2h, x.h2, ep);
        float f = (k == MKTYY) ? (pba + pbb) * (pca + pcb) : (pba * pca + pbb * pcb);
        v += x.wa * f;
    }
    return v;
}

// ── main v3: cfv[node][h] = payoff · mass3({h1,h2}) / nc — O(1) per thread. ──
kernel void vcfr_np4cf_main(
    device float*          cfv            [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const FlatNode* nodes          [[buffer(2)]],
    device const int32_t*  contributions  [[buffer(3)]],
    device const uint16_t* folded_masks   [[buffer(4)]],
    device const float*    reach          [[buffer(5)]],
    device const uint8_t*  hand_cards     [[buffer(6)]],
    device const int32_t*  p2h            [[buffer(7)]],
    device const float*    tab            [[buffer(8)]],
    constant LoneTermParams& p            [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x);
    int h  = int(gid.y);
    if (ti >= p.n_term || h >= p.nh) return;
    int nh = p.nh; int np = p.np; int traverser = p.traverser;
    uint node_id = term_nodes[ti];
    uint16_t fold_mask = folded_masks[node_id];
    FlatNode node = nodes[node_id];

    // payoff — identical to vcfr_np4_lone_main.
    int c_t = contributions[node_id * uint(np) + uint(traverser)];
    float traverser_stake = float(p.starting_pot) / float(np) + float(c_t);
    bool traverser_folded = (fold_mask & (uint16_t)(1u << traverser)) != 0;
    bool flop_seen = (node.board_state != 3);
    float eff_rake_rate = flop_seen ? p.rake_rate : 0.0f;
    float eff_rake_cap  = flop_seen ? p.rake_cap  : 0.0f;
    int total_pot = p.starting_pot;
    for (int q = 0; q < np; q++) total_pot += contributions[node_id * uint(np) + uint(q)];
    int min_contrib = contributions[node_id * uint(np) + 0u];
    for (int q = 1; q < np; q++) { int c = contributions[node_id * uint(np) + uint(q)]; if (c < min_contrib) min_contrib = c; }
    int num_main = 0;
    for (int q = 0; q < np; q++) if (contributions[node_id * uint(np) + uint(q)] >= min_contrib) num_main++;
    int main_pot_amount = min_contrib * num_main + p.starting_pot;
    float rake = fmax(0.0f, fmin(float(main_pot_amount) * eff_rake_rate, eff_rake_cap));
    float payoff = traverser_folded ? (-traverser_stake) : ((float(total_pot) - rake) - traverser_stake);

    Np4cfCtx x;
    x.t = tab + size_t(ti) * NP4CF_STRIDE;
    x.ra = np4_opp_reach(reach, node_id, np, nh, traverser, p.inner_role_base + 0);
    x.rb = np4_opp_reach(reach, node_id, np, nh, traverser, p.inner_role_base + 1);
    x.rc = np4_opp_reach(reach, node_id, np, nh, traverser, p.inner_role_base + 2);
    x.p2h = p2h;
    x.h1 = hand_cards[h * 2];
    x.h2 = hand_cards[h * 2 + 1];
    x.hp = p2h[x.h1 * 52 + x.h2];
    x.wa = (x.hp >= 0) ? x.ra[x.hp] : 0.0f;

    // add-back scalar features of the h-pair hand (computed once).
    float hp_sk[13];
    if (x.wa != 0.0f) np4cf_sk_feats(x.t, x.rb, x.rc, x.hp, x.h1, x.h2, hp_sk);
    else for (int k = 0; k < 13; k++) hp_sk[k] = 0.0f;

    const device float* t = x.t;
    int h1 = x.h1, h2 = x.h2;
    int xs[2] = { h1, h2 };
    float pbhh = np4cf_pr(x.rb, p2h, h1, h2);
    float pchh = np4cf_pr(x.rc, p2h, h1, h2);

    float n_r = np4cf_rsk(x, SKN, hp_sk);

    // ── S_part ──
    float ab = t[NP4CF_GS + 0] - t[NP4CF_PB + h1] - t[NP4CF_PB + h2] + pbhh;
    float ac = t[NP4CF_GS + 1] - t[NP4CF_PC + h1] - t[NP4CF_PC + h2] + pchh;
    float sb_sum = np4cf_rsk(x, SKQB, hp_sk) - np4cf_rsk(x, SKUB, hp_sk);
    float sc_sum = np4cf_rsk(x, SKQC, hp_sk) - np4cf_rsk(x, SKUC, hp_sk);
    for (int i = 0; i < 2; i++) {
        sb_sum += np4cf_rvk(x, VKYB, xs[i]);
        sc_sum += np4cf_rvk(x, VKYC, xs[i]);
    }
    float cross = np4cf_rsk(x, SKQBQC, hp_sk) - np4cf_rsk(x, SKQBUC, hp_sk)
                - np4cf_rsk(x, SKUBQC, hp_sk) + np4cf_rsk(x, SKUBUC, hp_sk);
    for (int i = 0; i < 2; i++) {
        cross += np4cf_rvk(x, VKQBYC, xs[i]);
        cross -= np4cf_rvk(x, VKUBYC, xs[i]);
        cross += np4cf_rvk(x, VKYBQC, xs[i]);
        cross -= np4cf_rvk(x, VKYBUC, xs[i]);
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            cross += np4cf_rmk(x, MKTYY, xs[i], xs[j]);
    float s_part = ab * ac * n_r + ab * sc_sum + ac * sb_sum + cross;

    // ── B_part ──
    float tbcx = t[NP4CF_GS + 3] - t[NP4CF_BCC + h1] - t[NP4CF_BCC + h2] + pbhh * pchh;
    float b_part = tbcx * n_r + np4cf_rsk(x, SKQBQC, hp_sk) - np4cf_rsk(x, SKVBC, hp_sk);
    for (int i = 0; i < 2; i++) b_part += np4cf_rvk(x, VKWBC, xs[i]);

    // ── D_part ──
    // ΦBX(c) = Σ_{e∈X} pB(c,e): for c=h1 → pB(h1,h2); c=h2 → pB(h2,h1).
    float phib[2] = { pbhh, pbhh };
    float phic[2] = { pchh, pchh };
    float dx = t[NP4CF_GS + 2]; // G0
    for (int i = 0; i < 2; i++) dx -= t[NP4CF_PB + xs[i]] * t[NP4CF_PC + xs[i]];
    for (int i = 0; i < 2; i++) dx -= t[NP4CF_G1 + xs[i]];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            dx += t[NP4CF_PB + xs[j]] * np4cf_pr(x.rc, p2h, xs[j], xs[i])
                + t[NP4CF_PC + xs[j]] * np4cf_pr(x.rb, p2h, xs[j], xs[i]);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            dx += t[NP4CF_G2 + xs[i] * 52 + xs[j]];
    for (int i = 0; i < 2; i++) dx -= phib[i] * phic[i];

    float d_part = dx * n_r;
    d_part -= np4cf_rsk(x, SKDP, hp_sk);
    d_part -= np4cf_rsk(x, SKG1, hp_sk);
    for (int i = 0; i < 2; i++) d_part += np4cf_rvk(x, VKKAP, xs[i]);
    for (int i = 0; i < 2; i++) {
        d_part += t[NP4CF_PB + xs[i]] * np4cf_rvk(x, VKYC, xs[i])
                + t[NP4CF_PC + xs[i]] * np4cf_rvk(x, VKYB, xs[i]);
    }
    d_part += np4cf_rsk(x, SKUBQC, hp_sk) + np4cf_rsk(x, SKQBUC, hp_sk);
    for (int i = 0; i < 2; i++) {
        d_part += np4cf_rvk(x, VKROW, xs[i]);
        d_part += np4cf_rvk(x, VKCOL, xs[i]);
    }
    d_part += np4cf_rsk(x, SKG2G, hp_sk);
    for (int i = 0; i < 2; i++) {
        d_part -= phib[i] * np4cf_rvk(x, VKYC, xs[i]);
        d_part -= phic[i] * np4cf_rvk(x, VKYB, xs[i]);
        d_part -= np4cf_rmk(x, MKTYY, xs[i], xs[i]);
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            d_part -= np4cf_rmk(x, MKZSAME, xs[i], xs[j]);
    for (int i = 0; i < 2; i++) {
        d_part -= np4cf_rvk(x, VKYBQC, xs[i]);
        d_part -= np4cf_rvk(x, VKQBYC, xs[i]);
    }
    d_part -= 2.0f * np4cf_rsk(x, SKQBQC, hp_sk);

    float mass = s_part + b_part - d_part;
    float v = payoff * mass;
    cfv[node_id * uint(nh) + uint(h)] = (p.num_combinations > 0.0f) ? (v / p.num_combinations) : v;
}

// ============================================================================
// np=5 (K=4 opponents) lone-survivor terminal:
//   mass4(h) = Σ_{g0⊥h} r0[g0] · mass3_{r1,r2,r3}(h ∪ g0)
// Inner = the CLOSED-FORM mass3 with a GENERAL 4-card exclusion X (validated
// 2.1e-14 in Rust `mod cf`; |X|=4 case gated there). Aggregates built by the
// np4cf prep kernels with inner_role_base=1 (roles = opponents 1,2,3); the
// outer opponent 0 is enumerated (mc_samples=0) or MC-sampled via a CDF
// (validated unbiased, CV 7.1% @ M=128). Math gate: k4_mass4_* tests.
// ============================================================================

// single-k scalar feature of hand g=(a,b) — mirrors cf::sk_of.
static inline float np5_sk_of(
    const device float* t, const device float* rb, const device float* rc,
    const device int32_t* p2h, int g, int a, int b, int k
) {
    switch (k) {
        case SKN: return 1.0f;
        case SKQB: return rb[g];
        case SKQC: return rc[g];
        case SKUB: return t[NP4CF_PB + a] + t[NP4CF_PB + b];
        case SKUC: return t[NP4CF_PC + a] + t[NP4CF_PC + b];
        case SKDP: return t[NP4CF_PB + a] * t[NP4CF_PC + a] + t[NP4CF_PB + b] * t[NP4CF_PC + b];
        case SKG1: return t[NP4CF_G1 + a] + t[NP4CF_G1 + b];
        case SKG2G: return t[NP4CF_G2 + a * 52 + a] + t[NP4CF_G2 + a * 52 + b]
                         + t[NP4CF_G2 + b * 52 + a] + t[NP4CF_G2 + b * 52 + b];
        case SKVBC: return t[NP4CF_BCC + a] + t[NP4CF_BCC + b];
        case SKQBQC: return rb[g] * rc[g];
        case SKQBUC: return rb[g] * (t[NP4CF_PC + a] + t[NP4CF_PC + b]);
        case SKUBQC: return (t[NP4CF_PB + a] + t[NP4CF_PB + b]) * rc[g];
        default: return (t[NP4CF_PB + a] + t[NP4CF_PB + b]) * (t[NP4CF_PC + a] + t[NP4CF_PC + b]); // SKUBUC
    }
    (void)p2h;
}

// single-k vector feature at e — mirrors cf::vk_of.
static inline float np5_vk_of(
    const device float* t, const device float* rb, const device float* rc,
    const device int32_t* p2h, int g, int a, int b, int k, int e
) {
    float pba = np4cf_pr(rb, p2h, a, e), pbb = np4cf_pr(rb, p2h, b, e);
    float pca = np4cf_pr(rc, p2h, a, e), pcb = np4cf_pr(rc, p2h, b, e);
    switch (k) {
        case VKYB: return pba + pbb;
        case VKYC: return pca + pcb;
        case VKWBC: return pba * pca + pbb * pcb;
        case VKKAP: return t[NP4CF_PB + a] * pca + t[NP4CF_PC + a] * pba
                        + t[NP4CF_PB + b] * pcb + t[NP4CF_PC + b] * pbb;
        case VKROW: return t[NP4CF_G2 + e * 52 + a] + t[NP4CF_G2 + e * 52 + b];
        case VKCOL: return t[NP4CF_G2 + a * 52 + e] + t[NP4CF_G2 + b * 52 + e];
        case VKQBYC: return rb[g] * (pca + pcb);
        case VKYBQC: return (pba + pbb) * rc[g];
        case VKUBYC: return (t[NP4CF_PB + a] + t[NP4CF_PB + b]) * (pca + pcb);
        default: return (pba + pbb) * (t[NP4CF_PC + a] + t[NP4CF_PC + b]); // VKYBUC
    }
}

static inline float np5_mk_of(
    const device float* rb, const device float* rc, const device int32_t* p2h,
    int a, int b, int k, int e, int ep
) {
    float pba = np4cf_pr(rb, p2h, a, e), pbb = np4cf_pr(rb, p2h, b, e);
    float pca = np4cf_pr(rc, p2h, a, ep), pcb = np4cf_pr(rc, p2h, b, ep);
    return (k == MKTYY) ? (pba + pbb) * (pca + pcb) : (pba * pca + pbb * pcb);
}

struct Np5Ctx {
    const device float* t;
    const device float* ra;   // inner aggregate weight = opponent 1
    const device float* rb;   // opponent 2
    const device float* rc;   // opponent 3
    const device int32_t* p2h;
    const device uint8_t* hand_cards;
    int x[4];
};

// Σ_{g⊥X} rA·feature — total − 4 rows + 6 pair add-backs. Mirrors cf::rsk/rvk/rmk.
static inline float np5_rsk(thread const Np5Ctx& c, int k) {
    float v = c.t[NP4CF_SKT + k];
    for (int i = 0; i < 4; i++) v -= c.t[NP4CF_SKR + k * 52 + c.x[i]];
    for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) {
        int idx = c.p2h[c.x[i] * 52 + c.x[j]];
        if (idx >= 0 && c.ra[idx] != 0.0f)
            v += c.ra[idx] * np5_sk_of(c.t, c.rb, c.rc, c.p2h, idx, c.x[i], c.x[j], k);
    }
    return v;
}
static inline float np5_rvk(thread const Np5Ctx& c, int k, int e) {
    float v = c.t[NP4CF_VKT + k * 52 + e];
    for (int i = 0; i < 4; i++) v -= c.t[NP4CF_VKR + k * 2704 + c.x[i] * 52 + e];
    for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) {
        int idx = c.p2h[c.x[i] * 52 + c.x[j]];
        if (idx >= 0 && c.ra[idx] != 0.0f)
            v += c.ra[idx] * np5_vk_of(c.t, c.rb, c.rc, c.p2h, idx, c.x[i], c.x[j], k, e);
    }
    return v;
}
static inline float np5_rmk(thread const Np5Ctx& c, int k, int e, int ep) {
    float v = c.t[NP4CF_MKT + k * 2704 + e * 52 + ep];
    for (int i = 0; i < 4; i++) v -= c.t[NP4CF_MKR + k * 140608 + c.x[i] * 2704 + e * 52 + ep];
    for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) {
        int idx = c.p2h[c.x[i] * 52 + c.x[j]];
        if (idx >= 0 && c.ra[idx] != 0.0f)
            v += c.ra[idx] * np5_mk_of(c.rb, c.rc, c.p2h, c.x[i], c.x[j], k, e, ep);
    }
    return v;
}

// mass3 with |X|=4 — literal port of cf::{s_part, b_part, d_part}.
static inline float np5_mass3_closed(thread const Np5Ctx& c) {
    const device float* t = c.t;
    // ── s_part ──
    float ab = t[NP4CF_GS + 0], ac = t[NP4CF_GS + 1];
    for (int i = 0; i < 4; i++) { ab -= t[NP4CF_PB + c.x[i]]; ac -= t[NP4CF_PC + c.x[i]]; }
    for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) {
        ab += np4cf_pr(c.rb, c.p2h, c.x[i], c.x[j]);
        ac += np4cf_pr(c.rc, c.p2h, c.x[i], c.x[j]);
    }
    float n = np5_rsk(c, SKN);
    float sb_sum = np5_rsk(c, SKQB) - np5_rsk(c, SKUB);
    float sc_sum = np5_rsk(c, SKQC) - np5_rsk(c, SKUC);
    for (int i = 0; i < 4; i++) {
        sb_sum += np5_rvk(c, VKYB, c.x[i]);
        sc_sum += np5_rvk(c, VKYC, c.x[i]);
    }
    float cross = np5_rsk(c, SKQBQC) - np5_rsk(c, SKQBUC) - np5_rsk(c, SKUBQC) + np5_rsk(c, SKUBUC);
    for (int i = 0; i < 4; i++) {
        cross += np5_rvk(c, VKQBYC, c.x[i]);
        cross -= np5_rvk(c, VKUBYC, c.x[i]);
        cross += np5_rvk(c, VKYBQC, c.x[i]);
        cross -= np5_rvk(c, VKYBUC, c.x[i]);
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
        cross += np5_rmk(c, MKTYY, c.x[i], c.x[j]);
    float s_part = ab * ac * n + ab * sc_sum + ac * sb_sum + cross;

    // ── b_part ──
    float tbcx = t[NP4CF_GS + 3];
    for (int i = 0; i < 4; i++) tbcx -= t[NP4CF_BCC + c.x[i]];
    for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) {
        int idx = c.p2h[c.x[i] * 52 + c.x[j]];
        if (idx >= 0) tbcx += c.rb[idx] * c.rc[idx];
    }
    float b_part = tbcx * n + np5_rsk(c, SKQBQC) - np5_rsk(c, SKVBC);
    for (int i = 0; i < 4; i++) b_part += np5_rvk(c, VKWBC, c.x[i]);

    // ── d_part ──
    float phib[4], phic[4];
    for (int i = 0; i < 4; i++) {
        float pb = 0.0f, pc = 0.0f;
        for (int j = 0; j < 4; j++) {
            pb += np4cf_pr(c.rb, c.p2h, c.x[i], c.x[j]);
            pc += np4cf_pr(c.rc, c.p2h, c.x[i], c.x[j]);
        }
        phib[i] = pb; phic[i] = pc;
    }
    float dx = t[NP4CF_GS + 2];
    for (int i = 0; i < 4; i++) dx -= t[NP4CF_PB + c.x[i]] * t[NP4CF_PC + c.x[i]];
    for (int i = 0; i < 4; i++) dx -= t[NP4CF_G1 + c.x[i]];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
        dx += t[NP4CF_PB + c.x[j]] * np4cf_pr(c.rc, c.p2h, c.x[j], c.x[i])
            + t[NP4CF_PC + c.x[j]] * np4cf_pr(c.rb, c.p2h, c.x[j], c.x[i]);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
        dx += t[NP4CF_G2 + c.x[i] * 52 + c.x[j]];
    for (int i = 0; i < 4; i++) dx -= phib[i] * phic[i];

    float d_part = dx * n;
    d_part -= np5_rsk(c, SKDP);
    d_part -= np5_rsk(c, SKG1);
    for (int i = 0; i < 4; i++) d_part += np5_rvk(c, VKKAP, c.x[i]);
    for (int i = 0; i < 4; i++)
        d_part += t[NP4CF_PB + c.x[i]] * np5_rvk(c, VKYC, c.x[i])
                + t[NP4CF_PC + c.x[i]] * np5_rvk(c, VKYB, c.x[i]);
    d_part += np5_rsk(c, SKUBQC) + np5_rsk(c, SKQBUC);
    for (int i = 0; i < 4; i++) {
        d_part += np5_rvk(c, VKROW, c.x[i]);
        d_part += np5_rvk(c, VKCOL, c.x[i]);
    }
    d_part += np5_rsk(c, SKG2G);
    for (int i = 0; i < 4; i++) {
        d_part -= phib[i] * np5_rvk(c, VKYC, c.x[i]);
        d_part -= phic[i] * np5_rvk(c, VKYB, c.x[i]);
        d_part -= np5_rmk(c, MKTYY, c.x[i], c.x[i]);
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
        d_part -= np5_rmk(c, MKZSAME, c.x[i], c.x[j]);
    for (int i = 0; i < 4; i++) {
        d_part -= np5_rvk(c, VKYBQC, c.x[i]);
        d_part -= np5_rvk(c, VKQBYC, c.x[i]);
    }
    d_part -= 2.0f * np5_rsk(c, SKQBQC);

    return s_part + b_part - d_part;
}

// ── CDF prep for the MC outer: thread per terminal, prefix-sum r0. ──
kernel void vcfr_np5_cdf(
    device float*          cdf            [[buffer(0)]],  // [n_term * nh]
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const float*    reach          [[buffer(2)]],
    constant LoneTermParams& p            [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int ti = int(gid);
    if (ti >= p.n_term) return;
    const device float* r0 = np4_opp_reach(reach, term_nodes[ti], p.np, p.nh, p.traverser, 0);
    device float* out = cdf + size_t(ti) * p.nh;
    float s = 0.0f;
    for (int g = 0; g < p.nh; g++) { s += r0[g]; out[g] = s; }
}

// splitmix-ish per-(iter, terminal, sample) uniform in [0,1).
static inline float np5_rand(uint32_t seed, uint ti, uint s) {
    uint32_t z = seed ^ (ti * 0x9E3779B9u) ^ (s * 0x85EBCA6Bu);
    z ^= z >> 16; z *= 0x7FEB352Du; z ^= z >> 15; z *= 0x846CA68Bu; z ^= z >> 16;
    return float(z >> 8) * (1.0f / 16777216.0f);
}

// ── main: cfv[node][h] = payoff · mass4(h) / nc. Grid (n_term, nh). ──
kernel void vcfr_np5_lone_main(
    device float*          cfv            [[buffer(0)]],
    device const uint32_t* term_nodes     [[buffer(1)]],
    device const FlatNode* nodes          [[buffer(2)]],
    device const int32_t*  contributions  [[buffer(3)]],
    device const uint16_t* folded_masks   [[buffer(4)]],
    device const float*    reach          [[buffer(5)]],
    device const uint8_t*  hand_cards     [[buffer(6)]],
    device const int32_t*  p2h            [[buffer(7)]],
    device const float*    tab            [[buffer(8)]],
    device const float*    cdf            [[buffer(9)]],
    constant LoneTermParams& p            [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ti = int(gid.x);
    int h  = int(gid.y);
    if (ti >= p.n_term || h >= p.nh) return;
    int nh = p.nh; int np = p.np; int traverser = p.traverser;
    uint node_id = term_nodes[ti];
    uint16_t fold_mask = folded_masks[node_id];
    FlatNode node = nodes[node_id];

    // payoff — identical conventions to the np3/np4 lone kernels.
    int c_t = contributions[node_id * uint(np) + uint(traverser)];
    float traverser_stake = float(p.starting_pot) / float(np) + float(c_t);
    bool traverser_folded = (fold_mask & (uint16_t)(1u << traverser)) != 0;
    bool flop_seen = (node.board_state != 3);
    float eff_rake_rate = flop_seen ? p.rake_rate : 0.0f;
    float eff_rake_cap  = flop_seen ? p.rake_cap  : 0.0f;
    int total_pot = p.starting_pot;
    for (int q = 0; q < np; q++) total_pot += contributions[node_id * uint(np) + uint(q)];
    int min_contrib = contributions[node_id * uint(np) + 0u];
    for (int q = 1; q < np; q++) { int cq = contributions[node_id * uint(np) + uint(q)]; if (cq < min_contrib) min_contrib = cq; }
    int num_main = 0;
    for (int q = 0; q < np; q++) if (contributions[node_id * uint(np) + uint(q)] >= min_contrib) num_main++;
    int main_pot_amount = min_contrib * num_main + p.starting_pot;
    float rake = fmax(0.0f, fmin(float(main_pot_amount) * eff_rake_rate, eff_rake_cap));
    float payoff = traverser_folded ? (-traverser_stake) : ((float(total_pot) - rake) - traverser_stake);

    Np5Ctx c;
    c.t = tab + size_t(ti) * NP4CF_STRIDE;
    // outer opponent 0; inner aggregate roles 1,2,3 (inner_role_base=1).
    const device float* r0 = np4_opp_reach(reach, node_id, np, nh, traverser, 0);
    c.ra = np4_opp_reach(reach, node_id, np, nh, traverser, 1);
    c.rb = np4_opp_reach(reach, node_id, np, nh, traverser, 2);
    c.rc = np4_opp_reach(reach, node_id, np, nh, traverser, 3);
    c.p2h = p2h;
    c.hand_cards = hand_cards;
    int h1 = hand_cards[h * 2], h2 = hand_cards[h * 2 + 1];
    c.x[0] = h1; c.x[1] = h2;

    float mass = 0.0f;
    if (p.mc_samples <= 0) {
        // FULL outer enumeration (the parity mode).
        for (int g0 = 0; g0 < nh; g0++) {
            float w = r0[g0];
            if (w == 0.0f) continue;
            int a = hand_cards[g0 * 2], b = hand_cards[g0 * 2 + 1];
            if (a == h1 || a == h2 || b == h1 || b == h2) continue;
            c.x[2] = a; c.x[3] = b;
            mass += w * np5_mass3_closed(c);
        }
    } else {
        // MC outer: M CDF draws ∝ r0, SHARED across h (same seed per terminal).
        const device float* tcdf = cdf + size_t(ti) * nh;
        float w_total = tcdf[nh - 1];
        if (w_total > 0.0f) {
            float acc = 0.0f;
            for (int s = 0; s < p.mc_samples; s++) {
                float u = np5_rand(p.mc_seed, ti, uint(s)) * w_total;
                // binary search first index with cdf >= u
                int lo = 0, hi = nh - 1;
                while (lo < hi) { int mid = (lo + hi) >> 1; if (tcdf[mid] < u) lo = mid + 1; else hi = mid; }
                int g0 = lo;
                int a = hand_cards[g0 * 2], b = hand_cards[g0 * 2 + 1];
                if (a == h1 || a == h2 || b == h1 || b == h2) continue;
                c.x[2] = a; c.x[3] = b;
                acc += np5_mass3_closed(c);
            }
            mass = acc * w_total / float(p.mc_samples);
        }
    }

    float v = payoff * mass;
    cfv[node_id * uint(nh) + uint(h)] = (p.num_combinations > 0.0f) ? (v / p.num_combinations) : v;
}
