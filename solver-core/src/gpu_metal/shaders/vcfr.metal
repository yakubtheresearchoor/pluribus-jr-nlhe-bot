// Core CFR kernels ported from vcfr.cu to Metal Shading Language.
// Phase 1-2: All kernels needed for vector CFR on river/turn-start games.

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
    thread float* out                              // [nh]
) {
    int num_opp = np - 1;
    int32_t c_t = contributions[traverser];

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
                            payoff_unit = -1.0f;
                        } else if (max_opp == h_str) {
                            uint t = 1;
                            if (s0 == h_str) t++;
                            if (s1 == h_str) t++;
                            payoff_unit = (k + 1.0f - (float)t) / (float)t;
                        } else {
                            payoff_unit = k;
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
        float payoff;
        if (traverser_folded) {
            payoff = -traverser_stake;
        } else {
            payoff = (float)total_pot - traverser_stake;
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
                            if (pc == 0) { prev_l = lev; continue; }
                            int num_contrib = 0;
                            for (int p = 0; p < np; p++) {
                                if (contributions[p] >= lev) num_contrib++;
                            }
                            float pot_l = float(pc * num_contrib);
                            if (li == 0) pot_l += float(starting_pot);

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
                                cash += pot_l / float(tied);
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
        int opp_a = (traverser == 0) ? 1 : 0;
        thread const float* reach_a = opp_reach_local + 0 * nh;
        int32_t c_opp_a = contributions[opp_a];
        bool a_folded = (fold_mask & (uint16_t)(1u << opp_a)) != 0;

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

                float net;
                if (traverser_folded) {
                    net = -traverser_stake;
                } else {
                    float cash = 0.0f;
                    int prev_l = 0;
                    for (int li = 0; li < num_levels; li++) {
                        int lev = levels[li];
                        int pc = lev - prev_l;
                        if (pc == 0) { prev_l = lev; continue; }
                        int num_contrib = 0;
                        for (int p = 0; p < np; p++) {
                            if (contributions[p] >= lev) num_contrib++;
                        }
                        float pot_l = float(pc * num_contrib);
                        if (li == 0) pot_l += float(starting_pot);
                        bool trav_elig = c_t >= lev;
                        bool a_elig = !a_folded && c_opp_a >= lev;
                        if (!trav_elig) { prev_l = lev; continue; }
                        if (!a_elig) {
                            cash += pot_l;
                        } else {
                            uint16_t max_str = (h_str > s_a) ? h_str : s_a;
                            int tied = 0;
                            if (h_str == max_str) tied++;
                            if (s_a == max_str) tied++;
                            if (h_str == max_str) cash += pot_l / float(tied);
                        }
                        prev_l = lev;
                    }
                    net = cash - traverser_stake;
                }
                accum += ra * net;
            }
            out[h] = accum;
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
            if (pc == 0) { prev_l = lev; continue; }

            int num_contrib = 0;
            for (int p = 0; p < np; p++) {
                if (contributions[p] >= lev) num_contrib++;
            }
            float pot_l = (float)(pc * num_contrib);
            if (li == 0) pot_l += (float)starting_pot;

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
                static_cash += pot_l;
            } else if (!has_active_elig && !trav_elig) {
                if (contributions[traverser] >= lev) {
                    float trav_contrib = (float)pc;
                    if (li == 0) trav_contrib += (float)starting_pot / (float)np;
                    static_cash += trav_contrib;
                }
            } else if (!trav_elig) {
                // Case D: traverser ineligible at contested level — no cash.
            } else {
                float share = factored_share_for_level_thread(
                    num_opp, elig_opps, 0u, h_m, h_str,
                    opp_reach_local, hand_cards, hand_strength, nh);
                case_c += pot_l * share;
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
    // ─── Slice 2 rake (CPU↔Metal parity) ───
    // Mirror CPU `tree.rake_rate, tree.rake_cap`. Per-terminal gating is
    // applied at evaluation site as: `eff_rate/eff_cap = flop_seen ? rake : 0`.
    float rake_rate;
    float rake_cap;
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
        // side_pot_showdown_cfv). Returns raw CFV; we apply /num_combinations
        // below as the caller normalization.
        float local_out[1326];
        multiway_brute_force_showdown(
            nh, np, int(params.traverser),
            params.starting_pot, fold_mask,
            contribs_local, opp_reach_local,
            hand_cards, sorted_pl_strength, sorted_pl_indices,
            local_out
        );
        for (int h = 0; h < nh; h++) out[h] = local_out[h];

        if (params.num_combinations > 0.0f) {
            for (int h = 0; h < nh; h++) out[h] /= params.num_combinations;
        }
        return;
    }


    // ═══ CHANCE NODE ═══
    if (node.node_type == NODE_TYPE_CHANCE) {
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
// vcfr_streaming_level
//
// Processes all outcomes for one level SEQUENTIALLY within each thread.
// Grid = level_count, blockDim = 1.
// Each thread handles one node, looping over all outcomes.
// At traverser player nodes, applies per-outcome DCFR discount:
//   regret = coef * regret + cp * inst_regret
// cum_strategy is accumulated into cum_accum with cp weighting.
// ============================================================================

struct StreamingParams {
    int level_count;
    int num_outcomes;
    int cfv_batch_stride;
    int sorted_opp_stride;
    int num_players;
    int nh;
    uint32_t traverser;
    float alpha_t;
    float beta_t;
    float gamma_t;
    float regret_floor;
    int32_t starting_pot;
    float num_combinations;
};

kernel void vcfr_streaming_level(
    device const uint32_t* level_nodes       [[buffer(0)]],
    constant StreamingParams& params         [[buffer(1)]],
    device const FlatNode* nodes             [[buffer(2)]],
    device const uint32_t* children          [[buffer(3)]],
    device const int32_t* contributions      [[buffer(4)]],
    device const uint16_t* folded_masks      [[buffer(5)]],
    device const float* strategy             [[buffer(6)]],
    device const uint32_t* infoset_offsets   [[buffer(7)]],
    device const float* reach                [[buffer(8)]],
    device float* cfv                        [[buffer(9)]],
    device float* regrets                    [[buffer(10)]],
    device float* cum_accum                  [[buffer(11)]],
    device const float* initial_weight       [[buffer(12)]],
    device const uint16_t* sorted_opp_strength  [[buffer(13)]],
    device const uint16_t* sorted_opp_indices    [[buffer(14)]],
    device const uint16_t* sorted_pl_strength   [[buffer(15)]],
    device const uint16_t* sorted_pl_indices     [[buffer(16)]],
    device const uint8_t* hand_cards          [[buffer(17)]],
    device const float* chance_prob           [[buffer(18)]],
    uint gid [[thread_position_in_grid]]
) {
    int node_in_level = int(gid);
    if (node_in_level >= params.level_count) return;

    uint node_id = level_nodes[node_in_level];
    FlatNode node = nodes[node_id];
    int np = params.num_players;
    int nh = params.nh;
    int num_opp = np - 1;

    // ═══ TERMINAL NODE ═══
    if (node.node_type == NODE_TYPE_TERMINAL) {
        int node_reach_base = node_id * np * nh;
        int32_t c_t = contributions[node_id * np + params.traverser];
        uint16_t fold_mask = folded_masks[node_id];

        int num_active = 0;
        for (int p = 0; p < np; p++) {
            if (!(fold_mask & (1 << p))) num_active++;
        }

        for (int outcome = 0; outcome < params.num_outcomes; outcome++) {
            int sos_off = outcome * params.sorted_opp_stride;
            int sps_off = outcome * params.sorted_opp_stride; // same layout as opp: num_opp * nh per outcome
            const device uint16_t* opp_str = sorted_opp_strength + sos_off;
            const device uint16_t* opp_idx = sorted_opp_indices + sos_off;
            const device uint16_t* pl_str = sorted_pl_strength + sps_off;
            const device uint16_t* pl_idx = sorted_pl_indices + sps_off;
            device float* out = cfv + outcome * params.cfv_batch_stride + node_id * nh;

            if (num_active <= 1 || (fold_mask & (1 << params.traverser))) {
                int32_t total_pot = params.starting_pot;
                for (int p = 0; p < np; p++) total_pot += contributions[node_id * np + p];
                float traverser_investment = (float)params.starting_pot / (float)np + (float)c_t;
                float payoff;
                if (fold_mask & (1 << params.traverser)) {
                    payoff = -traverser_investment;
                } else {
                    payoff = (float)total_pot - traverser_investment;
                }

                float opp_reach_sum = 0.0f;
                float opp_reach_minus[52];
                for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;
                for (int oi = 0; oi < num_opp; oi++) {
                    int opp = (oi < (int)params.traverser) ? oi : (oi + 1);
                    const device float* opp_r = reach + node_reach_base + opp * nh;
                    for (int ho = 0; ho < nh; ho++) {
                        float r = opp_r[ho];
                        if (r != 0.0f) {
                            opp_reach_sum += r;
                            opp_reach_minus[hand_cards[ho * 2]] += r;
                            opp_reach_minus[hand_cards[ho * 2 + 1]] += r;
                        }
                    }
                }
                if (opp_reach_sum > 0.0f) {
                    for (int h = 0; h < nh; h++) {
                        float cfreach = opp_reach_sum
                            - opp_reach_minus[hand_cards[h * 2]]
                            - opp_reach_minus[hand_cards[h * 2 + 1]];
                        out[h] = payoff * cfreach;
                    }
                } else {
                    for (int h = 0; h < nh; h++) out[h] = 0.0f;
                }
                if (params.num_combinations > 0.0f) { for (int h = 0; h < nh; h++) out[h] /= params.num_combinations; }
                continue;
            }

            bool all_equal = true;
            for (int p = 0; p < np; p++) {
                if (fold_mask & (1 << p)) continue;
                if (contributions[node_id * np + p] != c_t) { all_equal = false; break; }
            }

            if (all_equal) {
                int num_active_opp = 0;
                for (int p = 0; p < np; p++) {
                    if (p == (int)params.traverser) continue;
                    if (!(fold_mask & (1 << p))) num_active_opp++;
                }

                if (num_active_opp == 0) {
                    int32_t total_pot = params.starting_pot;
                    for (int p = 0; p < np; p++) total_pot += contributions[node_id * np + p];
                    float traverser_investment = (float)params.starting_pot / (float)np + (float)c_t;
                    float payoff = (float)total_pot - traverser_investment;
                    float opp_reach_sum = 0.0f;
                    float opp_reach_minus[52];
                    for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;
                    for (int oi = 0; oi < num_opp; oi++) {
                        int opp = (oi < (int)params.traverser) ? oi : (oi + 1);
                        const device float* opp_r = reach + node_reach_base + opp * nh;
                        for (int ho = 0; ho < nh; ho++) {
                            float r = opp_r[ho];
                            if (r != 0.0f) {
                                opp_reach_sum += r;
                                opp_reach_minus[hand_cards[ho * 2]] += r;
                                opp_reach_minus[hand_cards[ho * 2 + 1]] += r;
                            }
                        }
                    }
                    if (opp_reach_sum > 0.0f) {
                        for (int h = 0; h < nh; h++) {
                            float cfreach = opp_reach_sum - opp_reach_minus[hand_cards[h * 2]] - opp_reach_minus[hand_cards[h * 2 + 1]];
                            out[h] = payoff * cfreach;
                        }
                    } else {
                        for (int h = 0; h < nh; h++) out[h] = 0.0f;
                    }
                    if (params.num_combinations > 0.0f) { for (int h = 0; h < nh; h++) out[h] /= params.num_combinations; }
                    continue;
                }

                float opp_reach_local[5 * 1326];
                for (int oi = 0; oi < num_opp; oi++) {
                    int opp = (oi < (int)params.traverser) ? oi : (oi + 1);
                    if (fold_mask & (1 << opp)) {
                        for (int h = 0; h < nh; h++) opp_reach_local[oi * nh + h] = 0.0f;
                    } else {
                        const device float* opp_r = reach + node_reach_base + opp * nh;
                        for (int h = 0; h < nh; h++) opp_reach_local[oi * nh + h] = opp_r[h];
                    }
                }

                if (num_active_opp == 1) {
                    float local_cfv[1326];
                    sorted_sweep_showdown_vcfr_local(
                        opp_reach_local, num_opp, nh,
                        opp_str, opp_idx, pl_str, pl_idx,
                        hand_cards, local_cfv
                    );
                    float pot_size = (float)params.starting_pot / (float)np + (float)c_t;
                    for (int h = 0; h < nh; h++) out[h] = local_cfv[h] * pot_size;
                } else {
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
                                if (r != 0.0f) { cfreach_sum += r; cfreach_minus[hand_cards[ho * 2]] += r; cfreach_minus[hand_cards[ho * 2 + 1]] += r; }
                                i++;
                            }
                            cw[h] = cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]];
                        }
                        while (i < nh) {
                            uint16_t ho = opp_idx[oi * nh + i];
                            float r = opp_r[ho];
                            if (r != 0.0f) { cfreach_sum += r; cfreach_minus[hand_cards[ho * 2]] += r; cfreach_minus[hand_cards[ho * 2 + 1]] += r; }
                            i++;
                        }
                        for (int h = 0; h < nh; h++) {
                            float eff = cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]] + opp_r[h];
                            if (oi == 0) { cum_weaker[h] = cw[h]; eff_total[h] = eff; }
                            else { cum_weaker[h] *= cw[h]; eff_total[h] *= eff; }
                        }
                    }
                    for (int h = 0; h < nh; h++) {
                        out[h] = ((float)params.starting_pot / (float)np + (float)c_t) * ((float)(num_active_opp + 1) * cum_weaker[h] - eff_total[h]);
                    }
                }
            } else {
                // Side pot: product formula per pot level (exact multiway)
                for (int h = 0; h < nh; h++) out[h] = 0.0f;
                int levels[8]; int num_levels = 0;
                for (int p = 0; p < np && num_levels < 8; p++) {
                    int32_t c = contributions[node_id * np + p];
                    bool found = false;
                    for (int l = 0; l < num_levels; l++) { if (levels[l] == c) { found = true; break; } }
                    if (!found) levels[num_levels++] = c;
                }
                for (int i = 0; i < num_levels - 1; i++) for (int j = i + 1; j < num_levels; j++) if (levels[j] < levels[i]) { int tmp = levels[i]; levels[i] = levels[j]; levels[j] = tmp; }
                int prev_level = 0;
                for (int li = 0; li < num_levels; li++) {
                    int level = levels[li];
                    int pot_contribution = level - prev_level;
                    if (pot_contribution == 0) { prev_level = level; continue; }
                    int total_counted = 0; int eligible_opp_count = 0; bool traverser_eligible = false;
                    for (int p = 0; p < np; p++) { if (contributions[node_id * np + p] >= level) { total_counted++; if (!(fold_mask & (1 << p))) { if (p == (int)params.traverser) traverser_eligible = true; else eligible_opp_count++; } } }
                    int pot_at_level = pot_contribution * total_counted;
                    if (li == 0) pot_at_level += params.starting_pot;
                    if (eligible_opp_count == 0) { if (traverser_eligible) { for (int h = 0; h < nh; h++) out[h] += (float)pot_at_level; } prev_level = level; continue; }
                    if (traverser_eligible) {
                        if (eligible_opp_count == 1) {
                            // Single eligible opponent: pairwise sweep (exact)
                            for (int opp_p = 0; opp_p < np; opp_p++) {
                                if (opp_p == (int)params.traverser || (fold_mask & (1 << opp_p)) || contributions[node_id * np + opp_p] < level) continue;
                                int oi = (opp_p < (int)params.traverser) ? opp_p : (opp_p - 1);
                                const device float* opp_r = reach + node_reach_base + opp_p * nh;
                                const device uint16_t* o_str = opp_str + oi * nh;
                                const device uint16_t* o_idx = opp_idx + oi * nh;
                                float cfreach_sum = 0.0f; float cfreach_minus[52]; for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;
                                int i = 0;
                                for (int si = 0; si < nh; si++) { uint16_t str_h = pl_str[si]; uint16_t h = pl_idx[si]; while (i < nh && o_str[i] < str_h) { uint16_t ho = o_idx[i]; float r = opp_r[ho]; if (r != 0.0f) { cfreach_sum += r; cfreach_minus[hand_cards[ho * 2]] += r; cfreach_minus[hand_cards[ho * 2 + 1]] += r; } i++; } float cfreach = cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]]; out[h] += (float)pot_at_level * cfreach; }
                                cfreach_sum = 0.0f; for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f; i = nh - 1;
                                for (int si = nh - 1; si >= 0; si--) { uint16_t str_h = pl_str[si]; uint16_t h = pl_idx[si]; while (i >= 0 && o_str[i] > str_h) { uint16_t ho = o_idx[i]; float r = opp_r[ho]; if (r != 0.0f) { cfreach_sum += r; cfreach_minus[hand_cards[ho * 2]] += r; cfreach_minus[hand_cards[ho * 2 + 1]] += r; } i--; } float cfreach = cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]]; out[h] -= (float)pot_at_level * cfreach; }
                            }
                        } else {
                            // Multiple eligible opponents: product formula (exact)
                            float cum_weaker[1326];
                            float eff_total[1326];
                            for (int h = 0; h < nh; h++) { cum_weaker[h] = 1.0f; eff_total[h] = 1.0f; }
                            for (int opp_p = 0; opp_p < np; opp_p++) {
                                if (opp_p == (int)params.traverser || (fold_mask & (1 << opp_p)) || contributions[node_id * np + opp_p] < level) continue;
                                int oi = (opp_p < (int)params.traverser) ? opp_p : (opp_p - 1);
                                const device float* opp_r = reach + node_reach_base + opp_p * nh;
                                const device uint16_t* o_str = opp_str + oi * nh;
                                const device uint16_t* o_idx = opp_idx + oi * nh;
                                float cw[1326]; float cfreach_sum = 0.0f; float cfreach_minus[52]; for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f; int i = 0;
                                for (int si = 0; si < nh; si++) { uint16_t str_h = pl_str[si]; uint16_t h = pl_idx[si]; while (i < nh && o_str[i] < str_h) { uint16_t ho = o_idx[i]; float r = opp_r[ho]; if (r != 0.0f) { cfreach_sum += r; cfreach_minus[hand_cards[ho * 2]] += r; cfreach_minus[hand_cards[ho * 2 + 1]] += r; } i++; } cw[h] = cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]]; }
                                while (i < nh) { uint16_t ho = o_idx[i]; float r = opp_r[ho]; if (r != 0.0f) { cfreach_sum += r; cfreach_minus[hand_cards[ho * 2]] += r; cfreach_minus[hand_cards[ho * 2 + 1]] += r; } i++; }
                                for (int h = 0; h < nh; h++) { float eff = cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]] + opp_r[h]; cum_weaker[h] *= cw[h]; eff_total[h] *= eff; }
                            }
                            for (int h = 0; h < nh; h++) { out[h] += float(pot_at_level) * (float(eligible_opp_count + 1) * cum_weaker[h] - eff_total[h]); }
                        }
                    }
                    prev_level = level;
                }
                for (int h = 0; h < nh; h++) out[h] -= ((float)params.starting_pot / (float)np + (float)c_t);
            }
            if (params.num_combinations > 0.0f) { for (int h = 0; h < nh; h++) out[h] /= params.num_combinations; }
        }
        return;
    }

    // ═══ CHANCE NODE ═══
    if (node.node_type == NODE_TYPE_CHANCE) {
        for (int outcome = 0; outcome < params.num_outcomes; outcome++) {
            device float* cfv_o = cfv + outcome * params.cfv_batch_stride;
            device float* out = cfv_o + node_id * nh;
            for (int h = 0; h < nh; h++) out[h] = 0.0f;
            for (int a = 0; a < (int)node.num_children; a++) {
                uint child = children[node.children_start + a];
                for (int h = 0; h < nh; h++) out[h] += cfv_o[child * nh + h];
            }
        }
        return;
    }

    // ═══ PLAYER NODE ═══
    int owner = (int)node.player_id;
    int na = (int)node.num_children;
    uint infoset_id = infoset_offsets[node_id];
    int stride = MAX_NA * nh;
    const device float* sigma = strategy + infoset_id * stride;
    int offset = infoset_id * stride;

    if (owner == (int)params.traverser) {
        // Accumulate weighted instantaneous regrets across ALL outcomes,
        // then apply DCFR discount ONCE at the end.
        // This avoids the compounding discount bug where coef^N would
        // destroy information at early iterations.
        float inst_regret_accum[MAX_NA * 1326]; // [a * nh + h]
        for (int a = 0; a < na; a++)
            for (int h = 0; h < nh; h++)
                inst_regret_accum[a * nh + h] = 0.0f;

        for (int outcome = 0; outcome < params.num_outcomes; outcome++) {
            device float* cfv_o = cfv + outcome * params.cfv_batch_stride;
            int cp_base = outcome * nh;

            // Compute strategy-weighted average CFV
            float cfv_avg[1326];
            for (int h = 0; h < nh; h++) cfv_avg[h] = 0.0f;
            for (int a = 0; a < na; a++) {
                uint child = children[node.children_start + a];
                for (int h = 0; h < nh; h++) cfv_avg[h] += sigma[a * nh + h] * cfv_o[child * nh + h];
            }

            // Accumulate cp-weighted instantaneous regrets
            for (int a = 0; a < na; a++) {
                uint child = children[node.children_start + a];
                for (int h = 0; h < nh; h++) {
                    float cp = chance_prob[cp_base + h];
                    inst_regret_accum[a * nh + h] += cp * (cfv_o[child * nh + h] - cfv_avg[h]);
                }
            }

            // Accumulate cum_strategy with cp weighting
            for (int a = 0; a < na; a++) {
                for (int h = 0; h < nh; h++) {
                    uint cidx = offset + a * nh + h;
                    float cp = chance_prob[cp_base + h];
                    cum_accum[cidx] += cp * sigma[a * nh + h];
                }
            }

            device float* out = cfv_o + node_id * nh;
            for (int h = 0; h < nh; h++) out[h] = cfv_avg[h];
        }

        // Apply DCFR discount ONCE after all outcomes
        for (int a = 0; a < na; a++) {
            for (int h = 0; h < nh; h++) {
                uint ridx = offset + a * nh + h;
                float coef = (regrets[ridx] >= 0.0f) ? params.alpha_t : params.beta_t;
                regrets[ridx] = coef * regrets[ridx] + inst_regret_accum[a * nh + h];
                if (regrets[ridx] < params.regret_floor) regrets[ridx] = params.regret_floor;
            }
        }
    } else {
        // Opponent: sum child CFVs (no regret update)
        for (int outcome = 0; outcome < params.num_outcomes; outcome++) {
            device float* cfv_o = cfv + outcome * params.cfv_batch_stride;
            float cfv_avg[1326];
            for (int h = 0; h < nh; h++) cfv_avg[h] = 0.0f;
            for (int a = 0; a < na; a++) {
                uint child = children[node.children_start + a];
                for (int h = 0; h < nh; h++) cfv_avg[h] += cfv_o[child * nh + h];
            }
            device float* out = cfv_o + node_id * nh;
            for (int h = 0; h < nh; h++) out[h] = cfv_avg[h];
        }
    }
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

        // Run the brute-force showdown helper.
        float local_out[1326];
        multiway_brute_force_showdown(
            nh, np, int(params.traverser),
            params.starting_pot, fold_mask,
            contribs_local, opp_reach_local,
            hand_cards, pl_str, pl_idx,
            local_out
        );
        for (int h = 0; h < nh; h++) out[h] = local_out[h];

        if (params.num_combinations > 0.0f) {
            for (int h = 0; h < nh; h++) out[h] /= params.num_combinations;
        }
        return;
    }

    // (Legacy showdown evaluation code removed; replaced by brute-force above.)
    if (false) {
        int32_t c_t = contributions[int(node_id) * np + int(params.traverser)];
        uint16_t fold_mask = folded_masks[int(node_id)];
        int node_reach_base = int(node_id) * np * nh;
        device float* out = cfv_o + int(node_id) * nh;

        int num_active = 0;
        for (int p = 0; p < np; p++) {
            if (!(fold_mask & (1 << p))) num_active++;
        }

        // Fold win or traverser folded
        if (num_active <= 1 || (fold_mask & (1 << params.traverser))) {
            int32_t total_pot = params.starting_pot;
            for (int p = 0; p < np; p++) total_pot += contributions[int(node_id) * np + p];
            float traverser_investment = float(params.starting_pot) / float(np) + float(c_t);
            float payoff;
            if (fold_mask & (1 << params.traverser)) {
                payoff = -traverser_investment;
            } else {
                payoff = float(total_pot) - traverser_investment;
            }

            float opp_reach_sum = 0.0f;
            float opp_reach_minus[52];
            for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;
            for (int oi = 0; oi < num_opp; oi++) {
                int opp = (oi < int(params.traverser)) ? oi : (oi + 1);
                const device float* opp_r = reach + node_reach_base + opp * nh;
                for (int ho = 0; ho < nh; ho++) {
                    if (chance_prob[ho] == 0.0f) continue;
                    float r = opp_r[ho];
                    if (r != 0.0f) {
                        opp_reach_sum += r;
                        opp_reach_minus[hand_cards[ho * 2]] += r;
                        opp_reach_minus[hand_cards[ho * 2 + 1]] += r;
                    }
                }
            }
            if (opp_reach_sum > 0.0f) {
                for (int h = 0; h < nh; h++) {
                    float cfreach = opp_reach_sum
                        - opp_reach_minus[hand_cards[h * 2]]
                        - opp_reach_minus[hand_cards[h * 2 + 1]];
                    out[h] = payoff * cfreach;
                }
            } else {
                for (int h = 0; h < nh; h++) out[h] = 0.0f;
            }
            if (params.num_combinations > 0.0f) {
                for (int h = 0; h < nh; h++) out[h] /= params.num_combinations;
            }
            return;
        }

        // Showdown: check equal contributions
        bool all_equal = true;
        for (int p = 0; p < np; p++) {
            if (fold_mask & (1 << p)) continue;
            if (contributions[int(node_id) * np + p] != c_t) { all_equal = false; break; }
        }

        if (all_equal) {
            int num_active_opp = 0;
            for (int p = 0; p < np; p++) {
                if (p == int(params.traverser)) continue;
                if (!(fold_mask & (1 << p))) num_active_opp++;
            }

            if (num_active_opp == 0) {
                int32_t total_pot = params.starting_pot;
                for (int p = 0; p < np; p++) total_pot += contributions[int(node_id) * np + p];
                float traverser_investment = float(params.starting_pot) / float(np) + float(c_t);
                float payoff = float(total_pot) - traverser_investment;
                float opp_reach_sum = 0.0f;
                float opp_reach_minus[52];
                for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;
                for (int oi = 0; oi < num_opp; oi++) {
                    int opp = (oi < int(params.traverser)) ? oi : (oi + 1);
                    const device float* opp_r = reach + node_reach_base + opp * nh;
                    for (int ho = 0; ho < nh; ho++) {
                        if (chance_prob[ho] == 0.0f) continue;
                        float r = opp_r[ho];
                        if (r != 0.0f) {
                            opp_reach_sum += r;
                            opp_reach_minus[hand_cards[ho * 2]] += r;
                            opp_reach_minus[hand_cards[ho * 2 + 1]] += r;
                        }
                    }
                }
                if (opp_reach_sum > 0.0f) {
                    for (int h = 0; h < nh; h++) {
                        out[h] = payoff * (opp_reach_sum - opp_reach_minus[hand_cards[h * 2]] - opp_reach_minus[hand_cards[h * 2 + 1]]);
                    }
                } else {
                    for (int h = 0; h < nh; h++) out[h] = 0.0f;
                }
                if (params.num_combinations > 0.0f) {
                    for (int h = 0; h < nh; h++) out[h] /= params.num_combinations;
                }
                return;
            }

            // Build local opp reach (filtered by chance probability for board card blocking)
            float opp_reach_local[5 * 1326];
            for (int oi = 0; oi < num_opp; oi++) {
                int opp = (oi < int(params.traverser)) ? oi : (oi + 1);
                if (fold_mask & (1 << opp)) {
                    for (int h = 0; h < nh; h++) opp_reach_local[oi * nh + h] = 0.0f;
                } else {
                    const device float* opp_r = reach + node_reach_base + opp * nh;
                    for (int h = 0; h < nh; h++) {
                        opp_reach_local[oi * nh + h] = (chance_prob[h] == 0.0f) ? 0.0f : opp_r[h];
                    }
                }
            }

            if (num_active_opp == 1) {
                float local_cfv[1326];
                sorted_sweep_showdown_vcfr_local(
                    opp_reach_local, num_opp, nh,
                    opp_str, opp_idx, pl_str, pl_idx,
                    hand_cards, local_cfv
                );
                float pot_size = float(params.starting_pot) / float(np) + float(c_t);
                for (int h = 0; h < nh; h++) out[h] = local_cfv[h] * pot_size;
            } else {
                // Multiway probabilistic
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
                    out[h] = (float(params.starting_pot) / float(np) + float(c_t))
                        * (float(num_active_opp + 1) * cum_weaker[h] - eff_total[h]);
                }
            }
        } else {
            // Unequal contributions
            if (np == 2) {
                // 2-player: use half_pot * sweep (correct side pot handling)
                int32_t min_active = 0x7fffffff;
                for (int p = 0; p < np; p++) {
                    if (!(fold_mask & (1 << p))) {
                        if (contributions[int(node_id) * np + p] < min_active)
                            min_active = contributions[int(node_id) * np + p];
                    }
                }
                float half_pot = float(params.starting_pot) / float(np) + float(min_active);

                int num_active_opp = 0;
                for (int opp_p = 0; opp_p < np; opp_p++) {
                    if (opp_p == int(params.traverser)) continue;
                    if (!(fold_mask & (1 << opp_p))) num_active_opp++;
                }

                if (num_active_opp == 0) {
                    int32_t total_pot = params.starting_pot;
                    for (int p = 0; p < np; p++) total_pot += contributions[int(node_id) * np + p];
                    float payoff = float(total_pot) - (float(params.starting_pot) / float(np) + float(c_t));
                    for (int h = 0; h < nh; h++) out[h] = payoff;
                } else {
                    float opp_reach_local[5 * 1326];
                    for (int oi = 0; oi < num_opp; oi++) {
                        int opp = (oi < int(params.traverser)) ? oi : (oi + 1);
                        if (fold_mask & (1 << opp)) {
                            for (int h = 0; h < nh; h++) opp_reach_local[oi * nh + h] = 0.0f;
                        } else {
                            const device float* opp_r = reach + node_reach_base + opp * nh;
                            for (int h = 0; h < nh; h++) {
                                opp_reach_local[oi * nh + h] = (chance_prob[h] == 0.0f) ? 0.0f : opp_r[h];
                            }
                        }
                    }
                    float local_cfv[1326];
                    sorted_sweep_showdown_vcfr_local(
                        opp_reach_local, num_opp, nh,
                        opp_str, opp_idx,
                        pl_str, pl_idx,
                        hand_cards, local_cfv
                    );
                    for (int h = 0; h < nh; h++) out[h] = half_pot * local_cfv[h];
                }
            } else {
            // Multiway side pot handling (3+ players)
            // Uses product formula per pot level (exact multiway computation)
            for (int h = 0; h < nh; h++) out[h] = 0.0f;
            int levels[8];
            int num_levels = 0;
            for (int p = 0; p < np && num_levels < 8; p++) {
                int32_t c = contributions[int(node_id) * np + p];
                bool found = false;
                for (int l = 0; l < num_levels; l++) {
                    if (levels[l] == c) { found = true; break; }
                }
                if (!found) levels[num_levels++] = c;
            }
            // Sort levels ascending
            for (int i = 0; i < num_levels - 1; i++) {
                for (int j = i + 1; j < num_levels; j++) {
                    if (levels[j] < levels[i]) {
                        int tmp = levels[i]; levels[i] = levels[j]; levels[j] = tmp;
                    }
                }
            }

            int prev_level = 0;
            for (int li = 0; li < num_levels; li++) {
                int level = levels[li];
                int pot_contribution = level - prev_level;
                if (pot_contribution == 0) { prev_level = level; continue; }

                int total_counted = 0;
                int eligible_opp_count = 0;
                bool traverser_eligible = false;
                for (int p = 0; p < np; p++) {
                    if (contributions[int(node_id) * np + p] >= level) {
                        total_counted++;
                        if (!(fold_mask & (1 << p))) {
                            if (p == int(params.traverser)) {
                                traverser_eligible = true;
                            } else {
                                eligible_opp_count++;
                            }
                        }
                    }
                }

                int pot_at_level = pot_contribution * total_counted;
                if (li == 0) pot_at_level += params.starting_pot;

                if (eligible_opp_count == 0) {
                    if (traverser_eligible) {
                        for (int h = 0; h < nh; h++) out[h] += float(pot_at_level);
                    }
                    prev_level = level;
                    continue;
                }

                if (traverser_eligible) {
                    if (eligible_opp_count == 1) {
                        // Single eligible opponent: pairwise sweep (exact)
                        for (int opp_p = 0; opp_p < np; opp_p++) {
                            if (opp_p == int(params.traverser)) continue;
                            if (fold_mask & (1 << opp_p)) continue;
                            if (contributions[int(node_id) * np + opp_p] < level) continue;

                            int oi = (opp_p < int(params.traverser)) ? opp_p : (opp_p - 1);
                            const device float* opp_r = reach + node_reach_base + opp_p * nh;
                            const device uint16_t* o_str = opp_str + oi * nh;
                            const device uint16_t* o_idx = opp_idx + oi * nh;

                            // Wins sweep (ascending)
                            float cfreach_sum = 0.0f;
                            float cfreach_minus[52];
                            for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;
                            int i = 0;
                            for (int si = 0; si < nh; si++) {
                                uint16_t str_h = pl_str[si];
                                uint16_t h = pl_idx[si];
                                while (i < nh && o_str[i] < str_h) {
                                    uint16_t ho = o_idx[i];
                                    if (chance_prob[ho] != 0.0f) {
                                        float r = opp_r[ho];
                                        if (r != 0.0f) {
                                            cfreach_sum += r;
                                            cfreach_minus[hand_cards[ho * 2]] += r;
                                            cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                                        }
                                    }
                                    i++;
                                }
                                float cfreach = cfreach_sum
                                    - cfreach_minus[hand_cards[h * 2]]
                                    - cfreach_minus[hand_cards[h * 2 + 1]];
                                out[h] += float(pot_at_level) * cfreach;
                            }

                            // Losses sweep (descending)
                            cfreach_sum = 0.0f;
                            for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;
                            i = nh - 1;
                            for (int si = nh - 1; si >= 0; si--) {
                                uint16_t str_h = pl_str[si];
                                uint16_t h = pl_idx[si];
                                while (i >= 0 && o_str[i] > str_h) {
                                    uint16_t ho = o_idx[i];
                                    if (chance_prob[ho] != 0.0f) {
                                        float r = opp_r[ho];
                                        if (r != 0.0f) {
                                            cfreach_sum += r;
                                            cfreach_minus[hand_cards[ho * 2]] += r;
                                            cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                                        }
                                    }
                                    i--;
                                }
                                float cfreach = cfreach_sum
                                    - cfreach_minus[hand_cards[h * 2]]
                                    - cfreach_minus[hand_cards[h * 2 + 1]];
                                out[h] -= float(pot_at_level) * cfreach;
                            }
                        }
                    } else {
                        // Multiple eligible opponents: product formula (exact)
                        float cum_weaker[1326];
                        float eff_total[1326];
                        for (int h = 0; h < nh; h++) { cum_weaker[h] = 1.0f; eff_total[h] = 1.0f; }

                        for (int opp_p = 0; opp_p < np; opp_p++) {
                            if (opp_p == int(params.traverser)) continue;
                            if (fold_mask & (1 << opp_p)) continue;
                            if (contributions[int(node_id) * np + opp_p] < level) continue;

                            int oi = (opp_p < int(params.traverser)) ? opp_p : (opp_p - 1);
                            const device float* opp_r = reach + node_reach_base + opp_p * nh;
                            const device uint16_t* o_str = opp_str + oi * nh;
                            const device uint16_t* o_idx = opp_idx + oi * nh;

                            float cw[1326];
                            float cfreach_sum = 0.0f;
                            float cfreach_minus[52];
                            for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;
                            int i = 0;

                            for (int si = 0; si < nh; si++) {
                                uint16_t str_h = pl_str[si];
                                uint16_t h = pl_idx[si];
                                while (i < nh && o_str[i] < str_h) {
                                    uint16_t ho = o_idx[i];
                                    float r = (chance_prob[ho] == 0.0f) ? 0.0f : opp_r[ho];
                                    if (r != 0.0f) {
                                        cfreach_sum += r;
                                        cfreach_minus[hand_cards[ho * 2]] += r;
                                        cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                                    }
                                    i++;
                                }
                                cw[h] = cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]];
                            }

                            // Finish sweep for total effective reach
                            while (i < nh) {
                                uint16_t ho = o_idx[i];
                                float r = (chance_prob[ho] == 0.0f) ? 0.0f : opp_r[ho];
                                if (r != 0.0f) {
                                    cfreach_sum += r;
                                    cfreach_minus[hand_cards[ho * 2]] += r;
                                    cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                                }
                                i++;
                            }

                            for (int h = 0; h < nh; h++) {
                                float r_h = (chance_prob[h] == 0.0f) ? 0.0f : opp_r[h];
                                float eff = cfreach_sum
                                    - cfreach_minus[hand_cards[h * 2]]
                                    - cfreach_minus[hand_cards[h * 2 + 1]]
                                    + r_h;
                                cum_weaker[h] *= cw[h];
                                eff_total[h] *= eff;
                            }
                        }

                        for (int h = 0; h < nh; h++) {
                            out[h] += float(pot_at_level)
                                * (float(eligible_opp_count + 1) * cum_weaker[h] - eff_total[h]);
                        }
                    }
                }
                prev_level = level;
            }

            // Subtract investment
            for (int h = 0; h < nh; h++) {
                out[h] -= (float(params.starting_pot) / float(np) + float(c_t));
            }
            } // end np != 2
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
