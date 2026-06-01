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

    // Fallback for np >= 4: leave at zero (not supported).
    for (int h = 0; h < nh; h++) out[h] = 0.0f;
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
            for (int h = 0; h < nh; h++) {
                float inst_regret = cfv_o[int(child) * nh + h] - cfv_avg[h];
                int ridx = regret_base + a * nh + h;

                // Inline DCFR discount (matches CPU bottom_up_zone)
                float old_r = regrets[ridx];
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
