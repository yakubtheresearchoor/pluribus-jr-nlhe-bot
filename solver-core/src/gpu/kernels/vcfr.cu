#define NODE_TYPE_TERMINAL 0
#define NODE_TYPE_CHANCE   1
#define NODE_TYPE_PLAYER   2

#define MAX_NA 4
#define UNUSED_U32 0xFFFFFFFFu

struct FlatNode {
    uint8_t node_type;
    uint8_t player_id;
    uint8_t board_state;
    uint8_t _pad0;
    uint16_t num_children;
    uint16_t _pad1;
    uint32_t children_start;
    int32_t amount;
    uint8_t action_label;
    uint8_t _pad2[3];
};

__device__ void sorted_sweep_showdown_vcfr(
    const float* opp_reach_all, int num_opp, int nh,
    const uint16_t* opp_strength, const uint16_t* opp_indices,
    const uint16_t* player_strength, const uint16_t* player_indices,
    const uint8_t* hand_cards,
    float* returned_cfv
) {
    for (int h = 0; h < nh; h++) returned_cfv[h] = 0.0f;

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

extern "C" __global__ void vcfr_compute_strategies(
    const float* __restrict__ regrets,
    float* __restrict__ strategy,
    const uint32_t* __restrict__ decision_node_ids,
    const FlatNode* __restrict__ nodes,
    const uint32_t* __restrict__ infoset_offsets,
    int num_infosets,
    int nh
) {
    int infoset_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (infoset_id >= num_infosets) return;

    uint32_t node_id = decision_node_ids[infoset_id];
    const FlatNode& node = nodes[node_id];
    int na = (int)node.num_children;
    int stride = MAX_NA * nh;
    const float* r = regrets + infoset_id * stride;
    float* s = strategy + infoset_id * stride;

    for (int h = 0; h < nh; h++) {
        float pos_sum = 0.0f;
        for (int a = 0; a < na; a++) {
            float rv = r[a * nh + h];
            if (rv > 0.0f) pos_sum += rv;
        }
        if (pos_sum > 0.0f) {
            for (int a = 0; a < na; a++) {
                float rv = r[a * nh + h];
                s[a * nh + h] = (rv > 0.0f) ? rv / pos_sum : 0.0f;
            }
        } else {
            float u = 1.0f / (float)na;
            for (int a = 0; a < na; a++) {
                s[a * nh + h] = u;
            }
        }
        for (int a = na; a < MAX_NA; a++) {
            s[a * nh + h] = 0.0f;
        }
    }
}

extern "C" __global__ void vcfr_top_down_reach(
    const uint32_t* __restrict__ level_nodes,
    int level_count,
    const FlatNode* __restrict__ nodes,
    const uint32_t* __restrict__ children,
    const float* __restrict__ strategy,
    const uint32_t* __restrict__ infoset_offsets,
    float* __restrict__ reach,
    int num_players,
    int nh
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= level_count) return;

    uint32_t node_id = level_nodes[idx];
    const FlatNode& node = nodes[node_id];
    int np = num_players;
    int node_reach_base = node_id * np * nh;

    if (node.node_type == NODE_TYPE_PLAYER) {
        int player = (int)node.player_id;
        uint32_t infoset_id = infoset_offsets[node_id];
        int stride = MAX_NA * nh;
        const float* sigma = strategy + infoset_id * stride;

        for (int a = 0; a < (int)node.num_children; a++) {
            uint32_t child = children[node.children_start + a];
            int child_reach_base = child * np * nh;

            for (int p = 0; p < np; p++) {
                for (int h = 0; h < nh; h++) {
                    reach[child_reach_base + p * nh + h] = reach[node_reach_base + p * nh + h];
                }
            }
            for (int h = 0; h < nh; h++) {
                reach[child_reach_base + player * nh + h] *= sigma[a * nh + h];
            }
        }
    } else {
        for (int a = 0; a < (int)node.num_children; a++) {
            uint32_t child = children[node.children_start + a];
            int child_reach_base = child * np * nh;
            for (int p = 0; p < np; p++) {
                for (int h = 0; h < nh; h++) {
                    reach[child_reach_base + p * nh + h] = reach[node_reach_base + p * nh + h];
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BATCHED bottom-up: processes num_outcomes × level_count nodes in one
// launch. Grid = (num_outcomes × level_count) blocks, 1 thread per block.
// blockIdx.x encodes (outcome * level_count + node_in_level).
// Sorted arrays and CFV are offset by outcome.
// Regret updates use atomicAdd into regret_accum.
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void vcfr_bottom_up_batched(
    const uint32_t* __restrict__ level_nodes,
    int level_count,
    int num_outcomes,
    int cfv_batch_stride,        // nn * nh — stride per outcome in cfv buffer
    int sorted_opp_stride,      // num_opp * nh — stride per outcome in sorted arrays
    const FlatNode* __restrict__ nodes,
    const uint32_t* __restrict__ children,
    const int32_t* __restrict__ contributions,
    const uint16_t* __restrict__ folded_masks,
    const float* __restrict__ strategy,
    const uint32_t* __restrict__ infoset_offsets,
    const float* __restrict__ reach,
    float* __restrict__ cfv,             // [num_outcomes * nn * nh]
    float* __restrict__ regrets,
    float* __restrict__ regret_accum,    // [num_infosets * MAX_NA * nh]
    float* __restrict__ cum_strategy,
    float* __restrict__ cum_accum,        // [num_infosets * MAX_NA * nh] — batched accumulation
    const float* __restrict__ initial_weight,
    const uint16_t* __restrict__ sorted_opp_strength,  // [num_outcomes * sorted_opp_stride]
    const uint16_t* __restrict__ sorted_opp_indices,
    const uint16_t* __restrict__ sorted_pl_strength,   // [num_outcomes * nh]
    const uint16_t* __restrict__ sorted_pl_indices,
    const uint8_t* __restrict__ hand_cards,
    const float* __restrict__ chance_prob,   // [num_outcomes * nh]
    int num_players,
    int nh,
    uint32_t traverser,
    float alpha_t,
    float beta_t,
    float gamma_t,
    float regret_floor,
    int32_t starting_pot,
    float num_combinations
) {
    int idx = blockIdx.x;
    int outcome = idx / level_count;
    int node_in_level = idx % level_count;
    if (outcome >= num_outcomes) return;

    uint32_t node_id = level_nodes[node_in_level];
    const FlatNode& node = nodes[node_id];
    int np = num_players;
    int num_opp = np - 1;

    // Per-outcome offsets
    int cfv_off = outcome * cfv_batch_stride;
    int sos_off = outcome * sorted_opp_stride;
    int sps_off = outcome * nh;

    const uint16_t* opp_str = sorted_opp_strength + sos_off;
    const uint16_t* opp_idx = sorted_opp_indices + sos_off;
    const uint16_t* pl_str = sorted_pl_strength + sps_off;
    const uint16_t* pl_idx = sorted_pl_indices + sps_off;
    float* cfv_o = cfv + cfv_off;

    if (node.node_type == NODE_TYPE_TERMINAL) {
        int node_reach_base = node_id * np * nh;
        int32_t c_t = contributions[node_id * np + traverser];
        uint16_t fold_mask = folded_masks[node_id];

        int num_active = 0;
        for (int p = 0; p < np; p++) {
            if (!(fold_mask & (1 << p))) num_active++;
        }

        float* out = cfv_o + node_id * nh;

        if (num_active <= 1 || (fold_mask & (1 << traverser))) {
            int32_t total_pot = starting_pot;
            for (int p = 0; p < np; p++) total_pot += contributions[node_id * np + p];
            float traverser_investment = (float)starting_pot / (float)np + (float)c_t;
            float payoff;
            if (fold_mask & (1 << traverser)) {
                payoff = -traverser_investment;
            } else {
                payoff = (float)total_pot - traverser_investment;
            }

            float opp_reach_sum = 0.0f;
            float opp_reach_minus[52];
            for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;

            for (int oi = 0; oi < num_opp; oi++) {
                int opp = (oi < (int)traverser) ? oi : (oi + 1);
                const float* opp_r = reach + node_reach_base + opp * nh;
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
            if (num_combinations > 0.0f) { for (int h = 0; h < nh; h++) out[h] /= num_combinations; }
            return;
        }

        bool all_equal = true;
        for (int p = 0; p < np; p++) {
            if (fold_mask & (1 << p)) continue;
            if (contributions[node_id * np + p] != c_t) { all_equal = false; break; }
        }

        if (all_equal) {
            int num_active_opp = 0;
            for (int p = 0; p < np; p++) {
                if (p == (int)traverser) continue;
                if (!(fold_mask & (1 << p))) num_active_opp++;
            }

            if (num_active_opp == 0) {
                int32_t total_pot = starting_pot;
                for (int p = 0; p < np; p++) total_pot += contributions[node_id * np + p];
                float traverser_investment = (float)starting_pot / (float)np + (float)c_t;
                float payoff = (float)total_pot - traverser_investment;

                float opp_reach_sum = 0.0f;
                float opp_reach_minus[52];
                for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;
                for (int oi = 0; oi < num_opp; oi++) {
                    int opp = (oi < (int)traverser) ? oi : (oi + 1);
                    const float* opp_r = reach + node_reach_base + opp * nh;
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
                if (num_combinations > 0.0f) { for (int h = 0; h < nh; h++) out[h] /= num_combinations; }
                return;
            }

            float opp_reach_local[MAX_NA * 1326];
            for (int oi = 0; oi < num_opp; oi++) {
                int opp = (oi < (int)traverser) ? oi : (oi + 1);
                if (fold_mask & (1 << opp)) {
                    for (int h = 0; h < nh; h++) opp_reach_local[oi * nh + h] = 0.0f;
                } else {
                    const float* opp_r = reach + node_reach_base + opp * nh;
                    for (int h = 0; h < nh; h++) opp_reach_local[oi * nh + h] = opp_r[h];
                }
            }

            if (num_active_opp == 1) {
                sorted_sweep_showdown_vcfr(
                    opp_reach_local, num_opp, nh,
                    opp_str, opp_idx,
                    pl_str, pl_idx,
                    hand_cards,
                    out
                );
                for (int h = 0; h < nh; h++) out[h] *= ((float)starting_pot / (float)np + (float)c_t);
            } else {
                float cum_weaker[5 * 1326];
                float eff_total[5 * 1326];
                for (int h = 0; h < nh; h++) { cum_weaker[h] = 0.0f; eff_total[h] = 0.0f; }

                for (int oi = 0; oi < num_opp; oi++) {
                    const float* opp_r = opp_reach_local + oi * nh;
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
                        cw[h] = cfreach_sum
                            - cfreach_minus[hand_cards[h * 2]]
                            - cfreach_minus[hand_cards[h * 2 + 1]];
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
                    out[h] = ((float)starting_pot / (float)np + (float)c_t) * ((float)(num_active_opp + 1) * cum_weaker[h] - eff_total[h]);
                }
            }
        } else {
            // Side pot
            for (int h = 0; h < nh; h++) out[h] = 0.0f;
            int levels[8];
            int num_levels = 0;
            for (int p = 0; p < np && num_levels < 8; p++) {
                int32_t c = contributions[node_id * np + p];
                bool found = false;
                for (int l = 0; l < num_levels; l++) {
                    if (levels[l] == c) { found = true; break; }
                }
                if (!found) levels[num_levels++] = c;
            }
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
                    if (contributions[node_id * np + p] >= level) {
                        total_counted++;
                        if (!(fold_mask & (1 << p))) {
                            if (p == (int)traverser) {
                                traverser_eligible = true;
                            } else {
                                eligible_opp_count++;
                            }
                        }
                    }
                }

                int pot_at_level = pot_contribution * total_counted;
                if (li == 0) pot_at_level += starting_pot;

                if (eligible_opp_count == 0) {
                    if (traverser_eligible) {
                        for (int h = 0; h < nh; h++) out[h] += (float)pot_at_level;
                    }
                    prev_level = level;
                    continue;
                }

                if (traverser_eligible) {
                    for (int opp_p = 0; opp_p < np; opp_p++) {
                        if (opp_p == (int)traverser) continue;
                        if (fold_mask & (1 << opp_p)) continue;
                        if (contributions[node_id * np + opp_p] < level) continue;

                        int oi = (opp_p < (int)traverser) ? opp_p : (opp_p - 1);
                        const float* opp_r = reach + node_reach_base + opp_p * nh;
                        const uint16_t* o_str = opp_str + oi * nh;
                        const uint16_t* o_idx = opp_idx + oi * nh;

                        float cfreach_sum = 0.0f;
                        float cfreach_minus[52];
                        for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

                        int i = 0;
                        for (int si = 0; si < nh; si++) {
                            uint16_t str_h = pl_str[si];
                            uint16_t h = pl_idx[si];
                            while (i < nh && o_str[i] < str_h) {
                                uint16_t ho = o_idx[i];
                                float r = opp_r[ho];
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
                            out[h] += (float)pot_at_level * cfreach;
                        }

                        cfreach_sum = 0.0f;
                        for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;
                        i = nh - 1;
                        for (int si = nh - 1; si >= 0; si--) {
                            uint16_t str_h = pl_str[si];
                            uint16_t h = pl_idx[si];
                            while (i >= 0 && o_str[i] > str_h) {
                                uint16_t ho = o_idx[i];
                                float r = opp_r[ho];
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
                            out[h] -= (float)pot_at_level * cfreach;
                        }
                    }
                }

                prev_level = level;
            }

            for (int h = 0; h < nh; h++) out[h] -= ((float)starting_pot / (float)np + (float)c_t);
        }
        if (num_combinations > 0.0f) { for (int h = 0; h < nh; h++) out[h] /= num_combinations; }
        return;
    }

    if (node.node_type == NODE_TYPE_CHANCE) {
        float* out = cfv_o + node_id * nh;
        for (int h = 0; h < nh; h++) out[h] = 0.0f;
        for (int a = 0; a < (int)node.num_children; a++) {
            uint32_t child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                out[h] += cfv_o[child * nh + h];
            }
        }
        return;
    }

    // Player node
    int owner = (int)node.player_id;
    int na = (int)node.num_children;
    uint32_t infoset_id = infoset_offsets[node_id];
    int stride = MAX_NA * nh;
    const float* sigma = strategy + infoset_id * stride;

    float cfv_avg[1326];
    for (int h = 0; h < nh; h++) cfv_avg[h] = 0.0f;

    if (owner == (int)traverser) {
        for (int a = 0; a < na; a++) {
            uint32_t child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                cfv_avg[h] += sigma[a * nh + h] * cfv_o[child * nh + h];
            }
        }

        int offset = infoset_id * stride;
        for (int a = 0; a < na; a++) {
            uint32_t child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                float inst_regret = cfv_o[child * nh + h] - cfv_avg[h];
                uint32_t ridx = offset + a * nh + h;
                // Weight by chance probability for correct expected regret
                float cp = chance_prob[outcome * nh + h];
                atomicAdd(&regret_accum[ridx], cp * inst_regret);
            }
        }

        for (int a = 0; a < na; a++) {
            for (int h = 0; h < nh; h++) {
                uint32_t cidx = offset + a * nh + h;
                float cp = chance_prob[outcome * nh + h];
                atomicAdd(&cum_accum[cidx], cp * sigma[a * nh + h]);
            }
        }
    } else {
        for (int a = 0; a < na; a++) {
            uint32_t child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                cfv_avg[h] += cfv_o[child * nh + h];
            }
        }
    }

    float* out = cfv_o + node_id * nh;
    for (int h = 0; h < nh; h++) {
        out[h] = cfv_avg[h];
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Original single-outcome bottom-up (unchanged, used by turn-start)
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void vcfr_bottom_up(
    const uint32_t* __restrict__ level_nodes,
    int level_count,
    const FlatNode* __restrict__ nodes,
    const uint32_t* __restrict__ children,
    const int32_t* __restrict__ contributions,
    const uint16_t* __restrict__ folded_masks,
    const float* __restrict__ strategy,
    const uint32_t* __restrict__ infoset_offsets,
    const float* __restrict__ reach,
    float* __restrict__ cfv,
    float* __restrict__ regrets,
    float* __restrict__ cum_strategy,
    const float* __restrict__ initial_weight,
    const uint16_t* __restrict__ sorted_opp_strength,
    const uint16_t* __restrict__ sorted_opp_indices,
    const uint16_t* __restrict__ sorted_pl_strength,
    const uint16_t* __restrict__ sorted_pl_indices,
    const uint8_t* __restrict__ hand_cards,
    int num_players,
    int nh,
    uint32_t traverser,
    float alpha_t,
    float beta_t,
    float gamma_t,
    float regret_floor,
    int32_t starting_pot,
    float num_combinations
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= level_count) return;

    uint32_t node_id = level_nodes[idx];
    const FlatNode& node = nodes[node_id];
    int np = num_players;
    int num_opp = np - 1;

    if (node.node_type == NODE_TYPE_TERMINAL) {
        int node_reach_base = node_id * np * nh;
        int32_t c_t = contributions[node_id * np + traverser];
        uint16_t fold_mask = folded_masks[node_id];
        float* out = cfv + node_id * nh;

        int num_active = 0;
        for (int p = 0; p < np; p++) {
            if (!(fold_mask & (1 << p))) num_active++;
        }

        if (num_active <= 1 || (fold_mask & (1 << traverser))) {
            int32_t total_pot = starting_pot;
            for (int p = 0; p < np; p++) total_pot += contributions[node_id * np + p];
            float traverser_investment = (float)starting_pot / (float)np + (float)c_t;
            float payoff = (fold_mask & (1 << traverser)) ? -traverser_investment : ((float)total_pot - traverser_investment);

            float opp_reach_sum = 0.0f;
            float opp_reach_minus[52];
            for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;
            for (int oi = 0; oi < num_opp; oi++) {
                int opp = (oi < (int)traverser) ? oi : (oi + 1);
                const float* opp_r = reach + node_reach_base + opp * nh;
                for (int ho = 0; ho < nh; ho++) {
                    float r = opp_r[ho];
                    if (r != 0.0f) { opp_reach_sum += r; opp_reach_minus[hand_cards[ho * 2]] += r; opp_reach_minus[hand_cards[ho * 2 + 1]] += r; }
                }
            }
            if (opp_reach_sum > 0.0f) { for (int h = 0; h < nh; h++) out[h] = payoff * (opp_reach_sum - opp_reach_minus[hand_cards[h * 2]] - opp_reach_minus[hand_cards[h * 2 + 1]]); }
            else { for (int h = 0; h < nh; h++) out[h] = 0.0f; }
            if (num_combinations > 0.0f) { for (int h = 0; h < nh; h++) out[h] /= num_combinations; }
            return;
        }

        bool all_equal = true;
        for (int p = 0; p < np; p++) { if (fold_mask & (1 << p)) continue; if (contributions[node_id * np + p] != c_t) { all_equal = false; break; } }

        if (all_equal) {
            int num_active_opp = 0;
            for (int p = 0; p < np; p++) { if (p == (int)traverser) continue; if (!(fold_mask & (1 << p))) num_active_opp++; }

            if (num_active_opp == 0) {
                int32_t total_pot = starting_pot;
                for (int p = 0; p < np; p++) total_pot += contributions[node_id * np + p];
                float traverser_investment = (float)starting_pot / (float)np + (float)c_t;
                float payoff = (float)total_pot - traverser_investment;
                float opp_reach_sum = 0.0f;
                float opp_reach_minus[52];
                for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;
                for (int oi = 0; oi < num_opp; oi++) {
                    int opp = (oi < (int)traverser) ? oi : (oi + 1);
                    const float* opp_r = reach + node_reach_base + opp * nh;
                    for (int ho = 0; ho < nh; ho++) { float r = opp_r[ho]; if (r != 0.0f) { opp_reach_sum += r; opp_reach_minus[hand_cards[ho * 2]] += r; opp_reach_minus[hand_cards[ho * 2 + 1]] += r; } }
                }
                if (opp_reach_sum > 0.0f) { for (int h = 0; h < nh; h++) out[h] = payoff * (opp_reach_sum - opp_reach_minus[hand_cards[h * 2]] - opp_reach_minus[hand_cards[h * 2 + 1]]); }
                else { for (int h = 0; h < nh; h++) out[h] = 0.0f; }
                if (num_combinations > 0.0f) { for (int h = 0; h < nh; h++) out[h] /= num_combinations; }
                return;
            }

            float opp_reach_local[MAX_NA * 1326];
            for (int oi = 0; oi < num_opp; oi++) {
                int opp = (oi < (int)traverser) ? oi : (oi + 1);
                if (fold_mask & (1 << opp)) { for (int h = 0; h < nh; h++) opp_reach_local[oi * nh + h] = 0.0f; }
                else { const float* opp_r = reach + node_reach_base + opp * nh; for (int h = 0; h < nh; h++) opp_reach_local[oi * nh + h] = opp_r[h]; }
            }

            if (num_active_opp == 1) {
                sorted_sweep_showdown_vcfr(opp_reach_local, num_opp, nh, sorted_opp_strength, sorted_opp_indices, sorted_pl_strength, sorted_pl_indices, hand_cards, out);
                for (int h = 0; h < nh; h++) out[h] *= ((float)starting_pot / (float)np + (float)c_t);
            } else {
                float cum_weaker[5 * 1326]; float eff_total[5 * 1326];
                for (int h = 0; h < nh; h++) { cum_weaker[h] = 0.0f; eff_total[h] = 0.0f; }
                for (int oi = 0; oi < num_opp; oi++) {
                    const float* opp_r = opp_reach_local + oi * nh;
                    float cw[1326]; float cfreach_sum = 0.0f; float cfreach_minus[52]; for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;
                    int i = 0;
                    for (int si = 0; si < nh; si++) {
                        uint16_t str_h = sorted_pl_strength[si]; uint16_t h = sorted_pl_indices[si];
                        while (i < nh && sorted_opp_strength[oi * nh + i] < str_h) { uint16_t ho = sorted_opp_indices[oi * nh + i]; float r = opp_r[ho]; if (r != 0.0f) { cfreach_sum += r; cfreach_minus[hand_cards[ho * 2]] += r; cfreach_minus[hand_cards[ho * 2 + 1]] += r; } i++; }
                        cw[h] = cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]];
                    }
                    while (i < nh) { uint16_t ho = sorted_opp_indices[oi * nh + i]; float r = opp_r[ho]; if (r != 0.0f) { cfreach_sum += r; cfreach_minus[hand_cards[ho * 2]] += r; cfreach_minus[hand_cards[ho * 2 + 1]] += r; } i++; }
                    for (int h = 0; h < nh; h++) {
                        float eff = cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]] + opp_r[h];
                        if (oi == 0) { cum_weaker[h] = cw[h]; eff_total[h] = eff; } else { cum_weaker[h] *= cw[h]; eff_total[h] *= eff; }
                    }
                }
                for (int h = 0; h < nh; h++) out[h] = ((float)starting_pot / (float)np + (float)c_t) * ((float)(num_active_opp + 1) * cum_weaker[h] - eff_total[h]);
            }
        } else {
            for (int h = 0; h < nh; h++) out[h] = 0.0f;
            int levels[8]; int num_levels = 0;
            for (int p = 0; p < np && num_levels < 8; p++) { int32_t c = contributions[node_id * np + p]; bool found = false; for (int l = 0; l < num_levels; l++) { if (levels[l] == c) { found = true; break; } } if (!found) levels[num_levels++] = c; }
            for (int i = 0; i < num_levels - 1; i++) for (int j = i + 1; j < num_levels; j++) if (levels[j] < levels[i]) { int tmp = levels[i]; levels[i] = levels[j]; levels[j] = tmp; }
            int prev_level = 0;
            for (int li = 0; li < num_levels; li++) {
                int level = levels[li]; int pot_contribution = level - prev_level;
                if (pot_contribution == 0) { prev_level = level; continue; }
                int total_counted = 0; int eligible_opp_count = 0; bool traverser_eligible = false;
                for (int p = 0; p < np; p++) { if (contributions[node_id * np + p] >= level) { total_counted++; if (!(fold_mask & (1 << p))) { if (p == (int)traverser) traverser_eligible = true; else eligible_opp_count++; } } }
                int pot_at_level = pot_contribution * total_counted;
                if (li == 0) pot_at_level += starting_pot;
                if (eligible_opp_count == 0) { if (traverser_eligible) { for (int h = 0; h < nh; h++) out[h] += (float)pot_at_level; } prev_level = level; continue; }
                if (traverser_eligible) {
                    for (int opp_p = 0; opp_p < np; opp_p++) {
                        if (opp_p == (int)traverser || (fold_mask & (1 << opp_p)) || contributions[node_id * np + opp_p] < level) continue;
                        int oi = (opp_p < (int)traverser) ? opp_p : (opp_p - 1);
                        const float* opp_r = reach + node_reach_base + opp_p * nh;
                        const uint16_t* o_str = sorted_opp_strength + oi * nh;
                        const uint16_t* o_idx = sorted_opp_indices + oi * nh;
                        float cfreach_sum = 0.0f; float cfreach_minus[52]; for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;
                        int i = 0;
                        for (int si = 0; si < nh; si++) { uint16_t str_h = sorted_pl_strength[si]; uint16_t h = sorted_pl_indices[si]; while (i < nh && o_str[i] < str_h) { uint16_t ho = o_idx[i]; float r = opp_r[ho]; if (r != 0.0f) { cfreach_sum += r; cfreach_minus[hand_cards[ho * 2]] += r; cfreach_minus[hand_cards[ho * 2 + 1]] += r; } i++; } out[h] += (float)pot_at_level * (cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]]); }
                        cfreach_sum = 0.0f; for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f; i = nh - 1;
                        for (int si = nh - 1; si >= 0; si--) { uint16_t str_h = sorted_pl_strength[si]; uint16_t h = sorted_pl_indices[si]; while (i >= 0 && o_str[i] > str_h) { uint16_t ho = o_idx[i]; float r = opp_r[ho]; if (r != 0.0f) { cfreach_sum += r; cfreach_minus[hand_cards[ho * 2]] += r; cfreach_minus[hand_cards[ho * 2 + 1]] += r; } i--; } out[h] -= (float)pot_at_level * (cfreach_sum - cfreach_minus[hand_cards[h * 2]] - cfreach_minus[hand_cards[h * 2 + 1]]); }
                    }
                }
                prev_level = level;
            }
            for (int h = 0; h < nh; h++) out[h] -= ((float)starting_pot / (float)np + (float)c_t);
        }
        if (num_combinations > 0.0f) { for (int h = 0; h < nh; h++) out[h] /= num_combinations; }
        return;
    }

    if (node.node_type == NODE_TYPE_CHANCE) {
        for (int h = 0; h < nh; h++) cfv[node_id * nh + h] = 0.0f;
        for (int a = 0; a < (int)node.num_children; a++) { uint32_t child = children[node.children_start + a]; for (int h = 0; h < nh; h++) cfv[node_id * nh + h] += cfv[child * nh + h]; }
        return;
    }

    int owner = (int)node.player_id;
    int na = (int)node.num_children;
    uint32_t infoset_id = infoset_offsets[node_id];
    int stride = MAX_NA * nh;
    const float* sigma = strategy + infoset_id * stride;
    float cfv_avg[1326]; for (int h = 0; h < nh; h++) cfv_avg[h] = 0.0f;

    if (owner == (int)traverser) {
        for (int a = 0; a < na; a++) { uint32_t child = children[node.children_start + a]; for (int h = 0; h < nh; h++) cfv_avg[h] += sigma[a * nh + h] * cfv[child * nh + h]; }
        int offset = infoset_id * stride;
        for (int a = 0; a < na; a++) { uint32_t child = children[node.children_start + a]; for (int h = 0; h < nh; h++) { float inst_regret = cfv[child * nh + h] - cfv_avg[h]; uint32_t ridx = offset + a * nh + h; float coef = (regrets[ridx] >= 0.0f) ? alpha_t : beta_t; regrets[ridx] = coef * regrets[ridx] + inst_regret; if (regrets[ridx] < regret_floor) regrets[ridx] = regret_floor; } }
        for (int a = 0; a < na; a++) { for (int h = 0; h < nh; h++) { uint32_t cidx = offset + a * nh + h; cum_strategy[cidx] = gamma_t * cum_strategy[cidx] + sigma[a * nh + h]; } }
    } else {
        for (int a = 0; a < na; a++) { uint32_t child = children[node.children_start + a]; for (int h = 0; h < nh; h++) cfv_avg[h] += cfv[child * nh + h]; }
    }
    for (int h = 0; h < nh; h++) cfv[node_id * nh + h] = cfv_avg[h];
}

extern "C" __global__ void vcfr_chance_accumulate(
    float* __restrict__ cfv_accum,
    const float* __restrict__ cfv,
    const float* __restrict__ chance_prob,
    const uint32_t* __restrict__ chance_child_ids,
    int num_chance_children,
    int nh,
    int outcome
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = num_chance_children * nh;
    if (idx >= total) return;
    int cn = idx / nh;
    int h = idx % nh;
    uint32_t child_id = chance_child_ids[cn];
    float prob = chance_prob[outcome * nh + h];
    cfv_accum[child_id * nh + h] += prob * cfv[child_id * nh + h];
}

extern "C" __global__ void vcfr_chance_finalize(
    float* __restrict__ cfv,
    const float* __restrict__ cfv_accum,
    const uint32_t* __restrict__ chance_child_ids,
    int num_chance_children,
    int nh
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = num_chance_children * nh;
    if (idx >= total) return;
    int cn = idx / nh;
    int h = idx % nh;
    uint32_t child_id = chance_child_ids[cn];
    cfv[child_id * nh + h] = cfv_accum[child_id * nh + h];
}

// ═══════════════════════════════════════════════════════════════════════
// BATCHED chance accumulation: accumulates per-outcome CFVs into
// per-group accumulators using atomicAdd. Each outcome belongs to a
// group (e.g., turn card). The group's accumulator stores the
// probability-weighted sum of CFVs for that group's outcomes.
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void vcfr_chance_accumulate_grouped(
    float* __restrict__ cfv_accum,       // [num_groups * nn * nh]
    const float* __restrict__ cfv_batch, // [num_outcomes * nn * nh]
    const float* __restrict__ chance_prob, // [num_outcomes * nh]
    const uint32_t* __restrict__ chance_child_ids,
    const int32_t* __restrict__ outcome_to_group, // [num_outcomes]
    int num_outcomes,
    int num_chance_children,
    int nn,
    int nh
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = num_outcomes * num_chance_children * nh;
    if (idx >= total) return;

    int h = idx % nh;
    int cn = (idx / nh) % num_chance_children;
    int outcome = idx / (num_chance_children * nh);

    uint32_t child_id = chance_child_ids[cn];
    int group = outcome_to_group[outcome];
    float prob = chance_prob[outcome * nh + h];
    float val = prob * cfv_batch[outcome * nn * nh + child_id * nh + h];

    atomicAdd(&cfv_accum[group * nn * nh + child_id * nh + h], val);
}

// ═══════════════════════════════════════════════════════════════════════
// Apply regret discount after batched accumulation:
// regrets[ridx] = coef * regrets[ridx] + regret_accum[ridx]
// Then reset regret_accum to 0.
// ═══════════════════════════════════════════════════════════════════════

extern "C" __global__ void vcfr_regret_apply(
    float* __restrict__ regrets,
    float* __restrict__ regret_accum,
    int total_size,
    float alpha_t,
    float beta_t,
    float regret_floor
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_size) return;

    float ir = regret_accum[idx];
    float old_r = regrets[idx];
    float coef = (old_r >= 0.0f) ? alpha_t : beta_t;
    regrets[idx] = coef * old_r + ir;
    if (regrets[idx] < regret_floor) regrets[idx] = regret_floor;
    regret_accum[idx] = 0.0f;
}

// Apply cum_strategy accumulation with gamma discount:
// cum_strategy[idx] = gamma_t * cum_strategy[idx] + cum_accum[idx]
// Then reset cum_accum to 0.
extern "C" __global__ void vcfr_cum_apply(
    float* __restrict__ cum_strategy,
    float* __restrict__ cum_accum,
    int total_size,
    float gamma_t
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_size) return;

    cum_strategy[idx] = gamma_t * cum_strategy[idx] + cum_accum[idx];
    cum_accum[idx] = 0.0f;
}

// Initialize reach buffer: zero everything, then copy initial_weight
// into reach[0..np*nh-1] (node 0's reach).
// grid = ceil(np*nh / 256), block = 256
extern "C" __global__ void vcfr_init_reach(
    float* __restrict__ reach,
    const float* __restrict__ initial_weight,
    int total_reach_size,  // nn * np * nh
    int np_nh              // np * nh (size of initial_weight)
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < np_nh) {
        reach[idx] = initial_weight[idx];
    } else if (idx < total_reach_size) {
        reach[idx] = 0.0f;
    }
}

// Zero a buffer. Graph-capturable alternative to cuMemsetD8Async.
extern "C" __global__ void vcfr_zero_buffer(
    float* __restrict__ buf,
    int size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) buf[idx] = 0.0f;
}
