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

// Reused from mccfr.cu: O(NH) sorted-sweep showdown.
// Returns unscaled net win/loss fraction per hand in returned_cfv.
// Caller must multiply by c_t to get actual CFV.
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

// Kernel 1: Compute strategies from regrets at all decision nodes.
// One thread per infoset. Regret-matching: positive regrets normalized.
// strategy layout: infosets * MAX_NA * nh, stride MAX_NA*nh per infoset.
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

// Kernel 2: Top-down reach propagation for one level.
// Processes all nodes at 'level'. Copies reach from parent, modifies
// for the acting player at player nodes.
// reach layout: num_nodes * num_players * nh
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

// Kernel 3: Bottom-up CFV + regret update for one level, one traverser.
// Processes all nodes at 'level'. Terminal: evaluate. Player: CFV from children.
// Traverser nodes: update regrets and cumulative strategy.
// CRITICAL: At opponent nodes, SUM child CFVs (no strategy weighting).
// At traverser nodes, WEIGHT by strategy (standard CFR).
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
    float weight,
    float regret_floor
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

        int num_active = 0;
        for (int p = 0; p < np; p++) {
            if (!(fold_mask & (1 << p))) num_active++;
        }

        float* out = cfv + node_id * nh;

        if (num_active <= 1 || (fold_mask & (1 << traverser))) {
            int32_t pot = 0;
            for (int p = 0; p < np; p++) pot += contributions[node_id * np + p];
            float payoff;
            if (fold_mask & (1 << traverser)) {
                payoff = -(float)c_t;
            } else {
                payoff = (float)(pot - c_t);
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
            return;
        }

        bool all_equal = true;
        for (int p = 0; p < np; p++) {
            if (fold_mask & (1 << p)) continue;
            if (contributions[node_id * np + p] != c_t) { all_equal = false; break; }
        }

        if (all_equal) {
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

            sorted_sweep_showdown_vcfr(
                opp_reach_local, num_opp, nh,
                sorted_opp_strength, sorted_opp_indices,
                sorted_pl_strength, sorted_pl_indices,
                hand_cards,
                out
            );
            for (int h = 0; h < nh; h++) out[h] *= (float)c_t;
        } else {
            // Side pot: simplified — treat as pot-level showdown
            // Full side pot would need per-level processing (future work)
            for (int h = 0; h < nh; h++) out[h] = 0.0f;
        }
        return;
    }

    if (node.node_type == NODE_TYPE_CHANCE) {
        // Chance nodes handled during bottom-up pass for non-compound trees
        // For river-only trees, chance nodes shouldn't appear
        for (int h = 0; h < nh; h++) {
            cfv[node_id * nh + h] = 0.0f;
        }
        for (int a = 0; a < (int)node.num_children; a++) {
            uint32_t child = children[node.children_start + a];
            // For single-child chance with multiple outcomes, caller handles
            for (int h = 0; h < nh; h++) {
                cfv[node_id * nh + h] += cfv[child * nh + h];
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
                cfv_avg[h] += sigma[a * nh + h] * cfv[child * nh + h];
            }
        }

        int offset = infoset_id * stride;
        for (int a = 0; a < na; a++) {
            uint32_t child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                float inst_regret = cfv[child * nh + h] - cfv_avg[h];
                uint32_t ridx = offset + a * nh + h;
                regrets[ridx] += weight * inst_regret;
                if (regrets[ridx] < regret_floor) regrets[ridx] = regret_floor;
            }
        }

        int node_reach_base = node_id * np * nh;
        for (int a = 0; a < na; a++) {
            for (int h = 0; h < nh; h++) {
                uint32_t cidx = offset + a * nh + h;
                float t_reach = reach[node_reach_base + traverser * nh + h];
                cum_strategy[cidx] += weight * t_reach * sigma[a * nh + h];
            }
        }
    } else {
        // CRITICAL: At opponent nodes, SUM child CFVs.
        // The pre-computed reach at each child already includes the opponent's
        // strategy contribution. Weighting by strategy would double-count.
        for (int a = 0; a < na; a++) {
            uint32_t child = children[node.children_start + a];
            for (int h = 0; h < nh; h++) {
                cfv_avg[h] += cfv[child * nh + h];
            }
        }
    }

    for (int h = 0; h < nh; h++) {
        cfv[node_id * nh + h] = cfv_avg[h];
    }
}

// Kernel 4: Update average strategy (no-op in sequential mode — done inline
// in bottom_up). Kept as placeholder for potential future batch mode.
extern "C" __global__ void vcfr_update_avg_strategy(
    float* __restrict__ cum_strategy,
    const float* __restrict__ strategy,
    const uint32_t* __restrict__ decision_node_ids,
    const FlatNode* __restrict__ nodes,
    const uint32_t* __restrict__ infoset_offsets,
    const float* __restrict__ reach,
    int num_infosets,
    int num_players,
    int nh,
    uint32_t traverser,
    float weight
) {
    int infoset_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (infoset_id >= num_infosets) return;

    uint32_t node_id = decision_node_ids[infoset_id];
    const FlatNode& node = nodes[node_id];

    if (node.player_id != traverser) return;

    int na = (int)node.num_children;
    int stride = MAX_NA * nh;
    const float* sigma = strategy + infoset_id * stride;
    float* cum = cum_strategy + infoset_id * stride;
    int node_reach_base = node_id * num_players * nh;

    for (int a = 0; a < na; a++) {
        for (int h = 0; h < nh; h++) {
            float t_reach = reach[node_reach_base + traverser * nh + h];
            cum[a * nh + h] += weight * t_reach * sigma[a * nh + h];
        }
    }
}
