#define NODE_TYPE_TERMINAL 0
#define NODE_TYPE_CHANCE   1
#define NODE_TYPE_PLAYER   2

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

template <int NH>
__device__ void compute_strategy(
    const float* regrets, uint32_t offset, int num_actions,
    float* strategy
) {
    for (int h = 0; h < NH; h++) {
        float pos_sum = 0.0f;
        for (int a = 0; a < num_actions; a++) {
            float r = regrets[offset + a * NH + h];
            if (r > 0.0f) pos_sum += r;
        }
        if (pos_sum > 0.0f) {
            for (int a = 0; a < num_actions; a++) {
                float r = regrets[offset + a * NH + h];
                strategy[a * NH + h] = (r > 0.0f) ? r / pos_sum : 0.0f;
            }
        } else {
            float u = 1.0f / (float)num_actions;
            for (int a = 0; a < num_actions; a++) {
                strategy[a * NH + h] = u;
            }
        }
    }
}

enum FrameState : uint8_t {
    STATE_ENTER = 0,
    STATE_RETURN = 1,
};

template <int NH, int NA, int ND>
__device__ void vanilla_walk(
    const FlatNode* __restrict__ nodes,
    const uint32_t* __restrict__ children,
    const int32_t* __restrict__ contributions,
    float* regrets,
    float* cum_strategy,
    const uint32_t* __restrict__ node_offsets,
    const float* __restrict__ sign_table,
    uint32_t num_players,
    float weight,
    float regret_floor,
    uint32_t traverser,
    float* __restrict__ opp_reach,
    float* __restrict__ treach
) {
    struct Frame {
        uint32_t node_idx;
        FrameState state;
        int num_actions;
        int child_idx;
        float strategy[NA * NH];
        float cfv_actions[NA * NH];
        float saved_reach[NH];
    };

    Frame stack[ND];
    int sp = 0;

    float returned_cfv[NH];
    for (int h = 0; h < NH; h++) returned_cfv[h] = 0.0f;

    stack[sp].node_idx = 0;
    stack[sp].state = STATE_ENTER;
    sp++;

    while (sp > 0) {
        Frame& f = stack[sp - 1];
        const FlatNode& node = nodes[f.node_idx];

        if (f.state == STATE_ENTER) {
            if (node.node_type == NODE_TYPE_TERMINAL) {
                int32_t c_t = contributions[f.node_idx * num_players + traverser];
                uint32_t opp = 1 - traverser;
                int32_t c_o_val = contributions[f.node_idx * num_players + opp];
                bool is_showdown = (c_t == c_o_val);

                if (is_showdown) {
                    float c = (float)c_o_val;
                    for (int h = 0; h < NH; h++) {
                        float v = 0.0f;
                        for (int ho = 0; ho < NH; ho++) {
                            if (ho != h) {
                                v += opp_reach[ho] * c * sign_table[h * NH + ho];
                            }
                        }
                        returned_cfv[h] = v;
                    }
                } else {
                    float cf;
                    if (c_t < c_o_val) {
                        cf = -(float)c_t;
                    } else {
                        cf = (float)c_o_val;
                    }
                    for (int h = 0; h < NH; h++) {
                        float v = 0.0f;
                        for (int ho = 0; ho < NH; ho++) {
                            if (ho != h) v += opp_reach[ho] * cf;
                        }
                        returned_cfv[h] = v;
                    }
                }
                sp--;
                continue;
            }

            if (node.node_type == NODE_TYPE_CHANCE) {
                returned_cfv[0] = 0.0f;
                sp--;
                continue;
            }

            uint32_t player = node.player_id;
            int na = (int)node.num_children;
            uint32_t offset = node_offsets[f.node_idx];

            compute_strategy<NH>(regrets, offset, na, f.strategy);
            f.num_actions = na;
            f.child_idx = 0;
            for (int i = 0; i < NA * NH; i++) f.cfv_actions[i] = 0.0f;

            float* reach_to_modify = (player == traverser) ? treach : opp_reach;
            for (int h = 0; h < NH; h++) {
                f.saved_reach[h] = reach_to_modify[h];
                reach_to_modify[h] *= f.strategy[0 * NH + h];
            }

            f.state = STATE_RETURN;
            stack[sp].node_idx = children[node.children_start + 0];
            stack[sp].state = STATE_ENTER;
            sp++;
            continue;
        }

        if (f.state == STATE_RETURN) {
            const FlatNode& node2 = nodes[f.node_idx];
            uint32_t player = node2.player_id;
            uint32_t offset = node_offsets[f.node_idx];

            for (int h = 0; h < NH; h++) {
                f.cfv_actions[f.child_idx * NH + h] = returned_cfv[h];
            }

            f.child_idx++;
            if (f.child_idx < f.num_actions) {
                float* reach_to_modify = (player == traverser) ? treach : opp_reach;
                for (int h = 0; h < NH; h++) {
                    reach_to_modify[h] = f.saved_reach[h] * f.strategy[f.child_idx * NH + h];
                }

                stack[sp].node_idx = children[node2.children_start + f.child_idx];
                stack[sp].state = STATE_ENTER;
                sp++;
                continue;
            }

            float* reach_to_restore = (player == traverser) ? treach : opp_reach;
            for (int h = 0; h < NH; h++) reach_to_restore[h] = f.saved_reach[h];

            if (player == traverser) {
                float cfv_avg[NH];
                for (int h = 0; h < NH; h++) {
                    cfv_avg[h] = 0.0f;
                    for (int a = 0; a < f.num_actions; a++) {
                        cfv_avg[h] += f.strategy[a * NH + h] * f.cfv_actions[a * NH + h];
                    }
                }

                for (int h = 0; h < NH; h++) {
                    for (int a = 0; a < f.num_actions; a++) {
                        float inst_regret = f.cfv_actions[a * NH + h] - cfv_avg[h];
                        uint32_t idx = offset + a * NH + h;
                        atomicAdd(&regrets[idx], weight * inst_regret);
                    }
                }

                for (int a2 = 0; a2 < f.num_actions; a2++) {
                    for (int h = 0; h < NH; h++) {
                        uint32_t ridx = offset + a2 * NH + h;
                        if (regrets[ridx] < regret_floor) regrets[ridx] = regret_floor;
                    }
                }

                for (int h = 0; h < NH; h++) {
                    for (int a = 0; a < f.num_actions; a++) {
                        uint32_t idx = offset + a * NH + h;
                        atomicAdd(&cum_strategy[idx], weight * treach[h] * f.strategy[a * NH + h]);
                    }
                }

                for (int h = 0; h < NH; h++) returned_cfv[h] = cfv_avg[h];
            } else {
                for (int h = 0; h < NH; h++) {
                    returned_cfv[h] = 0.0f;
                    for (int a = 0; a < f.num_actions; a++) {
                        returned_cfv[h] += f.cfv_actions[a * NH + h];
                    }
                }
            }

            sp--;
            continue;
        }
    }
}

extern "C" __global__ void mccfr_kuhn(
    const FlatNode* nodes, const uint32_t* children,
    const int32_t* contributions, float* regrets, float* cum_strategy,
    const uint32_t* node_offsets, const float* sign_table,
    uint32_t num_players, uint32_t batch_size,
    float weight, float regret_floor, const uint32_t* seeds
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size) return;

    uint32_t rng = seeds[tid];
    rng = rng * 1103515245 + 12345;
    uint32_t traverser = tid % num_players;

    float opp_reach[3];
    float treach[3];
    for (int h = 0; h < 3; h++) {
        opp_reach[h] = 1.0f;
        treach[h] = 1.0f;
    }

    vanilla_walk<3, 4, 16>(nodes, children, contributions, regrets, cum_strategy,
                           node_offsets, sign_table, num_players, weight, regret_floor,
                           traverser, opp_reach, treach);
}

extern "C" __global__ void mccfr_leduc(
    const FlatNode* nodes, const uint32_t* children,
    const int32_t* contributions, float* regrets, float* cum_strategy,
    const uint32_t* node_offsets, const float* sign_table,
    uint32_t num_players, uint32_t batch_size,
    float weight, float regret_floor, const uint32_t* seeds
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size) return;

    uint32_t rng = seeds[tid];
    rng = rng * 1103515245 + 12345;
    uint32_t traverser = tid % num_players;

    float opp_reach[6];
    float treach[6];
    for (int h = 0; h < 6; h++) {
        opp_reach[h] = 1.0f;
        treach[h] = 1.0f;
    }

    vanilla_walk<6, 4, 16>(nodes, children, contributions, regrets, cum_strategy,
                           node_offsets, sign_table, num_players, weight, regret_floor,
                           traverser, opp_reach, treach);
}

#define MAX_DEPTH 16
#define MAX_NA 4
#define MAX_NH 1326
#define MAX_NP 10
#define FRAME_STRIDE 11934
#define FRAME_STRAT_OFF 0
#define FRAME_CFV_OFF (MAX_NA * MAX_NH)
#define FRAME_REACH_OFF (FRAME_CFV_OFF + MAX_NA * MAX_NH)
#define BOARD_STATE_RIVER 2

__device__ void compute_strategy_nplayer(
    const float* regrets, uint32_t offset, int num_actions, int nh,
    float* strategy
) {
    for (int h = 0; h < nh; h++) {
        float pos_sum = 0.0f;
        for (int a = 0; a < num_actions; a++) {
            float r = regrets[offset + a * nh + h];
            if (r > 0.0f) pos_sum += r;
        }
        if (pos_sum > 0.0f) {
            for (int a = 0; a < num_actions; a++) {
                float r = regrets[offset + a * nh + h];
                strategy[a * nh + h] = (r > 0.0f) ? r / pos_sum : 0.0f;
            }
        } else {
            float u = 1.0f / (float)num_actions;
            for (int a = 0; a < num_actions; a++) {
                strategy[a * nh + h] = u;
            }
        }
    }
}

__device__ void sorted_sweep_showdown(
    const float* opp_reach_all, int num_opp, int nh,
    const uint16_t* opp_strength, const uint16_t* opp_indices,
    const uint16_t* player_strength, const uint16_t* player_indices,
    const uint8_t* hand_cards,
    const int32_t* contributions, uint32_t node_idx,
    uint32_t traverser, uint32_t num_players,
    float* returned_cfv
) {
    int32_t c_t = contributions[node_idx * num_players + traverser];

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

    for (int h = 0; h < nh; h++) {
        returned_cfv[h] *= (float)c_t;
    }
}

extern "C" __global__ void mccfr_nplayer(
    const FlatNode* nodes, const uint32_t* children,
    const int32_t* contributions, float* regrets, float* cum_strategy,
    const uint32_t* node_offsets,
    const uint16_t* hand_ranks,
    const uint16_t* sorted_opp_strength,
    const uint16_t* sorted_opp_indices,
    const uint16_t* sorted_player_strength,
    const uint16_t* sorted_player_indices,
    const uint16_t* same_hand_idx,
    const uint8_t* hand_cards,
    const float* initial_weight,
    const uint16_t* chance_ranks_table,
    const uint8_t* remaining_deck,
    const uint16_t* chance_sorted_strength,
    const uint16_t* chance_sorted_indices,
    float* frame_data,
    uint32_t num_players, uint32_t batch_size,
    int32_t nh, int32_t num_remaining,
    float weight, float regret_floor, const uint32_t* seeds
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size) return;

    uint32_t rng = seeds[tid];
    rng = rng * 1103515245 + 12345;
    uint32_t traverser = tid % num_players;

    int num_opp = (int)num_players - 1;

    float opp_reach_all[MAX_NP * MAX_NH];
    float treach_buf[MAX_NH];
    for (int h = 0; h < nh; h++) {
        treach_buf[h] = initial_weight[traverser * nh + h];
        for (int oi = 0; oi < num_opp; oi++) {
            uint32_t opp = (oi < (int)traverser) ? (uint32_t)oi : (uint32_t)(oi + 1);
            opp_reach_all[oi * nh + h] = initial_weight[opp * nh + h];
        }
    }

    const uint16_t* current_hr = hand_ranks;
    const uint16_t* active_opp_str = sorted_opp_strength;
    const uint16_t* active_opp_idx = sorted_opp_indices;
    const uint16_t* active_pl_str = sorted_player_strength;
    const uint16_t* active_pl_idx = sorted_player_indices;

    struct Frame {
        uint32_t node_idx;
        FrameState state;
        int num_actions;
        int child_idx;
        const uint16_t* saved_hr;
        const uint16_t* saved_sorted_opp_str;
        const uint16_t* saved_sorted_opp_idx;
        const uint16_t* saved_sorted_pl_str;
        const uint16_t* saved_sorted_pl_idx;
        uint8_t chance_card;
        int chance_deck_idx;
        int saved_deck_size;
    };

    Frame stack[MAX_DEPTH];
    int sp = 0;

    float returned_cfv[MAX_NH];
    for (int h = 0; h < nh; h++) returned_cfv[h] = 0.0f;

    float* my_frames = frame_data + tid * MAX_DEPTH * FRAME_STRIDE;

    stack[sp].node_idx = 0;
    stack[sp].state = STATE_ENTER;
    sp++;

    while (sp > 0) {
        const int cur = sp - 1;
        Frame& f = stack[cur];
        float* f_strat = my_frames + cur * FRAME_STRIDE + FRAME_STRAT_OFF;
        float* f_cfv   = my_frames + cur * FRAME_STRIDE + FRAME_CFV_OFF;
        float* f_reach = my_frames + cur * FRAME_STRIDE + FRAME_REACH_OFF;
        const FlatNode& node = nodes[f.node_idx];

        if (f.state == STATE_ENTER) {
            if (node.node_type == NODE_TYPE_TERMINAL) {
                for (int h = 0; h < nh; h++) returned_cfv[h] = 0.0f;

                int32_t c_t = contributions[f.node_idx * num_players + traverser];
                bool all_equal = true;
                for (uint32_t p = 0; p < num_players; p++) {
                    if (p == traverser) continue;
                    int32_t cp = contributions[f.node_idx * num_players + p];
                    if (cp != c_t) { all_equal = false; break; }
                }

                if (all_equal) {
                    sorted_sweep_showdown(opp_reach_all, num_opp, nh,
                                          active_opp_str, active_opp_idx,
                                          active_pl_str, active_pl_idx,
                                          hand_cards,
                                          contributions, f.node_idx,
                                          traverser, num_players,
                                          returned_cfv);
                } else {
                    int32_t min_contrib = 0x7fffffff;
                    for (uint32_t p = 0; p < num_players; p++) {
                        int32_t cp = contributions[f.node_idx * num_players + p];
                        if (cp < min_contrib) min_contrib = cp;
                    }

                    float payoff;
                    if (c_t == min_contrib) {
                        payoff = -(float)c_t;
                    } else {
                        int32_t pot = 0;
                        for (uint32_t p = 0; p < num_players; p++) {
                            pot += contributions[f.node_idx * num_players + p];
                        }
                        payoff = (float)(pot - c_t);
                    }

                    float opp_reach_sum = 0.0f;
                    float opp_reach_minus[52];
                    for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;

                    for (int oi = 0; oi < num_opp; oi++) {
                        const float* opp_reach = opp_reach_all + oi * nh;
                        for (int ho = 0; ho < nh; ho++) {
                            float r = opp_reach[ho];
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
                            returned_cfv[h] = payoff * cfreach;
                        }
                    }
                }
                sp--;
                continue;
            }

            if (node.node_type == NODE_TYPE_CHANCE) {
                rng = rng * 1103515245 + 12345;
                {
                    int card_idx = (rng >> 16) % num_remaining;
                    uint8_t sampled = remaining_deck[card_idx];

                    f.saved_sorted_opp_str = active_opp_str;
                    f.saved_sorted_opp_idx = active_opp_idx;
                    f.saved_sorted_pl_str = active_pl_str;
                    f.saved_sorted_pl_idx = active_pl_idx;

                    if (node.board_state == BOARD_STATE_RIVER && chance_sorted_strength != nullptr) {
                        int opp_stride = num_opp * nh;
                        active_opp_str = &chance_sorted_strength[sampled * opp_stride];
                        active_opp_idx = &chance_sorted_indices[sampled * opp_stride];
                        active_pl_str = &chance_sorted_strength[sampled * opp_stride];
                        active_pl_idx = &chance_sorted_indices[sampled * opp_stride];
                    }
                }

                f.state = STATE_RETURN;
                stack[sp].node_idx = children[node.children_start + 0];
                stack[sp].state = STATE_ENTER;
                sp++;
                continue;
            }

            uint32_t player = node.player_id;
            int na = (int)node.num_children;
            uint32_t offset = node_offsets[f.node_idx];

            compute_strategy_nplayer(regrets, offset, na, nh, f_strat);
            f.num_actions = na;
            f.child_idx = 0;
            for (int i = 0; i < na * nh; i++) f_cfv[i] = 0.0f;

            float* reach_to_modify;
            if (player == traverser) {
                reach_to_modify = treach_buf;
            } else {
                int oi = (player < traverser) ? (int)player : (int)(player - 1);
                reach_to_modify = opp_reach_all + oi * nh;
            }
            for (int h = 0; h < nh; h++) {
                f_reach[h] = reach_to_modify[h];
                reach_to_modify[h] *= f_strat[0 * nh + h];
            }

            f.state = STATE_RETURN;
            stack[sp].node_idx = children[node.children_start + 0];
            stack[sp].state = STATE_ENTER;
            sp++;
            continue;
        }

        if (f.state == STATE_RETURN) {
            const FlatNode& node2 = nodes[f.node_idx];

            if (node2.node_type == NODE_TYPE_CHANCE) {
                active_opp_str = f.saved_sorted_opp_str;
                active_opp_idx = f.saved_sorted_opp_idx;
                active_pl_str = f.saved_sorted_pl_str;
                active_pl_idx = f.saved_sorted_pl_idx;
                sp--;
                continue;
            }

            uint32_t player = node2.player_id;
            uint32_t offset = node_offsets[f.node_idx];

            for (int h = 0; h < nh; h++) {
                f_cfv[f.child_idx * nh + h] = returned_cfv[h];
            }

            f.child_idx++;
            if (f.child_idx < f.num_actions) {
                float* reach_to_modify;
                if (player == traverser) {
                    reach_to_modify = treach_buf;
                } else {
                    int oi = (player < traverser) ? (int)player : (int)(player - 1);
                    reach_to_modify = opp_reach_all + oi * nh;
                }
                for (int h = 0; h < nh; h++) {
                    reach_to_modify[h] = f_reach[h] * f_strat[f.child_idx * nh + h];
                }

                stack[sp].node_idx = children[node2.children_start + f.child_idx];
                stack[sp].state = STATE_ENTER;
                sp++;
                continue;
            }

            float* reach_to_restore;
            if (player == traverser) {
                reach_to_restore = treach_buf;
            } else {
                int oi = (player < traverser) ? (int)player : (int)(player - 1);
                reach_to_restore = opp_reach_all + oi * nh;
            }
            for (int h = 0; h < nh; h++) reach_to_restore[h] = f_reach[h];

            if (player == traverser) {
                for (int h = 0; h < nh; h++) {
                    float avg = 0.0f;
                    for (int a = 0; a < f.num_actions; a++) {
                        avg += f_strat[a * nh + h] * f_cfv[a * nh + h];
                    }
                    returned_cfv[h] = avg;
                }

                for (int h = 0; h < nh; h++) {
                    for (int a = 0; a < f.num_actions; a++) {
                        float inst_regret = f_cfv[a * nh + h] - returned_cfv[h];
                        uint32_t idx = offset + a * nh + h;
                        atomicAdd(&regrets[idx], weight * inst_regret);
                    }
                }

                for (int a2 = 0; a2 < f.num_actions; a2++) {
                    for (int h = 0; h < nh; h++) {
                        uint32_t ridx = offset + a2 * nh + h;
                        if (regrets[ridx] < regret_floor) regrets[ridx] = regret_floor;
                    }
                }

                for (int h = 0; h < nh; h++) {
                    for (int a = 0; a < f.num_actions; a++) {
                        uint32_t idx = offset + a * nh + h;
                        atomicAdd(&cum_strategy[idx], weight * treach_buf[h] * f_strat[a * nh + h]);
                    }
                }
            } else {
                for (int h = 0; h < nh; h++) {
                    returned_cfv[h] = 0.0f;
                    for (int a = 0; a < f.num_actions; a++) {
                        returned_cfv[h] += f_cfv[a * nh + h];
                    }
                }
            }

            sp--;
            continue;
        }
    }
}

extern "C" __global__ void mccfr_nplayer_extsamp(
    const FlatNode* nodes, const uint32_t* children,
    const int32_t* contributions, float* regrets, float* cum_strategy,
    const uint32_t* node_offsets,
    const uint16_t* hand_ranks,
    const uint16_t* sorted_opp_strength,
    const uint16_t* sorted_opp_indices,
    const uint16_t* sorted_player_strength,
    const uint16_t* sorted_player_indices,
    const uint16_t* same_hand_idx,
    const uint8_t* hand_cards,
    const float* initial_weight,
    const uint16_t* chance_ranks_table,
    const uint8_t* remaining_deck,
    const uint16_t* chance_sorted_strength,
    const uint16_t* chance_sorted_indices,
    float* frame_data,
    uint32_t num_players, uint32_t batch_size,
    int32_t nh, int32_t num_remaining,
    float weight, float regret_floor, const uint32_t* seeds
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size) return;

    uint32_t rng = seeds[tid];
    rng = rng * 1103515245 + 12345;
    uint32_t traverser = (rng >> 16) % num_players;

    int num_opp = (int)num_players - 1;

    float opp_reach_all[MAX_NP * MAX_NH];
    float treach_buf[MAX_NH];
    for (int h = 0; h < nh; h++) {
        treach_buf[h] = initial_weight[traverser * nh + h];
        for (int oi = 0; oi < num_opp; oi++) {
            uint32_t opp = (oi < (int)traverser) ? (uint32_t)oi : (uint32_t)(oi + 1);
            opp_reach_all[oi * nh + h] = initial_weight[opp * nh + h];
        }
    }

    const uint16_t* current_hr = hand_ranks;
    const uint16_t* active_opp_str = sorted_opp_strength;
    const uint16_t* active_opp_idx = sorted_opp_indices;
    const uint16_t* active_pl_str = sorted_player_strength;
    const uint16_t* active_pl_idx = sorted_player_indices;

    struct Frame {
        uint32_t node_idx;
        FrameState state;
        int num_actions;
        int child_idx;
        const uint16_t* saved_hr;
        const uint16_t* saved_sorted_opp_str;
        const uint16_t* saved_sorted_opp_idx;
        const uint16_t* saved_sorted_pl_str;
        const uint16_t* saved_sorted_pl_idx;
        uint8_t chance_card;
        int chance_deck_idx;
        int saved_deck_size;
        bool is_traverser;
        int sampled_action;
    };

    Frame stack[MAX_DEPTH];
    int sp = 0;

    float returned_cfv[MAX_NH];
    for (int h = 0; h < nh; h++) returned_cfv[h] = 0.0f;

    float* my_frames = frame_data + tid * MAX_DEPTH * FRAME_STRIDE;

    stack[sp].node_idx = 0;
    stack[sp].state = STATE_ENTER;
    sp++;

    while (sp > 0) {
        const int cur = sp - 1;
        Frame& f = stack[cur];
        float* f_strat = my_frames + cur * FRAME_STRIDE + FRAME_STRAT_OFF;
        float* f_cfv   = my_frames + cur * FRAME_STRIDE + FRAME_CFV_OFF;
        float* f_reach = my_frames + cur * FRAME_STRIDE + FRAME_REACH_OFF;
        const FlatNode& node = nodes[f.node_idx];

        if (f.state == STATE_ENTER) {
            if (node.node_type == NODE_TYPE_TERMINAL) {
                for (int h = 0; h < nh; h++) returned_cfv[h] = 0.0f;

                int32_t c_t = contributions[f.node_idx * num_players + traverser];
                bool all_equal = true;
                for (uint32_t p = 0; p < num_players; p++) {
                    if (p == traverser) continue;
                    int32_t cp = contributions[f.node_idx * num_players + p];
                    if (cp != c_t) { all_equal = false; break; }
                }

                if (all_equal) {
                    sorted_sweep_showdown(opp_reach_all, num_opp, nh,
                                          active_opp_str, active_opp_idx,
                                          active_pl_str, active_pl_idx,
                                          hand_cards,
                                          contributions, f.node_idx,
                                          traverser, num_players,
                                          returned_cfv);
                } else {
                    int32_t min_contrib = 0x7fffffff;
                    for (uint32_t p = 0; p < num_players; p++) {
                        int32_t cp = contributions[f.node_idx * num_players + p];
                        if (cp < min_contrib) min_contrib = cp;
                    }

                    float payoff;
                    if (c_t == min_contrib) {
                        payoff = -(float)c_t;
                    } else {
                        int32_t pot = 0;
                        for (uint32_t p = 0; p < num_players; p++) {
                            pot += contributions[f.node_idx * num_players + p];
                        }
                        payoff = (float)(pot - c_t);
                    }

                    float opp_reach_sum = 0.0f;
                    float opp_reach_minus[52];
                    for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;

                    for (int oi = 0; oi < num_opp; oi++) {
                        const float* opp_reach = opp_reach_all + oi * nh;
                        for (int ho = 0; ho < nh; ho++) {
                            float r = opp_reach[ho];
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
                            returned_cfv[h] = payoff * cfreach;
                        }
                    }
                }
                sp--;
                continue;
            }

            if (node.node_type == NODE_TYPE_CHANCE) {
                rng = rng * 1103515245 + 12345;
                {
                    int card_idx = (rng >> 16) % num_remaining;
                    uint8_t sampled = remaining_deck[card_idx];

                    f.saved_sorted_opp_str = active_opp_str;
                    f.saved_sorted_opp_idx = active_opp_idx;
                    f.saved_sorted_pl_str = active_pl_str;
                    f.saved_sorted_pl_idx = active_pl_idx;

                    if (node.board_state == BOARD_STATE_RIVER && chance_sorted_strength != nullptr) {
                        int opp_stride = num_opp * nh;
                        active_opp_str = &chance_sorted_strength[sampled * opp_stride];
                        active_opp_idx = &chance_sorted_indices[sampled * opp_stride];
                        active_pl_str = &chance_sorted_strength[sampled * opp_stride];
                        active_pl_idx = &chance_sorted_indices[sampled * opp_stride];
                    }
                }

                f.state = STATE_RETURN;
                stack[sp].node_idx = children[node.children_start + 0];
                stack[sp].state = STATE_ENTER;
                sp++;
                continue;
            }

            uint32_t player = node.player_id;
            int na = (int)node.num_children;
            uint32_t offset = node_offsets[f.node_idx];
            bool is_trav = (player == traverser);

            compute_strategy_nplayer(regrets, offset, na, nh, f_strat);
            f.num_actions = na;
            f.is_traverser = is_trav;

            if (is_trav) {
                f.child_idx = 0;
                for (int i = 0; i < na * nh; i++) f_cfv[i] = 0.0f;

                for (int h = 0; h < nh; h++) {
                    f_reach[h] = treach_buf[h];
                    treach_buf[h] *= f_strat[0 * nh + h];
                }

                f.state = STATE_RETURN;
                stack[sp].node_idx = children[node.children_start + 0];
                stack[sp].state = STATE_ENTER;
                sp++;
            } else {
                rng = rng * 1103515245 + 12345;
                int sampled_a = (int)((rng >> 16) % (uint32_t)na);
                f.sampled_action = sampled_a;

                int oi = (player < traverser) ? (int)player : (int)(player - 1);
                float* reach_arr = opp_reach_all + oi * nh;
                for (int h = 0; h < nh; h++) {
                    f_reach[h] = reach_arr[h];
                    reach_arr[h] *= f_strat[sampled_a * nh + h];
                }

                f.state = STATE_RETURN;
                stack[sp].node_idx = children[node.children_start + sampled_a];
                stack[sp].state = STATE_ENTER;
                sp++;
            }
            continue;
        }

        if (f.state == STATE_RETURN) {
            const FlatNode& node2 = nodes[f.node_idx];

            if (node2.node_type == NODE_TYPE_CHANCE) {
                active_opp_str = f.saved_sorted_opp_str;
                active_opp_idx = f.saved_sorted_opp_idx;
                active_pl_str = f.saved_sorted_pl_str;
                active_pl_idx = f.saved_sorted_pl_idx;
                sp--;
                continue;
            }

            uint32_t player = node2.player_id;
            uint32_t offset = node_offsets[f.node_idx];

            if (f.is_traverser) {
                for (int h = 0; h < nh; h++) {
                    f_cfv[f.child_idx * nh + h] = returned_cfv[h];
                }

                f.child_idx++;
                if (f.child_idx < f.num_actions) {
                    for (int h = 0; h < nh; h++) {
                        treach_buf[h] = f_reach[h] * f_strat[f.child_idx * nh + h];
                    }

                    stack[sp].node_idx = children[node2.children_start + f.child_idx];
                    stack[sp].state = STATE_ENTER;
                    sp++;
                    continue;
                }

                for (int h = 0; h < nh; h++) treach_buf[h] = f_reach[h];

                for (int h = 0; h < nh; h++) {
                    float avg = 0.0f;
                    for (int a = 0; a < f.num_actions; a++) {
                        avg += f_strat[a * nh + h] * f_cfv[a * nh + h];
                    }
                    returned_cfv[h] = avg;
                }

                for (int h = 0; h < nh; h++) {
                    for (int a = 0; a < f.num_actions; a++) {
                        float inst_regret = f_cfv[a * nh + h] - returned_cfv[h];
                        uint32_t idx = offset + a * nh + h;
                        atomicAdd(&regrets[idx], weight * inst_regret);
                    }
                }

                for (int a2 = 0; a2 < f.num_actions; a2++) {
                    for (int h = 0; h < nh; h++) {
                        uint32_t ridx = offset + a2 * nh + h;
                        if (regrets[ridx] < regret_floor) regrets[ridx] = regret_floor;
                    }
                }

                for (int h = 0; h < nh; h++) {
                    for (int a = 0; a < f.num_actions; a++) {
                        uint32_t idx = offset + a * nh + h;
                        atomicAdd(&cum_strategy[idx], weight * treach_buf[h] * f_strat[a * nh + h]);
                    }
                }
            } else {
                int na = f.num_actions;
                float importance_weight = (float)na;
                for (int h = 0; h < nh; h++) {
                    returned_cfv[h] *= importance_weight;
                }

                int oi = (player < traverser) ? (int)player : (int)(player - 1);
                float* reach_arr = opp_reach_all + oi * nh;
                for (int h = 0; h < nh; h++) reach_arr[h] = f_reach[h];
            }

            sp--;
            continue;
        }
    }
}

// --- Compact external sampling kernel ---
// Traverser frames: strategy[NA*NH] + cfv_actions[NA*NH] + saved_reach[NH] = FRAME_STRIDE
// Opponent frames:  strategy[NA*NH] + saved_reach[NH] = OPP_FRAME_STRIDE (no cfv_actions)
// Cursor-based allocation: frames only occupy what they need, cursor resets on pop.
// Per-trajectory buffer: MAX_FRAME_DATA = MAX_DEPTH * FRAME_STRIDE floats.

#define OPP_FRAME_STRIDE (MAX_NA * MAX_NH + MAX_NH)
#define OPP_FRAME_STRAT_OFF 0
#define OPP_FRAME_REACH_OFF (MAX_NA * MAX_NH)
#define MAX_FRAME_DATA_COMPACT (MAX_DEPTH * FRAME_STRIDE)

extern "C" __global__ void mccfr_nplayer_extsamp_compact(
    const FlatNode* nodes, const uint32_t* children,
    const int32_t* contributions, const uint16_t* folded_masks,
    float* regrets, float* cum_strategy,
    const uint32_t* node_offsets,
    const uint16_t* hand_ranks,
    const uint16_t* sorted_opp_strength,
    const uint16_t* sorted_opp_indices,
    const uint16_t* sorted_player_strength,
    const uint16_t* sorted_player_indices,
    const uint16_t* same_hand_idx,
    const uint8_t* hand_cards,
    const float* initial_weight,
    const uint16_t* chance_ranks_table,
    const uint8_t* remaining_deck,
    const uint16_t* chance_sorted_strength,
    const uint16_t* chance_sorted_indices,
    float* frame_data,
    uint32_t num_players, uint32_t batch_size,
    int32_t nh, int32_t num_remaining,
    float weight, float regret_floor, const uint32_t* seeds,
    uint32_t* peak_cursor_out,
    uint32_t frame_stride_per_traj
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size) return;

    uint32_t rng = seeds[tid];
    rng = rng * 1103515245 + 12345;
    uint32_t traverser = (rng >> 16) % num_players;

    int num_opp = (int)num_players - 1;

    float opp_reach_all[MAX_NP * MAX_NH];
    float treach_buf[MAX_NH];
    for (int h = 0; h < nh; h++) {
        treach_buf[h] = initial_weight[traverser * nh + h];
        for (int oi = 0; oi < num_opp; oi++) {
            uint32_t opp = (oi < (int)traverser) ? (uint32_t)oi : (uint32_t)(oi + 1);
            opp_reach_all[oi * nh + h] = initial_weight[opp * nh + h];
        }
    }

    const uint16_t* active_opp_str = sorted_opp_strength;
    const uint16_t* active_opp_idx = sorted_opp_indices;
    const uint16_t* active_pl_str = sorted_player_strength;
    const uint16_t* active_pl_idx = sorted_player_indices;

    struct Frame {
        uint32_t node_idx;
        FrameState state;
        int num_actions;
        int child_idx;
        bool is_traverser;
        int sampled_action;
        uint32_t frame_offset;
        const uint16_t* saved_sorted_opp_str;
        const uint16_t* saved_sorted_opp_idx;
        const uint16_t* saved_sorted_pl_str;
        const uint16_t* saved_sorted_pl_idx;
        uint8_t chance_card;
    };

    Frame stack[MAX_DEPTH];
    int sp = 0;

    float returned_cfv[MAX_NH];
    for (int h = 0; h < nh; h++) returned_cfv[h] = 0.0f;

    float* my_base = frame_data + tid * frame_stride_per_traj;
    uint32_t cursor = 0;
    uint32_t peak_cursor = 0;

    stack[sp].node_idx = 0;
    stack[sp].state = STATE_ENTER;
    sp++;

    while (sp > 0) {
        const int cur = sp - 1;
        Frame& f = stack[cur];
        const FlatNode& node = nodes[f.node_idx];

        if (f.state == STATE_ENTER) {
            if (node.node_type == NODE_TYPE_TERMINAL) {
                for (int h = 0; h < nh; h++) returned_cfv[h] = 0.0f;

                int32_t c_t = contributions[f.node_idx * num_players + traverser];
                uint16_t fold_mask = folded_masks[f.node_idx];

                int num_active_players = 0;
                for (uint32_t p = 0; p < num_players; p++) {
                    if (!(fold_mask & (1u << p))) num_active_players++;
                }

                bool all_active_equal = true;
                if (num_active_players > 1) {
                    int32_t ref = -1;
                    for (uint32_t p = 0; p < num_players; p++) {
                        if (fold_mask & (1u << p)) continue;
                        int32_t cp = contributions[f.node_idx * num_players + p];
                        if (ref < 0) ref = cp;
                        else if (cp != ref) { all_active_equal = false; break; }
                    }
                }

                if (num_active_players <= 1 || (fold_mask & (1u << traverser))) {
                    int32_t pot = 0;
                    for (uint32_t p = 0; p < num_players; p++) {
                        pot += contributions[f.node_idx * num_players + p];
                    }
                    float payoff;
                    if (fold_mask & (1u << traverser)) {
                        payoff = -(float)c_t;
                    } else {
                        payoff = (float)(pot - c_t);
                    }

                    float opp_reach_sum = 0.0f;
                    float opp_reach_minus[52];
                    for (int c = 0; c < 52; c++) opp_reach_minus[c] = 0.0f;

                    for (int oi = 0; oi < num_opp; oi++) {
                        const float* opp_reach = opp_reach_all + oi * nh;
                        for (int ho = 0; ho < nh; ho++) {
                            float r = opp_reach[ho];
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
                            returned_cfv[h] = payoff * cfreach;
                        }
                    }
                } else if (all_active_equal) {
                    float amount_win = (float)c_t;
                    float amount_lose = -(float)c_t;

                    for (int oi = 0; oi < num_opp; oi++) {
                        int opp_p = (oi < (int)traverser) ? oi : (oi + 1);
                        if (fold_mask & (1u << opp_p)) continue;

                        const uint16_t* opp_str = active_opp_str + oi * nh;
                        const uint16_t* opp_idx = active_opp_idx + oi * nh;
                        const float* opp_reach = opp_reach_all + oi * nh;

                        float cfreach_sum = 0.0f;
                        float cfreach_minus[52];
                        for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

                        int ii = 0;
                        for (int si = 0; si < nh; si++) {
                            uint16_t str_h = active_pl_str[si];
                            uint16_t h = active_pl_idx[si];
                            while (ii < nh && opp_str[ii] < str_h) {
                                int ho = opp_idx[ii];
                                float r = opp_reach[ho];
                                if (r != 0.0f) {
                                    cfreach_sum += r;
                                    cfreach_minus[hand_cards[ho * 2]] += r;
                                    cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                                }
                                ii++;
                            }
                            float cfreach = cfreach_sum
                                - cfreach_minus[hand_cards[h * 2]]
                                - cfreach_minus[hand_cards[h * 2 + 1]];
                            returned_cfv[h] += amount_win * cfreach;
                        }

                        cfreach_sum = 0.0f;
                        for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

                        ii = nh - 1;
                        for (int si = nh - 1; si >= 0; si--) {
                            uint16_t str_h = active_pl_str[si];
                            uint16_t h = active_pl_idx[si];
                            while (ii >= 0 && opp_str[ii] > str_h) {
                                int ho = opp_idx[ii];
                                float r = opp_reach[ho];
                                if (r != 0.0f) {
                                    cfreach_sum += r;
                                    cfreach_minus[hand_cards[ho * 2]] += r;
                                    cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                                }
                                ii--;
                            }
                            float cfreach = cfreach_sum
                                - cfreach_minus[hand_cards[h * 2]]
                                - cfreach_minus[hand_cards[h * 2 + 1]];
                            returned_cfv[h] += amount_lose * cfreach;
                        }
                    }
                } else {
                        int32_t levels[MAX_NP];
                        int num_levels = 0;
                        {
                            int32_t tmp[MAX_NP];
                            int nt = 0;
                            for (uint32_t p = 0; p < num_players; p++) {
                                int32_t cp = contributions[f.node_idx * num_players + p];
                                if (cp > 0) { tmp[nt++] = cp; }
                            }
                            for (int i = 1; i < nt; i++) {
                                int32_t key = tmp[i];
                                int j = i - 1;
                                while (j >= 0 && tmp[j] > key) { tmp[j+1] = tmp[j]; j--; }
                                tmp[j+1] = key;
                            }
                            levels[0] = tmp[0];
                            num_levels = 1;
                            for (int i = 1; i < nt; i++) {
                                if (tmp[i] != levels[num_levels-1]) {
                                    levels[num_levels++] = tmp[i];
                                }
                            }
                        }

                        int32_t prev_level = 0;
                        for (int li = 0; li < num_levels; li++) {
                            int32_t level = levels[li];
                            int32_t pot_contrib = level - prev_level;
                            if (pot_contrib <= 0) { prev_level = level; continue; }

                            int num_eligible = 0;
                            int eligible_oi[MAX_NP];
                            int num_eligible_opp = 0;
                            bool trav_eligible = (c_t >= level) && !(fold_mask & (1u << traverser));

                            for (uint32_t p = 0; p < num_players; p++) {
                                if (fold_mask & (1u << p)) continue;
                                int32_t cp = contributions[f.node_idx * num_players + p];
                                if (cp >= level) {
                                    num_eligible++;
                                    if (p != traverser) {
                                        int oi = (p < traverser) ? (int)p : (int)(p - 1);
                                        eligible_oi[num_eligible_opp++] = oi;
                                    }
                                }
                            }

                            int num_contributors = 0;
                            for (uint32_t p = 0; p < num_players; p++) {
                                if (contributions[f.node_idx * num_players + p] >= level) {
                                    num_contributors++;
                                }
                            }

                            float pot_at_level = (float)(pot_contrib * num_contributors);

                            if (trav_eligible && num_eligible_opp > 0) {
                                for (int eoi = 0; eoi < num_eligible_opp; eoi++) {
                                    int oi = eligible_oi[eoi];
                                    const float* opp_reach = opp_reach_all + oi * nh;
                                    const uint16_t* o_str = active_opp_str + oi * nh;
                                    const uint16_t* o_idx = active_opp_idx + oi * nh;

                                    float cfreach_sum = 0.0f;
                                    float cfreach_minus[52];
                                    for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

                                    int idx = 0;
                                    for (int si = 0; si < nh; si++) {
                                        uint16_t str_h = active_pl_str[si];
                                        uint16_t h = active_pl_idx[si];
                                        while (idx < nh && o_str[idx] < str_h) {
                                            int ho = o_idx[idx];
                                            float r = opp_reach[ho];
                                            if (r != 0.0f) {
                                                cfreach_sum += r;
                                                cfreach_minus[hand_cards[ho * 2]] += r;
                                                cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                                            }
                                            idx++;
                                        }
                                        float cfreach = cfreach_sum
                                            - cfreach_minus[hand_cards[h * 2]]
                                            - cfreach_minus[hand_cards[h * 2 + 1]];
                                        returned_cfv[h] += pot_at_level * cfreach;
                                    }

                                    cfreach_sum = 0.0f;
                                    for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

                                    idx = nh - 1;
                                    for (int si = nh - 1; si >= 0; si--) {
                                        uint16_t str_h = active_pl_str[si];
                                        uint16_t h = active_pl_idx[si];
                                        while (idx >= 0 && o_str[idx] > str_h) {
                                            int ho = o_idx[idx];
                                            float r = opp_reach[ho];
                                            if (r != 0.0f) {
                                                cfreach_sum += r;
                                                cfreach_minus[hand_cards[ho * 2]] += r;
                                                cfreach_minus[hand_cards[ho * 2 + 1]] += r;
                                            }
                                            idx--;
                                        }
                                        float cfreach = cfreach_sum
                                            - cfreach_minus[hand_cards[h * 2]]
                                            - cfreach_minus[hand_cards[h * 2 + 1]];
                                        returned_cfv[h] -= pot_at_level * cfreach;
                                    }
                                }
                            } else if (trav_eligible && num_eligible_opp == 0) {
                                for (int h = 0; h < nh; h++) {
                                    returned_cfv[h] += pot_at_level;
                                }
                            }

                            prev_level = level;
                        }

                        for (int h = 0; h < nh; h++) {
                            returned_cfv[h] -= (float)c_t;
                        }
                    }
                sp--;
                continue;
            }

            if (node.node_type == NODE_TYPE_CHANCE) {
                rng = rng * 1103515245 + 12345;
                {
                    int card_idx = (rng >> 16) % num_remaining;
                    uint8_t sampled = remaining_deck[card_idx];

                    f.saved_sorted_opp_str = active_opp_str;
                    f.saved_sorted_opp_idx = active_opp_idx;
                    f.saved_sorted_pl_str = active_pl_str;
                    f.saved_sorted_pl_idx = active_pl_idx;

                    if (node.board_state == BOARD_STATE_RIVER && chance_sorted_strength != nullptr) {
                        int opp_stride = num_opp * nh;
                        active_opp_str = &chance_sorted_strength[sampled * opp_stride];
                        active_opp_idx = &chance_sorted_indices[sampled * opp_stride];
                        active_pl_str = &chance_sorted_strength[sampled * opp_stride];
                        active_pl_idx = &chance_sorted_indices[sampled * opp_stride];
                    }
                }
 

                f.state = STATE_RETURN;
                stack[sp].node_idx = children[node.children_start + 0];
                stack[sp].state = STATE_ENTER;
                sp++;
                continue;
            }

            uint32_t player = node.player_id;
            int na = (int)node.num_children;
            uint32_t offset = node_offsets[f.node_idx];
            bool is_trav = (player == traverser);

            f.num_actions = na;
            f.is_traverser = is_trav;

            if (is_trav) {
                f.frame_offset = cursor;
                cursor += FRAME_STRIDE;
                if (cursor > peak_cursor) peak_cursor = cursor;
                float* f_strat = my_base + f.frame_offset + FRAME_STRAT_OFF;
                float* f_cfv   = my_base + f.frame_offset + FRAME_CFV_OFF;
                float* f_reach = my_base + f.frame_offset + FRAME_REACH_OFF;

                compute_strategy_nplayer(regrets, offset, na, nh, f_strat);
                f.child_idx = 0;
                for (int i = 0; i < na * nh; i++) f_cfv[i] = 0.0f;

                for (int h = 0; h < nh; h++) {
                    f_reach[h] = treach_buf[h];
                    treach_buf[h] *= f_strat[0 * nh + h];
                }

                f.state = STATE_RETURN;
                stack[sp].node_idx = children[node.children_start + 0];
                stack[sp].state = STATE_ENTER;
                sp++;
            } else {
                f.frame_offset = cursor;
                cursor += OPP_FRAME_STRIDE;
                if (cursor > peak_cursor) peak_cursor = cursor;
                float* f_strat = my_base + f.frame_offset + OPP_FRAME_STRAT_OFF;
                float* f_reach = my_base + f.frame_offset + OPP_FRAME_REACH_OFF;

                compute_strategy_nplayer(regrets, offset, na, nh, f_strat);

                rng = rng * 1103515245 + 12345;
                int sampled_a = (int)((rng >> 16) % (uint32_t)na);
                f.sampled_action = sampled_a;

                int oi = (player < traverser) ? (int)player : (int)(player - 1);
                float* reach_arr = opp_reach_all + oi * nh;
                for (int h = 0; h < nh; h++) {
                    f_reach[h] = reach_arr[h];
                    reach_arr[h] *= f_strat[sampled_a * nh + h];
                }

                f.state = STATE_RETURN;
                stack[sp].node_idx = children[node.children_start + sampled_a];
                stack[sp].state = STATE_ENTER;
                sp++;
            }
            continue;
        }

        if (f.state == STATE_RETURN) {
            const FlatNode& node2 = nodes[f.node_idx];

            if (node2.node_type == NODE_TYPE_CHANCE) {
                active_opp_str = f.saved_sorted_opp_str;
                active_opp_idx = f.saved_sorted_opp_idx;
                active_pl_str = f.saved_sorted_pl_str;
                active_pl_idx = f.saved_sorted_pl_idx;
                sp--;
                continue;
            }

            uint32_t player = node2.player_id;
            uint32_t offset = node_offsets[f.node_idx];

            if (f.is_traverser) {
                float* f_strat = my_base + f.frame_offset + FRAME_STRAT_OFF;
                float* f_cfv   = my_base + f.frame_offset + FRAME_CFV_OFF;
                float* f_reach = my_base + f.frame_offset + FRAME_REACH_OFF;

                for (int h = 0; h < nh; h++) {
                    f_cfv[f.child_idx * nh + h] = returned_cfv[h];
                }

                f.child_idx++;
                if (f.child_idx < f.num_actions) {
                    for (int h = 0; h < nh; h++) {
                        treach_buf[h] = f_reach[h] * f_strat[f.child_idx * nh + h];
                    }

                    stack[sp].node_idx = children[node2.children_start + f.child_idx];
                    stack[sp].state = STATE_ENTER;
                    sp++;
                    continue;
                }

                cursor = f.frame_offset;

                for (int h = 0; h < nh; h++) treach_buf[h] = f_reach[h];

                for (int h = 0; h < nh; h++) {
                    float avg = 0.0f;
                    for (int a = 0; a < f.num_actions; a++) {
                        avg += f_strat[a * nh + h] * f_cfv[a * nh + h];
                    }
                    returned_cfv[h] = avg;
                }

                for (int h = 0; h < nh; h++) {
                    for (int a = 0; a < f.num_actions; a++) {
                        float inst_regret = f_cfv[a * nh + h] - returned_cfv[h];
                        uint32_t idx = offset + a * nh + h;
                        atomicAdd(&regrets[idx], weight * inst_regret);
                    }
                }

                for (int a2 = 0; a2 < f.num_actions; a2++) {
                    for (int h = 0; h < nh; h++) {
                        uint32_t ridx = offset + a2 * nh + h;
                        if (regrets[ridx] < regret_floor) regrets[ridx] = regret_floor;
                    }
                }

                for (int h = 0; h < nh; h++) {
                    for (int a = 0; a < f.num_actions; a++) {
                        uint32_t idx = offset + a * nh + h;
                        atomicAdd(&cum_strategy[idx], weight * treach_buf[h] * f_strat[a * nh + h]);
                    }
                }
            } else {
                float* f_strat = my_base + f.frame_offset + OPP_FRAME_STRAT_OFF;
                float* f_reach = my_base + f.frame_offset + OPP_FRAME_REACH_OFF;

                int na = f.num_actions;
                float importance_weight = (float)na;
                for (int h = 0; h < nh; h++) {
                    returned_cfv[h] *= importance_weight;
                }

                cursor = f.frame_offset;

                int oi = (player < traverser) ? (int)player : (int)(player - 1);
                float* reach_arr = opp_reach_all + oi * nh;
                for (int h = 0; h < nh; h++) reach_arr[h] = f_reach[h];
            }

            sp--;
            continue;
        }
    }

    if (peak_cursor_out != nullptr) {
        peak_cursor_out[tid] = peak_cursor;
    }
}
