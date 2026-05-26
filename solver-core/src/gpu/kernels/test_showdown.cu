#include <cstdint>

#define MAX_NH 1326

extern "C" __global__ void test_showdown(
    const uint16_t* sorted_opp_strength,
    const uint16_t* sorted_opp_indices,
    const uint16_t* sorted_player_strength,
    const uint16_t* sorted_player_indices,
    const uint8_t* hand_cards,
    const float* opp_reach,
    int32_t nh,
    float contribution,
    float* output
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= 1) return;

    float amount_win = contribution;
    float amount_lose = -contribution;

    float cfreach_sum = 0.0f;
    float cfreach_minus[52];
    for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

    // Ascending pass
    int i = 0;
    for (int si = 0; si < nh; si++) {
        uint16_t str_h = sorted_player_strength[si];
        uint16_t h = sorted_player_indices[si];
        while (i < nh && sorted_opp_strength[i] < str_h) {
            int ho = sorted_opp_indices[i];
            float r = opp_reach[ho];
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
        output[h] = amount_win * cfreach;
    }

    cfreach_sum = 0.0f;
    for (int c = 0; c < 52; c++) cfreach_minus[c] = 0.0f;

    // Descending pass
    i = nh - 1;
    for (int si = nh - 1; si >= 0; si--) {
        uint16_t str_h = sorted_player_strength[si];
        uint16_t h = sorted_player_indices[si];
        while (i >= 0 && sorted_opp_strength[i] > str_h) {
            int ho = sorted_opp_indices[i];
            float r = opp_reach[ho];
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
        output[h] += amount_lose * cfreach;
    }
}
