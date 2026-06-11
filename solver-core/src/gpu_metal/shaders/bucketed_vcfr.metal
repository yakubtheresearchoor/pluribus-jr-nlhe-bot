// Bucketed Design-1-collapsed terminal kernel (G2 step 3).
//
// One threadgroup per terminal node of the current zone walk
// (tg_size = 32, Fix-A). The grid covers all terminals of one
// (traverser, zone outcome) walk in a single dispatch (one GPU sync
// per zone walk — the Fix-B lesson applied at the offload boundary).
//
// ONE kernel, TWO dispatch configs (the G3 gate distinguishes configs,
// not code):
//   params.stripes == 1: the UNSTRIPED REFERENCE. Lanes tid < nb each
//     own traverser-bucket bt = tid and enumerate the full B^K tuple
//     space in exactly the CPU's order; no cross-lane reduction, so no
//     reorder — bit-exact-compatible with the CPU collapsed arm at
//     singletons (the middle link of the three-point gate chain).
//   params.stripes == S > 1: PRODUCTION. Lane tid = bt·S + s owns the
//     contiguous bo[0] range [s·nb/S, (s+1)·nb/S); partials reduce in
//     FIXED stripe order (s0+s1+…, manual, not simd_sum) so the result
//     is deterministic run-to-run; order differs from CPU → gated by
//     the f64-reference + trajectory standard (already ratified).
//
// Phases per threadgroup:
//   A. reduce: threads 0..num_opp each serially sum their opponent's
//      per-hand reach (board-filtered, h ascending — CPU order) into
//      bucket_reach_tg. Each element is read exactly once, so per-hand
//      reach never stages in threadgroup memory.
//   B. tables: the runout's four fraction tables → threadgroup (4 KB
//      at MAX_BUCKETS_GPU = 16).
//   C. enumeration: per-lane iterative odometer (the CPU recursion's
//      exact arithmetic; per-lane state < ~600 B — the 26 KB-spill
//      lesson honored at design time).
//   D. fixed-order stripe reduction → per-bucket cfv.
//   E. expansion: cfv[node·nh + h] = cfv_bucket[map[h]] (/nc), strided.
//
// Memory budget (threadgroup): bucket_reach 9·16·4 = 576 B + tables
// 4·16·16·4 = 4 KB + partials 32·4 = 128 B + cfv_bucket 16·4 = 64 B
// ≈ 4.8 KB. Per-lane: bo[9] + state 11·(9+1) + sc 9·4 + w[10] + dp[10]
// + level block ≈ < 700 B.

#pragma METAL fp contract(off)

#include <metal_stdlib>
#include "max_na_generated.metal"
#include "bucketed_generated.metal"
using namespace metal;

constant uint NO_BUCKET_GPU = 0xFFFFu;

struct BucketedTermParams {
    uint num_terminals;
    uint np;
    uint nh;
    uint nb;
    uint num_opp;
    uint stripes;       // 1 = unstriped reference; S>1 = production
    uint traverser;
    uint board0;        // dealt turn card or 0xFFFF (flop zone)
    uint board1;        // dealt river card or 0xFFFF
    uint table_off;     // float offset of this runout's [fw|ft|fl|fn] block
    uint map_off;       // u16 offset of this runout's hand->bucket map
    float nc;           // num_combinations (1.0 in production tables)
    int starting_pot;
    float rake_rate;
    float rake_cap;
    uint _pad;
};

// ── Arm-1 leaf: payoff constants, verbatim from recurse_eq_buckets ──
inline float arm1_leaf(thread const float* state, uint num_opp, float k,
                       float rake_per_unit_stake) {
    float accum = 0.0f;
    if (state[0] != 0.0f) {
        accum += state[0] * -1.0f;
    }
    for (uint j = 0; j <= num_opp; j++) {
        float s = state[1 + j];
        if (s == 0.0f) continue;
        float net_unit;
        if (j == 0) {
            net_unit = k - rake_per_unit_stake;
        } else {
            float t_f = float(j + 1);
            net_unit = (k + 1.0f - t_f) / t_f - rake_per_unit_stake / t_f;
        }
        accum += s * net_unit;
    }
    return accum;
}

// ── Arm-2 per-terminal level block (relation-independent precompute,
//    identical in every lane — verbatim Arm2Ctx fields) ──
struct LevelBlock {
    int n_levels;
    int levels[8];
    float pot_l[8];
    float pot_after_rake[8];
    bool trav_elig[8];
    float trav_return[8];   // elig_count==0 side-pot return (0 if none)
    uint elig_mask[8];      // bit oi set if opponent oi eligible at level
    float traverser_stake;
    float s_dummy;
};

inline LevelBlock build_level_block(
    thread const int* contribs, uint np, uint traverser, uint num_opp,
    thread const bool* opp_folded, thread const int* opp_contrib,
    int starting_pot, float eff_rake_rate, float eff_rake_cap, bool traverser_folded)
{
    LevelBlock lb;
    // levels = sorted dedup contributions
    int lv[8];
    int n = 0;
    for (uint p = 0; p < np; p++) {
        int c = contribs[p];
        bool seen = false;
        for (int i = 0; i < n; i++) if (lv[i] == c) { seen = true; break; }
        if (!seen) lv[n++] = c;
    }
    // insertion sort
    for (int i = 1; i < n; i++) {
        int key = lv[i];
        int j = i - 1;
        while (j >= 0 && lv[j] > key) { lv[j + 1] = lv[j]; j--; }
        lv[j + 1] = key;
    }
    lb.n_levels = n;
    int c_t = contribs[traverser];
    // main pot rake (site main-pot-only rule, verbatim)
    int main_pot_amount;
    {
        int nmc = 0;
        for (uint p = 0; p < np; p++) if (contribs[p] >= lv[0]) nmc++;
        main_pot_amount = lv[0] * nmc + starting_pot;
    }
    float main_pot_rake = max(min(float(main_pot_amount) * eff_rake_rate, eff_rake_cap), 0.0f);
    lb.traverser_stake = float(starting_pot) / float(np) + float(c_t);

    int prev_l = 0;
    for (int li = 0; li < n; li++) {
        int lev = lv[li];
        int pc = lev - prev_l;
        int num_contrib = 0;
        for (uint p = 0; p < np; p++) if (contribs[p] >= lev) num_contrib++;
        float pot_l = float(pc * num_contrib);
        if (li == 0) pot_l += float(starting_pot);
        lb.levels[li] = lev;
        lb.pot_l[li] = pot_l;
        bool trav_elig = !traverser_folded && c_t >= lev;
        lb.trav_elig[li] = trav_elig;
        uint emask = 0;
        uint elig_count = trav_elig ? 1u : 0u;
        for (uint oi = 0; oi < num_opp; oi++) {
            if (opp_folded[oi]) continue;
            if (opp_contrib[oi] < lev) continue;
            emask |= (1u << oi);
            elig_count++;
        }
        lb.elig_mask[li] = emask;
        lb.trav_return[li] = 0.0f;
        if (pot_l != 0.0f && elig_count == 0 && contribs[traverser] >= lev) {
            lb.trav_return[li] =
                float(pc) + (li == 0 ? float(starting_pot) / float(np) : 0.0f);
        }
        lb.pot_after_rake[li] = (li == 0) ? (pot_l - main_pot_rake) : pot_l;
        prev_l = lev;
    }
    return lb;
}

// ── Arm-2 leaf: net_expected, verbatim CPU arithmetic per tuple ──
inline float net_expected(
    thread const LevelBlock& lb, thread const float (*sc)[4],
    uint num_opp, thread const bool* opp_folded)
{
    // S_total = Π active (w+t+l) × Π folded n, in opponent order.
    float s_total = 1.0f;
    for (uint oi = 0; oi < num_opp; oi++) {
        if (opp_folded[oi]) {
            s_total *= sc[oi][3];
        } else {
            s_total *= sc[oi][0] + sc[oi][1] + sc[oi][2];
        }
    }
    float cash = 0.0f;
    float dp[MAX_OPP_BUCKETED + 1];
    for (int li = 0; li < lb.n_levels; li++) {
        float pot_l = lb.pot_l[li];
        if (pot_l == 0.0f) continue;
        uint emask = lb.elig_mask[li];
        bool trav_elig = lb.trav_elig[li];
        uint elig_count = (trav_elig ? 1u : 0u) + popcount(emask);
        if (elig_count == 0) {
            cash += lb.trav_return[li] * s_total;
            continue;
        }
        if (!trav_elig) continue;
        // m_out: active non-eligible (w+t+l) × folded n, opponent order.
        float m_out = 1.0f;
        for (uint oi = 0; oi < num_opp; oi++) {
            if (opp_folded[oi]) {
                m_out *= sc[oi][3];
                continue;
            }
            if ((emask & (1u << oi)) == 0) {
                m_out *= sc[oi][0] + sc[oi][1] + sc[oi][2];
            }
        }
        // dp[j] = P(no lose among eligible, j ties), in-place descending.
        dp[0] = 1.0f;
        uint ne = 0;
        for (uint oi = 0; oi < num_opp; oi++) {
            if ((emask & (1u << oi)) == 0) continue;
            float w = sc[oi][0];
            float t = sc[oi][1];
            dp[ne + 1] = 0.0f;
            for (int j = int(ne); j >= 0; j--) {
                float d = dp[j];
                if (d != 0.0f && t != 0.0f) dp[j + 1] += d * t;
                dp[j] = (d != 0.0f && w != 0.0f) ? d * w : 0.0f;
            }
            ne++;
        }
        float par = lb.pot_after_rake[li];
        for (uint j = 0; j <= ne; j++) {
            float d = dp[j];
            if (d == 0.0f) continue;
            float tied = float(j + 1);
            cash += m_out * d * (par / tied);
        }
    }
    return cash - lb.traverser_stake * s_total;
}

kernel void bucketed_terminal_collapsed(
    constant BucketedTermParams& params      [[buffer(0)]],
    const device uint*     term_node_ids     [[buffer(1)]],
    const device int*      term_contribs     [[buffer(2)]],  // [nt × np]
    const device uint*     term_fold_masks   [[buffer(3)]],  // [nt]
    const device float*    reach             [[buffer(4)]],  // [nn × np × nh]
    const device float*    fractions         [[buffer(5)]],  // concat runouts
    const device ushort*   bucket_map        [[buffer(6)]],  // concat runouts
    const device uchar*    hand_cards        [[buffer(7)]],  // [nh × 2]
    device float*          cfv               [[buffer(8)]],  // [nn × nh]
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= params.num_terminals) return;
    const uint np = params.np;
    const uint nh = params.nh;
    const uint nb = params.nb;
    const uint num_opp = params.num_opp;
    const uint S = params.stripes;
    const uint node = term_node_ids[tgid];
    const uint trav = params.traverser;

    threadgroup float bucket_reach_tg[MAX_OPP_BUCKETED * MAX_BUCKETS_GPU];
    threadgroup float fw_tg[MAX_BUCKETS_GPU * MAX_BUCKETS_GPU];
    threadgroup float ft_tg[MAX_BUCKETS_GPU * MAX_BUCKETS_GPU];
    threadgroup float fl_tg[MAX_BUCKETS_GPU * MAX_BUCKETS_GPU];
    threadgroup float fn_tg[MAX_BUCKETS_GPU * MAX_BUCKETS_GPU];
    threadgroup float partials[32];
    threadgroup float cfv_bucket_tg[MAX_BUCKETS_GPU];

    // ── Phase B: fraction tables → threadgroup (strided) ──
    const device float* tab = fractions + params.table_off;
    for (uint i = tid; i < nb * nb; i += tg_size) {
        fw_tg[i] = tab[i];
        ft_tg[i] = tab[nb * nb + i];
        fl_tg[i] = tab[2 * nb * nb + i];
        fn_tg[i] = tab[3 * nb * nb + i];
    }
    // ── Phase A: per-opponent reach reduce (CPU order: h ascending) ──
    const device ushort* map = bucket_map + params.map_off;
    if (tid < num_opp) {
        uint oi = tid;
        uint p = (oi < trav) ? oi : oi + 1;
        const device float* r_p = reach + (ulong(node) * np + p) * nh;
        for (uint b = 0; b < nb; b++) bucket_reach_tg[oi * nb + b] = 0.0f;
        for (uint h = 0; h < nh; h++) {
            float r = r_p[h];
            // Board-card filter (verbatim CPU: zero, then add — adds of
            // 0.0 preserved so the op sequence matches at singletons).
            if (r != 0.0f) {
                uint c1 = hand_cards[h * 2];
                uint c2 = hand_cards[h * 2 + 1];
                if (c1 == params.board0 || c2 == params.board0 ||
                    c1 == params.board1 || c2 == params.board1) {
                    r = 0.0f;
                }
            }
            ushort b = map[h];
            if (b == NO_BUCKET_GPU) continue;
            bucket_reach_tg[oi * nb + b] += r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Per-terminal scalars (identical in every lane) ──
    int contribs[8];
    for (uint p = 0; p < np; p++) contribs[p] = term_contribs[tgid * np + p];
    uint fold_mask = term_fold_masks[tgid];
    int c_t = contribs[trav];
    float eff_rake_rate = params.rake_rate;   // flop_seen always true here
    float eff_rake_cap = params.rake_cap;

    bool all_active_equal = true;
    {
        bool have_ref = false;
        int ref = 0;
        for (uint p = 0; p < np; p++) {
            if (fold_mask & (1u << p)) continue;
            if (!have_ref) { ref = contribs[p]; have_ref = true; }
            else if (contribs[p] != ref) { all_active_equal = false; break; }
        }
    }
    bool arm1 = all_active_equal && fold_mask == 0;

    bool opp_folded[MAX_OPP_BUCKETED];
    int opp_contrib[MAX_OPP_BUCKETED];
    for (uint oi = 0; oi < num_opp; oi++) {
        uint p = (oi < trav) ? oi : oi + 1;
        opp_folded[oi] = (fold_mask & (1u << p)) != 0;
        opp_contrib[oi] = contribs[p];
    }

    // ── Phase C: per-lane enumeration ──
    uint lanes_used = nb * S;
    float accum = 0.0f;
    bool lane_active = tid < lanes_used;
    uint bt = lane_active ? tid / S : 0;
    uint stripe = lane_active ? tid % S : 0;
    uint lo0 = stripe * nb / S;
    uint hi0 = (stripe + 1) * nb / S;

    if (lane_active && arm1) {
        float k = float(num_opp);
        float half_pot = float(params.starting_pot) / float(np) + float(c_t);
        int total_pot = params.starting_pot;
        for (uint p = 0; p < np; p++) total_pot += contribs[p];
        float rake = max(min(float(total_pot) * eff_rake_rate, eff_rake_cap), 0.0f);
        float rpus = half_pot > 0.0f ? rake / half_pot : 0.0f;

        int bo[MAX_OPP_BUCKETED];
        float s_stack[(MAX_OPP_BUCKETED + 1) * STATE_LEN];
        // depth-0 input state: point mass at "0 ties".
        for (int i = 0; i < STATE_LEN; i++) s_stack[i] = 0.0f;
        s_stack[1] = 1.0f;

        int d = 0;
        bo[0] = int(lo0) - 1;
        while (true) {
            bo[d]++;
            uint hi = (d == 0) ? hi0 : nb;
            if (bo[d] >= int(hi)) {
                if (d == 0) break;
                d--;
                continue;
            }
            uint b = uint(bo[d]);
            float r = bucket_reach_tg[uint(d) * nb + b];
            if (r == 0.0f) continue;
            float m = r;
            bool blocked = false;
            for (int j = 0; j < d; j++) {
                float f = fn_tg[uint(bo[j]) * nb + b];
                if (f == 0.0f) { blocked = true; break; }
                m *= f;
            }
            if (blocked) continue;
            uint idx = bt * nb + b;
            float fn_ = fn_tg[idx];
            if (fn_ == 0.0f) continue;
            float fw = fw_tg[idx];
            float ft = ft_tg[idx];
            float fl = fl_tg[idx];

            thread float* s_in = s_stack + d * STATE_LEN;
            thread float* s_out = s_stack + (d + 1) * STATE_LEN;
            for (int i = 0; i < STATE_LEN; i++) s_out[i] = 0.0f;
            if (s_in[0] != 0.0f) s_out[0] += s_in[0] * (m * fn_);
            for (int j = 0; j <= d; j++) {
                float s = s_in[1 + j];
                if (s == 0.0f) continue;
                if (fl != 0.0f) s_out[0] += s * (m * fl);
                if (ft != 0.0f) s_out[1 + j + 1] += s * (m * ft);
                if (fw != 0.0f) s_out[1 + j] += s * (m * fw);
            }
            if (uint(d + 1) == num_opp) {
                // Leaf. NOTE (declared, not discovered): arm1_leaf sums
                // the ≤(num_opp+2) state payoffs leaf-locally before the
                // single accum add; the CPU adds each term to the per-bt
                // chain directly. At singletons exactly one term is
                // nonzero → identical bits (the unstriped gate's case);
                // at general B this regrouping is inside the f64-ref
                // standard like every other reorder.
                accum += arm1_leaf(s_out, num_opp, k, rpus);
            } else {
                d++;
                bo[d] = -1;
            }
        }
        // CPU: cfv[bt] = half_pot * accum — same final multiply, per lane.
        accum = half_pot * accum;
    } else if (lane_active) {
        // Arm 2 (folds / unequal): level block + odometer with sc.
        bool traverser_folded = (fold_mask & (1u << trav)) != 0;
        LevelBlock lb = build_level_block(
            contribs, np, trav, num_opp, opp_folded, opp_contrib,
            params.starting_pot, eff_rake_rate, eff_rake_cap, traverser_folded);

        int bo[MAX_OPP_BUCKETED];
        float w_chain[MAX_OPP_BUCKETED + 1];
        float sc[MAX_OPP_BUCKETED][4];
        w_chain[0] = 1.0f;
        int d = 0;
        bo[0] = int(lo0) - 1;
        while (true) {
            bo[d]++;
            uint hi = (d == 0) ? hi0 : nb;
            if (bo[d] >= int(hi)) {
                if (d == 0) break;
                d--;
                continue;
            }
            uint b = uint(bo[d]);
            float r = bucket_reach_tg[uint(d) * nb + b];
            if (r == 0.0f) continue;
            float base = w_chain[d] * r;
            bool blocked = false;
            for (int j = 0; j < d; j++) {
                float f = fn_tg[uint(bo[j]) * nb + b];
                if (f == 0.0f) { blocked = true; break; }
                base *= f;
            }
            if (blocked) continue;
            uint idx = bt * nb + b;
            if (opp_folded[uint(d)]) {
                float n = fn_tg[idx];
                if (n == 0.0f) continue;
                sc[d][0] = 0.0f; sc[d][1] = 0.0f; sc[d][2] = 0.0f; sc[d][3] = n;
            } else {
                float fw = fw_tg[idx];
                float ft = ft_tg[idx];
                float fl = fl_tg[idx];
                if (fw == 0.0f && ft == 0.0f && fl == 0.0f) continue;
                sc[d][0] = fw; sc[d][1] = ft; sc[d][2] = fl; sc[d][3] = 0.0f;
            }
            if (uint(d + 1) == num_opp) {
                w_chain[d + 1] = base;
                accum += w_chain[d + 1] * net_expected(lb, sc, num_opp, opp_folded);
            } else {
                w_chain[d + 1] = base;
                d++;
                bo[d] = -1;
            }
        }
    }

    // ── Phase D: fixed-order stripe reduction ──
    partials[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane_active && stripe == 0) {
        float total = partials[tid];
        for (uint s = 1; s < S; s++) {
            total += partials[tid + s];   // fixed order: s1, s2, …
        }
        cfv_bucket_tg[bt] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase E: expansion (strided over h) ──
    device float* out = cfv + ulong(node) * nh;
    float nc = params.nc;
    for (uint h = tid; h < nh; h += tg_size) {
        ushort b = map[h];
        float v;
        if (b == NO_BUCKET_GPU) {
            v = 0.0f;
        } else if (nc > 0.0f) {
            v = cfv_bucket_tg[b] / nc;
        } else {
            v = cfv_bucket_tg[b];
        }
        out[h] = v;
    }
}
