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

// ── G6 lever 1: function-constant specialization (batched kernel).
// When the host sets these (uniform per-street B — the production
// case), the compiler constant-folds nb/np/num_opp/S: loop unrolling,
// stripe-logic elimination, better register allocation. Same
// operations in the same order — bit-exact with the generic pipeline
// by construction; the identity gates run THROUGH the specialized
// pipeline (uniform dims), N2's divergent dims exercise the generic
// fallback. ──
constant uint FC_NB        [[function_constant(0)]];
constant uint FC_NP        [[function_constant(1)]];
constant uint FC_NUM_OPP   [[function_constant(2)]];
constant uint FC_STRIPES   [[function_constant(3)]];
constant bool FC_UNIFORM = is_function_constant_defined(FC_NB);

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

// ════════════════════════════════════════════════════════════════════
// Batched variant (G4 step 3 — occupancy fix): one dispatch covers ALL
// (zone, outcome) terminal walks of one traverser pass. Same phases,
// same arithmetic as bucketed_terminal_collapsed; only the indexing
// changes: per-threadgroup job → walk descriptor → packed reach/cfv.
// The occupancy probe measured ~100-550 threadgroups/dispatch as the
// binding constraint (kernels overlap across queues but don't stack);
// this lifts a dispatch to ~1k-3.3k threadgroups.
// ════════════════════════════════════════════════════════════════════

struct BucketedWalkDesc {
    uint reach_off;    // float offset: packed per-terminal reach rows
    uint cfv_off;      // float offset: packed per-terminal cfv out
    uint contrib_off;  // int offset into concatenated zone contribs
    uint fold_off;     // uint offset into concatenated zone fold masks
    uint table_off;    // float offset of runout fraction block
    uint map_off;      // u16 offset of runout bucket map
    uint board0;
    uint board1;
    uint nb;
    uint stripes;
    uint _pad0;
    uint _pad1;
};

struct BucketedBatchedParams {
    uint num_jobs;
    uint np;
    uint nh;
    uint num_opp;
    uint traverser;
    float nc;
    int starting_pot;
    float rake_rate;
    float rake_cap;
    // 0 = packed mode (hybrid path: reach rows packed per terminal,
    // cfv packed per terminal); 1 = resident mode (G5 native: reach
    // read from the walk's resident buffer by NODE id via
    // node_ids_all[desc.node_off + lt], cfv written to the walk's
    // resident buffer at the node row).
    uint mode;
};

kernel void bucketed_terminal_collapsed_batched(
    constant BucketedBatchedParams& params   [[buffer(0)]],
    const device uint2*    jobs              [[buffer(1)]],  // (walk, local_term)
    const device BucketedWalkDesc* walks     [[buffer(2)]],
    const device int*      contribs_all      [[buffer(3)]],
    const device uint*     folds_all         [[buffer(4)]],
    const device float*    reach_packed      [[buffer(5)]],
    const device float*    fractions         [[buffer(6)]],
    const device ushort*   bucket_map        [[buffer(7)]],
    const device uchar*    hand_cards        [[buffer(8)]],
    device float*          cfv_packed        [[buffer(9)]],
    const device uint*     node_ids_all      [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= params.num_jobs) return;
    const uint2 job = jobs[tgid];
    const BucketedWalkDesc desc = walks[job.x];
    const uint lt = job.y;
    const uint node = node_ids_all[desc._pad0 + lt];  // _pad0 = zone node_off
    const uint np = FC_UNIFORM ? FC_NP : params.np;
    const uint nh = params.nh;
    const uint nb = FC_UNIFORM ? FC_NB : desc.nb;
    const uint num_opp = FC_UNIFORM ? FC_NUM_OPP : params.num_opp;
    const uint S = FC_UNIFORM ? FC_STRIPES : desc.stripes;
    const uint trav = params.traverser;

    threadgroup float bucket_reach_tg[MAX_OPP_BUCKETED * MAX_BUCKETS_GPU];
    threadgroup float fw_tg[MAX_BUCKETS_GPU * MAX_BUCKETS_GPU];
    threadgroup float ft_tg[MAX_BUCKETS_GPU * MAX_BUCKETS_GPU];
    threadgroup float fl_tg[MAX_BUCKETS_GPU * MAX_BUCKETS_GPU];
    threadgroup float fn_tg[MAX_BUCKETS_GPU * MAX_BUCKETS_GPU];
    threadgroup float partials[32];
    threadgroup float cfv_bucket_tg[MAX_BUCKETS_GPU];

    const device float* tab = fractions + desc.table_off;
    for (uint i = tid; i < nb * nb; i += tg_size) {
        fw_tg[i] = tab[i];
        ft_tg[i] = tab[nb * nb + i];
        fl_tg[i] = tab[2 * nb * nb + i];
        fn_tg[i] = tab[3 * nb * nb + i];
    }
    const device ushort* map = bucket_map + desc.map_off;
    const device float* reach_term = (params.mode == 1)
        ? reach_packed + desc.reach_off + ulong(node) * np * nh
        : reach_packed + desc.reach_off + ulong(lt) * np * nh;
    if (tid < num_opp) {
        uint oi = tid;
        uint p = (oi < trav) ? oi : oi + 1;
        const device float* r_p = reach_term + p * nh;
        for (uint b = 0; b < nb; b++) bucket_reach_tg[oi * nb + b] = 0.0f;
        for (uint h = 0; h < nh; h++) {
            float r = r_p[h];
            if (r != 0.0f) {
                uint c1 = hand_cards[h * 2];
                uint c2 = hand_cards[h * 2 + 1];
                if (c1 == desc.board0 || c2 == desc.board0 ||
                    c1 == desc.board1 || c2 == desc.board1) {
                    r = 0.0f;
                }
            }
            ushort b = map[h];
            if (b == NO_BUCKET_GPU) continue;
            bucket_reach_tg[oi * nb + b] += r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int contribs[8];
    for (uint p = 0; p < np; p++) contribs[p] = contribs_all[desc.contrib_off + lt * np + p];
    uint fold_mask = folds_all[desc.fold_off + lt];
    int c_t = contribs[trav];
    float eff_rake_rate = params.rake_rate;
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
                accum += arm1_leaf(s_out, num_opp, k, rpus);
            } else {
                d++;
                bo[d] = -1;
            }
        }
        accum = half_pot * accum;
    } else if (lane_active) {
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

    partials[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane_active && stripe == 0) {
        float total = partials[tid];
        for (uint s = 1; s < S; s++) {
            total += partials[tid + s];
        }
        cfv_bucket_tg[bt] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out = (params.mode == 1)
        ? cfv_packed + desc.cfv_off + ulong(node) * nh
        : cfv_packed + desc.cfv_off + ulong(lt) * nh;
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

// ════════════════════════════════════════════════════════════════════
// G5: fully GPU-native bucketed iteration — walk kernels.
// Bit-exactness design (the same argument as every prior port): each
// output element is computed by exactly one thread, with inner loops
// in the CPU's iteration order (actions ascending, hands ascending,
// children ascending, outcomes ascending), so at singleton buckets the
// computation graphs coincide with the CPU walk element-for-element.
// ════════════════════════════════════════════════════════════════════

struct NativeInfosetDesc {  // mirrors Rust InfosetDesc
    uint base;
    uint na;
    uint owner;
    uint node;
};

struct NativeBuDesc {  // mirrors Rust BuDesc
    uint node;
    uint kind;  // 0 player, 1 chance
    uint na;
    uint owner;
    uint base;
    uint child_off;
};

struct NativeReachEdge {  // mirrors Rust ReachEdge
    uint parent;
    uint child;
    uint action;  // MAX = chance pass-through
    uint owner;
};

struct NativeWalkParams {
    uint count;      // items in this dispatch (descs / edges / rows)
    uint np;
    uint nh;
    uint nb;         // zone bucket count
    uint map_off;    // runout map offset for this walk's zone outcome
    uint omega_off;  // runout omega offset
    uint traverser;
    float alpha_t;
    float beta_t;
    float gamma_t;
    float regret_floor;
    uint _pad;
};

// regret matching, one thread per (infoset desc, bucket column).
kernel void bucketed_native_strategies(
    const device float* regrets              [[buffer(0)]],
    device float* strategy                   [[buffer(1)]],
    const device NativeInfosetDesc* descs    [[buffer(2)]],
    constant NativeWalkParams& p             [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint di = gid / p.nb;
    uint col = gid % p.nb;
    if (di >= p.count) return;
    NativeInfosetDesc d = descs[di];
    // REGRET_MATCH_EPS mirror — MUST match the CPU's 1e-5
    // (flop_start_vector_cfr::regret_matching_into) or the first
    // regret-matched iteration diverges.
    const float EPS = 1e-5f;
    float pos_sum = 0.0f;
    for (uint a = 0; a < d.na; a++) {
        float r = regrets[d.base + a * p.nb + col];
        if (r > EPS) pos_sum += r;
    }
    if (pos_sum > 0.0f) {
        for (uint a = 0; a < d.na; a++) {
            float r = regrets[d.base + a * p.nb + col];
            strategy[d.base + a * p.nb + col] = (r > EPS) ? r / pos_sum : 0.0f;
        }
    } else {
        float uniform = 1.0f / float(d.na);
        for (uint a = 0; a < d.na; a++) {
            strategy[d.base + a * p.nb + col] = uniform;
        }
    }
}

kernel void bucketed_native_zero(
    device float* buf       [[buffer(0)]],
    constant uint& n        [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) buf[gid] = 0.0f;
}

// reach[p*nh + h] = initial_weights[p*nh + h] at the root (flop walk).
kernel void bucketed_native_root_init(
    device float* reach              [[buffer(0)]],
    const device float* weights     [[buffer(1)]],
    constant NativeWalkParams& p     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.np * p.nh) return;
    reach[gid] = weights[gid];
}

// Copy seed rows (all players) from the parent walk's reach buffer.
kernel void bucketed_native_seed(
    device float* dst                [[buffer(0)]],
    const device float* src         [[buffer(1)]],
    const device uint* rows          [[buffer(2)]],  // node ids
    constant NativeWalkParams& p     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint per_row = p.np * p.nh;
    uint ri = gid / per_row;
    uint k = gid % per_row;
    if (ri >= p.count) return;
    ulong base = ulong(rows[ri]) * per_row;
    dst[base + k] = src[base + k];
}

// One reach level: thread per (edge, h); loops players in order.
// child[p] = parent[p]; child[owner] = parent[owner] * sigma — the
// CPU's copy-then-multiply collapses to one expression with the same
// float value (x stored then x*sigma  ≡  x*sigma).
kernel void bucketed_native_reach_level(
    device float* reach                      [[buffer(0)]],
    const device float* strategy             [[buffer(1)]],
    const device uint* node_local_base       [[buffer(2)]],
    const device ushort* bucket_map          [[buffer(3)]],
    const device NativeReachEdge* edges      [[buffer(4)]],
    constant NativeWalkParams& p             [[buffer(5)]],
    constant uint& zone_outcome_base         [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint ei = gid / p.nh;
    uint h = gid % p.nh;
    if (ei >= p.count) return;
    NativeReachEdge e = edges[ei];
    ulong src = ulong(e.parent) * p.np * p.nh;
    ulong dst = ulong(e.child) * p.np * p.nh;
    if (e.action == 0xFFFFFFFFu) {
        for (uint pl = 0; pl < p.np; pl++) {
            reach[dst + pl * p.nh + h] = reach[src + pl * p.nh + h];
        }
        return;
    }
    const device ushort* map = bucket_map + p.map_off;
    for (uint pl = 0; pl < p.np; pl++) {
        float v = reach[src + pl * p.nh + h];
        if (pl == e.owner) {
            ushort b = map[h];
            if (b == NO_BUCKET_GPU) {
                v = 0.0f;  // CPU: NO_BUCKET hands zeroed at strategy step
            } else {
                v = v * strategy[zone_outcome_base + node_local_base[e.parent]
                                 + e.action * p.nb + b];
            }
        }
        reach[dst + pl * p.nh + h] = v;
    }
}

// One bottom-up cfv level: thread per (bu desc, h). Player traverser:
// sigma-weighted child sum; player other: plain child sum; chance:
// child sum. Children ascending — the CPU's order.
kernel void bucketed_native_cfv_level(
    device float* cfv                        [[buffer(0)]],
    const device float* strategy             [[buffer(1)]],
    const device ushort* bucket_map          [[buffer(2)]],
    const device NativeBuDesc* descs         [[buffer(3)]],
    const device uint* children              [[buffer(4)]],
    constant NativeWalkParams& p             [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint di = gid / p.nh;
    uint h = gid % p.nh;
    if (di >= p.count) return;
    NativeBuDesc d = descs[di];
    const device ushort* map = bucket_map + p.map_off;
    float acc = 0.0f;
    if (d.kind == 0 && d.owner == p.traverser) {
        ushort b = map[h];
        if (b != NO_BUCKET_GPU) {
            for (uint a = 0; a < d.na; a++) {
                uint child = children[d.child_off + a];
                acc += strategy[d.base + a * p.nb + b] * cfv[ulong(child) * p.nh + h];
            }
        }
    } else {
        for (uint a = 0; a < d.na; a++) {
            uint child = children[d.child_off + a];
            acc += cfv[ulong(child) * p.nh + h];
        }
    }
    cfv[ulong(d.node) * p.nh + h] = acc;
}

// Regret + cum update for traverser-owned infosets of one zone walk,
// after all cfv levels: thread per (desc, action, bucket); member
// hands summed h-ascending (the CPU's aggregation order).
kernel void bucketed_native_regret_update(
    const device float* cfv                  [[buffer(0)]],
    device float* regrets                    [[buffer(1)]],
    device float* cum                        [[buffer(2)]],
    const device float* strategy             [[buffer(3)]],
    const device ushort* bucket_map          [[buffer(4)]],
    const device float* omegas               [[buffer(5)]],
    const device NativeInfosetDesc* descs    [[buffer(6)]],
    const device uint* children              [[buffer(7)]],
    const device uint* node_child_off        [[buffer(8)]],
    constant NativeWalkParams& p             [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint per_desc = MAX_NA_POSTFLOP * p.nb;   // a-major slot grid
    uint di = gid / per_desc;
    uint rem = gid % per_desc;
    uint a = rem / p.nb;
    uint b = rem % p.nb;
    if (di >= p.count) return;
    NativeInfosetDesc d = descs[di];
    if (d.owner != p.traverser) return;
    if (a >= d.na) return;
    const device ushort* map = bucket_map + p.map_off;
    const device float* omega = omegas + p.omega_off;
    uint child = children[node_child_off[d.node] + a];
    const device float* cfv_child = cfv + ulong(child) * p.nh;
    const device float* cfv_avg = cfv + ulong(d.node) * p.nh;
    float inst = 0.0f;
    for (uint h = 0; h < p.nh; h++) {
        if (map[h] != b) continue;
        inst += omega[h] * (cfv_child[h] - cfv_avg[h]);
    }
    uint ridx = d.base + a * p.nb + b;
    float old_r = regrets[ridx];
    float coef = old_r >= 0.0f ? p.alpha_t : p.beta_t;
    float nr = coef * old_r + inst;
    if (nr < p.regret_floor) nr = p.regret_floor;
    regrets[ridx] = nr;
    cum[ridx] = p.gamma_t * cum[ridx] + strategy[ridx];
}

// Cross-walk chance accumulation: dst[row, h] = sum over outcomes
// (ascending — the CPU's loop order) of cp[o, h] * src_o[row, h].
// src walk cfv buffers live in one mega buffer at walk_offs[o].
kernel void bucketed_native_chance_accum(
    device float* dst_cfv                    [[buffer(0)]],
    const device float* cfv_mega             [[buffer(1)]],
    const device uint* walk_offs             [[buffer(2)]],  // float offsets
    const device float* cp                   [[buffer(3)]],  // [o*nh + h]
    const device uint* rows                  [[buffer(4)]],  // child node ids
    constant NativeWalkParams& p             [[buffer(5)]],
    constant uint& n_outcomes                [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint ri = gid / p.nh;
    uint h = gid % p.nh;
    if (ri >= p.count) return;
    ulong row = ulong(rows[ri]) * p.nh + h;
    float acc = 0.0f;
    for (uint o = 0; o < n_outcomes; o++) {
        acc += cp[o * p.nh + h] * cfv_mega[walk_offs[o] + row];
    }
    dst_cfv[row] = acc;
}

// SHARED-BUFFER SEQUENTIAL MODE (2026-06-12, big-tree unlock): one
// outcome's contribution to the chance sum, accumulated into a small
// per-boundary-row buffer while the shared cfv buffer still holds that
// outcome's values. acc starts zeroed; outcome-ascending dispatch order
// reproduces the mega-mode kernel's in-register loop bit-exactly
// (acc = ((cp0·v0) + cp1·v1) + ... — same f32 op sequence).
kernel void bucketed_native_chance_accum_step(
    device float* acc                        [[buffer(0)]],  // [r*nh + h]
    const device float* cfv                  [[buffer(1)]],  // shared, offset 0
    const device float* cp                   [[buffer(2)]],  // pre-offset: [h]
    const device uint* rows                  [[buffer(3)]],  // child node ids
    constant NativeWalkParams& p             [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint r = gid / p.nh;
    uint h = gid % p.nh;
    if (r >= p.count) return;
    acc[gid] += cp[h] * cfv[ulong(rows[r]) * p.nh + h];
}

// Companion: publish the accumulated sums into the shared cfv buffer
// at the boundary rows (so the parent zone's bottom-up reads them as
// child values), and re-zero acc for the next outcome group.
kernel void bucketed_native_rows_publish(
    device float* cfv                        [[buffer(0)]],
    device float* acc                        [[buffer(1)]],
    const device uint* rows                  [[buffer(2)]],
    constant NativeWalkParams& p             [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint r = gid / p.nh;
    uint h = gid % p.nh;
    if (r >= p.count) return;
    cfv[ulong(rows[r]) * p.nh + h] = acc[gid];
    acc[gid] = 0.0f;
}

// Root cfv accumulation across iterations (traverser 0): one thread
// per h, dst += flop_cfv[root, h].
kernel void bucketed_native_root_accum(
    device float* dst                        [[buffer(0)]],
    const device float* flop_cfv             [[buffer(1)]],
    constant uint& nh                        [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= nh) return;
    dst[gid] += flop_cfv[gid];
}
