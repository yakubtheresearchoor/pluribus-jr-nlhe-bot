// MCCFR ACCESS-PATTERN throughput benchmark (bandwidth-vs-divergence probe).
// Each GPU thread runs one synthetic external-sampling trajectory: a divergent
// walk through DEPTH betting levels, doing the work that dominates real MCCFR —
// scattered regret reads + regret-matching + atomic regret read-modify-write —
// over a regret buffer sized to exceed cache (so memory bandwidth is the
// bottleneck, as measured on CPU). Identical logic runs on the CPU side in Rust
// for a fair traj/s comparison. NOT a correct solver — a faithful probe of the
// access pattern that decides whether the GPU's bandwidth beats its divergence.
#include <metal_stdlib>
using namespace metal;

constant uint MCB_DEPTH = 8;  // betting levels per trajectory
constant uint MCB_NA = 4;     // actions per decision

inline uint mcb_xs(thread ulong &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (uint)s;
}

kernel void mccfr_bench(
    device atomic_float *regret  [[buffer(0)]],
    constant uint       &ninfo   [[buffer(1)]],
    device const ulong  *seeds   [[buffer(2)]],
    device float        *out     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    ulong s = seeds[tid];
    uint path = tid;
    float v = 0.0f;
    for (uint d = 0; d < MCB_DEPTH; d++) {
        // scattered infoset index (data-dependent ⇒ divergence + random access)
        uint info = (path * 2654435761u + d * 40503u) % ninfo;
        uint base = info * MCB_NA;
        // read regrets + regret-match
        float r[MCB_NA]; float sum = 0.0f;
        for (uint a = 0; a < MCB_NA; a++) {
            float ra = max(atomic_load_explicit(&regret[base + a], memory_order_relaxed), 0.0f);
            r[a] = ra; sum += ra;
        }
        // sample an action (external sampling)
        uint act = mcb_xs(s) % MCB_NA;
        path = path * MCB_NA + act;
        v += (sum > 0.0f) ? r[act] / sum : 0.0f;
        // atomic regret update — the scattered read-modify-write (bandwidth-bound)
        for (uint a = 0; a < MCB_NA; a++) {
            float delta = (a == act) ? 1.0f : -0.3333f;
            atomic_fetch_add_explicit(&regret[base + a], delta, memory_order_relaxed);
        }
    }
    out[tid] = v;
}

// PHASE 1, step 1: the bucketed-showdown DP ported to Metal (the hardest piece).
// Per thread = one terminal scenario: state-carrying (beaten, tie-count) DP over
// the sampled active-opponent river buckets, exactly mirroring the CPU
// `Mccfr::terminal` showdown. Validated bit-against the CPU on random scenarios.
constant uint SD_MAXNP = 6;

kernel void showdown_dp(
    device const float *f_w        [[buffer(0)]],  // nb*nb win fraction
    device const float *f_t        [[buffer(1)]],  // tie
    device const float *f_l        [[buffer(2)]],  // lose
    device const float *f_n        [[buffer(3)]],  // norm (compatible mass)
    constant uint       &nb        [[buffer(4)]],
    constant uint       &np        [[buffer(5)]],
    device const uint  *bt_arr     [[buffer(6)]],  // per-scenario traverser bucket
    device const uint  *na_arr     [[buffer(7)]],  // per-scenario #active opponents
    device const uint  *opp_arr    [[buffer(8)]],  // per-scenario SD_MAXNP opp buckets
    device const float *half_arr   [[buffer(9)]],
    device const float *net_arr    [[buffer(10)]],
    device float        *out       [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    uint bt = bt_arr[tid];
    uint na = na_arr[tid];
    float half_pot = half_arr[tid];
    float net_pot = net_arr[tid];
    float state[SD_MAXNP + 2];
    for (uint i = 0; i < np + 2; i++) state[i] = 0.0f;
    state[1] = 1.0f;
    for (uint k = 0; k < na; k++) {
        uint bo = opp_arr[tid * SD_MAXNP + k];
        uint idx = bt * nb + bo;
        float fn_ = f_n[idx];
        float norm = fn_ > 0.0f ? fn_ : 1.0f;
        float pw = f_w[idx] / norm, pt = f_t[idx] / norm, pl = f_l[idx] / norm;
        float ns[SD_MAXNP + 2];
        for (uint i = 0; i < np + 2; i++) ns[i] = 0.0f;
        if (state[0] != 0.0f) ns[0] += state[0];
        for (uint j = 0; j < np; j++) {
            float s = state[1 + j];
            if (s == 0.0f) continue;
            ns[0] += s * pl;
            ns[1 + j + 1] += s * pt;
            ns[1 + j] += s * pw;
        }
        for (uint i = 0; i < np + 2; i++) state[i] = ns[i];
    }
    float value = state[0] * (-half_pot);
    for (uint j = 0; j < np; j++) {
        float s = state[1 + j];
        if (s == 0.0f) continue;
        float t = (float)(j + 1);
        value += s * (net_pot / t - half_pot);
    }
    out[tid] = value;
}

// PHASE 1, steps 2-3: SINGLE-CELL GPU MCCFR (one fixed runout). Each thread runs
// one external-sampling trajectory over the flattened betting tree: iterative
// (explicit-stack, branch only at traverser nodes) traversal + the validated
// showdown DP + atomic regret/cum. Converges to the same equilibrium as the CPU
// `Mccfr` (the Phase-1 gate). All structural fields pre-marshaled per node.
struct CellParams {
    uint np; uint nb; uint maxna; int starting_pot;
    float rake_rate; float rake_cap;
    uint nt; uint nr; uint nh;   // runout grid: nt turns × nr rivers, nh = hand universe
    float prune_c; uint prune_active;  // negative-regret pruning (skip regret≤prune_c subtrees)
};
constant uint CL_MAXNA = 8;
constant uint CL_STACK = 32;

// Linear/Discounted-CFR: periodically scale accumulated regret + average strategy by
// d=k/(k+1) (Pluribus discounts both every ~10 min for the first 400 min) so early,
// poorly-trained iterations decay and later ones dominate → ~3× faster convergence.
// Regret stays ≥0 (CFR+) since d>0. Run as a separate pass (no atomic contention).
kernel void discount_buf(
    device float *reg [[buffer(0)]],
    device float *cum [[buffer(1)]],
    constant float &d  [[buffer(2)]],
    constant uint  &n  [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    reg[i] *= d;
    cum[i] *= d;
}

kernel void mccfr_cell(
    device const uchar  *n_type    [[buffer(0)]],
    device const uchar  *n_player  [[buffer(1)]],
    device const uchar  *n_bs      [[buffer(2)]],
    device const ushort *n_nch     [[buffer(3)]],
    device const uint   *n_chstart [[buffer(4)]],
    device const uint   *children  [[buffer(5)]],
    device const int    *n_local   [[buffer(6)]],
    device const ushort *n_fold    [[buffer(7)]],
    device const int    *n_contrib [[buffer(8)]],   // node*np + p
    constant CellParams &P         [[buffer(9)]],
    device const ushort *flop_b    [[buffer(10)]],
    device const ushort *turn_b    [[buffer(11)]],
    device const ushort *river_b   [[buffer(12)]],
    device const uchar  *t_alive   [[buffer(13)]],
    device const uchar  *r_alive   [[buffer(14)]],
    device const int    *strength  [[buffer(15)]],  // [run*nh + hand] exact 7-card rank (EXACT showdown)
    device const float  *f_t       [[buffer(16)]],  // (unused — exact showdown)
    device const float  *f_l       [[buffer(17)]],  // (unused)
    device const float  *f_n       [[buffer(18)]],  // (unused)
    device atomic_float *regret    [[buffer(19)]],
    device atomic_float *cum       [[buffer(20)]],
    device const uint   *hands_in  [[buffer(21)]],  // tid*np + p
    device const ulong  *seeds     [[buffer(22)]],
    device float        *root_out  [[buffer(23)]],  // traverser's value at the cell root (for preflop)
    uint tid [[thread_position_in_grid]])
{
    uint np = P.np, nb = P.nb, maxna = P.maxna, nh = P.nh;
    uint h[6];
    for (uint p = 0; p < np; p++) h[p] = hands_in[tid * np + p];
    uint traverser = tid % np;
    ulong s = seeds[tid];

    // External-sampling over CHANCE: sample THIS trajectory's runout (turn,river)
    // from the nt×nr grid (Pluribus samples the board per iteration — a FIXED 1×1
    // runout is a clairvoyant game, the bug this replaces). All board-dependent
    // lookups (turn/river bucket maps, per-street alive, showdown table) index by
    // the sampled runout.
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; uint ti = (uint)(s % (ulong)P.nt);
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; uint ri = (uint)(s % (ulong)P.nr);
    uint run  = ti * P.nr + ri;   // flat runout index
    uint roff = run * nh;         // per-(ti,ri) hand-map / river-alive offset
    uint taoff = ti * nh;         // per-ti turn-map / turn-alive offset
    uint toff = run * nb * nb;    // per-(ti,ri) showdown-table offset
    // PRUNING: this trajectory prunes iff active (past warmup) AND a 95% coin lands.
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    bool prune_this = (P.prune_active != 0) && ((float)((uint)(s >> 32)) * (1.0f / 4294967296.0f) < 0.95f);

    // ---- showdown at a terminal node, returns traverser value ----
    // (inlined; validated separately as showdown_dp)
    // strategy via regret-matching for (local,bucket): handled inline below.

    // explicit-stack iterative external sampling.
    struct Frame { uint node; uint ai; uint na; uint base; uint pruned; float cv[CL_MAXNA]; float st[CL_MAXNA]; };
    Frame stack[CL_STACK];
    int sp = -1;
    float ret = 0.0f;
    uint cur = 0;
    bool returning = false;

    for (uint guard = 0; guard < 100000u; guard++) {
        if (!returning) {
            // descend: follow linear (opp/chance sampled) until a traverser node or terminal
            bool stop = false;
            while (!stop) {
                uchar nt = n_type[cur];
                if (nt == 0) { // terminal → showdown
                    ushort fm = n_fold[cur];
                    uint base_c = cur * np;
                    int c_t = n_contrib[base_c + traverser];
                    int total = P.starting_pot;
                    for (uint p = 0; p < np; p++) total += n_contrib[base_c + p];
                    float half_pot = (float)P.starting_pot / (float)np + (float)c_t;
                    uchar bs = n_bs[cur];
                    // per-street death of any active hand
                    bool dead = false;
                    for (uint p = 0; p < np; p++) {
                        if (((fm >> p) & 1) == 0) {
                            uint hp = h[p];
                            if (bs >= 1 && t_alive[taoff + hp] == 0) dead = true;  // Turn=1
                            if (bs >= 2 && r_alive[roff + hp] == 0) dead = true;   // River=2
                        }
                    }
                    if (dead) { ret = 0.0f; stop = true; returning = true; break; }
                    if (((fm >> traverser) & 1) == 1) { ret = -half_pot; stop = true; returning = true; break; }
                    // main-pot rake
                    int minlev = 2147483647;
                    for (uint p = 0; p < np; p++) { int c = n_contrib[base_c + p]; if (c < minlev) minlev = c; }
                    int cnt = 0; for (uint p = 0; p < np; p++) if (n_contrib[base_c + p] >= minlev) cnt++;
                    int main_pot = minlev * cnt + P.starting_pot;
                    float rake = min(max((float)main_pot * P.rake_rate, 0.0f), P.rake_cap);
                    float net_pot = (float)total - rake;
                    // active opponents
                    // EXACT showdown (Pluribus-faithful): score the actual sampled hands
                    // on the sampled board. strength[roff + hand] = 7-card rank.
                    int s_t = strength[roff + h[traverser]];
                    uint better = 0, equal = 0; bool anyopp = false;
                    for (uint p = 0; p < np; p++) if (p != traverser && ((fm >> p) & 1) == 0) {
                        anyopp = true; int s_o = strength[roff + h[p]];
                        if (s_o > s_t) better++; else if (s_o == s_t) equal++;
                    }
                    if (!anyopp) ret = net_pot - half_pot;
                    else if (better > 0) ret = -half_pot;
                    else ret = net_pot / (float)(equal + 1) - half_pot;
                    stop = true; returning = true; break;
                }
                if (nt == 1) { cur = children[n_chstart[cur]]; continue; } // chance: single child
                // player node
                uint player = n_player[cur];
                uint na = n_nch[cur];
                uchar bs = n_bs[cur];
                uint bk = (bs == 0) ? flop_b[h[player]] : ((bs == 1) ? turn_b[taoff + h[player]] : river_b[roff + h[player]]);  // Flop=0 Turn=1 River=2
                // dead actor (hand conflicts the sampled runout) → NO_BUCKET → impossible
                // world, value 0 (matches CPU `if !alive { return 0 }`); without this the
                // index (local*nb + 65535)*maxna is far OOB — input/layout-sensitive blowup.
                if (bk >= nb) { ret = 0.0f; stop = true; returning = true; break; }
                uint base = ((uint)n_local[cur] * nb + bk) * maxna;
                // regret-match strategy
                float st[CL_MAXNA]; float sum = 0.0f;
                for (uint a = 0; a < na; a++) { float r = max(atomic_load_explicit(&regret[base+a], memory_order_relaxed), 0.0f); st[a] = r; sum += r; }
                if (sum > 0.0f) { for (uint a = 0; a < na; a++) st[a] /= sum; } else { for (uint a = 0; a < na; a++) st[a] = 1.0f/(float)na; }
                if (player == traverser) {
                    // PRUNING: mark actions with regret ≤ prune_c (skip their subtrees this
                    // iter); never prune the last betting round (river) or ALL actions.
                    uint pmask = 0;
                    if (prune_this && bs != 2) {
                        uint cnt = 0;
                        for (uint a = 0; a < na; a++) {
                            if (atomic_load_explicit(&regret[base + a], memory_order_relaxed) <= P.prune_c) { pmask |= (1u << a); cnt++; }
                        }
                        if (cnt >= na) pmask = 0;
                    }
                    sp++;
                    stack[sp].node = cur; stack[sp].na = na; stack[sp].base = base; stack[sp].pruned = pmask;
                    for (uint a = 0; a < na; a++) { stack[sp].st[a] = st[a]; stack[sp].cv[a] = 0.0f; }
                    uint a0 = 0; while (a0 < na && ((pmask >> a0) & 1)) a0++;
                    stack[sp].ai = a0;
                    cur = children[n_chstart[cur] + a0];
                    continue;
                } else {
                    // sample one action
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                    float r = (float)((uint)(s >> 32)) * (1.0f / 4294967296.0f);  // high 32 bits (xorshift64 low bits are low-quality)
                    float acc = 0.0f; uint a = na - 1;
                    for (uint i = 0; i < na; i++) { acc += st[i]; if (r <= acc) { a = i; break; } }
                    cur = children[n_chstart[cur] + a];
                    continue;
                }
            }
        } else {
            // returning: ret holds the finished subtree's value
            if (sp < 0) { root_out[tid] = ret; break; } // root done — emit the value
            stack[sp].cv[stack[sp].ai] = ret;
            stack[sp].ai++;
            while (stack[sp].ai < stack[sp].na && ((stack[sp].pruned >> stack[sp].ai) & 1)) stack[sp].ai++;
            if (stack[sp].ai < stack[sp].na) {
                cur = children[n_chstart[stack[sp].node] + stack[sp].ai];
                returning = false;
                continue;
            }
            // all (unpruned) children done → v + regret update, pop
            float v = 0.0f;
            for (uint a = 0; a < stack[sp].na; a++) v += stack[sp].st[a] * stack[sp].cv[a];
            for (uint a = 0; a < stack[sp].na; a++) {
                if ((stack[sp].pruned >> a) & 1) continue; // pruned: regret/cum unchanged this traj
                // CFR+: regret = max(regret + (cv-v), 0), applied atomically via CAS
                // (matches the CPU `.max(0)`; vanilla atomic-add converges to a
                // DIFFERENT equilibrium — the bug the step-4 gate caught).
                float delta = stack[sp].cv[a] - v;
                float old = atomic_load_explicit(&regret[stack[sp].base + a], memory_order_relaxed);
                float newv;
                do { newv = max(old + delta, 0.0f); }
                while (!atomic_compare_exchange_weak_explicit(&regret[stack[sp].base + a], &old, newv, memory_order_relaxed, memory_order_relaxed));
                atomic_fetch_add_explicit(&cum[stack[sp].base + a], stack[sp].st[a], memory_order_relaxed);
            }
            ret = v; sp--;
            // stay in returning mode
        }
    }
}

// PHASE 2: CONNECTED GPU MCCFR — one thread = one whole preflop→cell→showdown
// trajectory, fully on-device. Combined node array (preflop nodes + all cells'
// nodes); combined regret/cum (preflop region + per-cell regions). At the seam
// (c_seam=1 chance node) the live preflop seats are remapped to postflop slots.
// Preflop nodes (c_pre=1) bucket by preflop CLASS (169); postflop by card bucket.
struct ConnParams { uint np; uint maxna; float rake_rate; float rake_cap; uint nt; uint nr; uint nh; uint nf; uint post_stride; float prune_c; uint prune_active; uint freeze_pre; uint n_prefix; float target_q; uint prefix_stride; float target_eps; };

kernel void mccfr_conn(
    device const uchar  *c_type    [[buffer(0)]],
    device const uchar  *c_pre     [[buffer(1)]],   // 1 = preflop node
    device const uchar  *c_seam    [[buffer(2)]],   // 1 = preflop→postflop seam chance
    device const uchar  *c_player  [[buffer(3)]],
    device const uchar  *c_bs      [[buffer(4)]],
    device const ushort *c_nch     [[buffer(5)]],
    device const uint   *c_chstart [[buffer(6)]],
    device const uint   *children  [[buffer(7)]],
    device const int    *c_local   [[buffer(8)]],
    device const ushort *c_fold    [[buffer(9)]],
    device const int    *c_contrib [[buffer(10)]],  // node*np + p
    device const int    *c_spot    [[buffer(11)]],  // node starting pot
    device const uint   *c_regbase [[buffer(12)]],  // regret region offset
    device const uint   *c_nb      [[buffer(13)]],  // 169 (pre) or nb (post)
    constant ConnParams &P         [[buffer(14)]],
    device const ushort *flop_b    [[buffer(15)]],
    device const ushort *turn_b    [[buffer(16)]],
    device const ushort *river_b   [[buffer(17)]],
    device const uchar  *t_alive   [[buffer(18)]],
    device const uchar  *r_alive   [[buffer(19)]],
    device const int    *strength  [[buffer(20)]],  // [run*nh + hand] exact 7-card rank (EXACT showdown)
    // TARGETED EXPLORATION (2026-07-03, repurposed unused slots 21/22):
    // prefix_act[pi*stride + hop] = forced action INDEX at the hop'th player
    // node; prefix_meta[pi*2] = len, [pi*2+1] = target player (traverser).
    device const uchar  *prefix_act [[buffer(21)]],
    device const uint   *prefix_meta[[buffer(22)]],
    device const float  *f_n       [[buffer(23)]],  // (unused)
    constant uint       &nb_riv    [[buffer(24)]],  // (unused — exact showdown)
    device atomic_float *regret    [[buffer(25)]],
    device atomic_float *cum       [[buffer(26)]],
    device const uint   *hands_in  [[buffer(27)]],  // tid*np + p
    device const uint   *hand_cls  [[buffer(28)]],  // per hand → preflop class
    device const ulong  *seeds     [[buffer(29)]],
    device const uint   *fi_in     [[buffer(30)]],  // per-trajectory dealt flop (nf=1 ⇒ 0)
    uint tid [[thread_position_in_grid]])
{
    uint np = P.np, maxna = P.maxna, nh = P.nh;
    uint h[6];
    for (uint p = 0; p < np; p++) h[p] = hands_in[tid * np + p];
    uint trav = tid % np;
    ulong s = seeds[tid];
    // TARGETED EXPLORATION: with prob target_q, force this trajectory down a
    // rare defense line (folds→open→3-bet, defender to act). The defender
    // becomes the traverser; every FORCED sampled action multiplies the
    // importance weight w by the actor's current σ(a) — so the defender trains
    // against the TRUE weighted 3-bet range, not "any two cards". Unbiased:
    // targeting reallocates variance, not the fixed point.
    uint t_hop = 0u; uint t_len = 0u; uint t_pi = 0u; float w = 1.0f;
    if (P.n_prefix > 0u && P.target_q > 0.0f) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        if ((float)((uint)(s >> 32)) * (1.0f / 4294967296.0f) < P.target_q) {
            // Simple uniform pick + ε-FLOORED weights at the hops (a rejection
            // pre-walk was tried and REVERTED: it filters dead-by-σ prefixes
            // before the floor can rescue them — the two mechanisms cancel).
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            t_pi = (uint)(s % (ulong)P.n_prefix);
            t_len = prefix_meta[t_pi * 2u];
            trav = prefix_meta[t_pi * 2u + 1u];
        }
    }
    // ALL-FLOPS: this trajectory's flop (host-sampled + host-dealt flop-relative hands).
    // postflop card-map/strength/regret lookups offset by fi. nf=1 ⇒ fi=0 ⇒ all 0.
    uint fi = fi_in[tid];
    uint foff_f = fi * nh;               // flop_b offset
    uint foff_t = fi * P.nt * nh;        // turn_b / t_alive offset
    uint foff_r = fi * P.nt * P.nr * nh; // river_b / strength / r_alive offset
    uint freg = fi * P.post_stride;      // per-flop postflop regret-region offset
    // PRUNING: 95%-coin per trajectory when active.
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    bool prune_this = (P.prune_active != 0) && ((float)((uint)(s >> 32)) * (1.0f / 4294967296.0f) < 0.95f);
    // External-sampling over CHANCE: sample this trajectory's runout (turn,river)
    // from the full nt×nr grid — Pluribus samples the board per iteration (a fixed
    // 1×1 runout is a clairvoyant game and produces literally-wrong strategies).
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; uint ti = (uint)(s % (ulong)P.nt);
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; uint ri = (uint)(s % (ulong)P.nr);
    uint run = ti * P.nr + ri;    // flat runout index
    uint roff = run * nh;         // per-(ti,ri) hand-map / river-alive offset
    uint taoff = ti * nh;         // per-ti turn-map / turn-alive offset
    // postflop context (set at the seam): live-slot → hand, traverser slot, #live.
    uint post_hand[6]; uint post_trav = 0; uint nlive = 0;
    // SEAM MONEY (2026-07-03 pathology fix): the cell trees are SHARED across
    // preflop lines, so postflop terminals read the CELL's contributions/pot —
    // the actual line's preflop investment was never charged past the seam
    // (raising was free ⇒ systematic looseness; HU showed pair inversion).
    // Carry the line-vs-cell deltas across the seam: set at each crossing,
    // valid for the whole cell subtree under DFS (same pattern as post_hand).
    float seam_extra[6]; for (uint p = 0; p < 6; p++) seam_extra[p] = 0.0f;
    float seam_pot_extra = 0.0f;

    struct Frame { uint node; uint ai; uint na; uint base; uint pruned; float cv[CL_MAXNA]; float st[CL_MAXNA]; };
    Frame stack[CL_STACK];
    int sp = -1; float ret = 0.0f; uint cur = 0; bool returning = false;

    for (uint guard = 0; guard < 200000u; guard++) {
        if (!returning) {
            bool stop = false;
            while (!stop) {
                uchar ty = c_type[cur];
                if (ty == 1) { // chance
                    if (c_seam[cur] == 1) {
                        // SEAM: live preflop seats → postflop slots
                        ushort fm = c_fold[cur];
                        uint cell_root = children[c_chstart[cur]];
                        uint bc_pre = cur * np;
                        uint bc_cell = cell_root * np;
                        // Real line money: blinds (pre spot) + every seat's
                        // contribution (folders' dead money included).
                        float real_pot = (float)c_spot[cur];
                        for (uint p = 0; p < np; p++) real_pot += (float)c_contrib[bc_pre + p];
                        nlive = 0;
                        for (uint p = 0; p < np; p++) {
                            if (((fm >> p) & 1) == 0) {
                                if (p == trav) post_trav = nlive;
                                post_hand[nlive] = h[p];
                                // my real share (blind split + line contribution)
                                // minus what the cell root will claim for me.
                                seam_extra[nlive] = ((float)c_spot[cur] / (float)np + (float)c_contrib[bc_pre + p])
                                    - ((float)c_spot[cell_root] / max(1.0f, (float)(np - (uint)popcount((uint)fm))) + (float)c_contrib[bc_cell + nlive]);
                                nlive++;
                            }
                        }
                        // pot delta: real line pot vs the cell's claimed pot.
                        float cell_pot = (float)c_spot[cell_root];
                        for (uint q = 0; q < nlive; q++) cell_pot += (float)c_contrib[bc_cell + q];
                        seam_pot_extra = real_pot - cell_pot;
                    }
                    cur = children[c_chstart[cur]];
                    continue;
                }
                bool pre = (c_pre[cur] == 1);
                uint actor = c_player[cur];
                if (ty == 0) { // terminal
                    // who is "traverser" + the hand set in this region
                    uint who = pre ? trav : post_trav;
                    uint useN = pre ? np : nlive;
                    ushort fm = c_fold[cur];
                    uint bc = cur * np;
                    int c_t = c_contrib[bc + who];
                    int total_i = c_spot[cur];
                    for (uint p = 0; p < useN; p++) total_i += c_contrib[bc + p];
                    // SEAM MONEY: postflop terminals live in a SHARED cell tree —
                    // re-base pot + my-share to the ACTUAL entry line's money.
                    float total = (float)total_i + (pre ? 0.0f : seam_pot_extra);
                    float half_pot = (float)c_spot[cur] / (float)useN + (float)c_t + (pre ? 0.0f : seam_extra[who]);
                    if (((fm >> who) & 1) == 1) { ret = -half_pot; stop = true; returning = true; break; }
                    if (pre) {
                        // preflop fold-win terminal (uncontested) — no showdown, no rake
                        ret = total - half_pot; stop = true; returning = true; break;
                    }
                    // postflop showdown (over live slots)
                    uchar bs = c_bs[cur];
                    bool dead = false;
                    for (uint p = 0; p < useN; p++) {
                        if (((fm >> p) & 1) == 0) {
                            uint hp = post_hand[p];
                            if (bs >= 1 && t_alive[foff_t + taoff + hp] == 0) dead = true;
                            if (bs >= 2 && r_alive[foff_r + roff + hp] == 0) dead = true;
                        }
                    }
                    if (dead) { ret = 0.0f; stop = true; returning = true; break; }
                    int minlev = 2147483647;
                    for (uint p = 0; p < useN; p++) { int c = c_contrib[bc + p]; if (c < minlev) minlev = c; }
                    int cnt = 0; for (uint p = 0; p < useN; p++) if (c_contrib[bc + p] >= minlev) cnt++;
                    float main_pot = (float)(minlev * cnt + c_spot[cur]) + seam_pot_extra;
                    float rake = min(max(main_pot * P.rake_rate, 0.0f), P.rake_cap);
                    float net_pot = total - rake;
                    // EXACT showdown (Pluribus-faithful): score actual sampled hands.
                    int s_t = strength[foff_r + roff + post_hand[who]];
                    uint better = 0, equal = 0; bool anyopp = false;
                    for (uint p = 0; p < useN; p++) if (p != who && ((fm >> p) & 1) == 0) {
                        anyopp = true; int s_o = strength[foff_r + roff + post_hand[p]];
                        if (s_o > s_t) better++; else if (s_o == s_t) equal++;
                    }
                    if (!anyopp) ret = net_pot - half_pot;
                    else if (better > 0) ret = -half_pot;
                    else ret = net_pot / (float)(equal + 1) - half_pot;
                    stop = true; returning = true; break;
                }
                // player node
                uint na = c_nch[cur];
                uint hand = pre ? h[actor] : post_hand[actor];
                uint bucket = pre ? hand_cls[foff_f + hand] : ((c_bs[cur] == 0) ? flop_b[foff_f + hand] : ((c_bs[cur] == 1) ? turn_b[foff_t + taoff + hand] : river_b[foff_r + roff + hand]));
                // A player-node bucket of NO_BUCKET (65535) means the actor's hand
                // conflicts the sampled runout → impossible world. The dead-check only
                // guards terminals, so without this a turn/river player node would index
                // regret at (local*nb + 65535)*maxna — far OOB (the batch/layout-sensitive
                // blowup). Treat as dead: value 0. (Defensive in both bucketings.)
                if (bucket >= c_nb[cur]) { ret = 0.0f; stop = true; returning = true; break; }
                // postflop regret region is per-flop (preflop is shared ⇒ no fi offset).
                uint base = c_regbase[cur] + (pre ? 0u : freg) + ((uint)c_local[cur] * c_nb[cur] + bucket) * maxna;
                float st[CL_MAXNA]; float sum = 0.0f;
                for (uint a = 0; a < na; a++) { float r = max(atomic_load_explicit(&regret[base+a], memory_order_relaxed), 0.0f); st[a] = r; sum += r; }
                if (sum > 0.0f) { for (uint a = 0; a < na; a++) st[a] /= sum; } else { for (uint a = 0; a < na; a++) st[a] = 1.0f/(float)na; }
                uint who2 = pre ? trav : post_trav;
                // forced prefix hop (targeted trajectory, preflop, non-traverser
                // actor by construction): take the prescribed action, accumulate
                // the importance weight, descend.
                if (pre && t_hop < t_len) {
                    uint fa = (uint)prefix_act[t_pi * P.prefix_stride + t_hop];
                    t_hop++;
                    // ε-FLOORED importance weight: pure w=σ zeroes out whenever
                    // the dealt hand never takes the forced action (converged
                    // pure strategies ⇒ nearly all trajectories wasted). The
                    // floor trains the defender vs "true range + ε·any-two" —
                    // a benignly-wider defense, and every trajectory counts.
                    w *= max(st[fa], P.target_eps);
                    cur = children[c_chstart[cur] + fa];
                    continue;
                }
                if (actor == who2) {
                    // PRUNING: skip regret≤prune_c subtrees (not river; never all actions).
                    uint pmask = 0;
                    if (prune_this && c_bs[cur] != 2) {
                        uint cnt = 0;
                        for (uint a = 0; a < na; a++) {
                            if (atomic_load_explicit(&regret[base + a], memory_order_relaxed) <= P.prune_c) { pmask |= (1u << a); cnt++; }
                        }
                        if (cnt >= na) pmask = 0;
                    }
                    sp++;
                    stack[sp].node = cur; stack[sp].na = na; stack[sp].base = base; stack[sp].pruned = pmask;
                    for (uint a = 0; a < na; a++) { stack[sp].st[a] = st[a]; stack[sp].cv[a] = 0.0f; }
                    uint a0 = 0; while (a0 < na && ((pmask >> a0) & 1)) a0++;
                    stack[sp].ai = a0;
                    cur = children[c_chstart[cur] + a0];
                    continue;
                } else {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                    float r = (float)((uint)(s >> 32)) * (1.0f / 4294967296.0f);  // high 32 bits (xorshift64 low bits are low-quality)
                    float acc = 0.0f; uint a = na - 1;
                    for (uint i = 0; i < na; i++) { acc += st[i]; if (r <= acc) { a = i; break; } }
                    cur = children[c_chstart[cur] + a];
                    continue;
                }
            }
        } else {
            if (sp < 0) break;
            stack[sp].cv[stack[sp].ai] = ret;
            stack[sp].ai++;
            while (stack[sp].ai < stack[sp].na && ((stack[sp].pruned >> stack[sp].ai) & 1)) stack[sp].ai++;
            if (stack[sp].ai < stack[sp].na) {
                cur = children[c_chstart[stack[sp].node] + stack[sp].ai];
                returning = false;
                continue;
            }
            float v = 0.0f;
            for (uint a = 0; a < stack[sp].na; a++) v += stack[sp].st[a] * stack[sp].cv[a];
            // FREEZE-PREFLOP (reach-prior per-flop solve): when P.freeze_pre, preflop
            // nodes are READ for action selection but never updated — the preflop
            // strategy stays fixed while this flop's postflop converges to its reach.
            bool frozen = (P.freeze_pre != 0u) && (c_pre[stack[sp].node] == 1);
            if (!frozen) {
                for (uint a = 0; a < stack[sp].na; a++) {
                    if ((stack[sp].pruned >> a) & 1) continue; // pruned: regret/cum unchanged
                    float delta = w * (stack[sp].cv[a] - v);
                    float old = atomic_load_explicit(&regret[stack[sp].base + a], memory_order_relaxed);
                    float newv;
                    do { newv = max(old + delta, 0.0f); }
                    while (!atomic_compare_exchange_weak_explicit(&regret[stack[sp].base + a], &old, newv, memory_order_relaxed, memory_order_relaxed));
                    atomic_fetch_add_explicit(&cum[stack[sp].base + a], w * stack[sp].st[a], memory_order_relaxed);
                }
            }
            ret = v; sp--;
        }
    }
}
