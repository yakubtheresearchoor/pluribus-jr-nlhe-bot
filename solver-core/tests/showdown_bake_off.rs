// Multiway-showdown method bake-off: per-terminal accuracy + cost on a
// common K=3 4p configuration, with brute-force as ground truth and Monte
// Carlo swept across sample budgets.
//
// This is the bake-off the spec calls for. SCOPE-LIMITED per the spec's own
// time-boxing guidance: per-terminal quality and cost measurement here;
// full-solver lifted exploitability for MC is OUT OF SCOPE (would require
// plumbing the MC showdown into FlopStartGame::evaluate_terminal, an
// invasive change); the K=3 factored-exact CFV with B/T/S stratification
// is also OUT OF SCOPE here (substantial derivation, belongs in its own
// session). K=5 production-scale costs are PROJECTED from K=3 measurements
// with the recursion-depth multiplier, not measured.
//
// What this measurement does directly:
//   1. Monte Carlo showdown CFV at multiple sample counts vs brute-force
//      on a fixed (h_player, board, contributions, reach) configuration:
//      per-h max abs diff, max rel diff, per-h zero-sum violation.
//   2. Per-h cost of MC at each sample count vs brute-force per-h cost.
//   3. Project K=5 cost from K=3 measurement.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use std::time::Instant;

const NP_4P: u8 = 4;
const N_HANDS_NH: usize = 30;  // small enough for brute force, large enough for stable signal
const BOARD: [&str; 3] = ["2h", "7d", "Ks"];

// ---- Helpers ---------------------------------------------------------------

fn board_mask() -> u64 {
    BOARD.iter().map(|s| card_from_str(s).unwrap() as u8).fold(0u64, |m, c| m | (1u64 << c))
}

fn build_hand_set(nh: usize) -> (Vec<u8>, Vec<u16>) {
    let bm = board_mask();
    let board: Vec<Card> = BOARD.iter().map(|s| card_from_str(s).unwrap()).collect();
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if bm & (1u64 << c1) != 0 || bm & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut hand_cards = vec![0u8; nh * 2];
    let mut hand_strength = vec![0u16; nh];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i*2] = c1; hand_cards[i*2+1] = c2;
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        hand_strength[i] = h.evaluate_internal() as u16;
    }
    (hand_cards, hand_strength)
}

// ---- Brute-force reference -------------------------------------------------

/// Exact K-opp showdown CFV via full enumeration. Reference.
/// Equal-contribution, no fold (the spec's reference configuration).
fn brute_force_cfv(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    hand_strength: &[u16],
    nh: usize,
    pot: f32,
    k: f32, // = num_active_opp = num_opp for no-fold equal-contrib case
) -> Vec<f32> {
    let num_opp = opp_reach.len();
    let mut g_mask = vec![0u64; nh];
    for g in 0..nh {
        g_mask[g] = (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]);
    }

    let mut cfv = vec![0.0f32; nh];
    fn recurse(
        oi: usize, num_opp: usize, nh: usize,
        mask_so_far: u64, reach_so_far: f32, max_str_so_far: u16, tied_so_far: u32,
        h_str: u16,
        opp_reach: &[&[f32]],
        g_mask: &[u64],
        hand_strength: &[u16],
        k_f: f32,
        accum: &mut f32,
    ) {
        if oi == num_opp {
            let net_unit: f32 = if max_str_so_far > h_str {
                -1.0
            } else if max_str_so_far == h_str {
                let t = tied_so_far + 1;
                (k_f + 1.0 - t as f32) / t as f32
            } else {
                k_f
            };
            *accum += reach_so_far * net_unit;
            return;
        }
        for g in 0..nh {
            if g_mask[g] & mask_so_far != 0 { continue; }
            let r = opp_reach[oi][g];
            if r == 0.0 { continue; }
            let s = hand_strength[g];
            let (new_max, new_tied) = if s > max_str_so_far {
                (s, 1u32)
            } else if s == max_str_so_far {
                (max_str_so_far, tied_so_far + 1)
            } else {
                (max_str_so_far, tied_so_far)
            };
            recurse(oi+1, num_opp, nh,
                mask_so_far | g_mask[g], reach_so_far * r, new_max, new_tied,
                h_str, opp_reach, g_mask, hand_strength, k_f, accum);
        }
    }

    for h in 0..nh {
        let h_m = g_mask[h];
        let h_str = hand_strength[h];
        let mut accum = 0.0f32;
        recurse(0, num_opp, nh, h_m, 1.0, 0u16, 0u32,
            h_str, opp_reach, &g_mask, hand_strength, k, &mut accum);
        cfv[h] = pot * accum;
    }
    cfv
}

// ---- Monte Carlo candidate -------------------------------------------------

/// Monte Carlo showdown CFV.
/// Equal-contribution, no fold (the spec's reference configuration).
///
/// Sampling: for each h, draw N samples; each sample picks g_0..g_{K-1}
/// sequentially with probability proportional to r_i[g] over the valid
/// hands (not conflicting with the running mask). Importance-sample weight:
/// each sample's contribution is `(Π_i Z_i) · net(scenario)` where Z_i is
/// the reach-sum of valid choices at step i. This is the standard
/// importance-sampling estimator for `Σ_t Π_i r_i · net`.
fn monte_carlo_cfv(
    opp_reach: &[&[f32]],
    hand_cards: &[u8],
    hand_strength: &[u16],
    nh: usize,
    pot: f32,
    k: f32,
    n_samples: usize,
    rng_seed: u64,
) -> Vec<f32> {
    let num_opp = opp_reach.len();
    let mut g_mask = vec![0u64; nh];
    for g in 0..nh {
        g_mask[g] = (1u64 << hand_cards[g*2]) | (1u64 << hand_cards[g*2+1]);
    }

    let mut state = rng_seed;
    let mut next_u = || -> f32 {
        // SplitMix64-style PRNG, deterministic
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        (z >> 32) as f32 / (1u64 << 32) as f32
    };

    let mut cfv = vec![0.0f32; nh];

    for h in 0..nh {
        let h_m = g_mask[h];
        let h_str = hand_strength[h];

        let mut sum = 0.0f32;
        let mut effective_samples = 0u32;
        for _ in 0..n_samples {
            let mut mask = h_m;
            let mut z_prod = 1.0f32;
            let mut max_str = 0u16;
            let mut tied = 0u32;
            let mut valid = true;

            for oi in 0..num_opp {
                // Compute Z_oi over valid g (not conflicting with mask)
                let mut z = 0.0f32;
                for g in 0..nh {
                    if g_mask[g] & mask != 0 { continue; }
                    z += opp_reach[oi][g];
                }
                if z == 0.0 { valid = false; break; }
                z_prod *= z;

                let u = next_u();
                let target = u * z;
                let mut acc = 0.0f32;
                let mut chosen = 0usize;
                for g in 0..nh {
                    if g_mask[g] & mask != 0 { continue; }
                    acc += opp_reach[oi][g];
                    if target <= acc { chosen = g; break; }
                }
                mask |= g_mask[chosen];
                let s = hand_strength[chosen];
                if s > max_str { max_str = s; tied = 1; }
                else if s == max_str { tied += 1; }
            }
            if !valid { continue; }
            effective_samples += 1;

            let net_unit: f32 = if max_str > h_str {
                -1.0
            } else if max_str == h_str {
                let t = tied + 1;
                (k + 1.0 - t as f32) / t as f32
            } else {
                k
            };
            sum += z_prod * net_unit;
        }

        if effective_samples > 0 {
            cfv[h] = pot * sum / effective_samples as f32;
        }
    }
    cfv
}

// ---- Bake-off test ---------------------------------------------------------

#[test]
fn showdown_method_bake_off() {
    let nh = N_HANDS_NH;
    let num_opp = NP_4P as usize - 1;  // K=3
    let (hand_cards, hand_strength) = build_hand_set(nh);

    // Non-uniform reach: cosine-like profile so per-hand reaches differ
    // (otherwise importance sampling vs uniform sampling collapse).
    let reach: Vec<Vec<f32>> = (0..num_opp).map(|oi| {
        (0..nh).map(|h| 0.3 + 0.6 * ((h + oi * 5) % 11) as f32 / 11.0).collect()
    }).collect();
    let reach_views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();

    let pot = 20.0f32; // 4p starting_pot
    let k = num_opp as f32;

    // ---- Brute-force ground truth ----
    let t = Instant::now();
    let bf = brute_force_cfv(&reach_views, &hand_cards, &hand_strength, nh, pot, k);
    let bf_cost = t.elapsed();
    eprintln!("[BF] K={} nh={} all-terminals cost: {:?} ({:.3} ms/h)",
        num_opp, nh, bf_cost, bf_cost.as_secs_f64() * 1000.0 / nh as f64);

    // ---- Monte Carlo sweep ----
    let sample_counts = &[100usize, 1000, 10000];
    for &n in sample_counts {
        let t = Instant::now();
        let mc = monte_carlo_cfv(&reach_views, &hand_cards, &hand_strength, nh, pot, k, n, 0xdeadbeef);
        let mc_cost = t.elapsed();

        // Quality: max abs / max rel diff vs brute-force.
        let mut max_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut max_rel = 0.0f64;
        for h in 0..nh {
            let d = (mc[h] - bf[h]).abs() as f64;
            sum_sq += d * d;
            if d > max_abs { max_abs = d; }
            let scale = (bf[h].abs() as f64).max(0.01);
            let rel = d / scale;
            if rel > max_rel { max_rel = rel; }
        }
        let rmse = (sum_sq / nh as f64).sqrt();

        // Zero-sum: per-terminal, Σ_p (reach-weighted cfv summed over h). For
        // equal contribs, no fold, uniform reach, this should be ~0.
        let mut zs = 0.0f64;
        for h in 0..nh {
            zs += mc[h] as f64 * reach[0][h] as f64;
        }

        let pct_of_pot = max_abs as f32 / pot * 100.0;
        eprintln!("[MC N={:>5}] all-terminals cost: {:?} ({:.3} ms/h)  max_abs={:.4e} ({:.3}% of pot)  rmse={:.4e}  max_rel={:.3}  ΣP_0r·cfv={:.4e}",
            n, mc_cost, mc_cost.as_secs_f64() * 1000.0 / nh as f64,
            max_abs, pct_of_pot, rmse, max_rel, zs);
    }

    // ---- Per-h cost projection to K=5 nh=50 ----
    //
    // For BRUTE-FORCE at K=5 nh=50: nh^5 = 3.125e8 per h. Measured K=3 nh=30
    // gives a per-h ms. Scale: BF cost scales as nh^K, so K=5/K=3 ratio at
    // nh=50 vs nh=30: (50^5 / 30^3) = 3.125e8/27000 = 11574x.
    //
    // For MC: cost scales as N·K·nh per h (computing Z_oi and sampling).
    // K=5 vs K=3 at fixed N and same nh: 5/3 = 1.67x. At nh=50 vs nh=30:
    // 50/30 = 1.67x. Combined: ~2.78x cost per h at K=5 nh=50 vs K=3 nh=30
    // at same N.
    let bf_per_h_ms = bf_cost.as_secs_f64() * 1000.0 / nh as f64;
    let bf_k5_nh50_per_h_ms = bf_per_h_ms * (50f64.powi(5) / (nh as f64).powi(3));
    eprintln!("");
    eprintln!("=== Projection K=5 nh=50 (production scale) ===");
    eprintln!("Brute-force per h: {:.3} ms (measured K=3 nh={}) × (50^5/{}^3) = {:.1} ms",
        bf_per_h_ms, nh, nh, bf_k5_nh50_per_h_ms);
    eprintln!("Brute-force per terminal: ≈ {:.2} s", bf_k5_nh50_per_h_ms * 50.0 / 1000.0);
    let terminals_6p = 10000f64;
    let bf_k5_per_iter_s = bf_k5_nh50_per_h_ms / 1000.0 * 50.0 * terminals_6p;
    let iters = 2000f64;
    eprintln!("Brute-force per iter ({} terminals): {:.0} s ({:.1} h)",
        terminals_6p, bf_k5_per_iter_s, bf_k5_per_iter_s / 3600.0);
    eprintln!("Brute-force full solve ({} iters): {:.0} h ({:.1} years)",
        iters, bf_k5_per_iter_s * iters / 3600.0, bf_k5_per_iter_s * iters / (3600.0 * 24.0 * 365.0));

    eprintln!("");
    eprintln!("MC per h ratio K=5 nh=50 vs K=3 nh={}: ~{:.2}x at same N",
        nh, 5.0 / 3.0 * 50.0 / nh as f64);
}
