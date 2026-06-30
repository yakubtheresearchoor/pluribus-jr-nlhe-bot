//! De-risk the GPU multiway Arm-2 (side-pot) kernel by replicating its exact
//! per-tuple net_expected logic in Rust and comparing — EXHAUSTIVELY over all
//! bucket tuples, no MC — to the CPU reference
//! `bucketed_showdown_cfv_design1_collapsed`. Non-degenerate reach + genuine
//! Arm-2 configs (a folded player, unequal contributions). If the port logic
//! matches here, the Metal kernel (a transliteration of this Rust) is correct;
//! the GPU MC path then only adds sampling noise.

use solver_core::solver::bucketed_showdown::{
    bucketed_showdown_cfv_design1_collapsed, BucketedRunoutTables,
};

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
}

fn random_tables(nb: usize, rng: &mut Lcg) -> BucketedRunoutTables {
    let (mut f_w, mut f_t, mut f_l, mut f_n) =
        (vec![0.0f32; nb*nb], vec![0.0f32; nb*nb], vec![0.0f32; nb*nb], vec![0.0f32; nb*nb]);
    for i in 0..nb*nb {
        let compat = 0.4 + 0.6 * rng.f();
        let (a, b, c) = (rng.f(), rng.f(), rng.f());
        let s = a + b + c;
        f_w[i] = compat*a/s; f_t[i] = compat*b/s; f_l[i] = compat*c/s; f_n[i] = f_w[i]+f_t[i]+f_l[i];
    }
    BucketedRunoutTables { nb, f_w, f_t, f_l, f_n }
}

/// Rust replica of the Metal `vcfr_continuation_showdown_mw` Arm-2 path,
/// EXHAUSTIVE over bucket tuples (the M→∞ limit of the kernel's sampler).
#[allow(clippy::too_many_arguments)]
fn kernel_arm2_replica(
    bucket_reach: &[Vec<f32>], // [num_opp][nb]
    t: &BucketedRunoutTables,
    contribs: &[i32],
    fold_mask: u16,
    traverser: usize,
    np: usize,
    starting_pot: i32,
) -> Vec<f32> {
    let nb = t.nb;
    let num_opp = np - 1;
    let c_t = contribs[traverser];
    let trav_folded = fold_mask & (1 << traverser) != 0;
    let traverser_stake = starting_pot as f32 / np as f32 + c_t as f32;
    let opp_pl: Vec<usize> = (0..num_opp).map(|oi| if oi < traverser { oi } else { oi + 1 }).collect();
    let opp_c: Vec<i32> = opp_pl.iter().map(|&p| contribs[p]).collect();
    let opp_f: Vec<bool> = opp_pl.iter().map(|&p| fold_mask & (1 << p) != 0).collect();

    // levels (sorted unique contributions)
    let mut lv: Vec<i32> = contribs.to_vec();
    lv.sort(); lv.dedup();
    let main_amt = if lv.is_empty() { starting_pot } else {
        let num_main = (0..np).filter(|&p| contribs[p] >= lv[0]).count() as i32;
        lv[0]*num_main + starting_pot
    };
    let _ = main_amt; // rake 0 in this test

    let mut cfv = vec![0.0f32; nb];
    let mut bo = vec![0usize; num_opp];
    for bt in 0..nb {
        cfv[bt] = exhaustive_arm2(bt, num_opp, nb, &mut bo, bucket_reach, t, &opp_f, &opp_c,
            contribs, np, traverser, c_t, trav_folded, traverser_stake, &lv, starting_pot);
    }
    cfv
}

#[allow(clippy::too_many_arguments)]
fn exhaustive_arm2(
    bt: usize, num_opp: usize, nb: usize, bo: &mut [usize], br: &[Vec<f32>],
    t: &BucketedRunoutTables, opp_f: &[bool], opp_c: &[i32], contribs: &[i32], np: usize,
    traverser: usize, c_t: i32, trav_folded: bool, traverser_stake: f32, lv: &[i32], starting_pot: i32,
) -> f32 {
    // iterative odometer over num_opp buckets
    let mut accum = 0.0f32;
    let total: usize = nb.pow(num_opp as u32);
    for code in 0..total {
        let mut c = code;
        let mut reach_w = 1.0f32;
        for o in 0..num_opp { bo[o] = c % nb; c /= nb; reach_w *= br[o][bo[o]]; }
        if reach_w == 0.0 { continue; }
        // sc + wpair
        let mut scw = vec![0.0f32; num_opp]; let mut sct = vec![0.0f32; num_opp];
        let mut scl = vec![0.0f32; num_opp]; let mut scn = vec![0.0f32; num_opp];
        let mut wpair = 1.0f32; let mut dead = false;
        for o in 0..num_opp {
            let b = bo[o];
            for p in 0..o { let f = t.f_n[bo[p]*nb + b]; if f == 0.0 { dead = true; break; } wpair *= f; }
            if dead { break; }
            let i = bt*nb + b;
            if opp_f[o] { let n = t.f_n[i]; if n == 0.0 { dead = true; break; } scn[o] = n; }
            else { let (w,tt,l) = (t.f_w[i], t.f_t[i], t.f_l[i]); if w==0.0&&tt==0.0&&l==0.0 { dead = true; break; } scw[o]=w; sct[o]=tt; scl[o]=l; }
        }
        if dead { continue; }
        let mut s_total = 1.0f32;
        for o in 0..num_opp { s_total *= if opp_f[o] { scn[o] } else { scw[o]+sct[o]+scl[o] }; }
        let mut cash = 0.0f32; let mut prev_l = 0i32;
        for (liv, &lev) in lv.iter().enumerate() {
            let pc = lev - prev_l;
            let num_contrib = (0..np).filter(|&p| contribs[p] >= lev).count() as i32;
            let mut pot_l = (pc*num_contrib) as f32;
            if liv == 0 { pot_l += starting_pot as f32; }
            if pot_l == 0.0 { prev_l = lev; continue; }
            let trav_elig = !trav_folded && c_t >= lev;
            let mut elig = trav_elig as i32;
            for o in 0..num_opp { if opp_f[o] { continue; } if opp_c[o] < lev { continue; } elig += 1; }
            if elig == 0 {
                if contribs[traverser] >= lev {
                    let tcl = pc as f32 + if liv==0 { starting_pot as f32/np as f32 } else { 0.0 };
                    cash += tcl * s_total;
                }
                prev_l = lev; continue;
            }
            if !trav_elig { prev_l = lev; continue; }
            let mut m_out = 1.0f32;
            for o in 0..num_opp { if opp_f[o] { m_out *= scn[o]; } else if opp_c[o] < lev { m_out *= scw[o]+sct[o]+scl[o]; } }
            let mut dp = vec![0.0f32; num_opp+2]; dp[0] = 1.0; let mut ne = 0usize;
            for o in 0..num_opp {
                if opp_f[o] || opp_c[o] < lev { continue; }
                let (w, tt) = (scw[o], sct[o]); dp[ne+1] = 0.0;
                for j in (0..=ne).rev() { let d = dp[j];
                    if d != 0.0 && tt != 0.0 { dp[j+1] += d*tt; }
                    dp[j] = if d != 0.0 && w != 0.0 { d*w } else { 0.0 }; }
                ne += 1;
            }
            for j in 0..=ne { let d = dp[j]; if d == 0.0 { continue; } cash += m_out*d*(pot_l/(j+1) as f32); }
            prev_l = lev;
        }
        let val = cash - traverser_stake * s_total;
        accum += reach_w * wpair * val;
    }
    accum
}

#[test]
fn mw_arm2_kernel_replica_matches_cpu() {
    let mut rng = Lcg(0xA11CE_5EED);
    let nb = 6usize;
    for trial in 0..30 {
        let np = if trial % 2 == 0 { 3 } else { 4 };
        let num_opp = np - 1;
        let tables = random_tables(nb, &mut rng);
        let reach: Vec<Vec<f32>> = (0..num_opp)
            .map(|_| (0..nb).map(|_| if rng.f() < 0.15 { 0.0 } else { rng.f() }).collect())
            .collect();
        // genuine Arm-2: one opponent folds (lower contribution), others equal.
        let traverser = np - 1;
        let mut contribs = vec![40i32; np];
        let folder = trial % np; // some player folds (INCLUDING the traverser)
        let fold_mask = 1u16 << folder;
        contribs[folder] = if trial % 3 == 0 { 0 } else { 15 };

        let views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
        let cpu = bucketed_showdown_cfv_design1_collapsed(
            &views, &tables, &contribs, fold_mask, traverser, np as u8, 30, 0.0, 0.0, true,
        );
        let mine = kernel_arm2_replica(&reach, &tables, &contribs, fold_mask, traverser, np, 30);
        let max_abs: f32 = cpu.iter().zip(&mine).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
        let scale: f32 = cpu.iter().map(|x| x.abs()).fold(1e-4, f32::max);
        assert!(max_abs/scale < 1e-4,
            "trial={trial} np={np} fold={folder} contribs={contribs:?}: max_abs={max_abs} scale={scale}\n cpu={:?}\n mine={:?}",
            &cpu[..nb.min(6)], &mine[..nb.min(6)]);
    }
}
