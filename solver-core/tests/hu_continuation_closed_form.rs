//! De-risk the GPU continuation-leaf kernel: prove the closed-form HU Arm-1
//! continuation matches the CPU recursion `bucketed_showdown_cfv_design1_collapsed`.
//!
//! Continuation chance leaves are always reached by check/call closing the
//! street ⇒ equal contributions, fold_mask=0 ⇒ Arm 1. For HU (num_opp=1) the
//! `recurse_eq_buckets` DP collapses to, per traverser bucket bt:
//!
//!   cfv[bt] = half_pot · Σ_bo reach[bo] · [ (f_w − f_l) − rps·(f_w + f_t/2) ]
//!
//! where rps = rake_per_unit_stake, half_pot = starting_pot/np + c_t. If this
//! matches the CPU recursion bit-close on random tables/reach, the GPU port is
//! a mechanical B×B reduction.

use solver_core::solver::bucketed_showdown::{
    bucketed_showdown_cfv_design1_collapsed, BucketedRunoutTables,
};

/// Small deterministic LCG so the test needs no rng crate.
struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
}

/// Build random but VALID HU runout tables: per (bt,bo), f_w+f_t+f_l = f_n,
/// all in [0,1]. (Mirrors the win/tie/lose/compat invariant.)
fn random_tables(nb: usize, rng: &mut Lcg) -> BucketedRunoutTables {
    let mut f_w = vec![0.0f32; nb * nb];
    let mut f_t = vec![0.0f32; nb * nb];
    let mut f_l = vec![0.0f32; nb * nb];
    let mut f_n = vec![0.0f32; nb * nb];
    for i in 0..nb * nb {
        let compat = 0.3 + 0.7 * rng.f(); // non-zero compatible mass
        let (a, b, c) = (rng.f(), rng.f(), rng.f());
        let s = a + b + c;
        f_w[i] = compat * a / s;
        f_t[i] = compat * b / s;
        f_l[i] = compat * c / s;
        f_n[i] = f_w[i] + f_t[i] + f_l[i];
    }
    BucketedRunoutTables { nb, f_w, f_t, f_l, f_n }
}

fn closed_form_hu(
    reach: &[f32],
    t: &BucketedRunoutTables,
    starting_pot: i32,
    c_t: i32,
    c_opp: i32,
    np: usize,
    rake_rate: f32,
    rake_cap: f32,
) -> Vec<f32> {
    let nb = t.nb;
    let half_pot = starting_pot as f32 / np as f32 + c_t as f32;
    let total_pot = starting_pot + c_t + c_opp;
    let rake = (total_pot as f32 * rake_rate).min(rake_cap).max(0.0);
    let rps = if half_pot > 0.0 { rake / half_pot } else { 0.0 };
    let mut cfv = vec![0.0f32; nb];
    for bt in 0..nb {
        let mut accum = 0.0f32;
        for bo in 0..nb {
            let r = reach[bo];
            if r == 0.0 { continue; }
            let i = bt * nb + bo;
            let fw = t.f_w[i];
            let ft = t.f_t[i];
            let fl = t.f_l[i];
            let fn_ = t.f_n[i];
            if fn_ == 0.0 { continue; }
            accum += r * ((fw - fl) - rps * (fw + ft / 2.0));
        }
        cfv[bt] = half_pot * accum;
    }
    cfv
}

#[test]
fn hu_arm1_closed_form_matches_cpu_recursion() {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let np = 2u8;

    for nb in [4usize, 8, 16, 32, 64] {
        for trial in 0..20 {
            let tables = random_tables(nb, &mut rng);
            let reach: Vec<f32> = (0..nb).map(|_| if rng.f() < 0.2 { 0.0 } else { rng.f() }).collect();

            // Equal contributions (Arm 1 precondition), varied across trials.
            let c = (trial % 5) * 7;
            let starting_pot = 6 + (trial % 3) as i32 * 4;
            // Two rake regimes: none, and a capped rake.
            for &(rake_rate, rake_cap) in &[(0.0f32, 0.0f32), (0.05, 3.0)] {
                let cpu = bucketed_showdown_cfv_design1_collapsed(
                    &[reach.as_slice()],
                    &tables,
                    &[c as i32, c as i32],
                    0, // fold_mask
                    0, // traverser
                    np,
                    starting_pot,
                    rake_rate,
                    rake_cap,
                    true, // flop_seen
                );
                let cf = closed_form_hu(
                    &reach, &tables, starting_pot, c as i32, c as i32, np as usize,
                    rake_rate, rake_cap,
                );
                let max_abs: f32 = cpu.iter().zip(&cf).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                let scale: f32 = cpu.iter().map(|x| x.abs()).fold(1e-6, f32::max);
                assert!(
                    max_abs / scale < 1e-5,
                    "nb={nb} trial={trial} rake=({rake_rate},{rake_cap}): max_abs={max_abs} scale={scale}\n cpu={:?}\n cf ={:?}",
                    &cpu[..nb.min(6)], &cf[..nb.min(6)]
                );
            }
        }
    }
}
