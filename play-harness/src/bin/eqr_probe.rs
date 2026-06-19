//! EQR realized-EV GATE (2026-06-18, preflop-RTS pivot): sanity-check the
//! standalone equity-realization terminal BEFORE wiring it into the preflop CFR.
//! Prints flop-marginalized realized EV vs all-in equity for signature hands at
//! live-2/3/5. Gates: AA ≫ trash; AA realized EV DROPS L2→L5 (multiway penalty);
//! 87s realizes > A5o (draws); realized EV < raw equity (the realization gap).
//!
//! Config knobs (the standing requirement): MC_RAKE, MC_RAKECAP (bb), MC_STACK
//! (bb), MC_ANTE (bb), plus MC_BET (bet/pot), MC_NODRAWS, MC_FLOPS, MC_SAMPLES.

use play_harness::eqr::{allin_equity_on_flop, class_combo, realized_ev_on_flop, EqrPolicy};

const UNITS_PER_BB: f32 = 2.0;

#[inline]
fn xs(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

fn rand_flop(s: &mut u64, hole: [u8; 2]) -> [u8; 3] {
    let mut used = (1u64 << hole[0]) | (1u64 << hole[1]);
    let mut f = [0u8; 3];
    for fc in f.iter_mut() {
        loop {
            let c = (xs(s) % 52) as u8;
            if used & (1u64 << c) == 0 {
                used |= 1u64 << c;
                *fc = c;
                break;
            }
        }
    }
    f
}

fn env_f(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn env_u(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() {
    let rake_rate = env_f("MC_RAKE", 0.05);
    let rake_cap_bb = env_f("MC_RAKECAP", 10.0);
    let stack_bb = env_f("MC_STACK", 100.0);
    let ante_bb = env_f("MC_ANTE", 0.0);
    let rake_cap = rake_cap_bb * UNITS_PER_BB;
    let n_flops = env_u("MC_FLOPS", 80);
    let policy = EqrPolicy {
        bet_frac: env_f("MC_BET", 1.0),
        use_draws: std::env::var("MC_NODRAWS").is_err(),
        mc_samples: env_u("MC_SAMPLES", 1500),
    };

    println!(
        "EQR GATE | rake {:.0}% cap {}bb stack {}bb ante {}bb | bet {}×pot draws={} | {} flops × {} MC",
        rake_rate * 100.0, rake_cap_bb, stack_bb, ante_bb, policy.bet_frac, policy.use_draws, n_flops, policy.mc_samples
    );
    println!("(realized EV in chip units = net vs flop-start; 1bb = {UNITS_PER_BB} units. eq = raw all-in equity.)\n");

    // (name, r0, r1, suited) — ranks 0=2 .. 12=A, r0>=r1.
    let hands: [(&str, u8, u8, bool); 12] = [
        ("AA", 12, 12, false), ("KK", 11, 11, false), ("QQ", 10, 10, false),
        ("99", 7, 7, false), ("AKs", 12, 11, true), ("AJs", 12, 9, true),
        ("KQs", 11, 10, true), ("87s", 6, 5, true), ("A5o", 12, 3, false),
        ("T9o", 8, 7, false), ("72o", 5, 0, false), ("32o", 1, 0, false),
    ];
    let lives = [2usize, 3, 5];

    println!("{:<6} {}", "hand", "  L2 rEV / eq     L3 rEV / eq     L5 rEV / eq");
    for (name, r0, r1, suited) in hands {
        let hole = class_combo(r0, r1, suited);
        print!("{name:<6} ");
        for &live in &lives {
            // limped live-way pot: each puts in 1bb + ante; pot is dead at flop.
            let pot = live as f32 * (UNITS_PER_BB + ante_bb * UNITS_PER_BB);
            let mut s = 0x1234_5678u64
                ^ ((name.len() as u64) << 8)
                ^ ((r0 as u64) << 16)
                ^ ((r1 as u64) << 24)
                ^ ((live as u64) << 32);
            let mut rev = 0.0f32;
            let mut aeq = 0.0f32;
            for fi in 0..n_flops {
                let flop = rand_flop(&mut s, hole);
                let seed = s ^ ((fi as u64).wrapping_mul(0x9E3779B97F4A7C15));
                rev += realized_ev_on_flop(hole, flop, live, pot, rake_rate, rake_cap, policy, seed);
                aeq += allin_equity_on_flop(hole, flop, live, 600, seed ^ 0xABCD);
            }
            rev /= n_flops as f32;
            aeq /= n_flops as f32;
            print!("  {rev:+6.2} / {:>3.0}%", aeq * 100.0);
        }
        println!();
    }
    println!("\nGATES:");
    println!("  • AA/KK rEV ≫ 72o/32o rEV (strength ordering)");
    println!("  • AA rEV DROPS L2→L5 while eq also drops (multiway equity-thinning + realization)");
    println!("  • 87s rEV > A5o rEV (draws realize even at similar raw equity)");
    println!("  • rEV reflects realization, NOT raw equity (trash's small eq realizes even less)");
}
