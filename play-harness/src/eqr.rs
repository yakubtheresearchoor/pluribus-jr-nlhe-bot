//! EQUITY-REALIZATION (EQR) terminal model for the standalone preflop solver.
//!
//! The preflop terminal value is NOT raw all-in equity (too dumb — overvalues
//! strong hands multiway and ignores playability) and NOT check-down-to-showdown
//! (which collapses EXACTLY to all-in equity: with no betting, folding air loses
//! the same as showing it down). Realization needs a BET: folding to a bet saves
//! chips, betting when ahead wins extra.
//!
//! v0 policy (the fidelity lever — tune + validate): on the flop, each player
//! CONTINUES (invests one `bet_frac`-of-pot bet) iff they paired with their own
//! cards or hold a flush/OESD draw; otherwise they FOLD (forfeit their pot share
//! but save the bet). Continuers contest `pot + Σbets`; the best final 7-card hand
//! wins (rake on the contested pot). This captures: air folds (saves the bet),
//! made hands/draws realize, multiway thins strong hands (more continuers → more
//! ways to be beaten), and "raise to isolate" emerges without a postflop solve.
//!
//! Rake %, rake cap, stack, ante all flow in via the cell pot + GameSpec.

use clean_rules::best5;
use clean_rules::eval::{rank_of, suit_of};

#[derive(Clone, Copy)]
pub struct EqrPolicy {
    /// Bet size as a fraction of the pot — the realization "risk" a continuer pays.
    pub bet_frac: f32,
    /// Count flush / open-ended straight draws as continue (so 87s realizes > A5o).
    pub use_draws: bool,
    /// Monte-Carlo samples (opponents × runout) per (hand, flop, cell).
    pub mc_samples: usize,
}

impl Default for EqrPolicy {
    fn default() -> Self {
        EqrPolicy { bet_frac: 1.0, use_draws: true, mc_samples: 3000 }
    }
}

#[inline]
fn xorshift(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

#[inline]
fn draw_card(s: &mut u64, used: &mut u64) -> u8 {
    loop {
        let c = (xorshift(s) % 52) as u8;
        if *used & (1u64 << c) == 0 {
            *used |= 1u64 << c;
            return c;
        }
    }
}

/// Flop-visible continue test: paired WITH MY cards (pocket pair or hit the board),
/// or — if draws on — a flush draw (4+ of a suit) or open-ended straight draw.
#[inline]
fn flop_continue(hole: [u8; 2], flop: [u8; 3], use_draws: bool) -> bool {
    let (h0, h1) = (rank_of(hole[0]), rank_of(hole[1]));
    if h0 == h1 {
        return true; // pocket pair
    }
    let br = [rank_of(flop[0]), rank_of(flop[1]), rank_of(flop[2])];
    if br.contains(&h0) || br.contains(&h1) {
        return true; // paired the board with a hole card
    }
    if !use_draws {
        return false;
    }
    let cards = [hole[0], hole[1], flop[0], flop[1], flop[2]];
    // flush draw: 4+ of one suit among the 5 known cards.
    let mut suits = [0u8; 4];
    for &c in &cards {
        suits[suit_of(c) as usize] += 1;
    }
    if suits.iter().any(|&n| n >= 4) {
        return true;
    }
    // open-ended straight draw: 4 distinct consecutive ranks, both ends extendable.
    let mut present = [false; 13];
    for &c in &cards {
        present[rank_of(c) as usize] = true;
    }
    for lo in 1..=8usize {
        if present[lo] && present[lo + 1] && present[lo + 2] && present[lo + 3] {
            return true;
        }
    }
    false
}

#[inline]
fn rank7(hole: [u8; 2], board: [u8; 5]) -> clean_rules::HandRank {
    best5(&[hole[0], hole[1], board[0], board[1], board[2], board[3], board[4]])
}

/// One realized-EV sample under the v0 one-bet policy. Returns net chips vs
/// flop-start (baseline = each live player's pot share, `pot/live`).
#[allow(clippy::too_many_arguments)]
fn realized_ev_sample(
    my_hole: [u8; 2],
    opp_holes: &[[u8; 2]],
    flop: [u8; 3],
    turn: u8,
    river: u8,
    pot: f32,
    live: usize,
    bet: f32,
    rake_rate: f32,
    rake_cap: f32,
    use_draws: bool,
) -> f32 {
    let half_pot = pot / live as f32; // showdown-convention investment baseline
    if !flop_continue(my_hole, flop, use_draws) {
        return -half_pot; // fold the flop: forfeit pot share, save the bet
    }
    let continuers: Vec<usize> = (0..opp_holes.len())
        .filter(|&j| flop_continue(opp_holes[j], flop, use_draws))
        .collect();
    let n_cont = continuers.len() + 1; // including me
    if n_cont == 1 {
        // everyone else folded → win the pot uncontested (my bet returned).
        let rake = (pot * rake_rate).min(rake_cap).max(0.0);
        return (pot - rake) - half_pot;
    }
    let board = [flop[0], flop[1], flop[2], turn, river];
    let my_rank = rank7(my_hole, board);
    let mut maxr = my_rank;
    for &j in &continuers {
        let r = rank7(opp_holes[j], board);
        if r > maxr {
            maxr = r;
        }
    }
    let mut winners = if my_rank == maxr { 1u32 } else { 0 };
    let me_wins = my_rank == maxr;
    for &j in &continuers {
        if rank7(opp_holes[j], board) == maxr {
            winners += 1;
        }
    }
    let contested = pot + n_cont as f32 * bet;
    let rake = (contested * rake_rate).min(rake_cap).max(0.0);
    let net = contested - rake;
    let my_invest = half_pot + bet;
    if me_wins {
        net / winners as f32 - my_invest
    } else {
        -my_invest
    }
}

/// Flop-marginalized realized EV for `my_hole` in a `live`-way pot of size `pot`,
/// Monte-Carlo over opponents + runout (flop passed in; caller marginalizes flops).
#[allow(clippy::too_many_arguments)]
pub fn realized_ev_on_flop(
    my_hole: [u8; 2],
    flop: [u8; 3],
    live: usize,
    pot: f32,
    rake_rate: f32,
    rake_cap: f32,
    policy: EqrPolicy,
    seed: u64,
) -> f32 {
    let mut s = seed | 1;
    let base_used: u64 = (1u64 << my_hole[0])
        | (1u64 << my_hole[1])
        | (1u64 << flop[0])
        | (1u64 << flop[1])
        | (1u64 << flop[2]);
    let bet = policy.bet_frac * pot;
    let mut acc = 0.0f64;
    for _ in 0..policy.mc_samples {
        let mut used = base_used;
        let mut opp = Vec::with_capacity(live - 1);
        for _ in 0..live - 1 {
            let a = draw_card(&mut s, &mut used);
            let b = draw_card(&mut s, &mut used);
            opp.push([a, b]);
        }
        let turn = draw_card(&mut s, &mut used);
        let river = draw_card(&mut s, &mut used);
        acc += realized_ev_sample(
            my_hole, &opp, flop, turn, river, pot, live, bet, rake_rate, rake_cap, policy.use_draws,
        ) as f64;
    }
    (acc / policy.mc_samples as f64) as f32
}

/// Realized EV for ALL combos in `layout` on one `flop`, AMORTIZED MC: deal
/// opponents + runout once per sample, evaluate every non-conflicting combo
/// against that shared sample. Returns per-combo realized EV in layout order, on
/// the SAME chip scale as `root_cfv_from_avg` (net vs flop-start, baseline
/// pot/live). This is what the EQR oracle returns per (cell, canonical flop); the
/// preflop CFR then marginalizes over flops automatically.
#[allow(clippy::too_many_arguments)]
pub fn realized_ev_table_on_flop(
    layout: &[(u8, u8)],
    flop: [u8; 3],
    live: usize,
    pot: f32,
    rake_rate: f32,
    rake_cap: f32,
    policy: EqrPolicy,
    seed: u64,
) -> Vec<f32> {
    use rayon::prelude::*;
    let n = layout.len();
    let half_pot = pot / live as f32;
    let bet = policy.bet_frac * pot;
    let flop_mask = (1u64 << flop[0]) | (1u64 << flop[1]) | (1u64 << flop[2]);
    // Parallel over samples: each sample is independent with its own seeded RNG;
    // fold into per-thread (sum,cnt) accumulators, then reduce.
    let (sum, cnt) = (0..policy.mc_samples)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; n], vec![0u32; n]),
            |(mut sum, mut cnt), i| {
                let mut s = (seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15)) | 1;
                let mut used = flop_mask;
        let mut opp = [[0u8; 2]; 6];
        for o in opp.iter_mut().take(live - 1) {
            *o = [draw_card(&mut s, &mut used), draw_card(&mut s, &mut used)];
        }
        let turn = draw_card(&mut s, &mut used);
        let river = draw_card(&mut s, &mut used);
        let dealt = used;
        let board = [flop[0], flop[1], flop[2], turn, river];
        // precompute opponent continue + final rank (combo-independent).
        let mut opp_cont = [false; 6];
        let mut opp_rank = [clean_rules::HandRank(0); 6];
        let mut n_opp_cont = 0usize;
        let mut max_opp = clean_rules::HandRank(0);
        let mut any_opp = false;
        for j in 0..live - 1 {
            opp_cont[j] = flop_continue(opp[j], flop, policy.use_draws);
            opp_rank[j] = rank7(opp[j], board);
            if opp_cont[j] {
                n_opp_cont += 1;
                if !any_opp || opp_rank[j] > max_opp {
                    max_opp = opp_rank[j];
                    any_opp = true;
                }
            }
        }
        for (ci, &(a, b)) in layout.iter().enumerate() {
            let cm = (1u64 << a) | (1u64 << b);
            if dealt & cm != 0 {
                continue; // my combo conflicts with this sample's cards
            }
            if !flop_continue([a, b], flop, policy.use_draws) {
                sum[ci] += (-half_pot) as f64;
                cnt[ci] += 1;
                continue;
            }
            let n_cont = n_opp_cont + 1;
            if n_cont == 1 {
                let rake = (pot * rake_rate).min(rake_cap).max(0.0);
                sum[ci] += ((pot - rake) - half_pot) as f64;
                cnt[ci] += 1;
                continue;
            }
            let my_rank = rank7([a, b], board);
            let maxr = if any_opp { my_rank.max(max_opp) } else { my_rank };
            let me_wins = my_rank == maxr;
            let mut winners = if me_wins { 1u32 } else { 0 };
            for j in 0..live - 1 {
                if opp_cont[j] && opp_rank[j] == maxr {
                    winners += 1;
                }
            }
            let contested = pot + n_cont as f32 * bet;
            let rake = (contested * rake_rate).min(rake_cap).max(0.0);
            let net = contested - rake;
            let my_invest = half_pot + bet;
            let ev = if me_wins { net / winners as f32 - my_invest } else { -my_invest };
            sum[ci] += ev as f64;
            cnt[ci] += 1;
        }
                (sum, cnt)
            },
        )
        .reduce(
            || (vec![0.0f64; n], vec![0u32; n]),
            |(mut sa, mut ca), (sb, cb)| {
                for i in 0..n {
                    sa[i] += sb[i];
                    ca[i] += cb[i];
                }
                (sa, ca)
            },
        );
    (0..n).map(|i| if cnt[i] > 0 { (sum[i] / cnt[i] as f64) as f32 } else { -half_pot }).collect()
}

/// All-in / showdown equity (share of pot won, no folding) — the DUMB baseline
/// the realization model must beat, for the EQR comparison.
pub fn allin_equity_on_flop(
    my_hole: [u8; 2],
    flop: [u8; 3],
    live: usize,
    samples: usize,
    seed: u64,
) -> f32 {
    let mut s = seed | 1;
    let base_used: u64 = (1u64 << my_hole[0])
        | (1u64 << my_hole[1])
        | (1u64 << flop[0])
        | (1u64 << flop[1])
        | (1u64 << flop[2]);
    let mut acc = 0.0f64;
    for _ in 0..samples {
        let mut used = base_used;
        let mut opp = Vec::with_capacity(live - 1);
        for _ in 0..live - 1 {
            let a = draw_card(&mut s, &mut used);
            let b = draw_card(&mut s, &mut used);
            opp.push([a, b]);
        }
        let turn = draw_card(&mut s, &mut used);
        let river = draw_card(&mut s, &mut used);
        let board = [flop[0], flop[1], flop[2], turn, river];
        let my_rank = rank7(my_hole, board);
        let mut maxr = my_rank;
        for o in &opp {
            let r = rank7(*o, board);
            if r > maxr {
                maxr = r;
            }
        }
        let mut winners = if my_rank == maxr { 1u32 } else { 0 };
        for o in &opp {
            if rank7(*o, board) == maxr {
                winners += 1;
            }
        }
        if my_rank == maxr {
            acc += 1.0 / winners as f64;
        }
    }
    (acc / samples as f64) as f32
}

/// Two concrete cards for a 169-class probe (suits chosen non-conflicting).
/// kind: 0=pair, 1=suited, 2=offsuit. ranks r0 >= r1 (0=2 .. 12=A).
pub fn class_combo(r0: u8, r1: u8, suited: bool) -> [u8; 2] {
    if r0 == r1 {
        [r0 * 4, r0 * 4 + 1] // pair: two suits
    } else if suited {
        [r0 * 4, r1 * 4] // same suit (0)
    } else {
        [r0 * 4, r1 * 4 + 1] // different suits
    }
}
