//! Variance-reduced A/B experiment driver (harness components 2 + 3):
//! mirrored-seat duplicate (cancels hole-card luck) + a stratified scorer
//! (mean ± stderr per live-count family, in mbb/100). The whole point of
//! the duplicate is that two strategies must be compared on the SAME cards,
//! or poker's enormous card variance buries any real edge.
//!
//! The reducer is itself an instrument, so it is anchored against ground
//! truth before it is trusted (harness_variance_gate): the variance-reduced
//! edge must MATCH the raw edge (a reducer that shifts the mean gives tight
//! bars around the wrong number — worse than none), just with tighter bars.

use crate::match_play::{splitmix64, MatchEnv, Policy};
use std::collections::BTreeMap;

/// Streaming mean / standard-error accumulator for one stratum.
#[derive(Default, Clone)]
pub struct Stat {
    pub n: u64,
    sum: f64,
    sumsq: f64,
}
impl Stat {
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        self.sumsq += x * x;
    }
    pub fn mean(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.sum / self.n as f64 }
    }
    /// Standard error of the mean.
    pub fn stderr(&self) -> f64 {
        if self.n < 2 { return f64::INFINITY; }
        let n = self.n as f64;
        let var = ((self.sumsq - self.sum * self.sum / n) / (n - 1.0)).max(0.0);
        (var / n).sqrt()
    }
}

/// chips/hand → mbb/100 (milli-big-blinds per 100 hands) given the big
/// blind in chips. Oracle ante game: bb defaults to 2 units (the ante).
pub fn mbb_per_100(chips_per_hand: f64, bb_chips: f64) -> f64 {
    chips_per_hand / bb_chips * 1000.0 * 100.0
}

/// Outcome of a mirrored A/B duplicate run.
pub struct AbResult {
    /// RAW estimator: each seating's A−B edge as an independent sample (no
    /// card-luck cancellation) — the unbiased reference the reducer is gated
    /// against.
    pub raw: Stat,
    /// REDUCED estimator: the mirrored paired difference (card luck cancels).
    pub reduced: Stat,
    /// Reduced edge stratified by live-count at showdown (the family key).
    pub by_live: BTreeMap<u8, Stat>,
}

/// Mirrored-seat duplicate A/B. Strategy `a` takes seats {0,1,2}, `b`
/// {3,4,5} (assign1); then swapped (assign2). Both seatings deal the SAME
/// six hole-hands (same deck), so across the two assignments each strategy
/// plays ALL six hands — hole-card luck cancels in the paired average.
///
/// RAW pushes each assignment's edge as a separate sample (so raw and
/// reduced consume the SAME play_hand calls — the variance difference is the
/// cancellation, not more data). REDUCED pushes the paired average, which is
/// negatively-correlated across the two assignments and therefore tighter.
pub fn ab_duplicate(
    env: &MatchEnv,
    a: &Policy,
    b: &Policy,
    n_decks: usize,
    rng: &mut u64,
) -> AbResult {
    let mut res = AbResult {
        raw: Stat::default(),
        reduced: Stat::default(),
        by_live: BTreeMap::new(),
    };
    let pols1 = [*a, *a, *a, *b, *b, *b];
    let pols2 = [*b, *b, *b, *a, *a, *a];
    for _ in 0..n_decks {
        let holes = env.deal_holes(rng);
        // COMMON board across the mirror (cancels runout variance too, not
        // just hole-card luck); independent action streams per assignment.
        let Some(runout) = env.draw_runout(&holes, rng) else { continue };
        let mut r1 = splitmix64(rng);
        let mut r2 = splitmix64(rng);
        let Some((o1, l1)) = env.play_hand_live(&pols1, &holes, &mut r1, Some(runout)) else { continue };
        let Some((o2, l2)) = env.play_hand_live(&pols2, &holes, &mut r2, Some(runout)) else { continue };
        // A−B edge in each seating (A at {0,1,2} in assign1, at {3,4,5} in assign2).
        let e1 = (o1[0] + o1[1] + o1[2] - o1[3] - o1[4] - o1[5]) as f64;
        let e2 = (o2[3] + o2[4] + o2[5] - o2[0] - o2[1] - o2[2]) as f64;
        res.raw.push(e1);
        res.raw.push(e2);
        res.reduced.push((e1 + e2) / 2.0); // mirrored paired average
        res.by_live.entry(l1).or_default().push(e1);
        res.by_live.entry(l2).or_default().push(e2);
    }
    res
}
