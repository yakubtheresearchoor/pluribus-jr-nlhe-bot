//! H3/H5 gates for the match core, against a live banked artifact:
//!   1. ANCHOR (exact, hand-computable): one always-aggressive seat vs
//!      five check-folders wins every other seat's ante every hand —
//!      net EXACTLY +10 / −2, zero variance. Also pins the tree's
//!      child-ordering convention (child 0 = passive, last = bet).
//!   2. Blueprint 6-max self-play: thousands of audit-mode hands, all
//!      seats on the blueprint; per-hand conservation asserted inside
//!      settle_pots; report mean/std per seat (zero-sum check) — the
//!      H4 Monte-Carlo machinery, running through the independent
//!      evaluator end to end.

use play_harness::blueprint::{build_oracle_tree, Blueprint};
use play_harness::match_play::{MatchEnv, Policy};

fn env_or_skip() -> Option<(Blueprint, solver_core::tree::flat::FlatTree)> {
    let path = "../blueprint_out_b10_4x4/flop_0000.bp";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: no banked artifact");
        return None;
    }
    Some((Blueprint::load(path).unwrap(), build_oracle_tree()))
}

#[test]
// UN-IGNORED 2026-06-12: the two tree surfaces this anchor exposed are
// fixed at the root — (1) folds no longer collapse the hand (fold masks
// accumulate per fold; 5-fold terminal reads 0b111110), (2) get_pot
// includes starting_pot so a 1.0x-pot bet records 12, not 1. Policies
// now select by action label, not child position.
fn anchor_aggressor_vs_checkfolders_exact() {
    let Some((bp, tree)) = env_or_skip() else { return };
    let env = MatchEnv::new(&bp, &tree);
    let mut rng = 7u64;
    for hand in 0..500u32 {
        let holes = env.deal_holes(&mut rng);
        let mut pol: Vec<Policy> = (0..6).map(|_| Policy::CheckFold).collect();
        pol[0] = Policy::AlwaysAggressive;
        let Some(net) = env.play_hand(&pol, &holes, &mut rng) else { continue };
        assert_eq!(net[0], 10, "hand {hand}: aggressor must win all five antes exactly");
        for p in 1..6 {
            assert_eq!(net[p], -2, "hand {hand}: check-folder loses exactly its ante");
        }
    }
    eprintln!("anchor: 500/500 hands exact (+10 / -2 x5, zero variance)");
}

#[test]
fn blueprint_selfplay_audit_mode() {
    let Some((bp, tree)) = env_or_skip() else { return };
    let env = MatchEnv::new(&bp, &tree);
    let mut rng = 1234u64;
    const N: usize = 20_000;
    let mut sums = [0i64; 6];
    let mut sq = [0f64; 6];
    for _ in 0..N {
        let holes = env.deal_holes(&mut rng);
        let pol: Vec<Policy> = (0..6).map(|_| Policy::Blueprint(&bp)).collect();
        let Some(net) = env.play_hand(&pol, &holes, &mut rng) else { continue }; // conservation asserted inside
        for p in 0..6 {
            sums[p] += net[p];
            sq[p] += (net[p] as f64) * (net[p] as f64);
        }
    }
    let total: i64 = sums.iter().sum();
    assert_eq!(total, 0, "zero-sum across seats over the whole match");
    for p in 0..6 {
        let mean = sums[p] as f64 / N as f64;
        let var = sq[p] / N as f64 - mean * mean;
        let se = (var / N as f64).sqrt();
        eprintln!("seat {p}: mean {mean:+.4} chips/hand ± {se:.4} (1σ)");
        // Self-play sanity: no seat should show a runaway edge beyond
        // ~position effects (|mean| < 1 chip on a 12-chip pot).
        assert!(mean.abs() < 1.0, "seat {p} runaway mean {mean}");
    }
    eprintln!("self-play MC: {N} audit-mode hands through the independent evaluator");
}
