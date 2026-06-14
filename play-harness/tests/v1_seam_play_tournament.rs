//! STEP 3 DECIDER — bot-vs-bot PLAY tournament (the forgiving measure the whole
//! duplicate/reach-weight apparatus exists for). Exploitability is worst-case;
//! this measures mbb/100 EDGE between bucketed candidates head-to-head, with a
//! mirrored duplicate (same cards+board, seats swapped) cancelling card luck.
//!
//! Read AGAINST the exploitability sweep to get the WORST-CASE→PLAY TRANSLATION
//! FACTOR: exploitability says candidates differ by tens of % pot; if the PLAY
//! edge is a small fraction of that, the abstraction is fine in practice and the
//! multiway "crisis" was largely an exploitability-measure artifact. Equal
//! absolute bucket count across families (the de-confound lesson) so the
//! player-count axis stays clean. Bot-vs-bot FIRST (no field model needed); the
//! field model is built only if this says fidelity matters in play.

use play_harness::experiment::Stat;
use play_harness::v1_seam::{SeamBlueprint, SeamGame};
use play_harness::v1_seam::splitmix64;

fn flop() -> [u8; 3] { [47, 21, 0] }
fn mbb100(chips_per_seat_hand: f64) -> f64 { chips_per_seat_hand / 2.0 * 1000.0 * 100.0 }

/// Mirrored-duplicate per-seat-hand edge (chips) of A over B, head-to-head.
/// A in even seats / B in odd, then swapped; same dealt cards+board both ways
/// (audit deck shared), so card luck cancels in the paired average.
fn play_edge(game: &SeamGame, a: &SeamBlueprint, b: &SeamBlueprint, n_decks: usize, rng: &mut u64) -> Stat {
    let n = game.live as usize;
    let mut stat = Stat::default();
    for _ in 0..n_decks {
        let Some((holes, board)) = game.deal_audit(a, rng) else { continue };
        let mut r1 = splitmix64(rng);
        let mut r2 = splitmix64(rng);
        let bps1: Vec<&SeamBlueprint> = (0..n).map(|s| if s % 2 == 0 { a } else { b }).collect();
        let bps2: Vec<&SeamBlueprint> = (0..n).map(|s| if s % 2 == 0 { b } else { a }).collect();
        let (o1, _) = game.play_blueprints(&bps1, &holes, &board, &mut r1);
        let (o2, _) = game.play_blueprints(&bps2, &holes, &board, &mut r2);
        let (mut a_net, mut b_net) = (0i64, 0i64);
        for s in 0..n {
            if s % 2 == 0 { a_net += o1[s]; b_net += o2[s]; }
            else { a_net += o2[s]; b_net += o1[s]; }
        }
        stat.push((a_net - b_net) as f64 / n as f64); // per seat-hand
    }
    stat
}

#[test]
#[ignore = "round-robin non-transitivity check; --ignored --nocapture --release"]
fn play_round_robin_consistency() {
    // Is "fine loses to coarse" a clean ranking, or matchup-specific
    // non-transitivity? Complete the round-robin (exact/fine/coarse, all three
    // pairs) per family. Clean mis-rank ⇒ consistent fine<coarse everywhere +
    // transitive within family. Matchup-specific ⇒ sign flips across families
    // and/or a cycle (rock-paper-scissors) within one.
    const NH: usize = 10;
    let (nbc, nbf) = (5usize, 7);
    const D: usize = 30_000;
    eprintln!("\n═══ ROUND-ROBIN (mbb/100; +A>B): is the multiway head-to-head a clean ranking? ═══");
    eprintln!("family | exact>fine | exact>coarse | fine>coarse | within-family order");
    for live in 3u8..=5 {
        let g = SeamGame::new(live, 2, 12, flop());
        let exact = SeamBlueprint::solve_research(&g, NH, 700);
        let coarse = SeamBlueprint::solve_research_bucketed(&g, NH, nbc, 500);
        let fine = SeamBlueprint::solve_research_bucketed(&g, NH, nbf, 500);
        let mut rng = 0x9Au64 ^ live as u64;
        let ef = mbb100(play_edge(&g, &exact, &fine, D, &mut rng).mean());
        let ec = mbb100(play_edge(&g, &exact, &coarse, D, &mut rng).mean());
        let fc = mbb100(play_edge(&g, &fine, &coarse, D, &mut rng).mean());
        // Derive the order: who beats whom. exact should top; fine vs coarse is
        // the question; a cycle is exact>fine, fine>coarse, coarse>exact (etc).
        let order = if ef > 0.0 && ec > 0.0 {
            if fc > 0.0 { "exact>fine>coarse (transitive)" } else { "exact>coarse>fine (transitive, fine LAST)" }
        } else { "CYCLE (non-transitive!)" };
        eprintln!("  live-{live} | {ef:+8.0} | {ec:+8.0} | {fc:+8.0} | {order}");
    }
    eprintln!("\n→ fine<coarse sign FLIPS across families OR a cycle ⇒ head-to-head is matchup-specific,");
    eprintln!("  not a clean ranking ⇒ exploitability didn't 'mis-rank', rather NO cheap proxy ranks for");
    eprintln!("  deployment ⇒ the field is the only real measure (gated in for the clean reason).");
}

#[test]
fn play_edge_self_is_zero() {
    // Control: A vs A must be ~0 (the mirrored duplicate cancels position;
    // a non-zero self-edge means uncancelled bias and the tournament numbers
    // are untrustworthy). Cheap (one solve, live-3).
    let g = SeamGame::new(3, 2, 12, flop());
    let a = SeamBlueprint::solve_research_bucketed(&g, 10, 5, 300);
    let mut rng = 0x5E1Fu64;
    let s = play_edge(&g, &a, &a, 20_000, &mut rng);
    let (m, e) = (mbb100(s.mean()), mbb100(s.stderr()));
    eprintln!("self-play edge (A vs A): {m:+.1} ± {e:.1} mbb/100 ({} decks)", s.n);
    assert!(m.abs() < 4.0 * e.max(1.0), "self-edge {m:+.1} ≠ 0 (± {e:.1}) — duplicate not unbiased");
    eprintln!("PLAY-EDGE CONTROL PASS: A-vs-A ≈ 0 (machinery unbiased).");
}

#[test]
#[ignore = "play tournament (research-scale, minutes); --ignored --nocapture --release"]
fn multiway_play_tournament() {
    const NH: usize = 10;        // equal across families (de-confound)
    let (nb_coarse, nb_fine) = (5usize, 7); // equal absolute bucket counts
    const ITERS: u32 = 500;
    const DECKS: usize = 30_000;
    eprintln!("\n═══ BOT-vs-BOT PLAY TOURNAMENT (mbb/100, equal nh={NH} + equal buckets) ═══");
    eprintln!("read vs exploitability gap (worst-case): translation = play-edge / expl-gap");
    eprintln!("family | expl: coarse/fine (%pot) | PLAY edges (mbb/100): fine>coarse | exact>coarse");
    for live in 3u8..=5 {
        let g = SeamGame::new(live, 2, 12, flop());
        let exact = SeamBlueprint::solve_research(&g, NH, ITERS + 200);
        let coarse = SeamBlueprint::solve_research_bucketed(&g, NH, nb_coarse, ITERS);
        let fine = SeamBlueprint::solve_research_bucketed(&g, NH, nb_fine, ITERS);
        let (ec, ef) = (coarse.exploitability(&g, NH, 12), fine.exploitability(&g, NH, 12));
        let mut rng = 0x71u64 ^ live as u64;
        let fc = play_edge(&g, &fine, &coarse, DECKS, &mut rng);
        let xc = play_edge(&g, &exact, &coarse, DECKS, &mut rng);
        let (fc_m, fc_s) = (mbb100(fc.mean()), mbb100(fc.stderr()));
        let (xc_m, xc_s) = (mbb100(xc.mean()), mbb100(xc.stderr()));
        eprintln!("  live-{live} | {ec:6.1} / {ef:6.1} | fine>coarse {fc_m:+8.0} ±{fc_s:.0} | exact>coarse {xc_m:+8.0} ±{xc_s:.0}");
    }
    eprintln!("\n→ if PLAY edges are a SMALL fraction of the exploitability gap (in EV terms) ⇒ the");
    eprintln!("  worst-case gap translates WEAKLY to play ⇒ abstraction fine in practice, multiway");
    eprintln!("  'crisis' was an exploitability artifact ⇒ build the cheap candidate. If PLAY edges");
    eprintln!("  are LARGE ⇒ fidelity matters in play ⇒ build the field model to find the absolute bar.");
}
