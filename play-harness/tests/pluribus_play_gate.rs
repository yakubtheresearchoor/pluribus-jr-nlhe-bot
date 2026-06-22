//! End-to-end gate for the Pluribus per-street searched play loop.
//! Loads one real blueprint cell, deals multiway hands (bot + pool), and plays
//! them with `play_seam_pluribus` (the bot re-searches every street). Validates:
//!   (1) EXACT chip conservation with rake=0 → Σ net == dead, every hand;
//!   (2) a sane postflop bb/100 with the production rake.
//!
//! Run: BP_ROOT=$PWD/blueprint_out_v1 \
//!   cargo test --release -p play-harness --test pluribus_play_gate -- --ignored --nocapture

use play_harness::blueprint::Blueprint;
use play_harness::pluribus_play::{
    deal_from_range, flop_search_exploitability, play_seam_pair, play_seam_pluribus, SearchCfg,
};
use play_harness::preflop_player::splitmix64;

/// Parse a cell dir name "live3_c7_p25_b15" → (live, commit, pot).
fn parse_cell(name: &str) -> Option<(usize, i32, i32)> {
    let live = name.strip_prefix("live")?.split('_').next()?.parse().ok()?;
    let mut commit = None;
    let mut pot = None;
    for tok in name.split('_') {
        if let Some(c) = tok.strip_prefix('c') {
            commit = c.parse().ok();
        } else if let Some(p) = tok.strip_prefix('p') {
            pot = p.parse().ok();
        }
    }
    Some((live, commit?, pot?))
}

#[test]
#[ignore = "needs blueprint_out_v1; run --ignored --nocapture --release"]
fn pluribus_play_gate() {
    let bp_root = std::env::var("BP_ROOT").unwrap_or_else(|_| "blueprint_out_v1".into());
    // CELL env overrides; else pick the LOWEST-commit live-3 cell with a
    // flop_0000.bp (highest SPR ⇒ real bet sizing survives in the tree, vs a
    // low-SPR 3bet pot where pot bets merge to all-in).
    let cell_dir = if let Ok(c) = std::env::var("CELL") {
        Some(c)
    } else {
        let mut cands: Vec<String> = std::fs::read_dir(&bp_root)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("live3_"))
            .filter(|n| std::path::Path::new(&format!("{bp_root}/{n}/flop_0000.bp")).exists())
            .collect();
        // sort by commit (the cN token) ascending.
        cands.sort_by_key(|n| parse_cell(n).map(|(_, c, _)| c).unwrap_or(i32::MAX));
        cands.into_iter().next()
    };
    let cell_dir = match cell_dir {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no live3 cell with flop_0000.bp under {bp_root}");
            return;
        }
    };
    let (live, commit, pot) = parse_cell(&cell_dir).expect("parse cell");
    let bp_path = format!("{bp_root}/{cell_dir}/flop_0000.bp");
    let bp = Blueprint::load(&bp_path).expect("load bp");
    eprintln!("cell {cell_dir}: live={live} commit={commit} pot={pot} flop={:?}", bp.flop);
    eprintln!(
        "  bk nb_flop={} nb_turn={} nb_river={} nh={}",
        bp.bk.nb_flop, bp.bk.nb_turn, bp.bk.nb_river, bp.nh
    );
    assert_eq!(bp.np, live, "blueprint np matches cell live");

    let dead = (pot - live as i32 * commit).max(0) as u32;
    let fmask: u64 = bp.flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let iters: u32 = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let sample_m: u32 = std::env::var("SM").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let cfg = SearchCfg { iters, lambda: 300.0, opp_lambda: 300.0, sample_m, seed: 0xC0FFEE };

    // ---- (1) exact conservation, rake = 0 ----
    let mut rng = 0xBEEF_u64;
    let mut bad = 0u32;
    let n_cons = std::env::var("NC").ok().and_then(|s| s.parse().ok()).unwrap_or(8u64);
    for _ in 0..n_cons {
        let holes = deal_from_range(&bp, &mut rng);
        if let Some((net, _live)) =
            play_seam_pluribus(&bp, &holes, 0, commit as u32, dead, (0, 0), &cfg, &mut rng, false)
        {
            let s: i64 = net.iter().sum();
            if s != dead as i64 {
                bad += 1;
                if bad <= 3 {
                    eprintln!("  CONSERVATION FAIL: Σnet={s} != dead={dead}  net={net:?}");
                }
            }
        }
    }
    assert_eq!(bad, 0, "chip conservation violated in {bad}/{n_cons} hands");
    eprintln!("conservation OK: Σnet==dead for all {n_cons} hands (rake=0)");

    // ---- (1b) EXPLOITABILITY of the flop search (opponent-independent quality) ----
    // How far the per-street solve is from Nash within the subgame. Chip units;
    // also normalized by the pot for an absolute read (≪1% of pot = near-Nash).
    let exploit = flop_search_exploitability(&bp, commit, pot, &cfg);
    eprintln!(
        "flop-search exploitability = {exploit:.4} chips ({:+.2} bb, {:.2}% of pot {pot}) @ iters={}",
        exploit / 2.0,
        100.0 * exploit / pot as f32,
        cfg.iters
    );

    // ---- (2) postflop bb/100 vs the pool, sweeping the OPPONENT λ ----
    // Bot λ stays sharp (cfg.lambda); the bot MODELS opponents at opp_λ. Lower
    // opp_λ ⇒ the bot best-responds to softer/looser opponents ⇒ more
    // exploitative (the exploit lever vs the loose-passive pool). opp_λ == bot λ
    // is the GTO-ish baseline. CRN: every λ replays the SAME dealt hands (rng
    // reseeded per λ) so the comparison is paired, not confounded by variance.
    let rake_spec = (50u32, 20u32); // 5% capped at 10bb (20 units)
    let bb = 2.0;
    let n = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(8u64);
    let selfplay = std::env::var("SELF").is_ok();
    // SWEEP list (opponent λ); default = the single sharp λ=300 baseline (λ is not
    // the exploit lever — see commit 04de76d). Override with LAMSWEEP for a sweep.
    let sweep: Vec<f32> = match std::env::var("LAMSWEEP") {
        Ok(s) => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => vec![300.0],
    };
    eprintln!("=== opp-λ sweep (bot λ={:.0}, N={n}/λ, CRN) ===", cfg.lambda);
    for &olam in &sweep {
        let mut scfg = cfg;
        scfg.opp_lambda = olam;
        let mut rng = 0x1234_u64; // CRN: same hands across λ values
        let (mut sum, mut cnt, mut botwin) = (0i64, 0u64, 0u64);
        for _ in 0..n {
            let holes = deal_from_range(&bp, &mut rng);
            if let Some((net, _)) = play_seam_pluribus(
                &bp, &holes, 0, commit as u32, dead, rake_spec, &scfg, &mut rng, selfplay,
            ) {
                sum += net[0];
                cnt += 1;
                if net[0] > 0 {
                    botwin += 1;
                }
            }
        }
        let avg = sum as f64 / cnt.max(1) as f64;
        eprintln!(
            "  opp-λ={olam:>6.1}: hands={cnt} bot-avg={avg:+.2} ({:+.1} bb/100) win={:.1}%",
            avg / bb * 100.0,
            100.0 * botwin as f64 / cnt.max(1) as f64
        );
    }

    // ---- (3) PAIRED bots (seats 0,1) vs pool (seat 2): share card info or not ----
    // Proper CRN: pre-deal N (holes + a FIXED runout), and replay each under
    // share=off and share=on with a per-hand action seed. The ONLY difference
    // between the two conditions is the bots' INFORMATION, so the paired diff
    // isolates the value of shared hole-card range-blocking from variance.
    if std::env::var("PAIR").is_ok() && live >= 3 {
        let nh_pair = std::env::var("PAIR_N").ok().and_then(|s| s.parse().ok()).unwrap_or(60u64);
        // pre-deal holes + a valid runout per hand.
        let mut drng = 0x9A11_u64;
        let mut hands: Vec<(Vec<[u8; 2]>, (usize, usize))> = Vec::new();
        for _ in 0..nh_pair {
            let holes = deal_from_range(&bp, &mut drng);
            let blk = |c: u8| holes.iter().any(|h| h[0] == c || h[1] == c);
            let to: Vec<usize> = (0..bp.turns.len()).filter(|&t| !blk(bp.turns[t])).collect();
            if to.is_empty() {
                continue;
            }
            let t = to[(splitmix64(&mut drng) % to.len() as u64) as usize];
            let ro: Vec<usize> = (0..bp.rivers[t].len()).filter(|&r| !blk(bp.rivers[t][r])).collect();
            if ro.is_empty() {
                continue;
            }
            let r = ro[(splitmix64(&mut drng) % ro.len() as u64) as usize];
            hands.push((holes, (t, r)));
        }
        eprintln!("=== PAIRED bots (seats 0,1 vs pool 2), N={} dealt, CRN ===", hands.len());
        let mut avgs = [0.0f64; 2];
        for (si, &share) in [false, true].iter().enumerate() {
            let (mut pair_sum, mut cnt) = (0i64, 0u64);
            for (hi, (holes, ro)) in hands.iter().enumerate() {
                let mut prng = 0xD00D_u64 ^ (hi as u64).wrapping_mul(0x9E37_79B9);
                if let Some((net, _)) = play_seam_pair(
                    &bp, holes, 0, 1, share, commit as u32, dead, rake_spec, &cfg, &mut prng, Some(*ro),
                ) {
                    pair_sum += net[0] + net[1];
                    cnt += 1;
                }
            }
            let avg = pair_sum as f64 / cnt.max(1) as f64;
            avgs[si] = avg;
            eprintln!(
                "  share={share:5}: hands={cnt} pair-avg={avg:+.2} chips ({:+.1} bb/100 combined, 2 bots)",
                avg / bb * 100.0
            );
        }
        eprintln!(
            "  Δ(share−indep) = {:+.2} chips ({:+.1} bb/100) — value of shared card info",
            avgs[1] - avgs[0],
            (avgs[1] - avgs[0]) / bb * 100.0
        );
    }
}

/// Deal `live` holes (2 cards each) from the deck minus the flop.
fn deal(live: usize, fmask: u64, rng: &mut u64) -> Vec<[u8; 2]> {
    let mut used = fmask;
    let mut holes = vec![[0u8; 2]; live];
    for p in 0..live {
        for k in 0..2 {
            loop {
                let c = (splitmix64(rng) % 52) as u8;
                if used & (1 << c) == 0 {
                    used |= 1 << c;
                    holes[p][k] = c;
                    break;
                }
            }
        }
    }
    holes
}
