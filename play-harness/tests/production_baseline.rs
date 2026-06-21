//! FULL-TABLE production baseline (piece 5): bot (EQR preflop + v1 postflop)
//! vs the NL10 pool, full hands preflop->river, bb/100. FIRST end-to-end number
//! covers fold-out (live-1, NFND) + multiway (live-3/4/5 via play_seam). live-2
//! HU (.bp2) and live-6 (rollout) postflop are not yet wired, so bot-live-in-2/6
//! pots are SKIPPED (reported) — the number is conditional on those.
//!
//! Run: PF_STRAT=$PWD/preflop_eqr_bbfix BP_ROOT=$PWD/blueprint_out_v1 \
//!   cargo test --release -p play-harness --test production_baseline -- --ignored --nocapture

use play_harness::blueprint::Blueprint;
use play_harness::full_hand::{FlopRouter, FullHandSim, Seat};
use play_harness::match_play::{MatchEnv, Policy};
use play_harness::pool_preflop::PoolPreflop;
use play_harness::preflop_player::{splitmix64, PreflopPlayer};
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::collections::HashMap;

fn seam_tree(live: u8, commit: i32, pot: i32) -> FlatTree {
    let spec = production_game_v1();
    let cfg = spec.flop_seam_config(
        live,
        commit,
        pot,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
    );
    build_tree(&cfg).expect("seam tree")
}

#[test]
#[ignore = "needs preflop artifact + blueprint_out_v1; --ignored --nocapture --release"]
fn production_baseline_bb100() {
    let base = std::env::var("PF_STRAT").unwrap_or_else(|_| "preflop_eqr_bbfix".into());
    let bp_root = std::env::var("BP_ROOT").unwrap_or_else(|_| "blueprint_out_v1".into());
    let cells = format!("{bp_root}/cells.txt");
    if !std::path::Path::new(&format!("{base}.f32")).exists() || !std::path::Path::new(&cells).exists() {
        eprintln!("SKIP: missing {base}.f32 or {cells}");
        return;
    }
    let pf = PreflopPlayer::load(&base).unwrap();
    let sim = FullHandSim::new(pf, PoolPreflop::new(), 200, 1, 2);
    let router = FlopRouter::load(&bp_root, &cells, 200).unwrap();
    let rake_spec = (
        (sim.econ.rake_rate * 1000.0).round() as u32,
        sim.econ.rake_cap as u32,
    );

    // Canonical flops: the blueprint's universe. Use a small representative set
    // (the postflop loads one MatchEnv per (cell, flop)).
    let pt = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let k_flops = std::env::var("BP_FLOPS").ok().and_then(|s| s.parse().ok()).unwrap_or(6usize);
    let flops: Vec<(usize, [u8; 3])> =
        (0..k_flops).map(|fi| (fi, pt.canonical_flops[fi])).collect();

    // owned caches (MatchEnv borrows; rebuilt per hand from cached bp+tree)
    let mut bps: HashMap<(String, usize), Blueprint> = HashMap::new();
    let mut trees: HashMap<(u8, i32, i32), FlatTree> = HashMap::new();

    let n: u64 = std::env::var("BP_HANDS").ok().and_then(|s| s.parse().ok()).unwrap_or(60_000);
    let mut rng = 0xBA5E_u64;
    let mut sum_delta = 0.0f64;
    let mut counted = 0u64;
    let mut skipped = 0u64;
    let mut by_live = [0u64; 7];
    // localize: (sum, count) for folded / uncontested-win / postflop
    let mut cat = [[0.0f64, 0.0f64]; 3];
    let (mut pf_hands, mut pf_foldout, mut pf_botwin) = (0.0f64, 0.0f64, 0.0f64);

    for h in 0..n {
        let (fi, flop) = flops[(h as usize) % flops.len()];
        let fmask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
        // deal 12 distinct cards (deck minus flop) → 6 holes
        let mut used = fmask;
        let mut holes = [[0u8; 2]; 6];
        for p in 0..6 {
            for k in 0..2 {
                loop {
                    let c = (splitmix64(&mut rng) % 52) as u8;
                    if used & (1 << c) == 0 {
                        used |= 1 << c;
                        holes[p][k] = c;
                        break;
                    }
                }
            }
        }
        let bot_pos = (h % 6) as usize;
        let mut seats = [Seat::Pool; 6];
        seats[bot_pos] = Seat::Bot;

        let fe = sim.play_preflop(&seats, &holes, &mut rng);
        by_live[fe.live as usize] += 1;

        let mut catg = 0usize;
        let delta: Option<f64> = if fe.folded[bot_pos] {
            catg = 0;
            Some(-(fe.commit[bot_pos] as f64))
        } else if fe.live == 1 {
            // bot sole survivor (uncontested) — NFND, no rake
            catg = 1;
            Some((fe.pot - fe.commit[bot_pos]) as f64)
        } else if matches!(fe.live, 3 | 4 | 5) {
            catg = 2;
            let (route, _) = router.route(&fe.cell);
            let (commit, pot, dir) = route.expect("live-3/4/5 routes");
            let bp_path = format!("{dir}/flop_{fi:04}.bp");
            if !std::path::Path::new(&bp_path).exists() {
                skipped += 1;
                None
            } else {
                let live = fe.live;
                trees.entry((live, commit, pot)).or_insert_with(|| seam_tree(live, commit, pot));
                bps.entry((dir.clone(), fi)).or_insert_with(|| Blueprint::load(&bp_path).unwrap());
                let tree = &trees[&(live, commit, pot)];
                let bp = &bps[&(dir, fi)];
                let env = MatchEnv::new(bp, tree);
                // live seats in order → seam seats 0..live-1
                let live_seats: Vec<usize> = (0..6).filter(|&p| !fe.folded[p]).collect();
                let bot_seam = live_seats.iter().position(|&p| p == bot_pos).unwrap();
                let sh: Vec<[u8; 2]> = live_seats.iter().map(|&p| holes[p]).collect();
                let sp: Vec<Policy> = live_seats
                    .iter()
                    .map(|&p| if p == bot_pos { Policy::Blueprint(bp) } else { Policy::Population })
                    .collect();
                let dead = (pot - live as i32 * commit).max(0) as u32;
                match env.play_seam(&sp, &sh, commit as u32, dead, rake_spec, &mut rng, None) {
                    Some((net, term_live)) => {
                        pf_hands += 1.0;
                        if term_live == 1 { pf_foldout += 1.0; }
                        if net[bot_seam] > 0 { pf_botwin += 1.0; }
                        Some(net[bot_seam] as f64)
                    }
                    None => { skipped += 1; None }
                }
            }
        } else {
            // bot live in live-2 (HU) or live-6 — postflop not wired yet
            skipped += 1;
            None
        };

        if let Some(d) = delta {
            sum_delta += d;
            counted += 1;
            cat[catg][0] += d;
            cat[catg][1] += 1.0;
        }
    }
    eprintln!("postflop: hands={pf_hands:.0} foldout={:.1}% bot-win={:.1}%", 100.0*pf_foldout/pf_hands.max(1.0), 100.0*pf_botwin/pf_hands.max(1.0));
    let names = ["folded-pf", "uncontested", "postflop-3/4/5"];
    for c in 0..3 {
        let cnt = cat[c][1].max(1.0);
        eprintln!(
            "  [{}] n={:.0} avg={:+.2} chips ({:+.1} bb/100)",
            names[c], cat[c][1], cat[c][0] / cnt, (cat[c][0] / cnt) / 2.0 * 100.0
        );
    }

    let bb = 2.0;
    let bb100 = (sum_delta / counted as f64) / bb * 100.0;
    eprintln!("\n=== PRODUCTION BASELINE (full table, partial: live-1 + live-3/4/5) ===");
    eprintln!("hands={n} counted={counted} skipped(bot-live HU/6way)={skipped}");
    eprintln!("flops={} | rake={:?}", flops.len(), rake_spec);
    eprintln!("live-count seen: {:?}", &by_live[1..]);
    eprintln!("BOT bb/100 = {bb100:+.2}  (over counted hands; live-2 HU postflop excluded)");
    assert!(counted > n / 4, "too few counted hands");
}
