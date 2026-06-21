//! Validation gate for the preflop state machine (piece 3a): the flop-entry
//! distribution must be SANE — most pots heads-up/3-way, occasional multiway
//! limped pots, pots in a reasonable chip range, and the resolved SeamCells
//! should land in live-counts the v1 fill covers (2..6).
//!
//! Run: PF_STRAT=$PWD/preflop_eqr_bbfix cargo test --release -p play-harness \
//!      --test full_hand_preflop_gate -- --ignored --nocapture

use play_harness::full_hand::{FlopRouter, FullHandSim, RouteKind, Seat};
use play_harness::pool_preflop::PoolPreflop;
use play_harness::preflop_player::PreflopPlayer;
use solver_core::card::NUM_POSSIBLE_HANDS;
use solver_core::card::index_to_card_pair;

fn deal(rng: &mut u64) -> [[u8; 2]; 6] {
    // 12 distinct cards → 6 holes.
    let mut used = 0u64;
    let mut holes = [[0u8; 2]; 6];
    let mut pick = || -> u8 {
        loop {
            let c = (play_harness::preflop_player::splitmix64(rng) % 52) as u8;
            if used & (1 << c) == 0 {
                used |= 1 << c;
                return c;
            }
        }
    };
    for p in 0..6 {
        holes[p] = [pick(), pick()];
    }
    holes
}

#[test]
#[ignore = "trace; FH_TRACE=1 --ignored --nocapture --release"]
fn trace_few_hands() {
    let base = std::env::var("PF_STRAT").unwrap_or_else(|_| "preflop_eqr_bbfix".into());
    if !std::path::Path::new(&format!("{base}.f32")).exists() { return; }
    let pf = PreflopPlayer::load(&base).unwrap();
    let sim = FullHandSim::new(pf, PoolPreflop::new(), 200, 1, 2);
    let seats = [Seat::Bot, Seat::Pool, Seat::Pool, Seat::Pool, Seat::Pool, Seat::Pool];
    let mut rng = 0x9999_u64;
    for h in 0..8 {
        let holes = deal(&mut rng);
        eprintln!("--- hand {h} ---");
        let fe = sim.play_preflop(&seats, &holes, &mut rng);
        eprintln!("  => live={} pot={} commit_max={} folded={:?}", fe.live, fe.pot, fe.cell.commit, fe.folded);
    }
}

#[test]
#[ignore = "needs preflop artifact + blueprint_out_v1/cells.txt; --ignored --nocapture --release"]
fn flop_routing_coverage() {
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
    let seats = [Seat::Bot, Seat::Pool, Seat::Pool, Seat::Pool, Seat::Pool, Seat::Pool];

    let n = 30_000;
    let mut rng = 0x55_u64;
    // per live-count: [exact, fallback, uncovered]
    let mut cov: std::collections::HashMap<u8, [u64; 3]> = std::collections::HashMap::new();
    for _ in 0..n {
        let holes = deal(&mut rng);
        let fe = sim.play_preflop(&seats, &holes, &mut rng);
        if fe.live < 2 {
            continue;
        }
        let (_, kind) = router.route(&fe.cell);
        let e = cov.entry(fe.live).or_default();
        match kind {
            RouteKind::Exact => e[0] += 1,
            RouteKind::Fallback => e[1] += 1,
            RouteKind::Uncovered => e[2] += 1,
        }
    }

    eprintln!("\n=== flop routing coverage (live-3/4/5 = solved cells) ===");
    let mut lives: Vec<u8> = cov.keys().copied().collect();
    lives.sort();
    for l in lives {
        let [ex, fb, un] = cov[&l];
        let tot = ex + fb + un;
        eprintln!(
            "  live-{l}: {tot:>6} pots | exact {:5.1}% | fallback {:5.1}% | uncovered {:5.1}%",
            100.0 * ex as f64 / tot as f64,
            100.0 * fb as f64 / tot as f64,
            100.0 * un as f64 / tot as f64
        );
    }
    eprintln!("(live-2 .bp2 + live-6 rollout are separate paths, shown as 'uncovered' here)");
    // live-3/4/5 should mostly route (exact+fallback), not be uncovered.
    for l in [3u8, 4, 5] {
        if let Some(&[ex, fb, un]) = cov.get(&l) {
            let routed = ex + fb;
            assert!(routed > un, "live-{l}: more uncovered than routed ({un} vs {routed})");
        }
    }
    eprintln!("✓ live-3/4/5 pots route to v1 cells (exact or nearest-bin fallback)");
}

#[test]
#[ignore = "needs preflop artifact; --ignored --nocapture --release"]
fn preflop_flop_entry_distribution_sane() {
    let base = std::env::var("PF_STRAT").unwrap_or_else(|_| "preflop_eqr_bbfix".into());
    if !std::path::Path::new(&format!("{base}.f32")).exists() {
        eprintln!("SKIP: no preflop artifact at {base}.f32");
        return;
    }
    let _ = (NUM_POSSIBLE_HANDS, index_to_card_pair as fn(usize) -> (u8, u8));
    let pf = PreflopPlayer::load(&base).unwrap();
    let pool = PoolPreflop::new();
    let stack = 200;
    let sim = FullHandSim::new(pf, pool, stack, 1, 2);

    let n = 20_000;
    let mut rng = 0x1234_u64;
    let mut live_hist = [0u64; 7]; // index by live count 0..6
    let mut commit_sum = 0i64;
    let mut pot_sum = 0i64;
    let mut bot_folded = 0u64;
    let mut both_blinds_only = 0u64; // walk (everyone folds to BB)
    // Bot at UTG seat (position 0); rotate isn't needed for distribution sanity.
    let seats = [Seat::Bot, Seat::Pool, Seat::Pool, Seat::Pool, Seat::Pool, Seat::Pool];

    for _ in 0..n {
        let holes = deal(&mut rng);
        let fe = sim.play_preflop(&seats, &holes, &mut rng);
        live_hist[fe.live as usize] += 1;
        pot_sum += fe.pot as i64;
        commit_sum += fe.cell.commit as i64;
        if fe.folded[0] {
            bot_folded += 1;
        }
        if fe.live <= 1 {
            both_blinds_only += 1;
        }
    }

    eprintln!("\n=== flop-entry distribution over {n} hands (bot=UTG) ===");
    for l in 0..=6 {
        let pct = 100.0 * live_hist[l] as f64 / n as f64;
        eprintln!("  live-{l}: {:6.2}%  ({} hands)", pct, live_hist[l]);
    }
    eprintln!("  avg pot   = {:.1} chips ({:.1} bb)", pot_sum as f64 / n as f64, pot_sum as f64 / n as f64 / 2.0);
    eprintln!("  avg commit= {:.1} chips", commit_sum as f64 / n as f64);
    eprintln!("  bot(UTG) folded preflop: {:.1}%", 100.0 * bot_folded as f64 / n as f64);
    eprintln!("  hand ended preflop (≤1 live): {:.1}%", 100.0 * both_blinds_only as f64 / n as f64);

    // Sanity: multiway (live≥3) should happen a meaningful fraction (loose pool
    // limps → multiway), but heads-up + 3-way should dominate. live≥2 the norm.
    let flops: u64 = (2..=6).map(|l| live_hist[l]).sum();
    let multiway: u64 = (3..=6).map(|l| live_hist[l]).sum();
    assert!(flops > n / 2, "too few hands reach a flop: {flops}/{n}");
    assert!(multiway > n / 20, "loose pool should make some multiway pots: {multiway}/{n}");
    assert!(live_hist[6] < n / 2, "implausibly many 6-way pots: {}", live_hist[6]);
    // UTG should fold a lot (tight EQR UTG range ~ opens 11%, but faces 3bets):
    assert!(bot_folded > n / 3, "UTG bot should fold often: {bot_folded}/{n}");
    eprintln!("✓ flop-entry distribution sane");
}
