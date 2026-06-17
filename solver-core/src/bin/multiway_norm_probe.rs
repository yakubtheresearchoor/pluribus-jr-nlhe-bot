//! MULTIWAY NORMALIZATION PROBE (2026-06-17): the banked live-subset oracle
//! source (bucketed_live_subset_source) extracts live≥3 CFVs via
//! `run_all_root_cfv`, which — unlike its sibling `root_cfv_from_avg` — does NOT
//! call `normalize_opponent_reach`. Hypothesis: the banked multiway values are
//! therefore at unnormalized ×nh^num_opp scale, which dominates the preflop's
//! action values and swamps the per-traverser-hand equity signal (AA vs 72o) →
//! the loose, undifferentiated preflop the in-solver UTG emit revealed.
//!
//! This solves ONE live-3 flop subgame and extracts the traverser-0 flop-root
//! CFV BOTH ways: `run_all_root_cfv` (the bank path, unnormalized) and
//! `root_cfv_from_avg` (the normalized path). It prints magnitude + the AA / KK
//! / QQ / 72o / 32o values for each. If the bank path is huge + flat while the
//! normalized path is chip-scale + differentiated, the root cause is confirmed
//! and the fix is to normalize the banked extraction.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{card_from_str, Card};
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

fn main() {
    let spec = production_game_v1();
    let live: u8 = std::env::var("PROBE_LIVE").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let nb: usize = std::env::var("PROBE_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let iters: u32 = std::env::var("PROBE_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(64);

    let ptable = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let canonical = ptable.canonical_flops[0];
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let turns = vec![deck[0]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[deck[0] as usize] = vec![deck[1]];
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };

    let tree = build_tree(&spec.flop_seam_config(live, 2, 12, bets)).unwrap();
    let table = FlopChanceTable::build_full_nh_sampled(canonical, live, &turns, &river_decks);
    let hand_cards = table.hand_cards.clone();
    let nh = table.num_valid;
    let bk = FlopBucketing::quantile(&table, nb);
    let game = FlopStartGame::new(table);

    let mut s = BucketedFlopCfr::new(&tree, game.table(), &bk);
    s.set_terminal_design(TerminalDesign::Design1Collapsed);
    s.run(&tree, &game, &bk, iters);

    // Normalized path (what bank time SHOULD give the preflop).
    let norm = s.root_cfv_from_avg(&tree, &game, &bk);
    // Bank path (what the live-subset oracle source actually uses).
    let bank = s.run_all_root_cfv(&tree, &game, &bk, iters);

    println!("live-{live} @B{nb}, {iters} iters, canonical {:?}, nh={nh}\n", canonical);
    report("run_all_root_cfv  (BANK PATH, unnormalized)", &bank[0], &hand_cards, nh);
    println!();
    report("root_cfv_from_avg (NORMALIZED PATH)", &norm[0], &hand_cards, nh);
    println!("\n(if BANK is huge + AA≈72o while NORMALIZED is chip-scale + AA≫72o, the loose");
    println!(" preflop is confirmed: the banked oracle bakes unnormalized multiway CFVs.)");
}

fn report(label: &str, v: &[f32], hand_cards: &[Card], nh: usize) {
    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!("{label}");
    println!("  scale: min {mn:.3e} | max {mx:.3e} | mean {mean:.3e} | spread {:.3e}", mx - mn);
    let look = |name: &str, a: &str, b: &str| -> String {
        let want = norm_key(card_from_str(a).unwrap(), card_from_str(b).unwrap());
        let mut vals = vec![];
        for h in 0..nh {
            if norm_key(hand_cards[h * 2], hand_cards[h * 2 + 1]) == want {
                vals.push(v[h]);
            }
        }
        if vals.is_empty() {
            format!("{name} n/a")
        } else {
            format!("{name} {:.4e}", vals.iter().sum::<f32>() / vals.len() as f32)
        }
    };
    println!("  {} | {} | {} | {} | {}",
        look("AA", "Ac", "Ad"), look("KK", "Kc", "Kd"), look("QQ", "Qc", "Qd"),
        look("72o", "7h", "2d"), look("32o", "3h", "2d"));
}

fn norm_key(a: Card, b: Card) -> (u8, u8, bool) {
    let (ra, rb) = (a / 4, b / 4);
    let suited = (a % 4) == (b % 4);
    let (hi, lo) = if ra >= rb { (ra, rb) } else { (rb, ra) };
    (hi, lo, suited)
}
