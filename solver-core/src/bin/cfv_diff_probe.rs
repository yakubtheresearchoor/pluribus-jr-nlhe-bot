//! CFV DIFFERENTIATION PROBE (2026-06-17): the in-solver UTG emit shows the
//! chance-node CFV is ~FLAT across the traverser's own hand (AA cl0 ≈ 72o cl158,
//! differing <0.3%) and slightly INVERTED, with magnitude ~-1e11 scaling with
//! the traverser's bet size — the signature of −investment dominating while the
//! won-pot/equity signal is annihilated. This is the root cause of the loose
//! preflop: a preflop CFR fed near-identical values for AA and 72o cannot build
//! a range.
//!
//! This probe isolates WHERE the flatness enters. It solves the EXACT HU (live-2)
//! flop subgame and extracts the per-traverser-hand flop-root CFV, then prints
//! the value for AA / KK / AKs / 72o / 32o holdings. If the exact HU extraction
//! ALREADY fails to differentiate AA from 72o, the bug is in the postflop
//! solve/extraction; if it differentiates cleanly, the flatness is introduced
//! downstream in the multiway aggregation / opponent-reach normalization.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{card_from_str, Card};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::{extract_v_flop_root_from_cum, PreflopChanceTable};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

fn main() {
    let spec = production_game_v1();
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

    let tree = build_tree(&spec.flop_seam_config(2, 2, 12, bets)).unwrap();
    let t = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
    let game = FlopStartGame::new(t);
    let mut s = FlopStartVectorCfr::new(&tree, game.table());
    s.run(&tree, &game, iters);
    let (v, layout) = extract_v_flop_root_from_cum(&mut s, &tree, &game, 0);

    println!("EXACT HU flop-root CFV — canonical {:?}, {} iters, {} traverser hands\n", canonical, iters, v.len());
    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!("overall: min {mn:.4} | max {mx:.4} | mean {mean:.4} | spread {:.4}\n", mx - mn);

    // For a named holding, find the matching traverser hand in the layout and
    // average its CFV (a holding may map to multiple board-compatible combos).
    let look = |name: &str, a: &str, b: &str| {
        let (ca, cb) = (card_from_str(a).unwrap(), card_from_str(b).unwrap());
        let want = norm(ca, cb);
        let mut vals: Vec<f32> = vec![];
        for (h, &(x, y)) in layout.iter().enumerate() {
            if norm(x, y) == want {
                vals.push(v[h]);
            }
        }
        if vals.is_empty() {
            println!("  {name:<5}: (no board-compatible combo on this flop)");
        } else {
            let m = vals.iter().sum::<f32>() / vals.len() as f32;
            println!("  {name:<5}: CFV {m:>10.4}  ({} combos, range [{:.3}, {:.3}])",
                vals.iter().cloned().fold(f32::INFINITY, f32::min),
                m,
                vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }
    };
    println!("per-holding flop-root CFV (traverser=0), strong → weak:");
    look("AA", "Ac", "Ad");
    look("KK", "Kc", "Kd");
    look("AKs", "Ah", "Kh");
    look("QQ", "Qc", "Qd");
    look("JTs", "Jh", "Th");
    look("T9s", "Th", "9h");
    look("72o", "7h", "2d");
    look("32o", "3h", "2d");
    println!("\n(if AA ≫ 72o here, the exact HU extraction differentiates and the flatness is");
    println!(" downstream — multiway aggregation / opponent-reach normalization. If AA ≈ 72o,");
    println!(" the postflop solve/extraction itself is the bug.)");
}

// Order-independent suit-collapsed holding key: (hi_rank, lo_rank, suited?).
fn norm(a: Card, b: Card) -> (u8, u8, bool) {
    let (ra, rb) = (a / 4, b / 4);
    let suited = (a % 4) == (b % 4);
    let (hi, lo) = if ra >= rb { (ra, rb) } else { (rb, ra) };
    (hi, lo, suited)
}
