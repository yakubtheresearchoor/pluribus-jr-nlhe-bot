//! BLUEPRINT RE-SOLVE COST PROBE: how does a per-flop multiway bucketed solve scale
//! with the number of sampled runouts (turns × rivers)? The 1×1 build only covers
//! one runout per flop; full coverage needs many. Measure the per-runout cost so the
//! full-coverage re-solve can be projected from the known 1×1 build (~20 h GPU).
//!
//! Run: cargo run --release -p play-harness --bin bp_resolve_cost_probe

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

fn sm64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Sample `nt` distinct turns and `nr` distinct rivers per turn from a flop's deck.
fn seeded_nxm(flop: [Card; 3], nt: usize, nr: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let bm = flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let mut x = sm64(12345);
    let mut pool = deck.clone();
    let mut turns = Vec::new();
    for _ in 0..nt.min(pool.len()) {
        x = sm64(x);
        turns.push(pool.remove((x % pool.len() as u64) as usize));
    }
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &t in &turns {
        let mut rpool: Vec<u8> = deck.iter().copied().filter(|&c| c != t).collect();
        let mut rivers = Vec::new();
        for _ in 0..nr.min(rpool.len()) {
            x = sm64(x);
            rivers.push(rpool.remove((x % rpool.len() as u64) as usize));
        }
        river_decks[t as usize] = rivers;
    }
    (turns, river_decks)
}

fn main() {
    let spec = production_game_v1();
    let live: u8 = std::env::var("LIVE").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let commit = 6i32;
    let pot = commit * live as i32 + 8; // must cover live commits + dead money
    let iters: u32 = 34;
    let nb = 15;
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let tree = build_tree(&spec.flop_seam_config(live, commit, pot, bets)).unwrap();
    let canon = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    )
    .canonical_flops
    .clone();
    let flop = canon[0];

    println!("live-{live} c{commit}/p{pot}, B={nb}, {iters} iters, flop {flop:?}, tree {} nodes", tree.num_nodes());
    println!("{:<8} {:>6} {:>10} {:>12}", "runouts", "n_run", "solve_s", "s/runout");
    let maxrun: usize = std::env::var("MAXRUN").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let mut base = 0f64;
    for (i, &(nt, nr)) in [(1usize, 1usize), (2, 2), (4, 4), (7, 7)].iter().enumerate() {
        if nt * nr > maxrun {
            continue;
        }
        let (turns, rd) = seeded_nxm(flop, nt, nr);
        let nrun: usize = turns.iter().map(|&t| rd[t as usize].len()).sum();
        let t0 = Instant::now();
        let table = FlopChanceTable::build_full_nh_sampled(flop, live, &turns, &rd);
        let bk = FlopBucketing::quantile(&table, nb);
        let game = FlopStartGame::new(table);
        let mut s = BucketedFlopCfr::new(&tree, game.table(), &bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        s.run(&tree, &game, &bk, iters);
        let dt = t0.elapsed().as_secs_f64();
        if i == 0 {
            base = dt;
        }
        println!("{:<8} {:>6} {:>10.2} {:>12.3}  ({:.1}× the 1×1)", format!("{nt}x{nr}"), nrun, dt, dt / nrun as f64, dt / base);
    }
}
