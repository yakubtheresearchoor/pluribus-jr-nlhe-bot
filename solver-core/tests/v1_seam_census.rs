//! V1 SEAM CENSUS (measurement instrument, 2026-06-12): enumerate the
//! production-game-v1 preflop tree's flop-entry chance nodes and bin
//! them into SEAM CELLS (live players, per-live commit, pot). Each
//! distinct cell is a flop-start game the postflop oracle must cover
//! (pot-bucket-keyed frozen-CFV design); the census tells us how many
//! cells exist, their multiplicities, and (for representative cells)
//! the flop-tree node counts that drive blueprint pricing.
//!
//! Also gates the v1 preflop config itself: builds, poker-rules clean
//! (the standing tree_correctness_gate covers legality on its own
//! configs; here we assert structural seam invariants — every flop
//! entry has equal live commits and pot = Σ contribs).

use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::{build_tree, build_tree_preflop_only};
use solver_core::tree::flat::{NODE_TYPE_CHANCE, MAX_NA_PREFLOP};
use std::collections::BTreeMap;

fn preflop_bets() -> BetSizeOptions {
    // The bootstrap's production preflop abstraction (verbatim shape):
    // one open size + a pot-relative raise ladder filling MAX_NA_PREFLOP.
    let max_raise_count = MAX_NA_PREFLOP.saturating_sub(2);
    BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..max_raise_count)
            .map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64))
            .collect(),
    }
}

#[test]
#[ignore = "census instrument (~minutes on the fold-continuation tree); run with --ignored --nocapture"]
fn v1_seam_census() {
    let spec = production_game_v1();
    let cfg = spec.preflop_tree_config(preflop_bets());
    let t0 = std::time::Instant::now();
    let tree = build_tree_preflop_only(&cfg).expect("v1 preflop tree");
    let nn = tree.nodes.len();
    eprintln!("v1 preflop tree: {} nodes (built in {:.1?})", nn, t0.elapsed());

    let np = spec.num_players as usize;
    // Seam cells: (live, commit, pot) → count of flop-entry chance nodes.
    let mut cells: BTreeMap<(u8, i32, i32), usize> = BTreeMap::new();
    let mut n_chance = 0usize;
    for idx in 0..nn {
        let n = &tree.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.board_state != BoardState::Flop as u8 {
            continue;
        }
        n_chance += 1;
        let mask = tree.get_folded_mask(idx);
        let contribs: Vec<i32> =
            (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let live: Vec<usize> = (0..np).filter(|&p| mask & (1 << p) == 0).collect();
        let pot: i32 = cfg.starting_pot + contribs.iter().sum::<i32>();
        // SEAM INVARIANT: all live players have equal commits at the
        // flop, each at least the dead-money max, unless all-in short.
        let c0 = contribs[live[0]];
        for &p in &live {
            assert!(
                contribs[p] == c0 || contribs[p] >= spec.stack,
                "node {idx}: unequal live commits {:?} (live {:?})",
                contribs,
                live
            );
        }
        *cells.entry((live.len() as u8, c0, pot)).or_default() += 1;
    }
    eprintln!("flop-entry chance nodes: {n_chance}");
    let mut by_live: BTreeMap<u8, usize> = BTreeMap::new();
    for (&(live, _, _), &count) in &cells {
        *by_live.entry(live).or_default() += count;
    }
    eprintln!("cells: {} distinct (live, commit, pot) | entries by live count:", cells.len());
    for (live, count) in &by_live {
        let ncells = cells.keys().filter(|k| k.0 == *live).count();
        eprintln!("  {live} players: {count} flop entries across {ncells} cells");
    }
    eprintln!("top 20 cells by multiplicity:");
    let mut ranked: Vec<_> = cells.iter().collect();
    ranked.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    for (&(live, commit, pot), &count) in ranked.iter().take(20) {
        eprintln!(
            "  live {live}  commit {commit:>3} ({:>5.1} bb)  pot {pot:>4} ({:>6.1} bb)  ×{count}",
            commit as f64 / 2.0,
            pot as f64 / 2.0
        );
    }

    // Flop-tree sizes (drives pricing): oracle-shape abstraction
    // (1.0x pot bet, no raises). Build every distinct cell, report
    // per-live aggregate node counts + the largest cells.
    let flop_bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let mut sizes: Vec<((u8, i32, i32), usize)> = Vec::new();
    for &(live, commit, pot) in cells.keys() {
        let fcfg = spec.flop_seam_config(live, commit, pot, flop_bets.clone());
        let ft = build_tree(&fcfg).expect("flop seam tree");
        sizes.push(((live, commit, pot), ft.nodes.len()));
    }
    eprintln!("flop-start tree sizes (oracle-shape abstraction), per live count:");
    for (&live, _) in &by_live {
        let s: Vec<usize> =
            sizes.iter().filter(|((l, _, _), _)| *l == live).map(|&(_, n)| n).collect();
        let (min, max) = (s.iter().min().unwrap(), s.iter().max().unwrap());
        let sum: usize = s.iter().sum();
        eprintln!(
            "  live {live}: {} cells, nodes min {min} / max {max} / mean {:.0}",
            s.len(),
            sum as f64 / s.len() as f64
        );
    }
    let mut top: Vec<_> = sizes.iter().collect();
    top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    eprintln!("largest flop trees:");
    for ((live, commit, pot), n) in top.iter().take(8) {
        eprintln!("  live {live} commit {commit} pot {pot}: {n} nodes");
    }
}
