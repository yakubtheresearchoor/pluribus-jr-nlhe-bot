//! V1 POSTFLOP BILL (2026-06-12): the end-to-end postflop fill cost
//! under the VALIDATED cell-set policy (log2-SPR width 0.25,
//! bin-center representatives, shape-split, all-in cells free) and the
//! VALIDATED per-family prices. This answers the question the cost arc
//! was chasing: does v1 need the GPU, or did the depth cap (preflop
//! memory) + SPR bucketing (cell count ÷80) make CPU fit — making the
//! GPU the v2 quality lever it originally was?
//!
//! Pricing model: per-iter cost scales LINEARLY IN NODES within a
//! family (measured support: GPU live-6 pair 45,711→10.25 s/iter vs
//! 58,479→12.88 — nodes ×1.28, time ×1.26). Anchors are the VALIDATED
//! at-size prices (all four families passed GPU↔CPU parity at
//! nh=1176):
//!   live-2: CPU exact   0.047 s/iter @ 85 nodes  (no GPU arm needed)
//!   live-3: CPU 0.146 / GPU 0.022   @ 501
//!   live-4: CPU 2.524 / GPU 0.099   @ 2,443
//!   live-5: CPU 102.5 / GPU 1.006   @ 14,753
//!   live-6: CPU 2537  / GPU 10.25   @ 45,711
//! Bill = Σ over representatives of s/iter × 34 iters × 1755 flops.
//! CPU parallelizes across flops (14-thread effective ≈ ÷10, the
//! measured convention from the B=8 production-run pricing); GPU is
//! single-stream (wall=busy: concurrency buys nothing).

use solver_core::tree::action::{
    production_game_v1, BetCap, BetSize, BetSizeOptions, BoardState,
};
use solver_core::tree::builder::{build_tree, build_tree_preflop_only};
use solver_core::tree::flat::{NODE_TYPE_CHANCE, MAX_NA_PREFLOP};
use std::collections::BTreeMap;

fn preflop_bets() -> BetSizeOptions {
    let max_raise_count = MAX_NA_PREFLOP.saturating_sub(2);
    BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..max_raise_count)
            .map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64))
            .collect(),
    }
}

/// (CPU s/iter, GPU s/iter) anchors at (nodes) per family.
fn family_anchor(live: u8) -> (f64, f64, f64) {
    match live {
        2 => (85.0, 0.047, 0.047), // exact CPU; "GPU" column = CPU (not worth a GPU pass)
        3 => (501.0, 0.146, 0.022),
        4 => (2443.0, 2.524, 0.099),
        5 => (14753.0, 102.5, 1.006),
        6 => (45711.0, 2537.0, 10.25),
        _ => unreachable!(),
    }
}

#[test]
#[ignore = "costing arithmetic; run with --ignored --nocapture --release"]
fn v1_postflop_bill() {
    let spec = production_game_v1();
    let mut cfg = spec.preflop_tree_config(preflop_bets());
    cfg.max_bets_per_street = BetCap::all(3);
    let tree = build_tree_preflop_only(&cfg).expect("v1 preflop tree (cap 3)");
    let np = spec.num_players as usize;

    // Cells with multiplicity.
    let mut cells: BTreeMap<(u8, i32, i32), usize> = BTreeMap::new();
    for idx in 0..tree.nodes.len() {
        let n = &tree.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.board_state != BoardState::Flop as u8 {
            continue;
        }
        let mask = tree.get_folded_mask(idx);
        let contribs: Vec<i32> =
            (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let live: Vec<usize> = (0..np).filter(|&p| mask & (1 << p) == 0).collect();
        let pot: i32 = contribs.iter().sum();
        *cells.entry((live.len() as u8, contribs[live[0]], pot)).or_default() += 1;
    }

    // Bucket: (live, log2-SPR bin at width 0.25); all-in cells free.
    // Within bucket: shape-split by tree node count; representative =
    // bin-center cell (median SPR member of the shape group).
    let flop_bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let mut groups: BTreeMap<(u8, i64, usize), Vec<(f64, usize)>> = BTreeMap::new(); // (live, bin, nodes) -> [(spr, mult)]
    let mut allin_entries = 0usize;
    let mut total_entries = 0usize;
    for (&(live, commit, pot), &mult) in &cells {
        total_entries += mult;
        let behind = spec.stack - commit;
        if behind <= 0 {
            allin_entries += mult;
            continue;
        }
        let spr = behind as f64 / pot as f64;
        let bin = (spr.log2() / 0.25).floor() as i64;
        let ft = build_tree(&spec.flop_seam_config(live, commit, pot, flop_bets.clone()))
            .expect("flop tree");
        groups.entry((live, bin, ft.nodes.len())).or_default().push((spr, mult));
    }

    eprintln!(
        "cells {} | flop entries {} ({:.1}% all-in = free) | solve units {}",
        cells.len(),
        total_entries,
        100.0 * allin_entries as f64 / total_entries as f64,
        groups.len()
    );

    // Price each representative: family anchor × nodes ratio.
    const ITERS: f64 = 34.0;
    const FLOPS: f64 = 1755.0;
    let mut by_live: BTreeMap<u8, (usize, f64, f64, usize)> = BTreeMap::new(); // live -> (units, cpu_h, gpu_h, entries)
    for (&(live, _bin, nodes), members) in &groups {
        let (anchor_nodes, cpu_anchor, gpu_anchor) = family_anchor(live);
        let scale = nodes as f64 / anchor_nodes;
        let cpu_h = cpu_anchor * scale * ITERS * FLOPS / 3600.0;
        let gpu_h = gpu_anchor * scale * ITERS * FLOPS / 3600.0;
        let entries: usize = members.iter().map(|&(_, m)| m).sum();
        let e = by_live.entry(live).or_insert((0, 0.0, 0.0, 0));
        e.0 += 1;
        e.1 += cpu_h;
        e.2 += gpu_h;
        e.3 += entries;
    }

    let (mut units, mut cpu_total, mut gpu_total) = (0usize, 0.0f64, 0.0f64);
    eprintln!("family | units | entries | CPU 1-core h | GPU h");
    for (&live, &(u, cpu, gpu, entries)) in &by_live {
        eprintln!(
            "live {live} | {u:>4} | {entries:>7} | {cpu:>12.1} | {gpu:>8.1}"
        );
        units += u;
        cpu_total += cpu;
        gpu_total += gpu;
    }
    eprintln!("─────────────────────────────────────────────");
    eprintln!(
        "TOTAL  | {units:>4} |         | {cpu_total:>12.1} | {gpu_total:>8.1}"
    );
    eprintln!(
        "CPU 14-thread effective (÷10, the measured production convention): {:.1}h",
        cpu_total / 10.0
    );
    eprintln!(
        "VERDICT INPUTS: 24h budget — CPU-14t {} | GPU single-stream {}",
        if cpu_total / 10.0 <= 24.0 { "FITS" } else { "DOES NOT FIT" },
        if gpu_total <= 24.0 { "FITS" } else { "DOES NOT FIT" },
    );
}
