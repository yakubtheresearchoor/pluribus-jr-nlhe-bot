//! ROLLOUT CFV PRE-BANK (2026-06-15): bank the equity-rollout flop-root CFVs
//! the production fill never solves — LIVE-6 (full table; census is live 2..=5)
//! and ALL-IN live≥3 (behind ≤ 0). Both are check-down-to-showdown, so their
//! CFV is reach- and strategy-independent: compute once per (bin × flop), in
//! parallel, and store under <bp>/cfv/L{live}_S{bin}/, so the preflop reads
//! them (fast) instead of solving them single-threaded inside the oracle.
//! (live-2, incl. all-in, stays exact-on-the-fly — it's cheap.)
//!
//! Env: PF_BLUEPRINT (default blueprint_out_v1), RB_THREADS (14).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use play_harness::preflop_oracle::{cfv_rollout_live3plus, rollout_seam_tree, write_prebanked};
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::postflop_oracle::SeamCell;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BoardState};
use solver_core::tree::flat::{FlatTree, NODE_TYPE_CHANCE};

/// SAME unified tree as `cfv_prebank` / `preflop_runner` / the connected
/// blueprint — `build_conn_preflop_tree(6, 5)`. Must match so the live-6 /
/// all-in rollout bins discovered here are the ones the runner queries.
fn cap3_preflop_tree() -> FlatTree {
    solver_core::blueprint::build_conn_preflop_tree(6, 5).0
}

fn main() {
    let spec = production_game_v1();
    let root = std::env::var("PF_BLUEPRINT").unwrap_or_else(|_| "blueprint_out_v1".into());
    let cfv_root = format!("{root}/cfv");
    let threads: usize = std::env::var("RB_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(14);

    // Discover the rollout bins routed to cfv_rollout_live3plus:
    //   live==6 (any SPR)  ∪  all-in (behind ≤ 0) with live ≥ 3.
    let pft = cap3_preflop_tree();
    let mut buckets: HashMap<(u8, i64), SeamCell> = HashMap::new();
    for idx in 0..pft.num_nodes() {
        let n = &pft.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.board_state != BoardState::Flop as u8 { continue; }
        let cell = SeamCell::at_chance_node(&pft, idx, 6);
        let allin = spec.stack - cell.commit <= 0;
        let rollout = cell.live == 6 || (allin && cell.live >= 3);
        if !rollout { continue; }
        buckets.entry(cell.bucket_key(spec.stack)).or_insert(cell);
    }
    let mut blist: Vec<((u8, i64), SeamCell)> = buckets.into_iter().collect();
    blist.sort_by_key(|(k, _)| (k.0, k.1));

    let canon = PreflopChanceTable::new(6, vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6])
        .canonical_flops.clone();
    let nflop = canon.len();
    let nb_for = |live: u8| -> usize { match live { 3 | 4 => 15, _ => 8 } };

    eprintln!("rollout pre-bank: {} bins × {nflop} flops, {threads} threads → {cfv_root}", blist.len());
    for (k, c) in &blist {
        eprintln!("  L{} S{} (commit={} pot={}{})", k.0, k.1, c.commit, c.pot,
            if spec.stack - c.commit <= 0 { " ALLIN" } else { "" });
    }

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let total = blist.len();
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads.min(total.max(1)) {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= total { break; }
                let (key, cell) = &blist[k];
                let tree = rollout_seam_tree(&spec, cell.live, cell.commit, cell.pot);
                let nb = nb_for(cell.live);
                for (fi, &canonical) in canon.iter().enumerate() {
                    let per_live = cfv_rollout_live3plus(canonical, fi, &tree, cell.live, nb);
                    write_prebanked(&cfv_root, *key, fi, &per_live);
                }
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                let el = t0.elapsed().as_secs_f64();
                eprintln!("[{d}/{total}] L{} S{} done | {:.1} min | ETA {:.1} min",
                    key.0, key.1, el / 60.0, el / d as f64 * (total - d) as f64 / 60.0);
            });
        }
    });
    eprintln!("ROLLOUT_PREBANK_COMPLETE: {total} bins in {:.1} min", t0.elapsed().as_secs_f64() / 60.0);
}
