//! LIVE-2 STRATEGY FILL (2026-06-15): bank the full-nh heads-up postflop
//! strategy for every (live-2 SPR bucket × flop), so live-2 spots play from a
//! banked policy instead of a real-time resolve (measured unviable at
//! ~156–780 s/decision). Live-2 can't use the bucketed `.bp` pipeline (HU
//! showdown is exact, np≥3 only there), so it's solved with the exact HU
//! solver and serialized via `live2_bank`.
//!
//! Buckets are auto-discovered from the cap-3 preflop tree's flop-entry chance
//! nodes (the same set the CFV pre-bank uses), one rep per (live=2) SPR bin;
//! all-in bins (behind ≤ 0) are skipped — those are equity rollouts, solved on
//! the fly by the preflop oracle, not banked. Output:
//!   <out>/live2/S{bin}/flop_NNNN.bp2   +   <out>/live2/manifest.txt
//!
//! Env: PF_BLUEPRINT (default blueprint_out_v1) = output root (writes under
//! its live2/ subdir), L2_THREADS (14), L2_CFV (1 ⇒ also (re)bank the live-2
//! CFV inline so future blueprints skip the separate pre-bank; 0 ⇒ strategy
//! only, for when CFV is already banked).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use play_harness::live2_bank::{live2_bet_menu, save_live2_v2, solve_live2};
use play_harness::preflop_oracle::{cfv_live2, write_prebanked};
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::postflop_oracle::SeamCell;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::{build_tree, build_tree_preflop_only};
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP, NODE_TYPE_CHANCE};

fn cap3_preflop_tree() -> FlatTree {
    let spec = production_game_v1();
    let mrc = MAX_NA_PREFLOP.saturating_sub(2);
    let mut cfg = spec.preflop_tree_config(BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect(),
    });
    cfg.max_bets_per_street = BetCap::all(3);
    build_tree_preflop_only(&cfg).expect("cap-3 preflop tree")
}

fn main() {
    let spec = production_game_v1();
    let root = std::env::var("PF_BLUEPRINT").unwrap_or_else(|_| "blueprint_out_v1".into());
    // L2_SUBDIR lets a re-bank write to a fresh dir (e.g. live2_m2) without clobbering
    // the live bank; swap it in once validated. Default "live2".
    let subdir = std::env::var("L2_SUBDIR").unwrap_or_else(|_| "live2".into());
    let out = format!("{root}/{subdir}");
    let cfv_root = format!("{root}/cfv");
    let threads: usize = std::env::var("L2_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(14);
    // CFV is keyed to the SOLVE, independent of the strategy bank's bet menu, and is
    // already banked from the original pot-only fill — default OFF on a re-bank to
    // avoid redundant work (set L2_CFV=1 to also refresh it).
    let bank_cfv: bool = std::env::var("L2_CFV").as_deref() == Ok("1");
    let bets = live2_bet_menu();

    // Discover the live-2 SPR buckets (rep = first cell per bin), skip all-in.
    let pft = cap3_preflop_tree();
    let mut buckets: HashMap<(u8, i64), SeamCell> = HashMap::new();
    for idx in 0..pft.num_nodes() {
        let n = &pft.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.board_state != BoardState::Flop as u8 { continue; }
        let cell = SeamCell::at_chance_node(&pft, idx, 6);
        if cell.live != 2 || spec.stack - cell.commit <= 0 { continue; } // all-in → rollout
        buckets.entry(cell.bucket_key(spec.stack)).or_insert(cell);
    }
    let mut blist: Vec<((u8, i64), SeamCell)> = buckets.into_iter().collect();
    blist.sort_by_key(|(k, _)| k.1);
    // Optional caps for a smoke fill (validate the pipeline before the full run).
    if let Some(mb) = std::env::var("L2_MAXBINS").ok().and_then(|s| s.parse::<usize>().ok()) {
        blist.truncate(mb);
    }
    let max_flops = std::env::var("L2_MAXFLOPS").ok().and_then(|s| s.parse::<usize>().ok());

    let canon = PreflopChanceTable::new(6, vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6])
        .canonical_flops.clone();
    let nflop = max_flops.map(|m| m.min(canon.len())).unwrap_or(canon.len());
    std::fs::create_dir_all(&out).expect("mkdir live2 out");
    // Manifest: bin → (commit, pot) rep, for the deployment loader.
    let mut manifest = String::from("# live-2 strategy bank: bin commit pot dir\n");
    for (key, c) in &blist {
        manifest.push_str(&format!("S{} commit={} pot={} dir=S{}\n", key.1, c.commit, c.pot, key.1));
    }
    std::fs::write(format!("{out}/manifest.txt"), &manifest).expect("write manifest");

    eprintln!(
        "live-2 strategy fill: {} buckets × {nflop} flops, {threads} threads, cfv={} → {out}",
        blist.len(), bank_cfv,
    );
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let total = blist.len();
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads.min(total) {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= total { break; }
                let (key, cell) = &blist[k];
                let tree = build_tree(&spec.flop_seam_config(2, cell.commit, cell.pot, bets.clone()))
                    .expect("live-2 seam tree");
                let dir = format!("{out}/S{}", key.1);
                std::fs::create_dir_all(&dir).expect("mkdir bin");
                for (fi, &canonical) in canon.iter().enumerate().take(nflop) {
                    let solver = solve_live2(canonical, fi, &tree);
                    save_live2_v2(&format!("{dir}/flop_{fi:04}.bp2"), &solver, cell.commit, cell.pot, fi)
                        .expect("save live-2 blob");
                    if bank_cfv {
                        let per_live = cfv_live2(canonical, fi, &tree);
                        write_prebanked(&cfv_root, *key, fi, &per_live);
                    }
                }
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                let el = t0.elapsed().as_secs_f64();
                eprintln!("[{d}/{total}] bin S{} (c={} p={}) done | {:.1} min | ETA {:.1} min",
                    key.1, cell.commit, cell.pot, el / 60.0, el / d as f64 * (total - d) as f64 / 60.0);
            });
        }
    });
    eprintln!("LIVE2_FILL_COMPLETE: {total} buckets in {:.1} min → {out}", t0.elapsed().as_secs_f64() / 60.0);
}
