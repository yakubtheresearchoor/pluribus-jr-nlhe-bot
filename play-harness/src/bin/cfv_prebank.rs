//! CFV PRE-BANK (2026-06-15, option 1): compute every (SPR bucket × flop)
//! flop-root CFV ONCE, in parallel, and store it — so the preflop solve's fill
//! iteration is a pure read instead of per-node extraction/re-solve. Reads the
//! postflop fill for live-3/4/5 (`cfv_from_banked`), re-solves live-2 exactly
//! (`cfv_live2`); live-2 buckets are auto-discovered by walking the cap-3
//! preflop tree's flop-entry chance nodes. Output: <bp>/cfv/L{live}_S{bin}/cfv_NNNN.f32.
//!
//! Env: PF_BLUEPRINT (default blueprint_out_v1), CFV_THREADS (14),
//! CFV_LIMIT (process only the first N buckets — throughput probe).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use play_harness::preflop_oracle::{cfv_from_banked, cfv_live2, write_prebanked};
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::postflop_oracle::SeamCell;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, NODE_TYPE_CHANCE};

/// SAME unified tree as the preflop solve (`preflop_runner`) and the connected
/// blueprint — `build_conn_preflop_tree(6, 5)`, limp-inclusive + all-ins. The
/// pre-bank MUST discover live-2 buckets from the IDENTICAL tree the runner
/// queries, else the runner hits non-pre-banked buckets and falls back to the
/// slow on-the-fly re-solve. (Was a divergent raise-or-fold menu — tree drift.)
fn cap3_preflop_tree() -> FlatTree {
    solver_core::blueprint::build_conn_preflop_tree(6, 5).0
}

fn load_cells(root: &str) -> Vec<(u8, i32, i32, usize)> {
    std::fs::read_to_string(format!("{root}/cells.txt"))
        .expect("cells.txt")
        .lines()
        .filter(|l| l.starts_with("CELL live="))
        .map(|l| {
            let g = |k: &str| -> i64 {
                let s = &l[l.find(&format!("{k}=")).unwrap() + k.len() + 1..];
                s.split_whitespace().next().unwrap().parse().unwrap()
            };
            (g("live") as u8, g("commit") as i32, g("pot") as i32, g("b") as usize)
        })
        .collect()
}

fn main() {
    let spec = production_game_v1();
    let bp_root = std::env::var("PF_BLUEPRINT").unwrap_or_else(|_| "blueprint_out_v1".into());
    let cfv_root = format!("{bp_root}/cfv");
    let threads: usize = std::env::var("CFV_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(14);
    let limit: usize = std::env::var("CFV_LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };

    // live-3/4/5 reps (read .bp) from cells.txt.
    let mut buckets: HashMap<(u8, i64), (SeamCell, Option<String>)> = HashMap::new();
    for (live, commit, pot, b) in load_cells(&bp_root) {
        let key = SeamCell { live, commit, pot }.bucket_key(spec.stack);
        buckets.insert(key, (SeamCell { live, commit, pot }, Some(format!("{bp_root}/live{live}_c{commit}_p{pot}_b{b}"))));
    }
    // live-2 buckets: auto-discover from the preflop tree (re-solve, no .bp).
    let pft = cap3_preflop_tree();
    for idx in 0..pft.num_nodes() {
        let n = &pft.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.board_state != BoardState::Flop as u8 { continue; }
        let cell = SeamCell::at_chance_node(&pft, idx, 6);
        if cell.live != 2 || spec.stack - cell.commit <= 0 { continue; } // all-in → rollout
        buckets.entry(cell.bucket_key(spec.stack)).or_insert((cell, None));
    }

    let canon = PreflopChanceTable::new(6, vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6])
        .canonical_flops.clone();
    let mut blist: Vec<((u8, i64), (SeamCell, Option<String>))> = buckets.into_iter().collect();
    blist.sort_by_key(|(k, _)| (k.0, k.1));
    let total = blist.len().min(limit);
    let n2 = blist[..total].iter().filter(|(_, (c, _))| c.live == 2).count();
    eprintln!("CFV pre-bank: {total} buckets ({n2} live-2 re-solve) × {} flops, {threads} threads → {cfv_root}", canon.len());

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads.min(total) {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= total { break; }
                let (key, (cell, dir)) = &blist[k];
                // SKIP already-complete buckets (resume): the flop-root CFVs are
                // stride-independent VALUE vectors, so a bucket with all `canon.len()`
                // .f32 files is done and reusable. Lets a re-run finish only the
                // missing buckets (e.g. the slow deepest live-5) instead of redoing
                // the expensive live-2/3/4 re-solves.
                let bdir = format!("{cfv_root}/L{}_S{}", key.0, key.1);
                let have = std::fs::read_dir(&bdir)
                    .map(|rd| rd.filter(|e| e.as_ref().ok().and_then(|e| e.path().extension().map(|x| x == "f32")).unwrap_or(false)).count())
                    .unwrap_or(0);
                if have >= canon.len() {
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 5 == 0 || d == total { eprintln!("[{d}/{total}] (skip — complete)"); }
                    continue;
                }
                let tree = build_tree(&spec.flop_seam_config(cell.live, cell.commit, cell.pot, bets.clone()))
                    .expect("seam tree");
                for (fi, &canonical) in canon.iter().enumerate() {
                    let per_live = match dir {
                        Some(d) => cfv_from_banked(d, fi, &tree, canonical),
                        None => cfv_live2(canonical, fi, &tree),
                    };
                    write_prebanked(&cfv_root, *key, fi, &per_live);
                }
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 5 == 0 || d == total {
                    let el = t0.elapsed().as_secs_f64();
                    eprintln!("[{d}/{total}] {:.1} min | ETA {:.1} min", el / 60.0, el / d as f64 * (total - d) as f64 / 60.0);
                }
            });
        }
    });
    eprintln!("DONE: {total} buckets in {:.1} min", t0.elapsed().as_secs_f64() / 60.0);
}
