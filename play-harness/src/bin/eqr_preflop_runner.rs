//! STANDALONE PREFLOP SOLVE on the EQR realized-EV terminal (2026-06-18,
//! preflop-RTS pivot). The flop-entry continuation values come from the
//! equity-realization Monte Carlo (`eqr::realized_ev_table_on_flop`) — NOT a
//! coupled postflop solve. Decoupled: preflop solves against flop-AVERAGED
//! realized EV; postflop is handled at the table by RTS. Kills the co-solve cost
//! and the board-abstraction poison.
//!
//! Config knobs (standing requirement): EQR_RAKE, EQR_RAKECAP_BB, EQR_STACK_BB,
//! EQR_ANTE_BB (game economics) + MC_BET, MC_SAMPLES, MC_NODRAWS (policy) +
//! PF_FLOPS (flop subset, default 120), PF_ITERS (default 60).
//!
//! GATE: the solved ranges — AA/KK raise, trash folds, UTG tighter than the field.

use std::collections::HashMap;

use play_harness::eqr::{realized_ev_table_on_flop, spr_to_opp_tier, EqrPolicy};
use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::{card_from_str, Card};
use solver_core::solver::postflop_oracle::{BucketKeyedOracle, SeamCell};
use solver_core::solver::preflop_cfr::{make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr};
use solver_core::solver::preflop_start_game::{flop_combo_layout, PreflopChanceTable};
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, GameSpec, UNITS_PER_BB};
use solver_core::tree::builder::build_tree_preflop_only;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

fn cap3_preflop_tree(spec: &GameSpec, n_raises: usize) -> FlatTree {
    // n_raises raise SIZES per decision; default = MAX_NA_PREFLOP-2 (=14 at na=16),
    // a rich big-action menu 0.5×..7.0× pot. Depth stays shallow via cap-3.
    let mrc = n_raises.min(MAX_NA_PREFLOP.saturating_sub(2)).max(1);
    let mut cfg = spec.preflop_tree_config(BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect(),
    });
    cfg.max_bets_per_street = BetCap::all(3);
    cfg.no_open_limp = true; // raise-or-fold opens (open-limping is dominated in 6-max)
    cfg.threebet_or_fold = true; // 3bet-or-fold facing an open (no flat, no jam — sized 3bet plays the EQR postflop)
    build_tree_preflop_only(&cfg).expect("cap-3 preflop tree")
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn eqr_file(dir: &str, key: (u8, i64), fi: usize) -> String {
    format!("{dir}/L{}_S{}/eqr_{fi:04}.f32", key.0, key.1)
}

fn read_eqr(dir: &str, key: (u8, i64), fi: usize, want_len: usize) -> Option<Vec<f32>> {
    let bytes = std::fs::read(eqr_file(dir, key, fi)).ok()?;
    if bytes.len() != want_len * 4 {
        return None; // partial / mismatched write — recompute
    }
    Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn write_eqr(dir: &str, key: (u8, i64), fi: usize, table: &[f32]) {
    let d = format!("{dir}/L{}_S{}", key.0, key.1);
    std::fs::create_dir_all(&d).expect("mkdir eqr cache");
    let mut bytes = Vec::with_capacity(table.len() * 4);
    for &x in table {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    // atomic-ish: write tmp then rename, so a kill mid-write never leaves a
    // half file that reads as valid (length check also guards this).
    let path = eqr_file(dir, key, fi);
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, &bytes).expect("write eqr tmp");
    std::fs::rename(&tmp, &path).expect("rename eqr");
}

/// EQR continuation source (banked_read_source shape). Realized-EV table per
/// (SPR-bucket cell, canonical flop) — reach- & traverser-independent (symmetric
/// limped pot). PERSISTENT: each table is read from / written to `dir` so a stall
/// or restart never recomputes a filled (cell, flop) — it just re-reads. Progress
/// is logged every `LOG_EVERY` fresh computes.
#[allow(clippy::too_many_arguments)]
fn eqr_source(
    dir: String,
    flop_index: HashMap<[Card; 3], usize>,
    stack: i32,
    rake_rate: f32,
    rake_cap: f32,
    policy: EqrPolicy,
) -> impl FnMut(SeamCell, u16, [Card; 3], &[Vec<f32>], u8) -> Vec<f32> {
    const LOG_EVERY: u64 = 500;
    let mut mem: HashMap<((u8, i64), [Card; 3]), Vec<f32>> = HashMap::new();
    let mut computed = 0u64;
    let mut read = 0u64;
    let t0 = std::time::Instant::now();
    move |cell, _folded_mask, canonical, _combo_reaches, _traverser| {
        let key = cell.bucket_key(stack);
        if let Some(v) = mem.get(&(key, canonical)) {
            return v.clone();
        }
        let fi = flop_index[&canonical];
        let layout: Vec<(u8, u8)> = flop_combo_layout(canonical);
        let table = if let Some(v) = read_eqr(&dir, key, fi, layout.len()) {
            read += 1;
            v
        } else {
            let flop = [canonical[0], canonical[1], canonical[2]];
            let seed = splitmix(
                (key.0 as u64)
                    ^ ((key.1 as u64) << 8)
                    ^ ((canonical[0] as u64) << 24)
                    ^ ((canonical[1] as u64) << 32)
                    ^ ((canonical[2] as u64) << 40),
            );
            let spr = (stack - cell.commit) as f32 / (cell.pot as f32).max(1.0);
            let opp_tier = spr_to_opp_tier(spr, policy.continue_min_made);
            let t = realized_ev_table_on_flop(
                &layout, flop, cell.live as usize, cell.pot as f32, rake_rate, rake_cap, policy, opp_tier, seed,
            );
            write_eqr(&dir, key, fi, &t);
            computed += 1;
            if computed % LOG_EVERY == 0 {
                eprintln!(
                    "    [EQR fill] {computed} computed + {read} read-from-disk | {:.0}/s | {:.1} min",
                    computed as f64 / t0.elapsed().as_secs_f64().max(1e-3),
                    t0.elapsed().as_secs_f64() / 60.0
                );
            }
            t
        };
        mem.insert((key, canonical), table.clone());
        table
    }
}

fn env_f(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn env_u(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() {
    // ── configurable game economics ──
    let mut spec = production_game_v1();
    spec.rake_rate = env_f("EQR_RAKE", spec.rake_rate as f32) as f64;
    spec.rake_cap = (env_f("EQR_RAKECAP_BB", spec.rake_cap as f32 / UNITS_PER_BB as f32) * UNITS_PER_BB as f32) as i32;
    spec.stack = (env_f("EQR_STACK_BB", spec.stack as f32 / UNITS_PER_BB as f32) * UNITS_PER_BB as f32) as i32;
    spec.ante = (env_f("EQR_ANTE_BB", spec.ante as f32 / UNITS_PER_BB as f32) * UNITS_PER_BB as f32) as i32;

    let policy = EqrPolicy {
        bet_frac: env_f("MC_BET", 1.0),
        use_draws: std::env::var("MC_NODRAWS").is_err(),
        continue_min_made: env_u("MC_TIER", 2) as u8,
        mc_samples: env_u("MC_SAMPLES", 120),
    };
    let n_flops = env_u("PF_FLOPS", 120);
    let iters = env_u("PF_ITERS", 60) as u32;
    let check = env_u("PF_CHECK", 10) as u32;

    let n_raises = env_u("PF_RAISES", MAX_NA_PREFLOP.saturating_sub(2)); // default full na=16 menu
    let tree = cap3_preflop_tree(&spec, n_raises);
    let table = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let _ = n_flops; // (full-flop cell-aware path; subset path is cell-blind, incompatible w/ BucketKeyedOracle)
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);

    // flop → index (for the on-disk EQR cache filename).
    let flop_index: HashMap<[Card; 3], usize> =
        table.canonical_flops.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    // cache dir namespaced by the config that changes the table values, so a
    // different rake/stack/ante/policy never reads a stale fill. Same config ⇒
    // a restart reads every already-filled (cell, flop) instead of recomputing.
    let base = std::env::var("EQR_CACHE").unwrap_or_else(|_| "eqr_cache".into());
    let tag = format!(
        "rof3_nr{}_r{}_c{}_s{}_a{}_b{}_d{}_t{}_m{}",
        n_raises, (spec.rake_rate * 1000.0) as i32, spec.rake_cap, spec.stack, spec.ante,
        (policy.bet_frac * 100.0) as i32, policy.use_draws as u8, policy.continue_min_made, policy.mc_samples
    );
    let dir = format!("{base}/{tag}");
    std::fs::create_dir_all(&dir).expect("mkdir eqr cache base");
    eprintln!("EQR table cache: {dir} (rerunnable — restart reads filled tables)");

    let source = eqr_source(dir.clone(), flop_index, spec.stack, spec.rake_rate as f32, spec.rake_cap as f32, policy);
    let mut oracle = BucketKeyedOracle::new(spec.stack, 6, 0, source);
    let mut solver = PreflopVectorCfr::new(&tree);

    eprintln!(
        "EQR PREFLOP SOLVE | rake {:.0}% cap {}u stack {}u ante {}u | bet {}×pot draws={} mc={} | ALL flops × {} iters\n\
         {} nodes, {} infosets (iter-1 builds the EQR table cache, then cheap)",
        spec.rake_rate * 100.0, spec.rake_cap, spec.stack, spec.ante,
        policy.bet_frac, policy.use_draws, policy.mc_samples, iters,
        tree.num_nodes(), solver.infoset_count
    );

    // ── PARALLEL PRE-FILL: every (SPR-cell × flop) EQR table, across ALL cores ──
    // (each table serial, many tables concurrent = full utilization). Skips tables
    // already on disk, so it's rerunnable. Then the CFR oracle just reads.
    {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let mut cells: HashMap<(u8, i64), SeamCell> = HashMap::new();
        for i in 0..tree.num_nodes() {
            if tree.nodes[i].is_chance() {
                let cell = SeamCell::at_chance_node(&tree, i, 6);
                cells.entry(cell.bucket_key(spec.stack)).or_insert(cell);
            }
        }
        let jobs: Vec<((u8, i64), SeamCell, usize, [Card; 3])> = cells
            .iter()
            .flat_map(|(&k, &c)| {
                table.canonical_flops.iter().enumerate().map(move |(fi, &f)| (k, c, fi, f))
            })
            .collect();
        let total = jobs.len();
        eprintln!(
            "PRE-FILL: {} cells × {} flops = {total} tables, parallel across cores",
            cells.len(),
            table.canonical_flops.len()
        );
        let done = AtomicUsize::new(0);
        let tf = std::time::Instant::now();
        jobs.par_iter().for_each(|&(key, cell, fi, canonical)| {
            let layout: Vec<(u8, u8)> = flop_combo_layout(canonical);
            if read_eqr(&dir, key, fi, layout.len()).is_some() {
                return; // already filled — rerunnable
            }
            let seed = splitmix(
                (key.0 as u64)
                    ^ ((key.1 as u64) << 8)
                    ^ ((canonical[0] as u64) << 24)
                    ^ ((canonical[1] as u64) << 32)
                    ^ ((canonical[2] as u64) << 40),
            );
            let spr = (spec.stack - cell.commit) as f32 / (cell.pot as f32).max(1.0);
            let opp_tier = spr_to_opp_tier(spr, policy.continue_min_made);
            let vals = realized_ev_table_on_flop(
                &layout, [canonical[0], canonical[1], canonical[2]], cell.live as usize,
                cell.pot as f32, spec.rake_rate as f32, spec.rake_cap as f32, policy, opp_tier, seed,
            );
            write_eqr(&dir, key, fi, &vals);
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if d % 5000 == 0 {
                eprintln!(
                    "  [pre-fill] {d} computed | {:.0}/s | {:.1} min",
                    d as f64 / tf.elapsed().as_secs_f64().max(1e-3),
                    tf.elapsed().as_secs_f64() / 60.0
                );
            }
        });
        eprintln!(
            "PRE-FILL done: {} fresh tables in {:.1}s (rest already cached)",
            done.load(Ordering::Relaxed),
            tf.elapsed().as_secs_f64()
        );
    }

    let mt = std::env::var("MC_MT").map(|v| v != "0").unwrap_or(true);
    eprintln!("CFR: {} (MC_MT=0 to force serial)", if mt { "PARALLEL (per-traverser)" } else { "serial" });
    // CONVERGENCE STOP: run to a cap, but halt early when the REACH-WEIGHTED change
    // in the average strategy stops moving (the canonical CFR convergence measure;
    // plain max|Δ| bounces on near-zero-reach infosets). PF_EPS=0 ⇒ run to PF_ITERS.
    let eps: f32 = env_f("PF_EPS", 0.002);
    let t0 = std::time::Instant::now();
    let mut chance_cache: Vec<HashMap<u64, Vec<f32>>> = Vec::new();
    let mut prev_avg: Option<Vec<f32>> = None;
    for it in 1..=iters {
        let key_fn = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, spec.stack);
        if mt {
            solver.run_one_iteration_shared_chance_cached_par(&tree, &table, &mut oracle, &term_fn, key_fn, &mut chance_cache);
        } else {
            solver.run_one_iteration_shared_chance_cached(&tree, &table, &mut oracle, &term_fn, key_fn, &mut chance_cache);
        }
        if it == 1 || it % check == 0 {
            print_ranges(&solver, &tree, it, t0.elapsed().as_secs_f64());
            let reach = solver.compute_preflop_reach(&tree, None);
            let avg = solver.average_strategy(&tree);
            if let Some(p) = &prev_avg {
                let conv = solver.avg_reach_weighted_delta(&tree, &reach, &avg, p);
                eprintln!("    conv(reach-wt Δavg) {conv:.5}");
                if eps > 0.0 && conv < eps && it > 20 {
                    eprintln!("CONVERGED: reach-wt Δavg {conv:.5} < eps {eps} at iter {it}");
                    break;
                }
            }
            prev_avg = Some(avg);
        }
    }
    eprintln!("\ndone in {:.1}s", t0.elapsed().as_secs_f64());
}

/// UTG (node 0) raise/fold% for signature classes from the average strategy.
fn print_ranges(solver: &PreflopVectorCfr, tree: &FlatTree, it: u32, secs: f64) {
    let avg = solver.average_strategy(tree);
    let nc = NUM_PREFLOP_CLASSES;
    let local = solver.local_offset[0];
    let na = tree.nodes[0].num_children as usize;
    let off = local * MAX_NA_PREFLOP * nc;
    let labels: Vec<u8> = tree.node_children(0).iter().map(|&c| tree.nodes[c as usize].action_label).collect();
    let rf = |cl: usize, want: u8| -> f32 {
        let s: f32 = (0..na).map(|a| avg[off + a * nc + cl].max(0.0)).sum();
        if s <= 0.0 {
            return 0.0;
        }
        let r: f32 = (0..na).filter(|&a| labels[a] == want).map(|a| avg[off + a * nc + cl].max(0.0)).sum();
        r / s
    };
    let ix = |a: &str, b: &str| PreflopClass::from_combo(card_from_str(a).unwrap(), card_from_str(b).unwrap()).index();
    // label 4 = raise, 0 = fold (builder convention).
    let pr = |a: &str, b: &str| {
        let cl = ix(a, b);
        (rf(cl, 4) * 100.0, rf(cl, 0) * 100.0)
    };
    let hands = [
        ("AA", "Ac", "Ad"), ("KK", "Kc", "Kd"), ("QQ", "Qc", "Qd"),
        ("AKs", "Ac", "Kc"), ("AJs", "Ac", "Jc"), ("99", "9c", "9d"),
        ("T9s", "Tc", "9c"), ("87s", "8c", "7c"), ("A5o", "Ac", "5d"),
        ("KJo", "Kc", "Jd"), ("72o", "7c", "2d"), ("32o", "3c", "2d"),
    ];
    // ALL OPENING POSITIONS via the fold-chain (UTG → HJ → CO → BTN → SB): each
    // opener is the fold-child of the prior (everyone before folded). BB closes the
    // action — it has no open, only defense. raise% (label 4 or all-in 5) per hand.
    eprintln!("  iter {it:>3} ({secs:.0}s) — OPEN raise% by position:");
    let pos_names = ["UTG", "HJ", "CO", "BTN", "SB "];
    let mut pnode = 0usize;
    for pname in pos_names {
        if !tree.nodes[pnode].is_player() {
            break;
        }
        let plocal = solver.local_offset[pnode];
        let pna = tree.nodes[pnode].num_children as usize;
        let poff = plocal * MAX_NA_PREFLOP * nc;
        let plabels: Vec<u8> =
            tree.node_children(pnode).iter().map(|&c| tree.nodes[c as usize].action_label).collect();
        let praise = |cl: usize| -> f32 {
            let s: f32 = (0..pna).map(|a| avg[poff + a * nc + cl].max(0.0)).sum();
            if s <= 0.0 {
                return 0.0;
            }
            let r: f32 = (0..pna)
                .filter(|&a| plabels[a] == 4 || plabels[a] == 5)
                .map(|a| avg[poff + a * nc + cl].max(0.0))
                .sum();
            r / s
        };
        let mut pline = format!("    {pname}");
        for (name, a, b) in hands {
            pline.push_str(&format!(" {name} {:.0}", praise(ix(a, b)) * 100.0));
        }
        eprintln!("{pline}");
        match tree
            .node_children(pnode)
            .iter()
            .find(|&&k| tree.nodes[k as usize].action_label == 0 && tree.nodes[k as usize].is_player())
        {
            Some(&fc) => pnode = fc as usize,
            None => break,
        }
    }

    // FACING-A-RAISE / 3-BET gate: the next player after UTG's open-raise. Follow a
    // mid-size raise child of node 0 to the player who now faces it; report
    // 3bet%(label4) / call%(label2) / fold%(label0). Watch (per review): if strong
    // hands FLAT (high call%) instead of 3betting, the multiway-call attractor needs
    // the same raise-or-fold fix one node deeper.
    let raise_kids: Vec<usize> = tree
        .node_children(0)
        .iter()
        .filter(|&&k| tree.nodes[k as usize].action_label == 4)
        .map(|&k| k as usize)
        .collect();
    if let Some(&fr) = raise_kids.get(raise_kids.len() / 2) {
        if tree.nodes[fr].is_player() {
            let flocal = solver.local_offset[fr];
            let fna = tree.nodes[fr].num_children as usize;
            let foff = flocal * MAX_NA_PREFLOP * nc;
            let flabels: Vec<u8> =
                tree.node_children(fr).iter().map(|&c| tree.nodes[c as usize].action_label).collect();
            let frac = |cl: usize, want: u8| -> f32 {
                let s: f32 = (0..fna).map(|a| avg[foff + a * nc + cl].max(0.0)).sum();
                if s <= 0.0 {
                    return 0.0;
                }
                let r: f32 =
                    (0..fna).filter(|&a| flabels[a] == want).map(|a| avg[foff + a * nc + cl].max(0.0)).sum();
                r / s
            };
            eprintln!(
                "          facing-a-raise (seat {}) — 3bet%/call%/fold%:",
                tree.nodes[fr].player_id
            );
            let mut fl = String::new();
            for (name, a, b) in hands {
                let cl = ix(a, b);
                fl.push_str(&format!(
                    "{name} {:.0}/{:.0}/{:.0}   ",
                    frac(cl, 4) * 100.0,
                    frac(cl, 2) * 100.0,
                    frac(cl, 0) * 100.0
                ));
            }
            eprintln!("          {fl}");
        }
    }
}
