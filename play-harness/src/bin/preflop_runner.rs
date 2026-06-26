//! PRODUCTION PREFLOP RUNNER (2026-06-14): solve the cap-3 preflop blueprint
//! against the banked postflop fill. The flop-entry continuation values come
//! from the READ-BANKED oracle (`banked_read_source` → load rep .bp →
//! root_cfv_from_avg), NOT a re-solve — so the preflop trains against exactly
//! the deployed postflop values, and the fill's solves are reused (~1/34 the
//! cost). Frozen oracle (refresh_every=0): one cache fill, then cheap cached
//! iterations converge the preflop strategy.
//!
//! Env: PF_BLUEPRINT (dir of the fill, default blueprint_out_v1), PF_OUT
//! (output dir, default preflop_out_v1), PF_ITERS (default 60).
//!
//! NOTE: needs the postflop fill COMPLETE (reads every (cell, flop)) and
//! ~10.6 GB RAM for the cap-3 CFR buffers (MAX_NA_PREFLOP=8). Do NOT run
//! alongside the GPU fill.

use std::io::Write;
use std::time::Instant;

use play_harness::preflop_oracle::banked_read_source;
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::postflop_oracle::{BucketKeyedOracle, PostflopValueOracle, SeamCell};
use solver_core::solver::preflop_cfr::{
    make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr,
};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::production_game_v1;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

/// The production preflop tree — the SINGLE SOURCE OF TRUTH shared with the
/// connected solve (`gpu_conn_solve`) and the runtime loader: 5 pot-relative raise
/// sizes + an explicit ALL-IN (na=8), cap-3, LIMP-INCLUSIVE (no_open_limp /
/// threebet_or_fold left FALSE so the loose-passive pool's limps/cold-calls are
/// modelled). Using `build_conn_preflop_tree` here makes the preflop strategy
/// interchangeable with the connected blueprint — the prerequisite for co-solving.
fn cap3_preflop_tree() -> FlatTree {
    solver_core::blueprint::build_conn_preflop_tree(6, 5).0
}

fn load_cells(root: &str) -> Vec<(u8, i32, i32, usize)> {
    let txt = std::fs::read_to_string(format!("{root}/cells.txt")).expect("cells.txt");
    txt.lines()
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
    let out = std::env::var("PF_OUT").unwrap_or_else(|_| "preflop_out_v1".into());
    // CONVERGENCE STOP: halt when the current strategy stops moving —
    // max|Δσ| < PF_EPS between checks (PF_EPS=0 ⇒ run to cap, logging the
    // curve). Checked every PF_CHECK iters, capped at PF_CAP.
    let cap: u32 = std::env::var("PF_CAP").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let eps: f32 = std::env::var("PF_EPS").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let check: u32 = std::env::var("PF_CHECK").ok().and_then(|s| s.parse().ok()).unwrap_or(5);

    let tree = cap3_preflop_tree();
    let table = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let canon = table.canonical_flops.clone();
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&tree);

    let cells = load_cells(&bp_root);
    let source = banked_read_source(bp_root.clone(), cells, canon, spec.stack);
    let mut oracle = BucketKeyedOracle::new(spec.stack, 6, 0, source);
    let mut solver = PreflopVectorCfr::new(&tree);
    if std::env::var("PF_DEBUG_UTG").is_ok() {
        use solver_core::abstraction::preflop_class::PreflopClass;
        use solver_core::card::card_from_str;
        let cl = |a: &str, b: &str| PreflopClass::from_combo(card_from_str(a).unwrap(), card_from_str(b).unwrap()).index();
        // emit per-action cfv at the UTG node (0) for AA, KK, T9s, 72o, 32o.
        // T9s included to decompose WHY the premium (AA) limps but the marginal
        // (T9s) raises — call-vs-raise gap + its SIZE distinguishes a genuine
        // frozen-oracle consequence (big AA call≫raise) from a Layer-2 residual
        // investment-bias (tiny AA call≳raise).
        solver.debug_emit = Some((0, vec![cl("Ac","Ad"), cl("Kc","Kd"), cl("Tc","9c"), cl("7c","2d"), cl("3c","2d")]));
        eprintln!("PF_DEBUG_UTG: emitting UTG action-cfv for AA,KK,T9s,72o,32o");
    }

    eprintln!(
        "preflop runner: {} nodes, {} infosets | cap {cap}, eps {eps:.3e} (0=anchor run), check/{check} | reading {bp_root}",
        tree.num_nodes(),
        solver.infoset_count
    );
    let t_all = Instant::now();
    let mut prev_avg: Option<Vec<f32>> = None;
    let mut it = 0u32;
    // Cross-iteration cache of the reach-independent live-traverser chance CFV
    // (frozen oracle ⇒ identical every iter; computed once on iter 1). Shared
    // with the exploitability check. This is the ~280s/iter → seconds/iter win.
    let mut chance_cache: Vec<std::collections::HashMap<u64, Vec<f32>>> = Vec::new();
    loop {
        if it >= cap {
            eprintln!("hit cap {cap} (no stop trigger — raise cap or inspect the curve)");
            break;
        }
        let key_fn = PreflopVectorCfr::seam_bucket_chance_key(&tree, 6, spec.stack);
        let t0 = Instant::now();
        solver.run_one_iteration_shared_chance_cached(&tree, &table, &mut oracle, &term_fn, key_fn, &mut chance_cache);
        it += 1;
        let train_s = t0.elapsed().as_secs_f64();
        if it == 1 || it % check == 0 {
            // CONVERGENCE SIGNAL: REACH-WEIGHTED change in the normalized
            // average strategy (the deployed object). Plain max|Δ avg| is
            // dominated by near-zero-reach noise (bounces ~1.0); the current
            // strategy oscillates; the summed exploitability is f32-saturated
            // by the live-6 ~1e21 BR. The reach-weighted average change is the
            // canonical CFR convergence measure — → 0 as the reached average
            // settles. (raw max logged alongside for context.)
            let reach = solver.compute_preflop_reach(&tree, None);
            let avg = solver.average_strategy(&tree);
            let (conv, raw_max) = match prev_avg.as_ref() {
                Some(p) => (
                    solver.avg_reach_weighted_delta(&tree, &reach, &avg, p),
                    avg.iter().zip(p).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max),
                ),
                None => (f32::INFINITY, f32::INFINITY),
            };
            // UTG open-frequency trajectory (the looseness signal): does trash
            // fold off as the average converges?
            {
                use solver_core::abstraction::preflop_class::PreflopClass;
                use solver_core::card::card_from_str;
                let nc = NUM_PREFLOP_CLASSES;
                let local = solver.local_offset[0];
                let na = tree.nodes[0].num_children as usize;
                let off = local * MAX_NA_PREFLOP * nc;
                let labels: Vec<u8> =
                    tree.node_children(0).iter().map(|&c| tree.nodes[c as usize].action_label).collect();
                // (raise%, fold%) per class from the average strategy: raise =
                // label 4, fold = label 0. Sane preflop ⇒ AA raises, 72o folds.
                let rf = |cl: usize, want: u8| -> f32 {
                    let s: f32 = (0..na).map(|a| avg[off + a * nc + cl].max(0.0)).sum();
                    if s <= 0.0 {
                        return 0.0;
                    }
                    let r: f32 =
                        (0..na).filter(|&a| labels[a] == want).map(|a| avg[off + a * nc + cl].max(0.0)).sum();
                    r / s
                };
                let ix = |a: &str, b: &str| {
                    PreflopClass::from_combo(card_from_str(a).unwrap(), card_from_str(b).unwrap()).index()
                };
                let pr = |a: &str, b: &str| (rf(ix(a, b), 4) * 100.0, rf(ix(a, b), 0) * 100.0);
                let (aa_r, aa_f) = pr("Ac", "Ad");
                let (t9_r, t9_f) = pr("Tc", "9c");
                let (o72_r, o72_f) = pr("7c", "2d");
                let (o32_r, o32_f) = pr("3c", "2d");
                eprintln!(
                    "    UTG raise/fold%: AA {aa_r:.0}/{aa_f:.0} | T9s {t9_r:.0}/{t9_f:.0} | 72o {o72_r:.0}/{o72_f:.0} | 32o {o32_r:.0}/{o32_f:.0}",
                );
            }
            prev_avg = Some(avg);
            eprintln!(
                "  iter {it:>3}: conv(reach-wt) {conv:.4e} | raw-max {raw_max:.4e} | {train_s:.1}s/iter | {:.1} min",
                t_all.elapsed().as_secs_f64() / 60.0,
            );
            if eps > 0.0 && conv < eps && it > check {
                eprintln!("CONVERGED: reach-wt Δ avg {conv:.4e} < eps {eps:.4e} at iter {it}");
                break;
            }
        } else {
            eprintln!("  [iter {it} trained in {train_s:.1}s | {:.1} min total]", t_all.elapsed().as_secs_f64() / 60.0);
        }
    }
    let iters = it;

    // Bank the average strategy (cum_strategy) + a header.
    std::fs::create_dir_all(&out).expect("mkdir out");
    let path = format!("{out}/preflop.blob");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("create"));
    f.write_all(b"SSPF1\n").unwrap();
    writeln!(
        f,
        "{{\"infosets\":{},\"max_na\":{},\"classes\":{},\"iters\":{iters},\"cap\":3,\"reads\":\"{bp_root}\"}}",
        solver.infoset_count, MAX_NA_PREFLOP, NUM_PREFLOP_CLASSES
    ).unwrap();
    for v in &solver.cum_strategy {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
    eprintln!("banked preflop avg strategy → {path} ({} done in {:.1} min)", solver.cum_strategy.len(), t_all.elapsed().as_secs_f64() / 60.0);
}
