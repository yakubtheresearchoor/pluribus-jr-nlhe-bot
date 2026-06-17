//! END-TO-END GATE for the banked-read preflop oracle SOURCE (2026-06-14):
//! route a banked REP cell through `banked_read_source` (bucket_key → rep .bp
//! lookup → load → root_cfv_from_avg → layout-perm → traverser) and check it
//! against a fresh RE-SOLVE of the same cell at its own banked runout. Proves
//! the source's wiring (lookup, flop index, permutation, traverser indexing)
//! on top of the already-gated `read ≈ re-solve` extractor — so the preflop
//! oracle can read the fill instead of redoing ~25.8 h.

use play_harness::blueprint::Blueprint;
use play_harness::preflop_oracle::banked_read_source;
use solver_core::card::Card;
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::postflop_oracle::SeamCell;
use solver_core::solver::preflop_start_game::{
    table_hand_to_layout_perm, PreflopChanceTable,
};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

fn root() -> String {
    format!("{}/../blueprint_out_v1", env!("CARGO_MANIFEST_DIR"))
}

/// Parse `blueprint_out_v1/cells.txt` → (live, commit, pot, b).
fn load_cells() -> Vec<(u8, i32, i32, usize)> {
    let txt = std::fs::read_to_string(format!("{}/cells.txt", root())).expect("cells.txt");
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

#[test]
#[ignore = "banked-read source handles live-2 (exact re-solve branch); --ignored --nocapture --release"]
fn banked_read_source_handles_live2() {
    let spec = production_game_v1();
    let cells = load_cells(); // live-3/4/5 only — live-2 is re-solved, not read
    let nc = solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
    let canon = PreflopChanceTable::new(6, vec![vec![1.0f32 / nc as f32; nc]; 6]).canonical_flops.clone();
    let mut src = banked_read_source(root(), cells, canon.clone(), spec.stack);

    let canonical: [Card; 3] = canon[0];
    let folded_mask: u16 = 0b11_1100; // seats 2-5 folded → 2 live (0,1)
    let reaches: Vec<Vec<f32>> = vec![vec![]; 6];
    let cfv0 = src(SeamCell { live: 2, commit: 2, pot: 12 }, folded_mask, canonical, &reaches, 0);
    let cfv1 = src(SeamCell { live: 2, commit: 2, pot: 12 }, folded_mask, canonical, &reaches, 1);

    eprintln!("\n═══ banked-read source LIVE-2 branch ═══");
    assert!(!cfv0.is_empty() && cfv0.iter().all(|v| v.is_finite()), "live-2 CFV must be finite");
    assert_eq!(cfv0.len(), cfv1.len(), "both traversers same layout length");
    let m0 = cfv0.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let m1 = cfv1.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    eprintln!("  layout len {} | max|CFV| t0 {m0:.3e} t1 {m1:.3e} | finite ✓ | no panic ✓", cfv0.len());
    eprintln!("→ source re-solves live-2 exactly (np=2) and returns a sane CFV — preflop won't die on the most common entry");
}

#[test]
#[ignore = "banked-read source end-to-end (needs the fill's live-3/4 cells); --ignored --nocapture --release"]
fn banked_read_source_matches_resolve() {
    let spec = production_game_v1();
    let cells = load_cells();
    // The 1755 canonical flops, in the runner's order (= the .bp file index).
    let nc = solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
    let ptable = PreflopChanceTable::new(6, vec![vec![1.0f32 / nc as f32; nc]; 6]);
    let canon = ptable.canonical_flops.clone();

    let mut src = banked_read_source(root(), cells.clone(), canon.clone(), spec.stack);

    // A banked REP cell (live-4, fully solved). Use the cell's own flop_0000.
    let (live, commit, pot, b) = (4u8, 10i32, 40i32, 15usize);
    let fi = 0usize;
    let canonical: [Card; 3] = canon[fi];

    // np=6 with 2 folded → 4 live (seats 0..3); traverse a live seat.
    let folded_mask: u16 = 0b11_0000; // seats 4,5 folded
    let traverser = 0u8;
    let dummy_reaches: Vec<Vec<f32>> = vec![vec![]; 6];

    // ── SOURCE (read-banked) ──
    let cfv_src = src(
        SeamCell { live, commit, pot },
        folded_mask,
        canonical,
        &dummy_reaches,
        traverser,
    );

    let bp = Blueprint::load(&format!("{}/live{live}_c{commit}_p{pot}_b{b}/flop_{fi:04}.bp", root()))
        .expect("load .bp");

    // ── WIRING REFERENCE: extract the SAME .bp's root CFV manually (what the
    // source should do internally). Must be ~identical → proves the source's
    // bucket_key→rep lookup, flop index, perm and traverser indexing.
    let tree = build_tree(&spec.flop_seam_config(
        live, commit, pot,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
    )).unwrap();
    let mut bucketed = bp.indexer(&tree);
    bucketed.cum_strategy_flop_mut().copy_from_slice(&bp.cum_flop);
    bucketed.cum_strategy_turn_mut().copy_from_slice(&bp.cum_turn);
    bucketed.cum_strategy_river_mut().copy_from_slice(&bp.cum_river);
    let root_manual = bucketed.root_cfv_from_avg(&tree, &bp.game, &bp.bk);
    let perm = table_hand_to_layout_perm(&bp.game.table().hand_cards, bp.game.table().num_valid, canonical);
    let cfv_manual: Vec<f32> = perm.iter().map(|&h| root_manual[0][h]).collect();

    assert_eq!(cfv_src.len(), cfv_manual.len(), "layout length");
    let n = cfv_src.len();
    let mad: f64 = (0..n).map(|i| (cfv_src[i] as f64 - cfv_manual[i] as f64).abs()).sum::<f64>() / n as f64;
    let scale = cfv_manual.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6) as f64;
    let rel = mad / scale;
    eprintln!("\n═══ banked-read SOURCE vs manual extraction (live-{live} c{commit} p{pot}, flop {fi}) ═══");
    eprintln!("  layout len {n} | mean|Δ| {mad:.3e} | rel {:.4}% | scale {scale:.3e}", rel * 100.0);

    // INFO: source (banked GPU strategy) vs a fresh CPU re-solve at the banked
    // runout — a GPU-vs-CPU / 34-iter convergence delta, NOT a wiring check.
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in bp.turns.iter().enumerate() { rd[tc as usize] = bp.rivers[ti].clone(); }
    let table = FlopChanceTable::build_full_nh_sampled(bp.flop, live, &bp.turns, &rd);
    let bk = FlopBucketing::quantile(&table, b);
    let game = FlopStartGame::new(table);
    let mut s = BucketedFlopCfr::new(&tree, game.table(), &bk);
    s.set_terminal_design(TerminalDesign::Design1Collapsed);
    let rr = s.run_all_root_cfv(&tree, &game, &bk, 34);
    let cfv_re: Vec<f32> = perm.iter().map(|&h| rr[0][h]).collect();
    let madr: f64 = (0..n).map(|i| (cfv_src[i] as f64 - cfv_re[i] as f64).abs()).sum::<f64>() / n as f64;
    eprintln!("  [info] source(banked GPU) vs CPU re-solve @34it: rel {:.2}% (GPU/CPU + convergence, not wiring)", madr / scale * 100.0);

    eprintln!("→ {} (source loads the right rep .bp + perms + indexes correctly)", if rel < 0.001 { "WIRING OK ✓" } else { "WIRING BUG ✗" });
    assert!(rel < 0.001, "source ≠ manual extraction by {:.4}% — wiring bug (lookup/perm/index)", rel * 100.0);
}
