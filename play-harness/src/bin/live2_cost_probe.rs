//! LIVE-2 RICHER-MENU COST PROBE: measure the tree size, per-flop solve time, and
//! banked-buffer bytes of candidate HU bet menus, to project the full re-bank cost
//! (1755 flops × ~26 SPR bins) BEFORE launching. The current bank is pot-only
//! (bet:[1.0], raise:[]); this prints the multiplier each richer menu would cost.
//!
//! Run: cargo run --release -p play-harness --bin live2_cost_probe

use std::time::Instant;

use play_harness::live2_bank::solve_live2;
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::{build_tree, build_tree_preflop_only};
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP, NODE_TYPE_CHANCE};
use solver_core::solver::postflop_oracle::SeamCell;

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
    // pick a representative live-2 SPR bin (mid SPR).
    let pft = cap3_preflop_tree();
    let mut rep: Option<SeamCell> = None;
    for idx in 0..pft.num_nodes() {
        let n = &pft.nodes[idx];
        if n.node_type != NODE_TYPE_CHANCE || n.board_state != BoardState::Flop as u8 {
            continue;
        }
        let cell = SeamCell::at_chance_node(&pft, idx, 6);
        if cell.live != 2 || spec.stack - cell.commit <= 0 {
            continue;
        }
        // mid SPR-ish: commit ~10-14
        if cell.commit >= 8 && cell.commit <= 16 {
            rep = Some(cell);
            break;
        }
    }
    let rep = rep.expect("a live-2 rep cell");
    eprintln!("rep cell: live=2 commit={} pot={}", rep.commit, rep.pot);

    let canon = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    )
    .canonical_flops
    .clone();
    let nflop = canon.len();
    let nbins = 26usize; // observed in the deployed bank

    let menus: Vec<(&str, BetSizeOptions)> = vec![
        ("M0 current  bet[1.0] raise[]", BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] }),
        ("M1 search    bet[.5,1] raise[1]", BetSizeOptions { bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)], raise: vec![BetSize::PotRelative(1.0)] }),
        ("M2 rich      bet[.33,.66,1] raise[.66,1]", BetSizeOptions { bet: vec![BetSize::PotRelative(0.33), BetSize::PotRelative(0.66), BetSize::PotRelative(1.0)], raise: vec![BetSize::PotRelative(0.66), BetSize::PotRelative(1.0)] }),
    ];

    // Solve a few flops per menu to average out per-flop variance.
    let probe_flops = [0usize, 100, 800, 1500];
    println!("\n{:<42} {:>7} {:>9} {:>8} {:>8} {:>8}", "menu", "nodes", "solve/flop", "flopMB", "turnMB", "rivMB");
    for (name, menu) in &menus {
        let tree = build_tree(&spec.flop_seam_config(2, rep.commit, rep.pot, menu.clone()))
            .expect("seam tree");
        let nodes = tree.num_nodes();
        let mut tot = 0f64;
        let (mut fl, mut tl, mut rl) = (0usize, 0usize, 0usize);
        for &fi in &probe_flops {
            let t0 = Instant::now();
            let s = solve_live2(canon[fi], fi, &tree);
            tot += t0.elapsed().as_secs_f64();
            fl = s.cum_strategy_flop().len() * 4;
            tl = s.cum_strategy_turn().len() * 4;
            rl = s.cum_strategy_river().len() * 4;
        }
        let per_flop = tot / probe_flops.len() as f64;
        println!(
            "{:<42} {:>7} {:>7.3}s {:>8.2} {:>8.2} {:>8.2}",
            name, nodes, per_flop, fl as f64 / 1e6, tl as f64 / 1e6, rl as f64 / 1e6
        );
    }

    // ── Compression study on the M2 flop buffer: f32 → normalized → u8 → zstd ──
    let m2 = BetSizeOptions {
        bet: vec![BetSize::PotRelative(0.33), BetSize::PotRelative(0.66), BetSize::PotRelative(1.0)],
        raise: vec![BetSize::PotRelative(0.66), BetSize::PotRelative(1.0)],
    };
    let tree = build_tree(&spec.flop_seam_config(2, rep.commit, rep.pot, m2)).expect("m2 tree");
    let mut s = solve_live2(canon[0], 0, &tree);
    s.freeze_average_strategy(&tree); // normalize cum → avg into strategy_flop
    let avg: &[f32] = s.strategy_flop(); // values in [0,1]
    let f32_bytes = avg.len() * 4;
    // quantize each prob to u8 (×255 round). u16 variant for comparison.
    let u8buf: Vec<u8> = avg.iter().map(|&p| (p.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    let u16buf: Vec<u8> = avg
        .iter()
        .flat_map(|&p| ((p.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes())
        .collect();
    let f32_le: Vec<u8> = avg.iter().flat_map(|&p| p.to_le_bytes()).collect();
    let z_f32 = zstd::encode_all(&f32_le[..], 9).unwrap().len();
    let z_u8 = zstd::encode_all(&u8buf[..], 9).unwrap().len();
    let z_u16 = zstd::encode_all(&u16buf[..], 9).unwrap().len();
    let proj = |bytes: usize| bytes as f64 * nflop as f64 * nbins as f64 / 1e9;
    println!("\n── M2 FLOP-ONLY buffer compression (one flop) ──");
    println!("  raw f32          {:>9.3} MB   → bank {:>7.1} GB", f32_bytes as f64 / 1e6, proj(f32_bytes));
    println!("  zstd(f32)        {:>9.3} MB   → bank {:>7.1} GB", z_f32 as f64 / 1e6, proj(z_f32));
    println!("  u8 quant         {:>9.3} MB   → bank {:>7.1} GB", u8buf.len() as f64 / 1e6, proj(u8buf.len()));
    println!("  zstd(u8)         {:>9.3} MB   → bank {:>7.1} GB   [{:.1}× vs raw f32]", z_u8 as f64 / 1e6, proj(z_u8), f32_bytes as f64 / z_u8 as f64);
    println!("  zstd(u16)        {:>9.3} MB   → bank {:>7.1} GB", z_u16 as f64 / 1e6, proj(z_u16));
    println!("\n(bank = per-flop bytes × {nflop} flops × {nbins} bins; FLOP-ONLY since decide_live2 never reads turn/river)");
}
