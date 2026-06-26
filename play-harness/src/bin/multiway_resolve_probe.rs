//! END-TO-END multiway turn/river re-solve probe (Piece 1, P0 #1).
//!
//! Measures whether a full-nh multiway (live-3/4) turn/river re-solve fits a
//! real-time budget once the showdown is the FACTORED O(nh·2^K) one instead of
//! the exact O(nh^K) brute force. Two parts:
//!
//!   LATENCY: factored re-solve wall-time for live-3/4 river + turn at the full
//!            M2 bet menu (turn nests a check-only river, like live-2).
//!   FIDELITY: on a SMALL menu (so exact is tractable), solve a live-3 river
//!            both ways and report root avg-strategy L1 distance — the
//!            strategy-level gate the showdown EV bound (<1% pot) predicts small.
//!
//! Run: cargo run --release -p play-harness --bin multiway_resolve_probe
//! Env: MODE=latency|fidelity|both (default both), ITERS (default 64).

use std::time::Instant;

use play_harness::live2_bank::live2_bet_menu;
use solver_core::card::{card_from_str, Card};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::{build_tree, build_tree_with_bet_override};

/// Solve a multiway street subgame; return (root avg strategy [na×nh], na, nh, wall_ms).
fn solve(
    board: &[u8],
    np: u8,
    commit: i32,
    pot: i32,
    menu: BetSizeOptions,
    factored: bool,
    iters: u32,
) -> (Vec<Vec<f32>>, usize, usize, f64) {
    let spec = production_game_v1();
    let board_c: Vec<Card> = board.to_vec();
    let ranges = vec![vec![1.0f32; 1326]; np as usize];
    let (state, table) = match board.len() {
        4 => (BoardState::Turn, ChanceTable::compute_turn_start(&board_c, &ranges, np)),
        5 => (BoardState::River, ChanceTable::compute_river_start(&board_c, &ranges, np)),
        _ => panic!("board must be 4 or 5 cards"),
    };
    let nh = table.num_valid;
    let mut game = TurnStartGame::new(table);
    if factored {
        game = game.with_factored();
    }
    let cfg = spec.street_seam_config(state, np, commit, pot, menu);
    let tree = if board.len() == 4 {
        let check_only = BetSizeOptions { bet: vec![], raise: vec![] };
        build_tree_with_bet_override(&cfg, &[(BoardState::River, check_only)]).unwrap()
    } else {
        build_tree(&cfg).unwrap()
    };
    let na = tree.nodes[0].num_children as usize;
    let nnodes = tree.num_nodes();
    if std::env::var("VERBOSE").as_deref() == Ok("1") {
        let mut nchance = 0;
        let mut nterm = 0;
        let mut nplayer = 0;
        let mut chance_fanout = 0;
        for i in 0..nnodes {
            match tree.nodes[i].node_type {
                1 => {
                    nchance += 1;
                    chance_fanout += tree.nodes[i].num_children as usize;
                }
                0 => nterm += 1,
                _ => nplayer += 1,
            }
        }
        eprintln!(
            "    [debug] built: nodes={nnodes} (player={nplayer} term={nterm} chance={nchance} chance_fanout={chance_fanout}) nh={nh} np={np} na={na} (solving {iters}it)"
        );
    }
    // TEST: flush denormals to zero (FPCR.FZ, bit 24) on arm64 — reach products
    // underflow to subnormals in a real solve, and subnormal arithmetic is ~100×
    // slower microcode. Gated by FTZ=1 to A/B it.
    #[cfg(target_arch = "aarch64")]
    if std::env::var("FTZ").as_deref() == Ok("1") {
        unsafe {
            let mut fpcr: u64;
            core::arch::asm!("mrs {}, fpcr", out(reg) fpcr);
            fpcr |= 1 << 24;
            core::arch::asm!("msr fpcr, {}", in(reg) fpcr);
        }
    }
    let mut cfr = VectorCfr::new(&tree, vec![nh; np as usize]);
    let t = Instant::now();
    cfr.run(&tree, &game, iters);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    if std::env::var("VERBOSE").as_deref() == Ok("1") {
        eprintln!("    [debug] nodes={nnodes} nh={nh} np={np} iters={iters} solve={ms:.0}ms");
    }
    (cfr.get_average_strategy(0, na, nh), na, nh, ms)
}

fn main() {
    let card = |s: &str| card_from_str(s).unwrap();
    let mode = std::env::var("MODE").unwrap_or_else(|_| "both".into());
    let iters: u32 = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(64);

    // live-3: 3 players, each committed 20 entering the street, pot=60 (equal).
    let turn4: Vec<u8> = ["Ks", "9d", "4c", "2h"].iter().map(|s| card(s)).collect();
    let river5: Vec<u8> = ["Ks", "9d", "4c", "2h", "Ah"].iter().map(|s| card(s)).collect();

    if mode == "latency" || mode == "both" {
        // live-4 turn is the heaviest (3 active opp × 46 rivers); gate it behind
        // WITH_LIVE4 so a quick live-3 pass doesn't pay for it.
        let with_live4 = std::env::var("WITH_LIVE4").as_deref() == Ok("1");
        let cells: &[(u8, i32, i32)] = if with_live4 {
            &[(3, 20, 60), (4, 20, 80)]
        } else {
            &[(3, 20, 60)]
        };
        eprintln!("FACTORED MULTIWAY RE-SOLVE LATENCY (full M2 menu, {iters} iters)");
        eprintln!("{:<8} {:<6} {:>5} {:>4} {:>12} {:>12}", "street", "live", "nh", "na", "total ms", "ms/iter");
        for &(np, commit, pot) in cells {
            for (label, board) in [("river", river5.as_slice()), ("turn", turn4.as_slice())] {
                let (_s, na, nh, ms) = solve(board, np, commit, pot, live2_bet_menu(), true, iters);
                eprintln!(
                    "{:<8} {:<6} {:>5} {:>4} {:>12.1} {:>12.2}",
                    label, format!("live-{np}"), nh, na, ms, ms / iters as f64
                );
            }
        }
        eprintln!();
    }

    if mode == "fidelity" || mode == "both" {
        // Small menu: single pot-sized bet, no raises ⇒ few terminals ⇒ exact
        // brute force is tractable for a one-off reference.
        let small = BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        };
        let fi: u32 = std::env::var("FID_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(48);
        println!("FIDELITY: live-3 river, small menu (bet pot / check), {fi} iters — exact vs factored");
        let (s_exact, na, nh, ms_e) = solve(&river5, 3, 20, 60, small.clone(), false, fi);
        let (s_fact, _, _, ms_f) = solve(&river5, 3, 20, 60, small, true, fi);
        let mut l1 = 0.0f64;
        for a in 0..na {
            for h in 0..nh {
                l1 += (s_exact[a][h] - s_fact[a][h]).abs() as f64;
            }
        }
        let mean_l1 = l1 / (na * nh) as f64;
        println!("  na={na} nh={nh}  exact {ms_e:.0} ms   factored {ms_f:.0} ms   ({:.0}× faster)", ms_e / ms_f.max(0.001));
        println!("  root avg-strategy mean |Δ| per (action,hand): {mean_l1:.5}");
        println!("  (mean action-prob disagreement; <0.01 ≈ within solve noise)");
    }
}
