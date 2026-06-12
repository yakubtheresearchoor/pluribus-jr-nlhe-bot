//! G6 step 0: per-stage GPU-busy attribution for the native iteration
//! (the measure-before-build discipline — same-day compaction lesson).
//! Five tagged stages via per-stage command buffers on one queue:
//! strategies / reach / terminal / cfv+chance / regret. Captured at
//! it2 (dense) and it34 (converged plateau) for the two cells that
//! matter (production B=8 4×4, challenger B=10 4×4).
//!
//! Within-terminal phases (A reduce / B tables / C odometer / E
//! expand) cannot be split by timestamps; the it2→it34 delta isolates
//! C (the decaying part) — the plateau terminal number IS A+B+E+
//! residual-C.
//!
//! ═══ MEASURED 2026-06-11 (quiet box, release) ═══
//!   B=8  4×4 it2 1.032s: terminal 89% | reach 8% | rest ≤3%
//!   B=8  4×4 it34 0.362s: terminal 0.250 (69%) | reach 0.082 (23%)
//!   B=10 4×4 it2 4.659s: terminal 98%
//!   B=10 4×4 it34 0.871s: terminal 0.759 (87%) | reach 0.083 (9%)
//!   READINGS: terminal dominates every regime. Cross-B algebra on
//!   the plateaus (A/B/E is B-independent; 0.250 vs 0.759 differ 3×)
//!   ⇒ A+B+E ≈ 0: the plateau terminal is RESIDUAL PHASE C — lever 3
//!   (parallel phase A) dropped without being built. Reach stage is
//!   flat 0.083s in B and iteration (zero + per-level small grids) —
//!   lever 4's target, material only at the B=8 plateau (23%).
//!   C-levers (function constants, FMA) multiply the dominant slice.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, Card};
use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const NP: u8 = 6;

fn build_m2_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    build_tree(&config).unwrap()
}

fn quantile_maps(
    table: &FlopChanceTable,
    nb: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8]| -> Vec<u16> {
        let alive: Vec<usize> = pl_idx[..nh]
            .iter()
            .map(|&i| i as usize)
            .filter(|&h| !conflicts(h, dead))
            .collect();
        let n = alive.len();
        assert!(n >= nb);
        let mut map = vec![NO_BUCKET; nh];
        for (pos, &h) in alive.iter().enumerate() {
            map[h] = ((pos * nb) / n) as u16;
        }
        map
    };
    let (_, _, _, base_pi, _) = table.sorted_opp_arrays_base();
    let flop_map = map_for(&base_pi, &[]);
    let mut turn_maps = Vec::new();
    let mut river_maps = Vec::new();
    for &tc_card in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
        turn_maps.push(map_for(pi, &[tc_card]));
        let mut rms = Vec::new();
        for &rc_card in &table.river_decks[tc_card as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
            rms.push(map_for(pi, &[tc_card, rc_card]));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
}

fn build_table(n_turn: usize, n_river: usize) -> FlopChanceTable {
    let flop: [Card; 3] = [
        card_from_str("2h").unwrap(),
        card_from_str("7d").unwrap(),
        card_from_str("Ks").unwrap(),
    ];
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let turn_pos: &[usize] = match n_turn {
        1 => &[12],
        2 => &[12, 36],
        4 => &[6, 18, 30, 42],
        _ => unreachable!(),
    };
    let river_pos: &[usize] = match n_river {
        1 => &[10],
        2 => &[10, 30],
        4 => &[8, 20, 32, 44],
        _ => unreachable!(),
    };
    let turn_cards: Vec<u8> = turn_pos.iter().map(|&p| deck[p]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let rdeck: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        river_decks[tc as usize] = river_pos.iter().map(|&p| rdeck[p]).collect();
    }
    FlopChanceTable::build_full_nh_sampled(flop, NP, &turn_cards, &river_decks)
}

fn fmt_stages(b: &[f64; 5], total: f64) -> String {
    BucketedNativeGpu::STAGE_NAMES
        .iter()
        .zip(b.iter())
        .map(|(n, v)| format!("{n} {v:.3}s ({:.0}%)", 100.0 * v / total.max(1e-12)))
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
#[ignore = "G6 stage probe; run with --ignored --nocapture --release"]
fn g6_stage_attribution() {
    eprintln!("\n════ G6 step 0: per-stage busy attribution (quiet box) ════");
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_m2_tree();
    for (nb, nt, nr) in [(8usize, 4usize, 4usize), (10, 4, 4)] {
        let table = build_table(nt, nr);
        let (fm, tm, rm) = quantile_maps(&table, nb);
        let game = FlopStartGame::new(table);
        let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
        let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
        solver.set_terminal_design(TerminalDesign::Design1Collapsed);
        let mut native = BucketedNativeGpu::new(
            &ctx, &tree, game.table(), &bk, &solver, (32 / nb) as u32,
        )
        .expect("native gpu");
        native.set_stage_timing(true);

        native.run(1); // it1 (warm)
        native.reset_stage_busy();
        native.run(1); // it2 — dense
        let dense = native.stage_busy();
        let dense_total: f64 = dense.iter().sum();
        native.run(31); // it3..it33
        native.reset_stage_busy();
        native.run(1); // it34 — converged plateau
        let plateau = native.stage_busy();
        let plateau_total: f64 = plateau.iter().sum();

        eprintln!(
            "B={nb} {nt}×{nr} it2  (busy {dense_total:.3}s): {}",
            fmt_stages(&dense, dense_total)
        );
        eprintln!(
            "B={nb} {nt}×{nr} it34 (busy {plateau_total:.3}s): {}",
            fmt_stages(&plateau, plateau_total)
        );
        eprintln!(
            "B={nb} {nt}×{nr} terminal decay it2→it34: {:.3}s → {:.3}s \
             (decayed share = phase C; plateau terminal = A+B+E+residual)",
            dense[2], plateau[2]
        );
    }
}
