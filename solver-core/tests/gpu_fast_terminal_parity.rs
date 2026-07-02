//! GPU fast lone-survivor terminal parity (np=3): `vcfr_lone_terminal_par`
//! (parallel over hands) must reproduce the base `vcfr_bottom_up` terminal
//! showdown BIT-FOR-BIT (it runs the identical g0×g1 loop in the same order).
//! We solve the same np=3 depth-limited tree with the fast path OFF and ON and
//! compare the full cfv slice after one batched pass.

#![cfg(feature = "metal")]

use solver_core::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::{MetalContext, MetalVectorCfr};
use solver_core::solver::flop_start_game::FlopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree_depth_limited;

fn build_with_rake(rake_rate: f64, rake_cap: f64) -> (solver_core::tree::flat::FlatTree, FlopChanceTable) {
    let np = 3u8;
    let board: Vec<Card> = vec![3, 19, 35];
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let valid: Vec<u16> = (0..NUM_POSSIBLE_HANDS)
        .filter(|&hi| { let (c1, c2) = index_to_card_pair(hi);
            board_mask & (1u64 << c1) == 0 && board_mask & (1u64 << c2) == 0 })
        .map(|hi| hi as u16).collect();
    let nh_target = 60usize;
    let step = valid.len() / nh_target;
    let hands: Vec<u16> = valid.iter().step_by(step).copied().take(nh_target).collect();
    let nbc: Vec<u8> = (0..52u8).filter(|&c| board_mask & (1u64 << c) == 0).collect();
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    rd[nbc[0] as usize] = vec![nbc[1]];
    let ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![1.0f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS]).collect();
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(&board, &ranges, np, &hands, &[nbc[0]], &rd);
    let cfg = TreeConfig {
        num_players: np, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![400; np as usize], initial_contributions: vec![0; np as usize],
        rake_rate, rake_cap,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
        max_bets_per_street: None, no_open_limp: false, threebet_or_fold: false,
    };
    let tree = build_tree_depth_limited(&cfg).expect("tree");
    (tree, table)
}

#[test]
fn gpu_fast_lone_terminal_bit_exact() {
    run_parity(0.0, 0.0, true);
}

/// Rake-ON variant: production trees carry rake, and the fast-terminal params
/// previously HARDCODED rake 0.0 (silently un-raked fold pots) while every
/// parity tree used rake 0 — invisible. This variant keeps the rake path live.
#[test]
fn gpu_fast_lone_terminal_parity_with_rake() {
    run_parity(0.05, 20.0, false); // brute uses one final multiply; base path per-branch — not bit-exact under rake path differences, tolerance-gated
}

/// METHOD NOTE: compare at ITERATION 1 on FRESH solvers per traverser — after
/// CFR updates, dominated fold-lines get zero reach and fold-terminal cfv is
/// legitimately zero in both paths, making an N-iteration comparison VACUOUS
/// (0==0) at exactly the nodes under test. Nonzero scale is asserted.
fn run_parity(rake_rate: f64, rake_cap: f64, expect_bit_exact: bool) {
    let np = 3u8;
    let (tree, table) = build_with_rake(rake_rate, rake_cap);
    let nh = table.num_valid;
    let (sos, soi, sps, spi, _) = table.sorted_opp_arrays_base();
    let iw: Vec<Vec<f32>> = (0..np as usize).map(|p| table.initial_weights[p].clone()).collect();
    let nc = table.num_combinations;
    let ctx = MetalContext::new().expect("Metal");

    let lone: Vec<u32> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_terminal())
        .filter(|&n| { let fm = tree.get_folded_mask(n);
            (0..np).filter(|&p| fm & (1 << p) == 0).count() <= 1 })
        .map(|n| n as u32).collect();
    assert!(!lone.is_empty(), "no lone-survivor terminals");

    for t in 0..np as u32 {
        let mut slow = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, nc);
        let snap_slow = slow.run_one_iteration_diagnostic(&ctx, &tree, t);
        let mut fast = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, nc);
        fast.set_fast_lone_terminals(&ctx, &lone);
        let snap_fast = fast.run_one_iteration_diagnostic(&ctx, &tree, t);
        let mut fac = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, nc);
        fac.set_fast_lone_terminals_ex(&ctx, &lone, true);
        let snap_fac = fac.run_one_iteration_diagnostic(&ctx, &tree, t);

        let mut max_abs = 0.0f32;
        let mut f_rel = 0.0f32;
        for &term in &lone {
            let base = term as usize * nh;
            let mut scale = 1e-6f32;
            for h in 0..nh { scale = scale.max(snap_slow.cfv[base + h].abs()); }
            assert!(scale > 1e-6, "t={t} terminal {term}: base cfv all-zero (VACUOUS parity)");
            for h in 0..nh {
                max_abs = max_abs.max((snap_slow.cfv[base + h] - snap_fast.cfv[base + h]).abs());
                f_rel = f_rel.max((snap_slow.cfv[base + h] - snap_fac.cfv[base + h]).abs() / scale);
            }
        }
        eprintln!("t={t} rake={rake_rate}: brute max_abs={max_abs:.3e}  factored rel={f_rel:.3e}");
        if expect_bit_exact {
            assert!(max_abs == 0.0, "t={t}: brute fast terminal NOT bit-exact: {max_abs}");
        } else {
            let mut scale = 1e-6f32;
            for v in &snap_slow.cfv { scale = scale.max(v.abs()); }
            assert!(max_abs / scale < 1e-4, "t={t}: brute fast terminal diverges under rake: {max_abs}");
        }
        assert!(f_rel < 1e-3, "t={t}: factored terminal diverges: rel={f_rel:.3e}");
    }
}
