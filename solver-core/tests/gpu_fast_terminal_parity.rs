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

fn build() -> (solver_core::tree::flat::FlatTree, FlopChanceTable) {
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
        rake_rate: 0.0, rake_cap: 0.0,
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
    let np = 3u8;
    let (tree, table) = build();
    let nh = table.num_valid;
    let (sos, soi, sps, spi, _) = table.sorted_opp_arrays_base();
    let iw: Vec<Vec<f32>> = (0..np as usize).map(|p| table.initial_weights[p].clone()).collect();
    let nc = table.num_combinations;
    let ctx = MetalContext::new().expect("Metal");

    // lone-survivor terminals (num_active <= 1)
    let lone: Vec<u32> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_terminal())
        .filter(|&n| { let fm = tree.get_folded_mask(n);
            (0..np).filter(|&p| fm & (1 << p) == 0).count() <= 1 })
        .map(|n| n as u32).collect();
    assert!(!lone.is_empty(), "no lone-survivor terminals");

    // slow path (fast terminals OFF)
    let mut slow = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, nc);
    slow.run_batched(&ctx, &tree, 1);
    let cfv_slow = slow.cfv_slice();

    // fast path (fast terminals ON)
    let mut fast = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, nc);
    fast.set_fast_lone_terminals(&ctx, &lone);
    fast.run_batched(&ctx, &tree, 1);
    let cfv_fast = fast.cfv_slice();

    // The fast kernel runs the IDENTICAL g0×g1 loop ⇒ bit-exact at the lone
    // terminals; the rest of the tree is untouched ⇒ identical everywhere.
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0u32;
    for (a, b) in cfv_slow.iter().zip(&cfv_fast) {
        max_abs = max_abs.max((a - b).abs());
        let ua = a.to_bits() as i64; let ub = b.to_bits() as i64;
        max_ulp = max_ulp.max((ua - ub).unsigned_abs() as u32);
    }
    eprintln!("fast-terminal parity: {} lone terminals, nh={nh}, max_abs={max_abs:.3e}, max_ulp={max_ulp}", lone.len());
    assert!(max_abs == 0.0, "fast terminal NOT bit-exact to slow path: max_abs={max_abs}, max_ulp={max_ulp}");

    // FACTORED kernel: O(nh) inclusion-exclusion — same value within f32 tolerance.
    let mut fac = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, nc);
    fac.set_fast_lone_terminals_ex(&ctx, &lone, true);
    fac.run_batched(&ctx, &tree, 1);
    let cfv_fac = fac.cfv_slice();
    let mut f_abs = 0.0f32; let mut scale = 1e-6f32;
    for (a, b) in cfv_slow.iter().zip(&cfv_fac) {
        f_abs = f_abs.max((a - b).abs());
        scale = scale.max(a.abs());
    }
    eprintln!("factored parity: max_abs={f_abs:.3e}, rel={:.3e}", f_abs / scale);
    assert!(f_abs / scale < 1e-4, "factored terminal diverges: max_abs={f_abs}, scale={scale}");
}
