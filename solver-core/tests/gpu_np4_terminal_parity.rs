//! np=4 (K=3 opponents) factored lone-survivor terminal parity: the
//! `vcfr_np4_lone_*` pipeline (table-prep + per-(terminal,hand) mass kernel,
//! ported from the 1e-14-validated factored_mass3 reference) must reproduce the
//! base `vcfr_bottom_up` K>=3 path (per-node factored level-walk, exact) within
//! f32 tolerance on the same np=4 depth-limited tree.
//!
//! METHOD NOTE (learned the hard way): compare at ITERATION 1 on a FRESH solver
//! per traverser. After CFR updates, dominated fold-lines get ~zero reach and
//! their terminals' cfv is LEGITIMATELY zero in both paths — an N-iteration
//! comparison converges to vacuous 0≡0 at exactly the nodes under test. At
//! iteration 1 strategies are uniform ⇒ nonzero reach everywhere ⇒ real values
//! (guarded by the nonzero assertions below).

#![cfg(feature = "metal")]

use solver_core::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::{MetalContext, MetalVectorCfr};
use solver_core::solver::flop_start_game::FlopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree_depth_limited;

fn build() -> (solver_core::tree::flat::FlatTree, FlopChanceTable) {
    let np = 4u8;
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
        num_players: np, initial_state: BoardState::Flop, starting_pot: 40,
        starting_stacks: vec![400; np as usize], initial_contributions: vec![0; np as usize],
        rake_rate: 0.05, rake_cap: 20.0, // exercise the payoff rake path
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
        max_bets_per_street: None, no_open_limp: false, threebet_or_fold: false,
    };
    let tree = build_tree_depth_limited(&cfg).expect("tree");
    (tree, table)
}

#[test]
fn gpu_np4_factored_terminal_matches_base() {
    let np = 4u8;
    let (tree, table) = build();
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
    assert!(!lone.is_empty(), "no lone-survivor terminals in the np=4 tree");
    eprintln!("np=4 tree: {} nodes, {} lone-survivor terminals, nh={nh}", tree.num_nodes(), lone.len());

    // One FRESH solver pair per traverser (uniform strategies each time), so
    // every opponent-index mapping (oi -> player) is exercised on live reach.
    for t in 0..np as u32 {
        let mut slow = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, nc);
        let snap_slow = slow.run_one_iteration_diagnostic(&ctx, &tree, t);

        let mut fast = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, nc);
        fast.set_fast_lone_terminals_ex(&ctx, &lone, true);
        let snap_fast = fast.run_one_iteration_diagnostic(&ctx, &tree, t);

        let mut worst_rel = 0.0f32;
        for &term in &lone {
            let base = term as usize * nh;
            let mut scale = 1e-6f32;
            for h in 0..nh { scale = scale.max(snap_slow.cfv[base + h].abs()); }
            // NONZERO guard: iteration-1 uniform reach ⇒ real fold values. A zero
            // row would make this comparison vacuous (the failure mode that hid
            // an inert kernel during development).
            assert!(scale > 1e-6, "t={t} terminal {term}: base-path cfv all-zero (vacuous)");
            for h in 0..nh {
                let d = (snap_slow.cfv[base + h] - snap_fast.cfv[base + h]).abs();
                worst_rel = worst_rel.max(d / scale);
            }
        }
        // Full-buffer drift (terminals feed the tree above them).
        let mut all_rel = 0.0f32;
        let mut scale = 1e-6f32;
        for v in &snap_slow.cfv { scale = scale.max(v.abs()); }
        for (a, b) in snap_slow.cfv.iter().zip(&snap_fast.cfv) {
            all_rel = all_rel.max((a - b).abs() / scale);
        }
        eprintln!("t={t}: lone-terminal worst_rel={worst_rel:.3e}  full-cfv rel={all_rel:.3e}");
        assert!(worst_rel < 1e-3, "t={t}: np4 factored terminal diverges: rel={worst_rel:.3e}");
        assert!(all_rel < 1e-3, "t={t}: full-tree cfv diverged: rel={all_rel:.3e}");
    }
}

/// Live-4 per-iter timing at FULL production scale (nh≈1176, production seam
/// tree, rich menu) with the np4 fast terminals — the number that decides
/// whether live-4 fits the real-time budget on GPU. (Without the fast path the
/// base K>=3 terminals are O(nh^3)/node single-threaded — minutes/iter.)
#[test]
#[ignore = "full-scale live-4 bench (~seconds). Run on demand."]
fn gpu_np4_full_scale_bench() {
    use solver_core::tree::action::{production_game_v1, BetCap};
    let np = 4u8;
    let board: Vec<Card> = vec![3, 19, 35];
    // Build the table the way the RUNTIME adapter does (build_full_nh_sampled +
    // reduced runout grid) — compute_flop_start at np=4 grinds a combinatorial
    // CPU setup for ~hours (learned the hard way; killed at 31min).
    let canonical = [board[0], board[1], board[2]];
    let (turns, river_decks) = solver_core::blueprint::runout_grid(canonical, 12, 12);
    let turns_u8: Vec<u8> = turns.iter().map(|&c| c as u8).collect();
    let table = FlopChanceTable::build_full_nh_sampled(canonical, np, &turns_u8, &river_decks);
    let nh = table.num_valid;
    let spec = production_game_v1();
    // Mirror search_decision's tree: rich menu + 3-bet cap, live-4 seam.
    let mut cfg = spec.street_seam_config(BoardState::Flop, np, 6, 24,
        BetSizeOptions { bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)], raise: vec![BetSize::PotRelative(1.0)] });
    cfg.max_bets_per_street = BetCap::all(3);
    let tree = build_tree_depth_limited(&cfg).expect("tree");
    let lone: Vec<u32> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_terminal())
        .filter(|&n| { let fm = tree.get_folded_mask(n);
            (0..np).filter(|&p| fm & (1 << p) == 0).count() <= 1 })
        .map(|n| n as u32).collect();
    let n_term_all = (0..tree.num_nodes()).filter(|&n| tree.nodes[n].is_terminal()).count();
    eprintln!("live-4 seam tree: {} nodes, {} terminals ({} lone-survivor), nh={nh}",
        tree.num_nodes(), n_term_all, lone.len());

    let (sos, soi, sps, spi, _) = table.sorted_opp_arrays_base();
    let iw: Vec<Vec<f32>> = (0..np as usize).map(|p| table.initial_weights[p].clone()).collect();
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, table.num_combinations);
    gpu.set_fast_lone_terminals_ex(&ctx, &lone, true);

    gpu.run_batched(&ctx, &tree, 2); // warm (pipeline + first-touch)
    let t0 = std::time::Instant::now();
    let iters = 20u32;
    gpu.run_batched(&ctx, &tree, iters);
    let per_iter_ms = t0.elapsed().as_millis() as f64 / iters as f64;
    eprintln!("live-4 per-iter (np4 fast terminals, no continuation): {per_iter_ms:.1} ms/iter");
    eprintln!("  -> 150 iters = {:.1}s, 300 iters = {:.1}s", per_iter_ms * 150.0 / 1000.0, per_iter_ms * 300.0 / 1000.0);
}
