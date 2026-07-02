//! np=5 (K=4 opponents) GPU lone-terminal PARITY — vs the validated Rust f64
//! reference (common/closed_mass.rs; math gates k4_mass4_* at 1.47e-14 vs
//! brute). There is NO GPU base path at K=4 (that O(nh^K) wall is why this
//! kernel exists), so parity reads back reach_after_topdown at ITERATION 1
//! (fresh solver per traverser — the vacuous-0≡0 trap fix) and compares the
//! kernel's terminal cfv against payoff·mass4(h)/nc computed in Rust.
#![cfg(feature = "metal")]
#[path = "common/closed_mass.rs"]
mod cf;

use solver_core::gpu_metal::{MetalContext, MetalVectorCfr};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::build_tree_depth_limited;

#[test]
fn gpu_np5_terminal_matches_rust_reference() {
    let np = 5u8;
    let board: Vec<u8> = vec![3, 19, 35];
    let canonical = [board[0], board[1], board[2]];
    let (turns, river_decks) = solver_core::blueprint::runout_grid(canonical, 12, 12);
    let turns_u8: Vec<u8> = turns.iter().map(|&c| c as u8).collect();
    let table = FlopChanceTable::build_full_nh_sampled(canonical, np, &turns_u8, &river_decks);
    let nh = table.num_valid;
    let hands: Vec<(u8, u8)> = (0..nh)
        .map(|h| (table.hand_cards[h * 2], table.hand_cards[h * 2 + 1]))
        .collect();

    let spec = production_game_v1();
    let mut cfg = spec.street_seam_config(BoardState::Flop, np, 6, 30,
        BetSizeOptions { bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)], raise: vec![BetSize::PotRelative(1.0)] });
    cfg.max_bets_per_street = BetCap::all(3);
    let tree = build_tree_depth_limited(&cfg).expect("tree");
    let lone: Vec<u32> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_terminal())
        .filter(|&n| { let fm = tree.get_folded_mask(n);
            (0..np).filter(|&p| fm & (1 << p) == 0).count() <= 1 })
        .map(|n| n as u32).collect();
    eprintln!("np5 tree: {} nodes, {} lone terminals, nh={nh}", tree.num_nodes(), lone.len());

    let ctx = MetalContext::new().expect("Metal");
    let (sos, soi, sps, spi, _) = table.sorted_opp_arrays_base();
    let iw: Vec<Vec<f32>> = (0..np as usize).map(|p| table.initial_weights[p].clone()).collect();
    let game = FlopStartGame::new(table);
    let nc = game.table().num_combinations;

    let mut worst = 0.0f64;
    for t in 0..np as u32 {
        let mut gpu = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &game.table().hand_cards, nc);
        gpu.set_fast_lone_terminals_ex(&ctx, &lone, true);
        gpu.set_np5_mc_samples(0); // FULL outer enumeration = deterministic parity
        let snap = gpu.run_one_iteration_diagnostic(&ctx, &tree, t);

        // subset of terminals × subset of hands (the Rust f64 reference is slow)
        for (li, &term) in lone.iter().enumerate().step_by(7) {
            let node = term as usize;
            // opponent reach rows in seat order (kernel np4_opp_reach semantics)
            let opps: Vec<usize> = (0..np as usize).filter(|&p| p != t as usize).collect();
            let row = |p: usize| -> Vec<f64> {
                let base = (node * np as usize + p) * nh;
                snap.reach_after_topdown[base..base + nh].iter().map(|&v| v as f64).collect()
            };
            let r0 = row(opps[0]);
            let roles = cf::roles(&hands, &row(opps[1]), &row(opps[2]), &row(opps[3]));
            let agg = cf::build(&roles, &hands);

            // payoff mirror (byte-identical conventions to the kernel)
            let fm = tree.get_folded_mask(node);
            let c_t = tree.contributions[node * np as usize + t as usize];
            let stake = tree.starting_pot as f64 / np as f64 + c_t as f64;
            let folded = fm & (1 << t) != 0;
            let flop_seen = tree.nodes[node].board_state != 3;
            let (rr, rc2) = if flop_seen { (tree.rake_rate, tree.rake_cap) } else { (0.0, 0.0) };
            let total_pot: i32 = tree.starting_pot + (0..np as usize).map(|q| tree.contributions[node * np as usize + q]).sum::<i32>();
            let min_c = (0..np as usize).map(|q| tree.contributions[node * np as usize + q]).min().unwrap();
            let n_main = (0..np as usize).filter(|&q| tree.contributions[node * np as usize + q] >= min_c).count() as i32;
            let main_pot = min_c * n_main + tree.starting_pot;
            let rake = (main_pot as f64 * rr).min(rc2).max(0.0);
            let payoff = if folded { -stake } else { total_pot as f64 - rake - stake };

            let mut scale = 1e-30f64;
            let mut hs: Vec<usize> = (li % 61..nh).step_by(97).collect();
            hs.truncate(12);
            let refs: Vec<f64> = hs.iter().map(|&h| {
                payoff * cf::mass4_closed_inner(&hands, h, &r0, &roles, &agg) / nc as f64
            }).collect();
            for &r in &refs { scale = scale.max(r.abs()); }
            assert!(scale > 1e-9, "t={t} term {node}: reference all-zero (vacuous)");
            for (k, &h) in hs.iter().enumerate() {
                let g = snap.cfv[node * nh + h] as f64;
                let rel = (g - refs[k]).abs() / scale;
                worst = worst.max(rel);
                assert!(rel < 2e-3, "t={t} term {node} h={h}: gpu {g} vs ref {}", refs[k]);
            }
        }
    }
    eprintln!("np5 GPU vs Rust-reference worst rel = {worst:.3e}");
}
