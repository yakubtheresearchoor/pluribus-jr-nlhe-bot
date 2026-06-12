// Phase 0.D: find iteration count where exploitability reaches blueprint
// quality (low single-digit percent of pot, "good enough to anchor search").
//
// The prior 6-9 day projection assumed 200 iters. If blueprint quality is
// reached at 100-120 iters, the baseline is proportionally shorter at zero
// quality cost. This is the cheapest possible time saving and requires no
// code change.
//
// Method: 30 iters at nh=12 with exploitability measured at sparse
// checkpoints {5, 10, 15, 20, 25, 30}. Reports the descent curve so we can
// read where exploitability hits ~1% of pot (typical blueprint floor before
// search refinement). The prior nh=8 game reached <2% at iter 3, but
// nh=8 with simple bet structure has trivial equilibrium; nh=12 should be
// more representative.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::MetalFlopStartSolver;
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::io::Write;
use std::time::Instant;

fn build_6p_asymmetric_table(nh: usize) -> (FlatTree, FlopChanceTable) {
    let board: Vec<Card> = ["2h", "7d", "Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 6u8;
    let num_opp = 5usize;
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / nh;
    let chosen: Vec<u16> = (0..nh).map(|i| all_valid[i * step]).collect();
    let mut hand_cards = vec![0u8; nh * 2];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        hand_cards[i*2] = c1; hand_cards[i*2+1] = c2;
    }
    let mut conflict = vec![0u8; nh*nh];
    for i in 0..nh { for j in 0..nh {
        if i == j { conflict[i*nh+j] = 1; continue; }
        let (a1,a2) = index_to_card_pair(chosen[i] as usize);
        let (b1,b2) = index_to_card_pair(chosen[j] as usize);
        if a1==b1||a1==b2||a2==b1||a2==b2 { conflict[i*nh+j] = 1; }
    }}
    let mut hr = vec![0u16; nh];
    for (i, &hi) in chosen.iter().enumerate() {
        let (c1, c2) = index_to_card_pair(hi as usize);
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        hr[i] = h.evaluate_internal() as u16;
    }
    let tc = vec![card_from_str("3c").unwrap() as u8];
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    rd[tc[0] as usize] = vec![card_from_str("5s").unwrap() as u8];
    let mut turn_ranks = vec![0u16; 52 * nh];
    let mut turn_sorted_str = vec![0u16; 52 * num_opp * nh];
    let mut turn_sorted_idx = vec![0u16; 52 * num_opp * nh];
    for &t in &tc {
        for (i, &hi) in chosen.iter().enumerate() {
            let (c1, c2) = index_to_card_pair(hi as usize);
            let tm = board_mask | (1u64 << t);
            if tm & (1u64 << c1) != 0 || tm & (1u64 << c2) != 0 { continue; }
            let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
            for &bc in &board { h = h.add_card(bc as usize); }
            h = h.add_card(t as usize);
            turn_ranks[t as usize * nh + i] = h.evaluate_internal() as u16;
        }
        let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (turn_ranks[t as usize * nh + h] + 1, h as u16)).collect();
        items.sort_by_key(|&(s, _)| s);
        for oi in 0..num_opp {
            let off = t as usize * num_opp * nh + oi * nh;
            for h in 0..nh { turn_sorted_str[off + h] = items[h].0; turn_sorted_idx[off + h] = items[h].1; }
        }
    }
    let mut river_ranks = vec![0u16; 52 * 52 * nh];
    let mut river_sorted_str = vec![0u16; 52 * 52 * num_opp * nh];
    let mut river_sorted_idx = vec![0u16; 52 * 52 * num_opp * nh];
    for &t in &tc {
        let tm = board_mask | (1u64 << t);
        for &r in &rd[t as usize] {
            let fm = tm | (1u64 << r);
            for (i, &hi) in chosen.iter().enumerate() {
                let (c1, c2) = index_to_card_pair(hi as usize);
                if fm & (1u64 << c1) != 0 || fm & (1u64 << c2) != 0 { continue; }
                let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
                for &bc in &board { h = h.add_card(bc as usize); }
                h = h.add_card(t as usize).add_card(r as usize);
                river_ranks[t as usize * 52 * nh + r as usize * nh + i] = h.evaluate_internal() as u16;
            }
            let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (river_ranks[t as usize * 52 * nh + r as usize * nh + h] + 1, h as u16)).collect();
            items.sort_by_key(|&(s, _)| s);
            for oi in 0..num_opp {
                let off = t as usize * 52 * num_opp * nh + r as usize * num_opp * nh + oi * nh;
                for h in 0..nh { river_sorted_str[off + h] = items[h].0; river_sorted_idx[off + h] = items[h].1; }
            }
        }
    }
    let iw = vec![vec![1.0f32; nh]; num_players as usize];
    fn enum_nc(player: usize, np: usize, nh: usize, combined: u64, hand_cards: &[u8], weight: f64) -> f64 {
        if player == np { return weight; }
        let mut total = 0.0;
        for h in 0..nh {
            let m = (1u64 << hand_cards[h * 2]) | (1u64 << hand_cards[h * 2 + 1]);
            if combined & m != 0 { continue; }
            total += enum_nc(player + 1, np, nh, combined | m, hand_cards, weight);
        }
        total
    }
    let nc = enum_nc(0, num_players as usize, nh, 0, &hand_cards[..], 1.0);
    let table = FlopChanceTable {
        hand_ranks_base: hr, valid_hand_indices: chosen, num_valid: nh, conflict, hand_cards,
        remaining_deck: tc, turn_ranks, turn_sorted_str, turn_sorted_idx,
        river_ranks, river_sorted_str, river_sorted_idx,
        initial_weights: iw, num_players, num_combinations: nc, river_decks: rd,
    };
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&config).unwrap();
    (tree, table)
}

fn measure_exploitability(cpu: &FlopStartVectorCfr, tree: &FlatTree, game: &FlopStartGame, np: usize) -> f32 {
    let mut total = 0.0f32;
    for p in 0..np {
        let br = cpu.best_response_value_debug(tree, game, p as u8);
        let sv = cpu.strategy_value_debug(tree, game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total += (br[h] - sv[h]).max(0.0);
        }
    }
    total
}

fn upload_gpu_to_cpu(cpu: &mut FlopStartVectorCfr, gpu_reg: &[f32], gpu_cum: &[f32]) {
    let fl = cpu.regrets_flop().len();
    let tl = cpu.regrets_turn().len();
    { let r = cpu.regrets_flop_mut(); for i in 0..fl { r[i] = gpu_reg[i]; } }
    { let r = cpu.regrets_turn_mut(); for i in 0..tl { r[i] = gpu_reg[fl + i]; } }
    { let r = cpu.regrets_river_mut(); for i in 0..r.len() {
        if fl + tl + i < gpu_reg.len() { r[i] = gpu_reg[fl + tl + i]; } } }
    { let c = cpu.cum_strategy_flop_mut(); for i in 0..fl { c[i] = gpu_cum[i]; } }
    { let c = cpu.cum_strategy_turn_mut(); for i in 0..tl { c[i] = gpu_cum[fl + i]; } }
    { let c = cpu.cum_strategy_river_mut(); for i in 0..c.len() {
        if fl + tl + i < gpu_cum.len() { c[i] = gpu_cum[fl + tl + i]; } } }
}

#[test]
#[ignore = "Phase 0.D: 30 iter convergence at nh=12 with exploitability; ~2-3 hours"]
fn phase0d_blueprint_quality_nh12() {
    let nh = 12usize;
    let np = 6usize;
    // SPARSE checkpoints to keep CPU best-response cost manageable.
    // Original 7-checkpoint version timed out at 1h48min on broken tree;
    // corrected tree's 3× fewer terminals helps but BR still expensive.
    let checkpoints: &[u32] = &[3, 10, 30];
    let pot_total = 65.0f32;

    eprintln!("\n=== Phase 0.D: convergence curve at nh={} to find blueprint-quality iter ===\n", nh);
    let (tree, table) = build_6p_asymmetric_table(nh);
    let game = FlopStartGame::new(table);
    let mut cpu_proxy = FlopStartVectorCfr::new(&tree, &game.table());
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu_proxy);
    eprintln!("Tree: {} nodes; pot: {}", tree.num_nodes(), pot_total);
    std::io::stderr().flush().ok();

    let mut prev = 0u32;
    let t0 = Instant::now();
    let mut trajectory: Vec<(u32, f32, f64)> = Vec::new();
    eprintln!();
    eprintln!("{:>5} | {:>14} | {:>12} | {:>14}", "iter", "expl (% pot)", "batch (min)", "total (h)");
    eprintln!("{}", "-".repeat(60));

    for &cp in checkpoints {
        let n_to_run = cp - prev;
        let it0 = Instant::now();
        gpu.run(&ctx, &tree, &game, n_to_run);
        let batch_min = it0.elapsed().as_secs_f64() / 60.0;
        prev = cp;

        cpu_proxy.set_iteration(gpu.iteration());
        let gpu_reg = gpu.download_regrets();
        let gpu_cum = gpu.download_cum_strategy();
        upload_gpu_to_cpu(&mut cpu_proxy, &gpu_reg, &gpu_cum);
        let t_expl = Instant::now();
        let expl = measure_exploitability(&cpu_proxy, &tree, &game, np);
        let _expl_s = t_expl.elapsed().as_secs_f64();
        let pct = expl / pot_total * 100.0;
        trajectory.push((cp, pct, batch_min));
        eprintln!("{:>5} | {:>13.4}% | {:>12.2} | {:>14.2}",
            cp, pct, batch_min, t0.elapsed().as_secs_f64() / 3600.0);
        std::io::stderr().flush().ok();
    }

    eprintln!();
    eprintln!("=== Blueprint-quality reading ===");
    // "Blueprint quality" thresholds — values commonly cited as good-enough
    // anchors before search refinement
    let thresholds = [10.0f32, 5.0, 2.0, 1.0, 0.5, 0.1];
    for thr in &thresholds {
        let first_hit = trajectory.iter().find(|&&(_, pct, _)| pct <= *thr);
        match first_hit {
            Some(&(iter, pct, _)) => {
                eprintln!("  exploitability <= {:.1}% reached at iter {} ({:.4}%)", thr, iter, pct);
            }
            None => {
                eprintln!("  exploitability <= {:.1}% NOT reached in 30 iters", thr);
            }
        }
    }
    eprintln!();
    eprintln!("Compare to assumed 200 iters: if blueprint quality is reached at iter N,");
    eprintln!("baseline is (N/200)x the projected runtime at zero quality cost.");
    eprintln!("For example: blueprint at iter 100 → 50% baseline; at iter 50 → 25% baseline.");
}
