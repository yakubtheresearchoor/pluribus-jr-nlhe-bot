//! GPU depth-limited search vs CPU search: `gpu_search_flop_strat` (the runtime
//! entry point = MetalVectorCfr + bucketed continuation + fast terminals) must
//! converge to ~the same flop strategy as the CPU `BucketedContinuationGame` +
//! `CpuMccfr` path on identical inputs (same tree, bucketing, reach prior).
//!
//! Not bit-exact (the GPU CFR loop uses its built-in DCFR schedule, beta=0.5,
//! vs the CPU's set_dcfr(1.5,0,2.0); HU continuation is exact on both). After
//! enough iters both sit near the same equilibrium ⇒ small mean-L1.

#![cfg(feature = "metal")]

use solver_core::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::{gpu_search_flop_strat, GpuSearchCfg, MetalContext};
use solver_core::solver::bucketed_flop_cfr::FlopBucketing;
use solver_core::solver::bucketed_search::{BucketedContinuationGame, ContStreet};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree_depth_limited;

#[test]
fn gpu_flop_search_matches_cpu_search() {
    let np = 2u8;
    let board: Vec<Card> = vec![3, 19, 35];
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let valid: Vec<u16> = (0..NUM_POSSIBLE_HANDS)
        .filter(|&hi| { let (c1, c2) = index_to_card_pair(hi);
            board_mask & (1u64 << c1) == 0 && board_mask & (1u64 << c2) == 0 })
        .map(|hi| hi as u16).collect();
    let nh_target = 24usize;
    let step = valid.len() / nh_target;
    let hands: Vec<u16> = valid.iter().step_by(step).copied().take(nh_target).collect();
    let nbc: Vec<u8> = (0..52u8).filter(|&c| board_mask & (1u64 << c) == 0).collect();
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    rd[nbc[0] as usize] = vec![nbc[1]];
    let ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![1.0f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS]).collect();
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(&board, &ranges, np, &hands, &[nbc[0]], &rd);
    let nh = table.num_valid;
    let bk = FlopBucketing::identity(&table); // nb = nh: exact per-hand continuation

    let cfg = TreeConfig {
        num_players: np, initial_state: BoardState::Flop, starting_pot: 20,
        starting_stacks: vec![100; np as usize], initial_contributions: vec![0; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(0.75)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
        max_bets_per_street: None, no_open_limp: false, threebet_or_fold: false,
    };
    let tree = build_tree_depth_limited(&cfg).expect("tree");
    let leaf_nodes: Vec<usize> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_chance() && tree.node_children(n).is_empty())
        .collect();

    let fsg = FlopStartGame::new(table);
    let reach: Vec<Vec<f32>> = (0..np as usize).map(|p| fsg.table().initial_weights[p].clone()).collect();
    let iters = 800u32;

    let ctx = MetalContext::new().expect("Metal");
    let flop = BoardState::Flop as u8;

    let compare = |cpu: &CpuMccfr, gpu_strat: &std::collections::HashMap<usize, Vec<Vec<f32>>>| -> (f64, f32) {
        let mut total = 0.0f64; let mut count = 0usize; let mut worst = 0.0f32;
        for n in 0..tree.num_nodes() {
            if !(tree.nodes[n].is_player() && tree.nodes[n].board_state == flop) { continue; }
            let na = tree.nodes[n].num_children as usize;
            let cs = cpu.get_average_strategy(n, na, nh);
            let gs = gpu_strat.get(&n).expect("gpu node strategy");
            let mut l1 = 0.0f32;
            for a in 0..na { for h in 0..nh { l1 += (cs[a][h] - gs[a][h]).abs(); } }
            l1 /= (na * nh) as f32;
            worst = worst.max(l1); total += l1 as f64; count += 1;
        }
        (total / count as f64, worst)
    };

    // ── DCFR (no λ): both converge to ~Nash; schedule differs (loose). ──
    let cpu_game = BucketedContinuationGame::new_street(&fsg, &bk, ContStreet::Flop, 0, 7);
    let mut cpu = CpuMccfr::new(&tree, vec![nh; np as usize]);
    cpu.set_depth_limit(&leaf_nodes);
    cpu.set_dcfr(1.5, 0.0, 2.0);
    cpu.run(&tree, &cpu_game, iters);
    let gcfg = GpuSearchCfg { iters, sample_m: 0, seed: 7, factored_terminals: false, lambda: 0.0 , budget_ms: 120_000 };
    let gpu_strat = gpu_search_flop_strat(&ctx, &tree, fsg.table(), &bk, &reach, &gcfg);
    let (mean_l1, worst) = compare(&cpu, &gpu_strat);
    eprintln!("DCFR: GPU vs CPU mean_L1={mean_l1:.4}, worst={worst:.4}");
    assert!(mean_l1 < 0.05, "GPU DCFR search diverges: mean_L1={mean_l1}");

    // ── QRE (λ>0): the GPU logit-strategy must match the CPU QRE closely. ──
    let lambda = 5.0f32;
    let cpu_game2 = BucketedContinuationGame::new_street(&fsg, &bk, ContStreet::Flop, 0, 7);
    let mut cpu_q = CpuMccfr::new(&tree, vec![nh; np as usize]);
    cpu_q.set_depth_limit(&leaf_nodes);
    cpu_q.set_lambda(vec![lambda; np as usize]);
    cpu_q.run(&tree, &cpu_game2, iters);
    let gcfg_q = GpuSearchCfg { iters, sample_m: 0, seed: 7, factored_terminals: false, lambda , budget_ms: 120_000 };
    let gpu_q = gpu_search_flop_strat(&ctx, &tree, fsg.table(), &bk, &reach, &gcfg_q);
    let (q_l1, q_worst) = compare(&cpu_q, &gpu_q);
    eprintln!("QRE λ={lambda}: GPU vs CPU mean_L1={q_l1:.4}, worst={q_worst:.4}");
    // Same QRE rule on both ⇒ should match tightly (continuation exact for HU,
    // only MC/float order differ). A broken QRE port gives ~0.3 (uniform/Nash).
    assert!(q_l1 < 0.03, "GPU QRE search diverges from CPU QRE: mean_L1={q_l1}");

    // ── REAL quantile bucketing (nb<nh) + QRE, sample_m=0 (both exact). This is
    // the live config MINUS the sampled continuation — isolates whether the GPU
    // consumes the real nb-bucket flop_map/flop_tables correctly (vs my synthetic
    // nb=8 continuation test). If this matches, the live HU divergence is the
    // CPU's sampled vs the GPU's exact continuation, not a bucketing bug. ──
    let qbk = FlopBucketing::quantile(fsg.table(), 16);
    let cpu_game3 = BucketedContinuationGame::new_street(&fsg, &qbk, ContStreet::Flop, 0, 7);
    let mut cpu_qb = CpuMccfr::new(&tree, vec![nh; np as usize]);
    cpu_qb.set_depth_limit(&leaf_nodes);
    cpu_qb.set_lambda(vec![lambda; np as usize]);
    cpu_qb.run(&tree, &cpu_game3, iters);
    let gcfg_qb = GpuSearchCfg { iters, sample_m: 0, seed: 7, factored_terminals: false, lambda , budget_ms: 120_000 };
    let gpu_qb = gpu_search_flop_strat(&ctx, &tree, fsg.table(), &qbk, &reach, &gcfg_qb);
    let (qb_l1, qb_worst) = compare(&cpu_qb, &gpu_qb);
    eprintln!("QRE λ={lambda} + quantile nb=16: GPU vs CPU mean_L1={qb_l1:.4}, worst={qb_worst:.4}");
    assert!(qb_l1 < 0.04, "GPU vs CPU QRE+quantile diverges (real-bucketing bug?): mean_L1={qb_l1}");
}

/// TURN GPU search vs CPU: the generalized `gpu_search_street_strat` with
/// ContStreet::Turn(ti) must match the CPU turn search (BucketedContinuationGame
/// rooted on the turn, continuation = the river runout for that turn). Validates
/// the #11 turn generalization (bk.turn_tables/turn_map selection).
#[test]
fn gpu_turn_search_matches_cpu() {
    use solver_core::gpu_metal::gpu_search_street_strat;
    let np = 2u8;
    let board: Vec<Card> = vec![3, 19, 35];
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let valid: Vec<u16> = (0..NUM_POSSIBLE_HANDS)
        .filter(|&hi| { let (c1, c2) = index_to_card_pair(hi);
            board_mask & (1u64 << c1) == 0 && board_mask & (1u64 << c2) == 0 })
        .map(|hi| hi as u16).collect();
    let nh_target = 24usize;
    let step = valid.len() / nh_target;
    let hands: Vec<u16> = valid.iter().step_by(step).copied().take(nh_target).collect();
    let nbc: Vec<u8> = (0..52u8).filter(|&c| board_mask & (1u64 << c) == 0).collect();
    let turn = nbc[0];
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    // a couple rivers for the turn's runout tables
    rd[turn as usize] = vec![nbc[1], nbc[2]];
    let ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![1.0f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS]).collect();
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(&board, &ranges, np, &hands, &[turn], &rd);
    let nh = table.num_valid;
    let bk = FlopBucketing::identity(&table);

    // TURN-rooted depth-limited tree (truncates at the river deal).
    let cfg = TreeConfig {
        num_players: np, initial_state: BoardState::Turn, starting_pot: 20,
        starting_stacks: vec![100; np as usize], initial_contributions: vec![0; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(0.75)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
        max_bets_per_street: None, no_open_limp: false, threebet_or_fold: false,
    };
    let tree = build_tree_depth_limited(&cfg).expect("turn tree");
    let leaf_nodes: Vec<usize> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_chance() && tree.node_children(n).is_empty())
        .collect();
    assert!(!leaf_nodes.is_empty(), "turn tree should have river-deal continuation leaves");

    let fsg = FlopStartGame::new(table);
    let reach: Vec<Vec<f32>> = (0..np as usize).map(|p| fsg.table().initial_weights[p].clone()).collect();
    let iters = 800u32;
    let lambda = 5.0f32;

    let cpu_game = BucketedContinuationGame::new_street(&fsg, &bk, ContStreet::Turn(0), 0, 7);
    let mut cpu = CpuMccfr::new(&tree, vec![nh; np as usize]);
    cpu.set_depth_limit(&leaf_nodes);
    cpu.set_lambda(vec![lambda; np as usize]);
    cpu.run(&tree, &cpu_game, iters);

    let ctx = MetalContext::new().expect("Metal");
    let gcfg = GpuSearchCfg { iters, sample_m: 0, seed: 7, factored_terminals: false, lambda , budget_ms: 120_000 };
    let gpu = gpu_search_street_strat(&ctx, &tree, fsg.table(), &bk, ContStreet::Turn(0), &reach, &gcfg);

    let turn = BoardState::Turn as u8;
    let mut total = 0.0f64; let mut count = 0usize; let mut worst = 0.0f32;
    for n in 0..tree.num_nodes() {
        if !(tree.nodes[n].is_player() && tree.nodes[n].board_state == turn) { continue; }
        let na = tree.nodes[n].num_children as usize;
        let cs = cpu.get_average_strategy(n, na, nh);
        let gs = gpu.get(&n).expect("gpu turn node strategy");
        let mut l1 = 0.0f32;
        for a in 0..na { for h in 0..nh { l1 += (cs[a][h] - gs[a][h]).abs(); } }
        l1 /= (na * nh) as f32;
        worst = worst.max(l1); total += l1 as f64; count += 1;
    }
    let mean_l1 = total / count as f64;
    eprintln!("TURN QRE λ={lambda}: GPU vs CPU mean_L1={mean_l1:.4}, worst={worst:.4}, nodes={count}");
    assert!(count > 0, "no turn player nodes found");
    assert!(mean_l1 < 0.03, "GPU turn search diverges from CPU: mean_L1={mean_l1}");
}
