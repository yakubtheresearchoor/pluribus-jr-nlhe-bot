#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::{ChanceGpuData, GpuContext};
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::game::GameSpec;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;

use postflop_solver::*;
use postflop_solver::{
    BetSizeOptions as ExtBet, TreeConfig as ExtTreeConfig,
    BoardState as ExtBoard, ActionTree, CardConfig,
    flop_from_str, card_from_str as ext_card, solve, solve_step, NOT_DEALT,
};

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }
fn full_range() -> &'static str {
    "22+,A2s+,A2o+,K2s+,K2o+,Q2s+,Q2o+,J2s+,J2o+,T2s+,T2o+,92s+,92o+,82s+,82o+,72s+,72o+,62s+,62o+,52s+,52o+,42s+,42o+,32s,32o"
}
fn bets(s: &str) -> Vec<BetSize> {
    s.split(',').map(|x| BetSize::PotRelative(x.trim_end_matches('%').parse().unwrap())).collect()
}
fn offsets(tree: &solver_core::tree::flat::FlatTree, nh: usize) -> Vec<usize> {
    (0..tree.num_nodes()).map(|i| {
        let o = tree.infoset_offsets[i]; if o == u32::MAX { usize::MAX } else { o as usize * MAX_NA * nh }
    }).collect()
}
fn chance_probs(table: &ChanceTable) -> Vec<f32> {
    let nh = table.num_valid;
    let nd = table.remaining_deck.len();
    let mut p = vec![0.0f32; nd * nh];
    for o in 0..nd {
        let c = table.remaining_deck[o];
        for h in 0..nh {
            let (c1,c2) = index_to_card_pair(table.valid_hand_indices[h] as usize);
            if c == c1 || c == c2 { continue; }
            let bl = table.remaining_deck.iter().filter(|&&r| r == c1 || r == c2).count();
            p[o * nh + h] = 1.0 / (nd as f32 - bl as f32);
        }
    }
    p
}

// ─── River 50% pot ────────────────────────────────────────────────────────

#[test]
fn sc1_river_halfpot() {
    let board: Vec<Card> = ["2h","7d","Ks","4c","Qs"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let tree = build_tree(&TreeConfig {
        num_players: 2, initial_state: BoardState::River,
        starting_pot: 200, starting_stacks: vec![9500,9500],
        initial_contributions: vec![0,0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: bets("50%"), raise: vec![] },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    }).unwrap();
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_hands(0);
    let off = offsets(&tree, nh);

    println!("\n=== SC1: River 50%pot ===");
    println!("Tree: {} nodes, nh={}", tree.num_nodes(), nh);

    // VCFR CPU
    let mut vcfr = VectorCfr::new(&tree, vec![nh, nh]);
    let t0 = std::time::Instant::now();
    let mut vi = 0u32;
    loop { vcfr.run_sequential(&tree, &game, 10); vi += 10; if t0.elapsed().as_secs_f64() >= 5.0 { break; } }
    let vt = t0.elapsed().as_secs_f64();
    let p = StrategyProfile::from_usize_offsets(vcfr.cum_strategy_slice(), &off, nh);
    let ve = exploitability(&tree, &game, &p);
    println!("VCFR: {} iters {:.1}s, exp={:.6} ({:.3}%pot)", vi, vt, ve, ve/2.0);

    // External
    let eb = ExtBet::try_from(("50%","")).unwrap();
    let mut eg = PostFlopGame::with_config(CardConfig {
        range: [full_range().parse().unwrap(), full_range().parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(), turn: ext_card("Qs").unwrap(), river: ext_card("4c").unwrap(),
    }, ActionTree::new(ExtTreeConfig {
        initial_state: ExtBoard::River, starting_pot: 200, effective_stack: 9500,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [eb.clone(),eb.clone()], turn_bet_sizes: [eb.clone(),eb.clone()],
        river_bet_sizes: [eb.clone(),eb], turn_donk_sizes: None, river_donk_sizes: None,
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    }).unwrap()).unwrap();
    eg.allocate_memory(false);
    let t1 = std::time::Instant::now();
    let mut ei = 0u32;
    loop { solve_step(&eg, ei); ei += 1; if t1.elapsed().as_secs_f64() >= 5.0 { break; } }
    let et = t1.elapsed().as_secs_f64();
    let ee = { let mut g = PostFlopGame::with_config(CardConfig {
        range: [full_range().parse().unwrap(), full_range().parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(), turn: ext_card("Qs").unwrap(), river: ext_card("4c").unwrap(),
    }, ActionTree::new(ExtTreeConfig {
        initial_state: ExtBoard::River, starting_pot: 200, effective_stack: 9500,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [ExtBet::try_from(("50%","")).unwrap(),ExtBet::try_from(("50%","")).unwrap()],
        turn_bet_sizes: [ExtBet::try_from(("50%","")).unwrap(),ExtBet::try_from(("50%","")).unwrap()],
        river_bet_sizes: [ExtBet::try_from(("50%","")).unwrap(),ExtBet::try_from(("50%","")).unwrap()],
        turn_donk_sizes: None, river_donk_sizes: None,
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    }).unwrap()).unwrap(); g.allocate_memory(false); solve(&mut g, ei, 0.0, false) };
    println!("Ext:  {} iters {:.1}s, exp={:.6} ({:.3}%pot)", ei, et, ee, ee/2.0);
    println!("VCFR/Ext ratio: {:.2}x, throughput: {:.1}x", ve/ee.max(0.001), (vi as f64/vt)/(ei as f64/et));
    assert!(ve < ee * 5.0 + 1.0, "VCFR too far: {:.4} vs {:.4}", ve, ee);
}

// ─── T2D turn: production config ──────────────────────────────────────────

#[test]
fn sc1_t2d_production() {
    let board: Vec<Card> = ["2h","7d","Ks","4c"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let tree = build_tree(&TreeConfig {
        num_players: 2, initial_state: BoardState::Turn,
        starting_pot: 200, starting_stacks: vec![9500,9500],
        initial_contributions: vec![0,0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: bets("50%,100%"), raise: bets("50%") },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    }).unwrap();
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid;
    let game = TurnStartGame::new(ChanceTable::compute_turn_start(&board, &ranges, 2));
    let off = offsets(&tree, nh);
    let cp = chance_probs(&table);

    let nt = tree.nodes.iter().filter(|n| n.is_terminal()).count();
    let nc = tree.nodes.iter().filter(|n| n.is_chance()).count();
    let np = tree.nodes.iter().filter(|n| n.is_player()).count();

    println!("\n=== SC1: T2D production (25s) ===");
    println!("Tree: {} nodes (T:{} C:{} P:{}), nh={}", tree.num_nodes(), nt, nc, np, nh);

    // External
    let eb = ExtBet::try_from(("50%,100%","50%")).unwrap();
    let mut eg = PostFlopGame::with_config(CardConfig {
        range: [full_range().parse().unwrap(), full_range().parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(), turn: ext_card("4c").unwrap(), river: NOT_DEALT,
    }, ActionTree::new(ExtTreeConfig {
        initial_state: ExtBoard::Turn, starting_pot: 200, effective_stack: 9500,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [eb.clone(),eb.clone()], turn_bet_sizes: [eb.clone(),eb.clone()],
        river_bet_sizes: [eb.clone(),eb], turn_donk_sizes: None, river_donk_sizes: None,
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    }).unwrap()).unwrap();
    eg.allocate_memory(false);
    let t0 = std::time::Instant::now();
    let mut ei = 0u32;
    loop { solve_step(&eg, ei); ei += 1; if t0.elapsed().as_secs_f64() >= 25.0 { break; } }
    let et = t0.elapsed().as_secs_f64();
    let ee = { let mut g = PostFlopGame::with_config(CardConfig {
        range: [full_range().parse().unwrap(), full_range().parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(), turn: ext_card("4c").unwrap(), river: NOT_DEALT,
    }, ActionTree::new(ExtTreeConfig {
        initial_state: ExtBoard::Turn, starting_pot: 200, effective_stack: 9500,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [ExtBet::try_from(("50%,100%","50%")).unwrap(),ExtBet::try_from(("50%,100%","50%")).unwrap()],
        turn_bet_sizes: [ExtBet::try_from(("50%,100%","50%")).unwrap(),ExtBet::try_from(("50%,100%","50%")).unwrap()],
        river_bet_sizes: [ExtBet::try_from(("50%,100%","50%")).unwrap(),ExtBet::try_from(("50%,100%","50%")).unwrap()],
        turn_donk_sizes: None, river_donk_sizes: None,
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    }).unwrap()).unwrap(); g.allocate_memory(false); solve(&mut g, ei, 0.0, false) };
    println!("Ext:  {} iters {:.1}s ({:.0}ms/i), exp={:.4} ({:.2}%pot)", ei, et, et/ei as f64*1000.0, ee, ee/2.0);

    // GPU VCFR
    let (os, oi, ps, pi, _) = table.sorted_opp_arrays();
    let hc = table.hand_cards_gpu();
    let iw = table.initial_weight_flat();
    let (cs, ci) = table.chance_sorted_arrays_gpu();
    let gpu = GpuContext::new().expect("gpu");
    let mut cal = gpu.create_vcfr_solver_normalized(&tree, nh, &os,&oi,&ps,&pi,&hc,&iw,
        Some(ChanceGpuData { chance_sorted_strength: cs.clone(), chance_sorted_indices: ci.clone(),
            chance_probabilities: cp.clone(), remaining_deck: table.remaining_deck.clone() }),
        table.num_combinations).unwrap();
    let t1 = std::time::Instant::now(); cal.run(3).unwrap();
    let ms = t1.elapsed().as_secs_f64() / 3.0 * 1000.0;
    let tgt = (25.0 / (ms/1000.0)) as u32;
    println!("VCFR calib: {:.0}ms/i → {} target", ms, tgt);

    let mut vcfr = gpu.create_vcfr_solver_normalized(&tree, nh, &os,&oi,&ps,&pi,&hc,&iw,
        Some(ChanceGpuData { chance_sorted_strength: cs, chance_sorted_indices: ci,
            chance_probabilities: cp, remaining_deck: table.remaining_deck.clone() }),
        table.num_combinations).unwrap();
    let t2 = std::time::Instant::now(); vcfr.run(tgt).unwrap();
    let vt = t2.elapsed().as_secs_f64();
    let cum = vcfr.download_cum_strategy().unwrap();
    let prof = StrategyProfile::from_usize_offsets(&cum, &off, nh);
    let ve = exploitability(&tree, &game, &prof);
    println!("VCFR: {} iters {:.1}s ({:.0}ms/i), exp={:.4} ({:.2}%pot)", tgt, vt, vt/tgt as f64*1000.0, ve, ve/2.0);
    println!("VCFR/Ext ratio: {:.2}x, throughput: {:.1}x", ve/ee.max(0.001), (tgt as f64/vt)/(ei as f64/et));
    assert!(ve < ee * 20.0 + 1.0, "VCFR too far: {:.4} vs {:.4}", ve, ee);
}

// ─── T2D small: tighter game ──────────────────────────────────────────────

#[test]
fn sc1_t2d_small() {
    let board: Vec<Card> = ["2h","7d","Ks","4c"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let tree = build_tree(&TreeConfig {
        num_players: 2, initial_state: BoardState::Turn,
        starting_pot: 100, starting_stacks: vec![200,200],
        initial_contributions: vec![0,0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: bets("50%"), raise: vec![] },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    }).unwrap();
    let table = ChanceTable::compute_turn_start(&board, &ranges, 2);
    let nh = table.num_valid;
    let game = TurnStartGame::new(ChanceTable::compute_turn_start(&board, &ranges, 2));
    let off = offsets(&tree, nh);
    let cp = chance_probs(&table);

    println!("\n=== SC1: T2D small (10s) ===");
    println!("Tree: {} nodes, nh={}, pot=100", tree.num_nodes(), nh);

    // External
    let eb = ExtBet::try_from(("50%","")).unwrap();
    let mut eg = PostFlopGame::with_config(CardConfig {
        range: [full_range().parse().unwrap(), full_range().parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(), turn: ext_card("4c").unwrap(), river: NOT_DEALT,
    }, ActionTree::new(ExtTreeConfig {
        initial_state: ExtBoard::Turn, starting_pot: 100, effective_stack: 200,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [eb.clone(),eb.clone()], turn_bet_sizes: [eb.clone(),eb.clone()],
        river_bet_sizes: [eb.clone(),eb], turn_donk_sizes: None, river_donk_sizes: None,
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    }).unwrap()).unwrap();
    eg.allocate_memory(false);
    let t0 = std::time::Instant::now(); let mut ei = 0u32;
    loop { solve_step(&eg, ei); ei += 1; if t0.elapsed().as_secs_f64() >= 10.0 { break; } }
    let et = t0.elapsed().as_secs_f64();
    let ee = { let mut g = PostFlopGame::with_config(CardConfig {
        range: [full_range().parse().unwrap(), full_range().parse().unwrap()],
        flop: flop_from_str("2h7dKs").unwrap(), turn: ext_card("4c").unwrap(), river: NOT_DEALT,
    }, ActionTree::new(ExtTreeConfig {
        initial_state: ExtBoard::Turn, starting_pot: 100, effective_stack: 200,
        rake_rate: 0.0, rake_cap: 0.0,
        flop_bet_sizes: [ExtBet::try_from(("50%","")).unwrap(),ExtBet::try_from(("50%","")).unwrap()],
        turn_bet_sizes: [ExtBet::try_from(("50%","")).unwrap(),ExtBet::try_from(("50%","")).unwrap()],
        river_bet_sizes: [ExtBet::try_from(("50%","")).unwrap(),ExtBet::try_from(("50%","")).unwrap()],
        turn_donk_sizes: None, river_donk_sizes: None,
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    }).unwrap()).unwrap(); g.allocate_memory(false); solve(&mut g, ei, 0.0, false) };
    println!("Ext:  {} iters {:.1}s, exp={:.4} ({:.2}%pot)", ei, et, ee, ee/1.0);

    // GPU VCFR
    let (os,oi,ps,pi,_) = table.sorted_opp_arrays();
    let hc = table.hand_cards_gpu(); let iw = table.initial_weight_flat();
    let (cs,ci) = table.chance_sorted_arrays_gpu();
    let gpu = GpuContext::new().expect("gpu");
    let mut cal = gpu.create_vcfr_solver_normalized(&tree,nh,&os,&oi,&ps,&pi,&hc,&iw,
        Some(ChanceGpuData{chance_sorted_strength:cs.clone(),chance_sorted_indices:ci.clone(),
            chance_probabilities:cp.clone(),remaining_deck:table.remaining_deck.clone()}),
        table.num_combinations).unwrap();
    let t1 = std::time::Instant::now(); cal.run(3).unwrap();
    let ms = t1.elapsed().as_secs_f64()/3.0*1000.0;
    let tgt = (10.0/(ms/1000.0)) as u32;
    let mut vcfr = gpu.create_vcfr_solver_normalized(&tree,nh,&os,&oi,&ps,&pi,&hc,&iw,
        Some(ChanceGpuData{chance_sorted_strength:cs,chance_sorted_indices:ci,
            chance_probabilities:cp,remaining_deck:table.remaining_deck.clone()}),
        table.num_combinations).unwrap();
    let t2 = std::time::Instant::now(); vcfr.run(tgt).unwrap(); let vt = t2.elapsed().as_secs_f64();
    let cum = vcfr.download_cum_strategy().unwrap();
    let prof = StrategyProfile::from_usize_offsets(&cum, &off, nh);
    let ve = exploitability(&tree, &game, &prof);
    println!("VCFR: {} iters {:.1}s, exp={:.4} ({:.2}%pot)", tgt, vt, ve, ve/1.0);
    println!("VCFR/Ext ratio: {:.2}x", ve/ee.max(0.001));
    assert!(ve < ee * 20.0 + 1.0, "VCFR too far: {:.4} vs {:.4}", ve, ee);
}
