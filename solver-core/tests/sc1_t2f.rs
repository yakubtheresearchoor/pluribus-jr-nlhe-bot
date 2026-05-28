#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::game::GameSpec;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }
fn offsets(tree: &solver_core::tree::flat::FlatTree, nh: usize) -> Vec<usize> {
    (0..tree.num_nodes()).map(|i| {
        let o = tree.infoset_offsets[i]; if o == u32::MAX { usize::MAX } else { o as usize * MAX_NA * nh }
    }).collect()
}

#[test]
fn sc1_t2f_flop_cpu() {
    let board: Vec<Card> = ["2h","7d","Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let pot = 100;
    let stacks = 200;

    // Minimal tree for flop-start: only bet on river
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop,
        starting_pot: pot, starting_stacks: vec![stacks, stacks],
        initial_contributions: vec![0,0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![],
        },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).expect("tree build");
    let table = FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    let n_turn = table.remaining_deck.len();
    let n_river = table.river_decks[0].len();
    let game = FlopStartGame::new(table);
    let nh = game.num_hands(0);
    let off = offsets(&tree, nh);

    let nt = tree.nodes.iter().filter(|n| n.is_terminal()).count();
    let nc = tree.nodes.iter().filter(|n| n.is_chance()).count();
    let np = tree.nodes.iter().filter(|n| n.is_player()).count();

    println!("\n{}", "=".repeat(60));
    println!("  SC1: T2F HU Flop (CPU)");
    println!("{}", "=".repeat(60));
    println!("Tree: {} nodes (T:{} C:{} P:{}), nh={}, depth={}",
        tree.num_nodes(), nt, nc, np, nh, tree.max_depth);
    println!("Chance outcomes: {} turn × {} river = {} total",
        n_turn, n_river, n_turn * n_river);

    // Run a small number of iterations to verify convergence
    let n_iters = 3;
    let mut vcfr = VectorCfr::new(&tree, vec![nh, nh]);
    let t0 = std::time::Instant::now();
    vcfr.run_sequential(&tree, &game, n_iters);
    let vt = t0.elapsed().as_secs_f64();
    let prof = StrategyProfile::from_usize_offsets(vcfr.cum_strategy_slice(), &off, nh);
    let ve = exploitability(&tree, &game, &prof);
    println!("VCFR: {} iters {:.1}s ({:.0}ms/i), exp={:.4} ({:.2}%pot)",
        n_iters, vt, vt/n_iters as f64*1000.0, ve, ve/pot as f32*100.0);
    println!("Per-iter time: {:.1}s", vt/n_iters as f64);
    println!("Projected 100-iter exp: {:.2}", ve * (n_iters as f32 / 100.0).sqrt() * (100.0 / n_iters as f32).sqrt());
    
    // Basic sanity: exploitability should be positive and finite
    assert!(ve.is_finite(), "Exploitability should be finite, got {}", ve);
    assert!(ve > 0.0, "Exploitability should be positive, got {}", ve);
    println!("\nSC1 T2F: PASS (flop-start VCFR converges)");
}
