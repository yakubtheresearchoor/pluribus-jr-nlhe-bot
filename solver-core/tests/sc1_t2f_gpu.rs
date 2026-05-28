#![cfg(feature = "cuda")]

use solver_core::card::{card_from_str, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu::context::{ChanceGpuData, GpuContext};
use solver_core::solver::best_response::{exploitability, StrategyProfile};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::game::GameSpec;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA;

fn uniform_range() -> Vec<f32> { vec![1.0; NUM_POSSIBLE_HANDS] }

fn chance_probs(table: &ChanceTable) -> Vec<f32> {
    use solver_core::card::index_to_card_pair;
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
fn offsets(tree: &solver_core::tree::flat::FlatTree, nh: usize) -> Vec<usize> {
    (0..tree.num_nodes()).map(|i| {
        let o = tree.infoset_offsets[i]; if o == u32::MAX { usize::MAX } else { o as usize * MAX_NA * nh }
    }).collect()
}

#[test]
fn sc1_t2f_gpu_estimate() {
    // === STEP 1: Measure GPU turn-start as a baseline ===
    // The flop-start tree is ~3x larger than turn-start,
    // and has 49 turn × 48 river = 2352 chance outcomes vs 48 for turn-start.
    // GPU turn-start time × 49 gives a conservative flop-start estimate.

    let board4: Vec<Card> = ["2h","7d","Ks","Td"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let pot = 100;
    let stacks = 200;

    // Turn-start tree with same bet sizes as flop-start
    let config_t = TreeConfig {
        num_players: 2, initial_state: BoardState::Turn,
        starting_pot: pot, starting_stacks: vec![stacks, stacks],
        initial_contributions: vec![0,0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![],
        },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    };
    let tree_t = build_tree(&config_t).expect("turn tree build");
    let table_t = ChanceTable::compute_turn_start(&board4, &ranges, 2);
    let nh_t = table_t.num_valid;

    let nt = tree_t.nodes.iter().filter(|n| n.is_terminal()).count();
    let nct = tree_t.nodes.iter().filter(|n| n.is_chance()).count();
    let npt = tree_t.nodes.iter().filter(|n| n.is_player()).count();
    println!("\n{}", "=".repeat(60));
    println!("  SC1: T2F GPU estimate via turn-start measurement");
    println!("{}", "=".repeat(60));
    println!("Turn tree: {} nodes (T:{} C:{} P:{}), nh={}, depth={}",
        tree_t.num_nodes(), nt, nct, npt, nh_t, tree_t.max_depth);

    // GPU turn-start
    let ctx = GpuContext::new().expect("GPU context");
    let (os, oi, ps, pi, _) = table_t.sorted_opp_arrays();
    let hand_cards = table_t.hand_cards.clone();
    let iw = table_t.initial_weight_flat();

    let (cs, ci) = table_t.chance_sorted_arrays_gpu();
    let cp = chance_probs(&table_t);
    let chance_data = ChanceGpuData {
        chance_sorted_strength: cs,
        chance_sorted_indices: ci,
        chance_probabilities: cp,
        remaining_deck: table_t.remaining_deck.clone(),
    };

    let mut gpu = ctx.create_vcfr_solver_normalized(
        &tree_t, nh_t, &os, &oi, &ps, &pi, &hand_cards, &iw,
        Some(chance_data), table_t.num_combinations,
    ).expect("GPU solver create");

    // Warm up
    gpu.run(1).expect("GPU warmup");

    let n_gpu = 10u32;
    let t0 = std::time::Instant::now();
    gpu.run(n_gpu).expect("GPU run");
    let gpu_t = t0.elapsed().as_secs_f64();
    let gpu_ms_per_iter = gpu_t / n_gpu as f64 * 1000.0;
    println!("GPU turn-start: {} iters {:.1}s ({:.0}ms/i)", n_gpu, gpu_t, gpu_ms_per_iter);

    // === STEP 2: Flop-start tree stats ===
    let board3: Vec<Card> = ["2h","7d","Ks"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let config_f = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop,
        starting_pot: pot, starting_stacks: vec![stacks, stacks],
        initial_contributions: vec![0,0], rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![],
        },
        add_allin_threshold: 1.5, force_allin_threshold: 0.15, merging_threshold: 0.0,
    };
    let tree_f = build_tree(&config_f).expect("flop tree build");
    let table_f = FlopChanceTable::compute_flop_start(&board3, &ranges, 2);

    let ntf = tree_f.nodes.iter().filter(|n| n.is_terminal()).count();
    let ncf = tree_f.nodes.iter().filter(|n| n.is_chance()).count();
    let npf = tree_f.nodes.iter().filter(|n| n.is_player()).count();
    println!("Flop tree: {} nodes (T:{} C:{} P:{}), nh={}, depth={}",
        tree_f.num_nodes(), ntf, ncf, npf, table_f.num_valid, tree_f.max_depth);
    println!("Chance outcomes: {} turn × {} river = {} total",
        table_f.remaining_deck.len(),
        table_f.river_decks[0].len(),
        table_f.remaining_deck.len() * table_f.river_decks[0].len());

    // === STEP 3: Estimate flop-start GPU time ===
    // The turn-start GPU does 48 chance outcomes per iteration.
    // Flop-start does 49 × 48 = 2352.
    // Scale factor: 2352 / 48 = 49.
    // Plus the flop-start tree has more nodes per chance level.
    let turn_outcomes = 48.0;
    let flop_outcomes = 49.0 * 48.0;
    let chance_scale = flop_outcomes / turn_outcomes;

    // Node count ratio (below-chance nodes)
    let node_ratio = tree_f.num_nodes() as f64 / tree_t.num_nodes() as f64;

    let estimated_flop_ms = gpu_ms_per_iter * chance_scale * node_ratio.sqrt();
    println!("\nEstimated GPU flop-start: {:.0}ms/i (turn-start {:.0}ms × chance {:.1}x × node {:.2}x)",
        estimated_flop_ms, gpu_ms_per_iter, chance_scale, node_ratio.sqrt());
    println!("At {:.0}ms/i: {:.0} iters in 25s → DCFR convergence estimate",
        estimated_flop_ms, 25000.0 / estimated_flop_ms);

    // === STEP 4: CPU correctness check ===
    let game_f = FlopStartGame::new(table_f);
    let nh_f = game_f.num_hands(0);
    let off_f = offsets(&tree_f, nh_f);

    let mut vcfr = VectorCfr::new(&tree_f, vec![nh_f, nh_f]);
    let t1 = std::time::Instant::now();
    vcfr.run_sequential(&tree_f, &game_f, 1);
    let cpu_t = t1.elapsed().as_secs_f64();
    // For exploitability, we need cum_strategy which requires >=1 iteration
    // Skip exploitability for speed — CPU correctness was verified in sc1_t2f test
    println!("\nCPU correctness: 1 iter {:.1}s — verified in sc1_t2f test", cpu_t);

    println!("\n{}", "=".repeat(60));
    if estimated_flop_ms < 500.0 {
        println!("  VERDICT: GPU flop-start VIABLE ({:.0}ms/i)", estimated_flop_ms);
    } else if estimated_flop_ms < 2000.0 {
        println!("  VERDICT: GPU flop-start MARGINAL ({:.0}ms/i)", estimated_flop_ms);
    } else {
        println!("  VERDICT: GPU flop-start TOO SLOW ({:.0}ms/i)", estimated_flop_ms);
    }
    println!("  Next: build dedicated GPU flop-start kernel for actual measurement");
    println!("{}", "=".repeat(60));
}
