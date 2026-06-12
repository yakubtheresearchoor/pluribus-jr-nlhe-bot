// Step 1 — Identity test for the allocation-hoist fix in
// FlopStartVectorCfr::run.
//
// This test runs `solver.run(&tree, &game, n_iters)` on three small
// completable HU configurations and captures the bit-exact CFV output
// (root_cfv_sum returned by run). The recorded values are the golden
// reference; after the allocation-hoist fix to run(), this test must
// reproduce them EXACTLY.
//
// Rationale: the fix hoists per-iter allocations and uses slice-zeroing
// + overwrite-by-bottom_up_zone. The correctness argument relies on
// "every read after bottom_up_zone is a value just-written by this same
// call." If that's wrong anywhere, stale data leaks in and downstream
// CFV values drift. Bit-exact comparison is the most discriminating
// possible check.
//
// Three configs:
//   1. Minimal: HU, 1 bet + 0 raise, n_iters=1 (smallest tree, smoke).
//   2. Multi-iter: HU, 1 bet + 1 raise, n_iters=3 (exercises iteration loop).
//   3. Wider: HU, 2 bet + 0 raise, n_iters=2 (more chance children + iters).
//
// Workflow:
//   Phase A (pre-fix, captured NOW): print current run() output, hardcode
//     into the golden_* arrays below.
//   Phase B (post-fix): run this test, must match golden values bit-exact.
//
// Note: HU only at this scale because 6-max + production-size flop trees
// are infeasible on CPU even at n_iters=1 (the bug we're fixing). The
// HU identity is sufficient: bottom_up_zone's correctness over the river
// /turn/flop zones is np-agnostic for the allocation-pattern fix.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn run_one(label: &str, flop_tree: &FlatTree, n_iters: u32) -> Vec<f32> {
    let np = flop_tree.num_players;
    let class_weights: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let pre_table = PreflopChanceTable::new(np, class_weights);
    let canonical: [Card; 3] = pre_table.canonical_flops[0];
    let combo_ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS])
        .collect();
    let board: Vec<Card> = canonical.iter().copied().collect();

    let table = FlopChanceTable::compute_flop_start(&board, &combo_ranges, np);
    let nh = table.num_valid;
    let game = FlopStartGame::new(table);
    let mut solver = FlopStartVectorCfr::new(flop_tree, game.table());
    let cfv = solver.run(flop_tree, &game, n_iters);
    eprintln!("\n{}: nh={}, run({} iters) root_cfv_sum[0..6] = {:?}",
              label, nh, n_iters, cfv.iter().take(6).collect::<Vec<_>>());
    eprintln!("  full length = {}, sum = {:.6}, |max| = {:.6e}",
              cfv.len(), cfv.iter().sum::<f32>(),
              cfv.iter().map(|x| x.abs()).fold(0.0_f32, f32::max));
    cfv
}

fn cfg_hu_1bet_0raise(starting_stacks: i32, starting_pot: i32) -> TreeConfig {
    TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot,
        starting_stacks: vec![starting_stacks, starting_stacks],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    }
}

fn cfg_hu_1bet_1raise(starting_stacks: i32, starting_pot: i32) -> TreeConfig {
    TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot,
        starting_stacks: vec![starting_stacks, starting_stacks],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    }
}

fn cfg_hu_2bet_0raise(starting_stacks: i32, starting_pot: i32) -> TreeConfig {
    TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot,
        starting_stacks: vec![starting_stacks, starting_stacks],
        initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
            max_bets_per_street: None,
    }
}

// Phase A (pre-fix): RUN NOW to capture golden values. The asserts on
// `expected_*` are commented out until we have the captures in hand.
// After capture, uncomment + hardcode the values; the test then becomes
// a bit-exact post-fix gate.

#[test]
#[ignore = "Step 1 fix identity: HU run() pre/post-fix bit-exact gate. Run on demand."]
fn identity_pre_post_fix_hu_three_configs() {
    eprintln!("\n=== Identity test for FlopStartVectorCfr::run alloc-hoist fix ===\n");
    eprintln!("Phase A (PRE-FIX): captures current output as golden reference.");
    eprintln!("Phase B (POST-FIX): same test must produce bit-exact match.\n");

    // Config 1: smallest tree, 1 iter — the smoke discriminator.
    let cfg1 = cfg_hu_1bet_0raise(20, 4);
    let tree1 = build_tree(&cfg1).expect("cfg1 builds");
    eprintln!("Config 1: HU 1+0, stacks=20, pot=4, tree={} nodes, 1 iter", tree1.num_nodes());
    let cfv1 = run_one("cfg1", &tree1, 1);

    // Config 2: multi-iter, exercises the iteration loop.
    let cfg2 = cfg_hu_1bet_1raise(15, 4);
    let tree2 = build_tree(&cfg2).expect("cfg2 builds");
    eprintln!("\nConfig 2: HU 1+1, stacks=15, pot=4, tree={} nodes, 3 iters", tree2.num_nodes());
    let cfv2 = run_one("cfg2", &tree2, 3);

    // Config 3: 2 bet sizes, 2 iters — exercises wider chance + per-iter.
    let cfg3 = cfg_hu_2bet_0raise(20, 6);
    let tree3 = build_tree(&cfg3).expect("cfg3 builds");
    eprintln!("\nConfig 3: HU 2+0, stacks=20, pot=6, tree={} nodes, 2 iters", tree3.num_nodes());
    let cfv3 = run_one("cfg3", &tree3, 2);

    // Phase A capture: print bit-precise representations for hardcoding.
    eprintln!("\n=== Phase A capture (paste these into the golden hardcodes) ===");
    for (label, cfv) in [("cfg1", &cfv1), ("cfg2", &cfv2), ("cfg3", &cfv3)] {
        eprintln!("// {} ({} entries)", label, cfv.len());
        eprintln!("let golden_{}_bits: Vec<u32> = vec![{}];", label,
                  cfv.iter().map(|x| format!("{:#x}", x.to_bits())).collect::<Vec<_>>().join(", "));
    }

    // Phase B asserts (TO BE FILLED): bit-exact comparison.
    // After applying the fix, uncomment and the test acts as the gate.
    //
    // assert_eq!(cfv1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), golden_cfg1_bits);
    // assert_eq!(cfv2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), golden_cfg2_bits);
    // assert_eq!(cfv3.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), golden_cfg3_bits);

    let _ = (cfv1, cfv2, cfv3);
}
