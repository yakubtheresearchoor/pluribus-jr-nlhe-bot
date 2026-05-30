/// Compare terminal CFVs between our solver and b1nary at a specific terminal.
/// Both should give identical CFVs for uniform strategy on the same game.
///
/// Run:
///   cargo test -p solver-core --features metal --test terminal_cfvs_both -- --test-threads=1 --nocapture --ignored

use solver_core::card::card_from_str;
use solver_core::solver::flop_start_game::FlopStartGame;
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

/// Compare the CFV at the root for P0 with uniform strategy.
/// This should be computable by hand for both solvers.
#[test]
#[ignore]
fn root_cfv_comparison() {
    let board: Vec<solver_core::card::Card> = ["2h", "7d", "Ks"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![vec![1.0f32; 1326], vec![1.0; 1326]];
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&config).unwrap();
    let table = solver_core::solver::flop_start_game::FlopChanceTable::compute_flop_start(&board, &ranges, 2);
    let game = FlopStartGame::new(table);
    let solver = FlopStartVectorCfr::new(&tree, game.table());

    let nh = solver.num_hands();
    let nc = game.table().num_combinations as f32;

    let sv0 = solver.strategy_value(&tree, &game, 0);
    let sv1 = solver.strategy_value(&tree, &game, 1);

    println!("\n=== Root CFV comparison ===");
    println!("nh={}, nc={:.0}", nh, nc);

    // Print first 10 hands
    println!("\nFirst 10 hands SV (P0):");
    for h in 0..10.min(nh) {
        println!("  h={}: SV0={:.6} SV1={:.6}", h, sv0[h], sv1[h]);
    }

    // Sum statistics
    let sv0_sum: f32 = sv0.iter().sum();
    let sv1_sum: f32 = sv1.iter().sum();
    let sv0_abs_sum: f32 = sv0.iter().map(|v| v.abs()).sum();
    println!("\n  SV0 sum = {:.6}, avg = {:.6}", sv0_sum, sv0_sum / nh as f32);
    println!("  SV1 sum = {:.6}, avg = {:.6}", sv1_sum, sv1_sum / nh as f32);
    println!("  SV0 abs_sum = {:.6}", sv0_abs_sum);

    // The exploitability for uniform strategy is:
    // expl = sum_h w[h] * (BR[h] - SV[h]) / np for each player
    // With uniform strategy, SV[h] = cfv[h] (since sigma = 1/na for all actions)
    // And BR[h] = max_a cfv_a[h]
    // At the root with uniform strategy, the first action is always the best
    // (since all actions have equal weight), so BR ≈ SV for flat strategy trees.
    //
    // Wait, that's only if the tree has one action. With 3 actions (check, bet, fold?),
    // the BR picks the best action per hand.

    // Actually the key question: does our exploitability divide by nc the right
    // number of times? Let me trace through:
    //
    // 1. evaluate_terminal divides by nc → SV[h] already /nc
    // 2. compute_exploitability sums reach[h] * (BR[h] - SV[h]) / np
    //    where reach = initial_weight = 1.0
    // 3. So expl = sum_h (BR[h] - SV[h]) / np / nc? No...
    //    expl = (sum_h w0[h]*(BR0[h]-SV0[h]) + sum_h w1[h]*(BR1[h]-SV1[h])) / np
    //    where BR and SV are already /nc
    //
    // b1nary:
    // 1. evaluate divides by nc → amount_win/nc
    // 2. compute_exploitability sums cfvalue * reach / nc? Or just cfvalue * reach?
    //
    // Let me check b1nary's formula:
    // compute_exploitability in utility.rs:
    //   expl = sum_p (weighted_sum(cfvalues_br[p], reach[p]) - weighted_sum(cfvalues_sv[p], reach[p])) / np
    // where cfvalues are from compute_cfvalue_recursive
    // and the terminal divides by nc
    //
    // So b1nary: expl = sum_p sum_h reach[p][h] * (cfv_br[p][h] - cfv_sv[p][h]) / np
    // Our:       expl = sum_p sum_h w[p][h] * (BR[p][h] - SV[p][h]) / np
    //
    // Both divide by np. Both use per-hand cfvs already /nc. The formulas are identical.
    //
    // UNLESS: our BR/SV values are NOT the same as b1nary's cfv_br/cfv_sv.
    // The difference could be in the chance node handling.

    let br0 = solver.best_response_value(&tree, &game, 0);
    let br1 = solver.best_response_value(&tree, &game, 1);

    let w0 = &game.table().initial_weights[0];

    let expl_manual = (0..nh).map(|h| w0[h] * (br0[h] - sv0[h])).sum::<f32>()
        + (0..nh).map(|h| w0[h] * (br1[h] - sv1[h])).sum::<f32>();
    let expl = expl_manual / 2.0;

    println!("\n  Our exploitability = {:.6e}", expl);

    // Now compute b1nary's exploitability for comparison
    use postflop_solver::{Game as B1Game, compute_exploitability as b1_expl};
    use postflop_solver::{PostFlopGame, BetSizeOptions as BBetSizeOptions, flop_from_str, Range,
        CardConfig, TreeConfig as BTreeConfig, ActionTree};

    let one_pot = BBetSizeOptions { bet: vec![postflop_solver::BetSize::PotRelative(1.0)], raise: vec![] };
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("2h7dKs").unwrap(),
        ..Default::default()
    };
    let tree_config = BTreeConfig {
        starting_pot: 10, effective_stack: 95,
        flop_bet_sizes: [one_pot.clone(), one_pot.clone()],
        turn_bet_sizes: [one_pot.clone(), one_pot.clone()],
        river_bet_sizes: [one_pot.clone(), one_pot.clone()],
        ..Default::default()
    };
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut b1game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    b1game.allocate_memory(false);

    let b1_expl_val = b1_expl(&b1game);
    println!("  b1nary exploitability = {:.6e}", b1_expl_val);
    println!("  Ratio = {:.2}x", expl / b1_expl_val);

    // Check: what is the ratio of nc values?
    // b1nary: nc = sum of w0[h0]*w1[h1] for non-conflicting pairs
    // our: nc = same computation = 1271256
    // They should be identical since same game.

    // Let me check b1nary's chance_factor:
    // For flop-start game (node.turn == NOT_DEALT), chance_factor = 45.
    // For each turn card, chance_factor_river = 44.
    // cfreach *= 1/chance_factor at chance nodes.
    //
    // Our solver: prob(h,o) = 1/(49 - 2) = 1/47 for turn, 1/(48-2) = 1/46 for river.
    // b1nary: 1/45 for turn, 1/44 for river.
    //
    // 1/47 vs 1/45 → ratio = 45/47 ≈ 0.957
    // Two chance nodes: (45/47) * (44/46) ≈ 0.917
    // Total ratio: 1/0.917 ≈ 1.09
    //
    // But 4.28x >> 1.09x. So chance_factor difference is NOT the main cause.
    //
    // Wait — the chance_factor affects cfreach, not the CFV directly.
    // Let me think again about how CFVs propagate through chance nodes.
    //
    // In our solver: cfv[h] = sum_o prob(h,o) * child_cfv[h,o]
    // With prob = 1/47 for turn, the cfv is 1/47 * sum_child.
    //
    // In b1nary: the chance node multiplies cfreach by 1/chance_factor.
    // The terminal CFV is proportional to cfreach. So the CFV is
    // (1/chance_factor) * (sum over children of child_cfv)
    //
    // Wait, b1nary's chance node handling is different. It doesn't compute
    // cfv at the chance node. Instead, it modifies cfreach and recurses.
    // The cfv at the chance node is the sum of children's cfvs.
    // But cfreach is modified BEFORE recursing, so children see reduced cfreach.
    //
    // Actually, let me re-read b1nary's utility.rs chance handling:
    println!("\n  Need to check b1nary's chance handling in detail...");
}
