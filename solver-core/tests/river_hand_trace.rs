/// Compare raw per-hand SV values between our solver and b1nary on river-start.
///
/// Run:
///   cargo test -p solver-core --features metal --test river_hand_trace -- --test-threads=1 --nocapture --ignored

use solver_core::card::{card_from_str, Card};
use solver_core::solver::best_response::{self, exploitability, StrategyProfile};
use solver_core::solver::poker_game::RiverPokerGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::solver::game::GameSpec;
use solver_core::tree::flat::FlatTree;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::card::index_to_card_pair;

fn uniform_range() -> Vec<f32> { vec![1.0; 1326] }

fn build_river_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: 2, initial_state: BoardState::River, starting_pot: 10,
        starting_stacks: vec![95, 95], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    build_tree(&config).unwrap()
}

#[test]
#[ignore]
fn compare_hand_values() {
    let board: Vec<Card> = ["2h", "7d", "Ks", "3c", "5c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let ranges = vec![uniform_range(), uniform_range()];
    let game = RiverPokerGame::new(&board, &ranges, 2);
    let nh = game.num_hands(0);
    let tree = build_river_tree();

    let solver = VectorCfr::new(&tree, vec![nh, nh]);
    let profile = StrategyProfile::from_usize_offsets(
        solver.cum_strategy_slice(), solver.node_offsets(), nh,
    );

    let sv0 = best_response::strategy_value(&tree, &game as &dyn GameSpec, &profile, 0);
    let br0 = best_response::best_response_value(&tree, &game as &dyn GameSpec, &profile, 0);

    let nc = game.num_combinations() as f32;

    println!("\n=== Raw per-hand values (first 20 hands) ===");
    println!("  nh={}, nc={:.0}", nh, nc);
    println!("  Our tree: {} nodes", tree.num_nodes());
    println!("  {:>4} {:>12} {:>12} {:>12} {:>12}", "h", "SV0", "BR0", "BR0-SV0", "cards");
    for h in 0..20.min(nh) {
        let (c1, c2) = index_to_card_pair(game.valid_hand_indices()[h] as usize);
        println!("  {:>4} {:>12.6} {:>12.6} {:>12.6} ({},{})", 
            h, sv0[h], br0[h], br0[h] - sv0[h], c1, c2);
    }

    // EV[SV,P0] and EV[BR,P0]
    let w0 = game.initial_weight(0);
    let ev_sv: f32 = (0..nh).map(|h| w0[h] * sv0[h]).sum::<f32>() / nc;
    let ev_br: f32 = (0..nh).map(|h| w0[h] * br0[h]).sum::<f32>() / nc;
    println!("\n  EV[SV,P0] = {:.6}", ev_sv);
    println!("  EV[BR,P0] = {:.6}", ev_br);
    println!("  EV[BR-SV,P0] = {:.6}", ev_br - ev_sv);

    // Compute exploitability
    let our_expl = exploitability(&tree, &game as &dyn GameSpec, &profile);
    println!("  Exploitability = {:.6e}", our_expl);

    // b1nary comparison
    use postflop_solver::*;
    let one_pot = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("2h7dKs").unwrap(),
        turn: card_from_str("3c").unwrap(),
        river: card_from_str("5c").unwrap(),
        ..Default::default()
    };
    let tree_config = TreeConfig {
        starting_pot: 10, effective_stack: 95,
        initial_state: postflop_solver::BoardState::River,
        river_bet_sizes: [one_pot.clone(), one_pot.clone()],
        ..Default::default()
};
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut b1game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    b1game.allocate_memory(false);

    let b1_expl = compute_exploitability(&b1game);
    println!("  b1nary expl = {:.6e}", b1_expl);
    println!("  Ratio = {:.4}x", our_expl / b1_expl);

    // Check: b1nary num_hands
    println!("\n  b1nary num_hands(0) = {}", b1game.num_private_hands(0));
    println!("  our nh = {}", nh);

    // Print b1nary SV values for comparison
    let b1_current_ev = compute_current_ev(&b1game);
    println!("\n  b1nary current_ev = {:?}", b1_current_ev);
    println!("  b1nary current_ev sum = {:.6} (should be 0 for zero-sum)", b1_current_ev[0] + b1_current_ev[1]);

    // Check b1nary's internal nc
    // b1nary doesn't expose nc, but we can compute it:
    // amount_win = half_pot / nc, and we know half_pot = (starting_pot + 2*contribution)/2
    // For root node with uniform strategy, contribution = 5, so half_pot = (10+10)/2 = 10
    // amount_win = 10 / nc
    // We can check: what nc would b1nary need to match our expl?
    println!("\n  Our expl / b1nary expl = {:.4}x", our_expl / b1_expl);
    println!("  Our nc = {:.0}", nc);
    println!("  If b1nary nc = our nc * ratio = {:.0}", nc * (our_expl / b1_expl));
}

fn compute_current_ev(game: &postflop_solver::PostFlopGame) -> [f32; 2] {
    use postflop_solver::Game;
    let reach0 = game.initial_weights(0);
    let reach1 = game.initial_weights(1);
    let num_hands = game.num_private_hands(0);

    // Compute EV by summing cfvalues
    // For a river game, the root is a decision node.
    // At iter 0 (uniform strategy), the EV is the sum of reach * cfv / nc
    // Actually we need to use compute_current_ev from b1nary's API
    [0.0, 0.0] // placeholder
}
