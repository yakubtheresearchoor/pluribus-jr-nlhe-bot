use postflop_solver::{PostFlopGame, Range, CardConfig, TreeConfig, ActionTree, Game};

fn main() {
    let one_pot = postflop_solver::BetSizeOptions {
        bet: vec![postflop_solver::BetSize::PotRelative(1.0)],
        raise: vec![],
    };
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: postflop_solver::flop_from_str("2h7dKs").unwrap(),
        ..Default::default()
    };
    let tree_config = TreeConfig {
        starting_pot: 10,
        effective_stack: 95,
        flop_bet_sizes: [one_pot.clone(), one_pot.clone()],
        turn_bet_sizes: [one_pot.clone(), one_pot.clone()],
        river_bet_sizes: [one_pot.clone(), one_pot.clone()],
        ..Default::default()
};
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);

    println!("b1nary num_private_hands(0) = {}", game.num_private_hands(0));
    println!("b1nary num_private_hands(1) = {}", game.num_private_hands(1));
    println!("b1nary initial_weights(0) sum = {:.0}", game.initial_weights(0).iter().sum::<f32>());
}
