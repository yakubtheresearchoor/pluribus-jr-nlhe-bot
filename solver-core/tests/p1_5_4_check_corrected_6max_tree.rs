// Quick: just build the corrected 6-max preflop tree and report its size.
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
fn check_6max_corrected_tree_size() {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; 6],
        initial_contributions: vec![1, 2, 0, 0, 0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("builds");
    let total = tree.num_nodes();
    let mut p = 0; let mut c = 0; let mut t = 0;
    for n in &tree.nodes {
        if n.is_player() { p += 1; }
        else if n.is_chance() { c += 1; }
        else { t += 1; }
    }
    eprintln!("6-max preflop simplest CORRECTED tree:");
    eprintln!("  Total nodes: {}", total);
    eprintln!("  Player: {}, Chance: {}, Terminal: {}", p, c, t);
    eprintln!("  vs B.3 pre-fix measurement: 29,882 nodes (wrong-game)");
}
