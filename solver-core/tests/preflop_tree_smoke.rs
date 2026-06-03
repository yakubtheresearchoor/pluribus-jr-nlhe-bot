// P1 smoke test: build a preflop tree, inspect structure.
// Confirms preflop tree builds, contains preflop + flop + turn + river
// zones, and the chance transitions are structurally correct.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{ACTION_LABEL_CHANCE, ACTION_LABEL_FOLD, ACTION_LABEL_CHECK, ACTION_LABEL_CALL, ACTION_LABEL_BET, ACTION_LABEL_RAISE, ACTION_LABEL_ALLIN};

fn label_name(l: u8) -> &'static str {
    match l {
        ACTION_LABEL_FOLD => "F", ACTION_LABEL_CHECK => "K", ACTION_LABEL_CALL => "C",
        ACTION_LABEL_BET => "B", ACTION_LABEL_RAISE => "R", ACTION_LABEL_ALLIN => "A",
        ACTION_LABEL_CHANCE => "+", _ => "?",
    }
}

fn board_name(b: u8) -> &'static str {
    match b {
        0 => "FLOP", 1 => "TURN", 2 => "RIVER", 3 => "PRE", _ => "?",
    }
}

#[test]
fn preflop_tree_builds_and_has_four_zones() {
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![2, 1],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    };
    let tree = build_tree(&cfg).expect("preflop tree builds");
    eprintln!("\nHU preflop [BB=2, SB=1] tree: {} nodes total", tree.num_nodes());

    let mut by_zone: std::collections::BTreeMap<&str, (usize, usize, usize)> =
        std::collections::BTreeMap::new(); // (player, chance, terminal)
    for n in &tree.nodes {
        let zone = board_name(n.board_state);
        let entry = by_zone.entry(zone).or_insert((0, 0, 0));
        if n.is_player() { entry.0 += 1; }
        else if n.is_chance() { entry.1 += 1; }
        else { entry.2 += 1; }
    }
    for (zone, (p, c, t)) in &by_zone {
        eprintln!("  {}: {} PLAYER, {} CHANCE, {} TERMINAL", zone, p, c, t);
    }

    // Confirm all four zones are present in the tree.
    assert!(by_zone.contains_key("PRE"), "no Preflop zone nodes");
    assert!(by_zone.contains_key("FLOP"), "no Flop zone nodes");
    assert!(by_zone.contains_key("TURN"), "no Turn zone nodes");
    assert!(by_zone.contains_key("RIVER"), "no River zone nodes");

    // Root is preflop, acts player 1 (button = SB).
    assert_eq!(tree.nodes[0].board_state, BoardState::Preflop as u8);
    assert_eq!(tree.nodes[0].player_id, 1);

    // First chance transition (preflop → flop) must produce a CHANCE node
    // with board_state = Flop (destination). Walk to find it.
    let mut first_chance_idx: Option<usize> = None;
    for (i, n) in tree.nodes.iter().enumerate() {
        if n.is_chance() && n.board_state == BoardState::Flop as u8 {
            first_chance_idx = Some(i);
            break;
        }
    }
    let pf_to_flop = first_chance_idx.expect("no Preflop→Flop chance node found");
    eprintln!("\nFirst Preflop→Flop chance: node[{}]", pf_to_flop);
    let n = &tree.nodes[pf_to_flop];
    eprintln!("  type={} board={} children={}", n.node_type, board_name(n.board_state), n.num_children);
    assert_eq!(n.num_children, 1, "CHANCE node must have exactly 1 structural child");

    // The chance child must be a PLAYER on the FLOP, acting player 0 (BB)
    // per postflop convention (postflop, BB acts first).
    let child_idx = tree.children[n.children_start as usize] as usize;
    let child = &tree.nodes[child_idx];
    eprintln!("  → child node[{}]: type={} player={} board={}",
        child_idx, child.node_type, child.player_id, board_name(child.board_state));
    assert!(child.is_player(), "post-chance child must be PLAYER");
    assert_eq!(child.board_state, BoardState::Flop as u8, "post-chance child on Flop");
    assert_eq!(child.player_id, 0,
        "post-flop-chance child must act player 0 (BB acts first POSTFLOP), got {}",
        child.player_id);
}
