//! Integration test for the connected-blueprint /decide path (ConnDecider),
//! including TURN decisions (full postflop path replay). Skipped if the solved
//! blueprint / GS14 cache aren't present.

use play_harness::api::{ActionInput, DecideRequest};
use play_harness::api_conn::ConnDecider;
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::blueprint::ConnCellLayout;
use solver_core::solver::preflop_start_game::PreflopChanceTable;

#[test]
fn conn_decider_serves_preflop_and_turn() {
    let Ok(dec) = ConnDecider::load("blueprint_conn_v1", "gs14_blueprint_cache", 6, 1, 200, 3) else { return; };

    // PREFLOP: AA should rarely fold.
    let pre = DecideRequest { board: vec![], hero_cards: [48, 49], live: 6, hero_idx: 3, ..Default::default() };
    let r = dec.decide(&pre).expect("preflop");
    assert_eq!(r.street, "preflop");
    let fold = r.actions.iter().find(|a| a.label == 0).map(|a| a.prob).unwrap_or(0.0);
    assert!(fold < 0.1, "AA fold {fold} too high");

    // TURN: build a valid turn request by walking a cell tree to its first turn node
    // (the flop-closing betting = prior_actions), then ask ConnDecider to serve it.
    let layout = ConnCellLayout::build(6, 1, 200);
    if layout.keys.is_empty() { return; }
    let ki = (0..layout.keys.len()).max_by_key(|&i| layout.cell_trees[i].num_nodes()).unwrap();
    let (live, commit, pot) = layout.keys[ki];
    let tree = &layout.cell_trees[ki];

    let mut found: Option<Vec<(u8, i32)>> = None;
    let mut stack: Vec<(usize, Vec<(u8, i32)>)> = vec![(0, vec![])];
    while let Some((mut node, hist)) = stack.pop() {
        if hist.len() > 10 { continue; }
        while tree.nodes[node].is_chance() { node = tree.node_children(node)[0] as usize; }
        if !tree.nodes[node].is_player() { continue; }
        if tree.nodes[node].board_state == 1 { found = Some(hist); break; }
        for &k in tree.node_children(node) {
            let kn = &tree.nodes[k as usize];
            let mut h2 = hist.clone();
            h2.push((kn.action_label, kn.amount));
            stack.push((k as usize, h2));
        }
    }
    let Some(path) = found else { return; };

    let ranges = vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6];
    let flop = PreflopChanceTable::new(6, ranges).canonical_flops[0];
    let bmask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let deck: Vec<u8> = (0..52u8).filter(|c| bmask & (1u64 << c) == 0).collect();
    let (h1, h2) = (deck[0], deck[1]);
    let turn = *deck.iter().find(|&&c| c != h1 && c != h2).unwrap();

    let req = DecideRequest {
        board: vec![flop[0], flop[1], flop[2], turn],
        hero_cards: [h1, h2],
        live,
        commit_entry: commit as u32,
        pot_entry: pot as u32,
        flop_id: 0,
        cell_dir: format!("live{live}_c{commit}_p{pot}"),
        prior_actions: path.iter().map(|&(l, a)| ActionInput { label: l, to_total: a as u32 }).collect(),
        ..Default::default()
    };
    let resp = dec.decide(&req).expect("turn decision should resolve");
    assert_eq!(resp.street, "turn", "expected a turn decision");
    let s: f32 = resp.actions.iter().map(|a| a.prob).sum();
    assert!((s - 1.0).abs() < 1e-3, "turn dist sums to {s}");
    assert!(resp.actions.iter().all(|a| a.prob.is_finite() && a.prob >= 0.0));
}
