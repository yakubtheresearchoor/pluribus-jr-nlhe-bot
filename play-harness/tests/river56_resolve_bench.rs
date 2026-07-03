//! MEASURE-FIRST: the exact multiway river re-solve at np=5/6 (single street,
//! exact factored showdown, NO continuation/grid) — decides whether live-5/6
//! river decisions can be served by solve_multiway_street.
use play_harness::live2_bank::{solve_multiway_street, LIVE2_RT_BUDGET_MS, LIVE2_RT_RIVER_ITERS};

#[test]
#[ignore = "measurement bench"]
fn river_resolve_np56() {
    let board: Vec<u8> = vec![51, 46, 20, 9, 30]; // A K 7 4 9 rainbow-ish river
    for live in [5u8, 6] {
        let t0 = std::time::Instant::now();
        let s = solve_multiway_street(&board, live, 4, (live as i32) * 4 + 4, LIVE2_RT_RIVER_ITERS, LIVE2_RT_BUDGET_MS);
        let el = t0.elapsed().as_secs_f64();
        match &s {
            Some(sv) => eprintln!("river resolve live-{live}: {el:.2}s ({} nodes, iters={})", sv.tree.num_nodes(), LIVE2_RT_RIVER_ITERS),
            None => eprintln!("river resolve live-{live}: {el:.2}s -> None"),
        }
    }
}

#[test]
#[ignore = "production-config river resolve"]
fn river_resolve_prod_cfg() {
    let board: Vec<u8> = vec![51, 50, 20, 9, 30]; // the e2e canonical-ish board (paired aces)
    for (live, iters, bd) in [(5u8, 120u32, 8_000u128), (6, 40, 9_000)] {
        let t0 = std::time::Instant::now();
        let s = solve_multiway_street(&board, live, 4, (live as i32) * 4 + 4, iters, bd);
        let el = t0.elapsed().as_secs_f64();
        match &s {
            Some(sv) => eprintln!("river prod-cfg live-{live} iters={iters}: {el:.2}s ({} nodes)", sv.tree.num_nodes()),
            None => eprintln!("river prod-cfg live-{live}: {el:.2}s -> None"),
        }
    }
}

#[test]
#[ignore = "turn resolve breakdown"]
fn turn_resolve_breakdown() {
    // np=3/4 TURN boards through solve_multiway_street (the off-grid fallback).
    let board: Vec<u8> = vec![47, 20, 2, 9]; // K 7 2 4 turn
    for np in [3u8, 4] {
        let t0 = std::time::Instant::now();
        let s = solve_multiway_street(&board, np, 6, (np as i32) * 6, 600, 9_000);
        let el = t0.elapsed().as_secs_f64();
        match &s {
            Some(sv) => eprintln!("turn resolve np={np}: {el:.2}s total ({} nodes)", sv.tree.num_nodes()),
            None => eprintln!("turn resolve np={np}: {el:.2}s -> None"),
        }
    }
}

#[test]
#[ignore = "setup isolation"]
fn turn_table_setup_cost() {
    use solver_core::solver::chance_table::ChanceTable;
    let board: Vec<solver_core::card::Card> = vec![47, 20, 2, 9];
    for np in [3u8, 4] {
        let ranges = vec![vec![1.0f32; 1326]; np as usize];
        let t0 = std::time::Instant::now();
        let table = ChanceTable::compute_turn_start(&board, &ranges, np);
        eprintln!("compute_turn_start np={np}: {:.2}s (nh={})", t0.elapsed().as_secs_f64(), table.num_valid);
    }
}

#[test]
#[ignore = "live-6 river engagement debug"]
fn live6_river_engagement() {
    use play_harness::api::{decide_postflop_resolve_ranged, route_to_canonical, DecideRequest, ActionInput};
    // the exact e2e request that fell through to the lookup
    let mut req = DecideRequest {
        board: vec![51, 50, 20, 9, 30],
        hero_cards: [48, 49],
        live: 6, hero_idx: 1,
        commit_entry: 4, pot_entry: 28, to_call: Some(28),
        street_actions: vec![ActionInput { label: 3, to_total: 28 }],
        route: true,
        ..Default::default()
    };
    route_to_canonical(&mut req).expect("route");
    let t0 = std::time::Instant::now();
    let r = decide_postflop_resolve_ranged(&req, 40, 9_000, None);
    eprintln!("live-6 river resolve: {:.1}s -> {}", t0.elapsed().as_secs_f64(),
        if let Some(rr) = &r { format!("Some({} actions)", rr.actions.len()) } else { "None".into() });
    if let Some(rr) = r {
        for a in &rr.actions { eprintln!("  {} {:.2}", a.action, a.prob); }
    }
}
