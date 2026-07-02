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
