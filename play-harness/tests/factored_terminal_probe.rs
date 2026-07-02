//! P0 SHARP PROBE: one direct evaluate_terminal call, factored vs EXACT, at
//! the same all-live showdown terminal, uniform reach — per-hand cfv diff for
//! true-air / bottom-pair / nuts on a dry river. np=3.
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::game::GameSpec as GameTrait;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::tree::action::{BetCap, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::build_tree;

#[test]
fn factored_vs_exact_terminal_cfv() {
    let board: Vec<solver_core::card::Card> = vec![47, 20, 3, 9, 30]; // K 7 2 4 9
    let np = 3u8;
    let ranges = vec![vec![1.0f32; 1326]; np as usize];
    let spec = solver_core::tree::action::production_game_v1();
    let table = ChanceTable::compute_river_start(&board, &ranges, np);
    let nh = table.num_valid;
    let hc = table.hand_cards.clone();
    let mut cfg = spec.street_seam_config(BoardState::River, np, 6, 18,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] });
    cfg.max_bets_per_street = BetCap::all(1);
    let tree = build_tree(&cfg).expect("tree");
    // find an ALL-LIVE showdown terminal (check-check-check)
    let node = (0..tree.num_nodes())
        .find(|&n| tree.nodes[n].is_terminal() && tree.get_folded_mask(n) == 0)
        .expect("all-live showdown");
    let reach: Vec<Vec<f32>> = (0..np as usize).map(|_| vec![1.0f32; nh]).collect();

    let g_exact = TurnStartGame::new(ChanceTable::compute_river_start(&board, &ranges, np));
    let g_fact = TurnStartGame::new(table).with_factored();
    let t0 = std::time::Instant::now();
    let cfv_f = g_fact.evaluate_terminal(0, node, &tree, &reach);
    let tf = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let cfv_e = g_exact.evaluate_terminal(0, node, &tree, &reach);
    let te = t1.elapsed().as_secs_f64();

    let rank = |c: u8| c >> 2;
    let find2 = |r1: u8, r2: u8| (0..nh).find(|&h| {
        let (a, b) = (rank(hc[h*2]), rank(hc[h*2+1]));
        (a == r1 && b == r2) || (a == r2 && b == r1)
    }).unwrap();
    let air = find2(1, 3);   // 3-5: no pair, 5-high air
    let bp = find2(0, 4);    // 2-6: bottom pair (board 2)
    let kk = find2(11, 11);  // KK top set
    eprintln!("timing: factored {tf:.2}s, exact {te:.2}s (node {node}, nh={nh})");
    for (name, h) in [("AIR 53o", air), ("botpair 62", bp), ("KK set", kk)] {
        eprintln!("  {name}: factored={:.1}  exact={:.1}", cfv_f[h], cfv_e[h]);
    }
    // aggregate agreement
    let mut worst = 0.0f64; let mut scale = 1e-9f64;
    for h in 0..nh { scale = scale.max((cfv_e[h] as f64).abs()); }
    for h in 0..nh { worst = worst.max(((cfv_f[h] - cfv_e[h]) as f64).abs() / scale); }
    eprintln!("  per-hand worst rel diff = {worst:.3}");
}

#[test]
fn factored_vs_exact_fold_terminals() {
    let board: Vec<solver_core::card::Card> = vec![47, 20, 3, 9, 30];
    let np = 3u8;
    let ranges = vec![vec![1.0f32; 1326]; np as usize];
    let spec = solver_core::tree::action::production_game_v1();
    let table = ChanceTable::compute_river_start(&board, &ranges, np);
    let nh = table.num_valid;
    let mut cfg = spec.street_seam_config(BoardState::River, np, 6, 18,
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] });
    cfg.max_bets_per_street = BetCap::all(1);
    let tree = build_tree(&cfg).expect("tree");
    let reach: Vec<Vec<f32>> = (0..np as usize).map(|_| vec![1.0f32; nh]).collect();
    let g_exact = TurnStartGame::new(ChanceTable::compute_river_start(&board, &ranges, np));
    let g_fact = TurnStartGame::new(table).with_factored();

    // every terminal with folds, for traverser 0 AND 1: compare per-hand
    for node in 0..tree.num_nodes() {
        if !tree.nodes[node].is_terminal() { continue; }
        let fm = tree.get_folded_mask(node);
        if fm == 0 { continue; }
        for trav in [0u8, 1] {
            let f = g_fact.evaluate_terminal(trav, node, &tree, &reach);
            let e = g_exact.evaluate_terminal(trav, node, &tree, &reach);
            let mut scale = 1e-9f64;
            for h in 0..nh { scale = scale.max((e[h] as f64).abs()); }
            let mut worst = 0.0f64; let mut wh = 0;
            for h in 0..nh {
                let d = ((f[h] - e[h]) as f64).abs() / scale;
                if d > worst { worst = d; wh = h; }
            }
            eprintln!("node {node} fm={fm:03b} trav={trav}: worst rel {worst:.3} (h={wh}: f={:.0} e={:.0})",
                f[wh], e[wh]);
        }
    }
}
