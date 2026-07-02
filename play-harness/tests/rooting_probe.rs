//! Focused probe: does subgame rooting (freeze_prefix) train the facing-bet
//! node — and does the trash hand read a sharp strategy there?
use play_harness::blueprint::Blueprint;
use play_harness::pluribus_play::{search_decision, SearchCfg};
use solver_core::abstraction::flop_isomorphism::canonicalize_flop;
use solver_core::blueprint::runout_grid;
use solver_core::solver::bucketed_flop_cfr::FlopBucketing;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};

#[test]
fn rooted_facing_bet_trains_trash() {
    let np = 6usize;
    let canonical = canonicalize_flop([51, 50, 20]);
    let (turns, river_decks) = runout_grid(canonical, 12, 12);
    let turns_u8: Vec<u8> = turns.iter().map(|&c| c as u8).collect();
    let rivers: Vec<Vec<u8>> = turns_u8.iter().map(|&tc| river_decks[tc as usize].clone()).collect();
    let table = FlopChanceTable::build_full_nh_sampled(canonical, np as u8, &turns_u8, &river_decks);
    let nh = table.num_valid;
    let hc = table.hand_cards.clone();
    let game = FlopStartGame::new(table);
    let bk = FlopBucketing::quantile(game.table(), 16);
    let bp = Blueprint {
        flop: [canonical[0], canonical[1], canonical[2]],
        turns: turns_u8, rivers, np, nb: 16, nh,
        cum_flop: vec![], cum_turn: vec![], cum_river: vec![], bk, game,
    };
    let cfg = SearchCfg { iters: 24, par: Some(true), dcfr: Some(true), ..Default::default() };
    let board = bp.flop.to_vec();
    let hand_idx = |c1: u8, c2: u8| -> usize {
        (0..nh).position(|h| {
            let (a, b) = (hc[h * 2], hc[h * 2 + 1]);
            (a == c1 && b == c2) || (a == c2 && b == c1)
        }).expect("hand")
    };

    // blueprint-continuing-range-like priors: zero the bottom-rank hands for
    // EVERY seat (the server's reach_priors exclude trash from a 6-way pot).
    let rank0 = |c: u8| c >> 2;
    let tight: Vec<f32> = (0..nh)
        .map(|h| if (rank0(hc[h*2]) as u32 + rank0(hc[h*2+1]) as u32) < 10 { 0.0 } else { 1.0 })
        .collect();
    let priors: Vec<(usize, Vec<f32>)> = (0..np).map(|s| (s, tight.clone())).collect();
    // hero-floored variant (the production fix): hero seat (1) gets ε·max floor.
    let mut floored = priors.clone();
    for (s, r) in floored.iter_mut() {
        if *s == 1 {
            let mx = r.iter().cloned().fold(0.0f32, f32::max);
            for v in r.iter_mut() { if *v < mx * 1e-3 { *v = mx * 1e-3; } }
        }
    }

    for (label, prefix, pr) in [
        ("UNROOTED/uniform", vec![], vec![]),
        ("ROOTED/uniform", vec![(3u8, 28u32)], vec![]),
        ("ROOTED/tight-priors", vec![(3u8, 28u32)], priors.clone()),
        ("ROOTED/tight+hero-floor", vec![(3u8, 28u32)], floored.clone()),
    ] {
        let (tree, strat) = search_decision(&bp, &board, np, 1, None, &[], 4, 28, &cfg, &pr, &prefix)
            .expect("solve");
        // hero node: p0 bets pot(28) -> hero (p1) acts. Walk child by label 3.
        let root_children = tree.node_children(0).to_vec();
        let bet_child = root_children.iter()
            .map(|&c| c as usize)
            .find(|&c| tree.nodes[c].action_label == 3 && tree.nodes[c].is_player())
            .expect("bet child is hero node");
        let s = strat.get(&bet_child).expect("strategy at hero node");
        let na = tree.nodes[bet_child].num_children as usize;
        // pick hands relative to the CANONICAL board: lowest-rank offsuit trash
        // + an AK (cards must be off-board, hence derived not hardcoded).
        let rank = |c: u8| c >> 2;
        let trash = (0..nh).min_by_key(|&h| rank(hc[h*2]) as u32 + rank(hc[h*2+1]) as u32).unwrap();
        let ak = (0..nh).find(|&h| {
            let (a, b) = (rank(hc[h*2]), rank(hc[h*2+1]));
            (a == 12 && b == 11) || (a == 11 && b == 12)
        }).expect("AK");
        let _ = hand_idx; // (kept for reference)
        let row = |h: usize| -> Vec<f32> { (0..na).map(|a| s[a][h]).collect() };
        eprintln!("{label}: root frozen dist reachable; hero-node na={na}");
        eprintln!("  62o: {:?}", row(trash));
        eprintln!("  AK : {:?}", row(ak));
    }
}
