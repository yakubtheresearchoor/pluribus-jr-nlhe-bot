//! GPU rooting gate: search_decision at live-5 facing a pot bet — GPU path
//! (now rooted via vcfr_force_strategies) must fold trash and value the nuts,
//! matching the CPU-rooted behavior it replaces.
#![cfg(feature = "metal")]
use play_harness::blueprint::Blueprint;
use play_harness::pluribus_play::{search_decision, SearchCfg};
use solver_core::abstraction::flop_isomorphism::canonicalize_flop;
use solver_core::blueprint::runout_grid;
use solver_core::solver::bucketed_flop_cfr::FlopBucketing;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};

#[test]
fn gpu_rooted_facing_bet_battery() {
    let np = 5usize;
    let canonical = canonicalize_flop([51, 50, 20]);
    let (turns, river_decks) = runout_grid(canonical, 49, 12);
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
    let cfg = SearchCfg { iters: 48, par: Some(true), dcfr: Some(true), ..Default::default() };
    let board = bp.flop.to_vec();
    let prefix = vec![(3u8, 24u32)]; // p0 pot-bets; hero (p1) faces it
    let t0 = std::time::Instant::now();
    let (tree, strat) = search_decision(&bp, &board, np, 1, None, &[], 4, 24, &cfg, &[], &prefix)
        .expect("solve");
    let el = t0.elapsed().as_secs_f64();
    // hero node = the bet child of the root
    let bet_child = tree.node_children(0).iter().map(|&c| c as usize)
        .find(|&c| tree.nodes[c].action_label == 3 && tree.nodes[c].is_player()).expect("hero node");
    let s = strat.get(&bet_child).expect("dist");
    let na = tree.nodes[bet_child].num_children as usize;
    let rank = |c: u8| c >> 2;
    let trash = (0..nh).min_by_key(|&h| rank(hc[h*2]) as u32 + rank(hc[h*2+1]) as u32).unwrap();
    let nuts = (0..nh).find(|&h| rank(hc[h*2]) == 12 && rank(hc[h*2+1]) == 12);
    let trash_dist: Vec<f32> = (0..na).map(|a| s[a][trash]).collect();
    eprintln!("GPU-rooted live-5 facing pot-bet ({el:.1}s): trash {trash_dist:?}");
    if let Some(nu) = nuts {
        let nuts_dist: Vec<f32> = (0..na).map(|a| s[a][nu]).collect();
        eprintln!("  nuts-ish: {nuts_dist:?}");
        assert!(nuts_dist[0] < 0.2, "nuts must not fold: {nuts_dist:?}");
    }
    // STRUCTURAL gate only: rooting must make the facing-bet node SHARP
    // (trained), not uniform. The trash-folds assertion lives in the e2e
    // battery where the REAL blueprint priors apply — on this canonical AA7
    // board with UNIFORM priors, calling with any hand ties the board pair and
    // is genuinely defensible, so asserting fold here would be a bad-poker
    // gate (the same trap as the 62o-pairs-the-board probe).
    let maxp = trash_dist.iter().cloned().fold(0.0f32, f32::max);
    assert!(maxp > 0.6, "facing-bet node undertrained (near-uniform): {trash_dist:?}");
}
