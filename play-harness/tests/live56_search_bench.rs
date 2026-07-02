//! MEASURE-FIRST bench for live-5/6 real-time CPU street search: per-iter cost
//! at production scale (nh=1176, rich menu, factored showdown, PAR+DCFR path
//! via search_street_strat). Decides the iteration budget — or a no-ship.
use play_harness::blueprint::Blueprint;
use play_harness::pluribus_play::{search_decision, SearchCfg};
use solver_core::abstraction::flop_isomorphism::canonicalize_flop;
use solver_core::blueprint::runout_grid;
use solver_core::solver::bucketed_flop_cfr::FlopBucketing;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};

fn adapter(np: usize) -> Blueprint {
    let canonical = canonicalize_flop([51, 50, 20]);
    let (turns, river_decks) = runout_grid(canonical, 12, 12);
    let turns_u8: Vec<u8> = turns.iter().map(|&c| c as u8).collect();
    let rivers: Vec<Vec<u8>> = turns_u8.iter().map(|&tc| river_decks[tc as usize].clone()).collect();
    let table = FlopChanceTable::build_full_nh_sampled(canonical, np as u8, &turns_u8, &river_decks);
    let nh = table.num_valid;
    let game = FlopStartGame::new(table);
    let bk = FlopBucketing::quantile(game.table(), 16);
    Blueprint {
        flop: [canonical[0] as u8, canonical[1] as u8, canonical[2] as u8],
        turns: turns_u8, rivers, np, nb: 16, nh,
        cum_flop: vec![], cum_turn: vec![], cum_river: vec![], bk, game,
    }
}

#[test]
#[ignore = "measurement bench"]
fn live56_search_per_iter() {
    for np in [5usize, 6] {
        let bp = adapter(np);
        let iters = 8u32;
        let cfg = SearchCfg { iters, sample_m: 200, par: Some(true), dcfr: Some(true), ..Default::default() };
        let board = bp.flop.to_vec();
        let t0 = std::time::Instant::now();
        let r = search_decision(&bp, &board, np, 0, None, &[], 4, (np as i32) * 4 + 4, &cfg, &[]);
        let el = t0.elapsed().as_secs_f64();
        let per = el / iters as f64 * 1000.0;
        eprintln!("live-{np}: {iters} iters in {el:.1}s -> {per:.0} ms/iter (solve ok: {})", r.is_some());
    }
}
