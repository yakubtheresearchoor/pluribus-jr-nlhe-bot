//! LIVE-2 BANK ROUND-TRIP GATE (2026-06-15): solve a live-2 spot, save the
//! strategy blob, load it into a fresh solver, and assert the cum buffers come
//! back bit-identical and the average strategy matches. Proves the serializer
//! before the full fill runs.

use play_harness::live2_bank::{load_live2, save_live2, solve_live2};
use solver_core::card::{card_from_str, Card};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

#[test]
fn live2_bank_roundtrip() {
    let spec = production_game_v1();
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let (commit, pot) = (10i32, 32i32);
    let tree = build_tree(&spec.flop_seam_config(2, commit, pot, bets)).unwrap();
    let canonical: [Card; 3] = ["Ks", "7d", "2c"].map(|s| card_from_str(s).unwrap());
    let fi = 17usize;

    let s = solve_live2(canonical, fi, &tree);
    let (f0, t0, r0) = (
        s.cum_strategy_flop().to_vec(),
        s.cum_strategy_turn().to_vec(),
        s.cum_strategy_river().to_vec(),
    );
    assert!(f0.iter().any(|&v| v != 0.0), "flop cum all-zero — solve didn't accumulate");

    let path = format!("{}/live2_rt.bp2", std::env::temp_dir().display());
    save_live2(&path, &s, commit, pot, fi).unwrap();
    let sz = std::fs::metadata(&path).unwrap().len();

    let s2 = load_live2(&path, canonical, fi, &tree).unwrap();
    assert_eq!(s2.cum_strategy_flop(), &f0[..], "flop cum mismatch after round-trip");
    assert_eq!(s2.cum_strategy_turn(), &t0[..], "turn cum mismatch after round-trip");
    assert_eq!(s2.cum_strategy_river(), &r0[..], "river cum mismatch after round-trip");

    eprintln!(
        "live-2 round-trip OK: blob {:.2} KB | cum lens f/t/r = {}/{}/{}",
        sz as f64 / 1e3, f0.len(), t0.len(), r0.len()
    );
    std::fs::remove_file(&path).ok();
}
