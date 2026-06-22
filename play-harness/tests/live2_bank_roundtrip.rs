//! LIVE-2 BANK ROUND-TRIP GATE: solve a live-2 spot with the rich M2 menu, save
//! the COMPRESSED flop-only blob (SSL3), load it back, and assert the flop cum
//! buffer round-trips BIT-IDENTICAL (zstd is lossless) while turn/river come back
//! zero (not stored — decide_live2 never queries them). Also reports the achieved
//! compression vs the raw flop f32. Proves the serializer before the full fill.

use play_harness::live2_bank::{live2_bet_menu, load_live2, save_live2_v2, solve_live2};
use solver_core::card::{card_from_str, Card};
use solver_core::tree::action::production_game_v1;
use solver_core::tree::builder::build_tree;

#[test]
fn live2_bank_roundtrip() {
    let spec = production_game_v1();
    let (commit, pot) = (10i32, 32i32);
    let tree = build_tree(&spec.flop_seam_config(2, commit, pot, live2_bet_menu())).unwrap();
    let canonical: [Card; 3] = ["Ks", "7d", "2c"].map(|s| card_from_str(s).unwrap());
    let fi = 17usize;

    let s = solve_live2(canonical, fi, &tree);
    let f0 = s.cum_strategy_flop().to_vec();
    assert!(f0.iter().any(|&v| v != 0.0), "flop cum all-zero — solve didn't accumulate");

    let path = format!("{}/live2_rt_m2.bp2", std::env::temp_dir().display());
    save_live2_v2(&path, &s, commit, pot, fi).unwrap();
    let sz = std::fs::metadata(&path).unwrap().len();

    let s2 = load_live2(&path, canonical, fi, &tree).unwrap();
    // Flop cum must be bit-exact (zstd lossless); turn/river are not stored → zero.
    assert_eq!(s2.cum_strategy_flop(), &f0[..], "flop cum mismatch after round-trip");
    assert!(
        s2.cum_strategy_turn().iter().all(|&v| v == 0.0),
        "turn cum should be zero in a flop-only blob"
    );
    assert!(
        s2.cum_strategy_river().iter().all(|&v| v == 0.0),
        "river cum should be zero in a flop-only blob"
    );

    let raw_flop = f0.len() * 4;
    eprintln!(
        "live-2 SSL3 round-trip OK: blob {:.1} KB vs raw flop f32 {:.1} KB ({:.1}× smaller) | flop cum len {}",
        sz as f64 / 1e3,
        raw_flop as f64 / 1e3,
        raw_flop as f64 / sz as f64,
        f0.len()
    );
    std::fs::remove_file(&path).ok();
}
