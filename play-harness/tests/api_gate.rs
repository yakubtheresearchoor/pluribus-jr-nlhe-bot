//! Smoke test for the runtime decision API: load a real blueprint cell, build a
//! flop DecideRequest, and check decide_postflop returns a sane action
//! distribution — solo and in pair mode (partner-card blocking).
//!
//! Run: BP_ROOT=$PWD/blueprint_out_v1 PAR=1 \
//!   cargo test --release -p play-harness --test api_gate -- --ignored --nocapture

use play_harness::api::{decide_postflop, ActionInput, CellKey, DecideRequest};
use play_harness::blueprint::Blueprint;
use play_harness::pluribus_play::SearchCfg;

#[test]
#[ignore = "needs blueprint_out_v1; --ignored --nocapture --release"]
fn api_decide_smoke() {
    let bp_root = std::env::var("BP_ROOT").unwrap_or_else(|_| "blueprint_out_v1".into());
    // lowest-commit live3 cell with a flop_0000.bp (high SPR ⇒ real bet sizing).
    let cell_dir = std::fs::read_dir(&bp_root)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("live3_"))
        .filter(|n| std::path::Path::new(&format!("{bp_root}/{n}/flop_0000.bp")).exists())
        .min_by_key(|n| {
            n.split('_').find_map(|t| t.strip_prefix('c')?.parse::<i32>().ok()).unwrap_or(9999)
        });
    let cell_dir = match cell_dir {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no live3 cell under {bp_root}");
            return;
        }
    };
    // parse commit/pot from the dir name
    let parse = |p: char| cell_dir.split('_').find_map(|t| t.strip_prefix(p)?.parse::<u32>().ok());
    let (commit, pot) = (parse('c').unwrap(), parse('p').unwrap());
    let bp = Blueprint::load(&format!("{bp_root}/{cell_dir}/flop_0000.bp")).unwrap();
    eprintln!("cell {cell_dir}: live={} commit={commit} pot={pot} flop={:?}", bp.np, bp.flop);

    let board: Vec<u8> = bp.flop.to_vec();
    let bmask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    // pick three non-conflicting hole pairs (hero, partner, pool) from the deck.
    let deck: Vec<u8> = (0..52u8).filter(|&c| bmask & (1u64 << c) == 0).collect();
    let hero = [deck[0], deck[1]];
    let partner = [deck[2], deck[3]];

    let cfg = SearchCfg { iters: 120, ..Default::default() };
    let cell = CellKey { live: bp.np as u8, commit, pot };

    // --- SOLO: hero first to act on the flop (no prior street action) ---
    let req = DecideRequest {
        board: board.clone(),
        hero_cards: hero,
        partner_cards: None,
        live: bp.np as u8,
        hero_idx: 0,
        partner_idx: None,
        commit_entry: commit,
        pot_entry: pot,
        street_actions: vec![],
        cell,
        seed: Some(0x1234),
    };
    let resp = decide_postflop(&bp, &req, &cfg).expect("solo decision");
    let z: f32 = resp.actions.iter().map(|a| a.prob).sum();
    eprintln!("SOLO {} live={}: {} actions, Σp={z:.3}, {}ms", resp.street, resp.live, resp.actions.len(), resp.search_ms);
    for a in &resp.actions {
        eprintln!("  {:<5} amt={:<4} p={:.3}", a.action, a.amount, a.prob);
    }
    eprintln!("  chosen: {} amt={}", resp.chosen.action, resp.chosen.amount);
    assert!((z - 1.0).abs() < 1e-3, "solo probs must sum to 1, got {z}");
    assert!(!resp.paired);

    // --- PAIR MODE: hero seat 0, partner seat 1, partner cards blocked from pool ---
    let req_pair = DecideRequest {
        board,
        hero_cards: hero,
        partner_cards: Some(partner),
        live: bp.np as u8,
        hero_idx: 0,
        partner_idx: Some(1),
        commit_entry: commit,
        pot_entry: pot,
        street_actions: vec![ActionInput { label: 1, to_total: commit }], // seat 1 checks
        cell,
        seed: Some(0x1234),
    };
    // (with one prior action by seat 1, the node should be hero=seat? — depends on
    // action order; if the mapped node isn't hero's, decide returns None. Use the
    // no-action form to guarantee hero-to-act, but exercise the blocker path.)
    let req_pair = DecideRequest { street_actions: vec![], ..req_pair };
    let resp_pair = decide_postflop(&bp, &req_pair, &cfg).expect("pair decision");
    let zp: f32 = resp_pair.actions.iter().map(|a| a.prob).sum();
    eprintln!("PAIR {} live={}: {} actions, Σp={zp:.3}, paired={}, {}ms", resp_pair.street, resp_pair.live, resp_pair.actions.len(), resp_pair.paired, resp_pair.search_ms);
    assert!((zp - 1.0).abs() < 1e-3, "pair probs must sum to 1, got {zp}");
    assert!(resp_pair.paired);
    eprintln!("OK: solo + pair decisions return valid action distributions.");
}
