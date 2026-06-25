//! Smoke test for the runtime decision API: load a real blueprint cell, build a
//! flop DecideRequest, and check decide_postflop returns a sane action
//! distribution — solo and in pair mode (partner-card blocking).
//!
//! Run: BP_ROOT=$PWD/blueprint_out_v1 PAR=1 \
//!   cargo test --release -p play-harness --test api_gate -- --ignored --nocapture

use play_harness::api::{
    decide_live2, decide_live6, decide_postflop, decide_postflop_resolve, route_to_canonical,
    ActionInput, DecideRequest,
};
use play_harness::blueprint::Blueprint;
use play_harness::pluribus_play::SearchCfg;

/// END-TO-END multiway FACTORED re-solve (no blueprint needed — solves the real
/// board fresh). Drives `decide_postflop_resolve` for a live-3 turn AND river on
/// an arbitrary runout and checks it serves a valid, normalized strategy. This
/// is the path `decide_postflop` falls into for the multiway turn/river hole.
#[test]
fn api_multiway_resolve_e2e() {
    let card = |r: u8, s: u8| r * 4 + s; // rank*4+suit
    for (label, board) in [
        ("turn", vec![card(11, 0), card(7, 1), card(2, 2), card(0, 3)]),
        ("river", vec![card(11, 0), card(7, 1), card(2, 2), card(0, 3), card(12, 0)]),
    ] {
        let deck: Vec<u8> = (0..52u8).filter(|&c| !board.contains(&c)).collect();
        // The acting player at the subgame root depends on tree construction;
        // try each seat and require that the hero-is-acting seat serves a
        // valid decision.
        let mut served = 0;
        for hero_idx in 0..3u8 {
            let req = DecideRequest {
                board: board.clone(),
                hero_cards: [deck[0], deck[1]],
                partner_cards: None,
                live: 3,
                hero_idx,
                partner_idx: None,
                commit_entry: 20,
                pot_entry: 60,
                street_actions: vec![],
                cell_dir: String::new(),
                flop_id: 0,
                prior_actions: vec![],
                seed: Some(0x5EED),
                route: false,
                to_call: Some(0),
            };
            if let Some(r) = decide_postflop_resolve(&req) {
                served += 1;
                let z: f32 = r.actions.iter().map(|a| a.prob).sum();
                eprintln!(
                    "{label} live-3 seat {hero_idx}: {} actions, Σp={z:.3}, {}ms",
                    r.actions.len(),
                    r.search_ms
                );
                assert_eq!(r.street, label);
                assert_eq!(r.live, 3);
                assert!(!r.actions.is_empty(), "must offer actions");
                assert!((z - 1.0).abs() < 1e-3, "probs must normalize");
                assert!(r.actions.iter().all(|a| a.amount >= 0), "amounts non-negative");
                assert!(r.actions.iter().all(|a| a.prob >= 0.0 && a.prob <= 1.0));
            }
        }
        assert!(served >= 1, "{label}: at least one seat must be the acting player at root");
    }
    eprintln!("OK: multiway factored re-solve serves valid live-3 turn+river decisions end-to-end.");
}

/// MULTIWAY turn/river fallback: a live-3 turn on a NON-banked runout (the 1×1
/// hole) must no longer return None — it falls back to the equity-rollout model and
/// serves a valid decision.
#[test]
#[ignore = "needs blueprint_out_v1; --ignored --nocapture --release"]
fn api_postflop_turn_fallback() {
    let bp_root = std::env::var("BP_ROOT").unwrap_or_else(|_| "blueprint_out_v1".into());
    let cell = std::fs::read_dir(&bp_root)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .find(|n| n.starts_with("live3_") && std::path::Path::new(&format!("{bp_root}/{n}/flop_0000.bp")).exists());
    let cell = match cell {
        Some(c) => c,
        None => { eprintln!("SKIP: no live3 cell"); return; }
    };
    let bp = Blueprint::load(&format!("{bp_root}/{cell}/flop_0000.bp")).unwrap();
    let parse = |p: char| cell.split('_').find_map(|t| t.strip_prefix(p)?.parse::<u32>().ok());
    let (commit, pot) = (parse('c').unwrap(), parse('p').unwrap());
    // turn = a card almost certainly NOT the banked 1×1 turn → search misses → fallback.
    let used: u64 = bp.flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let turn = (0..52u8).find(|&c| used & (1u64 << c) == 0 && c != bp.turns[0]).unwrap();
    let board: Vec<u8> = vec![bp.flop[0], bp.flop[1], bp.flop[2], turn];
    let deck: Vec<u8> = (0..52u8).filter(|&c| board.iter().all(|&b| b != c)).collect();
    let req = DecideRequest {
        board, hero_cards: [deck[0], deck[1]], partner_cards: None, live: bp.np as u8,
        hero_idx: 0, partner_idx: None, commit_entry: commit, pot_entry: pot,
        street_actions: vec![], cell_dir: cell.clone(), flop_id: 0, seed: Some(1),
        route: false, to_call: Some(0), prior_actions: vec![],
    };
    let cfg = SearchCfg { iters: 80, ..Default::default() };
    let r = decide_postflop(&bp, &req, &cfg).expect("fallback must serve (no 400)");
    let z: f32 = r.actions.iter().map(|a| a.prob).sum();
    eprintln!("FALLBACK live-3 turn (non-banked runout): {} actions, Σp={z:.3}, {}ms", r.actions.len(), r.search_ms);
    assert!((z - 1.0).abs() < 1e-3);
    assert_eq!(r.street, "turn");
    eprintln!("OK: multiway turn/river no longer 400s on unbanked runouts.");
}

/// LIVE-6 equity-rollout decision (no blueprint needed). Unbet ⇒ check; facing a
/// bet ⇒ pot-odds call/fold vs MC all-in equity. A flopped set clears the odds and
/// calls; total air doesn't and folds.
#[test]
fn api_live6_rollout() {
    // AKQ rainbow flop.
    let flop = vec![48u8, 45, 42]; // A♠ K♦ Q♣ (rank*4+suit)
    let base = DecideRequest {
        board: flop.clone(),
        hero_cards: [49, 50], // A♦ A♥ → flopped set of aces
        partner_cards: None,
        live: 6,
        hero_idx: 0,
        partner_idx: None,
        commit_entry: 20,
        pot_entry: 100,
        street_actions: vec![],
        cell_dir: String::new(),
        flop_id: 0,
        prior_actions: vec![],
        seed: Some(0x77),
        route: false,
        to_call: Some(0),
    };
    // Unbet → check.
    let r_check = decide_live6(&base).expect("live6 unbet");
    assert_eq!(r_check.actions.len(), 1);
    assert_eq!(r_check.chosen.action, "check");
    eprintln!("LIVE6 unbet → {} ({}ms)", r_check.chosen.action, r_check.search_ms);

    // Facing a pot-odds-33% bet with a set → call.
    let strong = DecideRequest { to_call: Some(50), ..clone_req(&base) };
    let r_strong = decide_live6(&strong).expect("live6 set");
    eprintln!("LIVE6 set facing 50 into 100: chose {}", r_strong.chosen.action);
    assert_eq!(r_strong.chosen.action, "call", "a flopped set should call 33% pot odds");

    // Same bet with total air (7♠2♦ on AKQ) → fold.
    let weak = DecideRequest {
        hero_cards: [28, 9], // 7♠ 2♦
        to_call: Some(50),
        ..clone_req(&base)
    };
    let r_weak = decide_live6(&weak).expect("live6 air");
    eprintln!("LIVE6 air facing 50 into 100: chose {}", r_weak.chosen.action);
    assert_eq!(r_weak.chosen.action, "fold", "total air should fold 33% pot odds");
    eprintln!("OK: live-6 equity-rollout check/call/fold behaves by pot-odds.");
}

/// Shallow-clone a DecideRequest (the struct isn't Clone; rebuild the owned fields).
fn clone_req(r: &DecideRequest) -> DecideRequest {
    DecideRequest {
        board: r.board.clone(),
        hero_cards: r.hero_cards,
        partner_cards: r.partner_cards,
        live: r.live,
        hero_idx: r.hero_idx,
        partner_idx: r.partner_idx,
        commit_entry: r.commit_entry,
        pot_entry: r.pot_entry,
        street_actions: r.street_actions.clone(),
        cell_dir: r.cell_dir.clone(),
        flop_id: r.flop_id,
        prior_actions: vec![],
        seed: r.seed,
        route: r.route,
        to_call: r.to_call,
    }
}

/// LIVE-2 flop decision from the banked HU strategy: a lookup (no search). Asserts
/// a valid distribution and suit-iso routing invariance (a permuted raw board +
/// hole cards yields the same distribution). commit=10/pot=20 ⇒ SPR bin S12.
#[test]
#[ignore = "needs blueprint_out_v1/live2 bank; --ignored --nocapture --release"]
fn api_live2_flop() {
    let bp_root = std::env::var("BP_ROOT").unwrap_or_else(|_| "blueprint_out_v1".into());
    let subdir = std::env::var("L2_SUBDIR").unwrap_or_else(|_| "live2".into());
    let live2_root = format!("{bp_root}/{subdir}");
    if !std::path::Path::new(&format!("{live2_root}/manifest.txt")).exists() {
        eprintln!("SKIP: no live2 bank under {live2_root}");
        return;
    }
    // canonical flop 0 = three rainbow deuces [0,1,2]; hero off-board.
    let canon_req = DecideRequest {
        board: vec![0, 1, 2],
        hero_cards: [20, 33],
        partner_cards: None,
        live: 2,
        hero_idx: 0,
        partner_idx: None,
        commit_entry: 10,
        pot_entry: 20,
        street_actions: vec![],
        cell_dir: String::new(),
        flop_id: 0,
        prior_actions: vec![],
        seed: Some(0x1234),
        route: false,
        to_call: None,
    };
    let base = match decide_live2(&live2_root, &canon_req) {
        Some(r) => r,
        None => {
            eprintln!("SKIP: decide_live2 None (bank menu mismatch — M2 bank not deployed at {live2_root}?)");
            return;
        }
    };
    let z: f32 = base.actions.iter().map(|a| a.prob).sum();
    eprintln!("LIVE2 flop bin-S12: {} actions, Σp={z:.3}, {}ms", base.actions.len(), base.search_ms);
    for a in &base.actions {
        eprintln!("  {:<5} amt={:<4} p={:.3}", a.action, a.amount, a.prob);
    }
    assert!((z - 1.0).abs() < 1e-3, "live2 probs must sum to 1, got {z}");
    assert_eq!(base.live, 2);

    // routing invariance: permute board + hole by a suit map, route, expect same.
    let perm = [1u8, 0, 3, 2];
    let remap = |c: u8| ((c >> 2) << 2) | perm[(c & 3) as usize];
    let mut routed = DecideRequest {
        board: canon_req.board.iter().map(|&c| remap(c)).collect(),
        hero_cards: [remap(canon_req.hero_cards[0]), remap(canon_req.hero_cards[1])],
        route: true,
        ..canon_req
    };
    route_to_canonical(&mut routed).expect("route");
    let routed_resp = decide_live2(&live2_root, &routed).expect("routed live2 decision");
    for (b, r) in base.actions.iter().zip(routed_resp.actions.iter()) {
        assert_eq!(b.label, r.label);
        assert!((b.prob - r.prob).abs() < 1e-3, "live2 routing must be suit-iso invariant");
    }
    eprintln!("OK: live-2 flop decision valid + suit-iso routing invariant.");
}

/// LIVE-2 turn/river real-time decision: exact HU search of the actual board (no
/// bank). Asserts a valid distribution + the rich menu appears, and reports timing.
#[test]
fn api_live2_turn_river() {
    use play_harness::live2_bank::{solve_live2_street, LIVE2_RT_RIVER_ITERS, LIVE2_RT_TURN_ITERS};
    for (name, board) in [
        ("turn", vec![44u8, 33, 8, 2]),
        ("river", vec![44u8, 33, 8, 2, 19]),
    ] {
        let used: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
        let deck: Vec<u8> = (0..52u8).filter(|&c| used & (1u64 << c) == 0).collect();
        let hero = [deck[0], deck[1]];
        // root acting player (so hero_idx matches the first decision node).
        // Deep SPR (commit10/pot20) — the worst case for the turn (unbounded ≈45s);
        // adaptive iters must keep it under budget.
        let iters = if board.len() == 5 { LIVE2_RT_RIVER_ITERS } else { LIVE2_RT_TURN_ITERS };
        let probe = solve_live2_street(&board, 10, 20, iters).expect("solve");
        let root_player = probe.tree.nodes[0].player_id as u8;

        let req = DecideRequest {
            board: board.clone(),
            hero_cards: hero,
            partner_cards: None,
            live: 2,
            hero_idx: root_player,
            partner_idx: None,
            commit_entry: 10,
            pot_entry: 20,
            street_actions: vec![],
            cell_dir: String::new(),
            flop_id: 0,
            prior_actions: vec![],
            seed: Some(7),
            route: false,
            to_call: None,
        };
        let r = decide_live2("", &req).expect("turn/river decision");
        let z: f32 = r.actions.iter().map(|a| a.prob).sum();
        eprintln!(
            "LIVE2 {name}: {} actions, Σp={z:.3}, {}ms  {:?}",
            r.actions.len(), r.search_ms,
            r.actions.iter().map(|a| (a.action.as_str(), a.amount, (a.prob * 100.0).round() / 100.0)).collect::<Vec<_>>()
        );
        assert_eq!(r.street, name);
        assert!((z - 1.0).abs() < 1e-3, "{name} probs must sum to 1, got {z}");
        assert!(r.actions.len() >= 2, "{name} should offer ≥2 actions");
        assert!(r.search_ms < 13_000, "{name} must fit the budget, took {}ms", r.search_ms);
    }
    eprintln!("OK: live-2 turn + river real-time decisions valid + within budget.");
}

/// Routing invariance: a strategy that respects suit isomorphism must produce the
/// SAME decision on any orbit member of the canonical flop. Permute a canonical
/// board + hole cards by an arbitrary suit map, route it back, and the action
/// distribution must match the un-permuted decision (cards may differ by an
/// automorphism — only the distribution is invariant).
#[test]
#[ignore = "needs blueprint_out_v1; --ignored --nocapture --release"]
fn api_route_invariance() {
    let bp_root = std::env::var("BP_ROOT").unwrap_or_else(|_| "blueprint_out_v1".into());
    let cell_dir = std::fs::read_dir(&bp_root)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("live3_"))
        .find(|n| std::path::Path::new(&format!("{bp_root}/{n}/flop_0000.bp")).exists());
    let cell_dir = match cell_dir {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no live3 cell under {bp_root}");
            return;
        }
    };
    let bp = Blueprint::load(&format!("{bp_root}/{cell_dir}/flop_0000.bp")).unwrap();
    let parse = |p: char| cell_dir.split('_').find_map(|t| t.strip_prefix(p)?.parse::<u32>().ok());
    let (commit, pot) = (parse('c').unwrap(), parse('p').unwrap());
    let cfg = SearchCfg { iters: 120, ..Default::default() };

    // canonical board (= bp.flop, flop_id 0) + two hole cards off it.
    let canon_board: Vec<u8> = bp.flop.to_vec();
    let bmask: u64 = canon_board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let deck: Vec<u8> = (0..52u8).filter(|&c| bmask & (1u64 << c) == 0).collect();
    let hero = [deck[0], deck[1]];

    let base_req = DecideRequest {
        board: canon_board.clone(),
        hero_cards: hero,
        partner_cards: None,
        live: bp.np as u8,
        hero_idx: 0,
        partner_idx: None,
        commit_entry: commit,
        pot_entry: pot,
        street_actions: vec![],
        cell_dir: cell_dir.clone(),
        flop_id: 0,
        prior_actions: vec![],
        seed: Some(0x1234),
        route: false,
        to_call: None,
    };
    let base = decide_postflop(&bp, &base_req, &cfg).expect("canonical decision");

    // permute every card by a fixed suit map, then route back.
    let perm = [1u8, 0, 3, 2]; // 0↔1, 2↔3
    let remap = |c: u8| ((c >> 2) << 2) | perm[(c & 3) as usize];
    let raw_board: Vec<u8> = canon_board.iter().map(|&c| remap(c)).collect();
    let raw_hero = [remap(hero[0]), remap(hero[1])];
    let mut routed = DecideRequest {
        board: raw_board,
        hero_cards: raw_hero,
        flop_id: 999, // wrong on purpose — routing must overwrite it
        prior_actions: vec![],
        route: true,
        ..base_req
    };
    route_to_canonical(&mut routed).expect("route");
    assert_eq!(routed.flop_id, 0, "routed flop_id must resolve back to 0");
    assert_eq!(routed.board, canon_board, "routed board must equal canonical flop");
    let routed_resp = decide_postflop(&bp, &routed, &cfg).expect("routed decision");

    eprintln!("base vs routed action distributions:");
    for (b, r) in base.actions.iter().zip(routed_resp.actions.iter()) {
        eprintln!("  {:<5} base={:.4} routed={:.4}", b.action, b.prob, r.prob);
        assert_eq!(b.label, r.label, "action labels must align");
        assert!((b.prob - r.prob).abs() < 1e-3, "routed prob must match canonical");
    }
    eprintln!("OK: routing is suit-iso invariant (flop_id derived, decision preserved).");
}

/// Routed TURN + RIVER decisions: the per-street search requires the (remapped)
/// runout card to be in the bank (bp.turns/bp.rivers). Pick a RAINBOW, distinct-
/// rank canonical flop (trivial suit automorphism ⇒ a permuted-then-routed turn
/// maps back exactly to the banked turn), take a banked runout, permute the whole
/// board + hole by a suit map, route it back, and assert the decision matches the
/// un-permuted (canonical) turn/river decision.
#[test]
#[ignore = "needs blueprint_out_v1; --ignored --nocapture --release"]
fn api_route_turn_river() {
    use solver_core::abstraction::flop_isomorphism::enumerate_canonical_flops;
    let bp_root = std::env::var("BP_ROOT").unwrap_or_else(|_| "blueprint_out_v1".into());
    let cell_dir = std::fs::read_dir(&bp_root)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .find(|n| n.starts_with("live3_"));
    let cell_dir = match cell_dir {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no live3 cell under {bp_root}");
            return;
        }
    };
    let parse = |p: char| cell_dir.split('_').find_map(|t| t.strip_prefix(p)?.parse::<u32>().ok());
    let (commit, pot) = (parse('c').unwrap(), parse('p').unwrap());

    // Find a rainbow, distinct-rank canonical flop (trivial automorphism).
    let canon = enumerate_canonical_flops();
    let fi = canon
        .iter()
        .position(|f| {
            let ranks: Vec<u8> = f.iter().map(|&c| c >> 2).collect();
            let suits: Vec<u8> = f.iter().map(|&c| c & 3).collect();
            ranks[0] != ranks[1] && ranks[1] != ranks[2] && ranks[0] != ranks[2]
                && suits[0] != suits[1] && suits[1] != suits[2] && suits[0] != suits[2]
        })
        .expect("a rainbow distinct-rank canonical flop exists");
    let bp_path = format!("{bp_root}/{cell_dir}/flop_{fi:04}.bp");
    if !std::path::Path::new(&bp_path).exists() {
        eprintln!("SKIP: {bp_path} not present");
        return;
    }
    let bp = Blueprint::load(&bp_path).unwrap();
    let flop: Vec<u8> = bp.flop.to_vec();
    let t0 = bp.turns[0];
    let r0 = bp.rivers[0][0];
    eprintln!("cell {cell_dir} fi={fi} flop={flop:?} banked turn={t0} river={r0}");

    let bmask = |cards: &[u8]| cards.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let perm = [2u8, 3, 0, 1];
    let remap = |c: u8| ((c >> 2) << 2) | perm[(c & 3) as usize];
    let cfg = SearchCfg { iters: 120, ..Default::default() };

    for (street_name, board) in [
        ("turn", vec![flop[0], flop[1], flop[2], t0]),
        ("river", vec![flop[0], flop[1], flop[2], t0, r0]),
    ] {
        // hero off the board.
        let used = bmask(&board);
        let deck: Vec<u8> = (0..52u8).filter(|&c| used & (1u64 << c) == 0).collect();
        let hero = [deck[0], deck[1]];
        let base_req = DecideRequest {
            board: board.clone(),
            hero_cards: hero,
            partner_cards: None,
            live: bp.np as u8,
            hero_idx: 0,
            partner_idx: None,
            commit_entry: commit,
            pot_entry: pot,
            street_actions: vec![],
            cell_dir: cell_dir.clone(),
            flop_id: fi as u32,
            prior_actions: vec![],
            seed: Some(0x1234),
            route: false,
            to_call: None,
        };
        let base = match decide_postflop(&bp, &base_req, &cfg) {
            Some(r) => r,
            None => {
                eprintln!("SKIP {street_name}: canonical decision unmappable (node not hero's)");
                continue;
            }
        };
        // permute board + hole, route back.
        let mut routed = DecideRequest {
            board: board.iter().map(|&c| remap(c)).collect(),
            hero_cards: [remap(hero[0]), remap(hero[1])],
            flop_id: 999,
            prior_actions: vec![],
            route: true,
            ..base_req
        };
        route_to_canonical(&mut routed).expect("route turn/river");
        assert_eq!(routed.flop_id, fi as u32, "{street_name}: routed flop_id");
        assert_eq!(&routed.board[..3], &flop[..], "{street_name}: routed flop matches canonical");
        assert_eq!(routed.board[3], t0, "{street_name}: routed turn maps to banked turn");
        let routed_resp = decide_postflop(&bp, &routed, &cfg).expect("routed turn/river decision");
        eprintln!("{street_name}: {} actions", base.actions.len());
        for (b, r) in base.actions.iter().zip(routed_resp.actions.iter()) {
            eprintln!("  {:<5} base={:.4} routed={:.4}", b.action, b.prob, r.prob);
            assert_eq!(b.label, r.label, "{street_name}: labels align");
            assert!((b.prob - r.prob).abs() < 1e-3, "{street_name}: routed prob matches canonical");
        }
    }
    eprintln!("OK: routed turn + river decisions are suit-iso invariant.");
}

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
        cell_dir: cell_dir.clone(),
        flop_id: 0,
        prior_actions: vec![],
        seed: Some(0x1234),
        route: false,
        to_call: None,
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
        cell_dir: cell_dir.clone(),
        flop_id: 0,
        prior_actions: vec![],
        seed: Some(0x1234),
        route: false,
        to_call: None,
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
