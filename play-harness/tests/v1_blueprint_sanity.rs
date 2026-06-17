//! V1 BLUEPRINT POKER-SANITY (2026-06-15): decode the banked preflop.blob and
//! read the actual average opening strategy for recognizable hand classes. A
//! real solve must open premiums (AA/KK/AKs) near-always and fold trash
//! (72o/32o) near-always; mid hands in between. This is a "does it look like
//! poker" eyeball gate, not a numeric pin.
//!
//! Run: cargo test -p play-harness --release --test v1_blueprint_sanity -- --nocapture

use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::{card_from_str, Card};
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree_preflop_only;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

fn cap3_preflop_tree() -> FlatTree {
    let spec = production_game_v1();
    let mrc = MAX_NA_PREFLOP.saturating_sub(2);
    let mut cfg = spec.preflop_tree_config(BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect(),
    });
    cfg.max_bets_per_street = BetCap::all(3);
    build_tree_preflop_only(&cfg).expect("cap-3 preflop tree")
}

fn cls(a: &str, b: &str) -> usize {
    PreflopClass::from_combo(card_from_str(a).unwrap(), card_from_str(b).unwrap()).index()
}

#[test]
fn v1_blueprint_sanity() {
    let tree = cap3_preflop_tree();
    let mut solver = PreflopVectorCfr::new(&tree);
    let nc = NUM_PREFLOP_CLASSES;

    // Load the banked blob: magic line + header line + f32-LE cum payload.
    let path = format!("{}/../preflop_out_v1/preflop.blob", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let nl1 = bytes.iter().position(|&b| b == b'\n').unwrap();
    let nl2 = nl1 + 1 + bytes[nl1 + 1..].iter().position(|&b| b == b'\n').unwrap();
    eprintln!("header: {}", std::str::from_utf8(&bytes[nl1 + 1..nl2]).unwrap());
    let payload = &bytes[nl2 + 1..];
    let floats: Vec<f32> = payload.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(floats.len(), solver.cum_strategy.len(), "blob size != solver cum size");
    solver.cum_strategy.copy_from_slice(&floats);
    let avg = solver.average_strategy(&tree);

    // First few preflop decision nodes: print acting seat, action labels, and
    // the average strategy for recognizable classes. Action labels:
    // 0=fold 1=check 2=call 3=bet 4=raise 5=allin.
    let named: &[(&str, usize)] = &[
        ("AA ", cls("Ac", "Ad")), ("KK ", cls("Kc", "Kd")), ("QQ ", cls("Qc", "Qd")),
        ("JJ ", cls("Jc", "Jd")), ("TT ", cls("Tc", "Td")), ("55 ", cls("5c", "5d")),
        ("22 ", cls("2c", "2d")), ("AKs", cls("Ac", "Kc")), ("AKo", cls("Ac", "Kd")),
        ("AQs", cls("Ac", "Qc")), ("KQs", cls("Kc", "Qc")), ("T9s", cls("Tc", "9c")),
        ("QJo", cls("Qc", "Jd")), ("72o", cls("7c", "2d")), ("32o", cls("3c", "2d")),
        ("J4o", cls("Jc", "4d")),
    ];
    let lab = |l: u8| -> &'static str { match l { 0 => "fold", 1 => "check", 2 => "call", 3 => "bet", 4 => "raise", 5 => "allin", _ => "?" } };

    let mut shown = 0;
    for &nid in &tree.decision_node_ids {
        let idx = nid as usize;
        let local = solver.local_offset[idx];
        if local == usize::MAX { continue; }
        let na = tree.nodes[idx].num_children as usize;
        if na < 2 { continue; }
        let pl = tree.nodes[idx].player_id;
        let off = local * MAX_NA_PREFLOP * nc;
        let labels: Vec<&str> = tree.node_children(idx).iter().map(|&c| lab(tree.nodes[c as usize].action_label)).collect();
        eprintln!("\n── decision node {idx} | seat {pl} | actions: {labels:?} ──");
        eprintln!("  {:<4} {}", "hand", labels.iter().map(|l| format!("{l:>6}")).collect::<String>());
        for (name, ci) in named {
            let probs: Vec<String> = (0..na).map(|a| format!("{:>6.2}", avg[off + a * nc + ci])).collect();
            eprintln!("  {name}  {}", probs.join(""));
        }
        shown += 1;
        if shown >= 3 { break; }
    }
    assert!(shown > 0, "no decision nodes found");
}
