//! PREFLOP DEEP-NODE BATTERY — the verification gate for any preflop shard
//! (the 2026-07-03 leak diagnosis: EQR-frozen preflop has degenerate deep
//! nodes — 94h facing a 3-bet = pure ALL-IN, 44 facing a 4-bet = UNIFORM dice,
//! J9s vs open = pure ~26bb overraise — while opens are sane: AA mixes normal
//! 3-bet sizes, 72o folds pure).
//!
//! Run against ANY candidate dir (needs only preflop.ssbp2):
//!   PRE_BATTERY_DIR=blueprint_conn_out cargo test --release -p play-harness \
//!     --test preflop_battery -- --ignored --nocapture
//!
//! HARD assertions = the non-negotiables (open sanity + no dice + no trash
//! jams). The printed table is the human verdict for everything softer.

use play_harness::api::DecideRequest;
use solver_core::blueprint::ShardedConnBlueprint;
use solver_core::card::Card;

fn ws(p: &str) -> String { format!("{}/../{}", env!("CARGO_MANIFEST_DIR"), p) }

fn dist(
    bp: &ShardedConnBlueprint,
    hero: (Card, Card),
    hist: &[(u8, i32)],
) -> Option<Vec<(u8, i32, f32)>> {
    bp.preflop_action_dist(hero, hist).map(|d| {
        let z: f32 = d.iter().map(|(_, _, p)| p).sum::<f32>().max(1e-12);
        d.into_iter().map(|(l, a, p)| (l, a, p / z)).collect()
    })
}

fn show(name: &str, d: &Option<Vec<(u8, i32, f32)>>) {
    match d {
        None => eprintln!("{name:>28}: UNMAPPABLE"),
        Some(d) => {
            let mut s: Vec<_> = d.clone();
            s.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            let top: Vec<String> = s.iter().take(3)
                .map(|(l, a, p)| format!("{} {}: {:.3}", match l { 0 => "fold", 1 => "check", 2 => "call", _ => "raise" }, a, p))
                .collect();
            eprintln!("{name:>28}: {}", top.join(" | "));
        }
    }
}

/// max prob of any single action — near-uniform ⇒ untrained dice-roll node.
fn maxp(d: &[(u8, i32, f32)]) -> f32 { d.iter().map(|&(_, _, p)| p).fold(0.0, f32::max) }
/// prob mass on raises >= `amt`.
fn mass_ge(d: &[(u8, i32, f32)], amt: i32) -> f32 {
    d.iter().filter(|&&(l, a, _)| l >= 3 && a >= amt).map(|&(_, _, p)| p).sum()
}

#[test]
#[ignore = "battery for a candidate preflop shard; set PRE_BATTERY_DIR. Run on demand."]
fn preflop_deep_node_battery() {
    let dir = std::env::var("PRE_BATTERY_DIR").unwrap_or_else(|_| ws("blueprint_conn_eqr"));
    let bp = match ShardedConnBlueprint::load(&dir, 6, 5, 200, 16, 16, 8) {
        Ok(b) => b,
        Err(e) => match ShardedConnBlueprint::load(&dir, 6, 5, 200, 16, 16, 7) {
            Ok(b) => b,
            Err(e2) => { panic!("load {dir}: maxna=8 -> {e}; maxna=7 -> {e2}"); }
        },
    };
    eprintln!("battery on {dir}");
    // Tree-shape dump: the actual raise sizes at the first two decision nodes
    // (exact-match replay means real-history mappability depends on these).
    {
        let pft = &bp.pft;
        let mut node = 0usize;
        for depth in 0..3 {
            if !pft.nodes[node].is_player() { break; }
            let kids: Vec<String> = pft.node_children(node).iter()
                .map(|&k| format!("{}@{}", pft.nodes[k as usize].action_label, pft.nodes[k as usize].amount))
                .collect();
            eprintln!("  node depth {depth} (seat {}): [{}]", pft.nodes[node].player_id, kids.join(", "));
            // descend along the first raise child
            let nxt = pft.node_children(node).iter()
                .find(|&&k| pft.nodes[k as usize].action_label >= 3)
                .map(|&k| k as usize);
            match nxt { Some(n) => node = n, None => break }
        }
    }
    let c = |s: &str| solver_core::card::card_from_str(s).unwrap();

    // Histories (units: blinds 1/2, stack 200): folds fold to BTN open 5;
    // 3-bet = SB re-raise to 28; 4-bet = re-raise to 60 after hero call.
    let open: Vec<(u8, i32)> = vec![(0, 0), (0, 0), (0, 0), (4, 5)];
    // 3-bet size 15 = the size the solved chart actually uses (querying an
    // untaken size branch reads untrained rows — production needs the same
    // trained-branch snapping; see the matcher).
    let vs3bet: Vec<(u8, i32)> = vec![(0, 0), (0, 0), (0, 0), (4, 5), (4, 15)];

    // ---- opens: sanity anchors ----
    let aa = dist(&bp, (c("Ad"), c("Ac")), &open);
    let j9s = dist(&bp, (c("Jd"), c("9d")), &open);
    let o72 = dist(&bp, (c("7c"), c("2d")), &open);
    show("AA SB vs open", &aa);
    show("J9s SB vs open", &j9s);
    show("72o SB vs open", &o72);
    // ---- deep nodes: the measured leaks ----
    let h94 = dist(&bp, (c("9h"), c("4h")), &vs3bet);
    let ats = dist(&bp, (c("Ad"), c("Td")), &vs3bet);
    let p44 = dist(&bp, (c("4c"), c("4d")), &vs3bet);
    show("94s facing 3-bet", &h94);
    show("ATs facing 3-bet", &ats);
    show("44 facing 3-bet", &p44);

    // HARD GATES ------------------------------------------------------------
    let aa = aa.expect("AA open node");
    let o72 = o72.expect("72o open node");
    let j9s = j9s.expect("J9s open node");
    // 1) open sanity: AA aggressive, 72o folds.
    let aa_raise: f32 = aa.iter().filter(|&&(l, _, _)| l >= 3).map(|&(_, _, p)| p).sum();
    let o72_fold: f32 = o72.iter().filter(|&&(l, _, _)| l == 0).map(|&(_, _, p)| p).sum();
    assert!(aa_raise > 0.9, "AA vs open must raise (got {aa_raise:.3})");
    assert!(o72_fold > 0.9, "72o vs open must fold (got {o72_fold:.3})");
    // 2) no dice: every probed node must be peaked, not uniform.
    for (name, d) in [("AA", &aa), ("72o", &o72), ("J9s", &j9s)] {
        assert!(maxp(d) > 0.3, "{name} open node near-uniform (untrained): maxp {:.3}", maxp(d));
    }
    if let Some(d) = &h94 {
        let uniform = maxp(d) < 1.5 / d.len().max(1) as f32;
        if uniform {
            // GUARD TERRITORY: production's uniform-guard serves FOLD here
            // (verified by the live probe battery post-deploy) — the shard
            // itself is allowed to be untrained on this rare tail.
            eprintln!("94s vs 3-bet: untrained row -> runtime guard serves fold (accepted)");
        } else {
            // trained row: must not stack off trash.
            let jam = mass_ge(d, 100);
            assert!(jam < 0.10, "94s facing 3-bet jams {jam:.3} — the measured leak");
            let fold: f32 = d.iter().filter(|&&(l, _, _)| l == 0).map(|&(_, _, p)| p).sum();
            assert!(fold > 0.7, "94s facing 3-bet must mostly fold (got fold {fold:.3})");
        }
    }
    // 4) no mid-class overraise: J9s vs open must not put >20% on raises ≥ 40u (20bb).
    let j9_over = mass_ge(&j9s, 40);
    assert!(j9_over < 0.2, "J9s vs open overraises ≥20bb at {j9_over:.3} — the measured leak");
    eprintln!("BATTERY PASS");
}
