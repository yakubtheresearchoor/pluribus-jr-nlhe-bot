//! V1 POSTFLOP POKER-SANITY (2026-06-15): decode a banked live-2 (.bp2) cell
//! and check that the flop strategy tracks hand strength — strong made hands
//! bet/value, air checks. Eyeball gate (does the postflop side look like
//! poker), not a numeric pin. Uses the exact HU solver's per-hand flop avg.

use clean_rules::eval::best5;
use play_harness::live2_bank::load_live2;
use play_harness::preflop_oracle::seeded_1x1;
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;

fn card_to_str(c: Card) -> String {
    let r = b"23456789TJQKA"[(c >> 2) as usize] as char;
    let s = b"cdhs"[(c & 3) as usize] as char;
    format!("{r}{s}")
}
use solver_core::solver::flop_start_game::FlopChanceTable;
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA_POSTFLOP;

#[test]
fn v1_postflop_sanity() {
    let spec = production_game_v1();
    let (commit, pot, fi) = (64i32, 128i32, 0usize); // live2/S0 rep
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let tree = build_tree(&spec.flop_seam_config(2, commit, pot, bets)).unwrap();

    let canon = PreflopChanceTable::new(6, vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6]).canonical_flops.clone();
    let _ = fi;
    // Pick a DRY, UNPAIRED, broadway flop (rank=c>>2): a clean spot where
    // bet-freq vs strength is informative (vs paired/trips boards where
    // slowplaying inverts it). Distinct ranks, top rank ≥ Q (idx ≥ 10).
    let fi = canon.iter().position(|b| {
        let r: Vec<u8> = b.iter().map(|&c| c >> 2).collect();
        let s: Vec<u8> = b.iter().map(|&c| c & 3).collect();
        let distinct_rank = r[0] != r[1] && r[1] != r[2] && r[0] != r[2];
        let rainbow = s[0] != s[1] && s[1] != s[2] && s[0] != s[2];
        // dry: top card broadway, no two ranks within 4 (unconnected)
        let mut rs = r.clone(); rs.sort();
        let unconnected = rs[1] - rs[0] >= 4 && rs[2] - rs[1] >= 4;
        distinct_rank && rainbow && *r.iter().max().unwrap() >= 10 && unconnected
    }).expect("a dry rainbow broadway flop");
    let board = canon[fi];
    let path = format!("{}/../blueprint_out_v1/live2/S0/flop_{fi:04}.bp2", env!("CARGO_MANIFEST_DIR"));
    let solver = load_live2(&path, board, fi, &tree).unwrap_or_else(|e| panic!("load {path}: {e}"));

    // Rebuild the identical table for the hand order.
    let (turns, rd) = seeded_1x1(board, fi);
    let table = FlopChanceTable::build_full_nh_sampled(board, 2, &turns, &rd);
    let nh = table.num_valid;

    // Flop root = first flop-zone player decision node that can BET (the open).
    eprintln!("flop player nodes (idx: action-labels):");
    for i in 0..tree.num_nodes() {
        if tree.nodes[i].is_player() && solver.flop_local_offset_at(i).is_some() {
            let l: Vec<u8> = tree.node_children(i).iter().map(|&c| tree.nodes[c as usize].action_label).collect();
            eprintln!("  {i}: {l:?}");
        }
    }
    // The OPEN: a flop node offering check (1) + an aggressive action (bet 3 or
    // all-in 5). At low SPR the aggressive line is a jam, not a pot-bet.
    let is_open = |i: usize| tree.nodes[i].is_player() && solver.flop_local_offset_at(i).is_some()
        && tree.node_children(i).iter().any(|&c| tree.nodes[c as usize].action_label == 1)
        && tree.node_children(i).iter().any(|&c| matches!(tree.nodes[c as usize].action_label, 3 | 5));
    let root = (0..tree.num_nodes()).find(|&i| is_open(i)).expect("a flop open node");
    let na = tree.nodes[root].num_children as usize;
    let labels: Vec<u8> = tree.node_children(root).iter().map(|&c| tree.nodes[c as usize].action_label).collect();
    let bet_a = labels.iter().position(|&l| matches!(l, 3 | 5)).unwrap();
    let local = solver.flop_local_offset_at(root).unwrap();
    let off = local * MAX_NA_POSTFLOP * nh;
    let cum = solver.cum_strategy_flop();

    eprintln!("\n=== live-2 postflop sanity | board {} {} {} | actions {labels:?} ===",
        card_to_str(board[0]), card_to_str(board[1]), card_to_str(board[2]));

    // Per-hand: flop bet frequency + made-hand rank on this board.
    let mut rows: Vec<(u32, f32, [Card; 2])> = Vec::with_capacity(nh);
    for h in 0..nh {
        let (c1, c2) = (table.hand_cards[h * 2], table.hand_cards[h * 2 + 1]);
        let sum: f32 = (0..na).map(|a| cum[off + a * nh + h].max(0.0)).sum();
        let bet_freq = if sum > 0.0 { cum[off + bet_a * nh + h].max(0.0) / sum } else { 0.0 };
        let rank = best5(&[c1, c2, board[0], board[1], board[2]]).0;
        rows.push((rank, bet_freq, [c1, c2]));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    let q = nh / 5;
    let top_bet: f32 = rows[..q].iter().map(|r| r.1).sum::<f32>() / q as f32;
    let bot_bet: f32 = rows[nh - q..].iter().map(|r| r.1).sum::<f32>() / q as f32;
    let all_bet: f32 = rows.iter().map(|r| r.1).sum::<f32>() / nh as f32;
    eprintln!("flop bet-freq: strongest 20% = {top_bet:.3} | weakest 20% = {bot_bet:.3} | overall = {all_bet:.3}");
    eprintln!("strongest 5 hands:");
    for (rk, bf, c) in rows.iter().take(5) { eprintln!("  {}{} rank{rk} bet={bf:.2}", card_to_str(c[0]), card_to_str(c[1])); }
    eprintln!("weakest 5 hands:");
    for (rk, bf, c) in rows.iter().rev().take(5) { eprintln!("  {}{} rank{rk} bet={bf:.2}", card_to_str(c[0]), card_to_str(c[1])); }
    assert!(all_bet > 0.001 && all_bet < 0.999, "degenerate flop bet freq {all_bet} (all-bet or all-check)");
}
