//! GPU HU turn solve (#12) regression: on the real production seam config, the
//! GPU depth-limited (per-hand continuation) turn solve must VALUE-BET the nuts
//! — quad aces first-to-act on the turn bets ~always (not check down). This is
//! the fast, fully-converged alternative to the budget-capped CPU exact solve.
//!
//! The exact ground truth (real per-hand river showdown) mixes ~78% bet / ~22%
//! trap; the GPU over-bets (~100%, the continuation leaf drops the river betting
//! round so there's no future street to trap for) — an understood approximation,
//! not the old check-down defect. Also asserts the GPU solve fits the ~14s budget.

#![cfg(feature = "metal")]

use std::time::Instant;

use solver_core::card::{card_from_str, index_to_card_pair};
use solver_core::gpu_metal::{gpu_hu_turn_strat, GpuSearchCfg, MetalContext};
use solver_core::tree::action::{BoardState, production_game_v1};
use solver_core::tree::builder::build_tree_depth_limited;

use play_harness::live2_bank::live2_bet_menu;

fn card(s: &str) -> u8 { card_from_str(s).unwrap() as u8 }

#[test]
fn gpu_hu_turn_quads_value_bets() {
    // Quad aces on the turn: As Ah 7c 2d; hero = Ac Ad.
    let board = [card("As"), card("Ah"), card("7c"), card("2d")];
    let (commit, pot) = (20i32, 40i32);

    // Real production seam config (same game the runtime solves) — depth-limited
    // to the turn (river deal = continuation leaf).
    let cfg = production_game_v1().street_seam_config(BoardState::Turn, 2, commit, pot, live2_bet_menu());
    let tree = build_tree_depth_limited(&cfg).expect("turn tree");

    let ctx = MetalContext::new().expect("Metal");
    let bmask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let nh = (0..(52 * 51 / 2))
        .filter(|&idx| { let (c1, c2) = index_to_card_pair(idx);
            bmask & (1u64 << c1) == 0 && bmask & (1u64 << c2) == 0 })
        .count();
    let reach = vec![vec![1.0f32; nh]; 2];

    // Continuation resolution vs cost (measured, this scenario):
    //   nb=nh(1128)/600it → nuts bet 1.000, 10.7s  (per-hand, most accurate, tight)
    //   nb=200/300it      → nuts bet 0.964,  5.1s  (DEPLOY: value-bets, budget-safe)
    //   nb=64/300it       → nuts bet 0.775,  5.1s  (too coarse — tail averaged away)
    // Coarser nb averages the nuts with its neighbors; nb=200 keeps the tail sharp
    // enough to value-bet while the reduce kernel (O(n_leaf·nb·nh)/iter) stays cheap.
    let nb: usize = std::env::var("GPU_TURN_NB").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let iters: u32 = std::env::var("GPU_TURN_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    // River-integrated continuation (default): exact check-to-showdown model, so
    // the nuts value-bets at the proper ~78% instead of the proxy's ~100%.
    let river: bool = std::env::var("GPU_TURN_PROXY").is_err();
    let gcfg = GpuSearchCfg { iters, sample_m: 0, seed: 7, factored_terminals: false, lambda: 0.0 };
    let t = Instant::now();
    let (hand_cards, strat) = gpu_hu_turn_strat(&ctx, &board, &tree, &reach, nb, river, &gcfg);
    let elapsed = t.elapsed().as_secs_f32();

    let (a, b) = (card("Ac").min(card("Ad")), card("Ac").max(card("Ad")));
    let h = (0..nh).find(|&i| hand_cards[i * 2] == a && hand_cards[i * 2 + 1] == b)
        .expect("quad-aces present");
    let root = (0..tree.num_nodes())
        .find(|&n| tree.nodes[n].is_player() && tree.nodes[n].board_state == BoardState::Turn as u8)
        .expect("turn player node");
    let na = tree.nodes[root].num_children as usize;
    let s = strat.get(&root).expect("root strategy");
    let children = tree.node_children(root);
    let bet: f32 = (0..na)
        .filter(|&a| { let l = tree.nodes[children[a] as usize].action_label; l != 0 && l != 1 })
        .map(|a| s[a][h]).sum();

    eprintln!("GPU HU turn quad-aces: bet_prob={bet:.3} river={river} nb={nb} it={iters} in {elapsed:.2}s  (actions: {:?})",
        (0..na).map(|a| (tree.nodes[children[a] as usize].action_label, (s[a][h]*1000.0).round()/1000.0)).collect::<Vec<_>>());
    // River-integrated: the nuts value-bets at the proper ~78% (matches the exact
    // solve, see hu_turn_gpu_vs_exact) — the deploy defect was CHECKING it DOWN, so
    // the gate is "value-bets substantially", not "over-bets to ~1.0".
    assert!(bet > 0.5, "GPU HU turn quads should value-bet the majority, got {bet}");
    assert!(elapsed < 14.0, "GPU HU turn solve must fit the ~14s budget, took {elapsed:.2}s");
}
