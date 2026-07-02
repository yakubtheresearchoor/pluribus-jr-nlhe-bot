//! #12 fidelity validation: compare the GPU depth-limited turn solve (proxy vs
//! river-integrated continuation) against the CONVERGED exact solve (real
//! per-hand river showdown, check-only river betting) on the same production
//! seam config. Two things established here:
//!   1. The nuts value-bets ~always when converged (the exact's earlier "78%" at
//!      250it was under-convergence; at 1000it it's ~1.0 — matching the GPU).
//!   2. The river-integrated continuation is MORE faithful than the turn-strength
//!      proxy: lower full-strategy L1 vs the converged exact, at ~zero extra cost.
//!
//! Heavy (exact 1000it ≈ 200s). Ignored; run on demand.

use std::collections::HashMap;

use solver_core::card::{card_from_str, index_to_card_pair, Card};
use solver_core::gpu_metal::{gpu_hu_turn_strat, GpuSearchCfg, MetalContext};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSizeOptions, BoardState, production_game_v1};
use solver_core::tree::builder::{build_tree_depth_limited, build_tree_with_bet_override};
use solver_core::tree::flat::FlatTree;

use play_harness::live2_bank::live2_bet_menu;

fn card(s: &str) -> u8 { card_from_str(s).unwrap() as u8 }

fn root_turn(tree: &FlatTree) -> usize {
    (0..tree.num_nodes())
        .find(|&n| tree.nodes[n].is_player() && tree.nodes[n].board_state == BoardState::Turn as u8)
        .expect("turn player node")
}
fn hand_index(hand_cards: &[u8], nh: usize) -> HashMap<(u8, u8), usize> {
    (0..nh).map(|i| ((hand_cards[i * 2], hand_cards[i * 2 + 1]), i)).collect()
}
fn nuts_bet(tree: &FlatTree, s: &[Vec<f32>], h: usize) -> f32 {
    let root = root_turn(tree);
    let ch = tree.node_children(root);
    (0..s.len()).filter(|&a| { let l = tree.nodes[ch[a] as usize].action_label; l != 0 && l != 1 })
        .map(|a| s[a][h]).sum()
}

#[test]
#[ignore = "converged exact (1000it ≈ 200s) + GPU proxy/river fidelity compare. Run on demand."]
fn gpu_vs_exact_turn_fidelity() {
    let board = [card("As"), card("Ah"), card("7c"), card("2d")];
    let (commit, pot) = (20i32, 40i32);
    let exact_iters: u32 = std::env::var("EXACT_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let spec = production_game_v1();
    let cfg = spec.street_seam_config(BoardState::Turn, 2, commit, pot, live2_bet_menu());

    // ---- CONVERGED EXACT: real per-hand river showdown, check-only river ----
    let board_c: Vec<Card> = board.iter().map(|&c| c as Card).collect();
    let ranges = vec![vec![1.0f32; 1326]; 2];
    let table = ChanceTable::compute_turn_start(&board_c, &ranges, 2);
    let nh_e = table.num_valid;
    let hc_e = table.hand_cards.clone();
    let game = TurnStartGame::new(table);
    let check_only = BetSizeOptions { bet: vec![], raise: vec![] };
    let tree_e = build_tree_with_bet_override(&cfg, &[(BoardState::River, check_only)]).expect("exact tree");
    let mut cfr = VectorCfr::new(&tree_e, vec![nh_e; 2]);
    cfr.run(&tree_e, &game, exact_iters);
    let re = root_turn(&tree_e);
    let na = tree_e.nodes[re].num_children as usize;
    let strat_e = cfr.get_average_strategy(re, na, nh_e);
    let idx_e = hand_index(&hc_e, nh_e);

    // ---- GPU proxy + river (nb=200, 400it) ----
    let tree_g = build_tree_depth_limited(&cfg).expect("gpu tree");
    let ctx = MetalContext::new().expect("Metal");
    let bmask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let nh_g = (0..(52 * 51 / 2))
        .filter(|&idx| { let (c1, c2) = index_to_card_pair(idx);
            bmask & (1u64 << c1) == 0 && bmask & (1u64 << c2) == 0 })
        .count();
    let reach = vec![vec![1.0f32; nh_g]; 2];
    let gcfg = GpuSearchCfg { iters: 400, sample_m: 0, seed: 7, factored_terminals: false, lambda: 0.0 , budget_ms: 120_000 };
    let (hc_p, st_p) = gpu_hu_turn_strat(&ctx, &board, &tree_g, &reach, 200, false, &gcfg);
    let (hc_r, st_r) = gpu_hu_turn_strat(&ctx, &board, &tree_g, &reach, 200, true, &gcfg);
    let rg = root_turn(&tree_g);
    let sp = st_p.get(&rg).unwrap();
    let sr = st_r.get(&rg).unwrap();

    // Full-strategy mean-L1 (over hands, summed over actions) vs the converged exact.
    // Same seam config ⇒ same root action structure; align hands by their two cards.
    let na_g = sp.len();
    assert_eq!(na, na_g, "action count mismatch exact={na} gpu={na_g}");
    let mean_l1 = |sg: &[Vec<f32>], hc_g: &[u8]| -> f32 {
        let mut sum = 0.0f32; let mut cnt = 0;
        for hg in 0..nh_g {
            let key = (hc_g[hg * 2], hc_g[hg * 2 + 1]);
            let Some(&he) = idx_e.get(&key) else { continue };
            let l1: f32 = (0..na).map(|a| (sg[a][hg] - strat_e[a][he]).abs()).sum();
            sum += l1; cnt += 1;
        }
        sum / cnt as f32
    };
    let l1_p = mean_l1(sp, &hc_p);
    let l1_r = mean_l1(sr, &hc_r);

    // Nuts bet for each.
    let hn_e = *idx_e.get(&(card("Ac").min(card("Ad")), card("Ac").max(card("Ad")))).unwrap();
    let hn_p = *hand_index(&hc_p, nh_g).get(&(card("Ac").min(card("Ad")), card("Ac").max(card("Ad")))).unwrap();
    let hn_r = *hand_index(&hc_r, nh_g).get(&(card("Ac").min(card("Ad")), card("Ac").max(card("Ad")))).unwrap();

    eprintln!("CONVERGED exact ({exact_iters}it) vs GPU (nb=200/400it):");
    eprintln!("  nuts bet:  exact={:.3}  proxy={:.3}  river={:.3}",
        nuts_bet(&tree_e, &strat_e, hn_e), nuts_bet(&tree_g, sp, hn_p), nuts_bet(&tree_g, sr, hn_r));
    eprintln!("  mean-L1 vs exact:  proxy={l1_p:.4}   river={l1_r:.4}   (lower = more faithful)");
    assert!(l1_r <= l1_p + 1e-3,
        "river-integrated should be at least as faithful as the proxy: proxy={l1_p:.4} river={l1_r:.4}");
}
