//! GROUND TRUTH for the #12 turn-nuts question: does the *exact* HU turn solve
//! (real per-hand river showdown; only river BETTING is check-only) value-bet
//! quad aces on the turn, or slowplay it? The runtime `solve_live2_street` is
//! budget-capped (~40-60 iters) and reported ~60%; the GPU bucketed-continuation
//! solve (600 iters) reported ~0% (pure check). This test runs the exact solve
//! to convergence (250 iters, no budget cap) to decide which is right — i.e.
//! whether "the nuts slowplays the turn" is real GTO or a continuation artifact.
//!
//! Heavy (~250 * ~208ms ≈ 50s). Ignored by default; run on demand.

use solver_core::card::{card_from_str, Card};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{BetSizeOptions, BoardState, production_game_v1};
use solver_core::tree::builder::build_tree_with_bet_override;

use play_harness::live2_bank::live2_bet_menu;

fn card(s: &str) -> u8 { card_from_str(s).unwrap() as u8 }

#[test]
#[ignore = "exact turn solve to convergence (~50s). Run on demand: ground truth for the turn-nuts question."]
fn exact_hu_turn_nuts_bet_frequency() {
    let board = [card("As"), card("Ah"), card("7c"), card("2d")];
    let (commit, pot) = (20i32, 40i32);
    let iters: u32 = std::env::var("EXACT_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(250);

    let spec = production_game_v1();
    let board_c: Vec<Card> = board.iter().map(|&c| c as Card).collect();
    let ranges = vec![vec![1.0f32; 1326]; 2];
    let table = ChanceTable::compute_turn_start(&board_c, &ranges, 2);
    let nh = table.num_valid;
    let hand_cards = table.hand_cards.clone();
    let game = TurnStartGame::new(table);
    let cfg = spec.street_seam_config(BoardState::Turn, 2, commit, pot, live2_bet_menu());
    // Same nested check-only river continuation the runtime uses (river showdown
    // is EXACT per-hand; only river betting is truncated).
    let check_only = BetSizeOptions { bet: vec![], raise: vec![] };
    let tree = build_tree_with_bet_override(&cfg, &[(BoardState::River, check_only)]).expect("turn tree");

    let mut cfr = VectorCfr::new(&tree, vec![nh; 2]);
    cfr.run(&tree, &game, iters);

    // Quad-aces hand index (Ac, Ad; stored c1<c2).
    let (a, b) = (card("Ac").min(card("Ad")), card("Ac").max(card("Ad")));
    let h = (0..nh).find(|&i| hand_cards[i * 2] == a && hand_cards[i * 2 + 1] == b)
        .expect("quad-aces present");
    // First-to-act turn node.
    let root = (0..tree.num_nodes())
        .find(|&n| tree.nodes[n].is_player() && tree.nodes[n].board_state == BoardState::Turn as u8)
        .expect("turn player node");
    let na = tree.nodes[root].num_children as usize;
    let strat = cfr.get_average_strategy(root, na, nh);
    let children = tree.node_children(root);
    let bet: f32 = (0..na)
        .filter(|&a| { let l = tree.nodes[children[a] as usize].action_label; l != 0 && l != 1 })
        .map(|a| strat[a][h]).sum();
    eprintln!("EXACT HU turn quad-aces first-act ({iters} iters): bet_prob={bet:.4}");
    eprintln!("  actions: {:?}", (0..na)
        .map(|a| (tree.nodes[children[a] as usize].action_label, (strat[a][h]*1000.0).round()/1000.0))
        .collect::<Vec<_>>());
    // No assertion on direction — this is a MEASUREMENT. It prints the ground-truth
    // converged bet frequency of the nuts on the turn.
}
