//! ANYTIME-CONVERGENCE PROBE: when a turn re-solve is over budget (live-5), is the
//! best-in-budget partial re-solve of the ACTUAL board better than the transfer
//! fallback (another runout's converged strategy via this board's buckets)? Solve
//! turn B to convergence (ref), snapshot its average strategy at increasing iters
//! (anytime), and compare each snapshot's distance-to-ref against the transfer
//! baseline. The crossover iter = how many iters a re-solve needs to beat the
//! fallback; map that onto live-5's budget (~few iters) to decide.
//!
//! Run: cargo run --release -p play-harness --bin anytime_probe

use play_harness::eqr::allin_equity_on_board;
use play_harness::live2_bank::live2_bet_menu;
use solver_core::card::{card_from_str, Card};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{production_game_v1, BetSizeOptions, BoardState};
use solver_core::tree::builder::{build_tree, build_tree_with_bet_override};

fn buckets(board: &[u8], hand_cards: &[u8], nh: usize, nb: usize) -> Vec<usize> {
    let eq: Vec<f32> = (0..nh)
        .map(|h| allin_equity_on_board([hand_cards[2 * h], hand_cards[2 * h + 1]], board, 2, 200, 0x9 + h as u64))
        .collect();
    let mut order: Vec<usize> = (0..nh).collect();
    order.sort_by(|&a, &b| eq[a].partial_cmp(&eq[b]).unwrap());
    let mut bk = vec![0usize; nh];
    for (r, &h) in order.iter().enumerate() {
        bk[h] = (r * nb / nh).min(nb - 1);
    }
    bk
}

fn per_bucket(strat: &[Vec<f32>], bk: &[usize], na: usize, nh: usize, nb: usize) -> Vec<Vec<f32>> {
    let mut s = vec![vec![0.0f32; na]; nb];
    let mut c = vec![0usize; nb];
    for h in 0..nh {
        c[bk[h]] += 1;
        for a in 0..na {
            s[bk[h]][a] += strat[a][h];
        }
    }
    for b in 0..nb {
        if c[b] > 0 {
            for a in 0..na {
                s[b][a] /= c[b] as f32;
            }
        }
    }
    s
}

/// Build a nested turn subgame (rich turn + check-only river) for `board`: tree +
/// game + dims. Create fresh `VectorCfr::new(&tree, vec![nh;2])` per solve arm.
fn turn_pieces(board: &[u8]) -> (solver_core::tree::flat::FlatTree, TurnStartGame, usize, usize, Vec<u8>) {
    let spec = production_game_v1();
    let ranges = vec![vec![1.0f32; 1326]; 2];
    let table = ChanceTable::compute_turn_start(&board.iter().map(|&c| c as Card).collect::<Vec<_>>(), &ranges, 2);
    let nh = table.num_valid;
    let hc = table.hand_cards.clone();
    let game = TurnStartGame::new(table);
    let cfg = spec.street_seam_config(BoardState::Turn, 2, 10, 20, live2_bet_menu());
    let ck = BetSizeOptions { bet: vec![], raise: vec![] };
    let _ = build_tree;
    let tree = build_tree_with_bet_override(&cfg, &[(BoardState::River, ck)]).unwrap();
    let na = tree.nodes[0].num_children as usize;
    (tree, game, na, nh, hc)
}

fn main() {
    let nb = 15;
    let snaps = [1u32, 2, 4, 8, 16, 32, 64, 128];
    let refn = 256u32;
    let card = |s: &str| card_from_str(s).unwrap();
    let flop = [card("Ks"), card("9d"), card("4c")];
    let b_board: Vec<u8> = vec![flop[0], flop[1], flop[2], card("2h")]; // target turn B
    let a_board: Vec<u8> = vec![flop[0], flop[1], flop[2], card("Ah")]; // other runout A (fallback source)

    let weight: f32 = std::env::var("W").ok().and_then(|s| s.parse().ok()).unwrap_or(8.0);

    // Target turn B: converged ref (for distance) + B's buckets.
    let (tree, game, na, nh, hc) = turn_pieces(&b_board);
    let bk_b = buckets(&b_board, &hc, nh, nb);
    let ref_pb = {
        let mut c = VectorCfr::new(&tree, vec![nh; 2]);
        c.run(&tree, &game, refn);
        per_bucket(&c.get_average_strategy(0, na, nh), &bk_b, na, nh, nb)
    };

    // Turn A converged → per-NODE per-bucket strategy (the warm-start prior + the
    // root-only transfer fallback).
    let (ta, ga, _naa, nha, hca) = turn_pieces(&a_board);
    let mut cfa = VectorCfr::new(&ta, vec![nha; 2]);
    cfa.run(&ta, &ga, refn);
    let bk_a = buckets(&a_board, &hca, nha, nb);
    let mut pb_a_node: std::collections::HashMap<usize, Vec<Vec<f32>>> = std::collections::HashMap::new();
    for &nid in &ta.decision_node_ids {
        let n = nid as usize;
        let nan = ta.nodes[n].num_children as usize;
        pb_a_node.insert(n, per_bucket(&cfa.get_average_strategy(n, nan, nha), &bk_a, nan, nha, nb));
    }
    let pb_a_root = pb_a_node[&0].clone();

    let dist = |x: &[Vec<f32>]| -> f32 {
        let mut e = 0.0;
        for b in 0..nb {
            for a in 0..na {
                e += (x[b][a] - ref_pb[b][a]).abs();
            }
        }
        e / (nb * na) as f32
    };
    let transfer = dist(&pb_a_root);

    // COLD anytime (uniform init).
    let cold: Vec<(u32, f32)> = {
        let mut c = VectorCfr::new(&tree, vec![nh; 2]);
        let mut out = Vec::new();
        let mut done = 0u32;
        for &k in &snaps {
            c.run(&tree, &game, k - done);
            done = k;
            out.push((k, dist(&per_bucket(&c.get_average_strategy(0, na, nh), &bk_b, na, nh, nb))));
        }
        out
    };

    // WARM anytime (seed from A's per-node bucket strategy via B's buckets).
    let warm: Vec<(u32, f32)> = {
        let mut c = VectorCfr::new(&tree, vec![nh; 2]);
        let bk_b2 = bk_b.clone();
        c.warm_start(&tree, weight, |node, nan| {
            let pb = pb_a_node.get(&node);
            let mut s = vec![0.0f32; nan * nh];
            for h in 0..nh {
                let bkt = bk_b2[h];
                for a in 0..nan {
                    s[a * nh + h] = pb.map(|p| p[bkt][a]).unwrap_or(1.0 / nan as f32);
                }
            }
            s
        });
        let mut out = Vec::new();
        let mut done = 0u32;
        // iter 0 = the warm-started average (before any run) = the transfer.
        out.push((0, dist(&per_bucket(&c.get_average_strategy(0, na, nh), &bk_b, na, nh, nb))));
        for &k in &snaps {
            c.run(&tree, &game, k - done);
            done = k;
            out.push((k, dist(&per_bucket(&c.get_average_strategy(0, na, nh), &bk_b, na, nh, nb))));
        }
        out
    };

    println!("turn B=Ks9d4c2h, fallback=transfer from Ah, B={nb}, na={na}, warm weight={weight}\n");
    println!("fallback (transfer A→B) distance-to-converged: {transfer:.4}\n");
    println!("{:>6} {:>14} {:>14}", "iters", "COLD dist", "WARM dist");
    let warm_map: std::collections::HashMap<u32, f32> = warm.iter().cloned().collect();
    if let Some(&w0) = warm_map.get(&0) {
        println!("{:>6} {:>14} {:>14.4}  (warm iter-0 = the seed)", 0, "-", w0);
    }
    for (k, cd) in &cold {
        let wd = warm_map.get(k).copied().unwrap_or(f32::NAN);
        let cm = if *cd < transfer { "✓" } else { " " };
        let wm = if wd < transfer { "✓" } else { " " };
        println!("{:>6} {:>12.4}{} {:>12.4}{}", k, cd, cm, wd, wm);
    }
}
