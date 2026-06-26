//! BUCKET-TRANSFER PROBE: does one solved turn's per-bucket strategy transfer to a
//! DIFFERENT turn (the stopgap for our 1×1 runout gap)? Solve several turns exactly
//! (HU), bucket each by equity quantile (bucket k = k-th strength tier, as the
//! blueprint does), and compare:
//!   - ABSTRACTION error: turn B per-hand strategy vs B's OWN per-bucket average
//!     (the irreducible cost of bucketing at all).
//!   - TRANSFER error: turn B per-hand strategy vs turn A's per-bucket strategy
//!     applied via B's buckets (the cost of REUSING A's solve for B).
//! If TRANSFER ≈ ABSTRACTION, reusing one runout is ~free; if TRANSFER >>, re-solve.
//!
//! Run: cargo run --release -p play-harness --bin bucket_transfer_probe

use play_harness::eqr::allin_equity_on_board;
use play_harness::live2_bank::solve_live2_street;
use solver_core::card::{card_from_str, Card};

struct TurnData {
    turn: Card,
    na: usize,
    nh: usize,
    strat: Vec<Vec<f32>>, // [na][nh] average strategy at the turn root
    bucket: Vec<usize>,   // per-hand quantile bucket
    pb: Vec<Vec<f32>>,    // [nb][na] per-bucket mean strategy
}

fn per_bucket(strat: &[Vec<f32>], bucket: &[usize], na: usize, nh: usize, nb: usize) -> Vec<Vec<f32>> {
    let mut sum = vec![vec![0.0f32; na]; nb];
    let mut cnt = vec![0usize; nb];
    for h in 0..nh {
        let b = bucket[h];
        cnt[b] += 1;
        for a in 0..na {
            sum[b][a] += strat[a][h];
        }
    }
    for b in 0..nb {
        if cnt[b] > 0 {
            for a in 0..na {
                sum[b][a] /= cnt[b] as f32;
            }
        }
    }
    sum
}

fn solve_turn(flop: &[u8], turn: Card, commit: i32, pot: i32, iters: u32, nb: usize) -> TurnData {
    let mut board = flop.to_vec();
    board.push(turn);
    let s = solve_live2_street(&board, commit, pot, iters).expect("solve_live2_street");
    let na = s.tree.nodes[0].num_children as usize;
    let nh = s.nh;
    let strat = s.cfr.get_average_strategy(0, na, nh);
    // equity per hand (river rollout vs a random opponent) → strength ordering.
    let equity: Vec<f32> = (0..nh)
        .map(|h| allin_equity_on_board([s.hand_cards[2 * h], s.hand_cards[2 * h + 1]], &board, 2, 200, 0x51 + h as u64))
        .collect();
    let mut order: Vec<usize> = (0..nh).collect();
    order.sort_by(|&a, &b| equity[a].partial_cmp(&equity[b]).unwrap());
    let mut bucket = vec![0usize; nh];
    for (rank, &h) in order.iter().enumerate() {
        bucket[h] = (rank * nb / nh).min(nb - 1);
    }
    let pb = per_bucket(&strat, &bucket, na, nh, nb);
    TurnData { turn, na, nh, strat, bucket, pb }
}

fn main() {
    let nb = 15;
    let iters: u32 = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let (commit, pot) = (10i32, 20i32); // deep SPR → full M2 turn menu
    let flop: Vec<u8> = ["Ks", "9d", "4c"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let turns: Vec<Card> = ["2h", "7s", "Td", "Qc", "Ah"].iter().map(|s| card_from_str(s).unwrap()).collect();

    let mut data: Vec<TurnData> = Vec::new();
    for &t in &turns {
        let d = solve_turn(&flop, t, commit, pot, iters, nb);
        eprintln!("solved turn {t}: na={} nh={}", d.na, d.nh);
        data.push(d);
    }
    let na = data[0].na;

    println!("\nflop Ks9d4c, commit={commit} pot={pot} (deep SPR), B={nb}, {iters} iters, na={na}");

    println!("\nABSTRACTION baseline (per-hand vs own-bucket avg), per turn:");
    for d in &data {
        let mut e = 0.0f32;
        let mut c = 0;
        for h in 0..d.nh {
            for a in 0..d.na {
                e += (d.strat[a][h] - d.pb[d.bucket[h]][a]).abs();
                c += 1;
            }
        }
        println!("  turn {}: {:.4}", d.turn, e / c as f32);
    }

    println!("\nTRANSFER vs ABSTRACTION (mean |Δprob| per (hand,action)):");
    let mut sum_trans = 0.0f32;
    let mut sum_abst = 0.0f32;
    let mut pairs = 0;
    for i in 0..data.len() {
        for j in 0..data.len() {
            if i == j || data[i].na != data[j].na {
                continue;
            }
            let (a_pb, b) = (&data[i].pb, &data[j]);
            let (mut trans, mut abst, mut c) = (0.0f32, 0.0f32, 0);
            for h in 0..b.nh {
                let bk = b.bucket[h];
                for a in 0..na {
                    trans += (b.strat[a][h] - a_pb[bk][a]).abs();
                    abst += (b.strat[a][h] - b.pb[bk][a]).abs();
                    c += 1;
                }
            }
            let (tr, ab) = (trans / c as f32, abst / c as f32);
            println!("  {}→{}: transfer {:.4}  abstraction {:.4}  (extra {:+.4})", data[i].turn, b.turn, tr, ab, tr - ab);
            sum_trans += tr;
            sum_abst += ab;
            pairs += 1;
        }
    }
    println!(
        "\nMEAN over {pairs} pairs: transfer {:.4}  abstraction {:.4}  → reusing another runout adds {:+.4} mean |Δprob| ({:.0}% over the bucketing baseline)",
        sum_trans / pairs as f32,
        sum_abst / pairs as f32,
        (sum_trans - sum_abst) / pairs as f32,
        100.0 * (sum_trans - sum_abst) / sum_abst.max(1e-6)
    );
}
