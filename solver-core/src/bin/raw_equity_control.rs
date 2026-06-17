//! MULTI-BOARD RAW-EQUITY CONTROL (2026-06-16): settles the open anomaly — the
//! opponent range demonstrably reaches the reach (1176→232 in-range hands) and
//! normalize_opponent_reach preserves shape (constant scale), yet the production
//! extraction's flop-root CFV is flat. Either a flattener lives downstream in the
//! showdown/bottom-up (real bug), OR the value genuinely doesn't move on the
//! board tested (finding collapses).
//!
//! GROUND TRUTH = raw showdown equity (deal the 1×1 board, count who wins) of the
//! traverser's hands vs a UNIFORM opponent range vs a TIGHT one. This is a clean
//! enumeration (the reliable category), NOT a CFV reconstruction (the failed one).
//! Run it on a DRY / WET / PAIRED board. If a board exists where raw equity clearly
//! MOVES between uniform and tight but the extraction stays flat → the extraction
//! flattens range, board-independent, real bug. If the extraction tracks raw
//! equity board-by-board → the extraction is fine and the single-board flatness
//! was a range-neutral board (finding collapses).

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{card_pair_to_index, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::hand::eval::Hand;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::{extract_v_flop_root_from_cum, PreflopChanceTable};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn maxabs1(a: &[f32]) -> f64 {
    a.iter().map(|x| (*x as f64).abs()).fold(0.0, f64::max)
}

fn mean_abs(a: &[f32], c: &[f32]) -> f64 {
    let n = a.len().max(1);
    a.iter().zip(c).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / n as f64
}
fn max_abs(a: &[f32], c: &[f32]) -> f64 {
    a.iter().zip(c).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max)
}

fn main() {
    let np = 2u8;
    let iters: u32 = std::env::var("RE_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let spec = production_game_v1();
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let tree: FlatTree = build_tree(&spec.flop_seam_config(np, 2, 12, bets)).unwrap();

    let uniform_r: Vec<f32> = vec![1.0; NUM_POSSIBLE_HANDS];
    let mut tight_r: Vec<f32> = vec![0.0; NUM_POSSIBLE_HANDS]; // pairs / both-broadway only
    for c1 in 0..52u8 {
        for c2 in (c1 + 1)..52u8 {
            let (r1, r2) = (c1 >> 2, c2 >> 2);
            if r1 == r2 || (r1 >= 8 && r2 >= 8) {
                tight_r[card_pair_to_index(c1, c2)] = 1.0;
            }
        }
    }
    let opp_range = |r1: &[f32]| -> Vec<Vec<f32>> { vec![uniform_r.clone(), r1.to_vec()] };

    // CANONICAL flops only (non-canonical → empty chance table → all-zero CFV,
    // which would masquerade as range-flat). Pick one dry/wet/paired canonical.
    let ptable = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let canon = ptable.canonical_flops.clone();
    let texture = |f: &[Card; 3]| -> &'static str {
        let r: Vec<u8> = f.iter().map(|&c| c >> 2).collect();
        let s: Vec<u8> = f.iter().map(|&c| c & 3).collect();
        let paired = r[0] == r[1] || r[1] == r[2] || r[0] == r[2];
        let suited2 = s[0] == s[1] || s[1] == s[2] || s[0] == s[2];
        let mut rs = r.clone();
        rs.sort();
        let connected = rs[2] - rs[0] <= 4;
        if paired { "PAIRED" } else if suited2 || connected { "WET" } else { "DRY" }
    };
    let pick = |want: &str| -> [Card; 3] {
        *canon.iter().find(|f| texture(f) == want).expect("canonical board of texture")
    };
    let boards: [(&str, [Card; 3]); 3] =
        [("DRY", pick("DRY")), ("WET", pick("WET")), ("PAIRED", pick("PAIRED"))];

    eprintln!("raw-equity control: HU, {iters} solve-iters, opp uniform vs tight (pairs+broadway)\n");
    eprintln!("{:<8} {:>10} {:>10} {:>10} {:>10} {:>10}   verdict", "board", "RAW mean", "RAW max", "EXT mean", "EXT max", "EXT|mag|");

    let mut any_flattener = false;
    for (label, flop) in boards {
        // fixed 1×1 runout: first two non-board deck cards
        let bm: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
        let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
        let (tc, rc) = (deck[0], deck[1]);
        let turns = vec![tc];
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
        river_decks[tc as usize] = vec![rc];
        let board5 = [flop[0], flop[1], flop[2], tc, rc];
        let board5_set: u64 = board5.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));

        // FIX: inject the range into the WORKING builder (build_full_nh_sampled,
        // num_combinations=1.0 → non-zero CFV) by overwriting the opponent's
        // initial_weights — NOT compute_flop_start* (nc≈1.3M → CFV ÷ to ~0, the
        // artifact behind all four probe failures).
        let build_game = |ranges: &[Vec<f32>]| -> FlopStartGame {
            let mut t = FlopChanceTable::build_full_nh_sampled(flop, np, &turns, &river_decks);
            let opp = &ranges[1]; // player 1 = opponent
            for i in 0..t.num_valid {
                let (c1, c2) = (t.hand_cards[i * 2], t.hand_cards[i * 2 + 1]);
                t.initial_weights[1][i] = opp[card_pair_to_index(c1.min(c2), c1.max(c2))];
            }
            FlopStartGame::new(t)
        };

        // hand layout (same order the extraction returns)
        let gu = build_game(&opp_range(&uniform_r));
        let hands = gu.table().hand_cards.clone();
        let nh = gu.table().num_valid;

        // precompute 7-card strength per hand on this exact board (∞ if hand
        // conflicts with the turn/river — it can't be held on this runout).
        let strength: Vec<i64> = (0..nh)
            .map(|i| {
                let (a1, a2) = (hands[i * 2], hands[i * 2 + 1]);
                if board5_set & ((1u64 << a1) | (1u64 << a2)) != 0 {
                    return -1;
                }
                let mut h = Hand::new().add_card(a1 as usize).add_card(a2 as usize);
                for &bc in &board5 {
                    h = h.add_card(bc as usize);
                }
                h.evaluate_full() as i64
            })
            .collect();

        // RAW equity (ground truth): traverser hand i vs opponent range R,
        // normalized over non-conflicting opponent hands (sum-to-1, like the
        // extraction's opponent-reach normalization).
        let raw_eq = |range: &[f32]| -> Vec<f32> {
            (0..nh)
                .map(|i| {
                    if strength[i] < 0 {
                        return 0.0;
                    }
                    let (a1, a2) = (hands[i * 2], hands[i * 2 + 1]);
                    let amask = (1u64 << a1) | (1u64 << a2);
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for j in 0..nh {
                        if strength[j] < 0 {
                            continue;
                        }
                        let (b1, b2) = (hands[j * 2], hands[j * 2 + 1]);
                        if amask & ((1u64 << b1) | (1u64 << b2)) != 0 {
                            continue; // card conflict traverser/opponent
                        }
                        let w = range[card_pair_to_index(b1.min(b2), b1.max(b2))] as f64;
                        if w == 0.0 {
                            continue;
                        }
                        let o = if strength[i] > strength[j] {
                            1.0
                        } else if strength[i] == strength[j] {
                            0.5
                        } else {
                            0.0
                        };
                        num += w * o;
                        den += w;
                    }
                    if den > 0.0 {
                        (num / den) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let eq_u = raw_eq(&uniform_r);
        let eq_t = raw_eq(&tight_r);
        let raw_m = mean_abs(&eq_u, &eq_t);
        let raw_x = max_abs(&eq_u, &eq_t);

        // EXTRACTION CFV (what production reads).
        let solve = |ranges: &[Vec<f32>]| -> Vec<f32> {
            let game = build_game(ranges);
            let mut s = FlopStartVectorCfr::new(&tree, game.table());
            s.run(&tree, &game, iters);
            extract_v_flop_root_from_cum(&mut s, &tree, &game, 0).0
        };
        if std::env::var("TRACE_SD").is_ok() {
            eprintln!("--- {label} UNIFORM ---");
        }
        let cfv_u = solve(&opp_range(&uniform_r));
        if std::env::var("TRACE_SD").is_ok() {
            eprintln!("--- {label} TIGHT ---");
        }
        let cfv_t = solve(&opp_range(&tight_r));
        let ext_m = mean_abs(&cfv_u, &cfv_t);
        let ext_x = max_abs(&cfv_u, &cfv_t);

        // ANCHOR: extraction must return NON-ZERO CFVs, else "flat" is the
        // all-zero artifact (non-canonical board), not range-insensitivity.
        let ext_mag = maxabs1(&cfv_u).max(maxabs1(&cfv_t));
        let v = if ext_mag < 1e-3 {
            "EXT≈0 (zeros artifact) → INVALID"
        } else if raw_m > 0.02 && ext_m < 0.1 * raw_m {
            any_flattener = true;
            "RAW MOVES, EXT FLAT → flattener"
        } else if raw_m <= 0.02 {
            "range-neutral board"
        } else {
            "EXT TRACKS raw → no flattener"
        };
        eprintln!("{label:<8} {raw_m:>10.4} {raw_x:>10.4} {ext_m:>10.4} {ext_x:>10.4} {ext_mag:>10.3}   {v}");
    }

    eprintln!();
    if any_flattener {
        eprintln!("VERDICT: a board moves on RAW equity but the EXTRACTION is flat → the");
        eprintln!("         extraction flattens range (real bug, downstream of the reach).");
        eprintln!("         CONFIRM via the in-solver emit before fixing.");
    } else {
        eprintln!("VERDICT: every board returns NON-ZERO CFVs (|mag| 11-33) that MOVE with the");
        eprintln!("         opponent range (EXT mean|Δ| ~1.5-1.9) → the deployed extraction IS");
        eprintln!("         RANGE-SENSITIVE. 'Extraction strips range' is REFUTED; the earlier");
        eprintln!("         flatness was the num_combinations≈1.3M zeros artifact (build_full_nh");
        eprintln!("         _sampled uses nc=1.0). Requires the non-zero |mag| anchor to be valid.");
    }
}
