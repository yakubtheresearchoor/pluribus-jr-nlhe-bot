//! CO-SOLVE PREMISE PROBE (2026-06-17): before committing weeks to the live
//! co-solve, confirm its PREMISE — that a fold-capable (tighter) opponent flips
//! AA from limp to raise. AA limps in the frozen game because vs UNIFORM (wide)
//! never-tightening opponents, the MULTIWAY pot extracts more than the isolated
//! HU pot (more weak hands pay off AA's dominance). If that edge is range-driven,
//! a tighter opponent should collapse it and AA's HU (raise/isolate) value should
//! catch or beat its multiway (limp) value.
//!
//! Measures AA's flop-root EV at the SAME per-player pot (isolating field+range
//! from pot-size), for HU (live-2, exact) and MULTIWAY (live-3, bucketed), each
//! vs a UNIFORM opponent range and a TIGHT one (pairs + both-broadway). The
//! signal: does (multiway − HU) shrink or REVERSE from uniform → tight?
//!   reverse/shrink to ≤0  ⇒ co-solve premise CONFIRMED (tight opp ⇒ AA isolates)
//!   multiway still ≫ HU   ⇒ premise NOT confirmed — the co-solve may not fix AA.

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::{card_from_str, card_pair_to_index, Card};
use solver_core::card::NUM_POSSIBLE_HANDS;
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::preflop_start_game::{extract_v_flop_root_from_cum, PreflopChanceTable};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

fn main() {
    let iters: u32 = std::env::var("CP_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(48);
    let nb: usize = std::env::var("CP_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    let spec = production_game_v1();
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };

    // Uniform and TIGHT (pairs + both-broadway) opponent ranges.
    let uniform_r: Vec<f32> = vec![1.0; NUM_POSSIBLE_HANDS];
    let mut tight_r: Vec<f32> = vec![0.0; NUM_POSSIBLE_HANDS];
    for c1 in 0..52u8 {
        for c2 in (c1 + 1)..52u8 {
            let (r1, r2) = (c1 >> 2, c2 >> 2);
            if r1 == r2 || (r1 >= 8 && r2 >= 8) {
                tight_r[card_pair_to_index(c1, c2)] = 1.0;
            }
        }
    }

    let ptable = PreflopChanceTable::new(6, vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6]);
    let texture = |f: &[Card; 3]| -> &'static str {
        let r: Vec<u8> = f.iter().map(|&c| c >> 2).collect();
        let paired = r[0] == r[1] || r[1] == r[2] || r[0] == r[2];
        let mut rs = r.clone();
        rs.sort();
        if paired { "PAIRED" } else if rs[2] - rs[0] <= 4 { "WET" } else { "DRY" }
    };
    let pick = |want: &str| -> [Card; 3] {
        *ptable.canonical_flops.iter().find(|f| texture(f) == want).expect("board")
    };
    let boards = [("DRY", pick("DRY")), ("WET", pick("WET")), ("PAIRED", pick("PAIRED"))];

    // AA holding (suits not on the board; resolved per board below).
    println!("CO-SOLVE PREMISE: AA flop-root EV, HU(live-2) vs MULTIWAY(live-3), opp uniform vs tight");
    println!("(same pot=12; signal: multiway−HU should shrink/reverse from uniform→tight)\n");
    println!("{:<7} {:>10} {:>10} {:>10}   {:>10} {:>10} {:>10}", "board",
        "HU/unif", "HU/tight", "Δrange", "MW/unif", "MW/tight", "Δrange");

    for (label, flop) in boards {
        let bm: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
        let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
        let (tc, rc) = (deck[0], deck[1]);
        let turns = vec![tc];
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
        river_decks[tc as usize] = vec![rc];

        // AA combo not conflicting with board+runout.
        let used: u64 = bm | (1u64 << tc) | (1u64 << rc);
        let aces: Vec<u8> = (0..52u8).filter(|&c| c >> 2 == 12 && used & (1u64 << c) == 0).collect();
        let (aa1, aa2) = (aces[0], aces[1]);
        let aa_idx_in = |hand_cards: &[u8], nh: usize| -> usize {
            (0..nh).find(|&i| {
                let (a, b) = (hand_cards[i * 2], hand_cards[i * 2 + 1]);
                (a == aa1 && b == aa2) || (a == aa2 && b == aa1)
            }).expect("AA in layout")
        };

        // ---- HU (live-2, exact) ----
        let tree2 = build_tree(&spec.flop_seam_config(2, 2, 12, bets.clone())).unwrap();
        let hu_ev = |opp: &[f32]| -> f32 {
            let mut t = FlopChanceTable::build_full_nh_sampled(flop, 2, &turns, &river_decks);
            for i in 0..t.num_valid {
                let (c1, c2) = (t.hand_cards[i * 2], t.hand_cards[i * 2 + 1]);
                t.initial_weights[1][i] = opp[card_pair_to_index(c1.min(c2), c1.max(c2))];
            }
            let hc = t.hand_cards.clone();
            let nh = t.num_valid;
            let game = FlopStartGame::new(t);
            let mut s = FlopStartVectorCfr::new(&tree2, game.table());
            s.run(&tree2, &game, iters);
            let v = extract_v_flop_root_from_cum(&mut s, &tree2, &game, 0).0;
            v[aa_idx_in(&hc, nh)]
        };
        let hu_u = hu_ev(&uniform_r);
        let hu_t = hu_ev(&tight_r);

        // ---- MULTIWAY (live-3, bucketed) ----
        let tree3 = build_tree(&spec.flop_seam_config(3, 2, 12, bets.clone())).unwrap();
        let mw_ev = |opp: &[f32]| -> f32 {
            let mut t = FlopChanceTable::build_full_nh_sampled(flop, 3, &turns, &river_decks);
            for i in 0..t.num_valid {
                let (c1, c2) = (t.hand_cards[i * 2], t.hand_cards[i * 2 + 1]);
                let w = opp[card_pair_to_index(c1.min(c2), c1.max(c2))];
                t.initial_weights[1][i] = w; // both opponents tight
                t.initial_weights[2][i] = w;
            }
            let hc = t.hand_cards.clone();
            let nh = t.num_valid;
            let bk = FlopBucketing::quantile(&t, nb);
            let game = FlopStartGame::new(t);
            let mut s = BucketedFlopCfr::new(&tree3, game.table(), &bk);
            s.set_terminal_design(TerminalDesign::Design1Collapsed);
            s.run(&tree3, &game, &bk, iters);
            let root = s.root_cfv_from_avg(&tree3, &game, &bk);
            root[0][aa_idx_in(&hc, nh)]
        };
        let mw_u = mw_ev(&uniform_r);
        let mw_t = mw_ev(&tight_r);

        println!("{label:<7} {hu_u:>10.3} {hu_t:>10.3} {:>10.3}   {mw_u:>10.3} {mw_t:>10.3} {:>10.3}",
            hu_t - hu_u, mw_t - mw_u);
        println!("        multiway−HU:  uniform {:+.3}   tight {:+.3}   {}",
            mw_u - hu_u, mw_t - hu_t,
            if (mw_u - hu_u) > 0.0 && (mw_t - hu_t) <= (mw_u - hu_u) * 0.5 {
                "→ tight collapses multiway edge ⇒ premise SUPPORTED"
            } else if (mw_t - hu_t) > 0.0 {
                "→ multiway still wins vs tight ⇒ premise NOT supported"
            } else {
                "→ HU wins vs tight ⇒ premise SUPPORTED"
            });
    }
    println!("\n(AA limps because multiway > HU vs uniform. If multiway−HU collapses/reverses");
    println!(" vs a tight opp, a co-solve (which tightens opponents) would make AA isolate/raise.)");
}
