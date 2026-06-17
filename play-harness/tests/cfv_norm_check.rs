//! CFV NORMALIZATION CHECK (2026-06-16): directly extract root_cfv_from_avg
//! (via cfv_from_banked) for one rep .bp per live count on ONE dry flop, and
//! print per-class AA / 98s / 72o. After the opponent-reach-sum-1 fix these
//! should be CHIP-SCALE per-hand EVs (AA modestly +, 72o clearly −, magnitudes
//! comparable across live counts and proportional to pot) — not nh^num_opp.

use play_harness::preflop_oracle::cfv_from_banked;
use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::card_from_str;
use solver_core::solver::preflop_start_game::{flop_combo_layout, PreflopChanceTable};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;

#[test]
fn cfv_norm_check() {
    let spec = production_game_v1();
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let canon = PreflopChanceTable::new(6, vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6]).canonical_flops.clone();
    // a dry rainbow broadway flop
    let fi = canon.iter().position(|b| {
        let r: Vec<u8> = b.iter().map(|&c| c >> 2).collect();
        let s: Vec<u8> = b.iter().map(|&c| c & 3).collect();
        r[0] != r[1] && r[1] != r[2] && r[0] != r[2] && s[0] != s[1] && s[1] != s[2] && s[0] != s[2] && *r.iter().max().unwrap() >= 10
    }).unwrap();
    let board = canon[fi];
    let layout = flop_combo_layout(board);
    let aa = PreflopClass::from_combo(card_from_str("Ac").unwrap(), card_from_str("Ad").unwrap()).index();
    let mid = PreflopClass::from_combo(card_from_str("9c").unwrap(), card_from_str("8c").unwrap()).index();
    let trash = PreflopClass::from_combo(card_from_str("7c").unwrap(), card_from_str("2d").unwrap()).index();

    let reps: &[(u8, i32, i32, &str)] = &[
        (3, 10, 32, "live3_c10_p32_b15"),
        (4, 10, 40, "live4_c10_p40_b15"),
        (5, 10, 50, "live5_c10_p50_b8"),
    ];
    eprintln!("\n=== per-class flop-root CFV (traverser 0), dry flop fi={fi} ===");
    for &(live, commit, pot, dir) in reps {
        let tree = build_tree(&spec.flop_seam_config(live, commit, pot, bets.clone())).unwrap();
        let full = format!("{}/../blueprint_out_v1/{dir}", env!("CARGO_MANIFEST_DIR"));
        let per_live = cfv_from_banked(&full, fi, &tree, board);
        let v = &per_live[0];
        let mut sums = vec![0.0f64; NUM_PREFLOP_CLASSES];
        let mut cnts = vec![0u32; NUM_PREFLOP_CLASSES];
        for (i, &(c1, c2)) in layout.iter().enumerate() {
            let cl = PreflopClass::from_combo(c1, c2).index();
            sums[cl] += v[i] as f64;
            cnts[cl] += 1;
        }
        let avg = |c: usize| if cnts[c] > 0 { sums[c] / cnts[c] as f64 } else { 0.0 };
        eprintln!("  live-{live} (pot {pot}): AA={:+.3}  98s={:+.3}  72o={:+.3}", avg(aa), avg(mid), avg(trash));
    }

    // live-2 (exact HU) and a live-6 rollout — must land on the SAME chip scale.
    let per_class = |v: &[f32]| -> (f64, f64, f64) {
        let mut s = vec![0.0f64; NUM_PREFLOP_CLASSES];
        let mut c = vec![0u32; NUM_PREFLOP_CLASSES];
        for (i, &(c1, c2)) in layout.iter().enumerate() {
            let cl = PreflopClass::from_combo(c1, c2).index();
            s[cl] += v[i] as f64; c[cl] += 1;
        }
        let a = |k: usize| if c[k] > 0 { s[k] / c[k] as f64 } else { 0.0 };
        (a(aa), a(mid), a(trash))
    };
    let l2tree = build_tree(&spec.flop_seam_config(2, 10, 24, bets.clone())).unwrap();
    let (aa2, m2, t2) = per_class(&play_harness::preflop_oracle::cfv_live2(board, fi, &l2tree)[0]);
    eprintln!("  live-2 (pot 24): AA={:+.3}  98s={:+.3}  72o={:+.3}", aa2, m2, t2);
    let l6tree = play_harness::preflop_oracle::rollout_seam_tree(&spec, 6, 2, 12);
    let (aa6, m6, t6) = per_class(&play_harness::preflop_oracle::cfv_rollout_live3plus(board, fi, &l6tree, 6, 8)[0]);
    eprintln!("  live-6 rollout (pot 12): AA={:+.3}  98s={:+.3}  72o={:+.3}", aa6, m6, t6);
}
