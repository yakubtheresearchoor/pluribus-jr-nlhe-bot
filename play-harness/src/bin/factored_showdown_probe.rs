//! VALIDATION GATE for the factored multiway showdown (Piece 1, P0 #1).
//!
//! `factored_showdown_eq_cfv` approximates the exact O(nh^K) brute-force
//! multiway showdown by treating opponents as INDEPENDENT (ignoring inter-
//! opponent card removal). This probe measures that approximation error on a
//! live-3 river terminal (2 active opponents, equal contributions) before the
//! factored showdown is wired into any real-time solve.
//!
//! For each board it computes per-hand EV in chips two ways:
//!   exact:    brute-force cfv / joint reach-product normalizer (the truth)
//!   factored: factored  cfv / independence (Π tot) normalizer
//! and reports max/mean |EV_exact − EV_fact|, both absolute and as a fraction
//! of the pot. Small error ⇒ the factored showdown is safe to use.
//!
//! Run: cargo run --release -p play-harness --bin factored_showdown_probe

use solver_core::card::{card_from_str, Card};
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::showdown::{
    factored_showdown_eq_cfv, factored_total_reach_product, precompute_opp_masses,
    side_pot_showdown_cfv_with_rake, total_valid_reach_product,
};

fn main() {
    let card = |s: &str| card_from_str(s).unwrap();
    // Varied river textures: dry/paired/connected/flush/broadway.
    let boards = [
        ["Ks", "9d", "4c", "2h", "Ah"],
        ["7s", "7d", "2c", "Jh", "Qd"],
        ["6s", "5d", "4c", "3h", "9c"],
        ["As", "Ks", "Qs", "2s", "7s"],
        ["Td", "Jd", "Qh", "Kc", "9s"],
        ["2c", "2d", "2h", "5s", "9d"],
    ];

    let starting_pot = 30i32;
    let contrib = 50i32; // equal contributions for all 3 players
    let contributions = [contrib, contrib, contrib];
    // total pot at showdown = starting_pot + 3*contrib; pot for relative error.
    let pot = (starting_pot + 3 * contrib) as f32;

    println!(
        "FACTORED MULTIWAY SHOWDOWN VALIDATION (live-3 river, 2 active opp, equal contrib)\n\
         per-hand EV in chips; error = |exact − factored|; pot={pot:.0}\n"
    );
    println!(
        "{:<22} {:>5} {:>12} {:>12} {:>10} {:>10}",
        "board", "nh", "max|Δ| chip", "mean|Δ|chip", "max %pot", "mean %pot"
    );

    let mut worst_max = 0.0f32;
    for b in &boards {
        let board: Vec<Card> = b.iter().map(|s| card(s)).collect();
        let ranges = vec![vec![1.0f32; 1326]; 3];
        let table = ChanceTable::compute_river_start(&board, &ranges, 3);
        let nh = table.num_valid;
        let (so_str, so_idx, pl_str, pl_idx, _) = table.sorted_opp_arrays();
        let hc = &table.hand_cards;

        // Uniform reach for the two opponents (players 1, 2).
        let r0: Vec<f32> = table.initial_weights[1].clone();
        let r1: Vec<f32> = table.initial_weights[2].clone();
        let opp_reach: Vec<&[f32]> = vec![&r0, &r1];

        let exact = side_pot_showdown_cfv_with_rake(
            &opp_reach, hc, nh, &so_str, &so_idx, &pl_str, &pl_idx,
            &contributions, 0, 0, 3, starting_pot, 0.0, 0.0, true,
        );
        let fact = factored_showdown_eq_cfv(
            &opp_reach, hc, nh, &so_str, &so_idx, &pl_str, &pl_idx,
            &contributions, 0, 3, starting_pot, 0.0, 0.0, true,
        );

        // Per-hand normalizers (the per-hand total reach mass each path divides by):
        //   joint   = total_valid_reach_product (exact, excludes inter-opp conflicts)
        //   independence = Π_oi masses.r(oi, h)  (the factored path's implicit mass)
        let hand_strength: Vec<u16> = (0..nh).map(|h| table.hand_ranks_base[h] + 1).collect();
        let masses = precompute_opp_masses(&opp_reach, hc, &hand_strength, 0);
        let joint = total_valid_reach_product(&masses, &opp_reach);

        let mut max_d = 0.0f32;
        let mut sum_d = 0.0f32;
        let mut cnt = 0usize;
        for h in 0..nh {
            let joint_nc = joint[h];
            let indep_nc = (masses.r(0, h) * masses.r(1, h)) as f64;
            if indep_nc < 1.0 || joint_nc < 1.0 {
                continue; // degenerate (h dominates removal); skip
            }
            let ev_exact = exact[h] as f64 / joint_nc;
            let ev_fact = fact[h] as f64 / indep_nc;
            let d = (ev_exact - ev_fact).abs() as f32;
            if d > max_d {
                max_d = d;
            }
            sum_d += d;
            cnt += 1;
        }
        let mean_d = if cnt > 0 { sum_d / cnt as f32 } else { 0.0 };
        worst_max = worst_max.max(max_d);
        println!(
            "{:<22} {:>5} {:>12.4} {:>12.4} {:>9.3}% {:>9.3}%",
            b.join(""),
            nh,
            max_d,
            mean_d,
            100.0 * max_d / pot,
            100.0 * mean_d / pot,
        );
    }
    println!("\nworst max |Δ| across boards: {worst_max:.4} chips ({:.3}% of pot)", 100.0 * worst_max / pot);

    // ---- Constant-payoff reach product: factored Π tot vs exact joint TVRP ----
    // Validates the OTHER factored approximation (the fold / uncontested-win
    // path: cfv = payoff × reach_product). Relative error in the reach product.
    println!("\nCONSTANT-PAYOFF REACH PRODUCT (factored Π tot vs exact joint TVRP):");
    println!("{:<22} {:>5} {:>12} {:>12}", "board", "nh", "max rel%", "mean rel%");
    for b in &boards {
        let board: Vec<Card> = b.iter().map(|s| card(s)).collect();
        let ranges = vec![vec![1.0f32; 1326]; 3];
        let table = ChanceTable::compute_river_start(&board, &ranges, 3);
        let nh = table.num_valid;
        let hc = &table.hand_cards;
        let r0 = table.initial_weights[1].clone();
        let r1 = table.initial_weights[2].clone();
        let opp_reach: Vec<&[f32]> = vec![&r0, &r1];
        let hand_strength: Vec<u16> = (0..nh).map(|h| table.hand_ranks_base[h] + 1).collect();
        let masses = precompute_opp_masses(&opp_reach, hc, &hand_strength, 0);
        let joint = total_valid_reach_product(&masses, &opp_reach);
        let fact = factored_total_reach_product(&opp_reach, hc, nh);
        let mut max_r = 0.0f64;
        let mut sum_r = 0.0f64;
        let mut cnt = 0usize;
        for h in 0..nh {
            if joint[h] < 1.0 {
                continue;
            }
            let rel = ((fact[h] as f64 - joint[h]) / joint[h]).abs();
            if rel > max_r {
                max_r = rel;
            }
            sum_r += rel;
            cnt += 1;
        }
        let mean_r = if cnt > 0 { sum_r / cnt as f64 } else { 0.0 };
        println!("{:<22} {:>5} {:>11.3}% {:>11.3}%", b.join(""), nh, 100.0 * max_r, 100.0 * mean_r);
    }

    if std::env::var("LAT").as_deref() != Ok("1") {
        println!("\n(set LAT=1 for the per-terminal latency section)");
        return;
    }

    // ---- LATENCY: per-terminal showdown cost, exact O(nh^K) vs factored O(nh·2^K) ----
    // This is the cost that dominates a full-nh multiway turn/river re-solve.
    use std::time::Instant;
    println!("\nPER-TERMINAL SHOWDOWN COST (exact brute-force vs factored):");
    println!("{:<10} {:>4} {:>5} {:>14} {:>14} {:>9}", "players", "K", "nh", "exact ms/call", "fact ms/call", "speedup");
    let board5: Vec<Card> = ["Ks", "9d", "4c", "2h", "Ah"].iter().map(|s| card(s)).collect();
    for np in [3u8, 4] {
        let ranges = vec![vec![1.0f32; 1326]; np as usize];
        let table = ChanceTable::compute_river_start(&board5, &ranges, np);
        let nh = table.num_valid;
        let (so_str, so_idx, pl_str, pl_idx, _) = table.sorted_opp_arrays();
        let hc = &table.hand_cards;
        let k = np as usize - 1;
        let reach: Vec<Vec<f32>> = (0..k).map(|i| table.initial_weights[i + 1].clone()).collect();
        let opp_reach: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
        let contributions = vec![contrib; np as usize];

        // exact: K=3 (np=4) brute force is O(nh^3) ≈ 1.3e9 — UNUSABLE per terminal
        // (measured: minutes/call). Only time the K=2 (live-3) exact; for K≥3
        // report N/A (each call is O(nh) × the K=2 cost ≈ thousands of seconds).
        let exact_ms = if k >= 3 {
            f64::NAN
        } else {
            let reps_exact = 5;
            let t = Instant::now();
            for _ in 0..reps_exact {
                let _ = side_pot_showdown_cfv_with_rake(
                    &opp_reach, hc, nh, &so_str, &so_idx, &pl_str, &pl_idx,
                    &contributions, 0, 0, np, starting_pot, 0.0, 0.0, true,
                );
            }
            t.elapsed().as_secs_f64() * 1000.0 / reps_exact as f64
        };

        let reps_fact = 50;
        let t = Instant::now();
        for _ in 0..reps_fact {
            let _ = factored_showdown_eq_cfv(
                &opp_reach, hc, nh, &so_str, &so_idx, &pl_str, &pl_idx,
                &contributions, 0, np, starting_pot, 0.0, 0.0, true,
            );
        }
        let fact_ms = t.elapsed().as_secs_f64() * 1000.0 / reps_fact as f64;
        if exact_ms.is_nan() {
            println!(
                "{:<10} {:>4} {:>5} {:>14} {:>14.3} {:>9}",
                format!("live-{}", np), k, nh, "N/A (~1000s)", fact_ms, "—"
            );
        } else {
            println!(
                "{:<10} {:>4} {:>5} {:>14.3} {:>14.3} {:>8.0}x",
                format!("live-{}", np), k, nh, exact_ms, fact_ms, exact_ms / fact_ms
            );
        }
    }
}
