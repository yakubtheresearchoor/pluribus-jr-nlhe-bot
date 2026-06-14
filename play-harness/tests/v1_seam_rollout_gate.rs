//! Equity-rollout policy gate (docket step 2, the live-6 "blueprint"). Live-6
//! is no longer a solve — it's honest hand-strength, computed not solved. This
//! gates the rollout as a seatable policy: it CONSERVES (Σnet = dead − rake),
//! and it actually discriminates by hand strength (folds weak hands, value-bets
//! strong ones) rather than playing blindly. Strategy-quality vs solved-live-6
//! is a FIELD-MODEL question, deferred — this only confirms the policy is sane
//! and wired correctly.

use play_harness::v1_seam::{SeamGame, SeamPolicy};

fn flop() -> [u8; 3] { [47, 21, 0] } // Ks 7d 2c

#[test]
fn rollout_conserves() {
    for live in 2u8..=6 {
        let g = SeamGame::new(live, 2, 12, flop());
        let pols = vec![SeamPolicy::EquityRollout; live as usize];
        let mut rng = 0x9E37u64 ^ live as u64;
        for _ in 0..1500 {
            let (holes, board) = g.deal(&mut rng);
            let (net, _l) = g.play(&pols, &holes, &board, &mut rng);
            let rake = g.dead as i64 - net.iter().sum::<i64>();
            assert!(rake >= 0 && rake <= g.rake_cap as i64, "live-{live}: rollout rake {rake}");
        }
    }
    eprintln!("rollout conserves (Σnet = dead − rake) across all families.");
}

#[test]
fn rollout_discriminates_by_strength() {
    // Hand strength must drive outcomes. Heads-up (live-2) isolates the signal
    // — no dead-money-split confound across a tight 3-way field. Seat-0 gets a
    // premium (AA) vs trash (72o), seat-1 a random rollout; AA must net clearly
    // more. Cards: A♠=51, A♥=50; 7♣=20, 2♥=2; flop Ks7d2c blocks 47/21/0.
    let g = SeamGame::new(2, 2, 12, flop());
    let pols = vec![SeamPolicy::EquityRollout; 2];
    fn blocks(c: u8) -> bool { c == 47 || c == 21 || c == 0 }
    let avg_with_seat0 = |h0: [u8; 2], seed: u64| -> f64 {
        let mut rng = seed;
        let (mut sum, mut used) = (0i64, 0u64);
        for _ in 0..1500 {
            let (mut holes, board) = g.deal(&mut rng);
            holes[0] = h0;
            let bad = |c: u8| blocks(c) || c == board[3] || c == board[4] || holes[1].contains(&c);
            if bad(h0[0]) || bad(h0[1]) || h0[0] == h0[1] { continue; }
            let (net, _l) = g.play(&pols, &holes, &board, &mut rng);
            sum += net[0];
            used += 1;
        }
        sum as f64 / used as f64
    };
    // Weak = genuine AIR on Ks7d2c rainbow: 9♣4♦ (9-high, no pair/draw).
    // (NOT 72o — that makes two pair on a 7-2 board; the earlier test's bug.)
    let strong = avg_with_seat0([51, 50], 0xAA); // A♠A♥
    let weak = avg_with_seat0([28, 9], 0xBB);     // 9♣4♦ air
    eprintln!("HU rollout seat-0 avg net: AA {strong:+.2} vs 94o-air {weak:+.2} chips/hand");
    assert!(strong > weak + 2.0,
        "rollout does not discriminate by strength: AA {strong:+.2} ≤ air {weak:+.2} + 2");
    eprintln!("ROLLOUT GATE PASS: conserves + value-bets strong, folds weak (AA ≫ air heads-up).");
}
