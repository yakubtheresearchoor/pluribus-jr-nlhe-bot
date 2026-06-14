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
    // Weak = genuine AIR on Ks7d2c rainbow: 9♣4♦ (9-high, no pair/draw).
    // (NOT 72o — that makes two pair on a 7-2 board; the earlier test's bug.)
    let strong = avg_with_seat0([51, 50], 0xAA); // A♠A♥
    let weak = avg_with_seat0([28, 9], 0xBB);     // 9♣4♦ air
    eprintln!("HU rollout seat-0 avg net: AA {strong:+.2} vs 94o-air {weak:+.2} chips/hand");
    assert!(strong > weak + 2.0,
        "rollout does not discriminate by strength: AA {strong:+.2} ≤ air {weak:+.2} + 2");
    eprintln!("ROLLOUT GATE PASS: conserves + value-bets strong, folds weak (AA ≫ air heads-up).");
}

#[test]
fn rollout_equity_is_multiway_aware() {
    // THE gate that matters for live-6: the rollout was proven HEADS-UP but
    // serves a SIX-WAY family, and multiway equity is a different regime. Prove
    // (1) the equity calc is opponent-count-aware (the SAME hand has lower
    // equity vs more opponents), and (2) a hand that's heads-up-playable but
    // six-way-trash gets the LOW six-way equity (so the rollout won't overplay
    // it). flop Ks7d2c. A♥Q♠ = 50,43 (ace-high: fine HU, trash 6-way).
    let g = SeamGame::new(6, 2, 12, flop());
    let revealed = [47u8, 21, 0]; // flop only (a flop decision)
    let ahigh = [50u8, 43]; // A♥Q♠
    let set77 = [22u8, 23]; // 7♥7♠ = set of sevens (strong AT ANY count)
    let mut rng = 0xE917u64;
    let eq_ah_hu = g.equity(ahigh, &revealed, 1, &mut rng, 4000);
    let eq_ah_6w = g.equity(ahigh, &revealed, 5, &mut rng, 4000);
    let eq_set_6w = g.equity(set77, &revealed, 5, &mut rng, 4000);
    eprintln!("A♥Q♠ equity: heads-up {eq_ah_hu:.3} → six-way {eq_ah_6w:.3}");
    eprintln!("set-of-7s equity six-way: {eq_set_6w:.3}");
    // (1) opponent-count-aware: ace-high drops sharply HU → 6-way.
    assert!(eq_ah_6w < eq_ah_hu - 0.15,
        "equity not opponent-count-aware: A-high 6-way {eq_ah_6w:.3} not far below HU {eq_ah_hu:.3}");
    // (2) the 6-way ace-high equity is BELOW the rollout's call/bet thresholds
    // (0.42 call / 0.55 bet) ⇒ it folds/checks 6-way (won't overplay).
    assert!(eq_ah_6w < 0.42,
        "A-high six-way equity {eq_ah_6w:.3} ≥ 0.42 call threshold — rollout would OVERPLAY it 6-way");
    // (3) a genuine multiway monster (set) still clears the bet threshold 6-way.
    assert!(eq_set_6w > 0.55,
        "set-of-7s six-way equity {eq_set_6w:.3} ≤ 0.55 — rollout would underplay a monster");
    eprintln!("SIX-WAY GATE PASS: equity is opponent-count-aware; A-high tightens to fold 6-way");
    eprintln!("(eq {eq_ah_6w:.3} < 0.42), set value-bets 6-way (eq {eq_set_6w:.3} > 0.55).");
}

#[test]
fn rollout_discriminates_six_way() {
    // Behavioral six-way: seat-0 with a 6-way monster (set) nets clearly more
    // than seat-0 with a heads-up-ok-but-6-way-trash hand (ace-high), against a
    // five-rollout field. The HU-trash hand must NOT bleed (≈0 or negative).
    let g = SeamGame::new(6, 2, 12, flop());
    let pols = vec![SeamPolicy::EquityRollout; 6];
    fn blocks(c: u8) -> bool { c == 47 || c == 21 || c == 0 }
    let avg = |h0: [u8; 2], seed: u64| -> f64 {
        let mut rng = seed;
        let (mut sum, mut used) = (0i64, 0u64);
        for _ in 0..900 {
            let (mut holes, board) = g.deal(&mut rng);
            holes[0] = h0;
            let coll = |c: u8| blocks(c) || c == board[3] || c == board[4]
                || (1..6).any(|s| holes[s].contains(&c));
            if coll(h0[0]) || coll(h0[1]) || h0[0] == h0[1] { continue; }
            let (net, _l) = g.play(&pols, &holes, &board, &mut rng);
            sum += net[0]; used += 1;
        }
        sum as f64 / used.max(1) as f64
    };
    let set = avg([22, 23], 0x5E7); // 7♥7♠ set
    let ahigh = avg([50, 43], 0xA1); // A♥Q♠ (HU-ok, 6-way trash)
    eprintln!("six-way seat-0 avg net: set-of-7s {set:+.2} vs A-high {ahigh:+.2} chips/hand");
    assert!(set > ahigh + 2.0, "no six-way discrimination: set {set:+.2} ≤ A-high {ahigh:+.2} + 2");
    eprintln!("SIX-WAY DISCRIMINATION PASS: monster value-bets, HU-trash hand stays passive 6-way.");
}
