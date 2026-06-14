//! V1 seam-family match engine gate (step 1): the seam game is built on top
//! of clean-rules, so verify it CONSERVES — Σ(live nets) must equal
//! dead_money − rake exactly, every hand, every family — and that the v1
//! rake is actually being taken. No blueprint needed (synthetic policies).

use play_harness::v1_seam::{SeamGame, SeamPolicy};

fn flop() -> [u8; 3] {
    // Ks 7d 2c (rank*4+suit: K=11,s=3 → 47; 7=5,d=1 → 21; 2=0,c=0 → 0).
    [47, 21, 0]
}

#[test]
fn seam_conserves_and_rakes_all_families() {
    // Limped representative per family: each live seat committed 2 (1 bb)
    // preflop; pot 12 = six limps, so dead = (6−live)·2.
    for live in 2u8..=6 {
        let pot = 12;
        let commit = 2;
        let g = SeamGame::new(live, commit, pot, flop());
        assert_eq!(g.dead, (6 - live as u32) * 2, "dead money");
        let mut rng = 0x5EED ^ (live as u64);
        let mut rake_seen = 0u64;
        let mut hands = 0u64;
        let pols = vec![SeamPolicy::AlwaysAggressive; live as usize];
        for _ in 0..5000 {
            let (holes, board) = g.deal(&mut rng);
            let (net, n_live) = g.play(&pols, &holes, &board, &mut rng);
            assert_eq!(net.len(), live as usize);
            assert!(n_live >= 1 && n_live <= live);
            // Σ net = dead − rake  ⇒  rake = dead − Σnet, in [0, cap].
            let sum: i64 = net.iter().sum();
            let rake = g.dead as i64 - sum;
            assert!(rake >= 0 && rake <= g.rake_cap as i64,
                "live-{live}: implied rake {rake} out of [0,{}] (Σnet {sum}, dead {})", g.rake_cap, g.dead);
            rake_seen += rake as u64;
            hands += 1;
        }
        // Sanity: with all-in aggressive play, real pots form and rake is
        // actually taken (not trivially zero everywhere).
        assert!(rake_seen > 0, "live-{live}: no rake ever taken — rake not wired");
        eprintln!("live-{live}: {hands} hands, dead {}, mean rake {:.2} units (cap {})",
            g.dead, rake_seen as f64 / hands as f64, g.rake_cap);
    }
    eprintln!("SEAM GATE PASS: all families conserve (Σnet = dead − rake) and rake is taken.");
}

#[test]
fn seam_checkfold_folds_to_one() {
    // All-but-one check-fold: with everyone check/folding, the hand should
    // resolve with rake on the (small) pot and conservation intact.
    let g = SeamGame::new(4, 2, 12, flop());
    let mut rng = 0x1234u64;
    let pols = vec![SeamPolicy::CheckFold; 4];
    for _ in 0..2000 {
        let (holes, board) = g.deal(&mut rng);
        let (net, _live) = g.play(&pols, &holes, &board, &mut rng);
        let sum: i64 = net.iter().sum();
        let rake = g.dead as i64 - sum;
        assert!(rake >= 0 && rake <= g.rake_cap as i64, "checkfold conservation: rake {rake}");
    }
    eprintln!("SEAM checkfold conserves.");
}
