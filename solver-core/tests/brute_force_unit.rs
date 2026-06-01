/// Minimal unit test of the K=2 brute-force showdown logic.
/// Constructs 3 hands with NO card conflicts, NO ties, and verifies zero-sum.
use solver_core::solver::showdown::side_pot_showdown_cfv;

fn run_scenario(
    hand_cards: &[u8],
    sorted_pl_str: &[u16],
    sorted_pl_idx: &[u16],
    nh: usize,
    label: &str,
) -> f64 {
    let num_players = 3u8;
    let num_opp = 2;
    let mut sorted_opp_str: Vec<u16> = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx: Vec<u16> = Vec::with_capacity(num_opp * nh);
    for _ in 0..num_opp {
        sorted_opp_str.extend(sorted_pl_str);
        sorted_opp_idx.extend(sorted_pl_idx);
    }
    let uniform: Vec<f32> = vec![1.0; nh];
    let opp_reach_vec: Vec<Vec<f32>> = vec![uniform.clone(); 3];
    let contributions: Vec<i32> = vec![5, 5, 5];

    let mut total: f64 = 0.0;
    let mut per_h_total: Vec<f64> = vec![0.0; nh];
    for traverser in 0..3 {
        let opp_reach_views: Vec<&[f32]> = (0..num_players as usize)
            .filter(|&p| p != traverser)
            .map(|p| opp_reach_vec[p].as_slice())
            .collect();
        let cfv = side_pot_showdown_cfv(
            &opp_reach_views, hand_cards, nh,
            &sorted_opp_str, &sorted_opp_idx,
            sorted_pl_str, sorted_pl_idx,
            &contributions, 0, traverser, num_players, 15,
        );
        let s: f64 = cfv.iter().map(|&v| v as f64).sum();
        total += s;
        for h in 0..nh { per_h_total[h] += cfv[h] as f64; }
        eprintln!("  Traverser {}: cfv = {:?}, sum = {:.4}", traverser, cfv, s);
    }
    eprintln!("  [{}] TOTAL = {:.6}", label, total);
    total
}

/// Compute hand_strength array (the brute-force uses sorted_pl_str/idx to derive it)
fn make_sorted(strengths: &[u16]) -> (Vec<u16>, Vec<u16>) {
    let nh = strengths.len();
    let mut items: Vec<(u16, u16)> = (0..nh).map(|h| (strengths[h], h as u16)).collect();
    items.sort_by_key(|&(s, _)| s);
    let mut s_str = vec![0u16; nh];
    let mut s_idx = vec![0u16; nh];
    for i in 0..nh {
        s_str[i] = items[i].0;
        s_idx[i] = items[i].1;
    }
    (s_str, s_idx)
}

#[test]
fn brute_force_with_conflicts_no_ties() {
    // 4 hands with one conflict pair
    let nh = 4;
    // h0=[0,1], h1=[0,2] (conflict with h0 via 0), h2=[3,4], h3=[5,6]
    let hand_cards: Vec<u8> = vec![0,1, 0,2, 3,4, 5,6];
    let strengths: Vec<u16> = vec![10, 15, 20, 25];
    let (s_str, s_idx) = make_sorted(&strengths);
    eprintln!("\n=== 4 hands, h0-h1 conflict, no ties ===");
    let total = run_scenario(&hand_cards, &s_str, &s_idx, nh, "conflict");
    assert!(total.abs() < 1e-3, "Should be zero-sum: {}", total);
}

#[test]
fn brute_force_with_ties_no_conflicts() {
    let nh = 4;
    let hand_cards: Vec<u8> = vec![0,1, 2,3, 4,5, 6,7];
    // h0 ties h1 at strength 20; h2 = 10; h3 = 30
    let strengths: Vec<u16> = vec![20, 20, 10, 30];
    let (s_str, s_idx) = make_sorted(&strengths);
    eprintln!("\n=== 4 hands, no conflicts, h0=h1 tie ===");
    let total = run_scenario(&hand_cards, &s_str, &s_idx, nh, "tie");
    assert!(total.abs() < 1e-3, "Should be zero-sum with tie handling: {}", total);
}

/// Test 3-player [190, 5, 190] with P1 folded.
/// Per-scenario zero-sum must hold.
#[test]
fn brute_force_3p_fold_unequal() {
    use solver_core::solver::showdown::side_pot_showdown_cfv;

    let nh = 3;
    let num_players = 3u8;
    let hand_cards: Vec<u8> = vec![0,1, 2,3, 4,5];
    let strengths: Vec<u16> = vec![10, 20, 30];
    let (s_str, s_idx) = make_sorted(&strengths);
    let num_opp = 2;
    let mut sorted_opp_str: Vec<u16> = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx: Vec<u16> = Vec::with_capacity(num_opp * nh);
    for _ in 0..num_opp {
        sorted_opp_str.extend(&s_str);
        sorted_opp_idx.extend(&s_idx);
    }
    let uniform: Vec<f32> = vec![1.0; nh];
    let opp_reach_vec: Vec<Vec<f32>> = vec![uniform.clone(); 3];
    let contributions: Vec<i32> = vec![190, 5, 190];
    let fold_mask: u16 = 0b0010; // P1 folded

    eprintln!("\n=== 3-player [190, 5, 190] fm=010 (P1 folded) ===");
    let mut total: f64 = 0.0;
    for traverser in 0..3 {
        let opp_reach_views: Vec<&[f32]> = (0..num_players as usize)
            .filter(|&p| p != traverser)
            .map(|p| opp_reach_vec[p].as_slice())
            .collect();
        let cfv = side_pot_showdown_cfv(
            &opp_reach_views, &hand_cards, nh,
            &sorted_opp_str, &sorted_opp_idx,
            &s_str, &s_idx,
            &contributions, fold_mask, traverser, num_players, 15,
        );
        let s: f64 = cfv.iter().map(|&v| v as f64).sum();
        total += s;
        eprintln!("  Traverser {}: cfv = {:?}, sum = {:.4}", traverser, cfv, s);
    }
    eprintln!("  TOTAL = {:.6}", total);
    assert!(total.abs() < 1e-3, "Should be zero-sum: {}", total);
}

#[test]
fn brute_force_with_ties_at_top() {
    let nh = 3;
    let hand_cards: Vec<u8> = vec![0,1, 2,3, 4,5];
    // All three hands tie at strength 20
    let strengths: Vec<u16> = vec![20, 20, 20];
    let (s_str, s_idx) = make_sorted(&strengths);
    eprintln!("\n=== 3 hands, no conflicts, all tie ===");
    let total = run_scenario(&hand_cards, &s_str, &s_idx, nh, "all-tie");
    assert!(total.abs() < 1e-3, "Should be zero-sum with all-tie split: {}", total);
}

#[test]
fn brute_force_3p_no_ties_no_conflicts() {
    // 3 hands with distinct cards (no conflicts) and distinct strengths.
    let nh = 3;
    let num_players = 3u8;
    // Hand cards: each hand has 2 cards, all 6 cards distinct
    // h0 = [card 0, card 1]
    // h1 = [card 2, card 3]
    // h2 = [card 4, card 5]
    let hand_cards: Vec<u8> = vec![0, 1, 2, 3, 4, 5];

    // Strengths: h0 = 10, h1 = 20, h2 = 30 (h2 strongest)
    // sorted_pl_str (ascending): [10, 20, 30]
    // sorted_pl_idx: [0, 1, 2]
    let sorted_pl_str: Vec<u16> = vec![10, 20, 30];
    let sorted_pl_idx: Vec<u16> = vec![0, 1, 2];
    // Same arrays per opponent: shape [num_opp * nh]
    let num_opp = 2;
    let mut sorted_opp_str: Vec<u16> = Vec::with_capacity(num_opp * nh);
    let mut sorted_opp_idx: Vec<u16> = Vec::with_capacity(num_opp * nh);
    for _ in 0..num_opp {
        sorted_opp_str.extend(&sorted_pl_str);
        sorted_opp_idx.extend(&sorted_pl_idx);
    }

    // Uniform reach = 1.0
    let uniform: Vec<f32> = vec![1.0; nh];
    let opp_reach_vec: Vec<Vec<f32>> = vec![uniform.clone(); 3];

    // 3-way showdown, no folds, equal contributions
    let contributions: Vec<i32> = vec![5, 5, 5];
    let starting_pot = 15;

    let mut per_player_sum: Vec<f64> = vec![0.0; 3];
    let mut per_hand_per_traverser: Vec<Vec<f32>> = Vec::new();
    for traverser in 0..3 {
        let opp_reach_views: Vec<&[f32]> = (0..num_players as usize)
            .filter(|&p| p != traverser)
            .map(|p| opp_reach_vec[p].as_slice())
            .collect();
        let cfv = side_pot_showdown_cfv(
            &opp_reach_views, &hand_cards, nh,
            &sorted_opp_str, &sorted_opp_idx,
            &sorted_pl_str, &sorted_pl_idx,
            &contributions, 0, traverser, num_players, starting_pot,
        );
        let s: f64 = cfv.iter().map(|&v| v as f64).sum();
        per_player_sum[traverser] = s;
        per_hand_per_traverser.push(cfv);
        eprintln!("Traverser {}: cfv = {:?}, sum = {}", traverser, per_hand_per_traverser.last().unwrap(), s);
    }

    // Per-hand cross-player check (should be zero for SAME hand):
    // CFV_p[h] should be the same regardless of p (symmetric uniform game).
    for h in 0..nh {
        let v0 = per_hand_per_traverser[0][h];
        let v1 = per_hand_per_traverser[1][h];
        let v2 = per_hand_per_traverser[2][h];
        eprintln!("h={}: P0={:.4}, P1={:.4}, P2={:.4}, max diff = {:.6}",
            h, v0, v1, v2, (v0 - v1).abs().max((v1 - v2).abs()).max((v0 - v2).abs()));
    }

    let total: f64 = per_player_sum.iter().sum();
    eprintln!("\nTotal sum across all (player, hand): {}", total);
    eprintln!("Expected: 0 (zero-sum at terminal, uniform reach, no ties, no card conflicts among players)");

    // Hand-level check:
    // Only one physical scenario: P0=h0(10), P1=h1(20), P2=h2(30). h2 wins.
    // P0 payoff = -half_pot = -10
    // P1 payoff = -half_pot = -10
    // P2 payoff = K*half_pot = 2*10 = 20
    // Sum = 0
    //
    // But CFV[h] sums over ALL possible opponent hand assignments:
    // CFV_0[0] = sum over (g0, g1) of payoff_unit(h=0/str=10, g0, g1) * half_pot
    //          = (1,2): payoff(10, 20, 30) = -1 (loses to 30) → -10
    //          + (2,1): payoff(10, 30, 20) = -1 → -10
    //          = -20
    // CFV_0[1] = (0, 2): payoff(20, 10, 30) = -1 → -10
    //          + (2, 0): payoff(20, 30, 10) = -1 → -10
    //          = -20
    // CFV_0[2] = (0, 1): payoff(30, 10, 20) = 2 → 20
    //          + (1, 0): payoff(30, 20, 10) = 2 → 20
    //          = 40
    // Sum_h CFV_0[h] = -20 + -20 + 40 = 0
    //
    // Same for P1, P2. So total = 0.

    assert!(total.abs() < 1e-3, "Brute-force not zero-sum at terminal: {}", total);
}
