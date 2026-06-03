// Standing showdown-validation gate built around a parameterized
// first-principles independent enumerator. This is the generalization of
// tests/independent_fold_enumerator.rs (HU fold) and
// tests/independent_enumerator_3p.rs (3p fold) into a full coverage gate.
//
// METHODOLOGY (the durable discipline from #37):
// The enumerator computes per-hand showdown CFV by direct enumeration
// over all valid opponent hand assignments — no formulas, no shortcuts,
// just explicit card-conflict checks and direct strength comparisons.
// CPU side_pot_showdown_cfv is then validated against this oracle at f32
// floor for every (np, terminal type, fold pattern, contribution config)
// case in the test battery. Cases that previously passed against
// side_pot_showdown_cfv as the oracle but fail against this independent
// reference would reveal where the buggy formula had been the reference.
//
// AFTER #37 FIX, all cases must agree. Going forward, any new bug in
// any showdown code path fails this gate.

use solver_core::solver::showdown::side_pot_showdown_cfv;

/// Outcome of comparing one hand-pair: 1.0 = traverser strict win,
/// 0.0 = strict loss, 0.5 = tie (split). For multiway, this generalizes
/// to (num_winners_inclusive, is_traverser_winner).
fn pair_outcome(trav_str: u16, opp_str: u16) -> f32 {
    if trav_str > opp_str { 1.0 }
    else if trav_str < opp_str { 0.0 }
    else { 0.5 }
}

/// Test case: fully specifies the showdown scenario.
#[derive(Clone, Debug)]
struct ShowdownCase {
    name: &'static str,
    np: usize,
    nh: usize,
    traverser: usize,
    starting_pot: i32,
    contributions: Vec<i32>,
    fold_mask: u16,
    hand_cards: Vec<u8>,        // [nh * 2] distinct cards across hands
    hand_strength: Vec<u16>,    // [nh] arbitrary distinct ranks (descending strength = larger u16)
    opp_reach: Vec<Vec<f32>>,   // [num_opp][nh]
    num_combinations: f32,
}

/// Independent first-principles enumerator. Computes CFV(h) directly by
/// enumerating all valid joint opponent hand assignments. No clever
/// formulas — just explicit conflict checks and explicit pot-share math.
///
/// For each opponent hand g_i, skip if any card conflicts with the
/// traverser's hand h or with any earlier opponent's hand. Sum joint
/// reach product weighted by the per-scenario payoff.
fn independent_showdown_cfv(tc: &ShowdownCase) -> Vec<f32> {
    let np = tc.np;
    let nh = tc.nh;
    let num_opp = np - 1;
    let c_t = tc.contributions[tc.traverser];
    let traverser_folded = tc.fold_mask & (1u16 << tc.traverser) != 0;

    let mut cfv = vec![0.0f32; nh];

    for h in 0..nh {
        let hc1 = tc.hand_cards[h * 2];
        let hc2 = tc.hand_cards[h * 2 + 1];

        // Enumerate joint opp hand assignments recursively.
        let mut accum = 0.0f32;
        enumerate_opp_assignments(
            tc, h, hc1, hc2, 0, 1.0,
            &mut vec![0usize; num_opp],
            &mut accum,
            traverser_folded, c_t,
        );

        cfv[h] = accum / tc.num_combinations;
    }
    cfv
}

fn enumerate_opp_assignments(
    tc: &ShowdownCase,
    h: usize,
    hc1: u8, hc2: u8,
    oi: usize,
    reach_product: f32,
    assignment: &mut Vec<usize>,
    accum: &mut f32,
    traverser_folded: bool,
    c_t: i32,
) {
    let np = tc.np;
    let num_opp = np - 1;

    if oi == num_opp {
        // Complete assignment. Compute per-scenario payoff for traverser
        // holding hand h with this opponent hand assignment.
        let payoff = scenario_payoff(tc, h, assignment, traverser_folded, c_t);
        *accum += reach_product * payoff;
        return;
    }

    let p = if oi < tc.traverser { oi } else { oi + 1 };
    let p_folded = tc.fold_mask & (1u16 << p) != 0;

    for g in 0..tc.nh {
        // Conflict with traverser?
        let gc1 = tc.hand_cards[g * 2];
        let gc2 = tc.hand_cards[g * 2 + 1];
        if gc1 == hc1 || gc1 == hc2 || gc2 == hc1 || gc2 == hc2 { continue; }
        // Conflict with earlier opponents in this assignment?
        let mut conflict = false;
        for k in 0..oi {
            let earlier_oi = k;
            let earlier_p = if earlier_oi < tc.traverser { earlier_oi } else { earlier_oi + 1 };
            // Folded opponents still removed cards from the deck.
            let _ = earlier_p;
            let prev_g = assignment[k];
            let pc1 = tc.hand_cards[prev_g * 2];
            let pc2 = tc.hand_cards[prev_g * 2 + 1];
            if gc1 == pc1 || gc1 == pc2 || gc2 == pc1 || gc2 == pc2 { conflict = true; break; }
        }
        if conflict { continue; }

        let r = tc.opp_reach[oi][g];
        if r == 0.0 { continue; }

        // For folded opponents, reach still contributes (they got here
        // via fold, which has its own strategy probability), but they
        // can't WIN.
        assignment[oi] = g;
        enumerate_opp_assignments(
            tc, h, hc1, hc2, oi + 1, reach_product * r,
            assignment, accum, traverser_folded, c_t,
        );

        // Suppress unused
        let _ = p_folded;
    }
}

/// Per-scenario payoff: given traverser holds h and opponents hold
/// `assignment[oi]`, compute traverser's expected utility.
/// Handles fold-forfeit, single-pot showdown, side-pot multi-level.
fn scenario_payoff(
    tc: &ShowdownCase,
    h: usize,
    assignment: &[usize],
    traverser_folded: bool,
    c_t: i32,
) -> f32 {
    let np = tc.np;
    let starting_pot = tc.starting_pot;
    let traverser_investment = starting_pot as f32 / np as f32 + c_t as f32;

    if traverser_folded {
        return -traverser_investment;
    }

    // Identify per-player contributions and fold status.
    let mut is_folded = vec![false; np];
    for p in 0..np {
        is_folded[p] = tc.fold_mask & (1u16 << p) != 0;
    }

    let num_active = (0..np).filter(|&p| !is_folded[p]).count();
    if num_active == 1 {
        // Only traverser active → wins everything not yet returned.
        let total_pot: i32 = starting_pot + tc.contributions.iter().sum::<i32>();
        return total_pot as f32 - traverser_investment;
    }

    // Multi-level side-pot showdown. Build sorted unique contribution levels
    // (ascending), then for each level: pot at that level = (level - prev_level) *
    // (num players contributing >= level). Add starting_pot to first level.
    // Traverser eligible at level iff c_t >= level. Active eligible opponents at
    // each level determined by their contribution and fold status.
    let mut levels: Vec<i32> = (0..np).map(|p| tc.contributions[p]).collect();
    levels.sort();
    levels.dedup();

    let mut cash: f32 = 0.0;
    let mut prev_l = 0i32;
    for (li, &lev) in levels.iter().enumerate() {
        let pc = lev - prev_l;
        if pc == 0 { prev_l = lev; continue; }
        let num_contrib = (0..np).filter(|&p| tc.contributions[p] >= lev).count();
        let mut pot_l = (pc * num_contrib as i32) as f32;
        if li == 0 { pot_l += starting_pot as f32; }

        let trav_elig = c_t >= lev;
        if !trav_elig { prev_l = lev; continue; }

        // Find active eligible opponents at this level.
        let elig_opps: Vec<usize> = (0..np)
            .filter(|&p| p != tc.traverser && !is_folded[p] && tc.contributions[p] >= lev)
            .collect();

        // Compare traverser strength vs eligible opps' strengths at this level.
        let trav_str = tc.hand_strength[h];
        // Find max strength among eligible (traverser + opps).
        let mut max_str = trav_str;
        for &p in &elig_opps {
            // Map player p to its opponent-index oi = (p < traverser) ? p : p-1
            let oi = if p < tc.traverser { p } else { p - 1 };
            let opp_str = tc.hand_strength[assignment[oi]];
            if opp_str > max_str { max_str = opp_str; }
        }
        // Tie count at top.
        let mut tied = 0;
        if trav_str == max_str { tied += 1; }
        for &p in &elig_opps {
            let oi = if p < tc.traverser { p } else { p - 1 };
            let opp_str = tc.hand_strength[assignment[oi]];
            if opp_str == max_str { tied += 1; }
        }
        if trav_str == max_str {
            cash += pot_l / (tied as f32);
        }
        prev_l = lev;
    }
    cash - traverser_investment
}

/// Cross-validate CPU side_pot_showdown_cfv against the independent
/// enumerator. Asserts agreement at f32 floor.
fn validate_case(tc: &ShowdownCase) {
    let oracle = independent_showdown_cfv(tc);

    // Build sorted_pl arrays (descending strength, in pair-index order)
    // — required by side_pot_showdown_cfv even though some paths don't
    // use them. ASCENDING per production convention.
    let mut items: Vec<(u16, u16)> = (0..tc.nh)
        .map(|h| (tc.hand_strength[h], h as u16))
        .collect();
    items.sort_by_key(|&(s, _)| s);
    let sorted_str: Vec<u16> = items.iter().map(|&(s, _)| s).collect();
    let sorted_idx: Vec<u16> = items.iter().map(|&(_, i)| i).collect();

    let opp_reach_views: Vec<&[f32]> = tc.opp_reach.iter().map(|v| v.as_slice()).collect();
    let cpu_raw = side_pot_showdown_cfv(
        &opp_reach_views, &tc.hand_cards, tc.nh,
        &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
        &tc.contributions, tc.fold_mask, tc.traverser, tc.np as u8,
        tc.starting_pot,
    );
    let cpu: Vec<f32> = cpu_raw.iter().map(|&v| v / tc.num_combinations).collect();

    let mut max_diff = 0.0f32;
    for h in 0..tc.nh {
        let d = (oracle[h] - cpu[h]).abs();
        if d > max_diff { max_diff = d; }
    }
    eprintln!(
        "  [{:50}] oracle={:?} cpu={:?} max_diff={:.6e}",
        tc.name, oracle, cpu, max_diff
    );
    assert!(
        max_diff < 1e-4,
        "[{}] CPU disagrees with independent enumerator at f32 floor. \
         oracle = {:?}, cpu = {:?}, max_diff = {}",
        tc.name, oracle, cpu, max_diff
    );
}

/// Standard hand-card layout for nh=4: 8 distinct cards.
fn distinct_hand_cards_nh4() -> (Vec<u8>, Vec<u16>) {
    let hand_cards = vec![46u8, 50, 38, 42, 30, 34, 18, 26];
    let hand_strength = vec![4136u16, 3624, 2472, 2168]; // ascending in pair-index order → descending strength
    (hand_cards, hand_strength)
}

/// Larger hand-card layout for nh=12: 24 distinct cards, supports up to
/// 6p meaningful joint enumerations (5 opp × 2 = 10 cards from 22
/// non-traverser-cards). Strengths chosen distinct, descending order.
fn distinct_hand_cards_nh12() -> (Vec<u8>, Vec<u16>) {
    // 12 hands × 2 cards = 24 distinct cards (indices 0..24).
    let hand_cards: Vec<u8> = (0u8..24).collect();
    // Strengths: arbitrary distinct values, hand 0 weakest, hand 11 strongest.
    let hand_strength: Vec<u16> = (0u16..12).map(|i| 100 + i * 50).collect();
    (hand_cards, hand_strength)
}

#[test]
fn standing_showdown_oracle_battery() {
    eprintln!("\n=== Standing showdown-oracle validation battery ===");

    let (hand_cards, hand_strength) = distinct_hand_cards_nh4();
    let nh = 4;
    let r_uniform = vec![0.125f32; nh];

    let mut cases: Vec<ShowdownCase> = Vec::new();

    // ---- HU (2p) ----
    // Showdown, equal contributions, no folds.
    cases.push(ShowdownCase {
        name: "HU showdown equal",
        np: 2, nh, traverser: 0, starting_pot: 10,
        contributions: vec![5, 5], fold_mask: 0,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_uniform.clone()],
        num_combinations: 16.0,
    });
    // Fold-forfeit, traverser=0 folded after committing 5; p1 reached at 15.
    cases.push(ShowdownCase {
        name: "HU fold p0 folded (trav)",
        np: 2, nh, traverser: 0, starting_pot: 10,
        contributions: vec![5, 15], fold_mask: 0b01,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_uniform.clone()],
        num_combinations: 16.0,
    });
    // Fold-forfeit, traverser=0 active, p1 folded at 5 after p0 bet to 15.
    cases.push(ShowdownCase {
        name: "HU fold p1 folded (opp)",
        np: 2, nh, traverser: 0, starting_pot: 10,
        contributions: vec![15, 5], fold_mask: 0b10,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_uniform.clone()],
        num_combinations: 16.0,
    });
    // Showdown with non-uniform opp_reach (some hands have 0 reach).
    cases.push(ShowdownCase {
        name: "HU showdown non-uniform opp_reach",
        np: 2, nh, traverser: 0, starting_pot: 10,
        contributions: vec![5, 5], fold_mask: 0,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![vec![0.5f32, 0.25, 0.0, 0.25]],
        num_combinations: 16.0,
    });
    // Side pot: HU asymmetric contributions (p0=5, p1=15) but both active.
    // (Tests the min-active-contrib half_pot logic with unequal contribs.)
    cases.push(ShowdownCase {
        name: "HU side-pot asymmetric (5 vs 15)",
        np: 2, nh, traverser: 0, starting_pot: 10,
        contributions: vec![5, 15], fold_mask: 0,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_uniform.clone()],
        num_combinations: 16.0,
    });

    // ---- 3p ----
    let r_quarter = vec![0.25f32; nh];
    // Showdown, equal contributions, no folds.
    cases.push(ShowdownCase {
        name: "3p showdown equal",
        np: 3, nh, traverser: 0, starting_pot: 15,
        contributions: vec![5, 5, 5], fold_mask: 0,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
        num_combinations: 64.0,
    });
    // Fold-forfeit, traverser=0 folded.
    cases.push(ShowdownCase {
        name: "3p fold p0 folded (trav)",
        np: 3, nh, traverser: 0, starting_pot: 5,
        contributions: vec![5, 15, 15], fold_mask: 0b001,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
        num_combinations: 64.0,
    });
    // Single fold: p1 folded, p0 and p2 active at showdown.
    cases.push(ShowdownCase {
        name: "3p single fold (p1 folded, p0+p2 showdown)",
        np: 3, nh, traverser: 0, starting_pot: 5,
        contributions: vec![15, 5, 15], fold_mask: 0b010,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
        num_combinations: 64.0,
    });
    // Side-pot 3p: p2 short-stack all-in at 5, p0 and p1 at 15.
    cases.push(ShowdownCase {
        name: "3p side-pot (p2 short-stack)",
        np: 3, nh, traverser: 0, starting_pot: 5,
        contributions: vec![15, 15, 5], fold_mask: 0,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
        num_combinations: 64.0,
    });

    // ---- 4p, 5p, 6p ----
    // Showdown equal, traverser=0.
    let cfg_np = |np: usize, label: &'static str, starting_pot: i32, num_c: f32| {
        let mut reach = Vec::new();
        for _ in 0..(np - 1) { reach.push(r_uniform.clone()); }
        ShowdownCase {
            name: label,
            np, nh, traverser: 0, starting_pot,
            contributions: vec![5; np], fold_mask: 0,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: reach,
            num_combinations: num_c,
        }
    };
    cases.push(cfg_np(4, "4p showdown equal", 20, 256.0));
    cases.push(cfg_np(5, "5p showdown equal", 25, 1024.0));
    cases.push(cfg_np(6, "6p showdown equal", 30, 4096.0));

    // 6p single fold (p3 folded, traverser=0 active).
    {
        let mut reach = Vec::new();
        for _ in 0..5 { reach.push(r_uniform.clone()); }
        cases.push(ShowdownCase {
            name: "6p single fold (p3 folded)",
            np: 6, nh, traverser: 0, starting_pot: 30,
            contributions: vec![5, 5, 5, 5, 5, 5], fold_mask: 0b001000,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: reach,
            num_combinations: 4096.0,
        });
    }

    // 6p traverser folded.
    {
        let mut reach = Vec::new();
        for _ in 0..5 { reach.push(r_uniform.clone()); }
        cases.push(ShowdownCase {
            name: "6p traverser folded",
            np: 6, nh, traverser: 0, starting_pot: 30,
            contributions: vec![5, 15, 15, 15, 15, 15], fold_mask: 0b000001,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: reach,
            num_combinations: 4096.0,
        });
    }

    // ---- 4p, 5p, 6p with larger nh (meaningful joint enumerations) ----
    let (hc12, hs12) = distinct_hand_cards_nh12();
    let nh12 = 12;
    let r12 = vec![0.1f32; nh12];

    let cfg_np_big = |np: usize, label: &'static str, starting_pot: i32, num_c: f32| {
        let mut reach = Vec::new();
        for _ in 0..(np - 1) { reach.push(r12.clone()); }
        ShowdownCase {
            name: label,
            np, nh: nh12, traverser: 0, starting_pot,
            contributions: vec![5; np], fold_mask: 0,
            hand_cards: hc12.clone(), hand_strength: hs12.clone(),
            opp_reach: reach,
            num_combinations: (nh12 as f32).powi((np - 1) as i32),
        }
    };
    cases.push(cfg_np_big(4, "4p showdown equal (nh=12)", 20, 12u32.pow(3) as f32));
    cases.push(cfg_np_big(5, "5p showdown equal (nh=12)", 25, 12u32.pow(4) as f32));
    cases.push(cfg_np_big(6, "6p showdown equal (nh=12)", 30, 12u32.pow(5) as f32));

    // 6p single fold with nh=12 (meaningful enumeration).
    {
        let mut reach = Vec::new();
        for _ in 0..5 { reach.push(r12.clone()); }
        cases.push(ShowdownCase {
            name: "6p single fold p3 (nh=12)",
            np: 6, nh: nh12, traverser: 0, starting_pot: 30,
            contributions: vec![5, 5, 5, 5, 5, 5], fold_mask: 0b001000,
            hand_cards: hc12.clone(), hand_strength: hs12.clone(),
            opp_reach: reach,
            num_combinations: 12u32.pow(5) as f32,
        });
    }

    // 6p traverser folded (nh=12 meaningful enumeration).
    {
        let mut reach = Vec::new();
        for _ in 0..5 { reach.push(r12.clone()); }
        cases.push(ShowdownCase {
            name: "6p traverser folded (nh=12)",
            np: 6, nh: nh12, traverser: 0, starting_pot: 30,
            contributions: vec![5, 15, 15, 15, 15, 15], fold_mask: 0b000001,
            hand_cards: hc12.clone(), hand_strength: hs12.clone(),
            opp_reach: reach,
            num_combinations: 12u32.pow(5) as f32,
        });
    }

    // 6p side-pot multi-level (p5 short-stack at 5, others at 15).
    {
        let mut reach = Vec::new();
        for _ in 0..5 { reach.push(r12.clone()); }
        cases.push(ShowdownCase {
            name: "6p side-pot p5 short-stack (nh=12)",
            np: 6, nh: nh12, traverser: 0, starting_pot: 30,
            contributions: vec![15, 15, 15, 15, 15, 5], fold_mask: 0,
            hand_cards: hc12.clone(), hand_strength: hs12.clone(),
            opp_reach: reach,
            num_combinations: 12u32.pow(5) as f32,
        });
    }

    // 6p multi-fold + multi-level side-pot (p2 and p4 folded, p5 short).
    {
        let mut reach = Vec::new();
        for _ in 0..5 { reach.push(r12.clone()); }
        cases.push(ShowdownCase {
            name: "6p multi-fold + side-pot (nh=12)",
            np: 6, nh: nh12, traverser: 0, starting_pot: 30,
            contributions: vec![15, 15, 5, 15, 5, 5],
            fold_mask: 0b010100, // p2 and p4 folded
            hand_cards: hc12.clone(), hand_strength: hs12.clone(),
            opp_reach: reach,
            num_combinations: 12u32.pow(5) as f32,
        });
    }

    // Run them all.
    eprintln!("\nRunning {} cases against independent enumerator...\n", cases.len());
    for tc in &cases {
        validate_case(tc);
    }
    eprintln!("\n✓ All {} showdown configurations match independent enumerator at f32 floor.", cases.len());
}
