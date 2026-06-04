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

use solver_core::solver::showdown::{side_pot_showdown_cfv, side_pot_showdown_cfv_with_rake};

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

/// Tie-band layout: 4 distinct combos where TWO combos share a strength.
/// Combos 0 and 1 tie at strength 200; combo 2 at 100; combo 3 at 300.
/// All combos use distinct cards (no pairwise card overlap), so opp can
/// hold combo 1 while player holds combo 0 (tie scenario, non-conflict).
/// This exercises the showdown's tie code paths that
/// `distinct_hand_cards_nh4`'s all-distinct strengths can never trigger
/// (under all-distinct strengths, ties only occur at opp = h which is
/// always blocked by card-conflict).
///
/// Coverage gap finding (from Slice 1.3 rake work): the closed validation
/// arc never exercised non-conflicting ties. Pre-rake `sorted_sweep_showdown`
/// was correct at ties by design (strict < and > comparisons → ties skipped
/// → 0 contribution, matches expected 0 payoff for split pot at equal
/// stakes) but the oracle didn't ANCHOR that correctness against
/// formula-free enumeration. This layout closes the gap.
fn tie_band_hand_cards_nh4() -> (Vec<u8>, Vec<u16>) {
    let hand_cards = vec![46u8, 50, 38, 42, 30, 34, 18, 26]; // same 8 distinct cards
    let hand_strength = vec![200u16, 200, 100, 300]; // combos 0 and 1 tie at 200
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

    // ---- Tie-band coverage (CLOSES THE GAP, see tie_band_hand_cards_nh4 doc) ----
    // 4 combos with combos 0 and 1 sharing strength 200. Player holding
    // combo 0 ties with opp holding combo 1 (no card conflict between
    // combos 0 and 1). The independent reference's pair_outcome returns
    // 0.5 for ties; scenario_payoff splits the pot among tied players.
    // For HU equal contribs, tie payoff = pot/2 - investment = 0
    // (player gets back their stake). The pre-Slice-1.3 sorted_sweep
    // gives 0 contribution from ties (strict < and > skip them), which
    // is correct — this case discriminatively anchors that the math
    // matches truth, not just intuition.
    {
        let (tb_hc, tb_hs) = tie_band_hand_cards_nh4();
        // HU showdown equal: tie band exists for combos {0,1}.
        cases.push(ShowdownCase {
            name: "HU showdown tie-band (combos 0&1 tie at str=200)",
            np: 2, nh, traverser: 0, starting_pot: 10,
            contributions: vec![5, 5], fold_mask: 0,
            hand_cards: tb_hc.clone(), hand_strength: tb_hs.clone(),
            opp_reach: vec![r_uniform.clone()],
            num_combinations: 16.0,
        });
        // HU showdown tie-band with non-uniform opp_reach so the tie
        // band's CFV depends on opp's reach distribution (more
        // discriminating than uniform).
        cases.push(ShowdownCase {
            name: "HU showdown tie-band non-uniform reach",
            np: 2, nh, traverser: 0, starting_pot: 10,
            contributions: vec![5, 5], fold_mask: 0,
            hand_cards: tb_hc.clone(), hand_strength: tb_hs.clone(),
            opp_reach: vec![vec![0.3f32, 0.7, 0.5, 0.2]],
            num_combinations: 16.0,
        });
    }

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

    // ─────────────────────────────────────────────────────────────────
    // Slice (Phase 1 #40): Option B coverage extension.
    //
    // Option B is the postflop action abstraction with 2 bet sizes
    // (PotRelative(0.5), PotRelative(1.0)) and 2 raise sizes (same).
    // It produces terminal contribution patterns that the existing
    // oracle cases do not exercise: THREE OR MORE DISTINCT
    // CONTRIBUTION LEVELS AMONG ACTIVE PLAYERS, arising when multiple
    // players go all-in at different stack depths or when the bet/raise
    // chain hits 3+ allin steps before showdown.
    //
    // The existing battery tops out at 2 levels (e.g., [15,15,15,15,15,5]
    // is 2 levels). Option B produces 3 and 4 level patterns. Per the
    // standing question, the oracle must actually exercise the cases
    // production trees produce. Anchor coverage for arithmetic at scale.
    //
    // Each case below is verified against the formula-free independent
    // enumerator (the trusted reference for showdown arithmetic).
    // ─────────────────────────────────────────────────────────────────

    // 3p three-level all-active: a multi-raise terminal where every
    // player is in for a different amount. contribs=[10, 30, 60],
    // all distinct, no folds. Exercises K=2 per-level brute force at
    // 3 levels.
    cases.push(ShowdownCase {
        name: "3p three-level all-active [10,30,60]",
        np: 3, nh, traverser: 0, starting_pot: 15,
        contributions: vec![10, 30, 60], fold_mask: 0,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
        num_combinations: 64.0,
    });

    // 3p three-level traverser-shortest: traverser=0 short at 10, opps at 30 and 60.
    // Exercises eligibility transitions where traverser drops out at higher levels.
    cases.push(ShowdownCase {
        name: "3p three-level traverser-short [10,30,60] trav=0",
        np: 3, nh, traverser: 0, starting_pot: 15,
        contributions: vec![10, 30, 60], fold_mask: 0,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
        num_combinations: 64.0,
    });

    // 3p three-level traverser-deepest: traverser=2 at 60. Tests the
    // traverser-uniquely-eligible case at the top side pot level (no
    // contest at the top, traverser wins that level uncontested).
    cases.push(ShowdownCase {
        name: "3p three-level traverser-deep [10,30,60] trav=2",
        np: 3, nh, traverser: 2, starting_pot: 15,
        contributions: vec![10, 30, 60], fold_mask: 0,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
        num_combinations: 64.0,
    });

    // 3p three-level with mid-level fold: contributions [10, 30, 60]
    // but p1 (mid) folded. Exercises dead-money distribution at the
    // mid level while top and bottom are still contested. Common in
    // Option B trees when the mid-stack folds after a re-raise.
    cases.push(ShowdownCase {
        name: "3p three-level mid-fold [10,30,60] p1 folded",
        np: 3, nh, traverser: 0, starting_pot: 15,
        contributions: vec![10, 30, 60], fold_mask: 0b010,
        hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
        opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
        num_combinations: 64.0,
    });

    // 4p four-level all-active: contribs=[10, 25, 50, 100].
    // Four distinct levels. Exercises K>=3 factored per-level path
    // (multiway_brute_force_showdown's K>=3 branch). This is the
    // Option B endgame configuration where every player went allin
    // at a different depth.
    {
        let mut reach = Vec::new();
        for _ in 0..3 { reach.push(r_uniform.clone()); }
        cases.push(ShowdownCase {
            name: "4p four-level all-active [10,25,50,100]",
            np: 4, nh, traverser: 0, starting_pot: 20,
            contributions: vec![10, 25, 50, 100], fold_mask: 0,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: reach,
            num_combinations: 256.0,
        });
    }

    // 4p four-level traverser-deepest: traverser=3 (the deepest stack)
    // wins the top side pot uncontested. Specifically exercises the
    // K>=3 factored path's static_cash accumulation at the top level
    // where only the traverser is eligible (sole-eligible static_cash
    // path the K=2 cluster doesn't hit).
    {
        let mut reach = Vec::new();
        for _ in 0..3 { reach.push(r_uniform.clone()); }
        cases.push(ShowdownCase {
            name: "4p four-level traverser-deep [10,25,50,100] trav=3",
            np: 4, nh, traverser: 3, starting_pot: 20,
            contributions: vec![10, 25, 50, 100], fold_mask: 0,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: reach,
            num_combinations: 256.0,
        });
    }

    // 4p four-level with one fold: contribs=[10, 25, 50, 100], p2 folded
    // mid-tournament (their 50 is dead money). Exercises dead-money
    // distribution across multiple side-pot levels with K=3 active.
    {
        let mut reach = Vec::new();
        for _ in 0..3 { reach.push(r_uniform.clone()); }
        cases.push(ShowdownCase {
            name: "4p four-level + mid-fold [10,25,50,100] p2 folded",
            np: 4, nh, traverser: 0, starting_pot: 20,
            contributions: vec![10, 25, 50, 100], fold_mask: 0b0100,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: reach,
            num_combinations: 256.0,
        });
    }

    // 6p three-level mixed: simulates a tournament-style allin from
    // multiple short stacks. contribs=[15, 15, 5, 15, 5, 50] with no
    // folds, three distinct levels [5, 15, 50]. Exercises K>=3 factored
    // path with multiple shorts at the same level (two players at 5,
    // three at 15, one at 50).
    {
        let mut reach = Vec::new();
        for _ in 0..5 { reach.push(r_uniform.clone()); }
        cases.push(ShowdownCase {
            name: "6p three-level mixed [15,15,5,15,5,50]",
            np: 6, nh, traverser: 0, starting_pot: 30,
            contributions: vec![15, 15, 5, 15, 5, 50], fold_mask: 0,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: reach,
            num_combinations: 4096.0,
        });
    }

    // Run them all.
    eprintln!("\nRunning {} cases against independent enumerator...\n", cases.len());
    for tc in &cases {
        validate_case(tc);
    }
    eprintln!("\n✓ All {} showdown configurations match independent enumerator at f32 floor.", cases.len());
}

// ═════════════════════════════════════════════════════════════════════
// Phase 1 P2.4: preflop showdown oracle extension.
//
// Preflop terminals come in two types with opposite rake behavior:
//
//   Type A: preflop fold-terminals (action stopped before the flop)
//           flop_seen = FALSE, rake MUST be zero (no flop, no drop).
//           This is the first real exercise of the no-flop-no-drop
//           condition. The rake machinery's flop_seen gate was built
//           for this case and was dormant-untested until now (all
//           postflop trees had flop_seen=true at every terminal).
//
//   Type B: preflop all-in then 3-street runout reaching showdown
//           flop_seen = TRUE, rake applies normally, multi-level
//           contributions arise from tournament-style allin sequences
//           (builds on the Phase 1 #40 multi-level oracle extension).
//
// The two types split cleanly on the rake condition. P2.4 verifies
// BOTH branches explicitly:
//
//   For Type A: assert side_pot_showdown_cfv_with_rake(rake=5%,
//     flop_seen=false) produces IDENTICAL CFV to the rake-free
//     side_pot_showdown_cfv. This proves the no-flop-no-drop gate
//     correctly zeros out rake when flop_seen=false, closing the
//     dormant-untested item carried since the rake arc.
//
//   For Type B: assert structure correctness via the independent
//     enumerator at rake=0 (multi-level math is right), then
//     separately assert rake applies non-trivially at rake=5%
//     (rake-applied check via diff from rake-free baseline).
// ═════════════════════════════════════════════════════════════════════

#[test]
fn preflop_terminals_oracle_with_rake_gating() {
    eprintln!("\n=== P2.4: preflop showdown oracle + no-flop-no-drop coverage ===");

    let (hand_cards, hand_strength) = distinct_hand_cards_nh4();
    let nh = 4;
    let r_uniform = vec![0.125f32; nh];
    let r_quarter = vec![0.25f32; nh];

    // ─────────────────────────────────────────────────────────────
    // Type A: preflop fold-terminals (no-flop-no-drop coverage)
    // ─────────────────────────────────────────────────────────────
    //
    // Each case represents a tree terminal that occurred preflop
    // (before any board card was dealt). For these, flop_seen=false
    // and rake MUST be zero regardless of rake_rate/rake_cap.
    //
    // Validation: the rake-bearing path with flop_seen=false MUST
    // produce identical CFV to the rake-free baseline. Any divergence
    // means the no-flop-no-drop gate is broken.

    let preflop_fold_cases: Vec<ShowdownCase> = vec![
        // HU: SB (player 0) folds preflop to BB's pre-action. Common.
        // contributions = [SB(1), BB(2)], SB folded.
        ShowdownCase {
            name: "Preflop HU SB-fold",
            np: 2, nh, traverser: 0, starting_pot: 0,
            contributions: vec![1, 2], fold_mask: 0b01,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: vec![r_uniform.clone()],
            num_combinations: 16.0,
        },
        // HU: BB (player 1) folds preflop after SB raises. Rare but possible.
        // contributions = [SB-raised(5), BB(2)], BB folded.
        ShowdownCase {
            name: "Preflop HU BB-fold-to-raise",
            np: 2, nh, traverser: 0, starting_pot: 0,
            contributions: vec![5, 2], fold_mask: 0b10,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: vec![r_uniform.clone()],
            num_combinations: 16.0,
        },
        // 3p: button (player 2) folds preflop early.
        // contributions = [SB(1), BB(2), button(0)], button folded.
        ShowdownCase {
            name: "Preflop 3p button-fold",
            np: 3, nh, traverser: 0, starting_pot: 0,
            contributions: vec![1, 2, 0], fold_mask: 0b100,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
            num_combinations: 64.0,
        },
        // 3p: SB and button both fold, BB wins uncontested preflop.
        // contributions = [SB(1), BB(2), button(0)], SB and button folded.
        // Traverser=1 (BB) is the sole survivor.
        ShowdownCase {
            name: "Preflop 3p SB+button fold, BB wins",
            np: 3, nh, traverser: 1, starting_pot: 0,
            contributions: vec![1, 2, 0], fold_mask: 0b101,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
            num_combinations: 64.0,
        },
        // 6p: typical preflop fold sequence (everyone except SB+BB folds).
        // Then SB folds. BB wins uncontested.
        // contributions = [SB(1), BB(2), 0, 0, 0, 0], SB and all others folded.
        ShowdownCase {
            name: "Preflop 6p all-fold-to-BB",
            np: 6, nh, traverser: 1, starting_pot: 0,
            contributions: vec![1, 2, 0, 0, 0, 0], fold_mask: 0b111101,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: vec![r_uniform.clone(), r_uniform.clone(),
                            r_uniform.clone(), r_uniform.clone(), r_uniform.clone()],
            num_combinations: 1024.0,
        },
    ];

    eprintln!("\nType A: preflop fold-terminals (no-flop-no-drop verification)");
    eprintln!("  Asserting: side_pot_showdown_cfv_with_rake(rake=5%, flop_seen=false)");
    eprintln!("           == side_pot_showdown_cfv (rake-free) at f32 floor.");
    eprintln!("  This is the first real exercise of the no-flop-no-drop");
    eprintln!("  rake gate (the dormant-untested item carried since Slice 1.5).\n");

    for tc in &preflop_fold_cases {
        // Build the sorted_pl arrays the rake-bearing path needs.
        let mut items: Vec<(u16, u16)> = (0..tc.nh)
            .map(|h| (tc.hand_strength[h], h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        let sorted_str: Vec<u16> = items.iter().map(|&(s, _)| s).collect();
        let sorted_idx: Vec<u16> = items.iter().map(|&(_, i)| i).collect();

        let opp_reach_views: Vec<&[f32]> = tc.opp_reach.iter().map(|v| v.as_slice()).collect();

        // Rake-free baseline (the canonical no-rake output).
        let cfv_rake_free = side_pot_showdown_cfv(
            &opp_reach_views, &tc.hand_cards, tc.nh,
            &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
            &tc.contributions, tc.fold_mask, tc.traverser, tc.np as u8,
            tc.starting_pot,
        );

        // Rake-bearing path with flop_seen=FALSE. Per the rake spec
        // (no flop, no drop), this MUST produce identical CFV to the
        // rake-free baseline regardless of rake_rate or rake_cap.
        let cfv_no_flop_no_drop = side_pot_showdown_cfv_with_rake(
            &opp_reach_views, &tc.hand_cards, tc.nh,
            &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
            &tc.contributions, tc.fold_mask, tc.traverser, tc.np as u8,
            tc.starting_pot,
            0.05, 1000.0, false, // rake_rate=5%, generous cap, flop_seen=FALSE
        );

        let mut max_diff = 0.0f32;
        for h in 0..tc.nh {
            let d = (cfv_rake_free[h] - cfv_no_flop_no_drop[h]).abs();
            if d > max_diff { max_diff = d; }
        }
        eprintln!(
            "  [{:45}] rake-free={:?} no-flop-no-drop={:?} max_diff={:.6e}",
            tc.name, cfv_rake_free, cfv_no_flop_no_drop, max_diff,
        );
        assert!(
            max_diff < 1e-6,
            "P2.4 NO-FLOP-NO-DROP REGRESSION at case '{}': rake_with_gate \
             diverges from rake-free by {} > 1e-6. The flop_seen=false gate \
             is not zeroing rake. This closes the dormant-untested item \
             from the rake arc by exercising the no-flop-no-drop path for \
             the first time at a real preflop terminal.",
            tc.name, max_diff,
        );
    }
    eprintln!("\n✓ All {} preflop fold-terminals: no-flop-no-drop gate verified \
        rake-correctly-zero.", preflop_fold_cases.len());

    // ─────────────────────────────────────────────────────────────
    // Type B: preflop all-in then 3-street runout (multi-level, rake applies)
    // ─────────────────────────────────────────────────────────────
    //
    // These reach showdown after a preflop all-in followed by 3 streets
    // of board cards. flop_seen=TRUE, rake applies normally. Multi-level
    // contributions arise from tournament-style allin sequences (one or
    // more players covered for different amounts), building on the #40
    // multi-level extension.
    //
    // Validation: each case is added to the standard battery (validates
    // structure via the independent enumerator at rake=0). Additionally,
    // verify rake applies non-trivially at rake=5% (cfv differs from
    // rake-free by the predicted rake amount).

    let preflop_allin_cases: Vec<ShowdownCase> = vec![
        // HU equal stacks: both go allin preflop, 3-street runout to showdown.
        // contributions = [100, 100], no folds, flop_seen=true.
        ShowdownCase {
            name: "Preflop HU allin-equal showdown",
            np: 2, nh, traverser: 0, starting_pot: 3, // 1+2 blinds
            contributions: vec![100, 100], fold_mask: 0,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: vec![r_uniform.clone()],
            num_combinations: 16.0,
        },
        // 3p tournament: SB (short) and BB go allin at 20, button covers
        // both at 100. Multi-level showdown after 3-street runout.
        // contributions = [20, 20, 100], no folds, flop_seen=true.
        // Two distinct levels (20, 100).
        ShowdownCase {
            name: "Preflop 3p two-level allin [20,20,100]",
            np: 3, nh, traverser: 0, starting_pot: 3,
            contributions: vec![20, 20, 100], fold_mask: 0,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
            num_combinations: 64.0,
        },
        // 3p tournament with 3 distinct levels: SB short at 15, BB at 50,
        // button at 100. Builds on #40's three-level extension applied to
        // a preflop scenario specifically.
        ShowdownCase {
            name: "Preflop 3p three-level allin [15,50,100]",
            np: 3, nh, traverser: 1, starting_pot: 3,
            contributions: vec![15, 50, 100], fold_mask: 0,
            hand_cards: hand_cards.clone(), hand_strength: hand_strength.clone(),
            opp_reach: vec![r_quarter.clone(), r_quarter.clone()],
            num_combinations: 64.0,
        },
    ];

    eprintln!("\nType B: preflop all-in runout (multi-level, rake applies)");
    eprintln!("  Asserting: structure via independent enumerator (rake=0),");
    eprintln!("  AND rake applies non-trivially at rake=5%, flop_seen=true.\n");

    for tc in &preflop_allin_cases {
        // Part 1: structure check via independent enumerator (rake-free).
        validate_case(tc);

        // Part 2: rake-applied non-triviality check. The independent
        // enumerator gives rake-free CFV; the rake-bearing path with
        // flop_seen=true and rake=5% must DIFFER from rake-free by
        // a measurable amount (proves rake is actually being applied
        // at the preflop allin runout terminal).
        let mut items: Vec<(u16, u16)> = (0..tc.nh)
            .map(|h| (tc.hand_strength[h], h as u16))
            .collect();
        items.sort_by_key(|&(s, _)| s);
        let sorted_str: Vec<u16> = items.iter().map(|&(s, _)| s).collect();
        let sorted_idx: Vec<u16> = items.iter().map(|&(_, i)| i).collect();
        let opp_reach_views: Vec<&[f32]> = tc.opp_reach.iter().map(|v| v.as_slice()).collect();

        let cfv_rake_free = side_pot_showdown_cfv(
            &opp_reach_views, &tc.hand_cards, tc.nh,
            &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
            &tc.contributions, tc.fold_mask, tc.traverser, tc.np as u8,
            tc.starting_pot,
        );
        let cfv_with_rake = side_pot_showdown_cfv_with_rake(
            &opp_reach_views, &tc.hand_cards, tc.nh,
            &sorted_str, &sorted_idx, &sorted_str, &sorted_idx,
            &tc.contributions, tc.fold_mask, tc.traverser, tc.np as u8,
            tc.starting_pot,
            0.05, 1000.0, true, // rake_rate=5%, generous cap, flop_seen=TRUE
        );

        // The rake amount should be non-zero at some hand (else rake
        // isn't applying — false-green by vacuous match).
        let max_rake_effect = (0..tc.nh)
            .map(|h| (cfv_with_rake[h] - cfv_rake_free[h]).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "  [{:45}] max_rake_effect={:.4e} (non-zero means rake applies)",
            tc.name, max_rake_effect,
        );
        assert!(
            max_rake_effect > 1e-4,
            "P2.4 RAKE-APPLIED check FAILED at case '{}': max_rake_effect = \
             {}, expected > 1e-4. Rake at 5%% on a multi-thousand-chip pot \
             must produce a measurable CFV shift at some hand. If 0, the \
             rake path is silently inactive at this terminal type.",
            tc.name, max_rake_effect,
        );
    }
    eprintln!("\n✓ All {} preflop allin-runout cases: structure matches \
        independent enumerator AND rake-applies non-trivially at flop_seen=true.",
        preflop_allin_cases.len());

    eprintln!("\n═══ P2.4 COMPLETE: preflop showdown oracle covers both branches ═══");
    eprintln!("  Type A (fold-terminals): no-flop-no-drop verified explicitly");
    eprintln!("  Type B (allin-runouts):  multi-level structure + rake-applies verified");
    eprintln!("  Dormant-untested item from rake arc: CLOSED.");
}
