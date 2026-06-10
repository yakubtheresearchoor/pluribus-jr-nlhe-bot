//! B4 Design-2 gate, part 2 — THE VERDICT: equilibrium-quality A/B.
//!
//! CFV parity (part 1) is the proxy; the number that decides whether
//! Design 2 is acceptable is equilibrium quality. A CFV bias that is
//! systematic and near-symmetric across strategies can be large in
//! absolute terms while barely moving the equilibrium (part 1 measured
//! exactly that structure: ~16× uniform mass scale, 0.2-0.5% shape),
//! and a small bias in the wrong place can compound through the regret
//! loop. So: SAME config, solved twice — once per terminal design —
//! both lifted to hand granularity, both scored in the EXACT game
//! through the existing quality-gate machinery. The difference in
//! lifted exploitability IS Design 2's equilibrium cost, in the unit
//! that matters.
//!
//! Config: Phase 4 wet-deep, B=4 quantile maps, 30 iters — the exact
//! setting where the B3 quality gate measured Design 1's lifted
//! exploitability at 2.0864% pot, giving a banked cross-check that
//! this harness reproduces before trusting its Design-2 arm.
//!
//! ═══ MEASURED 2026-06-10 — FORK OPEN ═══
//!   Design 1 lifted:               2.0864% pot (reproduces B3 banked)
//!   Design 2 RAW factored:         6.9802% pot → cost +4.89% — REJECTED
//!   Design 2 + pairwise renorm:   15.8392% pot → cost +13.75% — WORSE
//! The proxy and the verdict DIVERGED twice: renormalization collapsed
//! terminal-level CFV deviation (1490% → 0.2-0.5% on the parity
//! fixture) and tripled the equilibrium damage. The regret loop is
//! sensitive to a component of the error the parity fixture does not
//! exercise.
//!
//! Regime caveat (probed by the companion tests below): this config is
//! the EXTREME card-removal regime — 5 opponents drawing from a 6-hand
//! universe is musical chairs; the joint "all hands distinct"
//! constraint is the dominant physics, maximally adversarial to any
//! factored approximation, and structurally absent at production
//! (5 opponents over ~1176 hands; pairwise compat ≈ 0.919). The
//! identity-B probe isolates factorization-alone cost at zero
//! coarseness; the NH=16 probe measures how the cost moves as the
//! regime relaxes toward production. NOTE the instrument wall: this
//! verdict measurement cannot run at production nh — the exact scorer
//! is O(nh^(K+1)) — so the regime TREND plus a terminal-level deviation
//! probe at production nh is the best available evidence.

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::bucketed_flop_cfr::{
    lift_cum_to_exact, BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const NP: u8 = 6;
const NH: usize = 6;
const NB: usize = 4;
const STACKS: i32 = 500;
const STARTING_POT: i32 = 30;
const STARTING_CONTRIB: i32 = 5;
const ITERS: u32 = 30;

// ── Wet-deep fixture (same as the B3 quality gate) ──

fn build_table() -> FlopChanceTable {
    let board: Vec<Card> = ["Th", "9d", "8c"].iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let mut all_valid: Vec<u16> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
            continue;
        }
        all_valid.push(idx as u16);
    }
    let step = all_valid.len() / NH;
    let chosen: Vec<u16> = (0..NH).map(|i| all_valid[i * step]).collect();
    let mut ranges: Vec<Vec<f32>> = (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for p in 0..NP as usize {
        for &hi in &chosen {
            ranges[p][hi as usize] = 1.0;
        }
    }
    let turn_cards: Vec<u8> =
        ["2c", "Jd"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    let river_strs_per_turn: [&[&str]; 2] = [&["4s", "7h"], &["3s", "Qc"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] =
            river_strs_per_turn[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

fn build_ab_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: STARTING_POT,
        starting_stacks: vec![STACKS; NP as usize],
        initial_contributions: vec![STARTING_CONTRIB; NP as usize],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.33), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    build_tree(&config).unwrap()
}

fn quantile_maps(
    table: &FlopChanceTable,
    nb: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8]| -> Vec<u16> {
        let alive: Vec<usize> = pl_idx[..nh]
            .iter()
            .map(|&i| i as usize)
            .filter(|&h| !conflicts(h, dead))
            .collect();
        let n = alive.len();
        assert!(n >= nb);
        let mut map = vec![NO_BUCKET; nh];
        for (pos, &h) in alive.iter().enumerate() {
            map[h] = ((pos * nb) / n) as u16;
        }
        map
    };
    let (_, _, _, base_pi, _) = table.sorted_opp_arrays_base();
    let flop_map = map_for(&base_pi, &[]);
    let mut turn_maps = Vec::new();
    let mut river_maps = Vec::new();
    for &tc_card in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
        turn_maps.push(map_for(pi, &[tc_card]));
        let mut rms = Vec::new();
        for &rc_card in &table.river_decks[tc_card as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
            rms.push(map_for(pi, &[tc_card, rc_card]));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
}

fn expl_pct(cpu: &FlopStartVectorCfr, tree: &FlatTree, game: &FlopStartGame) -> f32 {
    let np = tree.num_players as usize;
    let mut total = 0.0f32;
    for p in 0..np {
        let br = cpu.best_response_value_debug(tree, game, p as u8);
        let sv = cpu.strategy_value_debug(tree, game, p as u8);
        for h in 0..br.len().min(sv.len()) {
            total += (br[h] - sv[h]).max(0.0);
        }
    }
    total / STARTING_POT as f32 * 100.0
}

fn solve_lift_score(tree: &FlatTree, design: TerminalDesign) -> f32 {
    let game = FlopStartGame::new(build_table());
    let (fm, tm, rm) = quantile_maps(game.table(), NB);
    let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
    let mut bucketed = BucketedFlopCfr::new(tree, game.table(), &bk);
    bucketed.set_terminal_design(design);
    bucketed.run(tree, &game, &bk, ITERS);

    let game_score = FlopStartGame::new(build_table());
    let mut scorer = FlopStartVectorCfr::new(tree, game_score.table());
    lift_cum_to_exact(tree, &bucketed, &bk, &mut scorer);
    expl_pct(&scorer, tree, &game_score)
}

#[test]
#[ignore = "FORK OPEN: Design 2 fails this gate (raw +4.89%, renormalized \
            +13.75% pot vs the 0.25% acceptance line). This is the gate a \
            future Design-2 variant must turn green; numbers in header."]
fn equilibrium_ab_design2_vs_design1() {
    let tree = build_ab_tree();

    let d1 = solve_lift_score(&tree, TerminalDesign::Design1Brute);
    eprintln!("Design 1 (brute) lifted exploitability:    {d1:.4}% pot");
    // Cross-check against the B3 quality gate's banked number for this
    // exact config (2.0864% pot) — the harness must reproduce it
    // before its Design-2 arm is trusted.
    assert!(
        (d1 - 2.0864).abs() < 0.01,
        "Design-1 arm does not reproduce the B3 banked number \
         (got {d1:.4}%, banked 2.0864%) — harness drift, fix before \
         reading the A/B"
    );

    let d2 = solve_lift_score(&tree, TerminalDesign::Design2Factored);
    eprintln!("Design 2 (factored) lifted exploitability: {d2:.4}% pot");

    let cost = d2 - d1;
    eprintln!("equilibrium cost of factorization: {cost:+.4}% pot");
    eprintln!("(bucketing coarseness itself at B={NB}: {d1:.2}% — the factorization \
               cost is judged against that scale and the sweep's resolution)");

    // The verdict bound: the factorization's equilibrium cost must be
    // small against the bucketing coarseness it rides on. 0.25% pot
    // (≈ 12% of the B=4 coarseness cost) is the acceptance line set
    // BEFORE the Design-2 measurement was read.
    assert!(
        cost.abs() < 0.25,
        "Design 2 equilibrium cost {cost:+.4}% pot exceeds the acceptance \
         line (0.25% pot) — factorization error is material at the \
         equilibrium level; design fork re-opens"
    );
}

/// Probe (a): factorization-alone equilibrium cost at ZERO bucketing
/// coarseness — Design 2 at B = nh through identity maps. Design 1 at
/// identity is bit-exact to the exact solver (B3 gate), whose lifted
/// exploitability at this config is ~0.0001% pot, so Design 2's
/// absolute lifted number here IS the pure factorization cost.
///
/// MEASURED 2026-06-10: Design 2 @ identity lifted = 14.8696% pot.
/// Factorization alone is equilibrium-fatal at THIS card-removal
/// regime (5 opponents over 6 hands) even with PERFECT maps — the
/// damage is not a coarseness interaction; it is the factorization.
#[test]
#[ignore = "fork-diagnostic probe (~1 min); run with --ignored --nocapture"]
fn probe_factorization_cost_at_identity() {
    let tree = build_ab_tree();
    let game = FlopStartGame::new(build_table());
    let bk = FlopBucketing::identity(game.table());
    let mut bucketed = BucketedFlopCfr::new(&tree, game.table(), &bk);
    bucketed.set_terminal_design(TerminalDesign::Design2Factored);
    bucketed.run(&tree, &game, &bk, ITERS);

    let game_score = FlopStartGame::new(build_table());
    let mut scorer = FlopStartVectorCfr::new(&tree, game_score.table());
    lift_cum_to_exact(&tree, &bucketed, &bk, &mut scorer);
    let d2 = expl_pct(&scorer, &tree, &game_score);
    eprintln!("probe (a): Design 2 @ B=nh identity, lifted exploitability = {d2:.4}% pot");
    eprintln!("           (Design 1 @ identity ≡ exact ≈ 0.0001% — so this IS the");
    eprintln!("            pure factorization equilibrium cost at this regime)");
}

/// Probe (b): the same A/B in a MILDER card-removal regime — NH=16
/// hands (5 opponents over 16: pairwise compat ≈ 0.76 vs 6-hand
/// musical chairs), lean 1-bet tree so Design 1 at B=6 and the exact
/// O(nh^6) scorer both stay affordable. Measures how the factorization
/// cost moves as the regime relaxes toward production (compat 0.919).
///
/// MEASURED 2026-06-10 (B=6, 15 iters, 1569-node tree):
///   Design 1 lifted: 9.7496% | Design 2 lifted: 11.9819%
///   factorization cost: +2.2323% pot (vs +13.75% at NH=6, B=4)
/// The cost falls ~6× as the hand universe grows 6 → 16 — trend
/// direction favorable — but remains ~9× over the 0.25% acceptance
/// line, and NO production-nh verdict exists (instrument wall: exact
/// scorer is O(nh^6)). Extrapolation is explicitly NOT banked.
/// Design 2 stands REJECTED at every measurable regime.
#[test]
#[ignore = "fork-diagnostic probe (~10-20 min, exact BR at nh=16); run with --ignored --nocapture"]
fn probe_regime_relaxation_nh16() {
    const NH16: usize = 16;
    const NB16: usize = 6;
    const ITERS16: u32 = 15;

    let build_table16 = || -> FlopChanceTable {
        let board: Vec<Card> =
            ["Th", "9d", "8c"].iter().map(|s| card_from_str(s).unwrap()).collect();
        let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
        let mut all_valid: Vec<u16> = Vec::new();
        for idx in 0..NUM_POSSIBLE_HANDS {
            let (c1, c2) = index_to_card_pair(idx);
            if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 {
                continue;
            }
            all_valid.push(idx as u16);
        }
        let step = all_valid.len() / NH16;
        let chosen: Vec<u16> = (0..NH16).map(|i| all_valid[i * step]).collect();
        let mut ranges: Vec<Vec<f32>> =
            (0..NP).map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
        for p in 0..NP as usize {
            for &hi in &chosen {
                ranges[p][hi as usize] = 1.0;
            }
        }
        let turn_cards: Vec<u8> =
            ["2c", "Jd"].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
        let river_strs_per_turn: [&[&str]; 2] = [&["4s", "7h"], &["3s", "Qc"]];
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
        for (ti, &tc) in turn_cards.iter().enumerate() {
            river_decks[tc as usize] =
                river_strs_per_turn[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
        }
        FlopChanceTable::compute_flop_start_subset_with_decks(
            &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
        )
    };
    // Lean 1-bet tree keeps both the Design-1 reference solve and the
    // exact O(nh^6) scorer affordable at NH=16.
    let tree = {
        let config = TreeConfig {
            num_players: NP,
            initial_state: BoardState::Flop,
            starting_pot: STARTING_POT,
            starting_stacks: vec![STACKS; NP as usize],
            initial_contributions: vec![STARTING_CONTRIB; NP as usize],
            rake_rate: 0.0,
            rake_cap: 0.0,
            bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
            add_allin_threshold: 1.0,
            force_allin_threshold: 1.0,
            merging_threshold: 0.0,
            button_player: None,
        };
        build_tree(&config).unwrap()
    };
    eprintln!("probe (b) tree: {} nodes", tree.num_nodes());

    let run_one = |design: TerminalDesign| -> f32 {
        let game = FlopStartGame::new(build_table16());
        let (fm, tm, rm) = {
            // quantile maps at NH16 (same construction as the B=4 A/B)
            let table = game.table();
            let nh = table.num_valid;
            let conflicts = |h: usize, cards: &[u8]| -> bool {
                let c1 = table.hand_cards[h * 2];
                let c2 = table.hand_cards[h * 2 + 1];
                cards.iter().any(|&bc| bc == c1 || bc == c2)
            };
            let map_for = |pl_idx: &[u16], dead: &[u8]| -> Vec<u16> {
                let alive: Vec<usize> = pl_idx[..nh]
                    .iter()
                    .map(|&i| i as usize)
                    .filter(|&h| !conflicts(h, dead))
                    .collect();
                let n = alive.len();
                assert!(n >= NB16);
                let mut map = vec![NO_BUCKET; nh];
                for (pos, &h) in alive.iter().enumerate() {
                    map[h] = ((pos * NB16) / n) as u16;
                }
                map
            };
            let (_, _, _, base_pi, _) = table.sorted_opp_arrays_base();
            let flop_map = map_for(&base_pi, &[]);
            let mut turn_maps = Vec::new();
            let mut river_maps = Vec::new();
            for &tc_card in &table.remaining_deck {
                let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
                turn_maps.push(map_for(pi, &[tc_card]));
                let mut rms = Vec::new();
                for &rc_card in &table.river_decks[tc_card as usize] {
                    let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
                    rms.push(map_for(pi, &[tc_card, rc_card]));
                }
                river_maps.push(rms);
            }
            (flop_map, turn_maps, river_maps)
        };
        let bk = FlopBucketing::from_maps(game.table(), NB16, NB16, NB16, fm, tm, rm);
        let mut bucketed = BucketedFlopCfr::new(&tree, game.table(), &bk);
        bucketed.set_terminal_design(design);
        bucketed.run(&tree, &game, &bk, ITERS16);

        let game_score = FlopStartGame::new(build_table16());
        let mut scorer = FlopStartVectorCfr::new(&tree, game_score.table());
        lift_cum_to_exact(&tree, &bucketed, &bk, &mut scorer);
        expl_pct(&scorer, &tree, &game_score)
    };

    let d1 = run_one(TerminalDesign::Design1Brute);
    eprintln!("probe (b) NH=16 B={NB16}: Design 1 lifted = {d1:.4}% pot");
    let d2 = run_one(TerminalDesign::Design2Factored);
    eprintln!("probe (b) NH=16 B={NB16}: Design 2 lifted = {d2:.4}% pot");
    eprintln!("probe (b) factorization cost: {:+.4}% pot (vs +13.75% at NH=6/B=4)", d2 - d1);
}
