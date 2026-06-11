//! G3 remainder for the GPU bucketed terminal: the f64-reference
//! same-quantity gate and trajectory parity at DIVERGENT per-street
//! dims (per directive: uniform-B trajectories cannot exercise the
//! stride divergence the layout was built for).
//!
//! Gate 3 (f64 reference): one flop-zone walk's terminals evaluated
//! three ways — GPU striped, GPU unstriped, CPU f64 (the collapse-gate
//! reference arithmetic, all accumulation in f64) — at general B with
//! mixed fold/side-pot terminals. BOTH f32 arms must sit at
//! accumulated-f32-rounding distance from the one f64 quantity (the
//! same-quantity proof, the standard ratified at the collapse gate).
//!
//! Gate 4 (trajectory, divergent dims): nb_flop=4, nb_turn=3,
//! nb_river=5 through 8 full DCFR iterations, GPU striped vs pure CPU
//! — per-buffer drift pinned at accumulated-rounding scale. This is
//! the M1-pattern iteration gate run where the strides actually
//! diverge per street.
//!
//! ═══ MEASURED 2026-06-11 ═══
//!   Gate 3 (B=5, S=6, flop-zone walk, 90 terminals): max rel
//!     |GPU_striped − f64| 4.9e-13, |GPU_unstriped − f64| 2.3e-12,
//!     |CPU − f64| 2.3e-12 — all three f32 arms essentially ON the
//!     f64 quantity (this fixture's small dyadic sums round exactly;
//!     nonzero, so the paths are exercised). Same-quantity ✓.
//!   Gate 4 (4/3/5 divergent dims, 8 iters, GPU striped vs pure CPU):
//!     per-buffer max rel drift 2.1e-6 .. 7.7e-5, root 8.9e-7 —
//!     accumulated rounding through 8 CFR iterations (cum_flop is the
//!     deepest accumulator, hence largest); bug line 1e-3.
//!     Divergent-dims trajectory ✓ — the host handles per-street B
//!     natively (per-runout table offsets, per-zone nb/stripes).

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::bucketed_terminal::BucketedTerminalGpu;
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::bucketed_showdown::BucketedRunoutTables;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::Zone;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

const NP: u8 = 6;
const NH: usize = 16;

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
    let river_strs: [&[&str]; 2] = [&["4s", "7h"], &["3s", "Qc"]];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for (ti, &tc) in turn_cards.iter().enumerate() {
        river_decks[tc as usize] =
            river_strs[ti].iter().map(|s| card_from_str(s).unwrap() as u8).collect();
    }
    FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, NP, &chosen, &turn_cards, &river_decks,
    )
}

/// Rich-enough tree to carry fold + side-pot terminals in the flop zone.
fn build_g3_tree() -> FlatTree {
    let config = TreeConfig {
        num_players: NP,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![200; NP as usize],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.05,
        rake_cap: 3.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    build_tree(&config).unwrap()
}

fn quantile_maps(
    table: &FlopChanceTable,
    nb_f: usize,
    nb_t: usize,
    nb_r: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8], nb: usize| -> Vec<u16> {
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
    let flop_map = map_for(&base_pi, &[], nb_f);
    let mut turn_maps = Vec::new();
    let mut river_maps = Vec::new();
    for &tc_card in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
        turn_maps.push(map_for(pi, &[tc_card], nb_t));
        let mut rms = Vec::new();
        for &rc_card in &table.river_decks[tc_card as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
            rms.push(map_for(pi, &[tc_card, rc_card], nb_r));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
}

// ── f64 reference: per-terminal, both arms, all accumulation in f64.
//    The collapse-gate reference arithmetic (self-contained per the
//    existing gate-test convention), generalized to arm 1. ──

#[allow(clippy::too_many_arguments)]
fn terminal_reference_f64(
    bucket_reach: &[Vec<f64>],
    t: &BucketedRunoutTables,
    contributions: &[i32],
    fold_mask: u16,
    traverser: usize,
    starting_pot: i32,
    rake_rate: f64,
    rake_cap: f64,
) -> Vec<f64> {
    let nb = t.nb;
    let np = NP as usize;
    let num_opp = np - 1;
    let c_t = contributions[traverser];

    let mut all_active_equal = true;
    {
        let mut refc: Option<i32> = None;
        for p in 0..np {
            if fold_mask & (1u16 << p) != 0 {
                continue;
            }
            match refc {
                None => refc = Some(contributions[p]),
                Some(r) => {
                    if contributions[p] != r {
                        all_active_equal = false;
                        break;
                    }
                }
            }
        }
    }
    let arm1 = all_active_equal && fold_mask == 0;

    let opp_player: Vec<usize> =
        (0..num_opp).map(|oi| if oi < traverser { oi } else { oi + 1 }).collect();
    let opp_folded: Vec<bool> =
        opp_player.iter().map(|&p| fold_mask & (1u16 << p) != 0).collect();
    let opp_contrib: Vec<i32> = opp_player.iter().map(|&p| contributions[p]).collect();

    let fw = |bt: usize, b: usize| t.f_w[bt * nb + b] as f64;
    let ft_ = |bt: usize, b: usize| t.f_t[bt * nb + b] as f64;
    let fl = |bt: usize, b: usize| t.f_l[bt * nb + b] as f64;
    let fn_ = |a: usize, b: usize| t.f_n[a * nb + b] as f64;

    let mut out = vec![0.0f64; nb];

    if arm1 {
        let k = num_opp as f64;
        let half_pot = starting_pot as f64 / np as f64 + c_t as f64;
        let total_pot: i32 = starting_pot + contributions.iter().sum::<i32>();
        let rake = (total_pot as f64 * rake_rate).min(rake_cap).max(0.0);
        let rpus = if half_pot > 0.0 { rake / half_pot } else { 0.0 };
        #[allow(clippy::too_many_arguments)]
        fn rec1(
            d: usize,
            num_opp: usize,
            nb: usize,
            bt: usize,
            prefix: &mut Vec<usize>,
            state: &[f64],
            reach: &[Vec<f64>],
            t: &BucketedRunoutTables,
            k: f64,
            rpus: f64,
            acc: &mut f64,
        ) {
            if d == num_opp {
                if state[0] != 0.0 {
                    *acc += state[0] * -1.0;
                }
                for j in 0..=num_opp {
                    let s = state[1 + j];
                    if s == 0.0 {
                        continue;
                    }
                    let nu = if j == 0 {
                        k - rpus
                    } else {
                        let tf = (j + 1) as f64;
                        (k + 1.0 - tf) / tf - rpus / tf
                    };
                    *acc += s * nu;
                }
                return;
            }
            for b in 0..nb {
                let r = reach[d][b];
                if r == 0.0 {
                    continue;
                }
                let mut m = r;
                let mut blocked = false;
                for &pb in prefix.iter() {
                    let f = t.f_n[pb * nb + b] as f64;
                    if f == 0.0 {
                        blocked = true;
                        break;
                    }
                    m *= f;
                }
                if blocked {
                    continue;
                }
                let i = bt * nb + b;
                let (w, ti_, l, n) =
                    (t.f_w[i] as f64, t.f_t[i] as f64, t.f_l[i] as f64, t.f_n[i] as f64);
                if n == 0.0 {
                    continue;
                }
                let mut ns = vec![0.0f64; state.len()];
                if state[0] != 0.0 {
                    ns[0] += state[0] * (m * n);
                }
                for j in 0..=d {
                    let s = state[1 + j];
                    if s == 0.0 {
                        continue;
                    }
                    if l != 0.0 {
                        ns[0] += s * (m * l);
                    }
                    if ti_ != 0.0 {
                        ns[1 + j + 1] += s * (m * ti_);
                    }
                    if w != 0.0 {
                        ns[1 + j] += s * (m * w);
                    }
                }
                prefix.push(b);
                rec1(d + 1, num_opp, nb, bt, prefix, &ns, reach, t, k, rpus, acc);
                prefix.pop();
            }
        }
        for (bt, slot) in out.iter_mut().enumerate() {
            let mut state = vec![0.0f64; num_opp + 2];
            state[1] = 1.0;
            let mut acc = 0.0f64;
            rec1(0, num_opp, nb, bt, &mut Vec::new(), &state, bucket_reach, t, k, rpus, &mut acc);
            *slot = half_pot * acc;
        }
        let _ = (fw, ft_, fl, fn_);
        return out;
    }

    // Arm 2: tuple × relation enumeration, f64 (collapse-gate reference).
    let mut levels: Vec<i32> = (0..np).map(|p| contributions[p]).collect();
    levels.sort();
    levels.dedup();
    let main_pot_amount: i32 = {
        let nmc = (0..np).filter(|&p| contributions[p] >= levels[0]).count();
        levels[0] * nmc as i32 + starting_pot
    };
    let main_pot_rake = (main_pot_amount as f64 * rake_rate).min(rake_cap).max(0.0);
    let traverser_stake = starting_pot as f64 / np as f64 + c_t as f64;
    let traverser_folded = fold_mask & (1u16 << traverser) != 0;

    let net = |rel: &[u8]| -> f64 {
        let mut cash = 0.0f64;
        let mut prev_l = 0i32;
        for (li, &lev) in levels.iter().enumerate() {
            let pc = lev - prev_l;
            let nc = (0..np).filter(|&p| contributions[p] >= lev).count();
            let mut pot_l = (pc * nc as i32) as f64;
            if li == 0 {
                pot_l += starting_pot as f64;
            }
            if pot_l == 0.0 {
                prev_l = lev;
                continue;
            }
            let trav_elig = !traverser_folded && c_t >= lev;
            let mut elig: u32 = trav_elig as u32;
            let mut beats = false;
            for oi in 0..num_opp {
                if opp_folded[oi] || opp_contrib[oi] < lev {
                    continue;
                }
                elig += 1;
                if rel[oi] == 2 {
                    beats = true;
                }
            }
            if elig == 0 {
                if contributions[traverser] >= lev {
                    cash +=
                        pc as f64 + if li == 0 { starting_pot as f64 / np as f64 } else { 0.0 };
                }
                prev_l = lev;
                continue;
            }
            if !trav_elig {
                prev_l = lev;
                continue;
            }
            if !beats {
                let mut tied = 1u32;
                for oi in 0..num_opp {
                    if opp_folded[oi] || opp_contrib[oi] < lev {
                        continue;
                    }
                    if rel[oi] == 1 {
                        tied += 1;
                    }
                }
                let par = if li == 0 { pot_l - main_pot_rake } else { pot_l };
                cash += par / tied as f64;
            }
            prev_l = lev;
        }
        cash - traverser_stake
    };

    #[allow(clippy::too_many_arguments)]
    fn rec2(
        d: usize,
        num_opp: usize,
        nb: usize,
        bt: usize,
        w: f64,
        prefix: &mut Vec<usize>,
        rel: &mut [u8],
        reach: &[Vec<f64>],
        t: &BucketedRunoutTables,
        opp_folded: &[bool],
        net: &dyn Fn(&[u8]) -> f64,
        acc: &mut f64,
    ) {
        if d == num_opp {
            *acc += w * net(rel);
            return;
        }
        for b in 0..nb {
            let r = reach[d][b];
            if r == 0.0 {
                continue;
            }
            let mut base = w * r;
            let mut blocked = false;
            for &pb in prefix.iter() {
                let f = t.f_n[pb * nb + b] as f64;
                if f == 0.0 {
                    blocked = true;
                    break;
                }
                base *= f;
            }
            if blocked {
                continue;
            }
            let i = bt * nb + b;
            prefix.push(b);
            if opp_folded[d] {
                let n = t.f_n[i] as f64;
                if n != 0.0 {
                    rel[d] = 3;
                    rec2(d + 1, num_opp, nb, bt, base * n, prefix, rel, reach, t, opp_folded,
                         net, acc);
                }
            } else {
                for (code, f) in
                    [(0u8, t.f_w[i] as f64), (1, t.f_t[i] as f64), (2, t.f_l[i] as f64)]
                {
                    if f == 0.0 {
                        continue;
                    }
                    rel[d] = code;
                    rec2(d + 1, num_opp, nb, bt, base * f, prefix, rel, reach, t, opp_folded,
                         net, acc);
                }
            }
            prefix.pop();
        }
    }
    let mut rel = vec![0u8; num_opp];
    for (bt, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        rec2(0, num_opp, nb, bt, 1.0, &mut Vec::new(), &mut rel, bucket_reach, t, &opp_folded,
             &net, &mut acc);
        *slot = acc;
    }
    out
}

/// Gate 3: one flop-zone walk's terminals, three f32 arms vs one f64
/// quantity.
#[test]
fn gate3_f64_reference_same_quantity() {
    const NB: usize = 5;
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_g3_tree();
    let game = FlopStartGame::new(build_table());
    let (fm, tm, rm) = quantile_maps(game.table(), NB, NB, NB);
    let bk = FlopBucketing::from_maps(game.table(), NB, NB, NB, fm, tm, rm);
    let solver = BucketedFlopCfr::new(&tree, game.table(), &bk);

    // Deterministic non-uniform per-hand reach at every node (the
    // terminal consumes node-local reach; synthetic is fine — the gate
    // is about the terminal arithmetic, not the walk).
    let nn = tree.num_nodes();
    let np = NP as usize;
    let nh = game.table().num_valid;
    let mut reach = vec![0.0f32; nn * np * nh];
    for (i, r) in reach.iter_mut().enumerate() {
        let v = (i as u32).wrapping_mul(2654435761) % 13;
        *r = if v == 0 { 0.0 } else { v as f32 / 16.0 };
    }

    // GPU arms.
    let mut cfv_striped = vec![0.0f32; nn * nh];
    let mut cfv_unstriped = vec![0.0f32; nn * nh];
    for (stripes, cfv) in
        [((32 / NB) as u32, &mut cfv_striped), (1u32, &mut cfv_unstriped)]
    {
        let mut gpu =
            BucketedTerminalGpu::new(&ctx, &tree, game.table(), &bk, &solver, stripes)
                .expect("gpu");
        assert!(gpu.fill_terminals(Zone::Flop, None, None, 0, &reach, cfv));
    }

    // CPU f32 arm + f64 reference, per terminal.
    let mut n_term = 0usize;
    let mut max_s = 0.0f64;
    let mut max_u = 0.0f64;
    let mut max_c = 0.0f64;
    for idx in 0..nn {
        // Flop zone: nodes not below any turn/river chance — reuse the
        // layout's zone classification.
        let layout = solver.gpu_layout(&bk);
        if layout.zone_of(idx) != solver_core::solver::flop_start_vector_cfr::Zone::Flop
            || !tree.nodes[idx].is_terminal()
        {
            continue;
        }
        n_term += 1;
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
        let fold_mask = tree.get_folded_mask(idx);

        // Bucket reach (f32 CPU order → also f64 copy).
        let mut br32: Vec<Vec<f32>> = vec![vec![0.0; NB]; np - 1];
        let mut br64: Vec<Vec<f64>> = vec![vec![0.0; NB]; np - 1];
        for oi in 0..np - 1 {
            let p = oi + 1; // traverser = 0, so opp slot oi = player oi + 1
            let base = (idx * np + p) * nh;
            for h in 0..nh {
                let b = bk.flop_map[h];
                if b == NO_BUCKET {
                    continue;
                }
                br32[oi][b as usize] += reach[base + h];
                br64[oi][b as usize] += reach[base + h] as f64;
            }
        }
        let views: Vec<&[f32]> = br32.iter().map(|v| v.as_slice()).collect();
        let cpu = solver_core::solver::bucketed_showdown::bucketed_showdown_cfv_design1_collapsed(
            &views, &bk.flop_tables, &contribs, fold_mask, 0, NP,
            tree.starting_pot, tree.rake_rate as f32, tree.rake_cap as f32, true,
        );
        let reference = terminal_reference_f64(
            &br64, &bk.flop_tables, &contribs, fold_mask, 0,
            tree.starting_pot, tree.rake_rate, tree.rake_cap,
        );
        let scale = reference.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-30);
        // GPU outputs are expanded per-hand; compare via map (cfv[h] =
        // bucket value, nc = full-table num_combinations).
        let nc = game.table().num_combinations;
        for h in 0..nh {
            let b = bk.flop_map[h];
            if b == NO_BUCKET {
                continue;
            }
            let r = reference[b as usize] / nc;
            let ds = (cfv_striped[idx * nh + h] as f64 - r).abs() / scale;
            let du = (cfv_unstriped[idx * nh + h] as f64 - r).abs() / scale;
            let dc = (cpu[b as usize] as f64 / nc - r).abs() / scale;
            if ds > max_s {
                max_s = ds;
            }
            if du > max_u {
                max_u = du;
            }
            if dc > max_c {
                max_c = dc;
            }
        }
    }
    eprintln!(
        "gate 3 ({n_term} flop terminals, B={NB}): max rel |striped−f64| {max_s:.2e}, \
         |unstriped−f64| {max_u:.2e}, |CPU−f64| {max_c:.2e}"
    );
    for (arm, d) in [("striped", max_s), ("unstriped", max_u), ("cpu", max_c)] {
        assert!(
            d < 1e-4,
            "{arm} arm {d:.2e} beyond accumulated f32 rounding from the f64 \
             reference — same-quantity violation"
        );
    }
}

/// Gate 4: trajectory parity at DIVERGENT per-street dims (4/3/5),
/// 8 full DCFR iterations, GPU striped vs pure CPU.
#[test]
fn gate4_trajectory_divergent_dims() {
    let ctx = MetalContext::new().expect("Metal");
    let tree = build_g3_tree();
    const ITERS: u32 = 8;
    const S: u32 = 6; // 32 / max(nb) = 32/5

    let run = |use_gpu: bool| -> (BucketedFlopCfr, Vec<f32>) {
        let game = FlopStartGame::new(build_table());
        let (fm, tm, rm) = quantile_maps(game.table(), 4, 3, 5);
        let bk = FlopBucketing::from_maps(game.table(), 4, 3, 5, fm, tm, rm);
        let mut solver = BucketedFlopCfr::new(&tree, game.table(), &bk);
        solver.set_terminal_design(TerminalDesign::Design1Collapsed);
        if use_gpu {
            // Host handles divergent per-street B natively (per-runout
            // table offsets, per-zone nb/stripes in params).
            let gpu =
                BucketedTerminalGpu::new(&ctx, &tree, game.table(), &bk, &solver, S)
                    .expect("gpu");
            solver.set_terminal_offload_hook(Some(gpu.into_hook()));
        }
        let root = solver.run(&tree, &game, &bk, ITERS);
        (solver, root)
    };
    let (gpu_arm, root_gpu) = run(true);
    let (cpu_arm, root_cpu) = run(false);

    let scale = |xs: &[f32]| xs.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;
    let mut max_drift = 0.0f64;
    for (label, ga, ca) in [
        ("regrets_flop", gpu_arm.regrets_flop(), cpu_arm.regrets_flop()),
        ("cum_flop", gpu_arm.cum_strategy_flop(), cpu_arm.cum_strategy_flop()),
        ("regrets_turn", gpu_arm.regrets_turn(), cpu_arm.regrets_turn()),
        ("cum_turn", gpu_arm.cum_strategy_turn(), cpu_arm.cum_strategy_turn()),
        ("regrets_river", gpu_arm.regrets_river(), cpu_arm.regrets_river()),
        ("cum_river", gpu_arm.cum_strategy_river(), cpu_arm.cum_strategy_river()),
    ] {
        let s = scale(ca).max(1e-30);
        let d = ga
            .iter()
            .zip(ca.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).abs() / s)
            .fold(0.0, f64::max);
        eprintln!("gate 4 {label}: max rel drift {d:.2e}");
        max_drift = max_drift.max(d);
    }
    let root_d = root_gpu
        .iter()
        .zip(&root_cpu)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0, f64::max)
        / scale(&root_cpu).max(1e-30);
    eprintln!("gate 4 root: {root_d:.2e}");
    assert!(
        max_drift.max(root_d) < 1e-3,
        "divergent-dims trajectory drift {max_drift:.2e} beyond rounding — breakage"
    );
}
