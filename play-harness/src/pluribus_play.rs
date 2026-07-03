//! Pluribus per-street real-time search PLAY loop.
//!
//! The bot re-solves the CURRENT street at each decision (rich sizing, depth-
//! limited to the bucketed blueprint continuation) instead of reading a banked
//! 1×pot strategy. Each street is played on its OWN subgame tree, stitched at
//! the street boundary: when a street completes, the survivors (+ dealt next
//! card) seed a fresh subgame for the next street. Re-indexing therefore happens
//! only at street transitions — no per-fold re-rooting.
//!
//! The blueprint is used ONLY for the bucket-keyed continuation VALUE
//! (`bp.game`/`bp.bk`), never tree-indexed — so the rich search sidesteps the
//! 1×pot blueprint's global turn/river offset alignment entirely.
//!
//! v1 approximations (documented):
//!   - reach prior = the street-entering range minus board conflicts; NO Bayes
//!     update for the prior street's betting (the Pluribus reach-prior refinement).
//!   - river search values showdown leaves via the bucketed river tables (the
//!     ACTUAL hand result at settle uses real cards — exact).

use crate::blueprint::Blueprint;
use crate::match_play::{pop_postflop_action, splitmix64};
use clean_rules::eval::best5;
use clean_rules::table::settle_pots;
use solver_core::solver::best_response::{exploitability, exploitability_dl, StrategyProfile};
use solver_core::solver::bucketed_search::{BucketedContinuationGame, ContStreet};
use solver_core::solver::game::GameSpec;
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::{
    build_tree_depth_limited, build_tree_depth_limited_2, build_tree_depth_limited_2_branched,
};
use solver_core::tree::flat::FlatTree;
use std::collections::HashMap;

/// Per-decision search depth/sharpness.
#[derive(Clone, Copy)]
pub struct SearchCfg {
    pub iters: u32,
    /// QRE λ for the BOT seat (sharp ≈ best-response).
    pub lambda: f32,
    /// QRE λ for the OPPONENT seats in the search. Lower ⇒ the bot models
    /// opponents as softer/looser and best-responds more EXPLOITATIVELY (value-
    /// bet thinner, call down more) — the exploit lever vs a loose pool. Equal to
    /// `lambda` ⇒ symmetric (GTO-ish, robust vs tough opponents).
    pub opp_lambda: f32,
    pub sample_m: u32,
    pub seed: u64,
    /// Run the parallel snapshot walk (rayon over traversers). `None` ⇒ defer to
    /// the `PAR` env var (back-compat); `Some(b)` ⇒ explicit.
    pub par: Option<bool>,
    /// Run Discounted-CFR (α=1.5,β=0,γ=2 — faster convergence, fewer iters).
    /// `None` ⇒ defer to the `DCFR` env var; `Some(b)` ⇒ explicit.
    pub dcfr: Option<bool>,
}

impl Default for SearchCfg {
    fn default() -> Self {
        SearchCfg {
            iters: 160,
            lambda: 300.0,
            opp_lambda: 300.0,
            sample_m: 200,
            seed: 0x5EA12C,
            par: None,
            dcfr: None,
        }
    }
}

impl SearchCfg {
    /// Latency schedule per live-player count, sized to the ~14s real-time
    /// budget (measured: PAR live-5 ≈4.9s, live-4 ≈3.4s, live-3 ≈0.3s; DCFR
    /// converges in fewer iters). Parallel + discounted for the heavy multiway
    /// counts; cheaper counts get more iters for a tighter strategy. `base`
    /// carries λ / sample_m / seed (the non-latency knobs) unchanged.
    pub fn for_live(live: usize, base: &SearchCfg) -> SearchCfg {
        let (iters, par, dcfr) = match live {
            0..=2 => (base.iters.max(240), false, false), // cheap exact showdown
            3 => (base.iters.max(200), true, false),
            4 => (base.iters, true, true),
            _ => (base.iters.min(140), true, true), // live-5+: trim iters to fit budget
        };
        SearchCfg { iters, par: Some(par), dcfr: Some(dcfr), ..*base }
    }
}

/// Search bet menu: 2 bet sizes (half / pot) + 1 raise (pot). Trimmed from
/// 3 bets + 2 raises — rich betting × multiway × lossless nh blows the tree up
/// exponentially in the branching factor, so fewer sizes is the lever that keeps
/// live-3+ search under the 14s budget while staying multi-size (≫ 1×pot).
/// live-6 LEAN menu: one pot-sized bet + pot raise. The rich menu measured
/// 695 ms/iter at np=6 (over budget at any usable iter count); facing-bet
/// decision quality needs fold/call/raise-pot, not sizing breadth — 6-way pots
/// play simple and the pool is passive. Used by search_decision for live >= 6.
fn lean_bets_l6() -> BetSizeOptions {
    BetSizeOptions {
        bet: vec![BetSize::PotRelative(1.0)],
        raise: vec![BetSize::PotRelative(1.0)],
    }
}

fn bets_for_live(live: usize) -> BetSizeOptions {
    if live >= 6 { lean_bets_l6() } else { rich_bets() }
}

fn rich_bets() -> BetSizeOptions {
    BetSizeOptions {
        bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        raise: vec![BetSize::PotRelative(1.0)],
    }
}

/// Depth-limit leaves for a street-rooted subgame `tree`:
///   - flop/turn: the next-street CHANCE node (continuation integrates the
///     remaining runout via the street's bucket tables);
///   - river: the SHOWDOWN terminals (≥2 live) — there is no next chance, so the
///     bucketed river showdown values them; fold terminals stay exact.
fn depth_leaves(tree: &FlatTree, street: u8, is_river: bool, np: usize) -> Vec<usize> {
    let mut depth = Vec::new();
    if is_river {
        for n in 0..tree.num_nodes() {
            if tree.nodes[n].is_terminal() {
                let fm = tree.get_folded_mask(n);
                let live = (0..np).filter(|&p| fm & (1 << p) == 0).count();
                if live >= 2 {
                    depth.push(n);
                }
            }
        }
    } else {
        for n in 0..tree.num_nodes() {
            if tree.nodes[n].is_player() && tree.nodes[n].board_state == street {
                for &c in tree.node_children(n) {
                    let cn = &tree.nodes[c as usize];
                    if cn.is_chance() || cn.board_state != street {
                        depth.push(c as usize);
                    }
                }
            }
        }
    }
    depth.sort_unstable();
    depth.dedup();
    depth
}

/// Play ONE multiway postflop hand with per-street search for the bot.
/// `holes[i]` = seam seat i's hole cards (order = seam seats 0..np). `bot_seam`
/// = the bot's seam index. `base_commit` = each live seat's matched preflop
/// commit; `dead` = preflop dead money (folders' partials). `rake_spec` =
/// (rate_milli, cap). Returns (net chips per seam seat, live-at-showdown), or
/// None if a dealt runout is blocked (audit-style reject).
#[allow(clippy::too_many_arguments)]
pub fn play_seam_pluribus(
    bp: &Blueprint,
    holes: &[[u8; 2]],
    bot_seam: usize,
    base_commit: u32,
    dead: u32,
    rake_spec: (u32, u32),
    cfg: &SearchCfg,
    rng: &mut u64,
    selfplay: bool,
) -> Option<(Vec<i64>, u8)> {
    let np = holes.len();
    let nh = bp.nh;
    // hand index lookup in the blueprint's hand universe.
    let hc = &bp.game.table().hand_cards;
    let mut hand_of: HashMap<(u8, u8), usize> = HashMap::new();
    for h in 0..nh {
        let (a, b) = (hc[h * 2], hc[h * 2 + 1]);
        hand_of.insert((a.min(b), a.max(b)), h);
    }
    let hkey = |hole: [u8; 2]| (hole[0].min(hole[1]), hole[0].max(hole[1]));
    let blocked = |c: u8| holes.iter().take(np).any(|h| h[0] == c || h[1] == c);

    let spec = production_game_v1();
    let mut total_commit: Vec<i64> = vec![base_commit as i64; np];
    let mut folded: Vec<bool> = vec![false; np];
    let mut live_seats: Vec<usize> = (0..np).collect();
    let mut board: Vec<u8> = bp.flop.to_vec();
    let mut street = BoardState::Flop;
    let (mut ti, mut ri): (Option<usize>, Option<usize>) = (None, None);

    loop {
        let l = live_seats.len();
        if l == 1 {
            break; // uncontested — everyone else folded
        }
        let street_u8 = street as u8;
        let is_river = street == BoardState::River;
        let cont = match street {
            BoardState::Flop => ContStreet::Flop,
            BoardState::Turn => ContStreet::Turn(ti.unwrap()),
            BoardState::River => ContStreet::River(ti.unwrap(), ri.unwrap()),
            BoardState::Preflop => unreachable!(),
        };
        // survivors are matched ⇒ equal total_commit; pot = all commits + dead.
        let commit_entry = total_commit[live_seats[0]] as i32;
        let pot_entry = total_commit.iter().sum::<i64>() as i32 + dead as i32;
        let mut tcfg = spec.street_seam_config(street, l as u8, commit_entry, pot_entry, rich_bets());
        tcfg.max_bets_per_street = BetCap::all(3);
        // Depth-limited build: truncate flop/turn subgames at the next-street
        // chance (river is full — no next street). The frozen region below the
        // limit is never visited, so not building it is the big perf win.
        let tree = build_tree_depth_limited(&tcfg).ok()?;
        let depth = depth_leaves(&tree, street_u8, is_river, l);

        // bot decision strategy for THIS street (if the bot is still live).
        let bot_j = live_seats.iter().position(|&s| s == bot_seam);
        let dbg = std::env::var("PLDBG").is_ok();
        if dbg && bot_j.is_some() {
            eprintln!(
                "  street={street_u8} L={l} tree_nodes={} depth_leaves={} (building strat...)",
                tree.num_nodes(),
                depth.len()
            );
        }
        // PLK env: number of Pluribus continuation variants (1 = skip the k=4
        // multi-continuation robustification — diagnostic for its cost).
        let plk: usize = std::env::var("PLK").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let t0 = std::time::Instant::now();
        // Self-play needs the strategy for ALL seats (any acting player reads it);
        // normal play only when the bot is live.
        let need_strat = selfplay || bot_j.is_some();
        let bot_strat: Option<HashMap<usize, Vec<Vec<f32>>>> = if need_strat {
            let mut game = BucketedContinuationGame::new_street(
                &bp.game,
                &bp.bk,
                cont,
                cfg.sample_m,
                cfg.seed ^ (board.iter().fold(0u64, |a, &c| a.wrapping_mul(53).wrapping_add(c as u64))),
            );
            game.set_player_count(l as u8);
            let mut s = CpuMccfr::new(&tree, vec![nh; l]);
            s.set_depth_limit(&depth);
            if plk > 1 {
                s.setup_pluribus_continuations(&tree, plk, 5.0);
            }
            // Asymmetric λ: the bot seat sharp (cfg.lambda); opponent seats at
            // cfg.opp_lambda (lower ⇒ bot best-responds to softer/looser opponents
            // ⇒ exploits the loose pool). In self-play every seat is "the bot".
            let lam: Vec<f32> = (0..l)
                .map(|j| if selfplay || Some(j) == bot_j { cfg.lambda } else { cfg.opp_lambda })
                .collect();
            if std::env::var("DCFR").is_ok() {
                s.set_dcfr(1.5, 0.0, 2.0);
                s.enable_parallel();
            } else {
                s.set_lambda(lam);
                if std::env::var("PAR").is_ok() {
                    s.enable_parallel();
                }
            }
            s.run(&tree, &game, cfg.iters);
            let mut m = HashMap::new();
            for n in 0..tree.num_nodes() {
                if tree.nodes[n].is_player() && tree.nodes[n].board_state == street_u8 {
                    let na = tree.nodes[n].num_children as usize;
                    m.insert(n, s.get_average_strategy_fb(n, na, nh, tree.nodes[n].player_id));
                }
            }
            Some(m)
        } else {
            None
        };
        if dbg && bot_j.is_some() {
            eprintln!(
                "  search street={street_u8} L={l} nodes={} depth-leaves={} {:.2}s",
                tree.num_nodes(),
                depth.len(),
                t0.elapsed().as_secs_f64()
            );
        }

        // walk this street's betting on the subgame tree.
        let mut node = 0usize;
        loop {
            let n = &tree.nodes[node];
            if n.is_chance() || n.is_terminal() {
                break;
            }
            let j = n.player_id as usize;
            let seat = live_seats[j];
            let na = n.num_children as usize;
            let a = if (seat == bot_seam || selfplay) && bot_strat.is_some() {
                let strat = bot_strat.as_ref().unwrap().get(&node).expect("bot flop node");
                let h = *hand_of.get(&hkey(holes[seat])).expect("hand in universe");
                let mut x = (splitmix64(rng) % 1_000_000) as f32 / 1_000_000.0;
                let mut sel = na - 1;
                for a in 0..na {
                    let v = strat[a][h];
                    if x < v {
                        sel = a;
                        break;
                    }
                    x -= v;
                }
                if dbg {
                    let labels: Vec<u8> = tree
                        .node_children(node)
                        .iter()
                        .map(|&c| tree.nodes[c as usize].action_label)
                        .collect();
                    let dist: Vec<String> =
                        (0..na).map(|a| format!("{:.2}", strat[a][h])).collect();
                    eprintln!(
                        "    BOT s={street_u8} node={node} labels={labels:?} dist=[{}] -> a{sel}(L{})",
                        dist.join(","),
                        labels[sel]
                    );
                }
                sel
            } else {
                pop_postflop_action(&tree, node, holes[seat], &board, rng)
            };
            node = tree.node_children(node)[a] as usize;
        }

        // record this street's per-player contributions into the running totals.
        for j in 0..l {
            total_commit[live_seats[j]] += tree.get_contribution(node, j as u8) as i64;
        }
        let fmask = tree.get_folded_mask(node);
        let survivors: Vec<usize> =
            (0..l).filter(|&j| fmask & (1 << j) == 0).map(|j| live_seats[j]).collect();
        for j in 0..l {
            if fmask & (1 << j) != 0 {
                folded[live_seats[j]] = true;
            }
        }

        if tree.nodes[node].is_terminal() {
            break; // resolved this street (fold-out or river showdown)
        }
        // chance → advance to the next street with the survivors.
        live_seats = survivors;
        match street {
            BoardState::Flop => {
                let opts: Vec<usize> =
                    (0..bp.turns.len()).filter(|&t| !blocked(bp.turns[t])).collect();
                if opts.is_empty() {
                    return None;
                }
                let t = opts[(splitmix64(rng) % opts.len() as u64) as usize];
                ti = Some(t);
                board.push(bp.turns[t]);
                street = BoardState::Turn;
            }
            BoardState::Turn => {
                let t = ti.unwrap();
                let opts: Vec<usize> =
                    (0..bp.rivers[t].len()).filter(|&r| !blocked(bp.rivers[t][r])).collect();
                if opts.is_empty() {
                    return None;
                }
                let r = opts[(splitmix64(rng) % opts.len() as u64) as usize];
                ri = Some(r);
                board.push(bp.rivers[t][r]);
                street = BoardState::River;
            }
            BoardState::River => unreachable!(),
            BoardState::Preflop => unreachable!(),
        }
    }

    // ---- settle (mirrors match_play::play_seam) ----
    let n_live = folded.iter().filter(|&&f| !f).count();
    if n_live == 0 {
        return None;
    }
    let commits: Vec<u32> = (0..np).map(|p| total_commit[p] as u32).collect();
    let ranks: Vec<Option<u32>> = (0..np)
        .map(|p| {
            if folded[p] {
                None
            } else if n_live == 1 {
                Some(0)
            } else {
                let mut c = holes[p].to_vec();
                c.extend_from_slice(&board);
                Some(best5(&c).0)
            }
        })
        .collect();
    let mut net = settle_pots(&commits, &folded, &ranks, 0, (0, 0));
    let total_pot: u32 = commits.iter().sum::<u32>() + dead;
    let rake = ((total_pot as u64 * rake_spec.0 as u64) / 1000).min(rake_spec.1 as u64) as i64;
    let gain = dead as i64 - rake;
    let best = (0..np).filter(|&p| !folded[p]).map(|p| ranks[p].unwrap()).max().unwrap();
    let winners: Vec<usize> =
        (0..np).filter(|&p| !folded[p] && ranks[p].unwrap() == best).collect();
    let share = gain.div_euclid(winners.len() as i64);
    let mut odd = gain - share * winners.len() as i64;
    for &w in &winners {
        let extra = if odd > 0 { 1 } else { 0 };
        odd -= extra;
        net[w] += share + extra;
    }
    Some((net, n_live as u8))
}

/// Deal `bp.np` holes by SAMPLING each seat's hand from the cell's ENTERING
/// RANGE (`bp.game.initial_weight`), with card removal across seats + the board.
/// This replaces uniform random dealing — the bot's search assumes opponents
/// hold the cell range, so testing on range-weighted hands removes the biggest
/// simulation artifact (off-distribution garbage hands). Uses seat 0's range for
/// all seats (the postflop seam range is symmetric across seats).
pub fn deal_from_range(bp: &Blueprint, rng: &mut u64) -> Vec<[u8; 2]> {
    let np = bp.np;
    let nh = bp.nh;
    let hc = &bp.game.table().hand_cards;
    let w = bp.game.initial_weight(0); // board-blocking hands already 0-weight
    let total: f32 = w.iter().sum();
    let mut used: u64 = bp.flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let mut holes = vec![[0u8; 2]; np];
    for hole in holes.iter_mut().take(np) {
        loop {
            let r = (splitmix64(rng) % 1_000_000) as f32 / 1_000_000.0 * total;
            let mut acc = 0.0f32;
            let mut h = nh - 1;
            for (hh, &wh) in w.iter().enumerate() {
                acc += wh;
                if acc >= r {
                    h = hh;
                    break;
                }
            }
            let (c1, c2) = (hc[h * 2], hc[h * 2 + 1]);
            if used & (1 << c1) == 0 && used & (1 << c2) == 0 {
                used |= 1 << c1;
                used |= 1 << c2;
                *hole = [c1, c2];
                break;
            }
        }
    }
    holes
}

/// Build the FLOP subgame for a cell, search it, and return the EXPLOITABILITY
/// of the searched strategy (chip units, averaged over seats, reach-weighted) —
/// an OPPONENT-INDEPENDENT quality number: how far the per-street solve is from
/// Nash within the subgame. Low ⇒ the search converged to a good equilibrium;
/// high ⇒ few iters / weak continuation. `commit`/`pot` = the cell's flop-entry.
pub fn flop_search_exploitability(bp: &Blueprint, commit: i32, pot: i32, cfg: &SearchCfg) -> f32 {
    let np = bp.np;
    let nh = bp.nh;
    let spec = production_game_v1();
    let mut tcfg = spec.street_seam_config(BoardState::Flop, np as u8, commit, pot, rich_bets());
    tcfg.max_bets_per_street = BetCap::all(3);
    let tree = build_tree_depth_limited(&tcfg).expect("flop tree");
    let depth = depth_leaves(&tree, BoardState::Flop as u8, false, np);
    let game = BucketedContinuationGame::new(&bp.game, &bp.bk, cfg.sample_m, cfg.seed);
    let mut s = CpuMccfr::new(&tree, vec![nh; np]);
    s.set_depth_limit(&depth);
    if std::env::var("DCFR").is_ok() {
        // Nash regret-matching + DCFR (needs snapshot) + parallel.
        s.set_dcfr(1.5, 0.0, 2.0);
        s.enable_parallel();
    } else {
        s.set_lambda(vec![cfg.lambda; np]);
        if std::env::var("PAR").is_ok() {
            s.enable_parallel();
        }
    }
    s.run(&tree, &game, cfg.iters);
    let profile = StrategyProfile::from_usize_offsets(s.cum_strategy_slice(), s.node_offsets(), nh);
    // Exploitability with SUM-1 reach (the search used raw reach; normalization is
    // a uniform scale ⇒ the strategy's optimality is unchanged, but values land in
    // chip units rather than exploding by ~nh per opponent).
    let mut eval_game = BucketedContinuationGame::new(&bp.game, &bp.bk, cfg.sample_m, cfg.seed);
    eval_game.set_normalize_reach(true);
    exploitability(&tree, &eval_game, &profile)
}

/// Convergence sweep: the flop-subgame exploitability at each iteration checkpoint,
/// so we can read off how close the deployed iteration count gets to Nash and how
/// much further iterations would buy. Mirrors `flop_search_exploitability` (same
/// tree / DCFR / normalized-reach chip units) but solves fresh per checkpoint and
/// values leaves with `exploitability_dl` — which stops at the search's depth
/// limit exactly as CpuMccfr does, so the number is the search's convergence on
/// the game it actually solved (not a deeper game it never explored). Uniform
/// entering ranges (like `flop_search_exploitability`); the magnitude is a solver-
/// convergence measure, ~independent of the reach prior. Returns (iters, chips).
pub fn flop_search_exploitability_sweep(
    bp: &Blueprint,
    commit: i32,
    pot: i32,
    cfg: &SearchCfg,
    checkpoints: &[u32],
) -> Vec<(u32, f32)> {
    let np = bp.np;
    let nh = bp.nh;
    let spec = production_game_v1();
    let mut tcfg = spec.street_seam_config(BoardState::Flop, np as u8, commit, pot, rich_bets());
    tcfg.max_bets_per_street = BetCap::all(3);
    let tree = build_tree_depth_limited(&tcfg).expect("flop tree");
    let depth = depth_leaves(&tree, BoardState::Flop as u8, false, np);
    let mut depth_bool = vec![false; tree.num_nodes()];
    for &n in &depth {
        depth_bool[n] = true;
    }
    let mut out = Vec::with_capacity(checkpoints.len());
    for &iters in checkpoints {
        let game = BucketedContinuationGame::new(&bp.game, &bp.bk, cfg.sample_m, cfg.seed);
        let mut s = CpuMccfr::new(&tree, vec![nh; np]);
        s.set_depth_limit(&depth);
        // PLK>1 attempts Pluribus k-continuation robustification. NOTE: this is a
        // no-op under set_depth_limit — setup_pluribus_continuations biases ONLY
        // `freeze_node`-frozen continuation nodes, and a depth-limited (truncated)
        // tree has none (the recurse short-circuits at depth_limit → static
        // evaluate_continuation). Exposed here purely to demonstrate that inertness
        // empirically (PLK=4 must equal PLK=1).
        let plk: usize = std::env::var("PLK").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        if plk > 1 {
            s.setup_pluribus_continuations(&tree, plk, 5.0);
        }
        s.set_dcfr(1.5, 0.0, 2.0); // match the deployed search (DCFR on)
        s.enable_parallel();
        s.run(&tree, &game, iters);
        let profile = StrategyProfile::from_usize_offsets(s.cum_strategy_slice(), s.node_offsets(), nh);
        let mut eval_game = BucketedContinuationGame::new(&bp.game, &bp.bk, cfg.sample_m, cfg.seed);
        eval_game.set_normalize_reach(true);
        out.push((iters, exploitability_dl(&tree, &eval_game, &profile, &depth_bool)));
    }
    out
}

/// 2-STREET convergence sweep (validation harness for the Pluribus safe-continuation
/// machinery): build a flop tree that SEARCHES flop + turn and truncates at the
/// river-deal (the bucketed game now values that leaf with the river runout for the
/// in-tree-dealt turn). Solves fresh per checkpoint and returns (iters, exploit
/// chips) on the 2-street game. Used to confirm the 2-street game converges (Gate 2)
/// before freeze/k=4 are layered on.
pub fn flop_search_exploitability_2street_sweep(
    bp: &Blueprint,
    commit: i32,
    pot: i32,
    cfg: &SearchCfg,
    checkpoints: &[u32],
) -> Vec<(u32, f32, usize)> {
    let np = bp.np;
    let nh = bp.nh;
    let spec = production_game_v1();
    let mut tcfg = spec.street_seam_config(BoardState::Flop, np as u8, commit, pot, rich_bets());
    tcfg.max_bets_per_street = BetCap::all(3);
    // BRANCHED chance: the turn-deal node forks into one subtree per runout turn
    // (branch factor = the game's chance-outcome count), so the searched turn street
    // gets CARD-AWARE per-outcome storage — without this the turn strategy is shared
    // across all turns (card-agnostic) and a card-aware best-response leaves a ~25%
    // structural gap that never converges.
    let nturns = {
        let g = BucketedContinuationGame::new(&bp.game, &bp.bk, cfg.sample_m, cfg.seed);
        g.num_chance_outcomes()
    };
    let tree = if std::env::var("BRANCHED").map(|v| v != "0").unwrap_or(true) {
        build_tree_depth_limited_2_branched(&tcfg, nturns).expect("branched 2-street flop tree")
    } else {
        build_tree_depth_limited_2(&tcfg).expect("2-street flop tree")
    };
    // Depth leaf = the TURN→river transition (the river-deal chance). The flop AND
    // turn player nodes are searched; the river runout is the bucketed continuation.
    let depth = depth_leaves(&tree, BoardState::Turn as u8, false, np);
    let mut depth_bool = vec![false; tree.num_nodes()];
    for &n in &depth {
        depth_bool[n] = true;
    }
    let nodes = tree.num_nodes();
    // Per-node turn map (read-only) so the continuation reads the in-tree turn from
    // the tree, not a shared mutable Cell — required for parallel branch valuation.
    let node_turn = solver_core::solver::bucketed_search::build_node_turn(&tree);
    let mut out = Vec::with_capacity(checkpoints.len());
    for &iters in checkpoints {
        let mut game = BucketedContinuationGame::new(&bp.game, &bp.bk, cfg.sample_m, cfg.seed);
        game.set_node_turn(node_turn.clone());
        game.set_cell_free(true);
        let mut s = CpuMccfr::new(&tree, vec![nh; np]);
        s.set_depth_limit(&depth);
        s.set_dcfr(1.5, 0.0, 2.0);
        // PAR2: parallelize the disjoint turn branches (cell_free game ⇒ Cell-free walk).
        if std::env::var("PAR2").map(|v| v != "0").unwrap_or(false) {
            s.enable_parallel();
        }
        s.run(&tree, &game, iters);
        let profile = StrategyProfile::from_usize_offsets(s.cum_strategy_slice(), s.node_offsets(), nh);
        let mut eval_game = BucketedContinuationGame::new(&bp.game, &bp.bk, cfg.sample_m, cfg.seed);
        eval_game.set_node_turn(node_turn.clone());
        eval_game.set_cell_free(true);
        eval_game.set_normalize_reach(true);
        out.push((iters, exploitability_dl(&tree, &eval_game, &profile, &depth_bool), nodes));
    }
    out
}

/// Self-consistent Pluribus k-continuation FLOP search on the card-aware branched
/// 2-street tree. (1) WARM-UP: solve flop+turn to a coherent equilibrium. (2) FREEZE
/// every turn (next-street) node to that strategy. (3) `setup_pluribus_continuations(k)`
/// biases the frozen turn into blueprint + fold/call/raise variants the opponent picks
/// among. (4) RE-SOLVE the flop against the frozen+biased turn — so the flop strategy
/// is robust to the opponent deviating in the continuation (the Pluribus safe-
/// continuation property). SEQUENTIAL (frozen/boundary nodes are unsupported by the
/// parallel walk). Returns (root reach-uniform aggression, root avg strategy [na][nh],
/// tree node count).
pub fn flop_search_k4(
    bp: &Blueprint,
    commit: i32,
    pot: i32,
    cfg: &SearchCfg,
    k: usize,
    warm_iters: u32,
    solve_iters: u32,
) -> (f32, Vec<Vec<f32>>, usize) {
    let np = bp.np;
    let nh = bp.nh;
    let spec = production_game_v1();
    let mut tcfg = spec.street_seam_config(BoardState::Flop, np as u8, commit, pot, rich_bets());
    tcfg.max_bets_per_street = BetCap::all(3);
    let nturns = {
        let g = BucketedContinuationGame::new(&bp.game, &bp.bk, cfg.sample_m, cfg.seed);
        g.num_chance_outcomes()
    };
    let tree = build_tree_depth_limited_2_branched(&tcfg, nturns).expect("branched 2-street");
    let depth = depth_leaves(&tree, BoardState::Turn as u8, false, np);
    let turn_u8 = BoardState::Turn as u8;
    let flop_u8 = BoardState::Flop as u8;
    let node_turn = solver_core::solver::bucketed_search::build_node_turn(&tree);

    // (1) WARM-UP solve → coherent turn continuation. Cell-free game ⇒ the disjoint
    // turn branches parallelize (no frozen nodes here, so the parallel walk applies).
    let mut warm_game = BucketedContinuationGame::new(&bp.game, &bp.bk, cfg.sample_m, cfg.seed);
    warm_game.set_node_turn(node_turn.clone());
    warm_game.set_cell_free(true);
    let mut warm = CpuMccfr::new(&tree, vec![nh; np]);
    warm.set_depth_limit(&depth);
    warm.set_dcfr(1.5, 0.0, 2.0);
    warm.enable_parallel();
    warm.run(&tree, &warm_game, warm_iters);

    // (2) FREEZE every turn node to the warm-up average strategy.
    let mut s = CpuMccfr::new(&tree, vec![nh; np]);
    s.set_depth_limit(&depth);
    for n in 0..tree.num_nodes() {
        if tree.nodes[n].is_player() && tree.nodes[n].board_state == turn_u8 {
            let na = tree.nodes[n].num_children as usize;
            let avg = warm.get_average_strategy(n, na, nh); // [na][nh]
            let mut flat = vec![0f32; na * nh];
            for a in 0..na {
                for h in 0..nh {
                    flat[a * nh + h] = avg[a][h];
                }
            }
            s.freeze_node(n, &flat);
        }
    }
    // (3) BIAS the frozen turn into k continuation variants (opponent selects).
    if k > 1 {
        s.setup_pluribus_continuations(&tree, k, 5.0);
    }
    s.set_dcfr(1.5, 0.0, 2.0);
    // Parallelize the disjoint frozen turn branches (walk_cfr_par now handles
    // frozen/boundary nodes); Cell-free game keeps the branches off the inner Cell.
    s.enable_parallel();
    // (4) RE-SOLVE the flop against the frozen + biased continuation.
    let mut game = BucketedContinuationGame::new(&bp.game, &bp.bk, cfg.sample_m, cfg.seed);
    game.set_node_turn(node_turn.clone());
    game.set_cell_free(true);
    s.run(&tree, &game, solve_iters);

    // (5) Root summary: reach-uniform aggression (P(bet/raise/allin)) + the strategy.
    let na0 = tree.nodes[0].num_children as usize;
    let root = s.get_average_strategy(0, na0, nh);
    let labels: Vec<u8> = tree.node_children(0).iter().map(|&c| tree.nodes[c as usize].action_label).collect();
    let mut aggr = 0.0f32;
    for h in 0..nh {
        for a in 0..na0 {
            if matches!(labels[a], 3 | 4 | 5) {
                aggr += root[a][h];
            }
        }
    }
    aggr /= nh as f32;
    (aggr, root, tree.num_nodes())
}

/// Run ONE per-street search and extract the strategy for `street_u8` nodes.
/// `overrides` = per-subgame-seat reach overrides (paired-bot shared info).
#[allow(clippy::too_many_arguments)]

/// SUBGAME ROOTING: freeze the OBSERVED street-action prefix to probability 1
/// so every solve iteration trains the ACTUAL line. Without this, equilibrium
/// range-checking leaves facing-bet subtrees off-path/under-trained (measured:
/// trash hands read UNIFORM facing 6-way pot bets — worse than pot-odds).
/// Matching mirrors api::walk_to_node (label → class → non-fold; nearest
/// amount) so the solve trains exactly the node the read will land on.
/// Walk the observed street-action prefix through `tree` (label → class →
/// nearest-amount matching, identical to api::walk_to_node) and return the
/// (node, action_idx, num_actions) picks — shared by CPU freeze and GPU force.
pub(crate) fn prefix_picks(tree: &FlatTree, prefix: &[(u8, u32)]) -> Vec<(usize, usize, usize)> {
    let class = |l: u8| -> u8 { match l { 0 => 0, 1 | 2 => 1, _ => 2 } };
    let mut out = Vec::new();
    let mut node = 0usize;
    for &(label, to_total) in prefix {
        if !tree.nodes[node].is_player() { break; }
        let children = tree.node_children(node).to_vec();
        let lbl = |i: usize| tree.nodes[children[i] as usize].action_label;
        let mut cands: Vec<usize> = (0..children.len()).filter(|&i| lbl(i) == label).collect();
        if cands.is_empty() {
            cands = (0..children.len()).filter(|&i| class(lbl(i)) == class(label)).collect();
        }
        if cands.is_empty() {
            cands = (0..children.len()).filter(|&i| label == 0 || lbl(i) != 0).collect();
        }
        let pick = match cands.len() {
            0 => break,
            1 => cands[0],
            _ => *cands.iter().min_by_key(|&&i| {
                (tree.nodes[children[i] as usize].amount - to_total as i32).abs()
            }).unwrap(),
        };
        out.push((node, pick, children.len()));
        node = children[pick] as usize;
    }
    out
}

fn freeze_prefix(solver: &mut CpuMccfr, tree: &FlatTree, nh: usize, prefix: &[(u8, u32)]) {
    for (node, pick, na) in prefix_picks(tree, prefix) {
        let mut one_hot = vec![0.0f32; na * nh];
        for h in 0..nh { one_hot[pick * nh + h] = 1.0; }
        solver.freeze_node(node, &one_hot);
    }
}

pub fn search_street_strat(
    bp: &Blueprint,
    tree: &FlatTree,
    depth: &[usize],
    cont: ContStreet,
    l: usize,
    street_u8: u8,
    cfg: &SearchCfg,
    seed_salt: u64,
    overrides: &[(usize, Vec<f32>)],
    prefix: &[(u8, u32)],
) -> HashMap<usize, Vec<Vec<f32>>> {
    let nh = bp.nh;
    let mut game =
        BucketedContinuationGame::new_street(&bp.game, &bp.bk, cont, cfg.sample_m, cfg.seed ^ seed_salt);
    game.set_player_count(l as u8);
    for (p, r) in overrides {
        game.set_reach_override(*p, r.clone());
    }
    let mut s = CpuMccfr::new(tree, vec![nh; l]);
    s.set_depth_limit(depth);
    s.set_lambda(vec![cfg.lambda; l]);
    // DCFR (explicit cfg, else DCFR env): discounted regret-matching, faster
    // convergence. Must be set before enable_parallel (the snapshot walk folds
    // the discount into its update).
    let dcfr = cfg.dcfr.unwrap_or_else(|| std::env::var("DCFR").is_ok());
    if dcfr {
        s.set_dcfr(1.5, 0.0, 2.0);
    }
    let par = cfg.par.unwrap_or_else(|| std::env::var("PAR").is_ok());
    if par {
        s.enable_parallel();
    }
    if !prefix.is_empty() {
        freeze_prefix(&mut s, tree, nh, prefix);
    }
    s.run(tree, &game, cfg.iters);
    let mut m = HashMap::new();
    for n in 0..tree.num_nodes() {
        if tree.nodes[n].is_player() && tree.nodes[n].board_state == street_u8 {
            let na = tree.nodes[n].num_children as usize;
            m.insert(n, s.get_average_strategy(n, na, nh));
        }
    }
    m
}

/// Play ONE multiway hand with a PAIR of bots (`bot_a`, `bot_b`) vs the pool,
/// optionally SHARING hole-card info for range blocking. With `share`, each bot's
/// per-street search models the partner as a POINT MASS (its known hand) and the
/// pool as a BLOCKER-NARROWED range (the partner's 2 cards removed) — a sharper
/// continuation read. Without `share`, the bots play the plain equilibrium (one
/// shared search), knowing only their own cards. Returns (net per seam seat,
/// live-at-showdown). Compare combined bot EV across `share` on/off to measure
/// the value of shared information.
#[allow(clippy::too_many_arguments)]
pub fn play_seam_pair(
    bp: &Blueprint,
    holes: &[[u8; 2]],
    bot_a: usize,
    bot_b: usize,
    share: bool,
    base_commit: u32,
    dead: u32,
    rake_spec: (u32, u32),
    cfg: &SearchCfg,
    rng: &mut u64,
    runout: Option<(usize, usize)>,
) -> Option<(Vec<i64>, u8)> {
    let np = holes.len();
    let nh = bp.nh;
    let hc = &bp.game.table().hand_cards;
    let mut hand_of: HashMap<(u8, u8), usize> = HashMap::new();
    for h in 0..nh {
        hand_of.insert((hc[h * 2].min(hc[h * 2 + 1]), hc[h * 2].max(hc[h * 2 + 1])), h);
    }
    let hkey = |hole: [u8; 2]| (hole[0].min(hole[1]), hole[0].max(hole[1]));
    let hand_idx = |seat: usize| *hand_of.get(&hkey(holes[seat])).expect("hand in universe");
    let is_bot = |seat: usize| seat == bot_a || seat == bot_b;

    let spec = production_game_v1();
    let mut total_commit: Vec<i64> = vec![base_commit as i64; np];
    let mut folded = vec![false; np];
    let mut live_seats: Vec<usize> = (0..np).collect();
    let mut board: Vec<u8> = bp.flop.to_vec();
    let mut street = BoardState::Flop;
    let (mut ti, mut ri): (Option<usize>, Option<usize>) = (None, None);

    loop {
        let l = live_seats.len();
        if l == 1 {
            break;
        }
        let street_u8 = street as u8;
        let is_river = street == BoardState::River;
        let cont = match street {
            BoardState::Flop => ContStreet::Flop,
            BoardState::Turn => ContStreet::Turn(ti.unwrap()),
            BoardState::River => ContStreet::River(ti.unwrap(), ri.unwrap()),
            BoardState::Preflop => unreachable!(),
        };
        let commit_entry = total_commit[live_seats[0]] as i32;
        let pot_entry = total_commit.iter().sum::<i64>() as i32 + dead as i32;
        let mut tcfg = spec.street_seam_config(street, l as u8, commit_entry, pot_entry, rich_bets());
        tcfg.max_bets_per_street = BetCap::all(3);
        let tree = build_tree_depth_limited(&tcfg).ok()?;
        let depth = depth_leaves(&tree, street_u8, is_river, l);
        let seed_salt = board.iter().fold(0u64, |a, &c| a.wrapping_mul(53).wrapping_add(c as u64));

        // Build the per-bot search strategies for this street.
        // overrides for bot X (partner Y): partner seat → point mass at Y's hand;
        // pool seats → default range with Y's 2 cards removed (the shared blocker).
        let make_overrides = |me: usize, partner: usize| -> Vec<(usize, Vec<f32>)> {
            let mut ov = Vec::new();
            if !share {
                return ov;
            }
            let partner_live = live_seats.contains(&partner);
            // base range (board-filtered) for this street, from a scratch game.
            let scratch =
                BucketedContinuationGame::new_street(&bp.game, &bp.bk, cont, cfg.sample_m, cfg.seed);
            let base = scratch.initial_weight(0);
            let blockers = if partner_live { holes[partner] } else { [255u8, 255u8] };
            let blocked: Vec<f32> = (0..nh)
                .map(|h| {
                    let (c1, c2) = (hc[h * 2], hc[h * 2 + 1]);
                    if c1 == blockers[0] || c2 == blockers[0] || c1 == blockers[1] || c2 == blockers[1] {
                        0.0
                    } else {
                        base[h]
                    }
                })
                .collect();
            // Shared info = RANGE BLOCKING on the POOL only: the pool can't hold
            // the partner's 2 cards, so narrow its range. The partner (a teammate)
            // is left at its default range — modeling it as a point-mass ADVERSARY
            // in adversarial CFR distorts the strategy (each bot best-responds to a
            // known-hand "opponent" that is really cooperating). True collusion
            // would need a cooperative/team solve; this captures only the blocking.
            for (j, &seat) in live_seats.iter().enumerate() {
                if seat == me || (seat == partner && partner_live) {
                    continue; // bot + teammate: default range
                }
                ov.push((j, blocked.clone())); // pool seat: partner-card-blocked
            }
            ov
        };

        let a_live = live_seats.contains(&bot_a);
        let b_live = live_seats.contains(&bot_b);
        // share=false ⇒ one shared equilibrium search; share=true ⇒ per-bot.
        let strat_shared = if !share && (a_live || b_live) {
            Some(search_street_strat(bp, &tree, &depth, cont, l, street_u8, cfg, seed_salt, &[], &[]))
        } else {
            None
        };
        let strat_a = if share && a_live {
            Some(search_street_strat(bp, &tree, &depth, cont, l, street_u8, cfg, seed_salt ^ 0xA, &make_overrides(bot_a, bot_b), &[]))
        } else {
            None
        };
        let strat_b = if share && b_live {
            Some(search_street_strat(bp, &tree, &depth, cont, l, street_u8, cfg, seed_salt ^ 0xB, &make_overrides(bot_b, bot_a), &[]))
        } else {
            None
        };

        let mut node = 0usize;
        loop {
            let n = &tree.nodes[node];
            if n.is_chance() || n.is_terminal() {
                break;
            }
            let seat = live_seats[n.player_id as usize];
            let na = n.num_children as usize;
            let a = if is_bot(seat) {
                let strat = if share {
                    if seat == bot_a { strat_a.as_ref() } else { strat_b.as_ref() }
                } else {
                    strat_shared.as_ref()
                }
                .and_then(|m| m.get(&node))
                .expect("bot node strat");
                let h = hand_idx(seat);
                let mut x = (splitmix64(rng) % 1_000_000) as f32 / 1_000_000.0;
                let mut sel = na - 1;
                for a in 0..na {
                    let v = strat[a][h];
                    if x < v {
                        sel = a;
                        break;
                    }
                    x -= v;
                }
                sel
            } else {
                pop_postflop_action(&tree, node, holes[seat], &board, rng)
            };
            node = tree.node_children(node)[a] as usize;
        }

        for (j, &seat) in live_seats.iter().enumerate() {
            total_commit[seat] += tree.get_contribution(node, j as u8) as i64;
        }
        let fmask = tree.get_folded_mask(node);
        let survivors: Vec<usize> =
            (0..l).filter(|&j| fmask & (1 << j) == 0).map(|j| live_seats[j]).collect();
        for j in 0..l {
            if fmask & (1 << j) != 0 {
                folded[live_seats[j]] = true;
            }
        }
        if tree.nodes[node].is_terminal() {
            break;
        }
        live_seats = survivors;
        let blocked = |c: u8| holes.iter().take(np).any(|h| h[0] == c || h[1] == c);
        match street {
            BoardState::Flop => {
                // Fixed runout (CRN) when provided, else sample a valid turn.
                let t = if let Some((ft, _)) = runout {
                    if blocked(bp.turns[ft]) {
                        return None;
                    }
                    ft
                } else {
                    let opts: Vec<usize> =
                        (0..bp.turns.len()).filter(|&t| !blocked(bp.turns[t])).collect();
                    if opts.is_empty() {
                        return None;
                    }
                    opts[(splitmix64(rng) % opts.len() as u64) as usize]
                };
                ti = Some(t);
                board.push(bp.turns[t]);
                street = BoardState::Turn;
            }
            BoardState::Turn => {
                let t = ti.unwrap();
                let r = if let Some((_, fr)) = runout {
                    if fr >= bp.rivers[t].len() || blocked(bp.rivers[t][fr]) {
                        return None;
                    }
                    fr
                } else {
                    let opts: Vec<usize> =
                        (0..bp.rivers[t].len()).filter(|&r| !blocked(bp.rivers[t][r])).collect();
                    if opts.is_empty() {
                        return None;
                    }
                    opts[(splitmix64(rng) % opts.len() as u64) as usize]
                };
                ri = Some(r);
                board.push(bp.rivers[t][r]);
                street = BoardState::River;
            }
            _ => unreachable!(),
        }
    }

    // settle (same as play_seam_pluribus)
    let n_live = folded.iter().filter(|&&f| !f).count();
    if n_live == 0 {
        return None;
    }
    let commits: Vec<u32> = (0..np).map(|p| total_commit[p] as u32).collect();
    let ranks: Vec<Option<u32>> = (0..np)
        .map(|p| {
            if folded[p] {
                None
            } else if n_live == 1 {
                Some(0)
            } else {
                let mut c = holes[p].to_vec();
                c.extend_from_slice(&board);
                Some(best5(&c).0)
            }
        })
        .collect();
    let mut net = settle_pots(&commits, &folded, &ranks, 0, (0, 0));
    let total_pot: u32 = commits.iter().sum::<u32>() + dead;
    let rake = ((total_pot as u64 * rake_spec.0 as u64) / 1000).min(rake_spec.1 as u64) as i64;
    let gain = dead as i64 - rake;
    let best = (0..np).filter(|&p| !folded[p]).map(|p| ranks[p].unwrap()).max().unwrap();
    let winners: Vec<usize> = (0..np).filter(|&p| !folded[p] && ranks[p].unwrap() == best).collect();
    let share_amt = gain.div_euclid(winners.len() as i64);
    let mut odd = gain - share_amt * winners.len() as i64;
    for &w in &winners {
        let extra = if odd > 0 { 1 } else { 0 };
        odd -= extra;
        net[w] += share_amt + extra;
    }
    Some((net, n_live as u8))
}

/// SINGLE-DECISION postflop search for the runtime API. Builds the current
/// street's subgame, searches it (optionally with pair-mode range blocking), and
/// returns (subgame tree, per-node strategy map). The caller maps the street's
/// betting history onto the tree to find the hero's decision node and reads its
/// action distribution for the hero's actual hand.
///
/// `board` = full board (3/4/5 cards). `live` = players still in (the subgame is
/// built for this count). `hero_idx`/`partner_idx` = indices WITHIN the live set
/// (0..live). `blocker_cards` = cards to remove from the POOL (non-hero,
/// non-partner) ranges — pair mode passes the partner's 2 hole cards (the solver
/// already filters the hero's own + board via the showdown). `commit_entry` =
/// each live seat's matched commit ENTERING this street; `pot_entry` = total pot
/// entering this street (≥ live·commit; excess is dead money). Search params come
/// from `cfg` + the PAR/DCFR env toggles (set them when launching the server).
#[allow(clippy::too_many_arguments)]
pub fn search_decision(
    bp: &Blueprint,
    board: &[u8],
    live: usize,
    hero_idx: usize,
    partner_idx: Option<usize>,
    blocker_cards: &[u8],
    commit_entry: i32,
    pot_entry: i32,
    cfg: &SearchCfg,
    reach_priors: &[(usize, Vec<f32>)],
    prefix: &[(u8, u32)],
) -> Option<(FlatTree, std::collections::HashMap<usize, Vec<Vec<f32>>>)> {
    let nh = bp.nh;
    let hc = &bp.game.table().hand_cards;
    let (street, cont) = match board.len() {
        3 => (BoardState::Flop, ContStreet::Flop),
        4 => {
            let ti = bp.turns.iter().position(|&c| c == board[3])?;
            (BoardState::Turn, ContStreet::Turn(ti))
        }
        5 => {
            let ti = bp.turns.iter().position(|&c| c == board[3])?;
            let ri = bp.rivers[ti].iter().position(|&c| c == board[4])?;
            (BoardState::River, ContStreet::River(ti, ri))
        }
        _ => return None,
    };
    let is_river = street == BoardState::River;
    let street_u8 = street as u8;
    // Per-blueprint economics: the seam tree (rake, stack depth) must match the
    // game class the blueprint was solved under (game.json manifest).
    let spec = crate::runtime_spec::runtime_game_spec().clone();
    let mut tcfg = spec.street_seam_config(street, live as u8, commit_entry, pot_entry, bets_for_live(live));
    tcfg.max_bets_per_street = BetCap::all(3);
    let tree = build_tree_depth_limited(&tcfg).ok()?;
    let depth = depth_leaves(&tree, street_u8, is_river, live);

    // Pair-mode range blocking: pool seats (not hero, not partner) lose every
    // hand containing a blocker card.
    let mut overrides: Vec<(usize, Vec<f32>)> = Vec::new();
    if !blocker_cards.is_empty() {
        let scratch =
            BucketedContinuationGame::new_street(&bp.game, &bp.bk, cont, cfg.sample_m, cfg.seed);
        let base = scratch.initial_weight(0);
        let blocked: Vec<f32> = (0..nh)
            .map(|h| {
                let (c1, c2) = (hc[h * 2], hc[h * 2 + 1]);
                if blocker_cards.iter().any(|&b| b == c1 || b == c2) {
                    0.0
                } else {
                    base[h]
                }
            })
            .collect();
        for j in 0..live {
            if j == hero_idx || Some(j) == partner_idx {
                continue;
            }
            overrides.push((j, blocked.clone()));
        }
    }

    // REACH-PRIOR (piece 3): entering ranges from the connected blueprint's
    // strategy (Bayes-ish posterior — preflop-continuing range) instead of uniform.
    // Applied AFTER pair-mode so blocking composes; reach-prior seats not pair-
    // blocked get the prior, pair-blocked seats keep their (already-narrowed) range.
    for (seat, reach) in reach_priors {
        if !overrides.iter().any(|(s, _)| s == seat) {
            overrides.push((*seat, reach.clone()));
        }
    }
    let seed_salt = board.iter().fold(0u64, |a, &c| a.wrapping_mul(53).wrapping_add(c as u64));
    // GPU depth-limited search (FLOP + TURN; opt-in via GPU_SEARCH + the `metal`
    // feature). River stays on CPU — its terminals are actual-board showdowns the
    // GPU's flop-base sorted arrays can't value. Falls back to the CPU search.
    //
    // ★ live ≤ 4 (2026-07-01): live-4 re-enabled after the np=4 factored
    // lone-survivor terminal kernels landed (commits 39232a2 + 778defd) — the
    // kernel-panic root cause was np≥4 terminals hitting the base K≥3 path's
    // O(nh³)-per-node single-thread dispatches (seconds long, non-preemptible ⇒
    // WindowServer starved >120s ⇒ macOS watchdog panic, twice). With the fast
    // pipeline: live-4 = 81ms/iter, longest dispatch ~15-20ms (compositor-
    // interleavable), parity 2.5e-7 vs the base path. Measured/validated paths:
    // HU ~12ms/iter, live-3 ~43ms/iter, live-4 ~81ms/iter. live-5+ has no fast
    // terminal (K=4) and stays on the CPU/rollout path.
    // live-5 GPU (2026-07-02): np5 CLUSTER terminal kernels (104.8 ms/iter,
    // in-solver parity 5.4e-8, 98s watchdog soak clean). Pairs-only cluster
    // mass = 6.5× MORE accurate than the factored mass the CPU search uses for
    // the same fold terminals — GPU live-5 is faster AND more accurate.
    // SUBGAME ROOTING now runs on GPU too (vcfr_force_strategies one-hots the
    // observed prefix after every strategy refresh), so facing-bet solves are
    // GPU-eligible at every gated live count.
    let gpu_ok = live <= 5 && matches!(cont, ContStreet::Flop | ContStreet::Turn(_));
    let strat = if gpu_ok && std::env::var("GPU_SEARCH").is_ok() {
        try_gpu_search(bp, &tree, cont, live, &overrides, cfg, prefix)
            .unwrap_or_else(|| search_street_strat(bp, &tree, &depth, cont, live, street_u8, cfg, seed_salt, &overrides, prefix))
    } else {
        search_street_strat(bp, &tree, &depth, cont, live, street_u8, cfg, seed_salt, &overrides, prefix)
    };
    Some((tree, strat))
}

/// GPU depth-limited search (Metal) for FLOP or TURN. Per-request `MetalContext`
/// (thread-safe, ~ms setup). Returns None when the `metal` feature is off or the
/// GPU is unavailable, so the caller falls back to the CPU search.
#[cfg(feature = "metal")]
fn try_gpu_search(
    bp: &Blueprint,
    tree: &FlatTree,
    cont: ContStreet,
    live: usize,
    overrides: &[(usize, Vec<f32>)],
    cfg: &SearchCfg,
    prefix: &[(u8, u32)],
) -> Option<std::collections::HashMap<usize, Vec<Vec<f32>>>> {
    use solver_core::gpu_metal::{gpu_search_street_strat, shared_context, GpuSearchCfg};
    // Shared process-wide context (metallib + queue loaded ONCE) — NOT a per-request
    // MetalContext::new(), which leaked wired GPU memory the driver never reclaimed.
    let ctx = shared_context()?;
    // Board-filtered base range (= the CPU game's initial_weight for this street),
    // overlaid with the per-seat overrides → [live][nh] reach prior.
    let scratch = BucketedContinuationGame::new_street(&bp.game, &bp.bk, cont, cfg.sample_m, cfg.seed);
    let base = scratch.initial_weight(0);
    let mut reach: Vec<Vec<f32>> = (0..live).map(|_| base.clone()).collect();
    for (seat, r) in overrides {
        if *seat < live { reach[*seat] = r.clone(); }
    }
    // budget_ms: HARD runaway guard — a deep multiway tree must stop at the
    // real-time budget (matches the CPU path's adaptive trim), and an orphaned
    // solve (client disconnect) dies within it instead of pinning the GPU.
    let gcfg = GpuSearchCfg { iters: cfg.iters, sample_m: cfg.sample_m, seed: cfg.seed, factored_terminals: true, lambda: cfg.lambda, budget_ms: 16_000 };
    let picks = prefix_picks(tree, prefix);
    Some(gpu_search_street_strat(ctx, tree, bp.game.table(), &bp.bk, cont, &reach, &gcfg, &picks))
}

#[cfg(not(feature = "metal"))]
fn try_gpu_search(
    _bp: &Blueprint, _tree: &FlatTree, _cont: ContStreet, _live: usize,
    _overrides: &[(usize, Vec<f32>)], _cfg: &SearchCfg,
    _prefix: &[(u8, u32)],
) -> Option<std::collections::HashMap<usize, Vec<Vec<f32>>>> {
    None
}

/// Hand index of `cards` in the blueprint universe (for reading the searched
/// strategy at the hero's actual hand). None if not a valid (non-board) hand.
pub fn hand_index(bp: &Blueprint, cards: [u8; 2]) -> Option<usize> {
    let hc = &bp.game.table().hand_cards;
    let key = (cards[0].min(cards[1]), cards[0].max(cards[1]));
    (0..bp.nh).find(|&h| (hc[h * 2].min(hc[h * 2 + 1]), hc[h * 2].max(hc[h * 2 + 1])) == key)
}
