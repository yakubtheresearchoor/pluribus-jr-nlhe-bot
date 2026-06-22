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
use solver_core::solver::bucketed_search::{BucketedContinuationGame, ContStreet};
use solver_core::solver::mccfr::CpuMccfr;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::build_tree_depth_limited;
use solver_core::tree::flat::FlatTree;
use std::collections::HashMap;

/// Per-decision search depth/sharpness.
#[derive(Clone, Copy)]
pub struct SearchCfg {
    pub iters: u32,
    pub lambda: f32,
    pub sample_m: u32,
    pub seed: u64,
}

impl Default for SearchCfg {
    fn default() -> Self {
        SearchCfg { iters: 160, lambda: 300.0, sample_m: 200, seed: 0x5EA12C }
    }
}

/// Search bet menu: 2 bet sizes (half / pot) + 1 raise (pot). Trimmed from
/// 3 bets + 2 raises — rich betting × multiway × lossless nh blows the tree up
/// exponentially in the branching factor, so fewer sizes is the lever that keeps
/// live-3+ search under the 14s budget while staying multi-size (≫ 1×pot).
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
            s.set_lambda(vec![cfg.lambda; l]);
            s.run(&tree, &game, cfg.iters);
            let mut m = HashMap::new();
            for n in 0..tree.num_nodes() {
                if tree.nodes[n].is_player() && tree.nodes[n].board_state == street_u8 {
                    let na = tree.nodes[n].num_children as usize;
                    m.insert(n, s.get_average_strategy(n, na, nh));
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
