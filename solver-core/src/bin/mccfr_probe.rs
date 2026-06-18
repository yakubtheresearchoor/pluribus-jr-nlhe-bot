//! BATCHED MCCFR vs DCFR — WALL-ESCAPE MEASUREMENT (2026-06-17, branch
//! mccfr-cosolve-probe). Does external-sampling MCCFR's per-iter cost skip the
//! nb^num_opp multiway-showdown enumeration that makes the DCFR connected solve
//! cost explode with player count? Measurement branch, NOT production. The DCFR
//! path (BucketedFlopCfr) is untouched; this binary measures it head-to-head
//! against a batched external-sampling MCCFR on the IDENTICAL shrunk game.
//!
//! This file (step 1): the shrunk-game harness + the DCFR per-iter baseline by
//! live-count, establishing the wall (DCFR iter time should grow steeply with
//! live-count as nb^num_opp). MCCFR engine + the full-tree-enumerated
//! exploitability anchor land in subsequent steps.
//!
//! SHRINK KNOBS (recorded explicitly, env-overridable):
//!   MC_NB     buckets per street (default 6)
//!   MC_ITERS  DCFR iters for the per-iter timing (default 16)
//!   MC_NT/NR  turn/river runout samples (default 1/1 — single runout)
//! Both algorithms run on the game these knobs define; identical footing.

use std::time::Instant;

use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::card::Card;
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

/// The shrunk game for one live-count: small bucketed flop→river subgame on a
/// single canonical flop with an nt×nr runout. Returns the tree + table +
/// bucketing so DCFR and (later) MCCFR run on the IDENTICAL object.
pub struct ShrunkGame {
    pub tree: FlatTree,
    pub game: FlopStartGame,
    pub bk: FlopBucketing,
    pub live: u8,
    pub nb: usize,
}

pub fn build_shrunk(live: u8, nb: usize, nt: usize, nr: usize) -> ShrunkGame {
    let spec = production_game_v1();
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let ptable = PreflopChanceTable::new(
        6,
        vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6],
    );
    let canonical = ptable.canonical_flops[0];
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    // nt turn samples, nr river samples per turn (deterministic positions).
    let tp: &[usize] = match nt { 1 => &[12], 2 => &[12, 36], _ => &[12] };
    let turns: Vec<Card> = tp.iter().map(|&p| deck[p]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turns {
        let rd: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        let rp: &[usize] = match nr { 1 => &[10], 2 => &[10, 30], _ => &[10] };
        river_decks[tc as usize] = rp.iter().map(|&p| rd[p]).collect();
    }
    let tree = build_tree(&spec.flop_seam_config(live, 2, 12, bets)).unwrap();
    let table = FlopChanceTable::build_full_nh_sampled(canonical, live, &turns, &river_decks);
    let bk = FlopBucketing::quantile(&table, nb);
    let game = FlopStartGame::new(table);
    ShrunkGame { tree, game, bk, live, nb }
}

// ─────────────────────────────────────────────────────────────────────
// CPU batched external-sampling MCCFR (step 2). Reuses BucketedFlopCfr's
// river_tables + bucketing so the showdown is the IDENTICAL bucketed game as
// DCFR. External sampling: sample a hand per player (→ bucket trajectory),
// traverse the traverser's own actions, sample opponent actions; at the river
// run the per-tuple showdown DP for the ONE sampled opponent-bucket-tuple —
// O(num_opp) — instead of DCFR's O(nb^num_opp) enumeration. NOTE: this step
// targets the PER-ITER (wall-escape) number; the enumerated-exploitability
// convergence anchor + the garbage check are the explicitly-next milestone,
// so the convergence/quality of this engine is NOT yet trusted.
// ─────────────────────────────────────────────────────────────────────
struct Mccfr {
    node_local: Vec<i32>, // [nn], dense player-node index or -1
    n_info: usize,
    max_na: usize,
    nb: usize,
    np: usize,
    regret: Vec<f32>, // [n_info * nb * max_na]
    cum: Vec<f32>,
    flop_b: Vec<u16>,
    turn_b: Vec<u16>,
    river_b: Vec<u16>,
    tbl: solver_core::solver::bucketed_showdown::BucketedRunoutTables,
    // valid hands (no board/runout conflict) + their 2 cards.
    hand_cards: Vec<(u8, u8)>,
    valid: Vec<usize>, // hands with a real bucket on every street
    rng: u64,
}

impl Mccfr {
    fn new(g: &ShrunkGame) -> Self {
        use solver_core::tree::action::BoardState;
        let tree = &g.tree;
        let nn = tree.num_nodes();
        let mut node_local = vec![-1i32; nn];
        let mut n_info = 0usize;
        let mut max_na = 0usize;
        for i in 0..nn {
            if tree.nodes[i].is_player() {
                node_local[i] = n_info as i32;
                n_info += 1;
                max_na = max_na.max(tree.nodes[i].num_children as usize);
            }
        }
        // tables + maps straight from the bucketing (identical game as DCFR).
        let tbl = clone_tables(&g.bk.river_tables[0][0]);
        let nb = g.bk.nb_flop.max(g.bk.nb_turn).max(g.bk.nb_river);
        let table = g.game.table();
        let nh = table.num_valid;
        let hand_cards: Vec<(u8, u8)> =
            (0..nh).map(|h| (table.hand_cards[h * 2], table.hand_cards[h * 2 + 1])).collect();
        let no = u16::MAX;
        let valid: Vec<usize> = (0..nh)
            .filter(|&h| {
                g.bk.flop_map[h] != no && g.bk.turn_map[0][h] != no && g.bk.river_map[0][0][h] != no
            })
            .collect();
        let _ = BoardState::Flop;
        Mccfr {
            node_local,
            n_info,
            max_na,
            nb,
            np: g.live as usize,
            regret: vec![0.0; n_info * nb * max_na],
            cum: vec![0.0; n_info * nb * max_na],
            flop_b: g.bk.flop_map.clone(),
            turn_b: g.bk.turn_map[0].clone(),
            river_b: g.bk.river_map[0][0].clone(),
            tbl,
            hand_cards,
            valid,
            rng: 0x9E3779B97F4A7C15,
        }
    }

    #[inline]
    fn rand(&mut self) -> u64 {
        // xorshift64
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    #[inline]
    fn bucket_of(&self, board_state: u8, hand: usize) -> usize {
        use solver_core::tree::action::BoardState;
        let m = if board_state == BoardState::Flop as u8 {
            &self.flop_b
        } else if board_state == BoardState::Turn as u8 {
            &self.turn_b
        } else {
            &self.river_b
        };
        m[hand] as usize
    }

    /// regret-match a node's strategy into `out` (len na).
    fn strategy(&self, local: usize, bucket: usize, na: usize, out: &mut [f32]) {
        let base = (local * self.nb + bucket) * self.max_na;
        let mut sum = 0.0f32;
        for a in 0..na {
            let r = self.regret[base + a].max(0.0);
            out[a] = r;
            sum += r;
        }
        if sum > 0.0 {
            for a in 0..na {
                out[a] /= sum;
            }
        } else {
            let u = 1.0 / na as f32;
            for a in 0..na {
                out[a] = u;
            }
        }
    }

    /// Sample one distinct hand per player (no shared cards), valid for the runout.
    fn sample_deal(&mut self) -> Vec<usize> {
        let nv = self.valid.len();
        let mut hands = vec![0usize; self.np];
        let mut used: u64 = 0;
        for p in 0..self.np {
            loop {
                let r = (self.rand() as usize) % nv;
                let h = self.valid[r];
                let (c1, c2) = self.hand_cards[h];
                let m = (1u64 << c1) | (1u64 << c2);
                if used & m == 0 {
                    used |= m;
                    hands[p] = h;
                    break;
                }
            }
        }
        hands
    }

    /// External-sampling traverse for `traverser`; returns traverser CFV.
    fn traverse(&mut self, tree: &FlatTree, node: usize, traverser: usize, hands: &[usize]) -> f32 {
        let n = &tree.nodes[node];
        if n.is_terminal() {
            return self.terminal(tree, node, traverser, hands);
        }
        let kids = tree.node_children(node).to_vec();
        if n.is_chance() {
            // 1×1 runout: follow the single sampled child.
            return self.traverse(tree, kids[0] as usize, traverser, hands);
        }
        let player = n.player_id as usize;
        let na = kids.len();
        let bs = n.board_state;
        let local = self.node_local[node] as usize;
        let bucket = self.bucket_of(bs, hands[player]);
        let mut strat = [0.0f32; 16];
        self.strategy(local, bucket, na, &mut strat[..na]);
        if player == traverser {
            let mut cv = [0.0f32; 16];
            let mut v = 0.0f32;
            for a in 0..na {
                cv[a] = self.traverse(tree, kids[a] as usize, traverser, hands);
                v += strat[a] * cv[a];
            }
            let base = (local * self.nb + bucket) * self.max_na;
            for a in 0..na {
                self.regret[base + a] += cv[a] - v;
                self.cum[base + a] += strat[a];
            }
            v
        } else {
            // sample opponent action from their current strategy
            let r = (self.rand() as f64 / u64::MAX as f64) as f32;
            let mut acc = 0.0;
            let mut a = na - 1;
            for (i, &p) in strat[..na].iter().enumerate() {
                acc += p;
                if r <= acc {
                    a = i;
                    break;
                }
            }
            self.traverse(tree, kids[a] as usize, traverser, hands)
        }
    }

    /// Terminal value for the traverser: fold payoff or the per-tuple bucketed
    /// showdown (the recurse_eq_buckets DP run once for the sampled tuple).
    fn terminal(&self, tree: &FlatTree, node: usize, traverser: usize, hands: &[usize]) -> f32 {
        let fold_mask = tree.get_folded_mask(node);
        let np = self.np;
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(node, p as u8)).collect();
        let c_t = contribs[traverser];
        let half_pot = tree.starting_pot as f32 / np as f32 + c_t as f32;
        let total_pot: i32 = tree.starting_pot + contribs.iter().sum::<i32>();
        let trav_folded = (fold_mask >> traverser) & 1 == 1;
        if trav_folded {
            return -half_pot; // forfeit own stake (per the oracle's zero-sum convention)
        }
        // active opponents (not folded, not traverser)
        let active_opp: Vec<usize> =
            (0..np).filter(|&p| p != traverser && (fold_mask >> p) & 1 == 0).collect();
        if active_opp.is_empty() {
            // everyone else folded → traverser wins the pot (rake if flop seen)
            let rake = (total_pot as f32 * tree.rake_rate as f32).min(tree.rake_cap as f32).max(0.0);
            // win net = pot - own stake, in half_pot units: k = num_opp_folded share
            return half_pot * (np as f32 - 1.0) - rake; // approx: collect others' equal stakes
        }
        // showdown: per-tuple DP over the SAMPLED opponent river buckets.
        let nb = self.tbl.nb; // river table stride
        let bt = self.river_b[hands[traverser]] as usize;
        let k = active_opp.len() as f32;
        let rake = (total_pot as f32 * tree.rake_rate as f32).min(tree.rake_cap as f32).max(0.0);
        let rake_per_unit = if half_pot > 0.0 { rake / half_pot } else { 0.0 };
        // state-carrying (beaten, ties) DP, ONE tuple.
        let mut state = vec![0.0f32; np + 2];
        state[1] = 1.0;
        for &op in &active_opp {
            let bo = self.river_b[hands[op]] as usize;
            let i = bt * nb + bo;
            let (fw, ft, fl) = (self.tbl.f_w[i], self.tbl.f_t[i], self.tbl.f_l[i]);
            let fn_ = self.tbl.f_n[i];
            let norm = if fn_ > 0.0 { fn_ } else { 1.0 };
            let (pw, pt, pl) = (fw / norm, ft / norm, fl / norm); // condition on compatible
            let mut ns = vec![0.0f32; np + 2];
            if state[0] != 0.0 {
                ns[0] += state[0];
            }
            for j in 0..np {
                let s = state[1 + j];
                if s == 0.0 {
                    continue;
                }
                ns[0] += s * pl;
                ns[1 + j + 1] += s * pt;
                ns[1 + j] += s * pw;
            }
            state = ns;
        }
        let mut net = 0.0f32;
        if state[0] != 0.0 {
            net += state[0] * -1.0;
        }
        for j in 0..np {
            let s = state[1 + j];
            if s == 0.0 {
                continue;
            }
            let net_unit = if j == 0 {
                k - rake_per_unit
            } else {
                let t_f = (j + 1) as f32;
                (k + 1.0 - t_f) / t_f - rake_per_unit / t_f
            };
            net += s * net_unit;
        }
        half_pot * net
    }

    fn run_iter(&mut self, tree: &FlatTree, batch: usize) {
        for b in 0..batch {
            let traverser = b % self.np;
            let hands = self.sample_deal();
            self.traverse(tree, 0, traverser, &hands);
        }
    }
}

fn clone_tables(
    t: &solver_core::solver::bucketed_showdown::BucketedRunoutTables,
) -> solver_core::solver::bucketed_showdown::BucketedRunoutTables {
    solver_core::solver::bucketed_showdown::BucketedRunoutTables {
        nb: t.nb,
        f_w: t.f_w.clone(),
        f_t: t.f_t.clone(),
        f_l: t.f_l.clone(),
        f_n: t.f_n.clone(),
    }
}

/// DCFR per-iter timing. live==2 is exact HU (FlopStartVectorCfr, O(nh^2)
/// showdown — the bucketed designs force HU to exact); live>=3 is the bucketed
/// multiway wall (BucketedFlopCfr Design1Collapsed, O(nb^num_opp) showdown).
fn dcfr_per_iter(g: &ShrunkGame, iters: u32) -> (f64, usize) {
    use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
    let nn = g.tree.num_nodes();
    if g.live == 2 {
        let mut s = FlopStartVectorCfr::new(&g.tree, g.game.table());
        s.run(&g.tree, &g.game, 1);
        let t0 = Instant::now();
        s.run(&g.tree, &g.game, iters);
        return (t0.elapsed().as_secs_f64() / iters as f64, nn);
    }
    let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
    s.set_terminal_design(TerminalDesign::Design1Collapsed);
    s.run(&g.tree, &g.game, &g.bk, 1);
    let t0 = Instant::now();
    s.run(&g.tree, &g.game, &g.bk, iters);
    (t0.elapsed().as_secs_f64() / iters as f64, nn)
}

fn main() {
    let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let iters: u32 = std::env::var("MC_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let nt: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let nr: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(1);

    let batch: usize = std::env::var("MC_B").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let mc_iters: usize = std::env::var("MC_MCITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);

    println!("SHRUNK GAME: nb={nb} runout={nt}x{nr} | B={batch} | DCFR (enumerate) vs MCCFR (sample)\n");
    println!("{:>5} {:>8} {:>10} {:>13} {:>15} {:>14} {:>10}",
        "live", "num_opp", "nodes", "DCFR s/iter", "MCCFR µs/traj", "nb^num_opp", "DCFR/MC*");
    for live in 2u8..=5 {
        let g = build_shrunk(live, nb, nt, nr);
        let (dper, nodes) = dcfr_per_iter(&g, iters);
        // MCCFR: time `mc_iters` batches, report per-trajectory µs.
        let mut mc = Mccfr::new(&g);
        mc.run_iter(&g.tree, batch); // warm
        let t0 = Instant::now();
        for _ in 0..mc_iters {
            mc.run_iter(&g.tree, batch);
        }
        let total_traj = (mc_iters * batch) as f64;
        let mc_us = t0.elapsed().as_secs_f64() / total_traj * 1e6;
        // Equal-work ratio: DCFR per-iter vs MCCFR per (batch=nodes-ish) — report
        // DCFR-iter / MCCFR-per-traj as the raw per-unit speed ratio.
        let ratio = dper / (mc_us * 1e-6);
        let wall = (nb as f64).powi((live - 1) as i32);
        println!("{:>5} {:>8} {:>10} {:>13.4} {:>15.3} {:>14.0} {:>10.0}",
            live, live - 1, nodes, dper, mc_us, wall, ratio);
    }
    // SANITY: confirm the engine does REAL work (not a no-op that would also
    // look flat+fast). Run live-3 for a bit and report regret occupancy + a
    // sample of strategy spread. NOT a convergence claim — the anchor is next.
    {
        let g = build_shrunk(3, nb, nt, nr);
        let mut mc = Mccfr::new(&g);
        for _ in 0..50 {
            mc.run_iter(&g.tree, 4096);
        }
        let nz = mc.regret.iter().filter(|&&r| r != 0.0).count();
        let touched = mc.cum.iter().filter(|&&c| c > 0.0).count();
        let rmax = mc.regret.iter().cloned().fold(0.0f32, |m, x| m.max(x.abs()));
        println!(
            "\nSANITY (live-3, 50×4096 traj): regret non-zero {}/{} ({:.0}%), cum touched {} | max|regret| {:.3e}",
            nz, mc.regret.len(), 100.0 * nz as f64 / mc.regret.len() as f64, touched, rmax
        );
        println!("(non-trivial occupancy ⇒ the traversal really visits infosets + updates regret,");
        println!(" so the per-traj timing is real work, not a short-circuit. Convergence still UNVALIDATED.)");
    }

    println!("\nWALL-ESCAPE READ: DCFR s/iter should track nb^num_opp; MCCFR µs/traj should stay ~FLAT");
    println!("across live-count (it samples ONE tuple, O(num_opp), not the nb^num_opp enumeration).");
    println!("NOT YET MEASURED (next milestone): iters-to-converge vs the full-tree-enumerated");
    println!("exploitability anchor, + the AA-raises garbage check. Per-iter alone is half the answer.");
}
