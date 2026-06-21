//! FULL-HAND production baseline (piece 3): a 6-max preflop betting state
//! machine where the BOT follows its EQR raise-or-fold strategy and the POOL
//! injects limp/flat/3bet/fold, resolving to a flop SeamCell (live, commit, pot)
//! → the v1 postflop blueprint. OPTION A for limpers: the bot has no trained
//! limp response, so first-in (incl. only-limpers-behind) it plays its OPEN
//! (RFI) node as an iso-raise; the bot's EQR node when facing a raise is found
//! by skeleton-replay (limps/flats = fold-edges, raises = raise-edges).
//!
//! This module is the PREFLOP half (+ flop-entry resolution). Postflop play and
//! settlement build on top once the flop-entry distribution is validated.

use crate::pool_preflop::{PoolPreflop, PreAction};
use crate::preflop_player::{splitmix64, PreflopPlayer};
use solver_core::solver::postflop_oracle::SeamCell;
use solver_core::tree::flat::MAX_NA_PREFLOP;
use std::collections::HashMap;

/// Routes a flop SeamCell → the v1 postflop cell dir for live-3/4/5 (the solved
/// families), via `SeamCell::bucket_key`, with nearest-SPR-bin fallback for
/// situations the census didn't cover (the pool's limps shift the distribution
/// off the raise-or-fold census). live-2 (.bp2) and live-6 (rollout) are
/// separate paths handled by the caller.
pub struct FlopRouter {
    map: HashMap<(u8, i64), (i32, i32, String)>, // (live, spr_bin) -> (commit, pot, dir)
    live_bins: HashMap<u8, Vec<i64>>,            // per-live sorted spr_bins
    stack: i32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RouteKind {
    Exact,
    Fallback,
    Uncovered, // live-count not in the solved cell set (2 / 6)
}

impl FlopRouter {
    /// Parse a `cells.txt` (lines "CELL live=L commit=C pot=P b=B") under
    /// `blueprint_root` into the bucket map.
    pub fn load(blueprint_root: &str, cells_txt: &str, stack: i32) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(cells_txt)?;
        let mut map = HashMap::new();
        let mut live_bins: HashMap<u8, Vec<i64>> = HashMap::new();
        for line in text.lines() {
            if !line.starts_with("CELL live=") {
                continue;
            }
            let f = |k: &str| -> i64 {
                line.split_whitespace()
                    .find_map(|t| t.strip_prefix(k))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0)
            };
            let live = f("live=") as u8;
            let commit = f("commit=") as i32;
            let pot = f("pot=") as i32;
            let b = f("b=");
            let key = SeamCell { live, commit, pot }.bucket_key(stack);
            let dir = format!("{blueprint_root}/live{live}_c{commit}_p{pot}_b{b}");
            map.insert(key, (commit, pot, dir));
            live_bins.entry(live).or_default().push(key.1);
        }
        for v in live_bins.values_mut() {
            v.sort_unstable();
            v.dedup();
        }
        Ok(FlopRouter { map, live_bins, stack })
    }

    /// Route a flop entry → (cell dir, exact/fallback/uncovered).
    pub fn route(&self, cell: &SeamCell) -> (Option<String>, RouteKind) {
        let key = cell.bucket_key(self.stack);
        if let Some(v) = self.map.get(&key) {
            return (Some(v.2.clone()), RouteKind::Exact);
        }
        // nearest SPR bin for this live-count
        if let Some(bins) = self.live_bins.get(&cell.live) {
            if let Some(&nb) = bins.iter().min_by_key(|&&b| (b - key.1).abs()) {
                if let Some(v) = self.map.get(&(cell.live, nb)) {
                    return (Some(v.2.clone()), RouteKind::Fallback);
                }
            }
        }
        (None, RouteKind::Uncovered)
    }
}

// Position (action order) → constants. Seat index 0..5 = UTG..BB.
pub const UTG: usize = 0;
pub const SB: usize = 4;
pub const BB: usize = 5;

#[derive(Clone, Copy, PartialEq)]
pub enum Seat {
    Bot,
    Pool,
}

/// Outcome of the preflop round → the flop entry.
pub struct FlopEntry {
    pub folded: [bool; 6],
    pub commit: [i32; 6], // chips each seat put in preflop
    pub pot: i32,
    pub live: u8,
    pub cell: SeamCell,
}

/// Adjustable game economics. Production v1 = {rake_rate 0.05, rake_cap 20
/// (=10bb at bb=2), ante 0}. All flex so we can test other rake/ante structures.
#[derive(Clone, Copy)]
pub struct GameEcon {
    pub rake_rate: f64,
    pub rake_cap: i32, // chips
    pub ante: i32,     // chips, posted by every seat preflop
}

impl Default for GameEcon {
    fn default() -> Self {
        GameEcon { rake_rate: 0.05, rake_cap: 20, ante: 0 }
    }
}

pub struct FullHandSim {
    pf: PreflopPlayer,
    pool: PoolPreflop,
    pub stack: i32,
    sb: i32,
    bb: i32,
    pub econ: GameEcon,
    /// EQR open-decision node per opening position UTG..SB (BB never opens).
    open_node: [usize; 5],
}

impl FullHandSim {
    pub fn new(pf: PreflopPlayer, pool: PoolPreflop, stack: i32, sb: i32, bb: i32) -> Self {
        Self::with_econ(pf, pool, stack, sb, bb, GameEcon::default())
    }

    pub fn with_econ(
        pf: PreflopPlayer,
        pool: PoolPreflop,
        stack: i32,
        sb: i32,
        bb: i32,
        econ: GameEcon,
    ) -> Self {
        // Open node per position = follow the FOLD edge from the root that many
        // times (UTG=root; each fold advances to the next position's open).
        let mut open_node = [0usize; 5];
        let mut n = 0usize;
        for pos in 0..5 {
            open_node[pos] = n;
            if pos < 4 {
                let fold = pf.tree.node_children(n)
                    .iter()
                    .find(|&&c| pf.tree.nodes[c as usize].action_label == 0)
                    .copied()
                    .expect("fold edge");
                n = fold as usize;
            }
        }
        FullHandSim { pf, pool, stack, sb, bb, econ, open_node }
    }

    /// Bot's EQR decision node by skeleton-replay of `skel` (the fold/raise edge
    /// taken by each prior actor this hand). Returns None if the skeleton ended
    /// (terminal/chance) before the bot — e.g. a fully-limped pot reaching BB.
    fn bot_node(&self, skel: &[usize]) -> Option<usize> {
        let mut n = 0usize;
        for &child in skel {
            if !self.pf.tree.nodes[n].is_player() {
                return None;
            }
            n = child;
        }
        if self.pf.tree.nodes[n].is_player() {
            Some(n)
        } else {
            None
        }
    }

    /// Pot-relative raise-TO in chips for EQR raise action index `ai` at a node
    /// whose raise sizes are PotRelative(0.5 + 0.5*i) (i = ai-1, since action 0
    /// is fold). Pot-relative = raise the pot by f after calling.
    fn raise_to_chips(ai: usize, pot: i32, call_to: i32, my_commit: i32) -> i32 {
        let f = 0.5 + 0.5 * (ai.saturating_sub(1)) as f64;
        let to_call = (call_to - my_commit).max(0) as f64;
        let pot_after_call = pot as f64 + to_call;
        let raise_by = (f * pot_after_call).round() as i32;
        call_to + raise_by
    }

    /// Play one preflop round. `seats[pos]` = Bot/Pool; `holes[pos]` = the seat's
    /// hole cards (for the bot's hand class / pool's class). Returns the flop
    /// entry (who's live, commits, pot, SeamCell).
    pub fn play_preflop(
        &self,
        seats: &[Seat; 6],
        holes: &[[u8; 2]; 6],
        rng: &mut u64,
    ) -> FlopEntry {
        // Every seat antes (dead baseline); blinds post on top. to-call excludes
        // the ante (it's a baseline all share), so current_bet = ante + bb.
        let ante = self.econ.ante;
        let mut commit = [ante; 6];
        let mut folded = [false; 6];
        commit[SB] = ante + self.sb;
        commit[BB] = ante + self.bb;
        let mut current_bet = ante + self.bb;
        let mut num_raises = 0u32; // raises beyond the blind
        // need[p] = player p still owes an action this round. Set false when they
        // act (fold/call/check); a raise re-sets need=true for all other live
        // players. Round ends when no live player still needs to act. The BB
        // option falls out naturally (BB starts needing to act).
        let mut need = [true; 6];
        let mut skel: Vec<usize> = Vec::new(); // bot EQR skeleton edges

        let class: Vec<usize> = (0..6)
            .map(|p| PreflopPlayer::hand_class(holes[p][0], holes[p][1]))
            .collect();
        let mut buf = [0f32; MAX_NA_PREFLOP];

        let next = |p: usize| (p + 1) % 6;
        let live = |folded: &[bool; 6]| folded.iter().filter(|&&f| !f).count();
        let trace = std::env::var("FH_TRACE").is_ok();

        let mut pos = UTG;
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 200 {
                break; // safety
            }
            if live(&folded) <= 1 {
                break;
            }
            // Round ends when no live player still owes an action.
            if !(0..6).any(|p| !folded[p] && need[p]) {
                break;
            }
            // Advance to the next live player who still needs to act (skip
            // folded / done / all-in seats).
            let mut g2 = 0;
            while (folded[pos] || !need[pos] || commit[pos] >= self.stack) && g2 < 12 {
                pos = next(pos);
                g2 += 1;
            }

            let facing_raise = num_raises > 0;
            let pot: i32 = commit.iter().sum();

            // ---- decide the action ----
            // skeleton edge this seat contributes (for the bot's future lookups).
            let mut fold_seat = false;
            let mut raise_to: Option<i32> = None;

            match seats[pos] {
                Seat::Bot => {
                    // bot's EQR node
                    let node = if !facing_raise {
                        // open / iso over limpers: use this position's OPEN node
                        // (BB with no raise = limped-to: fall back to a check by
                        // folding the skeleton — handled as "limp/check" below).
                        if pos <= SB {
                            Some(self.open_node[pos])
                        } else {
                            None // BB option vs limpers — no trained node
                        }
                    } else {
                        self.bot_node(&skel)
                    };
                    match node {
                        Some(nd) => {
                            let na = self.pf.action_dist(nd, class[pos], &mut buf);
                            // sample
                            let mut x = (splitmix64(rng) % 1_000_000) as f32 / 1_000_000.0;
                            let mut ai = na - 1;
                            for a in 0..na {
                                if x < buf[a] {
                                    ai = a;
                                    break;
                                }
                                x -= buf[a];
                            }
                            let lbl = self.pf.tree.nodes[self.pf.tree.node_children(nd)[ai] as usize].action_label;
                            if lbl == 0 {
                                fold_seat = true;
                            } else {
                                raise_to = Some(Self::raise_to_chips(ai, pot, current_bet, commit[pos]));
                            }
                        }
                        None => {
                            // BB vs limpers (no trained node): check (stay, no raise).
                            // treated as a limp/check — match (free for BB).
                        }
                    }
                }
                Seat::Pool => {
                    let act = if !facing_raise {
                        self.pool.first_in(pos, class[pos], rng)
                    } else {
                        self.pool.facing_raise(pos, class[pos], rng)
                    };
                    match act {
                        PreAction::Fold => fold_seat = true,
                        PreAction::Limp | PreAction::Call => { /* match below */ }
                        PreAction::Raise => {
                            // pool sizing: ~3x open / 3x the bet when 3betting
                            let pot2: i32 = commit.iter().sum();
                            let raise_by = if facing_raise {
                                (current_bet as f64 * 2.0).round() as i32
                            } else {
                                (pot2 as f64 * 1.0).round() as i32 // ~pot-sized open
                            };
                            raise_to = Some(current_bet + raise_by);
                        }
                    }
                }
            }

            // CAP-3: no raises beyond the 3rd (matches the production
            // BetCap::all(3) the blueprint was solved under). A would-be raise
            // over the cap, or by an all-in player, becomes a call.
            if num_raises >= 3 || commit[pos] >= self.stack {
                raise_to = None;
            }

            // ---- apply the action + advance the bot skeleton ----
            let fold_child = |n: usize| -> usize {
                self.pf.tree.node_children(n)
                    .iter()
                    .find(|&&c| self.pf.tree.nodes[c as usize].action_label == 0)
                    .copied()
                    .unwrap_or(self.pf.tree.node_children(n)[0]) as usize
            };
            let raise_child = |n: usize| -> usize {
                // first raise edge (skeleton only needs "a raise happened")
                self.pf.tree.node_children(n)
                    .iter()
                    .find(|&&c| self.pf.tree.nodes[c as usize].action_label >= 3)
                    .copied()
                    .unwrap_or(self.pf.tree.node_children(n)[0]) as usize
            };
            // skeleton head = replay so far
            let skel_head = self.bot_node(&skel);

            if fold_seat {
                folded[pos] = true;
                if let Some(h) = skel_head {
                    if self.pf.tree.nodes[h].is_player() {
                        skel.push(fold_child(h));
                    }
                }
            } else if let Some(rt) = raise_to {
                commit[pos] = rt.min(self.stack);
                current_bet = commit[pos];
                num_raises += 1;
                if let Some(h) = skel_head {
                    if self.pf.tree.nodes[h].is_player() {
                        skel.push(raise_child(h));
                    }
                }
            } else {
                // call / limp / check: match the current bet (limp = match bb)
                commit[pos] = current_bet.min(self.stack);
                if let Some(h) = skel_head {
                    if self.pf.tree.nodes[h].is_player() {
                        skel.push(fold_child(h)); // limp/flat = fold-edge in skeleton
                    }
                }
            }

            if trace {
                let act = if fold_seat { "fold".to_string() }
                    else if let Some(rt) = raise_to { format!("raise->{rt}") }
                    else { "call/check".to_string() };
                eprintln!("  pos{pos} ({}): {act}  commit={} bet={current_bet} pot={}",
                    if seats[pos]==Seat::Bot {"BOT"} else {"pool"}, commit[pos], commit.iter().sum::<i32>());
            }
            // ---- update need-to-act ----
            need[pos] = false;
            if raise_to.is_some() && !fold_seat {
                for q in 0..6 {
                    // a raise re-opens action for live, not-yet-all-in players
                    if q != pos && !folded[q] && commit[q] < self.stack {
                        need[q] = true;
                    }
                }
            }
            pos = next(pos);
        }

        let pot: i32 = commit.iter().sum();
        let live_n = live(&folded) as u8;
        // SeamCell: live players' common commit (max among live), total pot.
        let commit_max = (0..6).filter(|&p| !folded[p]).map(|p| commit[p]).max().unwrap_or(0);
        let cell = SeamCell { live: live_n, commit: commit_max, pot };

        FlopEntry { folded, commit, pot, live: live_n, cell }
    }
}
