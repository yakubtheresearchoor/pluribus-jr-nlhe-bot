//! Duplicate-play match core (H3 slice 2): blueprint agents walk the
//! oracle tree (the tree IS the state machine); chance nodes deal
//! turn/river in AUDIT mode (from the blueprint's sampled runout set,
//! excluding cards held by any player — the exact game the solver
//! solved); terminals settle through clean-rules (`settle_pots` +
//! `best5` — never solver code). Per-hand conservation is asserted
//! inside settle_pots.

use crate::blueprint::Blueprint;
use clean_rules::eval::best5;
use clean_rules::table::settle_pots;
use solver_core::solver::bucketed_flop_cfr::BucketedFlopCfr;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};
use std::collections::HashMap;

pub fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// A seat's policy.
#[derive(Clone, Copy)]
pub enum Policy<'a> {
    /// Play the blueprint's normalized average strategy.
    Blueprint(&'a Blueprint),
    /// Check when possible, fold to any bet (the exact-EV anchor).
    CheckFold,
    /// Most aggressive action available (bet, else call).
    AlwaysAggressive,
    /// NL10 loose-passive pool (postflop), calibrated to reference_6max_pool_priors
    /// (NL10 row): when given the option, continuation-bets 52.4% (else checks);
    /// facing a bet, folds 29.1% (fold-to-cbet), raises ~7.3% (chk-raise proxy),
    /// else calls. Street-uniform first model. The realistic fish the bot is judged against.
    Population,
}

pub struct MatchEnv<'a> {
    pub bp: &'a Blueprint, // defines the game (flop, runouts, maps)
    pub tree: &'a FlatTree,
    pub idx: BucketedFlopCfr,
    /// (c1, c2) → hand index in the table's universe.
    pub hand_of: HashMap<(u8, u8), usize>,
}

impl<'a> MatchEnv<'a> {
    pub fn new(bp: &'a Blueprint, tree: &'a FlatTree) -> Self {
        let idx = bp.indexer(tree);
        let hc = &bp.game.table().hand_cards;
        let mut hand_of = HashMap::new();
        for h in 0..bp.nh {
            let (a, b) = (hc[h * 2], hc[h * 2 + 1]);
            hand_of.insert((a.min(b), a.max(b)), h);
        }
        MatchEnv { bp, tree, idx, hand_of }
    }

    /// Normalized average-strategy distribution at (node, street ctx,
    /// hand) — uniform where no cum mass (the solver's own fallback).
    fn strategy(
        &self,
        node: usize,
        ti: Option<usize>,
        ri: Option<usize>,
        h: usize,
        out: &mut [f32],
    ) {
        let na = self.tree.nodes[node].num_children as usize;
        let nb = self.bp.nb;
        let bp = self.bp;
        let (cum, base, b): (&[f32], usize, u16) = if let Some(ri) = ri {
            let ti = ti.unwrap();
            let local = self.idx.river_local_offset_at(node).expect("river infoset");
            (
                &bp.cum_river,
                (ti * self.idx.max_river_outcomes() + ri) * self.idx.river_stride()
                    + local * MAX_NA_POSTFLOP * nb,
                bp.bk.river_map[ti][ri][h],
            )
        } else if let Some(ti) = ti {
            let local = self.idx.turn_local_offset_at(node).expect("turn infoset");
            (
                &bp.cum_turn,
                ti * self.idx.turn_stride() + local * MAX_NA_POSTFLOP * nb,
                bp.bk.turn_map[ti][h],
            )
        } else {
            let local = self.idx.flop_local_offset_at(node).expect("flop infoset");
            (&bp.cum_flop, local * MAX_NA_POSTFLOP * nb, bp.bk.flop_map[h])
        };
        let uniform = 1.0 / na as f32;
        if b == u16::MAX {
            out[..na].fill(uniform);
            return;
        }
        let mut sum = 0.0f32;
        for a in 0..na {
            let v = cum[base + a * nb + b as usize].max(0.0);
            out[a] = v;
            sum += v;
        }
        if sum > 0.0 {
            for v in out[..na].iter_mut() {
                *v /= sum;
            }
        } else {
            out[..na].fill(uniform);
        }
    }

    /// Play ONE hand in AUDIT mode with the given per-seat policies
    /// and dealt hole cards; returns net chips per seat (ante 2 each
    /// included). `rng` drives action sampling and runout draws.
    /// Returns None when the dealt hands block every sampled runout
    /// (audit mode rejection-samples such deals).
    pub fn play_hand(&self, policies: &[Policy], holes: &[[u8; 2]; 6], rng: &mut u64) -> Option<Vec<i64>> {
        self.play_hand_live(policies, holes, rng, None).map(|(c, _)| c)
    }

    /// Draw a runout (turn index, river index) valid for `holes` — the
    /// COMMON board a duplicate run replays across mirrored seatings so the
    /// runout variance (the dominant swing for all-in lines) cancels too,
    /// not just the hole-card luck.
    pub fn draw_runout(&self, holes: &[[u8; 2]; 6], rng: &mut u64) -> Option<(usize, usize)> {
        let blocked = |c: u8| holes.iter().any(|h| h[0] == c || h[1] == c);
        let to: Vec<usize> = (0..self.bp.turns.len()).filter(|&t| !blocked(self.bp.turns[t])).collect();
        if to.is_empty() { return None; }
        let t = to[(splitmix64(rng) % to.len() as u64) as usize];
        let ro: Vec<usize> = (0..self.bp.rivers[t].len()).filter(|&r| !blocked(self.bp.rivers[t][r])).collect();
        if ro.is_empty() { return None; }
        let r = ro[(splitmix64(rng) % ro.len() as u64) as usize];
        Some((t, r))
    }

    /// Like `play_hand` but also returns the LIVE count at showdown (the
    /// family stratification key). `runout` Some pins the board (for
    /// duplicate-play common runouts); None draws it as in plain audit mode.
    pub fn play_hand_live(&self, policies: &[Policy], holes: &[[u8; 2]; 6], rng: &mut u64, runout: Option<(usize, usize)>) -> Option<(Vec<i64>, u8)> {
        let blocked = |c: u8| holes.iter().any(|h| h[0] == c || h[1] == c);
        let mut node = 0usize;
        let (mut ti, mut ri): (Option<usize>, Option<usize>) = (None, None);
        let mut board: Vec<u8> = self.bp.flop.to_vec();
        let mut buf = [0f32; MAX_NA_POSTFLOP];
        loop {
            let n = &self.tree.nodes[node];
            if n.is_terminal() {
                break;
            }
            if n.is_chance() {
                // AUDIT mode: draw uniformly from the sampled runouts
                // that don't collide with any dealt hand.
                if ti.is_none() {
                    let t = if let Some((ft, _)) = runout {
                        if blocked(self.bp.turns[ft]) { return None; }
                        ft
                    } else {
                        let opts: Vec<usize> = (0..self.bp.turns.len())
                            .filter(|&t| !blocked(self.bp.turns[t]))
                            .collect();
                        if opts.is_empty() { return None; }
                        opts[(splitmix64(rng) % opts.len() as u64) as usize]
                    };
                    ti = Some(t);
                    board.push(self.bp.turns[t]);
                } else {
                    let t = ti.unwrap();
                    let r = if let Some((_, fr)) = runout {
                        if fr >= self.bp.rivers[t].len() || blocked(self.bp.rivers[t][fr]) { return None; }
                        fr
                    } else {
                        let opts: Vec<usize> = (0..self.bp.rivers[t].len())
                            .filter(|&r| !blocked(self.bp.rivers[t][r]))
                            .collect();
                        if opts.is_empty() { return None; }
                        opts[(splitmix64(rng) % opts.len() as u64) as usize]
                    };
                    ri = Some(r);
                    board.push(self.bp.rivers[t][r]);
                }
                node = self.tree.node_children(node)[0] as usize;
                continue;
            }
            // Player node.
            let p = n.player_id as usize;
            let na = n.num_children as usize;
            // Select by ACTION LABEL, not child position (2026-06-12):
            // positional selection silently inverts when the builder's
            // ordering convention shifts. Labels are pinned tree facts
            // (FOLD=0 CHECK=1 CALL=2 BET=3 RAISE=4 ALLIN=5).
            let children = self.tree.node_children(node);
            let pick = |prefs: &[u8]| -> usize {
                prefs
                    .iter()
                    .find_map(|&want| {
                        children.iter().position(|&c| {
                            self.tree.nodes[c as usize].action_label == want
                        })
                    })
                    .expect("policy found no matching action label")
            };
            let a = match &policies[p] {
                Policy::AlwaysAggressive => pick(&[3, 4, 5, 2]), // bet/raise/allin, else call
                Policy::CheckFold => pick(&[1, 0]),              // check, else fold
                Policy::Population => {
                    // NL10 loose-passive pool (postflop frequencies).
                    let has = |want: u8| {
                        children.iter().any(|&c| self.tree.nodes[c as usize].action_label == want)
                    };
                    let r = (splitmix64(rng) % 1_000_000) as f64 / 1_000_000.0;
                    if has(1) {
                        // open / checked-to: c-bet 52.4%, else check
                        if r < 0.524 { pick(&[3, 4, 5]) } else { pick(&[1]) }
                    } else {
                        // facing a bet: fold 29.1%, raise ~7.3%, else call
                        if r < 0.291 {
                            pick(&[0])
                        } else if r < 0.291 + 0.073 {
                            pick(&[4, 5, 3, 2])
                        } else {
                            pick(&[2, 0])
                        }
                    }
                }
                Policy::Blueprint(_) => {
                    let hkey = {
                        let h = holes[p];
                        (h[0].min(h[1]), h[0].max(h[1]))
                    };
                    let h = *self.hand_of.get(&hkey).expect("hand in universe");
                    self.strategy(node, ti, ri, h, &mut buf);
                    let mut x = (splitmix64(rng) % 1_000_000) as f32 / 1_000_000.0;
                    let mut pick = na - 1;
                    for (i, &v) in buf[..na].iter().enumerate() {
                        if x < v {
                            pick = i;
                            break;
                        }
                        x -= v;
                    }
                    pick
                }
            };
            node = self.tree.node_children(node)[a] as usize;
        }
        // Terminal: settle through clean-rules.
        let fold_mask = self.tree.get_folded_mask(node);
        let folded: Vec<bool> = (0..6).map(|p| fold_mask & (1 << p) != 0).collect();
        let commits: Vec<u32> =
            (0..6).map(|p| 2 + self.tree.get_contribution(node, p as u8) as u32).collect();
        let live = folded.iter().filter(|&&f| !f).count();
        let ranks: Vec<Option<u32>> = (0..6)
            .map(|p| {
                if folded[p] {
                    None
                } else if live == 1 {
                    Some(0)
                } else {
                    let mut cards = holes[p].to_vec();
                    cards.extend_from_slice(&board);
                    Some(best5(&cards).0)
                }
            })
            .collect();
        Some((settle_pots(&commits, &folded, &ranks, 0, (0, 0)), live as u8))
    }

    /// Deal hole cards from the deck minus the flop (seeded).
    pub fn deal_holes(&self, rng: &mut u64) -> [[u8; 2]; 6] {
        let fm: u64 =
            self.bp.flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
        let mut pool: Vec<u8> = (0..52u8).filter(|&c| fm & (1u64 << c) == 0).collect();
        for i in (1..pool.len()).rev() {
            let j = (splitmix64(rng) % (i as u64 + 1)) as usize;
            pool.swap(i, j);
        }
        let mut holes = [[0u8; 2]; 6];
        for p in 0..6 {
            holes[p] = [pool[2 * p], pool[2 * p + 1]];
        }
        holes
    }
}
