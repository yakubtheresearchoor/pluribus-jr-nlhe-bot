//! V1 SEAM-FAMILY match engine (harness re-pointing, step 1). The ante
//! flop-start oracle was the WRONG GAME for the multiway-quality question —
//! its "live-count" is flop-betting survival, not the v1 PREFLOP-survival
//! seam families (live-2..6). This plays the v1 seam games directly, ONE
//! family per game (seeded), so per-family hand counts are CONTROLLED
//! (dissolving the rare-family sampling problem) and the v1 RAKE is in the
//! measurement (load-bearing — the whole stakes argument is a rake argument).
//!
//! Built ON TOP of clean-rules (settle_pots), never modifying the verified
//! ground-truth engine. Dead money D (= pot − live·commit, from preflop
//! folders) is free to the live players, so Σ(live nets) = D − rake (not
//! −rake): the seam is a subgame competing for a pot with free dead money.

use clean_rules::eval::best5;
use clean_rules::table::settle_pots;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};

#[inline]
pub fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Synthetic seat policy (no blueprint needed — power/variance is a property
/// of the game, measured on synthetic strategies before any production solve).
#[derive(Clone, Copy)]
pub enum SeamPolicy {
    CheckFold,
    AlwaysAggressive,
    /// Stochastic TAG-ish mix — a BALLPARK for real blueprint mixing (the
    /// real variance depends on how much the actual blueprints mix, which is
    /// unknown until they exist; this gives an order-of-magnitude estimate
    /// far below the all-in upper bound). Open node: check 60% / bet 40%.
    /// Facing a bet: fold 35% / call 50% / raise 15%.
    Mixed,
    /// EQUITY ROLLOUT — honest hand-strength, COMPUTED not solved (the live-6
    /// "blueprint": no buckets, no balance, no bluffing). Estimates equity vs
    /// the field by Monte-Carlo over opponent hands + remaining runout on the
    /// revealed board, then value-bets / calls good equity, checks / folds bad.
    /// Board-DEPENDENT (actions use the revealed turn/river), so it is played
    /// through `play`/`play_board` (which know the board), NOT `play_aivat`
    /// (which fixes actions and varies the runout — invalid for this policy).
    EquityRollout,
}

/// One v1 seam family as a seeded game.
pub struct SeamGame {
    pub live: u8,
    pub tree: FlatTree,
    pub flop: [u8; 3],
    pub commit: u32, // each live player's preflop commit (in the reduced stack)
    pub dead: u32,   // pot − live·commit, free to the live players
    pub rake_milli: u32,
    pub rake_cap: u32,
}

impl SeamGame {
    pub fn new(live: u8, commit: i32, pot: i32, flop: [u8; 3]) -> Self {
        let spec = production_game_v1();
        let cfg = spec.flop_seam_config(
            live,
            commit,
            pot,
            BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        );
        let tree = build_tree(&cfg).expect("seam tree");
        SeamGame {
            live,
            tree,
            flop,
            commit: commit as u32,
            dead: (pot - live as i32 * commit) as u32,
            rake_milli: (spec.rake_rate * 1000.0).round() as u32,
            rake_cap: spec.rake_cap as u32,
        }
    }

    /// Deal `live` hole-hands + a turn + river from the full deck minus flop.
    pub fn deal(&self, rng: &mut u64) -> (Vec<[u8; 2]>, [u8; 5]) {
        let fm: u64 = self.flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
        let mut pool: Vec<u8> = (0..52u8).filter(|&c| fm & (1u64 << c) == 0).collect();
        // Fisher-Yates partial.
        let need = self.live as usize * 2 + 2;
        for i in 0..need {
            let j = i + (splitmix64(rng) % (pool.len() - i) as u64) as usize;
            pool.swap(i, j);
        }
        let holes: Vec<[u8; 2]> = (0..self.live as usize).map(|p| [pool[2 * p], pool[2 * p + 1]]).collect();
        let tr = pool[self.live as usize * 2];
        let rv = pool[self.live as usize * 2 + 1];
        let board = [self.flop[0], self.flop[1], self.flop[2], tr, rv];
        (holes, board)
    }

    /// Play one seam hand to a terminal; returns (net per live seat, live
    /// count at showdown). Σ(net) = dead − rake.
    pub fn play(&self, policies: &[SeamPolicy], holes: &[[u8; 2]], board: &[u8; 5], rng: &mut u64) -> (Vec<i64>, u8) {
        let node = self.walk_to_terminal(policies, holes, board, rng);
        self.settle_at(node, holes, board)
    }

    /// Monte-Carlo equity of `my` vs `n_opp` random opponents on the
    /// `revealed` board, over remaining runouts. Strategy-independent — the
    /// equity-rollout's "computed not solved" value estimate. `n_opp` is the
    /// ACTUAL live-opponent count (live − 1), so the rollout tightens
    /// correctly multiway (a hand that is strong heads-up can be trash 6-way).
    /// Public so the six-way gate can prove the opponent-count-awareness.
    pub fn equity(&self, my: [u8; 2], revealed: &[u8], n_opp: usize, rng: &mut u64, samples: usize) -> f64 {
        let mut blocked = (1u64 << my[0]) | (1u64 << my[1]);
        for &c in revealed { blocked |= 1u64 << c; }
        let deck: Vec<u8> = (0..52u8).filter(|&c| blocked & (1u64 << c) == 0).collect();
        let need = 5 - revealed.len();
        let draw = n_opp * 2 + need;
        let mut win = 0.0f64;
        let mut d = deck.clone();
        for _ in 0..samples {
            for i in 0..draw {
                let j = i + (splitmix64(rng) % (d.len() - i) as u64) as usize;
                d.swap(i, j);
            }
            let mut board: Vec<u8> = revealed.to_vec();
            for k in 0..need { board.push(d[n_opp * 2 + k]); }
            let mut mine = my.to_vec();
            mine.extend_from_slice(&board);
            let myr = best5(&mine).0;
            let mut tie = false;
            let mut lose = false;
            for o in 0..n_opp {
                let mut oc = vec![d[2 * o], d[2 * o + 1]];
                oc.extend_from_slice(&board);
                let or = best5(&oc).0;
                if or > myr { lose = true; break; }
                if or == myr { tie = true; }
            }
            if lose { continue; }
            win += if tie { 0.5 } else { 1.0 };
        }
        win / samples as f64
    }

    /// Walk the betting to a terminal. `board` + `holes` are used only by the
    /// board-dependent EquityRollout policy; synthetic policies ignore them.
    fn walk_to_terminal(&self, policies: &[SeamPolicy], holes: &[[u8; 2]], board: &[u8; 5], rng: &mut u64) -> usize {
        let mut node = 0usize;
        loop {
            let n = &self.tree.nodes[node];
            if n.is_terminal() {
                break;
            }
            if n.is_chance() {
                node = self.tree.node_children(node)[0] as usize; // board is external
                continue;
            }
            let p = n.player_id as usize;
            let children = self.tree.node_children(node);
            let pick = |prefs: &[u8]| -> usize {
                prefs
                    .iter()
                    .find_map(|&want| {
                        children.iter().position(|&c| self.tree.nodes[c as usize].action_label == want)
                    })
                    .expect("policy found no matching action label")
            };
            let a = match policies[p] {
                SeamPolicy::AlwaysAggressive => pick(&[3, 4, 5, 2]), // bet/raise/allin else call
                SeamPolicy::CheckFold => pick(&[1, 0]),              // check else fold
                SeamPolicy::Mixed => {
                    let labels: Vec<u8> =
                        children.iter().map(|&c| self.tree.nodes[c as usize].action_label).collect();
                    let has = |l: u8| labels.contains(&l);
                    let x = (splitmix64(rng) % 1000) as f64 / 1000.0;
                    let want: u8 = if has(1) {
                        // open node (can check): check 60% / bet 40%.
                        if x < 0.60 || !has(3) { 1 } else { 3 }
                    } else {
                        // facing a bet: fold 35% / call 50% / raise 15%.
                        if x < 0.35 { 0 } else if x < 0.85 || !has(4) { 2 } else { 4 }
                    };
                    labels.iter().position(|&l| l == want)
                        .or_else(|| labels.iter().position(|&l| l == 2 || l == 1 || l == 0))
                        .expect("mixed: no fallback action")
                }
                SeamPolicy::EquityRollout => {
                    let labels: Vec<u8> =
                        children.iter().map(|&c| self.tree.nodes[c as usize].action_label).collect();
                    let has = |l: u8| labels.contains(&l);
                    let street = self.tree.nodes[node].board_state as usize; // Flop0/Turn1/River2
                    let revealed = &board[0..(3 + street).min(5)];
                    let eq = self.equity(holes[p], revealed, self.live as usize - 1, rng, 60);
                    // Honest hand-strength: open ⇒ bet good equity else check;
                    // facing a bet ⇒ call decent equity else fold. No bluff/raise.
                    let want: u8 = if has(1) {
                        if eq > 0.55 && has(3) { 3 } else { 1 }
                    } else if eq > 0.42 { 2 } else { 0 };
                    labels.iter().position(|&l| l == want)
                        .or_else(|| labels.iter().position(|&l| l == 1 || l == 2 || l == 0))
                        .expect("rollout: no fallback action")
                }
            };
            node = self.tree.node_children(node)[a] as usize;
        }
        node
    }

    /// Settle a terminal node given holes + a full board. Σ(net) = dead − rake.
    fn settle_at(&self, node: usize, holes: &[[u8; 2]], board: &[u8]) -> (Vec<i64>, u8) {
        let np = self.live as usize;
        // Settle through clean-rules on the live commits (preflop commit +
        // flop contribution); dead money + v1 rake layered on top.
        let fold_mask = self.tree.get_folded_mask(node);
        let folded: Vec<bool> = (0..np).map(|p| fold_mask & (1 << p) != 0).collect();
        let live_commits: Vec<u32> =
            (0..np).map(|p| self.commit + self.tree.get_contribution(node, p as u8) as u32).collect();
        let n_live = folded.iter().filter(|&&f| !f).count();
        let ranks: Vec<Option<u32>> = (0..np)
            .map(|p| {
                if folded[p] {
                    None
                } else if n_live == 1 {
                    Some(0)
                } else {
                    let mut c = holes[p].to_vec();
                    c.extend_from_slice(board);
                    Some(best5(&c).0)
                }
            })
            .collect();
        // Contested distribution (no rake here — applied to the total below).
        let mut net = settle_pots(&live_commits, &folded, &ranks, 0, (0, 0));
        // Main-pot winners (best live hand) collect the dead money minus the
        // total rake (rake comes off the main pot — site main-pot rule).
        let best = (0..np).filter(|&p| !folded[p]).map(|p| ranks[p].unwrap()).max().unwrap();
        let winners: Vec<usize> = (0..np).filter(|&p| !folded[p] && ranks[p].unwrap() == best).collect();
        let total_pot: u32 = live_commits.iter().sum::<u32>() + self.dead;
        let rake = ((total_pot as u64 * self.rake_milli as u64) / 1000).min(self.rake_cap as u64) as i64;
        let gain = self.dead as i64 - rake;
        let share = gain.div_euclid(winners.len() as i64);
        let mut odd = gain - share * winners.len() as i64;
        for &w in &winners {
            let extra = if odd > 0 { 1 } else { 0 };
            odd -= extra;
            net[w] += share + extra;
        }
        (net, n_live as u8)
    }

    /// AIVAT runout control variate (exact Rao-Blackwellization over the
    /// turn+river). Plays the actions, then returns BOTH the raw net under
    /// one drawn runout AND the EXACT expectation of the net over ALL
    /// enumerable runouts. The expectation is unbiased by construction (it
    /// IS E[net | actions, holes]) and carries ZERO runout variance, so it
    /// removes the dominant showdown swing without changing the mean. Needs
    /// no value function — buildable + gateable now; the blueprint's
    /// action-node corrections compose on top at option 3.
    pub fn play_aivat(
        &self, policies: &[SeamPolicy], holes: &[[u8; 2]], action_rng: &mut u64, runout_rng: &mut u64,
    ) -> (Vec<i64>, Vec<f64>, u8) {
        let np = self.live as usize;
        // play_aivat is for board-INDEPENDENT (synthetic) policies — actions
        // must not depend on the runout (AIVAT fixes them and varies it).
        let dummy = [self.flop[0], self.flop[1], self.flop[2], 0u8, 1u8];
        let node = self.walk_to_terminal(policies, holes, &dummy, action_rng);
        let fold_mask = self.tree.get_folded_mask(node);
        let n_live = (0..np).filter(|&p| fold_mask & (1 << p) == 0).count();
        let mut blocked = self.flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
        for h in holes { blocked |= (1u64 << h[0]) | (1u64 << h[1]); }
        let deck: Vec<u8> = (0..52u8).filter(|&c| blocked & (1u64 << c) == 0).collect();
        let mk = |a: u8, b: u8| [self.flop[0], self.flop[1], self.flop[2], a, b];
        // No showdown ⇒ board irrelevant ⇒ raw == aivat (no runout variance).
        if n_live <= 1 {
            let (net, live) = self.settle_at(node, holes, &mk(deck[0], deck[1]));
            let aivat = net.iter().map(|&x| x as f64).collect();
            return (net, aivat, live);
        }
        // RAW: one drawn runout.
        let i = (splitmix64(runout_rng) % deck.len() as u64) as usize;
        let mut j = (splitmix64(runout_rng) % (deck.len() - 1) as u64) as usize;
        if j >= i { j += 1; }
        let (raw, live) = self.settle_at(node, holes, &mk(deck[i], deck[j]));
        // AIVAT: exact mean over ALL unordered {turn, river}.
        let mut acc = vec![0f64; np];
        let mut cnt = 0u64;
        for a in 0..deck.len() {
            for b in (a + 1)..deck.len() {
                let (net, _) = self.settle_at(node, holes, &mk(deck[a], deck[b]));
                for p in 0..np { acc[p] += net[p] as f64; }
                cnt += 1;
            }
        }
        let aivat: Vec<f64> = acc.iter().map(|&s| s / cnt as f64).collect();
        (raw, aivat, live)
    }
}
