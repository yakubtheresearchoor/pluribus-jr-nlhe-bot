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
    // MC_IDENTITY=1: identity bucketing (nb=nh) ⇒ the bucketed game IS the exact
    // game, so converged DCFR is a TRUE Nash (exploitability→0) — the localization
    // test for whether the anchor floor is structural or bucketing-specific.
    let bk = if std::env::var("MC_IDENTITY").is_ok() {
        FlopBucketing::identity(&table)
    } else {
        FlopBucketing::quantile(&table, nb)
    };
    let mut tree = tree;
    if std::env::var("MC_NORAKE").is_ok() {
        tree.rake_rate = 0.0;
        tree.rake_cap = 0.0;
    }
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
/// Pluribus negative-regret ("deep") pruning config — Brown & Sandholm 2019,
/// Science Supplement, Algorithm 1 (MCCFR with Negative-Regret Pruning).
/// VERBATIM PARAMS (absolute, Pluribus-scale): prune traverser actions with
/// regret < -300,000,000 AND their whole subtree, in 95% of iters, EXCEPT the
/// last betting round and actions immediately leading to a terminal; regret
/// floor -310,000,000 (below the prune threshold so pruned actions can recover);
/// warm-up before pruning starts; Linear-CFR discount d=(t/DI)/(t/DI+1) for
/// t<LCFR_Threshold. The absolute -300M/-310M are tuned to Pluribus's regret
/// magnitude (trillions of iters); they do NOT transfer to the shrunk game
/// (measured max|regret| ~1e6 here), so we carry them as game-scaled knobs and
/// SWEEP the threshold (it trades the speed and the quality being measured) —
/// per the convergence-measurement discipline. NOT yet wired into the traversal;
/// recorded so the convergence runs match Pluribus exactly, not from memory.
#[derive(Clone, Copy)]
struct PruneConfig {
    /// prune threshold C (game-scaled; default a multiple of the regret scale).
    c: f32,
    /// regret floor (must be < c so pruned actions can climb back to un-prune).
    floor: f32,
    /// probability an iteration prunes (Pluribus: 0.95).
    prune_prob: f32,
    /// iterations of warm-up before pruning engages.
    warmup: u32,
    /// never prune on the final betting round or terminal-adjacent actions.
    protect_last_round: bool,
}

impl Default for PruneConfig {
    fn default() -> Self {
        // Defaults scaled to the shrunk game's regret magnitude; SWEEP c.
        PruneConfig { c: -3.0e5, floor: -3.1e5, prune_prob: 0.95, warmup: 0, protect_last_round: true }
    }
}

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
    valid: Vec<usize>, // FLOP-live hands (deal set) — per-street death handled in traverse
    turn_card: u8,
    river_card: u8,
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
        // PER-STREET (mirror DCFR): deal from FLOP-live hands (incl. runout-
        // colliders, which are turn/river-live until their card appears). A
        // collider DIES at its collision street (handled in traverse: 0 from
        // there) — a sampled tuple containing a collider is an impossible world
        // → 0, which reproduces DCFR's river-valid-weighted CFV in expectation
        // (no reweighting: sampling already weights by the flop-valid distrib).
        let valid: Vec<usize> = (0..nh).filter(|&h| g.bk.flop_map[h] != no).collect();
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
            turn_card: table.remaining_deck[0],
            river_card: table.river_decks[table.remaining_deck[0] as usize][0],
            rng: 0x9E3779B97F4A7C15,
        }
    }

    /// hand is live at street `bs` iff it doesn't hold a board card dealt by then
    /// (flop already excluded at deal). Mirrors DCFR's per-street card removal.
    #[inline]
    fn alive(&self, hand: usize, bs: u8) -> bool {
        use solver_core::tree::action::BoardState;
        let (c1, c2) = self.hand_cards[hand];
        if bs >= BoardState::Turn as u8 && (c1 == self.turn_card || c2 == self.turn_card) {
            return false;
        }
        if bs >= BoardState::River as u8 && (c1 == self.river_card || c2 == self.river_card) {
            return false;
        }
        true
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
        // per-street death: the acting player's hand is impossible at this street
        // (holds a board card dealt by now) → impossible world → 0. Also guards the
        // NO_BUCKET strategy index under quantile.
        if !self.alive(hands[player], bs) {
            return 0.0;
        }
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
                // CFR+: floor cumulative regret at 0 (converges far faster than
                // vanilla, which let regrets drift and the average stay ~uniform).
                self.regret[base + a] = (self.regret[base + a] + cv[a] - v).max(0.0);
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
        let bs = tree.nodes[node].board_state;
        let fold_mask = tree.get_folded_mask(node);
        let np = self.np;
        // per-street death at the terminal: any ACTIVE player (traverser or a non-
        // folded opponent) whose hand is impossible at this board → impossible
        // world → 0 (catches showdowns reached without a river action). Mirrors
        // DCFR: collider tuples have 0 reach.
        for p in 0..np {
            if (fold_mask >> p) & 1 == 0 && !self.alive(hands[p], bs) {
                return 0.0;
            }
        }
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

// ─────────────────────────────────────────────────────────────────────
// TRUE best-response exploitability anchor (step 3). EXACT, full-tree,
// re-optimizing the deviation at every evaluee node (NOT one-step CFV reads —
// one-step is blind to chained pruning holes). Models DCFR's bucketed game BY
// CONSTRUCTION: per-hand reach with σ read via the bucket map (matching
// propagate_player_reach), the bucketed showdown at terminals. The opponents'
// counterfactual reach is computed ONCE per evaluee and SHARED between the BR
// and on-policy passes, so the gap is a consistent exploitability regardless of
// the absolute normalization — taming the normalize_opponent_reach seam.
// MULTIWAY (live≥3) only here (bucketed_showdown_cfv asserts np≥3); live-2 (the
// 6-max two-handed SUBGAME) folds in later via the exact HU showdown.
// VALIDATION (two-sided, the acceptance test): converged DCFR must read ~0
// (no over-report — the normalization-seam test) AND under-converged DCFR must
// read HIGH-and-FALLING (catches real exploitation — no under-report).
// ─────────────────────────────────────────────────────────────────────
struct Anchor {
    nb: usize,
    np: usize,
    nh: usize,
    // per-street hand→bucket maps for the 1×1 runout.
    fmap: Vec<u16>,
    tmap: Vec<u16>,
    rmap: Vec<u16>,
    ftbl: solver_core::solver::bucketed_showdown::BucketedRunoutTables,
    ttbl: solver_core::solver::bucketed_showdown::BucketedRunoutTables,
    rtbl: solver_core::solver::bucketed_showdown::BucketedRunoutTables,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    valid: Vec<usize>,       // flop-live hands (board-3 excluded) — root set
    valid_turn: Vec<usize>,  // flop-live minus turn-card colliders
    valid_river: Vec<usize>, // turn-live minus river-card colliders
    nc: f32,           // num_combinations — DCFR divides the showdown by this
    // per-node one-step regret accumulator (localization: WHERE the BR diverges
    // from on-policy on converged DCFR — the floor-contributing nodes).
    node_regret: std::cell::RefCell<Vec<f32>>,
    // per-node per-action aggregated cv (which ACTION the BR over-values).
    node_action_cv: std::cell::RefCell<Vec<Vec<f32>>>,
}

impl Anchor {
    fn new(g: &ShrunkGame) -> Self {
        let table = g.game.table();
        let nh = table.num_valid;
        let no = u16::MAX;
        // PER-STREET card removal, mirroring DCFR's reach convention. DCFR's
        // compute_reach_turn only zeros TURN-conflicts (via turn_map); a river-
        // collider (e.g. holds the river card) carries full reach into TURN
        // terminals and is zeroed only at the RIVER (river_map / chance-prob). So
        // the anchor must be per-street too: a hand is live at a street iff it
        // doesn't collide with the board DEALT SO FAR. All-street `valid` excluded
        // river-colliders at turn terminals where DCFR includes them — the cap-
        // independent residual. Card-check (not just map!=NO_BUCKET) so it also
        // holds under identity bucketing, which never emits NO_BUCKET.
        let turn_card = table.remaining_deck[0] as usize;
        let river_card = table.river_decks[turn_card][0] as usize;
        let hcards = |h: usize| -> u64 {
            (1u64 << table.hand_cards[h * 2]) | (1u64 << table.hand_cards[h * 2 + 1])
        };
        let tmask = 1u64 << turn_card;
        let rmask = 1u64 << river_card;
        // flop-valid: every dealt hand (board-3 already excluded at build time).
        let valid: Vec<usize> = (0..nh).filter(|&h| g.bk.flop_map[h] != no).collect();
        // turn-valid: flop-valid, not colliding the turn card.
        let valid_turn: Vec<usize> = valid.iter().copied()
            .filter(|&h| hcards(h) & tmask == 0 && g.bk.turn_map[0][h] != no)
            .collect();
        // river-valid: turn-valid, not colliding the river card.
        let valid_river: Vec<usize> = valid_turn.iter().copied()
            .filter(|&h| hcards(h) & rmask == 0 && g.bk.river_map[0][0][h] != no)
            .collect();
        Anchor {
            nb: g.bk.nb_flop.max(g.bk.nb_turn).max(g.bk.nb_river),
            np: g.live as usize,
            nh,
            fmap: g.bk.flop_map.clone(),
            tmap: g.bk.turn_map[0].clone(),
            rmap: g.bk.river_map[0][0].clone(),
            ftbl: clone_tables(&g.bk.flop_tables),
            ttbl: clone_tables(&g.bk.turn_tables[0]),
            rtbl: clone_tables(&g.bk.river_tables[0][0]),
            starting_pot: g.tree.starting_pot,
            rake_rate: g.tree.rake_rate as f32,
            rake_cap: g.tree.rake_cap as f32,
            valid,
            valid_turn,
            valid_river,
            nc: table.num_combinations as f32,
            node_regret: std::cell::RefCell::new(vec![0.0; g.tree.num_nodes()]),
            node_action_cv: std::cell::RefCell::new(vec![Vec::new(); g.tree.num_nodes()]),
        }
    }

    /// hands live at a street = not colliding the board dealt so far (mirrors
    /// DCFR's per-street reach: river-colliders are live through the turn).
    #[inline]
    fn valid_at(&self, bs: u8) -> &[usize] {
        use solver_core::tree::action::BoardState;
        if bs == BoardState::Flop as u8 { &self.valid }
        else if bs == BoardState::Turn as u8 { &self.valid_turn }
        else { &self.valid_river }
    }

    #[inline]
    fn street_map(&self, bs: u8) -> &[u16] {
        use solver_core::tree::action::BoardState;
        if bs == BoardState::Flop as u8 { &self.fmap }
        else if bs == BoardState::Turn as u8 { &self.tmap }
        else { &self.rmap }
    }
    #[inline]
    fn street_tbl(&self, bs: u8) -> &solver_core::solver::bucketed_showdown::BucketedRunoutTables {
        use solver_core::tree::action::BoardState;
        if bs == BoardState::Flop as u8 { &self.ftbl }
        else if bs == BoardState::Turn as u8 { &self.ttbl }
        else { &self.rtbl }
    }

    /// total exploitability = Σ_i (BR_i − onpolicy_i), σ in canonical
    /// node-major layout [node*MAX_NA*nb + a*nb + b] (MAX_NA = max_na arg).
    fn exploitability(&self, tree: &FlatTree, sigma: &[f32], max_na: usize) -> f32 {
        let mut total = 0.0f32;
        for i in 0..self.np {
            // opponents' counterfactual reach at the root: sum-1 per opponent
            // over valid hands (i's slot carried but unused for the gap).
            let w = 1.0 / self.valid.len() as f32;
            let mut reach = vec![vec![0.0f32; self.nh]; self.np];
            for p in 0..self.np {
                for &h in &self.valid {
                    reach[p][h] = w;
                }
            }
            let v_br = self.walk(tree, 0, i as u8, true, sigma, max_na, &reach);
            let v_on = self.walk(tree, 0, i as u8, false, sigma, max_na, &reach);
            // V_i = mean over valid i-hands (uniform prior).
            let mut gap = 0.0f32;
            for &h in &self.valid {
                gap += (v_br[h] - v_on[h]) * w;
            }
            if std::env::var("MC_VERBOSE").is_ok() {
                eprintln!("    evaluee {i}: BR-gap {:.4e}", gap);
            }
            total += gap.max(0.0);
        }
        total
    }

    fn walk(&self, tree: &FlatTree, node: usize, i: u8, br: bool, sigma: &[f32], max_na: usize, reach: &[Vec<f32>]) -> Vec<f32> {
        let n = &tree.nodes[node];
        if n.is_terminal() {
            return self.terminal(tree, node, i, reach);
        }
        let kids: Vec<u32> = tree.node_children(node).to_vec();
        if n.is_chance() {
            // 1×1: single child; remove hands conflicting the dealt card. The
            // hand identity persists (no bucket-transition matrix). Card-removal
            // is captured by the next street's map (NO_BUCKET handled at showdown
            // + here we simply carry reach forward — the map zeroes conflicts).
            return self.walk(tree, kids[0] as usize, i, br, sigma, max_na, reach);
        }
        let player = n.player_id;
        let na = kids.len();
        let bs = n.board_state;
        let map = self.street_map(bs);
        let vh = self.valid_at(bs); // per-street live hands
        let off = node * max_na * self.nb;
        if player == i {
            let cvs: Vec<Vec<f32>> =
                kids.iter().map(|&c| self.walk(tree, c as usize, i, br, sigma, max_na, reach)).collect();
            let mut v = vec![0.0f32; self.nh];
            if br {
                // per-bucket argmax (uniform prior over hands in the bucket).
                let mut best_a = vec![0usize; self.nb];
                let mut best_v = vec![f32::NEG_INFINITY; self.nb];
                let mut sum = vec![vec![0.0f32; self.nb]; na];
                for &h in vh {
                    let b = map[h] as usize;
                    if b >= self.nb { continue; }
                    for a in 0..na { sum[a][b] += cvs[a][h]; }
                }
                for b in 0..self.nb {
                    for a in 0..na {
                        if sum[a][b] > best_v[b] { best_v[b] = sum[a][b]; best_a[b] = a; }
                    }
                }
                for &h in vh {
                    let b = map[h] as usize;
                    if b >= self.nb { continue; }
                    v[h] = cvs[best_a[b]][h];
                }
            } else {
                // per-bucket aggregates for v + the one-step regret localization.
                let mut sum_ab = vec![vec![0.0f32; self.nb]; na];
                for &h in vh {
                    let b = map[h] as usize;
                    if b >= self.nb { continue; }
                    for a in 0..na { sum_ab[a][b] += cvs[a][h]; }
                }
                let mut node_reg = 0.0f32;
                for b in 0..self.nb {
                    let onp: f32 = (0..na).map(|a| sigma[off + a * self.nb + b] * sum_ab[a][b]).sum();
                    let mx = (0..na).map(|a| sum_ab[a][b]).fold(f32::NEG_INFINITY, f32::max);
                    node_reg += (mx - onp).max(0.0);
                }
                self.node_regret.borrow_mut()[node] += node_reg;
                // aggregated per-action cv (sum over buckets+valid hands) — only
                // the evaluee-0 pass is inspected, overwrite is fine.
                let agg: Vec<f32> = (0..na).map(|a| (0..self.nb).map(|b| sum_ab[a][b]).sum()).collect();
                self.node_action_cv.borrow_mut()[node] = agg;
                for &h in vh {
                    let b = map[h] as usize;
                    if b >= self.nb { continue; }
                    let mut acc = 0.0;
                    for a in 0..na { acc += sigma[off + a * self.nb + b] * cvs[a][h]; }
                    v[h] = acc;
                }
            }
            v
        } else {
            // opponent: split this opponent's reach by σ into each child, recurse,
            // SUM the evaluee values (factored convention — reach carries σ).
            let p = player as usize;
            let mut v = vec![0.0f32; self.nh];
            for a in 0..na {
                let mut r2 = reach.to_vec();
                for &h in vh {
                    let b = map[h] as usize;
                    if b >= self.nb { r2[p][h] = 0.0; }
                    else { r2[p][h] *= sigma[off + a * self.nb + b]; }
                }
                let cv = self.walk(tree, kids[a] as usize, i, br, sigma, max_na, &r2);
                for &h in vh { v[h] += cv[h]; }
            }
            v
        }
    }

    /// evaluee value per i-hand at a terminal via the bucketed showdown.
    fn terminal(&self, tree: &FlatTree, node: usize, i: u8, reach: &[Vec<f32>]) -> Vec<f32> {
        use solver_core::solver::bucketed_showdown::bucketed_showdown_cfv_design1_collapsed;
        let bs = tree.nodes[node].board_state;
        let map = self.street_map(bs);
        let tbl = self.street_tbl(bs);
        let vh = self.valid_at(bs); // per-street live hands
        let fold_mask = tree.get_folded_mask(node);
        let np = self.np;
        // opponents' bucket reach (all p≠i), reduced from hand reach.
        let mut bucket_reach: Vec<Vec<f32>> = vec![vec![0.0f32; self.nb]; np - 1];
        let mut oi = 0;
        for p in 0..np {
            if p == i as usize { continue; }
            for &h in vh {
                let b = map[h] as usize;
                if b < self.nb { bucket_reach[oi][b] += reach[p][h]; }
            }
            oi += 1;
        }
        let views: Vec<&[f32]> = bucket_reach.iter().map(|v| v.as_slice()).collect();
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(node, p as u8)).collect();
        let cfv = bucketed_showdown_cfv_design1_collapsed(
            &views, tbl, &contribs, fold_mask, i as usize, np as u8,
            self.starting_pot, self.rake_rate, self.rake_cap, true,
        );
        // v[i-hand] = cfv at i-hand's bucket on this street, in DCFR's units
        // (DCFR divides the showdown by num_combinations — match it so the
        // exploitability magnitude is interpretable and side-1 reads true ~0).
        let inv_nc = if self.nc > 0.0 { 1.0 / self.nc } else { 1.0 };
        let mut v = vec![0.0f32; self.nh];
        for &h in vh {
            let b = map[h] as usize;
            if b < self.nb { v[h] = cfv[b] * inv_nc; }
        }
        v
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

/// Two-sided anchor validation: DCFR avg at increasing iters. The anchor passes
/// iff exploitability is HIGH at low iters and FALLS toward ~0 as DCFR converges
/// (converged ~0 = no over-report / normalization-seam OK; high-and-falling =
/// catches real exploitation / no under-report). One-sided would pass a liar.
fn anchor_validation(nb: usize, nt: usize, nr: usize) {
    use solver_core::tree::flat::MAX_NA_POSTFLOP;
    let live: u8 = std::env::var("MC_LIVE").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let g = build_shrunk(live, nb, nt, nr);
    let anchor = Anchor::new(&g);
    // MC_MCCONV: convergence of the MCCFR AVERAGE (per-street engine) measured by
    // the trusted anchor. Two-way gate: identity → ~0 (per-street implemented
    // right), nb=6 → DCFR's per-family residual (same game). Run BOTH before any
    // speed comparison (same-residual proves same game; speed is the verdict).
    if std::env::var("MC_MCCONV").is_ok() {
        let nn = g.tree.num_nodes();
        let nb_a = anchor.nb;
        let mut m = Mccfr::new(&g);
        println!("MCCFR CONVERGENCE — true-BR exploit of the MCCFR average, live-{live}, nb={nb}");
        println!("{:>10} {:>18}", "traj", "true-BR exploit");
        let batch = 8192usize;
        let mut total = 0usize;
        for &target in &[8192usize, 65536, 262144, 1048576, 4194304] {
            while total < target { m.run_iter(&g.tree, batch); total += batch; }
            let mut sigma = vec![0.0f32; nn * MAX_NA_POSTFLOP * nb_a];
            for node in 0..nn {
                if !g.tree.nodes[node].is_player() { continue; }
                let local = m.node_local[node] as usize;
                let na = g.tree.nodes[node].num_children as usize;
                let off = node * MAX_NA_POSTFLOP * nb_a;
                let use_current = std::env::var("MC_CURRENT").is_ok();
                for bb in 0..nb_a {
                    let base = (local * m.nb + bb) * m.max_na;
                    let val = |a: usize| -> f32 {
                        if use_current { m.regret[base + a].max(0.0) } else { m.cum[base + a] }
                    };
                    let sum: f32 = (0..na).map(val).sum();
                    if sum > 0.0 {
                        for a in 0..na { sigma[off + a * nb_a + bb] = val(a) / sum; }
                    } else {
                        let u = 1.0 / na as f32;
                        for a in 0..na { sigma[off + a * nb_a + bb] = u; }
                    }
                }
            }
            let expl = anchor.exploitability(&g.tree, &sigma, MAX_NA_POSTFLOP);
            // SELF-CONSISTENT convergence signal (the discriminator): the engine's
            // own CFR+ regret bound. maxR/T → 0 ⇒ engine converges to SOME Nash by
            // its OWN values (so a high anchor reading = terminal/value mismatch).
            // maxR/T FLAT ⇒ regrets grow linearly ⇒ dynamics don't converge to any
            // equilibrium (terminal reconciliation would be premature).
            let maxr = m.regret.iter().cloned().fold(0.0f32, f32::max);
            let meanr: f64 = m.regret.iter().map(|&r| r.max(0.0) as f64).sum::<f64>() / m.regret.len() as f64;
            println!("{total:>10} {expl:>18.5e}   maxR {maxr:>11.3e}  maxR/T {:>10.4e}  meanR/T {:>10.4e}",
                maxr as f64 / total as f64, meanr / total as f64);
        }
        return;
    }
    println!("ANCHOR VALIDATION — true-BR exploitability of the DCFR average, live-{live}, nb={nb}");
    println!("two-sided test: under-converged → HIGH & FALLING (catches exploitation, no under-report);");
    println!("converged → ~0 (no over-report — the normalize_opponent_reach-seam test). pass = both.\n");
    println!("{:>6} {:>18}", "DCFR iters", "true-BR exploit");
    // MC_ROOTCMP: is the anchor's value model = DCFR's? Compare anchor on-policy
    // root value V_i (per hand) vs DCFR's root_cfv_from_avg. Constant ratio ⇒
    // value model matches (floor is not a value-model bug); varying ratio ⇒
    // value-model divergence (the bug).
    if std::env::var("MC_ROOTCMP").is_ok() {
        use solver_core::tree::flat::MAX_NA_POSTFLOP;
        let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        s.run(&g.tree, &g.game, &g.bk, 1024);
        let sigma = s.average_strategy_canonical(&g.tree, &g.bk);
        let dcfr_root = s.root_cfv_from_avg(&g.tree, &g.game, &g.bk); // [np][nh]
        let nv = anchor.valid.len();
        let w = 1.0 / nv as f32;
        let mut reach = vec![vec![0.0f32; anchor.nh]; anchor.np];
        for p in 0..anchor.np { for &h in &anchor.valid { reach[p][h] = w; } }
        let v_on = anchor.walk(&g.tree, 0, 0, false, &sigma, MAX_NA_POSTFLOP, &reach);
        println!("ROOT VALUE CMP (evaluee 0): anchor on-policy vs DCFR root_cfv_from_avg");
        // card-removal diagnostic: print each hand's cards + the board/runout so
        // the sign-flipped vs exact contrast can be read (does the wrong hand
        // share a card with board/runout?).
        let cs = |c: u8| -> String {
            let r = "23456789TJQKA".as_bytes()[(c >> 2) as usize] as char;
            let s = "cdhs".as_bytes()[(c & 3) as usize] as char;
            format!("{r}{s}")
        };
        let tbl_hc = &g.game.table().hand_cards;
        let turn = g.game.table().remaining_deck.clone();
        let tc = turn[0];
        let rc = g.game.table().river_decks[tc as usize][0];
        println!("board: turn {} river {}  (flop = canonical[0])", cs(tc), cs(rc));
        println!("{:>6} {:>7} {:>14} {:>14} {:>10}", "hand", "cards", "anchor", "DCFR", "ratio");
        let mut shown = 0;
        for &h in anchor.valid.iter() {
            let (a, d) = (v_on[h], dcfr_root[0][h]);
            if d.abs() > 1e-6 && shown < 16 {
                let cards = format!("{}{}", cs(tbl_hc[h * 2]), cs(tbl_hc[h * 2 + 1]));
                let flag = if (a / d) < 0.0 { " <-SIGN-FLIP" } else if (a - d).abs() < 1e-3 { " <-exact" } else { "" };
                println!("{h:>6} {cards:>7} {a:>14.5e} {d:>14.5e} {:>10.4}{flag}", a / d);
                shown += 1;
            }
        }
        return;
    }
    // MC_NODECMP: per-ACTION cv at top fold nodes, anchor vs DCFR's internal
    // bottom_up cfv (turn_cfv[child]). Pins WHICH action's cv carries the
    // non-cancelling (floor) error — fold vs call ratio is the tell.
    if std::env::var("MC_NODECMP").is_ok() {
        let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        s.run(&g.tree, &g.game, &g.bk, 1024);
        let sigma = s.average_strategy_canonical(&g.tree, &g.bk);
        let _ = anchor.exploitability(&g.tree, &sigma, MAX_NA_POSTFLOP); // populates node_action_cv
        let tcfv = s.debug_turn_cfv(&g.tree, &g.game, &g.bk, 0); // DCFR per-node cfv, evaluee 0
        let acv = anchor.node_action_cv.borrow();
        let nh = anchor.nh;
        println!("PER-ACTION cv: anchor vs DCFR internal (evaluee 0). fold/call ratio = the tell.");
        for &node in &[468usize, 506, 369, 329, 230] {
            let n = &g.tree.nodes[node];
            if n.player_id != 0 { continue; } // DCFR cfv is for traverser 0
            let kids = g.tree.node_children(node);
            println!(" node {node} street {} p{}:", n.board_state, n.player_id);
            for (a, &c) in kids.iter().enumerate() {
                let dcfr: f32 = anchor.valid.iter().map(|&h| tcfv[c as usize * nh + h]).sum();
                let anc = acv[node].get(a).copied().unwrap_or(0.0);
                let lbl = g.tree.nodes[c as usize].action_label;
                let term = g.tree.nodes[c as usize].is_terminal();
                println!("   a{a} (label {lbl}{}) anchor {anc:>11.3e}  DCFR {dcfr:>11.3e}  ratio {:>8.4}",
                    if term { ",T" } else { "" }, anc / dcfr);
            }
        }
        return;
    }
    let iter_list: Vec<u32> = if let Ok(s) = std::env::var("MC_ITERS") {
        s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
    } else if std::env::var("MC_FAST").is_ok() { vec![256] } else { vec![1, 64, 1024] };
    for iters in iter_list {
        let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        s.run(&g.tree, &g.game, &g.bk, iters);
        let sigma = s.average_strategy_canonical(&g.tree, &g.bk);
        let expl = anchor.exploitability(&g.tree, &sigma, MAX_NA_POSTFLOP);
        println!("{iters:>6} {expl:>18.5e}");
    }
    if std::env::var("MC_NOLOC").is_ok() { return; }
    // ── FLOOR LOCALIZATION: where does the BR diverge on converged DCFR? ──
    let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
    s.set_terminal_design(TerminalDesign::Design1Collapsed);
    s.run(&g.tree, &g.game, &g.bk, 2048);
    let sigma = s.average_strategy_canonical(&g.tree, &g.bk);
    anchor.node_regret.borrow_mut().iter_mut().for_each(|x| *x = 0.0);
    let _ = anchor.exploitability(&g.tree, &sigma, MAX_NA_POSTFLOP);
    let nr = anchor.node_regret.borrow().clone();
    let mut idx: Vec<usize> = (0..nr.len()).filter(|&i| nr[i] > 1e-3).collect();
    idx.sort_by(|&a, &b| nr[b].partial_cmp(&nr[a]).unwrap());
    println!("\nFLOOR LOCALIZATION (converged DCFR) — top one-step-regret nodes:");
    println!("(if concentrated at a node TYPE — street/terminal-adjacent/na — that's the bug locus)");
    let acv = anchor.node_action_cv.borrow();
    for &node in idx.iter().take(10) {
        let n = &g.tree.nodes[node];
        let labels: Vec<u8> = g.tree.node_children(node).iter().map(|&c| g.tree.nodes[c as usize].action_label).collect();
        // per-action cv + which child is a terminal (fold→forfeit) with its contribs.
        let kids = g.tree.node_children(node);
        let kid_info: Vec<String> = kids.iter().map(|&c| {
            let cn = &g.tree.nodes[c as usize];
            if cn.is_terminal() {
                format!("T(c_p{}={})", n.player_id, g.tree.get_contribution(c as usize, n.player_id))
            } else { "·".into() }
        }).collect();
        let cv: Vec<String> = acv[node].iter().map(|v| format!("{v:+.2e}")).collect();
        println!("  node {node:>6} street {} p{} labels {labels:?}: cv [{}] kids [{}]",
            n.board_state, n.player_id, cv.join(", "), kid_info.join(", "));
    }
    let nonzero: usize = nr.iter().filter(|&&r| r > 1e-3).count();
    println!("  ({nonzero} nodes with regret > 1e-3; total = {:.4e})", nr.iter().sum::<f32>());
}

fn main() {
    let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let nt: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let nr: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    if std::env::var("MC_ANCHOR").is_ok() {
        anchor_validation(nb, nt, nr);
        return;
    }
    let iters: u32 = std::env::var("MC_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);

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
