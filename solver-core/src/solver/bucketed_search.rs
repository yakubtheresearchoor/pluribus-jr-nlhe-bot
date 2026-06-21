//! Pluribus-faithful real-time search GameSpec for MULTIWAY flop spots.
//!
//! The search runs `CpuMccfr` on the flop with **lossless current-round
//! abstraction** (real hands, full `nh`), QRE λ, and the k=4 multi-continuation
//! — exactly the production decision setup. The one change vs the exact game is
//! the DEPTH-LIMIT leaf value: at the turn-entry boundary the searcher does NOT
//! recurse into (and re-enumerate the O(nh^np) showdown of) the next streets.
//! Instead it values the leaf via the **bucketed + Monte-Carlo-sampled**
//! blueprint continuation:
//!
//!   1. reduce each opponent's per-hand reach → per-bucket reach (flop_map),
//!   2. `bucketed_showdown_cfv_design1_collapsed_sampled` → per-bucket CFV
//!      (sampled over opponent bucket-tuples ⇒ FLAT in player count),
//!   3. expand per-bucket CFV back to per-hand (lossless traverser side).
//!
//! This is the lever that makes 6-max real-time search tractable: the showdown
//! cost moves out of the searched tree and into the sampled continuation, which
//! is flat in `np` rather than `O(nh^np)`. The current round stays lossless, so
//! the decision being made suffers no card abstraction.
//!
//! Flop folds (someone folds before the turn) are still exact ordinary
//! terminals — delegated to the wrapped `FlopStartGame`. Only the turn-entry
//! depth-limit nodes route through `evaluate_continuation`.

use crate::solver::bucketed_flop_cfr::{FlopBucketing, NO_BUCKET};
use crate::solver::bucketed_showdown::bucketed_showdown_cfv_design1_collapsed_sampled;
use crate::solver::flop_start_game::FlopStartGame;
use crate::solver::game::GameSpec;
use crate::tree::flat::FlatTree;

/// Which street the search is rooted on — selects the bucket map + runout
/// tables the depth-limit continuation integrates over:
///   - `Flop`        → `flop_map` / `flop_tables` (turn+river runout)
///   - `Turn(ti)`    → `turn_map[ti]` / `turn_tables[ti]` (river runout)
///   - `River(ti,ri)`→ `river_map[ti][ri]` / `river_tables[ti][ri]` (showdown)
/// The reach→bucket→sampled-showdown→expand pipeline is identical across
/// streets; only the (map, tables, nb) triple changes. This is what makes the
/// per-street Pluribus search reuse one continuation implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContStreet {
    Flop,
    Turn(usize),
    River(usize, usize),
}

pub struct BucketedContinuationGame<'a> {
    /// Exact flop game (BORROWED — e.g. a loaded blueprint's `bp.game`) —
    /// supplies num_hands (lossless), initial weights, the flop fold terminals.
    inner: &'a FlopStartGame,
    /// Bucketing (BORROWED — e.g. `bp.bk`): per-street `*_map[h]` = bucket of
    /// hand h (or `NO_BUCKET`), `*_tables` = per-bucket runout showdown fractions.
    bucketing: &'a FlopBucketing,
    /// Rooted street — selects which map/tables the continuation integrates.
    street: ContStreet,
    np: u8,
    /// MC samples for the collapsed terminal. 0 ⇒ exact enumeration (only sane
    /// at small nb / low np; multiway must sample).
    sample_m: u32,
    rng_seed: u64,
}

impl<'a> BucketedContinuationGame<'a> {
    /// Flop-rooted search (the original entry point — unchanged behavior).
    pub fn new(
        inner: &'a FlopStartGame,
        bucketing: &'a FlopBucketing,
        sample_m: u32,
        rng_seed: u64,
    ) -> Self {
        Self::new_street(inner, bucketing, ContStreet::Flop, sample_m, rng_seed)
    }

    /// Per-street search: root on `street` (Flop/Turn/River); the depth-limit
    /// continuation integrates that street's runout tables.
    pub fn new_street(
        inner: &'a FlopStartGame,
        bucketing: &'a FlopBucketing,
        street: ContStreet,
        sample_m: u32,
        rng_seed: u64,
    ) -> Self {
        let np = inner.table().num_players;
        BucketedContinuationGame { inner, bucketing, street, np, sample_m, rng_seed }
    }

    /// Override the player count (default = the inner game's np). The pairwise
    /// bucket-vs-bucket showdown tables are np-independent, so the same blueprint
    /// continuation values a subgame with FEWER live players (e.g. after a
    /// mid-hand fold dropped a multiway pot from np to L). The search subgame
    /// tree must be built with this same L.
    pub fn set_player_count(&mut self, np: u8) {
        self.np = np;
    }
}

impl<'a> GameSpec for BucketedContinuationGame<'a> {
    fn num_hands(&self, player: u8) -> usize {
        self.inner.num_hands(player)
    }

    fn initial_weight(&self, player: u8) -> Vec<f32> {
        let mut w = self.inner.initial_weight(player);
        // Turn/River: zero hands that collide with the dealt board card(s) —
        // the flop game's reach still carries them. `*_map[h] == NO_BUCKET`
        // flags a board conflict. NOTE (v1): this filters by board only; it
        // does NOT Bayes-update the range for the prior street's betting (the
        // Pluribus reach-prior refinement — a documented follow-up). The search
        // roots on the prior-street entering range minus board conflicts.
        let mask: Option<&Vec<u16>> = match self.street {
            ContStreet::Flop => None,
            ContStreet::Turn(ti) => Some(&self.bucketing.turn_map[ti]),
            ContStreet::River(ti, ri) => Some(&self.bucketing.river_map[ti][ri]),
        };
        if let Some(map) = mask {
            for (h, wh) in w.iter_mut().enumerate() {
                if map[h] == NO_BUCKET {
                    *wh = 0.0;
                }
            }
        }
        w
    }

    fn evaluate_terminal(
        &self,
        traverser: u8,
        node_idx: usize,
        tree: &FlatTree,
        cfreach: &[Vec<f32>],
    ) -> Vec<f32> {
        // Flop folds (no showdown reached) — exact, cheap.
        self.inner.evaluate_terminal(traverser, node_idx, tree, cfreach)
    }

    fn evaluate_continuation(
        &self,
        traverser: u8,
        node_idx: usize,
        tree: &FlatTree,
        cfreach: &[Vec<f32>],
    ) -> Vec<f32> {
        let nh = self.inner.table().num_valid;
        let np = self.np as usize;
        let num_opp = np - 1;
        // Select the rooted street's bucket map + runout tables.
        let (nb, map, tables) = match self.street {
            ContStreet::Flop => (
                self.bucketing.nb_flop,
                &self.bucketing.flop_map,
                &self.bucketing.flop_tables,
            ),
            ContStreet::Turn(ti) => (
                self.bucketing.nb_turn,
                &self.bucketing.turn_map[ti],
                &self.bucketing.turn_tables[ti],
            ),
            ContStreet::River(ti, ri) => (
                self.bucketing.nb_river,
                &self.bucketing.river_map[ti][ri],
                &self.bucketing.river_tables[ti][ri],
            ),
        };
        let fold_mask = tree.get_folded_mask(node_idx);

        // (1) per-opponent reach over hands → over flop buckets.
        let mut bucket_reach: Vec<Vec<f32>> = vec![vec![0.0f32; nb]; num_opp];
        let mut oi = 0usize;
        for p in 0..np {
            if p == traverser as usize {
                continue;
            }
            let r = &cfreach[p];
            for h in 0..nh {
                let b = map[h];
                if b != NO_BUCKET {
                    bucket_reach[oi][b as usize] += r[h];
                }
            }
            oi += 1;
        }
        let views: Vec<&[f32]> = bucket_reach.iter().map(|v| v.as_slice()).collect();

        let contributions: Vec<i32> =
            (0..np).map(|p| tree.get_contribution(node_idx, p as u8)).collect();

        // (2) bucketed + MC-sampled showdown over the runout — per-bucket CFV.
        // Per-node seed ⇒ an independent deterministic stream at each leaf.
        let term_seed =
            self.rng_seed ^ (node_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let cfv_bucket = bucketed_showdown_cfv_design1_collapsed_sampled(
            &views,
            tables,
            &contributions,
            fold_mask,
            traverser as usize,
            self.np,
            tree.starting_pot,
            tree.rake_rate as f32,
            tree.rake_cap as f32,
            true,
            self.sample_m,
            term_seed,
        );

        // (3) expand per-bucket CFV → per-hand (lossless traverser), same /nc
        // normalization as the exact walk.
        let nc = self.inner.table().num_combinations as f32;
        let mut cfv = vec![0.0f32; nh];
        for h in 0..nh {
            let b = map[h];
            if b != NO_BUCKET {
                let v = cfv_bucket[b as usize];
                cfv[h] = if nc > 0.0 { v / nc } else { v };
            }
        }
        cfv
    }

    fn chance_probability(&self, outcome: usize, hand: usize) -> f32 {
        self.inner.chance_probability(outcome, hand)
    }

    fn num_combinations(&self) -> f64 {
        self.inner.num_combinations()
    }

    fn num_chance_outcomes(&self) -> usize {
        // Not crossed in a depth-limited flop search (the turn-deal is the
        // boundary), but delegate for correctness if used elsewhere.
        self.inner.num_chance_outcomes()
    }

    fn set_chance_outcome(&self, outcome: usize) {
        self.inner.set_chance_outcome(outcome)
    }

    fn clear_chance_outcome(&self) {
        self.inner.clear_chance_outcome()
    }
}
