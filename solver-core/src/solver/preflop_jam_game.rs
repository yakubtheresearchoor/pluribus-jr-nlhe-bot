//! Real-time preflop JAM-SUBGAME (Option A): re-solve a low-SPR heads-up preflop
//! decision with the rich raise menu PLUS an explicit all-in, which the lean na=8
//! production blueprint omits (see `blueprint::build_conn_preflop_tree`). The whole
//! point is the high-SPR jam the blueprint can't represent.
//!
//! Sound BECAUSE of the SPR gate: a called jam goes straight to a 5-card runout
//! (pure all-in equity — no flop continuation needed), and at LOW SPR even a "call
//! to see a flop" is ≈ all-in equity (little postflop maneuvering left). So EVERY
//! non-fold leaf of this subgame is valued by all-in equity; folds by the chip
//! delta. The caller must enforce the SPR gate (this proxy is invalid at high SPR).
//!
//! Class level (169): reuses `preflop_allin_equity` (HU equity table, combo
//! counts) + `preflop_terminal::build_class_blocking_matrix`. HU only (np=2) —
//! jams in 4-bet/5-bet pots are HU; multiway falls back to the blueprint upstream.
//!
//! Sign convention mirrors the rest of the preflop code (showdown_oracle / fold
//! terminal, see `preflop_terminal`): `investment = starting_pot/np + c[t]`,
//! traverser value = (equity·total_pot_after_rake) − investment, zero-sum at
//! no-rake. Dead money (starting_pot) is shared investment-wise.

use crate::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use crate::solver::game::GameSpec as GameSpecTrait;
use crate::solver::mccfr::CpuMccfr;
use crate::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, BoardState};
use crate::tree::action::GameSpec as GameConfig;
use crate::tree::builder::build_tree_preflop_only;
use crate::tree::flat::FlatTree;

const NC: usize = NUM_PREFLOP_CLASSES; // 169

/// Per-decision search settings for the jam subgame.
#[derive(Clone)]
pub struct PreflopJamCfg {
    pub iters: u32,
    pub lambda: f32,     // hero QRE sharpness (logit). 0 = plain CFR average.
    pub opp_lambda: f32, // opponent QRE sharpness.
    pub plk: usize,      // Pluribus continuation variants (1 = off).
    pub nraises: usize,  // rich pot-relative raise sizes before the explicit all-in.
}

impl Default for PreflopJamCfg {
    fn default() -> Self {
        PreflopJamCfg { iters: 256, lambda: 0.0, opp_lambda: 0.0, plk: 1, nraises: 6 }
    }
}

/// Result of a jam-subgame solve: the hero's per-class strategy at the root and
/// the action menu (label + hero's TOTAL contribution after each action).
pub struct PreflopJamResult {
    pub root: usize,
    pub actions: Vec<(u8, i32)>,   // (action_label, hero total contribution)
    pub strategy: Vec<Vec<f32>>,   // [na][169] class-level probabilities
}

/// HU class-level preflop game with chip-delta folds, all-in-equity all-in leaves,
/// and (when `cont` is supplied) v_flop CONTINUATION matrices at sees-a-flop leaves
/// — the difference between the cheap jam-subgame (A, `cont=None` ⇒ all-in-equity
/// proxy everywhere, valid only at low SPR) and the rigorous full re-solve (B/C,
/// `cont=Some` ⇒ real postflop EV at any SPR).
struct PreflopJamGame<'a> {
    reach: [Vec<f32>; 2],  // combo-count-weighted per-class reach, [hero, opp]
    equity: &'a [f32],     // 169×169 HU all-in equity (row = acting class, col = opp)
    blocking: &'a [f32],   // 169×169 non-blocking fractions
    /// Per-(commit, total_pot) v_flop value matrix: `cont[&(commit,pot)][h*169+oc]`
    /// = NET postflop chip-EV of acting class `h` vs opp class `oc` entering the
    /// flop at that seam (already net of investment, same units as the all-in
    /// branch). `None`, or a missing key ⇒ fall back to all-in equity (A proxy).
    cont: Option<&'a std::collections::HashMap<(i32, i32), Vec<f32>>>,
    stack: i32, // for all-in detection at a flop-entry leaf (commit >= stack)
    rake_rate: f32,
    rake_cap: f32,
}

impl<'a> PreflopJamGame<'a> {
    /// Value the leaf at `node_idx` for `traverser`: fold terminal → chip delta;
    /// otherwise (both live) → all-in equity showdown. Used for BOTH terminals
    /// and depth-limited flop-entry leaves (robust to which CFR path calls it).
    fn value_leaf(&self, traverser: u8, node_idx: usize, tree: &FlatTree, cfreach: &[Vec<f32>]) -> Vec<f32> {
        let t = traverser as usize;
        let o = 1 - t;
        let fold_mask = tree.get_folded_mask(node_idx);
        let c = [tree.get_contribution(node_idx, 0), tree.get_contribution(node_idx, 1)];
        let sp = tree.starting_pot;
        let total_pot = sp + c[0] + c[1];
        let investment = (sp as f32) / 2.0 + c[t] as f32;
        let opp_reach = &cfreach[o];
        let blk = self.blocking;
        let mut cfv = vec![0.0f32; NC];

        let t_folded = (fold_mask >> t) & 1 == 1;
        let o_folded = (fold_mask >> o) & 1 == 1;

        if t_folded || o_folded {
            // Fold terminal: constant chip delta × compatible opp reach mass.
            // (no flop ⇒ no rake.)
            let delta = if o_folded {
                total_pot as f32 - investment // t wins
            } else {
                -investment // t folded
            };
            for h in 0..NC {
                let mut mass = 0.0f32;
                let row = h * NC;
                for oc in 0..NC {
                    mass += opp_reach[oc] * blk[row + oc];
                }
                cfv[h] = delta * mass;
            }
        } else {
            // Flop-entry leaf. SEES A FLOP (chips behind) → real v_flop continuation
            // if supplied (B/C). TRUE all-in (commit ≥ stack) OR no continuation →
            // all-in equity (exact for the called jam; A's low-SPR proxy otherwise).
            let commit = c[0].min(c[1]); // HU: matched
            let all_in = self.stack > 0 && commit >= self.stack;
            if !all_in {
                if let Some(v) = self.cont.and_then(|m| m.get(&(commit, total_pot))) {
                    for h in 0..NC {
                        let row = h * NC;
                        let mut acc = 0.0f32;
                        for oc in 0..NC {
                            let w = opp_reach[oc] * blk[row + oc];
                            if w != 0.0 {
                                acc += w * v[row + oc];
                            }
                        }
                        cfv[h] = acc;
                    }
                    return cfv;
                }
            }
            // All-in equity (rake applies — the flop runs out).
            let rake = (self.rake_rate * total_pot as f32).min(self.rake_cap);
            let pot_eff = total_pot as f32 - rake;
            for h in 0..NC {
                let row = h * NC;
                let mut acc = 0.0f32;
                for oc in 0..NC {
                    let w = opp_reach[oc] * blk[row + oc];
                    if w == 0.0 {
                        continue;
                    }
                    acc += w * (self.equity[row + oc] * pot_eff - investment);
                }
                cfv[h] = acc;
            }
        }
        cfv
    }
}

impl<'a> GameSpecTrait for PreflopJamGame<'a> {
    fn num_hands(&self, _player: u8) -> usize {
        NC
    }
    fn initial_weight(&self, player: u8) -> Vec<f32> {
        self.reach[player as usize].clone()
    }
    fn evaluate_terminal(&self, traverser: u8, node_idx: usize, tree: &FlatTree, cfreach: &[Vec<f32>]) -> Vec<f32> {
        self.value_leaf(traverser, node_idx, tree, cfreach)
    }
    fn evaluate_continuation(&self, traverser: u8, node_idx: usize, tree: &FlatTree, cfreach: &[Vec<f32>]) -> Vec<f32> {
        self.value_leaf(traverser, node_idx, tree, cfreach)
    }
    fn chance_probability(&self, _outcome: usize, _hand: usize) -> f32 {
        1.0
    }
}

/// Build the rich preflop menu used by the jam search: the production raise ladder
/// (`build_conn_preflop_tree`'s sizes) PLUS an explicit all-in.
fn jam_bets(nraises: usize) -> BetSizeOptions {
    const RICH: [f64; 13] = [1.0, 0.5, 0.75, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 11.0];
    let mut raise: Vec<BetSize> =
        RICH.iter().take(nraises).map(|&f| BetSize::PotRelative(f)).collect();
    raise.push(BetSize::AllIn);
    BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise }
}

/// Construct the synthetic HU preflop tree rooted at the current decision state.
/// `c_hero`/`c_opp` = each player's current preflop contribution; `dead` = chips
/// already in the pot from folded seats/antes; `stack` = starting total stack.
/// Hero is player 0 and (by `c_hero < c_opp`) faces a bet, so the root is hero's.
fn build_jam_tree(c_hero: i32, c_opp: i32, dead: i32, stack: i32, nraises: usize) -> Result<FlatTree, String> {
    let g: GameConfig = production_game_v1();
    let cfg = crate::tree::action::TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: dead,
        starting_stacks: vec![stack - c_hero, stack - c_opp],
        initial_contributions: vec![c_hero, c_opp],
        rake_rate: g.rake_rate,
        rake_cap: g.rake_cap as f64,
        bet_sizes: jam_bets(nraises),
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(0), // hero = button = SB, acts first preflop HU
        max_bets_per_street: BetCap::all(3),
        no_open_limp: false,
        threebet_or_fold: false, // MUST be false: we WANT the jam available
    };
    build_tree_preflop_only(&cfg)
}

/// RIGOROUS full preflop re-solve (B/C): re-solve the HU preflop decision with the
/// rich menu + all-in, depth-limited at the flop. `cont` supplies per-(commit,pot)
/// v_flop value matrices for sees-a-flop leaves (None ⇒ the cheap A proxy = all-in
/// equity everywhere, valid only at low SPR). `*_reach` are per-class continuing
/// weights (this fn applies the combo counts). `equity`/`blocking` are 169×169.
/// Returns `None` if the synthetic state doesn't produce a hero-to-act root.
#[allow(clippy::too_many_arguments)]
pub fn solve_preflop_search(
    c_hero: i32,
    c_opp: i32,
    dead: i32,
    stack: i32,
    hero_reach: &[f32],
    opp_reach: &[f32],
    equity: &[f32],
    blocking: &[f32],
    cont: Option<&std::collections::HashMap<(i32, i32), Vec<f32>>>,
    cfg: &PreflopJamCfg,
) -> Option<PreflopJamResult> {
    if c_hero >= c_opp {
        return None; // v1 handles facing-a-bet decisions only
    }
    let tree = build_jam_tree(c_hero, c_opp, dead, stack, cfg.nraises).ok()?;

    // Combo-count-weight the reach so class aggregation = combo-level sum.
    let counts = crate::solver::preflop_allin_equity::class_combo_counts();
    let weight = |r: &[f32]| -> Vec<f32> { (0..NC).map(|i| r[i] * counts[i]).collect() };
    let game = PreflopJamGame {
        reach: [weight(hero_reach), weight(opp_reach)],
        equity,
        blocking,
        cont,
        stack,
        rake_rate: tree.rake_rate as f32,
        rake_cap: tree.rake_cap as f32,
    };

    let mut s = CpuMccfr::new(&tree, vec![NC, NC]);
    // Depth-limit (= value as all-in equity) every flop-entry leaf: a preflop-only
    // tree truncates at the flop, leaving childless chance nodes.
    let depth: Vec<usize> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_chance() && tree.node_children(n).is_empty())
        .collect();
    s.set_depth_limit(&depth);
    if cfg.plk > 1 {
        s.setup_pluribus_continuations(&tree, cfg.plk, 5.0);
    }
    s.set_lambda(vec![cfg.lambda, cfg.opp_lambda]);
    s.run(&tree, &game, cfg.iters);

    // Root = first player node; it MUST be hero (player 0).
    let root = (0..tree.num_nodes()).find(|&n| tree.nodes[n].is_player())?;
    if tree.nodes[root].player_id != 0 {
        return None;
    }
    let na = tree.nodes[root].num_children as usize;
    let children = tree.node_children(root);
    let actions: Vec<(u8, i32)> = (0..na)
        .map(|a| {
            let ch = children[a] as usize;
            (tree.nodes[ch].action_label, tree.get_contribution(ch, 0))
        })
        .collect();
    let strategy = s.get_average_strategy(root, na, NC);
    Some(PreflopJamResult { root, actions, strategy })
}

/// Option A: the cheap low-SPR jam subgame (all flop-entry leaves valued by all-in
/// equity). Thin wrapper over `solve_preflop_search` with no v_flop continuation.
#[allow(clippy::too_many_arguments)]
pub fn solve_hu_preflop_jam(
    c_hero: i32,
    c_opp: i32,
    dead: i32,
    stack: i32,
    hero_reach: &[f32],
    opp_reach: &[f32],
    equity: &[f32],
    blocking: &[f32],
    cfg: &PreflopJamCfg,
) -> Option<PreflopJamResult> {
    solve_preflop_search(
        c_hero, c_opp, dead, stack, hero_reach, opp_reach, equity, blocking, None, cfg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::preflop_class::PreflopClass;
    use crate::card::Card;
    use crate::solver::preflop_allin_equity::{class_combo_counts, class_hu_equity_table};
    use crate::solver::preflop_terminal::build_class_blocking_matrix;

    fn cls(r1: u8, s1: u8, r2: u8, s2: u8) -> usize {
        PreflopClass::from_combo((r1 * 4 + s1) as Card, (r2 * 4 + s2) as Card).index()
    }
    fn one_hot(c: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; NC];
        v[c] = 1.0;
        v
    }
    fn fold_prob(res: &PreflopJamResult, c: usize) -> f32 {
        for (a, &(label, _)) in res.actions.iter().enumerate() {
            if label == 0 {
                return res.strategy[a][c];
            }
        }
        0.0
    }

    /// Behavioral gate: at a low-SPR facing-a-bet spot, AA must (almost) never
    /// fold, while 72o vs a premium range folds heavily, and the all-in action
    /// must exist. Validates tree+reach+equity+strategy end to end.
    #[test]
    fn jam_behavior() {
        let g = production_game_v1();
        let stack = g.stack;
        let eq = class_hu_equity_table(80, 0xABCDEF); // coarse MC: behavior is noise-robust
        let blk = build_class_blocking_matrix();
        let counts = class_combo_counts();

        let aa = cls(12, 0, 12, 1);
        let o72 = cls(5, 0, 0, 1); // 7-2 offsuit
        let kk = cls(11, 0, 11, 1);
        let qq = cls(10, 0, 10, 1);
        let aks = cls(12, 0, 11, 0);

        // Low SPR: opp has put in 60% of stack, hero 25% facing it.
        let c_opp = (0.60 * stack as f32) as i32;
        let c_hero = (0.25 * stack as f32) as i32;
        let dead = stack / 20;

        let cfg = PreflopJamCfg { iters: 400, lambda: 8.0, opp_lambda: 8.0, plk: 1, nraises: 6 };

        // Opp = uniform over all classes (combo-weighted handled inside).
        let opp_uniform: Vec<f32> = counts.iter().map(|_| 1.0).collect();
        let res_aa = solve_hu_preflop_jam(
            c_hero, c_opp, dead, stack, &one_hot(aa), &opp_uniform, &eq, &blk, &cfg,
        )
        .expect("AA solve");
        // The all-in action must be present (label 5 = ALLIN).
        assert!(res_aa.actions.iter().any(|&(l, _)| l == 5), "no all-in action");
        let f_aa = fold_prob(&res_aa, aa);
        assert!(f_aa < 0.05, "AA should not fold, P(fold)={f_aa}");

        // 72o facing the same bet vs a tight premium range should fold a lot.
        let mut opp_prem = vec![0.0f32; NC];
        for &c in &[aa, kk, qq, aks] {
            opp_prem[c] = 1.0;
        }
        let res_72 = solve_hu_preflop_jam(
            c_hero, c_opp, dead, stack, &one_hot(o72), &opp_prem, &eq, &blk, &cfg,
        )
        .expect("72o solve");
        let f_72 = fold_prob(&res_72, o72);
        assert!(f_72 > 0.4, "72o vs premiums should fold, P(fold)={f_72}");
    }

    /// B/C generalization: a v_flop CONTINUATION at sees-a-flop leaves must drive
    /// the strategy. With "seeing a flop = +EV" the hero plays on (low fold); with
    /// "seeing a flop = −EV" it folds — proving `cont` is consumed (not the A proxy).
    #[test]
    fn continuation_drives_strategy() {
        let g = production_game_v1();
        let stack = g.stack;
        let eq = class_hu_equity_table(60, 0x5151);
        let blk = build_class_blocking_matrix();
        let counts = class_combo_counts();
        let opp_uniform: Vec<f32> = counts.iter().map(|_| 1.0).collect();

        // HIGH SPR: hero faces a small raise deep (A's all-in proxy is invalid here).
        let c_hero = stack / 100; // ~1bb
        let c_opp = stack / 25; // ~4bb raise
        let dead = stack / 200;
        let mid = cls(7, 0, 5, 1); // 97o — a marginal flop-playing hand

        // Flat continuation matrix for EVERY HU matched seam (commit, dead+2·commit).
        let cont_with = |val: f32| {
            let mut m: std::collections::HashMap<(i32, i32), Vec<f32>> =
                std::collections::HashMap::new();
            for commit in 0..=stack {
                m.insert((commit, dead + 2 * commit), vec![val; NC * NC]);
            }
            m
        };
        let cfg = PreflopJamCfg { iters: 300, lambda: 6.0, opp_lambda: 6.0, plk: 1, nraises: 6 };

        let good = cont_with(0.5 * stack as f32); // seeing a flop is great
        let bad = cont_with(-0.5 * stack as f32); // seeing a flop is terrible
        let res_good = solve_preflop_search(
            c_hero, c_opp, dead, stack, &one_hot(mid), &opp_uniform, &eq, &blk, Some(&good), &cfg,
        )
        .expect("good cont solve");
        let res_bad = solve_preflop_search(
            c_hero, c_opp, dead, stack, &one_hot(mid), &opp_uniform, &eq, &blk, Some(&bad), &cfg,
        )
        .expect("bad cont solve");
        let f_good = fold_prob(&res_good, mid);
        let f_bad = fold_prob(&res_bad, mid);
        assert!(
            f_good + 0.2 < f_bad,
            "continuation must drive folds: P(fold|+EV)={f_good} should be << P(fold|−EV)={f_bad}"
        );
    }
}
