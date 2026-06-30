use crate::tree::action::{Action, BetSize, BetSizeOptions, BoardState, TreeConfig};
use crate::tree::flat::{FlatNode, FlatTree, NODE_TYPE_PLAYER, NODE_TYPE_TERMINAL};

const MAX_DEPTH: usize = 64;

pub fn build_tree(config: &TreeConfig) -> Result<FlatTree, String> {
    build_tree_inner(config, None, &[])
}

/// Build a tree where specific board states use a DIFFERENT bet menu than
/// `config.bet_sizes` (per-street action abstraction). Used for NESTED solving:
/// a turn-rooted re-solve keeps the rich menu on the TURN but a coarse (or
/// check-only) menu on the RIVER continuation — the river is re-solved exactly on
/// arrival, so its in-lookahead abstraction can be cheap. `overrides` maps a
/// board state to its menu; states not listed use `config.bet_sizes`.
pub fn build_tree_with_bet_override(
    config: &TreeConfig,
    overrides: &[(BoardState, BetSizeOptions)],
) -> Result<FlatTree, String> {
    build_tree_inner(config, None, overrides)
}

/// DEPTH-LIMITED build for per-street real-time search: truncate the tree one
/// street past `config.initial_state` (its next-street chance nodes become
/// childless leaves). A flop-rooted config stops at the turn-deal chance; the
/// depth-limited solver values those leaves via the bucketed continuation and
/// never descends. This avoids building (and `setup_pluribus_continuations`
/// over) the full multi-street betting subtree, which is never visited below
/// the depth limit and is the dominant cost for multiway rich-betting trees.
/// River-rooted configs are unchanged (no next chance — showdown terminals are
/// depth-limited by the caller instead).
pub fn build_tree_depth_limited(config: &TreeConfig) -> Result<FlatTree, String> {
    build_tree_inner(config, config.initial_state.next(), &[])
}

/// 2-STREET depth-limited build: truncate TWO streets past `config.initial_state`
/// (a flop-rooted config stops at the RIVER-deal chance). Used by the Pluribus
/// safe-continuation search: the current street is SEARCHED, the next street is the
/// FROZEN + k-biased continuation region (`freeze_node` + `setup_pluribus_continuations`),
/// and the deeper chance is the depth leaf valued by the bucketed continuation. If
/// there is no second street ahead (river-rooted), this is the full tree.
pub fn build_tree_depth_limited_2(config: &TreeConfig) -> Result<FlatTree, String> {
    let truncate_at = config.initial_state.next().and_then(|s| s.next());
    build_tree_inner(config, truncate_at, &[])
}

/// 2-STREET depth-limited build with BRANCHED chance: the next-street chance node
/// branches into `chance_branch` distinct subtrees (one per runout outcome), giving
/// the searched next street CARD-AWARE per-outcome storage. `chance_branch` MUST
/// equal the game's `num_chance_outcomes()` for that street (the solver pairs branch
/// i with outcome i). This is the tree a faithful Pluribus 2-street / safe-continuation
/// search runs on.
pub fn build_tree_depth_limited_2_branched(
    config: &TreeConfig,
    chance_branch: usize,
) -> Result<FlatTree, String> {
    let truncate_at = config.initial_state.next().and_then(|s| s.next());
    build_tree_inner_branched(config, truncate_at, &[], Some(chance_branch))
}

/// PREFLOP-ONLY build (2026-06-12, v1 game derivation): truncate the
/// tree at the preflop→flop chance nodes (they become childless CHANCE
/// leaves). This is the tree the PRODUCTION preflop layer actually
/// walks — `PreflopVectorCfr` evaluates flop-entry chance nodes via
/// the frozen postflop ORACLE and never descends past them, and the
/// preflop raise ladder (MAX_NA_PREFLOP=16) is NOT a legal postflop
/// abstraction (it overflows MAX_NA_POSTFLOP), so fully expanding the
/// postflop zones under the preflop config is both wasted memory and a
/// build-time cap violation at production depth. Postflop zones get
/// their own seam-derived trees (`GameSpec::flop_seam_config`).
pub fn build_tree_preflop_only(config: &TreeConfig) -> Result<FlatTree, String> {
    if config.initial_state != BoardState::Preflop {
        return Err("preflop-only build requires initial_state = Preflop".into());
    }
    build_tree_inner(config, Some(BoardState::Flop), &[])
}

fn build_tree_inner(
    config: &TreeConfig,
    truncate_at: Option<BoardState>,
    bet_override: &[(BoardState, BetSizeOptions)],
) -> Result<FlatTree, String> {
    build_tree_inner_branched(config, truncate_at, bet_override, None)
}

fn build_tree_inner_branched(
    config: &TreeConfig,
    truncate_at: Option<BoardState>,
    bet_override: &[(BoardState, BetSizeOptions)],
    chance_branch: Option<usize>,
) -> Result<FlatTree, String> {
    let num_players = config.num_players as usize;
    if num_players < 2 || num_players > 10 {
        return Err(format!("num_players must be 2-10, got {}", num_players));
    }
    if config.starting_stacks.len() != num_players {
        return Err(format!(
            "starting_stacks.len() must be {}, got {}",
            num_players,
            config.starting_stacks.len()
        ));
    }
    if config.initial_contributions.len() != num_players {
        return Err(format!(
            "initial_contributions.len() must be {}, got {}",
            num_players,
            config.initial_contributions.len()
        ));
    }
    // 2026-06-12: starting_pot is DEAD money from before this tree;
    // live blinds belong in initial_contributions. A preflop tree has
    // starting_pot 0 legitimately — the invariant is that SOME money
    // is in play.
    if config.starting_pot < 0
        || config.starting_pot + config.initial_contributions.iter().sum::<i32>() <= 0
    {
        return Err(format!(
            "starting_pot + initial contributions must be > 0 (pot {}, contribs {:?})",
            config.starting_pot, config.initial_contributions
        ));
    }
    for (i, &s) in config.starting_stacks.iter().enumerate() {
        if s < 0 {
            return Err(format!("starting_stacks[{}] = {} is negative", i, s));
        }
    }

    let mut builder = TreeBuilder {
        config,
        truncate_at,
        bet_override,
        chance_branch,
        tree: FlatTree::new(
            config.num_players,
            config.starting_pot,
            config.starting_stacks.clone(),
            config.rake_rate,
            config.rake_cap,
        ),
    };

    let initial_contributions: Vec<i32> = config.initial_contributions.clone();
    let active_players: Vec<bool> = vec![true; num_players];
    // C1 convention: `committed[p]` = TOTAL chips player p has put in the pot
    // from the start of the hand. Initialized to initial contributions
    // (blinds/antes). Increases monotonically as the player calls or bets.
    // Bounded above by `max_committable(p) = starting_stacks[p] + initial_contributions[p]`.
    let stacks: Vec<i32> = config.initial_contributions.clone();

    // Dispatch first-actor by initial street: preflop reverses postflop's
    // action order (button acts first preflop in HU vs BB acting first
    // postflop). See `first_preflop_player` for the HU convention and the
    // multiway caveat.
    let first_player = match config.initial_state {
        BoardState::Preflop => builder.first_preflop_player(&active_players),
        _ => builder.first_postflop_player_with_button(&active_players),
    };

    let root = FlatNode::player(first_player, config.initial_state, 0);
    let root_idx = builder.tree.alloc_node(root);
    for p in 0..num_players {
        builder
            .tree
            .set_contribution(root_idx, p as u8, initial_contributions[p]);
    }

    // committed_at_round_start initialization.
    //
    // **Postflop trees (initial_state != Preflop):** at flop-start /
    // turn-start / river-start, the round begins with all PRIOR-STREET
    // chips already committed. `initial_contributions` represents what
    // each player put in pre-this-street (e.g., at flop-start, the
    // pre-flop pot share per player). So committed_at_round_start =
    // initial_contributions = current stacks. Then per_street_committed
    // = stacks - committed_at_round_start = 0 at the root, correctly
    // signaling "no bets THIS street yet".
    //
    // **Preflop trees (initial_state == Preflop):** the blinds ARE
    // first-round actions, not pre-existing chips from a prior street.
    // No round preceded preflop, so committed_at_round_start at the
    // preflop root should be 0 for every player — meaning blinds are
    // counted as IN-ROUND chips. Then per_street_committed = stacks =
    // initial_contributions = blinds, so BB's per_street = 2, SB's = 1,
    // others = 0, and UTG (per_street = 0) faces a max_other_per_street
    // of 2 → is_facing_bet = true → action set = {Fold, Call, Raise},
    // matching real preflop poker.
    //
    // The previous initialization (committed_at_round_start = stacks
    // for ALL initial states) treated preflop blinds as pre-existing
    // dead money, leading to per_street = 0 for everyone at the
    // preflop root, is_facing_bet = false for UTG, and an action set
    // of {Check, Bet, AllIn} where Check let SB / UTG see the flop
    // without matching the BB — the free-flop wrong-game bug surfaced
    // by the chip trace and the seam test (2026-06-04, the lead).
    let committed_at_round_start = match config.initial_state {
        BoardState::Preflop => vec![0_i32; num_players],
        _ => stacks.clone(),
    };
    let info = BuildInfo {
        committed_at_round_start,
        stacks,
        active: active_players,
        folded: vec![false; num_players],
        has_acted_this_round: vec![false; num_players],
        round_starter: first_player as usize,
        num_bets: 0,
        allin_flag: false,
        board_state: config.initial_state,
        depth: 0,
    };

    builder.build_recursive(root_idx, info);

    // Phase 3 fix: the Phase-2-era post-build TERMINAL→PLAYER fixup loop has
    // been REMOVED. With the rewritten player-advancement (each child is
    // allocated with the correct player_id directly, no in-place reassignment),
    // a TERMINAL node should never end up with children. If it does, the
    // rewrite has regressed — assert loud rather than silently relabel.
    for (i, n) in builder.tree.nodes.iter().enumerate() {
        debug_assert!(
            !(n.node_type == NODE_TYPE_TERMINAL && n.num_children > 0),
            "Tree-builder regression: TERMINAL node[{}] has {} children. The \
             rewrite was supposed to allocate each child with the correct \
             node_type up-front; if this fires, something is still doing \
             in-place player_id reassignment that leaves a children-bearing \
             TERMINAL behind.",
            i, n.num_children
        );
    }

    // Build-time abstraction-cap assert: every PLAYER node's child count
    // must fit within the action-slot stride of the buffer it'll be stored
    // in. With per-stage MAX_NA (Phase 2), preflop allows MAX_NA_PREFLOP
    // actions (=16) and postflop allows MAX_NA_POSTFLOP (=6, tuned in
    // Phase 4). Hitting these caps means the bet_sizes config produces a
    // legal action set wider than the stride bound for that stage — force
    // an explicit abstraction decision (cap actions, or bump the constant
    // in src/tree/flat.rs which auto-regenerates all derived strides via
    // build.rs codegen).
    use crate::tree::action::BoardState;
    use crate::tree::flat::{MAX_NA_POSTFLOP, MAX_NA_PREFLOP};
    for (i, n) in builder.tree.nodes.iter().enumerate() {
        if n.is_player() {
            let is_preflop = n.board_state == BoardState::Preflop as u8;
            let cap = if is_preflop { MAX_NA_PREFLOP } else { MAX_NA_POSTFLOP };
            let stage_name = if is_preflop { "MAX_NA_PREFLOP" } else { "MAX_NA_POSTFLOP" };
            assert!(
                (n.num_children as usize) <= cap,
                "PLAYER node[{}] (board_state={}) has {} children, exceeds {}={}. \
                 The abstraction (bet_sizes config) produces a legal action set \
                 wider than the per-stage stride. Either cap the action set \
                 (Option A) or raise {} in src/tree/flat.rs (build.rs auto- \
                 regenerates the Metal header; Option B).",
                i, n.board_state, n.num_children, stage_name, cap, stage_name
            );
        }
    }

    builder.tree.compute_levels();

    Ok(builder.tree)
}

struct TreeBuilder<'a> {
    config: &'a TreeConfig,
    /// If set, chance nodes transitioning INTO this board state become childless
    /// leaves (the caller's depth-limited solver values them). `Some(Flop)` =
    /// preflop-only (oracle continuation); `Some(Turn)`/`Some(River)` = per-street
    /// search depth limit. `None` = full tree. See `build_tree_depth_limited`.
    truncate_at: Option<BoardState>,
    /// Per-board-state bet-menu overrides (nested solving). A board state listed
    /// here uses its menu instead of `config.bet_sizes`; empty = uniform.
    bet_override: &'a [(BoardState, BetSizeOptions)],
    /// If `Some(n)`, a non-truncated chance node BRANCHES into `n` distinct,
    /// structurally-identical child subtrees (one per runout outcome) instead of a
    /// single replayed child. Each branch gets its OWN nodes ⇒ its own regret /
    /// strategy storage ⇒ a CARD-AWARE searched street below the chance (the
    /// per-outcome storage a faithful 2-street / Pluribus search needs). The
    /// solver pairs branch `i` with chance outcome `i` (`set_chance_outcome(i)`).
    /// `None` = the legacy single replayed child (card-agnostic). `n` MUST equal
    /// the game's `num_chance_outcomes()` for that street.
    chance_branch: Option<usize>,
    tree: FlatTree,
}

struct BuildInfo {
    // Under the C1 convention from the prior tree-builder fix, `stacks[p]`
    // tracks the TOTAL chips player p has put into the pot from the start of
    // the hand (cumulative committed). Initialized to initial_contributions.
    stacks: Vec<i32>,
    active: Vec<bool>,
    folded: Vec<bool>,
    has_acted_this_round: Vec<bool>,
    round_starter: usize,
    num_bets: i32,
    allin_flag: bool,
    board_state: BoardState,
    depth: usize,
    // Per-street snapshot: cumulative committed at the start of the current
    // betting round (i.e. the value `stacks` had just after the most recent
    // street transition). On the first street, equals initial_contributions.
    // Refreshed in `add_chance_child` after copying parent contribs to the
    // chance node. Used by the per-street classifier to decide facing-bet
    // vs not-facing-bet (a player is facing a bet iff their per-street
    // commit is less than some other active player's per-street commit;
    // cumulative comparison is incorrect across asymmetric blinds and
    // street transitions).
    committed_at_round_start: Vec<i32>,
}

impl BuildInfo {
    fn folded_mask(&self) -> u16 {
        let mut mask: u16 = 0;
        for (p, &f) in self.folded.iter().enumerate() {
            if f {
                mask |= 1u16 << p;
            }
        }
        mask
    }

    fn clone_for_child(&self) -> Self {
        BuildInfo {
            stacks: self.stacks.clone(),
            active: self.active.clone(),
            folded: self.folded.clone(),
            has_acted_this_round: self.has_acted_this_round.clone(),
            round_starter: self.round_starter,
            num_bets: self.num_bets,
            allin_flag: self.allin_flag,
            board_state: self.board_state,
            depth: self.depth + 1,
            committed_at_round_start: self.committed_at_round_start.clone(),
        }
    }

    fn num_active(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }

    fn contributions_sum(&self, base_contributions: &[i32]) -> i32 {
        base_contributions.iter().sum()
    }
}

/// Action-set classification for the player whose turn it is, derived from
/// the rules of no-limit poker:
///   - `AllInForcedCheck`: the acting player has no chips remaining
///     (`max_committable - cumulative_committed == 0`). They cannot fold,
///     call, bet, or raise. Their only "action" is a pass-through CHECK.
///   - `FacingBet`: the acting player has put in fewer chips THIS STREET
///     than at least one other still-eligible (active, not-yet-folded)
///     player. To remain in the hand they must match the highest
///     this-street commit; legal actions are FOLD, CALL (or all-in for
///     less if their remaining chips don't cover the full call), and any
///     RAISE the abstraction allows.
///   - `NotFacingBet`: the acting player's this-street commit equals (or
///     exceeds) every other active player's this-street commit. No one
///     has bet yet, or everyone has matched. Legal actions are CHECK and
///     any BET the abstraction allows. FOLD is NOT legal — there is
///     nothing on the table that requires a fold or a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActionClass {
    AllInForcedCheck,
    FacingBet,
    NotFacingBet,
}

impl<'a> TreeBuilder<'a> {
    /// Physical maximum chips this player can ever commit to the pot:
    /// their starting stack plus any blinds/antes already posted.
    /// Used to cap bet/raise amounts and call-matches at physical reality.
    fn max_committable(&self, player: usize) -> i32 {
        self.config.starting_stacks[player] + self.config.initial_contributions[player]
    }

    // ─── Per-street classifier helpers — INDEPENDENT from gate code ───
    //
    // These functions implement the action-class classifier derived directly
    // from the poker rules documented above on `ActionClass`. They are NOT
    // copied from `tests/tree_correctness_gate.rs` or
    // `tests/tree_correctness_gate_hand_built.rs`. When the builder and the
    // gate (each an independent derivation of the same rules) agree on a
    // given tree, the agreement is real evidence; if either had been written
    // by copying the other, agreement would be evidence of nothing.

    /// Chips that player p has committed during the CURRENT betting round
    /// (cumulative committed minus the snapshot taken at the start of the
    /// round / most recent chance transition).
    fn per_street_committed(info: &BuildInfo, p: usize) -> i32 {
        info.stacks[p] - info.committed_at_round_start[p]
    }

    /// Largest per-street commit among all active players who are NOT the
    /// `acting_player`. (Inactive/folded players are not counted — they no
    /// longer affect what the acting player owes this street.)
    fn max_other_per_street(info: &BuildInfo, acting_player: usize) -> i32 {
        let np = info.stacks.len();
        let mut m = 0i32;
        for p in 0..np {
            if p == acting_player || !info.active[p] {
                continue;
            }
            let s = Self::per_street_committed(info, p);
            if s > m {
                m = s;
            }
        }
        m
    }

    /// True iff the acting player has zero chips remaining (their cumulative
    /// committed equals their physical maximum).
    fn is_all_in(&self, info: &BuildInfo, player: usize) -> bool {
        info.stacks[player] >= self.max_committable(player)
    }

    /// True iff the acting player has put in fewer chips this street than at
    /// least one other active player. (Strict inequality — equal commits
    /// mean no one has raised over you; you're not facing a bet.)
    fn is_facing_bet(info: &BuildInfo, player: usize) -> bool {
        Self::per_street_committed(info, player) < Self::max_other_per_street(info, player)
    }

    /// Classify the action set the acting player is legally entitled to,
    /// per poker rules, given the current build state.
    fn compute_legal_action_class(&self, info: &BuildInfo, player: usize) -> ActionClass {
        if self.is_all_in(info, player) {
            ActionClass::AllInForcedCheck
        } else if Self::is_facing_bet(info, player) {
            ActionClass::FacingBet
        } else {
            ActionClass::NotFacingBet
        }
    }

    fn first_postflop_player(&self, active: &[bool]) -> u8 {
        let num_players = self.config.num_players as usize;
        for i in 0..num_players {
            if active[i] {
                return i as u8;
            }
        }
        0
    }

    /// First player to act PREFLOP.
    ///
    /// Behavior depends on `config.button_player`:
    ///
    /// **Explicit button (`button_player = Some(b)`)**: derives positions
    /// by rotation per the standard poker convention.
    ///   - HU (np=2): button == SB, so the button itself acts first
    ///     preflop. Returns `b`.
    ///   - Multiway (np>=3): UTG = (button + 3) mod np acts first.
    ///     Returns `(b + 3) % np`.
    ///
    /// **Legacy inference (`button_player = None`)**: returns the
    /// highest-indexed active player. HU-correct under the convention
    /// "higher-indexed seat is the button" (e.g., `initial_contributions
    /// = [2, 1]` with BB at player 0 and SB at player 1). Multiway-
    /// incorrect (returns the button instead of UTG); all multiway
    /// preflop callers MUST set `button_player` explicitly.
    fn first_preflop_player(&self, active: &[bool]) -> u8 {
        let num_players = self.config.num_players as usize;
        if let Some(button) = self.config.button_player {
            let button = button as usize;
            assert!(button < num_players,
                "button_player {} out of range for num_players {}", button, num_players);
            let first = if num_players == 2 {
                // HU: button == SB acts first preflop.
                button
            } else {
                // Multiway: UTG = (button + 3) % np acts first preflop.
                (button + 3) % num_players
            };
            assert!(active[first],
                "first preflop player (computed from button {}) is inactive at index {}",
                button, first);
            return first as u8;
        }
        // Legacy: highest-indexed active player. HU-correct under the
        // higher-indexed-is-button convention; multiway-incorrect.
        for i in (0..num_players).rev() {
            if active[i] {
                return i as u8;
            }
        }
        0
    }

    /// First player to act POSTFLOP.
    ///
    /// **Explicit button (`button_player = Some(b)`)**: SB = (button + 1)
    /// mod np acts first. At HU (np=2), this is BB = (button + 1) mod 2
    /// since SB == button; the formula collapses correctly.
    ///
    /// **Legacy inference (`button_player = None`)**: returns the lowest-
    /// indexed active player. HU-correct under the convention that BB is
    /// the lowest-indexed seat; multiway-incorrect if SB isn't player 0.
    fn first_postflop_player_with_button(&self, active: &[bool]) -> u8 {
        let num_players = self.config.num_players as usize;
        if let Some(button) = self.config.button_player {
            let button = button as usize;
            let mut idx = (button + 1) % num_players;
            // Skip folded/inactive players in clockwise order.
            for _ in 0..num_players {
                if active[idx] { return idx as u8; }
                idx = (idx + 1) % num_players;
            }
            return 0;
        }
        // Legacy: lowest-indexed active.
        self.first_postflop_player(active)
    }

    fn next_active_player(&self, current: usize, active: &[bool]) -> Option<usize> {
        let num_players = self.config.num_players as usize;
        for offset in 1..=num_players {
            let next = (current + offset) % num_players;
            if active[next] {
                return Some(next);
            }
        }
        None
    }

    fn is_round_complete(&self, info: &BuildInfo) -> bool {
        // Round is complete iff every active player has acted this round AND
        // either (a) all active players have matched cumulative commits (the
        // standard case — post-bet/post-call equalization), or (b) no betting
        // has occurred this street (per-street commits all 0 for active
        // players — handles the asymmetric-blind case where players check
        // around but cumulative commits remain unequal because of the blinds).
        //
        // The cumulative-equal check is the conventional Model A semantics
        // used by Call/Bet handlers; we keep it as the primary check.
        // The no-betting-this-street special case fills the gap for
        // asymmetric blinds: without it, on the flop with [10,5,5,5,5,5]
        // initial contribs, no one can ever "end the round" by checking
        // because cumulative stays unequal.
        let num_players = self.config.num_players as usize;
        for p in 0..num_players {
            if info.active[p] && !info.has_acted_this_round[p] {
                return false;
            }
        }

        let active: Vec<usize> = (0..num_players)
            .filter(|&p| info.active[p])
            .collect();
        if active.is_empty() {
            return true;
        }

        // (a) Standing-bet rule (corrected from cum_eq 2026-06-04, the lead):
        // The round is complete when every active player has matched the
        // standing bet (the max contribution among active players) OR is
        // all-in at their personal max_committable. The previous
        // "all stacks equal" check rejected legal round-end states where
        // some active players were all-in for less than the standing bet
        // (capped by their max_committable, with the excess from larger
        // stacks being uncalled-bet-returned per poker rules). The
        // unequal commits at the seam from those all-in-at-less states
        // were the foundation bug surviving past the initial
        // committed_at_round_start fix; this is the parallel correction
        // in is_round_complete.
        let standing_bet = active.iter().map(|&p| info.stacks[p]).max().unwrap();
        let all_matched_or_allin = active.iter().all(|&p| {
            info.stacks[p] == standing_bet || info.stacks[p] >= self.max_committable(p)
        });
        if all_matched_or_allin {
            return true;
        }

        // (b) No betting this street — all active have per-street commit 0.
        // Handles the postflop check-around case (no actions, round ends
        // when all checked).
        if active.iter().all(|&p| Self::per_street_committed(info, p) == 0) {
            return true;
        }

        false
    }

    fn only_one_active(&self, info: &BuildInfo) -> bool {
        info.num_active() <= 1
    }

    fn get_pot(&self, node_idx: usize) -> i32 {
        // BUG FIX 2026-06-12 (harness anchor chain): the pot MUST
        // include config.starting_pot — the solver terminal math
        // treats starting_pot + contributions as additive money, but
        // pot-relative bet sizing here ignored starting_pot, so e.g.
        // the oracle tree (pot 12, contribs 0) sized every "pot" bet
        // from pot 0 and floored it to 1 chip.
        let num_players = self.config.num_players as usize;
        self.config.starting_pot
            + (0..num_players)
                .map(|p| self.tree.get_contribution(node_idx, p as u8))
                .sum::<i32>()
    }

    fn build_recursive(&mut self, node_idx: usize, info: BuildInfo) {
        // MAX_DEPTH was previously a SILENT return — it silently dropped
        // recursion past depth 64, producing incomplete trees with no
        // signal. This hid the all-in-mixed-commits round-completion bug
        // for the entire history of this work (the lead's MAX_DEPTH lesson
        // 2026-06-04: "a silent cap that no enumerator happens to
        // overflow against would never be found"). Made LOUD (panic):
        // any depth exceeded now signals a real round-termination
        // issue, not silently truncates the tree.
        assert!(
            info.depth <= MAX_DEPTH,
            "build_recursive depth {} exceeds MAX_DEPTH {}. This signals a \
             round-termination bug (the round-completion logic is rejecting \
             legal round-end states; the recursion never terminates). \
             Previously a silent return that hid this bug as truncated trees; \
             made loud per the lead's audit-silent-truncations directive. \
             Investigate is_round_complete / all-in handling for this \
             configuration before raising MAX_DEPTH.",
            info.depth, MAX_DEPTH
        );

        // EARLY RETURN: if this node was already constructed as TERMINAL or
        // CHANCE by make_child_node (e.g., a FOLD that leaves only one
        // player active is typed TERMINAL), don't re-process it here. Re-processing
        // adds spurious children and forces the post-build TERMINAL→PLAYER
        // fixup, creating empty PLAYER nodes that the standing gate flags.
        // This is a key Phase 3 structural fix.
        let nt = self.tree.nodes[node_idx].node_type;
        if nt == NODE_TYPE_TERMINAL {
            // already terminal; nothing more to do
            return;
        }
        if nt == crate::tree::flat::NODE_TYPE_CHANCE {
            // chance nodes are constructed by add_chance_child which already
            // handles the post-chance player recursion; don't re-process here.
            return;
        }

        if self.only_one_active(&info) {
            self.tree.nodes[node_idx].node_type = NODE_TYPE_TERMINAL;
            self.tree.set_folded_mask(node_idx, info.folded_mask());
            return;
        }

        if info.allin_flag {
            // All-in check: every active player has reached their physical
            // maximum (no chips remaining). Use max_committable per player,
            // NOT cumulative-equal — with asymmetric blinds players have
            // different max_committable values (e.g., big-blind=210,
            // small-blind=205), so cumulative-equal fails even when all
            // are genuinely all-in. The correct check is per-player
            // physical-max comparison.
            let all_allin = (0..self.config.num_players as usize)
                .all(|p| !info.active[p] || info.stacks[p] >= self.max_committable(p));
            if all_allin {
                match info.board_state.next() {
                    Some(next_street) => {
                        // All-in on Preflop/Flop/Turn → chance to next street.
                        let mut child_info = info.clone_for_child();
                        child_info.board_state = next_street;
                        self.add_chance_child(node_idx, child_info);
                    }
                    None => {
                        // All all-in on river → showdown terminal.
                        self.tree.nodes[node_idx].node_type = NODE_TYPE_TERMINAL;
                        self.tree.set_folded_mask(node_idx, info.folded_mask());
                    }
                }
                return;
            }
        }

        // PHASE 3 FIX: removed the `info.prev_action == Action::Call` shortcut.
        // It was correct for heads-up (after a call, no one else can act) but
        // wrong for 3+ players — after p1 calls p0's bet, p2 still needs to
        // act. The shortcut was incorrectly advancing to the next street and
        // leaving the intended p2-decision node as an empty PLAYER. Now rely
        // solely on `is_round_complete_after_action` which checks that ALL
        // active players have acted and all have equal stacks.
        if self.is_round_complete_after_action(&info) {
            match info.board_state.next() {
                None => {
                    // River round complete → terminal showdown.
                    self.tree.nodes[node_idx].node_type = NODE_TYPE_TERMINAL;
                    self.tree.set_folded_mask(node_idx, info.folded_mask());
                }
                Some(next_street) => {
                    let mut child_info = info.clone_for_child();
                    child_info.board_state = next_street;
                    self.add_chance_child(node_idx, child_info);
                }
            }
            return;
        }

        let player = self.tree.nodes[node_idx].player_id as usize;
        if !info.active[player] {
            if let Some(next) = self.next_active_player(player, &info.active) {
                self.tree.nodes[node_idx].player_id = next as u8;
                self.build_recursive(node_idx, info);
            } else {
                // No active player remaining — this should be a TERMINAL, not
                // an empty PLAYER node. The only_one_active check above should
                // already have caught num_active <= 1, but defending against
                // any state where next_active_player returns None.
                self.tree.nodes[node_idx].node_type = NODE_TYPE_TERMINAL;
                self.tree.set_folded_mask(node_idx, info.folded_mask());
            }
            return;
        }

        let mut child_infos_and_actions = Vec::new();
        self.compute_actions(node_idx, &info, &mut child_infos_and_actions);

        if child_infos_and_actions.is_empty() {
            self.tree.nodes[node_idx].node_type = NODE_TYPE_TERMINAL;
            self.tree.set_folded_mask(node_idx, info.folded_mask());
            return;
        }

        let mut child_indices = Vec::with_capacity(child_infos_and_actions.len());
        for (action, child_info) in &child_infos_and_actions {
            let child_idx = self.make_child_node(node_idx, action, child_info);
            child_indices.push(child_idx as u32);
        }
        self.tree.set_children(node_idx, child_indices);

        for (i, (_, child_info)) in child_infos_and_actions.into_iter().enumerate() {
            let child_idx = self.tree.nodes[node_idx].children_start as usize + i;
            let child_idx = self.tree.children[child_idx] as usize;
            self.build_recursive(child_idx, child_info);
        }
    }

    fn is_round_complete_after_action(&self, info: &BuildInfo) -> bool {
        // Phase 4 cleanup: the historical `prev_action != None/Chance` guard
        // was redundant. `is_round_complete` itself requires every active
        // player to have acted this round (`has_acted_this_round[p]`), which
        // is `[false; np]` at both the root (init) and after any chance
        // transition (reset in `add_chance_child`). So round-complete cannot
        // fire at round-start regardless of `prev_action`. Function kept as
        // a named alias for readability at the build_recursive callsite.
        self.is_round_complete(info)
    }

    fn add_chance_child(&mut self, parent_idx: usize, mut info: BuildInfo) {
        // STRUCTURAL FIX: when this is called during build_recursive on a node
        // that was originally allocated as a PLAYER (round-complete advance,
        // all-allin advance), the PARENT NODE itself should become a CHANCE
        // node — not stay a PLAYER node with a chance child. The previous
        // behavior left a degenerate "PLAYER with single CHANCE child" that
        // the gate (correctly) flags as having no legal player actions.
        //
        // Approach: mutate parent_idx into a CHANCE node directly, then
        // allocate the post-chance player as its child. This consumes
        // parent_idx as the chance node itself rather than adding an
        // intermediate chance child below it.
        self.tree.nodes[parent_idx].node_type =
            crate::tree::flat::NODE_TYPE_CHANCE;
        self.tree.nodes[parent_idx].board_state = info.board_state as u8;
        // Record the fold state at the chance boundary (informational —
        // terminals remain the authoritative settlement surface; seam
        // instruments read this to recover the live set at flop entry).
        self.tree.set_folded_mask(parent_idx, info.folded_mask());
        // Note: the action_label on parent_idx is whatever brought us here
        // (e.g., CALL, CHECK). The gate uses this when propagating folded_mask
        // from the grandparent; CALL/CHECK don't trigger FOLD propagation, so
        // that's correct.

        // DEPTH-LIMITED BUILD: the chance node entering `truncate_at` is a leaf
        // of this tree; the caller's depth-limited solver supplies its value
        // (preflop oracle, or the per-street bucketed continuation).
        if self.truncate_at == Some(info.board_state) {
            return;
        }

        // Reset round state at the chance boundary (shared by all branches — the
        // betting structure below a chance is identical; only the dealt card
        // differs, which the SOLVER supplies per branch via set_chance_outcome).
        info.has_acted_this_round = vec![false; self.config.num_players as usize];
        info.num_bets = 0;
        info.round_starter = self.first_postflop_player_with_button(&info.active) as usize;
        // Refresh per-street commit snapshot at the chance boundary. The new
        // round begins with all players' current cumulative committed values
        // as their "starting" position for the next betting round.
        info.committed_at_round_start = info.stacks.clone();

        // BRANCHED chance (chance_branch = Some(n)): n distinct, structurally-
        // identical post-chance subtrees, one per runout outcome — each with its
        // own nodes ⇒ its own storage ⇒ a CARD-AWARE searched street. Default
        // (None) = a single replayed child (legacy card-agnostic). Branch i is
        // paired with chance outcome i by the solver's multi-child chance walk.
        let nbranch = self.chance_branch.unwrap_or(1);
        let mut child_indices = Vec::with_capacity(nbranch);
        for _ in 0..nbranch {
            let child_node = {
                let first = self.first_postflop_player_with_button(&info.active);
                FlatNode::player(first, info.board_state, self.tree.nodes[parent_idx].amount)
            };
            let child_idx = self.tree.alloc_node(child_node);
            // Post-chance player's incoming "action" is CHANCE, not FOLD (else the
            // gate's FOLD-propagation would treat it as reached via a fold).
            self.tree.nodes[child_idx].action_label = crate::tree::flat::ACTION_LABEL_CHANCE;
            for p in 0..self.config.num_players as usize {
                let contrib = self.tree.get_contribution(parent_idx, p as u8);
                self.tree.set_contribution(child_idx, p as u8, contrib);
            }
            child_indices.push(child_idx as u32);
        }
        // parent_idx IS the chance node now; its children are the post-chance players.
        self.tree.set_children(parent_idx, child_indices.clone());
        for &cidx in &child_indices {
            self.build_recursive(cidx as usize, info.clone_for_child());
        }
    }

    fn compute_actions(
        &self,
        node_idx: usize,
        info: &BuildInfo,
        out: &mut Vec<(Action, BuildInfo)>,
    ) {
        let player = self.tree.nodes[node_idx].player_id as usize;
        let num_players = self.config.num_players as usize;

        // C1: info.stacks[p] is now "total chips committed by p so far".
        let mut max_other_committed = 0i32;
        for p in 0..num_players {
            if p != player && info.active[p] {
                max_other_committed = max_other_committed.max(info.stacks[p]);
            }
        }

        let player_committed = info.stacks[player];
        let max_player_total = self.max_committable(player);
        let player_remaining = (max_player_total - player_committed).max(0);
        let to_call = (max_other_committed - player_committed).max(0);
        let pot = self.get_pot(node_idx);
        // prev_amount: the highest commitment any other player has put in.
        // Used by add_*_size_action and clamp_and_force_allin to translate
        // a configured bet/raise delta into a TOTAL post-action commitment.
        let prev_amount = max_other_committed;
        // max_amount: the most this player can commit in TOTAL — capped at
        // their physical chip total (starting_stack + initial_contribution).
        let max_amount = max_player_total;
        // All-in (or zero-stack seam cell): forced check. MUST precede
        // the min_amount computation — with max_committable = 0 (e.g. a
        // flop-seam tree where everyone is all-in preflop) the
        // clamp(1, 0) below panics (caught by the v1 seam census
        // 2026-06-12).
        if player_remaining <= 0 {
            let mut child_info = info.clone_for_child();
            child_info.has_acted_this_round[player] = true;
            out.push((Action::Check, child_info));
            return;
        }

        // min_amount: the minimum legal TOTAL commitment to participate.
        // Must at least match the largest committer (but no more than this
        // player can physically pay). A short-stack call is bounded above
        // by max_player_total, which the clamp routine then forces all-in.
        let min_amount = (player_committed + to_call.min(player_remaining))
            .clamp(1, max_amount);

        let spr_after_call =
            (player_remaining - to_call.min(player_remaining)) as f64
                / (pot + to_call.min(player_remaining) * (info.num_active() as i32)) as f64;

        let num_remaining_streets = info.board_state.num_remaining_streets();

        // Per-street override (nested solving): a coarse/check-only menu on a later
        // street; defaults to the config's uniform menu.
        let bet_options = self
            .bet_override
            .iter()
            .find(|(bs, _)| *bs == info.board_state)
            .map(|(_, m)| m)
            .unwrap_or(&self.config.bet_sizes);

        let mut actions = Vec::new();

        // PHASE 2 REWRITE: branch on per-street commit comparison (poker rules),
        // not on prev_action (which is incorrect across asymmetric blinds and
        // post-chance states with stale prev_action). The classifier helpers
        // are independently derived from poker rules; see `compute_legal_action_class`.
        //
        // Note: the all-in-forced-check case is handled by the early return
        // above at `player_remaining <= 0`. So here we only need to handle
        // FacingBet and NotFacingBet.
        // RAISE-DEPTH cap (2026-06-12, Pluribus-style — see the
        // TreeConfig field doc): once num_bets reaches the per-street
        // cap, aggression is pruned; defensive actions (fold / call /
        // check) are NEVER pruned — defense-completeness holds by
        // construction, only the cap VALUE needs the exploitability
        // measurement.
        let may_aggress = self.config.max_bets_per_street.map_or(true, |bc| {
            let applies = bc.seat.map_or(true, |s| s as usize == player);
            !applies || info.num_bets < bc.cap as i32
        });

        let action_class = self.compute_legal_action_class(info, player);
        match action_class {
            ActionClass::AllInForcedCheck => {
                // Unreachable in practice — handled by early-return above.
                // Belt-and-suspenders: emit forced CHECK.
                actions.push(Action::Check);
            }
            ActionClass::NotFacingBet => {
                // Open to act: legal = {CHECK, BET sizes, optional ALLIN if threshold}.
                // FOLD is NOT legal (nothing to call).
                actions.push(Action::Check);

                if may_aggress {
                    for &bet_size in &bet_options.bet {
                        // Pass prev_amount (max other committed) so bet sizes produce
                        // TOTAL post-action commitment (C1), not raw delta.
                        self.add_bet_size_action(
                            &bet_size,
                            pot,
                            prev_amount,
                            max_amount,
                            min_amount,
                            num_remaining_streets,
                            0,
                            spr_after_call,
                            &mut actions,
                        );
                    }

                    if max_amount
                        <= (pot as f64 * self.config.add_allin_threshold).round() as i32
                    {
                        actions.push(Action::AllIn(max_amount));
                    }
                }
            }
            ActionClass::FacingBet => {
                // Owes chips this street: legal = {FOLD, CALL/ALLIN, raises if !allin_flag}.
                // CHECK is NOT legal (cannot pass when owing).
                actions.push(Action::Fold);
                // RAISE-OR-FOLD opens: when `no_open_limp` is set, suppress CALL at an
                // OPEN (facing only the forced blind ⇒ num_bets==0, preflop) — open-
                // limping is strictly dominated in 6-max GTO and is otherwise an
                // attractor the static EQR terminal can't price. Calling a RAISE
                // (num_bets>=1) is still allowed.
                // Suppress CALL at the open (no_open_limp, num_bets==0) OR at the
                // 3-bet decision (threebet_or_fold, num_bets==1) — 3bet-or-fold.
                //
                // EXCEPTION — the BB's defensive flat vs a single open: 3bet-or-
                // fold from the blind makes the BB over-fold to steals (it folds
                // everything it can't profitably 3-bet), which hands the late
                // positions a free steal — BTN/SB opens explode to ~100% of trash
                // (the inverse-to-opponents-behind looseness gradient is the tell).
                // The BB closing the action with a price MUST be allowed to flat-
                // call wide; that's what disciplines the steal. Early-position
                // 3bet-or-fold cleanup (HJ/CO/BTN facing an open) is preserved.
                // First cut: BB only (SB doesn't close — BB can squeeze behind —
                // so SB stays 3bet-or-fold; revisit from the output if SB over-folds).
                let bb_seat = self.config.button_player
                    .map(|b| (b as usize + 2) % num_players);
                let is_bb_defending_open = self.config.threebet_or_fold
                    && info.num_bets == 1
                    && info.board_state == BoardState::Preflop
                    && bb_seat == Some(player);
                let suppress_call = (self.config.no_open_limp && info.num_bets == 0)
                    || (self.config.threebet_or_fold && info.num_bets == 1 && !is_bb_defending_open);
                if !suppress_call {
                    actions.push(Action::Call);
                }
                // Suppress the JAM at the 3-bet decision: a sized 3-bet must play the
                // EQR postflop, not collapse to a clean all-in showdown (which the
                // static terminal over-values for strong hands). Deeper jams unaffected.
                let suppress_jam = self.config.threebet_or_fold && info.num_bets == 1;

                if !info.allin_flag && may_aggress {
                    for &bet_size in &bet_options.raise {
                        self.add_raise_size_action(
                            &bet_size,
                            pot,
                            prev_amount,
                            max_amount,
                            min_amount,
                            num_remaining_streets,
                            info.num_bets,
                            spr_after_call,
                            &mut actions,
                        );
                    }

                    let allin_threshold = pot as f64 * self.config.add_allin_threshold;
                    if !suppress_jam && max_amount <= prev_amount + allin_threshold.round() as i32 {
                        actions.push(Action::AllIn(max_amount));
                    }
                }
            }
        }

        // At a 3-bet-or-fold decision, keep size-clamping but DON'T force the big
        // sized 3-bets into all-in jams (suppress_jam there).
        let allow_force_allin = !(self.config.threebet_or_fold && info.num_bets == 1);
        self.clamp_and_force_allin(&mut actions, pot, prev_amount, max_amount, min_amount, allow_force_allin);

        actions.sort();
        actions.dedup();

        actions = merge_bet_actions(actions, pot, prev_amount, self.config.merging_threshold);

        for action in actions {
            let mut child_info = info.clone_for_child();

            match action {
                Action::Fold => {
                    child_info.active[player] = false;
                    child_info.folded[player] = true;
                    child_info.has_acted_this_round[player] = true;
                }
                Action::Check => {
                    child_info.has_acted_this_round[player] = true;
                }
                Action::Call => {
                    // C1: caller matches the largest committer, capped by
                    // their own physical chip total (short-call all-in).
                    child_info.stacks[player] = max_other_committed.min(max_player_total);
                    child_info.has_acted_this_round[player] = true;
                }
                Action::Bet(amount) | Action::Raise(amount) => {
                    // C1: amount is the player's total commitment after the action.
                    debug_assert!(amount <= max_player_total,
                        "Bet/Raise amount {} exceeds player {}'s max committable {}",
                        amount, player, max_player_total);
                    child_info.stacks[player] = amount;
                    child_info.num_bets += 1;
                    child_info.has_acted_this_round = vec![false; num_players];
                    child_info.has_acted_this_round[player] = true;
                    child_info.round_starter = player;
                }
                Action::AllIn(amount) => {
                    // `BetSize::AllIn` emits an all-in TOTAL = max of the configured
                    // bet/raise totals (add_bet_size_action), which a large pot-relative
                    // size can push ABOVE this player's physical cap in a short-stacked
                    // cell (e.g. an asymmetric-commit flop seam, max_committable < stack).
                    // `clamp_and_force_allin` only clamps Bet/Raise, so the over-cap AllIn
                    // survives. The FlatTree write already clamps via `.min(cap)`
                    // (make_child_node); mirror it here so BuildInfo stays consistent.
                    // All-in closes the street (the sole child is a capped call/fold), so
                    // the clamp cannot change tree SHAPE — it keeps the BuildInfo's pot/stack
                    // bookkeeping byte-exact with what the (release) solve built.
                    let amount = amount.min(max_player_total);
                    child_info.stacks[player] = amount;
                    child_info.allin_flag = true;
                    child_info.has_acted_this_round = vec![false; num_players];
                    child_info.has_acted_this_round[player] = true;
                    child_info.round_starter = player;
                }
                _ => {}
            }

            out.push((action, child_info));
        }
    }

    fn add_bet_size_action(
        &self,
        bet_size: &BetSize,
        pot: i32,
        prev_amount: i32,
        _max_amount: i32,
        _min_amount: i32,
        num_remaining_streets: i32,
        _num_bets: i32,
        spr_after_call: f64,
        actions: &mut Vec<Action>,
    ) {
        // C1: action amount = TOTAL chips committed after the action
        //   = prev_amount (max other committed = what we'd match by calling)
        //   + delta (the additional chips this bet adds over the call amount)
        let compute_geometric = |n: i32, max_ratio: f64| -> i32 {
            let ratio =
                ((2.0 * spr_after_call + 1.0).powf(1.0 / n as f64) - 1.0) / 2.0;
            (pot as f64 * ratio.min(max_ratio)).round() as i32
        };

        match bet_size {
            BetSize::PotRelative(ratio) => {
                let delta = (pot as f64 * ratio).round() as i32;
                actions.push(Action::Bet(prev_amount + delta));
            }
            BetSize::PrevBetRelative(_) => {}
            BetSize::Additive(adder, _) => {
                actions.push(Action::Bet(prev_amount + *adder));
            }
            BetSize::Geometric(n, max_ratio) => {
                let n = if *n == 0 { num_remaining_streets } else { *n };
                let delta = compute_geometric(n, *max_ratio);
                actions.push(Action::Bet(prev_amount + delta));
            }
            BetSize::AllIn => {
                // amounts in actions[] are already C1 totals; max is the
                // biggest existing total commitment (we want all-in to be at
                // least as much as any other configured bet).
                let max_total = actions.iter().filter_map(|a| match a {
                    Action::Bet(v) | Action::Raise(v) | Action::AllIn(v) => Some(*v),
                    _ => None,
                }).max().unwrap_or(prev_amount);
                actions.push(Action::AllIn(max_total));
            }
        }
    }

    fn add_raise_size_action(
        &self,
        bet_size: &BetSize,
        pot: i32,
        prev_amount: i32,
        _max_amount: i32,
        _min_amount: i32,
        num_remaining_streets: i32,
        num_bets: i32,
        spr_after_call: f64,
        actions: &mut Vec<Action>,
    ) {
        let compute_geometric = |n: i32, max_ratio: f64| -> i32 {
            let ratio =
                ((2.0 * spr_after_call + 1.0).powf(1.0 / n as f64) - 1.0) / 2.0;
            (pot as f64 * ratio.min(max_ratio)).round() as i32
        };

        match bet_size {
            BetSize::PotRelative(ratio) => {
                let amount = prev_amount + (pot as f64 * ratio).round() as i32;
                actions.push(Action::Raise(amount));
            }
            BetSize::PrevBetRelative(ratio) => {
                let amount = (prev_amount as f64 * ratio).round() as i32;
                actions.push(Action::Raise(amount));
            }
            BetSize::Additive(adder, raise_cap) => {
                if *raise_cap == 0 || num_bets <= *raise_cap {
                    actions.push(Action::Raise(prev_amount + adder));
                }
            }
            BetSize::Geometric(n, max_ratio) => {
                let n = if *n == 0 {
                    (num_remaining_streets - num_bets + 1).max(1)
                } else {
                    (*n - num_bets + 1).max(1)
                };
                let amount = compute_geometric(n, *max_ratio);
                actions.push(Action::Raise(prev_amount + amount));
            }
            BetSize::AllIn => {}
        }
    }

    fn clamp_and_force_allin(
        &self,
        actions: &mut Vec<Action>,
        pot: i32,
        prev_amount: i32,
        max_amount: i32,
        min_amount: i32,
        allow_force_allin: bool,
    ) {
        for action in actions.iter_mut() {
            match *action {
                Action::Bet(amount) => {
                    let clamped = amount.clamp(min_amount, max_amount);
                    let new_amount_diff = clamped - prev_amount;
                    let new_pot = pot + 2 * new_amount_diff;
                    let threshold =
                        (new_pot as f64 * self.config.force_allin_threshold).round() as i32;
                    if allow_force_allin && max_amount <= clamped + threshold {
                        *action = Action::AllIn(max_amount);
                    } else if clamped != amount {
                        *action = Action::Bet(clamped);
                    }
                }
                Action::Raise(amount) => {
                    let clamped = amount.clamp(min_amount, max_amount);
                    let new_amount_diff = clamped - prev_amount;
                    let new_pot = pot + 2 * new_amount_diff;
                    let threshold =
                        (new_pot as f64 * self.config.force_allin_threshold).round() as i32;
                    if allow_force_allin && max_amount <= clamped + threshold {
                        *action = Action::AllIn(max_amount);
                    } else if clamped != amount {
                        *action = Action::Raise(clamped);
                    }
                }
                _ => {}
            }
        }
    }

    fn make_child_node(&mut self, parent_idx: usize, action: &Action, info: &BuildInfo) -> usize {
        let parent = &self.tree.nodes[parent_idx];
        let player = parent.player_id;

        // BUG FIX 2026-06-12 (harness anchor chain): FOLD was typed
        // TERMINAL unconditionally — a heads-up assumption. In multiway
        // trees this ended the entire hand on ANY fold (p0 bets, p1
        // folds → terminal while p2..p5 never act), which is a wrong
        // game. It also produced the "fold-mask under-reporting"
        // symptom the harness caught: single-fold masks at terminals
        // that should represent longer fold sequences that were never
        // built. `info` here is the POST-action child info (the fold
        // already cleared active[player]), so the uniform rule applies:
        // terminal iff at most one player remains.
        let child_type = if self.only_one_active(info) {
            NODE_TYPE_TERMINAL
        } else {
            NODE_TYPE_PLAYER
        };

        let next_player = if child_type == NODE_TYPE_TERMINAL {
            0
        } else {
            self.determine_next_player(player as usize, action, info)
        };

        let child = FlatNode {
            node_type: child_type,
            player_id: next_player as u8,
            board_state: parent.board_state,
            num_children: 0,
            children_start: 0,
            amount: parent.amount,
            action_label: action_to_label(action),
        };

        let child_idx = self.tree.alloc_node(child);
        for p in 0..self.config.num_players as usize {
            let contrib = self.tree.get_contribution(parent_idx, p as u8);
            self.tree.set_contribution(child_idx, p as u8, contrib);
        }

        match action {
            Action::Bet(a) | Action::Raise(a) | Action::AllIn(a) => {
                // C1: `a` is the total chips the player has committed after the
                // action. With the physical-cap enforcement in compute_actions
                // (max_amount = max_committable) and clamp_and_force_allin,
                // `a` is already <= max_committable. Set directly.
                let cap = self.max_committable(player as usize);
                // Bet/Raise are pre-clamped by clamp_and_force_allin; a `BetSize::AllIn`
                // total can legitimately exceed `cap` in a short-stacked cell (see the
                // AllIn arm in compute_actions) — the `.min(cap)` below is the clamp.
                debug_assert!(*a <= cap || matches!(action, Action::AllIn(_)),
                    "post-action commitment {} exceeds player {}'s max {} (action {:?})",
                    a, player, cap, action);
                self.tree.set_contribution(child_idx, player, (*a).min(cap));
            }
            Action::Call => {
                // C1: caller matches the largest committer, but capped by
                // their own physical chip total (short-call all-in case).
                let max_other = (0..self.config.num_players as usize)
                    .filter(|&p| p != player as usize && info.active[p])
                    .map(|p| info.stacks[p])
                    .max()
                    .unwrap_or(0);
                let cap = self.max_committable(player as usize);
                self.tree
                    .set_contribution(child_idx, player, max_other.min(cap));
            }
            _ => {}
        }

        self.tree.set_folded_mask(child_idx, info.folded_mask());

        child_idx
    }

    fn determine_next_player(
        &self,
        current: usize,
        action: &Action,
        info: &BuildInfo,
    ) -> usize {
        if matches!(action, Action::Fold) {
            return self
                .next_active_player(current, &info.active)
                .unwrap_or(0);
        }

        if matches!(action, Action::Check) {
            if self.is_round_complete(info) {
                return self.first_postflop_player_with_button(&info.active) as usize;
            }
            return self
                .next_active_player(current, &info.active)
                .unwrap_or(0);
        }

        self.next_active_player(current, &info.active)
            .unwrap_or(0)
    }
}

fn action_to_label(action: &Action) -> u8 {
    match action {
        Action::Fold => 0,
        Action::Check => 1,
        Action::Call => 2,
        Action::Bet(_) => 3,
        Action::Raise(_) => 4,
        Action::AllIn(_) => 5,
        Action::Chance(_) => 6,
        _ => 0,
    }
}

fn merge_bet_actions(actions: Vec<Action>, pot: i32, offset: i32, param: f64) -> Vec<Action> {
    if param <= 0.0 {
        return actions;
    }
    const EPS: f64 = 1e-12;

    let get_amount = |action: Action| match action {
        Action::Bet(amount) | Action::Raise(amount) | Action::AllIn(amount) => amount,
        _ => -1,
    };

    let mut cur_amount = i32::MAX;
    let mut ret = Vec::new();

    for &action in actions.iter().rev() {
        let amount = get_amount(action);
        if amount > 0 {
            let ratio = (amount - offset) as f64 / pot as f64;
            let cur_ratio = (cur_amount - offset) as f64 / pot as f64;
            let threshold_ratio = (cur_ratio - param) / (1.0 + param);
            if ratio < threshold_ratio * (1.0 - EPS) {
                ret.push(action);
                cur_amount = amount;
            }
        } else {
            ret.push(action);
        }
    }

    ret.reverse();
    ret
}
