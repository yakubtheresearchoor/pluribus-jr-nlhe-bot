//! Runtime decision API: turn a runtime game-state request into the bot's action
//! distribution. POSTFLOP decisions run the per-street search (`search_decision`)
//! with optional PAIR-MODE range blocking (up to 4 known hole cards: the hero's 2
//! + the partner's 2 — the partner's are removed from the pool's range). Preflop
//! is handled separately by the EQR preflop player (wired by the server).
//!
//! Contract design (we own the schema): the runtime — which tracks the live game
//! — sends the DERIVED per-decision subgame state (live count, the hero's index
//! among the live set, the matched commit + pot ENTERING this street, and the
//! ordered betting actions THIS street). The API builds the street subgame,
//! replays those actions to the hero's decision node, searches, and returns the
//! per-action probabilities for the hero's actual hand. This keeps the bot from
//! re-implementing the full betting engine the runtime already runs.

use crate::blueprint::Blueprint;
use crate::live2_bank::load_live2;
use crate::pluribus_play::{hand_index, search_decision, SearchCfg};
use crate::preflop_player::{splitmix64, PreflopPlayer};
use serde::{Deserialize, Serialize};
use solver_core::abstraction::flop_isomorphism::{canonicalize_flop, enumerate_canonical_flops};
use solver_core::solver::preflop_start_game::flop_combo_layout;
use solver_core::tree::action::production_game_v1;
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};
use std::sync::OnceLock;

/// Action labels (pinned tree facts): FOLD=0 CHECK=1 CALL=2 BET=3 RAISE=4 ALLIN=5.
pub fn action_name(label: u8) -> &'static str {
    match label {
        0 => "fold",
        1 => "check",
        2 => "call",
        3 => "bet",
        4 => "raise",
        5 => "allin",
        _ => "?",
    }
}

/// Map a canonical flop (blueprint frame) to its `flop_id` — the index into the
/// 1,755-entry canonical-flop list (which is exactly how the blueprint cells are
/// named, `flop_{id:04}.bp`). The lookup map is built once.
fn canon_flop_id(canon: [u8; 3]) -> Option<u32> {
    static MAP: OnceLock<std::collections::HashMap<[u8; 3], u32>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        enumerate_canonical_flops()
            .into_iter()
            .enumerate()
            .map(|(i, f)| {
                let mut s = f;
                s.sort_unstable();
                (s, i as u32)
            })
            .collect()
    });
    let mut key = canon;
    key.sort_unstable();
    map.get(&key).copied()
}

/// All 24 suit permutations (source-suit index → target suit).
fn all_suit_perms() -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(24);
    for a in 0..4u8 {
        for b in 0..4u8 {
            if b == a {
                continue;
            }
            for c in 0..4u8 {
                if c == a || c == b {
                    continue;
                }
                out.push([a, b, c, 6 - a - b - c]); // 0+1+2+3 = 6 ⇒ the 4th suit
            }
        }
    }
    out
}

fn remap_card(c: u8, perm: &[u8; 4]) -> u8 {
    let rank = c >> 2;
    let suit = (c & 3) as usize;
    (rank << 2) | perm[suit]
}

/// Find the full suit permutation that maps `raw_flop` onto its canonical
/// representative `canon` (set-wise). Canonicalization is `relabel ∘ perm`, itself
/// a permutation, so one of the 24 always works. The SAME map is then applied to
/// the hole / turn / river cards to bring the whole hand into the canonical frame.
fn suit_perm_to_canonical(raw_flop: [u8; 3], canon: [u8; 3]) -> Option<[u8; 4]> {
    let mut canon_sorted = canon;
    canon_sorted.sort_unstable();
    for p in all_suit_perms() {
        let mut mapped = [
            remap_card(raw_flop[0], &p),
            remap_card(raw_flop[1], &p),
            remap_card(raw_flop[2], &p),
        ];
        mapped.sort_unstable();
        if mapped == canon_sorted {
            return Some(p);
        }
    }
    None
}

/// Rewrite a raw-board request into the blueprint's canonical frame IN PLACE:
/// derive `flop_id` from the board's flop and remap the board + hero/partner hole
/// cards by the canonicalizing suit permutation. Returns None if the board is
/// shorter than a flop or the canonical flop isn't in the bank's list. Idempotent
/// effect for an already-canonical board (perm = identity, flop_id unchanged).
pub fn route_to_canonical(req: &mut DecideRequest) -> Option<()> {
    if req.board.len() < 3 {
        return None;
    }
    let raw_flop = [req.board[0], req.board[1], req.board[2]];
    let canon = canonicalize_flop(raw_flop);
    let flop_id = canon_flop_id(canon)?;
    let perm = suit_perm_to_canonical(raw_flop, canon)?;
    // The flop portion becomes the canonical flop in its EXACT order (= bp.flop),
    // so board[0..3] matches the blueprint frame precisely. Turn/river (board[3..])
    // and the hole cards are remapped by the same canonicalizing permutation.
    for c in req.board[3..].iter_mut() {
        *c = remap_card(*c, &perm);
    }
    req.board[0] = canon[0];
    req.board[1] = canon[1];
    req.board[2] = canon[2];
    for c in req.hero_cards.iter_mut() {
        *c = remap_card(*c, &perm);
    }
    if let Some(pc) = req.partner_cards.as_mut() {
        for c in pc.iter_mut() {
            *c = remap_card(*c, &perm);
        }
    }
    req.flop_id = flop_id;
    Some(())
}

/// One action taken on the current street (for replaying to the hero's node).
/// `label` is the action label; `to_total` is the player's total contribution
/// AFTER the action (used to disambiguate multiple bet/raise sizes).
#[derive(Deserialize, Clone, Copy)]
pub struct ActionInput {
    pub label: u8,
    #[serde(default)]
    pub to_total: u32,
}

/// A postflop decision request. `board` = 3/4/5 cards (0..51). `hero_cards` =
/// the hero's 2 hole cards. `partner_cards` (+ `partner_idx`) enable PAIR MODE:
/// the partner's cards are blocked from the pool's range. `live` = players still
/// in; `hero_idx`/`partner_idx` index the live set (0..live). `commit_entry`/
/// `pot_entry` = the matched commit / total pot ENTERING the current street.
/// `cell` routes the blueprint (the flop-entry live/commit/pot). `street_actions`
/// = the ordered actions THIS street.
#[derive(Deserialize, Default)]
pub struct DecideRequest {
    pub board: Vec<u8>,
    pub hero_cards: [u8; 2],
    #[serde(default)]
    pub partner_cards: Option<[u8; 2]>,
    #[serde(default)]
    pub live: u8,
    #[serde(default)]
    pub hero_idx: u8,
    #[serde(default)]
    pub partner_idx: Option<u8>,
    // Postflop-only (preflop requests — empty board — may omit these).
    #[serde(default)]
    pub commit_entry: u32,
    #[serde(default)]
    pub pot_entry: u32,
    #[serde(default)]
    pub street_actions: Vec<ActionInput>,
    /// Blueprint routing (postflop): the flop-entry cell dir (e.g.
    /// `live3_c6_p20_b15`) and the canonical flop id. The runtime — which tracks the
    /// hand — supplies these; the server loads `{bp_root}/{cell_dir}/flop_{id:04}.bp`
    /// (cached). (Board→canonical-flop routing is a later refinement.)
    #[serde(default)]
    pub cell_dir: String,
    #[serde(default)]
    pub flop_id: u32,
    /// Postflop betting on PRIOR streets (flop[, turn]) in order — for the connected
    /// blueprint, which replays the whole-postflop cell tree from the flop root to a
    /// turn/river node. Empty for flop decisions / the street-local search path. The
    /// runtime (which tracks the hand) supplies it for turn/river.
    #[serde(default)]
    pub prior_actions: Vec<ActionInput>,
    /// Optional RNG seed for the (deterministic) action sample.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Amount the hero must put in to call THIS decision (chips), for the live-6
    /// equity-rollout path which has no betting tree to derive it from. 0 / unset
    /// ⇒ unbet (the bot checks). When unset, the live-6 path derives it from
    /// `street_actions` (max opponent to_total this street − hero's).
    #[serde(default)]
    pub to_call: Option<u32>,
    /// Board→canonical-flop routing. When true, the server DERIVES `flop_id` from
    /// the raw `board` (suit-isomorphism canonicalization) and remaps the board +
    /// hero/partner hole cards into the blueprint's canonical suit frame before
    /// searching — so the runtime can send real cards and omit `flop_id`. When
    /// false (default), the caller supplies `flop_id` and pre-canonicalized cards.
    #[serde(default)]
    pub route: bool,
    /// BAYESIAN reach prior (optional): the full PREFLOP action sequence `(label,
    /// to_total)` of the hand. When supplied with `seat_positions`, the postflop
    /// search seeds each opponent's range with the blueprint's Bayesian posterior
    /// conditioned on that seat's preflop line (raiser ≠ caller), instead of the
    /// symmetric continuing-range v1. Empty ⇒ v1 fallback.
    #[serde(default)]
    pub preflop_actions: Vec<ActionInput>,
    /// BAYESIAN reach prior (optional): map from postflop SEAM seat (0..live, the
    /// search's relabeled seats) to BLUEPRINT position (0=SB,1=BB,2=UTG,3=HJ,4=CO,
    /// 5=BTN). Needed because the seam tree is position-agnostic but `preflop_seat_reach`
    /// is position-specific. Must have ≥ `live` entries to enable the Bayesian prior.
    #[serde(default)]
    pub seat_positions: Vec<u8>,
}

#[derive(Serialize, Clone)]
pub struct ActionProb {
    pub label: u8,
    pub action: String,
    /// Total contribution after the action (chips) — what the runtime should post.
    pub amount: i32,
    pub prob: f32,
}

#[derive(Serialize)]
pub struct DecideResponse {
    pub street: String,
    pub live: u8,
    pub actions: Vec<ActionProb>,
    pub chosen: ActionProb,
    pub search_ms: u64,
    pub paired: bool,
}

/// Walk the subgame from the root, replaying `actions`, to the hero's decision
/// node. For a uniquely-labelled child (fold/check/call) match the label; for
/// multiple bet/raise sizes pick the child whose `amount` is closest to the
/// reported total. Returns None if an action can't be matched (malformed state).
fn walk_to_node(tree: &FlatTree, actions: &[ActionInput]) -> Option<usize> {
    let mut node = 0usize;
    for act in actions {
        if tree.nodes[node].is_terminal() || tree.nodes[node].is_chance() {
            return None;
        }
        let children = tree.node_children(node);
        let cands: Vec<usize> = (0..children.len())
            .filter(|&i| tree.nodes[children[i] as usize].action_label == act.label)
            .collect();
        let pick = match cands.len() {
            0 => return None,
            1 => cands[0],
            _ => *cands.iter().min_by_key(|&&i| {
                (tree.nodes[children[i] as usize].amount - act.to_total as i32).abs()
            })?,
        };
        node = children[pick] as usize;
    }
    Some(node)
}

/// Run a POSTFLOP decision: search the current street's subgame (pair-blocked if
/// partner cards given), locate the hero's node, return per-action probabilities
/// + a sampled choice. None on malformed state / unmappable node / blocked board.
/// Run a LIVE-3/4/5 postflop decision. Primary path: the per-street bucketed-
/// continuation search (`decide_postflop_search`). FALLBACK: if the search can't
/// serve the spot — chiefly a TURN/RIVER runout not in the 1×1 bank (the multiway
/// hole) — degrade to the equity-rollout model so the runtime gets a sane check /
/// pot-odds decision instead of a 400. The flop always resolves via search (no
/// runout dependency), so a None there is a genuine malformed-state error, not a
/// runout miss — only turn/river (board ≥ 4) fall back.
pub fn decide_postflop(bp: &Blueprint, req: &DecideRequest, cfg: &SearchCfg) -> Option<DecideResponse> {
    decide_postflop_with_reach(bp, req, cfg, &[])
}

/// `decide_postflop` + per-seat REACH PRIORS for the search's entering ranges
/// (empty = uniform `initial_weight`). The connected-blueprint path supplies the
/// preflop-continuing range here (Pluribus reach-prior).
pub fn decide_postflop_with_reach(bp: &Blueprint, req: &DecideRequest, cfg: &SearchCfg, reach_priors: &[(usize, Vec<f32>)]) -> Option<DecideResponse> {
    // 1. Bucketed blueprint search (the per-cell 1×1-runout strategy). Returns
    //    None on the multiway turn/river hole (an arbitrary runout the blueprint
    //    never solved).
    if let Some(r) = decide_postflop_search(bp, req, cfg, reach_priors) {
        return Some(r);
    }
    // 2. Multiway (live ≥ 3) turn/river: FACTORED full-nh re-solve of the actual
    //    board — a real solve where the blueprint has no cell, ahead of the crude
    //    equity rollout. (HU turn/river is served by decide_live2, not here.)
    if req.board.len() >= 4 && req.live >= 3 {
        if let Some(r) = decide_postflop_resolve(req) {
            return Some(r);
        }
        // 3. Last resort: equity-rollout (check / pot-odds call-fold).
        return decide_live6(req);
    }
    if req.board.len() >= 4 {
        return decide_live6(req);
    }
    None
}

/// MULTIWAY (live ≥ 3) turn/river FACTORED full-nh re-solve fallback. Solves the
/// ACTUAL board with the rich M2 menu, valued by the factored O(nh·2^K) showdown
/// (`solve_multiway_street`), and reads the hero's average strategy at the
/// decision node. This covers the multiway turn/river hole left by the
/// 1×1-runout blueprint with a real strategy instead of a check/pot-odds rollout.
/// Uniform entering ranges (a real-time prior); the factored showdown is an
/// independent-opponent approximation (<1% of pot per-hand EV vs exact).
pub fn decide_postflop_resolve(req: &DecideRequest) -> Option<DecideResponse> {
    let t0 = std::time::Instant::now();
    if req.live < 3 || (req.board.len() != 4 && req.board.len() != 5) {
        return None;
    }
    let (commit, pot) = (req.commit_entry as i32, req.pot_entry as i32);
    let iters = if req.board.len() == 5 {
        crate::live2_bank::LIVE2_RT_RIVER_ITERS
    } else {
        crate::live2_bank::LIVE2_RT_TURN_ITERS
    };
    let solve = crate::live2_bank::solve_multiway_street(
        &req.board,
        req.live,
        commit,
        pot,
        iters,
        crate::live2_bank::LIVE2_RT_BUDGET_MS,
    )?;
    let node = walk_to_node(&solve.tree, &req.street_actions)?;
    if solve.tree.nodes[node].player_id as usize != req.hero_idx as usize {
        return None; // hero is not the acting player at this node
    }
    // Hero's hand index in the table layout (c1 < c2, not on board).
    let (a, b) = (
        req.hero_cards[0].min(req.hero_cards[1]),
        req.hero_cards[0].max(req.hero_cards[1]),
    );
    let h = (0..solve.nh)
        .find(|&i| solve.hand_cards[i * 2] == a && solve.hand_cards[i * 2 + 1] == b)?;
    let na = solve.tree.nodes[node].num_children as usize;
    let strat = solve.cfr.get_average_strategy(node, na, solve.nh); // [na][nh]
    let children = solve.tree.node_children(node);
    let actions: Vec<ActionProb> = (0..na)
        .map(|a| {
            let child = children[a] as usize;
            let label = solve.tree.nodes[child].action_label;
            let amount = commit + solve.tree.get_contribution(child, req.hero_idx);
            ActionProb { label, action: action_name(label).to_string(), amount, prob: strat[a][h] }
        })
        .collect();
    let street = if req.board.len() == 5 { "river" } else { "turn" };
    Some(finalize_decision(actions, street, req.live, false, req.seed, t0))
}

fn decide_postflop_search(
    bp: &Blueprint,
    req: &DecideRequest,
    cfg: &SearchCfg,
    reach_priors: &[(usize, Vec<f32>)],
) -> Option<DecideResponse> {
    let t0 = std::time::Instant::now();
    let blockers: Vec<u8> = req.partner_cards.map(|c| c.to_vec()).unwrap_or_default();
    let paired = req.partner_cards.is_some();
    let (tree, strat) = search_decision(
        bp,
        &req.board,
        req.live as usize,
        req.hero_idx as usize,
        req.partner_idx.map(|x| x as usize),
        &blockers,
        req.commit_entry as i32,
        req.pot_entry as i32,
        cfg,
        reach_priors,
    )?;
    let node = walk_to_node(&tree, &req.street_actions)?;
    // The hero must be the acting player at this node.
    if tree.nodes[node].player_id as usize != req.hero_idx as usize {
        return None;
    }
    let h = hand_index(bp, req.hero_cards)?;
    let na = tree.nodes[node].num_children as usize;
    let s = strat.get(&node)?;
    let children = tree.node_children(node);
    let mut actions: Vec<ActionProb> = (0..na)
        .map(|a| {
            let child = children[a] as usize;
            let label = tree.nodes[child].action_label;
            // The hero's TOTAL contribution after the action = the matched commit
            // entering this street + what this action adds (the subgame tracks
            // per-street contributions from 0). The runtime posts the delta from
            // the hero's current street contribution.
            let amount = req.commit_entry as i32 + tree.get_contribution(child, req.hero_idx);
            ActionProb { label, action: action_name(label).to_string(), amount, prob: s[a][h] }
        })
        .collect();
    // Sample a choice (deterministic given seed).
    let mut rng = req.seed.unwrap_or(0xA17C0DE);
    let mut x = (splitmix64(&mut rng) % 1_000_000) as f32 / 1_000_000.0;
    let mut sel = na - 1;
    for (a, ap) in actions.iter().enumerate() {
        if x < ap.prob {
            sel = a;
            break;
        }
        x -= ap.prob;
    }
    // Normalize displayed probs (guard tiny drift).
    let z: f32 = actions.iter().map(|a| a.prob).sum();
    if z > 0.0 {
        for a in actions.iter_mut() {
            a.prob /= z;
        }
    }
    let street = match req.board.len() {
        3 => "flop",
        4 => "turn",
        5 => "river",
        _ => "?",
    };
    Some(DecideResponse {
        street: street.to_string(),
        live: req.live,
        chosen: actions[sel].clone(),
        actions,
        search_ms: t0.elapsed().as_millis() as u64,
        paired,
    })
}

/// The live-2 SPR-bin rep (commit, pot) used to build that bin's HU seam tree.
/// Parsed once per `live2_root` from `manifest.txt` (lines `S{bin} commit=C
/// pot=P dir=...`). The bank solves ONE rep cell per SPR bin, so a decision must
/// rebuild the tree with the REP commit/pot (not the live game's) or the blob's
/// buffers won't fit.
fn live2_bin_rep(live2_root: &str, bin: i64) -> Option<(i32, i32)> {
    let text = std::fs::read_to_string(format!("{live2_root}/manifest.txt")).ok()?;
    for line in text.lines() {
        if !line.starts_with('S') {
            continue;
        }
        let b: i64 = match line.split_whitespace().next()?.trim_start_matches('S').parse() {
            Ok(b) => b,
            Err(_) => continue,
        };
        if b != bin {
            continue;
        }
        let get = |k: &str| line.split_whitespace().find_map(|t| t.strip_prefix(k)?.parse::<i32>().ok());
        return Some((get("commit=")?, get("pot=")?));
    }
    None
}

/// Sample a choice + normalize + wrap into a live-2 DecideResponse (shared by the
/// flop-bank and turn/river-search paths).
fn finalize_live2(
    actions: Vec<ActionProb>,
    street: &str,
    seed: Option<u64>,
    t0: std::time::Instant,
) -> DecideResponse {
    finalize_decision(actions, street, 2, false, seed, t0)
}

/// Sample a (deterministic-given-seed) chosen action from `actions`, normalize
/// the displayed probabilities, and assemble the response. Shared by the live-2
/// real-time path and the multiway factored re-solve fallback.
fn finalize_decision(
    mut actions: Vec<ActionProb>,
    street: &str,
    live: u8,
    paired: bool,
    seed: Option<u64>,
    t0: std::time::Instant,
) -> DecideResponse {
    let na = actions.len();
    let mut rng = seed.unwrap_or(0xA17C0DE);
    let mut x = (splitmix64(&mut rng) % 1_000_000) as f32 / 1_000_000.0;
    let mut sel = na.saturating_sub(1);
    for (a, ap) in actions.iter().enumerate() {
        if x < ap.prob {
            sel = a;
            break;
        }
        x -= ap.prob;
    }
    let z: f32 = actions.iter().map(|a| a.prob).sum();
    if z > 0.0 {
        for a in actions.iter_mut() {
            a.prob /= z;
        }
    }
    DecideResponse {
        street: street.to_string(),
        live,
        chosen: actions[sel.min(na.saturating_sub(1))].clone(),
        actions,
        search_ms: t0.elapsed().as_millis() as u64,
        paired,
    }
}

/// Run a LIVE-2 (heads-up) postflop decision. FLOP: banked exact HU strategy
/// (`.bp2`, SPR bin × canonical flop) — a lookup, runout-independent. TURN/RIVER:
/// the bank is 1×1 (one seeded runout) so it can't serve an arbitrary board; instead
/// we SEARCH the actual board in real time — an exact HU subgame solve (no
/// abstraction, plays to exact showdown): river ≈100ms, turn ≈5s (see
/// `solve_live2_street`). So heads-up now has a rich-sizing decision on every street.
pub fn decide_live2(live2_root: &str, req: &DecideRequest) -> Option<DecideResponse> {
    let t0 = std::time::Instant::now();

    // TURN (4) / RIVER (5): real-time exact HU search of the actual board. No bank,
    // no SPR routing, no canonicalization — the solve is exact on the real cards.
    if req.board.len() == 4 || req.board.len() == 5 {
        let (commit, pot) = (req.commit_entry as i32, req.pot_entry as i32);
        let iters = if req.board.len() == 5 {
            crate::live2_bank::LIVE2_RT_RIVER_ITERS
        } else {
            crate::live2_bank::LIVE2_RT_TURN_ITERS
        };
        let solve = crate::live2_bank::solve_live2_street(&req.board, commit, pot, iters)?;
        let node = walk_to_node(&solve.tree, &req.street_actions)?;
        if solve.tree.nodes[node].player_id as usize != req.hero_idx as usize {
            return None;
        }
        // Hero's hand index in the table layout (c1<c2, not on board).
        let (a, b) = (
            req.hero_cards[0].min(req.hero_cards[1]),
            req.hero_cards[0].max(req.hero_cards[1]),
        );
        let h = (0..solve.nh)
            .find(|&i| solve.hand_cards[i * 2] == a && solve.hand_cards[i * 2 + 1] == b)?;
        let na = solve.tree.nodes[node].num_children as usize;
        let strat = solve.cfr.get_average_strategy(node, na, solve.nh); // [na][nh]
        let children = solve.tree.node_children(node);
        let actions: Vec<ActionProb> = (0..na)
            .map(|a| {
                let child = children[a] as usize;
                let label = solve.tree.nodes[child].action_label;
                // The tree is built in the live commit/pot frame → exact amounts.
                let amount = commit + solve.tree.get_contribution(child, req.hero_idx);
                ActionProb { label, action: action_name(label).to_string(), amount, prob: strat[a][h] }
            })
            .collect();
        let street = if req.board.len() == 5 { "river" } else { "turn" };
        return Some(finalize_live2(actions, street, req.seed, t0));
    }

    if req.board.len() != 3 {
        return None;
    }
    let spec = production_game_v1();
    let (commit, pot) = (req.commit_entry as i32, req.pot_entry as i32);
    let behind = spec.stack - commit;
    if behind <= 0 || pot <= 0 {
        return None; // all-in / degenerate → equity rollout, not banked
    }
    // SPR bin (replicates SeamCell::bucket_key) → rep cell for the tree shape.
    let bin = ((behind as f64 / pot as f64).log2() / 0.25).floor() as i64;
    let (rep_commit, rep_pot) = live2_bin_rep(live2_root, bin)?;

    // Canonicalize the flop locally (works whether or not the caller pre-routed):
    // derive flop_id + the perm to bring the hero's hole cards into the bank frame.
    let raw_flop = [req.board[0], req.board[1], req.board[2]];
    let canon = canonicalize_flop(raw_flop);
    let fi = canon_flop_id(canon)? as usize;
    let perm = suit_perm_to_canonical(raw_flop, canon)?;
    let hero = [remap_card(req.hero_cards[0], &perm), remap_card(req.hero_cards[1], &perm)];

    // Rebuild the HU seam tree on the REP cell (with the SAME bet menu the bank was
    // filled with), load the banked strategy blob.
    let tree = build_tree(&spec.flop_seam_config(2, rep_commit, rep_pot, crate::live2_bank::live2_bet_menu())).ok()?;
    let path = format!("{live2_root}/S{bin}/flop_{fi:04}.bp2");
    let canon_cards: [u8; 3] = canon;
    let solver = load_live2(&path, canon_cards, fi, &tree).ok()?;

    // Hero's hand index in the full-nh layout (skip-flop, c1<c2 — identical to the
    // table's hand order, so it indexes the solver's strategy directly).
    let (a, b) = (hero[0].min(hero[1]), hero[0].max(hero[1]));
    let layout = flop_combo_layout(canon);
    let h = layout.iter().position(|&(x, y)| x == a && y == b)?;

    // Walk this street's betting to the hero's node, read the average strategy.
    let node = walk_to_node(&tree, &req.street_actions)?;
    if tree.nodes[node].player_id as usize != req.hero_idx as usize {
        return None;
    }
    let na = tree.nodes[node].num_children as usize;
    let dist = solver.avg_action_dist(node, na, None, None, h);
    let children = tree.node_children(node);
    // Bets are pot-relative; the rep pot ≈ the live pot within the SPR bin but not
    // exactly, so rescale the contribution into the live game's chip frame.
    let scale = pot as f32 / rep_pot.max(1) as f32;
    let actions: Vec<ActionProb> = (0..na)
        .map(|a| {
            let child = children[a] as usize;
            let label = tree.nodes[child].action_label;
            let rep_contrib = tree.get_contribution(child, req.hero_idx);
            let amount = commit + (rep_contrib as f32 * scale).round() as i32;
            ActionProb { label, action: action_name(label).to_string(), amount, prob: dist[a] }
        })
        .collect();
    Some(finalize_live2(actions, "flop", req.seed, t0))
}

/// Number of MC showdown samples for the live-6 equity estimate.
const LIVE6_EQUITY_SAMPLES: usize = 20_000;

/// Run a LIVE-6 (full-ring) postflop decision via the EQUITY-ROLLOUT model: the
/// six-way game has no per-cell blueprint (the postflop tree is valued by check-
/// down showdown equity, not solved). Faithful to that model: the bot never bets
/// (no betting strategy exists), so when UNBET it CHECKS; when FACING A BET it
/// calls or folds by pot-odds vs its Monte-Carlo all-in equity against the field.
/// Works on flop/turn/river (rolls out the remaining board). No blueprint needed.
pub fn decide_live6(req: &DecideRequest) -> Option<DecideResponse> {
    let t0 = std::time::Instant::now();
    if req.board.len() < 3 || req.board.len() > 5 {
        return None;
    }
    let live = req.live as usize;
    if live < 2 {
        return None;
    }
    // Amount to call: explicit (the contract for live-6), else best-effort from
    // this street's actions = the largest opponent to_total seen.
    let to_call = req
        .to_call
        .unwrap_or_else(|| req.street_actions.iter().map(|a| a.to_total).max().unwrap_or(0));

    let seed = req.seed.unwrap_or(0xA17C0DE);
    let equity = crate::eqr::allin_equity_on_board(
        req.hero_cards,
        &req.board,
        live,
        LIVE6_EQUITY_SAMPLES,
        seed,
    );

    let street = match req.board.len() {
        3 => "flop",
        4 => "turn",
        _ => "river",
    };
    let commit = req.commit_entry as i32;
    let pot = req.pot_entry as i32;

    let actions: Vec<ActionProb> = if to_call == 0 {
        // Unbet: the rollout model checks down.
        vec![ActionProb {
            label: 1,
            action: "check".into(),
            amount: commit,
            prob: 1.0,
        }]
    } else {
        // Facing a bet: pot-odds break-even = to_call / (pot + to_call). Call iff the
        // hero's equity clears it (pure decision — the rollout model has no bluffs).
        let req_eq = to_call as f32 / (pot + to_call as i32).max(1) as f32;
        let call = equity >= req_eq;
        vec![
            ActionProb {
                label: 0,
                action: "fold".into(),
                amount: commit,
                prob: if call { 0.0 } else { 1.0 },
            },
            ActionProb {
                label: 2,
                action: "call".into(),
                amount: commit + to_call as i32,
                prob: if call { 1.0 } else { 0.0 },
            },
        ]
    };
    let chosen = actions.iter().max_by(|a, b| a.prob.total_cmp(&b.prob))?.clone();
    Some(DecideResponse {
        street: street.to_string(),
        live: req.live,
        actions,
        chosen,
        search_ms: t0.elapsed().as_millis() as u64,
        paired: false,
    })
}

/// Run a PREFLOP decision (board empty): replay the preflop betting on the EQR
/// player's cap-3 tree to the hero's node and return the action distribution for
/// the hero's 169-class hand. None if the node can't be mapped.
pub fn decide_preflop(pf: &PreflopPlayer, req: &DecideRequest) -> Option<DecideResponse> {
    let t0 = std::time::Instant::now();
    let tree = &pf.tree;
    let node = walk_to_node(tree, &req.street_actions)?;
    if tree.nodes[node].is_terminal() || tree.nodes[node].is_chance() {
        return None;
    }
    let hand_class = PreflopPlayer::hand_class(req.hero_cards[0], req.hero_cards[1]);
    let mut out = [0f32; MAX_NA_PREFLOP];
    let na = pf.action_dist(node, hand_class, &mut out);
    let children = tree.node_children(node);
    let acting = tree.nodes[node].player_id; // the player to act at this node = hero
    let mut actions: Vec<ActionProb> = (0..na)
        .map(|a| {
            let child = children[a] as usize;
            let label = tree.nodes[child].action_label;
            // Hero's TOTAL chips in after the action (preflop contributions include
            // posted blinds), so the runtime knows the raise-to / call amount.
            let amount = tree.get_contribution(child, acting);
            ActionProb { label, action: action_name(label).to_string(), amount, prob: out[a] }
        })
        .collect();
    let mut rng = req.seed.unwrap_or(0xA17C0DE);
    let mut x = (splitmix64(&mut rng) % 1_000_000) as f32 / 1_000_000.0;
    let mut sel = na.saturating_sub(1);
    for (a, ap) in actions.iter().enumerate() {
        if x < ap.prob {
            sel = a;
            break;
        }
        x -= ap.prob;
    }
    let z: f32 = actions.iter().map(|a| a.prob).sum();
    if z > 0.0 {
        for a in actions.iter_mut() {
            a.prob /= z;
        }
    }
    Some(DecideResponse {
        street: "preflop".to_string(),
        live: req.live,
        chosen: actions[sel.min(na.saturating_sub(1))].clone(),
        actions,
        search_ms: t0.elapsed().as_millis() as u64,
        paired: false,
    })
}
