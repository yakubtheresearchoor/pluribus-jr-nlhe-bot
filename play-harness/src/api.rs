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
use crate::pluribus_play::{hand_index, search_decision, SearchCfg};
use crate::preflop_player::{splitmix64, PreflopPlayer};
use serde::{Deserialize, Serialize};
use solver_core::abstraction::flop_isomorphism::{canonicalize_flop, enumerate_canonical_flops};
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
#[derive(Deserialize)]
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
    /// Optional RNG seed for the (deterministic) action sample.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Board→canonical-flop routing. When true, the server DERIVES `flop_id` from
    /// the raw `board` (suit-isomorphism canonicalization) and remaps the board +
    /// hero/partner hole cards into the blueprint's canonical suit frame before
    /// searching — so the runtime can send real cards and omit `flop_id`. When
    /// false (default), the caller supplies `flop_id` and pre-canonicalized cards.
    #[serde(default)]
    pub route: bool,
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
pub fn decide_postflop(
    bp: &Blueprint,
    req: &DecideRequest,
    cfg: &SearchCfg,
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
