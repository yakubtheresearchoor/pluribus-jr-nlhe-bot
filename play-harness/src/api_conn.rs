//! Connected-blueprint `/decide` path. Serves decisions by LOOKUP from the
//! sharded connected MCCFR blueprint (`blueprint_conn_v1/`) — a different artifact
//! than the per-cell search blueprint the rest of `api` uses. Currently serves
//! PREFLOP and FLOP decisions (the cell root is the flop, so a street-local flop
//! request maps directly). TURN/RIVER need the full postflop betting path, which
//! the street-local `DecideRequest` doesn't carry — a noted API extension.

use crate::api::{action_name, decide_postflop, ActionProb, DecideRequest, DecideResponse};
use crate::blueprint::Blueprint;
use crate::pluribus_play::SearchCfg;
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::blueprint::{
    gs14_cache_path, load_gs14_cache, runout_grid, ConnCellLayout, FlopLayout, ShardedConnBlueprint,
};
use solver_core::card::Card;
use solver_core::solver::bucketed_flop_cfr::FlopBucketing;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A `Blueprint` adapter built from the connected GS14 cache. `Blueprint` holds
/// chance `Cell` state (`!Sync`) that the DEPTH-LIMITED search only reads, so
/// sharing it across request threads is sound — same argument as the server's
/// `SyncBp`. Asserted here for the per-flop adapter cache.
struct SyncBlueprint(Blueprint);
unsafe impl Send for SyncBlueprint {}
unsafe impl Sync for SyncBlueprint {}

/// Holds the loaded connected blueprint + reconstructed cell layout + the GS14
/// bucket cache dir. Read-only at decision time — share across request threads.
pub struct ConnDecider {
    bp: ShardedConnBlueprint,
    layout: ConnCellLayout,
    gs14_dir: String,
    nb: usize,
    bnt: usize, // cache runout dims (full 49×48 — buckets are full-fidelity for ANY board)
    bnr: usize,
    canonical_flops: Vec<[Card; 3]>,
    /// Base real-time search config (QRE λ etc); per-decision use `for_live`.
    cfg: SearchCfg,
    /// Per-(flop_id, live) adapter Blueprint cache — built lazily from the GS14
    /// cache (the depth-limited search's bucketing + ranges).
    bp_cache: Mutex<HashMap<(usize, usize), Arc<SyncBlueprint>>>,
}

/// Parse a cell dir `live{N}_c{C}_p{P}_…` into the flop-entry seam `(live, commit,
/// pot)`. None if the string isn't in that form.
fn parse_cell_dir(s: &str) -> Option<(u8, i32, i32)> {
    let mut it = s.split('_');
    let live = it.next()?.strip_prefix("live")?.parse().ok()?;
    let commit = it.next()?.strip_prefix('c')?.parse().ok()?;
    let pot = it.next()?.strip_prefix('p')?.parse().ok()?;
    Some((live, commit, pot))
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl ConnDecider {
    /// Load the sharded blueprint + reconstruct the cell layout. `(np, nraises, nb,
    /// maxna)` = the solve config (6, 1, 200, 3 for blueprint_conn_v1). `gs14_dir` =
    /// the full 49×48 GS14 bucket cache.
    pub fn load(bp_dir: &str, gs14_dir: &str, np: usize, nraises: usize, nb: usize, maxna: usize) -> std::io::Result<Self> {
        let bp = ShardedConnBlueprint::load(bp_dir, np, nraises, nb, 16, 16, maxna)?;
        let layout = ConnCellLayout::build(np, nraises, nb);
        let ranges = vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6];
        let canonical_flops = PreflopChanceTable::new(6, ranges).canonical_flops;
        let mut cfg = SearchCfg::default();
        // DCFR + parallel by default (faster convergence ⇒ fewer iters for a given
        // quality). iters is the latency↔convergence knob (lossless live-3+ search is
        // ~0.1-0.2s/iter; default sized to a few seconds, env-tunable).
        cfg.par = Some(true);
        cfg.dcfr = Some(true);
        cfg.iters = 48;
        if let Some(v) = std::env::var("CONN_ITERS").ok().and_then(|s| s.parse::<u32>().ok()) { cfg.iters = v; }
        if let Some(v) = std::env::var("CONN_SAMPLE_M").ok().and_then(|s| s.parse::<u32>().ok()) { cfg.sample_m = v; }
        if let Some(v) = std::env::var("CONN_LAMBDA").ok().and_then(|s| s.parse::<f32>().ok()) { cfg.lambda = v; }
        if let Some(v) = std::env::var("CONN_OPP_LAMBDA").ok().and_then(|s| s.parse::<f32>().ok()) { cfg.opp_lambda = v; }
        Ok(ConnDecider {
            bp, layout, gs14_dir: gs14_dir.to_string(), nb, bnt: 49, bnr: 48, canonical_flops,
            cfg,
            bp_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Build (or fetch cached) the per-flop `Blueprint` ADAPTER for the search: the
    /// GS14 buckets + full 49×48 runout tables + FlopStartGame ranges, from the
    /// connected GS14 cache. `cum_*` are empty (the search re-solves; it reads only
    /// bucketing + initial ranges). Cached per (flop_id, live).
    fn adapter(&self, flop_id: usize, live: usize) -> Option<Arc<SyncBlueprint>> {
        let key = (flop_id, live);
        if let Some(b) = self.bp_cache.lock().unwrap().get(&key) {
            return Some(b.clone());
        }
        let canonical = *self.canonical_flops.get(flop_id)?;
        // Adapter runout for the continuation SHOWDOWN tables (MC-sampled by the
        // search anyway, so full 49×48 is overkill + ~37s/flop cold). Reduced grid
        // (CONN_ADAPTER_RUNOUT, default 12) keeps full-fidelity BUCKETS (subset of
        // the 49×48 cache) at ~budget cold-build cost. Cached per flop after.
        let ar: usize = std::env::var("CONN_ADAPTER_RUNOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
        let (turns, river_decks) = runout_grid(canonical, ar, ar);
        let turns_u8: Vec<u8> = turns.iter().map(|&c| c as u8).collect();
        let rivers: Vec<Vec<u8>> = turns_u8.iter().map(|&tc| river_decks[tc as usize].clone()).collect();
        let table = FlopChanceTable::build_full_nh_sampled(canonical, live as u8, &turns_u8, &river_decks);
        let nh = table.num_valid;
        let game = FlopStartGame::new(table);
        let full = load_gs14_cache(
            &gs14_cache_path(&self.gs14_dir, flop_id, self.nb, self.bnt, self.bnr),
            self.nb, self.bnt, self.bnr,
        )?;
        let (fm, tm, rm) = solver_core::blueprint::subset_gs14(&full, canonical, self.bnt, self.bnr, ar, ar);
        let bk = FlopBucketing::from_maps(game.table(), self.nb, self.nb, self.nb, fm, tm, rm);
        let blueprint = Blueprint {
            flop: [canonical[0] as u8, canonical[1] as u8, canonical[2] as u8],
            turns: turns_u8, rivers, np: live, nb: self.nb, nh,
            cum_flop: vec![], cum_turn: vec![], cum_river: vec![], bk, game,
        };
        let arc = Arc::new(SyncBlueprint(blueprint));
        self.bp_cache.lock().unwrap().insert(key, arc.clone());
        Some(arc)
    }

    /// POSTFLOP decision via real-time depth-limited search over the connected
    /// blueprint's buckets (Pluribus-style: postflop = search, not lookup). The
    /// connected lookup (preflop + postflop_action_dist) is the baseline/continuation.
    fn decide_postflop_search(&self, req: &DecideRequest) -> Option<DecideResponse> {
        let flop_id = req.flop_id as usize;
        let adapter = self.adapter(flop_id, req.live as usize)?;
        // Use the tuned base cfg directly (par+dcfr+iters) rather than the 200-iter
        // for_live schedule, which was sized for the precomputed-table blueprint.
        decide_postflop(&adapter.0, req, &self.cfg)
    }

    /// Select the `(live, commit, pot)` cell: exact match, else nearest among
    /// same-live cells by |Δcommit| + |Δpot| (SPR-ish fallback).
    fn select_cell(&self, live: u8, commit: i32, pot: i32) -> Option<(u8, i32, i32)> {
        if self.layout.key_idx.contains_key(&(live, commit, pot)) {
            return Some((live, commit, pot));
        }
        self.layout
            .keys
            .iter()
            .filter(|k| k.0 == live)
            .min_by_key(|k| (k.1 - commit).abs() + (k.2 - pot).abs())
            .copied()
    }

    /// The flop-entry seam (cell key) for this hand. From `cell_dir`
    /// (`live{N}_c{C}_p{P}_…`) when supplied — stable across all postflop streets —
    /// else the current-street `(live, commit_entry, pot_entry)` (flop-correct only).
    fn seam(&self, req: &DecideRequest) -> Option<(u8, i32, i32)> {
        if let Some((l, c, p)) = parse_cell_dir(&req.cell_dir) {
            return self.select_cell(l, c, p);
        }
        self.select_cell(req.live, req.commit_entry as i32, req.pot_entry as i32)
    }

    /// Serve a decision. Returns None for spots this path can't serve (turn/river,
    /// unmapped node, missing cache) so the caller can fall back.
    pub fn decide(&self, req: &DecideRequest) -> Option<DecideResponse> {
        let t0 = std::time::Instant::now();
        let hero = (req.hero_cards[0] as Card, req.hero_cards[1] as Card);
        let hist: Vec<(u8, i32)> = req.street_actions.iter().map(|a| (a.label, a.to_total as i32)).collect();

        // POSTFLOP = real-time depth-limited QRE search over the connected buckets
        // (Pluribus: preflop=blueprint lookup, postflop=search). Falls back to the
        // connected lookup below if the search can't serve the spot.
        if (3..=5).contains(&req.board.len()) {
            if let Some(r) = self.decide_postflop_search(req) {
                return Some(r);
            }
        }

        let (street, raw): (&str, Vec<(u8, i32, f32)>) = if req.board.is_empty() {
            ("preflop", self.bp.preflop_action_dist(hero, &hist)?)
        } else if (3..=5).contains(&req.board.len()) {
            // POSTFLOP (flop/turn/river): replay the WHOLE-postflop cell tree from the
            // flop root — prior_actions (earlier streets) ++ this street's actions —
            // to the hero's node, indexing with the node's street bucket. Buckets come
            // from the FULL 49×48 GS14 cache (full-fidelity for ANY actual board card).
            let flop_id = req.flop_id as usize;
            if flop_id >= self.canonical_flops.len() {
                return None;
            }
            let canonical = self.canonical_flops[flop_id];
            let maps = load_gs14_cache(
                &gs14_cache_path(&self.gs14_dir, flop_id, self.nb, self.bnt, self.bnr),
                self.nb, self.bnt, self.bnr,
            )?;
            let fl = FlopLayout::for_canonical(canonical, self.bnt, self.bnr);
            let b = req.board.len();
            let buckets = [
                fl.flop_bucket(&maps, hero)? as usize,
                if b >= 4 { fl.turn_bucket(&maps, hero, req.board[3] as Card)? as usize } else { 0 },
                if b == 5 { fl.river_bucket(&maps, hero, req.board[3] as Card, req.board[4] as Card)? as usize } else { 0 },
            ];
            let post = self.bp.postflop_cum(flop_id).ok()?;
            let (live, commit, pot) = self.seam(req)?;
            // full postflop history: prior streets ++ this street.
            let mut full: Vec<(u8, i32)> = req.prior_actions.iter().map(|a| (a.label, a.to_total as i32)).collect();
            full.extend(hist.iter().copied());
            let street = match b { 3 => "flop", 4 => "turn", _ => "river" };
            (street, self.layout.postflop_action_dist(&post, live, commit, pot, buckets, &full)?)
        } else {
            return None;
        };

        // Build action probs (normalize defensively) + sample a deterministic choice.
        let z: f32 = raw.iter().map(|(_, _, p)| p).sum::<f32>().max(1e-12);
        let actions: Vec<ActionProb> = raw
            .iter()
            .map(|&(label, amount, prob)| ActionProb {
                label,
                action: action_name(label).to_string(),
                amount,
                prob: prob / z,
            })
            .collect();
        if actions.is_empty() {
            return None;
        }
        let mut rng = req.seed.unwrap_or(0xA17C0DE);
        let mut x = (splitmix(&mut rng) % 1_000_000) as f32 / 1_000_000.0;
        let mut sel = actions.len() - 1;
        for (a, ap) in actions.iter().enumerate() {
            if x < ap.prob {
                sel = a;
                break;
            }
            x -= ap.prob;
        }
        Some(DecideResponse {
            street: street.to_string(),
            live: req.live,
            chosen: actions[sel].clone(),
            actions,
            search_ms: t0.elapsed().as_millis() as u64,
            paired: false,
        })
    }
}
