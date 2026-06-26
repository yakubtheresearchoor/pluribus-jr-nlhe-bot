//! Connected-blueprint `/decide` path. Serves decisions by LOOKUP from the
//! sharded connected MCCFR blueprint (`blueprint_conn_v1/`) — a different artifact
//! than the per-cell search blueprint the rest of `api` uses. Currently serves
//! PREFLOP and FLOP decisions (the cell root is the flop, so a street-local flop
//! request maps directly). TURN/RIVER need the full postflop betting path, which
//! the street-local `DecideRequest` doesn't carry — a noted API extension.

use crate::api::{action_name, decide_postflop_with_reach, ActionProb, DecideRequest, DecideResponse};
use crate::blueprint::Blueprint;
use crate::pluribus_play::SearchCfg;
use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::blueprint::{
    gs14_cache_path, load_gs14_cache, runout_grid, ConnCellLayout, FlopLayout, ShardedConnBlueprint,
};
use solver_core::card::Card;
use solver_core::solver::bucketed_flop_cfr::FlopBucketing;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::preflop_allin_equity::load_or_build_class_equity_table;
use solver_core::solver::preflop_jam_game::{solve_hu_preflop_jam, PreflopJamCfg};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::solver::preflop_terminal::build_class_blocking_matrix;
use solver_core::tree::action::production_game_v1;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The 169×169 HU all-in equity + non-blocking tables for the preflop jam search.
/// Built once (lazily) and shared read-only.
struct JamTables {
    equity: Vec<f32>,
    blocking: Vec<f32>,
}

/// Preflop jam-subgame config (env-tunable). `enable`+`spr` gate WHEN the search
/// runs; the rest are search params.
struct JamCfg {
    enable: bool,
    spr: f32, // run the jam search only when post-call SPR ≤ this
    samples: usize,
    cache: String,
    iters: u32,
    lambda: f32,
    opp_lambda: f32,
}

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
    /// Preflop jam-subgame (Option A): restore the high-SPR all-in the lean na=8
    /// blueprint omits. Tables built lazily on first low-SPR HU preflop decision.
    jam_cfg: JamCfg,
    jam_tables: Mutex<Option<Arc<JamTables>>>,
    /// One representative `(Card, Card)` combo per preflop class — for the
    /// per-class reach prior (blueprint continuing range).
    rep_combos: Vec<(Card, Card)>,
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
        let env_f = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let env_u = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let jam_cfg = JamCfg {
            // Default ON: the whole point of the lean blueprint was to defer the
            // preflop jam to search. Disable with CONN_PRE_JAM=0.
            enable: std::env::var("CONN_PRE_JAM").map(|v| v != "0").unwrap_or(true),
            spr: env_f("CONN_PRE_JAM_SPR", 2.0),
            samples: env_u("CONN_PRE_EQUITY_SAMPLES", 2000),
            cache: std::env::var("CONN_PRE_EQUITY_CACHE").unwrap_or_else(|_| "preflop_hu_equity.bin".into()),
            iters: cfg.iters.max(256),
            lambda: cfg.lambda,
            opp_lambda: cfg.opp_lambda,
        };
        // One representative combo per class (first card-pair that maps to it).
        let mut rep_combos = vec![(0u8, 1u8); NUM_PREFLOP_CLASSES];
        let mut seen = vec![false; NUM_PREFLOP_CLASSES];
        'outer: for a in 0..52u8 {
            for b in (a + 1)..52u8 {
                let ci = PreflopClass::from_combo(a as Card, b as Card).index();
                if !seen[ci] {
                    seen[ci] = true;
                    rep_combos[ci] = (a as Card, b as Card);
                    if seen.iter().all(|&s| s) { break 'outer; }
                }
            }
        }
        Ok(ConnDecider {
            bp, layout, gs14_dir: gs14_dir.to_string(), nb, bnt: 49, bnr: 48, canonical_flops,
            cfg,
            bp_cache: Mutex::new(HashMap::new()),
            jam_cfg,
            jam_tables: Mutex::new(None),
            rep_combos,
        })
    }

    /// Build (or load from disk) the 169×169 HU equity + blocking tables, cached
    /// in-memory after first use.
    fn jam_tables(&self) -> Arc<JamTables> {
        if let Some(t) = self.jam_tables.lock().unwrap().as_ref() {
            return t.clone();
        }
        let equity = load_or_build_class_equity_table(&self.jam_cfg.cache, self.jam_cfg.samples, 0xC0FFEE);
        let blocking = build_class_blocking_matrix();
        let arc = Arc::new(JamTables { equity, blocking });
        *self.jam_tables.lock().unwrap() = Some(arc.clone());
        arc
    }

    /// Per-class reach prior = the blueprint's preflop CONTINUING (non-fold) mass
    /// for a representative combo of each class. Symmetric v1 (same for hero/opp);
    /// a Bayesian per-node update is a refinement.
    fn preflop_continue_reach(&self) -> Vec<f32> {
        (0..NUM_PREFLOP_CLASSES)
            .map(|cls| {
                let (c1, c2) = self.rep_combos[cls];
                self.bp
                    .preflop_action_dist((c1, c2), &[])
                    .map(|d| d.iter().filter(|(l, _, _)| *l != 0).map(|(_, _, p)| p).sum())
                    .unwrap_or(1.0)
            })
            .collect()
    }

    /// PREFLOP jam-subgame search (Option A): at a low-SPR HU preflop decision,
    /// re-solve with the rich menu + explicit all-in (which the blueprint omits).
    /// Returns None — fall back to the blueprint lookup — when the gate fails
    /// (not HU, high SPR, or the runtime didn't supply the betting state).
    fn decide_preflop_jam(&self, req: &DecideRequest) -> Option<DecideResponse> {
        if !self.jam_cfg.enable || req.live != 2 {
            return None;
        }
        // Synthetic HU state from the runtime-supplied betting state.
        let pot = req.pot_entry as i32;
        let c_hero = req.commit_entry as i32;
        let to_call = req.to_call? as i32;
        if pot <= 0 || to_call <= 0 {
            return None; // no explicit state ⇒ can't root the subgame safely
        }
        let c_opp = c_hero + to_call;
        let dead = pot - c_hero - c_opp;
        if dead < 0 {
            return None;
        }
        let stack = production_game_v1().stack;
        // SPR gate (post-call): the all-in-equity leaf proxy is only valid low.
        let pot_after = (pot + to_call) as f32;
        let rem = (stack - c_opp) as f32;
        if pot_after <= 0.0 || rem <= 0.0 || rem / pot_after > self.jam_cfg.spr {
            return None;
        }
        let t0 = std::time::Instant::now();
        let reach = self.preflop_continue_reach();
        let tables = self.jam_tables();
        let cfg = PreflopJamCfg {
            iters: self.jam_cfg.iters,
            lambda: self.jam_cfg.lambda,
            opp_lambda: self.jam_cfg.opp_lambda,
            plk: 1,
            nraises: 6,
        };
        let hero = (req.hero_cards[0] as Card, req.hero_cards[1] as Card);
        let res = solve_hu_preflop_jam(
            c_hero, c_opp, dead, stack, &reach, &reach, &tables.equity, &tables.blocking, &cfg,
        )?;
        let hcls = PreflopClass::from_combo(hero.0, hero.1).index();
        let z: f32 = res.actions.iter().enumerate().map(|(a, _)| res.strategy[a][hcls]).sum::<f32>().max(1e-12);
        let actions: Vec<ActionProb> = res
            .actions
            .iter()
            .enumerate()
            .map(|(a, &(label, amount))| ActionProb {
                label,
                action: action_name(label).to_string(),
                amount,
                prob: res.strategy[a][hcls] / z,
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
            street: "preflop".to_string(),
            live: req.live,
            chosen: actions[sel].clone(),
            actions,
            search_ms: t0.elapsed().as_millis() as u64,
            paired: req.partner_cards.is_some(),
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
        // REACH-PRIOR (piece 3): per-seat entering ranges = the connected preflop's
        // CONTINUING range (per-class non-fold weight at the open node), not uniform.
        // Same range for all live seats (symmetric v1; position-specific is a
        // refinement). Hands fold trash out → the search solves realistic ranges.
        let hc = &adapter.0.game.table().hand_cards;
        let nh = adapter.0.nh;
        let mut class_cont: HashMap<usize, f32> = HashMap::new();
        let reach: Vec<f32> = (0..nh).map(|h| {
            let (c1, c2) = (hc[h * 2], hc[h * 2 + 1]);
            let cls = PreflopClass::from_combo(c1 as Card, c2 as Card).index();
            *class_cont.entry(cls).or_insert_with(|| {
                self.bp.preflop_action_dist((c1 as Card, c2 as Card), &[])
                    .map(|d| d.iter().filter(|(l, _, _)| *l != 0).map(|(_, _, p)| p).sum())
                    .unwrap_or(1.0)
            })
        }).collect();
        let reach_priors: Vec<(usize, Vec<f32>)> = (0..req.live as usize).map(|s| (s, reach.clone())).collect();
        // Tuned base cfg (par+dcfr+iters); QRE λ from CONN_LAMBDA/CONN_OPP_LAMBDA.
        decide_postflop_with_reach(&adapter.0, req, &self.cfg, &reach_priors)
    }

    /// Select the `(live, commit, pot)` cell: exact match, else nearest among
    /// same-live cells by |Δcommit| + |Δpot| (SPR-ish fallback).
    fn select_cell(&self, live: u8, commit: i32, pot: i32) -> Option<(u8, i32, i32)> {
        // SPR-bin the state; if that bin has a cell, pass the state through
        // (postflop_action_dist re-bins it). Else fall back to the nearest same-live
        // rep (which bins to an existing cell).
        let stack = solver_core::tree::action::production_game_v1().stack;
        let bin = solver_core::blueprint::conn_seam_bin_from(live, commit, pot, stack);
        if self.layout.key_idx.contains_key(&bin) {
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

        // PREFLOP = blueprint lookup, EXCEPT low-SPR HU spots where the jam-subgame
        // search restores the explicit all-in the lean na=8 blueprint omits.
        if req.board.is_empty() {
            if let Some(r) = self.decide_preflop_jam(req) {
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
