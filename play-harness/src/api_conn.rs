//! Connected-blueprint `/decide` path. Serves decisions from the sharded connected
//! MCCFR blueprint (`blueprint_conn_eqr/`) — a different artifact than the per-cell
//! search blueprint the rest of `api` uses. PREFLOP is a blueprint lookup (the
//! EQR-frozen raise-or-fold preflop the postflop was solved against), POSTFLOP is a
//! real-time depth-limited QRE search over the connected buckets with the
//! preflop-continuing reach-prior (Pluribus: preflop=lookup, postflop=search).

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
    // LRU-BOUNDED per-(flop_id, live) adapter cache. Each adapter is ~100-250MB
    // (per-runout tables + bucketing); an UNBOUNDED map OOM-killed the server in
    // ~3 min under fleet load touching many distinct flops (RSS 4.8->24GB,
    // SIGKILL, launchd restart loop). Values are (adapter, lru_stamp); the
    // second tuple field of the mutex is the stamp counter. Evicted adapters
    // still in use by in-flight solves stay alive via their Arc.
    bp_cache: Mutex<(HashMap<(usize, usize), (Arc<SyncBlueprint>, u64)>, u64)>,
    /// Max cached adapters (CONN_ADAPTER_CACHE, default 24 ~= 3-6GB).
    adapter_cache_cap: usize,
    /// STREET-SOLVE CACHE: one solve serves every decision the hero makes on
    /// that street (Pluribus: solve the round once, play from it). Key includes
    /// hero cards (they BLOCK opponents' ranges in the solve), the street board,
    /// pot state, and a reach-prior hash. LRU (CONN_SOLVE_CACHE, default 16
    /// ≈ 200MB; a live-5 (tree,strat) ≈ 10-15MB). street_actions are READ-time.
    #[allow(clippy::type_complexity)]
    solve_cache: Mutex<(HashMap<(usize, u8, Vec<u8>, u32, u32, u8, [u8; 2], u64), (Arc<(solver_core::tree::flat::FlatTree, HashMap<usize, Vec<Vec<f32>>>)>, u64)>, u64)>,
    solve_cache_cap: usize,
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
    /// maxna)` = the solve config (6, 5, 200, 7 for blueprint_conn_eqr). `gs14_dir` =
    /// the full 49×48 GS14 bucket cache.
    pub fn load(bp_dir: &str, gs14_dir: &str, np: usize, nraises: usize, nb: usize, maxna: usize) -> std::io::Result<Self> {
        // Per-blueprint game economics: install this blueprint's `game.json`
        // (rake rate/cap, ante, stack, blinds) as the PROCESS-WIDE runtime spec —
        // every decision path (seam trees, GPU/CPU terminal rake, all-in sizing)
        // must refine under the SAME game class the blueprint was solved with
        // (stakes vary caps/antes/depth). Legacy dirs without a manifest fall
        // back to production_game_v1 (the class every pre-manifest blueprint
        // was solved under) with a WARNING so the gap is visible.
        match solver_core::tree::action::GameSpec::load_from_dir(bp_dir) {
            Some(Ok(spec)) => {
                let installed = crate::runtime_spec::set_runtime_game_spec(spec.clone());
                eprintln!(
                    "loaded game spec from {bp_dir}/game.json: stack={}u rake={}%/cap {}u ante={} {}",
                    spec.stack, spec.rake_rate * 100.0, spec.rake_cap, spec.ante,
                    if installed { "(installed)" } else { "(runtime spec already set — IGNORED; one process, one game class)" }
                );
            }
            Some(Err(e)) => {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{bp_dir}/game.json: {e}")));
            }
            None => {
                eprintln!("WARNING: {bp_dir}/game.json missing — assuming production_game_v1 economics (legacy blueprint)");
            }
        }
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
            bp_cache: Mutex::new((HashMap::new(), 0)),
            adapter_cache_cap: std::env::var("CONN_ADAPTER_CACHE").ok().and_then(|s| s.parse().ok()).unwrap_or(24).max(1),
            solve_cache: Mutex::new((HashMap::new(), 0)),
            solve_cache_cap: std::env::var("CONN_SOLVE_CACHE").ok().and_then(|s| s.parse().ok()).unwrap_or(16).max(1),
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
        let stack = crate::runtime_spec::runtime_game_spec().stack;
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

    /// The loaded sharded blueprint (for preflop-line construction / inspection).
    pub fn blueprint(&self) -> &ShardedConnBlueprint {
        &self.bp
    }

    /// Reach priors for a cell, building the adapter (test/inspection helper).
    pub fn reach_priors_for_cell(
        &self,
        flop_id: usize,
        live: usize,
        req: &DecideRequest,
    ) -> Option<Vec<(usize, Vec<f32>)>> {
        let adapter = self.adapter(flop_id, live)?;
        Some(self.reach_priors(req, &adapter.0))
    }

    /// Build (or fetch cached) the per-flop `Blueprint` ADAPTER for the search: the
    /// GS14 buckets + full 49×48 runout tables + FlopStartGame ranges, from the
    /// connected GS14 cache. `cum_*` are empty (the search re-solves; it reads only
    /// bucketing + initial ranges). Cached per (flop_id, live).
    fn adapter(&self, flop_id: usize, live: usize) -> Option<Arc<SyncBlueprint>> {
        let key = (flop_id, live);
        {
            let mut cache = self.bp_cache.lock().unwrap();
            let (map, stamp) = &mut *cache;
            if let Some((b, s)) = map.get_mut(&key) {
                *stamp += 1;
                *s = *stamp;
                return Some(b.clone());
            }
        }
        let canonical = *self.canonical_flops.get(flop_id)?;
        // Adapter runout for the continuation SHOWDOWN tables (MC-sampled by the
        // search anyway, so full 49×48 is overkill + ~37s/flop cold). Reduced grid
        // (CONN_ADAPTER_RUNOUT, default 12) keeps full-fidelity BUCKETS (subset of
        // the 49×48 cache) at ~budget cold-build cost. Cached per flop after.
        let ar: usize = std::env::var("CONN_ADAPTER_RUNOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
        // live >= 5: FULL 49-turn grid (12 sampled rivers/turn). The quantile
        // bucketing needs no GS14 subset, so the full turn set is affordable —
        // and it makes EVERY real turn card exact in the tree (no nearest-turn
        // mapping), unblocking the live-5 TURN search. Flop continuation
        // integrates more turns as a bonus.
        let (nt, nr) = if live >= 5 { (49usize, ar) } else { (ar, ar) };
        let (turns, river_decks) = runout_grid(canonical, nt, nr);
        let turns_u8: Vec<u8> = turns.iter().map(|&c| c as u8).collect();
        let rivers: Vec<Vec<u8>> = turns_u8.iter().map(|&tc| river_decks[tc as usize].clone()).collect();
        let table = FlopChanceTable::build_full_nh_sampled(canonical, live as u8, &turns_u8, &river_decks);
        let nh = table.num_valid;
        let game = FlopStartGame::new(table);
        // NAMED APPROXIMATION (live ≥ 5): coarse nb=16 QUANTILE continuation
        // instead of the nb=200 GS14 buckets. Continuation evaluation scales
        // with nb — at np=5 the nb=200 search MEASURED ~2s/iter (32 iters would
        // blow the budget ~12×); nb=16 measures 159 ms/iter ⇒ 32 iters ≈ 5.1s.
        // The in-street solve stays LOSSLESS full-nh either way; only the
        // depth-limit leaf values coarsen (B-ladder-studied knob; CONN_L5_NB).
        let (bk, nb) = if live >= 5 {
            let l5nb: usize = std::env::var("CONN_L5_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
            (FlopBucketing::quantile(game.table(), l5nb), l5nb)
        } else {
            let full = load_gs14_cache(
                &gs14_cache_path(&self.gs14_dir, flop_id, self.nb, self.bnt, self.bnr),
                self.nb, self.bnt, self.bnr,
            )?;
            let (fm, tm, rm) = solver_core::blueprint::subset_gs14(&full, canonical, self.bnt, self.bnr, ar, ar);
            (FlopBucketing::from_maps(game.table(), self.nb, self.nb, self.nb, fm, tm, rm), self.nb)
        };
        let blueprint = Blueprint {
            flop: [canonical[0] as u8, canonical[1] as u8, canonical[2] as u8],
            turns: turns_u8, rivers, np: live, nb, nh,
            cum_flop: vec![], cum_turn: vec![], cum_river: vec![], bk, game,
        };
        let arc = Arc::new(SyncBlueprint(blueprint));
        {
            let mut cache = self.bp_cache.lock().unwrap();
            let (map, stamp) = &mut *cache;
            while map.len() >= self.adapter_cache_cap {
                // Evict least-recently-used (O(cap) scan; cap is small).
                let Some((&old, _)) = map.iter().min_by_key(|(_, (_, s))| *s) else { break };
                map.remove(&old);
            }
            *stamp += 1;
            map.insert(key, (arc.clone(), *stamp));
        }
        Some(arc)
    }

    /// POSTFLOP decision via real-time depth-limited search over the connected
    /// blueprint's buckets (Pluribus-style: postflop = search, not lookup). The
    /// connected lookup (preflop + postflop_action_dist) is the baseline/continuation.
    pub fn decide_postflop_search(&self, req: &DecideRequest) -> Option<DecideResponse> {
        // REAL-TIME SEARCH IS live ≤ 5. The blueprint's 5-6-way cells are ONE
        // shared cell per live count (single 1×pot bet, no raises, cap 1, SPR
        // collapsed, solved as "~never reached") — but vs THIS loose pool
        // multiway pots are COMMON, so live-5 gets a real CPU search at the
        // actual (commit, pot) with the rich menu. MEASURED: 159 ms/iter ⇒ 32
        // iters ≈ 5.1s (GPU stays off at np=5 — no K≥4 fast terminal; watchdog).
        // live-6 stays on lookup/rollout: MEASURED 695 ms/iter — even 16 iters
        // ≈ 11s, over budget for too little convergence.
        if req.live > 6 {
            return None;
        }
        // live-6: search ONLY when FACING a bet/raise — that's where the equity
        // rollout loses money (naive pot-odds, never raises). First-to-act 6-way
        // range-checks at equilibrium (confirmed by the live-5 solves), so the
        // instant lookup serves it. Flop + turn only (the 49-turn adapter covers
        // live>=6 too); LEAN menu via bets_for_live (MEASURED 203 ms/iter vs 695
        // rich ⇒ 24 iters ≈ 4.9s).
        // live-6: search only FACING a bet/raise (first-in range-checks at
        // equilibrium — the lookup serves it instantly). Lean menu (203 ms/iter),
        // flop+turn. The off-path defect chain is closed by SUBGAME ROOTING
        // (observed prefix frozen to prob 1) + the HERO-SEAT PRIOR FLOOR in
        // reach_priors (zero-prior hands got no cfv anywhere ⇒ QRE uniform;
        // probe-validated: trash folds 0.999, trips raise 0.75 facing pot bets).
        if req.live == 6 {
            let facing = req.street_actions.iter().any(|a| matches!(a.label, 3 | 4 | 5));
            if !facing || req.board.len() > 4 {
                return None;
            }
        }
        // RIVER at live-5/6: EXACT single-street re-solve (factored multiway
        // showdown on the KNOWN 5-card board — no continuation, no runout grid).
        // MEASURED: ~48 ms/iter np=5, ~224 ms/iter np=6 ⇒ right-sized iters fit
        // the budget (CONN_RIVER_ITERS_L5/L6). live-6 first-in stays on the
        // instant lookup (range-checks at equilibrium — same policy as the flop).
        // None (walk/hero mismatch) falls through to the lookup as before.
        // live-5 RIVER: exact rooted+ranged re-solve (P0 fixed: folded-mass +
        // dead-money in the factored showdown). live-6 river stays lookup.
        if req.board.len() == 5 && req.live == 5 {
            let (it, bd) = (std::env::var("CONN_RIVER_ITERS_L5").ok().and_then(|s| s.parse().ok()).unwrap_or(120u32), 8_000u128);
            let live = req.live as usize;
            let seat_reach = if !req.preflop_actions.is_empty() && req.seat_positions.len() >= live {
                let history: Vec<(u8, i32)> = req.preflop_actions.iter().map(|a| (a.label, a.to_total as i32)).collect();
                Some(self.bp.preflop_seat_reach(&history))
            } else { None };
            let mut class_cont: HashMap<usize, f32> = HashMap::new();
            let mut ranges: Vec<Vec<f32>> = (0..live).map(|seat| {
                (0..1326usize).map(|hi| {
                    let (c1, c2) = solver_core::card::index_to_card_pair(hi);
                    let cls = PreflopClass::from_combo(c1 as Card, c2 as Card).index();
                    match &seat_reach {
                        Some(sr) => {
                            let pos = (req.seat_positions[seat] as usize).min(sr.len() - 1);
                            sr[pos][cls]
                        }
                        None => *class_cont.entry(cls).or_insert_with(|| {
                            self.bp.preflop_action_dist((c1 as Card, c2 as Card), &[])
                                .map(|d| d.iter().filter(|(l, _, _)| *l != 0).map(|(_, _, p)| p).sum())
                                .unwrap_or(1.0)
                        }),
                    }
                }).collect()
            }).collect();
            let hero = (req.hero_idx as usize).min(live.saturating_sub(1));
            let mx = ranges[hero].iter().cloned().fold(0.0f32, f32::max);
            let floor = (mx * 1e-3).max(f32::MIN_POSITIVE);
            for r in ranges[hero].iter_mut() { if *r < floor { *r = floor; } }
            return crate::api::decide_postflop_resolve_ranged(req, it, bd, Some(ranges));
        }
        if req.board.len() == 5 && req.live == 6 {
            return None; // live-6 river: lookup (resolve engagement unverified there)
        }
        // live-5 searches FLOP + TURN (the adapter now carries the FULL 49-turn
        // grid, so any real turn card resolves exactly). RIVER stays on the
        // lookup/rollout path: rivers are 12-sampled per turn (a full 49×48
        // adapter is the ~37s/flop cold-build hole), and an off-grid river
        // would fall into decide_postflop_resolve which GRINDS at np=5.
        if req.live == 5 && req.board.len() > 4 {
            return None;
        }
        let flop_id = req.flop_id as usize;
        let adapter = self.adapter(flop_id, req.live as usize)?;
        let reach_priors = self.reach_priors(req, &adapter.0);
        // Tuned base cfg (par+dcfr+iters); QRE λ from CONN_LAMBDA/CONN_OPP_LAMBDA.
        // live-5 trims iterations to its measured budget (CONN_ITERS_L5 overrides).
        let mut cfg = self.cfg;
        if req.live == 5 {
            // GPU serves live-5 (cluster kernels, ~105-140 ms/iter incl
            // continuation) for first-in AND facing-bet (GPU rooting): 48 iters
            // ≈ 5-7s — a 50% convergence raise over the CPU-era 32.
            let l5 = std::env::var("CONN_ITERS_L5").ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(48);
            cfg.iters = cfg.iters.min(l5);
        }
        if req.live == 6 {
            let l6 = std::env::var("CONN_ITERS_L6").ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(24);
            cfg.iters = cfg.iters.min(l6);
        }
        // STREET-SOLVE CACHE: same (hero, street, pot, priors) ⇒ reuse the solved
        // street strategy for later decisions on this street (ms reads).
        let prior_hash = {
            use std::hash::{Hash, Hasher};
            let mut hs = std::collections::hash_map::DefaultHasher::new();
            for a in &req.preflop_actions { (a.label, a.to_total).hash(&mut hs); }
            // ROOTED solves are conditioned on the observed street line — key it.
            for a in &req.street_actions { (a.label, a.to_total).hash(&mut hs); }
            req.seat_positions.hash(&mut hs);
            req.partner_cards.hash(&mut hs);
            req.partner_idx.hash(&mut hs);
            hs.finish()
        };
        let skey = (flop_id, req.live, req.board.clone(), req.commit_entry, req.pot_entry,
                    req.hero_idx, req.hero_cards, prior_hash);
        if let Some(solved) = {
            let mut c = self.solve_cache.lock().unwrap();
            let (map, stamp) = &mut *c;
            map.get_mut(&skey).map(|(s, st)| { *stamp += 1; *st = *stamp; s.clone() })
        } {
            let t0 = std::time::Instant::now();
            if let Some(r) = crate::api::read_street_decision(&adapter.0, &solved.0, &solved.1, req, t0.elapsed().as_millis() as u64) {
                return Some(r);
            }
            // read miss (off-tree action) → fall through to the full chain below.
        } else if let Some((tree, strat)) = crate::api::solve_street(&adapter.0, req, &cfg, &reach_priors) {
            let t0 = std::time::Instant::now();
            let solved = Arc::new((tree, strat));
            {
                let mut c = self.solve_cache.lock().unwrap();
                let (map, stamp) = &mut *c;
                while map.len() >= self.solve_cache_cap {
                    let Some(old) = map.iter().min_by_key(|(_, (_, s))| *s).map(|(k, _)| k.clone()) else { break };
                    map.remove(&old);
                }
                *stamp += 1;
                map.insert(skey, (solved.clone(), *stamp));
            }
            if let Some(r) = crate::api::read_street_decision(&adapter.0, &solved.0, &solved.1, req, t0.elapsed().as_millis() as u64) {
                return Some(r);
            }
        }
        decide_postflop_with_reach(&adapter.0, req, &cfg, &reach_priors)
    }

    /// Per-seat entering ranges for the postflop search. BAYESIAN (Pluribus reach
    /// prior) when the request supplies the preflop line + seat→position map: each
    /// seat's range = the blueprint posterior conditioned on that seat's preflop
    /// actions (raiser ≠ caller). Otherwise the SYMMETRIC v1 "continuing range"
    /// (per-class non-fold mass at the open) for every seat.
    pub fn reach_priors(&self, req: &DecideRequest, bp: &Blueprint) -> Vec<(usize, Vec<f32>)> {
        let mut priors = self.reach_priors_raw(req, bp);
        // HERO-SEAT PRIOR FLOOR (measured 2026-07-02, rooting_probe): a hand
        // with ZERO own-prior gets NO cfv computed anywhere (terminal evals skip
        // zero-reach hands), so QRE reads UNIFORM for it — trash called 6-way
        // pot bets 1/3 of the time. Reality overrides the range model for the
        // HERO: he is holding the hand, whatever the blueprint thinks — floor
        // his own prior with ε·max so every hand trains. Opponents' ranges stay
        // tight (the Bayes value). ε=1e-3 perturbs the equilibrium negligibly.
        let hero = req.hero_idx as usize;
        for (seat, reach) in priors.iter_mut() {
            if *seat != hero {
                continue;
            }
            let mx = reach.iter().cloned().fold(0.0f32, f32::max);
            let floor = (mx * 1e-3).max(f32::MIN_POSITIVE);
            for r in reach.iter_mut() {
                if *r < floor {
                    *r = floor;
                }
            }
        }
        priors
    }

    fn reach_priors_raw(&self, req: &DecideRequest, bp: &Blueprint) -> Vec<(usize, Vec<f32>)> {
        let hc = &bp.game.table().hand_cards;
        let nh = bp.nh;
        let live = req.live as usize;
        if !req.preflop_actions.is_empty() && req.seat_positions.len() >= live {
            // BAYESIAN: blueprint posterior per blueprint position, mapped to seam seats.
            let history: Vec<(u8, i32)> = req
                .preflop_actions
                .iter()
                .map(|a| (a.label, a.to_total as i32))
                .collect();
            let seat_reach = self.bp.preflop_seat_reach(&history); // [position][class]
            return (0..live)
                .map(|seam_seat| {
                    let pos = (req.seat_positions[seam_seat] as usize).min(seat_reach.len() - 1);
                    let pr = &seat_reach[pos];
                    let reach: Vec<f32> = (0..nh)
                        .map(|h| {
                            let (c1, c2) = (hc[h * 2], hc[h * 2 + 1]);
                            pr[PreflopClass::from_combo(c1 as Card, c2 as Card).index()]
                        })
                        .collect();
                    (seam_seat, reach)
                })
                .collect();
        }
        // v1 SYMMETRIC continuing range (per-class non-fold weight at the open node).
        let mut class_cont: HashMap<usize, f32> = HashMap::new();
        let reach: Vec<f32> = (0..nh)
            .map(|h| {
                let (c1, c2) = (hc[h * 2], hc[h * 2 + 1]);
                let cls = PreflopClass::from_combo(c1 as Card, c2 as Card).index();
                *class_cont.entry(cls).or_insert_with(|| {
                    self.bp
                        .preflop_action_dist((c1 as Card, c2 as Card), &[])
                        .map(|d| d.iter().filter(|(l, _, _)| *l != 0).map(|(_, _, p)| p).sum())
                        .unwrap_or(1.0)
                })
            })
            .collect();
        (0..live).map(|s| (s, reach.clone())).collect()
    }

    /// Convergence diagnostic: flop-subgame exploitability vs iteration count for a
    /// cell, using the SAME adapter (buckets + continuation) the runtime search
    /// builds. Returns (iters, exploit_chips, pct_of_pot). Empty if no adapter.
    pub fn flop_exploitability_sweep(
        &self,
        flop_id: usize,
        live: usize,
        commit: i32,
        pot: i32,
        checkpoints: &[u32],
    ) -> Vec<(u32, f32, f32)> {
        let adapter = match self.adapter(flop_id, live) {
            Some(a) => a,
            None => return vec![],
        };
        crate::pluribus_play::flop_search_exploitability_sweep(&adapter.0, commit, pot, &self.cfg, checkpoints)
            .into_iter()
            .map(|(it, chips)| {
                let pct = if pot > 0 { chips / pot as f32 * 100.0 } else { 0.0 };
                (it, chips, pct)
            })
            .collect()
    }

    /// 2-STREET convergence diagnostic (validation of the Pluribus safe-continuation
    /// machinery): flop+turn searched, river continuation. Returns (iters, pct_pot,
    /// tree_nodes). Empty if no adapter.
    pub fn flop_exploitability_sweep_2street(
        &self,
        flop_id: usize,
        live: usize,
        commit: i32,
        pot: i32,
        checkpoints: &[u32],
    ) -> Vec<(u32, f32, usize)> {
        let adapter = match self.adapter(flop_id, live) {
            Some(a) => a,
            None => return vec![],
        };
        crate::pluribus_play::flop_search_exploitability_2street_sweep(&adapter.0, commit, pot, &self.cfg, checkpoints)
            .into_iter()
            .map(|(it, chips, nodes)| {
                let pct = if pot > 0 { chips / pot as f32 * 100.0 } else { 0.0 };
                (it, pct, nodes)
            })
            .collect()
    }

    /// k-continuation probe: run the self-consistent Pluribus k=`k` safe-continuation
    /// FLOP search on a cell and return (root aggression, root strategy, tree nodes).
    /// Compare k=1 vs k=4 to confirm the biased continuations are live (non-inert) and
    /// see their effect on the flop strategy. None if no adapter.
    pub fn flop_k4_probe(
        &self,
        flop_id: usize,
        live: usize,
        commit: i32,
        pot: i32,
        k: usize,
        warm_iters: u32,
        solve_iters: u32,
    ) -> Option<(f32, Vec<Vec<f32>>, usize)> {
        let adapter = self.adapter(flop_id, live)?;
        Some(crate::pluribus_play::flop_search_k4(
            &adapter.0, commit, pot, &self.cfg, k, warm_iters, solve_iters,
        ))
    }

    /// Select the `(live, commit, pot)` cell: exact match, else nearest among
    /// same-live cells by |Δcommit| + |Δpot| (SPR-ish fallback).
    fn select_cell(&self, live: u8, commit: i32, pot: i32) -> Option<(u8, i32, i32)> {
        // SPR-bin the state; if that bin has a cell, pass the state through
        // (postflop_action_dist re-bins it). Else fall back to the nearest same-live
        // rep (which bins to an existing cell).
        let stack = crate::runtime_spec::runtime_game_spec().stack;
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
