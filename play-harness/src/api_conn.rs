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
            // 2.25 (2026-07-04, was 4.5 for one day): the 4.5 widening made the
            // jam game serve SPR 2.7-4.5 facing-3-bet spots, and its fold/call/
            // JAM menu + equity-only call model overvalues jamming there —
            // MEASURED live: 44/Q9c/JTs all pure-jammed 100bb. Jam-or-fold is
            // only sound at low SPR; deeper spots now go to the trained v5 rows
            // (real 4-bet sizes) or the equity guard (fold/call vs posterior).
            spr: env_f("CONN_PRE_JAM_SPR", 2.25),
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
    /// Hero equity vs the ANALYTIC value-heavy 3-bet prior (top 10% classes by
    /// eq-vs-random at weight 1 + 0.02 junk mix) — the measured pool 3-bets
    /// value-only, so this is the realistic range behind preflop aggression.
    fn eq_vs_3bet_prior(&self, hero: (Card, Card)) -> f32 {
        self.eq_vs_range_prior(hero, 0.10)
    }

    /// Like eq_vs_3bet_prior with a caller-chosen value-range width (e.g. the
    /// villain's measured 3-bet frequency: a 4% nit ⇒ tighter range ⇒ fold
    /// more; a 15% aggro ⇒ wider ⇒ defend/jam wider).
    fn eq_vs_range_prior(&self, hero: (Card, Card), frac: f32) -> f32 {
        let t = self.jam_tables();
        let nc = NUM_PREFLOP_CLASSES;
        let mut avg: Vec<(usize, f32)> = (0..nc)
            .map(|c| {
                let row = &t.equity[c * nc..(c + 1) * nc];
                (c, row.iter().sum::<f32>() / nc as f32)
            })
            .collect();
        avg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut range = vec![0.02f32; nc];
        let take = ((nc as f32) * frac.clamp(0.03, 0.35)) as usize;
        for &(c, _) in avg.iter().take(take.max(3)) {
            range[c] = 1.0;
        }
        let hcls = PreflopClass::from_combo(hero.0, hero.1).index();
        let (mut eq_num, mut den) = (0.0f64, 0.0f64);
        for oc in 0..nc {
            let wgt = (range[oc] as f64)
                * PreflopClass::new(oc as u8).num_combos() as f64
                * t.blocking[hcls * nc + oc] as f64;
            eq_num += wgt * t.equity[hcls * nc + oc] as f64;
            den += wgt;
        }
        if den > 0.0 { (eq_num / den) as f32 } else { 0.35 }
    }

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
        // Effective stack: the client's per-hand value when sent (short tables
        // break every SPR computation at the spec's 200u assumption), else spec.
        let stack = req.eff_stack.map(|s| s as i32)
            .unwrap_or_else(|| crate::runtime_spec::runtime_game_spec().stack);
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
        // JAM RANGE GATE (2026-07-05, client-caught: J9o 5-bet jam into KK): the
        // jam SOLVE models an equilibrium 3-bettor with bluffs that fold to jams;
        // the measured pool 3-bets VALUE-ONLY (river data: 0% air NL10) — no fold
        // equity, and their calling range crushes light jams. If the solve picks
        // jam but hero equity vs the value-heavy prior is below JAM_MIN_EQ
        // (default 0.40 ≈ break-even called-at-SPR≈1.5 with residual FE),
        // downgrade to the guard's priced fold/call.
        if actions[sel].label >= 3 {
            let hero = (req.hero_cards[0] as Card, req.hero_cards[1] as Card);
            // size the assumed 3-bet range by the villain's own measured stat
            // (blended toward the pool 9%) — HU, so the seat is unambiguous.
            let opp_seat = (1 - req.hero_idx.min(1)) as u8;
            let s = req.opponent_stats.iter().find(|o| o.seat_idx == opp_seat);
            let n = s.and_then(|o| o.sample_size).unwrap_or(0);
            let frac = Self::blend(s.and_then(|o| o.three_bet), n, 0.09);
            let eq = self.eq_vs_range_prior(hero, frac);
            let min_eq: f32 = std::env::var("JAM_MIN_EQ").ok().and_then(|s| s.parse().ok()).unwrap_or(0.40);
            if eq < min_eq {
                let price = to_call as f32 / ((pot + to_call) as f32 + to_call as f32);
                let call_ok = eq > price + 0.04;
                eprintln!("[pre-jam] jam GATED (eq {eq:.3} < {min_eq}) -> {} (price {price:.3})",
                    if call_ok { "call" } else { "fold" });
                let (l, amt) = if call_ok { (2u8, c_hero + to_call) } else { (0u8, c_hero) };
                let acts = vec![ActionProb { label: l, action: action_name(l).to_string(), amount: amt, prob: 1.0 }];
                return Some(DecideResponse {
                    street: "preflop".to_string(),
                    live: req.live,
                    chosen: acts[0].clone(),
                    actions: acts,
                    search_ms: t0.elapsed().as_millis() as u64,
                    paired: req.partner_cards.is_some(),
                });
            }
        }
        eprintln!("[pre-jam] served: {} {} (pot {} to_call {:?})",
            actions[sel].action, actions[sel].amount, req.pot_entry, req.to_call);
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
            // WEIGHT-AWARE eviction (hammer 2026-07-02: 24 entries filled with
            // live-5 adapters ≈ 20GB RSS, grazing the ~27GB Metal silent-failure
            // zone). live>=5 adapters hold the full 49-turn grid ≈ 4x a
            // 12-runout live<=4 adapter (measured ratio), so cap total WEIGHT
            // (live<=4 = 1 unit, live>=5 = 4) alongside the entry cap.
            // CONN_ADAPTER_CACHE_UNITS default 20 ≈ 6-7GB of adapters.
            let unit = |live: usize| if live >= 5 { 4usize } else { 1 };
            let cap_units: usize = std::env::var("CONN_ADAPTER_CACHE_UNITS")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(20).max(4);
            let new_units = unit(live);
            loop {
                let used: usize = map.keys().map(|&(_, l)| unit(l)).sum();
                if map.len() < self.adapter_cache_cap && used + new_units <= cap_units {
                    break;
                }
                // Evict least-recently-used (O(cap) scan; cap is small).
                let Some((&old, _)) = map.iter().min_by_key(|(_, (_, s))| *s) else { break };
                map.remove(&old);
            }
            *stamp += 1;
            map.insert(key, (arc.clone(), *stamp));
        }
        Some(arc)
    }

    /// LOOKUP-LANE CLASSIFIER: would `decide_postflop_search` engage for this
    /// request? MUST mirror its gates exactly (kept adjacent on purpose) — the
    /// server uses this to route lookup-served spots around the solve queue
    /// (no admission permit, ms latency). Cheap by construction: no adapter is
    /// built — the off-grid turn test recomputes the deterministic runout grid
    /// (µs) instead of touching the adapter cache.
    pub fn search_engages(&self, req: &DecideRequest) -> bool {
        if !(3..=5).contains(&req.board.len()) {
            return false;
        }
        if req.live > 6 {
            return false;
        }
        if req.live == 6 {
            if std::env::var("CONN_L6_FACING_SEARCH").map(|v| v == "1") != Ok(true) {
                return false;
            }
            let facing = req.street_actions.iter().any(|a| matches!(a.label, 3 | 4 | 5));
            return facing && req.board.len() <= 4;
        }
        if req.board.len() == 5 {
            // live-5 river resolve engages; live<=4 rivers run the street search.
            return true;
        }
        // OFF-GRID live-3/4 turn -> lookup (same rule as the search body; the
        // grid is a pure function of (canonical flop, CONN_ADAPTER_RUNOUT)).
        if req.board.len() == 4 && req.live <= 4 {
            let flop_id = req.flop_id as usize;
            let Some(&canonical) = self.canonical_flops.get(flop_id) else {
                return false;
            };
            let ar: usize = std::env::var("CONN_ADAPTER_RUNOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
            let (turns, _) = runout_grid(canonical, ar, ar);
            return turns.iter().any(|&t| t as u8 == req.board[3]);
        }
        true
    }

    /// POSTFLOP decision via real-time depth-limited search over the connected
    /// blueprint's buckets (Pluribus-style: postflop = search, not lookup). The
    /// connected lookup (preflop + postflop_action_dist) is the baseline/continuation.
    /// Is the hero's CLASS effectively impossible on the observed preflop line
    /// under the solved chart? Off-model hands get the reach FLOOR in the search
    /// — their strategy rows train at ~zero weight and are NOISE (measured live:
    /// QTs open-shoving an A-high flop 3-way, 75o jamming air in a 3-bet pot).
    fn hero_off_model(&self, req: &DecideRequest) -> bool {
        let hero = (req.hero_cards[0] as Card, req.hero_cards[1] as Card);
        let cls = PreflopClass::from_combo(hero.0, hero.1).index();
        let live = req.live as usize;
        if !req.preflop_actions.is_empty() && req.seat_positions.len() >= live {
            let history: Vec<(u8, i32)> = req.preflop_actions.iter().map(|a| (a.label, a.to_total as i32)).collect();
            let sr = self.bp.preflop_seat_reach(&history);
            let hero_seat = (req.hero_idx as usize).min(live.saturating_sub(1));
            let pos = (req.seat_positions[hero_seat] as usize).min(sr.len().saturating_sub(1));
            let r = &sr[pos];
            let mx = r.iter().cloned().fold(0.0f32, f32::max);
            return r[cls] < mx * 1e-3;
        }
        // v1 symmetric: hero class's preflop CONTINUING mass under the chart.
        let cont = self
            .bp
            .preflop_action_dist(hero, &[])
            .map(|d| d.iter().filter(|(l, _, _)| *l != 0).map(|(_, _, p)| p).sum::<f32>())
            .unwrap_or(1.0);
        cont < 1e-3
    }

    /// Exact-strength river pricing vs a pool-calibrated BETTING range on the
    /// actual board: top `value_frac` of villain combos + `bluff_frac` weakest
    /// (the bluff mix). Returns (hero_equity_vs_that_range, price).
    fn river_bet_gate(&self, req: &DecideRequest, to_call: u32, value_frac: f32, bluff_frac: f32) -> Option<(f32, f32)> {
        if req.board.len() != 5 || to_call == 0 {
            return None;
        }
        use solver_core::hand::eval::Hand;
        let b = &req.board;
        let mut base = Hand::new();
        for &c in b.iter() {
            base = base.add_card(c as usize);
        }
        let dead: u64 = b.iter().chain(req.hero_cards.iter()).fold(0u64, |m, &c| m | (1u64 << c));
        let hero_s = base
            .add_card(req.hero_cards[0] as usize)
            .add_card(req.hero_cards[1] as usize)
            .evaluate_internal();
        let mut vs: Vec<i32> = Vec::with_capacity(1081);
        for c1 in 0..52u8 {
            if dead & (1 << c1) != 0 { continue; }
            for c2 in (c1 + 1)..52 {
                if dead & (1 << c2) != 0 { continue; }
                vs.push(base.add_card(c1 as usize).add_card(c2 as usize).evaluate_internal());
            }
        }
        vs.sort_unstable_by(|a, b| b.cmp(a));
        let n = vs.len();
        let value_n = ((n as f32) * value_frac) as usize;
        let bluff_n = ((n as f32) * bluff_frac) as usize;
        let (mut win, mut tot) = (0f64, 0f64);
        for (i, &s) in vs.iter().enumerate() {
            let in_range = i < value_n || i >= n - bluff_n;
            if !in_range { continue; }
            tot += 1.0;
            if hero_s > s { win += 1.0 } else if hero_s == s { win += 0.5 }
        }
        let eq = if tot > 0.0 { (win / tot) as f32 } else { 0.0 };
        let pot_after = (req.pot_entry as f32 + to_call as f32).max(1.0);
        let price = to_call as f32 / (pot_after + to_call as f32);
        Some((eq, price))
    }

    /// POOL EXPLOIT OVERLAY (2026-07-04, POOL_EXPLOIT=0 disables): directional
    /// serve-time corrections for the two measured pool deviations that a QRE
    /// temperature cannot express simultaneously (noise is symmetric; the pool
    /// is DIRECTIONAL — honest big bets, loose calls):
    ///   1. RIVER HONESTY GATE: pool pot-sized river bets are ~10-15% bluffs
    ///      (equilibrium assumes ~33), so marginal river CALLS are re-priced vs
    ///      a value-weighted betting range and flipped to fold when they lose.
    ///   2. AIR-BLUFF SUPPRESSOR: aggressive actions with pure-air equity
    ///      (<0.28 MC vs random; semi-bluff draws sit above) become check/fold —
    ///      bluffing a station is a donation.
    /// Applied at the single postflop choke point, so search, resolve and lookup
    /// serves all pass through it. Value bets/calls are untouched.
    /// Blend an individual stat toward the pool prior by sample size
    /// (n/(n+K), K=200 hands — under ~200 hands the pool dominates).
    fn blend(stat: Option<f32>, n: u32, pool: f32) -> f32 {
        match stat {
            Some(s) => {
                let w = n as f32 / (n as f32 + 200.0);
                w * s + (1.0 - w) * pool
            }
            None => pool,
        }
    }

    /// Villain profile for a seat: (af, wtsd, fold_to_cbet, allin_freq),
    /// blended toward pool priors (NL10 measured: af 2.27, wtsd .326, ftc .314).
    fn villain_profile(&self, req: &DecideRequest, seat: u8) -> (f32, f32, f32, f32) {
        let s = req.opponent_stats.iter().find(|o| o.seat_idx == seat);
        let n = s.and_then(|o| o.sample_size).unwrap_or(0);
        let af = Self::blend(s.and_then(|o| o.af), n, 2.27);
        let wtsd = Self::blend(s.and_then(|o| o.wtsd), n, 0.326);
        let ftc = Self::blend(s.and_then(|o| o.fold_to_cbet), n, 0.314);
        let ai = Self::blend(s.and_then(|o| o.allin), n, 0.02);
        (af, wtsd, ftc, ai)
    }

    pub fn pool_overlay(&self, req: &DecideRequest, resp: DecideResponse) -> DecideResponse {
        if std::env::var("POOL_EXPLOIT").map(|v| v == "0") == Ok(true) {
            return resp;
        }
        if !(3..=5).contains(&req.board.len()) {
            return resp;
        }
        let to_call = req.to_call.unwrap_or_else(|| {
            req.street_actions.iter().map(|a| a.to_total).max().unwrap_or(0)
        });
        if std::env::var("POOL_DEBUG").is_ok() {
            eprintln!("[pool-dbg] overlay: board {} chosen {}@{} to_call {to_call} pot {}",
                req.board.len(), resp.chosen.label, resp.chosen.amount, req.pot_entry);
        }
        let chosen = &resp.chosen;
        let mk = |label: u8, amount: i32, live: u8, street: &str| -> DecideResponse {
            let actions = vec![ActionProb { label, action: action_name(label).to_string(), amount, prob: 1.0 }];
            DecideResponse { street: street.into(), live, actions: actions.clone(), chosen: actions[0].clone(), paired: false, search_ms: resp.search_ms }
        };
        // 1) river honesty gate on CALLS of meaningful bets (>= 40% pot).
        if req.board.len() == 5 && chosen.label == 2 && to_call > 0 {
            let big = (to_call as f32) >= 0.4 * (req.pot_entry as f32).max(1.0);
            if big {
                // β priority: per-request (stake-measured) > env > default.
                let mut bluff_share: f32 = req.pool_river_bluff.or_else(|| {
                    std::env::var("POOL_RIVER_BLUFF").ok().and_then(|s| s.parse().ok())
                }).unwrap_or(0.12);
                // MANIAC RELAXATION (HU only — multiway bettor identity unknown):
                // a hyper-aggressive opponent's big river bets DO contain bluffs;
                // folding the pool-honest way vs a maniac is the overlay's worst
                // failure mode. af>4 or allin-freq>10% ⇒ β floor 0.25.
                if req.live == 2 {
                    let opp_seat = (1 - req.hero_idx.min(1)) as u8;
                    let (af, _, _, ai) = self.villain_profile(req, opp_seat);
                    if af > 4.0 || ai > 0.10 {
                        bluff_share = bluff_share.max(0.25);
                        eprintln!("[pool-gate] maniac bettor (af {af:.1}, allin {ai:.2}) -> beta floored at {bluff_share}");
                    }
                }
                if let Some((eq, price)) = self.river_bet_gate(req, to_call, 0.35, bluff_share) {
                    if std::env::var("POOL_DEBUG").is_ok() {
                        eprintln!("[pool-dbg] river gate: hero {:?} board {:?} eq {eq:.3} price {price:.3}",
                            req.hero_cards, req.board);
                    }
                    // NO extra margin: the value-weighted range IS the caution
                    // (a +0.04 stack folded correct thin catches — battery-caught).
                    if eq < price {
                        eprintln!("[pool-gate] river call overridden -> FOLD (eq {eq:.3} vs price {price:.3}, pool bluff share {bluff_share})");
                        return mk(0, req.commit_entry as i32, resp.live, "river");
                    }
                }
            }
        }
        // 2) air-bluff suppressor on aggressive serves.
        if chosen.label >= 3 {
            if let Some(eq) = crate::api::rollout_equity(req) {
                let air: f32 = std::env::var("POOL_AIR_EQ").ok().and_then(|s| s.parse().ok()).unwrap_or(0.28);
                // FOLDY-TABLE EXCEPTION: bluffs print vs opponents who fold.
                // If EVERY live villain's blended fold-to-cbet ≥ 0.45 and
                // wtsd ≤ 0.28, let the solver's bluff through (the pool prior
                // ftc=.314 keeps the suppressor ON by default).
                let foldy = (0..req.live).filter(|&s2| s2 != req.hero_idx).all(|s2| {
                    let (_, wtsd, ftc, _) = self.villain_profile(req, s2);
                    ftc >= 0.45 && wtsd <= 0.28
                });
                if eq < air && !foldy {
                    let street = match req.board.len() { 3 => "flop", 4 => "turn", _ => "river" };
                    eprintln!("[pool-gate] air {} (eq {eq:.3}) overridden -> {}", chosen.action, if to_call == 0 { "check" } else { "fold" });
                    return if to_call == 0 {
                        mk(1, req.commit_entry as i32, resp.live, street)
                    } else {
                        mk(0, req.commit_entry as i32, resp.live, street)
                    };
                }
            }
        }
        resp
    }

    /// OFF-MODEL POSTFLOP GUARD (2026-07-04): honest play for hands whose search
    /// rows are floor-reach noise. Unbet: CHECK (an off-model hand must never
    /// bluff). Facing a flop/turn bet: MC pot-odds with a margin. Facing a RIVER
    /// bet: exact-strength bluff-catcher gate — hero equity vs a VALUE-WEIGHTED
    /// betting range on the actual board (top 35% of combos + 5% bluff mix), so
    /// bottom pair folds a pot bet while a real bluff-catcher still calls.
    fn decide_offmodel_guard(&self, req: &DecideRequest) -> Option<DecideResponse> {
        let to_call = req.to_call.unwrap_or_else(|| {
            let mx = req.street_actions.iter().map(|a| a.to_total).max().unwrap_or(0);
            mx
        });
        let serve = |label: u8, amount: i32| -> Option<DecideResponse> {
            let actions = vec![ActionProb {
                label,
                action: action_name(label).to_string(),
                amount,
                prob: 1.0,
            }];
            Some(DecideResponse {
                street: match req.board.len() { 3 => "flop", 4 => "turn", _ => "river" }.into(),
                live: req.live,
                actions: actions.clone(),
                chosen: actions[0].clone(),
                paired: false,
                search_ms: 0,
            })
        };
        if to_call == 0 {
            eprintln!("[post-guard] off-model hand, unbet -> check");
            return serve(1, 0);
        }
        let pot_after = (req.pot_entry as f32 + to_call as f32).max(1.0);
        let price = to_call as f32 / (pot_after + to_call as f32);
        if req.board.len() == 5 {
            // exact-strength river gate vs a value-weighted betting range
            use solver_core::hand::eval::Hand;
            let b = &req.board;
            let mut base = Hand::new();
            for &c in b.iter() {
                base = base.add_card(c as usize);
            }
            let dead: u64 = b.iter().chain(req.hero_cards.iter()).fold(0u64, |m, &c| m | (1u64 << c));
            let hero_s = base
                .add_card(req.hero_cards[0] as usize)
                .add_card(req.hero_cards[1] as usize)
                .evaluate_internal();
            let mut vs: Vec<i32> = Vec::with_capacity(1081);
            for c1 in 0..52u8 {
                if dead & (1 << c1) != 0 { continue; }
                for c2 in (c1 + 1)..52 {
                    if dead & (1 << c2) != 0 { continue; }
                    vs.push(base.add_card(c1 as usize).add_card(c2 as usize).evaluate_internal());
                }
            }
            vs.sort_unstable_by(|a, b| b.cmp(a)); // strongest first
            let n = vs.len();
            let value_n = n * 35 / 100;
            let bluff_n = n * 5 / 100;
            // betting range = top 35% + 5% random bluffs (weakest tail).
            let mut win = 0f64;
            let mut tot = 0f64;
            for (i, &s) in vs.iter().enumerate() {
                let w = if i < value_n { 1.0 } else if i >= n - bluff_n { 1.0 } else { 0.0 };
                if w == 0.0 { continue; }
                tot += 1.0;
                if hero_s > s { win += 1.0 } else if hero_s == s { win += 0.5 }
            }
            let eq = if tot > 0.0 { (win / tot) as f32 } else { 0.0 };
            let call_ok = eq > price + 0.04;
            eprintln!("[post-guard] off-model RIVER facing bet: eq-vs-betting-range {eq:.3} vs price {price:.3} -> {}",
                if call_ok { "call" } else { "fold" });
            return serve(if call_ok { 2 } else { 0 }, if call_ok { to_call as i32 } else { 0 });
        }
        // flop/turn: defer to the rollout's MC pot-odds with a stiffer margin by
        // shrinking the price headroom: require clear equity (margin 0.08).
        let eq = crate::api::rollout_equity(req)?;
        let call_ok = eq > price + 0.08;
        eprintln!("[post-guard] off-model {} facing bet: MC eq {eq:.3} vs price {price:.3} -> {}",
            if req.board.len() == 3 { "FLOP" } else { "TURN" },
            if call_ok { "call" } else { "fold" });
        serve(if call_ok { 2 } else { 0 }, if call_ok { to_call as i32 } else { 0 })
    }

    pub fn decide_postflop_search(&self, req: &DecideRequest) -> Option<DecideResponse> {
        // OFF-MODEL GUARD: floored-reach hands read noise from the search — serve
        // honest check/fold/priced-call instead (the postflop twin of the preflop
        // noise-row guard; measured leaks: air jams multiway, bottom-pair river
        // hero-calls).
        if self.hero_off_model(req) {
            if let Some(r) = self.decide_offmodel_guard(req) {
                return Some(r);
            }
        }
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
            // HAMMER VERDICT (2026-07-02): the live-6 facing-bet search measured
            // 20% success at concurrency-4 (capped exits of 2-8/24 iters after
            // 30s+ — below lookup quality at 60x the latency) and its zombie
            // solves were the runaway class. Default = LOOKUP; the search stays
            // available behind CONN_L6_FACING_SEARCH=1 for solo/light loads.
            if std::env::var("CONN_L6_FACING_SEARCH").map(|v| v == "1") != Ok(true) {
                return None;
            }
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
            let (it, bd) = (
                std::env::var("CONN_RIVER_ITERS_L5").ok().and_then(|s| s.parse().ok()).unwrap_or(120u32),
                8_000u128.min(req.budget_ms.map_or(u128::MAX, |b| b as u128)),
            );
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
        // OFF-GRID live-3/4 TURNS -> the conn LOOKUP (measured 2026-07-02): the
        // exact turn resolve behind the street search costs ~4s/ITERATION (the
        // turn tree recurses all 48 river branches) - 30s np=3 / 171s np=4 in
        // production, and a budget-starved 6-iter solve is slow AND near-
        // uniform. live-3/4 cells are RICH converged blueprint strategy; the
        // instant lookup strictly dominates. (live>=5 adapters carry all 49
        // turns, so they never off-grid.)
        if req.board.len() == 4 && req.live <= 4 {
            let tc = req.board[3];
            if !adapter.0.turns.contains(&tc) {
                return None;
            }
        }
        let reach_priors = self.reach_priors(req, &adapter.0);
        // Tuned base cfg (par+dcfr+iters); QRE λ from CONN_LAMBDA/CONN_OPP_LAMBDA.
        // live-5 trims iterations to its measured budget (CONN_ITERS_L5 overrides).
        let mut cfg = self.cfg;
        cfg.budget_ms = req.budget_ms; // deadline hand-me-down (queue-admitted late => shallower solve)
        if req.live == 5 {
            // GPU serves live-5 (cluster kernels, ~105-140 ms/iter incl
            // continuation) for first-in AND facing-bet (GPU rooting): 48 iters
            // ≈ 5-7s — a 50% convergence raise over the CPU-era 32.
            let l5 = std::env::var("CONN_ITERS_L5").ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(40);
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

    /// WARM-START GATE (item #6): warm-vs-cold exploitability curves for a flop
    /// cell. The warm run seeds the search's CUMULATIVE strategy (average only —
    /// regrets untouched) from the connected blueprint's own per-bucket action
    /// dist at every flop node, at `weight` virtual iterations. Returns
    /// (iters, cold_pct_pot, warm_pct_pot) per checkpoint. Ship warm-start ONLY
    /// if warm ≤ cold everywhere (or within noise) AND both converge to the same
    /// floor — the naive warm-start (regret seeding) was measured BAD before.
    pub fn warm_start_flop_sweep(
        &self,
        flop_id: usize,
        live: usize,
        commit: i32,
        pot: i32,
        checkpoints: &[u32],
        weight: f32,
    ) -> Vec<(u32, f32, f32)> {
        let adapter = match self.adapter(flop_id, live) {
            Some(a) => a,
            None => return vec![],
        };
        let Ok(post) = self.bp.postflop_cum(flop_id) else { return vec![] };
        let Some((cl, cc, cp)) = self.select_cell(live as u8, commit, pot) else { return vec![] };
        // Per-hand flop bucket in the CELL's nb space (same full-fidelity GS14
        // path as /decide). live>=5 adapters bucket in quantile-16 space — the
        // seed would be in the wrong space, so gate on the GS14 lives only.
        if live > 4 {
            return vec![];
        }
        let canonical = self.canonical_flops[flop_id];
        let Some(maps) = load_gs14_cache(
            &gs14_cache_path(&self.gs14_dir, flop_id, self.nb, self.bnt, self.bnr),
            self.nb, self.bnt, self.bnr,
        ) else { return vec![] };
        let fl = FlopLayout::for_canonical(canonical, self.bnt, self.bnr);
        let hc = &adapter.0.game.table().hand_cards;
        let nh = adapter.0.nh;
        let hb: Vec<Option<u16>> = (0..nh)
            .map(|h| fl.flop_bucket(&maps, (hc[h * 2] as Card, hc[h * 2 + 1] as Card)))
            .collect();
        let nb = self.nb;
        let layout = &self.layout;
        let post_ref = &post;

        // Seed builder: for every player node in the search tree, look the node's
        // action history up in the blueprint (per flop bucket), map the blueprint
        // actions onto the node's children (label -> class -> nearest amount, the
        // same matcher /decide + rooting use), and expand buckets -> hands.
        let seed_fn = move |tree: &solver_core::tree::flat::FlatTree| -> Vec<(usize, Vec<f32>)> {
            let class = |l: u8| -> u8 { match l { 0 => 0, 1 | 2 => 1, _ => 2 } };
            let mut out = Vec::new();
            let mut stack: Vec<(usize, Vec<(u8, i32)>)> = vec![(0, vec![])];
            while let Some((n, hist)) = stack.pop() {
                if !tree.nodes[n].is_player() {
                    continue;
                }
                let children = tree.node_children(n).to_vec();
                let na = children.len();
                let child_lbl = |i: usize| tree.nodes[children[i] as usize].action_label;
                let child_amt = |i: usize| tree.nodes[children[i] as usize].amount;
                // Per-bucket child probabilities from the blueprint dist.
                let per_bucket: Vec<Option<Vec<f32>>> = (0..nb)
                    .map(|b| {
                        layout
                            .postflop_action_dist(post_ref, cl, cc, cp, [b, 0, 0], &hist)
                            .map(|dist| {
                                let mut probs = vec![0f32; na];
                                for &(label, amount, p) in &dist {
                                    let mut cands: Vec<usize> =
                                        (0..na).filter(|&i| child_lbl(i) == label).collect();
                                    if cands.is_empty() {
                                        cands = (0..na)
                                            .filter(|&i| class(child_lbl(i)) == class(label))
                                            .collect();
                                    }
                                    if let Some(&i) = cands
                                        .iter()
                                        .min_by_key(|&&i| (child_amt(i) - amount).abs())
                                    {
                                        probs[i] += p;
                                    }
                                }
                                let z: f32 = probs.iter().sum();
                                if z > 1e-9 {
                                    for v in probs.iter_mut() {
                                        *v /= z;
                                    }
                                }
                                probs
                            })
                    })
                    .collect();
                if per_bucket.iter().any(|x| x.is_some()) {
                    let mut strat = vec![0f32; na * nh];
                    for h in 0..nh {
                        if let Some(pb) = hb[h].and_then(|b| per_bucket[b as usize].as_ref()) {
                            for a in 0..na {
                                strat[a * nh + h] = pb[a];
                            }
                        }
                    }
                    out.push((n, strat));
                }
                for &c in &children {
                    let child = c as usize;
                    let mut h2 = hist.clone();
                    h2.push((tree.nodes[child].action_label, tree.nodes[child].amount));
                    stack.push((child, h2));
                }
            }
            out
        };

        let cold = crate::pluribus_play::flop_search_exploitability_sweep_seeded(
            &adapter.0, commit, pot, &self.cfg, checkpoints, &seed_fn, 0.0,
        );
        let warm = crate::pluribus_play::flop_search_exploitability_sweep_seeded(
            &adapter.0, commit, pot, &self.cfg, checkpoints, &seed_fn, weight,
        );
        cold.iter()
            .zip(warm.iter())
            .map(|(&(it, c), &(_, w))| {
                let pp = |chips: f32| if pot > 0 { chips / pot as f32 * 100.0 } else { 0.0 };
                (it, pp(c), pp(w))
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
                return Some(self.pool_overlay(req, r));
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
            // UNMAPPABLE-SHAPE GUARD (2026-07-04, caught live at the 100-guard
            // checkpoint): histories the conn tree cannot express — above all
            // LIMPED pots (the tree is raise-or-fold; this pool limps constantly)
            // — used to fall through to the EQR player, whose deep nodes are the
            // original measured leaks (raise-63 overraises observed serving in
            // production). The equity guard needs no tree node: price the call
            // vs the aggressor posterior on a SYNTHETIC fold/call menu instead.
            let d_opt = self.bp.preflop_action_dist(hero, &hist);
            if d_opt.is_none() {
                let facing = hist.iter().any(|&(l, _)| matches!(l, 3 | 4 | 5));
                if facing {
                    if let Some(to_call) = req.to_call.filter(|&c| c > 0) {
                        let t = self.jam_tables();
                        // The Bayes walk cannot cross a limp (the conn tree is
                        // raise-or-fold), so no posterior exists for the raiser
                        // here. ANALYTIC AGGRESSOR PRIOR instead: top ~10% of
                        // classes by equity-vs-random at weight 1 + everything
                        // else at 0.02. ε calibration matters through COMBO
                        // MULTIPLICITY: junk is ~152 classes mostly ×12 combos,
                        // so ε=0.08 made the range ~60% junk by mass (94s read
                        // eq 0.364 -> call). ε=0.02 ≈ 25-30% junk — the value-
                        // heavy 3-bet mix loose-passive pools actually show.
                        let range: Vec<f32> = {
                            let nc = NUM_PREFLOP_CLASSES;
                            let mut avg: Vec<(usize, f32)> = (0..nc)
                                .map(|c| {
                                    let row = &t.equity[c * nc..(c + 1) * nc];
                                    (c, row.iter().sum::<f32>() / nc as f32)
                                })
                                .collect();
                            avg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            let mut r = vec![0.02f32; nc];
                            for &(c, _) in avg.iter().take(nc / 10) {
                                r[c] = 1.0;
                            }
                            r
                        };
                        let hcls = PreflopClass::from_combo(hero.0, hero.1).index();
                        let (mut eq_num, mut den) = (0.0f64, 0.0f64);
                        for oc in 0..NUM_PREFLOP_CLASSES {
                            let wgt = (range[oc] as f64)
                                * PreflopClass::new(oc as u8).num_combos() as f64
                                * t.blocking[hcls * NUM_PREFLOP_CLASSES + oc] as f64;
                            eq_num += wgt * t.equity[hcls * NUM_PREFLOP_CLASSES + oc] as f64;
                            den += wgt;
                        }
                        let eq = if den > 0.0 { (eq_num / den) as f32 } else { 0.35 };
                        let pot_after = (req.pot_entry as f32 + to_call as f32).max(1.0);
                        let price = to_call as f32 / (pot_after + to_call as f32);
                        let call_ok = eq > price + 0.04;
                        eprintln!("[pre-guard] UNMAPPABLE shape ({} actions): eq {eq:.3} vs price {price:.3} -> {}",
                            hist.len(), if call_ok { "call" } else { "fold" });
                        let (l, amt) = if call_ok {
                            (2u8, req.commit_entry as i32 + to_call as i32)
                        } else {
                            (0u8, req.commit_entry as i32)
                        };
                        let actions = vec![ActionProb {
                            label: l,
                            action: action_name(l).to_string(),
                            amount: amt,
                            prob: 1.0,
                        }];
                        return Some(DecideResponse {
                            street: "preflop".into(),
                            live: req.live,
                            actions: actions.clone(),
                            chosen: actions[0].clone(),
                            paired: false,
                            search_ms: 0,
                        });
                    }
                    // facing aggression, no to_call: safe fold.
                    eprintln!("[pre-guard] UNMAPPABLE shape, no to_call -> fold");
                    let actions = vec![ActionProb { label: 0, action: action_name(0).to_string(), amount: req.commit_entry as i32, prob: 1.0 }];
                    return Some(DecideResponse { street: "preflop".into(), live: req.live, actions: actions.clone(), chosen: actions[0].clone(), paired: false, search_ms: 0 });
                }
                // not facing a wager (limped pot, option to open/iso): fall to
                // the EQR player — its OPEN tiers are its good tiers (the leaks
                // were the deep facing-aggression nodes, taken over above).
                return None;
            }
            let d = d_opt?;
            // UNIFORM-GUARD (2026-07-03): a near-uniform row = an UNTRAINED node
            // (MCCFR trains only the played skeleton; the residual tail is
            // late-position raise-war defense). Serving it would be dice — the
            // measured leak class (44 jam-called a 4-bet at 1/8). Facing
            // aggression: FOLD (loose-pool-safe; premiums live on trained rows).
            // Not facing: passive check/call. The HU low-SPR spots never get
            // here (decide_preflop_jam serves them above).
            let z: f32 = d.iter().map(|(_, _, p)| p).sum::<f32>().max(1e-12);
            let maxp = d.iter().map(|(_, _, p)| p / z).fold(0.0f32, f32::max);
            let na = d.len().max(1);
            // PER-CLASS confidence (2026-07-04, client-caught): a node can be
            // trained overall while THIS class's row holds a few noisy visits —
            // measured live: 94s/J2s "called" 3-bets off rows that passed the
            // uniform test on ~single-digit cum. Below the mass floor, the row
            // is noise: route to the guard like an untrained node.
            let min_mass: f32 = std::env::var("CONN_PRE_MIN_MASS")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(50.0);
            let thin = self.bp.preflop_row_mass(hero, &hist).map_or(true, |m| m < min_mass);
            if maxp < 1.5 / na as f32 || thin {
                let facing = hist.iter().any(|&(l, _)| matches!(l, 3 | 4 | 5));
                // EQUITY-vs-POSTERIOR upgrade (2026-07-03): facing aggression at
                // an untrained node, don't blind-fold — pot-odds call/fold vs the
                // AGGRESSOR'S BAYESIAN RANGE (preflop_seat_reach posterior over
                // the observed line — live since the matcher fix) using the
                // 169×169 class-equity table. Targeted-CFR fill was measured
                // unable to beat line rarity (IS weight = the rarity), so the
                // runtime range solve IS the principled answer for this tail.
                if facing {
                    if let Some(to_call) = req.to_call.filter(|&c| c > 0) {
                        let seat_reach = self.bp.preflop_seat_reach(&hist);
                        // aggressor = seat of the LAST raise in the history walk;
                        // approximate with the max-to_total entry's seat via
                        // street_actions (seam seat), mapped through seat_positions
                        // when present; else fall back to a continuing-range mix.
                        // aggressor seat: walk the preflop tree along the history
                        // (class-tolerant nearest, mirroring the replay matcher)
                        // and record the seat of the last raise taken.
                        let agg_seat = {
                            let pft = &self.bp.pft;
                            let mut node = 0usize;
                            let mut agg: Option<usize> = None;
                            'walk: for &(label, to_total) in &hist {
                                if !pft.nodes[node].is_player() { break; }
                                let acting = pft.nodes[node].player_id;
                                let aggro = matches!(label, 3 | 4 | 5);
                                let mut best: Option<(i32, usize)> = None;
                                for &k in pft.node_children(node) {
                                    let kn = &pft.nodes[k as usize];
                                    let k_aggro = matches!(kn.action_label, 3 | 4 | 5);
                                    if (aggro && !k_aggro) || (!aggro && kn.action_label != label) {
                                        continue;
                                    }
                                    if !aggro {
                                        best = Some((0, k as usize));
                                        break;
                                    }
                                    let sz = pft.get_contribution(k as usize, acting).max(kn.amount);
                                    let dd = (sz - to_total).abs();
                                    if best.map_or(true, |(bd, _)| dd < bd) {
                                        best = Some((dd, k as usize));
                                    }
                                }
                                match best {
                                    Some((_, k)) => {
                                        if aggro { agg = Some(acting as usize); }
                                        node = k;
                                    }
                                    None => break 'walk,
                                }
                            }
                            agg
                        };
                        let range: Vec<f32> = match agg_seat {
                            Some(s) if s < seat_reach.len() => seat_reach[s].clone(),
                            _ => vec![1.0; NUM_PREFLOP_CLASSES],
                        };
                        let t = self.jam_tables();
                        let hcls = PreflopClass::from_combo(hero.0, hero.1).index();
                        let (mut eq_num, mut den) = (0.0f64, 0.0f64);
                        for oc in 0..NUM_PREFLOP_CLASSES {
                            let wgt = (range[oc] as f64)
                                * PreflopClass::new(oc as u8).num_combos() as f64
                                * t.blocking[hcls * NUM_PREFLOP_CLASSES + oc] as f64;
                            eq_num += wgt * t.equity[hcls * NUM_PREFLOP_CLASSES + oc] as f64;
                            den += wgt;
                        }
                        let eq = if den > 0.0 { (eq_num / den) as f32 } else { 0.35 };
                        let pot_after = (req.pot_entry as f32 + to_call as f32).max(1.0);
                        let price = to_call as f32 / (pot_after + to_call as f32);
                        // small margin: untrained spots play honest, not thin.
                        let call_ok = eq > price + 0.04;
                        let want = if call_ok { 2u8 } else { 0u8 };
                        if let Some(&(l, amt, _)) = d.iter().find(|&&(l, _, _)| l == want)
                            .or_else(|| d.iter().find(|&&(l, _, _)| l == 0)) {
                            eprintln!("[pre-guard] untrained facing node: eq {eq:.3} vs price {price:.3} -> {}",
                                if l == 2 { "call" } else { "fold" });
                            let actions = vec![ActionProb {
                                label: l,
                                action: action_name(l).to_string(),
                                amount: amt,
                                prob: 1.0,
                            }];
                            return Some(DecideResponse {
                                street: "preflop".into(),
                                live: req.live,
                                actions: actions.clone(),
                                chosen: actions[0].clone(),
                                paired: false,
                                search_ms: 0,
                            });
                        }
                    }
                }
                let want = if facing { 0u8 } else { 1u8 }; // fold : check
                let pick = d.iter().find(|&&(l, _, _)| l == want)
                    .or_else(|| d.iter().find(|&&(l, _, _)| l <= 2));
                if let Some(&(l, amt, _)) = pick {
                    eprintln!("[pre-guard] untrained node (maxp {maxp:.3}, na {na}) -> {} {}",
                        if l == 0 { "fold" } else { "check/call" }, amt);
                    let actions = vec![ActionProb {
                        label: l,
                        action: action_name(l).to_string(),
                        amount: amt,
                        prob: 1.0,
                    }];
                    return Some(DecideResponse {
                        street: "preflop".into(),
                        live: req.live,
                        actions: actions.clone(),
                        chosen: actions[0].clone(),
                        paired: false,
                        search_ms: 0,
                    });
                }
                return None;
            }
            // PREFLOP_TRACE=1: one line per trained-row preflop decision, so
            // client-observed anomalies (e.g. a class calling where probes fold)
            // can be pinned to the exact node/row without the client's request.
            if std::env::var("PREFLOP_TRACE").map(|v| v == "1") == Ok(true) {
                let mass = self.bp.preflop_row_mass(hero, &hist).unwrap_or(-1.0);
                let cls = PreflopClass::from_combo(hero.0, hero.1).index();
                eprintln!("[pre-trace] trained row: cls {cls} hist_len {} maxp {maxp:.3} row_mass {mass:.0}",
                    hist.len());
            }
            ("preflop", d)
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
        let resp = DecideResponse {
            street: street.to_string(),
            live: req.live,
            chosen: actions[sel].clone(),
            actions,
            search_ms: t0.elapsed().as_millis() as u64,
            paired: false,
        };
        // postflop LOOKUP serves pass through the pool overlay too.
        if (3..=5).contains(&req.board.len()) {
            return Some(self.pool_overlay(req, resp));
        }
        Some(resp)
    }
}
