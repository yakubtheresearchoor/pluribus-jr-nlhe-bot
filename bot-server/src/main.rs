//! Runtime decision HTTP server for the poker bot.
//!
//! POST /decide  — body = DecideRequest (see play_harness::api). Loads (caches)
//! the blueprint for the request's cell+flop, runs the per-street search on a
//! blocking thread (CPU-bound, ~0.3-17s depending on live count), and returns the
//! action distribution + a sampled choice. Pair mode = include partner_cards.
//!
//! Env: BP_ROOT (blueprint dir), BIND (addr, default 127.0.0.1:8080), ITERS
//! (search iterations), and PAR=1 / DCFR=1 to toggle the parallel / discounted
//! solver (set these when launching for live-3+ to fit the budget).
//!
//! Run: BP_ROOT=$PWD/blueprint_out_v1 PAR=1 cargo run --release -p bot-server

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use play_harness::api::{decide_postflop, decide_preflop, DecideRequest, DecideResponse};
use play_harness::blueprint::Blueprint;
use play_harness::pluribus_play::SearchCfg;
use play_harness::preflop_player::PreflopPlayer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The blueprint holds `Cell` chance state (FlopStartGame), making it `!Sync`.
/// That state is NEVER written during a DEPTH-LIMITED search (no chance recursion;
/// evaluate_* only read it), so sharing a blueprint across request threads — and
/// concurrent reads of the Cell — are sound. Assert it for the server.
struct SyncBp(Blueprint);
unsafe impl Send for SyncBp {}
unsafe impl Sync for SyncBp {}
impl std::ops::Deref for SyncBp {
    type Target = Blueprint;
    fn deref(&self) -> &Blueprint {
        &self.0
    }
}

/// The EQR preflop player is read-only at decision time (`action_dist(&self)`);
/// share it across request threads.
struct SyncPf(PreflopPlayer);
unsafe impl Send for SyncPf {}
unsafe impl Sync for SyncPf {}

#[derive(Clone)]
struct AppState {
    bp_root: String,
    cfg: SearchCfg,
    // (cell_dir, flop_id) -> loaded blueprint (loaded once, reused).
    cache: Arc<Mutex<HashMap<(String, u32), Arc<SyncBp>>>>,
    // EQR preflop strategy (for board-empty requests); None if PF_STRAT unset.
    pf: Option<Arc<SyncPf>>,
    // Connected blueprint decider (CONN_BP set): serves preflop + flop by lookup.
    conn: Option<Arc<play_harness::api_conn::ConnDecider>>,
    // Admission control for HEAVY postflop solves (CFR search / re-solve). Each
    // solve wants most of the machine (all cores + the GPU + GBs of buffers), so
    // running many at once oversubscribes CPU/GPU and balloons memory — the cause
    // of the timeout-storm + OOM. Bounds concurrent solves to a small N; excess
    // requests wait briefly then fast-fail 503 (better than hanging past the
    // client timeout). Preflop lookups bypass it entirely.
    solve_sema: Arc<tokio::sync::Semaphore>,
}

#[tokio::main]
async fn main() {
    let bp_root = std::env::var("BP_ROOT").unwrap_or_else(|_| "blueprint_out_v1".into());
    let iters = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(160u32);
    let cfg = SearchCfg { iters, ..Default::default() };
    let bind = std::env::var("BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let par = std::env::var("PAR").is_ok();
    let dcfr = std::env::var("DCFR").is_ok();

    // EQR preflop strategy (optional — preflop /decide needs it).
    let pf = std::env::var("PF_STRAT").ok().map(|base| {
        let p = PreflopPlayer::load(&base).expect("load PF_STRAT preflop strategy");
        eprintln!("loaded preflop strategy from {base} ({} nodes)", p.tree.num_nodes());
        Arc::new(SyncPf(p))
    });

    // Connected blueprint (CONN_BP=dir): preflop lookup + postflop search. Needs
    // CONN_GS14 (the 49×48 GS14 bucket cache). blueprint_conn_eqr = EQR-frozen
    // raise-or-fold preflop (no longer contaminated — see §11) + postflop re-solved
    // against it ⇒ self-consistent. Params (np=6, nraises=5, nb=200, maxna=7).
    let conn = std::env::var("CONN_BP").ok().map(|dir| {
        let gs14 = std::env::var("CONN_GS14").unwrap_or_else(|_| "gs14_blueprint_cache".into());
        let d = play_harness::api_conn::ConnDecider::load(&dir, &gs14, 6, 5, 200, 7)
            .expect("load CONN_BP connected blueprint");
        eprintln!("loaded connected blueprint from {dir} (gs14={gs14})");
        Arc::new(d)
    });

    // Max concurrent heavy solves. Each wants ~the whole machine, so keep this
    // small (default 2 — one CPU-bound + one GPU-bound overlap well). Tune via
    // BOTSERVER_MAX_SOLVES.
    let max_solves = std::env::var("BOTSERVER_MAX_SOLVES").ok().and_then(|s| s.parse().ok()).unwrap_or(2usize).max(1);
    let state = AppState {
        bp_root: bp_root.clone(),
        cfg,
        cache: Arc::new(Mutex::new(HashMap::new())),
        pf,
        conn,
        solve_sema: Arc::new(tokio::sync::Semaphore::new(max_solves)),
    };
    // WARM-UP: trigger every expensive one-time lazy init BEFORE the port opens,
    // so the first burst of real requests doesn't all block on cold `get_or_init`
    // and time out (the "fails ~20 times on startup" symptom). Guarded so a warm-up
    // failure degrades to a cold (but running) server rather than aborting startup.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| warmup(&state)));

    let app = Router::new()
        .route("/", get(|| async { "poker-bot decision server — POST /decide (see play_harness::api::DecideRequest)" }))
        .route("/decide", post(decide_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await.expect("bind");
    eprintln!("decision server on http://{bind}  (bp_root={bp_root}, iters={iters}, PAR={par}, DCFR={dcfr}, max_solves={max_solves})");
    axum::serve(listener, app).await.expect("serve");
}

/// Exercise every expensive one-time lazy init on the decision path so the first
/// real requests don't pay (and time out on) the cold build. Triggers: the 2.6M-
/// entry hand-eval `FULL_TABLE`, the board→canonical-flop MAP, the preflop-jam
/// equity table, a per-flop adapter (GS14 load + search structures), and — when
/// built `--features metal` with `GPU_SEARCH` — the Metal metallib + pipeline
/// compilation via a real HU-turn GPU solve. Results are ignored; each path just
/// needs to run once. Called before the port opens.
fn warmup(state: &AppState) {
    use play_harness::api::{route_to_canonical, DecideRequest};
    let t = std::time::Instant::now();

    // 1. FULL_TABLE — the big one (all 5-card hand values, sorted+deduped).
    let _ = solver_core::hand::eval::Hand::new()
        .add_card(0).add_card(1).add_card(2).add_card(3)
        .add_card(4).add_card(5).add_card(6)
        .evaluate_full();

    // 2. board→canonical-flop MAP.
    let mut r = DecideRequest { board: vec![51, 50, 20], route: true, live: 3, ..Default::default() };
    let _ = route_to_canonical(&mut r);

    // 3. Real decision paths via the connected decider (equity table, adapters,
    //    GS14 load, and the GPU pipeline). Each returns None-or-Ok harmlessly.
    if let Some(conn) = state.conn.as_ref() {
        // preflop lookup
        let _ = conn.decide(&DecideRequest { board: vec![], hero_cards: [51, 48], live: 6, hero_idx: 2, ..Default::default() });
        // low-SPR HU preflop → builds/loads the preflop-jam equity table
        let _ = conn.decide(&DecideRequest {
            board: vec![], hero_cards: [51, 48], live: 2, hero_idx: 0,
            commit_entry: 40, pot_entry: 80, to_call: Some(40), ..Default::default()
        });
        // multiway flop → per-flop adapter (GS14 deserialize + FlopChanceTable + bucketing)
        let mut f = DecideRequest {
            board: vec![51, 50, 20], hero_cards: [48, 49], live: 3, hero_idx: 0,
            commit_entry: 6, pot_entry: 18, to_call: Some(0), route: true, ..Default::default()
        };
        if route_to_canonical(&mut f).is_some() {
            let _ = conn.decide(&f);
        }
        // HU turn → decide_live2_resolve, which warms the Metal metallib + pipeline
        // compilation (the pricey first-GPU-call cost) when GPU_SEARCH is set.
        let _ = conn.decide(&DecideRequest {
            board: vec![51, 50, 20, 1], hero_cards: [48, 49], live: 2, hero_idx: 0,
            commit_entry: 20, pot_entry: 40, to_call: Some(0), ..Default::default()
        });
    }
    eprintln!("warmup complete in {:.1}s (lazy inits primed)", t.elapsed().as_secs_f32());
}

async fn decide_handler(
    State(st): State<AppState>,
    Json(mut req): Json<DecideRequest>,
) -> Result<Json<DecideResponse>, (StatusCode, String)> {
    let t_start = std::time::Instant::now();
    // PREFLOP (empty board): connected blueprint if loaded, else EQR player.
    if req.board.is_empty() {
        if let Some(conn) = st.conn.as_ref() {
            if let Some(r) = conn.decide(&req) {
                return Ok(Json(r));
            }
        }
        let pf = st.pf.as_ref().ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "no preflop strategy loaded (set PF_STRAT)".to_string(),
        ))?;
        return match decide_preflop(&pf.0, &req) {
            Some(r) => {
                // The conn layer could not serve this shape — the EQR PLAYER
                // (the pre-v5 chart with the measured deep-node leaks) answered.
                // Loud on purpose: any volume here is a mapping gap to fix.
                eprintln!(
                    "[pre-fallback] EQR player served: {} {} (live {}, {} street_actions, to_call {:?})",
                    r.chosen.action, r.chosen.amount, req.live, req.street_actions.len(), req.to_call
                );
                Ok(Json(r))
            }
            None => Err((StatusCode::BAD_REQUEST, "unmappable preflop node".into())),
        };
    }

    // Board→canonical-flop routing: derive flop_id + remap cards from the raw
    // board so the runtime can omit flop_id and send real cards. Run BEFORE the
    // live-count dispatch so both the search and live-2 paths get canonical cards.
    if req.route {
        play_harness::api::route_to_canonical(&mut req).ok_or((
            StatusCode::BAD_REQUEST,
            "route: board < 3 cards or canonical flop not in bank".to_string(),
        ))?;
    }

    // POSTFLOP input validation: the seam-tree builders assert
    // `pot >= live*commit`, `commit <= stack`, `2 <= live <= num_players`
    // (street_seam_config). An HTTP request violating these would panic a worker
    // (the task is isolated, but the caller gets a dropped connection, not a
    // diagnosable error). Reject with 400 up front so every postflop path is safe.
    if (3..=5).contains(&req.board.len()) {
        let stack = play_harness::runtime_game_spec().stack;
        let (live, commit, pot) = (req.live as i32, req.commit_entry as i32, req.pot_entry as i32);
        if req.live < 2 || req.live > 6 {
            return Err((StatusCode::BAD_REQUEST, format!("postflop: live must be 2..=6, got {}", req.live)));
        }
        if commit > stack {
            return Err((StatusCode::BAD_REQUEST, format!("postflop: commit_entry {commit} exceeds stack {stack}")));
        }
        if pot < live * commit {
            return Err((StatusCode::BAD_REQUEST, format!("postflop: pot_entry {pot} < live*commit ({live}*{commit})")));
        }
    }

    // ADMISSION CONTROL (REQUIRED with the shared MetalContext). Every GPU solve
    // shares ONE Metal command queue; concurrent command-buffer creation on it
    // from multiple threads deadlocks Metal (0% CPU hang). The postflop conn.decide
    // below also runs its solve on the async worker, so unbounded concurrency
    // starves the runtime. Gating heavy solves (BOTSERVER_MAX_SOLVES; keep at 1
    // while the GPU path shares one queue) means one solve at a time — no
    // concurrent GPU submission, and waiters await the permit ASYNC (they don't
    // block worker threads, so health/other routes stay responsive). Preflop
    // lookups returned above and never reach here. Saturated ⇒ wait 2s then 503.
    // Cancel guard first — it also covers the LOOKUP lane below. Dropping this
    // future (client disconnect) flips the flag and any in-flight work bails.
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _cancel_guard = play_harness::cancel::CancelOnDrop(cancel.clone());

    // ---------- LOOKUP LANE (no admission permit) ----------
    // Spots whose PRODUCTION answer is a lookup/rollout (turn/river cell reads,
    // live-6, off-grid turns) never queue behind heavy solves: they are ms-scale
    // reads that touch neither the GPU nor a core-bound solve. `search_engages`
    // mirrors the search gates exactly, so this cannot serve a lookup where the
    // search is the production path. live-2 stays on the heavy lane (its bank
    // lookup is cheap but its turn/river paths are exact solves).
    if (3..=5).contains(&req.board.len()) && req.live >= 3 {
        if let Some(conn) = st.conn.clone() {
            if !conn.search_engages(&req) {
                let flag = cancel.clone();
                let (served, returned): (Option<DecideResponse>, DecideRequest) =
                    tokio::task::spawn_blocking(move || {
                        let r = play_harness::cancel::with_cancel_flag(flag, || {
                            conn.decide(&req)
                                .or_else(|| play_harness::api::decide_live6(&req))
                        });
                        (r, req)
                    })
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("lookup join: {e}")))?;
                req = returned;
                if let Some(r) = served {
                    return Ok(Json(r));
                }
                // Unservable by lookup — fall through to the heavy lane.
            }
        }
    }

    // ---------- HEAVY LANE: deadline-aware queue ----------
    // The client tells us (or BOTSERVER_DEADLINE_MS defaults) how long it will
    // wait. We queue for a permit as long as a USEFUL solve can still finish
    // before that deadline, instead of the old flat 2s-then-503: at game pace a
    // collision is nearly always depth-1 and a few seconds of patience converts
    // a lost decision into a served one. The permit wait is fair-FIFO (tokio
    // semaphore). If the wait outlives the deadline math, fail FAST — the
    // client's fallback wants the 503 as early as possible.
    let deadline_total_ms: u64 = req.deadline_ms.unwrap_or_else(|| {
        std::env::var("BOTSERVER_DEADLINE_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000)
    });
    // Margin for response serialization + network + client-side slack.
    const DEADLINE_MARGIN_MS: u64 = 700;
    // Below this a solve is near-uniform — shed instead of serving noise.
    const MIN_USEFUL_SOLVE_MS: u64 = 2_500;
    let deadline = t_start + std::time::Duration::from_millis(deadline_total_ms.saturating_sub(DEADLINE_MARGIN_MS));
    let wait_cap = deadline
        .checked_duration_since(std::time::Instant::now())
        .and_then(|d| d.checked_sub(std::time::Duration::from_millis(MIN_USEFUL_SOLVE_MS)))
        .unwrap_or_default();
    #[allow(unused_mut)]
    let mut permit = match st.solve_sema.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => match tokio::time::timeout(wait_cap, st.solve_sema.clone().acquire_owned()).await {
            Ok(Ok(p)) => p,
            _ => {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "queue: cannot finish before the {deadline_total_ms}ms deadline (waited {}ms)",
                        t_start.elapsed().as_millis()
                    ),
                ))
            }
        },
    };
    // BUDGET HAND-ME-DOWN: whatever deadline remains after queueing is THE
    // solve budget — every runaway cap downstream (GPU batched budget, CPU
    // per-iter cap, resolve chunks) clamps to it, so an admitted-late solve
    // finishes shallow-but-in-time instead of full-depth-but-late.
    let remaining_ms = deadline
        .checked_duration_since(std::time::Instant::now())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    if remaining_ms < MIN_USEFUL_SOLVE_MS {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "queue: deadline exhausted while waiting for a solve slot".into(),
        ));
    }
    req.budget_ms = Some(remaining_ms);
    // HAMMER FIX (2026-07-02): the permit rides INSIDE each spawn_blocking
    // closure, NOT this future. When a client disconnects, axum drops this
    // future — a future-held permit released while its blocking solve kept
    // running, admitting the next solve on top (measured: 17 concurrent zombie
    // resolves past max_solves=3). The closures below own the permit for the
    // true lifetime of the work; the cancel guard reaps abandoned solves.

    // POSTFLOP (flop/turn/river) via the connected blueprint (lookup) when loaded.
    // Turn/river need `prior_actions` (the runtime supplies the full postflop path).
    // Falls through to the existing search/live paths if it can't serve.
    if (3..=5).contains(&req.board.len()) {
        if let Some(conn) = st.conn.clone() {
            // Run the solve on a BLOCKING thread. conn.decide runs the CPU/GPU
            // search (seconds); on an async worker it stalls the whole runtime
            // (health/other routes hang), and — sharing the one Metal command
            // queue — a concurrent GPU solve deadlocks/pins the GPU. The permit
            // above caps this to one at a time; spawn_blocking keeps it off the
            // async workers. Move `req` in and hand it back so the fallthrough
            // live-2/live-6/legacy paths below can still use it if conn returns None.
            let flag = cancel.clone();
            let (served, returned, permit_back) =
                tokio::task::spawn_blocking(move || {
                    let r = play_harness::cancel::with_cancel_flag(flag, || conn.decide(&req));
                    (r, req, permit)
                })
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("conn solve join: {e}")))?;
            req = returned;
            permit = permit_back;
            if let Some(r) = served {
                return Ok(Json(r));
            }
        }
    }

    // LIVE-2 (heads-up): banked exact HU strategy (.bp2 under {bp_root}/live2),
    // a lookup not a search. Only the flop is runout-independent (1×1 bank).
    if req.live == 2 {
        let subdir = std::env::var("L2_SUBDIR").unwrap_or_else(|_| "live2".into());
        let live2_root = format!("{}/{}", st.bp_root, subdir);
        // Try the bank; if it can't serve the spot (unbanked SPR bin, or a betting
        // sequence deeper than the search tree's cap — e.g. a flop raise-war past
        // 3 bets/street, which makes walk_to_node fall off the tree), fall to the
        // equity rollout (sane check / pot-odds) rather than a 422 the live game
        // can't act on. A proper HU flop facing-bet solve is the follow-up (§11).
        let flag = cancel.clone();
        let req2 = Some(req.clone());
        let out = tokio::task::spawn_blocking(move || {
            let _permit = permit; // held for the work's true lifetime
            play_harness::cancel::with_cancel_flag(flag, || {
                play_harness::api::decide_live2(&live2_root, &req)
                    .or_else(|| play_harness::api::decide_live6(&req))
            })
        })
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("live2 join: {e}")))?;
        return match out {
            Some(r) => {
                let r = match (st.conn.as_ref(), req2.as_ref()) {
                    (Some(c), Some(rq)) => c.pool_overlay(rq, r),
                    _ => r,
                };
                Ok(Json(r))
            }
            None => Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                "live-2: unresolvable postflop spot (board not 3-5 cards)".into(),
            )),
        };
    }

    // LIVE-6+ (full ring): equity-rollout model — no blueprint, MC showdown equity
    // → check when unbet, pot-odds call/fold when facing a bet. CPU-bound MC →
    // blocking pool.
    if req.live >= 6 {
        let flag = cancel.clone();
        let out = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            play_harness::cancel::with_cancel_flag(flag, || play_harness::api::decide_live6(&req))
        })
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("live6 join: {e}")))?;
        return match out {
            Some(r) => Ok(Json(r)),
            None => Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                "live-6: board must be 3-5 cards".into(),
            )),
        };
    }

    // POSTFLOP (live 3/4/5): get-or-load the LEGACY per-cell blueprint for this
    // cell+flop. In a CONNECTED-blueprint deployment (CONN_BP set) these legacy
    // `flop_NNNN.bp` files do NOT exist — so a spot the connected decider can't
    // serve (e.g. a 3-way flop FACING A BET whose size doesn't map to the cell
    // tree) used to hard-fail with "load .../flop_NNNN.bp: missing". Fall to the
    // graceful equity-rollout (decide_live6) instead of a 400. (A real facing-a-
    // bet 3-way flop strategy from the connected path is the proper follow-up.)
    let key = (req.cell_dir.clone(), req.flop_id);
    // None ⇒ the legacy cell is absent in a connected deployment ⇒ fall to the
    // rollout (handled AFTER the cache lock is released, so no MutexGuard is held
    // across the .await).
    let bp_opt: Option<Arc<SyncBp>> = {
        let mut cache = st.cache.lock().unwrap();
        if let Some(bp) = cache.get(&key) {
            Some(bp.clone())
        } else {
            let path = format!("{}/{}/flop_{:04}.bp", st.bp_root, req.cell_dir, req.flop_id);
            match Blueprint::load(&path) {
                Ok(bp) => {
                    let bp = Arc::new(SyncBp(bp));
                    cache.insert(key, bp.clone());
                    Some(bp)
                }
                Err(e) => {
                    if st.conn.is_some() {
                        None // connected deployment: legacy cell absent by design
                    } else {
                        return Err((StatusCode::BAD_REQUEST, format!("load {path}: {e}")));
                    }
                }
            }
        }
    };
    let bp: Arc<SyncBp> = match bp_opt {
        Some(bp) => bp,
        None => {
            let flag = cancel.clone();
            let out = tokio::task::spawn_blocking(move || {
                let _permit = permit;
                play_harness::cancel::with_cancel_flag(flag, || play_harness::api::decide_live6(&req))
            })
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("live6 join: {e}")))?;
            return match out {
                Some(r) => Ok(Json(r)),
                None => Err((StatusCode::UNPROCESSABLE_ENTITY, "postflop: unservable spot".into())),
            };
        }
    };
    // Per-live latency schedule (parallel + discounted for heavy multiway counts)
    // so each live count fits the ~14s real-time budget without manual env tuning.
    let cfg = play_harness::pluribus_play::SearchCfg::for_live(req.live as usize, &st.cfg);
    // CPU-bound search → blocking pool so the async runtime isn't stalled.
    let flag = cancel.clone();
    let out = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        play_harness::cancel::with_cancel_flag(flag, || decide_postflop(&bp, &req, &cfg))
    })
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("search join: {e}")))?;
    match out {
        Some(r) => Ok(Json(r)),
        None => Err((
            StatusCode::BAD_REQUEST,
            "unmappable decision (bad state / node not hero's / blocked board)".into(),
        )),
    }
}
