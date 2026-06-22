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

    let state = AppState {
        bp_root: bp_root.clone(),
        cfg,
        cache: Arc::new(Mutex::new(HashMap::new())),
        pf,
    };
    let app = Router::new()
        .route("/", get(|| async { "poker-bot decision server — POST /decide (see play_harness::api::DecideRequest)" }))
        .route("/decide", post(decide_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await.expect("bind");
    eprintln!("decision server on http://{bind}  (bp_root={bp_root}, iters={iters}, PAR={par}, DCFR={dcfr})");
    axum::serve(listener, app).await.expect("serve");
}

async fn decide_handler(
    State(st): State<AppState>,
    Json(mut req): Json<DecideRequest>,
) -> Result<Json<DecideResponse>, (StatusCode, String)> {
    // PREFLOP (empty board): EQR player — fast tree lookup, run inline.
    if req.board.is_empty() {
        let pf = st.pf.as_ref().ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "no preflop strategy loaded (set PF_STRAT)".to_string(),
        ))?;
        return match decide_preflop(&pf.0, &req) {
            Some(r) => Ok(Json(r)),
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

    // LIVE-2 (heads-up): banked exact HU strategy (.bp2 under {bp_root}/live2),
    // a lookup not a search. Only the flop is runout-independent (1×1 bank).
    if req.live == 2 {
        let live2_root = format!("{}/live2", st.bp_root);
        let out = tokio::task::spawn_blocking(move || {
            play_harness::api::decide_live2(&live2_root, &req)
        })
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("live2 join: {e}")))?;
        return match out {
            Some(r) => Ok(Json(r)),
            None => Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                "live-2: not a flop decision / SPR bin not banked / all-in (rollout)".into(),
            )),
        };
    }

    // live-6+ is the equity-rollout model (no blueprint) — not wired yet.
    if req.live >= 6 {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "live-6+ uses the equity-rollout model — no per-cell search wired".to_string(),
        ));
    }

    // POSTFLOP (live 3/4/5): get-or-load the blueprint for this cell+flop.
    let key = (req.cell_dir.clone(), req.flop_id);
    let bp: Arc<SyncBp> = {
        let mut cache = st.cache.lock().unwrap();
        if let Some(bp) = cache.get(&key) {
            bp.clone()
        } else {
            let path = format!("{}/{}/flop_{:04}.bp", st.bp_root, req.cell_dir, req.flop_id);
            let bp = Blueprint::load(&path)
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("load {path}: {e}")))?;
            let bp = Arc::new(SyncBp(bp));
            cache.insert(key, bp.clone());
            bp
        }
    };
    // Per-live latency schedule (parallel + discounted for heavy multiway counts)
    // so each live count fits the ~14s real-time budget without manual env tuning.
    let cfg = play_harness::pluribus_play::SearchCfg::for_live(req.live as usize, &st.cfg);
    // CPU-bound search → blocking pool so the async runtime isn't stalled.
    let out = tokio::task::spawn_blocking(move || decide_postflop(&bp, &req, &cfg))
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
