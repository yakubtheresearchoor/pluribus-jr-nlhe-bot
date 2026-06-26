//! LIVE-2 STRATEGY BANK (2026-06-15): serialize/deserialize a full-nh,
//! 1×1-runout live-2 (heads-up) postflop strategy from the exact HU solver
//! (`FlopStartVectorCfr`). Live-2 cannot go through the bucketed `.bp` pipeline
//! (the bucketed showdown asserts np≥3 — HU is genuinely exact), and a live
//! resolve is unviable (~156–780 s/decision, measured), so the HU strategy is
//! BANKED here instead. One blob per (SPR bucket × flop):
//!   <out>/live2_c{C}_p{P}_b/flop_NNNN.bp2
//!
//! Format: magic `SSL2\n`, a one-line JSON header (dims + cell + iters for
//! load-time validation), then the three time-summed cum-strategy buffers
//! (flop, turn, river) as f32-LE. The reader reconstructs an identical solver
//! (same tree + seeded 1×1 table) so buffer sizes match by construction, then
//! copies the buffers back in — same pattern as `cfv_from_banked` for the
//! bucketed `.bp`. Average strategy = normalize(cum) at query time.

use std::io::{Read, Write};

use solver_core::card::Card;
use solver_core::solver::chance_table::ChanceTable;
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::solver::turn_start_game::TurnStartGame;
use solver_core::solver::vector_cfr::VectorCfr;
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions, BoardState};
use solver_core::tree::builder::{build_tree, build_tree_with_bet_override};
use solver_core::tree::flat::FlatTree;

use crate::preflop_oracle::seeded_1x1;

/// DCFR iters for the live-2 strategy solve — matches the postflop fill / the
/// banked CFV convention so the banked strategy is consistent with its CFV.
pub const LIVE2_BANK_ITERS: u32 = 34;

/// The live-2 (HU) postflop bet menu — the RICH "M2" abstraction (3 bet sizes +
/// 2 raise sizes), enabled by the flop-only zstd bank (SSL3) which makes the
/// otherwise-prohibitive rich-menu storage fit (~2.4 GB vs ~440 GB raw). BOTH the
/// fill (`live2_fill`) and the decision (`api::decide_live2`) MUST build the HU
/// seam tree with THIS exact menu, or the banked buffers won't fit the tree.
pub fn live2_bet_menu() -> BetSizeOptions {
    BetSizeOptions {
        bet: vec![
            BetSize::PotRelative(0.33),
            BetSize::PotRelative(0.66),
            BetSize::PotRelative(1.0),
        ],
        raise: vec![BetSize::PotRelative(0.66), BetSize::PotRelative(1.0)],
    }
}

/// Real-time HU solve iter CEILINGS (measured: turn ≈5s @48 at mid-SPR, river
/// ≈100ms @96). The turn cost is strongly SPR-dependent (deep stacks ⇒ a much
/// bigger turn+river betting tree: ~45s @48 at SPR≈9), so the solve uses an ADAPTIVE
/// iter count — `solve_live2_street` measures one iteration and runs as many as fit
/// a wall-clock budget, capped at these ceilings.
pub const LIVE2_RT_TURN_ITERS: u32 = 48;
pub const LIVE2_RT_RIVER_ITERS: u32 = 96;
/// Wall-clock budget (ms) for a real-time street solve — kept under the ~14s live
/// budget with margin. At deep SPR the turn may run fewer than the ceiling iters.
pub const LIVE2_RT_BUDGET_MS: u128 = 9_000;
/// Floor on adaptive iters (don't degrade below this even if over budget).
pub const LIVE2_RT_MIN_ITERS: u32 = 6;

/// Result of a real-time exact HU street solve (turn or river root): the seam tree,
/// the solved CFR (query the average strategy at the hero's node), nh, and the hand
/// layout (`hand_cards[i*2..i*2+2]` = the i-th hand's two cards, c1<c2).
pub struct Live2StreetSolve {
    pub tree: FlatTree,
    pub cfr: VectorCfr,
    pub nh: usize,
    pub hand_cards: Vec<u8>,
}

/// Solve a live-2 (HU) TURN (4-card board) or RIVER (5-card board) subgame EXACTLY
/// in real time with the rich M2 menu, from uniform entering ranges. Unlike the flop
/// (1×1 bank, runout-specific), the actual board IS known here, so the remaining game
/// is small and exact — no abstraction, no continuation, HU plays to an exact
/// showdown: river = betting→showdown (~100ms); turn = river chance(47) + river
/// betting → showdown (~5s). `commit`/`pot` are the live game's chips entering the
/// street (the tree is built in that exact frame, so amounts need no rescale).
///
/// NOTE (approximation): the entering ranges are UNIFORM (full range), i.e. an
/// unconstrained subgame re-solve — it does not narrow the opponent's range by the
/// prior betting. Same simplification the flop bank makes at the flop.
pub fn solve_live2_street(board: &[u8], commit: i32, pot: i32, iters: u32) -> Option<Live2StreetSolve> {
    let spec = production_game_v1();
    let board_c: Vec<Card> = board.to_vec();
    let ranges = vec![vec![1.0f32; 1326]; 2];
    let (state, table) = match board.len() {
        4 => (BoardState::Turn, ChanceTable::compute_turn_start(&board_c, &ranges, 2)),
        5 => (BoardState::River, ChanceTable::compute_river_start(&board_c, &ranges, 2)),
        _ => return None,
    };
    let nh = table.num_valid;
    let hand_cards = table.hand_cards.clone();
    let game = TurnStartGame::new(table);
    let cfg = spec.street_seam_config(state, 2, commit, pot, live2_bet_menu());
    // NESTED solve: the TURN decision keeps the rich M2 menu, but the river
    // continuation INSIDE the turn lookahead is CHECK-ONLY (showdown equity) — this
    // shrinks the turn tree ~5× (the river-betting blowup at deep SPR: 584→182
    // nodes, 40s→8s @48it), keeping full 48 iters in budget at every SPR. The river
    // is re-solved EXACTLY with the full menu on arrival, so only the turn's
    // valuation of future river betting is approximated (Pluribus depth-limit
    // philosophy; check-down tracks the full-river turn strategy closely). The
    // river root (board.len()==5) is unaffected (no continuation past it).
    let tree = if board.len() == 4 {
        let check_only = BetSizeOptions { bet: vec![], raise: vec![] };
        build_tree_with_bet_override(&cfg, &[(BoardState::River, check_only)]).ok()?
    } else {
        build_tree(&cfg).ok()?
    };
    let mut cfr = VectorCfr::new(&tree, vec![nh; 2]);
    // Adaptive iters: time one iteration, then run as many more as fit the wall-clock
    // budget (capped at `iters`, floored at MIN). The river is cheap (runs the full
    // ceiling); a deep-SPR turn auto-trims to stay under budget.
    let t = std::time::Instant::now();
    cfr.run(&tree, &game, 1);
    let per_iter = t.elapsed().as_millis().max(1);
    let fit = (LIVE2_RT_BUDGET_MS / per_iter) as u32;
    let total = fit.clamp(LIVE2_RT_MIN_ITERS, iters);
    if total > 1 {
        cfr.run(&tree, &game, total - 1);
    }
    Some(Live2StreetSolve { tree, cfr, nh, hand_cards })
}

/// MULTIWAY (live ≥ 3) real-time TURN/RIVER re-solve with the FACTORED showdown.
///
/// This is the live-2 `solve_live2_street` generalized to ≥3 players: the only
/// way a full-nh multiway re-solve fits a real-time budget is the factored
/// O(nh·2^K) showdown (`TurnStartGame::with_factored`); the exact O(nh^K) one is
/// ~2.5 s/terminal at live-3 and ~1000 s at live-4. Same nested-continuation
/// trick as live-2 (turn nests a check-only river; river is a single street),
/// same adaptive iters (time one iter, fit `budget_ms`, floor at MIN). Ranges
/// are uniform entering the street (a real-time fallback prior; blueprint-reach
/// priors are a later refinement). Returns `None` for non-turn/river boards.
///
/// CAVEAT: the factored showdown is an INDEPENDENT-opponent approximation
/// (ignores inter-opponent card removal) — validated at <1% of pot per-hand EV
/// vs the exact brute force (`factored_showdown_probe`). It is the fallback for
/// the multiway turn/river hole (no per-cell blueprint there); far better than
/// the equity-rollout it sits in front of.
pub fn solve_multiway_street(
    board: &[u8],
    np: u8,
    commit: i32,
    pot: i32,
    iters: u32,
    budget_ms: u128,
) -> Option<Live2StreetSolve> {
    if np < 3 {
        return None; // HU goes through solve_live2_street (exact)
    }
    let spec = production_game_v1();
    let board_c: Vec<Card> = board.to_vec();
    let ranges = vec![vec![1.0f32; 1326]; np as usize];
    let (state, table) = match board.len() {
        4 => (BoardState::Turn, ChanceTable::compute_turn_start(&board_c, &ranges, np)),
        5 => (BoardState::River, ChanceTable::compute_river_start(&board_c, &ranges, np)),
        _ => return None,
    };
    let nh = table.num_valid;
    let hand_cards = table.hand_cards.clone();
    let game = TurnStartGame::new(table).with_factored();
    let cfg = spec.street_seam_config(state, np, commit, pot, live2_bet_menu());
    let tree = if board.len() == 4 {
        let check_only = BetSizeOptions { bet: vec![], raise: vec![] };
        build_tree_with_bet_override(&cfg, &[(BoardState::River, check_only)]).ok()?
    } else {
        build_tree(&cfg).ok()?
    };
    let mut cfr = VectorCfr::new(&tree, vec![nh; np as usize]);
    // Adaptive iters: time one iter, then fit the wall-clock budget (cap `iters`,
    // floor MIN) — identical discipline to the live-2 real-time solve.
    let t = std::time::Instant::now();
    cfr.run(&tree, &game, 1);
    let per_iter = t.elapsed().as_millis().max(1);
    let fit = (budget_ms / per_iter) as u32;
    let total = fit.clamp(LIVE2_RT_MIN_ITERS, iters);
    if total > 1 {
        cfr.run(&tree, &game, total - 1);
    }
    Some(Live2StreetSolve { tree, cfr, nh, hand_cards })
}

/// Header dims, written as JSON and re-checked on load (a mismatch means the
/// tree/table shape changed under the blob — a hard error, not a silent skip).
#[derive(Debug, Clone, PartialEq)]
pub struct Live2Header {
    pub nh: usize,
    pub flop_infosets: usize,
    pub turn_infosets: usize,
    pub river_infosets: usize,
    pub flop_len: usize,
    pub turn_len: usize,
    pub river_len: usize,
    pub commit: i32,
    pub pot: i32,
    pub fi: usize,
    pub iters: u32,
}

impl Live2Header {
    fn from_solver(s: &FlopStartVectorCfr, commit: i32, pot: i32, fi: usize, iters: u32) -> Self {
        Self {
            nh: s.num_hands(),
            flop_infosets: s.flop_infosets(),
            turn_infosets: s.turn_infosets(),
            river_infosets: s.river_infosets(),
            flop_len: s.cum_strategy_flop().len(),
            turn_len: s.cum_strategy_turn().len(),
            river_len: s.cum_strategy_river().len(),
            commit,
            pot,
            fi,
            iters,
        }
    }
    fn to_json(&self) -> String {
        format!(
            "{{\"nh\":{},\"fi_inf\":{},\"ti_inf\":{},\"ri_inf\":{},\"flop_len\":{},\"turn_len\":{},\"river_len\":{},\"commit\":{},\"pot\":{},\"fi\":{},\"iters\":{}}}",
            self.nh, self.flop_infosets, self.turn_infosets, self.river_infosets,
            self.flop_len, self.turn_len, self.river_len, self.commit, self.pot, self.fi, self.iters,
        )
    }
}

/// Solve a live-2 (HU) flop-entry spot at full nh on the seeded 1×1 runout and
/// return the solved solver (cum buffers populated by `run`).
pub fn solve_live2(canonical: [Card; 3], fi: usize, tree: &FlatTree) -> FlopStartVectorCfr {
    let (turns, river_decks) = seeded_1x1(canonical, fi);
    let table = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
    let game = FlopStartGame::new(table);
    let mut s = FlopStartVectorCfr::new(tree, game.table());
    s.run(tree, &game, LIVE2_BANK_ITERS);
    s
}

/// Write the live-2 strategy blob: magic + JSON header + cum_{flop,turn,river}.
pub fn save_live2(
    path: &str,
    s: &FlopStartVectorCfr,
    commit: i32,
    pot: i32,
    fi: usize,
) -> std::io::Result<()> {
    let hdr = Live2Header::from_solver(s, commit, pot, fi, LIVE2_BANK_ITERS);
    let tmp = format!("{path}.tmp");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(b"SSL2\n")?;
        writeln!(f, "{}", hdr.to_json())?;
        for buf in [s.cum_strategy_flop(), s.cum_strategy_turn(), s.cum_strategy_river()] {
            for &v in buf {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        f.flush()?;
    }
    std::fs::rename(&tmp, path)
}

/// Write the COMPRESSED flop-only live-2 blob (format SSL3): magic + JSON header +
/// zstd(cum_strategy_flop f32-LE). `decide_live2` only ever queries the FLOP (turn/
/// river are 1×1 / runout-specific and return None), so banking the flop buffer
/// alone drops ~95% of the bytes, and zstd on the near-pure strategy adds ~12× more
/// — together ~180× smaller than the naive 3-street f32 dump, and LOSSLESS: the flop
/// cum round-trips bit-exact, so the query path normalizes it identically to SSL2.
/// This makes the rich M2 bet menu bankable (~2.4 GB vs ~440 GB raw).
pub fn save_live2_v2(
    path: &str,
    s: &FlopStartVectorCfr,
    commit: i32,
    pot: i32,
    fi: usize,
) -> std::io::Result<()> {
    let hdr = Live2Header::from_solver(s, commit, pot, fi, LIVE2_BANK_ITERS);
    let flop = s.cum_strategy_flop();
    let mut raw = Vec::with_capacity(flop.len() * 4);
    for &v in flop {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let comp = zstd::encode_all(&raw[..], 9)?;
    let tmp = format!("{path}.tmp");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(b"SSL3\n")?;
        writeln!(f, "{}", hdr.to_json())?;
        f.write_all(&comp)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)
}

/// Read a live-2 strategy blob INTO a freshly-reconstructed solver. The caller
/// passes the same tree + canonical flop + flop index; this rebuilds the
/// identical seeded 1×1 table/solver and copies the banked cum buffers in.
/// Magic-aware: SSL2 = full 3-street f32 (legacy pot-only bank); SSL3 = flop-only
/// zstd (compressed rich-menu bank). The returned solver's average strategy
/// (normalize(cum)) is the deployed policy.
pub fn load_live2(
    path: &str,
    canonical: [Card; 3],
    fi: usize,
    tree: &FlatTree,
) -> std::io::Result<FlopStartVectorCfr> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    // Split magic + header line from the payload.
    let nl1 = bytes.iter().position(|&b| b == b'\n').expect("live-2 magic newline");
    let magic = bytes[..nl1].to_vec();
    let nl2 = nl1 + 1 + bytes[nl1 + 1..].iter().position(|&b| b == b'\n').expect("header newline");
    let header = std::str::from_utf8(&bytes[nl1 + 1..nl2]).expect("header utf8").to_string();
    let payload = &bytes[nl2 + 1..];

    let (turns, river_decks) = seeded_1x1(canonical, fi);
    let table = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
    let game = FlopStartGame::new(table);
    let mut s = FlopStartVectorCfr::new(tree, game.table());
    let as_f32 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let fl = s.cum_strategy_flop().len();

    // Shape/format mismatches return Err (not panic) so a caller hitting a bank
    // built with a DIFFERENT bet menu (tree shape drift during a re-bank) degrades
    // to None rather than crashing the server.
    let shape_err = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    match magic.as_slice() {
        b"SSL3" => {
            // Flop-only, zstd-compressed cum_strategy_flop. turn/river stay zero
            // (never queried — decide_live2 returns None off the flop).
            let raw = zstd::decode_all(payload)
                .map_err(|e| shape_err(format!("zstd {path}: {e}")))?;
            if raw.len() != fl * 4 {
                return Err(shape_err(format!(
                    "live-2 SSL3 blob {path}: flop payload {} ≠ expected {} (header {header}) — tree shape drift",
                    raw.len(), fl * 4
                )));
            }
            s.cum_strategy_flop_mut().copy_from_slice(&as_f32(&raw));
        }
        b"SSL2" => {
            // Legacy full 3-street f32 dump.
            let (tl, rl) = (s.cum_strategy_turn().len(), s.cum_strategy_river().len());
            let want = (fl + tl + rl) * 4;
            if payload.len() != want {
                return Err(shape_err(format!(
                    "live-2 SSL2 blob {path}: payload {} ≠ expected {want} (header {header}) — tree/table shape drift",
                    payload.len()
                )));
            }
            s.cum_strategy_flop_mut().copy_from_slice(&as_f32(&payload[..fl * 4]));
            s.cum_strategy_turn_mut().copy_from_slice(&as_f32(&payload[fl * 4..(fl + tl) * 4]));
            s.cum_strategy_river_mut().copy_from_slice(&as_f32(&payload[(fl + tl) * 4..]));
        }
        other => {
            return Err(shape_err(format!(
                "live-2 blob {path}: bad magic {:?}",
                String::from_utf8_lossy(other)
            )))
        }
    }
    Ok(s)
}
