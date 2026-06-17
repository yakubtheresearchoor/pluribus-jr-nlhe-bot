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
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
use solver_core::tree::flat::FlatTree;

use crate::preflop_oracle::seeded_1x1;

/// DCFR iters for the live-2 strategy solve — matches the postflop fill / the
/// banked CFV convention so the banked strategy is consistent with its CFV.
pub const LIVE2_BANK_ITERS: u32 = 34;

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

/// Read a live-2 strategy blob INTO a freshly-reconstructed solver. The caller
/// passes the same tree + canonical flop + flop index; this rebuilds the
/// identical seeded 1×1 table/solver and copies the banked cum buffers in. The
/// returned solver's average strategy (normalize(cum)) is the deployed policy.
pub fn load_live2(
    path: &str,
    canonical: [Card; 3],
    fi: usize,
    tree: &FlatTree,
) -> std::io::Result<FlopStartVectorCfr> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    // Split magic + header line from the f32 payload.
    let nl1 = bytes.iter().position(|&b| b == b'\n').expect("SSL2 magic newline");
    assert_eq!(&bytes[..nl1], b"SSL2", "live-2 blob bad magic in {path}");
    let nl2 = nl1 + 1 + bytes[nl1 + 1..].iter().position(|&b| b == b'\n').expect("header newline");
    let header = std::str::from_utf8(&bytes[nl1 + 1..nl2]).expect("header utf8").to_string();
    let payload = &bytes[nl2 + 1..];

    let (turns, river_decks) = seeded_1x1(canonical, fi);
    let table = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
    let game = FlopStartGame::new(table);
    let mut s = FlopStartVectorCfr::new(tree, game.table());

    let (fl, tl, rl) = (
        s.cum_strategy_flop().len(),
        s.cum_strategy_turn().len(),
        s.cum_strategy_river().len(),
    );
    let want = (fl + tl + rl) * 4;
    assert_eq!(
        payload.len(), want,
        "live-2 blob {path}: payload {} ≠ expected {want} (header {header}) — tree/table shape drift",
        payload.len()
    );
    let as_f32 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    s.cum_strategy_flop_mut().copy_from_slice(&as_f32(&payload[..fl * 4]));
    s.cum_strategy_turn_mut().copy_from_slice(&as_f32(&payload[fl * 4..(fl + tl) * 4]));
    s.cum_strategy_river_mut().copy_from_slice(&as_f32(&payload[(fl + tl) * 4..]));
    Ok(s)
}
