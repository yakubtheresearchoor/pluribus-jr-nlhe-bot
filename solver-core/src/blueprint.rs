//! Connected-MCCFR blueprint (BLP2) loader + decision lookup.
//!
//! The blueprint is a flat average-strategy table produced by the connected GPU
//! MCCFR solve (`mccfr_probe`'s `gpu_conn_solve`, written via `MC_EXTRACT`). It is
//! SELF-CONTAINED: a shared 169-class preflop region followed by per-flop bucketed
//! postflop regions. Unlike the per-cell search blueprint
//! (`play_harness::blueprint::Blueprint`), serving a decision from it is a LOOKUP,
//! not a re-solve.
//!
//! The on-disk header carries no tree structure, so the loader rebuilds the EXACT
//! preflop tree the solver used (same builder, same config) to recover the node ↔
//! betting-situation mapping. See [`build_conn_preflop_tree`], which MUST stay in
//! lock-step with `gpu_conn_solve`'s tree setup — both call it.
//!
//! BLP2 header (12 × u32 LE): `[magic=0x424C5032, np, nraises, pre_ninfo, 169,
//! maxna, nf, post_stride, nh, nb, nt, nr]`, then `gcum` (`pre_ninfo·169·maxna +
//! nf·post_stride` × f32 LE).

use crate::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use crate::card::Card;
use crate::solver::postflop_oracle::SeamCell;
use crate::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions, BoardState};
use crate::tree::builder::{build_tree_preflop_only, build_tree_with_bet_override};
use crate::tree::flat::FlatTree;
use std::io;

/// Per-live SPR bin WIDTHS (log2-SPR units). Finer = more cells = more memory.
/// The real ceiling is GPU memory (2 buffers × reg_buf × 4B ≤ ~27GB default),
/// tighter than u32. Tuned to fit default memory while beating v3's ~24 bins/live.
/// MUST be identical at solve + runtime; env is TUNING-ONLY — hardcode final.
/// Tuned to fit ~24.5GB GPU (under the ~27GB default wired limit) with 119 three-
/// way + 100 four-way bins (≈5× v3). HARDCODED — must match solve + runtime; never
/// env-driven (a mismatch would corrupt the layout).
const SPR_W3: f64 = 0.04;
const SPR_W4: f64 = 0.05;

/// PER-LIVE seam CELL KEY — SHARED by the connected solve (`gpu_conn_solve`) and
/// the runtime (`ConnCellLayout`) so the cell layout is byte-exact:
///   - live = 2 (HU): EXACT (commit, pot) — cheap (0.33B) + the most volume.
///   - live = 3: fine SPR bin (`spr_w3`) — finer than v3.
///   - live = 4: fine SPR bin (`spr_w4`) — finer than v3.
///   - live ≥ 5: ONE shared cell per live (rarely reached under raise-or-fold).
/// SPR = (stack − commit)/pot. `stack` = production start stack.
pub fn conn_seam_bin(pft: &FlatTree, node: usize, np: usize, stack: i32) -> (u8, i64) {
    let sc = SeamCell::at_chance_node(pft, node, np);
    conn_seam_bin_from(sc.live, sc.commit, pft.starting_pot + sc.pot, stack)
}
/// The representative `(live, commit, pot)` for a seam node — the FIRST node seen
/// for a bin defines that cell's betting tree (deterministic given tree order, so
/// solve + runtime agree).
pub fn conn_seam_rep(pft: &FlatTree, node: usize, np: usize) -> (u8, i32, i32) {
    let sc = SeamCell::at_chance_node(pft, node, np);
    (sc.live, sc.commit, pft.starting_pot + sc.pot)
}
/// PER-LIVE cell key from an explicit `(live, commit, pot)` (`pot` = full pot incl.
/// dead). See `conn_seam_bin` for the per-live policy.
pub fn conn_seam_bin_from(live: u8, commit: i32, pot: i32, stack: i32) -> (u8, i64) {
    let spr_bin = |w: f64| -> i64 {
        let behind = stack - commit;
        if behind <= 0 {
            return i64::MIN;
        }
        ((behind as f64 / pot as f64).log2() / w).floor() as i64
    };
    match live {
        // EXACT: unique i64 per (commit, pot). commit,pot ≥ 0 and ≪ 2^31.
        0..=2 => (live, ((commit as i64) << 32) | (pot as i64 & 0xFFFF_FFFF)),
        3 => (live, spr_bin(SPR_W3)),
        4 => (live, spr_bin(SPR_W4)),
        // live ≥ 5: one shared cell per live.
        _ => (live, 0),
    }
}

pub const BLP2_MAGIC: u32 = 0x42_4C_50_32;
pub const GS14_MAGIC: u32 = 0x4753_3134;

// ─────────────────────────────────────────────────────────────────────
// SSBP2 cum codec — per-slab u8-quant + zstd. Byte-identical to play-harness's
// `reencode_to_v2` / `quantize_roundtrip` (money-test-proven play-safe, ~16×).
// Used to compress each per-flop postflop `cum` slab to disk in the per-flop
// single-pass solve, so the 447GB fully-connected working set never materializes:
// one flop's f32 lives at a time, only its FINAL output is quantized (never live
// regrets → no requantization-convergence risk).
// Format: [lo:f32 LE][scale:f32 LE][zstd(u8 values)], u8 = round((v-lo)/scale·255).
// ─────────────────────────────────────────────────────────────────────

/// Encode an f32 slab as SSBP2 (`[lo][scale][zstd(u8)]`). `zlevel` = zstd level
/// (9 is the production setting).
pub fn ssbp2_encode_cum(slab: &[f32], zlevel: i32) -> Vec<u8> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in slab {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if !lo.is_finite() {
        lo = 0.0;
        hi = 0.0;
    }
    let scale = (hi - lo).max(1e-12);
    let q: Vec<u8> = slab
        .iter()
        .map(|&v| (((v - lo) / scale) * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    let z = zstd::encode_all(&q[..], zlevel).expect("zstd encode");
    let mut out = Vec::with_capacity(8 + z.len());
    out.extend_from_slice(&lo.to_le_bytes());
    out.extend_from_slice(&scale.to_le_bytes());
    out.extend_from_slice(&z);
    out
}

/// Decode an SSBP2 cum slab back to f32 (dequantized: `lo + q/255·scale`).
pub fn ssbp2_decode_cum(bytes: &[u8]) -> io::Result<Vec<f32>> {
    if bytes.len() < 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "ssbp2 slab < 8 bytes"));
    }
    let lo = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let scale = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let q = zstd::decode_all(&bytes[8..]).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(q.iter().map(|&b| lo + (b as f32 / 255.0) * scale).collect())
}

/// GS14 potential-aware + EMD bucket maps for one canonical flop, indexed by the
/// flop's hand/runout enumeration: `(flop_map[h], turn_map[ti][h],
/// river_map[ti][ri][h])`. A bucket of `u16::MAX` marks a hand dead on that
/// runout (its hole cards collide with the dealt board). This is the same tuple
/// the solver caches (`save_gs14`).
pub type Gs14Maps = (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>);

/// Per-flop GS14 cache path — mirrors the solver's `gs14_cache_path`:
/// `{dir}/gs14_f{flop}_nb{nb}_{nt}x{nr}.bin`.
pub fn gs14_cache_path(dir: &str, flop_idx: usize, nb: usize, nt: usize, nr: usize) -> std::path::PathBuf {
    std::path::PathBuf::from(dir).join(format!("gs14_f{flop_idx}_nb{nb}_{nt}x{nr}.bin"))
}

fn read_u16_vec(data: &[u8], off: &mut usize, n: usize) -> Option<Vec<u16>> {
    if *off + n * 2 > data.len() { return None; }
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(u16::from_le_bytes([data[*off], data[*off + 1]]));
        *off += 2;
    }
    Some(v)
}

/// Load a GS14 bucket-map cache file. Byte-compatible with the solver's
/// `save_gs14`: header `[magic=0x47533134, nh, nt, nr, nb]` (u32 LE) then
/// `flop_map`, `turn_map[ti]`, `river_map[ti][ri]` (u16 LE). Returns `None` on a
/// short file or a header that disagrees with the requested `(nb, nt, nr)`.
pub fn load_gs14_cache(path: &std::path::Path, nb: usize, nt: usize, nr: usize) -> Option<Gs14Maps> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 20 {
        return None;
    }
    let rd = |i: usize| u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
    if rd(0) != GS14_MAGIC as usize || rd(8) != nt || rd(12) != nr || rd(16) != nb {
        return None;
    }
    let nh = rd(4);
    let mut off = 20usize;
    let fm = read_u16_vec(&data, &mut off, nh)?;
    let mut tm = Vec::with_capacity(nt);
    for _ in 0..nt {
        tm.push(read_u16_vec(&data, &mut off, nh)?);
    }
    let mut rm = Vec::with_capacity(nt);
    for _ in 0..nt {
        let mut per_turn = Vec::with_capacity(nr);
        for _ in 0..nr {
            per_turn.push(read_u16_vec(&data, &mut off, nh)?);
        }
        rm.push(per_turn);
    }
    Some((fm, tm, rm))
}

/// Subset full (bnt×bnr) GS14 maps to an (nt×nr) runout subset, preserving the
/// full-fidelity bucket VALUES (runout_grid(nt,nr) cards ⊂ the (bnt,bnr) grid).
/// Shared by the solver and the runtime (search adapter at a reduced runout).
pub fn subset_gs14(full: &Gs14Maps, flop: [Card; 3], bnt: usize, bnr: usize, nt: usize, nr: usize) -> Gs14Maps {
    let (bturns, _) = runout_grid(flop, bnt, bnr);
    let (sturns, _) = runout_grid(flop, nt, nr);
    let pos = |hay: &[u8], needle: u8| hay.iter().position(|&x| x == needle).expect("subset card not in full grid");
    let bturns_u8: Vec<u8> = bturns.iter().map(|&c| c as u8).collect();
    let tpos: Vec<usize> = sturns.iter().map(|&t| pos(&bturns_u8, t as u8)).collect();
    let tm: Vec<Vec<u16>> = tpos.iter().map(|&p| full.1[p].clone()).collect();
    let (_, b_rivers) = runout_grid(flop, bnt, bnr);
    let (_, s_rivers) = runout_grid(flop, nt, nr);
    let rm: Vec<Vec<Vec<u16>>> = sturns.iter().enumerate().map(|(si, &stc)| {
        let p_t = tpos[si];
        let brd = &b_rivers[stc as u8 as usize];
        s_rivers[stc as u8 as usize].iter().map(|&src| {
            let p_r = pos(brd, src);
            full.2[p_t][p_r].clone()
        }).collect()
    }).collect();
    (full.0.clone(), tm, rm)
}

/// Build the connected-solve preflop tree + the player-node → local-infoset map.
///
/// MUST mirror `gpu_conn_solve` exactly: the blueprint's preflop indexing depends
/// on identical node ordering (player nodes numbered in tree order). Returns
/// `(tree, pre_local, pre_ninfo, pre_maxna)` where `pre_local[node]` is the local
/// infoset index of a player node (or `-1` for chance/terminal).
pub fn build_conn_preflop_tree(np: usize, nraises: usize) -> (FlatTree, Vec<i32>, usize, usize) {
    let mut spec = production_game_v1();
    spec.num_players = np as u8;
    // RICH preflop menu: `nraises` pot-relative raise sizes PLUS an explicit ALL-IN
    // (na = fold + call + nraises + all-in; nraises=5 ⇒ na=8 = CL_MAXNA ceiling). The
    // preflop all-in is now AFFORDABLE because SPR-binning + the multiway postflop
    // taper (build_conn_cell_tree) freed the regret budget — earlier it busted u32
    // only because the exact-(commit,pot) postflop explosion left no room.
    const RICH: [f64; 13] = [1.0, 0.5, 0.75, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 11.0];
    let pre_bets = if nraises > 0 {
        let mut raises: Vec<BetSize> =
            RICH.iter().take(nraises).map(|&f| BetSize::PotRelative(f)).collect();
        raises.push(BetSize::AllIn);
        BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: raises }
    } else {
        BetSizeOptions { bet: vec![], raise: vec![] }
    };
    let mut pcfg = spec.preflop_tree_config(pre_bets);
    if nraises > 0 {
        pcfg.max_bets_per_street = BetCap::all(3);
        // RAISE-OR-FOLD opens + FLAT-CALL defense — MATCHES the deployed EQR preflop
        // (position-aware-tier solve, preflop_eqr_*). The connected blueprint FREEZES that
        // preflop, so this tree must equal the one EQR was solved on, byte-for-byte. Open-
        // limping is dominated; the loose-pool limp model (was no_open_limp=false) is dropped.
        pcfg.no_open_limp = true;
        pcfg.threebet_or_fold = false;
    }
    let pft = build_tree_preflop_only(&pcfg).expect("preflop tree");
    let pnn = pft.num_nodes();
    let mut pre_local = vec![-1i32; pnn];
    let mut pre_ninfo = 0usize;
    let mut pre_maxna = 0usize;
    for i in 0..pnn {
        if pft.nodes[i].is_player() {
            pre_local[i] = pre_ninfo as i32;
            pre_ninfo += 1;
            pre_maxna = pre_maxna.max(pft.nodes[i].num_children as usize);
        }
    }
    (pft, pre_local, pre_ninfo, pre_maxna)
}

/// Tapered postflop bet menu per street (Pluribus). FLOP "more coarse": fold/call
/// + {0.33, 0.66, 1, 1.5}× pot + all-in (re-raise {0.66, 1}× + all-in). TURN/RIVER
/// = the PUBLISHED Pluribus sizes: first {0.5, 1}× + all-in, subsequent {1}× +
/// all-in. cap-3 raises/street. SHARED by `build_shrunk_cell` (solve) and
/// `ConnCellLayout` (runtime) so the cell trees match byte-for-byte.
pub fn build_conn_cell_tree(live: u8, commit: i32, pot: i32) -> FlatTree {
    let pr = BetSize::PotRelative;
    // PER-LIVE TAPER. Multiway postflop trees explode combinatorially (6-way is
    // ~600× HU) and dominate the regret budget (live 5-6 = 92% of the floats), yet
    // rich multiway sizing is both rare AND low-EV (multiway pots play simpler).
    // So: rich HU/3-way (where the EV + volume is), medium 4-way, minimal 5-6-way.
    // SHARED by solve + runtime ⇒ byte-exact. The flop, turn/river menus and the
    // raise CAP all taper together.
    let (flop, tr, cap) = match live {
        // HU / 3-way: FULL rich menu (exact cells — the bulk of postflop volume).
        0..=3 => (
            BetSizeOptions {
                bet: vec![pr(0.33), pr(0.66), pr(1.0), pr(1.5), BetSize::AllIn],
                raise: vec![pr(0.66), pr(1.0), BetSize::AllIn],
            },
            BetSizeOptions {
                bet: vec![pr(0.5), pr(1.0), BetSize::AllIn],
                raise: vec![pr(1.0), BetSize::AllIn],
            },
            3u8,
        ),
        // 4-way: medium menu (3 sizes + all-in, cap-2) so the cells stay small
        // enough to afford MANY fine SPR bins within the leftover u32 budget — a
        // better fidelity trade for 4-way than a few rich bins.
        4 => (
            BetSizeOptions {
                bet: vec![pr(0.66), pr(1.0), BetSize::AllIn],
                raise: vec![pr(1.0), BetSize::AllIn],
            },
            BetSizeOptions { bet: vec![pr(1.0), BetSize::AllIn], raise: vec![BetSize::AllIn] },
            2,
        ),
        // 5-6-way: one shared cell each (~never reached). Lean menu — a single pot
        // bet, no raise war, no all-in (all-in-only multiway overflows the builder).
        _ => (
            BetSizeOptions { bet: vec![pr(1.0)], raise: vec![] },
            BetSizeOptions { bet: vec![pr(1.0)], raise: vec![] },
            1,
        ),
    };
    let mut cfg = production_game_v1().flop_seam_config(live, commit, pot, flop);
    cfg.max_bets_per_street = BetCap::all(cap);
    build_tree_with_bet_override(&cfg, &[(BoardState::Turn, tr.clone()), (BoardState::River, tr)])
        .expect("conn cell tree")
}

/// Deterministic runout grid for a canonical flop: `nt` evenly-spaced turn cards,
/// `nr` evenly-spaced rivers per turn (1×/2× positions pinned to the anchor-
/// certified ones). SINGLE SOURCE shared by `build_shrunk_cell`, the GS14
/// precompute, and the runtime `FlopLayout` — so cached maps and runtime lookups
/// index the same ordering. At the full grid (nt=49, nr=48) this is just the
/// remaining deck in card order.
pub fn runout_grid(canonical: [Card; 3], nt: usize, nr: usize) -> (Vec<Card>, Vec<Vec<u8>>) {
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let nd = deck.len();
    let tp: Vec<usize> = match nt {
        1 => vec![12],
        2 => vec![12, 36],
        n => (0..n).map(|i| i * nd / n).collect(),
    };
    let turns: Vec<Card> = tp.iter().map(|&p| deck[p]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turns {
        let rd: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        let nrd = rd.len();
        let rp: Vec<usize> = match nr {
            1 => vec![10],
            2 => vec![10, 30],
            n => (0..n).map(|j| j * nrd / n).collect(),
        };
        river_decks[tc as usize] = rp.iter().map(|&p| rd[p]).collect();
    }
    (turns, river_decks)
}

/// Runtime hand/runout ordering for one canonical flop, mirroring exactly the
/// enumeration the GS14 maps were built against (`FlopChanceTable` hand order =
/// `index_to_card_pair` over non-board-colliding hands; turn/river order =
/// [`runout_grid`]). Lets a raw `(hero cards, turn, river)` be turned into the
/// `(hand_idx, ti, ri)` triple that indexes a [`Gs14Maps`]. Cheap to build
/// (combinatorial only — no strengths) and intended to be cached per flop.
pub struct FlopLayout {
    /// global hand index (`card_pair_to_index`) → local hand position, or -1 if
    /// the hand collides with the board / is absent.
    hand_pos: Vec<i32>,
    /// turn card (0..52) → turn index `ti`, or -1 if not in the grid.
    turn_pos: Vec<i32>,
    /// `[ti]`: river card (0..52) → river index `ri`, or -1.
    river_pos: Vec<Vec<i32>>,
}

impl FlopLayout {
    pub fn for_canonical(flop: [Card; 3], nt: usize, nr: usize) -> Self {
        use crate::card::{index_to_card_pair, NUM_POSSIBLE_HANDS};
        let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
        // Hand order: 0..NUM_POSSIBLE_HANDS, skipping board collisions (verbatim
        // FlopChanceTable::build_full_nh_sampled ordering).
        let mut hand_pos = vec![-1i32; NUM_POSSIBLE_HANDS];
        let mut local = 0i32;
        for idx in 0..NUM_POSSIBLE_HANDS {
            let (c1, c2) = index_to_card_pair(idx);
            if board_mask & ((1u64 << c1) | (1u64 << c2)) == 0 {
                hand_pos[idx] = local;
                local += 1;
            }
        }
        let (turns, river_decks) = runout_grid(flop, nt, nr);
        let mut turn_pos = vec![-1i32; 52];
        let mut river_pos = vec![vec![-1i32; 52]; turns.len()];
        for (ti, &tc) in turns.iter().enumerate() {
            turn_pos[tc as usize] = ti as i32;
            for (ri, &rc) in river_decks[tc as usize].iter().enumerate() {
                river_pos[ti][rc as usize] = ri as i32;
            }
        }
        FlopLayout { hand_pos, turn_pos, river_pos }
    }

    /// Local hand index for a hero hole-card pair (order-independent), or None if
    /// the hand collides with the board.
    pub fn hand_idx(&self, hero: (Card, Card)) -> Option<usize> {
        let hi = crate::card::card_pair_to_index(hero.0, hero.1);
        let p = self.hand_pos[hi];
        if p < 0 { None } else { Some(p as usize) }
    }

    pub fn turn_idx(&self, turn: Card) -> Option<usize> {
        let p = self.turn_pos[turn as usize];
        if p < 0 { None } else { Some(p as usize) }
    }

    pub fn river_idx(&self, ti: usize, river: Card) -> Option<usize> {
        let p = *self.river_pos.get(ti)?.get(river as usize)?;
        if p < 0 { None } else { Some(p as usize) }
    }

    /// GS14 flop bucket for the hero on this flop (None if hand dead / off-grid).
    pub fn flop_bucket(&self, maps: &Gs14Maps, hero: (Card, Card)) -> Option<u16> {
        let h = self.hand_idx(hero)?;
        match maps.0.get(h).copied()? {
            u16::MAX => None,
            b => Some(b),
        }
    }

    /// GS14 turn bucket (None if hand dead / turn off-grid).
    pub fn turn_bucket(&self, maps: &Gs14Maps, hero: (Card, Card), turn: Card) -> Option<u16> {
        let h = self.hand_idx(hero)?;
        let ti = self.turn_idx(turn)?;
        match maps.1.get(ti)?.get(h).copied()? {
            u16::MAX => None,
            b => Some(b),
        }
    }

    /// GS14 river bucket (None if hand dead / runout off-grid).
    pub fn river_bucket(&self, maps: &Gs14Maps, hero: (Card, Card), turn: Card, river: Card) -> Option<u16> {
        let h = self.hand_idx(hero)?;
        let ti = self.turn_idx(turn)?;
        let ri = self.river_idx(ti, river)?;
        match maps.2.get(ti)?.get(ri)?.get(h).copied()? {
            u16::MAX => None,
            b => Some(b),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConnHeader {
    pub np: usize,
    pub nraises: usize,
    pub pre_ninfo: usize,
    pub nclass: usize,
    pub maxna: usize,
    pub nf: usize,
    pub post_stride: usize,
    pub nh: usize,
    pub nb: usize,
    pub nt: usize,
    pub nr: usize,
}

/// A loaded connected blueprint: header + rebuilt preflop tree + the avg-strategy
/// cumulant buffer (`gcum`).
pub struct ConnBlueprint {
    pub hdr: ConnHeader,
    pub pft: FlatTree,
    pub pre_local: Vec<i32>,
    /// Average-strategy cumulant: preflop region (`pre_ninfo·169·maxna`) then the
    /// per-flop postflop regions (`nf·post_stride`).
    pub gcum: Vec<f32>,
}

impl ConnBlueprint {
    /// Parse a BLP2 file and rebuild the preflop tree.
    pub fn load(path: &str) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        let err = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        if data.len() < 48 {
            return Err(err("blueprint shorter than BLP2 header"));
        }
        let rd = |i: usize| {
            u32::from_le_bytes([data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]]) as usize
        };
        if rd(0) as u32 != BLP2_MAGIC {
            return Err(err("not a BLP2 blueprint (bad magic)"));
        }
        let hdr = ConnHeader {
            np: rd(1),
            nraises: rd(2),
            pre_ninfo: rd(3),
            nclass: rd(4),
            maxna: rd(5),
            nf: rd(6),
            post_stride: rd(7),
            nh: rd(8),
            nb: rd(9),
            nt: rd(10),
            nr: rd(11),
        };
        if hdr.nclass != NUM_PREFLOP_CLASSES {
            return Err(err("blueprint nclass != 169"));
        }
        let nfloats = hdr.pre_ninfo * hdr.nclass * hdr.maxna + hdr.nf * hdr.post_stride;
        let off = 48usize;
        if data.len() < off + nfloats * 4 {
            return Err(err("blueprint float region truncated"));
        }
        let mut gcum = Vec::with_capacity(nfloats);
        for i in 0..nfloats {
            let b = off + i * 4;
            gcum.push(f32::from_le_bytes([data[b], data[b + 1], data[b + 2], data[b + 3]]));
        }
        let (pft, pre_local, pre_ninfo, pre_maxna) = build_conn_preflop_tree(hdr.np, hdr.nraises);
        if pre_ninfo != hdr.pre_ninfo {
            return Err(err("rebuilt preflop tree disagrees with header pre_ninfo (np/nraises mismatch)"));
        }
        if pre_maxna > hdr.maxna {
            return Err(err("rebuilt preflop tree maxna exceeds header maxna"));
        }
        Ok(ConnBlueprint { hdr, pft, pre_local, gcum })
    }

    /// The preflop root (first player to act) — node 0 by tree construction.
    pub fn root(&self) -> usize {
        0
    }

    /// Normalized average-strategy action distribution at a preflop PLAYER node
    /// for a hand class. CFR+ convention: floor cumulants at 0 then normalize;
    /// uniform over the node's actions if the infoset is untrained.
    pub fn preflop_dist_at(&self, node: usize, class: usize) -> Vec<f32> {
        let na = self.pft.nodes[node].num_children as usize;
        let local = self.pre_local[node];
        assert!(local >= 0, "node {node} is not a player node");
        let base = (local as usize * self.hdr.nclass + class) * self.hdr.maxna;
        let raw: Vec<f32> = (0..na).map(|a| self.gcum[base + a].max(0.0)).collect();
        let sum: f32 = raw.iter().sum();
        if sum > 0.0 {
            raw.iter().map(|x| x / sum).collect()
        } else {
            vec![1.0 / na as f32; na]
        }
    }

    /// Replay an ordered preflop betting history from the root to the acting
    /// node, then return the hero's action distribution as `(action_label, prob)`
    /// pairs (one per child of the node, in tree order).
    ///
    /// Each history entry is `(action_label, to_total)`: `to_total` is the actor's
    /// TOTAL contribution (units) after the action, used to disambiguate raise
    /// sizes. Returns `None` if the history does not lead to a player node (e.g.
    /// the hand is already closed, or an action doesn't match any child).
    pub fn preflop_action_dist(
        &self,
        hero: (Card, Card),
        history: &[(u8, i32)],
    ) -> Option<Vec<(u8, i32, f32)>> {
        let mut node = self.root();
        for &(label, to_total) in history {
            if !self.pft.nodes[node].is_player() {
                return None;
            }
            let kids = self.pft.node_children(node);
            let mut next = None;
            for &k in kids {
                let kn = &self.pft.nodes[k as usize];
                if kn.action_label != label {
                    continue;
                }
                // bet(3)/raise(4)/allin(5) carry a size — match the total contribution.
                if matches!(label, 3 | 4 | 5) {
                    if kn.amount == to_total {
                        next = Some(k as usize);
                        break;
                    }
                } else {
                    next = Some(k as usize);
                    break;
                }
            }
            node = next?;
        }
        if !self.pft.nodes[node].is_player() {
            return None;
        }
        let class = PreflopClass::from_combo(hero.0, hero.1).index();
        let dist = self.preflop_dist_at(node, class);
        let acting = self.pft.nodes[node].player_id;
        let kids = self.pft.node_children(node);
        Some(
            kids.iter()
                .enumerate()
                .map(|(a, &k)| (self.pft.nodes[k as usize].action_label, self.pft.get_contribution(k as usize, acting), dist[a]))
                .collect(),
        )
    }

    /// Replay an ordered preflop betting history from the root to the acting node
    /// (shared by ConnBlueprint and ShardedConnBlueprint). Returns the node index,
    /// or None if the history doesn't lead to a player node.
    fn replay_preflop_node(pft: &FlatTree, history: &[(u8, i32)]) -> Option<usize> {
        let mut node = 0usize; // root
        for &(label, to_total) in history {
            if !pft.nodes[node].is_player() {
                return None;
            }
            let mut next = None;
            for &k in pft.node_children(node) {
                let kn = &pft.nodes[k as usize];
                if kn.action_label != label {
                    continue;
                }
                if matches!(label, 3 | 4 | 5) {
                    if kn.amount == to_total { next = Some(k as usize); break; }
                } else {
                    next = Some(k as usize); break;
                }
            }
            node = next?;
        }
        if pft.nodes[node].is_player() { Some(node) } else { None }
    }

    /// Find the first preflop player node whose action set includes a fold (the
    /// open-or-fold decision), and the index of that fold action. Mirrors the
    /// solver's gate probe — useful for sanity checks / tests.
    pub fn first_fold_node(&self) -> Option<(usize, usize)> {
        for i in 0..self.pft.num_nodes() {
            if !self.pft.nodes[i].is_player() {
                continue;
            }
            let actor = self.pft.nodes[i].player_id;
            for (a, &k) in self.pft.node_children(i).iter().enumerate() {
                let kn = &self.pft.nodes[k as usize];
                if kn.is_terminal() && (self.pft.get_folded_mask(k as usize) >> actor) & 1 == 1 {
                    return Some((i, a));
                }
            }
        }
        None
    }
}

/// Normalized avg-strategy at a preflop player node from a preflop cum slice.
/// CFR+: floor at 0 + normalize; uniform if untrained. Shared indexing helper.
fn normalize_pre_dist(pre_cum: &[f32], pre_local: &[i32], maxna: usize, node: usize, na: usize, class: usize) -> Vec<f32> {
    let local = pre_local[node];
    let base = (local as usize * NUM_PREFLOP_CLASSES + class) * maxna;
    let raw: Vec<f32> = (0..na).map(|a| pre_cum[base + a].max(0.0)).collect();
    let sum: f32 = raw.iter().sum();
    if sum > 0.0 { raw.iter().map(|x| x / sum).collect() } else { vec![1.0 / na as f32; na] }
}

/// A sharded connected blueprint on disk: `preflop.ssbp2` (shared preflop avg
/// strategy) + `flop_NNNN.ssbp2` (per-flop postflop), as written by the per-flop
/// single-pass builder (`MC_BLUEPRINT_BUILD`). The preflop loads + the tree
/// rebuilds at construction; postflop shards decode on demand.
pub struct ShardedConnBlueprint {
    pub dir: String,
    pub np: usize,
    pub nraises: usize,
    pub nb: usize,
    pub nt: usize,
    pub nr: usize,
    pub maxna: usize,
    pub pft: FlatTree,
    pub pre_local: Vec<i32>,
    pub pre_ninfo: usize,
    /// Decoded `preflop.ssbp2` (pre_ninfo · 169 · maxna floats).
    pub pre_cum: Vec<f32>,
}

impl ShardedConnBlueprint {
    /// Load the preflop shard + rebuild the preflop tree. `(np, nraises, nb, nt, nr,
    /// maxna)` are the solve's config (the runtime knows them; a future manifest can
    /// carry them). Validates the decoded preflop length against the rebuilt tree.
    pub fn load(dir: &str, np: usize, nraises: usize, nb: usize, nt: usize, nr: usize, maxna: usize) -> io::Result<Self> {
        let (pft, pre_local, pre_ninfo, pre_maxna) = build_conn_preflop_tree(np, nraises);
        if pre_maxna > maxna {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "tree pre_maxna exceeds maxna"));
        }
        let bytes = std::fs::read(format!("{dir}/preflop.ssbp2"))?;
        let pre_cum = ssbp2_decode_cum(&bytes)?;
        let expect = pre_ninfo * NUM_PREFLOP_CLASSES * maxna;
        if pre_cum.len() != expect {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("preflop shard len {} != pre_ninfo·169·maxna {expect}", pre_cum.len())));
        }
        Ok(ShardedConnBlueprint { dir: dir.to_string(), np, nraises, nb, nt, nr, maxna, pft, pre_local, pre_ninfo, pre_cum })
    }

    /// Preflop action distribution after replaying `history`, for `hero` cards:
    /// `(action_label, hero-total-contribution, prob)` per child of the acting node.
    pub fn preflop_action_dist(&self, hero: (Card, Card), history: &[(u8, i32)]) -> Option<Vec<(u8, i32, f32)>> {
        let node = ConnBlueprint::replay_preflop_node(&self.pft, history)?;
        let na = self.pft.nodes[node].num_children as usize;
        let class = PreflopClass::from_combo(hero.0, hero.1).index();
        let dist = normalize_pre_dist(&self.pre_cum, &self.pre_local, self.maxna, node, na, class);
        let acting = self.pft.nodes[node].player_id;
        let kids = self.pft.node_children(node);
        Some(kids.iter().enumerate().map(|(a, &k)| {
            let kn = &self.pft.nodes[k as usize];
            (kn.action_label, self.pft.get_contribution(k as usize, acting), dist[a])
        }).collect())
    }

    /// Decode a flop's postflop cum shard (post_stride floats). Reads disk each
    /// call — the caller should cache hot flops.
    pub fn postflop_cum(&self, flop_id: usize) -> io::Result<Vec<f32>> {
        let bytes = std::fs::read(format!("{}/flop_{flop_id:04}.ssbp2", self.dir))?;
        ssbp2_decode_cum(&bytes)
    }

    /// FlopLayout (hand/runout → bucket) for a canonical flop, matching the solve's
    /// runout subset (nt × nr).
    pub fn flop_layout(&self, canonical: [Card; 3]) -> FlopLayout {
        FlopLayout::for_canonical(canonical, self.nt, self.nr)
    }
}

/// Reconstructed postflop CELL LAYOUT for a connected blueprint — the runtime-side
/// twin of `gpu_conn_solve`'s setup, built from lib functions only (no
/// `FlopChanceTable`/strengths: the runtime needs the betting trees + node→infoset
/// map + regret offsets, not the showdown machinery). Lets a postflop decision be
/// indexed into a flop's postflop shard. The reconstructed `post_stride` MUST equal
/// the decoded shard length (a byte-exact correctness gate).
pub struct ConnCellLayout {
    /// Representative `(live, commit, pot)` per cell (the first seam seen for a bin).
    pub keys: Vec<(u8, i32, i32)>,
    /// `(live, SPR-bin)` → cell index. SPR-binned so it matches `gpu_conn_solve`.
    pub key_idx: std::collections::HashMap<(u8, i64), usize>,
    pub cell_trees: Vec<FlatTree>,
    pub cell_local: Vec<Vec<i32>>,
    pub cell_reg_base: Vec<usize>,
    pub post_stride: usize,
    pub maxna: usize,
    pub nb: usize,
}

impl ConnCellLayout {
    /// Reconstruct the (live, commit, pot) cells, their betting trees, node→infoset
    /// maps, and regret offsets. `real=false` matches the simple bet-pot blueprint
    /// (the production solve); `real=true` (cap-3 rich postflop) is not yet wired.
    pub fn build(np: usize, nraises: usize, nb: usize) -> Self {
        let (pft, _pl, _pn, pre_maxna) = build_conn_preflop_tree(np, nraises);
        let stack = production_game_v1().stack;
        let mut keys = Vec::new();
        let mut key_idx = std::collections::HashMap::new();
        for i in 0..pft.num_nodes() {
            if pft.nodes[i].is_chance() {
                let bk = conn_seam_bin(&pft, i, np, stack);
                if bk.0 >= 2 && !key_idx.contains_key(&bk) {
                    key_idx.insert(bk, keys.len());
                    keys.push(conn_seam_rep(&pft, i, np));
                }
            }
        }
        let mut cell_trees = Vec::with_capacity(keys.len());
        let mut cell_local = Vec::with_capacity(keys.len());
        let mut cell_ninfo = Vec::with_capacity(keys.len());
        let mut maxna = pre_maxna;
        for &(live, commit, pot) in &keys {
            let tree = build_conn_cell_tree(live, commit, pot);
            let nn = tree.num_nodes();
            let mut local = vec![-1i32; nn];
            let mut ni = 0usize;
            for i in 0..nn {
                if tree.nodes[i].is_player() {
                    local[i] = ni as i32;
                    ni += 1;
                    maxna = maxna.max(tree.nodes[i].num_children as usize);
                }
            }
            cell_trees.push(tree);
            cell_local.push(local);
            cell_ninfo.push(ni);
        }
        let mut cell_reg_base = vec![0usize; keys.len()];
        let mut off = 0usize;
        for ki in 0..keys.len() {
            cell_reg_base[ki] = off;
            off += cell_ninfo[ki] * nb * maxna;
        }
        ConnCellLayout { keys, key_idx, cell_trees, cell_local, cell_reg_base, post_stride: off, maxna, nb }
    }

    /// Postflop action distribution at ANY street. Replays postflop betting
    /// `history` (all actions, flop→turn→river in order) in the `(live, commit, pot)`
    /// cell's tree — auto-advancing through deal (chance) nodes — to the hero's
    /// decision node, then indexes the shard at `(cell, local-infoset, bucket)` using
    /// the bucket for THAT node's street. `buckets = [flop, turn, river]` from
    /// `FlopLayout` (only the decision street's entry must be valid). Returns
    /// `(action_label, prob)` per child, or None if it doesn't resolve.
    pub fn postflop_action_dist(&self, post_cum: &[f32], live: u8, commit: i32, pot: i32, buckets: [usize; 3], history: &[(u8, i32)]) -> Option<Vec<(u8, i32, f32)>> {
        // SPR-bin the decision's (live, commit, pot) to find its cell (the rep tree).
        let stack = production_game_v1().stack;
        let ki = *self.key_idx.get(&conn_seam_bin_from(live, commit, pot, stack))?;
        let tree = &self.cell_trees[ki];
        let mut node = 0usize;
        let mut hi = 0usize;
        loop {
            if tree.nodes[node].is_chance() {
                // deal node (turn/river) — single child in the 1×1 cell tree.
                node = tree.node_children(node)[0] as usize;
                continue;
            }
            if !tree.nodes[node].is_player() {
                return None; // terminal reached before the decision
            }
            if hi >= history.len() {
                break; // hero's decision node
            }
            let (label, to_total) = history[hi];
            let mut next = None;
            for &k in tree.node_children(node) {
                let kn = &tree.nodes[k as usize];
                if kn.action_label != label {
                    continue;
                }
                if matches!(label, 3 | 4 | 5) {
                    if kn.amount == to_total { next = Some(k as usize); break; }
                } else {
                    next = Some(k as usize); break;
                }
            }
            node = next?;
            hi += 1;
        }
        let na = tree.nodes[node].num_children as usize;
        let local = self.cell_local[ki][node];
        if local < 0 {
            return None;
        }
        let street = (tree.nodes[node].board_state as usize).min(2); // 0=flop,1=turn,2=river
        let bucket = buckets[street];
        let base = self.cell_reg_base[ki] + (local as usize * self.nb + bucket) * self.maxna;
        if base + na > post_cum.len() {
            return None;
        }
        let raw: Vec<f32> = (0..na).map(|a| post_cum[base + a].max(0.0)).collect();
        let sum: f32 = raw.iter().sum();
        let dist: Vec<f32> = if sum > 0.0 { raw.iter().map(|x| x / sum).collect() } else { vec![1.0 / na as f32; na] };
        let acting = tree.nodes[node].player_id;
        let kids = tree.node_children(node);
        Some(kids.iter().enumerate().map(|(a, &k)| {
            let kn = &tree.nodes[k as usize];
            (kn.action_label, tree.get_contribution(k as usize, acting), dist[a])
        }).collect())
    }

    /// Convenience: a FLOP-street decision (bucket = hero's flop bucket).
    pub fn flop_action_dist(&self, post_cum: &[f32], live: u8, commit: i32, pot: i32, bucket: usize, history: &[(u8, i32)]) -> Option<Vec<(u8, i32, f32)>> {
        self.postflop_action_dist(post_cum, live, commit, pot, [bucket, 0, 0], history)
    }
}
