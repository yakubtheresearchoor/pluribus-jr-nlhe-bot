//! BATCHED MCCFR vs DCFR — WALL-ESCAPE MEASUREMENT (2026-06-17, branch
//! mccfr-cosolve-probe). Does external-sampling MCCFR's per-iter cost skip the
//! nb^num_opp multiway-showdown enumeration that makes the DCFR connected solve
//! cost explode with player count? Measurement branch, NOT production. The DCFR
//! path (BucketedFlopCfr) is untouched; this binary measures it head-to-head
//! against a batched external-sampling MCCFR on the IDENTICAL shrunk game.
//!
//! This file (step 1): the shrunk-game harness + the DCFR per-iter baseline by
//! live-count, establishing the wall (DCFR iter time should grow steeply with
//! live-count as nb^num_opp). MCCFR engine + the full-tree-enumerated
//! exploitability anchor land in subsequent steps.
//!
//! SHRINK KNOBS (recorded explicitly, env-overridable):
//!   MC_NB     buckets per street (default 6)
//!   MC_ITERS  DCFR iters for the per-iter timing (default 16)
//!   MC_NT/NR  turn/river runout samples (default 1/1 — single runout)
//! Both algorithms run on the game these knobs define; identical footing.

use std::time::Instant;

use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::Card;
use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
use solver_core::tree::builder::{build_tree, build_tree_preflop_only};
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};

/// The shrunk game for one live-count: small bucketed flop→river subgame on a
/// single canonical flop with an nt×nr runout. Returns the tree + table +
/// bucketing so DCFR and (later) MCCFR run on the IDENTICAL object.
pub struct ShrunkGame {
    pub tree: FlatTree,
    pub game: FlopStartGame,
    pub bk: FlopBucketing,
    pub live: u8,
    pub nb: usize,
    /// EXACT 7-card hand strength per (ti, ri, hand) — for Pluribus-faithful showdown
    /// (score actual hands, not bucket-average equity). i32::MIN for runout-conflicting.
    pub strengths: Vec<Vec<Vec<i32>>>,
}

pub fn build_shrunk(live: u8, nb: usize, nt: usize, nr: usize) -> ShrunkGame {
    build_shrunk_cell(live, 2, 12, nb, nt, nr)
}

type Gs14Maps = (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>);
thread_local! {
    // GS14 buckets are live-INDEPENDENT (clustered on the hand's own equity
    // distribution), so for a fixed (flop, nb, runout-grid) the maps are identical
    // across cells. Cache them so the expensive k-means runs once, not per cell.
    static GS14_CACHE: std::cell::RefCell<std::collections::HashMap<(usize, usize, usize, usize), Gs14Maps>>
        = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Per-flop GS14 map cache file: {GS14_CACHE_DIR}/gs14_f{flop}_nb{nb}_{nt}x{nr}.bin.
fn gs14_cache_path(flop_idx: usize, nb: usize, nt: usize, nr: usize) -> std::path::PathBuf {
    let dir = std::env::var("GS14_CACHE_DIR").unwrap_or_else(|_| "gs14_cache".into());
    std::path::PathBuf::from(dir).join(format!("gs14_f{flop_idx}_nb{nb}_{nt}x{nr}.bin"))
}
fn read_u16_vec(data: &[u8], off: &mut usize, n: usize) -> Vec<u16> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n { v.push(u16::from_le_bytes([data[*off], data[*off + 1]])); *off += 2; }
    v
}
/// Binary: magic, nh, nt, nr, nb (u32 LE) then flop_map, turn_map[ti], river_map[ti][ri] (u16 LE).
fn save_gs14(path: &std::path::Path, m: &Gs14Maps, nb: usize, nt: usize, nr: usize) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
    let tmp = path.with_extension("tmp");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    let nh = m.0.len();
    for v in [0x4753_3134u32, nh as u32, nt as u32, nr as u32, nb as u32] { w.write_all(&v.to_le_bytes())?; }
    let mut wr = |s: &[u16]| -> std::io::Result<()> {
        let mut b = Vec::with_capacity(s.len() * 2);
        for &x in s { b.extend_from_slice(&x.to_le_bytes()); }
        w.write_all(&b)
    };
    wr(&m.0)?;
    for ti in 0..nt { wr(&m.1[ti])?; }
    for ti in 0..nt { for ri in 0..nr { wr(&m.2[ti][ri])?; } }
    drop(wr); w.flush()?; drop(w);
    std::fs::rename(&tmp, path) // atomic publish
}
fn load_gs14(path: &std::path::Path, nb: usize, nt: usize, nr: usize) -> Option<Gs14Maps> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 20 { return None; }
    let rd = |i: usize| u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
    if rd(0) != 0x4753_3134 || rd(8) != nt || rd(12) != nr || rd(16) != nb { return None; }
    let nh = rd(4);
    let mut off = 20usize;
    let fm = read_u16_vec(&data, &mut off, nh);
    let tm: Vec<Vec<u16>> = (0..nt).map(|_| read_u16_vec(&data, &mut off, nh)).collect();
    let rm: Vec<Vec<Vec<u16>>> = (0..nt).map(|_| (0..nr).map(|_| read_u16_vec(&data, &mut off, nh)).collect()).collect();
    Some((fm, tm, rm))
}

// subset_gs14 moved to solver_core::blueprint (shared with the runtime search adapter).

/// GS14 potential-aware + EMD hand→bucket maps (Pluribus's information abstraction),
/// fit to the FlopChanceTable's exact hand/runout ordering so they drop into
/// `FlopBucketing::from_maps`. Cached in-process (thread-local) AND on disk.
/// MC_BUCKET_NT/MC_BUCKET_NR: when set (and ≠ nt/nr), load the full-fidelity buckets
/// from that cache and SUBSET to the (nt×nr) solve runout (runout-subset solve).
fn gs14_maps(
    flop_idx: usize, nb: usize, nt: usize, nr: usize,
    table: &FlopChanceTable,
    flop: [solver_core::card::Card; 3],
) -> Gs14Maps {
    use solver_core::abstraction::postflop_buckets::build_postflop_bucketing_for_hands;
    use solver_core::card::Card;
    if let Some(m) = GS14_CACHE.with(|c| c.borrow().get(&(flop_idx, nb, nt, nr)).cloned()) { return m; }
    // Runout-subset: full-fidelity buckets from the (bnt×bnr) cache → (nt×nr) subset.
    let bnt: usize = std::env::var("MC_BUCKET_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(nt);
    let bnr: usize = std::env::var("MC_BUCKET_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(nr);
    if (bnt, bnr) != (nt, nr) {
        let full = load_gs14(&gs14_cache_path(flop_idx, nb, bnt, bnr), nb, bnt, bnr)
            .unwrap_or_else(|| panic!("runout-subset: bucket cache {} missing (need full {bnt}×{bnr} precompute)", gs14_cache_path(flop_idx, nb, bnt, bnr).display()));
        let m = subset_gs14(&full, flop, bnt, bnr, nt, nr);
        GS14_CACHE.with(|c| c.borrow_mut().insert((flop_idx, nb, nt, nr), m.clone()));
        return m;
    }
    let path = gs14_cache_path(flop_idx, nb, nt, nr);
    if let Some(m) = load_gs14(&path, nb, nt, nr) {
        GS14_CACHE.with(|c| c.borrow_mut().insert((flop_idx, nb, nt, nr), m.clone()));
        return m;
    }
    let hands: Vec<(Card, Card)> = (0..table.num_valid)
        .map(|h| (table.hand_cards[h * 2] as Card, table.hand_cards[h * 2 + 1] as Card)).collect();
    let turn_cards: Vec<Card> = table.remaining_deck.iter().map(|&c| c as Card).collect();
    let river_cards: Vec<Vec<Card>> = table.remaining_deck.iter()
        .map(|&tc| table.river_decks[tc as usize].iter().map(|&c| c as Card).collect()).collect();
    let restarts = std::env::var("MC_GS14_RESTARTS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let t0 = Instant::now();
    let pb = build_postflop_bucketing_for_hands(hands, &flop, &turn_cards, &river_cards, nb, nb, nb, restarts, 0xABCDEF01);
    eprintln!("[GS14] flop={flop_idx} nb={nb} runouts={nt}×{nr} built in {:.1}s (potential-aware+EMD, {restarts} restarts)", t0.elapsed().as_secs_f64());
    let m = (pb.flop_map, pb.turn_map, pb.river_map);
    let _ = save_gs14(&path, &m, nb, nt, nr);
    GS14_CACHE.with(|c| c.borrow_mut().insert((flop_idx, nb, nt, nr), m.clone()));
    m
}

// runout_grid moved to solver_core::blueprint (single source shared with the
// runtime FlopLayout, so cached maps and runtime lookups index one ordering).
use solver_core::blueprint::{runout_grid, subset_gs14};

/// Build + disk-cache the GS14 maps for one flop (no env; parallel-safe). Returns the
/// build time in seconds (0.0 if already cached). `canonical` must come from the SAME
/// list build_shrunk_cell uses (PreflopChanceTable.canonical_flops) so flop_idx aligns.
fn precompute_flop(canonical: [solver_core::card::Card; 3], flop_idx: usize, nb: usize, nt: usize, nr: usize) -> f64 {
    if gs14_cache_path(flop_idx, nb, nt, nr).exists() { return 0.0; }
    let (turns, river_decks) = runout_grid(canonical, nt, nr);
    let table = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
    let t0 = Instant::now();
    let _ = gs14_maps(flop_idx, nb, nt, nr, &table, canonical); // builds + saves to disk
    t0.elapsed().as_secs_f64()
}

/// MC_GS14_PRECOMPUTE: parallel precompute of per-flop GS14 nb-bucket maps → disk cache.
/// Range via MC_FLOP_LO/MC_FLOP_HI (default all 1755 canonical flops). Resumable
/// (skips already-cached). Each flop independent ⇒ rayon across flops (k-means also
/// rayon — shared pool, work-stealing balances).
fn gs14_precompute() {
    use rayon::prelude::*;
    let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let nt: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(49);
    let nr: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(48);
    let flops = PreflopChanceTable::new(6, vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6]).canonical_flops;
    let nflop = flops.len();
    let lo: usize = std::env::var("MC_FLOP_LO").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let hi: usize = std::env::var("MC_FLOP_HI").ok().and_then(|s| s.parse().ok()).unwrap_or(nflop).min(nflop);
    let todo: Vec<usize> = (lo..hi).filter(|&fi| !gs14_cache_path(fi, nb, nt, nr).exists()).collect();
    println!("GS14 PRECOMPUTE — nb={nb} runout={nt}×{nr}: {} flops to build (of {nflop} canonical, range {lo}..{hi})", todo.len());
    let done = std::sync::atomic::AtomicUsize::new(0);
    let t0 = Instant::now();
    todo.par_iter().for_each(|&fi| {
        let secs = precompute_flop(flops[fi], fi, nb, nt, nr);
        let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        println!("  [{d}/{}] flop {fi} built in {secs:.1}s (elapsed {:.0}s)", todo.len(), t0.elapsed().as_secs_f64());
    });
    println!("DONE — {} flops in {:.0}s ({:.1}s/flop avg)", todo.len(), t0.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64() / todo.len().max(1) as f64);
}

/// Build the shrunk postflop subgame for an explicit seam cell (commit/pot), so a
/// connected solve can match the postflop subgame to the preflop line that enters it.
/// Process-cached canonical flop list (1755 suit-iso classes). Deterministic, so
/// build it ONCE — `PreflopChanceTable::new` enumerates all flops and was being
/// re-run on every `build_shrunk_cell` call (×#cells ×#flops → the np=6 setup
/// wall). The cached vec is keyed only by the (fixed) 6-player uniform table.
fn canonical_flops_cached() -> &'static Vec<[solver_core::card::Card; 3]> {
    use std::sync::OnceLock;
    static CF: OnceLock<Vec<[solver_core::card::Card; 3]>> = OnceLock::new();
    CF.get_or_init(|| {
        PreflopChanceTable::new(6, vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6]).canonical_flops
    })
}

/// Write an f32 slab as an SSBP2 file (u8-quant + zstd; level from MC_BP_ZLEVEL, default 9).
fn ssbp2_write(path: &str, slab: &[f32]) {
    let z: i32 = std::env::var("MC_BP_ZLEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(9);
    std::fs::write(path, solver_core::blueprint::ssbp2_encode_cum(slab, z)).expect("write ssbp2");
}

pub fn build_shrunk_cell(live: u8, commit: i32, pot: i32, nb: usize, nt: usize, nr: usize) -> ShrunkGame {
    let spec = production_game_v1();
    // MC_REAL: the PRODUCTION action set — bet=pot, raise={0.5,1.0,...}pot (mrc =
    // MAX_NA_POSTFLOP-2), cap-3 (BetCap::all(3)). A bigger tree (re-raise sequences,
    // more terminal types) than the simple single-bet tree — Phase-0 re-certifies
    // the anchor on THIS, since the card-removal/per-street bugs were found on the
    // simpler tree and the real tree must be re-checked.
    let cflops = canonical_flops_cached();
    let flop_idx: usize = std::env::var("MC_FLOP").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let canonical = cflops[flop_idx.min(cflops.len() - 1)];
    let (turns, river_decks) = runout_grid(canonical, nt, nr);
    // TAPERED postflop menu (Pluribus) — SHARED with the runtime ConnCellLayout.
    let tree = solver_core::blueprint::build_conn_cell_tree(live, commit, pot);
    let table = FlopChanceTable::build_full_nh_sampled(canonical, live, &turns, &river_decks);
    // MC_IDENTITY=1: identity bucketing (nb=nh) ⇒ the bucketed game IS the exact
    // game, so converged DCFR is a TRUE Nash (exploitability→0) — the localization
    // test for whether the anchor floor is structural or bucketing-specific.
    let bk = if std::env::var("MC_IDENTITY").is_ok() {
        FlopBucketing::identity(&table)
    } else if std::env::var("MC_GS14").is_ok() {
        // FAITHFUL: GS14 potential-aware + earth-mover-distance buckets (Pluribus's
        // information abstraction) instead of quantile. Live-INDEPENDENT (clusters on
        // the hand's own equity distribution), so one build serves all cells — cache
        // the maps, rebuild the cheap per-runout showdown tables via from_maps.
        let (fm, tm, rm) = gs14_maps(flop_idx, nb, nt, nr, &table, [canonical[0], canonical[1], canonical[2]]);
        FlopBucketing::from_maps(&table, nb, nb, nb, fm, tm, rm)
    } else {
        FlopBucketing::quantile(&table, nb)
    };
    let mut tree = tree;
    if std::env::var("MC_NORAKE").is_ok() {
        tree.rake_rate = 0.0;
        tree.rake_cap = 0.0;
    }
    // EXACT showdown support: per-(ti,ri) 7-card strength for every hand (Pluribus
    // scores actual hands at showdown; abstraction is for infosets only, not payoffs).
    let hands_v: Vec<(Card, Card)> = (0..table.num_valid)
        .map(|h| (table.hand_cards[h * 2] as Card, table.hand_cards[h * 2 + 1] as Card)).collect();
    let strengths: Vec<Vec<Vec<i32>>> = turns.iter().map(|&tc| {
        river_decks[tc as usize].iter().map(|&rc| {
            solver_core::abstraction::postflop_buckets::strengths_at_river(&hands_v, &canonical, tc, rc)
        }).collect()
    }).collect();
    let game = FlopStartGame::new(table);
    ShrunkGame { tree, game, bk, live, nb, strengths }
}

// ─────────────────────────────────────────────────────────────────────
// CPU batched external-sampling MCCFR (step 2). Reuses BucketedFlopCfr's
// river_tables + bucketing so the showdown is the IDENTICAL bucketed game as
// DCFR. External sampling: sample a hand per player (→ bucket trajectory),
// traverse the traverser's own actions, sample opponent actions; at the river
// run the per-tuple showdown DP for the ONE sampled opponent-bucket-tuple —
// O(num_opp) — instead of DCFR's O(nb^num_opp) enumeration. NOTE: this step
// targets the PER-ITER (wall-escape) number; the enumerated-exploitability
// convergence anchor + the garbage check are the explicitly-next milestone,
// so the convergence/quality of this engine is NOT yet trusted.
// ─────────────────────────────────────────────────────────────────────
/// Pluribus negative-regret ("deep") pruning config — Brown & Sandholm 2019,
/// Science Supplement, Algorithm 1 (MCCFR with Negative-Regret Pruning).
/// VERBATIM PARAMS (absolute, Pluribus-scale): prune traverser actions with
/// regret < -300,000,000 AND their whole subtree, in 95% of iters, EXCEPT the
/// last betting round and actions immediately leading to a terminal; regret
/// floor -310,000,000 (below the prune threshold so pruned actions can recover);
/// warm-up before pruning starts; Linear-CFR discount d=(t/DI)/(t/DI+1) for
/// t<LCFR_Threshold. The absolute -300M/-310M are tuned to Pluribus's regret
/// magnitude (trillions of iters); they do NOT transfer to the shrunk game
/// (measured max|regret| ~1e6 here), so we carry them as game-scaled knobs and
/// SWEEP the threshold (it trades the speed and the quality being measured) —
/// per the convergence-measurement discipline. NOT yet wired into the traversal;
/// recorded so the convergence runs match Pluribus exactly, not from memory.
#[derive(Clone, Copy)]
struct PruneConfig {
    /// prune threshold C (game-scaled; default a multiple of the regret scale).
    c: f32,
    /// regret floor (must be < c so pruned actions can climb back to un-prune).
    floor: f32,
    /// probability an iteration prunes (Pluribus: 0.95).
    prune_prob: f32,
    /// iterations of warm-up before pruning engages.
    warmup: u32,
    /// never prune on the final betting round or terminal-adjacent actions.
    protect_last_round: bool,
}

impl Default for PruneConfig {
    fn default() -> Self {
        // Defaults scaled to the shrunk game's regret magnitude; SWEEP c.
        PruneConfig { c: -3.0e5, floor: -3.1e5, prune_prob: 0.95, warmup: 0, protect_last_round: true }
    }
}

struct Mccfr {
    node_local: Vec<i32>, // [nn], dense player-node index or -1
    n_info: usize,
    max_na: usize,
    nb: usize,
    np: usize,
    regret: Vec<f32>, // [n_info * nb * max_na]
    cum: Vec<f32>,
    flop_b: Vec<u16>,
    // FULL runout set (indexed by sampled turn/river outcome): turn_maps[ti],
    // river_maps[ti][ri], river_tabs[ti][ri]. Single-runout (nt=nr=1) ⇒ len-1 vecs
    // ⇒ behaviour identical to the prior fixed-runout engine (anchor-certified).
    turn_maps: Vec<Vec<u16>>,
    river_maps: Vec<Vec<Vec<u16>>>,
    river_tabs: Vec<Vec<solver_core::solver::bucketed_showdown::BucketedRunoutTables>>,
    river_strength: Vec<Vec<Vec<i32>>>, // [ti][ri][hand] exact 7-card strength (EXACT showdown)
    remaining_deck: Vec<u8>,    // turn candidate cards (ti → card)
    river_decks: Vec<Vec<u8>>,  // [turn_card] → river candidate cards (ri → card)
    nh: usize,
    // RUNOUT SAMPLING: each trajectory pre-samples one (turn,river) at deal time
    // (chance is independent of betting ⇒ standard external sampling). All bucket/
    // showdown lookups for the trajectory use (cur_ti, cur_ri).
    cur_ti: usize,
    cur_ri: usize,
    // valid hands (no board/runout conflict) + their 2 cards.
    hand_cards: Vec<(u8, u8)>,
    valid: Vec<usize>, // FLOP-live hands (deal set) — per-street death handled in traverse
    rng: u64,
    // Pluribus pruning (MC_PRUNE): in prune_prob of trajectories, skip traverser
    // actions whose cumulative regret ≤ prune_c (CFR+ floors bad actions to 0, so
    // ≤0 captures the negative-regret-pruning effect) and their subtrees — the
    // compute saving. 5% full traversal lets pruned actions recover; the last
    // betting round (river) is never pruned; warmup before pruning engages.
    prune: bool,
    prune_c: f32,
    prune_warmup: u64,
    iter: u64,
    linear: bool, // MC_LINEAR: weight avg-strategy by iteration t (CFR+ needs this)
    prune_this: bool,
    pruned_nodes: u64, // diagnostics: subtrees skipped (compute saved)
    visited_nodes: u64,
    // VR-MCCFR (MC_VR): control-variate baseline b(opp-infoset, action) — a running
    // estimate of the value returned by sampling that opponent action. At an opp
    // node the returned value = Σ σ(j)·b(j) + (sampled − b(a*)) — UNBIASED (the
    // control variate), lower-variance if b is well-fit. Same layout as regret/cum.
    vr: bool,
    vr_alpha: f32,
    baseline: Vec<f32>,
    // PLURIBUS negative-regret pruning: regret floored at `regret_floor` (Pluribus
    // −3.1e8, game-scaled here; 0 = CFR+, the old behaviour). Allowing regret to go
    // negative is what lets deeply-bad actions STAY pruned (vs CFR+ flooring at 0,
    // which un-prunes the instant regret recovers to 0). Prune below `prune_c`.
    regret_floor: f32,
}

impl Mccfr {
    fn new(g: &ShrunkGame) -> Self {
        use solver_core::tree::action::BoardState;
        let tree = &g.tree;
        let nn = tree.num_nodes();
        let mut node_local = vec![-1i32; nn];
        let mut n_info = 0usize;
        let mut max_na = 0usize;
        for i in 0..nn {
            if tree.nodes[i].is_player() {
                node_local[i] = n_info as i32;
                n_info += 1;
                max_na = max_na.max(tree.nodes[i].num_children as usize);
            }
        }
        // tables + maps straight from the bucketing (identical game as DCFR).
        let nb = g.bk.nb_flop.max(g.bk.nb_turn).max(g.bk.nb_river);
        let table = g.game.table();
        let nh = table.num_valid;
        let hand_cards: Vec<(u8, u8)> =
            (0..nh).map(|h| (table.hand_cards[h * 2], table.hand_cards[h * 2 + 1])).collect();
        let no = u16::MAX;
        // PER-STREET (mirror DCFR): deal from FLOP-live hands (incl. runout-
        // colliders, which are turn/river-live until their card appears). A
        // collider DIES at its collision street (handled in traverse: 0 from
        // there) — a sampled tuple containing a collider is an impossible world
        // → 0, which reproduces DCFR's river-valid-weighted CFV in expectation
        // (no reweighting: sampling already weights by the flop-valid distrib).
        let valid: Vec<usize> = (0..nh).filter(|&h| g.bk.flop_map[h] != no).collect();
        let _ = BoardState::Flop;
        Mccfr {
            node_local,
            n_info,
            max_na,
            nb,
            np: g.live as usize,
            regret: vec![0.0; n_info * nb * max_na],
            cum: vec![0.0; n_info * nb * max_na],
            flop_b: g.bk.flop_map.clone(),
            turn_maps: g.bk.turn_map.clone(),
            river_maps: g.bk.river_map.clone(),
            river_tabs: g
                .bk
                .river_tables
                .iter()
                .map(|rv| rv.iter().map(clone_tables).collect())
                .collect(),
            river_strength: g.strengths.clone(),
            remaining_deck: table.remaining_deck.clone(),
            river_decks: table.river_decks.clone(),
            nh,
            cur_ti: 0,
            cur_ri: 0,
            hand_cards,
            valid,
            rng: 0x9E3779B97F4A7C15,
            prune: std::env::var("MC_PRUNE").is_ok(),
            prune_c: std::env::var("MC_PRUNE_C").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            prune_warmup: std::env::var("MC_PRUNE_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000),
            iter: 0,
            linear: std::env::var("MC_LINEAR").is_ok(),
            prune_this: false,
            pruned_nodes: 0,
            visited_nodes: 0,
            vr: std::env::var("MC_VR").is_ok(),
            vr_alpha: std::env::var("MC_VR_ALPHA").ok().and_then(|s| s.parse().ok()).unwrap_or(0.05),
            baseline: vec![0.0; n_info * nb * max_na],
            regret_floor: std::env::var("MC_FLOOR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0),
        }
    }

    /// hand is live at street `bs` iff it doesn't hold a board card dealt by then
    /// (flop already excluded at deal). Mirrors DCFR's per-street card removal.
    #[inline]
    fn turn_card(&self) -> u8 {
        self.remaining_deck[self.cur_ti]
    }
    #[inline]
    fn river_card(&self) -> u8 {
        self.river_decks[self.turn_card() as usize][self.cur_ri]
    }

    #[inline]
    fn alive(&self, hand: usize, bs: u8) -> bool {
        use solver_core::tree::action::BoardState;
        let (c1, c2) = self.hand_cards[hand];
        if bs >= BoardState::Turn as u8 {
            let tc = self.turn_card();
            if c1 == tc || c2 == tc {
                return false;
            }
        }
        if bs >= BoardState::River as u8 {
            let rc = self.river_card();
            if c1 == rc || c2 == rc {
                return false;
            }
        }
        true
    }

    /// Pre-sample this trajectory's runout (turn,river) uniformly over the in-table
    /// outcomes. nt=nr=1 ⇒ always (0,0). Hands colliding with the sampled board die
    /// per-street via `alive` (matches the fixed-runout convention the anchor certified;
    /// validated against DCFR on the identical table).
    #[inline]
    fn sample_runout(&mut self) {
        let nt = self.remaining_deck.len();
        self.cur_ti = (self.rand() as usize) % nt;
        let tc = self.remaining_deck[self.cur_ti] as usize;
        let nr = self.river_decks[tc].len();
        self.cur_ri = (self.rand() as usize) % nr;
    }

    #[inline]
    fn rand(&mut self) -> u64 {
        // xorshift64
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    #[inline]
    fn bucket_of(&self, board_state: u8, hand: usize) -> usize {
        use solver_core::tree::action::BoardState;
        let m: &[u16] = if board_state == BoardState::Flop as u8 {
            &self.flop_b
        } else if board_state == BoardState::Turn as u8 {
            &self.turn_maps[self.cur_ti]
        } else {
            &self.river_maps[self.cur_ti][self.cur_ri]
        };
        m[hand] as usize
    }

    /// regret-match a node's strategy into `out` (len na).
    fn strategy(&self, local: usize, bucket: usize, na: usize, out: &mut [f32]) {
        let base = (local * self.nb + bucket) * self.max_na;
        let mut sum = 0.0f32;
        for a in 0..na {
            let r = self.regret[base + a].max(0.0);
            out[a] = r;
            sum += r;
        }
        if sum > 0.0 {
            for a in 0..na {
                out[a] /= sum;
            }
        } else {
            let u = 1.0 / na as f32;
            for a in 0..na {
                out[a] = u;
            }
        }
    }

    /// Pure cell rollout under the AVERAGE (cum) strategy: every player samples from
    /// their normalized cumulative strategy; returns `hero`'s terminal value. Used by
    /// the EV-delta check (NOT training — no regret/cum updates). Runout must be
    /// pre-sampled by the caller. Dead actor → impossible world → 0 (matches traverse).
    fn rollout_cell(&mut self, tree: &FlatTree, hero: usize, hands: &[usize]) -> f32 {
        let mut node = 0usize;
        loop {
            let n = &tree.nodes[node];
            if n.is_terminal() {
                return self.terminal(tree, node, hero, hands);
            }
            if n.is_chance() {
                node = tree.node_children(node)[0] as usize;
                continue;
            }
            let player = n.player_id as usize;
            let bs = n.board_state;
            if !self.alive(hands[player], bs) {
                return 0.0;
            }
            let bucket = self.bucket_of(bs, hands[player]);
            let local = self.node_local[node] as usize;
            let na = n.num_children as usize;
            let base = (local * self.nb + bucket) * self.max_na;
            let sum: f32 = (0..na).map(|a| self.cum[base + a].max(0.0)).sum();
            let r = (self.rand() as f64 / u64::MAX as f64) as f32;
            let mut acc = 0.0f32;
            let mut sel = na - 1;
            if sum > 0.0 {
                for i in 0..na {
                    acc += self.cum[base + i].max(0.0) / sum;
                    if r <= acc { sel = i; break; }
                }
            } else {
                sel = ((r * na as f32) as usize).min(na - 1);
            }
            node = tree.node_children(node)[sel] as usize;
        }
    }

    // ── HU exact best-response / exploitability over the bucketed cell ──────────
    // σ = this engine's AVERAGE strategy (cum). Propagates the opponent's per-hand
    // reach down the tree; enumerates the runout grid at chance nodes; at each hero
    // infoset picks the per-bucket argmax action. exploitability = ½(BR0+BR1) → 0 at
    // the abstract Nash. (Showdown ignores hero↔opp card removal — 2nd-order; the
    // bucketed table already encodes pairwise card removal.)
    #[inline]
    fn bucket_at(&self, bs: u8, h: usize, ti: i32, ri: i32) -> usize {
        (if bs == 0 { self.flop_b[h] }
         else if bs == 1 { self.turn_maps[ti as usize][h] }
         else { self.river_maps[ti as usize][ri as usize][h] }) as usize
    }
    #[inline]
    fn avg_action_prob(&self, local: usize, bucket: usize, na: usize, a: usize) -> f64 {
        let base = (local * self.nb + bucket) * self.max_na;
        let sum: f32 = (0..na).map(|x| self.cum[base + x].max(0.0)).sum();
        if sum > 0.0 { (self.cum[base + a].max(0.0) / sum) as f64 } else { 1.0 / na as f64 }
    }

    fn br_terminal(&self, tree: &FlatTree, node: usize, hero: usize, opp_reach: &[f64], ti: i32, ri: i32) -> Vec<f64> {
        let nh = self.nh;
        let mut out = vec![0.0f64; nh];
        let fm = tree.get_folded_mask(node);
        let opp = 1 - hero;
        let c_h = tree.get_contribution(node, hero as u8) as f64;
        let c_o = tree.get_contribution(node, opp as u8) as f64;
        let sp = tree.starting_pot as f64;
        let half_pot = sp / 2.0 + c_h;
        let total = sp + c_h + c_o;
        let minlev = c_h.min(c_o);
        let main_pot = minlev * 2.0 + sp;
        let rake = (main_pot * tree.rake_rate as f64).min(tree.rake_cap as f64).max(0.0);
        let net = total - rake;
        // total opp reach + per-card sums (for hero↔opp card-removal on fold/win mass)
        let mut mass = 0.0f64; let mut rc = [0.0f64; 52];
        for &oh in &self.valid {
            let w = opp_reach[oh]; if w == 0.0 { continue; }
            if !self.alive(oh, tree.nodes[node].board_state) { continue; }
            mass += w;
            let (a, b) = self.hand_cards[oh];
            rc[a as usize] += w; rc[b as usize] += w;
        }
        let hero_folded = (fm >> hero) & 1 == 1;
        let opp_folded = (fm >> opp) & 1 == 1;
        let bs = tree.nodes[node].board_state;
        if hero_folded || opp_folded {
            let sign = if hero_folded { -half_pot } else { net - half_pot };
            for &hh in &self.valid {
                if self.bucket_at(bs, hh, ti, ri) >= self.nb { continue; }
                let (c1, c2) = self.hand_cards[hh];
                let m = (mass - rc[c1 as usize] - rc[c2 as usize] + opp_reach[hh]).max(0.0);
                out[hh] = sign * m;
            }
            return out;
        }
        // EXACT showdown (Pluribus-faithful): per hero hand, integrate the opponent's
        // reach over ACTUAL pairwise outcomes (strength compare), with hero↔opp card
        // removal. O(nh²) per terminal — fine for a one-time exploitability measure.
        let st = &self.river_strength[ti as usize][ri as usize];
        for &hh in &self.valid {
            let s_h = st[hh]; if s_h == i32::MIN { continue; } // hero dead at this runout
            let (hc1, hc2) = self.hand_cards[hh];
            let mut v = 0.0f64;
            for &oh in &self.valid {
                let w = opp_reach[oh]; if w == 0.0 { continue; }
                let (oc1, oc2) = self.hand_cards[oh];
                if oc1 == hc1 || oc1 == hc2 || oc2 == hc1 || oc2 == hc2 { continue; } // shared card → impossible
                let s_o = st[oh]; if s_o == i32::MIN { continue; }
                v += w * if s_h > s_o { net - half_pot } else if s_h == s_o { net / 2.0 - half_pot } else { -half_pot };
            }
            out[hh] = v;
        }
        out
    }

    fn br(&self, tree: &FlatTree, node: usize, hero: usize, opp_reach: &[f64], ti: i32, ri: i32) -> Vec<f64> {
        let n = &tree.nodes[node];
        if n.is_terminal() { return self.br_terminal(tree, node, hero, opp_reach, ti, ri); }
        let nh = self.nh;
        if n.is_chance() {
            let child = tree.node_children(node)[0] as usize;
            let cbs = tree.nodes[child].board_state;
            let mut acc = vec![0.0f64; nh];
            if cbs == 1 {
                let nt = self.remaining_deck.len();
                for tii in 0..nt {
                    let tc = self.remaining_deck[tii];
                    let mut r2 = opp_reach.to_vec();
                    for &h in &self.valid { let (a, b) = self.hand_cards[h]; if a == tc || b == tc { r2[h] = 0.0; } }
                    let v = self.br(tree, child, hero, &r2, tii as i32, -1);
                    for h in 0..nh { acc[h] += v[h] / nt as f64; }
                }
            } else {
                let tc = self.remaining_deck[ti as usize];
                let rd = &self.river_decks[tc as usize];
                let nr = rd.len();
                for rii in 0..nr {
                    let rc = rd[rii];
                    let mut r2 = opp_reach.to_vec();
                    for &h in &self.valid { let (a, b) = self.hand_cards[h]; if a == rc || b == rc { r2[h] = 0.0; } }
                    let v = self.br(tree, child, hero, &r2, ti, rii as i32);
                    for h in 0..nh { acc[h] += v[h] / nr as f64; }
                }
            }
            return acc;
        }
        let player = n.player_id as usize;
        let bs = n.board_state;
        let na = n.num_children as usize;
        let local = self.node_local[node] as usize;
        let kids = tree.node_children(node).to_vec();
        if player != hero {
            let mut acc = vec![0.0f64; nh];
            for a in 0..na {
                let mut ra = vec![0.0f64; nh];
                for &oh in &self.valid {
                    let w = opp_reach[oh]; if w == 0.0 { continue; }
                    let bk = self.bucket_at(bs, oh, ti, ri); if bk >= self.nb { continue; }
                    ra[oh] = w * self.avg_action_prob(local, bk, na, a);
                }
                let v = self.br(tree, kids[a] as usize, hero, &ra, ti, ri);
                for h in 0..nh { acc[h] += v[h]; }
            }
            acc
        } else {
            let child_vals: Vec<Vec<f64>> = (0..na).map(|a| self.br(tree, kids[a] as usize, hero, opp_reach, ti, ri)).collect();
            let nb = self.nb;
            let prior = 1.0 / self.valid.len() as f64;
            let mut bucket_val = vec![vec![0.0f64; na]; nb];
            for &hh in &self.valid {
                let bk = self.bucket_at(bs, hh, ti, ri); if bk >= nb { continue; }
                for a in 0..na { bucket_val[bk][a] += prior * child_vals[a][hh]; }
            }
            let best_a: Vec<usize> = (0..nb).map(|b| {
                let (mut ba, mut bv) = (0usize, f64::NEG_INFINITY);
                for a in 0..na { if bucket_val[b][a] > bv { bv = bucket_val[b][a]; ba = a; } }
                ba
            }).collect();
            let mut out = vec![0.0f64; nh];
            for &hh in &self.valid {
                let bk = self.bucket_at(bs, hh, ti, ri); if bk >= nb { continue; }
                out[hh] = child_vals[best_a[bk]][hh];
            }
            out
        }
    }

    /// Exploitability of σ (=cum) in the bucketed HU cell, in chips. → 0 at abstract Nash.
    fn hu_exploitability(&self, tree: &FlatTree) -> f64 {
        assert_eq!(self.np, 2, "HU only");
        let prior = 1.0 / self.valid.len() as f64;
        let mut init = vec![0.0f64; self.nh];
        for &h in &self.valid { init[h] = prior; }
        let br0 = self.br(tree, 0, 0, &init, -1, -1);
        let br1 = self.br(tree, 0, 1, &init, -1, -1);
        let v0: f64 = self.valid.iter().map(|&h| prior * br0[h]).sum();
        let v1: f64 = self.valid.iter().map(|&h| prior * br1[h]).sum();
        (v0 + v1) / 2.0
    }

    /// Sample one distinct hand per player (no shared cards), and pre-sample the
    /// trajectory's runout.
    fn sample_deal(&mut self) -> Vec<usize> {
        self.sample_runout();
        let nv = self.valid.len();
        let mut hands = vec![0usize; self.np];
        let mut used: u64 = 0;
        for p in 0..self.np {
            loop {
                let r = (self.rand() as usize) % nv;
                let h = self.valid[r];
                let (c1, c2) = self.hand_cards[h];
                let m = (1u64 << c1) | (1u64 << c2);
                if used & m == 0 {
                    used |= m;
                    hands[p] = h;
                    break;
                }
            }
        }
        hands
    }

    /// External-sampling traverse for `traverser`; returns traverser CFV.
    // `reach` = the TRAVERSER's own reach π_i to this node (product of its strategy
    // probs on the path). The average strategy MUST be accumulated reach-weighted
    // (cum += π_i·σ) — naive cum += σ does NOT converge to Nash (exploitability grows
    // with T; the bug the HU exploitability check caught). Mirrors the trusted DCFR.
    fn traverse(&mut self, tree: &FlatTree, node: usize, traverser: usize, hands: &[usize], reach: f32) -> f32 {
        let n = &tree.nodes[node];
        if n.is_terminal() {
            return self.terminal(tree, node, traverser, hands);
        }
        let kids = tree.node_children(node).to_vec();
        if n.is_chance() {
            // 1×1 runout: follow the single sampled child (chance doesn't change π_i).
            return self.traverse(tree, kids[0] as usize, traverser, hands, reach);
        }
        let player = n.player_id as usize;
        let na = kids.len();
        let bs = n.board_state;
        // per-street death: the acting player's hand is impossible at this street
        // (holds a board card dealt by now) → impossible world → 0. Also guards the
        // NO_BUCKET strategy index under quantile.
        if !self.alive(hands[player], bs) {
            return 0.0;
        }
        let local = self.node_local[node] as usize;
        let bucket = self.bucket_of(bs, hands[player]);
        let mut strat = [0.0f32; 16];
        self.strategy(local, bucket, na, &mut strat[..na]);
        if player == traverser {
            let base = (local * self.nb + bucket) * self.max_na;
            // PLURIBUS PRUNING: in a prune trajectory, skip the subtrees of actions
            // whose cumulative regret ≤ prune_c (≈0-probability under regret-match,
            // CFR+ floors bad actions to 0). Protect the last betting round (river):
            // never prune there. Skipped actions contribute ≈0 to v (strat≈0) so the
            // value is ≈unbiased; the saving is the un-recursed subtree.
            let river = solver_core::tree::action::BoardState::River as u8;
            let prunable = self.prune_this && bs != river;
            let mut cv = [0.0f32; 16];
            let mut v = 0.0f32;
            let mut pruned = [false; 16];
            for a in 0..na {
                if prunable && self.regret[base + a] <= self.prune_c {
                    pruned[a] = true; // skip subtree
                    self.pruned_nodes += 1;
                    continue;
                }
                self.visited_nodes += 1;
                cv[a] = self.traverse(tree, kids[a] as usize, traverser, hands, reach * strat[a]);
                v += strat[a] * cv[a];
            }
            for a in 0..na {
                if pruned[a] { continue; } // pruned actions: regret/cum unchanged this traj
                // Floor cumulative regret (Pluribus: regret_floor<0 ⇒ negative-regret
                // pruning; 0 ⇒ CFR+).
                self.regret[base + a] = (self.regret[base + a] + cv[a] - v).max(self.regret_floor);
                // CFR+ needs LINEAR averaging (weight σ_t by t) for the average strategy
                // to converge; uniform averaging makes exploitability GROW with T.
                let w = if self.linear { self.iter as f32 } else { 1.0 };
                self.cum[base + a] += w * reach * strat[a];
            }
            v
        } else {
            // sample opponent action from their current strategy
            let r = (self.rand() as f64 / u64::MAX as f64) as f32;
            let mut acc = 0.0;
            let mut a = na - 1;
            for (i, &p) in strat[..na].iter().enumerate() {
                acc += p;
                if r <= acc {
                    a = i;
                    break;
                }
            }
            let sampled = self.traverse(tree, kids[a] as usize, traverser, hands, reach);
            if self.vr {
                // VR-MCCFR control variate: value = Σ σ(j)·b(j) + (sampled − b(a*)).
                // Unbiased regardless of b; lower variance when b tracks the value.
                // The value is the TRAVERSER's, so index b by the TRAVERSER's bucket
                // bt (not the opponent's) — captures the opponent-action variance for
                // THIS traverser hand. σ is the opponent's strategy (per opp bucket).
                let bt = self.bucket_of(bs, hands[traverser]);
                if bt >= self.nb {
                    sampled // traverser hand invalid this street → no VR
                } else {
                    let base = (local * self.nb + bt) * self.max_na;
                    let bexp: f32 = (0..na).map(|j| strat[j] * self.baseline[base + j]).sum();
                    let corrected = bexp + (sampled - self.baseline[base + a]);
                    self.baseline[base + a] += self.vr_alpha * (sampled - self.baseline[base + a]);
                    corrected
                }
            } else {
                sampled
            }
        }
    }

    /// Terminal value for the traverser: fold payoff or the per-tuple bucketed
    /// showdown (the recurse_eq_buckets DP run once for the sampled tuple).
    fn terminal(&self, tree: &FlatTree, node: usize, traverser: usize, hands: &[usize]) -> f32 {
        let bs = tree.nodes[node].board_state;
        let fold_mask = tree.get_folded_mask(node);
        let np = self.np;
        // per-street death at the terminal: any ACTIVE player (traverser or a non-
        // folded opponent) whose hand is impossible at this board → impossible
        // world → 0 (catches showdowns reached without a river action). Mirrors
        // DCFR: collider tuples have 0 reach.
        for p in 0..np {
            if (fold_mask >> p) & 1 == 0 && !self.alive(hands[p], bs) {
                return 0.0;
            }
        }
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(node, p as u8)).collect();
        let c_t = contribs[traverser];
        let half_pot = tree.starting_pot as f32 / np as f32 + c_t as f32;
        let total_pot: i32 = tree.starting_pot + contribs.iter().sum::<i32>();
        let trav_folded = (fold_mask >> traverser) & 1 == 1;
        if trav_folded {
            return -half_pot; // forfeit own stake (per the oracle's zero-sum convention)
        }
        // active opponents (not folded, not traverser)
        let active_opp: Vec<usize> =
            (0..np).filter(|&p| p != traverser && (fold_mask >> p) & 1 == 0).collect();
        // POT ACCOUNTING (reconciled to design1_collapsed 2026-06-18): the win is the
        // ACTUAL pot (starting_pot + ALL contribs, incl FOLDED players' dead money)
        // minus rake minus the traverser's own investment (=half_pot). The old
        // half_pot*(active_count) only equalled this when contribs were symmetric and
        // nobody folded — at fold terminals it mis-counted the dead money.
        // Rake the MAIN pot only (design1_collapsed convention): unmatched bets are
        // returned, not raked. main_pot = levels[0]×(#contrib ≥ levels[0]) + starting_pot
        // (levels = sorted-distinct contributions). Engine previously raked total_pot.
        let mut levels: Vec<i32> = (0..np).map(|p| contribs[p]).collect();
        levels.sort();
        levels.dedup();
        let main_pot = if levels.is_empty() {
            tree.starting_pot
        } else {
            levels[0] * (0..np).filter(|&p| contribs[p] >= levels[0]).count() as i32 + tree.starting_pot
        };
        let rake = (main_pot as f32 * tree.rake_rate as f32).min(tree.rake_cap as f32).max(0.0);
        let net_pot = total_pot as f32 - rake;
        if active_opp.is_empty() {
            // everyone else folded → traverser wins the whole pot uncontested.
            return net_pot - half_pot;
        }
        // EXACT showdown (Pluribus-faithful): score the ACTUAL sampled hands on the
        // sampled board (cur_ti, cur_ri). Infoset abstraction does NOT touch payoffs —
        // terminal utility is the real chip outcome. (Replaces the bucket-average DP.)
        let st = &self.river_strength[self.cur_ti][self.cur_ri];
        let s_t = st[hands[traverser]];
        let (mut better, mut equal) = (0usize, 0usize);
        for &op in &active_opp {
            let s_o = st[hands[op]];
            if s_o > s_t { better += 1; } else if s_o == s_t { equal += 1; }
        }
        if better > 0 {
            -half_pot // a live opponent outranks the traverser → forfeit own stake
        } else {
            net_pot / (equal + 1) as f32 - half_pot // outright win or (equal+1)-way chop
        }
    }

    fn run_iter(&mut self, tree: &FlatTree, batch: usize) {
        for b in 0..batch {
            self.iter += 1;
            // Pluribus 95/5: this trajectory prunes iff pruning is on, past warmup,
            // and a 95% coin lands. The 5% full-traversal trajectories let pruned
            // actions recover (get explored + regret-updated).
            self.prune_this = self.prune
                && self.iter > self.prune_warmup
                && (self.rand() as f64 / u64::MAX as f64) < 0.95;
            let traverser = b % self.np;
            let hands = self.sample_deal();
            self.traverse(tree, 0, traverser, &hands, 1.0);
        }
    }
}

fn clone_tables(
    t: &solver_core::solver::bucketed_showdown::BucketedRunoutTables,
) -> solver_core::solver::bucketed_showdown::BucketedRunoutTables {
    solver_core::solver::bucketed_showdown::BucketedRunoutTables {
        nb: t.nb,
        f_w: t.f_w.clone(),
        f_t: t.f_t.clone(),
        f_l: t.f_l.clone(),
        f_n: t.f_n.clone(),
    }
}

// ─────────────────────────────────────────────────────────────────────
// TRUE best-response exploitability anchor (step 3). EXACT, full-tree,
// re-optimizing the deviation at every evaluee node (NOT one-step CFV reads —
// one-step is blind to chained pruning holes). Models DCFR's bucketed game BY
// CONSTRUCTION: per-hand reach with σ read via the bucket map (matching
// propagate_player_reach), the bucketed showdown at terminals. The opponents'
// counterfactual reach is computed ONCE per evaluee and SHARED between the BR
// and on-policy passes, so the gap is a consistent exploitability regardless of
// the absolute normalization — taming the normalize_opponent_reach seam.
// MULTIWAY (live≥3) only here (bucketed_showdown_cfv asserts np≥3); live-2 (the
// 6-max two-handed SUBGAME) folds in later via the exact HU showdown.
// VALIDATION (two-sided, the acceptance test): converged DCFR must read ~0
// (no over-report — the normalization-seam test) AND under-converged DCFR must
// read HIGH-and-FALLING (catches real exploitation — no under-report).
// ─────────────────────────────────────────────────────────────────────
struct Anchor {
    nb: usize,
    np: usize,
    nh: usize,
    // per-street hand→bucket maps for the 1×1 runout.
    fmap: Vec<u16>,
    tmap: Vec<u16>,
    rmap: Vec<u16>,
    ftbl: solver_core::solver::bucketed_showdown::BucketedRunoutTables,
    ttbl: solver_core::solver::bucketed_showdown::BucketedRunoutTables,
    rtbl: solver_core::solver::bucketed_showdown::BucketedRunoutTables,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    valid: Vec<usize>,       // flop-live hands (board-3 excluded) — root set
    valid_turn: Vec<usize>,  // flop-live minus turn-card colliders
    valid_river: Vec<usize>, // turn-live minus river-card colliders
    nc: f32,           // num_combinations — DCFR divides the showdown by this
    // per-node one-step regret accumulator (localization: WHERE the BR diverges
    // from on-policy on converged DCFR — the floor-contributing nodes).
    node_regret: std::cell::RefCell<Vec<f32>>,
    // per-node per-action aggregated cv (which ACTION the BR over-values).
    node_action_cv: std::cell::RefCell<Vec<Vec<f32>>>,
}

impl Anchor {
    fn new(g: &ShrunkGame) -> Self {
        let table = g.game.table();
        let nh = table.num_valid;
        let no = u16::MAX;
        // PER-STREET card removal, mirroring DCFR's reach convention. DCFR's
        // compute_reach_turn only zeros TURN-conflicts (via turn_map); a river-
        // collider (e.g. holds the river card) carries full reach into TURN
        // terminals and is zeroed only at the RIVER (river_map / chance-prob). So
        // the anchor must be per-street too: a hand is live at a street iff it
        // doesn't collide with the board DEALT SO FAR. All-street `valid` excluded
        // river-colliders at turn terminals where DCFR includes them — the cap-
        // independent residual. Card-check (not just map!=NO_BUCKET) so it also
        // holds under identity bucketing, which never emits NO_BUCKET.
        let turn_card = table.remaining_deck[0] as usize;
        let river_card = table.river_decks[turn_card][0] as usize;
        let hcards = |h: usize| -> u64 {
            (1u64 << table.hand_cards[h * 2]) | (1u64 << table.hand_cards[h * 2 + 1])
        };
        let tmask = 1u64 << turn_card;
        let rmask = 1u64 << river_card;
        // flop-valid: every dealt hand (board-3 already excluded at build time).
        let valid: Vec<usize> = (0..nh).filter(|&h| g.bk.flop_map[h] != no).collect();
        // turn-valid: flop-valid, not colliding the turn card.
        let valid_turn: Vec<usize> = valid.iter().copied()
            .filter(|&h| hcards(h) & tmask == 0 && g.bk.turn_map[0][h] != no)
            .collect();
        // river-valid: turn-valid, not colliding the river card.
        let valid_river: Vec<usize> = valid_turn.iter().copied()
            .filter(|&h| hcards(h) & rmask == 0 && g.bk.river_map[0][0][h] != no)
            .collect();
        Anchor {
            nb: g.bk.nb_flop.max(g.bk.nb_turn).max(g.bk.nb_river),
            np: g.live as usize,
            nh,
            fmap: g.bk.flop_map.clone(),
            tmap: g.bk.turn_map[0].clone(),
            rmap: g.bk.river_map[0][0].clone(),
            ftbl: clone_tables(&g.bk.flop_tables),
            ttbl: clone_tables(&g.bk.turn_tables[0]),
            rtbl: clone_tables(&g.bk.river_tables[0][0]),
            starting_pot: g.tree.starting_pot,
            rake_rate: g.tree.rake_rate as f32,
            rake_cap: g.tree.rake_cap as f32,
            valid,
            valid_turn,
            valid_river,
            nc: table.num_combinations as f32,
            node_regret: std::cell::RefCell::new(vec![0.0; g.tree.num_nodes()]),
            node_action_cv: std::cell::RefCell::new(vec![Vec::new(); g.tree.num_nodes()]),
        }
    }

    /// hands live at a street = not colliding the board dealt so far (mirrors
    /// DCFR's per-street reach: river-colliders are live through the turn).
    #[inline]
    fn valid_at(&self, bs: u8) -> &[usize] {
        use solver_core::tree::action::BoardState;
        if bs == BoardState::Flop as u8 { &self.valid }
        else if bs == BoardState::Turn as u8 { &self.valid_turn }
        else { &self.valid_river }
    }

    #[inline]
    fn street_map(&self, bs: u8) -> &[u16] {
        use solver_core::tree::action::BoardState;
        if bs == BoardState::Flop as u8 { &self.fmap }
        else if bs == BoardState::Turn as u8 { &self.tmap }
        else { &self.rmap }
    }
    #[inline]
    fn street_tbl(&self, bs: u8) -> &solver_core::solver::bucketed_showdown::BucketedRunoutTables {
        use solver_core::tree::action::BoardState;
        if bs == BoardState::Flop as u8 { &self.ftbl }
        else if bs == BoardState::Turn as u8 { &self.ttbl }
        else { &self.rtbl }
    }

    /// total exploitability = Σ_i (BR_i − onpolicy_i), σ in canonical
    /// node-major layout [node*MAX_NA*nb + a*nb + b] (MAX_NA = max_na arg).
    fn exploitability(&self, tree: &FlatTree, sigma: &[f32], max_na: usize) -> f32 {
        let mut total = 0.0f32;
        for i in 0..self.np {
            // opponents' counterfactual reach at the root: sum-1 per opponent
            // over valid hands (i's slot carried but unused for the gap).
            let w = 1.0 / self.valid.len() as f32;
            let mut reach = vec![vec![0.0f32; self.nh]; self.np];
            for p in 0..self.np {
                for &h in &self.valid {
                    reach[p][h] = w;
                }
            }
            let v_br = self.walk(tree, 0, i as u8, true, sigma, max_na, &reach);
            let v_on = self.walk(tree, 0, i as u8, false, sigma, max_na, &reach);
            // V_i = mean over valid i-hands (uniform prior).
            let mut gap = 0.0f32;
            for &h in &self.valid {
                gap += (v_br[h] - v_on[h]) * w;
            }
            if std::env::var("MC_VERBOSE").is_ok() {
                eprintln!("    evaluee {i}: BR-gap {:.4e}", gap);
            }
            total += gap.max(0.0);
        }
        total
    }

    fn walk(&self, tree: &FlatTree, node: usize, i: u8, br: bool, sigma: &[f32], max_na: usize, reach: &[Vec<f32>]) -> Vec<f32> {
        let n = &tree.nodes[node];
        if n.is_terminal() {
            return self.terminal(tree, node, i, reach);
        }
        let kids: Vec<u32> = tree.node_children(node).to_vec();
        if n.is_chance() {
            // 1×1: single child; remove hands conflicting the dealt card. The
            // hand identity persists (no bucket-transition matrix). Card-removal
            // is captured by the next street's map (NO_BUCKET handled at showdown
            // + here we simply carry reach forward — the map zeroes conflicts).
            return self.walk(tree, kids[0] as usize, i, br, sigma, max_na, reach);
        }
        let player = n.player_id;
        let na = kids.len();
        let bs = n.board_state;
        let map = self.street_map(bs);
        let vh = self.valid_at(bs); // per-street live hands
        let off = node * max_na * self.nb;
        if player == i {
            let cvs: Vec<Vec<f32>> =
                kids.iter().map(|&c| self.walk(tree, c as usize, i, br, sigma, max_na, reach)).collect();
            let mut v = vec![0.0f32; self.nh];
            if br {
                // per-bucket argmax (uniform prior over hands in the bucket).
                let mut best_a = vec![0usize; self.nb];
                let mut best_v = vec![f32::NEG_INFINITY; self.nb];
                let mut sum = vec![vec![0.0f32; self.nb]; na];
                for &h in vh {
                    let b = map[h] as usize;
                    if b >= self.nb { continue; }
                    for a in 0..na { sum[a][b] += cvs[a][h]; }
                }
                for b in 0..self.nb {
                    for a in 0..na {
                        if sum[a][b] > best_v[b] { best_v[b] = sum[a][b]; best_a[b] = a; }
                    }
                }
                for &h in vh {
                    let b = map[h] as usize;
                    if b >= self.nb { continue; }
                    v[h] = cvs[best_a[b]][h];
                }
            } else {
                // per-bucket aggregates for v + the one-step regret localization.
                let mut sum_ab = vec![vec![0.0f32; self.nb]; na];
                for &h in vh {
                    let b = map[h] as usize;
                    if b >= self.nb { continue; }
                    for a in 0..na { sum_ab[a][b] += cvs[a][h]; }
                }
                let mut node_reg = 0.0f32;
                for b in 0..self.nb {
                    let onp: f32 = (0..na).map(|a| sigma[off + a * self.nb + b] * sum_ab[a][b]).sum();
                    let mx = (0..na).map(|a| sum_ab[a][b]).fold(f32::NEG_INFINITY, f32::max);
                    node_reg += (mx - onp).max(0.0);
                }
                self.node_regret.borrow_mut()[node] += node_reg;
                // aggregated per-action cv (sum over buckets+valid hands) — only
                // the evaluee-0 pass is inspected, overwrite is fine.
                let agg: Vec<f32> = (0..na).map(|a| (0..self.nb).map(|b| sum_ab[a][b]).sum()).collect();
                self.node_action_cv.borrow_mut()[node] = agg;
                for &h in vh {
                    let b = map[h] as usize;
                    if b >= self.nb { continue; }
                    let mut acc = 0.0;
                    for a in 0..na { acc += sigma[off + a * self.nb + b] * cvs[a][h]; }
                    v[h] = acc;
                }
            }
            v
        } else {
            // opponent: split this opponent's reach by σ into each child, recurse,
            // SUM the evaluee values (factored convention — reach carries σ).
            let p = player as usize;
            let mut v = vec![0.0f32; self.nh];
            for a in 0..na {
                let mut r2 = reach.to_vec();
                for &h in vh {
                    let b = map[h] as usize;
                    if b >= self.nb { r2[p][h] = 0.0; }
                    else { r2[p][h] *= sigma[off + a * self.nb + b]; }
                }
                let cv = self.walk(tree, kids[a] as usize, i, br, sigma, max_na, &r2);
                for &h in vh { v[h] += cv[h]; }
            }
            v
        }
    }

    /// evaluee value per i-hand at a terminal via the bucketed showdown.
    fn terminal(&self, tree: &FlatTree, node: usize, i: u8, reach: &[Vec<f32>]) -> Vec<f32> {
        use solver_core::solver::bucketed_showdown::bucketed_showdown_cfv_design1_collapsed;
        let bs = tree.nodes[node].board_state;
        let map = self.street_map(bs);
        let tbl = self.street_tbl(bs);
        let vh = self.valid_at(bs); // per-street live hands
        let fold_mask = tree.get_folded_mask(node);
        let np = self.np;
        // opponents' bucket reach (all p≠i), reduced from hand reach.
        let mut bucket_reach: Vec<Vec<f32>> = vec![vec![0.0f32; self.nb]; np - 1];
        let mut oi = 0;
        for p in 0..np {
            if p == i as usize { continue; }
            for &h in vh {
                let b = map[h] as usize;
                if b < self.nb { bucket_reach[oi][b] += reach[p][h]; }
            }
            oi += 1;
        }
        let views: Vec<&[f32]> = bucket_reach.iter().map(|v| v.as_slice()).collect();
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(node, p as u8)).collect();
        let cfv = bucketed_showdown_cfv_design1_collapsed(
            &views, tbl, &contribs, fold_mask, i as usize, np as u8,
            self.starting_pot, self.rake_rate, self.rake_cap, true,
        );
        // v[i-hand] = cfv at i-hand's bucket on this street, in DCFR's units
        // (DCFR divides the showdown by num_combinations — match it so the
        // exploitability magnitude is interpretable and side-1 reads true ~0).
        let inv_nc = if self.nc > 0.0 { 1.0 / self.nc } else { 1.0 };
        let mut v = vec![0.0f32; self.nh];
        for &h in vh {
            let b = map[h] as usize;
            if b < self.nb { v[h] = cfv[b] * inv_nc; }
        }
        v
    }
}

/// DCFR per-iter timing. live==2 is exact HU (FlopStartVectorCfr, O(nh^2)
/// showdown — the bucketed designs force HU to exact); live>=3 is the bucketed
/// multiway wall (BucketedFlopCfr Design1Collapsed, O(nb^num_opp) showdown).
fn dcfr_per_iter(g: &ShrunkGame, iters: u32) -> (f64, usize) {
    use solver_core::solver::flop_start_vector_cfr::FlopStartVectorCfr;
    let nn = g.tree.num_nodes();
    if g.live == 2 {
        let mut s = FlopStartVectorCfr::new(&g.tree, g.game.table());
        s.run(&g.tree, &g.game, 1);
        let t0 = Instant::now();
        s.run(&g.tree, &g.game, iters);
        return (t0.elapsed().as_secs_f64() / iters as f64, nn);
    }
    let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
    // MC_DCFR_FACTORED=1 ⇒ baseline DCFR uses the precomputed-equity terminal
    // (Design2Factored, np≥4) — the FAIR comparison now that DCFR no longer needs
    // the B^K Design1Collapsed wall MCCFR was built to dodge.
    let term = match (std::env::var("MC_DCFR_FACTORED").is_ok(), g.live >= 4) {
        (true, true) => TerminalDesign::Design2Factored,
        _ => TerminalDesign::Design1Collapsed,
    };
    s.set_terminal_design(term);
    s.run(&g.tree, &g.game, &g.bk, 1);
    let t0 = Instant::now();
    s.run(&g.tree, &g.game, &g.bk, iters);
    (t0.elapsed().as_secs_f64() / iters as f64, nn)
}

/// Two-sided anchor validation: DCFR avg at increasing iters. The anchor passes
/// iff exploitability is HIGH at low iters and FALLS toward ~0 as DCFR converges
/// (converged ~0 = no over-report / normalization-seam OK; high-and-falling =
/// catches real exploitation / no under-report). One-sided would pass a liar.
fn anchor_validation(nb: usize, nt: usize, nr: usize) {
    use solver_core::tree::flat::MAX_NA_POSTFLOP;
    let live: u8 = std::env::var("MC_LIVE").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let g = build_shrunk(live, nb, nt, nr);
    let anchor = Anchor::new(&g);
    // MC_TERMCMP: per-tuple terminal value, ENGINE vs design1_collapsed, at IDENTITY
    // (singletons → binary fractions → matched conditioning, so a divergence is a
    // REAL value bug not a representation artifact). Aim: pot/dead-money accounting
    // (engine half_pot*(np-1) win vs design1_collapsed's pot handling).
    if std::env::var("MC_TERMCMP").is_ok() {
        use solver_core::solver::bucketed_showdown::bucketed_showdown_cfv_design1_collapsed;
        use solver_core::tree::action::BoardState;
        let m = Mccfr::new(&g);
        let np = g.live as usize;
        // a full-showdown river terminal (all active, no folds)
        // np distinct river-valid hands (no shared cards)
        let mut hands: Vec<usize> = Vec::new();
        let mut used = 0u64;
        for &h in &anchor.valid_river {
            let (c1, c2) = m.hand_cards[h];
            let mask = (1u64 << c1) | (1u64 << c2);
            if used & mask == 0 { used |= mask; hands.push(h); if hands.len() == np { break; } }
        }
        let inv_nc = if anchor.nc > 0.0 { 1.0 / anchor.nc } else { 1.0 };
        println!("TERMINAL CMP (identity): np={np} starting_pot={} hands={hands:?} — DIVERGENT terminals only:", anchor.starting_pot);
        let mut shown = 0;
        let mut checked = 0;
        for node in 0..g.tree.num_nodes() {
            let n = &g.tree.nodes[node];
            if !n.is_terminal() { continue; }
            // only RIVER terminals (river tables) where the active players hold our hands
            if n.board_state != BoardState::River as u8 { continue; }
            checked += 1;
            let fold_mask = g.tree.get_folded_mask(node);
            let contribs: Vec<i32> = (0..np).map(|p| g.tree.get_contribution(node, p as u8)).collect();
            for trav in 0..np {
                if (fold_mask >> trav) & 1 == 1 { continue; } // traverser folded → engine returns -half_pot, skip (no showdown)
                let eng = m.terminal(&g.tree, node, trav, &hands);
                let mut reach = vec![vec![0.0f32; anchor.nb]; np - 1];
                let mut oi = 0;
                for p in 0..np {
                    if p == trav { continue; }
                    reach[oi][anchor.rmap[hands[p]] as usize] = 1.0;
                    oi += 1;
                }
                let views: Vec<&[f32]> = reach.iter().map(|v| v.as_slice()).collect();
                let cfv = bucketed_showdown_cfv_design1_collapsed(
                    &views, &anchor.rtbl, &contribs, fold_mask, trav, np as u8,
                    anchor.starting_pot, anchor.rake_rate, anchor.rake_cap, true,
                );
                let bt = anchor.rmap[hands[trav]] as usize;
                let d1 = cfv[bt] * inv_nc;
                if (eng - d1).abs() > 0.01 && shown < 14 {
                    println!("  node {node:>4} fold_mask {fold_mask:#05b} contribs {contribs:?} trav {trav}: ENGINE {eng:>9.3}  design1 {d1:>9.3}  diff {:>9.3}", eng - d1);
                    shown += 1;
                }
            }
        }
        println!("  ({shown} divergent of {checked} river terminals checked × {np} travs)");
        return;
    }
    // MC_MCCONV: convergence of the MCCFR AVERAGE (per-street engine) measured by
    // the trusted anchor. Two-way gate: identity → ~0 (per-street implemented
    // right), nb=6 → DCFR's per-family residual (same game). Run BOTH before any
    // speed comparison (same-residual proves same game; speed is the verdict).
    if std::env::var("MC_MCCONV").is_ok() {
        let nn = g.tree.num_nodes();
        let nb_a = anchor.nb;
        let mut m = Mccfr::new(&g);
        println!("MCCFR CONVERGENCE — true-BR exploit of the MCCFR average, live-{live}, nb={nb}");
        println!("{:>10} {:>18}", "traj", "true-BR exploit");
        let batch = 8192usize;
        let mut total = 0usize;
        let mut run_secs = 0.0f64; // time spent in run_iter only (NOT the anchor evals)
        for &target in &[8192usize, 65536, 262144, 1048576, 4194304] {
            let t_run = Instant::now();
            while total < target { m.run_iter(&g.tree, batch); total += batch; }
            run_secs += t_run.elapsed().as_secs_f64();
            let mut sigma = vec![0.0f32; nn * MAX_NA_POSTFLOP * nb_a];
            for node in 0..nn {
                if !g.tree.nodes[node].is_player() { continue; }
                let local = m.node_local[node] as usize;
                let na = g.tree.nodes[node].num_children as usize;
                let off = node * MAX_NA_POSTFLOP * nb_a;
                let use_current = std::env::var("MC_CURRENT").is_ok();
                for bb in 0..nb_a {
                    let base = (local * m.nb + bb) * m.max_na;
                    let val = |a: usize| -> f32 {
                        if use_current { m.regret[base + a].max(0.0) } else { m.cum[base + a] }
                    };
                    let sum: f32 = (0..na).map(val).sum();
                    if sum > 0.0 {
                        for a in 0..na { sigma[off + a * nb_a + bb] = val(a) / sum; }
                    } else {
                        let u = 1.0 / na as f32;
                        for a in 0..na { sigma[off + a * nb_a + bb] = u; }
                    }
                }
            }
            let expl = anchor.exploitability(&g.tree, &sigma, MAX_NA_POSTFLOP);
            // SELF-CONSISTENT convergence signal (the discriminator): the engine's
            // own CFR+ regret bound. maxR/T → 0 ⇒ engine converges to SOME Nash by
            // its OWN values (so a high anchor reading = terminal/value mismatch).
            // maxR/T FLAT ⇒ regrets grow linearly ⇒ dynamics don't converge to any
            // equilibrium (terminal reconciliation would be premature).
            let maxr = m.regret.iter().cloned().fold(0.0f32, f32::max);
            // pruning diagnostics: fraction of traverser-action recursions skipped
            // (the per-trajectory compute saved). 0 when MC_PRUNE is off.
            let total_actions = m.pruned_nodes + m.visited_nodes;
            let prune_frac = if total_actions > 0 { m.pruned_nodes as f64 / total_actions as f64 } else { 0.0 };
            let us_traj = run_secs / total as f64 * 1e6; // decoupled MCCFR µs/traj
            println!("{total:>10} {expl:>18.5e}   maxR/T {:>10.4e}  prune_frac {:>7.4}  µs/traj {:>7.3}",
                maxr as f64 / total as f64, prune_frac, us_traj);
        }
        return;
    }
    println!("ANCHOR VALIDATION — true-BR exploitability of the DCFR average, live-{live}, nb={nb}");
    println!("two-sided test: under-converged → HIGH & FALLING (catches exploitation, no under-report);");
    println!("converged → ~0 (no over-report — the normalize_opponent_reach-seam test). pass = both.\n");
    println!("{:>6} {:>18}", "DCFR iters", "true-BR exploit");
    // MC_ROOTCMP: is the anchor's value model = DCFR's? Compare anchor on-policy
    // root value V_i (per hand) vs DCFR's root_cfv_from_avg. Constant ratio ⇒
    // value model matches (floor is not a value-model bug); varying ratio ⇒
    // value-model divergence (the bug).
    if std::env::var("MC_ROOTCMP").is_ok() {
        use solver_core::tree::flat::MAX_NA_POSTFLOP;
        let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        s.run(&g.tree, &g.game, &g.bk, 1024);
        let sigma = s.average_strategy_canonical(&g.tree, &g.bk);
        let dcfr_root = s.root_cfv_from_avg(&g.tree, &g.game, &g.bk); // [np][nh]
        let nv = anchor.valid.len();
        let w = 1.0 / nv as f32;
        let mut reach = vec![vec![0.0f32; anchor.nh]; anchor.np];
        for p in 0..anchor.np { for &h in &anchor.valid { reach[p][h] = w; } }
        let v_on = anchor.walk(&g.tree, 0, 0, false, &sigma, MAX_NA_POSTFLOP, &reach);
        println!("ROOT VALUE CMP (evaluee 0): anchor on-policy vs DCFR root_cfv_from_avg");
        // card-removal diagnostic: print each hand's cards + the board/runout so
        // the sign-flipped vs exact contrast can be read (does the wrong hand
        // share a card with board/runout?).
        let cs = |c: u8| -> String {
            let r = "23456789TJQKA".as_bytes()[(c >> 2) as usize] as char;
            let s = "cdhs".as_bytes()[(c & 3) as usize] as char;
            format!("{r}{s}")
        };
        let tbl_hc = &g.game.table().hand_cards;
        let turn = g.game.table().remaining_deck.clone();
        let tc = turn[0];
        let rc = g.game.table().river_decks[tc as usize][0];
        println!("board: turn {} river {}  (flop = canonical[0])", cs(tc), cs(rc));
        println!("{:>6} {:>7} {:>14} {:>14} {:>10}", "hand", "cards", "anchor", "DCFR", "ratio");
        let mut shown = 0;
        for &h in anchor.valid.iter() {
            let (a, d) = (v_on[h], dcfr_root[0][h]);
            if d.abs() > 1e-6 && shown < 16 {
                let cards = format!("{}{}", cs(tbl_hc[h * 2]), cs(tbl_hc[h * 2 + 1]));
                let flag = if (a / d) < 0.0 { " <-SIGN-FLIP" } else if (a - d).abs() < 1e-3 { " <-exact" } else { "" };
                println!("{h:>6} {cards:>7} {a:>14.5e} {d:>14.5e} {:>10.4}{flag}", a / d);
                shown += 1;
            }
        }
        return;
    }
    // MC_NODECMP: per-ACTION cv at top fold nodes, anchor vs DCFR's internal
    // bottom_up cfv (turn_cfv[child]). Pins WHICH action's cv carries the
    // non-cancelling (floor) error — fold vs call ratio is the tell.
    if std::env::var("MC_NODECMP").is_ok() {
        let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        s.run(&g.tree, &g.game, &g.bk, 1024);
        let sigma = s.average_strategy_canonical(&g.tree, &g.bk);
        let _ = anchor.exploitability(&g.tree, &sigma, MAX_NA_POSTFLOP); // populates node_action_cv
        let tcfv = s.debug_turn_cfv(&g.tree, &g.game, &g.bk, 0); // DCFR per-node cfv, evaluee 0
        let acv = anchor.node_action_cv.borrow();
        let nh = anchor.nh;
        println!("PER-ACTION cv: anchor vs DCFR internal (evaluee 0). fold/call ratio = the tell.");
        for &node in &[468usize, 506, 369, 329, 230] {
            let n = &g.tree.nodes[node];
            if n.player_id != 0 { continue; } // DCFR cfv is for traverser 0
            let kids = g.tree.node_children(node);
            println!(" node {node} street {} p{}:", n.board_state, n.player_id);
            for (a, &c) in kids.iter().enumerate() {
                let dcfr: f32 = anchor.valid.iter().map(|&h| tcfv[c as usize * nh + h]).sum();
                let anc = acv[node].get(a).copied().unwrap_or(0.0);
                let lbl = g.tree.nodes[c as usize].action_label;
                let term = g.tree.nodes[c as usize].is_terminal();
                println!("   a{a} (label {lbl}{}) anchor {anc:>11.3e}  DCFR {dcfr:>11.3e}  ratio {:>8.4}",
                    if term { ",T" } else { "" }, anc / dcfr);
            }
        }
        return;
    }
    let iter_list: Vec<u32> = if let Ok(s) = std::env::var("MC_ITERS") {
        s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
    } else if std::env::var("MC_FAST").is_ok() { vec![256] } else { vec![1, 64, 1024] };
    for iters in iter_list {
        let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        s.run(&g.tree, &g.game, &g.bk, iters);
        let sigma = s.average_strategy_canonical(&g.tree, &g.bk);
        let expl = anchor.exploitability(&g.tree, &sigma, MAX_NA_POSTFLOP);
        println!("{iters:>6} {expl:>18.5e}");
    }
    if std::env::var("MC_NOLOC").is_ok() { return; }
    // ── FLOOR LOCALIZATION: where does the BR diverge on converged DCFR? ──
    let mut s = BucketedFlopCfr::new(&g.tree, g.game.table(), &g.bk);
    s.set_terminal_design(TerminalDesign::Design1Collapsed);
    s.run(&g.tree, &g.game, &g.bk, 2048);
    let sigma = s.average_strategy_canonical(&g.tree, &g.bk);
    anchor.node_regret.borrow_mut().iter_mut().for_each(|x| *x = 0.0);
    let _ = anchor.exploitability(&g.tree, &sigma, MAX_NA_POSTFLOP);
    let nr = anchor.node_regret.borrow().clone();
    let mut idx: Vec<usize> = (0..nr.len()).filter(|&i| nr[i] > 1e-3).collect();
    idx.sort_by(|&a, &b| nr[b].partial_cmp(&nr[a]).unwrap());
    println!("\nFLOOR LOCALIZATION (converged DCFR) — top one-step-regret nodes:");
    println!("(if concentrated at a node TYPE — street/terminal-adjacent/na — that's the bug locus)");
    let acv = anchor.node_action_cv.borrow();
    for &node in idx.iter().take(10) {
        let n = &g.tree.nodes[node];
        let labels: Vec<u8> = g.tree.node_children(node).iter().map(|&c| g.tree.nodes[c as usize].action_label).collect();
        // per-action cv + which child is a terminal (fold→forfeit) with its contribs.
        let kids = g.tree.node_children(node);
        let kid_info: Vec<String> = kids.iter().map(|&c| {
            let cn = &g.tree.nodes[c as usize];
            if cn.is_terminal() {
                format!("T(c_p{}={})", n.player_id, g.tree.get_contribution(c as usize, n.player_id))
            } else { "·".into() }
        }).collect();
        let cv: Vec<String> = acv[node].iter().map(|v| format!("{v:+.2e}")).collect();
        println!("  node {node:>6} street {} p{} labels {labels:?}: cv [{}] kids [{}]",
            n.board_state, n.player_id, cv.join(", "), kid_info.join(", "));
    }
    let nonzero: usize = nr.iter().filter(|&&r| r > 1e-3).count();
    println!("  ({nonzero} nodes with regret > 1e-3; total = {:.4e})", nr.iter().sum::<f32>());
}

// ─────────────────────────────────────────────────────────────────────
// CONNECTED MCCFR (the probe the branch was NAMED for, finally run).
// External-sampling MCCFR over the WHOLE game: preflop betting → flop deal →
// postflop, ONE trajectory updating regret at BOTH the preflop (169-class)
// infosets AND the postflop (bucket) infosets. This structurally AVOIDS the
// DCFR co-solve's N×fill wall: the postflop subgame co-adapts with preflop along
// sampled trajectories — there is no "re-solve every subgame to convergence each
// preflop iteration". Pluribus pruning runs on BOTH layers.
//
// MILESTONE 1 (this): HU, a single fixed canonical flop, call/fold-only preflop
// ⇒ ONE seam cell (both call → flop pot=4, commit=2; or SB folds → blinds). Real
// cap-3 postflop. Gated by preflop-strength MONOTONICITY + postflop plateau
// (NOT a full connected BR yet). Known reduction: single fixed runout (nt=nr=1)
// makes turn/river-colliding holes die postflop — a clairvoyance-flavored artifact
// shared with the subgame probe; used directionally, not as exact equity.
// ─────────────────────────────────────────────────────────────────────
struct ConnectedHu {
    pre_tree: FlatTree,
    pre_local: Vec<i32>, // [nn] preflop player-node → infoset idx, else -1
    pre_max_na: usize,
    pre_regret: Vec<f32>, // [pre_ninfo * NUM_PREFLOP_CLASSES * pre_max_na]
    pre_cum: Vec<f32>,
    post: Mccfr,           // postflop engine (one cell, fixed flop) — co-adapting
    post_tree: FlatTree,
    card_to_hand: Vec<i32>, // [52*52] (c1,c2)→postflop hand idx, -1 invalid
    deck: Vec<u8>,          // 49 non-flop cards (the preflop deal pool)
    rng: u64,
    prune: bool,
    prune_c: f32,
    prune_warmup: u64,
    iter: u64,
    prune_this: bool,
}

impl ConnectedHu {
    fn new(nb: usize) -> Self {
        // HU postflop cell entered by both calling: commit=2 (1bb each), pot=4.
        // Runout sampled (MC_NT turns × MC_NR rivers) to de-clairvoyant the showdown.
        let nt: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        let nr: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        let g = build_shrunk_cell(2, 2, 4, nb, nt, nr);
        let post_tree = g.tree.clone();
        let post = Mccfr::new(&g);
        // (c1,c2) → postflop hand index, both orders.
        let mut card_to_hand = vec![-1i32; 52 * 52];
        for (h, &(a, b)) in post.hand_cards.iter().enumerate() {
            card_to_hand[a as usize * 52 + b as usize] = h as i32;
            card_to_hand[b as usize * 52 + a as usize] = h as i32;
        }
        // The fixed flop = the 3 cards excluded from every valid hand. Recover it
        // as the cards never appearing in any hand's 2-card combo.
        let mut on_board = [true; 52];
        for &(a, b) in &post.hand_cards {
            on_board[a as usize] = false;
            on_board[b as usize] = false;
        }
        // flop ∪ turn ∪ river are "never in a flop-live hand"; the preflop deal
        // pool excludes the FLOP cards only (turn/river collide per-street in post).
        // Conservative: exclude everything not appearing — the deal pool is the
        // cards that DO appear in some hand (flop-compatible). That's exactly the
        // hands' card support.
        let deck: Vec<u8> = (0..52u8).filter(|&c| !on_board[c as usize]).collect();

        // preflop HU tree: fold/call only (no raises) ⇒ single seam cell.
        let mut spec = production_game_v1();
        spec.num_players = 2; // HU
        let bets = BetSizeOptions { bet: vec![], raise: vec![] };
        let cfg = spec.preflop_tree_config(bets);
        let pre_tree = build_tree_preflop_only(&cfg).expect("preflop HU tree");
        let nn = pre_tree.num_nodes();
        let mut pre_local = vec![-1i32; nn];
        let mut pre_ninfo = 0usize;
        let mut pre_max_na = 0usize;
        for i in 0..nn {
            if pre_tree.nodes[i].is_player() {
                pre_local[i] = pre_ninfo as i32;
                pre_ninfo += 1;
                pre_max_na = pre_max_na.max(pre_tree.nodes[i].num_children as usize);
            }
        }
        let stride = NUM_PREFLOP_CLASSES * pre_max_na.max(1);
        ConnectedHu {
            pre_tree,
            pre_local,
            pre_max_na: pre_max_na.max(1),
            pre_regret: vec![0.0; pre_ninfo * stride],
            pre_cum: vec![0.0; pre_ninfo * stride],
            post,
            post_tree,
            card_to_hand,
            deck,
            rng: 0xD1B54A32D192ED03,
            prune: std::env::var("MC_PRUNE").is_ok(),
            prune_c: std::env::var("MC_PRUNE_C").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            prune_warmup: std::env::var("MC_PRUNE_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(0),
            iter: 0,
            prune_this: false,
        }
    }

    #[inline]
    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    /// preflop regret-match into out[..na] for (infoset local, class).
    fn pre_strategy(&self, local: usize, class: usize, na: usize, out: &mut [f32]) {
        let base = (local * NUM_PREFLOP_CLASSES + class) * self.pre_max_na;
        let mut sum = 0.0f32;
        for a in 0..na {
            let r = self.pre_regret[base + a].max(0.0);
            out[a] = r;
            sum += r;
        }
        if sum > 0.0 {
            for a in 0..na {
                out[a] /= sum;
            }
        } else {
            let u = 1.0 / na as f32;
            for a in 0..na {
                out[a] = u;
            }
        }
    }

    /// Preflop fold terminal value for the traverser (HU uncontested; no rake — the
    /// preflop has no flop-seen drop). Mirrors the postflop half_pot convention.
    fn pre_terminal(&self, node: usize, traverser: usize) -> f32 {
        let np = 2usize;
        let fold_mask = self.pre_tree.get_folded_mask(node);
        let contribs: Vec<i32> =
            (0..np).map(|p| self.pre_tree.get_contribution(node, p as u8)).collect();
        let half_pot = self.pre_tree.starting_pot as f32 / np as f32 + contribs[traverser] as f32;
        if (fold_mask >> traverser) & 1 == 1 {
            return -half_pot; // folded: forfeit own stake
        }
        let total: i32 = self.pre_tree.starting_pot + contribs.iter().sum::<i32>();
        total as f32 - half_pot // uncontested win (opponent folded)
    }

    /// Connected external-sampling traverse over the preflop tree; at the flop-entry
    /// chance leaf it recurses into the (co-adapting) postflop engine.
    /// `class[p]` = preflop class index, `hand[p]` = postflop hand index.
    fn traverse_pre(&mut self, node: usize, traverser: usize, class: &[usize], hand: &[usize]) -> f32 {
        let n = &self.pre_tree.nodes[node];
        if n.is_terminal() {
            return self.pre_terminal(node, traverser);
        }
        if n.is_chance() {
            // FLOP-ENTRY SEAM: recurse into the postflop subgame with this trajectory's
            // concrete hands. Share the prune decision across the seam.
            self.post.prune_this = self.prune_this;
            return self.post.traverse(&self.post_tree, 0, traverser, hand, 1.0);
        }
        let kids: Vec<u32> = self.pre_tree.node_children(node).to_vec();
        let na = kids.len();
        let player = n.player_id as usize;
        let local = self.pre_local[node] as usize;
        let cls = class[player];
        let mut strat = [0.0f32; 16];
        self.pre_strategy(local, cls, na, &mut strat[..na]);
        if player == traverser {
            let base = (local * NUM_PREFLOP_CLASSES + cls) * self.pre_max_na;
            let mut cv = [0.0f32; 16];
            let mut v = 0.0f32;
            let mut pruned = [false; 16];
            for a in 0..na {
                if self.prune_this && self.pre_regret[base + a] <= self.prune_c {
                    pruned[a] = true;
                    continue;
                }
                cv[a] = self.traverse_pre(kids[a] as usize, traverser, class, hand);
                v += strat[a] * cv[a];
            }
            for a in 0..na {
                if pruned[a] {
                    continue;
                }
                self.pre_regret[base + a] = (self.pre_regret[base + a] + cv[a] - v).max(0.0);
                self.pre_cum[base + a] += strat[a];
            }
            v
        } else {
            // sample opponent action
            let r = (self.rand() as f64 / u64::MAX as f64) as f32;
            let mut acc = 0.0;
            let mut a = na - 1;
            for (i, &p) in strat[..na].iter().enumerate() {
                acc += p;
                if r <= acc {
                    a = i;
                    break;
                }
            }
            self.traverse_pre(kids[a] as usize, traverser, class, hand)
        }
    }

    /// Sample HU hole cards (4 distinct cards from the flop-compatible deck), return
    /// per-player (preflop class, postflop hand index).
    fn deal(&mut self) -> ([usize; 2], [usize; 2]) {
        let nd = self.deck.len();
        let mut used = [255u8; 4];
        let mut cards = [0u8; 4];
        let mut k = 0;
        while k < 4 {
            let idx = (self.rand() as usize) % nd;
            let c = self.deck[idx];
            if used[..k].iter().any(|&u| u == c) {
                continue;
            }
            used[k] = c;
            cards[k] = c;
            k += 1;
        }
        let mut class = [0usize; 2];
        let mut hand = [0usize; 2];
        for p in 0..2 {
            let (c1, c2) = (cards[2 * p], cards[2 * p + 1]);
            class[p] = PreflopClass::from_combo(c1 as Card, c2 as Card).index();
            hand[p] = self.card_to_hand[c1 as usize * 52 + c2 as usize] as usize;
        }
        (class, hand)
    }

    fn run_iter(&mut self, batch: usize) {
        for b in 0..batch {
            self.iter += 1;
            self.prune_this = self.prune
                && self.iter > self.prune_warmup
                && (self.rand() as f64 / u64::MAX as f64) < 0.95;
            let traverser = b % 2;
            self.post.sample_runout(); // sample this trajectory's turn+river
            let (class, hand) = self.deal();
            self.traverse_pre(0, traverser, &class, &hand);
        }
    }

    /// The preflop fold/call DECISION node, detected structurally: the player node
    /// that has a terminal child in which the acting player is folded. Returns
    /// (node_idx, fold_action_idx).
    fn fold_decision_node(&self) -> Option<(usize, usize)> {
        for i in 0..self.pre_tree.num_nodes() {
            let n = &self.pre_tree.nodes[i];
            if !n.is_player() {
                continue;
            }
            let actor = n.player_id;
            let kids: Vec<u32> = self.pre_tree.node_children(i).to_vec();
            for (a, &k) in kids.iter().enumerate() {
                let kn = &self.pre_tree.nodes[k as usize];
                if kn.is_terminal() && (self.pre_tree.get_folded_mask(k as usize) >> actor) & 1 == 1 {
                    return Some((i, a));
                }
            }
        }
        None
    }

    /// preflop CALL (= non-fold) frequency for a given class at the fold/call node.
    fn pre_call_freq(&self, class: usize) -> Option<f32> {
        let (node, fold_a) = self.fold_decision_node()?;
        let local = self.pre_local[node];
        if local < 0 {
            return None;
        }
        let na = self.pre_tree.nodes[node].num_children as usize;
        let base = (local as usize * NUM_PREFLOP_CLASSES + class) * self.pre_max_na;
        let sum: f32 = (0..na).map(|a| self.pre_cum[base + a]).sum();
        if sum <= 0.0 {
            return None;
        }
        Some(1.0 - self.pre_cum[base + fold_a] / sum)
    }

    /// CONNECTED regret bound: max cumulative (CFR+ floored) regret / T over BOTH
    /// the preflop AND postflop infosets, T = total trajectories. This is the
    /// self-consistent convergence proof — CFR's regret bound → 0 at ~1/√T iff the
    /// connected co-solve is reaching a Nash of the connected game by its own values
    /// (the same check that validated the isolated engine). Returns (pre, post).
    fn regret_bound(&self) -> (f32, f32) {
        let t = self.iter.max(1) as f32;
        let pre_max = self.pre_regret.iter().cloned().fold(0.0f32, f32::max) / t;
        let post_max = self.post.regret.iter().cloned().fold(0.0f32, f32::max) / t;
        (pre_max, post_max)
    }
}

/// MULTIWAY connected co-solver: np-way fold/call preflop threaded into a postflop
/// MCCFR engine PER live-count cell (live ∈ 2..=np). One trajectory deals np hands,
/// traverses the preflop tree (players fold/call), and at the flop-entry chance node
/// routes the LIVE players into the cell engine for their live count — co-adapting
/// preflop AND every postflop cell in ONE pass, no per-subgame re-convergence.
struct ConnectedMW {
    np: usize,
    pre_tree: FlatTree,
    pre_local: Vec<i32>,
    pre_max_na: usize,
    pre_regret: Vec<f32>,
    pre_cum: Vec<f32>,
    cells: Vec<Option<Mccfr>>,      // [live] → postflop engine (Some for 2..=np)
    cell_tree: Vec<Option<FlatTree>>,
    card_to_hand: Vec<i32>,
    deck: Vec<u8>,
    rng: u64,
    iter: u64,
    prune: bool,
    prune_warmup: u64,
    prune_this: bool,
}

impl ConnectedMW {
    fn new(np: usize, nb: usize) -> Self {
        let nt: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        let nr: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        let mut cells: Vec<Option<Mccfr>> = (0..=np).map(|_| None).collect();
        let mut cell_tree: Vec<Option<FlatTree>> = (0..=np).map(|_| None).collect();
        let mut card_to_hand: Vec<i32> = Vec::new();
        let mut deck: Vec<u8> = Vec::new();
        for live in 2..=np {
            // fold/call limped cell: every live player in for 2 (blind+call) ⇒ pot=2·live.
            let g = build_shrunk_cell(live as u8, 2, 2 * live as i32, nb, nt, nr);
            let post = Mccfr::new(&g);
            if card_to_hand.is_empty() {
                card_to_hand = vec![-1i32; 52 * 52];
                for (h, &(a, b)) in post.hand_cards.iter().enumerate() {
                    card_to_hand[a as usize * 52 + b as usize] = h as i32;
                    card_to_hand[b as usize * 52 + a as usize] = h as i32;
                }
                let mut on_board = [true; 52];
                for &(a, b) in &post.hand_cards {
                    on_board[a as usize] = false;
                    on_board[b as usize] = false;
                }
                deck = (0..52u8).filter(|&c| !on_board[c as usize]).collect();
            }
            cell_tree[live] = Some(g.tree.clone());
            cells[live] = Some(post);
        }
        let mut spec = production_game_v1();
        spec.num_players = np as u8;
        let cfg = spec.preflop_tree_config(BetSizeOptions { bet: vec![], raise: vec![] });
        let pre_tree = build_tree_preflop_only(&cfg).expect("preflop MW tree");
        let nn = pre_tree.num_nodes();
        let mut pre_local = vec![-1i32; nn];
        let mut pre_ninfo = 0usize;
        let mut pre_max_na = 0usize;
        for i in 0..nn {
            if pre_tree.nodes[i].is_player() {
                pre_local[i] = pre_ninfo as i32;
                pre_ninfo += 1;
                pre_max_na = pre_max_na.max(pre_tree.nodes[i].num_children as usize);
            }
        }
        let stride = NUM_PREFLOP_CLASSES * pre_max_na.max(1);
        ConnectedMW {
            np,
            pre_tree,
            pre_local,
            pre_max_na: pre_max_na.max(1),
            pre_regret: vec![0.0; pre_ninfo * stride],
            pre_cum: vec![0.0; pre_ninfo * stride],
            cells,
            cell_tree,
            card_to_hand,
            deck,
            rng: 0xD1B54A32D192ED03,
            iter: 0,
            prune: std::env::var("MC_PRUNE").is_ok(),
            prune_warmup: std::env::var("MC_PRUNE_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000),
            prune_this: false,
        }
    }

    #[inline]
    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    fn pre_strategy(&self, local: usize, class: usize, na: usize, out: &mut [f32]) {
        let base = (local * NUM_PREFLOP_CLASSES + class) * self.pre_max_na;
        let mut sum = 0.0f32;
        for a in 0..na {
            let r = self.pre_regret[base + a].max(0.0);
            out[a] = r;
            sum += r;
        }
        if sum > 0.0 {
            for a in 0..na { out[a] /= sum; }
        } else {
            let u = 1.0 / na as f32;
            for a in 0..na { out[a] = u; }
        }
    }

    fn pre_terminal(&self, node: usize, traverser: usize) -> f32 {
        let np = self.np;
        let fold_mask = self.pre_tree.get_folded_mask(node);
        let contribs: Vec<i32> = (0..np).map(|p| self.pre_tree.get_contribution(node, p as u8)).collect();
        let half_pot = self.pre_tree.starting_pot as f32 / np as f32 + contribs[traverser] as f32;
        if (fold_mask >> traverser) & 1 == 1 {
            return -half_pot;
        }
        let total: i32 = self.pre_tree.starting_pot + contribs.iter().sum::<i32>();
        total as f32 - half_pot
    }

    fn traverse_pre(&mut self, node: usize, traverser: usize, class: &[usize], hand: &[usize]) -> f32 {
        let n = &self.pre_tree.nodes[node];
        if n.is_terminal() {
            return self.pre_terminal(node, traverser);
        }
        if n.is_chance() {
            // FLOP-ENTRY SEAM: route the live players into their live-count cell.
            let fold_mask = self.pre_tree.get_folded_mask(node);
            let live_seats: Vec<usize> = (0..self.np).filter(|&p| (fold_mask >> p) & 1 == 0).collect();
            let live = live_seats.len();
            let tpost = match live_seats.iter().position(|&s| s == traverser) {
                Some(i) => i,
                None => return self.pre_terminal(node, traverser), // traverser not live (shouldn't reach here)
            };
            let post_hands: Vec<usize> = live_seats.iter().map(|&s| hand[s]).collect();
            let tree = self.cell_tree[live].as_ref().unwrap().clone();
            let pt = self.prune_this;
            let cell = self.cells[live].as_mut().unwrap();
            cell.prune_this = pt;
            cell.sample_runout();
            return cell.traverse(&tree, 0, tpost, &post_hands, 1.0);
        }
        let kids: Vec<u32> = self.pre_tree.node_children(node).to_vec();
        let na = kids.len();
        let player = n.player_id as usize;
        let local = self.pre_local[node] as usize;
        let cls = class[player];
        let mut strat = [0.0f32; 16];
        self.pre_strategy(local, cls, na, &mut strat[..na]);
        if player == traverser {
            let base = (local * NUM_PREFLOP_CLASSES + cls) * self.pre_max_na;
            let mut cv = [0.0f32; 16];
            let mut v = 0.0f32;
            for a in 0..na {
                cv[a] = self.traverse_pre(kids[a] as usize, traverser, class, hand);
                v += strat[a] * cv[a];
            }
            for a in 0..na {
                self.pre_regret[base + a] = (self.pre_regret[base + a] + cv[a] - v).max(0.0);
                self.pre_cum[base + a] += strat[a];
            }
            v
        } else {
            let r = (self.rand() as f64 / u64::MAX as f64) as f32;
            let mut acc = 0.0;
            let mut a = na - 1;
            for (i, &p) in strat[..na].iter().enumerate() {
                acc += p;
                if r <= acc { a = i; break; }
            }
            self.traverse_pre(kids[a] as usize, traverser, class, hand)
        }
    }

    fn deal(&mut self) -> (Vec<usize>, Vec<usize>) {
        let nd = self.deck.len();
        let mut cards = vec![0u8; 2 * self.np];
        let mut k = 0;
        while k < 2 * self.np {
            let idx = (self.rand() as usize) % nd;
            let c = self.deck[idx];
            if cards[..k].iter().any(|&u| u == c) { continue; }
            cards[k] = c;
            k += 1;
        }
        let mut class = vec![0usize; self.np];
        let mut hand = vec![0usize; self.np];
        for p in 0..self.np {
            let (c1, c2) = (cards[2 * p], cards[2 * p + 1]);
            class[p] = PreflopClass::from_combo(c1 as Card, c2 as Card).index();
            hand[p] = self.card_to_hand[c1 as usize * 52 + c2 as usize] as usize;
        }
        (class, hand)
    }

    fn run_iter(&mut self, batch: usize) {
        for b in 0..batch {
            self.iter += 1;
            self.prune_this = self.prune
                && self.iter > self.prune_warmup
                && (self.rand() as f64 / u64::MAX as f64) < 0.95;
            let traverser = b % self.np;
            let (class, hand) = self.deal();
            self.traverse_pre(0, traverser, &class, &hand);
        }
    }

    fn prune_frac(&self) -> f64 {
        let mut p = 0u64;
        let mut v = 0u64;
        for c in self.cells.iter().flatten() {
            p += c.pruned_nodes;
            v += c.visited_nodes;
        }
        if p + v == 0 { 0.0 } else { p as f64 / (p + v) as f64 }
    }

    fn regret_bound(&self) -> (f32, f32) {
        let t = self.iter.max(1) as f32;
        let pre_max = self.pre_regret.iter().cloned().fold(0.0f32, f32::max) / t;
        let mut post_max = 0.0f32;
        for c in self.cells.iter().flatten() {
            // pruning floors regret negative; the bound is over POSITIVE regret.
            post_max = post_max.max(c.regret.iter().cloned().fold(0.0f32, f32::max));
        }
        (pre_max, post_max / t)
    }

    fn pre_call_freq(&self, class: usize) -> Option<f32> {
        // first fold/call decision node for the acting player.
        for i in 0..self.pre_tree.num_nodes() {
            let n = &self.pre_tree.nodes[i];
            if !n.is_player() { continue; }
            let actor = n.player_id;
            let kids: Vec<u32> = self.pre_tree.node_children(i).to_vec();
            for (a, &k) in kids.iter().enumerate() {
                if self.pre_tree.nodes[k as usize].is_terminal()
                    && (self.pre_tree.get_folded_mask(k as usize) >> actor) & 1 == 1
                {
                    let local = self.pre_local[i];
                    if local < 0 { return None; }
                    let na = n.num_children as usize;
                    let base = (local as usize * NUM_PREFLOP_CLASSES + class) * self.pre_max_na;
                    let sum: f32 = (0..na).map(|x| self.pre_cum[base + x]).sum();
                    if sum <= 0.0 { return None; }
                    return Some(1.0 - self.pre_cum[base + a] / sum);
                }
            }
        }
        None
    }

    fn fold_node(&self) -> Option<(usize, usize, usize)> {
        for i in 0..self.pre_tree.num_nodes() {
            let n = &self.pre_tree.nodes[i];
            if !n.is_player() { continue; }
            let actor = n.player_id as usize;
            for (a, &k) in self.pre_tree.node_children(i).iter().enumerate() {
                if self.pre_tree.nodes[k as usize].is_terminal()
                    && (self.pre_tree.get_folded_mask(k as usize) >> actor) & 1 == 1 {
                    return Some((i, a, actor));
                }
            }
        }
        None
    }

    /// Pure preflop rollout under the AVERAGE strategy, hero's action at its
    /// fold-decision node forced to `forced`. Returns hero's terminal value.
    fn rollout(&mut self, fnode: usize, forced: usize, hero: usize, class: &[usize], hand: &[usize]) -> f32 {
        let mut node = 0usize;
        loop {
            let n = &self.pre_tree.nodes[node];
            if n.is_terminal() { return self.pre_terminal(node, hero); }
            if n.is_chance() {
                let fm = self.pre_tree.get_folded_mask(node);
                let live_seats: Vec<usize> = (0..self.np).filter(|&p| (fm >> p) & 1 == 0).collect();
                let live = live_seats.len();
                let tpost = match live_seats.iter().position(|&s| s == hero) {
                    Some(i) => i, None => return self.pre_terminal(node, hero),
                };
                let post_hands: Vec<usize> = live_seats.iter().map(|&s| hand[s]).collect();
                let tree = self.cell_tree[live].as_ref().unwrap().clone();
                let cell = self.cells[live].as_mut().unwrap();
                cell.sample_runout();
                return cell.rollout_cell(&tree, tpost, &post_hands);
            }
            let player = n.player_id as usize;
            let local = self.pre_local[node] as usize;
            let na = n.num_children as usize;
            let kids = self.pre_tree.node_children(node).to_vec();
            let a = if node == fnode && player == hero {
                forced
            } else {
                let mut st = [0.0f32; 16];
                self.pre_avg(local, class[player], na, &mut st[..na]);
                let r = (self.rand() as f64 / u64::MAX as f64) as f32;
                let (mut acc, mut sel) = (0.0f32, na - 1);
                for i in 0..na { acc += st[i]; if r <= acc { sel = i; break; } }
                sel
            };
            node = kids[a] as usize;
        }
    }

    fn pre_avg(&self, local: usize, cls: usize, na: usize, out: &mut [f32]) {
        let base = (local * NUM_PREFLOP_CLASSES + cls) * self.pre_max_na;
        let sum: f32 = (0..na).map(|a| self.pre_cum[base + a].max(0.0)).sum();
        if sum > 0.0 { for a in 0..na { out[a] = self.pre_cum[base + a].max(0.0) / sum; } }
        else { for a in 0..na { out[a] = 1.0 / na as f32; } }
    }

    /// MC estimate of EV(call) and EV(fold) at the fold-decision infoset for a hero
    /// holding `hero_cards`, under THIS engine's average strategy. EV(call)≈EV(fold)
    /// ⇒ indifferent ⇒ any call-freq is a valid equilibrium (not a bug).
    fn ev_call_fold(&mut self, hero_cards: (u8, u8), n_samples: usize) -> (f32, f32) {
        let (fnode, fold_a, hero) = self.fold_node().unwrap();
        let na = self.pre_tree.nodes[fnode].num_children as usize;
        let call_a = (0..na).find(|&a| a != fold_a).unwrap();
        let fold_child = self.pre_tree.node_children(fnode)[fold_a] as usize;
        let ev_fold = self.pre_terminal(fold_child, hero);
        let hh = self.card_to_hand[hero_cards.0 as usize * 52 + hero_cards.1 as usize];
        if hh < 0 { return (f32::NAN, ev_fold); }
        let hero_hand = hh as usize;
        let hero_cls = PreflopClass::from_combo(hero_cards.0 as Card, hero_cards.1 as Card).index();
        let nd = self.deck.len();
        let (mut sum, mut cnt) = (0.0f64, 0usize);
        for _ in 0..n_samples {
            let mut used = [false; 52];
            used[hero_cards.0 as usize] = true; used[hero_cards.1 as usize] = true;
            let mut class = vec![0usize; self.np];
            let mut hand = vec![0usize; self.np];
            class[hero] = hero_cls; hand[hero] = hero_hand;
            let mut ok = true;
            for p in 0..self.np {
                if p == hero { continue; }
                let mut tries = 0;
                loop {
                    tries += 1; if tries > 1000 { ok = false; break; }
                    let (r1, r2) = (self.rand() as usize, self.rand() as usize);
                    let c1 = self.deck[r1 % nd];
                    let c2 = self.deck[r2 % nd];
                    if c1 == c2 || used[c1 as usize] || used[c2 as usize] { continue; }
                    let h = self.card_to_hand[c1 as usize * 52 + c2 as usize];
                    if h < 0 { continue; }
                    used[c1 as usize] = true; used[c2 as usize] = true;
                    class[p] = PreflopClass::from_combo(c1 as Card, c2 as Card).index();
                    hand[p] = h as usize;
                    break;
                }
                if !ok { break; }
            }
            if !ok { continue; }
            sum += self.rollout(fnode, call_a, hero, &class, &hand) as f64;
            cnt += 1;
        }
        ((sum / cnt.max(1) as f64) as f32, ev_fold)
    }
}

/// Per-flop postflop cell set (one flop's bucketing → cells by live count).
struct FlopCells {
    cells: Vec<Option<Mccfr>>,
    cell_tree: Vec<Option<FlatTree>>,
    card_to_hand: Vec<i32>,
    deck: Vec<u8>,
}

/// MULTI-FLOP connected co-solver: samples one flop per trajectory from NF flops,
/// preflop regrets SHARED across flops (flop unseen preflop), postflop cells
/// per-flop. Measures whether trajectories-to-converge scales with flop count
/// (the load-bearing assumption in the all-flops wall-time projection).
struct ConnectedMWF {
    np: usize,
    pre_tree: FlatTree,
    pre_local: Vec<i32>,
    pre_max_na: usize,
    pre_regret: Vec<f32>,
    pre_cum: Vec<f32>,
    flops: Vec<FlopCells>,
    rng: u64,
    iter: u64,
}

impl ConnectedMWF {
    fn new(np: usize, nb: usize, flop_indices: &[usize]) -> Self {
        let nt: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        let nr: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        let mut flops = Vec::new();
        for &fi in flop_indices {
            std::env::set_var("MC_FLOP", fi.to_string());
            let mut cells: Vec<Option<Mccfr>> = (0..=np).map(|_| None).collect();
            let mut cell_tree: Vec<Option<FlatTree>> = (0..=np).map(|_| None).collect();
            let mut card_to_hand: Vec<i32> = Vec::new();
            let mut deck: Vec<u8> = Vec::new();
            for live in 2..=np {
                let g = build_shrunk_cell(live as u8, 2, 2 * live as i32, nb, nt, nr);
                let post = Mccfr::new(&g);
                if card_to_hand.is_empty() {
                    card_to_hand = vec![-1i32; 52 * 52];
                    for (h, &(a, b)) in post.hand_cards.iter().enumerate() {
                        card_to_hand[a as usize * 52 + b as usize] = h as i32;
                        card_to_hand[b as usize * 52 + a as usize] = h as i32;
                    }
                    let mut on_board = [true; 52];
                    for &(a, b) in &post.hand_cards {
                        on_board[a as usize] = false;
                        on_board[b as usize] = false;
                    }
                    deck = (0..52u8).filter(|&c| !on_board[c as usize]).collect();
                }
                cell_tree[live] = Some(g.tree.clone());
                cells[live] = Some(post);
            }
            flops.push(FlopCells { cells, cell_tree, card_to_hand, deck });
        }
        let mut spec = production_game_v1();
        spec.num_players = np as u8;
        let cfg = spec.preflop_tree_config(BetSizeOptions { bet: vec![], raise: vec![] });
        let pre_tree = build_tree_preflop_only(&cfg).expect("preflop MW tree");
        let nn = pre_tree.num_nodes();
        let mut pre_local = vec![-1i32; nn];
        let mut pre_ninfo = 0usize;
        let mut pre_max_na = 0usize;
        for i in 0..nn {
            if pre_tree.nodes[i].is_player() {
                pre_local[i] = pre_ninfo as i32;
                pre_ninfo += 1;
                pre_max_na = pre_max_na.max(pre_tree.nodes[i].num_children as usize);
            }
        }
        let stride = NUM_PREFLOP_CLASSES * pre_max_na.max(1);
        ConnectedMWF {
            np,
            pre_tree,
            pre_local,
            pre_max_na: pre_max_na.max(1),
            pre_regret: vec![0.0; pre_ninfo * stride],
            pre_cum: vec![0.0; pre_ninfo * stride],
            flops,
            rng: 0xD1B54A32D192ED03,
            iter: 0,
        }
    }

    #[inline]
    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    fn pre_strategy(&self, local: usize, class: usize, na: usize, out: &mut [f32]) {
        let base = (local * NUM_PREFLOP_CLASSES + class) * self.pre_max_na;
        let mut sum = 0.0f32;
        for a in 0..na {
            let r = self.pre_regret[base + a].max(0.0);
            out[a] = r;
            sum += r;
        }
        if sum > 0.0 {
            for a in 0..na { out[a] /= sum; }
        } else {
            let u = 1.0 / na as f32;
            for a in 0..na { out[a] = u; }
        }
    }

    fn pre_terminal(&self, node: usize, traverser: usize) -> f32 {
        let np = self.np;
        let fold_mask = self.pre_tree.get_folded_mask(node);
        let contribs: Vec<i32> = (0..np).map(|p| self.pre_tree.get_contribution(node, p as u8)).collect();
        let half_pot = self.pre_tree.starting_pot as f32 / np as f32 + contribs[traverser] as f32;
        if (fold_mask >> traverser) & 1 == 1 {
            return -half_pot;
        }
        let total: i32 = self.pre_tree.starting_pot + contribs.iter().sum::<i32>();
        total as f32 - half_pot
    }

    fn traverse_pre(&mut self, fi: usize, node: usize, traverser: usize, class: &[usize], hand: &[usize]) -> f32 {
        let n = &self.pre_tree.nodes[node];
        if n.is_terminal() {
            return self.pre_terminal(node, traverser);
        }
        if n.is_chance() {
            let fold_mask = self.pre_tree.get_folded_mask(node);
            let live_seats: Vec<usize> = (0..self.np).filter(|&p| (fold_mask >> p) & 1 == 0).collect();
            let live = live_seats.len();
            let tpost = match live_seats.iter().position(|&s| s == traverser) {
                Some(i) => i,
                None => return self.pre_terminal(node, traverser),
            };
            let post_hands: Vec<usize> = live_seats.iter().map(|&s| hand[s]).collect();
            let tree = self.flops[fi].cell_tree[live].as_ref().unwrap().clone();
            let cell = self.flops[fi].cells[live].as_mut().unwrap();
            cell.sample_runout();
            return cell.traverse(&tree, 0, tpost, &post_hands, 1.0);
        }
        let kids: Vec<u32> = self.pre_tree.node_children(node).to_vec();
        let na = kids.len();
        let player = n.player_id as usize;
        let local = self.pre_local[node] as usize;
        let cls = class[player];
        let mut strat = [0.0f32; 16];
        self.pre_strategy(local, cls, na, &mut strat[..na]);
        if player == traverser {
            let base = (local * NUM_PREFLOP_CLASSES + cls) * self.pre_max_na;
            let mut cv = [0.0f32; 16];
            let mut v = 0.0f32;
            for a in 0..na {
                cv[a] = self.traverse_pre(fi, kids[a] as usize, traverser, class, hand);
                v += strat[a] * cv[a];
            }
            for a in 0..na {
                self.pre_regret[base + a] = (self.pre_regret[base + a] + cv[a] - v).max(0.0);
                self.pre_cum[base + a] += strat[a];
            }
            v
        } else {
            let r = (self.rand() as f64 / u64::MAX as f64) as f32;
            let mut acc = 0.0;
            let mut a = na - 1;
            for (i, &p) in strat[..na].iter().enumerate() {
                acc += p;
                if r <= acc { a = i; break; }
            }
            self.traverse_pre(fi, kids[a] as usize, traverser, class, hand)
        }
    }

    fn run_iter(&mut self, batch: usize) {
        let nf = self.flops.len();
        for b in 0..batch {
            self.iter += 1;
            let traverser = b % self.np;
            let fi = (self.rand() as usize) % nf;
            // deal np hands from this flop's deck
            let nd = self.flops[fi].deck.len();
            let mut cards = vec![0u8; 2 * self.np];
            let mut k = 0;
            while k < 2 * self.np {
                let idx = (self.rand() as usize) % nd;
                let c = self.flops[fi].deck[idx];
                if cards[..k].iter().any(|&u| u == c) { continue; }
                cards[k] = c;
                k += 1;
            }
            let mut class = vec![0usize; self.np];
            let mut hand = vec![0usize; self.np];
            for p in 0..self.np {
                let (c1, c2) = (cards[2 * p], cards[2 * p + 1]);
                class[p] = PreflopClass::from_combo(c1 as Card, c2 as Card).index();
                hand[p] = self.flops[fi].card_to_hand[c1 as usize * 52 + c2 as usize] as usize;
            }
            self.traverse_pre(fi, 0, traverser, &class, &hand);
        }
    }

    fn regret_bound(&self) -> (f32, f32) {
        let t = self.iter.max(1) as f32;
        let pre_max = self.pre_regret.iter().cloned().fold(0.0f32, f32::max) / t;
        let mut post_max = 0.0f32;
        for f in &self.flops {
            for c in f.cells.iter().flatten() {
                post_max = post_max.max(c.regret.iter().cloned().fold(0.0f32, f32::max));
            }
        }
        (pre_max, post_max / t)
    }

    fn pre_call_freq(&self, class: usize) -> Option<f32> {
        for i in 0..self.pre_tree.num_nodes() {
            let n = &self.pre_tree.nodes[i];
            if !n.is_player() { continue; }
            let actor = n.player_id;
            for (a, &k) in self.pre_tree.node_children(i).iter().enumerate() {
                if self.pre_tree.nodes[k as usize].is_terminal()
                    && (self.pre_tree.get_folded_mask(k as usize) >> actor) & 1 == 1 {
                    let local = self.pre_local[i];
                    if local < 0 { return None; }
                    let na = n.num_children as usize;
                    let base = (local as usize * NUM_PREFLOP_CLASSES + class) * self.pre_max_na;
                    let sum: f32 = (0..na).map(|x| self.pre_cum[base + x]).sum();
                    if sum <= 0.0 { return None; }
                    return Some(1.0 - self.pre_cum[base + a] / sum);
                }
            }
        }
        None
    }
}

fn connected_mwf_probe(np: usize, nb: usize) {
    let batch: usize = std::env::var("MC_B").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let nfs: Vec<usize> = std::env::var("MC_NFS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);
    // target regret bound to "converge" to (post maxR/T).
    let target_eps: f32 = std::env::var("MC_EPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3e-3);
    println!("ALL-FLOPS SCALING — traj to reach post maxR/T < {target_eps:e} vs flop count (np={np}, nb={nb})");
    println!("{:>6} {:>14} {:>14} {:>12}", "NF", "traj_to_eps", "traj/flop", "µs/traj");
    let all_flops: Vec<usize> = (0..1755).collect();
    for &nf in &nfs {
        // distinct flops spread across the canonical set.
        let step = (all_flops.len() / nf).max(1);
        let idxs: Vec<usize> = (0..nf).map(|i| (i * step) % all_flops.len()).collect();
        let mut c = ConnectedMWF::new(np, nb, &idxs);
        let mut total: u64 = 0;
        let t0 = Instant::now();
        let max_traj: u64 = 200_000_000;
        let mut hit = 0u64;
        loop {
            c.run_iter(batch);
            total += batch as u64;
            let (_pre, post) = c.regret_bound();
            if post < target_eps { hit = total; break; }
            if total >= max_traj { break; }
        }
        let secs = t0.elapsed().as_secs_f64();
        let per_flop = if nf > 0 { hit as f64 / nf as f64 } else { 0.0 };
        println!("{:>6} {:>14} {:>14.0} {:>12.3}", nf,
            if hit > 0 { hit.to_string() } else { format!(">{}", max_traj) },
            per_flop, secs / total as f64 * 1e6);
    }
    println!("\nSCALING READ: traj_to_eps ∝ NF (linear) ⇒ traj/flop ~CONSTANT ⇒ all-flops = ×1755 (projection holds).");
    println!("If traj/flop falls with NF ⇒ SUB-linear (bucketing generalizes) ⇒ projection is CONSERVATIVE.");
}

/// ACTION-ABSTRACTED connected co-solver: preflop with RAISES (Pluribus action
/// gradient), postflop cells keyed by (live, commit, pot) so every distinct
/// preflop betting line routes to its own seam cell. This is the structural piece
/// that deepens trajectories + multiplies infosets toward Pluribus realism.
struct ConnectedMWA {
    np: usize,
    pre_tree: FlatTree,
    pre_local: Vec<i32>,
    pre_max_na: usize,
    pre_regret: Vec<f32>,
    pre_cum: Vec<f32>,
    cells: std::collections::HashMap<(u8, i32, i32), Mccfr>,
    cell_tree: std::collections::HashMap<(u8, i32, i32), FlatTree>,
    card_to_hand: Vec<i32>,
    deck: Vec<u8>,
    rng: u64,
    iter: u64,
    prune: bool,
    prune_warmup: u64,
    prune_this: bool,
}

impl ConnectedMWA {
    fn cell_key(pre_tree: &FlatTree, node: usize, np: usize) -> (u8, i32, i32) {
        let fold_mask = pre_tree.get_folded_mask(node);
        let contribs: Vec<i32> = (0..np).map(|p| pre_tree.get_contribution(node, p as u8)).collect();
        let live: Vec<usize> = (0..np).filter(|&p| (fold_mask >> p) & 1 == 0).collect();
        let commit = contribs[live[0]]; // live players matched the last raise ⇒ equal
        let pot = pre_tree.starting_pot + contribs.iter().sum::<i32>();
        (live.len() as u8, commit, pot)
    }

    fn new(np: usize, nb: usize) -> Self {
        let nt: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        let nr: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        // PREFLOP action abstraction: fold/call + pot-sized raise, cap-3 (re-raises
        // limited). A reduced stand-in for Pluribus's up-to-14 raise sizes — captures
        // the gradient (preflop has raises → deeper + more seam cells) at bounded cell
        // count. MC_PRERAISES overrides the raise count.
        let nraises: usize = std::env::var("MC_PRERAISES").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let mut spec = production_game_v1();
        spec.num_players = np as u8;
        let raises: Vec<BetSize> = (0..nraises).map(|i| BetSize::PotRelative(1.0 + i as f64)).collect();
        let mut cfg = spec.preflop_tree_config(BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: raises });
        cfg.max_bets_per_street = BetCap::all(3);
        let pre_tree = build_tree_preflop_only(&cfg).expect("preflop MWA tree");
        let nn = pre_tree.num_nodes();
        // distinct flop-entry cells.
        let mut keys: std::collections::HashSet<(u8, i32, i32)> = std::collections::HashSet::new();
        for i in 0..nn {
            if pre_tree.nodes[i].is_chance() {
                let k = Self::cell_key(&pre_tree, i, np);
                if k.0 >= 2 { keys.insert(k); }
            }
        }
        let mut cells = std::collections::HashMap::new();
        let mut cell_tree = std::collections::HashMap::new();
        let mut card_to_hand: Vec<i32> = Vec::new();
        let mut deck: Vec<u8> = Vec::new();
        for &(live, commit, pot) in &keys {
            let g = build_shrunk_cell(live, commit, pot, nb, nt, nr);
            let post = Mccfr::new(&g);
            if card_to_hand.is_empty() {
                card_to_hand = vec![-1i32; 52 * 52];
                for (h, &(a, b)) in post.hand_cards.iter().enumerate() {
                    card_to_hand[a as usize * 52 + b as usize] = h as i32;
                    card_to_hand[b as usize * 52 + a as usize] = h as i32;
                }
                let mut on_board = [true; 52];
                for &(a, b) in &post.hand_cards { on_board[a as usize] = false; on_board[b as usize] = false; }
                deck = (0..52u8).filter(|&c| !on_board[c as usize]).collect();
            }
            cell_tree.insert((live, commit, pot), g.tree.clone());
            cells.insert((live, commit, pot), post);
        }
        let mut pre_local = vec![-1i32; nn];
        let mut pre_ninfo = 0usize;
        let mut pre_max_na = 0usize;
        for i in 0..nn {
            if pre_tree.nodes[i].is_player() {
                pre_local[i] = pre_ninfo as i32;
                pre_ninfo += 1;
                pre_max_na = pre_max_na.max(pre_tree.nodes[i].num_children as usize);
            }
        }
        let stride = NUM_PREFLOP_CLASSES * pre_max_na.max(1);
        ConnectedMWA {
            np, pre_tree, pre_local, pre_max_na: pre_max_na.max(1),
            pre_regret: vec![0.0; pre_ninfo * stride],
            pre_cum: vec![0.0; pre_ninfo * stride],
            cells, cell_tree, card_to_hand, deck,
            rng: 0xD1B54A32D192ED03, iter: 0,
            prune: std::env::var("MC_PRUNE").is_ok(),
            prune_warmup: std::env::var("MC_PRUNE_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000),
            prune_this: false,
        }
    }

    #[inline]
    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.rng = x; x
    }

    fn pre_strategy(&self, local: usize, class: usize, na: usize, out: &mut [f32]) {
        let base = (local * NUM_PREFLOP_CLASSES + class) * self.pre_max_na;
        let mut sum = 0.0f32;
        for a in 0..na { let r = self.pre_regret[base + a].max(0.0); out[a] = r; sum += r; }
        if sum > 0.0 { for a in 0..na { out[a] /= sum; } } else { let u = 1.0 / na as f32; for a in 0..na { out[a] = u; } }
    }

    fn pre_terminal(&self, node: usize, traverser: usize) -> f32 {
        let np = self.np;
        let fold_mask = self.pre_tree.get_folded_mask(node);
        let contribs: Vec<i32> = (0..np).map(|p| self.pre_tree.get_contribution(node, p as u8)).collect();
        let half_pot = self.pre_tree.starting_pot as f32 / np as f32 + contribs[traverser] as f32;
        if (fold_mask >> traverser) & 1 == 1 { return -half_pot; }
        let total: i32 = self.pre_tree.starting_pot + contribs.iter().sum::<i32>();
        total as f32 - half_pot
    }

    fn traverse_pre(&mut self, node: usize, traverser: usize, class: &[usize], hand: &[usize]) -> f32 {
        let n = &self.pre_tree.nodes[node];
        if n.is_terminal() { return self.pre_terminal(node, traverser); }
        if n.is_chance() {
            let fold_mask = self.pre_tree.get_folded_mask(node);
            let live_seats: Vec<usize> = (0..self.np).filter(|&p| (fold_mask >> p) & 1 == 0).collect();
            let tpost = match live_seats.iter().position(|&s| s == traverser) {
                Some(i) => i, None => return self.pre_terminal(node, traverser),
            };
            let key = Self::cell_key(&self.pre_tree, node, self.np);
            let post_hands: Vec<usize> = live_seats.iter().map(|&s| hand[s]).collect();
            let tree = match self.cell_tree.get(&key) { Some(t) => t.clone(), None => return self.pre_terminal(node, traverser) };
            let pt = self.prune_this;
            let cell = self.cells.get_mut(&key).unwrap();
            cell.prune_this = pt;
            cell.sample_runout();
            return cell.traverse(&tree, 0, tpost, &post_hands, 1.0);
        }
        let kids: Vec<u32> = self.pre_tree.node_children(node).to_vec();
        let na = kids.len();
        let player = n.player_id as usize;
        let local = self.pre_local[node] as usize;
        let cls = class[player];
        let mut strat = [0.0f32; 16];
        self.pre_strategy(local, cls, na, &mut strat[..na]);
        if player == traverser {
            let base = (local * NUM_PREFLOP_CLASSES + cls) * self.pre_max_na;
            let mut cv = [0.0f32; 16];
            let mut v = 0.0f32;
            for a in 0..na { cv[a] = self.traverse_pre(kids[a] as usize, traverser, class, hand); v += strat[a] * cv[a]; }
            for a in 0..na {
                self.pre_regret[base + a] = (self.pre_regret[base + a] + cv[a] - v).max(0.0);
                self.pre_cum[base + a] += strat[a];
            }
            v
        } else {
            let r = (self.rand() as f64 / u64::MAX as f64) as f32;
            let mut acc = 0.0; let mut a = na - 1;
            for (i, &p) in strat[..na].iter().enumerate() { acc += p; if r <= acc { a = i; break; } }
            self.traverse_pre(kids[a] as usize, traverser, class, hand)
        }
    }

    fn run_iter(&mut self, batch: usize) {
        for b in 0..batch {
            self.iter += 1;
            self.prune_this = self.prune && self.iter > self.prune_warmup && (self.rand() as f64 / u64::MAX as f64) < 0.95;
            let traverser = b % self.np;
            let nd = self.deck.len();
            let mut cards = vec![0u8; 2 * self.np];
            let mut k = 0;
            while k < 2 * self.np {
                let idx = (self.rand() as usize) % nd;
                let c = self.deck[idx];
                if cards[..k].iter().any(|&u| u == c) { continue; }
                cards[k] = c; k += 1;
            }
            let mut class = vec![0usize; self.np];
            let mut hand = vec![0usize; self.np];
            for p in 0..self.np {
                let (c1, c2) = (cards[2 * p], cards[2 * p + 1]);
                class[p] = PreflopClass::from_combo(c1 as Card, c2 as Card).index();
                hand[p] = self.card_to_hand[c1 as usize * 52 + c2 as usize] as usize;
            }
            self.traverse_pre(0, traverser, &class, &hand);
        }
    }

    fn prune_frac(&self) -> f64 {
        let (mut p, mut v) = (0u64, 0u64);
        for c in self.cells.values() { p += c.pruned_nodes; v += c.visited_nodes; }
        if p + v == 0 { 0.0 } else { p as f64 / (p + v) as f64 }
    }

    fn regret_bound(&self) -> (f32, f32) {
        let t = self.iter.max(1) as f32;
        let pre_max = self.pre_regret.iter().cloned().fold(0.0f32, f32::max) / t;
        let mut post_max = 0.0f32;
        for c in self.cells.values() { post_max = post_max.max(c.regret.iter().cloned().fold(0.0f32, f32::max)); }
        (pre_max, post_max / t)
    }

    fn pre_call_freq(&self, class: usize) -> Option<f32> {
        for i in 0..self.pre_tree.num_nodes() {
            let n = &self.pre_tree.nodes[i];
            if !n.is_player() { continue; }
            let actor = n.player_id;
            let kids: Vec<u32> = self.pre_tree.node_children(i).to_vec();
            for (a, &k) in kids.iter().enumerate() {
                if self.pre_tree.nodes[k as usize].is_terminal() && (self.pre_tree.get_folded_mask(k as usize) >> actor) & 1 == 1 {
                    let local = self.pre_local[i];
                    if local < 0 { return None; }
                    let na = n.num_children as usize;
                    let base = (local as usize * NUM_PREFLOP_CLASSES + class) * self.pre_max_na;
                    let sum: f32 = (0..na).map(|x| self.pre_cum[base + x]).sum();
                    if sum <= 0.0 { return None; }
                    return Some(1.0 - self.pre_cum[base + a] / sum);
                }
            }
        }
        None
    }
}

fn connected_mwa_probe(np: usize, nb: usize) {
    let batch: usize = std::env::var("MC_B").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let mut c = ConnectedMWA::new(np, nb);
    let mut tot_infosets = c.pre_local.iter().filter(|&&x| x >= 0).count() * NUM_PREFLOP_CLASSES;
    for cell in c.cells.values() { tot_infosets += cell.n_info * cell.nb; }
    println!("CONNECTED MCCFR — ACTION-ABSTRACTED (np={np}, preflop raises+cap3, nb={nb}, prune={})", c.prune);
    println!("preflop infosets={} (×{} classes); {} distinct (live,commit,pot) cells; ~{} total infosets",
        c.pre_local.iter().filter(|&&x| x >= 0).count(), NUM_PREFLOP_CLASSES, c.cells.len(), tot_infosets);
    println!("\n{:>12} {:>14} {:>14} {:>10} {:>10} {:>9} {:>9}", "traj", "pre maxR/T", "post maxR/T", "AA call", "72o call", "prune%", "µs/traj");
    let mut total: u64 = 0;
    let targets: [u64; 6] = [16_384, 131_072, 524_288, 2_097_152, 8_388_608, 33_554_432];
    let mut run_secs = 0.0f64;
    for &target in &targets {
        let tr = Instant::now();
        while total < target { c.run_iter(batch); total += batch as u64; }
        run_secs += tr.elapsed().as_secs_f64();
        let (pre_r, post_r) = c.regret_bound();
        let aa = c.pre_call_freq(PreflopClass::from_combo(48, 49).index()).unwrap_or(-1.0);
        let o72 = c.pre_call_freq(PreflopClass::from_combo(20, 0).index()).unwrap_or(-1.0);
        println!("{:>12} {:>14.4e} {:>14.4e} {:>10.4} {:>10.4} {:>8.1}% {:>9.3}",
            total, pre_r, post_r, aa, o72, 100.0 * c.prune_frac(), run_secs / total as f64 * 1e6);
    }
    println!("\nACTION-ABSTRACTED: preflop raises deepen trajectories + multiply cells/infosets — the realistic Pluribus number.");
}

/// PARALLEL throughput: MCCFR trajectories are independent ⇒ embarrassingly
/// parallel on CPU cores (Pluribus's 64-core approach; GPU is the WRONG target —
/// divergent/scattered, why their paper used no GPU). Measures aggregate traj/s
/// at increasing thread counts to get the REAL multi-core number (single-thread ×
/// cores) the wall-time projection needs. Each thread runs an independent
/// connected co-solver (same game, different seed) for MC_SECS; we sum throughput.
fn connected_parallel_probe(np: usize, nb: usize) {
    let secs: f64 = std::env::var("MC_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let ncores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let tcounts: Vec<usize> = std::env::var("MC_THREADS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, ncores / 2, ncores]);
    println!("PARALLEL MCCFR THROUGHPUT (connected co-solver, np={np}, nb={nb}, {secs}s/run, {ncores} cores avail)");
    println!("{:>8} {:>16} {:>16} {:>10}", "threads", "M traj/s total", "M traj/s/thread", "speedup");
    let mut base = 0.0f64;
    for (i, &nt) in tcounts.iter().enumerate() {
        // build the solvers OUTSIDE the timer.
        let mut solvers: Vec<ConnectedMW> = (0..nt)
            .map(|tid| {
                let mut c = ConnectedMW::new(np, nb);
                c.rng ^= (tid as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
                c
            })
            .collect();
        let counts: Vec<u64> = std::thread::scope(|s| {
            let handles: Vec<_> = solvers
                .iter_mut()
                .map(|c| {
                    s.spawn(move || {
                        let t0 = Instant::now();
                        let mut n = 0u64;
                        while t0.elapsed().as_secs_f64() < secs {
                            c.run_iter(4096);
                            n += 4096;
                        }
                        n
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let total: f64 = counts.iter().sum::<u64>() as f64 / secs / 1e6;
        if i == 0 { base = total; }
        println!("{:>8} {:>16.2} {:>16.2} {:>9.1}x", nt, total, total / nt as f64, total / base);
    }
    println!("\nREAD: near-linear speedup ⇒ MCCFR parallelizes cleanly on CPU cores (it's embarrassingly");
    println!("parallel — the Pluribus approach). The multi-core traj/s is what sets the real wall-time.");
}

fn connected_mw_probe(np: usize, nb: usize) {
    let batch: usize = std::env::var("MC_B").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let mut c = ConnectedMW::new(np, nb);
    let live_cells: Vec<usize> = (2..=np).collect();
    println!("CONNECTED MCCFR — MULTIWAY (np={np}, fold/call preflop, cells live={live_cells:?}, nb={nb})");
    println!("pre infosets={} (×{} classes); postflop cells:", c.pre_local.iter().filter(|&&x| x >= 0).count(), NUM_PREFLOP_CLASSES);
    for live in 2..=np {
        if let Some(cell) = &c.cells[live] {
            println!("  live-{live}: {} infosets, nb^num_opp={}", cell.n_info, (nb as u64).pow((live - 1) as u32));
        }
    }
    println!("\nCONVERGENCE — regret bound (maxR/T) preflop + max over cells (prune={}):", c.prune);
    println!("{:>12} {:>14} {:>14} {:>10} {:>10} {:>9} {:>9}", "traj", "pre maxR/T", "post maxR/T", "AA call", "72o call", "prune%", "µs/traj");
    let mut total: u64 = 0;
    let targets: [u64; 6] = [16_384, 131_072, 524_288, 2_097_152, 8_388_608, 33_554_432];
    let mut run_secs = 0.0f64;
    for &target in &targets {
        let tr = Instant::now();
        while total < target { c.run_iter(batch); total += batch as u64; }
        run_secs += tr.elapsed().as_secs_f64();
        let (pre_r, post_r) = c.regret_bound();
        let aa = c.pre_call_freq(PreflopClass::from_combo(48, 49).index()).unwrap_or(-1.0);
        let o72 = c.pre_call_freq(PreflopClass::from_combo(20, 0).index()).unwrap_or(-1.0);
        println!("{:>12} {:>14.4e} {:>14.4e} {:>10.4} {:>10.4} {:>8.1}% {:>9.3}",
            total, pre_r, post_r, aa, o72, 100.0 * c.prune_frac(), run_secs / total as f64 * 1e6);
    }
    println!("\nCONVERGENCE READ: pre+post maxR/T fall ~1/√T ⇒ multiway connected co-solve reaches Nash.");
}

fn connected_probe(nb: usize) {
    let iters: usize = std::env::var("MC_CITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let batch: usize = std::env::var("MC_B").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let mut c = ConnectedHu::new(nb);
    println!(
        "CONNECTED MCCFR (HU, fixed flop, fold/call preflop, real cap-3 postflop, nb={nb})\n\
         pruning={} prune_c={} | {iters}×{batch} traj batches\n",
        c.prune, c.prune_c
    );
    println!("pre infosets={} (×{} classes), post infosets={}",
        c.pre_local.iter().filter(|&&x| x >= 0).count(), NUM_PREFLOP_CLASSES,
        c.post.n_info);
    {
        // fixed flop = the 3 cards in no flop-live hand (complement of the deal deck).
        let in_deck = |x: u8| c.deck.contains(&x);
        let flop: Vec<u8> = (0..52u8).filter(|&x| !in_deck(x)).collect();
        let rc = ['2','3','4','5','6','7','8','9','T','J','Q','K','A'];
        let sc = ['c','d','h','s'];
        let s: Vec<String> = flop.iter().map(|&x| format!("{}{}", rc[(x/4) as usize], sc[(x%4) as usize])).collect();
        println!("FIXED FLOP: {:?} (cards {:?})  [runout sampled {}×{}]",
            s, flop, c.post.remaining_deck.len(),
            c.post.river_decks.iter().map(|v| v.len()).max().unwrap_or(0));
    }
    if std::env::var("MC_CDBG").is_ok() {
        println!("\n-- preflop tree dump --");
        for i in 0..c.pre_tree.num_nodes() {
            let n = &c.pre_tree.nodes[i];
            let kind = if n.is_terminal() { "TERM" } else if n.is_chance() { "CHANCE" } else { "PLAYER" };
            let kids: Vec<u32> = c.pre_tree.node_children(i).to_vec();
            let contribs: Vec<i32> = (0..2).map(|p| c.pre_tree.get_contribution(i, p as u8)).collect();
            println!("  node {i}: {kind} player={} bs={} nkids={} fold_mask={:02b} contribs={:?} kids={:?}",
                n.player_id, n.board_state, n.num_children, c.pre_tree.get_folded_mask(i), contribs, kids);
        }
        println!("-- end dump --\n");
    }

    // CONVERGENCE PROOF: regret bound (maxR/T, pre+post) at increasing trajectory
    // counts. If both fall ~1/√T, the connected co-solve reaches a Nash of the
    // connected game (self-consistent CFR convergence — the same check that
    // validated the isolated engine). AA call-freq tracked alongside as a sanity.
    println!("\nCONNECTED CONVERGENCE — regret bound (maxR/T) over preflop+postflop infosets:");
    println!("{:>12} {:>14} {:>14} {:>12} {:>12}", "traj", "pre maxR/T", "post maxR/T", "AA call", "µs/traj");
    let t0 = Instant::now();
    let mut total: u64 = 0;
    let targets: [u64; 6] = [16_384, 131_072, 524_288, 2_097_152, 8_388_608, 33_554_432];
    let mut run_secs = 0.0f64;
    for &target in &targets {
        let tr = Instant::now();
        while total < target {
            c.run_iter(batch);
            total += batch as u64;
        }
        run_secs += tr.elapsed().as_secs_f64();
        let (pre_r, post_r) = c.regret_bound();
        let aa = c.pre_call_freq(PreflopClass::from_combo(48, 49).index()).unwrap_or(-1.0);
        println!("{:>12} {:>14.4e} {:>14.4e} {:>12.4} {:>12.3}",
            total, pre_r, post_r, aa, run_secs / total as f64 * 1e6);
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("\nwall {:.2}s | {:.3} µs/traj | post pruned {} / visited {} ({:.0}% skipped)",
        secs, run_secs / total as f64 * 1e6, c.post.pruned_nodes, c.post.visited_nodes,
        100.0 * c.post.pruned_nodes as f64 / (c.post.pruned_nodes + c.post.visited_nodes).max(1) as f64);
    println!("CONVERGENCE READ: pre+post maxR/T should fall ~1/√T (×4 traj ⇒ ÷2 regret) → connected Nash.");

    // MONOTONICITY GATE: call-freq should rise with hand strength.
    println!("\nPREFLOP CALL-FREQ by class (monotonicity gate):");
    let probes: [(&str, u8, u8); 7] = [
        ("AA", 48, 49), ("KK", 44, 45), ("QQ", 40, 41),
        ("AKs", 48, 44), ("87s", 24, 20), ("72o", 20, 0), ("32o", 4, 0),
    ];
    for (name, a, b) in probes {
        let cl = PreflopClass::from_combo(a as Card, b as Card).index();
        match c.pre_call_freq(cl) {
            Some(f) => println!("  {name:<5} call={f:.4}"),
            None => println!("  {name:<5} (unvisited)"),
        }
    }
    println!("\nGATE: call-freq should be MONOTONE in strength (AA≥KK≥…≥72o) and AA call-freq should");
    println!("PLATEAU (Δ→0). Caveat: single fixed runout ⇒ directional, not exact equity.");
}

/// First canonical-flop index that is rainbow + 3 distinct ranks + disconnected
/// (no two ranks within 2) — a representative non-degenerate board for the verdict.
fn representative_flop_idx() -> usize {
    use solver_core::abstraction::flop_isomorphism::enumerate_canonical_flops;
    let flops = enumerate_canonical_flops();
    for (i, f) in flops.iter().enumerate() {
        let r = [f[0] >> 2, f[1] >> 2, f[2] >> 2];
        let s = [f[0] & 3, f[1] & 3, f[2] & 3];
        let distinct = r[0] != r[1] && r[1] != r[2] && r[0] != r[2];
        let rainbow = s[0] != s[1] && s[1] != s[2] && s[0] != s[2];
        let mut rr = r;
        rr.sort();
        let disc = rr[1] - rr[0] > 2 && rr[2] - rr[1] > 2;
        if distinct && rainbow && disc {
            return i;
        }
    }
    0
}

/// DCFR CO-SOLVE of the identical HU connected game (the N×fill baseline the
/// connected-MCCFR verdict must beat) — postflop fills dispatched on the GPU
/// (BucketedNativeGpu, the production fill engine), as the real co-solve runs.
/// Alternates: inject the SB calling range as the postflop entry weights → run K
/// WARM GPU iters → take per-hand SB root CFV (run()'s return) → regret-match SB's
/// preflop fold/call per class. Measures N = refreshes to plateau. CHEAP TEST: small
/// warm N ⇒ N×fill modest ⇒ connected-MCCFR case weakens (cheap-test-first).
#[cfg(not(feature = "metal"))]
fn dcfr_cosolve(_nb: usize) {
    eprintln!("dcfr_cosolve requires --features metal (the co-solve fills dispatch on the GPU).");
}

#[cfg(feature = "metal")]
fn dcfr_cosolve(nb: usize) {
    use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
    use solver_core::gpu_metal::context::MetalContext;

    let k: u32 = std::env::var("MC_K").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let max_refresh: usize = std::env::var("MC_REFRESH").ok().and_then(|s| s.parse().ok()).unwrap_or(120);
    let tol: f32 = std::env::var("MC_TOL").ok().and_then(|s| s.parse().ok()).unwrap_or(2e-3);
    let nt: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let nr: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    // MULTIWAY: live-k limped pot (each posts 1 blind, plays=commit 2 ⇒ pot=2k), so
    // the GPU bucketed engine (np≥3) drives the fill — the real co-solve dispatch.
    let live: u8 = std::env::var("MC_LIVE").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let np = live as usize;

    let g = build_shrunk_cell(live, 2, 2 * live as i32, nb, nt, nr);
    let ShrunkGame { tree, mut game, bk, .. } = g;
    let mut cpu = {
        let mut s = BucketedFlopCfr::new(&tree, game.table(), &bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        s
    };
    let ctx = MetalContext::new().expect("Metal context (set OUT_DIR to dir holding solver.metallib)");
    let stripes = ((32 / nb).max(1)) as u32;
    let mut gpu = BucketedNativeGpu::new(&ctx, &tree, game.table(), &bk, &cpu, stripes).expect("native gpu");

    let nh = game.table().num_valid;
    let hand_class: Vec<usize> = (0..nh)
        .map(|h| {
            let c1 = game.table().hand_cards[h * 2];
            let c2 = game.table().hand_cards[h * 2 + 1];
            PreflopClass::from_combo(c1 as Card, c2 as Card).index()
        })
        .collect();

    println!(
        "DCFR CO-SOLVE [GPU BucketedNativeGpu] (live-{live} multiway, flop idx={}, play/fold preflop, real cap-3, nb={nb})\n\
         WARM | K={k} GPU iters/refresh | runout {nt}×{nr} | tol={tol} | wall=nb^{} = {}\n",
        representative_flop_idx(),
        np - 1,
        (nb as u64).pow((np - 1) as u32)
    );

    let mut rc = vec![0.0f32; NUM_PREFLOP_CLASSES];
    let mut rf = vec![0.0f32; NUM_PREFLOP_CLASSES];
    let mut call_now = vec![1.0f32; NUM_PREFLOP_CLASSES];
    let mut cum_call = vec![0.0f32; NUM_PREFLOP_CLASSES];
    let mut cum_w = 0.0f32;
    let fold_ev = -1.0f32; // SB folds ⇒ loses posted SB blind (1 unit)

    let t0 = Instant::now();
    let mut prev_avg = vec![1.0f32; NUM_PREFLOP_CLASSES];
    let mut n_used = 0usize;
    for refresh in 0..max_refresh {
        // entry range: ALL k players share the symmetric playing range (limped k-way
        // pot; range co-adapts with the postflop). Set it on BOTH the GPU reach buffer
        // (for the warm GPU solve) and the game table (for the normalized CPU extract).
        let mut flat = vec![0.0f32; np * nh];
        for p in 0..np {
            for h in 0..nh {
                let w = call_now[hand_class[h]];
                flat[p * nh + h] = w;
                game.table_mut().initial_weights[p][h] = w;
            }
        }
        gpu.set_initial_weights(&flat);
        gpu.run(k); // warm GPU solve (device regrets/cum persist)

        // NORMALIZED per-hand EV: copy the GPU's converged cum → CPU solver, extract
        // root_cfv_from_avg (does normalize_opponent_reach ⇒ true per-hand EV, NOT the
        // reach-weighted CFV the GPU run() returns). This is the gated extraction path.
        cpu.cum_strategy_flop_mut().copy_from_slice(gpu.cum_strategy_flop());
        cpu.cum_strategy_turn_mut().copy_from_slice(gpu.cum_strategy_turn());
        cpu.cum_strategy_river_mut().copy_from_slice(gpu.cum_strategy_river());
        let cfv = cpu.root_cfv_from_avg(&tree, &game, &bk); // [np][nh] normalized

        let mut sum = vec![0.0f32; NUM_PREFLOP_CLASSES];
        let mut cnt = vec![0u32; NUM_PREFLOP_CLASSES];
        for h in 0..nh {
            sum[hand_class[h]] += cfv[0][h];
            cnt[hand_class[h]] += 1;
        }
        for cl in 0..NUM_PREFLOP_CLASSES {
            if cnt[cl] == 0 {
                continue;
            }
            let ev_call = sum[cl] / cnt[cl] as f32;
            let v = call_now[cl] * ev_call + (1.0 - call_now[cl]) * fold_ev;
            rc[cl] = (rc[cl] + ev_call - v).max(0.0);
            rf[cl] = (rf[cl] + fold_ev - v).max(0.0);
            let s = rc[cl] + rf[cl];
            call_now[cl] = if s > 0.0 { rc[cl] / s } else { 0.5 };
        }
        let wt = (refresh + 1) as f32;
        cum_w += wt;
        for cl in 0..NUM_PREFLOP_CLASSES {
            cum_call[cl] += wt * call_now[cl];
        }
        let avg: Vec<f32> = (0..NUM_PREFLOP_CLASSES).map(|c| cum_call[c] / cum_w).collect();
        let delta = (0..NUM_PREFLOP_CLASSES).map(|c| (avg[c] - prev_avg[c]).abs()).fold(0.0f32, f32::max);
        n_used = refresh + 1;
        if refresh == 0 || (refresh + 1) % 10 == 0 {
            let aa = avg[PreflopClass::from_combo(48, 49).index()];
            println!("  refresh {:>3} | avg-strat Δ {:.5} | AA call {:.4}", refresh + 1, delta, aa);
        }
        if delta < tol && refresh > 3 {
            break;
        }
        prev_avg = avg;
    }
    let secs = t0.elapsed().as_secs_f64();

    let avg: Vec<f32> = (0..NUM_PREFLOP_CLASSES).map(|c| cum_call[c] / cum_w).collect();
    println!(
        "\n⇒ DCFR co-solve PLATEAU at N = {n_used} refreshes (×{k} GPU iters = {} total) | wall {:.2}s | GPU busy {:.2}s",
        n_used as u32 * k,
        secs,
        gpu.gpu_busy_seconds()
    );
    println!("\nCALLING RANGE (avg strategy) — signature classes:");
    let probes: [(&str, u8, u8); 7] = [
        ("AA", 48, 49), ("KK", 44, 45), ("QQ", 40, 41),
        ("AKs", 48, 44), ("87s", 24, 20), ("76o", 20, 17), ("32o", 4, 1),
    ];
    for (name, a, b) in probes {
        let cl = PreflopClass::from_combo(a as Card, b as Card).index();
        println!("  {name:<5} call={:.4}", avg[cl]);
    }
    println!("\nCHEAP-TEST READ: small warm N (few refreshes) ⇒ N×fill modest ⇒ connected-MCCFR's");
    println!("N×fill-avoidance advantage is small (per-traj capped ~2×) ⇒ DCFR co-solve feasible.");
    println!("Large N ⇒ N×fill brutal ⇒ connected MCCFR's case strong, full comparison warranted.");
}

/// CPU side of the MCCFR access-pattern benchmark — IDENTICAL work to the Metal
/// kernel (scattered relaxed-atomic regret read/modify/write over a cache-busting
/// buffer), multi-threaded. Returns M traj/s.
fn cpu_mccfr_bench(ninfo: u32, ntraj: usize, nthreads: usize, regret: &[std::sync::atomic::AtomicU32]) -> f64 {
    use std::sync::atomic::Ordering::Relaxed;
    const DEPTH: usize = 8;
    const NA: usize = 4;
    let per = ntraj / nthreads;
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let regret = &regret;
            s.spawn(move || {
                for j in 0..per {
                    let mut seed = 0x9E3779B97F4A7C15u64.wrapping_mul((t * per + j) as u64 + 1) | 1;
                    let mut path = (t * per + j) as u32;
                    for d in 0..DEPTH {
                        let info = (path.wrapping_mul(2654435761).wrapping_add(d as u32 * 40503)) % ninfo;
                        let base = info as usize * NA;
                        let mut sum = 0.0f32;
                        let mut r = [0.0f32; NA];
                        for a in 0..NA {
                            let ra = f32::from_bits(regret[base + a].load(Relaxed)).max(0.0);
                            r[a] = ra; sum += ra;
                        }
                        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                        let act = (seed as usize) % NA;
                        path = path.wrapping_mul(NA as u32).wrapping_add(act as u32);
                        let _ = if sum > 0.0 { r[act] / sum } else { 0.0 };
                        for a in 0..NA {
                            let delta = if a == act { 1.0f32 } else { -0.3333 };
                            let cur = f32::from_bits(regret[base + a].load(Relaxed));
                            regret[base + a].store((cur + delta).to_bits(), Relaxed); // racy (Hogwild)
                        }
                    }
                }
            });
        }
    });
    (per * nthreads) as f64 / t0.elapsed().as_secs_f64() / 1e6
}

#[cfg(feature = "metal")]
fn gpu_mccfr_bench() {
    use metal::{MTLResourceOptions, MTLSize};
    use solver_core::gpu_metal::context::MetalContext;
    let ninfo: u32 = std::env::var("MC_GPU_NINFO").ok().and_then(|s| s.parse().ok()).unwrap_or(4_000_000);
    let ntraj: usize = std::env::var("MC_GPU_NTRAJ").ok().and_then(|s| s.parse().ok()).unwrap_or(16_000_000);
    let reps: usize = std::env::var("MC_GPU_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let ncores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    println!("MCCFR ACCESS-PATTERN BENCH — GPU (Metal) vs CPU, identical work");
    println!("ninfo={ninfo} (regret buf {} MB), {ntraj} traj/dispatch × {reps} reps, DEPTH=8 NA=4\n",
        ninfo as usize * 4 * 4 / 1_000_000);
    let ctx = MetalContext::new().expect("metal ctx");
    let dev = ctx.device();
    let regret_g = dev.new_buffer((ninfo as usize * 4 * 4) as u64, MTLResourceOptions::StorageModeShared);
    let seeds: Vec<u64> = (0..ntraj).map(|i| 0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1) | 1).collect();
    let seeds_g = dev.new_buffer_with_data(seeds.as_ptr() as *const _, (ntraj * 8) as u64, MTLResourceOptions::StorageModeShared);
    let out_g = dev.new_buffer((ntraj * 4) as u64, MTLResourceOptions::StorageModeShared);
    let ninfo_g = dev.new_buffer_with_data(&ninfo as *const u32 as *const _, 4, MTLResourceOptions::StorageModeShared);
    let pipeline = ctx.create_pipeline("mccfr_bench").expect("mccfr_bench pipeline");
    let tg = (pipeline.max_total_threads_per_threadgroup() as usize).min(256);
    let dispatch = |reps: usize| {
        let t0 = Instant::now();
        for _ in 0..reps {
            let cmd = ctx.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&regret_g), 0);
            enc.set_buffer(1, Some(&ninfo_g), 0);
            enc.set_buffer(2, Some(&seeds_g), 0);
            enc.set_buffer(3, Some(&out_g), 0);
            enc.dispatch_threads(MTLSize::new(ntraj as u64, 1, 1), MTLSize::new(tg as u64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t0.elapsed().as_secs_f64()
    };
    dispatch(2); // warm
    let gpu_s = dispatch(reps);
    let gpu_tps = (reps * ntraj) as f64 / gpu_s / 1e6;
    println!("GPU (Metal, M4 Max): {:>8.1} M traj/s", gpu_tps);

    // CPU, identical work, 1 + ncores threads.
    let regret_c: Vec<std::sync::atomic::AtomicU32> = (0..ninfo as usize * 4).map(|_| std::sync::atomic::AtomicU32::new(0)).collect();
    let cpu1 = cpu_mccfr_bench(ninfo, ntraj / 8, 1, &regret_c);
    let cpun = cpu_mccfr_bench(ninfo, ntraj, ncores, &regret_c);
    println!("CPU 1 thread:        {:>8.1} M traj/s", cpu1);
    println!("CPU {ncores} threads:       {:>8.1} M traj/s", cpun);
    println!("\nGPU vs CPU-{ncores}thread: {:.2}×   |   GPU vs CPU-1thread: {:.1}×", gpu_tps / cpun, gpu_tps / cpu1);
    println!("(bandwidth argues FOR the GPU; divergence + atomics argue AGAINST — this settles it for the access pattern.)");
}

#[cfg(not(feature = "metal"))]
fn gpu_mccfr_bench() {
    eprintln!("MC_GPU requires --features metal (build: cargo build --release -p solver-core --features metal --bin mccfr_probe)");
}

/// CPU reference for the showdown DP — IDENTICAL to `Mccfr::terminal`'s DP and the
/// Metal `showdown_dp` kernel. Used to validate the GPU port bit-against the CPU.
fn cpu_sd_dp(
    bt: usize, opp: &[u32], na: usize,
    tbl: &solver_core::solver::bucketed_showdown::BucketedRunoutTables,
    nb: usize, np: usize, half_pot: f32, net_pot: f32,
) -> f32 {
    let mut state = vec![0.0f32; np + 2];
    state[1] = 1.0;
    for &boo in opp.iter().take(na) {
        let idx = bt * nb + boo as usize;
        let fn_ = tbl.f_n[idx];
        let norm = if fn_ > 0.0 { fn_ } else { 1.0 };
        let (pw, pt, pl) = (tbl.f_w[idx] / norm, tbl.f_t[idx] / norm, tbl.f_l[idx] / norm);
        let mut ns = vec![0.0f32; np + 2];
        if state[0] != 0.0 { ns[0] += state[0]; }
        for j in 0..np {
            let s = state[1 + j];
            if s == 0.0 { continue; }
            ns[0] += s * pl;
            ns[1 + j + 1] += s * pt;
            ns[1 + j] += s * pw;
        }
        state = ns;
    }
    let mut value = state[0] * (-half_pot);
    for j in 0..np {
        let s = state[1 + j];
        if s == 0.0 { continue; }
        value += s * (net_pot / (j + 1) as f32 - half_pot);
    }
    value
}

#[cfg(feature = "metal")]
fn gpu_showdown_validate() {
    use metal::{MTLResourceOptions, MTLSize};
    use solver_core::gpu_metal::context::MetalContext;
    let np: usize = std::env::var("MC_NP").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let nscen: usize = std::env::var("MC_SCEN").ok().and_then(|s| s.parse().ok()).unwrap_or(2_000_000);
    if std::env::var("MC_FLOP").is_err() {
        std::env::set_var("MC_FLOP", representative_flop_idx().to_string());
    }
    println!("PHASE 1 / step 1 — SHOWDOWN DP: GPU (Metal) vs CPU bit-validation (np={np}, nb={nb}, {nscen} scenarios)");
    let g = build_shrunk_cell(np as u8, 2, 2 * np as i32, nb, 1, 1);
    let m = Mccfr::new(&g);
    let tbl = &m.river_tabs[0][0];
    let tnb = tbl.nb;
    // random terminal scenarios.
    let mut rng = 0x1234567u64;
    let mut nx = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let mut bt = vec![0u32; nscen];
    let mut na = vec![0u32; nscen];
    let mut opp = vec![0u32; nscen * 6];
    let mut half = vec![0.0f32; nscen];
    let mut net = vec![0.0f32; nscen];
    for i in 0..nscen {
        bt[i] = (nx() as usize % tnb) as u32;
        let k = 1 + (nx() as usize % (np - 1).max(1));
        na[i] = k as u32;
        for j in 0..k { opp[i * 6 + j] = (nx() as usize % tnb) as u32; }
        half[i] = (nx() as usize % 20) as f32 + 2.0;
        net[i] = half[i] * np as f32 * 1.5;
    }
    let t_cpu = Instant::now();
    let cpu: Vec<f32> = (0..nscen)
        .map(|i| cpu_sd_dp(bt[i] as usize, &opp[i * 6..i * 6 + 6], na[i] as usize, tbl, tnb, np, half[i], net[i]))
        .collect();
    let cpu_s = t_cpu.elapsed().as_secs_f64();

    let ctx = MetalContext::new().expect("metal ctx");
    let dev = ctx.device();
    let upf = |d: &[f32]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len() * 4).max(4) as u64, MTLResourceOptions::StorageModeShared);
    let upu = |d: &[u32]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len() * 4).max(4) as u64, MTLResourceOptions::StorageModeShared);
    let (fw, ft, fl, fn_) = (upf(&tbl.f_w), upf(&tbl.f_t), upf(&tbl.f_l), upf(&tbl.f_n));
    let nb_b = upu(&[tnb as u32]);
    let np_b = upu(&[np as u32]);
    let (bt_b, na_b, opp_b) = (upu(&bt), upu(&na), upu(&opp));
    let (half_b, net_b) = (upf(&half), upf(&net));
    let out = dev.new_buffer((nscen * 4) as u64, MTLResourceOptions::StorageModeShared);
    let pipeline = ctx.create_pipeline("showdown_dp").expect("showdown_dp pipeline");
    let tg = (pipeline.max_total_threads_per_threadgroup() as usize).min(256);
    let t_gpu = Instant::now();
    let cmd = ctx.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    for (i, b) in [&fw, &ft, &fl, &fn_, &nb_b, &np_b, &bt_b, &na_b, &opp_b, &half_b, &net_b, &out].iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.dispatch_threads(MTLSize::new(nscen as u64, 1, 1), MTLSize::new(tg as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let gpu_s = t_gpu.elapsed().as_secs_f64();
    let gpu: &[f32] = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, nscen) };

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut sum_abs = 0.0f64;
    for i in 0..nscen {
        let d = (gpu[i] - cpu[i]).abs();
        max_abs = max_abs.max(d);
        let denom = cpu[i].abs().max(1e-3);
        max_rel = max_rel.max(d / denom);
        sum_abs += d as f64;
    }
    println!("  max |GPU-CPU| = {max_abs:.3e}   max rel = {max_rel:.3e}   mean |Δ| = {:.3e}", sum_abs / nscen as f64);
    println!("  CPU {:.0}M scen/s | GPU {:.0}M scen/s ({:.1}× )", nscen as f64 / cpu_s / 1e6, nscen as f64 / gpu_s / 1e6, cpu_s / gpu_s);
    let pass = max_abs < 1e-2; // float-order tolerance
    println!("  GATE: {} (showdown DP ported correctly)", if pass { "PASS ✓" } else { "FAIL ✗" });
}

#[cfg(not(feature = "metal"))]
fn gpu_showdown_validate() {
    eprintln!("MC_GPU_SD requires --features metal");
}

#[cfg(feature = "metal")]
fn gpu_cell_solve() {
    use metal::{MTLResourceOptions, MTLSize};
    use solver_core::gpu_metal::context::MetalContext;
    use solver_core::tree::action::BoardState;
    let np: usize = std::env::var("MC_NP").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let batch: usize = std::env::var("MC_B").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let nt_r: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let nr_r: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    if std::env::var("MC_FLOP").is_err() { std::env::set_var("MC_FLOP", representative_flop_idx().to_string()); }
    println!("PHASE 1 / steps 2-3 — SINGLE-CELL GPU MCCFR (np={np}, nb={nb}, {nt_r}×{nr_r} SAMPLED runout, batch={batch})");
    let g = build_shrunk_cell(np as u8, 2, 2 * np as i32, nb, nt_r, nr_r);
    let mut m = Mccfr::new(&g);
    let tree = g.tree.clone();
    let nn = tree.num_nodes();
    let maxna = m.max_na; let mnb = m.nb;
    // marshal tree
    let (mut nt, mut npl, mut nbs) = (vec![0u8; nn], vec![0u8; nn], vec![0u8; nn]);
    let (mut nch, mut nchs) = (vec![0u16; nn], vec![0u32; nn]);
    let mut children: Vec<u32> = Vec::new();
    let (mut nloc, mut nfold) = (vec![-1i32; nn], vec![0u16; nn]);
    let mut ncon = vec![0i32; nn * np];
    for i in 0..nn {
        let node = &tree.nodes[i];
        nt[i] = node.node_type; npl[i] = node.player_id; nbs[i] = node.board_state;
        nch[i] = node.num_children; nchs[i] = children.len() as u32;
        for &c in tree.node_children(i) { children.push(c); }
        nloc[i] = m.node_local[i];
        nfold[i] = tree.get_folded_mask(i);
        for p in 0..np { ncon[i * np + p] = tree.get_contribution(i, p as u8); }
    }
    let flop_b = g.bk.flop_map.clone();
    let nh = m.nh;
    // Flatten the nt×nr runout grid (Pluribus samples the board per trajectory; the
    // kernel picks (ti,ri) per thread and indexes these). turn maps/alive per ti;
    // river maps/alive + showdown tables per (ti,ri).
    let mut turn_b: Vec<u16> = Vec::with_capacity(nt_r * nh);
    let mut t_alive: Vec<u8> = Vec::with_capacity(nt_r * nh);
    for ti in 0..nt_r {
        turn_b.extend_from_slice(&g.bk.turn_map[ti]);
        m.cur_ti = ti; m.cur_ri = 0;
        for h in 0..nh { t_alive.push(m.alive(h, BoardState::Turn as u8) as u8); }
    }
    let mut river_b: Vec<u16> = Vec::with_capacity(nt_r * nr_r * nh);
    let mut r_alive: Vec<u8> = Vec::with_capacity(nt_r * nr_r * nh);
    let (mut ft, mut fl, mut fn_): (Vec<f32>, Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new());
    let mut strengths: Vec<i32> = Vec::with_capacity(nt_r * nr_r * nh); // EXACT showdown
    for ti in 0..nt_r {
        for ri in 0..nr_r {
            river_b.extend_from_slice(&g.bk.river_map[ti][ri]);
            m.cur_ti = ti; m.cur_ri = ri;
            for h in 0..nh { r_alive.push(m.alive(h, BoardState::River as u8) as u8); }
            strengths.extend_from_slice(&m.river_strength[ti][ri]); // [run*nh + hand]
            let tbl = &m.river_tabs[ti][ri];
            ft.extend_from_slice(&tbl.f_t); fl.extend_from_slice(&tbl.f_l); fn_.extend_from_slice(&tbl.f_n);
        }
    }
    let arr_len = m.n_info * mnb * maxna;

    #[repr(C)]
    struct CellParams { np: u32, nb: u32, maxna: u32, starting_pot: i32, rake_rate: f32, rake_cap: f32, nt: u32, nr: u32, nh: u32, prune_c: f32, prune_active: u32 }
    let prune_c: f32 = std::env::var("MC_PRUNE_C").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let prune_active: u32 = if std::env::var("MC_PRUNE").is_ok() { 1 } else { 0 };
    let params = CellParams { np: np as u32, nb: mnb as u32, maxna: maxna as u32, starting_pot: tree.starting_pot, rake_rate: tree.rake_rate as f32, rake_cap: tree.rake_cap as f32, nt: nt_r as u32, nr: nr_r as u32, nh: nh as u32, prune_c, prune_active };

    let ctx = MetalContext::new().expect("metal ctx");
    let dev = ctx.device();
    let so = MTLResourceOptions::StorageModeShared;
    let upu8 = |d: &[u8]| dev.new_buffer_with_data(d.as_ptr() as *const _, d.len().max(1) as u64, so);
    let upu16 = |d: &[u16]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len() * 2).max(2) as u64, so);
    let upu32 = |d: &[u32]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len() * 4).max(4) as u64, so);
    let upi32 = |d: &[i32]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len() * 4).max(4) as u64, so);
    let upf = |d: &[f32]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len() * 4).max(4) as u64, so);
    let (b_nt, b_npl, b_nbs) = (upu8(&nt), upu8(&npl), upu8(&nbs));
    let (b_nch, b_nchs, b_ch) = (upu16(&nch), upu32(&nchs), upu32(&children));
    let (b_nloc, b_nfold, b_ncon) = (upi32(&nloc), upu16(&nfold), upi32(&ncon));
    let b_par = dev.new_buffer_with_data(&params as *const _ as *const _, std::mem::size_of::<CellParams>() as u64, so);
    let (b_fb, b_tb, b_rb) = (upu16(&flop_b), upu16(&turn_b), upu16(&river_b));
    let (b_ta, b_ra) = (upu8(&t_alive), upu8(&r_alive));
    let b_strength = upi32(&strengths);
    let (b_ft, b_fl, b_fn) = (upf(&ft), upf(&fl), upf(&fn_));
    let b_reg = dev.new_buffer((arr_len * 4).max(4) as u64, so);
    let b_cum = dev.new_buffer((arr_len * 4).max(4) as u64, so);
    let b_root = dev.new_buffer((batch * 4) as u64, so); // root values (unused in Phase 1)
    let pipeline = ctx.create_pipeline("mccfr_cell").expect("mccfr_cell pipeline");
    let tg = (pipeline.max_total_threads_per_threadgroup() as usize).min(256);
    // Linear/Discounted-CFR: every MC_DISCOUNT iters scale regret+cum by k/(k+1) (0=off).
    let disc_int: u64 = std::env::var("MC_DISCOUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let disc_pipe = ctx.create_pipeline("discount_buf").expect("discount_buf pipeline");
    let mut disc_k = 0u64;

    println!("  nodes={nn} infosets={} nb={mnb} maxna={maxna} regret_len={arr_len} discount={disc_int}", m.n_info);
    println!("{:>12} {:>14} {:>10}", "traj", "GPU maxR/T", "M traj/s");
    let mut total: u64 = 0;
    let mut run_s = 0.0f64;
    for &target in &[batch as u64, 4 * batch as u64, 16 * batch as u64, 64 * batch as u64] {
        while total < target {
            // deal a batch on CPU (distinct cards) + seeds
            let mut hands = vec![0u32; batch * np];
            let mut seeds = vec![0u64; batch];
            for b in 0..batch {
                let mut rng = 0x9E3779B97F4A7C15u64.wrapping_mul(total + b as u64 + 1) | 1;
                let mut nx = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
                seeds[b] = nx() | 1;
                let mut used = 0u64; let mut k = 0;
                while k < np {
                    let h = m.valid[(nx() as usize) % m.valid.len()];
                    let (c1, c2) = m.hand_cards[h];
                    let mk = (1u64 << c1) | (1u64 << c2);
                    if used & mk != 0 { continue; }
                    used |= mk; hands[b * np + k] = h as u32; k += 1;
                }
            }
            let b_h = upu32(&hands);
            let b_s = dev.new_buffer_with_data(seeds.as_ptr() as *const _, (batch * 8) as u64, so);
            let t0 = Instant::now();
            let cmd = ctx.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            let bufs: [&metal::BufferRef; 24] = [&b_nt, &b_npl, &b_nbs, &b_nch, &b_nchs, &b_ch, &b_nloc, &b_nfold, &b_ncon, &b_par, &b_fb, &b_tb, &b_rb, &b_ta, &b_ra, &b_strength, &b_ft, &b_fl, &b_fn, &b_reg, &b_cum, &b_h, &b_s, &b_root];
            for (i, b) in bufs.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
            enc.dispatch_threads(MTLSize::new(batch as u64, 1, 1), MTLSize::new(tg as u64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            run_s += t0.elapsed().as_secs_f64();
            total += batch as u64;
            // periodic Linear-CFR discount of regret+cum by k/(k+1)
            if disc_int > 0 && total / disc_int > disc_k {
                disc_k = total / disc_int;
                let d = disc_k as f32 / (disc_k as f32 + 1.0);
                let n = arr_len as u32;
                let cmd = ctx.queue().new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&disc_pipe);
                enc.set_buffer(0, Some(&b_reg), 0); enc.set_buffer(1, Some(&b_cum), 0);
                enc.set_bytes(2, 4, &d as *const f32 as *const _);
                enc.set_bytes(3, 4, &n as *const u32 as *const _);
                let dtg = (disc_pipe.max_total_threads_per_threadgroup() as usize).min(256);
                enc.dispatch_threads(MTLSize::new(arr_len as u64, 1, 1), MTLSize::new(dtg as u64, 1, 1));
                enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
            }
        }
        let rg: &[f32] = unsafe { std::slice::from_raw_parts(b_reg.contents() as *const f32, arr_len) };
        let maxr = rg.iter().cloned().fold(0.0f32, f32::max);
        println!("{:>12} {:>14.4e} {:>10.1}", total, maxr / total as f32, total as f64 / run_s / 1e6);
    }
    // STEP 4 GATE. Multiplayer CFR has NON-UNIQUE equilibria, so raw strategy-match
    // is the wrong test — instead compare GPU-vs-CPU L1 against the CPU-vs-CPU L1
    // BASELINE (two CPU runs, different seeds, reaching different valid equilibria).
    // GPU L1 ≈ CPU-CPU L1 ⇒ the GPU is as-correct-as-the-CPU (same non-uniqueness).
    let cpu_target = (total / 4).min(16_000_000);
    let run_cpu = |seed: u64| -> Vec<f32> {
        let mut mc = Mccfr::new(&g);
        mc.cur_ti = 0; mc.cur_ri = 0; mc.rng = seed;
        let mut done = 0u64;
        while done < cpu_target { mc.run_iter(&tree, 4096); done += 4096; }
        mc.cum.clone()
    };
    let cum_a = run_cpu(0x9E3779B97F4A7C15);
    let cum_b = run_cpu(0xD1B54A32D192ED03);
    let gcum: &[f32] = unsafe { std::slice::from_raw_parts(b_cum.contents() as *const f32, arr_len) };
    let l1_of = |x: &[f32], y: &[f32]| -> f64 {
        let (mut l1, mut cnt) = (0.0f64, 0usize);
        for inf in 0..m.n_info {
            for bk in 0..mnb {
                let base = (inf * mnb + bk) * maxna;
                let xs: f32 = (0..maxna).map(|a| x[base + a].max(0.0)).sum();
                let ys: f32 = (0..maxna).map(|a| y[base + a].max(0.0)).sum();
                if xs > 1e-6 && ys > 1e-6 {
                    for a in 0..maxna { l1 += (x[base + a] / xs - y[base + a] / ys).abs() as f64; }
                    cnt += 1;
                }
            }
        }
        l1 / cnt.max(1) as f64
    };
    let cpu_cpu = l1_of(&cum_a, &cum_b);
    let gpu_cpu = l1_of(gcum, &cum_a);
    println!("\nSTEP 4 GATE — average-strategy L1 (np={np} is multiplayer ⇒ non-unique equilibria):");
    println!("  CPU-vs-CPU baseline (2 seeds): {cpu_cpu:.4}");
    println!("  GPU-vs-CPU:                    {gpu_cpu:.4}");
    let ok = gpu_cpu <= cpu_cpu * 1.5 + 0.02;
    println!("  GATE: {}  (GPU L1 ≈ baseline ⇒ GPU as-correct-as-CPU; showdown bit-exact ✓ + maxR/T~1/√T ✓)",
        if ok { "PASS ✓" } else { "FAIL ✗ — real divergence" });

    // ── GOLD-STANDARD (HU only): exact abstract-game EXPLOITABILITY of each solution.
    // The right test under non-uniqueness — L1 can't distinguish bug from valid-different
    // equilibrium, exploitability can. → 0 ⇒ converged to the abstract Nash.
    if np == 2 {
        let mut mg = Mccfr::new(&g); mg.cum.copy_from_slice(gcum);
        let mut mc = Mccfr::new(&g); mc.cum.copy_from_slice(&cum_a);
        // BR ANCHOR: a uniform strategy's exploitability. Any converged solve MUST be
        // well below this; if the solve reads ≥ uniform, the BR or engine is broken.
        let mut mu = Mccfr::new(&g);
        for x in mu.cum.iter_mut() { *x = 1.0; }
        let eu = mu.hu_exploitability(&tree);
        let eg = mg.hu_exploitability(&tree);
        let ec = mc.hu_exploitability(&tree);
        println!("\n  [BR anchor] UNIFORM-strategy exploitability = {:.5} chips ({:.2} bb/100)", eu, eu / 2.0 * 100.0);
        // 1 BB = 2 chips (limped cell); report bb and bb/100.
        println!("\nHU EXPLOITABILITY (exact abstract-game BR; →0 = abstract Nash):");
        println!("  GPU:  {:.5} chips  = {:.5} bb  = {:.2} bb/100", eg, eg / 2.0, eg / 2.0 * 100.0);
        println!("  CPU:  {:.5} chips  = {:.5} bb  = {:.2} bb/100", ec, ec / 2.0, ec / 2.0 * 100.0);
        println!("  (MC_IDENTITY ⇒ exact game: a converged solve should read ≈0 — BR sanity check)");
    }
}

#[cfg(not(feature = "metal"))]
fn gpu_cell_solve() { eprintln!("MC_GPU_CELL requires --features metal"); }

#[cfg(feature = "metal")]
fn gpu_conn_solve() {
    use metal::{MTLResourceOptions, MTLSize};
    use solver_core::gpu_metal::context::MetalContext;
    use solver_core::abstraction::preflop_class::PreflopClass;
    use solver_core::card::Card;
    let np: usize = std::env::var("MC_NP").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let batch: usize = std::env::var("MC_B").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    // Full-fidelity runout by default (49 turns × 48 rivers for a fixed flop) — Pluribus
    // samples the full board support. 1×1 is a clairvoyant game (literally-wrong strats).
    let nt_r: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(49);
    let nr_r: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(48);
    if std::env::var("MC_FLOP").is_err() { std::env::set_var("MC_FLOP", representative_flop_idx().to_string()); }
    let nraises: usize = std::env::var("MC_PRERAISES").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    // preflop tree: fold/call (+ optional pot-relative raises, cap-3) — mirror ConnectedMWA.
    // SHARED with the runtime loader (ConnBlueprint) so the blueprint's preflop
    // node indexing can never drift between solve and serve.
    let (pft, pre_local, pre_ninfo, pre_maxna) =
        solver_core::blueprint::build_conn_preflop_tree(np, nraises);
    let pnn = pft.num_nodes();
    // SEAM CELLS keyed by (live, SPR-bin) — NOT exact (commit, pot). The rich-preflop
    // exact key explodes to ~9366 cells ⇒ 48.6B floats/flop, 11× over the u32 offset
    // ceiling (this corrupted blueprint_conn_v2). SPR-binning collapses it ~230× while
    // keeping the rich sizing: postflop play is ~SPR-determined, so seams with the same
    // (live, SPR-bin) share one cell built at the FIRST-seen representative (commit,pot).
    let stack = production_game_v1().stack;
    // COARSE Phase-A cells (god-tier preflop): collapse all SPR bins per live to ONE cell
    // (bin 0), built at the first-seen (commit,pot) rep. Shrinks post_stride ~600× so the
    // shared preflop can train over a LARGE orbit-weighted MC_NF flop set within the u32
    // offset + GPU-memory ceilings. Phase B re-solves each flop's postflop at FULL v4
    // fidelity against the frozen preflop, so the lean Phase-A postflop only has to be good
    // enough to converge the preflop — not to ship. (Off ⇒ the rich v4 cell set.)
    let conn_coarse = std::env::var("MC_CONN_COARSE").is_ok();
    let mut keys: Vec<(u8, i32, i32)> = Vec::new();
    let mut key_idx: std::collections::HashMap<(u8, i64), usize> = std::collections::HashMap::new();
    for i in 0..pnn {
        if pft.nodes[i].is_chance() {
            let mut bk = solver_core::blueprint::conn_seam_bin(&pft, i, np, stack);
            if conn_coarse { bk.1 = 0; }
            if bk.0 >= 2 {
                key_idx.entry(bk).or_insert_with(|| {
                    keys.push(solver_core::blueprint::conn_seam_rep(&pft, i, np));
                    keys.len() - 1
                });
            }
        }
    }
    let mut cells: Vec<Mccfr> = Vec::new();
    let mut cell_trees: Vec<FlatTree> = Vec::new();
    eprintln!("[SETUP] building {} seam cells (np={np}, preraises={nraises})…", keys.len());
    let t_cells = Instant::now();
    for &(live, commit, pot) in &keys {
        let g = build_shrunk_cell(live, commit, pot, nb, 1, 1);
        cells.push(Mccfr::new(&g));
        cell_trees.push(g.tree.clone());
    }
    eprintln!("[SETUP] {} seam cells built in {:.1}s", keys.len(), t_cells.elapsed().as_secs_f64());
    let mnb = cells[0].nb;
    let cell_maxna = cells.iter().map(|c| c.max_na).max().unwrap_or(2);
    let maxna = pre_maxna.max(cell_maxna);
    assert!(maxna <= 8, "maxna {maxna} > CL_MAXNA=8 in kernel — raise CL_MAXNA");
    println!("PHASE 2A — CONNECTED GPU MCCFR + RAISES (np={np}, nb={nb}, preraises={nraises}, {} seam cells, maxna={maxna}, {nt_r}×{nr_r} runout)", keys.len());
    // ALL-FLOPS: NF flops (MC_NF; default 1 = single MC_FLOP). Maps/strengths/alive/
    // hand_cls are PER-FLOP (fi-major); preflop region SHARED; postflop regret ×NF.
    let nf_flops: usize = std::env::var("MC_NF").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let flop_indices: Vec<usize> = if nf_flops > 1 {
        let step = (1755 / nf_flops).max(1);
        (0..nf_flops).map(|i| (i * step) % 1755).collect()
    } else {
        vec![std::env::var("MC_FLOP").ok().and_then(|s| s.parse().ok()).unwrap_or(representative_flop_idx())]
    };
    // ORBIT-WEIGHTED flop frequencies (god-tier preflop): a canonical flop stands for
    // `orbit_of().len()` actual flops (×4 monotone / ×12 two-tone / ×24 rainbow). UNIFORM-
    // over-canonical sampling over-weights rare suit patterns and biases the SHARED preflop;
    // sampling each selected flop ∝ its orbit size trains the preflop against the TRUE flop
    // distribution. NF=1 collapses to weight 1. The CDF gives O(log NF) host-side sampling.
    let flop_w: Vec<f64> = {
        use solver_core::abstraction::flop_isomorphism::{enumerate_canonical_flops, orbit_of};
        let canon = enumerate_canonical_flops();
        flop_indices.iter().map(|&fi| orbit_of(canon[fi]).len() as f64).collect()
    };
    let flop_cdf: Vec<f64> = {
        let tot: f64 = flop_w.iter().sum();
        let mut c = 0.0;
        flop_w.iter().map(|&w| { c += w / tot; c }).collect()
    };
    let mut hand_cls: Vec<u32> = Vec::new();
    let mut flop_b: Vec<u16> = Vec::new();
    let mut turn_b: Vec<u16> = Vec::new();
    let mut t_alive: Vec<u8> = Vec::new();
    let mut river_b: Vec<u16> = Vec::new();
    let mut r_alive: Vec<u8> = Vec::new();
    let mut strengths: Vec<i32> = Vec::new();
    let (mut ft, mut fl, mut fn_): (Vec<f32>, Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new());
    let mut valid_f: Vec<Vec<usize>> = Vec::new();   // per-flop valid hands (for dealing)
    let mut hcards_f: Vec<Vec<(u8, u8)>> = Vec::new();
    let (mut nh, mut tnb) = (0usize, 0usize);
    let t_maps = Instant::now();
    for (fii, &flop) in flop_indices.iter().enumerate() {
        std::env::set_var("MC_FLOP", flop.to_string());
        let g2 = build_shrunk_cell(2, 2, 4, nb, nt_r, nr_r);
        let mut mref = Mccfr::new(&g2);
        nh = mref.nh; tnb = mref.river_tabs[0][0].nb;
        for h in 0..nh { let (c1, c2) = mref.hand_cards[h]; hand_cls.push(PreflopClass::from_combo(c1 as Card, c2 as Card).index() as u32); }
        flop_b.extend_from_slice(&g2.bk.flop_map);
        for ti in 0..nt_r {
            turn_b.extend_from_slice(&g2.bk.turn_map[ti]);
            mref.cur_ti = ti; mref.cur_ri = 0;
            for h in 0..nh { t_alive.push(mref.alive(h, 1) as u8); }
        }
        for ti in 0..nt_r {
            for ri in 0..nr_r {
                river_b.extend_from_slice(&g2.bk.river_map[ti][ri]);
                mref.cur_ti = ti; mref.cur_ri = ri;
                for h in 0..nh { r_alive.push(mref.alive(h, 2) as u8); }
                strengths.extend_from_slice(&mref.river_strength[ti][ri]); // [(fi,run)*nh + hand]
                if fii == 0 { let tbl = &mref.river_tabs[ti][ri]; ft.extend_from_slice(&tbl.f_t); fl.extend_from_slice(&tbl.f_l); fn_.extend_from_slice(&tbl.f_n); }
            }
        }
        valid_f.push(mref.valid.clone());
        hcards_f.push(mref.hand_cards.clone());
    }
    eprintln!("[SETUP] {} flops' maps built in {:.1}s ({:.2}s/flop)", flop_indices.len(), t_maps.elapsed().as_secs_f64(), t_maps.elapsed().as_secs_f64() / flop_indices.len() as f64);
    if std::env::var("MC_SETUP_ONLY").is_ok() { return; }

    // ── build combined node array: preflop [0..pnn) then each keyed cell appended ──
    let nn_total = pnn + cell_trees.iter().map(|t| t.num_nodes()).sum::<usize>();
    let (mut c_type, mut c_pre, mut c_seam) = (vec![0u8; nn_total], vec![0u8; nn_total], vec![0u8; nn_total]);
    let (mut c_player, mut c_bs) = (vec![0u8; nn_total], vec![0u8; nn_total]);
    let (mut c_nch, mut c_chstart) = (vec![0u16; nn_total], vec![0u32; nn_total]);
    let mut children: Vec<u32> = Vec::new();
    let (mut c_local, mut c_fold) = (vec![-1i32; nn_total], vec![0u16; nn_total]);
    let mut c_contrib = vec![0i32; nn_total * np];
    let mut c_spot = vec![0i32; nn_total];
    let (mut c_regbase, mut c_nb) = (vec![0u32; nn_total], vec![0u32; nn_total]);
    // regret regions: preflop first, then per-cell (by key index).
    let pre_region = pre_ninfo * NUM_PREFLOP_CLASSES * maxna;
    let mut cell_node_base = vec![0usize; keys.len()];
    let mut cell_reg_base = vec![0u32; keys.len()];
    // Accumulate offsets in USIZE (not u32) so the guard below can SEE an overflow.
    // The regret buffer is indexed by u32 offsets in the kernel; if the true region
    // exceeds the u32 ceiling the offsets silently wrap and cells alias onto each
    // other (and the preflop) → corrupt solve. This bit the first rich-na=8 attempt
    // (9366 exact-(commit,pot) cells ⇒ ~48.6B floats/flop, 11× over u32). Fix is
    // SPR-bin cells; until then, FAIL LOUDLY rather than write garbage.
    let mut node_off = pnn;
    let mut reg_off: usize = pre_region;
    for ki in 0..keys.len() {
        cell_node_base[ki] = node_off;
        cell_reg_base[ki] = reg_off as u32;
        node_off += cell_trees[ki].num_nodes();
        reg_off += cells[ki].n_info * mnb * maxna;
    }
    let reg_total = reg_off;
    assert!(
        reg_total <= u32::MAX as usize,
        "postflop regret region {reg_total} floats EXCEEDS the u32 offset ceiling {} \
         ({} seam cells × nb={mnb} × maxna={maxna}) — exact-(commit,pot) cell explosion; \
         the u32 kernel offsets would wrap + alias. Need SPR-bin cells (collapse cells) \
         or u64 offsets before solving this abstraction.",
        u32::MAX, keys.len(),
    );
    // preflop nodes
    for i in 0..pnn {
        let n = &pft.nodes[i];
        c_type[i] = n.node_type; c_pre[i] = 1; c_player[i] = n.player_id; c_bs[i] = n.board_state;
        c_local[i] = pre_local[i]; c_fold[i] = pft.get_folded_mask(i);
        for p in 0..np { c_contrib[i * np + p] = pft.get_contribution(i, p as u8); }
        c_spot[i] = pft.starting_pot; c_regbase[i] = 0; c_nb[i] = NUM_PREFLOP_CLASSES as u32;
        if n.is_chance() {
            // SEAM: rewire to the cell for this line's (live, SPR-bin) key.
            let mut bk = solver_core::blueprint::conn_seam_bin(&pft, i, np, stack);
            if conn_coarse { bk.1 = 0; }
            let ki = key_idx[&bk];
            c_seam[i] = 1; c_nch[i] = 1; c_chstart[i] = children.len() as u32;
            children.push(cell_node_base[ki] as u32);
        } else {
            c_nch[i] = n.num_children; c_chstart[i] = children.len() as u32;
            for &ch in pft.node_children(i) { children.push(ch); }
        }
    }
    // cell nodes (per key)
    for ki in 0..keys.len() {
        let live = keys[ki].0 as usize;
        let ct = &cell_trees[ki];
        let cm = &cells[ki];
        let base = cell_node_base[ki];
        for i in 0..ct.num_nodes() {
            let gi = base + i; let n = &ct.nodes[i];
            c_type[gi] = n.node_type; c_pre[gi] = 0; c_player[gi] = n.player_id; c_bs[gi] = n.board_state;
            c_local[gi] = cm.node_local[i]; c_fold[gi] = ct.get_folded_mask(i);
            for p in 0..live { c_contrib[gi * np + p] = ct.get_contribution(i, p as u8); }
            c_spot[gi] = ct.starting_pot; c_regbase[gi] = cell_reg_base[ki]; c_nb[gi] = mnb as u32;
            c_nch[gi] = n.num_children; c_chstart[gi] = children.len() as u32;
            for &ch in ct.node_children(i) { children.push(base as u32 + ch); }
        }
    }

    #[repr(C)] struct ConnParams { np: u32, maxna: u32, rake_rate: f32, rake_cap: f32, nt: u32, nr: u32, nh: u32, nf: u32, post_stride: u32, prune_c: f32, prune_active: u32, freeze_pre: u32 }
    let post_stride = reg_total - pre_region;       // one flop's postflop regret region
    let reg_buf_len = pre_region + nf_flops * post_stride; // shared preflop + NF×postflop
    let prune_c: f32 = std::env::var("MC_PRUNE_C").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let prune_active: u32 = if std::env::var("MC_PRUNE").is_ok() { 1 } else { 0 };
    // FREEZE-PREFLOP (reach-prior per-flop solve): preflop read, never updated.
    let freeze_pre: u32 = if std::env::var("MC_FREEZE_PRE").is_ok() { 1 } else { 0 };
    let params = ConnParams { np: np as u32, maxna: maxna as u32, rake_rate: pft.rake_rate as f32, rake_cap: pft.rake_cap as f32, nt: nt_r as u32, nr: nr_r as u32, nh: nh as u32, nf: nf_flops as u32, post_stride: post_stride as u32, prune_c, prune_active, freeze_pre };

    let ctx = MetalContext::new().expect("metal");
    let dev = ctx.device();
    let so = MTLResourceOptions::StorageModeShared;
    let u8b = |d: &[u8]| dev.new_buffer_with_data(d.as_ptr() as *const _, d.len().max(1) as u64, so);
    let u16b = |d: &[u16]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len()*2).max(2) as u64, so);
    let u32b = |d: &[u32]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len()*4).max(4) as u64, so);
    let i32b = |d: &[i32]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len()*4).max(4) as u64, so);
    let fb = |d: &[f32]| dev.new_buffer_with_data(d.as_ptr() as *const _, (d.len()*4).max(4) as u64, so);
    let bufs_static: Vec<metal::Buffer> = vec![
        u8b(&c_type), u8b(&c_pre), u8b(&c_seam), u8b(&c_player), u8b(&c_bs), u16b(&c_nch), u32b(&c_chstart),
        u32b(&children), i32b(&c_local), u16b(&c_fold), i32b(&c_contrib), i32b(&c_spot), u32b(&c_regbase), u32b(&c_nb),
        dev.new_buffer_with_data(&params as *const _ as *const _, std::mem::size_of::<ConnParams>() as u64, so),
        u16b(&flop_b), u16b(&turn_b), u16b(&river_b), u8b(&t_alive), u8b(&r_alive),
        i32b(&strengths), fb(&ft), fb(&fl), fb(&fn_), u32b(&[tnb as u32]),
    ];
    let b_reg = dev.new_buffer((reg_buf_len*4).max(4) as u64, so);
    let b_cum = dev.new_buffer((reg_buf_len*4).max(4) as u64, so);
    let b_cls = u32b(&hand_cls);
    let pipeline = ctx.create_pipeline("mccfr_conn").expect("mccfr_conn pipeline");
    let tg = (pipeline.max_total_threads_per_threadgroup() as usize).min(256);
    let disc_int: u64 = std::env::var("MC_DISCOUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let disc_pipe = ctx.create_pipeline("discount_buf").expect("discount_buf pipeline");
    let mut disc_k = 0u64;
    println!("  combined nodes={nn_total} (pre {pnn}) regret_len={reg_buf_len} maxna={maxna} cells={} nf={nf_flops} discount={disc_int}", np-1);
    println!("{:>12} {:>14} {:>10}", "traj", "GPU maxR/T", "M traj/s");
    let mut total = 0u64; let mut run_s = 0.0f64;
    for &target in &[batch as u64, 4*batch as u64, 16*batch as u64, 64*batch as u64] {
        while total < target {
            // per trajectory: sample a flop fi, deal np distinct hands from THAT flop's deck.
            let mut hands = vec![0u32; batch*np]; let mut seeds = vec![0u64; batch];
            let mut fis = vec![0u32; batch];
            for b in 0..batch {
                let mut rng = 0x9E3779B97F4A7C15u64.wrapping_mul(total + b as u64 + 1) | 1;
                let mut nx = || { rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; rng };
                seeds[b] = nx() | 1;
                // ORBIT-WEIGHTED flop draw (∝ suit-isomorphism multiplicity) so the shared
                // preflop trains against the true flop distribution, not uniform-over-canonical.
                let u = (nx() >> 11) as f64 / ((1u64 << 53) as f64);
                let fi = flop_cdf.partition_point(|&c| c < u).min(nf_flops - 1);
                fis[b] = fi as u32;
                let (valid, hcards) = (&valid_f[fi], &hcards_f[fi]);
                let mut used = 0u64; let mut k = 0;
                while k < np {
                    let h = valid[(nx() as usize) % valid.len()];
                    let (c1, c2) = hcards[h]; let mk = (1u64<<c1)|(1u64<<c2);
                    if used & mk != 0 { continue; } used |= mk; hands[b*np+k] = h as u32; k += 1;
                }
            }
            let b_h = u32b(&hands);
            let b_fi = u32b(&fis);
            let b_s = dev.new_buffer_with_data(seeds.as_ptr() as *const _, (batch*8) as u64, so);
            // Sub-dispatch: split the deal into chunks of `dispatch_sz`. Threads within a
            // chunk update regrets concurrently against the chunk-start values (stale);
            // regrets DO carry between chunks (shared buffer + atomics). Smaller chunk ⇒
            // fresher regrets ⇒ closer to sequential CFR (trades throughput).
            let dispatch_sz: usize = std::env::var("MC_DISPATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(batch);
            let t0 = Instant::now();
            let mut off = 0usize;
            while off < batch {
                let chunk = dispatch_sz.min(batch - off);
                let cmd = ctx.queue().new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                for (i, b) in bufs_static.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
                enc.set_buffer(25, Some(&b_reg), 0); enc.set_buffer(26, Some(&b_cum), 0);
                enc.set_buffer(27, Some(&b_h), (off*np*4) as u64); enc.set_buffer(28, Some(&b_cls), 0); enc.set_buffer(29, Some(&b_s), (off*8) as u64);
                enc.set_buffer(30, Some(&b_fi), (off*4) as u64);
                enc.dispatch_threads(MTLSize::new(chunk as u64, 1, 1), MTLSize::new(tg as u64, 1, 1));
                enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
                off += chunk;
            }
            run_s += t0.elapsed().as_secs_f64(); total += batch as u64;
            if disc_int > 0 && total / disc_int > disc_k {
                disc_k = total / disc_int;
                let d = disc_k as f32 / (disc_k as f32 + 1.0);
                let n = reg_buf_len as u32;
                let cmd = ctx.queue().new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&disc_pipe);
                enc.set_buffer(0, Some(&b_reg), 0); enc.set_buffer(1, Some(&b_cum), 0);
                enc.set_bytes(2, 4, &d as *const f32 as *const _);
                enc.set_bytes(3, 4, &n as *const u32 as *const _);
                let dtg = (disc_pipe.max_total_threads_per_threadgroup() as usize).min(256);
                enc.dispatch_threads(MTLSize::new(reg_buf_len as u64, 1, 1), MTLSize::new(dtg as u64, 1, 1));
                enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
            }
        }
        let rg: &[f32] = unsafe { std::slice::from_raw_parts(b_reg.contents() as *const f32, reg_buf_len) };
        let maxr = rg.iter().cloned().fold(0.0f32, f32::max);
        println!("{:>12} {:>14.4e} {:>10.1}", total, maxr / total as f32, total as f64 / run_s / 1e6);
        // PHASE-A PREFLOP RANGE (validation): open-node raise/fold% for key hands from the
        // SHARED preflop avg-strategy (b_cum over pre_region). Sanity for a god-tier preflop:
        // AA/KK raise ~100% & never fold, AKs raises heavily, T9s mixes, 72o/32o fold. The
        // NF=1 bias (pairs fold ~0, non-pairs over-fold) shows here as broken non-pair folds.
        {
            use solver_core::abstraction::preflop_class::PreflopClass;
            use solver_core::card::card_from_str;
            let cum: &[f32] = unsafe { std::slice::from_raw_parts(b_cum.contents() as *const f32, pre_region) };
            let nc = NUM_PREFLOP_CLASSES;
            let open = (0..pnn).find(|&i| pre_local[i] >= 0).unwrap_or(0);
            let local = pre_local[open] as usize;
            let na = pft.nodes[open].num_children as usize;
            let off = local * nc * maxna;
            let labels: Vec<u8> = pft.node_children(open).iter().map(|&c| pft.nodes[c as usize].action_label).collect();
            let rf = |cl: usize, want: u8| -> f32 {
                let s: f32 = (0..na).map(|a| cum[off + a*nc + cl].max(0.0)).sum();
                if s <= 0.0 { return 0.0; }
                let r: f32 = (0..na).filter(|&a| labels[a]==want).map(|a| cum[off + a*nc + cl].max(0.0)).sum();
                r / s
            };
            let ix = |a:&str,b:&str| PreflopClass::from_combo(card_from_str(a).unwrap(), card_from_str(b).unwrap()).index();
            let pr = |a:&str,b:&str| (rf(ix(a,b),4)*100.0, rf(ix(a,b),0)*100.0);
            let (aar,aaf)=pr("Ac","Ad"); let (kkr,kkf)=pr("Kc","Kd"); let (aksr,aksf)=pr("Ac","Kc");
            let (t9r,t9f)=pr("Tc","9c"); let (o72r,o72f)=pr("7c","2d"); let (o32r,o32f)=pr("3c","2d");
            println!("  PHASE-A open raise/fold%: AA {aar:.0}/{aaf:.0} | KK {kkr:.0}/{kkf:.0} | AKs {aksr:.0}/{aksf:.0} | T9s {t9r:.0}/{t9f:.0} | 72o {o72r:.0}/{o72f:.0} | 32o {o32r:.0}/{o32f:.0}");
        }
    }

    // ── PER-FLOP BLUEPRINT BUILD (single-pass + reach-prior) ──────────────
    // The solve above is PHASE A: it converged the shared preflop over the
    // representative flop set (MC_NF flops). PHASE B now solves EACH flop's
    // postflop one at a time with the preflop FROZEN (reach-prior), so only one
    // flop's f32 lives at once (the 447GB fully-connected set never forms), and
    // compresses each output via SSBP2. Reuses the flop-INDEPENDENT node/ft/fl/fn
    // buffers; swaps only the ~2MB per-flop maps + an nf=1 regret buffer.
    if std::env::var("MC_BLUEPRINT_BUILD").is_ok() {
        use solver_core::blueprint::ssbp2_encode_cum;
        let out_dir = std::env::var("MC_BP_OUT").unwrap_or_else(|_| "blueprint_conn_out".into());
        std::fs::create_dir_all(&out_dir).expect("bp out dir");
        // Save the converged preflop (regret + avg-strategy) from Phase A.
        let preA_reg: Vec<f32> = unsafe { std::slice::from_raw_parts(b_reg.contents() as *const f32, pre_region) }.to_vec();
        let preA_cum: Vec<f32> = unsafe { std::slice::from_raw_parts(b_cum.contents() as *const f32, pre_region) }.to_vec();
        // Phase-A continuation/leaf prior: write the preflop block too (shared by all flops).
        ssbp2_write(&format!("{out_dir}/preflop.ssbp2"), &preA_cum);
        let ncan = canonical_flops_cached().len();
        let lo: usize = std::env::var("MC_FLOP_LO").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let hi: usize = std::env::var("MC_FLOP_HI").ok().and_then(|s| s.parse().ok()).unwrap_or(ncan).min(ncan);
        // ANYTIME convergence: run each flop until maxR/T (postflop max-regret per
        // iter, ~1/√T) stops improving — i.e. diminishing returns — then call it
        // done. Knobs: MC_BP_ITERS = hard cap; MC_BP_MIN = floor before stopping;
        // MC_BP_CHECK = check interval; MC_BP_EPS = min fractional maxR/T drop per
        // check to count as "still improving" (below it for 2 checks ⇒ plateau ⇒ stop).
        let bp_cap: u64 = std::env::var("MC_BP_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(64_000_000);
        let bp_min: u64 = std::env::var("MC_BP_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(4 * batch as u64);
        let bp_check: u64 = std::env::var("MC_BP_CHECK").ok().and_then(|s| s.parse().ok()).unwrap_or(8 * batch as u64);
        let bp_eps: f32 = std::env::var("MC_BP_EPS").ok().and_then(|s| s.parse().ok()).unwrap_or(0.03);
        // nf=1 params with the preflop frozen.
        let params_b = ConnParams { nf: 1, freeze_pre: 1, post_stride: post_stride as u32, ..params };
        let b_params_b = dev.new_buffer_with_data(&params_b as *const _ as *const _, std::mem::size_of::<ConnParams>() as u64, so);
        let reg1_len = pre_region + post_stride;
        let b_reg1 = dev.new_buffer((reg1_len * 4).max(4) as u64, so);
        let b_cum1 = dev.new_buffer((reg1_len * 4).max(4) as u64, so);
        println!("  PHASE B: per-flop frozen-preflop solve, flops {lo}..{hi}, anytime (cap {bp_cap}, eps {bp_eps}) → {out_dir}/");
        let t_b = Instant::now();
        for flop in lo..hi {
            if std::path::Path::new(&format!("{out_dir}/flop_{flop:04}.ssbp2")).exists() { continue; } // resumable
            // build this flop's maps (subset path → full-fidelity buckets, nt_r×nr_r runout).
            std::env::set_var("MC_FLOP", flop.to_string());
            let g2 = build_shrunk_cell(2, 2, 4, nb, nt_r, nr_r);
            let mut mref = Mccfr::new(&g2);
            let nhf = mref.nh;
            let mut cls = Vec::with_capacity(nhf);
            for h in 0..nhf { let (c1, c2) = mref.hand_cards[h]; cls.push(PreflopClass::from_combo(c1 as Card, c2 as Card).index() as u32); }
            let mut fb = g2.bk.flop_map.clone();
            let (mut tb, mut ta) = (Vec::new(), Vec::new());
            for ti in 0..nt_r { tb.extend_from_slice(&g2.bk.turn_map[ti]); mref.cur_ti = ti; mref.cur_ri = 0; for h in 0..nhf { ta.push(mref.alive(h, 1) as u8); } }
            let (mut rb, mut ra, mut st) = (Vec::new(), Vec::new(), Vec::new());
            for ti in 0..nt_r { for ri in 0..nr_r { rb.extend_from_slice(&g2.bk.river_map[ti][ri]); mref.cur_ti = ti; mref.cur_ri = ri; for h in 0..nhf { ra.push(mref.alive(h, 2) as u8); } st.extend_from_slice(&mref.river_strength[ti][ri]); } }
            fb.shrink_to_fit();
            let (mfb, mtb, mrb, mta, mra, mst, mcls) = (u16b(&fb), u16b(&tb), u16b(&rb), u8b(&ta), u8b(&ra), i32b(&st), u32b(&cls));
            let valid = mref.valid.clone(); let hcards = mref.hand_cards.clone();
            // reset nf=1 buffers: preload frozen preflop, zero postflop.
            {
                let r1 = unsafe { std::slice::from_raw_parts_mut(b_reg1.contents() as *mut f32, reg1_len) };
                let c1 = unsafe { std::slice::from_raw_parts_mut(b_cum1.contents() as *mut f32, reg1_len) };
                r1[..pre_region].copy_from_slice(&preA_reg); for x in &mut r1[pre_region..] { *x = 0.0; }
                c1[..pre_region].copy_from_slice(&preA_cum); for x in &mut c1[pre_region..] { *x = 0.0; }
            }
            // frozen-preflop solve for this flop.
            let mut done = 0u64; let mut disc_k1 = 0u64;
            let mut prev_rt = f32::INFINITY; let mut slow = 0u32; let mut next_check = bp_min.max(bp_check);
            while done < bp_cap {
                let mut hands = vec![0u32; batch * np]; let mut seeds = vec![0u64; batch]; let fis = vec![0u32; batch];
                for b in 0..batch {
                    let mut rng = 0x9E3779B97F4A7C15u64.wrapping_mul(done + b as u64 + 1 + (flop as u64) << 20) | 1;
                    let mut nx = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
                    seeds[b] = nx() | 1;
                    let mut used = 0u64; let mut k = 0;
                    while k < np { let h = valid[(nx() as usize) % valid.len()]; let (c1, c2) = hcards[h]; let mk = (1u64 << c1) | (1u64 << c2); if used & mk != 0 { continue; } used |= mk; hands[b * np + k] = h as u32; k += 1; }
                }
                let b_h = u32b(&hands); let b_fi = u32b(&fis);
                let b_s = dev.new_buffer_with_data(seeds.as_ptr() as *const _, (batch * 8) as u64, so);
                let cmd = ctx.queue().new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                for (i, b) in bufs_static.iter().enumerate() { enc.set_buffer(i as u64, Some(b), 0); }
                enc.set_buffer(14, Some(&b_params_b), 0); // nf=1 + freeze_pre
                enc.set_buffer(15, Some(&mfb), 0); enc.set_buffer(16, Some(&mtb), 0); enc.set_buffer(17, Some(&mrb), 0);
                enc.set_buffer(18, Some(&mta), 0); enc.set_buffer(19, Some(&mra), 0); enc.set_buffer(20, Some(&mst), 0);
                enc.set_buffer(25, Some(&b_reg1), 0); enc.set_buffer(26, Some(&b_cum1), 0);
                enc.set_buffer(27, Some(&b_h), 0); enc.set_buffer(28, Some(&mcls), 0); enc.set_buffer(29, Some(&b_s), 0); enc.set_buffer(30, Some(&b_fi), 0);
                enc.dispatch_threads(MTLSize::new(batch as u64, 1, 1), MTLSize::new(tg as u64, 1, 1));
                enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
                done += batch as u64;
                if disc_int > 0 && done / disc_int > disc_k1 {
                    disc_k1 = done / disc_int; let d = disc_k1 as f32 / (disc_k1 as f32 + 1.0); let n = reg1_len as u32;
                    let cmd = ctx.queue().new_command_buffer(); let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&disc_pipe); enc.set_buffer(0, Some(&b_reg1), 0); enc.set_buffer(1, Some(&b_cum1), 0);
                    enc.set_bytes(2, 4, &d as *const f32 as *const _); enc.set_bytes(3, 4, &n as *const u32 as *const _);
                    let dtg = (disc_pipe.max_total_threads_per_threadgroup() as usize).min(256);
                    enc.dispatch_threads(MTLSize::new(reg1_len as u64, 1, 1), MTLSize::new(dtg as u64, 1, 1));
                    enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
                }
                // ANYTIME plateau check: maxR/T over the POSTFLOP region (scan is cheap
                // vs a check interval). Stop once the fractional drop stays < eps.
                if done >= next_check {
                    let r1 = unsafe { std::slice::from_raw_parts(b_reg1.contents() as *const f32, reg1_len) };
                    let maxr = r1[pre_region..].iter().cloned().fold(0.0f32, f32::max);
                    let rt = maxr / done as f32;
                    let impr = if prev_rt.is_finite() && prev_rt > 0.0 { (prev_rt - rt) / prev_rt } else { 1.0 };
                    if impr < bp_eps { slow += 1; } else { slow = 0; }
                    prev_rt = rt; next_check += bp_check;
                    if slow >= 2 { break; } // converging slow as hell ⇒ done
                }
            }
            // extract this flop's postflop avg-strategy slice → SSBP2.
            let c1 = unsafe { std::slice::from_raw_parts(b_cum1.contents() as *const f32, reg1_len) };
            ssbp2_write(&format!("{out_dir}/flop_{flop:04}.ssbp2"), &c1[pre_region..]);
            if (flop - lo) % 20 == 0 { println!("    flop {flop} done @{done} iters (maxR/T {prev_rt:.3e}, {:.1}s elapsed)", t_b.elapsed().as_secs_f64()); }
        }
        println!("  PHASE B complete: flops {lo}..{hi} in {:.1}s", t_b.elapsed().as_secs_f64());
        return;
    }
    // STEP GATE: preflop call-freq (AA/72o), GPU vs CPU ConnectedMW.
    let gcum: &[f32] = unsafe { std::slice::from_raw_parts(b_cum.contents() as *const f32, reg_buf_len) };
    // find the fold/call decision node + fold action in pft.
    let mut node = usize::MAX; let mut fold_a = 0usize;
    'outer: for i in 0..pnn {
        if !pft.nodes[i].is_player() { continue; }
        let actor = pft.nodes[i].player_id;
        for (a, &k) in pft.node_children(i).iter().enumerate() {
            if pft.nodes[k as usize].is_terminal() && (pft.get_folded_mask(k as usize) >> actor) & 1 == 1 { node = i; fold_a = a; break 'outer; }
        }
    }
    let gpu_callfreq = |cls: usize| -> f32 {
        let local = pre_local[node] as usize;
        let na = pft.nodes[node].num_children as usize;
        let base = (local * NUM_PREFLOP_CLASSES + cls) * maxna;
        let sum: f32 = (0..na).map(|a| gcum[base + a].max(0.0)).sum();
        if sum <= 0.0 { -1.0 } else { 1.0 - gcum[base + fold_a] / sum }
    };
    // GATE: GPU preflop NON-FOLD freq per class vs CPU ConnectedMWA (same MC_PRERAISES),
    // two CPU seeds for the non-uniqueness baseline. GPU as-correct-as-CPU iff
    // GPU-vs-CPU |Δ| ≈ the CPU-vs-CPU baseline.
    let cpu_target = total.min(16_000_000);
    let probes: [(&str, u8, u8); 5] = [("AA", 48, 49), ("KK", 44, 45), ("AKs", 48, 44), ("87s", 24, 20), ("72o", 20, 0)];
    let classes: Vec<usize> = probes.iter().map(|&(_, a, b)| PreflopClass::from_combo(a as Card, b as Card).index()).collect();
    // CPU reference: ALL-FLOPS ConnectedMWF when nf>1 (fold/call only ⇒ use preraises=0),
    // else ConnectedMWA. Returns per-class non-fold freq.
    let cpu_freqs = |seed: u64| -> Vec<f32> {
        if nf_flops > 1 {
            let mut c = ConnectedMWF::new(np, nb, &flop_indices); c.rng = seed;
            let mut done = 0u64; while done < cpu_target { c.run_iter(4096); done += 4096; }
            classes.iter().map(|&cls| c.pre_call_freq(cls).unwrap_or(-1.0)).collect()
        } else {
            let mut c = ConnectedMWA::new(np, nb); c.rng = seed;
            let mut done = 0u64; while done < cpu_target { c.run_iter(4096); done += 4096; }
            classes.iter().map(|&cls| c.pre_call_freq(cls).unwrap_or(-1.0)).collect()
        }
    };
    // MC_SKIP_GATE: bypass the (slow at nb=200) CPU baseline — used when
    // comparing two GPU runs against each other (e.g. emd_1d vs exact buckets).
    let skip_gate = std::env::var("MC_SKIP_GATE").map(|v| v == "1").unwrap_or(false);
    let (ca_all, cb_all) = if skip_gate {
        (vec![-1.0f32; classes.len()], vec![-1.0f32; classes.len()])
    } else {
        (cpu_freqs(0x9E3779B97F4A7C15), cpu_freqs(0xD1B54A32D192ED03))
    };
    let cref_name = if skip_gate { "SKIPPED" } else if nf_flops > 1 { "ConnectedMWF all-flops" } else { "ConnectedMWA" };
    println!("\nGATE — preflop non-fold freq (np={np}, preraises={nraises}, nf={nf_flops}; GPU vs CPU {cref_name}, 2-seed baseline):");
    println!("  {:<5} {:>9} {:>9} {:>9} {:>9} {:>9}", "class", "GPU", "CPU-A", "CPU-B", "GPU-Δ", "base-Δ");
    let (mut max_gpu_d, mut max_base_d) = (0.0f32, 0.0f32);
    for (i, (nm, _, _)) in probes.iter().enumerate() {
        let gp = gpu_callfreq(classes[i]);
        let (ca, cb) = (ca_all[i], cb_all[i]);
        let gd = if ca >= 0.0 { (gp - ca).abs() } else { 0.0 };
        let bd = if ca >= 0.0 && cb >= 0.0 { (ca - cb).abs() } else { 0.0 };
        max_gpu_d = max_gpu_d.max(gd); max_base_d = max_base_d.max(bd);
        println!("  {nm:<5} {gp:>9.4} {ca:>9.4} {cb:>9.4} {gd:>9.4} {bd:>9.4}");
    }
    let ok = max_gpu_d <= max_base_d * 1.5 + 0.05;
    println!("  GATE: {}  (max GPU-Δ {max_gpu_d:.4} vs CPU-CPU baseline {max_base_d:.4}; regret ~1/√T ✓)",
        if ok { "PASS ✓" } else { "FAIL ✗ — divergence beyond non-uniqueness" });

    // ── STRATEGY SANITY (dry-run usability): full preflop action distribution per hand
    // + NaN/Inf/degenerate scan. Eyeball: premiums aggressive, trash folds, none NaN.
    let nan = gcum.iter().filter(|x| x.is_nan() || x.is_infinite()).count();
    let nd_local = pre_local[node] as usize;
    let nd_na = pft.nodes[node].num_children as usize;
    // action labels at the fold-decision node (mark the fold action; others = call/raises)
    let labels: Vec<String> = (0..nd_na).map(|a| if a == fold_a { "FOLD".into() } else { format!("a{a}") }).collect();
    println!("\n  STRATEGY SANITY: NaN/Inf in blueprint = {nan}  |  actions = [{}]", labels.join(" "));
    let mut degenerate = 0usize; let mut untrained = 0usize;
    for (nm, a, b) in probes {
        let cls = PreflopClass::from_combo(a as Card, b as Card).index();
        let base = (nd_local * NUM_PREFLOP_CLASSES + cls) * maxna;
        let sum: f32 = (0..nd_na).map(|x| gcum[base + x].max(0.0)).sum();
        if sum <= 0.0 { untrained += 1; }
        let dist: Vec<String> = (0..nd_na).map(|x| format!("{:.3}", if sum > 0.0 { gcum[base + x].max(0.0) / sum } else { -1.0 })).collect();
        // degenerate = a pure (all-mass-on-one) strategy where a mix is expected (rough flag)
        let maxp = (0..nd_na).map(|x| if sum > 0.0 { gcum[base + x].max(0.0) / sum } else { 0.0 }).fold(0.0f32, f32::max);
        if maxp > 0.999 { degenerate += 1; }
        println!("    {nm:<5} [{}]", dist.join(" "));
    }
    println!("  (untrained classes={untrained}, ~pure classes={degenerate}/5; expect AA/KK aggressive, 72o fold-heavy)");

    // ── #5 BLUEPRINT EXTRACTION: write the converged AVERAGE strategy (cum) to disk.
    // Layout: header [magic, pre_ninfo, NUM_CLASSES, maxna, nf, post_stride, nh] then the
    // raw cum buffer (preflop region + NF postflop regions). The runtime normalizes per
    // infoset on read (strategy = cum[base..base+na] / Σ) and looks up (node,bucket via the
    // cached GS14 map for the dealt flop). This IS the blueprint (Pluribus's avg strategy).
    if let Ok(path) = std::env::var("MC_EXTRACT") {
        use std::io::Write;
        // BLP2 — self-describing header so the runtime loader (ConnBlueprint) can
        // rebuild the EXACT preflop tree + postflop map dims with no out-of-band
        // config. Layout: [magic, np, nraises, pre_ninfo, 169, maxna, nf, post_stride,
        // nh, nb, nt, nr] (12 × u32 LE) then gcum (reg_buf_len × f32 LE).
        let hdr: [u32; 12] = [0x42_4C_50_32, np as u32, nraises as u32, pre_ninfo as u32,
            NUM_PREFLOP_CLASSES as u32, maxna as u32, nf_flops as u32, post_stride as u32,
            nh as u32, nb as u32, nt_r as u32, nr_r as u32];
        let mut w = std::io::BufWriter::new(std::fs::File::create(&path).expect("blueprint file"));
        for v in hdr { w.write_all(&v.to_le_bytes()).unwrap(); }
        let mut bytes = Vec::with_capacity(reg_buf_len * 4);
        for &x in gcum { bytes.extend_from_slice(&x.to_le_bytes()); }
        w.write_all(&bytes).unwrap(); w.flush().unwrap();
        println!("  BLUEPRINT extracted → {path} (BLP2, {:.1} MB avg strategy, {reg_buf_len} floats)",
            (48 + reg_buf_len * 4) as f64 / 1e6);
    }
}

#[cfg(not(feature = "metal"))]
fn gpu_conn_solve() { eprintln!("MC_GPU_CONN requires --features metal"); }

fn main() {
    if std::env::var("MC_GS14_PRECOMPUTE").is_ok() {
        gs14_precompute();
        return;
    }
    if std::env::var("MC_GPU").is_ok() {
        gpu_mccfr_bench();
        return;
    }
    if std::env::var("MC_GPU_SD").is_ok() {
        gpu_showdown_validate();
        return;
    }
    if std::env::var("MC_GPU_CELL").is_ok() {
        gpu_cell_solve();
        return;
    }
    if std::env::var("MC_GPU_CONN").is_ok() {
        gpu_conn_solve();
        return;
    }
    if std::env::var("MC_DCFR_COSOLVE").is_ok() {
        // representative flop for the verdict (overridable via MC_FLOP).
        if std::env::var("MC_FLOP").is_err() {
            std::env::set_var("MC_FLOP", representative_flop_idx().to_string());
        }
        let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        dcfr_cosolve(nb);
        return;
    }
    if std::env::var("MC_CONNECTED").is_ok() {
        if std::env::var("MC_FLOP").is_err() {
            std::env::set_var("MC_FLOP", representative_flop_idx().to_string());
        }
        let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        connected_probe(nb);
        return;
    }
    if std::env::var("MC_CONNECTED_MW").is_ok() {
        if std::env::var("MC_FLOP").is_err() {
            std::env::set_var("MC_FLOP", representative_flop_idx().to_string());
        }
        let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        let np: usize = std::env::var("MC_NP").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        connected_mw_probe(np, nb);
        return;
    }
    if std::env::var("MC_PARALLEL").is_ok() {
        if std::env::var("MC_FLOP").is_err() {
            std::env::set_var("MC_FLOP", representative_flop_idx().to_string());
        }
        let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        let np: usize = std::env::var("MC_NP").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        connected_parallel_probe(np, nb);
        return;
    }
    if std::env::var("MC_CONNECTED_MWA").is_ok() {
        if std::env::var("MC_FLOP").is_err() {
            std::env::set_var("MC_FLOP", representative_flop_idx().to_string());
        }
        let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        let np: usize = std::env::var("MC_NP").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        connected_mwa_probe(np, nb);
        return;
    }
    if std::env::var("MC_CONNECTED_MWF").is_ok() {
        let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        let np: usize = std::env::var("MC_NP").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        connected_mwf_probe(np, nb);
        return;
    }
    let nb: usize = std::env::var("MC_NB").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let nt: usize = std::env::var("MC_NT").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let nr: usize = std::env::var("MC_NR").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    if std::env::var("MC_ANCHOR").is_ok() {
        anchor_validation(nb, nt, nr);
        return;
    }
    let iters: u32 = std::env::var("MC_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);

    let batch: usize = std::env::var("MC_B").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let mc_iters: usize = std::env::var("MC_MCITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);

    println!("SHRUNK GAME: nb={nb} runout={nt}x{nr} | B={batch} | DCFR (enumerate) vs MCCFR (sample)\n");
    println!("{:>5} {:>8} {:>10} {:>13} {:>15} {:>14} {:>10}",
        "live", "num_opp", "nodes", "DCFR s/iter", "MCCFR µs/traj", "nb^num_opp", "DCFR/MC*");
    for live in 2u8..=5 {
        let g = build_shrunk(live, nb, nt, nr);
        let (dper, nodes) = dcfr_per_iter(&g, iters);
        // MCCFR: time `mc_iters` batches, report per-trajectory µs.
        let mut mc = Mccfr::new(&g);
        mc.run_iter(&g.tree, batch); // warm
        let t0 = Instant::now();
        for _ in 0..mc_iters {
            mc.run_iter(&g.tree, batch);
        }
        let total_traj = (mc_iters * batch) as f64;
        let mc_us = t0.elapsed().as_secs_f64() / total_traj * 1e6;
        // Equal-work ratio: DCFR per-iter vs MCCFR per (batch=nodes-ish) — report
        // DCFR-iter / MCCFR-per-traj as the raw per-unit speed ratio.
        let ratio = dper / (mc_us * 1e-6);
        let wall = (nb as f64).powi((live - 1) as i32);
        println!("{:>5} {:>8} {:>10} {:>13.4} {:>15.3} {:>14.0} {:>10.0}",
            live, live - 1, nodes, dper, mc_us, wall, ratio);
    }
    // SANITY: confirm the engine does REAL work (not a no-op that would also
    // look flat+fast). Run live-3 for a bit and report regret occupancy + a
    // sample of strategy spread. NOT a convergence claim — the anchor is next.
    {
        let g = build_shrunk(3, nb, nt, nr);
        let mut mc = Mccfr::new(&g);
        for _ in 0..50 {
            mc.run_iter(&g.tree, 4096);
        }
        let nz = mc.regret.iter().filter(|&&r| r != 0.0).count();
        let touched = mc.cum.iter().filter(|&&c| c > 0.0).count();
        let rmax = mc.regret.iter().cloned().fold(0.0f32, |m, x| m.max(x.abs()));
        println!(
            "\nSANITY (live-3, 50×4096 traj): regret non-zero {}/{} ({:.0}%), cum touched {} | max|regret| {:.3e}",
            nz, mc.regret.len(), 100.0 * nz as f64 / mc.regret.len() as f64, touched, rmax
        );
        println!("(non-trivial occupancy ⇒ the traversal really visits infosets + updates regret,");
        println!(" so the per-traj timing is real work, not a short-circuit. Convergence still UNVALIDATED.)");
    }

    println!("\nWALL-ESCAPE READ: DCFR s/iter should track nb^num_opp; MCCFR µs/traj should stay ~FLAT");
    println!("across live-count (it samples ONE tuple, O(num_opp), not the nb^num_opp enumeration).");
    println!("NOT YET MEASURED (next milestone): iters-to-converge vs the full-tree-enumerated");
    println!("exploitability anchor, + the AA-raises garbage check. Per-iter alone is half the answer.");
}
