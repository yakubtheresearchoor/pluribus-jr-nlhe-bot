//! BANKED-READ preflop oracle source (2026-06-14): serves flop-root CFVs to
//! the preflop solve by READING the banked postflop fill — load the SPR-bucket
//! representative's .bp, set its average strategy, extract the root CFV via
//! `root_cfv_from_avg` — instead of re-solving. Drop-in for
//! `BucketKeyedOracle::new(.., src)`; consistent with the deployed values
//! (same banked runouts + reach-weighted B), ~1/34 the cost of re-solving.
//!
//! Alignment: `SeamCell::bucket_key(stack)` is the SAME (live, log2-SPR/0.25)
//! binning the cell-set census used to pick the reps, so an incoming cell maps
//! to its banked representative exactly. Gated `read ≈ re-solve` by
//! `root_cfv_from_avg_gate` (extractor) + `preflop_oracle_banked_gate` (wiring).

use std::collections::HashMap;
use std::sync::Arc;

use solver_core::card::Card;
use solver_core::solver::flop_start_game::FlopChanceTable;
use solver_core::solver::postflop_oracle::SeamCell;
use solver_core::solver::preflop_start_game::{
    compute_v_flop_at_root_converged_with_table, flop_combo_layout, table_hand_to_layout_perm,
};
use solver_core::tree::action::{production_game_v1, BetSize, BetSizeOptions, GameSpec};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

use crate::blueprint::Blueprint;

/// DCFR iters for the exact live-2 re-solve (matches the postflop fill).
const LIVE2_ITERS: u32 = 34;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Reproduce the runner's seeded 1×1 runout draw (turn x0=fi; river
/// x0=fi^0xDEADBEEF^(tc<<40)) so live-2 re-solves use the same runout
/// convention the live-3/4/5 fill banked — consistent if live-2 is later baked.
pub fn seeded_1x1(canonical: [Card; 3], fi: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let bm: u64 = canonical.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| bm & (1u64 << c) == 0).collect();
    let mut pool = deck.clone();
    let x = splitmix64(fi as u64);
    let tc = pool.remove((x % pool.len() as u64) as usize);
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    let mut rpool: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
    let rx = splitmix64((fi as u64) ^ 0xDEAD_BEEF ^ ((tc as u64) << 40));
    let rc = rpool.remove((rx % rpool.len() as u64) as usize);
    river_decks[tc as usize] = vec![rc];
    (vec![tc], river_decks)
}

/// Per-live layout-ordered flop-root CFV by READING a banked rep `.bp`
/// (live-3/4/5): load → set avg → `root_cfv_from_avg`. `tree` is the rep's
/// seam tree. Shared by the oracle source and the CFV pre-bank.
pub fn cfv_from_banked(dir: &str, fi: usize, tree: &FlatTree, canonical: [Card; 3]) -> Vec<Vec<f32>> {
    let bp = Blueprint::load(&format!("{dir}/flop_{fi:04}.bp"))
        .unwrap_or_else(|e| panic!("load {dir}/flop_{fi:04}.bp: {e}"));
    let mut bucketed = bp.indexer(tree);
    bucketed.cum_strategy_flop_mut().copy_from_slice(&bp.cum_flop);
    bucketed.cum_strategy_turn_mut().copy_from_slice(&bp.cum_turn);
    bucketed.cum_strategy_river_mut().copy_from_slice(&bp.cum_river);
    let root_all = bucketed.root_cfv_from_avg(tree, &bp.game, &bp.bk);
    let perm = table_hand_to_layout_perm(&bp.game.table().hand_cards, bp.game.table().num_valid, canonical);
    root_all.iter().map(|ph| perm.iter().map(|&h| ph[h]).collect()).collect()
}

/// Build the CHECK-ONLY seam tree for an equity rollout: empty bet options AND
/// all-in disabled. `flop_seam_config` otherwise adds an all-in action when SPR
/// ≤ 1 (`add_allin_threshold`), which at np=6 explodes the tree from 21 → 1713
/// nodes and the collapsed terminal from 0.04 s → ~14 s/flop. An equity rollout
/// is pure check-down-to-showdown, so suppress all betting. 21 nodes at every
/// commit; the showdown CFV is the only content.
pub fn rollout_seam_tree(spec: &GameSpec, live: u8, commit: i32, pot: i32) -> FlatTree {
    let mut cfg = spec.flop_seam_config(
        live,
        commit.min(spec.stack),
        pot,
        BetSizeOptions { bet: vec![], raise: vec![] },
    );
    cfg.add_allin_threshold = 0.0;
    cfg.force_allin_threshold = 0.0;
    build_tree(&cfg).expect("rollout seam tree")
}

/// DCFR iters for an EQUITY ROLLOUT solve. The rollout tree is CHECK-ONLY (no
/// betting), so every infoset has one forced action — the strategy is constant
/// across iters and the root showdown CFV is iter-independent. A handful of
/// iters suffices (the cum-average is the same forced strategy at every step).
const ROLLOUT_ITERS: u32 = 2;

/// EQUITY-ROLLOUT flop-root CFV for live≥3, bucketed. Used for cells the fill
/// never banks: ALL-IN cells (behind ≤ 0, any live — no chips to bet) and
/// LIVE-6 (the full table; the cell census is live 2..=5, so six-way is always
/// rolled out, never solved — the betting tree is 45k nodes vs 21 check-only).
/// The caller passes a CHECK-ONLY seam tree (empty bet options) so the game is
/// check-down-to-showdown = an equity rollout; the showdown CFV comes from the
/// SAME bucketed terminal the solved cells use, on the seeded 1×1 runout, so
/// the convention matches `cfv_from_banked`. Mirrors the keystone source's
/// live≥3 path (`bucketed_live_subset_source`) with a degenerate tree.
pub fn cfv_rollout_live3plus(
    canonical: [Card; 3],
    fi: usize,
    checkonly_tree: &FlatTree,
    live: u8,
    nb: usize,
) -> Vec<Vec<f32>> {
    use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
    use solver_core::solver::flop_start_game::FlopStartGame;
    let (turns, river_decks) = seeded_1x1(canonical, fi);
    let table = FlopChanceTable::build_full_nh_sampled(canonical, live, &turns, &river_decks);
    let bk = FlopBucketing::quantile(&table, nb);
    let perm = table_hand_to_layout_perm(&table.hand_cards, table.num_valid, canonical);
    let game = FlopStartGame::new(table);
    let mut solver = BucketedFlopCfr::new(checkonly_tree, game.table(), &bk);
    solver.set_terminal_design(TerminalDesign::Design1Collapsed);
    // Train (trivial on a check-only tree) then extract via root_cfv_from_avg,
    // which applies the opponent-reach-sum-1 normalization (per-hand EV). Using
    // the same extraction as the live-3/4/5 banked path keeps the rollout CFV on
    // the SAME chip scale — essential so the preflop compares rollout (live-6 /
    // all-in) cells against solved cells consistently.
    solver.run(checkonly_tree, &game, &bk, ROLLOUT_ITERS);
    let root_all = solver.root_cfv_from_avg(checkonly_tree, &game, &bk);
    root_all.iter().map(|ph| perm.iter().map(|&h| ph[h]).collect()).collect()
}

/// Per-live (2) layout-ordered flop-root CFV by EXACT re-solve (live-2; np=2
/// O(nh²), no wall). Sampled 1×1 table through the keystone's extraction, so
/// the CFV convention matches `cfv_from_banked`. Shared by source + pre-bank.
pub fn cfv_live2(canonical: [Card; 3], fi: usize, tree: &FlatTree) -> Vec<Vec<f32>> {
    let (turns, river_decks) = seeded_1x1(canonical, fi);
    let layout = flop_combo_layout(canonical);
    (0..2u8)
        .map(|t| {
            let table = FlopChanceTable::build_full_nh_sampled(canonical, 2, &turns, &river_decks);
            let (v_tbl, lay_tbl) =
                compute_v_flop_at_root_converged_with_table(table, tree, t, LIVE2_ITERS);
            let mut pos: HashMap<(Card, Card), usize> = HashMap::new();
            for (i, &(a, b)) in lay_tbl.iter().enumerate() {
                pos.insert((a.min(b), a.max(b)), i);
            }
            layout.iter().map(|&(a, b)| v_tbl[pos[&(a.min(b), a.max(b))]]).collect()
        })
        .collect()
}

/// Path of a pre-banked CFV file for SPR bucket `key` at flop index `fi`.
pub fn cfv_path(cfv_root: &str, key: (u8, i64), fi: usize) -> String {
    format!("{cfv_root}/L{}_S{}/cfv_{fi:04}.f32", key.0, key.1)
}

/// Read a pre-banked per-live CFV (live × nlay f32-LE), if present.
fn read_prebanked(cfv_root: &str, key: (u8, i64), fi: usize, live: u8) -> Option<Vec<Vec<f32>>> {
    let bytes = std::fs::read(cfv_path(cfv_root, key, fi)).ok()?;
    let floats: Vec<f32> =
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let nlay = floats.len() / live as usize;
    Some((0..live as usize).map(|t| floats[t * nlay..(t + 1) * nlay].to_vec()).collect())
}

/// Write a pre-banked per-live CFV (the CFV pre-bank pass).
pub fn write_prebanked(cfv_root: &str, key: (u8, i64), fi: usize, per_live: &[Vec<f32>]) {
    let dir = format!("{cfv_root}/L{}_S{}", key.0, key.1);
    std::fs::create_dir_all(&dir).expect("mkdir cfv bucket");
    let mut bytes = Vec::new();
    for v in per_live {
        for &x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(format!("{dir}/cfv_{fi:04}.f32"), bytes).expect("write cfv");
}

/// Build the read-banked source closure.
///   `blueprint_root` — dir holding `live{N}_c{C}_p{P}_b{B}/flop_NNNN.bp`.
///   `cells` — the banked reps `(live, commit, pot, b)` (one per SPR bucket).
///   `canonical_flops` — the 1755 ordered canonical flops (index = .bp number).
///   `stack` — game-class stack (for bucket_key alignment).
pub fn banked_read_source(
    blueprint_root: String,
    cells: Vec<(u8, i32, i32, usize)>,
    canonical_flops: Vec<[Card; 3]>,
    stack: i32,
) -> impl FnMut(SeamCell, u16, [Card; 3], &[Vec<f32>], u8) -> Vec<f32> {
    // bucket_key -> (rep live, commit, pot, dir)
    let mut bucket_map: HashMap<(u8, i64), (u8, i32, i32, String)> = HashMap::new();
    // per-live bucket width (for the on-the-fly all-in solve, which has no rep).
    let mut live_nb: HashMap<u8, usize> = HashMap::new();
    for &(live, commit, pot, b) in &cells {
        let key = SeamCell { live, commit, pot }.bucket_key(stack);
        bucket_map.insert(key, (live, commit, pot, format!("{blueprint_root}/live{live}_c{commit}_p{pot}_b{b}")));
        live_nb.entry(live).or_insert(b);
    }
    // canonical flop -> fi (the .bp file index)
    let mut flop_index: HashMap<[Card; 3], usize> = HashMap::new();
    for (fi, f) in canonical_flops.iter().enumerate() {
        flop_index.insert(*f, fi);
    }

    let cfv_root = format!("{blueprint_root}/cfv"); // pre-banked CFVs, if present
    let bets = BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] };
    let spec = production_game_v1();
    let mut trees: HashMap<(u8, i32, i32), Arc<FlatTree>> = HashMap::new();
    // (bucket_key, canonical) -> per-live layout-ordered CFV
    let mut cache: HashMap<((u8, i64), [Card; 3]), Vec<Vec<f32>>> = HashMap::new();
    // PROGRESS: instrument the lazy cache-fill so a short run reveals the fill
    // rate (cells/sec) — the iter-1 oracle fill is the dominant cost and its
    // size scales with how many distinct SPR-bucket seams the preflop tree
    // reaches. Gated on PF_ORACLE_PROGRESS to stay silent in production.
    let progress = std::env::var("PF_ORACLE_PROGRESS").is_ok();
    let t_fill = std::time::Instant::now();
    let mut misses: usize = 0;
    let mut live2_misses: usize = 0;

    move |cell, folded_mask, canonical, combo_reaches, traverser| {
        let np = combo_reaches.len();
        let live_seats: Vec<usize> = (0..np).filter(|&p| (folded_mask >> p) & 1 == 0).collect();
        let trav_live = live_seats
            .iter()
            .position(|&p| p == traverser as usize)
            .expect("traverser must be live");
        let key = cell.bucket_key(stack);

        if !cache.contains_key(&(key, canonical)) {
            let fi = *flop_index
                .get(&canonical)
                .unwrap_or_else(|| panic!("canonical flop {canonical:?} not in canonical list"));

            // Fast path: a pre-banked CFV file (cfv_out/L{live}_S{bin}/cfv_NNNN.f32),
            // if the CFV pre-bank pass has run — pure read, no extraction.
            let per_live: Vec<Vec<f32>> = if let Some(pb) = read_prebanked(&cfv_root, key, fi, cell.live) {
                pb
            } else if cell.live == 2 {
                let tree = trees
                    .entry((cell.live, cell.commit, cell.pot))
                    .or_insert_with(|| {
                        Arc::new(build_tree(&spec.flop_seam_config(cell.live, cell.commit, cell.pot, bets.clone())).expect("live-2 seam tree"))
                    })
                    .clone();
                cfv_live2(canonical, fi, &tree)
            } else if cell.live < 5 && bucket_map.contains_key(&key) {
                let (rlive, rcommit, rpot, dir) = bucket_map.get(&key).cloned().unwrap();
                let tree = trees
                    .entry((rlive, rcommit, rpot))
                    .or_insert_with(|| {
                        Arc::new(build_tree(&spec.flop_seam_config(rlive, rcommit, rpot, bets.clone())).expect("seam tree"))
                    })
                    .clone();
                cfv_from_banked(&dir, fi, &tree, canonical)
            } else {
                // ROLLOUT (check-down-to-showdown) for cells without a cheap exact
                // CFV here:
                //   • all-in (behind ≤ 0, any live): no chips to bet,
                //   • live-6 (full table): never solved, and
                //   • live-5 NOT pre-banked: the exact 5-way root_cfv_from_avg is
                //     pathologically slow (~26 min/bucket) and 5-way flops are <1%
                //     of volume — a check-down rollout EV is a fine preflop prior.
                //     (Pre-banked live-5 still read exact via read_prebanked above.)
                let is_allin = key.1 == i64::MIN;
                assert!(
                    is_allin || cell.live >= 5,
                    "non-rollout SPR bucket {key:?} missing from banked fill (cell {cell:?})"
                );
                let nb = live_nb.get(&cell.live).copied().unwrap_or(8);
                let tree = trees
                    .entry((cell.live, cell.commit, cell.pot))
                    .or_insert_with(|| Arc::new(rollout_seam_tree(&spec, cell.live, cell.commit, cell.pot)))
                    .clone();
                cfv_rollout_live3plus(canonical, fi, &tree, cell.live, nb)
            };
            cache.insert((key, canonical), per_live);
            misses += 1;
            if cell.live == 2 {
                live2_misses += 1;
            }
            if progress && misses % 200 == 0 {
                let s = t_fill.elapsed().as_secs_f64();
                eprintln!(
                    "    [oracle fill] {misses} cells ({live2_misses} live-2 re-solves) in {s:.1}s = {:.1} cells/s",
                    misses as f64 / s,
                );
            }
        }
        cache[&(key, canonical)][trav_live].clone()
    }
}
