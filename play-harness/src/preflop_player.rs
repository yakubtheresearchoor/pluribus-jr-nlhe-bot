//! Loads the dumped EQR preflop strategy (`<base>.f32` + `<base>.json`) and
//! samples per-(decision-node, hand-class) actions — the bot's PREFLOP play for
//! the full-hand production baseline. Rebuilds the EXACT cap-3 preflop tree the
//! strategy was solved on (`production_game_v1` + no-open-limp + 3bet-or-fold +
//! cap-3), so node indices and `local_offset` align with the saved array.
//!
//! Strategy layout (from `PreflopVectorCfr::average_strategy`, already
//! normalized): `avg[local_offset[node] * MAX_NA_PREFLOP * NC + a*NC + class]`.

use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::card::Card;
use solver_core::solver::preflop_cfr::PreflopVectorCfr;
use solver_core::tree::action::{production_game_v1, BetCap, BetSize, BetSizeOptions};
use solver_core::tree::builder::build_tree_preflop_only;
use solver_core::tree::flat::{FlatTree, MAX_NA_PREFLOP};

/// `PreflopVectorCfr::local_offset` sentinel for non-decision nodes.
const UNUSED: usize = usize::MAX;

/// Backing store for the ~3.2 GB `.f32` average-strategy array. `Mapped` is a
/// zero-copy, read-only memory map of the file — the strategy lives in evictable,
/// file-backed page cache (macOS drops cold pages for free and re-reads on touch)
/// instead of a resident `Vec<f32>` heap allocation (which can only be *compressed*
/// when cold, costing CPU + committed memory). A decision reads a handful of cells,
/// so the resident working set is a few pages, not 3.2 GB. `Owned` is the fallback
/// when the file can't be mapped (kept byte-identical to the old path).
enum StratStore {
    Mapped(memmap2::Mmap),
    Owned(Vec<f32>),
}

impl StratStore {
    #[inline]
    fn as_slice(&self) -> &[f32] {
        match self {
            // The `.f32` is native-endian little-endian f32; the file is page-aligned
            // (mmap), so the cast is valid + byte-identical to the old from_le_bytes
            // load on little-endian hosts (all our targets).
            StratStore::Mapped(m) => bytemuck::cast_slice(&m[..]),
            StratStore::Owned(v) => v.as_slice(),
        }
    }
}

pub struct PreflopPlayer {
    pub tree: FlatTree,
    solver: PreflopVectorCfr,
    strat: StratStore,
}

impl PreflopPlayer {
    /// Rebuild the cap-3 preflop tree from the artifact header and load the
    /// `.f32` average strategy. `base` = path stem (no extension).
    pub fn load(base: &str) -> std::io::Result<Self> {
        let header = std::fs::read_to_string(format!("{base}.json"))?;
        let n_raises = json_int(&header, "n_raises") as usize;
        let avg_len = json_int(&header, "avg_len") as usize;
        let num_nodes = json_int(&header, "num_nodes") as usize;
        // Read the tree-shape flags from the header so the loader rebuilds the EXACT
        // tree the chart was solved on (default true/true for older artifacts that
        // predate flat-call defense). Mismatch is caught by the num_nodes assert below.
        let no_open_limp = json_bool(&header, "no_open_limp", true);
        let threebet_or_fold = json_bool(&header, "threebet_or_fold", true);
        // conn_tree: the chart was solved on the SHARED build_conn_preflop_tree (the one
        // the connected blueprint freezes) rather than the standalone cap3 menu — rebuild
        // the identical tree so node/action indexing maps 1:1.
        let conn_tree = json_bool(&header, "conn_tree", false);

        let spec = production_game_v1();
        let tree = if conn_tree {
            solver_core::blueprint::build_conn_preflop_tree(6, n_raises).0
        } else {
            let mrc = n_raises.min(MAX_NA_PREFLOP.saturating_sub(2)).max(1);
            let mut cfg = spec.preflop_tree_config(BetSizeOptions {
                bet: vec![BetSize::PotRelative(1.0)],
                raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect(),
            });
            cfg.max_bets_per_street = BetCap::all(3);
            cfg.no_open_limp = no_open_limp;
            cfg.threebet_or_fold = threebet_or_fold;
            build_tree_preflop_only(&cfg).expect("preflop tree")
        };
        assert_eq!(
            tree.num_nodes(),
            num_nodes,
            "preflop tree ({} nodes) != artifact header ({num_nodes}) — config drift",
            tree.num_nodes()
        );

        let solver = PreflopVectorCfr::new(&tree);
        // Memory-map the ~3.2 GB `.f32` strategy (zero-copy, evictable page cache)
        // rather than reading it into a resident `Vec<f32>`. Falls back to an owned
        // read if the map fails, so behavior is unchanged when mmap is unavailable.
        let path = format!("{base}.f32");
        let expect_bytes = avg_len * 4;
        let strat = match std::fs::File::open(&path).and_then(|f| {
            // Safety: the file is opened read-only and not truncated while mapped
            // (it's a static artifact); the Mmap is owned by the returned struct.
            let m = unsafe { memmap2::Mmap::map(&f)? };
            Ok(m)
        }) {
            Ok(m) if m.len() == expect_bytes && (m.as_ptr() as usize) % 4 == 0 => {
                StratStore::Mapped(m)
            }
            _ => {
                let bytes = std::fs::read(&path)?;
                assert_eq!(bytes.len(), expect_bytes, "strategy .f32 length mismatch vs header avg_len");
                StratStore::Owned(
                    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                )
            }
        };

        Ok(PreflopPlayer { tree, solver, strat })
    }

    /// 169-class index for a hole-card combo (order-independent).
    pub fn hand_class(c1: Card, c2: Card) -> usize {
        PreflopClass::from_combo(c1, c2).index()
    }

    /// Write the `na` action probabilities at (node, hand_class) into `out`;
    /// returns `na`. Uniform fallback for unused/no-action nodes.
    pub fn action_dist(&self, node: usize, hand_class: usize, out: &mut [f32]) -> usize {
        let na = self.tree.nodes[node].num_children as usize;
        let local = self.solver.local_offset[node];
        if local == UNUSED || na == 0 {
            let u = 1.0 / na.max(1) as f32;
            for v in out[..na].iter_mut() {
                *v = u;
            }
            return na;
        }
        let off = local * MAX_NA_PREFLOP * NUM_PREFLOP_CLASSES;
        let strat = self.strat.as_slice();
        for a in 0..na {
            out[a] = strat[off + a * NUM_PREFLOP_CLASSES + hand_class];
        }
        na
    }

    /// Sample an action index by the strategy at (node, hand_class).
    pub fn sample_action(&self, node: usize, hand_class: usize, rng: &mut u64) -> usize {
        let mut buf = [0f32; MAX_NA_PREFLOP];
        let na = self.action_dist(node, hand_class, &mut buf);
        let mut x = (splitmix64(rng) % 1_000_000) as f32 / 1_000_000.0;
        for a in 0..na {
            if x < buf[a] {
                return a;
            }
            x -= buf[a];
        }
        na.saturating_sub(1)
    }
}

#[inline]
pub fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Parse `"key":true|false` from the header, falling back to `default` if absent.
fn json_bool(header: &str, key: &str, default: bool) -> bool {
    let pat = format!("\"{key}\":");
    match header.find(&pat) {
        Some(i) => header[i + pat.len()..].trim_start().starts_with("true"),
        None => default,
    }
}

fn json_int(header: &str, key: &str) -> i64 {
    let pat = format!("\"{key}\":");
    let s = &header[header.find(&pat).unwrap_or_else(|| panic!("missing {key}")) + pat.len()..];
    let end = s.find(|c: char| !c.is_ascii_digit() && c != '-').unwrap_or(s.len());
    s[..end].parse().unwrap_or_else(|_| panic!("bad int for {key}"))
}

#[cfg(test)]
mod mmap_tests {
    use super::StratStore;
    use std::io::Write;

    /// The mmap zero-copy cast must be byte-identical to the old `from_le_bytes`
    /// load on this (little-endian) host — the correctness gate for swapping the
    /// resident `Vec<f32>` for a memory-mapped `&[f32]`.
    #[test]
    fn mmap_cast_matches_from_le_bytes() {
        let vals: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.5 - 7.0).collect();
        let mut bytes = Vec::new();
        for v in &vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let path = std::env::temp_dir().join(format!("pf_mmap_test_{}.f32", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();

        let f = std::fs::File::open(&path).unwrap();
        let m = unsafe { memmap2::Mmap::map(&f).unwrap() };
        assert_eq!(m.as_ptr() as usize % 4, 0, "mmap must be 4-aligned for the f32 cast");
        let mapped = StratStore::Mapped(m);

        let owned = StratStore::Owned(
            bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        );
        assert_eq!(mapped.as_slice(), owned.as_slice(), "mmap cast != from_le_bytes");
        assert_eq!(mapped.as_slice(), vals.as_slice(), "mmap values wrong");
        std::fs::remove_file(&path).ok();
    }
}
