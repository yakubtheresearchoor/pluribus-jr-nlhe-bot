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

pub struct PreflopPlayer {
    pub tree: FlatTree,
    solver: PreflopVectorCfr,
    strat: Vec<f32>,
}

impl PreflopPlayer {
    /// Rebuild the cap-3 preflop tree from the artifact header and load the
    /// `.f32` average strategy. `base` = path stem (no extension).
    pub fn load(base: &str) -> std::io::Result<Self> {
        let header = std::fs::read_to_string(format!("{base}.json"))?;
        let n_raises = json_int(&header, "n_raises") as usize;
        let avg_len = json_int(&header, "avg_len") as usize;
        let num_nodes = json_int(&header, "num_nodes") as usize;

        let spec = production_game_v1();
        let mrc = n_raises.min(MAX_NA_PREFLOP.saturating_sub(2)).max(1);
        let mut cfg = spec.preflop_tree_config(BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: (0..mrc).map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64)).collect(),
        });
        cfg.max_bets_per_street = BetCap::all(3);
        cfg.no_open_limp = true;
        cfg.threebet_or_fold = true;
        let tree = build_tree_preflop_only(&cfg).expect("preflop tree");
        assert_eq!(
            tree.num_nodes(),
            num_nodes,
            "preflop tree ({} nodes) != artifact header ({num_nodes}) — config drift",
            tree.num_nodes()
        );

        let solver = PreflopVectorCfr::new(&tree);
        let bytes = std::fs::read(format!("{base}.f32"))?;
        assert_eq!(bytes.len(), avg_len * 4, "strategy .f32 length mismatch vs header avg_len");
        let strat: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

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
        for a in 0..na {
            out[a] = self.strat[off + a * NUM_PREFLOP_CLASSES + hand_class];
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

fn json_int(header: &str, key: &str) -> i64 {
    let pat = format!("\"{key}\":");
    let s = &header[header.find(&pat).unwrap_or_else(|| panic!("missing {key}")) + pat.len()..];
    let end = s.find(|c: char| !c.is_ascii_digit() && c != '-').unwrap_or(s.len());
    s[..end].parse().unwrap_or_else(|_| panic!("bad int for {key}"))
}
