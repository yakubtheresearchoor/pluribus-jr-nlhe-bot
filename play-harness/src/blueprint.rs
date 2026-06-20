//! SSBP1 blueprint loader: parses a banked per-flop artifact and
//! reconstructs everything an agent needs to PLAY it — the chance
//! table (rebuilt from the header's flop + pinned runouts), the
//! bucketing (from the banked maps), and the solver-side indexing
//! (BucketedFlopCfr layout, so cum sections index exactly as banked).
//!
//! The header is one JSON line; we extract the few numeric fields with
//! a minimal hand-rolled scanner (arrays of integers + integers) to
//! keep the harness dependency-free.

use solver_core::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalDesign};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::io::Read;

pub struct Blueprint {
    pub flop: [u8; 3],
    pub turns: Vec<u8>,
    pub rivers: Vec<Vec<u8>>, // per turn (same order as `turns`)
    pub np: usize,            // players at the flop (oracle=6, per-family seam=live)
    pub nb: usize,
    pub nh: usize,
    pub cum_flop: Vec<f32>,
    pub cum_turn: Vec<f32>,
    pub cum_river: Vec<f32>,
    pub bk: FlopBucketing,
    pub game: FlopStartGame,
}

/// The oracle-shape postflop tree every blueprint in the current
/// family is solved on (pinned in the artifact header).
pub fn build_oracle_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 12,
        starting_stacks: vec![94; 6],
        initial_contributions: vec![0; 6],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
            no_open_limp: false,
            threebet_or_fold: false,
    };
    build_tree(&cfg).expect("oracle tree")
}

/// Extract `"key":<integer>` from the header line.
fn json_int(header: &str, key: &str) -> i64 {
    let pat = format!("\"{key}\":");
    let s = &header[header.find(&pat).unwrap_or_else(|| panic!("missing {key}")) + pat.len()..];
    let end = s.find(|c: char| !c.is_ascii_digit() && c != '-').unwrap_or(s.len());
    s[..end].parse().expect("int field")
}

/// Extract `"key":[ ... ]` raw bracket body (balanced).
fn json_array_body<'a>(header: &'a str, key: &str) -> &'a str {
    let pat = format!("\"{key}\":[");
    let start =
        header.find(&pat).unwrap_or_else(|| panic!("missing {key}")) + pat.len();
    let mut depth = 1;
    for (i, c) in header[start..].char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return &header[start..start + i];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced array for {key}");
}

fn ints(body: &str) -> Vec<i64> {
    body.split(|c: char| !c.is_ascii_digit() && c != '-')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse().unwrap())
        .collect()
}

impl Blueprint {
    /// In-place u8 quantization round-trip of the cumulative strategy
    /// (per-array linear min..max → 0..255 → back). Models the deployed
    /// u8-quantized blueprint at LOAD time: the bot then plays from the
    /// quantization-degraded strategy, so a money test with this on proves
    /// quant is play-safe without rewriting the 44GB artifact to disk.
    /// Measured round-trip error on real blueprints: ~0.1% on EV-relevant
    /// (high-reach) strategy mass (the cum values are bounded ~[0,iters],
    /// so a single linear scale preserves within-infoset ratios).
    pub fn quantize_roundtrip(&mut self) {
        fn q8(a: &mut [f32]) {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in a.iter() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let scale = (hi - lo).max(1e-12);
            for v in a.iter_mut() {
                let q = (((*v - lo) / scale) * 255.0).round().clamp(0.0, 255.0) as u8;
                *v = lo + (q as f32 / 255.0) * scale;
            }
        }
        q8(&mut self.cum_flop);
        q8(&mut self.cum_turn);
        q8(&mut self.cum_river);
    }

    pub fn load(path: &str) -> std::io::Result<Blueprint> {
        let mut raw = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut raw)?;
        assert_eq!(&raw[..6], b"SSBP1\n", "bad magic");
        let hdr_end = 6 + raw[6..].iter().position(|&b| b == b'\n').expect("header line");
        let header = std::str::from_utf8(&raw[6..hdr_end]).expect("utf8 header").to_string();

        // Sections: name '\n' u64-LE len, bytes.
        let mut sections = std::collections::HashMap::new();
        let mut pos = hdr_end + 1;
        while pos < raw.len() {
            let name_end = pos + raw[pos..].iter().position(|&b| b == b'\n').unwrap();
            let name = std::str::from_utf8(&raw[pos..name_end]).unwrap().to_string();
            let mut len8 = [0u8; 8];
            len8.copy_from_slice(&raw[name_end + 1..name_end + 9]);
            let len = u64::from_le_bytes(len8) as usize;
            let data = raw[name_end + 9..name_end + 9 + len].to_vec();
            sections.insert(name, data);
            pos = name_end + 9 + len;
        }
        let f32s = |name: &str| -> Vec<f32> {
            sections[name]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let u16s = |name: &str| -> Vec<u16> {
            sections[name]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };

        let flop_v = ints(json_array_body(&header, "flop"));
        let flop = [flop_v[0] as u8, flop_v[1] as u8, flop_v[2] as u8];
        let turns: Vec<u8> = ints(json_array_body(&header, "turn")).iter().map(|&x| x as u8).collect();
        let rivers_body = json_array_body(&header, "rivers");
        let rivers: Vec<Vec<u8>> = rivers_body
            .split(']')
            .filter(|s| s.contains(|c: char| c.is_ascii_digit()))
            .map(|s| ints(s).iter().map(|&x| x as u8).collect())
            .collect();
        let nb = ints(json_array_body(&header, "b"))[0] as usize;
        let nh = json_int(&header, "nh") as usize;
        // Players at the flop: header `"tree":{"np":N,...}`. Old oracle
        // artifacts carry np=6; per-family seam cells carry np=live (3/4/5).
        let np = json_int(&header, "np") as usize;

        // Rebuild the table from pinned runouts (same constructor the
        // runner used, at the cell's np) and bucketing from banked maps.
        let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
        for (ti, &tc) in turns.iter().enumerate() {
            river_decks[tc as usize] = rivers[ti].clone();
        }
        let table = FlopChanceTable::build_full_nh_sampled(flop, np as u8, &turns, &river_decks);
        assert_eq!(table.num_valid, nh, "rebuilt table nh mismatch");
        let game = FlopStartGame::new(table);

        let flop_map = u16s("map_flop");
        let turn_cat = u16s("map_turn");
        let river_cat = u16s("map_river");
        assert_eq!(flop_map.len(), nh);
        assert_eq!(turn_cat.len(), turns.len() * nh);
        let turn_map: Vec<Vec<u16>> =
            turn_cat.chunks_exact(nh).map(|c| c.to_vec()).collect();
        let mut river_map: Vec<Vec<Vec<u16>>> = Vec::new();
        let mut off = 0;
        for r in &rivers {
            let mut per_t = Vec::new();
            for _ in r {
                per_t.push(river_cat[off..off + nh].to_vec());
                off += nh;
            }
            river_map.push(per_t);
        }
        assert_eq!(off, river_cat.len(), "river map section length");
        let bk =
            FlopBucketing::from_maps(game.table(), nb, nb, nb, flop_map, turn_map, river_map);

        Ok(Blueprint {
            flop,
            turns,
            rivers,
            np,
            nb,
            nh,
            cum_flop: f32s("cum_flop"),
            cum_turn: f32s("cum_turn"),
            cum_river: f32s("cum_river"),
            bk,
            game,
        })
    }

    /// Solver-side indexing for the cum sections (the EXACT layout
    /// they were banked from).
    pub fn indexer(&self, tree: &FlatTree) -> BucketedFlopCfr {
        let mut s = BucketedFlopCfr::new(tree, self.game.table(), &self.bk);
        s.set_terminal_design(TerminalDesign::Design1Collapsed);
        s
    }
}
