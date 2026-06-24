//! Connected-blueprint `/decide` path. Serves decisions by LOOKUP from the
//! sharded connected MCCFR blueprint (`blueprint_conn_v1/`) — a different artifact
//! than the per-cell search blueprint the rest of `api` uses. Currently serves
//! PREFLOP and FLOP decisions (the cell root is the flop, so a street-local flop
//! request maps directly). TURN/RIVER need the full postflop betting path, which
//! the street-local `DecideRequest` doesn't carry — a noted API extension.

use crate::api::{action_name, ActionProb, DecideRequest, DecideResponse};
use solver_core::abstraction::preflop_class::NUM_PREFLOP_CLASSES;
use solver_core::blueprint::{
    gs14_cache_path, load_gs14_cache, ConnCellLayout, FlopLayout, ShardedConnBlueprint,
};
use solver_core::card::Card;
use solver_core::solver::preflop_start_game::PreflopChanceTable;

/// Holds the loaded connected blueprint + reconstructed cell layout + the GS14
/// bucket cache dir. Read-only at decision time — share across request threads.
pub struct ConnDecider {
    bp: ShardedConnBlueprint,
    layout: ConnCellLayout,
    gs14_dir: String,
    nb: usize,
    bnt: usize, // cache runout dims (full 49×48 — buckets are full-fidelity for ANY board)
    bnr: usize,
    canonical_flops: Vec<[Card; 3]>,
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl ConnDecider {
    /// Load the sharded blueprint + reconstruct the cell layout. `(np, nraises, nb,
    /// maxna)` = the solve config (6, 1, 200, 3 for blueprint_conn_v1). `gs14_dir` =
    /// the full 49×48 GS14 bucket cache.
    pub fn load(bp_dir: &str, gs14_dir: &str, np: usize, nraises: usize, nb: usize, maxna: usize) -> std::io::Result<Self> {
        let bp = ShardedConnBlueprint::load(bp_dir, np, nraises, nb, 16, 16, maxna)?;
        let layout = ConnCellLayout::build(np, nraises, nb);
        let ranges = vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6];
        let canonical_flops = PreflopChanceTable::new(6, ranges).canonical_flops;
        Ok(ConnDecider { bp, layout, gs14_dir: gs14_dir.to_string(), nb, bnt: 49, bnr: 48, canonical_flops })
    }

    /// Select the `(live, commit, pot)` cell: exact match, else nearest among
    /// same-live cells by |Δcommit| + |Δpot| (SPR-ish fallback).
    fn select_cell(&self, live: u8, commit: i32, pot: i32) -> Option<(u8, i32, i32)> {
        if self.layout.key_idx.contains_key(&(live, commit, pot)) {
            return Some((live, commit, pot));
        }
        self.layout
            .keys
            .iter()
            .filter(|k| k.0 == live)
            .min_by_key(|k| (k.1 - commit).abs() + (k.2 - pot).abs())
            .copied()
    }

    /// Serve a decision. Returns None for spots this path can't serve (turn/river,
    /// unmapped node, missing cache) so the caller can fall back.
    pub fn decide(&self, req: &DecideRequest) -> Option<DecideResponse> {
        let t0 = std::time::Instant::now();
        let hero = (req.hero_cards[0] as Card, req.hero_cards[1] as Card);
        let hist: Vec<(u8, i32)> = req.street_actions.iter().map(|a| (a.label, a.to_total as i32)).collect();

        let (street, raw): (&str, Vec<(u8, i32, f32)>) = if req.board.is_empty() {
            ("preflop", self.bp.preflop_action_dist(hero, &hist)?)
        } else if req.board.len() == 3 {
            // FLOP: the cell root is the flop, so the flop street_actions map directly.
            let flop_id = req.flop_id as usize;
            if flop_id >= self.canonical_flops.len() {
                return None;
            }
            let canonical = self.canonical_flops[flop_id];
            let maps = load_gs14_cache(
                &gs14_cache_path(&self.gs14_dir, flop_id, self.nb, self.bnt, self.bnr),
                self.nb, self.bnt, self.bnr,
            )?;
            let fl = FlopLayout::for_canonical(canonical, self.bnt, self.bnr);
            let bucket = fl.flop_bucket(&maps, hero)? as usize;
            let post = self.bp.postflop_cum(flop_id).ok()?;
            let (live, commit, pot) = self.select_cell(req.live, req.commit_entry as i32, req.pot_entry as i32)?;
            ("flop", self.layout.flop_action_dist(&post, live, commit, pot, bucket, &hist)?)
        } else {
            // turn/river: needs full postflop betting path (API extension).
            return None;
        };

        // Build action probs (normalize defensively) + sample a deterministic choice.
        let z: f32 = raw.iter().map(|(_, _, p)| p).sum::<f32>().max(1e-12);
        let actions: Vec<ActionProb> = raw
            .iter()
            .map(|&(label, amount, prob)| ActionProb {
                label,
                action: action_name(label).to_string(),
                amount,
                prob: prob / z,
            })
            .collect();
        if actions.is_empty() {
            return None;
        }
        let mut rng = req.seed.unwrap_or(0xA17C0DE);
        let mut x = (splitmix(&mut rng) % 1_000_000) as f32 / 1_000_000.0;
        let mut sel = actions.len() - 1;
        for (a, ap) in actions.iter().enumerate() {
            if x < ap.prob {
                sel = a;
                break;
            }
            x -= ap.prob;
        }
        Some(DecideResponse {
            street: street.to_string(),
            live: req.live,
            chosen: actions[sel].clone(),
            actions,
            search_ms: t0.elapsed().as_millis() as u64,
            paired: false,
        })
    }
}
