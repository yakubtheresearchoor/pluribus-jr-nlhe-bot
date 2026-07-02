//! Process-wide RUNTIME GAME SPEC — the game economics (rake rate/cap, ante,
//! stack depth, blinds, no-flop-no-drop) every real-time decision path builds
//! its seam trees and values its terminals with.
//!
//! Set ONCE at startup from the loaded blueprint's `game.json` manifest
//! (`ConnDecider::load` / the server), because the search must refine under the
//! SAME economics its blueprint was solved with — stakes vary rake caps and
//! antes, and a 50bb blueprint needs 50bb seam trees. Falls back to
//! `production_game_v1` (the NL10 class every pre-manifest blueprint was
//! solved under) when never set.

use solver_core::tree::action::{production_game_v1, GameSpec};
use std::sync::OnceLock;

static RUNTIME_SPEC: OnceLock<GameSpec> = OnceLock::new();

/// Install the runtime game spec (first call wins; later calls return false
/// and are ignored — one process serves one game class).
pub fn set_runtime_game_spec(spec: GameSpec) -> bool {
    RUNTIME_SPEC.set(spec).is_ok()
}

/// The game spec every decision path must use. Defaults to production v1 when
/// no blueprint manifest installed one (legacy dirs).
pub fn runtime_game_spec() -> &'static GameSpec {
    RUNTIME_SPEC.get_or_init(production_game_v1)
}

#[cfg(test)]
mod tests {
    use solver_core::tree::action::{production_game_v1, GameSpec};

    #[test]
    fn game_json_round_trips() {
        let spec = GameSpec {
            num_players: 6, sb: 1, bb: 2, ante: 1, stack: 100, // 50bb, with ante
            rake_rate: 0.04, rake_cap: 12, no_flop_no_drop: true,
        };
        let s = spec.to_json();
        let back = GameSpec::from_json(&s).expect("parse");
        assert_eq!(back, spec, "round-trip mismatch: {s}");
        // v1 too
        let v1 = production_game_v1();
        assert_eq!(GameSpec::from_json(&v1.to_json()).unwrap(), v1);
    }

    #[test]
    fn seam_tree_inherits_manifest_economics() {
        use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState};
        // A different-stakes spec: the seam TreeConfig must carry ITS rake and
        // stack (this is what makes GPU/CPU terminal rake + all-in sizing right).
        let spec = GameSpec {
            num_players: 6, sb: 1, bb: 2, ante: 0, stack: 100,
            rake_rate: 0.03, rake_cap: 8, no_flop_no_drop: true,
        };
        let cfg = spec.street_seam_config(BoardState::Flop, 3, 6, 18,
            BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] });
        assert_eq!(cfg.rake_rate, 0.03);
        assert_eq!(cfg.rake_cap, 8.0);
        assert_eq!(cfg.starting_stacks, vec![100 - 6; 3]);
    }
}
