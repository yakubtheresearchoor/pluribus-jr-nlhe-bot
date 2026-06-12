use crate::card::Card;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Action {
    #[default]
    None,
    Fold,
    Check,
    Call,
    Bet(i32),
    Raise(i32),
    AllIn(i32),
    Chance(Card),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum BoardState {
    #[default]
    Flop = 0,
    Turn = 1,
    River = 2,
    // Preflop = 3 (appended for backward compatibility with hardcoded
    // board_state == 1/2 checks in gpu/context.rs and flop_start_vector_cfr.rs).
    // The repr value (3) is distinct from the street ordinal (0); see
    // street() and num_remaining_streets() for chronological semantics.
    Preflop = 3,
}

impl BoardState {
    /// Chronological street ordinal: Preflop=0, Flop=1, Turn=2, River=3.
    /// Distinct from the repr value (where Preflop=3 for backward compat).
    pub fn street(&self) -> u8 {
        match self {
            BoardState::Preflop => 0,
            BoardState::Flop => 1,
            BoardState::Turn => 2,
            BoardState::River => 3,
        }
    }

    /// Number of streets remaining INCLUDING this one.
    /// Preflop=4, Flop=3, Turn=2, River=1.
    pub fn num_remaining_streets(&self) -> i32 {
        match self {
            BoardState::Preflop => 4,
            BoardState::Flop => 3,
            BoardState::Turn => 2,
            BoardState::River => 1,
        }
    }

    /// Next street in chronological order, or None if River (no next street).
    pub fn next(&self) -> Option<BoardState> {
        match self {
            BoardState::Preflop => Some(BoardState::Flop),
            BoardState::Flop => Some(BoardState::Turn),
            BoardState::Turn => Some(BoardState::River),
            BoardState::River => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BetSize {
    PotRelative(f64),
    PrevBetRelative(f64),
    Additive(i32, i32),
    Geometric(i32, f64),
    AllIn,
}

impl PartialOrd for BetSize {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.to_pot_fraction().partial_cmp(&other.to_pot_fraction())
    }
}

impl BetSize {
    fn to_pot_fraction(&self) -> f64 {
        match self {
            BetSize::PotRelative(f) => *f,
            BetSize::AllIn => f64::INFINITY,
            BetSize::PrevBetRelative(f) => *f,
            BetSize::Additive(v, _) => *v as f64 / 100.0,
            BetSize::Geometric(n, max) => max.min(*n as f64 + 1.0),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BetSizeOptions {
    pub bet: Vec<BetSize>,
    pub raise: Vec<BetSize>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DonkSizeOptions {
    pub donk: Vec<BetSize>,
}

/// THE single authoritative chip-unit ↔ big-blind conversion
/// (pinned 2026-06-12, user directive: "our units should be in bb").
///
/// All integer money in TreeConfig / FlatTree / harness settlement is
/// in CHIP UNITS, where 1 bb = `UNITS_PER_BB` units (2, so the small
/// blind = 1 unit stays integral). Every human-facing report (EVs,
/// match results, exploitability) must divide by this constant —
/// nothing else may define its own conversion. The standing
/// `bb_units_gate` test pins the production configs against it.
///
/// Oracle flop game in bb: ante 2 units = 1 bb each, starting pot
/// 12 units = 6 bb, stacks 94 units = 47 bb, flop 1.0x-pot bet
/// 12 units = 6 bb.
pub const UNITS_PER_BB: i32 = 2;

#[derive(Debug, Clone)]
pub struct TreeConfig {
    pub num_players: u8,
    pub initial_state: BoardState,
    pub starting_pot: i32,
    pub starting_stacks: Vec<i32>,
    pub initial_contributions: Vec<i32>,
    pub rake_rate: f64,
    pub rake_cap: f64,
    pub bet_sizes: BetSizeOptions,
    pub add_allin_threshold: f64,
    pub force_allin_threshold: f64,
    pub merging_threshold: f64,
    /// Player index of the dealer button. Other positions are derived
    /// by rotation:
    ///   - SB = (button + 1) mod num_players
    ///   - BB = (button + 2) mod num_players
    ///   - UTG = (button + 3) mod num_players (multiway only)
    ///
    /// Preflop first actor:
    ///   - HU (np=2): button itself acts first (button = SB acts first preflop)
    ///   - Multiway (np>=3): UTG acts first
    ///
    /// Postflop first actor: SB = (button + 1) mod np acts first (HU collapses
    /// to BB = (button + 1) mod 2 since SB = button at HU).
    ///
    /// `None` means "legacy inference: highest-indexed active player is
    /// the button". This is HU-correct (with the convention that the
    /// higher-indexed seat is the button) but MULTIWAY-INCORRECT (it
    /// makes the button act first preflop instead of UTG). All multiway
    /// preflop callers MUST set this explicitly per the lead's directive
    /// (2026-06-04): positions are what matter for the blueprint, and
    /// the button is a parameter that determines who is in each
    /// position, not a fixed convention. Existing HU code paths leave
    /// it `None` for backward compatibility; new 6-max code paths set
    /// it explicitly.
    pub button_player: Option<u8>,
}
