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

/// A GAME CLASS (table structure + rake), as distinct from a tree
/// abstraction (bet sizes, runout sampling). Blueprints are solved per
/// game class; dollar stakes are NOT an axis (poker is scale-invariant
/// in money — only bb-denominated structure matters).
#[derive(Debug, Clone, PartialEq)]
pub struct GameSpec {
    pub num_players: u8,
    /// All money in chip units (UNITS_PER_BB units = 1 bb).
    pub sb: i32,
    pub bb: i32,
    pub ante: i32,
    /// Starting TOTAL stack per player (blinds come out of this).
    pub stack: i32,
    /// Rake fraction of the pot (e.g. 0.05) and cap in chip units.
    pub rake_rate: f64,
    pub rake_cap: i32,
    /// No flop, no drop: pots that never see a flop are unraked.
    pub no_flop_no_drop: bool,
}

impl GameSpec {
    /// Derive the PREFLOP-rooted TreeConfig for this game class.
    /// Conventions (all pinned by the tree-correctness + bb-units
    /// gates):
    ///   - button = seat np−1, SB = seat 0, BB = seat 1 (UTG = seat 2
    ///     acts first preflop via `button_player`).
    ///   - starting_pot = ante money only (DEAD); the blinds are LIVE
    ///     first-round commits in `initial_contributions` (2026-06-12
    ///     semantics — the bootstrap's `starting_pot: 3` double-counted).
    ///   - EQUAL TOTAL stacks: `starting_stacks[p] = stack − contrib[p]`
    ///     so max_committable = `stack` for every seat (the bootstrap's
    ///     flat `[100;6]` gave the blinds deeper totals).
    ///   - Rake fields threaded from the spec. Under no-flop-no-drop
    ///     the preflop-zone terminals (folds) are unraked BY
    ///     CONSTRUCTION in the layered architecture: rake is applied in
    ///     postflop terminal evaluation only (flop_seen = true there).
    pub fn preflop_tree_config(&self, bet_sizes: BetSizeOptions) -> TreeConfig {
        let np = self.num_players as usize;
        let mut contrib = vec![self.ante; np];
        contrib[0] += self.sb;
        contrib[1] += self.bb;
        TreeConfig {
            num_players: self.num_players,
            initial_state: BoardState::Preflop,
            starting_pot: 0,
            starting_stacks: contrib.iter().map(|&c| self.stack - c).collect(),
            initial_contributions: contrib,
            rake_rate: self.rake_rate,
            rake_cap: self.rake_cap as f64,
            bet_sizes,
            add_allin_threshold: 1.0,
            force_allin_threshold: 1.0,
            merging_threshold: 0.0,
            button_player: Some(self.num_players - 1),
            max_bets_per_street: None,
        }
    }

    /// Derive a FLOP-START TreeConfig for one preflop→flop SEAM CELL:
    /// `live` players reach the flop having each committed `commit`
    /// units, with `pot` total units in the middle (pot ≥ live×commit;
    /// the excess is dead money from folders/antes). All flop money is
    /// dead at flop start (contributions 0, pot additive), stacks are
    /// what remains behind. Rake threaded; flop trees evaluate
    /// terminals with flop_seen = true (the flop has been dealt by
    /// definition of the seam).
    pub fn flop_seam_config(
        &self,
        live: u8,
        commit: i32,
        pot: i32,
        bet_sizes: BetSizeOptions,
    ) -> TreeConfig {
        assert!(live >= 2 && live <= self.num_players);
        assert!(pot >= live as i32 * commit, "pot must cover live commits");
        assert!(commit <= self.stack);
        TreeConfig {
            num_players: live,
            initial_state: BoardState::Flop,
            starting_pot: pot,
            starting_stacks: vec![self.stack - commit; live as usize],
            initial_contributions: vec![0; live as usize],
            rake_rate: self.rake_rate,
            rake_cap: self.rake_cap as f64,
            bet_sizes,
            add_allin_threshold: 1.0,
            force_allin_threshold: 1.0,
            merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,
        }
    }
}

/// PRODUCTION GAME CLASS v1 (pinned with user 2026-06-12):
/// 6-max, no ante, blinds 0.5bb/1bb, 100bb starting stacks,
/// 5% rake capped at 10bb, no flop no drop.
///
/// Single source of truth — runners derive TreeConfigs from this and
/// the harness derives its clean-rules TableConfig from this; neither
/// may restate the numbers. Gated by `bb_units_gate` (numeric pins)
/// and the harness `production_game_gate` (behavioral pins through
/// the independent rules engine, incl. no-flop-no-drop and the cap).
pub fn production_game_v1() -> GameSpec {
    GameSpec {
        num_players: 6,
        sb: 1,                       // 0.5 bb
        bb: UNITS_PER_BB,            // 1 bb
        ante: 0,
        stack: 100 * UNITS_PER_BB,   // 100 bb
        rake_rate: 0.05,
        rake_cap: 10 * UNITS_PER_BB, // 10 bb
        no_flop_no_drop: true,
    }
}

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
    /// RAISE-DEPTH cap (2026-06-12, Pluribus-style): maximum number of
    /// aggressive actions (bets + raises) per STREET; once reached,
    /// only fold/call/check remain. `None` = unlimited. This caps the
    /// raise-WAR depth (how many times betting re-escalates in a
    /// round), NOT the raise sizes per decision — the size menu stays
    /// at full width at every decision that is still allowed to raise.
    /// Blinds are not actions and don't count; an all-in ends the
    /// aggression chain via allin_flag regardless of this cap.
    ///
    /// DISCIPLINE PIN: the production value is MEASURED, not assumed —
    /// action-abstraction changes get validated (the postflop MAX_NA
    /// saga: pruning the wrong action creates catastrophic
    /// exploitability). Depth 3 vs 4 is decided by the exploitability
    /// A/B; pick the smallest cap that is defense-complete.
    pub max_bets_per_street: Option<BetCap>,
}

/// See `TreeConfig::max_bets_per_street`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetCap {
    /// Maximum bets + raises per street.
    pub cap: u8,
    /// Apply the cap to ONE seat only (`None` = all seats — the
    /// production shape). The single-seat form builds the A/B's
    /// asymmetric measurement game: the capped seat cannot escalate
    /// past `cap` while the exploiter seat keeps full depth, so every
    /// defensive infoset the capped seat needs (facing deep raises)
    /// stays IN the tree and the equilibrium is well-defined — no
    /// action-translation lift required to measure the cap's cost.
    pub seat: Option<u8>,
}

impl BetCap {
    pub fn all(cap: u8) -> Option<BetCap> {
        Some(BetCap { cap, seat: None })
    }
    pub fn seat_only(cap: u8, seat: u8) -> Option<BetCap> {
        Some(BetCap { cap, seat: Some(seat) })
    }
}
