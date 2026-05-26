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
}

impl BoardState {
    pub fn street(&self) -> u8 {
        *self as u8
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
}
