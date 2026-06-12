//! Clean-room No-Limit Hold'em rules engine: dealer, betting state
//! machine, side-pot settlement, and hand evaluator — written fresh
//! from the rules of poker. SHARES ZERO CODE with solver-core (the
//! crate has no dependencies; see Cargo.toml for the mandate). The
//! play harness scores every self-play hand through this engine, so
//! any systematic showdown bug in the solver's terminal machinery
//! shows up as a cross-implementation disagreement (a fire alarm)
//! instead of being invisible because all six bots share it.
//!
//! Design rule: simple and obviously-correct over fast. The evaluator
//! scores every 5-card subset of the 7 cards; the settlement builds
//! layered side pots the way a floor manager would.
//!
//! Card encoding (deliberately NOT the solver's): `card = rank * 4 +
//! suit`, rank 0..=12 meaning 2..=Ace, suit 0..=3 (clubs, diamonds,
//! hearts, spades). Conversions happen at the harness boundary.

pub mod eval;
pub mod table;

pub use eval::{best5, HandRank};
pub use table::{Action, HandState, Settlement, Street, TableConfig};
