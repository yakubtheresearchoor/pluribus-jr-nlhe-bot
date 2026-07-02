//! Play harness: duplicate-play matches between blueprint agents,
//! scored through the clean-rules independent engine. Carries the
//! deferred A/Bs (map family, B count, cell, runout residual, GPU
//! challenger), the Monte-Carlo value audit, and the analytic anchors.
//!
//! Status: H1 (clean-rules engine) + H2 (fire alarm) + H3 slice 1
//! (SSBP1 loader, round-trip gated) DONE. The fire alarm's first run
//! caught the rank-truncation bug (see 1ee5c98).
//!
//! ═══ H3 DESIGN (pinned 2026-06-12, before the agent is built) ═══
//!
//! v1 match game: the blueprint's OWN flop-start game (oracle tree:
//! 6p, pot 12, stacks 94, 1.0×pot bet, no raises), on the blueprint's
//! flop. The oracle tree IS the state machine the agents traverse;
//! ALL money math and showdown scoring happen in clean-rules (a
//! `settle_pots`-style free function refactored out of
//! HandState::settle — never re-implement pot logic here).
//!
//! TWO DEAL MODES, deliberately distinct:
//!   - AUDIT mode (H4): turn/river dealt only from the blueprint's
//!     SAMPLED runout set with the solver's chance probabilities —
//!     this is exactly the game the solver solved, so sampled EVs are
//!     comparable to solver root CFVs within error bars.
//!   - PLAY mode (A/Bs): full-deck runouts + RUNOUT PROJECTION v1
//!     (named approximation): dealt turn maps to the sampled turn
//!     nearest by rank (ties: matching suit-class relative to board),
//!     same per-turn for rivers. The audit-vs-play EV gap MEASURES the
//!     runout residual — one of the five A/Bs, for free.
//!
//! Duplicate-play control: same shuffled deck replayed with seat
//! rotation (each strategy plays every seat on the same cards);
//! per-deck paired differences give the variance-reduced A/B stat.
//!
//! Hand→index: h = position of (c1,c2) among the table's
//! `hand_cards` (the loader rebuilds the table, so the mapping is the
//! solver's own). Bucket: banked maps per (street, runout).
//!
//! Card-encoding boundary: conversions live in `convert`, pinned by
//! test; nothing else may convert ad hoc.

pub mod blueprint;
pub mod runtime_spec;
pub use runtime_spec::{runtime_game_spec, set_runtime_game_spec};
pub mod eqr;
pub mod preflop_oracle;
pub mod live2_bank;
pub mod match_play;
pub mod pluribus_play;
pub mod api;
pub mod api_conn;
pub mod convert;
pub mod experiment;
pub mod v1_seam;
pub mod preflop_player;
pub mod pool_preflop;
pub mod full_hand;
