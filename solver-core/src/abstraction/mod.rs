//! Board abstraction layer.
//!
//! This module is the home for board-abstraction machinery — currently
//! just lossless flop suit-isomorphism canonicalization (which preflop
//! needs to make the 22,100 flop combinations tractable as a ~1,755
//! canonical-board chance step), with room for additional abstractions
//! (chosen-subset, postflop bucketing) as the project grows.
//!
//! VALIDATION DISCIPLINE: each abstraction in this module is validated
//! independently against the un-abstracted reference at f32 floor on
//! small cases, per the maintenance principle established in the
//! validation arc. Lossless abstractions (flop_isomorphism) need
//! correctness validation only. Lossy abstractions (future bucketing)
//! need correctness AND EV-cost measurement.

pub mod flop_isomorphism;
