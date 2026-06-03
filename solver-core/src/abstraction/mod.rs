//! Board and hand abstraction layer.
//!
//! Current contents:
//! - `flop_isomorphism` — lossless suit-canonicalization of the 22,100
//!   flops to 1,755 canonical forms, with orbit weights × 4 / × 12 / × 24
//! - `preflop_class` — lossless 169-class preflop hand layout (13 pairs
//!   + 78 suited + 78 offsuit), with per-flop expansion maps mapping
//!   each class to its surviving 2-card combos under a given flop
//!
//! These two pieces together make the preflop → flop chance integration
//! EXACT against the un-canonicalized 22,100 × 1326 enumeration. The
//! flop iso alone (at full nh = 1326) introduces an orbit-weighted
//! approximation error because the joint flop-hand suit symmetry
//! requires the hand to be in its canonical class too. Validation
//! at #44 anchors at max_diff = 0 against the joint enumeration.
//!
//! VALIDATION DISCIPLINE: each abstraction in this module is validated
//! independently against the un-abstracted reference at f32 floor on
//! small cases, per the maintenance principle established in the
//! validation arc. Lossless abstractions (flop_isomorphism,
//! preflop_class) need correctness validation only. Lossy abstractions
//! (future bucketing) need correctness AND EV-cost measurement.

pub mod flop_isomorphism;
pub mod preflop_class;
