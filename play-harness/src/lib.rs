//! Play harness: duplicate-play matches between blueprint agents,
//! scored through the clean-rules independent engine. Carries the
//! deferred A/Bs (map family, B count, cell, runout residual, GPU
//! challenger), the Monte-Carlo value audit, and the analytic anchors.
//!
//! Card-encoding boundary: solver cards are `suit * 13 + rank`-style?
//! NO — conversions live in `convert` and are pinned by a test against
//! both crates' string formats; nothing else may convert ad hoc.

pub mod convert;
pub mod blueprint;
