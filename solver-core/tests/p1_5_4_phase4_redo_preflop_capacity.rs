//! Phase 4 redo — preflop CAPACITY smoke test (item 1 of the post-bank
//! dot-the-i list).
//!
//! The user flagged that the postflop side was rigorously bootstrapped
//! (commits 239be58 → 0b3ac50, landing at MAX_NA_POSTFLOP = 4 on a verified
//! shape with cross-action-space cost ≤ 0.001%) but the PREFLOP side
//! (MAX_NA_PREFLOP = 16) was set by reference to Pluribus (up to 14 raise
//! sizes) and never empirically bootstrapped the same way. The user's
//! framing: "preflop is cheap (CPU-only, small tree, no multiway board
//! runout), so over-provisioned preflop costs almost nothing; the missing
//! piece is confirming 16 is sufficient (not too LEAN)."
//!
//! This file delivers the PARTIAL bootstrap that fits in a single
//! iteration: it confirms the tree-builder's STRUCTURAL CAPACITY at
//! MAX_NA_PREFLOP = 16 (a preflop tree with 14 raise sizes builds cleanly
//! and the per-stage cap assertion in builder.rs doesn't fire). The FULL
//! bootstrap — solving the preflop game to convergence and observing
//! which raise sizes the equilibrium actually uses — is documented as a
//! deferred follow-up with a concrete plan below.
//!
//! The deferred full bootstrap is the proper analogue of what Phase 4
//! did for postflop. Without it, MAX_NA_PREFLOP = 16 is grounded in
//! Pluribus's empirical work (which is reasonable but not specifically
//! validated against our deployment) and the structural capacity check
//! (which confirms the tree machinery supports up to 14 raise sizes
//! without crashing).
//!
//! === Deferred FULL PREFLOP BOOTSTRAP plan ===
//!
//! Per the existing preflop infrastructure (PreflopVectorCfr,
//! PreflopChanceTable, make_production_terminal_value_fn_multiway,
//! UnabstractedPostflopOracle — see e.g.
//! p1_5_4_step1_reduced_scale_endtoend.rs for the wiring pattern):
//!
//! 1. Build a rich preflop tree with the MAXIMUM raise sizes that
//!    MAX_NA_PREFLOP supports: 14 = MAX_NA_PREFLOP - 2 (= fold + call +
//!    14 raises). Use raise sizes spanning 0.5× pot to e.g. 12× pot (the
//!    Pluribus range), at increments of ≈0.5-1×.
//!
//! 2. Build a postflop oracle (UnabstractedPostflopOracle or a stub) and
//!    a PreflopChanceTable with realistic class_weights (the existing
//!    test uses 1/NUM_PREFLOP_CLASSES uniform; a real bootstrap should
//!    use card-distribution weights).
//!
//! 3. Run PreflopVectorCfr::run_one_iteration_subset for ~50-100 iters
//!    (preflop is small, fast — minutes per iter). Track convergence
//!    via self-exploitability (analogous to the postflop self-expl floor
//!    of 0.005% pot used in phase4_redo_measurement.rs).
//!
//! 4. After convergence: walk preflop player nodes, normalize cum_strategy
//!    to per-(infoset, hand-class) σ, aggregate σ by raise-size action
//!    index across all preflop infosets weighted by reach. Output a
//!    histogram: "raise size R was chosen with σ=X across the equilibrium."
//!
//! 5. Verdict logic:
//!    - If only K << 14 raise sizes have non-trivial σ (say K ≤ 8), then
//!      MAX_NA_PREFLOP could shrink to K+2 = e.g. 10 with no quality
//!      loss. The bank moves down.
//!    - If 12-14 raise sizes all have meaningful σ (the abstraction is
//!      "saturated"), bump MAX_NA_PREFLOP temporarily higher (e.g. 20)
//!      and re-run with 18 raise sizes to see if any sizes ABOVE the
//!      original 14 get used. If yes, 16 is too LEAN. If no, 16 is
//!      sufficient.
//!
//! 6. Bank: update flat.rs comment with the empirical preflop verdict,
//!    matching the postflop framing.
//!
//! Cost: ~3-6 hours of compute + a few hundred LOC of glue. The deferred-
//! work flag in flat.rs is updated below.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA_PREFLOP;

#[test]
fn phase4_redo_preflop_structural_capacity_smoke() {
    eprintln!("\n=== Phase 4 REDO: preflop STRUCTURAL CAPACITY smoke test ===");
    eprintln!("Goal: confirm tree builder accepts MAX_NA_PREFLOP - 2 raise sizes");
    eprintln!("      without firing the per-stage cap assertion. This is a");
    eprintln!("      sanity check, NOT a full bootstrap (see file docstring).");
    eprintln!("MAX_NA_PREFLOP = {}", MAX_NA_PREFLOP);

    // Maximum raise sizes that fit in MAX_NA_PREFLOP. fold + call accounts
    // for 2 slots, so the cap is MAX_NA_PREFLOP - 2 raise sizes per node.
    let max_raise_count = MAX_NA_PREFLOP.saturating_sub(2);
    eprintln!("Building preflop tree with {} raise sizes ({} - fold - call)", max_raise_count, MAX_NA_PREFLOP);

    // Pluribus-ish range: 0.5× pot to (max_raise_count × 0.5 + 0.5)× pot.
    // For MAX_NA = 16 → 14 raises spanning 0.5×p to 7.5×p in 0.5×p steps.
    let raises: Vec<BetSize> = (0..max_raise_count)
        .map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64))
        .collect();
    eprintln!("Raise sizes ({}): {:?}", raises.len(),
        raises.iter().take(5).chain(raises.iter().rev().take(2)).collect::<Vec<_>>());

    // Standard 6-max preflop config (per p1_5_4_step1_reduced_scale_endtoend.rs).
    let pre_cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; 6],
        initial_contributions: vec![1, 2, 0, 0, 0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: raises,
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
    };

    let preflop_tree = build_tree(&pre_cfg);
    match preflop_tree {
        Ok(tree) => {
            eprintln!("✓ Preflop tree built successfully:");
            eprintln!("    nodes = {}", tree.num_nodes());
            eprintln!("    infosets = {}", tree.num_infosets);
            eprintln!("    structural capacity at MAX_NA_PREFLOP = {} CONFIRMED.", MAX_NA_PREFLOP);
        }
        Err(e) => {
            panic!("Preflop tree FAILED to build at MAX_NA_PREFLOP = {} with {} raises: {:?}\n\
                    This means the per-stage cap fires; the bank value is structurally over-provisioned\n\
                    relative to the tree builder's expectations or this test's raise set is mis-sized.",
                MAX_NA_PREFLOP, max_raise_count, e);
        }
    }
    eprintln!("\nThis is a STRUCTURAL CAPACITY check only.");
    eprintln!("The empirical bootstrap CLOSED 2026-06-10 — see");
    eprintln!("p1_5_4_phase4_redo_preflop_bootstrap.rs: 4-5 of 14 sizes carry ~98% of");
    eprintln!("raise mass at both fidelities → MAX_NA_PREFLOP = 16 is NOT too lean.");
}
