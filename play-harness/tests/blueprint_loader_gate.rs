//! H3 gate 1: SSBP1 loader round-trip against a REAL banked artifact
//! (the challenger run banks onto disk as this test runs). Structural
//! invariants: section lengths must equal the strides derived from the
//! rebuilt (tree, table, bucketing) — i.e., the artifact indexes
//! exactly as the solver banked it — maps in range, and normalized
//! strategies are valid distributions wherever cum mass exists.

use play_harness::blueprint::{build_oracle_tree, Blueprint};

#[test]
fn loader_roundtrip_against_banked_artifact() {
    let path = std::env::var("BP_ARTIFACT")
        .unwrap_or_else(|_| "../blueprint_out_b10_4x4/flop_0000.bp".into());
    let path = path.as_str();
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: no banked artifact yet at {path}");
        return;
    }
    let bp = Blueprint::load(path).expect("load");
    let tree = build_oracle_tree();
    let idx = bp.indexer(&tree);

    // Stride round-trip: banked section lengths == solver layout.
    assert_eq!(bp.cum_flop.len(), idx.flop_stride(), "cum_flop stride");
    assert_eq!(
        bp.cum_turn.len(),
        bp.turns.len() * idx.turn_stride(),
        "cum_turn stride (n_turn × per-outcome)"
    );
    assert_eq!(
        bp.cum_river.len(),
        bp.turns.len() * idx.max_river_outcomes() * idx.river_stride(),
        "cum_river stride"
    );

    // Maps in range; every live hand mapped.
    for &m in &bp.bk.flop_map {
        assert!(m == u16::MAX || (m as usize) < bp.nb);
    }
    let live = bp.bk.flop_map.iter().filter(|&&m| m != u16::MAX).count();
    assert_eq!(live, bp.nh, "flop map total (no board conflicts at flop)");

    // Normalized root-zone strategies are distributions where mass
    // exists (cum >= 0 and rows sum to ~1 after normalization).
    use solver_core::tree::flat::MAX_NA_POSTFLOP;
    let mut rows = 0;
    for &nid in &tree.decision_node_ids {
        let Some(local) = idx.flop_local_offset_at(nid as usize) else { continue };
        let na = tree.nodes[nid as usize].num_children as usize;
        let off = local * MAX_NA_POSTFLOP * bp.nb;
        for b in 0..bp.nb {
            let sum: f32 = (0..na).map(|a| bp.cum_flop[off + a * bp.nb + b]).sum();
            if sum > 0.0 {
                for a in 0..na {
                    let v = bp.cum_flop[off + a * bp.nb + b];
                    assert!(v >= 0.0, "negative cum mass");
                    assert!(v / sum <= 1.0 + 1e-6);
                }
                rows += 1;
            }
        }
    }
    assert!(rows > 0, "no strategy mass anywhere");
    eprintln!(
        "loader round-trip OK: flop {:?}, {} turns, B={}, nh={}, {} flop strategy rows",
        bp.flop, bp.turns.len(), bp.nb, bp.nh, rows
    );
}
