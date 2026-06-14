//! PRODUCTION-FIDELITY DECISION TABLE (assembled from banked numbers — NO fresh
//! production solves). Cost = banked GPU per-family B8 prices × banked cost-ladder
//! multipliers (nh=1176). Quality = the research-tournament RELATIONSHIP (read as
//! "is finer meaningfully better than cheaper", not absolute — research magnitudes
//! are inflated, only the relationship transfers; absolute "does it win" is graded
//! live play's call). Reach-weighted architecture: live-6 = equity ROLLOUT (free),
//! live-5 + live-2 fixed at B8, live-3/4 = the varying axis (where reach + spend are).

#[test]
fn production_fidelity_decision_table() {
    // Banked GPU per-family blueprint cost at B8 (hours). live-6 = ROLLOUT = 0.
    let (l2, l3, l4, l5, l6) = (0.05f64, 0.36, 1.6, 16.7, 0.0);
    // Banked cost ladder (CPU per-iter, nh=1176): B8 10.14 / B10 37.94 / B15 400 /
    // B20 2195 / B25 8453 s/iter → multiplier vs B8 (≈GPU B-scaling; GPU B8→B10
    // measured ~3.9× ≈ CPU 3.74×; high-B GPU multiplier is the one estimated cell).
    let ladder = [("B8", 147, 1.0f64), ("B10", 118, 3.74), ("B15", 78, 39.45),
                  ("B20", 59, 216.5), ("B25", 49, 833.6)];
    // Research distance-from-exact (de-confounded, equal nh=10, % pot) for the
    // common families — the RELATIONSHIP anchor (super-linear; finer buys a lot
    // early, diminishing). Production B is FAR coarser than these research ratios,
    // so this sizes the SHAPE, not the absolute. live-3: 1.4:1→14% 2:1→27% 3.3:1→170%.
    let fixed = l2 + l5 + l6; // 16.75h: live-2@B8 + live-5@B8 + live-6 rollout
    eprintln!("\n═══ PRODUCTION-FIDELITY DECISION TABLE (reach-weighted, live-6 rollout) ═══");
    eprintln!("fixed across rows: live-2@B8 {l2}h + live-5@B8 {l5}h + live-6 rollout 0h = {fixed}h");
    eprintln!("varying axis: live-3/4 fidelity (the reach + spend concentration)\n");
    eprintln!("live-3/4 @ (prod ratio) | GPU-h total | vs 24h | quality vs cheaper row");
    let mut prev_h = 0.0f64;
    for (i, (name, ratio, mult)) in ladder.iter().enumerate() {
        let var = (l3 + l4) * mult;
        let total = fixed + var;
        let line24 = if total <= 24.0 { "UNDER".to_string() }
            else { format!("{:.0}× over (~{:.0}d)", total / 24.0, total / 24.0) };
        let qual = match i {
            0 => "FLOOR — coarsest (147:1); research worst-distance regime".to_string(),
            1 => "marginal step over B8 (118:1 ≈ B8); small fidelity gain".to_string(),
            2 => "meaningfully finer (78:1); super-linear distance drop in research".to_string(),
            3 => format!("+{:.0}h over B15 for a DIMINISHING fidelity step (cost cliff)", total - prev_h),
            _ => format!("+{:.0}h over B20 for MARGINAL gain", total - prev_h),
        };
        eprintln!("  {name} ({ratio}:1) | {total:8.1}h | {line24:>14} | {qual}");
        prev_h = total;
    }
    eprintln!("\nQUALITY-PER-HOUR (where the tradeoff flattens):");
    eprintln!("  B8→B10:  +5.4h  small fidelity step (118:1≈147:1)");
    eprintln!("  B10→B15: +70h   the meaningful fidelity jump (research: super-linear, big early gain)");
    eprintln!("  B15→B20: +347h  DIMINISHING — 4.7× the B15 cost for a smaller fidelity step");
    eprintln!("  B20→B25: +1210h MARGINAL");
    eprintln!("\nREAD (decision is the lead's): the 24h floor (B8/B10) is the cheap reach-weighted base;");
    eprintln!("  B15 (~94h/4d) is the 'over-24h-for-real-gain' candidate (in scope per the lead); B20+");
    eprintln!("  (~18d+) is weeks for diminishing return (out of scope). live-5 sensitivity: @B10 adds");
    eprintln!("  ~+46h (16.7→62.8) for the modest live-5 residual — optional, rare family. CAVEATS:");
    eprintln!("  quality is research-RELATIONSHIP (absolute = graded live play); production B coarser");
    eprintln!("  than research tested; exploitability ranks vs exploiter, soft field is matchup-specific.");
}
