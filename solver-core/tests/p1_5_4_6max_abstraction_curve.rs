// Re-baseline 6-max preflop tree size across the action-abstraction curve, on
// the CORRECTED foundation (post round-completion fix, MAX_DEPTH loud).
//
// The 2,649-node "simplest" measurement is the FLOOR (1 bet, 0 raise). The
// production decision on "is preflop-tree abstraction still mandatory at
// 6-max" must be made on a realistic abstraction, not the floor. This sweep
// maps the curve so the architectural conclusion lands on data.
//
// Reference points for context:
//   - HU Option-B postflop (15,029 nodes) — the largest tree we've solved end to end
//   - 6-max wrong-game simplest (29,882 nodes) — the pre-fix measurement that
//     drove the "abstraction is mandatory" conclusion in slice 7a / B.3
//
// Each config below builds the CORRECTED tree with MAX_DEPTH loud (assert
// fires if exceeded). A failed assert here would itself be a finding.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

fn measure(name: &str, bets: Vec<BetSize>, raises: Vec<BetSize>) -> usize {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; 6],
        initial_contributions: vec![1, 2, 0, 0, 0, 0],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: bets,
            raise: raises,
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("builds");
    let total = tree.num_nodes();
    let (mut p, mut c, mut t) = (0, 0, 0);
    for n in &tree.nodes {
        if n.is_player() {
            p += 1;
        } else if n.is_chance() {
            c += 1;
        } else {
            t += 1;
        }
    }
    eprintln!(
        "  {:<40} {:>9} nodes  ({:>6} player, {:>4} chance, {:>6} terminal)",
        name, total, p, c, t
    );
    total
}

#[test]
fn six_max_abstraction_curve() {
    eprintln!("\n=== 6-max preflop tree size across action-abstraction curve ===\n");
    eprintln!("All on CORRECTED foundation (round-completion fix + MAX_DEPTH loud).\n");

    // 1-bet, 0-raise: the existing floor measurement
    let floor = measure(
        "simplest (1 bet, 0 raise)",
        vec![BetSize::PotRelative(1.0)],
        vec![],
    );

    // 1 bet, 1 raise: minimal raise machinery
    let one_one = measure(
        "1 bet, 1 raise (pot, pot)",
        vec![BetSize::PotRelative(1.0)],
        vec![BetSize::PotRelative(1.0)],
    );

    // 2 bet, 0 raise
    let two_zero = measure(
        "2 bet (0.5p, 1p), 0 raise",
        vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        vec![],
    );

    // 2 bet, 1 raise
    let two_one = measure(
        "2 bet (0.5p, 1p), 1 raise (pot)",
        vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        vec![BetSize::PotRelative(1.0)],
    );

    // 2 bet, 2 raise: Option-B-equivalent (the postflop abstraction verified
    // sufficient by max_na_option_b_completeness, applied to preflop)
    let opt_b = measure(
        "Option-B-equiv (2 bet 0.5p/1p, 2 raise 0.5p/1p)",
        vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
    );

    eprintln!("\n=== Comparison points ===");
    eprintln!("  HU Option-B postflop reference:           15,029 nodes");
    eprintln!("  6-max wrong-game simplest (pre-fix):      29,882 nodes");
    eprintln!("  6-max corrected simplest (floor):       {:>9} nodes", floor);
    eprintln!("  6-max corrected 1 bet + 1 raise:        {:>9} nodes", one_one);
    eprintln!("  6-max corrected 2 bet + 0 raise:        {:>9} nodes", two_zero);
    eprintln!("  6-max corrected 2 bet + 1 raise:        {:>9} nodes", two_one);
    eprintln!("  6-max corrected Option-B-equivalent:    {:>9} nodes", opt_b);
    eprintln!();
    eprintln!("Question for the architectural decision: where does the realistic");
    eprintln!("6-max corrected tree sit relative to HU Option-B's 15,029?");
}
