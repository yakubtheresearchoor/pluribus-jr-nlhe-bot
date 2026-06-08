// MAX_NA_POSTFLOP decision (#31), step 1: measure tree sizes for Option A (current
// MAX_NA_POSTFLOP=4) vs candidate Option B configurations. This is the precondition
// for the MAX_NA_POSTFLOP-vs-blueprint-EV measurement — if Option B's tree is
// computationally intractable, the EV gap may not be measurable, which
// itself informs the decision.
//
// The maintenance principle from the validation arc applies here: when
// MAX_NA_POSTFLOP changes, the standing showdown oracle's action-count combinations
// must extend to cover the new action sets. Tree sizing is the first step
// to scope that oracle extension.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn tree_stats(label: &str, cfg: &TreeConfig) -> Result<(usize, usize, usize, usize), String> {
    let tree = build_tree(cfg).map_err(|e| format!("[{}] build failed: {}", label, e))?;
    let nn = tree.num_nodes();
    let np = cfg.num_players as usize;
    let mut n_player = 0;
    let mut n_chance = 0;
    let mut n_terminal = 0;
    let mut max_children = 0;
    let mut by_nc: std::collections::BTreeMap<u16, usize> = std::collections::BTreeMap::new();
    for n in &tree.nodes {
        if n.is_player() {
            n_player += 1;
            *by_nc.entry(n.num_children).or_insert(0) += 1;
            if n.num_children as usize > max_children { max_children = n.num_children as usize; }
        } else if n.is_chance() {
            n_chance += 1;
        } else {
            n_terminal += 1;
        }
    }
    eprintln!(
        "[{}] {} nodes total (player={}, chance={}, terminal={}), max_children={}",
        label, nn, n_player, n_chance, n_terminal, max_children
    );
    eprintln!("  player nodes by num_children: {:?}", by_nc);
    Ok((nn, n_player, n_chance, n_terminal))
}

#[test]
fn max_na_tree_sizing_hu_symmetric_options() {
    eprintln!("\n=== MAX_NA_POSTFLOP tree sizing: HU symmetric [5,5] ===\n");

    // CURRENT (existing infrastructure): bet=[PotRel(1.0)], raise=[]
    //   facing-bet action set: fold, call → 2 actions (sometimes +allin = 3)
    //   not-facing-bet: check, bet → 2 actions (sometimes +allin = 3)
    //   max per node: ≤3. Fits MAX_NA_POSTFLOP=4 with one slot to spare.
    let cur = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,

    };
    let _ = tree_stats("Current: 1 bet PotRel(1.0), 0 raise", &cur).unwrap();
    eprintln!();

    // OPTION A (proposed cap at MAX_NA_POSTFLOP=4): bet=[PotRel(1.0)], raise=[PotRel(1.0)]
    //   facing-bet: fold, call, raise → 3 (+allin = 4 max). Fits MAX_NA_POSTFLOP=4.
    //   not-facing-bet: check, bet → 2 (+allin = 3). Fits.
    //   This is the richest abstraction that fits MAX_NA_POSTFLOP=4 with raises.
    let opt_a = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,

    };
    let _ = tree_stats("Option A: 1 bet + 1 raise PotRel(1.0)", &opt_a).unwrap();
    eprintln!();

    // OPTION B candidate (needs MAX_NA_POSTFLOP ≥ 5 to even build):
    //   bet=[PotRel(0.5), PotRel(1.0)], raise=[PotRel(0.5), PotRel(1.0)]
    //   facing-bet: fold, call, raise_0.5, raise_1.0 → 4 (+allin = 5 max).
    //   Exceeds MAX_NA_POSTFLOP=4 → build-time assert fires (#37 Phase 5 hardening).
    // We attempt to build it; expect the assert.
    let opt_b = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,

    };
    let opt_b_result = std::panic::catch_unwind(|| {
        build_tree(&opt_b)
    });
    match opt_b_result {
        Ok(Ok(_)) => {
            // Re-run as proper tree_stats now that we know it builds.
            let _ = tree_stats("Option B: 2 bet + 2 raise PotRel(0.5/1.0)", &opt_b).unwrap();
            eprintln!();
            eprintln!("  IMPORTANT: Option B BUILDS at MAX_NA_POSTFLOP=4 — action sets collapse to ≤4 via");
            eprintln!("  some combination of clamp_and_force_allin + sort + dedup. This means the");
            eprintln!("  Option A vs B framing was based on a wrong assumption about which configs");
            eprintln!("  fit MAX_NA_POSTFLOP=4. The richer Pluribus-style abstraction may already be");
            eprintln!("  expressible within the current stride budget. This changes the MAX_NA_POSTFLOP");
            eprintln!("  decision: if richer abstractions fit, there's no kernel re-touch needed.");
        }
        Ok(Err(e)) => {
            eprintln!("[Option B] tree build error (not assert): {}", e);
        }
        Err(_) => {
            eprintln!("[Option B (2 bet + 2 raise)] build-time MAX_NA_POSTFLOP assert FIRED (expected).");
            eprintln!("  This confirms Option B requires MAX_NA_POSTFLOP ≥ 5 to even build the tree.");
        }
    }

    // Even richer: 3 bet sizes + 3 raise sizes (Pluribus first-raise abstraction).
    eprintln!();
    let opt_c = TreeConfig {
        num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
        starting_stacks: vec![100, 100], initial_contributions: vec![5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0), BetSize::PotRelative(2.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0), BetSize::PotRelative(2.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,

    };
    let opt_c_result = std::panic::catch_unwind(|| { build_tree(&opt_c) });
    match opt_c_result {
        Ok(Ok(_)) => {
            let _ = tree_stats("Option C: 3 bet + 3 raise PotRel(0.5/1.0/2.0)", &opt_c).unwrap();
        }
        Ok(Err(e)) => eprintln!("[Option C] tree build error: {}", e),
        Err(_) => eprintln!("[Option C] MAX_NA_POSTFLOP assert FIRED — exceeds MAX_NA_POSTFLOP=4."),
    }
}
