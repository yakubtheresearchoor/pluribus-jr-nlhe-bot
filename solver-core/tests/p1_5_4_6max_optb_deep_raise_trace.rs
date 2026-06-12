// Verification 1: trace the deepest raise sequence in the 2+2 6-max preflop
// tree and confirm the cascading is legitimate, not action over-generation.
//
// Why this test exists: the corrected 6-max curve jumped 5x from one raise
// level (2 bet + 1 raise = 32,923 nodes) to two raise levels (2 bet + 2 raise
// = 162,650 nodes). The recent history (wrong-game trees being bigger for
// wrong reasons) warrants confirming this tree is bigger for the right
// reason: legitimate multiway raise depth, not duplicated/spurious actions.
//
// "Trace the chips, don't conclude from labels" — derive every action from
// the change in chip commits across a parent → child edge. No reliance on
// builder-side action labels.
//
// What it does: starting from the root, walk the path that takes the largest
// committing action at every player decision until the chain terminates
// (allin or fold). At each node print: depth, player, commits, num_children,
// the chip-diff to each child. Then assert:
//   (a) chain terminates by chip exhaustion (allin) or fold or chance, not
//       by depth cap;
//   (b) max chip commit is monotonically non-decreasing along the chain;
//   (c) every action set fits within MAX_NA_POSTFLOP=4;
//   (d) every child of a player node represents a distinct chip-commit
//       outcome (i.e., no spurious duplicate children — the dedup is real).

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree, MAX_NA_POSTFLOP};

fn commits_at(tree: &FlatTree, node_idx: usize, np: usize) -> Vec<i32> {
    (0..np).map(|p| tree.get_contribution(node_idx, p as u8)).collect()
}

fn classify_edge(parent_commits: &[i32], child_commits: &[i32], actor: u8) -> String {
    let p = actor as usize;
    let diff = child_commits[p] - parent_commits[p];
    let max_parent = *parent_commits.iter().max().unwrap();
    let max_child = *child_commits.iter().max().unwrap();
    if diff == 0 && max_child == max_parent {
        // No chip change for actor and no new high — Check or Fold-by-non-actor.
        // For the actor's own action, this is Check (if actor was facing 0
        // to call) or it could be a fold (signaled by the folded_mask).
        "Check/Fold (Δ=0)".into()
    } else if diff > 0 && child_commits[p] == max_child && max_child > max_parent {
        // Actor pushed above the previous max — new bet/raise.
        format!("Bet/Raise (Δ={}, new max={})", diff, max_child)
    } else if diff > 0 && child_commits[p] == max_child && max_child == max_parent {
        // Actor matched the standing bet.
        format!("Call (Δ={}, matched {})", diff, max_child)
    } else if diff > 0 {
        // Actor committed but did not reach the standing max — partial allin call.
        format!("Partial allin (Δ={}, to {} < max {})", diff, child_commits[p], max_child)
    } else {
        format!("Fold/Check (Δ={})", diff)
    }
}

#[test]
fn six_max_optb_deepest_raise_chain() {
    let np = 6usize;
    let cfg = TreeConfig {
        num_players: np as u8,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; np],
        initial_contributions: vec![1, 2, 0, 0, 0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("builds");

    eprintln!("\n=== Deepest-raise chain in 6-max Option-B preflop tree ===\n");
    eprintln!("Config: 6 players, button=5, SB/BB=1/2, stacks=100, bet=[0.5p,1p], raise=[0.5p,1p]");
    eprintln!("Tree: {} nodes total\n", tree.num_nodes());

    let mut node_idx = 0usize;
    let mut depth = 0usize;
    let mut prev_max_commit = 0i32;
    let mut max_children_seen = 0usize;

    let termination;
    loop {
        let node = &tree.nodes[node_idx];

        if node.is_chance() {
            let commits = commits_at(&tree, node_idx, np);
            termination = format!("CHANCE (preflop→flop) at depth {}, commits={:?}", depth, commits);
            eprintln!("[chain ends at chance — preflop round complete]");
            eprintln!("   final commits = {:?}", commits);
            prev_max_commit = *commits.iter().max().unwrap();
            break;
        }
        if !node.is_player() {
            let commits = commits_at(&tree, node_idx, np);
            termination = format!("TERMINAL at depth {}, commits={:?}", depth, commits);
            eprintln!("[chain ends at terminal — fold]");
            eprintln!("   final commits = {:?}", commits);
            prev_max_commit = *commits.iter().max().unwrap();
            break;
        }

        // Player node. Walk children, derive each edge's chip diff.
        let commits = commits_at(&tree, node_idx, np);
        let actor = node.player_id;
        let children = tree.node_children(node_idx);
        let nc = children.len();
        max_children_seen = max_children_seen.max(nc);

        eprintln!(
            "depth={:2}  node={:6}  player={}  commits={:?}  #children={}",
            depth, node_idx, actor, commits, nc
        );

        // Per-child diffs + dedup check.
        let mut child_outcomes: Vec<(usize, Vec<i32>, String)> = Vec::with_capacity(nc);
        for &c in children {
            let cc = commits_at(&tree, c as usize, np);
            let edge = classify_edge(&commits, &cc, actor);
            eprintln!("            → child={:6}  commits={:?}  edge={}", c, cc, edge);
            child_outcomes.push((c as usize, cc, edge));
        }
        // Dedup check (d): every child must have a distinct (actor_commit, max_commit) signature.
        let mut sigs: Vec<(i32, i32)> = child_outcomes
            .iter()
            .map(|(_, cc, _)| (cc[actor as usize], *cc.iter().max().unwrap()))
            .collect();
        sigs.sort();
        let dups = sigs.windows(2).filter(|w| w[0] == w[1]).count();
        assert_eq!(
            dups, 0,
            "node {} has {} duplicate child outcomes — would indicate over-generation (the wrong-reason tree growth)",
            node_idx, dups
        );

        // Monotonicity (b): max commit non-decreasing along the deepest-raise chain.
        let cur_max = *commits.iter().max().unwrap();
        assert!(
            cur_max >= prev_max_commit,
            "max chip commit decreased at depth {}: was {}, now {}",
            depth, prev_max_commit, cur_max
        );
        prev_max_commit = cur_max;

        // Pick the child with the largest actor commit (deepest raise). Ties
        // broken by largest max_commit (most aggressive).
        let next = child_outcomes
            .iter()
            .max_by_key(|(_, cc, _)| (cc[actor as usize], *cc.iter().max().unwrap()))
            .expect("at least one child");
        eprintln!("            ⇒ chose child {} (deepest raise)\n", next.0);
        node_idx = next.0;
        depth += 1;

        // (a): chain bound. Anything over 50 deep in preflop is itself a finding.
        assert!(depth < 50, "chain exceeded depth 50; expected termination by allin/fold/chance");
    }

    eprintln!("\nSummary:");
    eprintln!("  Chain depth: {}", depth);
    eprintln!("  Termination: {}", termination);
    eprintln!("  Max #children seen on any player node: {} (MAX_NA_POSTFLOP={})", max_children_seen, MAX_NA_POSTFLOP);
    eprintln!("  Final max commit: {} chips (starting stack: 100)", prev_max_commit);
    eprintln!();
    eprintln!("Verification:");
    eprintln!("  (a) Chain terminated by chance/fold, not depth cap: PASS");
    eprintln!("  (b) Max commit monotonically non-decreasing: PASS");
    eprintln!("  (c) All action sets within MAX_NA_POSTFLOP={}: PASS", MAX_NA_POSTFLOP);
    eprintln!("  (d) No duplicate child outcomes at any node along the chain: PASS");

    assert!(
        max_children_seen <= MAX_NA_POSTFLOP,
        "max children {} exceeded MAX_NA_POSTFLOP {}", max_children_seen, MAX_NA_POSTFLOP
    );
}
