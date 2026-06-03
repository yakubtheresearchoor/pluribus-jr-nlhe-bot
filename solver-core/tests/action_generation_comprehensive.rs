// Comprehensive validation of compute_actions against poker rules.
// After two distinct action-generation bugs (bet-collapse + force-fold), the
// function's output gets exhaustive validation across all node-type
// situations rather than case-by-case patches.
//
// For each player node in the tree, classify the situation from game state
// and compare the generated action set to what poker rules require:
//
//   Situation 1: Player is all-in (remaining = 0)
//     Legal: {CHECK only} (forced pass)
//
//   Situation 2: Player has chips, facing a bet (committed < max_other_committed)
//     Legal: {FOLD, CALL} at minimum (raises optional based on config)
//
//   Situation 3: Player has chips, NOT facing a bet (committed >= max_other_committed)
//     Legal: {CHECK, BET sizes} at minimum (allin only if threshold triggered)
//
// Report every situation × violation type combination.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{FlatTree,
    ACTION_LABEL_CHECK, ACTION_LABEL_CALL, ACTION_LABEL_FOLD,
    ACTION_LABEL_BET, ACTION_LABEL_RAISE, ACTION_LABEL_ALLIN};

fn label_name(l: u8) -> &'static str {
    match l {
        ACTION_LABEL_FOLD => "FOLD",
        ACTION_LABEL_CHECK => "CHECK",
        ACTION_LABEL_CALL => "CALL",
        ACTION_LABEL_BET => "BET",
        ACTION_LABEL_RAISE => "RAISE",
        ACTION_LABEL_ALLIN => "ALLIN",
        _ => "OTHER",
    }
}

#[derive(Default, Clone, Debug)]
struct Stats {
    count: usize,
    has_check: usize,
    has_call: usize,
    has_fold: usize,
    has_bet: usize,
    has_raise: usize,
    has_allin: usize,
    only_fold: usize,
    only_check: usize,
    only_call: usize,
}

fn classify_actions(tree: &FlatTree, node_idx: usize, stats: &mut Stats) {
    let n = &tree.nodes[node_idx];
    stats.count += 1;
    let nc = n.num_children as usize;
    let mut has = [false; 6];  // [FOLD, CHECK, CALL, BET, RAISE, ALLIN]
    for i in 0..nc {
        let child_idx = tree.children[n.children_start as usize + i] as usize;
        let lbl = tree.nodes[child_idx].action_label;
        if lbl < 6 { has[lbl as usize] = true; }
    }
    if has[ACTION_LABEL_FOLD as usize] { stats.has_fold += 1; }
    if has[ACTION_LABEL_CHECK as usize] { stats.has_check += 1; }
    if has[ACTION_LABEL_CALL as usize] { stats.has_call += 1; }
    if has[ACTION_LABEL_BET as usize] { stats.has_bet += 1; }
    if has[ACTION_LABEL_RAISE as usize] { stats.has_raise += 1; }
    if has[ACTION_LABEL_ALLIN as usize] { stats.has_allin += 1; }
    if nc == 1 {
        let lbl = tree.nodes[tree.children[n.children_start as usize] as usize].action_label;
        match lbl {
            ACTION_LABEL_FOLD => stats.only_fold += 1,
            ACTION_LABEL_CHECK => stats.only_check += 1,
            ACTION_LABEL_CALL => stats.only_call += 1,
            _ => {}
        }
    }
}

fn print_stats(name: &str, s: &Stats) {
    if s.count == 0 {
        eprintln!("  {}: 0 nodes", name);
        return;
    }
    eprintln!("  {} ({} nodes)", name, s.count);
    eprintln!("    has CHECK: {:>10} ({:.1}%)", s.has_check, s.has_check as f64 / s.count as f64 * 100.0);
    eprintln!("    has CALL : {:>10} ({:.1}%)", s.has_call, s.has_call as f64 / s.count as f64 * 100.0);
    eprintln!("    has FOLD : {:>10} ({:.1}%)", s.has_fold, s.has_fold as f64 / s.count as f64 * 100.0);
    eprintln!("    has BET  : {:>10} ({:.1}%)", s.has_bet, s.has_bet as f64 / s.count as f64 * 100.0);
    eprintln!("    has RAISE: {:>10} ({:.1}%)", s.has_raise, s.has_raise as f64 / s.count as f64 * 100.0);
    eprintln!("    has ALLIN: {:>10} ({:.1}%)", s.has_allin, s.has_allin as f64 / s.count as f64 * 100.0);
    eprintln!("    1-child only FOLD : {:>10}", s.only_fold);
    eprintln!("    1-child only CHECK: {:>10}", s.only_check);
    eprintln!("    1-child only CALL : {:>10}", s.only_call);
}

#[test]
fn comprehensive_action_generation_audit() {
    let cfg = TreeConfig {
        num_players: 6, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    };
    let tree = build_tree(&cfg).unwrap();
    let np = 6usize;
    let starting_stacks = vec![200i32; 6];
    let initial_contribs = vec![10i32, 5, 5, 5, 5, 5];
    let max_committable: Vec<i32> = starting_stacks.iter().zip(&initial_contribs)
        .map(|(s, c)| s + c).collect();

    eprintln!("\n=== Comprehensive compute_actions audit against poker rules ===\n");

    let mut s_allin = Stats::default();
    let mut s_facing_bet_has_chips = Stats::default();
    let mut s_not_facing_bet_has_chips = Stats::default();
    let mut s_inactive_or_other = Stats::default();

    // Detailed violation lists
    let mut violations_facing_bet_missing_call: Vec<usize> = Vec::new();
    let mut violations_facing_bet_missing_fold: Vec<usize> = Vec::new();
    let mut violations_not_facing_bet_missing_check: Vec<usize> = Vec::new();
    let mut violations_not_facing_bet_missing_bet_options: Vec<usize> = Vec::new();
    let mut violations_not_facing_bet_has_fold: Vec<usize> = Vec::new();
    let mut violations_all_in_not_check: Vec<usize> = Vec::new();

    for (idx, n) in tree.nodes.iter().enumerate() {
        if !n.is_player() { continue; }
        let player = n.player_id as usize;

        // Compute game state at this node
        let player_committed = tree.get_contribution(idx, player as u8);
        let player_remaining = max_committable[player] - player_committed;

        // Find max_other_committed over OTHER players (any player except this one;
        // can't directly tell which are folded from tree, but contribs of folded
        // players are frozen at their last commit so they still represent the
        // "highest competing commit" that this player needs to match).
        let mut max_other = 0i32;
        for p in 0..np {
            if p == player { continue; }
            let c = tree.get_contribution(idx, p as u8);
            if c > max_other { max_other = c; }
        }

        // Classify situation
        let actions: Vec<u8> = (0..n.num_children as usize)
            .map(|i| tree.nodes[tree.children[n.children_start as usize + i] as usize].action_label)
            .collect();
        let has = |l: u8| actions.contains(&l);

        if player_remaining == 0 {
            // Situation 1: all-in player, expected {CHECK only}
            classify_actions(&tree, idx, &mut s_allin);
            if !has(ACTION_LABEL_CHECK) || actions.len() != 1 ||
               actions[0] != ACTION_LABEL_CHECK {
                if violations_all_in_not_check.len() < 10 {
                    violations_all_in_not_check.push(idx);
                }
            }
        } else if player_committed < max_other {
            // Situation 2: has chips, facing a bet, expected {FOLD, CALL} minimum
            classify_actions(&tree, idx, &mut s_facing_bet_has_chips);
            if !has(ACTION_LABEL_CALL) && !has(ACTION_LABEL_ALLIN) {
                if violations_facing_bet_missing_call.len() < 10 {
                    violations_facing_bet_missing_call.push(idx);
                }
            }
            if !has(ACTION_LABEL_FOLD) {
                if violations_facing_bet_missing_fold.len() < 10 {
                    violations_facing_bet_missing_fold.push(idx);
                }
            }
        } else {
            // Situation 3: has chips, NOT facing a bet, expected {CHECK, BET sizes} minimum
            classify_actions(&tree, idx, &mut s_not_facing_bet_has_chips);
            if !has(ACTION_LABEL_CHECK) {
                if violations_not_facing_bet_missing_check.len() < 10 {
                    violations_not_facing_bet_missing_check.push(idx);
                }
            }
            if !has(ACTION_LABEL_BET) && !has(ACTION_LABEL_ALLIN) {
                if violations_not_facing_bet_missing_bet_options.len() < 10 {
                    violations_not_facing_bet_missing_bet_options.push(idx);
                }
            }
            if has(ACTION_LABEL_FOLD) {
                if violations_not_facing_bet_has_fold.len() < 10 {
                    violations_not_facing_bet_has_fold.push(idx);
                }
            }
        }
    }

    eprintln!("--- Situation 1: ALL-IN player (remaining=0), expected {{CHECK only}} ---");
    print_stats("ALL-IN", &s_allin);

    eprintln!();
    eprintln!("--- Situation 2: FACING-BET (committed < max_other), expected {{FOLD, CALL}} min ---");
    print_stats("FACING-BET", &s_facing_bet_has_chips);

    eprintln!();
    eprintln!("--- Situation 3: NOT-FACING-BET (committed >= max_other), expected {{CHECK, BET}} min ---");
    print_stats("NOT-FACING-BET", &s_not_facing_bet_has_chips);

    eprintln!();
    eprintln!("=== VIOLATIONS ===");

    let print_violations = |label: &str, indices: &[usize]| {
        if indices.is_empty() {
            eprintln!("  ✓ {}: 0 violations", label);
        } else {
            eprintln!("  ✗ {}: ≥{} violations (showing up to 10)", label, indices.len());
            for &idx in indices.iter().take(10) {
                let n = &tree.nodes[idx];
                let player = n.player_id;
                let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();
                let actions: Vec<String> = (0..n.num_children as usize)
                    .map(|i| {
                        let c = &tree.nodes[tree.children[n.children_start as usize + i] as usize];
                        format!("{}({})", label_name(c.action_label), c.amount)
                    })
                    .collect();
                eprintln!("    node[{}] p{} contribs={:?} → actions={:?}",
                    idx, player, contribs, actions);
            }
        }
    };

    print_violations("Facing-bet: missing CALL", &violations_facing_bet_missing_call);
    print_violations("Facing-bet: missing FOLD", &violations_facing_bet_missing_fold);
    print_violations("Not-facing-bet: missing CHECK", &violations_not_facing_bet_missing_check);
    print_violations("Not-facing-bet: missing BET options", &violations_not_facing_bet_missing_bet_options);
    print_violations("Not-facing-bet: has FOLD (FOLD illegal when no bet to call)",
        &violations_not_facing_bet_has_fold);
    print_violations("All-in: action set is not exactly {CHECK}", &violations_all_in_not_check);

    eprintln!();
    eprintln!("=== ROLLUP ===");
    let total_violations = violations_facing_bet_missing_call.len()
        + violations_facing_bet_missing_fold.len()
        + violations_not_facing_bet_missing_check.len()
        + violations_not_facing_bet_missing_bet_options.len()
        + violations_not_facing_bet_has_fold.len()
        + violations_all_in_not_check.len();
    if total_violations == 0 {
        eprintln!("  ✓ All player nodes generate the rule-required minimum action set.");
        eprintln!("    compute_actions is fully poker-rules-compliant. Safe to optimize.");
    } else {
        eprintln!("  ✗ compute_actions has at least {} violation classes producing non-rule-compliant action sets.",
            total_violations);
        eprintln!("    DO NOT optimize (elision / right-sizing) until compute_actions is fixed.");
        eprintln!("    The current solver is solving a game whose action set is wrong; convergence");
        eprintln!("    on this game is convergence to a non-poker equilibrium (too-tight / too-foldy).");
    }
}
