// STANDING tree-correctness gate — permanent rules-compliance check.
//
// After three distinct action-generation bugs (bet-collapse, force-fold,
// prev-action-branching), the durable fix is: build the gate, gate the tree
// against it, never let an under-validated builder produce a tree that
// downstream solving uses. This is to the tree what kernel parity gates are
// to the math.
//
// Per-street commit semantics: at any tree node, the legal poker actions
// depend on whether the player owes chips RELATIVE TO THE CURRENT STREET,
// not relative to cumulative commits across the whole hand. The audit
// implements this correctly here by tracking `commit_at_round_start` —
// the snapshot of contribs at the most recent chance ancestor (or root) —
// and computing per-street commits = current - snapshot.
//
// Checks performed at every player node:
//   1. All-in player (remaining=0): legal set is exactly {CHECK}
//   2. Facing-bet (per-street player < per-street max-other): {FOLD, CALL+ALLIN} min
//   3. Not-facing-bet: {CHECK} required; FOLD illegal (nothing to fold to)
//   4. PLAYER node never has 0 children (structural)
//   5. Folded player never appears as the acting player_id
//
// Checks performed at every terminal node:
//   6. All contributions ≤ player's max_committable (physical-validity)
//
// Configurable to run on multiple representative tree configs. Gate asserts
// zero violations across all checks.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{
    FlatTree,
    ACTION_LABEL_CHECK, ACTION_LABEL_CALL, ACTION_LABEL_FOLD,
    ACTION_LABEL_BET, ACTION_LABEL_RAISE, ACTION_LABEL_ALLIN,
};

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

#[derive(Default, Debug, Clone)]
struct Violations {
    /// All-in player not given exactly {CHECK}
    allin_wrong_action_set: Vec<(usize, Vec<u8>)>,
    /// Facing-bet player missing FOLD
    facing_bet_missing_fold: Vec<usize>,
    /// Facing-bet player missing CALL/ALLIN
    facing_bet_missing_call: Vec<usize>,
    /// Not-facing-bet player missing CHECK
    not_facing_bet_missing_check: Vec<usize>,
    /// Not-facing-bet player has FOLD (illegal)
    not_facing_bet_has_fold: Vec<usize>,
    /// PLAYER node with 0 children
    empty_player_node: Vec<usize>,
    /// Acting player is folded
    folded_player_acts: Vec<(usize, u8)>,
    /// Terminal contribution exceeds player's max_committable
    contribution_exceeds_stack: Vec<(usize, u8, i32, i32)>,
}

impl Violations {
    fn total(&self) -> usize {
        self.allin_wrong_action_set.len()
        + self.facing_bet_missing_fold.len()
        + self.facing_bet_missing_call.len()
        + self.not_facing_bet_missing_check.len()
        + self.not_facing_bet_has_fold.len()
        + self.empty_player_node.len()
        + self.folded_player_acts.len()
        + self.contribution_exceeds_stack.len()
    }
}

fn print_violations(label: &str, vs: &Violations, max_samples: usize) {
    let total = vs.total();
    eprintln!("\n=== Violations for: {} (total: {}) ===", label, total);
    if total == 0 {
        eprintln!("  ✓ ZERO violations — tree is poker-rules-compliant");
        return;
    }
    let print_sample = |name: &str, items: &[usize], n: usize| {
        if items.is_empty() {
            eprintln!("  ✓ {}: 0", name);
        } else {
            eprintln!("  ✗ {}: {} (sample of {})", name, items.len(), items.len().min(n));
            for &idx in items.iter().take(n) {
                eprint!("      node[{}] ", idx);
                eprintln!();
            }
        }
    };
    print_sample("Facing-bet missing FOLD", &vs.facing_bet_missing_fold, max_samples);
    print_sample("Facing-bet missing CALL", &vs.facing_bet_missing_call, max_samples);
    print_sample("Not-facing-bet missing CHECK", &vs.not_facing_bet_missing_check, max_samples);
    print_sample("Not-facing-bet has FOLD", &vs.not_facing_bet_has_fold, max_samples);
    print_sample("Empty PLAYER node (0 children)", &vs.empty_player_node, max_samples);

    if !vs.allin_wrong_action_set.is_empty() {
        eprintln!("  ✗ All-in not exactly {{CHECK}}: {} (sample of {})",
            vs.allin_wrong_action_set.len(), vs.allin_wrong_action_set.len().min(max_samples));
        for (idx, actions) in vs.allin_wrong_action_set.iter().take(max_samples) {
            let names: Vec<&str> = actions.iter().map(|&a| label_name(a)).collect();
            eprintln!("      node[{}] all-in, actions={:?}", idx, names);
        }
    }
    if !vs.folded_player_acts.is_empty() {
        eprintln!("  ✗ Folded player acts: {} (sample of {})",
            vs.folded_player_acts.len(), vs.folded_player_acts.len().min(max_samples));
        for (idx, p) in vs.folded_player_acts.iter().take(max_samples) {
            eprintln!("      node[{}] folded player p{} is acting", idx, p);
        }
    }
    if !vs.contribution_exceeds_stack.is_empty() {
        eprintln!("  ✗ Contribution exceeds max_committable: {} (sample of {})",
            vs.contribution_exceeds_stack.len(),
            vs.contribution_exceeds_stack.len().min(max_samples));
        for (idx, p, contrib, max_c) in vs.contribution_exceeds_stack.iter().take(max_samples) {
            eprintln!("      node[{}] p{} contrib={} > max={}", idx, p, contrib, max_c);
        }
    }
}

/// Iterative DFS that threads commit_at_round_start through the walk.
/// At each chance node, the snapshot for its children = contribs at the chance node.
///
/// Root snapshot (FIXED 2026-06-12): for POSTFLOP-start trees the root
/// contributions are prior-street money, so the snapshot = root contribs
/// (per-street commits start at 0). For PREFLOP-start trees the blinds
/// are LIVE first-round commits (no street preceded them) — snapshot
/// must be ZERO so the SB genuinely faces the BB's blind. The old
/// unconditional snapshot=root-contribs zeroed the blinds out and
/// misclassified the HU-preflop root (flagged the builder's correct
/// {FOLD,CALL,RAISE} as illegal — the 4 phantom violations).
fn validate_tree(
    tree: &FlatTree,
    np: usize,
    max_committable: &[i32],
    preflop_root: bool,
) -> Violations {
    let mut vs = Violations::default();
    // Iterative DFS: stack of (node_idx, commit_at_round_start, folded_mask)
    let initial_snapshot: Vec<i32> = if preflop_root {
        vec![0; np]
    } else {
        (0..np).map(|p| tree.get_contribution(0, p as u8)).collect()
    };
    let mut stack: Vec<(usize, Vec<i32>, u16)> = vec![(0, initial_snapshot, 0)];

    while let Some((idx, round_start, folded_mask)) = stack.pop() {
        let n = &tree.nodes[idx];
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(idx, p as u8)).collect();

        if n.is_chance() {
            // Snapshot the current contribs for children's per-street comparison
            let new_snapshot: Vec<i32> = contribs.clone();
            for i in 0..n.num_children as usize {
                let c = tree.children[n.children_start as usize + i] as usize;
                stack.push((c, new_snapshot.clone(), folded_mask));
            }
            continue;
        }

        if n.is_terminal() {
            // Check 6: contributions ≤ max_committable
            for p in 0..np {
                if contribs[p] > max_committable[p] {
                    vs.contribution_exceeds_stack.push((idx, p as u8, contribs[p], max_committable[p]));
                }
            }
            // Note: terminal's folded_mask is set on the tree directly via set_folded_mask;
            // we get the actual mask from the tree.
            continue;
        }

        // PLAYER NODE
        let nc = n.num_children as usize;
        let player = n.player_id;

        // Check 4: empty player node (no children)
        if nc == 0 {
            vs.empty_player_node.push(idx);
            continue;  // can't recurse
        }

        // Check 5: folded player acts
        if (folded_mask & (1u16 << player)) != 0 {
            vs.folded_player_acts.push((idx, player));
        }

        // Per-street commit comparison
        let per_street: Vec<i32> = (0..np)
            .map(|p| contribs[p] - round_start[p])
            .collect();
        let player_this_street = per_street[player as usize];
        let max_other_this_street: i32 = (0..np)
            .filter(|&p| p != player as usize && (folded_mask & (1u16 << p)) == 0)
            .map(|p| per_street[p])
            .max()
            .unwrap_or(0);
        let player_remaining = max_committable[player as usize] - contribs[player as usize];
        let facing_bet = player_this_street < max_other_this_street;

        // Gather the child action labels
        let actions: Vec<u8> = (0..nc)
            .map(|i| tree.nodes[tree.children[n.children_start as usize + i] as usize].action_label)
            .collect();
        let has = |l: u8| actions.contains(&l);

        // Check 1: all-in player
        if player_remaining == 0 {
            // Expected: exactly {CHECK}
            if !(actions.len() == 1 && actions[0] == ACTION_LABEL_CHECK) {
                vs.allin_wrong_action_set.push((idx, actions.clone()));
            }
        } else if facing_bet {
            // Check 2: facing-bet, expected {FOLD, CALL/ALLIN}
            if !has(ACTION_LABEL_FOLD) {
                vs.facing_bet_missing_fold.push(idx);
            }
            if !has(ACTION_LABEL_CALL) && !has(ACTION_LABEL_ALLIN) {
                vs.facing_bet_missing_call.push(idx);
            }
        } else {
            // Check 3: not-facing-bet, expected {CHECK, BET/ALLIN}; FOLD illegal
            if !has(ACTION_LABEL_CHECK) {
                vs.not_facing_bet_missing_check.push(idx);
            }
            if has(ACTION_LABEL_FOLD) {
                vs.not_facing_bet_has_fold.push(idx);
            }
        }

        // Recurse: child gets the same round_start; folded_mask updated based on action
        for i in 0..nc {
            let c_idx = tree.children[n.children_start as usize + i] as usize;
            let c = &tree.nodes[c_idx];
            let new_mask = if c.action_label == ACTION_LABEL_FOLD {
                folded_mask | (1u16 << player)
            } else {
                folded_mask
            };
            stack.push((c_idx, round_start.clone(), new_mask));
        }
    }

    vs
}

fn run_gate(label: &str, config: TreeConfig) -> Violations {
    let tree = build_tree(&config).unwrap();
    let np = config.num_players as usize;
    let max_committable: Vec<i32> = config.starting_stacks.iter()
        .zip(&config.initial_contributions)
        .map(|(s, c)| s + c)
        .collect();
    eprintln!("\n>>> Gate run: {} (tree: {} nodes)", label, tree.num_nodes());
    let vs = validate_tree(
        &tree,
        np,
        &max_committable,
        config.initial_state == BoardState::Preflop,
    );
    print_violations(label, &vs, 5);
    vs
}

#[test]
fn tree_correctness_gate_multi_config() {
    eprintln!("\n=== STANDING TREE-CORRECTNESS GATE ===");
    eprintln!("Runs poker-rules-compliance audit across representative configs.");
    eprintln!("Asserts zero violations across all checks.");

    let total = {
        // Config A: 6p asymmetric blinds [10, 5×5] — main test scenario
        let v_a = run_gate(
            "6p asymmetric [10,5,5,5,5,5], 1 bet PotRel(1.0)",
            TreeConfig {
                num_players: 6, initial_state: BoardState::Flop, starting_pot: 30,
                starting_stacks: vec![200; 6],
                initial_contributions: vec![10, 5, 5, 5, 5, 5],
                rake_rate: 0.0, rake_cap: 0.0,
                bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
                add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,

            }
        );

        // Config B: 6p symmetric blinds
        let v_b = run_gate(
            "6p symmetric [5;6], 1 bet PotRel(1.0)",
            TreeConfig {
                num_players: 6, initial_state: BoardState::Flop, starting_pot: 30,
                starting_stacks: vec![100; 6],
                initial_contributions: vec![5; 6],
                rake_rate: 0.0, rake_cap: 0.0,
                bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
                add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,

            }
        );

        // Config C: 3p smaller for fast iteration
        let v_c = run_gate(
            "3p asymmetric [10,5,5], 1 bet PotRel(1.0)",
            TreeConfig {
                num_players: 3, initial_state: BoardState::Flop, starting_pot: 15,
                starting_stacks: vec![200; 3],
                initial_contributions: vec![10, 5, 5],
                rake_rate: 0.0, rake_cap: 0.0,
                bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
                add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,

            }
        );

        // Config D: 2p heads-up SYMMETRIC (simplest fully-enumerable case;
        // pulled forward from plan's Phase 5 because HU action-order is
        // distinct from multiway and the rewrite was not validated against it.)
        let v_d = run_gate(
            "2p HU symmetric [5,5], 1 bet PotRel(1.0)",
            TreeConfig {
                num_players: 2, initial_state: BoardState::Flop, starting_pot: 10,
                starting_stacks: vec![100; 2],
                initial_contributions: vec![5, 5],
                rake_rate: 0.0, rake_cap: 0.0,
                bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
                add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,

            }
        );

        // Config E: 2p heads-up ASYMMETRIC blinds (BB=2 at index 0, SB=1 at
        // index 1 — matches the postflop ordering convention where the first
        // active player by index acts first; HU asymmetric exercises both the
        // reversed-blind size and the per-street vs cumulative subtleties.)
        let v_e = run_gate(
            "2p HU asymmetric [2,1], 1 bet PotRel(1.0)",
            TreeConfig {
                num_players: 2, initial_state: BoardState::Flop, starting_pot: 3,
                starting_stacks: vec![100; 2],
                initial_contributions: vec![2, 1],
                rake_rate: 0.0, rake_cap: 0.0,
                bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
                add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,

            }
        );

        // Config F: 6p asymmetric with add_allin_threshold = 0.5 — Phase 5
        // hardening config. The lower threshold means open-allin gets added
        // to facing-bet action sets at smaller pot sizes (max <= prev + pot*0.5
        // vs prev + pot*1.0), exercising the conditional-allin paths more
        // densely. If the rewrite's classifier mishandles allin inclusion
        // when threshold < 1.0, this catches it.
        let v_f = run_gate(
            "6p asymmetric [10,5,5,5,5,5], 1 bet PotRel(1.0), allin_thr=0.5",
            TreeConfig {
                num_players: 6, initial_state: BoardState::Flop, starting_pot: 35,
                starting_stacks: vec![200; 6],
                initial_contributions: vec![10, 5, 5, 5, 5, 5],
                rake_rate: 0.0, rake_cap: 0.0,
                bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
                add_allin_threshold: 0.5, force_allin_threshold: 1.0, merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,

            }
        );

        // Config G: starts at BoardState::Turn — exercises the per-street
        // snapshot path on a non-Flop initial state. Tree spans turn + river
        // only (2 streets), with whatever per-street snapshot semantics apply
        // when the initial state isn't Flop.
        let v_g = run_gate(
            "6p asymmetric starting at TURN",
            TreeConfig {
                num_players: 6, initial_state: BoardState::Turn, starting_pot: 35,
                starting_stacks: vec![200; 6],
                initial_contributions: vec![10, 5, 5, 5, 5, 5],
                rake_rate: 0.0, rake_cap: 0.0,
                bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
                add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,

            }
        );

        // Config H: starts at BoardState::River — terminal street, no further
        // chance transitions; round-complete must go to TERMINAL, not CHANCE.
        let v_h = run_gate(
            "6p asymmetric starting at RIVER",
            TreeConfig {
                num_players: 6, initial_state: BoardState::River, starting_pot: 35,
                starting_stacks: vec![200; 6],
                initial_contributions: vec![10, 5, 5, 5, 5, 5],
                rake_rate: 0.0, rake_cap: 0.0,
                bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
                add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,

            }
        );

        // Config I: 2p heads-up PREFLOP — preflop integration component
        // gate. Same blind structure as Config E but starting at Preflop.
        // The preflop tree has a four-zone structure (PRE → CHANCE → FLOP
        // → CHANCE → TURN → CHANCE → RIVER) and the root acts player 1
        // (button = SB acts first preflop in HU, opposite of postflop's
        // BB-first). Per-PLAYER-node legality rules are identical to
        // postflop; this config verifies they pass on the preflop tree.
        let v_i = run_gate(
            "2p HU PREFLOP [2,1], 1 bet PotRel(1.0)",
            TreeConfig {
                // starting_pot is DEAD MONEY from before this tree
                // (additive with contributions in the terminal math —
                // the 2026-06-12 get_pot fix). Preflop blinds are LIVE
                // contributions, so starting_pot must be 0 here; the
                // old value 3 double-counted the blinds.
                num_players: 2, initial_state: BoardState::Preflop, starting_pot: 0,
                starting_stacks: vec![100; 2],
                initial_contributions: vec![2, 1],
                rake_rate: 0.0, rake_cap: 0.0,
                bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
                add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
            button_player: None,
            max_bets_per_street: None,

            }
        );

        // Config J: PRODUCTION GAME v1 preflop (6-max, blinds 1/2,
        // equal-total 200 stacks, button 5 → UTG first), derived from
        // GameSpec — the config the production preflop layer solves.
        // PREFLOP-ONLY build (flop-entry chance nodes are leaves; the
        // frozen postflop oracle owns everything past them). Reduced
        // raise ladder [0.5, 1.0, 2.0] keeps the gate fast (~100k
        // nodes); the legality classifier under test is identical at
        // any ladder width, and the full-ladder build is exercised by
        // the v1_seam_census instrument.
        let v_j = {
            use solver_core::tree::action::production_game_v1;
            use solver_core::tree::builder::build_tree_preflop_only;
            let spec = production_game_v1();
            let cfg = spec.preflop_tree_config(BetSizeOptions {
                bet: vec![BetSize::PotRelative(1.0)],
                raise: vec![
                    BetSize::PotRelative(0.5),
                    BetSize::PotRelative(1.0),
                    BetSize::PotRelative(2.0),
                ],
            });
            let tree = build_tree_preflop_only(&cfg).unwrap();
            let np = cfg.num_players as usize;
            let max_committable: Vec<i32> = cfg
                .starting_stacks
                .iter()
                .zip(&cfg.initial_contributions)
                .map(|(s, c)| s + c)
                .collect();
            // Equal-total invariant from the GameSpec derivation.
            for (p, &m) in max_committable.iter().enumerate() {
                assert_eq!(m, spec.stack, "seat {p} max_committable != spec.stack");
            }
            eprintln!(
                "\n>>> Gate run: v1 PRODUCTION PREFLOP (preflop-only, reduced ladder) (tree: {} nodes)",
                tree.num_nodes()
            );
            let vs = validate_tree(&tree, np, &max_committable, true);
            print_violations("v1 PRODUCTION PREFLOP", &vs, 5);
            vs
        };

        v_a.total() + v_b.total() + v_c.total() + v_d.total() + v_e.total()
            + v_f.total() + v_g.total() + v_h.total() + v_i.total() + v_j.total()
    };

    eprintln!("\n=== ROLLUP ===");
    eprintln!("Total violations across all configs: {}", total);
    if total == 0 {
        eprintln!("✓ Tree-builder is poker-rules-compliant. SAFE to optimize.");
    } else {
        eprintln!("✗ Tree-builder has {} violations. DO NOT optimize until fixed.", total);
        eprintln!("  Downstream solving will compute correct math on the WRONG GAME.");
    }
    // Phase 5 hardening: gate is now an ASSERT, not a print-only audit.
    // Any future regression in tree-builder legality fails this test rather
    // than silently printing for someone to read. This is the discipline the
    // session's rewrite arc established: the gate drove zero violations and
    // must hold zero going forward.
    assert_eq!(
        total, 0,
        "Tree-builder regression: {} violations across all configs. \
         See per-config violation breakdown above for the offending categories. \
         The standing rule from the rewrite arc: any non-zero count must be \
         driven back to zero before any optimization, because downstream \
         solving will compute correct math on the WRONG GAME otherwise.",
        total
    );
}
