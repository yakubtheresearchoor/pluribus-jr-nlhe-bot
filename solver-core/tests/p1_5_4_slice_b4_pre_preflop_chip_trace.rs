// P1.5.4 Slice B.4 pre: trace chips through preflop action sequences.
//
// Per the lead (2026-06-04): "do not fix anything yet, because you have
// not established it's a bug. Trace a concrete preflop sequence
// through the current builder and check the chips: when UTG 'checks'
// and action proceeds, can a player reach the flop without putting
// in the BB amount? ... If a player can see the flop without matching
// the BB (free flops), it's wrong-game ... If the labels {Check, Bet,
// AllIn} are a dead-money representation that still puts the correct
// chips in, it's a naming convention and there's no bug."
//
// The discriminating check is the CHIPS at the flop, not the action
// labels. This test walks HU preflop with the apparent {Check, Bet,
// AllIn} sequence (SB checks → BB checks → chance to flop) and
// prints the contributions at each node, so we can SEE whether
// "Check" lets SB into the flop without matching the BB.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;

#[test]
fn b4_pre_trace_hu_preflop_chips_through_check_check() {
    // HU preflop. SB = player 1 (highest-indexed via the None legacy
    // path); BB = player 0. SB acts first preflop per the existing
    // convention.
    let cfg = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![2, 1],  // BB=player 0 (2 chips), SB=player 1 (1 chip)
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&cfg).expect("HU preflop tree builds");

    eprintln!("\n═══ HU preflop chip trace ═══");
    eprintln!("Initial contributions per config: [BB=2, SB=1]");
    eprintln!("Starting pot per config: 3 chips");
    eprintln!("Starting stacks per config: [100, 100]\n");

    // Helper: show node summary.
    let dump = |idx: usize, label: &str| {
        let n = &tree.nodes[idx];
        let np = tree.num_players;
        let c: Vec<i32> = (0..np)
            .map(|p| tree.get_contribution(idx, p))
            .collect();
        let kind = if n.is_player() { format!("PLAYER(p={})", n.player_id) }
                   else if n.is_chance() { "CHANCE".to_string() }
                   else { "TERMINAL".to_string() };
        let bs = match n.board_state {
            0 => "Flop", 1 => "Turn", 2 => "River", 3 => "Preflop",
            x => return panic!("unknown board state {}", x),
        };
        let children = tree.node_children(idx);
        eprintln!("  {} | node {:>3} | {:<12} | {:>7} | contributions={:?} | children={:?}",
            label, idx, kind, bs, c, children);
    };

    eprintln!("Initial tree node:");
    dump(0, "root        ");

    let root = &tree.nodes[0];
    assert!(root.is_player(), "root should be a player node");
    eprintln!("\nRoot is player_id={} (expect SB=1 with HU legacy inference)", root.player_id);
    assert_eq!(root.player_id, 1, "HU legacy: SB=player 1 acts first preflop");

    // ── Walk action 0 (Check, per the wrong-game hypothesis) from the SB ──
    let root_children = tree.node_children(0).to_vec();
    eprintln!("\nRoot's {} children (the SB's available actions):", root_children.len());
    for (a, &ch) in root_children.iter().enumerate() {
        let cidx = ch as usize;
        let kind = if tree.nodes[cidx].is_player() { "PLAYER" }
                   else if tree.nodes[cidx].is_chance() { "CHANCE" }
                   else { "TERMINAL" };
        let c: Vec<i32> = (0..tree.num_players)
            .map(|p| tree.get_contribution(cidx, p))
            .collect();
        eprintln!("  action {}: child node {} ({}) | contributions={:?}",
            a, cidx, kind, c);
    }

    // Take SB's first action (whatever it is — could be Fold, Check, Call,
    // or Bet depending on the builder's wrong-game vs correct-game choice).
    // KEY OBSERVATION: if action 0 is Check (no chips added), contributions
    // stay [2, 1]. If action 0 is Call (SB matches BB), SB's contribution
    // jumps from 1 to 2.
    let after_sb_act0 = root_children[0] as usize;
    eprintln!("\nAfter SB takes action 0:");
    dump(after_sb_act0, "post-act0   ");
    let sb_contrib_after = tree.get_contribution(after_sb_act0, 1);
    let bb_contrib_after = tree.get_contribution(after_sb_act0, 0);
    eprintln!("  SB contribution went from 1 → {}", sb_contrib_after);
    eprintln!("  BB contribution went from 2 → {}", bb_contrib_after);

    // ── Now follow action 0 from whatever-node-this-is ──
    let after_acts = tree.node_children(after_sb_act0).to_vec();
    if !after_acts.is_empty() {
        let next = after_acts[0] as usize;
        eprintln!("\nAfter next player's action 0:");
        dump(next, "post-bothact0");
        let sb_final = tree.get_contribution(next, 1);
        let bb_final = tree.get_contribution(next, 0);
        eprintln!("  SB contribution: {}", sb_final);
        eprintln!("  BB contribution: {}", bb_final);

        // Continue walking until we hit a chance node (the preflop→flop boundary)
        // taking action[0] each time.
        let mut cur = next;
        for step in 1..10 {
            if tree.nodes[cur].is_chance() || tree.nodes[cur].is_terminal() {
                eprintln!("\nReached non-player node at step {}:", step);
                dump(cur, &format!("step {:>2}    ", step));
                break;
            }
            let cs = tree.node_children(cur).to_vec();
            if cs.is_empty() { break; }
            cur = cs[0] as usize;
            eprintln!("After step {} (action 0):", step);
            dump(cur, &format!("step {:>2}    ", step));
        }

        // Check if cur is a chance node — if so, look at the postflop start contributions.
        if tree.nodes[cur].is_chance() {
            eprintln!("\n══ Chance node reached ══");
            let cs = tree.node_children(cur).to_vec();
            if !cs.is_empty() {
                let postflop_root = cs[0] as usize;
                eprintln!("First postflop node below chance:");
                dump(postflop_root, "flop start  ");
                let sb_at_flop = tree.get_contribution(postflop_root, 1);
                let bb_at_flop = tree.get_contribution(postflop_root, 0);
                let pot_at_flop = sb_at_flop + bb_at_flop + tree.starting_pot;
                eprintln!("\n══ CHIP VERDICT ══");
                eprintln!("  At flop start:");
                eprintln!("    SB contributed: {} chips (was 1, expected {} for matched)",
                    sb_at_flop, if sb_at_flop == 2 { "✓ 2" } else { "✗ NOT 2 — free flop?" });
                eprintln!("    BB contributed: {} chips (was 2, expected 2)", bb_at_flop);
                eprintln!("    starting_pot field: {}", tree.starting_pot);
                eprintln!("    Total chips in pot at flop: {}", pot_at_flop);
                eprintln!("");
                if sb_at_flop == 2 && bb_at_flop == 2 {
                    eprintln!("  ✓ CORRECT-GAME: SB's contribution rose from 1 to 2 along the");
                    eprintln!("    sequence, even though the action was labeled 'Check'. The");
                    eprintln!("    {{Check, Bet, AllIn}} action set IS the codebase's dead-money");
                    eprintln!("    representation; 'Check' here means 'complete to the standing");
                    eprintln!("    bet' (i.e., functionally Call). The labels are non-standard");
                    eprintln!("    but the chips are correct. NOT a foundation bug.");
                } else if sb_at_flop == 1 {
                    eprintln!("  ✗ WRONG-GAME: SB saw the flop with only 1 chip in (the SB blind),");
                    eprintln!("    without matching the BB. This is a FREE FLOP — the builder is");
                    eprintln!("    modeling preflop incorrectly. Foundation bug confirmed.");
                } else {
                    eprintln!("  ?? UNEXPECTED: SB contribution = {}; neither 1 nor 2.", sb_at_flop);
                }
            }
        }
    }
}
