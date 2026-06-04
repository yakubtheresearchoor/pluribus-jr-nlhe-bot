// Phase 1 P3: rich preflop action abstraction completeness check.
//
// P2.3 (count_hu_preflop_asymmetric in hu_completeness.rs) verified the
// SIMPLEST preflop abstraction: bet=[PotRelative(1.0)], raise=[]. That
// confirmed the preflop tree builder fully expands the simplest action
// set per the abstraction. P3 extends this to the richer abstractions
// production preflop solves require: multiple bet sizes plus raise sizes
// (3-bet, 4-bet variants), exercising the collapse machinery that
// MAX_NA Option B verified for postflop.
//
// THE DISCRIMINATING QUESTION (per the standing project discipline):
// when the rich preflop action set produces 5+ nominal actions per node
// (fold, call, multiple raise sizes, allin), the abstraction expects
// the collapse to reduce them to <= MAX_NA=4 by ACTION-COINCIDENCE
// (multiple formal actions clamping to the same physical amount, merged
// lossless into one tree child) rather than SILENT DROP (formal actions
// at distinct amounts dropped to fit the budget, losing sequences from
// the abstraction's intended space). The reference enumerator implements
// the documented collapse from first principles. A match with the tree
// builder's count proves the builder implements the same collapse
// correctly. A mismatch (reference > tree) means the builder is dropping
// sequences silently; a mismatch (reference < tree) means the builder is
// adding nodes the abstraction does not imply.
//
// INDEPENDENCE OF THE REFERENCE (per the carry): the reference is
// derived from the abstraction SPEC, not from the tree builder. It
// enumerates the formal action set per the configured bet/raise sizes,
// applies clamp (a physical constraint, not arbitrary), and dedupes
// by (kind, amount) which IS the action-coincidence collapse at its
// finest granularity (two formal actions with the same clamped amount
// produce IDENTICAL successor states by physics, so merging them is
// lossless). The reference does NOT call into builder code. Match with
// the builder is two implementations of the same SPEC agreeing,
// catching builder bugs that drop sequences via short-circuit or that
// add bogus children. The dedup-by-amount granularity makes the
// collapse lossless by physics (not by spec choice), so the reference
// is correct-by-construction for the collapse step.
//
// CARRIED FORWARD from earlier preflop work:
//   P0: BoardState::Preflop variant exists (action.rs)
//   P1: first_preflop_player implemented (builder.rs)
//   P2.1: preflop_action_order verified button acts first preflop
//   P2.2: tree_correctness_gate covers preflop configs
//   P2.3: count_hu_preflop_asymmetric verified simplest preflop
//         (no raises) builds correctly
//   P2.4: preflop showdown oracle + no-flop-no-drop verified
//   THIS (P3): preflop with RICH action abstraction (raises) is
//              complete per the SPEC; collapse to MAX_NA=4 is
//              action-coincidence lossless, not silent drop.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

#[derive(Clone, Debug)]
struct RefState {
    street: u8,
    commits: Vec<i32>,
    round_start: Vec<i32>,
    folded: Vec<bool>,
    has_acted: Vec<bool>,
    allin_flag: bool,
    to_act: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefAct {
    Fold,
    Check,
    Call,
    Bet(i32),
    Raise(i32),
    AllIn(i32),
}

fn ref_pot(s: &RefState) -> i32 {
    s.commits.iter().sum()
}

fn ref_max_committable(cfg: &TreeConfig, p: usize) -> i32 {
    cfg.starting_stacks[p] + cfg.initial_contributions[p]
}

fn ref_next_active(s: &RefState, current: usize) -> Option<usize> {
    let np = s.commits.len();
    for offset in 1..=np {
        let next = (current + offset) % np;
        if !s.folded[next] {
            return Some(next);
        }
    }
    None
}

fn ref_first_active(s: &RefState) -> Option<usize> {
    (0..s.commits.len()).find(|&p| !s.folded[p])
}

/// Reference compute_actions for the rich abstraction (bet sizes +
/// raise sizes). Mirrors max_na_option_b_completeness ref_actions
/// pattern but explicitly handles BOTH bet and raise sizes (vs
/// hu_completeness which asserts raise is empty).
///
/// IMPORTANT: this is the spec-implementing enumerator. It does not
/// call into the tree builder. The dedup by (kind, amount) at the
/// end is the action-coincidence collapse: two formal actions with
/// the same clamped target produce the same successor state, so
/// merging them is lossless by physics.
fn ref_actions(cfg: &TreeConfig, s: &RefState) -> Vec<RefAct> {
    let np = s.commits.len();
    let p = s.to_act as usize;
    let player_committed = s.commits[p];
    let max_amount = ref_max_committable(cfg, p);
    let player_remaining = (max_amount - player_committed).max(0);
    let pot = ref_pot(s);

    let max_other_cumulative = (0..np)
        .filter(|&q| q != p && !s.folded[q])
        .map(|q| s.commits[q])
        .max()
        .unwrap_or(0);
    let prev_amount = max_other_cumulative;

    let per_street_p = s.commits[p] - s.round_start[p];
    let max_other_per_street = (0..np)
        .filter(|&q| q != p && !s.folded[q])
        .map(|q| s.commits[q] - s.round_start[q])
        .max()
        .unwrap_or(0);

    let mut actions: Vec<RefAct> = Vec::new();

    if player_remaining <= 0 {
        return vec![RefAct::Check];
    }

    let facing_bet = per_street_p < max_other_per_street;

    if !facing_bet {
        // NotFacingBet: Check, bet sizes, optional AllIn.
        actions.push(RefAct::Check);
        for bs in &cfg.bet_sizes.bet {
            match bs {
                BetSize::PotRelative(r) => {
                    let delta = (pot as f64 * r).round() as i32;
                    let target = prev_amount + delta;
                    actions.push(RefAct::Bet(target));
                }
                BetSize::AllIn => {
                    actions.push(RefAct::AllIn(max_amount));
                }
                _ => panic!("unsupported BetSize in bet vec"),
            }
        }
        if max_amount <= (pot as f64 * cfg.add_allin_threshold).round() as i32 {
            actions.push(RefAct::AllIn(max_amount));
        }
    } else {
        // FacingBet: Fold, Call, raise sizes if any, optional AllIn.
        actions.push(RefAct::Fold);
        actions.push(RefAct::Call);
        if !s.allin_flag {
            for bs in &cfg.bet_sizes.raise {
                match bs {
                    BetSize::PotRelative(r) => {
                        let delta = (pot as f64 * r).round() as i32;
                        let target = prev_amount + delta;
                        actions.push(RefAct::Raise(target));
                    }
                    BetSize::AllIn => {
                        actions.push(RefAct::AllIn(max_amount));
                    }
                    _ => panic!("unsupported BetSize in raise vec"),
                }
            }
            let allin_thr = (pot as f64 * cfg.add_allin_threshold).round() as i32;
            if max_amount <= prev_amount + allin_thr {
                actions.push(RefAct::AllIn(max_amount));
            }
        }
    }

    // Clamp + force_allin: convert Bet/Raise to AllIn when force threshold met.
    let to_call = (max_other_cumulative - player_committed).max(0);
    let min_amount = (player_committed + to_call.min(player_remaining))
        .max(1)
        .min(max_amount);
    for a in actions.iter_mut() {
        let amt_opt = match a {
            RefAct::Bet(v) | RefAct::Raise(v) => Some(*v),
            _ => None,
        };
        if let Some(amt) = amt_opt {
            let clamped = amt.clamp(min_amount, max_amount);
            let new_diff = clamped - prev_amount;
            let new_pot = pot + 2 * new_diff;
            let force_thr = (new_pot as f64 * cfg.force_allin_threshold).round() as i32;
            if max_amount <= clamped + force_thr {
                *a = RefAct::AllIn(max_amount);
            } else {
                *a = match a {
                    RefAct::Bet(_) => RefAct::Bet(clamped),
                    RefAct::Raise(_) => RefAct::Raise(clamped),
                    _ => unreachable!(),
                };
            }
        }
    }

    // Sort + dedup: action-coincidence collapse at finest granularity.
    // Two formal actions with the same (kind, amount) produce identical
    // successor states by physics. Merging is lossless.
    actions.sort_by(|a, b| {
        let key = |x: &RefAct| match x {
            RefAct::Fold => (0, 0),
            RefAct::Check => (1, 0),
            RefAct::Call => (2, 0),
            RefAct::Bet(amt) => (3, *amt),
            RefAct::Raise(amt) => (4, *amt),
            RefAct::AllIn(amt) => (5, *amt),
        };
        key(a).cmp(&key(b))
    });
    actions.dedup();

    assert_eq!(cfg.merging_threshold, 0.0,
        "ref enumerator only handles merging_threshold = 0.0");
    actions
}

fn ref_apply(cfg: &TreeConfig, s: &RefState, a: RefAct) -> RefState {
    let np = s.commits.len();
    let mut n = s.clone();
    let p = s.to_act as usize;
    let max_amount = ref_max_committable(cfg, p);
    let max_other = (0..np)
        .filter(|&q| q != p && !s.folded[q])
        .map(|q| s.commits[q])
        .max()
        .unwrap_or(0);

    match a {
        RefAct::Fold => {
            n.folded[p] = true;
            n.has_acted[p] = true;
        }
        RefAct::Check => { n.has_acted[p] = true; }
        RefAct::Call => {
            n.commits[p] = max_other.min(max_amount);
            n.has_acted[p] = true;
            if n.commits[p] == max_amount { n.allin_flag = true; }
        }
        RefAct::Bet(amt) | RefAct::Raise(amt) => {
            n.commits[p] = amt;
            n.has_acted[p] = true;
            for q in 0..np {
                if q != p && !n.folded[q] {
                    n.has_acted[q] = false;
                }
            }
        }
        RefAct::AllIn(amt) => {
            n.commits[p] = amt;
            n.has_acted[p] = true;
            n.allin_flag = true;
            for q in 0..np {
                if q != p && !n.folded[q] {
                    n.has_acted[q] = false;
                }
            }
        }
    }
    n.to_act = ref_next_active(&n, p).unwrap_or(0) as u8;
    n
}

/// Recursively enumerate the legal tree. Returns (players, chances, terminals).
/// Structure mirrors hu_completeness::ref_count (the proven postflop version)
/// with no semantic changes — only the action set differs (this enumerator
/// handles raises via ref_actions above). The street field is the
/// CHRONOLOGICAL ordinal (Preflop=0, Flop=1, Turn=2, River=3) per
/// BoardState::street(), not the enum repr value.
fn ref_count(cfg: &TreeConfig, s: &RefState) -> (usize, usize, usize) {
    let np = s.commits.len();
    let num_unfolded = s.folded.iter().filter(|&&f| !f).count();

    // Terminal classification: only one (or zero) unfolded player.
    if num_unfolded <= 1 {
        return (0, 0, 1);
    }

    let unfolded: Vec<usize> = (0..np).filter(|p| !s.folded[*p]).collect();

    // All-in shortcut (mirrors builder lines 359-388): if every non-folded
    // player has committed their max, skip forced-check round entirely.
    if s.allin_flag {
        let all_allin = unfolded
            .iter()
            .all(|&p| s.commits[p] >= ref_max_committable(cfg, p));
        if all_allin {
            if s.street == 3 {
                return (0, 0, 1);  // river end
            }
            let mut next = s.clone();
            next.street += 1;
            next.round_start = next.commits.clone();
            next.has_acted = vec![false; np];
            next.to_act = ref_first_active(&next).unwrap_or(0) as u8;
            let (cp, cc, ct) = ref_count(cfg, &next);
            return (cp, cc + 1, ct);
        }
    }

    // Round-complete check: all non-folded must have acted, AND either
    // cumulative commits equal OR per-street commits all zero (no-betting
    // case for asymmetric blinds).
    let all_acted = unfolded.iter().all(|&p| s.has_acted[p]);
    let cum_eq = unfolded
        .iter()
        .all(|&p| s.commits[p] == s.commits[unfolded[0]]);
    let no_betting = unfolded
        .iter()
        .all(|&p| s.commits[p] - s.round_start[p] == 0);
    let round_complete = all_acted && (cum_eq || no_betting);

    if round_complete {
        if s.street == 3 {
            return (0, 0, 1);
        }
        let mut next = s.clone();
        next.street += 1;
        next.round_start = next.commits.clone();
        next.has_acted = vec![false; np];
        next.to_act = ref_first_active(&next).unwrap_or(0) as u8;
        let (cp, cc, ct) = ref_count(cfg, &next);
        return (cp, cc + 1, ct);
    }

    // PLAYER node. Multi-player convention (builder make_child_node):
    // Fold ALWAYS produces TERMINAL. Critical for multi-player; no-op in HU
    // (Fold reduces num_unfolded to 1, terminal check catches it next call).
    let acts = ref_actions(cfg, s);
    let mut tot_p = 1usize;
    let mut tot_c = 0usize;
    let mut tot_t = 0usize;
    for a in acts {
        if matches!(a, RefAct::Fold) {
            tot_t += 1;
            continue;
        }
        let child = ref_apply(cfg, s, a);
        let (cp, cc, ct) = ref_count(cfg, &child);
        tot_p += cp; tot_c += cc; tot_t += ct;
    }
    (tot_p, tot_c, tot_t)
}

/// Preflop-aware initial state: to_act respects button-first preflop convention.
fn ref_initial(cfg: &TreeConfig) -> RefState {
    let np = cfg.num_players as usize;
    // HU preflop: button (player 1) acts first per builder::first_preflop_player.
    // Multiway preflop ordering is deferred per the plan; this test focuses on HU.
    // Postflop: leftmost-active acts first (player 0 in unfolded state).
    let to_act = match cfg.initial_state {
        BoardState::Preflop => {
            assert_eq!(np, 2, "P3 reference only handles HU preflop currently \
                (multiway preflop ordering deferred per plan)");
            // HU: button is the higher-indexed player.
            1u8
        }
        _ => 0u8,
    };
    RefState {
        street: cfg.initial_state.street(),  // chronological ordinal
        commits: cfg.initial_contributions.clone(),
        round_start: cfg.initial_contributions.clone(),
        folded: vec![false; np],
        has_acted: vec![false; np],
        allin_flag: false,
        to_act,
    }
}

fn count_tree(tree: &FlatTree) -> (usize, usize, usize) {
    let mut p = 0; let mut c = 0; let mut t = 0;
    for n in &tree.nodes {
        if n.is_player() { p += 1; }
        else if n.is_chance() { c += 1; }
        else { t += 1; }
    }
    (p, c, t)
}

fn run_completeness_check(cfg: &TreeConfig, label: &str) {
    let tree = build_tree(cfg).expect("tree build should succeed");
    let (tp, tc, tt) = count_tree(&tree);
    let init = ref_initial(cfg);
    let (rp, rc, rt) = ref_count(cfg, &init);

    eprintln!("\n=== P3 rich preflop completeness: {} ===", label);
    eprintln!(
        "  Tree:      player={:5} chance={:4} terminal={:5} total={}",
        tp, tc, tt, tp + tc + tt
    );
    eprintln!(
        "  Reference: player={:5} chance={:4} terminal={:5} total={}",
        rp, rc, rt, rp + rc + rt
    );
    eprintln!(
        "  Delta:     player={:+}  chance={:+}  terminal={:+}  total={:+}",
        tp as i64 - rp as i64,
        tc as i64 - rc as i64,
        tt as i64 - rt as i64,
        (tp + tc + tt) as i64 - (rp + rc + rt) as i64
    );

    assert_eq!(tp, rp,
        "[{}] PLAYER mismatch. Tree<ref: builder is dropping sequences the rich \
         abstraction spec says exist (lossy collapse, silent drop, NOT action-\
         coincidence). Tree>ref: builder is adding nodes the spec does not imply.",
        label);
    assert_eq!(tc, rc, "[{}] CHANCE mismatch.", label);
    assert_eq!(tt, rt, "[{}] TERMINAL mismatch.", label);
    eprintln!(
        "  ✓ Rich preflop abstraction is COMPLETE under the spec: tree count \
         matches independent reference. Collapse to MAX_NA budget is action-\
         coincidence (lossless), not silent drop."
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test configs: rich preflop abstractions
// ─────────────────────────────────────────────────────────────────────
//
// Each config exercises a progressively richer preflop action set, with
// the reference enumerator verifying the tree builder fully expands the
// abstraction without silent drops.

fn config_hu_preflop_with_one_raise() -> TreeConfig {
    // Step 1 of richness: add a single raise size to the simplest preflop
    // config. P2.3 already verified bet=[PotRelative(1.0)], raise=[] passes;
    // this is the first step into rich abstractions.
    TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![2, 1],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    }
}

fn config_hu_preflop_option_b_style() -> TreeConfig {
    // Step 2: Option B applied to preflop — 2 bet sizes + 2 raise sizes.
    // This is the configuration that approaches MAX_NA=4 budget tightly,
    // where the collapse machinery has to do real work. If the collapse
    // is action-coincidence (lossless), tree count matches reference. If
    // it is silent drop (lossy), reference > tree.
    TreeConfig {
        num_players: 2,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100, 100],
        initial_contributions: vec![2, 1],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
    }
}

#[test]
fn count_hu_preflop_with_one_raise() {
    run_completeness_check(
        &config_hu_preflop_with_one_raise(),
        "HU preflop, 1 bet + 1 raise (rich-abstraction step 1)",
    );
}

#[test]
fn count_hu_preflop_option_b_style() {
    run_completeness_check(
        &config_hu_preflop_option_b_style(),
        "HU preflop, 2 bet + 2 raise (Option B applied to preflop)",
    );
}
