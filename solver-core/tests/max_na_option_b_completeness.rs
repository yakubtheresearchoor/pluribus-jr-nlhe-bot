// MAX_NA_POSTFLOP decision (#31) verification step: completeness check on Option B.
//
// The convenient structural fact to verify: "Option B's 2 bet + 2 raise
// abstraction fits within MAX_NA_POSTFLOP=4 with max_children=4." The five-actions-
// look-like-they-should-be-but-fit-in-four claim. The bet-collapse bug
// the rewrite arc taught us to verify rather than accept.
//
// Methodology: extend the hu_completeness.rs reference enumerator to
// support raise sizes (production builder generates raises via
// add_raise_size_action; the existing enumerator only handles bets).
// Then build Option B's tree and confirm the unshortcuted reference
// count matches.
//
// MATCH → legitimate action-coincidence dedup (clamp+force_allin makes
//   raise_1.0 and explicit-allin both target max_committable, dedupe to
//   single AllIn → 5 actions → 4 children). Option B is complete; the
//   EV win is real; adopt with the showdown oracle extended.
//
// MISMATCH → silent drop. The fit-within-4 is the old bug in new clothes,
//   measured EV is on an incomplete tree, decision is wrong. Don't adopt.

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

fn ref_pot(s: &RefState) -> i32 { s.commits.iter().sum() }

fn ref_max_committable(cfg: &TreeConfig, p: usize) -> i32 {
    cfg.starting_stacks[p] + cfg.initial_contributions[p]
}

fn ref_next_active(s: &RefState, current: usize) -> Option<usize> {
    let np = s.commits.len();
    for offset in 1..=np {
        let next = (current + offset) % np;
        if !s.folded[next] { return Some(next); }
    }
    None
}

fn ref_first_active(s: &RefState) -> Option<usize> {
    (0..s.commits.len()).find(|&p| !s.folded[p])
}

/// Extended ref_actions: now handles raise sizes via the same logic as
/// production add_raise_size_action (PotRel(ratio) → prev_amount + pot*ratio).
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

    if player_remaining <= 0 {
        return vec![RefAct::Check];
    }
    let facing_bet = per_street_p < max_other_per_street;
    let mut actions: Vec<RefAct> = Vec::new();

    if !facing_bet {
        actions.push(RefAct::Check);
        for bs in &cfg.bet_sizes.bet {
            let delta = match bs {
                BetSize::PotRelative(r) => (pot as f64 * r).round() as i32,
                _ => unimplemented!("only PotRelative bet sizes"),
            };
            actions.push(RefAct::Bet(prev_amount + delta));
        }
        if max_amount <= (pot as f64 * cfg.add_allin_threshold).round() as i32 {
            actions.push(RefAct::AllIn(max_amount));
        }
    } else {
        actions.push(RefAct::Fold);
        actions.push(RefAct::Call);
        if !s.allin_flag {
            // EXTENDED: raise sizes.
            for bs in &cfg.bet_sizes.raise {
                let delta = match bs {
                    BetSize::PotRelative(r) => (pot as f64 * r).round() as i32,
                    _ => unimplemented!("only PotRelative raise sizes"),
                };
                actions.push(RefAct::Raise(prev_amount + delta));
            }
            let allin_thr = (pot as f64 * cfg.add_allin_threshold).round() as i32;
            if max_amount <= prev_amount + allin_thr {
                actions.push(RefAct::AllIn(max_amount));
            }
        }
    }

    // clamp_and_force_allin (same for Bet and Raise — production treats them
    // identically in clamp).
    let to_call = (max_other_cumulative - player_committed).max(0);
    let min_amount = (player_committed + to_call.min(player_remaining))
        .max(1)
        .min(max_amount);
    for a in actions.iter_mut() {
        let amt_opt = match *a {
            RefAct::Bet(v) | RefAct::Raise(v) => Some(v),
            _ => None,
        };
        if let Some(amt) = amt_opt {
            let clamped = amt.clamp(min_amount, max_amount);
            let new_diff = clamped - prev_amount;
            let new_pot = pot + 2 * new_diff;
            let force_thr = (new_pot as f64 * cfg.force_allin_threshold).round() as i32;
            if max_amount <= clamped + force_thr {
                *a = RefAct::AllIn(max_amount);
            } else if clamped != amt {
                // Re-wrap in same variant type.
                *a = match a {
                    RefAct::Bet(_) => RefAct::Bet(clamped),
                    RefAct::Raise(_) => RefAct::Raise(clamped),
                    _ => unreachable!(),
                };
            }
        }
    }

    // Sort + dedup (production builder.rs lines 639-640). Production's
    // Action enum derives Ord with variant ordering: Fold < Check < Call
    // < Bet < Raise < AllIn. Match that here.
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
        }
        RefAct::Bet(amt) | RefAct::Raise(amt) => {
            n.commits[p] = amt;
            n.has_acted = vec![false; np];
            n.has_acted[p] = true;
        }
        RefAct::AllIn(amt) => {
            n.commits[p] = amt;
            n.has_acted = vec![false; np];
            n.has_acted[p] = true;
            n.allin_flag = true;
        }
    }
    n.to_act = ref_next_active(&n, p).unwrap_or(0) as u8;
    n
}

fn ref_count(cfg: &TreeConfig, s: &RefState) -> (usize, usize, usize) {
    let np = s.commits.len();
    let num_unfolded = s.folded.iter().filter(|&&f| !f).count();
    if num_unfolded <= 1 { return (0, 0, 1); }

    let unfolded: Vec<usize> = (0..np).filter(|p| !s.folded[*p]).collect();
    if s.allin_flag {
        let all_allin = unfolded.iter()
            .all(|&p| s.commits[p] >= ref_max_committable(cfg, p));
        if all_allin {
            if s.street == 2 { return (0, 0, 1); }
            let mut next = s.clone();
            next.street += 1;
            next.round_start = next.commits.clone();
            next.has_acted = vec![false; np];
            next.to_act = ref_first_active(&next).unwrap_or(0) as u8;
            let (cp, cc, ct) = ref_count(cfg, &next);
            return (cp, cc + 1, ct);
        }
    }

    let all_acted = unfolded.iter().all(|&p| s.has_acted[p]);
    let cum_eq = unfolded.iter()
        .all(|&p| s.commits[p] == s.commits[unfolded[0]]);
    let no_betting = unfolded.iter()
        .all(|&p| s.commits[p] - s.round_start[p] == 0);
    let round_complete = all_acted && (cum_eq || no_betting);

    if round_complete {
        if s.street == 2 { return (0, 0, 1); }
        let mut next = s.clone();
        next.street += 1;
        next.round_start = next.commits.clone();
        next.has_acted = vec![false; np];
        next.to_act = ref_first_active(&next).unwrap_or(0) as u8;
        let (cp, cc, ct) = ref_count(cfg, &next);
        return (cp, cc + 1, ct);
    }

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
        tot_p += cp;
        tot_c += cc;
        tot_t += ct;
    }
    (tot_p, tot_c, tot_t)
}

fn ref_initial(cfg: &TreeConfig) -> RefState {
    let np = cfg.num_players as usize;
    RefState {
        street: cfg.initial_state as u8,
        commits: cfg.initial_contributions.clone(),
        round_start: cfg.initial_contributions.clone(),
        folded: vec![false; np],
        has_acted: vec![false; np],
        allin_flag: false,
        to_act: 0,
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

#[test]
fn option_b_completeness_check_hu() {
    eprintln!("\n=== Option B completeness check (HU symmetric [5,5]) ===");
    eprintln!("Verifying that 2 bet + 2 raise abstraction is fully expanded by the");
    eprintln!("builder, not silently collapsed to fit MAX_NA_POSTFLOP=4.\n");

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
    let tree = build_tree(&opt_b).unwrap();
    let (tp, tc, tt) = count_tree(&tree);
    let (rp, rc, rt) = ref_count(&opt_b, &ref_initial(&opt_b));

    eprintln!("Tree:       PLAYER={:5}  CHANCE={:5}  TERMINAL={:5}  total={}",
        tp, tc, tt, tp + tc + tt);
    eprintln!("Reference:  PLAYER={:5}  CHANCE={:5}  TERMINAL={:5}  total={}",
        rp, rc, rt, rp + rc + rt);
    eprintln!("Delta:      PLAYER={:+}  CHANCE={:+}  TERMINAL={:+}  total={:+}",
        tp as i64 - rp as i64, tc as i64 - rc as i64,
        tt as i64 - rt as i64,
        (tp + tc + tt) as i64 - (rp + rc + rt) as i64);
    eprintln!();

    if tp == rp && tc == rc && tt == rt {
        eprintln!("✓ COMPLETENESS CONFIRMED: tree exactly matches the unshortcuted");
        eprintln!("  reference enumeration. The fit-within-MAX_NA_POSTFLOP=4 is achieved by");
        eprintln!("  legitimate action coincidence (clamp + force_allin + dedup), not");
        eprintln!("  by silently dropping abstraction lines. Option B can be adopted.");
        eprintln!();
        eprintln!("  Next: extend the standing showdown oracle to cover Option B's");
        eprintln!("  contribution patterns (the maintenance principle), then proceed");
        eprintln!("  with blueprint solving on Option B.");
    } else {
        eprintln!("✗ INCOMPLETE: the builder is producing fewer nodes than the");
        eprintln!("  abstraction implies. The fit-within-MAX_NA_POSTFLOP=4 is achieved by");
        eprintln!("  silently dropping bet/raise sizes. The 5× exploitability win");
        eprintln!("  measured by max_na_exploitability.rs is on an INCOMPLETE Option B");
        eprintln!("  tree. The decision is WRONG until this is resolved.");
        eprintln!();
        eprintln!("  Next: identify which abstraction lines are being dropped. Either");
        eprintln!("  fix the builder (if the drop is a bug) or stay with Option A (if");
        eprintln!("  the drop is a design choice masquerading as fit-within-budget).");
    }

    assert_eq!(tp, rp, "Option B PLAYER count mismatch (tree {} vs ref {})", tp, rp);
    assert_eq!(tc, rc, "Option B CHANCE count mismatch (tree {} vs ref {})", tc, rc);
    assert_eq!(tt, rt, "Option B TERMINAL count mismatch (tree {} vs ref {})", tt, rt);
}
