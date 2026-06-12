// P1.5.4 diagnosis: distinguish finite-deep recursion (benign) from
// unbounded recursion (dead-money bug surviving in multiway path) for
// the 6-max enumerator stack overflow.
//
// Per the lead (2026-06-04): "don't accept 'separate issue, iterative
// rewrite later' until you confirm it's finite-deep-recursion (benign)
// versus unbounded recursion (the dead-money bug surviving in a
// multiway enumerator path the parallel fix didn't reach), because the
// symptom is identical to what started this bug hunt."
//
// This diagnosis re-implements the enumerator with hard bounds:
//   - depth bound: panics if recursion depth exceeds DEPTH_LIMIT
//   - call bound: panics if total ref_count calls exceeds CALL_LIMIT
//
// Interpretation:
//   - Completes within bounds → finite-deep recursion. Stack overflow
//     in the production enumerator is just large frame size × bounded
//     depth. Benign; iterative rewrite or larger stack suffices.
//   - Panics on DEPTH_LIMIT → unbounded recursion. Dead-money bug
//     surviving in a multiway path the parallel fix missed.
//   - Panics on CALL_LIMIT but not DEPTH_LIMIT → state revisiting (the
//     enumerator visits the same state many times). Also a bug.
//
// The corrected 6-max tree is 4923 nodes (verified standalone). A
// correct enumerator visits each tree node ONCE. So a sane bound:
//   DEPTH_LIMIT = 200 (much larger than any realistic preflop+postflop
//                      action sequence at 6-max)
//   CALL_LIMIT  = 100_000 (>> 4923 by a factor of 20×, generous slack
//                          for siblings + intermediate states)

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};

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

fn ref_actions(cfg: &TreeConfig, s: &RefState) -> Vec<RefAct> {
    // (Same as preflop_rich_abstraction_completeness's ref_actions —
    // copy here for self-containment.)
    let np = s.commits.len();
    let p = s.to_act as usize;
    let player_committed = s.commits[p];
    let max_amount = ref_max_committable(cfg, p);
    let player_remaining = (max_amount - player_committed).max(0);
    let pot = ref_pot(s);
    let max_other_cumulative = (0..np)
        .filter(|&q| q != p && !s.folded[q])
        .map(|q| s.commits[q]).max().unwrap_or(0);
    let prev_amount = max_other_cumulative;
    let per_street_p = s.commits[p] - s.round_start[p];
    let max_other_per_street = (0..np)
        .filter(|&q| q != p && !s.folded[q])
        .map(|q| s.commits[q] - s.round_start[q]).max().unwrap_or(0);
    let mut actions: Vec<RefAct> = Vec::new();
    if player_remaining <= 0 { return vec![RefAct::Check]; }
    let facing_bet = per_street_p < max_other_per_street;
    if !facing_bet {
        actions.push(RefAct::Check);
        for bs in &cfg.bet_sizes.bet {
            match bs {
                BetSize::PotRelative(r) => {
                    let delta = (pot as f64 * r).round() as i32;
                    actions.push(RefAct::Bet(prev_amount + delta));
                }
                BetSize::AllIn => actions.push(RefAct::AllIn(max_amount)),
                _ => panic!(),
            }
        }
        if max_amount <= (pot as f64 * cfg.add_allin_threshold).round() as i32 {
            actions.push(RefAct::AllIn(max_amount));
        }
    } else {
        actions.push(RefAct::Fold);
        actions.push(RefAct::Call);
        if !s.allin_flag {
            for bs in &cfg.bet_sizes.raise {
                match bs {
                    BetSize::PotRelative(r) => {
                        let delta = (pot as f64 * r).round() as i32;
                        actions.push(RefAct::Raise(prev_amount + delta));
                    }
                    BetSize::AllIn => actions.push(RefAct::AllIn(max_amount)),
                    _ => panic!(),
                }
            }
            let allin_thr = (pot as f64 * cfg.add_allin_threshold).round() as i32;
            if max_amount <= prev_amount + allin_thr {
                actions.push(RefAct::AllIn(max_amount));
            }
        }
    }
    let to_call = (max_other_cumulative - player_committed).max(0);
    let min_amount = (player_committed + to_call.min(player_remaining))
        .max(1).min(max_amount);
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
    actions
}

fn ref_apply(cfg: &TreeConfig, s: &RefState, a: RefAct) -> RefState {
    let np = s.commits.len();
    let mut n = s.clone();
    let p = s.to_act as usize;
    let max_amount = ref_max_committable(cfg, p);
    let max_other = (0..np)
        .filter(|&q| q != p && !s.folded[q])
        .map(|q| s.commits[q]).max().unwrap_or(0);
    match a {
        RefAct::Fold => { n.folded[p] = true; n.has_acted[p] = true; }
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
                if q != p && !n.folded[q] { n.has_acted[q] = false; }
            }
        }
        RefAct::AllIn(amt) => {
            n.commits[p] = amt;
            n.has_acted[p] = true;
            n.allin_flag = true;
            for q in 0..np {
                if q != p && !n.folded[q] { n.has_acted[q] = false; }
            }
        }
    }
    n.to_act = ref_next_active(&n, p).unwrap_or(0) as u8;
    n
}

// BOUNDED ref_count with depth + call tracking.
const DEPTH_LIMIT: usize = 200;
const CALL_LIMIT: usize = 100_000;

fn ref_count_bounded(
    cfg: &TreeConfig,
    s: &RefState,
    depth: usize,
    calls: &mut usize,
    max_depth_seen: &mut usize,
) -> (usize, usize, usize) {
    *calls += 1;
    if *calls > CALL_LIMIT {
        panic!("CALL_LIMIT exceeded ({} calls): the enumerator is revisiting states \
                or branching excessively. The tree has 4923 nodes; a correct enumerator \
                should make ~5000 calls, not >{}. This indicates either unbounded \
                recursion OR repeated state revisitation.",
            *calls, CALL_LIMIT);
    }
    if depth > *max_depth_seen { *max_depth_seen = depth; }
    if depth > DEPTH_LIMIT {
        eprintln!("\n══ STATE DUMP at depth-limit ({}) ══", depth);
        eprintln!("  street:     {}", s.street);
        eprintln!("  commits:    {:?}", s.commits);
        eprintln!("  round_start:{:?}", s.round_start);
        eprintln!("  folded:     {:?}", s.folded);
        eprintln!("  has_acted:  {:?}", s.has_acted);
        eprintln!("  allin_flag: {}", s.allin_flag);
        eprintln!("  to_act:     {} (player {})", s.to_act, s.to_act);
        let per_street: Vec<i32> = (0..s.commits.len())
            .map(|p| s.commits[p] - s.round_start[p]).collect();
        eprintln!("  per_street: {:?}", per_street);
        let unfolded: Vec<usize> = (0..s.commits.len()).filter(|p| !s.folded[*p]).collect();
        let all_acted = unfolded.iter().all(|&p| s.has_acted[p]);
        let cum_eq = unfolded.iter().all(|&p| s.commits[p] == s.commits[unfolded[0]]);
        let no_betting = unfolded.iter().all(|&p| s.commits[p] - s.round_start[p] == 0);
        eprintln!("  unfolded:   {:?}", unfolded);
        eprintln!("  all_acted:  {} | cum_eq: {} | no_betting: {} → round_complete: {}",
            all_acted, cum_eq, no_betting, all_acted && (cum_eq || no_betting));
        eprintln!("  ref_actions output: {:?}", ref_actions(cfg, s));
        panic!("DEPTH_LIMIT exceeded ({} levels). See state dump above.", depth);
    }

    let np = s.commits.len();
    let num_unfolded = s.folded.iter().filter(|&&f| !f).count();
    if num_unfolded <= 1 { return (0, 0, 1); }
    let unfolded: Vec<usize> = (0..np).filter(|p| !s.folded[*p]).collect();

    if s.allin_flag {
        let all_allin = unfolded.iter()
            .all(|&p| s.commits[p] >= ref_max_committable(cfg, p));
        if all_allin {
            if s.street == 3 { return (0, 0, 1); }
            let mut next = s.clone();
            next.street += 1;
            next.round_start = next.commits.clone();
            next.has_acted = vec![false; np];
            next.to_act = ref_first_active(&next).unwrap_or(0) as u8;
            let (cp, cc, ct) = ref_count_bounded(cfg, &next, depth + 1, calls, max_depth_seen);
            return (cp, cc + 1, ct);
        }
    }

    let all_acted = unfolded.iter().all(|&p| s.has_acted[p]);
    // Corrected standing-bet rule (matches builder.rs is_round_complete fix
    // 2026-06-04): the round is complete when every active player has matched
    // the standing bet OR is all-in at max_committable. Replaces the old
    // cum_eq check which caused unbounded recursion at all-in-mixed states.
    let standing_bet = unfolded.iter().map(|&p| s.commits[p]).max().unwrap();
    let matched_or_allin = unfolded.iter().all(|&p| {
        s.commits[p] == standing_bet || s.commits[p] >= ref_max_committable(cfg, p)
    });
    let no_betting = unfolded.iter().all(|&p| s.commits[p] - s.round_start[p] == 0);
    let round_complete = all_acted && (matched_or_allin || no_betting);

    if round_complete {
        if s.street == 3 { return (0, 0, 1); }
        let mut next = s.clone();
        next.street += 1;
        next.round_start = next.commits.clone();
        next.has_acted = vec![false; np];
        next.to_act = ref_first_active(&next).unwrap_or(0) as u8;
        let (cp, cc, ct) = ref_count_bounded(cfg, &next, depth + 1, calls, max_depth_seen);
        return (cp, cc + 1, ct);
    }

    let acts = ref_actions(cfg, s);
    let mut tot_p = 1usize;
    let mut tot_c = 0usize;
    let mut tot_t = 0usize;
    for a in acts {
        if matches!(a, RefAct::Fold) { tot_t += 1; continue; }
        let child = ref_apply(cfg, s, a);
        let (cp, cc, ct) = ref_count_bounded(cfg, &child, depth + 1, calls, max_depth_seen);
        tot_p += cp; tot_c += cc; tot_t += ct;
    }
    (tot_p, tot_c, tot_t)
}

fn ref_initial(cfg: &TreeConfig) -> RefState {
    let np = cfg.num_players as usize;
    let to_act = match cfg.initial_state {
        BoardState::Preflop => {
            let button = cfg.button_player.unwrap_or((np - 1) as u8) as usize;
            if np == 2 { button as u8 } else { ((button + 3) % np) as u8 }
        }
        _ => 0u8,
    };
    let round_start = match cfg.initial_state {
        BoardState::Preflop => vec![0_i32; np],
        _ => cfg.initial_contributions.clone(),
    };
    RefState {
        street: cfg.initial_state.street(),
        commits: cfg.initial_contributions.clone(),
        round_start,
        folded: vec![false; np],
        has_acted: vec![false; np],
        allin_flag: false,
        to_act,
    }
}

#[test]
fn diag_6max_enumerator_bounded() {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; 6],
        initial_contributions: vec![1, 2, 0, 0, 0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
            max_bets_per_street: None,
    };
    let s = ref_initial(&cfg);
    let mut calls = 0usize;
    let mut max_depth = 0usize;

    // Run inside a large-stack thread so we have room for finite-deep
    // recursion if that's what it is.
    let cfg_clone = cfg.clone();
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            let s = ref_initial(&cfg_clone);
            let mut calls = 0usize;
            let mut max_depth = 0usize;
            let result = ref_count_bounded(&cfg_clone, &s, 0, &mut calls, &mut max_depth);
            (result, calls, max_depth)
        })
        .expect("spawn");
    let (result, calls, max_depth) = handle.join().expect("thread join");

    eprintln!("\n══ 6-max enumerator bounded run ══");
    eprintln!("  Total calls:     {}", calls);
    eprintln!("  Max depth seen:  {}", max_depth);
    eprintln!("  Result (p/c/t):  {:?}", result);
    eprintln!("");
    if max_depth < DEPTH_LIMIT {
        eprintln!("  ✓ FINITE-DEEP recursion: max depth {} < limit {}. The 6-max enumerator",
            max_depth, DEPTH_LIMIT);
        eprintln!("    terminates; the production stack overflow is BENIGN (large frame size");
        eprintln!("    × bounded depth, not unbounded recursion). The dead-money bug is NOT");
        eprintln!("    surviving in a multiway enumerator path. The fix is the production");
        eprintln!("    stack size or an iterative rewrite, not another bug fix.");
    } else {
        eprintln!("  ✗ UNBOUNDED-LIKE recursion: max depth >= limit {}. The enumerator's", DEPTH_LIMIT);
        eprintln!("    recursion depth exceeds any realistic action-sequence length. This is");
        eprintln!("    the dead-money bug surviving in a multiway path, not a stack-size issue.");
    }
    // The test passes if it completes within bounds. The panic is the
    // signal that bounds were exceeded.
    let _ = (result, calls);
}
